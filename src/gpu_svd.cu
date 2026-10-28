// GPU thin SVD via the Gram trick (Phase A). See gpu_svd.hpp for the contract.
//
// Per block M (m x n, complex double; k = min(m,n)):
//   1. Form the smaller Gram matrix G (k x k Hermitian):
//        m <= n :  G = M Mᴴ   (cublasZherk, OP_N)
//        m >  n :  G = Mᴴ M   (cublasZherk, OP_C)
//   2. Eigendecompose G = Q Λ Qᴴ (cusolverDnZheevd; eigenvalues ascending).
//   3. The "other" factor by one GEMM:
//        m <= n :  P = Mᴴ Q  (n x k)   -> later V = P diag(1/sv)
//        m >  n :  P = M  Q  (m x k)   -> later U = P diag(1/sv)
//   4. Copy Q, Λ, P back; on the host reverse to descending order and scale the
//      P columns by 1/sv (guarding tiny sv exactly as the CPU thin_svd_gram does).
//
// This keeps every O(k^3) operation on the GPU (Zherk + Zheevd + Zgemm) and leaves
// only the O(k*max(m,n)) reversal/scaling on the host. Eigen MatrixXcd and cuBLAS
// are both column-major and std::complex<double> is bit-compatible with
// cuDoubleComplex, so host<->device copies are straight cudaMemcpy (no transpose).

#include "gpu_svd.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

namespace qx_gpu {
namespace {

// ---- Persistent context: handles + reusable device buffers, guarded by a mutex.
// Created once (cuSOLVER/cuBLAS handle creation is expensive). block_svd_truncate
// calls this from an OpenMP region, so a single mutex serialises GPU access -- a
// correct first version (the dominant cost is one big block per bond anyway).
struct Ctx {
    bool                ok    = false;
    cusolverDnHandle_t  solver = nullptr;
    cublasHandle_t      blas   = nullptr;
    std::mutex          mtx;
    // Reusable device buffers; grown on demand to the max block size seen.
    void*    dM    = nullptr; size_t cM   = 0;  // M     (m*n cuDoubleComplex)
    void*    dG    = nullptr; size_t cG   = 0;  // Gram  (k*k cuDoubleComplex)
    void*    dP    = nullptr; size_t cP   = 0;  // M(ᴴ)Q (max(m,n)*k cuDoubleComplex)
    double*  dW    = nullptr; size_t cW   = 0;  // eigenvalues (k double)
    void*    dWork = nullptr; size_t cWk  = 0;  // Zheevd workspace
    int*     dInfo = nullptr;
    // Buffers for form_theta_block_z (Phase B): A sub-block, Ai1, theta.
    void*    tA    = nullptr; size_t ctA  = 0;
    void*    tB    = nullptr; size_t ctB  = 0;
    void*    tT    = nullptr; size_t ctT  = 0;
    // FP32 buffers for thin_svd_gram_gpu_c (Phase C mixed precision).
    void*    fM    = nullptr; size_t cfM  = 0;
    void*    fG    = nullptr; size_t cfG  = 0;
    void*    fP    = nullptr; size_t cfP  = 0;
    float*   fW    = nullptr; size_t cfW  = 0;
    void*    fWk   = nullptr; size_t cfWk = 0;
    // Jacobi eigensolver params for the FP32 path: Cheevd (divide&conquer) fails to
    // converge (info>0) on the steep/clustered Gram spectra of theta blocks in FP32;
    // Cheevj (Jacobi) is robust there.
    syevjInfo_t fjacobi = nullptr;
};

bool init_ctx(Ctx& c) {
    if (cudaSetDevice(0) != cudaSuccess) return false;
    if (cusolverDnCreate(&c.solver) != CUSOLVER_STATUS_SUCCESS) return false;
    if (cublasCreate(&c.blas) != CUBLAS_STATUS_SUCCESS) return false;
    if (cusolverDnCreateSyevjInfo(&c.fjacobi) != CUSOLVER_STATUS_SUCCESS) return false;
    cusolverDnXsyevjSetTolerance(c.fjacobi, 1e-5);   // FP32-appropriate
    cusolverDnXsyevjSetMaxSweeps(c.fjacobi, 12);     // kept (large) eigenpairs converge early
    if (cudaMalloc(&c.dInfo, sizeof(int)) != cudaSuccess) return false;
    return true;
}

Ctx& ctx() {
    static Ctx c;
    static std::once_flag f;
    std::call_once(f, [&]() { c.ok = init_ctx(c); });
    return c;
}

// Ensure device buffer *p holds >= bytes; (re)allocate if not. Returns false on OOM.
bool ensure(void** p, size_t* cap, size_t bytes) {
    if (*cap >= bytes) return true;
    if (*p) cudaFree(*p);
    *p = nullptr; *cap = 0;
    if (cudaMalloc(p, bytes) != cudaSuccess) return false;
    *cap = bytes;
    return true;
}

}  // namespace

bool gpu_available() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return false;
    return ctx().ok;
}

bool thin_svd_gram_gpu_z(const std::complex<double>* M, int m, int n,
                         std::complex<double>* U, std::complex<double>* V,
                         double* sv) {
    Ctx& c = ctx();
    if (!c.ok || m <= 0 || n <= 0) return false;
    std::lock_guard<std::mutex> lock(c.mtx);

    const int k     = (m <= n) ? m : n;                 // = min(m,n)
    const int rowsP = (m <= n) ? n : m;                 // rows of P (the GEMM result)
    const cuDoubleComplex* dMc;
    cuDoubleComplex* dGc; cuDoubleComplex* dPc;

    // -- allocate / reuse device buffers
    if (!ensure(&c.dM, &c.cM, sizeof(cuDoubleComplex) * (size_t)m * n)) return false;
    if (!ensure(&c.dG, &c.cG, sizeof(cuDoubleComplex) * (size_t)k * k)) return false;
    if (!ensure(&c.dP, &c.cP, sizeof(cuDoubleComplex) * (size_t)rowsP * k)) return false;
    if (!ensure((void**)&c.dW, &c.cW, sizeof(double) * (size_t)k)) return false;
    dMc = reinterpret_cast<const cuDoubleComplex*>(c.dM);
    dGc = reinterpret_cast<cuDoubleComplex*>(c.dG);
    dPc = reinterpret_cast<cuDoubleComplex*>(c.dP);

    // -- upload M (column-major, contiguous)
    if (cudaMemcpy(c.dM, M, sizeof(cuDoubleComplex) * (size_t)m * n,
                   cudaMemcpyHostToDevice) != cudaSuccess) return false;

    const double one_r = 1.0, zero_r = 0.0;
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    // -- step 1: Gram matrix G (k x k Hermitian) via Zherk
    //    m<=n: G = M Mᴴ  (OP_N, leading dim m);  m>n: G = Mᴴ M (OP_C, leading dim m)
    cublasOperation_t herk_op = (m <= n) ? CUBLAS_OP_N : CUBLAS_OP_C;
    int herk_contract = (m <= n) ? n : m;               // contracted dimension
    if (cublasZherk(c.blas, uplo, herk_op, k, herk_contract,
                    &one_r, dMc, m, &zero_r, dGc, k) != CUBLAS_STATUS_SUCCESS)
        return false;

    // -- step 2: eigendecomposition of G (eigenvalues ascending, eigenvectors in dG)
    int lwork = 0;
    if (cusolverDnZheevd_bufferSize(c.solver, CUSOLVER_EIG_MODE_VECTOR, uplo,
                                    k, dGc, k, c.dW, &lwork) != CUSOLVER_STATUS_SUCCESS)
        return false;
    if (!ensure(&c.dWork, &c.cWk, sizeof(cuDoubleComplex) * (size_t)lwork)) return false;
    if (cusolverDnZheevd(c.solver, CUSOLVER_EIG_MODE_VECTOR, uplo, k, dGc, k, c.dW,
                         reinterpret_cast<cuDoubleComplex*>(c.dWork), lwork,
                         c.dInfo) != CUSOLVER_STATUS_SUCCESS)
        return false;
    int info = 0;
    if (cudaMemcpy(&info, c.dInfo, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    if (info != 0) return false;                        // eigensolver did not converge

    // -- step 3: the other factor P by one GEMM
    //    m<=n: P = Mᴴ Q  (n x k);  m>n: P = M Q  (m x k)
    const cuDoubleComplex c_one  = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex c_zero = make_cuDoubleComplex(0.0, 0.0);
    cublasOperation_t gemm_opA = (m <= n) ? CUBLAS_OP_C : CUBLAS_OP_N;
    if (cublasZgemm(c.blas, gemm_opA, CUBLAS_OP_N, rowsP, k, k,
                    &c_one, dMc, m, dGc, k, &c_zero, dPc, rowsP) != CUBLAS_STATUS_SUCCESS)
        return false;

    // -- step 4: bring Q (eigenvectors), Λ (eigenvalues), P back to the host
    std::vector<std::complex<double>> hQ((size_t)k * k);
    std::vector<std::complex<double>> hP((size_t)rowsP * k);
    std::vector<double>               hW((size_t)k);
    if (cudaMemcpy(hQ.data(), c.dG, sizeof(cuDoubleComplex) * (size_t)k * k,
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    if (cudaMemcpy(hP.data(), c.dP, sizeof(cuDoubleComplex) * (size_t)rowsP * k,
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    if (cudaMemcpy(hW.data(), c.dW, sizeof(double) * (size_t)k,
                   cudaMemcpyDeviceToHost) != cudaSuccess) return false;

    // -- reverse to descending order; scale P columns by 1/sv (guarding tiny sv).
    // For m<=n: U = reversed Q (m x k, m==k), V = reversed P / sv (n x k).
    // For m>n : V = reversed Q (n x k, n==k), U = reversed P / sv (m x k).
    std::complex<double>* eigDst = (m <= n) ? U : V;    // gets the eigenvectors Q
    std::complex<double>* sclDst = (m <= n) ? V : U;    // gets P / sv
    for (int j = 0; j < k; ++j) {
        double lam = hW[k - 1 - j];
        sv[j] = std::sqrt(lam > 0.0 ? lam : 0.0);
    }
    double smax = (k > 0 && sv[0] > 0.0) ? sv[0] : 1.0;
    for (int j = 0; j < k; ++j) {
        int src = k - 1 - j;                            // descending: reverse columns
        // eigenvector column (length k) -> eigDst column j
        const std::complex<double>* qcol = hQ.data() + (size_t)src * k;
        std::complex<double>*       dcol = eigDst + (size_t)j * k;
        for (int r = 0; r < k; ++r) dcol[r] = qcol[r];
        // P column (length rowsP) scaled by 1/sv -> sclDst column j
        const std::complex<double>* pcol = hP.data() + (size_t)src * rowsP;
        std::complex<double>*       scol = sclDst + (size_t)j * rowsP;
        if (sv[j] > 1e-14 * smax) {
            double inv = 1.0 / sv[j];
            for (int r = 0; r < rowsP; ++r) scol[r] = pcol[r] * inv;
        } else {
            for (int r = 0; r < rowsP; ++r) scol[r] = std::complex<double>(0.0, 0.0);
            sv[j] = 0.0;   // mode dropped: a zeroed factor column must carry sv=0 so the
                           // global truncation neither keeps it nor mis-counts trunc error
        }
    }
    return true;
}

bool thin_svd_gram_gpu_c(const std::complex<double>* M, int m, int n,
                         std::complex<double>* U, std::complex<double>* V,
                         double* sv) {
    Ctx& c = ctx();
    if (!c.ok || m <= 0 || n <= 0) return false;
    std::lock_guard<std::mutex> lock(c.mtx);

    const int k     = (m <= n) ? m : n;
    const int rowsP = (m <= n) ? n : m;

    // host: cast M (complex<double>) -> complex<float>
    std::vector<std::complex<float>> Mf((size_t)m * n);
    for (size_t i = 0; i < (size_t)m * n; ++i)
        Mf[i] = std::complex<float>((float)M[i].real(), (float)M[i].imag());

    if (!ensure(&c.fM, &c.cfM, sizeof(cuComplex) * (size_t)m * n)) return false;
    if (!ensure(&c.fG, &c.cfG, sizeof(cuComplex) * (size_t)k * k)) return false;
    if (!ensure(&c.fP, &c.cfP, sizeof(cuComplex) * (size_t)rowsP * k)) return false;
    if (!ensure((void**)&c.fW, &c.cfW, sizeof(float) * (size_t)k)) return false;
    auto* dM = reinterpret_cast<cuComplex*>(c.fM);
    auto* dG = reinterpret_cast<cuComplex*>(c.fG);
    auto* dP = reinterpret_cast<cuComplex*>(c.fP);

    if (cudaMemcpy(dM, Mf.data(), sizeof(cuComplex)*(size_t)m*n, cudaMemcpyHostToDevice) != cudaSuccess) return false;

    const float one_r = 1.0f, zero_r = 0.0f;
    const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    cublasOperation_t herk_op = (m <= n) ? CUBLAS_OP_N : CUBLAS_OP_C;
    int herk_contract = (m <= n) ? n : m;
    if (cublasCherk(c.blas, uplo, herk_op, k, herk_contract,
                    &one_r, dM, m, &zero_r, dG, k) != CUBLAS_STATUS_SUCCESS) return false;

    // Jacobi eigensolver (robust in FP32; Cheevd D&C fails to converge here).
    int lwork = 0;
    if (cusolverDnCheevj_bufferSize(c.solver, CUSOLVER_EIG_MODE_VECTOR, uplo,
                                    k, dG, k, c.fW, &lwork, c.fjacobi) != CUSOLVER_STATUS_SUCCESS) return false;
    if (!ensure(&c.fWk, &c.cfWk, sizeof(cuComplex) * (size_t)lwork)) return false;
    if (cusolverDnCheevj(c.solver, CUSOLVER_EIG_MODE_VECTOR, uplo, k, dG, k, c.fW,
                         reinterpret_cast<cuComplex*>(c.fWk), lwork, c.dInfo, c.fjacobi) != CUSOLVER_STATUS_SUCCESS) return false;
    int info = 0;
    if (cudaMemcpy(&info, c.dInfo, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    // Accept only a fully-converged Jacobi decomposition (info==0). info>0 means the
    // sweep cap was hit without reaching tolerance -- cuSOLVER does not guarantee a
    // valid eigendecomposition there, so reject and let the caller fall back to the
    // FP64 path (correctness over the modest FP32 speedup).
    if (info != 0) return false;

    const cuComplex c_one  = make_cuComplex(1.0f, 0.0f);
    const cuComplex c_zero = make_cuComplex(0.0f, 0.0f);
    cublasOperation_t gemm_opA = (m <= n) ? CUBLAS_OP_C : CUBLAS_OP_N;
    if (cublasCgemm(c.blas, gemm_opA, CUBLAS_OP_N, rowsP, k, k,
                    &c_one, dM, m, dG, k, &c_zero, dP, rowsP) != CUBLAS_STATUS_SUCCESS) return false;

    std::vector<std::complex<float>> hQ((size_t)k * k);
    std::vector<std::complex<float>> hP((size_t)rowsP * k);
    std::vector<float>               hW((size_t)k);
    if (cudaMemcpy(hQ.data(), dG, sizeof(cuComplex)*(size_t)k*k, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    if (cudaMemcpy(hP.data(), dP, sizeof(cuComplex)*(size_t)rowsP*k, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    if (cudaMemcpy(hW.data(), c.fW, sizeof(float)*(size_t)k, cudaMemcpyDeviceToHost) != cudaSuccess) return false;

    // reorder to descending + scale; assemble outputs in double.
    std::complex<double>* eigDst = (m <= n) ? U : V;
    std::complex<double>* sclDst = (m <= n) ? V : U;
    for (int j = 0; j < k; ++j) {
        double lam = (double)hW[k - 1 - j];
        sv[j] = std::sqrt(lam > 0.0 ? lam : 0.0);
    }
    double smax = (k > 0 && sv[0] > 0.0) ? sv[0] : 1.0;
    for (int j = 0; j < k; ++j) {
        int src = k - 1 - j;
        const std::complex<float>* qcol = hQ.data() + (size_t)src * k;
        std::complex<double>*       dcol = eigDst + (size_t)j * k;
        for (int r = 0; r < k; ++r) dcol[r] = std::complex<double>(qcol[r].real(), qcol[r].imag());
        const std::complex<float>* pcol = hP.data() + (size_t)src * rowsP;
        std::complex<double>*       scol = sclDst + (size_t)j * rowsP;
        if (sv[j] > 1e-6 * smax) {          // FP32 guard: ~single-precision floor
            double inv = 1.0 / sv[j];
            for (int r = 0; r < rowsP; ++r) scol[r] = std::complex<double>(pcol[r].real()*inv, pcol[r].imag()*inv);
        } else {
            for (int r = 0; r < rowsP; ++r) scol[r] = std::complex<double>(0.0, 0.0);
            sv[j] = 0.0;   // mode dropped: a zeroed factor column must carry sv=0 so the
                           // global truncation neither keeps it nor mis-counts trunc error
        }
    }
    return true;
}

bool form_theta_block_z(const std::complex<double>* Aup, int dLup,
                        const std::complex<double>* Adn, int dLdn,
                        const std::complex<double>* Ai1, int kmid, int ncol,
                        std::complex<double>* theta) {
    Ctx& c = ctx();
    const int nrow = dLup + dLdn;
    if (!c.ok || nrow <= 0 || ncol <= 0 || kmid <= 0) return false;
    std::lock_guard<std::mutex> lock(c.mtx);

    const int maxdL = (dLup > dLdn) ? dLup : dLdn;
    if (!ensure(&c.tA, &c.ctA, sizeof(cuDoubleComplex) * (size_t)maxdL * kmid)) return false;
    if (!ensure(&c.tB, &c.ctB, sizeof(cuDoubleComplex) * (size_t)kmid * ncol)) return false;
    if (!ensure(&c.tT, &c.ctT, sizeof(cuDoubleComplex) * (size_t)nrow * ncol)) return false;
    auto* dA = reinterpret_cast<cuDoubleComplex*>(c.tA);
    auto* dB = reinterpret_cast<cuDoubleComplex*>(c.tB);
    auto* dT = reinterpret_cast<cuDoubleComplex*>(c.tT);

    // Ai1 uploaded once, shared by both spin sectors.
    if (cudaMemcpy(dB, Ai1, sizeof(cuDoubleComplex)*(size_t)kmid*ncol, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    // zero theta so a missing spin sector leaves its rows = 0 (matches CPU).
    if (cudaMemset(dT, 0, sizeof(cuDoubleComplex)*(size_t)nrow*ncol) != cudaSuccess) return false;

    const cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
    // s1=up -> theta rows [0, dLup): C = Aup * Ai1, written with ldc = nrow.
    if (dLup > 0 && Aup) {
        if (cudaMemcpy(dA, Aup, sizeof(cuDoubleComplex)*(size_t)dLup*kmid, cudaMemcpyHostToDevice) != cudaSuccess) return false;
        if (cublasZgemm(c.blas, CUBLAS_OP_N, CUBLAS_OP_N, dLup, ncol, kmid,
                        &one, dA, dLup, dB, kmid, &zero, dT, nrow) != CUBLAS_STATUS_SUCCESS) return false;
    }
    // s1=dn -> theta rows [dLup, nrow): write into dT + dLup (ldc = nrow).
    if (dLdn > 0 && Adn) {
        if (cudaMemcpy(dA, Adn, sizeof(cuDoubleComplex)*(size_t)dLdn*kmid, cudaMemcpyHostToDevice) != cudaSuccess) return false;
        if (cublasZgemm(c.blas, CUBLAS_OP_N, CUBLAS_OP_N, dLdn, ncol, kmid,
                        &one, dA, dLdn, dB, kmid, &zero, dT + dLup, nrow) != CUBLAS_STATUS_SUCCESS) return false;
    }
    if (cudaMemcpy(theta, dT, sizeof(cuDoubleComplex)*(size_t)nrow*ncol, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    return true;
}

// Arbitrary local dimension d (spin-1, ...): nsec row groups instead of up/dn. As[s]
// is a (dLs[s] x kmid) column-major sub-block (nullptr / dLs[s]=0 = absent, zero rows);
// theta row s starts at the cumulative offset of the earlier groups. One GEMM per
// present group into a shared (nrow x ncol) theta, sharing one Ai1 upload.
bool form_theta_block_z_multi(const std::complex<double>* const* As, const int* dLs,
                              int nsec, const std::complex<double>* Ai1, int kmid,
                              int ncol, int nrow, std::complex<double>* theta) {
    Ctx& c = ctx();
    if (!c.ok || nrow <= 0 || ncol <= 0 || kmid <= 0 || nsec <= 0) return false;
    std::lock_guard<std::mutex> lock(c.mtx);

    int maxdL = 0;
    for (int s = 0; s < nsec; ++s) if (dLs[s] > maxdL) maxdL = dLs[s];
    if (maxdL <= 0) return false;
    if (!ensure(&c.tA, &c.ctA, sizeof(cuDoubleComplex) * (size_t)maxdL * kmid)) return false;
    if (!ensure(&c.tB, &c.ctB, sizeof(cuDoubleComplex) * (size_t)kmid * ncol)) return false;
    if (!ensure(&c.tT, &c.ctT, sizeof(cuDoubleComplex) * (size_t)nrow * ncol)) return false;
    auto* dA = reinterpret_cast<cuDoubleComplex*>(c.tA);
    auto* dB = reinterpret_cast<cuDoubleComplex*>(c.tB);
    auto* dT = reinterpret_cast<cuDoubleComplex*>(c.tT);

    if (cudaMemcpy(dB, Ai1, sizeof(cuDoubleComplex)*(size_t)kmid*ncol, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cudaMemset(dT, 0, sizeof(cuDoubleComplex)*(size_t)nrow*ncol) != cudaSuccess) return false;

    const cuDoubleComplex one  = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);
    int row_off = 0;
    for (int s = 0; s < nsec; ++s) {
        int dL = dLs[s];
        if (dL > 0 && As[s]) {   // theta rows [row_off, row_off+dL): C = As[s] * Ai1
            if (cudaMemcpy(dA, As[s], sizeof(cuDoubleComplex)*(size_t)dL*kmid, cudaMemcpyHostToDevice) != cudaSuccess) return false;
            if (cublasZgemm(c.blas, CUBLAS_OP_N, CUBLAS_OP_N, dL, ncol, kmid,
                            &one, dA, dL, dB, kmid, &zero, dT + row_off, nrow) != CUBLAS_STATUS_SUCCESS) return false;
        }
        row_off += dL;
    }
    if (cudaMemcpy(theta, dT, sizeof(cuDoubleComplex)*(size_t)nrow*ncol, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    return true;
}

// ---- Batched Hermitian eigensolves for the B-form TEBD layer (M12) -----------
namespace {

// The padded size of a k x k matrix, i.e. its class. Each class is one
// cusolverDnXsyevBatched, whose time is mostly latency growing with the size, so
// the classes step by 1/4 to 1/2: padding costs at most that, fewer classes save
// launches, and the classes run concurrently anyway.
int eig_class(int k) {
    if (k <= 64)  return ((k + 15) / 16) * 16;
    if (k <= 256) return ((k + 31) / 32) * 32;
    if (k <= 512) return ((k + 63) / 64) * 64;
    return ((k + 127) / 128) * 128;
}

struct BatchSlot {           // one size class: its own stream, handle and buffers
    cudaStream_t        st = nullptr;
    cusolverDnHandle_t  h  = nullptr;
    cusolverDnParams_t  p  = nullptr;
    void* dA = nullptr;    size_t cA  = 0;
    void* dW = nullptr;    size_t cW  = 0;
    void* dInfo = nullptr; size_t cI  = 0;
    void* dWork = nullptr; size_t cWk = 0;
    std::vector<char> hWork;
};
struct BatchCtx {
    std::mutex             mtx;
    std::vector<BatchSlot> slots;           // grown to the most classes seen at once
    void* hBuf = nullptr; size_t cH = 0;    // pinned staging, the matrices out and back
};
BatchCtx& bctx() { static BatchCtx b; return b; }

bool ensure_slot(BatchSlot& s) {
    if (s.h) return true;
    if (!s.st && cudaStreamCreateWithFlags(&s.st, cudaStreamNonBlocking) != cudaSuccess)
        return false;
    if (!s.p && cusolverDnCreateParams(&s.p) != CUSOLVER_STATUS_SUCCESS) return false;
    cusolverDnHandle_t h = nullptr;
    if (cusolverDnCreate(&h) != CUSOLVER_STATUS_SUCCESS) return false;
    if (cusolverDnSetStream(h, s.st) != CUSOLVER_STATUS_SUCCESS) { cusolverDnDestroy(h); return false; }
    s.h = h;
    return true;
}
bool ensure_pinned(void** p, size_t* cap, size_t bytes) {
    if (*cap >= bytes) return true;
    if (*p) cudaFreeHost(*p);
    *p = nullptr; *cap = 0;
    if (cudaMallocHost(p, bytes) != cudaSuccess) return false;
    *cap = bytes;
    return true;
}

// Pack matrix G (k x k) into the N x N slot A: G top left, -pad on the rest of the
// diagonal, zeros elsewhere.
template <class T>
void pack_padded(const std::complex<double>* G, int k, int N, double pad, T* A) {
    std::fill(A, A + (size_t)N * N, T(0));
    for (int c = 0; c < k; ++c)
        for (int r = 0; r < k; ++r) {
            const std::complex<double> g = G[(size_t)c * k + r];
            A[(size_t)c * N + r] = T((typename T::value_type)g.real(),
                                     (typename T::value_type)g.imag());
        }
    for (int d = k; d < N; ++d) A[(size_t)d * N + d] = T((typename T::value_type)pad, 0);
}
// The k real eigenpairs back out of slot A / W: the last k (ascending; the padding's
// eigenvalues are the lowest), rows 0..k-1 of their eigenvectors.
template <class T, class R>
void unpack_padded(const T* A, const R* W, int k, int N, std::complex<double>* G, double* w) {
    for (int c = 0; c < k; ++c) {
        w[c] = (double)W[N - k + c];
        const T* src = A + (size_t)(N - k + c) * N;
        for (int r = 0; r < k; ++r)
            G[(size_t)c * k + r] = std::complex<double>((double)src[r].real(), (double)src[r].imag());
    }
}

}  // namespace

bool eigh_batched_z(int count, std::complex<double>* const* G, const int* n,
                    double* const* w, bool fp32, int* ok) {
    for (int i = 0; i < count; ++i) ok[i] = 0;
    if (count <= 0 || !ctx().ok) return false;     // ctx() initialises the device
    BatchCtx& b = bctx();
    std::lock_guard<std::mutex> lock(b.mtx);

    std::map<int, std::vector<int>> cls;           // class size N -> matrices
    for (int i = 0; i < count; ++i) if (n[i] > 0) cls[eig_class(n[i])].push_back(i);
    const size_t es = fp32 ? sizeof(std::complex<float>) : sizeof(std::complex<double>);
    const size_t ws = fp32 ? sizeof(float) : sizeof(double);
    // Staging layout, per class: the matrices, the eigenvalues, the info flags.
    std::vector<size_t> offA, offW, offI;
    size_t total = 0;
    for (auto& [N, ids] : cls) {
        offA.push_back(total); total += es * (size_t)N * N * ids.size();
        offW.push_back(total); total += ws * (size_t)N * ids.size();
        offI.push_back(total); total += sizeof(int) * ids.size();
        total = (total + 255) & ~(size_t)255;
    }
    if (!ensure_pinned(&b.hBuf, &b.cH, total)) return false;
    if (b.slots.size() < cls.size()) b.slots.resize(cls.size());
    const cudaDataType tA = fp32 ? CUDA_C_32F : CUDA_C_64F;
    const cudaDataType tW = fp32 ? CUDA_R_32F : CUDA_R_64F;
    const auto vec = CUSOLVER_EIG_MODE_VECTOR;
    const auto low = CUBLAS_FILL_MODE_LOWER;

    // Pack and launch class by class: the host packs the next class while the device
    // solves the last, and the classes' streams run concurrently.
    bool fine = true;
    int c = 0;
    for (auto it = cls.begin(); it != cls.end() && fine; ++it, ++c) {
        const int N = it->first;
        const std::vector<int>& ids = it->second;
        const int64_t nb = (int64_t)ids.size();
        BatchSlot& s = b.slots[c];
        char* hA = static_cast<char*>(b.hBuf) + offA[c];
        for (size_t t = 0; t < ids.size(); ++t) {
            const int i = ids[t], k = n[i];
            double tr = 0;
            for (int r = 0; r < k; ++r) tr += G[i][(size_t)r * k + r].real();
            const double pad = -(tr > 0 ? tr : 1.0);   // below every eigenvalue of a PSD G
            if (fp32) pack_padded(G[i], k, N, pad, reinterpret_cast<std::complex<float>*>(hA) + t * (size_t)N * N);
            else      pack_padded(G[i], k, N, pad, reinterpret_cast<std::complex<double>*>(hA) + t * (size_t)N * N);
        }
        size_t wsD = 0, wsH = 0;
        fine = ensure_slot(s)
            && ensure(&s.dA, &s.cA, es * (size_t)N * N * nb)
            && ensure(&s.dW, &s.cW, ws * (size_t)N * nb)
            && ensure(&s.dInfo, &s.cI, sizeof(int) * nb)
            && cusolverDnXsyevBatched_bufferSize(s.h, s.p, vec, low, N, tA, s.dA, N, tW, s.dW,
                                                 tA, &wsD, &wsH, nb) == CUSOLVER_STATUS_SUCCESS
            && ensure(&s.dWork, &s.cWk, wsD > 0 ? wsD : 1);
        if (!fine) break;
        if (s.hWork.size() < wsH) s.hWork.resize(wsH);
        fine = cudaMemcpyAsync(s.dA, hA, es * (size_t)N * N * nb, cudaMemcpyHostToDevice,
                               s.st) == cudaSuccess
            && cusolverDnXsyevBatched(s.h, s.p, vec, low, N, tA, s.dA, N, tW, s.dW, tA,
                                      s.dWork, wsD, s.hWork.empty() ? nullptr : s.hWork.data(),
                                      wsH, static_cast<int*>(s.dInfo), nb) == CUSOLVER_STATUS_SUCCESS
            && cudaMemcpyAsync(hA, s.dA, es * (size_t)N * N * nb, cudaMemcpyDeviceToHost,
                               s.st) == cudaSuccess
            && cudaMemcpyAsync(static_cast<char*>(b.hBuf) + offW[c], s.dW, ws * (size_t)N * nb,
                               cudaMemcpyDeviceToHost, s.st) == cudaSuccess
            && cudaMemcpyAsync(static_cast<char*>(b.hBuf) + offI[c], s.dInfo, sizeof(int) * nb,
                               cudaMemcpyDeviceToHost, s.st) == cudaSuccess;
    }
    // Wait for every launched class (even after a failure: the staging buffer must
    // not be reused while a copy into it is pending).
    for (int u = 0; u < c && u < (int)b.slots.size(); ++u)
        if (cudaStreamSynchronize(b.slots[u].st) != cudaSuccess) fine = false;
    if (!fine) return false;

    c = 0;
    for (auto& [N, ids] : cls) {
        const char* hA = static_cast<const char*>(b.hBuf) + offA[c];
        const char* hW = static_cast<const char*>(b.hBuf) + offW[c];
        const int*  hI = reinterpret_cast<const int*>(static_cast<const char*>(b.hBuf) + offI[c]);
        for (size_t t = 0; t < ids.size(); ++t) {
            const int i = ids[t], k = n[i];
            if (hI[t] != 0) continue;              // not converged: the caller solves it
            if (fp32)
                unpack_padded(reinterpret_cast<const std::complex<float>*>(hA) + t * (size_t)N * N,
                              reinterpret_cast<const float*>(hW) + t * (size_t)N, k, N, G[i], w[i]);
            else
                unpack_padded(reinterpret_cast<const std::complex<double>*>(hA) + t * (size_t)N * N,
                              reinterpret_cast<const double*>(hW) + t * (size_t)N, k, N, G[i], w[i]);
            ok[i] = 1;
        }
        ++c;
    }
    return true;
}

}  // namespace qx_gpu
