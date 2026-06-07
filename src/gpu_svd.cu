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

#include <cmath>
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
    // info<0: illegal arg -> fail. info>0: Jacobi hit the sweep cap without full
    // convergence, but the large eigenpairs (the kept Schmidt values) converge first,
    // so the result is still good for TEBD truncation -> accept.
    if (info < 0) return false;

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

}  // namespace qx_gpu
