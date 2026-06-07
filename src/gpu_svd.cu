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
};

bool init_ctx(Ctx& c) {
    if (cudaSetDevice(0) != cudaSuccess) return false;
    if (cusolverDnCreate(&c.solver) != CUSOLVER_STATUS_SUCCESS) return false;
    if (cublasCreate(&c.blas) != CUBLAS_STATUS_SUCCESS) return false;
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
        }
    }
    return true;
}

}  // namespace qx_gpu
