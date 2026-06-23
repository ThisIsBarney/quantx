// GPU (cuSOLVER/cuBLAS) thin SVD via the Gram trick -- Phase A of the QuantX GPU
// acceleration plan (docs/gpu_acceleration_plan.md). Host-callable interface with
// plain types only (no CUDA or Eigen types here) so it can be included from the
// MSVC-compiled dmrg_blocks.cpp; the implementation lives in gpu_svd.cu (nvcc).
//
// Semantics match the CPU thin_svd_gram in dmrg_blocks.cpp exactly: it computes a
// thin SVD M = U diag(sv) Vᴴ via the Hermitian eigendecomposition of the smaller
// Gram matrix (M Mᴴ if m<=n, else Mᴴ M). Squares the condition number -- the tiny
// tail singular values lose precision (TEBD truncates them anyway); the kept
// Schmidt values are accurate.
//
// All matrices are COLUMN-MAJOR (Eigen / cuBLAS convention). With k = min(m,n):
//   M : m x n   (input)
//   U : m x k   (output, caller-allocated, size m*k)
//   V : n x k   (output, caller-allocated, size n*k)
//   sv: k       (output, caller-allocated, descending)
// Returns true on success; false on any device error or if no GPU is usable, in
// which case the caller falls back to the CPU path.
#pragma once
#include <complex>

namespace qx_gpu {

// True if a CUDA device is present and the cuSOLVER/cuBLAS context initialised.
bool gpu_available();

// Complex-double Gram-trick thin SVD on the GPU. See header comment for layout.
bool thin_svd_gram_gpu_z(const std::complex<double>* M, int m, int n,
                         std::complex<double>* U, std::complex<double>* V,
                         double* sv);

// Mixed-precision variant (Phase C): identical interface and layout (host buffers
// are complex<double>), but the O(k³) GPU linear algebra (Gram, Hermitian
// eigensolve, back-multiply) runs in FP32 (cuComplex: Cherk/Cheevd/Cgemm). The
// 4090's FP32 throughput is ~64× its FP64, and the dominant TEBD SVD is FP64-bound,
// so this is the big win on consumer GPUs. Precision: the Gram trick already
// squares the condition number; FP32 on top keeps ~3–4 digits in the kept Schmidt
// values — fine for real-time TEBD truncation (Trotter error dominates), NOT for
// energy-exact DMRG/TDVP. The final reorder + 1/σ scaling is done in double.
bool thin_svd_gram_gpu_c(const std::complex<double>* M, int m, int n,
                         std::complex<double>* U, std::complex<double>* V,
                         double* sv);

// Build one block-sparse theta block on the GPU (Phase B residency). Computes the
// two spin-sector contributions of form_theta entirely on-device, sharing a single
// upload of Ai1, so the O(n_row·kmid·n_col) GEMMs amortise the O(·²) transfers --
// the right granularity (whole big block, both GEMMs) vs the failed per-GEMM offload.
//   theta (n_row x n_col, n_row = dLup+dLdn), column-major:
//     rows [0, dLup)        = Aup (dLup x kmid) * Ai1 (kmid x n_col)   (s1=up)
//     rows [dLup, n_row)    = Adn (dLdn x kmid) * Ai1                   (s1=dn)
//   A missing side: pass its pointer = nullptr and dL = 0 -> those rows are zero.
//   Aup/Adn are dense column-major (lda = dLup/dLdn); Ai1 is dense (lda = kmid).
// Returns false on any device error -> caller falls back to the CPU form_theta.
bool form_theta_block_z(const std::complex<double>* Aup, int dLup,
                        const std::complex<double>* Adn, int dLdn,
                        const std::complex<double>* Ai1, int kmid, int ncol,
                        std::complex<double>* theta);

// Arbitrary local dimension d: nsec row groups (As[s] = (dLs[s] x kmid) column-major
// sub-block, nullptr/dLs[s]=0 = absent). theta row group s starts at the cumulative
// offset of the earlier dLs; one GEMM per present group, shared Ai1 upload. Used by
// the d>2 path; the 2-sector form_theta_block_z above stays the spin-1/2 route.
bool form_theta_block_z_multi(const std::complex<double>* const* As, const int* dLs,
                              int nsec, const std::complex<double>* Ai1, int kmid,
                              int ncol, int nrow, std::complex<double>* theta);

}  // namespace qx_gpu
