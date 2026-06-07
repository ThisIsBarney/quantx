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

}  // namespace qx_gpu
