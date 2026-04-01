// src/phi4.cpp — M24: φ⁴ Scalar Quantum Field Theory  (Lanczos + Krylov)
//
// Simulates the φ⁴ lattice QFT Hamiltonian:
//
//   H = Σᵢ [½πᵢ² + ½m²φᵢ² + (λ/4)φᵢ⁴] − κ Σᵢ φᵢφᵢ₊₁
//
// using a truncated Fock (harmonic oscillator) basis per site with d_local levels.
//
// This version replaces the dense matrix + power iteration with:
//   • CSR sparse Hamiltonian (build_phi4_csr)
//   • Lanczos eigensolver with full reorthogonalization (lanczos_eig)
//   • Arnoldi-Krylov time propagator exp(-i·dt·H)|ψ⟩  (arnoldi_expm)
//   • φ⁴ quantum quench dynamics ("dynamics" scenario)

#include "qx/engine.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qx {

namespace {

using cx   = std::complex<double>;
using Mat  = std::vector<double>;    // real, column-major
using MatC = std::vector<cx>;        // complex, column-major

// ─────────────────────────────────────────────────────────────────────────────
// A.  Local (single-site) operator matrices in truncated Fock space
// ─────────────────────────────────────────────────────────────────────────────

// â|n⟩ = √n |n-1⟩  →  a[n-1, n] = √n  (column-major)
static std::vector<double> make_a(int d) {
  std::vector<double> a(static_cast<std::size_t>(d) * d, 0.0);
  for (int n = 1; n < d; ++n)
    a[static_cast<std::size_t>(n - 1) + static_cast<std::size_t>(n) * d] =
        std::sqrt(static_cast<double>(n));
  return a;
}

// φ = (a + a†)/√2
static std::vector<double> make_phi(int d) {
  auto a = make_a(d);
  std::vector<double> phi(static_cast<std::size_t>(d) * d, 0.0);
  for (int r = 0; r < d; ++r)
    for (int c = 0; c < d; ++c) {
      std::size_t idx  = static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * d;
      std::size_t idxT = static_cast<std::size_t>(c) + static_cast<std::size_t>(r) * d;
      phi[idx] = (a[idx] + a[idxT]) / std::sqrt(2.0);
    }
  return phi;
}

// π² = -(a - a†)² / 2
static std::vector<double> make_pi2(int d) {
  auto a = make_a(d);
  std::vector<double> diff(static_cast<std::size_t>(d) * d, 0.0);
  for (int r = 0; r < d; ++r)
    for (int c = 0; c < d; ++c) {
      std::size_t idx  = static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * d;
      std::size_t idxT = static_cast<std::size_t>(c) + static_cast<std::size_t>(r) * d;
      diff[idx] = a[idx] - a[idxT];
    }
  std::vector<double> pi2(static_cast<std::size_t>(d) * d, 0.0);
  for (int r = 0; r < d; ++r)
    for (int c = 0; c < d; ++c) {
      double acc = 0.0;
      for (int k = 0; k < d; ++k)
        acc += diff[static_cast<std::size_t>(r) + static_cast<std::size_t>(k) * d] *
               diff[static_cast<std::size_t>(k) + static_cast<std::size_t>(c) * d];
      pi2[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * d] = -0.5 * acc;
    }
  return pi2;
}

// Dense d×d matmul (used to build phi², phi⁴ locally)
static std::vector<double> matmul(const std::vector<double>& A,
                                   const std::vector<double>& B, int d) {
  std::vector<double> C(static_cast<std::size_t>(d) * d, 0.0);
  for (int r = 0; r < d; ++r)
    for (int c = 0; c < d; ++c) {
      double acc = 0.0;
      for (int k = 0; k < d; ++k)
        acc += A[static_cast<std::size_t>(r) + static_cast<std::size_t>(k) * d] *
               B[static_cast<std::size_t>(k) + static_cast<std::size_t>(c) * d];
      C[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * d] = acc;
    }
  return C;
}

// ─────────────────────────────────────────────────────────────────────────────
// A.  CSR sparse matrix
// ─────────────────────────────────────────────────────────────────────────────

struct CsrMatrix {
  int dim{0};
  std::vector<double> vals;
  std::vector<int>    cols;
  std::vector<int>    row_ptr;   // length dim+1
};

// Build full Hamiltonian in CSR format.
static CsrMatrix build_phi4_csr(int N, int d, double m2, double lam,
                                  double kappa, double eps) {
  int dim_full = 1;
  for (int i = 0; i < N; ++i) dim_full *= d;

  // Local operators
  auto phi_loc  = make_phi(d);
  auto pi2_loc  = make_pi2(d);
  auto phi2_loc = matmul(phi_loc, phi_loc, d);
  auto phi4_loc = matmul(phi2_loc, phi2_loc, d);

  // COO accumulator: (row,col) → value
  std::map<std::pair<int,int>, double> coo;

  // Precompute site strides: stride[i] = d^i
  std::vector<int> strides(N, 1);
  for (int i = 1; i < N; ++i) strides[i] = strides[i - 1] * d;

  // ── Helper: add single-site operator at site `site` with coefficient `coeff` ──
  auto add_local_sparse = [&](const std::vector<double>& op, int site, double coeff) {
    int stride = strides[site];
    int block  = stride * d;
    int n_outer = dim_full / block;
    for (int outer = 0; outer < n_outer; ++outer) {
      for (int inner = 0; inner < stride; ++inner) {
        for (int r = 0; r < d; ++r) {
          for (int c = 0; c < d; ++c) {
            double v = op[static_cast<std::size_t>(r) +
                          static_cast<std::size_t>(c) * d];
            if (v == 0.0) continue;
            int row_full = outer * block + r * stride + inner;
            int col_full = outer * block + c * stride + inner;
            coo[{row_full, col_full}] += coeff * v;
          }
        }
      }
    }
  };

  // ── Helper: add two-site operator at sites s1, s2 ────────────────────────
  auto add_twosite_sparse = [&](const std::vector<double>& op1, int s1,
                                 const std::vector<double>& op2, int s2,
                                 double coeff) {
    int dim_other = dim_full / (d * d);
    for (int r1 = 0; r1 < d; ++r1)
      for (int c1 = 0; c1 < d; ++c1) {
        double v1 = op1[static_cast<std::size_t>(r1) +
                        static_cast<std::size_t>(c1) * d];
        if (v1 == 0.0) continue;
        for (int r2 = 0; r2 < d; ++r2)
          for (int c2 = 0; c2 < d; ++c2) {
            double v2 = op2[static_cast<std::size_t>(r2) +
                            static_cast<std::size_t>(c2) * d];
            if (v2 == 0.0) continue;
            int base_row = r1 * strides[s1] + r2 * strides[s2];
            int base_col = c1 * strides[s1] + c2 * strides[s2];
            for (int other = 0; other < dim_other; ++other) {
              int row_full = base_row, col_full = base_col;
              int tmp = other;
              for (int i = 0; i < N; ++i) {
                if (i == s1 || i == s2) continue;
                int di = tmp % d; tmp /= d;
                row_full += di * strides[i];
                col_full += di * strides[i];
              }
              coo[{row_full, col_full}] += coeff * v1 * v2;
            }
          }
      }
  };

  // ── Local terms ──────────────────────────────────────────────────────────
  for (int i = 0; i < N; ++i) {
    add_local_sparse(pi2_loc,  i, 0.5);
    add_local_sparse(phi2_loc, i, 0.5 * m2);
    add_local_sparse(phi4_loc, i, 0.25 * lam);
    add_local_sparse(phi_loc,  i, eps);
  }

  // ── Hopping terms: -κ φᵢφᵢ₊₁ ────────────────────────────────────────────
  for (int i = 0; i < N - 1; ++i)
    add_twosite_sparse(phi_loc, i, phi_loc, i + 1, -kappa);

  // ── COO → CSR ─────────────────────────────────────────────────────────────
  CsrMatrix H;
  H.dim = dim_full;
  H.row_ptr.resize(static_cast<std::size_t>(dim_full) + 1, 0);

  // Count non-zeros per row
  for (auto& kv : coo) {
    if (kv.second != 0.0)
      H.row_ptr[kv.first.first + 1]++;
  }
  // Prefix sum
  for (int r = 0; r < dim_full; ++r)
    H.row_ptr[r + 1] += H.row_ptr[r];

  int nnz = H.row_ptr[dim_full];
  H.vals.resize(nnz);
  H.cols.resize(nnz);

  // Fill values
  std::vector<int> pos(H.row_ptr.begin(), H.row_ptr.end() - 1);
  for (auto& kv : coo) {
    if (kv.second == 0.0) continue;
    int r = kv.first.first;
    int p = pos[r]++;
    H.vals[p] = kv.second;
    H.cols[p] = kv.first.second;
  }

  return H;
}

// y = H * x  (real)
static void csr_matvec(const CsrMatrix& H,
                        const std::vector<double>& x,
                        std::vector<double>& y) {
  int dim = H.dim;
  y.assign(dim, 0.0);
  for (int r = 0; r < dim; ++r) {
    double acc = 0.0;
    for (int p = H.row_ptr[r]; p < H.row_ptr[r + 1]; ++p)
      acc += H.vals[p] * x[H.cols[p]];
    y[r] = acc;
  }
}

// y = H * x  (complex)
static void csr_matvec_cx(const CsrMatrix& H,
                           const std::vector<cx>& x,
                           std::vector<cx>& y) {
  int dim = H.dim;
  y.assign(dim, cx{0.0, 0.0});
  for (int r = 0; r < dim; ++r) {
    cx acc{0.0, 0.0};
    for (int p = H.row_ptr[r]; p < H.row_ptr[r + 1]; ++p)
      acc += H.vals[p] * x[H.cols[p]];
    y[r] = acc;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  Tridiagonal eigensolver  (implicit symmetric QR with Wilkinson shift)
// ─────────────────────────────────────────────────────────────────────────────
//
// alpha:    diagonal,     length k
// beta_sub: sub-diagonal, length k-1
// Returns (eigenvalues sorted ascending,
//          eigenvectors as columns in k×k column-major matrix)

static std::pair<std::vector<double>, std::vector<double>>
tridiag_eig(const std::vector<double>& alpha_in,
            const std::vector<double>& beta_in) {
  int k = static_cast<int>(alpha_in.size());
  std::vector<double> a = alpha_in;
  std::vector<double> b = beta_in;   // length k-1
  // Eigenvector accumulator: k×k identity
  std::vector<double> Q(static_cast<std::size_t>(k) * k, 0.0);
  for (int i = 0; i < k; ++i)
    Q[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * k] = 1.0;

  int max_iter = 30 * k;
  for (int iter = 0; iter < max_iter; ++iter) {
    // Deflate: find the largest l such that b[l-1] is negligible
    int n = k;
    while (n > 1 &&
           std::abs(b[n - 2]) <
               1e-10 * (std::abs(a[n - 2]) + std::abs(a[n - 1])))
      --n;
    if (n == 1) break;

    // Find smallest m (start of unreduced block)
    int m = n - 1;
    while (m > 0 &&
           std::abs(b[m - 1]) >=
               1e-10 * (std::abs(a[m - 1]) + std::abs(a[m])))
      --m;

    // Wilkinson shift from bottom-right 2×2 submatrix of [m..n-1]
    double d_w = (a[n - 2] - a[n - 1]) * 0.5;
    double sign_d = (d_w >= 0.0) ? 1.0 : -1.0;
    double mu = a[n - 1] -
                b[n - 2] * b[n - 2] /
                    (d_w + sign_d * std::sqrt(d_w * d_w + b[n - 2] * b[n - 2]));

    double x = a[m] - mu;
    double z = b[m];

    for (int i = m; i < n - 1; ++i) {
      // Compute Givens rotation
      double r_g = std::sqrt(x * x + z * z);
      if (r_g < 1e-300) break;
      double c = x / r_g, s = z / r_g;

      // Apply rotation to tridiagonal
      if (i > m) b[i - 1] = r_g;
      double a_i   = a[i],   a_i1 = a[i + 1];
      double b_i   = b[i];
      a[i]   = c * c * a_i + 2.0 * c * s * b_i + s * s * a_i1;
      a[i+1] = s * s * a_i - 2.0 * c * s * b_i + c * c * a_i1;
      b[i]   = c * s * (a_i1 - a_i) + (c * c - s * s) * b_i;
      if (i < n - 2) {
        double b_i1 = b[i + 1];
        b[i + 1] = c * b_i1;
        z = s * b_i1;
        x = b[i];
      }

      // Accumulate rotation into Q: Q[:, i:i+2] *= G
      for (int j = 0; j < k; ++j) {
        double qi  = Q[static_cast<std::size_t>(j) + static_cast<std::size_t>(i) * k];
        double qi1 = Q[static_cast<std::size_t>(j) + static_cast<std::size_t>(i + 1) * k];
        Q[static_cast<std::size_t>(j) + static_cast<std::size_t>(i)     * k] =  c * qi + s * qi1;
        Q[static_cast<std::size_t>(j) + static_cast<std::size_t>(i + 1) * k] = -s * qi + c * qi1;
      }
    }
  }

  // Sort eigenvalues (and permute eigenvectors accordingly)
  std::vector<int> idx(k);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int i, int j) { return a[i] < a[j]; });

  std::vector<double> evals(k);
  std::vector<double> evecs(static_cast<std::size_t>(k) * k);
  for (int j = 0; j < k; ++j) {
    evals[j] = a[idx[j]];
    for (int r = 0; r < k; ++r)
      evecs[static_cast<std::size_t>(r) + static_cast<std::size_t>(j) * k] =
          Q[static_cast<std::size_t>(r) + static_cast<std::size_t>(idx[j]) * k];
  }
  return {evals, evecs};
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  Lanczos eigensolver
// ─────────────────────────────────────────────────────────────────────────────

struct EigPair {
  double eval{0.0};
  std::vector<double> evec;
};

// Returns {ground state, first excited state}
static std::pair<EigPair, EigPair>
lanczos_eig(const CsrMatrix& H, int seed_val = 42) {
  int dim = H.dim;

  // Guard: too large
  if (dim > 2000000) {
    EigPair bad;
    bad.eval = 0.0;
    return {bad, bad};
  }

  // Determine k_max
  int k_max = std::min(200, dim / 4 + 10);
  // Memory guard: cap if needed
  {
    long long mem_bytes = static_cast<long long>(k_max) * dim * 8;
    if (dim > 500000 || mem_bytes > 1500LL * 1024 * 1024)
      k_max = std::min(50, dim / 10);
  }
  k_max = std::max(k_max, 4);

  // Random start vector via simple LCG
  std::vector<double> v(dim);
  {
    std::uint64_t lcg = static_cast<std::uint64_t>(seed_val) * 6364136223846793005ULL + 1442695040888963407ULL;
    for (int i = 0; i < dim; ++i) {
      lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
      // Map to [-1,1]
      v[i] = static_cast<double>(static_cast<std::int64_t>(lcg >> 33)) / static_cast<double>(1LL << 30);
    }
    double norm = 0.0;
    for (auto x : v) norm += x * x;
    norm = std::sqrt(norm);
    for (auto& x : v) x /= norm;
  }

  // Lanczos recurrence storage
  std::vector<double> alpha, beta;     // Ritz tridiagonal
  std::vector<std::vector<double>> Q;  // all Lanczos vectors
  Q.reserve(k_max + 1);

  Q.push_back(v);                       // Q[0] = v
  std::vector<double> w(dim);
  double beta_prev = 0.0;

  int k_actual = 0;
  for (int j = 0; j < k_max; ++j) {
    // w = H * Q[j]
    csr_matvec(H, Q[j], w);

    // alpha_j = <Q[j], w>
    double alpha_j = 0.0;
    for (int i = 0; i < dim; ++i) alpha_j += Q[j][i] * w[i];
    alpha.push_back(alpha_j);

    // w -= alpha_j * Q[j]
    for (int i = 0; i < dim; ++i) w[i] -= alpha_j * Q[j][i];
    // w -= beta_prev * Q[j-1]
    if (j > 0)
      for (int i = 0; i < dim; ++i) w[i] -= beta_prev * Q[j - 1][i];

    // Full reorthogonalisation against all stored vectors (modified Gram-Schmidt)
    for (int l = 0; l <= j; ++l) {
      double dot = 0.0;
      for (int i = 0; i < dim; ++i) dot += Q[l][i] * w[i];
      for (int i = 0; i < dim; ++i) w[i] -= dot * Q[l][i];
    }
    // Second pass for numerical stability
    for (int l = 0; l <= j; ++l) {
      double dot = 0.0;
      for (int i = 0; i < dim; ++i) dot += Q[l][i] * w[i];
      for (int i = 0; i < dim; ++i) w[i] -= dot * Q[l][i];
    }

    // beta_{j+1} = ||w||
    double beta_j = 0.0;
    for (auto x : w) beta_j += x * x;
    beta_j = std::sqrt(beta_j);

    k_actual = j + 1;
    if (beta_j < 1e-12) break;

    beta.push_back(beta_j);
    beta_prev = beta_j;

    // Normalise and store
    std::vector<double> qnew(dim);
    for (int i = 0; i < dim; ++i) qnew[i] = w[i] / beta_j;
    Q.push_back(qnew);
  }

  // Solve tridiagonal eigenproblem of size k_actual × k_actual
  auto [evals, evecs_k] = tridiag_eig(alpha, beta);

  // Back-transform Ritz vectors: evec_k = sum_j Y[j, col] * Q[j]
  auto make_ritz = [&](int col) -> EigPair {
    if (col >= k_actual) col = k_actual - 1;
    EigPair ep;
    ep.eval = evals[col];
    ep.evec.assign(dim, 0.0);
    for (int j = 0; j < k_actual; ++j) {
      double y = evecs_k[static_cast<std::size_t>(j) +
                          static_cast<std::size_t>(col) * k_actual];
      for (int i = 0; i < dim; ++i)
        ep.evec[i] += y * Q[j][i];
    }
    // Normalise
    double n2 = 0.0;
    for (auto x : ep.evec) n2 += x * x;
    n2 = std::sqrt(n2);
    if (n2 > 1e-14)
      for (auto& x : ep.evec) x /= n2;
    return ep;
  };

  return {make_ritz(0), make_ritz(1)};
}

// ─────────────────────────────────────────────────────────────────────────────
// E.  Arnoldi-Krylov matrix exponential  exp(-i·dt·H)|ψ⟩
// ─────────────────────────────────────────────────────────────────────────────

// Dense matrix exponential of a small complex m×m matrix A (column-major).
// Uses Taylor series with scaling-and-squaring.
static MatC dense_expm(const MatC& A, int m) {
  // 1-norm of A
  double norm1 = 0.0;
  for (int c = 0; c < m; ++c) {
    double col_sum = 0.0;
    for (int r = 0; r < m; ++r)
      col_sum += std::abs(A[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * m]);
    norm1 = std::max(norm1, col_sum);
  }

  int s = 1;
  if (norm1 > 1.0) s = static_cast<int>(std::ceil(norm1));

  // Scale A by 1/s
  MatC As(static_cast<std::size_t>(m) * m);
  for (std::size_t i = 0; i < static_cast<std::size_t>(m) * m; ++i)
    As[i] = A[i] / static_cast<double>(s);

  // Taylor series: E = I + A + A²/2! + ... up to order 14
  // E = sum_{k=0}^{14} A^k / k!
  MatC E(static_cast<std::size_t>(m) * m, cx{0.0, 0.0});
  // Set E = I
  for (int i = 0; i < m; ++i)
    E[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * m] = cx{1.0, 0.0};

  MatC Ak = E;  // A^k, starts as I
  double fact = 1.0;
  for (int k = 1; k <= 14; ++k) {
    fact /= static_cast<double>(k);
    // Ak = Ak * As
    MatC Ak_new(static_cast<std::size_t>(m) * m, cx{0.0, 0.0});
    for (int r = 0; r < m; ++r)
      for (int c = 0; c < m; ++c) {
        cx acc{0.0, 0.0};
        for (int l = 0; l < m; ++l)
          acc += Ak [static_cast<std::size_t>(r) + static_cast<std::size_t>(l) * m] *
                 As [static_cast<std::size_t>(l) + static_cast<std::size_t>(c) * m];
        Ak_new[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * m] = acc;
      }
    Ak = Ak_new;
    // E += fact * Ak
    for (std::size_t i = 0; i < static_cast<std::size_t>(m) * m; ++i)
      E[i] += fact * Ak[i];
  }

  // Squaring phase: exp(A) = exp(A/s)^s via s-1 matrix multiplications by exp(A/s).
  // Save E_orig = exp(A/s) and accumulate E = E * E_orig repeatedly so that after
  // sq iterations E = exp(A/s)^(sq+1).  Using E = E * E would give exp(A/s)^(2^(sq))
  // which is wrong for non-power-of-two s.
  if (s > 1) {
    MatC E_orig = E;  // exp(A/s)
    for (int sq = 1; sq < s; ++sq) {
      MatC E2(static_cast<std::size_t>(m) * m, cx{0.0, 0.0});
      for (int r = 0; r < m; ++r)
        for (int c = 0; c < m; ++c) {
          cx acc{0.0, 0.0};
          for (int l = 0; l < m; ++l)
            acc += E[static_cast<std::size_t>(r) + static_cast<std::size_t>(l) * m] *
                   E_orig[static_cast<std::size_t>(l) + static_cast<std::size_t>(c) * m];
          E2[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * m] = acc;
        }
      E = E2;
    }
  }
  return E;
}

// psi_new ≈ exp(-i*dt*H)*psi  using m-step Arnoldi
static std::pair<std::vector<cx>, double>
arnoldi_expm(const CsrMatrix& H,
             const std::vector<cx>& psi,
             double dt, int m = 20) {
  int dim = H.dim;
  m = std::min(m, dim);

  // beta = ||psi||
  double beta = 0.0;
  for (auto& z : psi) beta += std::norm(z);
  beta = std::sqrt(beta);
  if (beta < 1e-300) return {psi, 0.0};

  // Arnoldi basis V[0..m-1] and upper Hessenberg H_m (m×m, col-major)
  std::vector<std::vector<cx>> V;
  V.reserve(m + 1);
  {
    std::vector<cx> v0(dim);
    for (int i = 0; i < dim; ++i) v0[i] = psi[i] / beta;
    V.push_back(v0);
  }

  MatC Hm(static_cast<std::size_t>(m) * m, cx{0.0, 0.0});
  std::vector<cx> w(dim);
  int m_actual = m;

  for (int j = 0; j < m; ++j) {
    csr_matvec_cx(H, V[j], w);

    // Modified Gram-Schmidt
    for (int i = 0; i <= j; ++i) {
      cx hij{0.0, 0.0};
      for (int l = 0; l < dim; ++l)
        hij += std::conj(V[i][l]) * w[l];
      Hm[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * m] = hij;
      for (int l = 0; l < dim; ++l)
        w[l] -= hij * V[i][l];
    }

    // h_{j+1,j}
    double hnext = 0.0;
    for (auto& z : w) hnext += std::norm(z);
    hnext = std::sqrt(hnext);

    if (hnext < 1e-14) {
      m_actual = j + 1;
      break;
    }
    if (j < m - 1) {
      Hm[static_cast<std::size_t>(j + 1) + static_cast<std::size_t>(j) * m] =
          cx{hnext, 0.0};
      std::vector<cx> vnew(dim);
      for (int l = 0; l < dim; ++l) vnew[l] = w[l] / hnext;
      V.push_back(vnew);
    }
  }

  // Build the m_actual × m_actual submatrix of Hm scaled by -i*dt
  MatC Hdt(static_cast<std::size_t>(m_actual) * m_actual, cx{0.0, 0.0});
  for (int r = 0; r < m_actual; ++r)
    for (int c = 0; c < m_actual; ++c)
      Hdt[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * m_actual] =
          cx{0.0, -dt} * Hm[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * m];

  // exp(-i*dt*H_m)
  MatC expHm = dense_expm(Hdt, m_actual);

  // phi_m = expHm * e1 * beta   (e1 = [1,0,...,0])
  std::vector<cx> phi_m(m_actual);
  for (int r = 0; r < m_actual; ++r)
    phi_m[r] = expHm[static_cast<std::size_t>(r)] * beta;   // col 0

  // psi_new = sum_j phi_m[j] * V[j]
  std::vector<cx> psi_new(dim, cx{0.0, 0.0});
  for (int j = 0; j < m_actual; ++j)
    for (int i = 0; i < dim; ++i)
      psi_new[i] += phi_m[j] * V[j][i];

  // Error estimate
  double err = 0.0;
  if (m_actual >= 2) {
    double hm_sub = std::abs(
        Hm[static_cast<std::size_t>(m_actual - 1) +
           static_cast<std::size_t>(m_actual - 2) * m]);
    err = hm_sub * std::abs(phi_m[m_actual - 1]) * std::abs(dt);
  }

  return {psi_new, err};
}

// ─────────────────────────────────────────────────────────────────────────────
// F.  Bipartite entanglement entropy
// ─────────────────────────────────────────────────────────────────────────────

// Householder reduction of a real symmetric n×n matrix (column-major) to
// tridiagonal form.  Returns (alpha, beta) for tridiag_eig.
static std::pair<std::vector<double>, std::vector<double>>
householder_tridiag_real(std::vector<double>& A, int n) {
  for (int k = 0; k < n - 2; ++k) {
    // Reflection to zero out A[k+2..n-1, k]
    double sigma = 0.0;
    for (int i = k + 1; i < n; ++i)
      sigma += A[static_cast<std::size_t>(i) + static_cast<std::size_t>(k) * n] *
               A[static_cast<std::size_t>(i) + static_cast<std::size_t>(k) * n];
    double alpha_hh = std::sqrt(sigma);
    double x1 = A[static_cast<std::size_t>(k + 1) + static_cast<std::size_t>(k) * n];
    if (x1 > 0.0) alpha_hh = -alpha_hh;

    double r2 = sigma - x1 * alpha_hh;  // = 0.5 * ||v||²  where v = x - alpha*e1
    if (r2 < 1e-30) continue;

    // Build Householder vector v = x - alpha*e1  (stored in place starting at k+1)
    A[static_cast<std::size_t>(k + 1) + static_cast<std::size_t>(k) * n] -= alpha_hh;

    // p = A[k+1:, k+1:] * v / r2
    std::vector<double> v(n - k - 1);
    for (int i = 0; i < n - k - 1; ++i)
      v[i] = A[static_cast<std::size_t>(k + 1 + i) + static_cast<std::size_t>(k) * n];

    // q = A_sub * v
    std::vector<double> q(n - k - 1, 0.0);
    for (int r = 0; r < n - k - 1; ++r)
      for (int c = 0; c < n - k - 1; ++c)
        q[r] += A[static_cast<std::size_t>(k + 1 + r) +
                  static_cast<std::size_t>(k + 1 + c) * n] *
                v[c];

    double vTq = 0.0;
    for (int i = 0; i < n - k - 1; ++i) vTq += v[i] * q[i];

    // w = q/r2 - (vTq/(2*r2^2)) * v
    std::vector<double> w(n - k - 1);
    for (int i = 0; i < n - k - 1; ++i)
      w[i] = q[i] / r2 - (vTq / (2.0 * r2 * r2)) * v[i];

    // A_sub -= v*w^T + w*v^T
    for (int r = 0; r < n - k - 1; ++r)
      for (int c = 0; c < n - k - 1; ++c) {
        A[static_cast<std::size_t>(k + 1 + r) +
          static_cast<std::size_t>(k + 1 + c) * n] -=
            v[r] * w[c] + w[r] * v[c];
      }

    // Restore off-diag so diagonal holds the tridiagonal value
    A[static_cast<std::size_t>(k + 1) + static_cast<std::size_t>(k) * n] = alpha_hh;
    A[static_cast<std::size_t>(k) + static_cast<std::size_t>(k + 1) * n] = alpha_hh;
  }

  std::vector<double> diag(n), sub(n - 1);
  for (int i = 0; i < n; ++i)
    diag[i] = A[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * n];
  for (int i = 0; i < n - 1; ++i)
    sub[i] = A[static_cast<std::size_t>(i + 1) + static_cast<std::size_t>(i) * n];
  return {diag, sub};
}

// Householder reduction of a complex Hermitian n×n matrix (column-major) to
// real symmetric tridiagonal form.  Returns (alpha, beta) for tridiag_eig.
// Reference: Golub & Van Loan, "Matrix Computations", 4th ed., §8.3.
static std::pair<std::vector<double>, std::vector<double>>
householder_tridiag_herm(std::vector<cx>& A, int n) {
  for (int k = 0; k < n - 2; ++k) {
    // Compute ||x|| where x = A[k+1:, k]
    double sigma = 0.0;
    for (int i = k + 1; i < n; ++i)
      sigma += std::norm(A[static_cast<std::size_t>(i) + static_cast<std::size_t>(k) * n]);
    double alpha_hh = std::sqrt(sigma);

    cx x1 = A[static_cast<std::size_t>(k + 1) + static_cast<std::size_t>(k) * n];
    // Choose sign to avoid cancellation: multiply by -phase(x1)
    cx phase_x1 = (std::abs(x1) < 1e-300) ? cx{1.0, 0.0} : x1 / std::abs(x1);
    alpha_hh = std::abs(alpha_hh);  // real positive

    double r2 = sigma - std::abs(x1) * alpha_hh;  // 0.5 * ||v||²
    if (r2 < 1e-30) continue;

    // Build Householder vector v (stored in place at column k, rows k+1..n-1)
    A[static_cast<std::size_t>(k + 1) + static_cast<std::size_t>(k) * n] -= phase_x1 * alpha_hh;

    std::vector<cx> v(n - k - 1);
    for (int i = 0; i < n - k - 1; ++i)
      v[i] = A[static_cast<std::size_t>(k + 1 + i) + static_cast<std::size_t>(k) * n];

    // q = A_sub * v  (A_sub is Hermitian)
    std::vector<cx> q(n - k - 1, cx{0.0, 0.0});
    for (int r = 0; r < n - k - 1; ++r)
      for (int c = 0; c < n - k - 1; ++c)
        q[r] += A[static_cast<std::size_t>(k + 1 + r) +
                  static_cast<std::size_t>(k + 1 + c) * n] * v[c];

    cx vHq{0.0, 0.0};
    for (int i = 0; i < n - k - 1; ++i) vHq += std::conj(v[i]) * q[i];

    // w = q/r2 - (Re(vHq) / (2*r2^2)) * v  — uses real part only since A Hermitian
    std::vector<cx> w(n - k - 1);
    for (int i = 0; i < n - k - 1; ++i)
      w[i] = q[i] / r2 - (vHq.real() / (2.0 * r2 * r2)) * v[i];

    // A_sub -= v*w† + w*v†  (keeps Hermitian structure)
    for (int r = 0; r < n - k - 1; ++r)
      for (int c = 0; c < n - k - 1; ++c) {
        A[static_cast<std::size_t>(k + 1 + r) +
          static_cast<std::size_t>(k + 1 + c) * n] -=
            v[r] * std::conj(w[c]) + w[r] * std::conj(v[c]);
      }

    // Write tridiagonal off-diagonal (real by construction after the reflection)
    cx off = phase_x1 * alpha_hh;
    A[static_cast<std::size_t>(k + 1) + static_cast<std::size_t>(k) * n] = off;
    A[static_cast<std::size_t>(k) + static_cast<std::size_t>(k + 1) * n] = std::conj(off);
  }

  std::vector<double> diag(n), sub(n - 1);
  for (int i = 0; i < n; ++i)
    diag[i] = A[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * n].real();
  for (int i = 0; i < n - 1; ++i)
    sub[i] = std::abs(A[static_cast<std::size_t>(i + 1) + static_cast<std::size_t>(i) * n]);
  return {diag, sub};
}

static double entanglement_entropy(const std::vector<cx>& psi, int N, int d) {
  int dimA = 1, dimB = 1;
  for (int i = 0; i < N / 2; ++i)     dimA *= d;
  for (int i = N / 2; i < N; ++i)     dimB *= d;

  if (dimA > 400) return -1.0;

  // Build M†M  (dimA × dimA complex Hermitian)
  // M[a, b] = psi[a * dimB + b]
  // (M†M)[a1, a2] = sum_b conj(psi[a1*dimB+b]) * psi[a2*dimB+b]
  std::vector<cx> MtM(static_cast<std::size_t>(dimA) * dimA, cx{0.0, 0.0});
  for (int a1 = 0; a1 < dimA; ++a1)
    for (int a2 = a1; a2 < dimA; ++a2) {
      cx val{0.0, 0.0};
      for (int b = 0; b < dimB; ++b) {
        cx p1 = psi[static_cast<std::size_t>(a1) * dimB + b];
        cx p2 = psi[static_cast<std::size_t>(a2) * dimB + b];
        val += std::conj(p1) * p2;
      }
      MtM[static_cast<std::size_t>(a1) + static_cast<std::size_t>(a2) * dimA] = val;
      MtM[static_cast<std::size_t>(a2) + static_cast<std::size_t>(a1) * dimA] = std::conj(val);
    }

  if (dimA == 1) {
    double p = MtM[0].real();
    return (p > 1e-14) ? -p * std::log(p) : 0.0;
  }

  auto [diag, sub] = householder_tridiag_herm(MtM, dimA);
  auto [evals, _evecs] = tridiag_eig(diag, sub);

  // Compute trace (norm²) for normalisation
  double trace = 0.0;
  for (auto e : evals) trace += (e > 0.0 ? e : 0.0);
  if (trace < 1e-28) return 0.0;

  double S = 0.0;
  for (auto e : evals) {
    if (e <= 0.0) continue;
    double p = e / trace;
    if (p > 1e-14) S -= p * std::log(p);
  }
  return S;
}

// ─────────────────────────────────────────────────────────────────────────────
// Observable expectation values
// ─────────────────────────────────────────────────────────────────────────────

// Real psi version
static double expect_local_real(const std::vector<double>& psi,
                                  const std::vector<double>& op,
                                  int site, int N, int d) {
  int dim_full = static_cast<int>(psi.size());
  double result = 0.0;
  int stride = 1;
  for (int i = 0; i < site; ++i) stride *= d;
  int block = stride * d;
  for (int outer = 0; outer < dim_full / block; ++outer)
    for (int inner = 0; inner < stride; ++inner)
      for (int r = 0; r < d; ++r)
        for (int c = 0; c < d; ++c) {
          double v = op[static_cast<std::size_t>(r) +
                        static_cast<std::size_t>(c) * d];
          if (v == 0.0) continue;
          int rf = outer * block + r * stride + inner;
          int cf = outer * block + c * stride + inner;
          result += psi[rf] * v * psi[cf];
        }
  return result;
}

// Complex psi version — O is real so <psi|O|psi> is purely real
static double expect_local_cx(const std::vector<cx>& psi,
                                const std::vector<double>& op,
                                int site, int N, int d) {
  int dim_full = static_cast<int>(psi.size());
  double result = 0.0;
  int stride = 1;
  for (int i = 0; i < site; ++i) stride *= d;
  int block = stride * d;
  for (int outer = 0; outer < dim_full / block; ++outer)
    for (int inner = 0; inner < stride; ++inner)
      for (int r = 0; r < d; ++r)
        for (int c = 0; c < d; ++c) {
          double v = op[static_cast<std::size_t>(r) +
                        static_cast<std::size_t>(c) * d];
          if (v == 0.0) continue;
          int rf = outer * block + r * stride + inner;
          int cf = outer * block + c * stride + inner;
          // Re(conj(psi[rf]) * v * psi[cf])
          result += (psi[rf].real() * psi[cf].real() +
                     psi[rf].imag() * psi[cf].imag()) * v;
        }
  return result;
}

// Energy <psi|H|psi> for complex psi
static double energy_expect_cx(const CsrMatrix& H,
                                 const std::vector<cx>& psi) {
  int dim = H.dim;
  std::vector<cx> Hpsi(dim);
  csr_matvec_cx(H, psi, Hpsi);
  double re = 0.0;
  for (int i = 0; i < dim; ++i)
    re += psi[i].real() * Hpsi[i].real() + psi[i].imag() * Hpsi[i].imag();
  return re;
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

Phi4Result run_phi4(const Phi4Options& opts) {
  auto t0 = std::chrono::steady_clock::now();

  const int N = static_cast<int>(opts.n_sites);
  const int d = static_cast<int>(opts.d_local);

  auto make_err = [&](const std::string& msg) {
    Phi4Result r;
    r.ok = false;
    r.n_sites = static_cast<std::uint32_t>(N);
    r.d_local = static_cast<std::uint32_t>(d);
    r.m2 = opts.m2;
    r.lambda_coupling = opts.lambda_coupling;
    r.message = msg;
    return r;
  };

  if (N < 2 || N > 8)
    return make_err("n_sites must be in [2, 8]");
  if (d < 4 || d > 12)
    return make_err("d_local must be in [4, 12]");

  int dim_full = 1;
  for (int i = 0; i < N; ++i) dim_full *= d;

  if (dim_full > 2000000)
    return make_err("system too large (dim > 2M); reduce n_sites or d_local");

  Phi4Result result;
  result.ok     = true;
  result.n_sites = static_cast<std::uint32_t>(N);
  result.d_local = static_cast<std::uint32_t>(d);
  result.m2      = opts.m2;
  result.lambda_coupling = opts.lambda_coupling;

  // ── Pre-compute local operators for observables ───────────────────────────
  auto phi_loc  = make_phi(d);
  auto phi2_loc = matmul(phi_loc, phi_loc, d);
  std::vector<double> n_loc(static_cast<std::size_t>(d) * d, 0.0);
  for (int n = 0; n < d; ++n)
    n_loc[static_cast<std::size_t>(n) + static_cast<std::size_t>(n) * d] =
        static_cast<double>(n);

  // ── Helper: run Lanczos for a given m2 value ──────────────────────────────
  auto run_scenario = [&](double m2_val) ->
      std::tuple<double, double, double,
                 std::vector<double>, std::vector<double>, std::vector<double>> {
    CsrMatrix H = build_phi4_csr(N, d, m2_val, opts.lambda_coupling,
                                  opts.kappa, opts.epsilon);
    auto [gs, es] = lanczos_eig(H, static_cast<int>(opts.seed));

    double gap = es.eval - gs.eval;

    std::vector<double> phi_exp(N), phi2_exp(N), n_exp(N);
    for (int i = 0; i < N; ++i) {
      phi_exp[i]  = expect_local_real(gs.evec, phi_loc,  i, N, d);
      phi2_exp[i] = expect_local_real(gs.evec, phi2_loc, i, N, d);
      n_exp[i]    = expect_local_real(gs.evec, n_loc,    i, N, d);
    }

    double op = 0.0;
    for (auto x : phi2_exp) op += x;
    op = std::sqrt(op / N);

    return {gs.eval, gap, op, phi_exp, phi2_exp, n_exp};
  };

  // ── Dynamics scenario ─────────────────────────────────────────────────────
  if (opts.scenario == "dynamics") {
    // Build pre-quench Hamiltonian and find ground state
    CsrMatrix H_init = build_phi4_csr(N, d, opts.m2_initial, opts.lambda_coupling,
                                       opts.kappa, opts.epsilon);
    auto [gs_init, _es_init] = lanczos_eig(H_init, static_cast<int>(opts.seed));

    // Build post-quench Hamiltonian
    CsrMatrix H_final = build_phi4_csr(N, d, opts.m2_final, opts.lambda_coupling,
                                        opts.kappa, opts.epsilon);
    auto [gs_final, es_final] = lanczos_eig(H_final, static_cast<int>(opts.seed));
    result.ground_energy  = gs_final.eval;
    result.energy_gap     = es_final.eval - gs_final.eval;

    // Set initial state as complex vector
    std::vector<cx> psi(dim_full);
    for (int i = 0; i < dim_full; ++i)
      psi[i] = cx{gs_init.evec[i], 0.0};

    const int n_steps = static_cast<int>(opts.n_steps);
    result.dynamics.reserve(n_steps + 1);

    for (int step = 0; step <= n_steps; ++step) {
      Phi4DynamicsFrame frame;
      frame.time = step * opts.dt;

      // Per-site observables
      frame.phi_expect.resize(N);
      frame.occupation.resize(N);
      double phi2_sum = 0.0;
      for (int i = 0; i < N; ++i) {
        frame.phi_expect[i] = expect_local_cx(psi, phi_loc,  i, N, d);
        frame.occupation[i] = expect_local_cx(psi, n_loc,    i, N, d);
        phi2_sum += expect_local_cx(psi, phi2_loc, i, N, d);
      }
      frame.order_param = std::sqrt(phi2_sum / N);
      frame.entropy     = entanglement_entropy(psi, N, d);
      frame.energy      = energy_expect_cx(H_final, psi);

      result.dynamics.push_back(std::move(frame));

      if (step < n_steps) {
        auto [psi_new, _err] = arnoldi_expm(H_final, psi, opts.dt, 20);
        // Renormalise
        double norm2 = 0.0;
        for (auto& z : psi_new) norm2 += std::norm(z);
        double inv_norm = 1.0 / std::sqrt(norm2);
        for (auto& z : psi_new) z *= inv_norm;
        psi = std::move(psi_new);
      }
    }

    auto t1 = std::chrono::steady_clock::now();
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
  }

  // ── Static scenarios: symmetric / broken / sweep ──────────────────────────
  auto [E0, gap, op, phi_exp, phi2_exp, n_exp] = run_scenario(opts.m2);
  result.ground_energy   = E0;
  result.energy_gap      = gap;
  result.order_parameter = op;
  result.phi_expectation  = phi_exp;
  result.phi2_expectation = phi2_exp;
  result.occupation       = n_exp;

  // Field distribution at site 0 (from ground-state Ritz vector)
  // Rebuild H and run Lanczos one more time to get evec for distribution
  {
    CsrMatrix H = build_phi4_csr(N, d, opts.m2, opts.lambda_coupling,
                                  opts.kappa, opts.epsilon);
    auto [gs, _es] = lanczos_eig(H, static_cast<int>(opts.seed));
    // p(φ₀) ∝ Σ_{i₁..i_{N-1}} |⟨i₀,i₁,..,i_{N-1}|ψ⟩|²  binned by i₀
    result.field_distribution.assign(d, 0.0);
    int stride0 = 1;  // site 0 has stride = d^0 = 1
    int block0  = d;
    int n_outer = dim_full / block0;
    for (int outer = 0; outer < n_outer; ++outer)
      for (int i0 = 0; i0 < d; ++i0) {
        double amp = gs.evec[outer * block0 + i0];
        result.field_distribution[i0] += amp * amp;
      }
  }

  // m² sweep
  if (opts.scenario == "sweep" && opts.n_sweep > 0) {
    for (std::uint32_t i = 0; i < opts.n_sweep; ++i) {
      double m2_val = opts.m2_min +
                      (opts.m2_max - opts.m2_min) *
                          static_cast<double>(i) /
                          std::max(1u, opts.n_sweep - 1);
      auto [Ei, gi, opi, pe, p2e, ne] = run_scenario(m2_val);
      result.sweep.push_back({m2_val, Ei, opi, gi});
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  result.elapsed_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  return result;
}

}  // namespace qx
