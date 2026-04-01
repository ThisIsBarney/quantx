// src/floquet.cpp — M27-D: Floquet Kicked-Ising Model
//
// Simulates the kicked-Ising Hamiltonian:
//
//   H₀ = −J Σ_{⟨i,i+1⟩} Z_i Z_{i+1}  − h_z Σ_i Z_i
//
// under periodic delta-function kicks:
//
//   U_kick = exp(−i h_x Σ_i X_i)  = ⊗_i exp(−i h_x X_i)
//
// One Floquet period: U(T) = exp(−i·(T/2)·H₀) · U_kick · exp(−i·(T/2)·H₀)
//   with T = 2π/ω.
//
// Hilbert space: N qubits, dim = 2^N (Z-basis |↑↓…⟩).
// Quasi-energies: eigenphases of U(T) in [−π, π).
// Dynamics scenario: stroboscopic snapshots at multiples of T.
// Spectrum scenario: Floquet quasi-energy spectrum vs drive amplitude.
// Heating scenario: entanglement entropy growth vs period number.
//
// Implementation:
//   • CSR sparse H₀ for the Ising-longitudinal part
//   • Arnoldi-Krylov matrix exponential exp(−i τ H₀)|ψ⟩  (τ = T/2)
//   • Single-site X kicks applied exactly (tensor product)
//   • Floquet spectrum via full diagonalization (dim ≤ 4096 for N ≤ 12)
//   • Von Neumann bipartite entropy at N/2 cut via Householder + tridiag_eig

#include "qx/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qx {
namespace {

using cx   = std::complex<double>;
using MatC = std::vector<cx>;

static constexpr double PI = 3.141592653589793238462643383279502884;

// ─────────────────────────────────────────────────────────────────────────────
// A.  CSR for H₀ = −J Σ Z_i Z_{i+1}  − h_z Σ Z_i  (diagonal in Z basis)
// ─────────────────────────────────────────────────────────────────────────────
// Since H₀ is entirely diagonal in the computational basis, we store just the
// diagonal vector (not a full CSR) — this gives exact matvec in O(dim).

static std::vector<double> build_h0_diag(int n_sites, double J, double h_z) {
  int dim = 1 << n_sites;
  std::vector<double> diag(dim, 0.0);
  for (int s = 0; s < dim; ++s) {
    double e = 0.0;
    // ZZ terms
    for (int i = 0; i < n_sites - 1; ++i) {
      int zi = ((s >> i) & 1) ? 1 : -1;
      int zj = ((s >> (i + 1)) & 1) ? 1 : -1;
      e -= J * zi * zj;
    }
    // longitudinal field
    for (int i = 0; i < n_sites; ++i) {
      int zi = ((s >> i) & 1) ? 1 : -1;
      e -= h_z * zi;
    }
    diag[s] = e;
  }
  return diag;
}

// y = H₀ * x  (diagonal)
static void h0_matvec(const std::vector<double>& diag, const std::vector<cx>& x, std::vector<cx>& y) {
  for (std::size_t i = 0; i < diag.size(); ++i) y[i] = diag[i] * x[i];
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  Kick operator  U_kick = ⊗_i exp(−i h_x X_i)  applied in-place
//
// For each qubit i:  exp(−i θ X) = cos(θ)I − i sin(θ)X
//   |s⟩ → cos(θ)|s⟩ − i sin(θ)|flip_i(s)⟩
// We iterate over all pairs (s, flip_i(s)) with s < flip_i(s).
// ─────────────────────────────────────────────────────────────────────────────

static void apply_kick(std::vector<cx>& psi, int n_sites, double h_x) {
  cx c{std::cos(h_x), 0.0};
  cx ms{0.0, -std::sin(h_x)};
  for (int i = 0; i < n_sites; ++i) {
    int dim = static_cast<int>(psi.size());
    int bit = 1 << i;
    for (int s = 0; s < dim; ++s) {
      if (s & bit) continue;  // process each pair once
      int sf = s | bit;
      cx a = psi[s], b = psi[sf];
      psi[s]  = c * a + ms * b;
      psi[sf] = ms * a + c * b;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  Arnoldi-Krylov: psi ← exp(−i τ H₀) psi
// ─────────────────────────────────────────────────────────────────────────────

// Dense complex matrix exponential (Taylor + scaling-and-squaring)
static MatC dense_expm_cx(MatC A, int m) {
  double norm1 = 0.0;
  for (int c = 0; c < m; ++c) {
    double cs = 0.0;
    for (int r = 0; r < m; ++r)
      cs += std::abs(A[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * m]);
    norm1 = std::max(norm1, cs);
  }
  int s = 1;
  if (norm1 > 1.0) s = static_cast<int>(std::ceil(norm1));
  MatC As(static_cast<std::size_t>(m) * m);
  for (std::size_t i = 0; i < static_cast<std::size_t>(m) * m; ++i)
    As[i] = A[i] / static_cast<double>(s);

  MatC E(static_cast<std::size_t>(m) * m, cx{0.0, 0.0});
  for (int i = 0; i < m; ++i)
    E[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * m] = {1.0, 0.0};

  MatC Ak = E;
  double fact = 1.0;
  for (int k = 1; k <= 14; ++k) {
    fact /= static_cast<double>(k);
    MatC Ak_new(static_cast<std::size_t>(m) * m, cx{0.0, 0.0});
    for (int r = 0; r < m; ++r)
      for (int c = 0; c < m; ++c) {
        cx acc{0.0, 0.0};
        for (int l = 0; l < m; ++l)
          acc += Ak[static_cast<std::size_t>(r) + static_cast<std::size_t>(l) * m] *
                 As[static_cast<std::size_t>(l) + static_cast<std::size_t>(c) * m];
        Ak_new[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * m] = acc;
      }
    Ak = Ak_new;
    for (std::size_t i = 0; i < static_cast<std::size_t>(m) * m; ++i)
      E[i] += fact * Ak[i];
  }

  // Squaring: E = exp(A/s)^s via linear accumulation
  if (s > 1) {
    MatC E_orig = E;
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

// m-step Arnoldi for exp(−i τ diag) |ψ⟩
// Since H₀ is diagonal, exp(−i τ H₀) is exact:  psi[s] *= exp(−i τ diag[s])
static void apply_exp_h0(const std::vector<double>& diag, std::vector<cx>& psi, double tau) {
  for (std::size_t s = 0; s < diag.size(); ++s) {
    double phase = -tau * diag[s];
    psi[s] *= cx{std::cos(phase), std::sin(phase)};
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// D.  Bipartite entanglement entropy
// ─────────────────────────────────────────────────────────────────────────────

// Householder tridiagonalization of complex Hermitian n×n matrix (col-major)
static std::pair<std::vector<double>, std::vector<double>>
householder_herm(std::vector<cx>& A, int n) {
  for (int k = 0; k < n - 2; ++k) {
    double sigma = 0.0;
    for (int i = k + 1; i < n; ++i)
      sigma += std::norm(A[static_cast<std::size_t>(i) + static_cast<std::size_t>(k) * n]);
    double alpha_hh = std::sqrt(sigma);
    cx x1 = A[static_cast<std::size_t>(k + 1) + static_cast<std::size_t>(k) * n];
    cx phase_x1 = (std::abs(x1) < 1e-300) ? cx{1.0, 0.0} : x1 / std::abs(x1);
    double r2 = sigma - std::abs(x1) * alpha_hh;
    if (r2 < 1e-30) continue;
    A[static_cast<std::size_t>(k + 1) + static_cast<std::size_t>(k) * n] -= phase_x1 * alpha_hh;
    std::vector<cx> v(n - k - 1);
    for (int i = 0; i < n - k - 1; ++i)
      v[i] = A[static_cast<std::size_t>(k + 1 + i) + static_cast<std::size_t>(k) * n];
    std::vector<cx> q(n - k - 1, cx{0.0, 0.0});
    for (int r = 0; r < n - k - 1; ++r)
      for (int c = 0; c < n - k - 1; ++c)
        q[r] += A[static_cast<std::size_t>(k + 1 + r) + static_cast<std::size_t>(k + 1 + c) * n] * v[c];
    cx vHq{0.0, 0.0};
    for (int i = 0; i < n - k - 1; ++i) vHq += std::conj(v[i]) * q[i];
    std::vector<cx> w(n - k - 1);
    for (int i = 0; i < n - k - 1; ++i)
      w[i] = q[i] / r2 - (vHq.real() / (2.0 * r2 * r2)) * v[i];
    for (int r = 0; r < n - k - 1; ++r)
      for (int c = 0; c < n - k - 1; ++c)
        A[static_cast<std::size_t>(k + 1 + r) + static_cast<std::size_t>(k + 1 + c) * n] -=
            v[r] * std::conj(w[c]) + w[r] * std::conj(v[c]);
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

// Wilkinson QR for tridiagonal eigensolver (reused from phi4 logic)
static std::vector<double>
tridiag_eigenvalues(std::vector<double> a, std::vector<double> b) {
  int k = static_cast<int>(a.size());
  for (int iter = 0; iter < 30 * k; ++iter) {
    int n = k;
    while (n > 1 && std::abs(b[n - 2]) < 1e-10 * (std::abs(a[n - 2]) + std::abs(a[n - 1])))
      --n;
    if (n == 1) break;
    int m = n - 1;
    while (m > 0 && std::abs(b[m - 1]) >= 1e-10 * (std::abs(a[m - 1]) + std::abs(a[m])))
      --m;
    double d_w = (a[n - 2] - a[n - 1]) * 0.5;
    double sd  = (d_w >= 0.0) ? 1.0 : -1.0;
    double mu  = a[n - 1] - b[n - 2] * b[n - 2] /
                 (d_w + sd * std::sqrt(d_w * d_w + b[n - 2] * b[n - 2]));
    double x = a[m] - mu, z = b[m];
    for (int i = m; i < n - 1; ++i) {
      double rg = std::sqrt(x * x + z * z);
      if (rg < 1e-300) break;
      double c = x / rg, s = z / rg;
      if (i > m) b[i - 1] = rg;
      double ai = a[i], ai1 = a[i + 1], bi = b[i];
      a[i]   = c * c * ai + 2.0 * c * s * bi + s * s * ai1;
      a[i+1] = s * s * ai - 2.0 * c * s * bi + c * c * ai1;
      b[i]   = c * s * (ai1 - ai) + (c * c - s * s) * bi;
      if (i < n - 2) { double bi1 = b[i + 1]; b[i + 1] = c * bi1; z = s * bi1; x = b[i]; }
    }
  }
  return a;
}

static double floquet_entropy(const std::vector<cx>& psi, int n_sites) {
  int dimA = 1 << (n_sites / 2);
  int dimB = 1 << (n_sites - n_sites / 2);
  // Build M†M (dimA × dimA complex Hermitian)
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
  auto [dg, sb] = householder_herm(MtM, dimA);
  auto evals = tridiag_eigenvalues(dg, sb);
  double trace = 0.0;
  for (auto e : evals) if (e > 0.0) trace += e;
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
// E.  Floquet unitary diagonalization (full dense, for quasi-energies)
// ─────────────────────────────────────────────────────────────────────────────
// Build U(T) column by column by applying U to each basis state, then
// diagonalize with Schur/QR iteration for unitary matrices.
// For small dim (≤ 4096), use Hessenberg QR on U.
//
// Quasi-energies: θ_j = arg(λ_j) / T, where λ_j are eigenvalues of U(T).
// We compute arg only (no eigenvectors needed for gap ratio).

// Apply one Floquet period to psi in-place
static void apply_floquet_period(const std::vector<double>& h0_diag,
                                  std::vector<cx>& psi,
                                  int n_sites,
                                  double h_x,
                                  double half_T) {
  apply_exp_h0(h0_diag, psi, half_T);
  apply_kick(psi, n_sites, h_x);
  apply_exp_h0(h0_diag, psi, half_T);
}

// Build dense Floquet matrix U(T) (dim×dim, column-major)
static MatC build_floquet_matrix(const std::vector<double>& h0_diag,
                                  int n_sites, double h_x, double half_T) {
  int dim = 1 << n_sites;
  MatC U(static_cast<std::size_t>(dim) * dim, cx{0.0, 0.0});
  for (int col = 0; col < dim; ++col) {
    std::vector<cx> psi(dim, cx{0.0, 0.0});
    psi[col] = {1.0, 0.0};
    apply_floquet_period(h0_diag, psi, n_sites, h_x, half_T);
    for (int r = 0; r < dim; ++r)
      U[static_cast<std::size_t>(r) + static_cast<std::size_t>(col) * dim] = psi[r];
  }
  return U;
}

// QR iteration to find eigenvalues of an upper Hessenberg unitary matrix.
// We use the Francis double-step QR shift adapted for unitary matrices.
// Here we use a simple power-method-free approach: explicitly Hessenberg-reduce
// then iterate.  For dim ≤ 4096 this is acceptable.
//
// In practice: reduce U → upper Hessenberg, then run QR iterations until
// quasi-diagonal.  Return diagonal elements (eigenvalues).
//
// Implementation: complex Householder to reduce to upper Hessenberg,
// then Francis QR on the Hessenberg form.  We only need eigenvalues (phases).

static std::vector<cx>
hessenberg_eigenvalues(MatC& U, int n) {
  // Reduce to upper Hessenberg via unitary similarity
  for (int k = 0; k < n - 2; ++k) {
    // Build Householder for column k, rows k+1..n-1
    double sigma = 0.0;
    for (int i = k + 1; i < n; ++i)
      sigma += std::norm(U[static_cast<std::size_t>(i) + static_cast<std::size_t>(k) * n]);
    if (sigma < 1e-28) continue;
    double alpha_hh = std::sqrt(sigma);
    cx x1 = U[static_cast<std::size_t>(k + 1) + static_cast<std::size_t>(k) * n];
    cx ph = (std::abs(x1) < 1e-300) ? cx{1.0, 0.0} : x1 / std::abs(x1);
    cx tau_hh = ph * alpha_hh;
    U[static_cast<std::size_t>(k + 1) + static_cast<std::size_t>(k) * n] += tau_hh;
    double r2 = std::abs(U[static_cast<std::size_t>(k + 1) + static_cast<std::size_t>(k) * n]);
    if (r2 < 1e-300) { U[static_cast<std::size_t>(k + 1) + static_cast<std::size_t>(k) * n] -= tau_hh; continue; }
    r2 = r2 * r2;
    // v = column k, rows k+1..n-1
    std::vector<cx> v(n - k - 1);
    for (int i = 0; i < n - k - 1; ++i)
      v[i] = U[static_cast<std::size_t>(k + 1 + i) + static_cast<std::size_t>(k) * n];
    // Apply from left: U[k+1:, :] -= (2/r2) v (v† U[k+1:, :])
    for (int c = 0; c < n; ++c) {
      cx dot{0.0, 0.0};
      for (int i = 0; i < n - k - 1; ++i)
        dot += std::conj(v[i]) * U[static_cast<std::size_t>(k + 1 + i) + static_cast<std::size_t>(c) * n];
      for (int i = 0; i < n - k - 1; ++i)
        U[static_cast<std::size_t>(k + 1 + i) + static_cast<std::size_t>(c) * n] -= (2.0 / r2) * v[i] * dot;
    }
    // Apply from right: U[:, k+1:] -= (2/r2) (U[:, k+1:] v) v†
    for (int r = 0; r < n; ++r) {
      cx dot{0.0, 0.0};
      for (int i = 0; i < n - k - 1; ++i)
        dot += U[static_cast<std::size_t>(r) + static_cast<std::size_t>(k + 1 + i) * n] * v[i];
      for (int i = 0; i < n - k - 1; ++i)
        U[static_cast<std::size_t>(r) + static_cast<std::size_t>(k + 1 + i) * n] -= (2.0 / r2) * dot * std::conj(v[i]);
    }
  }

  // QR iteration on upper Hessenberg U (Francis single-step with Wilkinson-like shift)
  for (int iter = 0; iter < 30 * n; ++iter) {
    int nn = n;
    while (nn > 1 && std::abs(U[static_cast<std::size_t>(nn - 1) + static_cast<std::size_t>(nn - 2) * n]) <
           1e-10 * (std::abs(U[static_cast<std::size_t>(nn - 1) + static_cast<std::size_t>(nn - 1) * n]) +
                    std::abs(U[static_cast<std::size_t>(nn - 2) + static_cast<std::size_t>(nn - 2) * n])))
      --nn;
    if (nn == 1) break;
    // Shift = U[nn-1, nn-1]
    cx mu = U[static_cast<std::size_t>(nn - 1) + static_cast<std::size_t>(nn - 1) * n];
    // Apply shift, do one QR step via Givens rotations
    int mm = nn - 1;
    while (mm > 0 && std::abs(U[static_cast<std::size_t>(mm) + static_cast<std::size_t>(mm - 1) * n]) >=
           1e-10 * (std::abs(U[static_cast<std::size_t>(mm) + static_cast<std::size_t>(mm) * n]) +
                    std::abs(U[static_cast<std::size_t>(mm - 1) + static_cast<std::size_t>(mm - 1) * n])))
      --mm;
    cx x = U[static_cast<std::size_t>(mm) + static_cast<std::size_t>(mm) * n] - mu;
    cx z = U[static_cast<std::size_t>(mm + 1) + static_cast<std::size_t>(mm) * n];
    for (int i = mm; i < nn - 1; ++i) {
      double rg = std::sqrt(std::norm(x) + std::norm(z));
      if (rg < 1e-300) break;
      cx c_g = std::conj(x) / rg, s_g = std::conj(z) / rg;
      // Apply Givens from left: rows i, i+1
      for (int c = 0; c < n; ++c) {
        cx ui  = U[static_cast<std::size_t>(i)     + static_cast<std::size_t>(c) * n];
        cx ui1 = U[static_cast<std::size_t>(i + 1) + static_cast<std::size_t>(c) * n];
        U[static_cast<std::size_t>(i)     + static_cast<std::size_t>(c) * n] =  std::conj(c_g) * ui + std::conj(s_g) * ui1;
        U[static_cast<std::size_t>(i + 1) + static_cast<std::size_t>(c) * n] = -s_g * ui + c_g * ui1;
      }
      // Apply Givens from right: cols i, i+1
      for (int r = 0; r < n; ++r) {
        cx ui  = U[static_cast<std::size_t>(r) + static_cast<std::size_t>(i)     * n];
        cx ui1 = U[static_cast<std::size_t>(r) + static_cast<std::size_t>(i + 1) * n];
        U[static_cast<std::size_t>(r) + static_cast<std::size_t>(i)     * n] = c_g * ui + s_g * ui1;
        U[static_cast<std::size_t>(r) + static_cast<std::size_t>(i + 1) * n] = -std::conj(s_g) * ui + std::conj(c_g) * ui1;
      }
      if (i < nn - 2) {
        x = U[static_cast<std::size_t>(i + 1) + static_cast<std::size_t>(i) * n];
        z = U[static_cast<std::size_t>(i + 2) + static_cast<std::size_t>(i) * n];
      }
    }
  }

  // Extract diagonal eigenvalues
  std::vector<cx> evals(n);
  for (int i = 0; i < n; ++i)
    evals[i] = U[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * n];
  return evals;
}

// Mean gap ratio r = mean(min(δ_n, δ_{n+1}) / max(δ_n, δ_{n+1}))
// where δ_n = θ_{n+1} − θ_n (sorted quasi-energies)
static double gap_ratio(const std::vector<double>& qe) {
  int n = static_cast<int>(qe.size());
  if (n < 3) return 0.0;
  std::vector<double> sorted = qe;
  std::sort(sorted.begin(), sorted.end());
  double r = 0.0;
  int cnt = 0;
  for (int i = 0; i < n - 2; ++i) {
    double d1 = sorted[i + 1] - sorted[i];
    double d2 = sorted[i + 2] - sorted[i + 1];
    if (d1 < 1e-14 || d2 < 1e-14) continue;
    r += std::min(d1, d2) / std::max(d1, d2);
    ++cnt;
  }
  return (cnt > 0) ? r / cnt : 0.0;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

FloquetResult run_floquet(const FloquetOptions& opts) {
  auto t0 = std::chrono::steady_clock::now();
  FloquetResult res;

  try {
    int N = static_cast<int>(opts.n_sites);
    if (N < 2 || N > 12)
      throw std::runtime_error("n_sites must be in [2, 12]");
    if (opts.drive_omega <= 0.0)
      throw std::runtime_error("drive_omega must be positive");

    int dim = 1 << N;
    double T     = 2.0 * PI / opts.drive_omega;
    double halfT = T * 0.5;

    auto h0_diag = build_h0_diag(N, opts.J, opts.h_z);

    res.n_sites     = opts.n_sites;
    res.J           = opts.J;
    res.h_z         = opts.h_z;
    res.drive_amp   = opts.drive_amp;
    res.drive_omega = opts.drive_omega;

    // ── Dynamics scenario ───────────────────────────────────────────────────
    if (opts.scenario == "dynamics" || opts.scenario == "heating") {
      // Initial state: Néel state |↑↓↑↓…⟩ = alternating 1 and 0 bits
      int neel = 0;
      for (int i = 0; i < N; i += 2) neel |= (1 << i);
      std::vector<cx> psi(dim, cx{0.0, 0.0});
      psi[neel] = {1.0, 0.0};

      auto record = [&](std::uint32_t period, double t) {
        FloquetFrame f;
        f.period = period;
        f.time   = t;
        // ⟨Z_i⟩/N averaged = magnetization
        double mz = 0.0;
        for (int s = 0; s < dim; ++s) {
          double p = std::norm(psi[s]);
          double sz = 0.0;
          for (int i = 0; i < N; ++i)
            sz += ((s >> i) & 1) ? 1.0 : -1.0;
          mz += p * sz / N;
        }
        f.magnetization_z = mz;
        // ⟨H₀⟩
        double en = 0.0;
        for (int s = 0; s < dim; ++s)
          en += std::norm(psi[s]) * h0_diag[s];
        f.energy  = en;
        f.entropy = floquet_entropy(psi, N);
        res.frames.push_back(f);
      };

      record(0, 0.0);
      for (std::uint32_t p = 1; p <= opts.n_periods; ++p) {
        apply_floquet_period(h0_diag, psi, N, opts.drive_amp, halfT);
        record(p, static_cast<double>(p) * T);
      }
    }

    // ── Spectrum scenario ───────────────────────────────────────────────────
    if (opts.scenario == "spectrum" || opts.scenario == "dynamics") {
      if (dim <= 4096) {
        // Build U(T) and diagonalize for quasi-energies
        MatC U = build_floquet_matrix(h0_diag, N, opts.drive_amp, halfT);
        auto eigs = hessenberg_eigenvalues(U, dim);
        res.quasi_energies.resize(dim);
        for (int i = 0; i < dim; ++i)
          res.quasi_energies[i] = std::arg(eigs[i]) / T;
        std::sort(res.quasi_energies.begin(), res.quasi_energies.end());
        res.mean_gap_ratio = gap_ratio(res.quasi_energies);
      }
    }

    // ── Scan scenario ───────────────────────────────────────────────────────
    if (opts.scenario == "spectrum") {
      int ns = static_cast<int>(opts.n_scan);
      for (int si = 0; si < ns; ++si) {
        double amp = opts.scan_amp_max * si / std::max(ns - 1, 1);
        res.scan_amps.push_back(amp);
        if (dim <= 1024) {
          MatC U = build_floquet_matrix(h0_diag, N, amp, halfT);
          auto eigs = hessenberg_eigenvalues(U, dim);
          std::vector<double> qe(dim);
          for (int i = 0; i < dim; ++i)
            qe[i] = std::arg(eigs[i]) / T;
          std::sort(qe.begin(), qe.end());
          res.scan_qe.push_back(std::move(qe));
        } else {
          res.scan_qe.push_back({});
        }
      }
    }

    res.ok = true;
  } catch (const std::exception& ex) {
    res.ok      = false;
    res.message = ex.what();
  }

  auto t1 = std::chrono::steady_clock::now();
  res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return res;
}

}  // namespace qx
