// src/phi4.cpp — M24: φ⁴ Scalar Quantum Field Theory
//
// Simulates the φ⁴ lattice QFT Hamiltonian:
//
//   H = Σᵢ [½πᵢ² + ½m²φᵢ² + (λ/4)φᵢ⁴] − κ Σᵢ φᵢφᵢ₊₁
//
// using a truncated Fock (harmonic oscillator) basis per site with d_local levels.
//
// Local operators in terms of creation/annihilation operators (â, â†):
//   φᵢ = (âᵢ + â†ᵢ)/√2        (field)
//   πᵢ = i(â†ᵢ − âᵢ)/√2       (conjugate momentum)
//
// So:
//   πᵢ² = (2N̂ᵢ + 1 − â†²ᵢ − â²ᵢ)/2
//   φᵢ² = (â²ᵢ + â†²ᵢ + 2N̂ᵢ + 1)/2
//   φᵢ⁴ = (âᵢ + â†ᵢ)⁴/4
//
// The full Hamiltonian is built as a sparse tensor product and the ground state
// is found via a 500-step shifted power iteration (spectral shift method).
// For n_sites ≤ 5 and d_local ≤ 8 (dim ≤ 32768) this runs in a few seconds.
//
// The C++ engine handles "symmetric" and "broken" scenarios (ground state +
// first excited state + sweep). Dynamics are delegated to the Python solver.

#include "qx/engine.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace qx {

namespace {

using cx   = std::complex<double>;
using Mat  = std::vector<double>;   // real, column-major, dim×dim
using MatC = std::vector<cx>;

// ── Local (single-site) operator matrices in truncated Fock space ───────────

// â (lowering): â|n⟩ = √n |n−1⟩
// Matrix element: a[n-1, n] = √n  (col n, row n-1)
std::vector<double> make_a(int d) {
  std::vector<double> a(static_cast<std::size_t>(d) * d, 0.0);
  for (int n = 1; n < d; ++n) {
    a[static_cast<std::size_t>(n - 1) + static_cast<std::size_t>(n) * d] =
        std::sqrt(static_cast<double>(n));
  }
  return a;
}

// φ = (a + a†)/√2, symmetric tridiagonal matrix.
std::vector<double> make_phi(int d) {
  auto a = make_a(d);
  std::vector<double> phi(static_cast<std::size_t>(d) * d, 0.0);
  for (int r = 0; r < d; ++r)
    for (int c = 0; c < d; ++c) {
      std::size_t idx = static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * d;
      phi[idx] = (a[idx] + a[static_cast<std::size_t>(c) +
                                static_cast<std::size_t>(r) * d]) /
                 std::sqrt(2.0);
    }
  return phi;
}

// π² = (2N + 1 − a†² − a²)/2 — returns diagonal + off-diagonal terms.
// Equivalently: π² = −(a − a†)²/2 which equals I − adag·a†† (numerically same).
// We compute it directly as -(a - adag)^2 / 2.
std::vector<double> make_pi2(int d) {
  auto a = make_a(d);
  // (a - a†): antisymmetric.
  std::vector<double> diff(static_cast<std::size_t>(d) * d, 0.0);
  for (int r = 0; r < d; ++r)
    for (int c = 0; c < d; ++c) {
      std::size_t idx = static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * d;
      diff[idx] = a[idx] - a[static_cast<std::size_t>(c) + static_cast<std::size_t>(r) * d];
    }
  // -(diff)^2 / 2 = -diff·diff / 2.
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

// Matrix multiply A·B (d×d, col-major).
std::vector<double> matmul(const std::vector<double>& A, const std::vector<double>& B, int d) {
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

// Matrix add in-place: A += s*B.
void madd(std::vector<double>& A, const std::vector<double>& B, double s, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) A[i] += s * B[i];
}

// ── Full Hamiltonian as dense real matrix (column-major, dim×dim) ─────────

Mat build_phi4_hamiltonian(int N, int d, double m2, double lam, double kappa, double eps) {
  const int dim = 1;
  int dim_full = 1;
  for (int i = 0; i < N; ++i) dim_full *= d;

  const std::size_t sz = static_cast<std::size_t>(dim_full) * dim_full;
  Mat H(sz, 0.0);

  // Local operators.
  auto phi_loc = make_phi(d);
  auto pi2_loc = make_pi2(d);
  auto phi2_loc = matmul(phi_loc, phi_loc, d);
  auto phi4_loc = matmul(phi2_loc, phi2_loc, d);

  // Helper: apply single-site operator at site s to H.
  // H += coeff * (I ⊗ ... ⊗ op_loc ⊗ ... ⊗ I)
  // Using tensor product structure: for basis index i = i_{N-1}...i_0 (i_s is the s-th digit in base d),
  // the operator only mixes the s-th index.
  auto add_local = [&](const std::vector<double>& op, int site, double coeff) {
    // stride_before = d^site, stride_after = d^(N-site-1)
    int stride = 1;
    for (int i = 0; i < site; ++i) stride *= d;
    int block = stride * d;
    // Iterate over all basis states.
    for (int outer = 0; outer < dim_full / block; ++outer) {
      for (int inner = 0; inner < stride; ++inner) {
        // Row and column in the local space.
        for (int r = 0; r < d; ++r) {
          for (int c = 0; c < d; ++c) {
            double v = op[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * d];
            if (v == 0.0) continue;
            int row_full = outer * block + r * stride + inner;
            int col_full = outer * block + c * stride + inner;
            H[static_cast<std::size_t>(row_full) +
              static_cast<std::size_t>(col_full) * dim_full] += coeff * v;
          }
        }
      }
    }
  };

  // Helper: two-site operator at sites s1, s2.
  // Precompute per-site strides (stride[i] = d^i).
  auto add_twosite = [&](const std::vector<double>& op1, int s1,
                         const std::vector<double>& op2, int s2, double coeff) {
    // Build strides: stride[i] = d^i.
    std::vector<int> strides(N, 1);
    for (int i = 1; i < N; ++i) strides[i] = strides[i - 1] * d;

    // Iterate only over the N-2 "other" sites (not s1 or s2) to avoid the d²
    // overcount that occurs when iterating all dim_full states and overwriting
    // the digits at s1 and s2.
    int dim_other = dim_full / (d * d);

    for (int r1 = 0; r1 < d; ++r1)
      for (int c1 = 0; c1 < d; ++c1) {
        double v1 = op1[static_cast<std::size_t>(r1) + static_cast<std::size_t>(c1) * d];
        if (v1 == 0.0) continue;
        for (int r2 = 0; r2 < d; ++r2)
          for (int c2 = 0; c2 < d; ++c2) {
            double v2 = op2[static_cast<std::size_t>(r2) + static_cast<std::size_t>(c2) * d];
            if (v2 == 0.0) continue;
            // Contribution from the two target sites.
            int base_row = r1 * strides[s1] + r2 * strides[s2];
            int base_col = c1 * strides[s1] + c2 * strides[s2];
            // Enumerate all combinations of the remaining N-2 site indices.
            for (int other = 0; other < dim_other; ++other) {
              int row_full = base_row, col_full = base_col;
              int tmp = other;
              for (int i = 0; i < N; ++i) {
                if (i == s1 || i == s2) continue;
                int di = tmp % d; tmp /= d;
                row_full += di * strides[i];
                col_full += di * strides[i];
              }
              H[static_cast<std::size_t>(row_full) +
                static_cast<std::size_t>(col_full) * dim_full] += coeff * v1 * v2;
            }
          }
      }
  };

  // ── Local terms ──────────────────────────────────────────────────────────
  for (int i = 0; i < N; ++i) {
    add_local(pi2_loc,  i, 0.5);          // ½πᵢ²
    add_local(phi2_loc, i, 0.5 * m2);     // ½m²φᵢ²
    add_local(phi4_loc, i, 0.25 * lam);   // λ/4 · φᵢ⁴
    add_local(phi_loc,  i, eps);           // ε·φᵢ (explicit symmetry breaking)
  }

  // ── Hopping terms: −κ φᵢφᵢ₊₁ ────────────────────────────────────────────
  for (int i = 0; i < N - 1; ++i) {
    add_twosite(phi_loc, i, phi_loc, i + 1, -kappa);
  }

  return H;
}

// ── Ground state via shifted power iteration ─────────────────────────────

struct EigPair {
  double eval;
  std::vector<double> evec;
};

EigPair power_iter(const Mat& H, int dim, std::uint64_t seed, int n_iter = 500) {
  // Spectral shift: estimate max row 1-norm for shift.
  double shift = 0.0;
  for (int r = 0; r < dim; ++r) {
    double rs = 0.0;
    for (int c = 0; c < dim; ++c)
      rs += std::abs(H[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * dim]);
    shift = std::max(shift, rs);
  }
  shift *= 1.01;

  // Random initial vector.
  std::vector<double> v(dim);
  for (int i = 0; i < dim; ++i) {
    v[i] = std::cos(static_cast<double>((seed * 6364136223846793005ULL + i) % 1000) * 0.00628);
  }
  double norm = 0.0;
  for (auto x : v) norm += x * x;
  norm = std::sqrt(norm);
  for (auto& x : v) x /= norm;

  std::vector<double> w(dim);
  for (int iter = 0; iter < n_iter; ++iter) {
    // w = (shift·I - H)·v
    for (int r = 0; r < dim; ++r) {
      double acc = shift * v[r];
      for (int c = 0; c < dim; ++c)
        acc -= H[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * dim] * v[c];
      w[r] = acc;
    }
    double n2 = 0.0;
    for (auto x : w) n2 += x * x;
    n2 = std::sqrt(n2);
    if (n2 < 1e-14) break;
    for (int i = 0; i < dim; ++i) v[i] = w[i] / n2;
  }

  // Rayleigh quotient.
  double e = 0.0;
  for (int r = 0; r < dim; ++r) {
    double acc = 0.0;
    for (int c = 0; c < dim; ++c)
      acc += H[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * dim] * v[c];
    e += v[r] * acc;
  }
  return {e, std::move(v)};
}

// Deflate H to find second eigenvalue: H' = H - E0·v0·v0ᵀ + 2*|E0|·v0·v0ᵀ
// = H + (2|E0| - E0)·v0v0ᵀ — shifts ground state eigenvalue above the rest.
EigPair second_eigenvalue(const Mat& H, int dim, const std::vector<double>& v0,
                           double E0, std::uint64_t seed) {
  // Deflated H: H_def = H - E0 * |v0><v0| + shift_gs * |v0><v0|
  // Choose shift_gs >> spectrum to push ground state out.
  double shift_gs = std::abs(E0) * 3.0 + 10.0;
  Mat Hd = H;
  for (int r = 0; r < dim; ++r)
    for (int c = 0; c < dim; ++c)
      Hd[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * dim] +=
          shift_gs * v0[r] * v0[c];
  return power_iter(Hd, dim, seed + 1, 500);
}

// ── Observable computation ─────────────────────────────────────────────────

// Compute ⟨op⟩ = ⟨ψ|O_site ⊗ I...|ψ⟩ for a single-site local operator.
double expect_local(const std::vector<double>& psi, const std::vector<double>& op,
                    int site, int N, int d) {
  int dim_full = static_cast<int>(psi.size());
  double result = 0.0;
  int stride = 1;
  for (int i = 0; i < site; ++i) stride *= d;
  int block = stride * d;
  for (int outer = 0; outer < dim_full / block; ++outer) {
    for (int inner = 0; inner < stride; ++inner) {
      for (int r = 0; r < d; ++r) {
        for (int c = 0; c < d; ++c) {
          double v = op[static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * d];
          if (v == 0.0) continue;
          int row_full = outer * block + r * stride + inner;
          int col_full = outer * block + c * stride + inner;
          result += psi[row_full] * v * psi[col_full];
        }
      }
    }
  }
  return result;
}

}  // namespace

// ── Public API ─────────────────────────────────────────────────────────────

Phi4Result run_phi4(const Phi4Options& opts) {
  auto t0 = std::chrono::steady_clock::now();

  const int N = static_cast<int>(opts.n_sites);
  const int d = static_cast<int>(opts.d_local);
  if (N < 2 || N > 5) {
    return Phi4Result{false, 0, 0, 0, 0, 0, 0, 0, {}, {}, {}, {}, {},
                      0.0, "n_sites must be in [2, 5]"};
  }
  if (d < 4 || d > 12) {
    return Phi4Result{false, 0, 0, 0, 0, 0, 0, 0, {}, {}, {}, {}, {},
                      0.0, "d_local must be in [4, 12]"};
  }

  int dim_full = 1;
  for (int i = 0; i < N; ++i) dim_full *= d;
  if (dim_full > 60000) {
    return Phi4Result{false, 0, 0, 0, 0, 0, 0, 0, {}, {}, {}, {}, {},
                      0.0, "system too large (n_sites^d_local > 60000); reduce n_sites or d_local"};
  }

  Phi4Result result;
  result.n_sites = static_cast<std::uint32_t>(N);
  result.d_local = static_cast<std::uint32_t>(d);
  result.m2 = opts.m2;
  result.lambda_coupling = opts.lambda_coupling;

  // Local operators for observables.
  auto phi_loc  = make_phi(d);
  auto phi2_loc = matmul(phi_loc, phi_loc, d);
  // N̂ = a†a = (φ² + π² - 1)/2 locally; easier: N̂[n,n] = n.
  std::vector<double> n_loc(static_cast<std::size_t>(d) * d, 0.0);
  for (int n = 0; n < d; ++n)
    n_loc[static_cast<std::size_t>(n) + static_cast<std::size_t>(n) * d] =
        static_cast<double>(n);

  auto run_scenario = [&](double m2_val) -> std::tuple<double, double, double,
                                                         std::vector<double>, std::vector<double>,
                                                         std::vector<double>> {
    Mat H = build_phi4_hamiltonian(N, d, m2_val, opts.lambda_coupling, opts.kappa, opts.epsilon);
    auto gs = power_iter(H, dim_full, opts.seed);
    auto es = second_eigenvalue(H, dim_full, gs.evec, gs.eval, opts.seed);

    double gap = es.eval - gs.eval;

    std::vector<double> phi_exp(N), phi2_exp(N), n_exp(N);
    for (int i = 0; i < N; ++i) {
      phi_exp[i]  = expect_local(gs.evec, phi_loc,  i, N, d);
      phi2_exp[i] = expect_local(gs.evec, phi2_loc, i, N, d);
      n_exp[i]    = expect_local(gs.evec, n_loc,    i, N, d);
    }

    // Order parameter: √(mean ⟨φᵢ²⟩).
    double op = 0.0;
    for (auto x : phi2_exp) op += x;
    op = std::sqrt(op / N);

    return {gs.eval, gap, op, phi_exp, phi2_exp, n_exp};
  };

  // ── Main scenario ──────────────────────────────────────────────────────
  auto [E0, gap, op, phi_exp, phi2_exp, n_exp] = run_scenario(opts.m2);
  result.ground_energy  = E0;
  result.energy_gap     = gap;
  result.order_parameter = op;
  result.phi_expectation  = phi_exp;
  result.phi2_expectation = phi2_exp;
  result.occupation       = n_exp;

  // ── m² sweep ──────────────────────────────────────────────────────────
  if (opts.scenario == "sweep" && opts.n_sweep > 0) {
    for (std::uint32_t i = 0; i < opts.n_sweep; ++i) {
      double m2_val = opts.m2_min + (opts.m2_max - opts.m2_min) *
                      static_cast<double>(i) / std::max(1u, opts.n_sweep - 1);
      auto [Ei, gi, opi, pe, p2e, ne] = run_scenario(m2_val);
      result.sweep.push_back({m2_val, Ei, opi, gi});
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return result;
}

}  // namespace qx
