// src/qfield.cpp — M23: Schwinger model (1+1D QED) lattice simulation
//
// Implements the Kogut-Susskind staggered Hamiltonian for the Schwinger model
// after integrating out the gauge field via Gauss's law (open BCs):
//
//   H = -½ Σ_n (σ+_n σ-_{n+1} + h.c.)          [hopping / kinetic]
//     + (m/2) Σ_n (-1)^n σz_n                   [staggered mass]
//     + (g²/2) Σ_{l=0}^{N-2} L_l²               [electric field energy]
//
// where L_l = α + Σ_{k=0}^{l} (σz_k + (-1)^k)/2  (Gauss's law resolved).
//
// The long-range part from expanding L_l² produces ZZ interactions:
//   H_E = (g²/2) Σ_{l} [c_l I + Σ_k a_{lk} σz_k + Σ_{k≤k'} b_{lkk'} σz_k σz_{k'}]²
//
// For small N (≤ 12 sites, dim ≤ 4096) we build the full Hamiltonian matrix
// and call a minimal power-iteration ground-state finder.
//
// The C++ engine handles: vacuum scenario (ground state + mass sweep).
// Dynamics (time evolution) is delegated to the Python reference solver
// which uses numpy's full matrix exponentiation.

#include "qx/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace qx {

namespace {

using cx = std::complex<double>;
using Mat = std::vector<cx>;  // column-major dense matrix, dim×dim

// ── Sparse Pauli helpers ───────────────────────────────────────────────────

// Single-site σz: diagonal element for basis state |s⟩ (s is an N-bit integer).
// Qubit n (0=leftmost) contributes +1 if bit n of s is 0, -1 if bit n is 1
// (|0⟩ = spin up = +1 in σz convention).
inline double sz_element(std::uint32_t s, int n) {
  return ((s >> (n)) & 1u) ? -1.0 : +1.0;
}

// Build the full N-qubit Hamiltonian as a dense complex matrix.
// dim = 2^N, column-major: H[row + col*dim].
Mat build_hamiltonian(int N, double mass, double coupling, double background) {
  const int dim = 1 << N;
  Mat H(static_cast<std::size_t>(dim) * dim, cx{0.0, 0.0});

  auto idx = [&](int r, int c) -> std::size_t {
    return static_cast<std::size_t>(r) + static_cast<std::size_t>(c) * dim;
  };

  // ── Hopping: -½ Σ_n (σ+_n σ-_{n+1} + σ-_n σ+_{n+1}) ──────────────────
  // σ+_n σ-_{n+1}|s⟩: flip bit n from 1→0 and bit n+1 from 0→1.
  // The operator gives non-zero iff bit n=1 and bit n+1=0.
  for (int n = 0; n < N - 1; ++n) {
    for (int s = 0; s < dim; ++s) {
      bool bit_n  = (s >> n)       & 1;
      bool bit_n1 = (s >> (n + 1)) & 1;
      if (bit_n && !bit_n1) {
        // σ+_n σ-_{n+1}: clear bit n, set bit n+1 → target t
        int t = (s & ~(1 << n)) | (1 << (n + 1));
        H[idx(t, s)] -= 0.5;  // row t, col s
      }
      if (!bit_n && bit_n1) {
        // σ-_n σ+_{n+1}: set bit n, clear bit n+1
        int t = (s | (1 << n)) & ~(1 << (n + 1));
        H[idx(t, s)] -= 0.5;
      }
    }
  }

  // ── Mass: (m/2) Σ_n (-1)^n σz_n  ── diagonal ──────────────────────────
  for (int s = 0; s < dim; ++s) {
    double diag = 0.0;
    for (int n = 0; n < N; ++n) {
      double sign = (n % 2 == 0) ? +1.0 : -1.0;
      diag += (mass * 0.5) * sign * sz_element(s, n);
    }
    H[idx(s, s)] += cx{diag, 0.0};
  }

  // ── Electric field: (g²/2) Σ_{l=0}^{N-2} L_l²  ──────────────────────
  // L_l = background + Σ_{k=0}^{l} Q_k  where Q_k = (σz_k + (-1)^k) / 2.
  // For a basis state |s⟩: L_l(s) = background + Σ_{k=0}^{l} (sz(s,k) + (-1)^k)/2.
  // L_l² is diagonal → add to diagonal.
  for (int l = 0; l < N - 1; ++l) {
    for (int s = 0; s < dim; ++s) {
      double L = background;
      for (int k = 0; k <= l; ++k) {
        double sgn = (k % 2 == 0) ? +1.0 : -1.0;
        L += 0.5 * (sz_element(s, k) + sgn);
      }
      H[idx(s, s)] += cx{(coupling * 0.5) * L * L, 0.0};
    }
  }

  return H;
}

// ── Minimal power-iteration for ground state (Lanczos not needed for dim≤4096) ──
// Returns (eigenvalue, eigenvector) via 200-step inverse-free shifted power method.
// For production correctness we use the simplest approach: compute H*v - shift*v
// for the lowest eigenvalue via repeated application.
//
// Actually: for dim ≤ 4096, directly finding the ground state via explicit
// matrix-vector iterations converges reliably.  We use the shift-invert trick:
// lowest eigenvector of (H - λ_min I) → we estimate λ_min first via Gershgorin,
// then shift and iterate on the negative-definite version.
//
// Simpler approach for correctness: just do 300 iterations of the power method
// on (shift*I - H) where shift > max(|eigenvalue|).  The ground state is the
// eigenvector of the largest eigenvalue of (shift*I - H).

struct GroundState {
  double energy;
  std::vector<cx> psi;
};

GroundState find_ground_state(const Mat& H, int dim, std::uint64_t seed) {
  // Estimate spectral radius via row 1-norms for shift.
  double shift = 0.0;
  for (int r = 0; r < dim; ++r) {
    double row_sum = 0.0;
    for (int c = 0; c < dim; ++c) {
      row_sum += std::abs(H[static_cast<std::size_t>(r) +
                            static_cast<std::size_t>(c) * dim]);
    }
    shift = std::max(shift, row_sum);
  }
  shift *= 1.01;

  // Initialize random vector.
  std::vector<cx> v(dim);
  // Simple deterministic init: equal superposition with phase from seed.
  for (int i = 0; i < dim; ++i) {
    double angle = static_cast<double>((seed * 6364136223846793005ULL + i) % 1000) * 0.006283;
    v[i] = cx{std::cos(angle), std::sin(angle)};
  }
  // Normalise.
  double norm = 0.0;
  for (auto& x : v) norm += std::norm(x);
  norm = std::sqrt(norm);
  for (auto& x : v) x /= norm;

  // Power iteration on (shift*I - H): 400 iterations.
  std::vector<cx> w(dim);
  for (int iter = 0; iter < 400; ++iter) {
    // w = (shift*I - H) * v
    for (int r = 0; r < dim; ++r) {
      cx acc{0.0, 0.0};
      for (int c = 0; c < dim; ++c) {
        acc += H[static_cast<std::size_t>(r) +
                  static_cast<std::size_t>(c) * dim] * v[c];
      }
      w[r] = shift * v[r] - acc;
    }
    // Normalise.
    double n2 = 0.0;
    for (auto& x : w) n2 += std::norm(x);
    n2 = std::sqrt(n2);
    if (n2 < 1e-14) break;
    for (int i = 0; i < dim; ++i) v[i] = w[i] / n2;
  }

  // Rayleigh quotient: E = ⟨v|H|v⟩.
  double energy = 0.0;
  for (int r = 0; r < dim; ++r) {
    cx acc{0.0, 0.0};
    for (int c = 0; c < dim; ++c) {
      acc += H[static_cast<std::size_t>(r) +
                static_cast<std::size_t>(c) * dim] * v[c];
    }
    energy += std::real(std::conj(v[r]) * acc);
  }

  return {energy, std::move(v)};
}

// ── Compute observables from a state vector ────────────────────────────────

SchwingerFrame compute_observables(const std::vector<cx>& psi, int N,
                                   double coupling, double background,
                                   double time) {
  const int dim = 1 << N;
  SchwingerFrame f;
  f.time = time;
  f.charge_density.resize(N, 0.0);
  f.electric_field.resize(N - 1, 0.0);
  double chiral = 0.0;

  // σz expectation per site.
  std::vector<double> sz_exp(N, 0.0);
  for (int s = 0; s < dim; ++s) {
    double prob = std::norm(psi[s]);
    for (int n = 0; n < N; ++n) {
      sz_exp[n] += prob * sz_element(s, n);
    }
  }

  // Charge density Q_n = (σz_n + (-1)^n) / 2.
  for (int n = 0; n < N; ++n) {
    double sgn = (n % 2 == 0) ? +1.0 : -1.0;
    f.charge_density[n] = 0.5 * (sz_exp[n] + sgn);
  }

  // Electric field per link: <L_l> = background + Σ_{k=0}^{l} <Q_k>.
  double cum_charge = 0.0;
  for (int l = 0; l < N - 1; ++l) {
    cum_charge += f.charge_density[l];
    f.electric_field[l] = background + cum_charge;
  }

  // Chiral condensate: Σ = -(1/N) Σ_n (-1)^n <σz_n>.
  for (int n = 0; n < N; ++n) {
    double sgn = (n % 2 == 0) ? +1.0 : -1.0;
    chiral -= sgn * sz_exp[n];
  }
  f.chiral_condensate = chiral / N;

  // Particle number: Σ_n max(0, charge_density[n]).
  for (auto q : f.charge_density) {
    if (q > 0.0) f.particle_number += q;
  }

  return f;
}

}  // namespace

// ── Public API ─────────────────────────────────────────────────────────────

QfieldResult run_qfield(const QfieldOptions& opts) {
  auto t0 = std::chrono::steady_clock::now();

  const int N = static_cast<int>(opts.n_sites);
  if (N < 2 || N > 14) {
    return QfieldResult{false, 0, 0, 0, 0, 0, {}, {}, {},
                        0.0, "n_sites must be in [2, 14]"};
  }
  if (N % 2 != 0) {
    return QfieldResult{false, 0, 0, 0, 0, 0, {}, {}, {},
                        0.0, "n_sites must be even (staggered fermions)"};
  }

  QfieldResult result;
  result.n_sites = static_cast<std::uint32_t>(N);
  result.mass = opts.mass;
  result.coupling = opts.coupling;

  // ── Ground state ─────────────────────────────────────────────────────────
  Mat H = build_hamiltonian(N, opts.mass, opts.coupling, opts.background);
  const int dim = 1 << N;
  auto gs = find_ground_state(H, dim, opts.seed);

  result.ground_energy = gs.energy;
  auto gs_frame = compute_observables(gs.psi, N, opts.coupling, opts.background, 0.0);
  result.chiral_condensate = gs_frame.chiral_condensate;

  // ── Mass sweep ────────────────────────────────────────────────────────────
  if (opts.n_mass_sweep > 0) {
    double m_min = -2.0, m_max = 2.0;
    for (std::uint32_t i = 0; i < opts.n_mass_sweep; ++i) {
      double m = m_min + (m_max - m_min) * i / std::max(1u, opts.n_mass_sweep - 1);
      Mat Hi = build_hamiltonian(N, m, opts.coupling, opts.background);
      auto gsi = find_ground_state(Hi, dim, opts.seed);
      auto fi = compute_observables(gsi.psi, N, opts.coupling, opts.background, 0.0);
      result.mass_sweep.push_back({m, gsi.energy, fi.chiral_condensate});
    }
  }

  // ── String energies (vacuum energy + energy with one extra link charge) ──
  if (opts.scenario == "string") {
    // For each separation R (1..N/2), compute energy of state with
    // background field α = 1 on links [0..R-1] relative to vacuum.
    // This approximates the string tension: E(R) ≈ (g²/2)·R.
    double E_vac = result.ground_energy;
    for (int R = 1; R <= N / 2; ++R) {
      // Uniform background field 1.0 on first R links — approximated by
      // increasing α on a restricted region.  Full treatment requires
      // a position-dependent background which the C++ engine delegates to
      // the Python solver.  Here we use a global α = 1/(N-1) * R approximation.
      double alpha_eff = static_cast<double>(R) / (N - 1);
      Mat Hs = build_hamiltonian(N, opts.mass, opts.coupling, alpha_eff);
      auto gss = find_ground_state(Hs, dim, opts.seed);
      result.string_energies.push_back({static_cast<double>(R),
                                         gss.energy - E_vac});
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  result.elapsed_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  return result;
}

}  // namespace qx
