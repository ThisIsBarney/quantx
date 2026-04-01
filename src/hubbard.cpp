// src/hubbard.cpp — M27-C: 1D Hubbard Model
//
//   H = −t Σ_{⟨i,j⟩,σ} (c†_{i,σ} c_{j,σ} + h.c.) + U Σ_i n_{i↑} n_{i↓}
//
// Basis: product |σ_up⟩⊗|σ_dn⟩ at fixed electron numbers.
// CSR sparse Hamiltonian + real Lanczos with full reorthogonalization.
// Observables: E₀, gap, double occupancy, spin correlations, n(k).

#include "qx/engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qx {
namespace {

// ─────────────────────────────────────────────────────────────────────────────
// A.  Basis enumeration
// ─────────────────────────────────────────────────────────────────────────────

// All N-bit integers with exactly n bits set (Gosper's hack), ascending.
static std::vector<int> enum_states(int N, int n) {
  std::vector<int> st;
  if (n == 0) { st.push_back(0); return st; }
  if (n > N)  return st;
  int s = (1 << n) - 1;
  int limit = 1 << N;
  while (s < limit) {
    st.push_back(s);
    int c = s & (-s);
    int r = s + c;
    s = (((r ^ s) >> 2) / c) | r;
  }
  return st;
}

// Jordan-Wigner sign for c†_i c_j:
// (−1)^(# bits strictly between min(i,j) and max(i,j) in state).
static int jw_sign(int state, int i, int j) {
  int lo = std::min(i, j), hi = std::max(i, j);
  if (hi - lo <= 1) return 1;  // nearest-neighbour: no intermediate sites
  int mask = ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1);
  return (__builtin_popcount(static_cast<unsigned>(state & mask)) % 2 == 0) ? 1 : -1;
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  CSR sparse matrix (real, symmetric)
// ─────────────────────────────────────────────────────────────────────────────

struct HCsr {
  int dim{0};
  std::vector<double> vals;
  std::vector<int>    cols;
  std::vector<int>    row_ptr;  // length dim+1
};

static void hcsr_matvec(const HCsr& A, const std::vector<double>& x, std::vector<double>& y) {
  for (int r = 0; r < A.dim; ++r) {
    double s = 0.0;
    for (int p = A.row_ptr[r]; p < A.row_ptr[r + 1]; ++p)
      s += A.vals[p] * x[A.cols[p]];
    y[r] = s;
  }
}

struct CooEntry { int row, col; double val; };

static HCsr coo_to_hcsr(std::vector<CooEntry> coo, int dim) {
  std::sort(coo.begin(), coo.end(), [](const CooEntry& a, const CooEntry& b) {
    return a.row < b.row || (a.row == b.row && a.col < b.col);
  });
  // Merge duplicates
  std::vector<CooEntry> merged;
  merged.reserve(coo.size());
  for (auto& e : coo) {
    if (!merged.empty() && merged.back().row == e.row && merged.back().col == e.col)
      merged.back().val += e.val;
    else
      merged.push_back(e);
  }
  HCsr csr;
  csr.dim = dim;
  csr.row_ptr.resize(static_cast<std::size_t>(dim) + 1, 0);
  for (auto& e : merged) csr.row_ptr[e.row + 1]++;
  for (int r = 0; r < dim; ++r) csr.row_ptr[r + 1] += csr.row_ptr[r];
  csr.vals.resize(merged.size());
  csr.cols.resize(merged.size());
  std::vector<int> pos(csr.row_ptr.begin(), csr.row_ptr.end() - 1);
  for (auto& e : merged) {
    int p = pos[e.row]++;
    csr.vals[p] = e.val;
    csr.cols[p] = e.col;
  }
  return csr;
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  Build Hubbard Hamiltonian (CSR)
// ─────────────────────────────────────────────────────────────────────────────

static HCsr build_hubbard_csr(int n_sites, double t_hop, double U,
                               const std::vector<int>& up_st,
                               const std::vector<int>& dn_st,
                               const std::map<int, int>& up_map,
                               const std::map<int, int>& dn_map) {
  int n_up = static_cast<int>(up_st.size());
  int n_dn = static_cast<int>(dn_st.size());
  int dim  = n_up * n_dn;

  std::vector<CooEntry> coo;
  coo.reserve(static_cast<std::size_t>(dim) * 2 * n_sites);

  for (int iu = 0; iu < n_up; ++iu) {
    int us = up_st[iu];
    for (int id = 0; id < n_dn; ++id) {
      int ds  = dn_st[id];
      int idx = iu * n_dn + id;

      // Diagonal: U * double occupancy
      int docc = __builtin_popcount(static_cast<unsigned>(us & ds));
      if (docc) coo.push_back({idx, idx, U * docc});

      // Hopping in spin-up sector: c†_{site+1} c_{site}
      for (int site = 0; site < n_sites - 1; ++site) {
        int b0 = 1 << site, b1 = 1 << (site + 1);
        if ((us & b0) && !(us & b1)) {
          int us_new = (us ^ b0) | b1;
          auto it = up_map.find(us_new);
          if (it != up_map.end()) {
            int iu2 = it->second;
            int idx2 = iu2 * n_dn + id;
            double sg = static_cast<double>(jw_sign(us, site, site + 1));
            coo.push_back({idx2, idx,  -t_hop * sg});
            coo.push_back({idx,  idx2, -t_hop * sg});
          }
        }
      }

      // Hopping in spin-down sector: c†_{site+1} c_{site}
      for (int site = 0; site < n_sites - 1; ++site) {
        int b0 = 1 << site, b1 = 1 << (site + 1);
        if ((ds & b0) && !(ds & b1)) {
          int ds_new = (ds ^ b0) | b1;
          auto it = dn_map.find(ds_new);
          if (it != dn_map.end()) {
            int id2  = it->second;
            int idx2 = iu * n_dn + id2;
            double sg = static_cast<double>(jw_sign(ds, site, site + 1));
            coo.push_back({idx2, idx,  -t_hop * sg});
            coo.push_back({idx,  idx2, -t_hop * sg});
          }
        }
      }
    }
  }

  return coo_to_hcsr(std::move(coo), dim);
}

// ─────────────────────────────────────────────────────────────────────────────
// D.  Tridiagonal eigensolver (Wilkinson-shift implicit QR)
// ─────────────────────────────────────────────────────────────────────────────

static std::pair<std::vector<double>, std::vector<double>>
tridiag_eig_h(const std::vector<double>& alpha_in, const std::vector<double>& beta_in) {
  int k = static_cast<int>(alpha_in.size());
  std::vector<double> a = alpha_in, b = beta_in;
  std::vector<double> Q(static_cast<std::size_t>(k) * k, 0.0);
  for (int i = 0; i < k; ++i)
    Q[static_cast<std::size_t>(i) + static_cast<std::size_t>(i) * k] = 1.0;

  for (int iter = 0; iter < 30 * k; ++iter) {
    int n = k;
    while (n > 1 && std::abs(b[n - 2]) < 1e-10 * (std::abs(a[n - 2]) + std::abs(a[n - 1])))
      --n;
    if (n == 1) break;
    int m = n - 1;
    while (m > 0 && std::abs(b[m - 1]) >= 1e-10 * (std::abs(a[m - 1]) + std::abs(a[m])))
      --m;
    double d_w   = (a[n - 2] - a[n - 1]) * 0.5;
    double sign_d = (d_w >= 0.0) ? 1.0 : -1.0;
    double mu    = a[n - 1] - b[n - 2] * b[n - 2] /
                   (d_w + sign_d * std::sqrt(d_w * d_w + b[n - 2] * b[n - 2]));
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
      if (i < n - 2) { double bi1 = b[i+1]; b[i+1] = c * bi1; z = s * bi1; x = b[i]; }
      for (int j = 0; j < k; ++j) {
        double qi  = Q[static_cast<std::size_t>(j) + static_cast<std::size_t>(i)     * k];
        double qi1 = Q[static_cast<std::size_t>(j) + static_cast<std::size_t>(i + 1) * k];
        Q[static_cast<std::size_t>(j) + static_cast<std::size_t>(i)     * k] =  c * qi + s * qi1;
        Q[static_cast<std::size_t>(j) + static_cast<std::size_t>(i + 1) * k] = -s * qi + c * qi1;
      }
    }
  }
  std::vector<int> idx(k);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int i, int j) { return a[i] < a[j]; });
  std::vector<double> evals(k), evecs(static_cast<std::size_t>(k) * k);
  for (int j = 0; j < k; ++j) {
    evals[j] = a[idx[j]];
    for (int r = 0; r < k; ++r)
      evecs[static_cast<std::size_t>(r) + static_cast<std::size_t>(j) * k] =
          Q[static_cast<std::size_t>(r) + static_cast<std::size_t>(idx[j]) * k];
  }
  return {evals, evecs};
}

// ─────────────────────────────────────────────────────────────────────────────
// E.  Lanczos eigensolver (full reorthogonalization, real)
// ─────────────────────────────────────────────────────────────────────────────

struct HEigPair { double eval{0.0}; std::vector<double> evec; };

static std::pair<HEigPair, HEigPair>
hubbard_lanczos(const HCsr& H, int seed_val = 42) {
  int dim = H.dim;
  if (dim <= 0) { HEigPair bad; return {bad, bad}; }

  int k_max = std::min(std::min(200, dim / 4 + 10), dim);
  k_max = std::max(k_max, 2);

  // Random start vector (LCG)
  std::vector<double> v(dim);
  {
    std::uint64_t lcg = static_cast<std::uint64_t>(seed_val) * 6364136223846793005ULL + 1;
    for (int i = 0; i < dim; ++i) {
      lcg = lcg * 6364136223846793005ULL + 1442695040888963407ULL;
      v[i] = static_cast<double>(static_cast<std::int64_t>(lcg >> 33)) /
             static_cast<double>(1LL << 30);
    }
  }
  double norm = 0.0;
  for (auto x : v) norm += x * x;
  norm = std::sqrt(norm);
  for (auto& x : v) x /= norm;

  std::vector<double> alpha, beta;
  std::vector<std::vector<double>> Q;
  Q.reserve(k_max + 1);
  Q.push_back(v);
  std::vector<double> w(dim);
  double beta_prev = 0.0;
  int k_actual = 0;

  for (int j = 0; j < k_max; ++j) {
    hcsr_matvec(H, Q[j], w);
    double alpha_j = 0.0;
    for (int i = 0; i < dim; ++i) alpha_j += Q[j][i] * w[i];
    alpha.push_back(alpha_j);
    for (int i = 0; i < dim; ++i) w[i] -= alpha_j * Q[j][i];
    if (j > 0)
      for (int i = 0; i < dim; ++i) w[i] -= beta_prev * Q[j - 1][i];
    // Two-pass reorthogonalization
    for (int pass = 0; pass < 2; ++pass)
      for (int l = 0; l <= j; ++l) {
        double dot = 0.0;
        for (int i = 0; i < dim; ++i) dot += Q[l][i] * w[i];
        for (int i = 0; i < dim; ++i) w[i] -= dot * Q[l][i];
      }
    double beta_j = 0.0;
    for (auto x : w) beta_j += x * x;
    beta_j = std::sqrt(beta_j);
    k_actual = j + 1;
    if (beta_j < 1e-12) break;
    beta.push_back(beta_j);
    beta_prev = beta_j;
    std::vector<double> qnew(dim);
    for (int i = 0; i < dim; ++i) qnew[i] = w[i] / beta_j;
    Q.push_back(qnew);
  }

  auto [evals, evecs_k] = tridiag_eig_h(alpha, beta);
  int k = k_actual;

  auto make_ritz = [&](int col) -> HEigPair {
    col = std::min(col, k - 1);
    HEigPair ep;
    ep.eval = evals[col];
    ep.evec.assign(dim, 0.0);
    for (int j = 0; j < k; ++j) {
      double y = evecs_k[static_cast<std::size_t>(j) + static_cast<std::size_t>(col) * k];
      for (int i = 0; i < dim; ++i) ep.evec[i] += y * Q[j][i];
    }
    // normalize
    double nn = 0.0;
    for (auto x : ep.evec) nn += x * x;
    nn = std::sqrt(nn);
    if (nn > 1e-300) for (auto& x : ep.evec) x /= nn;
    return ep;
  };

  return {make_ritz(0), make_ritz(1)};
}

// ─────────────────────────────────────────────────────────────────────────────
// F.  Observable computation
// ─────────────────────────────────────────────────────────────────────────────

struct HubbardObs {
  double double_occupancy{0.0};
  double spin_structure_pi{0.0};
  std::vector<double> spin_corr;
  std::vector<double> momentum_dist;
};

static HubbardObs compute_obs(int n_sites,
                               const std::vector<int>& up_st,
                               const std::vector<int>& dn_st,
                               const std::map<int, int>& up_map,
                               const std::map<int, int>& dn_map,
                               const std::vector<double>& psi) {
  HubbardObs obs;
  int n_up = static_cast<int>(up_st.size());
  int n_dn = static_cast<int>(dn_st.size());
  obs.spin_corr.assign(n_sites, 0.0);

  double docc = 0.0;
  for (int iu = 0; iu < n_up; ++iu) {
    int us = up_st[iu];
    for (int id = 0; id < n_dn; ++id) {
      int ds = dn_st[id];
      int idx = iu * n_dn + id;
      double p = psi[idx] * psi[idx];
      docc += p * __builtin_popcount(static_cast<unsigned>(us & ds));
      // spin correlations
      double sz0 = 0.5 * (((us >> 0) & 1) - ((ds >> 0) & 1));
      for (int r = 0; r < n_sites; ++r) {
        double szr = 0.5 * (((us >> r) & 1) - ((ds >> r) & 1));
        obs.spin_corr[r] += p * sz0 * szr;
      }
    }
  }
  obs.double_occupancy = docc / n_sites;

  double s_pi = 0.0;
  for (int r = 0; r < n_sites; ++r)
    s_pi += ((r % 2 == 0) ? 1.0 : -1.0) * obs.spin_corr[r];
  obs.spin_structure_pi = s_pi / n_sites;

  // One-body density matrix: rho[r,s] = <c†_r c_s> (up + dn)
  std::vector<double> rho(static_cast<std::size_t>(n_sites) * n_sites, 0.0);

  // spin-up contribution
  for (int iu = 0; iu < n_up; ++iu) {
    int us = up_st[iu];
    for (int id = 0; id < n_dn; ++id) {
      int idx = iu * n_dn + id;
      double pv = psi[idx];
      for (int s = 0; s < n_sites; ++s) {
        if (!(us & (1 << s))) continue;
        // diagonal
        rho[static_cast<std::size_t>(s) * n_sites + s] += pv * pv;
        // off-diagonal
        for (int r = 0; r < n_sites; ++r) {
          if (r == s || (us & (1 << r))) continue;
          int us_new = (us & ~(1 << s)) | (1 << r);
          auto it = up_map.find(us_new);
          if (it == up_map.end()) continue;
          int idx2 = it->second * n_dn + id;
          double sg = static_cast<double>(jw_sign(us, s, r));
          rho[static_cast<std::size_t>(r) * n_sites + s] += psi[idx2] * pv * sg;
        }
      }
    }
  }
  // spin-down contribution
  for (int iu = 0; iu < n_up; ++iu) {
    int us = up_st[iu];
    for (int id = 0; id < n_dn; ++id) {
      int ds = dn_st[id];
      int idx = iu * n_dn + id;
      double pv = psi[idx];
      for (int s = 0; s < n_sites; ++s) {
        if (!(ds & (1 << s))) continue;
        rho[static_cast<std::size_t>(s) * n_sites + s] += pv * pv;
        for (int r = 0; r < n_sites; ++r) {
          if (r == s || (ds & (1 << r))) continue;
          int ds_new = (ds & ~(1 << s)) | (1 << r);
          auto it = dn_map.find(ds_new);
          if (it == dn_map.end()) continue;
          int idx2 = iu * n_dn + it->second;
          double sg = static_cast<double>(jw_sign(ds, s, r));
          rho[static_cast<std::size_t>(r) * n_sites + s] += psi[idx2] * pv * sg;
        }
      }
    }
  }

  // n(k_m) = (1/N) Σ_{r,s} e^{i k_m (r−s)} ρ_{rs}
  // For real symmetric ρ: n(k) = (1/N)[Σ_r ρ_{rr} + 2 Σ_{r<s} cos(k(r−s)) ρ_{rs}]
  obs.momentum_dist.assign(n_sites, 0.0);
  for (int m = 0; m < n_sites; ++m) {
    double km = 2.0 * M_PI * m / n_sites;
    double nk = 0.0;
    for (int r = 0; r < n_sites; ++r)
      for (int s = 0; s < n_sites; ++s)
        nk += std::cos(km * (r - s)) * rho[static_cast<std::size_t>(r) * n_sites + s];
    obs.momentum_dist[m] = nk / n_sites;
  }

  return obs;
}

// ─────────────────────────────────────────────────────────────────────────────
// G.  Single-point Hubbard solve
// ─────────────────────────────────────────────────────────────────────────────

struct HubbardPoint {
  double E0{0.0}, E1{0.0};
  std::vector<double> psi0;
};

static HubbardPoint run_point(int n_sites, double t_hop, double U,
                               const std::vector<int>& up_st,
                               const std::vector<int>& dn_st,
                               const std::map<int, int>& up_map,
                               const std::map<int, int>& dn_map) {
  HCsr H = build_hubbard_csr(n_sites, t_hop, U, up_st, dn_st, up_map, dn_map);
  auto [gs, es] = hubbard_lanczos(H);
  return {gs.eval, es.eval, gs.evec};
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

HubbardResult run_hubbard(const HubbardOptions& opts) {
  auto t0 = std::chrono::steady_clock::now();
  HubbardResult res;

  try {
    int N = static_cast<int>(opts.n_sites);
    if (N < 2 || N > 14)
      throw std::runtime_error("n_sites must be in [2, 14]");

    int n_up = static_cast<int>(opts.n_up > 0 ? opts.n_up : opts.n_sites / 2);
    int n_dn = static_cast<int>(opts.n_dn > 0 ? opts.n_dn : opts.n_sites / 2);

    auto up_st = enum_states(N, n_up);
    auto dn_st = enum_states(N, n_dn);
    std::map<int, int> up_map, dn_map;
    for (int i = 0; i < static_cast<int>(up_st.size()); ++i) up_map[up_st[i]] = i;
    for (int i = 0; i < static_cast<int>(dn_st.size()); ++i) dn_map[dn_st[i]] = i;

    res.n_sites = opts.n_sites;
    res.n_up    = static_cast<std::uint32_t>(n_up);
    res.n_dn    = static_cast<std::uint32_t>(n_dn);
    res.t_hop   = opts.t_hop;
    res.U       = opts.U;

    if (opts.scenario == "sweep") {
      int ns = static_cast<int>(opts.n_sweep);
      for (int si = 0; si < ns; ++si) {
        double U_val = opts.U_min + (opts.U_max - opts.U_min) * si / std::max(ns - 1, 1);
        auto pt = run_point(N, opts.t_hop, U_val, up_st, dn_st, up_map, dn_map);
        auto obs = compute_obs(N, up_st, dn_st, up_map, dn_map, pt.psi0);
        HubbardSweepPoint sp;
        sp.U                  = U_val;
        sp.ground_energy      = pt.E0;
        sp.double_occupancy   = obs.double_occupancy;
        sp.spin_structure_pi  = obs.spin_structure_pi;
        sp.energy_gap         = pt.E1 - pt.E0;
        res.sweep.push_back(sp);
      }
      // Final ground-state observables at opts.U
      auto pt  = run_point(N, opts.t_hop, opts.U, up_st, dn_st, up_map, dn_map);
      auto obs = compute_obs(N, up_st, dn_st, up_map, dn_map, pt.psi0);
      res.ground_energy     = pt.E0;
      res.energy_gap        = pt.E1 - pt.E0;
      res.double_occupancy  = obs.double_occupancy;
      res.spin_structure_pi = obs.spin_structure_pi;
      res.spin_corr         = obs.spin_corr;
      res.momentum_dist     = obs.momentum_dist;
    } else {
      // ground_state scenario
      auto pt  = run_point(N, opts.t_hop, opts.U, up_st, dn_st, up_map, dn_map);
      auto obs = compute_obs(N, up_st, dn_st, up_map, dn_map, pt.psi0);
      res.ground_energy     = pt.E0;
      res.energy_gap        = pt.E1 - pt.E0;
      res.double_occupancy  = obs.double_occupancy;
      res.spin_structure_pi = obs.spin_structure_pi;
      res.spin_corr         = obs.spin_corr;
      res.momentum_dist     = obs.momentum_dist;
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
