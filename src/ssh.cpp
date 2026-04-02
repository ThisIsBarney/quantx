// src/ssh.cpp — M29-H: Su-Schrieffer-Heeger (SSH) Topological Insulator
//
// The 1D SSH model describes spinless fermions on a dimerized chain:
//
//   H = -t₁ Σ_{i=0}^{N-2,even} (c†_{i} c_{i+1} + h.c.)
//       -t₂ Σ_{i=1}^{N-2,odd}  (c†_{i} c_{i+1} + h.c.)
//
// with N = 2*n_cells sites, alternating intra-cell (t₁) and inter-cell (t₂)
// hoppings.  The dimerization parameter δ = (t₂ - t₁) / (t₁ + t₂) controls
// the topology:
//   δ < 0  →  t₁ > t₂  →  trivial insulator (Zak phase = 0)
//   δ > 0  →  t₂ > t₁  →  topological insulator (Zak phase = π, edge states)
//
// Everything here is single-particle: we build and diagonalize the 2N×2N
// hopping matrix H directly.  The Zak (Berry) phase is computed via the
// discrete Wilson loop over the occupied bands in k-space.
//
// Scenarios:
//   spectrum    — diagonalize at (t, δ), report spectrum, gap, Zak phase, edge
//   sweep       — scan δ from delta_min to delta_max, track gap and Zak phase

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace qx {
namespace {

using cx   = std::complex<double>;
using MatC = std::vector<cx>;
using MatR = std::vector<double>;

// ─────────────────────────────────────────────────────────────────────────────
// A.  Build SSH hopping matrix  (real, symmetric, 2n_cells × 2n_cells)
// ─────────────────────────────────────────────────────────────────────────────

static MatR build_ssh_matrix(int n_cells, double t, double delta) {
    int N = 2 * n_cells;
    double t1 = t * (1.0 - delta);  // intra-cell hopping
    double t2 = t * (1.0 + delta);  // inter-cell hopping

    MatR H(static_cast<std::size_t>(N) * N, 0.0);
    for (int i = 0; i < N - 1; ++i) {
        double tij = (i % 2 == 0) ? t1 : t2;
        H[i * N + (i + 1)] = -tij;
        H[(i + 1) * N + i] = -tij;
    }
    return H;
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  Dense real symmetric eigensolver (Householder + Wilkinson QR)
//     (same algorithm as tfi.cpp — reproduced to keep files self-contained)
// ─────────────────────────────────────────────────────────────────────────────

// householder_tridiag: T = V^T A V with V orthogonal (Householder reflectors).
// If V_out != nullptr the accumulated V is stored so that eigenvectors of A can
// be recovered as V * (eigenvectors of T).
static void householder_tridiag(const MatR& A, std::vector<double>& alpha,
                                  std::vector<double>& beta, int n,
                                  MatR* V_out = nullptr) {
    alpha.resize(n); beta.resize(n - 1);
    MatR Q = A;

    MatR V;
    if (V_out) {
        V.assign(static_cast<std::size_t>(n) * n, 0.0);
        for (int i = 0; i < n; ++i) V[i * n + i] = 1.0;
    }

    for (int k = 0; k < n - 2; ++k) {
        double norm2 = 0.0;
        for (int i = k + 1; i < n; ++i) norm2 += Q[i * n + k] * Q[i * n + k];
        double norm = std::sqrt(norm2);
        if (norm < 1e-15) continue;

        std::vector<double> v(n, 0.0);
        for (int i = k + 1; i < n; ++i) v[i] = Q[i * n + k];
        v[k + 1] += (v[k + 1] >= 0.0 ? 1.0 : -1.0) * norm;
        double vnorm2 = 0.0;
        for (int i = k + 1; i < n; ++i) vnorm2 += v[i] * v[i];
        if (vnorm2 < 1e-30) continue;

        std::vector<double> p(n, 0.0);
        for (int r = 0; r < n; ++r)
            for (int c = k + 1; c < n; ++c)
                p[r] += Q[r * n + c] * v[c];
        for (int r = 0; r < n; ++r) p[r] /= (vnorm2 / 2.0);

        double vTpv = 0.0;
        for (int i = k + 1; i < n; ++i) vTpv += v[i] * p[i];
        std::vector<double> w(n);
        for (int i = 0; i < n; ++i) w[i] = p[i] - (vTpv / vnorm2) * v[i];

        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c)
                Q[r * n + c] -= v[r] * w[c] + w[r] * v[c];

        // Accumulate right reflector: V ← V P  (VP)[row,col] = V[row,col] - 2/vnorm2*(V·v)[row]*v[col]
        if (V_out) {
            std::vector<double> Vv(n, 0.0);
            for (int row = 0; row < n; ++row)
                for (int j = k + 1; j < n; ++j)
                    Vv[row] += V[row * n + j] * v[j];
            for (int row = 0; row < n; ++row)
                for (int c2 = k + 1; c2 < n; ++c2)
                    V[row * n + c2] -= (2.0 / vnorm2) * Vv[row] * v[c2];
        }
    }
    for (int i = 0; i < n; ++i) alpha[i] = Q[i * n + i];
    for (int i = 0; i < n - 1; ++i) beta[i] = Q[(i + 1) * n + i];
    if (V_out) *V_out = V;
}

static std::vector<double> tridiag_eig_evecs(std::vector<double> alpha,
                                               std::vector<double> beta,
                                               MatR& evecs) {
    int n = static_cast<int>(alpha.size());
    evecs.assign(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) evecs[i * n + i] = 1.0;

    for (int end = n - 1; end > 0; ) {
        for (int i = 0; i < end; ++i)
            if (std::abs(beta[i]) < 1e-14 * (std::abs(alpha[i]) + std::abs(alpha[i + 1])))
                beta[i] = 0.0;

        int start = end;
        while (start > 0 && beta[start - 1] != 0.0) --start;
        if (start == end) { --end; continue; }

        double d  = (alpha[end - 1] - alpha[end]) * 0.5;
        double sg = (d >= 0.0) ? 1.0 : -1.0;
        double mu = alpha[end] - sg * beta[end - 1] * beta[end - 1] /
                    (std::abs(d) + std::sqrt(d * d + beta[end - 1] * beta[end - 1]));
        double x = alpha[start] - mu, z = beta[start];

        for (int i = start; i < end; ++i) {
            double r = std::sqrt(x * x + z * z);
            double c = x / r, s = z / r;
            if (i > start) beta[i - 1] = r;
            double a0 = alpha[i], a1 = alpha[i + 1], b = beta[i];
            alpha[i]     = c*c*a0 + 2*c*s*b + s*s*a1;
            alpha[i + 1] = s*s*a0 - 2*c*s*b + c*c*a1;
            beta[i]      = c*s*(a1 - a0) + (c*c - s*s)*b;
            if (i < end - 1) { x = beta[i]; z = -s * beta[i + 1]; beta[i + 1] *= c; }
            for (int row = 0; row < n; ++row) {
                double qi = evecs[row*n+i], qi1 = evecs[row*n+i+1];
                evecs[row*n+i]   =  c*qi + s*qi1;
                evecs[row*n+i+1] = -s*qi + c*qi1;
            }
        }
    }

    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return alpha[a] < alpha[b]; });
    std::vector<double> sorted(n);
    for (int i = 0; i < n; ++i) sorted[i] = alpha[idx[i]];
    MatR Qsorted(static_cast<std::size_t>(n) * n);
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < n; ++row)
            Qsorted[row*n+col] = evecs[row*n+idx[col]];
    evecs = Qsorted;
    return sorted;
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  Zak (Berry) phase via Wilson loop in k-space
//
//  For a finite open chain we can't directly use k-space periodicity.
//  Instead we use the r-space approach: Zak phase = π × winding number
//  of the occupied-band projector.  For N even with N/2 occupied bands,
//  the Zak phase is:
//
//    θ_Zak = -Im log det [⟨u_i^occ|u_j^occ⟩] restricted to unit-cell A-sublattice
//
//  In the open-chain limit we use the "entanglement-based" indicator:
//  count the number of eigenvalues of the entanglement Hamiltonian near 0.5,
//  which signals edge modes.  For practical output we compute:
//
//   • Whether the mid-gap states exist (|E| < gap/4)
//   • Approximate Zak phase as 0 (no edge states) or π (edge states present)
//
// ─────────────────────────────────────────────────────────────────────────────

// Discrete Wilson loop for the Zak phase on a periodic chain (k-space).
// Build periodic Bloch Hamiltonian H(k) = t1 + t2 e^{ik} (lower off-diag A→B)
// for k = 2πm/n_cells, m=0..n_cells-1.
// Occupied band |u_k⟩ (lower eigenvalue of 2×2 H(k)).
// Zak phase = -Im log Π_{k} ⟨u_k | u_{k+1}⟩.
static double zak_phase_periodic(int n_cells, double t, double delta) {
    double t1 = t * (1.0 - delta);
    double t2 = t * (1.0 + delta);

    int nk = n_cells;
    // Occupied eigenvector at each k (lower band = negative energy)
    // 2×2 BdG Hamiltonian in (A,B) sublattice:
    //   H(k) = [[0, d(k)], [d*(k), 0]]
    //   d(k) = -(t1 + t2 e^{ik})
    std::vector<cx> u_occ(nk);
    for (int m = 0; m < nk; ++m) {
        double k = 2.0 * M_PI * m / nk;
        cx dk = cx{-t1, 0.0} + cx{-t2 * std::cos(k), -t2 * std::sin(k)};
        double absdk = std::abs(dk);
        // Eigenvector of lower eigenvalue (-|d|): [d*/|d|, -1]/sqrt(2) up to phase
        cx phase_factor = (absdk > 1e-14) ? dk / absdk : cx{1.0, 0.0};
        u_occ[m] = phase_factor;  // store only the (B component ratio)
    }

    // Wilson loop: Π ⟨u_k | u_{k+1}⟩  (overlap of 1D unit vector phases)
    cx W{1.0, 0.0};
    for (int m = 0; m < nk; ++m) {
        int m1 = (m + 1) % nk;
        // Full 2-component dot product: ⟨u_m|u_{m1}⟩ = conj(u_m) · u_{m1}
        // Using the representation u = [d*/|d|, -1]/√2:
        cx dot = std::conj(u_occ[m]) * u_occ[m1] + cx{1.0, 0.0};
        // Normalize: |dot|² = |phase1* phase2|² + 1 = 2 → |dot|/√2
        double nd = std::abs(dot);
        if (nd > 1e-14) W *= dot / nd;
    }
    // Zak phase = -Im log W = arg(W)  (in [−π, π])
    double zak = -std::imag(std::log(W));
    // Round to 0 or π
    return (std::abs(zak) > M_PI / 2.0) ? M_PI : 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// D.  Solve SSH open chain: spectrum + edge states
// ─────────────────────────────────────────────────────────────────────────────

struct SSHSol {
    std::vector<double> energies;
    double gap{0.0};
    double zak_phase{0.0};
    std::vector<double> edge_weight;  // |ψ|² per site for lowest positive energy mode
};

static SSHSol solve_ssh(int n_cells, double t, double delta) {
    int N = 2 * n_cells;
    MatR H = build_ssh_matrix(n_cells, t, delta);
    std::vector<double> alpha, beta;
    MatR V;
    householder_tridiag(H, alpha, beta, N, &V);
    MatR Q_tri;
    auto evals = tridiag_eig_evecs(alpha, beta, Q_tri);
    // Back-transform: eigenvectors of H = V * (eigenvectors of T)
    MatR evecs(static_cast<std::size_t>(N) * N, 0.0);
    for (int row = 0; row < N; ++row)
        for (int col = 0; col < N; ++col)
            for (int k = 0; k < N; ++k)
                evecs[row * N + col] += V[row * N + k] * Q_tri[k * N + col];

    SSHSol sol;
    sol.energies = evals;

    // Bulk gap: edge states in the topological phase have |E| exponentially small
    // (~(t2/t1)^{-n_cells}), while bulk bands start at |E| ~ min(t1, t2).
    // Use a threshold of 0.3 * min(t1, t2) to exclude edge states and measure
    // the physical band gap between valence and conduction bands.
    double t1_abs = std::abs(t * (1.0 - delta));
    double t2_abs = std::abs(t * (1.0 + delta));
    double edge_thr = 0.3 * std::min(t1_abs, t2_abs);
    if (edge_thr < 1e-10) edge_thr = 1e-10;  // fallback for near-zero hopping
    double emax_neg = -1e30, emin_pos = 1e30;
    for (double e : evals) {
        if (e < -edge_thr) emax_neg = std::max(emax_neg, e);
        if (e >  edge_thr) emin_pos = std::min(emin_pos, e);
    }
    sol.gap = (emin_pos > emax_neg) ? emin_pos - emax_neg : 0.0;

    // Zak phase from k-space Wilson loop (uses periodic BdG)
    sol.zak_phase = zak_phase_periodic(n_cells, t, delta);

    // Edge weight: find mid-gap modes (|E| < edge_thr), report |ψ|² per site
    // for the mode closest to zero energy. Use edge_thr (same as gap threshold)
    // so that bulk-band states are excluded even when gap/4 would be too small.
    double thr = edge_thr;
    int edge_idx = -1;
    double best = 1e30;
    for (int i = 0; i < N; ++i) {
        if (std::abs(evals[i]) < thr && std::abs(evals[i]) < best) {
            best = std::abs(evals[i]);
            edge_idx = i;
        }
    }
    sol.edge_weight.resize(N, 0.0);
    if (edge_idx >= 0) {
        for (int site = 0; site < N; ++site)
            sol.edge_weight[site] = evecs[site * N + edge_idx] *
                                    evecs[site * N + edge_idx];
    }
    return sol;
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// E.  Public entry point
// ─────────────────────────────────────────────────────────────────────────────

SSHResult run_ssh(const SSHOptions& opts) {
    auto t0 = std::chrono::steady_clock::now();

    SSHResult res;
    res.n_cells = opts.n_cells;
    res.t = opts.t;
    res.delta = opts.delta;

    if (opts.n_cells < 2 || opts.n_cells > 200)
        throw std::runtime_error("SSH: n_cells must be in [2, 200]");
    if (opts.t <= 0.0)
        throw std::runtime_error("SSH: t must be positive");
    if (std::abs(opts.delta) >= 1.0)
        throw std::runtime_error("SSH: |delta| must be < 1");

    if (opts.scenario == "spectrum" || opts.scenario == "sweep") {
        SSHSol sol = solve_ssh(opts.n_cells, opts.t, opts.delta);
        res.energies    = sol.energies;
        res.gap         = sol.gap;
        res.zak_phase   = sol.zak_phase;
        res.edge_weight = sol.edge_weight;
    }

    if (opts.scenario == "sweep") {
        if (opts.n_sweep < 2)
            throw std::runtime_error("SSH: n_sweep must be >= 2");
        res.sweep.reserve(opts.n_sweep);
        for (int k = 0; k < opts.n_sweep; ++k) {
            double dk = opts.delta_min +
                        (opts.delta_max - opts.delta_min) * k / (opts.n_sweep - 1);
            if (std::abs(dk) >= 1.0) continue;
            SSHSol s = solve_ssh(opts.n_cells, opts.t, dk);
            SSHSweepPoint pt;
            pt.delta       = dk;
            pt.gap         = s.gap;
            pt.zak_phase   = s.zak_phase;
            // edge_weight: sum over left 2 sites
            pt.edge_weight = 0.0;
            for (int site = 0; site < 2 && site < (int)s.edge_weight.size(); ++site)
                pt.edge_weight += s.edge_weight[site];
            res.sweep.push_back(pt);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.ok = true;
    return res;
}

}  // namespace qx
