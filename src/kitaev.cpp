// src/kitaev.cpp — M30-I: Kitaev Chain (p-wave topological superconductor)
//
// The 1D Kitaev chain (Kitaev 2001) is the simplest model realising unpaired
// Majorana zero modes at the chain ends.  In the spinless-fermion language:
//
//   H = -μ Σ_i (c†_i c_i - 1/2)
//       - t Σ_{i=0}^{N-2} (c†_i c_{i+1} + h.c.)
//       + Δ Σ_{i=0}^{N-2} (c†_i c†_{i+1} + h.c.)
//
// Topological phase: |μ| < 2t  (two Majorana zero modes at chain ends)
// Trivial phase:     |μ| > 2t
//
// The model is solved via the Bogoliubov-de Gennes (BdG) transformation
// using the Nambu spinor basis (c, c†)^T.  The BdG Hamiltonian is the
// 2N × 2N real anti-symmetric matrix (in Nambu basis):
//
//   H_BdG = ( H_0    Δ  )   where H_0[i,j] = -t δ_{i,j±1} - μ/2 δ_{ij}
//            (-Δ   -H_0 )         Δ[i,j]   = +Δ δ_{i,j-1} - Δ δ_{i,j+1}  (antisym)
//
// Quasi-energies come in ±E pairs; the gap is the minimum positive E.
//
// Topological invariant Q: sign of the Pfaffian of the skew-symmetric BdG
// matrix at k=0 and k=π.  For a finite open chain we use the simpler
// criterion: Q = -1 if near-zero modes exist (|E| < gap/4), +1 otherwise.
//
// Majorana weight: the Majorana operator γ_L = c_0 + c†_0 at site 0.
// For a near-zero mode with BdG eigenvector (u_i, v_i), the Majorana
// content at site i is |u_i + v_i|² (normalised).
//
// Scenarios:
//   spectrum — single (t, Δ, μ) BdG spectrum + Majorana weight profile
//   sweep    — scan μ from μ_min to μ_max across the topological QPT at |μ|=2t

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace qx {
namespace {

using MatR = std::vector<double>;

// ─────────────────────────────────────────────────────────────────────────────
// A.  BdG Hamiltonian builder  (2N × 2N real symmetric)
//
// Nambu basis: ψ = (c_0, ..., c_{N-1}, c†_0, ..., c†_{N-1})^T
//   H_BdG[i,j]      = H_0[i,j]          (upper-left N×N block)
//   H_BdG[i+N,j+N]  = -H_0[i,j]         (lower-right block, ph symmetry)
//   H_BdG[i,j+N]    = Δ_ij              (upper-right pairing block)
//   H_BdG[i+N,j]    = Δ_ij              (lower-left, real → symmetric)
//
// H_0[i,i] = -μ/2,  H_0[i,i±1] = -t/2 (factor 1/2 from Nambu doubling)
// Δ[i,i+1] = +Δ/2,  Δ[i+1,i]   = -Δ/2 (antisymmetric, real)
// ─────────────────────────────────────────────────────────────────────────────

static MatR build_bdg(int N, double t, double delta, double mu) {
    int dim = 2 * N;
    MatR H(static_cast<std::size_t>(dim) * dim, 0.0);

    auto set = [&](int r, int c, double v) {
        H[r * dim + c] += v;
        H[c * dim + r] += v;  // symmetric
    };

    for (int i = 0; i < N; ++i) {
        // Diagonal: -μ/2 in (c,c) block, +μ/2 in (c†,c†) block
        H[i * dim + i]               += -mu / 2.0;
        H[(i + N) * dim + (i + N)]   +=  mu / 2.0;
    }
    for (int i = 0; i < N - 1; ++i) {
        // Hopping: -t/2 in (c,c) and -(−t/2) = +t/2 in (c†,c†)
        set(i,     i + 1,           -t / 2.0);  // (c,c) block; also adds (i+1,i)
        // (c†,c†) block: opposite sign
        H[(i + N) * dim + (i + 1 + N)] += t / 2.0;
        H[(i + 1 + N) * dim + (i + N)] += t / 2.0;

        // Pairing: upper-right block Δ[i,i+1] = +Δ/2 (antisymmetric: Δ[i+1,i]=-Δ/2)
        // In the real symmetric BdG the upper-right off-diagonal is Δ_ij (antisym in i,j)
        // H[i, j+N] = Δ_ij and H[j, i+N] = -Δ_ij (to keep antisymmetry of Δ)
        // Full BdG is symmetric so H[j+N,i] = H[i,j+N]
        H[i * dim + (i + 1 + N)]     += +delta / 2.0;
        H[(i + 1 + N) * dim + i]     += +delta / 2.0;
        H[(i + 1) * dim + (i + N)]   += -delta / 2.0;
        H[(i + N) * dim + (i + 1)]   += -delta / 2.0;
    }
    return H;
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  Real symmetric eigensolver (Householder + Wilkinson QR)
//     with reflector back-transform — identical pattern to tfi.cpp
// ─────────────────────────────────────────────────────────────────────────────

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

static std::vector<double> tridiag_eig(std::vector<double> alpha,
                                        std::vector<double> beta,
                                        MatR* evecs = nullptr) {
    int n = static_cast<int>(alpha.size());
    MatR Q;
    if (evecs) { Q.assign(static_cast<std::size_t>(n) * n, 0.0); for (int i = 0; i < n; ++i) Q[i * n + i] = 1.0; }
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
            if (evecs) {
                for (int row = 0; row < n; ++row) {
                    double qi = Q[row*n+i], qi1 = Q[row*n+i+1];
                    Q[row*n+i]   =  c*qi + s*qi1;
                    Q[row*n+i+1] = -s*qi + c*qi1;
                }
            }
        }
    }
    if (evecs) *evecs = Q;
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return alpha[a] < alpha[b]; });
    std::vector<double> sorted(n);
    for (int i = 0; i < n; ++i) sorted[i] = alpha[idx[i]];
    if (evecs) {
        MatR Qs(static_cast<std::size_t>(n) * n);
        for (int col = 0; col < n; ++col)
            for (int row = 0; row < n; ++row)
                Qs[row*n+col] = (*evecs)[row*n+idx[col]];
        *evecs = Qs;
    }
    return sorted;
}

// Full diagonalization with back-transform
struct BdGSol {
    std::vector<double> evals;  // all 2N energies sorted ascending
    MatR evecs;                 // columns = eigenvectors
};

static BdGSol solve_bdg(int N, double t, double delta, double mu) {
    int dim = 2 * N;
    MatR H = build_bdg(N, t, delta, mu);
    std::vector<double> alpha, beta;
    MatR V;
    householder_tridiag(H, alpha, beta, dim, &V);
    MatR Q_tri;
    auto evals = tridiag_eig(alpha, beta, &Q_tri);
    BdGSol sol;
    sol.evals = evals;
    sol.evecs.assign(static_cast<std::size_t>(dim) * dim, 0.0);
    for (int row = 0; row < dim; ++row)
        for (int col = 0; col < dim; ++col)
            for (int k = 0; k < dim; ++k)
                sol.evecs[row * dim + col] += V[row * dim + k] * Q_tri[k * dim + col];
    return sol;
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  Observables
// ─────────────────────────────────────────────────────────────────────────────

struct KitaevObs {
    double gap{0.0};
    double topo_inv{0.0};          // -1 (topo) or +1 (trivial)
    std::vector<double> majorana;  // |u_i + v_i|² per site, for lowest |E| mode
};

static KitaevObs compute_obs(int N, const BdGSol& sol) {
    KitaevObs obs;
    int dim = 2 * N;

    // Gap: smallest positive quasi-energy (BdG spectrum is ±E symmetric)
    double min_pos = 1e30;
    for (double e : sol.evals) if (e > 1e-10) min_pos = std::min(min_pos, e);
    obs.gap = (min_pos < 1e29) ? min_pos : 0.0;

    // Topological invariant: sign of parity = -1 if near-zero mode exists
    double thr = std::max(0.3 * std::min(std::abs(sol.evals.front()), min_pos), 1e-8);
    int n_zero = 0;
    for (double e : sol.evals) if (std::abs(e) < thr) ++n_zero;
    obs.topo_inv = (n_zero > 0) ? -1.0 : 1.0;

    // Majorana weight: collect all near-zero mode indices
    // In the topological phase there are exactly 2 (one ±E pair at ~0).
    // A single eigenvector from a degenerate pair can have arbitrary phase /
    // mixing, yielding all-zero u+v components.  We must combine BOTH modes.
    std::vector<int> zero_inds;
    for (int i = 0; i < dim; ++i)
        if (std::abs(sol.evals[i]) < thr) zero_inds.push_back(i);

    // BdG eigenvector: (u_0, ..., u_{N-1}, v_0, ..., v_{N-1})
    // Majorana operator at site i: γ_i = u_i + v_i
    obs.majorana.assign(N, 0.0);
    if (zero_inds.size() >= 2) {
        // Two near-zero modes (idx1 and idx2) may be arbitrarily mixed.
        // Build symmetric (+) and antisymmetric (-) Majorana combinations:
        //   majA[i] = |γ^(+)_i|² = |(u¹_i+u²_i) + (v¹_i+v²_i)|²
        //   majB[i] = |γ^(-)_i|² = |(u¹_i−u²_i) + (v¹_i−v²_i)|²
        // Pick the combination with larger weight at site 0 (left boundary).
        int idx1 = zero_inds[0], idx2 = zero_inds[1];
        std::vector<double> majA(N, 0.0), majB(N, 0.0);
        double nA = 0.0, nB = 0.0;
        for (int i = 0; i < N; ++i) {
            double u1 = sol.evecs[i * dim + idx1], v1 = sol.evecs[(i + N) * dim + idx1];
            double u2 = sol.evecs[i * dim + idx2], v2 = sol.evecs[(i + N) * dim + idx2];
            double gA = (u1 + u2) + (v1 + v2);
            double gB = (u1 - u2) + (v1 - v2);
            majA[i] = gA * gA;
            majB[i] = gB * gB;
            nA += majA[i];  nB += majB[i];
        }
        if (nA > 1e-14) for (auto& m : majA) m /= nA;
        if (nB > 1e-14) for (auto& m : majB) m /= nB;
        // Pick the combination with higher weight on either boundary (site 0 or site N-1)
        double bndA = std::max(majA[0], majA[N - 1]);
        double bndB = std::max(majB[0], majB[N - 1]);
        obs.majorana = (bndA >= bndB) ? majA : majB;
    } else {
        // Trivial phase or single mode: use the closest-to-zero eigenvector directly
        int zero_idx = 0;
        double best_e = 1e30;
        for (int i = 0; i < dim; ++i)
            if (std::abs(sol.evals[i]) < best_e) { best_e = std::abs(sol.evals[i]); zero_idx = i; }
        double norm = 0.0;
        for (int i = 0; i < N; ++i) {
            double u = sol.evecs[i * dim + zero_idx];
            double v = sol.evecs[(i + N) * dim + zero_idx];
            obs.majorana[i] = (u + v) * (u + v);
            norm += obs.majorana[i];
        }
        if (norm > 1e-14) for (auto& m : obs.majorana) m /= norm;
    }
    return obs;
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// D.  Public entry point
// ─────────────────────────────────────────────────────────────────────────────

KitaevResult run_kitaev(const KitaevOptions& opts) {
    auto t0 = std::chrono::steady_clock::now();

    if (opts.n_sites < 2 || opts.n_sites > 200)
        throw std::runtime_error("Kitaev: n_sites must be in [2, 200]");
    if (opts.t <= 0.0)
        throw std::runtime_error("Kitaev: t must be positive");
    if (opts.delta < 0.0)
        throw std::runtime_error("Kitaev: delta must be non-negative");

    KitaevResult res;
    res.n_sites = opts.n_sites;
    res.t = opts.t; res.delta = opts.delta; res.mu = opts.mu;

    int N = opts.n_sites;

    if (opts.scenario == "spectrum" || opts.scenario == "sweep") {
        BdGSol sol = solve_bdg(N, opts.t, opts.delta, opts.mu);
        KitaevObs obs = compute_obs(N, sol);
        res.energies       = sol.evals;
        res.gap            = obs.gap;
        res.topo_inv       = obs.topo_inv;
        res.majorana_weight = obs.majorana;
    }

    if (opts.scenario == "sweep") {
        if (opts.n_sweep < 2)
            throw std::runtime_error("Kitaev: n_sweep must be >= 2");
        res.sweep.reserve(opts.n_sweep);
        for (int k = 0; k < opts.n_sweep; ++k) {
            double muk = opts.mu_min + (opts.mu_max - opts.mu_min) * k / (opts.n_sweep - 1);
            BdGSol sol = solve_bdg(N, opts.t, opts.delta, muk);
            KitaevObs obs = compute_obs(N, sol);
            KitaevSweepPoint pt;
            pt.mu = muk;
            pt.gap = obs.gap;
            pt.topo_inv = obs.topo_inv;
            // boundary Majorana weight: max of left (sites 0,1) or right (sites N-2,N-1)
            {
                int sz = static_cast<int>(obs.majorana.size());
                double bdy_l = 0.0, bdy_r = 0.0;
                for (int i = 0; i < 2 && i < sz; ++i) bdy_l += obs.majorana[i];
                for (int i = sz - 1; i >= std::max(0, sz - 2); --i) bdy_r += obs.majorana[i];
                pt.majorana_weight = std::max(bdy_l, bdy_r);
            }
            res.sweep.push_back(pt);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.ok = true;
    return res;
}

}  // namespace qx
