// src/anderson.cpp — M30-J: 1D Anderson Localization
//
// The 1D Anderson model of a disordered tight-binding chain:
//
//   H = Σ_i ε_i c†_i c_i  −  t Σ_{⟨i,i+1⟩} (c†_i c_{i+1} + h.c.)
//
// where ε_i ~ Uniform[-W/2, W/2] are uncorrelated on-site energies.
//
// In 1D all eigenstates are exponentially localised for any W > 0
// (Abrahams et al. 1979).  We diagnose localisation via:
//
//   IPR(n) = Σ_i |ψ_n(i)|^4   (Inverse Participation Ratio)
//            → 1 for perfectly localised, 1/N for extended
//
//   Localisation length ξ: in 1D, ξ ≈ 105.2 (t/W)² for W << t
//                          estimated from mean_IPR ~ C/ξ
//
//   Level-spacing ratio r = min(δ_n, δ_{n+1}) / max(δ_n, δ_{n+1})
//                          Poisson (localised): ⟨r⟩ ≈ 0.386
//                          GOE    (ergodic):    ⟨r⟩ ≈ 0.530
//
// Scenarios:
//   single — one disorder realisation at (t, W, seed): spectrum + IPR
//   sweep  — scan W from W_min to W_max, averaging IPR and r over n_samples

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace qx {
namespace {

using MatR = std::vector<double>;

// ─────────────────────────────────────────────────────────────────────────────
// A.  Portable popcount (SWAR method)  — for LCG seed mixing
// ─────────────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────────────
// B.  Build disordered tight-binding Hamiltonian (tridiagonal real symmetric)
//     Stored as diagonal + off-diagonal (no dense matrix needed — use Lanczos).
// ─────────────────────────────────────────────────────────────────────────────

struct TBChain {
    std::vector<double> diag;  // ε_i (on-site)
    std::vector<double> off;   // -t coupling (length N-1)
};

static TBChain build_chain(int N, double t, double W, std::mt19937& rng) {
    std::uniform_real_distribution<double> uni(-W / 2.0, W / 2.0);
    TBChain chain;
    chain.diag.resize(N);
    chain.off.resize(N - 1, -t);
    for (int i = 0; i < N; ++i) chain.diag[i] = uni(rng);
    return chain;
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  Exact diagonalization of real symmetric tridiagonal (LAPACK-free)
//     Wilkinson QR on the tridiagonal directly — no Householder needed.
// ─────────────────────────────────────────────────────────────────────────────

// Diagonalize tridiagonal with eigenvector accumulation.
static std::vector<double> tridiag_eig_full(std::vector<double> alpha,
                                              std::vector<double> beta,
                                              MatR& evecs_out) {
    int n = static_cast<int>(alpha.size());
    MatR Q(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) Q[i * n + i] = 1.0;

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
                double qi = Q[row*n+i], qi1 = Q[row*n+i+1];
                Q[row*n+i]   =  c*qi + s*qi1;
                Q[row*n+i+1] = -s*qi + c*qi1;
            }
        }
    }
    // Sort ascending
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return alpha[a] < alpha[b]; });
    std::vector<double> sorted(n);
    for (int i = 0; i < n; ++i) sorted[i] = alpha[idx[i]];
    MatR Qs(static_cast<std::size_t>(n) * n);
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < n; ++row)
            Qs[row*n+col] = Q[row*n+idx[col]];
    evecs_out = Qs;
    return sorted;
}

// ─────────────────────────────────────────────────────────────────────────────
// D.  Observables: IPR, level-spacing ratio, localisation length
// ─────────────────────────────────────────────────────────────────────────────

// IPR of each eigenstate column (col = eigenvector index)
static std::vector<double> compute_ipr(const MatR& evecs, int N) {
    std::vector<double> ipr(N, 0.0);
    for (int col = 0; col < N; ++col) {
        double s = 0.0;
        for (int row = 0; row < N; ++row) {
            double v = evecs[row * N + col];
            s += v * v * v * v;
        }
        ipr[col] = s;
    }
    return ipr;
}

// Mean level-spacing ratio ⟨r⟩ from sorted eigenvalues
static double mean_gap_ratio(const std::vector<double>& evals) {
    int n = static_cast<int>(evals.size());
    if (n < 3) return 0.0;
    double sum = 0.0; int cnt = 0;
    for (int i = 1; i < n - 1; ++i) {
        double d1 = evals[i] - evals[i - 1];
        double d2 = evals[i + 1] - evals[i];
        if (d1 > 1e-14 && d2 > 1e-14) {
            sum += std::min(d1, d2) / std::max(d1, d2);
            ++cnt;
        }
    }
    return cnt > 0 ? sum / cnt : 0.0;
}

// Localisation length: in 1D ξ = 1 / ⟨IPR⟩ when states are exponentially
// localised (⟨IPR⟩ ≈ 1/ξ for a box-localised state of length ξ).
static double loc_length_from_ipr(double mean_ipr, int N) {
    if (mean_ipr < 1e-14 || mean_ipr > 1.0) return 0.0;
    double xi = 1.0 / mean_ipr;
    return std::min(xi, static_cast<double>(N));
}

// ─────────────────────────────────────────────────────────────────────────────
// E.  Single-disorder solve
// ─────────────────────────────────────────────────────────────────────────────

struct AndersonSample {
    std::vector<double> evals;
    std::vector<double> ipr;
    double mean_ipr{0.0};
    double mean_r{0.0};
};

static AndersonSample solve_sample(int N, double t, double W, std::mt19937& rng) {
    TBChain chain = build_chain(N, t, W, rng);
    MatR evecs;
    auto evals = tridiag_eig_full(chain.diag, chain.off, evecs);
    auto ipr = compute_ipr(evecs, N);
    double sum_ipr = 0.0;
    for (double v : ipr) sum_ipr += v;
    AndersonSample s;
    s.evals = evals;
    s.ipr = ipr;
    s.mean_ipr = sum_ipr / N;
    s.mean_r = mean_gap_ratio(evals);
    return s;
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// F.  Public entry point
// ─────────────────────────────────────────────────────────────────────────────

AndersonResult run_anderson(const AndersonOptions& opts) {
    auto t0_wall = std::chrono::steady_clock::now();

    if (opts.n_sites < 4 || opts.n_sites > 2000)
        throw std::runtime_error("Anderson: n_sites must be in [4, 2000]");
    if (opts.t <= 0.0)
        throw std::runtime_error("Anderson: t must be positive");
    if (opts.W < 0.0)
        throw std::runtime_error("Anderson: W must be non-negative");
    if (opts.n_samples < 1)
        throw std::runtime_error("Anderson: n_samples must be >= 1");

    AndersonResult res;
    res.n_sites = opts.n_sites;
    res.t = opts.t;
    res.W = opts.W;

    int N = opts.n_sites;
    std::mt19937 rng(opts.seed);

    if (opts.scenario == "single" || opts.scenario == "sweep") {
        // Single realisation (for spectrum/IPR per state display)
        std::mt19937 rng0(opts.seed);
        auto samp = solve_sample(N, opts.t, opts.W, rng0);
        res.energies    = samp.evals;
        res.ipr_spectrum = samp.ipr;
        res.mean_ipr    = samp.mean_ipr;
        res.mean_r      = samp.mean_r;
        res.loc_length  = loc_length_from_ipr(res.mean_ipr, N);
    }

    if (opts.scenario == "sweep") {
        if (opts.n_sweep < 2)
            throw std::runtime_error("Anderson: n_sweep must be >= 2");
        res.sweep.reserve(opts.n_sweep);
        for (int k = 0; k < opts.n_sweep; ++k) {
            double Wk = opts.W_min + (opts.W_max - opts.W_min) * k / (opts.n_sweep - 1);
            if (Wk < 0.0) Wk = 0.0;
            // Average over n_samples disorder realisations
            double sum_ipr = 0.0, sum_r = 0.0;
            for (int samp = 0; samp < opts.n_samples; ++samp) {
                auto s = solve_sample(N, opts.t, Wk, rng);
                sum_ipr += s.mean_ipr;
                sum_r   += s.mean_r;
            }
            double avg_ipr = sum_ipr / opts.n_samples;
            double avg_r   = sum_r   / opts.n_samples;
            AndersonSweepPoint pt;
            pt.W          = Wk;
            pt.mean_ipr   = avg_ipr;
            pt.mean_r     = avg_r;
            pt.loc_length = loc_length_from_ipr(avg_ipr, N);
            res.sweep.push_back(pt);
        }
    }

    auto t1_wall = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1_wall - t0_wall).count();
    res.ok = true;
    return res;
}

}  // namespace qx
