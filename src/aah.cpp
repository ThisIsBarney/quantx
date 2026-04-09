// src/aah.cpp  —  M46: Aubry-André-Harper (AAH) Quasicrystal
//
// 1D tight-binding chain with quasiperiodic on-site potential (Aubry-André model):
//
//   H = Σ_n t (c†_{n+1}c_n + h.c.) + 2V cos(2πβn + φ) c†_n c_n
//
// The model is exactly self-dual under a Fourier transform when β is irrational
// (paradigmatic example: β = golden ratio = (√5−1)/2 ≈ 0.6180).
//
// Metal-insulator transition (MIT) at V = t:
//   V < t : all eigenstates extended  (IPR ~ N^{-1} → 0, metallic)
//   V = t : all eigenstates critical  (IPR ~ N^{-D₂}, multifractal)
//   V > t : all eigenstates localised (IPR = O(1), insulating)
//
// Lyapunov exponent (inverse localisation length) for V > t, any energy:
//   γ = ln(V/t)   (uniform across the spectrum — Aubry-André self-duality)
//
// For rational β = p/q the spectrum splits into q bands (Hofstadter column).
// For irrational β the spectrum is a Cantor set; the gap labels obey the
// Diophantine equation  N(E) = n + m·β  (mod 1) with m, n integers (TKNN).
//
// Scenarios:
//   "spectrum"  — eigenvalues and IPR at the supplied (V, β)
//   "phase"     — eigenvalue spectrum as V sweeps 0 → V_max (fractal butterfly)
//   "both"      — both

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

// Golden ratio: β = (√5 − 1)/2 ≈ 0.6180339887
static constexpr double GOLDEN_RATIO = 0.6180339887498948482;

namespace qx {

// ── Jacobi diagonalisation of a real symmetric N×N matrix ────────────────────
// Returns eigenvalues (unsorted) and overwrites M with eigenvectors (columns).
static std::vector<double> jacobi_real_sym(std::vector<double>& M, int N) {
    // M stored row-major; eigenvectors accumulated in V (identity → rotation product)
    std::vector<double> V(N * N, 0.0);
    for (int i = 0; i < N; ++i) V[i*N+i] = 1.0;

    const int max_sweep = 100 * N * N;
    for (int sweep = 0; sweep < max_sweep; ++sweep) {
        // Find off-diagonal maximum
        int p = 0, q = 1; double amax = 0.0;
        for (int i = 0; i < N; ++i)
            for (int j = i+1; j < N; ++j)
                if (std::abs(M[i*N+j]) > amax) { amax = std::abs(M[i*N+j]); p = i; q = j; }
        if (amax < 1e-13) break;

        const double app = M[p*N+p], aqq = M[q*N+q], apq = M[p*N+q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t   = (tau >= 0)
            ? 1.0 / (tau + std::sqrt(1.0 + tau*tau))
            : 1.0 / (tau - std::sqrt(1.0 + tau*tau));
        const double c   = 1.0 / std::sqrt(1.0 + t*t);
        const double s   = t * c;

        // Update diagonal
        M[p*N+p] = app - t*apq;
        M[q*N+q] = aqq + t*apq;
        M[p*N+q] = M[q*N+p] = 0.0;

        // Update off-diagonal rows/cols
        for (int r = 0; r < N; ++r) {
            if (r == p || r == q) continue;
            const double arp = M[r*N+p], arq = M[r*N+q];
            M[r*N+p] = M[p*N+r] = c*arp - s*arq;
            M[r*N+q] = M[q*N+r] = s*arp + c*arq;
        }

        // Accumulate eigenvectors
        for (int r = 0; r < N; ++r) {
            const double vrp = V[r*N+p], vrq = V[r*N+q];
            V[r*N+p] = c*vrp - s*vrq;
            V[r*N+q] = s*vrp + c*vrq;
        }
    }

    // Extract eigenvalues from diagonal
    std::vector<double> evals(N);
    for (int i = 0; i < N; ++i) evals[i] = M[i*N+i];
    M = std::move(V);  // M now holds eigenvectors (columns)
    return evals;
}

// ── Build AAH Hamiltonian (N×N real symmetric, flat row-major) ────────────────
static std::vector<double> build_aah(double t, double V, double beta, double phi, int N) {
    std::vector<double> H(N * N, 0.0);
    for (int n = 0; n < N; ++n) {
        // On-site: 2V cos(2π β n + φ)  (n is 1-indexed in the formula, 0-indexed here)
        H[n*N+n] = 2.0 * V * std::cos(2.0 * M_PI * beta * static_cast<double>(n) + phi);
        // Nearest-neighbour hopping (OBC)
        if (n + 1 < N) {
            H[n*N+(n+1)] = t;
            H[(n+1)*N+n] = t;
        }
    }
    return H;
}

// ── Inverse participation ratio: IPR = Σ_n |ψ_n|⁴ ────────────────────────────
// V holds eigenvectors as columns (N×N, row-major: V[row*N+col] = col-th evec, row-th component)
static std::vector<double> compute_ipr_from_evecs(const std::vector<double>& V, int N) {
    std::vector<double> ipr(N, 0.0);
    for (int col = 0; col < N; ++col) {
        double s = 0.0;
        for (int row = 0; row < N; ++row) {
            const double c = V[row*N+col];
            s += c*c*c*c;
        }
        ipr[col] = s;
    }
    return ipr;
}

// ── Spectrum at fixed (V, beta) ───────────────────────────────────────────────
static void compute_spectrum(const AAHOptions& opts, AAHResult& res) {
    const int N = opts.n_sites;
    const double beta = (opts.beta == 0.0) ? GOLDEN_RATIO : opts.beta;

    std::vector<double> H = build_aah(opts.t, opts.V, beta, opts.phi, N);
    // Jacobi: H is overwritten with eigenvectors after call
    std::vector<double> evals = jacobi_real_sym(H, N);
    // H now holds eigenvectors (columns)

    // Sort by eigenvalue
    std::vector<int> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return evals[a] < evals[b]; });

    res.eigenvalues.resize(N);
    std::vector<double> evecs_sorted(N * N);
    for (int i = 0; i < N; ++i) {
        res.eigenvalues[i] = evals[idx[i]];
        for (int r = 0; r < N; ++r)
            evecs_sorted[r*N+i] = H[r*N+idx[i]];
    }

    res.ipr = compute_ipr_from_evecs(evecs_sorted, N);
    res.mean_ipr = 0.0;
    for (double v : res.ipr) res.mean_ipr += v;
    res.mean_ipr /= N;

    // Analytic Lyapunov exponent (Aubry-André self-duality result)
    const double ratio = opts.V / opts.t;
    res.lyapunov = (ratio > 1.0) ? std::log(ratio) : 0.0;

    // Phase label
    const double eps = 0.05;
    if (std::abs(opts.V - opts.t) < eps * opts.t)
        res.phase_label = "Critical (V=t)";
    else if (opts.V < opts.t)
        res.phase_label = "Metal (V<t)";
    else
        res.phase_label = "Insulator (V>t)";
}

// ── Phase diagram: eigenvalues vs V sweep ─────────────────────────────────────
static void compute_phase(const AAHOptions& opts, AAHResult& res) {
    const int N    = opts.n_sites;
    const int nV   = opts.n_V;
    const double beta = (opts.beta == 0.0) ? GOLDEN_RATIO : opts.beta;

    res.pd_V.resize(nV);
    res.pd_E.resize(nV * N);
    res.pd_ipr_mean.resize(nV);

    for (int vi = 0; vi < nV; ++vi) {
        const double Vv = (nV > 1) ? opts.V_max * vi / (nV - 1) : opts.V_max;
        res.pd_V[vi] = Vv;

        std::vector<double> H = build_aah(opts.t, Vv, beta, opts.phi, N);
        std::vector<double> evals = jacobi_real_sym(H, N);
        // H now holds eigenvectors
        const auto ipr = compute_ipr_from_evecs(H, N);

        std::sort(evals.begin(), evals.end());
        for (int i = 0; i < N; ++i)
            res.pd_E[vi*N+i] = evals[i];

        double s = 0.0;
        for (double v : ipr) s += v;
        res.pd_ipr_mean[vi] = s / N;
    }
}

// ── Dispatcher ───────────────────────────────────────────────────────────────
AAHResult run_aah(const AAHOptions& opts) {
    if (opts.n_sites < 4)  throw std::runtime_error("AAH: n_sites must be >= 4");
    if (opts.n_V < 1)      throw std::runtime_error("AAH: n_V must be >= 1");
    if (opts.t <= 0.0)     throw std::runtime_error("AAH: t must be positive");
    if (opts.V < 0.0)      throw std::runtime_error("AAH: V must be non-negative");

    const std::string& sc = opts.scenario;
    if (sc != "spectrum" && sc != "phase" && sc != "both")
        throw std::runtime_error(
            "AAH: unknown scenario \"" + sc +
            "\"; valid values: spectrum, phase, both");

    auto t0 = std::chrono::steady_clock::now();

    AAHResult res;
    res.scenario   = sc;
    res.t          = opts.t;
    res.V          = opts.V;
    res.beta_used  = (opts.beta == 0.0) ? GOLDEN_RATIO : opts.beta;

    if (sc == "spectrum" || sc == "both") compute_spectrum(opts, res);
    if (sc == "phase"    || sc == "both") compute_phase(opts, res);

    // Ensure phase_label set even when sc=="phase"
    if (res.phase_label.empty()) {
        const double ratio = opts.V / opts.t;
        const double eps   = 0.05;
        if (std::abs(ratio - 1.0) < eps)
            res.phase_label = "Critical (V=t)";
        else if (ratio < 1.0)
            res.phase_label = "Metal (V<t)";
        else
            res.phase_label = "Insulator (V>t)";
        res.lyapunov = (ratio > 1.0) ? std::log(ratio) : 0.0;
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.message = "t=" + std::to_string(opts.t) +
                  " V=" + std::to_string(opts.V) +
                  " beta=" + std::to_string(res.beta_used) +
                  " N=" + std::to_string(opts.n_sites) +
                  " phase=" + res.phase_label;
    res.ok = true;
    return res;
}

}  // namespace qx
