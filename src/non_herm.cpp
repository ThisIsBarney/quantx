// src/non_herm.cpp  —  M45: Non-Hermitian Topology
//
// Two canonical non-Hermitian topological phenomena:
//
// 1. Hatano-Nelson skin effect (1D asymmetric hopping chain):
//    H = Σ_n J_R c†_{n+1}c_n + J_L c†_n c_{n+1},  J_R=J+γ/2, J_L=J-γ/2
//
//    Under PBC: E(k) = 2J cos(k) + iγ sin(k)
//      → traces an ellipse in the complex plane
//      → winding W = sign(γ) (integer, measured around E=0)
//
//    Under OBC: E_n = 2√(J_R J_L) cos(nπ/(N+1)),  n = 1,...,N
//      → all eigenvalues are real (skin effect collapses imaginary parts)
//      → eigenstates: |ψ_n(r)|² ∝ (J_R/J_L)^r sin²(nπr/(N+1))
//        — all N states pile up at the same boundary (non-Hermitian skin effect)
//
// 2. Exceptional point (2×2 NH Hamiltonian with off-diagonal gain-loss):
//    H(δ,γ) = [[δ, J+γ],[J-γ, -δ]]
//    E_± = ±√(δ² + J² - γ²)
//    EP when discriminant = 0: δ=0 and γ=J  (eigenvalues AND eigenvectors coalesce)
//    For δ=0: γ<J → real ±√(J²-γ²); γ>J → purely imaginary ±i√(γ²-J²)
//
// Scenarios: "skin" | "ep" | "both"

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace qx {

// ── Winding number of PBC spectrum around E=0 ─────────────────────────────────
// E(k) = 2J cos(k) + i*gamma*sin(k)  traces an ellipse; winding = sign(gamma).
// Computed numerically via phase accumulation.
// Special case: gamma=0 → Hermitian limit, E(k) is real, W=0.  Without this
// guard the contour passes through the origin (E=0 at k=π/2), producing a
// spurious ±2π jump and incorrectly rounding to W=±1.
static int skin_winding(double J, double gamma, int nk) {
    if (std::abs(gamma) < 1e-12) return 0;
    const double dkx = 2.0 * M_PI / nk;
    double total_phase = 0.0;
    double prev_re = 2.0*J;  // E(0) = 2J
    double prev_im = 0.0;
    for (int i = 1; i <= nk; ++i) {
        const double k  = i * dkx;
        const double re = 2.0 * J * std::cos(k);
        const double im = gamma * std::sin(k);
        // arg(curr / prev) = arg(curr * conj(prev))
        const double prod_re = re*prev_re + im*prev_im;
        const double prod_im = im*prev_re - re*prev_im;
        total_phase += std::atan2(prod_im, prod_re);
        prev_re = re; prev_im = im;
    }
    return static_cast<int>(std::round(total_phase / (2.0 * M_PI)));
}

// ── Skin effect computation ───────────────────────────────────────────────────
static void compute_skin(const NonHermOptions& opts, NonHermResult& res) {
    const double J     = opts.J;
    const double gam   = opts.gamma;
    const double J_R   = J + gam * 0.5;
    const double J_L   = J - gam * 0.5;
    const int    nk    = opts.n_k;
    const int    N     = opts.n_sites;

    // ─ PBC complex spectrum ─
    res.k_pbc.resize(nk);
    res.e_re_pbc.resize(nk);
    res.e_im_pbc.resize(nk);
    for (int i = 0; i < nk; ++i) {
        const double k  = (2.0 * M_PI * i) / nk - M_PI;
        res.k_pbc[i]    = k;
        res.e_re_pbc[i] = 2.0 * J * std::cos(k);
        res.e_im_pbc[i] = gam * std::sin(k);
    }

    // ─ OBC real spectrum: E_n = 2√(J_R J_L) cos(nπ/(N+1)) ─
    // Guard against J_R J_L < 0 (over-driven regime): use 0 for imaginary sqrt.
    const double E0 = (J_R * J_L > 0.0) ? 2.0 * std::sqrt(J_R * J_L) : 0.0;
    res.e_re_obc.resize(N);
    for (int n = 1; n <= N; ++n) {
        res.e_re_obc[n-1] = E0 * std::cos(static_cast<double>(n) * M_PI / (N + 1));
    }
    std::sort(res.e_re_obc.begin(), res.e_re_obc.end());

    // ─ Skin density: |ψ_r|² ∝ (J_R/J_L)^r sin²(θ_mid r), θ_mid for n=N/2 ─
    {
        const int n_mid = N / 2 + 1;
        const double theta = static_cast<double>(n_mid) * M_PI / (N + 1);
        const double log_ratio = (std::abs(J_L) > 1e-14)
            ? std::log(std::abs(J_R / J_L)) * (J_R >= 0 ? 1.0 : -1.0)
            : 30.0;  // effectively infinite skin at J_L → 0

        // Find max log(rho) for numerical stability
        double max_log = -1e300;
        for (int r = 1; r <= N; ++r) {
            const double s = std::sin(theta * static_cast<double>(r));
            if (std::abs(s) > 1e-10) {
                const double lv = static_cast<double>(r) * log_ratio
                                  + 2.0 * std::log(std::abs(s));
                if (lv > max_log) max_log = lv;
            }
        }
        if (max_log < -1e299) max_log = 0.0;

        std::vector<double> rho(N);
        double Z = 0.0;
        for (int r = 1; r <= N; ++r) {
            const double s = std::sin(theta * static_cast<double>(r));
            const double lv = static_cast<double>(r) * log_ratio
                              + 2.0 * std::log(std::max(1e-300, std::abs(s)))
                              - max_log;
            rho[r-1] = std::exp(lv);
            Z += rho[r-1];
        }
        if (Z < 1e-30) Z = 1.0;

        res.skin_site.resize(N);
        res.skin_density.resize(N);
        for (int r = 0; r < N; ++r) {
            res.skin_site[r]    = static_cast<double>(r + 1);
            res.skin_density[r] = rho[r] / Z;
        }
    }

    // ─ Winding number ─
    res.W = skin_winding(J, gam, nk);
}

// ── Winding-number sweep: W vs gamma ─────────────────────────────────────────
static void compute_winding_sweep(const NonHermOptions& opts, NonHermResult& res) {
    const int ng = opts.n_gamma;
    const double gmax = opts.gamma_max;
    res.sweep_gamma.resize(ng);
    res.sweep_W.resize(ng);
    for (int i = 0; i < ng; ++i) {
        const double gv = (ng > 1) ? gmax * i / (ng - 1) : gmax;
        res.sweep_gamma[i] = gv;
        res.sweep_W[i]     = skin_winding(opts.J, gv, opts.n_k);
    }
}

// ── Exceptional point: E(gamma) at delta=0 ────────────────────────────────────
// H = [[0, J+gamma],[J-gamma, 0]]  →  E = ±sqrt(J^2 - gamma^2)
// EP at gamma = J: both eigenvalues = 0, eigenvectors coalesce.
static void compute_ep(const NonHermOptions& opts, NonHermResult& res) {
    const int ng     = opts.n_gamma;
    const double gmax = opts.gamma_max;
    const double J    = opts.J;

    res.ep_gamma.resize(ng);
    res.ep_e_re_plus.resize(ng);  res.ep_e_im_plus.resize(ng);
    res.ep_e_re_minus.resize(ng); res.ep_e_im_minus.resize(ng);

    for (int i = 0; i < ng; ++i) {
        const double gv = (ng > 1) ? gmax * i / (ng - 1) : gmax;
        res.ep_gamma[i] = gv;
        // Full discriminant includes on-site detuning delta:
        // E = ±√(δ² + J² − γ²);  EP when δ² + J² = γ²
        const double disc = opts.delta*opts.delta + J*J - gv*gv;
        double re, im;
        if (disc >= 0.0) {
            // Below EP: real eigenvalues ±√(δ²+J²-γ²)
            re = std::sqrt(disc); im = 0.0;
        } else {
            // Above EP: purely imaginary ±i√(γ²-δ²-J²)
            re = 0.0; im = std::sqrt(-disc);
        }
        res.ep_e_re_plus[i]  =  re;  res.ep_e_im_plus[i]  =  im;
        res.ep_e_re_minus[i] = -re;  res.ep_e_im_minus[i] = -im;
    }
}

// ── Dispatcher ───────────────────────────────────────────────────────────────
NonHermResult run_non_herm(const NonHermOptions& opts) {
    if (opts.n_k < 4)      throw std::runtime_error("NonHerm: n_k must be >= 4");
    if (opts.n_sites < 4)  throw std::runtime_error("NonHerm: n_sites must be >= 4");
    if (opts.n_gamma < 1)  throw std::runtime_error("NonHerm: n_gamma must be >= 1");
    if (opts.J <= 0.0)     throw std::runtime_error("NonHerm: J must be positive");

    const std::string& sc = opts.scenario;
    if (sc != "skin" && sc != "ep" && sc != "both")
        throw std::runtime_error(
            "NonHerm: unknown scenario \"" + sc +
            "\"; valid values: skin, ep, both");

    auto t0 = std::chrono::steady_clock::now();

    NonHermResult res;
    res.scenario = sc;
    res.J     = opts.J;
    res.gamma = opts.gamma;
    res.delta = opts.delta;

    if (sc == "skin" || sc == "both") {
        compute_skin(opts, res);
        compute_winding_sweep(opts, res);
        if (res.W != 0)
            res.phase_label = "Skin (W=" + std::to_string(res.W) + ")";
        else
            res.phase_label = "Trivial (W=0)";
    }
    if (sc == "ep" || sc == "both") {
        compute_ep(opts, res);
        if (sc == "ep") {
            // Compute W for labelling when not already done
            res.W = skin_winding(opts.J, opts.gamma, opts.n_k);
            res.phase_label = (std::abs(opts.gamma) < opts.J)
                ? "Below EP (real spectrum)"
                : "Above EP (imaginary spectrum)";
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.message = "J=" + std::to_string(opts.J) +
                  " gamma=" + std::to_string(opts.gamma) +
                  " W=" + std::to_string(res.W) +
                  " phase=" + res.phase_label;
    res.ok = true;
    return res;
}

}  // namespace qx
