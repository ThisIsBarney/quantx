// src/kitaev_honeycomb.cpp  —  M42: Kitaev Honeycomb Model
//
// Exactly solvable spin-1/2 model on the honeycomb lattice:
//   H = -Jx Σ_{x} σ_i^x σ_j^x  -Jy Σ_{y} σ_i^y σ_j^y  -Jz Σ_{z} σ_i^z σ_j^z
//
// Solved via Majorana representation (Kitaev 2006).  In the zero-flux sector
// the 2-band Bloch Hamiltonian is:
//
//   H(k) = [[m(k),  i f(k) ],
//            [-i f*(k), -m(k)]]
//
//   f(k) = Jz + Jx e^{i k·a1} + Jy e^{i k·a2}
//          (a1 = (1,0),  a2 = (1/2, √3/2))
//
//   m(k) = 2κ(sin(k·a1) + sin(k·a2) + sin(k·(a2−a1)))
//          (NNN hopping induced by magnetic field h; κ ∝ hx hy hz / J²)
//
// Energy spectrum: E±(k) = ±√(|f(k)|² + m(k)²)
//
// Phase diagram on the (Jx,Jy,Jz) simplex (Jx+Jy+Jz=1):
//   A_x: Jx > Jy+Jz  (gapped, |gap| = Jx−Jy−Jz)
//   A_y: Jy > Jx+Jz  (gapped)
//   A_z: Jz > Jx+Jy  (gapped)
//   B:   triangle inequality met  (gapless Dirac; with κ≠0 → C=±1 topo SC)
//
// Chern number via FHS lattice gauge on the lower Bloch band.
//
// Scenarios: "spectrum" | "phase_diagram" | "chern" | "both"

#include "qx/engine.hpp"
#include "qx/types.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace qx {

using cx = std::complex<double>;

// ── Honeycomb lattice vectors ─────────────────────────────────────────────
// a1 = (1, 0),  a2 = (1/2, √3/2)
// Reciprocal vectors (b·a = 2π δ):
//   b1 = 2π(1, −1/√3),  b2 = 2π(0, 2/√3)
static constexpr double SQRT3 = 1.7320508075688772935;

// ── 2×2 Kitaev Hamiltonian at k=(kx,ky) ──────────────────────────────────
// Returns [f, m] where f = f(k), m = m(k).
static void kitaev_fk(double kx, double ky,
                       double Jx, double Jy, double Jz, double kappa,
                       cx& f, double& m) {
    // f(k) = Jz + Jx e^{i k·a1} + Jy e^{i k·a2}
    //   k·a1 = kx*1 + ky*0 = kx
    //   k·a2 = kx/2 + ky*√3/2
    const double dot_a1 = kx;
    const double dot_a2 = kx * 0.5 + ky * (SQRT3 * 0.5);
    f = cx{Jz} + cx{Jx} * cx{std::cos(dot_a1), std::sin(dot_a1)}
               + cx{Jy} * cx{std::cos(dot_a2), std::sin(dot_a2)};

    // m(k) = 2κ(sin(k·a1) + sin(k·a2) + sin(k·(a2−a1)))
    //   k·(a2−a1) = k·a2 − k·a1 = dot_a2 − dot_a1
    m = 2.0 * kappa * (std::sin(dot_a1) + std::sin(dot_a2)
                       + std::sin(dot_a2 - dot_a1));
}

// E+(k) = sqrt(|f|^2 + m^2)
static double e_plus(double kx, double ky,
                     double Jx, double Jy, double Jz, double kappa) {
    cx f; double m;
    kitaev_fk(kx, ky, Jx, Jy, Jz, kappa, f, m);
    return std::sqrt(std::norm(f) + m * m);
}

// ── Phase classification ──────────────────────────────────────────────────
static std::string classify_phase(double Jx, double Jy, double Jz, double kappa,
                                   double gap) {
    if (Jx >= Jy + Jz) return "A_x";
    if (Jy >= Jx + Jz) return "A_y";
    if (Jz >= Jx + Jy) return "A_z";
    // B phase
    if (kappa != 0.0 && gap > 1e-6)
        return "B (topo, C=+-1)";
    return "B (gapless)";
}

// ── BZ sampling on honeycomb reciprocal lattice ───────────────────────────
// k = (s/nk) b1 + (t/nk) b2,  s,t = 0..nk-1
// b1 = 2π(1, -1/√3),  b2 = 2π(0, 2/√3)
static void bz_point(int s, int t, int nk, double& kx, double& ky) {
    const double b1x = 2.0 * M_PI,        b1y = -2.0 * M_PI / SQRT3;
    const double b2x = 0.0,               b2y =  4.0 * M_PI / SQRT3;
    const double fs = static_cast<double>(s) / nk;
    const double ft = static_cast<double>(t) / nk;
    kx = fs * b1x + ft * b2x;
    ky = fs * b1y + ft * b2y;
}

// ── Band structure on BZ grid ─────────────────────────────────────────────
static void compute_spectrum(const KitaevHcOptions& opts, KitaevHcResult& res) {
    const int nk = opts.n_k;
    const int N  = nk * nk;
    res.kx_bz.resize(N);  res.ky_bz.resize(N);
    res.E_plus.resize(N); res.E_minus.resize(N);

    for (int s = 0; s < nk; ++s) {
        for (int t = 0; t < nk; ++t) {
            double kx, ky;
            bz_point(s, t, nk, kx, ky);
            const double ep = e_plus(kx, ky, opts.Jx, opts.Jy, opts.Jz, opts.kappa);
            const int idx = s * nk + t;
            res.kx_bz[idx]  = kx;
            res.ky_bz[idx]  = ky;
            res.E_plus[idx]  =  ep;
            res.E_minus[idx] = -ep;
        }
    }
    res.gap = *std::min_element(res.E_plus.begin(), res.E_plus.end());
}

// ── Chern number via FHS lattice gauge ────────────────────────────────────
// Lower band eigenvector of H(k): u(k) = normalize([-if*, E+ + m]) (unnorm)
// Link U_μ(k) = ⟨u(k)|u(k+δμ)⟩ / |⟨u(k)|u(k+δμ)⟩|
static cx lower_evec_overlap(double kx1, double ky1, double kx2, double ky2,
                               double Jx, double Jy, double Jz, double kappa) {
    // Lower eigenvector of [[m, if],[-if*, -m]]:
    // For E- = -sqrt(|f|^2+m^2), eigvec = normalize([-if*, E-+m])
    // But E-+m may be zero → use upper component if denominator is tiny
    auto lower_vec = [&](double kx, double ky, cx& v0, cx& v1) {
        cx f; double m;
        kitaev_fk(kx, ky, Jx, Jy, Jz, kappa, f, m);
        const double ep = std::sqrt(std::norm(f) + m * m);
        // Eigenvector (v0, v1) for eigenvalue -ep:
        //   [[m, if],[-if*,-m]] [v0,v1]^T = -ep [v0,v1]^T
        //   (m+ep) v0 + if v1 = 0  → v1 = i(m+ep)/f * v0  (if |f|>0)
        // or use: v0 = -conj(f), v1 = (m+ep)*(-i) normalised
        const double denom = std::abs(f);
        if (denom > 1e-14) {
            v0 = -std::conj(f);          // −f*
            v1 = cx{0, -(m + ep)};       // −i(m+ep)
        } else {
            // f≈0: diagonal, eigvec of -m is (0,1) if m>0 else (1,0)
            v0 = (m >= 0) ? cx{0} : cx{1};
            v1 = (m >= 0) ? cx{1} : cx{0};
        }
        // normalise
        double norm = std::sqrt(std::norm(v0) + std::norm(v1));
        if (norm > 1e-14) { v0 /= norm; v1 /= norm; }
    };
    cx u0, u1, v0, v1;
    lower_vec(kx1, ky1, u0, u1);
    lower_vec(kx2, ky2, v0, v1);
    return std::conj(u0)*v0 + std::conj(u1)*v1;  // inner product ⟨u|v⟩
}

static int compute_chern(const KitaevHcOptions& opts) {
    const int nk = opts.n_k;
    const double b1x = 2.0 * M_PI, b1y = -2.0 * M_PI / SQRT3;
    const double b2x = 0.0,        b2y =  4.0 * M_PI / SQRT3;
    const double dk1x = b1x / nk, dk1y = b1y / nk;
    const double dk2x = b2x / nk, dk2y = b2y / nk;

    double F_sum = 0.0;
    for (int s = 0; s < nk; ++s) {
        for (int t = 0; t < nk; ++t) {
            const int sp = (s + 1) % nk, tp = (t + 1) % nk;
            double kx, ky, kxs, kys;
            bz_point(s,  t,  nk, kx,  ky);
            double kx_b1, ky_b1, kx_b2, ky_b2, kx_b12, ky_b12;
            bz_point(sp, t,  nk, kx_b1,  ky_b1);
            bz_point(s,  tp, nk, kx_b2,  ky_b2);
            bz_point(sp, tp, nk, kx_b12, ky_b12);
            (void)kxs; (void)kys;

            cx U1  = lower_evec_overlap(kx,     ky,     kx_b1,  ky_b1,  opts.Jx, opts.Jy, opts.Jz, opts.kappa);
            cx U2  = lower_evec_overlap(kx,     ky,     kx_b2,  ky_b2,  opts.Jx, opts.Jy, opts.Jz, opts.kappa);
            cx U1p = lower_evec_overlap(kx_b2,  ky_b2,  kx_b12, ky_b12, opts.Jx, opts.Jy, opts.Jz, opts.kappa);
            cx U2p = lower_evec_overlap(kx_b1,  ky_b1,  kx_b12, ky_b12, opts.Jx, opts.Jy, opts.Jz, opts.kappa);

            auto link = [](cx z) -> cx {
                double r = std::abs(z);
                return (r > 1e-12) ? z / r : cx{1};
            };
            cx F_plaq = link(U1) * link(U2p) * std::conj(link(U1p)) * std::conj(link(U2));
            F_sum += std::arg(F_plaq);
        }
    }
    return static_cast<int>(std::round(F_sum / (2.0 * M_PI)));
}

// ── Phase diagram on simplex ──────────────────────────────────────────────
static void compute_phase_diagram(const KitaevHcOptions& opts, KitaevHcResult& res) {
    const int n = opts.n_pd;
    // Parametrize the simplex: Jx = (n-i-j)/n, Jy = i/n, Jz = j/n for i+j<=n
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n - i; ++j) {
            const double Jx = static_cast<double>(n - i - j) / n;
            const double Jy = static_cast<double>(i) / n;
            const double Jz = static_cast<double>(j) / n;
            const double norm = Jx + Jy + Jz;
            if (norm < 1e-10) continue;

            // Gap at k=0 (analytic lower bound)
            double gap_analytic;
            if (Jx >= Jy + Jz) gap_analytic = Jx - Jy - Jz;
            else if (Jy >= Jx + Jz) gap_analytic = Jy - Jx - Jz;
            else if (Jz >= Jx + Jy) gap_analytic = Jz - Jx - Jy;
            else gap_analytic = 0.0;  // gapless

            const std::string ph = classify_phase(Jx, Jy, Jz, opts.kappa, gap_analytic);
            res.pd_Jx.push_back(Jx);
            res.pd_Jy.push_back(Jy);
            res.pd_Jz.push_back(Jz);
            res.pd_phase.push_back(ph);
            res.pd_gap.push_back(gap_analytic);
        }
    }
}

// ── Main dispatcher ───────────────────────────────────────────────────────
KitaevHcResult run_kitaev_honeycomb(const KitaevHcOptions& opts) {
    if (opts.n_k < 4)  throw std::runtime_error("Kitaev: n_k must be >= 4");
    if (opts.n_pd < 2) throw std::runtime_error("Kitaev: n_pd must be >= 2");
    if (opts.Jx < 0 || opts.Jy < 0 || opts.Jz < 0)
        throw std::runtime_error("Kitaev: Jx, Jy, Jz must be non-negative");

    const std::string& sc = opts.scenario;
    if (sc != "spectrum" && sc != "phase_diagram" && sc != "chern" && sc != "both")
        throw std::runtime_error(
            "Kitaev: unknown scenario \"" + sc +
            "\"; valid values: spectrum, phase_diagram, chern, both");

    auto t0 = std::chrono::steady_clock::now();

    KitaevHcResult res;
    res.scenario = sc;
    res.Jx = opts.Jx;  res.Jy = opts.Jy;  res.Jz = opts.Jz;  res.kappa = opts.kappa;

    // Compute gap analytically for phase label
    double gap_analytic = 0.0;
    if (opts.Jx >= opts.Jy + opts.Jz) gap_analytic = opts.Jx - opts.Jy - opts.Jz;
    else if (opts.Jy >= opts.Jx + opts.Jz) gap_analytic = opts.Jy - opts.Jx - opts.Jz;
    else if (opts.Jz >= opts.Jx + opts.Jy) gap_analytic = opts.Jz - opts.Jx - opts.Jy;

    if (sc == "spectrum" || sc == "both") {
        compute_spectrum(opts, res);
    } else {
        // Still need gap for phase label
        res.gap = gap_analytic;
    }

    res.chern = 0;
    if (sc == "chern" || sc == "both") {
        res.chern = compute_chern(opts);
    }

    res.phase_label = classify_phase(opts.Jx, opts.Jy, opts.Jz, opts.kappa, res.gap);

    if (sc == "phase_diagram" || sc == "both") {
        compute_phase_diagram(opts, res);
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.message = "Jx=" + std::to_string(opts.Jx) +
                  " Jy=" + std::to_string(opts.Jy) +
                  " Jz=" + std::to_string(opts.Jz) +
                  " kappa=" + std::to_string(opts.kappa) +
                  " phase=" + res.phase_label +
                  " gap=" + std::to_string(res.gap);
    res.ok = true;
    return res;
}

}  // namespace qx
