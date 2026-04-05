// src/haldane.cpp -- M38: Haldane Model / Quantum Anomalous Hall Effect
//
// 2-band tight-binding model on the honeycomb lattice with complex NNN hopping
// e^{iφ} (Haldane flux).  Realizes the quantum anomalous Hall effect — an
// integer Hall effect without Landau levels.
//
// Bloch Hamiltonian (2×2, row-major, h0 I₂ + hx σx + hy σy + hz σz):
//   NN bond vectors (A→B):  δ₁=(0,1), δ₂=(√3/2,-1/2), δ₃=(-√3/2,-1/2)
//   NNN vectors (A→A CCW):  b₁=(√3,0), b₂=(√3/2,3/2), b₃=(-√3/2,3/2)
//
//   h₀ = 2t₂cos(φ)·[cos(k·b₁)+cos(k·b₂)+cos(k·b₃)]
//   hx = t₁·Re[Σ e^{ik·δᵢ}]
//   hy = t₁·Im[Σ e^{ik·δᵢ}]
//   hz = M - 2t₂sin(φ)·[sin(k·b₁)+sin(k·b₂)+sin(k·b₃)]
//
// Chern number C₁ via Fukui-Hatsugai-Suzuki (FHS) lattice gauge formula on
// a uniform grid over the BZ parameterized by reciprocal vectors
//   G₁=(2π/√3, -2π/3),  G₂=(0, 4π/3).
//
// Topological phase: C₁=±1 when |M/(t₂)| < 3√3|sin(φ)|, C₁=0 otherwise.

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace qx {
namespace {

using cx = std::complex<double>;

// ─────────────────────────────────────────────────────────────────────────────
// A.  Haldane Bloch Hamiltonian
// ─────────────────────────────────────────────────────────────────────────────

struct HCoeffs {
    double h0, hx, hy, hz;
};

static HCoeffs haldane_coeffs(double kx, double ky,
                               double t1, double t2, double phi, double M_val) {
    const double s3 = std::sqrt(3.0);

    // NN vectors: δ₁=(0,1), δ₂=(√3/2,-1/2), δ₃=(-√3/2,-1/2)
    const double d1x = 0.0,     d1y = 1.0;
    const double d2x = s3/2.0,  d2y = -0.5;
    const double d3x = -s3/2.0, d3y = -0.5;

    // NNN vectors: b₁=(√3,0), b₂=(√3/2,3/2), b₃=(-√3/2,3/2)
    const double b1x = s3,     b1y = 0.0;
    const double b2x = s3/2.0, b2y = 1.5;
    const double b3x = -s3/2.0, b3y = 1.5;

    // NN sum: f(k) = Σ e^{ik·δᵢ}
    cx f = std::exp(cx(0, kx*d1x + ky*d1y))
         + std::exp(cx(0, kx*d2x + ky*d2y))
         + std::exp(cx(0, kx*d3x + ky*d3y));

    double hx = t1 * f.real();
    double hy = t1 * f.imag();

    // NNN sums
    double g_cos = std::cos(kx*b1x + ky*b1y)
                 + std::cos(kx*b2x + ky*b2y)
                 + std::cos(kx*b3x + ky*b3y);
    double g_sin = std::sin(kx*b1x + ky*b1y)
                 + std::sin(kx*b2x + ky*b2y)
                 + std::sin(kx*b3x + ky*b3y);

    double h0 = 2.0 * t2 * std::cos(phi) * g_cos;
    double hz = M_val - 2.0 * t2 * std::sin(phi) * g_sin;

    return {h0, hx, hy, hz};
}

// Build 2×2 Bloch Hamiltonian (row-major, complex):
//   H = [[h0+hz, hx-i*hy], [hx+i*hy, h0-hz]]
static void haldane_ham(double kx, double ky,
                        double t1, double t2, double phi, double M_val,
                        cx H[4]) {
    auto c = haldane_coeffs(kx, ky, t1, t2, phi, M_val);
    H[0] = cx{c.h0 + c.hz, 0.0};
    H[1] = cx{c.hx, -c.hy};
    H[2] = cx{c.hx, +c.hy};
    H[3] = cx{c.h0 - c.hz, 0.0};
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  Analytical lower eigenvector of 2×2 Hermitian matrix
//     H = [[A, B],[B*, C]] with A,C real.
//     Eigenvalues: E± = (A+C)/2 ± √(((A-C)/2)² + |B|²)
//     Lower eigenvector for E- :
//       if |B| > eps or A ≠ C:  q = normalize([-B*, E- - A])
//       else (diagonal, A≤C):   q = (1,0)
// ─────────────────────────────────────────────────────────────────────────────
static void lower_evec(double kx, double ky,
                       double t1, double t2, double phi, double M_val,
                       cx q[2]) {
    cx H[4];
    haldane_ham(kx, ky, t1, t2, phi, M_val, H);
    // H[0] = A (real), H[1] = B, H[2] = B*, H[3] = C (real)
    double A = H[0].real();
    double C = H[3].real();
    cx B = H[1];

    double half_diff = (A - C) * 0.5;
    double absB2 = std::norm(B);
    double R = std::sqrt(half_diff * half_diff + absB2);  // ≥ 0

    if (R < 1e-14) {
        // Degenerate (gap collapsed): return (1,0)
        q[0] = cx{1.0, 0.0};
        q[1] = cx{0.0, 0.0};
        return;
    }

    // E- = (A+C)/2 - R
    double E_minus = 0.5*(A + C) - R;

    // Lower eigenvector: q0 = -conj(B), q1 = E- - A
    cx q0 = -std::conj(B);
    cx q1 = cx{E_minus - A, 0.0};

    double norm = std::sqrt(std::norm(q0) + std::norm(q1));
    if (norm < 1e-14) {
        q[0] = cx{1.0, 0.0};
        q[1] = cx{0.0, 0.0};
        return;
    }
    q[0] = q0 / norm;
    q[1] = q1 / norm;
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  Phase label from Chern number
// ─────────────────────────────────────────────────────────────────────────────
static std::string phase_label_from_chern(int C1) {
    if (C1 > 0) {
        return "QAH (C\u2081=+" + std::to_string(C1) + ")";
    } else if (C1 < 0) {
        return "QAH (C\u2081=" + std::to_string(C1) + ")";
    }
    return "Trivial (C\u2081=0)";
}

// ─────────────────────────────────────────────────────────────────────────────
// D.  FHS Chern number computation
//     BZ grid: k(i,j) = (i/n_k)*G1 + (j/n_k)*G2
//     G1 = (2π/√3, -2π/3),  G2 = (0, 4π/3)
//     Link: U_μ(i,j) = <q(i,j)|q(i,j)+e_μ> / |...|
//     Field: F(i,j) = arg(U1(i,j)*U2(i+1,j)*conj(U1(i,j+1))*conj(U2(i,j)))
//     C1 = round(ΣF / (2π))
// ─────────────────────────────────────────────────────────────────────────────
static int compute_chern(double t1, double t2, double phi, double M_val, int n_k,
                         std::vector<double>& kx_bz,
                         std::vector<double>& ky_bz,
                         std::vector<double>& berry_curv) {
    const double s3 = std::sqrt(3.0);
    // Reciprocal vectors
    const double G1x = 2.0*M_PI / s3,  G1y = -2.0*M_PI / 3.0;
    const double G2x = 0.0,            G2y =  4.0*M_PI / 3.0;

    const int N = n_k;
    const int N2 = N * N;

    // Compute grid k-points and lower eigenvectors
    std::vector<double> kx_grid(N2), ky_grid(N2);
    std::vector<cx> q0_grid(N2), q1_grid(N2);  // q[0] and q[1] per grid point

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double si = static_cast<double>(i) / N;
            double sj = static_cast<double>(j) / N;
            double kx = si*G1x + sj*G2x;
            double ky = si*G1y + sj*G2y;
            int idx = i*N + j;
            kx_grid[idx] = kx;
            ky_grid[idx] = ky;
            cx q[2];
            lower_evec(kx, ky, t1, t2, phi, M_val, q);
            q0_grid[idx] = q[0];
            q1_grid[idx] = q[1];
        }
    }

    kx_bz.resize(N2);
    ky_bz.resize(N2);
    berry_curv.resize(N2);

    double F_sum = 0.0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int idx   = i*N + j;
            int ip    = ((i+1) % N);
            int jp    = ((j+1) % N);
            int idx_i1j  = ip*N + j;
            int idx_ij1  = i*N + jp;
            int idx_i1j1 = ip*N + jp;

            // Link U1(i,j) = <q(i,j) | q(i+1,j)>
            cx U1 = std::conj(q0_grid[idx]) * q0_grid[idx_i1j]
                  + std::conj(q1_grid[idx]) * q1_grid[idx_i1j];
            double aU1 = std::abs(U1);
            if (aU1 < 1e-14) aU1 = 1.0;
            U1 /= aU1;

            // Link U2(i,j) = <q(i,j) | q(i,j+1)>
            cx U2 = std::conj(q0_grid[idx]) * q0_grid[idx_ij1]
                  + std::conj(q1_grid[idx]) * q1_grid[idx_ij1];
            double aU2 = std::abs(U2);
            if (aU2 < 1e-14) aU2 = 1.0;
            U2 /= aU2;

            // Link U1(i,j+1) = <q(i,j+1) | q(i+1,j+1)>
            cx U1p = std::conj(q0_grid[idx_ij1]) * q0_grid[idx_i1j1]
                   + std::conj(q1_grid[idx_ij1]) * q1_grid[idx_i1j1];
            double aU1p = std::abs(U1p);
            if (aU1p < 1e-14) aU1p = 1.0;
            U1p /= aU1p;

            // Link U2(i+1,j) = <q(i+1,j) | q(i+1,j+1)>
            cx U2p = std::conj(q0_grid[idx_i1j]) * q0_grid[idx_i1j1]
                   + std::conj(q1_grid[idx_i1j]) * q1_grid[idx_i1j1];
            double aU2p = std::abs(U2p);
            if (aU2p < 1e-14) aU2p = 1.0;
            U2p /= aU2p;

            // F(i,j) = arg(U1 * U2p * conj(U1p) * conj(U2))
            cx prod = U1 * U2p * std::conj(U1p) * std::conj(U2);
            double F = std::arg(prod);

            berry_curv[idx] = F;
            kx_bz[idx] = kx_grid[idx];
            ky_bz[idx] = ky_grid[idx];
            F_sum += F;
        }
    }

    int C1 = static_cast<int>(std::round(F_sum / (2.0 * M_PI)));
    return C1;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────
HaldaneResult run_haldane(const HaldaneOptions& opts) {
    auto t_start = std::chrono::steady_clock::now();
    HaldaneResult res;

    // ── Validate ──────────────────────────────────────────────────────────
    if (opts.n_k < 4 || opts.n_k > 100) {
        throw std::runtime_error("haldane: n_k must be in [4, 100]");
    }
    if (opts.n_y < 4 || opts.n_y > 40) {
        throw std::runtime_error("haldane: n_y must be in [4, 40]");
    }
    if (opts.sweep_param != "M" && opts.sweep_param != "phi") {
        throw std::runtime_error("haldane: sweep_param must be \"M\" or \"phi\"");
    }

    res.t1      = opts.t1;
    res.t2      = opts.t2;
    res.phi     = opts.phi;
    res.M_val   = opts.M_val;
    res.n_k     = opts.n_k;
    res.scenario = opts.scenario;

    // ── Compute Chern number and Berry curvature map ───────────────────────
    res.chern = compute_chern(opts.t1, opts.t2, opts.phi, opts.M_val, opts.n_k,
                              res.kx_bz, res.ky_bz, res.berry_curv);
    res.phase_label = phase_label_from_chern(res.chern);

    // ── Phase diagram sweep ────────────────────────────────────────────────
    if (opts.n_sweep >= 2) {
        int n = opts.n_sweep;
        res.sweep_vals.resize(n);
        res.sweep_chern.resize(n);
        res.sweep_phase.resize(n);
        for (int s = 0; s < n; ++s) {
            double v = opts.param_min + s * (opts.param_max - opts.param_min) / (n - 1);
            res.sweep_vals[s] = v;

            double t1s = opts.t1, t2s = opts.t2, phis = opts.phi, Ms = opts.M_val;
            if (opts.sweep_param == "M") {
                Ms = v;
            } else {
                phis = v;
            }

            std::vector<double> kx_tmp, ky_tmp, bc_tmp;
            int C1s = compute_chern(t1s, t2s, phis, Ms, opts.n_k, kx_tmp, ky_tmp, bc_tmp);
            res.sweep_chern[s] = C1s;
            res.sweep_phase[s] = phase_label_from_chern(C1s);
        }
    }

    // ── Ribbon: not implemented — reject "spectrum" scenario explicitly ────
    if (opts.scenario == "spectrum") {
        throw std::runtime_error(
            "haldane: scenario \"spectrum\" (zigzag ribbon) is not yet implemented; "
            "use \"chern\" or \"phase_diagram\"");
    }
    res.n_ribbon_bands = 0;

    auto t_end = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Build summary message
    std::ostringstream msg;
    msg << "Haldane model: t1=" << opts.t1
        << " t2=" << opts.t2
        << " phi=" << opts.phi
        << " M=" << opts.M_val
        << " => C1=" << res.chern
        << " (" << res.phase_label << ")";
    res.message = msg.str();
    res.ok = true;
    return res;
}

}  // namespace qx
