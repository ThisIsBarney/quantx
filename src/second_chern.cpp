// src/second_chern.cpp -- M35: Second Chern Number & 4D Quantum Hall Effect
//
// Computes the second Chern number C₂ for the Zhang-Hu 4D quantum Hall model
// (also called the 4D topological insulator) using the lattice gauge field
// (Fukui-Hatsugai-Suzuki) method extended to four dimensions.
//
// Model: H(k) = d₁(k)Γ₁ + d₂(k)Γ₂ + d₃(k)Γ₃ + d₄(k)Γ₄ + d₅(k)Γ₅
//   d₁ = sin k₁,  d₂ = sin k₂,  d₃ = sin k₃,  d₄ = sin k₄
//   d₅ = m + t(cos k₁ + cos k₂ + cos k₃ + cos k₄)
//
// Gamma matrices (5-component, 4×4, {Γ_a,Γ_b}=2δ_{ab}I₄):
//   Γ₁ = σ₁⊗σ₁,  Γ₂ = σ₁⊗σ₂,  Γ₃ = σ₁⊗σ₃,  Γ₄ = σ₂⊗I,  Γ₅ = σ₃⊗I
//
// Eigenvalues: ±|d|, each doubly degenerate.
// Lower projector: P = (I₄ - H/|d|) / 2
//
// Phase diagram (gap closes at m = 0, ±2, ±4):
//   |m| > 4:         C₂ = 0  (trivial)
//   2 < |m| < 4:    |C₂| = 1
//   0 < |m| < 2:    |C₂| = 3
//
// Second Chern number via Abelian lattice gauge field:
//   U_μ(k) = det( Q(k)† Q(k+e_μ) )            [scalar link]
//   f_μν(k) = Im log( U_μ(k) U_ν(k+eμ) conj(U_μ(k+eν)) conj(U_ν(k)) )
//   C₂ = round( (1/4π²) Σ_k (f₁₂f₃₄ - f₁₃f₂₄ + f₁₄f₂₃) )
//
// This Abelian approximation is exact for the Zhang-Hu model due to the
// U(1)×SU(2) structure of the occupied bands.

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
namespace {

using cx = std::complex<double>;

// ─────────────────────────────────────────────────────────────────────────────
// A.  4×4 complex Hamiltonian
//     H stored row-major: H[i*4+j] = H_{ij}
// ─────────────────────────────────────────────────────────────────────────────

// Explicit 4×4 Hamiltonian from the five real d-components.
// Γ₁=σ₁⊗σ₁, Γ₂=σ₁⊗σ₂, Γ₃=σ₁⊗σ₃, Γ₄=σ₂⊗I, Γ₅=σ₃⊗I
// H = [[d₅,    0,         d₃-id₄,  d₁-id₂],
//      [0,     d₅,        d₁+id₂, -d₃-id₄],
//      [d₃+id₄, d₁-id₂, -d₅,     0       ],
//      [d₁+id₂, -d₃+id₄, 0,      -d₅     ]]
static void build_ham(double d1, double d2, double d3, double d4, double d5,
                      cx H[16]) {
    const cx id1 = cx{0,d1}, id2 = cx{0,d2}, id3 = cx{0,d3}, id4 = cx{0,d4};
    H[0]  =  d5;        H[1]  =  0;
    H[2]  = d3 - id4;   H[3]  = d1 - id2;
    H[4]  =  0;         H[5]  =  d5;
    H[6]  = d1 + id2;   H[7]  = -(d3 + id4);
    H[8]  = d3 + id4;   H[9]  = d1 - id2;
    H[10] = -d5;        H[11] =  0;
    H[12] = d1 + id2;   H[13] = -(d3 - id4);
    H[14] =  0;         H[15] = -d5;
    (void)id1; (void)id2; (void)id3; (void)id4;
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  Extract 2 lower eigenvectors via projector + Gram-Schmidt
//     P_lower = (I₄ - H/r) / 2,  then Gram-Schmidt on columns of P.
//     Q[row*2 + col]:  4 rows, 2 columns.
// ─────────────────────────────────────────────────────────────────────────────

static void lower_basis(const cx H[16], double r, cx Q[8]) {
    // Build P = (I - H/r)/2
    cx P[16];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            P[i*4+j] = ((i == j ? 1.0 : 0.0) - H[i*4+j] / r) * 0.5;

    // Gram-Schmidt on the 4 columns of P, keeping first 2 independent ones
    cx basis[2][4] = {};
    int n_found = 0;
    for (int col = 0; col < 4 && n_found < 2; ++col) {
        cx v[4];
        for (int row = 0; row < 4; ++row) v[row] = P[row*4 + col];
        // Subtract projections onto already-found basis vectors
        for (int b = 0; b < n_found; ++b) {
            cx proj{0};
            for (int row = 0; row < 4; ++row)
                proj += std::conj(basis[b][row]) * v[row];
            for (int row = 0; row < 4; ++row)
                v[row] -= proj * basis[b][row];
        }
        double nrm = 0;
        for (int row = 0; row < 4; ++row) nrm += std::norm(v[row]);
        nrm = std::sqrt(nrm);
        if (nrm > 1e-10) {
            for (int row = 0; row < 4; ++row)
                basis[n_found][row] = v[row] / nrm;
            ++n_found;
        }
    }
    // Store column-major in Q: Q[row*2 + col]
    for (int row = 0; row < 4; ++row) {
        Q[row*2 + 0] = (n_found > 0) ? basis[0][row] : cx{0};
        Q[row*2 + 1] = (n_found > 1) ? basis[1][row] : cx{0};
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  Scalar link determinant
//     U_μ(k) = det( Q(k)† Q(k+eμ) )  — normalised to unit circle
// ─────────────────────────────────────────────────────────────────────────────

// 2×2 overlap matrix M = Q0† Q1, return det M
static cx link_det(const cx Q0[8], const cx Q1[8]) {
    // M[i][j] = sum_row conj(Q0[row*2+i]) * Q1[row*2+j]
    cx M[4] = {};
    for (int row = 0; row < 4; ++row) {
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                M[i*2+j] += std::conj(Q0[row*2+i]) * Q1[row*2+j];
    }
    return M[0]*M[3] - M[1]*M[2];  // det of 2×2
}

// ─────────────────────────────────────────────────────────────────────────────
// D.  Build k-grid and eigenvectors
//     Grid: [0, 2π) with N_k points, periodic.
// ─────────────────────────────────────────────────────────────────────────────

// Precompute basis Q at all (N+1)^4 grid points (periodic copies needed).
// Q_grid[(i1*(N+1)^3 + i2*(N+1)^2 + i3*(N+1) + i4) * 8 + ...]

static std::vector<std::array<cx,8>>
build_grid(int N_k, double m, double t) {
    int N1 = N_k + 1;  // +1 for periodic copies
    long total = static_cast<long>(N1) * N1 * N1 * N1;
    std::vector<std::array<cx,8>> grid(total);

    for (int i1 = 0; i1 < N1; ++i1) {
        double k1 = 2.0 * M_PI * (i1 % N_k) / N_k;
        for (int i2 = 0; i2 < N1; ++i2) {
            double k2 = 2.0 * M_PI * (i2 % N_k) / N_k;
            for (int i3 = 0; i3 < N1; ++i3) {
                double k3 = 2.0 * M_PI * (i3 % N_k) / N_k;
                for (int i4 = 0; i4 < N1; ++i4) {
                    double k4 = 2.0 * M_PI * (i4 % N_k) / N_k;

                    double d1 = std::sin(k1);
                    double d2 = std::sin(k2);
                    double d3 = std::sin(k3);
                    double d4 = std::sin(k4);
                    double d5 = m + t * (std::cos(k1)+std::cos(k2)+
                                         std::cos(k3)+std::cos(k4));
                    double r  = std::sqrt(d1*d1+d2*d2+d3*d3+d4*d4+d5*d5);

                    long idx = ((long)i1*N1*N1*N1 + i2*N1*N1 + i3*N1 + i4);
                    if (r < 1e-14) {
                        // Gap closes — return identity-like basis
                        grid[idx] = {cx{1},0,0,0, 0,cx{1},0,0};
                    } else {
                        cx H[16];
                        build_ham(d1,d2,d3,d4,d5, H);
                        lower_basis(H, r, grid[idx].data());
                    }
                }
            }
        }
    }
    return grid;
}

// ─────────────────────────────────────────────────────────────────────────────
// E.  Second Chern number via Abelian 4D FHS
// ─────────────────────────────────────────────────────────────────────────────

static int compute_c2(int N_k, double m, double t) {
    int N1 = N_k + 1;
    auto grid = build_grid(N_k, m, t);

    auto idx = [&](int i1, int i2, int i3, int i4) -> long {
        return (long)i1*N1*N1*N1 + i2*N1*N1 + i3*N1 + i4;
    };

    // Precompute scalar link determinants  U_μ(k)  for all k and μ=0..3
    // ul[μ * N_k^4 + flat_idx]
    long N4 = (long)N_k * N_k * N_k * N_k;
    std::vector<cx> ul(4 * N4);

    auto flat4 = [&](int i1,int i2,int i3,int i4) -> long {
        return (long)i1*N_k*N_k*N_k + i2*N_k*N_k + i3*N_k + i4;
    };

    for (int i1 = 0; i1 < N_k; ++i1)
    for (int i2 = 0; i2 < N_k; ++i2)
    for (int i3 = 0; i3 < N_k; ++i3)
    for (int i4 = 0; i4 < N_k; ++i4) {
        long f = flat4(i1,i2,i3,i4);
        const cx* Q0 = grid[idx(i1,i2,i3,i4)].data();
        ul[0*N4 + f] = link_det(Q0, grid[idx(i1+1,i2,i3,i4)].data());
        ul[1*N4 + f] = link_det(Q0, grid[idx(i1,i2+1,i3,i4)].data());
        ul[2*N4 + f] = link_det(Q0, grid[idx(i1,i2,i3+1,i4)].data());
        ul[3*N4 + f] = link_det(Q0, grid[idx(i1,i2,i3,i4+1)].data());
    }

    // Compute plaquette fluxes and accumulate C₂
    // f_μν(k) = Im log( U_μ(k) U_ν(k+eμ) conj(U_μ(k+eν)) conj(U_ν(k)) )
    double c2_sum = 0.0;
    for (int i1 = 0; i1 < N_k; ++i1)
    for (int i2 = 0; i2 < N_k; ++i2)
    for (int i3 = 0; i3 < N_k; ++i3)
    for (int i4 = 0; i4 < N_k; ++i4) {
        long f   = flat4(i1,i2,i3,i4);
        long f12 = flat4((i1+1)%N_k, i2, i3, i4);
        long f13 = flat4(i1, (i2+1)%N_k, i3, i4);
        long f14 = flat4(i1, i2, (i3+1)%N_k, i4);
        long f23 = flat4(i1, i2, i3, (i4+1)%N_k);
        // μ=0(k₁),ν=1(k₂): flux12
        auto plaq = [&](int mu, int nu, long km, long kn, long kmn) -> double {
            cx p = ul[mu*N4+f] * ul[nu*N4+km]
                 * std::conj(ul[mu*N4+kn]) * std::conj(ul[nu*N4+f]);
            double r = std::abs(p);
            if (r < 1e-14) return 0.0;
            return std::arg(p / r);
        };
        double fl12 = plaq(0,1, f12, f13, flat4((i1+1)%N_k,(i2+1)%N_k,i3,i4));
        double fl34 = plaq(2,3, flat4(i1,i2,(i3+1)%N_k,i4),
                               flat4(i1,i2,i3,(i4+1)%N_k),
                               flat4(i1,i2,(i3+1)%N_k,(i4+1)%N_k));
        double fl13 = plaq(0,2, f12, f14, flat4((i1+1)%N_k,i2,(i3+1)%N_k,i4));
        double fl24 = plaq(1,3, f13, f23, flat4(i1,(i2+1)%N_k,i3,(i4+1)%N_k));
        double fl14 = plaq(0,3, f12, f23, flat4((i1+1)%N_k,i2,i3,(i4+1)%N_k));
        double fl23 = plaq(1,2, f13, f14, flat4(i1,(i2+1)%N_k,(i3+1)%N_k,i4));

        c2_sum += fl12*fl34 - fl13*fl24 + fl14*fl23;
    }

    return static_cast<int>(std::round(c2_sum / (4.0 * M_PI * M_PI)));
}

// ─────────────────────────────────────────────────────────────────────────────
// F.  Phase label helper
// ─────────────────────────────────────────────────────────────────────────────

static std::string phase_label_c2(int c2) {
    if (c2 == 0) return "Trivial (C\342\202\202=0)";
    return "4D QHE (C\342\202\202=" + std::to_string(c2) + ")";
}

}  // namespace (anonymous)

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

SecondChernResult run_second_chern(const SecondChernOptions& opts) {
    auto t0 = std::chrono::steady_clock::now();

    if (opts.n_k < 2)
        throw std::runtime_error("SecondChern: n_k must be >= 2");
    if (opts.n_k > 16)
        throw std::runtime_error("SecondChern: n_k must be <= 16 (4D grid limit)");

    SecondChernResult res;
    res.m        = opts.m;
    res.t        = opts.t;
    res.scenario = opts.scenario;

    if (opts.scenario == "invariant") {
        int c2 = compute_c2(opts.n_k, opts.m, opts.t);
        res.second_chern = c2;
        res.phase_label  = phase_label_c2(c2);
        res.message = "m=" + std::to_string(opts.m) +
                      ", C2=" + std::to_string(c2);

    } else if (opts.scenario == "phase_diagram") {
        int n = opts.n_sweep;
        if (n < 2) throw std::runtime_error("SecondChern: n_sweep must be >= 2");
        res.sweep_vals.resize(n);
        res.sweep_c2.resize(n);
        res.sweep_phase.resize(n);

        for (int k = 0; k < n; ++k) {
            double p = opts.param_min +
                       (opts.param_max - opts.param_min) * k / (n - 1);
            res.sweep_vals[k] = p;
            double m_k = (opts.sweep_param == "m") ? p : opts.m;
            double t_k = (opts.sweep_param == "t") ? p : opts.t;
            if (t_k == 0.0) { res.sweep_c2[k] = 0; }
            else { res.sweep_c2[k] = compute_c2(opts.n_k, m_k, t_k); }
            res.sweep_phase[k] = phase_label_c2(res.sweep_c2[k]);
        }
        if (n > 0) {
            res.second_chern = res.sweep_c2[n / 2];
            res.phase_label  = res.sweep_phase[n / 2];
        }
        res.message = "phase diagram: swept " + opts.sweep_param +
                      " in [" + std::to_string(opts.param_min) +
                      ", " + std::to_string(opts.param_max) + "]";

    } else {
        throw std::runtime_error(
            "SecondChern: unknown scenario '" + opts.scenario +
            "' — use invariant | phase_diagram");
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.ok = true;
    return res;
}

}  // namespace qx
