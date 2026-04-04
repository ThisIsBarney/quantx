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
// Second Chern number via non-Abelian 4D FHS (rank-2 subspace):
//   U_μ(k) = Q(k)† Q(k+e_μ) / sqrt(|det|)     [2×2 link matrix, |det|=1]
//   F_μν(k) = U_μ U_ν(k+eμ) U_μ(k+eν)† U_ν†  [2×2 plaquette]
//   f_μν(k) = log( F_μν(k) )                   [2×2 matrix log]
//   C₂ = round( (1/8π²) Σ_k Re Tr(f₁₂f₃₄ - f₁₃f₂₄ + f₁₄f₂₃) )
//
// The 2×2 trace captures both the U(1) determinant bundle and the SU(2)
// Berry curvature, correctly reproducing C₂ for any rank-2 subspace.

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
// C.  Non-Abelian 2×2 link matrices and matrix logarithm
//
// Full FHS for a rank-2 occupied subspace:
//   U_μ(k) = Q(k)† Q(k+eμ) / sqrt(|det(Q†Q')|)   [2×2, |det|=1]
//   F_μν   = U_μ(k) U_ν(k+eμ) U_μ(k+eν)† U_ν(k)† [2×2 plaquette]
//   f_μν   = log(F_μν)                              [matrix logarithm]
//   C₂     = round( (1/8π²) Σ_k Tr(f₁₂f₃₄ - f₁₃f₂₄ + f₁₄f₂₃) )
//
// The factor 8π² (not 4π²) comes from the 2×2 matrix trace picking up
// both the U(1) determinant bundle and the SU(2) Berry curvature.
// ─────────────────────────────────────────────────────────────────────────────

using M2 = std::array<cx, 4>;  // row-major 2×2: [a,b,c,d] = [[a,b],[c,d]]

static M2 m2mul(const M2& A, const M2& B) {
    return { A[0]*B[0]+A[1]*B[2], A[0]*B[1]+A[1]*B[3],
             A[2]*B[0]+A[3]*B[2], A[2]*B[1]+A[3]*B[3] };
}
static M2 m2dag(const M2& A) {
    return { std::conj(A[0]), std::conj(A[2]),
             std::conj(A[1]), std::conj(A[3]) };
}
static cx m2tr(const M2& A) { return A[0] + A[3]; }

// 2×2 matrix logarithm for an approximately unitary matrix F.
// Decomposes F = e^(iφ/2) W with W ∈ SU(2), computes log W analytically.
static M2 mat2_log(const M2& F) {
    cx det = F[0]*F[3] - F[1]*F[2];
    double phi = std::arg(det);
    // W = F * e^(-iφ/2)
    cx ei = std::exp(cx{0.0, -0.5 * phi});
    M2 W = { F[0]*ei, F[1]*ei, F[2]*ei, F[3]*ei };
    // Tr(W) = 2cos(α)
    double cos_a = std::clamp(0.5 * (W[0] + W[3]).real(), -1.0, 1.0);
    double alpha  = std::acos(cos_a);
    double c = (std::abs(std::sin(alpha)) > 1e-10)
               ? alpha / std::sin(alpha) : 1.0;
    // log(W) = c * (W - cos(α) I),  log(F) = log(W) + (iφ/2) I
    cx iph = cx{0.0, 0.5 * phi};
    return { cx{c}*(W[0]-cx{cos_a}) + iph,
             cx{c}* W[1],
             cx{c}* W[2],
             cx{c}*(W[3]-cx{cos_a}) + iph };
}

// 2×2 link matrix Q0† Q1, normalized so |det| = 1.
static M2 link_mat(const cx Q0[8], const cx Q1[8]) {
    M2 M{};
    for (int row = 0; row < 4; ++row)
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                M[i*2+j] += std::conj(Q0[row*2+i]) * Q1[row*2+j];
    cx d = M[0]*M[3] - M[1]*M[2];
    double r = std::abs(d);
    if (r > 1e-14) {
        double s = 1.0 / std::sqrt(r);
        for (auto& v : M) v *= s;
    }
    return M;
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
// E.  Second Chern number via non-Abelian 4D FHS
// ─────────────────────────────────────────────────────────────────────────────

static int compute_c2(int N_k, double m, double t) {
    int N1 = N_k + 1;
    auto grid = build_grid(N_k, m, t);

    auto idx = [&](int i1, int i2, int i3, int i4) -> long {
        return (long)i1*N1*N1*N1 + i2*N1*N1 + i3*N1 + i4;
    };

    // Precompute 2×2 link matrices U_μ(k) = Q(k)† Q(k+eμ) (normalized)
    // ul[μ * N_k^4 + flat_idx]
    long N4 = (long)N_k * N_k * N_k * N_k;
    std::vector<M2> ul(4 * N4);

    auto flat4 = [&](int i1,int i2,int i3,int i4) -> long {
        return (long)i1*N_k*N_k*N_k + i2*N_k*N_k + i3*N_k + i4;
    };

    for (int i1 = 0; i1 < N_k; ++i1)
    for (int i2 = 0; i2 < N_k; ++i2)
    for (int i3 = 0; i3 < N_k; ++i3)
    for (int i4 = 0; i4 < N_k; ++i4) {
        long f = flat4(i1,i2,i3,i4);
        const cx* Q0 = grid[idx(i1,i2,i3,i4)].data();
        ul[0*N4+f] = link_mat(Q0, grid[idx(i1+1,i2,i3,i4)].data());
        ul[1*N4+f] = link_mat(Q0, grid[idx(i1,i2+1,i3,i4)].data());
        ul[2*N4+f] = link_mat(Q0, grid[idx(i1,i2,i3+1,i4)].data());
        ul[3*N4+f] = link_mat(Q0, grid[idx(i1,i2,i3,i4+1)].data());
    }

    // Plaquette F_μν = U_μ(k) U_ν(k+eμ) U_μ(k+eν)† U_ν(k)†
    // Field strength f_μν = log(F_μν)
    // C₂ = round( (1/8π²) Σ_k Re Tr(f₁₂f₃₄ - f₁₃f₂₄ + f₁₄f₂₃) )
    double c2_sum = 0.0;

    for (int i1 = 0; i1 < N_k; ++i1)
    for (int i2 = 0; i2 < N_k; ++i2)
    for (int i3 = 0; i3 < N_k; ++i3)
    for (int i4 = 0; i4 < N_k; ++i4) {
        long k0  = flat4(i1, i2, i3, i4);
        // shifted flat indices (one step along each axis)
        long k1p = flat4((i1+1)%N_k, i2, i3, i4);
        long k2p = flat4(i1, (i2+1)%N_k, i3, i4);
        long k3p = flat4(i1, i2, (i3+1)%N_k, i4);
        long k4p = flat4(i1, i2, i3, (i4+1)%N_k);

        // F_μν(k) = U_μ(k) U_ν(k+eμ) U_μ(k+eν)† U_ν(k)†
        auto plaq = [&](int mu, int nu, long km, long kn) -> M2 {
            return m2mul(m2mul(m2mul(ul[mu*N4+k0], ul[nu*N4+km]),
                               m2dag(ul[mu*N4+kn])),
                         m2dag(ul[nu*N4+k0]));
        };

        M2 f12 = mat2_log(plaq(0, 1, k1p, k2p));
        M2 f34 = mat2_log(plaq(2, 3, k3p, k4p));
        M2 f13 = mat2_log(plaq(0, 2, k1p, k3p));
        M2 f24 = mat2_log(plaq(1, 3, k2p, k4p));
        M2 f14 = mat2_log(plaq(0, 3, k1p, k4p));
        M2 f23 = mat2_log(plaq(1, 2, k2p, k3p));

        c2_sum += (m2tr(m2mul(f12,f34)) - m2tr(m2mul(f13,f24))
                 + m2tr(m2mul(f14,f23))).real();
    }

    return static_cast<int>(std::round(c2_sum / (8.0 * M_PI * M_PI)));
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
        if (opts.sweep_param != "m" && opts.sweep_param != "t")
            throw std::runtime_error(
                "SecondChern: unsupported sweep_param '" + opts.sweep_param +
                "' — use m or t");
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
