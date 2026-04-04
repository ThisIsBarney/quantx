// src/wilson_loop.cpp -- M36: Wilson Loop & Wannier Charge Centers
//
// Computes the Wilson loop spectrum (Wannier Charge Centers) for the
// Bernevig-Hughes-Zhang (BHZ) model — the canonical 2D Z₂ topological insulator
// describing HgTe/CdTe quantum wells.
//
// BHZ model (4×4, two-orbital, spin-up/down):
//   H(k) = ε(k)I₄ + d_a(k) Γ_a    (a = 1..5, no sum over index 0 = identity)
//   ε(k) = C - D(kx²+ky²)          [particle-hole asymmetry; set C=D=0 for simplicity]
//   d₁ = A kx,   d₂ = A ky,   d₃ = 0,   d₄ = 0
//   d₅ = M - B(kx² + ky²)           [M controls band inversion]
//
//   Γ₁ = σx ⊗ σx,  Γ₂ = σx ⊗ σy,  Γ₃ = σx ⊗ σz,  Γ₄ = σy ⊗ I,  Γ₅ = σz ⊗ I
//
// (On the lattice: kx → sin kx, ky → sin ky, kx² → 2(1-cos kx), etc.)
//
// Wilson loop (at fixed ky, along kx):
//   W(ky) = Π_{kx} P(kx, ky)   [product of overlap matrices]
//   P_mn(k) = ⟨u_m(k)|u_n(k+Δkx)⟩  [2×2 overlap of occupied bands]
//
// Wannier Charge Centers (WCC):
//   θ_i(ky) = -arg(λ_i(W(ky))) / (2π)   in [0, 1)
//
// Z₂ invariant: count of WCC crossings of θ = 0.5 as ky sweeps [0, π].
//   Odd number → Z₂ = 1 (topological);  Even → Z₂ = 0 (trivial).
//
// Phase diagram: gap closes at M=0 and M=4B (lattice); topological when 0 < M/B < 4.
//   Typical values: A=B=1. Topological: 0 < M < 4 (taking B=1).

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
// A.  BHZ 4×4 Hamiltonian (lattice regularised)
//
//   ε(k)  = C - 2D(2 - cos kx - cos ky)    [particle-hole asymmetry]
//   d₁    = A sin kx
//   d₂    = A sin ky
//   d₅    = M - 2B(2 - cos kx - cos ky)    [band inversion: gap closes at M=0, M=4B]
//   (d₃ = d₄ = 0 for the minimal BHZ model)
//
//   H = ε I₄ + d₁Γ₁ + d₂Γ₂ + d₅Γ₅
//
//   Γ₁ = σx⊗σx, Γ₂ = σx⊗σy, Γ₅ = σz⊗I
//   H = [[ (ε+d₅)I₂ ,  d₁σx + d₂σy ],
//        [ d₁σx - d₂σy, (ε-d₅)I₂  ]]
//   Expanding explicitly (row-major 4×4):
//    H[0][0]=ε+d₅, H[0][1]=0,        H[0][2]=d₁,      H[0][3]=-i d₂
//    H[1][0]=0,    H[1][1]=ε+d₅,     H[1][2]=i d₂,    H[1][3]=d₁   (wait, need to recheck)
//
// Let me be careful. σx⊗σx = [[0,σx],[σx,0]], σx⊗σy = [[0,-iσy... ]]
// Use direct tensor product:
//   σx = [[0,1],[1,0]], σy = [[0,-i],[i,0]], σz = [[1,0],[0,-1]]
//   I₂ = [[1,0],[0,1]]
//
//   Γ₁ = σx⊗σx = [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]
//   Γ₂ = σx⊗σy = [[0,0,0,-i],[0,0,i,0],[0,-i,0,0],[i,0,0,0]]
//   Γ₅ = σz⊗I  = [[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]]
//
//   H = ε I₄ + d₁ Γ₁ + d₂ Γ₂ + d₅ Γ₅
//   H[0] = (ε+d₅, 0,     0,         d₁-i*d₂)
//   H[1] = (0,     ε+d₅,  d₁+i*d₂,  0       )
//   H[2] = (0,     d₁-i*d₂, ε-d₅,  0       )  ← wait Γ₂[2][1]=-i, Γ₁[2][1]=1
//   Let me just compute row by row.
// ─────────────────────────────────────────────────────────────────────────────

static void bhz_ham(double kx, double ky,
                    double M, double A, double B, double C, double D,
                    cx H[16]) {
    double eps = C - 2.0*D*(2.0 - std::cos(kx) - std::cos(ky));
    double d1  = A * std::sin(kx);
    double d2  = A * std::sin(ky);
    double d5  = M - 2.0*B*(2.0 - std::cos(kx) - std::cos(ky));

    // Γ₁ = σx⊗σx:  only off-diag corners
    //   (0,3)=1, (1,2)=1, (2,1)=1, (3,0)=1
    // Γ₂ = σx⊗σy:  (0,3)=-i, (1,2)=i, (2,1)=-i, (3,0)=i
    // Γ₅ = σz⊗I:   diag (1,1,-1,-1)
    // H = ε I + d1 Γ1 + d2 Γ2 + d5 Γ5
    //
    // H[0][0] = ε + d5
    // H[0][1] = 0
    // H[0][2] = 0
    // H[0][3] = d1 * 1 + d2 * (-i) = d1 - i*d2
    // H[1][0] = 0
    // H[1][1] = ε + d5
    // H[1][2] = d1 * 1 + d2 * (i) = d1 + i*d2
    // H[1][3] = 0
    // H[2][0] = 0
    // H[2][1] = d1 * 1 + d2 * (-i) = d1 - i*d2
    // H[2][2] = ε - d5
    // H[2][3] = 0
    // H[3][0] = d1 * 1 + d2 * (i) = d1 + i*d2
    // H[3][1] = 0
    // H[3][2] = 0
    // H[3][3] = ε - d5

    const cx ep = eps;
    const cx id2{0.0, d2};

    H[0]  = ep + d5;      H[1]  = 0;
    H[2]  = 0;            H[3]  = cx{d1} - id2;

    H[4]  = 0;            H[5]  = ep + d5;
    H[6]  = cx{d1} + id2; H[7]  = 0;

    H[8]  = 0;            H[9]  = cx{d1} - id2;
    H[10] = ep - d5;      H[11] = 0;

    H[12] = cx{d1} + id2; H[13] = 0;
    H[14] = 0;            H[15] = ep - d5;
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  Diagonalise 4×4 Hermitian matrix → lower 2 eigenvectors
//     Uses explicit 4×4 eigensolver via Jacobi iterations.
// ─────────────────────────────────────────────────────────────────────────────

// Simple Jacobi-based eigensolver for 4×4 Hermitian matrices.
// Returns eigenvectors as columns of a 4×4 unitary matrix (row-major 16 elements).
// Eigenvalues are sorted in ascending order.

static void jacobi4(const cx H_in[16], double eigvals[4], cx eigvecs[16]) {
    // Copy to working array (Hermitian, stored as 4×4 complex)
    // Use Gram-Schmidt orthogonalization on H's columns as starting guess, etc.
    // For a production engine we'd use LAPACK; here we use power iteration / Householder.
    //
    // Since BHZ has only 3 non-zero "d" components and small matrix, use analytic structure:
    // The Hamiltonian H(k) = ε I + d1 Γ1 + d2 Γ2 + d5 Γ5 commutes with time-reversal T.
    // We use the block structure directly.
    //
    // H in block form (swapping rows/cols 1↔2 to reveal block structure):
    // With basis ordering (|E+↑⟩, |E-↑⟩, |E+↓⟩, |E-↓⟩), but our standard BHZ is already 4×4.
    //
    // Use Jacobi eigenvalue algorithm:
    static const int N = 4;
    // Working matrix (real part and imaginary part separated for Jacobi)
    // Actually, implement a simple power-method based approach using 4×4 complex Hermitian.
    // We'll do 50 sweeps of Jacobi rotations.

    // Copy H_in to A
    cx A[16];
    for (int i = 0; i < 16; ++i) A[i] = H_in[i];

    // V = identity
    for (int i = 0; i < 16; ++i) eigvecs[i] = (i / 4 == i % 4) ? cx{1.0} : cx{0.0};

    // Jacobi iterations — use unitary Jacobi (complex Jacobi)
    for (int sweep = 0; sweep < 30; ++sweep) {
        for (int p = 0; p < N-1; ++p) {
            for (int q = p+1; q < N; ++q) {
                cx apq = A[p*N+q];
                if (std::abs(apq) < 1e-14) continue;
                double app = A[p*N+p].real();
                double aqq = A[q*N+q].real();
                double tau = (aqq - app) / (2.0 * std::abs(apq));
                double t   = (tau >= 0) ?  1.0 / (tau + std::sqrt(1.0 + tau*tau))
                                         : -1.0 / (-tau + std::sqrt(1.0 + tau*tau));
                double c   = 1.0 / std::sqrt(1.0 + t*t);
                double s   = c * t;
                // Phase of apq
                cx phase = apq / std::abs(apq);

                // Apply rotation: A' = G† A G  where G is the Givens matrix
                // G_{pp}=c, G_{qq}=c, G_{pq}=s*conj(phase), G_{qp}=-s*phase
                for (int r = 0; r < N; ++r) {
                    cx arp = A[r*N+p], arq = A[r*N+q];
                    A[r*N+p] =  cx{c}*arp - cx{s}*std::conj(phase)*arq;
                    A[r*N+q] =  cx{s}*phase*arp + cx{c}*arq;
                }
                for (int r = 0; r < N; ++r) {
                    cx apr = A[p*N+r], aqr = A[q*N+r];
                    A[p*N+r] =  cx{c}*apr - cx{s}*phase*aqr;
                    A[q*N+r] =  cx{s}*std::conj(phase)*apr + cx{c}*aqr;
                }
                // Accumulate in eigvecs
                for (int r = 0; r < N; ++r) {
                    cx vrp = eigvecs[r*N+p], vrq = eigvecs[r*N+q];
                    eigvecs[r*N+p] =  cx{c}*vrp - cx{s}*std::conj(phase)*vrq;
                    eigvecs[r*N+q] =  cx{s}*phase*vrp + cx{c}*vrq;
                }
            }
        }
    }
    for (int i = 0; i < N; ++i) eigvals[i] = A[i*N+i].real();

    // Sort by eigenvalue (insertion sort, permute eigvecs columns)
    for (int i = 1; i < N; ++i) {
        double key = eigvals[i];
        cx col[N];
        for (int r = 0; r < N; ++r) col[r] = eigvecs[r*N+i];
        int j = i - 1;
        while (j >= 0 && eigvals[j] > key) {
            eigvals[j+1] = eigvals[j];
            for (int r = 0; r < N; ++r) eigvecs[r*N+j+1] = eigvecs[r*N+j];
            --j;
        }
        eigvals[j+1] = key;
        for (int r = 0; r < N; ++r) eigvecs[r*N+j+1] = col[r];
    }
}

// Extract lower 2 eigenvectors: Q is 4×2 (row*2+col)
static void lower_evecs(double kx, double ky,
                         double M, double A, double B, double C, double D,
                         cx Q[8]) {
    cx H[16];
    bhz_ham(kx, ky, M, A, B, C, D, H);
    double evals[4];
    cx evecs[16];
    jacobi4(H, evals, evecs);
    // Lower 2 eigenvectors = columns 0 and 1
    for (int row = 0; row < 4; ++row) {
        Q[row*2+0] = evecs[row*4+0];
        Q[row*2+1] = evecs[row*4+1];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  Wilson loop W(ky) = Π_{kx steps} M(kx, ky)
//     M = Q(kx)† Q(kx + Δkx)   [2×2 overlap, NOT normalized]
//     W is accumulated as a full 2×2 complex matrix.
//     WCC = -arg(eig(W)) / (2π)  mapped to [0, 1)
// ─────────────────────────────────────────────────────────────────────────────

// 2×2 matrix multiply (row-major, stored as cx[4])
static std::array<cx,4> m2mul(const std::array<cx,4>& A, const std::array<cx,4>& B) {
    return { A[0]*B[0]+A[1]*B[2], A[0]*B[1]+A[1]*B[3],
             A[2]*B[0]+A[3]*B[2], A[2]*B[1]+A[3]*B[3] };
}

// Eigenvalues of a 2×2 matrix via characteristic polynomial
// λ = (Tr ± sqrt(Tr² - 4 Det)) / 2
static void eig2(const std::array<cx,4>& M, cx evals[2]) {
    cx tr  = M[0] + M[3];
    cx det = M[0]*M[3] - M[1]*M[2];
    cx disc = std::sqrt(tr*tr - cx{4.0}*det);
    evals[0] = (tr - disc) * 0.5;
    evals[1] = (tr + disc) * 0.5;
}

// Compute Wilson loop WCC for a single ky value.
// Returns 2 WCC values in [0, 1).
static std::array<double,2> wilson_wcc(double ky, int n_kx,
                                        double M, double A, double B,
                                        double C, double D) {
    // Accumulate W = Π overlap matrices Q(kx)† Q(kx+Δkx)
    std::array<cx,4> W = {cx{1},0, 0,cx{1}};  // identity 2×2

    for (int ix = 0; ix < n_kx; ++ix) {
        double kx0 = 2.0*M_PI * ix / n_kx;
        double kx1 = 2.0*M_PI * (ix+1) / n_kx;  // periodic: kx1 wraps at n_kx

        cx Q0[8], Q1[8];
        lower_evecs(kx0, ky, M, A, B, C, D, Q0);
        lower_evecs(kx1, ky, M, A, B, C, D, Q1);

        // Overlap M = Q0† Q1  (2×2)
        std::array<cx,4> Ov{};
        for (int row = 0; row < 4; ++row)
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    Ov[i*2+j] += std::conj(Q0[row*2+i]) * Q1[row*2+j];

        W = m2mul(W, Ov);
    }

    // Eigenvalues of W → phases → WCC
    cx evals[2];
    eig2(W, evals);
    std::array<double,2> wcc;
    for (int i = 0; i < 2; ++i) {
        double theta = std::arg(evals[i]);   // in (-π, π]
        wcc[i] = -theta / (2.0 * M_PI);     // WCC in (-0.5, 0.5]
        if (wcc[i] < 0) wcc[i] += 1.0;      // shift to [0, 1)
    }
    // Sort for consistent plotting
    if (wcc[0] > wcc[1]) std::swap(wcc[0], wcc[1]);
    return wcc;
}

// ─────────────────────────────────────────────────────────────────────────────
// D.  Z₂ invariant from WCC evolution
//     Sweep ky from 0 to π; count crossings of θ = 0.5 by the lower WCC.
//     Odd count → Z₂ = 1.
// ─────────────────────────────────────────────────────────────────────────────

static int z2_from_wcc(const std::vector<std::array<double,2>>& wcc_grid) {
    // Count sign changes of (wcc[0] - 0.5) across ky in [0, π]
    // (Only need half BZ for TR-invariant system)
    int crossings = 0;
    int n = static_cast<int>(wcc_grid.size());
    for (int i = 0; i < n - 1; ++i) {
        double w0 = wcc_grid[i][0]   - 0.5;
        double w1 = wcc_grid[i+1][0] - 0.5;
        if (w0 * w1 < 0) ++crossings;
    }
    return crossings % 2;
}

// ─────────────────────────────────────────────────────────────────────────────
// E.  Energy gap at high-symmetry points (for gap diagnostic)
// ─────────────────────────────────────────────────────────────────────────────

static double gap_at(double kx, double ky,
                     double M, double A, double B, double C, double D) {
    cx H[16];
    bhz_ham(kx, ky, M, A, B, C, D, H);
    double evals[4]; cx evecs[16];
    jacobi4(H, evals, evecs);
    // Gap = E2 - E1 (between upper and lower pair)
    return std::abs(evals[2] - evals[1]);
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

WilsonLoopResult run_wilson_loop(const WilsonLoopOptions& opts) {
    auto t0 = std::chrono::steady_clock::now();

    if (opts.n_kx < 4)
        throw std::runtime_error("WilsonLoop: n_kx must be >= 4");
    if (opts.n_ky < 4)
        throw std::runtime_error("WilsonLoop: n_ky must be >= 4");

    const double M = opts.M;
    const double A = opts.A;
    const double B = opts.B;
    const double C = opts.C;
    const double D = opts.D;

    WilsonLoopResult res;
    res.M = M; res.A = A; res.B = B;
    res.scenario = opts.scenario;

    if (opts.scenario == "wcc") {
        // ── Compute WCC spectrum over ky in [0, π] ──────────────────────────
        int n_ky = opts.n_ky;
        res.ky_vals.resize(n_ky);
        res.wcc1.resize(n_ky);
        res.wcc2.resize(n_ky);

        std::vector<std::array<double,2>> wcc_grid(n_ky);
        for (int iy = 0; iy < n_ky; ++iy) {
            double ky = M_PI * iy / (n_ky - 1);   // 0 to π
            res.ky_vals[iy] = ky;
            wcc_grid[iy] = wilson_wcc(ky, opts.n_kx, M, A, B, C, D);
            res.wcc1[iy] = wcc_grid[iy][0];
            res.wcc2[iy] = wcc_grid[iy][1];
        }
        res.z2 = z2_from_wcc(wcc_grid);
        res.phase_label = (res.z2 == 1) ? "Topological (Z\342\202\202=1)"
                                        : "Trivial (Z\342\202\202=0)";
        res.gap_gamma = gap_at(0.0, 0.0, M, A, B, C, D);
        res.gap_m_pt  = gap_at(M_PI, M_PI, M, A, B, C, D);
        res.message   = "M=" + std::to_string(M) + ", Z2=" + std::to_string(res.z2);

    } else if (opts.scenario == "phase_diagram") {
        // ── Sweep M, compute Z₂ at each point ───────────────────────────────
        if (opts.sweep_param != "M")
            throw std::runtime_error(
                "WilsonLoop: unsupported sweep_param '" + opts.sweep_param +
                "' — only 'M' is supported for phase_diagram");
        int n = opts.n_sweep;
        if (n < 2) throw std::runtime_error("WilsonLoop: n_sweep must be >= 2");

        res.sweep_vals.resize(n);
        res.sweep_z2.resize(n);
        res.sweep_phase.resize(n);

        // Use fewer ky points for the sweep to keep it tractable
        int n_ky_sweep = std::max(8, opts.n_ky / 4);

        for (int k = 0; k < n; ++k) {
            double Mp = opts.param_min +
                        (opts.param_max - opts.param_min) * k / (n - 1);
            res.sweep_vals[k] = Mp;

            std::vector<std::array<double,2>> wcc_g(n_ky_sweep);
            for (int iy = 0; iy < n_ky_sweep; ++iy) {
                double ky = M_PI * iy / (n_ky_sweep - 1);
                wcc_g[iy] = wilson_wcc(ky, opts.n_kx, Mp, A, B, C, D);
            }
            res.sweep_z2[k]    = z2_from_wcc(wcc_g);
            res.sweep_phase[k] = (res.sweep_z2[k] == 1) ? "Topological"
                                                         : "Trivial";
        }
        // Also compute full WCC at the nominal M for the page
        {
            int n_ky = opts.n_ky;
            res.ky_vals.resize(n_ky);
            res.wcc1.resize(n_ky);
            res.wcc2.resize(n_ky);
            std::vector<std::array<double,2>> wcc_grid(n_ky);
            for (int iy = 0; iy < n_ky; ++iy) {
                double ky = M_PI * iy / (n_ky - 1);
                res.ky_vals[iy] = ky;
                wcc_grid[iy] = wilson_wcc(ky, opts.n_kx, M, A, B, C, D);
                res.wcc1[iy] = wcc_grid[iy][0];
                res.wcc2[iy] = wcc_grid[iy][1];
            }
            res.z2 = z2_from_wcc(wcc_grid);
        }
        res.phase_label = (res.z2 == 1) ? "Topological (Z\342\202\202=1)"
                                        : "Trivial (Z\342\202\202=0)";
        res.gap_gamma = gap_at(0.0, 0.0, M, A, B, C, D);
        res.gap_m_pt  = gap_at(M_PI, M_PI, M, A, B, C, D);
        res.message   = "phase diagram: swept M in ["
                      + std::to_string(opts.param_min) + ", "
                      + std::to_string(opts.param_max) + "]";

    } else {
        throw std::runtime_error(
            "WilsonLoop: unknown scenario '" + opts.scenario +
            "' — use wcc | phase_diagram");
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
    res.ok = true;
    return res;
}

}  // namespace qx
