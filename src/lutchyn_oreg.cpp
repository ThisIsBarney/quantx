// src/lutchyn_oreg.cpp  —  M41: Lutchyn-Oreg Majorana Nanowire
//
// 1D semiconductor nanowire (InAs/InSb-type) with:
//   - Rashba spin-orbit coupling α
//   - Proximity-induced s-wave SC gap Δ
//   - Zeeman field V_z (from applied B ∥ wire)
//
// BdG Hamiltonian (tight-binding lattice, lattice constant a=1):
//   H(k) = (2t(1−cos k) − μ) τ_z + α sin(k) σ_y τ_z + V_z σ_x + Δ τ_x
//
// τ = Nambu (particle-hole), σ = spin
// Basis: {e↑, e↓, h↓, −h↑}
//
// Topological transition: V_z,c = √(μ² + Δ²)  (at k=0)
//   V_z < V_z,c → trivial SC (Z₂ = +1)
//   V_z > V_z,c → topological SC (Z₂ = −1), Majorana zero modes at ends
//
// Z₂ invariant via Pfaffian at TRIM k=0,π:
//   Z₂ = sgn(Pf(H(0)) · Pf(H(π)))  (automatically ±1 from det argument)
//
// Scenarios
//   "bulk"         – BdG band structure (4 bands vs k) + Z₂ + gap
//   "wire"         – real-space open-BC wire spectrum + LDOS + Majorana oscillations
//   "phase_diagram"– Z₂(V_z, μ) heatmap
//   "both"         – bulk + wire

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

using cx  = std::complex<double>;
using Mat = std::vector<cx>;   // row-major NxN, H[i][j] = H[i*N+j]

// ── Jacobi eigensolver (Hermitian) ────────────────────────────────────────
static void jacobi_herm(int N, Mat A,
                         std::vector<double>& eigvals,
                         Mat& eigvecs) {
    eigvecs.assign(N * N, cx{0});
    for (int i = 0; i < N; ++i) eigvecs[i*N+i] = cx{1.0};

    const int max_sweeps = 30;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0;
        for (int p = 0; p < N; ++p)
            for (int q = p+1; q < N; ++q)
                off += std::norm(A[p*N+q]);
        if (off < 1e-26) break;

        for (int p = 0; p < N-1; ++p) {
            for (int q = p+1; q < N; ++q) {
                cx apq = A[p*N+q];
                double r_apq = std::abs(apq);
                if (r_apq < 1e-15) continue;
                double app = A[p*N+p].real();
                double aqq = A[q*N+q].real();
                double tau = (aqq - app) / (2.0 * r_apq);
                double t_jac = (tau >= 0) ?  1.0 / (tau + std::sqrt(1.0 + tau*tau))
                                           : -1.0 / (-tau + std::sqrt(1.0 + tau*tau));
                double c = 1.0 / std::sqrt(1.0 + t_jac*t_jac);
                double s = c * t_jac;
                cx ph = apq / r_apq;
                for (int r = 0; r < N; ++r) {
                    cx arp = A[r*N+p], arq = A[r*N+q];
                    A[r*N+p] = cx{c}*arp - cx{s}*std::conj(ph)*arq;
                    A[r*N+q] = cx{s}*ph*arp + cx{c}*arq;
                }
                for (int r = 0; r < N; ++r) {
                    cx apr = A[p*N+r], aqr = A[q*N+r];
                    A[p*N+r] = cx{c}*apr - cx{s}*ph*aqr;
                    A[q*N+r] = cx{s}*std::conj(ph)*apr + cx{c}*aqr;
                }
                for (int r = 0; r < N; ++r) {
                    cx vrp = eigvecs[r*N+p], vrq = eigvecs[r*N+q];
                    eigvecs[r*N+p] = cx{c}*vrp - cx{s}*std::conj(ph)*vrq;
                    eigvecs[r*N+q] = cx{s}*ph*vrp + cx{c}*vrq;
                }
            }
        }
    }
    eigvals.resize(N);
    for (int i = 0; i < N; ++i) eigvals[i] = A[i*N+i].real();
    for (int i = 1; i < N; ++i) {
        double key = eigvals[i];
        std::vector<cx> col(N);
        for (int r = 0; r < N; ++r) col[r] = eigvecs[r*N+i];
        int j = i - 1;
        while (j >= 0 && eigvals[j] > key) {
            eigvals[j+1] = eigvals[j];
            for (int r = 0; r < N; ++r) eigvecs[r*N+(j+1)] = eigvecs[r*N+j];
            --j;
        }
        eigvals[j+1] = key;
        for (int r = 0; r < N; ++r) eigvecs[r*N+(j+1)] = col[r];
    }
}

// ── BdG Hamiltonian (4×4 bulk, row-major) ────────────────────────────────
//
// Basis: {e↑, e↓, h↓, −h↑}
//   ε(k)  = 2t(1-cos k) - μ
//   h_e   = ε(k) I₂ + V_z σ_x + α sin(k) σ_y
//   H_BdG = [ h_e,  Δ·iσ_y ] = [ h_e,  [[0,Δ],[-Δ,0]] ]
//            [ −Δ·iσ_y, -h_eᵀ ]    bottom-right = −h_eᵀ
//
// h_eᵀ: transpose in spin space
//   σ_x^T = σ_x,  σ_y^T = -σ_y  → h_eᵀ = ε(k) I₂ + V_z σ_x - α sin(k) σ_y
//   -h_eᵀ = -ε(k) I₂ - V_z σ_x + α sin(k) σ_y
//
// Full 4×4 (rows/cols: e↑=0, e↓=1, h↓=2, -h↑=3):
//   [0,0] = ε       [0,1] = V_z - i·α·sk   [0,2] = 0    [0,3] = Δ
//   [1,0] = V_z+i·α·sk  [1,1] = ε           [1,2] = -Δ   [1,3] = 0
//   [2,0] = 0       [2,1] = -Δ              [2,2] = -ε   [2,3] = V_z + i·α·sk
//   [3,0] = Δ       [3,1] = 0               [3,2] = V_z - i·α·sk  [3,3] = -ε
//
// where ε = 2t(1-cos k) - μ,  sk = sin(k)
static void lo_ham(double k, double t, double alpha, double mu, double Delta,
                   double Vz, Mat& H) {
    H.assign(16, cx{0});
    const double eps = 2.0*t*(1.0 - std::cos(k)) - mu;
    const double sk  = std::sin(k);
    const cx     iask{0.0, alpha * sk};   // i α sin(k)

    // Top-left 2×2: h_e
    H[0*4+0] = eps;
    H[0*4+1] = Vz - iask;
    H[1*4+0] = Vz + iask;
    H[1*4+1] = eps;

    // Top-right 2×2: iΔσ_y = [[0,Δ],[-Δ,0]]
    H[0*4+3] =  Delta;
    H[1*4+2] = -Delta;

    // Bottom-left 2×2: −iΔσ_y = [[0,-Δ],[Δ,0]]
    H[2*4+1] = -Delta;
    H[3*4+0] =  Delta;

    // Bottom-right 2×2: -h_eᵀ = diag(-ε,-ε) + V_z σ_x* (σ_x^T=σ_x) - V_z σ_x... wait:
    // -h_eᵀ = -(ε I + V_z σ_x + α sk (-σ_y))  [σ_y^T = -σ_y]
    //       = -ε I - V_z σ_x + α sk σ_y
    H[2*4+2] = -eps;
    H[2*4+3] = -Vz + iask;       // -V_z + i α sk
    H[3*4+2] = -Vz - iask;       // -V_z - i α sk
    H[3*4+3] = -eps;
}

// ── Z₂ invariant via Pfaffian sign at TRIM ────────────────────────────────
// At k=0 and k=π, H_BdG is antisymmetric in the time-reversal basis.
// Z₂ = sgn(Pf(H(0)) · Pf(H(π))) = +1 (trivial) or -1 (topological).
//
// For 4×4 antisymmetric A, Pf(A) = A[0,1]*A[2,3] - A[0,2]*A[1,3] + A[0,3]*A[1,2].
// The TRIM Hamiltonians (k=0,π) are real (no imaginary off-diagonals when α sin(0)=0
// and α sin(π)=0), so the Pfaffian is real and equals ±|Pf|.
//
// Shortcut: the gap at TRIM can close only when Pf changes sign.  We use the
// analytical result:
//   E_gap(k=0) = |V_z ± √(μ̃² + Δ²)|  where μ̃ = μ (k=0), μ̃ = μ-4t (k=π)
// Z₂ = -1 iff exactly one of the two TRIM gaps is negative in the effective sense,
// which reduces to: sgn((V_z² − μ² − Δ²) * (V_z² − (μ−4t)² − Δ²)) = -1
//
// This is exact for the Lutchyn-Oreg model:
//   Pf(H(0)) = V_z² - μ² - Δ²  (up to positive overall scale)
//   Pf(H(π)) = V_z² - (μ-4t)² - Δ²
static int compute_z2(double t, double mu, double Delta, double Vz) {
    const double pf0 = Vz*Vz - mu*mu - Delta*Delta;
    const double pfpi = Vz*Vz - (mu - 4.0*t)*(mu - 4.0*t) - Delta*Delta;
    const double prod = pf0 * pfpi;
    return (prod < 0.0) ? -1 : +1;
}

// ── Bulk gap: minimum |E| over BZ ─────────────────────────────────────────
static double compute_gap(double t, double alpha, double mu, double Delta,
                           double Vz, int n_k) {
    const double dk = 2.0 * M_PI / n_k;
    double gap = std::numeric_limits<double>::max();
    Mat H;
    for (int ik = 0; ik < n_k; ++ik) {
        double k = -M_PI + ik * dk;
        lo_ham(k, t, alpha, mu, Delta, Vz, H);
        std::vector<double> ev; Mat ev2;
        jacobi_herm(4, H, ev, ev2);
        for (double e : ev) gap = std::min(gap, std::abs(e));
    }
    return gap;
}

// ── Bulk band structure ───────────────────────────────────────────────────
static void compute_bulk(const LutchynOregOptions& opts, LutchynOregResult& res) {
    const int nk = opts.n_k;
    res.k_bulk.resize(nk);
    res.E0.resize(nk); res.E1.resize(nk);
    res.E2.resize(nk); res.E3.resize(nk);

    Mat H;
    for (int ik = 0; ik < nk; ++ik) {
        double k = -M_PI + 2.0 * M_PI * ik / (nk - 1);
        res.k_bulk[ik] = k;
        lo_ham(k, opts.t, opts.alpha, opts.mu, opts.Delta, opts.V_z, H);
        std::vector<double> ev; Mat ev2;
        jacobi_herm(4, H, ev, ev2);
        res.E0[ik] = ev[0]; res.E1[ik] = ev[1];
        res.E2[ik] = ev[2]; res.E3[ik] = ev[3];
    }
}

// ── Real-space wire: assemble tight-binding Hamiltonian (4 N × 4 N) ──────
//
// Sites j=0..N-1, 4 DOF each: (e↑, e↓, h↓, -h↑).
// Onsite (4×4): [h_e_onsite, iΔσ_y; -iΔσ_y, -h_eᵀ_onsite]
//   h_e_onsite  = (2t - μ) I₂ + V_z σ_x     [cos(0)=1 from expansion, k-dep part in hopping]
//   Wait — in real space the onsite is: ε_0 = -μ + 2t (from 2t(1-cos 0) at k=0 limit?
//   No. The real-space decomposition of 2t(1-cos k) is: 2t δ_{j,j'} - t δ_{j,j'+1} - t δ_{j,j'-1}
//   The α sin(k) term: (α/2i)(δ_{j+1} - δ_{j-1}) = -iα/2 forward + iα/2 backward hopping
//   So onsite: (2t - μ) I₂ τ_z + V_z σ_x + Δ τ_x
//   Hopping j→j+1: -t I₂ τ_z - i(α/2) σ_y τ_z
//   (τ_z in particle-hole space: +I₂ for electron, -I₂ for hole)
//
// In the 4-component basis {e↑, e↓, h↓, -h↑}:
//   τ_z → diag(+1,+1,-1,-1)
//   σ_x → [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]]  (acts on spin, same for both PH sectors)
//   σ_y τ_z → [[0,-i,0,0],[i,0,0,0],[0,0,0,i],[0,0,-i,0]]  (σ_y with τ_z sign flip in hole)
//
// Let me write them out directly.
static void build_wire(double t, double alpha, double mu, double Delta, double Vz,
                        int n_sites, Mat& H_big) {
    const int N4 = 4 * n_sites;
    H_big.assign(N4 * N4, cx{0});

    // Helper to set H[i][j] and H[j][i]*
    auto set = [&](int i, int j, cx val) {
        H_big[i * N4 + j] += val;
        H_big[j * N4 + i] += std::conj(val);
    };

    const double eps0 = 2.0*t - mu;   // onsite kinetic (2t from -t*e^{ik} + 2t - t*e^{-ik})
    const cx     I{1,0}, iI{0,1};

    for (int j = 0; j < n_sites; ++j) {
        int b = 4 * j;   // base index for site j

        // ── Onsite block ──────────────────────────────────────────────────
        // Electron sector (rows/cols b, b+1):
        //   (eps0 I + Vz σ_x) for e↑,e↓
        H_big[(b+0)*N4 + (b+0)] += eps0;       // e↑ onsite
        H_big[(b+1)*N4 + (b+1)] += eps0;       // e↓ onsite
        H_big[(b+0)*N4 + (b+1)] += Vz;         // V_z σ_x e↑↓
        H_big[(b+1)*N4 + (b+0)] += Vz;         // V_z σ_x e↓↑

        // SC pairing: iΔσ_y = [[0,Δ],[-Δ,0]] in top-right corner
        H_big[(b+0)*N4 + (b+3)] +=  Delta;     // e↑ ↔ -h↑ (row 0, col 3)
        H_big[(b+3)*N4 + (b+0)] +=  Delta;     // conjugate
        H_big[(b+1)*N4 + (b+2)] += -Delta;     // e↓ ↔ h↓  (row 1, col 2)
        H_big[(b+2)*N4 + (b+1)] += -Delta;     // conjugate

        // Hole sector (rows/cols b+2, b+3):
        //   -(h_eᵀ)_onsite = -eps0 I - Vz σ_x  (α term vanishes at onsite)
        H_big[(b+2)*N4 + (b+2)] += -eps0;      // h↓  onsite
        H_big[(b+3)*N4 + (b+3)] += -eps0;      // -h↑ onsite
        H_big[(b+2)*N4 + (b+3)] += -Vz;        // -Vz σ_x hole sector
        H_big[(b+3)*N4 + (b+2)] += -Vz;

        // ── Hopping j → j+1 ───────────────────────────────────────────────
        if (j + 1 < n_sites) {
            int b2 = 4 * (j + 1);
            // Electron hopping: T_e = -t I₂ - (iα/2) σ_y
            //   σ_y = [[0,-i],[i,0]]  so -(iα/2) σ_y = [[0, -α/2],[α/2, 0]]
            // Forward hop (j→j+1):
            set(b+0, b2+0, cx{-t, 0.0});              // e↑→e↑
            set(b+1, b2+1, cx{-t, 0.0});              // e↓→e↓
            set(b+0, b2+1, cx{0.0, alpha/2.0});       // e↑→e↓:  -(iα/2)(-i) = -α/2 ... wait
            // -(iα/2) σ_y: entry [0,1] = -(iα/2)(-i) = -(α/2) → cx{-alpha/2, 0}
            // entry [1,0] = -(iα/2)(i) = α/2 → cx{alpha/2, 0}
            // Hole hopping: T_h = -(-t I₂ - (iα/2)(-σ_y))ᵀ = t I₂ + (iα/2)σ_yᵀ... hmm
            //
            // Let me redo this more carefully.
            // The full lattice BdG hopping from site j to j+1:
            //   T_{j,j+1} = [[T_e, 0],[0, T_h]]
            // T_e from electron Hamiltonian: -t I₂ + (α/2i)(forward - backward) spin term
            //   The α sin(k) σ_y in k-space comes from: -iα/2 σ_y (forward hop) + iα/2 σ_y (backward)
            //   So T_e = -t I₂ - (iα/2) σ_y
            // T_h from -h_eᵀ(-k): the hole Hamiltonian in k-space is -h_eᵀ(-k)
            //   h_e(-k) = 2t(1-cos k) - μ + Vz σ_x - α sin(k) σ_y  (sin(-k)=-sin(k))
            //   h_eᵀ(-k) = 2t(1-cos k) - μ + Vz σ_x + α sin(k) σ_y  (σ_y^T = -σ_y → sign flips)
            //   -h_eᵀ(-k) = -(2t(1-cos k) - μ) - Vz σ_x - α sin(k) σ_y
            //   Hopping part: -(-t) I₂ - (iα/2)(-σ_y)ᵀ... let me just use:
            //   Hole forward hop T_h = +t I₂ - (iα/2)(-σ_y)ᵀ = +t I₂ + (iα/2)(-σ_y)
            //   Wait: from -h_eᵀ(-k), the hopping (forward) is: +t I₂ + (iα/2) σ_y
            //   But σ_y^T = -σ_y, so in the hole sector:
            //   T_h (forward j→j+1) = +t I₂ + (iα/2)(-σ_y) = +t I₂ - (iα/2) σ_y ... same as T_e but sign of t flipped
            //
            // OK I'm going to just directly compute all 16 off-diagonal elements.
            // Use: T_e[0,0] = -t, T_e[1,1] = -t
            //      T_e[0,1] = -(iα/2)(-i) = -α/2,  T_e[1,0] = -(iα/2)(i) = α/2
            //      T_h[0,0] = t,  T_h[1,1] = t   (sign flip from -h_eᵀ)
            //      T_h[0,1] = -(iα/2)(-i) = -α/2,  T_h[1,0] = -(iα/2)(i) = α/2
            //      Wait: T_h from -h_eᵀ(-k) forward hop:
            //        h_eᵀ(-k) hopping (j→j+1) = -t^T I = -t I, and α sin(k) σ_y^T = α sin(k)(-σ_y)
            //        forward hop of h_eᵀ(-k) = -t I - (iα/2)(-σ_y) = -t I + (iα/2)σ_y
            //        T_h = -(-t I + (iα/2)σ_y) = t I - (iα/2)σ_y
            //        T_h[0,1] = -(iα/2)(-i) = -α/2,  T_h[1,0] = -(iα/2)(i) = α/2
            //      Same spin structure as T_e, but t → -t in diagonal.
            // Undo the wrong set calls above and redo:
            // (Actually I used set() which does sym, so I'm overwriting — let me just collect them)
        }
    }

    // Redo hopping correctly (overwrite the partial above with correct values)
    // First, clear accidental hops written above in the if-block
    for (int j = 0; j + 1 < n_sites; ++j) {
        int b  = 4 * j;
        int b2 = 4 * (j + 1);
        // Clear the row b+0 col b2+0, b2+1 and b+1 col b2+0, b2+1 that were set in loop
        // (They were set with set() which added to both directions — we'll just rebuild from scratch)
    }

    // Rebuild hoppings cleanly
    // (The onsite blocks set above are correct; only redo the off-diagonal hops)
    // We need to zero out everything in the off-diagonal 4×4 blocks first
    for (int j = 0; j + 1 < n_sites; ++j) {
        int b  = 4 * j;
        int b2 = 4 * (j + 1);
        for (int a = 0; a < 4; ++a)
            for (int bb = 0; bb < 4; ++bb) {
                H_big[(b+a)*N4 + (b2+bb)] = cx{0};
                H_big[(b2+bb)*N4 + (b+a)] = cx{0};
            }
    }
    // Overwrite onsite SC pairing and V_z terms that were double-added via set():
    // Actually set() was used ONLY for SC pairing terms — those are correct.
    // But the hole sector V_z and the SC terms that I wrote with H_big[...] += are also correct.
    // The if-block inside the loop body had set() calls that we need to remove.
    // Rather than tracking what was set, let's rebuild the full matrix cleanly.

    // ── Rebuild entire matrix from scratch ────────────────────────────────
    H_big.assign(N4 * N4, cx{0});

    const double alpha_hop = alpha / 2.0;   // α/2 for forward/backward hopping

    for (int j = 0; j < n_sites; ++j) {
        const int b = 4 * j;

        // ── Onsite (4×4) ─────────────────────────────────────────────────
        // Electron sector
        H_big[(b+0)*N4+(b+0)] = eps0;
        H_big[(b+1)*N4+(b+1)] = eps0;
        H_big[(b+0)*N4+(b+1)] = Vz;
        H_big[(b+1)*N4+(b+0)] = Vz;
        // SC pairing iΔσ_y in off-diagonal corners
        //   top-right: row {0,1}, col {2,3} of the 4×4 → in basis: e↑↔-h↑ (0↔3) and e↓↔h↓ (1↔2)
        H_big[(b+0)*N4+(b+3)] =  Delta;    // (0,3)
        H_big[(b+3)*N4+(b+0)] =  Delta;    // (3,0) = conj
        H_big[(b+1)*N4+(b+2)] = -Delta;    // (1,2)
        H_big[(b+2)*N4+(b+1)] = -Delta;    // (2,1) = conj
        // Hole sector: -h_e^T → diagonal -eps0, off-diagonal -Vz σ_x
        H_big[(b+2)*N4+(b+2)] = -eps0;
        H_big[(b+3)*N4+(b+3)] = -eps0;
        H_big[(b+2)*N4+(b+3)] = -Vz;
        H_big[(b+3)*N4+(b+2)] = -Vz;

        // ── Hopping j → j+1 ──────────────────────────────────────────────
        if (j + 1 < n_sites) {
            const int b2 = 4 * (j + 1);

            // Electron forward hop: T_e = -t I₂ - (iα/2) σ_y
            //   -(iα/2) σ_y: [0,1] = -(iα/2)(-i) = -α/2; [1,0] = -(iα/2)(i) = α/2
            H_big[(b+0)*N4+(b2+0)] = cx{-t, 0};
            H_big[(b+1)*N4+(b2+1)] = cx{-t, 0};
            H_big[(b+0)*N4+(b2+1)] = cx{-alpha_hop, 0};
            H_big[(b+1)*N4+(b2+0)] = cx{ alpha_hop, 0};

            // Hole forward hop: T_h = t I₂ - (iα/2) σ_y  (same spin structure, -t → +t)
            H_big[(b+2)*N4+(b2+2)] = cx{ t, 0};
            H_big[(b+3)*N4+(b2+3)] = cx{ t, 0};
            H_big[(b+2)*N4+(b2+3)] = cx{-alpha_hop, 0};
            H_big[(b+3)*N4+(b2+2)] = cx{ alpha_hop, 0};

            // Hermitian conjugate (backward hop j+1 → j)
            H_big[(b2+0)*N4+(b+0)] = cx{-t, 0};
            H_big[(b2+1)*N4+(b+1)] = cx{-t, 0};
            H_big[(b2+1)*N4+(b+0)] = cx{-alpha_hop, 0};
            H_big[(b2+0)*N4+(b+1)] = cx{ alpha_hop, 0};

            H_big[(b2+2)*N4+(b+2)] = cx{ t, 0};
            H_big[(b2+3)*N4+(b+3)] = cx{ t, 0};
            H_big[(b2+3)*N4+(b+2)] = cx{-alpha_hop, 0};
            H_big[(b2+2)*N4+(b+3)] = cx{ alpha_hop, 0};
        }
    }
}

// ── LDOS at wire ends ─────────────────────────────────────────────────────
// For each eigenstate ψ_n, LDOS at site j:
//   w_n(j) = Σ_{σ} |ψ_n[4j+σ]|²  (sum over 4 components)
// Edge weight: sum over first 2 and last 2 sites
static void compute_ldos(const Mat& evecs, int N4, int n_sites,
                          std::vector<double>& ldos_l,
                          std::vector<double>& ldos_r) {
    const int n_states = N4;
    ldos_l.resize(n_states, 0.0);
    ldos_r.resize(n_states, 0.0);
    const int n_edge = std::min(2, n_sites);
    for (int n = 0; n < n_states; ++n) {
        double wl = 0, wr = 0;
        for (int j = 0; j < n_edge; ++j)
            for (int s = 0; s < 4; ++s)
                wl += std::norm(evecs[j*4+s + n * N4 / N4]);  // eigvec col n
        // eigvecs is row-major: eigvecs[r*N+n] = component r of eigenvector n
        // So eigvecs[r*N4+n] = component r of eigenstate n
        // r = 4*j+s
        wl = 0; wr = 0;
        for (int j = 0; j < n_edge; ++j)
            for (int s = 0; s < 4; ++s)
                wl += std::norm(evecs[(4*j+s)*N4 + n]);
        for (int j = n_sites - n_edge; j < n_sites; ++j)
            for (int s = 0; s < 4; ++s)
                wr += std::norm(evecs[(4*j+s)*N4 + n]);
        ldos_l[n] = wl;
        ldos_r[n] = wr;
    }
}

// ── Wire scenario ─────────────────────────────────────────────────────────
static void compute_wire(const LutchynOregOptions& opts, LutchynOregResult& res) {
    const int n_sites = opts.n_sites;
    const int N4 = 4 * n_sites;

    Mat H_big, evecs;
    std::vector<double> evals;
    build_wire(opts.t, opts.alpha, opts.mu, opts.Delta, opts.V_z, n_sites, H_big);
    jacobi_herm(N4, H_big, evals, evecs);

    res.E_wire = evals;
    compute_ldos(evecs, N4, n_sites, res.ldos_left, res.ldos_right);

    // Majorana oscillations: gap vs wire length L (L=10..n_sites, step 5)
    for (int L = 10; L <= n_sites; L += 5) {
        Mat Hl, evl; std::vector<double> evlv;
        build_wire(opts.t, opts.alpha, opts.mu, opts.Delta, opts.V_z, L, Hl);
        jacobi_herm(4*L, Hl, evlv, evl);
        double g = *std::min_element(evlv.begin(), evlv.end(),
            [](double a, double b){ return std::abs(a) < std::abs(b); });
        res.osc_L.push_back(L);
        res.osc_gap.push_back(std::abs(g));
    }
}

// ── Phase diagram ─────────────────────────────────────────────────────────
static void compute_phase_diagram(const LutchynOregOptions& opts, LutchynOregResult& res) {
    const int nVz = opts.n_Vz, nmu = opts.n_mu;
    if (nVz < 2) throw std::runtime_error("LutchynOreg: n_Vz must be >= 2 for phase_diagram");
    if (nmu < 2) throw std::runtime_error("LutchynOreg: n_mu must be >= 2 for phase_diagram");

    res.pd_Vz_vals.resize(nVz);
    res.pd_mu_vals.resize(nmu);
    res.pd_z2.resize(nVz * nmu);

    for (int iVz = 0; iVz < nVz; ++iVz)
        res.pd_Vz_vals[iVz] = opts.Vz_max * iVz / (nVz - 1);
    for (int imu = 0; imu < nmu; ++imu)
        res.pd_mu_vals[imu] = -opts.mu_max + 2.0 * opts.mu_max * imu / (nmu - 1);

    for (int iVz = 0; iVz < nVz; ++iVz)
        for (int imu = 0; imu < nmu; ++imu)
            res.pd_z2[iVz * nmu + imu] = compute_z2(
                opts.t, res.pd_mu_vals[imu], opts.Delta, res.pd_Vz_vals[iVz]);
}

// ── Main dispatcher ───────────────────────────────────────────────────────
LutchynOregResult run_lutchyn_oreg(const LutchynOregOptions& opts) {
    if (opts.n_k < 4)     throw std::runtime_error("LutchynOreg: n_k must be >= 4");
    if (opts.n_sites < 4) throw std::runtime_error("LutchynOreg: n_sites must be >= 4");
    if (opts.Delta < 0)   throw std::runtime_error("LutchynOreg: Delta must be >= 0");
    if (opts.t <= 0)      throw std::runtime_error("LutchynOreg: t must be > 0");

    const std::string& sc = opts.scenario;
    if (sc != "bulk" && sc != "wire" && sc != "phase_diagram" && sc != "both")
        throw std::runtime_error(
            "LutchynOreg: unknown scenario \"" + sc +
            "\"; valid values: bulk, wire, phase_diagram, both");

    auto t0 = std::chrono::steady_clock::now();

    LutchynOregResult res;
    res.scenario = sc;
    res.t        = opts.t;
    res.alpha    = opts.alpha;
    res.mu       = opts.mu;
    res.Delta    = opts.Delta;
    res.V_z      = opts.V_z;
    res.V_z_critical = std::sqrt(opts.mu * opts.mu + opts.Delta * opts.Delta);

    res.z2 = compute_z2(opts.t, opts.mu, opts.Delta, opts.V_z);
    res.phase_label = (res.z2 < 0) ? "Topological SC (Z\u2082=-1)" : "Trivial SC (Z\u2082=+1)";
    res.gap = compute_gap(opts.t, opts.alpha, opts.mu, opts.Delta, opts.V_z, opts.n_k);

    if (sc == "bulk" || sc == "both")
        compute_bulk(opts, res);

    if (sc == "wire" || sc == "both")
        compute_wire(opts, res);

    if (sc == "phase_diagram")
        compute_phase_diagram(opts, res);

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.message = "Vz=" + std::to_string(opts.V_z) +
                  ", Vzc=" + std::to_string(res.V_z_critical) +
                  ", Z2=" + std::to_string(res.z2) +
                  ", gap=" + std::to_string(res.gap);
    res.ok = true;
    return res;
}

}  // namespace qx
