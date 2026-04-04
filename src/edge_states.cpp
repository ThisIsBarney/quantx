// src/edge_states.cpp -- M37: Edge States / Surface Spectrum (BHZ strip geometry)
//
// Computes the energy spectrum of the BHZ model in a strip geometry:
//   - y direction: finite, N_y sites, OPEN boundary conditions
//   - x direction: periodic, kx is a good quantum number
//
// For each kx, the strip Hamiltonian is a block-tridiagonal (4*N_y)×(4*N_y)
// Hermitian matrix, diagonalized to yield 4*N_y energy eigenvalues.
//
// Topological edge states (Z₂=1, 0 < M/B < 4) appear as in-gap bands
// connecting valence to conduction band — the "smoking gun" of the BHZ TI.
//
// Strip Hamiltonian construction (from the lattice BHZ model):
//   H(kx,ky) = (C-2D(2-cos kx-cos ky))I + A sin kx Γ₁ + A sin ky Γ₂
//             + (M-2B(2-cos kx-cos ky))Γ₅
//
//   → Fourier transform in ky → real-space y hoppings:
//
//   H_onsite(kx) = [C - 2D(2-cos kx)] I + A sin kx Γ₁ + [M - 2B(2-cos kx)] Γ₅
//                  (the constant -2D and -2B come from two y-neighbours)
//   H_hop        = -D I + B Γ₅ - (iA/2) Γ₂      [|j><j+1| hopping]
//   H_hop†       = -D I + B Γ₅ + (iA/2) Γ₂      [|j+1><j| hopping]
//
// Edge localization: each eigenstate ψ is scored by the probability weight
// on the two outermost sites:
//   edge_weight = Σ_{j ∈ {0,1,N-2,N-1}} Σ_α |ψ(j,α)|²   ∈ [0,1]
// Bulk states → 0, edge states → 1.
//
// For the phase_diagram scenario: sweep M, compute Wilson loop Z₂ at each
// value to label the phase (re-uses BHZ eigensolver inline).

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
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
// A.  BHZ matrices: on-site block and nearest-neighbour hopping in y
// ─────────────────────────────────────────────────────────────────────────────

// Γ matrices (4×4, row-major):
//   Γ₁ = σx⊗σx: (0,3)=1,(1,2)=1,(2,1)=1,(3,0)=1
//   Γ₂ = σx⊗σy: (0,3)=-i,(1,2)=i,(2,1)=-i,(3,0)=i
//   Γ₅ = σz⊗I:  diag(1,1,-1,-1)
//
// H_onsite(kx):
//   diag: (ε+d5, ε+d5, ε-d5, ε-d5)
//   off:  (0,3)=d1-id2_0, (1,2)=d1+id2_0, (2,1)=d1-id2_0, (3,0)=d1+id2_0
//   where d2_0=0 (no sin ky on-site), ε=C-2D(2-cos kx)-2D=-C+2D... wait:
//
//   ε_on = C - 2D*(2 - cos kx) — this already includes both y-neighbours' D
//   d5_on= M - 2B*(2 - cos kx)  — same for B
//   d1_on= A*sin kx
//   (no d2 on-site since sin(ky) term lives entirely in the hoppings)

static void fill_onsite(double kx, double M, double A, double B, double C, double D,
                        cx out[16]) {
    double eps = C - 2.0*D*(2.0 - std::cos(kx));
    double d1  = A * std::sin(kx);
    double d5  = M - 2.0*B*(2.0 - std::cos(kx));
    // Same structure as bhz_ham with d2=0
    out[0]  = cx{eps + d5}; out[1]  = 0;
    out[2]  = 0;            out[3]  = cx{d1};
    out[4]  = 0;            out[5]  = cx{eps + d5};
    out[6]  = cx{d1};       out[7]  = 0;
    out[8]  = 0;            out[9]  = cx{d1};
    out[10] = cx{eps - d5}; out[11] = 0;
    out[12] = cx{d1};       out[13] = 0;
    out[14] = 0;            out[15] = cx{eps - d5};
}

// H_hop = -D I + B Γ₅ - (iA/2) Γ₂
//   Γ₂ contributes: (0,3)=-i, (1,2)=i, (2,1)=-i, (3,0)=i
//   so -(iA/2)*Γ₂: (0,3)=-(iA/2)(-i)=-A/2, (1,2)=-(iA/2)(i)=A/2,
//                  (2,1)=-(iA/2)(-i)=-A/2, (3,0)=-(iA/2)(i)=A/2
//   Γ₅ diag: (1,1,-1,-1)
//   -D I diag: (-D,-D,-D,-D)
static void fill_hop(double A, double B, double D,
                     cx hop[16]) {
    // hop = -D I + B Γ5 - (iA/2) Γ2
    double bmd = B - D;   // B*Γ5 diagonal (+B) minus D*I → upper block diag = -D+B
    double mbd = -B - D;  // lower block diag = -D-B
    double ha  = A * 0.5;
    for (int i = 0; i < 16; ++i) hop[i] = cx{0};
    hop[0]  = cx{bmd};   // (0,0)
    hop[5]  = cx{bmd};   // (1,1)
    hop[10] = cx{mbd};   // (2,2)
    hop[15] = cx{mbd};   // (3,3)
    // off-diagonal from -(iA/2)Γ₂:
    // Γ₂: (0,3)=-i → -(iA/2)*(-i) = -(iA/2)*(-i) = -A/2
    hop[3]  = cx{-ha};   // (0,3)
    hop[6]  = cx{ ha};   // (1,2)
    hop[9]  = cx{-ha};   // (2,1)
    hop[12] = cx{ ha};   // (3,0)
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  General N×N complex Hermitian Jacobi eigensolver
// ─────────────────────────────────────────────────────────────────────────────

static void jacobi_herm(int N, std::vector<cx>& A,
                         std::vector<double>& eigvals,
                         std::vector<cx>& eigvecs) {
    eigvecs.assign(N * N, cx{0});
    for (int i = 0; i < N; ++i) eigvecs[i*N+i] = cx{1.0};

    const int max_sweeps = 30;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        // Check off-diagonal norm for early exit
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
                double t   = (tau >= 0) ?  1.0 / (tau + std::sqrt(1.0 + tau*tau))
                                         : -1.0 / (-tau + std::sqrt(1.0 + tau*tau));
                double c   = 1.0 / std::sqrt(1.0 + t*t);
                double s   = c * t;
                cx ph = apq / r_apq;   // phase of A[p,q]

                // Column update: A → A G  (right multiply)
                for (int r = 0; r < N; ++r) {
                    cx arp = A[r*N+p], arq = A[r*N+q];
                    A[r*N+p] = cx{c}*arp - cx{s}*std::conj(ph)*arq;
                    A[r*N+q] = cx{s}*ph*arp + cx{c}*arq;
                }
                // Row update: A → G† A  (left multiply)
                for (int r = 0; r < N; ++r) {
                    cx apr = A[p*N+r], aqr = A[q*N+r];
                    A[p*N+r] = cx{c}*apr - cx{s}*ph*aqr;
                    A[q*N+r] = cx{s}*std::conj(ph)*apr + cx{c}*aqr;
                }
                // Accumulate eigvecs (columns of eigvecs)
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

    // Insertion sort by eigenvalue, permuting eigvecs columns
    for (int i = 1; i < N; ++i) {
        double key = eigvals[i];
        std::vector<cx> col(N);
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

// ─────────────────────────────────────────────────────────────────────────────
// C.  Build strip Hamiltonian and diagonalize
// ─────────────────────────────────────────────────────────────────────────────

// Build (4*n_y)×(4*n_y) block-tridiagonal strip Hamiltonian at given kx.
static void build_strip(double kx, double M, double A, double B, double C, double D,
                        int n_y, std::vector<cx>& Hmat) {
    int NB = 4;      // block size (4 orbitals per site)
    int N  = NB * n_y;
    Hmat.assign(N * N, cx{0});

    cx Hon[16], Hhop[16];
    fill_onsite(kx, M, A, B, C, D, Hon);
    fill_hop(A, B, D, Hhop);

    // Place diagonal blocks (on-site)
    for (int j = 0; j < n_y; ++j) {
        int base = j * NB;
        for (int r = 0; r < NB; ++r)
            for (int c2 = 0; c2 < NB; ++c2)
                Hmat[(base+r)*N + (base+c2)] += Hon[r*NB+c2];
    }

    // Place off-diagonal blocks (hoppings)
    for (int j = 0; j < n_y - 1; ++j) {
        int bj  = j * NB;
        int bj1 = (j+1) * NB;
        for (int r = 0; r < NB; ++r) {
            for (int c2 = 0; c2 < NB; ++c2) {
                // |j><j+1|: H_hop
                Hmat[(bj+r)*N + (bj1+c2)] += Hhop[r*NB+c2];
                // |j+1><j|: H_hop†
                Hmat[(bj1+r)*N + (bj+c2)] += std::conj(Hhop[c2*NB+r]);
            }
        }
    }
}

// Edge weight of eigenvector (column col of eigvecs matrix, N×N row-major).
// = probability density on first 2 and last 2 physical sites.
static double edge_weight(const std::vector<cx>& eigvecs, int N, int col, int n_y) {
    double w = 0;
    int NB = 4;
    int edge_sites = std::min(2, n_y / 2);
    for (int j = 0; j < edge_sites; ++j) {
        for (int alpha = 0; alpha < NB; ++alpha) {
            int row0 = j * NB + alpha;
            int rowN = (n_y - 1 - j) * NB + alpha;
            w += std::norm(eigvecs[row0 * N + col]);
            w += std::norm(eigvecs[rowN * N + col]);
        }
    }
    return std::min(w, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// D.  Wilson loop Z₂ (inline — same logic as M36, independent of n_y)
//     Used for the phase_diagram scenario.
// ─────────────────────────────────────────────────────────────────────────────

static void bhz_lower_q(double kx, double ky,
                        double M, double A, double B, double C, double D,
                        cx Q[8]) {
    // Reuse strip on-site logic with full (kx,ky) dependence
    double eps = C - 2.0*D*(2.0 - std::cos(kx) - std::cos(ky));
    double d1  = A * std::sin(kx);
    double d2  = A * std::sin(ky);
    double d5  = M - 2.0*B*(2.0 - std::cos(kx) - std::cos(ky));
    cx H4[16];
    H4[0]=cx{eps+d5}; H4[1]=0;        H4[2]=0;           H4[3]=cx{d1}-cx{0,d2};
    H4[4]=0;          H4[5]=cx{eps+d5}; H4[6]=cx{d1}+cx{0,d2}; H4[7]=0;
    H4[8]=0;          H4[9]=cx{d1}-cx{0,d2}; H4[10]=cx{eps-d5}; H4[11]=0;
    H4[12]=cx{d1}+cx{0,d2}; H4[13]=0; H4[14]=0;          H4[15]=cx{eps-d5};

    // Jacobi 4×4
    std::vector<cx> Av(H4, H4+16), ev; std::vector<double> evals;
    jacobi_herm(4, Av, evals, ev);
    for (int row = 0; row < 4; ++row) {
        Q[row*2+0] = ev[row*4+0];
        Q[row*2+1] = ev[row*4+1];
    }
}

static std::array<double,2> wl_wcc(double ky, int n_kx,
                                    double M, double A, double B, double C, double D) {
    std::array<cx,4> W = {cx{1},0, 0,cx{1}};
    for (int ix = 0; ix < n_kx; ++ix) {
        double kx0 = 2.0*M_PI * ix / n_kx;
        double kx1 = 2.0*M_PI * (ix+1) / n_kx;
        cx Q0[8], Q1[8];
        bhz_lower_q(kx0, ky, M, A, B, C, D, Q0);
        bhz_lower_q(kx1, ky, M, A, B, C, D, Q1);
        std::array<cx,4> Ov{};
        for (int row = 0; row < 4; ++row)
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j)
                    Ov[i*2+j] += std::conj(Q0[row*2+i]) * Q1[row*2+j];
        W = { W[0]*Ov[0]+W[1]*Ov[2], W[0]*Ov[1]+W[1]*Ov[3],
              W[2]*Ov[0]+W[3]*Ov[2], W[2]*Ov[1]+W[3]*Ov[3] };
    }
    cx tr = W[0]+W[3], det = W[0]*W[3]-W[1]*W[2];
    cx disc = std::sqrt(tr*tr - cx{4.0}*det);
    cx ev0 = (tr-disc)*0.5, ev1 = (tr+disc)*0.5;
    auto to_wcc = [](cx e) {
        double th = std::arg(e);
        double w = -th / (2.0*M_PI);
        if (w < 0) w += 1.0;
        return w;
    };
    std::array<double,2> wcc = {to_wcc(ev0), to_wcc(ev1)};
    if (wcc[0] > wcc[1]) std::swap(wcc[0], wcc[1]);
    return wcc;
}

static int z2_strip(double M, double A, double B, double C, double D,
                    int n_kx_wl, int n_ky_wl) {
    std::vector<std::array<double,2>> grid(n_ky_wl);
    for (int iy = 0; iy < n_ky_wl; ++iy) {
        double ky = M_PI * iy / (n_ky_wl - 1);
        grid[iy] = wl_wcc(ky, n_kx_wl, M, A, B, C, D);
    }
    // Branch-continuity crossing count (same as M36 fix)
    auto circ_dist = [](double a, double b) {
        double d = std::abs(a - b);
        return d > 0.5 ? 1.0 - d : d;
    };
    double track[2] = {grid[0][0], grid[0][1]};
    int crossings = 0;
    for (int i = 1; i < n_ky_wl; ++i) {
        double c0 = grid[i][0], c1 = grid[i][1];
        double ds = circ_dist(c0,track[0]) + circ_dist(c1,track[1]);
        double dc = circ_dist(c1,track[0]) + circ_dist(c0,track[1]);
        double next[2] = (ds <= dc) ? std::array<double,2>{c0,c1}.data()
                                    : std::array<double,2>{c1,c0}.data();
        // write inline to avoid VLA
        double n0, n1;
        if (ds <= dc) { n0=c0; n1=c1; } else { n0=c1; n1=c0; }
        if ((track[0]-0.5)*(n0-0.5) < 0) ++crossings;
        if ((track[1]-0.5)*(n1-0.5) < 0) ++crossings;
        track[0]=n0; track[1]=n1;
    }
    (void)next;
    return crossings % 2;
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

EdgeStatesResult run_edge_states(const EdgeStatesOptions& opts) {
    auto t0 = std::chrono::steady_clock::now();

    if (opts.n_y < 4)
        throw std::runtime_error("EdgeStates: n_y must be >= 4");
    if (opts.n_y > 60)
        throw std::runtime_error("EdgeStates: n_y must be <= 60");
    if (opts.n_kx < 2)
        throw std::runtime_error("EdgeStates: n_kx must be >= 2");

    const double M = opts.M, A = opts.A, B = opts.B, C = opts.C, D = opts.D;
    const int n_y = opts.n_y, n_kx = opts.n_kx;
    const int n_bands = 4 * n_y;

    EdgeStatesResult res;
    res.M = M; res.A = A; res.B = B;
    res.n_y = n_y; res.n_bands = n_bands;
    res.scenario = opts.scenario;

    // ── Compute strip spectrum at all kx ──────────────────────────────────
    res.kx_vals.resize(n_kx);
    res.energies.resize(n_kx * n_bands);
    res.edge_weight.resize(n_kx * n_bands);

    for (int ix = 0; ix < n_kx; ++ix) {
        double kx = -M_PI + 2.0*M_PI * ix / (n_kx - 1);
        res.kx_vals[ix] = kx;

        std::vector<cx> Hmat;
        build_strip(kx, M, A, B, C, D, n_y, Hmat);

        std::vector<double> evals;
        std::vector<cx> evecs;
        jacobi_herm(n_bands, Hmat, evals, evecs);

        for (int b = 0; b < n_bands; ++b) {
            res.energies[ix * n_bands + b]    = evals[b];
            res.edge_weight[ix * n_bands + b] = edge_weight(evecs, n_bands, b, n_y);
        }
    }

    // ── Detect in-gap states and determine Z₂ ────────────────────────────
    // Use Wilson loop Z₂ with coarser grid (quick)
    int z2 = z2_strip(M, A, B, C, D, /*n_kx_wl=*/20, /*n_ky_wl=*/16);
    res.z2 = z2;
    res.phase_label = (z2 == 1) ? "Topological (Z\342\202\202=1)"
                                 : "Trivial (Z\342\202\202=0)";

    if (opts.scenario == "spectrum") {
        res.message = "M=" + std::to_string(M) +
                      ", Z2=" + std::to_string(z2) +
                      ", n_y=" + std::to_string(n_y);

    } else if (opts.scenario == "phase_diagram") {
        if (opts.sweep_param != "M")
            throw std::runtime_error(
                "EdgeStates: unsupported sweep_param '" + opts.sweep_param +
                "' — only 'M' is supported");
        int n = opts.n_sweep;
        if (n < 2) throw std::runtime_error("EdgeStates: n_sweep must be >= 2");
        res.sweep_vals.resize(n);
        res.sweep_z2.resize(n);
        res.sweep_phase.resize(n);
        for (int k = 0; k < n; ++k) {
            double Mp = opts.param_min +
                        (opts.param_max - opts.param_min) * k / (n - 1);
            res.sweep_vals[k] = Mp;
            int z2k = z2_strip(Mp, A, B, C, D, 16, 12);
            res.sweep_z2[k]    = z2k;
            res.sweep_phase[k] = (z2k == 1) ? "Topological" : "Trivial";
        }
        res.message = "phase diagram: swept M in ["
                    + std::to_string(opts.param_min) + ", "
                    + std::to_string(opts.param_max) + "]";

    } else {
        throw std::runtime_error(
            "EdgeStates: unknown scenario '" + opts.scenario +
            "' — use spectrum | phase_diagram");
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    res.ok = true;
    return res;
}

}  // namespace qx
