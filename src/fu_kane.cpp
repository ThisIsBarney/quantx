// src/fu_kane.cpp  —  M40: Fu-Kane Superconductor / Majorana Zero Modes
//
// BdG Hamiltonian for the surface of a 3D topological insulator (Bi₂Se₃-type)
// proximitized with an s-wave superconductor and subject to a Zeeman exchange
// field M (from a magnetic TI cap-layer or applied field along ẑ):
//
//   h(k)   = vF(kx σy − ky σx) + M σz − μ I₂
//   H_BdG  = [ h(k),       iΔσy  ]     (4×4, Nambu⊗spin basis)
//             [−iΔσy*,  −hᵀ(−k)  ]
//
// Phase transition: M_c = √(μ² + Δ²).
//   M < M_c → topological SC (Chern C = 1), chiral Majorana edge mode at E = 0
//   M > M_c → trivial SC    (Chern C = 0)
//
// Scenarios
//   "bulk"         – BdG band structure (kx scan, ky=0) + Chern number
//   "edge"         – strip spectrum (n_y sites, open BCs in y) + Chern number
//   "phase_diagram"– Chern(M, μ) map over a 2D grid
//   "both"         – bulk + edge (skip phase-diagram heavy sweep)

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

namespace qx {

using cx  = std::complex<double>;
using Mat = std::vector<cx>;   // row-major NxN

// ── Jacobi eigensolver (Hermitian) ────────────────────────────────────────
// Identical copy to the one in edge_states.cpp / weyl.cpp to keep each TU
// self-contained (no shared translation-unit dependency).
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
                double t   = (tau >= 0) ?  1.0 / (tau + std::sqrt(1.0 + tau*tau))
                                         : -1.0 / (-tau + std::sqrt(1.0 + tau*tau));
                double c   = 1.0 / std::sqrt(1.0 + t*t);
                double s   = c * t;
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
    // sort
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

// ── BdG Hamiltonian (bulk, 4×4) ───────────────────────────────────────────
//
// Basis: {e↑, e↓, h↓, −h↑}  (standard Nambu spinor)
//
// h(k) = vF(kx σy − ky σx) + M σz − μ I
//       = [[ M−μ,        −vF(ky+ikx)  ],
//          [ vF(ikx−ky),  −M−μ        ]]
//
// iΔσy = [[0, Δ], [−Δ, 0]]
//
// −hᵀ(−k): replace k→−k then transpose-negate.
//   h(−k) = [[M−μ, vF(ky+ikx)],[ vF(−ikx+ky), −M−μ]]
//   hᵀ(−k) = [[M−μ, vF(ky−ikx)],[ vF(ky+ikx), −M−μ]]
//   −hᵀ(−k) = [[μ−M, −vF(ky−ikx)],[−vF(ky+ikx), M+μ]]
//
// Full 4×4:
//   row 0: [ M−μ,         −vF(ky+ikx),    0,          Δ          ]
//   row 1: [ vF(ikx−ky),  −M−μ,           −Δ,         0          ]
//   row 2: [ 0,           −Δ,              μ−M,       −vF(ky−ikx) ]
//   row 3: [ Δ,            0,             −vF(ky+ikx), M+μ        ]
//
static Mat fu_kane_ham(double kx, double ky,
                        double vF, double M, double mu, double Delta) {
    const cx ikx{0.0, kx}, iky{0.0, ky};
    const cx D{Delta};
    // off-diagonal surface terms
    const cx h01 = -vF * (cx{ky} + ikx);   // h(k)[0,1]
    const cx h10 =  vF * (ikx - cx{ky});    // h(k)[1,0]
    // hole-sector off-diagonals
    const cx hh23 = -vF * (cx{ky} - ikx);  // −hᵀ(−k)[2,3]
    const cx hh32 = -vF * (cx{ky} + ikx);  // −hᵀ(−k)[3,2]

    Mat H(16, cx{0});
    // row 0
    H[0*4+0] = cx{M - mu};   H[0*4+1] = h01;          H[0*4+2] = cx{0};  H[0*4+3] = D;
    // row 1
    H[1*4+0] = h10;          H[1*4+1] = cx{-M - mu};  H[1*4+2] = -D;     H[1*4+3] = cx{0};
    // row 2
    H[2*4+0] = cx{0};        H[2*4+1] = -D;           H[2*4+2] = cx{mu - M};  H[2*4+3] = hh23;
    // row 3
    H[3*4+0] = D;             H[3*4+1] = cx{0};        H[3*4+2] = hh32;         H[3*4+3] = cx{M + mu};
    return H;
}

// ── Diagonalise a 4×4 BdG, returning all 4 eigenvalues sorted ─────────────
static std::array<double,4> bdg_evals_4(const Mat& H) {
    Mat A = H;
    std::vector<double> ev;
    Mat evec;
    jacobi_herm(4, A, ev, evec);
    return {ev[0], ev[1], ev[2], ev[3]};
}

// ── Occupied eigenvectors (2 negative-energy cols of 4×4 BdG) ──────────────
static void bdg_occupied_2(const Mat& H, cx* q0, cx* q1) {
    Mat A = H;
    std::vector<double> ev;
    Mat evec;
    jacobi_herm(4, A, ev, evec);
    // cols 0 and 1 correspond to the two lowest (negative) eigenvalues
    for (int r = 0; r < 4; ++r) {
        q0[r] = evec[r*4+0];
        q1[r] = evec[r*4+1];
    }
}

// ── FHS Chern number (multi-band, 2 occupied bands of 4-band BdG) ─────────
//
// Uses the lattice-gauge FHS algorithm generalised to n_occ = 2:
//   det-link U_μ(k) = det(⟨u_a(k)|u_b(k+δμ)⟩) / |det(...)|
//   F12 = arg(det_U1(k) · det_U2(k+δ1) · conj(det_U1(k+δ2)) · conj(det_U2(k)))
//   C = round(ΣF12 / 2π)
//
static int chern_fu_kane(double vF, double M, double mu, double Delta, int nk) {
    // Grid: k = (i/nk) * 2π, j/nk * 2π  ∈ [0, 2π)
    const double dk = 2.0 * M_PI / nk;

    // Pre-compute occupied eigenvectors on (nk+1) × (nk+1) grid
    // (extra column/row for periodic wraparound)
    const int NP = nk + 1;
    std::vector<std::array<cx,8>> Q(NP * NP);  // 2 cols × 4 rows = 8 cx per point

    for (int i = 0; i < NP; ++i) {
        double kx = i * dk - M_PI;
        for (int j = 0; j < NP; ++j) {
            double ky = j * dk - M_PI;
            Mat H = fu_kane_ham(kx, ky, vF, M, mu, Delta);
            cx q0[4], q1[4];
            bdg_occupied_2(H, q0, q1);
            auto& q = Q[i*NP+j];
            for (int r = 0; r < 4; ++r) { q[r] = q0[r]; q[r+4] = q1[r]; }
        }
    }

    // FHS accumulation
    auto det_link = [&](int i0, int j0, int i1, int j1) -> cx {
        // 2×2 overlap matrix ⟨u_a(k0)|u_b(k1)⟩, a,b ∈ {0,1}
        const auto& q0 = Q[i0*NP+j0];
        const auto& q1 = Q[i1*NP+j1];
        // col a of Q0: rows 0-3 (a=0) or rows 4-7 (a=1)
        // det = o00*o11 − o01*o10
        cx o00{0}, o01{0}, o10{0}, o11{0};
        for (int r = 0; r < 4; ++r) {
            o00 += std::conj(q0[r])   * q1[r];
            o01 += std::conj(q0[r])   * q1[r+4];
            o10 += std::conj(q0[r+4]) * q1[r];
            o11 += std::conj(q0[r+4]) * q1[r+4];
        }
        cx d = o00*o11 - o01*o10;
        double ad = std::abs(d);
        return (ad > 1e-14) ? d / ad : cx{1.0};
    };

    double F_sum = 0.0;
    for (int i = 0; i < nk; ++i) {
        for (int j = 0; j < nk; ++j) {
            cx U1  = det_link(i,   j,   i+1, j  );
            cx U2  = det_link(i,   j,   i,   j+1);
            cx U1p = det_link(i,   j+1, i+1, j+1);
            cx U2p = det_link(i+1, j,   i+1, j+1);
            F_sum += std::arg(U1 * U2p * std::conj(U1p) * std::conj(U2));
        }
    }
    return static_cast<int>(std::round(F_sum / (2.0 * M_PI)));
}

// ── BdG band structure: kx scan at ky=0 ──────────────────────────────────
static void compute_bulk(const FuKaneOptions& opts, FuKaneResult& res) {
    const int nkx = opts.n_kx;
    const double vF = opts.v_F, M = opts.M_val, mu = opts.mu, D = opts.Delta;
    res.kx_bulk.resize(nkx);
    res.E0.resize(nkx); res.E1.resize(nkx); res.E2.resize(nkx); res.E3.resize(nkx);

    double min_gap = std::numeric_limits<double>::max();
    for (int i = 0; i < nkx; ++i) {
        double kx = -M_PI + 2.0 * M_PI * i / (nkx - 1);
        res.kx_bulk[i] = kx;
        auto H = fu_kane_ham(kx, 0.0, vF, M, mu, D);
        auto ev = bdg_evals_4(H);
        res.E0[i] = ev[0]; res.E1[i] = ev[1];
        res.E2[i] = ev[2]; res.E3[i] = ev[3];
        // gap = smallest |E|
        for (auto e : ev) min_gap = std::min(min_gap, std::abs(e));
    }
    res.gap = min_gap;
}

// ── Strip edge-state spectrum ──────────────────────────────────────────────
//
// Lattice model on N_y sites. Basis at each site j: (e↑, e↓, h↓, −h↑).
// Onsite (ky=0):  diagonal part of H_BdG at ky=0
// Y-hopping:      from -vF ky σx in electron sector → forward hop T^e = i vF/2 σx
//                 from +vF ky σx in hole sector   → forward hop T^h = -i vF/2 σx
//   ⟹ T_{j→j+1} = i vF/2 · diag_block(σx, −σx)  (4×4)
//
static void compute_edge(const FuKaneOptions& opts, FuKaneResult& res) {
    const int ny = opts.n_y, nkx = opts.n_kx;
    const int N  = 4 * ny;
    const double vF = opts.v_F, M = opts.M_val, mu = opts.mu, D = opts.Delta;

    res.kx_edge.resize(nkx);
    res.n_edge_evals = N;
    res.E_edge.resize(nkx * N);

    // Forward hopping matrix T (4×4): i vF/2 block-diag(σx, −σx)
    // σx = [[0,1],[1,0]]; T[e block] = i vF/2 σx, T[h block] = -i vF/2 σx
    const cx ivF2{0.0, vF * 0.5};
    // T[row][col] only has 4 non-zero entries
    // e-block (rows/cols 0-1): T[0,1]=T[1,0]=ivF2
    // h-block (rows/cols 2-3): T[2,3]=T[3,2]=-ivF2
    auto fill_hop = [&](Mat& H_big, int from, int to) {
        // H_big[to*4+r, from*4+c] += T[r,c]
        // T: (0,1)->ivF2, (1,0)->ivF2, (2,3)->-ivF2, (3,2)->-ivF2
        H_big[(to*4+0)*N + (from*4+1)] += ivF2;
        H_big[(to*4+1)*N + (from*4+0)] += ivF2;
        H_big[(to*4+2)*N + (from*4+3)] += -ivF2;
        H_big[(to*4+3)*N + (from*4+2)] += -ivF2;
        // Hermitian conjugate (backward hop T†):
        H_big[(from*4+1)*N + (to*4+0)] += std::conj(ivF2);
        H_big[(from*4+0)*N + (to*4+1)] += std::conj(ivF2);
        H_big[(from*4+3)*N + (to*4+2)] += std::conj(-ivF2);
        H_big[(from*4+2)*N + (to*4+3)] += std::conj(-ivF2);
    };

    for (int ik = 0; ik < nkx; ++ik) {
        double kx = -M_PI + 2.0 * M_PI * ik / (nkx - 1);
        res.kx_edge[ik] = kx;

        Mat H_big(N * N, cx{0});

        // Onsite blocks (ky=0 for each site — ky enters only via hopping)
        auto H4 = fu_kane_ham(kx, 0.0, vF, M, mu, D);
        for (int j = 0; j < ny; ++j) {
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    H_big[(j*4+r)*N + (j*4+c)] += H4[r*4+c];
        }

        // Hopping between adjacent sites
        for (int j = 0; j < ny - 1; ++j)
            fill_hop(H_big, j, j+1);

        // Diagonalise
        std::vector<double> evals;
        Mat evecs;
        jacobi_herm(N, H_big, evals, evecs);

        for (int b = 0; b < N; ++b)
            res.E_edge[ik * N + b] = evals[b];
    }
}

// ── Phase diagram: Chern(M, μ) on a 2D grid ───────────────────────────────
static void compute_phase_diagram(const FuKaneOptions& opts, FuKaneResult& res) {
    const int nM = opts.n_M, nmu = opts.n_mu, nk = opts.n_k;
    const double vF = opts.v_F, D = opts.Delta;

    res.pd_M_vals.resize(nM);
    res.pd_mu_vals.resize(nmu);
    res.pd_chern.resize(nM * nmu);

    for (int iM = 0; iM < nM; ++iM)
        res.pd_M_vals[iM] = opts.M_max * iM / (nM - 1);
    for (int imu = 0; imu < nmu; ++imu)
        res.pd_mu_vals[imu] = -opts.mu_max + 2.0 * opts.mu_max * imu / (nmu - 1);

    for (int iM = 0; iM < nM; ++iM) {
        double Mv = res.pd_M_vals[iM];
        for (int imu = 0; imu < nmu; ++imu) {
            double muv = res.pd_mu_vals[imu];
            res.pd_chern[iM * nmu + imu] = chern_fu_kane(vF, Mv, muv, D, nk);
        }
    }
}

// ── Main dispatcher ───────────────────────────────────────────────────────
FuKaneResult run_fu_kane(const FuKaneOptions& opts) {
    if (opts.n_k < 4)   throw std::runtime_error("FuKane: n_k must be >= 4");
    if (opts.n_y < 2)   throw std::runtime_error("FuKane: n_y must be >= 2");
    if (opts.n_kx < 2)  throw std::runtime_error("FuKane: n_kx must be >= 2");
    if (opts.Delta < 0) throw std::runtime_error("FuKane: Delta must be >= 0");

    const std::string& sc = opts.scenario;
    if (sc != "bulk" && sc != "edge" && sc != "phase_diagram" && sc != "both")
        throw std::runtime_error(
            "FuKane: unknown scenario \"" + sc +
            "\"; valid values: bulk, edge, phase_diagram, both");

    auto t0 = std::chrono::steady_clock::now();

    FuKaneResult res;
    res.scenario  = sc;
    res.v_F       = opts.v_F;
    res.M_val     = opts.M_val;
    res.mu        = opts.mu;
    res.Delta     = opts.Delta;
    res.M_critical = std::sqrt(opts.mu * opts.mu + opts.Delta * opts.Delta);

    // Always compute the Chern number at the requested operating point
    res.chern = chern_fu_kane(opts.v_F, opts.M_val, opts.mu, opts.Delta, opts.n_k);
    res.phase_label = (res.chern != 0) ? "Topological SC (C=1)" : "Trivial SC (C=0)";

    if (sc == "bulk" || sc == "both")
        compute_bulk(opts, res);

    if (sc == "edge" || sc == "both")
        compute_edge(opts, res);

    if (sc == "phase_diagram")
        compute_phase_diagram(opts, res);

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.message = "FuKane M=" + std::to_string(opts.M_val) +
                  " mu=" + std::to_string(opts.mu) +
                  " Delta=" + std::to_string(opts.Delta) +
                  " Mc=" + std::to_string(res.M_critical) +
                  " C=" + std::to_string(res.chern);
    res.ok = true;
    return res;
}

}  // namespace qx
