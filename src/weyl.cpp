// src/weyl.cpp -- M39: Weyl Semimetal & Surface Fermi Arcs
//
// Minimal 2-band cubic-lattice Weyl Hamiltonian:
//   H(k) = sin(kx) σx + sin(ky) σy + (m - cos(kx) - cos(ky) - cos(kz)) σz
//
// Weyl nodes at kx=ky=0, cos(kz)=m-2 (exist for 1 < m < 3).
//   Node at +k0: chirality +1 (source of Berry flux)
//   Node at -k0: chirality -1 (sink of Berry flux)
//
// Chern number on kx-ky slices: C1=+1 between nodes, C1=0 outside.
// (Computed via Fukui-Hatsugai-Suzuki lattice gauge formula.)
//
// Surface Fermi arcs: slab geometry (finite in z, periodic in x,y).
//   H_slab(kx,ky): (2*n_z) x (2*n_z) block-tridiagonal Hermitian matrix.
//   On-site block: sin(kx)σx + sin(ky)σy + (m - cos(kx) - cos(ky))σz
//   Forward z-hopping: -(1/2)σz
//   For each (kx,ky): eigenstate closest to Fermi energy → edge weight.

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
namespace {

using cx = std::complex<double>;

// ─────────────────────────────────────────────────────────────────────────────
// A.  2×2 Weyl Hamiltonian and lower eigenvector
// ─────────────────────────────────────────────────────────────────────────────

// H(kx,ky,kz) = dx*σx + dy*σy + dz*σz stored row-major as H[4]:
//   H = [[dz, dx-i*dy],[dx+i*dy, -dz]]
static void weyl_ham(double kx, double ky, double kz, double m, cx H[4]) {
    double dx = std::sin(kx);
    double dy = std::sin(ky);
    double dz = m - std::cos(kx) - std::cos(ky) - std::cos(kz);
    H[0] = cx{dz};
    H[1] = cx{dx, -dy};
    H[2] = cx{dx,  dy};
    H[3] = cx{-dz};
}

// Analytical lower eigenvector of 2×2 Hermitian H = [[A,B],[B*,C]].
// q ∝ (-B*, E- - A) normalized; handle degeneracy.
static void lower_evec_2x2(const cx H[4], cx q[2]) {
    double A = H[0].real(), C = H[3].real();
    cx B = H[1];
    double avg = (A + C) * 0.5;
    double r   = std::sqrt(((A - C) * 0.5) * ((A - C) * 0.5) + std::norm(B));
    double Em  = avg - r;
    cx q0 = -std::conj(B);
    cx q1 = cx{Em - A};
    double nm = std::sqrt(std::norm(q0) + std::norm(q1));
    if (nm < 1e-14) {
        q[0] = cx{1}; q[1] = cx{0};
        return;
    }
    q[0] = q0 / nm;
    q[1] = q1 / nm;
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  Fukui-Hatsugai-Suzuki Chern number on n_k × n_k kx-ky grid at fixed kz
// ─────────────────────────────────────────────────────────────────────────────

static int chern_kz_slice(double kz, double m, int n_k) {
    // kx, ky ∈ [-π, π) with spacing 2π/n_k
    const double dk = 2.0 * M_PI / n_k;

    // Store lower eigenvectors q[i,j] (2-component) in flat array
    // Index: (i * n_k + j) * 2 + spin
    std::vector<cx> vecs(n_k * n_k * 2);

    for (int i = 0; i < n_k; ++i) {
        double kx = -M_PI + i * dk;
        for (int j = 0; j < n_k; ++j) {
            double ky = -M_PI + j * dk;
            cx H[4];
            weyl_ham(kx, ky, kz, m, H);
            cx q[2];
            lower_evec_2x2(H, q);
            int base = (i * n_k + j) * 2;
            vecs[base + 0] = q[0];
            vecs[base + 1] = q[1];
        }
    }

    // Link variable U = <q|q'>/ |<q|q'>|
    auto link = [&](int i0, int j0, int i1, int j1) -> cx {
        int b0 = (i0 * n_k + j0) * 2;
        int b1 = (i1 * n_k + j1) * 2;
        cx z = std::conj(vecs[b0]) * vecs[b1] +
               std::conj(vecs[b0 + 1]) * vecs[b1 + 1];
        double az = std::abs(z);
        return (az > 1e-12) ? z / az : cx{1.0};
    };

    double F_sum = 0.0;
    for (int i = 0; i < n_k; ++i) {
        for (int j = 0; j < n_k; ++j) {
            int ip = (i + 1) % n_k;
            int jp = (j + 1) % n_k;
            cx U1  = link(i,  j,  ip, j);
            cx U2  = link(i,  j,  i,  jp);
            cx U1p = link(i,  jp, ip, jp);
            cx U2p = link(ip, j,  ip, jp);
            // F = arg(U1 * U2p * conj(U1p) * conj(U2))
            cx prod = U1 * U2p * std::conj(U1p) * std::conj(U2);
            F_sum += std::arg(prod);
        }
    }
    return static_cast<int>(std::round(F_sum / (2.0 * M_PI)));
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  General N×N complex Hermitian Jacobi eigensolver (reused from edge_states)
// ─────────────────────────────────────────────────────────────────────────────

static void jacobi_herm(int N, std::vector<cx>& A,
                        std::vector<double>& eigvals,
                        std::vector<cx>& eigvecs) {
    eigvecs.assign(N * N, cx{0});
    for (int i = 0; i < N; ++i) eigvecs[i * N + i] = cx{1.0};

    const int max_sweeps = 30;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0;
        for (int p = 0; p < N; ++p)
            for (int q = p + 1; q < N; ++q)
                off += std::norm(A[p * N + q]);
        if (off < 1e-26) break;

        for (int p = 0; p < N - 1; ++p) {
            for (int q = p + 1; q < N; ++q) {
                cx apq = A[p * N + q];
                double r_apq = std::abs(apq);
                if (r_apq < 1e-15) continue;

                double app = A[p * N + p].real();
                double aqq = A[q * N + q].real();
                double tau = (aqq - app) / (2.0 * r_apq);
                double t = (tau >= 0) ?  1.0 / (tau + std::sqrt(1.0 + tau * tau))
                                      : -1.0 / (-tau + std::sqrt(1.0 + tau * tau));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = c * t;
                cx ph = apq / r_apq;

                for (int r = 0; r < N; ++r) {
                    cx arp = A[r * N + p], arq = A[r * N + q];
                    A[r * N + p] = cx{c} * arp - cx{s} * std::conj(ph) * arq;
                    A[r * N + q] = cx{s} * ph  * arp + cx{c} * arq;
                }
                for (int r = 0; r < N; ++r) {
                    cx apr = A[p * N + r], aqr = A[q * N + r];
                    A[p * N + r] = cx{c} * apr - cx{s} * ph * aqr;
                    A[q * N + r] = cx{s} * std::conj(ph) * apr + cx{c} * aqr;
                }
                for (int r = 0; r < N; ++r) {
                    cx vrp = eigvecs[r * N + p], vrq = eigvecs[r * N + q];
                    eigvecs[r * N + p] = cx{c} * vrp - cx{s} * std::conj(ph) * vrq;
                    eigvecs[r * N + q] = cx{s} * ph  * vrp + cx{c} * vrq;
                }
            }
        }
    }

    eigvals.resize(N);
    for (int i = 0; i < N; ++i) eigvals[i] = A[i * N + i].real();

    for (int i = 1; i < N; ++i) {
        double key = eigvals[i];
        std::vector<cx> col(N);
        for (int r = 0; r < N; ++r) col[r] = eigvecs[r * N + i];
        int j = i - 1;
        while (j >= 0 && eigvals[j] > key) {
            eigvals[j + 1] = eigvals[j];
            for (int r = 0; r < N; ++r) eigvecs[r * N + j + 1] = eigvecs[r * N + j];
            --j;
        }
        eigvals[j + 1] = key;
        for (int r = 0; r < N; ++r) eigvecs[r * N + j + 1] = col[r];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// D.  Slab Hamiltonian (2*n_z) × (2*n_z) and edge weight
// ─────────────────────────────────────────────────────────────────────────────

// Build slab Hamiltonian for periodic (kx,ky), open z boundary conditions.
// H_onsite = sin(kx)σx + sin(ky)σy + (m - cos(kx) - cos(ky))σz
// H_hop (forward z) = -(1/2)σz  →  [[−1/2, 0],[0, 1/2]]
static void build_slab(double kx, double ky, double m, int n_z,
                       std::vector<cx>& Hmat) {
    const int NB = 2;   // block size (2 spin states per layer)
    const int N  = NB * n_z;
    Hmat.assign(N * N, cx{0});

    double dx   = std::sin(kx);
    double dy   = std::sin(ky);
    double dz   = m - std::cos(kx) - std::cos(ky);

    // On-site 2×2 block: [[dz, dx-i*dy],[dx+i*dy, -dz]]
    cx Hon[4];
    Hon[0] = cx{dz};
    Hon[1] = cx{dx, -dy};
    Hon[2] = cx{dx,  dy};
    Hon[3] = cx{-dz};

    // Forward z-hopping 2×2 block: -(1/2)σz = [[-1/2, 0],[0, 1/2]]
    cx Hhop[4];
    Hhop[0] = cx{-0.5};
    Hhop[1] = cx{0};
    Hhop[2] = cx{0};
    Hhop[3] = cx{0.5};

    // Place diagonal blocks (on-site)
    for (int iz = 0; iz < n_z; ++iz) {
        int base = iz * NB;
        for (int r = 0; r < NB; ++r)
            for (int c = 0; c < NB; ++c)
                Hmat[(base + r) * N + (base + c)] = Hon[r * NB + c];
    }

    // Place off-diagonal blocks (hoppings)
    for (int iz = 0; iz < n_z - 1; ++iz) {
        int bi  = iz * NB;
        int bi1 = (iz + 1) * NB;
        for (int r = 0; r < NB; ++r) {
            for (int c = 0; c < NB; ++c) {
                // |iz><iz+1|: Hhop
                Hmat[(bi + r) * N + (bi1 + c)] = Hhop[r * NB + c];
                // |iz+1><iz|: Hhop† (Hhop is real diagonal → same)
                Hmat[(bi1 + r) * N + (bi + c)] = std::conj(Hhop[c * NB + r]);
            }
        }
    }
}

// Edge weight of eigenvector (column col of row-major eigvecs, N = 2*n_z).
// = probability on first layer (sites 0,1) + last layer (sites 2*n_z-2, 2*n_z-1).
static double edge_weight_2band(const std::vector<cx>& eigvecs, int N, int col) {
    double w = std::norm(eigvecs[0 * N + col]) +
               std::norm(eigvecs[1 * N + col]) +
               std::norm(eigvecs[(N - 2) * N + col]) +
               std::norm(eigvecs[(N - 1) * N + col]);
    return std::min(w, 1.0);
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

WeylResult run_weyl(const WeylOptions& opts) {
    auto t0 = std::chrono::steady_clock::now();

    if (opts.n_k < 4)
        throw std::runtime_error("Weyl: n_k must be >= 4");
    if (opts.n_k > 60)
        throw std::runtime_error("Weyl: n_k must be <= 60");
    if (opts.n_z < 4)
        throw std::runtime_error("Weyl: n_z must be >= 4");
    if (opts.n_z > 60)
        throw std::runtime_error("Weyl: n_z must be <= 60");
    if (opts.n_kz < 2)
        throw std::runtime_error("Weyl: n_kz must be >= 2");

    const double m  = opts.m_param;
    const int n_k   = opts.n_k;
    const int n_z   = opts.n_z;
    const int n_kz  = opts.n_kz;

    WeylResult res;
    res.m_param  = m;
    res.n_k      = n_k;
    res.n_z      = n_z;
    res.scenario = opts.scenario;

    // ── Scenario validation ───────────────────────────────────────────────
    if (opts.scenario != "fermi_arc" &&
        opts.scenario != "chern_sweep" &&
        opts.scenario != "both") {
        throw std::runtime_error(
            "Weyl: unknown scenario \"" + opts.scenario +
            "\"; valid values: fermi_arc, chern_sweep, both");
    }

    // ── Weyl node positions ───────────────────────────────────────────────
    double val = m - 2.0;
    double node_kz = 0.0;
    bool has_nodes = (std::abs(val) < 1.0);
    if (has_nodes) {
        node_kz = std::acos(val);
        res.node_kz_pos =  node_kz;
        res.node_kz_neg = -node_kz;
    } else {
        res.node_kz_pos = std::numeric_limits<double>::quiet_NaN();
        res.node_kz_neg = std::numeric_limits<double>::quiet_NaN();
    }

    // ── Phase label ───────────────────────────────────────────────────────
    std::string phase;
    if (has_nodes) {
        // Format: "Weyl semimetal (kz0=±X.XXX)"
        char buf[64];
        std::snprintf(buf, sizeof(buf), "Weyl semimetal (kz\xE2\x82\x80=\xC2\xB1%.3f)", node_kz);
        phase = buf;
    } else {
        phase = "Trivial (no Weyl nodes)";
    }

    // ── Bulk Weyl cone spectrum: E±(0,0,kz) ──────────────────────────────
    {
        const int n_cone = 80;
        res.cone_kz.resize(n_cone);
        res.cone_E_plus.resize(n_cone);
        res.cone_E_minus.resize(n_cone);
        for (int i = 0; i < n_cone; ++i) {
            double kz = -M_PI + 2.0 * M_PI * i / (n_cone - 1);
            res.cone_kz[i] = kz;
            double dz = m - 1.0 - 1.0 - std::cos(kz);  // kx=ky=0 → cos(0)=1 each
            double e = std::sqrt(dz * dz);  // dx=dy=0 at kx=ky=0
            res.cone_E_plus[i]  =  e;
            res.cone_E_minus[i] = -e;
        }
    }

    // ── Chern number sweep over kz ────────────────────────────────────────
    {
        res.kz_vals.resize(n_kz);
        res.chern_kz.resize(n_kz);
        for (int ik = 0; ik < n_kz; ++ik) {
            double kz = -M_PI + 2.0 * M_PI * ik / (n_kz - 1);
            res.kz_vals[ik]  = kz;
            res.chern_kz[ik] = chern_kz_slice(kz, m, n_k);
        }
    }

    // ── Surface Fermi arc (slab diagonalization) ─────────────────────────
    if (opts.scenario == "fermi_arc" || opts.scenario == "both") {
        const int N_slab = 2 * n_z;
        const double dk  = 2.0 * M_PI / n_k;
        const double ef  = opts.fermi_energy;

        res.arc_kx.reserve(n_k * n_k);
        res.arc_ky.reserve(n_k * n_k);
        res.arc_weight.reserve(n_k * n_k);
        res.arc_energy.reserve(n_k * n_k);

        for (int ix = 0; ix < n_k; ++ix) {
            double kx = -M_PI + ix * dk;
            for (int iy = 0; iy < n_k; ++iy) {
                double ky = -M_PI + iy * dk;

                std::vector<cx> Hmat;
                build_slab(kx, ky, m, n_z, Hmat);

                std::vector<double> evals;
                std::vector<cx> evecs;
                jacobi_herm(N_slab, Hmat, evals, evecs);

                // Find eigenstate closest to Fermi energy
                int idx = 0;
                double best = std::abs(evals[0] - ef);
                for (int b = 1; b < N_slab; ++b) {
                    double d = std::abs(evals[b] - ef);
                    if (d < best) { best = d; idx = b; }
                }
                double w = edge_weight_2band(evecs, N_slab, idx);

                res.arc_kx.push_back(kx);
                res.arc_ky.push_back(ky);
                res.arc_weight.push_back(w);
                res.arc_energy.push_back(evals[idx]);
            }
        }
        res.n_arc_pts = n_k * n_k;
    }

    // ── m_param sweep: node kz position vs m ─────────────────────────────
    {
        int nsw = opts.n_sweep;
        if (nsw < 2) nsw = 2;
        res.sweep_vals.resize(nsw);
        res.sweep_node_kz.resize(nsw);
        res.sweep_phase.resize(nsw);
        for (int k = 0; k < nsw; ++k) {
            double ms = opts.param_min +
                        (opts.param_max - opts.param_min) * k / (nsw - 1);
            res.sweep_vals[k] = ms;
            double vs = ms - 2.0;
            if (std::abs(vs) < 1.0) {
                res.sweep_node_kz[k] = std::acos(vs);
                res.sweep_phase[k]   = "Weyl semimetal";
            } else {
                res.sweep_node_kz[k] = std::numeric_limits<double>::quiet_NaN();
                res.sweep_phase[k]   = "Trivial/Insulator";
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.message    = "m=" + std::to_string(m) + ", " + phase +
                     ", n_k=" + std::to_string(n_k) +
                     ", n_z=" + std::to_string(n_z);
    res.ok = true;
    return res;
}

}  // namespace qx
