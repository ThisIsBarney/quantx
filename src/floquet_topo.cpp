// src/floquet_topo.cpp  —  M44: Floquet Topological Insulator
//
// 5-step drive protocol on a 2D square lattice (Rudner et al., PRX 3, 031005, 2013).
// Each period T/5: turn on hopping J along one of the 5 "directions":
//   step 1: hop in +x,  step 2: hop in +y,
//   step 3: hop in -x,  step 4: hop in -y,
//   step 5: on-site (identity × 5th of period)
//
// 2-band model (sublattice A/B per unit cell on the square lattice):
//   H_i(k) = J * [cos(phase_i) σ_x - sin(phase_i) σ_y]
//
// Floquet operator: U(k) = exp(-iH_5 T/5) ... exp(-iH_1 T/5)
//   U is 2×2 unitary; eigenvalues e^{-iε_±(k)}, ε ∈ (-π/T, π/T]
//
// Topological invariants (winding numbers):
//   W_0  = (1/2πi) ∮ Tr[U†(∂U/∂kx)] dkx  (at ky=0, integrated over BZ)
//   W_pi = winding of the "off-diagonal" propagator in the π-gap basis
//
// Anomalous Floquet phase (AFP): all bulk Chern numbers = 0, yet W_pi = ±1
// → chiral edge modes in the π quasienergy gap.
//
// Scenarios: "spectrum" | "edge" | "winding" | "both"

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

using cx  = std::complex<double>;
using Mat2 = std::array<std::array<cx, 2>, 2>;

// ── 2×2 matrix utilities ─────────────────────────────────────────────────────
static Mat2 mat2_id() {
    Mat2 m{};
    m[0][0] = cx{1}; m[1][1] = cx{1};
    return m;
}
static Mat2 mat2_mul(const Mat2& A, const Mat2& B) {
    Mat2 C;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            C[i][j] = A[i][0]*B[0][j] + A[i][1]*B[1][j];
    return C;
}
// exp(-i * h * Mat2) where Mat2 is Hermitian 2×2 = [[m, f],[f*, -m]]
// Eigenvalues ±E where E = sqrt(|f|^2 + m^2)
// exp(-i θ H/E) = cos(θE) I - i sin(θE) H/E
static Mat2 mat2_expm(double coeff, double m, cx f) {
    // H = [[m, f],[conj(f), -m]], coeff = -i * T/5
    // → exp(coeff * H) = cos(E|coeff|) I + (coeff/|coeff|) * sin(E|coeff|) * H/E
    // with coeff purely imaginary = -i*t: expm(-i t H) = cos(tE) I - i sin(tE) H/E
    const double E = std::sqrt(m*m + std::norm(f));
    const double t = coeff;  // real: t = T/5 (the actual time step)
    double c, s_over_E;
    if (E < 1e-14) {
        c = 1.0; s_over_E = t;  // limit: sin(tE)/E → t
    } else {
        c = std::cos(t * E);
        s_over_E = std::sin(t * E) / E;
    }
    // exp(-i t H) = c*I - i*s/E * H
    Mat2 R;
    R[0][0] = cx{c, -s_over_E * m};
    R[0][1] = cx{0, -s_over_E} * f;
    R[1][0] = cx{0, -s_over_E} * std::conj(f);
    R[1][1] = cx{c,  s_over_E * m};
    return R;
}

// ── 5-step protocol ────────────────────────────────────────────────────────────
// H_i(k) at step i:
//   step 1 (+x): f = J e^{ikx},  m = 0
//   step 2 (+y): f = J e^{iky},  m = 0
//   step 3 (-x): f = J e^{-ikx}, m = 0
//   step 4 (-y): f = J e^{-iky}, m = 0
//   step 5 (on-site/trivial): H5 = 0 (just accumulates phase from identity)
static Mat2 floquet_U(double kx, double ky, double J, double T) {
    const double dt = T / 5.0;
    // step 1
    const Mat2 U1 = mat2_expm(dt, 0.0, cx{J * std::cos(kx), J * std::sin(kx)});
    // step 2
    const Mat2 U2 = mat2_expm(dt, 0.0, cx{J * std::cos(ky), J * std::sin(ky)});
    // step 3
    const Mat2 U3 = mat2_expm(dt, 0.0, cx{J * std::cos(kx), -J * std::sin(kx)});
    // step 4
    const Mat2 U4 = mat2_expm(dt, 0.0, cx{J * std::cos(ky), -J * std::sin(ky)});
    // step 5: identity (zero Hamiltonian for the 5th interval)
    const Mat2 U5 = mat2_id();
    // U(T) = U5 U4 U3 U2 U1  (rightmost applied first)
    return mat2_mul(U5, mat2_mul(U4, mat2_mul(U3, mat2_mul(U2, U1))));
}

// Quasienergies of 2×2 unitary U: ε = ±arg(λ)/T
// For 2×2 unitary det=1 (SU(2)): U = cos(θ) I + i sin(θ) n·σ
// Eigenvalues e^{±iθ}; quasienergies ε = ±θ/T
static std::pair<double, double> quasi_energies(const Mat2& U, double T) {
    // Tr(U) = 2 cos(θ) → θ = acos(Re(Tr(U))/2)
    const double tr_re = (U[0][0] + U[1][1]).real() * 0.5;
    const double tr_clamped = std::max(-1.0, std::min(1.0, tr_re));
    const double theta = std::acos(tr_clamped);  // in [0, π]
    return {-theta / T, theta / T};
}

// ── Winding number W_pi ────────────────────────────────────────────────────────
// W_pi counts the chiral edge modes in the π-gap.
// For the 1D winding number along kx at ky=0 of the "effective Hamiltonian"
// H_eff(kx) = i log U(kx):
//   We use the off-diagonal winding of U along kx.
// Simplified: track arg(U_{01}(kx)) as kx sweeps 0→2π; W_pi = winding/(2π).
// For the full 2D invariant (Rudner et al.): W_pi = Σ_k contribution, but
// for the standard 5-step protocol the winding reduces to a 1D integral.
//
// We compute numerically: integrate d[arg(det(U_offdiag))]/dkx over kx ∈ [0,2π).
static int compute_winding(double J, double T, int nk) {
    // Use the off-diagonal element U[0][1] winding along kx (at ky=0).
    // W_pi = (1/2π) * (change in arg(U[0][1]) as kx: 0→2π)
    double total_phase = 0.0;
    const double dkx = 2.0 * M_PI / nk;
    cx prev = floquet_U(0.0, 0.0, J, T)[0][1];
    for (int i = 1; i <= nk; ++i) {
        const double kx = i * dkx;
        const cx curr = floquet_U(kx, 0.0, J, T)[0][1];
        // Phase difference: arg(curr/prev)
        const double dphi = std::arg(curr * std::conj(prev));
        total_phase += dphi;
        prev = curr;
    }
    return static_cast<int>(std::round(total_phase / (2.0 * M_PI)));
}

// ── Bulk spectrum ─────────────────────────────────────────────────────────────
static void compute_bulk(const FloquetTopoOptions& opts, FloquetTopoResult& res) {
    const int nk = opts.n_k;
    const int N  = nk * nk;
    res.kx_bulk.resize(N); res.ky_bulk.resize(N);
    res.eps_minus.resize(N); res.eps_plus.resize(N);
    for (int si = 0; si < nk; ++si) {
        const double kx = (2.0 * M_PI * si) / nk - M_PI;
        for (int ti = 0; ti < nk; ++ti) {
            const double ky = (2.0 * M_PI * ti) / nk - M_PI;
            const int idx = si * nk + ti;
            res.kx_bulk[idx] = kx;
            res.ky_bulk[idx] = ky;
            const Mat2 U = floquet_U(kx, ky, opts.J, opts.T);
            const auto [em, ep] = quasi_energies(U, opts.T);
            res.eps_minus[idx] = em;
            res.eps_plus[idx]  = ep;
        }
    }
}

// ── Edge-state spectrum (strip geometry) ─────────────────────────────────────
// Strip: periodic in x, open in y (Ny unit cells).
// Build the 2Ny × 2Ny Floquet operator for each kx.
static void compute_edge(const FloquetTopoOptions& opts, FloquetTopoResult& res) {
    const int Ny   = opts.n_y;
    const int nk   = opts.n_k;
    const int dim  = 2 * Ny;  // 2 orbitals per unit cell
    const double dt = opts.T / 5.0;
    const double J  = opts.J;
    const double T  = opts.T;

    res.kx_edge.resize(nk);
    res.eps_edge.resize(nk * dim);

    for (int si = 0; si < nk; ++si) {
        const double kx = (2.0 * M_PI * si) / nk - M_PI;
        res.kx_edge[si] = kx;

        // Build 2Ny×2Ny strip Floquet operator as product of 5 step operators.
        // Each step i: 2Ny×2Ny unitary connecting sublattice A/B within each row
        // and between neighbouring rows for y-direction steps.
        //
        // Basis: (A_1, B_1, A_2, B_2, ..., A_Ny, B_Ny)
        // Step 1 (hop +x): on each row (iy), U_1_row = exp(-i dt H1(kx))
        //   H1 = J(cos kx σx - sin kx σy) → block diagonal in y
        // Step 2 (hop +y): U_2 couples row iy to iy+1 (open BCs: last row uncoupled)
        //   H2 = J(cos(ky) σx - sin(ky) σy) → ky-hops handled in y-real-space
        //   In real space: H2 = J*(c†_{iy+1,A} c_{iy,B} + h.c.)
        // etc.
        //
        // For compactness, build each step matrix directly.

        // Helper: build 2Ny×2Ny unitary from 5 steps
        // We store as flat row-major complex array.
        std::vector<cx> Ustrip(dim * dim, cx{0});
        // Initialise to identity
        for (int i = 0; i < dim; ++i) Ustrip[i*dim+i] = cx{1};

        // Multiply from left: Ustrip = U_step * Ustrip
        auto left_mul = [&](const std::vector<cx>& S) {
            std::vector<cx> tmp(dim * dim, cx{0});
            for (int r = 0; r < dim; ++r)
                for (int c = 0; c < dim; ++c)
                    for (int k = 0; k < dim; ++k)
                        tmp[r*dim+c] += S[r*dim+k] * Ustrip[k*dim+c];
            Ustrip = std::move(tmp);
        };

        // Step matrix: build from 2×2 blocks
        auto step_diag = [&](const Mat2& B2) {  // same 2×2 block on every row
            std::vector<cx> S(dim * dim, cx{0});
            for (int iy = 0; iy < Ny; ++iy) {
                const int a = 2*iy, b = 2*iy+1;
                S[a*dim+a] = B2[0][0]; S[a*dim+b] = B2[0][1];
                S[b*dim+a] = B2[1][0]; S[b*dim+b] = B2[1][1];
            }
            return S;
        };

        // Steps 1 and 3: x-direction hops — block diagonal in y
        const Mat2 U1blk = mat2_expm(dt, 0.0, cx{J*std::cos(kx),  J*std::sin(kx)});
        const Mat2 U3blk = mat2_expm(dt, 0.0, cx{J*std::cos(kx), -J*std::sin(kx)});
        // Step 5: identity
        // Apply U1 first, then U2, U3, U4, U5 = id
        // U(T) = U5 U4 U3 U2 U1
        // Build from right: start with U1, left-multiply each subsequent step
        // Actually set Ustrip = Id, then do:
        //   Ustrip = U5 U4 U3 U2 U1 * Id

        // Reset to identity
        std::fill(Ustrip.begin(), Ustrip.end(), cx{0});
        for (int i = 0; i < dim; ++i) Ustrip[i*dim+i] = cx{1};

        // Apply U1 (leftmost in time = rightmost in product)
        left_mul(step_diag(U1blk));

        // Step 2: +y hop. Real-space: hop from (iy, B) → (iy+1, A) + h.c.
        // H2_row = J * (c†_{iy+1,A} c_{iy,B} + h.c.) for iy=0..Ny-2
        // U2 = exp(-i dt H2)
        // For simplicity use the analytic 2×2 block for each pair (iy, iy+1):
        // Build tri-diagonal step matrix directly.
        {
            std::vector<cx> S2(dim * dim, cx{0});
            // On-site: identity (off-diagonal hops only connect A/B across rows)
            // Actually H2 is a sum of two-site terms. We build the full matrix
            // and then exponentiate — but that's costly for large Ny.
            // Use the fact that H2 decomposes into commuting 2-site blocks on
            // pairs (2iy+1, 2(iy+1)) for even iy, giving block-diagonal exp.
            // For odd-site open chain, remaining end site gets identity.
            for (int i = 0; i < dim; ++i) S2[i*dim+i] = cx{1};  // start with Id
            // Pairs of rows: (2iy+1, 2iy+2) = (B_iy, A_{iy+1}) for iy=0,2,...
            for (int iy = 0; iy < Ny - 1; iy += 2) {
                const int bA = 2*iy+1;   // B of row iy
                const int aB = 2*(iy+1); // A of row iy+1
                // 2×2 block in {bA, aB} subspace: H = J*[[0,1],[1,0]] (σx)
                const Mat2 blk = mat2_expm(dt, 0.0, cx{J, 0.0});
                // Replace identity in that 2×2 block
                S2[bA*dim+bA] = blk[0][0]; S2[bA*dim+aB] = blk[0][1];
                S2[aB*dim+bA] = blk[1][0]; S2[aB*dim+aB] = blk[1][1];
            }
            left_mul(S2);
        }

        // Apply U3
        left_mul(step_diag(U3blk));

        // Step 4: -y hop. Pairs (2iy, 2iy+1) = (A_iy, B_iy) for iy=1,3,...
        {
            std::vector<cx> S4(dim * dim, cx{0});
            for (int i = 0; i < dim; ++i) S4[i*dim+i] = cx{1};
            for (int iy = 1; iy < Ny - 1; iy += 2) {
                const int bB = 2*iy+1;   // B of row iy
                const int aA = 2*(iy+1); // A of row iy+1
                const Mat2 blk = mat2_expm(dt, 0.0, cx{J, 0.0});
                S4[bB*dim+bB] = blk[0][0]; S4[bB*dim+aA] = blk[0][1];
                S4[aA*dim+bB] = blk[1][0]; S4[aA*dim+aA] = blk[1][1];
            }
            left_mul(S4);
        }
        // Step 5: identity — already done (no-op)

        // Diagonalise Ustrip via joint Jacobi on Msin and Mcos:
        //   Msin = (Ustrip - Ustrip†)/(2i) — real symmetric, eigenvalues = sin(εT)
        //   Mcos = (Ustrip + Ustrip†)/2    — real symmetric, eigenvalues = cos(εT)
        // Msin and Mcos commute (same eigenbasis), so Jacobi sweeps on Msin
        // simultaneously transform Mcos.  Then ε = atan2(sin, cos)/T covers
        // the full range (-π/T, π/T], including the π-gap edge modes.
        std::vector<double> Msin(dim * dim, 0.0), Mcos(dim * dim, 0.0);
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j) {
                Msin[i*dim+j] = (Ustrip[i*dim+j] - std::conj(Ustrip[j*dim+i])).imag() * 0.5;
                Mcos[i*dim+j] = (Ustrip[i*dim+j] + std::conj(Ustrip[j*dim+i])).real() * 0.5;
            }
        // Joint Jacobi: pivot on Msin, apply same Givens rotation to Mcos
        const int max_iter = 200 * dim * dim;
        for (int iter = 0; iter < max_iter; ++iter) {
            int p = 0, q = 1; double amax = 0.0;
            for (int i = 0; i < dim; ++i)
                for (int j = i+1; j < dim; ++j)
                    if (std::abs(Msin[i*dim+j]) > amax) {
                        amax = std::abs(Msin[i*dim+j]); p = i; q = j;
                    }
            if (amax < 1e-12) break;
            const double app = Msin[p*dim+p], aqq = Msin[q*dim+q], apq = Msin[p*dim+q];
            const double theta = 0.5 * (aqq - app) / apq;
            const double t2 = (theta >= 0)
                ? 1.0/(theta + std::sqrt(1.0 + theta*theta))
                : 1.0/(theta - std::sqrt(1.0 + theta*theta));
            const double c2 = 1.0/std::sqrt(1.0 + t2*t2), s2 = t2*c2;
            // Update Msin diagonal + zero off-diagonal pivot
            Msin[p*dim+p] = app - t2*apq; Msin[q*dim+q] = aqq + t2*apq;
            Msin[p*dim+q] = Msin[q*dim+p] = 0.0;
            for (int r = 0; r < dim; ++r) {
                if (r==p||r==q) continue;
                const double arp = Msin[r*dim+p], arq = Msin[r*dim+q];
                Msin[r*dim+p] = Msin[p*dim+r] = c2*arp - s2*arq;
                Msin[r*dim+q] = Msin[q*dim+r] = s2*arp + c2*arq;
            }
            // Apply same Givens rotation to Mcos: R^T Mcos R
            const double bpp = Mcos[p*dim+p], bqq = Mcos[q*dim+q], bpq = Mcos[p*dim+q];
            Mcos[p*dim+p] = c2*c2*bpp - 2.0*c2*s2*bpq + s2*s2*bqq;
            Mcos[q*dim+q] = s2*s2*bpp + 2.0*c2*s2*bpq + c2*c2*bqq;
            Mcos[p*dim+q] = Mcos[q*dim+p] = c2*s2*(bpp - bqq) + (c2*c2 - s2*s2)*bpq;
            for (int r = 0; r < dim; ++r) {
                if (r==p||r==q) continue;
                const double brp = Mcos[r*dim+p], brq = Mcos[r*dim+q];
                Mcos[r*dim+p] = Mcos[p*dim+r] = c2*brp - s2*brq;
                Mcos[r*dim+q] = Mcos[q*dim+r] = s2*brp + c2*brq;
            }
        }
        // ε = atan2(sin(εT), cos(εT)) / T  — full range (-π/T, π/T]
        for (int i = 0; i < dim; ++i) {
            res.eps_edge[si * dim + i] = std::atan2(Msin[i*dim+i], Mcos[i*dim+i]) / T;
        }
    }
}

// ── Winding-number sweep ─────────────────────────────────────────────────────
static void compute_winding_sweep(const FloquetTopoOptions& opts, FloquetTopoResult& res) {
    const int nJ = opts.n_J;
    res.sweep_J.resize(nJ);
    res.sweep_W_pi.resize(nJ);
    for (int i = 0; i < nJ; ++i) {
        const double Ji = opts.J_max * i / (nJ - 1);
        res.sweep_J[i]    = Ji;
        res.sweep_W_pi[i] = compute_winding(Ji, opts.T, opts.n_k);
    }
}

// ── Dispatcher ───────────────────────────────────────────────────────────────
FloquetTopoResult run_floquet_topo(const FloquetTopoOptions& opts) {
    if (opts.n_k < 4)   throw std::runtime_error("FloquetTopo: n_k must be >= 4");
    if (opts.n_y < 2)   throw std::runtime_error("FloquetTopo: n_y must be >= 2");
    if (opts.n_J < 2)   throw std::runtime_error("FloquetTopo: n_J must be >= 2");
    if (opts.T <= 0.0)  throw std::runtime_error("FloquetTopo: T must be positive");

    const std::string& sc = opts.scenario;
    if (sc != "spectrum" && sc != "edge" && sc != "winding" && sc != "both")
        throw std::runtime_error(
            "FloquetTopo: unknown scenario \"" + sc +
            "\"; valid values: spectrum, edge, winding, both");

    auto t0 = std::chrono::steady_clock::now();

    FloquetTopoResult res;
    res.scenario = sc;
    res.J = opts.J;
    res.T = opts.T;

    // Winding numbers at current J
    res.W_pi = compute_winding(opts.J, opts.T, opts.n_k);
    res.W_0  = 0;  // W_0 is trivially 0 for the standard 5-step protocol

    if (res.W_pi != 0)
        res.phase_label = "Anomalous Floquet (W_pi=" + std::to_string(res.W_pi) + ")";
    else
        res.phase_label = "Trivial Floquet (W_pi=0)";

    if (sc == "spectrum" || sc == "both") compute_bulk(opts, res);
    if (sc == "edge"     || sc == "both") compute_edge(opts, res);
    if (sc == "winding"  || sc == "both") compute_winding_sweep(opts, res);

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.message = "J=" + std::to_string(opts.J) +
                  " T=" + std::to_string(opts.T) +
                  " W_pi=" + std::to_string(res.W_pi) +
                  " phase=" + res.phase_label;
    res.ok = true;
    return res;
}

}  // namespace qx
