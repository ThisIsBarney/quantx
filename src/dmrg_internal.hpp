#pragma once
// Internal header for DMRG implementation -- not part of the public API.
// Include this from dmrg_mpo.cpp, dmrg_dense.cpp, dmrg_blocks.cpp, dmrg_capi.cpp.

#include <complex>
#include <map>
#include <string>
#include <vector>
#include <array>
#include <stdexcept>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

namespace qx_dmrg {

using Cx   = std::complex<double>;
using Mat  = Eigen::MatrixXcd;
using Vec  = Eigen::VectorXcd;
using RVec = Eigen::VectorXd;
using Mat2 = Eigen::Matrix2cd;
using nlj  = nlohmann::json;

// -- Shared data structures ---------------------------------------------------

// MPO site tensor: ops[alpha][beta] is a d x d matrix W^{alpha,beta}_{s,s'} acting
// on the local Hilbert space. d=2 (spin-1/2) by default; d>2 for spin-1, Hubbard,
// etc. phys_charge[s] is the U(1)/Sz charge of local state s in 2*Sz integer units
// (spin-1/2 = {+1,-1}, spin-1 = {+2,0,-2}) -- the block-sparse path needs it.
struct MPOSite {
    int chi_l, chi_r;
    int d = 2;
    std::vector<int> phys_charge{+1, -1};
    std::vector<Mat> ops;   // each is d x d
    const Mat& W(int a, int b) const { return ops[a * chi_r + b]; }
    Mat& W(int a, int b)             { return ops[a * chi_r + b]; }
};

// The DMRG/TDVP contraction kernels still hard-loop the two spin-1/2 states (the
// d>2 contractions are phase 2b). Until then, fail loud at the public entry points
// rather than silently truncating a d>2 MPO to its s<2 subspace.
inline void require_local_dim2(const std::vector<MPOSite>& mpo, const char* where) {
    for (const MPOSite& W : mpo)
        if (W.d != 2)
            throw std::runtime_error(std::string(where) + ": local dimension d=" +
                std::to_string(W.d) + " not yet supported (phase 2a generalizes the MPO "
                "representation only; the contraction kernels are still d=2)");
}

// The arbitrary-d dense kernels read a single local dimension (mpo[0].d) and fold
// every two-site theta with it, so a non-uniform chain would mis-stride silently.
// All physical models are uniform; fail loud if that ever breaks.
inline void require_uniform_local_dim(const std::vector<MPOSite>& mpo, const char* where) {
    for (const MPOSite& W : mpo)
        if (W.d != mpo[0].d)
            throw std::runtime_error(std::string(where) + ": non-uniform local dimension "
                "across the chain is not supported");
}

struct MPS {
    int n;
    std::vector<Mat> A;    // A[i]: (chi_L*2, chi_R), left-canonical
    std::vector<RVec> sv;  // sv[i]: singular values at bond i
};

struct DMRGResult {
    double energy = 0;
    double truncation_error = 0;
    std::vector<double> sweep_energies;
    int n_sweeps_done = 0;
    double wall_time_s = 0;
};

// -- Function declarations (definitions in respective .cpp files) -------------

// dmrg_mpo.cpp
Mat2 pauli(char c);
// XXZ chain: H = J sum_i [ (X_i X_{i+1} + Y_i Y_{i+1}) + delta Z_i Z_{i+1} ]
// (delta=1 is the isotropic Heisenberg point). Same Sz bond structure as the
// plain Heisenberg MPO, so the block-sparse engine handles it unchanged.
std::vector<MPOSite> build_xxz_mpo(int n, double J = 1.0, double delta = 1.0);
inline std::vector<MPOSite> build_heisenberg_mpo(int n, double J = 1.0) {
    return build_xxz_mpo(n, J, 1.0);
}
// Total spin-current operator J_tot = sum_l i*J(sigma+_l sigma-_{l+1} - h.c.).
std::vector<MPOSite> build_current_mpo(int n, double J = 1.0);
// XXZ Hamiltonian on the *interleaved purified* chain of 2*n_phys sites: physical
// sites at even positions, ancillas at odd. Even sites carry the standard XXZ
// W (couplings between physical neighbours pass over the intervening ancilla);
// odd (ancilla) sites are a diagonal identity on all 5 MPO bond states, so the
// Sz bond structure (HB_DELTA) -- and therefore every block-sparse environment
// kernel -- is unchanged. H acts on physical sites only: ancillas are spectators.
std::vector<MPOSite> build_xxz_purified_mpo(int n_phys, double J = 1.0,
                                            double delta = 1.0);
// Spin-1 (d=3) Heisenberg on the interleaved purified chain (physical even, ancilla
// identity pass-through) -- the d>2 analogue of build_xxz_purified_mpo.
std::vector<MPOSite> build_spin1_purified_mpo(int n_phys, double J = 1.0);
// Total spin-current operator on the interleaved purified chain (physical sites
// even, ancilla identity pass-through), for finite-T transport. Same HB_DELTA
// bond structure as build_xxz_purified_mpo, so apply_mpo_to_bmps applies it
// unchanged. J_tot = sum_l i*J(sigma+_{2l} sigma-_{2l+2} - h.c.) over physical l.
std::vector<MPOSite> build_current_purified_mpo(int n_phys, double J = 1.0);

// J1-J2 (next-nearest-neighbour) XXZ chain:
//   H = sum_i [ J1 h_{i,i+1} + J2 h_{i,i+2} ],  h = XX+YY+delta ZZ.
// An 8-wide Sz-conserving MPO (each emitted operator can complete at range 1 or,
// after passing one identity, range 2) -- the first model that needs the general
// derived charge array (its bond structure is wider than Heisenberg's 5).
std::vector<MPOSite> build_j1j2_mpo(int n, double J1 = 1.0, double J2 = 0.0,
                                    double delta = 1.0);

// Spin-1 Heisenberg chain: H = J sum_i S_i.S_{i+1} with the d=3 spin-1 operators.
// The first model with local dimension > 2 (phys_charge = {+2,0,-2}); validates
// the arbitrary-local-dimension generalization of MPOSite / derive_mpo_sym.
std::vector<MPOSite> build_spin1_heisenberg_mpo(int n, double J = 1.0);

// -- U(1)/Sz symmetry of an MPO (for the block-sparse path) -------------------
// delta[b] = (bra accumulated Sz) - (ket accumulated Sz) carried by MPO bond b.
struct MPOSym {
    int chi = 1;                  // MPO bulk bond dimension
    std::vector<int> delta{0};    // per-bond-index Sz charge (sigma units)
};

// Derive the per-bond Sz charge from a uniform-bulk Sz-conserving MPO by
// propagating  delta[b] = delta[a] + sz(s) - sz(s')  over every nonzero
// W(a,b)(s,s')  (sz(0)=+1 up, sz(1)=-1 dn), anchored at delta[0]=0 (the standard
// lower-triangular 'done'/right-vacuum bond). Throws std::runtime_error if the
// MPO is not Sz-conserving (a bond carrying mixed/inconsistent charge). It
// reproduces HB_DELTA={0,+2,-2,0,0} for the Heisenberg MPO; for a wider MPO
// (J1-J2, longer range) it yields the charge array the block kernels consume.
inline MPOSym derive_mpo_sym(const std::vector<MPOSite>& mpo) {
    int bulk = -1, w = 0;                       // a full-width (square) bulk site
    for (int i = 0; i < (int)mpo.size(); ++i)
        if (mpo[i].chi_l == mpo[i].chi_r && mpo[i].chi_l >= w) { w = mpo[i].chi_l; bulk = i; }
    if (bulk < 0)   // n<=2: only boundary sites (chi_l!=chi_r) -> no bulk to read
        throw std::runtime_error("derive_mpo_sym: no square bulk MPO site (chain too short, need n>=3)");
    const MPOSite& W = mpo[bulk];
    const int chi = W.chi_r;
    const int ld = W.d;                          // local Hilbert-space dimension
    const std::vector<int>& qphys = W.phys_charge;   // Sz charge of each local state
    if ((int)qphys.size() != ld || W.W(0, 0).rows() != ld)
        throw std::runtime_error("derive_mpo_sym: MPOSite invariant violated "
                                 "(d, phys_charge size, and op rows must all agree)");

    std::vector<std::array<int,3>> edges;       // (a,b,c): bond a->b carries charge c
    for (int a = 0; a < chi; ++a)
        for (int b = 0; b < chi; ++b) {
            const Mat& m = W.W(a, b);
            int c = 0; bool any = false;
            for (int s = 0; s < ld; ++s)
                for (int sp = 0; sp < ld; ++sp)
                    if (std::abs(m(s, sp)) > 1e-13) {
                        int cc = qphys[s] - qphys[sp];   // bra charge - ket charge
                        if (!any) { c = cc; any = true; }
                        else if (cc != c)
                            throw std::runtime_error("derive_mpo_sym: MPO not Sz-conserving "
                                                     "(bond carries mixed charge)");
                    }
            if (any) edges.push_back({a, b, c});
        }

    MPOSym sym; sym.chi = chi; sym.delta.assign(chi, 0);
    std::vector<char> known(chi, 0); known[0] = 1;     // anchor delta[0]=0
    bool changed = true; int guard = 0;
    while (changed && guard++ <= chi + 1) {
        changed = false;
        for (auto& e : edges) {
            int a = e[0], b = e[1], d = e[2];
            if (known[a] && !known[b])      { sym.delta[b] = sym.delta[a] + d; known[b] = 1; changed = true; }
            else if (known[b] && !known[a]) { sym.delta[a] = sym.delta[b] - d; known[a] = 1; changed = true; }
            else if (known[a] && known[b] && sym.delta[b] - sym.delta[a] != d)
                throw std::runtime_error("derive_mpo_sym: MPO not Sz-conserving "
                                         "(inconsistent bond charges)");
        }
    }
    return sym;  // bonds unreachable from b=0 keep charge 0 (unused states)
}
std::vector<MPOSite> build_pauli_sum_mpo(int n,
    const std::vector<std::pair<double, std::string>>& terms);

// dmrg_dense.cpp
MPS  init_zero_state(int n, int d = 2);
double compute_energy(const MPS& mps, const std::vector<MPOSite>& mpo);
DMRGResult run_dmrg(const std::vector<MPOSite>& mpo, int n,
                    int max_bond, double cutoff, int n_sweeps, double tol);

// dmrg_blocks.cpp
// Den Nijs-Rommelse string-order parameter of the ground state (the Haldane
// hidden-order signature): O = -<gs| Sz_i0 (prod_{i0<k<j0} e^{i pi Sz_k}) Sz_j0 |gs>.
double string_order_blocks(const std::vector<MPOSite>& mpo, int n, int i0, int j0,
                           int max_bond, double cutoff, int n_sweeps);
DMRGResult run_dmrg_blocks(const std::vector<MPOSite>& mpo, int n,
                           int max_bond, double cutoff, int n_sweeps, double tol);

// Block-sparse (Sz-symmetric) TDVP2 time evolution from the Neel state.
struct TDVPBlockResult {
    double energy = 0;
    std::vector<double> times, stag;   // staggered magnetization time series
    int max_bond = 0;
    double wall_time_s = 0;
};
TDVPBlockResult run_tdvp2_blocks(const std::vector<MPOSite>& mpo, int n,
                                 double dt, int n_steps, int chi_max,
                                 double cutoff, bool imag);

// Block-sparse TEBD2 (Trotter-gate) time evolution from the Neel state -- the
// fast nearest-neighbour route (no environments / Krylov). measure_stride: record
// staggered magnetization every this many steps (<=0 = endpoints only).
// J and delta parameterise the XXZ bond gate exp(tau*h), h = J(XX+YY)+J*delta*ZZ;
// they must match the mpo passed in (whose energy is measured at the end).
TDVPBlockResult run_tebd2_blocks(const std::vector<MPOSite>& mpo, int n,
                                 double dt, int n_steps, int chi_max,
                                 double cutoff, bool imag, int measure_stride = 1,
                                 double J = 1.0, double delta = 1.0);

// Dynamical spin structure factor data: time-dependent correlations
// C_{j,l0}(t) = <psi0| sigma^z_j e^{-i(H-E0)t} sigma^z_{l0} |psi0> from a central
// source site l0, ready for a space+time Fourier transform into S(k,omega).
struct SpectralResult {
    std::vector<double> times;                  // length n_steps+1
    int l0 = 0, n = 0, max_bond = 0;
    double E0 = 0, wall_time_s = 0;
    std::vector<std::vector<double>> C_re, C_im; // C_re[t][j], C_im[t][j]
};
SpectralResult run_spectral_blocks(const std::vector<MPOSite>& mpo, int n,
                                   double dt, int n_steps, int chi_max,
                                   double cutoff, int n_sweeps,
                                   double J = 1.0, double delta = 1.0);

// Loschmidt echo of a quench: ground state of mpo_i evolved under XXZ(Jf, deltaf).
// L(t) = |<psi0|psi(t)>|^2 (return probability); rate(t) = -ln L(t) / n (its
// non-analyticities are dynamical quantum phase transitions).
struct LoschmidtResult {
    std::vector<double> times, L, rate;
    int n = 0, max_bond = 0;
    double E0 = 0, wall_time_s = 0;
};
LoschmidtResult run_loschmidt_blocks(const std::vector<MPOSite>& mpo_i, int n,
                                     double dt, int n_steps, int chi_max,
                                     double cutoff, int n_sweeps,
                                     double Jf, double deltaf);

// Spin conductivity: current-current correlator C(t) = <psi0|J e^{-i(H-E0)t}J|psi0>
// (normalised by <psi0|psi0>), J the total spin current. sigma(omega) is its FFT.
struct ConductivityResult {
    std::vector<double> times, C_re, C_im;
    int n = 0, max_bond = 0;
    double E0 = 0, wall_time_s = 0;
};
ConductivityResult run_conductivity_blocks(const std::vector<MPOSite>& mpo_H, int n,
                                           double dt, int n_steps, int chi_max,
                                           double cutoff, int n_sweeps,
                                           double J = 1.0, double delta = 1.0);

// Finite-temperature statics via ancilla purification (thermofield double):
// rho(beta) = e^{-beta H}/Z represented as the pure state |psi(beta)> =
// e^{-beta H/2} |psi0> on an interleaved 2*n_phys chain (physical at even sites,
// ancilla at odd; |psi0> = product of (|ud>-|du>)/sqrt(2) Bell pairs = beta=0).
// Cooling is imaginary-time block-TEBD2 with swap gates ferrying physical
// neighbours together. Each step advances beta by dbeta. Per checkpoint
// (every measure_stride steps; <=0 = endpoints only) it records, in Pauli
// (sigma^z) units with H = J sum [XX+YY+delta ZZ]:
//   E       = <H>_beta = Tr(H e^{-beta H})/Z                (per chain, not per site)
//   sz[j]   = <Z_j>_beta            (physical sites only, j = 0..n_phys-1)
//   szsz[j] = <Z_c Z_j>_beta        (c = n_phys/2, central physical site)
//   chi_zz  = beta*(<M^2> - <M>^2)/n_phys,  M = sum_j Z_j   (only if with_chi;
//             costs n_phys overlap sweeps per checkpoint)
struct ThermalResult {
    std::vector<double> betas, E, chi_zz;
    std::vector<std::vector<double>> sz, szsz;   // [checkpoint][j_phys]
    int n_phys = 0, c = 0, max_bond = 0;
    double wall_time_s = 0;
};
ThermalResult run_thermal_blocks(int n_phys, double J, double delta,
                                 double dbeta, int n_steps, int chi_max,
                                 double cutoff, int measure_stride = 1,
                                 bool with_chi = true, int d = 2);

// Finite-temperature dynamical structure factor (F1): the purified-state
// analogue of run_spectral_blocks. The state is cooled to |psi(beta)> =
// e^{-beta H/2}|psi0> (n_cool imaginary-time steps), the central physical site
// l0 is hit with sigma^z, and BOTH the unsourced state A = e^{-iHt}|psi(beta)>
// and the sourced state B = e^{-iHt} sigma^z_{l0}|psi(beta)> are real-time
// evolved (the finite-T bra is not stationary, unlike T=0). The emitted
//   C_{j,l0}(t;T) = <A(t)| sigma^z_j |B(t)> / Z = <sigma^z_j(t) sigma^z_{l0}(0)>_beta
// is ready for a space+time FFT into S(k,omega,T); chi''(k,omega,T) =
// (1 - e^{-beta omega}) S(k,omega,T).
//
// disentangle: apply the Karrasch-Bardarson-Moore backward-ancilla evolution
// (auxiliary +H on the ancilla sublattice). H_anc commutes with H_phys and with
// the physical sigma^z operators, so the correlator is exactly invariant in the
// absence of truncation, but the operator-space entanglement -- hence the bond
// dimension at fixed accuracy -- is substantially reduced (->longer reachable t
// / lower T). bond_t[t] records the max bond dimension at each step (for the
// disentangler benchmark). All indices j are physical-site indices 0..n_phys-1.
struct SpectralThermalResult {
    std::vector<double> times;                   // length n_steps+1
    std::vector<std::vector<double>> C_re, C_im; // C_re[t][j], C_im[t][j]
    std::vector<int> bond_t;                     // max(bond_a, bond_b) per step
    std::vector<int> bond_a_t, bond_b_t;         // bond dims of A (unsourced) / B (sourced)
    int l0 = 0, n_phys = 0, max_bond = 0;
    double beta = 0, E = 0, wall_time_s = 0;
    bool disentangled = false;
};
SpectralThermalResult run_spectral_thermal_blocks(
    int n_phys, double J, double delta, double beta, int n_cool,
    double dt, int n_steps, int chi_max, double cutoff,
    bool disentangle = true);

// Finite-temperature spin conductivity (F2): the current-current correlator
//   C(t;T) = <J(t) J(0)>_beta = <psi(beta)| e^{iHt} J e^{-iHt} J |psi(beta)> / Z
// on the purified chain, J the total spin current. Its FFT is the optical
// conductivity sigma(omega,T); the t->inf plateau / zero-frequency weight is the
// Drude weight D(T) (the integrable-XXZ ballistic-vs-diffusive question). Like
// the structure factor, the finite-T bra is not stationary, so both the
// unsourced A = e^{-iHt}|psi> and the sourced B = e^{-iHt} J|psi> are real-time
// evolved; C(t) = <A(t)|J|B(t)> = <J A(t)|B(t)>. disentangle: KBM ancilla-backward
// gauge (cancels A's real-time growth -- a compute win, exactly as in F1).
struct ConductivityThermalResult {
    std::vector<double> times, C_re, C_im;
    std::vector<int> bond_t;
    int n_phys = 0, max_bond = 0;
    double beta = 0, E = 0, wall_time_s = 0;
    bool disentangled = false;
};
ConductivityThermalResult run_conductivity_thermal_blocks(
    int n_phys, double J, double delta, double beta, int n_cool,
    double dt, int n_steps, int chi_max, double cutoff,
    bool disentangle = true);

} // namespace qx_dmrg
