#pragma once
// Internal header for DMRG implementation -- not part of the public API.
// Include this from dmrg_mpo.cpp, dmrg_dense.cpp, dmrg_blocks.cpp, dmrg_capi.cpp.

#include <complex>
#include <map>
#include <string>
#include <vector>
#include <array>
#include <memory>
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
    // Per-site charge profiles of the converged state, filled only when
    // QX_PROFILES=1 (one extra non-destructive centre sweep). On the packed
    // two-U(1) path site_n = <N_i> and site_2sz = <2Sz_i>; on a plain single-U(1)
    // path site_n carries the plain charge (2Sz or N) and site_2sz stays 0.
    std::vector<double> site_n, site_2sz;
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
// Spin-1 (d=3) total spin current on the interleaved purified chain -- the d>2
// analogue of build_current_purified_mpo.
std::vector<MPOSite> build_spin1_current_purified_mpo(int n_phys, double J = 1.0);
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

// -- Packed two-U(1) charges ---------------------------------------------------
// The block engine's charge is ONE additive integer everywhere (it is only ever added
// and compared), so a product of two U(1)s packs into it: q = N*QX_Q2K + 2Sz.
// Component-wise addition IS integer addition, with no aliasing as long as the
// accumulated |2Sz| stays below QX_Q2K/2 (|2Sz| <= N <= n sites; fine below ~2000
// sites). Observables that read a CHARGE as a VALUE (center_sz / measure_stag) decode
// the N component when |q| is in the packed range -- see obs_of_charge.
constexpr int QX_Q2K = 4096;
inline int pack_q2(int N, int twoSz) { return N * QX_Q2K + twoSz; }
// Charge -> scalar observable: packed charges (|q| >= QX_Q2K/2) yield the particle
// number N; plain spin charges (small) pass through as the 2*Sz value they are.
inline double obs_of_charge(int q) {
    if (std::abs(q) < QX_Q2K / 2) return (double)q;      // plain single-U(1) charge
    return (double)((q + QX_Q2K / 2) / QX_Q2K);          // packed: N component (q >= 0 here)
}
// Both components: packed charges decode to (N, 2Sz); plain charges pass through
// as (q, 0) -- the two-component profile is meaningful on the packed path.
// Like obs_of_charge, assumes packed q >= 0 (N >= |2Sz| always holds for states).
inline void obs2_of_charge(int q, double& Nv, double& twoSzv) {
    if (std::abs(q) < QX_Q2K / 2) { Nv = (double)q; twoSzv = 0.0; return; }
    int N = (q + QX_Q2K / 2) / QX_Q2K;
    Nv = (double)N; twoSzv = (double)(q - N * QX_Q2K);
}

// Fermionic Hubbard chain (d=4): H = -t sum_{i,s}(c^dag_{i s}c_{i+1 s}+h.c.) + U sum_i n_up n_dn.
// The first FERMIONIC model -- Fermi statistics carried in the MPO via Jordan-Wigner
// (parity F on every travelling operator), so the bosonic block engine runs it unchanged.
// two_u1 = false: conserve particle number N only (phys_charge = {0,1,1,2}).
// two_u1 = true:  conserve N AND 2Sz via the packed charge (phys_charge =
//   {pack(0,0), pack(1,+1), pack(1,-1), pack(2,0)}) -- several times finer blocks
//   (TeNPy-granularity), the Hubbard performance path. Target sector: total_charge =
//   pack_q2(N_total, 2Sz_total), e.g. pack_q2(n, 0) for the half-filling singlet.
// mu: chemical potential (H gains -mu sum n_i); mu = U/2 is the particle-hole
// symmetric point that pins <n> = 1 at EVERY temperature (the half-filled Mott
// ensemble for the finite-T paths). Default 0 (all T=0 sector-targeted work).
std::vector<MPOSite> build_hubbard_mpo(int n, double t = 1.0, double U = 0.0,
                                       bool two_u1 = false, double mu = 0.0);
// Hubbard on the interleaved purified chain (physical even, ancilla identity odd) --
// finite-T. JW string threads the physical modes only (ancillas bosonic, plain
// identity carries + plain SWAP in the cooling). Packed two-U(1) charges.
std::vector<MPOSite> build_hubbard_purified_mpo(int n_phys, double t = 1.0, double U = 0.0,
                                                double mu = 0.0);
// Total charge current of the Hubbard chain on the purified chain (JW form, identity
// ancillas): J = it sum_{i,s}(c^dag_{i s}c_{i+1 s} - h.c.). For finite-T transport.
std::vector<MPOSite> build_hubbard_current_purified_mpo(int n_phys, double t = 1.0);

// Two-leg Hubbard LADDER via rung fusion (one d=16 site per rung -> a uniform
// nearest-neighbour chain; JW strings inside the 16x16 operators). The first
// quasi-2D model; packed two-U(1) charges. Half filling target: pack_q2(2*Lx, 0).
std::vector<MPOSite> build_hubbard_ladder_mpo(int Lx, double t = 1.0, double tperp = 1.0,
                                              double U = 0.0, double mu = 0.0);
// The same ladder SITE-SPLIT: 2*Lx Electron sites (d=4), period-2 uniform-chi MPO
// (leg hops range-2 with F carries). Less local work per sweep than the d=16 fusion;
// the performance formulation. Half filling target: pack_q2(2*Lx, 0).
std::vector<MPOSite> build_hubbard_ladder_split_mpo(int Lx, double t = 1.0, double tperp = 1.0,
                                                    double U = 0.0, double mu = 0.0);
// Width-W Hubbard strip / CYLINDER (y_periodic), site-split d=4, period-W MPO
// (column-major, x-hops range-W with F-carry chains). Isotropic t.
// Half filling target: pack_q2(W*Lx, 0).
// hpin: staggered Zeeman pinning on the x=0 edge column, H -= hpin sum_y (-1)^y
// Sz_{0,y} (even legs pinned up). Sz-diagonal, so both U(1) charges stay conserved;
// standard practice for exposing stripe/AF order in a symmetric DMRG ground state.
// tp: next-nearest-neighbour diagonal hopping (t-t'-U model), H -= tp on every
// (x,y)-(x+1,y+-1) bond (cylinder: y wraps mod W). chi grows 34 -> 82 for W=4.
// Width limit (fail-loud, chi <= 128): W <= 15 at tp=0, W <= 10 open / W <= 6
// cylinder with tp != 0 (chi = 20W+2 there).
// vamp/vlam: charge-modulation BIAS, H += vamp sum_i cos(2 pi (x_i - vlam/2)/vlam) n_i
// -- imprints a stripe pattern of wavelength vlam (holes gather at x = vlam/2 + k*vlam).
// n-diagonal, so both charges stay conserved. The biased GS is not a state of the
// physical H; measure the UNBIASED H on it (variational upper bound) to compare
// competing stripe patterns fairly.
std::vector<MPOSite> build_hubbard_lattice_split_mpo(int Lx, int W, double t = 1.0,
                                                     double U = 0.0, double mu = 0.0,
                                                     bool y_periodic = false,
                                                     double hpin = 0.0,
                                                     double tp = 0.0,
                                                     double vamp = 0.0, int vlam = 0);
// Singlet pair-field correlator MPO O = Delta^dag_{b1} Delta_{b2} on the width-W
// lattice; bond b = (x, y, orient) with orient 'y' (rung, i'=i+1, y < W-1) or 'x'
// (leg, i'=i+W). b1 must lie strictly left of b2. chi=7; N- and Sz-conserving
// (overall charge 0), so <GS|O|GS> is measurable in the GS sector directly.
std::vector<MPOSite> build_pair_corr_mpo(int n, int W, int x1, int y1, char o1,
                                         int x2, int y2, char o2);

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
    const int chi = mpo[bulk].chi_r;
    const int ld = mpo[bulk].d;                  // local Hilbert-space dimension
    const std::vector<int>& qphys = mpo[bulk].phys_charge;
    if ((int)qphys.size() != ld || mpo[bulk].W(0, 0).rows() != ld)
        throw std::runtime_error("derive_mpo_sym: MPOSite invariant violated "
                                 "(d, phys_charge size, and op rows must all agree)");

    // Collect charge-constraint edges from EVERY full-width bulk site, not just one:
    // a period-2 (or generally position-dependent) uniform-chi MPO -- e.g. the
    // site-split Hubbard ladder, whose A-sublattice W emits the rung/leg-a channels
    // and whose B-sublattice W emits the leg-b ones -- determines some bond charges
    // only through one sublattice. All sites share ONE bond space, so one delta array
    // must satisfy the union of constraints (conflicts still throw). For a uniform
    // MPO every site contributes identical edges: behaviour unchanged.
    int first = -1;
    for (int i = 0; i < (int)mpo.size(); ++i)
        if (mpo[i].chi_l == chi && mpo[i].chi_r == chi) { first = i; break; }
    std::vector<std::array<int,3>> edges;       // (a,b,c): bond a->b carries charge c
    for (int i = first; i < (int)mpo.size(); ++i) {
        const MPOSite& W = mpo[i];
        if (W.chi_l != chi || W.chi_r != chi) continue;    // boundary sites
        // EVERY full-width site contributes edges (previously capped at 8 -- enough
        // for period-W cylinder MPOs, but measurement MPOs like the pair-field
        // correlator activate their channel slots only at arbitrary positions deep
        // in the chain, so the full scan is required; for periodic MPOs the extra
        // sites contribute identical edges and behaviour is unchanged).
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
    }

    MPOSym sym; sym.chi = chi; sym.delta.assign(chi, 0);
    std::vector<char> known(chi, 0); known[0] = 1;     // anchor delta[0]=0

    // Boundary sites pin ABSOLUTE charges: the first site's single row is the
    // start slot (left vacuum, charge 0), the last site's single column the
    // done/closure slot (total operator charge -- 0 for an Sz-conserving MPO).
    // Bulk edges alone can leave slots unreachable (e.g. a measurement MPO whose
    // final completion lands on the last site); these anchors close the gap, and
    // for Hamiltonian MPOs they add redundant-but-consistent constraints only.
    auto block_charge = [&](const Mat& m, int& c) -> bool {   // false = zero block
        bool any = false;
        for (int s = 0; s < ld; ++s)
            for (int sp = 0; sp < ld; ++sp)
                if (std::abs(m(s, sp)) > 1e-13) {
                    int cc = qphys[s] - qphys[sp];
                    if (!any) { c = cc; any = true; }
                    else if (cc != c)
                        throw std::runtime_error("derive_mpo_sym: MPO not Sz-conserving "
                                                 "(bond carries mixed charge)");
                }
        return any;
    };
    auto pin = [&](int slot, int val) {
        if (known[slot] && sym.delta[slot] != val)
            throw std::runtime_error("derive_mpo_sym: MPO not Sz-conserving "
                                     "(inconsistent bond charges)");
        sym.delta[slot] = val; known[slot] = 1;
    };
    if (mpo.front().chi_l == 1 && mpo.front().chi_r == chi)
        for (int b = 0; b < chi; ++b) {
            int c;
            if (block_charge(mpo.front().W(0, b), c)) pin(b, c);
        }
    // NOTE: the -c pin assumes TOTAL operator charge 0 (true for every Hamiltonian
    // and for particle-conserving correlators like Delta^dag Delta). A deliberately
    // CHARGED operator MPO (e.g. a bare Delta^dag excitation) would need this
    // anchor offset by its total charge -- extend here if such an MPO ever exists.
    if (mpo.back().chi_l == chi && mpo.back().chi_r == 1)
        for (int a = 0; a < chi; ++a) {
            int c;
            if (block_charge(mpo.back().W(a, 0), c)) pin(a, -c);
        }

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
// total_charge: the fixed total U(1) charge of the target sector (0 for the Sz spin
// models; e.g. half-filling N=n for Hubbard). init_config (optional): a local-state
// index per site for the initial product state, summing to total_charge (else the
// default |0,d-1,0,...> alternation is used).
DMRGResult run_dmrg_blocks(const std::vector<MPOSite>& mpo, int n,
                           int max_bond, double cutoff, int n_sweeps, double tol,
                           int total_charge = 0,
                           const std::vector<int>* init_config = nullptr);

// -- VQE / quantum-circuit simulation on the block-sparse engine ---------------
// Hamiltonian Variational Ansatz (HVA), the standard fermionic VQE circuit:
//   |psi(theta)> = prod_l e^{-i th_U^l H_U} e^{-i th_odd^l H_odd} e^{-i th_even^l H_even} |psi_0>
// Each layer's angles are {even-bond hopping, odd-bond hopping, on-site U}. The bonds
// within one parity class commute, so its exponential factorizes EXACTLY into two-site
// gates (no Trotter error inside a layer) and the on-site layer is a diagonal one-site
// gate. The ansatz conserves N and 2Sz, so the U(1)xU(1) block machinery applies to the
// circuit -- exact simulation at sizes far past current quantum hardware, with the
// bond dimension reached measuring the VQE state's entanglement.
struct HVAResult {
    double energy = 0;                 // <psi(theta)| H |psi(theta)>
    int max_bond = 0;                  // largest bond dimension the circuit reached
    double wall_s = 0;
    std::vector<int> layer_bond;       // max bond after each layer (entanglement growth)
    // LARGEST bond entropy of the prepared state (max over every cut, not the middle
    // bond: odd n has no middle bond, and reading n/2-1 there reports a lopsided cut
    // as if it were a half cut). THE number that decides
    // classical simulability: an MPS needs chi ~ e^S, so S saturating with depth means
    // the ansatz stays classically cheap no matter how many qubits it uses, while S
    // growing toward the volume-law value (~n) is what a quantum computer would be
    // needed for. Read from a LOSSLESS centre walk, so it is not distorted by the
    // cutoff -- but it is still the entropy of the chi-capped state and is bounded by
    // ln(chi); trust it only while trunc_err stays far below the scale of interest.
    double S_max = 0;
    // Entropy at the BALANCED cut (equal site counts either side). Stays 0 for ODD n,
    // where no such cut exists -- bond n/2-1 there is the lopsided one that the old
    // half-chain reading got wrong, so it is deliberately NOT stored. Worth having
    // next to S_max because a maximum can sit on a BOUNDARY bond, where it is capped
    // by ln(d) of one site -- ln 16 for the fused ladder -- so comparing maxima
    // across geometries partly compares local dimensions, not the states.
    double S_mid = 0;
    // LARGEST single discarded weight over the circuit's ~2(n-1)p truncations, not
    // their sum: the accumulated error can be up to that factor larger.
    double trunc_err = 0;
};
// Prepared initial state for repeated circuit runs. VQE evaluates the SAME circuit
// from the SAME |psi_0> thousands of times, so preparing the state once per
// optimisation instead of once per energy evaluation is the difference between
// seconds and hours. Opaque handle: the block MPS type is internal to dmrg_blocks.cpp.
struct HVAInitState;
// |psi_0> = the DMRG ground state of init_mpo -- the standard HVA state preparation
// is the U=0 (non-interacting) ground state. This is structural, not cosmetic: every
// HVA generator commutes with S^2, so <S^2> is conserved along the entire circuit and
// a Neel product state (<S^2> = 3 at n=6) can never reach the singlet ground state.
// prep_out (optional): the state-preparation DMRG's own result. Its truncation is
// NOT included in the circuit's trunc_err, and preparing at a smaller chi than the
// circuit runs at is normal, so check it when n is large.
std::shared_ptr<HVAInitState> hva_prepare_init(const std::vector<MPOSite>& init_mpo,
                                               int n, int chi_max, int total_charge = 0,
                                               const std::vector<int>* init_config = nullptr,
                                               DMRGResult* prep_out = nullptr);
// want_entropy: the entropy walk is a full extra lossless sweep, so a VQE optimiser
// (which calls this thousands of times but reads S only from the final run) should
// leave it off -- it doubled the wall time of the depth scan when it was always on.
HVAResult run_hva_circuit(const std::vector<MPOSite>& mpo, int n,
                          const std::vector<std::array<double,3>>& angles,
                          int chi_max, double cutoff, int total_charge = 0,
                          const std::vector<int>* init_config = nullptr,
                          const std::shared_ptr<HVAInitState>& init = nullptr,
                          bool want_entropy = true);

// HVA circuit on a width-W cylinder (site-split, column-major i = W*x + y). The
// x-hops are range-W on the JW chain and the y-wrap range-(W-1), so every layer goes
// through the fermionic-swap network.
// Bonds are partitioned into COMMUTING GROUPS (greedy edge colouring: no two bonds in
// a group share a site), one angle per group per layer, so each group's exponential
// factorises exactly -- no Trotter error inside a layer, matching how the chain and
// ladder circuits are defined. The number of groups is geometry-dependent, so a
// width-W circuit has MORE parameters per layer than the chain's three; that favours
// the wider geometry, so a worse result there is the stronger claim.
// n_groups() reports the count so a caller can size its angle vectors.
int hva_lattice_groups(int Lx, int W, bool y_periodic);

// Number of lattice bonds (y-NN, y-wrap when W>2, x-hops) -- reported alongside the
// group count so the doc can state how much of the geometry each layer touches.
int hva_lattice_nbonds(int Lx, int W, bool y_periodic);
// The groups themselves, so a test can drive a dense reference with the SAME bonds in
// the SAME order (groups do not commute with each other). Re-deriving the greedy
// colouring in the test would risk diverging from what the circuit actually applies;
// the test's independence comes from building each group's Hamiltonian from explicit
// JW operators instead.
std::vector<std::vector<std::pair<int,int>>>
hva_lattice_group_bonds(int Lx, int W, bool y_periodic);
// PRECONDITION, checked: lattice_mpo's on-site term must be diagonal and the same at
// every site, and hop_mpo must share its local dimension and charge basis.
// PRECONDITION, NOT checked: lattice_mpo's hopping graph must be exactly the y-NN /
// y-wrap / x-hop set that lattice_bonds enumerates. A diagonal-hop MPO (tp != 0 in
// build_hubbard_lattice_split_mpo) has terms this circuit has no generator for, so the
// ansatz is strictly weaker than the Hamiltonian being measured -- still a valid
// variational upper bound, but not the ansatz the caller named, and nothing here says
// so. Every in-tree caller passes tp = 0.
HVAResult run_hva_lattice(const std::vector<MPOSite>& lattice_mpo,
                          const std::vector<MPOSite>& hop_mpo, int Lx, int W,
                          bool y_periodic,
                          const std::vector<std::vector<double>>& angles,
                          int chi_max, double cutoff, int total_charge = 0,
                          const std::vector<int>* init_config = nullptr,
                          const std::shared_ptr<HVAInitState>& init = nullptr,
                          bool want_entropy = true);

// ---- U(1) qubit circuits (M16) --------------------------------------------------------
// One-qubit phases and nearest-neighbour two-qubit gates that conserve the number of
// qubits in |1>, on the block-sparse MPS: every bond carries that number as an exact
// charge, so the total is conserved at any truncation. A gate is a row-major matrix:
// 2x2 on qubit q (it must be diagonal), or 4x4 on qubits (q, q+1) in the basis
// |s_q s_{q+1}> = |00>, |01>, |10>, |11> (qubit q the high bit). run_u1_circuit checks
// every gate unitary and charge-conserving to 1e-10, and refuses it otherwise.
struct U1Gate {
    int q = 0;
    int arity = 2;                          // 1: phase on q; 2: gate on (q, q+1)
    std::vector<std::complex<double>> u;    // 4 or 16 entries, row-major
};
struct U1Measurement {
    int after = 0;                          // measured after this many gates
    std::vector<double> z;                  // <Z_i>, normalized by the state's norm
    double norm2 = 1;                       // squared norm: the weight the truncations left
    int max_bond = 1;
};
struct U1CircuitResult {
    std::vector<U1Measurement> measurements;
    int max_bond = 1;                       // largest bond any gate produced
    double trunc_sum = 0;                   // discarded weight, summed over the gates
    double trunc_max = 0;                   // the largest single discard
    int ones = 0;                           // the conserved number of |1>s, read off the MPS
    double wall_s = 0;
};
U1CircuitResult run_u1_circuit(int n, const std::vector<int>& bits,
                               const std::vector<U1Gate>& gates,
                               const std::vector<int>& measure_after,
                               int chi_max, double cutoff);

// Cost probe: apply ONE layer of x-hops (all W*(Lx-1) range-W bonds) of a width-W
// cylinder to a chi-sized state through the swap network, and return the worst
// discarded weight. Exists so the width scan can be sized from a measurement.
double hva_lattice_layer_cost(const std::vector<MPOSite>& lattice_mpo,
                              const std::vector<MPOSite>& hop_mpo, int n, int W,
                              int Lx, int chi_max, int total_charge,
                              const std::vector<int>& init_config);

// TEST HOOK for the fermionic-swap network. Prepares the product state init_config,
// applies exp(-i theta * h) for the hopping bond between MODES i and j (any distance;
// a swap network brings them adjacent and back), and returns <psi| probe |psi>.
// Exposed so the ED test can check the primitive itself -- a long-range fermionic
// rotation is exactly where a JW sign error hides.
double hva_swap_probe(const std::vector<MPOSite>& hop_mpo,
                      const std::vector<MPOSite>& probe, int n, int i, int j,
                      double theta, int chi_max, int total_charge,
                      const std::vector<int>& init_config);

// One block DMRG ground-state run + <GS|O|GS> for a list of measurement MPOs
// (e.g. pair-field correlators): the GS is computed once, then for each obs MPO
// the environments are rebuilt and the two-site H_eff expectation evaluated.
// Real-MPO path only (fail-loud otherwise). res_out (optional) receives the full
// DMRGResult of the GS run (energy, and profiles when QX_PROFILES=1).
std::vector<double> gs_mpo_expectations(const std::vector<MPOSite>& H, int n,
                                        int max_bond, double cutoff, int n_sweeps,
                                        double tol, int total_charge,
                                        const std::vector<int>* init_config,
                                        const std::vector<std::vector<MPOSite>>& obs,
                                        DMRGResult* res_out = nullptr);

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
// total_charge / init_config: the GS sector (see run_dmrg_blocks). op_wts: per-local-
// state weights of the DIAGONAL correlator operator W (default = the charge itself,
// i.e. sigma^z for spin-1/2); packed-charge models must pass explicit weights, e.g.
// Hubbard spin {0,+1,-1,0} (=2Sz) or charge fluctuation {-1,0,0,+1} (=n-1, half filling).
SpectralResult run_spectral_blocks(const std::vector<MPOSite>& mpo, int n,
                                   double dt, int n_steps, int chi_max,
                                   double cutoff, int n_sweeps,
                                   double J = 1.0, double delta = 1.0,
                                   int total_charge = 0,
                                   const std::vector<int>* init_config = nullptr,
                                   const std::vector<double>* op_wts = nullptr);

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
                                 bool with_chi = true, int d = 2, double mu = 0.0);

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
    bool disentangle = true, int d = 2, double mu = 0.0);

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
    bool disentangle = true, int d = 2, double mu = 0.0);

} // namespace qx_dmrg
