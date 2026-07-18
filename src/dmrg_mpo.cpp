#include "dmrg_internal.hpp"

#include <algorithm>
#include <cmath>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qx_dmrg {

// -- Pauli matrices -----------------------------------------------------------

Mat2 pauli(char c) {
    Mat2 m;
    switch (c) {
        case 'X': m << 0,1, 1,0; break;
        case 'Y': m << 0,Cx(0,-1), Cx(0,1),0; break;
        case 'Z': m << 1,0, 0,-1; break;
        default:  m = Mat2::Identity(); break;
    }
    return m;
}

// -- MPO ----------------------------------------------------------------------

// Build Heisenberg MPO: H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
//
// Encoded in the QN-resolved (raising/lowering) basis so that each MPO bond index
// carries a definite ΔSz.  This is required by the Sz-conserving block-sparse path
// (run_dmrg_blocks, which assumes HB_DELTA = {0,+2,-2,0,0}); it is mathematically
// identical to the X/Y/Z form and the dense path is agnostic to the choice.
//
//   X X + Y Y = 2 (σ+ σ- + σ- σ+),   with σ+ = |↑⟩⟨↓|, σ- = |↓⟩⟨↑|
//
//  W = [ I     0      0     0    0 ]
//      [ σ-    0      0     0    0 ]   bond 1: completes 2σ+ σ-  (ΔSz = +2)
//      [ σ+    0      0     0    0 ]   bond 2: completes 2σ- σ+  (ΔSz = -2)
//      [ Z     0      0     0    0 ]   bond 3: completes Z Z     (ΔSz =  0)
//      [ 0   2J*σ+  2J*σ-  J*Z   I ]
//
// Left  boundary: row 4 (last row)
// Right boundary: col 0 (first col)
std::vector<MPOSite> build_xxz_mpo(int n, double J, double delta) {
    const int chi = 5;
    Mat2 sp; sp << 0, 1, 0, 0;   // σ+ = |↑⟩⟨↓|  (ket ↓ → bra ↑, ΔSz = +2)
    Mat2 sm; sm << 0, 0, 1, 0;   // σ- = |↓⟩⟨↑|  (ΔSz = -2)
    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        int cl = (i == 0)     ? 1 : chi;
        int cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl;
        site.chi_r = cr;
        site.ops.resize(cl * cr, Mat2::Zero());

        // Full 5x5 W matrix at bulk sites
        std::array<std::array<Mat2, 5>, 5> W5 {};
        for (auto& row : W5) for (auto& e : row) e.setZero();
        W5[0][0] = pauli('I');
        W5[1][0] = sm;
        W5[2][0] = sp;
        W5[3][0] = pauli('Z');
        W5[4][1] = (2.0 * J) * sp;            // Jxy = J  (XX + YY)
        W5[4][2] = (2.0 * J) * sm;
        W5[4][3] = (J * delta) * pauli('Z');  // Jz  = J*delta  (delta ZZ)
        W5[4][4] = pauli('I');

        // Map cl x cr indices into the 5x5 matrix
        for (int a = 0; a < cl; ++a) {
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 4 : a;
                int br = (i == n - 1) ? 0 : b;
                site.W(a, b) = W5[ar][br];
            }
        }
    }
    return mpo;
}

// Spin-1 Heisenberg chain MPO: H = J sum_i S_i . S_{i+1}, with the spin-1 (d=3)
// operators. S.S = (1/2)(S+ S- + S- S+) + Sz Sz. Same 5-wide bond structure as the
// spin-1/2 Heisenberg MPO (so derive_mpo_sym yields {0,+2,-2,0,0}), but the local
// dimension is 3 and the operators are 3x3 -- the first d>2 model on the engine.
// Basis (|+1>,|0>,|-1>), Sz charges (2*Sz) = {+2,0,-2}.
std::vector<MPOSite> build_spin1_heisenberg_mpo(int n, double J) {
    const int chi = 5, d = 3;
    const double r2 = std::sqrt(2.0);
    Mat Sp = Mat::Zero(3, 3); Sp(0, 1) = r2; Sp(1, 2) = r2;   // S+ raises Sz
    Mat Sm = Mat::Zero(3, 3); Sm(1, 0) = r2; Sm(2, 1) = r2;   // S- = (S+)^dag
    Mat Sz = Mat::Zero(3, 3); Sz(0, 0) = 1.0; Sz(2, 2) = -1.0; // diag(+1,0,-1)
    Mat I3 = Mat::Identity(3, 3);
    std::array<std::array<Mat, 5>, 5> W5;
    for (auto& row : W5) for (auto& e : row) e = Mat::Zero(3, 3);
    W5[0][0] = I3;
    W5[1][0] = Sm; W5[2][0] = Sp; W5[3][0] = Sz;
    W5[4][1] = (0.5 * J) * Sp;   // emit (J/2)S+, complete with S-  -> (J/2) S+ S-
    W5[4][2] = (0.5 * J) * Sm;   // emit (J/2)S-, complete with S+  -> (J/2) S- S+
    W5[4][3] = J * Sz;           // emit J Sz, complete with Sz     -> J Sz Sz
    W5[4][4] = I3;

    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        int cl = (i == 0) ? 1 : chi, cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        site.d = d; site.phys_charge = {+2, 0, -2};
        site.ops.assign(cl * cr, Mat::Zero(d, d));
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 4 : a, br = (i == n - 1) ? 0 : b;
                site.W(a, b) = W5[ar][br];
            }
    }
    return mpo;
}

// Fermionic Hubbard chain (d=4, the first FERMIONIC model):
//   H = -t sum_{i,sigma} (c^dag_{i sigma} c_{i+1 sigma} + h.c.) + U sum_i n_{i up} n_{i dn}
// Local basis {|0>, |up>, |dn>, |up dn>} with |up dn> = c^dag_up c^dag_dn |0>. The
// engine is bosonic (Sz/charge U(1) only, no anticommutation), so the Fermi statistics
// are carried by the MPO itself via Jordan-Wigner: every operator that "travels" along
// an MPO bond is dressed with the on-site parity F = (-1)^{n_up+n_dn} = diag(1,-1,-1,1),
// and the on-site annihilators already carry the intra-site up-before-dn sign
// (C_dn|up dn> = -|up>). One then gets the uniform matrix-tensor form
//   c^dag_{i s} c_{i+1 s} = (C^dag_s F)_i (x) (C_s)_{i+1}    (both spins)
// so the bosonic MPO contraction reproduces the fermionic H exactly (validated vs an
// explicit Jordan-Wigner ED in qx_hubbard_test). The conserved U(1) is the particle
// number N (phys_charge = {0,1,1,2}); derive_mpo_sym reads it exactly as it reads Sz.
//
// Bulk bond dim 6, lower-triangular: 0=done, {1,2,3,4}=the four travelling hop
// half-terms (up-fwd, dn-fwd, up-bwd, dn-bwd), 5=start.
std::vector<MPOSite> build_hubbard_mpo(int n, double t, double U, bool two_u1, double mu) {
    const int chi = 6, d = 4;
    Mat I4 = Mat::Identity(4, 4);
    Mat F  = Mat::Zero(4, 4); F(0,0)=1; F(1,1)=-1; F(2,2)=-1; F(3,3)=1;  // (-1)^{n_up+n_dn}
    Mat Nu = Mat::Zero(4, 4); Nu(1,1)=1; Nu(3,3)=1;                       // n_up
    Mat Nd = Mat::Zero(4, 4); Nd(2,2)=1; Nd(3,3)=1;                       // n_dn
    Mat Cup = Mat::Zero(4, 4); Cup(0,1)=1; Cup(2,3)=1;                    // annihilate up
    Mat Cdn = Mat::Zero(4, 4); Cdn(0,2)=1; Cdn(1,3)=-1;                   // annihilate dn (on-site JW sign)
    Mat Cupd = Cup.adjoint();  // create up: (1,0)=1,(3,2)=1
    Mat Cdnd = Cdn.adjoint();  // create dn: (2,0)=1,(3,1)=-1

    std::array<std::array<Mat, 6>, 6> W;
    for (auto& row : W) for (auto& e : row) e = Mat::Zero(4, 4);
    W[0][0] = I4;
    W[5][5] = I4;
    W[5][0] = U * (Nu * Nd) - mu * (Nu + Nd);   // on-site U n_up n_dn - mu n  (mu=U/2 pins half filling)
    W[5][1] = -t * (Cupd * F); W[1][0] = Cup;   // c^dag_{i up} c_{i+1 up}
    W[5][2] = -t * (Cdnd * F); W[2][0] = Cdn;   // c^dag_{i dn} c_{i+1 dn}
    W[5][3] = -t * (F * Cup ); W[3][0] = Cupd;  // h.c. (up): c^dag_{i+1 up} c_{i up}
    W[5][4] = -t * (F * Cdn ); W[4][0] = Cdnd;  // h.c. (dn)

    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        int cl = (i == 0) ? 1 : chi, cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        // Single U(1): particle number N. Packed two-U(1): (N, 2Sz) -- injective again
        // (|up>/|dn> split), several times finer blocks. Same H either way.
        site.d = d;
        site.phys_charge = two_u1
            ? std::vector<int>{pack_q2(0,0), pack_q2(1,+1), pack_q2(1,-1), pack_q2(2,0)}
            : std::vector<int>{0, 1, 1, 2};
        site.ops.assign(cl * cr, Mat::Zero(d, d));
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 5 : a, br = (i == n - 1) ? 0 : b;
                site.W(a, b) = W[ar][br];
            }
    }
    return mpo;
}

// Fermionic Hubbard on the interleaved PURIFIED chain (physical even, ancilla odd) --
// the d>2 analogue of build_spin1_purified_mpo, for finite temperature. Convention:
// the Jordan-Wigner string threads the PHYSICAL modes only; the ancillas are bosonic
// spectators outside the string. The physical hopping's F parity factors therefore
// live entirely in the physical emit/complete operators (exactly as in
// build_hubbard_mpo), and the ancilla sites pass every MPO bond channel through with
// the PLAIN identity (not F). This is consistent with the swap-and-carry cooling,
// which only ever exchanges a physical mode with an ancilla (never two physical
// modes), so the plain SWAP permutation is exact. Always the packed two-U(1) charge.
std::vector<MPOSite> build_hubbard_purified_mpo(int n_phys, double t, double U, double mu) {
    const int chi = 6, d = 4, n = 2 * n_phys;
    Mat I4 = Mat::Identity(4, 4);
    Mat F  = Mat::Zero(4, 4); F(0,0)=1; F(1,1)=-1; F(2,2)=-1; F(3,3)=1;
    Mat Nu = Mat::Zero(4, 4); Nu(1,1)=1; Nu(3,3)=1;
    Mat Nd = Mat::Zero(4, 4); Nd(2,2)=1; Nd(3,3)=1;
    Mat Cup = Mat::Zero(4, 4); Cup(0,1)=1; Cup(2,3)=1;
    Mat Cdn = Mat::Zero(4, 4); Cdn(0,2)=1; Cdn(1,3)=-1;
    Mat Cupd = Cup.adjoint(), Cdnd = Cdn.adjoint();

    std::array<std::array<Mat, 6>, 6> Wp;   // physical site (same bulk as build_hubbard_mpo)
    for (auto& row : Wp) for (auto& e : row) e = Mat::Zero(4, 4);
    Wp[0][0] = I4; Wp[5][5] = I4;
    Wp[5][0] = U * (Nu * Nd) - mu * (Nu + Nd);
    Wp[5][1] = -t * (Cupd * F); Wp[1][0] = Cup;
    Wp[5][2] = -t * (Cdnd * F); Wp[2][0] = Cdn;
    Wp[5][3] = -t * (F * Cup ); Wp[3][0] = Cupd;
    Wp[5][4] = -t * (F * Cdn ); Wp[4][0] = Cdnd;
    std::array<std::array<Mat, 6>, 6> Wa;   // ancilla: identity pass-through, all channels
    for (auto& row : Wa) for (auto& e : row) e = Mat::Zero(4, 4);
    for (int b = 0; b < chi; ++b) Wa[b][b] = I4;

    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        const auto& W6 = (i % 2 == 0) ? Wp : Wa;
        int cl = (i == 0) ? 1 : chi, cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        site.d = d;
        site.phys_charge = {pack_q2(0,0), pack_q2(1,+1), pack_q2(1,-1), pack_q2(2,0)};
        site.ops.assign(cl * cr, Mat::Zero(d, d));
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 5 : a, br = (i == n - 1) ? 0 : b;
                site.W(a, b) = W6[ar][br];
            }
    }
    return mpo;
}

// Total CHARGE-current operator of the Hubbard chain on the interleaved purified
// chain -- the d=4 analogue of build_current_purified_mpo, for finite-T transport.
// From the continuity equation for H = -t sum(c^dag c + h.c.):
//   J = i t sum_{i,sigma} (c^dag_{i sigma} c_{i+1 sigma} - c^dag_{i+1 sigma} c_{i sigma})
// which in the Jordan-Wigner matrix-tensor form (F parity in the emit operators,
// exactly as in build_hubbard_mpo) is
//   J = i t sum_sigma [ (C^dag_sigma F)_i (x) (C_sigma)_{i+1} - (F C_sigma)_i (x) (C^dag_sigma)_{i+1} ].
// Hermitian; no on-site term (W(start,done)=0). Ancillas are identity pass-throughs.
std::vector<MPOSite> build_hubbard_current_purified_mpo(int n_phys, double t) {
    const int chi = 6, d = 4, n = 2 * n_phys;
    Mat I4 = Mat::Identity(4, 4);
    Mat F  = Mat::Zero(4, 4); F(0,0)=1; F(1,1)=-1; F(2,2)=-1; F(3,3)=1;
    Mat Cup = Mat::Zero(4, 4); Cup(0,1)=1; Cup(2,3)=1;
    Mat Cdn = Mat::Zero(4, 4); Cdn(0,2)=1; Cdn(1,3)=-1;
    Mat Cupd = Cup.adjoint(), Cdnd = Cdn.adjoint();
    const Cx it(0.0, t);

    std::array<std::array<Mat, 6>, 6> Wp;
    for (auto& row : Wp) for (auto& e : row) e = Mat::Zero(4, 4);
    Wp[0][0] = I4; Wp[5][5] = I4;
    Wp[5][1] = it * (Cupd * F);    Wp[1][0] = Cup;    // +it c^dag_up c_up (forward)
    Wp[5][2] = it * (Cdnd * F);    Wp[2][0] = Cdn;    // +it c^dag_dn c_dn
    Wp[5][3] = (-it) * (F * Cup);  Wp[3][0] = Cupd;   // -it c^dag_{i+1 up} c_{i up}
    Wp[5][4] = (-it) * (F * Cdn);  Wp[4][0] = Cdnd;
    std::array<std::array<Mat, 6>, 6> Wa;
    for (auto& row : Wa) for (auto& e : row) e = Mat::Zero(4, 4);
    for (int b = 0; b < chi; ++b) Wa[b][b] = I4;

    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        const auto& W6 = (i % 2 == 0) ? Wp : Wa;
        int cl = (i == 0) ? 1 : chi, cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        site.d = d;
        site.phys_charge = {pack_q2(0,0), pack_q2(1,+1), pack_q2(1,-1), pack_q2(2,0)};
        site.ops.assign(cl * cr, Mat::Zero(d, d));
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 5 : a, br = (i == n - 1) ? 0 : b;
                site.W(a, b) = W6[ar][br];
            }
    }
    return mpo;
}

// Two-leg fermionic Hubbard LADDER via RUNG FUSION -- the first quasi-2D model.
//   H = -t sum_{x,leg,s} (c^dag c + h.c.)_leg  - t_perp sum_{x,s} (c^dag_a c_b + h.c.)_rung
//       + U sum n_up n_dn - mu sum n
// Each rung (two Electron sites, legs a/b) is fused into ONE d=16 MPS site, so the
// ladder becomes a UNIFORM NEAREST-NEIGHBOUR chain of rungs: derive_mpo_sym, the block
// DMRG, the packed two-U(1) charges and the TEBD/TDVP2 gates all apply with no engine
// changes (the alternative -- a snake ordering with range-2 couplings -- would need
// period-2 non-uniform MPO support).
//
// Jordan-Wigner mode order: (x,a),(x,b),(x+1,a),(x+1,b),...  Rung basis |s_a, s_b>
// with index s = 4*s_a + s_b (a = most significant, matching the ED kron convention).
// Rung-local operators (strings within the rung):
//   c_{a s} = C_s (x) I,   c_{b s} = F (x) C_s          (string passes mode a)
// Inter-rung hopping (string covers the modes between the two legs' positions):
//   c^dag_{x,a s} c_{x+1,a s} = [c^dag_{a s} F2]_x (x) [c_{a s}]_{x+1},  F2 = F(x)F
//   c^dag_{x,b s} c_{x+1,b s} = [c^dag_{b s} (I(x)F)]_x (x) [c_{b s}]_{x+1}
// (the emit operator carries the parity of every mode from its own position to the
// rung end; verified against the explicit-JW ED to machine zero in qx_hubbard_test).
// 10-wide MPO: 0=done, 1..8 = the eight travelling leg-hop channels, 9=start.
// Charges: packed two-U(1) per rung, qphys[s] = pack(N_a+N_b, 2Sz_a+2Sz_b).
std::vector<MPOSite> build_hubbard_ladder_mpo(int Lx, double t, double tperp,
                                              double U, double mu) {
    const int chi = 10, d = 16;
    Mat I4 = Mat::Identity(4, 4);
    Mat F  = Mat::Zero(4, 4); F(0,0)=1; F(1,1)=-1; F(2,2)=-1; F(3,3)=1;
    Mat Nu = Mat::Zero(4, 4); Nu(1,1)=1; Nu(3,3)=1;
    Mat Nd = Mat::Zero(4, 4); Nd(2,2)=1; Nd(3,3)=1;
    Mat Cup = Mat::Zero(4, 4); Cup(0,1)=1; Cup(2,3)=1;
    Mat Cdn = Mat::Zero(4, 4); Cdn(0,2)=1; Cdn(1,3)=-1;
    auto kron16 = [](const Mat& A, const Mat& B) {          // A on leg a (major), B on b
        Mat K(16, 16);
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
            K.block(4*i, 4*j, 4, 4) = A(i, j) * B;
        return K;
    };
    const Mat I16 = Mat::Identity(16, 16);
    const Mat F2  = kron16(F, F);                            // rung parity
    const Mat IF  = kron16(I4, F);                           // parity of mode b only
    // rung-local fermion operators (JW within the rung)
    auto ca = [&](const Mat& C){ return kron16(C, I4); };    // leg a
    auto cb = [&](const Mat& C){ return kron16(F, C); };     // leg b (string over a)
    const Mat CaU = ca(Cup), CaD = ca(Cdn), CbU = cb(Cup), CbD = cb(Cdn);
    const Mat Na  = kron16(Nu + Nd, I4), Nb = kron16(I4, Nu + Nd);
    // rung-local Hamiltonian: rung hopping + U + mu
    Mat Hr = Mat::Zero(16, 16);
    Mat rung_hop = CaU.adjoint() * CbU + CaD.adjoint() * CbD;
    Hr += -tperp * (rung_hop + rung_hop.adjoint());
    Hr += U * (kron16(Nu * Nd, I4) + kron16(I4, Nu * Nd));
    Hr += -mu * (Na + Nb);

    // travelling channels: emit on rung x, complete on rung x+1
    struct Ch { Mat emit, comp; };
    std::vector<Ch> ch;
    for (const Mat* C : {&CaU, &CaD}) {          // leg a, both spins: fwd + bwd
        Mat cd = C->adjoint();
        ch.push_back({ -t * (cd * F2), *C });                    // c^dag_x c_{x+1}
        ch.push_back({ -t * (F2 * (*C)), cd });                  // h.c.
    }
    for (const Mat* C : {&Cup, &Cdn}) {          // leg b
        // Mode-level derivation: c^dag_{x,b s} c_{x+1,b s} = [I (x) (C^dag_s F)]_x
        // (string = b's own remaining modes) (x) [F (x) C_s]_{x+1} (string over the
        // intervening a modes on the NEXT rung). The emit uses the BARE b operator
        // (no a-string -- that string belongs to the completing rung), the complete
        // uses the a-strung one. Getting this split wrong is a 2t-sized error the
        // explicit-JW ED check catches immediately.
        Mat bare = kron16(I4, *C);               // b annihilator WITHOUT the a-string
        Mat strung = cb(*C);                     // b annihilator WITH the a-string
        ch.push_back({ -t * (bare.adjoint() * IF), strung });            // fwd
        ch.push_back({ -t * (IF * bare), Mat(strung.adjoint()) });       // bwd (h.c.)
    }

    std::array<std::array<Mat, 10>, 10> W;
    for (auto& row : W) for (auto& e : row) e = Mat::Zero(16, 16);
    W[0][0] = I16; W[9][9] = I16;
    W[9][0] = Hr;
    for (int k = 0; k < 8; ++k) { W[9][1 + k] = ch[k].emit; W[1 + k][0] = ch[k].comp; }

    // packed (N, 2Sz) rung charges: per-4-state N = {0,1,1,2}, 2Sz = {0,+1,-1,0}
    const int n4[4] = {0, 1, 1, 2}, z4[4] = {0, +1, -1, 0};
    std::vector<int> qphys(16);
    for (int sa = 0; sa < 4; ++sa)
        for (int sb = 0; sb < 4; ++sb)
            qphys[4*sa + sb] = pack_q2(n4[sa] + n4[sb], z4[sa] + z4[sb]);

    std::vector<MPOSite> mpo(Lx);
    for (int i = 0; i < Lx; ++i) {
        int cl = (i == 0) ? 1 : chi, cr = (i == Lx - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        site.d = d; site.phys_charge = qphys;
        site.ops.assign(cl * cr, Mat::Zero(d, d));
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 9 : a, br = (i == Lx - 1) ? 0 : b;
                site.W(a, b) = W[ar][br];
            }
    }
    return mpo;
}

// Two-leg Hubbard ladder, SITE-SPLIT formulation: the same Hamiltonian as
// build_hubbard_ladder_mpo, but on 2*Lx Electron sites (d=4) in the JW order
// A0,B0,A1,B1,... -- a PERIOD-2 uniform-chi MPO (the A sublattice emits the rung and
// leg-a channels, the B sublattice the leg-b ones; derive_mpo_sym reads the union of
// both sublattices' constraints). Rung hops are nearest-neighbour; leg hops are
// range-2 with an F (parity) carry on the intervening site:
//   c^dag_{A_x s} c_{A_{x+1} s} = (C^dag_s F)_{A_x} (x) F_{B_x} (x) (C_s)_{A_{x+1}}.
// This formulation does LESS local work per sweep than the rung-fused d=16 chain
// (d=4, twice the bonds) -- the performance mapping; both are the same physics.
// 14-wide: 0=done, 1..4 rung (A->B), 5..8 leg-a (A..A, diagonal F carry at B),
// 9..12 leg-b (B..B, diagonal F carry at A), 13=start.
std::vector<MPOSite> build_hubbard_ladder_split_mpo(int Lx, double t, double tperp,
                                                    double U, double mu) {
    const int chi = 14, d = 4, n = 2 * Lx;
    Mat I4 = Mat::Identity(4, 4);
    Mat F  = Mat::Zero(4, 4); F(0,0)=1; F(1,1)=-1; F(2,2)=-1; F(3,3)=1;
    Mat Nu = Mat::Zero(4, 4); Nu(1,1)=1; Nu(3,3)=1;
    Mat Nd = Mat::Zero(4, 4); Nd(2,2)=1; Nd(3,3)=1;
    Mat Cup = Mat::Zero(4, 4); Cup(0,1)=1; Cup(2,3)=1;
    Mat Cdn = Mat::Zero(4, 4); Cdn(0,2)=1; Cdn(1,3)=-1;
    Mat Cupd = Cup.adjoint(), Cdnd = Cdn.adjoint();
    Mat hs = U * (Nu * Nd) - mu * (Nu + Nd);     // on-site, both sublattices
    // the four hop "half"-operators, one per channel slot within a group:
    // emit {C^dag_up F, C^dag_dn F, F C_up, F C_dn}, complete {C_up, C_dn, C^dag_up, C^dag_dn}
    const Mat em[4] = { Mat(Cupd * F), Mat(Cdnd * F), Mat(F * Cup), Mat(F * Cdn) };
    const Mat co[4] = { Cup, Cdn, Cupd, Cdnd };

    std::array<std::array<Mat, 14>, 14> WA, WB;
    for (auto& row : WA) for (auto& e : row) e = Mat::Zero(4, 4);
    for (auto& row : WB) for (auto& e : row) e = Mat::Zero(4, 4);
    WA[0][0] = WA[13][13] = I4;  WB[0][0] = WB[13][13] = I4;
    WA[13][0] = hs;              WB[13][0] = hs;
    for (int k = 0; k < 4; ++k) {
        WA[13][1 + k] = -tperp * em[k];  WB[1 + k][0] = co[k];       // rung: A emits, B completes
        WA[13][5 + k] = -t * em[k];      WB[5 + k][5 + k] = F;       // leg a: A emits, B carries F,
        WA[5 + k][0]  = co[k];                                       //        next A completes
        WB[13][9 + k] = -t * em[k];      WA[9 + k][9 + k] = F;       // leg b: B emits, A carries F,
        WB[9 + k][0]  = co[k];                                       //        next B completes
    }

    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        const auto& W14 = (i % 2 == 0) ? WA : WB;
        int cl = (i == 0) ? 1 : chi, cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        site.d = d;
        site.phys_charge = {pack_q2(0,0), pack_q2(1,+1), pack_q2(1,-1), pack_q2(2,0)};
        site.ops.assign(cl * cr, Mat::Zero(d, d));
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 13 : a, br = (i == n - 1) ? 0 : b;
                site.W(a, b) = W14[ar][br];
            }
    }
    return mpo;
}

// Width-W Hubbard strip/CYLINDER, site-split (d=4): the general quasi-2D builder.
// Column-major order site i = W*x + y (x = 0..Lx-1 rungs/columns, y = 0..W-1 legs).
//   y-hops: (x,y)-(x,y+1) = chain-NN (emitted only from y < W-1);
//   y-wrap (y_periodic): (x,0)-(x,W-1) = chain range-(W-1), emitted only from y=0;
//   x-hops: (x,y)-(x+1,y) = chain range-W, emitted from EVERY site (uniform).
// Every intervening site of a range-r hop carries the parity F (Jordan-Wigner),
// exactly the (ED-validated) ladder-split pattern generalized: emit = C^dag F / F C
// forms, diagonal F-carry chains (o, rem) -> (o, rem-1), complete = C / C^dag.
// PERIOD-W MPO (only the start-row emits are y-dependent); derive_mpo_sym merges the
// constraints of up to 8 consecutive bulk sites, so W <= 8 is supported.
// chi = 2 + 4W (x-chains) + 4 (y-NN) + [y_periodic: 4(W-1) wrap chains].
// W=2 open reproduces build_hubbard_ladder_split_mpo's Hamiltonian (self-check).
std::vector<MPOSite> build_hubbard_lattice_split_mpo(int Lx, int W, double t,
                                                     double U, double mu,
                                                     bool y_periodic, double hpin) {
    const int d = 4, n = W * Lx;
    const int nx = 4 * W;                        // x-hop carry slots
    const int nw = y_periodic ? 4 * (W - 1) : 0; // y-wrap carry slots
    const int chi = 2 + nx + 4 + nw;
    Mat I4 = Mat::Identity(4, 4);
    Mat F  = Mat::Zero(4, 4); F(0,0)=1; F(1,1)=-1; F(2,2)=-1; F(3,3)=1;
    Mat Nu = Mat::Zero(4, 4); Nu(1,1)=1; Nu(3,3)=1;
    Mat Nd = Mat::Zero(4, 4); Nd(2,2)=1; Nd(3,3)=1;
    Mat Cup = Mat::Zero(4, 4); Cup(0,1)=1; Cup(2,3)=1;
    Mat Cdn = Mat::Zero(4, 4); Cdn(0,2)=1; Cdn(1,3)=-1;
    Mat hs = U * (Nu * Nd) - mu * (Nu + Nd);
    const Mat em[4] = { Mat(Cup.adjoint() * F), Mat(Cdn.adjoint() * F),
                        Mat(F * Cup), Mat(F * Cdn) };
    const Mat co[4] = { Cup, Cdn, Mat(Cup.adjoint()), Mat(Cdn.adjoint()) };
    // slot layout: 0 = done; x-chains 1 .. 4W (op o slot base 1+o*W, rem = 0..W-1 at
    // base+rem); y-NN 1+4W .. 4+4W; wrap chains next (op o base, rem = 0..W-2);
    // start = chi-1.
    const int xbase = 1, ybase = 1 + nx, wbase = ybase + 4, start = chi - 1, done = 0;

    // one W-matrix per leg index y (period W); only the start-row differs across y.
    std::vector<std::array<std::array<Mat, 64>, 64>> Wy;   // chi <= 2+32+4+28 = 66? guard below
    if (chi > 64) throw std::runtime_error("build_hubbard_lattice_split_mpo: W too large (chi > 64)");
    Wy.resize(W);
    for (int y = 0; y < W; ++y) {
        auto& Wm = Wy[y];
        for (int a = 0; a < chi; ++a) for (int b = 0; b < chi; ++b) Wm[a][b] = Mat::Zero(4, 4);
        Wm[done][done] = I4; Wm[start][start] = I4;
        Wm[start][done] = hs;
        for (int o = 0; o < 4; ++o) {
            // x-hops: uniform emit into rem=W-1, F-carry chain down, complete at rem=0
            Wm[start][xbase + o*W + (W - 1)] = -t * em[o];
            for (int rem = 1; rem < W; ++rem)
                Wm[xbase + o*W + rem][xbase + o*W + rem - 1] = F;
            Wm[xbase + o*W + 0][done] = co[o];
            // y-NN: emit only below the top leg; complete uniformly
            if (y < W - 1) Wm[start][ybase + o] = -t * em[o];
            Wm[ybase + o][done] = co[o];
            // y-wrap (cylinder): emit only from y=0, range W-1 (W-2 F-carries)
            if (y_periodic && W > 2) {
                if (y == 0) Wm[start][wbase + o*(W-1) + (W - 2)] = -t * em[o];
                for (int rem = 1; rem < W - 1; ++rem)
                    Wm[wbase + o*(W-1) + rem][wbase + o*(W-1) + rem - 1] = F;
                Wm[wbase + o*(W-1) + 0][done] = co[o];
            } else if (y_periodic && W == 2) {
                // W=2 wrap coincides with the y-NN bond: double the y coupling? No --
                // a 2-leg "cylinder" would double-count the rung; reject loudly.
                throw std::runtime_error("build_hubbard_lattice_split_mpo: y_periodic "
                                         "needs W >= 3 (W=2 wrap duplicates the rung)");
            }
        }
    }

    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        const auto& Wm = Wy[i % W];
        int cl = (i == 0) ? 1 : chi, cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        site.d = d;
        site.phys_charge = {pack_q2(0,0), pack_q2(1,+1), pack_q2(1,-1), pack_q2(2,0)};
        site.ops.assign(cl * cr, Mat::Zero(d, d));
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? start : a, br = (i == n - 1) ? done : b;
                site.W(a, b) = Wm[ar][br];
            }
    }
    // Edge pinning: H -= hpin (-1)^y Sz_{0,y} on the x=0 column only (sites 0..W-1).
    // Sz = diag(0, +1/2, -1/2, 0) is diagonal in both charges, so the bond charge
    // structure (and derive_mpo_sym) is untouched; only the on-site start->done
    // block of the first W site matrices changes.
    if (hpin != 0.0) {
        Mat Sz = Mat::Zero(4, 4); Sz(1,1) = 0.5; Sz(2,2) = -0.5;
        for (int y = 0; y < W && y < n; ++y) {
            double h = -hpin * ((y % 2 == 0) ? 1.0 : -1.0);
            mpo[y].W((y == 0) ? 0 : start, 0) += h * Sz;   // site 0: cl=1, row 0 IS start
        }
    }
    return mpo;
}

// Singlet pair-field correlator MPO on the width-W lattice (column-major sites):
//   O = Delta^dag_{b1} Delta_{b2},   Delta^dag_b = (1/sqrt2)(c^dag_{i up} c^dag_{i' dn}
//                                                          - c^dag_{i dn} c^dag_{i' up})
// for bonds b = (x, y, orient): orient 'y' -> i' = i+1 (rung bond, needs y < W-1),
// orient 'x' -> i' = i+W (leg bond). The SC order-parameter correlator of the doped
// Hubbard lattice; measured between y/x oriented bonds it exposes the d-wave sign
// structure (Phi_xy < 0). Chain-ordered JW form of each pair (p1 < p2):
//   c^#_{p1 s} c^#_{p2 s'} = (a^#_s F)_{p1} (x) F^(x)(p2-p1-1) (x) (a^#_s')_{p2}
// -- the ED-validated hop pattern (emit carries F, intervening F, bare complete),
// valid for any dagger combination. Each completed pair is parity-EVEN, so the
// stretch between b1 and b2 carries plain identity (no string).
// chi=7 slots (one charge per slot, derive_mpo_sym-compatible):
//   0=done(0); 1,2 = creation channels c^dag_up/c^dag_dn emitted; 3 = start carry(0);
//   4 = between-pairs carry (charge pack(2,0)); 5,6 = annihilation channels.
// Requires b1 strictly left of b2 (last site of b1 < first site of b2).
std::vector<MPOSite> build_pair_corr_mpo(int n, int W, int x1, int y1, char o1,
                                         int x2, int y2, char o2) {
    const int d = 4, chi = 7, start = 3, done = 0, mid = 4;
    auto bond_sites = [&](int x, int y, char o, int& a, int& b) {
        if (o != 'x' && o != 'y')
            throw std::runtime_error("build_pair_corr_mpo: orient must be 'x' or 'y'");
        if (y < 0 || y >= W || x < 0)
            throw std::runtime_error("build_pair_corr_mpo: bond (x,y) outside lattice");
        a = W * x + y;
        b = (o == 'y') ? a + 1 : a + W;
        if (o == 'y' && y >= W - 1)
            throw std::runtime_error("build_pair_corr_mpo: y-bond needs y < W-1");
        if (b >= n) throw std::runtime_error("build_pair_corr_mpo: bond leaves lattice");
    };
    int i1, i1e, i2, i2e;
    bond_sites(x1, y1, o1, i1, i1e);
    bond_sites(x2, y2, o2, i2, i2e);
    if (i1e >= i2)
        throw std::runtime_error("build_pair_corr_mpo: bonds must be disjoint, b1 < b2");

    Mat I4 = Mat::Identity(4, 4);
    Mat F  = Mat::Zero(4, 4); F(0,0)=1; F(1,1)=-1; F(2,2)=-1; F(3,3)=1;
    Mat Cup = Mat::Zero(4, 4); Cup(0,1)=1; Cup(2,3)=1;
    Mat Cdn = Mat::Zero(4, 4); Cdn(0,2)=1; Cdn(1,3)=-1;
    const double s2 = 1.0 / std::sqrt(2.0);

    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        std::array<std::array<Mat, 7>, 7> Wm;
        for (auto& row : Wm) for (auto& e : row) e = Mat::Zero(4, 4);
        Wm[done][done] = I4;
        if (i < i1) Wm[start][start] = I4;
        if (i == i1) {                                    // emit creation pair
            Wm[start][1] = s2 * Mat(Cup.adjoint() * F);
            Wm[start][2] = s2 * Mat(Cdn.adjoint() * F);
        }
        if (i > i1 && i < i1e) { Wm[1][1] = F; Wm[2][2] = F; }   // x-bond JW carry
        if (i == i1e) {                                   // complete creation pair
            Wm[1][mid] = Mat(Cdn.adjoint());
            Wm[2][mid] = -Mat(Cup.adjoint());
        }
        if (i > i1e && i < i2) Wm[mid][mid] = I4;         // even stretch: no string
        if (i == i2) {                                    // emit annihilation pair:
            // Delta_{b2} = (1/sqrt2)(-c_{j up}c_{j' dn} + c_{j dn}c_{j' up});
            // c_{p1}c_{p2} chain form = (a F)_{p1} (x) F... (x) (a)_{p2}, so the
            // emit is C*F (NOT the hop convention's F*C = -C*F, which encodes the
            // reordering sign of c^dag_{p2} c_{p1} -- absent here).
            Wm[mid][5] = -s2 * Mat(Cup * F);
            Wm[mid][6] =  s2 * Mat(Cdn * F);
        }
        if (i > i2 && i < i2e) { Wm[5][5] = F; Wm[6][6] = F; }
        if (i == i2e) {                                   // complete annihilation pair
            Wm[5][done] = Cdn;
            Wm[6][done] = Cup;
        }
        int cl = (i == 0) ? 1 : chi, cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        site.d = d;
        site.phys_charge = {pack_q2(0,0), pack_q2(1,+1), pack_q2(1,-1), pack_q2(2,0)};
        site.ops.assign(cl * cr, Mat::Zero(d, d));
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? start : a, br = (i == n - 1) ? done : b;
                site.W(a, b) = Wm[ar][br];
            }
    }
    return mpo;
}

// J1-J2 (next-nearest-neighbour) XXZ chain MPO: H = sum_i [J1 h_{i,i+1} + J2
// h_{i,i+2}], h = XX+YY+delta ZZ = 2(sigma+ sigma- + sigma- sigma+) + delta ZZ.
// 8-wide: each emitted operator completes at range 1, or passes one identity and
// completes at range 2. States: 0 done, {1,2}=sigma+ {range1,range2}, {3,4}=sigma-,
// {5,6}=Z, 7 start. Same lower-triangular convention as build_xxz_mpo (left
// boundary = start row, right boundary = done col 0), so derive_mpo_sym yields
// charges {0,+2,+2,-2,-2,0,0,0} and the block path can consume it.
std::vector<MPOSite> build_j1j2_mpo(int n, double J1, double J2, double delta) {
    const int chi = 8;
    Mat2 sp; sp << 0, 1, 0, 0;   // sigma+ = |up><dn|
    Mat2 sm; sm << 0, 0, 1, 0;   // sigma-
    Mat2 Z = pauli('Z'), I = pauli('I');
    std::array<std::array<Mat2, 8>, 8> W{};
    for (auto& row : W) for (auto& e : row) e.setZero();
    W[0][0] = I;                       // done -> done
    W[7][7] = I;                       // start -> start
    W[7][1] = sp;  W[7][3] = sm;  W[7][5] = Z;          // emit (bare) from start
    W[1][0] = (2.0 * J1) * sm;  W[1][2] = I;  W[2][0] = (2.0 * J2) * sm;   // sigma+ chain
    W[3][0] = (2.0 * J1) * sp;  W[3][4] = I;  W[4][0] = (2.0 * J2) * sp;   // sigma- chain
    W[5][0] = (J1 * delta) * Z; W[5][6] = I;  W[6][0] = (J2 * delta) * Z;  // Z chain

    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        int cl = (i == 0) ? 1 : chi, cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        site.ops.assign(cl * cr, Mat2::Zero());
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 7 : a, br = (i == n - 1) ? 0 : b;
                site.W(a, b) = W[ar][br];
            }
    }
    return mpo;
}

// Spin-current MPO for the XXZ chain: J_tot = sum_l i*J (sigma+_l sigma-_{l+1} -
// sigma-_l sigma+_{l+1}) (the Z term carries no spin current). Hermitian; same Sz
// bond structure as the Heisenberg MPO, so the block engine applies it unchanged.
std::vector<MPOSite> build_current_mpo(int n, double J) {
    const int chi = 5;
    Mat2 sp; sp << 0, 1, 0, 0;
    Mat2 sm; sm << 0, 0, 1, 0;
    Cx iJ(0.0, J);
    std::array<std::array<Mat2, 5>, 5> W5{};
    for (auto& row : W5) for (auto& e : row) e.setZero();
    W5[0][0] = pauli('I');
    W5[1][0] = sm;                 // complete  i*J sigma+_l sigma-_{l+1}
    W5[2][0] = sp;                 // complete -i*J sigma-_l sigma+_{l+1}
    W5[4][1] = iJ * sp;            // emit sigma+ (bond 1, DeltaSz +2)
    W5[4][2] = (-iJ) * sm;         // emit sigma- (bond 2, DeltaSz -2)
    W5[4][4] = pauli('I');
    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        int cl = (i == 0) ? 1 : chi, cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        site.ops.assign(cl * cr, Mat2::Zero());
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 4 : a, br = (i == n - 1) ? 0 : b;
                site.W(a, b) = W5[ar][br];
            }
    }
    return mpo;
}

// XXZ MPO on the interleaved purified chain (2*n_phys sites): even sites carry the
// physical XXZ couplings, odd (ancilla) sites are identity pass-throughs on every
// MPO bond state, so an "in-progress" term emitted at physical site 2i hops over
// ancilla 2i+1 and completes at physical site 2i+2. The identity is Sz-diagonal
// and keeps each bond state's HB_DELTA, so the block-sparse engine (environments,
// H_eff, energies) runs on the purified chain completely unchanged.
//
// Note the last *physical* site (2*n_phys-2) still uses the full bulk W (it can
// "emit" σ±/Z to its right), but those in-progress states die at the right
// boundary (col 0 of the final ancilla's identity), so H is exactly
//   J sum_{i<n_phys-1} [ 2(σ+_{2i}σ-_{2i+2} + h.c.) + delta Z_{2i} Z_{2i+2} ].
std::vector<MPOSite> build_xxz_purified_mpo(int n_phys, double J, double delta) {
    const int chi = 5, n = 2 * n_phys;
    Mat2 sp; sp << 0, 1, 0, 0;   // σ+ = |↑⟩⟨↓|  (ΔSz = +2)
    Mat2 sm; sm << 0, 0, 1, 0;   // σ- = |↓⟩⟨↑|  (ΔSz = -2)

    // Physical-site W (same as build_xxz_mpo bulk).
    std::array<std::array<Mat2, 5>, 5> Wp{};
    for (auto& row : Wp) for (auto& e : row) e.setZero();
    Wp[0][0] = pauli('I');
    Wp[1][0] = sm;
    Wp[2][0] = sp;
    Wp[3][0] = pauli('Z');
    Wp[4][1] = (2.0 * J) * sp;
    Wp[4][2] = (2.0 * J) * sm;
    Wp[4][3] = (J * delta) * pauli('Z');
    Wp[4][4] = pauli('I');
    // Ancilla-site W: diagonal identity (pass every bond state through).
    std::array<std::array<Mat2, 5>, 5> Wa{};
    for (auto& row : Wa) for (auto& e : row) e.setZero();
    for (int b = 0; b < chi; ++b) Wa[b][b] = pauli('I');

    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        const auto& W5 = (i % 2 == 0) ? Wp : Wa;
        int cl = (i == 0) ? 1 : chi, cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        site.ops.assign(cl * cr, Mat2::Zero());
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 4 : a, br = (i == n - 1) ? 0 : b;
                site.W(a, b) = W5[ar][br];
            }
    }
    return mpo;
}

// Spin-1 (d=3) Heisenberg MPO on the interleaved purified chain (2*n_phys sites):
// even sites carry the physical spin-1 couplings, odd (ancilla) sites are identity
// (I3) pass-throughs on every bond state -- same structure as build_xxz_purified_mpo
// but with d=3 / physical spin-1 operators, so the block engine runs the spin-1
// purified chain unchanged. H = J sum_{i<n_phys-1} S_{2i}.S_{2i+2}.
std::vector<MPOSite> build_spin1_purified_mpo(int n_phys, double J) {
    const int chi = 5, d = 3, n = 2 * n_phys;
    const double r2 = std::sqrt(2.0);
    Mat Sp = Mat::Zero(3, 3); Sp(0, 1) = r2; Sp(1, 2) = r2;
    Mat Sm = Mat::Zero(3, 3); Sm(1, 0) = r2; Sm(2, 1) = r2;
    Mat Sz = Mat::Zero(3, 3); Sz(0, 0) = 1.0; Sz(2, 2) = -1.0;
    Mat I3 = Mat::Identity(3, 3);
    // Physical-site W (same bulk as build_spin1_heisenberg_mpo).
    std::array<std::array<Mat, 5>, 5> Wp;
    for (auto& row : Wp) for (auto& e : row) e = Mat::Zero(3, 3);
    Wp[0][0] = I3; Wp[1][0] = Sm; Wp[2][0] = Sp; Wp[3][0] = Sz;
    Wp[4][1] = (0.5 * J) * Sp; Wp[4][2] = (0.5 * J) * Sm; Wp[4][3] = J * Sz; Wp[4][4] = I3;
    // Ancilla-site W: diagonal identity (pass every bond state through).
    std::array<std::array<Mat, 5>, 5> Wa;
    for (auto& row : Wa) for (auto& e : row) e = Mat::Zero(3, 3);
    for (int b = 0; b < chi; ++b) Wa[b][b] = I3;

    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        const auto& W5 = (i % 2 == 0) ? Wp : Wa;
        int cl = (i == 0) ? 1 : chi, cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        site.d = d; site.phys_charge = {+2, 0, -2};
        site.ops.assign(cl * cr, Mat::Zero(d, d));
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 4 : a, br = (i == n - 1) ? 0 : b;
                site.W(a, b) = W5[ar][br];
            }
    }
    return mpo;
}

// Total spin-current operator on the interleaved purified chain: even sites carry
// the current W (sigma+/sigma- emission, no Z term -- the Z bond carries no
// current), odd (ancilla) sites are identity pass-throughs. Same HB_DELTA bond
// structure as build_xxz_purified_mpo, so apply_mpo_to_bmps applies it unchanged.
std::vector<MPOSite> build_current_purified_mpo(int n_phys, double J) {
    const int chi = 5, n = 2 * n_phys;
    Mat2 sp; sp << 0, 1, 0, 0;
    Mat2 sm; sm << 0, 0, 1, 0;
    Cx iJ(0.0, J);
    std::array<std::array<Mat2, 5>, 5> Wp{};
    for (auto& row : Wp) for (auto& e : row) e.setZero();
    Wp[0][0] = pauli('I');
    Wp[1][0] = sm;                 // complete  i*J sigma+_l sigma-_{l+1}
    Wp[2][0] = sp;                 // complete -i*J sigma-_l sigma+_{l+1}
    Wp[4][1] = iJ * sp;            // emit sigma+ (bond 1, DeltaSz +2)
    Wp[4][2] = (-iJ) * sm;         // emit sigma- (bond 2, DeltaSz -2)
    Wp[4][4] = pauli('I');
    std::array<std::array<Mat2, 5>, 5> Wa{};
    for (auto& row : Wa) for (auto& e : row) e.setZero();
    for (int b = 0; b < chi; ++b) Wa[b][b] = pauli('I');

    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        const auto& W5 = (i % 2 == 0) ? Wp : Wa;
        int cl = (i == 0) ? 1 : chi, cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        site.ops.assign(cl * cr, Mat2::Zero());
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 4 : a, br = (i == n - 1) ? 0 : b;
                site.W(a, b) = W5[ar][br];
            }
    }
    return mpo;
}

// Spin-1 (d=3) total spin-current J = sum_l i J (S+_l S-_{l+1} - S-_l S+_{l+1}) on
// the interleaved purified chain -- the d>2 analogue of build_current_purified_mpo
// (same 5-wide structure, physical spin-1 S+- operators, ancilla I3 pass-through).
std::vector<MPOSite> build_spin1_current_purified_mpo(int n_phys, double J) {
    const int chi = 5, d = 3, n = 2 * n_phys;
    const double r2 = std::sqrt(2.0);
    Mat Sp = Mat::Zero(3, 3); Sp(0, 1) = r2; Sp(1, 2) = r2;
    Mat Sm = Mat::Zero(3, 3); Sm(1, 0) = r2; Sm(2, 1) = r2;
    Mat I3 = Mat::Identity(3, 3);
    Cx iJ(0.0, J);
    std::array<std::array<Mat, 5>, 5> Wp;
    for (auto& row : Wp) for (auto& e : row) e = Mat::Zero(3, 3);
    Wp[0][0] = I3;
    Wp[1][0] = Sm;                 // complete  i J S+_l S-_{l+1}
    Wp[2][0] = Sp;                 // complete -i J S-_l S+_{l+1}
    Wp[4][1] = iJ * Sp;            // emit S+ (bond 1, DeltaSz +2)
    Wp[4][2] = (-iJ) * Sm;         // emit S- (bond 2, DeltaSz -2)
    Wp[4][4] = I3;
    std::array<std::array<Mat, 5>, 5> Wa;
    for (auto& row : Wa) for (auto& e : row) e = Mat::Zero(3, 3);
    for (int b = 0; b < chi; ++b) Wa[b][b] = I3;

    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        const auto& W5 = (i % 2 == 0) ? Wp : Wa;
        int cl = (i == 0) ? 1 : chi, cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        site.d = d; site.phys_charge = {+2, 0, -2};
        site.ops.assign(cl * cr, Mat::Zero(d, d));
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 4 : a, br = (i == n - 1) ? 0 : b;
                site.W(a, b) = W5[ar][br];
            }
    }
    return mpo;
}

// -- Pauli label parser -------------------------------------------------------

// Convert indexed label ("X0X1") or plain label ("XXI") to per-site char array.
static std::vector<char> parse_pauli(const std::string& label, int n) {
    std::vector<char> ops(n, 'I');
    static const std::regex idx_re(R"(([XYZI])(\d+))");
    std::sregex_iterator it(label.begin(), label.end(), idx_re), end;
    if (it != end) {
        // Indexed format
        for (; it != end; ++it) {
            char op  = (*it)[1].str()[0];
            int  q   = std::stoi((*it)[2].str());
            if (q < n) ops[q] = op;
        }
    } else {
        // Plain format: pad/trim to length n
        for (int i = 0; i < n && i < (int)label.size(); ++i)
            ops[i] = std::toupper(label[i]);
    }
    return ops;
}

// Build MPO from a list of (coeff, pauli_label) terms.
// Naive construction: bond dim = n_terms + 2 (one state per term, plus boundaries).
// The MPO encodes H = sum_k c_k P^k_0 x P^k_1 x ... x P^k_{n-1}.
std::vector<MPOSite> build_pauli_sum_mpo(
        int n, const std::vector<std::pair<double, std::string>>& terms) {
    int nt = static_cast<int>(terms.size());
    // States: 0 = "left vac", 1..nt = "in progress for term k", nt+1 = "done"
    int full_chi = nt + 2;

    // Pre-parse all pauli strings
    std::vector<std::vector<char>> parsed(nt);
    std::vector<double> coeffs(nt);
    for (int k = 0; k < nt; ++k) {
        coeffs[k] = terms[k].first;
        parsed[k] = parse_pauli(terms[k].second, n);
    }

    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        int cl = (i == 0)     ? 1   : full_chi;
        int cr = (i == n - 1) ? 1   : full_chi;
        auto& site = mpo[i];
        site.chi_l = cl;
        site.chi_r = cr;
        site.ops.resize(cl * cr, Mat2::Zero());

        std::vector<std::vector<Mat2>> W(full_chi, std::vector<Mat2>(full_chi, Mat2::Zero()));
        W[0][0] = pauli('I');                   // left pass-through
        W[full_chi-1][full_chi-1] = pauli('I'); // right pass-through

        for (int k = 0; k < nt; ++k) {
            char p = parsed[k][i];
            if (i == 0) {
                W[0][k + 1] = coeffs[k] * pauli(p);
            } else if (i == n - 1) {
                W[k + 1][full_chi - 1] = pauli(p);
            } else {
                W[k + 1][k + 1] = pauli(p);
            }
            if (i == 0) {
                if (n == 1) W[0][full_chi - 1] = coeffs[k] * pauli(p);
            }
        }

        // Extract slice
        for (int a = 0; a < cl; ++a) {
            for (int b = 0; b < cr; ++b) {
                int ar2 = (i == 0) ? 0              : a;
                int br2 = (i == n-1) ? (full_chi-1) : b;
                site.W(a, b) = W[ar2][br2];
            }
        }
    }
    return mpo;
}

} // namespace qx_dmrg
