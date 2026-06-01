#pragma once
// SU(2)-symmetric DMRG core (stages 3-6): reduced MPS / environment tensors and
// the recoupling contractions, for the spin-1/2 Heisenberg chain.
//
// Design principle: every recoupling coefficient is computed by EXPLICIT
// Clebsch-Gordan summation (see recoup_*), not from a 6j/9j closed form. This
// removes all phase-convention risk -- the only SU(2) input is the validated CG
// of su2.hpp. (Closed-form 6j/9j can replace the CG sums later as an
// optimization.) Each contraction is validated by expanding the reduced result
// to full U(1) (Sz-resolved) tensors and comparing to a plain dense contraction.
//
// Conventions (doubled-integer two-j; physical spin-1/2 is tS=1, mu=+-1):
//   MPS site A: fuses left bond j_l (x) phys(1/2) -> right bond j_r.
//     full A^{mu}_{(jl ml al)(jr mr ar)} = <jl ml; 1/2 mu | jr mr> * Ared[jl,jr][al,ar]
//   Left environment L (legs: ket bond j_k, MPO bond w, bra bond j_b):
//     coupling  j_b (x) w -> j_k :
//     full L_{(jk mk ak)(w mw)(jb mb ab)} = <jb mb; w mw | jk mk> * Lred[jk,w,jb][ak,ab]
//   Right environment R (legs: ket bond j_k, MPO bond w, bra bond j_b):
//     coupling  j_k (x) w -> j_b :  (mirror of left)
//     full R_{(jk mk ak)(w mw)(jb mb ab)} = <jk mk; w mw | jb mb> * Rred[jk,w,jb][ak,ab]

#include "dmrg_su2.hpp"
#include <Eigen/Dense>
#include <map>
#include <tuple>
#include <vector>

namespace qx_su2 {

using Mat = Eigen::MatrixXd;

// ----- bond helpers --------------------------------------------------------
inline int deg_of(const SpinBond& b, int tS) { return b.deg_of(tS); }
inline void bond_add(SpinBond& b, int tS, int deg) {
    for (int i = 0; i < b.n(); ++i) if (b.tS[i] == tS) { b.deg[i] += deg; return; }
    b.tS.push_back(tS); b.deg.push_back(deg);
}

// ----- reduced MPS site ----------------------------------------------------
struct Site {
    SpinBond left, right;
    std::map<std::pair<int,int>, Mat> blk;  // (tjl,tjr) -> (deg_l x deg_r)
    Mat* find(int tjl, int tjr) {
        auto it = blk.find({tjl, tjr});
        return it == blk.end() ? nullptr : &it->second;
    }
    const Mat* find(int tjl, int tjr) const {
        auto it = blk.find({tjl, tjr});
        return it == blk.end() ? nullptr : &it->second;
    }
};

// ----- reduced environment -------------------------------------------------
// key = (tj_ket, mpo_bond_index, tj_bra) -> (deg_ket x deg_bra)
struct Env {
    std::map<std::tuple<int,int,int>, Mat> blk;
    Mat* find(int tjk, int w, int tjb) {
        auto it = blk.find({tjk, w, tjb});
        return it == blk.end() ? nullptr : &it->second;
    }
    const Mat* find(int tjk, int w, int tjb) const {
        auto it = blk.find({tjk, w, tjb});
        return it == blk.end() ? nullptr : &it->second;
    }
};

// ----- recoupling coefficients ---------------------------------------------
// Single convention for BOTH environments: every MPS/env bond couples as
//   jk (x) w -> jb     (m_b = m_k + m_w),
// because the MPO operator component q satisfies mu_bra = mu_ket + q and
// mw_r = mw_l + q, so summing along the chain gives m_bra = m_ket + m_w.
// The site fuses jl (x) 1/2 -> jr on BOTH ket and bra legs.
//
// recoup_left: absorb a site into the left env, old legs (kl,wl,bl) [INPUT,
// "old coupling"] -> new legs (kr,wr,br) [projected, "new coupling"]. Returns R:
//   Lnew_red[jkr,wr,jbr] += R * r_mpo * (Aket^T Lold_red Abra).
inline double recoup_left(int tjkl, int twl, int tjbl,
                          int tk,                       // operator rank (MPO)
                          int tjkr, int twr, int tjbr) {
    const int tphys = 1;
    double num = 0.0, den = 0.0;
    for (int mkr = -tjkr; mkr <= tjkr; mkr += 2)
    for (int mwr = -twr;  mwr <= twr;  mwr += 2) {
        int mbr = mkr + mwr;                  // new coupling: jkr (x) wr -> jbr
        if (mbr < -tjbr || mbr > tjbr) continue;
        double cg_new = clebsch_gordan(tjkr, mkr, twr, mwr, tjbr, mbr);
        if (cg_new == 0.0) continue;
        double val = 0.0;
        for (int mkl = -tjkl; mkl <= tjkl; mkl += 2)
        for (int mwl = -twl;  mwl <= twl;  mwl += 2) {
            int mbl = mkl + mwl;              // old coupling: jkl (x) wl -> jbl
            if (mbl < -tjbl || mbl > tjbl) continue;
            double cg_old = clebsch_gordan(tjkl, mkl, twl, mwl, tjbl, mbl);
            if (cg_old == 0.0) continue;
            int q = mwr - mwl;                // MPO bond: wl (x) k -> wr
            for (int mu = -tphys; mu <= tphys; mu += 2) {    // ket phys
                if (mkl + mu != mkr) continue;               // ket A: jkl (x) 1/2 -> jkr
                double cg_Aket = clebsch_gordan(tjkl, mkl, tphys, mu, tjkr, mkr);
                if (cg_Aket == 0.0) continue;
                int mup = mu + q;             // bra phys: mu_bra = mu_ket + q
                if (mup < -tphys || mup > tphys) continue;
                double cg_Wbond = clebsch_gordan(twl, mwl, tk, q, twr, mwr);
                if (cg_Wbond == 0.0) continue;
                double cg_Wphys = clebsch_gordan(tphys, mu, tk, q, tphys, mup);
                if (cg_Wphys == 0.0) continue;
                if (mbl + mup != mbr) continue;               // bra A: jbl (x) 1/2 -> jbr
                double cg_Abra = clebsch_gordan(tjbl, mbl, tphys, mup, tjbr, mbr);
                if (cg_Abra == 0.0) continue;
                val += cg_old * cg_Aket * cg_Wbond * cg_Wphys * cg_Abra;
            }
        }
        num += cg_new * val;
        den += cg_new * cg_new;
    }
    return den > 0 ? num / den : 0.0;
}

// recoup_right: absorb a site into the right env, old legs (kr,wr,br) [INPUT] ->
// new legs (kl,wl,bl) [projected]. Same coupling jk (x) w -> jb throughout.
inline double recoup_right(int tjkr, int twr, int tjbr,
                           int tk,
                           int tjkl, int twl, int tjbl) {
    const int tphys = 1;
    double num = 0.0, den = 0.0;
    for (int mkl = -tjkl; mkl <= tjkl; mkl += 2)
    for (int mwl = -twl;  mwl <= twl;  mwl += 2) {
        int mbl = mkl + mwl;                  // new coupling: jkl (x) wl -> jbl
        if (mbl < -tjbl || mbl > tjbl) continue;
        double cg_new = clebsch_gordan(tjkl, mkl, twl, mwl, tjbl, mbl);
        if (cg_new == 0.0) continue;
        double val = 0.0;
        for (int mkr = -tjkr; mkr <= tjkr; mkr += 2)
        for (int mwr = -twr;  mwr <= twr;  mwr += 2) {
            int mbr = mkr + mwr;              // old coupling: jkr (x) wr -> jbr
            if (mbr < -tjbr || mbr > tjbr) continue;
            double cg_old = clebsch_gordan(tjkr, mkr, twr, mwr, tjbr, mbr);
            if (cg_old == 0.0) continue;
            int q = mwr - mwl;
            for (int mu = -tphys; mu <= tphys; mu += 2) {
                if (mkl + mu != mkr) continue;               // ket A: jkl (x) 1/2 -> jkr
                double cg_Aket = clebsch_gordan(tjkl, mkl, tphys, mu, tjkr, mkr);
                if (cg_Aket == 0.0) continue;
                int mup = mu + q;
                if (mup < -tphys || mup > tphys) continue;
                double cg_Wbond = clebsch_gordan(twl, mwl, tk, q, twr, mwr);
                if (cg_Wbond == 0.0) continue;
                double cg_Wphys = clebsch_gordan(tphys, mu, tk, q, tphys, mup);
                if (cg_Wphys == 0.0) continue;
                if (mbl + mup != mbr) continue;               // bra A: jbl (x) 1/2 -> jbr
                double cg_Abra = clebsch_gordan(tjbl, mbl, tphys, mup, tjbr, mbr);
                if (cg_Abra == 0.0) continue;
                val += cg_old * cg_Aket * cg_Wbond * cg_Wphys * cg_Abra;
            }
        }
        num += cg_new * val;
        den += cg_new * cg_new;
    }
    return den > 0 ? num / den : 0.0;
}

// ----- grow left environment by absorbing one site -------------------------
// trans: MPO transitions; A: the (ket=bra) site tensor at this position.
inline Env grow_left(const Env& L, const Site& A,
                     const std::vector<SU2Heisenberg::Trans>& trans) {
    Env Lnew;
    for (auto& [key, Lblk] : L.blk) {
        int tjkl = std::get<0>(key), wl = std::get<1>(key), tjbl = std::get<2>(key);
        int twl = SU2Heisenberg::bond_tS(wl);
        for (auto& t : trans) {
            if (t.a != wl) continue;
            int wr = t.b, twr = SU2Heisenberg::bond_tS(wr);
            // ket block jkl -> jkr, bra block jbl -> jbr
            for (auto& [kk, Aket] : A.blk) {
                if (kk.first != tjkl) continue;
                int tjkr = kk.second;
                for (auto& [kb, Abra] : A.blk) {
                    if (kb.first != tjbl) continue;
                    int tjbr = kb.second;
                    double R = recoup_left(tjkl, twl, tjbl, t.tk, tjkr, twr, tjbr);
                    if (R == 0.0) continue;
                    Mat contrib = (R * t.r) * (Aket.transpose() * Lblk * Abra);
                    auto knew = std::make_tuple(tjkr, wr, tjbr);
                    auto it = Lnew.blk.find(knew);
                    if (it == Lnew.blk.end()) Lnew.blk[knew] = contrib;
                    else it->second += contrib;
                }
            }
        }
    }
    return Lnew;
}

// ----- grow right environment by absorbing one site ------------------------
inline Env grow_right(const Env& R, const Site& A,
                      const std::vector<SU2Heisenberg::Trans>& trans) {
    Env Rnew;
    for (auto& [key, Rblk] : R.blk) {
        int tjkr = std::get<0>(key), wr = std::get<1>(key), tjbr = std::get<2>(key);
        int twr = SU2Heisenberg::bond_tS(wr);
        for (auto& t : trans) {
            if (t.b != wr) continue;          // transition wl -> wr
            int wl = t.a, twl = SU2Heisenberg::bond_tS(wl);
            for (auto& [kk, Aket] : A.blk) {
                if (kk.second != tjkr) continue;
                int tjkl = kk.first;
                for (auto& [kb, Abra] : A.blk) {
                    if (kb.second != tjbr) continue;
                    int tjbl = kb.first;
                    double Rc = recoup_right(tjkr, twr, tjbr, t.tk, tjkl, twl, tjbl);
                    if (Rc == 0.0) continue;
                    Mat contrib = (Rc * t.r) * (Aket * Rblk * Abra.transpose());
                    auto knew = std::make_tuple(tjkl, wl, tjbl);
                    auto it = Rnew.blk.find(knew);
                    if (it == Rnew.blk.end()) Rnew.blk[knew] = contrib;
                    else it->second += contrib;
                }
            }
        }
    }
    return Rnew;
}

} // namespace qx_su2
