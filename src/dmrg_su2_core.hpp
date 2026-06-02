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
inline double recoup_left_impl(int tjkl, int twl, int tjbl,
                          int tk,                       // operator rank (MPO)
                          int tjkr, int twr, int tjbr, int tphys) {
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
// memoized wrapper (labels-only; recoupling is reused across sweeps)
inline double recoup_left(int tjkl, int twl, int tjbl, int tk,
                          int tjkr, int twr, int tjbr, int tphys = 1) {
    static std::map<std::array<int,8>, double> cache;
    std::array<int,8> key{tjkl, twl, tjbl, tk, tjkr, twr, tjbr, tphys};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    double r = recoup_left_impl(tjkl, twl, tjbl, tk, tjkr, twr, tjbr, tphys);
    cache.emplace(key, r);
    return r;
}

// recoup_right: absorb a site into the right env, old legs (kr,wr,br) [INPUT] ->
// new legs (kl,wl,bl) [projected]. Same coupling jk (x) w -> jb throughout.
inline double recoup_right_impl(int tjkr, int twr, int tjbr,
                           int tk,
                           int tjkl, int twl, int tjbl, int tphys) {
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
inline double recoup_right(int tjkr, int twr, int tjbr, int tk,
                           int tjkl, int twl, int tjbl, int tphys = 1) {
    static std::map<std::array<int,8>, double> cache;
    std::array<int,8> key{tjkr, twr, tjbr, tk, tjkl, twl, tjbl, tphys};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    double r = recoup_right_impl(tjkr, twr, tjbr, tk, tjkl, twl, tjbl, tphys);
    cache.emplace(key, r);
    return r;
}

// ----- grow left environment by absorbing one site -------------------------
// trans: MPO transitions; A: the (ket=bra) site tensor at this position.
inline Env grow_left(const Env& L, const Site& A,
                     const std::vector<SU2Heisenberg::Trans>& trans, int tphys = 1) {
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
                    double R = recoup_left(tjkl, twl, tjbl, t.tk, tjkr, twr, tjbr, tphys);
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
                      const std::vector<SU2Heisenberg::Trans>& trans, int tphys = 1) {
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
                    double Rc = recoup_right(tjkr, twr, tjbr, t.tk, tjkl, twl, tjbl, tphys);
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

// ----- two-site reduced wavefunction --------------------------------------
// Theta keyed by (tjl, tjc, tjr): sequential fusion tree jl (x) 1/2 -> jc,
// jc (x) 1/2 -> jr. Matrix is (deg_l x deg_r) over the outer MPS bond mults.
using Key3 = std::tuple<int,int,int>;
using Theta = std::map<Key3, Mat>;

// Recoupling for the two-site H_eff. Computed by CG-sum over the full network:
//   left env (jl (x) wl -> jl'), W1 (wl->wm, rank k1) on site1,
//   W2 (wm->wr, rank k2) on site2, right env (jr (x) wr -> jr').
// Input fusion tree (jl,jc,jr) -> output fusion tree (jl',jc',jr').
inline double recoup_2site_impl(int tjl, int tjc, int tjr,
                           int tjlp, int tjcp, int tjrp,
                           int twl, int twm, int twr,
                           int tk1, int tk2, int tp) {
    double num = 0.0, den = 0.0;
    for (int mlp = -tjlp; mlp <= tjlp; mlp += 2)
    for (int mu1p = -tp; mu1p <= tp; mu1p += 2) {
        int mcp = mlp + mu1p;
        if (mcp < -tjcp || mcp > tjcp) continue;
        double cgO1 = clebsch_gordan(tjlp, mlp, tp, mu1p, tjcp, mcp);
        if (cgO1 == 0.0) continue;
        for (int mu2p = -tp; mu2p <= tp; mu2p += 2) {
            int mrp = mcp + mu2p;
            if (mrp < -tjrp || mrp > tjrp) continue;
            double cgO2 = clebsch_gordan(tjcp, mcp, tp, mu2p, tjrp, mrp);
            if (cgO2 == 0.0) continue;
            double Pout = cgO1 * cgO2;
            double full = 0.0;
            for (int ml = -tjl; ml <= tjl; ml += 2)
            for (int mu1 = -tp; mu1 <= tp; mu1 += 2) {
                int mc = ml + mu1;
                if (mc < -tjc || mc > tjc) continue;
                double cgI1 = clebsch_gordan(tjl, ml, tp, mu1, tjc, mc);
                if (cgI1 == 0.0) continue;
                int mwl = mlp - ml;                    // Lenv: ml' = ml + mwl
                if (mwl < -twl || mwl > twl) continue;
                double cgL = clebsch_gordan(tjl, ml, twl, mwl, tjlp, mlp);
                if (cgL == 0.0) continue;
                int q1 = mu1p - mu1;                   // W1 phys: mu1' = mu1 + q1
                int mwm = mwl + q1;                    // W1 bond
                if (mwm < -twm || mwm > twm) continue;
                double cgWb1 = clebsch_gordan(twl, mwl, tk1, q1, twm, mwm);
                if (cgWb1 == 0.0) continue;
                double cgWp1 = clebsch_gordan(tp, mu1, tk1, q1, tp, mu1p);
                if (cgWp1 == 0.0) continue;
                for (int mu2 = -tp; mu2 <= tp; mu2 += 2) {
                    int mr = mc + mu2;
                    if (mr < -tjr || mr > tjr) continue;
                    double cgI2 = clebsch_gordan(tjc, mc, tp, mu2, tjr, mr);
                    if (cgI2 == 0.0) continue;
                    int q2 = mu2p - mu2;               // W2 phys
                    int mwr = mwm + q2;                // W2 bond
                    if (mwr < -twr || mwr > twr) continue;
                    if (mr + mwr != mrp) continue;     // Renv: mr' = mr + mwr
                    double cgWb2 = clebsch_gordan(twm, mwm, tk2, q2, twr, mwr);
                    if (cgWb2 == 0.0) continue;
                    double cgWp2 = clebsch_gordan(tp, mu2, tk2, q2, tp, mu2p);
                    if (cgWp2 == 0.0) continue;
                    double cgR = clebsch_gordan(tjr, mr, twr, mwr, tjrp, mrp);
                    if (cgR == 0.0) continue;
                    full += cgI1 * cgI2 * cgL * cgR * cgWb1 * cgWp1 * cgWb2 * cgWp2;
                }
            }
            num += Pout * full;
            den += Pout * Pout;
        }
    }
    return den > 0 ? num / den : 0.0;
}
inline double recoup_2site(int tjl, int tjc, int tjr,
                           int tjlp, int tjcp, int tjrp,
                           int twl, int twm, int twr, int tk1, int tk2, int tp = 1) {
    static std::map<std::array<int,12>, double> cache;
    std::array<int,12> key{tjl, tjc, tjr, tjlp, tjcp, tjrp, twl, twm, twr, tk1, tk2, tp};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    double r = recoup_2site_impl(tjl, tjc, tjr, tjlp, tjcp, tjrp, twl, twm, twr, tk1, tk2, tp);
    cache.emplace(key, r);
    return r;
}

// Two-site effective Hamiltonian matvec: Theta' = H_eff Theta.
// L: left env at site1's left bond (ket jl, MPO wl, bra jl'). R: right env at
// site2's right bond (ket jr, MPO wr, bra jr').
inline Theta apply_heff_2site(const Theta& th, const Env& L, const Env& R,
                              const std::vector<SU2Heisenberg::Trans>& trans,
                              int tphys = 1) {
    Theta out;
    for (auto& [kin, Th] : th) {
        int tjl = std::get<0>(kin), tjc = std::get<1>(kin), tjr = std::get<2>(kin);
        for (auto& [kL, Lb] : L.blk) {
            if (std::get<0>(kL) != tjl) continue;
            int wl = std::get<1>(kL), tjlp = std::get<2>(kL);
            int twl = SU2Heisenberg::bond_tS(wl);
            for (auto& [kR, Rb] : R.blk) {
                if (std::get<0>(kR) != tjr) continue;
                int wr = std::get<1>(kR), tjrp = std::get<2>(kR);
                int twr = SU2Heisenberg::bond_tS(wr);
                for (auto& t1 : trans) {
                    if (t1.a != wl) continue;
                    int wm = t1.b, twm = SU2Heisenberg::bond_tS(wm);
                    for (auto& t2 : trans) {
                        if (t2.a != wm || t2.b != wr) continue;
                        int lo = std::max(std::abs(tjlp - tphys), std::abs(tjrp - tphys));
                        int hi = std::min(tjlp + tphys, tjrp + tphys);
                        for (int tjcp = lo; tjcp <= hi; tjcp += 2) {
                            double Rc = recoup_2site(tjl, tjc, tjr, tjlp, tjcp, tjrp,
                                                     twl, twm, twr, t1.tk, t2.tk, tphys);
                            if (Rc == 0.0) continue;
                            Mat contrib = (Rc * t1.r * t2.r) * (Lb.transpose() * Th * Rb);
                            Key3 ko{tjlp, tjcp, tjrp};
                            auto it = out.find(ko);
                            if (it == out.end()) out[ko] = contrib;
                            else it->second += contrib;
                        }
                    }
                }
            }
        }
    }
    return out;
}

} // namespace qx_su2
