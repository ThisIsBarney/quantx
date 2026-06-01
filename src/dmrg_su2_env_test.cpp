// Validation of SU(2) environment growth (stage 3): grow a reduced left/right
// environment by one site, expand the reduced result to full U(1) (Sz-resolved)
// tensors, and compare to a plain dense contraction of the expanded inputs.
#include "dmrg_su2_core.hpp"
#include <cstdio>
#include <cmath>
#include <random>

using namespace qx_su2;

static int failures = 0;
static void check(const char* name, double maxerr, double tol = 1e-10) {
    bool ok = maxerr < tol;
    if (!ok) ++failures;
    printf("  %-44s maxerr=%.3e  %s\n", name, maxerr, ok ? "OK" : "FAIL");
}

static std::mt19937 rng(12345);
static double rnd() { std::uniform_real_distribution<double> d(-1, 1); return d(rng); }

// W_full(wl,wr; mwl,mwr; mu_bra,mu_ket): sum over MPO transitions wl->wr.
static double W_full(const std::vector<SU2Heisenberg::Trans>& trans,
                     int wl, int wr, int mwl, int mwr, int mu_bra, int mu_ket) {
    double v = 0;
    for (auto& t : trans)
        if (t.a == wl && t.b == wr) v += SU2Heisenberg::elem(t, mwl, mwr, mu_bra, mu_ket);
    return v;
}

int main() {
    auto trans = SU2Heisenberg::transitions(1.0);

    // ---- LEFT growth test --------------------------------------------------
    // Site A: left bond {1/2}, right bond {0,1} (deg 1 each), blocks (t1->t0),(t1->t2).
    {
        Site A;
        A.left = {{1}, {1}};
        A.right = {{0, 2}, {1, 1}};
        A.blk[{1, 0}] = (Mat(1,1) << rnd()).finished();
        A.blk[{1, 2}] = (Mat(1,1) << rnd()).finished();

        // Left env on ket/bra bond {1/2}; MPO bonds begin/op/done; deg 1.
        Env L;
        for (int w : {0, 1, 2}) {
            int tw = SU2Heisenberg::bond_tS(w);
            if (!tri_ok(1, tw, 1)) continue;        // jb (x) w -> jk = 1/2
            L.blk[{1, w, 1}] = (Mat(1,1) << rnd()).finished();
        }

        Env Lnew = grow_left(L, A, trans);

        // Validate: for every reduced output block and every component, compare
        // expand(Lnew) to the plain dense contraction of expanded inputs.
        double maxerr = 0;
        for (auto& [knew, Lnb] : Lnew.blk) {
            int tjkr = std::get<0>(knew), wr = std::get<1>(knew), tjbr = std::get<2>(knew);
            int twr = SU2Heisenberg::bond_tS(wr);
            for (int mkr = -tjkr; mkr <= tjkr; mkr += 2)
            for (int mwr = -twr;  mwr <= twr;  mwr += 2) {
                int mbr = mkr + mwr;                            // jkr (x) wr -> jbr
                if (mbr < -tjbr || mbr > tjbr) continue;
                double cg_out = clebsch_gordan(tjkr, mkr, twr, mwr, tjbr, mbr);
                double lhs = cg_out * Lnb(0, 0);    // expand(Lnew)

                // dense contraction
                double rhs = 0;
                for (auto& [kL, Lb] : L.blk) {
                    int tjkl = std::get<0>(kL), wl = std::get<1>(kL), tjbl = std::get<2>(kL);
                    int twl = SU2Heisenberg::bond_tS(wl);
                    const Mat* Aket = A.find(tjkl, tjkr);
                    const Mat* Abra = A.find(tjbl, tjbr);
                    if (!Aket || !Abra) continue;
                    for (int mkl = -tjkl; mkl <= tjkl; mkl += 2)
                    for (int mwl = -twl;  mwl <= twl;  mwl += 2) {
                        int mbl = mkl + mwl;                    // jkl (x) wl -> jbl
                        if (mbl < -tjbl || mbl > tjbl) continue;
                        double cgL = clebsch_gordan(tjkl, mkl, twl, mwl, tjbl, mbl);
                        if (cgL == 0) continue;
                        double Lval = cgL * Lb(0, 0);
                        for (int mu = -1; mu <= 1; mu += 2) {
                            if (mkl + mu != mkr) continue;
                            double cgAk = clebsch_gordan(tjkl, mkl, 1, mu, tjkr, mkr);
                            if (cgAk == 0) continue;
                            for (int mup = -1; mup <= 1; mup += 2) {
                                if (mbl + mup != mbr) continue;
                                double w = W_full(trans, wl, wr, mwl, mwr, mup, mu);
                                if (w == 0) continue;
                                double cgAb = clebsch_gordan(tjbl, mbl, 1, mup, tjbr, mbr);
                                if (cgAb == 0) continue;
                                rhs += Lval * cgAk * (*Aket)(0,0) * w * cgAb * (*Abra)(0,0);
                            }
                        }
                    }
                }
                maxerr = std::max(maxerr, std::abs(lhs - rhs));
            }
        }
        printf("LEFT environment growth (reduced vs full U(1) contraction):\n");
        check("left grow_left", maxerr);
    }

    // ---- RIGHT growth test -------------------------------------------------
    {
        Site A;
        A.left = {{0, 2}, {1, 1}};
        A.right = {{1}, {1}};
        A.blk[{0, 1}] = (Mat(1,1) << rnd()).finished();
        A.blk[{2, 1}] = (Mat(1,1) << rnd()).finished();

        Env R;
        for (int w : {0, 1, 2}) {
            int tw = SU2Heisenberg::bond_tS(w);
            if (!tri_ok(1, tw, 1)) continue;        // jk (x) w -> jb = 1/2
            R.blk[{1, w, 1}] = (Mat(1,1) << rnd()).finished();
        }

        Env Rnew = grow_right(R, A, trans);

        double maxerr = 0;
        for (auto& [knew, Rnb] : Rnew.blk) {
            int tjkl = std::get<0>(knew), wl = std::get<1>(knew), tjbl = std::get<2>(knew);
            int twl = SU2Heisenberg::bond_tS(wl);
            for (int mkl = -tjkl; mkl <= tjkl; mkl += 2)
            for (int mwl = -twl;  mwl <= twl;  mwl += 2) {
                int mbl = mkl + mwl;
                if (mbl < -tjbl || mbl > tjbl) continue;
                double cg_out = clebsch_gordan(tjkl, mkl, twl, mwl, tjbl, mbl);
                double lhs = cg_out * Rnb(0, 0);

                double rhs = 0;
                for (auto& [kR, Rb] : R.blk) {
                    int tjkr = std::get<0>(kR), wr = std::get<1>(kR), tjbr = std::get<2>(kR);
                    int twr = SU2Heisenberg::bond_tS(wr);
                    const Mat* Aket = A.find(tjkl, tjkr);
                    const Mat* Abra = A.find(tjbl, tjbr);
                    if (!Aket || !Abra) continue;
                    for (int mkr = -tjkr; mkr <= tjkr; mkr += 2)
                    for (int mwr = -twr;  mwr <= twr;  mwr += 2) {
                        int mbr = mkr + mwr;
                        if (mbr < -tjbr || mbr > tjbr) continue;
                        double cgR = clebsch_gordan(tjkr, mkr, twr, mwr, tjbr, mbr);
                        if (cgR == 0) continue;
                        double Rval = cgR * Rb(0, 0);
                        for (int mu = -1; mu <= 1; mu += 2) {
                            if (mkl + mu != mkr) continue;
                            double cgAk = clebsch_gordan(tjkl, mkl, 1, mu, tjkr, mkr);
                            if (cgAk == 0) continue;
                            for (int mup = -1; mup <= 1; mup += 2) {
                                if (mbl + mup != mbr) continue;
                                double w = W_full(trans, wl, wr, mwl, mwr, mup, mu);
                                if (w == 0) continue;
                                double cgAb = clebsch_gordan(tjbl, mbl, 1, mup, tjbr, mbr);
                                if (cgAb == 0) continue;
                                rhs += Rval * cgAk * (*Aket)(0,0) * w * cgAb * (*Abra)(0,0);
                            }
                        }
                    }
                }
                maxerr = std::max(maxerr, std::abs(lhs - rhs));
            }
        }
        printf("RIGHT environment growth (reduced vs full U(1) contraction):\n");
        check("right grow_right", maxerr);
    }

    printf("\n%s\n", failures == 0 ? "ALL PASSED" : "SOME FAILED");
    return failures == 0 ? 0 : 1;
}
