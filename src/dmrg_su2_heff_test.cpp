// Validation of the SU(2) two-site H_eff matvec (stage 4): apply H_eff to a
// reduced two-site wavefunction, expand the result to full U(1), and compare to
// a dense full-network contraction (left env . W1 . W2 . right env . Theta).
#include "dmrg_su2_core.hpp"
#include <cstdio>
#include <cmath>
#include <random>

using namespace qx_su2;

static int failures = 0;
static void check(const char* name, double maxerr, double tol = 1e-10) {
    bool ok = maxerr < tol && std::isfinite(maxerr);
    if (!ok) ++failures;
    printf("  %-44s maxerr=%.3e  %s\n", name, maxerr, ok ? "OK" : "FAIL");
}
static std::mt19937 rng(2024);
static double rnd() { std::uniform_real_distribution<double> d(-1, 1); return d(rng); }

static double Welem(const std::vector<SU2Heisenberg::Trans>& tr, int wa, int wb,
                    int mwa, int mwb, int mu_bra, int mu_ket) {
    double v = 0;
    for (auto& t : tr) if (t.a == wa && t.b == wb)
        v += SU2Heisenberg::elem(t, mwa, mwb, mu_bra, mu_ket);
    return v;
}

int main() {
    auto trans = SU2Heisenberg::transitions(1.0);

    // Bond structure (deg 1): left bond {1/2}, intermediate {0,1}, right {1/2,3/2}.
    // Left env on ket/bra jl = 1/2; right env on ket/bra jr in {1/2,3/2}.
    Env L, R;
    for (int w : {0, 1, 2}) {
        int tw = SU2Heisenberg::bond_tS(w);
        if (tri_ok(1, tw, 1)) L.blk[{1, w, 1}] = (Mat(1,1) << rnd()).finished();
    }
    for (int tjr : {1, 3})
        for (int w : {0, 1, 2}) {
            int tw = SU2Heisenberg::bond_tS(w);
            if (tri_ok(tjr, tw, tjr)) R.blk[{tjr, w, tjr}] = (Mat(1,1) << rnd()).finished();
        }

    // Theta blocks: (jl,jc,jr) valid fusion paths from jl=1/2.
    Theta th;
    for (int tjc : {0, 2})
        for (int tjr : {1, 3}) {
            if (!tri_ok(1, 1, tjc)) continue;       // jl=1/2 (x) 1/2 -> jc
            if (!tri_ok(tjc, 1, tjr)) continue;     // jc (x) 1/2 -> jr
            th[{1, tjc, tjr}] = (Mat(1,1) << rnd()).finished();
        }

    Theta out = apply_heff_2site(th, L, R, trans);

    // Validate: expand(out) == dense contraction, for all output components.
    double maxerr = 0;
    // Collect output (jl',jr') pairs present.
    for (int tjlp : {1})
    for (int tjrp : {1, 3}) {
        for (int mlp = -tjlp; mlp <= tjlp; mlp += 2)
        for (int mu1p = -1; mu1p <= 1; mu1p += 2)
        for (int mu2p = -1; mu2p <= 1; mu2p += 2) {
            int mcp = mlp + mu1p, mrp = mcp + mu2p;
            if (mrp < -tjrp || mrp > tjrp) continue;

            // lhs = expand(out): sum over jc'
            double lhs = 0;
            for (int tjcp : {0, 2}) {
                if (mcp < -tjcp || mcp > tjcp) continue;
                auto it = out.find({tjlp, tjcp, tjrp});
                if (it == out.end()) continue;
                double cg1 = clebsch_gordan(tjlp, mlp, 1, mu1p, tjcp, mcp);
                double cg2 = clebsch_gordan(tjcp, mcp, 1, mu2p, tjrp, mrp);
                lhs += cg1 * cg2 * it->second(0, 0);
            }

            // rhs = dense full-network contraction
            double rhs = 0;
            for (auto& [kin, Th] : th) {
                int tjl = std::get<0>(kin), tjc = std::get<1>(kin), tjr = std::get<2>(kin);
                for (int ml = -tjl; ml <= tjl; ml += 2)
                for (int mu1 = -1; mu1 <= 1; mu1 += 2) {
                    int mc = ml + mu1;
                    if (mc < -tjc || mc > tjc) continue;
                    double cgI1 = clebsch_gordan(tjl, ml, 1, mu1, tjc, mc);
                    if (cgI1 == 0) continue;
                    for (int mu2 = -1; mu2 <= 1; mu2 += 2) {
                        int mr = mc + mu2;
                        if (mr < -tjr || mr > tjr) continue;
                        double cgI2 = clebsch_gordan(tjc, mc, 1, mu2, tjr, mr);
                        if (cgI2 == 0) continue;
                        double thv = cgI1 * cgI2 * Th(0, 0);
                        // left env block (tjl, wl, tjlp)
                        for (auto& [kL, Lb] : L.blk) {
                            if (std::get<0>(kL) != tjl || std::get<2>(kL) != tjlp) continue;
                            int wl = std::get<1>(kL), twl = SU2Heisenberg::bond_tS(wl);
                            int mwl = mlp - ml;
                            if (mwl < -twl || mwl > twl) continue;
                            double cgL = clebsch_gordan(tjl, ml, twl, mwl, tjlp, mlp);
                            if (cgL == 0) continue;
                            // right env block (tjr, wr, tjrp)
                            for (auto& [kR, Rb] : R.blk) {
                                if (std::get<0>(kR) != tjr || std::get<2>(kR) != tjrp) continue;
                                int wr = std::get<1>(kR), twr = SU2Heisenberg::bond_tS(wr);
                                int mwr = mrp - mr;
                                if (mwr < -twr || mwr > twr) continue;
                                double cgR = clebsch_gordan(tjr, mr, twr, mwr, tjrp, mrp);
                                if (cgR == 0) continue;
                                int q1 = mu1p - mu1, mwm = mwl + q1;
                                for (auto& t1 : trans) {
                                    if (t1.a != wl) continue;
                                    int wm = t1.b, twm = SU2Heisenberg::bond_tS(wm);
                                    if (mwm < -twm || mwm > twm) continue;
                                    double w1 = Welem(trans, wl, wm, mwl, mwm, mu1p, mu1);
                                    if (w1 == 0) continue;
                                    double w2 = Welem(trans, wm, wr, mwm, mwr, mu2p, mu2);
                                    if (w2 == 0) continue;
                                    rhs += thv * cgL * Lb(0,0) * w1 * w2 * cgR * Rb(0,0);
                                }
                            }
                        }
                    }
                }
            }
            maxerr = std::max(maxerr, std::abs(lhs - rhs));
        }
    }
    printf("SU(2) two-site H_eff (reduced vs dense full-network contraction):\n");
    check("apply_heff_2site", maxerr);

    printf("\n%s\n", failures == 0 ? "ALL PASSED" : "SOME FAILED");
    return failures == 0 ? 0 : 1;
}
