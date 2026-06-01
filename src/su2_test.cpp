// Validation of the SU(2) angular-momentum algebra (Wigner 3j/6j/9j, CG).
#include "su2.hpp"
#include <cstdio>
#include <cmath>

using namespace qx_su2;

static int failures = 0;
static void check(const char* name, double got, double want, double tol = 1e-12) {
    bool ok = std::abs(got - want) < tol;
    if (!ok) ++failures;
    printf("  %-44s got=%+.10f want=%+.10f  %s\n", name, got, want, ok ? "OK" : "FAIL");
}

int main() {
    const double s2 = std::sqrt(2.0), s3 = std::sqrt(3.0);

    printf("Clebsch-Gordan (known values):\n");
    // <1/2 1/2; 1/2 1/2 | 1 1> = 1
    check("<1/2 1/2;1/2 1/2|1 1>", clebsch_gordan(1,1,1,1,2,2), 1.0);
    // triplet/singlet m=0 mixing
    check("<1/2 1/2;1/2 -1/2|1 0>", clebsch_gordan(1,1,1,-1,2,0), 1.0/s2);
    check("<1/2 1/2;1/2 -1/2|0 0>", clebsch_gordan(1,1,1,-1,0,0), 1.0/s2);
    check("<1/2 -1/2;1/2 1/2|0 0>", clebsch_gordan(1,-1,1,1,0,0), -1.0/s2);
    // spin-1 couplings
    check("<1 0;1 0|0 0>",  clebsch_gordan(2,0,2,0,0,0), -1.0/s3);
    check("<1 0;1 0|2 0>",  clebsch_gordan(2,0,2,0,4,0),  std::sqrt(2.0/3.0));
    check("<1 1;1 -1|0 0>", clebsch_gordan(2,2,2,-2,0,0),  1.0/s3);

    printf("Wigner 6j (known values):\n");
    check("{1 1 1;1 1 1}",        wigner6j(2,2,2,2,2,2), 1.0/6.0);
    check("{1/2 1/2 0;1/2 1/2 1}",wigner6j(1,1,0,1,1,2), 1.0/2.0);

    printf("Clebsch-Gordan orthogonality  sum_{m1,m2} <..|J M><..|J' M'> = d_JJ' d_MM':\n");
    {
        int tj1 = 1, tj2 = 1;  // two spin-1/2
        double err = 0;
        for (int tJ = 0; tJ <= tj1 + tj2; tJ += 2)
        for (int tJp = 0; tJp <= tj1 + tj2; tJp += 2)
        for (int tM = -tJ; tM <= tJ; tM += 2)
        for (int tMp = -tJp; tMp <= tJp; tMp += 2) {
            double s = 0;
            for (int tm1 = -tj1; tm1 <= tj1; tm1 += 2)
            for (int tm2 = -tj2; tm2 <= tj2; tm2 += 2)
                s += clebsch_gordan(tj1, tm1, tj2, tm2, tJ, tM)
                   * clebsch_gordan(tj1, tm1, tj2, tm2, tJp, tMp);
            double want = (tJ == tJp && tM == tMp) ? 1.0 : 0.0;
            err = std::max(err, std::abs(s - want));
        }
        check("CG orthogonality (max err)", err, 0.0, 1e-12);
    }

    printf("Wigner 6j orthogonality  sum_x (2x+1){a b x;c d p}{a b x;c d q} = d_pq/(2p+1):\n");
    {
        int ta = 2, tb = 2, tc = 2, td = 2;  // spins 1
        double err = 0;
        for (int tp = 0; tp <= 4; tp += 2)
        for (int tq = 0; tq <= 4; tq += 2) {
            double s = 0;
            for (int tx = 0; tx <= 4; tx += 2)
                s += (tx + 1) * wigner6j(ta, tb, tx, tc, td, tp)
                              * wigner6j(ta, tb, tx, tc, td, tq);
            double want = (tp == tq && tri_ok(ta, td, tp) && tri_ok(tb, tc, tp))
                          ? 1.0 / (tp + 1) : 0.0;
            if (tp == tq && !(tri_ok(ta, td, tp) && tri_ok(tb, tc, tp))) want = 0.0;
            err = std::max(err, std::abs(s - want));
        }
        check("6j orthogonality (max err)", err, 0.0, 1e-12);
    }

    printf("Wigner 9j reduction (one entry zero)  {a b c;d e f;g h 0}:\n");
    {
        // {a b c; d e f; g h 0} = d(c,f) d(g,h) (-1)^(b+c+d+g) / sqrt((2c+1)(2g+1)) {a b c; e d g}
        int ta=2,tb=2,tc=2, td=1,te=1,tf=2, tg=1,th=1;
        double w9 = wigner9j(ta,tb,tc, td,te,tf, tg,th,0);
        double phase = std::pow(-1.0, (tb+tc+td+tg)/2);
        double want = (tc==tf && tg==th)
            ? phase / std::sqrt((double)((tc+1)*(tg+1))) * wigner6j(ta,tb,tc, te,td,tg)
            : 0.0;
        check("9j reduction", w9, want, 1e-12);
    }

    printf("\n%s\n", failures == 0 ? "ALL PASSED" : "SOME FAILED");
    return failures == 0 ? 0 : 1;
}
