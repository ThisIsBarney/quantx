#pragma once
// SU(2) angular-momentum algebra: Wigner 3j / 6j / 9j symbols and Clebsch-Gordan
// coefficients. Foundation for SU(2)-symmetric (non-abelian) DMRG.
//
// Convention: all spins/projections are passed as DOUBLED integers ("two-j"), so a
// spin-1/2 is tj = 1, a spin-1 is tj = 2, projection m = +1/2 is tm = +1, etc. This
// keeps everything in exact integer arithmetic. Quantities like (j+m) or
// (j1+j2-j3) are then (tj+tm)/2, (tj1+tj2-tj3)/2, which are integers for valid
// (triangle-satisfying, integer-(j+m)) inputs.

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace qx_su2 {

// Cached log-factorials (numerically stable for moderate spins).
inline double lnfact(int n) {
    static std::vector<double> tbl = {0.0};
    while ((int)tbl.size() <= n) {
        int k = (int)tbl.size();
        tbl.push_back(tbl.back() + std::log((double)k));
    }
    return tbl[n];
}
inline double fact(int n) { return std::exp(lnfact(n)); }

// Triangle condition on doubled args: |a-b| <= c <= a+b and a+b+c even.
inline bool tri_ok(int ta, int tb, int tc) {
    return tc >= std::abs(ta - tb) && tc <= ta + tb && ((ta + tb + tc) % 2 == 0);
}

// log of the triangle coefficient Delta(a,b,c)^2 = (a+b-c)!(a-b+c)!(-a+b+c)!/(a+b+c+1)!
inline double ln_tri(int ta, int tb, int tc) {
    int s1 = (ta + tb - tc) / 2, s2 = (ta - tb + tc) / 2,
        s3 = (-ta + tb + tc) / 2, s4 = (ta + tb + tc) / 2 + 1;
    return lnfact(s1) + lnfact(s2) + lnfact(s3) - lnfact(s4);
}

// Wigner 3j symbol  ( ta tb tc ; tA tB tC )  (all doubled).
inline double wigner3j(int ta, int tA, int tb, int tB, int tc, int tC) {
    if (tA + tB + tC != 0) return 0.0;
    if (std::abs(tA) > ta || std::abs(tB) > tb || std::abs(tC) > tc) return 0.0;
    if ((ta + tA) % 2 || (tb + tB) % 2 || (tc + tC) % 2) return 0.0;  // j+m integer
    if (!tri_ok(ta, tb, tc)) return 0.0;

    int kmin = std::max({0, (tb - tc - tA) / 2, (ta - tc + tB) / 2});
    int kmax = std::min({(ta + tb - tc) / 2, (ta - tA) / 2, (tb + tB) / 2});
    if (kmin > kmax) return 0.0;

    double lnpre = 0.5 * (ln_tri(ta, tb, tc)
        + lnfact((ta + tA) / 2) + lnfact((ta - tA) / 2)
        + lnfact((tb + tB) / 2) + lnfact((tb - tB) / 2)
        + lnfact((tc + tC) / 2) + lnfact((tc - tC) / 2));

    double sum = 0.0;
    for (int k = kmin; k <= kmax; ++k) {
        double lnterm = lnfact(k) + lnfact((ta + tb - tc) / 2 - k)
            + lnfact((ta - tA) / 2 - k) + lnfact((tb + tB) / 2 - k)
            + lnfact((tc - tb + tA) / 2 + k) + lnfact((tc - ta - tB) / 2 + k);
        sum += ((k & 1) ? -1.0 : 1.0) * std::exp(lnpre - lnterm);
    }
    int phase = (ta - tb - tC) / 2;
    return ((phase & 1) ? -1.0 : 1.0) * sum;
}

// Clebsch-Gordan coefficient <ta tA ; tb tB | tc tC>  (all doubled).
inline double clebsch_gordan(int ta, int tA, int tb, int tB, int tc, int tC) {
    if (tA + tB != tC) return 0.0;
    double w = wigner3j(ta, tA, tb, tB, tc, -tC);
    if (w == 0.0) return 0.0;
    int phase = (ta - tb + tC) / 2;
    return ((phase & 1) ? -1.0 : 1.0) * std::sqrt((double)(tc + 1)) * w;
}

// Wigner 6j symbol  { ta tb tc ; td te tf }  (all doubled), via the Racah formula.
inline double wigner6j(int ta, int tb, int tc, int td, int te, int tf) {
    if (!tri_ok(ta, tb, tc) || !tri_ok(ta, te, tf) ||
        !tri_ok(td, tb, tf) || !tri_ok(td, te, tc)) return 0.0;

    double lnpre = 0.5 * (ln_tri(ta, tb, tc) + ln_tri(ta, te, tf)
                        + ln_tri(td, tb, tf) + ln_tri(td, te, tc));

    int a1 = (ta + tb + tc) / 2, a2 = (ta + te + tf) / 2,
        a3 = (td + tb + tf) / 2, a4 = (td + te + tc) / 2;
    int b1 = (ta + tb + td + te) / 2, b2 = (tb + tc + te + tf) / 2,
        b3 = (ta + tc + td + tf) / 2;

    int tmin = std::max({a1, a2, a3, a4});
    int tmax = std::min({b1, b2, b3});
    if (tmin > tmax) return 0.0;

    double sum = 0.0;
    for (int t = tmin; t <= tmax; ++t) {
        double lnterm = lnfact(t - a1) + lnfact(t - a2) + lnfact(t - a3) + lnfact(t - a4)
                      + lnfact(b1 - t) + lnfact(b2 - t) + lnfact(b3 - t);
        double term = std::exp(lnpre + lnfact(t + 1) - lnterm);
        sum += ((t & 1) ? -1.0 : 1.0) * term;
    }
    return sum;
}

// Wigner 9j symbol  { t11 t12 t13 ; t21 t22 t23 ; t31 t32 t33 }  (all doubled),
// expressed as a single sum over a 6j product.
inline double wigner9j(int t11, int t12, int t13,
                       int t21, int t22, int t23,
                       int t31, int t32, int t33) {
    int xmin = std::max({std::abs(t11 - t33), std::abs(t12 - t23), std::abs(t21 - t32)});
    int xmax = std::min({t11 + t33, t12 + t23, t21 + t32});
    double sum = 0.0;
    for (int tx = xmin; tx <= xmax; tx += 2) {
        double s = (tx + 1)
            * wigner6j(t11, t21, t31, t32, t33, tx)
            * wigner6j(t12, t22, t32, t21, tx, t23)
            * wigner6j(t13, t23, t33, tx, t11, t12);
        sum += ((tx & 1) ? -1.0 : 1.0) * s;
    }
    return sum;
}

} // namespace qx_su2
