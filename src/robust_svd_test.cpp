// Gate for the BDCSVD fixes (cmake/eigen_bdcsvd_fixes.cmake) and the JacobiSVD fallback
// (robust_svd.hpp).
//
// (a) Upper-bidiagonal matrices with an O(1) block in rows 0-32, weakly coupled to a block
//     scaled by ~1e-295 in rows 33-63, make perturbCol0 reach perm(l-1) at l == 0. Eigen 3.4.0
//     as released reads out of bounds there and crashes; with the fix it reports
//     NumericalIssue, and robust_thin_bdcsvd must answer with JacobiSVD. A build whose Eigen
//     was not fixed therefore crashes or fails here.
// (b) The total-deflation test must compare |col0|: on Eigen 3.4.1's own regression matrices
//     (its test compare_bdc_jacobi_instance, switch size 3), 3.4.0 as released was off by 1e-2.
#include "robust_svd.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <complex>
#include <cstdio>
#include <random>

using Eigen::MatrixXcd;
using Eigen::MatrixXd;

static MatrixXd guard_matrix(std::mt19937_64& rng, double s, double ratio, int n = 64) {
    std::uniform_real_distribution<double> u(0.5, 1.0);
    MatrixXd A = MatrixXd::Zero(n, n);
    for (int i = 0; i < n; ++i) {
        double d, e;
        if (i <= 32)      { d = u(rng); e = 0.3 * u(rng); }
        else if (i <= 47) { d = s * ratio * u(rng); e = s * u(rng); }
        else if (i == 48) { d = s * u(rng); e = s * u(rng); }
        else              { d = s * u(rng); e = s * 0.3 * u(rng); }
        if (i == 32) e = s * u(rng);   // the weak coupling of the two halves
        A(i, i) = d;
        if (i + 1 < n) A(i, i + 1) = e;
    }
    return A;
}

// Relative reconstruction error, orthonormality of U and V, and descending order.
template <class M>
static bool check(const char* label, const M& X, const qx::ThinSVD<M>& r) {
    const Eigen::Index k = std::min(X.rows(), X.cols());
    bool shapes = r.U.rows() == X.rows() && r.U.cols() == k && r.V.rows() == X.cols() &&
                  r.V.cols() == k && r.s.size() == k;
    if (!shapes) {
        std::printf("  %s: wrong shapes  FAIL\n", label);
        return false;
    }
    const M R = r.U * r.s.asDiagonal() * r.V.adjoint();
    const double recon = (X - R).norm() / X.norm();
    const double eu = (r.U.adjoint() * r.U - M::Identity(k, k)).norm();
    const double ev = (r.V.adjoint() * r.V - M::Identity(k, k)).norm();
    bool desc = true;
    for (Eigen::Index i = 1; i < k; ++i) desc = desc && r.s(i - 1) >= r.s(i);
    const bool ok = recon < 1e-12 && eu < 1e-12 && ev < 1e-12 && desc;
    std::printf("  %s: recon %.1e, |U*U-I| %.1e, |V*V-I| %.1e, descending %d  %s\n", label, recon,
                eu, ev, (int)desc, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: a crash in BDCSVD shows where
    int fails = 0;

    std::printf("(a) perturbCol0's l == 0 case: BDCSVD reports it, the fallback answers\n");
    std::mt19937_64 rng(12345);
    int tried = 0, reported = 0, bad = 0;
    MatrixXd first;
    for (double s : {1e-295, 1e-300})
        for (double ratio : {0.3, 0.03})
            for (int t = 0; t < 4; ++t) {
                const MatrixXd A = guard_matrix(rng, s, ratio);
                ++tried;
                Eigen::BDCSVD<MatrixXd> bdc(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
                if (bdc.info() != Eigen::Success) {
                    ++reported;
                    if (first.size() == 0) first = A;
                }
                if (!check("  real", A, qx::robust_thin_bdcsvd(A))) ++bad;
            }
    std::printf("  BDCSVD reported a numerical issue on %d of %d matrices\n", reported, tried);
    if (reported == 0) {
        std::printf("  the matrices no longer reach the guard, so this gate tests nothing  FAIL\n");
        ++fails;
    }
    if (bad) ++fails;
    if (first.size()) {
        const MatrixXcd C = first.cast<std::complex<double>>() * std::complex<double>(0.6, 0.8);
        if (!check("complex", C, qx::robust_thin_bdcsvd(C))) ++fails;
    }

    std::printf("(b) total deflation compares |col0|: Eigen 3.4.1's regression matrices\n");
    for (int structured = 0; structured < 2; ++structured) {
        MatrixXd m(4, 3);
        if (structured)
            m << 1.056293, 0, 0, -0.336468, 0.907359, 0, -1.566245, 0, 0.149150, -0.1, 0, 0;
        else
            m << 0.882336, 18.3914, -26.7921, -5.58135, 17.1931, -24.0892, -20.794, 8.68496,
                -4.83103, -8.4981, -10.5451, 23.9072;
        Eigen::BDCSVD<MatrixXd> b(m.rows(), m.cols(), 0);
        b.setSwitchSize(3);
        b.compute(m);
        Eigen::JacobiSVD<MatrixXd> j(m);
        const double diff =
            (b.singularValues() - j.singularValues()).norm() / j.singularValues().norm();
        const bool ok = b.info() == Eigen::Success && diff < 1e-12;
        std::printf("  structured=%d: |s_BDCSVD - s_Jacobi| / |s| = %.1e  %s\n", structured, diff,
                    ok ? "OK" : "FAIL");
        fails += !ok;
    }

    std::printf(fails ? "ROBUST SVD TEST: FAIL\n" : "ROBUST SVD TEST: OK\n");
    return fails ? 1 : 0;
}
