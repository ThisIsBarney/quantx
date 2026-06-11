// Finite-temperature spin conductivity (F2) validation against dense ED.
//
// run_conductivity_thermal_blocks cools the interleaved purified chain to
// |psi(beta)>, applies the total spin current J, real-time evolves the unsourced
// A and the sourced B = J|psi>, and emits the current-current correlator
//   C(t;T) = <J(t) J(0)>_beta = <A(t)|J|B(t)>/Z.
// This test compares it to the exact Lehmann representation from dense ED of the
// physical XXZ chain (N=6 -> 64x64):
//
//   C(t;T) = (1/Z) sum_{m,n} e^{-beta E_m} |<m|J|n>|^2 e^{i(E_m-E_n)t}
//
// with J = sum_l i*J(sigma+_l sigma-_{l+1} - sigma-_l sigma+_{l+1}).
//
// Checks:
//   * C(0) = <J^2>_beta real (imag part ~0): source/normalisation
//   * full C(t;T) re+im vs ED Lehmann at beta=0.5 and beta=2
//   * disentangler invariance: the ancilla gauge is unitary and commutes with the
//     physical current J, so C is exactly invariant (truncation-limited ~1e-9).
#include "dmrg_internal.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using namespace qx_dmrg;

static Mat kron(const Mat& A, const Mat& B) {
    Mat K(A.rows() * B.rows(), A.cols() * B.cols());
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            K.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
    return K;
}
static Mat site_op(int n, int site, const Mat2& o) {
    Mat O = Mat::Identity(1, 1);
    for (int i = 0; i < n; ++i)
        O = kron(O, (i == site) ? Mat(o) : Mat(Mat2::Identity()));
    return O;
}

// Exact Lehmann current-current correlator C(t;beta) over a time grid.
struct EDTransport {
    int dim;
    Eigen::VectorXd E;
    Mat Jeig;   // V^dag J V (current in the energy eigenbasis)

    EDTransport(int n, double Jc, double delta) : dim(1 << n) {
        Mat2 X = pauli('X'), Y = pauli('Y'), Z = pauli('Z');
        Mat2 sp; sp << 0, 1, 0, 0;     // sigma+ = |up><dn|
        Mat2 sm; sm << 0, 0, 1, 0;     // sigma-
        Mat H = Mat::Zero(dim, dim), Jt = Mat::Zero(dim, dim);
        Cx iJ(0.0, Jc);
        for (int i = 0; i < n - 1; ++i) {
            H += Jc * (site_op(n, i, X) * site_op(n, i + 1, X)
                     + site_op(n, i, Y) * site_op(n, i + 1, Y)
                     + delta * site_op(n, i, Z) * site_op(n, i + 1, Z));
            Jt += iJ * (site_op(n, i, sp) * site_op(n, i + 1, sm)
                      - site_op(n, i, sm) * site_op(n, i + 1, sp));
        }
        Eigen::SelfAdjointEigenSolver<Mat> es(H);
        E = es.eigenvalues();
        Jeig = es.eigenvectors().adjoint() * Jt * es.eigenvectors();
    }
    Cx at(double beta, double t) const {
        Eigen::VectorXd p = (-beta * (E.array() - E.minCoeff())).exp();
        double Z = p.sum();
        Cx C(0, 0);
        for (int m = 0; m < dim; ++m)
            for (int nn = 0; nn < dim; ++nn)
                C += (p[m] / Z) * std::norm(Jeig(m, nn))
                     * std::exp(Cx(0, (E[m] - E[nn]) * t));
        return C;
    }
};

static int check_case(int n_phys, double J, double delta, double beta, int n_cool,
                      double dt, int n_steps, int chi, double tol) {
    std::printf("=== sigma(w,T) XXZ N=%d J=%.2f d=%.2f beta=%.2f : cool=%d, dt=%.2f, "
                "%d steps (t->%.1f), chi=%d ===\n",
                n_phys, J, delta, beta, n_cool, dt, n_steps, dt * n_steps, chi);
    EDTransport ed(n_phys, J, delta);

    auto run_chk = [&](bool dis) -> int {
        auto r = run_conductivity_thermal_blocks(n_phys, J, delta, beta, n_cool,
                                                 dt, n_steps, chi, /*cutoff*/1e-12, dis);
        std::printf("  [disentangle=%d] E(beta)=%.5f max_bond=%d  C(0)=%.4f%+.1ei  %.2fs\n",
                    (int)dis, r.E, r.max_bond, r.C_re[0], r.C_im[0], r.wall_time_s);
        if (std::abs(r.C_im[0]) > 1e-7) {
            std::printf("    FAIL C(0) imag=%.2e (should be ~0)\n", r.C_im[0]);
            return 1;
        }
        double worst = 0; int wt = 0;
        for (size_t t = 0; t < r.times.size(); ++t) {
            Cx Ce = ed.at(beta, r.times[t]);
            double e = std::max(std::abs(r.C_re[t] - Ce.real()),
                                std::abs(r.C_im[t] - Ce.imag()));
            if (e > worst) { worst = e; wt = (int)t; }
        }
        bool ok = worst <= tol;
        std::printf("    vs ED Lehmann: worst err=%.2e at t=%.2f  %s\n",
                    worst, r.times[wt], ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    };

    int fails = run_chk(false) + run_chk(true);

    auto a = run_conductivity_thermal_blocks(n_phys, J, delta, beta, n_cool, dt,
                                             n_steps, chi, 1e-12, false);
    auto b = run_conductivity_thermal_blocks(n_phys, J, delta, beta, n_cool, dt,
                                             n_steps, chi, 1e-12, true);
    double dmax = 0;
    for (size_t t = 0; t < a.times.size(); ++t) {
        dmax = std::max(dmax, std::abs(a.C_re[t] - b.C_re[t]));
        dmax = std::max(dmax, std::abs(a.C_im[t] - b.C_im[t]));
    }
    bool inv_ok = dmax <= 1e-7;
    std::printf("  disentangler invariance |C_on - C_off|=%.2e  %s\n\n",
                dmax, inv_ok ? "OK" : "FAIL");
    if (!inv_ok) ++fails;
    return fails;
}

int main() {
    int fails = 0;
    // chi=256 >= 2^6 -> the purified evolution and the J-application are lossless,
    // so the ED match is Trotter-limited (worst ~1.2e-2 at dt=0.05, beta=2, where
    // the colder correlator has sharper features; tol 2e-2 with margin) and the
    // disentangler invariance is exact to ~1e-9 (gauge is unitary, commutes with J).
    fails += check_case(/*n_phys*/6, /*J*/1.0, /*delta*/1.0, /*beta*/0.5,
                        /*n_cool*/20, /*dt*/0.05, /*n_steps*/40, /*chi*/256,
                        /*tol*/2e-2);
    fails += check_case(/*n_phys*/6, /*J*/1.0, /*delta*/1.0, /*beta*/2.0,
                        /*n_cool*/40, /*dt*/0.05, /*n_steps*/40, /*chi*/256,
                        /*tol*/2e-2);
    std::printf("%s\n", fails ? "THERMAL TRANSPORT TEST: FAIL"
                              : "THERMAL TRANSPORT TEST: ALL OK");
    return fails ? 1 : 0;
}
