// Finite-temperature dynamical structure factor (F1) validation against dense ED.
//
// run_spectral_thermal_blocks cools the interleaved purified chain to
// |psi(beta)> = e^{-beta H/2}|psi0>, applies sigma^z at the central physical
// site l0, real-time evolves the unsourced state A and the sourced state B, and
// emits C_{j,l0}(t;T) = <sigma^z_j(t) sigma^z_{l0}(0)>_beta. This test compares
// every (t, j) against the exact Lehmann representation from dense ED of the
// physical XXZ chain (N=6 -> 64x64):
//
//   C_{j,l}(t;T) = (1/Z) sum_{m,n} e^{-beta E_m} <m|Z_j|n><n|Z_l|m> e^{i(E_m-E_n)t}
//
// Checks:
//   * C_{l0,l0}(0) = <(sigma^z)^2> = 1 exactly (source normalisation / Z)
//   * full C_{j,l0}(t;T) re+im vs ED Lehmann at beta=0.5 and beta=4 (the latter
//     is the T->0 limit -> ground-state dynamical correlator)
//   * disentangler invariance: the ancilla-backward gauge is unitary and commutes
//     with the physical sigma^z, so C is exactly invariant (truncation-limited).
//     At chi >= 2^N the evolution is lossless, so disentangle on == off to ~1e-9
//     -- this pins down that the disentangler is a true gauge, not an approximation.
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

// Exact Lehmann C_{j,l0}(t;beta) for all physical j over a time grid.
struct EDSpectral {
    int n, dim, l0;
    Eigen::VectorXd E;
    std::vector<Mat> Zeig;   // Zeig[j] = V^dag sigma^z_j V (eigenbasis)
    Mat Zl;                  // V^dag sigma^z_{l0} V

    EDSpectral(int n_, double J, double delta, int l0_) : n(n_), dim(1 << n_), l0(l0_) {
        Mat2 X = pauli('X'), Y = pauli('Y'), Z = pauli('Z');
        Mat H = Mat::Zero(dim, dim);
        for (int i = 0; i < n - 1; ++i)
            H += J * (site_op(n, i, X) * site_op(n, i + 1, X)
                    + site_op(n, i, Y) * site_op(n, i + 1, Y)
                    + delta * site_op(n, i, Z) * site_op(n, i + 1, Z));
        Eigen::SelfAdjointEigenSolver<Mat> es(H);
        E = es.eigenvalues();
        Mat V = es.eigenvectors();
        Zeig.resize(n);
        for (int j = 0; j < n; ++j) Zeig[j] = V.adjoint() * site_op(n, j, Z) * V;
        Zl = Zeig[l0];
    }
    // C_{j,l0}(t) for all j at one time t and inverse temperature beta.
    std::vector<Cx> at(double beta, double t) const {
        Eigen::VectorXd p = (-beta * (E.array() - E.minCoeff())).exp();
        double Z = p.sum();
        std::vector<Cx> C(n, Cx(0, 0));
        for (int m = 0; m < dim; ++m) {
            for (int nn = 0; nn < dim; ++nn) {
                Cx ph = std::exp(Cx(0, (E[m] - E[nn]) * t)) * (p[m] / Z);
                Cx zl = Zl(nn, m);
                for (int j = 0; j < n; ++j)
                    C[j] += ph * Zeig[j](m, nn) * zl;
            }
        }
        return C;
    }
};

static int check_case(int n_phys, double J, double delta, double beta, int n_cool,
                      double dt, int n_steps, int chi, double tol) {
    const int l0 = n_phys / 2;
    std::printf("=== S(k,w,T) XXZ N=%d J=%.2f d=%.2f beta=%.2f : cool=%d, dt=%.2f, "
                "%d steps (t->%.1f), chi=%d ===\n",
                n_phys, J, delta, beta, n_cool, dt, n_steps, dt * n_steps, chi);

    EDSpectral ed(n_phys, J, delta, l0);

    auto run_chk = [&](bool dis) -> int {
        auto r = run_spectral_thermal_blocks(n_phys, J, delta, beta, n_cool,
                                             dt, n_steps, chi, /*cutoff*/1e-12, dis);
        std::printf("  [disentangle=%d] E(beta)=%.5f max_bond=%d  %.2fs\n",
                    (int)dis, r.E, r.max_bond, r.wall_time_s);
        // C_{l0,l0}(0) == 1
        double c00 = r.C_re[0][l0], c00i = r.C_im[0][l0];
        if (std::abs(c00 - 1.0) > 1e-9 || std::abs(c00i) > 1e-9) {
            std::printf("    FAIL C_{l0,l0}(0)=%.3e+%.3ei (expected 1)\n", c00, c00i);
            return 1;
        }
        double worst = 0; int wt = 0, wj = 0;
        for (size_t t = 0; t < r.times.size(); ++t) {
            auto Ced = ed.at(beta, r.times[t]);
            for (int j = 0; j < n_phys; ++j) {
                double er = std::abs(r.C_re[t][j] - Ced[j].real());
                double ei = std::abs(r.C_im[t][j] - Ced[j].imag());
                double e = std::max(er, ei);
                if (e > worst) { worst = e; wt = (int)t; wj = j; }
            }
        }
        bool ok = worst <= tol;
        std::printf("    vs ED Lehmann: worst err=%.2e at (t=%.2f, j=%d)  %s\n",
                    worst, r.times[wt], wj, ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    };

    int fails = run_chk(false) + run_chk(true);

    // disentangler invariance: C(dis) == C(nodis) to truncation (lossless at chi>=2^N)
    auto a = run_spectral_thermal_blocks(n_phys, J, delta, beta, n_cool, dt, n_steps,
                                         chi, 1e-12, false);
    auto b = run_spectral_thermal_blocks(n_phys, J, delta, beta, n_cool, dt, n_steps,
                                         chi, 1e-12, true);
    double dmax = 0;
    for (size_t t = 0; t < a.times.size(); ++t)
        for (int j = 0; j < n_phys; ++j) {
            dmax = std::max(dmax, std::abs(a.C_re[t][j] - b.C_re[t][j]));
            dmax = std::max(dmax, std::abs(a.C_im[t][j] - b.C_im[t][j]));
        }
    bool inv_ok = dmax <= 1e-7;
    std::printf("  disentangler invariance |C_on - C_off|=%.2e  %s\n\n",
                dmax, inv_ok ? "OK" : "FAIL");
    if (!inv_ok) ++fails;
    return fails;
}

int main() {
    int fails = 0;
    // beta=0.5 (warm) and beta=4 (cold -> ground-state correlator). chi=256 >= 2^6
    // so the purified evolution is lossless: the ED match is Trotter-limited only
    // (2nd order -> ~(dt)^2, here ~1e-2 at dt=0.05, t->2; tol 1.5e-2 with margin),
    // and the disentangler invariance is exact to ~1e-13 (true unitary gauge, not
    // an approximation). The cold case is the T->0 ground-state correlator.
    fails += check_case(/*n_phys*/6, /*J*/1.0, /*delta*/1.0, /*beta*/0.5,
                        /*n_cool*/20, /*dt*/0.05, /*n_steps*/40, /*chi*/256,
                        /*tol*/1.5e-2);
    fails += check_case(/*n_phys*/6, /*J*/1.0, /*delta*/1.0, /*beta*/4.0,
                        /*n_cool*/80, /*dt*/0.05, /*n_steps*/40, /*chi*/256,
                        /*tol*/1.5e-2);
    std::printf("%s\n", fails ? "THERMAL SPECTRAL TEST: FAIL"
                              : "THERMAL SPECTRAL TEST: ALL OK");
    return fails ? 1 : 0;
}
