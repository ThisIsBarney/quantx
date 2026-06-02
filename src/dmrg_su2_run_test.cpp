// Validation of the full SU(2) two-site DMRG (stages 5-6): ground-state energy
// of the open Heisenberg chain  H = J sum_i sigma_i . sigma_{i+1}  versus exact
// dense diagonalization, for N = 4, 6, 8. No truncation (exact bonds) -> must
// match to ~1e-8.
#include "dmrg_su2_run.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <cmath>
#include <complex>
#include <vector>

using namespace qx_su2;

// dense ground energy of the spin-1 Heisenberg chain  H = J sum S_i.S_j  (OBC).
static double dense_ground_spin1(int N, double J) {
    using Cm = Eigen::MatrixXcd;
    using Cx = std::complex<double>;
    double s2 = std::sqrt(2.0);
    Cm Sz(3,3); Sz.setZero(); Sz(0,0) = 1; Sz(2,2) = -1;
    Cm Sp(3,3); Sp.setZero(); Sp(0,1) = s2; Sp(1,2) = s2;     // raising
    Cm Sm = Sp.adjoint();
    Cm Sx = 0.5 * (Sp + Sm);
    Cm Sy = Cx(0, -0.5) * (Sp - Sm);                          // (Sp-Sm)/(2i)
    Cm I3 = Cm::Identity(3, 3);
    auto kron = [](const Cm& A, const Cm& B) {
        Cm C(A.rows()*B.rows(), A.cols()*B.cols());
        for (int i = 0; i < A.rows(); ++i)
            for (int j = 0; j < A.cols(); ++j)
                C.block(i*B.rows(), j*B.cols(), B.rows(), B.cols()) = A(i,j) * B;
        return C;
    };
    auto embed = [&](const Cm& op, int site) {
        Cm r(1,1); r(0,0) = 1;
        for (int k = 0; k < N; ++k) r = kron(r, (k == site) ? op : I3);
        return r;
    };
    int d = 1; for (int i = 0; i < N; ++i) d *= 3;
    Cm H = Cm::Zero(d, d);
    for (int i = 0; i + 1 < N; ++i)
        for (const Cm* P : {&Sx, &Sy, &Sz})
            H += J * (embed(*P, i) * embed(*P, i + 1));
    Eigen::SelfAdjointEigenSolver<Cm> es(H);
    return es.eigenvalues()(0);
}

static int failures = 0;

// dense ground energy of H = J sum sigma_i . sigma_{i+1}, OBC, via full ED.
static double dense_ground(int N, double J) {
    int dim = 1 << N;
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(dim, dim);
    // sigma matrices (Pauli)
    auto bit = [](int s, int i){ return (s >> i) & 1; };  // 0 = up, 1 = down
    for (int s = 0; s < dim; ++s) {
        for (int i = 0; i + 1 < N; ++i) {
            int si = bit(s, i), sj = bit(s, i+1);
            // sigma_z sigma_z : +1 if equal, -1 if different
            H(s, s) += J * (si == sj ? 1.0 : -1.0);
            // sigma_x sigma_x + sigma_y sigma_y = 2 (flip both) when opposite
            if (si != sj) {
                int s2 = s ^ (1 << i) ^ (1 << (i+1));
                H(s2, s) += J * 2.0;
            }
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(H);
    return es.eigenvalues()(0);
}

static void run(int N) {
    double J = 1.0;
    double ref = dense_ground(N, J);
    double e = run_su2_dmrg(N, J, /*max_mult*/0, /*sweeps*/6, /*seed*/7);
    double err = std::abs(e - ref);
    bool ok = err < 1e-7;
    if (!ok) ++failures;
    printf("  N=%-2d  SU(2)=%+.10f  dense=%+.10f  err=%.2e  %s\n",
           N, e, ref, err, ok ? "OK" : "FAIL");
}

// Truncated run vs dense: with a modest multiplet cap the energy should still
// be very close to exact (gapless Heisenberg, but small N).
static void run_trunc(int N, int max_mult) {
    double J = 1.0;
    double ref = dense_ground(N, J);
    double e = run_su2_dmrg(N, J, max_mult, /*sweeps*/8, /*seed*/7);
    double err = std::abs(e - ref);
    bool ok = err < 1e-4;
    if (!ok) ++failures;
    printf("  N=%-2d (cap %d mult)  SU(2)=%+.8f  dense=%+.8f  err=%.2e  %s\n",
           N, max_mult, e, ref, err, ok ? "OK" : "FAIL");
}

// Show the SU(2) payoff: at the middle bond of the exact MPS, how many
// multiplets SU(2) tracks vs the equivalent U(1) (Sz) state count.
static void show_reduction(int N) {
    auto bonds = exact_bonds(N);
    const SpinBond& b = bonds[N/2];
    int n_mult = 0, n_u1 = 0;
    for (int i = 0; i < b.n(); ++i) {
        n_mult += b.deg[i];                 // reduced multiplets stored
        n_u1   += b.deg[i] * (b.tS[i] + 1); // each spin-S multiplet = 2S+1 = tS+1 states
    }
    printf("  N=%-3d middle bond: %3d SU(2) multiplets  vs %4d U(1) states  (%.2fx fewer)\n",
           N, n_mult, n_u1, (double)n_u1 / n_mult);
}

// spin-1 chain: SU(2) (tS_phys=2) vs dense ED of H = J sum S.S
static void run_spin1(int N, int max_mult) {
    double J = 1.0;
    double ref = dense_ground_spin1(N, J);
    double e = run_su2_dmrg(N, J, max_mult, /*sweeps*/8, /*seed*/7, /*tS_phys*/2);
    double err = std::abs(e - ref);
    bool ok = err < 1e-6;
    if (!ok) ++failures;
    printf("  N=%-2d (S=1, cap %d)  SU(2)=%+.8f  dense=%+.8f  err=%.2e  %s\n",
           N, max_mult, e, ref, err, ok ? "OK" : "FAIL");
}

int main() {
    printf("SU(2) two-site DMRG vs dense ED (H = J sum sigma.sigma, OBC):\n");
    run(4);
    run(6);
    run(8);
    printf("Truncated (multiplet cap):\n");
    run_trunc(10, 20);
    run_trunc(12, 30);
    printf("Spin-1 Heisenberg chain (H = J sum S.S, Haldane) vs dense ED:\n");
    run_spin1(2, 0);
    run_spin1(4, 0);
    run_spin1(6, 40);
    printf("SU(2) multiplet reduction (exact middle bond):\n");
    show_reduction(8);
    show_reduction(16);
    show_reduction(24);
    show_reduction(32);
    printf("\n%s\n", failures == 0 ? "ALL PASSED" : "SOME FAILED");
    return failures == 0 ? 0 : 1;
}
