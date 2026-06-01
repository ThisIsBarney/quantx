// Validation of the full SU(2) two-site DMRG (stages 5-6): ground-state energy
// of the open Heisenberg chain  H = J sum_i sigma_i . sigma_{i+1}  versus exact
// dense diagonalization, for N = 4, 6, 8. No truncation (exact bonds) -> must
// match to ~1e-8.
#include "dmrg_su2_run.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace qx_su2;

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

int main() {
    printf("SU(2) two-site DMRG vs dense ED (H = J sum sigma.sigma, OBC):\n");
    run(4);
    run(6);
    run(8);
    printf("Truncated (multiplet cap):\n");
    run_trunc(10, 20);
    run_trunc(12, 30);
    printf("\n%s\n", failures == 0 ? "ALL PASSED" : "SOME FAILED");
    return failures == 0 ? 0 : 1;
}
