// Timing harness: dense run_dmrg vs Sz-block-sparse run_dmrg_blocks.
// Isolates the algorithm cost (no Python/tenpy in the loop) and finds the
// chi at which the block-sparse path crosses over the dense path.
//
// cutoff is set tiny so the SVD keeps up to `chi` states wherever the local
// Schmidt rank allows, exercising genuine O(chi^3) work at the chain centre.

#include "dmrg_internal.hpp"
#include <cstdio>
#include <cmath>

using namespace qx_dmrg;

int main(int argc, char** argv) {
    int n_sweeps = (argc > 1) ? std::atoi(argv[1]) : 5;
    const double cutoff = (argc > 3) ? std::atof(argv[3]) : 1e-10;
    const double tol = 0.0;        // run all sweeps (fixed work)

    int n = (argc > 2) ? std::atoi(argv[2]) : 20;
    const int chis[] = {16, 32, 64, 128, 256};

    printf("n=%d  n_sweeps=%d  cutoff=%.0e\n", n, n_sweeps, cutoff);
    printf("%4s %5s %14s %9s %14s %9s %10s\n",
           "n", "chi", "E_dense", "t_dense", "E_block", "t_block", "speedup");
    printf("--------------------------------------------------------------------------------\n");
    fflush(stdout);

    auto mpo = build_heisenberg_mpo(n, 1.0);
    for (int chi : chis) {
        auto d = run_dmrg(mpo, n, chi, cutoff, n_sweeps, tol);
        auto b = run_dmrg_blocks(mpo, n, chi, cutoff, n_sweeps, tol);
        double sp = (b.wall_time_s > 0) ? d.wall_time_s / b.wall_time_s : 0.0;
        printf("%4d %5d %14.6f %9.3f %14.6f %9.3f %9.2fx\n",
               n, chi, d.energy, d.wall_time_s, b.energy, b.wall_time_s, sp);
        fflush(stdout);
    }
    return 0;
}
