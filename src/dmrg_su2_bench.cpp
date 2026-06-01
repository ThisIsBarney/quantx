// Timing harness for the SU(2) two-site DMRG: wall time per chain length /
// multiplet cap, plus the final energy and energy density.
#include "dmrg_su2_run.hpp"
#include <cstdio>
#include <chrono>

using namespace qx_su2;
using clk = std::chrono::steady_clock;

static void bench(int N, int max_mult, int sweeps) {
    auto t0 = clk::now();
    double e = run_su2_dmrg(N, 1.0, max_mult, sweeps, 7);
    auto t1 = clk::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  N=%-4d cap=%-4d sweeps=%-2d   E=%+.8f  E/N=%+.6f   %8.1f ms\n",
           N, max_mult, sweeps, e, e / N, ms);
}

int main() {
    printf("SU(2) two-site DMRG timing (H = J sum sigma.sigma, OBC):\n");
    bench(16,  0, 6);
    bench(24, 40, 6);
    bench(32, 60, 6);
    bench(48, 80, 6);
    bench(64, 100, 6);
    return 0;
}
