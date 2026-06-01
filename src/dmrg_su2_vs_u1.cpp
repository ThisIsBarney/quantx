// Head-to-head: SU(2) two-site DMRG vs the U(1) (Sz block-sparse) path on the
// same open Heisenberg chain (H = J sum sigma_i.sigma_{i+1}, J=1). Reports
// ground energy, kept bond size, and wall time, so the SU(2) bond-dimension
// reduction can be read against actual accuracy/time.
#include "dmrg_internal.hpp"      // U(1) path: build_heisenberg_mpo, run_dmrg_blocks
#include "dmrg_su2_run.hpp"       // SU(2) path
#include <cstdio>
#include <chrono>

using clk = std::chrono::steady_clock;

int main() {
    const int N = 32;
    printf("Heisenberg N=%d, H = J sum sigma.sigma (J=1), %d sweeps\n", N, 6);
    printf("%-8s %-6s %14s %10s\n", "path", "param", "E", "time(ms)");
    printf("----------------------------------------------------\n");

    // U(1) block-sparse path at several chi.
    auto mpo = qx_dmrg::build_heisenberg_mpo(N, 1.0);
    for (int chi : {32, 64, 128}) {
        auto t0 = clk::now();
        auto r = qx_dmrg::run_dmrg_blocks(mpo, N, chi, 1e-10, 6, 0.0);
        double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        printf("%-8s chi=%-4d %14.6f %10.1f\n", "U(1)", chi, r.energy, ms);
        fflush(stdout);
    }
    // SU(2) path at several multiplet caps.
    for (int mm : {20, 40, 80}) {
        auto t0 = clk::now();
        double e = qx_su2::run_su2_dmrg(N, 1.0, mm, 6, 7);
        double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        printf("%-8s mult=%-3d %14.6f %10.1f\n", "SU(2)", mm, e, ms);
        fflush(stdout);
    }
    return 0;
}
