// Correctness test for block-sparse (Sz-conserving) DMRG.
// Verifies run_dmrg_blocks reproduces the dense run_dmrg ground-state energy.

#include "dmrg_internal.hpp"
#include <cstdio>
#include <cmath>

using namespace qx_dmrg;

int main() {
    struct Case { int n; int max_bond; };
    const Case cases[] = {{2, 8}, {4, 64}, {6, 64}, {8, 64}};
    int failures = 0;

    for (auto& c : cases) {
        auto mpo = build_heisenberg_mpo(c.n, 1.0);
        auto dense  = run_dmrg(mpo, c.n, c.max_bond, 1e-10, 40, 1e-12);
        auto blocks = run_dmrg_blocks(mpo, c.n, c.max_bond, 1e-10, 40, 1e-12);
        double diff = std::abs(dense.energy - blocks.energy);
        bool ok = diff < 1e-6;
        if (!ok) ++failures;
        printf("n=%2d: dense E=%.8f  blocks E=%.8f  diff=%.2e  %s\n",
               c.n, dense.energy, blocks.energy, diff, ok ? "OK" : "FAIL");
    }

    printf("\n%s\n", failures == 0 ? "ALL PASSED" : "SOME FAILED");
    return failures == 0 ? 0 : 1;
}
