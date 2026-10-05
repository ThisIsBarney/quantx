// Correctness test for block-sparse (Sz-conserving) DMRG.
// Verifies run_dmrg_blocks reproduces the dense run_dmrg ground-state energy.

#include "dmrg_internal.hpp"
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <string>

using namespace qx_dmrg;

int main() {
    struct Case { int n; int max_bond; };
    const Case cases[] = {{2, 8}, {4, 64}, {6, 64}, {8, 64}};
    int failures = 0;

    for (auto& c : cases) {
        // One case per try, flushed: an exception escaping main ends in std::terminate ->
        // abort(), which the Windows CRT reports as 0xC0000409 -- the same code as a real
        // stack-buffer overrun -- with block-buffered stdout lost. That combination hid
        // which size failed, and why, and even what kind of failure it was.
        try {
            auto mpo = build_heisenberg_mpo(c.n, 1.0);
            auto dense  = run_dmrg(mpo, c.n, c.max_bond, 1e-10, 40, 1e-12);
            if (c.n < 3) {
                // The block engine refuses n < 3 by design: derive_mpo_sym needs a bulk
                // MPO site (qx_mpo_sym_test asserts the throw). This case used to ask it for
                // n=2 anyway, a leftover from before the universal-MPO rewrite. Check the
                // refusal is a clean exception, and that the dense answer is still right.
                bool threw = false;
                std::string msg;
                try { run_dmrg_blocks(mpo, c.n, c.max_bond, 1e-10, 40, 1e-12); }
                catch (const std::exception& e) { threw = true; msg = e.what(); }
                const bool ok = threw && msg.find("n>=3") != std::string::npos &&
                                std::abs(dense.energy - (-3.0)) < 1e-10;   // 2-site singlet
                if (!ok) ++failures;
                printf("n=%2d: dense E=%.8f  blocks refused n<3 (%s)  %s\n",
                       c.n, dense.energy, threw ? "clean exception" : "DID NOT THROW",
                       ok ? "OK" : "FAIL");
                fflush(stdout);
                continue;
            }
            auto blocks = run_dmrg_blocks(mpo, c.n, c.max_bond, 1e-10, 40, 1e-12);
            double diff = std::abs(dense.energy - blocks.energy);
            bool ok = diff < 1e-6;
            if (!ok) ++failures;
            printf("n=%2d: dense E=%.8f  blocks E=%.8f  diff=%.2e  %s\n",
                   c.n, dense.energy, blocks.energy, diff, ok ? "OK" : "FAIL");
        } catch (const std::exception& e) {
            ++failures;
            printf("n=%2d: THREW %s\n", c.n, e.what());
        }
        fflush(stdout);
    }

    printf("\n%s\n", failures == 0 ? "ALL PASSED" : "SOME FAILED");
    return failures == 0 ? 0 : 1;
}
