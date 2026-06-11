// Disentangler benchmark (F1 headline): the Karrasch-Bardarson-Moore
// backward-ancilla gauge leaves the finite-T correlator exactly invariant (a
// fact pinned to ~1e-13 by qx_thermal_spectral_test) but slows the growth of
// the purified state's bond dimension -- so at a fixed truncation it reaches
// longer times / lower temperatures, and at a fixed time it needs less chi.
//
// This driver cools a Heisenberg chain to beta and real-time evolves the
// finite-T structure-factor correlator with the disentangler OFF then ON, at a
// fixed cutoff and chi cap, and reports the max bond dimension per timestep and
// the wall time for both. The divergence of the two bond(t) curves IS the
// payoff: bond_on grows markedly slower than bond_off.
//
// Usage: qx_thermal_spectral_bench [N_phys] [beta] [dt] [n_steps] [chi] [cutoff] [n_cool]
// Writes docs/disentangler_bond_growth.csv (t, bond_off, bond_on) for plotting.
#include "dmrg_internal.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace qx_dmrg;

int main(int argc, char** argv) {
    int    N      = argc > 1 ? std::atoi(argv[1]) : 24;
    double beta   = argc > 2 ? std::atof(argv[2]) : 1.0;
    double dt     = argc > 3 ? std::atof(argv[3]) : 0.1;
    int    nsteps = argc > 4 ? std::atoi(argv[4]) : 30;
    int    chi    = argc > 5 ? std::atoi(argv[5]) : 1024;
    double cutoff = argc > 6 ? std::atof(argv[6]) : 1e-8;
    int    ncool  = argc > 7 ? std::atoi(argv[7]) : 40;

    std::printf("Disentangler benchmark: Heisenberg N=%d (2N=%d purified sites), "
                "beta=%.2f, dt=%.3f, %d steps (t->%.1f), chi<=%d, cutoff=%.0e, cool=%d\n",
                N, 2*N, beta, dt, nsteps, dt*nsteps, chi, cutoff, ncool);

    auto off = run_spectral_thermal_blocks(N, 1.0, 1.0, beta, ncool, dt, nsteps,
                                           chi, cutoff, /*disentangle*/false);
    auto on  = run_spectral_thermal_blocks(N, 1.0, 1.0, beta, ncool, dt, nsteps,
                                           chi, cutoff, /*disentangle*/true);

    // A = unsourced state e^{-iH t}|psi(b)> (its bond growth is what the
    // disentangler cancels outright); B = sourced state, carries the signal.
    // Cost of S(k,w,T) is set by max(bond_A, bond_B).
    std::printf("\n  %-6s | %-9s %-9s %-7s | %-9s %-9s %-7s | %-9s\n",
                "t", "A(off)", "A(on)", "A x", "B(off)", "B(on)", "B x", "max x");
    for (size_t k = 0; k < off.times.size(); ++k) {
        auto rat = [](int o, int n){ return n > 0 ? (double)o / n : 1.0; };
        std::printf("  %-6.2f | %-9d %-9d %-7.2f | %-9d %-9d %-7.2f | %-9.2f\n",
                    off.times[k],
                    off.bond_a_t[k], on.bond_a_t[k], rat(off.bond_a_t[k], on.bond_a_t[k]),
                    off.bond_b_t[k], on.bond_b_t[k], rat(off.bond_b_t[k], on.bond_b_t[k]),
                    rat(off.bond_t[k], on.bond_t[k]));
    }
    // Peak A-state reduction (the disentangler's direct effect: it cancels the
    // real-time entanglement growth of the unsourced state; the thermal-entropy
    // floor of |psi(beta)> itself is intrinsic and remains).
    double a_red = 1.0;
    for (size_t k = 0; k < off.times.size(); ++k)
        if (on.bond_a_t[k] > 0)
            a_red = std::max(a_red, (double)off.bond_a_t[k] / on.bond_a_t[k]);

    std::printf("\n  unsourced state A:  peak bond %dx smaller with disentangler "
                "(real-time growth cancelled)\n", (int)(a_red + 0.5));
    std::printf("  sourced state B:    operator-spreading entanglement is intrinsic"
                " -> peak (memory) ket-limited\n");
    std::printf("  max bond (memory):  off=%d  on=%d  (%.2fx)\n",
                off.max_bond, on.max_bond,
                on.max_bond > 0 ? (double)off.max_bond / on.max_bond : 1.0);
    std::printf("  wall time (compute):off=%.2fs  on=%.2fs  (%.2fx %s)\n",
                off.wall_time_s, on.wall_time_s,
                on.wall_time_s > 0 ? off.wall_time_s / on.wall_time_s : 1.0,
                on.wall_time_s < off.wall_time_s ? "faster" : "slower");
    std::printf("  E(beta)=%.5f  l0=%d  (C_{l0,l0}(0)=%.4f, should be 1)\n",
                off.E, off.l0, off.C_re[0][off.l0]);

    // CSV for plotting (mirrors benchmarks/plot_margin_vs_chi.py style tooling).
    if (FILE* f = std::fopen("docs/disentangler_bond_growth.csv", "w")) {
        std::fprintf(f, "t,A_off,A_on,B_off,B_on,max_off,max_on\n");
        for (size_t k = 0; k < off.times.size(); ++k)
            std::fprintf(f, "%.4f,%d,%d,%d,%d,%d,%d\n", off.times[k],
                         off.bond_a_t[k], on.bond_a_t[k],
                         off.bond_b_t[k], on.bond_b_t[k],
                         off.bond_t[k], on.bond_t[k]);
        std::fclose(f);
        std::printf("  wrote docs/disentangler_bond_growth.csv\n");
    }
    return 0;
}
