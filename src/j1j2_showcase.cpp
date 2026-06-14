// J1-J2 frustrated Heisenberg chain on the fast block-sparse path -- a showcase
// of the universal-MPO capability (the engine was hard-wired to nearest-neighbour
// XXZ; build_j1j2_mpo is an 8-wide MPO now consumed by run_dmrg_blocks unchanged,
// matching ED to ~1e-13 -- see qx_mpo_sym_test).
//
// H = J1 sum_i S_i.S_{i+1} + J2 sum_i S_i.S_{i+2}  (isotropic, Pauli convention:
// each S.S term is XX+YY+ZZ). Frustration grows the dimerization: near the
// Majumdar-Ghosh point J2 = J1/2 the ground state approaches a PRODUCT of
// nearest-neighbour singlets (bond dimension 2), so its entanglement collapses
// and a tiny-chi DMRG nearly reaches the converged energy. The gap between a
// low-chi probe and a converged run, plotted vs J2/J1, therefore DIPS toward
// J2/J1 = 0.5 -- the dimer-product signature, from energies alone, on the new
// block path. (Exactly at the MG point the GS is two-fold degenerate, a known
// DMRG-convergence corner case; the approach to it carries the physics.)
//
// Usage: qx_j1j2_showcase [N] [chi_conv] [chi_probe]
// Writes docs/j1j2_dimerization.csv  (j2, E/N converged, low-chi gap/site).
#include "dmrg_internal.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace qx_dmrg;

int main(int argc, char** argv) {
    int N        = argc > 1 ? std::atoi(argv[1]) : 32;
    int chi_conv = argc > 2 ? std::atoi(argv[2]) : 128;
    int chi_prb  = argc > 3 ? std::atoi(argv[3]) : 8;
    const double J1 = 1.0, d = 1.0;

    std::printf("J1-J2 frustrated chain (N=%d, isotropic): ground state on the fast\n"
                "block path, converged chi=%d vs low-chi probe chi=%d.\n",
                N, chi_conv, chi_prb);
    std::printf("\n  %-7s %-13s %-13s %-12s %-8s\n",
                "J2/J1", "E/N (chi=hi)", "E/N (chi=lo)", "gap/site", "wall(s)");

    // J2/J1 grid (skip exactly 0.5: the MG point's two-fold GS degeneracy is a
    // DMRG corner case -- the dimerization shows in the approach to it).
    const std::vector<double> j2s =
        {0.0, 0.10, 0.20, 0.30, 0.40, 0.45, 0.49, 0.51, 0.55, 0.60};

    FILE* f = std::fopen("docs/j1j2_dimerization.csv", "w");
    if (f) std::fprintf(f, "j2,E_per_site,gap_per_site\n");
    for (double j2 : j2s) {
        auto mpo = build_j1j2_mpo(N, J1, j2, d);
        auto t0 = std::chrono::steady_clock::now();
        auto hi = run_dmrg_blocks(mpo, N, chi_conv, 1e-11, 80, 1e-11);
        double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        auto lo = run_dmrg_blocks(mpo, N, chi_prb, 1e-11, 80, 1e-11);
        double eh = hi.energy / N, el = lo.energy / N, gap = el - eh;
        std::printf("  %-7.3f %-13.6f %-13.6f %-12.2e %-8.2f\n", j2, eh, el, gap, wall);
        if (f) std::fprintf(f, "%.4f,%.8f,%.8e\n", j2, eh, gap);
    }
    if (f) { std::fclose(f); std::printf("\n  wrote docs/j1j2_dimerization.csv\n"); }

    // Exact Majumdar-Ghosh check: at J2 = J1/2 the GS is a dimer product (chi=2),
    // so even a chi=8 DMRG reaches the exact energy E/N = -1.5 (Pauli units).
    auto mg = run_dmrg_blocks(build_j1j2_mpo(N, J1, 0.5, d), N, /*chi*/8, 1e-12, 80, 1e-12);
    std::printf("\n  Majumdar-Ghosh point J2=0.5 : E/N(chi=8) = %.6f  (exact dimer = -1.500000)\n",
                mg.energy / N);
    std::printf("  -> the gap/site collapses toward J2/J1=0.5: the GS dimerizes to a\n"
                "     bond-dimension-2 product, reachable at tiny chi.\n");
    return 0;
}
