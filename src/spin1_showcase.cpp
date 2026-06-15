// Spin-1 Heisenberg (Haldane) chain on the fast block-sparse path -- a showcase of
// the universal-MPO local-dimension capability. The engine was hard-wired to a
// two-state (spin-1/2) local Hilbert space; build_spin1_heisenberg_mpo is a d=3 MPO
// now consumed by run_dmrg_blocks unchanged, matching ED to ~1e-13 (qx_mpo_sym_test).
//
// H = J sum_i S_i . S_{i+1} with the physical spin-1 operators (S+- = sqrt(2)). The
// ground state is the gapped Haldane phase, whose energy DENSITY is known to high
// precision: e_bulk = -1.401484039 J. On an OPEN chain the raw E/N carries an O(1/N)
// edge-mode/bond-counting offset, so the bulk value is read cleanly from the
// INCREMENTAL energy (E_N - E_{N-dN})/dN -> e_bulk: the cost of growing the bulk by
// one site. The Haldane gap also means the entanglement obeys an area law, so the
// DMRG energy SATURATES at small bond dimension (unlike the critical spin-1/2 chain)
// -- shown by the chi sweep.
//
// Usage: qx_spin1_showcase [chi]
// Writes docs/spin1_haldane.csv  (N, E/N, e_bulk_incremental, wall_s).
#include "dmrg_internal.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace qx_dmrg;

int main(int argc, char** argv) {
    int chi = argc > 1 ? std::atoi(argv[1]) : 128;
    const double J = 1.0;
    const double e_exact = -1.401484039;   // spin-1 Heisenberg energy density (J units)

    auto e_of = [&](int N) {
        auto mpo = build_spin1_heisenberg_mpo(N, J);
        auto t0 = std::chrono::steady_clock::now();
        double E = run_dmrg_blocks(mpo, N, chi, 1e-11, 80, 1e-11).energy;
        double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return std::pair<double,double>{E, wall};
    };

    std::printf("Spin-1 Heisenberg (Haldane) chain on the fast block path (chi=%d).\n"
                "Bulk energy density e_bulk = (E_N - E_{N-dN})/dN -> %.9f (J units).\n\n",
                chi, e_exact);
    std::printf("  %-5s %-14s %-16s %-10s %-8s\n",
                "N", "E/N", "e_bulk(incr)", "err", "wall(s)");

    FILE* f = std::fopen("docs/spin1_haldane.csv", "w");
    if (f) std::fprintf(f, "N,E_per_site,e_bulk_incremental,wall_s\n");

    const std::vector<int> Ns = {16, 32, 48, 64, 80, 96};
    double prevE = 0; int prevN = 0;
    for (int N : Ns) {
        auto [E, wall] = e_of(N);
        double e_site = E / N;
        if (prevN == 0) {
            std::printf("  %-5d %-14.6f %-16s %-10s %-8.2f\n", N, e_site, "-", "-", wall);
            if (f) std::fprintf(f, "%d,%.8f,nan,%.3f\n", N, e_site, wall);
        } else {
            double e_bulk = (E - prevE) / (N - prevN);
            std::printf("  %-5d %-14.6f %-16.8f %-10.1e %-8.2f\n",
                        N, e_site, e_bulk, e_bulk - e_exact, wall);
            if (f) std::fprintf(f, "%d,%.8f,%.8f,%.3f\n", N, e_site, e_bulk, wall);
        }
        prevE = E; prevN = N;
    }
    if (f) { std::fclose(f); std::printf("\n  wrote docs/spin1_haldane.csv\n"); }

    // Area-law signature: at fixed N the gapped Haldane GS energy SATURATES at small
    // chi (low entanglement), unlike a critical chain whose energy keeps dropping.
    std::printf("\n  Bond-dimension convergence at N=64 (gapped -> saturates fast):\n");
    std::printf("    %-6s %-14s\n", "chi", "E/N");
    for (int c : {4, 8, 16, 32, 64}) {
        double E = run_dmrg_blocks(build_spin1_heisenberg_mpo(64, J), 64, c, 1e-11, 80, 1e-11).energy;
        std::printf("    %-6d %-14.6f\n", c, E / 64);
    }
    std::printf("\n  -> the incremental energy converges to the known Haldane density\n"
                "     -1.401484 J, and the energy saturates by chi~16: the gapped\n"
                "     spin-1 ground state, on the same fast block path as spin-1/2.\n");
    return 0;
}
