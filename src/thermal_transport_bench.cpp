// Finite-T transport benchmark (F2): the spin current-current correlator
// C(t;T) = <J(t)J(0)>_beta on the purified chain, the integrand of the optical
// conductivity sigma(omega,T) = Re integral e^{i omega t} C(t) dt. Its t=0 value
// is <J^2>_beta; its long-time behaviour distinguishes ballistic (a nonzero
// Drude weight, Re C -> const) from diffusive transport -- the live question for
// the integrable Heisenberg chain. The FFT to sigma(omega,T) is a downstream
// post-process of the emitted C(t).
//
// The disentangler (default on) is the F1-validated compute path; its bond-growth
// win on the unsourced state carries over unchanged (the extensive current makes
// the sourced ket dominate peak memory, exactly as for the structure factor).
//
// This driver sweeps a few temperatures and reports C(t;T) + a Drude-weight proxy
// (long-time average of Re C). Usage:
//   qx_thermal_transport_bench [N_phys] [dt] [n_steps] [chi] [cutoff] [n_cool] [beta...]
// Writes docs/transport_correlator.csv (t, then ReC/ImC per beta).
#include "dmrg_internal.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace qx_dmrg;

int main(int argc, char** argv) {
    int    N      = argc > 1 ? std::atoi(argv[1]) : 24;
    double dt     = argc > 2 ? std::atof(argv[2]) : 0.1;
    int    nsteps = argc > 3 ? std::atoi(argv[3]) : 30;
    int    chi    = argc > 4 ? std::atoi(argv[4]) : 1024;
    double cutoff = argc > 5 ? std::atof(argv[5]) : 1e-8;
    int    ncool  = argc > 6 ? std::atoi(argv[6]) : 40;
    std::vector<double> betas;
    for (int a = 7; a < argc; ++a) betas.push_back(std::atof(argv[a]));
    if (betas.empty()) betas = {0.5, 1.0, 2.0};
    const int d = std::getenv("QX_BENCH_D") ? std::atoi(std::getenv("QX_BENCH_D")) : 2;
    // QX_BENCH_DELTA: XXZ anisotropy for d=2/3; the on-site U for d=4 ((J,delta)->(t,U)).
    const double bdelta = std::getenv("QX_BENCH_DELTA") ? std::atof(std::getenv("QX_BENCH_DELTA")) : 1.0;
    // QX_BENCH_MU (d=4): chemical potential; U/2 pins half filling at every T.
    const double bmu = std::getenv("QX_BENCH_MU") ? std::atof(std::getenv("QX_BENCH_MU")) : 0.0;

    std::printf("Transport benchmark: %s N=%d (2N=%d purified sites), "
                "dt=%.3f, %d steps (t->%.1f), chi<=%d, cutoff=%.0e, cool=%d%s\n",
                d==4?"Hubbard":d==3?"spin-1":"spin-1/2", N, 2*N, dt, nsteps, dt*nsteps,
                chi, cutoff, ncool, d==4?" (delta=U)":"");

    // disentangle OFF: the correlator is disentangler-invariant (~1e-13), so the
    // cheaper single-sweep path emits identical C(t;T) at ~2x the speed -- the
    // disentangler's value (a compute win on the unsourced bra) is its own
    // benchmark (F1, qx_thermal_spectral_bench), not re-litigated here.
    std::vector<ConductivityThermalResult> res;
    for (double beta : betas)
        res.push_back(run_conductivity_thermal_blocks(N, 1.0, bdelta, beta, ncool, dt,
                                                      nsteps, chi, cutoff,
                                                      /*disentangle*/false, d, bmu));

    // per-temperature summary: <J^2>, Drude proxy, peak bond, wall time
    std::printf("\n  %-6s %-12s %-12s %-9s %-9s\n",
                "beta", "C(0)=<J^2>", "Drude~", "maxbond", "wall(s)");
    for (size_t b = 0; b < betas.size(); ++b) {
        const auto& r = res[b];
        double drude = 0; int dn = 0;
        for (size_t k = r.times.size() / 2; k < r.times.size(); ++k) { drude += r.C_re[k]; ++dn; }
        std::printf("  %-6.2f %-12.4f %-12.4f %-9d %-9.2f\n",
                    betas[b], r.C_re[0], dn ? drude / dn : 0.0, r.max_bond, r.wall_time_s);
    }

    // Re C(t;T) table across temperatures (the sigma(omega,T) integrand).
    std::printf("\n  %-6s", "t");
    for (double beta : betas) std::printf(" | ReC(b=%.1f)", beta);
    std::printf("\n");
    for (size_t k = 0; k < res[0].times.size(); ++k) {
        std::printf("  %-6.2f", res[0].times[k]);
        for (auto& r : res) std::printf(" | %-10.5f", r.C_re[k]);
        std::printf("\n");
    }

    // QX_BENCH_OUT overrides the output path (so d=3/d=4 runs don't clobber the
    // d=2 artifact -- the trap the spin-1 run hit once).
    const char* outp = std::getenv("QX_BENCH_OUT");
    if (!outp) outp = "docs/transport_correlator.csv";
    if (FILE* f = std::fopen(outp, "w")) {
        std::fprintf(f, "t");
        for (double beta : betas) std::fprintf(f, ",ReC_b%.2f,ImC_b%.2f", beta, beta);
        std::fprintf(f, "\n");
        for (size_t k = 0; k < res[0].times.size(); ++k) {
            std::fprintf(f, "%.4f", res[0].times[k]);
            for (auto& r : res) std::fprintf(f, ",%.6f,%.6f", r.C_re[k], r.C_im[k]);
            std::fprintf(f, "\n");
        }
        std::fclose(f);
        std::printf("\n  wrote %s\n", outp);
    }
    return 0;
}
