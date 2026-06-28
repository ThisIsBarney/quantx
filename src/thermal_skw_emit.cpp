// Emit the finite-T dynamical spin correlator C_{j,l0}(t;T) for a space+time FFT
// into the structure factor S(k,omega,T) (downstream Python in
// benchmarks/plot_skw.py). Runs run_spectral_thermal_blocks at one temperature
// and writes the full C[t][j] matrix (real + imag) plus the source site l0.
//
// Usage: qx_thermal_skw_emit [N_phys] [beta] [dt] [n_steps] [chi] [cutoff] [n_cool] [out.csv] [disentangle] [d]
// CSV layout: header "t,l0=<l0>,n_phys=<N>"; then per row: t, Re_0..Re_{N-1}, Im_0..Im_{N-1}.
// disentangle defaults OFF: the correlator is disentangler-invariant (~1e-13), so
// the cheaper single-sweep path emits identical figure data at ~2x the speed.
// d=2 -> spin-1/2 XXZ Heisenberg; d=3 -> spin-1 Heisenberg (Haldane, gapped S(k,w)).
#include "dmrg_internal.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace qx_dmrg;

int main(int argc, char** argv) {
    int    N      = argc > 1 ? std::atoi(argv[1]) : 24;
    double beta   = argc > 2 ? std::atof(argv[2]) : 1.0;
    double dt     = argc > 3 ? std::atof(argv[3]) : 0.1;
    int    nsteps = argc > 4 ? std::atoi(argv[4]) : 60;
    int    chi    = argc > 5 ? std::atoi(argv[5]) : 768;
    double cutoff = argc > 6 ? std::atof(argv[6]) : 1e-8;
    int    ncool  = argc > 7 ? std::atoi(argv[7]) : 40;
    std::string out = argc > 8 ? argv[8] : "docs/skw_correlator.csv";
    bool   diseng = argc > 9 ? std::atoi(argv[9]) != 0 : false;
    int    d      = argc > 10 ? std::atoi(argv[10]) : 2;

    std::printf("S(k,w,T) emit: %s N=%d beta=%.2f dt=%.3f %d steps (t->%.1f) "
                "chi<=%d cutoff=%.0e cool=%d\n", d==3?"spin-1":"spin-1/2",
                N, beta, dt, nsteps, dt*nsteps, chi, cutoff, ncool);

    auto r = run_spectral_thermal_blocks(N, 1.0, 1.0, beta, ncool, dt, nsteps,
                                         chi, cutoff, diseng, d);
    std::printf("  E(beta)=%.5f  l0=%d  max_bond=%d  C_{l0,l0}(0)=%.4f  %.1fs\n",
                r.E, r.l0, r.max_bond, r.C_re[0][r.l0], r.wall_time_s);

    FILE* f = std::fopen(out.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", out.c_str()); return 1; }
    std::fprintf(f, "t,l0=%d,n_phys=%d,beta=%.4f\n", r.l0, N, beta);
    for (size_t t = 0; t < r.times.size(); ++t) {
        std::fprintf(f, "%.4f", r.times[t]);
        for (int j = 0; j < N; ++j) std::fprintf(f, ",%.6e", r.C_re[t][j]);
        for (int j = 0; j < N; ++j) std::fprintf(f, ",%.6e", r.C_im[t][j]);
        std::fprintf(f, "\n");
    }
    std::fclose(f);
    std::printf("  wrote %s (%zu times x %d sites)\n", out.c_str(), r.times.size(), N);
    return 0;
}
