// T=0 dynamical structure factors of the fermionic Hubbard chain -- the raw data of
// SPIN-CHARGE SEPARATION, the hallmark of 1D interacting fermions. Emits the two
// correlators (ED-validated in qx_hubbard_test):
//   C_spin (j,l0;t) = <gs| (2Sz)_j  e^{-i(H-E0)t} (2Sz)_l0  |gs>,  2Sz  = diag(0,+1,-1,0)
//   C_charge(j,l0;t) = <gs| (n-1)_j e^{-i(H-E0)t} (n-1)_l0 |gs>,  n-1 = diag(-1,0,0,+1)
// on the half-filling ground state (packed two-U(1) fast path). A space+time FFT
// (benchmarks/plot_hubbard_skw.py) turns them into S_spin(k,w) / S_charge(k,w): the
// spin sector is GAPLESS (spinons, velocity v_s); the charge sector opens the MOTT GAP
// (holon-antiholon continuum above ~Delta_c) -- one electron, two very different
// collective excitations.
//
// Usage: qx_hubbard_spectra [N] [U] [dt] [n_steps] [chi] [cutoff]
// Writes docs/hubbard_spin_correlator.csv + docs/hubbard_charge_correlator.csv
// (CSV layout matches qx_thermal_skw_emit: header "t,l0=..,n_phys=..,U=.."; rows
// t, Re_0..Re_{N-1}, Im_0..Im_{N-1}).
#include "dmrg_internal.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace qx_dmrg;

static void emit(const char* path, const SpectralResult& r, int N, double U) {
    FILE* f = std::fopen(path, "w");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::fprintf(f, "t,l0=%d,n_phys=%d,U=%.4f\n", r.l0, N, U);
    for (size_t t = 0; t < r.times.size(); ++t) {
        std::fprintf(f, "%.4f", r.times[t]);
        for (int j = 0; j < N; ++j) std::fprintf(f, ",%.6e", r.C_re[t][j]);
        for (int j = 0; j < N; ++j) std::fprintf(f, ",%.6e", r.C_im[t][j]);
        std::fprintf(f, "\n");
    }
    std::fclose(f);
    std::printf("  wrote %s (%zu times x %d sites)\n", path, r.times.size(), N);
}

int main(int argc, char** argv) {
    int    N      = argc > 1 ? std::atoi(argv[1]) : 64;
    double U      = argc > 2 ? std::atof(argv[2]) : 4.0;
    double dt     = argc > 3 ? std::atof(argv[3]) : 0.05;
    int    nsteps = argc > 4 ? std::atoi(argv[4]) : 160;
    int    chi    = argc > 5 ? std::atoi(argv[5]) : 256;
    double cutoff = argc > 6 ? std::atof(argv[6]) : 1e-10;

    std::printf("Hubbard T=0 spectra: N=%d U=%.2f dt=%.3f %d steps (t->%.1f) chi<=%d "
                "cutoff=%.0e (packed two-U(1) fast path, half filling)\n",
                N, U, dt, nsteps, dt * nsteps, chi, cutoff);

    auto mpo = build_hubbard_mpo(N, 1.0, U, /*two_u1*/true);
    std::vector<int> cfg(N);
    for (int i = 0; i < N; ++i) cfg[i] = (i % 2 == 0) ? 1 : 2;      // SDW |up dn ...>
    const std::vector<double> w_spin = { 0.0, +1.0, -1.0, 0.0 };    // 2Sz
    const std::vector<double> w_chg  = {-1.0,  0.0,  0.0, +1.0 };   // n - 1 (half filling)

    auto sp = run_spectral_blocks(mpo, N, dt, nsteps, chi, cutoff, 80, 1.0, 1.0,
                                  pack_q2(N, 0), &cfg, &w_spin);
    std::printf("  spin  : E0=%.6f  l0=%d  max_bond=%d  C(0)=%.4f  %.1fs\n",
                sp.E0, sp.l0, sp.max_bond, sp.C_re[0][sp.l0], sp.wall_time_s);
    emit("docs/hubbard_spin_correlator.csv", sp, N, U);

    auto ch = run_spectral_blocks(mpo, N, dt, nsteps, chi, cutoff, 80, 1.0, 1.0,
                                  pack_q2(N, 0), &cfg, &w_chg);
    std::printf("  charge: E0=%.6f  l0=%d  max_bond=%d  C(0)=%.4f  %.1fs\n",
                ch.E0, ch.l0, ch.max_bond, ch.C_re[0][ch.l0], ch.wall_time_s);
    emit("docs/hubbard_charge_correlator.csv", ch, N, U);
    return 0;
}
