// Finite-T dynamics: GPU vs CPU SVD-throughput benchmark.
//
// The same block-sparse SVD hot loop that the T=0 dynamics uses (and where the
// GPU advantage grows with chi -- see docs/gpu_margin_vs_chi.svg) fires unchanged
// on the purified chain. Real-time evolution of the finite-T correlator saturates
// the bond dimension to chi_max within a few steps, so pushing chi_max puts the
// per-Sz-block SVDs into the large regime where cuSOLVER beats the CPU -- the
// "advantage grows with chi" property, now for finite-T dynamics. (Cooling alone
// is temperature-limited in bond dimension; the real-time response is what
// reaches the ceiling, so this benchmark times the evolution.)
//
// run_spectral_thermal_blocks times the real-time evolution only (cooling happens
// before its internal clock starts). The GPU path is selected per-block by
// QX_GPU_MIN_BLK: a small value routes large blocks to cuSOLVER, a huge value
// forces the CPU Eigen/LAPACK path. Run the same command under both envs at each
// chi to get the speedup:
//
//   QX_GPU_MIN_BLK=128     qx_thermal_gpu_bench 20 1024    # GPU
//   QX_GPU_MIN_BLK=9999999 qx_thermal_gpu_bench 20 1024    # CPU
//
// Usage: qx_thermal_gpu_bench [N_phys] [chi] [n_steps] [beta] [n_cool] [dt] [cutoff]
#include "dmrg_internal.hpp"
#include <cstdio>
#include <cstdlib>

using namespace qx_dmrg;

int main(int argc, char** argv) {
    int    N      = argc > 1 ? std::atoi(argv[1]) : 20;
    int    chi    = argc > 2 ? std::atoi(argv[2]) : 1024;
    int    nsteps = argc > 3 ? std::atoi(argv[3]) : 16;
    double beta   = argc > 4 ? std::atof(argv[4]) : 1.0;
    int    ncool  = argc > 5 ? std::atoi(argv[5]) : 20;
    double dt     = argc > 6 ? std::atof(argv[6]) : 0.1;
    double cutoff = argc > 7 ? std::atof(argv[7]) : 1e-9;

    const char* gmb = std::getenv("QX_GPU_MIN_BLK");
    std::printf("finite-T dynamics: N=%d (2N=%d sites) chi=%d, beta=%.1f, %d rt-steps, "
                "QX_GPU_MIN_BLK=%s\n", N, 2*N, chi, beta, nsteps,
                gmb ? gmb : "(default 128)");

    // disentangle=false: the correlator is gauge-invariant, so the single-sweep
    // path is the right throughput baseline (the evolution still saturates chi).
    auto r = run_spectral_thermal_blocks(N, /*J*/1.0, /*delta*/1.0, beta, ncool,
                                         dt, nsteps, chi, cutoff,
                                         /*disentangle*/false);

    std::printf("  max_bond=%d  E(beta)=%.5f  evolve_wall=%.3fs  (%.1f ms/rt-step)\n",
                r.max_bond, r.E, r.wall_time_s, 1000.0 * r.wall_time_s / nsteps);
    return 0;
}
