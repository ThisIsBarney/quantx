// Phase A end-to-end benchmark (docs/gpu_acceleration_plan.md, benchmark step
// "margin vs chi"): real-time Néel quench TEBD2 on the Heisenberg chain, timed at a
// given bond dimension chi. The GPU has TWO thresholds -- SVD (QX_GPU_MIN_BLK) and
// theta build (QX_GPU_THETA_MINWORK) -- so a TRUE CPU run must set BOTH high; setting
// only QX_GPU_MIN_BLK leaves the large theta blocks on the GPU (hybrid, not pure CPU).
// Pair with QX_TEBD_PROFILE=1 to see the SVD share / ngpu count.
//
// Usage: qx_tebd_bench [n] [chi] [steps] [dt] [d]   (defaults: 32 chi 60 0.05 2)
//   d=2 -> spin-1/2 Heisenberg (Pauli sigma.sigma); d=3 -> spin-1 Heisenberg (S.S).
#include "dmrg_internal.hpp"
#include <cstdio>
#include <cstdlib>

using namespace qx_dmrg;

int main(int argc, char** argv) {
    int    n     = argc > 1 ? std::atoi(argv[1]) : 32;
    int    chi   = argc > 2 ? std::atoi(argv[2]) : 60;
    int    steps = argc > 3 ? std::atoi(argv[3]) : 60;
    double dt    = argc > 4 ? std::atof(argv[4]) : 0.05;
    int    d     = argc > 5 ? std::atoi(argv[5]) : 2;

    const char* thr  = std::getenv("QX_GPU_MIN_BLK");
    const char* thrT = std::getenv("QX_GPU_THETA_MINWORK");
    auto mpo = (d == 3) ? build_spin1_heisenberg_mpo(n, 1.0) : build_heisenberg_mpo(n, 1.0);
    // real-time Néel quench (imag=false); endpoints only (stride=0); J=1, delta=1.
    auto r = run_tebd2_blocks(mpo, n, dt, steps, chi, 1e-12, /*imag*/false, /*stride*/0, 1.0, 1.0);

    std::printf("d=%d n=%-3d chi_max=%-5d steps=%-4d dt=%.4f | max_bond=%-5d wall=%8.3f s | %.4f s/step | E=%.9f | QX_GPU_MIN_BLK=%s QX_GPU_THETA_MINWORK=%s\n",
                d, n, chi, steps, dt, r.max_bond, r.wall_time_s, r.wall_time_s/steps, r.energy,
                thr ? thr : "(unset->128)", thrT ? thrT : "(unset->2097152)");
    return 0;
}
