// Step 4 spot-check (docs/gpu_acceleration_plan.md §7): confirm a dynamical
// observable inherits the GPU TEBD acceleration at large chi, and that the GPU path
// reproduces the CPU result. Runs the spin-conductivity current-current correlator
// C(t)=<J(t)J(0)> (the richest path: block DMRG ground state -> J|psi0> MPO apply ->
// real-time block TEBD evolution). GPU has TWO independent thresholds -- the SVD
// dispatch (QX_GPU_MIN_BLK) and the theta build (QX_GPU_THETA_MINWORK) -- so a TRUE
// CPU baseline must set BOTH high; setting only QX_GPU_MIN_BLK still routes large
// theta blocks to the GPU (a hybrid, not pure CPU).
//
// Usage: qx_obs_bench [n] [chi] [steps] [dt]   (defaults: 32 256 60 0.0125)
#include "dmrg_internal.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace qx_dmrg;

int main(int argc, char** argv) {
    int    n     = argc > 1 ? std::atoi(argv[1]) : 32;
    int    chi   = argc > 2 ? std::atoi(argv[2]) : 256;
    int    steps = argc > 3 ? std::atoi(argv[3]) : 60;
    double dt    = argc > 4 ? std::atof(argv[4]) : 0.0125;
    const double J = 1.0, delta = 1.0;

    const char* thr  = std::getenv("QX_GPU_MIN_BLK");
    const char* thrT = std::getenv("QX_GPU_THETA_MINWORK");
    auto mpo = build_xxz_mpo(n, J, delta);
    auto r = run_conductivity_blocks(mpo, n, dt, steps, chi, 1e-12, /*n_sweeps*/40, J, delta);

    // result fingerprint: C(0)=<J^2> and an L1 sum over the time series (GPU vs CPU
    // must match to FP tolerance even though there is no ED at this chi).
    double l1 = 0; for (size_t k = 0; k < r.C_re.size(); ++k) l1 += std::abs(r.C_re[k]) + std::abs(r.C_im[k]);
    // Report BOTH GPU thresholds so a run's config (true CPU vs hybrid) is unambiguous.
    std::printf("conductivity n=%-3d chi=%-5d steps=%-4d dt=%.4f | max_bond=%-5d E0=%.6f wall=%8.3f s | C(0)=%.6f  sum|C|=%.6f | QX_GPU_MIN_BLK=%s QX_GPU_THETA_MINWORK=%s\n",
                n, chi, steps, dt, r.max_bond, r.E0, r.wall_time_s,
                r.C_re.empty() ? 0.0 : r.C_re[0], l1,
                thr ? thr : "(unset->128)", thrT ? thrT : "(unset->2097152)");
    return 0;
}
