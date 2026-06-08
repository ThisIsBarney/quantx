// Large-block SVD-backend regression test.
//
// The ED suite (qx_tdvp_blocks_test, n=6) only exercises tiny Sz blocks (<96), so it
// never hits the large-block code paths -- the LAPACK xsyevd/xheevd eigensolve
// (>= QX_LAPACKE_MIN) or the GPU cuSOLVER SVD (>= QX_GPU_MIN_BLK). That gap let a
// wrong-results OpenBLAS LAPACKE path ship: at chi=512 it corrupted the DMRG ground
// state (E0 = -3395 / -9217 -- two different garbage values, i.e. an intermittent
// race, not a deterministic size bug, so it does not reproduce on every run). This
// test closes the *class* of gap: it runs the block-sparse DMRG at a chi large enough
// to create >=96 blocks and asserts the energy, so any large-block backend that
// diverges (deterministically, or caught by chance for an intermittent one) fails
// here -- on the shipping Eigen path and on any opted-in LAPACK/GPU backend.
//
// Run it under different backends to validate each:
//   (default)                  -> Eigen Gram eigensolve (reference)
//   QX_LAPACKE_MIN=96          -> LAPACK xsyevd path (catches a buggy OpenBLAS LAPACKE)
//   QX_GPU_MIN_BLK=1           -> GPU path (real DMRG is FP64-Eigen; see note)
//
// Note: run_dmrg_blocks for a real Hamiltonian uses BlockDMRG<double>; the GPU SVD
// path is complex-only, so the GPU backend is exercised by the complex evolution
// (qx_tdvp_blocks_test with QX_GPU_MIN_BLK=1), while THIS test targets the CPU
// large-block eigensolve (Eigen vs LAPACK) that the DMRG ground state uses.
//
// Usage: qx_large_block_test [n] [chi] [expected_E0] [tol]
#include "dmrg_internal.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace qx_dmrg;

int main(int argc, char** argv) {
    int    n        = argc > 1 ? std::atoi(argv[1]) : 32;
    int    chi      = argc > 2 ? std::atoi(argv[2]) : 256;
    bool   have_ref = argc > 3;
    double E0_ref   = have_ref ? std::atof(argv[3]) : 0.0;
    double tol      = argc > 4 ? std::atof(argv[4]) : 1e-3;

    const char* lpk = std::getenv("QX_LAPACKE_MIN");
    const char* gmb = std::getenv("QX_GPU_MIN_BLK");

    auto mpo = build_heisenberg_mpo(n, 1.0);   // XXZ delta=1 (Pauli convention)
    auto r = run_dmrg_blocks(mpo, n, /*max_bond*/chi, /*cutoff*/1e-10,
                             /*n_sweeps*/40, /*tol*/1e-10);

    // At n>=24, chi>=256 the central-Sz blocks reach ~0.5*chi (>=96), so the LAPACK /
    // GPU large-block paths are exercised (verified via QX_TEBD_PROFILE maxblk).
    std::printf("DMRG n=%d chi=%d : E0=%.6f  | QX_LAPACKE_MIN=%s QX_GPU_MIN_BLK=%s\n",
                n, chi, r.energy, lpk ? lpk : "(off)", gmb ? gmb : "(unset)");

    if (!have_ref) {
        std::printf("  (no reference E0 given; pass it as argv[3] to assert)\n");
        return 0;
    }
    double err = std::abs(r.energy - E0_ref);
    bool ok = err < tol;
    std::printf("  E0 vs reference %.6f : err=%.3e tol=%.1e  %s\n", E0_ref, err, tol, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
