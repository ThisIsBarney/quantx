#pragma once
// Internal header for DMRG implementation -- not part of the public API.
// Include this from dmrg_mpo.cpp, dmrg_dense.cpp, dmrg_blocks.cpp, dmrg_capi.cpp.

#include <complex>
#include <map>
#include <string>
#include <vector>
#include <array>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

namespace qx_dmrg {

using Cx   = std::complex<double>;
using Mat  = Eigen::MatrixXcd;
using Vec  = Eigen::VectorXcd;
using RVec = Eigen::VectorXd;
using Mat2 = Eigen::Matrix2cd;
using nlj  = nlohmann::json;

// -- Shared data structures ---------------------------------------------------

// MPO site tensor: ops[alpha][beta] is a 2x2 matrix W^{alpha,beta}_{s,s'}
struct MPOSite {
    int chi_l, chi_r;
    std::vector<Mat2> ops;
    const Mat2& W(int a, int b) const { return ops[a * chi_r + b]; }
    Mat2& W(int a, int b)             { return ops[a * chi_r + b]; }
};

struct MPS {
    int n;
    std::vector<Mat> A;    // A[i]: (chi_L*2, chi_R), left-canonical
    std::vector<RVec> sv;  // sv[i]: singular values at bond i
};

struct DMRGResult {
    double energy = 0;
    double truncation_error = 0;
    std::vector<double> sweep_energies;
    int n_sweeps_done = 0;
    double wall_time_s = 0;
};

// -- Function declarations (definitions in respective .cpp files) -------------

// dmrg_mpo.cpp
Mat2 pauli(char c);
// XXZ chain: H = J sum_i [ (X_i X_{i+1} + Y_i Y_{i+1}) + delta Z_i Z_{i+1} ]
// (delta=1 is the isotropic Heisenberg point). Same Sz bond structure as the
// plain Heisenberg MPO, so the block-sparse engine handles it unchanged.
std::vector<MPOSite> build_xxz_mpo(int n, double J = 1.0, double delta = 1.0);
inline std::vector<MPOSite> build_heisenberg_mpo(int n, double J = 1.0) {
    return build_xxz_mpo(n, J, 1.0);
}
std::vector<MPOSite> build_pauli_sum_mpo(int n,
    const std::vector<std::pair<double, std::string>>& terms);

// dmrg_dense.cpp
MPS  init_zero_state(int n);
double compute_energy(const MPS& mps, const std::vector<MPOSite>& mpo);
DMRGResult run_dmrg(const std::vector<MPOSite>& mpo, int n,
                    int max_bond, double cutoff, int n_sweeps, double tol);

// dmrg_blocks.cpp
DMRGResult run_dmrg_blocks(const std::vector<MPOSite>& mpo, int n,
                           int max_bond, double cutoff, int n_sweeps, double tol);

// Block-sparse (Sz-symmetric) TDVP2 time evolution from the Neel state.
struct TDVPBlockResult {
    double energy = 0;
    std::vector<double> times, stag;   // staggered magnetization time series
    int max_bond = 0;
    double wall_time_s = 0;
};
TDVPBlockResult run_tdvp2_blocks(const std::vector<MPOSite>& mpo, int n,
                                 double dt, int n_steps, int chi_max,
                                 double cutoff, bool imag);

// Block-sparse TEBD2 (Trotter-gate) time evolution from the Neel state -- the
// fast nearest-neighbour route (no environments / Krylov). measure_stride: record
// staggered magnetization every this many steps (<=0 = endpoints only).
// J and delta parameterise the XXZ bond gate exp(tau*h), h = J(XX+YY)+J*delta*ZZ;
// they must match the mpo passed in (whose energy is measured at the end).
TDVPBlockResult run_tebd2_blocks(const std::vector<MPOSite>& mpo, int n,
                                 double dt, int n_steps, int chi_max,
                                 double cutoff, bool imag, int measure_stride = 1,
                                 double J = 1.0, double delta = 1.0);

// Dynamical spin structure factor data: time-dependent correlations
// C_{j,l0}(t) = <psi0| sigma^z_j e^{-i(H-E0)t} sigma^z_{l0} |psi0> from a central
// source site l0, ready for a space+time Fourier transform into S(k,omega).
struct SpectralResult {
    std::vector<double> times;                  // length n_steps+1
    int l0 = 0, n = 0, max_bond = 0;
    double E0 = 0, wall_time_s = 0;
    std::vector<std::vector<double>> C_re, C_im; // C_re[t][j], C_im[t][j]
};
SpectralResult run_spectral_blocks(const std::vector<MPOSite>& mpo, int n,
                                   double dt, int n_steps, int chi_max,
                                   double cutoff, int n_sweeps,
                                   double J = 1.0, double delta = 1.0);

} // namespace qx_dmrg
