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
std::vector<MPOSite> build_heisenberg_mpo(int n, double J = 1.0);
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

} // namespace qx_dmrg
