#include "dmrg_internal.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace qx_dmrg {

// -- MPS initialization -------------------------------------------------------

// Initialize MPS to |0...0> (all qubits in |0> state), bond dim = 1
MPS init_zero_state(int n) {
    MPS mps;
    mps.n = n;
    mps.A.resize(n);
    mps.sv.resize(n - 1);
    for (int i = 0; i < n; ++i) {
        mps.A[i] = Mat::Zero(2, 1);
        mps.A[i](0, 0) = 1.0;  // all |up>
    }
    for (int i = 0; i < n - 1; ++i)
        mps.sv[i] = RVec::Ones(1);
    return mps;
}

// Initialize MPS to Neel state |up dn up dn...> (alternating spins), bond dim = 1
// This ensures Sz_total = 0 (for even n), placing the initial state in the
// correct symmetry sector for the Heisenberg ground state.
static MPS init_neel_state(int n) {
    MPS mps;
    mps.n = n;
    mps.A.resize(n);
    mps.sv.resize(n - 1);
    for (int i = 0; i < n; ++i) {
        mps.A[i] = Mat::Zero(2, 1);
        mps.A[i](i % 2, 0) = 1.0;  // even: s=0 (|up>), odd: s=1 (|dn>)
    }
    for (int i = 0; i < n - 1; ++i)
        mps.sv[i] = RVec::Ones(1);
    return mps;
}

// -- Environment tensors ------------------------------------------------------

// Env shape (chi_mps x chi_mpo x chi_mps) -- flat vector
// index(m1, a, m2) = m1 * chi_mpo * chi_mps_r + a * chi_mps_r + m2
struct Env {
    int chi1, chi_w, chi2;
    std::vector<Cx> data;

    Env() : chi1(0), chi_w(0), chi2(0) {}
    Env(int c1, int cw, int c2)
        : chi1(c1), chi_w(cw), chi2(c2), data(c1 * cw * c2, Cx(0)) {}

    Cx& at(int m1, int a, int m2) {
        return data[m1 * chi_w * chi2 + a * chi2 + m2];
    }
    const Cx& at(int m1, int a, int m2) const {
        return data[m1 * chi_w * chi2 + a * chi2 + m2];
    }
};

// -- Optimised environment contractions (M177) --------------------------------
//
// All three contractions (update_left, update_right, apply_heff) are
// rewritten to use Eigen matrix-matrix multiplications (DGEMM) instead of
// raw nested loops.

// Left environment update:
//   L_new[m2, b, m2'] = sum_{m1,a,s,s'} conj(A[s*chi+m1, m2]) W[a,b][s,s'] L[m1,a,m1'] A[s'*chi+m1', m2']
static Env update_left(const Env& L, const Mat& A, const MPOSite& W) {
    const int chi_new = A.cols();       // new MPS bond (= chi_R of A)
    const int chi_old = A.rows() / 2;  // old MPS bond (= chi_L of A)
    const int chi_wl  = W.chi_l;
    const int chi_wr  = W.chi_r;
    Env L_new(chi_new, chi_wr, chi_new);  // zero-initialised

    // L_flat[m1, a*chi_old + m1'] = L.at(m1, a, m1')
    Mat L_flat(chi_old, chi_wl * chi_old);
    for (int m1 = 0; m1 < chi_old; ++m1)
        for (int a = 0; a < chi_wl; ++a)
            for (int m1p = 0; m1p < chi_old; ++m1p)
                L_flat(m1, a * chi_old + m1p) = L.at(m1, a, m1p);

    // Phase 1: T1[s][m2, a*chi_old + m1'] = sum_{m1} conj(A[s*chi_old+m1, m2]) x L[m1, a, m1']
    std::array<Mat, 2> T1;
    for (int s = 0; s < 2; ++s) {
        Mat A_s_conj = A.block(s * chi_old, 0, chi_old, chi_new).conjugate();
        T1[s] = A_s_conj.transpose() * L_flat;  // [chi_new, chi_wl * chi_old]
    }

    // Phase 2: accumulate into L_new
    for (int a = 0; a < chi_wl; ++a) {
        for (int b = 0; b < chi_wr; ++b) {
            const auto& Wab = W.W(a, b);
            for (int s = 0; s < 2; ++s) {
                Mat T1_a_s = T1[s].block(0, a * chi_old, chi_new, chi_old);
                for (int sp = 0; sp < 2; ++sp) {
                    Cx w = Wab(s, sp);
                    if (std::abs(w) < 1e-15) continue;
                    Mat A_sp = A.block(sp * chi_old, 0, chi_old, chi_new);
                    Mat contrib = w * (T1_a_s * A_sp);  // O(chi^3) DGEMM
                    for (int m2 = 0; m2 < chi_new; ++m2)
                        for (int m2p = 0; m2p < chi_new; ++m2p)
                            L_new.at(m2, b, m2p) += contrib(m2, m2p);
                }
            }
        }
    }
    return L_new;
}

// Right environment update:
static Env update_right(const Env& R, const Mat& A, const MPOSite& W) {
    const int chi_r = A.cols();       // MPS right bond
    const int chi_l = A.rows() / 2;  // MPS left bond (new env bond)
    const int chi_wl = W.chi_l;
    const int chi_wr = W.chi_r;
    Env R_new(chi_l, chi_wl, chi_l);  // zero-initialised

    // R_flat_T[m2', b*chi_r + m2] = R.at(m2, b, m2')
    Mat R_flat_T(chi_r, chi_wr * chi_r);
    for (int m2 = 0; m2 < chi_r; ++m2)
        for (int b = 0; b < chi_wr; ++b)
            for (int m2p = 0; m2p < chi_r; ++m2p)
                R_flat_T(m2p, b * chi_r + m2) = R.at(m2, b, m2p);

    // Phase 1: T1_sp[m1', b*chi_r + m2] = sum_{m2'} conj(A_sp[m1', m2']) x R_flat_T[m2', b*chi_r+m2]
    std::array<Mat, 2> T1;
    for (int sp = 0; sp < 2; ++sp) {
        Mat A_sp_conj = A.block(sp * chi_l, 0, chi_l, chi_r).conjugate();
        T1[sp] = A_sp_conj * R_flat_T;  // [chi_l, chi_wr * chi_r]
    }

    // Phase 2: accumulate into R_new
    for (int a = 0; a < chi_wl; ++a) {
        for (int b = 0; b < chi_wr; ++b) {
            const auto& Wab = W.W(a, b);
            for (int s = 0; s < 2; ++s) {
                Mat A_s = A.block(s * chi_l, 0, chi_l, chi_r);
                for (int sp = 0; sp < 2; ++sp) {
                    Cx w = Wab(sp, s);
                    if (std::abs(w) < 1e-15) continue;
                    Mat T1_sp_b = T1[sp].block(0, b * chi_r, chi_l, chi_r);
                    Mat contrib = w * (A_s * T1_sp_b.transpose());  // O(chi^3) DGEMM
                    for (int m1 = 0; m1 < chi_l; ++m1)
                        for (int m1p = 0; m1p < chi_l; ++m1p)
                            R_new.at(m1, a, m1p) += contrib(m1, m1p);
                }
            }
        }
    }
    return R_new;
}

// -- Effective Hamiltonian ----------------------------------------------------
//
// Applies H_eff to the two-site tensor theta.
static Vec apply_heff(const Env& L, const MPOSite& Wi, const MPOSite& Wj,
                      const Env& R, const Vec& theta, int chi_L, int chi_R) {
    const int d2     = 4;    // d*d = 2*2
    const int chi_Wl = Wi.chi_l;
    const int chi_Wm = Wi.chi_r;  // = Wj.chi_l
    const int chi_Wr = Wj.chi_r;

    // Precompute 2-site MPO: W_comb[a][c] is a 4x4 complex matrix
    using Mat4 = Eigen::Matrix<Cx, 4, 4>;
    std::vector<std::vector<Mat4>> W_comb(chi_Wl, std::vector<Mat4>(chi_Wr, Mat4::Zero()));
    for (int a = 0; a < chi_Wl; ++a)
        for (int b = 0; b < chi_Wm; ++b)
            for (int c = 0; c < chi_Wr; ++c)
                for (int s1p = 0; s1p < 2; ++s1p)
                    for (int s2p = 0; s2p < 2; ++s2p)
                        for (int s1 = 0; s1 < 2; ++s1)
                            for (int s2 = 0; s2 < 2; ++s2)
                                W_comb[a][c](s1p*2+s2p, s1*2+s2) +=
                                    Wi.W(a,b)(s1p,s1) * Wj.W(b,c)(s2p,s2);

    // Step 1: zero-copy view of theta as [chi_L*d2, chi_R] -- ROW-MAJOR Map.
    using RMat = Eigen::Matrix<Cx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const RMat> theta_mat(theta.data(), chi_L * d2, chi_R);

    // R_mat[m2, c*chi_R + m2']
    Mat R_mat(chi_R, chi_Wr * chi_R);
    for (int m2 = 0; m2 < chi_R; ++m2)
        for (int c = 0; c < chi_Wr; ++c)
            for (int m2p = 0; m2p < chi_R; ++m2p)
                R_mat(m2, c*chi_R + m2p) = R.at(m2, c, m2p);

    // T1[chi_L*d2, chi_Wr*chi_R] -- single big DGEMM
    Mat T1 = theta_mat * R_mat;

    // Output accumulator [chi_L, d2*chi_R] stored ROW-MAJOR.
    RMat out_mat = RMat::Zero(chi_L, d2 * chi_R);

    for (int a = 0; a < chi_Wl; ++a) {
        // L_a[m1, m1'] -- build once per a
        Mat L_a(chi_L, chi_L);
        for (int m1 = 0; m1 < chi_L; ++m1)
            for (int m1p = 0; m1p < chi_L; ++m1p)
                L_a(m1, m1p) = L.at(m1p, a, m1);

        for (int c = 0; c < chi_Wr; ++c) {
            if (W_comb[a][c].isZero(1e-15)) continue;

            Mat T1_c = T1.block(0, c * chi_R, chi_L * d2, chi_R);

            Mat T1_c_perm(d2, chi_L * chi_R);
            for (int m1 = 0; m1 < chi_L; ++m1)
                for (int p = 0; p < d2; ++p)
                    T1_c_perm.block(p, m1 * chi_R, 1, chi_R) =
                        T1_c.block(m1 * d2 + p, 0, 1, chi_R);

            Mat T2_perm = W_comb[a][c] * T1_c_perm;

            Mat T2_flat(chi_L, d2 * chi_R);
            for (int m1 = 0; m1 < chi_L; ++m1)
                for (int p = 0; p < d2; ++p)
                    T2_flat.block(m1, p * chi_R, 1, chi_R) =
                        T2_perm.block(p, m1 * chi_R, 1, chi_R);

            out_mat.noalias() += L_a.transpose() * T2_flat;
        }
    }

    return Eigen::Map<Vec>(out_mat.data(), chi_L * d2 * chi_R);
}

// -- Eigensolver (Lanczos) ----------------------------------------------------

static std::pair<double, Vec> lanczos_ground_state(
        std::function<Vec(const Vec&)> matvec, int dim, const Vec& init,
        int max_iter = 100, double tol = 1e-10) {

    // Full diag for small dims
    if (dim <= 300) {
        Mat H(dim, dim);
        for (int j = 0; j < dim; ++j) {
            Vec e = Vec::Zero(dim);
            e[j] = 1.0;
            H.col(j) = matvec(e);
        }
        H = (H + H.adjoint()) * 0.5;
        Eigen::SelfAdjointEigenSolver<Mat> eig(H);
        return {eig.eigenvalues()[0], eig.eigenvectors().col(0)};
    }

    // Lanczos with Ritz-residual early exit
    constexpr int MIN_STEPS  = 10;
    constexpr int CHECK_FREQ =  5;

    int m = std::min(max_iter, dim);
    std::vector<Vec> q;
    q.reserve(m + 1);

    Vec v = init;
    double norm_v = v.norm();
    if (norm_v < 1e-14) { v = Vec::Random(dim); norm_v = v.norm(); }
    v /= norm_v;

    Eigen::VectorXd alpha(m), beta(m);
    q.push_back(v);

    int actual_m = m;

    for (int j = 0; j < m; ++j) {
        Vec w = matvec(q[j]);
        alpha[j] = w.dot(q[j]).real();
        w -= alpha[j] * q[j];
        if (j > 0) w -= beta[j - 1] * q[j - 1];

        double b = w.norm();
        beta[j] = b;

        if (b < 1e-14) { actual_m = j + 1; break; }

        if (j < m - 1) q.push_back(w / b);

        int cur = j + 1;
        if (cur >= MIN_STEPS && (cur % CHECK_FREQ == 0 || j == m - 1)) {
            Eigen::MatrixXd T_cur = Eigen::MatrixXd::Zero(cur, cur);
            for (int k = 0; k < cur;     ++k) T_cur(k, k)   = alpha[k];
            for (int k = 0; k < cur - 1; ++k) {
                T_cur(k, k+1) = beta[k];
                T_cur(k+1, k) = beta[k];
            }
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig_cur(T_cur);
            double ritz_res = b * std::abs(eig_cur.eigenvectors().col(0)[cur - 1]);
            if (ritz_res < tol) {
                actual_m = cur;
                break;
            }
        }
    }
    m = actual_m;

    Eigen::MatrixXd T(m, m);
    T.setZero();
    for (int j = 0; j < m; ++j)     T(j, j)   = alpha[j];
    for (int j = 0; j < m - 1; ++j) { T(j, j+1) = beta[j]; T(j+1, j) = beta[j]; }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(T);
    double E0 = eig.eigenvalues()[0];
    Eigen::VectorXd y = eig.eigenvectors().col(0);

    Vec psi = Vec::Zero(dim);
    int qsize = static_cast<int>(q.size());
    for (int j = 0; j < std::min(m, qsize); ++j)
        psi += y[j] * q[j];
    psi /= psi.norm();

    return {E0, psi};
}

// -- SVD truncation -----------------------------------------------------------

struct SVDResult {
    Mat U;
    RVec S;
    Mat Vt;
    double truncation_error;
    int chi_new;
};

static SVDResult svd_truncate(const Mat& theta, int max_bond, double cutoff) {
    Eigen::BDCSVD<Mat> svd(theta, Eigen::ComputeThinU | Eigen::ComputeThinV);
    RVec sv = svd.singularValues();

    int chi_new = sv.size();
    double sv_sq_sum = sv.squaredNorm();
    if (sv_sq_sum > 0) {
        double running = 0;
        for (int k = chi_new - 1; k >= 0; --k) {
            running += sv[k] * sv[k];
            if (running / sv_sq_sum > cutoff * cutoff) {
                chi_new = k + 1;
                break;
            }
        }
    }
    chi_new = std::min(chi_new, max_bond);
    chi_new = std::max(chi_new, 1);

    double trunc = 0;
    for (int k = chi_new; k < sv.size(); ++k)
        trunc += sv[k] * sv[k];

    SVDResult res;
    res.U    = svd.matrixU().leftCols(chi_new);
    res.S    = sv.head(chi_new);
    res.Vt   = svd.matrixV().adjoint().topRows(chi_new);
    res.truncation_error = trunc;
    res.chi_new = chi_new;
    return res;
}

// -- Compute energy -----------------------------------------------------------

double compute_energy(const MPS& mps, const std::vector<MPOSite>& mpo) {
    int n = mps.n;
    Env L(1, mpo[0].chi_l, 1);
    L.at(0, 0, 0) = 1.0;

    for (int i = 0; i < n; ++i) {
        L = update_left(L, mps.A[i], mpo[i]);
    }
    return L.at(0, 0, 0).real();
}

// -- DMRG sweep ---------------------------------------------------------------

DMRGResult run_dmrg(const std::vector<MPOSite>& mpo, int n,
                    int max_bond, double cutoff,
                    int n_sweeps, double tol) {
    auto t_start = std::chrono::steady_clock::now();

    MPS mps = init_neel_state(n);

    // Build initial right environments R[i] for i = n-1, n-2, ..., 0
    std::vector<Env> R(n + 1);
    {
        int chi_r = mpo[n-1].chi_r;
        R[n] = Env(1, chi_r, 1);
        R[n].at(0, 0, 0) = 1.0;
    }
    for (int i = n - 1; i >= 0; --i) {
        R[i] = update_right(R[i + 1], mps.A[i], mpo[i]);
    }

    // Left environments L[0] = boundary
    std::vector<Env> L(n + 1);
    {
        int chi_l = mpo[0].chi_l;
        L[0] = Env(1, chi_l, 1);
        L[0].at(0, 0, 0) = 1.0;
    }

    DMRGResult result;
    result.energy           = 0.0;
    result.truncation_error = 0.0;
    result.n_sweeps_done    = 0;
    result.wall_time_s      = 0.0;
    double prev_energy = 0.0;

    for (int sweep = 0; sweep < n_sweeps; ++sweep) {
        double sweep_trunc = 0;
        double E0 = 0;

        // -- Left-to-right half sweep -----------------------------------------
        for (int i = 0; i < n - 1; ++i) {
            const Mat& Ai  = mps.A[i];
            const Mat& Ai1 = mps.A[i + 1];
            int chi_L = Ai.rows() / 2;
            int chi_M = Ai.cols();
            int chi_R = Ai1.cols();

            // Absorb singular values into Ai+1 from the left
            Mat Ai1_s = Ai1;
            if (i > 0) {
                RVec& sv = mps.sv[i];
                Ai1_s = Ai1;
                for (int l = 0; l < chi_M; ++l)
                    for (int s = 0; s < 2; ++s)
                        for (int r = 0; r < chi_R; ++r)
                            Ai1_s(s * chi_M + l, r) *= (l < sv.size() ? mps.sv[i][l] : 0.0);
            }

            // Form two-site tensor theta
            int theta_dim = chi_L * 4 * chi_R;
            Vec theta(theta_dim);
            for (int m1 = 0; m1 < chi_L; ++m1)
                for (int s1 = 0; s1 < 2; ++s1)
                    for (int s2 = 0; s2 < 2; ++s2)
                        for (int m2 = 0; m2 < chi_R; ++m2) {
                            Cx v(0);
                            for (int l = 0; l < chi_M; ++l)
                                v += Ai(s1 * chi_L + m1, l) * Ai1_s(s2 * chi_M + l, m2);
                            theta[m1 * 4 * chi_R + s1 * 2 * chi_R + s2 * chi_R + m2] = v;
                        }

            // Find ground state of H_eff
            auto matvec = [&](const Vec& v) {
                return apply_heff(L[i], mpo[i], mpo[i+1], R[i+2], v, chi_L, chi_R);
            };
            auto [E, psi] = lanczos_ground_state(matvec, theta_dim, theta);
            E0 = E;

            // Reshape psi to (chi_L*2, 2*chi_R) for SVD
            Mat Theta_mat(chi_L * 2, 2 * chi_R);
            for (int m1 = 0; m1 < chi_L; ++m1)
                for (int s1 = 0; s1 < 2; ++s1)
                    for (int s2 = 0; s2 < 2; ++s2)
                        for (int m2 = 0; m2 < chi_R; ++m2)
                            Theta_mat(s1 * chi_L + m1, s2 * chi_R + m2) =
                                psi[m1 * 4 * chi_R + s1 * 2 * chi_R + s2 * chi_R + m2];

            // SVD truncation
            auto sv_res = svd_truncate(Theta_mat, max_bond, cutoff);
            sweep_trunc += sv_res.truncation_error;

            // Update MPS
            mps.A[i] = sv_res.U;  // shape (chi_L*2, chi_new)
            mps.sv[i] = sv_res.S;
            int chi_new = sv_res.chi_new;
            Mat Vt = sv_res.Vt;  // (chi_new, 2*chi_R)
            mps.A[i + 1] = Mat(2 * chi_new, chi_R);
            for (int l = 0; l < chi_new; ++l)
                for (int s2 = 0; s2 < 2; ++s2)
                    for (int m2 = 0; m2 < chi_R; ++m2)
                        mps.A[i + 1](s2 * chi_new + l, m2) = Vt(l, s2 * chi_R + m2);

            // Update left environment
            L[i + 1] = update_left(L[i], mps.A[i], mpo[i]);
        }

        // -- Right-to-left half sweep -----------------------------------------
        for (int i = n - 2; i >= 0; --i) {
            const Mat& Ai  = mps.A[i];
            const Mat& Ai1 = mps.A[i + 1];
            int chi_L = Ai.rows() / 2;
            int chi_M = Ai.cols();
            int chi_R = Ai1.cols();

            // Absorb singular values from right into A[i]
            Mat Ai_s = Ai;
            {
                for (int l = 0; l < chi_M; ++l)
                    for (int s = 0; s < 2; ++s)
                        for (int m1 = 0; m1 < chi_L; ++m1)
                            Ai_s(s * chi_L + m1, l) *= (l < mps.sv[i].size() ? mps.sv[i][l] : 0.0);
            }

            int theta_dim = chi_L * 4 * chi_R;
            Vec theta(theta_dim);
            for (int m1 = 0; m1 < chi_L; ++m1)
                for (int s1 = 0; s1 < 2; ++s1)
                    for (int s2 = 0; s2 < 2; ++s2)
                        for (int m2 = 0; m2 < chi_R; ++m2) {
                            Cx v(0);
                            for (int l = 0; l < chi_M; ++l)
                                v += Ai_s(s1 * chi_L + m1, l) * Ai1(s2 * chi_M + l, m2);
                            theta[m1 * 4 * chi_R + s1 * 2 * chi_R + s2 * chi_R + m2] = v;
                        }

            auto matvec = [&](const Vec& v) {
                return apply_heff(L[i], mpo[i], mpo[i+1], R[i+2], v, chi_L, chi_R);
            };
            auto [E, psi] = lanczos_ground_state(matvec, theta_dim, theta);
            E0 = E;

            Mat Theta_mat(chi_L * 2, 2 * chi_R);
            for (int m1 = 0; m1 < chi_L; ++m1)
                for (int s1 = 0; s1 < 2; ++s1)
                    for (int s2 = 0; s2 < 2; ++s2)
                        for (int m2 = 0; m2 < chi_R; ++m2)
                            Theta_mat(s1 * chi_L + m1, s2 * chi_R + m2) =
                                psi[m1 * 4 * chi_R + s1 * 2 * chi_R + s2 * chi_R + m2];

            auto sv_res = svd_truncate(Theta_mat, max_bond, cutoff);
            sweep_trunc += sv_res.truncation_error;

            int chi_new = sv_res.chi_new;
            mps.A[i] = sv_res.U;  // (chi_L*2, chi_new)
            mps.sv[i] = sv_res.S;
            Mat Vt = sv_res.Vt;
            mps.A[i + 1] = Mat(2 * chi_new, chi_R);
            for (int l = 0; l < chi_new; ++l)
                for (int s2 = 0; s2 < 2; ++s2)
                    for (int m2 = 0; m2 < chi_R; ++m2)
                        mps.A[i + 1](s2 * chi_new + l, m2) = Vt(l, s2 * chi_R + m2);

            // Update right environment
            R[i + 1] = update_right(R[i + 2], mps.A[i + 1], mpo[i + 1]);
        }

        result.sweep_energies.push_back(E0);
        result.truncation_error = sweep_trunc;

        // Convergence check
        if (sweep > 0 && std::abs(E0 - prev_energy) < tol) {
            result.n_sweeps_done = sweep + 1;
            result.energy = E0;
            break;
        }
        prev_energy = E0;
        result.energy = E0;
        result.n_sweeps_done = sweep + 1;
    }

    auto t_end = std::chrono::steady_clock::now();
    result.wall_time_s = std::chrono::duration<double>(t_end - t_start).count();
    return result;
}

} // namespace qx_dmrg
