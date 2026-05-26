/**
 * M176 — C++ MPS tensor contractions + two-site DMRG.
 *
 * Implements a complete two-site DMRG ground-state finder for 1D spin-1/2
 * chains, with a JSON C ABI matching the rest of the QuantX engine.
 *
 * Supported Hamiltonians
 * ----------------------
 * - Heisenberg XXX:   {"type": "heisenberg", "J": 1.0}
 * - General sum of Pauli strings (nearest-neighbour or longer range):
 *   {"type": "pauli_sum", "terms": [{"coeff": 1.0, "pauli": "X0X1"}, ...]}
 *   Pauli strings may be given in indexed format ("X0X1") or plain ("XXI..").
 *
 * MPS convention (left-canonical bulk)
 * -------------------------------------
 * Site tensor A[i] stored as Eigen::MatrixXcd of shape (chi_L * d, chi_R).
 * Row index:  s * chi_L + l   (s ∈ {0,1}, l ∈ {0..chi_L-1})
 * Col index:  r               (r ∈ {0..chi_R-1})
 *
 * Environment tensors
 * -------------------
 * L_env[i] : (chi_mps × chi_mpo × chi_mps) stored as flat vector.
 *            Flat index: m1 * (chi_mpo * chi_mps2) + a * chi_mps2 + m2
 * R_env[i] : same shape, reading from right.
 *
 * References
 * ----------
 * [1] Schollwöck (2011). Ann. Phys. 326, 96. DOI: 10.1016/j.aop.2010.09.012
 * [2] White (1992). Phys. Rev. Lett. 69, 2863. DOI: 10.1103/PhysRevLett.69.2863
 */
#include "qx/c_api.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <functional>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <nlohmann/json.hpp>

namespace {

using Cx      = std::complex<double>;
using Mat     = Eigen::MatrixXcd;
using Vec     = Eigen::VectorXcd;
using RVec    = Eigen::VectorXd;
using Mat2    = Eigen::Matrix2cd;
using nlj     = nlohmann::json;

// ── Pauli matrices ──────────────────────────────────────────────────────────

static Mat2 pauli(char c) {
    Mat2 m;
    switch (c) {
        case 'X': m << 0,1, 1,0; break;
        case 'Y': m << 0,Cx(0,-1), Cx(0,1),0; break;
        case 'Z': m << 1,0, 0,-1; break;
        default:  m = Mat2::Identity(); break;
    }
    return m;
}

// ── MPO ─────────────────────────────────────────────────────────────────────

// MPO site tensor: ops[alpha][beta] is a 2×2 matrix W^{alpha,beta}_{s,s'}
struct MPOSite {
    int chi_l, chi_r;
    // Flat storage: ops[a * chi_r + b] = 2x2 matrix
    std::vector<Mat2> ops;
    const Mat2& W(int a, int b) const { return ops[a * chi_r + b]; }
    Mat2& W(int a, int b)             { return ops[a * chi_r + b]; }
};

// Build Heisenberg MPO: H = J Σ_i (XᵢXᵢ₊₁ + YᵢYᵢ₊₁ + ZᵢZᵢ₊₁)
// Standard 5-bond-dim MPO (Schollwöck 2011 Table 4)
//
//  W = [ I    0    0    0   0 ]
//      [ X    0    0    0   0 ]
//      [ Y    0    0    0   0 ]
//      [ Z    0    0    0   0 ]
//      [ 0   J*X  J*Y  J*Z  I ]
//
// Left  boundary: row 4 (last row)
// Right boundary: col 0 (first col)
static std::vector<MPOSite> build_heisenberg_mpo(int n, double J) {
    const int chi = 5;
    const int d   = 2;
    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        int cl = (i == 0)     ? 1 : chi;
        int cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl;
        site.chi_r = cr;
        site.ops.resize(cl * cr, Mat2::Zero());

        // Helper lambdas for local alpha/beta indices in full chi x chi matrix
        // For first site: left boundary selects row 4 only → cl=1 means a=0 maps to row 4
        // For last  site: right boundary selects col 0 only → cr=1 means b=0 maps to col 0
        // We build the full 5x5 MPO and then extract the relevant slice.
        // Simpler: store the full bulk MPO and special-case boundaries via wrappers.

        // Full 5x5 W matrix at bulk sites
        std::array<std::array<Mat2, 5>, 5> W5 {};
        for (auto& row : W5) for (auto& e : row) e.setZero();
        W5[0][0] = pauli('I');
        W5[1][0] = pauli('X');
        W5[2][0] = pauli('Y');
        W5[3][0] = pauli('Z');
        W5[4][1] = J * pauli('X');
        W5[4][2] = J * pauli('Y');
        W5[4][3] = J * pauli('Z');
        W5[4][4] = pauli('I');

        // Map cl × cr indices into the 5×5 matrix
        // Left boundary: cl=1 → only row 4 is "active"
        // Right boundary: cr=1 → only col 0 is "active"
        int row_offset = (i == 0) ? 4 : 0;
        int col_offset = (i == n - 1) ? 0 : 0;  // col offset always 0, but limit to cr
        for (int a = 0; a < cl; ++a) {
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 4 : a;
                int br = (i == n - 1) ? 0 : b;
                site.W(a, b) = W5[ar][br];
            }
        }
    }
    return mpo;
}

// ── Pauli label parser ──────────────────────────────────────────────────────

// Convert indexed label ("X0X1") or plain label ("XXI") to per-site char array.
static std::vector<char> parse_pauli(const std::string& label, int n) {
    std::vector<char> ops(n, 'I');
    static const std::regex idx_re(R"(([XYZI])(\d+))");
    std::sregex_iterator it(label.begin(), label.end(), idx_re), end;
    if (it != end) {
        // Indexed format
        for (; it != end; ++it) {
            char op  = (*it)[1].str()[0];
            int  q   = std::stoi((*it)[2].str());
            if (q < n) ops[q] = op;
        }
    } else {
        // Plain format: pad/trim to length n
        for (int i = 0; i < n && i < (int)label.size(); ++i)
            ops[i] = std::toupper(label[i]);
    }
    return ops;
}

// Build MPO from a list of (coeff, pauli_label) terms.
// Naive construction: bond dim = n_terms + 2 (one state per term, plus boundaries).
// The MPO encodes H = Σ_k c_k P^k_0 ⊗ P^k_1 ⊗ ... ⊗ P^k_{n-1}.
static std::vector<MPOSite> build_pauli_sum_mpo(
        int n, const std::vector<std::pair<double, std::string>>& terms) {
    int nt = static_cast<int>(terms.size());
    // States: 0 = "left vac", 1..nt = "in progress for term k", nt+1 = "done"
    int full_chi = nt + 2;

    // Pre-parse all pauli strings
    std::vector<std::vector<char>> parsed(nt);
    std::vector<double> coeffs(nt);
    for (int k = 0; k < nt; ++k) {
        coeffs[k] = terms[k].first;
        parsed[k] = parse_pauli(terms[k].second, n);
    }

    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        int cl = (i == 0)     ? 1   : full_chi;
        int cr = (i == n - 1) ? 1   : full_chi;
        auto& site = mpo[i];
        site.chi_l = cl;
        site.chi_r = cr;
        site.ops.resize(cl * cr, Mat2::Zero());

        // Build full (full_chi × full_chi) W and then extract boundary slice.
        // W[done, done] = I       (pass-through for completed terms)
        // W[vac, vac]   = I       (pass-through for not-yet-started terms)
        // W[vac, in_k]  = coeff_k * P^k_i   for first occurrence
        // W[in_k, in_k] = P^k_i              for middle occurrences
        // W[in_k, done] = P^k_i              for last occurrence
        // But since Pauli strings are product operators, all sites are "in progress"
        // simultaneously. Better model: each term k lives independently.
        // Simplification: use W[0, k+1] = c_k * P^k_i on left boundary site,
        //                     W[k+1, k+1] = P^k_i on bulk,
        //                     W[k+1, full_chi-1] = P^k_i on right boundary site.
        // Plus W[0, 0] = I (lower-left pass), W[nt+1, nt+1] = I (upper-right pass).
        // This is the standard finite-state automaton construction.

        std::vector<std::vector<Mat2>> W(full_chi, std::vector<Mat2>(full_chi, Mat2::Zero()));
        W[0][0] = pauli('I');                   // left pass-through
        W[full_chi-1][full_chi-1] = pauli('I'); // right pass-through

        for (int k = 0; k < nt; ++k) {
            char p = parsed[k][i];
            if (i == 0) {
                W[0][k + 1] = coeffs[k] * pauli(p);
            } else if (i == n - 1) {
                W[k + 1][full_chi - 1] = pauli(p);
            } else {
                W[k + 1][k + 1] = pauli(p);
            }
            // On the left boundary site also seed the in-progress state
            if (i == 0) {
                // Already set above; also propagate "done" if n==1
                if (n == 1) W[0][full_chi - 1] = coeffs[k] * pauli(p);
            }
        }

        // Extract slice
        int row_start = (i == 0) ? 0 : 0;
        int col_start = 0;
        for (int a = 0; a < cl; ++a) {
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? (full_chi - 1) : a; // left bdy picks last row
                int br = (i == n-1) ? 0              : b; // right bdy picks first col
                // Actually: left boundary state is "vac"=0, but we want the operator
                // to start in the correct state. Let's use row 0 (vac) for first site.
                // For left boundary (i=0): cl=1, a=0 → maps to row 0 (vac state)
                // For right boundary (i=n-1): cr=1, b=0 → maps to col full_chi-1 (done state)
                int ar2 = (i == 0) ? 0              : a;
                int br2 = (i == n-1) ? (full_chi-1) : b;
                site.W(a, b) = W[ar2][br2];
            }
        }
    }
    return mpo;
}

// ── MPS ─────────────────────────────────────────────────────────────────────

struct MPS {
    int n;
    std::vector<Mat> A;    // A[i]: (chi_L*2, chi_R), left-canonical
    std::vector<RVec> sv;  // sv[i]: singular values at bond i  (n-1 bonds, bond i is between i and i+1)
};

// Initialize MPS to |0...0⟩ (all qubits in |0⟩ state), bond dim = 1
static MPS init_zero_state(int n) {
    MPS mps;
    mps.n = n;
    mps.A.resize(n);
    mps.sv.resize(n - 1);
    for (int i = 0; i < n; ++i) {
        // Shape: (1*2, 1) = (2, 1)
        mps.A[i] = Mat::Zero(2, 1);
        mps.A[i](0, 0) = 1.0;  // s=0 (|0⟩), l=0, r=0
    }
    for (int i = 0; i < n - 1; ++i)
        mps.sv[i] = RVec::Ones(1);
    return mps;
}

// ── Environment tensors ──────────────────────────────────────────────────────

// Env shape (chi_mps × chi_mpo × chi_mps) — flat vector
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

// Left environment: L_new[m2, b, m2'] = Σ_{m1,a,s,s'} conj(A[s*chi_L+m1, m2]) * W[a,b][s,s'] * L[m1,a,m1'] * A[s'*chi_L+m1', m2']
static Env update_left(const Env& L, const Mat& A, const MPOSite& W) {
    int chi_l = A.cols();                       // new left bond = chi_R of A
    int chi_l_old = A.rows() / 2;               // old left bond = chi_L of A
    int chi_w_new = W.chi_r;
    Env L_new(chi_l, chi_w_new, chi_l);

    for (int m2 = 0; m2 < chi_l; ++m2) {
        for (int b = 0; b < chi_w_new; ++b) {
            for (int m2p = 0; m2p < chi_l; ++m2p) {
                Cx val(0);
                for (int m1 = 0; m1 < chi_l_old; ++m1) {
                    for (int a = 0; a < W.chi_l; ++a) {
                        const Cx L_ma = L.at(m1, a, 0); // will fix indexing below
                        // Actually L has chi1 = chi_l_old, chi2 = chi_l_old
                        // L.at(m1, a, m1') for the old env
                        (void)L_ma;
                        for (int m1p = 0; m1p < chi_l_old; ++m1p) {
                            Cx Lval = L.at(m1, a, m1p);
                            if (Lval == Cx(0)) continue;
                            const Mat2& Wab = W.W(a, b);
                            for (int s = 0; s < 2; ++s) {
                                Cx Aconj = std::conj(A(s * chi_l_old + m1, m2));
                                if (Aconj == Cx(0)) continue;
                                for (int sp = 0; sp < 2; ++sp) {
                                    Cx Wval = Wab(s, sp);
                                    if (Wval == Cx(0)) continue;
                                    Cx Aval = A(sp * chi_l_old + m1p, m2p);
                                    val += Aconj * Wval * Lval * Aval;
                                }
                            }
                        }
                    }
                }
                L_new.at(m2, b, m2p) += val;
            }
        }
    }
    return L_new;
}

// Right environment: R_new[m1, a, m1'] = Σ_{m2,b,s,s'} A[s*chi_L+m1, m2] * W[a,b][s,s'] * R[m2,b,m2'] * conj(A[s'*chi_L+m1', m2'])
static Env update_right(const Env& R, const Mat& A, const MPOSite& W) {
    int chi_r_old = A.cols();    // right bond of current site = chi_R of A
    int chi_l    = A.rows() / 2; // left bond
    int chi_w_new = W.chi_l;
    Env R_new(chi_l, chi_w_new, chi_l);

    for (int m1 = 0; m1 < chi_l; ++m1) {
        for (int a = 0; a < chi_w_new; ++a) {
            for (int m1p = 0; m1p < chi_l; ++m1p) {
                Cx val(0);
                for (int m2 = 0; m2 < chi_r_old; ++m2) {
                    for (int b = 0; b < W.chi_r; ++b) {
                        Cx Rval = R.at(m2, b, 0); // fix below
                        (void)Rval;
                        for (int m2p = 0; m2p < chi_r_old; ++m2p) {
                            Cx Rv = R.at(m2, b, m2p);
                            if (Rv == Cx(0)) continue;
                            const Mat2& Wab = W.W(a, b);
                            for (int s = 0; s < 2; ++s) {
                                Cx Aval = A(s * chi_l + m1, m2);
                                if (Aval == Cx(0)) continue;
                                for (int sp = 0; sp < 2; ++sp) {
                                    Cx Wval = Wab(s, sp);
                                    if (Wval == Cx(0)) continue;
                                    Cx Aconj = std::conj(A(sp * chi_l + m1p, m2p));
                                    val += Aval * Wval * Rv * Aconj;
                                }
                            }
                        }
                    }
                }
                R_new.at(m1, a, m1p) += val;
            }
        }
    }
    return R_new;
}

// ── Effective Hamiltonian ────────────────────────────────────────────────────

// theta: length chi_L * 2 * 2 * chi_R
// index: m1 * (4 * chi_R) + s1 * (2 * chi_R) + s2 * chi_R + m2
static Vec apply_heff(const Env& L, const MPOSite& Wi, const MPOSite& Wj,
                      const Env& R, const Vec& theta, int chi_L, int chi_R) {
    const int d = 2;
    Vec out = Vec::Zero(chi_L * d * d * chi_R);

    auto idx = [&](int m, int s1, int s2, int r) {
        return m * (d * d * chi_R) + s1 * (d * chi_R) + s2 * chi_R + r;
    };

    for (int m1p = 0; m1p < chi_L; ++m1p) {
        for (int s1p = 0; s1p < d; ++s1p) {
            for (int s2p = 0; s2p < d; ++s2p) {
                for (int m2p = 0; m2p < chi_R; ++m2p) {
                    Cx val(0);
                    for (int m1 = 0; m1 < chi_L; ++m1) {
                        for (int m2 = 0; m2 < chi_R; ++m2) {
                            for (int a = 0; a < Wi.chi_l; ++a) {
                                for (int b = 0; b < Wi.chi_r; ++b) {
                                    for (int c = 0; c < Wj.chi_r; ++c) {
                                        Cx Lval = L.at(m1, a, m1p);
                                        if (Lval == Cx(0)) continue;
                                        Cx Rval = R.at(m2, c, m2p);
                                        if (Rval == Cx(0)) continue;
                                        for (int s1 = 0; s1 < d; ++s1) {
                                            Cx Wival = Wi.W(a,b)(s1p, s1);
                                            if (Wival == Cx(0)) continue;
                                            for (int s2 = 0; s2 < d; ++s2) {
                                                Cx Wjval = Wj.W(b,c)(s2p, s2);
                                                if (Wjval == Cx(0)) continue;
                                                val += Lval * Wival * Wjval * Rval
                                                       * theta[idx(m1, s1, s2, m2)];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    out[idx(m1p, s1p, s2p, m2p)] += val;
                }
            }
        }
    }
    return out;
}

// ── Eigensolver (Lanczos) ────────────────────────────────────────────────────

// Simple Lanczos to find ground state energy and vector.
// Falls back to full diagonalisation for tiny dim.
static std::pair<double, Vec> lanczos_ground_state(
        std::function<Vec(const Vec&)> matvec, int dim, const Vec& init,
        int max_iter = 200, double tol = 1e-12) {

    // Full diag for small dims
    if (dim <= 300) {
        Mat H(dim, dim);
        for (int j = 0; j < dim; ++j) {
            Vec e = Vec::Zero(dim);
            e[j] = 1.0;
            H.col(j) = matvec(e);
        }
        // Symmetrize (should already be Hermitian, but guard for numerics)
        H = (H + H.adjoint()) * 0.5;
        Eigen::SelfAdjointEigenSolver<Mat> eig(H);
        return {eig.eigenvalues()[0], eig.eigenvectors().col(0)};
    }

    // Lanczos
    int m = std::min(max_iter, dim);
    std::vector<Vec> q;
    q.reserve(m + 1);

    Vec v = init;
    double norm = v.norm();
    if (norm < 1e-14) {
        v = Vec::Random(dim);
        norm = v.norm();
    }
    v /= norm;

    Eigen::VectorXd alpha(m), beta(m - 1);
    q.push_back(v);

    for (int j = 0; j < m; ++j) {
        Vec w = matvec(q[j]);
        alpha[j] = w.dot(q[j]).real();
        w -= alpha[j] * q[j];
        if (j > 0) w -= beta[j - 1] * q[j - 1];

        double b = w.norm();
        if (j < m - 1) {
            beta[j] = b;
            if (b < 1e-14) { m = j + 1; break; }
            q.push_back(w / b);
        }
    }

    // Diagonalise the (m x m) tridiagonal matrix
    Eigen::MatrixXd T(m, m);
    T.setZero();
    for (int j = 0; j < m; ++j) T(j, j) = alpha[j];
    for (int j = 0; j < m - 1; ++j) { T(j, j+1) = beta[j]; T(j+1, j) = beta[j]; }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(T);
    double E0 = eig.eigenvalues()[0];
    Eigen::VectorXd y = eig.eigenvectors().col(0);

    // Reconstruct Ritz vector
    Vec psi = Vec::Zero(dim);
    int qsize = static_cast<int>(q.size());
    for (int j = 0; j < std::min(m, qsize); ++j)
        psi += y[j] * q[j];
    psi /= psi.norm();

    return {E0, psi};
}

// ── SVD truncation ───────────────────────────────────────────────────────────

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

    // Determine truncation
    int chi_new = sv.size();
    // Apply cutoff
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
    // Apply max_bond
    chi_new = std::min(chi_new, max_bond);
    chi_new = std::max(chi_new, 1);

    // Truncation error = discarded weight
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

// ── DMRG sweep ───────────────────────────────────────────────────────────────

struct DMRGResult {
    double energy;
    double truncation_error;
    std::vector<double> sweep_energies;
    int n_sweeps_done;
    double wall_time_s;
};

// Compute <psi|H|psi> using the left environments
static double compute_energy(const MPS& mps, const std::vector<MPOSite>& mpo) {
    int n = mps.n;
    // Build left boundary: L[0] = scalar 1, shape (1,1,1)
    Env L(1, mpo[0].chi_l, 1);
    L.at(0, 0, 0) = 1.0;

    for (int i = 0; i < n; ++i) {
        L = update_left(L, mps.A[i], mpo[i]);
    }
    // L is now a (1, 1, 1) scalar = energy
    return L.at(0, 0, 0).real();
}

static DMRGResult run_dmrg(const std::vector<MPOSite>& mpo, int n,
                           int max_bond, double cutoff,
                           int n_sweeps, double tol) {
    auto t_start = std::chrono::steady_clock::now();

    MPS mps = init_zero_state(n);

    // Build initial right environments R[i] for i = n-1, n-2, ..., 0
    // R[n] = boundary (1 × 1 × 1)
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

        // ── Left-to-right half sweep ──────────────────────────────────────
        for (int i = 0; i < n - 1; ++i) {
            // Form two-site tensor Theta = A[i] * diag(sv[i]) * A[i+1]
            // A[i]:   (chi_L * 2, chi_M)
            // A[i+1]: (chi_M * 2, chi_R)
            const Mat& Ai  = mps.A[i];
            const Mat& Ai1 = mps.A[i + 1];
            int chi_L = Ai.rows() / 2;
            int chi_M = Ai.cols();
            int chi_R = Ai1.cols();

            // Absorb singular values into Ai+1 from the left
            Mat Ai1_s = Ai1;
            if (i > 0) {
                RVec& sv = mps.sv[i];
                for (int r = 0; r < chi_M; ++r)
                    Ai1_s.row(r / 2 * 2 + r % 2) *= (r < sv.size() ? sv[r] : 0.0);
                // Actually the singular values scale the rows of Ai1 by bond index
                // Correct: Ai1_s(s*chi_M + l, r) *= sv[l]
                Ai1_s = Ai1;
                for (int l = 0; l < chi_M; ++l)
                    for (int s = 0; s < 2; ++s)
                        for (int r = 0; r < chi_R; ++r)
                            Ai1_s(s * chi_M + l, r) *= (l < sv.size() ? mps.sv[i][l] : 0.0);
            }

            // Theta[m1, s1, s2, m2] = Σ_l A[i](s1*chi_L+m1, l) * Ai1_s(s2*chi_M+l, m2)
            // Flatten Theta to chi_L*2*2*chi_R vector
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

            // Update MPS: A[i] = U (left-canonical), store sv, A[i+1] = Vt
            mps.A[i] = sv_res.U;  // shape (chi_L*2, chi_new)
            mps.sv[i] = sv_res.S;
            // A[i+1] gets Vt (chi_new x 2*chi_R) reshaped to (2*chi_new, chi_R)
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

        // ── Right-to-left half sweep ──────────────────────────────────────
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
            // A[i] = U (left-canonical), A[i+1] = Vt (right-canonical)
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

// ── JSON C ABI ────────────────────────────────────────────────────────────────

static thread_local std::string tl_dmrg_buf;

static const char* dmrg_error(const std::string& msg) {
    nlj out;
    out["ok"] = false;
    out["error"] = msg;
    tl_dmrg_buf = out.dump();
    return tl_dmrg_buf.c_str();
}

// Parse Hamiltonian from JSON into MPO
static std::vector<MPOSite> parse_ham(const nlj& prob, int n) {
    std::string type = prob.value("type", "pauli_sum");
    if (type == "heisenberg") {
        double J = prob.value("J", 1.0);
        return build_heisenberg_mpo(n, J);
    }
    // pauli_sum
    std::vector<std::pair<double, std::string>> terms;
    if (prob.contains("terms")) {
        for (const auto& t : prob["terms"]) {
            double c  = t.value("coeff", 1.0);
            std::string p = t.value("pauli", "I");
            terms.emplace_back(c, p);
        }
    } else if (prob.contains("hamiltonian_terms")) {
        for (const auto& t : prob["hamiltonian_terms"]) {
            double c = t[0].get<double>();
            std::string p = t[1].get<std::string>();
            terms.emplace_back(c, p);
        }
    }
    return build_pauli_sum_mpo(n, terms);
}

}  // namespace

// ── Public C API ─────────────────────────────────────────────────────────────

extern "C" {

QX_API const char* qx_dmrg_sweep(const char* problem_json,
                                  const char* options_json) {
    try {
        nlj prob  = nlj::parse(problem_json  ? problem_json  : "{}");
        nlj opts  = nlj::parse(options_json  ? options_json  : "{}");

        int  n        = prob.value("n_qubits",  8);
        int  max_bond = opts.value("max_bond",  64);
        double cutoff = opts.value("cutoff",    1e-10);
        int  n_sweeps = opts.value("n_sweeps",  10);
        double tol    = opts.value("tol",       1e-10);

        if (n < 2)        return dmrg_error("n_qubits must be >= 2");
        if (n > 200)      return dmrg_error("n_qubits exceeds max (200)");
        if (n_sweeps < 1) return dmrg_error("n_sweeps must be >= 1");

        auto mpo = parse_ham(prob, n);
        DMRGResult res = run_dmrg(mpo, n, max_bond, cutoff, n_sweeps, tol);

        nlj out;
        out["ok"]               = true;
        out["energy"]           = res.energy;
        out["truncation_error"] = res.truncation_error;
        out["n_sweeps_done"]    = res.n_sweeps_done;
        out["wall_time_s"]      = res.wall_time_s;
        out["sweep_energies"]   = res.sweep_energies;

        tl_dmrg_buf = out.dump();
        return tl_dmrg_buf.c_str();
    } catch (const std::exception& e) {
        return dmrg_error(e.what());
    }
}

QX_API const char* qx_mps_energy(const char* problem_json) {
    // For now, delegate to qx_dmrg_sweep with n_sweeps=0 and return energy
    // from a single expectation-value evaluation.
    try {
        nlj prob = nlj::parse(problem_json ? problem_json : "{}");
        int n    = prob.value("n_qubits", 8);
        auto mpo = parse_ham(prob, n);
        MPS mps  = init_zero_state(n);
        double E = compute_energy(mps, mpo);
        nlj out;
        out["ok"]     = true;
        out["energy"] = E;
        tl_dmrg_buf = out.dump();
        return tl_dmrg_buf.c_str();
    } catch (const std::exception& e) {
        return dmrg_error(e.what());
    }
}

}  // extern "C"
