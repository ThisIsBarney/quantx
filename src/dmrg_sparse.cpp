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

// ── Optimised environment contractions (M177) ──────────────────────────────
//
// All three contractions (update_left, update_right, apply_heff) are
// rewritten to use Eigen matrix-matrix multiplications (DGEMM) instead of
// raw nested loops.  This reduces the dominant cost from O(χ⁵·χ_W²·d⁴) to
// O(nnz_W · χ³ · d²), where nnz_W is the number of non-zero 2×2 blocks in
// the MPO tensor W (≈ 8 for the Heisenberg MPO with χ_W = 5).
//
// Reference: Schollwöck (2011), Sec. VI.C (environment update tensor network).

// Left environment update:
//   L_new[m2, b, m2'] = Σ_{m1,a,s,s'} conj(A[s·χ+m1, m2]) W[a,b][s,s'] L[m1,a,m1'] A[s'·χ+m1', m2']
//
// Algorithm:
//   Phase 1: For each spin s, compute T1_s = conj(A_s)ᵀ @ L_flat  [χ_new × χ_W·χ_old]
//   Phase 2: For each (a,b,s,sp) with W[a,b][s,sp] ≠ 0,
//            L_new[b] += w × T1_s_block(a) @ A_sp
//            where T1_s_block(a) = T1_s[:, a·χ_old:(a+1)·χ_old] [χ_new × χ_old]
static Env update_left(const Env& L, const Mat& A, const MPOSite& W) {
    const int chi_new = A.cols();       // new MPS bond (= chi_R of A)
    const int chi_old = A.rows() / 2;  // old MPS bond (= chi_L of A)
    const int chi_wl  = W.chi_l;
    const int chi_wr  = W.chi_r;
    Env L_new(chi_new, chi_wr, chi_new);  // zero-initialised

    // L_flat[m1, a·chi_old + m1'] = L.at(m1, a, m1')
    Mat L_flat(chi_old, chi_wl * chi_old);
    for (int m1 = 0; m1 < chi_old; ++m1)
        for (int a = 0; a < chi_wl; ++a)
            for (int m1p = 0; m1p < chi_old; ++m1p)
                L_flat(m1, a * chi_old + m1p) = L.at(m1, a, m1p);

    // Phase 1: T1[s][m2, a·chi_old + m1'] = Σ_{m1} conj(A[s·chi_old+m1, m2]) × L[m1, a, m1']
    std::array<Mat, 2> T1;
    for (int s = 0; s < 2; ++s) {
        Mat A_s_conj = A.block(s * chi_old, 0, chi_old, chi_new).conjugate();
        T1[s] = A_s_conj.transpose() * L_flat;  // [chi_new, chi_wl · chi_old]
    }

    // Phase 2: accumulate into L_new
    for (int a = 0; a < chi_wl; ++a) {
        for (int b = 0; b < chi_wr; ++b) {
            const Mat2& Wab = W.W(a, b);
            for (int s = 0; s < 2; ++s) {
                // T1[s] block for this a: [chi_new, chi_old]
                Mat T1_a_s = T1[s].block(0, a * chi_old, chi_new, chi_old);
                for (int sp = 0; sp < 2; ++sp) {
                    Cx w = Wab(s, sp);
                    if (std::abs(w) < 1e-15) continue;
                    // A_sp: [chi_old, chi_new]
                    Mat A_sp = A.block(sp * chi_old, 0, chi_old, chi_new);
                    // contrib[m2, m2'] = w × Σ_{m1'} T1_a_s[m2, m1'] × A_sp[m1', m2']
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
//   R_new[m1, a, m1'] = Σ_{m2,b,s,sp} A[s·χ+m1, m2] W[a,b][s,sp] R[m2,b,m2'] conj(A[sp·χ+m1', m2'])
//
// Algorithm:
//   Phase 1: For each spin sp, T1_sp = conj(A_sp) @ R_flat^T  [chi_L × chi_W·chi_R_old]
//            where R_flat_T[m2', b·chi_R_old + m2] = R[m2, b, m2']
//   Phase 2: For each (a,b,s,sp) with W[a,b][s,sp] ≠ 0,
//            R_new[a] += w × A_s @ T1_sp_block(b)ᵀ
static Env update_right(const Env& R, const Mat& A, const MPOSite& W) {
    const int chi_r = A.cols();       // MPS right bond
    const int chi_l = A.rows() / 2;  // MPS left bond (new env bond)
    const int chi_wl = W.chi_l;
    const int chi_wr = W.chi_r;
    Env R_new(chi_l, chi_wl, chi_l);  // zero-initialised

    // R_flat_T[m2', b·chi_r + m2] = R.at(m2, b, m2')
    Mat R_flat_T(chi_r, chi_wr * chi_r);
    for (int m2 = 0; m2 < chi_r; ++m2)
        for (int b = 0; b < chi_wr; ++b)
            for (int m2p = 0; m2p < chi_r; ++m2p)
                R_flat_T(m2p, b * chi_r + m2) = R.at(m2, b, m2p);

    // Phase 1: T1_sp[m1', b·chi_r + m2] = Σ_{m2'} conj(A_sp[m1', m2']) × R_flat_T[m2', b·chi_r+m2]
    std::array<Mat, 2> T1;
    for (int sp = 0; sp < 2; ++sp) {
        Mat A_sp_conj = A.block(sp * chi_l, 0, chi_l, chi_r).conjugate();
        T1[sp] = A_sp_conj * R_flat_T;  // [chi_l, chi_wr · chi_r]
    }

    // Phase 2: accumulate into R_new
    for (int a = 0; a < chi_wl; ++a) {
        for (int b = 0; b < chi_wr; ++b) {
            const Mat2& Wab = W.W(a, b);
            for (int s = 0; s < 2; ++s) {
                Mat A_s = A.block(s * chi_l, 0, chi_l, chi_r);
                for (int sp = 0; sp < 2; ++sp) {
                    Cx w = Wab(s, sp);
                    if (std::abs(w) < 1e-15) continue;
                    // T1_sp block for this b: [chi_l, chi_r]
                    Mat T1_sp_b = T1[sp].block(0, b * chi_r, chi_l, chi_r);
                    // contrib[m1, m1'] = w × Σ_{m2} A_s[m1, m2] × T1_sp_b[m1', m2]
                    //                  = w × A_s @ T1_sp_b.T
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

// ── Effective Hamiltonian ────────────────────────────────────────────────────
//
// Applies H_eff to the two-site tensor theta:
//   (H_eff |θ⟩)[m1', s1', s2', m2'] =
//     Σ_{m1,m2,a,b,c,s1,s2} L[m1,a,m1'] Wi[a,b][s1',s1] Wj[b,c][s2',s2] R[m2,c,m2'] θ[m1,s1,s2,m2]
//
// Algorithm (O(nnz_ac · χ³ · d²)):
//   Precompute W_comb[a][c][sp12, s12] = Σ_b Wi[a,b][s1p,s1] × Wj[b,c][s2p,s2]  (4×4 matrix)
//   Step 1: T1 = θ_mat @ R_mat  [chi_L·4, chi_Wj·chi_R]  O(chi²·chi_W·d²) DGEMM
//   Step 2: For each (a,c) with W_comb[a][c] ≠ 0:
//           T2_perm = W_comb[a][c] @ T1_c_perm          O(d⁴·chi²) small
//           out += L_a.T @ T2_flat                       O(chi³·d²) DGEMM  ← dominant
//   Total: O(nnz_ac · chi³ · d²) ≈ O(8 · chi³ · 4) for Heisenberg
static Vec apply_heff(const Env& L, const MPOSite& Wi, const MPOSite& Wj,
                      const Env& R, const Vec& theta, int chi_L, int chi_R) {
    const int d2     = 4;    // d*d = 2*2
    const int chi_Wl = Wi.chi_l;
    const int chi_Wm = Wi.chi_r;  // = Wj.chi_l
    const int chi_Wr = Wj.chi_r;

    // Precompute 2-site MPO: W_comb[a][c] is a 4×4 complex matrix
    // W_comb[a][c][s1p*2+s2p, s1*2+s2] = Σ_b Wi[a,b][s1p,s1] × Wj[b,c][s2p,s2]
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

    // Step 1: theta_mat[m1*4+phys, m2], R_mat[m2, c*chi_R+m2']
    //         T1 = theta_mat @ R_mat  → [chi_L*4, chi_Wr*chi_R]
    Mat theta_mat(chi_L * d2, chi_R);
    for (int m1 = 0; m1 < chi_L; ++m1)
        for (int p = 0; p < d2; ++p)
            for (int m2 = 0; m2 < chi_R; ++m2)
                theta_mat(m1*d2 + p, m2) = theta[m1*d2*chi_R + p*chi_R + m2];

    Mat R_mat(chi_R, chi_Wr * chi_R);
    for (int m2 = 0; m2 < chi_R; ++m2)
        for (int c = 0; c < chi_Wr; ++c)
            for (int m2p = 0; m2p < chi_R; ++m2p)
                R_mat(m2, c*chi_R + m2p) = R.at(m2, c, m2p);

    Mat T1 = theta_mat * R_mat;  // [chi_L*4, chi_Wr*chi_R]

    Vec out = Vec::Zero(chi_L * d2 * chi_R);

    // Step 2: for each non-zero (a,c) pair
    for (int a = 0; a < chi_Wl; ++a) {
        // L_a[m1, m1'] = L.at(m1, a, m1')
        Mat L_a(chi_L, chi_L);
        for (int m1 = 0; m1 < chi_L; ++m1)
            for (int m1p = 0; m1p < chi_L; ++m1p)
                L_a(m1, m1p) = L.at(m1, a, m1p);

        for (int c = 0; c < chi_Wr; ++c) {
            if (W_comb[a][c].isZero(1e-15)) continue;

            // T1_c: column block for this c → [chi_L*4, chi_R]
            Mat T1_c = T1.block(0, c*chi_R, chi_L*d2, chi_R);

            // Permute T1_c to T1_c_perm[phys, m1*chi_R+m2'] for W application
            Mat T1_c_perm(d2, chi_L * chi_R);
            for (int m1 = 0; m1 < chi_L; ++m1)
                for (int p = 0; p < d2; ++p)
                    for (int m2p = 0; m2p < chi_R; ++m2p)
                        T1_c_perm(p, m1*chi_R + m2p) = T1_c(m1*d2 + p, m2p);

            // Apply W_comb[a][c]: T2_perm = W_comb[a][c] @ T1_c_perm
            Mat T2_perm = W_comb[a][c] * T1_c_perm;  // [4, chi_L*chi_R]

            // Permute back: T2_flat[m1, phys*chi_R+m2']
            Mat T2_flat(chi_L, d2 * chi_R);
            for (int m1 = 0; m1 < chi_L; ++m1)
                for (int p = 0; p < d2; ++p)
                    for (int m2p = 0; m2p < chi_R; ++m2p)
                        T2_flat(m1, p*chi_R + m2p) = T2_perm(p, m1*chi_R + m2p);

            // Contract with L_a: out_contrib = L_a.T @ T2_flat  [chi_L, d2*chi_R]
            Mat out_contrib = L_a.transpose() * T2_flat;  // O(chi^3) DGEMM

            for (int m1p = 0; m1p < chi_L; ++m1p)
                for (int p = 0; p < d2; ++p)
                    for (int m2p = 0; m2p < chi_R; ++m2p)
                        out[m1p*d2*chi_R + p*chi_R + m2p] += out_contrib(m1p, p*chi_R + m2p);
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
