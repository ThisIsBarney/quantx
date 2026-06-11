#include "dmrg_internal.hpp"

#include <algorithm>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qx_dmrg {

// -- Pauli matrices -----------------------------------------------------------

Mat2 pauli(char c) {
    Mat2 m;
    switch (c) {
        case 'X': m << 0,1, 1,0; break;
        case 'Y': m << 0,Cx(0,-1), Cx(0,1),0; break;
        case 'Z': m << 1,0, 0,-1; break;
        default:  m = Mat2::Identity(); break;
    }
    return m;
}

// -- MPO ----------------------------------------------------------------------

// Build Heisenberg MPO: H = J sum_i (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
//
// Encoded in the QN-resolved (raising/lowering) basis so that each MPO bond index
// carries a definite ΔSz.  This is required by the Sz-conserving block-sparse path
// (run_dmrg_blocks, which assumes HB_DELTA = {0,+2,-2,0,0}); it is mathematically
// identical to the X/Y/Z form and the dense path is agnostic to the choice.
//
//   X X + Y Y = 2 (σ+ σ- + σ- σ+),   with σ+ = |↑⟩⟨↓|, σ- = |↓⟩⟨↑|
//
//  W = [ I     0      0     0    0 ]
//      [ σ-    0      0     0    0 ]   bond 1: completes 2σ+ σ-  (ΔSz = +2)
//      [ σ+    0      0     0    0 ]   bond 2: completes 2σ- σ+  (ΔSz = -2)
//      [ Z     0      0     0    0 ]   bond 3: completes Z Z     (ΔSz =  0)
//      [ 0   2J*σ+  2J*σ-  J*Z   I ]
//
// Left  boundary: row 4 (last row)
// Right boundary: col 0 (first col)
std::vector<MPOSite> build_xxz_mpo(int n, double J, double delta) {
    const int chi = 5;
    Mat2 sp; sp << 0, 1, 0, 0;   // σ+ = |↑⟩⟨↓|  (ket ↓ → bra ↑, ΔSz = +2)
    Mat2 sm; sm << 0, 0, 1, 0;   // σ- = |↓⟩⟨↑|  (ΔSz = -2)
    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        int cl = (i == 0)     ? 1 : chi;
        int cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl;
        site.chi_r = cr;
        site.ops.resize(cl * cr, Mat2::Zero());

        // Full 5x5 W matrix at bulk sites
        std::array<std::array<Mat2, 5>, 5> W5 {};
        for (auto& row : W5) for (auto& e : row) e.setZero();
        W5[0][0] = pauli('I');
        W5[1][0] = sm;
        W5[2][0] = sp;
        W5[3][0] = pauli('Z');
        W5[4][1] = (2.0 * J) * sp;            // Jxy = J  (XX + YY)
        W5[4][2] = (2.0 * J) * sm;
        W5[4][3] = (J * delta) * pauli('Z');  // Jz  = J*delta  (delta ZZ)
        W5[4][4] = pauli('I');

        // Map cl x cr indices into the 5x5 matrix
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

// Spin-current MPO for the XXZ chain: J_tot = sum_l i*J (sigma+_l sigma-_{l+1} -
// sigma-_l sigma+_{l+1}) (the Z term carries no spin current). Hermitian; same Sz
// bond structure as the Heisenberg MPO, so the block engine applies it unchanged.
std::vector<MPOSite> build_current_mpo(int n, double J) {
    const int chi = 5;
    Mat2 sp; sp << 0, 1, 0, 0;
    Mat2 sm; sm << 0, 0, 1, 0;
    Cx iJ(0.0, J);
    std::array<std::array<Mat2, 5>, 5> W5{};
    for (auto& row : W5) for (auto& e : row) e.setZero();
    W5[0][0] = pauli('I');
    W5[1][0] = sm;                 // complete  i*J sigma+_l sigma-_{l+1}
    W5[2][0] = sp;                 // complete -i*J sigma-_l sigma+_{l+1}
    W5[4][1] = iJ * sp;            // emit sigma+ (bond 1, DeltaSz +2)
    W5[4][2] = (-iJ) * sm;         // emit sigma- (bond 2, DeltaSz -2)
    W5[4][4] = pauli('I');
    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        int cl = (i == 0) ? 1 : chi, cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        site.ops.assign(cl * cr, Mat2::Zero());
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 4 : a, br = (i == n - 1) ? 0 : b;
                site.W(a, b) = W5[ar][br];
            }
    }
    return mpo;
}

// XXZ MPO on the interleaved purified chain (2*n_phys sites): even sites carry the
// physical XXZ couplings, odd (ancilla) sites are identity pass-throughs on every
// MPO bond state, so an "in-progress" term emitted at physical site 2i hops over
// ancilla 2i+1 and completes at physical site 2i+2. The identity is Sz-diagonal
// and keeps each bond state's HB_DELTA, so the block-sparse engine (environments,
// H_eff, energies) runs on the purified chain completely unchanged.
//
// Note the last *physical* site (2*n_phys-2) still uses the full bulk W (it can
// "emit" σ±/Z to its right), but those in-progress states die at the right
// boundary (col 0 of the final ancilla's identity), so H is exactly
//   J sum_{i<n_phys-1} [ 2(σ+_{2i}σ-_{2i+2} + h.c.) + delta Z_{2i} Z_{2i+2} ].
std::vector<MPOSite> build_xxz_purified_mpo(int n_phys, double J, double delta) {
    const int chi = 5, n = 2 * n_phys;
    Mat2 sp; sp << 0, 1, 0, 0;   // σ+ = |↑⟩⟨↓|  (ΔSz = +2)
    Mat2 sm; sm << 0, 0, 1, 0;   // σ- = |↓⟩⟨↑|  (ΔSz = -2)

    // Physical-site W (same as build_xxz_mpo bulk).
    std::array<std::array<Mat2, 5>, 5> Wp{};
    for (auto& row : Wp) for (auto& e : row) e.setZero();
    Wp[0][0] = pauli('I');
    Wp[1][0] = sm;
    Wp[2][0] = sp;
    Wp[3][0] = pauli('Z');
    Wp[4][1] = (2.0 * J) * sp;
    Wp[4][2] = (2.0 * J) * sm;
    Wp[4][3] = (J * delta) * pauli('Z');
    Wp[4][4] = pauli('I');
    // Ancilla-site W: diagonal identity (pass every bond state through).
    std::array<std::array<Mat2, 5>, 5> Wa{};
    for (auto& row : Wa) for (auto& e : row) e.setZero();
    for (int b = 0; b < chi; ++b) Wa[b][b] = pauli('I');

    std::vector<MPOSite> mpo(n);
    for (int i = 0; i < n; ++i) {
        const auto& W5 = (i % 2 == 0) ? Wp : Wa;
        int cl = (i == 0) ? 1 : chi, cr = (i == n - 1) ? 1 : chi;
        auto& site = mpo[i];
        site.chi_l = cl; site.chi_r = cr;
        site.ops.assign(cl * cr, Mat2::Zero());
        for (int a = 0; a < cl; ++a)
            for (int b = 0; b < cr; ++b) {
                int ar = (i == 0) ? 4 : a, br = (i == n - 1) ? 0 : b;
                site.W(a, b) = W5[ar][br];
            }
    }
    return mpo;
}

// -- Pauli label parser -------------------------------------------------------

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
// The MPO encodes H = sum_k c_k P^k_0 x P^k_1 x ... x P^k_{n-1}.
std::vector<MPOSite> build_pauli_sum_mpo(
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
            if (i == 0) {
                if (n == 1) W[0][full_chi - 1] = coeffs[k] * pauli(p);
            }
        }

        // Extract slice
        for (int a = 0; a < cl; ++a) {
            for (int b = 0; b < cr; ++b) {
                int ar2 = (i == 0) ? 0              : a;
                int br2 = (i == n-1) ? (full_chi-1) : b;
                site.W(a, b) = W[ar2][br2];
            }
        }
    }
    return mpo;
}

} // namespace qx_dmrg
