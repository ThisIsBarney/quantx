/**
 * @file lindblad_sparse.cpp
 * @brief Matrix-free sparse Lindblad solver (M173).
 *
 * Solves the Lindblad master equation
 *
 *   dρ/dt = −i[H,ρ] + Σ_k (L_k ρ L_k† − ½{L_k†L_k, ρ})
 *
 * without ever constructing the 4^n × 4^n Liouvillian superoperator.
 * Instead, each RHS evaluation is done via sparse Pauli-string operations
 * directly on the d×d density matrix ρ (d = 2^n).
 *
 * Memory usage: O(4^n) complex<double> = O(d²).
 *   n=10 →  8 MB     n=12 →  128 MB    n=14 →  2 GB    n=16 → 32 GB
 *
 * Time evolution: 4th-order Runge-Kutta with Hermitian re-symmetrisation.
 * Steady state  : column-replacement GMRES via Eigen's BiCGSTAB on the
 *                 sparse Liouvillian (built explicitly, feasible for n≤12).
 *
 * C ABI entry points (declared in include/qx/c_api.h):
 *   qx_lindblad_sparse_evolve(problem_json, options_json)  → result_json
 *   qx_lindblad_sparse_steady(problem_json)                → result_json
 *
 * References
 * ----------
 * [1] Lindblad (1976). Commun. Math. Phys. 48, 119.
 * [2] Johansson et al. (2012). QuTiP. Comput. Phys. Commun. 183, 1760.
 * [3] Breuer & Petruccione (2002). Open Quantum Systems. OUP.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <nlohmann/json.hpp>

#include "qx/c_api.h"
#include "qx/json_utils.hpp"
#include "qx/logging.hpp"

using json = nlohmann::json;

namespace qx {
namespace lindblad_sparse {

// ── Types ───────────────────────────────────────────────────────────────────

using Complex   = std::complex<double>;
using MatrixXcd = Eigen::MatrixXcd;
using VectorXcd = Eigen::VectorXcd;
using SparseMatrixXcd = Eigen::SparseMatrix<Complex, Eigen::ColMajor>;
using Triplet = Eigen::Triplet<Complex>;

static constexpr std::size_t kMaxQubitsMatFree = 16U;   // memory limit ~32 GB
static constexpr std::size_t kMaxQubitsForSteady = 12U; // explicit L: ~128 MB ρ

// ── Single-qubit sparse gate matrices ───────────────────────────────────────

SparseMatrixXcd make_pauli(char p) {
    SparseMatrixXcd m(2, 2);
    switch (p) {
        case 'X': m.insert(0,1) = Complex{1,0}; m.insert(1,0) = Complex{1,0}; break;
        case 'Y': m.insert(0,1) = Complex{0,-1}; m.insert(1,0) = Complex{0,1}; break;
        case 'Z': m.insert(0,0) = Complex{1,0};  m.insert(1,1) = Complex{-1,0}; break;
        default:  m.insert(0,0) = Complex{1,0};  m.insert(1,1) = Complex{1,0}; break; // I
    }
    m.makeCompressed();
    return m;
}

// Compute sparse Kronecker product A ⊗ B
SparseMatrixXcd kron(const SparseMatrixXcd& A, const SparseMatrixXcd& B) {
    const int mA = static_cast<int>(A.rows()), nA = static_cast<int>(A.cols());
    const int mB = static_cast<int>(B.rows()), nB = static_cast<int>(B.cols());
    SparseMatrixXcd C(mA * mB, nA * nB);
    std::vector<Triplet> trips;
    trips.reserve(static_cast<std::size_t>(A.nonZeros() * B.nonZeros()));
    for (int kA = 0; kA < A.outerSize(); ++kA) {
        for (SparseMatrixXcd::InnerIterator itA(A, kA); itA; ++itA) {
            for (int kB = 0; kB < B.outerSize(); ++kB) {
                for (SparseMatrixXcd::InnerIterator itB(B, kB); itB; ++itB) {
                    trips.emplace_back(
                        itA.row() * mB + itB.row(),
                        itA.col() * nB + itB.col(),
                        itA.value() * itB.value());
                }
            }
        }
    }
    C.setFromTriplets(trips.begin(), trips.end());
    return C;
}

// Build embedded 2^n × 2^n sparse matrix for single-qubit gate on qubit q.
// Qubit 0 = leftmost tensor factor (most significant bit convention).
SparseMatrixXcd embed(const SparseMatrixXcd& gate, int q, int n) {
    SparseMatrixXcd result(1, 1);
    result.insert(0, 0) = Complex{1, 0};
    result.makeCompressed();
    for (int i = 0; i < n; ++i) {
        SparseMatrixXcd factor = (i == q) ? gate : make_pauli('I');
        result = kron(result, factor);
    }
    return result;
}

// Build full n-qubit sparse Pauli-string matrix.
// pauli_str: e.g. "XZI" → X on qubit 0, Z on qubit 1, I on qubit 2.
SparseMatrixXcd build_pauli_string(const std::string& pauli_str, int n) {
    SparseMatrixXcd result(1, 1);
    result.insert(0, 0) = Complex{1, 0};
    result.makeCompressed();
    for (int i = 0; i < n; ++i) {
        char p = (i < static_cast<int>(pauli_str.size())) ? pauli_str[i] : 'I';
        result = kron(result, make_pauli(p));
    }
    return result;
}

// ── Hamiltonian and jump operator descriptions ───────────────────────────────

struct HamTerm {
    double coeff;
    std::string pauli;  // e.g. "XZI"
};

struct JumpOp {
    std::string type;   // "amplitude_damping" | "dephasing" | "depolarizing" | "matrix"
    int qubit;
    double rate;
    // For custom matrix jump operators (optional)
    std::vector<std::vector<Complex>> matrix_2x2;
};

// Build the n-qubit sparse matrix for a jump operator
SparseMatrixXcd build_jump_matrix(const JumpOp& op, int n) {
    SparseMatrixXcd L2(2, 2);
    const double g = std::sqrt(op.rate);

    if (op.type == "amplitude_damping") {
        // L = √γ σ⁻ = √γ |0⟩⟨1|
        L2.insert(0, 1) = Complex{g, 0};
    } else if (op.type == "dephasing") {
        // L = √(γ/2) σz
        L2.insert(0, 0) = Complex{g / std::sqrt(2.0), 0};
        L2.insert(1, 1) = Complex{-g / std::sqrt(2.0), 0};
    } else if (op.type == "depolarizing_x") {
        L2.insert(0, 1) = Complex{g / std::sqrt(3.0), 0};
        L2.insert(1, 0) = Complex{g / std::sqrt(3.0), 0};
    } else if (op.type == "depolarizing_y") {
        L2.insert(0, 1) = Complex{0, -g / std::sqrt(3.0)};
        L2.insert(1, 0) = Complex{0,  g / std::sqrt(3.0)};
    } else if (op.type == "depolarizing_z") {
        L2.insert(0, 0) = Complex{g / std::sqrt(3.0), 0};
        L2.insert(1, 1) = Complex{-g / std::sqrt(3.0), 0};
    } else {
        throw std::invalid_argument("Unknown jump operator type: " + op.type);
    }
    L2.makeCompressed();
    return embed(L2, op.qubit, n);
}

// ── Matrix-free RHS evaluation ───────────────────────────────────────────────
// Computes L(ρ) = -i[H,ρ] + Σ_k D_k(ρ) directly in O(m*d² + k*d²) without
// forming the 4^n × 4^n Liouvillian.

// Apply a sparse matrix M (d×d) to a dense matrix ρ (d×d) from the left: M*ρ
MatrixXcd apply_left(const SparseMatrixXcd& M, const MatrixXcd& rho) {
    return M * rho;
}

// Apply from the right: ρ*M
MatrixXcd apply_right(const MatrixXcd& rho, const SparseMatrixXcd& M) {
    return rho * M;
}

// Hamiltonian commutator contribution: -i[H,ρ] = -i(Hρ - ρH)
MatrixXcd ham_commutator(const std::vector<HamTerm>& terms,
                          const std::vector<SparseMatrixXcd>& Hmats,
                          const MatrixXcd& rho) {
    MatrixXcd result = MatrixXcd::Zero(rho.rows(), rho.cols());
    const Complex mi{0.0, -1.0};
    for (std::size_t k = 0; k < terms.size(); ++k) {
        const Complex c{terms[k].coeff, 0.0};
        const SparseMatrixXcd& P = Hmats[k];
        result += mi * c * (apply_left(P, rho) - apply_right(rho, P));
    }
    return result;
}

// Single jump operator dissipator: L ρ L† - ½(L†L ρ + ρ L†L)
MatrixXcd dissipator(const SparseMatrixXcd& L, const MatrixXcd& rho) {
    const SparseMatrixXcd Ldag = L.adjoint();
    const SparseMatrixXcd LdagL = (Ldag * L).eval();
    return apply_left(L, apply_right(rho, Ldag))
         - 0.5 * (apply_left(LdagL, rho) + apply_right(rho, LdagL));
}

// Full Lindblad RHS: dρ/dt = -i[H,ρ] + Σ_k D_k(ρ)
MatrixXcd lindblad_rhs(const std::vector<HamTerm>& ham,
                        const std::vector<SparseMatrixXcd>& Hmats,
                        const std::vector<SparseMatrixXcd>& Lops,
                        const MatrixXcd& rho) {
    MatrixXcd result = ham_commutator(ham, Hmats, rho);
    for (const auto& L : Lops) {
        result += dissipator(L, rho);
    }
    return result;
}

// ── RK4 time stepper ─────────────────────────────────────────────────────────

MatrixXcd rk4_step(const std::vector<HamTerm>& ham,
                   const std::vector<SparseMatrixXcd>& Hmats,
                   const std::vector<SparseMatrixXcd>& Lops,
                   const MatrixXcd& rho, double dt) {
    auto rhs = [&](const MatrixXcd& r) {
        return lindblad_rhs(ham, Hmats, Lops, r);
    };
    MatrixXcd k1 = rhs(rho);
    MatrixXcd k2 = rhs(rho + 0.5 * dt * k1);
    MatrixXcd k3 = rhs(rho + 0.5 * dt * k2);
    MatrixXcd k4 = rhs(rho + dt * k3);
    MatrixXcd rho_new = rho + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
    // Re-symmetrise to prevent drift from breaking Hermiticity
    rho_new = (rho_new + rho_new.adjoint()) * 0.5;
    return rho_new;
}

// ── Sparse Liouvillian for steady-state ──────────────────────────────────────
// Builds the 4^n × 4^n sparse Liouvillian L in vec(ρ) space.
// Used only for steady-state solver (n ≤ kMaxQubitsForSteady).

SparseMatrixXcd build_liouvillian(const std::vector<HamTerm>& ham,
                                   const std::vector<SparseMatrixXcd>& Hmats,
                                   const std::vector<SparseMatrixXcd>& Lops,
                                   int n) {
    const int d = 1 << n;
    const int d2 = d * d;

    // Identity matrix d×d (sparse)
    SparseMatrixXcd Id(d, d);
    Id.setIdentity();

    std::vector<Triplet> trips;
    trips.reserve(static_cast<std::size_t>(d2) * 4ULL);  // rough estimate

    // Helper: add contribution of a d²×d² sparse matrix to trips
    auto add_contribution = [&](const SparseMatrixXcd& M, Complex scale) {
        for (int k = 0; k < M.outerSize(); ++k) {
            for (SparseMatrixXcd::InnerIterator it(M, k); it; ++it) {
                trips.emplace_back(static_cast<int>(it.row()),
                                   static_cast<int>(it.col()),
                                   scale * it.value());
            }
        }
    };

    // Hamiltonian part: -i(I⊗H - H^T⊗I)
    // = -i Σ_k h_k (I⊗P_k - P_k^T⊗I)
    const Complex mi{0.0, -1.0};
    for (std::size_t k = 0; k < ham.size(); ++k) {
        const Complex c = mi * Complex{ham[k].coeff, 0.0};
        const SparseMatrixXcd& P  = Hmats[k];
        const SparseMatrixXcd  Pt = P.transpose();
        // I⊗P contribution (block diagonal: each block is P)
        add_contribution(kron(Id, P),  c);
        // P^T⊗I contribution (block structure: A_{ij}*I at block position ij)
        add_contribution(kron(Pt, Id), -c);
    }

    // Jump operator dissipator parts: L^*⊗L - ½(I⊗L†L) - ½((L†L)^T⊗I)
    for (const auto& L : Lops) {
        const SparseMatrixXcd Ldag    = L.adjoint();
        const SparseMatrixXcd LdagL   = (Ldag * L).eval();
        const SparseMatrixXcd Lconj   = L.conjugate();
        const SparseMatrixXcd LdagLt  = LdagL.transpose();

        add_contribution(kron(Lconj, L),    Complex{1.0, 0.0});
        add_contribution(kron(Id, LdagL),   Complex{-0.5, 0.0});
        add_contribution(kron(LdagLt, Id),  Complex{-0.5, 0.0});
    }

    SparseMatrixXcd Lmat(d2, d2);
    Lmat.setFromTriplets(trips.begin(), trips.end());
    return Lmat;
}

// Solve for steady state using trace-normalised linear system.
// Replaces last row of L with trace constraint: Σ_i ρ_{ii} = 1.
MatrixXcd solve_steady_state(const std::vector<HamTerm>& ham,
                              const std::vector<SparseMatrixXcd>& Hmats,
                              const std::vector<SparseMatrixXcd>& Lops,
                              int n) {
    const int d  = 1 << n;
    const int d2 = d * d;

    SparseMatrixXcd L = build_liouvillian(ham, Hmats, Lops, n);

    // Build trace-constraint row: coefficient 1 for each diagonal element ρ_{ii}
    // ρ_{ii} = vec(ρ)[i*d + i] in row-major vectorisation
    // We replace the last row with this constraint.
    std::vector<Triplet> trips;
    for (int k = 0; k < L.outerSize(); ++k) {
        for (SparseMatrixXcd::InnerIterator it(L, k); it; ++it) {
            if (static_cast<int>(it.row()) != d2 - 1) {
                trips.emplace_back(static_cast<int>(it.row()),
                                   static_cast<int>(it.col()),
                                   it.value());
            }
        }
    }
    // Trace constraint: vec(ρ)[i*d+i] for i=0..d-1
    for (int i = 0; i < d; ++i) {
        trips.emplace_back(d2 - 1, i * d + i, Complex{1.0, 0.0});
    }

    SparseMatrixXcd Lmod(d2, d2);
    Lmod.setFromTriplets(trips.begin(), trips.end());

    // RHS: all zeros except last element = 1 (trace = 1)
    VectorXcd rhs = VectorXcd::Zero(d2);
    rhs(d2 - 1) = Complex{1.0, 0.0};

    // Solve with BiCGSTAB (memory-efficient iterative solver)
    Eigen::BiCGSTAB<SparseMatrixXcd> solver;
    solver.setTolerance(1e-12);
    solver.setMaxIterations(10000);
    solver.compute(Lmod);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("BiCGSTAB: matrix analysis failed");
    }
    VectorXcd rho_vec = solver.solve(rhs);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("BiCGSTAB: solve failed (not converged). "
                                 "Residual: " +
                                 std::to_string(solver.error()));
    }

    // Reshape d²-vector back to d×d density matrix (row-major)
    MatrixXcd rho(d, d);
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            rho(i, j) = rho_vec(i * d + j);
        }
    }
    // Re-symmetrise for numerical cleanliness
    rho = (rho + rho.adjoint()) * 0.5;
    return rho;
}

// ── JSON parse helpers ───────────────────────────────────────────────────────

std::vector<HamTerm> parse_ham(const json& j) {
    std::vector<HamTerm> terms;
    if (!j.is_array()) return terms;
    for (const auto& t : j) {
        HamTerm h;
        h.coeff = t.value("coeff", t.value("coefficient", 0.0));
        h.pauli = t.value("pauli", t.value("label", std::string("I")));
        terms.push_back(h);
    }
    return terms;
}

std::vector<JumpOp> parse_jumps(const json& j, int n) {
    std::vector<JumpOp> ops;
    if (!j.is_array()) return ops;
    for (const auto& item : j) {
        std::string type = item.value("type", "amplitude_damping");
        double rate      = item.value("rate", 0.0);
        int qubit        = item.value("qubit", 0);

        if (type == "depolarizing") {
            // Expand into three jump operators
            for (const std::string& sub : {"depolarizing_x", "depolarizing_y", "depolarizing_z"}) {
                ops.push_back({sub, qubit, rate, {}});
            }
        } else {
            ops.push_back({type, qubit, rate, {}});
        }
        (void)n;  // may be used for validation in future
    }
    return ops;
}

// ── Result serialisation ─────────────────────────────────────────────────────

double compute_purity(const MatrixXcd& rho) {
    return std::abs((rho * rho).trace().real());
}

double compute_entropy(const MatrixXcd& rho) {
    Eigen::SelfAdjointEigenSolver<MatrixXcd> es(rho, Eigen::EigenvaluesOnly);
    double s = 0.0;
    for (int i = 0; i < es.eigenvalues().size(); ++i) {
        double lam = es.eigenvalues()(i);
        if (lam > 1e-15) s -= lam * std::log2(lam);
    }
    return s;
}

json rho_to_json(const MatrixXcd& rho) {
    int d = static_cast<int>(rho.rows());
    json j;
    j["purity"]   = compute_purity(rho);
    j["entropy"]  = compute_entropy(rho);
    json pops = json::array();
    for (int i = 0; i < d; ++i) pops.push_back(rho(i,i).real());
    j["populations"] = pops;
    return j;
}

// ── Public C++ entry points (called by C ABI) ────────────────────────────────

std::string run_evolve(const std::string& problem_json,
                        const std::string& options_json) {
    json prob = json::parse(problem_json);
    json opts = json::parse(options_json.empty() ? "{}" : options_json);

    const int n          = prob.value("n_qubits", prob.value("qubits", 1));
    const double dt      = prob.value("dt", opts.value("dt", 0.01));
    const double t_total = prob.value("total_time", opts.value("total_time", 1.0));
    const int n_steps    = static_cast<int>(std::max(1.0, std::round(t_total / dt)));

    if (n < 1 || static_cast<std::size_t>(n) > kMaxQubitsMatFree) {
        return json{{"error", "n_qubits must be in [1, " +
                               std::to_string(kMaxQubitsMatFree) + "]"}}.dump();
    }

    const int d = 1 << n;

    // Parse Hamiltonian and jump operators
    auto ham   = parse_ham(prob.value("hamiltonian_terms", json::array()));
    auto jumps = parse_jumps(prob.value("jump_operators", json::array()), n);

    // Pre-build sparse matrices
    std::vector<SparseMatrixXcd> Hmats;
    Hmats.reserve(ham.size());
    for (const auto& h : ham) {
        Hmats.push_back(build_pauli_string(h.pauli, n));
    }
    std::vector<SparseMatrixXcd> Lops;
    Lops.reserve(jumps.size());
    for (const auto& jop : jumps) {
        Lops.push_back(build_jump_matrix(jop, n));
    }

    // Initialise ρ = |0⟩⟨0|
    MatrixXcd rho = MatrixXcd::Zero(d, d);
    rho(0, 0) = Complex{1.0, 0.0};

    // Time evolve
    json frames = json::array();
    auto record = [&](int step, double t) {
        json f = rho_to_json(rho);
        f["step"] = step;
        f["time"] = t;
        frames.push_back(f);
    };
    record(0, 0.0);
    for (int s = 0; s < n_steps; ++s) {
        rho = rk4_step(ham, Hmats, Lops, rho, dt);
        record(s + 1, (s + 1) * dt);
    }

    json result;
    result["backend"]      = "lindblad_sparse";
    result["n_qubits"]     = n;
    result["n_steps"]      = n_steps;
    result["lindblad_frames"] = frames;
    result["final_purity"] = compute_purity(rho);
    result["final_entropy"]= compute_entropy(rho);
    return result.dump();
}

std::string run_steady(const std::string& problem_json) {
    json prob = json::parse(problem_json);
    const int n = prob.value("n_qubits", prob.value("qubits", 1));

    if (n < 1 || static_cast<std::size_t>(n) > kMaxQubitsForSteady) {
        return json{{"error", "Steady-state solver supports n_qubits ≤ " +
                               std::to_string(kMaxQubitsForSteady)}}.dump();
    }

    auto ham   = parse_ham(prob.value("hamiltonian_terms", json::array()));
    auto jumps = parse_jumps(prob.value("jump_operators", json::array()), n);

    std::vector<SparseMatrixXcd> Hmats;
    for (const auto& h : ham) {
        Hmats.push_back(build_pauli_string(h.pauli, n));
    }
    std::vector<SparseMatrixXcd> Lops;
    for (const auto& jop : jumps) {
        Lops.push_back(build_jump_matrix(jop, n));
    }

    MatrixXcd rho_ss = solve_steady_state(ham, Hmats, Lops, n);

    json result;
    result["backend"]     = "lindblad_sparse";
    result["n_qubits"]    = n;
    result["purity"]      = compute_purity(rho_ss);
    result["entropy"]     = compute_entropy(rho_ss);
    json pops = json::array();
    for (int i = 0; i < static_cast<int>(rho_ss.rows()); ++i) {
        pops.push_back(rho_ss(i,i).real());
    }
    result["populations"] = pops;
    return result.dump();
}

}  // namespace lindblad_sparse
}  // namespace qx

// ── C ABI ────────────────────────────────────────────────────────────────────

extern "C" {

QX_API const char* qx_lindblad_sparse_evolve(const char* problem_json,
                                               const char* options_json) {
    try {
        static thread_local std::string buf;
        buf = qx::lindblad_sparse::run_evolve(
            problem_json ? problem_json : "{}",
            options_json ? options_json : "{}");
        return buf.c_str();
    } catch (const std::exception& e) {
        static thread_local std::string err;
        err = nlohmann::json{{"error", e.what()}}.dump();
        return err.c_str();
    }
}

QX_API const char* qx_lindblad_sparse_steady(const char* problem_json) {
    try {
        static thread_local std::string buf;
        buf = qx::lindblad_sparse::run_steady(
            problem_json ? problem_json : "{}");
        return buf.c_str();
    } catch (const std::exception& e) {
        static thread_local std::string err;
        err = nlohmann::json{{"error", e.what()}}.dump();
        return err.c_str();
    }
}

}  // extern "C"
