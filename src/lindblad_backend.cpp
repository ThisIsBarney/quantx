#include "qx/backends.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <limits>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace qx {
namespace {

using Complex = std::complex<double>;
using MatrixXcd = Eigen::MatrixXcd;

constexpr std::uint32_t kMaxLindbladQubits = 10U;

std::size_t lindblad_dimension(std::uint32_t qubits) {
  if (qubits >= std::numeric_limits<std::size_t>::digits) {
    throw std::invalid_argument(
        "Lindblad backend does not support qubit counts >= " +
        std::to_string(std::numeric_limits<std::size_t>::digits) +
        " due to dimension overflow risk.");
  }
  if (qubits > kMaxLindbladQubits) {
    throw std::invalid_argument(
        "Lindblad backend supports at most " +
        std::to_string(kMaxLindbladQubits) +
        " qubits to avoid excessive dense matrix allocations.");
  }
  return static_cast<std::size_t>(1ULL) << qubits;
}

// ── Single-qubit Pauli matrices ─────────────────────────────────────────────

MatrixXcd pauli_x() {
  MatrixXcd m(2, 2);
  m << Complex{0.0, 0.0}, Complex{1.0, 0.0},
       Complex{1.0, 0.0}, Complex{0.0, 0.0};
  return m;
}

MatrixXcd pauli_y() {
  MatrixXcd m(2, 2);
  m << Complex{0.0, 0.0}, Complex{0.0, -1.0},
       Complex{0.0, 1.0}, Complex{0.0, 0.0};
  return m;
}

MatrixXcd pauli_z() {
  MatrixXcd m(2, 2);
  m << Complex{1.0, 0.0}, Complex{0.0, 0.0},
       Complex{0.0, 0.0}, Complex{-1.0, 0.0};
  return m;
}

// σ⁻ = |0⟩⟨1| (amplitude damping lowering operator)
MatrixXcd sigma_minus() {
  MatrixXcd m(2, 2);
  m << Complex{0.0, 0.0}, Complex{1.0, 0.0},
       Complex{0.0, 0.0}, Complex{0.0, 0.0};
  return m;
}

// ── Embed a 2×2 single-qubit operator in a 2^n × 2^n Hilbert space ─────────
// Qubit numbering: qubit 0 = least significant bit of basis index.

MatrixXcd embed_operator(const MatrixXcd& op2x2, std::uint32_t q, std::uint32_t n) {
  const std::size_t dim = lindblad_dimension(n);
  const std::size_t q_mask = static_cast<std::size_t>(1ULL) << q;
  MatrixXcd result = MatrixXcd::Zero(static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));
  for (std::size_t col = 0; col < dim; ++col) {
    const Eigen::Index c_q = static_cast<Eigen::Index>((col >> q) & 1U);
    const std::size_t col_rest = col & ~q_mask;
    for (std::size_t row = 0; row < dim; ++row) {
      if ((row & ~q_mask) != col_rest) {
        continue;
      }
      const Eigen::Index r_q = static_cast<Eigen::Index>((row >> q) & 1U);
      result(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) = op2x2(r_q, c_q);
    }
  }
  return result;
}

// ── Build Hamiltonian from Pauli-string HamiltonianTerms ─────────────────────
// Ignores collapse-operator labels (AmplitudeDamping*, Dephasing*, Depolarizing*).

MatrixXcd build_hamiltonian(const std::vector<HamiltonianTerm>& terms, std::uint32_t n) {
  const std::size_t dim = lindblad_dimension(n);
  MatrixXcd H = MatrixXcd::Zero(static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));

  const std::regex token_re("([XYZxyz])(\\d+)");
  for (const auto& term : terms) {
    const std::string& label = term.label;
    // Skip collapse-operator labels
    if (label.rfind("AmplitudeDamping", 0) == 0 ||
        label.rfind("Dephasing", 0) == 0 ||
        label.rfind("Depolarizing", 0) == 0) {
      continue;
    }
    if (label.empty()) {
      continue;
    }

    std::size_t flip_mask = 0;
    std::size_t yz_mask = 0;
    std::size_t y_count = 0;
    bool valid = true;

    auto begin = std::sregex_iterator(label.begin(), label.end(), token_re);
    const auto end = std::sregex_iterator();
    for (auto it = begin; it != end && valid; ++it) {
      const char p = static_cast<char>(std::toupper(static_cast<unsigned char>((*it)[1].str()[0])));
      const auto q = static_cast<std::uint32_t>(std::stoul((*it)[2].str()));
      if (q >= n || q >= 63U) {
        valid = false;
        break;
      }
      const std::size_t m = static_cast<std::size_t>(1ULL << q);
      if (p == 'X') {
        flip_mask |= m;
      } else if (p == 'Y') {
        flip_mask |= m;
        yz_mask |= m;
        ++y_count;
      } else if (p == 'Z') {
        yz_mask |= m;
      } else {
        valid = false;
      }
    }
    if (!valid) {
      continue;
    }

    Complex global_phase;
    switch (y_count % 4U) {
      case 0: global_phase = Complex{1.0, 0.0};  break;
      case 1: global_phase = Complex{0.0, 1.0};  break;
      case 2: global_phase = Complex{-1.0, 0.0}; break;
      default: global_phase = Complex{0.0, -1.0}; break;
    }

    for (std::size_t src = 0; src < dim; ++src) {
      const std::size_t dst = src ^ flip_mask;
      const std::size_t sign_bits = src & yz_mask;
      const bool odd = (std::popcount(sign_bits) & 1U) != 0U;
      const double sign = odd ? -1.0 : 1.0;
      H(static_cast<Eigen::Index>(dst), static_cast<Eigen::Index>(src)) +=
          term.coefficient * global_phase * sign;
    }
  }
  return H;
}

// ── Collapse operator record (precomputed L, L†, L†L) ─────────────────────

struct CollapseOp {
  MatrixXcd L;
  MatrixXcd Ldag;
  MatrixXcd LdagL;
};

// Extract the qubit index from a collapse-operator label, e.g. "AmplitudeDamping2" → 2.
// Returns an empty optional when the suffix is missing or not a valid integer.
std::optional<std::uint32_t> parse_collapse_qubit(const std::string& label,
                                                    const std::string& prefix) {
  if (label.size() <= prefix.size()) {
    return std::nullopt;
  }
  try {
    return static_cast<std::uint32_t>(std::stoul(label.substr(prefix.size())));
  } catch (...) {
    return std::nullopt;
  }
}

// Parse collapse-operator entries from HamiltonianTerms.
// Label formats:
//   "AmplitudeDamping<q>"  coefficient = γ  →  L = sqrt(γ) σ⁻_q
//   "Dephasing<q>"         coefficient = γ  →  L = sqrt(γ) σz_q / 2
//   "Depolarizing<q>"      coefficient = γ  →  3 ops: sqrt(γ/3)·{σx,σy,σz}_q

std::vector<CollapseOp> build_collapse_ops(const std::vector<HamiltonianTerm>& terms,
                                            std::uint32_t n) {
  std::vector<CollapseOp> ops;

  auto push_op = [&](const MatrixXcd& L) {
    CollapseOp op;
    op.L = L;
    op.Ldag = L.adjoint();
    op.LdagL = op.Ldag * op.L;
    ops.push_back(std::move(op));
  };

  for (const auto& term : terms) {
    const std::string& label = term.label;
    const double gamma = std::max(0.0, term.coefficient);
    const double sqrt_gamma = std::sqrt(gamma);

    if (label.rfind("AmplitudeDamping", 0) == 0) {
      auto q = parse_collapse_qubit(label, "AmplitudeDamping");
      if (!q || *q >= n) {
        continue;
      }
      push_op(sqrt_gamma * embed_operator(sigma_minus(), *q, n));
    } else if (label.rfind("Dephasing", 0) == 0) {
      auto q = parse_collapse_qubit(label, "Dephasing");
      if (!q || *q >= n) {
        continue;
      }
      // L = sqrt(γ) * σz / 2
      push_op((sqrt_gamma * 0.5) * embed_operator(pauli_z(), *q, n));
    } else if (label.rfind("Depolarizing", 0) == 0) {
      auto q = parse_collapse_qubit(label, "Depolarizing");
      if (!q || *q >= n) {
        continue;
      }
      const double sqrt_g3 = std::sqrt(gamma / 3.0);
      push_op(sqrt_g3 * embed_operator(pauli_x(), *q, n));
      push_op(sqrt_g3 * embed_operator(pauli_y(), *q, n));
      push_op(sqrt_g3 * embed_operator(pauli_z(), *q, n));
    }
  }
  return ops;
}

// ── Lindblad superoperator ─────────────────────────────────────────────────

MatrixXcd lindblad_rhs(const MatrixXcd& rho,
                        const MatrixXcd& H,
                        const std::vector<CollapseOp>& ops) {
  // Coherent part: -i[H, ρ]
  MatrixXcd result = Complex{0.0, -1.0} * (H * rho - rho * H);
  // Dissipative part
  for (const auto& op : ops) {
    result += op.L * rho * op.Ldag
              - 0.5 * (op.LdagL * rho + rho * op.LdagL);
  }
  return result;
}

// RK4 integration step.
MatrixXcd rk4_step(const MatrixXcd& rho,
                    double dt,
                    const MatrixXcd& H,
                    const std::vector<CollapseOp>& ops) {
  const MatrixXcd k1 = lindblad_rhs(rho, H, ops);
  const MatrixXcd k2 = lindblad_rhs(rho + (dt * 0.5) * k1, H, ops);
  const MatrixXcd k3 = lindblad_rhs(rho + (dt * 0.5) * k2, H, ops);
  const MatrixXcd k4 = lindblad_rhs(rho + dt * k3, H, ops);
  return rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

// ── Observables ────────────────────────────────────────────────────────────

double compute_purity(const MatrixXcd& rho) {
  return std::max(0.0, (rho * rho).trace().real());
}

double compute_entropy(const MatrixXcd& rho) {
  // -Tr(ρ log ρ) via eigendecomposition
  Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(rho, Eigen::EigenvaluesOnly);
  double entropy = 0.0;
  for (Eigen::Index i = 0; i < solver.eigenvalues().size(); ++i) {
    const double lam = std::max(0.0, solver.eigenvalues()(i));
    if (lam > 1e-15) {
      entropy -= lam * std::log(lam);
    }
  }
  return std::max(0.0, entropy);
}

// ── Entanglement measures for mixed states ──────────────────────────────────

// Partial trace over subsystem B: ρ_A[ia, ia'] = Σ_{ib} ρ[ia + ib*da, ia' + ib*da].
// Qubit 0 = LSB → subsystem A spans low n_a bits, B spans high bits.
MatrixXcd partial_trace_b(const MatrixXcd& rho, std::uint32_t n_a, std::uint32_t n_b) {
  const std::size_t da = static_cast<std::size_t>(1ULL << n_a);
  const std::size_t db = static_cast<std::size_t>(1ULL << n_b);
  MatrixXcd rho_a = MatrixXcd::Zero(static_cast<Eigen::Index>(da), static_cast<Eigen::Index>(da));
  for (std::size_t ib = 0; ib < db; ++ib) {
    for (std::size_t ia = 0; ia < da; ++ia) {
      for (std::size_t iap = 0; iap < da; ++iap) {
        rho_a(static_cast<Eigen::Index>(ia), static_cast<Eigen::Index>(iap)) +=
            rho(static_cast<Eigen::Index>(ia + ib * da),
                static_cast<Eigen::Index>(iap + ib * da));
      }
    }
  }
  return rho_a;
}

// Partial trace over subsystem A: ρ_B[ib, ib'] = Σ_{ia} ρ[ia + ib*da, ia + ib'*da].
MatrixXcd partial_trace_a(const MatrixXcd& rho, std::uint32_t n_a, std::uint32_t n_b) {
  const std::size_t da = static_cast<std::size_t>(1ULL << n_a);
  const std::size_t db = static_cast<std::size_t>(1ULL << n_b);
  MatrixXcd rho_b = MatrixXcd::Zero(static_cast<Eigen::Index>(db), static_cast<Eigen::Index>(db));
  for (std::size_t ia = 0; ia < da; ++ia) {
    for (std::size_t ib = 0; ib < db; ++ib) {
      for (std::size_t ibp = 0; ibp < db; ++ibp) {
        rho_b(static_cast<Eigen::Index>(ib), static_cast<Eigen::Index>(ibp)) +=
            rho(static_cast<Eigen::Index>(ia + ib * da),
                static_cast<Eigen::Index>(ia + ibp * da));
      }
    }
  }
  return rho_b;
}

// Partial transpose with respect to A:
// ρ^{T_A}[ia + ib*da, ia' + ib'*da] = ρ[ia' + ib*da, ia + ib'*da].
MatrixXcd partial_transpose_a(const MatrixXcd& rho, std::uint32_t n_a, std::uint32_t n_b) {
  const std::size_t da = static_cast<std::size_t>(1ULL << n_a);
  const std::size_t db = static_cast<std::size_t>(1ULL << n_b);
  const std::size_t dim = da * db;
  MatrixXcd out = MatrixXcd::Zero(static_cast<Eigen::Index>(dim), static_cast<Eigen::Index>(dim));
  for (std::size_t ib = 0; ib < db; ++ib) {
    for (std::size_t ia = 0; ia < da; ++ia) {
      for (std::size_t ibp = 0; ibp < db; ++ibp) {
        for (std::size_t iap = 0; iap < da; ++iap) {
          out(static_cast<Eigen::Index>(ia + ib * da),
              static_cast<Eigen::Index>(iap + ibp * da)) =
              rho(static_cast<Eigen::Index>(iap + ib * da),
                  static_cast<Eigen::Index>(ia + ibp * da));
        }
      }
    }
  }
  return out;
}

// Wootters concurrence for a 2-qubit mixed state.
double wootters_concurrence(const MatrixXcd& rho) {
  // spin-flip basis: σ_y ⊗ σ_y (in computational basis |00⟩,|01⟩,|10⟩,|11⟩)
  // R_tilde = ρ * (σ_y⊗σ_y) * ρ* * (σ_y⊗σ_y)
  static const Complex sy_sy_data[16] = {
      {0,0},{0,0},{0,0},{-1,0},
      {0,0},{0,0},{1,0},{0,0},
      {0,0},{1,0},{0,0},{0,0},
      {-1,0},{0,0},{0,0},{0,0}
  };
  MatrixXcd sy_sy = Eigen::Map<const MatrixXcd>(sy_sy_data, 4, 4).transpose();
  const MatrixXcd rho_tilde = sy_sy * rho.conjugate() * sy_sy;
  // R = ρ * rho_tilde, eigenvalues of R are sqrt'd in descending order
  const MatrixXcd R = rho * rho_tilde;
  // Use ComplexEigenSolver; eigenvalues may have small imaginary parts
  Eigen::ComplexEigenSolver<MatrixXcd> es(R, false);
  Eigen::VectorXd lambdas(4);
  for (Eigen::Index i = 0; i < 4; ++i) {
    lambdas(i) = std::sqrt(std::max(0.0, es.eigenvalues()(i).real()));
  }
  std::sort(lambdas.data(), lambdas.data() + 4, std::greater<double>());
  return std::max(0.0, lambdas(0) - lambdas(1) - lambdas(2) - lambdas(3));
}

// Compute bipartite entanglement measures from a Lindblad density matrix.
EntanglementFrame compute_entanglement_mixed(const MatrixXcd& rho,
                                              std::uint32_t n_a, std::uint32_t n,
                                              std::uint32_t step, double time) {
  EntanglementFrame frame;
  frame.step = step;
  frame.time = time;
  if (n_a == 0 || n_a >= n) return frame;

  const std::uint32_t n_b = n - n_a;
  frame.n_a = n_a;
  frame.n_b = n_b;

  // Partial traces
  const MatrixXcd rho_a = partial_trace_b(rho, n_a, n_b);
  const MatrixXcd rho_b = partial_trace_a(rho, n_a, n_b);

  frame.entropy_a = compute_entropy(rho_a);
  const double entropy_b = compute_entropy(rho_b);
  const double entropy_total = compute_entropy(rho);

  // Mutual information I(A:B) = S_A + S_B - S_total
  frame.mutual_info = frame.entropy_a + entropy_b - entropy_total;

  // Log-negativity: E_N = log2(||ρ^{T_A}||_1)
  const MatrixXcd rho_pt = partial_transpose_a(rho, n_a, n_b);
  Eigen::SelfAdjointEigenSolver<MatrixXcd> solver(rho_pt, Eigen::EigenvaluesOnly);
  double trace_norm = 0.0;
  for (Eigen::Index i = 0; i < solver.eigenvalues().size(); ++i) {
    trace_norm += std::abs(solver.eigenvalues()(i));
  }
  frame.log_negativity = std::log2(std::max(1.0, trace_norm));

  // Schmidt values from eigenvalues of ρ_A (square roots of eigenvalues)
  Eigen::SelfAdjointEigenSolver<MatrixXcd> sa(rho_a, Eigen::EigenvaluesOnly);
  const std::size_t ka = static_cast<std::size_t>(sa.eigenvalues().size());
  frame.schmidt_values.resize(ka);
  for (std::size_t i = 0; i < ka; ++i) {
    frame.schmidt_values[i] = std::sqrt(std::max(0.0, sa.eigenvalues()(static_cast<Eigen::Index>(i))));
  }
  std::sort(frame.schmidt_values.begin(), frame.schmidt_values.end(), std::greater<double>());

  // Concurrence for 2-qubit only
  if (n == 2U) {
    frame.concurrence = wootters_concurrence(rho);
  }

  return frame;
}

// ── LindbladBackend ─────────────────────────────────────────────────────────

class LindbladBackend final : public IBackend {
 public:
  std::string name() const override { return "lindblad"; }

  BackendCapabilities capabilities() const override {
    BackendCapabilities c;
    c.kind = BackendKind::Statevector;
    c.supports_gpu = false;
    c.supports_mpi = false;
    c.supports_noisy = true;
    c.max_qubits_recommended = 6;
    c.equation_scope = "Lindblad";
    return c;
  }

  void initialize(const QuantumProblem& problem, const RunOptions& options) override {
    n_ = problem.qubits;
    const auto dim =
        static_cast<Eigen::Index>(lindblad_dimension(n_));
    rho_ = MatrixXcd::Zero(dim, dim);
    rho_(0, 0) = Complex{1.0, 0.0};
    H_ = build_hamiltonian(problem.hamiltonian_terms, n_);
    ops_ = build_collapse_ops(problem.hamiltonian_terms, n_);
    dt_ = problem.evolution.dt;
    total_time_ = problem.evolution.total_time;
    entanglement_cut_ = options.entanglement_cut;
    frames_.clear();
    entanglement_frames_.clear();
  }

  void evolve(const QuantumProblem&, const RunOptions&) override {
    const auto steps = static_cast<std::uint32_t>(
        std::max(1.0, std::round(total_time_ / dt_)));

    record_frame(0U, 0.0);
    if (entanglement_cut_ > 0 && entanglement_cut_ < n_) {
      entanglement_frames_.push_back(
          compute_entanglement_mixed(rho_, entanglement_cut_, n_, 0U, 0.0));
    }

    for (std::uint32_t s = 0; s < steps; ++s) {
      rho_ = rk4_step(rho_, dt_, H_, ops_);
      // Re-symmetrise to prevent numerical drift from breaking Hermiticity.
      rho_ = (rho_ + rho_.adjoint()) * 0.5;
      const double t = static_cast<double>(s + 1U) * dt_;
      record_frame(s + 1U, t);
      if (entanglement_cut_ > 0 && entanglement_cut_ < n_) {
        entanglement_frames_.push_back(
            compute_entanglement_mixed(rho_, entanglement_cut_, n_, s + 1U, t));
      }
    }
  }

  std::map<std::string, double> measure(const QuantumProblem&) override {
    std::map<std::string, double> obs;
    obs["purity"] = compute_purity(rho_);
    obs["entropy"] = compute_entropy(rho_);
    const auto dim = static_cast<std::size_t>(rho_.rows());
    for (std::size_t i = 0; i < dim; ++i) {
      obs["population_" + std::to_string(i)] = std::max(0.0, rho_(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)).real());
    }
    return obs;
  }

  void checkpoint(const std::string&) override {}
  void restore(const std::string&) override {}

  std::vector<std::complex<double>> current_state() const override {
    // Return diagonal populations as a proxy statevector for compatibility.
    const auto dim = static_cast<std::size_t>(rho_.rows());
    std::vector<Complex> state(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      state[i] = Complex{std::max(0.0, rho_(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)).real()), 0.0};
    }
    return state;
  }

  std::vector<ContinuousDensityFrame> continuous_density_frames() const override { return {}; }

  std::vector<LindbladFrame> lindblad_frames() const override { return frames_; }

  std::vector<EntanglementFrame> entanglement_frames() const override {
    return entanglement_frames_;
  }

 private:
  void record_frame(std::uint32_t step, double t) {
    LindbladFrame f;
    f.step = step;
    f.time = t;
    f.purity = compute_purity(rho_);
    f.entropy = compute_entropy(rho_);

    const auto dim = static_cast<std::size_t>(rho_.rows());
    f.populations.resize(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      f.populations[i] = std::max(0.0, rho_(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)).real());
    }

    double max_coh = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
      for (std::size_t j = 0; j < dim; ++j) {
        if (i != j) {
          max_coh = std::max(max_coh, std::abs(rho_(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j))));
        }
      }
    }
    f.coherence = max_coh;
    frames_.push_back(f);
  }

  std::uint32_t n_{1};
  double dt_{0.01};
  double total_time_{1.0};
  std::uint32_t entanglement_cut_{0};
  MatrixXcd rho_;
  MatrixXcd H_;
  std::vector<CollapseOp> ops_;
  std::vector<LindbladFrame> frames_;
  std::vector<EntanglementFrame> entanglement_frames_;
};

}  // namespace

std::unique_ptr<IBackend> make_lindblad_backend() {
  return std::make_unique<LindbladBackend>();
}

}  // namespace qx
