#include "qx/backends.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <regex>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace qx {
namespace {

using Complex = std::complex<double>;
using MatrixXcd = Eigen::MatrixXcd;

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
  const std::size_t dim = static_cast<std::size_t>(1U) << n;
  const std::size_t q_mask = static_cast<std::size_t>(1U) << q;
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
  const std::size_t dim = static_cast<std::size_t>(1U) << n;
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

// Parse collapse-operator entries from HamiltonianTerms.
// Label formats:
//   "AmplitudeDamping<q>"  coefficient = γ  →  L = sqrt(γ) σ⁻_q
//   "Dephasing<q>"         coefficient = γ  →  L = sqrt(γ) σz_q / 2
//   "Depolarizing<q>"      coefficient = γ  →  3 ops: sqrt(γ/3)·{σx,σy,σz}_q

std::vector<CollapseOp> build_collapse_ops(const std::vector<HamiltonianTerm>& terms,
                                            std::uint32_t n) {
  std::vector<CollapseOp> ops;

  auto parse_qubit = [](const std::string& label, const std::string& prefix) -> std::optional<std::uint32_t> {
    if (label.size() <= prefix.size()) {
      return std::nullopt;
    }
    try {
      return static_cast<std::uint32_t>(std::stoul(label.substr(prefix.size())));
    } catch (...) {
      return std::nullopt;
    }
  };

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
      auto q = parse_qubit(label, "AmplitudeDamping");
      if (!q || *q >= n) {
        continue;
      }
      push_op(sqrt_gamma * embed_operator(sigma_minus(), *q, n));
    } else if (label.rfind("Dephasing", 0) == 0) {
      auto q = parse_qubit(label, "Dephasing");
      if (!q || *q >= n) {
        continue;
      }
      // L = sqrt(γ) * σz / 2
      push_op((sqrt_gamma * 0.5) * embed_operator(pauli_z(), *q, n));
    } else if (label.rfind("Depolarizing", 0) == 0) {
      auto q = parse_qubit(label, "Depolarizing");
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

  void initialize(const QuantumProblem& problem, const RunOptions&) override {
    n_ = problem.qubits;
    const auto dim = static_cast<Eigen::Index>(static_cast<std::size_t>(1U) << n_);
    rho_ = MatrixXcd::Zero(dim, dim);
    rho_(0, 0) = Complex{1.0, 0.0};
    H_ = build_hamiltonian(problem.hamiltonian_terms, n_);
    ops_ = build_collapse_ops(problem.hamiltonian_terms, n_);
    dt_ = problem.evolution.dt;
    total_time_ = problem.evolution.total_time;
    frames_.clear();
  }

  void evolve(const QuantumProblem&, const RunOptions&) override {
    const auto steps = static_cast<std::uint32_t>(
        std::max(1.0, std::round(total_time_ / dt_)));

    record_frame(0U, 0.0);

    for (std::uint32_t s = 0; s < steps; ++s) {
      rho_ = rk4_step(rho_, dt_, H_, ops_);
      // Re-symmetrise to prevent numerical drift from breaking Hermiticity.
      rho_ = (rho_ + rho_.adjoint()) * 0.5;
      record_frame(s + 1U, static_cast<double>(s + 1U) * dt_);
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
  MatrixXcd rho_;
  MatrixXcd H_;
  std::vector<CollapseOp> ops_;
  std::vector<LindbladFrame> frames_;
};

}  // namespace

std::unique_ptr<IBackend> make_lindblad_backend() {
  return std::make_unique<LindbladBackend>();
}

}  // namespace qx
