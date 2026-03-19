#include "qx/backends.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

namespace qx {
namespace {

// Lightweight MPS backend scaffold with configurable truncation controls.
class MpsBackend final : public IBackend {
 public:
  // Return backend name.
  std::string name() const override { return "mps"; }

  // Return static MPS backend capability metadata.
  BackendCapabilities capabilities() const override {
    BackendCapabilities c;
    c.kind = BackendKind::MPS;
    c.supports_gpu = true;
    c.supports_mpi = true;
    c.supports_noisy = false;
    c.max_qubits_recommended = 1000;
    c.equation_scope = "Schrodinger";
    return c;
  }

  // Initialize MPS surrogate state from problem configuration.
  void initialize(const QuantumProblem& problem, const RunOptions&) override {
    qubits_ = problem.qubits;
    // Start from a small bond and let runtime growth model decide expansion.
    bond_dim_ = 4;
    truncation_error_ = 0.0;
    method_ = problem.evolution.method.empty() ? "tebd" : problem.evolution.method;
    max_bond_ = std::max<std::uint32_t>(2U, problem.evolution.mps_max_bond);
    trunc_tol_ = std::max(1e-14, problem.evolution.mps_trunc_tol);
    observable_cache_.clear();
  }

  // Evolve surrogate MPS statistics across timesteps.
  void evolve(const QuantumProblem& problem, const RunOptions&) override {
    const std::size_t steps = static_cast<std::size_t>(problem.evolution.total_time / problem.evolution.dt);
    // Surrogate entanglement-growth factors used in v1 before full tensor contractions land.
    const double term_factor = 1.0 + static_cast<double>(std::max<std::size_t>(1, problem.hamiltonian_terms.size())) * 0.05;
    const double depth_factor = 1.0 + static_cast<double>(problem.expected_circuit_depth) * 0.002;
    const double method_factor = (method_ == "tdvp") ? 0.65 : 1.0;  // tdvp: smoother growth / less truncation
    const double base_growth = std::log1p(std::max(2U, problem.qubits) * 0.08) * term_factor * depth_factor * method_factor;

    for (std::size_t i = 0; i < steps; ++i) {
      const double candidate = static_cast<double>(bond_dim_) + base_growth;
      const double clipped = std::min<double>(static_cast<double>(max_bond_), candidate);
      const double overflow = std::max(0.0, candidate - clipped);

      // Bound effective bond dimension by user contract.
      bond_dim_ = static_cast<std::uint32_t>(std::max(2.0, std::round(clipped)));

      // Truncation increases when bond growth is clipped by max_bond.
      if (overflow > 0.0) {
        truncation_error_ += trunc_tol_ * (overflow / std::max(1.0, candidate));
      }
      // Background truncation floor to mimic periodic SVD truncation.
      truncation_error_ += trunc_tol_ * 0.01;
    }
    observable_cache_["bond_dim"] = static_cast<double>(bond_dim_);
    observable_cache_["truncation_error"] = truncation_error_;
    observable_cache_["mps_max_bond"] = static_cast<double>(max_bond_);
    observable_cache_["mps_trunc_tol"] = trunc_tol_;
    observable_cache_["entanglement_entropy_estimate"] =
        std::log2(std::max(2.0, static_cast<double>(bond_dim_)));
  }

  // Return cached observables emitted by this backend.
  std::map<std::string, double> measure(const QuantumProblem&) override { return observable_cache_; }

  // Persist surrogate MPS state for restart.
  void checkpoint(const std::string& path) override {
    // Persist model controls too, so restore is behaviorally consistent.
    std::ofstream os(path + "/mps.chk");
    os << qubits_ << "\n" << bond_dim_ << "\n" << truncation_error_ << "\n" << max_bond_ << "\n" << trunc_tol_ << "\n"
       << method_ << "\n";
  }

  // Restore surrogate MPS state from checkpoint if it exists.
  void restore(const std::string& path) override {
    std::ifstream is(path + "/mps.chk");
    if (!is.good()) {
      return;
    }
    is >> qubits_ >> bond_dim_ >> truncation_error_ >> max_bond_ >> trunc_tol_;
    if (!(is >> method_)) {
      // Backward compatibility for older checkpoints without method field.
      method_ = "tebd";
    }
    observable_cache_["bond_dim"] = static_cast<double>(bond_dim_);
    observable_cache_["truncation_error"] = truncation_error_;
    observable_cache_["mps_max_bond"] = static_cast<double>(max_bond_);
    observable_cache_["mps_trunc_tol"] = trunc_tol_;
  }

  // Dense state is not exposed for compressed MPS backend.
  std::vector<std::complex<double>> current_state() const override { return {}; }

  // MPS backend does not emit continuous density frames.
  std::vector<ContinuousDensityFrame> continuous_density_frames() const override { return {}; }

 private:
  std::uint32_t qubits_{0};
  std::uint32_t bond_dim_{4};
  std::uint32_t max_bond_{1024};
  double truncation_error_{0.0};
  double trunc_tol_{1e-8};
  std::string method_{"tebd"};
  std::map<std::string, double> observable_cache_{};
};

}  // namespace

// Construct MPS backend instance.
std::unique_ptr<IBackend> make_mps_backend() {
  return std::make_unique<MpsBackend>();
}

}  // namespace qx
