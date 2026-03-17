#include "qx/backends.hpp"

#include <cmath>
#include <fstream>

namespace qx {
namespace {

class MpsBackend final : public IBackend {
 public:
  std::string name() const override { return "mps"; }

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

  void initialize(const QuantumProblem& problem, const RunOptions&) override {
    qubits_ = problem.qubits;
    bond_dim_ = 4;
    truncation_error_ = 0.0;
    observable_cache_.clear();
  }

  void evolve(const QuantumProblem& problem, const RunOptions&) override {
    const std::size_t steps = static_cast<std::size_t>(problem.evolution.total_time / problem.evolution.dt);
    for (std::size_t i = 0; i < steps; ++i) {
      const double growth = std::log1p(static_cast<double>(problem.qubits) * 0.05) * 0.5;
      bond_dim_ = static_cast<std::uint32_t>(std::min<double>(1024.0, bond_dim_ + growth));
      truncation_error_ += 1e-10 * static_cast<double>(bond_dim_);
    }
    observable_cache_["bond_dim"] = static_cast<double>(bond_dim_);
    observable_cache_["truncation_error"] = truncation_error_;
    observable_cache_["entanglement_entropy_estimate"] =
        std::log2(std::max(2.0, static_cast<double>(bond_dim_)));
  }

  std::map<std::string, double> measure(const QuantumProblem&) override { return observable_cache_; }

  void checkpoint(const std::string& path) override {
    std::ofstream os(path + "/mps.chk");
    os << qubits_ << "\n" << bond_dim_ << "\n" << truncation_error_ << "\n";
  }

  void restore(const std::string& path) override {
    std::ifstream is(path + "/mps.chk");
    if (!is.good()) {
      return;
    }
    is >> qubits_ >> bond_dim_ >> truncation_error_;
    observable_cache_["bond_dim"] = static_cast<double>(bond_dim_);
    observable_cache_["truncation_error"] = truncation_error_;
  }

  std::vector<std::complex<double>> current_state() const override { return {}; }

 private:
  std::uint32_t qubits_{0};
  std::uint32_t bond_dim_{4};
  double truncation_error_{0.0};
  std::map<std::string, double> observable_cache_{};
};

}  // namespace

std::unique_ptr<IBackend> make_mps_backend() {
  return std::make_unique<MpsBackend>();
}

}  // namespace qx
