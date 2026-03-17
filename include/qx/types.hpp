#pragma once

#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace qx {

enum class EquationType {
  Schrodinger,
  Lindblad,
  Dirac,
  KleinGordon
};

enum class BackendKind {
  Statevector,
  MPS
};

struct ResourceConstraints {
  std::uint32_t max_threads{0};
  std::uint32_t max_gpus{0};
  std::uint32_t mpi_ranks{1};
  std::uint64_t max_memory_mb{0};
};

struct EvolutionConfig {
  double total_time{1.0};
  double dt{0.01};
  std::string method{"trotter"};
};

struct HamiltonianTerm {
  std::string label;
  double coefficient{0.0};
};

struct OperatorNode {
  std::string id;
  std::string op_type;
  std::vector<std::string> inputs;
  std::map<std::string, double> params;
};

struct OperatorGraph {
  std::vector<OperatorNode> nodes;
};

struct QuantumProblem {
  std::uint32_t qubits{1};
  EquationType equation{EquationType::Schrodinger};
  EvolutionConfig evolution{};
  std::vector<HamiltonianTerm> hamiltonian_terms{};
  OperatorGraph operator_graph{};
  std::vector<std::string> observables{"norm"};
  double target_error{1e-8};
  bool noisy{false};
  std::uint32_t expected_circuit_depth{0};
};

struct RunOptions {
  bool use_gpu{false};
  bool allow_mpi{false};
  bool enable_checkpoint{true};
  std::string checkpoint_path{"./checkpoints"};
  ResourceConstraints resources{};
};

struct Event {
  std::string name;
  std::map<std::string, double> metrics;
};

struct SimulationResult {
  bool ok{true};
  std::string backend;
  std::vector<std::complex<double>> state{};
  std::map<std::string, double> observables{};
  std::vector<Event> events{};
  std::string message;
};

struct BackendInfo {
  std::string selected_backend;
  std::vector<std::string> candidates;
  std::vector<std::string> reasons;
};

struct BackendCapabilities {
  BackendKind kind{BackendKind::Statevector};
  bool supports_gpu{false};
  bool supports_mpi{false};
  bool supports_noisy{false};
  std::uint32_t max_qubits_recommended{0};
  std::string equation_scope{"Schrodinger"};
};

}  // namespace qx
