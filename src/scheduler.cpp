#include "qx/scheduler.hpp"

#include "qx/backend.hpp"

namespace qx {

// Compute backend selection explanation from problem size/physics/runtime hints.
BackendInfo explain_backend_choice_internal(const QuantumProblem& problem, const RunOptions& options) {
  BackendInfo info;
  info.candidates = BackendRegistry::instance().names();

  if (problem.equation != EquationType::Schrodinger) {
    info.selected_backend = "statevector";
    info.reasons.push_back("Non-Schrodinger equations are reserved for future backends.");
    return info;
  }

  if (problem.qubits <= 28 && !problem.noisy) {
    info.selected_backend = "statevector";
    info.reasons.push_back("Qubit count fits dense statevector recommendation.");
    if (options.use_gpu) {
      info.reasons.push_back("GPU requested and statevector backend supports GPU acceleration.");
    }
    return info;
  }

  info.selected_backend = "mps";
  info.reasons.push_back("Large qubit count favors tensor network compression.");
  if (problem.expected_circuit_depth > 100) {
    info.reasons.push_back("Circuit depth indicates likely entanglement growth; monitor bond dimension.");
  }
  return info;
}

// Return only selected backend name from explanation output.
std::string choose_backend(const QuantumProblem& problem, const RunOptions& options) {
  return explain_backend_choice_internal(problem, options).selected_backend;
}

}  // namespace qx
