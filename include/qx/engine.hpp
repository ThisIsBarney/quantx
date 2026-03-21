#pragma once

#include <stdexcept>

#include "qx/types.hpp"

namespace qx {

// Structured circuit execution error with machine-readable code.
class CircuitError final : public std::runtime_error {
 public:
  // Build an error with a stable code and human-readable message.
  CircuitError(std::string code, std::string message);
  // Return the stable machine-readable code.
  const std::string& code() const noexcept;

 private:
  std::string code_;
};

// Run Hamiltonian evolution according to problem + runtime options.
SimulationResult run(const QuantumProblem& problem, const RunOptions& options);
// Explain which backend will be selected and why.
BackendInfo explain_backend_choice(const QuantumProblem& problem, const RunOptions& options);
// Return gate-model capability metadata for the current circuit executor.
CircuitCapabilities circuit_backend_capabilities();
// Execute quantum circuit and optionally sample shot statistics.
CircuitResult run_circuit(const QuantumCircuit& circuit, const RunOptions& options);

}  // namespace qx
