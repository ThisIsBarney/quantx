#pragma once

#include "qx/types.hpp"

namespace qx {

// Run Hamiltonian evolution according to problem + runtime options.
SimulationResult run(const QuantumProblem& problem, const RunOptions& options);
// Explain which backend will be selected and why.
BackendInfo explain_backend_choice(const QuantumProblem& problem, const RunOptions& options);
// Execute quantum circuit and optionally sample shot statistics.
CircuitResult run_circuit(const QuantumCircuit& circuit, const RunOptions& options);

}  // namespace qx
