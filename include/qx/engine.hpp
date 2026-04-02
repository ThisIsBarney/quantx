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
// Run variational eigensolver (VQE) or QAOA optimization.
VqeResult run_vqe(const VqeOptions& opts);
// Run Quantum Phase Estimation to extract eigenphases of a unitary.
QpeResult run_qpe(const QpeOptions& opts);
// Run Quantum Error Correction Monte Carlo simulation.
QecResult run_qec(const QecOptions& opts);
// Run Schwinger model (1+1D QED) lattice simulation.
QfieldResult run_qfield(const QfieldOptions& opts);
// Run φ⁴ scalar quantum field theory simulation.
Phi4Result run_phi4(const Phi4Options& opts);
// Run 1D Hubbard model (Mott insulator) simulation.
HubbardResult run_hubbard(const HubbardOptions& opts);
// Run Floquet kicked-Ising model simulation.
FloquetResult run_floquet(const FloquetOptions& opts);
// Run 1D XXZ Heisenberg spin chain simulation.
HeisenbergResult run_heisenberg(const HeisenbergOptions& opts);
// Run Bose-Hubbard model simulation.
BoseHubbardResult run_bose_hubbard(const BoseHubbardOptions& opts);
// Run Transverse-Field Ising model simulation.
TFIResult run_tfi(const TFIOptions& opts);
// Run SSH (Su-Schrieffer-Heeger) topological insulator simulation.
SSHResult run_ssh(const SSHOptions& opts);
// Run Kitaev chain (p-wave topological superconductor) simulation.
KitaevResult run_kitaev(const KitaevOptions& opts);
// Run Anderson localization simulation.
AndersonResult run_anderson(const AndersonOptions& opts);
// Run 2D Chern insulator (Qi-Wu-Zhang model) simulation.
ChernResult run_chern(const ChernOptions& opts);
// Run Hofstadter butterfly simulation.
HofstadterResult run_hofstadter(const HofstadterOptions& opts);
// Run Many-Body Localization (disordered XXZ chain) simulation.
MBLResult run_mbl(const MBLOptions& opts);

}  // namespace qx
