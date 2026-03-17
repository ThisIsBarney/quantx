#pragma once

#include "qx/types.hpp"

namespace qx {

SimulationResult run(const QuantumProblem& problem, const RunOptions& options);
BackendInfo explain_backend_choice(const QuantumProblem& problem, const RunOptions& options);

}  // namespace qx
