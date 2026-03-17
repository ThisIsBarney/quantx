#pragma once

#include "qx/types.hpp"

namespace qx {

// Internal backend explanation logic used by public API and tests.
BackendInfo explain_backend_choice_internal(const QuantumProblem& problem, const RunOptions& options);
// Return selected backend name for a given problem/options pair.
std::string choose_backend(const QuantumProblem& problem, const RunOptions& options);

}  // namespace qx
