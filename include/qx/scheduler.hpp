#pragma once

#include "qx/types.hpp"

namespace qx {

BackendInfo explain_backend_choice_internal(const QuantumProblem& problem, const RunOptions& options);
std::string choose_backend(const QuantumProblem& problem, const RunOptions& options);

}  // namespace qx
