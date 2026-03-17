#pragma once

#include <string>

#include "qx/types.hpp"

namespace qx {

QuantumProblem problem_from_json(const std::string& json);
RunOptions options_from_json(const std::string& json);
std::string result_to_json(const SimulationResult& result);
std::string backend_info_to_json(const BackendInfo& info);

}  // namespace qx
