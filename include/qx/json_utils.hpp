#pragma once

#include <string>

#include "qx/types.hpp"

namespace qx {

// Parse simulation problem JSON into strongly-typed IR.
QuantumProblem problem_from_json(const std::string& json);
// Parse runtime options JSON into execution options.
RunOptions options_from_json(const std::string& json);
// Parse a continuous-export contract snapshot from JSON and normalize it defensively.
ContinuousExportConfig continuous_export_config_from_json(const std::string& json);
// Clamp continuous-export contract fields to stable append-only bounds.
ContinuousExportConfig sanitize_continuous_export_config(const ContinuousExportConfig& config);
// Parse circuit JSON into gate-level circuit representation.
QuantumCircuit circuit_from_json(const std::string& json);
// Serialize simulation result to JSON string.
std::string result_to_json(const SimulationResult& result);
// Serialize circuit execution result to JSON string.
std::string circuit_result_to_json(const CircuitResult& result);
// Serialize circuit capability metadata to JSON string.
std::string circuit_capabilities_to_json(const CircuitCapabilities& capabilities);
// Serialize backend selection explanation to JSON string.
std::string backend_info_to_json(const BackendInfo& info);

}  // namespace qx
