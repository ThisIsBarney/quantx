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
// Parse VQE options from JSON string.
VqeOptions vqe_options_from_json(const std::string& json);
// Serialize VQE result to JSON string.
std::string vqe_result_to_json(const VqeResult& result);
// Parse QPE options from JSON string.
QpeOptions qpe_options_from_json(const std::string& json);
// Serialize QPE result to JSON string.
std::string qpe_result_to_json(const QpeResult& result);
// Parse QEC options from JSON string.
QecOptions qec_options_from_json(const std::string& json);
// Serialize QEC result to JSON string.
std::string qec_result_to_json(const QecResult& result);
// Parse QField (Schwinger model) options from JSON string.
QfieldOptions qfield_options_from_json(const std::string& json);
// Serialize QField result to JSON string.
std::string qfield_result_to_json(const QfieldResult& result);
// Parse φ⁴ QFT options from JSON string.
Phi4Options phi4_options_from_json(const std::string& json);
// Serialize φ⁴ QFT result to JSON string.
std::string phi4_result_to_json(const Phi4Result& result);
// Parse Hubbard model options from JSON string.
HubbardOptions hubbard_options_from_json(const std::string& json);
// Serialize Hubbard model result to JSON string.
std::string hubbard_result_to_json(const HubbardResult& result);
// Parse Floquet options from JSON string.
FloquetOptions floquet_options_from_json(const std::string& json);
// Serialize Floquet result to JSON string.
std::string floquet_result_to_json(const FloquetResult& result);
// Parse Heisenberg chain options from JSON string.
HeisenbergOptions heisenberg_options_from_json(const std::string& json);
// Serialize Heisenberg result to JSON string.
std::string heisenberg_result_to_json(const HeisenbergResult& result);
// Parse Bose-Hubbard options from JSON string.
BoseHubbardOptions bose_hubbard_options_from_json(const std::string& json);
// Serialize Bose-Hubbard result to JSON string.
std::string bose_hubbard_result_to_json(const BoseHubbardResult& result);
// Parse Transverse-Field Ising options from JSON string.
TFIOptions tfi_options_from_json(const std::string& json);
// Serialize Transverse-Field Ising result to JSON string.
std::string tfi_result_to_json(const TFIResult& result);
// Parse SSH options from JSON string.
SSHOptions ssh_options_from_json(const std::string& json);
// Serialize SSH result to JSON string.
std::string ssh_result_to_json(const SSHResult& result);
// Parse Kitaev chain options from JSON string.
KitaevOptions kitaev_options_from_json(const std::string& json);
// Serialize Kitaev chain result to JSON string.
std::string kitaev_result_to_json(const KitaevResult& result);
// Parse Anderson localization options from JSON string.
AndersonOptions anderson_options_from_json(const std::string& json);
// Serialize Anderson localization result to JSON string.
std::string anderson_result_to_json(const AndersonResult& result);
// Parse 2D Chern insulator options from JSON string.
ChernOptions chern_options_from_json(const std::string& json);
// Serialize 2D Chern insulator result to JSON string.
std::string chern_result_to_json(const ChernResult& result);
// Parse Hofstadter butterfly options from JSON string.
HofstadterOptions hofstadter_options_from_json(const std::string& json);
// Serialize Hofstadter butterfly result to JSON string.
std::string hofstadter_result_to_json(const HofstadterResult& result);
// Parse Many-Body Localization options from JSON string.
MBLOptions mbl_options_from_json(const std::string& json);
// Serialize Many-Body Localization result to JSON string.
std::string mbl_result_to_json(const MBLResult& result);

}  // namespace qx
