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
// Parse spectral function options from JSON string.
SpectralOptions spectral_options_from_json(const std::string& json);
// Serialize spectral function result to JSON string.
std::string spectral_result_to_json(const SpectralResult& result);
// Parse 10-fold way topological classification options from JSON string.
TenfoldOptions tenfold_options_from_json(const std::string& json);
// Serialize 10-fold way topological classification result to JSON string.
std::string tenfold_result_to_json(const TenfoldResult& result);
// Parse second Chern number (4D QHE) options from JSON string.
SecondChernOptions second_chern_options_from_json(const std::string& json);
// Serialize second Chern number result to JSON string.
std::string second_chern_result_to_json(const SecondChernResult& result);
// Parse Wilson loop / Wannier Charge Center options from JSON string.
WilsonLoopOptions wilson_loop_options_from_json(const std::string& json);
// Serialize Wilson loop result to JSON string.
std::string wilson_loop_result_to_json(const WilsonLoopResult& result);
// Parse edge states / surface spectrum options from JSON string.
EdgeStatesOptions edge_states_options_from_json(const std::string& json);
// Serialize edge states result to JSON string.
std::string edge_states_result_to_json(const EdgeStatesResult& result);
// Parse Haldane model (QAH effect) options from JSON string.
HaldaneOptions haldane_options_from_json(const std::string& json);
// Serialize Haldane model result to JSON string.
std::string    haldane_result_to_json(const HaldaneResult& result);
// Parse Weyl semimetal options from JSON string.
WeylOptions weyl_options_from_json(const std::string& json);
// Serialize Weyl semimetal result to JSON string.
std::string weyl_result_to_json(const WeylResult& result);

// Parse Fu-Kane superconductor options from JSON string.
FuKaneOptions fu_kane_options_from_json(const std::string& json);
// Serialize Fu-Kane superconductor result to JSON string.
std::string   fu_kane_result_to_json(const FuKaneResult& result);
// Parse Lutchyn-Oreg Majorana nanowire options from JSON string.
LutchynOregOptions lutchyn_oreg_options_from_json(const std::string& json);
// Serialize Lutchyn-Oreg Majorana nanowire result to JSON string.
std::string        lutchyn_oreg_result_to_json(const LutchynOregResult& result);
// Parse Kitaev honeycomb model options from JSON string.
KitaevHcOptions kitaev_hc_options_from_json(const std::string& json);
// Serialize Kitaev honeycomb model result to JSON string.
std::string     kitaev_hc_result_to_json(const KitaevHcResult& result);
// Parse BBH quadrupole insulator options from JSON string.
BBHOptions  bbh_options_from_json(const std::string& json);
// Serialize BBH quadrupole insulator result to JSON string.
std::string bbh_result_to_json(const BBHResult& result);
// Parse Floquet topological insulator options from JSON string.
FloquetTopoOptions  floquet_topo_options_from_json(const std::string& json);
// Serialize Floquet topological insulator result to JSON string.
std::string         floquet_topo_result_to_json(const FloquetTopoResult& result);
// Parse non-Hermitian topology options from JSON string.
NonHermOptions  non_herm_options_from_json(const std::string& json);
// Serialize non-Hermitian topology result to JSON string.
std::string     non_herm_result_to_json(const NonHermResult& result);
// Parse Aubry-Andre-Harper quasicrystal options from JSON string.
AAHOptions  aah_options_from_json(const std::string& json);
// Serialize Aubry-Andre-Harper quasicrystal result to JSON string.
std::string aah_result_to_json(const AAHResult& result);
// Parse Sachdev-Ye-Kitaev model options from JSON string.
SYKOptions  syk_options_from_json(const std::string& json);
// Serialize Sachdev-Ye-Kitaev model result to JSON string.
std::string syk_result_to_json(const SYKResult& result);
// Parse Kondo/SIAM options from JSON string.
KondoOptions  kondo_options_from_json(const std::string& json);
// Serialize Kondo/SIAM result to JSON string.
std::string   kondo_result_to_json(const KondoResult& result);
// Parse bipartite entanglement entropy options from JSON string.
EntOptions    ent_options_from_json(const std::string& json);
// Serialize bipartite entanglement entropy result to JSON string.
std::string   ent_result_to_json(const EntResult& result);

}  // namespace qx
