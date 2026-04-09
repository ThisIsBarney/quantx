#include <algorithm>
#include <cmath>
#include <limits>
#include "qx/json_utils.hpp"
#include "qx/engine.hpp"
#include "qx/version.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace qx {
namespace {

using json = nlohmann::json;

// Convert problem kind enum to stable JSON token.
std::string problem_kind_to_string(ProblemKind k) {
  switch (k) {
    case ProblemKind::Discrete:
      return "discrete";
    case ProblemKind::ContinuousSpace2D:
      return "continuous_space_2d";
    case ProblemKind::ContinuousSpace3D:
      return "continuous_space_3d";
  }
  return "discrete";
}

// Parse problem kind token to enum with discrete fallback.
ProblemKind problem_kind_from_string(const std::string& s) {
  if (s == "continuous_space_2d" || s == "continuous2d" || s == "continuous_2d") {
    return ProblemKind::ContinuousSpace2D;
  }
  if (s == "continuous_space_3d" || s == "continuous3d" || s == "continuous_3d") {
    return ProblemKind::ContinuousSpace3D;
  }
  return ProblemKind::Discrete;
}

// Convert equation enum to stable JSON token.
std::string equation_to_string(EquationType t) {
  switch (t) {
    case EquationType::Schrodinger:
      return "Schrodinger";
    case EquationType::Lindblad:
      return "Lindblad";
    case EquationType::Dirac:
      return "Dirac";
    case EquationType::KleinGordon:
      return "KleinGordon";
  }
  return "Schrodinger";
}

// Parse equation token to enum with Schrodinger fallback.
EquationType equation_from_string(const std::string& s) {
  if (s == "Lindblad") return EquationType::Lindblad;
  if (s == "Dirac") return EquationType::Dirac;
  if (s == "KleinGordon") return EquationType::KleinGordon;
  return EquationType::Schrodinger;
}

// Parse JSON text and throw a clear error message on invalid input.
json parse_json_or_throw(const std::string& text) {
  try {
    return json::parse(text.empty() ? "{}" : text);
  } catch (const json::parse_error& ex) {
    throw std::runtime_error(std::string("invalid json: ") + ex.what());
  }
}

// Read typed field if present, otherwise keep caller-supplied default.
template <typename T>
T get_or(const json& j, const char* key, const T& def) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) {
    return def;
  }
  return it->get<T>();
}

// Parse one Hamiltonian term object.
HamiltonianTerm parse_hamiltonian_term(const json& item) {
  HamiltonianTerm t;
  t.label = get_or<std::string>(item, "label", "");
  t.coefficient = get_or<double>(item, "coefficient", 0.0);
  return t;
}

// Parse one circuit operation object.
CircuitOp parse_circuit_op(const json& item) {
  CircuitOp op;
  op.name = get_or<std::string>(item, "name", "");
  op.targets = get_or<std::vector<std::uint32_t>>(item, "targets", {});
  op.controls = get_or<std::vector<std::uint32_t>>(item, "controls", {});
  op.params = get_or<std::vector<double>>(item, "params", {});
  op.param_names = get_or<std::vector<std::string>>(item, "param_names", {});
  return op;
}

// Parse the nested DrivingConfig payload if present.
void parse_driving_config(const json& parent, DrivingConfig& out) {
  auto it = parent.find("driving");
  if (it == parent.end() || !it->is_object()) {
    return;
  }
  const auto& d = *it;
  out.type = get_or<std::string>(d, "type", out.type);
  out.amplitude = get_or<double>(d, "amplitude", out.amplitude);
  out.frequency = get_or<double>(d, "frequency", out.frequency);
  out.phase = get_or<double>(d, "phase", out.phase);
  out.t_center = get_or<double>(d, "t_center", out.t_center);
  out.t_width = get_or<double>(d, "t_width", out.t_width);
  out.spatial_profile = get_or<std::string>(d, "spatial_profile", out.spatial_profile);
}

// Parse the nested continuous initial-state payload if present.
void parse_initial_state_config(const json& parent, ContinuousInitialStateConfig& out) {
  auto it = parent.find("initial_state");
  if (it == parent.end() || !it->is_object()) {
    return;
  }
  const auto& initial = *it;
  out.kind = get_or<std::string>(initial, "kind", out.kind);
  out.params = get_or<std::map<std::string, double>>(initial, "params", out.params);
  out.grid = get_or<std::vector<double>>(initial, "grid", out.grid);
  out.pi_kind = get_or<std::string>(initial, "pi_kind", out.pi_kind);
}

// Read a numeric field as an unsigned 32-bit value and clamp it to a minimum.
std::uint32_t read_clamped_u32(const json& j, const char* key, std::uint32_t def, std::uint32_t min_value);

// Read a numeric field as size_t and clamp it to a minimum.
std::size_t read_clamped_size(const json& j, const char* key, std::size_t def, std::size_t min_value);

// Read a density-floor ratio and clamp it to the inclusive range [0, 1].
double read_clamped_ratio(const json& j, const char* key, double def);

// Normalize a continuous-export contract so serialization and runtime share the same bounds.
ContinuousExportConfig sanitize_continuous_export_config_impl(const ContinuousExportConfig& config) {
  ContinuousExportConfig out = config;
  out.schema_version = std::max<std::uint32_t>(1U, out.schema_version);
  out.export_frame_stride = std::max<std::uint32_t>(1U, out.export_frame_stride);
  out.export_points_max = std::max<std::size_t>(1U, out.export_points_max);
  if (!std::isfinite(out.export_density_floor_ratio)) {
    out.export_density_floor_ratio = 0.0;
  }
  out.export_density_floor_ratio = std::clamp(out.export_density_floor_ratio, 0.0, 1.0);
  return out;
}

// Read a continuous-export contract from JSON, accepting either the wrapper or the bare config object.
ContinuousExportConfig parse_continuous_export_config_impl(const json& root) {
  if (!root.is_object()) {
    return sanitize_continuous_export_config_impl(ContinuousExportConfig{});
  }

  const json* node = &root;
  if (node->contains("continuous_export") && (*node)["continuous_export"].is_object()) {
    node = &(*node)["continuous_export"];
  }
  if (node->contains("export_config") && (*node)["export_config"].is_object()) {
    node = &(*node)["export_config"];
  }

  ContinuousExportConfig config;
  config.schema_version = read_clamped_u32(*node, "schema_version", config.schema_version, 1U);
  config.export_frame_stride = read_clamped_u32(*node, "export_frame_stride", config.export_frame_stride, 1U);
  config.export_points_max = read_clamped_size(*node, "export_points_max", config.export_points_max, 1U);
  config.export_density_floor_ratio =
      read_clamped_ratio(*node, "export_density_floor_ratio", config.export_density_floor_ratio);
  return sanitize_continuous_export_config_impl(config);
}

// Read a numeric field as an unsigned 32-bit value and clamp it to a minimum.
std::uint32_t read_clamped_u32(const json& j, const char* key, std::uint32_t def, std::uint32_t min_value) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) {
    return def;
  }
  double value = 0.0;
  try {
    value = it->get<double>();
  } catch (...) {
    return def;
  }
  if (!std::isfinite(value)) {
    return def;
  }
  if (value < static_cast<double>(min_value)) {
    return min_value;
  }
  if (value > static_cast<double>(std::numeric_limits<std::uint32_t>::max())) {
    return std::numeric_limits<std::uint32_t>::max();
  }
  return static_cast<std::uint32_t>(value);
}

// Read a numeric field as size_t and clamp it to a minimum.
std::size_t read_clamped_size(const json& j, const char* key, std::size_t def, std::size_t min_value) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) {
    return def;
  }
  long double value = 0.0L;
  try {
    value = it->get<long double>();
  } catch (...) {
    return def;
  }
  if (!std::isfinite(static_cast<double>(value))) {
    return def;
  }
  if (value < static_cast<long double>(min_value)) {
    return min_value;
  }
  const long double max_value = static_cast<long double>(std::numeric_limits<std::size_t>::max());
  if (value > max_value) {
    return std::numeric_limits<std::size_t>::max();
  }
  return static_cast<std::size_t>(value);
}

// Read a density-floor ratio and clamp it to the inclusive range [0, 1].
double read_clamped_ratio(const json& j, const char* key, double def) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) {
    return def;
  }
  double value = 0.0;
  try {
    value = it->get<double>();
  } catch (...) {
    return def;
  }
  if (!std::isfinite(value)) {
    return def;
  }
  return std::clamp(value, 0.0, 1.0);
}

// Read a numeric field as an unsigned 32-bit value and clamp it to a minimum.
std::uint32_t read_clamped_u32(const json& j, const char* key, std::uint32_t def, std::uint32_t min_value);

// Read a numeric field as size_t and clamp it to a minimum.
std::size_t read_clamped_size(const json& j, const char* key, std::size_t def, std::size_t min_value);

// Read a density-floor ratio and clamp it to the inclusive range [0, 1].
double read_clamped_ratio(const json& j, const char* key, double def);

// Parse the optional circuit noise payload and keep legacy fields synchronized.
void parse_noise_config(const json& parent, QuantumCircuit& out) {
  auto it = parent.find("noise");
  if (it == parent.end() || !it->is_object()) {
    return;
  }
  const auto& noise = *it;
  out.noise_model = get_or<std::string>(noise, "kind", out.noise_model);
  out.noise_params = get_or<std::map<std::string, double>>(noise, "params", out.noise_params);
  if (out.noise_model == "depolarizing") {
    out.depolarizing_prob = get_or<double>(noise, "prob", out.depolarizing_prob);
    if (out.noise_params.count("prob") != 0U) {
      out.depolarizing_prob = out.noise_params["prob"];
    }
  } else if (out.noise_model == "amplitude_damping") {
    out.amplitude_damping_gamma = get_or<double>(noise, "gamma", out.amplitude_damping_gamma);
    if (out.noise_params.count("gamma") != 0U) {
      out.amplitude_damping_gamma = out.noise_params["gamma"];
    }
  } else if (out.noise_model == "bit_flip" || out.noise_model == "phase_flip") {
    const double p = get_or<double>(noise, "prob", 0.0);
    out.noise_params["prob"] = (out.noise_params.count("prob") != 0U) ? out.noise_params["prob"] : p;
  } else if (out.noise_model == "t1_decay") {
    if (out.noise_params.find("T1") == out.noise_params.end()) {
      out.noise_params["T1"] = get_or<double>(noise, "T1", 0.0);
    }
    if (out.noise_params.find("dt") == out.noise_params.end()) {
      out.noise_params["dt"] = get_or<double>(noise, "dt", 0.0);
    }
  } else if (out.noise_model == "t2_decay") {
    if (out.noise_params.find("T2") == out.noise_params.end()) {
      out.noise_params["T2"] = get_or<double>(noise, "T2", 0.0);
    }
    if (out.noise_params.find("dt") == out.noise_params.end()) {
      out.noise_params["dt"] = get_or<double>(noise, "dt", 0.0);
    }
  } else if (out.noise_model != "none" && !out.noise_model.empty()) {
    throw CircuitError("QX_UNSUPPORTED_NOISE", "unsupported noise model: " + out.noise_model);
  }
}

// Serialize events to JSON array.
json events_to_json(const std::vector<Event>& events) {
  json out = json::array();
  for (const auto& ev : events) {
    json item = {{"name", ev.name}};
    if (!ev.metrics.empty()) {
      item["metrics"] = ev.metrics;
    }
    out.push_back(std::move(item));
  }
  return out;
}

// Serialize a continuous density frame metadata record.
json continuous_density_frame_to_json(const ContinuousDensityFrame& frame) {
  json out = json{
      {"step", frame.step},
      {"time", frame.time},
      {"nx", frame.nx},
      {"ny", frame.ny},
      {"nz", frame.nz},
      {"boundary", frame.boundary},
      {"norm", frame.norm},
      {"x_mean", frame.x_mean},
      {"y_mean", frame.y_mean},
      {"z_mean", frame.z_mean},
      {"sigma_x", frame.sigma_x},
      {"sigma_y", frame.sigma_y},
      {"sigma_z", frame.sigma_z},
      {"x_min", frame.x_min},
      {"x_max", frame.x_max},
      {"y_min", frame.y_min},
      {"y_max", frame.y_max},
      {"z_min", frame.z_min},
      {"z_max", frame.z_max},
      {"potential_kind", frame.potential_kind},
      {"driving_amplitude", frame.driving_amplitude},
  };
  if (!frame.density_xy_flat.empty()) {
    out["density_xy_flat"] = frame.density_xy_flat;
  }
  if (!frame.density_xz_flat.empty()) {
    out["density_xz_flat"] = frame.density_xz_flat;
  }
  if (!frame.density_yz_flat.empty()) {
    out["density_yz_flat"] = frame.density_yz_flat;
  }
  if (!frame.points_xyzd_flat.empty()) {
    out["points_xyzd_flat"] = frame.points_xyzd_flat;
  }
  // Phase slices (arg ψ₀) — 2D midplane projections
  if (!frame.phase_xy_flat.empty()) {
    out["phase_xy_flat"] = frame.phase_xy_flat;
  }
  if (!frame.phase_xz_flat.empty()) {
    out["phase_xz_flat"] = frame.phase_xz_flat;
  }
  if (!frame.phase_yz_flat.empty()) {
    out["phase_yz_flat"] = frame.phase_yz_flat;
  }
  // Dirac spinor component densities
  if (!frame.dirac_s0_xy_flat.empty()) {
    out["dirac_s0_xy_flat"] = frame.dirac_s0_xy_flat;
  }
  if (!frame.dirac_s1_xy_flat.empty()) {
    out["dirac_s1_xy_flat"] = frame.dirac_s1_xy_flat;
  }
  // Full 3D voxel grid for isosurface rendering
  if (!frame.density_xyz_flat.empty()) {
    out["density_xyz_flat"] = frame.density_xyz_flat;
  }
  return out;
}

// Serialize continuous density frame metadata to JSON array.
json continuous_density_frames_to_json(const std::vector<ContinuousDensityFrame>& frames) {
  json out = json::array();
  for (const auto& frame : frames) {
    out.push_back(continuous_density_frame_to_json(frame));
  }
  return out;
}

// Serialize one Lindblad density-matrix frame.
json lindblad_frame_to_json(const LindbladFrame& frame) {
  json out{
      {"step", frame.step},
      {"time", frame.time},
      {"purity", frame.purity},
      {"entropy", frame.entropy},
      {"coherence", frame.coherence},
      {"populations", frame.populations},
  };
  return out;
}

// Serialize all Lindblad frames to a JSON array.
json lindblad_frames_to_json(const std::vector<LindbladFrame>& frames) {
  json out = json::array();
  for (const auto& frame : frames) {
    out.push_back(lindblad_frame_to_json(frame));
  }
  return out;
}

// Serialize one WignerFrame to JSON. Large payload arrays are emitted conditionally.
json wigner_frame_to_json(const WignerFrame& frame) {
  json out{
      {"step", frame.step},
      {"time", frame.time},
      {"nx", frame.nx},
      {"np", frame.np},
      {"x_min", frame.x_min},
      {"x_max", frame.x_max},
      {"p_min", frame.p_min},
      {"p_max", frame.p_max},
      {"transmission", frame.transmission},
      {"reflection", frame.reflection},
      {"px_mean", frame.px_mean},
      {"py_mean", frame.py_mean},
      {"kinetic_energy", frame.kinetic_energy},
  };
  if (!frame.wigner_flat.empty()) {
    out["wigner_flat"] = frame.wigner_flat;
  }
  if (!frame.current_x_flat.empty()) {
    out["current_x_flat"] = frame.current_x_flat;
  }
  if (!frame.current_y_flat.empty()) {
    out["current_y_flat"] = frame.current_y_flat;
  }
  return out;
}

// Serialize one MPS diagnostics frame to JSON.
json mps_frame_to_json(const MpsFrame& f) {
  json out{{"step", f.step}, {"time", f.time}, {"n_sites", f.n_sites},
           {"max_bond_dim", f.max_bond_dim}, {"truncation_error", f.truncation_error}};
  if (!f.bond_dims.empty())     out["bond_dims"]      = f.bond_dims;
  if (!f.bond_entropies.empty()) out["bond_entropies"] = f.bond_entropies;
  if (!f.magnetization.empty())  out["magnetization"]  = f.magnetization;
  return out;
}

// Serialize all MPS frames to a JSON array.
json mps_frames_to_json(const std::vector<MpsFrame>& frames) {
  json out = json::array();
  for (const auto& f : frames) out.push_back(mps_frame_to_json(f));
  return out;
}

// Serialize all Wigner frames to a JSON array.
json wigner_frames_to_json(const std::vector<WignerFrame>& frames) {
  json out = json::array();
  for (const auto& frame : frames) {
    out.push_back(wigner_frame_to_json(frame));
  }
  return out;
}

// Serialize one entanglement frame to JSON.
json entanglement_frame_to_json(const EntanglementFrame& frame) {
  json out{{"step", frame.step},    {"time", frame.time},
           {"n_a", frame.n_a},      {"n_b", frame.n_b},
           {"entropy_a", frame.entropy_a}, {"mutual_info", frame.mutual_info},
           {"concurrence", frame.concurrence},
           {"log_negativity", frame.log_negativity}};
  if (!frame.schmidt_values.empty()) {
    out["schmidt_values"] = frame.schmidt_values;
  }
  return out;
}

// Serialize all entanglement frames to a JSON array.
json entanglement_frames_to_json(const std::vector<EntanglementFrame>& frames) {
  json out = json::array();
  for (const auto& f : frames) out.push_back(entanglement_frame_to_json(f));
  return out;
}

// Select a stride-sampled view of the continuous frame list for the public export wrapper.
std::vector<ContinuousDensityFrame> sample_continuous_export_frames(
    const std::vector<ContinuousDensityFrame>& frames,
    std::uint32_t stride) {
  if (stride <= 1U || frames.empty()) {
    return frames;
  }

  std::vector<ContinuousDensityFrame> sampled;
  sampled.reserve(frames.size() / stride + 2U);
  for (std::size_t i = 0; i < frames.size(); ++i) {
    const bool keep = (i == 0U) || (i + 1U == frames.size()) || (frames[i].step % stride == 0U);
    if (keep) {
      sampled.push_back(frames[i]);
    }
  }
  return sampled;
}

// Serialize the stable continuous-export contract metadata.
json continuous_export_config_to_json(const ContinuousExportConfig& config) {
  const ContinuousExportConfig sanitized = sanitize_continuous_export_config_impl(config);
  return json{{"schema_version", sanitized.schema_version},
              {"export_frame_stride", sanitized.export_frame_stride},
              {"export_points_max", sanitized.export_points_max},
              {"export_density_floor_ratio", sanitized.export_density_floor_ratio}};
}

// Serialize circuit capability metadata to a JSON object.
json circuit_capabilities_to_json_object(const CircuitCapabilities& capabilities) {
  return json{
      {"backend", capabilities.backend},
      {"supported_gates", capabilities.supported_gates},
      {"supported_noise_models", capabilities.supported_noise_models},
      {"max_qubits", capabilities.max_qubits},
      {"max_shots", capabilities.max_shots},
      {"supports_qiskit_like", capabilities.supports_qiskit_like},
      {"supports_cirq_like", capabilities.supports_cirq_like},
  };
}
}  // namespace

// Parse simulation problem JSON into IR.
QuantumProblem problem_from_json(const std::string& json_text) {
  const json j = parse_json_or_throw(json_text);

  QuantumProblem p;
  p.kind = problem_kind_from_string(get_or<std::string>(j, "kind", problem_kind_to_string(p.kind)));
  p.qubits = get_or<std::uint32_t>(j, "qubits", p.qubits);
  p.equation = equation_from_string(get_or<std::string>(j, "equation", equation_to_string(p.equation)));
  p.evolution.total_time = get_or<double>(j, "total_time", p.evolution.total_time);
  p.evolution.dt = get_or<double>(j, "dt", p.evolution.dt);
  p.evolution.method = get_or<std::string>(j, "method", p.evolution.method);
  p.evolution.krylov_dim = get_or<std::uint32_t>(j, "krylov_dim", p.evolution.krylov_dim);
  p.evolution.mps_max_bond = get_or<std::uint32_t>(j, "mps_max_bond", p.evolution.mps_max_bond);
  p.evolution.mps_trunc_tol = get_or<double>(j, "mps_trunc_tol", p.evolution.mps_trunc_tol);
  p.target_error = get_or<double>(j, "target_error", p.target_error);
  p.noisy = get_or<bool>(j, "noisy", p.noisy);
  p.expected_circuit_depth = get_or<std::uint32_t>(j, "expected_circuit_depth", p.expected_circuit_depth);
  p.observables = get_or<std::vector<std::string>>(j, "observables", p.observables);

  auto c2d_it = j.find("continuous_space_2d");
  if (c2d_it != j.end() && c2d_it->is_object()) {
    const auto& c2d = *c2d_it;
    p.continuous_2d.nx = get_or<std::uint32_t>(c2d, "nx", p.continuous_2d.nx);
    p.continuous_2d.ny = get_or<std::uint32_t>(c2d, "ny", p.continuous_2d.ny);
    p.continuous_2d.dx = get_or<double>(c2d, "dx", p.continuous_2d.dx);
    p.continuous_2d.dy = get_or<double>(c2d, "dy", p.continuous_2d.dy);
    p.continuous_2d.dt = get_or<double>(c2d, "dt", p.continuous_2d.dt);
    p.continuous_2d.steps = get_or<std::uint32_t>(c2d, "steps", p.continuous_2d.steps);
    p.continuous_2d.boundary = get_or<std::string>(c2d, "boundary", p.continuous_2d.boundary);
    p.continuous_2d.dispersion = get_or<std::string>(c2d, "dispersion", p.continuous_2d.dispersion);
    p.continuous_2d.mass = get_or<double>(c2d, "mass", p.continuous_2d.mass);
    p.continuous_2d.speed_of_light = get_or<double>(c2d, "speed_of_light", p.continuous_2d.speed_of_light);
    p.continuous_2d.potential_kind = get_or<std::string>(c2d, "potential_kind", p.continuous_2d.potential_kind);
    p.continuous_2d.potential_params =
        get_or<std::map<std::string, double>>(c2d, "potential_params", p.continuous_2d.potential_params);
    p.continuous_2d.potential_grid = get_or<std::vector<double>>(c2d, "potential_grid", p.continuous_2d.potential_grid);
    parse_driving_config(c2d, p.continuous_2d.driving);
    parse_initial_state_config(c2d, p.continuous_2d.initial_state);
    p.kind = ProblemKind::ContinuousSpace2D;
  }

  auto c3d_it = j.find("continuous_space_3d");
  if (c3d_it != j.end() && c3d_it->is_object()) {
    const auto& c3d = *c3d_it;
    p.continuous_3d.nx = get_or<std::uint32_t>(c3d, "nx", p.continuous_3d.nx);
    p.continuous_3d.ny = get_or<std::uint32_t>(c3d, "ny", p.continuous_3d.ny);
    p.continuous_3d.nz = get_or<std::uint32_t>(c3d, "nz", p.continuous_3d.nz);
    p.continuous_3d.dx = get_or<double>(c3d, "dx", p.continuous_3d.dx);
    p.continuous_3d.dy = get_or<double>(c3d, "dy", p.continuous_3d.dy);
    p.continuous_3d.dz = get_or<double>(c3d, "dz", p.continuous_3d.dz);
    p.continuous_3d.dt = get_or<double>(c3d, "dt", p.continuous_3d.dt);
    p.continuous_3d.steps = get_or<std::uint32_t>(c3d, "steps", p.continuous_3d.steps);
    p.continuous_3d.boundary = get_or<std::string>(c3d, "boundary", p.continuous_3d.boundary);
    p.continuous_3d.dispersion = get_or<std::string>(c3d, "dispersion", p.continuous_3d.dispersion);
    p.continuous_3d.mass = get_or<double>(c3d, "mass", p.continuous_3d.mass);
    p.continuous_3d.speed_of_light = get_or<double>(c3d, "speed_of_light", p.continuous_3d.speed_of_light);
    p.continuous_3d.potential_kind = get_or<std::string>(c3d, "potential_kind", p.continuous_3d.potential_kind);
    p.continuous_3d.potential_params =
        get_or<std::map<std::string, double>>(c3d, "potential_params", p.continuous_3d.potential_params);
    p.continuous_3d.potential_grid = get_or<std::vector<double>>(c3d, "potential_grid", p.continuous_3d.potential_grid);
    parse_driving_config(c3d, p.continuous_3d.driving);
    parse_initial_state_config(c3d, p.continuous_3d.initial_state);
    p.kind = ProblemKind::ContinuousSpace3D;
  }

  auto h_it = j.find("hamiltonian_terms");
  if (h_it != j.end() && h_it->is_array()) {
    p.hamiltonian_terms.clear();
    for (std::size_t i = 0; i < h_it->size(); ++i) {
      const auto& item = h_it->at(i);
      if (!item.is_object()) {
        continue;
      }
      const HamiltonianTerm t = parse_hamiltonian_term(item);
      if (!t.label.empty()) {
        p.hamiltonian_terms.push_back(t);
      }
    }
  }
  return p;
}

// Parse runtime options JSON.
RunOptions options_from_json(const std::string& json_text) {
  const json j = parse_json_or_throw(json_text);
  RunOptions o;
  o.use_gpu = get_or<bool>(j, "use_gpu", o.use_gpu);
  o.allow_mpi = get_or<bool>(j, "allow_mpi", o.allow_mpi);
  o.enable_checkpoint = get_or<bool>(j, "enable_checkpoint", o.enable_checkpoint);
  o.export_continuous_frames = get_or<bool>(j, "export_continuous_frames", o.export_continuous_frames);
  o.export_wigner = get_or<bool>(j, "export_wigner", o.export_wigner);
  o.entanglement_cut = get_or<std::uint32_t>(j, "entanglement_cut", o.entanglement_cut);
  o.checkpoint_path = get_or<std::string>(j, "checkpoint_path", o.checkpoint_path);
  o.resources.max_threads = get_or<std::uint32_t>(j, "max_threads", o.resources.max_threads);
  o.resources.max_gpus = get_or<std::uint32_t>(j, "max_gpus", o.resources.max_gpus);
  o.resources.mpi_ranks = get_or<std::uint32_t>(j, "mpi_ranks", o.resources.mpi_ranks);
  o.resources.max_memory_mb = get_or<std::uint64_t>(j, "max_memory_mb", o.resources.max_memory_mb);
  ContinuousExportConfig export_config;
  export_config.export_frame_stride = read_clamped_u32(j, "export_frame_stride", export_config.export_frame_stride, 1U);
  export_config.export_points_max = read_clamped_size(j, "export_points_max", export_config.export_points_max, 1U);
  export_config.export_density_floor_ratio =
      read_clamped_ratio(j, "export_density_floor_ratio", export_config.export_density_floor_ratio);
  export_config = sanitize_continuous_export_config_impl(export_config);
  o.export_frame_stride = export_config.export_frame_stride;
  o.export_points_max = export_config.export_points_max;
  o.export_density_floor_ratio = export_config.export_density_floor_ratio;
  return o;
}

// Parse a continuous-export contract snapshot and normalize append-only fields.
ContinuousExportConfig continuous_export_config_from_json(const std::string& json_text) {
  const json j = parse_json_or_throw(json_text);
  return parse_continuous_export_config_impl(j);
}

// Clamp continuous-export contract fields to stable append-only bounds.
ContinuousExportConfig sanitize_continuous_export_config(const ContinuousExportConfig& config) {
  return sanitize_continuous_export_config_impl(config);
}

// Parse circuit JSON and enforce at least one operation.
QuantumCircuit circuit_from_json(const std::string& json_text) {
  const json j = parse_json_or_throw(json_text);
  QuantumCircuit c;
  c.qubits = get_or<std::uint32_t>(j, "qubits", c.qubits);
  c.observables = get_or<std::vector<std::string>>(j, "observables", c.observables);
  c.shots = get_or<std::uint32_t>(j, "shots", c.shots);
  c.depolarizing_prob = get_or<double>(j, "depolarizing_prob", c.depolarizing_prob);
  c.amplitude_damping_gamma = get_or<double>(j, "amplitude_damping_gamma", c.amplitude_damping_gamma);
  c.seed = get_or<std::uint64_t>(j, "seed", c.seed);

  auto ops_it = j.find("operations");
  if (ops_it != j.end() && ops_it->is_array()) {
    c.operations.clear();
    for (std::size_t i = 0; i < ops_it->size(); ++i) {
      const auto& item = ops_it->at(i);
      if (!item.is_object()) {
        continue;
      }
      CircuitOp op = parse_circuit_op(item);
      if (!op.name.empty()) {
        c.operations.push_back(std::move(op));
      }
    }
  }
  c.clbits = get_or<std::uint32_t>(j, "clbits", c.clbits);
  c.parameter_map = get_or<std::map<std::string, double>>(j, "parameter_map", c.parameter_map);
  parse_noise_config(j, c);
  if (c.operations.empty()) {
    throw std::runtime_error("circuit must contain at least one operation");
  }
  return c;
}

// Serialize Hamiltonian simulation result as compact JSON.
std::string result_to_json(const SimulationResult& result) {
  json j;
  j["ok"] = result.ok;
  j["backend"] = result.backend;
  j["elapsed_ms"] = result.elapsed_ms;
  j["message"] = result.message;
  j["state_size"] = result.state.size();
  j["engine_version"] = version_string();
  j["observables"] = result.observables;
  j["continuous_density_frames"] = continuous_density_frames_to_json(result.continuous_density_frames);
  j["mps_frames"] = mps_frames_to_json(result.mps_frames);
  j["lindblad_frames"] = lindblad_frames_to_json(result.lindblad_frames);
  j["wigner_frames"] = wigner_frames_to_json(result.wigner_frames);
  j["entanglement_frames"] = entanglement_frames_to_json(result.entanglement_frames);
  const ContinuousExportConfig export_config = sanitize_continuous_export_config_impl(result.continuous_export);
  j["continuous_export"] = {
      {"schema_version", export_config.schema_version},
      {"export_config", continuous_export_config_to_json(export_config)},
      {"frames",
       continuous_density_frames_to_json(
           sample_continuous_export_frames(result.continuous_density_frames,
                                           export_config.export_frame_stride))},
  };
  j["events"] = events_to_json(result.events);
  return j.dump();
}

// Serialize circuit execution result as compact JSON.
std::string circuit_result_to_json(const CircuitResult& result) {
  json j;
  j["ok"] = result.ok;
  j["backend"] = result.backend;
  j["elapsed_ms"] = result.elapsed_ms;
  j["message"] = result.message;
  j["state_size"] = result.state.size();
  j["engine_version"] = version_string();
  // Structured metadata block for framework adapters; mirrors top-level fields for
  // backward compat but groups them under a stable key for transpiler integration.
  j["metadata"] = {
      {"engine_version", version_string()},
      {"backend", result.backend},
      {"elapsed_ms", result.elapsed_ms},
  };
  j["counts"] = result.counts;
  j["expectation_exact"] = result.expectation_exact;
  j["expectation_shot"] = result.expectation_shot;
  j["estimation_error"] = result.estimation_error;
  j["shot_variance"] = result.shot_variance;
  j["shot_stderr"] = result.shot_stderr;
  j["error_summary"] = result.error_summary;
  if (!result.measurement_results.empty()) {
    j["measurement_results"] = result.measurement_results;
  }
  return j.dump();
}

// Serialize circuit capability metadata as compact JSON.
std::string circuit_capabilities_to_json(const CircuitCapabilities& capabilities) {
  json j = circuit_capabilities_to_json_object(capabilities);
  j["engine_version"] = version_string();
  return j.dump();
}

// Serialize backend explanation as compact JSON.
std::string backend_info_to_json(const BackendInfo& info) {
  json j;
  j["selected_backend"] = info.selected_backend;
  j["candidates"] = info.candidates;
  j["reasons"] = info.reasons;
  j["engine_version"] = version_string();
  return j.dump();
}

// ── VQE serialization helpers ────────────────────────────────────────────────

static json vqe_frame_to_json_obj(const VqeFrame& f) {
  json j;
  j["iteration"]     = f.iteration;
  j["energy"]        = f.energy;
  j["gradient_norm"] = f.gradient_norm;
  j["accepted"]      = f.accepted;
  j["params"]        = f.params;
  return j;
}

// Parse VQE options from JSON.
VqeOptions vqe_options_from_json(const std::string& json_text) {
  const json j = parse_json_or_throw(json_text);
  VqeOptions o;
  o.n_qubits       = get_or<std::uint32_t>(j, "n_qubits",       o.n_qubits);
  o.ansatz_type    = get_or<std::string>  (j, "ansatz_type",    o.ansatz_type);
  o.n_layers       = get_or<std::uint32_t>(j, "n_layers",       o.n_layers);
  o.optimizer      = get_or<std::string>  (j, "optimizer",      o.optimizer);
  o.learning_rate  = get_or<double>       (j, "learning_rate",  o.learning_rate);
  o.adam_beta1     = get_or<double>       (j, "adam_beta1",     o.adam_beta1);
  o.adam_beta2     = get_or<double>       (j, "adam_beta2",     o.adam_beta2);
  o.adam_epsilon   = get_or<double>       (j, "adam_epsilon",   o.adam_epsilon);
  o.max_iterations = get_or<std::uint32_t>(j, "max_iterations", o.max_iterations);
  o.convergence_tol= get_or<double>       (j, "convergence_tol",o.convergence_tol);
  o.seed           = get_or<std::uint64_t>(j, "seed",           o.seed);
  o.initial_params = get_or<std::vector<double>>(j, "initial_params", o.initial_params);

  auto ht_it = j.find("cost_hamiltonian");
  if (ht_it != j.end() && ht_it->is_array()) {
    for (const auto& item : *ht_it) {
      if (item.is_object()) {
        o.cost_hamiltonian.push_back(parse_hamiltonian_term(item));
      }
    }
  }

  auto edges_it = j.find("qaoa_edges");
  if (edges_it != j.end() && edges_it->is_array()) {
    for (const auto& e : *edges_it) {
      if (e.is_array() && e.size() == 2) {
        o.qaoa_edges.emplace_back(e[0].get<std::uint32_t>(), e[1].get<std::uint32_t>());
      }
    }
  }
  return o;
}

// Parse QPE options from JSON.
QpeOptions qpe_options_from_json(const std::string& json_text) {
  const json j = parse_json_or_throw(json_text);
  QpeOptions o;
  o.n_ancilla      = get_or<std::uint32_t>(j, "n_ancilla",      o.n_ancilla);
  o.n_target       = get_or<std::uint32_t>(j, "n_target",       o.n_target);
  o.unitary_type   = get_or<std::string>  (j, "unitary_type",   o.unitary_type);
  o.phase_angle    = get_or<double>       (j, "phase_angle",    o.phase_angle);
  o.evolution_time = get_or<double>       (j, "evolution_time", o.evolution_time);
  o.n_shots        = get_or<std::uint32_t>(j, "n_shots",        o.n_shots);
  o.seed           = get_or<std::uint64_t>(j, "seed",           o.seed);
  auto ht_it = j.find("hamiltonian_terms");
  if (ht_it != j.end() && ht_it->is_array()) {
    for (const auto& item : *ht_it) {
      if (item.is_object()) {
        o.hamiltonian_terms.push_back(parse_hamiltonian_term(item));
      }
    }
  }
  return o;
}

// Serialize QPE result.
std::string qpe_result_to_json(const QpeResult& result) {
  json j;
  j["ok"]                 = result.ok;
  j["n_ancilla"]          = result.n_ancilla;
  j["n_target"]           = result.n_target;
  j["estimated_phase"]    = result.estimated_phase;
  j["true_phase"]         = result.true_phase;
  j["elapsed_ms"]         = result.elapsed_ms;
  j["message"]            = result.message;
  j["engine_version"]     = version_string();
  j["measurement_counts"] = result.measurement_counts;
  j["phase_probs"]        = result.phase_probs;
  return j.dump();
}

// Serialize VQE result.
std::string vqe_result_to_json(const VqeResult& result) {
  json j;
  j["ok"]             = result.ok;
  j["converged"]      = result.converged;
  j["n_iterations"]   = result.n_iterations;
  j["optimal_energy"] = result.optimal_energy;
  j["optimal_params"] = result.optimal_params;
  j["elapsed_ms"]     = result.elapsed_ms;
  j["message"]        = result.message;
  j["engine_version"] = version_string();
  json frames = json::array();
  for (const auto& f : result.frames) {
    frames.push_back(vqe_frame_to_json_obj(f));
  }
  j["frames"] = frames;
  return j.dump();
}

// ── M22: QEC JSON ────────────────────────────────────────────────────────────

// Parse QEC options from JSON.
QecOptions qec_options_from_json(const std::string& json_text) {
  const json j = parse_json_or_throw(json_text);
  QecOptions o;
  o.code                = get_or<std::string>  (j, "code",                o.code);
  o.error_type          = get_or<std::string>  (j, "error_type",          o.error_type);
  o.physical_error_rate = get_or<double>       (j, "physical_error_rate", o.physical_error_rate);
  o.n_trials            = get_or<std::uint32_t>(j, "n_trials",            o.n_trials);
  o.n_frames_max        = get_or<std::uint32_t>(j, "n_frames_max",        o.n_frames_max);
  o.seed                = get_or<std::uint64_t>(j, "seed",                o.seed);
  auto sw_it = j.find("sweep_error_rates");
  if (sw_it != j.end() && sw_it->is_array()) {
    for (const auto& v : *sw_it) {
      if (v.is_number()) o.sweep_error_rates.push_back(v.get<double>());
    }
  }
  return o;
}

// Serialize QEC result to JSON.
std::string qec_result_to_json(const QecResult& result) {
  json j;
  j["ok"]                         = result.ok;
  j["code"]                        = result.code;
  j["n_physical"]                  = result.n_physical;
  j["n_syndrome"]                  = result.n_syndrome;
  j["logical_error_rate"]          = result.logical_error_rate;
  j["logical_error_rate_theory"]   = result.logical_error_rate_theory;
  j["elapsed_ms"]                  = result.elapsed_ms;
  j["message"]                     = result.message;
  j["engine_version"]              = version_string();

  json frames = json::array();
  for (const auto& f : result.frames) {
    json fj;
    fj["trial"]           = f.trial;
    fj["errors_injected"] = f.errors_injected;
    fj["syndrome"]        = f.syndrome;
    fj["correction"]      = f.correction;
    fj["succeeded"]       = f.succeeded;
    frames.push_back(fj);
  }
  j["frames"] = frames;

  json tc = json::array();
  for (const auto& pt : result.threshold_curve) {
    json ptj;
    ptj["p_physical"]       = pt.p_physical;
    ptj["p_logical_mc"]     = pt.p_logical_mc;
    ptj["p_logical_theory"] = pt.p_logical_theory;
    tc.push_back(ptj);
  }
  j["threshold_curve"] = tc;

  return j.dump();
}

// ── M23: Schwinger model (1+1D QED) ──────────────────────────────────────────

QfieldOptions qfield_options_from_json(const std::string& json_text) {
  auto j = json::parse(json_text);
  QfieldOptions opts;
  if (j.contains("n_sites"))      opts.n_sites     = j["n_sites"].get<std::uint32_t>();
  if (j.contains("mass"))         opts.mass        = j["mass"].get<double>();
  if (j.contains("coupling"))     opts.coupling    = j["coupling"].get<double>();
  if (j.contains("background"))   opts.background  = j["background"].get<double>();
  if (j.contains("scenario"))     opts.scenario    = j["scenario"].get<std::string>();
  if (j.contains("dt"))           opts.dt          = j["dt"].get<double>();
  if (j.contains("n_steps"))      opts.n_steps     = j["n_steps"].get<std::uint32_t>();
  if (j.contains("n_mass_sweep")) opts.n_mass_sweep = j["n_mass_sweep"].get<std::uint32_t>();
  if (j.contains("seed"))         opts.seed        = j["seed"].get<std::uint64_t>();
  return opts;
}

std::string qfield_result_to_json(const QfieldResult& result) {
  json j;
  j["ok"]                = result.ok;
  j["n_sites"]           = result.n_sites;
  j["mass"]              = result.mass;
  j["coupling"]          = result.coupling;
  j["ground_energy"]     = result.ground_energy;
  j["chiral_condensate"] = result.chiral_condensate;
  j["elapsed_ms"]        = result.elapsed_ms;
  j["message"]           = result.message;
  j["engine_version"]    = version_string();

  // Dynamics frames.
  json frames = json::array();
  for (const auto& f : result.frames) {
    json fj;
    fj["time"]              = f.time;
    fj["charge_density"]    = f.charge_density;
    fj["electric_field"]    = f.electric_field;
    fj["chiral_condensate"] = f.chiral_condensate;
    fj["energy"]            = f.energy;
    fj["particle_number"]   = f.particle_number;
    frames.push_back(fj);
  }
  j["frames"] = frames;

  // Mass sweep.
  json ms = json::array();
  for (const auto& pt : result.mass_sweep) {
    ms.push_back({pt[0], pt[1], pt[2]});
  }
  j["mass_sweep"] = ms;

  // String energies.
  json se = json::array();
  for (const auto& pt : result.string_energies) {
    se.push_back({pt[0], pt[1]});
  }
  j["string_energies"] = se;

  return j.dump();
}

// ── M24: φ⁴ scalar QFT ───────────────────────────────────────────────────────

Phi4Options phi4_options_from_json(const std::string& json_text) {
  auto j = json::parse(json_text);
  Phi4Options opts;
  if (j.contains("n_sites"))         opts.n_sites          = j["n_sites"].get<std::uint32_t>();
  if (j.contains("m2"))              opts.m2               = j["m2"].get<double>();
  if (j.contains("lambda_coupling")) opts.lambda_coupling  = j["lambda_coupling"].get<double>();
  if (j.contains("kappa"))           opts.kappa            = j["kappa"].get<double>();
  if (j.contains("d_local"))         opts.d_local          = j["d_local"].get<std::uint32_t>();
  if (j.contains("scenario"))        opts.scenario         = j["scenario"].get<std::string>();
  if (j.contains("n_sweep"))         opts.n_sweep          = j["n_sweep"].get<std::uint32_t>();
  if (j.contains("m2_min"))          opts.m2_min           = j["m2_min"].get<double>();
  if (j.contains("m2_max"))          opts.m2_max           = j["m2_max"].get<double>();
  if (j.contains("epsilon"))         opts.epsilon          = j["epsilon"].get<double>();
  if (j.contains("seed"))            opts.seed             = j["seed"].get<std::uint64_t>();
  if (j.contains("m2_initial"))      opts.m2_initial       = j["m2_initial"].get<double>();
  if (j.contains("m2_final"))        opts.m2_final         = j["m2_final"].get<double>();
  if (j.contains("dt"))              opts.dt               = j["dt"].get<double>();
  if (j.contains("n_steps"))         opts.n_steps          = j["n_steps"].get<std::uint32_t>();
  return opts;
}

std::string phi4_result_to_json(const Phi4Result& result) {
  json j;
  j["ok"]               = result.ok;
  j["n_sites"]          = result.n_sites;
  j["d_local"]          = result.d_local;
  j["m2"]               = result.m2;
  j["lambda_coupling"]  = result.lambda_coupling;
  j["ground_energy"]    = result.ground_energy;
  j["energy_gap"]       = result.energy_gap;
  j["order_parameter"]  = result.order_parameter;
  j["phi_expectation"]  = result.phi_expectation;
  j["phi2_expectation"] = result.phi2_expectation;
  j["occupation"]       = result.occupation;
  j["field_distribution"] = result.field_distribution;
  j["elapsed_ms"]       = result.elapsed_ms;
  j["message"]          = result.message;
  j["engine_version"]   = version_string();

  json sw = json::array();
  for (const auto& pt : result.sweep) {
    json ptj;
    ptj["m2"]              = pt.m2;
    ptj["ground_energy"]   = pt.ground_energy;
    ptj["order_parameter"] = pt.order_parameter;
    ptj["energy_gap"]      = pt.energy_gap;
    sw.push_back(ptj);
  }
  j["sweep"] = sw;

  json dyn = json::array();
  for (const auto& f : result.dynamics) {
    json fj;
    fj["time"]        = f.time;
    fj["phi_expect"]  = f.phi_expect;
    fj["occupation"]  = f.occupation;
    fj["order_param"] = f.order_param;
    fj["entropy"]     = f.entropy;
    fj["energy"]      = f.energy;
    dyn.push_back(fj);
  }
  j["dynamics"] = dyn;

  return j.dump();
}

// ── M27-C: Hubbard model ─────────────────────────────────────────────────────

HubbardOptions hubbard_options_from_json(const std::string& json_text) {
  auto j = json::parse(json_text);
  HubbardOptions o;
  if (j.contains("n_sites"))  o.n_sites  = j["n_sites"].get<std::uint32_t>();
  if (j.contains("t_hop"))    o.t_hop    = j["t_hop"].get<double>();
  if (j.contains("U"))        o.U        = j["U"].get<double>();
  if (j.contains("n_up"))     o.n_up     = j["n_up"].get<std::uint32_t>();
  if (j.contains("n_dn"))     o.n_dn     = j["n_dn"].get<std::uint32_t>();
  if (j.contains("scenario")) o.scenario = j["scenario"].get<std::string>();
  if (j.contains("n_sweep"))  o.n_sweep  = j["n_sweep"].get<std::uint32_t>();
  if (j.contains("U_min"))    o.U_min    = j["U_min"].get<double>();
  if (j.contains("U_max"))    o.U_max    = j["U_max"].get<double>();
  return o;
}

std::string hubbard_result_to_json(const HubbardResult& r) {
  json j;
  j["ok"]                 = r.ok;
  j["n_sites"]            = r.n_sites;
  j["n_up"]               = r.n_up;
  j["n_dn"]               = r.n_dn;
  j["t_hop"]              = r.t_hop;
  j["U"]                  = r.U;
  j["ground_energy"]      = r.ground_energy;
  j["energy_gap"]         = r.energy_gap;
  j["double_occupancy"]   = r.double_occupancy;
  j["spin_structure_pi"]  = r.spin_structure_pi;
  j["spin_corr"]          = r.spin_corr;
  j["momentum_dist"]      = r.momentum_dist;
  j["elapsed_ms"]         = r.elapsed_ms;
  j["message"]            = r.message;
  j["engine_version"]     = version_string();

  json sw = json::array();
  for (const auto& pt : r.sweep) {
    json p;
    p["U"]                 = pt.U;
    p["ground_energy"]     = pt.ground_energy;
    p["double_occupancy"]  = pt.double_occupancy;
    p["spin_structure_pi"] = pt.spin_structure_pi;
    p["energy_gap"]        = pt.energy_gap;
    sw.push_back(p);
  }
  j["sweep"] = sw;
  return j.dump();
}

// ── M27-D: Floquet model ──────────────────────────────────────────────────────

FloquetOptions floquet_options_from_json(const std::string& json_text) {
  auto j = json::parse(json_text);
  FloquetOptions o;
  if (j.contains("n_sites"))     o.n_sites     = j["n_sites"].get<std::uint32_t>();
  if (j.contains("J"))           o.J           = j["J"].get<double>();
  if (j.contains("h_z"))         o.h_z         = j["h_z"].get<double>();
  if (j.contains("drive_amp"))   o.drive_amp   = j["drive_amp"].get<double>();
  if (j.contains("drive_omega")) o.drive_omega = j["drive_omega"].get<double>();
  if (j.contains("n_periods"))   o.n_periods   = j["n_periods"].get<std::uint32_t>();
  if (j.contains("scenario"))    o.scenario    = j["scenario"].get<std::string>();
  if (j.contains("n_scan"))      o.n_scan      = j["n_scan"].get<std::uint32_t>();
  if (j.contains("scan_amp_max")) o.scan_amp_max = j["scan_amp_max"].get<double>();
  return o;
}

std::string floquet_result_to_json(const FloquetResult& r) {
  json j;
  j["ok"]             = r.ok;
  j["n_sites"]        = r.n_sites;
  j["J"]              = r.J;
  j["h_z"]            = r.h_z;
  j["drive_amp"]      = r.drive_amp;
  j["drive_omega"]    = r.drive_omega;
  j["quasi_energies"] = r.quasi_energies;
  j["mean_gap_ratio"] = r.mean_gap_ratio;
  j["elapsed_ms"]     = r.elapsed_ms;
  j["message"]        = r.message;
  j["engine_version"] = version_string();

  json fr = json::array();
  for (const auto& f : r.frames) {
    json fj;
    fj["period"]         = f.period;
    fj["time"]           = f.time;
    fj["magnetization_z"] = f.magnetization_z;
    fj["energy"]         = f.energy;
    fj["entropy"]        = f.entropy;
    fr.push_back(fj);
  }
  j["frames"] = fr;

  json sa = json::array();
  json sq = json::array();
  for (std::size_t i = 0; i < r.scan_amps.size(); ++i) {
    sa.push_back(r.scan_amps[i]);
    sq.push_back(r.scan_qe[i]);
  }
  j["scan_amps"] = sa;
  j["scan_qe"]   = sq;

  return j.dump();
}

// ── M28-E: Heisenberg chain ───────────────────────────────────────────────

HeisenbergOptions heisenberg_options_from_json(const std::string& json_text) {
  auto j = json::parse(json_text);
  HeisenbergOptions o;
  if (j.contains("n_sites"))   o.n_sites   = j["n_sites"].get<std::uint32_t>();
  if (j.contains("J"))         o.J         = j["J"].get<double>();
  if (j.contains("Jz"))        o.Jz        = j["Jz"].get<double>();
  if (j.contains("h"))         o.h         = j["h"].get<double>();
  if (j.contains("scenario"))  o.scenario  = j["scenario"].get<std::string>();
  if (j.contains("n_sweep"))   o.n_sweep   = j["n_sweep"].get<std::uint32_t>();
  if (j.contains("Jz_min"))    o.Jz_min    = j["Jz_min"].get<double>();
  if (j.contains("Jz_max"))    o.Jz_max    = j["Jz_max"].get<double>();
  if (j.contains("h_initial")) o.h_initial = j["h_initial"].get<double>();
  if (j.contains("dt"))        o.dt        = j["dt"].get<double>();
  if (j.contains("n_steps"))   o.n_steps   = j["n_steps"].get<std::uint32_t>();
  return o;
}

std::string heisenberg_result_to_json(const HeisenbergResult& r) {
  json j;
  j["ok"]             = r.ok;
  j["n_sites"]        = r.n_sites;
  j["J"]              = r.J;
  j["Jz"]             = r.Jz;
  j["h"]              = r.h;
  j["ground_energy"]  = r.ground_energy;
  j["energy_gap"]     = r.energy_gap;
  j["structure_pi"]   = r.structure_pi;
  j["spin_corr"]      = r.spin_corr;
  j["Sz_site"]        = r.Sz_site;
  j["elapsed_ms"]     = r.elapsed_ms;
  j["message"]        = r.message;
  j["engine_version"] = version_string();

  json sw = json::array();
  for (const auto& pt : r.sweep) {
    json pj;
    pj["Jz"]            = pt.Jz;
    pj["ground_energy"] = pt.ground_energy;
    pj["energy_gap"]    = pt.energy_gap;
    pj["Sz_total"]      = pt.Sz_total;
    pj["structure_pi"]  = pt.structure_pi;
    sw.push_back(pj);
  }
  j["sweep"] = sw;

  json fr = json::array();
  for (const auto& f : r.frames) {
    json fj;
    fj["time"]     = f.time;
    fj["Sz_total"] = f.Sz_total;
    fj["energy"]   = f.energy;
    fj["entropy"]  = f.entropy;
    fj["Sz_site"]  = f.Sz_site;
    fr.push_back(fj);
  }
  j["frames"] = fr;

  return j.dump();
}

// ── M28-F: Bose-Hubbard model ─────────────────────────────────────────────

BoseHubbardOptions bose_hubbard_options_from_json(const std::string& json_text) {
  auto j = json::parse(json_text);
  BoseHubbardOptions o;
  if (j.contains("n_sites"))  o.n_sites  = j["n_sites"].get<std::uint32_t>();
  if (j.contains("n_bosons")) o.n_bosons = j["n_bosons"].get<std::uint32_t>();
  if (j.contains("n_max"))    o.n_max    = j["n_max"].get<std::uint32_t>();
  if (j.contains("J"))        o.J        = j["J"].get<double>();
  if (j.contains("U"))        o.U        = j["U"].get<double>();
  if (j.contains("mu"))       o.mu       = j["mu"].get<double>();
  if (j.contains("scenario")) o.scenario = j["scenario"].get<std::string>();
  if (j.contains("n_sweep"))  o.n_sweep  = j["n_sweep"].get<std::uint32_t>();
  if (j.contains("J_min"))    o.J_min    = j["J_min"].get<double>();
  if (j.contains("J_max"))    o.J_max    = j["J_max"].get<double>();
  return o;
}

std::string bose_hubbard_result_to_json(const BoseHubbardResult& r) {
  json j;
  j["ok"]               = r.ok;
  j["n_sites"]          = r.n_sites;
  j["n_bosons"]         = r.n_bosons;
  j["n_max"]            = r.n_max;
  j["J"]                = r.J;
  j["U"]                = r.U;
  j["ground_energy"]    = r.ground_energy;
  j["energy_gap"]       = r.energy_gap;
  j["superfluid_op"]    = r.superfluid_op;
  j["double_occupancy"] = r.double_occupancy;
  j["occupation"]       = r.occupation;
  j["number_dist"]      = r.number_dist;
  j["elapsed_ms"]       = r.elapsed_ms;
  j["message"]          = r.message;
  j["engine_version"]   = version_string();

  json sw = json::array();
  for (const auto& pt : r.sweep) {
    json pj;
    pj["J"]                = pt.J;
    pj["ground_energy"]    = pt.ground_energy;
    pj["energy_gap"]       = pt.energy_gap;
    pj["superfluid_op"]    = pt.superfluid_op;
    pj["double_occupancy"] = pt.double_occupancy;
    sw.push_back(pj);
  }
  j["sweep"] = sw;

  return j.dump();
}

// ── M29-G: Transverse-Field Ising ────────────────────────────────────────────

TFIOptions tfi_options_from_json(const std::string& json_text) {
  auto j = json::parse(json_text);
  TFIOptions o;
  if (j.contains("n_sites"))  o.n_sites  = j["n_sites"].get<int>();
  if (j.contains("J"))        o.J        = j["J"].get<double>();
  if (j.contains("h"))        o.h        = j["h"].get<double>();
  if (j.contains("scenario")) o.scenario = j["scenario"].get<std::string>();
  if (j.contains("n_sweep"))  o.n_sweep  = j["n_sweep"].get<int>();
  if (j.contains("h_min"))    o.h_min    = j["h_min"].get<double>();
  if (j.contains("h_max"))    o.h_max    = j["h_max"].get<double>();
  if (j.contains("h_quench")) o.h_quench = j["h_quench"].get<double>();
  if (j.contains("dt"))       o.dt       = j["dt"].get<double>();
  if (j.contains("n_steps"))  o.n_steps  = j["n_steps"].get<int>();
  return o;
}

std::string tfi_result_to_json(const TFIResult& r) {
  json j;
  j["ok"]            = r.ok;
  j["n_sites"]       = r.n_sites;
  j["J"]             = r.J;
  j["h"]             = r.h;
  j["ground_energy"] = r.ground_energy;
  j["energy_gap"]    = r.energy_gap;
  j["order_param"]   = r.order_param;
  j["corr_len"]      = r.corr_len;
  j["spin_corr"]     = r.spin_corr;
  j["elapsed_ms"]    = r.elapsed_ms;
  j["message"]       = r.message;
  j["engine_version"] = version_string();

  json sw = json::array();
  for (const auto& pt : r.sweep) {
    json pj;
    pj["h"]            = pt.h;
    pj["ground_energy"]= pt.ground_energy;
    pj["energy_gap"]   = pt.energy_gap;
    pj["order_param"]  = pt.order_param;
    pj["corr_len"]     = pt.corr_len;
    sw.push_back(pj);
  }
  j["sweep"] = sw;

  json fr = json::array();
  for (const auto& f : r.frames) {
    json fj;
    fj["time"]        = f.time;
    fj["h_now"]       = f.h_now;
    fj["order_param"] = f.order_param;
    fj["energy"]      = f.energy;
    fj["entropy"]     = f.entropy;
    fr.push_back(fj);
  }
  j["frames"] = fr;

  return j.dump();
}

// ── M29-H: SSH Model ─────────────────────────────────────────────────────────

SSHOptions ssh_options_from_json(const std::string& json_text) {
  auto j = json::parse(json_text);
  SSHOptions o;
  if (j.contains("n_cells"))   o.n_cells   = j["n_cells"].get<int>();
  if (j.contains("t"))         o.t         = j["t"].get<double>();
  if (j.contains("delta"))     o.delta     = j["delta"].get<double>();
  if (j.contains("scenario"))  o.scenario  = j["scenario"].get<std::string>();
  if (j.contains("n_sweep"))   o.n_sweep   = j["n_sweep"].get<int>();
  if (j.contains("delta_min")) o.delta_min = j["delta_min"].get<double>();
  if (j.contains("delta_max")) o.delta_max = j["delta_max"].get<double>();
  return o;
}

std::string ssh_result_to_json(const SSHResult& r) {
  json j;
  j["ok"]          = r.ok;
  j["n_cells"]     = r.n_cells;
  j["t"]           = r.t;
  j["delta"]       = r.delta;
  j["energies"]    = r.energies;
  j["gap"]         = r.gap;
  j["zak_phase"]   = r.zak_phase;
  j["edge_weight"] = r.edge_weight;
  j["elapsed_ms"]  = r.elapsed_ms;
  j["message"]     = r.message;
  j["engine_version"] = version_string();

  json sw = json::array();
  for (const auto& pt : r.sweep) {
    json pj;
    pj["delta"]       = pt.delta;
    pj["gap"]         = pt.gap;
    pj["zak_phase"]   = pt.zak_phase;
    pj["edge_weight"] = pt.edge_weight;
    sw.push_back(pj);
  }
  j["sweep"] = sw;

  return j.dump();
}

// ── M30-I: Kitaev Chain ───────────────────────────────────────────────────────

KitaevOptions kitaev_options_from_json(const std::string& json_text) {
  auto j = json::parse(json_text);
  KitaevOptions o;
  if (j.contains("n_sites"))  o.n_sites  = j["n_sites"].get<int>();
  if (j.contains("t"))        o.t        = j["t"].get<double>();
  if (j.contains("delta"))    o.delta    = j["delta"].get<double>();
  if (j.contains("mu"))       o.mu       = j["mu"].get<double>();
  if (j.contains("scenario")) o.scenario = j["scenario"].get<std::string>();
  if (j.contains("n_sweep"))  o.n_sweep  = j["n_sweep"].get<int>();
  if (j.contains("mu_min"))   o.mu_min   = j["mu_min"].get<double>();
  if (j.contains("mu_max"))   o.mu_max   = j["mu_max"].get<double>();
  return o;
}

std::string kitaev_result_to_json(const KitaevResult& r) {
  json j;
  j["ok"]              = r.ok;
  j["n_sites"]         = r.n_sites;
  j["t"]               = r.t;
  j["delta"]           = r.delta;
  j["mu"]              = r.mu;
  j["energies"]        = r.energies;
  j["gap"]             = r.gap;
  j["topo_inv"]        = r.topo_inv;
  j["majorana_weight"] = r.majorana_weight;
  j["elapsed_ms"]      = r.elapsed_ms;
  j["message"]         = r.message;
  j["engine_version"]  = version_string();

  json sw = json::array();
  for (const auto& pt : r.sweep) {
    json pj;
    pj["mu"]              = pt.mu;
    pj["gap"]             = pt.gap;
    pj["topo_inv"]        = pt.topo_inv;
    pj["majorana_weight"] = pt.majorana_weight;
    sw.push_back(pj);
  }
  j["sweep"] = sw;
  return j.dump();
}

// ── M30-J: Anderson Localization ─────────────────────────────────────────────

AndersonOptions anderson_options_from_json(const std::string& json_text) {
  auto j = json::parse(json_text);
  AndersonOptions o;
  if (j.contains("n_sites"))   o.n_sites   = j["n_sites"].get<int>();
  if (j.contains("t"))         o.t         = j["t"].get<double>();
  if (j.contains("W"))         o.W         = j["W"].get<double>();
  if (j.contains("n_samples")) o.n_samples = j["n_samples"].get<int>();
  if (j.contains("scenario"))  o.scenario  = j["scenario"].get<std::string>();
  if (j.contains("n_sweep"))   o.n_sweep   = j["n_sweep"].get<int>();
  if (j.contains("W_min"))     o.W_min     = j["W_min"].get<double>();
  if (j.contains("W_max"))     o.W_max     = j["W_max"].get<double>();
  if (j.contains("seed"))      o.seed      = j["seed"].get<unsigned int>();
  return o;
}

std::string anderson_result_to_json(const AndersonResult& r) {
  json j;
  j["ok"]           = r.ok;
  j["n_sites"]      = r.n_sites;
  j["t"]            = r.t;
  j["W"]            = r.W;
  j["mean_ipr"]     = r.mean_ipr;
  j["mean_r"]       = r.mean_r;
  j["loc_length"]   = r.loc_length;
  j["ipr_spectrum"] = r.ipr_spectrum;
  j["energies"]     = r.energies;
  j["elapsed_ms"]   = r.elapsed_ms;
  j["message"]      = r.message;
  j["engine_version"] = version_string();

  json sw = json::array();
  for (const auto& pt : r.sweep) {
    json pj;
    pj["W"]          = pt.W;
    pj["mean_ipr"]   = pt.mean_ipr;
    pj["mean_r"]     = pt.mean_r;
    pj["loc_length"] = pt.loc_length;
    sw.push_back(pj);
  }
  j["sweep"] = sw;
  return j.dump();
}

// ── M31-K: 2D Chern Insulator ─────────────────────────────────────────────

ChernOptions chern_options_from_json(const std::string& json_str) {
  auto j = json::parse(json_str);
  ChernOptions o;
  if (j.contains("n_sites"))  o.n_sites  = j["n_sites"].get<int>();
  if (j.contains("m"))        o.m        = j["m"].get<double>();
  if (j.contains("t"))        o.t        = j["t"].get<double>();
  if (j.contains("scenario")) o.scenario = j["scenario"].get<std::string>();
  if (j.contains("n_k"))      o.n_k      = j["n_k"].get<int>();
  if (j.contains("n_sweep"))  o.n_sweep  = j["n_sweep"].get<int>();
  if (j.contains("m_min"))    o.m_min    = j["m_min"].get<double>();
  if (j.contains("m_max"))    o.m_max    = j["m_max"].get<double>();
  return o;
}

std::string chern_result_to_json(const ChernResult& r) {
  json j;
  j["ok"]          = r.ok;
  j["n_sites"]     = r.n_sites;
  j["m"]           = r.m;
  j["t"]           = r.t;
  j["scenario"]    = r.scenario;
  j["chern"]       = r.chern;
  j["gap"]         = r.gap;
  j["elapsed_ms"]  = r.elapsed_ms;
  j["message"]     = r.message;
  j["engine_version"] = version_string();
  j["lower_band"]  = r.lower_band;
  j["upper_band"]  = r.upper_band;
  j["berry_curv"]  = r.berry_curv;
  j["kx_vals"]     = r.kx_vals;
  json es = json::array();
  for (const auto& sp : r.edge_spectrum) es.push_back(sp);
  j["edge_spectrum"] = es;
  json sw = json::array();
  for (const auto& pt : r.sweep) {
    json pj;
    pj["m"]     = pt.m;
    pj["gap"]   = pt.gap;
    pj["chern"] = pt.chern;
    sw.push_back(pj);
  }
  j["sweep"] = sw;
  return j.dump();
}

// ── M31-L: Hofstadter Butterfly ───────────────────────────────────────────

HofstadterOptions hofstadter_options_from_json(const std::string& json_str) {
  auto j = json::parse(json_str);
  HofstadterOptions o;
  if (j.contains("t"))        o.t        = j["t"].get<double>();
  if (j.contains("scenario")) o.scenario = j["scenario"].get<std::string>();
  if (j.contains("phi"))      o.phi      = j["phi"].get<double>();
  if (j.contains("q_max"))    o.q_max    = j["q_max"].get<int>();
  if (j.contains("n_k"))      o.n_k      = j["n_k"].get<int>();
  if (j.contains("n_ky"))     o.n_ky     = j["n_ky"].get<int>();
  return o;
}

std::string hofstadter_result_to_json(const HofstadterResult& r) {
  json j;
  j["ok"]         = r.ok;
  j["elapsed_ms"] = r.elapsed_ms;
  j["message"]    = r.message;
  j["engine_version"] = version_string();
  j["phi_vals"]   = r.phi_vals;
  j["evals"]      = r.evals;
  j["phi_used"]   = r.phi_used;
  j["n_bands"]    = r.n_bands;
  j["kx_vals"]    = r.kx_vals;
  json bd = json::array();
  for (const auto& band_row : r.bands) bd.push_back(band_row);
  j["bands"] = bd;
  return j.dump();
}

// ── M32: MBL ─────────────────────────────────────────────────────────────────

MBLOptions mbl_options_from_json(const std::string& json_str) {
  auto j = json::parse(json_str);
  MBLOptions o;
  if (j.contains("n_sites"))   o.n_sites   = j["n_sites"].get<int>();
  if (j.contains("J"))         o.J         = j["J"].get<double>();
  if (j.contains("delta"))     o.delta     = j["delta"].get<double>();
  if (j.contains("W"))         o.W         = j["W"].get<double>();
  if (j.contains("scenario"))  o.scenario  = j["scenario"].get<std::string>();
  if (j.contains("W_min"))     o.W_min     = j["W_min"].get<double>();
  if (j.contains("W_max"))     o.W_max     = j["W_max"].get<double>();
  if (j.contains("n_sweep"))   o.n_sweep   = j["n_sweep"].get<int>();
  if (j.contains("n_samples")) o.n_samples = j["n_samples"].get<int>();
  if (j.contains("seed"))      o.seed      = j["seed"].get<unsigned int>();
  return o;
}

std::string mbl_result_to_json(const MBLResult& r) {
  json j;
  j["ok"]           = r.ok;
  j["n_sites"]      = r.n_sites;
  j["J"]            = r.J;
  j["delta"]        = r.delta;
  j["W"]            = r.W;
  j["scenario"]     = r.scenario;
  j["evals"]        = r.evals;
  j["entropies"]    = r.entropies;
  j["iprs"]         = r.iprs;
  j["r_vals"]       = r.r_vals;
  j["entropy_mean"] = r.entropy_mean;
  j["r_mean"]       = r.r_mean;
  j["ipr_mean"]     = r.ipr_mean;
  j["elapsed_ms"]   = r.elapsed_ms;
  j["message"]      = r.message;
  j["engine_version"] = version_string();
  json sweep_arr = json::array();
  for (const auto& pt : r.sweep) {
    json sp;
    sp["W"]            = pt.W;
    sp["r_mean"]       = pt.r_mean;
    sp["entropy_mean"] = pt.entropy_mean;
    sp["ipr_mean"]     = pt.ipr_mean;
    sweep_arr.push_back(sp);
  }
  j["sweep"] = sweep_arr;
  return j.dump();
}

TenfoldOptions tenfold_options_from_json(const std::string& json_str) {
  auto j = json::parse(json_str);
  TenfoldOptions o;
  if (j.contains("az_class"))    o.az_class    = j["az_class"].get<std::string>();
  if (j.contains("dimension"))   o.dimension   = j["dimension"].get<int>();
  if (j.contains("model"))       o.model       = j["model"].get<std::string>();
  if (j.contains("scenario"))    o.scenario    = j["scenario"].get<std::string>();
  if (j.contains("t1"))          o.t1          = j["t1"].get<double>();
  if (j.contains("t2"))          o.t2          = j["t2"].get<double>();
  if (j.contains("m"))           o.m           = j["m"].get<double>();
  if (j.contains("t"))           o.t           = j["t"].get<double>();
  if (j.contains("t_hop"))       o.t_hop       = j["t_hop"].get<double>();
  if (j.contains("mu"))          o.mu          = j["mu"].get<double>();
  if (j.contains("n_k"))         o.n_k         = j["n_k"].get<int>();
  if (j.contains("param_min"))   o.param_min   = j["param_min"].get<double>();
  if (j.contains("param_max"))   o.param_max   = j["param_max"].get<double>();
  if (j.contains("n_sweep"))     o.n_sweep     = j["n_sweep"].get<int>();
  if (j.contains("sweep_param")) o.sweep_param = j["sweep_param"].get<std::string>();
  return o;
}

std::string tenfold_result_to_json(const TenfoldResult& r) {
  json j;
  j["ok"]              = r.ok;
  j["az_class"]        = r.az_class;
  j["dimension"]       = r.dimension;
  j["invariant_type"]  = r.invariant_type;
  j["scenario"]        = r.scenario;
  j["model"]           = r.model;
  j["invariant_value"] = r.invariant_value;
  j["invariant_z2"]    = r.invariant_z2;
  j["phase_label"]     = r.phase_label;
  j["k_bands"]         = r.k_bands;
  j["e_plus"]          = r.e_plus;
  j["e_minus"]         = r.e_minus;
  j["sweep_vals"]      = r.sweep_vals;
  j["sweep_invariant"] = r.sweep_invariant;
  j["sweep_z2"]        = r.sweep_z2;
  j["sweep_phase"]     = r.sweep_phase;
  j["elapsed_ms"]      = r.elapsed_ms;
  j["message"]         = r.message;
  j["engine_version"]  = version_string();
  return j.dump();
}

SecondChernOptions second_chern_options_from_json(const std::string& json_str) {
  auto j = json::parse(json_str);
  SecondChernOptions o;
  if (j.contains("m"))           o.m           = j["m"].get<double>();
  if (j.contains("t"))           o.t           = j["t"].get<double>();
  if (j.contains("n_k"))         o.n_k         = j["n_k"].get<int>();
  if (j.contains("scenario"))    o.scenario    = j["scenario"].get<std::string>();
  if (j.contains("param_min"))   o.param_min   = j["param_min"].get<double>();
  if (j.contains("param_max"))   o.param_max   = j["param_max"].get<double>();
  if (j.contains("n_sweep"))     o.n_sweep     = j["n_sweep"].get<int>();
  if (j.contains("sweep_param")) o.sweep_param = j["sweep_param"].get<std::string>();
  return o;
}

std::string second_chern_result_to_json(const SecondChernResult& r) {
  json j;
  j["ok"]           = r.ok;
  j["m"]            = r.m;
  j["t"]            = r.t;
  j["scenario"]     = r.scenario;
  j["second_chern"] = r.second_chern;
  j["phase_label"]  = r.phase_label;
  j["sweep_vals"]   = r.sweep_vals;
  j["sweep_c2"]     = r.sweep_c2;
  j["sweep_phase"]  = r.sweep_phase;
  j["elapsed_ms"]   = r.elapsed_ms;
  j["message"]      = r.message;
  j["engine_version"] = version_string();
  return j.dump();
}

WilsonLoopOptions wilson_loop_options_from_json(const std::string& json_str) {
  auto j = json::parse(json_str);
  WilsonLoopOptions o;
  if (j.contains("M"))           o.M           = j["M"].get<double>();
  if (j.contains("A"))           o.A           = j["A"].get<double>();
  if (j.contains("B"))           o.B           = j["B"].get<double>();
  if (j.contains("C"))           o.C           = j["C"].get<double>();
  if (j.contains("D"))           o.D           = j["D"].get<double>();
  if (j.contains("n_kx"))        o.n_kx        = j["n_kx"].get<int>();
  if (j.contains("n_ky"))        o.n_ky        = j["n_ky"].get<int>();
  if (j.contains("scenario"))    o.scenario    = j["scenario"].get<std::string>();
  if (j.contains("sweep_param")) o.sweep_param = j["sweep_param"].get<std::string>();
  if (j.contains("param_min"))   o.param_min   = j["param_min"].get<double>();
  if (j.contains("param_max"))   o.param_max   = j["param_max"].get<double>();
  if (j.contains("n_sweep"))     o.n_sweep     = j["n_sweep"].get<int>();
  return o;
}

std::string wilson_loop_result_to_json(const WilsonLoopResult& r) {
  json j;
  j["ok"]          = r.ok;
  j["M"]           = r.M;
  j["A"]           = r.A;
  j["B"]           = r.B;
  j["scenario"]    = r.scenario;
  j["z2"]          = r.z2;
  j["phase_label"] = r.phase_label;
  j["gap_gamma"]   = r.gap_gamma;
  j["gap_m_pt"]    = r.gap_m_pt;
  j["ky_vals"]     = r.ky_vals;
  j["wcc1"]        = r.wcc1;
  j["wcc2"]        = r.wcc2;
  j["sweep_vals"]  = r.sweep_vals;
  j["sweep_z2"]    = r.sweep_z2;
  j["sweep_phase"] = r.sweep_phase;
  j["elapsed_ms"]  = r.elapsed_ms;
  j["message"]     = r.message;
  j["engine_version"] = version_string();
  return j.dump();
}

EdgeStatesOptions edge_states_options_from_json(const std::string& json_str) {
  auto j = json::parse(json_str);
  EdgeStatesOptions o;
  if (j.contains("M"))           o.M           = j["M"].get<double>();
  if (j.contains("A"))           o.A           = j["A"].get<double>();
  if (j.contains("B"))           o.B           = j["B"].get<double>();
  if (j.contains("C"))           o.C           = j["C"].get<double>();
  if (j.contains("D"))           o.D           = j["D"].get<double>();
  if (j.contains("n_kx"))        o.n_kx        = j["n_kx"].get<int>();
  if (j.contains("n_y"))         o.n_y         = j["n_y"].get<int>();
  if (j.contains("scenario"))    o.scenario    = j["scenario"].get<std::string>();
  if (j.contains("sweep_param")) o.sweep_param = j["sweep_param"].get<std::string>();
  if (j.contains("param_min"))   o.param_min   = j["param_min"].get<double>();
  if (j.contains("param_max"))   o.param_max   = j["param_max"].get<double>();
  if (j.contains("n_sweep"))     o.n_sweep     = j["n_sweep"].get<int>();
  return o;
}

std::string edge_states_result_to_json(const EdgeStatesResult& r) {
  json j;
  j["ok"]          = r.ok;
  j["M"]           = r.M;
  j["A"]           = r.A;
  j["B"]           = r.B;
  j["n_y"]         = r.n_y;
  j["n_bands"]     = r.n_bands;
  j["scenario"]    = r.scenario;
  j["z2"]          = r.z2;
  j["phase_label"] = r.phase_label;
  j["kx_vals"]     = r.kx_vals;
  j["energies"]    = r.energies;
  j["edge_weight"] = r.edge_weight;
  j["sweep_vals"]  = r.sweep_vals;
  j["sweep_z2"]    = r.sweep_z2;
  j["sweep_phase"] = r.sweep_phase;
  j["elapsed_ms"]  = r.elapsed_ms;
  j["message"]     = r.message;
  j["engine_version"] = version_string();
  return j.dump();
}

SpectralOptions spectral_options_from_json(const std::string& json_str) {
  auto j = json::parse(json_str);
  SpectralOptions o;
  if (j.contains("n_sites"))   o.n_sites   = j["n_sites"].get<int>();
  if (j.contains("J"))         o.J         = j["J"].get<double>();
  if (j.contains("delta"))     o.delta     = j["delta"].get<double>();
  if (j.contains("h"))         o.h         = j["h"].get<double>();
  if (j.contains("scenario"))  o.scenario  = j["scenario"].get<std::string>();
  if (j.contains("n_k"))       o.n_k       = j["n_k"].get<int>();
  if (j.contains("n_omega"))   o.n_omega   = j["n_omega"].get<int>();
  if (j.contains("omega_min")) o.omega_min = j["omega_min"].get<double>();
  if (j.contains("omega_max")) o.omega_max = j["omega_max"].get<double>();
  if (j.contains("eta"))       o.eta       = j["eta"].get<double>();
  if (j.contains("tau_max"))   o.tau_max   = j["tau_max"].get<double>();
  if (j.contains("n_tau"))     o.n_tau     = j["n_tau"].get<int>();
  return o;
}

std::string spectral_result_to_json(const SpectralResult& r) {
  json j;
  j["ok"]           = r.ok;
  j["n_sites"]      = r.n_sites;
  j["J"]            = r.J;
  j["delta"]        = r.delta;
  j["h"]            = r.h;
  j["scenario"]     = r.scenario;
  j["E_ground"]     = r.E_ground;
  j["k_vals"]       = r.k_vals;
  j["omega_vals"]   = r.omega_vals;
  j["spectral"]     = r.spectral;
  j["tau_vals"]     = r.tau_vals;
  j["energy_tau"]   = r.energy_tau;
  j["fidelity_tau"] = r.fidelity_tau;
  j["elapsed_ms"]   = r.elapsed_ms;
  j["message"]      = r.message;
  j["engine_version"] = version_string();
  return j.dump();
}

HaldaneOptions haldane_options_from_json(const std::string& s) {
  auto j = json::parse(s);
  HaldaneOptions o;
  if (j.contains("t1"))          o.t1          = j["t1"].get<double>();
  if (j.contains("t2"))          o.t2          = j["t2"].get<double>();
  if (j.contains("phi"))         o.phi         = j["phi"].get<double>();
  if (j.contains("M"))           o.M_val       = j["M"].get<double>();
  if (j.contains("n_k"))         o.n_k         = j["n_k"].get<int>();
  if (j.contains("n_y"))         o.n_y         = j["n_y"].get<int>();
  if (j.contains("scenario"))    o.scenario    = j["scenario"].get<std::string>();
  if (j.contains("sweep_param")) o.sweep_param = j["sweep_param"].get<std::string>();
  if (j.contains("param_min"))   o.param_min   = j["param_min"].get<double>();
  if (j.contains("param_max"))   o.param_max   = j["param_max"].get<double>();
  if (j.contains("n_sweep"))     o.n_sweep     = j["n_sweep"].get<int>();
  return o;
}

std::string haldane_result_to_json(const HaldaneResult& r) {
  json j;
  j["ok"]           = r.ok;
  j["t1"]           = r.t1;
  j["t2"]           = r.t2;
  j["phi"]          = r.phi;
  j["M"]            = r.M_val;
  j["n_k"]          = r.n_k;
  j["scenario"]     = r.scenario;
  j["chern"]        = r.chern;
  j["phase_label"]  = r.phase_label;
  j["kx_bz"]        = r.kx_bz;
  j["ky_bz"]        = r.ky_bz;
  j["berry_curv"]   = r.berry_curv;
  j["sweep_vals"]   = r.sweep_vals;
  j["sweep_chern"]  = r.sweep_chern;
  j["sweep_phase"]  = r.sweep_phase;
  j["elapsed_ms"]   = r.elapsed_ms;
  j["message"]      = r.message;
  j["engine_version"] = version_string();
  return j.dump();
}

WeylOptions weyl_options_from_json(const std::string& s) {
  auto j = json::parse(s);
  WeylOptions o;
  if (j.contains("m"))            o.m_param      = j["m"].get<double>();
  if (j.contains("n_k"))          o.n_k          = j["n_k"].get<int>();
  if (j.contains("n_z"))          o.n_z          = j["n_z"].get<int>();
  if (j.contains("n_kz"))         o.n_kz         = j["n_kz"].get<int>();
  if (j.contains("fermi_energy")) o.fermi_energy = j["fermi_energy"].get<double>();
  if (j.contains("scenario"))     o.scenario     = j["scenario"].get<std::string>();
  if (j.contains("param_min"))    o.param_min    = j["param_min"].get<double>();
  if (j.contains("param_max"))    o.param_max    = j["param_max"].get<double>();
  if (j.contains("n_sweep"))      o.n_sweep      = j["n_sweep"].get<int>();
  return o;
}

std::string weyl_result_to_json(const WeylResult& r) {
  json j;
  j["ok"]            = r.ok;
  j["m"]             = r.m_param;
  j["n_k"]           = r.n_k;
  j["n_z"]           = r.n_z;
  j["scenario"]      = r.scenario;
  j["node_kz_pos"]   = r.node_kz_pos;
  j["node_kz_neg"]   = r.node_kz_neg;
  j["kz_vals"]       = r.kz_vals;
  j["chern_kz"]      = r.chern_kz;
  j["arc_kx"]        = r.arc_kx;
  j["arc_ky"]        = r.arc_ky;
  j["arc_weight"]    = r.arc_weight;
  j["arc_energy"]    = r.arc_energy;
  j["n_arc_pts"]     = r.n_arc_pts;
  j["cone_kz"]       = r.cone_kz;
  j["cone_E_plus"]   = r.cone_E_plus;
  j["cone_E_minus"]  = r.cone_E_minus;
  j["sweep_vals"]    = r.sweep_vals;
  j["sweep_node_kz"] = r.sweep_node_kz;
  j["sweep_phase"]   = r.sweep_phase;
  j["elapsed_ms"]    = r.elapsed_ms;
  j["message"]       = r.message;
  j["engine_version"] = version_string();
  return j.dump();
}



FuKaneOptions fu_kane_options_from_json(const std::string& s) {
  auto j = json::parse(s);
  FuKaneOptions o;
  if (j.contains("v_F"))      o.v_F      = j["v_F"].get<double>();
  if (j.contains("M"))        o.M_val    = j["M"].get<double>();
  if (j.contains("mu"))       o.mu       = j["mu"].get<double>();
  if (j.contains("Delta"))    o.Delta    = j["Delta"].get<double>();
  if (j.contains("n_k"))      o.n_k      = j["n_k"].get<int>();
  if (j.contains("n_y"))      o.n_y      = j["n_y"].get<int>();
  if (j.contains("n_kx"))     o.n_kx     = j["n_kx"].get<int>();
  if (j.contains("n_M"))      o.n_M      = j["n_M"].get<int>();
  if (j.contains("n_mu"))     o.n_mu     = j["n_mu"].get<int>();
  if (j.contains("M_max"))    o.M_max    = j["M_max"].get<double>();
  if (j.contains("mu_max"))   o.mu_max   = j["mu_max"].get<double>();
  if (j.contains("scenario")) o.scenario = j["scenario"].get<std::string>();
  return o;
}

std::string fu_kane_result_to_json(const FuKaneResult& r) {
  json j;
  j["ok"]           = r.ok;
  j["scenario"]     = r.scenario;
  j["v_F"]          = r.v_F;
  j["M"]            = r.M_val;
  j["mu"]           = r.mu;
  j["Delta"]        = r.Delta;
  j["chern"]        = r.chern;
  j["phase_label"]  = r.phase_label;
  j["gap"]          = r.gap;
  j["M_critical"]   = r.M_critical;
  j["kx_bulk"]      = r.kx_bulk;
  j["E0"]           = r.E0;
  j["E1"]           = r.E1;
  j["E2"]           = r.E2;
  j["E3"]           = r.E3;
  j["kx_edge"]      = r.kx_edge;
  j["E_edge"]       = r.E_edge;
  j["n_edge_evals"] = r.n_edge_evals;
  j["pd_M_vals"]    = r.pd_M_vals;
  j["pd_mu_vals"]   = r.pd_mu_vals;
  j["pd_chern"]     = r.pd_chern;
  j["elapsed_ms"]   = r.elapsed_ms;
  j["message"]      = r.message;
  j["engine_version"] = version_string();
  return j.dump();
}

LutchynOregOptions lutchyn_oreg_options_from_json(const std::string& s) {
  auto j = json::parse(s);
  LutchynOregOptions o;
  if (j.contains("t"))         o.t        = j["t"].get<double>();
  if (j.contains("alpha"))     o.alpha    = j["alpha"].get<double>();
  if (j.contains("mu"))        o.mu       = j["mu"].get<double>();
  if (j.contains("Delta"))     o.Delta    = j["Delta"].get<double>();
  if (j.contains("V_z"))       o.V_z      = j["V_z"].get<double>();
  if (j.contains("n_k"))       o.n_k      = j["n_k"].get<int>();
  if (j.contains("n_sites"))   o.n_sites  = j["n_sites"].get<int>();
  if (j.contains("n_Vz"))      o.n_Vz     = j["n_Vz"].get<int>();
  if (j.contains("n_mu"))      o.n_mu     = j["n_mu"].get<int>();
  if (j.contains("Vz_max"))    o.Vz_max   = j["Vz_max"].get<double>();
  if (j.contains("mu_max"))    o.mu_max   = j["mu_max"].get<double>();
  if (j.contains("scenario"))  o.scenario = j["scenario"].get<std::string>();
  return o;
}

std::string lutchyn_oreg_result_to_json(const LutchynOregResult& r) {
  json j;
  j["ok"]            = r.ok;
  j["scenario"]      = r.scenario;
  j["t"]             = r.t;
  j["alpha"]         = r.alpha;
  j["mu"]            = r.mu;
  j["Delta"]         = r.Delta;
  j["V_z"]           = r.V_z;
  j["z2"]            = r.z2;
  j["phase_label"]   = r.phase_label;
  j["gap"]           = r.gap;
  j["V_z_critical"]  = r.V_z_critical;
  j["k_bulk"]        = r.k_bulk;
  j["E0"]            = r.E0;
  j["E1"]            = r.E1;
  j["E2"]            = r.E2;
  j["E3"]            = r.E3;
  j["E_wire"]        = r.E_wire;
  j["ldos_left"]     = r.ldos_left;
  j["ldos_right"]    = r.ldos_right;
  j["osc_L"]         = r.osc_L;
  j["osc_gap"]       = r.osc_gap;
  j["pd_Vz_vals"]    = r.pd_Vz_vals;
  j["pd_mu_vals"]    = r.pd_mu_vals;
  j["pd_z2"]         = r.pd_z2;
  j["elapsed_ms"]    = r.elapsed_ms;
  j["message"]       = r.message;
  j["engine_version"] = version_string();
  return j.dump();
}

KitaevHcOptions kitaev_hc_options_from_json(const std::string& s) {
  auto j = json::parse(s);
  KitaevHcOptions o;
  if (j.contains("Jx"))       o.Jx       = j["Jx"].get<double>();
  if (j.contains("Jy"))       o.Jy       = j["Jy"].get<double>();
  if (j.contains("Jz"))       o.Jz       = j["Jz"].get<double>();
  if (j.contains("kappa"))    o.kappa    = j["kappa"].get<double>();
  if (j.contains("n_k"))      o.n_k      = j["n_k"].get<int>();
  if (j.contains("n_pd"))     o.n_pd     = j["n_pd"].get<int>();
  if (j.contains("scenario")) o.scenario = j["scenario"].get<std::string>();
  return o;
}

std::string kitaev_hc_result_to_json(const KitaevHcResult& r) {
  json j;
  j["ok"]          = r.ok;
  j["scenario"]    = r.scenario;
  j["Jx"]          = r.Jx;
  j["Jy"]          = r.Jy;
  j["Jz"]          = r.Jz;
  j["kappa"]       = r.kappa;
  j["phase_label"] = r.phase_label;
  j["gap"]         = r.gap;
  j["chern"]       = r.chern;
  j["kx_bz"]       = r.kx_bz;
  j["ky_bz"]       = r.ky_bz;
  j["E_plus"]      = r.E_plus;
  j["E_minus"]     = r.E_minus;
  j["pd_Jx"]       = r.pd_Jx;
  j["pd_Jy"]       = r.pd_Jy;
  j["pd_Jz"]       = r.pd_Jz;
  j["pd_phase"]    = r.pd_phase;
  j["pd_gap"]      = r.pd_gap;
  j["elapsed_ms"]  = r.elapsed_ms;
  j["message"]     = r.message;
  j["engine_version"] = version_string();
  return j.dump();
}

}  // namespace qx
