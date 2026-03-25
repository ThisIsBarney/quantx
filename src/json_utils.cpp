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

}  // namespace qx


