#include "qx/json_utils.hpp"
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
}

// Serialize events to JSON array.
json events_to_json(const std::vector<Event>& events) {
  json out = json::array();
  for (const auto& ev : events) {
    out.push_back({{"name", ev.name}});
  }
  return out;
}

// Serialize a continuous density frame metadata record.
json continuous_density_frame_to_json(const ContinuousDensityFrame& frame) {
  return json{
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
  };
}

// Serialize continuous density frame metadata to JSON array.
json continuous_density_frames_to_json(const std::vector<ContinuousDensityFrame>& frames) {
  json out = json::array();
  for (const auto& frame : frames) {
    out.push_back(continuous_density_frame_to_json(frame));
  }
  return out;
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
    for (const auto& item : *h_it) {
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
  o.checkpoint_path = get_or<std::string>(j, "checkpoint_path", o.checkpoint_path);
  o.resources.max_threads = get_or<std::uint32_t>(j, "max_threads", o.resources.max_threads);
  o.resources.max_gpus = get_or<std::uint32_t>(j, "max_gpus", o.resources.max_gpus);
  o.resources.mpi_ranks = get_or<std::uint32_t>(j, "mpi_ranks", o.resources.mpi_ranks);
  o.resources.max_memory_mb = get_or<std::uint64_t>(j, "max_memory_mb", o.resources.max_memory_mb);
  return o;
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
    for (const auto& item : *ops_it) {
      if (!item.is_object()) {
        continue;
      }
      CircuitOp op = parse_circuit_op(item);
      if (!op.name.empty()) {
        c.operations.push_back(std::move(op));
      }
    }
  }
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
  j["counts"] = result.counts;
  j["expectation_exact"] = result.expectation_exact;
  j["expectation_shot"] = result.expectation_shot;
  j["estimation_error"] = result.estimation_error;
  j["shot_variance"] = result.shot_variance;
  j["shot_stderr"] = result.shot_stderr;
  j["error_summary"] = result.error_summary;
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
