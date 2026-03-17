#include "qx/json_utils.hpp"

#include <regex>
#include <sstream>

namespace qx {
namespace {

std::string extract_string(const std::string& json, const std::string& key, const std::string& def) {
  std::regex rg("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"");
  std::smatch m;
  if (std::regex_search(json, m, rg) && m.size() > 1) {
    return m[1].str();
  }
  return def;
}

double extract_double(const std::string& json, const std::string& key, double def) {
  std::regex rg("\"" + key + "\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
  std::smatch m;
  if (std::regex_search(json, m, rg) && m.size() > 1) {
    return std::stod(m[1].str());
  }
  return def;
}

std::uint32_t extract_uint(const std::string& json, const std::string& key, std::uint32_t def) {
  std::regex rg("\"" + key + "\"\\s*:\\s*([0-9]+)");
  std::smatch m;
  if (std::regex_search(json, m, rg) && m.size() > 1) {
    return static_cast<std::uint32_t>(std::stoul(m[1].str()));
  }
  return def;
}

bool extract_bool(const std::string& json, const std::string& key, bool def) {
  std::regex rg("\"" + key + "\"\\s*:\\s*(true|false)");
  std::smatch m;
  if (std::regex_search(json, m, rg) && m.size() > 1) {
    return m[1].str() == "true";
  }
  return def;
}

std::vector<HamiltonianTerm> extract_hamiltonian_terms(const std::string& json) {
  std::vector<HamiltonianTerm> out;
  std::regex object_rg("\\{[^\\{\\}]*\\}");
  std::regex label_rg("\"label\"\\s*:\\s*\"([^\"]+)\"");
  std::regex coeff_rg("\"coefficient\"\\s*:\\s*(-?[0-9]+(?:\\.[0-9]+)?)");
  auto begin = std::sregex_iterator(json.begin(), json.end(), object_rg);
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end; ++it) {
    const std::string obj = (*it).str();
    std::smatch lm;
    std::smatch cm;
    if (!std::regex_search(obj, lm, label_rg) || !std::regex_search(obj, cm, coeff_rg)) {
      continue;
    }
    HamiltonianTerm t;
    t.label = lm[1].str();
    t.coefficient = std::stod(cm[1].str());
    out.push_back(t);
  }
  return out;
}

std::string escape(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (const char c : s) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  return out;
}

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

EquationType equation_from_string(const std::string& s) {
  if (s == "Lindblad") return EquationType::Lindblad;
  if (s == "Dirac") return EquationType::Dirac;
  if (s == "KleinGordon") return EquationType::KleinGordon;
  return EquationType::Schrodinger;
}

}  // namespace

QuantumProblem problem_from_json(const std::string& json) {
  QuantumProblem p;
  p.qubits = extract_uint(json, "qubits", p.qubits);
  p.equation = equation_from_string(extract_string(json, "equation", equation_to_string(p.equation)));
  p.evolution.total_time = extract_double(json, "total_time", p.evolution.total_time);
  p.evolution.dt = extract_double(json, "dt", p.evolution.dt);
  p.evolution.method = extract_string(json, "method", p.evolution.method);
  p.target_error = extract_double(json, "target_error", p.target_error);
  p.noisy = extract_bool(json, "noisy", p.noisy);
  p.expected_circuit_depth = extract_uint(json, "expected_circuit_depth", p.expected_circuit_depth);
  p.hamiltonian_terms = extract_hamiltonian_terms(json);
  return p;
}

RunOptions options_from_json(const std::string& json) {
  RunOptions o;
  o.use_gpu = extract_bool(json, "use_gpu", o.use_gpu);
  o.allow_mpi = extract_bool(json, "allow_mpi", o.allow_mpi);
  o.enable_checkpoint = extract_bool(json, "enable_checkpoint", o.enable_checkpoint);
  o.checkpoint_path = extract_string(json, "checkpoint_path", o.checkpoint_path);
  o.resources.max_threads = extract_uint(json, "max_threads", o.resources.max_threads);
  o.resources.max_gpus = extract_uint(json, "max_gpus", o.resources.max_gpus);
  o.resources.mpi_ranks = extract_uint(json, "mpi_ranks", o.resources.mpi_ranks);
  o.resources.max_memory_mb =
      static_cast<std::uint64_t>(extract_uint(json, "max_memory_mb", static_cast<std::uint32_t>(o.resources.max_memory_mb)));
  return o;
}

std::string result_to_json(const SimulationResult& result) {
  std::ostringstream os;
  os << "{";
  os << "\"ok\":" << (result.ok ? "true" : "false") << ",";
  os << "\"backend\":\"" << escape(result.backend) << "\",";
  os << "\"message\":\"" << escape(result.message) << "\",";
  os << "\"state_size\":" << result.state.size() << ",";
  os << "\"observables\":{";
  bool first = true;
  for (const auto& [k, v] : result.observables) {
    if (!first) os << ",";
    first = false;
    os << "\"" << escape(k) << "\":" << v;
  }
  os << "},";
  os << "\"events\":[";
  for (std::size_t i = 0; i < result.events.size(); ++i) {
    if (i > 0) os << ",";
    os << "{\"name\":\"" << escape(result.events[i].name) << "\"}";
  }
  os << "]";
  os << "}";
  return os.str();
}

std::string backend_info_to_json(const BackendInfo& info) {
  std::ostringstream os;
  os << "{";
  os << "\"selected_backend\":\"" << escape(info.selected_backend) << "\",";
  os << "\"candidates\":[";
  for (std::size_t i = 0; i < info.candidates.size(); ++i) {
    if (i > 0) os << ",";
    os << "\"" << escape(info.candidates[i]) << "\"";
  }
  os << "],";
  os << "\"reasons\":[";
  for (std::size_t i = 0; i < info.reasons.size(); ++i) {
    if (i > 0) os << ",";
    os << "\"" << escape(info.reasons[i]) << "\"";
  }
  os << "]";
  os << "}";
  return os.str();
}

}  // namespace qx
