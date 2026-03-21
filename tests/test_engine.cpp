#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numbers>
#include <sstream>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

#include <nlohmann/json.hpp>

#include "qx/fft_provider.hpp"
#include "qx/backends.hpp"
#include "qx/engine.hpp"
#include "qx/json_utils.hpp"
#include "qx/version.hpp"
#include "test_support.hpp"

namespace {

// Minimal assertion helper for smoke-test binary.
void expect(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    std::exit(1);
  }
}

// Write a UTF-8 text file for CLI smoke tests.
void write_text_file(const std::filesystem::path& path, const std::string& text) {
  std::ofstream os(path, std::ios::binary);
  if (!os.good()) {
    throw std::runtime_error("failed to write file: " + path.string());
  }
  os << text;
}

// Read a UTF-8 text file for CLI smoke tests.
std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream is(path, std::ios::binary);
  if (!is.good()) {
    throw std::runtime_error("failed to open file: " + path.string());
  }
  std::ostringstream os;
  os << is.rdbuf();
  return os.str();
}

// Wrap a filesystem path in quotes for a shell command.
std::string quote_path(const std::filesystem::path& path) {
  return "\"" + path.string() + "\"";
}

// Bind the C API entry points from the runtime DLL at test time.
struct CApiBindings {
  using RunCircuitFn = const char* (*)(const char*, const char*);
  using FreeStringFn = void (*)(const char*);

  HMODULE module{nullptr};
  RunCircuitFn run_circuit{nullptr};
  FreeStringFn free_string{nullptr};
};

// Load the DLL exports needed by the C API smoke checks.
const CApiBindings& c_api_bindings() {
  static const CApiBindings bindings = [] {
    const char* candidates[] = {"Debug\\quantx.dll", "quantx.dll"};
    HMODULE module = nullptr;
    for (const char* candidate : candidates) {
      module = LoadLibraryA(candidate);
      if (module != nullptr) {
        break;
      }
    }
    if (module == nullptr) {
      throw std::runtime_error("failed to load quantx.dll for C API smoke checks");
    }
    auto run_circuit = reinterpret_cast<CApiBindings::RunCircuitFn>(GetProcAddress(module, "qx_run_circuit"));
    auto free_string = reinterpret_cast<CApiBindings::FreeStringFn>(GetProcAddress(module, "qx_free_string"));
    if (run_circuit == nullptr || free_string == nullptr) {
      throw std::runtime_error("failed to resolve qx_run_circuit/qx_free_string from quantx.dll");
    }
    return CApiBindings{module, run_circuit, free_string};
  }();
  return bindings;
}

// Convert a heap-allocated C string JSON payload into a parsed JSON value.
nlohmann::json parse_owned_json(CApiBindings::RunCircuitFn fn, const char* circuit_json, const char* options_json) {
  const char* raw = fn(circuit_json, options_json);
  std::string text = raw ? raw : "";
  c_api_bindings().free_string(raw);
  return nlohmann::json::parse(text);
}

// Execute a command through the platform shell and return the raw exit code.
int run_command(const std::string& command) {
#ifdef _WIN32
  // Use CreateProcess instead of std::system so the smoke test stays shell-free and does not depend on cmd.exe parsing.
  std::vector<char> mutable_command(command.begin(), command.end());
  mutable_command.push_back('\0');

  STARTUPINFOA si{};
  si.cb = sizeof(si);
  PROCESS_INFORMATION pi{};
  if (!CreateProcessA(nullptr, mutable_command.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
    const DWORD err = GetLastError();
    std::cerr << "command launch failed (" << err << "): " << command << "\n";
    return 1;
  }

  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD exit_code = 1;
  if (!GetExitCodeProcess(pi.hProcess, &exit_code)) {
    exit_code = 1;
  }
  CloseHandle(pi.hProcess);
  CloseHandle(pi.hThread);
  if (exit_code != 0) {
    std::cerr << "command failed (" << exit_code << "): " << command << "\n";
  }
  return static_cast<int>(exit_code);
#else
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    std::cerr << "command failed (" << rc << "): " << command << "\n";
  }
  return rc;
#endif
}

// Locate qx_cli.exe across the supported local build layouts.
std::filesystem::path resolve_qx_cli_executable(const std::filesystem::path& repo_root) {
  const std::filesystem::path candidates[] = {
      repo_root / "build-nmake-local" / "qx_cli.exe",
      repo_root / "build-vs" / "Debug" / "qx_cli.exe",
      repo_root / "build-vs" / "qx_cli.exe",
      repo_root / "Debug" / "qx_cli.exe",
  };
  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  std::ostringstream msg;
  msg << "unable to locate qx_cli.exe; checked:";
  for (const auto& candidate : candidates) {
    msg << ' ' << candidate.string();
  }
  throw std::runtime_error(msg.str());
}

// Resolve the repository root by walking upward until the canonical top-level files appear.
std::filesystem::path resolve_repo_root(const std::filesystem::path& start_dir) {
  std::filesystem::path current = start_dir;
  for (int depth = 0; depth < 4; ++depth) {
    if (std::filesystem::exists(current / "examples") && std::filesystem::exists(current / "CMakeLists.txt")) {
      return current;
    }
    if (!current.has_parent_path()) {
      break;
    }
    current = current.parent_path();
  }
  throw std::runtime_error("unable to locate repository root from " + start_dir.string());
}

}  // namespace

// End-to-end smoke test for simulation and circuit APIs.
int main() {
  qx::test::configure_console_diagnostics();
  try {
    qx::QuantumProblem p;
  p.qubits = 8;
  p.evolution.total_time = 1.0;
  p.evolution.dt = 0.05;

  qx::RunOptions o;
  o.enable_checkpoint = false;

  const auto info = qx::explain_backend_choice(p, o);
  expect(info.selected_backend == "statevector", "small problem should choose statevector");

  const auto r = qx::run(p, o);
  expect(r.ok, "simulation should be ok");
  expect(r.backend == "statevector", "backend should be statevector");
  expect(r.elapsed_ms >= 0.0, "simulation elapsed_ms should be available");
  expect(r.observables.count("norm") == 1, "norm observable should exist");
  expect(std::abs(r.observables.at("norm") - 1.0) < 1e-8, "norm should be conserved");
  expect(!qx::version_string().empty(), "version string should not be empty");

  const std::string run_json = qx::result_to_json(r);
  expect(run_json.find("\"engine_version\"") != std::string::npos, "result json should include engine_version");

  p.qubits = 64;
  p.evolution.method = "tebd";
  p.evolution.mps_max_bond = 16;
  p.evolution.mps_trunc_tol = 1e-6;
  const auto info2 = qx::explain_backend_choice(p, o);
  expect(info2.selected_backend == "mps", "large problem should choose mps");
  const auto r2 = qx::run(p, o);
  expect(r2.backend == "mps", "backend should be mps");
  expect(r2.observables.count("bond_dim") == 1, "mps should report bond_dim");
  expect(r2.observables.count("truncation_error") == 1, "mps should report truncation_error");
  expect(r2.observables.count("mps_max_bond") == 1, "mps should report mps_max_bond");
  expect(r2.observables.count("mps_trunc_tol") == 1, "mps should report mps_trunc_tol");
  expect(r2.observables.count("entanglement_entropy_estimate") == 1, "mps should report entropy estimate");
  // Contract test: backend must honor user-provided bond cap in outputs.
  expect(r2.observables.at("bond_dim") <= 16.0 + 1e-9, "bond_dim should respect mps_max_bond");
  expect(std::abs(r2.observables.at("mps_max_bond") - 16.0) < 1e-9, "reported mps_max_bond should match input");
  expect(std::abs(r2.observables.at("mps_trunc_tol") - 1e-6) < 1e-12, "reported mps_trunc_tol should match input");
  expect(r2.observables.at("truncation_error") > 0.0, "truncation_error should accumulate");

  p.evolution.method = "tdvp";
  const auto r2_tdvp = qx::run(p, o);
  expect(r2_tdvp.backend == "mps", "tdvp large problem should run on mps");
  expect(r2_tdvp.observables.at("bond_dim") <= 16.0 + 1e-9, "tdvp bond_dim should respect mps_max_bond");
  // In our v1 surrogate model, tdvp is expected to accumulate less truncation than tebd.
  expect(r2_tdvp.observables.at("truncation_error") < r2.observables.at("truncation_error"),
         "tdvp should have smaller truncation_error than tebd in this surrogate model");

  // Regression for scientific notation parsing from JSON inputs.
  const auto parsed = qx::problem_from_json(R"({
    "qubits": 64,
    "method": "tebd",
    "mps_max_bond": 16,
    "mps_trunc_tol": 1e-6,
    "hamiltonian_terms": [{"label": "Z0", "coefficient": 1e-3}]
  })");
  expect(std::abs(parsed.evolution.mps_trunc_tol - 1e-6) < 1e-12, "json scientific notation should parse for mps_trunc_tol");
  expect(std::abs(parsed.hamiltonian_terms.at(0).coefficient - 1e-3) < 1e-12,
         "json scientific notation should parse for Hamiltonian coefficient");

  const auto parsed_options = qx::options_from_json(R"({
    "export_continuous_frames": true,
    "export_frame_stride": 5,
    "export_points_max": 42,
    "export_density_floor_ratio": 0.25
  })");
  expect(parsed_options.export_continuous_frames, "options json should parse export_continuous_frames");
  expect(parsed_options.export_frame_stride == 5U, "options json should parse export_frame_stride");
  expect(parsed_options.export_points_max == 42U, "options json should parse export_points_max");
  expect(std::abs(parsed_options.export_density_floor_ratio - 0.25) < 1e-12,
         "options json should parse export_density_floor_ratio");

  const auto parsed_options_boundary = qx::options_from_json(R"({
    "export_continuous_frames": true,
    "export_frame_stride": 0,
    "export_points_max": 0,
    "export_density_floor_ratio": 3.5
  })");
  expect(parsed_options_boundary.export_frame_stride == 1U, "export_frame_stride should clamp to a minimum of 1");
  expect(parsed_options_boundary.export_points_max == 1U, "export_points_max should clamp to a minimum of 1");
  expect(std::abs(parsed_options_boundary.export_density_floor_ratio - 1.0) < 1e-12,
         "export_density_floor_ratio should clamp to 1.0");

  const auto default_export_contract = qx::continuous_export_config_from_json(R"({})");
  expect(default_export_contract.schema_version == 1U, "default export contract schema_version should be 1");
  expect(default_export_contract.export_frame_stride == 1U, "default export contract frame stride should be 1");
  expect(default_export_contract.export_points_max == 2500U, "default export contract points cap should be 2500");
  expect(std::abs(default_export_contract.export_density_floor_ratio - 0.08) < 1e-12,
         "default export contract density floor ratio should be 0.08");

  const auto future_export_contract = qx::continuous_export_config_from_json(R"({
    "continuous_export": {
      "schema_version": 0,
      "export_frame_stride": 0,
      "export_points_max": 0,
      "export_density_floor_ratio": -0.5,
      "future_field": 99,
      "export_config": {
        "schema_version": 0,
        "export_frame_stride": 0,
        "export_points_max": 0,
        "export_density_floor_ratio": 4.0,
        "future_nested_field": "keep"
      }
    }
  })");
  expect(future_export_contract.schema_version == 1U, "future export contract schema_version should clamp to 1");
  expect(future_export_contract.export_frame_stride == 1U, "future export contract frame stride should clamp to 1");
  expect(future_export_contract.export_points_max == 1U, "future export contract points cap should clamp to 1");
  expect(std::abs(future_export_contract.export_density_floor_ratio - 1.0) < 1e-12,
         "future export contract density floor ratio should clamp to 1.0");

  const auto c2d = qx::problem_from_json(R"({
    "kind": "continuous_space_2d",
    "continuous_space_2d": {
      "nx": 64,
      "ny": 32,
      "dx": 0.05,
      "dy": 0.05,
      "dt": 0.001,
      "steps": 20,
      "boundary": "periodic",
      "potential_kind": "barrier",
      "potential_params": {"height": 1.0},
      "initial_state": {
        "kind": "gaussian",
        "params": {
          "psi_sigma": 0.2,
          "psi_kx": 0.0,
          "psi_ky": 0.0,
          "psi_x0": 0.0,
          "psi_y0": 0.0
        }
      }
    }
  })");
  expect(c2d.kind == qx::ProblemKind::ContinuousSpace2D, "continuous_space_2d kind should parse");
  expect(c2d.continuous_2d.nx == 64, "continuous nx should parse");
  expect(c2d.continuous_2d.ny == 32, "continuous ny should parse");
  expect(std::abs(c2d.continuous_2d.dt - 0.001) < 1e-12, "continuous dt should parse");
  expect(c2d.continuous_2d.initial_state.kind == "gaussian", "continuous2d initial_state kind should parse");
  expect(std::abs(c2d.continuous_2d.initial_state.params.at("psi_sigma") - 0.2) < 1e-12,
         "continuous2d initial_state params should parse");

  const auto c2d_run = qx::run(c2d, o);
  expect(c2d_run.ok, "continuous2d run should succeed");
  expect(c2d_run.backend == "continuous2d", "continuous2d backend should be selected");
  expect(c2d_run.observables.count("norm") == 1, "continuous2d should report norm");
  expect(std::abs(c2d_run.observables.at("norm") - 1.0) < 5e-2, "continuous2d norm should remain close to 1");
  expect(c2d_run.observables.count("sigma_x") == 1, "continuous2d should report sigma_x");
  expect(!c2d_run.continuous_density_frames.empty(), "continuous2d should export density frame metadata");
  expect(c2d_run.continuous_density_frames.front().step == 0, "continuous2d first frame should be step 0");
  expect(c2d_run.continuous_density_frames.back().step == c2d.continuous_2d.steps,
         "continuous2d last frame should match configured steps");
  const std::string c2d_json = qx::result_to_json(c2d_run);
  expect(c2d_json.find("\"continuous_density_frames\"") != std::string::npos,
         "continuous2d result json should include continuous density frames");
  expect(c2d_json.find("\"continuous_export\"") != std::string::npos,
         "continuous2d result json should include continuous export object");

  qx::SimulationResult contract_result;
  contract_result.continuous_export.schema_version = 0;
  contract_result.continuous_export.export_frame_stride = 0;
  contract_result.continuous_export.export_points_max = 0;
  contract_result.continuous_export.export_density_floor_ratio = 9.0;
  const auto contract_doc = nlohmann::json::parse(qx::result_to_json(contract_result));
  expect(contract_doc.at("continuous_export").at("schema_version").get<std::uint32_t>() == 1U,
         "serialized continuous export schema_version should clamp to 1");
  expect(contract_doc.at("continuous_export").at("export_config").at("export_frame_stride").get<std::uint32_t>() == 1U,
         "serialized continuous export frame stride should clamp to 1");
  expect(contract_doc.at("continuous_export").at("export_config").at("export_points_max").get<std::size_t>() == 1U,
         "serialized continuous export points cap should clamp to 1");
  expect(std::abs(contract_doc.at("continuous_export").at("export_config").at("export_density_floor_ratio").get<double>() - 1.0) < 1e-12,
         "serialized continuous export density floor ratio should clamp to 1.0");

  auto o_export = o;
  o_export.export_continuous_frames = true;
  const auto c2d_export_run = qx::run(c2d, o_export);
  expect(c2d_export_run.ok, "continuous2d export run should succeed");
  expect(!c2d_export_run.continuous_density_frames.empty(), "continuous2d export run should include frames");
  expect(!c2d_export_run.continuous_density_frames.front().density_xy_flat.empty(),
         "continuous2d export run should include density payload");
  expect(c2d_export_run.continuous_density_frames.front().density_xy_flat.size() ==
             static_cast<std::size_t>(c2d.continuous_2d.nx) * static_cast<std::size_t>(c2d.continuous_2d.ny),
         "continuous2d export density payload should match nx*ny");

  auto o_export_dirty = o_export;
  o_export_dirty.export_frame_stride = 0;
  o_export_dirty.export_points_max = 0;
  o_export_dirty.export_density_floor_ratio = 2.5;
  const auto c2d_dirty_run = qx::run(c2d, o_export_dirty);
  expect(c2d_dirty_run.ok, "continuous2d dirty export run should succeed");
  expect(c2d_dirty_run.continuous_export.export_frame_stride == 1U,
         "continuous2d dirty export should normalize frame stride to 1");
  expect(c2d_dirty_run.continuous_export.export_points_max == 1U,
         "continuous2d dirty export should normalize points max to 1");
  expect(std::abs(c2d_dirty_run.continuous_export.export_density_floor_ratio - 1.0) < 1e-12,
         "continuous2d dirty export should normalize density floor ratio to 1.0");
  bool saw_contract_event = false;
  for (const auto& ev : c2d_dirty_run.events) {
    if (ev.name == "continuous_export_contract") {
      saw_contract_event = true;
      expect(ev.metrics.count("normalized") == 1, "continuous_export_contract event should expose normalized flag");
      expect(ev.metrics.at("normalized") > 0.5, "continuous_export_contract event should mark normalization");
      expect(std::abs(ev.metrics.at("export_frame_stride") - 1.0) < 1e-12,
             "continuous_export_contract event should report effective frame stride");
      expect(std::abs(ev.metrics.at("export_points_max") - 1.0) < 1e-12,
             "continuous_export_contract event should report effective points cap");
      expect(std::abs(ev.metrics.at("export_density_floor_ratio") - 1.0) < 1e-12,
             "continuous_export_contract event should report effective density floor ratio");
    }
  }
  expect(saw_contract_event, "continuous2d dirty export should emit a normalization event");
  const auto c2d_dirty_doc = nlohmann::json::parse(qx::result_to_json(c2d_dirty_run));
  const auto c2d_dirty_events = c2d_dirty_doc.at("events");
  bool saw_contract_event_json = false;
  for (std::size_t i = 0; i < c2d_dirty_events.size(); ++i) {
    const auto& ev = c2d_dirty_events.at(i);
    if (ev.at("name").get<std::string>() == "continuous_export_contract") {
      saw_contract_event_json = true;
      expect(ev.at("metrics").at("normalized").get<double>() > 0.5,
             "continuous_export_contract JSON event should expose normalization");
    }
  }
  expect(saw_contract_event_json, "continuous_export_contract JSON event should be serialized");

  auto o_export_stride = o_export;
  o_export_stride.export_frame_stride = 4;
  const auto c2d_stride_run = qx::run(c2d, o_export_stride);
  expect(c2d_stride_run.ok, "continuous2d stride export run should succeed");
  expect(!c2d_stride_run.continuous_density_frames.empty(), "continuous2d stride run should include frames");
  expect(c2d_stride_run.continuous_density_frames.front().step == 0, "continuous2d stride first frame should be step 0");
  expect(c2d_stride_run.continuous_density_frames.back().step == c2d.continuous_2d.steps,
         "continuous2d stride last frame should match configured steps");
  expect(!c2d_stride_run.continuous_density_frames.front().density_xy_flat.empty(),
         "continuous2d stride run should keep payload on the first frame");
  expect(c2d_stride_run.continuous_density_frames.at(1).density_xy_flat.empty(),
         "continuous2d stride run should skip payload on non-stride frames");
  const std::string c2d_stride_json = qx::result_to_json(c2d_stride_run);
  const auto c2d_stride_doc = nlohmann::json::parse(c2d_stride_json);
  expect(c2d_stride_doc.at("continuous_export").at("schema_version").get<std::uint32_t>() == 1U,
         "continuous2d export wrapper should expose schema_version");
  expect(c2d_stride_doc.at("continuous_export").at("export_config").at("export_frame_stride").get<std::uint32_t>() == 4U,
         "continuous2d export wrapper should report export_frame_stride");
  const auto c2d_stride_frames = c2d_stride_doc.at("continuous_export").at("frames");
  expect(c2d_stride_frames.is_array(), "continuous2d export wrapper frames should be an array");
  expect(c2d_stride_frames.size() < c2d_stride_run.continuous_density_frames.size(),
         "continuous2d export wrapper should be stride-sampled");
  std::uint32_t previous_step = 0U;
  for (std::size_t i = 0; i < c2d_stride_frames.size(); ++i) {
    const auto& frame = c2d_stride_frames.at(i);
    const std::uint32_t step = frame.at("step").get<std::uint32_t>();
    if (i > 0U) {
      expect(step > previous_step, "continuous2d export wrapper steps should be strictly increasing");
    }
    previous_step = step;
    const auto density = frame.at("density_xy_flat");
    expect(density.is_array(), "continuous2d export wrapper payload should include density_xy_flat");
    expect(density.size() == static_cast<std::size_t>(c2d.continuous_2d.nx) *
                                static_cast<std::size_t>(c2d.continuous_2d.ny),
           "continuous2d export wrapper density payload should match nx*ny");
  }

  const auto workdir = std::filesystem::current_path();
  const auto repo_root = resolve_repo_root(workdir);
  const auto cli_path = resolve_qx_cli_executable(repo_root);
  const auto cli_problem = repo_root / "examples" / "problem_continuous2d_free.json";
  const auto cli_options = std::filesystem::temp_directory_path() / "quantx_cli_export_options.json";
  const auto cli_plain_out = std::filesystem::temp_directory_path() / "quantx_cli_plain.json";
  const auto cli_export_out = std::filesystem::temp_directory_path() / "quantx_cli_export.json";
  std::filesystem::remove(cli_options);
  std::filesystem::remove(cli_plain_out);
  std::filesystem::remove(cli_export_out);
  write_text_file(cli_options, R"({"enable_checkpoint": false})");

  {
    std::ostringstream cmd;
    cmd << quote_path(cli_path) << " --problem " << quote_path(cli_problem) << " --options "
        << quote_path(cli_options) << " --out " << quote_path(cli_plain_out);
    expect(run_command(cmd.str()) == 0, "CLI without export flag should succeed");
  }
  const std::string cli_plain_json = read_text_file(cli_plain_out);
  expect(cli_plain_json.find("\"continuous_export\"") != std::string::npos,
         "CLI without export flag should still emit continuous_export wrapper");
  expect(cli_plain_json.find("\"density_xy_flat\"") == std::string::npos,
         "CLI without export flag should not include dense frame payload");

  {
    std::ostringstream cmd;
    cmd << quote_path(cli_path) << " --problem " << quote_path(cli_problem) << " --options "
        << quote_path(cli_options) << " --out " << quote_path(cli_export_out)
        << " --export-continuous-frames --export-frame-stride 4 --export-points-max 32 --export-density-floor-ratio 0.2";
    expect(run_command(cmd.str()) == 0, "CLI with export flag should succeed");
  }
  const std::string cli_export_json = read_text_file(cli_export_out);
  const auto cli_export_doc = nlohmann::json::parse(cli_export_json);
  expect(cli_export_doc.at("continuous_export").at("schema_version").get<std::uint32_t>() == 1U,
         "CLI export wrapper should expose schema_version");
  expect(cli_export_doc.at("continuous_export").at("export_config").at("export_frame_stride").get<std::uint32_t>() == 4U,
         "CLI export wrapper should preserve export_frame_stride");
  expect(cli_export_doc.at("continuous_export").at("export_config").at("export_points_max").get<std::size_t>() == 32U,
         "CLI export wrapper should preserve export_points_max");
  expect(std::abs(cli_export_doc.at("continuous_export").at("export_config").at("export_density_floor_ratio").get<double>() - 0.2) < 1e-12,
         "CLI export wrapper should preserve export_density_floor_ratio");
  const auto cli_export_frames = cli_export_doc.at("continuous_export").at("frames");
  expect(cli_export_frames.is_array(), "CLI export wrapper frames should be an array");
  expect(cli_export_frames.size() < cli_export_doc.at("continuous_density_frames").size(),
         "CLI export wrapper should sample frames when stride > 1");
  for (std::size_t i = 0; i < cli_export_frames.size(); ++i) {
    const auto& frame = cli_export_frames.at(i);
    expect(frame.at("density_xy_flat").is_array(), "CLI export wrapper should retain 2D density payloads");
    expect(!frame.at("density_xy_flat").empty(), "CLI export wrapper payload should be non-empty");
  }

  auto o_gpu = o;
  o_gpu.use_gpu = true;
  const auto c2d_gpu_run = qx::run(c2d, o_gpu);
  expect(c2d_gpu_run.ok, "continuous2d gpu run should succeed");
  expect(c2d_gpu_run.observables.count("fft_provider_cufft") == 1,
         "continuous2d gpu run should expose fft_provider_cufft");
  expect(c2d_gpu_run.observables.count("fft_provider_eigen") == 1,
         "continuous2d gpu run should expose fft_provider_eigen");
  const double c2d_fft_cufft = c2d_gpu_run.observables.at("fft_provider_cufft");
  const double c2d_fft_eigen = c2d_gpu_run.observables.at("fft_provider_eigen");
  expect(std::abs((c2d_fft_cufft + c2d_fft_eigen) - 1.0) < 1e-9,
         "continuous2d gpu marker sum should stay normalized");
  if (qx::cufft_backend_available()) {
    expect(c2d_fft_cufft > 0.5, "continuous2d gpu run should select cufft provider marker when available");
  } else {
    expect(c2d_fft_eigen > 0.5, "continuous2d gpu run should fall back to eigen marker when cufft is unavailable");
  }

  auto c2d_free_early = qx::problem_from_json(R"({
    "kind": "continuous_space_2d",
    "continuous_space_2d": {
      "nx": 64,
      "ny": 64,
      "dx": 0.1,
      "dy": 0.1,
      "dt": 0.001,
      "steps": 2,
      "boundary": "periodic",
      "potential_kind": "none",
      "potential_params": {"psi_sigma": 0.2, "psi_kx": 0.0, "psi_ky": 0.0, "psi_x0": 0.0, "psi_y0": 0.0}
    }
  })");
  auto c2d_free_late = c2d_free_early;
  c2d_free_late.continuous_2d.steps = 120;

  const auto free_early = qx::run(c2d_free_early, o);
  const auto free_late = qx::run(c2d_free_late, o);
  expect(free_early.ok && free_late.ok, "free-packet continuous2d runs should succeed");
  const double sigma_x_early = free_early.observables.at("sigma_x");
  const double sigma_x_late = free_late.observables.at("sigma_x");
  expect(sigma_x_late > sigma_x_early + 1e-4, "free packet sigma_x should broaden over time");
  const double sigma0 = 0.2;
  const double t_late = static_cast<double>(c2d_free_late.continuous_2d.steps) * c2d_free_late.continuous_2d.dt;
  const double sigma_analytic = std::sqrt(sigma0 * sigma0 + (t_late * t_late) / (4.0 * sigma0 * sigma0));
  expect(std::abs(sigma_x_late - sigma_analytic) < 8e-2,
         "free packet sigma_x should be close to analytic broadening");

  auto c2d_abs = c2d;
  c2d_abs.continuous_2d.boundary = "absorbing";
  c2d_abs.continuous_2d.steps = 60;
  c2d_abs.continuous_2d.potential_params["absorb_width"] = 4.0;
  c2d_abs.continuous_2d.potential_params["absorb_strength"] = 4.0;
  const auto c2d_abs_run = qx::run(c2d_abs, o);
  expect(c2d_abs_run.ok, "continuous2d absorbing run should succeed");
  expect(c2d_abs_run.observables.count("boundary_absorbing") == 1,
         "continuous2d absorbing run should expose boundary flag");
  expect(c2d_abs_run.observables.at("boundary_absorbing") > 0.5,
         "continuous2d absorbing boundary flag should be enabled");
  expect(c2d_abs_run.observables.at("norm") < c2d_run.observables.at("norm"),
         "absorbing boundary should reduce norm versus periodic case");

  auto c2d_dir = c2d;
  c2d_dir.continuous_2d.boundary = "dirichlet";
  c2d_dir.continuous_2d.steps = 40;
  const auto c2d_dir_run = qx::run(c2d_dir, o);
  expect(c2d_dir_run.ok, "continuous2d dirichlet run should succeed");
  expect(c2d_dir_run.observables.count("boundary_dirichlet") == 1,
         "continuous2d dirichlet run should expose boundary flag");
  expect(c2d_dir_run.observables.at("boundary_dirichlet") > 0.5,
         "continuous2d dirichlet boundary flag should be enabled");
  expect(c2d_dir_run.observables.at("norm") > 0.90, "continuous2d dirichlet norm should stay reasonably close to 1");
  expect(std::abs(c2d_dir_run.state.front()) < 1e-10, "continuous2d dirichlet left edge should be zero");
  expect(std::abs(c2d_dir_run.state.at(c2d_dir.continuous_2d.nx - 1U)) < 1e-10,
         "continuous2d dirichlet right edge should be zero");
  expect(std::abs(c2d_dir_run.state.at((c2d_dir.continuous_2d.ny - 1U) * c2d_dir.continuous_2d.nx)) < 1e-10,
         "continuous2d dirichlet bottom edge should be zero");

  {
    auto checkpoint_problem = c2d;
    checkpoint_problem.continuous_2d.steps = 14;
    checkpoint_problem.continuous_2d.potential_kind = "none";
    checkpoint_problem.continuous_2d.boundary = "periodic";
    const auto checkpoint_backend = qx::make_continuous2d_backend();
    const auto checkpoint_dir = std::filesystem::temp_directory_path() / "quantx_continuous2d_restore_test";
    std::filesystem::remove_all(checkpoint_dir);
    std::filesystem::create_directories(checkpoint_dir);

    checkpoint_backend->initialize(checkpoint_problem, o);
    checkpoint_backend->evolve(checkpoint_problem, o);
    const auto checkpoint_obs = checkpoint_backend->measure(checkpoint_problem);
    const auto checkpoint_state = checkpoint_backend->current_state();
    checkpoint_backend->checkpoint(checkpoint_dir.string());

    const auto restored_backend = qx::make_continuous2d_backend();
    restored_backend->restore(checkpoint_dir.string());
    restored_backend->initialize(checkpoint_problem, o);
    restored_backend->evolve(checkpoint_problem, o);
    const auto restored_obs = restored_backend->measure(checkpoint_problem);
    const auto restored_state = restored_backend->current_state();
    expect(restored_state.size() == checkpoint_state.size(), "restored continuous2d state size should match");
    expect(std::abs(restored_obs.at("norm") - checkpoint_obs.at("norm")) < 1e-10,
           "restored continuous2d norm should match checkpoint");
    expect(std::abs(restored_state.front() - checkpoint_state.front()) < 1e-10,
           "restored continuous2d state should match checkpoint");
    expect(std::abs(restored_state.at(restored_state.size() / 2U) - checkpoint_state.at(checkpoint_state.size() / 2U)) <
               1e-10,
           "restored continuous2d state midpoint should match checkpoint");
  }

  const auto c3d = qx::problem_from_json(R"({
    "kind": "continuous_space_3d",
    "continuous_space_3d": {
      "nx": 24,
      "ny": 20,
      "nz": 16,
      "dx": 0.2,
      "dy": 0.2,
      "dz": 0.2,
      "dt": 0.001,
      "steps": 12,
      "boundary": "periodic",
      "potential_kind": "none",
      "potential_params": {"psi_sigma": 0.35, "psi_kx": 0.0, "psi_ky": 0.0, "psi_kz": 0.0},
      "initial_state": {
        "kind": "gaussian",
        "params": {
          "psi_sigma": 0.35,
          "psi_kx": 0.0,
          "psi_ky": 0.0,
          "psi_kz": 0.0,
          "psi_x0": 0.0,
          "psi_y0": 0.0,
          "psi_z0": 0.0
        }
      }
    }
  })");
  expect(c3d.kind == qx::ProblemKind::ContinuousSpace3D, "continuous_space_3d kind should parse");
  expect(c3d.continuous_3d.nz == 16, "continuous3d nz should parse");
  expect(std::abs(c3d.continuous_3d.dz - 0.2) < 1e-12, "continuous3d dz should parse");
  expect(c3d.continuous_3d.initial_state.kind == "gaussian", "continuous3d initial_state kind should parse");

  const auto c3d_info = qx::explain_backend_choice(c3d, o);
  expect(c3d_info.selected_backend == "continuous3d", "continuous3d problem should choose continuous3d backend");

  const auto c3d_run = qx::run(c3d, o);
  expect(c3d_run.ok, "continuous3d run should succeed");
  expect(c3d_run.backend == "continuous3d", "continuous3d backend should run");
  expect(c3d_run.observables.count("norm") == 1, "continuous3d should report norm");
  expect(std::abs(c3d_run.observables.at("norm") - 1.0) < 6e-2, "continuous3d norm should remain close to 1");
  expect(c3d_run.observables.count("sigma_z") == 1, "continuous3d should report sigma_z");
  expect(!c3d_run.continuous_density_frames.empty(), "continuous3d should export density frame metadata");
  expect(c3d_run.continuous_density_frames.back().step == c3d.continuous_3d.steps,
         "continuous3d last frame should match configured steps");
  const auto c3d_export_run = qx::run(c3d, o_export);
  expect(c3d_export_run.ok, "continuous3d export run should succeed");
  expect(!c3d_export_run.continuous_density_frames.empty(), "continuous3d export run should include frames");
  expect(!c3d_export_run.continuous_density_frames.front().density_xy_flat.empty(),
         "continuous3d export run should include xy slice payload");
  expect(c3d_export_run.continuous_density_frames.front().density_xy_flat.size() ==
             static_cast<std::size_t>(c3d.continuous_3d.nx) * static_cast<std::size_t>(c3d.continuous_3d.ny),
         "continuous3d xy slice payload should match nx*ny");
  expect(c3d_export_run.continuous_density_frames.front().density_xz_flat.size() ==
             static_cast<std::size_t>(c3d.continuous_3d.nx) * static_cast<std::size_t>(c3d.continuous_3d.nz),
         "continuous3d xz slice payload should match nx*nz");
  expect(c3d_export_run.continuous_density_frames.front().density_yz_flat.size() ==
             static_cast<std::size_t>(c3d.continuous_3d.ny) * static_cast<std::size_t>(c3d.continuous_3d.nz),
         "continuous3d yz slice payload should match ny*nz");
  expect(!c3d_export_run.continuous_density_frames.front().points_xyzd_flat.empty(),
         "continuous3d export run should include volumetric point payload");
  expect(c3d_export_run.continuous_density_frames.front().points_xyzd_flat.size() % 4U == 0U,
         "continuous3d point payload should be encoded as x,y,z,density quads");

  auto o_export_stride_3d = o_export;
  o_export_stride_3d.export_frame_stride = 5;
  const auto c3d_stride_run = qx::run(c3d, o_export_stride_3d);
  expect(c3d_stride_run.ok, "continuous3d stride export run should succeed");
  expect(!c3d_stride_run.continuous_density_frames.empty(), "continuous3d stride export run should include frames");
  expect(c3d_stride_run.continuous_density_frames.front().step == 0,
         "continuous3d stride first frame should be step 0");
  expect(c3d_stride_run.continuous_density_frames.back().step == c3d.continuous_3d.steps,
         "continuous3d stride last frame should match configured steps");
  expect(!c3d_stride_run.continuous_density_frames.front().points_xyzd_flat.empty(),
         "continuous3d stride run should keep payload on the first frame");
  expect(c3d_stride_run.continuous_density_frames.at(1).points_xyzd_flat.empty(),
         "continuous3d stride run should skip volumetric payload on non-stride frames");
  const auto c3d_stride_doc = nlohmann::json::parse(qx::result_to_json(c3d_stride_run));
  expect(c3d_stride_doc.at("continuous_export").at("export_config").at("export_frame_stride").get<std::uint32_t>() == 5U,
         "continuous3d export wrapper should preserve export_frame_stride");
  expect(c3d_stride_doc.at("continuous_export").at("frames").size() < c3d_stride_run.continuous_density_frames.size(),
         "continuous3d export wrapper should be stride-sampled");

  auto o_export_sparse = o_export;
  o_export_sparse.export_points_max = 16;
  o_export_sparse.export_density_floor_ratio = 0.9;
  const auto c3d_sparse_run = qx::run(c3d, o_export_sparse);
  expect(c3d_sparse_run.ok, "continuous3d sparse export run should succeed");
  expect(!c3d_sparse_run.continuous_density_frames.empty(), "continuous3d sparse export run should include frames");
  expect(c3d_sparse_run.continuous_density_frames.front().points_xyzd_flat.size() <= 16U * 4U,
         "continuous3d sparse export should respect export_points_max");
  expect(c3d_sparse_run.continuous_density_frames.front().points_xyzd_flat.size() <
             c3d_export_run.continuous_density_frames.front().points_xyzd_flat.size(),
         "continuous3d sparse export should reduce point-cloud size");
  const auto c3d_sparse_doc = nlohmann::json::parse(qx::result_to_json(c3d_sparse_run));
  expect(c3d_sparse_doc.at("continuous_export").at("export_config").at("export_points_max").get<std::size_t>() == 16U,
         "continuous3d export wrapper should preserve export_points_max");
  expect(std::abs(c3d_sparse_doc.at("continuous_export").at("export_config").at("export_density_floor_ratio").get<double>() - 0.9) < 1e-12,
         "continuous3d export wrapper should preserve export_density_floor_ratio");

  const auto c3d_gpu_run = qx::run(c3d, o_gpu);
  expect(c3d_gpu_run.ok, "continuous3d gpu run should succeed");
  expect(c3d_gpu_run.observables.count("fft_provider_cufft") == 1,
         "continuous3d gpu run should expose fft_provider_cufft");
  expect(c3d_gpu_run.observables.count("fft_provider_eigen") == 1,
         "continuous3d gpu run should expose fft_provider_eigen");
  const double c3d_fft_cufft = c3d_gpu_run.observables.at("fft_provider_cufft");
  const double c3d_fft_eigen = c3d_gpu_run.observables.at("fft_provider_eigen");
  expect(std::abs((c3d_fft_cufft + c3d_fft_eigen) - 1.0) < 1e-9,
         "continuous3d gpu marker sum should stay normalized");
  if (qx::cufft_backend_available()) {
    expect(c3d_fft_cufft > 0.5, "continuous3d gpu run should select cufft provider marker when available");
  } else {
    expect(c3d_fft_eigen > 0.5, "continuous3d gpu run should fall back to eigen marker when cufft is unavailable");
  }

  auto c3d_free_early = qx::problem_from_json(R"({
    "kind": "continuous_space_3d",
    "continuous_space_3d": {
      "nx": 24,
      "ny": 24,
      "nz": 24,
      "dx": 0.1,
      "dy": 0.1,
      "dz": 0.1,
      "dt": 0.001,
      "steps": 2,
      "boundary": "periodic",
      "potential_kind": "none",
      "potential_params": {
        "psi_sigma": 0.25,
        "psi_kx": 0.0,
        "psi_ky": 0.0,
        "psi_kz": 0.0,
        "psi_x0": 0.0,
        "psi_y0": 0.0,
        "psi_z0": 0.0
      }
    }
  })");
  auto c3d_free_late = c3d_free_early;
  c3d_free_late.continuous_3d.steps = 100;

  const auto c3d_free_early_run = qx::run(c3d_free_early, o);
  const auto c3d_free_late_run = qx::run(c3d_free_late, o);
  expect(c3d_free_early_run.ok && c3d_free_late_run.ok, "free-packet continuous3d runs should succeed");

  const double sigma0_3d = 0.25;
  const double t_late_3d = static_cast<double>(c3d_free_late.continuous_3d.steps) * c3d_free_late.continuous_3d.dt;
  const double sigma_analytic_3d = std::sqrt(sigma0_3d * sigma0_3d + (t_late_3d * t_late_3d) / (4.0 * sigma0_3d * sigma0_3d));

  const double sigma_x_early_3d = c3d_free_early_run.observables.at("sigma_x");
  const double sigma_y_early_3d = c3d_free_early_run.observables.at("sigma_y");
  const double sigma_z_early_3d = c3d_free_early_run.observables.at("sigma_z");
  const double sigma_x_late_3d = c3d_free_late_run.observables.at("sigma_x");
  const double sigma_y_late_3d = c3d_free_late_run.observables.at("sigma_y");
  const double sigma_z_late_3d = c3d_free_late_run.observables.at("sigma_z");

  expect(sigma_x_late_3d > sigma_x_early_3d + 1e-4, "free packet sigma_x should broaden over time in continuous3d");
  expect(sigma_y_late_3d > sigma_y_early_3d + 1e-4, "free packet sigma_y should broaden over time in continuous3d");
  expect(sigma_z_late_3d > sigma_z_early_3d + 1e-4, "free packet sigma_z should broaden over time in continuous3d");
  expect(std::abs(sigma_x_late_3d - sigma_analytic_3d) < 1.2e-1,
         "free packet sigma_x should be close to analytic broadening in continuous3d");
  expect(std::abs(sigma_y_late_3d - sigma_analytic_3d) < 1.2e-1,
         "free packet sigma_y should be close to analytic broadening in continuous3d");
  expect(std::abs(sigma_z_late_3d - sigma_analytic_3d) < 1.2e-1,
         "free packet sigma_z should be close to analytic broadening in continuous3d");

  auto c3d_abs = c3d;
  c3d_abs.continuous_3d.boundary = "absorbing";
  c3d_abs.continuous_3d.steps = 60;
  c3d_abs.continuous_3d.potential_params["absorb_width"] = 4.0;
  c3d_abs.continuous_3d.potential_params["absorb_strength"] = 4.0;
  const auto c3d_abs_run = qx::run(c3d_abs, o);
  expect(c3d_abs_run.ok, "continuous3d absorbing run should succeed");
  expect(c3d_abs_run.observables.count("boundary_absorbing") == 1,
         "continuous3d absorbing run should expose boundary flag");
  expect(c3d_abs_run.observables.at("boundary_absorbing") > 0.5,
         "continuous3d absorbing boundary flag should be enabled");
  expect(c3d_abs_run.observables.at("norm") < c3d_run.observables.at("norm"),
         "absorbing boundary should reduce norm versus periodic case");

  auto c3d_dir = c3d;
  c3d_dir.continuous_3d.boundary = "dirichlet";
  const auto c3d_dir_run = qx::run(c3d_dir, o);
  expect(c3d_dir_run.ok, "continuous3d dirichlet run should succeed");
  expect(c3d_dir_run.observables.count("boundary_dirichlet") == 1,
         "continuous3d dirichlet run should expose boundary flag");
  expect(c3d_dir_run.observables.at("boundary_dirichlet") > 0.5,
         "continuous3d dirichlet boundary flag should be enabled");
  expect(c3d_dir_run.observables.at("norm") > 0.90,
         "continuous3d dirichlet norm should stay reasonably close to 1 in smoke setup");

  {
    auto checkpoint_problem = c3d;
    checkpoint_problem.continuous_3d.steps = 10;
    const auto checkpoint_backend = qx::make_continuous3d_backend();
    const auto checkpoint_dir = std::filesystem::temp_directory_path() / "quantx_continuous3d_restore_test";
    std::filesystem::remove_all(checkpoint_dir);
    std::filesystem::create_directories(checkpoint_dir);

    checkpoint_backend->initialize(checkpoint_problem, o);
    checkpoint_backend->evolve(checkpoint_problem, o);
    const auto checkpoint_obs = checkpoint_backend->measure(checkpoint_problem);
    const auto checkpoint_state = checkpoint_backend->current_state();
    checkpoint_backend->checkpoint(checkpoint_dir.string());

    const auto restored_backend = qx::make_continuous3d_backend();
    restored_backend->restore(checkpoint_dir.string());
    restored_backend->initialize(checkpoint_problem, o);
    restored_backend->evolve(checkpoint_problem, o);
    const auto restored_obs = restored_backend->measure(checkpoint_problem);
    const auto restored_state = restored_backend->current_state();
    expect(restored_state.size() == checkpoint_state.size(), "restored continuous3d state size should match");
    expect(std::abs(restored_obs.at("norm") - checkpoint_obs.at("norm")) < 1e-10,
           "restored continuous3d norm should match checkpoint");
    expect(std::abs(restored_state.front() - checkpoint_state.front()) < 1e-10,
           "restored continuous3d state should match checkpoint");
    expect(std::abs(restored_state.at(restored_state.size() / 2U) - checkpoint_state.at(checkpoint_state.size() / 2U)) <
               1e-10,
           "restored continuous3d state midpoint should match checkpoint");
  }

  qx::QuantumProblem rabi;
  rabi.qubits = 1;
  rabi.evolution.total_time = std::numbers::pi / 2.0;
  rabi.evolution.dt = 0.005;
  rabi.evolution.method = "trotter";
  rabi.hamiltonian_terms = {qx::HamiltonianTerm{"X0", 1.0}};

  const auto rt = qx::run(rabi, o);
  expect(rt.backend == "statevector", "rabi should run on statevector");
  expect(rt.observables.count("population_1") == 1, "population_1 should exist");
  expect(std::abs(rt.observables.at("population_1") - 1.0) < 2e-2, "trotter x-drive should reach near |1>");

  rabi.evolution.method = "krylov";
  rabi.evolution.krylov_dim = 8;
  const auto rk = qx::run(rabi, o);
  expect(std::abs(rk.observables.at("population_1") - 1.0) < 5e-2, "krylov x-drive should reach near |1>");

  qx::QuantumProblem xx;
  xx.qubits = 2;
  xx.evolution.total_time = std::numbers::pi / 2.0;
  xx.evolution.dt = 0.005;
  xx.evolution.method = "trotter";
  xx.hamiltonian_terms = {qx::HamiltonianTerm{"X0X1", 1.0}};
  const auto rxx = qx::run(xx, o);
  expect(rxx.backend == "statevector", "two-qubit XX should run on statevector");
  expect(rxx.observables.count("population_last") == 1, "population_last should exist");
  expect(std::abs(rxx.observables.at("population_last") - 1.0) < 2e-2, "xx at pi/2 should reach near |11>");

  qx::QuantumCircuit bell;
  bell.qubits = 2;
  bell.observables = {"Z0", "Z1", "Z0Z1", "X0X1", "Y0Y1", "X0Y1"};
  bell.operations = {
      qx::CircuitOp{"H", {0}, {}, {}},
      qx::CircuitOp{"CNOT", {1}, {0}, {}},
  };
  const auto cb = qx::run_circuit(bell, o);
  expect(cb.ok, "bell circuit should be ok");
  expect(cb.state.size() == 4, "bell statevector size should be 4");
  expect(cb.elapsed_ms >= 0.0, "circuit elapsed_ms should be available");
  expect(std::abs(std::norm(cb.state[0]) - 0.5) < 2e-2, "bell p00 should be near 0.5");
  expect(std::abs(std::norm(cb.state[3]) - 0.5) < 2e-2, "bell p11 should be near 0.5");
  expect(std::abs(cb.expectation_exact.at("Z0")) < 2e-2, "bell <Z0> should be near 0");
  expect(std::abs(cb.expectation_exact.at("Z1")) < 2e-2, "bell <Z1> should be near 0");
  expect(std::abs(cb.expectation_exact.at("Z0Z1") - 1.0) < 2e-2, "bell <Z0Z1> should be near 1");
  expect(std::abs(cb.expectation_exact.at("X0X1") - 1.0) < 2e-2, "bell <X0X1> should be near 1");
  expect(std::abs(cb.expectation_exact.at("Y0Y1") + 1.0) < 2e-2, "bell <Y0Y1> should be near -1");
  expect(std::abs(cb.expectation_exact.at("X0Y1")) < 2e-2, "bell <X0Y1> should be near 0");
  const std::string circuit_json = qx::circuit_result_to_json(cb);
  expect(circuit_json.find("\"engine_version\"") != std::string::npos, "circuit result json should include engine_version");

  const auto caps = qx::circuit_backend_capabilities();
  expect(caps.backend == "statevector", "circuit capability backend should be statevector");
  expect(caps.max_qubits == 25, "circuit capability max_qubits should match the hard limit");
  expect(caps.max_shots >= 1000, "circuit capability max_shots should cover smoke sampling");
  expect(std::find(caps.supported_gates.begin(), caps.supported_gates.end(), "CNOT") != caps.supported_gates.end(),
         "circuit capability should list CNOT");
  expect(std::find(caps.supported_gates.begin(), caps.supported_gates.end(), "CX") != caps.supported_gates.end(),
         "circuit capability should list CX");
  expect(std::find(caps.supported_noise_models.begin(), caps.supported_noise_models.end(), "depolarizing") !=
             caps.supported_noise_models.end(),
         "circuit capability should list depolarizing noise");
  const std::string caps_json = qx::circuit_capabilities_to_json(caps);
  expect(caps_json.find("\"supported_gates\"") != std::string::npos,
         "circuit capability json should include supported_gates");

  try {
    qx::QuantumCircuit bad_gate;
    bad_gate.qubits = 1;
    bad_gate.operations = {qx::CircuitOp{"FOO", {0}, {}, {}}};
    (void)qx::run_circuit(bad_gate, o);
    expect(false, "unsupported gate should throw CircuitError");
  } catch (const qx::CircuitError& ex) {
    expect(ex.code() == "QX_UNSUPPORTED_GATE", "unsupported gate should use QX_UNSUPPORTED_GATE");
    expect(std::string(ex.what()).find("unsupported gate") != std::string::npos,
           "unsupported gate message should be human readable");
  }

  try {
    (void)qx::circuit_from_json(R"({"qubits":1,"operations":[{"name":"X","targets":[0]}],"noise":{"kind":"phase_flip"}})");
    expect(false, "unsupported noise should throw CircuitError during parse");
  } catch (const qx::CircuitError& ex) {
    expect(ex.code() == "QX_UNSUPPORTED_NOISE", "unsupported noise should use QX_UNSUPPORTED_NOISE");
    expect(std::string(ex.what()).find("unsupported noise model") != std::string::npos,
           "unsupported noise message should be human readable");
  }

  {
    const auto& c_api = c_api_bindings();
    const auto bad_gate_doc =
        parse_owned_json(c_api.run_circuit, R"({"qubits":1,"operations":[{"name":"FOO","targets":[0]}]})",
                         R"({"enable_checkpoint":false})");
    expect(!bad_gate_doc.at("ok").get<bool>(), "C API unsupported gate should return ok=false");
    expect(bad_gate_doc.at("error_kind").get<std::string>() == "circuit",
           "C API unsupported gate should mark circuit error kind");
    expect(bad_gate_doc.at("error_code").get<std::string>() == "QX_UNSUPPORTED_GATE",
           "C API unsupported gate should expose QX_UNSUPPORTED_GATE");
    expect(bad_gate_doc.at("error").get<std::string>() == bad_gate_doc.at("error_message").get<std::string>(),
           "C API unsupported gate should keep error and error_message aligned");
  }

  {
    const auto& c_api = c_api_bindings();
    const auto bad_noise_doc = parse_owned_json(
        c_api.run_circuit, R"({"qubits":1,"operations":[{"name":"X","targets":[0]}],"noise":{"kind":"phase_flip"}})",
        R"({"enable_checkpoint":false})");
    expect(!bad_noise_doc.at("ok").get<bool>(), "C API unsupported noise should return ok=false");
    expect(bad_noise_doc.at("error_kind").get<std::string>() == "circuit",
           "C API unsupported noise should mark circuit error kind");
    expect(bad_noise_doc.at("error_code").get<std::string>() == "QX_UNSUPPORTED_NOISE",
           "C API unsupported noise should expose QX_UNSUPPORTED_NOISE");
    expect(bad_noise_doc.at("error_message").get<std::string>().find("unsupported noise model") != std::string::npos,
           "C API unsupported noise message should be human readable");
  }

  {
    const auto& c_api = c_api_bindings();
    const auto bad_param_doc =
        parse_owned_json(c_api.run_circuit, R"({"qubits":1,"operations":[{"name":"CNOT","targets":[0]}]})",
                         R"({"enable_checkpoint":false})");
    expect(!bad_param_doc.at("ok").get<bool>(), "C API invalid parameter should return ok=false");
    expect(bad_param_doc.at("error_kind").get<std::string>() == "circuit",
           "C API invalid parameter should mark circuit error kind");
    expect(bad_param_doc.at("error_code").get<std::string>() == "QX_INVALID_CIRCUIT",
           "C API invalid parameter should expose QX_INVALID_CIRCUIT");
    expect(bad_param_doc.at("error_message").get<std::string>().find("requires 1 control and 1 target") !=
               std::string::npos,
           "C API invalid parameter message should be human readable");
  }

  qx::QuantumCircuit rx;
  rx.qubits = 1;
  rx.operations = {qx::CircuitOp{"RX", {0}, {}, {std::numbers::pi}}};
  const auto crx = qx::run_circuit(rx, o);
  expect(std::abs(std::norm(crx.state[1]) - 1.0) < 2e-2, "rx(pi) should rotate |0> near |1>");

  bell.shots = 2000;
  const auto cb_shots = qx::run_circuit(bell, o);
  expect(cb_shots.counts.size() >= 2, "bell sampling should have at least two outcomes");
  expect(cb_shots.counts.count("00") == 1, "bell sampling should contain 00");
  expect(cb_shots.counts.count("11") == 1, "bell sampling should contain 11");
  expect(std::abs(cb_shots.expectation_shot.at("Z0")) < 0.1, "shot <Z0> should be near 0");
  expect(std::abs(cb_shots.expectation_shot.at("Z1")) < 0.1, "shot <Z1> should be near 0");
  expect(std::abs(cb_shots.expectation_shot.at("Z0Z1") - 1.0) < 0.1, "shot <Z0Z1> should be near 1");
  expect(std::abs(cb_shots.expectation_shot.at("X0X1") - 1.0) < 0.1, "shot <X0X1> should be near 1");
  expect(std::abs(cb_shots.expectation_shot.at("Y0Y1") + 1.0) < 0.1, "shot <Y0Y1> should be near -1");
  expect(std::abs(cb_shots.expectation_shot.at("X0Y1")) < 0.1, "shot <X0Y1> should be near 0");
  expect(cb_shots.shot_stderr.count("Z0") == 1, "shot stderr should include Z0");
  expect(cb_shots.shot_variance.count("Z0") == 1, "shot variance should include Z0");
  expect(cb_shots.error_summary.count("mean_abs_error") == 1, "error summary should include mean_abs_error");
  expect(cb_shots.error_summary.count("max_abs_error") == 1, "error summary should include max_abs_error");

  qx::QuantumCircuit damp;
  damp.qubits = 1;
  damp.operations = {qx::CircuitOp{"X", {0}, {}, {}}};
  damp.amplitude_damping_gamma = 1.0;
  const auto cd = qx::run_circuit(damp, o);
  expect(std::abs(std::norm(cd.state[0]) - 1.0) < 1e-8, "full amplitude damping should relax |1> to |0>");
  expect(std::abs(std::norm(cd.state[1])) < 1e-8, "full amplitude damping should clear |1> population");

  qx::QuantumCircuit y_eigen;
  y_eigen.qubits = 1;
  y_eigen.shots = 1500;
  y_eigen.observables = {"Y0"};
  y_eigen.operations = {
      qx::CircuitOp{"H", {0}, {}, {}},
      qx::CircuitOp{"Z", {0}, {}, {}},
      qx::CircuitOp{"S", {0}, {}, {}},
  };
  const auto cy = qx::run_circuit(y_eigen, o);
  expect(std::abs(cy.expectation_exact.at("Y0") - 1.0) < 2e-2, "|+i> should satisfy <Y>=1");
  expect(std::abs(cy.expectation_shot.at("Y0") - 1.0) < 0.1, "shot <Y> should be near 1");

  std::cout << "qx_smoke passed\n";
  return 0;
  } catch (const std::exception& ex) {
    return qx::test::report_exception(ex);
  } catch (...) {
    return qx::test::report_unknown_exception();
  }
}
