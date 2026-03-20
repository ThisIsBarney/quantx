#include <cmath>
#ifdef _MSC_VER
#include <crtdbg.h>
#endif
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

#include "qx/fft_provider.hpp"
#include "qx/backends.hpp"
#include "qx/engine.hpp"
#include "qx/json_utils.hpp"
#include "qx/version.hpp"

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
#ifdef _MSC_VER
#ifdef _DEBUG
  // Route CRT assertions/errors to stderr to avoid interactive debug popups in CI and scripted runs.
  _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
#endif
#endif

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
  auto o_export = o;
  o_export.export_continuous_frames = true;
  const auto c2d_export_run = qx::run(c2d, o_export);
  expect(c2d_export_run.ok, "continuous2d export run should succeed");
  expect(!c2d_export_run.continuous_density_frames.empty(), "continuous2d export run should include frames");
  expect(!c2d_export_run.continuous_density_frames.front().density_xy_flat.empty(),
         "continuous2d export run should include density payload");

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
        << " --export-continuous-frames";
    expect(run_command(cmd.str()) == 0, "CLI with export flag should succeed");
  }
  const std::string cli_export_json = read_text_file(cli_export_out);
  expect(cli_export_json.find("\"density_xy_flat\"") != std::string::npos,
         "CLI export flag should include dense frame payload");

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
  expect(!c3d_export_run.continuous_density_frames.front().points_xyzd_flat.empty(),
         "continuous3d export run should include volumetric point payload");

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
    std::cerr << "EXCEPTION: " << ex.what() << "\n";
    return 2;
  } catch (...) {
    std::cerr << "EXCEPTION: unknown\n";
    return 2;
  }
}
