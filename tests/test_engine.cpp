#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>

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

}  // namespace

// End-to-end smoke test for simulation and circuit APIs.
int main() {
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
      "potential_params": {"height": 1.0}
    }
  })");
  expect(c2d.kind == qx::ProblemKind::ContinuousSpace2D, "continuous_space_2d kind should parse");
  expect(c2d.continuous_2d.nx == 64, "continuous nx should parse");
  expect(c2d.continuous_2d.ny == 32, "continuous ny should parse");
  expect(std::abs(c2d.continuous_2d.dt - 0.001) < 1e-12, "continuous dt should parse");

  const auto c2d_run = qx::run(c2d, o);
  expect(c2d_run.ok, "continuous2d run should succeed");
  expect(c2d_run.backend == "continuous2d", "continuous2d backend should be selected");
  expect(c2d_run.observables.count("norm") == 1, "continuous2d should report norm");
  expect(std::abs(c2d_run.observables.at("norm") - 1.0) < 5e-2, "continuous2d norm should remain close to 1");
  expect(c2d_run.observables.count("sigma_x") == 1, "continuous2d should report sigma_x");

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
      "potential_params": {"psi_sigma": 0.35, "psi_kx": 0.0, "psi_ky": 0.0, "psi_kz": 0.0}
    }
  })");
  expect(c3d.kind == qx::ProblemKind::ContinuousSpace3D, "continuous_space_3d kind should parse");
  expect(c3d.continuous_3d.nz == 16, "continuous3d nz should parse");
  expect(std::abs(c3d.continuous_3d.dz - 0.2) < 1e-12, "continuous3d dz should parse");

  const auto c3d_info = qx::explain_backend_choice(c3d, o);
  expect(c3d_info.selected_backend == "continuous3d", "continuous3d problem should choose continuous3d backend");

  const auto c3d_run = qx::run(c3d, o);
  expect(c3d_run.ok, "continuous3d run should succeed");
  expect(c3d_run.backend == "continuous3d", "continuous3d backend should run");
  expect(c3d_run.observables.count("norm") == 1, "continuous3d should report norm");
  expect(std::abs(c3d_run.observables.at("norm") - 1.0) < 6e-2, "continuous3d norm should remain close to 1");
  expect(c3d_run.observables.count("sigma_z") == 1, "continuous3d should report sigma_z");

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
}
