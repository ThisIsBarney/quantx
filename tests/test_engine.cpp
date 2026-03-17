#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numbers>

#include "qx/engine.hpp"

namespace {

void expect(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    std::exit(1);
  }
}

}  // namespace

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
  expect(r.observables.count("norm") == 1, "norm observable should exist");
  expect(std::abs(r.observables.at("norm") - 1.0) < 1e-8, "norm should be conserved");

  p.qubits = 64;
  const auto info2 = qx::explain_backend_choice(p, o);
  expect(info2.selected_backend == "mps", "large problem should choose mps");
  const auto r2 = qx::run(p, o);
  expect(r2.backend == "mps", "backend should be mps");
  expect(r2.observables.count("bond_dim") == 1, "mps should report bond_dim");

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
  bell.observables = {"Z0", "Z1", "Z0Z1", "X0X1"};
  bell.operations = {
      qx::CircuitOp{"H", {0}, {}, {}},
      qx::CircuitOp{"CNOT", {1}, {0}, {}},
  };
  const auto cb = qx::run_circuit(bell, o);
  expect(cb.ok, "bell circuit should be ok");
  expect(cb.state.size() == 4, "bell statevector size should be 4");
  expect(std::abs(std::norm(cb.state[0]) - 0.5) < 2e-2, "bell p00 should be near 0.5");
  expect(std::abs(std::norm(cb.state[3]) - 0.5) < 2e-2, "bell p11 should be near 0.5");
  expect(std::abs(cb.expectation_exact.at("Z0")) < 2e-2, "bell <Z0> should be near 0");
  expect(std::abs(cb.expectation_exact.at("Z1")) < 2e-2, "bell <Z1> should be near 0");
  expect(std::abs(cb.expectation_exact.at("Z0Z1") - 1.0) < 2e-2, "bell <Z0Z1> should be near 1");
  expect(std::abs(cb.expectation_exact.at("X0X1") - 1.0) < 2e-2, "bell <X0X1> should be near 1");

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
