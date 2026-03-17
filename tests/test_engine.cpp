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

  std::cout << "qx_smoke passed\n";
  return 0;
}
