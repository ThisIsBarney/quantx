#include "qx/engine.hpp"

#include <algorithm>
#include <cctype>
#include <complex>
#include <filesystem>
#include <random>
#include <stdexcept>

#include "qx/backend.hpp"
#include "qx/backends.hpp"
#include "qx/runtime.hpp"
#include "qx/scheduler.hpp"

namespace qx {
namespace {

using Complex = std::complex<double>;

void ensure_builtin_backends() {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  register_builtin_backends();
  initialized = true;
}

void require_qubit(std::uint32_t q, std::uint32_t n, const std::string& opname) {
  if (q >= n) {
    throw std::runtime_error("gate " + opname + " references out-of-range qubit");
  }
}

void apply_single_qubit(
    std::vector<Complex>& state,
    std::uint32_t qubit,
    Complex m00,
    Complex m01,
    Complex m10,
    Complex m11) {
  const std::size_t n = state.size();
  const std::size_t mask = static_cast<std::size_t>(1ULL << qubit);
  for (std::size_t i = 0; i < n; ++i) {
    if ((i & mask) == 0) {
      const std::size_t j = i | mask;
      const Complex a = state[i];
      const Complex b = state[j];
      state[i] = m00 * a + m01 * b;
      state[j] = m10 * a + m11 * b;
    }
  }
}

void apply_cnot(std::vector<Complex>& state, std::uint32_t control, std::uint32_t target) {
  const std::size_t n = state.size();
  const std::size_t cm = static_cast<std::size_t>(1ULL << control);
  const std::size_t tm = static_cast<std::size_t>(1ULL << target);
  for (std::size_t i = 0; i < n; ++i) {
    if ((i & cm) != 0 && (i & tm) == 0) {
      std::swap(state[i], state[i | tm]);
    }
  }
}

void apply_cz(std::vector<Complex>& state, std::uint32_t control, std::uint32_t target) {
  const std::size_t n = state.size();
  const std::size_t cm = static_cast<std::size_t>(1ULL << control);
  const std::size_t tm = static_cast<std::size_t>(1ULL << target);
  for (std::size_t i = 0; i < n; ++i) {
    if ((i & cm) != 0 && (i & tm) != 0) {
      state[i] = -state[i];
    }
  }
}

void apply_swap(std::vector<Complex>& state, std::uint32_t q0, std::uint32_t q1) {
  if (q0 == q1) return;
  const std::size_t n = state.size();
  const std::size_t m0 = static_cast<std::size_t>(1ULL << q0);
  const std::size_t m1 = static_cast<std::size_t>(1ULL << q1);
  for (std::size_t i = 0; i < n; ++i) {
    const bool b0 = (i & m0) != 0;
    const bool b1 = (i & m1) != 0;
    if (!b0 && b1) {
      const std::size_t j = (i | m0) & ~m1;
      std::swap(state[i], state[j]);
    }
  }
}

std::string bitstring(std::size_t idx, std::uint32_t qubits) {
  std::string out(qubits, '0');
  for (std::uint32_t q = 0; q < qubits; ++q) {
    if ((idx >> q) & 1ULL) {
      out[qubits - 1U - q] = '1';
    }
  }
  return out;
}

}  // namespace

void register_builtin_backends() {
  auto& registry = BackendRegistry::instance();
  registry.register_backend("statevector", &make_statevector_backend, make_statevector_backend()->capabilities());
  registry.register_backend("mps", &make_mps_backend, make_mps_backend()->capabilities());
}

BackendInfo explain_backend_choice(const QuantumProblem& problem, const RunOptions& options) {
  ensure_builtin_backends();
  return explain_backend_choice_internal(problem, options);
}

SimulationResult run(const QuantumProblem& problem, const RunOptions& options) {
  ensure_builtin_backends();

  SimulationResult result;
  RuntimeContext runtime(options);
  runtime.emit_event("runtime_initialized",
                     {{"threads", static_cast<double>(runtime.effective_threads())},
                      {"mpi_ranks", static_cast<double>(runtime.effective_mpi_ranks())}});

  const std::string backend_name = choose_backend(problem, options);
  auto backend = BackendRegistry::instance().create(backend_name);

  if (options.enable_checkpoint) {
    std::filesystem::create_directories(options.checkpoint_path);
    backend->restore(options.checkpoint_path);
    runtime.emit_event("checkpoint_restore_attempt", {{"success", 1.0}});
  }

  backend->initialize(problem, options);
  runtime.emit_event("backend_initialized", {{"qubits", static_cast<double>(problem.qubits)}});
  backend->evolve(problem, options);
  runtime.emit_event("evolution_complete", {{"total_time", problem.evolution.total_time}});

  result.ok = true;
  result.backend = backend->name();
  result.observables = backend->measure(problem);
  result.state = backend->current_state();
  result.events = runtime.events();
  result.message = "Simulation completed";

  if (options.enable_checkpoint) {
    backend->checkpoint(options.checkpoint_path);
    result.events.push_back(Event{"checkpoint_saved", {{"enabled", 1.0}}});
  }

  return result;
}

CircuitResult run_circuit(const QuantumCircuit& circuit, const RunOptions&) {
  if (circuit.qubits == 0) {
    throw std::runtime_error("circuit must have at least 1 qubit");
  }
  if (circuit.qubits >= 26) {
    throw std::runtime_error("circuit execution currently supports qubits < 26");
  }

  const std::size_t n = static_cast<std::size_t>(1ULL << circuit.qubits);
  std::vector<Complex> state(n, Complex{0.0, 0.0});
  state[0] = Complex{1.0, 0.0};

  const double inv_sqrt2 = 0.7071067811865475244;
  for (const auto& op : circuit.operations) {
    std::string g = op.name;
    std::transform(g.begin(), g.end(), g.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

    if (g == "X" || g == "Y" || g == "Z" || g == "H" || g == "S" || g == "T" || g == "RX" || g == "RY" ||
        g == "RZ") {
      if (op.targets.size() != 1) {
        throw std::runtime_error("single-qubit gate " + g + " requires 1 target");
      }
      const std::uint32_t t = op.targets[0];
      require_qubit(t, circuit.qubits, g);

      if (g == "X") {
        apply_single_qubit(state, t, {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0});
      } else if (g == "Y") {
        apply_single_qubit(state, t, {0.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}, {0.0, 0.0});
      } else if (g == "Z") {
        apply_single_qubit(state, t, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0});
      } else if (g == "H") {
        apply_single_qubit(state, t, {inv_sqrt2, 0.0}, {inv_sqrt2, 0.0}, {inv_sqrt2, 0.0}, {-inv_sqrt2, 0.0});
      } else if (g == "S") {
        apply_single_qubit(state, t, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 1.0});
      } else if (g == "T") {
        apply_single_qubit(state, t, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {inv_sqrt2, inv_sqrt2});
      } else if (g == "RX") {
        const double theta = op.params.empty() ? 0.0 : op.params[0];
        const double c = std::cos(theta * 0.5);
        const double s = std::sin(theta * 0.5);
        apply_single_qubit(state, t, {c, 0.0}, {0.0, -s}, {0.0, -s}, {c, 0.0});
      } else if (g == "RY") {
        const double theta = op.params.empty() ? 0.0 : op.params[0];
        const double c = std::cos(theta * 0.5);
        const double s = std::sin(theta * 0.5);
        apply_single_qubit(state, t, {c, 0.0}, {-s, 0.0}, {s, 0.0}, {c, 0.0});
      } else if (g == "RZ") {
        const double theta = op.params.empty() ? 0.0 : op.params[0];
        apply_single_qubit(state, t, std::exp(Complex{0.0, -theta * 0.5}), {0.0, 0.0}, {0.0, 0.0},
                           std::exp(Complex{0.0, theta * 0.5}));
      }
      continue;
    }

    if (g == "CNOT" || g == "CX") {
      if (op.controls.empty() || op.targets.size() != 1) {
        throw std::runtime_error("CNOT requires 1 control and 1 target");
      }
      const std::uint32_t c = op.controls[0];
      const std::uint32_t t = op.targets[0];
      require_qubit(c, circuit.qubits, g);
      require_qubit(t, circuit.qubits, g);
      apply_cnot(state, c, t);
      continue;
    }

    if (g == "CZ") {
      if (op.controls.empty() || op.targets.size() != 1) {
        throw std::runtime_error("CZ requires 1 control and 1 target");
      }
      const std::uint32_t c = op.controls[0];
      const std::uint32_t t = op.targets[0];
      require_qubit(c, circuit.qubits, g);
      require_qubit(t, circuit.qubits, g);
      apply_cz(state, c, t);
      continue;
    }

    if (g == "SWAP") {
      if (op.targets.size() != 2) {
        throw std::runtime_error("SWAP requires 2 targets");
      }
      const std::uint32_t q0 = op.targets[0];
      const std::uint32_t q1 = op.targets[1];
      require_qubit(q0, circuit.qubits, g);
      require_qubit(q1, circuit.qubits, g);
      apply_swap(state, q0, q1);
      continue;
    }

    throw std::runtime_error("unsupported gate: " + g);
  }

  CircuitResult out;
  out.ok = true;
  out.backend = "statevector";
  out.state = state;
  out.message = "Circuit executed";

  if (circuit.shots > 0) {
    std::vector<double> probs(state.size(), 0.0);
    for (std::size_t i = 0; i < state.size(); ++i) {
      probs[i] = std::norm(state[i]);
    }
    std::mt19937_64 rng(42);
    std::discrete_distribution<std::size_t> dist(probs.begin(), probs.end());
    for (std::uint32_t s = 0; s < circuit.shots; ++s) {
      const std::size_t idx = dist(rng);
      out.counts[bitstring(idx, circuit.qubits)] += 1;
    }
  }

  return out;
}

}  // namespace qx
