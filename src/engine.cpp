#include "qx/engine.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <complex>
#include <filesystem>
#include <functional>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "qx/backend.hpp"
#include "qx/backends.hpp"
#include "qx/logging.hpp"
#include "qx/runtime.hpp"
#include "qx/scheduler.hpp"

namespace qx {
namespace {

using Complex = std::complex<double>;

// Ensure built-in backends are registered exactly once.
void ensure_builtin_backends() {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  register_builtin_backends();
  initialized = true;
}

// Validate one qubit index against circuit width.
void require_qubit(std::uint32_t q, std::uint32_t n, const std::string& opname) {
  if (q >= n) {
    throw std::runtime_error("gate " + opname + " references out-of-range qubit");
  }
}

// Apply an arbitrary 2x2 single-qubit unitary to the full statevector.
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

// Apply controlled-NOT gate.
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

// Apply controlled-Z gate.
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

// Apply SWAP gate.
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

// Convert basis index to big-endian bitstring.
std::string bitstring(std::size_t idx, std::uint32_t qubits) {
  std::string out(qubits, '0');
  for (std::uint32_t q = 0; q < qubits; ++q) {
    if ((idx >> q) & 1ULL) {
      out[qubits - 1U - q] = '1';
    }
  }
  return out;
}

// Compute probability that selected qubit is in |1>.
double qubit_one_population(const std::vector<Complex>& state, std::uint32_t qubit) {
  const std::size_t mask = static_cast<std::size_t>(1ULL << qubit);
  double p1 = 0.0;
  for (std::size_t i = 0; i < state.size(); ++i) {
    if ((i & mask) != 0) {
      p1 += std::norm(state[i]);
    }
  }
  return p1;
}

// Normalize statevector to unit norm.
void renormalize(std::vector<Complex>& state) {
  double n = 0.0;
  for (const auto& a : state) {
    n += std::norm(a);
  }
  const double inv = n > 0.0 ? 1.0 / std::sqrt(n) : 1.0;
  for (auto& a : state) {
    a *= inv;
  }
}

// Apply one-qubit amplitude damping channel via quantum trajectories.
void apply_amplitude_damping(
    std::vector<Complex>& state, std::uint32_t qubit, double gamma, std::mt19937_64& rng) {
  if (gamma <= 0.0) return;
  gamma = std::clamp(gamma, 0.0, 1.0);

  const std::size_t n = state.size();
  const std::size_t mask = static_cast<std::size_t>(1ULL << qubit);
  const double p1 = qubit_one_population(state, qubit);
  const double p_jump = gamma * p1;

  std::uniform_real_distribution<double> u01(0.0, 1.0);
  const bool jumped = (p_jump > 0.0) && (u01(rng) < p_jump);

  std::vector<Complex> out(n, Complex{0.0, 0.0});
  if (jumped) {
    // E1 = sqrt(gamma) |0><1|; after normalization -> divide by sqrt(p1).
    const double inv = p1 > 0.0 ? 1.0 / std::sqrt(p1) : 1.0;
    for (std::size_t i = 0; i < n; ++i) {
      if ((i & mask) == 0) {
        const std::size_t j = i | mask;
        out[i] = state[j] * inv;
      }
    }
  } else {
    // E0 = |0><0| + sqrt(1-gamma)|1><1|
    const double s = std::sqrt(std::max(0.0, 1.0 - gamma));
    out = state;
    for (std::size_t i = 0; i < n; ++i) {
      if ((i & mask) != 0) {
        out[i] *= s;
      }
    }
    renormalize(out);
  }

  state.swap(out);
}

// Apply one-qubit depolarizing channel via random Pauli kick.
void apply_depolarizing(std::vector<Complex>& state, std::uint32_t qubit, double p, std::mt19937_64& rng) {
  if (p <= 0.0) return;
  p = std::clamp(p, 0.0, 1.0);
  std::uniform_real_distribution<double> u01(0.0, 1.0);
  if (u01(rng) >= p) return;

  std::uniform_int_distribution<int> pick(0, 2);
  const int pauli = pick(rng);
  if (pauli == 0) {  // X
    apply_single_qubit(state, qubit, {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0});
  } else if (pauli == 1) {  // Y
    apply_single_qubit(state, qubit, {0.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}, {0.0, 0.0});
  } else {  // Z
    apply_single_qubit(state, qubit, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0});
  }
}

// Apply configured noise channels after each gate.
void apply_noise_after_gate(std::vector<Complex>& state, const QuantumCircuit& circuit, std::mt19937_64& rng) {
  for (std::uint32_t q = 0; q < circuit.qubits; ++q) {
    apply_depolarizing(state, q, circuit.depolarizing_prob, rng);
    apply_amplitude_damping(state, q, circuit.amplitude_damping_gamma, rng);
  }
}

// Parsed Pauli operator on a target qubit.
struct PauliTarget {
  char pauli{'Z'};
  std::uint32_t qubit{0};
};

// Mean/variance/std-error for shot-based expectation estimate.
struct ShotStats {
  double mean{0.0};
  double variance{0.0};
  double std_err{0.0};
};

// Parse labels like "Z0X2Y5" to Pauli targets with validation.
std::vector<PauliTarget> parse_pauli_observable(const std::string& label, std::uint32_t qubits) {
  std::vector<PauliTarget> targets;
  std::regex rg("([XYZxyz])([0-9]+)");
  std::unordered_set<std::uint32_t> seen;
  auto begin = std::sregex_iterator(label.begin(), label.end(), rg);
  auto end = std::sregex_iterator();
  for (auto it = begin; it != end; ++it) {
    const char p = static_cast<char>(std::toupper(static_cast<unsigned char>((*it)[1].str()[0])));
    const std::uint32_t q = static_cast<std::uint32_t>(std::stoul((*it)[2].str()));
    if (q >= qubits) {
      throw std::runtime_error("observable references out-of-range qubit: " + label);
    }
    if (seen.find(q) != seen.end()) {
      throw std::runtime_error("observable repeats qubit index: " + label);
    }
    seen.insert(q);
    targets.push_back(PauliTarget{p, q});
  }
  if (targets.empty()) {
    throw std::runtime_error("unsupported observable (expected Pauli string): " + label);
  }
  return targets;
}

// Compute exact expectation value <psi|P|psi> for Pauli string P.
Complex exact_pauli_expectation(const std::vector<Complex>& state, const std::vector<PauliTarget>& obs) {
  Complex value{0.0, 0.0};
  for (std::size_t i = 0; i < state.size(); ++i) {
    std::size_t j = i;
    Complex phase{1.0, 0.0};
    for (const auto& t : obs) {
      const std::size_t mask = static_cast<std::size_t>(1ULL << t.qubit);
      const bool bit = (j & mask) != 0;
      if (t.pauli == 'X') {
        j ^= mask;
      } else if (t.pauli == 'Y') {
        phase *= bit ? Complex{0.0, -1.0} : Complex{0.0, 1.0};
        j ^= mask;
      } else {  // Z
        phase *= bit ? Complex{-1.0, 0.0} : Complex{1.0, 0.0};
      }
    }
    value += std::conj(state[i]) * phase * state[j];
  }
  return value;
}

// Apply Hadamard gate.
void apply_h_gate(std::vector<Complex>& state, std::uint32_t qubit) {
  const double inv_sqrt2 = 0.7071067811865475244;
  apply_single_qubit(state, qubit, {inv_sqrt2, 0.0}, {inv_sqrt2, 0.0}, {inv_sqrt2, 0.0}, {-inv_sqrt2, 0.0});
}

// Apply S phase gate.
void apply_s_gate(std::vector<Complex>& state, std::uint32_t qubit) {
  apply_single_qubit(state, qubit, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 1.0});
}

// Apply S-dagger phase gate.
void apply_sdg_gate(std::vector<Complex>& state, std::uint32_t qubit) {
  apply_single_qubit(state, qubit, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, -1.0});
}

// Rotate measurement basis so Pauli string can be sampled in Z basis.
void rotate_for_pauli_measurement(std::vector<Complex>& state, const std::vector<PauliTarget>& obs) {
  for (const auto& t : obs) {
    if (t.pauli == 'X') {
      apply_h_gate(state, t.qubit);
    } else if (t.pauli == 'Y') {
      // U = H*S^\dagger => U^\dagger Z U = Y
      apply_sdg_gate(state, t.qubit);
      apply_h_gate(state, t.qubit);
    }
  }
}

// Return +/-1 eigenvalue sign for Z-string on sampled basis index.
double z_sign_from_index(std::size_t idx, const std::vector<PauliTarget>& obs) {
  double sign = 1.0;
  for (const auto& t : obs) {
    if (((idx >> t.qubit) & 1ULL) != 0) {
      sign = -sign;
    }
  }
  return sign;
}

// Check if observable is purely Z-type (can use direct counts).
bool observable_all_z(const std::vector<PauliTarget>& obs) {
  for (const auto& t : obs) {
    if (t.pauli != 'Z') return false;
  }
  return true;
}

// Estimate observable stats directly from sampled bitstring counts.
ShotStats shot_stats_from_counts(
    const std::map<std::string, std::uint64_t>& counts, const std::vector<PauliTarget>& obs, std::uint32_t qubits) {
  std::uint64_t total = 0;
  double acc = 0.0;
  for (const auto& [bits, c] : counts) {
    if (bits.size() != qubits) {
      continue;
    }
    double sign = 1.0;
    for (const auto& t : obs) {
      const std::size_t pos = static_cast<std::size_t>(qubits - 1U - t.qubit);
      if (bits[pos] == '1') {
        sign = -sign;
      }
    }
    acc += sign * static_cast<double>(c);
    total += c;
  }
  ShotStats out;
  if (total == 0) {
    return out;
  }
  out.mean = acc / static_cast<double>(total);
  out.variance = std::max(0.0, 1.0 - out.mean * out.mean);
  out.std_err = std::sqrt(out.variance / static_cast<double>(total));
  return out;
}

// Estimate observable stats by rotating state and Monte-Carlo sampling.
ShotStats shot_stats_rotated_state(
    const std::vector<Complex>& state, const std::vector<PauliTarget>& obs, std::uint32_t shots, std::mt19937_64& rng) {
  ShotStats out;
  if (shots == 0) {
    return out;
  }
  std::vector<Complex> rotated = state;
  rotate_for_pauli_measurement(rotated, obs);
  std::vector<double> probs(rotated.size(), 0.0);
  for (std::size_t i = 0; i < rotated.size(); ++i) {
    probs[i] = std::norm(rotated[i]);
  }
  std::discrete_distribution<std::size_t> dist(probs.begin(), probs.end());
  double acc = 0.0;
  for (std::uint32_t s = 0; s < shots; ++s) {
    const std::size_t idx = dist(rng);
    acc += z_sign_from_index(idx, obs);
  }
  std::uint32_t y_count = 0;
  for (const auto& t : obs) {
    if (t.pauli == 'Y') {
      ++y_count;
    }
  }
  const double sign_correction = (y_count % 2 == 0) ? 1.0 : -1.0;
  out.mean = sign_correction * (acc / static_cast<double>(shots));
  out.variance = std::max(0.0, 1.0 - out.mean * out.mean);
  out.std_err = std::sqrt(out.variance / static_cast<double>(shots));
  return out;
}

}  // namespace

// Register built-in backends into global registry.
void register_builtin_backends() {
  auto& registry = BackendRegistry::instance();
  registry.register_backend("statevector", &make_statevector_backend, make_statevector_backend()->capabilities());
  registry.register_backend("mps", &make_mps_backend, make_mps_backend()->capabilities());
  registry.register_backend("continuous2d", &make_continuous2d_backend, make_continuous2d_backend()->capabilities());
  registry.register_backend("continuous3d", &make_continuous3d_backend, make_continuous3d_backend()->capabilities());
}

// Public API: explain backend choice.
BackendInfo explain_backend_choice(const QuantumProblem& problem, const RunOptions& options) {
  ensure_builtin_backends();
  log::init_logging();
  const BackendInfo info = explain_backend_choice_internal(problem, options);
  log::debug("backend explain selected: " + info.selected_backend);
  return info;
}

// Public API: run Hamiltonian simulation.
SimulationResult run(const QuantumProblem& problem, const RunOptions& options) {
  ensure_builtin_backends();
  log::init_logging();
  const auto t0 = std::chrono::high_resolution_clock::now();
  log::info("run started: qubits=" + std::to_string(problem.qubits));

  SimulationResult result;
  RuntimeContext runtime(options);
  runtime.emit_event("runtime_initialized",
                     {{"threads", static_cast<double>(runtime.effective_threads())},
                      {"mpi_ranks", static_cast<double>(runtime.effective_mpi_ranks())}});

  const std::string backend_name = choose_backend(problem, options);
  log::info("scheduler selected backend: " + backend_name);
  auto backend = BackendRegistry::instance().create(backend_name);

  if (options.enable_checkpoint) {
    std::filesystem::create_directories(options.checkpoint_path);
    backend->restore(options.checkpoint_path);
    runtime.emit_event("checkpoint_restore_attempt", {{"success", 1.0}});
    log::debug("checkpoint restore attempted at: " + options.checkpoint_path);
  }

  backend->initialize(problem, options);
  runtime.emit_event("backend_initialized", {{"qubits", static_cast<double>(problem.qubits)}});
  backend->evolve(problem, options);
  runtime.emit_event("evolution_complete", {{"total_time", problem.evolution.total_time}});

  result.ok = true;
  result.backend = backend->name();
  result.observables = backend->measure(problem);
  result.state = backend->current_state();
  result.continuous_density_frames = backend->continuous_density_frames();
  result.events = runtime.events();
  result.message = "Simulation completed";

  if (options.enable_checkpoint) {
    backend->checkpoint(options.checkpoint_path);
    result.events.push_back(Event{"checkpoint_saved", {{"enabled", 1.0}}});
    log::debug("checkpoint saved at: " + options.checkpoint_path);
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  {
    std::ostringstream os;
    os << "run finished: backend=" << result.backend << ", elapsed_ms=" << result.elapsed_ms;
    log::info(os.str());
  }

  return result;
}

// Public API: run gate-level circuit simulation.
CircuitResult run_circuit(const QuantumCircuit& circuit, const RunOptions&) {
  log::init_logging();
  const auto t0 = std::chrono::high_resolution_clock::now();
  log::info("circuit run started: qubits=" + std::to_string(circuit.qubits));
  if (circuit.qubits == 0) {
    throw std::runtime_error("circuit must have at least 1 qubit");
  }
  if (circuit.qubits >= 26) {
    throw std::runtime_error("circuit execution currently supports qubits < 26");
  }

  std::mt19937_64 rng(circuit.seed);

  const std::size_t n = static_cast<std::size_t>(1ULL << circuit.qubits);
  std::vector<Complex> state(n, Complex{0.0, 0.0});
  state[0] = Complex{1.0, 0.0};

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
        apply_h_gate(state, t);
      } else if (g == "S") {
        apply_single_qubit(state, t, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 1.0});
      } else if (g == "T") {
        const double inv_sqrt2 = 0.7071067811865475244;
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
      apply_noise_after_gate(state, circuit, rng);
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
      apply_noise_after_gate(state, circuit, rng);
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
      apply_noise_after_gate(state, circuit, rng);
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
      apply_noise_after_gate(state, circuit, rng);
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
    std::discrete_distribution<std::size_t> dist(probs.begin(), probs.end());
    for (std::uint32_t s = 0; s < circuit.shots; ++s) {
      const std::size_t idx = dist(rng);
      out.counts[bitstring(idx, circuit.qubits)] += 1;
    }
  }

  for (const auto& obs : circuit.observables) {
    const auto parsed = parse_pauli_observable(obs, circuit.qubits);
    const double exact = exact_pauli_expectation(state, parsed).real();
    out.expectation_exact[obs] = exact;
    if (circuit.shots > 0) {
      ShotStats stats;
      if (observable_all_z(parsed)) {
        stats = shot_stats_from_counts(out.counts, parsed, circuit.qubits);
      } else {
        std::mt19937_64 obs_rng(circuit.seed + static_cast<std::uint64_t>(std::hash<std::string>{}(obs)));
        stats = shot_stats_rotated_state(state, parsed, circuit.shots, obs_rng);
      }
      const double shot = stats.mean;
      out.expectation_shot[obs] = shot;
      out.estimation_error[obs] = shot - exact;
      out.shot_variance[obs] = stats.variance;
      out.shot_stderr[obs] = stats.std_err;
    }
  }

  if (!out.estimation_error.empty()) {
    double sum_abs = 0.0;
    double max_abs = 0.0;
    for (const auto& [_, e] : out.estimation_error) {
      const double a = std::abs(e);
      sum_abs += a;
      if (a > max_abs) {
        max_abs = a;
      }
    }
    out.error_summary["mean_abs_error"] = sum_abs / static_cast<double>(out.estimation_error.size());
    out.error_summary["max_abs_error"] = max_abs;
    out.error_summary["num_observables"] = static_cast<double>(out.estimation_error.size());
  }

  const auto t1 = std::chrono::high_resolution_clock::now();
  out.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  {
    std::ostringstream os;
    os << "circuit run finished: elapsed_ms=" << out.elapsed_ms;
    log::info(os.str());
  }

  return out;
}

}  // namespace qx
