// VQE / QAOA variational optimizer for QuantX.
//
// Implements:
//   - Hardware-efficient ansatz (RY + CNOT ladder, n_layers reps)
//   - QAOA ansatz (H init + cost unitary + mixer, p rounds, MaxCut)
//   - Parameter-shift gradient: ∂E/∂θ = (E(θ+π/2) - E(θ-π/2)) / 2
//   - Adam optimizer and vanilla gradient descent
//
// The optimizer reuses run_circuit() for all energy evaluations so it
// inherits the full statevector gate set (RY, RZ, RX, CNOT, H, …).

#include "qx/engine.hpp"
#include "qx/logging.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qx {
namespace {

using Params = std::vector<double>;

// ── Ansatz builders ───────────────────────────────────────────────────────────

// Build hardware-efficient ansatz circuit with given parameter vector.
// Structure: (n_layers) × [RY on each qubit, CNOT chain q0→q1→…→q_{n-1}]
//            + final RY layer (no entanglement after).
// n_params = (n_layers + 1) * n_qubits
QuantumCircuit build_hw_efficient(const VqeOptions& opts, const Params& params) {
  QuantumCircuit c;
  c.qubits = opts.n_qubits;
  const std::uint32_t n = opts.n_qubits;
  std::size_t p = 0;
  for (std::uint32_t layer = 0; layer < opts.n_layers; ++layer) {
    for (std::uint32_t q = 0; q < n; ++q) {
      CircuitOp ry;
      ry.name = "RY";
      ry.targets = {q};
      ry.params = {params[p++]};
      c.operations.push_back(ry);
    }
    if (n > 1) {
      for (std::uint32_t q = 0; q < n - 1; ++q) {
        CircuitOp cx;
        cx.name = "CNOT";
        cx.targets = {q + 1};
        cx.controls = {q};
        c.operations.push_back(cx);
      }
    }
  }
  // Final rotation layer.
  for (std::uint32_t q = 0; q < n; ++q) {
    CircuitOp ry;
    ry.name = "RY";
    ry.targets = {q};
    ry.params = {params[p++]};
    c.operations.push_back(ry);
  }
  return c;
}

// Return n_params for HEA ansatz.
std::size_t hw_efficient_n_params(const VqeOptions& opts) {
  return static_cast<std::size_t>((opts.n_layers + 1) * opts.n_qubits);
}

// Build QAOA circuit for MaxCut on opts.qaoa_edges.
// Initial state: |+⟩^n via H on all qubits.
// p rounds: cost unitary U_C(γ) then mixer U_B(β).
//   U_C(γ) = Π_{(i,j)∈E} CNOT(i,j) · RZ(-γ, j) · CNOT(i,j)
//            [implements exp(+iγ/2 ZiZj), i.e. exp(-iγ H_C) up to const]
//   U_B(β) = Π_i RX(2β, i)  [exp(-iβ Σ Xi)]
// n_params = 2 * n_layers
QuantumCircuit build_qaoa(const VqeOptions& opts, const Params& params) {
  QuantumCircuit c;
  c.qubits = opts.n_qubits;
  // Prepare |+⟩^n.
  for (std::uint32_t q = 0; q < opts.n_qubits; ++q) {
    CircuitOp h;
    h.name = "H";
    h.targets = {q};
    c.operations.push_back(h);
  }
  for (std::uint32_t layer = 0; layer < opts.n_layers; ++layer) {
    const double gamma = params[2 * layer];
    const double beta  = params[2 * layer + 1];
    // Cost unitary: CNOT·RZ(γ)·CNOT per edge → exp(-iγ/2 Z_iZ_j).
    // Keeps standard form exp(-iθG/2) with θ=γ, so parameter-shift shift=π/2 is valid.
    for (const auto& edge : opts.qaoa_edges) {
      const std::uint32_t qi = edge.first;
      const std::uint32_t qj = edge.second;
      CircuitOp cx1, rz, cx2;
      cx1.name = "CNOT"; cx1.targets = {qj}; cx1.controls = {qi};
      rz.name  = "RZ";   rz.targets  = {qj}; rz.params    = {gamma};
      cx2.name = "CNOT"; cx2.targets = {qj}; cx2.controls = {qi};
      c.operations.push_back(cx1);
      c.operations.push_back(rz);
      c.operations.push_back(cx2);
    }
    // Mixer unitary: RX(β) on all qubits → exp(-iβ/2 Xi), standard form.
    for (std::uint32_t q = 0; q < opts.n_qubits; ++q) {
      CircuitOp rx;
      rx.name   = "RX";
      rx.targets = {q};
      rx.params  = {beta};
      c.operations.push_back(rx);
    }
  }
  return c;
}

// Return n_params for QAOA.
std::size_t qaoa_n_params(const VqeOptions& opts) {
  return 2 * static_cast<std::size_t>(opts.n_layers);
}

// ── Energy evaluation ─────────────────────────────────────────────────────────

// Build the parameterized circuit from current params.
QuantumCircuit build_ansatz(const VqeOptions& opts, const Params& params) {
  if (opts.ansatz_type == "qaoa") {
    return build_qaoa(opts, params);
  }
  return build_hw_efficient(opts, params);
}

// Evaluate ⟨H_cost⟩ at given parameter point by running run_circuit().
double eval_energy(const VqeOptions& opts, const Params& params) {
  QuantumCircuit circuit = build_ansatz(opts, params);
  // Collect unique observable labels.
  for (const auto& term : opts.cost_hamiltonian) {
    if (!term.label.empty()) {
      bool found = false;
      for (const auto& o : circuit.observables) {
        if (o == term.label) { found = true; break; }
      }
      if (!found) circuit.observables.push_back(term.label);
    }
  }
  RunOptions ro;
  ro.enable_checkpoint = false;
  const CircuitResult cr = run_circuit(circuit, ro);
  double E = 0.0;
  for (const auto& term : opts.cost_hamiltonian) {
    if (term.label.empty()) {
      E += term.coefficient;  // constant / identity term
      continue;
    }
    auto it = cr.expectation_exact.find(term.label);
    if (it != cr.expectation_exact.end()) {
      E += term.coefficient * it->second;
    }
  }
  return E;
}

// ── Gradient ──────────────────────────────────────────────────────────────────

// Compute analytical gradient via the parameter-shift rule.
// ∂E/∂θ_k = (E(θ + π/2 * e_k) − E(θ − π/2 * e_k)) / 2
Params parameter_shift_gradient(const VqeOptions& opts, const Params& theta) {
  constexpr double shift = 1.5707963267948966;  // π/2
  Params grad(theta.size(), 0.0);
  Params tp = theta;
  Params tm = theta;
  for (std::size_t k = 0; k < theta.size(); ++k) {
    tp[k] += shift;
    tm[k] -= shift;
    grad[k] = (eval_energy(opts, tp) - eval_energy(opts, tm)) * 0.5;
    tp[k] = theta[k];
    tm[k] = theta[k];
  }
  return grad;
}

// ── Adam optimizer ────────────────────────────────────────────────────────────

struct AdamState {
  std::vector<double> m;  // first moment
  std::vector<double> v;  // second moment
  std::uint32_t t{0};     // step counter
  explicit AdamState(std::size_t n) : m(n, 0.0), v(n, 0.0) {}
};

Params adam_step(const VqeOptions& opts, Params theta, const Params& grad, AdamState& s) {
  const double b1 = opts.adam_beta1;
  const double b2 = opts.adam_beta2;
  const double eps = opts.adam_epsilon;
  ++s.t;
  for (std::size_t k = 0; k < theta.size(); ++k) {
    s.m[k] = b1 * s.m[k] + (1.0 - b1) * grad[k];
    s.v[k] = b2 * s.v[k] + (1.0 - b2) * grad[k] * grad[k];
    const double m_hat = s.m[k] / (1.0 - std::pow(b1, s.t));
    const double v_hat = s.v[k] / (1.0 - std::pow(b2, s.t));
    theta[k] -= opts.learning_rate * m_hat / (std::sqrt(v_hat) + eps);
  }
  return theta;
}

// ── Main optimizer loop ───────────────────────────────────────────────────────

std::size_t n_params_for(const VqeOptions& opts) {
  if (opts.ansatz_type == "qaoa") return qaoa_n_params(opts);
  return hw_efficient_n_params(opts);
}

}  // namespace

// Public API.
VqeResult run_vqe(const VqeOptions& opts) {
  log::init_logging();
  const auto t0 = std::chrono::high_resolution_clock::now();
  {
    std::ostringstream os;
    os << "VQE started: ansatz=" << opts.ansatz_type
       << ", n_qubits=" << opts.n_qubits
       << ", n_layers=" << opts.n_layers
       << ", optimizer=" << opts.optimizer;
    log::info(os.str());
  }

  VqeResult result;

  if (opts.n_qubits == 0) {
    result.ok = false;
    result.message = "n_qubits must be >= 1";
    return result;
  }
  if (opts.cost_hamiltonian.empty()) {
    result.ok = false;
    result.message = "cost_hamiltonian must not be empty";
    return result;
  }
  if (opts.ansatz_type == "qaoa" && opts.qaoa_edges.empty()) {
    result.ok = false;
    result.message = "qaoa ansatz requires non-empty qaoa_edges";
    return result;
  }

  const std::size_t np = n_params_for(opts);
  Params theta = opts.initial_params;
  if (theta.empty()) {
    std::mt19937_64 rng(opts.seed);
    std::uniform_real_distribution<double> uni(0.0, 6.283185307179586);
    theta.resize(np);
    for (auto& v : theta) v = uni(rng);
  } else if (theta.size() != np) {
    result.ok = false;
    {
      std::ostringstream os;
      os << "initial_params size " << theta.size()
         << " does not match expected " << np;
      result.message = os.str();
    }
    return result;
  }

  AdamState adam(np);
  double prev_energy = std::numeric_limits<double>::max();
  double energy = eval_energy(opts, theta);

  for (std::uint32_t iter = 0; iter < opts.max_iterations; ++iter) {
    // Record frame.
    VqeFrame frame;
    frame.iteration = iter;
    frame.energy    = energy;
    frame.params    = theta;

    const Params grad = parameter_shift_gradient(opts, theta);
    double gnorm = 0.0;
    for (double g : grad) gnorm += g * g;
    gnorm = std::sqrt(gnorm);
    frame.gradient_norm = gnorm;
    frame.accepted = true;
    result.frames.push_back(frame);

    // Log progress every 10 iterations.
    if (iter % 10 == 0) {
      std::ostringstream os;
      os << "VQE iter=" << iter << " energy=" << energy << " |grad|=" << gnorm;
      log::debug(os.str());
    }

    // Check convergence.
    if (gnorm < opts.convergence_tol) {
      result.converged = true;
      break;
    }
    if (iter > 0 && std::abs(energy - prev_energy) < opts.convergence_tol) {
      result.converged = true;
      break;
    }

    // Update parameters.
    Params new_theta;
    if (opts.optimizer == "gradient_descent") {
      new_theta = theta;
      for (std::size_t k = 0; k < np; ++k) {
        new_theta[k] -= opts.learning_rate * grad[k];
      }
    } else {
      // Default: Adam.
      new_theta = adam_step(opts, theta, grad, adam);
    }

    prev_energy = energy;
    theta = new_theta;
    energy = eval_energy(opts, theta);
  }

  // Final frame.
  {
    VqeFrame frame;
    frame.iteration = result.frames.empty() ? 0 : result.frames.back().iteration + 1;
    frame.energy    = energy;
    frame.params    = theta;
    frame.gradient_norm = 0.0;
    frame.accepted  = true;
    result.frames.push_back(frame);
  }

  result.ok = true;
  result.optimal_energy = energy;
  result.optimal_params = theta;
  result.n_iterations = static_cast<std::uint32_t>(result.frames.size());

  const auto t1 = std::chrono::high_resolution_clock::now();
  result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  {
    std::ostringstream os;
    os << "VQE finished: converged=" << result.converged
       << " optimal_energy=" << result.optimal_energy
       << " n_iterations=" << result.n_iterations
       << " elapsed_ms=" << result.elapsed_ms;
    log::info(os.str());
    result.message = result.converged ? "Converged" : "Max iterations reached";
  }

  return result;
}

}  // namespace qx
