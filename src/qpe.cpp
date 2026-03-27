// Quantum Phase Estimation for QuantX.
//
// Circuit structure (n_ancilla + n_target qubits):
//   Qubits 0..n_ancilla-1  — ancilla (precision) register
//   Qubits n_ancilla..n_ancilla+n_target-1 — target register
//
//   1. Hadamard on all ancilla qubits  → |+⟩^n_ancilla ⊗ |ψ⟩
//   2. Controlled-U^{2^k} for k = 0..n_ancilla-1
//      Ancilla qubit k controls U^{2^k} on the target register.
//   3. Inverse QFT on ancilla register (exact, using controlled-phase decomposition)
//   4. Measure all ancilla qubits → integer a, phase estimate = a / 2^n_ancilla
//
// Supported unitary types:
//   "phase_gate"  — P(θ): |1⟩ → e^{iθ}|1⟩  (single-qubit, n_target=1)
//   "hamiltonian" — U = exp(-iHt) approximated by first-order Trotter
//                   (n_target = number of qubits in Hamiltonian; up to 3)
//
// The C++ implementation constructs the QPE circuit as a QuantumCircuit and
// runs it via run_circuit() with n_shots samples.  For the Hamiltonian case,
// a first-order Trotter expansion of exp(-iHt) is interleaved into the circuit
// as CU gates built from the Pauli rotation decomposition.
//
// NOTE: The exact circuit-level Trotter decomposition for arbitrary Hamiltonians
// is intricate; the QPE C++ path therefore supports only the "phase_gate" case
// directly.  The "hamiltonian" case is handled by the Python reference solver
// (_qpe_frames) which uses full matrix exponentiation.

#include "qx/engine.hpp"
#include "qx/logging.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace qx {
namespace {

// ── QFT circuit builder ────────────────────────────────────────────────────────
//
// Inverse QFT on qubits [q_start, q_start + n) using the standard decomposition:
//   1. SWAP the register (reverse order)
//   2. For each qubit j from 0 to n-1:
//        H on qubit j
//        Controlled-P(-π/2^k) for k = 2..n-j  (ctrl = j+k-1)

// Append inverse-QFT gates on qubits [first, first+n) to the operation list.
void append_iqft(std::vector<CircuitOp>& ops, std::uint32_t first, std::uint32_t n) {
  // Reverse the qubit order via SWAP chain.
  for (std::uint32_t i = 0; i < n / 2; ++i) {
    CircuitOp sw;
    sw.name = "SWAP";
    sw.targets = {first + i, first + n - 1 - i};
    ops.push_back(sw);
  }
  // Standard inverse-QFT decomposition.
  for (std::uint32_t j = 0; j < n; ++j) {
    // H on qubit first+j.
    CircuitOp h;
    h.name = "H";
    h.targets = {first + j};
    ops.push_back(h);
    // Controlled-RZ(-π/2^k) for k = 2..n-j  (control = qubit first+j+k-1).
    for (std::uint32_t k = 2; k <= n - j; ++k) {
      const double angle = -2.0 * 3.141592653589793 / static_cast<double>(1ULL << k);
      // CRZ(θ) = CNOT·RZ(θ/2)·CNOT·RZ(-θ/2) on ctrl/tgt.
      // Use the phase-kickback form: for phase estimation, we only need
      // the diagonal controlled-RZ. Decompose as:
      //   RZ(-θ/2) on ctrl  ·  CNOT(ctrl,tgt)  ·  RZ(θ/2) on tgt  ·
      //   CNOT(ctrl,tgt)  ·  RZ(θ/2) on ctrl
      // which implements diag(1, 1, e^{iθ/2}, e^{-iθ/2}) × global_phase correction.
      // Simpler: just use RZ(θ) on target conditioned via CNOT sandwich.
      // CRZ(θ): CNOT(ctrl,tgt) · RZ(θ/2) · CNOT(ctrl,tgt) · RZ(-θ/2) on tgt = diag(1,1,e^{-iθ/2},e^{iθ/2})
      // But CPhase(θ) = diag(1,1,1,e^{iθ}) is what we actually want.
      // CPhase(θ) decomposition:
      //   RZ(θ/2) on ctrl · CNOT(ctrl,tgt) · RZ(-θ/2) on tgt · CNOT(ctrl,tgt) · RZ(θ/2) on tgt
      const std::uint32_t ctrl = first + j + k - 1;
      const std::uint32_t tgt  = first + j;
      // RZ(θ/2) on ctrl.
      CircuitOp rz_ctrl1;
      rz_ctrl1.name = "RZ"; rz_ctrl1.targets = {ctrl}; rz_ctrl1.params = {angle / 2.0};
      ops.push_back(rz_ctrl1);
      // CNOT(ctrl, tgt).
      CircuitOp cx1;
      cx1.name = "CNOT"; cx1.targets = {tgt}; cx1.controls = {ctrl};
      ops.push_back(cx1);
      // RZ(-θ/2) on tgt.
      CircuitOp rz_tgt_neg;
      rz_tgt_neg.name = "RZ"; rz_tgt_neg.targets = {tgt}; rz_tgt_neg.params = {-angle / 2.0};
      ops.push_back(rz_tgt_neg);
      // CNOT(ctrl, tgt).
      CircuitOp cx2;
      cx2.name = "CNOT"; cx2.targets = {tgt}; cx2.controls = {ctrl};
      ops.push_back(cx2);
      // RZ(θ/2) on tgt.
      CircuitOp rz_tgt_pos;
      rz_tgt_pos.name = "RZ"; rz_tgt_pos.targets = {tgt}; rz_tgt_pos.params = {angle / 2.0};
      ops.push_back(rz_tgt_pos);
    }
  }
}

// ── Phase-gate QPE circuit ────────────────────────────────────────────────────
//
// U = P(θ): |0⟩ → |0⟩, |1⟩ → e^{iθ}|1⟩
// U^k     = P(kθ)
// Target eigenstate: |1⟩ with eigenvalue e^{iθ} → phase φ = θ/(2π)
//
// Controlled-P(kθ) on (ancilla, target) decomposes as controlled-RZ:
//   CPhase(α) = RZ(α/2)_ctrl · CNOT(ctrl,tgt) · RZ(-α/2)_tgt · CNOT(ctrl,tgt) · RZ(α/2)_tgt
QuantumCircuit build_phase_gate_qpe(const QpeOptions& opts) {
  const std::uint32_t n_anc = opts.n_ancilla;
  const std::uint32_t n_tgt = opts.n_target;  // must be 1
  const std::uint32_t n = n_anc + n_tgt;
  // Target qubit is the last qubit.
  const std::uint32_t tgt = n - 1;

  QuantumCircuit c;
  c.qubits = n;
  c.seed = opts.seed;

  // Prepare ancilla in |+⟩^n_anc.
  for (std::uint32_t k = 0; k < n_anc; ++k) {
    CircuitOp h;
    h.name = "H"; h.targets = {k};
    c.operations.push_back(h);
  }
  // Target eigenstate |1⟩: X gate.
  {
    CircuitOp x;
    x.name = "X"; x.targets = {tgt};
    c.operations.push_back(x);
  }

  // Controlled-P(2^k * θ) for k = 0..n_anc-1, control = ancilla[k].
  double power_angle = opts.phase_angle;  // θ * 2^0
  for (std::uint32_t k = 0; k < n_anc; ++k) {
    const std::uint32_t ctrl = k;
    const double alpha = power_angle;
    // CPhase(alpha): diag(1,1,1,e^{i*alpha}) on {|ctrl=0,tgt=0>,|ctrl=0,tgt=1>,|ctrl=1,tgt=0>,|ctrl=1,tgt=1>}
    // Decomposition:
    //   RZ(alpha/2) on ctrl
    //   CNOT(ctrl, tgt)
    //   RZ(-alpha/2) on tgt
    //   CNOT(ctrl, tgt)
    //   RZ(alpha/2) on tgt
    CircuitOp rz_ctrl;
    rz_ctrl.name = "RZ"; rz_ctrl.targets = {ctrl}; rz_ctrl.params = {alpha / 2.0};
    c.operations.push_back(rz_ctrl);

    CircuitOp cx1;
    cx1.name = "CNOT"; cx1.targets = {tgt}; cx1.controls = {ctrl};
    c.operations.push_back(cx1);

    CircuitOp rz_tgt_neg;
    rz_tgt_neg.name = "RZ"; rz_tgt_neg.targets = {tgt}; rz_tgt_neg.params = {-alpha / 2.0};
    c.operations.push_back(rz_tgt_neg);

    CircuitOp cx2;
    cx2.name = "CNOT"; cx2.targets = {tgt}; cx2.controls = {ctrl};
    c.operations.push_back(cx2);

    CircuitOp rz_tgt_pos;
    rz_tgt_pos.name = "RZ"; rz_tgt_pos.targets = {tgt}; rz_tgt_pos.params = {alpha / 2.0};
    c.operations.push_back(rz_tgt_pos);

    power_angle *= 2.0;  // next power: θ * 2^{k+1}
  }

  // Inverse QFT on ancilla qubits [0, n_anc).
  append_iqft(c.operations, 0, n_anc);

  // Measure all ancilla qubits.
  for (std::uint32_t k = 0; k < n_anc; ++k) {
    CircuitOp m;
    m.name = "MEASURE"; m.targets = {k}; m.controls = {k};
    c.operations.push_back(m);
  }
  c.clbits = n_anc;
  c.shots = opts.n_shots;

  return c;
}

}  // namespace

// Public API.
QpeResult run_qpe(const QpeOptions& opts) {
  log::init_logging();
  const auto t0 = std::chrono::high_resolution_clock::now();
  {
    std::ostringstream os;
    os << "QPE started: unitary=" << opts.unitary_type
       << ", n_ancilla=" << opts.n_ancilla
       << ", n_target=" << opts.n_target
       << ", n_shots=" << opts.n_shots;
    log::info(os.str());
  }

  QpeResult result;
  result.n_ancilla = opts.n_ancilla;
  result.n_target  = opts.n_target;

  const std::uint32_t dim_anc = 1U << opts.n_ancilla;
  result.measurement_counts.assign(dim_anc, 0);
  result.phase_probs.assign(dim_anc, 0.0);

  if (opts.unitary_type != "phase_gate") {
    result.ok = false;
    result.message = "C++ QPE currently supports unitary_type='phase_gate' only; "
                     "use the Python reference solver for 'hamiltonian'.";
    return result;
  }

  if (opts.n_target != 1) {
    result.ok = false;
    result.message = "phase_gate QPE requires n_target=1";
    return result;
  }

  if (opts.n_ancilla == 0) {
    result.ok = false;
    result.message = "n_ancilla must be >= 1";
    return result;
  }

  // Build and run the QPE circuit.
  const QuantumCircuit circ = build_phase_gate_qpe(opts);
  RunOptions ro;
  ro.enable_checkpoint = false;
  const CircuitResult cr = run_circuit(circ, ro);

  // Decode measurement outcomes.
  // Each shot produces n_ancilla measurement results. We read from cr.counts
  // which maps binary strings → count.
  for (const auto& [bitstr, cnt] : cr.counts) {
    // The n_ancilla + n_target bit string: first n_ancilla bits are ancilla.
    // cr counts over the full n_ancilla+n_target qubit register (but target
    // collapses independently; we care only about ancilla bits).
    // Extract ancilla bits (qubits 0..n_ancilla-1 = first n_ancilla chars).
    if (bitstr.size() < opts.n_ancilla) continue;
    const std::string anc_str = bitstr.substr(0, opts.n_ancilla);
    std::uint32_t anc_val = 0;
    for (std::uint32_t b = 0; b < opts.n_ancilla; ++b) {
      if (anc_str[b] == '1') {
        // Qubit 0 is MSB in the bitstring from run_circuit.
        anc_val |= (1U << (opts.n_ancilla - 1 - b));
      }
    }
    if (anc_val < dim_anc) {
      result.measurement_counts[anc_val] += static_cast<std::uint64_t>(cnt);
    }
  }

  // Estimated phase = bin with highest count / 2^n_ancilla.
  const auto max_it = std::max_element(result.measurement_counts.begin(),
                                        result.measurement_counts.end());
  const std::uint32_t peak_bin = static_cast<std::uint32_t>(
      std::distance(result.measurement_counts.begin(), max_it));
  result.estimated_phase = static_cast<double>(peak_bin) / static_cast<double>(dim_anc);

  // True phase for "phase_gate": θ/(2π) mod 1.
  double true_phase = opts.phase_angle / (2.0 * 3.141592653589793);
  // Normalise to [0, 1).
  true_phase -= std::floor(true_phase);
  result.true_phase = true_phase;

  // Exact phase probabilities: run once without shots to get exact statevector.
  {
    QuantumCircuit circ_exact = build_phase_gate_qpe(opts);
    circ_exact.shots = 0;
    circ_exact.operations.erase(
        std::remove_if(circ_exact.operations.begin(), circ_exact.operations.end(),
                       [](const CircuitOp& op) { return op.name == "MEASURE"; }),
        circ_exact.operations.end());
    circ_exact.clbits = 0;
    const CircuitResult cr_exact = run_circuit(circ_exact, ro);
    const std::size_t dim_total = cr_exact.state.size();
    const std::size_t dim_tgt   = 1ULL << opts.n_target;
    for (std::uint32_t a = 0; a < dim_anc; ++a) {
      double prob = 0.0;
      for (std::size_t t = 0; t < dim_tgt; ++t) {
        const std::size_t idx = static_cast<std::size_t>(a) * dim_tgt + t;
        if (idx < dim_total) {
          prob += std::norm(cr_exact.state[idx]);
        }
      }
      result.phase_probs[a] = prob;
    }
  }

  result.ok = true;
  result.message = "QPE completed";

  const auto t1 = std::chrono::high_resolution_clock::now();
  result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  {
    std::ostringstream os;
    os << "QPE finished: estimated_phase=" << result.estimated_phase
       << " true_phase=" << result.true_phase
       << " elapsed_ms=" << result.elapsed_ms;
    log::info(os.str());
  }

  return result;
}

}  // namespace qx
