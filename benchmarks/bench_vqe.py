"""Benchmark 3: VQE / variational optimisation — QuantX vs. PennyLane.

Scenario A — Noise-free UCCSD-style VQE:
  Minimise the ground-state energy of a 2-orbital H2-like Hamiltonian using
  QuantX OOVQESolver (orbital-optimised VQE).

Scenario B — Noise-aware VQE:
  Same Hamiltonian but with depolarising noise injected at each layer.
  Compared against PennyLane DefaultQubit (if installed).
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import numpy as np
from bench_utils import BenchResult, print_table, save_results, timer

# 2-orbital active-space core Hamiltonian for a minimal H2 model
# h_core[i,j] = one-electron integral <i|h|j>
H2_H_CORE = np.array([[-1.2528, 0.0],
                       [0.0,    -0.4759]])
N_CIRCUIT_PARAMS = 2
N_QUBITS = 2


def bench_quantx_oovqe() -> BenchResult:
    """OO-VQE on a 2-orbital H2 core Hamiltonian."""
    try:
        from quantx_engine.oovqe import OOVQESolver
        solver = OOVQESolver(
            h_core=H2_H_CORE,
            n_orb=2,
            n_circuit_params=N_CIRCUIT_PARAMS,
            max_iterations=30,
            inner_steps=20,
        )
        with timer() as elapsed:
            result = solver.run(seed=42)
        return BenchResult(
            benchmark="vqe_h2",
            backend="quantx_oovqe",
            params={"n_orb": 2, "max_iter": 30},
            wall_time_s=elapsed[0],
            result_value=float(result.final_energy),
        )
    except Exception as exc:
        return BenchResult("vqe_h2", "quantx_oovqe", {"n_orb": 2}, 0.0, error=str(exc))


def bench_quantx_noise_vqe() -> BenchResult:
    """Noise-aware VQE on the 2-qubit H2 Hamiltonian."""
    try:
        from quantx_engine.noise_vqe import NoiseAwareVQE
        from quantx_engine.advanced_chemistry import build_hamiltonian_matrix
        from quantx_engine.chemistry import PauliHamiltonian, PauliTerm
        # Minimal H2 Pauli Hamiltonian: H = -1.05 I + 0.39 Z0 - 0.39 Z1 + 0.18 X0X1
        ham = PauliHamiltonian([
            PauliTerm(-1.05, {}),
            PauliTerm(0.39,  {0: 'Z'}),
            PauliTerm(-0.39, {1: 'Z'}),
            PauliTerm(0.18,  {0: 'X', 1: 'X'}),
        ])
        H_mat = build_hamiltonian_matrix(ham)
        solver = NoiseAwareVQE(
            hamiltonian=H_mat,
            n_qubits=N_QUBITS,
            n_layers=2,
            max_iter=80,
            seed=42,
        )
        with timer() as elapsed:
            result = solver.run()
        return BenchResult(
            benchmark="vqe_h2",
            backend="quantx_noise_vqe",
            params={"n_qubits": N_QUBITS, "n_layers": 2},
            wall_time_s=elapsed[0],
            result_value=float(result.energy),
        )
    except Exception as exc:
        return BenchResult("vqe_h2", "quantx_noise_vqe", {"n_qubits": N_QUBITS}, 0.0, error=str(exc))


def bench_pennylane() -> BenchResult:
    """PennyLane DefaultQubit VQE on the same Pauli Hamiltonian."""
    try:
        import pennylane as qml
        from pennylane import numpy as pnp

        H_coeffs = np.array([-1.05, 0.39, -0.39, 0.18])
        H_ops = [
            qml.Identity(0),
            qml.PauliZ(0),
            qml.PauliZ(1),
            qml.PauliX(0) @ qml.PauliX(1),
        ]
        H = qml.Hamiltonian(H_coeffs, H_ops)
        dev = qml.device("default.qubit", wires=N_QUBITS)

        @qml.qnode(dev)
        def circuit(params):
            qml.RY(params[0], wires=0)
            qml.RY(params[1], wires=1)
            qml.CNOT(wires=[0, 1])
            return qml.expval(H)

        params = pnp.array([0.1, 0.2], requires_grad=True)
        opt = qml.GradientDescentOptimizer(stepsize=0.1)

        with timer() as elapsed:
            for _ in range(80):
                params = opt.step(circuit, params)
            energy = float(circuit(params))

        return BenchResult(
            benchmark="vqe_h2",
            backend="pennylane",
            params={"n_qubits": N_QUBITS, "steps": 80},
            wall_time_s=elapsed[0],
            result_value=energy,
        )
    except ImportError:
        return BenchResult("vqe_h2", "pennylane", {"n_qubits": N_QUBITS}, 0.0,
                           skipped=True, skip_reason="pennylane not installed")
    except Exception as exc:
        return BenchResult("vqe_h2", "pennylane", {"n_qubits": N_QUBITS}, 0.0, error=str(exc))


def run(quick: bool = False) -> list[BenchResult]:
    results = []
    if not quick:
        # OO-VQE involves alternating orbital + circuit optimization — slower
        print("  OO-VQE (QuantX)", flush=True)
        results.append(bench_quantx_oovqe())
    print("  Noise-aware VQE (QuantX)", flush=True)
    results.append(bench_quantx_noise_vqe())
    print("  PennyLane DefaultQubit", flush=True)
    results.append(bench_pennylane())
    return results


if __name__ == "__main__":
    print("Benchmark 3: VQE / variational optimisation")
    results = run()
    print_table(results)
    save_results(results, "benchmarks/results_vqe.json")
