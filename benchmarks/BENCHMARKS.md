# QuantX Engine — Benchmark Report

_Generated on 2026-05-21 03:49 UTC_

## Hardware / Software Environment

| Key | Value |
|-----|-------|
| platform | Windows-11-10.0.26200-SP0 |
| python | 3.14.3 |
| numpy | 2.4.4 |
| cpu_count | 16 |
| quantx_version | 0.2.0 |
| qiskit_version | — |
| pennylane_version | — |
| qutip_version | — |

## Methodology

Wall-clock time is measured with `time.perf_counter()` around the **hot path
only** — matrix construction, I/O, and library imports are excluded.
Each benchmark runs a single trial (no warm-up) to reflect cold-start latency
typical in scientific workflows.  Competitor libraries (SciPy, QuTiP,
PennyLane, Stim, TeNPy) are imported at runtime; if a library is not installed
the row is marked **SKIP** and QuantX timings are still reported.

To reproduce:
```bash
# full suite
python benchmarks/run_all_benchmarks.py

# single suite
python benchmarks/run_all_benchmarks.py --suite lanczos

# smoke test (small sizes)
python benchmarks/run_all_benchmarks.py --quick
```

## Benchmark 1: Heisenberg Lanczos eigensolver

Ground-state energy of a 1D Heisenberg chain computed with
Lanczos (iterative sparse diagonalisation, k=1 eigenvalue).
Result column shows E₀ (lower is more negative for antiferromagnet).

| Benchmark | Backend | Params | Time (s) | Result |
|-----------|---------|--------|----------|--------|
| heisenberg_lanczos | quantx | n_sites=4 | 0.010 | -3.2321 |
| heisenberg_lanczos | scipy | n_sites=4 | — | SKIP (scipy not installed) |
| heisenberg_lanczos | qutip | n_sites=4 | — | SKIP (qutip not installed) |
| heisenberg_lanczos | quantx | n_sites=6 | 0.006 | -4.9872 |
| heisenberg_lanczos | scipy | n_sites=6 | — | SKIP (scipy not installed) |
| heisenberg_lanczos | qutip | n_sites=6 | — | SKIP (qutip not installed) |
| heisenberg_lanczos | quantx | n_sites=8 | 0.027 | -6.7499 |
| heisenberg_lanczos | scipy | n_sites=8 | — | SKIP (scipy not installed) |
| heisenberg_lanczos | qutip | n_sites=8 | — | SKIP (qutip not installed) |
| heisenberg_lanczos | quantx | n_sites=10 | 0.378 | -8.5161 |
| heisenberg_lanczos | scipy | n_sites=10 | — | SKIP (scipy not installed) |
| heisenberg_lanczos | qutip | n_sites=10 | — | SKIP (qutip not installed) |
| heisenberg_lanczos | quantx | n_sites=12 | 3.060 | -10.2842 |
| heisenberg_lanczos | scipy | n_sites=12 | — | SKIP (scipy not installed) |
| heisenberg_lanczos | qutip | n_sites=12 | — | SKIP (qutip not installed) |

## Benchmark 2: Ising Trotter time evolution

Norm-preserving second-order Trotter evolution of a transverse-field
Ising chain from t=0 to t=1 with n\_steps=100.  Result column shows
‖ψ(t=1)‖, which must equal 1.0 (unitarity check).

| Benchmark | Backend | Params | Time (s) | Result |
|-----------|---------|--------|----------|--------|
| ising_trotter | quantx | n_qubits=4, n_steps=100 | 0.000 | 1.0000 |
| ising_trotter | scipy_expm | n_qubits=4 | — | SKIP (scipy not installed) |
| ising_trotter | qutip | n_qubits=4 | — | SKIP (qutip not installed) |
| ising_trotter | quantx | n_qubits=6, n_steps=100 | 0.019 | 1.0000 |
| ising_trotter | scipy_expm | n_qubits=6 | — | SKIP (scipy not installed) |
| ising_trotter | qutip | n_qubits=6 | — | SKIP (qutip not installed) |
| ising_trotter | quantx | n_qubits=8, n_steps=100 | 0.092 | 1.0000 |
| ising_trotter | scipy_expm | n_qubits=8 | — | SKIP (scipy not installed) |
| ising_trotter | qutip | n_qubits=8 | — | SKIP (qutip not installed) |
| ising_trotter | quantx | n_qubits=10, n_steps=100 | 2.030 | 1.0000 |
| ising_trotter | scipy_expm | n_qubits=10 | — | SKIP (scipy not installed) |
| ising_trotter | qutip | n_qubits=10 | — | SKIP (qutip not installed) |

## Benchmark 3: VQE variational optimisation

Variational ground-state optimisation on a minimal H₂ Hamiltonian.
OO-VQE uses an orbital-rotation outer loop; Noise-aware VQE adds a
depolarising noise model.  Result column shows final energy (Hartree).

| Benchmark | Backend | Params | Time (s) | Result |
|-----------|---------|--------|----------|--------|
| vqe_h2 | quantx_oovqe | n_orb=2, max_iter=30 | 0.009 | -0.3884 |
| vqe_h2 | quantx_noise_vqe | n_qubits=2, n_layers=2 | 0.089 | -0.9659 |
| vqe_h2 | pennylane | n_qubits=2 | — | SKIP (pennylane not installed) |

## Benchmark 4: QEC logical error rate

Monte-Carlo simulation of syndrome extraction for a repetition code
at distances d = 3, 5, 7, 9 and three physical error rates.
Result column shows the logical error rate at the **highest** distance.

| Benchmark | Backend | Params | Time (s) | Result |
|-----------|---------|--------|----------|--------|
| qec_logical_rate | quantx | distances=[3, 5, 7, 9], p_phys=0.001, n_shots=500 | 0.002 | 0.0000 |
| qec_logical_rate | stim | distances=[3, 5, 7, 9], p_phys=0.001 | — | SKIP (stim not installed) |
| qec_logical_rate | quantx | distances=[3, 5, 7, 9], p_phys=0.05, n_shots=500 | 0.003 | 0.0000 |
| qec_logical_rate | stim | distances=[3, 5, 7, 9], p_phys=0.05 | — | SKIP (stim not installed) |
| qec_logical_rate | quantx | distances=[3, 5, 7, 9], p_phys=0.1, n_shots=500 | 0.003 | 0.0000 |
| qec_logical_rate | stim | distances=[3, 5, 7, 9], p_phys=0.1 | — | SKIP (stim not installed) |

## Benchmark 5: MPS GHZ fidelity

GHZ state fidelity computed via MPS with bond dimension 2 (exact
for GHZ).  Tests linear-in-n scaling of the MPS contraction engine.
Result column shows ⟨ψ_exact|ψ_MPS⟩² (should be 1.0 for all sizes).

| Benchmark | Backend | Params | Time (s) | Result |
|-----------|---------|--------|----------|--------|
| mps_ghz_fidelity | quantx | n_qubits=4, max_bond=2 | 0.001 | 1.0000 |
| mps_ghz_fidelity | tenpy | n_qubits=4, max_bond=2 | — | SKIP (tenpy not installed) |
| mps_ghz_fidelity | quantx | n_qubits=8, max_bond=2 | 0.001 | 1.0000 |
| mps_ghz_fidelity | tenpy | n_qubits=8, max_bond=2 | — | SKIP (tenpy not installed) |
| mps_ghz_fidelity | quantx | n_qubits=12, max_bond=2 | 0.002 | 1.0000 |
| mps_ghz_fidelity | tenpy | n_qubits=12, max_bond=2 | — | SKIP (tenpy not installed) |
| mps_ghz_fidelity | quantx | n_qubits=16, max_bond=2 | 0.001 | 1.0000 |
| mps_ghz_fidelity | tenpy | n_qubits=16, max_bond=2 | — | SKIP (tenpy not installed) |
| mps_ghz_fidelity | quantx | n_qubits=20, max_bond=2 | 0.002 | 1.0000 |
| mps_ghz_fidelity | tenpy | n_qubits=20, max_bond=2 | — | SKIP (tenpy not installed) |
| mps_ghz_fidelity | quantx | n_qubits=32, max_bond=2 | 0.003 | 1.0000 |
| mps_ghz_fidelity | tenpy | n_qubits=32, max_bond=2 | — | SKIP (tenpy not installed) |
| mps_ghz_fidelity | quantx | n_qubits=64, max_bond=2 | 0.006 | 1.0000 |
| mps_ghz_fidelity | tenpy | n_qubits=64, max_bond=2 | — | SKIP (tenpy not installed) |

