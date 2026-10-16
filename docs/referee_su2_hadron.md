# Referee feasibility: SU(2) hadron dynamics (M8)

*As of 2026-10-16. Every number below is from the sources listed at the end.*

**Verdict.** The [Quantum Advantage Tracker](https://quantum-advantage-tracker.github.io/)'s
SU(2) hadron instance is within reach of the U(1) block-sparse engine, but it is
already classically matched: an MPS simulation of the published circuits with quimb,
converged by χ = 800, ran in under an hour on a laptop, and the tracker lists it as
verified. A QuantX run would add an exactly conserved charge and an error bar on the
physical observable, where the published classical methods disagree at the last step.
That would refine a classical result, not provide a new one. It is also the tracker's
only instance with a conserved U(1) charge, so the tracker gives the block-sparse engine
little leverage today.

## The instance

| | |
|---|---|
| Tracker ids | `su2_hadron_dynamics_lsh_x_100_meson`, and `_SCV`, the vacuum it is subtracted from |
| Model | SU(2) lattice gauge theory in 1+1D, loop-string-hadron (LSH) encoding, at weak coupling x = 100 |
| Size | 60 sites on 120 qubits, two per site |
| Circuit | first-order Trotter, step 0.0015, 20 steps; two-qubit depth 259. As published it is transpiled: 11,840 CX, 10,720 RZ, 4,720 H, 2,462 X and 2,330 SWAP (meson circuit) |
| Conserved | the total charge Q = Σ(n_i + n_o) = 60 |
| Value submitted | Q itself: 59.994 ± 0.006 from the QPU at step 20 |
| Physical observable | the vacuum-subtracted fermion density n_f(t) |

## The published baselines, at step 20

| Method | n_f(20) | Run time | Hardware |
|---|---|---|---|
| QPU | −0.077 | 20 s per step | IBM Boston |
| Pauli propagation, CPU | 0.408 | 2,484 s | HP Z2 Tower G9 |
| Pauli propagation, GPU (cuPauliProp, coefficients below 1e-5 dropped) | 0.246 | 2,930 s | NVIDIA RTX 5000 Ada |
| Tensor network: 2-site TDVP of the full LSH Hamiltonian (ITensors.jl), D = 200 | 0.278 | 8,949 s | Mac Pro, 16-core Xeon W |
| MPS simulation of the circuits (quimb), χ = 800 | not reported | 3,003 s, both circuits | Apple M3 Pro laptop |

- The three published classical methods spread by 0.16 at step 20.
- The QPU sits 0.32 to 0.49 below them.
- The tensor network's truncation error reaches 1e-4 per step by step 18 at D = 200. At
  x = 200 it reaches 1e-2 by step 13, and the paper reports it breaking down after step 10.

The tracker's README for this instance swaps the run-time labels, giving the tensor
network as 2,484 s. The paper's Table 2 and the CPU Pauli-propagation submission give
2,484 s to Pauli propagation on the CPU.

## What a QuantX run would add, and what it needs

- **An exact charge.** The circuit conserves Q, and a U(1) block-sparse MPS keeps
  Q = 60 exactly at any truncation; the quimb run reports 59.9985.
- **An error bar on n_f(t).** The tracker records only Q. A χ-converged n_f(t), with
  the change from χ = 800 to 1600 as its error bar, would settle which published
  method is right at step 20.
- **A converter for the circuit.** The published QASM is transpiled, so its H, X and CX
  gates each break the U(1) symmetry, which only whole gate clusters keep. A converter
  would fuse the gates of each cluster back into one charge-conserving block, check it,
  and pick a qubit order in which the blocks are short-ranged. It is a general tool: any
  number-conserving circuit given as QASM would run on the block engine through it.

The run time is not measured. quimb's 3,003 s on a laptop, at χ = 800 without
symmetry, is the bound to beat.

## The tracker as a referee market

| Observable-estimation family | Instances | Qubits | Lattice | Conserved U(1) charge |
|---|---|---|---|---|
| `floquet_mixed_field_ising` | 5 | 51, 74 | heavy-hex | none: a transverse field |
| `operator_loschmidt_echo` | 4 | 49, 56, 70 | 2D (heavy-hex or square) | none: X rotations |
| `su2_hadron_dynamics_lsh` | 2 | 120 | chain | Q |

The classically verifiable problems (peaked circuits, and random-graph sampling) are
generic circuits, arbitrary single-qubit rotations between CZ gates, with no symmetry to
exploit either.

## Recommendation

1. **M16 keeps the SU(2) instance, with a narrower claim:** an exactly charge-conserving
   simulation with an error-barred n_f(t), not a refutation of an advantage claim. The
   work is the QASM-to-block converter, which M14 (the CUDA-Q backend) needs too.
2. **Ask the experiment's authors, or the tracker, for x = 200 or longer times.** That is
   where the paper's tensor network broke down after step 10, and where its Pauli
   propagation grew sensitive to truncation. A converged classical value would be new
   there.
3. **Refereeing needs symmetric candidates, and the tracker has one, already matched.**
   Watch for number-conserving fermionic or gauge-theory candidates, and do not build
   for the generic instances: on those the block-sparse engine has no edge over quimb
   or Pauli propagation.

## Sources

- The experiment: F. Ilčić, R. Majumdar, E. Mathew, M. O. Ali, N. Earnest-Noble and
  I. Raychowdhury, *Observation of Robust and Coherent Non-Abelian Hadron Dynamics on
  Noisy Quantum Processors*, [arXiv:2602.18080](https://arxiv.org/abs/2602.18080) (v3,
  2026-05-29). Table 2 gives the run times; the supplementary information gives the
  tensor-network truncation errors.
- The tracker instance and its README:
  [data/observable-estimations/circuit-models/su2_hadron_dynamics_lsh](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/tree/main/data/observable-estimations/circuit-models/su2_hadron_dynamics_lsh).
  The gate counts are from `x_100_meson.qasm` there.
- The submissions: the QPU result,
  [#149](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/issues/149);
  Pauli propagation on the CPU,
  [#202](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/issues/202);
  the quimb MPS simulation (R. Farrell, Caltech; verified),
  [#227](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/issues/227).
- The other instances: [circuit-models.json](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/blob/main/data/observable-estimations/circuit-models.json)
  and each family's README.
