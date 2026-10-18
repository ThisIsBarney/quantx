# Referee feasibility: SU(2) hadron dynamics (M8)

*As of 2026-10-18. The published numbers are from the sources listed at the end; the
bond-dimension scan was measured here, with
[`benchmarks/su2_hadron_quimb.py`](../benchmarks/su2_hadron_quimb.py).*

**Verdict.** The [Quantum Advantage Tracker](https://quantum-advantage-tracker.github.io/)'s
SU(2) hadron instance is within reach of the U(1) block-sparse engine, but a classical
simulation already matches it: an MPS simulation of the published circuits with quimb
ran in under an hour on a laptop, and the tracker accepted it after maintainer review.
That entry reported the conserved charge only. We measured what it left open, the bond
dimension the physical observable needs. The vacuum-subtracted fermion number after the
last step is n_f(20) = 0.12 ± 0.04: 0.630 at χ = 400, 0.154 at χ = 800 and 0.116 at
χ = 1600, where quimb's fidelity estimate reaches 0.995, the error bar being the last
doubling's change. All three published classical values lie above it, by 0.13 to 0.29,
and the QPU's −0.077 lies 0.19 below. A QuantX run would add an exactly conserved
charge: it would refine a classical result, not provide a new one. And this is the
tracker's only instance with a conserved U(1) charge, so the tracker gives the
block-sparse engine little leverage today.

## The instance

| | |
|---|---|
| Tracker instance | `su2_hadron_dynamics_lsh_x_100_meson`. Its observable is the difference from a second circuit, `x_100_SCV`, the vacuum |
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
| MPS simulation of the circuits (quimb), χ = 800, Q converged by its submitter's comparison | not reported (0.154 at its χ in the scan below) | 3,003 s, both circuits | Apple M3 Pro laptop |

- The three published classical methods spread by 0.16 at step 20.
- The QPU sits 0.32 to 0.49 below them.
- The tensor network's truncation error reaches 1e-4 per step by step 18 at D = 200. At
  x = 200 it reaches 1e-2 by step 13, and the paper reports it breaking down after step 10.

The tracker's README for this instance swaps the run-time labels, giving the tensor
network as 2,484 s. The paper's Table 2 and the CPU Pauli-propagation submission give
2,484 s to Pauli propagation on the CPU.

## Measured here: n_f(20) against the bond dimension

[`benchmarks/su2_hadron_quimb.py`](../benchmarks/su2_hadron_quimb.py) simulates both
published circuits as a quimb MPS truncated at bond dimension χ, and reads n_f(20) from
the final states. Its fidelity estimate is the norm the truncations leave.

| χ | n_f(20) | Fidelity estimate, meson / vacuum | Q, meson | Run time, both circuits |
|---|---|---|---|---|
| 200 | −2.77 | 0.010 / 0.005 | 59.934 | 7 min, GPU |
| 400 | 0.630 | 0.557 / 0.515 | 59.995 | 85 min, CPU (2 threads) |
| 800 | 0.154 | 0.863 / 0.833 | 59.9986 | 46 min, GPU |
| 1600 | 0.116 | 0.996 / 0.995 | 59.9998 | 3.2 h, GPU |

- **n_f(20) = 0.12 ± 0.04.** Each doubling of χ moves it less: by −0.48 from 400 to
  800, and by −0.038 from 800 to 1600, where no single site moves by more than 0.003.
  The error bar is that last change. If the convergence keeps its last ratio, about 0.08
  per doubling, what is left is 0.003, but the error bar does not assume it.
- **Against the published values,** at step 20: Pauli propagation on the CPU 0.408
  (0.29 above), the tensor network 0.278 (0.16 above), Pauli propagation on the GPU 0.246
  (0.13 above), the QPU −0.077 (0.19 below). The tensor network evolves the Hamiltonian
  rather than the circuit, so part of its difference can be the circuit's Trotter error.
  The Pauli propagations and the QPU run the circuit itself.
- **The accepted entry.** At χ = 800 the scan reproduces the tracker's quimb entry, Q =
  59.9986 against its 59.9985. Its n_f(20), which it did not report, would be 0.154: 0.04
  above the χ = 1600 value.
- **Why n_f converges slowly.** It sums 60 sites, and its truncation error sits where the
  quark and antiquark are. From χ = 400 to 800 it falls by about 0.025 on each of the nine
  sites at either end, while the 30 sites in between move by 0.003 in all.

How the circuits are read:

- The meson circuit is the vacuum circuit plus X on qubits 59 and 60: a quark on site 0
  and an antiquark on site 59, the two ends of the chain. By step 20 the difference from
  the vacuum has spread about 15 sites in from each end. At χ = 1600 the profile is
  mirror-symmetric to 2e-5.
- The transpiled circuits end with their qubits permuted. Their 2,330 SWAP gates leave 60
  of the 120 qubits elsewhere (pairs (1, 2), (5, 6), …), so the final ⟨Z⟩ are mapped back
  before n_f is formed. Read without the permutation, the sum stays at its starting value
  2.00 at every χ: a quantity the circuit conserves, not n_f.
- The GPU runs keep the MPS in complex128 on an RTX 4090, and split each gate by an
  eigendecomposition of its Gram matrix, which cuSOLVER does far faster than an SVD.
  At χ = 200 and 400 they agree with the CPU's SVD to 3e-3 in n_f(20), and at χ = 400
  the fidelity estimates agree to 2e-4. χ = 400 takes 15 min there, against 85 min on
  two CPU threads.

## What a QuantX run would add, and what it needs

- **An exact charge.** The circuit conserves Q, and a U(1) block-sparse MPS keeps
  Q = 60 exactly at any truncation; the quimb runs above lose 2e-4 of it at χ = 1600 and
  9e-2 at χ = 200.
- **The same error bar for less.** The scan above needed χ = 1600 for an error bar of
  0.04 on n_f(20). A charge-conserving MPS at that χ splits every tensor into charge
  blocks; how much faster it would run is not measured.
- **A converter for the circuit.** The published QASM is transpiled, so its H, X and CX
  gates each break the U(1) symmetry, which only whole gate clusters keep. A converter
  would fuse the gates of each cluster back into one charge-conserving block, check it,
  and pick a qubit order in which the blocks are short-ranged. It is a general tool: any
  number-conserving circuit given as QASM would run on the block engine through it.

quimb's run times, without symmetry, are the bound to beat. They are 46 min at χ = 800 and
3.2 h at χ = 1600 on one RTX 4090 in complex128, which that GPU runs at 1/64 of its
float32 rate. The submitter's χ = 800 took 3,003 s on a laptop. A QuantX run's time is
not measured: it needs the converter first.

## The tracker as a referee market

| Observable-estimation family | Instances | Qubits | Lattice | Conserved U(1) charge |
|---|---|---|---|---|
| `floquet_mixed_field_ising` | 5 | 51, 74 | heavy-hex | none: a transverse field |
| `operator_loschmidt_echo` | 4 | 49, 56, 70 | 2D (heavy-hex or square) | none: X rotations |
| `su2_hadron_dynamics_lsh` | 1, with its vacuum circuit | 120 | chain | Q |

The classically verifiable problems (peaked circuits, and random-graph sampling) are
generic circuits, arbitrary single-qubit rotations between CZ gates, with no symmetry to
exploit either.

## Recommendation

1. **Offer the tracker n_f(20) = 0.12 ± 0.04, with this scan.** It is the first value of
   the physical observable with an error bar, and every published classical value lies
   outside that bar. It refines the classical side of the record. It does not refute the
   advantage claim, which rests on Q.
2. **M16 keeps the SU(2) instance, with a narrower claim:** an exactly charge-conserving
   simulation of n_f(t), not a refutation of an advantage claim. The work is the
   QASM-to-block converter, which M14 (the CUDA-Q backend) needs too.
3. **Ask the experiment's authors, or the tracker, for x = 200 or longer times.** That is
   where the paper's tensor network broke down after step 10, and where its Pauli
   propagation grew sensitive to truncation. A converged classical value would be new
   there.
4. **Refereeing needs symmetric candidates, and the tracker has one, already matched.**
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
  the quimb MPS simulation (R. Farrell, Caltech; accepted after maintainer review, with
  no error bound),
  [#227](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/issues/227).
- The scan: [`benchmarks/su2_hadron_quimb.py`](../benchmarks/su2_hadron_quimb.py)
  (quimb 1.11.2; CuPy 14 for the GPU runs), its runs in
  [referee_su2_hadron_nf.csv](referee_su2_hadron_nf.csv), measured 2026-10-18. Its inputs
  are the two QASM files at the tracker's commit a62dc46 (2026-07-07), which the script
  downloads and checks by SHA-256.
- The other instances: [circuit-models.json](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/blob/main/data/observable-estimations/circuit-models.json)
  and each family's README.
