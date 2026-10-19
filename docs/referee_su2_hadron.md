# Referee feasibility: SU(2) hadron dynamics (M8, M16)

*As of 2026-10-20. The published numbers are from the sources listed at the end. The
bond-dimension scans were measured here, with quimb
([`benchmarks/su2_hadron_quimb.py`](../benchmarks/su2_hadron_quimb.py), M8) and with QuantX
([`benchmarks/su2_hadron_quantx.py`](../benchmarks/su2_hadron_quantx.py), M16).*

**Verdict.** The [Quantum Advantage Tracker](https://quantum-advantage-tracker.github.io/)'s
SU(2) hadron instance is within reach of the U(1) block-sparse engine, but a classical
simulation already matches it. An MPS simulation of the published circuits with quimb ran
in under an hour on a laptop, and the tracker accepted it after maintainer review. That
entry reported the conserved charge only. We measured what it left open, the physical
observable: the vacuum-subtracted fermion number after the last step, n_f(20).
- quimb's scan gives n_f(20) = 0.12 ± 0.04 at χ = 1600.
- QuantX's U(1) MPS of the same circuits keeps the charge exact. It gives
  n_f(20) = 0.117 ± 0.002 at χ = 1600, in 2.0 hours on one GPU.
- In both, the error bar is the change from the previous doubling of χ.

All three published classical values lie above it, by 0.13 to 0.29, and the QPU's −0.077
lies 0.19 below. That is no scatter around the answer. Step by step, the QuantX MPS has
converged through step 17, and every classical method already lies above it there, by up
to 0.20. The QuantX run refines a classical result; it does not provide a new one. And
this is the tracker's only instance with a conserved U(1) charge, so the tracker gives the
block-sparse engine little leverage today.

## The instance

| | |
|---|---|
| Tracker instance | `su2_hadron_dynamics_lsh_x_100_meson`. Its observable is the difference from a second circuit, `x_100_SCV`, the vacuum |
| Model | SU(2) lattice gauge theory in 1+1D, loop-string-hadron (LSH) encoding, at weak coupling x = 100 |
| Size | 60 sites on 120 qubits, two per site |
| Circuit | first-order Trotter, step 0.0015, 20 steps; two-qubit depth 259. As published it is transpiled: 11,840 CX, 10,720 RZ, 4,720 H, 2,462 X and 2,330 SWAP (meson circuit) |
| Conserved | the total charge Q = Σ(n_i + n_o) = 60 |
| Value tracked | Q: every entry, the QPU's included, submitted the conserved charge (59.994 ± 0.006 from the QPU), though the instance's README names n_f(20) its tracker-facing observable |
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

## Measured here with quimb: n_f(20) against the bond dimension

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
  per doubling, what is left is 0.003, but the error bar does not assume it. QuantX's
  scan below lands 0.0008 above quimb's χ = 1600 value.
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

## Measured here with QuantX's U(1) MPS (M16)

[`benchmarks/su2_hadron_quantx.py`](../benchmarks/su2_hadron_quantx.py) runs the same two
circuits on QuantX. The published QASM is transpiled, so its H, X and CX gates each break
the U(1) symmetry, which only whole gate clusters keep. `quantx_engine.u1_circuit` fuses
each cluster back into one gate that conserves the number of qubits in |1⟩. The meson
circuit's 32,072 gates (the vacuum's 32,070) become 8,290 such gates. The block-sparse MPS
then carries that number as an exact charge.

| χ | n_f(20) | Norm², meson / vacuum | Q, meson | Run time, both circuits |
|---|---|---|---|---|
| 200 | −0.91 | 0.044 / 0.026 | 60 | 3 min, CPU (2 threads) |
| 400 | 0.312 | 0.674 / 0.622 | 60 | 13 min, GPU |
| 800 | 0.119 | 0.966 / 0.959 | 60 | 38 min, GPU |
| 1600 | 0.117 | 0.9993 / 0.9992 | 60 | 2.0 h, GPU |

- **n_f(20) = 0.117 ± 0.002.** The error bar is the change from χ = 800, as in quimb's scan,
  where it was 0.04; quimb's χ = 1600 value, 0.116, lies inside it. QuantX's χ = 800 run is
  already within 0.003 of it, where quimb needed χ = 1600.
- **Q = 60 exactly,** at every χ and every step: no truncation can leave the charge
  sector. quimb's runs above lose 2e-4 of it at χ = 1600 and 9e-2 at χ = 200. Most of
  that leaks through the published circuit's intermediate states: given the fused gates,
  quimb keeps Q to 2e-7.
- **The gain at equal χ comes from the fusion, not the symmetry.** quimb truncates after
  every gate it is given. On the published circuit that is after each CX, where a
  cluster's intermediate states are more entangled than its end; on the fused circuit it
  is once per cluster. Given the fused gates, quimb keeps the norm QuantX keeps, to 3e-4
  in both circuits at every χ. It gives the same n_f(20) to 0.003: 0.315, 0.120 and 0.1170
  at χ = 400, 800 and 1600, against QuantX's 0.312, 0.119 and 0.1169. The symmetry
  contributes the exact charge.
- **Run time: the speed-up is the fusion's too.** On the same RTX 4090, both circuits
  take QuantX 13 min at χ = 400, 38 min at 800 and 2.0 h at 1600, against quimb's 15 min,
  46 min and 3.2 h on the published gates. Given the fused gates, quimb takes 10 min,
  28 min and 2.1 h: faster than QuantX up to χ = 800, and 6% slower at 1600. The engine's
  GPU path moves only its largest blocks to the GPU, so on this instance the charge blocks
  buy the exact charge, not speed. A block-sparse GPU backend (M12) is where they could.
  [referee_su2_hadron_quantx.csv](referee_su2_hadron_quantx.csv) holds every run's step 20.
- **quimb's default cutoff.** An open quimb MPS drops, at every split, up to cutoff times
  the state's weight (its cutoff_mode "rsum2"). At the scans' cutoff of 1e-10 and about
  700 splits per Trotter step, that is at most 7e-8 of the norm per step, and 6e-8 is
  lost. It moved quimb's n_f by up to 5e-5 at steps 7 to 10, well inside its 4e-4
  tolerance. With cutoff_mode "rel", which drops only singular values below the cutoff
  times the largest, quimb on the published gates and QuantX on the fused ones agree to
  6e-11 through step 9. The fusion reproduces the published circuits.
- **Every run twice, and once elsewhere.** This machine's memory can flip bits, so each
  QuantX run at χ ≥ 400 ran twice, and the repeats agree bit for bit. A CI job
  (`su2-hadron-quantx-repro.yml`) builds the engine on a GitHub runner. It runs χ = 400
  there on the CPU, where the runs above used the GPU, and holds steps 1 to 14 to the
  χ = 1600 trajectory within 1e-4.

## Step by step: the classical methods drift from the converged values

![n_f(t) of the SU(2) hadron circuits at every Trotter step: QuantX's U(1) MPS, converged through step 17, against the tensor network, both Pauli propagations and the QPU; below, each method minus the MPS](referee_su2_hadron_nf_t.svg)

The circuits are 20 blocks of 1,602 gates, one per Trotter step. After each of the first
19 blocks the SWAP gates have returned the qubits to their initial layout; the last block
lacks its final SWAP layer, which is the permutation above. So the `trajectory` commands
of both scripts read n_f after every step, and the tracker's README lists every method at
every step. [referee_su2_hadron_quantx_nf_t.csv](referee_su2_hadron_quantx_nf_t.csv) holds
the QuantX runs, which the figure shows; quimb's are in
[referee_su2_hadron_nf_t.csv](referee_su2_hadron_nf_t.csv). Both files hold the
published values too.

- **Converged through step 17.** QuantX's χ = 800 and 1600 agree to 3e-4 there, and at
  χ = 1600 the truncations leave the norm within 5e-6 of 1 in both circuits. Through step
  14 they agree to 6e-8. That is convergence to a stated tolerance, not an error bound,
  but the tolerance is a hundred times smaller than the drifts below. quimb's scan
  converged through step 14, to the same tolerance.
- **Step 1 checks the reading.** The MPS gives 1.8253, as both Pauli propagations do: to
  1e-10 against the CPU one's 1.825335615. The tensor network's 1.8242 evolves the
  Hamiltonian, not the circuit.
- **The classical methods drift above the converged values:** both Pauli propagations from
  step 2 on, and the tensor network from step 5 (before that it sits below, by up to
  0.005). The drifts at three steps:

  | | Step 10 | Step 17 | Step 20 |
  |---|---|---|---|
  | Pauli propagation, CPU | 0.157 | 0.20 | 0.29 |
  | Pauli propagation, GPU | 0.071 | 0.09 | 0.13 |
  | Tensor network | 0.051 | 0.08 | 0.16 |

  Part of the tensor network's difference is the circuit's Trotter error, which it does
  not simulate; the Pauli propagations run the circuit itself.
- **The QPU swings both ways,** by up to 0.32 (step 7).

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

1. **Offer the tracker n_f(20) = 0.117 ± 0.002, with both scans and the step-by-step
   comparison.** It is the first value of the physical observable with an error bar, from
   two independent MPS codes that agree inside it. Every published classical value lies
   outside that bar. The value the tracker records, Q, cannot separate the methods: a
   charge-conserving simulation gives exactly 60. This refines the classical side of the
   record; it does not refute the advantage claim.
2. **M16's simulation is done; the claim stays narrow:** an exactly charge-conserving
   simulation of n_f(t), not a refutation of an advantage claim. Whether to submit it is
   the maintainers' decision. The QASM-to-block converter it needed,
   `quantx_engine.u1_circuit`, is general: any number-conserving circuit given as QASM now
   runs on the block engine, which M14 (the CUDA-Q backend) needs too.
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
  downloads and checks by SHA-256. The step-by-step values are in
  [referee_su2_hadron_nf_t.csv](referee_su2_hadron_nf_t.csv), with the published ones from
  the instance's README; [`benchmarks/plot_su2_hadron.py`](../benchmarks/plot_su2_hadron.py)
  draws the figure and computes the drifts quoted above.
- The other instances: [circuit-models.json](https://github.com/quantum-advantage-tracker/quantum-advantage-tracker.github.io/blob/main/data/observable-estimations/circuit-models.json)
  and each family's README.
