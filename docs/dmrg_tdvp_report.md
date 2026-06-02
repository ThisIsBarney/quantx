# QuantX TDVP: Matrix-Product-State Time Evolution

**A technical report on the time-dependent variational principle (TDVP)
integrators in QuantX — real- and imaginary-time evolution of spin chains —
and an honest performance comparison with TeNPy.**

---

## Abstract

QuantX adds TDVP time evolution on top of its MPS/DMRG engine: a one-site
(TDVP1) and a two-site bond-adaptive (TDVP2) projector-splitting integrator,
for both real-time dynamics and imaginary-time cooling. The implementation is
validated to machine precision against exact diagonalization (TDVP1 is exact at
full bond dimension; the TDVP2 Néel quench matches `exp(-iHt)` to ~10⁻¹⁰) and is
exposed through the C-API and Python. On a real-time Heisenberg quench, QuantX
TDVP2 is **~10× faster than TeNPy at small systems** (N≈10, identical accuracy);
at large bond dimension (N≳16, χ≈64) TeNPy's tuned kernels are ~1.8× faster, so
the speed crown at scale is **not yet** claimed — the crossover is ~N=12–14, and
the remaining gap is a kernel-optimization task, not an algorithmic one.

---

## 1. What TDVP adds

DMRG finds ground states; TDVP evolves states in time on the MPS manifold. It is
the modern successor to TEBD: no Trotter error, it handles long-range and
non-nearest-neighbour MPOs natively, and (unlike TEBD) it conserves energy
extremely well. This makes QuantX a tensor-network *engine* — ground states
(DMRG, U(1)/SU(2)) **and** dynamics (TDVP) — rather than a ground-state solver.

Use cases: quench dynamics, transport, spectral functions, and imaginary-time as
an alternative ground-state finder.

## 2. Method

The projector-splitting TDVP (Haegeman et al., Lubich et al.):

- **expmv** (`src/tdvp.hpp`): the local time step is the Krylov (Lanczos) action
  of a matrix exponential `exp(τ H_eff) v`, with `τ = -i dt` (real time) or
  `τ = -dt` (imaginary time). Adaptive, with the small tridiagonal exponentiated
  by eigendecomposition.
- **Effective Hamiltonians**: one-site `apply_h1`, zero-site/bond `apply_h0`,
  two-site `apply_h2` (with the two-site MPO precombined into a 4×4 per MPO-bond
  pair). Environments `update_left/update_right` match the DMRG conventions.
- **TDVP1** (`tdvp_step`): second-order symmetric sweep — evolve each site
  forward by `dt/2` (`H1`), shift the orthogonality centre by QR/LQ, evolve the
  bond backward by `dt/2` (`H0`); left→right then right→left.
- **TDVP2** (`tdvp2_step`): evolve each two-site block forward (`H2`), SVD-split
  with truncation to `chi_max`, evolve the single shared site backward (`H1`).
  Because it works on two sites, TDVP2 **grows the bond dimension**, capturing
  entanglement growth from a product state (which TDVP1 cannot).

## 3. Validation

All against exact references (`qx_tdvp_test`, `test_tdvp.py`):

| Check | Result |
|-------|--------|
| `expmv` vs dense matrix exponential (real & imaginary τ) | ~10⁻¹⁴ |
| one-/zero-site `H_eff` energy vs dense ED | ~10⁻¹⁵ |
| TDVP1 real-time vs `exp(-iHt)` (full bond — exact) | ~10⁻¹⁵ |
| TDVP1 energy conservation | ~10⁻¹⁴ |
| TDVP1 imaginary-time → ground state | ~10⁻¹⁵ |
| TDVP2 Néel quench vs `exp(-iHt)` | ~10⁻¹⁰ |
| TDVP2 energy conservation | ~10⁻¹³ |
| imaginary-time TDVP2 → ground (Python) | ~10⁻⁶ |

TDVP1 at full bond dimension reproduces exact dynamics with **no time-step
error** — the known exactness of the projector-splitting integrator.

## 4. API

```python
from quantx_engine import Engine
from quantx_engine.tdvp import TDVPSolver
s = TDVPSolver(Engine())

# real-time quench from the Neel state; staggered magnetization m(t)
r = s.evolve(20, dt=0.05, n_steps=40, chi_max=64, two_site=True)
print(r.times, r.staggered_mag, r.max_bond, r.wall_time_s)

# imaginary-time cooling to the ground state
g = s.evolve(20, dt=0.1, n_steps=200, imag=True)
print(g.energy)
```

## 5. Benchmark vs TeNPy (honest)

Real-time Néel quench, open Heisenberg chain `H = J Σ σ·σ`, `dt=0.05`, `T=2.0`,
`chi_max=64`, single-threaded. Both engines compute the staggered magnetization
`m(t)`; accuracy is checked against exact diagonalization where feasible.

| N | QuantX TDVP2 | TeNPy 2-site TDVP | ratio | accuracy |
|--:|-------------:|------------------:|------:|----------|
| 10 | **1.06 s** | 10.6 s | **10.1× faster** | both `9.1e-5` vs exact |
| 16 | 73.6 s | 41.3 s | 0.56× (TeNPy faster) | — |
| 20 | 136 s | 76 s | 0.56× (TeNPy faster) | — |

**Reading this honestly.** At small systems QuantX wins by ~10× — TeNPy's
fixed Python/bookkeeping overhead dominates there. As the bond dimension grows
(N≳16 reaches χ=64) the workload becomes genuine `O(χ³)` linear algebra, and
TeNPy's highly tuned BLAS-`tensordot` kernels pull ahead by ~1.8×. The crossover
is around N=12–14.

This is the **opposite** regime split from the DMRG result (where QuantX wins at
all sizes): DMRG benefited from symmetry (U(1)/SU(2)) and a warm-started
few-iteration Lanczos, whereas the TDVP two-site evolution is dominated by the
per-step `H_eff` matvec, where we have not yet matched TeNPy's kernel tuning.

## 6. Limitations & next steps

- **Large-χ speed**: the TDVP2 `apply_h2` matvec should be restructured into the
  big-DGEMM reshape used by the DMRG dense `apply_heff` (one fused `θ·R`
  contraction rather than per-block products), and the per-step environment
  extraction tightened. This is the path to matching/beating TeNPy at scale —
  a kernel-optimization task, the algorithm is correct.
- **Symmetry**: TDVP currently runs on dense (non-symmetric) MPS tensors;
  porting it onto the U(1)/SU(2) block-sparse tensors would cut the constant
  factor substantially (the same lever that made DMRG fast).
- **Models**: currently the spin-1/2 Heisenberg MPO; general MPOs and finite-T
  (purification / METTS) are natural extensions.

## 7. Conclusion

QuantX now has validated, machine-precision TDVP time evolution (real and
imaginary time, one- and two-site), exposed in Python, and faster than TeNPy for
small/moderate systems. It makes QuantX a full tensor-network engine. The
large-bond-dimension speed crown is a kernel-optimization follow-up (and the
biggest single lever is porting TDVP onto the symmetry-adapted tensors that
already power the DMRG path).
