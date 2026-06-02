# QuantX TDVP: Matrix-Product-State Time Evolution

**A technical report on the time-dependent variational principle (TDVP)
integrators in QuantX — real- and imaginary-time evolution of spin chains —
and an honest performance comparison with TeNPy.**

---

## Abstract

QuantX adds TDVP time evolution on top of its MPS/DMRG engine: a one-site
(TDVP1) and a two-site bond-adaptive (TDVP2) projector-splitting integrator,
for both real-time dynamics and imaginary-time cooling, with a **Sz-symmetric
(block-sparse) TDVP2** as the default fast path. The implementation is validated
to machine precision against exact diagonalization (TDVP1 is exact at full bond
dimension; the TDVP2 Néel quench matches `exp(-iHt)` to ~10⁻⁶–10⁻¹⁰) and is
exposed through the C-API and Python. On a real-time Heisenberg quench
(single-thread, identical accuracy), **block-sparse QuantX TDVP2 is 8–50× faster
than TeNPy across all tested sizes** (N=10–20, χ up to 64). The block-sparsity is
the decisive lever — the same one that powers the DMRG path — turning a dense
implementation that *lost* at large bond dimension into one that wins everywhere.

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

**Block-sparse TDVP2 (default)** vs TeNPy, and the earlier dense path for
contrast:

| N | QuantX TDVP2 (block) | TeNPy 2-site TDVP | speedup | (dense TDVP2) |
|--:|---------------------:|------------------:|--------:|--------------:|
| 10 | **0.18 s** | 8.99 s | **50×** | 1.06 s (10×) |
| 16 | **3.78 s** | 41.1 s | **11×** | 73.6 s (0.6×) |
| 20 | **7.94 s** | 65.6 s | **8.3×** | 136 s (0.6×) |

Accuracy is identical (e.g. N=10 staggered-magnetization error `9.1e-5` vs exact
for both engines).

**The decisive lever.** The original *dense* TDVP2 won at small N (Python
overhead dominates TeNPy there) but lost ~1.8× at large bond dimension, where
the workload is genuine `O(χ³)` linear algebra. Exploiting **Sz conservation**
(block-sparse tensors) — so every contraction, SVD and environment update only
touches symmetry-allowed blocks — cut the constant factor enough to flip N=20
from **0.6× (losing) to 8.3× (winning)**, a ~17× improvement. This is exactly the
lever that makes the DMRG path fast; the block-sparse TDVP2 reuses the very same
DMRG block kernels (`form_theta`, `apply_heff_plan`, `apply_heff_1site`,
`block_svd_truncate`, block environments) with `expmv` as the local time step.

## 6. Limitations & next steps

- **Done — Sz block-sparse TDVP2**: the default fast path; wins at all tested
  sizes (§5). The dense path is retained for non-Néel initial states and one-site
  TDVP1.
- **SU(2) TDVP**: TDVP2 currently uses U(1) (Sz) symmetry; running it on the
  SU(2) multiplet tensors (as the SU(2) DMRG already does) would cut the constant
  factor further for spin-isotropic dynamics.
- **Models**: currently the spin-1/2 Heisenberg MPO; general MPOs and
  finite-temperature (purification / METTS) are natural extensions.
- **Multi-core**: the block matvec/SVD parallelism that the DMRG path uses can be
  enabled for TDVP2 as well.

## 7. Conclusion

QuantX now has validated, machine-precision TDVP time evolution (real and
imaginary time, one- and two-site) exposed in Python, with a **Sz-symmetric
block-sparse TDVP2** that is **8–50× faster than TeNPy across all tested system
sizes** at identical accuracy. This makes QuantX a full tensor-network engine —
fast ground states (DMRG, U(1)/SU(2)) *and* fast dynamics (TDVP) — and the TDVP
speed advantage is a genuine killer feature, secured by the same block-sparsity
lever that powers the DMRG path.
