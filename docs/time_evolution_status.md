# Time evolution & derived features vs TeNPy — honest status (2026-06-03)

This note records a course correction. An earlier benchmark claimed block-sparse
**TDVP2** beat TeNPy "8–50× at all sizes." That comparison was against TeNPy's
`TwoSiteTDVPEngine` (Python TDVP — slow). Against the method practitioners
actually use for nearest-neighbour real-time evolution, **TEBD**, our TDVP2 loses
15–22×. The cause is algorithmic: TDVP does a Krylov matrix-exponential per bond
(~15–20 matvecs) vs TEBD's single Trotter gate; a C++/symmetry implementation edge
cannot close a 15–20× algorithmic gap. Everything built on TDVP2 (finite-T,
spectral) inherited the loss.

## What genuinely beats TeNPy (apples-to-apples, single thread, identical results)

| Task | QuantX | TeNPy | speedup | notes |
|------|--------|-------|---------|-------|
| **DMRG ground state** N=32, χ=128 | 0.82 s | 4.89 s | **6.0×** | same energy, dE~1e-13 |
| **DMRG ground state** N=48, χ=128 | 1.71 s | 8.21 s | **4.8×** | vs `TwoSiteDMRGEngine` |

DMRG wins because it is the *same algorithm*: C++ + Sz/SU(2) block-sparsity beats
TeNPy's Python at equal work.

## The fix: block-sparse C++ TEBD (the algorithm TeNPy uses)

Implemented `tebd2_step_blocks` / `run_tebd2_blocks` (`src/dmrg_blocks.cpp`):
a precomputed two-site Sz-block gate + SVD per bond, symmetric L→R/R→L half-gate
sweep (2nd-order Trotter), no environments, no Krylov. Reuses the existing
`form_theta` / `block_svd_truncate` / center helpers. Exposed via
`qx_tdvp_evolve` (`engine="tebd"`, default) and `TDVPSolver.evolve(engine=...)`.

Validated vs exact diagonalization (n=6 Heisenberg): Néel-quench staggered
magnetization 2.3e-4, energy conserved 5.6e-5 (O(dt²) Trotter), imaginary-time →
ground 1.3e-6.

### Result: 10× over our own TDVP2

| Néel quench, χ=128, 60 steps | QX-TDVP2 | QX-TEBD | speedup |
|---|---|---|---|
| N=20 | 48.4 s | 4.42 s | 11× |
| N=32 | 148.5 s | 12.8 s | 12× |
| N=48 | 275.8 s | 24.7 s | 11× |

### Evolution-only vs TeNPy TEBD (the real opponent) — WINS AT ALL SIZES

The block SVD was 94% of the TEBD step (largest blocks ~77 wide), where Eigen's
`BDCSVD` is slower than TeNPy's LAPACK `gesdd`. Replacing it with the
Gram-matrix eigendecomposition (`thin_svd_gram`: SVD via the Hermitian
eigenproblem of the smaller `M M†`/`M† M` — the standard fast-TEBD route) roughly
halved the step:

| Néel quench, χ=128, 60 steps | QX-TEBD (BDCSVD) | QX-TEBD (Gram) | TeNPy-TEBD | QX/TeNPy |
|---|---|---|---|---|
| N=20 | 3.21 s | **1.16 s** | 2.74 s | **2.37×** |
| N=32 | 12.4 s | **5.28 s** | 6.25 s | **1.18×** |
| N=48 | 23.1 s | **10.0 s** | 10.85 s | **1.08×** |
| N=64 | — | **14.1 s** | 15.1 s | **1.07×** |

Block TEBD now beats TeNPy TEBD at every size. Accuracy preserved (TDVP2 energy
still 3.8e-13). The shared `block_svd_truncate` change also speeds up TDVP2.

## Dynamical structure factor S(k,ω): a win at all sizes

Built on the winning pieces (DMRG ground state → σ^z → block TEBD evolve →
block two-MPS overlap → FFT). `C_{j,l0}(t)` validated vs ED to ~6e-3 (Trotter).
End-to-end vs TeNPy (DMRG + evolve + per-step measurement), single thread:

| C_{j,l0}(t), 150 steps | QuantX (TDVP2, original) | QuantX (TEBD+Gram) | TeNPy | speedup |
|---|---|---|---|---|
| N=16 | (lost) | 0.6 s | 9.8 s | **16.8×** |
| N=24 | (lost) | 5.3 s | 20.9 s | **3.9×** |
| N=32 | 237 s (0.1×) | 16.8 s | 37.8 s | **2.25×** |
| N=48 | (lost) | 34.1 s | 62.7 s | **1.84×** |

The dynamical structure factor is a genuine killer feature: from a 5–10× loss
(TDVP2) to a 1.8–17× win across all tested sizes.

## Finite-T (for the record): correct but not winning

METTS lost 15–55× (sampling overhead); ancilla purification — done with the
existing block engine, **no general-MPO machinery needed** (an earlier "needs
generalisation" claim was wrong: it conflated the MPO bond's accumulated Sz with
the per-site Sz change) — lost 2–10× (2N-site d=2 chain vs TeNPy's N-site d=4).
Both are validated vs exact but remain TeNPy's strong suit, so they are kept on a
separate experimental branch (`feat/finite-t`) rather than this winners branch.

## Bottom line

- **DMRG ground-state solver** — genuine, verified killer feature (4.8–6×, more with SU(2)).
- **Block TEBD** — the right time-evolution engine (10× over our TDVP2); with the
  Gram-matrix SVD it **beats TeNPy TEBD at all sizes** (1.07–2.37×).
- **Dynamical structure factor S(k,ω)** — built on TEBD, **beats TeNPy 1.84–16.8×**
  across N=16–48. A genuine killer feature.
- TDVP2 is retained for its Krylov-exact local step (energy conservation to ~1e-13;
  the route for general / long-range MPOs where TEBD's Trotter split does not apply).
- Finite-T (METTS, purification) remains TeNPy's strong suit — correct but slower.
