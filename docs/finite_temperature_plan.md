# Finite-Temperature Dynamical Response — Implementation Plan (KF2)

**Status:** PLAN (2026-06-10). Greenfield, but built **on top of** the existing,
ED-validated block-sparse TEBD machinery — not a rewrite.

**Goal:** lift QuantX's killer feature from *T=0* dynamical response to **finite
temperature** — compute `S(k, ω, T)`, the dynamical susceptibility `χ''(k, ω, T)`,
and finite-T transport `σ(ω, T)` on the same GPU-accelerated block-sparse engine.
These are the quantities that **neutron scattering, NMR, and transport experiments
actually measure**, so this turns the engine from "T=0 spectroscopy" into
"experimentally comparable spectroscopy."

**Why this is the right next killer feature — it compounds the GPU moat.**
Finite-T via **ancilla purification** doubles the local Hilbert space and, more
importantly, *cooling to low T builds up thermal entanglement before t=0*, so the
bond dimension χ needed is substantially larger than at T=0 for the same accuracy.
Because QuantX's measured GPU advantage **grows with χ** (see
[`gpu_acceleration_plan.md`](gpu_acceleration_plan.md)), the finite-T regime makes
that advantage *bigger*, not just preserved. The headline becomes a two-axis claim:
the margin grows with χ **and** with inverse temperature β.

---

## 1. Context: what already exists to reuse

All in `src/dmrg_blocks.cpp` / `src/dmrg_internal.hpp`, all validated vs ED:

- **Imaginary-time block-TEBD** — `run_tebd2_blocks(..., bool imag, ...)` and
  `run_tdvp2_blocks(..., bool imag)`. This is the cooling primitive: `e^{-τH}`
  Trotter gates with block-sparse SVD truncation. Purification cools the **doubled**
  chain with exactly this.
- **T=0 dynamical correlators via TEBD** — the templates we generalize:
  - `run_spectral_blocks` → `C_{j,l0}(t) = ⟨ψ0|σ^z_j e^{-i(H-E0)t} σ^z_{l0}|ψ0⟩`,
    2D-FFT'd to `S(k,ω)`.
  - `run_conductivity_blocks` → current-current `C(t) = ⟨ψ0|J e^{-i(H-E0)t} J|ψ0⟩`
    → `σ(ω)`. `build_current_mpo` already exists.
  - `run_loschmidt_blocks` → DQPT rate function.
- **MPO builders** — `build_xxz_mpo`, `build_current_mpo`, and a **general
  `build_pauli_sum_mpo(n, terms)`** (arbitrary Pauli-string sums) — useful for the
  ancilla-extended Hamiltonian and for staggered/uniform field probes.
- **GPU SVD + theta residency** — `thin_svd_gram_gpu` (FP64/opt-in FP32) and
  `form_theta_block_z`, dispatched above `QX_GPU_MIN_BLK` / `QX_GPU_THETA_MINWORK`.
  Finite-T inherits all of this unchanged — it is the same per-bond SVD hot loop.

**What does NOT exist yet:** any purified / thermofield / METTS state, any
finite-T observable, any `β` parameter. That is what this plan adds.

---

## 2. The method: ancilla purification (thermofield)

Represent the thermal density matrix `ρ(β) = e^{-βH}/Z` as a pure state in an
enlarged Hilbert space (physical ⊗ ancilla):

1. **Infinite-T state** `|ψ(β=0)⟩`: a product of locally maximally-entangled
   physical–ancilla pairs, `⊗_i (|↑⟩_i|↓⟩_ā + |↓⟩_i|↑⟩_ā)/√2`. Tracing out the
   ancilla gives `ρ ∝ I` (infinite temperature). **Pairing the ancilla spin with
   *opposite* Sz keeps the whole purified state in the total-Sz=0 block** — i.e. it
   stays inside the existing U(1) block-sparse structure with no new symmetry code.
2. **Cool:** imaginary-time evolve `|ψ(β)⟩ = e^{-βH_phys/2}|ψ(0)⟩` with H acting on
   the **physical legs only** (identity on ancillae). Reuses the imag-time TEBD path.
   Then `⟨O⟩_β = ⟨ψ(β)|O_phys|ψ(β)⟩` for any physical operator O.
3. **Dynamics:** apply a source operator `O_{l0}`, real-time evolve, measure
   `C_{j,l0}(t; T) = ⟨ψ(β)| O_j e^{-i(H-⟨H⟩)t} O_{l0} |ψ(β)⟩`, 2D-FFT → `S(k,ω,T)`.

**Lattice-doubling design decision (the first thing to settle):**
- **(A) Composite d=4 site (recommended):** each MPS site carries physical⊗ancilla
  (two spin-½ → local dim 4), Sz quantum number = `sz_phys + sz_anc`. The XXZ TEBD
  gate acts as `exp(τ h_phys) ⊗ I_anc` on neighboring composite sites — i.e. the
  *nearest-neighbor* gate structure is preserved, which is exactly what the block
  TEBD engine wants. Cost: introduce a d=4 local site type + the doubled MPO/gate.
- **(B) Interleaved 2N chain:** physical and ancilla sites alternate on a length-2N
  chain (odd bonds = the β=0 Bell pairs, even bonds = H). Keeps d=2 but makes the
  physical coupling *next-nearest-neighbor*, needing swap gates or 3-site handling —
  more friction with the NN TEBD path.

Recommend **(A)**: it preserves the engine's NN-gate + U(1)-block invariants and
localizes all new code to "a doubled local site + its gate," leaving the SVD /
truncation / GPU hot loop untouched.

**Entanglement-reduction trick (Phase F1, the make-or-break):** the naive purified
real-time correlator's entanglement grows fast (you evolve a state that already
carries thermal entanglement). Use the **Karrasch–Bardarson–Moore** disentangler:
evolve the ancilla *backward* in time (auxiliary `-H_anc`), which is physically
irrelevant for `⟨O_phys(t)O_phys(0)⟩` but roughly **halves** the bond growth → reach
longer t (better ω-resolution) / lower T at fixed χ. This is the standard
finite-T-tNRG/purification optimization and is essential for useful spectra.

---

## 3. Design — phases

### Phase F0 — purified state + thermal **static** observables
Deliver `e^{-βH/2}` cooling on the doubled chain and static thermodynamics.
- Composite d=4 site type + β=0 maximally-entangled init in the Sz=0 block.
- Doubled XXZ gate `exp(τ h_phys)⊗I_anc`; reuse imag-time TEBD + block SVD.
- Outputs: `E(T)`, specific heat `C(T)=dE/dT`, uniform susceptibility `χ(T)`,
  static structure factor `⟨S^z_j S^z_l⟩_T`.
- **Validation:** vs ED finite-T (small N, exact `Tr[O e^{-βH}]/Z`); vs the known
  XXZ thermodynamics (Bethe-ansatz / QTM) for `E(T)`, `χ(T)` at larger N.
- New API: `run_thermal_blocks(mpo, n, beta, dtau, chi_max, cutoff, ...)`
  → `ThermalResult { temps/betas, E, C, chi, szsz... }`.

### Phase F1 — finite-T **dynamical** correlators → `S(k,ω,T)`, `χ''(ω,T)`
- After cooling to β/2, apply source `σ^z_{l0}`, real-time evolve, measure
  `C_{j,l0}(t;T)`; 2D-FFT → `S(k,ω,T)`. `χ''(k,ω,T) = (1-e^{-βω}) S(k,ω,T)`.
- Implement the **backward-ancilla disentangler** for reach.
- Optional: linear-prediction / windowing to extend the time series for ω-resolution.
- **Validation:** vs ED finite-T Lehmann (small N); **detailed balance**
  `S(k,-ω) = e^{-βω} S(k,ω)` (internal check, any N); f-sum rule; T→0 limit must
  reproduce the existing `run_spectral_blocks` result.
- New API: `run_spectral_thermal_blocks(...)` mirroring `run_spectral_blocks` + β.

### Phase F2 — finite-T **transport** + experimental observables
- Current-current correlator on the purified state → `σ(ω,T)`, Drude weight `D(T)`
  (the live XXZ ballistic-vs-diffusive transport question), and NMR `1/T1(T)` from
  the q-summed low-ω limit. Reuses `run_conductivity_blocks` + `build_current_mpo`.
- **Validation:** ED finite-T transport (small N); D(T)→ integrable-XXZ known
  behavior; sum rules.

### GPU / benchmark phase — the headline (the deliverable)
- Confirm the SVD/theta GPU paths fire unchanged on the doubled chain (they will —
  same per-Sz-block SVD; χ is just larger).
- **Headline benchmark:** QuantX-GPU vs TeNPy-CPU purification `S(k,ω,T)` for a
  Heisenberg chain, sweeping **both** χ and β. Expected: margin grows with χ (as at
  T=0) **and** the useful-χ for a fixed accuracy grows as T↓ → QuantX pulls further
  ahead in the experimentally relevant low-T regime. Extend
  `benchmarks/gpu_vs_tenpy_quench.py` style harness (min-of-N, cooldowns).

---

## 4. Where to change (file/function map)

- `src/dmrg_internal.hpp` — new result structs (`ThermalResult`,
  `SpectralThermalResult`) + declarations.
- `src/dmrg_mpo.cpp` — composite d=4 site type; doubled XXZ gate
  `exp(τ h_phys)⊗I_anc`; β=0 maximally-entangled init builder.
- `src/dmrg_blocks.cpp` — `run_thermal_blocks`, `run_spectral_thermal_blocks`,
  `run_conductivity_thermal_blocks`. These are **purified-state wrappers** around the
  existing imag/real TEBD step + `block_svd_truncate` (GPU-dispatched as today).
- New benchmark `benchmarks/finite_T_vs_tenpy.py`; new validation
  `src/thermal_blocks_test.cpp` (ED finite-T + detailed balance) added to ctest.
- No changes to the GPU kernels (`gpu_svd.cu`) — finite-T is pure reuse of the SVD
  hot loop at larger χ.

---

## 5. Validation plan (tiered, like the GPU work)

1. **ED finite-T (small N, exact):** `Tr[O e^{-βH}]/Z` and the Lehmann
   `S(k,ω,T)` for N≤12 — the purified result must match to TEBD-truncation tolerance.
2. **Internal physics invariants (any N):** detailed balance
   `S(k,-ω)=e^{-βω}S(k,ω)`, f-sum rule, `χ(T)` Kramers–Kronig consistency.
3. **Known thermodynamics:** XXZ `E(T)`, `χ(T)` vs Bethe-ansatz/QTM references.
4. **T→0 regression:** finite-T code at large β must reproduce the existing
   T=0 `run_spectral_blocks` / `run_conductivity_blocks` outputs.
5. **Literature spectra:** Heisenberg-chain finite-T `S(k,ω,T)` vs published
   purification/TEBD results (qualitative line-shape + sum-rule weight).

---

## 6. Pitfalls & mitigations

- **Entanglement growth is worse than at T=0** (the t=0 state is already thermally
  entangled) → backward-ancilla disentangler (F1) + larger χ (where GPU wins).
  This is *the* risk; F1 stands or falls on the disentangler working.
- **Low T needs large χ for the *state itself*** (imag-time cooling). Good news for
  the GPU story; budget χ accordingly and report the χ(β) needed.
- **β=0 state normalization / Sz bookkeeping** in the block basis — unit-test the
  infinite-T `⟨O⟩` equals the trace average before any cooling.
- **Finite-time ω-resolution** → window + optional linear prediction; report the
  effective ω-resolution honestly (don't over-resolve).
- **d=4 site doubles block sizes** → re-tune `QX_GPU_MIN_BLK` / `QX_GPU_THETA_MINWORK`
  for the doubled local dimension (more work per block → GPU fires sooner).

---

## 7. Recommended order of work

1. **Phase F0** — composite-site purification + cooling + static thermodynamics,
   validated vs ED and XXZ-thermo. (Establishes the purified state is correct.)
2. **Phase F1** — finite-T `S(k,ω,T)` with the disentangler; detailed-balance +
   ED-Lehmann validation. (The scientific payoff.)
3. **GPU headline benchmark** — margin vs (χ, β) vs TeNPy-CPU; write into
   `docs/gpu_acceleration_report.md` / a new finite-T report; update `positioning.md`.
4. **Phase F2** — finite-T transport `σ(ω,T)`, Drude weight, NMR `1/T1`.

Each phase → branch → PR → Codex review → admin-merge with local ED validation
(same workflow as the GPU phases; CI remains billing-blocked).

**Positioning upgrade this unlocks:** from "GPU dynamical response of spin chains"
to **"GPU finite-temperature spectroscopy of correlated spin chains — `S(k,ω,T)`,
`χ''`, `σ(ω,T)` — with an advantage that grows with both χ and β."**
