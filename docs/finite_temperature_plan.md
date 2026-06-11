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
   physical–ancilla pairs, `⊗_i (|↑⟩_i|↓⟩_ā − |↓⟩_i|↑⟩_ā)/√2` (the singlet; any
   relative phase works — it cancels in `Tr_anc`). Tracing out the ancilla gives
   `ρ ∝ I` (infinite temperature). **Pairing the ancilla spin with *opposite* Sz
   keeps the whole purified state in the total-Sz=0 block** — i.e. it stays
   inside the existing U(1) block-sparse structure with no new symmetry code.
2. **Cool:** imaginary-time evolve `|ψ(β)⟩ = e^{-βH_phys/2}|ψ(0)⟩` with H acting on
   the **physical legs only** (identity on ancillae). Reuses the imag-time TEBD path.
   Then `⟨O⟩_β = ⟨ψ(β)|O_phys|ψ(β)⟩` for any physical operator O.
3. **Dynamics:** apply a source operator `O_{l0}`, real-time evolve, measure
   `C_{j,l0}(t; T) = ⟨ψ(β)| O_j e^{-i(H-⟨H⟩)t} O_{l0} |ψ(β)⟩`, 2D-FFT → `S(k,ω,T)`.

**Lattice-doubling design decision (the first thing to settle):**
- **(A) Composite d=4 site:** each MPS site carries physical⊗ancilla
  (two spin-½ → local dim 4), Sz quantum number = `sz_phys + sz_anc`. The XXZ TEBD
  gate acts as `exp(τ h_phys) ⊗ I_anc` on neighboring composite sites — i.e. the
  *nearest-neighbor* gate structure is preserved. Cost: introduce a d=4 local site
  type + the doubled MPO/gate.
- **(B) Interleaved 2N chain:** physical and ancilla sites alternate on a length-2N
  chain. Keeps d=2 but makes the physical coupling *next-nearest-neighbor*,
  needing swap gates.

**DECIDED: (B) interleaved — implemented in Phase F0.** The plan originally
recommended (A), but reading the engine settled it the other way: **d=2 with ±1
charge shifts is the engine's pervasive structural contract**, not a local
parameter — `block_A[i][q_L] = Mat(dL, dR_up+dR_dn)` two-sector column layout,
theta's 2-group rows/cols, `bt_expand_theta` / `bt_A_from_U` / `bt_center_*`,
the overlap envs, and the GPU `form_theta` kernel all hard-code two spin
sectors. Option (A) would have rewritten every kernel (and re-validated the GPU
paths); option (B) turned out to need **zero engine changes**:
- **SWAP is an Sz-conserving 2-site gate** that fits the existing homogeneous
  `Gate` format exactly (`pdiag=1`, Sz=0 block `[[0,1],[1,0]]`), so
  `apply_gate_theta` applies it unmodified.
- **Swap-and-carry sweep:** L→R puts SWAP on even bonds (carrying each physical
  site past its ancilla) and the XXZ half-gate on odd bonds; R→L mirrors it,
  restoring the `p a p a …` layout each step. The sequence is a palindrome
  (2nd-order Trotter) and bond order is monotonic, so the canonical centre stays
  adjacent to the active bond — structurally identical to `tebd2_step_blocks`.
  Cost: one extra SVD per physical bond per half-sweep (the swap), a constant
  factor ≪ the d=4 rewrite risk.
- **Ancilla pass-through MPO** (`build_xxz_purified_mpo`): odd sites carry a
  diagonal identity on all 5 MPO bond states, preserving each state's
  `HB_DELTA`, so environments / `H_eff` / `bt_energy` run on the purified chain
  with no special cases.
- The β=0 singlet Bell pairs `⊗ (|↑↓⟩−|↓↑⟩)/√2` keep every pair at total Sz 0
  (bond QNs: 0 between pairs, ±1 inside), as required.

**Entanglement-reduction trick (Phase F1, the make-or-break):** the naive purified
real-time correlator's entanglement grows fast (you evolve a state that already
carries thermal entanglement). Use the **Karrasch–Bardarson–Moore** disentangler:
evolve the ancilla *backward* in time (auxiliary `-H_anc`), which is physically
irrelevant for `⟨O_phys(t)O_phys(0)⟩` but roughly **halves** the bond growth → reach
longer t (better ω-resolution) / lower T at fixed χ. This is the standard
finite-T-tNRG/purification optimization and is essential for useful spectra.

---

## 3. Design — phases

### Phase F0 — purified state + thermal **static** observables ✅ DONE
Deliver `e^{-βH/2}` cooling on the doubled chain and static thermodynamics.
- ~~Composite d=4 site~~ → **interleaved 2N chain** (see decision above):
  `purified_init` (β=0 singlet Bell pairs, right-canonical, centre on site 0) +
  `thermal_step_blocks` (swap-and-carry imag-time TEBD2) + `build_xxz_purified_mpo`.
- Outputs per checkpoint: `E(β)`, `⟨Z_j⟩`, `⟨Z_c Z_j⟩` (c = central physical
  site), `χ_zz = β·Var(Σ_j Z_j)/N` (all-pairs via overlap envs; `with_chi` flag —
  it costs N overlap sweeps per checkpoint). `C(T) = dE/dT` is a finite-difference
  post-process of `E(β)`.
- **Validation:** `src/thermal_blocks_test.cpp` (ctest `qx_thermal_blocks_test`)
  vs dense ED `Tr[O e^{-βH}]/Z`, N=8, Δ=1 (β→2) and Δ=0.5 (β→1): β=0 exact to
  machine precision; worst finite-β errors errE≈9e-4, errZcZ≈2e-4, errChi≈2e-4
  (Trotter-dominated at dβ=0.05); passes on Eigen-CPU, forced-GPU
  (`QX_GPU_MIN_BLK=1`), and forced-LAPACKE (`QX_LAPACKE_MIN=1`) backends.
- API: `run_thermal_blocks(n_phys, J, delta, dbeta, n_steps, chi_max, cutoff,
  measure_stride, with_chi)` → `ThermalResult { betas, E, chi_zz, sz, szsz, … }`.

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

- `src/dmrg_internal.hpp` — new result structs (`ThermalResult` ✅,
  `SpectralThermalResult`) + declarations.
- `src/dmrg_mpo.cpp` — `build_xxz_purified_mpo` ✅ (interleaved chain: physical
  XXZ W at even sites, identity pass-through at odd/ancilla sites).
- `src/dmrg_blocks.cpp` — `purified_init` + `thermal_step_blocks` +
  `run_thermal` ✅; F1/F2: `run_spectral_thermal_blocks`,
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
- **The 2N-site chain doubles the sweep length** (and the swap bonds add one SVD
  per physical bond per half-sweep) → a constant-factor overhead; the thermal χ
  growth with β is what actually sets the cost. If profiling ever shows the swap
  SVDs dominating, the swap bond's spectrum is a permutation of the unswapped
  one (no new truncation), so a cheaper exact-permutation path is possible.

---

## 7. Recommended order of work

1. **Phase F0** ✅ — interleaved-chain purification + swap-and-carry cooling +
   static thermodynamics, validated vs ED. (Establishes the purified state is
   correct.)
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
