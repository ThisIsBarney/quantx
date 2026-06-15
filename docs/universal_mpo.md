# Universal MPO — any Sz-conserving model on the fast engine

The block-sparse, GPU-accelerated engine was originally hard-wired to the
nearest-neighbour Heisenberg/XXZ bond structure (`HB_DELTA={0,+2,−2,0,0}` baked
into every environment / `H_eff` kernel). The **universal MPO** work makes the
bond structure *data-driven*: the per-bond U(1)/Sz charge is **derived from the
operator MPO itself**, so the same fast engine — DMRG, real-time dynamics,
`S(k,ω)`/`σ(ω)`, the whole finite-temperature stack, the grows-with-χ GPU
acceleration — runs **any Sz-conserving Hamiltonian**, not just XXZ.

## How

`derive_mpo_sym(mpo)` recovers each MPO bond index's Sz charge from the operator
structure alone, propagating `delta[b] = delta[a] + sz(s) − sz(s')` over every
nonzero `W(a,b)(s,s')` and anchoring the vacuum bond at 0; it throws if the MPO
isn't Sz-conserving. The env/`H_eff` kernels then consume that `delta` array in
place of the hard-coded constant. For Heisenberg the derived charges *equal*
`HB_DELTA`, so the rewiring is bit-identical there (the n=32 regression energy is
unchanged to the digit); for a wider MPO it just works.

## Showcase: the frustrated J1–J2 chain

`H = J1 Σ S_i·S_{i+1} + J2 Σ S_i·S_{i+2}` is the textbook frustrated chain — an
**8-wide MPO** (`build_j1j2_mpo`), wider than Heisenberg's 5. It now runs on the
fast block path, matching exact diagonalization to **~1e-13** (`qx_mpo_sym_test`,
N=8 and N=10).

![J1-J2 dimerization](j1j2_dimerization.svg)

The figure is a pure-energy signature of the physics, computed entirely on the
new path (`qx_j1j2_showcase`, N=32). As `J2/J1 → 0.5` (the **Majumdar–Ghosh
point**) the ground state dimerizes toward a product of nearest-neighbour
singlets — a **bond-dimension-2** state — so its entanglement collapses and a
χ=8 DMRG nearly reaches the converged energy: the χ=8 gap drops **~1300×**
(2.5e-4 → 1.9e-7) approaching MG. Exactly at J2=J1/2, χ=8 returns the **exact
Majumdar–Ghosh dimer energy** E/N = −1.500000.

## Showcase: the spin-1 Haldane chain (arbitrary local dimension)

Phase 1 generalized the *bond* structure; **phase 2 generalizes the *local*
dimension**. The same engine now runs `H = J Σ S_i·S_{i+1}` with the **physical
spin-1 operators** (local dimension d=3, `build_spin1_heisenberg_mpo`) — the
gapped **Haldane phase**, a different universality class from the critical
spin-1/2 chain. It matches exact diagonalization to **~1e-13** (`qx_mpo_sym_test`,
N=4 and N=6, on both the dense and the fast block path).

![Spin-1 Haldane energy density](spin1_haldane.svg)

`qx_spin1_showcase` runs the chain on the fast block path at growing N. The
**incremental bulk energy** `e_bulk(N) = (E_N − E_{N−ΔN})/ΔN` removes the
open-chain edge offset and converges to the known spin-1 Heisenberg energy
density **−1.401484039 J to ~2e-8** by N=96. And because the Haldane ground state
is gapped (area-law entanglement), the energy **saturates by χ≈16** — unlike the
critical spin-1/2 chain whose entanglement grows with N. Both signatures come
from ground-state energies alone, on the same fast path as spin-1/2.

## Scope and limits

- **Phase 1 (done):** any *uniform-bulk* Sz-conserving MPO at local dimension
  d=2 — next-nearest / longer-range, fields, etc. Non-uniform MPOs
  (`build_pauli_sum_mpo`) stay on the dense path; `derive_mpo_sym` throws
  fail-loud if one reaches the block path.
- **Phase 2 (done — ground state):** arbitrary local dimension d>2. The MPO
  representation (`MPOSite.d`/`phys_charge`), the dense DMRG, and the **fast
  block-sparse DMRG** all run any uniform d (spin-1 d=3 validated vs ED). The
  block path folds the local index into d charge-shifted column groups
  (`phys_groups`) in place of the old up/dn split.
- **Known issue:** the single-site DMRG refinement phase loses variationality at
  an *exactly degenerate* ground state with truncation (e.g. the MG point itself
  at large N); the two-site path is unaffected (use `QX_DMRG_NO_SINGLESITE=1`).
  Tracked separately.
- **Phase 2 (remaining):** the block-sparse *dynamics* (TEBD/TDVP) and the GPU
  `form_theta` are still d=2 (the d>2 entry points throw fail-loud); generalizing
  them brings spin-1 real-time/finite-T spectroscopy onto the fast path.
