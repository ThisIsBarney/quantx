# Finite-temperature spectroscopy

QuantX computes the thermal state of a chain and its **dynamical response** by
ancilla purification and real-time block TEBD on the block-sparse engine, the same
engine that runs the T=0 dynamics. From Python, `quantx_engine.thermal` covers the
XXZ, spin-1 Heisenberg and Fermi-Hubbard chains of `quantx_engine.models`:

| Python | what it gives | C++ routine |
|---|---|---|
| `thermal.statics(model, beta)` | E(β), ⟨S^z_j⟩ (⟨n_j⟩ for Hubbard), ⟨O_c O_j⟩ and the uniform susceptibility (the compressibility for Hubbard), on a grid from β = 0 | `run_thermal_blocks` |
| `thermal.structure_factor(...)`, then `.spectrum()` | C_j(t) = ⟨S^z_j(t) S^z_c⟩_β, then the dynamical structure factor **S(k, ω, T)** | `run_spectral_thermal_blocks` |
| `thermal.current_correlation(...)`, then `.conductivity()` | ⟨J(t) J⟩_β, then the optical conductivity **Re σ(ω, T)** per site | `run_conductivity_thermal_blocks` |

```python
from quantx_engine import thermal
from quantx_engine.models import HubbardChain, XXZChain

st = thermal.statics(XXZChain(L=16), beta=2.0)       # seconds
st.energy[-1], st.susceptibility[-1]                 # at beta = 2

sf = thermal.structure_factor(XXZChain(L=12), beta=1.0, t_max=4.0)
k, w, S = sf.spectrum()                              # S(k, omega, T)

cc = thermal.current_correlation(HubbardChain(L=6, U=4.0, mu=2.0), beta=1.0,
                                 t_max=3.0, max_bond=256)
w, sigma = cc.conductivity()                         # Re sigma(omega, T), omega >= 0
```

**The bond dimension.** `max_bond` (default 128) caps the purified states, and the
real-time evolution reaches it within a few steps, so raise it until the result
stops moving. The spin chains need far less than Hubbard: the XXZ structure factor
above moves by 1e-5 between `max_bond` 128 and 256, while the Hubbard conductivity
moves by 1% of its peak (and the late-time ⟨J(t)J⟩ by 5%), at the local dimension
d = 4. `max_bond` and `bond_dims` in the results say how full the bonds got.

**Conventions.** Those of `quantx_engine.models`: spins S = σ/2, open chains. The
ensemble is grand canonical, over every S^z sector and, for Hubbard, every particle
number; `mu` sets the filling (`mu = U/2` is half filling at every temperature). The
current is that of the conserved charge, from the continuity equation: for the spin
chains J = Σ_l (iJ/2)(S⁺_l S⁻_{l+1} − h.c.), and for Hubbard the particle current
J = Σ_{l,s} i t (c†_{l+1,s} c_{l,s} − h.c.). The structure factor of the Hubbard
chain is the spin one, S^z = (n↑ − n↓)/2.

**The spectra.** `spectrum()` is S(k, ω) = Σ_r e^{−ikr} ∫dt e^{iωt} C_{c+r}(t),
with the negative times from C(−t) = C(t)*, and a Gaussian window e^{−(ηt)²/2}
(default η = 3/t_max) against the cut at the last time. On an open chain the momentum
is measured from the central source, and the extension C(−t) = C(t)* is exact for the
local correlation (H is real) and at every k on an odd chain, whose centre is a
mirror point; on an even chain it is the usual open-chain approximation. At β > 0
both signs of ω carry weight, related by detailed balance, S(k, −ω) = e^{−βω} S(k, ω),
up to the window's broadening.

`conductivity()` is the Kubo formula,
Re σ(ω) = (1 − e^{−βω})/(2Lω) ∫dt e^{iωt} ⟨J(t) J⟩_β, which tends to
β/(2L) ∫dt ⟨J(t) J⟩_β at ω → 0. It is computed on ω ≥ 0 only: Re σ is even, and below
zero the factor grows as e^{β|ω|}, against a windowed transform that is not
exponentially small there.

J1-J2 and the Hubbard lattices have no purified form in the engine yet, and are
refused.

## How it works

The Gibbs state `ρ(β) = e^{−βH}/Z` is represented as a **pure state**
`|ψ(β)⟩ = e^{−βH/2}|ψ0⟩` on an **interleaved 2N chain** — physical sites at even
positions, ancillas at odd. `|ψ0⟩` is a product of maximally entangled pairs, one per
site, each of fixed total charge (a singlet for spin 1/2; for Hubbard every pair holds
two electrons at S^z = 0), which is the β=0 infinite-temperature state. Cooling is
imaginary-time TEBD with **swap-and-carry** gates that ferry physical neighbours
together, so the engine's block structure stays intact: local dimension d = 2, 3 and
4 for spin 1/2, spin 1 and Hubbard (see
[finite_temperature_plan.md](finite_temperature_plan.md) for the design pivot from
the originally-planned `d=4` composite site).

Dynamical correlators `C(t;T) = ⟨A(t) B⟩_β` cool to `|ψ(β)⟩`, apply the source
(`σ^z_{l0}` for `S(k,ω,T)`, the total current `J` for `σ(ω,T)`), and real-time
evolve **both** the unsourced and sourced states — the finite-T bra is not
stationary, unlike T=0. The **Karrasch–Bardarson–Moore backward-ancilla
disentangler** cancels the real-time entanglement growth of the unsourced state (a
true unitary gauge, exact to ~1e-13; a compute win — see
[disentangler figure](disentangler_bond_growth.svg)). The current correlator
⟨A(t)|J|B(t)⟩ is contracted exactly at every step, as an MPO sandwich.

## Validation

**The Python gate** (`python/quantx_engine/tests/test_thermal_ed.py`) compares
everything `quantx_engine.thermal` returns with an exact diagonalisation of the whole
Hilbert space, written independently of the engine: the Gibbs state over every sector,
its exact real-time evolution, and the operators in the conventions above (for
Hubbard, Jordan-Wigner fermions). It runs the XXZ chain (L = 6, Δ = 0.7), the spin-1
chain (L = 4) and the Hubbard chain (L = 4, U = 3, μ = 0.9, off half filling): the
statics on the whole grid up to β = 2, and both correlators at β = 1 up to t = 2. The
bond dimension holds each purified state whole, so the only error is the
second-order Trotter error of the steps: below 1e-3 at dβ = dt = 0.05 (measured
4.2e-4 at most, on the Hubbard current, whose C(0) is 3.8), and falling by 3.9 to
4.0 at half the step (the gate requires at least 3). A slip in a unit conversion would be
0.1 or more, and would not move with the step. The gate also checks the Fourier
transforms, `spectrum()` and `conductivity()`, on exact correlations against the
Lehmann sums they approximate, to 1e-8, without the engine.

That gate found a bug in the C++ current correlator, fixed with it: it used to apply J
to the evolving state and compress the product back to the bond cap at every step,
with a first sweep that truncated against a part of the chain that was not
orthonormal. It lost 3e-2 of ⟨J(t)J⟩ on the 4-site Hubbard chain at the exact bond
dimension, at every step size.

The C++ tests check every routine against dense ED at small N (the exact thermal
trace / Lehmann representation), on three SVD backends (Eigen-CPU, cuSOLVER-GPU,
LAPACK):

| test | reference | worst error |
|---|---|---|
| `qx_thermal_blocks_test` | `Tr[O e^{−βH}]/Z`, N=8 | β=0 exact; finite-β ~1e-3 (Trotter) |
| `qx_thermal_spectral_test` | ED Lehmann `S(k,ω,T)`, N=6 | ~6e-3 / 8e-3 (β=0.5 / 4) |
| `qx_thermal_transport_test` | ED Lehmann `σ(ω,T)`, N=6 | ~4e-3 / 1.2e-2 (β=0.5 / 2) |

The disentangler-invariance check (`|C_on − C_off| ~ 1e-13`) is the proof that the
ancilla-backward gauge is exact, not an approximation.

## Spectra

The figures below are space+time FFTs of the emitted correlators (Heisenberg
chain, N=16, χ=512 — demonstration scale; see the caveat). They show the
**temperature dependence** of the response directly.

**Dynamical structure factor `S(k,ω,T)`.** The dashed lines are the des
Cloizeaux–Pearson two-spinon continuum edges `ω = (π/2)|sin k|` and
`ω = π|sin(k/2)|`. Cold (`T = 0.5`) shows a **sharp, bright antiferromagnetic
spinon peak at `k = π`, low ω** — coherent spectral weight pinned to the lower
continuum edge (the brightest cells sit at `k ≈ π, ω ≈ 0.9`, inside the
continuum). Warm (`T = 2`) is **broad and washed out** — thermal incoherence
spreads the weight across the band. That sharp→broad evolution on warming is the
finite-T spectroscopy.

![S(k,omega) cold](skw_b2.0.svg)
![S(k,omega) warm](skw_b0.5.svg)

**Current spectral function `∝ σ(ω,T)`** (FFT of `⟨J(t)J(0)⟩_β`; the raw spectral
function, not Kubo-reweighted). The response peaks near the exchange scale
`ω ≈ J`; the total weight (f-sum `∝ ⟨J²⟩_T`) shrinks on cooling.

![current spectral function](conductivity_sigma_wT.svg)

Generated by `qx_thermal_skw_emit` / `qx_thermal_transport_bench` →
`benchmarks/plot_skw.py` / `benchmarks/plot_conductivity.py` (pure-stdlib SVG, no
numpy). **Caveat:** these are small-N / modest-χ *demonstrations* — a converged
S(k,ω,T) needs larger N (so finite-size revivals fall outside the time window)
and larger χ (so the long-time dynamics is truncation-free). That is exactly the
regime where the GPU large-χ engine pays off; the correlators themselves are
ED-validated (above).

## GPU acceleration — the advantage grows with χ, now at finite T

The block-sparse SVD hot loop fires unchanged on the purified chain, and the
real-time response **saturates the bond dimension to χ_max** within a few steps —
so finite-T dynamics lands directly in the large-χ regime where cuSOLVER beats the
CPU. Timing the purified real-time evolution (Heisenberg N=20, 40 purified sites,
RTX 4090 vs a single i5-13600KF thread, `QX_GPU_MIN_BLK` toggle, identical
`E(β) = −32.618` on both paths):

| χ | GPU | CPU | speedup |
|---|-----|-----|---------|
| 512 | 46.5 s | 67.6 s | **1.46×** |
| 1024 | 90.6 s | 258.7 s | **2.86×** |

![finite-T GPU margin](gpu_finiteT_margin.svg)

The margin **grows with χ** (1.46× → 2.86×), the same property as the T=0 dynamics
([gpu_margin_vs_chi.svg](gpu_margin_vs_chi.svg)) — and here the win at χ=1024 is
*larger* than the T=0 Néel-quench (~2.2×), because the finite-T correlator evolves
two states (a non-stationary bra and the sourced ket), doubling the SVD load. This
is the "bigger problem → bigger advantage" property exactly where finite-T needs
it: low T and long times both push χ up. Benchmark: `qx_thermal_gpu_bench`
(`QX_GPU_MIN_BLK=128` GPU vs `=9999999` CPU) → `benchmarks/plot_gpu_finiteT.py`.
