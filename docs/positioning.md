# QuantX — Positioning

## The one-liner

**QuantX is an integrated, GPU-accelerated engine for the *dynamical response functions*
of quantum spin chains — and its advantage over the reference code (TeNPy) grows with
bond dimension χ instead of reversing.**

The differentiator is not a single speedup multiple (those are hardware- and
thermal-dependent). It is a **capability + a scaling property**: a complete,
ED-validated pipeline for the observables experimentalists actually want, whose
relative advantage *increases* in the hard (large-χ) regime where it matters.

## Why it's a killer feature — three pillars

**1. A complete dynamical-observables pipeline, not just a fast kernel.**
One block-sparse (U(1) / SU(2)-symmetric) C++/CUDA engine carries the whole computation
end-to-end — DMRG ground state → operator application → real-time block-TEBD evolution
→ space–time FFT — and emits the experimentally relevant observables directly:

- dynamical structure factor **S(k, ω)**
- spin **conductivity σ(ω)** (current–current correlator)
- **Loschmidt echo / DQPT** rate function

All are XXZ-general (anisotropy Δ) and validated against exact diagonalization. TeNPy
can produce these, but the user assembles them in Python and the orchestration is
Python-bound; the packaged, native pipeline is QuantX's distinguishing surface.

**2. The advantage *grows with χ* (the regime that matters).**
On CPU alone, QuantX's lead over TeNPy *reverses* at large χ, where TeNPy's tuned BLAS
catches up on the O(χ³) SVD. Routing that SVD (and the θ-tensor build) to the GPU turns
the downward curve **upward**: against single-thread TeNPy (Intel MKL), the Néel-quench
TEBD margin climbs with χ to **~2.2× at χ=1024** (≈2.7× with the opt-in FP32 path),
consistently for N=32 and N=64. See the figure below. This "bigger problem → bigger
advantage" property is exactly what hard, convergence-limited simulations need.

![QuantX-GPU vs TeNPy-CPU margin vs χ](gpu_margin_vs_chi.svg)

**3. A strong ground-state floor.**
The DMRG ground state is a **4.8–6× win over TeNPy** on CPU (the hot loop is many small
Lanczos matvecs, where compiled C++ beats Python orchestration) — independent of the
GPU work, it anchors the package.

## What "complete" means

The killer feature is the *system*, not a component: a single engine where the
ground state, the operator algebra, the time evolution, and the spectral post-processing
are all block-sparse, GPU-aware, and consistent — so a user goes from a Hamiltonian to
S(k, ω) / σ(ω) / DQPT in one fast, validated path, and can push χ on a single consumer
GPU (RTX 4090) without the CPU-era large-χ reversal.

## Honest boundaries (deliberately not in the headline)

- The large-χ single-step TEBD speedup vs TeNPy is **~2.2×** (modest, and ±~2×
  thermally noisy on a consumer box without clock pinning). The robust claim is the
  *trend*, not a precise multiple.
- **Mid-χ (≈256) is roughly break-even.**
- The large small-χ multiples (~4.7×) are a **C++-vs-Python overhead** advantage, not an
  algorithmic one — labeled as such.
- FP32 is **opt-in** (real-time dynamics only); DMRG / energy-exact paths stay FP64.

## Suggested phrasings by audience

- **Paper / abstract:** "A block-sparse C++/CUDA engine for the dynamical response of
  XXZ spin chains — S(k, ω), σ(ω), DQPT — whose advantage over TeNPy *grows with bond
  dimension* (≈2–3× at χ=1024 on a single RTX 4090), atop a 4.8–6× DMRG ground-state
  speedup, all ED-validated."
- **README one-liner:** "Dynamical response functions for spin chains, GPU-accelerated —
  and it gets *relatively* faster the harder (larger χ) you push."
- **Pitch:** lead with "complete pipeline + grows-with-χ", show the margin-vs-χ figure.
