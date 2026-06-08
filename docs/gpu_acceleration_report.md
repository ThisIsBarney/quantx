# GPU Acceleration Report — Phase A (cuSOLVER SVD) + Phase B (theta residency) + Phase C (FP32, opt-in)

**Date:** 2026-06-06 **Hardware:** NVIDIA RTX 4090 (Ada, sm_89), 24 GB; CUDA 13.2;
MSVC 19.50 x64; single-thread CPU baseline. **Plan:** [`gpu_acceleration_plan.md`](gpu_acceleration_plan.md).

## TL;DR

**Phase A** routes the large per-Sz-sector SVDs of the block TEBD step to the GPU
(cuSOLVER `Zheevd` on the Gram matrix). **Phase B** additionally builds the large
`theta` blocks on the GPU (`form_theta_block_z`), removing the theta-merge CPU
bottleneck. Both leave the control flow + small-block tail on the CPU. The central
claim of the plan — **the QuantX advantage grows with bond dimension χ** — holds,
and Phase B pulls further ahead:

| level | χ=256 | χ=512 | χ=1024 |
|---|---|---|---|
| **bare SVD kernel** (GPU/CPU) | 5.6× | 17.4× | **45×** (86× at d=2048) |
| **TEBD end-to-end, Phase A** (GPU SVD) | 1.01× | 1.57× | **2.60×** |
| **TEBD end-to-end, Phase B** (GPU SVD + theta) | 0.98× | 2.27× | **4.42×** |
| **QuantX-GPU+OpenBLAS vs TeNPy-MKL** (N=32, min-of-2, post-fix) | 0.91× | 1.12× | **2.17×** (2.74× w/ FP32) |

(First three rows: N=32 Néel quench, 60 steps, GPU-vs-CPU same-process 3-ways. Last
row: the real opponent — QuantX(CUDA+OpenBLAS) vs single-thread TeNPy(MKL), min-of-2
controlled, **post-LAPACKE-fix**; also ~4.7× at χ=128 where C++ overhead beats TeNPy's
Python, and **N=64 χ=1024 = 2.22× / 2.66× FP32** — consistent with N=32 at large χ.
Absolute ratios are thermally noisy (±~2×) on this box; the χ-trend is robust.) The
internal GPU-vs-CPU curves
**grow with χ** — the goal — and Phase B widens the lead (Phase-B/Phase-A =
1.0→1.44→1.70× across the three χ). Small-χ stays ~1× because the
blocks are below the GPU thresholds (no small-problem regression). The bare-kernel
speedup is monotonic in size; Phase A captures its SVD part (4.3× on the SVD phase
at χ=1024) and Phase B adds the theta part (7× on that phase), together turning the
χ=1024 step from 7.2 s (CPU) into **1.63 s**.

## What was built

- `src/gpu_svd.cu` / `src/gpu_svd.hpp` — `thin_svd_gram_gpu_z`: the Gram-trick thin
  SVD for complex-double blocks. Forms `G = M Mᴴ` (or `Mᴴ M`) with `cublasZherk`,
  eigendecomposes with `cusolverDnZheevd`, recovers the other factor with one
  `cublasZgemm`, and does only the O(k·max(m,n)) descending-reorder + `1/σ` scaling
  on the host. Persistent cuSOLVER/cuBLAS handles + reused device buffers, guarded
  by a mutex (one GPU, OpenMP block loop → serialise; correct first version).
  Column-major throughout (Eigen ≡ cuBLAS), `std::complex<double>` ≡ `cuDoubleComplex`,
  so host↔device copies are straight `cudaMemcpy`.
- `src/dmrg_blocks.cpp` — `block_svd_truncate` dispatches a block to the GPU when
  `min(rows,cols) >= QX_GPU_MIN_BLK` (default **128**), else CPU (LAPACKE if built
  with `QX_ENABLE_BLAS`, else the Eigen Gram path). Falls back to CPU on any device
  error. Profiling counter `ngpu` (see `QX_TEBD_PROFILE=1`).
- Build: `option(QX_USE_CUDA)` → `enable_language(CUDA)`, `find_package(CUDAToolkit)`,
  static `qx_gpu` lib (cusolver/cublas/cudart) linked into `qx_capi` and
  `qx_tdvp_blocks_test`; `/bigobj` for the large `dmrg_blocks.cpp` TU.

## Correctness

`qx_gpu_svd_test` — random complex `m×n` of several shapes (m<n, m>n, square,
64…300), vs Eigen BDCSVD:

| check | worst over shapes |
|---|---|
| reconstruction ‖UΣVᴴ−M‖/‖M‖ | 2.9e-15 |
| U orthonormal ‖UᴴU−I‖ | 4.3e-14 |
| V orthonormal ‖VᴴV−I‖ | 3.0e-12 |
| singular values vs BDCSVD (rel) | 5.7e-15 |

`qx_tdvp_blocks_test` with `QX_GPU_MIN_BLK=1` **and** `QX_GPU_THETA_MINWORK=1` (every
SVD *and* every theta block forced through the GPU) — the full TEBD / S(k,ω) /
Loschmidt / conductivity / TDVP2 suite passes vs exact diagonalization at the **same
tolerances** as CPU: staggered-mag 5.3e-6, energy conservation 1.8e-13, S(k,ω) 5.8e-3,
Loschmidt 7.2e-4, conductivity 1.2e-3. (The GPU theta path is gated to the no-sv-
absorption / dense-block case and otherwise falls back to the CPU `form_theta`, so
the TDVP/spectral paths that pass `sv` are byte-for-byte the CPU result.)

## Benchmark 1 — bare Gram-SVD kernel (the thing Phase A changes)

`qx_gpu_svd_bench`: complex-double square `d×d`, CPU `thin_svd_gram` (Eigen
`SelfAdjointEigenSolver`) vs GPU `thin_svd_gram_gpu_z`, best of N reps.

| d | CPU ms | GPU ms | GPU/CPU |
|---|---|---|---|
| 32 | 0.13 | 1.00 | 0.13× |
| 64 | 0.85 | 3.06 | 0.28× |
| 96 | 2.71 | 4.39 | 0.62× |
| 128 | 6.22 | 5.39 | **1.15×** |
| 192 | 20.2 | 6.44 | 3.13× |
| 256 | 47.9 | 8.49 | 5.64× |
| 384 | 158 | 14.9 | 10.6× |
| 512 | 383 | 22.0 | 17.4× |
| 768 | 1298 | 44.0 | 29.5× |
| 1024 | 3053 | 67.6 | **45.2×** |
| 1536 | 10535 | 163 | 64.6× |
| 2048 | 26394 | 308 | **85.8×** |

Crossover at **d ≈ 128** (below it PCIe transfer + launch overhead lose — hence the
threshold default). Above it the GPU advantage grows monotonically: this is the
"advantage grows with χ" property, at the kernel level, in its purest form.

## Benchmark 2 — end-to-end TEBD step (`qx_tebd_bench`)

Real-time Néel quench, Heisenberg (Δ=1), N=32, dt=0.05, 60 steps; identical code
path, GPU on/off selected by `QX_GPU_MIN_BLK`. CPU baseline here is the **Eigen
Gram** path (this build has no OpenBLAS; an OpenBLAS CPU baseline would be ~1.5×
faster, so the ratios below are conservative against a tuned CPU).

| χ | max_bond | CPU wall (s) | GPU wall (s) | end-to-end | SVD-phase (GPU/CPU) |
|---|---|---|---|---|---|
| 128 | 128 | 4.85 | 4.67 | 1.04× | ~1.0× |
| 256 | 256 | 21.31 | 21.86 | 0.97× | ~1.0× |
| 512 | 512 | 99.48 | 59.53 | 1.67× | 1.23× |
| 1024 | 1024 | 403.1 | 128.9 | 3.13× | **4.51×** |

The end-to-end margin turns upward and is decisively a win by χ=1024. (These are
the Phase-A first pass — one run per config, and the χ=1024 GPU run happened to be
cool; the controlled same-process 3-way in the Phase-B section gives the honest
Phase-A χ=1024 number, 2.60×. The qualitative "grows with χ" trend is robust either
way; what's thermally noisy is the *absolute* CPU baseline, not the GPU.)

## Why end-to-end < kernel — the Amdahl picture (and the Phase-B case)

From `QX_TEBD_PROFILE=1` at χ=1024 (per the 60-step run):

- **CPU step:** SVD 84%, theta-merge 14%, gate/recombine ~2%.
- **GPU step:** SVD 59%, **theta-merge 36%**, gate/recombine ~6%.

Three effects cap the Phase-A win:
1. **Block min-dim ≈ 0.5·χ, not 0.6·χ, and many blocks.** The χ bond is spread
   across ~20 Sz sectors; the largest block has `min(rows,cols)`=142 (χ=256), 224
   (χ=512), 552 (χ=1024). A TEBD step does ~380 block-SVDs; only the ~10 largest
   clear the 128 threshold and reach the GPU. The long tail of small SVDs stays on
   the CPU.
2. **theta-merge is now the #2 cost (36% on GPU)** and is entirely on the CPU.
3. **Per-block transfer + mutex serialisation** eat into the medium blocks
   (d≈128–256), where the kernel win is only ~1–6×.

All three are exactly what **Phase B** removes: keep the MPS tensors resident on
the GPU across the sweep (kills per-block PCIe transfer), run the small-block SVDs
as one `cusolverDnZheevjBatched` call (kills the CPU tail), and move `form_theta` /
`apply_gate` onto the GPU (kills the 36%). The kernel benchmark (up to 86×) is the
ceiling Phase B chases.

## Phase B investigation — why the per-GEMM offload does NOT work (measured)

The tempting low-risk Phase-B step is to offload each `form_theta` block GEMM
(`Ai_s1 * Ai1`, the 36%) to cuBLAS *individually*, leaving the data flow otherwise
unchanged. It was implemented (a per-call `gpu_matmul_z`, threshold `QX_GPU_GEMM_MIN`)
and **measured to be a net loss**, then reverted in favour of the per-block builder:

| χ=512, N=32 (GPU SVD on) | theta time | step time |
|---|---|---|
| Phase A (theta on CPU, fused Eigen GEMM) | baseline | 0.99 s/step |
| + per-call GEMM offload, threshold 96 | **↑ (much worse)** | 2.45 s/step |
| + per-call GEMM offload, threshold 256–512 | no better | 1.78–1.87 s/step |

Two structural reasons, both confirming the plan's residency thesis:
1. **Per-call transfer dominates.** `form_theta` issues ~1000 GEMMs/step (≈20 Sz
   sectors × 2 spin terms × 62 bonds × 2 half-sweeps). Each offloaded GEMM pays
   H2D(A)+H2D(B)+D2H(C) + kernel-launch + a `cudaMemcpy` sync (~0.5–1 ms of fixed
   cost), which for d≈100–280 blocks swamps the O(d³) compute saved. ~1000 × ~1 ms
   ≈ 1 s/step of pure overhead. This is the exact "never transfer per-block"
   pitfall in the plan §8.
2. **Returning `C = A*B` by value breaks Eigen's fused `dest.noalias() += A*B`**
   (one GEMM, no temporary) into alloc + GEMM + a second add pass — so even the CPU
   fallback regressed.

**Conclusion:** theta-merge can only be accelerated at the right *granularity* —
not per GEMM, but **per large theta block, built entirely on the device** so a
block's two spin-sector GEMMs share one upload and the O(n_row·kmid·n_col) work
amortises the O(·²) transfer. That is Phase B (below).

## Phase B — theta residency (implemented; the real win)

`form_theta_block_z` (in `gpu_svd.cu`) builds one theta block on the GPU: upload
`Ai1` once, run the up-sector GEMM (`Aup·Ai1` → rows `[0,dLup)`) and the dn-sector
GEMM (`Adn·Ai1` → rows `[dLup,n_row)`) into a single device `theta` buffer (disjoint
row ranges, `ldc=n_row`), download once. `form_theta` routes a block here when it is
large (`n_row·kmid·n_col ≥ QX_GPU_THETA_MINWORK`, default ~2.1e6 ≈ 128³), the scalar
is complex, and there is no sv-absorption (the TEBD case) — otherwise it uses the CPU
path, so results are identical (validated against ED, and the GPU path is gated on the
A blocks being dense, `rows == left-sector dim`). The small-block tail stays on the
CPU (transfer-bound), exactly mirroring the SVD threshold.

**Clean 3-way, one process, χ=1024, N=32, 60 steps** (identical work `nsvd=26310`,
CPU run first so the single-thread baseline gets the coolest clocks → the GPU
speedups are a conservative floor):

| config | wall (s) | s/step | theta | svd | speedup vs CPU |
|---|---|---|---|---|---|
| Pure CPU (Eigen) | 431.8 | 7.20 | 87.1 | 337.1 | 1.0× |
| Phase A (GPU SVD, CPU theta) | 166.1 | 2.77 | 79.3 | 78.6 | **2.60×** |
| **Phase B (GPU SVD + GPU theta)** | **97.8** | **1.63** | **11.9** | 77.6 | **4.42×** |

Phase B is **4.42× over single-thread CPU** and **1.70× over Phase A**. The theta
phase collapses from a co-bottleneck (87 s, 20% of the CPU step / 48% of the Phase-A
step) to **11.9 s (12%)** — a 7.3× phase speedup — and SVD is the clean 4.29× from
Phase A (337→78.6 s). What's left (svd 80%, recomb 6%, gate 2%) is the next frontier:
the small-block SVD tail (batched `cusolverDnZheevjBatched`) and keeping the A/U/V
tensors resident across bonds to drop the remaining per-block transfers.

**The win grows with χ** (clean cool-clock 3-ways, N=32, 60 steps, end-to-end wall):

| χ | CPU (s) | Phase A (s) | Phase B (s) | A vs CPU | B vs CPU | B vs A |
|---|---|---|---|---|---|---|
| 256  | 22.3  | 22.1  | 22.8 | 1.01× | 0.98× | 0.97× |
| 512  | 102.7 | 65.4  | 45.3 | 1.57× | 2.27× | 1.44× |
| 1024 | 431.8 | 166.1 | 97.8 | 2.60× | **4.42×** | **1.70×** |

At χ=256 the blocks are below both GPU thresholds, so all three are ~equal (correct:
no transfer the compute can't pay back). From χ=512 up, both phases turn on and the
margin climbs — the "advantage grows with χ" property, now end-to-end and with the
theta bottleneck removed.

> **Measurement note:** single-thread CPU times here are thermally noisy (boost vs
> sustained clocks can swing absolute times ±~20% run to run); the SVD/GPU phase
> timers are stable. All ratios above are from one back-to-back run on identical
> work, which controls for that. The earlier Phase-A-only singleton (3.13× at χ=1024)
> came from a cooler GPU run; the controlled 2.60× here is the honest Phase-A number.

## Phase C — FP32 mixed precision (opt-in; a modest win, eigensolver-bound)

The 4090's FP64 is ~1/64 of its FP32 throughput and the step is ~80% SVD, so FP32
*looked* like the big lever. In practice it is a **modest** win, because the SVD is
**eigensolver-bound, not GEMM-bound**, and the only FP32 eigensolver that is robust
on the steep/clustered Gram spectra of theta blocks is **iterative**:

- `cusolverDnCheevd` (FP32 divide & conquer) is fast but **fails to converge**
  (`info>0`) on these spectra — almost every large block fell back to CPU, making
  FP32 *slower* than FP64. FP64 `Zheevd` is robust; FP32 D&C is not.
- `cusolverDnCheevj` (FP32 Jacobi, tol 1e-5, ≤12 sweeps) is robust — the kept large
  Schmidt values converge in the first few sweeps — but its iteration cost eats most
  of the FP32 throughput advantage.

**Correctness:** cuSOLVER only guarantees a valid eigendecomposition at `info==0`. A
block whose Jacobi hits the 12-sweep cap (`info>0`) is **rejected**, and
`thin_svd_gram_gpu` falls back **FP32 → FP64-GPU → CPU** — so an unconverged block is
re-solved in FP64 *on the GPU* (still fast), never used unconverged and never dropped
to the CPU. In practice most large blocks converge in ≤12 sweeps, so this costs little.

`thin_svd_gram_gpu_c` does the Gram/eigensolve/back-multiply in FP32 (host casts
double↔float; final reorder + 1/σ scaling stays in double). Measured (same-binary
A/B, Phase A+B on, toggle `QX_GPU_FP32`; SVD-phase ratio is the thermally-stable
metric):

| χ | FP64 SVD phase | FP32 SVD phase | SVD speedup |
|---|---|---|---|
| 512  | 28.6 s | 20.6 s | **1.39×** |
| 1024 | 58.4 s | 44.5 s | **1.31×** |

**OFF by default** (`QX_GPU_FP32`: unset/0 = FP64; 1 = FP32 on real-time TEBD only;
≥2 = FP32 everywhere, testing). Gated to where single precision is tolerable
(real-time TEBD / S(k,ω) / Loschmidt / conductivity, Trotter-error-dominated);
DMRG, TDVP2, imag-time cooling, and the ground-state / J|ψ₀⟩ prep inside the
dynamics paths stay FP64. ED suite passes with `QX_GPU_FP32=1` (real-time TEBD errors
unchanged; TDVP2 energy 1.8e-13). Forcing FP32 onto TDVP2 (`=2`) breaks its 1e-6
energy check — confirming the gating is necessary. Large-χ real-time accuracy is not
ED-validatable (no ED beyond n=6), which is the other reason it ships opt-in.

## How to build & run

```
cmd /c "<VS>\VC\Auxiliary\Build\vcvars64.bat && \
  cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DQX_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 && \
  cmake --build build-cuda --target qx_gpu_svd_test qx_gpu_svd_bench qx_tebd_bench qx_tdvp_blocks_test"
```

- `build-cuda/Release/qx_gpu_svd_test.exe` — correctness (exit 0 = pass).
- `build-cuda/Release/qx_gpu_svd_bench.exe` — kernel scaling table.
- `qx_tebd_bench.exe [n] [chi] [steps] [dt]` — end-to-end; runtime knobs:
  `QX_GPU_MIN_BLK` (SVD→GPU threshold, default 128; huge = CPU-only SVD),
  `QX_GPU_THETA_MINWORK` (theta→GPU work threshold, default ~2.1e6; huge = CPU theta),
  `QX_GPU_FP32` (0/unset = FP64, 1 = FP32 real-time TEBD, ≥2 = FP32 everywhere),
  and `QX_TEBD_PROFILE=1` (per-phase timings + `ngpu`).

For production / Python (`quantx.dll`), also build the `qx_capi` target; at runtime
the loader must find `cusolver64_*.dll`/`cublas64_*.dll`/`cudart64_*.dll` (the CUDA
`bin` dir is on PATH on this box).

**OpenBLAS threading caveat:** QuantX parallelises over Sz blocks with its own OpenMP,
so it pins OpenBLAS to 1 thread (mutex-serialised, restored after the loop) — a
multi-threaded BLAS *inside* that loop oversubscribes and crashes. Because OpenBLAS's
thread count is *process-global*, a concurrent external numpy/scipy BLAS call on
another host thread can be transiently affected. When embedding the OpenBLAS build in
a multithreaded-BLAS host, run with `OPENBLAS_NUM_THREADS=1` (standard for OpenMP
hosts) or build without `QX_ENABLE_BLAS`.

## The headline — QuantX-GPU+OpenBLAS vs TeNPy-CPU (the real opponent)

The plan's deliverable: does the QuantX advantage **grow with χ against TeNPy**? Same
Néel-quench physics on both engines (`benchmarks/gpu_vs_tenpy_quench.py`): TeNPy
`SpinChain` Jx=Jy=Jz=4 (= σ·σ), 2nd-order Trotter, `svd_min=1e-12`, **single-thread**
(TeNPy's optimal, here backed by **Intel MKL** — a top-tier opponent, not OpenBLAS as
earlier drafts said). QuantX is the CUDA **+ OpenBLAS** build (large blocks → GPU,
small/sub-threshold blocks → OpenBLAS GEMMs via `EIGEN_USE_BLAS`). Controlled protocol
(consumer box, no clock pinning): **min-of-2 trials + cooldowns** (best trial =
least-throttled = each side's true capability), 50 steps. `qx_tebd_bench` vs in-process
TeNPy. Numbers below are **post-LAPACKE-fix** (Step 4 found and disabled a wrong-results
LAPACKE path — see that section); they are lower than the buggy-LAPACKE drafts.

**N=32** (TeNPy / QuantX-GPU / QuantX-FP32 seconds, then ratios):

| χ | TeNPy(s) | QX-GPU(s) | QX-FP32(s) | GPU/TeNPy | FP32/TeNPy |
|---|---|---|---|---|---|
| 128  | 11.45 | 2.46 | 2.44 | **~4.7×** | ~4.7× |
| 256  | 11.48 | 12.66 | 9.89 | 0.91× | 1.16× |
| 512  | 34.53 | 30.70 | 21.96 | 1.12× | 1.57× |
| 1024 | 139.42 | 64.36 | 50.94 | **2.17×** | **2.74×** |

(χ=128 is from a separate session — no block clears the ≥96 LAPACKE band there, so the
fix doesn't change it; it's the small-χ C++-overhead regime.)

**N=64** (post-fix, min-of-2; χ=128 from the earlier session, LAPACKE-irrelevant there):

| χ | TeNPy(s) | QX-GPU(s) | QX-FP32(s) | GPU/TeNPy | FP32/TeNPy |
|---|---|---|---|---|---|
| 128  | 13.13 | 6.82 | 7.42 | ~1.9× | ~1.8× |
| 256  | 31.68 | 40.09 | 30.80 | 0.79× | 1.03× |
| 512  | 110.26 | 101.99 | 73.44 | 1.08× | 1.50× |
| 1024 | 517.59 | 232.99 | 194.33 | **2.22×** | **2.66×** |

The honest, controlled picture (these **supersede** all earlier drafts: the single-run
"4.04×" caught a *throttled* TeNPy, and the buggy-LAPACKE numbers were inflated):

- **The advantage grows with χ in the GPU regime** (the plan's point), consistently for
  both sizes: N=32 256→1024 0.91→1.12→**2.17×** (FP32 2.74×); N=64 0.79→1.08→**2.22×**
  (FP32 2.66×). At χ=1024 both land ~**2.2×** (FP32 ~2.7×).
- **Small χ is a separate C++-overhead win** (N=32 χ=128 ~4.7×): no block clears the GPU
  threshold, so this is QuantX-CPU vs TeNPy's Python orchestration.
- **Mid χ (256) is ~break-even-to-slightly-below** (0.79–0.91×): blocks below the GPU
  threshold, TeNPy's MKL efficient; this is the engine's weakest regime.
- Earlier drafts showed bigger large-χ multiples (up to 4.56×); those were **thermally
  inflated** (a throttled single-thread TeNPy ran 2–2.5× slower). Cool-clock min-of-2
  gives ~2.2× — the honest number.
- **A mid-χ dip** (χ=256) sits between the two regimes — **~break-even post-fix**
  (N=32 0.91×). `EIGEN_USE_BLAS` gives a small lift over CUDA-only-Eigen but doesn't
  clear 1× there. (An earlier draft showed 1.43×, but that came from the buggy LAPACKE
  eigensolve, now disabled for correctness — see Step 4.)

## Step 4 — observables inherit the speedup (spot-check) + a correctness fix

Spot-check (`qx_obs_bench`): the spin-conductivity C(t)=⟨J(t)J(0)⟩ (block DMRG ground
state → J|ψ₀⟩ MPO apply → real-time block TEBD), N=32, **true CPU (both GPU paths off)
vs full GPU, same session**, results compared by fingerprint (C(0)=⟨J²⟩, Σ|C|):

| χ | CPU (s) | GPU (s) | GPU/CPU | C(0) match |
|---|---|---|---|---|
| 512  | 74.1 | 54.9 | 1.35× | 4.857261 = 4.857261 ✓ |
| 1024 | 215.4 | 100.3 | **2.15×** | 4.857268 ≈ 4.857261 ✓ (~6 digits) |

So the observables **inherit the engine speedup** (2.15× at χ=1024, consistent with the
vs-TeNPy ~2.2×) and the GPU path **reproduces the CPU result** — there is no ED at this
χ, but GPU≡CPU to ~6 digits is the consistency proof. (Part of the observable is the
real-scalar DMRG ground state, which is CPU both ways, so the GPU speedup applies only
to the complex evolution — that is why the total ratio tracks the evolution's, not the
larger bare-SVD multiples.)

**Correctness fix found here (important).** The first χ=512 conductivity gave a *garbage*
DMRG ground state (E0=−3395 vs the correct −55.99) — and on the CPU(OpenBLAS) path, not
the GPU. Isolated to `thin_svd_lapacke`: this conda-forge OpenBLAS returns **wrong
results from `LAPACKE_dsyevd/zheevd` at some block sizes** (χ=512 hit it; χ=256/1024
happened to be fine). The ED test (n=6, blocks <96) never exercises the ≥96 LAPACKE
path, so #141 shipped it. **Fix:** the LAPACK eigensolve is now opt-in (`QX_LAPACKE_MIN`,
default disabled); `EIGEN_USE_BLAS` still routes the Gram trick's dominant GEMMs to
OpenBLAS (the bulk of the small-χ speedup, and correct), with the eigensolve back on
Eigen. χ=512 is now correct. **Follow-up: add a large-χ DMRG-energy regression test**
(the gap that hid this).

## Limitations / next

- **Thermal noise dominates the absolute ratios** on this consumer box (no clock
  pinning, single-thread CPU turbo + GPU boost both drift with session thermal state).
  Two min-of-2 measurements of N=64 χ=256/512 gave 0.55×/1.03× (hot) vs 1.28×/2.24×
  (cool) — a ~2× spread. **Trust the χ-scaling trend and the large-χ multi-× win; treat
  the exact multiple as ±~2×.** A clock-pinned/server box would tighten it.
- **Real scalar (DMRG) stays on CPU** by design (the wrapper returns false for
  `S=double`); DMRG is already a strong CPU win and not the target.
- **Mid-χ (χ≈256)** is the weakest regime (blocks below the GPU threshold, TeNPy's MKL
  efficient). Levers: lower `QX_GPU_MIN_BLK` (watch the transfer crossover) or batch the
  small blocks (`cusolverDnZheevjBatched`).
- **Phase D frontier:** the χ=1024 step is now ~80% SVD; batch the small-block SVD
  tail (`cusolverDnZheevjBatched`) and keep A/U/V resident across bonds (drop the
  per-bond host round-trips) for the next step up.
- **FP32 is opt-in and modest** (eigensolver-bound, see Phase C); a faster robust FP32
  eigensolver (or a batched Jacobi) is the lever to make it a bigger win.
