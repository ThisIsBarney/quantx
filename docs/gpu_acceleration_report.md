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
| **QuantX-GPU+OpenBLAS vs TeNPy-MKL** (N=32, min-of-2) | 1.43× | 1.59× | **2.38×** (2.92× w/ FP32) |

(First three rows: N=32 Néel quench, 60 steps, GPU-vs-CPU same-process 3-ways. Last
row: the real opponent — QuantX(CUDA+OpenBLAS) vs single-thread TeNPy(MKL), min-of-2
controlled; also 4.65× at χ=128 where C++ overhead beats TeNPy's Python, and **N=64
χ=1024 = 4.56× / 5.86× FP32** — bigger N, bigger win. Absolute ratios are thermally
noisy (±~2×) on this box; the χ-trend is robust.) The internal GPU-vs-CPU curves
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

## The headline — QuantX-GPU+OpenBLAS vs TeNPy-CPU (the real opponent)

The plan's deliverable: does the QuantX advantage **grow with χ against TeNPy**? Same
Néel-quench physics on both engines (`benchmarks/gpu_vs_tenpy_quench.py`): TeNPy
`SpinChain` Jx=Jy=Jz=4 (= σ·σ), 2nd-order Trotter, `svd_min=1e-12`, **single-thread**
(TeNPy's optimal, here backed by **Intel MKL** — a top-tier opponent, not OpenBLAS as
earlier drafts said). QuantX is the CUDA **+ OpenBLAS** build (large blocks → GPU,
small/sub-threshold blocks → OpenBLAS). Controlled protocol (consumer box, no clock
pinning): **min-of-2 trials + cooldowns** (best trial = least-throttled = each side's
true capability), 50 steps. `qx_tebd_bench` native exe vs in-process TeNPy.

**N=32** (TeNPy / QuantX-GPU / QuantX-FP32 seconds, then ratios):

| χ | TeNPy(s) | QX-GPU(s) | QX-FP32(s) | GPU/TeNPy | FP32/TeNPy |
|---|---|---|---|---|---|
| 128  | 11.45 | 2.46 | 2.44 | **4.65×** | 4.70× |
| 256  | 29.46 | 20.59 | 17.62 | 1.43× | 1.67× |
| 512  | 63.40 | 39.85 | 29.02 | 1.59× | 2.18× |
| 1024 | 198.47 | 83.44 | 68.02 | **2.38×** | **2.92×** |

**N=64** (large-χ tail re-measured in a separate, cooler session — see the noise note):

| χ | TeNPy(s) | QX-GPU(s) | QX-FP32(s) | GPU/TeNPy | FP32/TeNPy |
|---|---|---|---|---|---|
| 128  | 13.13 | 6.82 | 7.42 | 1.92× | 1.77× |
| 256  | 84.53 | 65.97 | 52.67 | 1.28× | 1.60× |
| 512  | 289.70 | 129.41 | 102.28 | 2.24× | 2.83× |
| 1024 | 1318.39 | 289.20 | 224.87 | **4.56×** | **5.86×** |

The honest, controlled picture (this **supersedes** the earlier single-run "4.04× at
N=32 χ=1024" — that caught a *throttled* TeNPy; clean best-vs-best is **2.38×**):

- **The advantage grows with χ in the GPU regime** (the plan's point). N=32: 256→1024
  1.43→1.59→**2.38×** (FP32 2.92×). N=64: 256→1024 1.28→2.24→**4.56×** (FP32 **5.86×**).
- **Bigger N → bigger large-χ win** (N=64 χ=1024 4.56× vs N=32 2.38×): at fixed χ more
  sites = more O(χ³) work where the GPU compounds, against more of TeNPy's per-bond
  Python overhead.
- **Small χ is a separate C++-overhead win** (N=32 χ=128 **4.65×**): no block clears the
  GPU threshold, so this is QuantX-CPU-OpenBLAS vs TeNPy's Python orchestration.
- **A shallow mid-χ dip** (χ=256) sits between the two regimes; **Step 1 (OpenBLAS)
  lifted it** (N=32 χ=256 ~0.8× CUDA-only → **1.43×**), so it's now a modest win rather
  than a loss.

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
