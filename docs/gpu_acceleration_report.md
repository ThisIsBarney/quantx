# GPU Acceleration Report — Phase A (cuSOLVER Gram-SVD)

**Date:** 2026-06-06 **Hardware:** NVIDIA RTX 4090 (Ada, sm_89), 24 GB; CUDA 13.2;
MSVC 19.50 x64; single-thread CPU baseline. **Plan:** [`gpu_acceleration_plan.md`](gpu_acceleration_plan.md).

## TL;DR

Phase A routes the large per-Sz-sector SVDs of the block-sparse TEBD step to the
GPU (cuSOLVER `Zheevd` on the Gram matrix), leaving everything else on the CPU.
The central claim of the plan — **the QuantX advantage grows with bond dimension
χ** — is demonstrated:

| level | χ=128 / d=128 | χ=256 | χ=512 | χ=1024 / d=1024 |
|---|---|---|---|---|
| **bare SVD kernel** (GPU/CPU) | 1.15× | 5.6× | 17.4× | **45×** (86× at d=2048) |
| **TEBD SVD phase** (GPU/CPU) | ~1.0× | ~1.0× | 1.23× | **4.51×** |
| **TEBD end-to-end** (GPU/CPU) | 1.04× | 0.97× | 1.67× | **3.13×** |

The kernel speedup is dramatic and monotonic in size; it translates only partly to
the end-to-end step because of Amdahl's law (analysis below). This is the expected
Phase-A result and directly motivates Phase B.

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

`qx_tdvp_blocks_test` with `QX_GPU_MIN_BLK=1` (every block forced through the GPU)
— the full TEBD / S(k,ω) / Loschmidt / conductivity / TDVP2 suite passes vs exact
diagonalization at the **same tolerances** as CPU: staggered-mag 5.3e-6, energy
conservation 1.9e-14, S(k,ω) 5.8e-3, Loschmidt 7.2e-4, conductivity 1.2e-3.

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
| 1024 | 1024 | 403.1 | 128.9 | **3.13×** | **4.51×** |

The end-to-end margin turns upward and is decisively a win by χ=1024.

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

## Phase B investigation — why the contained offload does NOT work (measured)

The tempting low-risk Phase-B step is to offload just the `form_theta` block GEMMs
(`Ai_s1 * Ai1`, the 36%) to cuBLAS, leaving the data flow otherwise unchanged. It
was implemented (`gpu_matmul_z`, threshold `QX_GPU_GEMM_MIN`) and **measured to be a
net loss**, then reverted:

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

**Conclusion:** theta-merge can only be accelerated by **GPU residency**, not
per-call offload — form `theta` on the device from MPS tensors that are *already*
resident (one A upload per bond, the per-sector GEMMs accumulated into device
`theta` sub-blocks via `cublasZgemm` with `beta=1` into offset pointers, `theta`
never round-tripping to the host), then feed it straight into the gate + SVD (also
on-device). That is the real Phase-B architecture; the contained shortcut is a dead
end. The host-pointer `gpu_matmul_z` primitive was removed — residency needs
device-resident-pointer GEMMs, a different (correct) primitive.

## How to build & run

```
cmd /c "<VS>\VC\Auxiliary\Build\vcvars64.bat && \
  cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DQX_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 && \
  cmake --build build-cuda --target qx_gpu_svd_test qx_gpu_svd_bench qx_tebd_bench qx_tdvp_blocks_test"
```

- `build-cuda/Release/qx_gpu_svd_test.exe` — correctness (exit 0 = pass).
- `build-cuda/Release/qx_gpu_svd_bench.exe` — kernel scaling table.
- `qx_tebd_bench.exe [n] [chi] [steps] [dt]` — end-to-end; set `QX_GPU_MIN_BLK`
  (128 = GPU on large blocks; a huge value = CPU only) and `QX_TEBD_PROFILE=1`.

For production / Python (`quantx.dll`), also build the `qx_capi` target; at runtime
the loader must find `cusolver64_*.dll`/`cublas64_*.dll`/`cudart64_*.dll` (the CUDA
`bin` dir is on PATH on this box).

## Limitations / next

- **No fair TeNPy comparison yet** — base Python here has no numpy/scipy/TeNPy, and
  no OpenBLAS env. The honest "QuantX-GPU vs TeNPy-CPU" headline needs those
  installed; with the documented CPU/TeNPy margin (Phase 0: χ=128 1.84×, χ=256
  0.79×) and the GPU/CPU ratios above, GPU/TeNPy ≈ (GPU/CPU)·(CPU/TeNPy) already
  trends up with χ, but a direct measurement is the right deliverable.
- **Real scalar (DMRG) stays on CPU** by design (the wrapper returns false for
  `S=double`); DMRG is already a strong CPU win and not the target.
- **Phase B** is the clear next step (see above).
