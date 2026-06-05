# GPU Acceleration Plan for QuantX Block-Sparse Tensor-Network Engine

**Status:** handoff spec for a session on a GPU machine. This machine has no GPU,
so the plan below is unimplemented and untested. Everything it references
(`src/dmrg_blocks.cpp`, the tests, the benchmarks) exists and is validated on CPU.

**Goal:** make QuantX's advantage over TeNPy **grow with system scale** (bond
dimension χ). On CPU single-thread we are at the LAPACK floor on the dominant SVD
(see "Why" below); the only way the margin grows with χ is to do the O(χ³) dense
linear algebra on hardware where our **block structure** is an asset — the GPU.

---

## 1. Context: what QuantX is and where the time goes

QuantX is a C++ block-sparse (Sz-/U(1)-symmetric) MPS/MPO engine with a Python
wrapper. The relevant winning features (all on `master`, validated vs exact
diagonalization, and benchmarked vs TeNPy):

- **DMRG** ground state (U(1)+SU(2)): 4.8–6× vs TeNPy.
- **Block-sparse TEBD** real-time evolution: 1.07–2.37× vs TeNPy TEBD.
- **Dynamical observables on TEBD**: S(k,ω) (1.84–16.8×), Loschmidt/DQPT
  (1.75–10×), spin conductivity σ(ω). All XXZ-general (anisotropy Δ).

**The bottleneck (measured with `QX_TEBD_PROFILE=1`):** a TEBD step at large χ is
**~87% SVD** (N=64, χ=128: theta-merge 11%, gate 1%, SVD 87%, recombine 1%). The
SVD is per-Sz-sector and currently runs on CPU via the Gram-matrix trick
(`thin_svd_gram` in `src/dmrg_blocks.cpp`): SVD of an `m×n` block via the
Hermitian eigendecomposition of the smaller Gram matrix `M Mᴴ` (or `Mᴴ M`) using
Eigen's `SelfAdjointEigenSolver`.

**Measured margin vs χ (Néel quench, N=32, single-thread):**

| χ | QuantX | TeNPy | margin |
|---|--------|-------|--------|
| 128 | 5.54 s | 6.71 s | **1.21×** |
| 256 | 28.7 s | 16.2 s | **0.56× (we LOSE ~1.8×)** |

**Root cause (important):** QuantX links **no optimized BLAS/LAPACK** — it uses
Eigen's *built-in* routines (verified: no `BLAS`/`LAPACK`/`EIGEN_USE_*` anywhere
in the build). TeNPy's numpy uses **OpenBLAS** (`scipy-openblas`). At large χ the
SVD/eigensolve is O(χ³) dense linear algebra, where OpenBLAS's tuned `gesdd`/
`heevd` crushes Eigen's generic eigensolver, *and* our Gram trick adds extra GEMMs
(form `M Mᴴ`, reconstruct `V = Mᴴ U / sv`). So the χ=256 reversal is **not a
fundamental floor** — we are *below* the floor because we never linked a fast
BLAS. That means there is a large, cheap **CPU** win available before any GPU work
(Phase 0). The bond is concentrated in a few central Sz sectors, so one **dominant
block** (~0.6·χ wide) dominates the cost.

**Why the block structure is a GPU asset:** at bond χ the central bond carries
~√χ–N Sz sectors → **many independent SVDs** (perfect for cuSOLVER *batched*
routines) plus one large eigensolve (perfect for a single big GPU `Zheevd`).
TeNPy's npc orchestration is Python-bound and its dense per-block SVD doesn't
expose this batch structure as cleanly. The GPU advantage on O(χ³) work **grows
with χ**, which is exactly the "bigger = more advantage" property we want.

---

## 2. Success criteria

1. **Correctness:** `qx_tdvp_blocks_test` and the Python test suite pass unchanged
   (GPU SVD must reproduce the CPU Gram-eigensolve to the same tolerances; the
   existing tests validate against ED — staggered mag ~5e-6, energy conservation
   ~1e-13, S(k,ω)/Loschmidt/conductivity vs ED).
2. **The margin grows with χ:** benchmark QuantX-GPU vs TeNPy-CPU TEBD on a Néel
   quench (N=32–64) at χ = 128, 256, 512, 1024 and show the speedup **increases**
   with χ (CPU baseline is flat ~1.1–1.4×; target GPU curve trends upward).
3. No regression on small problems (fall back to CPU below a size threshold where
   PCIe transfer dominates).

---

## 3. Design — phases

### Phase 0 (CPU) — DONE on this machine. Real ~1.5× win, but does NOT fix large-χ.

Implemented and measured (opt-in via `-DQX_ENABLE_BLAS=ON`, linking conda-forge
OpenBLAS). Two levers, in order of impact:

1. **`EIGEN_USE_BLAS` (the big one):** routing Eigen's GEMMs to OpenBLAS. The Gram
   SVD is GEMM-dominated (`M Mᴴ` + `Mᴴ U` ≈ 4m³ vs the m³ eigensolve), and Eigen's
   built-in GEMM was the bottleneck. This alone lifted the whole curve ~1.5×.
2. **LAPACK `zheevd` for the large-block Gram eigensolve** (`thin_svd_lapacke`,
   threshold `min(m,n) ≥ 96`); small blocks keep the pure-Eigen Gram path. A
   *direct* `zgesdd` was tried too but was **slower** at χ=256 (0.67× vs 0.79×) —
   the Gram trick is fewer FLOPs, so keep it and just put its eigensolve on OpenBLAS.

**Measured (N=32 quench, single thread), margin = TeNPy/QuantX:**

| χ | pre-BLAS | Phase 0 (OpenBLAS) |
|---|----------|--------------------|
| 128 | 1.21× | **1.84×** |
| 256 | 0.56× | **0.79×** |

So Phase 0 is a genuine ~1.5× uniform speedup — χ=128 becomes a strong win and
χ=256 goes from a clear loss to nearly even. **But the margin still *decreases*
with χ (1.84 → 0.79), i.e. Phase 0 does NOT make the advantage grow with scale.**
At large χ our Gram-on-OpenBLAS still does not beat TeNPy's tuned per-block
`gesdd` on its own OpenBLAS. The "advantage grows with χ" goal genuinely needs the
GPU (Phases A/B) — but now the CPU baseline is honest (OpenBLAS-backed), so the GPU
comparison will be fair.

**Build/runtime notes (Windows):** OpenBLAS from `conda create -n qxblas -c
conda-forge openblas`; configure with `-DQX_ENABLE_BLAS=ON
-DQX_BLAS_INCLUDE_DIR=<env>/Library/include/openblas
-DQX_BLAS_LIB=<env>/Library/lib/openblas.lib`. At runtime the loader must find
`openblas.dll` — Python 3.8+ ignores PATH for a loaded DLL's dependencies, so use
`os.add_dll_directory(<env>/Library/bin)` before importing (or copy the dll next
to `libquantx.dll`). Default build is `QX_ENABLE_BLAS=OFF` and unaffected.

### Phase A (GPU): GPU only the large-block SVDs

Keep the whole TEBD/DMRG control flow on CPU. Inside `block_svd_truncate`, route
**only the large Sz-sector blocks** (min dim above a threshold, e.g. ≥ 64) to the
GPU; leave the many tiny tail blocks on the CPU (they are cheap and transfer-
dominated).

Per large block `M` (`m×n`, complex double; let `m = min(m,n)`):
1. Transfer `M` to GPU (O(m·n) — the matrix; small vs the O(m³) compute).
2. Form the Gram matrix `G = M Mᴴ` (if m ≤ n) or `Mᴴ M` (m×m Hermitian) with
   cuBLAS `cublasZherk`.
3. Eigendecompose `G = U Λ Uᴴ` with cuSOLVER `cusolverDnZheevd` (descending sv =
   sqrt(Λ)). For the truncation we need all eigenpairs of the small Gram matrix.
4. `V = Mᴴ U diag(1/sv)` with cuBLAS `cublasZgemm` + a scaling kernel (guard tiny
   sv exactly as the CPU `thin_svd_gram` does: zero the column if sv < 1e-14·svmax).
5. Transfer back `U (m×m)`, `V (n×m)`, `sv (m)`.

The global truncation (cross-sector singular-value sort, `keep_per_q`) stays on
CPU — it needs all sectors' singular values together, which is a cheap O(total χ)
sort. So: GPU returns full per-block `U,V,sv`; CPU does the global sort + extracts
the kept columns (`U_up/U_dn/Vt`) exactly as today.

**Threshold + fallback:** `if (min(m,n) < QX_GPU_MIN_BLK) use CPU thin_svd_gram;
else GPU`. Tune `QX_GPU_MIN_BLK` (start ~64). Below it, transfer overhead loses.

**Expected outcome:** accelerates the dominant block (the Amdahl bottleneck),
which is the 87%. This alone should make the margin grow with χ.

### Phase B (if Phase A pays off): keep the state on GPU, batch everything

PCIe transfer per bond is the enemy. Phase B amortizes it by keeping the MPS
tensors (and DMRG environments) **resident on GPU** across the whole sweep, and
running `form_theta`, `apply_gate_theta`, the SVD, and `bt_A_from_U`/`bt_center_*`
on GPU. The many small block SVDs become one **batched** call
(`cusolverDnZheevjBatched` on the stacked small Gram matrices). Only scalars
(energies, truncation errors) and the final result come back to CPU.

This is a larger rewrite (all block kernels gain GPU paths) but is where the full
speedup lives. Gate it behind a runtime/compile flag so the CPU path remains the
default and the reference.

---

## 4. Where to change (file/function map)

All in `src/dmrg_blocks.cpp` unless noted:

- `thin_svd_gram(const Mat& M, Mat& U, Mat& V, RVec& s)` — the CPU Gram SVD. Add
  a sibling `thin_svd_gram_gpu(...)` with the same signature/semantics. This is
  the single numerical kernel to port for Phase A.
- `block_svd_truncate(...)` — in the per-sector loop
  (`#pragma omp parallel for ... thin_svd_gram(*svblk[i], ...)`), dispatch to
  `thin_svd_gram_gpu` when `min(rows,cols) >= QX_GPU_MIN_BLK`. The surrounding
  global-truncation logic is unchanged.
- `tebd2_step_blocks`, `tdvp2_step_blocks`, `run_tebd2`, `run_spectral`,
  `run_conductivity` — unchanged for Phase A (they call `block_svd_truncate`).
- New file `src/gpu_svd.cu` (+ `src/gpu_svd.hpp`) holding the CUDA/cuSOLVER/cuBLAS
  implementation; `thin_svd_gram_gpu` is a thin host wrapper declared in the hpp
  and called from `dmrg_blocks.cpp`. Keep CUDA out of `dmrg_blocks.cpp` (compile
  it as plain C++); only the `.cu` file needs `nvcc`.
- A persistent GPU context (cuSOLVER/cuBLAS handles, reusable device buffers) —
  create once (thread-local or a singleton), not per call. Per-call handle
  creation will dominate and kill performance.

The data layout matches: `Mat = Eigen::MatrixXcd` is column-major, as is cuBLAS/
cuSOLVER — so a host→device copy of the raw `.data()` pointer is a straight
`cudaMemcpy` with no transpose. `std::complex<double>` is bit-compatible with
`cuDoubleComplex`.

---

## 5. Build system

- CMake: `enable_language(CUDA)`, `find_package(CUDAToolkit)`, link
  `CUDA::cusolver CUDA::cublas CUDA::cudart`. Compile `src/gpu_svd.cu` with nvcc;
  the rest stays the host C++ compiler. Gate with `option(QX_USE_CUDA ...)` and a
  `#ifdef QX_USE_CUDA` around the dispatch so the CPU build is unaffected.
- Set `CMAKE_CUDA_ARCHITECTURES` for the target GPU (e.g. `80` for A100, `89` for
  4090, `90` for H100).

---

## 6. Validation plan

1. Build with `QX_USE_CUDA=ON`, run `qx_tdvp_blocks_test` — all checks must pass
   (they compare to ED). The GPU Gram-eigensolve vs the CPU one should agree to
   ~1e-10; if a test that was 5e-6 drifts, investigate the eigensolver tolerance.
2. Add a direct unit check: random complex `m×n` blocks, compare
   `thin_svd_gram_gpu` vs `thin_svd_gram` (reconstruct `U diag(sv) Vᴴ` ≈ M, and
   `‖UᴴU − I‖`, `‖VᴴV − I‖`).
3. Run the Python suite (`pytest python/quantx_engine/tests`); the
   TEBD/spectral/Loschmidt/conductivity/DMRG tests must stay green.

## 7. Benchmark plan (the deliverable)

Reuse the existing harness style (`benchmarks/spectral_vs_tenpy.py`,
`benchmarks/tdvp_vs_tenpy.py`). Single-thread CPU for TeNPy (its optimal config),
GPU for QuantX:

- **margin vs χ**: Néel quench, N=32 and 64, χ ∈ {128, 256, 512, 1024}, fixed
  step count. Plot QuantX-GPU / TeNPy-CPU speedup vs χ. **Expected: upward trend**
  (vs the flat ~1.1–1.4× CPU curve measured in `docs/time_evolution_status.md`).
- Also record QuantX-GPU vs QuantX-CPU to isolate the GPU speedup, and (if a
  recent TeNPy with a `cupy`/`torch` backend is available) QuantX-GPU vs
  TeNPy-GPU for the honest apples-to-apples.
- The same observables (S(k,ω), conductivity) inherit the engine speedup; spot-
  check one at large χ.

Write results into `docs/gpu_acceleration_report.md` and update
`docs/time_evolution_status.md` with the GPU curve.

---

## 8. Pitfalls & mitigations

- **PCIe transfer dominates small blocks** → threshold (Phase A) / keep state on
  GPU (Phase B). Never transfer per-tiny-block.
- **Per-call handle/allocation overhead** → persistent handles + a reused device
  workspace pool sized to the max block seen.
- **cuSOLVER `Zheevd` is one-matrix** → for the many small blocks use
  `cusolverDnZheevjBatched` (Jacobi, batched) in Phase B; in Phase A just leave
  small blocks on CPU.
- **Gram squares the condition number** — already true on CPU and accepted (the
  truncated tail loses precision, the kept Schmidt values are fine). GPU inherits
  this; don't "fix" it with a full SVD (slower). If a low-T/DMRG accuracy issue
  appears, fall back to `cusolverDnZgesvdj` (Jacobi SVD, no squaring) for DMRG.
- **Mixed precision (stretch):** real-time TEBD truncation tolerates single
  precision; a `Cheevd`/`Cgesvdj` path is ~2× on top and much more on consumer
  GPUs (FP32 >> FP64 throughput). Keep FP64 for DMRG/energy-exact paths.
- **Determinism/threads:** keep the CPU OpenMP block loop, but only the *small*
  blocks on CPU threads while a CUDA stream handles the large block concurrently
  (overlap), or serialize for a first correct version.

---

## 9. Recommended order of work

0. **Phase 0 (CPU, do first):** link OpenBLAS/MKL (`EIGEN_USE_BLAS`/`LAPACKE`) +
   size-adaptive SVD (direct `zgesdd` for large blocks). Re-run margin-vs-χ;
   expect χ=256 to recover from 0.56× to ≥1×. Cheap, and a prerequisite for a fair
   GPU comparison.
1. Phase A `thin_svd_gram_gpu` (large blocks only) + build flag + unit test +
   `qx_tdvp_blocks_test` green.
2. margin-vs-χ benchmark on GPU → confirm the speedup **grows** with χ. **Key result.**
3. If it pays: Phase B (state-resident-on-GPU, batched small blocks).
4. Stretch: mixed precision; TeNPy-GPU comparison.

Note: Phase 0 can (and should) be done on this CPU machine; only Phases A/B need
the GPU box.

The whole point is item 2: demonstrate, with data, that the QuantX advantage
**grows with χ** on GPU — turning "we tie LAPACK at large scale on CPU" into "we
pull away at large scale on GPU", which is the regime that actually matters.
