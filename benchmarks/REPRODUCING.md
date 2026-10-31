# Reproducing the comparisons with TeNPy and ITensor

The technical reports in `docs/` compare QuantX with TeNPy (and once with ITensor) on
the same problems. This page says which script produced which claim, what each needs,
and a small configuration that checks a script runs end to end before you commit to
the full run. The published numbers were measured on one desktop (Intel i5-13600KF,
NVIDIA RTX 4090, Windows), whose clocks throttle: expect the ratios to move by up to
2× between runs and machines, and the trend with χ to hold.

## What you need

- **TeNPy 1.0.6**, the version the reports used, with its NumPy (MKL on the reference
  machine). On Windows with conda, run from an *activated* environment: an
  un-activated conda Python finds the wrong BLAS DLLs, and TeNPy dies with a Windows
  exception 0xc06d007f.
- **QuantX**, in one of two forms, by script (the table says which):
  - *Python API*: the installed package (`pip install quantx-engine`), or
    `QUANTX_LIB_PATH` pointing at a library from a source build.
  - *Native program*: a CMake build of the named target, which times the engine
    without the Python layer:

    ```bash
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release --target qx_tebd_bench qx_hubbard_showcase
    ```

    Both run on the CPU; a CUDA build adds the GPU path to `qx_tebd_bench`. The
    scripts look under `QX_BUILD_DIR` (default `build/`); `QX_TEBD_BENCH` and
    `QX_HUB_SHOWCASE` override with a full path. `QX_DLL_DIR` goes first on PATH for
    the runs, for a Windows build whose OpenBLAS or CUDA DLLs are elsewhere.
- **The GPU comparison** needs a CUDA build (`-DQX_USE_CUDA=ON`) and an NVIDIA GPU.
- **The ITensor comparison** needs Julia with ITensors.jl and ITensorMPS.jl, `julia` on
  PATH or `JULIA_EXE` set to it.

Each script prints a table and writes `benchmarks/results_<name>.json` (not tracked).
The size, bond dimensions and repetitions come from `BENCH_*` variables; the quick
configurations below set them small.

## The comparisons

| Claim, and where | Script | QuantX side |
|---|---|---|
| Heisenberg ground state, U(1) DMRG 2.7–26× faster than TeNPy across accuracy; SU(2) faster still ([dmrg_su2_benchmark_report.md](../docs/dmrg_su2_benchmark_report.md) §4) | `dmrg_su2_vs_tenpy.py` | Python API |
| S(k, ω) of the Heisenberg chain, 1.84–16.8× faster than TeNPy's TEBD ([time_evolution_status.md](../docs/time_evolution_status.md)) | `spectral_vs_tenpy.py` | Python API |
| The retracted "TDVP2 8–50× faster" ([dmrg_tdvp_report.md](../docs/dmrg_tdvp_report.md), the correction at the top). **Superseded:** the engine now runs this nearest-neighbour quench with block TEBD (method `tebd2-block`), and TeNPy's fast method here is its TEBD, not the TDVP engine the script times. Kept as an accuracy check against exact diagonalisation | `tdvp_vs_tenpy.py` | Python API |
| Spin-1 Néel-quench TEBD: CPU 1.30–1.46×, with the GPU ≈2.0× at χ = 512 ([universal_mpo.md](../docs/universal_mpo.md), spin-1) | `spin1_tebd_vs_tenpy.py` | `qx_tebd_bench` |
| Hubbard CDW-quench TEBD, N = 32: 1.81× at χ = 256, 1.15× at 512 ([universal_mpo.md](../docs/universal_mpo.md), Hubbard TEBD) | `hubbard_tebd_vs_tenpy.py` | `qx_tebd_bench` |
| Hubbard chain ground state, N = 64: 2.17× at χ = 128, 1.37× at 256 ([universal_mpo.md](../docs/universal_mpo.md), Hubbard DMRG) | `hubbard_dmrg_vs_tenpy.py` | `qx_hubbard_showcase` |
| Two-leg Hubbard ladder, Lx = 16, U = 8: 1.36× site-split ([universal_mpo.md](../docs/universal_mpo.md), ladder) | `hubbard_ladder_vs_tenpy.py` | `qx_hubbard_showcase` |
| W = 4 Hubbard cylinder, 8×4, U = 8, 12 sweeps: 2.06× after the optimisation round, TeNPy lower in energy by 4.6e-4 ([universal_mpo.md](../docs/universal_mpo.md), cylinder) | `hubbard_cylinder_vs_tenpy.py` | `qx_hubbard_showcase` |
| Hubbard chain ground state against ITensor ([universal_mpo.md](../docs/universal_mpo.md), ITensor) | `hubbard_dmrg_vs_itensor.py` | `qx_hubbard_showcase` |
| GPU Néel-quench TEBD against single-thread TeNPy, N = 32: the default sweep 0.93× at χ = 256 and 2.20× at 1024; the even/odd scheme (M12) 1.33× at 256 (1.40× at N = 64) and 1.00× at 1024 ([gpu_acceleration_report.md](../docs/gpu_acceleration_report.md), M12) | `gpu_vs_tenpy_quench.py` | `qx_tebd_bench`, CUDA build |

## What the S(k, ω) comparison compares

`spectral_vs_tenpy.py` times both codes at the same Trotter step and order, not at the
same accuracy, and the two correlations differ by more than either's truncation. With
`H = Σ σ·σ` the step dt = 0.05 is 0.2 in units of the S·S exchange, a coarse step, and
both codes carry a second-order Trotter error to match. Measured against exact dynamics
at N = 12
(where χ = 64 holds the state whole), up to t = 3:

| dt | QuantX − exact | TeNPy − exact | QuantX − TeNPy |
|---|---|---|---|
| 0.05 | 2.7e-2 | 3.1e-2 | 2.7e-2 |
| 0.025 | 6.8e-3 | 7.7e-3 | 6.8e-3 |

Both errors fall by 4 at half the step, and the two codes differ from each other by
about as much as each differs from the exact result: their Trotter errors are of the
same size but not the same. At the published sizes the two correlations differ by up to
0.08 (N = 16) and 0.17 (N = 32, 48) by t = 7.5, unchanged when χ goes from 64 to 256 at
N = 16. The speed ratio is a fair one for equal step and order; a comparison at a given
accuracy needs the step chosen for it.

## Quick checks

Each runs its script end to end at a small size, in seconds unless noted; the
timings mean nothing at these sizes. They were run as listed, from an activated
TeNPy environment with `QX_BUILD_DIR` and `QX_DLL_DIR` set.

```bash
export BENCH_REPS=1 BENCH_COOLDOWN=0
BENCH_THREADS=1 python benchmarks/dmrg_su2_vs_tenpy.py --quick
BENCH_THREADS=1 python benchmarks/spectral_vs_tenpy.py      # the full grid: minutes
BENCH_THREADS=1 python benchmarks/tdvp_vs_tenpy.py          # minutes: TeNPy's TDVP is slow
BENCH_N=12 BENCH_PLAN=32:10 python benchmarks/spin1_tebd_vs_tenpy.py     # chi:steps
BENCH_N=8 BENCH_PLAN=32:10 python benchmarks/hubbard_tebd_vs_tenpy.py    # chi:steps
BENCH_N=16 BENCH_PLAN=32 python benchmarks/hubbard_dmrg_vs_tenpy.py
BENCH_LX=4 BENCH_PLAN=128 python benchmarks/hubbard_ladder_vs_tenpy.py
BENCH_LX=2 BENCH_PLAN=32 BENCH_SWEEPS=2 python benchmarks/hubbard_cylinder_vs_tenpy.py
BENCH_PLAN=12:32 BENCH_STEPS=5 python benchmarks/gpu_vs_tenpy_quench.py   # N:chi
BENCH_N=8 BENCH_PLAN=32 python benchmarks/hubbard_dmrg_vs_itensor.py     # minutes: Julia
```

TeNPy stops a ground-state run whose truncation error passes its `max_trunc_err`
(1e-4 by default), so the ladder's check uses a bond dimension that holds the small
ladder. The full runs leave out `BENCH_REPS` and `BENCH_COOLDOWN`: the published
minima came from repeated runs with cooldowns between them.
