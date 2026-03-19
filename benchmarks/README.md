# Benchmark Harness

Current suites:

1. `circuit_accuracy_smoke`
2. `circuit_noise_sweep`
3. `continuous2d_smoke`
4. `continuous3d_smoke`
5. `continuous3d_accuracy`

Recommended entrypoints:

```powershell
.\scripts\benchmark_all.ps1 -NoiseSweep
.\scripts\benchmark_all.ps1 -NoiseSweep -RunGates -EnforceGate
```

The unified benchmark flow writes reports under `benchmarks/reports/`:

- suite JSON report: `*.json`
- suite markdown summary: `*.md`
- cumulative index: `INDEX.md`
- trend report: `*_TREND.md`

All runs should pin random seeds and include hardware/software versions. Current report metadata includes `engine_version` for reproducibility and regression triage.
