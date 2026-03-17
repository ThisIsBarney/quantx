# Benchmark Harness (Scaffold)

Planned benchmark suites:

1. Infinite well wave packet evolution (analytic cross-check)
2. Two-level Rabi oscillation (analytic cross-check)
3. Small Ising chain dynamics (cross-check against reference simulators)

Expected output format:

- `reports/<date>/accuracy.json`
- `reports/<date>/performance.json`
- `reports/<date>/environment.json`

All runs should pin random seeds and include hardware/software versions.
