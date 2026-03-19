# Third-Party Dependencies and Reuse Policy

## Reuse policy

- Prefer mature open-source implementations over custom reimplementation when:
- algorithm correctness risk is non-trivial
- maintenance cost of self-built code is high
- license is business-friendly and compatible with project distribution
- Keep dependencies pinned by version in build logic.
- Record license type and usage scope for each dependency.
- Default allow-list is permissive licenses suitable for closed-source commercial distribution (`MIT`, `BSD`, `Apache-2.0`, `Zlib`, `BSL`).
- Any weak/strong copyleft dependency requires explicit review and approval before adoption.

## Current dependencies

1. `nlohmann/json` (`v3.11.3`)
- License: MIT
- Purpose: JSON parsing/serialization for problem/options/results C++ and C API boundary.

2. `Eigen` (`3.4.0`)
- License: MPL-2.0
- Purpose: robust linear algebra primitives in Krylov path (matrix exponential for reduced Hessenberg system).

3. `spdlog` (`v1.14.1`)
- License: MIT
- Purpose: unified runtime/CLI logging with configurable verbosity.

4. `CLI11` (`v2.4.2`)
- License: BSD-3-Clause
- Purpose: robust command-line parsing for `qx_cli`, replacing custom argument parsing.

5. `NumPy` (runtime, optional for 2D visualization)
- License: BSD-3-Clause
- Purpose: numerical kernel for 2D split-step Fourier spatial wavepacket visualization.

## Optional local toolchains (not redistributed)

- `CUDA Toolkit` / `cuFFT` may be installed locally to enable optional acceleration (`QX_ENABLE_CUDA=ON`).
- This toolkit is not bundled or redistributed by this repository and is outside the dependency-policy allow-list checks.

## Selection rationale (current iteration)

- Replaced custom JSON parser with `nlohmann/json` to reduce parsing edge-case bugs.
- Replaced custom small-matrix exponential code with `Eigen` `MatrixFunctions` to reduce numerical maintenance burden.
- Added `spdlog` for production-grade logging instead of custom logger implementation.
- Replaced manual CLI parsing with `CLI11` to reduce argument-handling bugs and improve maintainability.

## CI enforcement

- workflow: `.github/workflows/benchmark-gate.yml`
- gates include:
  - CLI argument contract checks
  - dependency license policy checks
