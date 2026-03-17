# Third-Party Dependencies and Reuse Policy

## Reuse policy

- Prefer mature open-source implementations over custom reimplementation when:
- algorithm correctness risk is non-trivial
- maintenance cost of self-built code is high
- license is business-friendly and compatible with project distribution
- Keep dependencies pinned by version in build logic.
- Record license type and usage scope for each dependency.
- Do not introduce copyleft dependencies that would force relicensing of the core without explicit approval.

## Current dependencies

1. `nlohmann/json` (`v3.11.3`)
- License: MIT
- Purpose: JSON parsing/serialization for problem/options/results C++ and C API boundary.

2. `Eigen` (`3.4.0`)
- License: MPL-2.0
- Purpose: robust linear algebra primitives in Krylov path (matrix exponential for reduced Hessenberg system).

## Selection rationale (current iteration)

- Replaced custom JSON parser with `nlohmann/json` to reduce parsing edge-case bugs.
- Replaced custom small-matrix exponential code with `Eigen` `MatrixFunctions` to reduce numerical maintenance burden.
