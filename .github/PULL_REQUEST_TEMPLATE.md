## Summary

<!-- What this PR changes and why, in a paragraph. Link the issue: Closes #NNN -->

## Changes

<!-- The key changes: new or changed APIs, files, behaviour. -->

- 

## Test plan

<!-- What you ran, and what it showed. See "The verification standard" in CONTRIBUTING.md. -->

- [ ] Python tests pass: `pytest python/quantx_engine/tests -q`
- [ ] Engine change: a gate compares it against an independent reference (exact
      diagonalisation, a state vector or an analytic result) at a stated tolerance, and it
      passes with `QX_REQUIRE_NATIVE=1` and the library loaded
- [ ] C++ change: `ctest --test-dir build --output-on-failure` passes in the default
      configuration (no BLAS, no CUDA), which the wheels use
- [ ] `CHANGELOG.md` has a line under `[Unreleased]` for anything a user would notice
- [ ] Docstrings and docs state only what a test checks

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change (changes existing behaviour or the C API)
- [ ] Documentation
- [ ] Performance or refactor, with no behaviour change

## Checklist

- [ ] The branch is up to date with `master`
- [ ] Each commit is signed off (`git commit -s`, DCO)
