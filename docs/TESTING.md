# Testing Notes

## Automated smoke coverage

- backend routing:
  - small qubit case -> `statevector`
  - large qubit case -> `mps`
- statevector physics sanity:
  - 1-qubit Rabi drive with `H = X0` at `t = pi/2`
    - `trotter` path reaches `population_1 ~= 1`
    - `krylov` path reaches `population_1 ~= 1`
  - 2-qubit XX coupling with `H = X0X1` at `t = pi/2`
    - reaches `population_last ~= 1` (`|11>`)
- norm conservation:
  - `|psi|^2` stays near 1.0 in smoke scenarios

## How to run

```powershell
.\scripts\test.ps1
```
