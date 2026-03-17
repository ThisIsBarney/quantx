# ML/LLM Integration Plan

This document captures how ML and LLM capabilities will be used without weakening numerical correctness.

## Priority

- phase priority: P2
- non-negotiable: ML policies are advisory unless explicitly promoted by validated gates

## P2.1 Learned scheduler (backend selection)

- input features:
  - qubits
  - operation count / Hamiltonian term count
  - estimated entanglement growth proxy
  - target error and resource constraints
- output:
  - backend recommendation (`statevector`, `mps`, future backends)
  - confidence score
- safety:
  - if confidence is low, fallback to heuristic policy
  - always emit explanation metadata

## P2.2 Adaptive evolution control

- objective:
  - reduce runtime while maintaining target error
- controls:
  - timestep recommendation (`dt`)
  - optional method hint (`trotter`, `krylov`)
- safety:
  - enforce hard error bounds and rollback if drift is detected

## P2.3 LLM workflow assistant

- generate initial `problem.json` / `circuit.json` templates from natural language
- suggest benchmark sets and summarize run results
- produce reproducible experiment plans with explicit seeds and version info

## Guardrails

- deterministic mode always available
- all ML decisions logged with feature snapshot and chosen action
- benchmark gates required before ML policy becomes default
