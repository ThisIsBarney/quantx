# quantx-engine

Python SDK for the [QuantX](https://github.com/ThisIsBarney/quantx) quantum simulation engine — 37 physics solvers exposed via a ctypes C ABI bridge.

## Installation

```bash
pip install quantx-engine
```

> **Note:** The package requires the compiled `qx` shared library. On Windows, place `qx.dll` (built from source) alongside the package or in your `PATH`. See the [main repository](https://github.com/ThisIsBarney/quantx) for build instructions.

## Quick Start

```python
from quantx_engine import Engine

engine = Engine()

# List all available solvers
solvers = engine.list_solvers()
print(f"{len(solvers)} solvers available")

# Run VQE on a custom Hamiltonian
result = engine.run_vqe({
    "cost_hamiltonian": [
        {"label": "Z0", "coefficient": -0.5},
        {"label": "Z1", "coefficient": -0.5},
        {"label": "Z0Z1", "coefficient": 0.25},
    ],
    "ansatz_type": "uccsd",
    "max_iterations": 200,
})
print(result)
```

## Quantum Chemistry

```python
from quantx_engine.chemistry import MolecularVQE

# Build H₂ at equilibrium geometry
problem = MolecularVQE.h2(bond_length=0.735)
print(f"Hamiltonian: {problem.hamiltonian}")
print(f"FCI reference energy: {problem.fci_energy:.6f} Hartree")

# Get VQE options compatible with Engine.run_vqe()
opts = problem.to_vqe_options(ansatz="uccsd", max_iterations=300)
result = engine.run_vqe(opts)
```

Supported molecules: **H₂** (4 qubits), **LiH** (10 qubits), **BeH₂** (12 qubits).

## OpenQASM 3.0

```python
from quantx_engine.qasm import parse_qasm_to_circuit

circuit = parse_qasm_to_circuit("""
OPENQASM 3.0;
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;
""")
result = engine.run_circuit(circuit)
```

## Noise Models

```python
from quantx_engine.noise import DepolarizingNoise, ThermalRelaxationNoise, NoiseModel

model = NoiseModel()
model.add(DepolarizingNoise(prob=0.01))
model.add(ThermalRelaxationNoise(T1=50e-6, T2=30e-6, t_gate=50e-9))

circuit = {"gates": [...]}
circuit.update(model.to_dict())
result = engine.run_circuit(circuit)
```

## IBM Quantum Connector

```python
from quantx_engine.ibm import IBMQuantumBackend

backend = IBMQuantumBackend(token="YOUR_IBM_TOKEN", backend_name="ibm_brisbane")
result = backend.run(circuit, shots=1024)
```

## Async API

All 37 solver methods have async variants:

```python
import asyncio
from quantx_engine import Engine

async def main():
    engine = Engine()
    result = await engine.run_vqe_async({"cost_hamiltonian": [...], "ansatz_type": "uccsd"})
    print(result)

asyncio.run(main())
```

## Available Solvers

`vqe`, `qpe`, `qec`, `qfield`, `phi4`, `hubbard`, `floquet`, `heisenberg`,
`bose_hubbard`, `tfi`, `ssh`, `kitaev`, `anderson`, `chern`, `hofstadter`,
`mbl`, `spectral`, `tenfold`, `second_chern`, `wilson_loop`, `edge_states`,
`haldane`, `weyl`, `fu_kane`, `lutchyn_oreg`, `kitaev_honeycomb`, `bbh`,
`floquet_topo`, `non_herm`, `aah`, `syk`, `kondo`, `luttinger`, `toric`,
`mipt`, `rg`, `entanglement`

## License

MIT — see [LICENSE](../LICENSE).
