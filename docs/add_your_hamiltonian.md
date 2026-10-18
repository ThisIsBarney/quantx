# Add Your Own Hamiltonian in 30 Minutes

This guide walks you through adding a new physics solver to QuantX from scratch —
a Python implementation, a literature-validation test, and optional C++ acceleration.

Estimated time: **20–30 minutes** for a simple Hamiltonian.

---

## Prerequisites

- Python ≥ 3.11, `git`, a GitHub account
- QuantX dev environment set up (see [CONTRIBUTING.md](../CONTRIBUTING.md))

```bash
git clone https://github.com/yourusername/quantx.git
cd quantx
pip install -e "python/[dev]"
```

---

## Step 1 — Choose your Hamiltonian

Pick something with a known analytic result so the test can validate against
the literature.  Good first choices:

| Hamiltonian | Known result | Complexity |
|-------------|-------------|------------|
| 1D XX chain | `E₀ = -J/π` per site (Bethe ansatz) | Easy |
| 1D TFIM at critical point | `E₀/N → -2J/π` | Easy |
| Kitaev honeycomb (toric code limit) | Exact topological gap | Medium |

We will use the **1D XX chain** as an example.

---

## Step 2 — Implement the solver

Create `python/quantx_engine/xx_chain.py`:

```python
"""XX spin chain solver.

The Hamiltonian is

    H = J Σ_i (Sˣ_i Sˣ_{i+1} + Sʸ_i Sʸ_{i+1})
      = (J/2) Σ_i (S⁺_i S⁻_{i+1} + S⁻_i S⁺_{i+1})

References
----------
.. [1] Lieb, Schultz, Mattis (1961). Ann. Phys. 16, 407.
       DOI: 10.1016/0003-4916(61)90115-4
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from quantx_engine.contrib.lanczos import LanczosSolver


def xx_hamiltonian(n_sites: int, J: float = 1.0) -> sp.csr_matrix:
    """Build the sparse XX Hamiltonian for an open chain of *n_sites* spins."""
    dim = 2 ** n_sites
    H = sp.lil_matrix((dim, dim), dtype=complex)

    for i in range(n_sites - 1):
        # S+_i S-_{i+1}  +  S-_i S+_{i+1}
        for state in range(dim):
            bit_i   = (state >> i)       & 1
            bit_ip1 = (state >> (i + 1)) & 1
            if bit_i == 1 and bit_ip1 == 0:
                flipped = state ^ (1 << i) ^ (1 << (i + 1))
                H[flipped, state] += J / 2
            if bit_i == 0 and bit_ip1 == 1:
                flipped = state ^ (1 << i) ^ (1 << (i + 1))
                H[flipped, state] += J / 2

    return H.tocsr()


class XXChainSolver:
    """Ground-state solver for the 1D XX spin chain."""

    def __init__(self, J: float = 1.0):
        self.J = J

    def solve(self, n_sites: int) -> dict:
        """Return ground-state energy and vector for an open XX chain."""
        H = xx_hamiltonian(n_sites, self.J)
        solver = LanczosSolver(k=1, max_iter=500, tol=1e-10)
        result = solver.solve(H)
        return {
            "ground_energy": float(result.eigenvalues[0]),
            "ground_state":  result.eigenvectors[:, 0],
            "n_sites":       n_sites,
        }
```

---

## Step 3 — Write a literature-validation test

Create `python/quantx_engine/tests/test_literature_xx_chain.py`:

```python
"""M_XX — Literature validation: 1D XX chain ground-state energy.

Reference: Lieb, Schultz, Mattis (1961). Ann. Phys. 16, 407.
DOI: 10.1016/0003-4916(61)90115-4

The exact ground-state energy per site for an infinite open XX chain is
    E₀/N = -J/π ≈ -0.3183 J
We verify that our finite-chain Lanczos result converges to this value.
"""
import pytest
import numpy as np


def test_xx_chain_ground_energy_n8():
    """Lanczos energy for N=8 XX chain is within 2% of the Bethe-ansatz limit."""
    from quantx_engine.xx_chain import XXChainSolver

    solver = XXChainSolver(J=1.0)
    result = solver.solve(n_sites=8)

    exact_per_site = -1.0 / np.pi          # Bethe ansatz, infinite chain
    numerical_per_site = result["ground_energy"] / 8

    assert abs(numerical_per_site - exact_per_site) < 0.02, (
        f"Energy per site {numerical_per_site:.4f} too far from "
        f"exact {exact_per_site:.4f}"
    )


def test_xx_chain_ground_state_normalised():
    """Ground-state vector returned by XXChainSolver has unit norm."""
    from quantx_engine.xx_chain import XXChainSolver

    result = XXChainSolver().solve(n_sites=6)
    norm = np.linalg.norm(result["ground_state"])
    assert abs(norm - 1.0) < 1e-10


def test_xx_hamiltonian_hermitian():
    """The XX Hamiltonian matrix is Hermitian."""
    from quantx_engine.xx_chain import xx_hamiltonian
    import scipy.sparse as sp

    H = xx_hamiltonian(n_sites=4)
    diff = H - H.conj().T
    assert sp.linalg.norm(diff) < 1e-12
```

Run the tests:

```bash
pytest python/quantx_engine/tests/test_literature_xx_chain.py -v
```

---

## Step 4 — Expose the solver in the public API

Add one line to `python/quantx_engine/__init__.py`:

```python
from quantx_engine.xx_chain import XXChainSolver, xx_hamiltonian
```

---

## Step 5 — Open a pull request

```bash
git checkout -b feat/xx-chain-solver
git add python/quantx_engine/xx_chain.py \
        python/quantx_engine/tests/test_literature_xx_chain.py \
        python/quantx_engine/__init__.py
git commit -m "feat: add 1D XX chain solver with Lanczos ground state"
git push -u origin feat/xx-chain-solver
```

Then open a PR on GitHub. The CI pipeline will run your tests automatically.

---

## Optional — C++ acceleration

If your Hamiltonian is performance-critical, you can add a native backend.
See [CONTRIBUTING.md § C++ backend](../CONTRIBUTING.md#building-the-native-engine)
for the full build instructions.  The pattern is:

1. Add a `qx_xx_chain_solve()` function in `src/solvers/xx_chain.cpp`
2. Register it in `src/registry.cpp`
3. Call it from Python via `quantx_engine.engine.Engine().run_solver("xx_chain", params)`

---

## Checklist

Before submitting your PR, verify:

- [ ] Solver file has a module docstring with a `References` section citing the paper DOI
- [ ] Test file name starts with `test_literature_` and cites the same paper
- [ ] At least one test reproduces a specific numerical result from the paper
- [ ] `pytest python/quantx_engine/tests/ -q` passes with no new failures
- [ ] Public symbols are added to `__init__.py`

That's it — welcome to the QuantX community!
