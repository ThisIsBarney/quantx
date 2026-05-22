"""Cross-Entropy Benchmarking (XEB) — M82.

Implements the linear cross-entropy benchmarking protocol as used in
Google's quantum supremacy experiments and IBM's heavy-output benchmarks.

Linear XEB fidelity:
    F_XEB = d · E_x~P_noisy[P_ideal(x)] - 1

where d = 2**n_qubits, P_ideal is the ideal probability distribution from a
noiseless circuit, and x is sampled from P_noisy (the noisy circuit).
For an exact density-matrix simulation both distributions are known, so the
fidelity simplifies to:

    F_XEB = d · ∑_x P_ideal(x) · P_noisy(x) - 1

For noiseless circuits F_XEB ≈ 1; for fully depolarised circuits F_XEB ≈ 0.

Quick reference::

    from quantx_engine.xeb import CrossEntropyBenchmarking
    result = CrossEntropyBenchmarking(n_qubits=2, p_error=0.01).run(
        depths=[1, 2, 4, 8], n_trials=50
    )
    print(result.mean_fidelity)  # ~ exp(-p * d * depth) for small p

References
----------
.. [1] Arute, F. et al. (2019). Quantum supremacy using a programmable
       superconducting processor. *Nature* 574, 505–510.
       https://doi.org/10.1038/s41586-019-1666-5

.. [2] Boixo, S. et al. (2018). Characterizing quantum supremacy in
       near-term devices. *Nat. Phys.* 14, 595–600.
       https://doi.org/10.1038/s41567-018-0124-x

.. [3] Morvan, A. et al. (2024). Phase transitions in random circuit sampling.
       *Nature* 634, 328–333.
       https://doi.org/10.1038/s41586-024-07998-6

Physical validation
-------------------
XEB fidelity F_XEB = <P_ideal> * D - 1, where D = 2^n is the Hilbert-space
dimension and P_ideal is the ideal distribution probability [Arute2019].
For a perfect noiseless device F_XEB = 1; for a maximally mixed output F_XEB = 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

from .rb_qv import RBQVError, _require_numpy, _haar_unitary, _embed_gate

__all__ = [
    "XEBError",
    "XEBResult",
    "CrossEntropyBenchmarking",
]


class XEBError(ValueError):
    """Raised for invalid inputs to XEB experiments."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class XEBResult:
    """Result of a Cross-Entropy Benchmarking experiment.

    Attributes
    ----------
    depths:
        Circuit depths evaluated.
    xeb_fidelities:
        Mean linear XEB fidelity at each depth.
    mean_fidelity:
        Average of ``xeb_fidelities`` over all depths (summary scalar).
    n_qubits:
        Number of qubits.
    n_trials:
        Number of random-circuit trials used per depth.
    """
    depths: "list[int]"
    xeb_fidelities: "list[float]"
    mean_fidelity: float
    n_qubits: int
    n_trials: int


# ---------------------------------------------------------------------------
# CrossEntropyBenchmarking
# ---------------------------------------------------------------------------

class CrossEntropyBenchmarking:
    """Linear Cross-Entropy Benchmarking (XEB).

    Generates random layered circuits (Haar-random 2-qubit unitaries in a
    brick-work pattern, identical to the QuantumVolume circuit model) of
    varying depths, and estimates the linear XEB fidelity via exact
    density-matrix simulation.

    Parameters
    ----------
    n_qubits:
        Number of qubits.  Must be >= 2 (XEB requires at least one 2-qubit gate).
    p_error:
        Depolarizing error probability per 2-qubit gate.  0 = noiseless.
    seed:
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        p_error: float = 0.001,
        seed: "int | None" = None,
    ) -> None:
        _require_numpy()
        if n_qubits < 2:
            raise XEBError(f"n_qubits must be >= 2 for XEB, got {n_qubits}.")
        if not (0.0 <= p_error < 1.0):
            raise XEBError(f"p_error must be in [0, 1), got {p_error}.")
        self.n_qubits = n_qubits
        self.p_error = p_error
        self.seed = seed

    def run(
        self,
        depths: "list[int]",
        n_trials: int = 50,
    ) -> XEBResult:
        """Run the XEB experiment.

        Parameters
        ----------
        depths:
            List of circuit depths (number of random-unitary layers) to evaluate.
        n_trials:
            Number of random circuits to average over at each depth.

        Returns
        -------
        XEBResult
        """
        _require_numpy()
        depths = list(depths)
        if not depths:
            raise XEBError("depths must not be empty.")
        if any(d < 1 for d in depths):
            raise XEBError("All depths must be >= 1.")
        if n_trials < 1:
            raise XEBError("n_trials must be >= 1.")

        rng = np.random.default_rng(self.seed)
        n = self.n_qubits
        d = 1 << n
        xeb_fidelities: "list[float]" = []

        for depth in depths:
            fid_sum = 0.0
            for _ in range(n_trials):
                # Build noiseless and noisy density matrices in parallel
                rho_ideal = np.zeros((d, d), dtype=complex)
                rho_ideal[0, 0] = 1.0
                rho_noisy = rho_ideal.copy()

                for _layer in range(depth):
                    perm = rng.permutation(n)
                    for k in range(0, n - 1, 2):
                        q1, q2 = int(perm[k]), int(perm[k + 1])
                        U2 = _haar_unitary(4, rng)
                        V = _embed_gate(U2, n, (q1, q2))
                        rho_ideal = V @ rho_ideal @ V.conj().T
                        rho_noisy = V @ rho_noisy @ V.conj().T
                        if self.p_error > 0.0:
                            rho_noisy = (1.0 - self.p_error) * rho_noisy + (
                                self.p_error / d
                            ) * np.eye(d, dtype=complex)

                # Exact probabilities from diagonal of density matrices
                p_ideal = np.maximum(np.real(np.diag(rho_ideal)), 0.0)
                p_noisy = np.maximum(np.real(np.diag(rho_noisy)), 0.0)

                # Linear XEB fidelity: d * sum_x(P_ideal(x) * P_noisy(x)) - 1
                fid_sum += float(d * np.dot(p_ideal, p_noisy) - 1.0)

            xeb_fidelities.append(fid_sum / n_trials)

        return XEBResult(
            depths=depths,
            xeb_fidelities=xeb_fidelities,
            mean_fidelity=float(np.mean(xeb_fidelities)),
            n_qubits=n,
            n_trials=n_trials,
        )
