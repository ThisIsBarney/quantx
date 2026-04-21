"""Quantum State and Process Tomography — M83.

Provides pure-NumPy implementations of:

- :class:`StateTomography`: reconstruct a density matrix from ideal Pauli
  expectation values via linear inversion; compute state fidelity.
- :class:`ProcessTomography`: compute the Pauli-basis chi (χ) process matrix
  for an n-qubit unitary, compute average gate fidelity and process fidelity.

No classical optimisation or maximum-likelihood estimation is performed —
all reconstruction uses exact linear inversion.  This makes the module
suitable for exact simulation workflows where shot noise is absent.

Quick reference::

    from quantx_engine.tomography import StateTomography, ProcessTomography
    import numpy as np

    # State tomography
    rho = np.array([[0.9, 0.05], [0.05, 0.1]], dtype=complex)
    st = StateTomography(n_qubits=1)
    exp_vals = st.pauli_expectations(rho)       # {'X': ..., 'Y': ..., 'Z': ...}
    rho_rec = st.reconstruct_linear(exp_vals)   # recovered density matrix
    print(st.state_fidelity(rho, rho_rec))      # ~1.0

    # Process tomography
    pt = ProcessTomography(n_qubits=1)
    chi = pt.chi_matrix(np.array([[0,1],[1,0]], dtype=complex))  # X gate
    print(pt.average_gate_fidelity(chi, chi))   # 1.0
"""
from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

__all__ = [
    "TomographyError",
    "StateTomography",
    "ProcessTomography",
]


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class TomographyError(ValueError):
    """Raised for invalid inputs to tomography routines."""


def _require_numpy() -> None:
    if not _HAS_NUMPY:  # pragma: no cover
        raise TomographyError("NumPy is required for tomography routines.")


# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------

_I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

_PAULIS_1Q: "dict[str, np.ndarray]" = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def _kron_n(matrices: "list[np.ndarray]") -> "np.ndarray":
    """Tensor product of a list of matrices (left to right)."""
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def _n_qubit_paulis(n: int) -> "dict[str, np.ndarray]":
    """Return all 4^n Pauli tensor products for n qubits.

    Returns a dict mapping label string (e.g. 'XZI') to the 2^n × 2^n matrix.
    """
    labels = list(itertools.product("IXYZ", repeat=n))
    paulis: "dict[str, np.ndarray]" = {}
    for label_tuple in labels:
        label = "".join(label_tuple)
        paulis[label] = _kron_n([_PAULIS_1Q[c] for c in label_tuple])
    return paulis


# ---------------------------------------------------------------------------
# StateTomography
# ---------------------------------------------------------------------------

class StateTomography:
    """Quantum state tomography via Pauli expectations.

    Uses the identity::

        ρ = (1/d) ∑_{P ∈ {I,X,Y,Z}^⊗n}  Tr(P ρ) · P

    to reconstruct a density matrix from its Pauli expectation values.
    Reconstruction is exact for noiseless (ideal) expectation values.

    Parameters
    ----------
    n_qubits:
        Number of qubits.
    """

    def __init__(self, n_qubits: int = 1) -> None:
        _require_numpy()
        if n_qubits < 1:
            raise TomographyError(f"n_qubits must be >= 1, got {n_qubits}.")
        self.n_qubits = n_qubits
        self._d = 1 << n_qubits
        self._paulis: "dict[str, np.ndarray] | None" = None  # lazy init

    def _get_paulis(self) -> "dict[str, np.ndarray]":
        if self._paulis is None:
            self._paulis = _n_qubit_paulis(self.n_qubits)
        return self._paulis

    def pauli_expectations(
        self,
        rho: "np.ndarray",
    ) -> "dict[str, float]":
        """Compute all Pauli expectation values Tr(P ρ) for ρ.

        Parameters
        ----------
        rho:
            Density matrix of shape ``(d, d)`` where d = 2**n_qubits.

        Returns
        -------
        dict mapping Pauli label to real expectation value.
        """
        _require_numpy()
        rho = np.asarray(rho, dtype=complex)
        if rho.shape != (self._d, self._d):
            raise TomographyError(
                f"rho must have shape ({self._d}, {self._d}), got {rho.shape}."
            )
        paulis = self._get_paulis()
        return {
            label: float(np.real(np.trace(P @ rho)))
            for label, P in paulis.items()
        }

    def reconstruct_linear(
        self,
        expectations: "dict[str, float]",
    ) -> "np.ndarray":
        """Reconstruct density matrix from Pauli expectations via linear inversion.

        Parameters
        ----------
        expectations:
            Dict mapping Pauli label to expectation value, as returned by
            :meth:`pauli_expectations`.

        Returns
        -------
        Reconstructed density matrix (Hermitian, trace-1, not necessarily PSD
        for noisy input expectations).
        """
        _require_numpy()
        paulis = self._get_paulis()
        d = self._d
        rho = np.zeros((d, d), dtype=complex)
        for label, P in paulis.items():
            exp_val = expectations.get(label, 0.0)
            rho += exp_val * P
        rho /= d
        return rho

    @staticmethod
    def state_fidelity(rho1: "np.ndarray", rho2: "np.ndarray") -> float:
        """Compute state fidelity F(ρ₁, ρ₂) = Tr(√(√ρ₁ ρ₂ √ρ₁))².

        For pure states |ψ⟩ this simplifies to F = ⟨ψ₁|ρ₂|ψ₁⟩.

        Parameters
        ----------
        rho1, rho2:
            Density matrices of the same shape.

        Returns
        -------
        Fidelity in [0, 1].
        """
        _require_numpy()
        rho1 = np.asarray(rho1, dtype=complex)
        rho2 = np.asarray(rho2, dtype=complex)
        if rho1.shape != rho2.shape:
            raise TomographyError(
                f"rho1 and rho2 must have the same shape, "
                f"got {rho1.shape} vs {rho2.shape}."
            )
        # sqrt(rho1) via eigendecomposition
        evals1, evecs1 = np.linalg.eigh(rho1)
        evals1 = np.maximum(evals1, 0.0)  # clamp numerical noise
        sqrt_rho1 = evecs1 @ np.diag(np.sqrt(evals1)) @ evecs1.conj().T
        M = sqrt_rho1 @ rho2 @ sqrt_rho1
        # Trace of sqrt(M)
        evals_m = np.linalg.eigvalsh(M)
        evals_m = np.maximum(evals_m, 0.0)
        fidelity = float(np.sum(np.sqrt(evals_m))) ** 2
        return min(1.0, max(0.0, fidelity))  # clamp to [0, 1]


# ---------------------------------------------------------------------------
# ProcessTomography
# ---------------------------------------------------------------------------

class ProcessTomography:
    """Quantum process tomography via the chi (χ) process matrix.

    The chi matrix in the Pauli basis is defined by::

        E(ρ) = ∑_{m,n}  χ_{mn}  P_m  ρ  P_n†

    For a unitary channel U·ρ·U†, the chi matrix has rank 1.

    Parameters
    ----------
    n_qubits:
        Number of qubits.
    """

    def __init__(self, n_qubits: int = 1) -> None:
        _require_numpy()
        if n_qubits < 1:
            raise TomographyError(f"n_qubits must be >= 1, got {n_qubits}.")
        self.n_qubits = n_qubits
        self._d = 1 << n_qubits
        self._paulis_list: "list[tuple[str, np.ndarray]] | None" = None

    def _get_paulis_list(self) -> "list[tuple[str, np.ndarray]]":
        if self._paulis_list is None:
            paulis = _n_qubit_paulis(self.n_qubits)
            # Ordered list for consistent indexing (I always first)
            ordered = sorted(paulis.items())
            self._paulis_list = ordered
        return self._paulis_list

    def chi_matrix(self, unitary: "np.ndarray") -> "np.ndarray":
        """Compute the Pauli-basis chi matrix for a unitary channel U·ρ·U†.

        Parameters
        ----------
        unitary:
            Unitary matrix of shape ``(d, d)`` where d = 2**n_qubits.

        Returns
        -------
        Chi matrix of shape ``(4**n_qubits, 4**n_qubits)``.
        """
        _require_numpy()
        U = np.asarray(unitary, dtype=complex)
        d = self._d
        if U.shape != (d, d):
            raise TomographyError(
                f"unitary must have shape ({d}, {d}), got {U.shape}."
            )
        paulis = self._get_paulis_list()
        n_ops = len(paulis)  # 4**n_qubits
        chi = np.zeros((n_ops, n_ops), dtype=complex)

        # chi_{mn} = (1/d^2) Tr(P_m† U) Tr(U† P_n)^* ... use direct formula:
        # For unitary channel: chi_{mn} = (1/d) * Tr(P_m U P_n† U†) ... no.
        # Correct formula: chi_{mn} = (1/d²) ∑_{i,j} <i|P_m|j> U_{ij} * conj(<i|P_n|j> U_{ij})
        # Simplest correct approach: project U onto Pauli basis.
        # E(ρ) = U ρ U†  =>  χ_mn = (1/d) Tr(P_m U) * conj((1/d) Tr(P_n U))
        # ... This is only valid for 1-qubit.
        # General: E(ρ) = ∑_mn χ_mn P_m ρ P_n†
        # => χ_mn = (1/d²) Tr(P_m U) Tr(P_n U)†  is WRONG for multi-qubit.
        #
        # Correct general formula (Choi-Kraus):
        # χ_mn = (1/d) Tr(P_m · U · P_n† · U†)
        # because the Kraus operator for a unitary is just K_0 = U,
        # and χ_mn = ∑_k (1/d) Tr(P_m K_k) Tr(P_n K_k)†
        #          = (1/d) Tr(P_m U) * conj(Tr(P_n U))
        #
        # For n-qubit unitary with a *single* Kraus operator:
        # The Pauli-basis representation is:
        #   U = ∑_k  u_k P_k / sqrt(d)  where u_k = Tr(P_k† U) / sqrt(d) ... hmm
        #
        # Let me use the standard textbook definition carefully:
        # Decompose U in the Pauli basis: U = ∑_m c_m P_m
        # where c_m = Tr(P_m† U) / d = Tr(P_m U) / d  (since P_m is Hermitian)
        #
        # Then E(ρ) = Uρ U† = (∑_m c_m P_m) ρ (∑_n c_n* P_n†)
        #           = ∑_{mn} c_m c_n* P_m ρ P_n
        # So χ_mn = c_m * c_n* = Tr(P_m U)/d * conj(Tr(P_n U)/d)
        #
        # This IS the correct formula for a unitary channel with single Kraus.

        # Pauli expansion: U = ∑_m c_m P_m where c_m = Tr(P_m U) / d.
        # For a unitary channel E(ρ) = UρU†: χ_mn = c_m c_n*.
        # With this normalisation ∑_m|c_m|² = 1, so Tr(χ χ†) = 1 and
        # process_fidelity = Tr(χ₁ χ₂†) lies in [0, 1] with 1 for identical unitaries.
        coeffs = np.array(
            [np.trace(P @ U) / d for _, P in paulis],
            dtype=complex,
        )
        chi = np.outer(coeffs, coeffs.conj())
        return chi

    def process_fidelity(
        self,
        chi1: "np.ndarray",
        chi2: "np.ndarray",
    ) -> float:
        """Compute process fidelity F_process(E1, E2) = Tr(χ1 χ2†).

        With the 1/d Pauli normalisation used by :meth:`chi_matrix` this
        equals |⟨c1|c2⟩|² ∈ [0, 1] with 1 when the two unitaries are equal.

        Parameters
        ----------
        chi1, chi2:
            Chi matrices of the same shape, as returned by :meth:`chi_matrix`.

        Returns
        -------
        Process fidelity in [0, 1].
        """
        _require_numpy()
        chi1 = np.asarray(chi1, dtype=complex)
        chi2 = np.asarray(chi2, dtype=complex)
        if chi1.shape != chi2.shape:
            raise TomographyError(
                f"chi1 and chi2 must have the same shape, "
                f"got {chi1.shape} vs {chi2.shape}."
            )
        fid = float(np.real(np.trace(chi1 @ chi2.conj().T)))
        return min(1.0, max(0.0, fid))

    def average_gate_fidelity(
        self,
        chi_actual: "np.ndarray",
        chi_target: "np.ndarray",
    ) -> float:
        """Compute average gate fidelity from chi matrices.

        Uses the relation::

            F_avg = (d · F_process + 1) / (d + 1)

        where d = 2**n_qubits.

        Parameters
        ----------
        chi_actual:
            Chi matrix of the implemented gate.
        chi_target:
            Chi matrix of the target (ideal) gate.

        Returns
        -------
        Average gate fidelity in [0, 1].
        """
        _require_numpy()
        d = self._d
        f_proc = self.process_fidelity(chi_actual, chi_target)
        f_avg = (d * f_proc + 1.0) / (d + 1.0)
        return min(1.0, max(0.0, f_avg))
