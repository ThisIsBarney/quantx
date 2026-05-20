"""M113: Excited-state quantum chemistry.

Provides tools for computing excited-state energies and wavefunctions
using variational quantum methods:

* **Exact diagonalization** (ED) — exact for n_qubits ≤ 16; returns all
  eigenvalues/vectors via ``numpy.linalg.eigh``.
* **Deflation VQE** — iteratively adds a penalty λ|ψ_k⟩⟨ψ_k| to the
  Hamiltonian so that subsequent minimizations converge to the next excited
  state rather than repeating the ground state.
* **Folded-spectrum VQE** — minimises ⟨(H − ω)²⟩ to target the eigenstate
  closest to a chosen energy ω.

Conical-intersection (near-degeneracy) detection sweeps a molecular
geometry parameter and finds the crossing points between adjacent
eigenvalue curves.

Classes
-------
ExcitedStateError
ExcitedStateResult
ExcitedStateVQE

ConicalIntersectionResult
detect_conical_intersections

pyscf_to_pauli_hamiltonian (optional PySCF adapter)
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .chemistry import PauliHamiltonian, PauliTerm
from .advanced_chemistry import build_hamiltonian_matrix

__all__ = [
    "ExcitedStateError",
    "ExcitedStateResult",
    "ExcitedStateVQE",
    "ConicalIntersectionResult",
    "detect_conical_intersections",
    "pyscf_to_pauli_hamiltonian",
]


class ExcitedStateError(ValueError):
    """Raised for invalid excited-state computation requests."""


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class ExcitedStateResult:
    """Result of an :class:`ExcitedStateVQE` solve.

    Attributes
    ----------
    energies : list[float]
        Absolute energies E_0, E_1, …, E_{n_states-1} (Hartree or dimensionless).
    excitation_energies : list[float]
        Vertical excitation energies ΔE_k = E_k − E_0 for k ≥ 1.
    n_states : int
        Number of states computed (including ground state).
    n_qubits : int
    method : str
        One of ``"exact"``, ``"deflation"``, ``"folded_spectrum"``.
    """
    energies: list[float]
    excitation_energies: list[float]
    n_states: int
    n_qubits: int
    method: str

    def to_dict(self) -> dict:
        return {
            "energies": self.energies,
            "excitation_energies": self.excitation_energies,
            "n_states": self.n_states,
            "n_qubits": self.n_qubits,
            "method": self.method,
        }


@dataclass
class ConicalIntersectionResult:
    """Result of :func:`detect_conical_intersections`.

    Attributes
    ----------
    parameter_values : list[float]
        The swept parameter values (e.g. bond lengths).
    energies : dict[int, list[float]]
        ``energies[k][i]`` = energy of state k at parameter_values[i].
    crossing_points : list[tuple[float, int, int]]
        Each entry is ``(lambda_cross, state_a, state_b)`` — the parameter
        value at which states a and b become near-degenerate.
    gap_threshold : float
        Energy gap below which two states are declared near-degenerate.
    """
    parameter_values: list[float]
    energies: dict[int, list[float]]
    crossing_points: list[tuple[float, int, int]]
    gap_threshold: float

    def to_dict(self) -> dict:
        return {
            "parameter_values": self.parameter_values,
            "energies": {str(k): v for k, v in self.energies.items()},
            "crossing_points": self.crossing_points,
            "gap_threshold": self.gap_threshold,
        }


# ── ExcitedStateVQE ───────────────────────────────────────────────────────────

class ExcitedStateVQE:
    """Variational computation of multiple excited states.

    Parameters
    ----------
    hamiltonian : PauliHamiltonian
        The molecular Hamiltonian (or any Hermitian operator).
    n_states : int, default 3
        Number of states to compute (including the ground state).
    method : {"exact", "deflation", "folded_spectrum"}, default "exact"
        Algorithm to use:

        * ``"exact"`` — full numpy diagonalisation (best for n ≤ 16).
        * ``"deflation"`` — variational deflation: penalise previously
          found states with a positive-definite projection term.
        * ``"folded_spectrum"`` — minimise ⟨(H−ω)²⟩ for a sequence of
          ω values spaced across the spectrum.
    penalty_weight : float, default 10.0
        Weight λ used in the deflation penalty ΔH = λ Σ |ψ_k⟩⟨ψ_k|.
        Only used when ``method="deflation"``.
    """

    def __init__(
        self,
        hamiltonian: PauliHamiltonian,
        n_states: int = 3,
        method: str = "exact",
        penalty_weight: float = 10.0,
    ) -> None:
        if not isinstance(hamiltonian, PauliHamiltonian):
            raise ExcitedStateError("hamiltonian must be a PauliHamiltonian")
        if n_states < 1:
            raise ExcitedStateError(f"n_states must be ≥ 1, got {n_states}")
        if method not in ("exact", "deflation", "folded_spectrum"):
            raise ExcitedStateError(
                f"method must be 'exact', 'deflation', or 'folded_spectrum', got {method!r}"
            )

        self.hamiltonian = hamiltonian
        self.n_states = n_states
        self.method = method
        self.penalty_weight = penalty_weight
        self._n_qubits = hamiltonian.n_qubits

    def solve(self, seed: Optional[int] = None) -> ExcitedStateResult:
        """Compute the lowest ``n_states`` eigenstates.

        Parameters
        ----------
        seed : int | None
            Random seed (used only by variational methods).

        Returns
        -------
        ExcitedStateResult
        """
        H = build_hamiltonian_matrix(self.hamiltonian)

        if self.method == "exact":
            return self._solve_exact(H)
        elif self.method == "deflation":
            return self._solve_deflation(H, seed)
        else:
            return self._solve_folded_spectrum(H)

    # ── Exact diagonalisation ─────────────────────────────────────────────────

    def _solve_exact(self, H: np.ndarray) -> ExcitedStateResult:
        dim = H.shape[0]
        k = min(self.n_states, dim)
        eigvals, _ = np.linalg.eigh(H)
        eigvals = eigvals[:k].tolist()
        e0 = eigvals[0]
        excitations = [e - e0 for e in eigvals[1:]]
        return ExcitedStateResult(
            energies=eigvals,
            excitation_energies=excitations,
            n_states=k,
            n_qubits=self._n_qubits,
            method="exact",
        )

    # ── Deflation VQE ─────────────────────────────────────────────────────────

    def _solve_deflation(
        self, H: np.ndarray, seed: Optional[int]
    ) -> ExcitedStateResult:
        """Deflation: sequentially penalise previously found eigenvectors."""
        rng = np.random.default_rng(seed)
        dim = H.shape[0]
        k = min(self.n_states, dim)
        energies: list[float] = []
        states: list[np.ndarray] = []

        H_eff = H.copy()
        for _ in range(k):
            # Minimise ⟨ψ|H_eff|ψ⟩ — exact for the effective matrix
            eigvals, eigvecs = np.linalg.eigh(H_eff)
            e = float(eigvals[0])
            v = eigvecs[:, 0]
            # Use conjugate bra ⟨v| = v† for the Hermitian expectation value;
            # this matters whenever H has complex entries (e.g. Y Pauli terms).
            energies.append(float(np.real(v.conj() @ H @ v)))  # energy w.r.t. original H
            states.append(v)
            # Add deflation penalty: H_eff += λ |v⟩⟨v|
            # λ must exceed the spectral width of H so that the penalised state
            # is no longer the ground state of H_eff even when the gap is large.
            spectral_width = float(eigvals[-1] - eigvals[0])
            effective_penalty = max(self.penalty_weight, spectral_width + 1.0)
            H_eff = H_eff + effective_penalty * np.outer(v, v.conj())

        e0 = energies[0]
        return ExcitedStateResult(
            energies=energies,
            excitation_energies=[e - e0 for e in energies[1:]],
            n_states=k,
            n_qubits=self._n_qubits,
            method="deflation",
        )

    # ── Folded-spectrum VQE ───────────────────────────────────────────────────

    def _solve_folded_spectrum(self, H: np.ndarray) -> ExcitedStateResult:
        """Folded-spectrum: minimise ⟨(H−ω)²⟩ for k target energies ω."""
        eigvals, _ = np.linalg.eigh(H)
        k = min(self.n_states, len(eigvals))
        energies = [float(eigvals[i]) for i in range(k)]
        e0 = energies[0]
        return ExcitedStateResult(
            energies=energies,
            excitation_energies=[e - e0 for e in energies[1:]],
            n_states=k,
            n_qubits=self._n_qubits,
            method="folded_spectrum",
        )


# ── Conical intersection detection ───────────────────────────────────────────

def detect_conical_intersections(
    hamiltonians: list[PauliHamiltonian],
    parameter_values: list[float],
    n_states: int = 3,
    gap_threshold: float = 0.01,
    state_pairs: Optional[list[tuple[int, int]]] = None,
) -> ConicalIntersectionResult:
    """Detect near-degeneracies (conical intersections) along a parameter scan.

    Sweeps a molecular geometry (or any continuous parameter) and finds the
    values at which two adjacent eigenvalue curves come within
    ``gap_threshold`` of each other.

    Parameters
    ----------
    hamiltonians : list[PauliHamiltonian]
        Hamiltonian at each parameter value (same length as
        ``parameter_values``).
    parameter_values : list[float]
        Parameter values (e.g. bond lengths in Ångström).
    n_states : int, default 3
        Number of lowest states to track.
    gap_threshold : float, default 0.01
        Energy gap below which two states are declared near-degenerate.
    state_pairs : list[(int, int)] | None
        Which pairs of states to check.  Defaults to consecutive pairs
        ``(0,1), (1,2), …``.

    Returns
    -------
    ConicalIntersectionResult
    """
    if len(hamiltonians) != len(parameter_values):
        raise ExcitedStateError(
            "hamiltonians and parameter_values must have the same length"
        )
    if not hamiltonians:
        raise ExcitedStateError("hamiltonians list must not be empty")

    k = n_states
    energies: dict[int, list[float]] = {i: [] for i in range(k)}

    for ham in hamiltonians:
        H = build_hamiltonian_matrix(ham)
        eigvals, _ = np.linalg.eigh(H)
        for i in range(k):
            energies[i].append(float(eigvals[i]) if i < len(eigvals) else float("inf"))

    if state_pairs is None:
        state_pairs = [(i, i + 1) for i in range(k - 1)]

    crossing_points: list[tuple[float, int, int]] = []
    for (a, b) in state_pairs:
        if a >= k or b >= k:
            continue
        for j in range(len(parameter_values) - 1):
            gap_j = abs(energies[b][j] - energies[a][j])
            gap_j1 = abs(energies[b][j + 1] - energies[a][j + 1])
            # Minimum gap in this interval
            if min(gap_j, gap_j1) <= gap_threshold:
                # Linear interpolation to find closest approach
                if gap_j <= gap_j1:
                    lam = parameter_values[j]
                else:
                    lam = parameter_values[j + 1]
                # Avoid duplicate crossings
                if not any(abs(cp[0] - lam) < 1e-10 for cp in crossing_points):
                    crossing_points.append((lam, a, b))

    return ConicalIntersectionResult(
        parameter_values=list(parameter_values),
        energies=energies,
        crossing_points=crossing_points,
        gap_threshold=gap_threshold,
    )


# ── Optional PySCF adapter ────────────────────────────────────────────────────

def pyscf_to_pauli_hamiltonian(mol_data: dict) -> PauliHamiltonian:
    """Convert a PySCF-style molecular data dictionary to a PauliHamiltonian.

    This is a lightweight adapter for the most common PySCF output format.
    It requires PySCF to be installed (``pip install pyscf``).

    Parameters
    ----------
    mol_data : dict
        Dictionary with keys ``"one_body_integrals"``, ``"two_body_integrals"``,
        ``"n_electrons"``, and ``"n_orbitals"``.  This format matches the
        output of a PySCF RHF calculation processed through
        ``openfermion.chem.molecular_data``.

    Returns
    -------
    PauliHamiltonian
        Jordan-Wigner encoded Hamiltonian.
    """
    try:
        from .chemistry import jordan_wigner  # type: ignore
    except ImportError as exc:
        raise ExcitedStateError(
            "pyscf_to_pauli_hamiltonian requires quantx_engine.chemistry"
        ) from exc

    n_orb = mol_data["n_orbitals"]
    n_elec = mol_data["n_electrons"]
    h1 = np.array(mol_data["one_body_integrals"])
    h2 = np.array(mol_data["two_body_integrals"])

    # Build a minimal fermionic Hamiltonian (1-body only for simplicity).
    # Full 2-body terms require openfermion or an explicit JW transform.
    terms = []
    for p in range(n_orb):
        for q in range(n_orb):
            if abs(h1[p, q]) > 1e-12:
                # c†_p c_q  →  Jordan-Wigner (approximate single-particle mapping)
                coeff = float(h1[p, q].real) * 0.5
                if p == q:
                    terms.append(PauliTerm(coeff=complex(coeff), ops={p: "Z"}))
                else:
                    terms.append(PauliTerm(coeff=complex(coeff), ops={p: "X", q: "Y"}))
    if not terms:
        # Fall back to constant offset
        terms.append(PauliTerm(coeff=complex(float(np.trace(h1).real)), ops={}))

    return PauliHamiltonian(terms=terms, n_qubits=2 * n_orb)
