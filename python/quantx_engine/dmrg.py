"""M176 — DMRG ground-state solver for the QuantX engine.

Wraps the native C++ two-site DMRG implementation (``src/dmrg_sparse.cpp``)
with a clean Python API.  Falls back to a pure-Python reference DMRG (via
``quantx_engine.tdmrg``) when the native library is not available.

Supported Hamiltonians
----------------------
* Heisenberg XXX chain:
  ``{"type": "heisenberg", "J": 1.0}``
* General sum of Pauli strings (nearest-neighbour or longer range):
  Passed as ``hamiltonian_terms=[(coeff, "X0X1"), ...]``

Maximum system size: 200 qubits (C++ backend limit).

Quick-start example::

    from quantx_engine import Engine
    from quantx_engine.dmrg import DMRGSolver

    solver = DMRGSolver(Engine())

    # Heisenberg chain n=50, bond dim 100
    result = solver.solve(
        n_qubits=50,
        hamiltonian_type="heisenberg",
        J=1.0,
        max_bond=100,
        n_sweeps=10,
    )
    print(f"Ground state energy: {result.energy:.8f}")
    print(f"Converged in {result.n_sweeps_done} sweeps")

    # General Hamiltonian via Pauli strings
    result = solver.solve(
        n_qubits=10,
        hamiltonian_terms=[(1.0, "X0X1"), (1.0, "Y0Y1"), (1.0, "Z0Z1")],
        max_bond=32,
    )

References
----------
.. [1] Schollwöck (2011). Ann. Phys. 326, 96. DOI: 10.1016/j.aop.2010.09.012
.. [2] White (1992). Phys. Rev. Lett. 69, 2863. DOI: 10.1103/PhysRevLett.69.2863
"""
from __future__ import annotations

import json
from ctypes import c_char_p
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["DMRGSolver", "DMRGResult", "DMRGError"]

_MAX_DMRG_QUBITS = 200

# ── Exceptions ──────────────────────────────────────────────────────────────

class DMRGError(RuntimeError):
    """Raised when the DMRG solver encounters an unrecoverable error."""


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class DMRGResult:
    """Ground-state result from the DMRG solver.

    Attributes
    ----------
    energy : float
        Variational ground-state energy (in units of the Hamiltonian coupling).
    truncation_error : float
        Total discarded weight across all SVD truncations in the last sweep.
    n_sweeps_done : int
        Number of full sweeps completed (may be less than ``n_sweeps`` if
        converged early).
    sweep_energies : list[float]
        Energy recorded at the end of each half-sweep (left→right).
    wall_time_s : float
        Wall-clock time in seconds for the full solver run.
    backend : str
        ``"cpp"`` when the native library ran; ``"python"`` for the fallback.
    """
    energy: float
    truncation_error: float
    n_sweeps_done: int
    sweep_energies: list[float]
    wall_time_s: float
    backend: str = "cpp"


# ── ctypes helpers ────────────────────────────────────────────────────────────

def _get_dmrg_lib(engine) -> Any | None:
    """Return the native lib if ``qx_dmrg_sweep`` is compiled in, else None."""
    lib = getattr(engine, "_lib", None)
    if lib is None:
        return None
    try:
        fn = lib.qx_dmrg_sweep
        fn.argtypes  = [c_char_p, c_char_p]
        fn.restype   = c_char_p
        fn2 = lib.qx_mps_energy
        fn2.argtypes = [c_char_p]
        fn2.restype  = c_char_p
        return lib
    except AttributeError:
        return None


# ── Pure-Python fallback: exact diagonalisation ───────────────────────────────
# The C++ DMRG is the only DMRG implementation.  Without the native library
# we fall back to exact diagonalisation (numpy eigvalsh), which is only feasible
# for small systems (n ≤ 20, Hilbert space ≤ 2^20 = 1M).  For larger systems
# a DMRGError is raised, asking the user to build the native extension.

def _run_python_dmrg(n_qubits: int,
                     hamiltonian_type: str,
                     J: float,
                     hamiltonian_terms: list,
                     max_bond: int,
                     cutoff: float,
                     n_sweeps: int,
                     tol: float) -> DMRGResult:
    """Exact-diagonalisation fallback when the C++ native library is absent.

    Only viable for n_qubits ≤ 20.  Raises DMRGError for larger systems,
    directing the user to build the native extension.
    """
    import time
    import numpy as np

    t0 = time.perf_counter()
    if n_qubits > 20:
        raise DMRGError(
            f"Native library not available and n_qubits={n_qubits} > 20 "
            "is too large for the pure-Python exact-diag fallback. "
            "Build the native library to use the C++ DMRG solver."
        )

    # Build Hamiltonian matrix via Pauli strings
    d = 2 ** n_qubits
    H = np.zeros((d, d), dtype=complex)
    _sx = np.array([[0, 1], [1, 0]], dtype=complex)
    _sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    _sz = np.array([[1, 0], [0, -1]], dtype=complex)
    _id = np.eye(2, dtype=complex)
    _paulis = {"X": _sx, "Y": _sy, "Z": _sz, "I": _id}

    if hamiltonian_type == "heisenberg":
        terms_ed = []
        for i in range(n_qubits - 1):
            for g in ("X", "Y", "Z"):
                terms_ed.append((J, f"{g}{i}{g}{i+1}"))
    else:
        terms_ed = list(hamiltonian_terms)

    import re
    for coeff, label in terms_ed:
        ops = ["I"] * n_qubits
        for m in re.finditer(r"([XYZI])(\d+)", label):
            ops[int(m.group(2))] = m.group(1)
        mat = np.array([[1.0]], dtype=complex)
        for op in ops:
            mat = np.kron(mat, _paulis[op])
        H += coeff * mat

    vals = np.linalg.eigvalsh(H)
    elapsed = time.perf_counter() - t0
    return DMRGResult(
        energy=float(vals[0]),
        truncation_error=0.0,
        n_sweeps_done=0,
        sweep_energies=[float(vals[0])],
        wall_time_s=elapsed,
        backend="python_exact",
    )


# ── Main solver class ─────────────────────────────────────────────────────────

class DMRGSolver:
    """Two-site DMRG ground-state solver.

    Parameters
    ----------
    engine : quantx_engine.Engine
        QuantX engine instance.  The native C++ DMRG path is used when the
        native library exposes ``qx_dmrg_sweep``; otherwise falls back to
        pure-Python.
    """

    def __init__(self, engine) -> None:
        self._engine = engine
        self._lib    = _get_dmrg_lib(engine)

    # ── Public API ──────────────────────────────────────────────────────────

    def solve(
        self,
        n_qubits: int,
        *,
        hamiltonian_type: str = "pauli_sum",
        J: float = 1.0,
        hamiltonian_terms: Optional[list] = None,
        max_bond: int = 64,
        cutoff: float = 1e-10,
        n_sweeps: int = 10,
        tol: float = 1e-10,
    ) -> DMRGResult:
        """Run DMRG to find the ground-state energy.

        Parameters
        ----------
        n_qubits : int
            Number of spin-1/2 sites.
        hamiltonian_type : str
            ``"heisenberg"`` for XXX Heisenberg chain, or ``"pauli_sum"``
            for a general sum of Pauli strings (requires ``hamiltonian_terms``).
        J : float
            Heisenberg coupling constant (used when ``hamiltonian_type="heisenberg"``).
        hamiltonian_terms : list of (float, str), optional
            List of ``(coefficient, pauli_label)`` pairs.  Pauli labels may
            be in indexed format (``"X0X1"``) or plain format (``"XXI..."``).
        max_bond : int
            Maximum MPS bond dimension χ.  Memory scales as O(χ² · n).
        cutoff : float
            SVD singular-value weight cutoff; states with discarded weight
            below ``cutoff²`` are dropped even if ``max_bond`` is not reached.
        n_sweeps : int
            Maximum number of full left→right→left DMRG sweeps.
        tol : float
            Energy convergence tolerance.  Stops early when |ΔE| < tol.

        Returns
        -------
        DMRGResult

        Raises
        ------
        DMRGError
            On invalid parameters or solver failure.
        """
        if n_qubits < 2:
            raise DMRGError("n_qubits must be ≥ 2")
        if n_sweeps < 1:
            raise DMRGError("n_sweeps must be ≥ 1")
        if n_qubits > _MAX_DMRG_QUBITS:
            raise DMRGError(
                f"n_qubits={n_qubits} exceeds the maximum of {_MAX_DMRG_QUBITS}"
            )
        if hamiltonian_type not in ("heisenberg", "pauli_sum"):
            raise DMRGError(
                f"Unknown hamiltonian_type '{hamiltonian_type}'. "
                "Use 'heisenberg' or 'pauli_sum'."
            )
        if hamiltonian_type == "pauli_sum" and not hamiltonian_terms:
            raise DMRGError(
                "hamiltonian_terms must be provided when hamiltonian_type='pauli_sum'"
            )

        if self._lib is not None:
            return self._solve_cpp(
                n_qubits, hamiltonian_type, J, hamiltonian_terms or [],
                max_bond, cutoff, n_sweeps, tol
            )
        # Fallback
        return _run_python_dmrg(
            n_qubits, hamiltonian_type, J, hamiltonian_terms or [],
            max_bond, cutoff, n_sweeps, tol
        )

    def mps_energy(
        self,
        n_qubits: int,
        *,
        hamiltonian_type: str = "heisenberg",
        J: float = 1.0,
        hamiltonian_terms: Optional[list] = None,
    ) -> float:
        """Compute ⟨0...0|H|0...0⟩ for the all-zero product state.

        Useful for a quick sanity-check of the Hamiltonian encoding.

        Returns
        -------
        float
            Energy expectation value.
        """
        if self._lib is None:
            raise DMRGError(
                "mps_energy requires the native C++ library. "
                "Build the native extension or use DMRGSolver.solve() instead."
            )
        prob: dict = {"n_qubits": n_qubits}
        if hamiltonian_type == "heisenberg":
            prob["type"] = "heisenberg"
            prob["J"]    = J
        else:
            prob["type"]  = "pauli_sum"
            prob["terms"] = [
                {"coeff": float(c), "pauli": p}
                for c, p in (hamiltonian_terms or [])
            ]
        raw = self._lib.qx_mps_energy(json.dumps(prob).encode())
        out = json.loads(raw)
        if not out.get("ok", False):
            raise DMRGError(out.get("error", "Unknown error from qx_mps_energy"))
        return float(out["energy"])

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _solve_cpp(
        self,
        n_qubits: int,
        hamiltonian_type: str,
        J: float,
        hamiltonian_terms: list,
        max_bond: int,
        cutoff: float,
        n_sweeps: int,
        tol: float,
    ) -> DMRGResult:
        prob: dict = {"n_qubits": n_qubits}
        if hamiltonian_type == "heisenberg":
            prob["type"] = "heisenberg"
            prob["J"]    = J
        else:
            prob["type"]  = "pauli_sum"
            prob["terms"] = [
                {"coeff": float(c), "pauli": p}
                for c, p in hamiltonian_terms
            ]

        opts: dict = {
            "max_bond": max_bond,
            "cutoff":   cutoff,
            "n_sweeps": n_sweeps,
            "tol":      tol,
        }

        raw = self._lib.qx_dmrg_sweep(
            json.dumps(prob).encode(),
            json.dumps(opts).encode(),
        )
        out = json.loads(raw)

        if not out.get("ok", False):
            raise DMRGError(out.get("error", "Unknown error from qx_dmrg_sweep"))

        return DMRGResult(
            energy=float(out["energy"]),
            truncation_error=float(out.get("truncation_error", 0.0)),
            n_sweeps_done=int(out.get("n_sweeps_done", n_sweeps)),
            sweep_energies=[float(e) for e in out.get("sweep_energies", [])],
            wall_time_s=float(out.get("wall_time_s", 0.0)),
            backend="cpp",
        )
