"""Lindblad open-system dynamics for the QuantX engine.

Wraps the native C++ ``lindblad_backend`` (RK4 density-matrix evolution,
Wootters concurrence, log-negativity, mutual information) with a clean
Python API.

The Lindblad master equation::

    dρ/dt = −i[H, ρ] + Σ_k γ_k (L_k ρ L_k† − ½{L_k†L_k, ρ})

is integrated by the C++ backend using a 4th-order Runge-Kutta scheme with
automatic Hermitian re-symmetrisation at every step to suppress numerical
drift.

Supported collapse channels (single-qubit, specified via
:class:`JumpOperator`):

* ``"amplitude_damping"`` — L = √γ σ⁻  (energy relaxation T₁)
* ``"dephasing"``         — L = √γ σz/2  (pure dephasing T₂)
* ``"depolarizing"``      — three operators √(γ/3)·{σx, σy, σz}

Maximum system size: 10 qubits (C++ backend limit; dense 2ⁿ × 2ⁿ matrix).

Quick-start example::

    from quantx_engine import Engine
    from quantx_engine.lindblad import LindbladSolver, JumpOperator

    engine = Engine()
    solver = LindbladSolver(engine)

    # Single qubit with amplitude damping (T1 decay)
    result = solver.solve(
        n_qubits=1,
        hamiltonian_terms=[],          # no coherent drive
        jump_operators=[JumpOperator("amplitude_damping", qubit=0, rate=0.5)],
        total_time=4.0,
        dt=0.05,
    )
    print(result.final_purity)         # → ~0 (fully mixed at long time)
    print(result.times[-1])            # → 4.0

    # Two-qubit system: Heisenberg XY + dephasing on both qubits
    result = solver.solve(
        n_qubits=2,
        hamiltonian_terms=[(0.5, "X0X1"), (0.5, "Y0Y1")],
        jump_operators=[
            JumpOperator("dephasing", qubit=0, rate=0.1),
            JumpOperator("dephasing", qubit=1, rate=0.1),
        ],
        total_time=10.0,
        dt=0.1,
        entanglement_cut=1,            # track bipartite entanglement A=[0], B=[1]
    )
    print(result.entanglement.concurrence[-1])  # concurrence decays to 0


References
----------
.. [1] Lindblad, G. (1976). On the generators of quantum dynamical semigroups.
       *Commun. Math. Phys.* 48, 119.
       https://doi.org/10.1007/BF01608499

.. [2] Gorini, V., Kossakowski, A. & Sudarshan, E. C. G. (1976). Completely
       positive dynamical semigroups of n-level systems.
       *J. Math. Phys.* 17, 821.
       https://doi.org/10.1063/1.522979

.. [3] Johansson, J. R., Nation, P. D. & Nori, F. (2012). QuTiP: An open-
       source Python framework for the dynamics of open quantum systems.
       *Comput. Phys. Commun.* 183, 1760.
       https://doi.org/10.1016/j.cpc.2012.02.021

Physical validation
-------------------
Single-qubit amplitude damping (L = sqrt(gamma)*sigma-): the excited state
population decays exponentially as p_e(t) = exp(-gamma*t) [Lindblad1976].
Trace preservation: Tr(rho(t)) = 1 for all t (Lindblad master equation
preserves positivity and trace) [Gorini1976].
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from quantx_engine.engine import Engine

__all__ = [
    "LindbladError",
    "JumpOperator",
    "LindbladFrame",
    "EntanglementTrace",
    "LindbladResult",
    "LindbladSolver",
]

# Maximum qubits supported by the dense C++ lindblad_backend (RK4).
_MAX_LINDBLAD_QUBITS_DENSE = 10

# Maximum qubits for the sparse (matrix-free) C++ backend (M173).
_MAX_LINDBLAD_QUBITS_SPARSE_EVOLVE = 16   # memory: ~32 GB at n=16
_MAX_LINDBLAD_QUBITS_SPARSE_STEADY = 12   # explicit Liouvillian: ~128 MB at n=12

# Backward-compat alias used throughout existing code.
_MAX_LINDBLAD_QUBITS = _MAX_LINDBLAD_QUBITS_SPARSE_EVOLVE

# Map Python channel names → C++ label prefixes (dense backend)
_CHANNEL_PREFIXES: dict[str, str] = {
    "amplitude_damping": "AmplitudeDamping",
    "dephasing":         "Dephasing",
    "depolarizing":      "Depolarizing",
}


# ---------------------------------------------------------------------------
# Sparse backend ctypes helpers (M173)
# ---------------------------------------------------------------------------

def _parse_pauli_label(label: str, n_qubits: int) -> str:
    """Convert an indexed Pauli label to a plain per-qubit string.

    The public API uses labels like ``"X0"``, ``"Z0Z1"``, ``"X0X1"``
    where each letter+digit pair specifies a Pauli on a qubit.  The sparse
    C++ backend expects a plain string of length n_qubits where character i
    is the Pauli on qubit i (e.g. ``"XI"`` for X on qubit 0, I on qubit 1).

    Examples
    --------
    >>> _parse_pauli_label("X0X1", 2)
    'XX'
    >>> _parse_pauli_label("Z0", 3)
    'ZII'
    >>> _parse_pauli_label("XXI", 3)   # already plain format — passed through
    'XXI'
    """
    import re
    # Detect indexed format: contains digits after Pauli letters
    if re.search(r'[XYZI]\d', label):
        result = ['I'] * n_qubits
        for m in re.finditer(r'([XYZI])(\d+)', label):
            op, q = m.group(1), int(m.group(2))
            if q < n_qubits:
                result[q] = op
        return ''.join(result)
    # Already plain format (e.g. "XZI") — pad or truncate to n_qubits
    plain = label.upper().replace(' ', '')
    if len(plain) < n_qubits:
        plain = plain + 'I' * (n_qubits - len(plain))
    return plain[:n_qubits]


def _get_sparse_lib(engine: "Engine"):
    """Return the ctypes library handle from *engine* if it exposes the
    sparse Lindblad symbols, otherwise return None."""
    import ctypes
    from ctypes import c_char_p, c_void_p
    lib = getattr(engine, "_lib", None)
    if lib is None:
        return None
    # Check that the sparse symbols were compiled in
    try:
        fn = lib.qx_lindblad_sparse_evolve
        fn.argtypes = [c_char_p, c_char_p]
        fn.restype  = c_char_p   # thread-local, no free needed
        fn2 = lib.qx_lindblad_sparse_steady
        fn2.argtypes = [c_char_p]
        fn2.restype  = c_char_p
        return lib
    except AttributeError:
        return None


def _sparse_evolve_raw(lib, problem_dict: dict) -> dict:
    """Call qx_lindblad_sparse_evolve and return parsed JSON."""
    import json as _json
    import ctypes
    prob_bytes = _json.dumps(problem_dict).encode()
    opts_bytes = b"{}"
    raw = lib.qx_lindblad_sparse_evolve(
        ctypes.c_char_p(prob_bytes),
        ctypes.c_char_p(opts_bytes),
    )
    if raw is None:
        raise LindbladError("qx_lindblad_sparse_evolve returned NULL")
    return _json.loads(raw.decode() if isinstance(raw, bytes) else raw)


def _sparse_steady_raw(lib, problem_dict: dict) -> dict:
    """Call qx_lindblad_sparse_steady and return parsed JSON."""
    import json as _json
    import ctypes
    prob_bytes = _json.dumps(problem_dict).encode()
    raw = lib.qx_lindblad_sparse_steady(ctypes.c_char_p(prob_bytes))
    if raw is None:
        raise LindbladError("qx_lindblad_sparse_steady returned NULL")
    return _json.loads(raw.decode() if isinstance(raw, bytes) else raw)


class LindbladError(ValueError):
    """Raised for invalid Lindblad simulation parameters."""


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------

class JumpOperator:
    """A collapse (jump) operator for the Lindblad equation.

    Can be constructed in two ways:

    1. **Named channel** (original API)::

        JumpOperator("amplitude_damping", qubit=0, rate=0.5)
        JumpOperator("dephasing",         qubit=1, rate=0.1)

    2. **Explicit matrix** (literature API)::

        L = np.sqrt(gamma) * sigma_minus
        JumpOperator(L)

    Parameters
    ----------
    kind_or_matrix:
        Either a channel-kind string (``"amplitude_damping"``,
        ``"dephasing"``, ``"depolarizing"``) or a numpy array representing
        the Kraus/jump operator matrix directly.
    qubit:
        Target qubit index (0-based).  Required for named-channel form.
    rate:
        Jump rate γ ≥ 0.  Required for named-channel form.
    """

    def __init__(
        self,
        kind_or_matrix,
        qubit: int | None = None,
        rate: float | None = None,
    ) -> None:
        import numpy as _np
        if isinstance(kind_or_matrix, _np.ndarray):
            # Matrix form
            self._matrix: "_np.ndarray | None" = kind_or_matrix
            self.kind: "str | None" = None
            self.qubit: "int | None" = None
            self.rate: "float | None" = None
        else:
            # Named-channel form (backwards-compatible dataclass behaviour)
            self._matrix = None
            self.kind = str(kind_or_matrix)
            if self.kind not in _CHANNEL_PREFIXES:
                raise LindbladError(
                    f"Unknown jump operator kind {self.kind!r}. "
                    f"Choose from: {list(_CHANNEL_PREFIXES)}"
                )
            if qubit is None:
                raise LindbladError("qubit must be provided for named JumpOperator")
            if rate is None:
                raise LindbladError("rate must be provided for named JumpOperator")
            self.qubit = int(qubit)
            self.rate = float(rate)
            if self.qubit < 0:
                raise LindbladError(f"qubit must be >= 0, got {self.qubit}")
            if self.rate < 0.0:
                raise LindbladError(f"rate must be >= 0, got {self.rate}")

    @property
    def matrix(self):
        """Kraus operator matrix (only set for matrix-form JumpOperators)."""
        return self._matrix

    def _to_hamiltonian_term(self) -> dict[str, Any]:
        """Convert to the C++ ``HamiltonianTerm`` JSON dict."""
        if self.kind is None:
            raise LindbladError(
                "Cannot convert matrix-form JumpOperator to Hamiltonian term. "
                "Use the named-channel form instead."
            )
        prefix = _CHANNEL_PREFIXES[self.kind]
        return {"label": f"{prefix}{self.qubit}", "coefficient": float(self.rate)}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class LindbladFrame:
    """Snapshot of the density matrix at one integration step.

    Attributes
    ----------
    step:
        Integration step index (0 = initial state).
    time:
        Physical time at this step.
    purity:
        Tr(ρ²) ∈ [1/d, 1] where d = 2ⁿ.  1 = pure state, 1/d = maximally mixed.
    entropy:
        Von Neumann entropy −Tr(ρ ln ρ) ≥ 0.
    coherence:
        Maximum off-diagonal element max_{i≠j} |ρ_{ij}|.
    populations:
        Diagonal elements ρ_{ii} (basis-state populations), length = 2ⁿ.
    """
    step: int
    time: float
    purity: float
    entropy: float
    coherence: float
    populations: list[float]


@dataclass
class EntanglementTrace:
    """Bipartite entanglement measures over time.

    Computed only when ``entanglement_cut > 0`` is passed to
    :meth:`LindbladSolver.solve`.  Subsystem A = qubits 0…(cut-1),
    subsystem B = qubits cut…(n-1).

    Attributes
    ----------
    times:
        Time axis (same length as other arrays).
    entropy_a:
        Von Neumann entropy of the reduced density matrix ρ_A.
    mutual_info:
        Mutual information I(A:B) = S_A + S_B − S_AB ≥ 0.
    log_negativity:
        Logarithmic negativity E_N = log₂ ‖ρ^{T_A}‖₁ (entanglement monotone).
    concurrence:
        Wootters concurrence C(ρ) ∈ [0, 1].  Only meaningful (non-zero) for
        2-qubit systems.
    """
    times: list[float]
    entropy_a: list[float]
    mutual_info: list[float]
    log_negativity: list[float]
    concurrence: list[float]

    def __len__(self) -> int:
        return len(self.times)


@dataclass
class LindbladResult:
    """Result of a Lindblad master equation integration.

    Attributes
    ----------
    times:
        Time axis, length = n_steps + 1 (includes t=0).
    purity:
        Tr(ρ²) at each time step.
    entropy:
        Von Neumann entropy at each time step.
    coherence:
        Max off-diagonal element at each time step.
    populations:
        populations[t] is a list of 2ⁿ basis-state populations at time t.
    final_purity:
        Purity of the final state.
    final_entropy:
        Von Neumann entropy of the final state.
    final_populations:
        Basis-state populations of the final state.
    frames:
        Full per-step snapshots (includes all of the above plus step index).
    entanglement:
        Bipartite entanglement traces.  ``None`` if ``entanglement_cut=0``.
    n_qubits:
        Number of qubits simulated.
    total_time:
        Total evolution time.
    dt:
        Integration time step.
    """
    times: list[float]
    purity: list[float]
    entropy: list[float]
    coherence: list[float]
    populations: list[list[float]]
    final_purity: float
    final_entropy: float
    final_populations: list[float]
    frames: list[LindbladFrame]
    entanglement: EntanglementTrace | None
    n_qubits: int
    total_time: float
    dt: float

    def steady_state_reached(self, tol: float = 1e-4) -> bool:
        """Return True if purity has converged within ``tol`` over the last 10%
        of the time axis.

        This is a simple heuristic: if the purity varies by less than ``tol``
        in the final 10% of the simulation the system is likely near its
        steady state.

        .. warning::
            This heuristic can give **false positives** for coherently driven
            systems whose purity is approximately constant (e.g. near-pure
            Rabi oscillations).  It is most reliable for dissipative dynamics
            where purity monotonically decays to a fixed-point value.
        """
        if len(self.purity) < 4:
            return False
        tail_start = max(1, int(0.9 * len(self.purity)))
        tail = self.purity[tail_start:]
        if len(tail) < 2:
            return False  # too few tail points to assess convergence
        return (max(tail) - min(tail)) < tol

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"LindbladResult(n_qubits={self.n_qubits}, "
            f"total_time={self.total_time}, dt={self.dt}, "
            f"n_steps={len(self.times)-1}, "
            f"final_purity={self.final_purity:.4g}, "
            f"final_entropy={self.final_entropy:.4g})"
        )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class LindbladSolver:
    """Solve the Lindblad master equation using the native C++ RK4 backend.

    Parameters
    ----------
    engine:
        An initialised :class:`~quantx_engine.engine.Engine` instance.

    Examples
    --------
    >>> from quantx_engine import Engine
    >>> from quantx_engine.lindblad import LindbladSolver, JumpOperator
    >>> solver = LindbladSolver(Engine())
    >>> result = solver.solve(
    ...     n_qubits=1,
    ...     hamiltonian_terms=[],
    ...     jump_operators=[JumpOperator("amplitude_damping", 0, rate=1.0)],
    ...     total_time=5.0,
    ... )
    >>> result.final_purity < 0.01   # fully mixed
    True
    """

    def __init__(self, engine: "Engine"):
        self._engine = engine

    def solve(
        self,
        n_qubits: int,
        hamiltonian_terms: list[tuple[float, str]],
        jump_operators: list[JumpOperator],
        total_time: float,
        dt: float = 0.01,
        entanglement_cut: int = 0,
    ) -> LindbladResult:
        """Integrate the Lindblad equation from |0…0⟩⟨0…0|.

        Parameters
        ----------
        n_qubits:
            Number of qubits.  Maximum 10 (dense RK4 backend limit).
        hamiltonian_terms:
            Coherent Hamiltonian as a list of ``(coefficient, pauli_string)``
            tuples.  The Pauli string uses the format ``"X0"``, ``"Z0Z1"``,
            ``"Y2"`` etc.  Example: ``[(1.0, "Z0"), (-0.5, "X0X1")]``.
        jump_operators:
            List of :class:`JumpOperator` instances describing the collapse
            channels.
        total_time:
            Total evolution time.
        dt:
            RK4 integration time step.  Smaller values give higher accuracy
            but increase runtime.  Rule of thumb: dt < 0.1 / max_rate.
        entanglement_cut:
            If > 0, compute bipartite entanglement between qubits
            0…(cut-1) and cut…(n-1) at every step.  Entanglement data
            is returned in :attr:`LindbladResult.entanglement`.
            Set to 0 (default) to skip (saves memory and compute).

        Returns
        -------
        LindbladResult

        Raises
        ------
        LindbladError
            On invalid parameters (too many qubits, negative time, etc.).
        """
        # ── Validate ──────────────────────────────────────────────────────────
        if n_qubits < 1:
            raise LindbladError(f"n_qubits must be >= 1, got {n_qubits}")
        if n_qubits > _MAX_LINDBLAD_QUBITS:
            raise LindbladError(
                f"LindbladSolver supports at most {_MAX_LINDBLAD_QUBITS} qubits "
                f"(sparse matrix-free C++ backend limit), got {n_qubits}"
            )
        if total_time <= 0.0:
            raise LindbladError(f"total_time must be > 0, got {total_time}")
        if dt <= 0.0:
            raise LindbladError(f"dt must be > 0, got {dt}")
        if dt > total_time:
            raise LindbladError(
                f"dt ({dt}) must be <= total_time ({total_time})"
            )
        if entanglement_cut != 0 and not (1 <= entanglement_cut <= n_qubits - 1):
            raise LindbladError(
                f"entanglement_cut must be 0 or in [1, n_qubits-1], "
                f"got {entanglement_cut} for n_qubits={n_qubits}"
            )
        for jop in jump_operators:
            if jop.qubit >= n_qubits:
                raise LindbladError(
                    f"JumpOperator targets qubit {jop.qubit} but n_qubits={n_qubits}"
                )

        # ── Build HamiltonianTerm list ─────────────────────────────────────────
        terms: list[dict[str, Any]] = []
        for coeff, pauli_str in hamiltonian_terms:
            terms.append({"label": pauli_str, "coefficient": float(coeff)})
        for jop in jump_operators:
            terms.append(jop._to_hamiltonian_term())

        # ── Construct problem dict ────────────────────────────────────────────
        problem: dict[str, Any] = {
            "qubits": n_qubits,
            "equation": "Lindblad",
            "total_time": float(total_time),
            "dt": float(dt),
            "hamiltonian_terms": terms,
        }
        options: dict[str, Any] = {}
        if entanglement_cut > 0:
            options["entanglement_cut"] = entanglement_cut

        # ── Route to sparse or dense backend ─────────────────────────────────
        sparse_lib = _get_sparse_lib(self._engine)

        # The sparse backend does NOT emit entanglement_frames, so whenever
        # the caller requests entanglement tracking we must use the dense path
        # (which supports it for n ≤ 10).
        needs_entanglement = entanglement_cut > 0

        if sparse_lib is not None and n_qubits > _MAX_LINDBLAD_QUBITS_DENSE:
            # Must use sparse — dense backend cannot handle this many qubits.
            if needs_entanglement:
                raise LindbladError(
                    f"entanglement_cut tracking is only supported for n_qubits ≤ "
                    f"{_MAX_LINDBLAD_QUBITS_DENSE} (dense backend limit). "
                    f"Got n_qubits={n_qubits}."
                )
            return self._solve_sparse(sparse_lib, n_qubits, hamiltonian_terms,
                                      jump_operators, total_time, dt)

        if sparse_lib is not None and n_qubits > 6 and not needs_entanglement:
            # Prefer sparse for n > 6 when entanglement tracking is not needed.
            return self._solve_sparse(sparse_lib, n_qubits, hamiltonian_terms,
                                      jump_operators, total_time, dt)

        # Dense C++ backend (n ≤ 10, via engine.run JSON RPC).
        # Supports entanglement_cut and is used when n ≤ 6 or entanglement requested.
        raw = self._engine.run(problem, options)
        return _parse_result(raw, n_qubits, total_time, dt)

    # ------------------------------------------------------------------
    def _solve_sparse(
        self,
        lib,
        n_qubits: int,
        hamiltonian_terms: list[tuple[float, str]],
        jump_operators: list["JumpOperator"],
        total_time: float,
        dt: float,
    ) -> "LindbladResult":
        """Run time evolution via the sparse matrix-free C++ backend (M173)."""
        import json as _json

        # Build problem dict for sparse backend.
        # Convert indexed labels ("X0X1") to plain per-qubit strings ("XX").
        ham_list = [{"coeff": float(c),
                     "pauli": _parse_pauli_label(p, n_qubits)}
                    for c, p in hamiltonian_terms]
        jop_list = []
        for jop in jump_operators:
            jop_list.append({
                "type":  jop.kind,
                "qubit": jop.qubit,
                "rate":  float(jop.rate),
            })

        prob = {
            "n_qubits":        n_qubits,
            "hamiltonian_terms": ham_list,
            "jump_operators":  jop_list,
            "total_time":      float(total_time),
            "dt":              float(dt),
        }

        raw = _sparse_evolve_raw(lib, prob)

        if "error" in raw:
            raise LindbladError(f"Sparse Lindblad backend error: {raw['error']}")

        # Convert to LindbladResult via existing frame parser
        return _parse_result(raw, n_qubits, total_time, dt)

    def steady_state(
        self,
        n_qubits: int,
        hamiltonian_terms: list[tuple[float, str]],
        jump_operators: list["JumpOperator"],
    ) -> dict:
        """Compute the steady-state density matrix ρ_ss via sparse BiCGSTAB.

        Requires the native library with M173 sparse backend compiled in.
        Supports up to 12 qubits.

        Returns
        -------
        dict with keys: purity, entropy, populations (diagonal of ρ_ss)
        """
        if n_qubits < 1 or n_qubits > _MAX_LINDBLAD_QUBITS_SPARSE_STEADY:
            raise LindbladError(
                f"steady_state() supports 1–{_MAX_LINDBLAD_QUBITS_SPARSE_STEADY} "
                f"qubits, got {n_qubits}"
            )
        sparse_lib = _get_sparse_lib(self._engine)
        if sparse_lib is None:
            raise LindbladError(
                "steady_state() requires the native library with M173 sparse "
                "Lindblad support. Build with cmake and set QUANTX_LIB_PATH."
            )
        ham_list = [{"coeff": float(c),
                     "pauli": _parse_pauli_label(p, n_qubits)}
                    for c, p in hamiltonian_terms]
        jop_list = [{"type": j.kind, "qubit": j.qubit, "rate": float(j.rate)}
                    for j in jump_operators]
        prob = {
            "n_qubits":          n_qubits,
            "hamiltonian_terms": ham_list,
            "jump_operators":    jop_list,
        }
        raw = _sparse_steady_raw(sparse_lib, prob)
        if "error" in raw:
            raise LindbladError(f"Sparse steady-state error: {raw['error']}")
        return raw


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _parse_lindblad_frame(d: dict[str, Any]) -> LindbladFrame:
    return LindbladFrame(
        step=int(d.get("step", 0)),
        time=float(d.get("time", 0.0)),
        purity=float(d.get("purity", 0.0)),
        entropy=float(d.get("entropy", 0.0)),
        coherence=float(d.get("coherence", 0.0)),   # 0.0 for sparse backend
        populations=[float(p) for p in d.get("populations", [])],
    )


def _parse_entanglement_frame(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "time": float(d.get("time", 0.0)),
        "entropy_a": float(d.get("entropy_a", 0.0)),
        "mutual_info": float(d.get("mutual_info", 0.0)),
        "log_negativity": float(d.get("log_negativity", 0.0)),
        "concurrence": float(d.get("concurrence", 0.0)),
    }


def _parse_result(raw: dict[str, Any], n_qubits: int,
                  total_time: float, dt: float) -> LindbladResult:
    """Convert raw Engine.run() output to a LindbladResult."""
    if not raw.get("ok", True):
        msg = raw.get("error_message") or raw.get("error") or "backend reported failure"
        raise LindbladError(f"Lindblad backend error: {msg}")

    raw_frames: list[dict] = raw.get("lindblad_frames", [])
    frames = [_parse_lindblad_frame(f) for f in raw_frames]

    if not frames:
        raise LindbladError(
            "Lindblad backend returned no frames — check that the native "
            "library is built with the Lindblad backend enabled."
        )

    times      = [f.time for f in frames]
    purity     = [f.purity for f in frames]
    entropy    = [f.entropy for f in frames]
    coherence  = [f.coherence for f in frames]
    populations = [f.populations for f in frames]

    last = frames[-1]

    # ── Entanglement trace (optional) ──────────────────────────────────────
    ent_trace: EntanglementTrace | None = None
    raw_ent = raw.get("entanglement_frames", [])
    if raw_ent:
        ef_list = [_parse_entanglement_frame(d) for d in raw_ent]
        ent_trace = EntanglementTrace(
            times=[e["time"] for e in ef_list],
            entropy_a=[e["entropy_a"] for e in ef_list],
            mutual_info=[e["mutual_info"] for e in ef_list],
            log_negativity=[e["log_negativity"] for e in ef_list],
            concurrence=[e["concurrence"] for e in ef_list],
        )

    return LindbladResult(
        times=times,
        purity=purity,
        entropy=entropy,
        coherence=coherence,
        populations=populations,
        final_purity=last.purity,
        final_entropy=last.entropy,
        final_populations=last.populations,
        frames=frames,
        entanglement=ent_trace,
        n_qubits=n_qubits,
        total_time=total_time,
        dt=dt,
    )
