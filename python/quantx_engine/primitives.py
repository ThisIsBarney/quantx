"""Estimator + Sampler hardware-agnostic primitives — M85.

Provides:

- :class:`SparsePauliOp`: lightweight observable specification (Pauli tensor
  products with complex coefficients).
- :class:`Estimator`: compute ⟨ψ|O|ψ⟩ expectation values for circuits and
  observables, with optional noise and shot sampling.
- :class:`Sampler`: sample quasi-probability distributions from circuits.
- :class:`EstimatorResult`, :class:`SamplerResult`: typed result containers.

Both primitives use a pure-NumPy statevector simulator (noiseless default)
or a density-matrix simulator when a :class:`~quantx_engine.noise.NoiseModel`
is supplied.  No C++ engine binary is required.

Qubit ordering convention (same as Qiskit): qubit 0 is the **least-significant
bit** of the statevector index.  Pauli strings are written MSB-first, so
``"ZI"`` means Z on qubit 1, I on qubit 0 (matching the index ordering of the
computational basis ket ``|q1 q0⟩``).

Quick reference::

    from quantx_engine.primitives import Estimator, Sampler, SparsePauliOp

    bell = {
        "n_qubits": 2,
        "operations": [
            {"name": "h",  "qubits": [0]},
            {"name": "cx", "qubits": [0, 1]},
        ],
    }

    # Exact expectation value  ⟨Bell|ZZ|Bell⟩ = 1
    result = Estimator().run([bell], [SparsePauliOp("ZZ")])
    print(result.values)  # [1.]

    # Probability distribution (exact)
    sresult = Sampler().run([bell])
    print(sresult.quasi_dists[0])  # {0: 0.5, 3: 0.5}  (|00⟩ and |11⟩)
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

__all__ = [
    "PrimitiveError",
    "SparsePauliOp",
    "EstimatorResult",
    "SamplerResult",
    "Estimator",
    "Sampler",
]


# ===========================================================================
# Exception
# ===========================================================================

class PrimitiveError(ValueError):
    """Raised for invalid inputs to Estimator or Sampler."""


def _require_numpy() -> None:
    if not _HAS_NUMPY:  # pragma: no cover
        raise PrimitiveError("NumPy is required for primitives.")


# ===========================================================================
# SparsePauliOp — observable specification
# ===========================================================================

class SparsePauliOp:
    """Lightweight sparse Pauli observable.

    Parameters
    ----------
    data:
        One of:

        - A Pauli string such as ``"IZZ"`` (implicit coefficient 1.0).
        - A list of ``(pauli_str, coeff)`` tuples.
        - Another :class:`SparsePauliOp` (copy).

    Examples
    --------
    ::

        SparsePauliOp("ZZ")
        SparsePauliOp([("ZZ", 0.5), ("XX", -0.5)])
    """

    def __init__(self, data) -> None:
        _require_numpy()
        if isinstance(data, SparsePauliOp):
            self._terms: list[tuple[str, complex]] = list(data._terms)
        elif isinstance(data, str):
            _validate_pauli_str(data)
            self._terms = [(data.upper(), 1.0 + 0j)]
        elif isinstance(data, (list, tuple)):
            self._terms = []
            for item in data:
                if (isinstance(item, (list, tuple)) and len(item) == 2
                        and isinstance(item[0], str)):
                    s, c = str(item[0]).upper(), complex(item[1])
                    _validate_pauli_str(s)
                    self._terms.append((s, c))
                else:
                    raise PrimitiveError(
                        f"SparsePauliOp list items must be (pauli_str, coeff) pairs, "
                        f"got {item!r}."
                    )
            if not self._terms:
                raise PrimitiveError("SparsePauliOp must have at least one term.")
        else:
            raise PrimitiveError(
                f"SparsePauliOp: unsupported type {type(data).__name__}. "
                "Pass a Pauli string, list of (str, coeff) pairs, or SparsePauliOp."
            )
        lens = {len(s) for s, _ in self._terms}
        if len(lens) > 1:
            raise PrimitiveError(
                f"All Pauli strings must have the same length; got lengths {sorted(lens)}."
            )

    @property
    def n_qubits(self) -> int:
        """Number of qubits this observable acts on."""
        return len(self._terms[0][0])

    @property
    def terms(self) -> list[tuple[str, complex]]:
        """List of (pauli_str, coeff) pairs."""
        return list(self._terms)

    @classmethod
    def from_pauli_hamiltonian(cls, hamiltonian) -> "SparsePauliOp":
        """Convert a :class:`~quantx_engine.chemistry.PauliHamiltonian` to
        :class:`SparsePauliOp`.

        The Pauli string is built MSB-first: index 0 = qubit n-1.
        """
        n = hamiltonian.n_qubits
        terms: list[tuple[str, complex]] = []
        for t in hamiltonian.terms:
            pauli = "".join(t.ops.get(n - 1 - i, "I") for i in range(n))
            terms.append((pauli, complex(t.coeff)))
        return cls(terms)

    def __repr__(self) -> str:
        return f"SparsePauliOp({self._terms!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SparsePauliOp):
            return NotImplemented
        return self._terms == other._terms


def _validate_pauli_str(s: str) -> None:
    invalid = set(s.upper()) - {"I", "X", "Y", "Z"}
    if invalid:
        raise PrimitiveError(
            f"Invalid characters in Pauli string {s!r}: {sorted(invalid)}."
        )
    if not s:
        raise PrimitiveError("Pauli string must not be empty.")


# ===========================================================================
# Gate matrix library
# ===========================================================================

_GATE_CACHE: dict = {}


def _gate_matrices() -> dict:
    global _GATE_CACHE
    if _GATE_CACHE:
        return _GATE_CACHE
    sq2 = math.sqrt(2)
    _GATE_CACHE = {
        # --- single-qubit fixed ---
        "id": np.eye(2, dtype=complex),
        "i":  np.eye(2, dtype=complex),
        "x":  np.array([[0, 1], [1, 0]], dtype=complex),
        "y":  np.array([[0, -1j], [1j, 0]], dtype=complex),
        "z":  np.array([[1, 0], [0, -1]], dtype=complex),
        "h":  np.array([[1, 1], [1, -1]], dtype=complex) / sq2,
        "s":  np.array([[1, 0], [0, 1j]], dtype=complex),
        "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
        "t":  np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=complex),
        "tdg": np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=complex),
        "sx":  np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex) / 2,
        "sxdg": np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=complex) / 2,
        # --- two-qubit fixed (qubit order: first listed = control/first gate qubit) ---
        "cx":   np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                           [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
        "cnot": np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                           [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
        "cy":   np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                           [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=complex),
        "cz":   np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                           [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex),
        "swap": np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                           [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex),
        "iswap": np.array([[1, 0, 0, 0], [0, 0, 1j, 0],
                            [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex),
    }
    return _GATE_CACHE


def _get_gate_matrix(name: str, params: list | None) -> "np.ndarray":
    """Return the unitary matrix for a named gate, optionally parametric."""
    key = name.lower()
    if params:
        theta = float(params[0])
        if key == "rx":
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
        if key == "ry":
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            return np.array([[c, -s], [s, c]], dtype=complex)
        if key == "rz":
            return np.array([[np.exp(-0.5j * theta), 0],
                              [0, np.exp(0.5j * theta)]], dtype=complex)
        if key in ("p", "phase", "u1"):
            return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)
        if key == "u2" and len(params) >= 2:
            phi, lam = float(params[0]), float(params[1])
            sq2 = math.sqrt(2)
            return np.array([
                [1, -np.exp(1j * lam)],
                [np.exp(1j * phi), np.exp(1j * (phi + lam))],
            ], dtype=complex) / sq2
        if key in ("u3", "u") and len(params) >= 3:
            th, phi, lam = float(params[0]), float(params[1]), float(params[2])
            c, s = math.cos(th / 2), math.sin(th / 2)
            return np.array([
                [c, -np.exp(1j * lam) * s],
                [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c],
            ], dtype=complex)
        if key == "rxx":
            c, s = math.cos(theta / 2), math.sin(theta / 2)
            return np.array([
                [c, 0, 0, -1j * s],
                [0, c, -1j * s, 0],
                [0, -1j * s, c, 0],
                [-1j * s, 0, 0, c],
            ], dtype=complex)
        if key == "rzz":
            p, m = np.exp(-0.5j * theta), np.exp(0.5j * theta)
            return np.diag([p, m, m, p]).astype(complex)
    matrices = _gate_matrices()
    if key in matrices:
        return matrices[key]
    raise PrimitiveError(
        f"Unknown gate {name!r}. Fixed gates: {sorted(matrices)}; "
        "parametric: rx, ry, rz, p/phase/u1, u2, u3/u, rxx, rzz."
    )


# ===========================================================================
# Statevector simulator
# ===========================================================================

def _apply_gate_sv(
    sv: "np.ndarray",
    U: "np.ndarray",
    qubits: list[int],
) -> "np.ndarray":
    """Apply gate U to a statevector at specified qubit positions.

    Qubit ordering: qubit 0 = LSB of ``sv`` index.  The reshaped tensor has
    axis ``j`` corresponding to qubit ``n-1-j`` (MSB-first, C order), so we
    map ``qubit q → tensor axis n-1-q``.

    Parameters
    ----------
    sv:
        Statevector of length ``2**n``.
    U:
        Gate matrix of shape ``(2**nq, 2**nq)``.
    qubits:
        Target qubit indices (qubit 0 = LSB).

    Returns
    -------
    New statevector of same shape as ``sv``.
    """
    n = int(round(math.log2(sv.shape[0])))
    nq = len(qubits)
    # tensor axis for qubit q = n-1-q  (C-order reshape puts MSB at axis 0)
    axes = [n - 1 - q for q in qubits]

    sv = sv.reshape([2] * n)
    U = U.reshape([2] * nq + [2] * nq)

    # Contract U's "ket" half (axes nq..2nq-1) with sv's qubit axes
    sv = np.tensordot(U, sv, axes=(list(range(nq, 2 * nq)), axes))
    # Result shape: [2]*nq (U bra) + [2]*(n-nq) (sv remaining axes, in original order)

    # Rebuild the correct axis ordering so that each qubit ends up at its
    # original tensor axis.
    sv_remaining = [i for i in range(n) if i not in axes]
    # target_pos[cur_axis] = final tensor position
    target_pos = [0] * n
    for i, a in enumerate(axes):
        target_pos[i] = a
    for i, a in enumerate(sv_remaining):
        target_pos[nq + i] = a
    # inv_perm[final_pos] = cur_axis  (numpy's np.transpose convention)
    inv_perm = [0] * n
    for cur, final in enumerate(target_pos):
        inv_perm[final] = cur

    return np.transpose(sv, inv_perm).reshape(-1)


def _resolve_params(
    raw: list,
    param_values: dict | None,
) -> list[float]:
    """Resolve a list of raw parameter values (floats or string names)."""
    out: list[float] = []
    for p in raw:
        if isinstance(p, str):
            if param_values is None or p not in param_values:
                raise PrimitiveError(
                    f"Parameter {p!r} not found in parameter_values dict."
                )
            out.append(float(param_values[p]))
        else:
            out.append(float(p))
    return out


def _sv_simulate(
    ops: list[dict],
    n_qubits: int,
    param_values: dict | None = None,
) -> "np.ndarray":
    """Simulate a circuit starting from |0…0⟩ and return the statevector."""
    sv = np.zeros(1 << n_qubits, dtype=complex)
    sv[0] = 1.0
    for op in ops:
        name: str = op.get("name", "").lower()
        if name in ("barrier", "measure", "reset", ""):
            continue
        raw_params = op.get("params", op.get("parameters", []))
        params = _resolve_params(raw_params, param_values)
        qubits: list[int] = op.get("qubits", op.get("targets", []))
        U = _get_gate_matrix(name, params if params else None)
        sv = _apply_gate_sv(sv, U, qubits)
    return sv


# ===========================================================================
# Density-matrix helpers (for noisy simulation)
# ===========================================================================

def _embed_gate(U: "np.ndarray", qubits: list[int], n: int) -> "np.ndarray":
    """Embed gate U acting on `qubits` into the full n-qubit Hilbert space.

    Returns a ``(2**n, 2**n)`` unitary matrix.
    """
    d = 1 << n
    columns = np.eye(d, dtype=complex)
    result = np.empty((d, d), dtype=complex)
    for j in range(d):
        result[:, j] = _apply_gate_sv(columns[j], U, qubits)
    return result


def _apply_gate_dm(
    rho: "np.ndarray",
    U: "np.ndarray",
    qubits: list[int],
    n: int,
) -> "np.ndarray":
    """Apply gate U to density matrix: ρ → U_full ρ U_full†."""
    U_full = _embed_gate(U, qubits, n)
    return U_full @ rho @ U_full.conj().T


def _depolarize_local(
    rho: "np.ndarray",
    p: float,
    qubits: list[int],
    n: int,
) -> "np.ndarray":
    """Apply independent single-qubit depolarizing noise to each qubit in
    `qubits`.

    Channel: ``E(ρ) = (1 − 3p/4)ρ + (p/4)(XρX + YρY + ZρZ)``
    which corresponds to per-qubit fidelity ``1 − p``.
    """
    if p <= 0.0:
        return rho
    mats = _gate_matrices()
    for q in qubits:
        fac = 1.0 - 3.0 * p / 4.0
        new_rho = fac * rho
        for P_name in ("x", "y", "z"):
            P_full = _embed_gate(mats[P_name], [q], n)
            new_rho = new_rho + (p / 4.0) * (P_full @ rho @ P_full.conj().T)
        rho = new_rho
    return rho


def _dm_simulate(
    ops: list[dict],
    n_qubits: int,
    noise_model,
    param_values: dict | None = None,
) -> "np.ndarray":
    """Simulate a circuit with noise using a density matrix.

    Applies single-qubit depolarizing noise after every gate; the noise
    probability is taken from the noise model's ``depolarizing_prob`` (or
    ``amplitude_damping_gamma``) field.
    """
    d = 1 << n_qubits
    rho = np.zeros((d, d), dtype=complex)
    rho[0, 0] = 1.0

    p_noise = 0.0
    if noise_model is not None:
        nd = noise_model.to_dict()
        p_noise = float(
            nd.get("depolarizing_prob", nd.get("amplitude_damping_gamma", 0.0))
        )

    for op in ops:
        name: str = op.get("name", "").lower()
        if name in ("barrier", "measure", "reset", ""):
            continue
        raw_params = op.get("params", op.get("parameters", []))
        params = _resolve_params(raw_params, param_values)
        qubits: list[int] = op.get("qubits", op.get("targets", []))
        U = _get_gate_matrix(name, params if params else None)
        rho = _apply_gate_dm(rho, U, qubits, n_qubits)
        if p_noise > 0.0:
            rho = _depolarize_local(rho, p_noise, qubits, n_qubits)

    return rho


# ===========================================================================
# Pauli → matrix utilities
# ===========================================================================

_PAULI1: dict | None = None


def _pauli1() -> dict:
    global _PAULI1
    if _PAULI1 is None:
        m = _gate_matrices()
        _PAULI1 = {"I": m["id"], "X": m["x"], "Y": m["y"], "Z": m["z"]}
    return _PAULI1


def _sparse_pauli_to_matrix(obs: SparsePauliOp) -> "np.ndarray":
    """Convert a SparsePauliOp to its dense matrix representation.

    The Pauli string is MSB-first: ``pauli_str[0]`` acts on qubit n-1,
    ``pauli_str[-1]`` acts on qubit 0.  The resulting matrix is in the
    standard computational basis with qubit 0 = LSB.
    """
    n = obs.n_qubits
    d = 1 << n
    mat = np.zeros((d, d), dtype=complex)
    p1 = _pauli1()
    for pauli_str, coeff in obs.terms:
        P = p1[pauli_str[0]]
        for c in pauli_str[1:]:
            P = np.kron(P, p1[c])
        mat += coeff * P
    return mat


# ===========================================================================
# Circuit normalisation
# ===========================================================================

def _normalize_circuit(circuit) -> tuple[list[dict], int]:
    """Accept a circuit dict or a plain ops list; return (ops, n_qubits)."""
    if isinstance(circuit, dict):
        ops = circuit.get(
            "operations",
            circuit.get("ops", circuit.get("gates", [])),
        )
        n = circuit.get(
            "n_qubits",
            circuit.get("num_qubits", circuit.get("qubits", None)),
        )
        if n is None:
            n = _infer_n_qubits(ops)
        return ops, int(n)
    if isinstance(circuit, (list, tuple)):
        ops = list(circuit)
        return ops, _infer_n_qubits(ops)
    raise PrimitiveError(
        f"Unsupported circuit type {type(circuit).__name__}. "
        "Pass a dict with 'n_qubits' + 'operations', or a plain list of gate dicts."
    )


def _infer_n_qubits(ops: list[dict]) -> int:
    max_q = -1
    for op in ops:
        for q in op.get("qubits", op.get("targets", [])):
            if q > max_q:
                max_q = q
    if max_q < 0:
        raise PrimitiveError(
            "Cannot infer n_qubits from an empty or qubit-less operation list. "
            "Provide n_qubits explicitly in the circuit dict."
        )
    return max_q + 1


def _normalize_observable(obs) -> SparsePauliOp:
    """Accept str / list / SparsePauliOp / PauliHamiltonian → SparsePauliOp."""
    if isinstance(obs, SparsePauliOp):
        return obs
    if isinstance(obs, str):
        return SparsePauliOp(obs)
    if isinstance(obs, (list, tuple)):
        return SparsePauliOp(list(obs))
    # Duck-type PauliHamiltonian
    if hasattr(obs, "terms") and hasattr(obs, "n_qubits"):
        return SparsePauliOp.from_pauli_hamiltonian(obs)
    raise PrimitiveError(
        f"Unsupported observable type {type(obs).__name__}. "
        "Use a Pauli string, list of (str, coeff), SparsePauliOp, or PauliHamiltonian."
    )


# ===========================================================================
# Result containers
# ===========================================================================

@dataclass
class EstimatorResult:
    """Result from :meth:`Estimator.run`.

    Attributes
    ----------
    values:
        Array of expectation values, shape ``(n,)``.
    stds:
        Standard deviation of each value.  Zero for exact (noiseless /
        infinite-shot) evaluation.
    metadata:
        List of per-pair metadata dicts.
    """
    values: "np.ndarray"
    stds: "np.ndarray"
    metadata: list = field(default_factory=list)


@dataclass
class SamplerResult:
    """Result from :meth:`Sampler.run`.

    Attributes
    ----------
    quasi_dists:
        List of quasi-probability dicts (one per circuit). Keys are
        bitstring integers (e.g. ``0b101 == 5`` for ``|101⟩``), values are
        quasi-probabilities (exact simulation → true probabilities ≥ 0).
    metadata:
        List of per-circuit metadata dicts.
    """
    quasi_dists: list
    metadata: list = field(default_factory=list)


# ===========================================================================
# Estimator
# ===========================================================================

class Estimator:
    """Hardware-agnostic expectation-value estimator.

    Computes ``⟨ψ|O|ψ⟩`` for each (circuit, observable) pair using a
    pure-NumPy statevector (noiseless) or density-matrix (noisy) simulator.

    Parameters
    ----------
    noise_model:
        Optional :class:`~quantx_engine.noise.NoiseModel` instance.  When
        provided a density-matrix simulation is used; otherwise the
        computation is exact.
    shots:
        When positive, draw ``shots`` measurement outcomes and estimate
        the expectation value empirically (adds shot noise).  ``None`` or
        ``0`` → exact value.
    seed:
        Random seed for reproducible shot sampling.

    Example
    -------
    ::

        from quantx_engine.primitives import Estimator, SparsePauliOp

        bell = {
            "n_qubits": 2,
            "operations": [
                {"name": "h",  "qubits": [0]},
                {"name": "cx", "qubits": [0, 1]},
            ],
        }
        result = Estimator().run([bell], ["ZZ"])
        # result.values ≈ [1.0]
    """

    def __init__(
        self,
        noise_model=None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        _require_numpy()
        self.noise_model = noise_model
        self.shots = shots
        self.seed = seed

    def run(
        self,
        circuits,
        observables,
        parameter_values=None,
    ) -> EstimatorResult:
        """Evaluate expectation values for (circuit, observable) pairs.

        Parameters
        ----------
        circuits:
            Single circuit or list of circuits.
        observables:
            Single observable or list of observables (Pauli strings,
            ``SparsePauliOp``, or ``PauliHamiltonian``).  Length must match
            ``circuits`` or be 1 (broadcast).
        parameter_values:
            Optional list of parameter-binding dicts (one per circuit),
            mapping parameter name → float.  May also be a single dict
            (applied to all circuits).

        Returns
        -------
        EstimatorResult
        """
        _require_numpy()

        # ── normalise inputs ────────────────────────────────────────────────
        if not isinstance(circuits, (list, tuple)):
            circuits = [circuits]
        if not isinstance(observables, (list, tuple)):
            observables = [observables]
        if len(circuits) == 1 and len(observables) > 1:
            circuits = circuits * len(observables)
        if len(observables) == 1 and len(circuits) > 1:
            observables = observables * len(circuits)
        n_jobs = len(circuits)
        if len(observables) != n_jobs:
            raise PrimitiveError(
                f"circuits ({len(circuits)}) and observables ({len(observables)}) "
                "must have the same length (or one must be length 1)."
            )
        if parameter_values is None:
            param_list = [None] * n_jobs
        elif isinstance(parameter_values, dict):
            param_list = [parameter_values] * n_jobs
        else:
            param_list = list(parameter_values)
            if len(param_list) < n_jobs:
                param_list += [None] * (n_jobs - len(param_list))

        # ── evaluate ────────────────────────────────────────────────────────
        rng = np.random.default_rng(self.seed)
        values = np.zeros(n_jobs, dtype=float)
        stds   = np.zeros(n_jobs, dtype=float)
        metadata: list[dict] = []

        for i, (circuit, observable) in enumerate(zip(circuits, observables)):
            ops, nq = _normalize_circuit(circuit)
            obs     = _normalize_observable(observable)
            pv      = param_list[i]

            if obs.n_qubits != nq:
                raise PrimitiveError(
                    f"Circuit {i} has {nq} qubits but observable has "
                    f"{obs.n_qubits} qubits."
                )

            obs_mat = _sparse_pauli_to_matrix(obs)

            # ── simulate state ───────────────────────────────────────────────
            noisy = self.noise_model is not None
            if noisy:
                rho = _dm_simulate(ops, nq, self.noise_model, pv)
                exact_val = float(np.real(np.trace(rho @ obs_mat)))
            else:
                sv = _sv_simulate(ops, nq, pv)
                exact_val = float(np.real(sv.conj() @ (obs_mat @ sv)))

            # ── shots (sample in observable eigenbasis) ───────────────────────
            if self.shots and self.shots > 0:
                evals, evecs = np.linalg.eigh(obs_mat)   # real eigenvalues
                if noisy:
                    probs_obs = np.real(np.array([
                        float((evecs[:, k].conj() @ rho @ evecs[:, k]))
                        for k in range(len(evals))
                    ]))
                else:
                    amps = evecs.conj().T @ sv
                    probs_obs = np.abs(amps) ** 2

                probs_obs = np.maximum(probs_obs, 0.0)
                probs_obs /= probs_obs.sum()
                counts = rng.multinomial(self.shots, probs_obs)
                sampled = float(np.dot(counts, evals)) / self.shots
                # Bessel-corrected std of the mean
                outcomes = np.repeat(evals, counts)
                shot_std = float(outcomes.std(ddof=1)) / math.sqrt(self.shots) \
                    if self.shots > 1 else 0.0
                values[i] = sampled
                stds[i]   = shot_std
            else:
                values[i] = exact_val
                stds[i]   = 0.0

            metadata.append({"circuit_index": i, "n_qubits": nq})

        return EstimatorResult(values=values, stds=stds, metadata=metadata)


# ===========================================================================
# Sampler
# ===========================================================================

class Sampler:
    """Hardware-agnostic circuit sampler.

    Runs circuits and returns quasi-probability distributions over bitstrings.

    Parameters
    ----------
    noise_model:
        Optional noise model (density-matrix simulation when set).
    shots:
        Number of measurement shots.  ``None`` or ``0`` → return exact
        probability distribution (no shot noise).
    seed:
        Random seed for reproducibility.

    Example
    -------
    ::

        from quantx_engine.primitives import Sampler

        circuit = {
            "n_qubits": 1,
            "operations": [{"name": "h", "qubits": [0]}],
        }
        result = Sampler(shots=1000, seed=42).run([circuit])
        # result.quasi_dists[0] ≈ {0: 0.5, 1: 0.5}
    """

    def __init__(
        self,
        noise_model=None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        _require_numpy()
        self.noise_model = noise_model
        self.shots = shots
        self.seed = seed

    def run(
        self,
        circuits,
        parameter_values=None,
    ) -> SamplerResult:
        """Sample from circuits.

        Parameters
        ----------
        circuits:
            Single circuit or list of circuits.
        parameter_values:
            Optional list of parameter-binding dicts (one per circuit).

        Returns
        -------
        SamplerResult
        """
        _require_numpy()

        if not isinstance(circuits, (list, tuple)):
            circuits = [circuits]
        n_jobs = len(circuits)

        if parameter_values is None:
            param_list = [None] * n_jobs
        elif isinstance(parameter_values, dict):
            param_list = [parameter_values] * n_jobs
        else:
            param_list = list(parameter_values)
            if len(param_list) < n_jobs:
                param_list += [None] * (n_jobs - len(param_list))

        rng = np.random.default_rng(self.seed)
        quasi_dists: list[dict] = []
        metadata: list[dict] = []

        for i, circuit in enumerate(circuits):
            ops, nq = _normalize_circuit(circuit)
            pv = param_list[i]

            noisy = self.noise_model is not None
            if noisy:
                rho = _dm_simulate(ops, nq, self.noise_model, pv)
                probs = np.real(np.diag(rho))
            else:
                sv = _sv_simulate(ops, nq, pv)
                probs = np.abs(sv) ** 2

            probs = np.maximum(probs, 0.0)
            total = probs.sum()
            if total > 0:
                probs = probs / total

            if self.shots and self.shots > 0:
                counts = rng.multinomial(self.shots, probs)
                qd = {
                    int(j): int(counts[j]) / self.shots
                    for j in range(len(counts))
                    if counts[j] > 0
                }
            else:
                qd = {
                    int(j): float(probs[j])
                    for j in range(len(probs))
                    if probs[j] > 1e-15
                }

            quasi_dists.append(qd)
            metadata.append({
                "circuit_index": i,
                "n_qubits": nq,
                "shots": self.shots,
            })

        return SamplerResult(quasi_dists=quasi_dists, metadata=metadata)
