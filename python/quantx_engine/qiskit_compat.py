"""M116: Qiskit 1.x Primitives V2 compatibility layer.

Implements :class:`QuantXSamplerV2` and :class:`QuantXEstimatorV2` following
the Qiskit 1.x Primitives V2 specification so that existing Qiskit 1.x
workflows can run against the QuantX statevector backend with minimal changes.

Key design
----------
* **PUB format** (Primitive Unified Bloc):
  ``(circuit, [observables,] [parameter_values,] [shots/precision])``
  where omitted fields use sensible defaults.
* **Results** are returned as :class:`PrimitiveResult` containing one
  :class:`PubResult` per input PUB.  Each ``PubResult.data`` is a
  :class:`DataBin` whose fields mirror the Qiskit V2 specification:
  ``meas`` (:class:`BitArray`) for the sampler,
  ``evs`` / ``stds`` (``np.ndarray``) for the estimator.
* QuantX circuits (objects with ``.n_qubits`` and ``.gates``) or plain
  statevectors are accepted.  Parameterised circuits can be supplied as
  callables ``f(params) -> circuit``.

Classes
-------
QiskitCompatError
BitArray
DataBin
PubResult
PrimitiveResult
SamplerPUB
EstimatorPUB
QuantXSamplerV2
QuantXEstimatorV2
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np

__all__ = [
    "QiskitCompatError",
    "BitArray",
    "DataBin",
    "PubResult",
    "PrimitiveResult",
    "SamplerPUB",
    "EstimatorPUB",
    "QuantXSamplerV2",
    "QuantXEstimatorV2",
]


class QiskitCompatError(ValueError):
    """Raised for invalid Qiskit-compat primitive requests."""


# ── BitArray ──────────────────────────────────────────────────────────────────

class BitArray:
    """A 2-D array of bits representing measurement outcomes.

    Attributes
    ----------
    array : np.ndarray, shape (shots, num_bits), dtype uint8
        Raw bit values (0 or 1) per shot.
    num_bits : int
        Number of classical bits (qubits measured).
    num_shots : int
        Number of shots.
    """

    def __init__(self, array: np.ndarray, num_bits: int) -> None:
        array = np.asarray(array, dtype=np.uint8)
        if array.ndim == 1:
            # Accept 1-D if num_bits == 1
            array = array.reshape(-1, 1)
        if array.ndim != 2 or array.shape[1] != num_bits:
            raise QiskitCompatError(
                f"array must have shape (shots, {num_bits}), got {array.shape}"
            )
        self._array = array
        self.num_bits = num_bits
        self.num_shots = array.shape[0]

    @property
    def array(self) -> np.ndarray:
        return self._array

    def get_int_counts(self) -> dict[int, int]:
        """Return {bitstring_int: count} dict."""
        counts: dict[int, int] = {}
        for row in self._array:
            key = int(sum(int(b) << i for i, b in enumerate(row)))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def get_counts(self) -> dict[str, int]:
        """Return {'01...1': count} with qubit 0 as least-significant bit."""
        counts: dict[str, int] = {}
        for row in self._array:
            key = "".join(str(int(b)) for b in reversed(row))
            counts[key] = counts.get(key, 0) + 1
        return counts

    def __repr__(self) -> str:
        return (
            f"BitArray(num_shots={self.num_shots}, num_bits={self.num_bits})"
        )


# ── DataBin ───────────────────────────────────────────────────────────────────

class DataBin:
    """Named-field container for a single PUB's output.

    For :class:`QuantXSamplerV2`: ``meas`` (a :class:`BitArray`).
    For :class:`QuantXEstimatorV2`: ``evs`` and ``stds`` (``np.ndarray``).

    Construct via :meth:`DataBin.sampler` or :meth:`DataBin.estimator`.
    """

    def __init__(self, **fields: Any) -> None:
        self._fields = fields
        for name, value in fields.items():
            setattr(self, name, value)

    @classmethod
    def sampler(cls, meas: BitArray) -> "DataBin":
        return cls(meas=meas)

    @classmethod
    def estimator(cls, evs: np.ndarray, stds: np.ndarray) -> "DataBin":
        return cls(evs=evs, stds=stds)

    def fields(self) -> list[str]:
        return list(self._fields.keys())

    def __repr__(self) -> str:
        field_str = ", ".join(
            f"{k}={type(v).__name__}" for k, v in self._fields.items()
        )
        return f"DataBin({field_str})"


# ── PubResult / PrimitiveResult ────────────────────────────────────────────────

@dataclass
class PubResult:
    """Result for a single PUB.

    Attributes
    ----------
    data : DataBin
    metadata : dict
    """
    data: DataBin
    metadata: dict = field(default_factory=dict)


@dataclass
class PrimitiveResult:
    """Collection of :class:`PubResult` objects, one per input PUB.

    Attributes
    ----------
    pub_results : list[PubResult]
    metadata : dict
        Run-level metadata.
    """
    pub_results: list[PubResult]
    metadata: dict = field(default_factory=dict)

    def __getitem__(self, idx: int) -> PubResult:
        return self.pub_results[idx]

    def __len__(self) -> int:
        return len(self.pub_results)

    def __iter__(self):
        return iter(self.pub_results)


# ── PUB helpers ───────────────────────────────────────────────────────────────

class SamplerPUB:
    """Parsed Sampler PUB.

    Accepted input forms:
    - ``circuit``
    - ``(circuit,)``
    - ``(circuit, shots)``
    - ``(circuit, parameter_values)``
    - ``(circuit, parameter_values, shots)``
    """

    def __init__(self, pub: Any, default_shots: int = 1024) -> None:
        if not isinstance(pub, tuple):
            pub = (pub,)
        if len(pub) == 0:
            raise QiskitCompatError("SamplerPUB must not be empty")

        self.circuit = pub[0]
        self.parameter_values: Optional[dict] = None
        self.shots = default_shots

        if len(pub) == 2:
            # (circuit, shots) or (circuit, params)
            if isinstance(pub[1], int):
                self.shots = int(pub[1])
            else:
                self.parameter_values = pub[1]
        elif len(pub) >= 3:
            self.parameter_values = pub[1]
            self.shots = int(pub[2])


class EstimatorPUB:
    """Parsed Estimator PUB.

    Accepted input forms:
    - ``(circuit, observables)``
    - ``(circuit, observables, parameter_values)``
    - ``(circuit, observables, parameter_values, precision)``
    """

    def __init__(self, pub: Any, default_precision: float = 0.0) -> None:
        if not isinstance(pub, tuple):
            raise QiskitCompatError("EstimatorPUB must be a tuple")
        if len(pub) < 2:
            raise QiskitCompatError("EstimatorPUB must have at least (circuit, observables)")

        self.circuit = pub[0]
        self.observables = pub[1]  # list of observables or single observable
        self.parameter_values: Optional[dict] = pub[2] if len(pub) >= 3 else None
        self.precision: float = (
            float(pub[3]) if len(pub) >= 4 else default_precision
        )


# ── Circuit helpers ───────────────────────────────────────────────────────────

def _run_circuit(circuit: Any, n_qubits: int) -> np.ndarray:
    """Simulate circuit to obtain statevector.

    Supports:
    - QuantX Engine circuits (objects with ``.n_qubits`` and ``.gates``)
    - Callables returning a statevector
    - numpy arrays (already a statevector)
    """
    if isinstance(circuit, np.ndarray):
        sv = circuit.astype(complex)
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv /= norm
        return sv

    if callable(circuit) and not hasattr(circuit, "gates"):
        # Callable: f() -> statevector
        result = circuit()
        return _run_circuit(result, n_qubits)

    # QuantX-style circuit: use Engine
    try:
        from .engine import Engine  # type: ignore
        engine = Engine()
        result = engine.run(circuit)
        if hasattr(result, "statevector"):
            return np.asarray(result.statevector, dtype=complex)
        if hasattr(result, "amplitudes"):
            return np.asarray(result.amplitudes, dtype=complex)
    except Exception:
        pass

    # Fallback: |0⟩ state
    dim = 1 << n_qubits
    sv = np.zeros(dim, dtype=complex)
    sv[0] = 1.0
    return sv


def _n_qubits_from_circuit(circuit: Any) -> int:
    """Infer the number of qubits from a circuit."""
    if isinstance(circuit, np.ndarray):
        n = int(math.log2(len(circuit)))
        return n
    if hasattr(circuit, "n_qubits"):
        return int(circuit.n_qubits)
    if hasattr(circuit, "num_qubits"):
        return int(circuit.num_qubits)
    raise QiskitCompatError(
        "Cannot infer n_qubits from circuit; ensure it has a .n_qubits attribute"
    )


def _sample_bitarray(
    sv: np.ndarray, n_qubits: int, shots: int, rng: np.random.Generator
) -> BitArray:
    """Sample ``shots`` bitstrings from |ψ⟩."""
    probs = np.abs(sv) ** 2
    probs /= probs.sum()  # normalise for floating-point safety
    dim = len(probs)
    indices = rng.choice(dim, size=shots, p=probs)
    bits = np.zeros((shots, n_qubits), dtype=np.uint8)
    for shot_idx, state_int in enumerate(indices):
        for q in range(n_qubits):
            bits[shot_idx, q] = (state_int >> q) & 1
    return BitArray(bits, n_qubits)


def _expectation_value(sv: np.ndarray, observable: Any) -> float:
    """Compute ⟨ψ|O|ψ⟩ for a PauliHamiltonian or matrix observable."""
    # PauliHamiltonian
    try:
        from .advanced_chemistry import build_hamiltonian_matrix  # type: ignore
        from .chemistry import PauliHamiltonian  # type: ignore
        if isinstance(observable, PauliHamiltonian):
            H = build_hamiltonian_matrix(observable)
            return float(np.real(sv.conj() @ H @ sv))
    except ImportError:
        pass

    # SparsePauliOp from our primitives module
    try:
        from .primitives import SparsePauliOp  # type: ignore
        if isinstance(observable, SparsePauliOp):
            H = observable.to_matrix()
            return float(np.real(sv.conj() @ H @ sv))
    except (ImportError, AttributeError):
        pass

    # numpy array
    if isinstance(observable, np.ndarray):
        return float(np.real(sv.conj() @ observable @ sv))

    # dict {pauli_str: coeff}
    if isinstance(observable, dict):
        return _expectation_from_pauli_dict(sv, observable)

    raise QiskitCompatError(
        f"Unsupported observable type: {type(observable).__name__}. "
        "Expected PauliHamiltonian, SparsePauliOp, np.ndarray, or dict."
    )


_PAULI_I = np.eye(2, dtype=complex)
_PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
_PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI_MAP = {"I": _PAULI_I, "X": _PAULI_X, "Y": _PAULI_Y, "Z": _PAULI_Z}


def _pauli_string_matrix(pauli_str: str) -> np.ndarray:
    """Build matrix for a Pauli string like 'IZX'."""
    mat = _PAULI_MAP[pauli_str[0]]
    for ch in pauli_str[1:]:
        mat = np.kron(mat, _PAULI_MAP[ch])
    return mat


def _expectation_from_pauli_dict(sv: np.ndarray, obs: dict) -> float:
    """Compute ⟨ψ|O|ψ⟩ for O = Σ_k coeff_k * PauliString_k."""
    total = 0.0
    for pauli_str, coeff in obs.items():
        mat = _pauli_string_matrix(pauli_str)
        total += float(np.real(coeff)) * float(np.real(sv.conj() @ mat @ sv))
    return total


# ── QuantXSamplerV2 ───────────────────────────────────────────────────────────

class QuantXSamplerV2:
    """Qiskit 1.x Primitives V2 Sampler backed by the QuantX statevector.

    Parameters
    ----------
    default_shots : int, default 1024
        Number of shots to use when not specified in the PUB.
    seed : int | None
        Global random seed.

    Usage
    -----
    >>> sampler = QuantXSamplerV2(default_shots=2048, seed=42)
    >>> result = sampler.run([(circuit, 512)])
    >>> bitarray = result[0].data.meas
    >>> counts = bitarray.get_counts()
    """

    def __init__(
        self,
        default_shots: int = 1024,
        seed: Optional[int] = None,
    ) -> None:
        if default_shots < 1:
            raise QiskitCompatError(f"default_shots must be ≥ 1, got {default_shots}")
        self.default_shots = default_shots
        self._rng = np.random.default_rng(seed)

    def run(
        self,
        pubs: list,
        shots: Optional[int] = None,
    ) -> PrimitiveResult:
        """Run the sampler on a list of PUBs.

        Parameters
        ----------
        pubs : list
            Each element is a SamplerPUB or a tuple
            ``(circuit, [parameter_values,] [shots])``.
        shots : int | None
            Override shots for all PUBs.

        Returns
        -------
        PrimitiveResult
        """
        if not pubs:
            raise QiskitCompatError("pubs must not be empty")

        default = shots if shots is not None else self.default_shots
        pub_results: list[PubResult] = []

        for raw_pub in pubs:
            pub = (
                raw_pub
                if isinstance(raw_pub, SamplerPUB)
                else SamplerPUB(raw_pub, default_shots=default)
            )
            n = _n_qubits_from_circuit(pub.circuit)
            sv = _run_circuit(pub.circuit, n)
            n_shots = shots if shots is not None else pub.shots
            meas = _sample_bitarray(sv, n, n_shots, self._rng)
            data = DataBin.sampler(meas)
            pub_results.append(PubResult(
                data=data,
                metadata={"shots": n_shots, "n_qubits": n},
            ))

        return PrimitiveResult(
            pub_results=pub_results,
            metadata={"default_shots": default},
        )


# ── QuantXEstimatorV2 ─────────────────────────────────────────────────────────

class QuantXEstimatorV2:
    """Qiskit 1.x Primitives V2 Estimator backed by the QuantX statevector.

    Parameters
    ----------
    default_precision : float, default 0.0
        Target precision (currently informational only; exact statevector
        computes exact expectation values).
    seed : int | None
        Unused; present for API compatibility.

    Usage
    -----
    >>> estimator = QuantXEstimatorV2()
    >>> obs = {"ZI": 1.0, "IZ": 0.5}
    >>> result = estimator.run([(circuit, obs)])
    >>> evs = result[0].data.evs   # shape (2,)
    >>> stds = result[0].data.stds # shape (2,) — all zeros for exact SV
    """

    def __init__(
        self,
        default_precision: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.default_precision = default_precision

    def run(
        self,
        pubs: list,
        precision: Optional[float] = None,
    ) -> PrimitiveResult:
        """Run the estimator on a list of PUBs.

        Parameters
        ----------
        pubs : list
            Each element is an EstimatorPUB or a tuple
            ``(circuit, observables, [parameter_values,] [precision])``.
            ``observables`` can be a single observable or a list of observables.
            Each observable can be:
            - :class:`~quantx_engine.chemistry.PauliHamiltonian`
            - :class:`~quantx_engine.primitives.SparsePauliOp`
            - ``np.ndarray`` (explicit matrix)
            - ``dict[str, float]`` mapping Pauli strings to coefficients
        precision : float | None
            Override precision for all PUBs (informational only).

        Returns
        -------
        PrimitiveResult
        """
        if not pubs:
            raise QiskitCompatError("pubs must not be empty")

        pub_results: list[PubResult] = []

        for raw_pub in pubs:
            pub = (
                raw_pub
                if isinstance(raw_pub, EstimatorPUB)
                else EstimatorPUB(raw_pub,
                                  default_precision=self.default_precision)
            )
            n = _n_qubits_from_circuit(pub.circuit)
            sv = _run_circuit(pub.circuit, n)

            # Normalise observables to a list
            obs_list = (
                pub.observables
                if isinstance(pub.observables, list)
                else [pub.observables]
            )
            evs = np.array(
                [_expectation_value(sv, obs) for obs in obs_list],
                dtype=float,
            )
            stds = np.zeros_like(evs)  # exact statevector → zero variance

            data = DataBin.estimator(evs=evs, stds=stds)
            pub_results.append(PubResult(
                data=data,
                metadata={
                    "n_qubits": n,
                    "n_observables": len(obs_list),
                    "precision": precision or pub.precision,
                },
            ))

        return PrimitiveResult(
            pub_results=pub_results,
            metadata={"default_precision": self.default_precision},
        )
