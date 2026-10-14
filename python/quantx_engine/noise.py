"""
Noise model library for the QuantX engine.

Provides structured noise configuration objects that generate the JSON noise
payloads accepted by Engine.run_circuit() and Engine.run_qasm().

Noise fields are placed **on the circuit dict** (not in options):

    from quantx_engine.noise import DepolarizingNoise
    circuit = engine.parse_qasm(src, shots=1000)
    circuit.update(DepolarizingNoise(0.01).to_dict())   # add noise to circuit
    engine.run_circuit(circuit)

Supported noise models and their native backend mapping:
- DepolarizingNoise     — emits top-level ``depolarizing_prob`` (fully supported)
- AmplitudeDampingNoise — emits top-level ``amplitude_damping_gamma`` (fully supported)
- ThermalRelaxationNoise— emits ``noise={kind:t1_decay,...}`` for T1; T2 dephasing
                          is stored in ``noise_metadata`` (informational only — native
                          backend does not yet apply T2 dephasing)
- ReadoutNoise          — emits ``noise={kind:bit_flip,prob:p}`` (supported as
                          bit-flip channel, not measurement-specific readout)
- CompositeNoise        — combine models whose fields are disjoint: at most one
                          ``noise``-kind model, and any shared field raises

References
----------
.. [1] Nielsen, M. A. & Chuang, I. L. (2000). *Quantum Computation and
       Quantum Information*. Cambridge University Press.
       https://doi.org/10.1017/CBO9780511976667
       (Depolarising, amplitude damping, phase flip channels: Ch. 8)

.. [2] Lindblad, G. (1976). On the generators of quantum dynamical semigroups.
       *Commun. Math. Phys.* 48, 119.
       https://doi.org/10.1007/BF01608499

.. [3] Preskill, J. (1998). *Lecture Notes on Quantum Information*.
       Available at http://theory.caltech.edu/~preskill/ph229/

Physical validation
-------------------
Depolarising channel: p(I)=1-3p/4, p(X)=p(Y)=p(Z)=p/4 leaves the maximally
mixed state rho=I/2 invariant for all p [Nielsen&Chuang, Ch. 8.3].
Amplitude damping: excited state |1> decays with probability gamma;
p_|1>(t) = exp(-gamma*t) [exponential Markovian decay].
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

__all__ = [
    "NoiseModel",
    "DepolarizingNoise",
    "AmplitudeDampingNoise",
    "ThermalRelaxationNoise",
    "ReadoutNoise",
    "CompositeNoise",
    "NoiseModelError",
    "DepolarizingChannel",
    "AmplitudeDampingChannel",
]


class NoiseModelError(ValueError):
    """Raised when a noise model has invalid parameters."""


class NoiseModel(ABC):
    """Base class for all QuantX noise models."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable noise payload for Engine.run_circuit()."""

    def __add__(self, other: "NoiseModel") -> "CompositeNoise":
        """Combine two noise models with the + operator."""
        return CompositeNoise([self, other])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.to_dict()})"


class DepolarizingNoise(NoiseModel):
    """Uniform depolarizing channel applied after every gate.

    The depolarizing channel replaces the qubit state with the maximally mixed
    state with probability p:  ρ → (1-p)ρ + p·I/2, which is
    (1 - 3p/4)ρ + (p/4)(XρX + YρY + ZρZ). The native circuit backend applies it
    to every qubit after every gate (idle qubits included), one trajectory per
    shot. The density-matrix simulator in :mod:`quantx_engine.primitives` and the
    dynamic-circuit simulator apply it the same way (see :func:`native_channels`).

    Parameters
    ----------
    prob:
        Depolarizing probability per gate, in [0, 1].
    """

    def __init__(self, prob: float):
        if not (0.0 <= prob <= 1.0):
            raise NoiseModelError(f"depolarizing prob must be in [0, 1], got {prob}")
        self._prob = float(prob)

    @property
    def prob(self) -> float:
        return self._prob

    def to_dict(self) -> dict[str, Any]:
        return {"depolarizing_prob": self._prob}


class AmplitudeDampingNoise(NoiseModel):
    """Amplitude damping (T1 relaxation) noise channel.

    Models energy dissipation: |1⟩ → |0⟩ with probability γ per gate step.

    Parameters
    ----------
    gamma:
        Amplitude damping parameter γ = 1 − exp(−t/T1), in [0, 1].
    """

    def __init__(self, gamma: float):
        if not (0.0 <= gamma <= 1.0):
            raise NoiseModelError(f"amplitude damping gamma must be in [0, 1], got {gamma}")
        self._gamma = float(gamma)

    @property
    def gamma(self) -> float:
        return self._gamma

    def to_dict(self) -> dict[str, Any]:
        return {"amplitude_damping_gamma": self._gamma}


class ThermalRelaxationNoise(NoiseModel):
    """Combined T1/T2 thermal relaxation noise model.

    Computes amplitude damping γ = 1 − exp(−t_gate / T1) from the longitudinal
    relaxation time T1 and gate duration t_gate.  The T2 dephasing parameter
    λ = 1 − exp(−t_gate / T2) is computed and stored in the output dict under
    ``noise_metadata`` for informational purposes, but is **not** currently
    consumed by the native circuit backend (which supports T1 amplitude damping
    only via the ``t1_decay`` noise kind).

    ``to_dict()`` emits ``{"noise": {"kind": "t1_decay", "T1": T1, "dt": t_gate},
    "noise_metadata": {...}}`` — only the ``noise`` sub-dict is executed.

    The constraint T2 ≤ 2·T1 (Lindblad bound) is enforced.

    Parameters
    ----------
    T1:
        Longitudinal relaxation time in nanoseconds.
    T2:
        Transverse relaxation time in nanoseconds. Must satisfy T2 ≤ 2·T1.
    t_gate:
        Gate duration in nanoseconds (default: 50 ns, typical for
        superconducting qubits).
    """

    def __init__(self, T1: float, T2: float, t_gate: float = 50.0):
        if T1 <= 0:
            raise NoiseModelError(f"T1 must be positive, got {T1}")
        if T2 <= 0:
            raise NoiseModelError(f"T2 must be positive, got {T2}")
        if T2 > 2 * T1:
            raise NoiseModelError(
                f"T2 ({T2}) must satisfy T2 ≤ 2·T1 ({2*T1:.4g}) (Lindblad bound)"
            )
        if t_gate <= 0:
            raise NoiseModelError(f"t_gate must be positive, got {t_gate}")
        self._T1 = float(T1)
        self._T2 = float(T2)
        self._t_gate = float(t_gate)

    @property
    def T1(self) -> float:
        return self._T1

    @property
    def T2(self) -> float:
        return self._T2

    @property
    def t_gate(self) -> float:
        return self._t_gate

    @property
    def gamma(self) -> float:
        """Amplitude damping parameter γ = 1 − exp(−t_gate / T1)."""
        return 1.0 - math.exp(-self._t_gate / self._T1)

    @property
    def lambda_dephasing(self) -> float:
        """Dephasing parameter λ = 1 − exp(−t_gate / T2)."""
        return 1.0 - math.exp(-self._t_gate / self._T2)

    def to_dict(self) -> dict[str, Any]:
        # "noise" sub-dict is parsed by the native backend (t1_decay kind is supported)
        # "noise_metadata" carries T2 info for user reference — not executed by backend
        return {
            "noise": {"kind": "t1_decay", "T1": self._T1, "dt": self._t_gate},
            "noise_metadata": {
                "T1_ns": self._T1,
                "T2_ns": self._T2,
                "t_gate_ns": self._t_gate,
                "gamma": self.gamma,
                "lambda_dephasing": self.lambda_dephasing,
                "note": "T2 dephasing not yet applied by native backend",
            },
        }


class ReadoutNoise(NoiseModel):
    """Symmetric bit-flip noise, intended to stand in for readout errors.

    The native circuit backend implements this as a ``bit_flip`` channel
    (``noise = {"kind": "bit_flip", "prob": p}``): after every gate except a
    measurement, each qubit flips with probability p.  So it acts as gate
    noise, not as a readout error.  A true readout error model would act on
    the measurement outcomes only; ``bit_flip`` is the closest mechanism the
    backend supports.

    Parameters
    ----------
    prob:
        Bit-flip probability per qubit per gate, in [0, 0.5].
        Values above 0.5 are anti-correlated and should be modeled differently.
    """

    def __init__(self, prob: float):
        if not (0.0 <= prob <= 0.5):
            raise NoiseModelError(
                f"readout error prob must be in [0, 0.5], got {prob} "
                "(values > 0.5 indicate anti-correlated error; model differently)"
            )
        self._prob = float(prob)

    @property
    def prob(self) -> float:
        return self._prob

    def to_dict(self) -> dict[str, Any]:
        # "bit_flip" is the supported noise kind in the native backend
        return {"noise": {"kind": "bit_flip", "prob": self._prob}}


class CompositeNoise(NoiseModel):
    """Combination of multiple noise models applied simultaneously.

    ``to_dict()`` is the union of the models' payloads, so the models must
    emit disjoint keys.  If two models emit the same key, building the
    composite (constructor or ``+``) raises :class:`NoiseModelError`; no entry
    is ever silently replaced.

    In practice this allows **at most one** ``noise``-kind model per
    composite.  :class:`ThermalRelaxationNoise` (``t1_decay``) and
    :class:`ReadoutNoise` (``bit_flip``) both put their channel under the
    single key ``noise``, because the native backend parses exactly one
    ``noise`` object per circuit (``parse_noise_config`` in
    ``src/json_utils.cpp``).  A second one has no slot to go to: merging the
    two would drop a channel.  :class:`DepolarizingNoise` and
    :class:`AmplitudeDampingNoise` emit the top-level fields
    ``depolarizing_prob`` and ``amplitude_damping_gamma``, which the backend
    applies alongside the ``noise`` channel, so they combine with each other
    and with one ``noise``-kind model::

        CompositeNoise([DepolarizingNoise(0.005),
                        ThermalRelaxationNoise(T1=1e5, T2=6e4, t_gate=50)])  # ok
        CompositeNoise([ThermalRelaxationNoise(T1=1e5, T2=6e4, t_gate=50),
                        ReadoutNoise(0.02)])     # NoiseModelError: both emit "noise"

    Every key is checked, not only ``noise``.  Each top-level field holds one
    value per circuit, so two :class:`DepolarizingNoise` models collide on
    ``depolarizing_prob`` (build one model instead).  ``noise_metadata`` is
    checked as well: no simulator executes it, but a merge would keep only one
    model's description.

    The check is on the fields a payload sets, not only on its keys.  A
    ``noise`` selector of kind ``depolarizing`` or ``amplitude_damping`` (no
    shipped model emits one, but the backend accepts both, so a custom model
    may) is another spelling of ``depolarizing_prob`` or
    ``amplitude_damping_gamma``: the backend writes its ``prob`` or ``gamma``
    (at the selector's top level or in its ``params``) into that field.  Such a
    selector therefore collides with a :class:`DepolarizingNoise` or
    :class:`AmplitudeDampingNoise` although the keys differ.  A selector that
    carries no value leaves the field alone, and combines.

    Parameters
    ----------
    models:
        List of NoiseModel instances to combine.

    Raises
    ------
    NoiseModelError
        If ``models`` is empty, holds a non-NoiseModel, or two models set the
        same field: they emit the same key, or one emits a ``noise`` selector
        that the backend writes into another's field.
    """

    def __init__(self, models: list[NoiseModel]):
        if not models:
            raise NoiseModelError("CompositeNoise requires at least one model")
        for m in models:
            if not isinstance(m, NoiseModel):
                raise NoiseModelError(f"expected NoiseModel, got {type(m).__name__}")
        self._models = list(models)
        self._merged()  # refuse colliding models here, not when the circuit is run

    @property
    def models(self) -> list[NoiseModel]:
        return list(self._models)

    def to_dict(self) -> dict[str, Any]:
        return self._merged()

    def __add__(self, other: NoiseModel) -> "CompositeNoise":
        return CompositeNoise(self._models + [other])

    def _merged(self) -> dict[str, Any]:
        """Union of the models' payloads; NoiseModelError if two set the same field."""
        merged: dict[str, Any] = {}
        owner: dict[str, tuple[int, str]] = {}  # field -> (model index, key that sets it)
        for i, m in enumerate(self._models):
            for key, value in m.to_dict().items():
                fields = _fields_set(key, value)
                for field in fields:
                    if field in owner and owner[field][0] != i:
                        self._refuse(owner[field], (i, key), field, merged, value)
                for field in fields:
                    owner.setdefault(field, (i, key))
                merged[key] = value
        return merged

    def _refuse(self, first: tuple[int, str], second: tuple[int, str], field: str,
                merged: dict[str, Any], value: Any) -> None:
        """Raise for the second model setting a field the first already set."""
        (j, key_j), (i, key_i) = first, second
        name_j = f"{type(self._models[j]).__name__} (models[{j}])"
        name_i = f"{type(self._models[i]).__name__} (models[{i}])"
        if key_j == key_i == "noise":
            kinds = [_kind(v) for v in (merged["noise"], value)]
            raise NoiseModelError(
                f"{name_j} and {name_i} both emit 'noise' (kinds {kinds[0]!r} and "
                f"{kinds[1]!r}); the native backend executes one 'noise' channel per "
                "circuit, so one would be silently dropped. Use at most one noise-kind "
                "model per CompositeNoise; DepolarizingNoise and AmplitudeDampingNoise "
                "use their own fields and combine with it."
            )
        if key_j == key_i:
            raise NoiseModelError(
                f"{name_j} and {name_i} both emit {key_i!r}; a circuit holds one value "
                "per key, so one would be silently dropped. Use a single model instead."
            )
        # One sets the field directly, the other through a 'noise' selector.
        def emits(key: str, v: Any) -> str:
            if key == "noise":
                return f"emits a 'noise' selector of kind {_kind(v)!r}"
            return f"emits {key!r}"
        raise NoiseModelError(
            f"{name_j} {emits(key_j, merged[key_j])} and {name_i} {emits(key_i, value)}; "
            f"the native backend writes both into {field!r}, so one would be silently "
            "dropped. Use a single model instead."
        )


# The native parser (parse_noise_config in src/json_utils.cpp) reads these two 'noise'
# kinds as another spelling of a top-level field: when the selector carries a value
# (kind -> field, value key), it writes that value into the field, over whatever a
# DepolarizingNoise or AmplitudeDampingNoise put there.
_NOISE_KIND_FIELDS = {"depolarizing": ("depolarizing_prob", "prob"),
                      "amplitude_damping": ("amplitude_damping_gamma", "gamma")}


def _kind(noise: Any) -> Any:
    return noise.get("kind") if isinstance(noise, dict) else None


def _fields_set(key: str, value: Any) -> list[str]:
    """The circuit fields one payload entry sets: its key, and for a 'noise' selector of
    an aliasing kind that carries a value, the top-level field the backend writes it into.

    This mirrors parse_noise_config: the value counts if it is non-null at the selector's
    top level, or present in its ``params``. (A value of the wrong type makes the backend
    reject the circuit, which is not a silent drop either way.) A selector without a value
    leaves the field alone, so it combines."""
    kind = _kind(value) if key == "noise" else None
    alias = _NOISE_KIND_FIELDS.get(kind) if isinstance(kind, str) else None
    if alias is None:
        return [key]
    field, value_key = alias
    params = value.get("params")
    carries = (value.get(value_key) is not None
               or (isinstance(params, dict) and value_key in params))
    return [key, field] if carries else [key]


def native_channels(model: "NoiseModel | dict[str, Any] | None") -> list[tuple[str, float]]:
    """The single-qubit channels the native circuit backend applies, in its order, to
    every qubit after every gate (never after a measurement), for a noise model or its
    ``to_dict()`` payload.

    Each entry is ``(kind, rate)``:

    - ``"depolarizing"``, p: ρ → (1 − p)ρ + p·I/2;
    - ``"amplitude_damping"``, γ: |1⟩ decays to |0⟩ with probability γ;
    - ``"bit_flip"`` or ``"phase_flip"``, p: X or Z with probability p.

    This mirrors the backend's parser and noise loop (``parse_noise_config`` in
    ``src/json_utils.cpp``, ``apply_noise_after_gate`` in ``src/engine.cpp``):

    - ``depolarizing_prob`` and ``amplitude_damping_gamma`` come first;
    - a ``noise`` selector of kind ``depolarizing`` or ``amplitude_damping`` that carries
      its value overwrites them;
    - then the ``noise`` kind runs: ``bit_flip``, ``phase_flip``, ``t1_decay`` (amplitude
      damping with γ = 1 − exp(−dt/T1)) or ``t2_decay`` (a phase flip with
      p = (1 − exp(−dt/T2))/2), each reading ``params`` before its own keys;
    - ``noise_metadata`` is information only, not executed;
    - every rate is clamped to [0, 1], as the backend clamps it.

    The Python simulators use this, so a model means the same channels everywhere.

    Raises
    ------
    NoiseModelError
        For a ``noise`` kind the backend does not support (it would refuse the circuit).
    """
    if model is None:
        return []
    payload = model.to_dict() if isinstance(model, NoiseModel) else dict(model)
    dep = float(payload.get("depolarizing_prob") or 0.0)
    gamma = float(payload.get("amplitude_damping_gamma") or 0.0)
    named: list[tuple[str, float]] = []
    noise = payload.get("noise")
    if isinstance(noise, dict):
        kind = noise.get("kind")
        params = noise.get("params")
        params = params if isinstance(params, dict) else {}

        def value(key: str, default: Any = None) -> Any:
            if key in params:
                return params[key]
            v = noise.get(key)
            return default if v is None else v

        if kind == "depolarizing":
            dep = float(value("prob", dep))
        elif kind == "amplitude_damping":
            gamma = float(value("gamma", gamma))
        elif kind in ("bit_flip", "phase_flip"):
            named.append((kind, float(value("prob", 0.0))))
        elif kind in ("t1_decay", "t2_decay"):
            t = float(value("T1" if kind == "t1_decay" else "T2", 0.0))
            dt = float(value("dt", 0.0))
            if t > 0.0 and dt > 0.0:
                if kind == "t1_decay":
                    named.append(("amplitude_damping", 1.0 - math.exp(-dt / t)))
                else:
                    named.append(("phase_flip", 0.5 * (1.0 - math.exp(-dt / t))))
        elif kind not in (None, "", "none"):
            raise NoiseModelError(f"unsupported noise kind: {kind!r}")
    channels: list[tuple[str, float]] = []
    if dep > 0.0:
        channels.append(("depolarizing", min(dep, 1.0)))
    if gamma > 0.0:
        channels.append(("amplitude_damping", min(gamma, 1.0)))
    channels.extend((kind, min(rate, 1.0)) for kind, rate in named if rate > 0.0)
    return channels


# ---------------------------------------------------------------------------
# Quantum-channel wrappers (operate directly on density matrices)
# These are separate from the engine-payload noise models above.
# ---------------------------------------------------------------------------

try:
    import numpy as _np
    _HAS_NP = True
except ImportError:  # pragma: no cover
    _HAS_NP = False


class DepolarizingChannel:
    """Depolarizing channel Φ(ρ) = (1-p)ρ + p·I/d applied to a density matrix.

    Parameters
    ----------
    p : float
        Depolarizing probability ∈ [0, 1].
    """

    def __init__(self, p: float) -> None:
        if not (0.0 <= p <= 1.0):
            raise NoiseModelError(f"p must be in [0, 1], got {p}")
        self.p = float(p)

    def apply(self, rho: "Any") -> "Any":
        """Apply channel to density matrix rho."""
        if not _HAS_NP:  # pragma: no cover
            raise NoiseModelError("numpy is required for DepolarizingChannel.apply()")
        rho = _np.asarray(rho, dtype=complex)
        d = rho.shape[0]
        return (1.0 - self.p) * rho + self.p / d * _np.eye(d, dtype=complex)

    def __repr__(self) -> str:
        return f"DepolarizingChannel(p={self.p})"


class AmplitudeDampingChannel:
    """Amplitude-damping (T1-relaxation) channel for a qubit density matrix.

    Kraus operators:
        K₀ = [[1, 0], [0, √(1-γ)]],  K₁ = [[0, √γ], [0, 0]]

    Parameters
    ----------
    gamma : float
        Amplitude-damping probability γ ∈ [0, 1].
    """

    def __init__(self, gamma: float) -> None:
        if not (0.0 <= gamma <= 1.0):
            raise NoiseModelError(f"gamma must be in [0, 1], got {gamma}")
        self.gamma = float(gamma)

    def apply(self, rho: "Any") -> "Any":
        """Apply channel to a 2×2 density matrix."""
        if not _HAS_NP:  # pragma: no cover
            raise NoiseModelError("numpy is required for AmplitudeDampingChannel.apply()")
        rho = _np.asarray(rho, dtype=complex)
        g = self.gamma
        K0 = _np.array([[1.0, 0.0], [0.0, _np.sqrt(1.0 - g)]], dtype=complex)
        K1 = _np.array([[0.0, _np.sqrt(g)], [0.0, 0.0]], dtype=complex)
        return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T

    def __repr__(self) -> str:
        return f"AmplitudeDampingChannel(gamma={self.gamma})"
