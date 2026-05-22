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
- CompositeNoise        — combine multiple noise models (fields merged)

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
    state with probability p:  ρ → (1-p)ρ + p·I/2

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
    """Symmetric bit-flip noise modelling readout errors.

    The native circuit backend implements this as a ``bit_flip`` channel
    (``noise = {"kind": "bit_flip", "prob": p}``), which applies a bit-flip
    with probability p after each measurement gate.  This is an approximation:
    a true readout error model would be measurement-specific, but ``bit_flip``
    is the closest currently-supported mechanism.

    Parameters
    ----------
    prob:
        Bit-flip (readout error) probability per measurement, in [0, 0.5].
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

    All noise models are merged into a single dict. Later models override
    earlier ones if they share the same key.

    Parameters
    ----------
    models:
        List of NoiseModel instances to combine.
    """

    def __init__(self, models: list[NoiseModel]):
        if not models:
            raise NoiseModelError("CompositeNoise requires at least one model")
        for m in models:
            if not isinstance(m, NoiseModel):
                raise NoiseModelError(f"expected NoiseModel, got {type(m).__name__}")
        self._models = list(models)

    @property
    def models(self) -> list[NoiseModel]:
        return list(self._models)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for m in self._models:
            result.update(m.to_dict())
        return result

    def __add__(self, other: NoiseModel) -> "CompositeNoise":
        return CompositeNoise(self._models + [other])
