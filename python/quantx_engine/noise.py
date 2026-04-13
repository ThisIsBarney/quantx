"""
Noise model library for the QuantX engine.

Provides structured noise configuration objects that generate the JSON noise
payloads accepted by Engine.run_circuit() and Engine.run_qasm().

Supported noise models:
- DepolarizingNoise     — uniform depolarizing channel p on all qubits
- AmplitudeDampingNoise — T1 amplitude damping γ on all qubits
- ThermalRelaxationNoise— combined T1/T2 relaxation (converts to γ/λ fields)
- CompositeNoise        — combine multiple noise models (fields merged)
- ReadoutNoise          — symmetric readout bit-flip error on measurement

Usage:
    from quantx_engine.noise import DepolarizingNoise
    engine.run_circuit(circuit, options={"noise": DepolarizingNoise(0.01).to_dict()})
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

    Computes amplitude damping γ = 1 − exp(−t_gate / T1) and dephasing
    λ = 1 − exp(−t_gate / T2) from physical relaxation times T1, T2
    and the gate duration t_gate.

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
        return {
            "amplitude_damping_gamma": self.gamma,
            "dephasing_lambda": self.lambda_dephasing,
            "T1_ns": self._T1,
            "T2_ns": self._T2,
            "t_gate_ns": self._t_gate,
        }


class ReadoutNoise(NoiseModel):
    """Symmetric readout bit-flip noise applied during measurement.

    Each measured bit is flipped with probability p_readout independently.

    Parameters
    ----------
    prob:
        Readout error probability per measurement bit, in [0, 0.5].
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
        return {"readout_error_prob": self._prob}


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
