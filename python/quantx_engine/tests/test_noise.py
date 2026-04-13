"""Tests for the QuantX noise model library."""
from __future__ import annotations

import math
import pytest
from quantx_engine.noise import (
    DepolarizingNoise,
    AmplitudeDampingNoise,
    ThermalRelaxationNoise,
    ReadoutNoise,
    CompositeNoise,
    NoiseModelError,
)


# ── DepolarizingNoise ────────────────────────────────────────────────────────

def test_depolarizing_valid():
    n = DepolarizingNoise(0.01)
    d = n.to_dict()
    assert d == {"depolarizing_prob": 0.01}


def test_depolarizing_boundary():
    assert DepolarizingNoise(0.0).to_dict()["depolarizing_prob"] == 0.0
    assert DepolarizingNoise(1.0).to_dict()["depolarizing_prob"] == 1.0


def test_depolarizing_invalid():
    with pytest.raises(NoiseModelError):
        DepolarizingNoise(-0.01)
    with pytest.raises(NoiseModelError):
        DepolarizingNoise(1.01)


# ── AmplitudeDampingNoise ────────────────────────────────────────────────────

def test_amplitude_damping_valid():
    n = AmplitudeDampingNoise(0.05)
    d = n.to_dict()
    assert d == {"amplitude_damping_gamma": 0.05}


def test_amplitude_damping_invalid():
    with pytest.raises(NoiseModelError):
        AmplitudeDampingNoise(-0.1)
    with pytest.raises(NoiseModelError):
        AmplitudeDampingNoise(1.1)


# ── ThermalRelaxationNoise ───────────────────────────────────────────────────

def test_thermal_relaxation_gamma_lambda():
    T1, T2, t = 50000.0, 30000.0, 50.0
    n = ThermalRelaxationNoise(T1, T2, t)
    expected_gamma = 1.0 - math.exp(-t / T1)
    expected_lambda = 1.0 - math.exp(-t / T2)
    d = n.to_dict()
    # Active backend key: t1_decay via noise sub-dict
    assert d["noise"]["kind"] == "t1_decay"
    assert d["noise"]["T1"] == T1
    assert d["noise"]["dt"] == t
    # Metadata for informational use
    meta = d["noise_metadata"]
    assert abs(meta["gamma"] - expected_gamma) < 1e-12
    assert abs(meta["lambda_dephasing"] - expected_lambda) < 1e-12
    assert meta["T1_ns"] == T1
    assert meta["T2_ns"] == T2
    assert meta["t_gate_ns"] == t


def test_thermal_relaxation_t2_equals_2t1():
    # T2 = 2*T1 is the Lindblad bound — must be accepted
    n = ThermalRelaxationNoise(T1=10000.0, T2=20000.0, t_gate=50.0)
    assert n.T2 == 20000.0


def test_thermal_relaxation_t2_exceeds_bound():
    with pytest.raises(NoiseModelError, match="Lindblad"):
        ThermalRelaxationNoise(T1=10000.0, T2=20001.0, t_gate=50.0)


def test_thermal_relaxation_invalid_times():
    with pytest.raises(NoiseModelError):
        ThermalRelaxationNoise(T1=0.0, T2=1000.0)
    with pytest.raises(NoiseModelError):
        ThermalRelaxationNoise(T1=1000.0, T2=-1.0)
    with pytest.raises(NoiseModelError):
        ThermalRelaxationNoise(T1=1000.0, T2=100.0, t_gate=0.0)


# ── ReadoutNoise ─────────────────────────────────────────────────────────────

def test_readout_valid():
    n = ReadoutNoise(0.02)
    d = n.to_dict()
    assert d == {"noise": {"kind": "bit_flip", "prob": 0.02}}


def test_readout_boundary():
    assert ReadoutNoise(0.0).to_dict()["noise"]["prob"] == 0.0
    assert ReadoutNoise(0.5).to_dict()["noise"]["prob"] == 0.5


def test_readout_invalid():
    with pytest.raises(NoiseModelError):
        ReadoutNoise(-0.01)
    with pytest.raises(NoiseModelError):
        ReadoutNoise(0.51)


# ── CompositeNoise ────────────────────────────────────────────────────────────

def test_composite_merges_dicts():
    n = CompositeNoise([DepolarizingNoise(0.01), AmplitudeDampingNoise(0.05)])
    d = n.to_dict()
    assert d["depolarizing_prob"] == 0.01
    assert d["amplitude_damping_gamma"] == 0.05


def test_composite_plus_operator():
    n = DepolarizingNoise(0.01) + ReadoutNoise(0.02)
    assert isinstance(n, CompositeNoise)
    d = n.to_dict()
    assert "depolarizing_prob" in d
    # ReadoutNoise emits {"noise": {"kind": "bit_flip", ...}}
    assert "noise" in d
    assert d["noise"]["kind"] == "bit_flip"


def test_composite_chaining():
    n = DepolarizingNoise(0.01) + AmplitudeDampingNoise(0.05) + ReadoutNoise(0.02)
    assert isinstance(n, CompositeNoise)
    d = n.to_dict()
    assert "depolarizing_prob" in d
    assert "amplitude_damping_gamma" in d
    assert "noise" in d  # ReadoutNoise wins (last one)


def test_composite_empty_raises():
    with pytest.raises(NoiseModelError):
        CompositeNoise([])


def test_composite_invalid_model_raises():
    with pytest.raises(NoiseModelError):
        CompositeNoise([DepolarizingNoise(0.01), "not_a_model"])


# ── Integration: to_dict used as circuit options ──────────────────────────────

def test_noise_dict_json_serialisable():
    """Ensure to_dict() output is fully JSON-serialisable (including nested dicts)."""
    import json
    models = [
        DepolarizingNoise(0.01),
        AmplitudeDampingNoise(0.05),
        ThermalRelaxationNoise(50000.0, 30000.0, 50.0),
        ReadoutNoise(0.02),
        DepolarizingNoise(0.01) + ReadoutNoise(0.02),
    ]
    for m in models:
        d = m.to_dict()
        json.dumps(d)  # must not raise — verifies full serializability
