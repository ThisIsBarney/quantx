"""Tests for the QuantX noise model library."""
from __future__ import annotations

import math
import os

import pytest
from quantx_engine.noise import (
    NoiseModel,
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
    assert "noise" in d  # ReadoutNoise is the only noise-kind model


def test_composite_depolarizing_plus_thermal_keeps_both_channels():
    """depolarizing_prob + one noise kind is what the native backend can execute."""
    import json
    depol = DepolarizingNoise(0.005)
    thermal = ThermalRelaxationNoise(T1=1e5, T2=6e4, t_gate=50.0)
    d = CompositeNoise([depol, thermal]).to_dict()
    assert d["depolarizing_prob"] == 0.005
    assert d["noise"] == {"kind": "t1_decay", "T1": 1e5, "dt": 50.0}
    assert d["noise_metadata"] == thermal.to_dict()["noise_metadata"]
    assert len(d) == 3
    assert json.loads(json.dumps(d)) == d  # unchanged through the JSON hop
    assert (depol + thermal).to_dict() == d


def test_composite_second_noise_kind_raises():
    """ReadoutNoise's bit_flip used to replace the T1 channel without a word."""
    with pytest.raises(NoiseModelError, match="both emit 'noise'") as excinfo:
        CompositeNoise([
            DepolarizingNoise(0.005),
            ThermalRelaxationNoise(T1=1e5, T2=6e4, t_gate=50.0),
            ReadoutNoise(0.02),
        ])
    assert "'t1_decay' and 'bit_flip'" in str(excinfo.value)


def test_composite_plus_operator_refuses_second_noise_kind():
    thermal = ThermalRelaxationNoise(T1=1e5, T2=6e4, t_gate=50.0)
    with pytest.raises(NoiseModelError, match="both emit 'noise'"):
        thermal + ReadoutNoise(0.02)
    with pytest.raises(NoiseModelError, match="both emit 'noise'"):
        (DepolarizingNoise(0.005) + thermal) + ReadoutNoise(0.02)
    with pytest.raises(NoiseModelError, match="both emit 'noise'"):
        ReadoutNoise(0.02) + (DepolarizingNoise(0.005) + thermal)


class _MetadataOnlyNoise(NoiseModel):
    """Emits only noise_metadata, which no shipped model does on its own."""

    def to_dict(self):
        return {"noise_metadata": {"note": "custom"}}


@pytest.mark.parametrize("models, key", [
    ([ReadoutNoise(0.01), ReadoutNoise(0.02)], "noise"),
    ([DepolarizingNoise(0.01), DepolarizingNoise(0.02)], "depolarizing_prob"),
    ([AmplitudeDampingNoise(0.01), AmplitudeDampingNoise(0.02)], "amplitude_damping_gamma"),
    ([ThermalRelaxationNoise(1e5, 6e4, 50.0), _MetadataOnlyNoise()], "noise_metadata"),
])
def test_composite_refuses_any_shared_key(models, key):
    with pytest.raises(NoiseModelError, match=f"both emit '{key}'"):
        CompositeNoise(models)


class _SelectorNoise(NoiseModel):
    """A custom model that spells a scalar channel as a named 'noise' selector, which
    the native backend accepts (parse_noise_config in src/json_utils.cpp)."""

    def __init__(self, kind, **fields):
        self._noise = {"kind": kind, **fields}

    def to_dict(self):
        return {"noise": dict(self._noise)}


@pytest.mark.parametrize("scalar, selector, field", [
    (DepolarizingNoise(0.01), _SelectorNoise("depolarizing", prob=0.02),
     "depolarizing_prob"),
    (AmplitudeDampingNoise(0.01), _SelectorNoise("amplitude_damping", gamma=0.02),
     "amplitude_damping_gamma"),
    # the backend reads the value from params too
    (DepolarizingNoise(0.01), _SelectorNoise("depolarizing", params={"prob": 0.02}),
     "depolarizing_prob"),
    (AmplitudeDampingNoise(0.01), _SelectorNoise("amplitude_damping", params={"gamma": 0.0}),
     "amplitude_damping_gamma"),
])
def test_composite_refuses_a_selector_aliasing_a_scalar_field(scalar, selector, field):
    """The keys differ, but the backend writes the selector's prob or gamma into the
    scalar field, so the composite would have run one of the two values."""
    for models in ([scalar, selector], [selector, scalar]):
        with pytest.raises(NoiseModelError, match=f"writes both into '{field}'"):
            CompositeNoise(models)
    with pytest.raises(NoiseModelError, match=f"writes both into '{field}'"):
        scalar + selector
    with pytest.raises(NoiseModelError, match=f"writes both into '{field}'"):
        CompositeNoise([CompositeNoise([scalar]), selector])  # nested


def test_composite_selector_combines_with_the_other_scalar_field():
    """A depolarizing selector sets depolarizing_prob only; amplitude damping combines."""
    d = CompositeNoise([AmplitudeDampingNoise(0.05),
                        _SelectorNoise("depolarizing", prob=0.02)]).to_dict()
    assert d == {"amplitude_damping_gamma": 0.05,
                 "noise": {"kind": "depolarizing", "prob": 0.02}}


@pytest.mark.parametrize("scalar, selector", [
    (DepolarizingNoise(0.01), _SelectorNoise("depolarizing")),
    (DepolarizingNoise(0.01), _SelectorNoise("depolarizing", prob=None)),
    # the backend reads gamma for this kind, not prob
    (AmplitudeDampingNoise(0.01), _SelectorNoise("amplitude_damping", prob=0.5)),
])
def test_composite_selector_without_a_value_combines(scalar, selector):
    """A selector that carries no value leaves the scalar field alone (the backend keeps
    it), so nothing would be dropped and the composite is allowed."""
    d = CompositeNoise([scalar, selector]).to_dict()
    assert d == {**scalar.to_dict(), **selector.to_dict()}


@pytest.mark.parametrize("noise", ["bit_flip", None, {"kind": ["depolarizing"]}, {"prob": 0.1}])
def test_composite_odd_noise_values_do_not_crash_the_check(noise):
    class _Raw(NoiseModel):
        def to_dict(self):
            return {"noise": noise}

    d = CompositeNoise([DepolarizingNoise(0.01), _Raw()]).to_dict()
    assert d == {"depolarizing_prob": 0.01, "noise": noise}


# ── The collision rules against the native backend ────────────────────────────
# The check above mirrors parse_noise_config by hand. These run the circuits, so a
# backend change that breaks the mirror fails here. Everything is deterministic: one
# qubit, X (|1>), then channels at rate 0 or 1, read through Z0.

def _native_engine():
    # CI's native job sets QX_REQUIRE_NATIVE=1: there a load failure must fail.
    unavailable = pytest.fail if os.environ.get("QX_REQUIRE_NATIVE") == "1" else pytest.skip
    try:
        from quantx_engine.engine import Engine
        return Engine()
    except Exception as exc:  # QuantXNativeUnavailable, or a missing dependency
        unavailable(f"native library unavailable: {exc}")


def _z_after_x(engine, noise_fields):
    circuit = {"qubits": 1, "shots": 0, "observables": ["Z0"], "seed": 1,
               "operations": [{"name": "X", "targets": [0]}], **noise_fields}
    return engine.run_circuit(circuit)["expectation_exact"]["Z0"]


@pytest.mark.parametrize("selector, overwrites", [
    ({"kind": "amplitude_damping", "gamma": 0.0}, True),
    ({"kind": "amplitude_damping", "params": {"gamma": 0.0}}, True),
    ({"kind": "amplitude_damping"}, False),
    ({"kind": "amplitude_damping", "prob": 0.0}, False),
])
def test_native_selector_alias_matches_the_check(selector, overwrites):
    """Damping at gamma = 1 sends |1> to |0> (Z0 = +1). A selector the check calls an
    alias must replace it (Z0 = -1); one it lets combine must leave it in place."""
    engine = _native_engine()
    assert _z_after_x(engine, {"amplitude_damping_gamma": 1.0}) == pytest.approx(1.0)
    z = _z_after_x(engine, {"amplitude_damping_gamma": 1.0, "noise": selector})
    assert z == pytest.approx(-1.0 if overwrites else 1.0)
    combine = [AmplitudeDampingNoise(1.0), _SelectorNoise(**selector)]
    if overwrites:
        with pytest.raises(NoiseModelError):
            CompositeNoise(combine)
    else:
        CompositeNoise(combine)


def test_native_scalar_field_and_noise_kind_both_run():
    """What the composite promises when it combines: damping (gamma = 1) takes |1> to |0>,
    then a bit flip (p = 1) back to |1>. Z0 = -1 only if both channels ran; either alone
    ends in |0>."""
    engine = _native_engine()
    model = CompositeNoise([AmplitudeDampingNoise(1.0), _SelectorNoise("bit_flip", prob=1.0)])
    assert _z_after_x(engine, model.to_dict()) == pytest.approx(-1.0)
    assert _z_after_x(engine, {"noise": {"kind": "bit_flip", "prob": 1.0}}) == pytest.approx(1.0)
    assert _z_after_x(engine, AmplitudeDampingNoise(1.0).to_dict()) == pytest.approx(1.0)


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
