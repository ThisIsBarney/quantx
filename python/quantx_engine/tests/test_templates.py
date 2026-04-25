"""Tests for Circuit Templates Library — M88."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quantx_engine.templates import (
    CircuitTemplate,
    TemplateError,
    ghz_circuit,
    hardware_efficient_ansatz,
    iqp_circuit,
    qaoa_circuit,
    quantum_fourier_transform,
    uccsd_ansatz,
)
from quantx_engine.primitives import Estimator, Sampler


# ===========================================================================
# Helpers
# ===========================================================================

def _is_valid_circuit_dict(d: dict) -> bool:
    """Check that d has the expected circuit-dict shape."""
    return (
        isinstance(d, dict)
        and "n_qubits" in d
        and "operations" in d
        and isinstance(d["n_qubits"], int)
        and isinstance(d["operations"], list)
    )


def _all_params_bound(d: dict) -> bool:
    """Return True if every op param is a float (no strings left)."""
    for op in d["operations"]:
        for p in op.get("params", []):
            if isinstance(p, str):
                return False
    return True


# ===========================================================================
# CircuitTemplate — core dataclass
# ===========================================================================

class TestCircuitTemplate:
    def _make_tmpl(self) -> CircuitTemplate:
        return CircuitTemplate(
            n_qubits=2,
            operations=[
                {"name": "ry", "qubits": [0], "params": ["alpha"]},
                {"name": "cx", "qubits": [0, 1]},
                {"name": "ry", "qubits": [1], "params": ["beta"]},
            ],
            parameter_names=["alpha", "beta"],
            name="TestTemplate",
        )

    def test_n_parameters(self):
        tmpl = self._make_tmpl()
        assert tmpl.n_parameters == 2

    def test_to_circuit_returns_valid_dict(self):
        tmpl = self._make_tmpl()
        d = tmpl.to_circuit()
        assert _is_valid_circuit_dict(d)
        assert d["n_qubits"] == 2

    def test_to_circuit_preserves_symbolic_params(self):
        tmpl = self._make_tmpl()
        d = tmpl.to_circuit()
        # symbolic strings should still be there
        assert d["operations"][0]["params"] == ["alpha"]

    def test_bind_list_returns_circuit_dict(self):
        tmpl = self._make_tmpl()
        d = tmpl.bind([1.0, 2.0])
        assert _is_valid_circuit_dict(d)
        assert _all_params_bound(d)

    def test_bind_list_correct_values(self):
        tmpl = self._make_tmpl()
        d = tmpl.bind([1.5, -0.3])
        assert d["operations"][0]["params"] == [pytest.approx(1.5)]
        assert d["operations"][2]["params"] == [pytest.approx(-0.3)]

    def test_bind_dict_returns_circuit_dict(self):
        tmpl = self._make_tmpl()
        d = tmpl.bind({"alpha": 0.1, "beta": 0.2})
        assert _is_valid_circuit_dict(d)
        assert _all_params_bound(d)

    def test_bind_dict_correct_values(self):
        tmpl = self._make_tmpl()
        d = tmpl.bind({"alpha": 3.14, "beta": -1.57})
        assert d["operations"][0]["params"] == [pytest.approx(3.14)]
        assert d["operations"][2]["params"] == [pytest.approx(-1.57)]

    def test_bind_list_wrong_length_raises(self):
        tmpl = self._make_tmpl()
        with pytest.raises(TemplateError, match="Expected 2 parameter values"):
            tmpl.bind([1.0])

    def test_bind_dict_missing_key_raises(self):
        tmpl = self._make_tmpl()
        with pytest.raises(TemplateError, match="Missing parameter values"):
            tmpl.bind({"alpha": 1.0})

    def test_bind_invalid_type_raises(self):
        tmpl = self._make_tmpl()
        with pytest.raises(TemplateError, match="list or dict"):
            tmpl.bind(42)  # type: ignore

    def test_no_params_bind_empty_list(self):
        tmpl = CircuitTemplate(
            n_qubits=1,
            operations=[{"name": "h", "qubits": [0]}],
            parameter_names=[],
            name="Fixed",
        )
        d = tmpl.bind([])
        assert _is_valid_circuit_dict(d)

    def test_no_params_bind_empty_dict(self):
        tmpl = CircuitTemplate(
            n_qubits=1,
            operations=[{"name": "h", "qubits": [0]}],
            parameter_names=[],
            name="Fixed",
        )
        d = tmpl.bind({})
        assert _is_valid_circuit_dict(d)


# ===========================================================================
# hardware_efficient_ansatz
# ===========================================================================

class TestHardwareEfficientAnsatz:
    def test_returns_circuit_template(self):
        tmpl = hardware_efficient_ansatz(3)
        assert isinstance(tmpl, CircuitTemplate)

    def test_n_qubits(self):
        tmpl = hardware_efficient_ansatz(4)
        assert tmpl.n_qubits == 4

    def test_parameter_count_1_layer(self):
        # n_qubits * (n_layers + 1) = 3 * 2 = 6
        tmpl = hardware_efficient_ansatz(3, n_layers=1)
        assert tmpl.n_parameters == 6

    def test_parameter_count_2_layers(self):
        # 4 * (2 + 1) = 12
        tmpl = hardware_efficient_ansatz(4, n_layers=2)
        assert tmpl.n_parameters == 12

    def test_parameter_count_single_qubit(self):
        # 1 * (3 + 1) = 4
        tmpl = hardware_efficient_ansatz(1, n_layers=3)
        assert tmpl.n_parameters == 4

    def test_to_circuit_valid(self):
        tmpl = hardware_efficient_ansatz(3, n_layers=2)
        d = tmpl.to_circuit()
        assert _is_valid_circuit_dict(d)

    def test_bind_zeros_valid_circuit(self):
        tmpl = hardware_efficient_ansatz(3, n_layers=1)
        d = tmpl.bind([0.0] * tmpl.n_parameters)
        assert _all_params_bound(d)
        assert d["n_qubits"] == 3

    def test_bind_dict_works(self):
        tmpl = hardware_efficient_ansatz(2, n_layers=1)
        values = {n: 0.5 for n in tmpl.parameter_names}
        d = tmpl.bind(values)
        assert _all_params_bound(d)

    def test_rotation_ry_default(self):
        tmpl = hardware_efficient_ansatz(2)
        rot_gates = [op["name"] for op in tmpl.operations if op["name"] == "ry"]
        assert len(rot_gates) > 0

    def test_rotation_rx(self):
        tmpl = hardware_efficient_ansatz(2, rotation="rx")
        rot_gates = [op["name"] for op in tmpl.operations if op["name"] == "rx"]
        assert len(rot_gates) > 0

    def test_rotation_rz(self):
        tmpl = hardware_efficient_ansatz(2, rotation="rz")
        rot_gates = [op["name"] for op in tmpl.operations if op["name"] == "rz"]
        assert len(rot_gates) > 0

    def test_cnot_ladder_present(self):
        tmpl = hardware_efficient_ansatz(3, n_layers=1)
        cx_gates = [op for op in tmpl.operations if op["name"] == "cx"]
        # Should have n_qubits-1 CNOTs per layer
        assert len(cx_gates) == 2  # (3-1) * 1 layer

    def test_invalid_n_qubits_zero(self):
        with pytest.raises(TemplateError):
            hardware_efficient_ansatz(0)

    def test_invalid_n_layers_zero(self):
        with pytest.raises(TemplateError):
            hardware_efficient_ansatz(2, n_layers=0)

    def test_invalid_rotation(self):
        with pytest.raises(TemplateError, match="rotation"):
            hardware_efficient_ansatz(2, rotation="rx2")

    def test_estimator_integration(self):
        tmpl = hardware_efficient_ansatz(2, n_layers=1)
        d = tmpl.bind([0.0] * tmpl.n_parameters)
        result = Estimator().run([d], ["ZI"])
        assert result.values.shape == (1,)
        assert math.isfinite(result.values[0])


# ===========================================================================
# qaoa_circuit
# ===========================================================================

class TestQAOACircuit:
    def test_returns_circuit_template(self):
        tmpl = qaoa_circuit(4, p_layers=1)
        assert isinstance(tmpl, CircuitTemplate)

    def test_parameter_count_p1(self):
        tmpl = qaoa_circuit(4, p_layers=1)
        assert tmpl.n_parameters == 2  # gamma_0, beta_0

    def test_parameter_count_p3(self):
        tmpl = qaoa_circuit(4, p_layers=3)
        assert tmpl.n_parameters == 6  # 2*3

    def test_parameter_names_format(self):
        tmpl = qaoa_circuit(4, p_layers=2)
        assert "gamma_0" in tmpl.parameter_names
        assert "gamma_1" in tmpl.parameter_names
        assert "beta_0" in tmpl.parameter_names
        assert "beta_1" in tmpl.parameter_names

    def test_to_circuit_valid(self):
        tmpl = qaoa_circuit(4, p_layers=1)
        d = tmpl.to_circuit()
        assert _is_valid_circuit_dict(d)

    def test_initial_hadamards(self):
        tmpl = qaoa_circuit(4, p_layers=1)
        d = tmpl.to_circuit()
        h_gates = [op for op in d["operations"] if op["name"] == "h"]
        assert len(h_gates) == 4

    def test_bind_and_estimator(self):
        tmpl = qaoa_circuit(4, p_layers=1)
        d = tmpl.bind([0.1, 0.2])
        result = Estimator().run([d], ["ZIII"])
        assert math.isfinite(result.values[0])

    def test_bind_dict_works(self):
        tmpl = qaoa_circuit(3, p_layers=1)
        d = tmpl.bind({"gamma_0": 0.5, "beta_0": 0.3})
        assert _all_params_bound(d)

    def test_invalid_n_qubits_one(self):
        with pytest.raises(TemplateError):
            qaoa_circuit(1, p_layers=1)

    def test_invalid_p_layers_zero(self):
        with pytest.raises(TemplateError):
            qaoa_circuit(4, p_layers=0)


# ===========================================================================
# ghz_circuit
# ===========================================================================

class TestGHZCircuit:
    def test_returns_circuit_template(self):
        tmpl = ghz_circuit(3)
        assert isinstance(tmpl, CircuitTemplate)

    def test_no_parameters(self):
        tmpl = ghz_circuit(4)
        assert tmpl.n_parameters == 0
        assert tmpl.parameter_names == []

    def test_to_circuit_valid(self):
        tmpl = ghz_circuit(3)
        d = tmpl.to_circuit()
        assert _is_valid_circuit_dict(d)
        assert d["n_qubits"] == 3

    def test_bind_empty_list(self):
        tmpl = ghz_circuit(3)
        d = tmpl.bind([])
        assert _is_valid_circuit_dict(d)

    def test_ghz_sampler_only_00_and_11(self):
        """GHZ |000...0⟩ + |111...1⟩ superposition."""
        n = 3
        tmpl = ghz_circuit(n)
        d = tmpl.to_circuit()
        result = Sampler().run([d])
        qd = result.quasi_dists[0]
        # Only |000⟩ = 0 and |111⟩ = 2^n - 1 should be present
        all_zero = 0
        all_one = (1 << n) - 1
        assert set(qd.keys()) == {all_zero, all_one}
        assert qd[all_zero] == pytest.approx(0.5, abs=1e-10)
        assert qd[all_one]  == pytest.approx(0.5, abs=1e-10)

    def test_ghz_sampler_2_qubits(self):
        tmpl = ghz_circuit(2)
        d = tmpl.to_circuit()
        result = Sampler().run([d])
        qd = result.quasi_dists[0]
        assert set(qd.keys()) == {0, 3}

    def test_ghz_structure_ops(self):
        tmpl = ghz_circuit(4)
        d = tmpl.to_circuit()
        ops = d["operations"]
        # First op should be H on qubit 0
        assert ops[0]["name"] == "h"
        assert ops[0]["qubits"] == [0]
        # Remaining ops should be CNOTs
        for op in ops[1:]:
            assert op["name"] == "cx"

    def test_invalid_n_qubits_one(self):
        with pytest.raises(TemplateError):
            ghz_circuit(1)


# ===========================================================================
# quantum_fourier_transform
# ===========================================================================

class TestQFT:
    def test_returns_circuit_template(self):
        tmpl = quantum_fourier_transform(3)
        assert isinstance(tmpl, CircuitTemplate)

    def test_no_parameters(self):
        tmpl = quantum_fourier_transform(4)
        assert tmpl.n_parameters == 0

    def test_to_circuit_valid(self):
        tmpl = quantum_fourier_transform(3)
        d = tmpl.to_circuit()
        assert _is_valid_circuit_dict(d)

    def test_qft_1qubit_is_hadamard(self):
        """QFT on |0⟩ for 1 qubit = H gate → uniform superposition."""
        tmpl = quantum_fourier_transform(1)
        d = tmpl.to_circuit()
        result = Sampler().run([d])
        qd = result.quasi_dists[0]
        assert qd.get(0, 0.0) == pytest.approx(0.5, abs=1e-10)
        assert qd.get(1, 0.0) == pytest.approx(0.5, abs=1e-10)

    def test_qft_uniform_superposition_from_zero(self):
        """QFT|0...0⟩ should give uniform superposition over all 2^n states."""
        n = 3
        tmpl = quantum_fourier_transform(n)
        d = tmpl.to_circuit()
        result = Sampler().run([d])
        qd = result.quasi_dists[0]
        expected_prob = 1.0 / (1 << n)
        # All 2^n states should have equal probability
        assert len(qd) == (1 << n)
        for prob in qd.values():
            assert prob == pytest.approx(expected_prob, abs=1e-8)

    def test_iqft_inverts_qft(self):
        """QFT followed by IQFT should return to |0...0⟩."""
        n = 2
        qft = quantum_fourier_transform(n)
        iqft = quantum_fourier_transform(n, inverse=True)
        # Compose circuits
        combined = {
            "n_qubits": n,
            "operations": qft.to_circuit()["operations"]
                        + iqft.to_circuit()["operations"],
        }
        result = Sampler().run([combined])
        qd = result.quasi_dists[0]
        # Should be back to |00⟩
        assert qd.get(0, 0.0) == pytest.approx(1.0, abs=1e-8)

    def test_qft_name(self):
        tmpl = quantum_fourier_transform(3)
        assert tmpl.name == "QFT"

    def test_iqft_name(self):
        tmpl = quantum_fourier_transform(3, inverse=True)
        assert tmpl.name == "IQFT"

    def test_qft_bind_empty(self):
        tmpl = quantum_fourier_transform(2)
        d = tmpl.bind([])
        assert _is_valid_circuit_dict(d)

    def test_invalid_n_qubits_zero(self):
        with pytest.raises(TemplateError):
            quantum_fourier_transform(0)


# ===========================================================================
# iqp_circuit
# ===========================================================================

class TestIQPCircuit:
    def test_returns_circuit_template(self):
        tmpl = iqp_circuit(3, depth=1, seed=42)
        assert isinstance(tmpl, CircuitTemplate)

    def test_n_qubits(self):
        tmpl = iqp_circuit(4, depth=1, seed=0)
        assert tmpl.n_qubits == 4

    def test_has_parameters(self):
        tmpl = iqp_circuit(3, depth=2, seed=1)
        assert tmpl.n_parameters > 0

    def test_to_circuit_valid(self):
        tmpl = iqp_circuit(3, depth=2, seed=2)
        d = tmpl.to_circuit()
        assert _is_valid_circuit_dict(d)

    def test_seed_reproducible(self):
        t1 = iqp_circuit(4, depth=2, seed=99)
        t2 = iqp_circuit(4, depth=2, seed=99)
        assert t1.parameter_names == t2.parameter_names
        assert len(t1.operations) == len(t2.operations)

    def test_bind_works(self):
        tmpl = iqp_circuit(3, depth=1, seed=7)
        vals = [0.1] * tmpl.n_parameters
        d = tmpl.bind(vals)
        assert _all_params_bound(d)

    def test_has_hadamards(self):
        tmpl = iqp_circuit(3, depth=2, seed=0)
        d = tmpl.to_circuit()
        h_gates = [op for op in d["operations"] if op["name"] == "h"]
        assert len(h_gates) > 0

    def test_invalid_n_qubits_one(self):
        with pytest.raises(TemplateError):
            iqp_circuit(1, depth=1)

    def test_invalid_depth_zero(self):
        with pytest.raises(TemplateError):
            iqp_circuit(3, depth=0)


# ===========================================================================
# uccsd_ansatz
# ===========================================================================

class TestUCCSDansatz:
    def test_returns_circuit_template(self):
        tmpl = uccsd_ansatz(n_electrons=1, n_qubits=4)
        assert isinstance(tmpl, CircuitTemplate)

    def test_n_qubits(self):
        tmpl = uccsd_ansatz(n_electrons=2, n_qubits=4)
        assert tmpl.n_qubits == 4

    def test_has_parameters(self):
        tmpl = uccsd_ansatz(n_electrons=1, n_qubits=4)
        assert tmpl.n_parameters > 0

    def test_single_electron_2_qubits(self):
        # 1 electron, 2 qubits: 1 occupied, 1 virtual → 1 single excitation
        tmpl = uccsd_ansatz(n_electrons=1, n_qubits=2)
        assert "theta_s_0_1" in tmpl.parameter_names

    def test_to_circuit_valid(self):
        tmpl = uccsd_ansatz(n_electrons=2, n_qubits=4)
        d = tmpl.to_circuit()
        assert _is_valid_circuit_dict(d)

    def test_hf_reference_x_gates(self):
        """First n_electrons ops should be X gates."""
        n_e = 2
        tmpl = uccsd_ansatz(n_electrons=n_e, n_qubits=4)
        d = tmpl.to_circuit()
        x_gates = [op for op in d["operations"] if op["name"] == "x"]
        assert len(x_gates) >= n_e

    def test_bind_works(self):
        tmpl = uccsd_ansatz(n_electrons=1, n_qubits=4)
        d = tmpl.bind([0.0] * tmpl.n_parameters)
        assert _all_params_bound(d)

    def test_estimator_integration(self):
        tmpl = uccsd_ansatz(n_electrons=1, n_qubits=2)
        d = tmpl.bind([0.0] * tmpl.n_parameters)
        result = Estimator().run([d], ["ZI"])
        assert math.isfinite(result.values[0])

    def test_invalid_n_electrons_zero(self):
        with pytest.raises(TemplateError):
            uccsd_ansatz(n_electrons=0, n_qubits=4)

    def test_invalid_n_electrons_equals_n_qubits(self):
        with pytest.raises(TemplateError, match="virtual orbital"):
            uccsd_ansatz(n_electrons=4, n_qubits=4)

    def test_invalid_n_electrons_exceeds_n_qubits(self):
        with pytest.raises(TemplateError, match="virtual orbital"):
            uccsd_ansatz(n_electrons=5, n_qubits=4)

    def test_invalid_n_qubits_one(self):
        with pytest.raises(TemplateError):
            uccsd_ansatz(n_electrons=1, n_qubits=1)


# ===========================================================================
# __init__ re-exports
# ===========================================================================

class TestTemplatesInit:
    def test_template_error_exported(self):
        from quantx_engine import TemplateError as TE
        assert TE is TemplateError

    def test_circuit_template_exported(self):
        from quantx_engine import CircuitTemplate as CT
        assert CT is CircuitTemplate

    def test_hardware_efficient_ansatz_exported(self):
        from quantx_engine import hardware_efficient_ansatz as hea
        assert hea is hardware_efficient_ansatz

    def test_qaoa_circuit_exported(self):
        from quantx_engine import qaoa_circuit as qc
        assert qc is qaoa_circuit

    def test_ghz_circuit_exported(self):
        from quantx_engine import ghz_circuit as gc
        assert gc is ghz_circuit

    def test_quantum_fourier_transform_exported(self):
        from quantx_engine import quantum_fourier_transform as qft
        assert qft is quantum_fourier_transform

    def test_uccsd_ansatz_exported(self):
        from quantx_engine import uccsd_ansatz as ua
        assert ua is uccsd_ansatz
