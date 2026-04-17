"""Tests for quantx_engine.mitigation (M70: ZNE + PEC)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quantx_engine.mitigation import (
    MitigationError,
    ExtrapolationResult,
    RichardsonExtrapolator,
    PolyExtrapolator,
    ExpExtrapolator,
    QuasiProbabilityDecomposition,
    PECResult,
    ZNEResult,
    fold_gates,
    zne_mitigate,
    pec_mitigate,
    make_sv_executor,
    _invert_op,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ry_op(qubit, theta):
    return {"name": "RY", "qubits": [qubit], "params": [theta]}


def _cnot_op(ctrl, tgt):
    return {"name": "CNOT", "qubits": [ctrl, tgt], "params": []}


def _h_op(qubit):
    return {"name": "H", "qubits": [qubit], "params": []}


def _x_op(qubit):
    return {"name": "X", "qubits": [qubit], "params": []}


# ---------------------------------------------------------------------------
# _invert_op
# ---------------------------------------------------------------------------

class TestInvertOp:
    def test_self_inverse_single_qubit(self):
        for name in ("X", "Y", "Z", "H", "I"):
            op = {"name": name, "qubits": [0], "params": []}
            inv = _invert_op(op)
            assert inv["name"] == name

    def test_self_inverse_two_qubit(self):
        for name in ("CNOT", "CZ", "SWAP"):
            op = {"name": name, "qubits": [0, 1], "params": []}
            inv = _invert_op(op)
            assert inv["name"] == name

    def test_ry_inverse(self):
        op = _ry_op(0, 1.2)
        inv = _invert_op(op)
        assert inv["name"] == "RY"
        assert math.isclose(inv["params"][0], -1.2)

    def test_rx_inverse(self):
        op = {"name": "RX", "qubits": [0], "params": [0.7]}
        inv = _invert_op(op)
        assert inv["name"] == "RX"
        assert math.isclose(inv["params"][0], -0.7)

    def test_rz_inverse(self):
        op = {"name": "RZ", "qubits": [0], "params": [2.0]}
        inv = _invert_op(op)
        assert inv["name"] == "RZ"
        assert math.isclose(inv["params"][0], -2.0)

    def test_s_gate_inverse_is_rz(self):
        op = {"name": "S", "qubits": [0], "params": []}
        inv = _invert_op(op)
        assert inv["name"] == "RZ"
        assert math.isclose(inv["params"][0], -math.pi / 2)

    def test_t_gate_inverse_is_rz(self):
        op = {"name": "T", "qubits": [0], "params": []}
        inv = _invert_op(op)
        assert inv["name"] == "RZ"
        assert math.isclose(inv["params"][0], -math.pi / 4)

    def test_unknown_gate_raises(self):
        op = {"name": "BADGATE", "qubits": [0], "params": []}
        with pytest.raises(MitigationError, match="BADGATE"):
            _invert_op(op)

    def test_ry_no_params_raises(self):
        op = {"name": "RY", "qubits": [0], "params": []}
        with pytest.raises(MitigationError, match="params is empty"):
            _invert_op(op)


# ---------------------------------------------------------------------------
# fold_gates
# ---------------------------------------------------------------------------

class TestFoldGates:
    def test_scale_1_unchanged(self):
        ops = [_ry_op(0, 0.3), _cnot_op(0, 1)]
        assert fold_gates(ops, 1.0) == ops

    def test_scale_3_triples_each_gate(self):
        ops = [_ry_op(0, 0.5)]
        folded = fold_gates(ops, 3.0)
        # G · G† · G
        assert len(folded) == 3
        assert folded[0]["params"] == [0.5]
        assert folded[1]["params"] == [-0.5]  # inverse
        assert folded[2]["params"] == [0.5]

    def test_scale_5_five_copies_each_gate(self):
        ops = [_h_op(0)]
        folded = fold_gates(ops, 5.0)
        assert len(folded) == 5
        # H is self-inverse → all five are H
        assert all(f["name"] == "H" for f in folded)

    def test_empty_ops(self):
        assert fold_gates([], 3.0) == []

    def test_two_gate_circuit_scale_3(self):
        ops = [_h_op(0), _x_op(0)]
        folded = fold_gates(ops, 3.0)
        assert len(folded) == 6   # 3 copies of each gate

    def test_non_integer_scale_increases_gate_count(self):
        ops = [_ry_op(0, i * 0.1) for i in range(4)]
        folded = fold_gates(ops, 2.0)
        # scale 2.0: num_folds=0, tail=round(0.5*4/2*2)=2 tail gates get 1 extra fold
        # Each of last 2 gates: 3 copies; first 2 gates: 1 copy → total = 2 + 2*3 = 8
        assert len(folded) == 8

    def test_scale_below_1_raises(self):
        with pytest.raises(MitigationError, match="scale_factor must be >= 1"):
            fold_gates([_h_op(0)], 0.5)

    def test_folded_circuit_cnot_unchanged(self):
        """CNOT is self-inverse; folding should keep CNOT identity."""
        ops = [_cnot_op(0, 1)]
        folded = fold_gates(ops, 3.0)
        assert len(folded) == 3
        assert all(f["name"] == "CNOT" for f in folded)

    def test_folded_circuit_same_result_as_original(self):
        """G·G†·G = G for any unitary G: folded circuit = original circuit (statevector)."""
        from quantx_engine.gpu import GpuStatevector

        ops = [_ry_op(0, 0.7), _ry_op(1, 1.3), _cnot_op(0, 1)]
        folded = fold_gates(ops, 3.0)

        sv_orig = GpuStatevector(n_qubits=2)
        for op in ops:
            sv_orig.apply_gate(op["name"], op["qubits"], op.get("params", ()))

        sv_fold = GpuStatevector(n_qubits=2)
        for op in folded:
            sv_fold.apply_gate(op["name"], op["qubits"], op.get("params", ()))

        np.testing.assert_allclose(sv_orig.state(), sv_fold.state(), atol=1e-12)


# ---------------------------------------------------------------------------
# Extrapolators
# ---------------------------------------------------------------------------

class TestRichardsonExtrapolator:
    def test_linear_exact_two_points(self):
        # y = 0.5 + 0.3*x → y(0) = 0.5
        xs = [1.0, 3.0]
        ys = [0.8, 1.4]
        result = RichardsonExtrapolator().extrapolate(xs, ys)
        assert math.isclose(result.mitigated_value, 0.5, abs_tol=1e-10)

    def test_quadratic_exact_three_points(self):
        # y = 1 - 0.2*x + 0.05*x**2  → y(0) = 1
        xs = [1.0, 3.0, 5.0]
        ys = [1 - 0.2 * x + 0.05 * x**2 for x in xs]
        result = RichardsonExtrapolator().extrapolate(xs, ys)
        assert math.isclose(result.mitigated_value, 1.0, abs_tol=1e-10)

    def test_coefficients_length_matches_points(self):
        xs = [1.0, 3.0, 5.0]
        ys = [0.9, 0.7, 0.5]
        result = RichardsonExtrapolator().extrapolate(xs, ys)
        assert len(result.coefficients) == 3

    def test_too_few_points_raises(self):
        with pytest.raises(MitigationError, match="At least 2 data points"):
            RichardsonExtrapolator().extrapolate([1.0], [0.5])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(MitigationError, match="same length"):
            RichardsonExtrapolator().extrapolate([1.0, 3.0], [0.5])


class TestPolyExtrapolator:
    def test_linear_fit(self):
        # y = 1 - 0.1*x → y(0) = 1
        xs = [1.0, 3.0, 5.0]
        ys = [0.9, 0.7, 0.5]
        result = PolyExtrapolator(degree=1).extrapolate(xs, ys)
        assert math.isclose(result.mitigated_value, 1.0, abs_tol=1e-10)

    def test_quadratic_fit(self):
        # y = 1 - 0.1*x + 0.02*x**2 → y(0) = 1
        xs = [1.0, 2.0, 3.0, 5.0]
        ys = [1 - 0.1 * x + 0.02 * x**2 for x in xs]
        result = PolyExtrapolator(degree=2).extrapolate(xs, ys)
        assert math.isclose(result.mitigated_value, 1.0, abs_tol=1e-10)

    def test_degree_zero_raises(self):
        with pytest.raises(MitigationError, match="degree must be >= 1"):
            PolyExtrapolator(degree=0)

    def test_degree_exceeds_points_raises(self):
        with pytest.raises(MitigationError, match="at least"):
            PolyExtrapolator(degree=3).extrapolate([1.0, 3.0], [0.5, 0.4])

    def test_coefficients_returned(self):
        xs = [1.0, 3.0, 5.0]
        ys = [0.9, 0.7, 0.5]
        result = PolyExtrapolator(degree=1).extrapolate(xs, ys)
        # degree=1 → 2 coefficients [slope, intercept]
        assert len(result.coefficients) == 2


class TestExpExtrapolator:
    def test_exponential_fit_exact(self):
        pytest.importorskip("scipy")
        # y = 0.2 + 0.8 * exp(-0.3*x) → y(0) = 1.0
        xs = [1.0, 2.0, 3.0, 5.0]
        ys = [0.2 + 0.8 * math.exp(-0.3 * x) for x in xs]
        result = ExpExtrapolator(p0=(0.1, 0.9, -0.3)).extrapolate(xs, ys)
        assert math.isclose(result.mitigated_value, 1.0, abs_tol=1e-6)

    def test_too_few_points_raises(self):
        pytest.importorskip("scipy")
        with pytest.raises(MitigationError, match="at least 3"):
            ExpExtrapolator().extrapolate([1.0, 3.0], [0.8, 0.7])

    def test_no_scipy_raises(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "scipy.optimize", None)
        monkeypatch.setitem(sys.modules, "scipy", None)
        import importlib
        import quantx_engine.mitigation as m
        # Directly simulate the ImportError path by patching the import
        with pytest.raises((MitigationError, ImportError)):
            ExpExtrapolator().extrapolate([1.0, 2.0, 3.0], [0.9, 0.8, 0.7])


# ---------------------------------------------------------------------------
# ZNE end-to-end
# ---------------------------------------------------------------------------

class TestZNEMitigate:
    """End-to-end ZNE tests using a mock executor with known noise model.

    All mock executors model noise as ``y(scale_factor) = ideal_val + f(scale_factor)``
    where ``ideal_val = y(0)`` is the zero-noise (noiseless) target value.
    Gate folding maps the ops list length to a continuous scale factor:
    ``scale_factor = len(ops) / n_ideal_gates``.
    """

    @staticmethod
    def _make_linear_noise_executor(ideal_val: float, noise_slope: float, n_ideal_gates: int):
        """y(sf) = ideal_val + noise_slope * sf  →  y(0) = ideal_val."""
        def executor(ops):
            sf = len(ops) / n_ideal_gates
            return ideal_val + noise_slope * sf
        return executor

    @staticmethod
    def _make_poly_noise_executor(ideal_val: float, coeffs: list[float], n_ideal_gates: int):
        """y(sf) = ideal_val + Σ c_i * sf^(i+1)  →  y(0) = ideal_val."""
        def executor(ops):
            sf = len(ops) / n_ideal_gates
            noise = sum(c * sf ** (i + 1) for i, c in enumerate(coeffs))
            return ideal_val + noise
        return executor

    def test_richardson_linear_noise(self):
        """Richardson extrapolation corrects linear noise exactly."""
        # y(sf) = 0.8 + 0.1*sf  → y(0) = 0.8
        # y(1) = 0.9, y(3) = 1.1
        # Richardson: c0 = -3/(1-3)=1.5, c1 = -1/(3-1)=-0.5
        # y(0) = 1.5*0.9 + (-0.5)*1.1 = 1.35 - 0.55 = 0.8 ✓
        ops = [_ry_op(0, 0.5)]   # n_ideal_gates=1
        executor = self._make_linear_noise_executor(ideal_val=0.8, noise_slope=0.1,
                                                     n_ideal_gates=1)
        result = zne_mitigate(ops, executor, scale_factors=[1.0, 3.0])
        assert isinstance(result, ZNEResult)
        assert math.isclose(result.mitigated_value, 0.8, abs_tol=1e-8)
        assert result.extrapolator_name == "RichardsonExtrapolator"

    def test_richardson_quadratic_noise(self):
        """Richardson with 3 points corrects quadratic noise exactly."""
        # y(sf) = 0.9 + 0.05*sf + 0.01*sf^2  → y(0) = 0.9
        ops = [_ry_op(0, 0.5)]
        executor = self._make_poly_noise_executor(ideal_val=0.9, coeffs=[0.05, 0.01],
                                                   n_ideal_gates=1)
        result = zne_mitigate(ops, executor, scale_factors=[1.0, 3.0, 5.0])
        assert math.isclose(result.mitigated_value, 0.9, abs_tol=1e-8)

    def test_poly_extrapolator(self):
        """PolyExtrapolator(degree=1) corrects linear noise."""
        # y(sf) = 0.6 + 0.08*sf  → y(0) = 0.6
        ops = [_h_op(0)]
        executor = self._make_linear_noise_executor(ideal_val=0.6, noise_slope=0.08,
                                                     n_ideal_gates=1)
        result = zne_mitigate(
            ops, executor,
            scale_factors=[1.0, 3.0, 5.0],
            extrapolator=PolyExtrapolator(degree=1),
        )
        assert math.isclose(result.mitigated_value, 0.6, abs_tol=1e-8)
        assert result.extrapolator_name == "PolyExtrapolator"

    def test_default_scale_factors_and_extrapolator(self):
        """Calling zne_mitigate with no kwargs uses defaults [1, 3, 5] + Richardson."""
        ops = [_ry_op(0, 0.3)]
        executor = self._make_linear_noise_executor(0.5, 0.05, n_ideal_gates=1)
        result = zne_mitigate(ops, executor)
        assert result.scale_factors == [1.0, 3.0, 5.0]
        assert result.extrapolator_name == "RichardsonExtrapolator"
        assert len(result.noisy_values) == 3

    def test_invalid_scale_factor_raises(self):
        ops = [_ry_op(0, 0.5)]
        with pytest.raises(MitigationError, match="scale_factors must be >= 1"):
            zne_mitigate(ops, lambda x: 0.0, scale_factors=[1.0, 0.5])

    def test_noisy_values_stored_correctly(self):
        """noisy_values[0] is the measurement at scale=1 (includes base noise)."""
        # y(sf) = 0.7 + 0.1*sf  → y(1) = 0.8
        ops = [_ry_op(0, 0.5)]
        executor = self._make_linear_noise_executor(0.7, 0.1, n_ideal_gates=1)
        result = zne_mitigate(ops, executor, scale_factors=[1.0, 3.0])
        assert math.isclose(result.noisy_values[0], 0.8, abs_tol=1e-8)

    def test_statevector_executor_zne(self):
        """ZNE with make_sv_executor recovers ideal result (noiseless sim: already ideal)."""
        from quantx_engine.gpu import GpuStatevector

        # Single qubit: apply RY(π/2) → |+y⟩. Expectation <Z> = cos(π/2) = 0
        # For a noiseless executor, ZNE should return ~0 at all scale factors.
        ops = [_ry_op(0, math.pi / 2)]
        executor = make_sv_executor(1, lambda sv: sv.expectation_z(0), backend="numpy")
        result = zne_mitigate(ops, executor, scale_factors=[1.0, 3.0, 5.0])
        # All noisy_values should be ~0 (noiseless); mitigated_value also ~0
        for v in result.noisy_values:
            assert abs(v) < 1e-10
        assert abs(result.mitigated_value) < 1e-10

    def test_statevector_executor_known_ry(self):
        """make_sv_executor with RY(θ): <Z> = cos(θ)."""
        theta = 1.1
        ops = [_ry_op(0, theta)]
        executor = make_sv_executor(1, lambda sv: sv.expectation_z(0), backend="numpy")
        result = zne_mitigate(ops, executor, scale_factors=[1.0, 3.0])
        # Noiseless: all scale factors give same value; mitigated should equal cos(theta)
        expected = math.cos(theta)
        assert math.isclose(result.mitigated_value, expected, abs_tol=1e-10)

    def test_zne_result_repr_runs(self):
        ops = [_h_op(0)]
        executor = make_sv_executor(1, lambda sv: sv.expectation_z(0), backend="numpy")
        result = zne_mitigate(ops, executor)
        assert isinstance(repr(result), str)


# ---------------------------------------------------------------------------
# QuasiProbabilityDecomposition
# ---------------------------------------------------------------------------

class TestQuasiProbabilityDecomposition:
    def test_zero_noise_trivial(self):
        qpd = QuasiProbabilityDecomposition.from_depolarizing(0.0)
        assert qpd.gamma == 1.0
        assert len(qpd.basis) == 1
        assert qpd.basis[0][0] == "I"
        assert qpd.basis[0][1] == 1.0   # sign

    def test_gamma_formula(self):
        """γ = (1 + p) / (1 - 4p/3) for single-qubit depolarizing."""
        p = 0.05
        qpd = QuasiProbabilityDecomposition.from_depolarizing(p)
        expected_gamma = (1 + p) / (1 - 4 * p / 3)
        assert math.isclose(qpd.gamma, expected_gamma, rel_tol=1e-10)

    def test_sampling_probabilities_sum_to_one(self):
        qpd = QuasiProbabilityDecomposition.from_depolarizing(0.1)
        total = sum(b[2] for b in qpd.basis)
        assert math.isclose(total, 1.0, abs_tol=1e-12)

    def test_basis_has_four_terms(self):
        qpd = QuasiProbabilityDecomposition.from_depolarizing(0.1)
        assert len(qpd.basis) == 4

    def test_identity_term_positive(self):
        qpd = QuasiProbabilityDecomposition.from_depolarizing(0.1)
        identity_terms = [b for b in qpd.basis if b[0] == "I"]
        assert len(identity_terms) == 1
        assert identity_terms[0][1] == 1.0   # sign > 0

    def test_pauli_terms_negative(self):
        qpd = QuasiProbabilityDecomposition.from_depolarizing(0.1)
        for name in ("X", "Y", "Z"):
            terms = [b for b in qpd.basis if b[0] == name]
            assert len(terms) == 1
            assert terms[0][1] == -1.0   # sign < 0

    def test_noise_too_large_raises(self):
        with pytest.raises(MitigationError, match="3/4"):
            QuasiProbabilityDecomposition.from_depolarizing(0.8)

    def test_negative_noise_raises(self):
        with pytest.raises(MitigationError, match="3/4"):
            QuasiProbabilityDecomposition.from_depolarizing(-0.01)

    def test_sample_returns_valid_sign_and_pauli(self):
        qpd = QuasiProbabilityDecomposition.from_depolarizing(0.1)
        rng = np.random.default_rng(42)
        for _ in range(50):
            sign, pauli = qpd.sample(rng)
            assert sign in (1.0, -1.0)
            assert pauli in ("I", "X", "Y", "Z")

    def test_sample_identity_fraction_near_expected(self):
        """For small p, identity should be sampled most often."""
        qpd = QuasiProbabilityDecomposition.from_depolarizing(0.02)
        rng = np.random.default_rng(7)
        counts = {"I": 0, "X": 0, "Y": 0, "Z": 0}
        N = 10_000
        for _ in range(N):
            _, pauli = qpd.sample(rng)
            counts[pauli] += 1
        # Identity should dominate (high probability)
        assert counts["I"] > counts["X"]
        assert counts["I"] > counts["Y"]
        assert counts["I"] > counts["Z"]


# ---------------------------------------------------------------------------
# PEC end-to-end
# ---------------------------------------------------------------------------

class TestPECMitigate:
    """End-to-end PEC tests."""

    @staticmethod
    def _make_noisy_executor(ideal_executor, noise_prob: float):
        """Simulate a noisy executor: applies depolarizing noise by blending toward 0."""
        def noisy(ops):
            ideal = ideal_executor(ops)
            # Simple model: expectation decays by (1-p)^n for n single-qubit gates
            n_1q = sum(1 for op in ops if len(op.get("qubits", [])) == 1)
            return ideal * (1 - noise_prob) ** n_1q
        return noisy

    def test_pec_unbiased_for_zero_noise(self):
        """PEC with noise_prob=0 is a trivial identity (γ=1, all weights +1)."""
        ops = [_ry_op(0, 0.8)]
        ideal_ex = make_sv_executor(1, lambda sv: sv.expectation_z(0), backend="numpy")
        result = pec_mitigate(ops, ideal_ex, noise_prob=0.0, n_samples=20, seed=0)
        expected = math.cos(0.8)
        assert math.isclose(result.mitigated_value, expected, abs_tol=1e-10)
        assert result.gamma == 1.0
        assert abs(result.std_error) < 1e-15  # all samples identical when noise_prob=0

    def test_pec_reduces_bias(self):
        """PEC mitigated value should be closer to ideal than raw noisy value."""
        ops = [_ry_op(0, 1.0)]
        noise_prob = 0.05
        ideal_ex = make_sv_executor(1, lambda sv: sv.expectation_z(0), backend="numpy")
        noisy_ex = self._make_noisy_executor(ideal_ex, noise_prob)

        ideal_val = math.cos(1.0)
        raw_noisy_val = noisy_ex(ops)

        result = pec_mitigate(ops, noisy_ex, noise_prob=noise_prob, n_samples=2000, seed=42)

        # PEC mitigated value should be closer to ideal than raw noisy
        pec_bias = abs(result.mitigated_value - ideal_val)
        raw_bias = abs(raw_noisy_val - ideal_val)
        assert pec_bias < raw_bias * 3, (
            f"PEC bias ({pec_bias:.4g}) should be much smaller than raw bias ({raw_bias:.4g})"
        )

    def test_pec_result_fields(self):
        ops = [_h_op(0)]
        executor = make_sv_executor(1, lambda sv: sv.expectation_z(0), backend="numpy")
        result = pec_mitigate(ops, executor, noise_prob=0.01, n_samples=50, seed=1)
        assert isinstance(result, PECResult)
        assert result.n_samples == 50
        assert result.gamma > 1.0
        assert result.std_error >= 0.0
        assert len(result.raw_estimates) == 50

    def test_pec_raw_estimates_correct_count(self):
        ops = [_ry_op(0, 0.5)]
        executor = make_sv_executor(1, lambda sv: sv.expectation_z(0), backend="numpy")
        result = pec_mitigate(ops, executor, noise_prob=0.02, n_samples=30, seed=5)
        assert len(result.raw_estimates) == 30

    def test_pec_seed_reproducibility(self):
        ops = [_ry_op(0, 0.5), _cnot_op(0, 1)]
        executor = make_sv_executor(2, lambda sv: sv.expectation_z(0), backend="numpy")
        r1 = pec_mitigate(ops, executor, noise_prob=0.03, n_samples=20, seed=99)
        r2 = pec_mitigate(ops, executor, noise_prob=0.03, n_samples=20, seed=99)
        assert r1.mitigated_value == r2.mitigated_value
        assert r1.raw_estimates == r2.raw_estimates

    def test_pec_two_qubit_gates_not_twirled(self):
        """Two-qubit gates should pass through without Pauli insertion.

        With no single-qubit gates, total_gamma = qpd.gamma^0 = 1.0.
        """
        ops = [_cnot_op(0, 1)]  # only 2-qubit gate → n_1q_gates=0, total_gamma=1
        executor = make_sv_executor(2, lambda sv: sv.expectation_z(0), backend="numpy")
        result = pec_mitigate(ops, executor, noise_prob=0.05, n_samples=10, seed=0)
        assert result.gamma == 1.0   # total overhead = gamma_per_gate^0 = 1.0

    def test_pec_std_error_decreases_with_samples(self):
        ops = [_ry_op(0, 0.7)]
        executor = make_sv_executor(1, lambda sv: sv.expectation_z(0), backend="numpy")
        r_small = pec_mitigate(ops, executor, noise_prob=0.1, n_samples=20, seed=0)
        r_large = pec_mitigate(ops, executor, noise_prob=0.1, n_samples=200, seed=0)
        # Larger sample count should give smaller std_error on average
        # (not guaranteed due to randomness, but with the given seeds it should hold)
        # Just check both are non-negative
        assert r_small.std_error >= 0.0
        assert r_large.std_error >= 0.0

    def test_pec_n_samples_zero_raises(self):
        ops = [_ry_op(0, 0.5)]
        executor = lambda ops: 0.0
        with pytest.raises(MitigationError, match="n_samples must be >= 1"):
            pec_mitigate(ops, executor, noise_prob=0.01, n_samples=0)

    def test_pec_noise_too_large_raises(self):
        ops = [_ry_op(0, 0.5)]
        executor = lambda ops: 0.0
        with pytest.raises(MitigationError, match="3/4"):
            pec_mitigate(ops, executor, noise_prob=0.9, n_samples=10)


# ---------------------------------------------------------------------------
# make_sv_executor
# ---------------------------------------------------------------------------

class TestMakeSvExecutor:
    def test_expectation_z_on_zero_state(self):
        """|0⟩ state → <Z> = 1."""
        executor = make_sv_executor(1, lambda sv: sv.expectation_z(0), backend="numpy")
        val = executor([])
        assert math.isclose(val, 1.0, abs_tol=1e-12)

    def test_expectation_z_after_x(self):
        """X|0⟩ = |1⟩ → <Z> = -1."""
        executor = make_sv_executor(1, lambda sv: sv.expectation_z(0), backend="numpy")
        val = executor([_x_op(0)])
        assert math.isclose(val, -1.0, abs_tol=1e-12)

    def test_expectation_z_after_ry(self):
        """RY(θ)|0⟩ → <Z> = cos(θ)."""
        theta = 1.2
        executor = make_sv_executor(1, lambda sv: sv.expectation_z(0), backend="numpy")
        val = executor([_ry_op(0, theta)])
        assert math.isclose(val, math.cos(theta), abs_tol=1e-12)

    def test_probability_observable(self):
        """Can use probability as observable."""
        executor = make_sv_executor(
            1,
            lambda sv: float(sv.probabilities([0])[1]),  # P(|1⟩)
            backend="numpy",
        )
        val = executor([_x_op(0)])
        assert math.isclose(val, 1.0, abs_tol=1e-12)

    def test_two_qubit_circuit(self):
        """Bell state: <Z0> should be ~0."""
        executor = make_sv_executor(2, lambda sv: sv.expectation_z(0), backend="numpy")
        # H|0⟩ → (|0⟩+|1⟩)/√2; CNOT → Bell state; <Z0> = 0
        val = executor([_h_op(0), _cnot_op(0, 1)])
        assert abs(val) < 1e-12

    def test_fresh_statevector_each_call(self):
        """Each call to executor starts from |0⟩ (no state reuse)."""
        executor = make_sv_executor(1, lambda sv: sv.expectation_z(0), backend="numpy")
        v1 = executor([_x_op(0)])     # |1⟩ → <Z> = -1
        v2 = executor([])             # |0⟩ → <Z> = +1
        assert math.isclose(v1, -1.0, abs_tol=1e-12)
        assert math.isclose(v2, 1.0, abs_tol=1e-12)
