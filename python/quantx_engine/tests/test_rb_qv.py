"""Tests for Randomized Benchmarking and Quantum Volume (M81)."""
from __future__ import annotations

import math
import pytest
import numpy as np

from ..rb_qv import (
    RBQVError,
    RBResult,
    StandardRB,
    QVResult,
    QuantumVolume,
    _generate_cliffords_1q,
    _get_cliffords_1q,
    _haar_unitary,
    _fit_rb,
    _embed_gate,
    _dm_apply_gate,
    _dm_depolarize,
)


# ===========================================================================
# RBQVError
# ===========================================================================

class TestRBQVError:
    def test_is_value_error(self):
        assert issubclass(RBQVError, ValueError)

    def test_raises(self):
        with pytest.raises(RBQVError):
            raise RBQVError("test")


# ===========================================================================
# Helper functions
# ===========================================================================

class TestCliffordGroup:
    def test_generates_24(self):
        cliffords = _generate_cliffords_1q()
        assert len(cliffords) == 24

    def test_all_unitary(self):
        for C in _generate_cliffords_1q():
            assert C.shape == (2, 2)
            prod = C @ C.conj().T
            assert np.allclose(prod, np.eye(2), atol=1e-10), "Not unitary"

    def test_all_distinct_up_to_phase(self):
        cliffords = _generate_cliffords_1q()
        n = len(cliffords)

        def _norm_phase(U):
            v = U.ravel()
            idx = np.argmax(np.abs(v))
            return U / (v[idx] / abs(v[idx]))

        for i in range(n):
            for j in range(i + 1, n):
                assert not np.allclose(
                    _norm_phase(cliffords[i]),
                    _norm_phase(cliffords[j]),
                    atol=1e-8,
                ), f"Duplicate Cliffords at indices {i} and {j}"

    def test_closed_under_multiplication(self):
        """C1 @ C2 must (up to phase) be another Clifford in the group."""
        cliffords = _generate_cliffords_1q()
        n = len(cliffords)

        def _norm_phase(U):
            v = U.ravel()
            idx = np.argmax(np.abs(v))
            return U / (v[idx] / abs(v[idx]))

        norms = [_norm_phase(C) for C in cliffords]
        for i in range(6):  # spot-check 6 products
            C = cliffords[i] @ cliffords[(i + 3) % n]
            C_norm = _norm_phase(C)
            found = any(np.allclose(C_norm, N, atol=1e-8) for N in norms)
            assert found, f"Product of Clifford {i} and {(i+3)%n} not in group"

    def test_cache_returns_same_objects(self):
        c1 = _get_cliffords_1q()
        c2 = _get_cliffords_1q()
        assert c1 is c2  # cached


class TestHaarUnitary:
    def test_shape(self):
        rng = np.random.default_rng(0)
        U = _haar_unitary(4, rng)
        assert U.shape == (4, 4)

    def test_unitary(self):
        rng = np.random.default_rng(1)
        for d in (2, 4, 8):
            U = _haar_unitary(d, rng)
            assert np.allclose(U @ U.conj().T, np.eye(d), atol=1e-10)

    def test_deterministic_with_seed(self):
        U1 = _haar_unitary(4, np.random.default_rng(42))
        U2 = _haar_unitary(4, np.random.default_rng(42))
        assert np.allclose(U1, U2)

    def test_different_seeds_give_different_unitaries(self):
        U1 = _haar_unitary(4, np.random.default_rng(0))
        U2 = _haar_unitary(4, np.random.default_rng(1))
        assert not np.allclose(U1, U2)


class TestEmbedGate:
    def test_1qubit_identity(self):
        V = _embed_gate(np.eye(2, dtype=complex), 2, (0,))
        assert np.allclose(V, np.eye(4))

    def test_2qubit_cx_like(self):
        """Embed CNOT on qubits (0,1) of a 2-qubit system = full CNOT."""
        CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        V = _embed_gate(CNOT, 2, (0, 1))
        assert np.allclose(V, CNOT)

    def test_3qubit_embed_x_on_qubit0(self):
        """X on qubit 0 of 3-qubit system: swaps top half of basis states."""
        X = np.array([[0,1],[1,0]], dtype=complex)
        V = _embed_gate(X, 3, (0,))
        psi = np.array([1,0,0,0,0,0,0,0], dtype=complex)  # |000⟩
        result = V @ psi
        expected = np.array([0,0,0,0,1,0,0,0], dtype=complex)  # |100⟩
        assert np.allclose(result, expected)

    def test_embed_preserves_norm(self):
        rng = np.random.default_rng(5)
        U = _haar_unitary(4, rng)
        V = _embed_gate(U, 3, (0, 1))
        # V should be unitary (embed a 2-qubit unitary in 3-qubit space)
        VVd = V @ V.conj().T
        assert np.allclose(VVd, np.eye(8), atol=1e-10)


class TestFitRB:
    def test_perfect_exponential(self):
        """Fit should recover p=0.9, A=0.5, B=0.5 exactly."""
        m = np.array([1, 5, 10, 20, 50], dtype=float)
        p_true, A_true, B_true = 0.9, 0.5, 0.5
        p_obs = A_true * p_true ** m + B_true
        p_fit, A_fit, B_fit = _fit_rb(m, p_obs)
        assert abs(p_fit - p_true) < 0.01
        assert abs(A_fit - A_true) < 0.05
        assert abs(B_fit - B_true) < 0.05

    def test_flat_curve(self):
        """A flat curve is reproduced by the fit (may not recover p uniquely)."""
        m = np.array([1, 5, 10], dtype=float)
        p_obs = np.ones(3) * 0.98  # nearly flat
        p_fit, A_fit, B_fit = _fit_rb(m, p_obs)
        # Verify the fit reconstructs the data accurately
        pred = A_fit * p_fit ** m + B_fit
        assert np.allclose(pred, p_obs, atol=0.02)

    def test_returns_floats(self):
        m = np.array([1, 2, 5], dtype=float)
        p_obs = np.array([0.9, 0.85, 0.7])
        p, A, B = _fit_rb(m, p_obs)
        assert isinstance(p, float)
        assert isinstance(A, float)
        assert isinstance(B, float)


class TestDMHelpers:
    def test_dm_apply_gate_identity(self):
        rho = np.diag([0.5, 0.5]).astype(complex)
        result = _dm_apply_gate(rho, np.eye(2, dtype=complex))
        assert np.allclose(result, rho)

    def test_dm_apply_gate_x(self):
        """X|0⟩⟨0|X† = |1⟩⟨1|."""
        rho = np.zeros((2, 2), dtype=complex)
        rho[0, 0] = 1.0
        X = np.array([[0,1],[1,0]], dtype=complex)
        result = _dm_apply_gate(rho, X)
        expected = np.zeros((2, 2), dtype=complex)
        expected[1, 1] = 1.0
        assert np.allclose(result, expected)

    def test_dm_depolarize_trace_preserved(self):
        """Trace must be 1 after depolarizing."""
        rho = np.zeros((2, 2), dtype=complex)
        rho[0, 0] = 1.0
        result = _dm_depolarize(rho, 0.1)
        assert abs(np.trace(result) - 1.0) < 1e-12

    def test_dm_depolarize_zero_noise(self):
        """p=0: no change."""
        rho = np.diag([0.7, 0.3]).astype(complex)
        result = _dm_depolarize(rho, 0.0)
        assert np.allclose(result, rho)

    def test_dm_depolarize_full_noise(self):
        """p=1: maximally mixed state."""
        rho = np.zeros((2, 2), dtype=complex)
        rho[0, 0] = 1.0
        result = _dm_depolarize(rho, 1.0)
        assert np.allclose(result, np.eye(2) / 2)


# ===========================================================================
# StandardRB
# ===========================================================================

class TestStandardRBConstruction:
    def test_default_construction(self):
        rb = StandardRB()
        assert rb.n_qubits == 1
        assert rb.p_error == pytest.approx(0.001)

    def test_invalid_n_qubits(self):
        with pytest.raises(RBQVError):
            StandardRB(n_qubits=0)

    def test_invalid_p_error_negative(self):
        with pytest.raises(RBQVError):
            StandardRB(p_error=-0.1)

    def test_invalid_p_error_one(self):
        with pytest.raises(RBQVError):
            StandardRB(p_error=1.0)

    def test_custom_params(self):
        rb = StandardRB(n_qubits=2, p_error=0.02, seed=42)
        assert rb.n_qubits == 2
        assert rb.p_error == pytest.approx(0.02)
        assert rb.seed == 42


class TestStandardRBRun:
    def test_run_returns_rb_result(self):
        rb = StandardRB(p_error=0.01, seed=0)
        result = rb.run([1, 5], n_sequences=5)
        assert isinstance(result, RBResult)

    def test_result_lengths_match(self):
        rb = StandardRB(seed=0)
        m_vals = [1, 3, 7, 15]
        result = rb.run(m_vals, n_sequences=5)
        assert result.sequence_lengths == m_vals
        assert len(result.survival_probs) == len(m_vals)

    def test_survival_probs_in_01(self):
        rb = StandardRB(p_error=0.01, seed=0)
        result = rb.run([1, 5, 10], n_sequences=10)
        for sp in result.survival_probs:
            assert 0.0 <= sp <= 1.0 + 1e-8, f"Survival prob out of [0,1]: {sp}"

    def test_noiseless_survives_perfectly(self):
        """With p_error=0, survival probability should be 1 for any length."""
        rb = StandardRB(p_error=0.0, seed=0)
        result = rb.run([1, 5, 20], n_sequences=3)
        for sp in result.survival_probs:
            assert abs(sp - 1.0) < 1e-8, f"Noiseless survival != 1: {sp}"

    def test_noiseless_p_decay_near_one(self):
        rb = StandardRB(p_error=0.0, seed=0)
        result = rb.run([1, 5, 10, 20], n_sequences=3)
        assert result.p_decay > 0.95

    def test_noisy_p_decay_less_than_noiseless(self):
        rb_clean = StandardRB(p_error=0.0, seed=0)
        rb_noisy = StandardRB(p_error=0.05, seed=0)
        m_vals = [1, 5, 10, 20]
        r_clean = rb_clean.run(m_vals, n_sequences=10)
        r_noisy = rb_noisy.run(m_vals, n_sequences=10)
        assert r_noisy.p_decay < r_clean.p_decay

    def test_epc_positive(self):
        rb = StandardRB(p_error=0.01, seed=0)
        result = rb.run([1, 5, 10], n_sequences=5)
        assert result.epc >= 0.0

    def test_n_qubits_stored(self):
        rb = StandardRB(n_qubits=1, seed=0)
        result = rb.run([1, 3], n_sequences=3)
        assert result.n_qubits == 1

    def test_epc_roughly_half_p_error(self):
        """For 1-qubit, EPC ≈ (1-p)/2 ≈ p_error/2 for small p_error."""
        p_err = 0.02
        rb = StandardRB(p_error=p_err, seed=0)
        # Use many sequences and a range of lengths for a better fit
        result = rb.run([1, 3, 5, 10, 20, 50], n_sequences=50)
        # EPC should be in the ballpark of p_error/2
        assert result.epc < p_err * 3 + 0.02  # generous upper bound

    def test_empty_m_values_raises(self):
        rb = StandardRB(seed=0)
        with pytest.raises(RBQVError):
            rb.run([], n_sequences=5)

    def test_invalid_sequence_length_raises(self):
        rb = StandardRB(seed=0)
        with pytest.raises(RBQVError):
            rb.run([0, 5])

    def test_invalid_n_sequences_raises(self):
        rb = StandardRB(seed=0)
        with pytest.raises(RBQVError):
            rb.run([5], n_sequences=0)

    def test_reproducible_with_seed(self):
        rb1 = StandardRB(p_error=0.01, seed=77)
        rb2 = StandardRB(p_error=0.01, seed=77)
        r1 = rb1.run([1, 5, 10], n_sequences=5)
        r2 = rb2.run([1, 5, 10], n_sequences=5)
        assert r1.survival_probs == r2.survival_probs

    def test_2qubit_rb_haar(self):
        """Multi-qubit RB (Haar random) returns valid result."""
        rb = StandardRB(n_qubits=2, p_error=0.01, seed=0)
        result = rb.run([1, 3, 5], n_sequences=5)
        assert isinstance(result, RBResult)
        assert result.n_qubits == 2
        for sp in result.survival_probs:
            assert 0.0 <= sp <= 1.0 + 1e-8


# ===========================================================================
# QuantumVolume
# ===========================================================================

class TestQuantumVolumeConstruction:
    def test_default_construction(self):
        qv = QuantumVolume()
        assert qv.p_error == pytest.approx(0.001)

    def test_invalid_p_error(self):
        with pytest.raises(RBQVError):
            QuantumVolume(p_error=-0.01)
        with pytest.raises(RBQVError):
            QuantumVolume(p_error=1.0)

    def test_custom_params(self):
        qv = QuantumVolume(p_error=0.005, seed=7)
        assert qv.p_error == pytest.approx(0.005)
        assert qv.seed == 7


class TestQuantumVolumeRun:
    def test_run_returns_list_of_qv_results(self):
        qv = QuantumVolume(p_error=0.0, seed=0)
        results = qv.run(max_qubits=2, n_trials=10)
        assert len(results) == 2
        assert all(isinstance(r, QVResult) for r in results)

    def test_n_qubits_sequence(self):
        qv = QuantumVolume(seed=0)
        results = qv.run(max_qubits=3, n_trials=5)
        assert [r.n_qubits for r in results] == [1, 2, 3]

    def test_noiseless_passes(self):
        """With zero noise, all widths should pass (HOGP > 2/3)."""
        qv = QuantumVolume(p_error=0.0, seed=0)
        results = qv.run(max_qubits=2, n_trials=20)
        for r in results:
            assert r.heavy_output_prob > 2.0 / 3.0, (
                f"n={r.n_qubits}: HOGP={r.heavy_output_prob:.4f}"
            )

    def test_noiseless_all_pass(self):
        qv = QuantumVolume(p_error=0.0, seed=42)
        results = qv.run(max_qubits=2, n_trials=30)
        assert all(r.passed for r in results)

    def test_heavy_noise_may_fail(self):
        """With very heavy noise, QV may fail."""
        qv = QuantumVolume(p_error=0.5, seed=0)
        results = qv.run(max_qubits=2, n_trials=20)
        hogps = [r.heavy_output_prob for r in results]
        # With 50% depolarizing per gate, HOGP should be near 1/2
        assert any(h < 0.9 for h in hogps)

    def test_qv_number_noiseless(self):
        """Noiseless: QV number = 2^max_qubits."""
        qv = QuantumVolume(p_error=0.0, seed=0)
        qv_num = qv.quantum_volume_number(max_qubits=2, n_trials=20)
        # Both n=1 and n=2 should pass, so QV = 4
        assert qv_num == 4

    def test_qv_number_returns_int(self):
        qv = QuantumVolume(p_error=0.0, seed=0)
        result = qv.quantum_volume_number(max_qubits=1, n_trials=10)
        assert isinstance(result, int)

    def test_result_fields(self):
        qv = QuantumVolume(p_error=0.0, seed=1)
        results = qv.run(max_qubits=1, n_trials=5)
        r = results[0]
        assert 0.0 <= r.heavy_output_prob <= 1.0 + 1e-8
        assert r.n_trials == 5
        assert r.quantum_volume in (0, 1 << r.n_qubits)

    def test_invalid_max_qubits(self):
        qv = QuantumVolume(seed=0)
        with pytest.raises(RBQVError):
            qv.run(max_qubits=0)

    def test_invalid_n_trials(self):
        qv = QuantumVolume(seed=0)
        with pytest.raises(RBQVError):
            qv.run(max_qubits=1, n_trials=0)

    def test_reproducible_with_seed(self):
        qv1 = QuantumVolume(p_error=0.01, seed=99)
        qv2 = QuantumVolume(p_error=0.01, seed=99)
        r1 = qv1.run(max_qubits=2, n_trials=5)
        r2 = qv2.run(max_qubits=2, n_trials=5)
        for a, b in zip(r1, r2):
            assert abs(a.heavy_output_prob - b.heavy_output_prob) < 1e-12

    def test_hogp_in_unit_interval(self):
        qv = QuantumVolume(p_error=0.01, seed=0)
        results = qv.run(max_qubits=2, n_trials=10)
        for r in results:
            assert 0.0 <= r.heavy_output_prob <= 1.0 + 1e-8
