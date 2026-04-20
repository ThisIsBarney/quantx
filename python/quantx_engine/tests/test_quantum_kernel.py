"""Tests for quantum kernel methods and QSVM (M80)."""
from __future__ import annotations

import math
import pytest
import numpy as np

from ..quantum_kernel import (
    QKernelError,
    QuantumFeatureMap,
    ZZFeatureMap,
    FidelityQuantumKernel,
    target_alignment,
    frobenius_alignment,
    QSVMResult,
    QSVM,
    _apply_single_qubit,
    _apply_cx,
    _rz,
    _H_GATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fm(n=2, reps=1):
    return ZZFeatureMap(n_qubits=n, n_reps=reps)


def _make_kernel(n=2, reps=1):
    return FidelityQuantumKernel(_make_fm(n, reps))


def _xy_data(n_per_class=4, seed=0):
    """Simple 2-D linearly separable data (in quantum feature space)."""
    rng = np.random.default_rng(seed)
    X_pos = rng.uniform(0.65, 0.95, size=(n_per_class, 2))
    X_neg = rng.uniform(0.05, 0.35, size=(n_per_class, 2))
    X = np.vstack([X_pos, X_neg])
    y = np.array([1.0] * n_per_class + [-1.0] * n_per_class)
    return X, y


# ===========================================================================
# QKernelError
# ===========================================================================

class TestQKernelError:
    def test_is_value_error(self):
        assert issubclass(QKernelError, ValueError)

    def test_raises(self):
        with pytest.raises(QKernelError):
            raise QKernelError("test")


# ===========================================================================
# Gate helpers
# ===========================================================================

class TestGateHelpers:
    def test_rz_diagonal(self):
        theta = math.pi / 3
        R = _rz(theta)
        assert R.shape == (2, 2)
        assert abs(R[0, 1]) < 1e-12
        assert abs(R[1, 0]) < 1e-12
        assert abs(abs(R[0, 0]) - 1.0) < 1e-12
        assert abs(abs(R[1, 1]) - 1.0) < 1e-12

    def test_rz_phases(self):
        theta = math.pi / 2
        R = _rz(theta)
        assert abs(R[0, 0] - np.exp(-1j * theta / 2)) < 1e-12
        assert abs(R[1, 1] - np.exp(+1j * theta / 2)) < 1e-12

    def test_h_gate_hadamard(self):
        """H|0⟩ = |+⟩ = [1/√2, 1/√2]."""
        psi = np.array([1.0, 0.0], dtype=complex)
        h_psi = _H_GATE @ psi
        assert np.allclose(h_psi, [1 / math.sqrt(2), 1 / math.sqrt(2)])

    def test_apply_single_qubit_1qubit(self):
        """Apply X gate to a 1-qubit state: X|0⟩ = |1⟩."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        psi = np.array([1.0, 0.0], dtype=complex)
        result = _apply_single_qubit(psi, 1, 0, X)
        assert np.allclose(result, [0.0, 1.0])

    def test_apply_single_qubit_2qubits_first(self):
        """Apply X to qubit 0 (MSB) of |00⟩ → |10⟩."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)  # |00⟩
        result = _apply_single_qubit(psi, 2, 0, X)
        expected = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex)  # |10⟩
        assert np.allclose(result, expected)

    def test_apply_single_qubit_2qubits_second(self):
        """Apply X to qubit 1 (LSB) of |00⟩ → |01⟩."""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        result = _apply_single_qubit(psi, 2, 1, X)
        expected = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)  # |01⟩
        assert np.allclose(result, expected)

    def test_apply_cx_flips_target(self):
        """CX(0,1)|10⟩ = |11⟩."""
        psi = np.zeros(4, dtype=complex)
        psi[2] = 1.0  # |10⟩ = index 2
        result = _apply_cx(psi, 2, 0, 1)
        expected = np.zeros(4, dtype=complex)
        expected[3] = 1.0  # |11⟩ = index 3
        assert np.allclose(result, expected)

    def test_apply_cx_no_flip_when_ctrl_zero(self):
        """CX(0,1)|01⟩ = |01⟩ (control is 0, target unchanged)."""
        psi = np.zeros(4, dtype=complex)
        psi[1] = 1.0  # |01⟩ = index 1
        result = _apply_cx(psi, 2, 0, 1)
        assert np.allclose(result, psi)

    def test_apply_cx_is_unitary(self):
        """CX must preserve norm."""
        psi = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        result = _apply_cx(psi, 2, 0, 1)
        assert abs(np.linalg.norm(result) - 1.0) < 1e-12

    def test_h_self_inverse(self):
        """H applied twice = identity."""
        psi = np.array([0.6, 0.8], dtype=complex)
        psi_back = _apply_single_qubit(
            _apply_single_qubit(psi, 1, 0, _H_GATE), 1, 0, _H_GATE
        )
        assert np.allclose(psi_back, psi, atol=1e-12)


# ===========================================================================
# ZZFeatureMap
# ===========================================================================

class TestZZFeatureMapConstruction:
    def test_basic_construction(self):
        fm = ZZFeatureMap(n_qubits=2, n_reps=2)
        assert fm.n_qubits == 2
        assert fm.n_reps == 2
        assert fm.n_features == 2

    def test_single_qubit(self):
        fm = ZZFeatureMap(n_qubits=1, n_reps=1)
        assert fm.n_qubits == 1
        assert len(fm._pairs) == 0  # no ZZ pairs for 1 qubit

    def test_invalid_n_qubits(self):
        with pytest.raises(QKernelError):
            ZZFeatureMap(n_qubits=0)

    def test_invalid_n_reps(self):
        with pytest.raises(QKernelError):
            ZZFeatureMap(n_qubits=2, n_reps=0)

    def test_pairs_count(self):
        fm = ZZFeatureMap(n_qubits=3)
        assert len(fm._pairs) == 3  # (0,1),(0,2),(1,2)


class TestZZFeatureMapStatevector:
    def test_correct_length(self):
        fm = ZZFeatureMap(n_qubits=2, n_reps=1)
        psi = fm.statevector(np.array([0.3, 0.7]))
        assert len(psi) == 4

    def test_unit_norm(self):
        fm = ZZFeatureMap(n_qubits=2, n_reps=2)
        for x in [[0.0, 0.0], [1.0, 1.0], [0.5, 0.3], [math.pi, 1.2]]:
            psi = fm.statevector(np.array(x))
            assert abs(np.linalg.norm(psi) - 1.0) < 1e-10, f"Not unit for x={x}"

    def test_complex_dtype(self):
        psi = ZZFeatureMap(2).statevector(np.array([0.1, 0.2]))
        assert np.iscomplexobj(psi)

    def test_wrong_input_length_raises(self):
        fm = ZZFeatureMap(n_qubits=2)
        with pytest.raises(QKernelError):
            fm.statevector(np.array([0.1, 0.2, 0.3]))

    def test_different_inputs_different_states(self):
        fm = ZZFeatureMap(n_qubits=2)
        psi1 = fm.statevector(np.array([0.1, 0.2]))
        psi2 = fm.statevector(np.array([0.8, 0.9]))
        assert not np.allclose(psi1, psi2)

    def test_same_input_same_state(self):
        fm = ZZFeatureMap(n_qubits=2)
        x = np.array([0.42, 0.58])
        psi1 = fm.statevector(x)
        psi2 = fm.statevector(x)
        assert np.allclose(psi1, psi2)

    def test_n_reps_affects_state(self):
        x = np.array([0.3, 0.7])
        fm1 = ZZFeatureMap(n_qubits=2, n_reps=1)
        fm2 = ZZFeatureMap(n_qubits=2, n_reps=2)
        assert not np.allclose(fm1.statevector(x), fm2.statevector(x))

    def test_larger_system(self):
        fm = ZZFeatureMap(n_qubits=4, n_reps=1)
        psi = fm.statevector(np.array([0.1, 0.3, 0.5, 0.7]))
        assert len(psi) == 16
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-10


# ===========================================================================
# FidelityQuantumKernel
# ===========================================================================

class TestFidelityQuantumKernel:
    def setup_method(self):
        self.fm = ZZFeatureMap(n_qubits=2, n_reps=1)
        self.k = FidelityQuantumKernel(self.fm)

    def test_self_kernel_is_one(self):
        """K(x, x) = 1 for any x."""
        for x in [[0.0, 0.0], [0.5, 0.5], [1.0, 0.3]]:
            assert abs(self.k.evaluate(x, x) - 1.0) < 1e-10

    def test_kernel_in_unit_interval(self):
        """K(x, x') ∈ [0, 1]."""
        x1 = np.array([0.1, 0.2])
        x2 = np.array([0.8, 0.9])
        val = self.k.evaluate(x1, x2)
        assert 0.0 <= val <= 1.0 + 1e-10

    def test_kernel_symmetric(self):
        x1 = np.array([0.2, 0.4])
        x2 = np.array([0.6, 0.8])
        assert abs(self.k.evaluate(x1, x2) - self.k.evaluate(x2, x1)) < 1e-10

    def test_matrix_shape_self(self):
        X = np.array([[0.1, 0.2], [0.5, 0.6], [0.9, 0.8]])
        K = self.k.matrix(X)
        assert K.shape == (3, 3)

    def test_matrix_diagonal_is_one(self):
        X = np.array([[0.1, 0.2], [0.5, 0.6], [0.9, 0.8]])
        K = self.k.matrix(X)
        assert np.allclose(np.diag(K), 1.0, atol=1e-10)

    def test_matrix_symmetric(self):
        X = np.array([[0.1, 0.2], [0.5, 0.6], [0.9, 0.8]])
        K = self.k.matrix(X)
        assert np.allclose(K, K.T, atol=1e-10)

    def test_matrix_psd(self):
        """Kernel matrix must be positive semi-definite."""
        X = np.array([[0.1, 0.2], [0.5, 0.6], [0.9, 0.8]])
        K = self.k.matrix(X)
        evals = np.linalg.eigvalsh(K)
        assert np.all(evals >= -1e-10), f"Negative eigenvalue: {evals.min()}"

    def test_matrix_cross(self):
        """Cross-kernel matrix K[i,j] = evaluate(X[i], Y[j])."""
        X = np.array([[0.1, 0.2], [0.5, 0.6]])
        Y = np.array([[0.9, 0.8]])
        K = self.k.matrix(X, Y)
        assert K.shape == (2, 1)
        assert abs(K[0, 0] - self.k.evaluate(X[0], Y[0])) < 1e-10
        assert abs(K[1, 0] - self.k.evaluate(X[1], Y[0])) < 1e-10

    def test_matrix_entries_in_unit_interval(self):
        X = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]])
        K = self.k.matrix(X)
        assert np.all(K >= -1e-10)
        assert np.all(K <= 1.0 + 1e-10)


# ===========================================================================
# Kernel alignment
# ===========================================================================

class TestTargetAlignment:
    def test_perfect_alignment(self):
        """KTA = 1 when K = yy^T."""
        y = np.array([1.0, 1.0, -1.0, -1.0])
        K = np.outer(y, y).astype(float)
        ta = target_alignment(K, y)
        assert abs(ta - 1.0) < 1e-10

    def test_range_minus1_to_1(self):
        rng = np.random.default_rng(7)
        y = np.array([1.0, -1.0, 1.0, -1.0])
        K = rng.uniform(0, 1, (4, 4))
        K = (K + K.T) / 2  # symmetrise
        ta = target_alignment(K, y)
        assert -1.0 - 1e-10 <= ta <= 1.0 + 1e-10

    def test_zero_kernel_returns_zero(self):
        y = np.array([1.0, -1.0])
        K = np.zeros((2, 2))
        assert target_alignment(K, y) == 0.0

    def test_identity_kernel(self):
        """KTA for I with balanced labels = 1/sqrt(n)."""
        n = 4
        y = np.array([1.0, 1.0, -1.0, -1.0])
        K = np.eye(n)
        ta = target_alignment(K, y)
        expected = 1.0 / math.sqrt(n)
        assert abs(ta - expected) < 1e-10


class TestFrobeniusAlignment:
    def test_self_alignment_is_one(self):
        rng = np.random.default_rng(3)
        K = rng.uniform(0, 1, (4, 4))
        assert abs(frobenius_alignment(K, K) - 1.0) < 1e-10

    def test_negated_alignment_is_minus_one(self):
        rng = np.random.default_rng(3)
        K = rng.uniform(0, 1, (4, 4))
        assert abs(frobenius_alignment(K, -K) - (-1.0)) < 1e-10

    def test_range(self):
        rng = np.random.default_rng(5)
        K1 = rng.uniform(0, 1, (4, 4))
        K2 = rng.uniform(0, 1, (4, 4))
        val = frobenius_alignment(K1, K2)
        assert -1.0 - 1e-10 <= val <= 1.0 + 1e-10

    def test_zero_kernel_returns_zero(self):
        K = np.zeros((3, 3))
        K2 = np.eye(3)
        assert frobenius_alignment(K, K2) == 0.0

    def test_orthogonal_kernels_near_zero(self):
        """Kernels with disjoint support have near-zero alignment."""
        K1 = np.zeros((4, 4))
        K1[0, 1] = K1[1, 0] = 1.0
        K2 = np.zeros((4, 4))
        K2[2, 3] = K2[3, 2] = 1.0
        assert frobenius_alignment(K1, K2) == pytest.approx(0.0, abs=1e-10)


# ===========================================================================
# QSVM construction
# ===========================================================================

class TestQSVMConstruction:
    def test_from_feature_map(self):
        fm = _make_fm()
        clf = QSVM(feature_map=fm)
        assert isinstance(clf.kernel, FidelityQuantumKernel)

    def test_from_kernel(self):
        k = _make_kernel()
        clf = QSVM(kernel=k)
        assert clf.kernel is k

    def test_kernel_takes_precedence(self):
        fm = _make_fm()
        k = _make_kernel()
        clf = QSVM(feature_map=fm, kernel=k)
        assert clf.kernel is k

    def test_no_args_raises(self):
        with pytest.raises(QKernelError):
            QSVM()

    def test_invalid_C_raises(self):
        with pytest.raises(QKernelError):
            QSVM(feature_map=_make_fm(), C=0.0)
        with pytest.raises(QKernelError):
            QSVM(feature_map=_make_fm(), C=-1.0)

    def test_predict_before_fit_raises(self):
        clf = QSVM(feature_map=_make_fm())
        with pytest.raises(QKernelError):
            clf.predict(np.array([[0.1, 0.2]]))

    def test_decision_function_before_fit_raises(self):
        clf = QSVM(feature_map=_make_fm())
        with pytest.raises(QKernelError):
            clf.decision_function(np.array([[0.1, 0.2]]))


# ===========================================================================
# QSVM fitting
# ===========================================================================

class TestQSVMFit:
    def setup_method(self):
        self.fm = ZZFeatureMap(n_qubits=2, n_reps=1)
        self.clf = QSVM(feature_map=self.fm, C=1.0)
        self.X, self.y = _xy_data(n_per_class=3)

    def test_fit_returns_qsvm_result(self):
        result = self.clf.fit(self.X, self.y)
        assert isinstance(result, QSVMResult)

    def test_result_fields(self):
        result = self.clf.fit(self.X, self.y)
        n = len(self.y)
        assert len(result.alphas) == n
        assert result.n_support == len(result.support_vector_indices)
        assert 0.0 <= result.train_accuracy <= 1.0
        assert math.isfinite(result.bias)
        assert result.kernel_matrix.shape == (n, n)

    def test_kernel_matrix_symmetric(self):
        result = self.clf.fit(self.X, self.y)
        K = result.kernel_matrix
        assert np.allclose(K, K.T, atol=1e-10)

    def test_kernel_matrix_diagonal_is_one(self):
        result = self.clf.fit(self.X, self.y)
        assert np.allclose(np.diag(result.kernel_matrix), 1.0, atol=1e-10)

    def test_invalid_labels_raises(self):
        with pytest.raises(QKernelError):
            self.clf.fit(self.X, np.zeros(len(self.y)))

    def test_mismatched_shapes_raises(self):
        with pytest.raises(QKernelError):
            self.clf.fit(self.X, self.y[:2])

    def test_labels_not_in_pm1_raises(self):
        y_bad = self.y.copy()
        y_bad[0] = 2.0
        with pytest.raises(QKernelError):
            self.clf.fit(self.X, y_bad)


# ===========================================================================
# QSVM predict / score
# ===========================================================================

class TestQSVMPredict:
    def setup_method(self):
        self.fm = ZZFeatureMap(n_qubits=2, n_reps=1)
        self.clf = QSVM(feature_map=self.fm, C=1.0)
        self.X, self.y = _xy_data(n_per_class=3)
        self.clf.fit(self.X, self.y)

    def test_predict_shape(self):
        preds = self.clf.predict(self.X)
        assert preds.shape == (len(self.y),)

    def test_predict_returns_pm1(self):
        preds = self.clf.predict(self.X)
        assert set(preds).issubset({-1.0, 1.0})

    def test_score_is_float_in_01(self):
        s = self.clf.score(self.X, self.y)
        assert 0.0 <= s <= 1.0

    def test_decision_function_shape(self):
        df = self.clf.decision_function(self.X)
        assert df.shape == (len(self.y),)

    def test_decision_function_sign_matches_predict(self):
        df = self.clf.decision_function(self.X)
        preds = self.clf.predict(self.X)
        # For non-zero decision values, sign should match prediction
        for d, p in zip(df, preds):
            if abs(d) > 1e-8:
                assert np.sign(d) == p


class TestQSVMSeparable:
    """Well-separated data should achieve high training accuracy."""

    def test_separable_data_high_accuracy(self):
        """Two clearly separated clusters should train to ≥ 75% accuracy."""
        fm = ZZFeatureMap(n_qubits=2, n_reps=2)
        clf = QSVM(feature_map=fm, C=10.0)
        X, y = _xy_data(n_per_class=4, seed=1)
        result = clf.fit(X, y, max_passes=50)
        assert result.train_accuracy >= 0.75

    def test_2_sample_fit(self):
        """Minimal case: 1 sample per class."""
        fm = ZZFeatureMap(n_qubits=2, n_reps=1)
        clf = QSVM(feature_map=fm, C=1.0)
        X = np.array([[0.1, 0.2], [0.9, 0.8]])
        y = np.array([-1.0, 1.0])
        result = clf.fit(X, y)
        assert isinstance(result, QSVMResult)
        preds = clf.predict(X)
        assert set(preds).issubset({-1.0, 1.0})


# ===========================================================================
# End-to-end: QSVMResult + kernel alignment on toy data
# ===========================================================================

class TestKernelAlignmentOnData:
    def test_quantum_kernel_alignment_positive(self):
        """A quantum kernel on well-separated data should align > 0 with target."""
        k = FidelityQuantumKernel(ZZFeatureMap(n_qubits=2, n_reps=2))
        X, y = _xy_data(n_per_class=4, seed=0)
        K = k.matrix(X)
        ta = target_alignment(K, y)
        # Quantum kernel should show positive alignment for this dataset
        assert ta > 0.0

    def test_ideal_kernel_alignment(self):
        """yy^T kernel should give KTA = 1.0."""
        y = np.array([1.0, 1.0, -1.0, -1.0])
        K_ideal = np.outer(y, y).astype(float)
        assert abs(target_alignment(K_ideal, y) - 1.0) < 1e-10

    def test_frobenius_alignment_with_target(self):
        k = FidelityQuantumKernel(ZZFeatureMap(n_qubits=2, n_reps=2))
        X, y = _xy_data(n_per_class=3, seed=0)
        K = k.matrix(X)
        K_target = np.outer(y, y).astype(float)
        fa = frobenius_alignment(K, K_target)
        # Should be a valid float in [-1, 1]
        assert -1.0 - 1e-10 <= fa <= 1.0 + 1e-10


# ===========================================================================
# Public package exports
# ===========================================================================

def test_quantum_kernel_importable_from_package():
    from quantx_engine import (
        QKernelError as QKE,
        ZZFeatureMap as ZZ,
        FidelityQuantumKernel as FQK,
        target_alignment as ta,
        frobenius_alignment as fa,
        QSVMResult as QR,
        QSVM as QS,
    )
    assert QKE is QKernelError
    assert ZZ is ZZFeatureMap
    assert FQK is FidelityQuantumKernel
    assert ta is target_alignment
    assert fa is frobenius_alignment
    assert QR is QSVMResult
    assert QS is QSVM
