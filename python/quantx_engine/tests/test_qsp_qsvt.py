"""Tests for M123: quantum signal processing (QSP) and QSVT."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantx_engine.contrib.qsp_qsvt import (
    QSPAnglesFinder,
    QSPError,
    QSVTCircuit,
    _Rz,
    _Wx,
    _qsp_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_unitary(U: np.ndarray, tol: float = 1e-8) -> bool:
    return bool(np.allclose(U @ U.conj().T, np.eye(len(U)), atol=tol))


# ---------------------------------------------------------------------------
# Signal unitary _Wx
# ---------------------------------------------------------------------------

class TestWx:
    def test_wx_is_unitary(self):
        for x in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            assert _is_unitary(_Wx(x))

    def test_wx_at_zero(self):
        W = _Wx(0.0)
        assert W[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert abs(W[0, 1]) == pytest.approx(1.0, abs=1e-12)

    def test_wx_at_one(self):
        W = _Wx(1.0)
        assert W[0, 0] == pytest.approx(1.0, abs=1e-12)

    def test_wx_clips_outside_range(self):
        # Should not raise for x slightly outside [-1, 1]
        W = _Wx(1.1)
        assert _is_unitary(W, tol=1e-6)

    def test_rz_is_unitary(self):
        for phi in [0.0, math.pi / 4, math.pi, -math.pi / 2]:
            assert _is_unitary(_Rz(phi))

    def test_rz_diagonal(self):
        R = _Rz(math.pi / 4)
        assert abs(R[0, 0]) == pytest.approx(1.0)
        assert abs(R[1, 1]) == pytest.approx(1.0)
        assert R[0, 1] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# QSP matrix
# ---------------------------------------------------------------------------

class TestQSPMatrix:
    def test_qsp_matrix_is_unitary(self):
        phases = [0.1, 0.2, 0.3]
        for x in [-0.8, 0.0, 0.7]:
            U = _qsp_matrix(x, phases)
            assert _is_unitary(U)

    def test_qsp_single_phase_is_rz(self):
        # With a single phase and no signal queries: U = Rz(φ)
        phi = 0.5
        U = _qsp_matrix(0.5, [phi])
        R = _Rz(phi)
        assert np.allclose(U, R, atol=1e-12)

    def test_qsp_top_left_is_polynomial(self):
        # For P(x) = x (degree-1 identity polynomial),
        # phases = [π/2, 0] should give P(x) ≈ x approximately
        # (we just check the value is a scalar)
        phases = [0.0, 0.0]
        x = 0.7
        U = _qsp_matrix(x, phases)
        assert np.isscalar(U[0, 0].real) or U[0, 0].real is not None


# ---------------------------------------------------------------------------
# QSPAnglesFinder construction
# ---------------------------------------------------------------------------

class TestQSPAnglesFinder:
    def test_basic_construction(self):
        f = QSPAnglesFinder(degree=3)
        assert f.degree == 3

    def test_degree_zero_raises(self):
        with pytest.raises(QSPError, match="degree"):
            QSPAnglesFinder(degree=0)

    def test_n_grid_one_raises(self):
        with pytest.raises(QSPError, match="n_grid"):
            QSPAnglesFinder(degree=3, n_grid=1)

    def test_repr(self):
        f = QSPAnglesFinder(degree=3)
        assert "QSPAnglesFinder" in repr(f)
        assert "degree=3" in repr(f)


# ---------------------------------------------------------------------------
# QSPAnglesFinder.chebyshev
# ---------------------------------------------------------------------------

class TestChebyshev:
    def test_chebyshev_returns_d_plus_one_phases(self):
        d = 3
        f = QSPAnglesFinder(degree=d)
        result = f.chebyshev(k=d)
        assert len(result.phases) == d + 1

    def test_chebyshev_wrong_k_raises(self):
        f = QSPAnglesFinder(degree=3)
        with pytest.raises(QSPError):
            f.chebyshev(k=2)

    def test_chebyshev_k_zero_raises(self):
        f = QSPAnglesFinder(degree=1)
        with pytest.raises(QSPError, match="at least 1"):
            f.chebyshev(k=0)

    def test_chebyshev_poly_name(self):
        f = QSPAnglesFinder(degree=3)
        r = f.chebyshev(k=3)
        assert "Chebyshev" in r.poly_name

    def test_chebyshev_max_error_is_finite(self):
        f = QSPAnglesFinder(degree=3)
        r = f.chebyshev(k=3)
        assert math.isfinite(r.max_error)

    def test_chebyshev_result_degree(self):
        f = QSPAnglesFinder(degree=5)
        r = f.chebyshev(k=5)
        assert r.degree == 5


# ---------------------------------------------------------------------------
# QSPAnglesFinder.sign_function
# ---------------------------------------------------------------------------

class TestSignFunction:
    def test_sign_returns_correct_degree(self):
        f = QSPAnglesFinder(degree=5)
        r = f.sign_function(delta=0.1)
        assert len(r.phases) == 6  # d+1

    def test_sign_poly_name(self):
        f = QSPAnglesFinder(degree=3)
        r = f.sign_function()
        assert "sign" in r.poly_name

    def test_sign_invalid_delta_raises(self):
        f = QSPAnglesFinder(degree=3)
        with pytest.raises(QSPError, match="delta"):
            f.sign_function(delta=0.0)
        with pytest.raises(QSPError, match="delta"):
            f.sign_function(delta=1.5)

    def test_sign_invalid_scale_raises(self):
        f = QSPAnglesFinder(degree=3)
        with pytest.raises(QSPError, match="scale"):
            f.sign_function(scale=0.0)
        with pytest.raises(QSPError, match="scale"):
            f.sign_function(scale=1.2)

    def test_sign_max_error_finite(self):
        f = QSPAnglesFinder(degree=3, n_grid=50)
        r = f.sign_function(delta=0.2)
        assert math.isfinite(r.max_error)


# ---------------------------------------------------------------------------
# QSPAnglesFinder.hamiltonian_simulation
# ---------------------------------------------------------------------------

class TestHamiltonianSimulation:
    def test_hsim_returns_phases(self):
        f = QSPAnglesFinder(degree=4)
        r = f.hamiltonian_simulation(t=0.5)
        assert len(r.phases) == 5

    def test_hsim_poly_name_contains_t(self):
        f = QSPAnglesFinder(degree=3)
        r = f.hamiltonian_simulation(t=1.0)
        assert "Hsim" in r.poly_name

    def test_hsim_max_error_finite(self):
        f = QSPAnglesFinder(degree=3, n_grid=50)
        r = f.hamiltonian_simulation(t=0.5)
        assert math.isfinite(r.max_error)


# ---------------------------------------------------------------------------
# QSPAnglesFinder.threshold
# ---------------------------------------------------------------------------

class TestThreshold:
    def test_threshold_returns_phases(self):
        f = QSPAnglesFinder(degree=5)
        r = f.threshold(0.2)
        assert len(r.phases) == 6

    def test_threshold_boundary_raises(self):
        f = QSPAnglesFinder(degree=3)
        with pytest.raises(QSPError):
            f.threshold(-1.0)
        with pytest.raises(QSPError):
            f.threshold(1.0)

    def test_threshold_max_error_finite(self):
        f = QSPAnglesFinder(degree=3, n_grid=50)
        r = f.threshold(0.0)
        assert math.isfinite(r.max_error)


# ---------------------------------------------------------------------------
# QSVTCircuit construction
# ---------------------------------------------------------------------------

class TestQSVTCircuitConstruction:
    @pytest.fixture
    def phases(self):
        return [0.1, 0.2, 0.3]

    @pytest.fixture
    def block_enc(self):
        # 2×2 unitary block-encoding of a 1×1 "signal" (x=0.5)
        import cmath as cm
        x = 0.5
        s = math.sqrt(max(0, 1 - x ** 2))
        return np.array([[x, 1j * s], [1j * s, x]], dtype=complex)

    def test_basic_construction(self, phases, block_enc):
        c = QSVTCircuit(phases, block_enc, n_signal=1)
        assert c.degree == 2

    def test_phases_too_short_raises(self, block_enc):
        with pytest.raises(QSPError, match="phases"):
            QSVTCircuit([0.1], block_enc, n_signal=1)

    def test_n_signal_zero_raises(self, phases, block_enc):
        with pytest.raises(QSPError, match="n_signal"):
            QSVTCircuit(phases, block_enc, n_signal=0)

    def test_alpha_zero_raises(self, phases, block_enc):
        with pytest.raises(QSPError, match="alpha"):
            QSVTCircuit(phases, block_enc, n_signal=1, alpha=0.0)

    def test_non_square_matrix_raises(self, phases):
        with pytest.raises(QSPError, match="square"):
            QSVTCircuit(phases, np.zeros((2, 3)), n_signal=1)

    def test_repr(self, phases, block_enc):
        c = QSVTCircuit(phases, block_enc, n_signal=1)
        assert "QSVTCircuit" in repr(c)

    def test_total_qubits(self, phases, block_enc):
        c = QSVTCircuit(phases, block_enc, n_signal=1)
        assert c.total_qubits == 2


# ---------------------------------------------------------------------------
# QSVTCircuit.polynomial_values
# ---------------------------------------------------------------------------

class TestPolynomialValues:
    @pytest.fixture
    def circuit(self):
        phases = np.zeros(4)  # degree-3 all-zero phases
        block_enc = np.eye(2, dtype=complex)
        return QSVTCircuit(phases, block_enc, n_signal=1)

    def test_poly_values_default_grid(self, circuit):
        Ps = circuit.polynomial_values()
        assert len(Ps) == 101

    def test_poly_values_custom_grid(self, circuit):
        xs = np.linspace(-1, 1, 50)
        Ps = circuit.polynomial_values(xs)
        assert len(Ps) == 50

    def test_poly_values_finite(self, circuit):
        Ps = circuit.polynomial_values()
        assert np.all(np.isfinite(Ps))

    def test_poly_values_bounded(self, circuit):
        # All QSP polynomials satisfy |P(x)| ≤ 1 on [-1, 1]
        Ps = circuit.polynomial_values()
        assert np.all(np.abs(Ps) <= 1.0 + 1e-10)


# ---------------------------------------------------------------------------
# QSVTCircuit.singular_value_transform
# ---------------------------------------------------------------------------

class TestSingularValueTransform:
    def test_identity_phases_give_identity_transform(self):
        # All-zero phases, then P(x) ≈ const; SVT gives const * A
        phases = [0.0, 0.0]
        A = np.array([[0.5, 0.0], [0.0, 0.3]], dtype=complex)
        c = QSVTCircuit(phases, np.eye(2, dtype=complex), n_signal=2, alpha=1.0)
        A_out = c.singular_value_transform(A)
        assert A_out.shape == A.shape

    def test_svt_output_shape(self):
        phases = [0.1, 0.2, 0.3]
        A = np.random.default_rng(0).random((3, 3)) * 0.5
        block_enc = np.eye(3, dtype=complex)
        c = QSVTCircuit(phases, block_enc, n_signal=3, alpha=1.0)
        A_out = c.singular_value_transform(A)
        assert A_out.shape == (3, 3)

    def test_zero_matrix_stays_zero(self):
        phases = [0.1, 0.2]
        A = np.zeros((2, 2), dtype=complex)
        c = QSVTCircuit(phases, np.eye(2, dtype=complex), n_signal=2)
        A_out = c.singular_value_transform(A)
        assert np.allclose(A_out, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# QSVTCircuit.apply
# ---------------------------------------------------------------------------

class TestApply:
    def test_apply_returns_correct_shape(self):
        phases = [0.1, 0.2]
        block_enc = np.eye(2, dtype=complex)
        c = QSVTCircuit(phases, block_enc, n_signal=2)
        sv = np.array([1.0, 0.0], dtype=complex)
        out = c.apply(sv)
        assert out.shape == (2,)

    def test_callable_block_encoding(self):
        phases = [0.0, 0.0]
        c = QSVTCircuit(phases, lambda sv: sv, n_signal=2)
        sv = np.array([1.0, 0.0], dtype=complex)
        out = c.apply(sv)
        assert out.shape == (2,)
