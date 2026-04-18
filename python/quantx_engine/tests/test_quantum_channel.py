"""Tests for QuantumChannel and SteadyStateSolver (M75).

All tests require numpy; skipped otherwise.
"""
from __future__ import annotations

import math

import pytest

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

pytestmark = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy required")

from quantx_engine.quantum_channel import (
    ChannelError,
    QuantumChannel,
    SteadyStateSolver,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _is_density_matrix(rho: np.ndarray, tol: float = 1e-8) -> bool:
    """Check that rho is a valid density matrix: Hermitian, PSD, trace 1."""
    if not np.allclose(rho, rho.conj().T, atol=tol):
        return False
    ev = np.linalg.eigvalsh(rho)
    if np.any(ev < -tol):
        return False
    if abs(np.real(np.trace(rho)) - 1.0) > tol:
        return False
    return True


def _rho_plus() -> np.ndarray:
    """Density matrix for |+⟩ = (|0⟩+|1⟩)/√2."""
    return np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)


def _rho_1() -> np.ndarray:
    """Density matrix for |1⟩."""
    return np.array([[0, 0], [0, 1]], dtype=complex)


def _rho_0() -> np.ndarray:
    """Density matrix for |0⟩."""
    return np.array([[1, 0], [0, 0]], dtype=complex)


def _rho_mixed() -> np.ndarray:
    """Maximally mixed state I/2."""
    return np.eye(2, dtype=complex) / 2


# ===========================================================================
# QuantumChannel — construction
# ===========================================================================

class TestConstruction:
    def test_from_single_kraus(self):
        ch = QuantumChannel([np.eye(2, dtype=complex)])
        assert ch.dim == 2
        assert ch.n_kraus == 1

    def test_from_multiple_kraus(self):
        K0 = np.array([[1, 0], [0, math.sqrt(0.7)]], dtype=complex)
        K1 = np.array([[0, math.sqrt(0.3)], [0, 0]], dtype=complex)
        ch = QuantumChannel([K0, K1])
        assert ch.n_kraus == 2

    def test_empty_kraus_raises(self):
        with pytest.raises(ChannelError, match="at least one"):
            QuantumChannel([])

    def test_non_square_kraus_raises(self):
        with pytest.raises(ChannelError, match="not square"):
            QuantumChannel([np.ones((2, 3), dtype=complex)])

    def test_inconsistent_shape_raises(self):
        K0 = np.eye(2, dtype=complex)
        K1 = np.eye(3, dtype=complex)
        with pytest.raises(ChannelError):
            QuantumChannel([K0, K1])

    def test_from_kraus_classmethod_list(self):
        ops = [np.eye(2, dtype=complex)]
        ch = QuantumChannel.from_kraus(ops)
        assert ch.dim == 2

    def test_from_kraus_classmethod_array(self):
        """from_kraus also accepts a (n_kraus, d, d) array."""
        arr = np.stack([np.eye(2, dtype=complex), np.zeros((2, 2), dtype=complex)])
        ch = QuantumChannel.from_kraus(arr)
        assert ch.n_kraus == 2


# ===========================================================================
# QuantumChannel — factory channels
# ===========================================================================

class TestFactoryChannels:
    # ── Identity ─────────────────────────────────────────────────────────

    def test_identity_preserves_state(self):
        ch = QuantumChannel.identity(2)
        rho = _rho_plus()
        np.testing.assert_allclose(ch.apply(rho), rho, atol=1e-12)

    def test_identity_is_unital(self):
        assert QuantumChannel.identity(2).is_unital()

    def test_identity_is_trace_preserving(self):
        assert QuantumChannel.identity(2).is_trace_preserving()

    # ── Amplitude damping ─────────────────────────────────────────────────

    def test_amplitude_damping_gamma0_identity(self):
        """γ = 0 → identity channel."""
        ch = QuantumChannel.amplitude_damping(0.0)
        rho = _rho_plus()
        np.testing.assert_allclose(ch.apply(rho), rho, atol=1e-12)

    def test_amplitude_damping_gamma1_maps_to_ground(self):
        """γ = 1 → any state maps to |0⟩⟨0|."""
        ch = QuantumChannel.amplitude_damping(1.0)
        for rho in [_rho_0(), _rho_1(), _rho_plus(), _rho_mixed()]:
            result = ch.apply(rho)
            np.testing.assert_allclose(result, _rho_0(), atol=1e-12)

    def test_amplitude_damping_partial_decay(self):
        """γ = 0.3: excited population decreases."""
        ch = QuantumChannel.amplitude_damping(0.3)
        rho = _rho_1()
        result = ch.apply(rho)
        assert result[1, 1].real == pytest.approx(0.7, abs=1e-12)
        assert result[0, 0].real == pytest.approx(0.3, abs=1e-12)

    def test_amplitude_damping_invalid_gamma_raises(self):
        with pytest.raises(ChannelError, match="gamma"):
            QuantumChannel.amplitude_damping(1.5)

    def test_amplitude_damping_not_unital(self):
        """Amplitude damping is not unital (γ > 0)."""
        assert not QuantumChannel.amplitude_damping(0.5).is_unital()

    # ── Depolarizing ──────────────────────────────────────────────────────

    def test_depolarizing_p0_identity(self):
        ch = QuantumChannel.depolarizing(2, 0.0)
        rho = _rho_plus()
        np.testing.assert_allclose(ch.apply(rho), rho, atol=1e-12)

    def test_depolarizing_p1_maps_to_mixed(self):
        """p = 1: any state maps to I/2."""
        ch = QuantumChannel.depolarizing(2, 1.0)
        for rho in [_rho_0(), _rho_1(), _rho_plus()]:
            result = ch.apply(rho)
            np.testing.assert_allclose(result, _rho_mixed(), atol=1e-12)

    def test_depolarizing_partial(self):
        """p = 0.5: state closer to I/2."""
        ch = QuantumChannel.depolarizing(2, 0.5)
        rho = _rho_0()
        result = ch.apply(rho)
        # Should be 0.5*(1-0.5)*[[1,0],[0,0]] + 0.5*I/2 = 0.5*[[1,0],[0,0]] + 0.25*I
        # = [[0.75, 0], [0, 0.25]]
        np.testing.assert_allclose(result, [[0.75, 0], [0, 0.25]], atol=1e-12)

    def test_depolarizing_is_unital(self):
        """Depolarizing channel is unital."""
        assert QuantumChannel.depolarizing(2, 0.5).is_unital()

    def test_depolarizing_d3(self):
        """d = 3 depolarizing channel is a valid CPTP map."""
        ch = QuantumChannel.depolarizing(3, 0.5)
        assert ch.is_trace_preserving()
        assert ch.is_completely_positive()

    def test_depolarizing_invalid_p_raises(self):
        with pytest.raises(ChannelError, match="p must be in"):
            QuantumChannel.depolarizing(2, -0.1)

    # ── Dephasing ─────────────────────────────────────────────────────────

    def test_dephasing_p0_identity(self):
        ch = QuantumChannel.dephasing(0.0)
        rho = _rho_plus()
        np.testing.assert_allclose(ch.apply(rho), rho, atol=1e-12)

    def test_dephasing_p1_kills_coherences(self):
        """p = 1: off-diagonals → 0."""
        ch = QuantumChannel.dephasing(1.0)
        rho = _rho_plus()  # [[0.5, 0.5],[0.5, 0.5]]
        result = ch.apply(rho)
        np.testing.assert_allclose(result, [[0.5, 0], [0, 0.5]], atol=1e-12)

    def test_dephasing_partial(self):
        """p = 0.5: off-diagonals halved."""
        ch = QuantumChannel.dephasing(0.5)
        rho = _rho_plus()
        result = ch.apply(rho)
        np.testing.assert_allclose(result[0, 1].real, 0.25, atol=1e-12)
        np.testing.assert_allclose(result[1, 1].real, 0.5, atol=1e-12)

    def test_dephasing_d3(self):
        """d = 3 dephasing is valid CPTP."""
        ch = QuantumChannel.dephasing(0.4, d=3)
        assert ch.is_trace_preserving()
        rho3 = np.ones((3, 3), dtype=complex) / 3
        result = ch.apply(rho3)
        # Diagonal should be preserved, off-diagonals scaled by (1-p)=0.6
        np.testing.assert_allclose(np.diag(result).real, [1/3, 1/3, 1/3], atol=1e-12)
        np.testing.assert_allclose(result[0, 1].real, 0.6 / 3, atol=1e-12)

    # ── Bit-flip and phase-flip ───────────────────────────────────────────

    def test_bit_flip_p0_identity(self):
        ch = QuantumChannel.bit_flip(0.0)
        rho = _rho_0()
        np.testing.assert_allclose(ch.apply(rho), rho, atol=1e-12)

    def test_bit_flip_p1_flips_state(self):
        """p = 1: |0⟩ → |1⟩ and vice versa."""
        ch = QuantumChannel.bit_flip(1.0)
        np.testing.assert_allclose(ch.apply(_rho_0()), _rho_1(), atol=1e-12)
        np.testing.assert_allclose(ch.apply(_rho_1()), _rho_0(), atol=1e-12)

    def test_phase_flip_p1_flips_phase(self):
        """p = 1: |+⟩⟨+| off-diagonals flip sign → |−⟩⟨−|."""
        ch = QuantumChannel.phase_flip(1.0)
        rho = _rho_plus()
        result = ch.apply(rho)
        np.testing.assert_allclose(result[0, 1].real, -0.5, atol=1e-12)

    # ── Reset ─────────────────────────────────────────────────────────────

    def test_reset_maps_everything_to_ground(self):
        """Reset channel → |0⟩⟨0| regardless of input."""
        ch = QuantumChannel.reset()
        for rho in [_rho_0(), _rho_1(), _rho_plus(), _rho_mixed()]:
            result = ch.apply(rho)
            np.testing.assert_allclose(result, _rho_0(), atol=1e-12)

    def test_reset_is_trace_preserving(self):
        assert QuantumChannel.reset().is_trace_preserving()


# ===========================================================================
# QuantumChannel — CPTP properties
# ===========================================================================

class TestCPTPProperties:
    @pytest.mark.parametrize("ch_fn", [
        lambda: QuantumChannel.identity(2),
        lambda: QuantumChannel.amplitude_damping(0.3),
        lambda: QuantumChannel.depolarizing(2, 0.5),
        lambda: QuantumChannel.dephasing(0.4),
        lambda: QuantumChannel.bit_flip(0.3),
        lambda: QuantumChannel.phase_flip(0.3),
        lambda: QuantumChannel.reset(),
    ])
    def test_is_trace_preserving(self, ch_fn):
        assert ch_fn().is_trace_preserving()

    @pytest.mark.parametrize("ch_fn", [
        lambda: QuantumChannel.identity(2),
        lambda: QuantumChannel.amplitude_damping(0.3),
        lambda: QuantumChannel.depolarizing(2, 0.5),
        lambda: QuantumChannel.dephasing(0.4),
        lambda: QuantumChannel.bit_flip(0.3),
        lambda: QuantumChannel.reset(),
    ])
    def test_is_completely_positive(self, ch_fn):
        assert ch_fn().is_completely_positive()

    def test_apply_preserves_trace(self):
        """Tr(Φ(ρ)) = Tr(ρ) for trace-preserving channels."""
        for gamma in [0.1, 0.5, 0.9]:
            ch = QuantumChannel.amplitude_damping(gamma)
            rho = _rho_plus()
            result = ch.apply(rho)
            assert abs(np.real(np.trace(result)) - 1.0) < 1e-12

    def test_apply_preserves_hermiticity(self):
        """Φ(ρ) is Hermitian if ρ is Hermitian."""
        ch = QuantumChannel.depolarizing(2, 0.4)
        rho = _rho_plus()
        result = ch.apply(rho)
        np.testing.assert_allclose(result, result.conj().T, atol=1e-12)

    def test_apply_output_is_density_matrix(self):
        """Φ(ρ) is a valid density matrix for various inputs."""
        channels = [
            QuantumChannel.amplitude_damping(0.3),
            QuantumChannel.depolarizing(2, 0.5),
            QuantumChannel.dephasing(0.6),
            QuantumChannel.bit_flip(0.2),
        ]
        inputs = [_rho_0(), _rho_1(), _rho_plus(), _rho_mixed()]
        for ch in channels:
            for rho in inputs:
                result = ch.apply(rho)
                assert _is_density_matrix(result), f"Not a density matrix: {result}"


# ===========================================================================
# QuantumChannel — superoperator and Choi matrix
# ===========================================================================

class TestSuperoperatorChoi:
    def test_superoperator_agrees_with_apply(self):
        """vec(Φ(ρ)) == L @ vec(ρ) for row-major vectorisation."""
        ch = QuantumChannel.amplitude_damping(0.4)
        L = ch.superoperator()
        rho = _rho_plus()
        result_apply = ch.apply(rho)
        result_L = (L @ rho.ravel()).reshape(2, 2)
        np.testing.assert_allclose(result_apply, result_L, atol=1e-12)

    def test_superoperator_shape(self):
        ch = QuantumChannel.depolarizing(3, 0.3)
        L = ch.superoperator()
        assert L.shape == (9, 9)

    def test_choi_matrix_is_psd(self):
        """Choi matrix of a CP channel is positive semi-definite."""
        for ch in [
            QuantumChannel.amplitude_damping(0.5),
            QuantumChannel.depolarizing(2, 0.3),
            QuantumChannel.dephasing(0.4),
        ]:
            J = ch.choi_matrix()
            evals = np.linalg.eigvalsh(J)
            assert np.all(evals >= -1e-10), f"Not PSD: min eval = {evals.min()}"

    def test_choi_matrix_trace_equals_d(self):
        """Tr(J) = d for trace-preserving channels."""
        for d, ch in [
            (2, QuantumChannel.amplitude_damping(0.5)),
            (2, QuantumChannel.depolarizing(2, 0.3)),
            (3, QuantumChannel.depolarizing(3, 0.2)),
        ]:
            J = ch.choi_matrix()
            assert np.real(np.trace(J)) == pytest.approx(d, abs=1e-10)

    def test_from_choi_round_trip(self):
        """from_choi(ch.choi_matrix()) gives the same channel."""
        ch = QuantumChannel.amplitude_damping(0.3)
        J = ch.choi_matrix()
        ch2 = QuantumChannel.from_choi(J)
        # Both channels should give the same output on test states
        for rho in [_rho_0(), _rho_1(), _rho_plus()]:
            np.testing.assert_allclose(ch.apply(rho), ch2.apply(rho), atol=1e-10)

    def test_from_choi_identity(self):
        """from_choi on the Choi matrix of identity gives back identity."""
        ch = QuantumChannel.identity(2)
        J = ch.choi_matrix()
        ch2 = QuantumChannel.from_choi(J)
        rho = _rho_plus()
        np.testing.assert_allclose(ch2.apply(rho), rho, atol=1e-10)

    def test_from_choi_non_square_raises(self):
        with pytest.raises(ChannelError, match="square"):
            QuantumChannel.from_choi(np.ones((4, 3), dtype=complex))

    def test_from_choi_non_perfect_square_raises(self):
        with pytest.raises(ChannelError, match="perfect square"):
            QuantumChannel.from_choi(np.eye(6, dtype=complex))


# ===========================================================================
# QuantumChannel — compose and tensor
# ===========================================================================

class TestComposeAndTensor:
    def test_compose_two_identity(self):
        """identity ∘ identity = identity."""
        ch = QuantumChannel.identity(2)
        composed = ch.compose(ch)
        rho = _rho_plus()
        np.testing.assert_allclose(composed.apply(rho), rho, atol=1e-12)

    def test_compose_amplitude_damping_twice(self):
        """Applying amplitude damping twice: γ_eff = 1 - (1-γ)²."""
        gamma = 0.3
        ch = QuantumChannel.amplitude_damping(gamma)
        composed = ch.compose(ch)
        # Apply twice to |1⟩: population of |1⟩ → (1-γ)² = 0.49
        rho = _rho_1()
        result = composed.apply(rho)
        assert result[1, 1].real == pytest.approx((1 - gamma) ** 2, abs=1e-10)

    def test_compose_dimension_mismatch_raises(self):
        ch2 = QuantumChannel.identity(2)
        ch3 = QuantumChannel.identity(3)
        with pytest.raises(ChannelError, match="dimensions"):
            ch2.compose(ch3)

    def test_tensor_product_product_state(self):
        """(Φ_A ⊗ Φ_B)(ρ_A ⊗ ρ_B) = Φ_A(ρ_A) ⊗ Φ_B(ρ_B)."""
        chA = QuantumChannel.amplitude_damping(0.4)
        chB = QuantumChannel.dephasing(0.3)
        ch_AB = chA.tensor(chB)

        rhoA = _rho_plus()
        rhoB = _rho_0()
        rho_AB = np.kron(rhoA, rhoB)

        result_direct = ch_AB.apply(rho_AB)
        expected = np.kron(chA.apply(rhoA), chB.apply(rhoB))
        np.testing.assert_allclose(result_direct, expected, atol=1e-10)

    def test_tensor_trace_preserving(self):
        """Tensor product of TP channels is TP."""
        chA = QuantumChannel.bit_flip(0.2)
        chB = QuantumChannel.phase_flip(0.3)
        ch_AB = chA.tensor(chB)
        assert ch_AB.is_trace_preserving()

    def test_compose_superoperator_is_product(self):
        """Superoperator of composed channel = L_other @ L_self."""
        chA = QuantumChannel.amplitude_damping(0.3)
        chB = QuantumChannel.dephasing(0.4)
        composed = chA.compose(chB)
        LA, LB = chA.superoperator(), chB.superoperator()
        np.testing.assert_allclose(
            composed.superoperator(), LB @ LA, atol=1e-10
        )


# ===========================================================================
# SteadyStateSolver
# ===========================================================================

class TestSteadyStateSolver:
    def test_amplitude_damping_steady_state_is_ground(self):
        """Amplitude damping (γ > 0) converges to |0⟩⟨0|."""
        ch = QuantumChannel.amplitude_damping(0.5)
        solver = SteadyStateSolver()
        rho_ss = solver.solve(ch)
        np.testing.assert_allclose(rho_ss, _rho_0(), atol=1e-8)

    def test_amplitude_damping_gamma1_steady_state(self):
        """γ = 1: steady state is exactly |0⟩⟨0|."""
        ch = QuantumChannel.amplitude_damping(1.0)
        solver = SteadyStateSolver()
        rho_ss = solver.solve(ch)
        np.testing.assert_allclose(rho_ss, _rho_0(), atol=1e-8)

    def test_depolarizing_steady_state_is_mixed(self):
        """Depolarizing (p > 0) steady state is I/2."""
        ch = QuantumChannel.depolarizing(2, 0.5)
        solver = SteadyStateSolver()
        rho_ss = solver.solve(ch)
        np.testing.assert_allclose(rho_ss, _rho_mixed(), atol=1e-8)

    def test_reset_steady_state_is_ground(self):
        """Reset channel steady state is |0⟩⟨0|."""
        ch = QuantumChannel.reset()
        solver = SteadyStateSolver()
        rho_ss = solver.solve(ch)
        np.testing.assert_allclose(rho_ss, _rho_0(), atol=1e-8)

    def test_steady_state_is_valid_density_matrix(self):
        """Steady state is a valid density matrix for various channels."""
        channels = [
            QuantumChannel.amplitude_damping(0.3),
            QuantumChannel.depolarizing(2, 0.6),
            QuantumChannel.dephasing(0.5),
            QuantumChannel.reset(),
        ]
        solver = SteadyStateSolver()
        for ch in channels:
            rho_ss = solver.solve(ch)
            assert _is_density_matrix(rho_ss), (
                f"Not a density matrix for {ch}: {rho_ss}"
            )

    def test_steady_state_is_fixed_point(self):
        """Φ(ρ_ss) ≈ ρ_ss."""
        channels = [
            QuantumChannel.amplitude_damping(0.4),
            QuantumChannel.depolarizing(2, 0.5),
            QuantumChannel.reset(),
        ]
        solver = SteadyStateSolver()
        for ch in channels:
            rho_ss = solver.solve(ch)
            rho_applied = ch.apply(rho_ss)
            np.testing.assert_allclose(
                rho_applied, rho_ss, atol=1e-6,
                err_msg=f"Steady state not a fixed point for {ch}"
            )

    def test_steady_state_d3_depolarizing(self):
        """d = 3 depolarizing: steady state is I/3."""
        ch = QuantumChannel.depolarizing(3, 0.5)
        solver = SteadyStateSolver()
        rho_ss = solver.solve(ch)
        np.testing.assert_allclose(rho_ss, np.eye(3) / 3, atol=1e-8)

    def test_invalid_tol_raises(self):
        with pytest.raises(ChannelError, match="tol must be > 0"):
            SteadyStateSolver(tol=-1.0)

    def test_dephasing_steady_state_is_diagonal(self):
        """Dephasing (p=1): fixed point is any diagonal state."""
        ch = QuantumChannel.dephasing(1.0)
        solver = SteadyStateSolver()
        rho_ss = solver.solve(ch)
        # Off-diagonals should be zero
        assert abs(rho_ss[0, 1]) < 1e-8
        assert abs(rho_ss[1, 0]) < 1e-8
        assert _is_density_matrix(rho_ss)


# ===========================================================================
# Integration: channel pipeline
# ===========================================================================

class TestIntegration:
    def test_depolarizing_then_amplitude_damping(self):
        """Composing depolarizing then amplitude damping gives valid CPTP."""
        ch = QuantumChannel.depolarizing(2, 0.3).compose(
            QuantumChannel.amplitude_damping(0.5)
        )
        assert ch.is_trace_preserving()
        assert ch.is_completely_positive()
        rho = _rho_plus()
        result = ch.apply(rho)
        assert _is_density_matrix(result)

    def test_tensor_two_qubits_then_steady_state(self):
        """Two-qubit tensor channel has correct steady state."""
        # Both qubits undergoing amplitude damping
        chA = QuantumChannel.amplitude_damping(0.5)
        chB = QuantumChannel.amplitude_damping(0.5)
        ch_AB = chA.tensor(chB)
        solver = SteadyStateSolver()
        rho_ss = solver.solve(ch_AB)
        # Steady state should be |00⟩⟨00|
        expected = np.zeros((4, 4), dtype=complex)
        expected[0, 0] = 1.0
        np.testing.assert_allclose(rho_ss, expected, atol=1e-8)

    def test_many_applications_converge_to_steady_state(self):
        """Repeated application of amplitude damping converges to |0⟩⟨0|."""
        ch = QuantumChannel.amplitude_damping(0.3)
        rho = _rho_plus()
        for _ in range(100):
            rho = ch.apply(rho)
        np.testing.assert_allclose(rho, _rho_0(), atol=1e-5)
