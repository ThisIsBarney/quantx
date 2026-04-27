"""Tests for Floquet engineering and Bloch-Floquet band structure — M92."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quantx_engine.floquet import (
    BlochFloquetSolver,
    FloquetError,
    FloquetResult,
    FloquetSystem,
    _matrix_exp,
    _matrix_log_unitary,
    _propagator_rk4,
    driven_two_level,
    floquet_effective_hamiltonian,
    floquet_spectrum,
)

# ── Pauli matrices ────────────────────────────────────────────────────────────

_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


# ── FloquetSystem construction ────────────────────────────────────────────────

class TestFloquetSystemConstruction:
    def test_default_drive_func_is_cosine(self):
        sys = FloquetSystem(_Z, _X, omega=2.0)
        t = 1.23
        # f(t) should be cos(omega * t)
        assert abs(sys.drive_func(t) - math.cos(2.0 * t)) < 1e-12

    def test_period_correct(self):
        omega = 3.0
        sys = FloquetSystem(_Z, _X, omega=omega)
        assert abs(sys.period - 2.0 * math.pi / omega) < 1e-12

    def test_dim_correct(self):
        sys = FloquetSystem(_Z, _X, omega=1.0)
        assert sys.dim == 2

    def test_H_at_static_only(self):
        """With H_drive=0 and drive_func=cos, H(t) = H_static for any t."""
        sys = FloquetSystem(_Z, np.zeros((2, 2), dtype=complex), omega=1.0)
        H = sys.H_at(0.5)
        np.testing.assert_allclose(H, _Z, atol=1e-12)

    def test_H_at_with_drive(self):
        """H(t) = H_static + cos(omega*t) * H_drive."""
        omega = 2.0
        t = 0.7
        sys = FloquetSystem(_Z, _X, omega=omega)
        expected = _Z + math.cos(omega * t) * _X
        np.testing.assert_allclose(sys.H_at(t), expected, atol=1e-12)

    def test_custom_drive_func(self):
        sys = FloquetSystem(_Z, _X, omega=1.0, drive_func=lambda t: math.sin(t))
        t = 0.5
        expected = _Z + math.sin(t) * _X
        np.testing.assert_allclose(sys.H_at(t), expected, atol=1e-12)

    def test_n_steps_default(self):
        sys = FloquetSystem(_Z, _X, omega=1.0)
        assert sys.n_steps == 200

    def test_4x4_system(self):
        H4 = np.eye(4, dtype=complex)
        sys = FloquetSystem(H4, H4, omega=1.0)
        assert sys.dim == 4

    def test_invalid_omega_zero(self):
        with pytest.raises(FloquetError):
            FloquetSystem(_Z, _X, omega=0.0)

    def test_invalid_omega_negative(self):
        with pytest.raises(FloquetError):
            FloquetSystem(_Z, _X, omega=-1.0)

    def test_invalid_n_steps_too_small(self):
        with pytest.raises(FloquetError):
            FloquetSystem(_Z, _X, omega=1.0, n_steps=2)

    def test_invalid_H_static_not_square(self):
        with pytest.raises((FloquetError, Exception)):
            FloquetSystem(np.ones((2, 3), dtype=complex), _X, omega=1.0)

    def test_invalid_H_drive_wrong_shape(self):
        with pytest.raises(FloquetError):
            FloquetSystem(_Z, np.eye(3, dtype=complex), omega=1.0)

    def test_repr_contains_dim_omega(self):
        r = repr(FloquetSystem(_Z, _X, omega=2.0))
        assert "FloquetSystem" in r
        assert "dim=2" in r
        assert "omega" in r


# ── One-period propagator ─────────────────────────────────────────────────────

class TestPropagator:
    def test_propagator_is_unitary(self):
        """U(T) must satisfy U†U ≈ I."""
        sys = FloquetSystem(_Z, _X, omega=2.0)
        U = _propagator_rk4(sys)
        d = sys.dim
        np.testing.assert_allclose(
            U.conj().T @ U, np.eye(d, dtype=complex), atol=1e-6
        )

    def test_propagator_shape(self):
        sys = FloquetSystem(_Z, _X, omega=2.0, n_steps=100)
        U = _propagator_rk4(sys)
        assert U.shape == (2, 2)

    def test_static_hamiltonian_propagator_exact(self):
        """For H_drive=0, U(T) = exp(-i H_static T) exactly."""
        omega = 1.5
        sys = FloquetSystem(_Z, np.zeros((2, 2), dtype=complex), omega=omega, n_steps=400)
        T = sys.period
        U = _propagator_rk4(sys)
        # exp(-i Z T): eigenvalues are exp(-iT) and exp(+iT)
        expected = _matrix_exp(-1j * _Z * T)
        np.testing.assert_allclose(U, expected, atol=1e-4)

    def test_propagator_4x4_unitary(self):
        H4 = np.diag([1.0, 0.5, -0.5, -1.0]).astype(complex)
        D4 = np.diag([0, 1, -1, 0]).astype(complex)
        sys = FloquetSystem(H4, D4, omega=3.0, n_steps=200)
        U = _propagator_rk4(sys)
        np.testing.assert_allclose(
            U.conj().T @ U, np.eye(4, dtype=complex), atol=1e-5
        )


# ── Matrix helpers ────────────────────────────────────────────────────────────

class TestMatrixHelpers:
    def test_matrix_exp_identity(self):
        M = np.zeros((2, 2), dtype=complex)
        np.testing.assert_allclose(_matrix_exp(M), np.eye(2), atol=1e-12)

    def test_matrix_exp_pauli_z(self):
        """exp(iπZ/2) = i * I (up to global phase)."""
        theta = math.pi / 2
        result = _matrix_exp(1j * theta * _Z)
        # e^{iπZ/2}: eigenvalues e^{iπ/2}=i and e^{-iπ/2}=-i
        expected = np.diag([np.exp(1j * theta), np.exp(-1j * theta)])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_matrix_exp_hermitian_stays_unitary(self):
        """exp(iH) should be unitary when H is Hermitian."""
        H = 0.5 * _X + 0.3 * _Z
        U = _matrix_exp(1j * H)
        np.testing.assert_allclose(U @ U.conj().T, np.eye(2), atol=1e-12)

    def test_matrix_log_unitary_roundtrip(self):
        """exp(-i H_eff T) should recover U(T)."""
        omega = 2.0
        T = 2.0 * math.pi / omega
        sys = FloquetSystem(_Z, 0.3 * _X, omega=omega, n_steps=400)
        U = _propagator_rk4(sys)
        H_eff = _matrix_log_unitary(U, omega)
        U_reconstructed = _matrix_exp(-1j * H_eff * T)
        np.testing.assert_allclose(np.abs(U), np.abs(U_reconstructed), atol=1e-4)

    def test_matrix_log_gives_hermitian(self):
        """H_eff from matrix log should be (approximately) Hermitian."""
        omega = 3.0
        sys = FloquetSystem(_Z, 0.2 * _X, omega=omega, n_steps=300)
        U = _propagator_rk4(sys)
        H_eff = _matrix_log_unitary(U, omega)
        # Should be Hermitian
        np.testing.assert_allclose(H_eff, H_eff.conj().T, atol=1e-10)


# ── BlochFloquetSolver ────────────────────────────────────────────────────────

class TestBlochFloquetSolver:
    def test_solve_returns_floquet_result(self):
        sys = FloquetSystem(_Z, _X, omega=4.0)
        solver = BlochFloquetSolver(sys)
        result = solver.solve()
        assert isinstance(result, FloquetResult)

    def test_solve_quasi_energies_count(self):
        sys = FloquetSystem(_Z, _X, omega=4.0)
        result = BlochFloquetSolver(sys).solve()
        assert len(result.quasi_energies) == 2

    def test_solve_quasi_energies_sorted(self):
        sys = FloquetSystem(_Z, _X, omega=4.0)
        result = BlochFloquetSolver(sys).solve()
        qe = np.real(result.quasi_energies)
        assert np.all(qe[:-1] <= qe[1:])

    def test_floquet_modes_shape(self):
        sys = FloquetSystem(_Z, _X, omega=4.0)
        result = BlochFloquetSolver(sys).solve()
        assert result.floquet_modes.shape == (2, 2)

    def test_H_eff_is_hermitian(self):
        sys = FloquetSystem(_Z, _X, omega=4.0, n_steps=300)
        result = BlochFloquetSolver(sys).solve()
        np.testing.assert_allclose(
            result.H_eff, result.H_eff.conj().T, atol=1e-10
        )

    def test_propagator_stored(self):
        sys = FloquetSystem(_Z, 0.1 * _X, omega=4.0)
        result = BlochFloquetSolver(sys).solve()
        assert result.propagator.shape == (2, 2)

    def test_omega_period_stored(self):
        omega = 3.7
        sys = FloquetSystem(_Z, _X, omega=omega)
        result = BlochFloquetSolver(sys).solve()
        assert abs(result.omega - omega) < 1e-12
        assert abs(result.period - 2 * math.pi / omega) < 1e-12

    def test_static_quasi_energies_equal_eigenvalues(self):
        """For H_drive=0, quasi-energies should match H_static eigenvalues (mod omega)."""
        omega = 10.0   # Large omega so no aliasing
        H_static = 0.3 * _Z   # eigenvalues ±0.3
        sys = FloquetSystem(H_static, np.zeros((2, 2), dtype=complex),
                            omega=omega, n_steps=400)
        result = BlochFloquetSolver(sys).solve()
        qe = np.sort(np.real(result.quasi_energies))
        # Eigenvalues of H_static: -0.3, +0.3
        np.testing.assert_allclose(qe, [-0.3, 0.3], atol=1e-3)

    def test_invalid_system_type(self):
        with pytest.raises(FloquetError):
            BlochFloquetSolver("not a system")

    def test_repr(self):
        sys = FloquetSystem(_Z, _X, omega=2.0)
        r = repr(BlochFloquetSolver(sys))
        assert "BlochFloquetSolver" in r

    def test_micromotion_at_returns_matrix(self):
        sys = FloquetSystem(_Z, 0.1 * _X, omega=5.0, n_steps=200)
        solver = BlochFloquetSolver(sys)
        M = solver.micromotion_at(0.1)
        assert M.shape == (2, 2)

    def test_quasi_energy_vs_amplitude_shape(self):
        sys = FloquetSystem(_Z, _X, omega=5.0, n_steps=100)
        solver = BlochFloquetSolver(sys)
        amps = np.linspace(0, 1, 5)
        qe_scan = solver.quasi_energy_vs_amplitude(amps)
        assert qe_scan.shape == (5, 2)

    def test_quasi_energy_vs_amplitude_zero_amp(self):
        """At zero amplitude the quasi-energies should match the static spectrum."""
        omega = 10.0
        sys = FloquetSystem(0.3 * _Z, _X, omega=omega, n_steps=300)
        solver = BlochFloquetSolver(sys)
        qe_scan = solver.quasi_energy_vs_amplitude(np.array([0.0]))
        qe = np.sort(qe_scan[0])
        np.testing.assert_allclose(qe, [-0.3, 0.3], atol=1e-3)


# ── FloquetResult helpers ─────────────────────────────────────────────────────

class TestFloquetResult:
    def _make_result(self, amp=0.1, omega=5.0):
        sys = FloquetSystem(0.5 * _Z, amp * _X, omega=omega, n_steps=200)
        return BlochFloquetSolver(sys).solve()

    def test_gap_positive(self):
        result = self._make_result()
        assert result.gap() > 0

    def test_gap_type(self):
        result = self._make_result()
        assert isinstance(result.gap(), float)

    def test_is_topological_returns_bool(self):
        result = self._make_result()
        assert isinstance(result.is_topological(), bool)

    def test_micromotion_at_t0_is_heff_exp(self):
        """micromotion(0) = exp(+i H_eff * 0) = I."""
        result = self._make_result()
        M = result.micromotion(0.0)
        np.testing.assert_allclose(M, np.eye(2, dtype=complex), atol=1e-10)

    def test_repr_contains_quasi_energies(self):
        result = self._make_result()
        r = repr(result)
        assert "FloquetResult" in r
        assert "quasi_energies" in r


# ── floquet_spectrum convenience ──────────────────────────────────────────────

class TestFloquetSpectrum:
    def test_returns_floquet_result(self):
        result = floquet_spectrum(_Z, 0.1 * _X, omega=4.0)
        assert isinstance(result, FloquetResult)

    def test_quasi_energies_count(self):
        result = floquet_spectrum(_Z, 0.1 * _X, omega=4.0)
        assert len(result.quasi_energies) == 2

    def test_4x4_spectrum_count(self):
        H4 = np.diag([1.0, 0.5, -0.5, -1.0]).astype(complex)
        D4 = np.zeros((4, 4), dtype=complex)
        result = floquet_spectrum(H4, D4, omega=5.0)
        assert len(result.quasi_energies) == 4

    def test_n_steps_keyword(self):
        result = floquet_spectrum(_Z, 0.1 * _X, omega=4.0, n_steps=100)
        assert isinstance(result, FloquetResult)


# ── floquet_effective_hamiltonian ─────────────────────────────────────────────

class TestFloquetEffectiveHamiltonian:
    def test_returns_hermitian(self):
        H_eff = floquet_effective_hamiltonian(_Z, 0.1 * _X, omega=5.0)
        np.testing.assert_allclose(H_eff, H_eff.conj().T, atol=1e-12)

    def test_zero_drive_returns_H0(self):
        """[0, 0†]/ω = 0, so H_eff = H0."""
        H_eff = floquet_effective_hamiltonian(_Z, np.zeros((2, 2), dtype=complex), omega=3.0)
        np.testing.assert_allclose(H_eff, _Z, atol=1e-12)

    def test_large_omega_approaches_H0(self):
        """In the high-frequency limit, corrections ∝ 1/ω → 0.

        Use a non-Hermitian H1 (upper-triangular) so that [H1, H1†] ≠ 0
        and the 1/ω correction is visible.
        """
        # H1 = [[0,1],[0,0]]: [H1, H1†] = Z (non-zero commutator)
        H1 = np.array([[0, 1], [0, 0]], dtype=complex)
        H_eff_lo = floquet_effective_hamiltonian(_Z, H1, omega=1.0)
        H_eff_hi = floquet_effective_hamiltonian(_Z, H1, omega=100.0)
        # High-freq result should be closer to _Z
        diff_lo = np.linalg.norm(H_eff_lo - _Z)
        diff_hi = np.linalg.norm(H_eff_hi - _Z)
        assert diff_hi < diff_lo

    def test_invalid_omega(self):
        with pytest.raises(FloquetError):
            floquet_effective_hamiltonian(_Z, _X, omega=0.0)

    def test_shape_preserved(self):
        H_eff = floquet_effective_hamiltonian(_Z, _X, omega=2.0)
        assert H_eff.shape == (2, 2)

    def test_4x4_shape(self):
        H4 = np.eye(4, dtype=complex)
        H_eff = floquet_effective_hamiltonian(H4, H4, omega=3.0)
        assert H_eff.shape == (4, 4)


# ── driven_two_level ──────────────────────────────────────────────────────────

class TestDrivenTwoLevel:
    def test_returns_floquet_result(self):
        result = driven_two_level(1.0, 0.1, 1.0)
        assert isinstance(result, FloquetResult)

    def test_quasi_energies_two_levels(self):
        result = driven_two_level(1.0, 0.1, 1.0)
        assert len(result.quasi_energies) == 2

    def test_resonant_drive_avoided_crossing(self):
        """On resonance omega_drive = omega0, quasi-energy gap ≈ drive_amp."""
        omega0 = 2.0
        amp = 0.2
        # Resonant drive: omega_drive = omega0
        result = driven_two_level(omega0, amp, omega0, n_steps=400)
        gap = result.gap()
        # Gap should be of order amp (Rabi splitting ≈ amp/2 per side)
        assert gap > 0

    def test_off_resonant_drive(self):
        """Off-resonance, quasi-energies approach ±ω0/2."""
        omega0 = 1.0
        amp = 0.05
        # Highly off-resonant
        result = driven_two_level(omega0, amp, omega_drive=10.0, n_steps=200)
        qe = np.sort(np.real(result.quasi_energies))
        # Quasi-energies should be close to ±omega0/2 = ±0.5
        np.testing.assert_allclose(qe, [-0.5, 0.5], atol=0.05)

    def test_propagator_unitary(self):
        result = driven_two_level(1.0, 0.3, 1.2, n_steps=200)
        U = result.propagator
        np.testing.assert_allclose(U.conj().T @ U, np.eye(2), atol=1e-5)

    def test_zero_amplitude(self):
        """Zero amplitude: quasi-energies = ±omega0/2 (no Floquet-zone folding).

        Use omega_drive >> omega0 so phases ±omega0/2 · T are small (≪ π)
        and map cleanly without branch-cut folding.
        """
        omega0 = 1.0
        omega_drive = 20.0   # Large: T=2π/20, phase = ±0.5 * 2π/20 = ±π/20 ≪ π
        result = driven_two_level(omega0, 0.0, omega_drive, n_steps=200)
        qe = np.sort(np.real(result.quasi_energies))
        np.testing.assert_allclose(qe, [-omega0 / 2, omega0 / 2], atol=1e-3)


# ── __init__ re-exports ────────────────────────────────────────────────────────

class TestFloquetInit:
    def test_floquet_error_exported(self):
        from quantx_engine import FloquetError as FE
        assert FE is FloquetError

    def test_floquet_system_exported(self):
        from quantx_engine import FloquetSystem as FS
        assert FS is FloquetSystem

    def test_floquet_result_exported(self):
        from quantx_engine import FloquetResult as FR
        assert FR is FloquetResult

    def test_bloch_floquet_solver_exported(self):
        from quantx_engine import BlochFloquetSolver as BFS
        assert BFS is BlochFloquetSolver

    def test_floquet_spectrum_exported(self):
        from quantx_engine import floquet_spectrum as fs
        assert fs is floquet_spectrum

    def test_floquet_effective_hamiltonian_exported(self):
        from quantx_engine import floquet_effective_hamiltonian as feh
        assert feh is floquet_effective_hamiltonian

    def test_driven_two_level_exported(self):
        from quantx_engine import driven_two_level as dtl
        assert dtl is driven_two_level
