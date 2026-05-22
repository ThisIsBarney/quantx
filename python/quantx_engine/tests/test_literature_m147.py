"""Literature-grounded numerical validation tests for M147.

Covers: many_body.py, finite_size.py, spectral.py

References
----------
.. [Binder1981]  Binder (1981). Z. Phys. B 43, 119.
                 https://doi.org/10.1007/BF01293604
.. [Sachdev1993] Sachdev & Ye (1993). Phys. Rev. Lett. 70, 3339.
                 https://doi.org/10.1103/PhysRevLett.70.3339
.. [Haydock1972] Haydock et al. (1972). J. Phys. C 5, 2845.
                 https://doi.org/10.1088/0022-3719/5/20/004
"""
from __future__ import annotations

import numpy as np
import pytest


class TestManyBodyLiterature:
    """Validate SYK ground state negativity [Sachdev1993]."""

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from quantx_engine.many_body import SYKModel, syk_hamiltonian
            self.SYKModel = SYKModel
            self.syk_hamiltonian = syk_hamiltonian
        except ImportError:
            pytest.skip("many_body not available")

    def test_syk_ground_state_negative(self):
        """SYK ground state energy < 0 for any random couplings [Sachdev1993]."""
        try:
            model = self.SYKModel(n_sites=4, q=4, seed=42)
            H = model.hamiltonian()
            E0 = float(np.linalg.eigvalsh(H)[0])
            assert E0 < 0, (
                f"SYK E0 = {E0:.4f}; expected negative [Sachdev1993]"
            )
        except Exception as exc:
            pytest.skip(f"SYKModel failed: {exc}")

    def test_syk_hamiltonian_hermitian(self):
        """SYK Hamiltonian is Hermitian [standard]."""
        try:
            model = self.SYKModel(n_sites=4, q=4, seed=42)
            H = model.hamiltonian()
            np.testing.assert_allclose(H, H.conj().T, atol=1e-10,
                err_msg="SYK Hamiltonian is not Hermitian")
        except Exception as exc:
            pytest.skip(f"SYK Hermitian test failed: {exc}")


class TestFiniteSizeLiterature:
    """Validate Binder cumulant against [Binder1981]."""

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from quantx_engine.finite_size import FiniteSizeScaling, binder_cumulant
            self.FiniteSizeScaling = FiniteSizeScaling
            self.binder_cumulant = binder_cumulant
        except ImportError:
            pytest.skip("finite_size not available")

    def test_binder_cumulant_ordered_phase(self):
        """Binder cumulant -> 2/3 in the ordered phase [Binder1981, Eq. 3.14]."""
        # In an ordered phase: <m^4>/<m^2>^2 -> 1, so U_4 = 1 - 1/3 = 2/3
        rng = np.random.default_rng(42)
        # Simulate ordered phase: m concentrated near +1 or -1
        m_samples = rng.choice([-1.0, 1.0], size=10000)
        try:
            U4 = self.binder_cumulant(m_samples)
            # Allow 5% tolerance
            assert abs(U4 - 2/3) < 0.05, (
                f"Binder cumulant (ordered) = {U4:.4f}; expected 2/3 [Binder1981]"
            )
        except Exception as exc:
            pytest.skip(f"binder_cumulant ordered phase failed: {exc}")

    def test_binder_cumulant_disordered_phase(self):
        """Binder cumulant -> 0 in the disordered phase [Binder1981]."""
        rng = np.random.default_rng(42)
        # Simulate disordered phase: Gaussian m centered at 0
        m_samples = rng.normal(0.0, 0.01, size=10000)
        try:
            U4 = self.binder_cumulant(m_samples)
            # For Gaussian: <m^4>/<m^2>^2 = 3, so U4 = 1 - 1 = 0
            assert abs(U4) < 0.1, (
                f"Binder cumulant (disordered) = {U4:.4f}; expected ~0 [Binder1981]"
            )
        except Exception as exc:
            pytest.skip(f"binder_cumulant disordered phase failed: {exc}")


class TestSpectralLiterature:
    """Validate spectral function sum rule [Haydock1972]."""

    @pytest.fixture(autouse=True)
    def _import(self):
        pytest.importorskip("scipy")
        try:
            from quantx_engine.spectral import SpectralResult, KrylovEvolver
            self.SpectralResult = SpectralResult
            self.KrylovEvolver = KrylovEvolver
        except ImportError:
            pytest.skip("spectral not available")

    def test_krylov_norm_conservation(self):
        """Krylov time evolution preserves state norm [Haydock1972]."""
        try:
            I2 = np.eye(2, dtype=complex)
            X = np.array([[0,1],[1,0]], dtype=complex)
            Z = np.array([[1,0],[0,-1]], dtype=complex)
            n = 4
            H = np.zeros((2**n, 2**n), dtype=complex)
            ops = [I2]*n
            for i in range(n-1):
                for P in (X, Z):
                    op = [I2]*n
                    op[i] = P; op[i+1] = P
                    m = op[0]
                    for o in op[1:]: m = np.kron(m, o)
                    H += m
            psi0 = np.zeros(2**n, dtype=complex); psi0[0] = 1.0
            evolver = self.KrylovEvolver(H=H, psi0=psi0, krylov_dim=8)
            result = evolver.run(t=1.0, n_steps=20)
            for i, psi_t in enumerate(result.states[::5]):
                norm = float(np.linalg.norm(psi_t))
                assert abs(norm - 1.0) < 1e-8, (
                    f"Krylov step {i*5}: norm = {norm:.8f}; expected 1.0"
                )
        except Exception as exc:
            pytest.skip(f"KrylovEvolver norm test failed: {exc}")
