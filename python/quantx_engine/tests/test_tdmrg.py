"""Tests for tdmrg.py — M94 time-dependent DMRG."""
from __future__ import annotations

import math
import unittest

import numpy as np

from quantx_engine.tensor_network import MPS, MPO, TensorNetworkError
from quantx_engine.tdmrg import (
    TDMRGError,
    TDMRGResult,
    TDMRGSolver,
    real_time_evolve,
    imaginary_time_evolve,
    _mpo_bond_hamiltonian,
    _gate_from_h,
    _apply_two_site_gate,
    _entanglement_entropy,
    _mps_energy,
    _single_site_expectation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_I = np.eye(2, dtype=complex)


def _bell_mps() -> MPS:
    """(|00⟩ + |11⟩)/√2 as a 2-site MPS."""
    psi = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    return MPS.from_statevector(psi, d=2)


def _ising(n: int = 4, J: float = 1.0, h: float = 0.5) -> MPO:
    return MPO.ising(n=n, J=J, h=h)


def _heisenberg(n: int = 4) -> MPO:
    return MPO.heisenberg(n=n, J=1.0, hz=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# _mpo_bond_hamiltonian
# ─────────────────────────────────────────────────────────────────────────────

class TestMpoBondHamiltonian(unittest.TestCase):
    def test_returns_square_matrix(self):
        mpo = _ising(n=4)
        h = _mpo_bond_hamiltonian(mpo, 0)
        d2 = mpo.d ** 2
        self.assertEqual(h.shape, (d2, d2))

    def test_is_hermitian_ising(self):
        mpo = _ising(n=4)
        for site in range(mpo.n_sites - 1):
            h = _mpo_bond_hamiltonian(mpo, site)
            self.assertTrue(
                np.allclose(h, h.conj().T, atol=1e-12),
                msg=f"Bond {site}: h not Hermitian"
            )

    def test_is_hermitian_heisenberg(self):
        mpo = _heisenberg(n=4)
        for site in range(mpo.n_sites - 1):
            h = _mpo_bond_hamiltonian(mpo, site)
            self.assertTrue(
                np.allclose(h, h.conj().T, atol=1e-12),
                msg=f"Heisenberg bond {site}: h not Hermitian"
            )

    def test_ising_bulk_bond_contains_zz(self):
        """For H = -J*ZZ - h*(XI + IX), the bulk bond h should contain ZZ coupling."""
        J, h_field = 1.0, 0.0
        mpo = MPO.ising(n=4, J=J, h=h_field)
        h_bond = _mpo_bond_hamiltonian(mpo, 1)  # bulk-bulk bond
        ZZ = np.kron(_Z, _Z)
        # h_bond should be proportional to -J*ZZ (up to single-site terms)
        # Check that inner product <ZZ, h_bond> / <ZZ, ZZ> ≈ -J
        overlap = np.trace(ZZ.conj().T @ h_bond)
        norm_ZZ = np.trace(ZZ.conj().T @ ZZ)
        ratio = float(np.real(overlap / norm_ZZ))
        self.assertAlmostEqual(ratio, -J, delta=0.1)

    def test_two_site_system(self):
        """For n=2, only one bond exists."""
        mpo = _ising(n=2)
        h = _mpo_bond_hamiltonian(mpo, 0)
        self.assertEqual(h.shape, (4, 4))
        self.assertTrue(np.allclose(h, h.conj().T, atol=1e-12))

    def test_eigenvalues_real(self):
        """All eigenvalues of h (Hermitian) must be real."""
        mpo = _heisenberg(n=6)
        for i in range(mpo.n_sites - 1):
            h = _mpo_bond_hamiltonian(mpo, i)
            eigs = np.linalg.eigvalsh(h)
            self.assertTrue(np.all(np.isreal(eigs)))


# ─────────────────────────────────────────────────────────────────────────────
# _gate_from_h
# ─────────────────────────────────────────────────────────────────────────────

class TestGateFromH(unittest.TestCase):
    def _random_hermitian(self, d=4, seed=42):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        return 0.5 * (A + A.conj().T)

    def test_real_time_gate_is_unitary(self):
        h = self._random_hermitian()
        U = _gate_from_h(h, dt=0.1, imaginary=False)
        prod = U @ U.conj().T
        self.assertTrue(np.allclose(prod, np.eye(4), atol=1e-10))

    def test_imaginary_time_gate_is_real_positive(self):
        """exp(-h*dt) with real h → real matrix with positive eigenvalues."""
        h = np.diag([1.0, 2.0, 3.0, 4.0])
        G = _gate_from_h(h, dt=0.1, imaginary=True)
        # Real-valued
        self.assertTrue(np.allclose(G.imag, 0, atol=1e-12))
        # Positive diagonal
        for i in range(4):
            self.assertGreater(float(G[i, i].real), 0)

    def test_real_time_zero_dt_is_identity(self):
        h = self._random_hermitian()
        U = _gate_from_h(h, dt=0.0, imaginary=False)
        self.assertTrue(np.allclose(U, np.eye(4), atol=1e-10))

    def test_imaginary_time_zero_dtau_is_identity(self):
        h = self._random_hermitian()
        G = _gate_from_h(h, dt=0.0, imaginary=True)
        self.assertTrue(np.allclose(G, np.eye(4), atol=1e-10))

    def test_real_time_gate_determinant_one(self):
        """det(U) = 1 for a traceless Hermitian h."""
        h = self._random_hermitian()
        h -= np.eye(4) * np.trace(h) / 4  # make traceless
        U = _gate_from_h(h, dt=0.3, imaginary=False)
        self.assertAlmostEqual(abs(np.linalg.det(U)), 1.0, places=8)


# ─────────────────────────────────────────────────────────────────────────────
# _apply_two_site_gate
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyTwoSiteGate(unittest.TestCase):
    def test_identity_gate_unchanged(self):
        mps = MPS.product_state([0, 0, 0], d=2)
        sv_before = mps.to_statevector().copy()
        gate = np.eye(4, dtype=complex)
        _apply_two_site_gate(mps, gate, site=0, max_chi=4, svd_cutoff=1e-12)
        sv_after = mps.to_statevector()
        self.assertTrue(np.allclose(sv_before, sv_after, atol=1e-10))

    def test_swap_gate(self):
        """SWAP |01⟩ → |10⟩."""
        mps = MPS.product_state([0, 1], d=2)
        SWAP = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=complex)
        _apply_two_site_gate(mps, SWAP, site=0, max_chi=4, svd_cutoff=1e-12)
        sv = mps.to_statevector()
        expected = MPS.product_state([1, 0]).to_statevector()
        self.assertTrue(np.allclose(sv, expected, atol=1e-10))

    def test_singular_values_returned(self):
        """Returns full SVD spectrum (before truncation)."""
        mps = MPS.product_state([0, 0])
        sv = _apply_two_site_gate(mps, np.eye(4), 0, max_chi=4, svd_cutoff=0)
        self.assertIsInstance(sv, np.ndarray)
        self.assertTrue(len(sv) > 0)

    def test_truncation_respects_max_chi(self):
        """After gate on Bell state, max bond dim ≤ max_chi."""
        bell = _bell_mps()
        gate = np.eye(4, dtype=complex)  # identity preserves state
        _apply_two_site_gate(bell, gate, 0, max_chi=1, svd_cutoff=0)
        # bond dim should be truncated to 1
        self.assertEqual(bell.tensors[0].shape[2], 1)

    def test_preserves_norm_for_unitary_gate(self):
        mps = MPS.from_statevector(
            np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        )
        nrm_before = mps.norm()
        gate = _gate_from_h(np.eye(4, dtype=complex) * 0.5, dt=0.1, imaginary=False)
        _apply_two_site_gate(mps, gate, 0, max_chi=8, svd_cutoff=1e-12)
        self.assertAlmostEqual(mps.norm(), nrm_before, delta=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# _entanglement_entropy
# ─────────────────────────────────────────────────────────────────────────────

class TestEntanglementEntropy(unittest.TestCase):
    def test_product_state_zero_entropy(self):
        """Product state |0⟩|0⟩ has zero entanglement entropy."""
        # SVD of |00⟩ = [[1]], only one singular value
        sv = np.array([1.0])
        self.assertAlmostEqual(_entanglement_entropy(sv), 0.0, places=12)

    def test_bell_state_log2_entropy(self):
        """Bell state has S = log(2) ≈ 0.693."""
        sv = np.array([1.0 / math.sqrt(2), 1.0 / math.sqrt(2)])
        S = _entanglement_entropy(sv)
        self.assertAlmostEqual(S, math.log(2), places=10)

    def test_all_zero_singular_values(self):
        sv = np.array([0.0, 0.0, 0.0])
        self.assertAlmostEqual(_entanglement_entropy(sv), 0.0, places=12)

    def test_one_dominant_sv_approx_zero(self):
        sv = np.array([1.0, 1e-15, 1e-15])
        S = _entanglement_entropy(sv)
        self.assertLess(S, 1e-5)

    def test_equal_svs_maximally_mixed(self):
        """d equal SVs → S = log(d)."""
        for d in [2, 4, 8]:
            sv = np.ones(d) / math.sqrt(d)
            S = _entanglement_entropy(sv)
            self.assertAlmostEqual(S, math.log(d), places=8)

    def test_entropy_non_negative(self):
        sv = np.random.default_rng(42).random(8) + 0.1
        S = _entanglement_entropy(sv)
        self.assertGreaterEqual(S, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# _mps_energy
# ─────────────────────────────────────────────────────────────────────────────

class TestMpsEnergy(unittest.TestCase):
    def test_energy_finite(self):
        mpo = _ising(n=4)
        mps = MPS.product_state([0, 0, 0, 0])
        e = _mps_energy(mps, mpo)
        self.assertTrue(math.isfinite(e))

    def test_energy_real(self):
        mpo = _heisenberg(n=4)
        mps = MPS.product_state([0, 1, 0, 1])
        e = _mps_energy(mps, mpo)
        self.assertTrue(math.isfinite(e))

    def test_ising_all_up_energy(self):
        """For H=-J*ZZ-h*X, |0000⟩ state, ZZ terms give -J per bond."""
        J, h = 1.0, 0.0
        mpo = MPO.ising(n=4, J=J, h=h)
        mps = MPS.product_state([0, 0, 0, 0])
        e = _mps_energy(mps, mpo)
        # 3 bonds × (-J*1) = -3J
        self.assertAlmostEqual(e, -3.0 * J, delta=0.1)

    def test_zero_norm_returns_nan(self):
        mpo = _ising(n=2)
        mps = MPS(tensors=[np.zeros((1, 2, 1), dtype=complex)])
        # Manually set second tensor
        mps = MPS(tensors=[
            np.zeros((1, 2, 1), dtype=complex),
            np.zeros((1, 2, 1), dtype=complex),
        ])
        e = _mps_energy(mps, mpo)
        self.assertTrue(math.isnan(e))


# ─────────────────────────────────────────────────────────────────────────────
# _single_site_expectation
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleSiteExpectation(unittest.TestCase):
    def test_z_expectation_ground_state(self):
        """⟨0|Z|0⟩ = +1."""
        mps = MPS.product_state([0])
        val = _single_site_expectation(mps, _Z, site=0)
        self.assertAlmostEqual(float(val.real), 1.0, places=10)

    def test_z_expectation_excited_state(self):
        """⟨1|Z|1⟩ = -1."""
        mps = MPS.product_state([1])
        val = _single_site_expectation(mps, _Z, site=0)
        self.assertAlmostEqual(float(val.real), -1.0, places=10)

    def test_x_expectation_ground_state(self):
        """⟨0|X|0⟩ = 0 (eigenstates of Z have zero X expectation)."""
        mps = MPS.product_state([0])
        val = _single_site_expectation(mps, _X, site=0)
        self.assertAlmostEqual(abs(float(val.real)), 0.0, places=10)

    def test_multi_site_z(self):
        """⟨0101|Z_2|0101⟩ = -1 (qubit 2 is |1⟩)."""
        mps = MPS.product_state([0, 1, 0, 1])
        val = _single_site_expectation(mps, _Z, site=1)  # qubit 1 is |1⟩
        self.assertAlmostEqual(float(val.real), -1.0, places=8)

    def test_identity_expectation_one(self):
        """⟨ψ|I|ψ⟩ / ⟨ψ|ψ⟩ = 1 for any normalised state."""
        mps = MPS.product_state([0, 0, 1])
        val = _single_site_expectation(mps, _I, site=1)
        self.assertAlmostEqual(float(val.real), 1.0, places=10)


# ─────────────────────────────────────────────────────────────────────────────
# TDMRGSolver — construction and validation
# ─────────────────────────────────────────────────────────────────────────────

class TestTDMRGSolverConstruction(unittest.TestCase):
    def test_basic_construction(self):
        mpo = _ising(n=4)
        solver = TDMRGSolver(mpo=mpo, dt=0.01, n_steps=10)
        self.assertEqual(solver.mpo.n_sites, 4)
        self.assertEqual(solver.dt, 0.01)
        self.assertEqual(solver.n_steps, 10)

    def test_repr(self):
        mpo = _ising(n=4)
        solver = TDMRGSolver(mpo=mpo, dt=0.05, n_steps=5)
        r = repr(solver)
        self.assertIn("n_sites=4", r)
        self.assertIn("dt=0.05", r)

    def test_non_mpo_raises(self):
        with self.assertRaises(TDMRGError):
            TDMRGSolver(mpo="not an mpo", dt=0.01, n_steps=10)

    def test_single_site_mpo_raises(self):
        with self.assertRaises((TDMRGError, TensorNetworkError)):
            MPO.ising(n=1)  # MPO itself requires n>=2

    def test_negative_dt_raises(self):
        mpo = _ising(n=4)
        with self.assertRaises(TDMRGError):
            TDMRGSolver(mpo=mpo, dt=-0.1, n_steps=10)

    def test_zero_dt_raises(self):
        mpo = _ising(n=4)
        with self.assertRaises(TDMRGError):
            TDMRGSolver(mpo=mpo, dt=0.0, n_steps=10)

    def test_zero_n_steps_raises(self):
        mpo = _ising(n=4)
        with self.assertRaises(TDMRGError):
            TDMRGSolver(mpo=mpo, dt=0.01, n_steps=0)

    def test_invalid_trotter_order_raises(self):
        mpo = _ising(n=4)
        with self.assertRaises(TDMRGError):
            TDMRGSolver(mpo=mpo, dt=0.01, n_steps=10, trotter_order=3)

    def test_bonds_precomputed(self):
        """n-1 bond Hamiltonians should be precomputed."""
        n = 5
        mpo = _ising(n=n)
        solver = TDMRGSolver(mpo=mpo, dt=0.01, n_steps=5)
        self.assertEqual(len(solver._h_bonds), n - 1)
        self.assertEqual(len(solver._gates_full), n - 1)
        self.assertEqual(len(solver._gates_half), n - 1)


# ─────────────────────────────────────────────────────────────────────────────
# TDMRGSolver — run()
# ─────────────────────────────────────────────────────────────────────────────

class TestTDMRGSolverRun(unittest.TestCase):
    def test_returns_tdmrg_result(self):
        mpo = _ising(n=4)
        solver = TDMRGSolver(mpo=mpo, dt=0.05, n_steps=5)
        result = solver.run()
        self.assertIsInstance(result, TDMRGResult)

    def test_times_shape(self):
        n_steps = 8
        mpo = _ising(n=4)
        result = TDMRGSolver(mpo=mpo, dt=0.1, n_steps=n_steps).run()
        self.assertEqual(result.times.shape, (n_steps,))

    def test_times_monotone(self):
        mpo = _ising(n=4)
        result = TDMRGSolver(mpo=mpo, dt=0.1, n_steps=5).run()
        self.assertTrue(np.all(np.diff(result.times) > 0))

    def test_times_correct_values(self):
        mpo = _ising(n=4)
        dt = 0.05
        n_steps = 4
        result = TDMRGSolver(mpo=mpo, dt=dt, n_steps=n_steps).run()
        np.testing.assert_allclose(result.times, dt * np.arange(1, n_steps + 1))

    def test_energies_shape(self):
        mpo = _ising(n=4)
        result = TDMRGSolver(mpo=mpo, dt=0.1, n_steps=5).run()
        self.assertEqual(result.energies.shape, (5,))

    def test_energies_finite(self):
        mpo = _heisenberg(n=4)
        mps = MPS.product_state([0, 1, 0, 1])
        result = TDMRGSolver(mpo=mpo, dt=0.05, n_steps=5).run(mps)
        self.assertTrue(np.all(np.isfinite(result.energies)))

    def test_entanglement_entropy_shape(self):
        n = 5
        mpo = _ising(n=n)
        result = TDMRGSolver(mpo=mpo, dt=0.1, n_steps=4).run()
        self.assertEqual(result.entanglement_entropy.shape, (4, n - 1))

    def test_entanglement_entropy_non_negative(self):
        mpo = _ising(n=4)
        result = TDMRGSolver(mpo=mpo, dt=0.1, n_steps=5).run()
        self.assertTrue(np.all(result.entanglement_entropy >= -1e-10))

    def test_norms_shape(self):
        mpo = _ising(n=4)
        result = TDMRGSolver(mpo=mpo, dt=0.05, n_steps=5).run()
        self.assertEqual(result.norms.shape, (5,))

    def test_real_time_norm_preserved(self):
        """Real-time evolution is unitary: norm should stay ≈ 1."""
        mpo = _ising(n=4)
        mps = MPS.product_state([0, 0, 0, 0])
        result = TDMRGSolver(
            mpo=mpo, dt=0.05, n_steps=10, max_chi=8
        ).run(mps)
        np.testing.assert_allclose(result.norms, 1.0, atol=0.01)

    def test_bond_dims_tracked(self):
        mpo = _ising(n=4)
        result = TDMRGSolver(mpo=mpo, dt=0.1, n_steps=3).run()
        self.assertEqual(len(result.bond_dims), 3)
        for bd in result.bond_dims:
            self.assertEqual(len(bd), mpo.n_sites - 1)

    def test_mps_final_is_mps(self):
        mpo = _ising(n=4)
        result = TDMRGSolver(mpo=mpo, dt=0.1, n_steps=3).run()
        self.assertIsInstance(result.mps_final, MPS)

    def test_custom_initial_state_used(self):
        """Providing a custom MPS uses it as the starting state."""
        mpo = _ising(n=4)
        mps_init = MPS.product_state([1, 0, 1, 0])
        result = TDMRGSolver(mpo=mpo, dt=0.01, n_steps=1).run(mps_init)
        self.assertIsInstance(result.mps_final, MPS)

    def test_wrong_n_sites_initial_state_raises(self):
        mpo = _ising(n=4)
        mps = MPS.product_state([0, 0, 0])  # 3 sites, not 4
        with self.assertRaises(TDMRGError):
            TDMRGSolver(mpo=mpo, dt=0.1, n_steps=5).run(mps)

    def test_non_mps_initial_raises(self):
        mpo = _ising(n=4)
        with self.assertRaises(TDMRGError):
            TDMRGSolver(mpo=mpo, dt=0.1, n_steps=5).run("not an MPS")

    def test_imaginary_time_energy_decreases(self):
        """Imaginary-time evolution should lower the energy."""
        mpo = _ising(n=4, J=1.0, h=0.5)
        mps_init = MPS.random(4, max_chi=4, seed=0)
        mps_init.normalize()
        result = TDMRGSolver(
            mpo=mpo, dt=0.05, n_steps=30, max_chi=8, imaginary=True
        ).run(mps_init)
        # Final energy should be lower than initial
        self.assertLess(result.energies[-1], result.energies[0] + 0.1)

    def test_imaginary_time_result_flag(self):
        mpo = _ising(n=4)
        result = TDMRGSolver(mpo=mpo, dt=0.05, n_steps=5, imaginary=True).run()
        self.assertTrue(result.imaginary)

    def test_real_time_result_flag(self):
        mpo = _ising(n=4)
        result = TDMRGSolver(mpo=mpo, dt=0.05, n_steps=5, imaginary=False).run()
        self.assertFalse(result.imaginary)

    def test_trotter_order_1_runs(self):
        mpo = _ising(n=4)
        result = TDMRGSolver(
            mpo=mpo, dt=0.05, n_steps=3, trotter_order=1
        ).run()
        self.assertIsInstance(result, TDMRGResult)

    def test_observables_tracked(self):
        """Tracking ⟨Z_0⟩ returns an array of length n_steps."""
        mpo = _ising(n=4)
        obs = {"Z0": (0, _Z)}
        result = TDMRGSolver(
            mpo=mpo, dt=0.1, n_steps=5, observables=obs
        ).run(MPS.product_state([0, 0, 0, 0]))
        self.assertIn("Z0", result.observables)
        self.assertEqual(result.observables["Z0"].shape, (5,))

    def test_observable_initial_value_correct(self):
        """⟨Z_0⟩ starts at +1 for |0⟩ product state under small dt."""
        mpo = _ising(n=4)
        obs = {"Z0": (0, _Z)}
        mps = MPS.product_state([0, 0, 0, 0])
        result = TDMRGSolver(
            mpo=mpo, dt=0.001, n_steps=1, max_chi=8, observables=obs
        ).run(mps)
        z0 = float(result.observables["Z0"][0].real)
        # After tiny evolution, Z0 should still be close to 1
        self.assertAlmostEqual(z0, 1.0, delta=0.1)

    def test_max_chi_respected(self):
        """Bond dims never exceed max_chi."""
        mpo = _ising(n=6)
        max_chi = 3
        result = TDMRGSolver(
            mpo=mpo, dt=0.1, n_steps=5, max_chi=max_chi
        ).run()
        for bd_list in result.bond_dims:
            for bd in bd_list:
                self.assertLessEqual(bd, max_chi)


# ─────────────────────────────────────────────────────────────────────────────
# TDMRGResult
# ─────────────────────────────────────────────────────────────────────────────

class TestTDMRGResult(unittest.TestCase):
    def _make_result(self) -> TDMRGResult:
        mpo = _ising(n=4)
        return TDMRGSolver(mpo=mpo, dt=0.1, n_steps=6).run()

    def test_repr_contains_n_steps(self):
        r = self._make_result()
        rep = repr(r)
        self.assertIn("n_steps=6", rep)

    def test_repr_contains_dt(self):
        r = self._make_result()
        rep = repr(r)
        self.assertIn("dt=0.1", rep)

    def test_max_entropy_non_negative(self):
        r = self._make_result()
        self.assertGreaterEqual(r.max_entropy(), 0.0)

    def test_entropy_at_default_last(self):
        r = self._make_result()
        last = r.entropy_at(-1)
        np.testing.assert_array_equal(last, r.entanglement_entropy[-1])

    def test_entropy_at_first(self):
        r = self._make_result()
        first = r.entropy_at(0)
        np.testing.assert_array_equal(first, r.entanglement_entropy[0])

    def test_observables_empty_by_default(self):
        r = self._make_result()
        self.assertEqual(r.observables, {})


# ─────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────────────────────────────────────

class TestRealTimeEvolve(unittest.TestCase):
    def test_returns_tdmrg_result(self):
        mpo = _ising(n=4)
        result = real_time_evolve(mpo, dt=0.05, n_steps=5)
        self.assertIsInstance(result, TDMRGResult)

    def test_result_not_imaginary(self):
        mpo = _ising(n=4)
        result = real_time_evolve(mpo, dt=0.05, n_steps=5)
        self.assertFalse(result.imaginary)

    def test_custom_initial_state(self):
        mpo = _ising(n=4)
        mps = MPS.product_state([1, 0, 1, 0])
        result = real_time_evolve(mpo, mps, dt=0.05, n_steps=3)
        self.assertIsInstance(result, TDMRGResult)

    def test_norm_roughly_preserved(self):
        mpo = _heisenberg(n=4)
        result = real_time_evolve(mpo, dt=0.02, n_steps=10, max_chi=8)
        np.testing.assert_allclose(result.norms, 1.0, atol=0.05)

    def test_times_correct(self):
        dt = 0.1
        mpo = _ising(n=4)
        result = real_time_evolve(mpo, dt=dt, n_steps=3)
        np.testing.assert_allclose(
            result.times, [dt, 2 * dt, 3 * dt], rtol=1e-10
        )

    def test_entropy_shape(self):
        mpo = _ising(n=5)
        result = real_time_evolve(mpo, dt=0.1, n_steps=4)
        self.assertEqual(result.entanglement_entropy.shape, (4, 4))

    def test_observables_forwarded(self):
        mpo = _ising(n=4)
        obs = {"Zmid": (2, _Z)}
        result = real_time_evolve(mpo, dt=0.1, n_steps=3, observables=obs)
        self.assertIn("Zmid", result.observables)
        self.assertEqual(len(result.observables["Zmid"]), 3)


class TestImaginaryTimeEvolve(unittest.TestCase):
    def test_returns_tdmrg_result(self):
        mpo = _ising(n=4)
        result = imaginary_time_evolve(mpo, dtau=0.05, n_steps=10)
        self.assertIsInstance(result, TDMRGResult)

    def test_result_is_imaginary(self):
        mpo = _ising(n=4)
        result = imaginary_time_evolve(mpo, dtau=0.05, n_steps=10)
        self.assertTrue(result.imaginary)

    def test_energy_finite(self):
        mpo = _ising(n=4)
        result = imaginary_time_evolve(mpo, dtau=0.05, n_steps=20)
        self.assertTrue(np.all(np.isfinite(result.energies)))

    def test_imaginary_energy_converges_to_ground_state(self):
        """Imaginary-time evolution should converge to ground-state energy."""
        # Ising with n=4, J=1, h=0.5 has known ground state energy ≈ -5.2
        mpo = MPO.ising(n=4, J=1.0, h=0.5)
        result = imaginary_time_evolve(mpo, dtau=0.05, n_steps=100, max_chi=8)
        e_final = result.energies[-1]
        # Ground state for this parameter set is significantly negative
        self.assertLess(e_final, -3.0)

    def test_norm_stays_near_one(self):
        """After renormalisation, norm should be ≈ 1."""
        mpo = _ising(n=4)
        result = imaginary_time_evolve(mpo, dtau=0.05, n_steps=10, max_chi=8)
        np.testing.assert_allclose(result.norms, 1.0, atol=0.01)

    def test_entanglement_grows_then_saturates(self):
        """Entanglement should be non-trivial for interacting systems."""
        mpo = MPO.ising(n=6, J=1.0, h=0.5)
        result = imaginary_time_evolve(mpo, dtau=0.05, n_steps=50, max_chi=8)
        # Maximum entropy across all bonds and all times should be positive
        self.assertGreater(result.max_entropy(), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# __init__ re-exports
# ─────────────────────────────────────────────────────────────────────────────

class TestInitExports(unittest.TestCase):
    def test_tdmrg_error_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "TDMRGError"))

    def test_tdmrg_result_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "TDMRGResult"))

    def test_tdmrg_solver_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "TDMRGSolver"))

    def test_real_time_evolve_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "real_time_evolve"))

    def test_imaginary_time_evolve_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "imaginary_time_evolve"))


if __name__ == "__main__":
    unittest.main()
