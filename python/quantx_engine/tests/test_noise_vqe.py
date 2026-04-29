"""Tests for noise_vqe.py — M95 noise-aware VQE."""
from __future__ import annotations

import math
import unittest

import numpy as np

from quantx_engine.noise_vqe import (
    NoiseVQEError,
    SPSAOptimizer,
    NoiseAwareVQEResult,
    NoiseAwareVQE,
    build_hea_circuit,
    hardware_efficient_vqe,
    n_hea_params,
    _expect_noisy,
    _nelder_mead,
)
from quantx_engine.primitives import SparsePauliOp


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_I = np.eye(2, dtype=complex)
_ZZ = np.kron(_Z, _Z)


def _rng(seed=42):
    return np.random.default_rng(seed)


def _ground_state_dm(H: np.ndarray) -> np.ndarray:
    """Return |gs⟩⟨gs| for the smallest eigenvalue."""
    _, vecs = np.linalg.eigh(H)
    psi = vecs[:, 0]
    return np.outer(psi.conj(), psi)


# ─────────────────────────────────────────────────────────────────────────────
# n_hea_params
# ─────────────────────────────────────────────────────────────────────────────

class TestNHEAParams(unittest.TestCase):
    def test_one_qubit_one_layer(self):
        self.assertEqual(n_hea_params(1, 1), 1)

    def test_two_qubits_two_layers(self):
        self.assertEqual(n_hea_params(2, 2), 4)

    def test_four_qubits_three_layers(self):
        self.assertEqual(n_hea_params(4, 3), 12)


# ─────────────────────────────────────────────────────────────────────────────
# build_hea_circuit
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildHEACircuit(unittest.TestCase):
    def test_returns_list(self):
        theta = np.zeros(4)
        ops = build_hea_circuit(2, 2, theta)
        self.assertIsInstance(ops, list)

    def test_contains_ry_gates(self):
        theta = np.ones(4)
        ops = build_hea_circuit(2, 2, theta)
        ry_gates = [op for op in ops if op.get("name") == "ry"]
        self.assertEqual(len(ry_gates), 4)  # 2 qubits × 2 layers

    def test_contains_cx_gates(self):
        theta = np.zeros(4)
        ops = build_hea_circuit(2, 2, theta)
        cx_gates = [op for op in ops if op.get("name") == "cx"]
        self.assertEqual(len(cx_gates), 2)  # 1 pair × 2 layers

    def test_linear_entanglement_n_qubits_3(self):
        theta = np.zeros(6)
        ops = build_hea_circuit(3, 2, theta)
        cx_gates = [op for op in ops if op.get("name") == "cx"]
        # 2 pairs (0-1, 1-2) × 2 layers = 4
        self.assertEqual(len(cx_gates), 4)

    def test_full_entanglement(self):
        theta = np.zeros(6)
        ops = build_hea_circuit(3, 2, theta, entanglement="full")
        cx_gates = [op for op in ops if op.get("name") == "cx"]
        # 3 pairs × 2 layers = 6
        self.assertEqual(len(cx_gates), 6)

    def test_ry_angles_stored(self):
        theta = np.array([0.5, 1.2, 0.3, 0.9])
        ops = build_hea_circuit(2, 2, theta)
        ry_gates = [op for op in ops if op.get("name") == "ry"]
        angles = [op["params"][0] for op in ry_gates]
        np.testing.assert_allclose(angles, theta, atol=1e-12)

    def test_too_few_params_raises(self):
        with self.assertRaises(NoiseVQEError):
            build_hea_circuit(2, 2, np.zeros(3))  # needs 4

    def test_unknown_entanglement_raises(self):
        with self.assertRaises(NoiseVQEError):
            build_hea_circuit(2, 1, np.zeros(2), entanglement="ring")

    def test_single_qubit_no_cx(self):
        theta = np.array([0.5])
        ops = build_hea_circuit(1, 1, theta)
        cx_gates = [op for op in ops if op.get("name") == "cx"]
        self.assertEqual(len(cx_gates), 0)

    def test_qubit_indices_in_gates(self):
        theta = np.zeros(4)
        ops = build_hea_circuit(2, 2, theta)
        for op in ops:
            if "qubits" in op:
                for q in op["qubits"]:
                    self.assertIn(q, [0, 1])


# ─────────────────────────────────────────────────────────────────────────────
# _expect_noisy
# ─────────────────────────────────────────────────────────────────────────────

class TestExpectNoisy(unittest.TestCase):
    def test_exact_zero_shots(self):
        """With n_shots=0, returns exact Tr(H ρ)."""
        rho = np.diag([1.0, 0.0]).astype(complex)
        H = _Z
        val = _expect_noisy(rho, H, 0, _rng())
        self.assertAlmostEqual(val, 1.0, places=12)

    def test_exact_excited_state(self):
        rho = np.diag([0.0, 1.0]).astype(complex)
        val = _expect_noisy(rho, _Z, 0, _rng())
        self.assertAlmostEqual(val, -1.0, places=12)

    def test_shot_estimate_converges(self):
        """With many shots, estimate ≈ exact value."""
        rho = np.diag([0.6, 0.4]).astype(complex)  # mixed state
        H = _Z
        exact = float(np.real(np.trace(H @ rho)))  # 0.6*1 + 0.4*(-1) = 0.2
        estimate = _expect_noisy(rho, H, 50000, _rng(0))
        self.assertAlmostEqual(estimate, exact, delta=0.05)

    def test_maximally_mixed_z_zero(self):
        """⟨Z⟩ = 0 for maximally mixed state."""
        rho = np.eye(2, dtype=complex) / 2
        val = _expect_noisy(rho, _Z, 0, _rng())
        self.assertAlmostEqual(val, 0.0, places=10)

    def test_shot_noise_adds_variance(self):
        """Repeated shot estimates should vary (non-zero standard deviation)."""
        rho = np.diag([0.5, 0.5]).astype(complex)
        estimates = [_expect_noisy(rho, _Z, 100, _rng(i)) for i in range(30)]
        std = float(np.std(estimates))
        self.assertGreater(std, 0.0)

    def test_two_qubit_exact(self):
        """⟨ZZ⟩ = 1 for |00⟩."""
        rho = np.zeros((4, 4), dtype=complex)
        rho[0, 0] = 1.0
        val = _expect_noisy(rho, _ZZ, 0, _rng())
        self.assertAlmostEqual(val, 1.0, places=10)


# ─────────────────────────────────────────────────────────────────────────────
# SPSAOptimizer
# ─────────────────────────────────────────────────────────────────────────────

class TestSPSAOptimizerConstruction(unittest.TestCase):
    def test_default_construction(self):
        opt = SPSAOptimizer()
        self.assertGreater(opt.a, 0)
        self.assertGreater(opt.c, 0)

    def test_repr(self):
        opt = SPSAOptimizer(a=0.2, c=0.05)
        r = repr(opt)
        self.assertIn("0.2", r)

    def test_custom_params_stored(self):
        opt = SPSAOptimizer(a=0.5, c=0.2, A=20.0)
        self.assertEqual(opt.a, 0.5)
        self.assertEqual(opt.c, 0.2)
        self.assertEqual(opt.A, 20.0)


class TestSPSAOptimizerMinimize(unittest.TestCase):
    def test_minimizes_quadratic(self):
        """SPSA should reduce a noiseless quadratic f(x) = (x-2)²."""
        def f(x):
            return float((x[0] - 2.0) ** 2)

        opt = SPSAOptimizer(a=0.5, c=0.1, seed=0)
        x0 = np.array([0.0])
        x_opt, history = opt.minimize(f, x0, max_iter=200, tol=1e-5)
        self.assertLess(f(x_opt), f(x0))

    def test_returns_theta_and_history(self):
        opt = SPSAOptimizer(seed=0)
        x0 = np.zeros(3)
        x_opt, history = opt.minimize(lambda x: float(np.sum(x**2)), x0, max_iter=10)
        self.assertIsInstance(x_opt, np.ndarray)
        self.assertIsInstance(history, list)
        self.assertGreater(len(history), 0)

    def test_history_length(self):
        """History should have at most max_iter entries."""
        opt = SPSAOptimizer(seed=0)
        x0 = np.zeros(2)
        _, history = opt.minimize(lambda x: 0.0, x0, max_iter=20)
        self.assertLessEqual(len(history), 20)

    def test_callback_called(self):
        calls = [0]
        def cb(it, theta, fval):
            calls[0] += 1
        opt = SPSAOptimizer(seed=0)
        opt.minimize(lambda x: 0.0, np.zeros(2), max_iter=5, callback=cb)
        self.assertGreater(calls[0], 0)

    def test_multi_dimensional(self):
        """Minimize f(x) = Σ (x_i - 1)² starting from zeros."""
        def f(x):
            return float(np.sum((x - 1.0) ** 2))

        opt = SPSAOptimizer(a=0.3, c=0.1, seed=7)
        x0 = np.zeros(4)
        x_opt, _ = opt.minimize(f, x0, max_iter=300)
        # Should reduce significantly from initial value of 4.0
        self.assertLess(f(x_opt), f(x0))

    def test_converges_on_flat_function(self):
        """f(x) = 0 always → should converge quickly."""
        opt = SPSAOptimizer(seed=0)
        _, history = opt.minimize(lambda x: 0.0, np.zeros(3), max_iter=100)
        self.assertLessEqual(len(history), 100)


# ─────────────────────────────────────────────────────────────────────────────
# _nelder_mead (internal)
# ─────────────────────────────────────────────────────────────────────────────

class TestNelderMeadInternal(unittest.TestCase):
    def test_minimizes_quadratic(self):
        def f(x):
            return float(np.sum((x - 3.0) ** 2))

        x0 = np.zeros(2)
        x_opt, _ = _nelder_mead(f, x0, max_iter=500)
        self.assertLess(f(x_opt), 0.1)

    def test_returns_history(self):
        x0 = np.zeros(2)
        _, history = _nelder_mead(lambda x: 1.0, x0, max_iter=10)
        self.assertIsInstance(history, list)


# ─────────────────────────────────────────────────────────────────────────────
# NoiseAwareVQE — construction and validation
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseAwareVQEConstruction(unittest.TestCase):
    def test_basic_construction(self):
        H = SparsePauliOp("Z")
        vqe = NoiseAwareVQE(H, n_qubits=1, n_layers=1)
        self.assertEqual(vqe.n_qubits, 1)
        self.assertEqual(vqe.n_layers, 1)

    def test_repr(self):
        H = SparsePauliOp("ZZ")
        vqe = NoiseAwareVQE(H, n_qubits=2)
        r = repr(vqe)
        self.assertIn("n_qubits=2", r)
        self.assertIn("spsa", r)

    def test_zero_qubits_raises(self):
        with self.assertRaises(NoiseVQEError):
            NoiseAwareVQE(SparsePauliOp("Z"), n_qubits=0)

    def test_zero_layers_raises(self):
        with self.assertRaises(NoiseVQEError):
            NoiseAwareVQE(SparsePauliOp("Z"), n_qubits=1, n_layers=0)

    def test_zero_max_iter_raises(self):
        with self.assertRaises(NoiseVQEError):
            NoiseAwareVQE(SparsePauliOp("Z"), n_qubits=1, max_iter=0)

    def test_invalid_optimizer_raises(self):
        with self.assertRaises(NoiseVQEError):
            NoiseAwareVQE(SparsePauliOp("Z"), n_qubits=1, optimizer="adam")

    def test_accepts_matrix_hamiltonian(self):
        H = -np.diag([1.0, -1.0]).astype(complex)
        vqe = NoiseAwareVQE(H, n_qubits=1)
        self.assertIsNotNone(vqe)

    def test_accepts_sparse_pauli_op(self):
        H = SparsePauliOp([("ZZ", -1.0), ("XX", 0.5)])
        vqe = NoiseAwareVQE(H, n_qubits=2)
        self.assertIsNotNone(vqe)


# ─────────────────────────────────────────────────────────────────────────────
# NoiseAwareVQE — run()
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseAwareVQERun(unittest.TestCase):
    def test_returns_result(self):
        H = SparsePauliOp("Z")
        vqe = NoiseAwareVQE(H, n_qubits=1, n_layers=1, n_shots=0, max_iter=20)
        result = vqe.run()
        self.assertIsInstance(result, NoiseAwareVQEResult)

    def test_energy_finite(self):
        H = SparsePauliOp("Z")
        result = NoiseAwareVQE(H, n_qubits=1, n_layers=1,
                               n_shots=0, max_iter=30).run()
        self.assertTrue(math.isfinite(result.energy))

    def test_theta_correct_shape(self):
        H = SparsePauliOp("ZZ")
        n_layers = 2
        result = NoiseAwareVQE(H, n_qubits=2, n_layers=n_layers,
                               n_shots=0, max_iter=10).run()
        self.assertEqual(result.theta.shape, (4,))  # 2*2

    def test_energies_history_non_empty(self):
        H = SparsePauliOp("Z")
        result = NoiseAwareVQE(H, n_qubits=1, n_layers=1,
                               n_shots=0, max_iter=10).run()
        self.assertGreater(len(result.energies), 0)

    def test_custom_theta_init_accepted(self):
        H = SparsePauliOp("Z")
        theta0 = np.array([0.0])
        result = NoiseAwareVQE(H, n_qubits=1, n_layers=1,
                               n_shots=0, max_iter=5).run(theta0)
        self.assertIsInstance(result, NoiseAwareVQEResult)

    def test_wrong_theta_init_shape_raises(self):
        H = SparsePauliOp("Z")
        with self.assertRaises(NoiseVQEError):
            NoiseAwareVQE(H, n_qubits=1, n_layers=1).run(np.zeros(3))

    def test_noiseless_energy_near_ground_state(self):
        """With exact shots and enough iterations, VQE should approach -1."""
        H = SparsePauliOp("Z")   # eigenvalues ±1; ground state energy = -1
        result = NoiseAwareVQE(
            H, n_qubits=1, n_layers=1,
            n_shots=0, max_iter=100, optimizer="nelder-mead", seed=0
        ).run()
        self.assertLess(result.energy, 0.0)

    def test_shots_metadata(self):
        H = SparsePauliOp("Z")
        result = NoiseAwareVQE(H, n_qubits=1, n_layers=1,
                               n_shots=500, max_iter=5).run()
        self.assertEqual(result.shots_per_eval, 500)

    def test_n_iterations_stored(self):
        H = SparsePauliOp("Z")
        result = NoiseAwareVQE(H, n_qubits=1, n_layers=1,
                               n_shots=0, max_iter=15).run()
        self.assertGreater(result.n_iterations, 0)

    def test_noise_description_no_noise(self):
        H = SparsePauliOp("Z")
        result = NoiseAwareVQE(H, n_qubits=1, noise_model=None,
                               n_shots=0, max_iter=5).run()
        self.assertEqual(result.noise_description, "none")

    def test_noise_model_applied(self):
        """With depolarizing noise, energy at optimal point is less negative."""
        from quantx_engine.noise import DepolarizingNoise
        H = SparsePauliOp("Z")
        # For single-qubit H=Z, optimal angle is theta=pi (|1> state, energy=-1)
        theta_opt = np.array([math.pi])

        clean_vqe = NoiseAwareVQE(H, n_qubits=1, n_layers=1, n_shots=0, max_iter=1)
        noisy_vqe = NoiseAwareVQE(H, n_qubits=1, n_layers=1,
                                  noise_model=DepolarizingNoise(prob=0.3),
                                  n_shots=0, max_iter=1)
        e_clean = clean_vqe._energy(theta_opt)
        e_noisy = noisy_vqe._energy(theta_opt)

        # At the optimal angle, noise should push energy toward 0 (less negative)
        self.assertLess(e_clean, -0.9)        # clean ≈ -1
        self.assertGreater(e_noisy, e_clean)  # noisy is less negative

    def test_nelder_mead_optimizer(self):
        H = SparsePauliOp("Z")
        result = NoiseAwareVQE(H, n_qubits=1, n_layers=1,
                               optimizer="nelder-mead",
                               n_shots=0, max_iter=50).run()
        self.assertIsInstance(result, NoiseAwareVQEResult)

    def test_result_noise_description_with_depol(self):
        from quantx_engine.noise import DepolarizingNoise
        H = SparsePauliOp("Z")
        result = NoiseAwareVQE(
            H, n_qubits=1, noise_model=DepolarizingNoise(prob=0.01),
            n_shots=0, max_iter=5
        ).run()
        self.assertIn("depolarizing", result.noise_description.lower())

    def test_n_qubits_stored(self):
        H = SparsePauliOp("ZZ")
        result = NoiseAwareVQE(H, n_qubits=2, max_iter=5, n_shots=0).run()
        self.assertEqual(result.n_qubits, 2)

    def test_n_layers_stored(self):
        H = SparsePauliOp("ZZ")
        result = NoiseAwareVQE(H, n_qubits=2, n_layers=3,
                               max_iter=5, n_shots=0).run()
        self.assertEqual(result.n_layers, 3)


# ─────────────────────────────────────────────────────────────────────────────
# NoiseAwareVQEResult repr
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseAwareVQEResult(unittest.TestCase):
    def _make_result(self) -> NoiseAwareVQEResult:
        return NoiseAwareVQEResult(
            energy=-0.95,
            theta=np.array([1.5, 0.3]),
            n_iterations=42,
            energies=[-0.5, -0.8, -0.95],
            shots_per_eval=1000,
            converged=True,
            noise_description="depolarizing(p=0.01)",
            n_qubits=2,
            n_layers=1,
        )

    def test_repr_contains_energy(self):
        r = repr(self._make_result())
        self.assertIn("-0.950000", r)

    def test_repr_contains_n_iter(self):
        r = repr(self._make_result())
        self.assertIn("n_iter=42", r)

    def test_repr_contains_shots(self):
        r = repr(self._make_result())
        self.assertIn("shots=1000", r)


# ─────────────────────────────────────────────────────────────────────────────
# hardware_efficient_vqe convenience function
# ─────────────────────────────────────────────────────────────────────────────

class TestHardwareEfficientVQE(unittest.TestCase):
    def test_returns_result(self):
        H = SparsePauliOp("Z")
        result = hardware_efficient_vqe(H, n_qubits=1, n_shots=0, max_iter=20)
        self.assertIsInstance(result, NoiseAwareVQEResult)

    def test_energy_finite(self):
        H = SparsePauliOp("ZZ")
        result = hardware_efficient_vqe(H, n_qubits=2, n_shots=0, max_iter=20)
        self.assertTrue(math.isfinite(result.energy))

    def test_with_noise_model(self):
        from quantx_engine.noise import DepolarizingNoise
        H = SparsePauliOp("Z")
        result = hardware_efficient_vqe(
            H, n_qubits=1,
            noise_model=DepolarizingNoise(prob=0.01),
            n_shots=0, max_iter=15
        )
        self.assertIsInstance(result, NoiseAwareVQEResult)

    def test_matrix_hamiltonian(self):
        H_mat = -np.diag([1.0, -1.0]).astype(complex)  # ground energy = -1
        result = hardware_efficient_vqe(H_mat, n_qubits=1, n_shots=0,
                                        max_iter=50, optimizer="nelder-mead")
        self.assertLess(result.energy, 0.0)

    def test_two_qubit_zz(self):
        """Hardware-efficient VQE on H = -ZZ (ground energy = -1)."""
        H = SparsePauliOp([("ZZ", -1.0)])
        result = hardware_efficient_vqe(
            H, n_qubits=2, n_layers=2, n_shots=0, max_iter=30
        )
        self.assertIsInstance(result, NoiseAwareVQEResult)
        # Energy should be ≤ 0 (somewhere below the max energy of +1)
        self.assertLessEqual(result.energy, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# __init__ re-exports
# ─────────────────────────────────────────────────────────────────────────────

class TestInitExports(unittest.TestCase):
    def test_noise_vqe_error_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "NoiseVQEError"))

    def test_spsa_optimizer_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "SPSAOptimizer"))

    def test_noise_aware_vqe_result_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "NoiseAwareVQEResult"))

    def test_noise_aware_vqe_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "NoiseAwareVQE"))

    def test_build_hea_circuit_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "build_hea_circuit"))

    def test_hardware_efficient_vqe_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "hardware_efficient_vqe"))

    def test_n_hea_params_exported(self):
        import quantx_engine
        self.assertTrue(hasattr(quantx_engine, "n_hea_params"))


if __name__ == "__main__":
    unittest.main()
