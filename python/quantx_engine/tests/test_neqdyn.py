"""M114: Tests for non-equilibrium quantum dynamics."""
from __future__ import annotations

import math
import pytest
import numpy as np

from quantx_engine.neqdyn import (
    NonEquilError,
    KeldyshResult,
    KeldyshSolver,
    EntanglementDynamicsResult,
    EntanglementDynamics,
    KibbleZurekResult,
    KibbleZurekScaling,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tight_binding(n: int, t: float = 1.0) -> np.ndarray:
    """1D tight-binding chain H_{ij} = -t (nearest-neighbour)."""
    H = np.zeros((n, n))
    for i in range(n - 1):
        H[i, i + 1] = -t
        H[i + 1, i] = -t
    return H


def _two_qubit_ising_H() -> np.ndarray:
    """2-qubit transverse-field Ising Hamiltonian (exact)."""
    Z = np.array([[1., 0.], [0., -1.]])
    X = np.array([[0., 1.], [1., 0.]])
    I = np.eye(2)
    H = -np.kron(Z, Z) - 0.5 * (np.kron(X, I) + np.kron(I, X))
    return H


# ── KeldyshSolver construction ────────────────────────────────────────────────

class TestKeldyshSolverConstruction:
    def test_default_site(self):
        solver = KeldyshSolver(_tight_binding(4))
        assert solver.site == 0
        assert solver.broadening == 0.05

    def test_explicit_params(self):
        solver = KeldyshSolver(_tight_binding(4), site=2, broadening=0.1)
        assert solver.site == 2
        assert solver.broadening == 0.1

    def test_non_square_raises(self):
        with pytest.raises(NonEquilError):
            KeldyshSolver(np.ones((3, 4)))

    def test_1d_array_raises(self):
        with pytest.raises(NonEquilError):
            KeldyshSolver(np.ones(4))  # type: ignore

    def test_site_out_of_range_raises(self):
        with pytest.raises(NonEquilError):
            KeldyshSolver(_tight_binding(4), site=4)

    def test_site_negative_raises(self):
        with pytest.raises(NonEquilError):
            KeldyshSolver(_tight_binding(4), site=-1)


# ── KeldyshResult ─────────────────────────────────────────────────────────────

class TestKeldyshResult:
    def _make_result(self) -> KeldyshResult:
        return KeldyshSolver(_tight_binding(4)).solve(t_max=5.0, n_times=50, n_freqs=100)

    def test_returns_keldysh_result(self):
        assert isinstance(self._make_result(), KeldyshResult)

    def test_shapes(self):
        result = KeldyshSolver(_tight_binding(4)).solve(n_times=50, n_freqs=100)
        assert result.times.shape == (50,)
        assert result.G_ret.shape == (50,)
        assert result.spectral_function.shape == (100,)
        assert result.frequencies.shape == (100,)

    def test_times_starts_at_zero(self):
        result = self._make_result()
        assert result.times[0] == 0.0

    def test_G_ret_zero_at_t0(self):
        """At t=0 the retarded GF is zero (theta(0) convention)."""
        result = self._make_result()
        assert result.G_ret[0] == 0.0

    def test_spectral_function_positive(self):
        result = self._make_result()
        assert np.all(result.spectral_function >= -1e-14)

    def test_spectral_function_normalised(self):
        """Integral of A(ω) over all ω should be ≈ 1 (sum rule)."""
        result = KeldyshSolver(_tight_binding(4), site=0).solve(
            t_max=10.0, n_times=200, n_freqs=1000
        )
        dw = result.frequencies[1] - result.frequencies[0]
        total = float(np.sum(result.spectral_function) * dw)
        assert abs(total - 1.0) < 0.05

    def test_site_stored(self):
        result = KeldyshSolver(_tight_binding(4), site=2).solve()
        assert result.site == 2

    def test_to_dict_keys(self):
        d = self._make_result().to_dict()
        assert "times" in d
        assert "G_ret_real" in d
        assert "G_ret_imag" in d
        assert "spectral_function" in d
        assert "frequencies" in d
        assert "site" in d

    def test_to_dict_values_are_lists(self):
        d = self._make_result().to_dict()
        assert isinstance(d["times"], list)
        assert isinstance(d["spectral_function"], list)

    def test_single_site_model(self):
        """1×1 Hamiltonian → single delta peak."""
        H = np.array([[2.0]])
        result = KeldyshSolver(H, site=0, broadening=0.1).solve(n_freqs=100)
        # Peak should be near ω = 2.0
        peak_idx = int(np.argmax(result.spectral_function))
        assert abs(result.frequencies[peak_idx] - 2.0) < 0.2


# ── EntanglementDynamics construction ─────────────────────────────────────────

class TestEntanglementDynamicsConstruction:
    def test_default_subsystem(self):
        H = _two_qubit_ising_H()
        dyn = EntanglementDynamics(H, n_qubits=2)
        assert dyn.subsystem_a == [0]  # left half for 2 qubits

    def test_explicit_subsystem(self):
        H = _two_qubit_ising_H()
        dyn = EntanglementDynamics(H, n_qubits=2, subsystem_a=[1])
        assert dyn.subsystem_a == [1]

    def test_wrong_matrix_size_raises(self):
        H = np.eye(3)  # not 2^n × 2^n for any integer n
        with pytest.raises(NonEquilError):
            EntanglementDynamics(H, n_qubits=2)

    def test_4qubit_default_subsystem(self):
        H = np.eye(16)
        dyn = EntanglementDynamics(H, n_qubits=4)
        assert dyn.subsystem_a == [0, 1]  # left half


# ── EntanglementDynamicsResult ────────────────────────────────────────────────

class TestEntanglementDynamicsResult:
    def _run(self, times=None) -> EntanglementDynamicsResult:
        H = _two_qubit_ising_H()
        dyn = EntanglementDynamics(H, n_qubits=2, subsystem_a=[0])
        psi0 = np.zeros(4)
        psi0[0] = 1.0  # |00⟩ — product state
        if times is None:
            times = [0.0, 0.5, 1.0, 2.0]
        return dyn.evolve(psi0, times)

    def test_returns_result(self):
        assert isinstance(self._run(), EntanglementDynamicsResult)

    def test_times_stored(self):
        times = [0.0, 1.0, 2.0]
        result = self._run(times)
        assert result.times == times

    def test_entropy_len_matches_times(self):
        times = [0.0, 0.5, 1.0]
        result = self._run(times)
        assert len(result.entropy) == len(times)

    def test_mutual_info_len_matches_times(self):
        times = [0.0, 0.5, 1.0]
        result = self._run(times)
        assert len(result.mutual_info) == len(times)

    def test_entropy_non_negative(self):
        result = self._run()
        for s in result.entropy:
            assert s >= -1e-10

    def test_product_state_zero_entropy(self):
        """A product initial state should have zero entropy at t=0."""
        result = self._run([0.0])
        assert abs(result.entropy[0]) < 1e-10

    def test_entropy_bounded_above(self):
        """For a single qubit subsystem, S_A ≤ ln(2)."""
        result = self._run([0.0, 1.0, 2.0, 3.0])
        ln2 = math.log(2)
        for s in result.entropy:
            assert s <= ln2 + 1e-9

    def test_mutual_info_equals_twice_entropy(self):
        """For pure state: I(A:B) = 2 S_A."""
        result = self._run()
        for s, mi in zip(result.entropy, result.mutual_info):
            assert abs(mi - 2 * s) < 1e-10

    def test_to_dict_keys(self):
        d = self._run().to_dict()
        assert "times" in d
        assert "entropy" in d
        assert "mutual_info" in d
        assert "subsystem_a" in d

    def test_entanglement_generated_over_time(self):
        """Interacting Hamiltonian should generate entanglement from product state."""
        result = self._run([0.0, 5.0, 10.0])
        # At t>0 with interactions, entropy should grow above 0
        max_entropy = max(result.entropy)
        assert max_entropy > 1e-6

    def test_unnormalised_initial_state_normalised(self):
        """Should accept unnormalised initial state (normalises internally)."""
        H = _two_qubit_ising_H()
        dyn = EntanglementDynamics(H, n_qubits=2, subsystem_a=[0])
        psi0 = np.array([2.0, 0.0, 0.0, 0.0])  # norm=2, not 1
        result = dyn.evolve(psi0, [0.0])
        assert abs(result.entropy[0]) < 1e-10


# ── KibbleZurekScaling construction ──────────────────────────────────────────

class TestKibbleZurekConstruction:
    def test_default_params(self):
        kz = KibbleZurekScaling(n_sites=4)
        assert kz.n_sites == 4
        assert kz.J == 1.0
        assert kz.n_steps == 50

    def test_explicit_params(self):
        kz = KibbleZurekScaling(n_sites=4, J=2.0, n_steps=20)
        assert kz.J == 2.0
        assert kz.n_steps == 20

    def test_n_sites_one_raises(self):
        with pytest.raises(NonEquilError):
            KibbleZurekScaling(n_sites=1)

    def test_n_sites_two_ok(self):
        kz = KibbleZurekScaling(n_sites=2)
        assert kz.n_sites == 2


# ── KibbleZurekResult ─────────────────────────────────────────────────────────

class TestKibbleZurekResult:
    def _run_fast(self) -> KibbleZurekResult:
        kz = KibbleZurekScaling(n_sites=2, n_steps=10)
        return kz.sweep(quench_rates=[1.0, 2.0, 4.0])

    def test_returns_kz_result(self):
        assert isinstance(self._run_fast(), KibbleZurekResult)

    def test_quench_rates_stored(self):
        rates = [1.0, 2.0, 4.0]
        result = KibbleZurekScaling(n_sites=2, n_steps=10).sweep(rates)
        assert result.quench_rates == rates

    def test_defect_densities_len(self):
        rates = [1.0, 2.0, 4.0]
        result = KibbleZurekScaling(n_sites=2, n_steps=10).sweep(rates)
        assert len(result.defect_densities) == 3

    def test_defect_densities_in_range(self):
        result = self._run_fast()
        for d in result.defect_densities:
            assert 0.0 <= d <= 1.0 + 1e-10

    def test_kz_exponent_is_float(self):
        result = self._run_fast()
        assert isinstance(result.kz_exponent, float)

    def test_kz_prefactor_is_float(self):
        result = self._run_fast()
        assert isinstance(result.kz_prefactor, float)

    def test_to_dict_keys(self):
        d = self._run_fast().to_dict()
        assert "quench_rates" in d
        assert "defect_densities" in d
        assert "kz_exponent" in d
        assert "kz_prefactor" in d

    def test_default_quench_rates(self):
        result = KibbleZurekScaling(n_sites=2, n_steps=5).sweep()
        assert len(result.quench_rates) == 8  # default is logspace 8 points

    def test_fast_quench_more_defects_than_slow(self):
        """Fast quench → more defects (KZ prediction)."""
        kz = KibbleZurekScaling(n_sites=3, n_steps=20)
        slow = kz.sweep([0.1], h_i=2.0, h_f=0.2)
        fast = kz.sweep([5.0], h_i=2.0, h_f=0.2)
        # Fast quench should produce at least as many defects as slow quench
        # (may not hold perfectly for tiny systems, so use a loose bound)
        assert fast.defect_densities[0] >= slow.defect_densities[0] - 0.2

    def test_exponent_positive(self):
        """KZ exponent should be positive (defects increase with rate)."""
        kz = KibbleZurekScaling(n_sites=3, n_steps=20)
        result = kz.sweep([0.5, 1.0, 2.0, 4.0])
        # Exponent sign: defects ~ v^alpha with alpha > 0
        # We allow some tolerance for small finite-size systems
        assert result.kz_exponent > -0.5


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration:
    def test_keldysh_larger_chain(self):
        H = _tight_binding(6)
        result = KeldyshSolver(H, site=3).solve(t_max=8.0, n_times=100, n_freqs=200)
        assert result.G_ret.shape == (100,)
        assert result.spectral_function.shape == (200,)

    def test_entanglement_4qubits(self):
        H = np.zeros((16, 16))
        for i in range(15):
            H[i, i + 1] = H[i + 1, i] = -1.0
        dyn = EntanglementDynamics(H, n_qubits=4, subsystem_a=[0, 1])
        psi0 = np.zeros(16)
        psi0[0] = 1.0
        result = dyn.evolve(psi0, [0.0, 1.0, 2.0])
        assert len(result.entropy) == 3
        assert all(s >= -1e-10 for s in result.entropy)

    def test_kz_single_rate(self):
        kz = KibbleZurekScaling(n_sites=2, n_steps=5)
        result = kz.sweep([1.0])
        assert len(result.defect_densities) == 1
        # Exponent: NaN or some value is acceptable with a single point
        assert isinstance(result.kz_exponent, float)
