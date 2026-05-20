"""M115: Tests for CVaR-QAOA combinatorial optimisation."""
from __future__ import annotations

import math
import pytest
import numpy as np

from quantx_engine.qaoa_optim import (
    CVaRQAOAError,
    QAOAResult,
    MaxCutProblem,
    TSPProblem,
    CVaRQAOA,
    QAOALandscapeResult,
    QAOALandscape,
    warm_start_qaoa,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _triangle_graph() -> MaxCutProblem:
    """3-node triangle graph; max-cut = 2."""
    return MaxCutProblem(n_nodes=3, edges=[(0, 1), (1, 2), (0, 2)])


def _simple_2node() -> MaxCutProblem:
    """2-node graph with one edge; max-cut = 1."""
    return MaxCutProblem(n_nodes=2, edges=[(0, 1)])


def _sym_dist(n: int) -> np.ndarray:
    """Symmetric distance matrix for n cities (integer distances)."""
    rng = np.random.default_rng(42)
    d = rng.integers(1, 10, (n, n)).astype(float)
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0.0)
    return d


# ── MaxCutProblem construction ────────────────────────────────────────────────

class TestMaxCutConstruction:
    def test_default_init(self):
        mc = _triangle_graph()
        assert mc.n_nodes == 3
        assert len(mc.edges) == 3

    def test_weighted_edges(self):
        mc = MaxCutProblem(n_nodes=3, edges=[(0, 1, 2.5), (1, 2, 1.0)])
        assert mc.edges[0][2] == 2.5

    def test_n_nodes_zero_raises(self):
        with pytest.raises(CVaRQAOAError):
            MaxCutProblem(n_nodes=0, edges=[])

    def test_edge_out_of_range_raises(self):
        with pytest.raises(CVaRQAOAError):
            MaxCutProblem(n_nodes=2, edges=[(0, 5)])

    def test_empty_edges_ok(self):
        mc = MaxCutProblem(n_nodes=4, edges=[])
        assert mc.n_nodes == 4


class TestMaxCutMethods:
    def test_cut_value_basic(self):
        mc = _simple_2node()
        # bitstring 01 (node 0 in set 0, node 1 in set 1) → cut = 1
        assert mc.cut_value(0b01) == 1.0
        assert mc.cut_value(0b10) == 1.0
        # 00 and 11 → cut = 0
        assert mc.cut_value(0b00) == 0.0
        assert mc.cut_value(0b11) == 0.0

    def test_exact_max_cut_2node(self):
        mc = _simple_2node()
        val, _ = mc.exact_max_cut()
        assert val == 1.0

    def test_exact_max_cut_triangle(self):
        mc = _triangle_graph()
        val, _ = mc.exact_max_cut()
        assert val == 2.0

    def test_cost_diagonal_shape(self):
        mc = _triangle_graph()
        diag = mc.cost_diagonal()
        assert diag.shape == (8,)  # 2^3

    def test_cost_diagonal_is_negative_cut(self):
        mc = _simple_2node()
        diag = mc.cost_diagonal()
        # diag[x] = -cut_value(x)
        for x in range(4):
            assert abs(diag[x] + mc.cut_value(x)) < 1e-12

    def test_approximation_ratio_optimal(self):
        mc = _simple_2node()
        _, best = mc.exact_max_cut()
        ratio = mc.approximation_ratio(best)
        assert abs(ratio - 1.0) < 1e-10

    def test_approximation_ratio_suboptimal(self):
        mc = _triangle_graph()
        ratio = mc.approximation_ratio(0b000)  # cut = 0
        assert ratio == 0.0

    def test_cost_hamiltonian_returns_hamiltonian(self):
        from quantx_engine.chemistry import PauliHamiltonian
        mc = _triangle_graph()
        ham = mc.cost_hamiltonian()
        assert isinstance(ham, PauliHamiltonian)
        assert ham.n_qubits == 3

    def test_no_edges_cut_always_zero(self):
        mc = MaxCutProblem(n_nodes=3, edges=[])
        for x in range(8):
            assert mc.cut_value(x) == 0.0


# ── TSPProblem construction ───────────────────────────────────────────────────

class TestTSPConstruction:
    def test_basic_init(self):
        d = _sym_dist(3)
        tsp = TSPProblem(d)
        assert tsp.n_cities == 3
        assert tsp.n_qubits == 9

    def test_non_square_raises(self):
        with pytest.raises(CVaRQAOAError):
            TSPProblem(np.ones((3, 4)))

    def test_single_city_raises(self):
        with pytest.raises(CVaRQAOAError):
            TSPProblem(np.array([[0.0]]))

    def test_default_penalty(self):
        tsp = TSPProblem(_sym_dist(2))
        assert tsp.penalty == 10.0


class TestTSPMethods:
    def test_tour_length_basic(self):
        d = np.array([[0., 2., 9.], [2., 0., 6.], [9., 6., 0.]])
        tsp = TSPProblem(d)
        # Tour 0→1→2→0: 2 + 6 + 9 = 17
        assert abs(tsp.tour_length([0, 1, 2]) - 17.0) < 1e-10

    def test_tour_length_invalid_raises(self):
        tsp = TSPProblem(_sym_dist(3))
        with pytest.raises(CVaRQAOAError):
            tsp.tour_length([0, 0, 1])  # not a permutation

    def test_decode_bitstring_valid(self):
        d = _sym_dist(2)
        tsp = TSPProblem(d)
        # Valid: city 0 at pos 0, city 1 at pos 1 → x_{0,0}=1, x_{1,1}=1
        # qubit(city, pos) = city * 2 + pos
        # qubit(0,0)=0, qubit(0,1)=1, qubit(1,0)=2, qubit(1,1)=3
        # x_{0,0}=1 (bit 0), x_{1,1}=1 (bit 3) → bitstring = 0b1001 = 9
        x = (1 << 0) | (1 << 3)  # bit 0 and bit 3 set
        tour = tsp.decode_bitstring(x)
        assert tour is not None
        assert len(tour) == 2
        assert set(tour) == {0, 1}

    def test_decode_bitstring_invalid(self):
        tsp = TSPProblem(_sym_dist(2))
        # All zeros → no city at any position → invalid
        tour = tsp.decode_bitstring(0)
        assert tour is None

    def test_cost_diagonal_shape(self):
        tsp = TSPProblem(_sym_dist(2))
        diag = tsp.cost_diagonal()
        assert diag.shape == (1 << tsp.n_qubits,)

    def test_cost_diagonal_penalty_for_zero(self):
        """All-zero bitstring violates all constraints → high cost."""
        tsp = TSPProblem(_sym_dist(2), penalty=10.0)
        diag = tsp.cost_diagonal()
        assert diag[0] > 0.0  # should have penalty

    def test_cost_hamiltonian(self):
        from quantx_engine.chemistry import PauliHamiltonian
        tsp = TSPProblem(_sym_dist(2))
        ham = tsp.cost_hamiltonian()
        assert isinstance(ham, PauliHamiltonian)
        assert ham.n_qubits == 4


# ── CVaRQAOA construction ─────────────────────────────────────────────────────

class TestCVaRQAOAConstruction:
    def _diag(self) -> np.ndarray:
        mc = _simple_2node()
        return mc.cost_diagonal()

    def test_default_params(self):
        qaoa = CVaRQAOA(self._diag(), n_qubits=2)
        assert qaoa.n_layers == 1
        assert qaoa.alpha == 0.25

    def test_explicit_params(self):
        qaoa = CVaRQAOA(self._diag(), n_qubits=2, n_layers=2, alpha=0.5)
        assert qaoa.n_layers == 2
        assert qaoa.alpha == 0.5

    def test_wrong_diag_shape_raises(self):
        with pytest.raises(CVaRQAOAError):
            CVaRQAOA(np.array([1.0, 2.0, 3.0]), n_qubits=2)

    def test_n_layers_zero_raises(self):
        with pytest.raises(CVaRQAOAError):
            CVaRQAOA(self._diag(), n_qubits=2, n_layers=0)

    def test_alpha_zero_raises(self):
        with pytest.raises(CVaRQAOAError):
            CVaRQAOA(self._diag(), n_qubits=2, alpha=0.0)

    def test_alpha_gt_1_raises(self):
        with pytest.raises(CVaRQAOAError):
            CVaRQAOA(self._diag(), n_qubits=2, alpha=1.1)

    def test_alpha_1_is_standard_qaoa(self):
        qaoa = CVaRQAOA(self._diag(), n_qubits=2, alpha=1.0)
        assert qaoa.alpha == 1.0


# ── CVaRQAOA objective ────────────────────────────────────────────────────────

class TestCVaRObjective:
    def _make_qaoa(self, alpha=0.25) -> CVaRQAOA:
        mc = _triangle_graph()
        return CVaRQAOA(mc.cost_diagonal(), n_qubits=3, alpha=alpha)

    def test_objective_returns_float(self):
        qaoa = self._make_qaoa()
        val = qaoa.objective(np.array([0.5, 0.3]))
        assert isinstance(val, float)

    def test_expectation_returns_float(self):
        qaoa = self._make_qaoa()
        val = qaoa.expectation(np.array([0.5, 0.3]))
        assert isinstance(val, float)

    def test_cvar_leq_expectation(self):
        """CVaR_α ≤ ⟨H⟩ for minimisation (lower = better, CVaR focuses on best)."""
        qaoa = self._make_qaoa(alpha=0.1)
        params = np.array([0.5, 0.3])
        assert qaoa.objective(params) <= qaoa.expectation(params) + 1e-10

    def test_cvar_alpha1_equals_expectation(self):
        """CVaR with alpha=1 equals standard expectation value."""
        qaoa = self._make_qaoa(alpha=1.0)
        params = np.array([0.5, 0.3])
        assert abs(qaoa.objective(params) - qaoa.expectation(params)) < 1e-10

    def test_objective_bounded(self):
        """CVaR must be between min and max cost."""
        mc = _triangle_graph()
        diag = mc.cost_diagonal()
        qaoa = CVaRQAOA(diag, n_qubits=3, alpha=0.25)
        val = qaoa.objective(np.array([1.0, 0.5]))
        assert diag.min() - 1e-10 <= val <= diag.max() + 1e-10

    def test_deeper_layers(self):
        """p=2 QAOA objective should be computable."""
        mc = _simple_2node()
        qaoa = CVaRQAOA(mc.cost_diagonal(), n_qubits=2, n_layers=2)
        params = np.array([0.5, 0.7, 0.3, 0.2])  # [γ1, γ2, β1, β2]
        val = qaoa.objective(params)
        assert isinstance(val, float)


# ── CVaRQAOA optimize ─────────────────────────────────────────────────────────

class TestCVaROptimize:
    def test_optimize_returns_qaoa_result(self):
        mc = _simple_2node()
        qaoa = CVaRQAOA(mc.cost_diagonal(), n_qubits=2, alpha=0.5,
                        exact_optimum=-1.0)
        result = qaoa.optimize(seed=0)
        assert isinstance(result, QAOAResult)

    def test_result_fields(self):
        mc = _simple_2node()
        qaoa = CVaRQAOA(mc.cost_diagonal(), n_qubits=2, alpha=0.5)
        result = qaoa.optimize(seed=42)
        assert result.n_layers == 1
        assert result.alpha == 0.5
        assert result.n_qubits == 2
        assert result.optimal_params.shape == (2,)

    def test_optimal_value_leq_initial(self):
        """Optimised value should be ≤ initial value (minimisation)."""
        mc = _triangle_graph()
        qaoa = CVaRQAOA(mc.cost_diagonal(), n_qubits=3, alpha=0.3)
        init_params = np.array([0.1, 0.2])
        init_val = qaoa.objective(init_params)
        result = qaoa.optimize(initial_params=init_params, seed=0)
        # Should be at most a tiny bit worse than initial (allow numerical noise)
        assert result.optimal_value <= init_val + 0.1

    def test_approximation_ratio_with_exact(self):
        mc = _simple_2node()
        exact_val, _ = mc.exact_max_cut()
        # exact_optimum is -exact_val (we minimise negative cut)
        qaoa = CVaRQAOA(mc.cost_diagonal(), n_qubits=2, alpha=1.0,
                        exact_optimum=-exact_val)
        result = qaoa.optimize(seed=0)
        assert result.approximation_ratio is not None

    def test_approximation_ratio_none_without_exact(self):
        mc = _simple_2node()
        qaoa = CVaRQAOA(mc.cost_diagonal(), n_qubits=2)
        result = qaoa.optimize(seed=0)
        assert result.approximation_ratio is None

    def test_to_dict_keys(self):
        mc = _simple_2node()
        qaoa = CVaRQAOA(mc.cost_diagonal(), n_qubits=2)
        d = qaoa.optimize(seed=0).to_dict()
        assert "optimal_params" in d
        assert "optimal_value" in d
        assert "expectation_value" in d
        assert "approximation_ratio" in d
        assert "n_layers" in d

    def test_reproducible_with_seed(self):
        mc = _triangle_graph()
        qaoa = CVaRQAOA(mc.cost_diagonal(), n_qubits=3, alpha=0.3)
        r1 = qaoa.optimize(seed=7)
        r2 = qaoa.optimize(seed=7)
        np.testing.assert_array_equal(r1.optimal_params, r2.optimal_params)

    def test_optimize_finds_low_energy(self):
        """For MaxCut with 1 edge, QAOA should find expectation < 0 (near -1)."""
        mc = _simple_2node()
        qaoa = CVaRQAOA(mc.cost_diagonal(), n_qubits=2, alpha=1.0)
        result = qaoa.optimize(seed=0)
        # Max-cut value = 1; cost_diag = -cut, so expectation should be < 0
        assert result.expectation_value < 0.0


# ── QAOALandscape ─────────────────────────────────────────────────────────────

class TestQAOALandscape:
    def _make_landscape(self) -> QAOALandscape:
        mc = _simple_2node()
        return QAOALandscape(mc.cost_diagonal(), n_qubits=2, alpha=0.3)

    def test_compute_returns_result(self):
        result = self._make_landscape().compute(n_gamma=5, n_beta=5)
        assert isinstance(result, QAOALandscapeResult)

    def test_surface_shapes(self):
        result = self._make_landscape().compute(n_gamma=8, n_beta=6)
        assert result.energy_surface.shape == (8, 6)
        assert result.cvar_surface.shape == (8, 6)
        assert result.gamma_values.shape == (8,)
        assert result.beta_values.shape == (6,)

    def test_energy_surface_bounded(self):
        mc = _simple_2node()
        diag = mc.cost_diagonal()
        landscape = QAOALandscape(diag, n_qubits=2)
        result = landscape.compute(n_gamma=5, n_beta=5)
        assert np.all(result.energy_surface >= diag.min() - 1e-10)
        assert np.all(result.energy_surface <= diag.max() + 1e-10)

    def test_cvar_leq_energy(self):
        """CVaR_α ≤ ⟨H⟩ pointwise (CVaR picks best-α outcomes)."""
        mc = _simple_2node()
        landscape = QAOALandscape(mc.cost_diagonal(), n_qubits=2, alpha=0.2)
        result = landscape.compute(n_gamma=4, n_beta=4)
        assert np.all(result.cvar_surface <= result.energy_surface + 1e-10)

    def test_optimal_gamma_beta_in_range(self):
        result = self._make_landscape().compute(
            gamma_range=(0.0, math.pi), beta_range=(0.0, math.pi / 2),
            n_gamma=10, n_beta=10
        )
        assert 0.0 <= result.optimal_gamma <= math.pi + 1e-10
        assert 0.0 <= result.optimal_beta <= math.pi / 2 + 1e-10

    def test_alpha_stored(self):
        landscape = QAOALandscape(np.zeros(4), n_qubits=2, alpha=0.15)
        result = landscape.compute(n_gamma=3, n_beta=3)
        assert result.alpha == 0.15

    def test_to_dict_keys(self):
        d = self._make_landscape().compute(n_gamma=3, n_beta=3).to_dict()
        assert "gamma_values" in d
        assert "beta_values" in d
        assert "energy_surface" in d
        assert "cvar_surface" in d
        assert "optimal_gamma" in d
        assert "optimal_beta" in d

    def test_uniform_cost_flat_landscape(self):
        """Uniform cost → energy surface is constant."""
        n = 2
        diag = np.zeros(1 << n)  # all costs = 0
        landscape = QAOALandscape(diag, n_qubits=n)
        result = landscape.compute(n_gamma=4, n_beta=4)
        assert np.allclose(result.energy_surface, 0.0)


# ── warm_start_qaoa ───────────────────────────────────────────────────────────

class TestWarmStart:
    def test_random_strategy(self):
        params = warm_start_qaoa(n_layers=3, strategy="random", seed=0)
        assert params.shape == (6,)

    def test_random_in_range(self):
        params = warm_start_qaoa(n_layers=4, strategy="random", seed=1)
        assert np.all(params >= 0)
        assert np.all(params <= math.pi + 1e-10)

    def test_random_reproducible(self):
        p1 = warm_start_qaoa(n_layers=2, strategy="random", seed=42)
        p2 = warm_start_qaoa(n_layers=2, strategy="random", seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_fixed_strategy(self):
        params = warm_start_qaoa(n_layers=3, strategy="fixed")
        assert params.shape == (6,)

    def test_fixed_values_reasonable(self):
        params = warm_start_qaoa(n_layers=2, strategy="fixed")
        gammas, betas = params[:2], params[2:]
        assert np.all(gammas >= 0)
        assert np.all(betas >= 0)

    def test_tqa_strategy(self):
        params = warm_start_qaoa(n_layers=4, strategy="tqa")
        assert params.shape == (8,)

    def test_tqa_gammas_increasing(self):
        params = warm_start_qaoa(n_layers=4, strategy="tqa")
        gammas = params[:4]
        assert np.all(np.diff(gammas) >= -1e-10)

    def test_tqa_betas_decreasing(self):
        params = warm_start_qaoa(n_layers=4, strategy="tqa")
        betas = params[4:]
        assert np.all(np.diff(betas) <= 1e-10)

    def test_interpolated_strategy(self):
        prev = np.array([0.5, 0.3])
        params = warm_start_qaoa(n_layers=3, strategy="interpolated", prev_params=prev)
        assert params.shape == (6,)

    def test_interpolated_without_prev_raises(self):
        with pytest.raises(CVaRQAOAError):
            warm_start_qaoa(n_layers=2, strategy="interpolated")

    def test_unknown_strategy_raises(self):
        with pytest.raises(CVaRQAOAError):
            warm_start_qaoa(n_layers=2, strategy="greedy")

    def test_n_layers_zero_raises(self):
        with pytest.raises(CVaRQAOAError):
            warm_start_qaoa(n_layers=0)

    def test_single_layer(self):
        for strategy in ("random", "fixed", "tqa"):
            params = warm_start_qaoa(n_layers=1, strategy=strategy)
            assert params.shape == (2,)


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration:
    def test_maxcut_qaoa_pipeline(self):
        """Full pipeline: MaxCut → cost_diagonal → CVaRQAOA → optimise."""
        mc = _triangle_graph()
        diag = mc.cost_diagonal()
        exact, _ = mc.exact_max_cut()

        qaoa = CVaRQAOA(diag, n_qubits=3, alpha=0.5, exact_optimum=-float(exact))
        init = warm_start_qaoa(n_layers=1, strategy="fixed")
        result = qaoa.optimize(initial_params=init, seed=0)

        assert result.expectation_value <= 0.0  # minimising negative cut
        assert isinstance(result.optimal_params, np.ndarray)

    def test_landscape_minimum_consistent_with_optimize(self):
        """Landscape optimal params should give comparable energy to optimised."""
        mc = _simple_2node()
        diag = mc.cost_diagonal()

        landscape = QAOALandscape(diag, n_qubits=2)
        ls_result = landscape.compute(n_gamma=15, n_beta=15)

        opt_params = np.array([ls_result.optimal_gamma, ls_result.optimal_beta])
        qaoa = CVaRQAOA(diag, n_qubits=2, alpha=1.0)
        landscape_energy = qaoa.expectation(opt_params)

        # The landscape minimum energy should be close to
        # the minimum of the energy surface
        assert abs(landscape_energy - ls_result.energy_surface.min()) < 1e-8

    def test_tsp_2city_trivial(self):
        """2-city TSP: only 2 valid tours (0→1→0 and 1→0→1, same length)."""
        d = np.array([[0., 3.], [3., 0.]])
        tsp = TSPProblem(d)
        assert tsp.tour_length([0, 1]) == pytest.approx(6.0)
        assert tsp.tour_length([1, 0]) == pytest.approx(6.0)

    def test_warm_start_interpolated_depth_growth(self):
        """Params at depth p-1 can be interpolated to depth p."""
        p1_params = warm_start_qaoa(n_layers=1, strategy="fixed")
        p2_params = warm_start_qaoa(n_layers=2, strategy="interpolated",
                                    prev_params=p1_params)
        assert p2_params.shape == (4,)
        # First and last point of interpolation should match original
        assert abs(p2_params[0] - p1_params[0]) < 1e-10
