"""Tests for M125: measurement-based quantum computation (MBQC)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantx_engine.mbqc import (
    GraphState,
    MBQCError,
    MBQCPattern,
    MeasurementResult,
)


# ---------------------------------------------------------------------------
# GraphState construction
# ---------------------------------------------------------------------------

class TestGraphStateConstruction:
    def test_basic(self):
        g = GraphState(3, [(0, 1), (1, 2)])
        assert g.n_qubits == 3

    def test_n_edges(self):
        g = GraphState(3, [(0, 1), (1, 2)])
        assert g.n_edges == 2

    def test_n_qubits_zero_raises(self):
        with pytest.raises(MBQCError, match="n_qubits"):
            GraphState(0)

    def test_out_of_range_edge_raises(self):
        with pytest.raises(MBQCError, match="out of range"):
            GraphState(2, [(0, 3)])

    def test_self_loop_raises(self):
        with pytest.raises(MBQCError, match="Self-loop"):
            GraphState(2, [(0, 0)])

    def test_too_many_qubits_raises(self):
        with pytest.raises(MBQCError, match="15"):
            GraphState(16)

    def test_repr(self):
        g = GraphState(3, [(0, 1)])
        assert "GraphState" in repr(g)
        assert "n_qubits=3" in repr(g)


class TestGraphStateProperties:
    def test_state_vector_normalized(self):
        g = GraphState(3, [(0, 1), (1, 2)])
        sv = g.state_vector
        assert abs(np.linalg.norm(sv) - 1.0) < 1e-12

    def test_state_vector_size(self):
        n = 4
        g = GraphState(n, [(0, 1), (2, 3)])
        assert g.state_vector.shape == (1 << n,)

    def test_adjacency_matrix_symmetric(self):
        g = GraphState(3, [(0, 1), (1, 2)])
        A = g.adjacency_matrix
        assert np.array_equal(A, A.T)

    def test_adjacency_matrix_values(self):
        g = GraphState(3, [(0, 1)])
        A = g.adjacency_matrix
        assert A[0, 1] == 1 and A[1, 0] == 1
        assert A[0, 2] == 0

    def test_no_edges_gives_plus_product_state(self):
        # Without edges, |G⟩ = |+⟩^⊗n (uniform superposition)
        g = GraphState(2)
        sv = g.state_vector
        expected = np.array([0.5, 0.5, 0.5, 0.5], dtype=complex)
        assert np.allclose(sv, expected, atol=1e-12)

    def test_fidelity_with_self_is_one(self):
        g = GraphState(3, [(0, 1), (1, 2)])
        assert g.fidelity_with(g) == pytest.approx(1.0)

    def test_fidelity_different_graphs_less_than_one(self):
        g1 = GraphState(2, [(0, 1)])
        g2 = GraphState(2)
        assert g1.fidelity_with(g2) < 1.0

    def test_fidelity_wrong_size_raises(self):
        g1 = GraphState(2, [(0, 1)])
        g2 = GraphState(3)
        with pytest.raises(MBQCError):
            g1.fidelity_with(g2)

    def test_entanglement_entropy_unentangled(self):
        g = GraphState(3)  # no edges → product state → S = 0
        S = g.entanglement_entropy([0])
        assert S == pytest.approx(0.0, abs=1e-10)

    def test_entanglement_entropy_bell_state(self):
        # Single CZ edge → entangled; entropy of half-system ≥ 0
        g = GraphState(2, [(0, 1)])
        S = g.entanglement_entropy([0])
        assert S > 0.0


class TestGraphStateFactories:
    def test_linear_cluster(self):
        g = GraphState.linear_cluster(4)
        assert g.n_qubits == 4
        assert g.n_edges == 3

    def test_linear_cluster_too_short(self):
        with pytest.raises(MBQCError):
            GraphState.linear_cluster(1)

    def test_grid_cluster(self):
        g = GraphState.grid_cluster(2, 3)
        assert g.n_qubits == 6
        # 2×3 grid: 2 horizontal edges per row × 2 rows = 4
        #            1 vertical edge per pair × 3 cols = 3  → total 7
        assert g.n_edges == 7

    def test_grid_cluster_too_large(self):
        with pytest.raises(MBQCError, match="15"):
            GraphState.grid_cluster(4, 4)  # 16 qubits

    def test_state_vector_normalized_linear(self):
        g = GraphState.linear_cluster(5)
        assert abs(np.linalg.norm(g.state_vector) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# MBQCPattern
# ---------------------------------------------------------------------------

class TestMBQCPatternConstruction:
    def test_basic(self):
        g = GraphState(3, [(0, 1), (1, 2)])
        p = MBQCPattern(g, input_qubits=[0], output_qubits=[2])
        assert p.n == 3

    def test_out_of_range_input_raises(self):
        g = GraphState(3)
        with pytest.raises(MBQCError):
            MBQCPattern(g, input_qubits=[5])

    def test_repr(self):
        g = GraphState(3)
        p = MBQCPattern(g)
        assert "MBQCPattern" in repr(p)


class TestMBQCPatternMeasure:
    @pytest.fixture
    def pattern(self):
        g = GraphState.linear_cluster(4)
        return MBQCPattern(g, input_qubits=[0], output_qubits=[3])

    def test_measure_returns_result(self, pattern):
        r = pattern.measure(1, angle=0.0, seed=0)
        assert isinstance(r, MeasurementResult)

    def test_measure_outcome_is_0_or_1(self, pattern):
        r = pattern.measure(1, angle=0.0, seed=0)
        assert r.outcome in (0, 1)

    def test_forced_outcome(self, pattern):
        r = pattern.measure(1, angle=0.0, outcome=1)
        assert r.outcome == 1

    def test_double_measure_raises(self, pattern):
        pattern.measure(1, angle=0.0, seed=0)
        with pytest.raises(MBQCError, match="already been measured"):
            pattern.measure(1, angle=0.0, seed=0)

    def test_measure_output_qubit_raises(self, pattern):
        with pytest.raises(MBQCError, match="output qubit"):
            pattern.measure(3, angle=0.0)

    def test_n_measured_increments(self, pattern):
        assert pattern.n_measured == 0
        pattern.measure(1, angle=0.0, seed=0)
        assert pattern.n_measured == 1

    def test_measurement_outcomes_populated(self, pattern):
        pattern.measure(1, angle=0.0, outcome=0)
        assert 1 in pattern.measurement_outcomes
        assert pattern.measurement_outcomes[1] == 0

    def test_corrected_angle_stored(self, pattern):
        r = pattern.measure(1, angle=math.pi / 4, seed=0)
        assert math.isfinite(r.corrected_angle)


class TestMBQCPatternRun:
    def test_run_multiple_measurements(self):
        g = GraphState.linear_cluster(5)
        p = MBQCPattern(g, output_qubits=[4])
        seq = [(1, 0.0), (2, math.pi / 4), (3, 0.0)]
        results = p.run(seq, seed=0)
        assert len(results) == 3
        assert all(r.outcome in (0, 1) for r in results)

    def test_run_with_forced_outcomes(self):
        g = GraphState.linear_cluster(4)
        p = MBQCPattern(g, output_qubits=[3])
        seq = [(1, 0.0, 0), (2, 0.0, 1)]
        results = p.run(seq)
        assert results[0].outcome == 0
        assert results[1].outcome == 1


class TestMBQCPatternOutput:
    def test_output_state_normalized(self):
        g = GraphState.linear_cluster(4)
        p = MBQCPattern(g, output_qubits=[3])
        p.run([(1, 0.0), (2, 0.0)], seed=0)
        out = p.output_state()
        assert abs(np.linalg.norm(out) - 1.0) < 1e-10

    def test_output_state_correct_size(self):
        # 4-qubit linear cluster, measure qubits 1 and 2; output_qubits=[3]
        # output_state() now returns only the declared output qubits.
        g = GraphState.linear_cluster(4)
        p = MBQCPattern(g, output_qubits=[3])
        p.run([(1, 0.0), (2, 0.0)], seed=0)
        out = p.output_state()
        # 1 declared output qubit → state of size 2^1 = 2
        assert out.shape == (2,)

    def test_output_state_no_measurements_returns_full(self):
        g = GraphState(3)
        p = MBQCPattern(g, output_qubits=[0, 1])
        # No measurements yet; output state involves remaining qubits
        out = p.output_state()
        assert out.shape[0] > 0


class TestMBQCCorrections:
    def test_add_x_correction_runs(self):
        g = GraphState.linear_cluster(3)
        p = MBQCPattern(g, output_qubits=[2])
        p.add_x_correction(target=2, depends_on=[0])
        p.measure(0, angle=0.0, seed=0)
        p.measure(1, angle=0.0, seed=1)
        p.apply_corrections(p.measurement_outcomes)
        # No error raised

    def test_add_z_correction_runs(self):
        g = GraphState.linear_cluster(3)
        p = MBQCPattern(g, output_qubits=[2])
        p.add_z_correction(target=2, depends_on=[1])
        p.measure(0, angle=0.0, seed=0)
        p.measure(1, angle=0.0, seed=1)
        p.apply_corrections(p.measurement_outcomes)

    def test_out_of_range_correction_target_raises(self):
        g = GraphState(3)
        p = MBQCPattern(g)
        with pytest.raises(MBQCError):
            p.add_x_correction(target=10, depends_on=[0])

    def test_z_correction_out_of_range_raises(self):
        g = GraphState(3)
        p = MBQCPattern(g)
        with pytest.raises(MBQCError):
            p.add_z_correction(target=10, depends_on=[0])
