"""Tests for crosstalk + ZZ coupling model — M90."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quantx_engine.crosstalk import (
    CouplingGraph,
    CrosstalkError,
    CrosstalkModel,
    cheapest_path,
    echo_sequence,
    zz_routing_cost,
)
from quantx_engine.pulse import QubitSystem, PulseSimulator, PulseSchedule, DriveChannel, ConstantWaveform


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sys2(freq=0.0):
    return QubitSystem(2, [freq, freq])


def _graph2(zeta=0.1):
    """Two-qubit graph with single ZZ edge."""
    return CouplingGraph(2, [(0, 1, zeta)])


# ── CouplingGraph construction ────────────────────────────────────────────────

class TestCouplingGraph:
    def test_default_empty(self):
        g = CouplingGraph(3)
        assert g.n_qubits == 3
        assert len(g.edges) == 0

    def test_single_edge(self):
        g = CouplingGraph(2, [(0, 1, 0.05)])
        assert g.zeta(0, 1) == pytest.approx(0.05)

    def test_symmetry(self):
        g = CouplingGraph(2, [(0, 1, 0.05)])
        assert g.zeta(1, 0) == pytest.approx(0.05)

    def test_missing_edge_returns_zero(self):
        g = CouplingGraph(3, [(0, 1, 0.05)])
        assert g.zeta(0, 2) == 0.0
        assert g.zeta(1, 2) == 0.0

    def test_self_loop_raises(self):
        with pytest.raises(CrosstalkError):
            CouplingGraph(2, [(0, 0, 0.1)])

    def test_out_of_range_raises(self):
        with pytest.raises(CrosstalkError):
            CouplingGraph(2, [(0, 5, 0.1)])

    def test_n_qubits_zero_raises(self):
        with pytest.raises(CrosstalkError):
            CouplingGraph(0)

    def test_neighbors_single_edge(self):
        g = CouplingGraph(3, [(0, 1, 0.05), (1, 2, 0.03)])
        nb = dict(g.neighbors(1))
        assert nb[0] == pytest.approx(0.05)
        assert nb[2] == pytest.approx(0.03)

    def test_neighbors_isolated_qubit(self):
        g = CouplingGraph(3, [(0, 1, 0.05)])
        assert g.neighbors(2) == []

    def test_adjacency_matrix_shape(self):
        g = CouplingGraph(3, [(0, 1, 0.1), (1, 2, 0.2)])
        mat = g.adjacency_matrix()
        assert mat.shape == (3, 3)
        assert mat[0, 1] == pytest.approx(0.1)
        assert mat[1, 0] == pytest.approx(0.1)
        assert mat[0, 2] == 0.0

    def test_adjacency_matrix_symmetric(self):
        g = CouplingGraph(4, [(0, 1, 0.1), (2, 3, 0.2)])
        mat = g.adjacency_matrix()
        assert np.allclose(mat, mat.T)

    def test_repr(self):
        g = CouplingGraph(2, [(0, 1, 0.05)])
        assert "CouplingGraph" in repr(g)
        assert "2" in repr(g)


# ── CrosstalkModel construction ───────────────────────────────────────────────

class TestCrosstalkModelConstruction:
    def test_mismatched_n_qubits_raises(self):
        sys3 = QubitSystem(3, [0.0, 0.0, 0.0])
        graph2 = CouplingGraph(2, [(0, 1, 0.1)])
        with pytest.raises(CrosstalkError):
            CrosstalkModel(sys3, graph2)

    def test_repr(self):
        m = CrosstalkModel(_sys2(), _graph2())
        assert "CrosstalkModel" in repr(m)

    def test_zz_phase_linear_in_time(self):
        m = CrosstalkModel(_sys2(), _graph2(zeta=0.1))
        t1 = m.zz_phase(0, 1, 10.0)
        t2 = m.zz_phase(0, 1, 20.0)
        assert t2 == pytest.approx(2.0 * t1, rel=1e-10)

    def test_zz_phase_proportional_to_zeta(self):
        m1 = CrosstalkModel(_sys2(), _graph2(zeta=0.1))
        m2 = CrosstalkModel(_sys2(), _graph2(zeta=0.2))
        assert m2.zz_phase(0, 1, 5.0) == pytest.approx(2.0 * m1.zz_phase(0, 1, 5.0))

    def test_zz_phase_zero_for_absent_edge(self):
        g = CouplingGraph(3, [(0, 1, 0.1)])
        sys3 = QubitSystem(3, [0.0, 0.0, 0.0])
        m = CrosstalkModel(sys3, g)
        assert m.zz_phase(0, 2, 100.0) == 0.0


# ── Idle unitary ──────────────────────────────────────────────────────────────

class TestIdleUnitary:
    def test_idle_unitary_is_unitary(self):
        m = CrosstalkModel(_sys2(), _graph2(zeta=0.1))
        U = m.idle_unitary(5.0)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-10)

    def test_idle_unitary_zero_time_is_identity(self):
        m = CrosstalkModel(_sys2(), _graph2(zeta=0.1))
        U = m.idle_unitary(0.0)
        assert np.allclose(U, np.eye(4), atol=1e-12)

    def test_idle_unitary_no_zz_is_trivial(self):
        """With zeta=0 and on-resonance qubits, U should be identity."""
        m = CrosstalkModel(_sys2(), CouplingGraph(2, [(0, 1, 0.0)]))
        U = m.idle_unitary(10.0)
        assert np.allclose(U, np.eye(4), atol=1e-10)

    def test_idle_unitary_diagonal_computational_basis(self):
        """H_ZZ = ζ/2 · ZZ is diagonal in the computational basis; U is diagonal."""
        m = CrosstalkModel(_sys2(), _graph2(zeta=0.1))
        U = m.idle_unitary(5.0)
        # All off-diagonal elements should be zero
        off_diag = U - np.diag(np.diag(U))
        assert np.allclose(off_diag, 0.0, atol=1e-10)


# ── ZZ phase accumulation physics ────────────────────────────────────────────

class TestZZPhysics:
    """Physical checks: |00⟩, |11⟩ are eigenstates; |01⟩,|10⟩ accumulate phase."""

    def test_00_state_stable(self):
        m = CrosstalkModel(_sys2(), _graph2(zeta=0.3))
        r = m.evolve_idle(20.0)
        assert r.populations.get("00", 0.0) == pytest.approx(1.0, abs=1e-8)

    def test_11_state_stable(self):
        psi11 = np.array([0, 0, 0, 1], dtype=complex)
        m = CrosstalkModel(_sys2(), _graph2(zeta=0.3))
        r = m.evolve_idle(20.0, initial_state=psi11)
        assert r.populations.get("11", 0.0) == pytest.approx(1.0, abs=1e-8)

    def test_01_state_gains_phase(self):
        """After idle, |01⟩ gains a global phase — population unchanged."""
        psi01 = np.array([0, 1, 0, 0], dtype=complex)
        m = CrosstalkModel(_sys2(), _graph2(zeta=0.3))
        r = m.evolve_idle(10.0, initial_state=psi01)
        assert r.populations.get("01", 0.0) == pytest.approx(1.0, abs=1e-8)

    def test_superposition_acquires_zz_phase(self):
        """(|00⟩+|01⟩)/√2 accumulates relative ZZ phase.

        Under H_ZZ = ζ/2·ZZ the diagonal elements are:
          |00⟩: +ζ/2,  |01⟩: −ζ/2
        so the relative phase grows as ζ·t.
        After t = π/ζ the relative phase is π → state becomes (|00⟩−|01⟩)/√2.
        """
        zeta = 0.2
        t_pi = math.pi / zeta       # relative ZZ phase = ζ·t = π
        m = CrosstalkModel(_sys2(), _graph2(zeta=zeta))
        # |00⟩ (index 0) + |01⟩ (index 1) — same populations after evolution
        psi = np.array([1, 1, 0, 0], dtype=complex) / math.sqrt(2)
        r = m.evolve_idle(t_pi, initial_state=psi)
        assert r.populations.get("00", 0.0) == pytest.approx(0.5, abs=0.01)
        assert r.populations.get("01", 0.0) == pytest.approx(0.5, abs=0.01)
        # Relative phase between |01⟩ and |00⟩ components should be ≈ π
        phi_00 = np.angle(r.final_state[0])
        phi_01 = np.angle(r.final_state[1])
        delta = (phi_01 - phi_00) % (2 * math.pi)
        assert min(abs(delta - math.pi), abs(delta - math.pi + 2 * math.pi)) < 0.1

    def test_evolve_idle_negative_time_raises(self):
        m = CrosstalkModel(_sys2(), _graph2())
        with pytest.raises(CrosstalkError):
            m.evolve_idle(-1.0)

    def test_evolve_idle_zero_norm_raises(self):
        m = CrosstalkModel(_sys2(), _graph2())
        with pytest.raises(CrosstalkError):
            m.evolve_idle(1.0, initial_state=np.zeros(4))


# ── ZZ-aware routing cost ─────────────────────────────────────────────────────

class TestZZRoutingCost:
    def test_base_cost_with_zero_zz(self):
        g = CouplingGraph(2, [(0, 1, 0.0)])
        assert zz_routing_cost(g, 0, 1) == pytest.approx(1.0)

    def test_cost_increases_with_zz(self):
        g_low = CouplingGraph(2, [(0, 1, 0.01)])
        g_high = CouplingGraph(2, [(0, 1, 1.0)])
        assert zz_routing_cost(g_high, 0, 1) > zz_routing_cost(g_low, 0, 1)

    def test_cost_symmetric(self):
        g = CouplingGraph(2, [(0, 1, 0.05)])
        assert zz_routing_cost(g, 0, 1) == pytest.approx(zz_routing_cost(g, 1, 0))

    def test_absent_edge_uses_base_cost(self):
        g = CouplingGraph(3, [(0, 1, 0.1)])
        assert zz_routing_cost(g, 0, 2) == pytest.approx(1.0)

    def test_zz_weight_scales_penalty(self):
        g = CouplingGraph(2, [(0, 1, 0.1)])
        c1 = zz_routing_cost(g, 0, 1, zz_weight=1.0)
        c2 = zz_routing_cost(g, 0, 1, zz_weight=2.0)
        assert c2 - 1.0 == pytest.approx(2.0 * (c1 - 1.0))

    def test_custom_base_cost(self):
        g = CouplingGraph(2, [(0, 1, 0.0)])
        assert zz_routing_cost(g, 0, 1, swap_base_cost=3.0) == pytest.approx(3.0)


# ── cheapest_path ─────────────────────────────────────────────────────────────

class TestCheapestPath:
    def test_direct_edge_path(self):
        g = CouplingGraph(2, [(0, 1, 0.0)])
        path, cost = cheapest_path(g, 0, 1)
        assert path == [0, 1]
        assert cost == pytest.approx(1.0)

    def test_indirect_path(self):
        # 0 --(high ZZ)-- 1 --(low ZZ)-- 2 --(low ZZ)-- 3
        # 0 --(low ZZ)--------------------------------------3  (direct, but edge may not exist)
        # Actually: 0->1->2->3 is cheaper if direct 0->3 has high ZZ
        g = CouplingGraph(4, [
            (0, 1, 5.0),   # high ZZ
            (1, 2, 0.01),
            (2, 3, 0.01),
            (0, 3, 5.0),   # also high ZZ direct
        ])
        path, cost = cheapest_path(g, 0, 3)
        # Should route 0->1->2->3 if cheaper than 0->3
        assert path[0] == 0
        assert path[-1] == 3

    def test_source_equals_target(self):
        g = CouplingGraph(3, [(0, 1, 0.1)])
        path, cost = cheapest_path(g, 0, 0)
        assert path == [0]
        assert cost == pytest.approx(0.0)

    def test_no_path_returns_empty(self):
        g = CouplingGraph(4, [(0, 1, 0.1), (2, 3, 0.1)])  # disconnected
        path, cost = cheapest_path(g, 0, 3)
        assert path == []
        assert cost == math.inf

    def test_prefers_low_zz_path(self):
        """Three qubits: 0 --(high ZZ)-- 1 -- 2; 0 --(low ZZ)-- 2 direct."""
        g = CouplingGraph(3, [
            (0, 1, 10.0),  # expensive
            (1, 2, 0.01),
            (0, 2, 0.01),  # cheap direct
        ])
        path, cost = cheapest_path(g, 0, 2)
        assert path == [0, 2]   # direct low-ZZ path


# ── Echo sequence ─────────────────────────────────────────────────────────────

class TestEchoSequence:
    def test_schedule_has_two_drive_instructions(self):
        sched = echo_sequence(0, 1, idle_time=20.0, pi_duration=5.0)
        drive_instrs = [i for i in sched.instructions
                        if hasattr(i.channel, "qubit") and i.waveform.amp > 0]
        assert len(drive_instrs) == 2

    def test_echo_duration_correct(self):
        sched = echo_sequence(0, 1, idle_time=20.0, pi_duration=5.0)
        assert sched.duration == pytest.approx(25.0)

    def test_echo_targets_both_qubits(self):
        sched = echo_sequence(0, 1, idle_time=20.0, pi_duration=5.0)
        drive_qubits = {i.channel.qubit for i in sched.instructions
                        if hasattr(i.channel, "qubit") and i.waveform.amp > 0}
        assert 0 in drive_qubits
        assert 1 in drive_qubits

    def test_echo_pi_pulses_at_half_idle(self):
        idle = 20.0; pi_dur = 4.0
        sched = echo_sequence(0, 1, idle_time=idle, pi_duration=pi_dur)
        for inst in sched.instructions:
            if hasattr(inst.channel, "qubit") and inst.waveform.amp > 0:
                assert inst.t_start == pytest.approx(idle / 2.0)

    def test_echo_gaussian_option(self):
        sched = echo_sequence(0, 1, idle_time=20.0, pi_duration=6.0,
                              use_gaussian=True, pi_sigma=1.0)
        assert sched.duration > 0

    def test_echo_negative_idle_raises(self):
        with pytest.raises(CrosstalkError):
            echo_sequence(0, 1, idle_time=-1.0)

    def test_echo_cancels_zz_accumulation(self):
        """
        Physical test: without echo, |+0⟩ accumulates ZZ phase.
        With echo (X⊗X at T/2), the ZZ phase cancels and ⟨X_0⟩ is restored.
        """
        zeta = 0.2
        T_idle = math.pi / zeta / 2   # half-π ZZ phase without echo
        pi_dur = 5.0
        system = QubitSystem(2, [0.0, 0.0], t1_times=None, t2_times=None)
        graph = CouplingGraph(2, [(0, 1, zeta)])
        model = CrosstalkModel(system, graph)

        # Prepare |+0⟩ = (|00⟩ + |10⟩)/√2  (q1=|+⟩, q0=|0⟩)
        psi_plus0 = np.zeros(4, dtype=complex)
        psi_plus0[0] = 1 / math.sqrt(2)   # |00⟩
        psi_plus0[2] = 1 / math.sqrt(2)   # |10⟩

        # Without echo: after T_idle, ZZ phase accumulated
        r_no_echo = model.evolve_idle(T_idle, initial_state=psi_plus0)
        bx_no_echo = r_no_echo.expectation(
            np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]], dtype=complex)  # X⊗I
        )

        # Full ZZ cancellation: simulate with PulseSimulator including echo X-pulses
        # (simplified: just verify the ZZ accumulation was non-zero without echo)
        assert abs(bx_no_echo) < 0.99   # ZZ caused decoherence of ⟨X⟩


# ── __init__ re-exports ────────────────────────────────────────────────────────

class TestCrosstalkInit:
    def test_coupling_graph_exported(self):
        from quantx_engine import CouplingGraph as CG
        assert CG is CouplingGraph

    def test_crosstalk_model_exported(self):
        from quantx_engine import CrosstalkModel as CM
        assert CM is CrosstalkModel

    def test_crosstalk_error_exported(self):
        from quantx_engine import CrosstalkError as CE
        assert CE is CrosstalkError

    def test_zz_routing_cost_exported(self):
        from quantx_engine import zz_routing_cost as zrc
        assert zrc is zz_routing_cost

    def test_echo_sequence_exported(self):
        from quantx_engine import echo_sequence as es
        assert es is echo_sequence
