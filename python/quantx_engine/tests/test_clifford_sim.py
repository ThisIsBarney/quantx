"""M110: Tests for large-scale Clifford (stabilizer) simulation.

Covers:
  - StabilizerState tableau initialisation and low-level gate operations
  - CliffordSimulator gate API and measurement
  - GHZ circuit correctness (all outcomes equal)
  - Bell-state correlations
  - Deterministic measurement outcomes (|0⟩, |1⟩)
  - Phase gates: S, S†, Z
  - 10 000-qubit scale test (runtime < 60 s)
  - StimCircuit: string parsing, gate execution, measurement
  - Error handling: out-of-range qubits, bad Stim syntax
"""
from __future__ import annotations

import textwrap
import time

import pytest

from quantx_engine.clifford_sim import (
    CliffordError,
    CliffordSimulator,
    StabilizerState,
    StimCircuit,
    StimCircuitError,
)


# ── StabilizerState low-level ─────────────────────────────────────────────────

class TestStabilizerStateInit:
    def test_single_qubit(self):
        st = StabilizerState(1)
        assert st.n == 1
        assert st.n_qubits == 1

    def test_tableau_shape(self):
        n = 5
        st = StabilizerState(n)
        assert st._tab.shape == (2 * n, 2 * n + 1)

    def test_initial_state_is_zero(self):
        """In |0...0⟩, destabilisers are X_i, stabilisers are Z_i."""
        n = 4
        st = StabilizerState(n)
        # destabiliser i has x[i,i]=1, all other x/z columns 0
        for i in range(n):
            assert st._tab[i, i] == 1               # x_i of destabiliser i
            assert st._tab[i, n + i] == 0            # z_i of destabiliser i
            assert st._tab[i, 2 * n] == 0            # phase of destabiliser i
        # stabiliser i has z[n+i, n+i]=1
        for i in range(n):
            assert st._tab[n + i, i] == 0            # x_i of stabiliser i
            assert st._tab[n + i, n + i] == 1        # z_i of stabiliser i
            assert st._tab[n + i, 2 * n] == 0        # phase of stabiliser i

    def test_zero_qubits_raises(self):
        with pytest.raises(CliffordError):
            StabilizerState(0)

    def test_negative_qubits_raises(self):
        with pytest.raises(CliffordError):
            StabilizerState(-1)

    def test_copy_is_independent(self):
        st = StabilizerState(3)
        st.h(0)
        copy = st.copy()
        # Mutate original — copy should not change
        st.s(0)
        assert (copy._tab != st._tab).any()


# ── Single-qubit gates ────────────────────────────────────────────────────────

class TestSingleQubitGates:
    """Apply gates and verify measurement outcomes on simple states."""

    def test_x_flips_zero_to_one(self):
        sim = CliffordSimulator(n_qubits=1)
        sim.x(0)
        assert sim.measure(0) == 1

    def test_double_x_is_identity(self):
        sim = CliffordSimulator(n_qubits=1)
        sim.x(0).x(0)
        assert sim.measure(0) == 0

    def test_z_leaves_zero_unchanged(self):
        """Z|0⟩ = |0⟩, so Z-measurement should give 0."""
        sim = CliffordSimulator(n_qubits=1)
        sim.z(0)
        assert sim.measure(0) == 0

    def test_y_on_zero(self):
        """Y|0⟩ = i|1⟩ → measures 1."""
        sim = CliffordSimulator(n_qubits=1)
        sim.y(0)
        assert sim.measure(0) == 1

    def test_s_then_sdg_is_identity(self):
        """S followed by S† = I."""
        sim = CliffordSimulator(n_qubits=1)
        sim.h(0)
        sim.s(0)
        sim.sdg(0)
        # Should still be in |+⟩ — measuring gives 0 or 1 but tableau unchanged
        sim2 = CliffordSimulator(n_qubits=1)
        sim2.h(0)
        # Both tableaux must be identical
        import numpy as np
        np.testing.assert_array_equal(sim._state._tab, sim2._state._tab)

    def test_h_creates_superposition(self):
        """H|0⟩ is in a superposition — measure 0 or 1 are both valid."""
        sim = CliffordSimulator(n_qubits=1)
        sim.h(0)
        outcome = sim.measure(0)
        assert outcome in (0, 1)

    def test_h_h_is_identity(self):
        """H·H = I: state must be deterministically |0⟩ after two Hadamards."""
        sim = CliffordSimulator(n_qubits=1)
        sim.h(0).h(0)
        assert sim.measure(0) == 0

    def test_out_of_range_qubit_raises(self):
        sim = CliffordSimulator(n_qubits=2)
        with pytest.raises(CliffordError):
            sim.h(2)
        with pytest.raises(CliffordError):
            sim.x(-1)

    def test_s_squared_is_z(self):
        """S² = Z: applying S twice and Z once should give the same state."""
        import numpy as np
        sim_ss = CliffordSimulator(n_qubits=1)
        sim_ss.h(0).s(0).s(0)

        sim_z = CliffordSimulator(n_qubits=1)
        sim_z.h(0).z(0)

        np.testing.assert_array_equal(sim_ss._state._tab, sim_z._state._tab)


# ── Measurement ───────────────────────────────────────────────────────────────

class TestMeasurement:
    def test_measure_zero_state_always_zero(self):
        for seed in range(10):
            sim = CliffordSimulator(n_qubits=3, seed=seed)
            for q in range(3):
                assert sim.measure(q) == 0

    def test_measure_x_state_always_one(self):
        for seed in range(5):
            sim = CliffordSimulator(n_qubits=2, seed=seed)
            sim.x(0).x(1)
            assert sim.measure(0) == 1
            assert sim.measure(1) == 1

    def test_measure_collapses_superposition(self):
        """After measuring |+⟩ to 0 or 1, second measurement is deterministic."""
        sim = CliffordSimulator(n_qubits=1, seed=7)
        sim.h(0)
        first = sim.measure(0)
        second = sim.measure(0)
        assert first == second

    def test_measure_all(self):
        sim = CliffordSimulator(n_qubits=4)
        outcomes = sim.measure_all()
        assert outcomes == [0, 0, 0, 0]

    def test_measurement_outcomes_property(self):
        sim = CliffordSimulator(n_qubits=3)
        sim.x(1)
        sim.measure(0)
        sim.measure(1)
        sim.measure(2)
        outcomes = sim.measurement_outcomes
        assert outcomes[0] == 0
        assert outcomes[1] == 1
        assert outcomes[2] == 0

    def test_product_superposition_independent_outcomes(self):
        """P1 regression: H|0⟩⊗H|0⟩ must produce all four outcomes 00/01/10/11.

        Before the fix, every measure(seed=X) recreated random.Random(X) and
        both qubits always collapsed to the same value (00 or 11 only).
        """
        seen = set()
        for seed in range(200):
            sim = CliffordSimulator(n_qubits=2, seed=seed)
            sim.h(0).h(1)
            outcomes = tuple(sim.measure_all())
            seen.add(outcomes)
        assert (0, 1) in seen, "Product state never produced 01"
        assert (1, 0) in seen, "Product state never produced 10"


# ── Bell state ────────────────────────────────────────────────────────────────

class TestBellState:
    def test_bell_state_correlated(self):
        """Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2: both qubits always equal."""
        for seed in range(20):
            sim = CliffordSimulator(n_qubits=2, seed=seed)
            sim.h(0).cx(0, 1)
            a = sim.measure(0)
            b = sim.measure(1)
            assert a == b, f"seed={seed}: got {a},{b}"

    def test_bell_state_both_outcomes_possible(self):
        """Over many seeds, we should see both 00 and 11."""
        outcomes = set()
        for seed in range(50):
            sim = CliffordSimulator(n_qubits=2, seed=seed)
            sim.h(0).cx(0, 1)
            a = sim.measure(0)
            b = sim.measure(1)
            outcomes.add((a, b))
        assert (0, 0) in outcomes
        assert (1, 1) in outcomes
        assert (0, 1) not in outcomes
        assert (1, 0) not in outcomes


# ── GHZ circuit ───────────────────────────────────────────────────────────────

class TestGHZ:
    @pytest.mark.parametrize("n_qubits", [3, 10, 50, 100])
    def test_ghz_all_equal(self, n_qubits: int):
        """GHZ state: all qubits must collapse to the same value."""
        sim = CliffordSimulator(n_qubits=n_qubits, seed=42)
        sim.h(0)
        for q in range(n_qubits - 1):
            sim.cx(q, q + 1)
        outcomes = [sim.measure(q) for q in range(n_qubits)]
        assert len(set(outcomes)) == 1, f"Outcomes not all equal: {outcomes[:10]}"

    def test_ghz_3_qubit_both_values(self):
        """Both 000 and 111 should appear across many seeds."""
        seen = set()
        for seed in range(100):
            sim = CliffordSimulator(n_qubits=3, seed=seed)
            sim.h(0).cx(0, 1).cx(0, 2)
            o = tuple(sim.measure(q) for q in range(3))
            seen.add(o)
        assert (0, 0, 0) in seen
        assert (1, 1, 1) in seen


# ── Two-qubit gates ───────────────────────────────────────────────────────────

class TestTwoQubitGates:
    def test_cx_ctrl_same_as_target_raises(self):
        sim = CliffordSimulator(n_qubits=2)
        with pytest.raises(CliffordError):
            sim.cx(0, 0)

    def test_cx_flips_target_when_ctrl_one(self):
        sim = CliffordSimulator(n_qubits=2)
        sim.x(0)          # |10⟩
        sim.cx(0, 1)      # → |11⟩
        assert sim.measure(0) == 1
        assert sim.measure(1) == 1

    def test_cx_no_flip_when_ctrl_zero(self):
        sim = CliffordSimulator(n_qubits=2)
        # ctrl is |0⟩, target is |0⟩
        sim.cx(0, 1)      # → |00⟩ unchanged
        assert sim.measure(0) == 0
        assert sim.measure(1) == 0

    def test_cz_symmetric(self):
        """CZ is symmetric: CZ(0,1) and CZ(1,0) give identical states."""
        import numpy as np
        sim_a = CliffordSimulator(n_qubits=2)
        sim_a.h(0).h(1).cz(0, 1)

        sim_b = CliffordSimulator(n_qubits=2)
        sim_b.h(0).h(1).cz(1, 0)

        np.testing.assert_array_equal(sim_a._state._tab, sim_b._state._tab)

    def test_swap(self):
        """SWAP(0,1) swaps |10⟩ to |01⟩."""
        sim = CliffordSimulator(n_qubits=2)
        sim.x(0)            # |10⟩
        sim.swap(0, 1)      # → |01⟩
        assert sim.measure(0) == 0
        assert sim.measure(1) == 1

    def test_swap_self_is_identity(self):
        """Applying SWAP twice is identity."""
        import numpy as np
        sim = CliffordSimulator(n_qubits=3)
        sim.h(0).cx(0, 1)
        tab_before = sim._state._tab.copy()
        sim.swap(1, 2).swap(1, 2)
        np.testing.assert_array_equal(sim._state._tab, tab_before)


# ── Large-scale test ──────────────────────────────────────────────────────────

class TestLargeScale:
    def test_10000_qubit_ghz_preparation(self):
        """10 000-qubit GHZ preparation + spot-check two qubits are correlated.

        Full measure-all would be O(n³) due to destabilizer chains; we instead
        verify that qubits 0 and n-1 collapse to the same value, which is
        sufficient to confirm the entanglement structure.
        """
        n = 10_000
        t0 = time.perf_counter()
        sim = CliffordSimulator(n_qubits=n, seed=0)
        sim.h(0)
        for q in range(n - 1):
            sim.cx(q, q + 1)
        # Measuring only qubit 0 (random) + qubit n-1 (deterministic, 2 rowmults)
        m0 = sim.measure(0)
        m_last = sim.measure(n - 1)
        elapsed = time.perf_counter() - t0
        assert m0 == m_last, f"Qubits 0 and {n-1} disagree: {m0} vs {m_last}"
        assert elapsed < 60.0, f"10000-qubit GHZ took {elapsed:.1f}s (> 60s limit)"

    def test_1000_qubit_ghz_all_equal(self):
        """1 000-qubit GHZ: all outcomes equal."""
        n = 1_000
        t0 = time.perf_counter()
        sim = CliffordSimulator(n_qubits=n, seed=0)
        sim.h(0)
        for q in range(n - 1):
            sim.cx(q, q + 1)
        outcomes = [sim.measure(q) for q in range(n)]
        elapsed = time.perf_counter() - t0
        assert len(set(outcomes)) == 1, "1000-qubit GHZ: not all equal"
        assert elapsed < 60.0, f"1000-qubit GHZ took {elapsed:.1f}s (> 60s limit)"

    def test_1000_qubit_all_x(self):
        """Apply X to all 1000 qubits; all should measure 1."""
        n = 1_000
        sim = CliffordSimulator(n_qubits=n)
        for q in range(n):
            sim.x(q)
        outcomes = sim.measure_all()
        assert all(o == 1 for o in outcomes)

    def test_10000_qubit_identity(self):
        """H·H = I on each of 10 000 qubits; all should measure 0.

        Gate application is O(n) vectorized per gate; measure_all on |0...0⟩
        is deterministic with no rowmult chains → total O(n²) numpy ops.
        """
        n = 10_000
        t0 = time.perf_counter()
        sim = CliffordSimulator(n_qubits=n)
        for q in range(n):
            sim.h(q).h(q)
        outcomes = sim.measure_all()
        elapsed = time.perf_counter() - t0
        assert all(o == 0 for o in outcomes)
        assert elapsed < 60.0, f"10000-qubit identity took {elapsed:.1f}s (> 60s limit)"


# ── StimCircuit parser ────────────────────────────────────────────────────────

class TestStimCircuitParsing:
    def test_empty_string(self):
        circ = StimCircuit.from_string("")
        assert circ.n_qubits == 1
        assert circ.instructions == []

    def test_comment_only(self):
        circ = StimCircuit.from_string("# This is a comment\n")
        assert circ.instructions == []

    def test_single_h_gate(self):
        circ = StimCircuit.from_string("H 0\n")
        assert circ.n_qubits == 1
        assert len(circ.instructions) == 1
        assert circ.instructions[0] == ("H", [0])

    def test_multi_qubit_line(self):
        """H 0 1 2 on one line → three separate instructions."""
        circ = StimCircuit.from_string("H 0 1 2\n")
        assert circ.n_qubits == 3
        assert len(circ.instructions) == 3

    def test_cnot_pair(self):
        circ = StimCircuit.from_string("CNOT 0 1\n")
        assert circ.instructions[0] == ("CNOT", [0, 1])

    def test_cx_alias(self):
        circ = StimCircuit.from_string("CX 2 3\n")
        assert circ.instructions[0] == ("CX", [2, 3])

    def test_skip_tick_and_coords(self):
        text = textwrap.dedent("""\
            QUBIT_COORDS(0, 0) 0
            QUBIT_COORDS(1, 0) 1
            TICK
            H 0
            CNOT 0 1
            TICK
            M 0 1
        """)
        circ = StimCircuit.from_string(text)
        # Only H, CNOT, M 0, M 1 should remain
        gates = [g for g, _ in circ.instructions]
        assert "QUBIT_COORDS" not in gates
        assert "TICK" not in gates
        assert "H" in gates
        assert "CNOT" in gates
        assert gates.count("M") == 2

    def test_measurement_with_parens(self):
        """M(0.001) 0 — probability arg in parens should be stripped."""
        circ = StimCircuit.from_string("M(0.001) 0\n")
        assert circ.instructions[0][0] == "M"

    def test_n_qubits_inferred(self):
        circ = StimCircuit.from_string("H 7\nCNOT 3 9\n")
        assert circ.n_qubits == 10  # max qubit index 9 → n_qubits = 10

    def test_unknown_gate_raises(self):
        with pytest.raises(StimCircuitError, match="unsupported"):
            StimCircuit.from_string("MAGIC_GATE 0\n")

    def test_odd_operands_two_qubit_gate_raises(self):
        with pytest.raises(StimCircuitError):
            StimCircuit.from_string("CNOT 0\n")

    def test_non_integer_operand_raises(self):
        with pytest.raises(StimCircuitError):
            StimCircuit.from_string("H foo\n")

    def test_s_dag_alias(self):
        circ = StimCircuit.from_string("S_DAG 0\n")
        assert circ.instructions[0] == ("S_DAG", [0])

    def test_from_file(self):
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".stim")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("H 0\nCNOT 0 1\nM 0 1\n")
            circ = StimCircuit.from_file(path)
            assert circ.n_qubits == 2
            assert len(circ.instructions) == 4  # H, CNOT, M 0, M 1
        finally:
            os.unlink(path)


# ── StimCircuit execution ─────────────────────────────────────────────────────

class TestStimCircuitRun:
    def test_bell_state_via_stim(self):
        """Bell circuit via Stim: both outcomes always equal."""
        text = "H 0\nCNOT 0 1\nM 0 1\n"
        circ = StimCircuit.from_string(text)
        for seed in range(20):
            outcomes = circ.run(seed=seed)
            assert len(outcomes) == 2
            assert outcomes[0] == outcomes[1], f"seed={seed}: {outcomes}"

    def test_ghz_3_via_stim(self):
        text = "H 0\nCNOT 0 1\nCNOT 0 2\nM 0 1 2\n"
        circ = StimCircuit.from_string(text)
        for seed in range(10):
            outcomes = circ.run(seed=seed)
            assert len(set(outcomes)) == 1

    def test_deterministic_zero_state(self):
        """Measuring the |0⟩ state always gives 0."""
        circ = StimCircuit.from_string("M 0 1 2\n")
        outcomes = circ.run(seed=0)
        assert outcomes == [0, 0, 0]

    def test_x_then_measure(self):
        """X followed by M always gives 1."""
        circ = StimCircuit.from_string("X 0\nM 0\n")
        outcomes = circ.run(seed=0)
        assert outcomes == [1]

    def test_s_gate_via_stim(self):
        circ = StimCircuit.from_string("H 0\nS 0\nM 0\n")
        outcomes = circ.run(seed=0)
        assert outcomes[0] in (0, 1)

    def test_cz_via_stim(self):
        text = "H 0\nH 1\nCZ 0 1\nM 0 1\n"
        circ = StimCircuit.from_string(text)
        outcomes = circ.run(seed=0)
        assert len(outcomes) == 2

    def test_swap_via_stim(self):
        """X on qubit 0, SWAP(0,1), M 0 → 0; M 1 → 1."""
        text = "X 0\nSWAP 0 1\nM 0\nM 1\n"
        circ = StimCircuit.from_string(text)
        outcomes = circ.run(seed=0)
        assert outcomes == [0, 1]

    def test_identity_gate_via_stim(self):
        """I gate is a no-op: state remains |0⟩."""
        circ = StimCircuit.from_string("I 0\nM 0\n")
        assert circ.run(seed=0) == [0]

    def test_mz_alias(self):
        circ = StimCircuit.from_string("X 0\nMZ 0\n")
        assert circ.run(seed=0) == [1]

    def test_multiple_runs_independent(self):
        """Each run() call is independent (fresh simulator)."""
        circ = StimCircuit.from_string("H 0\nM 0\n")
        results = [circ.run(seed=seed) for seed in range(30)]
        # Results should be a list of [0] or [1], all valid
        for r in results:
            assert r[0] in (0, 1)


# ── _g_table sanity checks ────────────────────────────────────────────────────

class TestGTable:
    """Verify the phase-lookup table against known Pauli products."""

    def test_identity_gives_zero(self):
        from quantx_engine.clifford_sim import _g_table
        for x2 in (0, 1):
            for z2 in (0, 1):
                assert _g_table(0, 0, x2, z2) == 0

    def test_xx_is_identity_phase_zero(self):
        from quantx_engine.clifford_sim import _g_table
        # X·X = I → g=0
        assert _g_table(1, 0, 1, 0) == 0

    def test_xy_gives_i(self):
        from quantx_engine.clifford_sim import _g_table
        # X·Y = iZ  (matrix: (i,0;0,-i) = iZ)
        # sigma(1,0)·sigma(1,1) = i^g · sigma(0,1), g=1
        assert _g_table(1, 0, 1, 1) == 1

    def test_yx_gives_minus_i(self):
        from quantx_engine.clifford_sim import _g_table
        # Y·X = -iZ  (matrix: (-i,0;0,i) = -iZ)
        # sigma(1,1)·sigma(1,0) = i^g · sigma(0,1), g=3
        assert _g_table(1, 1, 1, 0) == 3

    def test_yy_is_identity_phase_zero(self):
        from quantx_engine.clifford_sim import _g_table
        # Y·Y = I  →  i^g · sigma(0,0) = I  →  g=0
        assert _g_table(1, 1, 1, 1) == 0


# ── Phase tracking ────────────────────────────────────────────────────────────

class TestPhaseTracking:
    """Verify phase is tracked correctly for circuits with non-trivial phases."""

    def test_z_eigenvalue_after_h_s_h(self):
        """H·S·H = Rx(π/2)-like; check state is not |0⟩."""
        # After H·S·H on |0⟩ we get a superposition, not a Z eigenstate
        sim = CliffordSimulator(n_qubits=1)
        sim.h(0).s(0).h(0)
        outcome = sim.measure(0)
        assert outcome in (0, 1)

    def test_s4_is_identity(self):
        """S⁴ = I (S has order 4)."""
        import numpy as np
        sim = CliffordSimulator(n_qubits=1)
        sim.h(0)
        tab_before = sim._state._tab.copy()
        sim.s(0).s(0).s(0).s(0)
        np.testing.assert_array_equal(sim._state._tab, tab_before)

    def test_x_after_z_gives_minus_x(self):
        """Z·X·Z = -X (conjugation).  Test indirectly via stabiliser."""
        # Prepare: apply Z to |+⟩ (= H|0⟩) → |−⟩ which is H|1⟩
        # Then H → |1⟩ → measures 1
        sim = CliffordSimulator(n_qubits=1)
        sim.h(0).z(0).h(0)
        assert sim.measure(0) == 1
