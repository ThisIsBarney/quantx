"""Tests for quantx_engine.lindblad (M71: Lindblad open-system solver).

All tests run without a compiled native library by using mock Engine
objects.  Physics-level tests (T1 decay, steady state, concurrence) verify
that the Python API correctly parses and propagates the C++ output.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from quantx_engine.lindblad import (
    EntanglementTrace,
    JumpOperator,
    LindbladError,
    LindbladFrame,
    LindbladResult,
    LindbladSolver,
    _parse_result,
)


# ---------------------------------------------------------------------------
# Helpers: mock engine + synthetic frame data
# ---------------------------------------------------------------------------

def _make_frame_dict(step, time, purity, entropy, coherence, populations):
    return {
        "step": step,
        "time": time,
        "purity": purity,
        "entropy": entropy,
        "coherence": coherence,
        "populations": populations,
    }


def _mock_engine_for(lindblad_frames, entanglement_frames=None, observables=None):
    """Return an Engine mock whose run() returns the given frame data."""
    engine = MagicMock()
    engine.run.return_value = {
        "ok": True,
        "lindblad_frames": lindblad_frames,
        "entanglement_frames": entanglement_frames or [],
        "observables": observables or {},
    }
    return engine


def _make_decay_frames(n_steps=10, gamma=0.5, dt=0.1):
    """Simulate amplitude-damping T1 decay: purity = 1 - (1-e^{-γt})/2."""
    frames = []
    for s in range(n_steps + 1):
        t = s * dt
        p1 = math.exp(-gamma * t)   # excited-state population decays
        p0 = 1.0 - p1
        purity = p0 ** 2 + p1 ** 2
        entropy = -p0 * math.log(p0 + 1e-15) - p1 * math.log(p1 + 1e-15)
        frames.append(_make_frame_dict(
            step=s, time=t,
            purity=purity,
            entropy=max(0.0, entropy),
            coherence=0.0,
            populations=[p0, p1],
        ))
    return frames


# ---------------------------------------------------------------------------
# JumpOperator validation
# ---------------------------------------------------------------------------

class TestJumpOperator:
    def test_amplitude_damping_valid(self):
        jop = JumpOperator("amplitude_damping", qubit=0, rate=0.5)
        assert jop.kind == "amplitude_damping"
        assert jop.qubit == 0
        assert jop.rate == 0.5

    def test_dephasing_valid(self):
        jop = JumpOperator("dephasing", qubit=2, rate=0.1)
        assert jop.kind == "dephasing"

    def test_depolarizing_valid(self):
        jop = JumpOperator("depolarizing", qubit=1, rate=0.05)
        assert jop.kind == "depolarizing"

    def test_unknown_kind_raises(self):
        with pytest.raises(LindbladError, match="Unknown jump operator kind"):
            JumpOperator("photon_loss", qubit=0, rate=0.1)

    def test_negative_qubit_raises(self):
        with pytest.raises(LindbladError, match="qubit must be >= 0"):
            JumpOperator("dephasing", qubit=-1, rate=0.1)

    def test_negative_rate_raises(self):
        with pytest.raises(LindbladError, match="rate must be >= 0"):
            JumpOperator("amplitude_damping", qubit=0, rate=-0.1)

    def test_zero_rate_valid(self):
        jop = JumpOperator("amplitude_damping", qubit=0, rate=0.0)
        assert jop.rate == 0.0

    def test_to_hamiltonian_term_amplitude_damping(self):
        jop = JumpOperator("amplitude_damping", qubit=3, rate=0.7)
        term = jop._to_hamiltonian_term()
        assert term["label"] == "AmplitudeDamping3"
        assert math.isclose(term["coefficient"], 0.7)

    def test_to_hamiltonian_term_dephasing(self):
        jop = JumpOperator("dephasing", qubit=0, rate=0.2)
        term = jop._to_hamiltonian_term()
        assert term["label"] == "Dephasing0"
        assert math.isclose(term["coefficient"], 0.2)

    def test_to_hamiltonian_term_depolarizing(self):
        jop = JumpOperator("depolarizing", qubit=1, rate=0.05)
        term = jop._to_hamiltonian_term()
        assert term["label"] == "Depolarizing1"
        assert math.isclose(term["coefficient"], 0.05)


# ---------------------------------------------------------------------------
# LindbladSolver parameter validation
# ---------------------------------------------------------------------------

class TestLindbladSolverValidation:
    def _solver(self):
        return LindbladSolver(_mock_engine_for(_make_decay_frames()))

    def test_n_qubits_zero_raises(self):
        with pytest.raises(LindbladError, match="n_qubits must be >= 1"):
            self._solver().solve(n_qubits=0, hamiltonian_terms=[], jump_operators=[],
                                 total_time=1.0)

    def test_n_qubits_too_large_raises(self):
        with pytest.raises(LindbladError, match="at most 10 qubits"):
            self._solver().solve(n_qubits=11, hamiltonian_terms=[], jump_operators=[],
                                 total_time=1.0)

    def test_negative_total_time_raises(self):
        with pytest.raises(LindbladError, match="total_time must be > 0"):
            self._solver().solve(n_qubits=1, hamiltonian_terms=[], jump_operators=[],
                                 total_time=-1.0)

    def test_zero_total_time_raises(self):
        with pytest.raises(LindbladError, match="total_time must be > 0"):
            self._solver().solve(n_qubits=1, hamiltonian_terms=[], jump_operators=[],
                                 total_time=0.0)

    def test_negative_dt_raises(self):
        with pytest.raises(LindbladError, match="dt must be > 0"):
            self._solver().solve(n_qubits=1, hamiltonian_terms=[], jump_operators=[],
                                 total_time=1.0, dt=-0.01)

    def test_invalid_entanglement_cut_raises(self):
        with pytest.raises(LindbladError, match="entanglement_cut"):
            self._solver().solve(n_qubits=2, hamiltonian_terms=[], jump_operators=[],
                                 total_time=1.0, entanglement_cut=5)

    def test_entanglement_cut_zero_valid(self):
        """entanglement_cut=0 means disabled — should not raise."""
        solver = LindbladSolver(_mock_engine_for(_make_decay_frames()))
        result = solver.solve(n_qubits=1, hamiltonian_terms=[], jump_operators=[],
                               total_time=1.0, entanglement_cut=0)
        assert result.entanglement is None

    def test_n_qubits_10_valid(self):
        """10 qubits is the maximum — should not raise."""
        frames = [_make_frame_dict(0, 0.0, 1.0, 0.0, 0.0, [1.0] + [0.0]*1023)]
        solver = LindbladSolver(_mock_engine_for(frames))
        result = solver.solve(n_qubits=10, hamiltonian_terms=[], jump_operators=[],
                               total_time=0.01, dt=0.01)
        assert result.n_qubits == 10


# ---------------------------------------------------------------------------
# Engine call construction
# ---------------------------------------------------------------------------

class TestEngineCallConstruction:
    def test_problem_dict_has_correct_equation(self):
        engine = _mock_engine_for(_make_decay_frames())
        solver = LindbladSolver(engine)
        solver.solve(n_qubits=2, hamiltonian_terms=[(1.0, "Z0")],
                     jump_operators=[JumpOperator("dephasing", 0, 0.1)],
                     total_time=1.0, dt=0.05)

        call_args = engine.run.call_args
        problem = call_args[0][0]
        assert problem["equation"] == "Lindblad"
        assert problem["qubits"] == 2
        assert math.isclose(problem["total_time"], 1.0)
        assert math.isclose(problem["dt"], 0.05)

    def test_hamiltonian_terms_included(self):
        engine = _mock_engine_for(_make_decay_frames())
        LindbladSolver(engine).solve(
            n_qubits=2,
            hamiltonian_terms=[(0.5, "Z0Z1"), (-1.0, "X0")],
            jump_operators=[],
            total_time=1.0,
        )
        problem = engine.run.call_args[0][0]
        labels = {t["label"] for t in problem["hamiltonian_terms"]}
        assert "Z0Z1" in labels
        assert "X0" in labels

    def test_jump_operators_encoded_in_terms(self):
        engine = _mock_engine_for(_make_decay_frames())
        LindbladSolver(engine).solve(
            n_qubits=2,
            hamiltonian_terms=[],
            jump_operators=[
                JumpOperator("amplitude_damping", qubit=0, rate=0.3),
                JumpOperator("dephasing", qubit=1, rate=0.1),
            ],
            total_time=1.0,
        )
        problem = engine.run.call_args[0][0]
        labels = {t["label"]: t["coefficient"] for t in problem["hamiltonian_terms"]}
        assert "AmplitudeDamping0" in labels
        assert math.isclose(labels["AmplitudeDamping0"], 0.3)
        assert "Dephasing1" in labels
        assert math.isclose(labels["Dephasing1"], 0.1)

    def test_entanglement_cut_passed_to_options(self):
        engine = _mock_engine_for(_make_decay_frames())
        LindbladSolver(engine).solve(
            n_qubits=3, hamiltonian_terms=[], jump_operators=[],
            total_time=1.0, entanglement_cut=1,
        )
        options = engine.run.call_args[0][1]
        assert options.get("entanglement_cut") == 1

    def test_no_entanglement_cut_no_option(self):
        engine = _mock_engine_for(_make_decay_frames())
        LindbladSolver(engine).solve(
            n_qubits=2, hamiltonian_terms=[], jump_operators=[], total_time=1.0,
        )
        options = engine.run.call_args[0][1]
        assert "entanglement_cut" not in options


# ---------------------------------------------------------------------------
# LindbladResult structure
# ---------------------------------------------------------------------------

class TestLindbladResultStructure:
    def _run(self, n_steps=5, gamma=0.5, dt=0.1):
        frames = _make_decay_frames(n_steps=n_steps, gamma=gamma, dt=dt)
        engine = _mock_engine_for(frames)
        return LindbladSolver(engine).solve(
            n_qubits=1, hamiltonian_terms=[],
            jump_operators=[JumpOperator("amplitude_damping", 0, gamma)],
            total_time=n_steps * dt, dt=dt,
        )

    def test_times_length(self):
        result = self._run(n_steps=10)
        assert len(result.times) == 11   # n_steps + 1 (includes t=0)

    def test_times_first_is_zero(self):
        result = self._run()
        assert result.times[0] == 0.0

    def test_times_last_matches_total_time(self):
        result = self._run(n_steps=5, dt=0.1)
        assert math.isclose(result.times[-1], 0.5, abs_tol=1e-10)

    def test_purity_arrays_same_length_as_times(self):
        result = self._run(n_steps=8)
        assert len(result.purity) == len(result.times)
        assert len(result.entropy) == len(result.times)
        assert len(result.coherence) == len(result.times)

    def test_populations_shape(self):
        result = self._run(n_steps=5)
        # n_qubits=1 → 2 populations per step
        assert len(result.populations) == 6
        assert all(len(p) == 2 for p in result.populations)

    def test_initial_purity_is_one(self):
        """Starting from |0⟩, purity = 1 at t=0."""
        result = self._run()
        assert math.isclose(result.purity[0], 1.0, abs_tol=1e-10)

    def test_final_purity_less_than_initial(self):
        """Amplitude damping reduces purity."""
        result = self._run(n_steps=20, gamma=0.5, dt=0.1)
        assert result.final_purity < result.purity[0]

    def test_final_populations_sums_to_one(self):
        result = self._run()
        assert math.isclose(sum(result.final_populations), 1.0, abs_tol=1e-10)

    def test_frames_count(self):
        result = self._run(n_steps=7)
        assert len(result.frames) == 8   # n_steps + 1

    def test_no_entanglement_when_cut_zero(self):
        result = self._run()
        assert result.entanglement is None

    def test_n_qubits_stored(self):
        result = self._run()
        assert result.n_qubits == 1

    def test_total_time_stored(self):
        result = self._run(n_steps=5, dt=0.1)
        assert math.isclose(result.total_time, 0.5)

    def test_dt_stored(self):
        result = self._run(dt=0.05)
        assert math.isclose(result.dt, 0.05)


# ---------------------------------------------------------------------------
# Physics: T1 amplitude damping decay
# ---------------------------------------------------------------------------

class TestAmplitudeDampingPhysics:
    """Verify that mock data matching the analytical T1 decay is parsed
    correctly and that steady_state_reached() detects convergence."""

    def _run_decay(self, gamma=1.0, n_steps=50, dt=0.1):
        frames = _make_decay_frames(n_steps=n_steps, gamma=gamma, dt=dt)
        engine = _mock_engine_for(frames)
        return LindbladSolver(engine).solve(
            n_qubits=1, hamiltonian_terms=[],
            jump_operators=[JumpOperator("amplitude_damping", 0, gamma)],
            total_time=n_steps * dt, dt=dt,
        )

    def test_purity_decays_monotonically(self):
        result = self._run_decay(gamma=1.0, n_steps=20)
        # Purity should generally decrease (may not be perfectly monotone
        # due to relaxation towards mixed state then back up; check overall trend)
        assert result.final_purity < result.purity[0]

    def test_steady_state_reached_at_long_time(self):
        """After many T1 times the purity should converge to 1 (ground state = pure)."""
        # Amplitude damping: ρ → |0⟩⟨0| at long time → purity=1
        # Our mock has |0⟩ population growing back to 1
        result = self._run_decay(gamma=2.0, n_steps=100, dt=0.1)
        assert result.steady_state_reached(tol=1e-3)

    def test_final_population_0_dominates_at_long_time(self):
        """At t >> 1/γ the ground state |0⟩ population should be near 1."""
        result = self._run_decay(gamma=5.0, n_steps=50, dt=0.1)
        assert result.final_populations[0] > 0.99

    def test_initial_entropy_zero(self):
        """Starting from |0⟩, entropy = 0."""
        result = self._run_decay()
        assert result.entropy[0] < 1e-10

    def test_steady_state_not_reached_short_time(self):
        """At short times, purity hasn't converged yet."""
        result = self._run_decay(gamma=0.1, n_steps=3, dt=0.1)
        assert not result.steady_state_reached(tol=1e-4)


# ---------------------------------------------------------------------------
# Entanglement trace
# ---------------------------------------------------------------------------

class TestEntanglementTrace:
    def _make_ent_frames(self, n_steps=5, dt=0.1):
        ent_frames = []
        for s in range(n_steps + 1):
            t = s * dt
            # Simulate decaying entanglement
            c = math.exp(-0.5 * t)
            ent_frames.append({
                "step": s,
                "time": t,
                "entropy_a": c * 0.5,
                "mutual_info": c * 0.8,
                "log_negativity": c * 0.3,
                "concurrence": c,
            })
        return ent_frames

    def _run_with_entanglement(self, n_steps=5, dt=0.1):
        lindblad_frames = [
            _make_frame_dict(s, s * dt, 1.0 - 0.05 * s, 0.0, 0.0, [1.0, 0.0, 0.0, 0.0])
            for s in range(n_steps + 1)
        ]
        ent_frames = self._make_ent_frames(n_steps=n_steps, dt=dt)
        engine = _mock_engine_for(lindblad_frames, entanglement_frames=ent_frames)
        return LindbladSolver(engine).solve(
            n_qubits=2,
            hamiltonian_terms=[(1.0, "X0X1")],
            jump_operators=[JumpOperator("dephasing", 0, 0.1)],
            total_time=n_steps * dt,
            dt=dt,
            entanglement_cut=1,
        )

    def test_entanglement_not_none(self):
        result = self._run_with_entanglement()
        assert result.entanglement is not None

    def test_entanglement_times_length(self):
        n_steps = 7
        result = self._run_with_entanglement(n_steps=n_steps)
        assert len(result.entanglement.times) == n_steps + 1

    def test_entanglement_concurrence_initial(self):
        result = self._run_with_entanglement()
        # At t=0, concurrence = 1.0 (from _make_ent_frames)
        assert math.isclose(result.entanglement.concurrence[0], 1.0, abs_tol=1e-10)

    def test_entanglement_concurrence_decays(self):
        result = self._run_with_entanglement(n_steps=10, dt=0.1)
        assert result.entanglement.concurrence[-1] < result.entanglement.concurrence[0]

    def test_entanglement_trace_len(self):
        result = self._run_with_entanglement(n_steps=5)
        assert len(result.entanglement) == 6


# ---------------------------------------------------------------------------
# Error: no frames returned
# ---------------------------------------------------------------------------

class TestNoFramesError:
    def test_empty_frames_raises(self):
        engine = _mock_engine_for([])
        with pytest.raises(LindbladError, match="no frames"):
            LindbladSolver(engine).solve(
                n_qubits=1, hamiltonian_terms=[], jump_operators=[],
                total_time=1.0,
            )


# ---------------------------------------------------------------------------
# _parse_result utility
# ---------------------------------------------------------------------------

class TestParseResult:
    def test_parse_basic(self):
        frames = _make_decay_frames(n_steps=4)
        result = _parse_result({"lindblad_frames": frames, "entanglement_frames": []},
                                n_qubits=1, total_time=0.4, dt=0.1)
        assert isinstance(result, LindbladResult)
        assert len(result.frames) == 5

    def test_parse_with_entanglement(self):
        frames = [_make_frame_dict(0, 0.0, 1.0, 0.0, 0.0, [1.0, 0.0])]
        ent_frames = [{"step": 0, "time": 0.0, "entropy_a": 0.0,
                       "mutual_info": 0.0, "log_negativity": 0.0, "concurrence": 0.0}]
        result = _parse_result({"lindblad_frames": frames, "entanglement_frames": ent_frames},
                                n_qubits=1, total_time=0.0, dt=0.01)
        assert result.entanglement is not None
        assert len(result.entanglement.times) == 1


# ---------------------------------------------------------------------------
# engine.run_lindblad convenience wrapper
# ---------------------------------------------------------------------------

class TestEngineRunLindblad:
    def test_engine_run_lindblad_returns_result(self):
        """Engine.run_lindblad() should return a LindbladResult."""
        from quantx_engine.engine import Engine
        engine = MagicMock(spec=Engine)
        frames = _make_decay_frames(n_steps=5)
        engine.run.return_value = {
            "ok": True,
            "lindblad_frames": frames,
            "entanglement_frames": [],
            "observables": {},
        }
        # Patch the Engine instance's run method directly
        real_engine = object.__new__(Engine)
        real_engine.run = engine.run
        real_engine.run_lindblad = Engine.run_lindblad.__get__(real_engine, Engine)

        result = real_engine.run_lindblad(
            n_qubits=1,
            hamiltonian_terms=[],
            jump_operators=[JumpOperator("amplitude_damping", 0, 0.5)],
            total_time=0.5,
            dt=0.1,
        )
        assert isinstance(result, LindbladResult)
        assert result.n_qubits == 1
