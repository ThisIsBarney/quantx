"""M117: Tests for Jupyter interactive widgets (headless mode)."""
from __future__ import annotations

import math
import pytest
import numpy as np

from quantx_engine.jupyter_widgets import (
    WidgetError,
    CircuitWidgetState,
    CircuitWidget,
    BlochSphereWidgetState,
    BlochSphereWidget,
    JobMonitorWidgetState,
    JobMonitorWidget,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _zero_sv() -> np.ndarray:
    return np.array([1.0, 0.0], dtype=complex)


def _one_sv() -> np.ndarray:
    return np.array([0.0, 1.0], dtype=complex)


def _plus_sv() -> np.ndarray:
    return np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)


def _make_evolution(n_steps: int = 10) -> list[np.ndarray]:
    """Rotation from |0⟩ to |1⟩ in n_steps."""
    result = []
    for i in range(n_steps):
        theta = math.pi * i / (n_steps - 1)
        sv = np.array([math.cos(theta / 2), math.sin(theta / 2)], dtype=complex)
        result.append(sv)
    return result


# ── CircuitWidgetState ────────────────────────────────────────────────────────

class TestCircuitWidgetState:
    def test_default_gates(self):
        state = CircuitWidgetState(n_qubits=3, n_moments=5)
        assert "H" in state.available_gates
        assert "CX" in state.available_gates

    def test_add_gate(self):
        state = CircuitWidgetState(n_qubits=2, n_moments=4)
        state.add_gate("H", qubit=0, moment=0)
        assert len(state.gates) == 1
        assert state.gates[0]["gate"] == "H"

    def test_add_gate_out_of_range_qubit(self):
        state = CircuitWidgetState(n_qubits=2, n_moments=4)
        with pytest.raises(WidgetError):
            state.add_gate("H", qubit=2, moment=0)

    def test_add_gate_out_of_range_moment(self):
        state = CircuitWidgetState(n_qubits=2, n_moments=4)
        with pytest.raises(WidgetError):
            state.add_gate("H", qubit=0, moment=4)

    def test_add_unknown_gate_raises(self):
        state = CircuitWidgetState(n_qubits=2, n_moments=4)
        with pytest.raises(WidgetError):
            state.add_gate("UNKNOWN", qubit=0, moment=0)

    def test_remove_gate(self):
        state = CircuitWidgetState(n_qubits=2, n_moments=4)
        state.add_gate("H", qubit=0, moment=0)
        removed = state.remove_gate(qubit=0, moment=0)
        assert removed is True
        assert len(state.gates) == 0

    def test_remove_nonexistent_gate(self):
        state = CircuitWidgetState(n_qubits=2, n_moments=4)
        removed = state.remove_gate(qubit=0, moment=0)
        assert removed is False

    def test_clear(self):
        state = CircuitWidgetState(n_qubits=3, n_moments=5)
        state.add_gate("H", 0, 0)
        state.add_gate("X", 1, 1)
        state.clear()
        assert len(state.gates) == 0

    def test_to_circuit_ops_sorted_by_moment(self):
        state = CircuitWidgetState(n_qubits=3, n_moments=5)
        state.add_gate("X", 0, 2)
        state.add_gate("H", 1, 0)
        state.add_gate("Z", 2, 1)
        ops = state.to_circuit_ops()
        moments = [g["moment"] for g in ops]
        assert moments == sorted(moments)

    def test_to_dict(self):
        state = CircuitWidgetState(n_qubits=2, n_moments=3)
        state.add_gate("H", 0, 0)
        d = state.to_dict()
        assert "n_qubits" in d
        assert "gates" in d
        assert d["n_qubits"] == 2


# ── CircuitWidget ─────────────────────────────────────────────────────────────

class TestCircuitWidget:
    def test_default_init(self):
        w = CircuitWidget()
        assert w.state.n_qubits == 4
        assert w.state.n_moments == 8

    def test_explicit_init(self):
        w = CircuitWidget(n_qubits=2, n_moments=4)
        assert w.state.n_qubits == 2
        assert w.state.n_moments == 4

    def test_zero_qubits_raises(self):
        with pytest.raises(WidgetError):
            CircuitWidget(n_qubits=0)

    def test_zero_moments_raises(self):
        with pytest.raises(WidgetError):
            CircuitWidget(n_moments=0)

    def test_add_gate_chainable(self):
        w = CircuitWidget(n_qubits=3, n_moments=4)
        w.add_gate("H", 0, 0).add_gate("X", 1, 1)
        assert len(w.state.gates) == 2

    def test_remove_gate(self):
        w = CircuitWidget(n_qubits=2, n_moments=3)
        w.add_gate("H", 0, 0)
        removed = w.remove_gate(0, 0)
        assert removed is True
        assert len(w.state.gates) == 0

    def test_clear(self):
        w = CircuitWidget(n_qubits=3, n_moments=4)
        w.add_gate("H", 0, 0).add_gate("X", 1, 0)
        w.clear()
        assert len(w.state.gates) == 0

    def test_export_circuit(self):
        w = CircuitWidget(n_qubits=2, n_moments=4)
        w.add_gate("H", 0, 0).add_gate("CX", 1, 1)
        ops = w.export_circuit()
        assert len(ops) == 2
        gates = [o["gate"] for o in ops]
        assert "H" in gates and "CX" in gates

    def test_custom_gate_palette(self):
        w = CircuitWidget(n_qubits=2, available_gates=["RX", "RY"])
        w.add_gate("RX", 0, 0)
        assert w.state.gates[0]["gate"] == "RX"

    def test_on_change_callback(self):
        changes = []
        w = CircuitWidget(n_qubits=2, on_change=lambda s: changes.append(len(s.gates)))
        w.add_gate("H", 0, 0)
        w.add_gate("X", 1, 1)
        assert changes == [1, 2]

    def test_repr(self):
        w = CircuitWidget(n_qubits=3, n_moments=5)
        assert "CircuitWidget" in repr(w)
        assert "n_qubits=3" in repr(w)

    def test_show_raises_without_ipywidgets(self, monkeypatch):
        """show() should raise WidgetError when ipywidgets not installed."""
        import sys
        monkeypatch.setitem(sys.modules, "ipywidgets", None)
        w = CircuitWidget(n_qubits=2)
        with pytest.raises((WidgetError, ImportError, TypeError)):
            w.show()


# ── BlochSphereWidget helpers ─────────────────────────────────────────────────

class TestBlochCoords:
    def test_zero_state_is_north_pole(self):
        """⟨Z⟩=1, so z=1 (north pole) for |0⟩."""
        from quantx_engine.jupyter_widgets import _bloch_coords_from_sv
        x, y, z = _bloch_coords_from_sv(_zero_sv())
        assert abs(z - 1.0) < 1e-10
        assert abs(x) < 1e-10
        assert abs(y) < 1e-10

    def test_one_state_is_south_pole(self):
        from quantx_engine.jupyter_widgets import _bloch_coords_from_sv
        x, y, z = _bloch_coords_from_sv(_one_sv())
        assert abs(z + 1.0) < 1e-10

    def test_plus_state_on_equator(self):
        """|+⟩ lies on the equator, x=1."""
        from quantx_engine.jupyter_widgets import _bloch_coords_from_sv
        x, y, z = _bloch_coords_from_sv(_plus_sv())
        assert abs(x - 1.0) < 1e-10
        assert abs(z) < 1e-10

    def test_wrong_shape_raises(self):
        from quantx_engine.jupyter_widgets import _bloch_coords_from_sv
        with pytest.raises(WidgetError):
            _bloch_coords_from_sv(np.array([1.0, 0.0, 0.0]))


# ── BlochSphereWidget ─────────────────────────────────────────────────────────

class TestBlochSphereWidget:
    def test_default_init(self):
        w = BlochSphereWidget()
        # Default: |0⟩ → north pole
        assert abs(w.state.bloch_z - 1.0) < 1e-10

    def test_explicit_sv_init(self):
        w = BlochSphereWidget(_zero_sv())
        assert abs(w.state.bloch_z - 1.0) < 1e-10

    def test_one_state_init(self):
        w = BlochSphereWidget(_one_sv())
        assert abs(w.state.bloch_z + 1.0) < 1e-10

    def test_plus_state_init(self):
        w = BlochSphereWidget(_plus_sv())
        x, y, z = w.bloch_coordinates()
        assert abs(x - 1.0) < 1e-10

    def test_wrong_shape_raises(self):
        with pytest.raises(WidgetError):
            BlochSphereWidget(np.array([1.0, 0.0, 0.0]))

    def test_update(self):
        w = BlochSphereWidget(_zero_sv())
        w.update(_one_sv())
        assert abs(w.state.bloch_z + 1.0) < 1e-10

    def test_update_wrong_shape_raises(self):
        w = BlochSphereWidget()
        with pytest.raises(WidgetError):
            w.update(np.array([1.0, 0.0, 0.0]))

    def test_bloch_coordinates(self):
        w = BlochSphereWidget(_zero_sv())
        coords = w.bloch_coordinates()
        assert len(coords) == 3

    def test_angles_north_pole(self):
        w = BlochSphereWidget(_zero_sv())
        theta, phi = w.angles()
        assert abs(theta) < 1e-10  # θ=0 at north pole

    def test_angles_south_pole(self):
        w = BlochSphereWidget(_one_sv())
        theta, phi = w.angles()
        assert abs(theta - math.pi) < 1e-10

    def test_on_change_callback(self):
        states = []
        w = BlochSphereWidget(on_change=lambda s: states.append(s.bloch_z))
        w.update(_one_sv())
        assert abs(states[-1] + 1.0) < 1e-10

    def test_normalised_init(self):
        """Unnormalised statevectors should be normalised internally."""
        sv = np.array([2.0, 0.0], dtype=complex)
        w = BlochSphereWidget(sv)
        assert abs(w.state.bloch_z - 1.0) < 1e-10

    def test_evolution_init(self):
        evo = _make_evolution(5)
        w = BlochSphereWidget(evolution=evo)
        assert w._evolution is not None
        assert len(w._evolution) == 5

    def test_evolution_wrong_shape_raises(self):
        with pytest.raises(WidgetError):
            BlochSphereWidget(
                evolution=[np.array([1.0, 0.0, 0.0])]
            )

    def test_to_dict(self):
        w = BlochSphereWidget(_plus_sv())
        d = w.state.to_dict()
        assert "bloch_x" in d
        assert "bloch_y" in d
        assert "bloch_z" in d
        assert "theta" in d

    def test_repr(self):
        w = BlochSphereWidget()
        assert "BlochSphereWidget" in repr(w)

    def test_show_raises_without_ipywidgets(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "ipywidgets", None)
        w = BlochSphereWidget()
        with pytest.raises((WidgetError, ImportError, TypeError)):
            w.show()


# ── JobMonitorWidgetState ─────────────────────────────────────────────────────

class TestJobMonitorWidgetState:
    def test_initial_state(self):
        state = JobMonitorWidgetState(job_id="abc123")
        assert state.status == "queued"
        assert state.progress == 0.0
        assert state.poll_count == 0
        assert not state.is_terminal

    def test_is_terminal_done(self):
        state = JobMonitorWidgetState(job_id="x", status="done")
        assert state.is_terminal

    def test_is_terminal_failed(self):
        state = JobMonitorWidgetState(job_id="x", status="failed")
        assert state.is_terminal

    def test_is_terminal_running(self):
        state = JobMonitorWidgetState(job_id="x", status="running")
        assert not state.is_terminal

    def test_to_dict(self):
        state = JobMonitorWidgetState(job_id="j1", status="running", progress=0.5)
        d = state.to_dict()
        assert d["job_id"] == "j1"
        assert d["status"] == "running"
        assert d["progress"] == 0.5


# ── JobMonitorWidget ──────────────────────────────────────────────────────────

class TestJobMonitorWidget:
    def test_default_init(self):
        w = JobMonitorWidget(job_id="job-1")
        assert w.state.job_id == "job-1"
        assert w.state.status == "queued"

    def test_empty_job_id_raises(self):
        with pytest.raises(WidgetError):
            JobMonitorWidget(job_id="")

    def test_negative_interval_raises(self):
        with pytest.raises(WidgetError):
            JobMonitorWidget(job_id="j1", poll_interval_s=-1.0)

    def test_zero_interval_raises(self):
        with pytest.raises(WidgetError):
            JobMonitorWidget(job_id="j1", poll_interval_s=0.0)

    def test_manual_update(self):
        w = JobMonitorWidget(job_id="j1")
        w.update("running", progress=0.3, message="Processing…")
        assert w.state.status == "running"
        assert abs(w.state.progress - 0.3) < 1e-10
        assert w.state.poll_count == 1

    def test_update_progress_out_of_range_raises(self):
        w = JobMonitorWidget(job_id="j1")
        with pytest.raises(WidgetError):
            w.update("running", progress=1.5)

    def test_update_chainable(self):
        w = JobMonitorWidget(job_id="j1")
        w.update("running", 0.5).update("done", 1.0)
        assert w.state.status == "done"
        assert w.state.poll_count == 2

    def test_poll_without_fn_returns_none(self):
        w = JobMonitorWidget(job_id="j1")
        result = w.poll()
        assert result is None

    def test_poll_with_fn(self):
        poll_responses = [
            {"status": "running", "progress": 0.5, "message": "halfway"},
            {"status": "done", "progress": 1.0},
        ]
        call_count = [0]

        def mock_poll(job_id):
            resp = poll_responses[min(call_count[0], len(poll_responses) - 1)]
            call_count[0] += 1
            return resp

        w = JobMonitorWidget(job_id="j1", poll_fn=mock_poll)
        state = w.poll()
        assert state is not None
        assert state.status == "running"
        assert abs(state.progress - 0.5) < 1e-10

    def test_poll_fn_increments_poll_count(self):
        w = JobMonitorWidget(
            job_id="j1",
            poll_fn=lambda _: {"status": "running", "progress": 0.1},
        )
        w.poll()
        w.poll()
        assert w.state.poll_count == 2

    def test_on_change_callback(self):
        statuses = []
        w = JobMonitorWidget(
            job_id="j1",
            poll_fn=lambda _: {"status": "done", "progress": 1.0},
            on_change=lambda s: statuses.append(s.status),
        )
        w.poll()
        assert "done" in statuses

    def test_is_terminal_after_done(self):
        w = JobMonitorWidget(job_id="j1")
        w.update("done", 1.0)
        assert w.state.is_terminal

    def test_repr(self):
        w = JobMonitorWidget(job_id="my-job-123")
        r = repr(w)
        assert "JobMonitorWidget" in r
        assert "my-job-123" in r

    def test_show_raises_without_ipywidgets(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "ipywidgets", None)
        w = JobMonitorWidget(job_id="j1")
        with pytest.raises((WidgetError, ImportError, TypeError)):
            w.show()


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration:
    def test_circuit_widget_full_flow(self):
        """Build a small circuit and export it."""
        w = CircuitWidget(n_qubits=3, n_moments=5)
        w.add_gate("H", 0, 0)
        w.add_gate("CX", 1, 1)
        w.add_gate("X", 2, 2)
        ops = w.export_circuit()
        assert len(ops) == 3
        assert ops[0]["gate"] == "H"

    def test_bloch_sphere_evolution_tracking(self):
        """Track Bloch coordinates through an evolution."""
        evo = _make_evolution(5)
        w = BlochSphereWidget(evolution=evo)
        # Start: |0⟩ → z≈1
        assert abs(w.state.bloch_z - 1.0) < 1e-10
        # Update to |1⟩
        w.update(evo[-1])
        assert abs(w.state.bloch_z + 1.0) < 0.05  # approx south pole

    def test_job_monitor_lifecycle(self):
        """Simulate a job lifecycle queued → running → done."""
        responses = [
            {"status": "queued", "progress": 0.0},
            {"status": "running", "progress": 0.5},
            {"status": "done", "progress": 1.0},
        ]
        idx = [0]

        def poll_fn(job_id):
            r = responses[min(idx[0], len(responses) - 1)]
            idx[0] += 1
            return r

        w = JobMonitorWidget(job_id="job-xyz", poll_fn=poll_fn)
        for _ in range(3):
            w.poll()
        assert w.state.is_terminal
        assert w.state.status == "done"
        assert abs(w.state.progress - 1.0) < 1e-10
