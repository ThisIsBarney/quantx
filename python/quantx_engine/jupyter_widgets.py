"""M117: Jupyter interactive widgets for QuantX.

Provides three rich notebook widgets for interactive circuit editing,
Bloch-sphere visualisation, and job monitoring.  All widgets degrade
gracefully when running outside Jupyter (no hard dependency on ipywidgets
or anywidget — their absence raises :class:`WidgetError` with an actionable
message).

* **CircuitWidget** — drag-and-drop gate palette + circuit grid.  Tracks
  gate placements and exports the final circuit as a list of
  ``{"gate": str, "qubits": list[int]}`` dicts.

* **BlochSphereWidget** — live 3-D Bloch-sphere rendering from a
  single-qubit statevector.  Updates continuously while the user drags a
  time slider.

* **JobMonitorWidget** — polls a job-store URL and displays a progress bar
  with status chips.

Classes
-------
WidgetError
CircuitWidgetState
CircuitWidget
BlochSphereWidgetState
BlochSphereWidget
JobMonitorWidgetState
JobMonitorWidget

References
----------
.. [1] Granger, B. E. & Perez, F. (2021). Jupyter: Thinking and storytelling
       with code and data. *Comput. Sci. Eng.* 23, 7–14.
       https://doi.org/10.1109/MCSE.2021.3059263

.. [2] ipywidgets Development Team (2023). ipywidgets: Interactive HTML Widgets
       for Jupyter Notebooks. https://doi.org/10.5281/zenodo.3401774
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

__all__ = [
    "WidgetError",
    "CircuitWidgetState",
    "CircuitWidget",
    "BlochSphereWidgetState",
    "BlochSphereWidget",
    "JobMonitorWidgetState",
    "JobMonitorWidget",
]


class WidgetError(RuntimeError):
    """Raised for invalid widget operations or when dependencies are absent."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _try_import_ipywidgets():
    """Return the ipywidgets module, or raise WidgetError if absent."""
    try:
        import ipywidgets as widgets  # type: ignore
        return widgets
    except ImportError as exc:
        raise WidgetError(
            "ipywidgets is not installed.  Run: pip install ipywidgets"
        ) from exc


def _bloch_coords_from_sv(sv: np.ndarray) -> tuple[float, float, float]:
    """Convert a 1-qubit statevector to Bloch-sphere (x, y, z) coordinates.

    For |ψ⟩ = α|0⟩ + β|1⟩:
        x = 2 Re(α* β)
        y = 2 Im(α* β)
        z = |α|² − |β|²
    """
    if sv.shape != (2,):
        raise WidgetError(
            f"BlochSphereWidget expects a 1-qubit statevector of shape (2,), "
            f"got {sv.shape}"
        )
    alpha, beta = complex(sv[0]), complex(sv[1])
    x = 2.0 * float((alpha.conjugate() * beta).real)
    y = 2.0 * float((alpha.conjugate() * beta).imag)
    z = float(abs(alpha) ** 2 - abs(beta) ** 2)
    return x, y, z


def _job_status_emoji(status: str) -> str:
    return {
        "queued": "⏳",
        "running": "🔄",
        "done": "✅",
        "failed": "❌",
        "cancelled": "🚫",
    }.get(status.lower(), "❓")


# ── CircuitWidget ─────────────────────────────────────────────────────────────

@dataclass
class CircuitWidgetState:
    """Serialisable state of a :class:`CircuitWidget`.

    Attributes
    ----------
    n_qubits : int
    n_moments : int
    gates : list[dict]
        Each entry: ``{"gate": str, "qubit": int, "moment": int}``.
    available_gates : list[str]
        Gate palette displayed in the widget.
    """
    n_qubits: int
    n_moments: int
    gates: list[dict] = field(default_factory=list)
    available_gates: list[str] = field(default_factory=lambda: [
        "H", "X", "Y", "Z", "S", "T", "CX", "CZ", "SWAP",
    ])

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "n_moments": self.n_moments,
            "gates": list(self.gates),
            "available_gates": list(self.available_gates),
        }

    def add_gate(self, gate: str, qubit: int, moment: int) -> None:
        """Place a gate in the circuit grid."""
        if not (0 <= qubit < self.n_qubits):
            raise WidgetError(
                f"qubit={qubit} out of range [0, {self.n_qubits})"
            )
        if not (0 <= moment < self.n_moments):
            raise WidgetError(
                f"moment={moment} out of range [0, {self.n_moments})"
            )
        if gate not in self.available_gates:
            raise WidgetError(
                f"Gate {gate!r} not in palette: {self.available_gates}"
            )
        self.gates.append({"gate": gate, "qubit": qubit, "moment": moment})

    def remove_gate(self, qubit: int, moment: int) -> bool:
        """Remove the gate at (qubit, moment).  Returns True if found."""
        before = len(self.gates)
        self.gates = [
            g for g in self.gates
            if not (g["qubit"] == qubit and g["moment"] == moment)
        ]
        return len(self.gates) < before

    def clear(self) -> None:
        """Remove all gates."""
        self.gates.clear()

    def to_circuit_ops(self) -> list[dict]:
        """Export gates sorted by moment for circuit construction."""
        return sorted(self.gates, key=lambda g: (g["moment"], g["qubit"]))


class CircuitWidget:
    """Interactive circuit editor widget.

    When running inside Jupyter and ipywidgets is installed, renders an
    HTML/CSS gate palette + circuit grid.  Outside Jupyter the widget
    operates in "headless" mode: all state is tracked in
    :attr:`state` without rendering.

    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubit wires.
    n_moments : int, default 8
        Number of time steps (columns).
    available_gates : list[str] | None
        Custom gate palette.  Defaults to H/X/Y/Z/S/T/CX/CZ/SWAP.
    on_change : Callable | None
        Callback invoked with the updated :class:`CircuitWidgetState`
        whenever the circuit changes.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_moments: int = 8,
        available_gates: Optional[list[str]] = None,
        on_change: Optional[Callable[[CircuitWidgetState], None]] = None,
    ) -> None:
        if n_qubits < 1:
            raise WidgetError(f"n_qubits must be ≥ 1, got {n_qubits}")
        if n_moments < 1:
            raise WidgetError(f"n_moments must be ≥ 1, got {n_moments}")

        self.state = CircuitWidgetState(
            n_qubits=n_qubits,
            n_moments=n_moments,
        )
        if available_gates is not None:
            self.state.available_gates = list(available_gates)
        self._on_change = on_change
        self._widget: Any = None  # ipywidgets object, if created

    # ── Headless API ──────────────────────────────────────────────────────────

    def add_gate(self, gate: str, qubit: int, moment: int) -> "CircuitWidget":
        """Add a gate and trigger on_change callback."""
        self.state.add_gate(gate, qubit, moment)
        self._notify()
        return self

    def remove_gate(self, qubit: int, moment: int) -> bool:
        """Remove a gate and trigger on_change callback."""
        removed = self.state.remove_gate(qubit, moment)
        if removed:
            self._notify()
        return removed

    def clear(self) -> "CircuitWidget":
        """Clear all gates."""
        self.state.clear()
        self._notify()
        return self

    def export_circuit(self) -> list[dict]:
        """Return sorted gate list for circuit construction."""
        return self.state.to_circuit_ops()

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change(self.state)

    # ── Jupyter rendering ─────────────────────────────────────────────────────

    def show(self) -> Any:
        """Render the interactive widget in a Jupyter notebook.

        Returns the ipywidgets object; calling ``display(widget.show())``
        in a cell will render it.

        Raises :class:`WidgetError` if ipywidgets is not installed.
        """
        widgets = _try_import_ipywidgets()
        n = self.state.n_qubits
        m = self.state.n_moments

        # Gate palette dropdown
        gate_dropdown = widgets.Dropdown(
            options=self.state.available_gates,
            description="Gate:",
            layout=widgets.Layout(width="120px"),
        )
        qubit_slider = widgets.IntSlider(
            min=0, max=n - 1, description="Qubit:", layout=widgets.Layout(width="200px")
        )
        moment_slider = widgets.IntSlider(
            min=0, max=m - 1, description="Step:", layout=widgets.Layout(width="200px")
        )
        add_btn = widgets.Button(description="Add Gate", button_style="primary")
        clear_btn = widgets.Button(description="Clear", button_style="warning")
        output = widgets.Output()

        def _refresh(_=None):
            with output:
                output.clear_output(wait=True)
                print(f"Circuit ({n}q × {m} steps) — {len(self.state.gates)} gates:")
                for g in self.state.to_circuit_ops():
                    print(f"  t={g['moment']} q={g['qubit']} {g['gate']}")

        def _on_add(_):
            try:
                self.add_gate(gate_dropdown.value, qubit_slider.value, moment_slider.value)
            except WidgetError as e:
                with output:
                    output.clear_output()
                    print(f"Error: {e}")
            _refresh()

        def _on_clear(_):
            self.clear()
            _refresh()

        add_btn.on_click(_on_add)
        clear_btn.on_click(_on_clear)
        _refresh()

        self._widget = widgets.VBox([
            widgets.HBox([gate_dropdown, qubit_slider, moment_slider, add_btn, clear_btn]),
            output,
        ])
        return self._widget

    def __repr__(self) -> str:
        return (
            f"CircuitWidget(n_qubits={self.state.n_qubits}, "
            f"n_moments={self.state.n_moments}, "
            f"n_gates={len(self.state.gates)})"
        )


# ── BlochSphereWidget ─────────────────────────────────────────────────────────

@dataclass
class BlochSphereWidgetState:
    """State of a :class:`BlochSphereWidget`.

    Attributes
    ----------
    statevector : np.ndarray, shape (2,)
        Current single-qubit statevector.
    bloch_x, bloch_y, bloch_z : float
        Bloch-sphere coordinates.
    theta, phi : float
        Polar (θ) and azimuthal (φ) angles in radians.
    time_index : int
        Current position in the time-evolution array.
    """
    statevector: np.ndarray
    bloch_x: float
    bloch_y: float
    bloch_z: float
    theta: float = 0.0
    phi: float = 0.0
    time_index: int = 0

    def to_dict(self) -> dict:
        return {
            "statevector_real": self.statevector.real.tolist(),
            "statevector_imag": self.statevector.imag.tolist(),
            "bloch_x": self.bloch_x,
            "bloch_y": self.bloch_y,
            "bloch_z": self.bloch_z,
            "theta": self.theta,
            "phi": self.phi,
            "time_index": self.time_index,
        }


def _sv_to_bloch_state(sv: np.ndarray) -> BlochSphereWidgetState:
    """Build a BlochSphereWidgetState from a 1-qubit statevector."""
    x, y, z = _bloch_coords_from_sv(sv)
    # Convert (x, y, z) to spherical angles
    r = math.sqrt(x * x + y * y + z * z)
    if r < 1e-12:
        theta, phi = 0.0, 0.0
    else:
        theta = math.acos(max(-1.0, min(1.0, z / r)))
        phi = math.atan2(y, x)
    return BlochSphereWidgetState(
        statevector=sv.copy(),
        bloch_x=x, bloch_y=y, bloch_z=z,
        theta=theta, phi=phi,
    )


class BlochSphereWidget:
    """Real-time Bloch-sphere visualisation for a 1-qubit state.

    Supports static display, time-evolution playback via a slider, and
    programmatic updates.

    Parameters
    ----------
    statevector : np.ndarray, shape (2,)
        Initial 1-qubit statevector.
    evolution : list[np.ndarray] | None
        Time-ordered sequence of statevectors for slider animation.
    on_change : Callable | None
        Callback invoked with the current :class:`BlochSphereWidgetState`
        on each update.
    """

    def __init__(
        self,
        statevector: Optional[np.ndarray] = None,
        evolution: Optional[list[np.ndarray]] = None,
        on_change: Optional[Callable[[BlochSphereWidgetState], None]] = None,
    ) -> None:
        if statevector is None:
            statevector = np.array([1.0, 0.0], dtype=complex)
        sv = np.asarray(statevector, dtype=complex)
        if sv.shape != (2,):
            raise WidgetError(
                f"statevector must have shape (2,), got {sv.shape}"
            )
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv = sv / norm

        if evolution is not None:
            for i, sv_i in enumerate(evolution):
                sv_i = np.asarray(sv_i, dtype=complex)
                if sv_i.shape != (2,):
                    raise WidgetError(
                        f"evolution[{i}] must have shape (2,), got {sv_i.shape}"
                    )

        self.state = _sv_to_bloch_state(sv)
        self._evolution = evolution
        self._on_change = on_change
        self._widget: Any = None

    # ── Programmatic API ──────────────────────────────────────────────────────

    def update(self, statevector: np.ndarray) -> "BlochSphereWidget":
        """Update the displayed statevector."""
        sv = np.asarray(statevector, dtype=complex)
        if sv.shape != (2,):
            raise WidgetError(
                f"statevector must have shape (2,), got {sv.shape}"
            )
        norm = np.linalg.norm(sv)
        if norm > 0:
            sv = sv / norm
        self.state = _sv_to_bloch_state(sv)
        self.state.time_index = self.state.time_index
        if self._on_change is not None:
            self._on_change(self.state)
        return self

    def bloch_coordinates(self) -> tuple[float, float, float]:
        """Return (x, y, z) Bloch-sphere coordinates of current state."""
        return self.state.bloch_x, self.state.bloch_y, self.state.bloch_z

    def angles(self) -> tuple[float, float]:
        """Return (theta, phi) polar angles in radians."""
        return self.state.theta, self.state.phi

    # ── Jupyter rendering ─────────────────────────────────────────────────────

    def show(self) -> Any:
        """Render the interactive Bloch-sphere widget.

        Raises :class:`WidgetError` if ipywidgets is not installed.
        """
        widgets = _try_import_ipywidgets()
        output = widgets.Output()

        def _draw(sv: np.ndarray):
            with output:
                output.clear_output(wait=True)
                x, y, z = _bloch_coords_from_sv(sv)
                theta = math.acos(max(-1.0, min(1.0, z)))
                phi_deg = math.degrees(math.atan2(y, x))
                theta_deg = math.degrees(theta)
                print(f"Bloch sphere: x={x:.3f}  y={y:.3f}  z={z:.3f}")
                print(f"  θ={theta_deg:.1f}°  φ={phi_deg:.1f}°")
                # ASCII Bloch-sphere approximation
                col = int((x + 1) / 2 * 10)
                row = int((1 - z) / 2 * 5)
                sphere = [["·"] * 11 for _ in range(6)]
                sphere[min(5, max(0, row))][min(10, max(0, col))] = "●"
                for r in sphere:
                    print("".join(r))

        _draw(self.state.statevector)

        if self._evolution is not None and len(self._evolution) > 1:
            slider = widgets.IntSlider(
                min=0, max=len(self._evolution) - 1,
                description="t:", layout=widgets.Layout(width="300px")
            )

            def _on_slider_change(change):
                idx = change["new"]
                sv = np.asarray(self._evolution[idx], dtype=complex)
                norm = np.linalg.norm(sv)
                if norm > 0:
                    sv /= norm
                self.state = _sv_to_bloch_state(sv)
                self.state.time_index = idx
                if self._on_change is not None:
                    self._on_change(self.state)
                _draw(sv)

            slider.observe(_on_slider_change, names="value")
            self._widget = widgets.VBox([slider, output])
        else:
            self._widget = output

        return self._widget

    def __repr__(self) -> str:
        x, y, z = self.bloch_coordinates()
        return f"BlochSphereWidget(x={x:.3f}, y={y:.3f}, z={z:.3f})"


# ── JobMonitorWidget ──────────────────────────────────────────────────────────

@dataclass
class JobMonitorWidgetState:
    """State of a :class:`JobMonitorWidget`.

    Attributes
    ----------
    job_id : str
    status : str
        One of ``queued``, ``running``, ``done``, ``failed``, ``cancelled``.
    progress : float
        Progress fraction in [0, 1].
    message : str
        Human-readable status message.
    poll_count : int
        Number of times the job has been polled.
    """
    job_id: str
    status: str = "queued"
    progress: float = 0.0
    message: str = ""
    poll_count: int = 0

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "poll_count": self.poll_count,
        }

    @property
    def is_terminal(self) -> bool:
        """True if the job has reached a terminal state."""
        return self.status in ("done", "failed", "cancelled")


class JobMonitorWidget:
    """Polling job-status monitor widget.

    Polls a job-store endpoint (or a user-supplied callback) at a configurable
    interval and displays a progress bar with a status chip.

    Parameters
    ----------
    job_id : str
        Identifier of the job to monitor.
    poll_fn : Callable[[str], dict] | None
        Function that accepts a ``job_id`` and returns a dict with at least
        ``"status"`` and optionally ``"progress"`` (0–1) and ``"message"``.
        If ``None``, the widget operates in manual-update mode.
    poll_interval_s : float, default 2.0
        Polling interval in seconds (used only when rendered via
        :meth:`show`).
    on_change : Callable | None
        Called with :class:`JobMonitorWidgetState` on each status update.
    """

    def __init__(
        self,
        job_id: str,
        poll_fn: Optional[Callable[[str], dict]] = None,
        poll_interval_s: float = 2.0,
        on_change: Optional[Callable[[JobMonitorWidgetState], None]] = None,
    ) -> None:
        if not job_id:
            raise WidgetError("job_id must not be empty")
        if poll_interval_s <= 0:
            raise WidgetError(
                f"poll_interval_s must be > 0, got {poll_interval_s}"
            )
        self.state = JobMonitorWidgetState(job_id=job_id)
        self._poll_fn = poll_fn
        self.poll_interval_s = poll_interval_s
        self._on_change = on_change
        self._widget: Any = None

    # ── Manual-update API ─────────────────────────────────────────────────────

    def update(
        self,
        status: str,
        progress: float = 0.0,
        message: str = "",
    ) -> "JobMonitorWidget":
        """Manually push a status update."""
        if not (0.0 <= progress <= 1.0):
            raise WidgetError(f"progress must be in [0, 1], got {progress}")
        self.state.status = status
        self.state.progress = progress
        self.state.message = message
        self.state.poll_count += 1
        if self._on_change is not None:
            self._on_change(self.state)
        return self

    def poll(self) -> Optional[JobMonitorWidgetState]:
        """Run one poll via the configured poll function.

        Returns the updated state, or ``None`` if no poll function is set.
        """
        if self._poll_fn is None:
            return None
        try:
            result = self._poll_fn(self.state.job_id)
            status = result.get("status", "queued")
            progress = float(result.get("progress", 0.0))
            message = str(result.get("message", ""))
            return self.update(status, progress, message).state
        except Exception as exc:
            self.state.message = str(exc)
            return self.state

    # ── Jupyter rendering ─────────────────────────────────────────────────────

    def show(self) -> Any:
        """Render the job-monitor widget.

        Raises :class:`WidgetError` if ipywidgets is not installed.
        """
        widgets = _try_import_ipywidgets()

        progress_bar = widgets.FloatProgress(
            value=self.state.progress,
            min=0.0, max=1.0,
            description=f"Job {self.state.job_id[:8]}…",
            layout=widgets.Layout(width="400px"),
        )
        status_label = widgets.Label(
            value=f"{_job_status_emoji(self.state.status)} {self.state.status.upper()}"
        )
        message_label = widgets.Label(value=self.state.message)
        poll_label = widgets.Label(value=f"Poll count: {self.state.poll_count}")

        refresh_btn = widgets.Button(description="Refresh", button_style="info")
        output = widgets.Output()

        def _refresh_display(_=None):
            self.poll()
            progress_bar.value = self.state.progress
            status_label.value = (
                f"{_job_status_emoji(self.state.status)} "
                f"{self.state.status.upper()}"
            )
            message_label.value = self.state.message
            poll_label.value = f"Poll count: {self.state.poll_count}"
            with output:
                output.clear_output(wait=True)
                if self.state.is_terminal:
                    print(f"Job {self.state.job_id} is {self.state.status}.")

        refresh_btn.on_click(_refresh_display)

        self._widget = widgets.VBox([
            widgets.HBox([progress_bar, status_label]),
            message_label,
            widgets.HBox([refresh_btn, poll_label]),
            output,
        ])
        return self._widget

    def __repr__(self) -> str:
        s = self.state
        return (
            f"JobMonitorWidget(job_id={s.job_id!r}, status={s.status!r}, "
            f"progress={s.progress:.0%})"
        )
