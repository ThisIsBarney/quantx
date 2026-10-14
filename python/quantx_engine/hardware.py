"""Real hardware backend integration — M93.

Provides abstractions for submitting QuantX circuits to real quantum hardware
and pulling standardised calibration data, without tying callers to any single
vendor SDK.

Classes
-------
* :class:`HardwareError` — base exception for this module.
* :class:`BackendCalibration` — standardised calibration (T1, T2, gate errors,
  coupling map).  Converts to a QuantX :class:`~quantx_engine.noise.NoiseModel`
  and :class:`~quantx_engine.transpiler.CouplingMap`.
* :class:`HardwareJobResult` — completed-job result (counts, probabilities).
* :class:`HardwareJob` — non-blocking async job handle with status polling.
* :class:`IBMHardwareBackend` — extends the existing
  :class:`~quantx_engine.ibm.IBMQuantumBackend` with async submission,
  calibration fetching, and pulse-schedule submission.
* :class:`IonQBackend` — IonQ Cloud REST API (v0.3) backend; uses only
  ``urllib.request`` from the standard library.
* :class:`GoogleBackend` — Google Quantum AI backend via ``cirq-google``
  (lazy import).

Functions
---------
* :func:`calibration_to_noise_model` — convert a :class:`BackendCalibration`
  to a QuantX :class:`~quantx_engine.noise.ThermalRelaxationNoise` (or
  :class:`~quantx_engine.noise.CompositeNoise` when gate errors are large).
* :func:`available_backends` — scan environment variables for configured
  backends and return a summary dict.

References
----------
.. [1] Krantz, P. et al. (2019). A quantum engineer's guide to superconducting
       qubits. *Appl. Phys. Rev.* 6, 021318.
       https://doi.org/10.1063/1.5089550

.. [2] Gambetta, J. M. et al. (2017). Characterization of addressability by
       simultaneous randomized benchmarking.
       *Phys. Rev. Lett.* 109, 240504.
       https://doi.org/10.1103/PhysRevLett.109.240504

.. [3] Arute, F. et al. (2019). Quantum supremacy using a programmable
       superconducting processor. *Nature* 574, 505.
       https://doi.org/10.1038/s41586-019-1666-5
"""
from __future__ import annotations

import json
import math
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

__all__ = [
    "HardwareError",
    "BackendCalibration",
    "HardwareJobResult",
    "HardwareJob",
    "IBMHardwareBackend",
    "IonQBackend",
    "GoogleBackend",
    "calibration_to_noise_model",
    "available_backends",
]

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class HardwareError(RuntimeError):
    """Raised for hardware backend errors (connection, auth, job failure)."""


# ---------------------------------------------------------------------------
# BackendCalibration
# ---------------------------------------------------------------------------

@dataclass
class BackendCalibration:
    """Standardised device calibration snapshot.

    All time constants are in **nanoseconds** (consistent with
    :class:`~quantx_engine.noise.ThermalRelaxationNoise`).

    Parameters
    ----------
    backend_name:
        Human-readable name of the device (e.g. ``"ibm_sherbrooke"``).
    n_qubits:
        Total number of physical qubits.
    t1:
        Per-qubit T1 (longitudinal relaxation) in nanoseconds.
        Keys are qubit indices.  Missing qubits default to ``float("inf")``.
    t2:
        Per-qubit T2 (transverse relaxation) in nanoseconds.
        Keys are qubit indices.  Missing qubits default to ``float("inf")``.
    gate_errors:
        Per-gate-type average error rate (probability), e.g.
        ``{"cx": 0.005, "sx": 0.0003}``.
    coupling_edges:
        List of ``(q0, q1)`` pairs indicating connected qubit pairs.
    gate_times:
        Per-gate-type average duration in nanoseconds, e.g.
        ``{"cx": 350.0, "sx": 35.0}``.
    """

    backend_name: str
    n_qubits: int
    t1: dict[int, float] = field(default_factory=dict)
    t2: dict[int, float] = field(default_factory=dict)
    gate_errors: dict[str, float] = field(default_factory=dict)
    coupling_edges: list[tuple[int, int]] = field(default_factory=list)
    gate_times: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def mean_t1(self) -> float:
        """Average T1 across all qubits with finite T1, in nanoseconds."""
        vals = [v for v in self.t1.values() if math.isfinite(v) and v > 0]
        return float(sum(vals) / len(vals)) if vals else float("inf")

    def mean_t2(self) -> float:
        """Average T2 across all qubits with finite T2, in nanoseconds."""
        vals = [v for v in self.t2.values() if math.isfinite(v) and v > 0]
        return float(sum(vals) / len(vals)) if vals else float("inf")

    def mean_gate_time(self, gate: str = "cx") -> float:
        """Average gate time in nanoseconds for the requested gate type."""
        return float(self.gate_times.get(gate, 50.0))

    def to_noise_model(self, t_gate_ns: float | None = None):
        """Convert calibration to a QuantX
        :class:`~quantx_engine.noise.ThermalRelaxationNoise`.

        Uses the mean T1/T2 over all qubits.  If T2 > 2·T1 (violates the
        Lindblad bound) it is clamped to 2·T1.

        Parameters
        ----------
        t_gate_ns:
            Gate duration in nanoseconds used to compute the per-gate
            damping parameters.  Defaults to the calibration's mean CX time,
            falling back to 50 ns.

        Returns
        -------
        :class:`~quantx_engine.noise.ThermalRelaxationNoise`
        """
        from .noise import ThermalRelaxationNoise  # local import to avoid circulars

        t1 = self.mean_t1()
        t2 = self.mean_t2()
        # Enforce Lindblad bound: T2 ≤ 2·T1
        if math.isfinite(t1) and math.isfinite(t2):
            t2 = min(t2, 2.0 * t1)
        elif not math.isfinite(t2):
            t2 = 2.0 * t1 if math.isfinite(t1) else 1e9
        if not math.isfinite(t1):
            t1 = 1e9   # essentially noiseless

        tg = t_gate_ns if t_gate_ns is not None else self.mean_gate_time("cx")
        return ThermalRelaxationNoise(T1=t1, T2=t2, t_gate=tg)

    def to_coupling_map(self):
        """Convert to a QuantX :class:`~quantx_engine.transpiler.CouplingMap`."""
        from .transpiler import CouplingMap  # local import

        return CouplingMap(n_qubits=self.n_qubits, edges=list(self.coupling_edges))

    def __repr__(self) -> str:
        return (
            f"BackendCalibration(backend={self.backend_name!r}, "
            f"n_qubits={self.n_qubits}, "
            f"mean_T1={self.mean_t1():.0f}ns, mean_T2={self.mean_t2():.0f}ns)"
        )


# ---------------------------------------------------------------------------
# HardwareJobResult
# ---------------------------------------------------------------------------

@dataclass
class HardwareJobResult:
    """Result of a completed hardware job.

    Attributes
    ----------
    job_id:
        Provider-assigned job identifier.
    backend:
        Backend name the job ran on.
    counts:
        Raw bit-string counts, e.g. ``{"00": 512, "11": 488}``.
    probabilities:
        Normalised probabilities, e.g. ``{"00": 0.5, "11": 0.476}``.
    shots:
        Number of shots executed.
    metadata:
        Provider-specific metadata (timing, error codes, etc.).
    """

    job_id: str
    backend: str
    counts: dict[str, int]
    probabilities: dict[str, float]
    shots: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        top = sorted(self.probabilities.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{k}:{v:.3f}" for k, v in top)
        return (
            f"HardwareJobResult(job_id={self.job_id!r}, backend={self.backend!r}, "
            f"shots={self.shots}, top=[{top_str}])"
        )


# ---------------------------------------------------------------------------
# HardwareJob  (async handle)
# ---------------------------------------------------------------------------

class HardwareJob:
    """Non-blocking job handle returned by async ``submit()`` methods.

    Internally holds a callable ``_fetch_fn(job_id) -> HardwareJobResult``
    that the backend supplies; callers never instantiate this directly.

    Parameters
    ----------
    job_id:
        Provider-assigned job identifier.
    backend_name:
        Backend the job was submitted to.
    _status_fn:
        Callable ``(job_id: str) -> str`` returning the current status string
        (``"queued"``, ``"running"``, ``"completed"``, or ``"failed"``).
    _result_fn:
        Callable ``(job_id: str) -> HardwareJobResult`` returning the final
        result (called only when status is ``"completed"``).
    """

    def __init__(
        self,
        job_id: str,
        backend_name: str,
        _status_fn: Callable[[str], str],
        _result_fn: Callable[[str], "HardwareJobResult"],
    ) -> None:
        self._job_id = job_id
        self._backend_name = backend_name
        self._status_fn = _status_fn
        self._result_fn = _result_fn

    @property
    def job_id(self) -> str:
        return self._job_id

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def status(self) -> str:
        """Return current job status: ``"queued"``, ``"running"``,
        ``"completed"``, or ``"failed"``."""
        return self._status_fn(self._job_id)

    def fetch(
        self,
        timeout: float = 300.0,
        poll_interval: float = 2.0,
    ) -> "HardwareJobResult":
        """Block until the job completes and return the result.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait before raising :class:`HardwareError`.
        poll_interval:
            Seconds between status polls.

        Returns
        -------
        HardwareJobResult

        Raises
        ------
        HardwareError
            If the job fails or the timeout expires.
        """
        if timeout <= 0:
            raise HardwareError("timeout must be positive")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            s = self._status_fn(self._job_id)
            if s == "completed":
                return self._result_fn(self._job_id)
            if s == "failed":
                raise HardwareError(
                    f"Job {self._job_id!r} failed on backend {self._backend_name!r}."
                )
            remaining = deadline - time.monotonic()
            time.sleep(min(poll_interval, max(0.0, remaining)))
        raise HardwareError(
            f"Job {self._job_id!r} did not complete within {timeout}s "
            f"(last status: {self._status_fn(self._job_id)!r})."
        )

    def __repr__(self) -> str:
        return (
            f"HardwareJob(job_id={self._job_id!r}, "
            f"backend={self._backend_name!r})"
        )


# ---------------------------------------------------------------------------
# IBMHardwareBackend
# ---------------------------------------------------------------------------

class IBMHardwareBackend:
    """Extended IBM Quantum backend with async submission, calibration fetching,
    and pulse-schedule submission.

    This class wraps :class:`~quantx_engine.ibm.IBMQuantumBackend` and adds:

    * :meth:`submit` — non-blocking circuit submission returning a
      :class:`HardwareJob`.
    * :meth:`fetch_calibration` — pull T1/T2/gate-error calibration from the
      IBM device properties API and return a :class:`BackendCalibration`.
    * :meth:`run_pulse` — submit a QuantX
      :class:`~quantx_engine.pulse.PulseSchedule` to IBM as a native pulse job.
    * :meth:`list_backends` — list available IBM backends for the configured
      instance.

    Parameters
    ----------
    token:
        IBM Quantum API token.  Falls back to ``QISKIT_IBM_TOKEN`` env var.
    instance:
        IBM Quantum instance string (``"hub/group/project"``).
    backend_name:
        Target backend name.
    """

    def __init__(
        self,
        token: str | None = None,
        instance: str = "ibm-q/open/main",
        backend_name: str = "ibmq_qasm_simulator",
    ) -> None:
        from .ibm import IBMQuantumBackend  # local import

        self._base = IBMQuantumBackend(
            token=token, instance=instance, backend_name=backend_name
        )
        self._backend_name = backend_name

    # ------------------------------------------------------------------
    def run_circuit(
        self,
        circuit: dict[str, Any],
        shots: int = 1000,
        optimization_level: int = 1,
    ) -> "HardwareJobResult":
        """Submit and block until completion (synchronous convenience wrapper)."""
        raw = self._base.run_circuit(circuit, shots=shots,
                                     optimization_level=optimization_level)
        return HardwareJobResult(
            job_id=raw.get("ibm_job_id", ""),
            backend=self._backend_name,
            counts=raw.get("counts", {}),
            probabilities=raw.get("probabilities", {}),
            shots=raw.get("shots", shots),
            metadata=raw.get("raw_ibm_result", {}),
        )

    def submit(
        self,
        circuit: dict[str, Any],
        shots: int = 1000,
        optimization_level: int = 1,
    ) -> "HardwareJob":
        """Submit a circuit asynchronously and return a :class:`HardwareJob`.

        The job is submitted immediately via the Qiskit SamplerV2 primitive.
        The returned handle can be polled with :meth:`HardwareJob.fetch`.
        """
        try:
            from qiskit import QuantumCircuit  # type: ignore[import]
            from qiskit.qasm3 import loads as qasm3_loads  # type: ignore[import]
            from qiskit_ibm_runtime import SamplerV2 as Sampler  # type: ignore[import]
            from qiskit.transpiler.preset_passmanagers import (  # type: ignore[import]
                generate_preset_pass_manager,
            )
        except ImportError as exc:
            raise HardwareError(
                "IBMHardwareBackend requires qiskit>=1.0 and "
                "qiskit-ibm-runtime>=0.20."
            ) from exc

        backend_obj = self._base._backend
        if backend_obj is None:
            raise HardwareError(
                "Backend not connected. Provide token= or QISKIT_IBM_TOKEN."
            )
        from .ibm import circuit_to_qasm3, IBMConnectorError

        qasm_src = circuit_to_qasm3(circuit)
        try:
            qc = qasm3_loads(qasm_src)
        except Exception as exc:
            raise HardwareError(f"QASM parse failed: {exc}") from exc

        pm = generate_preset_pass_manager(
            optimization_level=optimization_level, backend=backend_obj
        )
        isa_qc = pm.run(qc)
        sampler = Sampler(mode=backend_obj)
        ibm_job = sampler.run([isa_qc], shots=shots)

        # Capture job_id immediately
        raw_id = getattr(ibm_job, "job_id", "")
        if callable(raw_id):
            try:
                raw_id = ibm_job.job_id()
            except Exception:
                raw_id = ""
        job_id: str = raw_id or ""

        def _status(jid: str) -> str:
            try:
                st = ibm_job.status()
                name = str(st.name).lower() if hasattr(st, "name") else str(st).lower()
                if "done" in name or "complete" in name:
                    return "completed"
                if "error" in name or "cancel" in name or "fail" in name:
                    return "failed"
                if "running" in name:
                    return "running"
                return "queued"
            except Exception:
                return "queued"

        def _result(jid: str) -> HardwareJobResult:
            ibm_res = ibm_job.result()
            raw = self._base._convert_result(ibm_res, shots, job_id=jid)
            return HardwareJobResult(
                job_id=jid,
                backend=self._backend_name,
                counts=raw.get("counts", {}),
                probabilities=raw.get("probabilities", {}),
                shots=raw.get("shots", shots),
                metadata=raw.get("raw_ibm_result", {}),
            )

        return HardwareJob(
            job_id=job_id,
            backend_name=self._backend_name,
            _status_fn=_status,
            _result_fn=_result,
        )

    def fetch_calibration(self) -> "BackendCalibration":
        """Pull T1/T2/gate-error calibration from the IBM device properties.

        Returns
        -------
        BackendCalibration

        Notes
        -----
        Requires a real IBM backend object (not the simulator stub).  Falls
        back to empty calibration for backends that do not expose
        ``properties()``.
        """
        backend_obj = self._base._backend
        if backend_obj is None:
            raise HardwareError("Backend not connected.")

        n_qubits: int = 0
        t1: dict[int, float] = {}
        t2: dict[int, float] = {}
        gate_errors: dict[str, float] = {}
        coupling_edges: list[tuple[int, int]] = []
        gate_times: dict[str, float] = {}

        # Qiskit backends expose .properties() and .configuration()
        try:
            cfg = backend_obj.configuration()
            n_qubits = int(getattr(cfg, "n_qubits", 0))
            raw_coupling = getattr(cfg, "coupling_map", []) or []
            seen: set[tuple[int, int]] = set()
            for pair in raw_coupling:
                if len(pair) >= 2:
                    e = (int(pair[0]), int(pair[1]))
                    rev = (e[1], e[0])
                    if e not in seen and rev not in seen:
                        coupling_edges.append(e)
                        seen.add(e)
        except Exception:
            pass

        try:
            props = backend_obj.properties()
            if props is not None:
                for q in range(n_qubits):
                    try:
                        t1_s = props.t1(q)         # seconds
                        t1[q] = t1_s * 1e9        # → nanoseconds
                    except Exception:
                        pass
                    try:
                        t2_s = props.t2(q)
                        t2[q] = t2_s * 1e9
                    except Exception:
                        pass
                # Gate errors and gate times
                for gate in props.gates:
                    gname = getattr(gate, "gate", "")
                    for param in getattr(gate, "parameters", []):
                        pname = getattr(param, "name", "")
                        pval = getattr(param, "value", None)
                        if pname == "gate_error" and pval is not None:
                            cur = gate_errors.get(gname, 0.0)
                            gate_errors[gname] = max(cur, float(pval))
                        elif pname == "gate_length" and pval is not None:
                            cur = gate_times.get(gname, 0.0)
                            # gate_length in seconds → ns
                            gate_times[gname] = max(cur, float(pval) * 1e9)
        except Exception:
            pass

        return BackendCalibration(
            backend_name=self._backend_name,
            n_qubits=n_qubits,
            t1=t1,
            t2=t2,
            gate_errors=gate_errors,
            coupling_edges=coupling_edges,
            gate_times=gate_times,
        )

    def run_pulse(
        self,
        schedule,  # quantx_engine.pulse.PulseSchedule
        shots: int = 1000,
    ) -> "HardwareJob":
        """Submit a QuantX :class:`~quantx_engine.pulse.PulseSchedule` to IBM.

        Converts the schedule to a serialisable representation and submits it
        via the Qiskit ``IBMBackend`` pulse job API.  Returns a
        :class:`HardwareJob`.

        Parameters
        ----------
        schedule:
            A :class:`~quantx_engine.pulse.PulseSchedule` object.
        shots:
            Number of repetitions.
        """
        try:
            import qiskit.pulse as qpulse  # type: ignore[import]
            from qiskit_ibm_runtime import IBMBackend  # type: ignore[import]
        except ImportError as exc:
            raise HardwareError(
                "run_pulse requires qiskit>=1.0 and qiskit-ibm-runtime>=0.20."
            ) from exc

        backend_obj = self._base._backend
        if backend_obj is None:
            raise HardwareError("Backend not connected.")

        # Convert PulseSchedule to a Qiskit Schedule
        qiskit_sched = _quantx_schedule_to_qiskit(schedule, backend_obj)

        # Submit
        try:
            job = backend_obj.run(qiskit_sched, shots=shots, meas_level=2)
        except Exception as exc:
            raise HardwareError(f"Pulse job submission failed: {exc}") from exc

        raw_id = getattr(job, "job_id", "")
        if callable(raw_id):
            try:
                raw_id = job.job_id()
            except Exception:
                raw_id = ""
        job_id: str = raw_id or ""

        def _status(jid: str) -> str:
            try:
                st = job.status()
                name = str(st.name if hasattr(st, "name") else st).lower()
                if "done" in name or "complete" in name:
                    return "completed"
                if "error" in name or "fail" in name or "cancel" in name:
                    return "failed"
                return "running" if "run" in name else "queued"
            except Exception:
                return "queued"

        def _result(jid: str) -> HardwareJobResult:
            try:
                res = job.result()
                counts_raw: dict[str, int] = {}
                try:
                    counts_raw = res.get_counts()
                except Exception:
                    pass
                total = sum(counts_raw.values()) or shots
                probs = {k: v / total for k, v in counts_raw.items()}
                return HardwareJobResult(
                    job_id=jid,
                    backend=self._backend_name,
                    counts=counts_raw,
                    probabilities=probs,
                    shots=shots,
                    metadata={},
                )
            except Exception as exc:
                raise HardwareError(f"Pulse job result fetch failed: {exc}") from exc

        return HardwareJob(
            job_id=job_id,
            backend_name=self._backend_name,
            _status_fn=_status,
            _result_fn=_result,
        )

    def list_backends(self) -> list[str]:
        """List available backend names for the configured IBM instance."""
        service = self._base._service
        if service is None:
            raise HardwareError("Not connected to IBM Quantum.")
        try:
            backends = service.backends()
            return [b.name for b in backends]
        except Exception as exc:
            raise HardwareError(f"list_backends failed: {exc}") from exc

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def __repr__(self) -> str:
        return (
            f"IBMHardwareBackend(backend={self._backend_name!r}, "
            f"instance={self._base._instance!r})"
        )


def _quantx_schedule_to_qiskit(schedule, backend_obj: Any) -> Any:
    """Convert a QuantX PulseSchedule to a minimal Qiskit Schedule.

    Only Gaussian and constant drive pulses are converted; other waveform
    types raise :class:`HardwareError`.
    """
    try:
        import qiskit.pulse as qpulse  # type: ignore[import]
    except ImportError as exc:
        raise HardwareError("qiskit is required for pulse submission.") from exc

    try:
        dt = backend_obj.configuration().dt  # sample time in seconds
    except Exception:
        dt = 2.2222e-10  # default IBM dt ≈ 0.222 ns

    qiskit_sched = qpulse.Schedule()
    t_current = 0

    for instr in schedule.instructions:
        ch = instr.channel
        waveform = instr.waveform

        # Map channel to Qiskit channel using the correct QuantX attribute names:
        # DriveChannel.qubit, MeasureChannel.qubit, ControlChannel.index
        ch_type = type(ch).__name__
        if ch_type == "DriveChannel":
            q_idx = int(getattr(ch, "qubit", 0))
            qch = qpulse.DriveChannel(q_idx)
        elif ch_type == "MeasureChannel":
            q_idx = int(getattr(ch, "qubit", 0))
            qch = qpulse.MeasureChannel(q_idx)
        elif ch_type == "ControlChannel":
            q_idx = int(getattr(ch, "index", 0))
            qch = qpulse.ControlChannel(q_idx)
        else:
            continue

        # Convert waveform to Qiskit waveform
        wf_type = type(waveform).__name__
        duration_ns: float = getattr(waveform, "duration", 100.0)
        n_samples = max(1, int(round(duration_ns * 1e-9 / dt)))

        if wf_type == "GaussianWaveform":
            sigma_ns: float = getattr(waveform, "sigma", duration_ns / 4)
            sigma_samples = max(1, int(round(sigma_ns * 1e-9 / dt)))
            amp: float = getattr(waveform, "amp", 0.5)
            qwf = qpulse.library.Gaussian(
                duration=n_samples, amp=amp, sigma=sigma_samples
            )
        elif wf_type == "ConstantWaveform":
            amp = getattr(waveform, "amp", 0.5)
            qwf = qpulse.library.Constant(duration=n_samples, amp=amp)
        else:
            # Fall back to sampled waveform for other types
            import numpy as np
            t_arr = np.linspace(0, duration_ns, n_samples)
            samples = waveform.envelope(t_arr).astype(complex)
            qwf = qpulse.library.Waveform(samples=samples)

        t_samples = max(0, int(round(t_current * 1e-9 / dt)))
        qiskit_sched |= qpulse.Play(qwf, qch) << t_samples
        t_current += duration_ns

    return qiskit_sched


# ---------------------------------------------------------------------------
# IonQBackend
# ---------------------------------------------------------------------------

#: Gate name mapping from QuantX to IonQ native gate names.
_QX_TO_IONQ: dict[str, str] = {
    "H": "h", "X": "x", "Y": "y", "Z": "z",
    "S": "s", "T": "t", "Sdg": "si", "Tdg": "ti",
    "CNOT": "cnot", "CZ": "cz", "SWAP": "swap",
    "RX": "rx", "RY": "ry", "RZ": "rz",
    "SX": "v", "SXdg": "vi",
    "I": "id",
}

_IONQ_BASE_URL = "https://api.ionq.co/v0.3"


def _circuit_to_ionq(circuit: dict[str, Any]) -> dict[str, Any]:
    """Convert a QuantX circuit dict to IonQ native circuit JSON format.

    Handles both capitalisation conventions (``"CNOT"`` / ``"cnot"`` / ``"cx"``)
    and both qubit-field layouts (``"targets"``/``"controls"`` or the standard
    QuantX ``"qubits"`` flat list).
    """
    n_qubits = int(circuit.get("qubits", 0))
    ops = circuit.get("operations", [])
    ionq_gates: list[dict[str, Any]] = []

    # Canonical-name aliases for common QuantX lowercase/alternative names
    _ALIASES: dict[str, str] = {
        "cx": "CNOT", "ccx": "CNOT",
        "cz": "CZ",
        "h": "H", "x": "X", "y": "Y", "z": "Z",
        "s": "S", "t": "T", "sdg": "Sdg", "tdg": "Tdg",
        "rx": "RX", "ry": "RY", "rz": "RZ",
        "sx": "SX", "sxdg": "SXdg",
        "swap": "SWAP", "id": "I",
        "measure": "MEASURE", "reset": "RESET", "barrier": "BARRIER",
    }

    for op in ops:
        name_raw = str(op.get("name", "")).strip()
        # Normalise: try exact match, then alias map, then uppercase
        name = _ALIASES.get(name_raw.lower(), name_raw) or name_raw.upper()

        if name.upper() in ("MEASURE", "RESET", "BARRIER"):
            continue  # IonQ measures all at end implicitly

        ionq_name = _QX_TO_IONQ.get(name) or _QX_TO_IONQ.get(name.upper())
        if ionq_name is None:
            raise HardwareError(
                f"Gate {name_raw!r} has no IonQ equivalent. "
                f"Supported: {sorted(_QX_TO_IONQ)}."
            )

        # Support both 'qubits' flat list and 'targets'/'controls' fields
        all_qubits = list(op.get("qubits", []))
        targets = list(op.get("targets", all_qubits))
        controls = list(op.get("controls", []))
        params = list(op.get("params", []))

        gate_dict: dict[str, Any] = {"gate": ionq_name}

        norm = name.upper()
        if norm in ("CNOT", "CZ"):
            # Two-qubit gate: if no explicit controls, first qubit is control
            if not controls and len(targets) >= 2:
                controls = targets[:1]
                targets = targets[1:]
            elif not controls and all_qubits and not targets:
                controls = all_qubits[:1]
                targets = all_qubits[1:]
            gate_dict["control"] = int(controls[0]) if controls else 0
            gate_dict["target"] = int(targets[0]) if targets else 1
        elif norm == "SWAP":
            swap_qubits = targets if len(targets) >= 2 else all_qubits
            if len(swap_qubits) >= 2:
                gate_dict["targets"] = [int(swap_qubits[0]), int(swap_qubits[1])]
            else:
                continue
        elif params:
            gate_dict["target"] = int(targets[0]) if targets else 0
            gate_dict["rotation"] = float(params[0])
        else:
            gate_dict["target"] = int(targets[0]) if targets else 0

        ionq_gates.append(gate_dict)

    return {"qubits": n_qubits, "circuit": ionq_gates}


def _ionq_request(
    method: str,
    url: str,
    token: str,
    body: dict | None = None,
) -> dict[str, Any]:
    """Make an IonQ REST API request using urllib.request (stdlib only)."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {
        "Authorization": f"apiKey {token}",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body_bytes = exc.read()
        try:
            msg = json.loads(body_bytes).get("error", body_bytes.decode())
        except Exception:
            msg = body_bytes.decode()
        raise HardwareError(
            f"IonQ API error {exc.code}: {msg}"
        ) from exc
    except urllib.error.URLError as exc:
        raise HardwareError(f"IonQ network error: {exc.reason}") from exc


class IonQBackend:
    """IonQ Cloud REST API (v0.3) backend.

    No extra dependencies beyond the standard library are required.

    Parameters
    ----------
    token:
        IonQ API key.  Falls back to the ``IONQ_API_KEY`` environment variable.
    device:
        IonQ target device, e.g. ``"simulator"`` (default), ``"qpu.harmony"``,
        ``"qpu.aria-1"``.
    base_url:
        IonQ API base URL.  Override for mocking in tests.
    """

    def __init__(
        self,
        token: str | None = None,
        device: str = "simulator",
        base_url: str = _IONQ_BASE_URL,
    ) -> None:
        self._token = token or os.environ.get("IONQ_API_KEY") or ""
        if not self._token:
            raise HardwareError(
                "IonQ API key required. Pass token= or set IONQ_API_KEY env var."
            )
        self._device = device
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    def _submit_job(self, circuit: dict[str, Any], shots: int) -> str:
        """Submit a job to IonQ and return the job id."""
        ionq_circuit = _circuit_to_ionq(circuit)
        payload: dict[str, Any] = {
            "target": self._device,
            "shots": shots,
            "input": ionq_circuit,
        }
        resp = _ionq_request(
            "POST", f"{self._base_url}/jobs", self._token, payload
        )
        job_id = resp.get("id") or resp.get("job_id") or ""
        if not job_id:
            raise HardwareError(f"IonQ did not return a job id. Response: {resp}")
        return str(job_id)

    def _get_job(self, job_id: str) -> dict[str, Any]:
        """Fetch job details from IonQ."""
        return _ionq_request("GET", f"{self._base_url}/jobs/{job_id}", self._token)

    def _status_fn(self, job_id: str) -> str:
        """Return standardised status string for a given IonQ job id."""
        try:
            resp = self._get_job(job_id)
            raw = str(resp.get("status", "")).lower()
            if raw in ("completed", "ready"):
                return "completed"
            if raw in ("failed", "canceled", "cancelled", "error"):
                return "failed"
            if raw == "running":
                return "running"
            return "queued"
        except HardwareError:
            raise
        except Exception:
            return "queued"

    def _result_fn(self, job_id: str) -> "HardwareJobResult":
        """Fetch and convert IonQ job result."""
        resp = self._get_job(job_id)
        shots_done = int(resp.get("shots", 1024))
        histogram: dict[str, float] = (
            resp.get("data", {}).get("histogram", {})
            or resp.get("results", {})
            or {}
        )
        n_qubits = resp.get("input", {}).get("qubits", 1)

        # IonQ histogram keys are decimal integers ("0", "3", …) or bitstrings
        counts: dict[str, int] = {}
        probs: dict[str, float] = {}
        for k, v in histogram.items():
            try:
                idx = int(k)
                bitstr = format(idx, f"0{n_qubits}b")
            except ValueError:
                bitstr = str(k)
            p = float(v)
            probs[bitstr] = p
            counts[bitstr] = max(1, int(round(p * shots_done)))

        return HardwareJobResult(
            job_id=job_id,
            backend=self._device,
            counts=counts,
            probabilities=probs,
            shots=shots_done,
            metadata={k: v for k, v in resp.items()
                      if k not in ("input", "data", "results")},
        )

    def submit(
        self,
        circuit: dict[str, Any],
        shots: int = 1024,
    ) -> "HardwareJob":
        """Submit a circuit asynchronously and return a :class:`HardwareJob`."""
        job_id = self._submit_job(circuit, shots)
        return HardwareJob(
            job_id=job_id,
            backend_name=self._device,
            _status_fn=self._status_fn,
            _result_fn=self._result_fn,
        )

    def run_circuit(
        self,
        circuit: dict[str, Any],
        shots: int = 1024,
        timeout: float = 300.0,
    ) -> "HardwareJobResult":
        """Submit a circuit and block until completion.

        Parameters
        ----------
        circuit:
            QuantX circuit dict.
        shots:
            Number of shots.
        timeout:
            Maximum wait time in seconds.

        Returns
        -------
        HardwareJobResult
        """
        job = self.submit(circuit, shots=shots)
        return job.fetch(timeout=timeout)

    def fetch_calibration(self) -> "BackendCalibration":
        """Fetch device calibration data from the IonQ backends endpoint.

        Returns a :class:`BackendCalibration` with the available fields
        populated; fields not exposed by IonQ are left at defaults.
        """
        try:
            resp = _ionq_request(
                "GET", f"{self._base_url}/backends/{self._device}", self._token
            )
        except HardwareError:
            # Fall back to empty calibration if endpoint not available
            resp = {}

        n_qubits = int(resp.get("qubits", 0))
        connectivity: list[tuple[int, int]] = []
        for pair in resp.get("connectivity", []):
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                connectivity.append((int(pair[0]), int(pair[1])))

        # IonQ reports fidelities; convert to error rates
        fidelities = resp.get("fidelity", {})
        gate_errors: dict[str, float] = {}
        for gate_name, fid in fidelities.items():
            try:
                gate_errors[gate_name] = 1.0 - float(fid)
            except (ValueError, TypeError):
                pass

        t1_us = resp.get("timing", {}).get("T1", None)
        t2_us = resp.get("timing", {}).get("T2", None)
        t1 = {q: float(t1_us) * 1e3 for q in range(n_qubits)} if t1_us else {}
        t2 = {q: float(t2_us) * 1e3 for q in range(n_qubits)} if t2_us else {}

        return BackendCalibration(
            backend_name=self._device,
            n_qubits=n_qubits,
            t1=t1,
            t2=t2,
            gate_errors=gate_errors,
            coupling_edges=connectivity,
            gate_times={},
        )

    def __repr__(self) -> str:
        return f"IonQBackend(device={self._device!r})"


# ---------------------------------------------------------------------------
# GoogleBackend
# ---------------------------------------------------------------------------

class GoogleBackend:
    """Google Quantum AI backend via ``cirq-google``.

    Converts QuantX circuits to Cirq objects using the existing
    :func:`~quantx_engine.adapters.to_cirq` adapter and submits them via the
    ``cirq-google`` SDK.

    Requirements
    ------------
    ``pip install cirq-google``

    Parameters
    ----------
    project_id:
        Google Cloud project id (e.g. ``"my-gcp-project"``).
    processor_id:
        Quantum processor id (e.g. ``"rainbow"``). Defaults to ``"simulator"``
        which routes to the in-process Cirq simulator.
    credentials:
        Optional Google OAuth2 credentials object.  If ``None``, application
        default credentials are used.
    """

    def __init__(
        self,
        project_id: str,
        processor_id: str = "simulator",
        credentials: Any = None,
    ) -> None:
        self._project_id = project_id
        self._processor_id = processor_id
        self._credentials = credentials
        self._engine: Any = None  # lazy-initialised

    # ------------------------------------------------------------------
    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            import cirq_google  # type: ignore[import]
        except ImportError as exc:
            raise HardwareError(
                "GoogleBackend requires cirq-google. "
                "Install with: pip install cirq-google"
            ) from exc
        try:
            kwargs: dict[str, Any] = {"project_id": self._project_id}
            if self._credentials is not None:
                kwargs["credentials"] = self._credentials
            self._engine = cirq_google.Engine(**kwargs)
        except Exception as exc:
            raise HardwareError(
                f"Google Quantum AI engine initialisation failed: {exc}"
            ) from exc
        return self._engine

    def _to_cirq_circuit(self, circuit: dict[str, Any]) -> Any:
        """Convert a QuantX circuit dict to a Cirq Circuit object."""
        from .adapters import to_cirq  # local import

        return to_cirq(circuit)

    def submit(
        self,
        circuit: dict[str, Any],
        shots: int = 1000,
    ) -> "HardwareJob":
        """Submit a circuit asynchronously and return a :class:`HardwareJob`."""
        if self._processor_id == "simulator":
            # Use Cirq's built-in simulator path — run synchronously then wrap
            return self._submit_simulator(circuit, shots)

        engine = self._get_engine()
        try:
            import cirq_google  # type: ignore[import]

            cirq_circuit = self._to_cirq_circuit(circuit)
            gjob = engine.run_async(
                program=cirq_circuit,
                processor_id=self._processor_id,
                repetitions=shots,
            )
        except Exception as exc:
            raise HardwareError(f"Google job submission failed: {exc}") from exc

        job_id: str = str(getattr(gjob, "job_id", "") or "")

        def _status(jid: str) -> str:
            try:
                raw = str(gjob.status()).lower()
                if "success" in raw or "done" in raw or "ready" in raw:
                    return "completed"
                if "fail" in raw or "error" in raw or "cancel" in raw:
                    return "failed"
                return "running"
            except Exception:
                return "queued"

        def _result(jid: str) -> HardwareJobResult:
            return _google_result_to_hardware(gjob.results(), shots,
                                              jid, self._processor_id)

        return HardwareJob(
            job_id=job_id,
            backend_name=self._processor_id,
            _status_fn=_status,
            _result_fn=_result,
        )

    def _submit_simulator(
        self, circuit: dict[str, Any], shots: int
    ) -> "HardwareJob":
        """Run on Cirq's local simulator and wrap result in HardwareJob."""
        try:
            import cirq  # type: ignore[import]
        except ImportError as exc:
            raise HardwareError("GoogleBackend simulator mode requires cirq.") from exc

        cirq_circuit = self._to_cirq_circuit(circuit)
        # Cirq simulator needs explicit measurement gates; add them if absent
        qubits = sorted(cirq_circuit.all_qubits())
        if not any(
            isinstance(op.gate, cirq.MeasurementGate)
            for moment in cirq_circuit
            for op in moment.operations
        ):
            cirq_circuit = cirq_circuit + cirq.measure(*qubits, key="m")

        sim = cirq.Simulator()
        result = sim.run(cirq_circuit, repetitions=shots)
        counts = _cirq_result_to_counts(result, shots)
        total = sum(counts.values()) or shots
        probs = {k: v / total for k, v in counts.items()}

        job_id = f"cirq-sim-{id(result)}"
        res = HardwareJobResult(
            job_id=job_id,
            backend="cirq-simulator",
            counts=counts,
            probabilities=probs,
            shots=shots,
            metadata={},
        )

        def _status(_: str) -> str:
            return "completed"

        def _result(_: str) -> HardwareJobResult:
            return res

        return HardwareJob(
            job_id=job_id,
            backend_name="cirq-simulator",
            _status_fn=_status,
            _result_fn=_result,
        )

    def run_circuit(
        self,
        circuit: dict[str, Any],
        shots: int = 1000,
        timeout: float = 300.0,
    ) -> "HardwareJobResult":
        """Submit and block until completion."""
        return self.submit(circuit, shots=shots).fetch(timeout=timeout)

    def fetch_calibration(self) -> "BackendCalibration":
        """Fetch calibration from cirq-google processor device specification.

        Returns a :class:`BackendCalibration`.  Raises :class:`HardwareError`
        if the processor is ``"simulator"`` (no calibration available).
        """
        if self._processor_id == "simulator":
            raise HardwareError(
                "Calibration is not available for the local Cirq simulator."
            )
        engine = self._get_engine()
        try:
            import cirq_google  # type: ignore[import]

            processor = engine.get_processor(self._processor_id)
            device = processor.get_device()
            qubits = list(device.qubits)
            n_qubits = len(qubits)
            # Map qubit index by row*ncols+col for grid qubits
            q_to_idx = {q: i for i, q in enumerate(sorted(qubits))}
            coupling: list[tuple[int, int]] = []
            for q in qubits:
                # Prefer device.neighbors_of(q) for physical adjacency;
                # fall back to GridQubit-style row/col neighbours.
                neighbors: list[Any] = []
                if hasattr(device, "neighbors_of"):
                    try:
                        neighbors = list(device.neighbors_of(q))
                    except Exception:
                        pass
                if not neighbors and hasattr(q, "neighbors"):
                    try:
                        neighbors = [nb for nb in q.neighbors()
                                     if nb in q_to_idx]
                    except Exception:
                        pass
                for nb in neighbors:
                    if nb in q_to_idx and q_to_idx[q] < q_to_idx[nb]:
                        coupling.append((q_to_idx[q], q_to_idx[nb]))

            return BackendCalibration(
                backend_name=self._processor_id,
                n_qubits=n_qubits,
                coupling_edges=coupling,
            )
        except Exception as exc:
            raise HardwareError(
                f"Google calibration fetch failed: {exc}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"GoogleBackend(project={self._project_id!r}, "
            f"processor={self._processor_id!r})"
        )


def _google_result_to_hardware(
    cirq_results: Any,
    shots: int,
    job_id: str,
    backend: str,
) -> "HardwareJobResult":
    """Convert cirq-google results to HardwareJobResult."""
    counts: dict[str, int] = {}
    for res in (cirq_results if isinstance(cirq_results, list) else [cirq_results]):
        partial = _cirq_result_to_counts(res, shots)
        for k, v in partial.items():
            counts[k] = counts.get(k, 0) + v
    total = sum(counts.values()) or shots
    return HardwareJobResult(
        job_id=job_id,
        backend=backend,
        counts=counts,
        probabilities={k: v / total for k, v in counts.items()},
        shots=shots,
        metadata={},
    )


def _cirq_result_to_counts(result: Any, shots: int) -> dict[str, int]:
    """Convert a Cirq Result object to a joint-bitstring count dict.

    All measurement keys are sorted alphabetically and concatenated per shot
    to produce a single bitstring per repetition, matching the hardware-style
    ``{"0110": 42, ...}`` format.
    """
    counts: dict[str, int] = {}
    try:
        keys = sorted(result.measurements.keys())
        if not keys:
            return counts
        # Build per-shot bitstrings by concatenating all measurement key arrays
        arrays = [result.measurements[k] for k in keys]  # each: (reps, n_bits)
        n_reps = len(arrays[0])
        for i in range(n_reps):
            bits = "".join(
                "".join(str(int(b)) for b in arr[i])
                for arr in arrays
            )
            counts[bits] = counts.get(bits, 0) + 1
        return counts
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# calibration_to_noise_model
# ---------------------------------------------------------------------------

def calibration_to_noise_model(
    calibration: "BackendCalibration",
    t_gate_ns: float | None = None,
) -> Any:
    """Convert a :class:`BackendCalibration` to a QuantX noise model.

    Uses :meth:`BackendCalibration.to_noise_model`.  If significant
    depolarizing gate error is present (max gate error > 0.1 %) a
    :class:`~quantx_engine.noise.CompositeNoise` combining
    :class:`~quantx_engine.noise.DepolarizingNoise` with the thermal
    relaxation model is returned.

    Parameters
    ----------
    calibration:
        Device calibration snapshot.
    t_gate_ns:
        Override gate duration in ns; defaults to the calibration value.

    Returns
    -------
    :class:`~quantx_engine.noise.NoiseModel`
    """
    from .noise import DepolarizingNoise, CompositeNoise

    thermal = calibration.to_noise_model(t_gate_ns=t_gate_ns)

    max_gate_err = max(calibration.gate_errors.values(), default=0.0)
    if max_gate_err > 0.001:
        # A gate error is an average gate infidelity r. For the one-qubit depolarizing
        # channel (1 - p)rho + p I/2 that DepolarizingNoise applies, r = p/2, so p = 2r.
        depol = DepolarizingNoise(prob=min(2.0 * max_gate_err, 1.0))
        return CompositeNoise([depol, thermal])
    return thermal


# ---------------------------------------------------------------------------
# available_backends
# ---------------------------------------------------------------------------

def available_backends() -> dict[str, list[str]]:
    """Scan environment variables and return a dict of available backends.

    Checks for:

    * ``QISKIT_IBM_TOKEN`` → IBM backends (uses ``"ibmq_qasm_simulator"`` as
      placeholder; real backend list requires a live connection).
    * ``IONQ_API_KEY`` → IonQ backends (simulator always available).
    * ``GOOGLE_CLOUD_PROJECT`` → Google backends (simulator always available).

    Returns
    -------
    dict mapping provider name to list of backend name strings.
    """
    result: dict[str, list[str]] = {}

    if os.environ.get("QISKIT_IBM_TOKEN"):
        result["ibm"] = ["ibmq_qasm_simulator"]

    if os.environ.get("IONQ_API_KEY"):
        result["ionq"] = ["simulator", "qpu.harmony", "qpu.aria-1", "qpu.forte-1"]

    if os.environ.get("GOOGLE_CLOUD_PROJECT"):
        result["google"] = ["simulator", "rainbow", "weber"]

    return result
