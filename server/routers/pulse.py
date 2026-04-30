"""Pulse-level simulation endpoints — M89 / M98.

Routes
------
POST /pulse/simulate   Simulate a pulse schedule on a driven qubit system
POST /pulse/pi-pulse   Compute Gaussian π-pulse area for a given σ
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/pulse", tags=["pulse-simulator"])


# ── Request models ────────────────────────────────────────────────────────────

class WaveformSpec(BaseModel):
    """Specification of a single waveform instruction."""
    channel_type: str = Field("drive", description="'drive', 'measure', or 'control'")
    channel_index: int = Field(0, ge=0, description="Channel qubit index")
    waveform_type: str = Field(..., description="'gaussian', 'drag', 'flattop', or 'constant'")
    t_start: float = Field(0.0, ge=0, description="Instruction start time (s)")
    duration: float = Field(..., gt=0, description="Waveform duration (s)")
    amp: float = Field(1.0, description="Peak amplitude (rad/s)")
    # Gaussian / DRAG params
    sigma: float | None = Field(None, gt=0, description="Gaussian σ (s); defaults to duration/4")
    beta: float | None = Field(None, description="DRAG β parameter (s); required for 'drag'")
    # Flat-top params
    rise_time: float | None = Field(None, gt=0, description="Flat-top rise time (s); defaults to duration*0.1")


class PulseSimulateRequest(BaseModel):
    n_qubits: int = Field(1, ge=1, le=4, description="Number of qubits to simulate")
    qubit_frequencies: list[float] = Field(
        default_factory=list,
        description="Qubit transition frequencies (rad/s). Defaults to [0.0]*n_qubits."
    )
    t_total: float = Field(..., gt=0, description="Total simulation time (s)")
    waveforms: list[WaveformSpec] = Field(default_factory=list, description="Pulse instructions")
    t1: float | None = Field(None, gt=0, description="T1 relaxation time (s)")
    t2: float | None = Field(None, gt=0, description="T2 dephasing time (s)")


class PiPulseRequest(BaseModel):
    sigma: float = Field(..., gt=0, description="Gaussian pulse σ (s)")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/simulate", summary="Simulate a pulse schedule on a qubit system")
async def pulse_simulate(req: PulseSimulateRequest) -> dict[str, Any]:
    """Run a pulse-level simulation via the Schrödinger / Lindblad ODE solver.

    Accepts a list of waveform instructions (Gaussian, DRAG, flat-top, or
    constant envelopes) and returns the final Bloch vector for each qubit and
    the density matrix.
    """
    from quantx_engine.pulse import (
        PulseError,
        DriveChannel,
        MeasureChannel,
        ControlChannel,
        GaussianWaveform,
        DRAGWaveform,
        FlatTopWaveform,
        ConstantWaveform,
        PulseSchedule,
        QubitSystem,
        PulseSimulator,
    )

    try:
        n = req.n_qubits
        freqs = req.qubit_frequencies or [0.0] * n
        if len(freqs) != n:
            raise HTTPException(
                status_code=400,
                detail={"ok": False, "error": "qubit_frequencies length must equal n_qubits"},
            )

        # QubitSystem
        t1_list = [req.t1] * n if req.t1 is not None else None
        t2_list = [req.t2] * n if req.t2 is not None else None
        system = QubitSystem(
            n_qubits=n,
            frequencies=freqs,
            t1_times=t1_list,
            t2_times=t2_list,
        )

        # Short-circuit: empty schedule → trivial |0⟩ state (no simulation needed)
        if not req.waveforms:
            return {
                "ok": True,
                "bloch_vectors": [[0.0, 0.0, 1.0]] * n,
                "density_matrix_real": [],
                "density_matrix_imag": [],
            }

        # Build schedule using PulseSchedule.add(channel, waveform, t_start)
        schedule = PulseSchedule(name="api_schedule")
        for wf in req.waveforms:
            ctype = wf.channel_type.lower()
            if ctype == "drive":
                channel = DriveChannel(qubit=wf.channel_index)
            elif ctype == "measure":
                channel = MeasureChannel(qubit=wf.channel_index)
            elif ctype == "control":
                channel = ControlChannel(index=wf.channel_index)
            else:
                raise HTTPException(
                    status_code=400,
                    detail={"ok": False, "error": f"Unknown channel_type: {wf.channel_type!r}. Use 'drive', 'measure', or 'control'."},
                )
            wtype = wf.waveform_type.lower()
            if wtype == "gaussian":
                sigma = wf.sigma if wf.sigma is not None else wf.duration / 4.0
                wave = GaussianWaveform(duration=wf.duration, amp=wf.amp, sigma=sigma)
            elif wtype == "drag":
                sigma = wf.sigma if wf.sigma is not None else wf.duration / 4.0
                beta = wf.beta if wf.beta is not None else 0.0
                wave = DRAGWaveform(duration=wf.duration, amp=wf.amp, sigma=sigma, beta=beta)
            elif wtype == "flattop":
                rise = wf.rise_time if wf.rise_time is not None else wf.duration * 0.1
                wave = FlatTopWaveform(duration=wf.duration, amp=wf.amp, rise_time=rise)
            elif wtype == "constant":
                wave = ConstantWaveform(duration=wf.duration, amp=wf.amp)
            else:
                raise HTTPException(
                    status_code=400,
                    detail={"ok": False, "error": f"Unknown waveform_type: {wf.waveform_type!r}"},
                )
            schedule.add(channel, wave, wf.t_start)

        # Extend schedule to req.t_total so idle T1/T2 evolution is included.
        # A zero-amplitude constant pulse padded to t_total ensures the ODE
        # solver integrates over the full requested time window.
        if schedule.duration < req.t_total:
            schedule.add(
                DriveChannel(qubit=0),
                ConstantWaveform(duration=req.t_total, amp=0.0),
                0.0,
            )

        sim = PulseSimulator()
        result = sim.run(schedule=schedule, system=system)

        # Collect Bloch vectors per qubit
        bloch_vectors = []
        for q in range(n):
            bv = result.bloch_vector(q)
            bloch_vectors.append(list(bv))

        dm = result.density_matrix
        return {
            "ok": True,
            "bloch_vectors": bloch_vectors,
            "density_matrix_real": dm.real.tolist() if hasattr(dm, "real") else [],
            "density_matrix_imag": dm.imag.tolist() if hasattr(dm, "imag") else [],
        }
    except HTTPException:
        raise
    except PulseError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})


@router.post("/pi-pulse", summary="Compute Gaussian π-pulse area")
async def pi_pulse_area_endpoint(req: PiPulseRequest) -> dict[str, Any]:
    """Return the pulse area for a Gaussian with given σ.

    The π-pulse area is Ω_R · σ · √(2π).  This endpoint computes and returns
    the total rotation angle in radians for a given σ.
    """
    from quantx_engine.pulse import pi_pulse_area, PulseError
    try:
        area = pi_pulse_area(req.sigma)
    except PulseError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    return {
        "ok": True,
        "sigma_s": req.sigma,
        "pi_pulse_area_rad": float(area),
    }
