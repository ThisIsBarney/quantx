"""Quantum network simulation endpoints — M96 / M98.

Routes
------
POST /network/repeater-chain          simulate_repeater_chain
POST /network/fidelity/dephasing      fidelity_after_dephasing
POST /network/fidelity/swapping       fidelity_after_swapping
POST /network/fidelity/purification   fidelity_after_purification
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/network", tags=["quantum-network"])


# ── Request / response models ─────────────────────────────────────────────────

class RepeaterChainRequest(BaseModel):
    n_hops: int = Field(0, ge=0, description="Number of intermediate repeater nodes")
    segment_length_km: float = Field(10.0, gt=0, description="Length of each fibre segment (km)")
    attenuation_db_km: float = Field(0.2, ge=0, description="Fibre loss (dB/km)")
    dephasing_per_segment: float = Field(0.01, ge=0, le=1, description="Depolarising probability per segment")
    bsm_fidelity: float = Field(0.99, gt=0, le=1, description="BSM gate fidelity at repeaters")
    purification_rounds: int = Field(0, ge=0, description="DEJMPS purification rounds per segment")
    memory_coherence_time: float | None = Field(None, gt=0, description="T2 memory coherence time (s)")
    t_segment_s: float | None = Field(None, gt=0, description="Time to generate one Bell pair per segment (s)")


class DephasingRequest(BaseModel):
    F: float = Field(..., ge=0, le=1, description="Input Werner fidelity")
    p_deph: float = Field(..., ge=0, le=1, description="Two-qubit depolarising probability")


class SwappingRequest(BaseModel):
    F_A: float = Field(..., ge=0, le=1, description="Fidelity of left Bell pair")
    F_B: float = Field(..., ge=0, le=1, description="Fidelity of right Bell pair")
    F_gate: float = Field(1.0, gt=0, le=1, description="BSM gate fidelity")


class PurificationRequest(BaseModel):
    F: float = Field(..., ge=0, le=1, description="Input Werner fidelity")
    n_rounds: int = Field(1, ge=0, description="Number of DEJMPS purification rounds")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/repeater-chain", summary="Simulate N-hop repeater chain")
async def repeater_chain(req: RepeaterChainRequest) -> dict[str, Any]:
    """Simulate an N-hop quantum repeater chain and return end-to-end fidelity.

    Uses the Werner-state model with optional memory dephasing and
    DEJMPS purification. Returns fidelity at each hop and the asymptotic
    BB84 secret key rate.
    """
    from quantx_engine.quantum_network import simulate_repeater_chain, NetworkError
    try:
        result = simulate_repeater_chain(
            n_hops=req.n_hops,
            segment_length_km=req.segment_length_km,
            attenuation_db_km=req.attenuation_db_km,
            dephasing_per_segment=req.dephasing_per_segment,
            bsm_fidelity=req.bsm_fidelity,
            purification_rounds=req.purification_rounds,
            memory_coherence_time=req.memory_coherence_time,
            t_segment_s=req.t_segment_s,
        )
    except NetworkError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})
    return {
        "ok": True,
        "end_to_end_fidelity": result.end_to_end_fidelity,
        "hop_fidelities": result.hop_fidelities,
        "total_length_km": result.total_length_km,
        "n_hops": result.n_hops,
        "secret_key_rate": result.secret_key_rate,
        "metadata": result.metadata,
    }


@router.post("/fidelity/dephasing", summary="Werner fidelity after dephasing")
async def fidelity_dephasing(req: DephasingRequest) -> dict[str, Any]:
    """Compute F' = (1 − p)·F + p/4 (two-qubit depolarising noise)."""
    from quantx_engine.quantum_network import fidelity_after_dephasing, NetworkError
    try:
        F_out = fidelity_after_dephasing(req.F, req.p_deph)
    except NetworkError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    return {"ok": True, "F_out": F_out}


@router.post("/fidelity/swapping", summary="Fidelity after entanglement swapping")
async def fidelity_swapping(req: SwappingRequest) -> dict[str, Any]:
    """Compute fidelity after BSM-based entanglement swapping.

    F_out = F_gate · (F_A·F_B + (1−F_A)(1−F_B)/3) + (1−F_gate)/4
    """
    from quantx_engine.quantum_network import fidelity_after_swapping, NetworkError
    try:
        F_out = fidelity_after_swapping(req.F_A, req.F_B, req.F_gate)
    except NetworkError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    return {"ok": True, "F_out": F_out}


@router.post("/fidelity/purification", summary="Fidelity after DEJMPS purification")
async def fidelity_purification(req: PurificationRequest) -> dict[str, Any]:
    """Compute fidelity after n rounds of DEJMPS entanglement purification.

    Each round: F' = F² / (F² + (1−F)²/3).
    """
    from quantx_engine.quantum_network import fidelity_after_purification, NetworkError
    try:
        F_out = fidelity_after_purification(req.F, req.n_rounds)
    except NetworkError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    return {"ok": True, "F_out": F_out}
