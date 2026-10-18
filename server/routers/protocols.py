"""Quantum communication protocol endpoints — M97 / M98.

Routes
------
POST /protocols/bb84           BB84 key distribution
POST /protocols/e91            Ekert-91 entanglement-based QKD
POST /protocols/teleportation  Quantum state teleportation
POST /protocols/dense-coding   Superdense coding
GET  /protocols/info           Key rate and fidelity utilities
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/protocols", tags=["quantum-protocols"])


# ── Request models ────────────────────────────────────────────────────────────

class BB84Request(BaseModel):
    n_bits: int = Field(1000, gt=0, description="Number of raw qubits Alice sends")
    channel_error_rate: float = Field(0.02, ge=0, le=0.5, description="Intrinsic QBER from channel noise")
    eve_intercept_fraction: float = Field(0.0, ge=0, le=1, description="Fraction of qubits intercepted by Eve")
    detection_threshold: float = Field(0.11, ge=0, le=0.5, description="QBER threshold for Eve detection")
    seed: int | None = Field(None, description="Random seed for reproducibility")


class E91Request(BaseModel):
    n_pairs: int = Field(2000, gt=3, description="Number of Bell pairs to consume")
    bell_pair_fidelity: float = Field(0.95, ge=0, le=1, description="Bell-pair fidelity F")
    channel_error_rate: float = Field(0.01, ge=0, le=0.5, description="Additional bit-flip error rate")
    seed: int | None = Field(None, description="Random seed")


class TeleportationRequest(BaseModel):
    n_states: int = Field(100, gt=0, description="Number of qubit states to teleport")
    channel_fidelity: float = Field(0.95, ge=0, le=1, description="Bell-pair fidelity of the shared channel")
    bsm_fidelity: float = Field(0.99, gt=0, le=1, description="Alice's BSM gate fidelity")
    seed: int | None = Field(None, description="Random seed")


class DenseCodingRequest(BaseModel):
    n_messages: int = Field(500, gt=0, description="Number of 2-bit messages to send")
    channel_fidelity: float = Field(0.95, ge=0, le=1, description="Bell-pair fidelity of the shared channel")
    seed: int | None = Field(None, description="Random seed")


class KeyRateRequest(BaseModel):
    qber: float = Field(..., ge=0, le=0.5, description="Quantum Bit Error Rate")


class TeleportationFidelityRequest(BaseModel):
    channel_fidelity: float = Field(..., ge=0, le=1, description="Werner channel Bell-pair fidelity")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/bb84", summary="Run BB84 key distribution protocol")
async def run_bb84(req: BB84Request) -> dict[str, Any]:
    """Simulate a full BB84 key exchange.

    Models an optional intercept-resend eavesdropper and computes the
    asymptotic Shor–Preskill secure key rate. Returns the sifted key
    bits (without the error-check sample) and Eve-detection flag.
    """
    from quantx_engine.contrib.quantum_protocols import BB84Protocol, ProtocolError
    try:
        proto = BB84Protocol(
            channel_error_rate=req.channel_error_rate,
            eve_intercept_fraction=req.eve_intercept_fraction,
            detection_threshold=req.detection_threshold,
            seed=req.seed,
        )
        result = proto.run(n_bits=req.n_bits)
    except ProtocolError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})
    return {
        "ok": True,
        "n_raw_bits": result.n_raw_bits,
        "n_sifted_bits": result.n_sifted_bits,
        "n_secure_bits": result.n_secure_bits,
        "qber": result.qber,
        "secure_key_rate": result.secure_key_rate,
        "eve_detected": result.eve_detected,
        "metadata": result.metadata,
    }


@router.post("/e91", summary="Run Ekert-91 entanglement-based QKD")
async def run_e91(req: E91Request) -> dict[str, Any]:
    """Simulate the E91 QKD protocol.

    Uses optimal CHSH measurement angles for |Φ+⟩ (S = 2√2 at unit
    fidelity). Returns the CHSH S-parameter, Bell violation flag, key
    bits extracted, and Shor–Preskill secure key rate.
    """
    from quantx_engine.contrib.quantum_protocols import E91Protocol, ProtocolError
    try:
        proto = E91Protocol(
            bell_pair_fidelity=req.bell_pair_fidelity,
            channel_error_rate=req.channel_error_rate,
            seed=req.seed,
        )
        result = proto.run(n_pairs=req.n_pairs)
    except ProtocolError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})
    return {
        "ok": True,
        "n_pairs": result.n_pairs,
        "n_key_bits": result.n_key_bits,
        "chsh_s": result.chsh_s,
        "bell_violation": result.bell_violation,
        "qber": result.qber,
        "secure_key_rate": result.secure_key_rate,
        "metadata": result.metadata,
    }


@router.post("/teleportation", summary="Simulate quantum state teleportation")
async def run_teleportation(req: TeleportationRequest) -> dict[str, Any]:
    """Simulate quantum teleportation over a Werner-state channel.

    Average fidelity = (2F+1)/3 where F is the effective channel fidelity
    after BSM imperfection. Returns per-protocol success rate vs the
    classical 2/3 fidelity limit.
    """
    from quantx_engine.contrib.quantum_protocols import TeleportationProtocol, ProtocolError
    try:
        proto = TeleportationProtocol(
            channel_fidelity=req.channel_fidelity,
            bsm_fidelity=req.bsm_fidelity,
            seed=req.seed,
        )
        result = proto.run(n_states=req.n_states)
    except ProtocolError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})
    return {
        "ok": True,
        "n_teleported": result.n_teleported,
        "average_fidelity": result.average_fidelity,
        "channel_fidelity": result.channel_fidelity,
        "classical_bits_sent": result.classical_bits_sent,
        "success_rate": result.success_rate,
        "metadata": result.metadata,
    }


@router.post("/dense-coding", summary="Simulate superdense coding")
async def run_dense_coding(req: DenseCodingRequest) -> dict[str, Any]:
    """Simulate superdense coding: 2 classical bits per qubit.

    Returns the channel capacity (bits/qubit), message error rate, and
    total classical bits encoded.
    """
    from quantx_engine.contrib.quantum_protocols import DenseCodingProtocol, ProtocolError
    try:
        proto = DenseCodingProtocol(
            channel_fidelity=req.channel_fidelity,
            seed=req.seed,
        )
        result = proto.run(n_messages=req.n_messages)
    except ProtocolError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})
    return {
        "ok": True,
        "n_messages": result.n_messages,
        "n_bits_sent_quantum": result.n_bits_sent_quantum,
        "n_classical_bits_encoded": result.n_classical_bits_encoded,
        "channel_capacity": result.channel_capacity,
        "error_rate": result.error_rate,
        "metadata": result.metadata,
    }


@router.post("/utils/key-rate", summary="Compute Shor-Preskill BB84 key rate")
async def compute_key_rate(req: KeyRateRequest) -> dict[str, Any]:
    """Return the asymptotic BB84 secure key rate: r = 1 − 2·h₂(QBER)."""
    from quantx_engine.contrib.quantum_protocols import bb84_key_rate, ProtocolError
    try:
        rate = bb84_key_rate(req.qber)
    except ProtocolError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    return {"ok": True, "qber": req.qber, "secure_key_rate": rate}


@router.post("/utils/teleportation-fidelity", summary="Compute teleportation average fidelity")
async def compute_teleportation_fidelity(req: TeleportationFidelityRequest) -> dict[str, Any]:
    """Return average teleportation fidelity: f = (2F + 1) / 3."""
    from quantx_engine.contrib.quantum_protocols import teleportation_fidelity, ProtocolError
    try:
        f = teleportation_fidelity(req.channel_fidelity)
    except ProtocolError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    return {
        "ok": True,
        "channel_fidelity": req.channel_fidelity,
        "average_teleportation_fidelity": f,
        "beats_classical": f > 2.0 / 3.0,
    }
