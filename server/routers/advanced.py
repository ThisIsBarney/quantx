"""Advanced solver endpoints — M91/M92/M94/M95 / M98.

Routes
------
POST /advanced/floquet/driven-two-level  Driven qubit Floquet quasi-energies
POST /advanced/tdmrg/evolve              TDMRG real-time evolution (Heisenberg / Ising)
POST /advanced/vqe/noise-aware           Noise-aware VQE
POST /advanced/fault-tolerant/logical    Logical qubit error rate (repetition code)
POST /advanced/fault-tolerant/magic-state  15-to-1 magic state distillation
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/advanced", tags=["advanced-solvers"])


# ── Floquet ───────────────────────────────────────────────────────────────────

class FloquetDrivenTwoLevelRequest(BaseModel):
    omega_0: float = Field(1.0, gt=0, description="Qubit natural frequency ω₀ (rad/s)")
    omega_drive: float = Field(1.0, gt=0, description="Drive frequency ω_d (rad/s)")
    drive_amplitude: float = Field(0.1, ge=0, description="Drive amplitude Ω_R (rad/s)")
    n_steps: int = Field(200, ge=10, le=2000, description="Number of integration steps per period")


# ── TDMRG ─────────────────────────────────────────────────────────────────────

class TDMRGEvolveRequest(BaseModel):
    n_sites: int = Field(4, ge=2, le=16, description="Number of lattice sites")
    model: str = Field("heisenberg", description="Model: 'heisenberg' or 'ising'")
    J: float = Field(1.0, description="Coupling strength J")
    h: float = Field(0.5, description="Transverse field h (Ising only)")
    t_total: float = Field(1.0, gt=0, description="Total real-time evolution (s)")
    dt: float = Field(0.05, gt=0, description="Trotter time step")
    chi: int = Field(16, ge=1, le=64, description="Max MPS bond dimension χ")


# ── Noise-aware VQE ───────────────────────────────────────────────────────────

class NoiseAwareVQERequest(BaseModel):
    n_qubits: int = Field(2, ge=1, le=8, description="Number of qubits")
    n_layers: int = Field(2, ge=1, description="Number of HEA ansatz layers")
    # Hamiltonian as flat real 2^n × 2^n matrix (row-major)
    hamiltonian_real: list[list[float]] = Field(
        ...,
        description="Real part of Hamiltonian matrix (2^n × 2^n, row-major)"
    )
    hamiltonian_imag: list[list[float]] | None = Field(
        None,
        description="Imaginary part of Hamiltonian matrix (optional; defaults to zero)"
    )
    depolarizing_noise: float = Field(0.01, ge=0, le=1, description="Depolarising noise per gate")
    max_iter: int = Field(100, ge=1, le=500, description="Maximum SPSA iterations")
    seed: int | None = Field(None, description="Random seed")


# ── Fault-tolerant ────────────────────────────────────────────────────────────

class LogicalQubitRequest(BaseModel):
    distance: int = Field(3, ge=3, description="Repetition code distance (must be odd, ≥ 3)")
    p_physical: float = Field(0.01, ge=0, le=0.5, description="Physical gate error probability")
    n_shots: int = Field(500, ge=1, le=10000, description="Monte Carlo shots for error rate estimation")
    seed: int | None = Field(None, description="Random seed")


class MagicStateRequest(BaseModel):
    p_physical: float = Field(0.01, ge=0, le=0.5, description="Input T-gate error probability")
    n_attempts: int = Field(500, ge=1, le=10000, description="Number of distillation rounds")
    seed: int | None = Field(None, description="Random seed")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/floquet/driven-two-level", summary="Driven two-level Floquet quasi-energies")
async def floquet_driven_two_level(req: FloquetDrivenTwoLevelRequest) -> dict[str, Any]:
    """Compute Floquet quasi-energies of a periodically driven qubit.

    Uses RK4 integration over one drive period to build the Floquet propagator,
    then extracts quasi-energies via matrix log.

    Returns quasi-energies ε ∈ (−ω/2, ω/2] and the effective Hamiltonian H_eff.
    """
    from quantx_engine.floquet import driven_two_level, FloquetError
    try:
        result = driven_two_level(
            omega0=req.omega_0,
            drive_amp=req.drive_amplitude,
            omega_drive=req.omega_drive,
            n_steps=req.n_steps,
        )
    except FloquetError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})

    qe = result.quasi_energies
    # gap is a method on FloquetResult
    gap_val = result.gap() if callable(result.gap) else float(result.gap)
    return {
        "ok": True,
        "quasi_energies": qe.tolist() if hasattr(qe, "tolist") else list(qe),
        "gap": float(gap_val),
        "omega_drive": req.omega_drive,
        "metadata": {
            "omega_0": req.omega_0,
            "drive_amplitude": req.drive_amplitude,
            "n_steps": req.n_steps,
        },
    }


@router.post("/tdmrg/evolve", summary="TDMRG real-time MPS evolution")
async def tdmrg_evolve(req: TDMRGEvolveRequest) -> dict[str, Any]:
    """Run real-time TDMRG (Suzuki-Trotter + SVD truncation) on a 1D spin chain.

    Supports Heisenberg (``H = J Σ S_i·S_{i+1}``) and transverse-field Ising
    (``H = −J Σ ZZ − h Σ X``) models. Initial state is the Néel state |↑↓↑↓…⟩.

    Returns the entanglement entropy profile at the final time step and the
    energy expectation value history.
    """
    from quantx_engine.tdmrg import real_time_evolve, TDMRGError
    from quantx_engine.tensor_network import MPO, TensorNetworkError
    import numpy as np

    model = req.model.lower()
    if model not in ("heisenberg", "ising"):
        raise HTTPException(
            status_code=400,
            detail={"ok": False, "error": f"Unknown model {req.model!r}. Use 'heisenberg' or 'ising'."},
        )

    try:
        if model == "heisenberg":
            mpo = MPO.heisenberg(n=req.n_sites, J=req.J, hz=0.0)
        else:
            mpo = MPO.ising(n=req.n_sites, J=req.J, h=req.h)

        n_steps = max(1, round(req.t_total / req.dt))
        result = real_time_evolve(
            mpo=mpo,
            dt=req.dt,
            n_steps=n_steps,
            max_chi=req.chi,
        )
    except (TDMRGError, TensorNetworkError) as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})

    final_entropy = result.entropy_at(-1)
    return {
        "ok": True,
        "n_sites": req.n_sites,
        "t_total": float(result.times[-1]),
        "n_steps": len(result.times),
        "final_energy": float(result.energies[-1]),
        "energies": result.energies.tolist(),
        "entanglement_entropy_final": final_entropy.tolist() if hasattr(final_entropy, "tolist") else list(final_entropy),
        "max_entropy": result.max_entropy(),
        "metadata": {
            "model": req.model,
            "J": req.J,
            "h": req.h if model == "ising" else None,
            "chi": req.chi,
            "dt": req.dt,
        },
    }


@router.post("/vqe/noise-aware", summary="Noise-aware VQE with SPSA optimizer")
async def noise_aware_vqe(req: NoiseAwareVQERequest) -> dict[str, Any]:
    """Run noise-aware VQE using a hardware-efficient ansatz and SPSA.

    Accepts the Hamiltonian as a 2^n × 2^n real (+ optional imaginary) matrix.
    Returns the ground-state energy estimate and converged parameters.
    """
    from quantx_engine.noise_vqe import NoiseAwareVQE, NoiseVQEError
    from quantx_engine.noise import DepolarizingNoise
    import numpy as np

    dim = 2 ** req.n_qubits
    if len(req.hamiltonian_real) != dim or any(len(row) != dim for row in req.hamiltonian_real):
        raise HTTPException(
            status_code=400,
            detail={"ok": False, "error": f"hamiltonian_real must be {dim}×{dim} for {req.n_qubits} qubits"},
        )

    try:
        H = np.array(req.hamiltonian_real, dtype=complex)
        if req.hamiltonian_imag is not None:
            if len(req.hamiltonian_imag) != dim or any(len(row) != dim for row in req.hamiltonian_imag):
                raise HTTPException(
                    status_code=400,
                    detail={"ok": False, "error": f"hamiltonian_imag must be {dim}×{dim}"},
                )
            H += 1j * np.array(req.hamiltonian_imag, dtype=float)

        # Build noise model from the requested depolarizing probability.
        # p=0 → noiseless (noise_model=None); p>0 → DepolarizingNoise per gate.
        noise_model = DepolarizingNoise(req.depolarizing_noise) if req.depolarizing_noise > 0 else None
        vqe = NoiseAwareVQE(
            hamiltonian=H,
            n_qubits=req.n_qubits,
            n_layers=req.n_layers,
            noise_model=noise_model,
            max_iter=req.max_iter,
            seed=req.seed if req.seed is not None else 42,
        )
        result = vqe.run()
    except HTTPException:
        raise
    except NoiseVQEError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})

    theta = result.theta
    return {
        "ok": True,
        "energy": float(result.energy),
        "optimal_params": theta.tolist() if hasattr(theta, "tolist") else list(theta),
        "n_iterations": int(result.n_iterations),
        "converged": bool(result.converged),
        "metadata": {
            "n_qubits": req.n_qubits,
            "n_layers": req.n_layers,
            "depolarizing_noise": req.depolarizing_noise,
        },
    }


@router.post("/fault-tolerant/logical", summary="Logical qubit error rate (repetition code)")
async def logical_qubit_error_rate(req: LogicalQubitRequest) -> dict[str, Any]:
    """Estimate logical error rate of a repetition-code logical qubit.

    The distance-d repetition code can correct up to ⌊(d−1)/2⌋ bit-flip errors.
    Returns the simulated logical error rate via Monte Carlo sampling.
    """
    from quantx_engine.fault_tolerant import LogicalQubit, FTError
    try:
        lq = LogicalQubit(
            distance=req.distance,
            physical_error_rate=req.p_physical,
            rng_seed=req.seed,
        )
        logical_error_rate = lq.logical_error_rate(n_shots=req.n_shots)
    except FTError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})

    return {
        "ok": True,
        "logical_error_rate": float(logical_error_rate),
        "physical_error_rate": req.p_physical,
        "code_distance": req.distance,
        "n_shots": req.n_shots,
        "t_correctable": lq.t_correctable,
    }


@router.post("/fault-tolerant/magic-state", summary="15-to-1 magic state distillation")
async def magic_state_distillation(req: MagicStateRequest) -> dict[str, Any]:
    """Run Bravyi-Kitaev 15-to-1 magic state distillation.

    Input T-gate error probability ``p_in`` is suppressed cubically:
    ``p_out ≈ 35 · p_in³``.  Returns the measured output error rate and
    theoretical prediction.
    """
    from quantx_engine.fault_tolerant import MagicStateDistillery, FTError
    try:
        distillery = MagicStateDistillery(p_in=req.p_physical, rng_seed=req.seed)
        result = distillery.distill(n_attempts=req.n_attempts)
    except FTError as exc:
        raise HTTPException(status_code=400, detail={"ok": False, "error": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=500, detail={"ok": False, "error": str(exc)})

    return {
        "ok": True,
        "p_input": req.p_physical,
        "p_output_measured": float(result.p_out_measured),
        "p_output_theoretical": float(result.p_out_theoretical),
        "suppression_factor": float(result.suppression_factor),
        "acceptance_rate": float(result.acceptance_rate),
        "n_attempts": req.n_attempts,
    }
