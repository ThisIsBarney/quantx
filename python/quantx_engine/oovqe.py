"""M124: Orbital-optimised VQE (OO-VQE).

Orbital-optimised VQE simultaneously minimises the energy with respect to
both the circuit (ansatz) parameters θ and the orbital rotation parameters κ.
The orbital rotation is parametrised as an anti-Hermitian matrix K = κ^a_p q
and applied as U(κ) = exp(K) to the one-body integral matrices.

Key components
--------------
:class:`OOVQEResult`
    Container for the OO-VQE optimisation trajectory.
:class:`OOVQESolver`
    Alternating-minimisation solver: inner VQE step (parametric circuit) +
    outer orbital gradient step.

Molecular Hamiltonian model
---------------------------
The Hamiltonian is specified in second quantisation as

    H = Σ_{pq} h_pq a†_p a_q  +  ½ Σ_{pqrs} g_pqrs a†_p a†_q a_r a_s

Orbital rotations transform the one-electron integrals by

    h̃_pq = (U^† h U)_pq,   Ũ = exp(K)

where K is a real skew-symmetric matrix.  For simplicity this module
operates in the qubit representation via Jordan–Wigner with two spin-orbitals
mapped to the Z and X Pauli terms, supporting up to 8 orbitals.

References
----------
* Sokolov, I. O. et al. (2020). Quantum orbital-optimised coupled-cluster
  doubles.  *J. Chem. Phys.* 152, 124107.
* Takeshita, T. et al. (2020). Increasing the representation accuracy of
  quantum simulations of chemistry without extra quantum resources.
  *Phys. Rev. X* 10, 011004.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Sequence

import numpy as np

__all__ = [
    "OOVQEError",
    "OOVQEResult",
    "OOVQESolver",
]

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class OOVQEError(Exception):
    """Raised for invalid OO-VQE operations."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OOVQEResult:
    """Result of an OO-VQE optimisation run.

    Attributes
    ----------
    energies : list[float]
        Energy at the end of each macro-iteration (orbital rotation step).
    circuit_params : ndarray
        Final circuit (ansatz) parameters θ.
    orbital_params : ndarray
        Final orbital rotation parameters κ (upper-triangular of K).
    converged : bool
        True if the energy change between the last two iterations fell below
        the convergence threshold.
    n_iterations : int
        Number of macro-iterations performed.
    """
    energies: list[float] = field(default_factory=list)
    circuit_params: np.ndarray = field(default_factory=lambda: np.array([]))
    orbital_params: np.ndarray = field(default_factory=lambda: np.array([]))
    converged: bool = False
    n_iterations: int = 0

    @property
    def final_energy(self) -> float:
        """Energy at the last iteration."""
        return float(self.energies[-1]) if self.energies else float("nan")

    @property
    def energy_history(self) -> np.ndarray:
        """Energy as a numpy array over iterations."""
        return np.array(self.energies)


# ---------------------------------------------------------------------------
# Orbital rotation helpers
# ---------------------------------------------------------------------------

def _skew_matrix(kappa: np.ndarray, n_orb: int) -> np.ndarray:
    """Build a real skew-symmetric matrix K from upper-triangular parameters."""
    K = np.zeros((n_orb, n_orb), dtype=float)
    idx = 0
    for i in range(n_orb):
        for j in range(i + 1, n_orb):
            K[i, j] = kappa[idx]
            K[j, i] = -kappa[idx]
            idx += 1
    return K


def _orbital_rotation(kappa: np.ndarray, n_orb: int) -> np.ndarray:
    """Return U = exp(K) for skew-symmetric K built from kappa."""
    K = _skew_matrix(kappa, n_orb)
    return _matrix_exp_real(K)


def _matrix_exp_real(K: np.ndarray) -> np.ndarray:
    """Matrix exponential of a real skew-symmetric matrix (result is orthogonal)."""
    # Use Padé approximant via scipy if available, else Taylor expansion.
    try:
        from scipy.linalg import expm  # type: ignore
        return expm(K)
    except ImportError:
        # Taylor expansion to order 10
        result = np.eye(K.shape[0])
        term = np.eye(K.shape[0])
        for n in range(1, 11):
            term = term @ K / n
            result = result + term
            if np.linalg.norm(term) < 1e-14:
                break
        return result


def _rotate_integrals(h: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Rotate one-electron integrals: h̃ = U^T h U."""
    return U.T @ h @ U


def _n_kappa(n_orb: int) -> int:
    """Number of independent orbital rotation parameters."""
    return n_orb * (n_orb - 1) // 2


# ---------------------------------------------------------------------------
# JW-mapped energy for a 2-orbital model
# ---------------------------------------------------------------------------

def _jw_energy_2orb(h_rot: np.ndarray, theta: np.ndarray) -> float:
    """Compute ⟨ψ(θ)|H(h̃)|ψ(θ)⟩ for a 2-orbital Jordan–Wigner Hamiltonian.

    Hamiltonian (2-qubit, 2-orbital, 2-electron):
        H = h̃₀₀ Z₀/2 + h̃₁₁ Z₁/2 + h̃₀₁ (X₀X₁ + Y₀Y₁)/2
    (particle-hole picture; simplified model)

    Ansatz: Ry(θ₀) ⊗ Ry(θ₁) applied to |01⟩ (Hartree–Fock reference).
    """
    # Build Hamiltonian matrix
    h00 = float(h_rot[0, 0])
    h11 = float(h_rot[1, 1])
    h01 = float(h_rot[0, 1]) if h_rot.shape[0] > 1 else 0.0

    # |00⟩=0, |01⟩=1, |10⟩=2, |11⟩=3
    H = np.zeros((4, 4), dtype=complex)
    H[0, 0] = h00 / 2 + h11 / 2        # Z0/2 + Z1/2 for |00⟩
    H[1, 1] = h00 / 2 - h11 / 2        # Z0/2 - Z1/2 for |01⟩
    H[2, 2] = -h00 / 2 + h11 / 2       # -Z0/2 + Z1/2 for |10⟩
    H[3, 3] = -h00 / 2 - h11 / 2       # -Z0/2 - Z1/2 for |11⟩
    H[1, 2] = H[2, 1] = h01 / 2        # off-diagonal (X₀X₁ + Y₀Y₁)/2

    # Ansatz state: HF |01⟩ rotated by Ry(θ₀) on qubit 0, Ry(θ₁) on qubit 1
    # |ψ⟩ = (cos(θ₀/2)|0⟩ + sin(θ₀/2)|1⟩) ⊗ (cos(θ₁/2)|0⟩ + sin(θ₁/2)|1⟩)
    # ... but for variational energy we start from |01⟩ HF reference.
    # The "rotation" mixes |01⟩ and |10⟩: a simple 2-param ansatz.
    c0 = math.cos(float(theta[0]) / 2) if len(theta) > 0 else 1.0
    s0 = math.sin(float(theta[0]) / 2) if len(theta) > 0 else 0.0
    c1 = math.cos(float(theta[1]) / 2) if len(theta) > 1 else 1.0
    s1 = math.sin(float(theta[1]) / 2) if len(theta) > 1 else 0.0

    # Start from |01⟩ = [0, 1, 0, 0]^T, apply Ry rotations
    psi = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    # Ry(θ₀) on qubit 0: mixes |0...⟩ ↔ |1...⟩ parts
    psi_new = np.zeros(4, dtype=complex)
    psi_new[0] = c0 * psi[0] - s0 * psi[2]  # |00⟩ component
    psi_new[1] = c0 * psi[1] - s0 * psi[3]  # |01⟩ component
    psi_new[2] = s0 * psi[0] + c0 * psi[2]  # |10⟩ component
    psi_new[3] = s0 * psi[1] + c0 * psi[3]  # |11⟩ component
    psi = psi_new
    # Ry(θ₁) on qubit 1
    psi_new = np.zeros(4, dtype=complex)
    psi_new[0] = c1 * psi[0] - s1 * psi[1]  # |00⟩ ↔ |01⟩
    psi_new[1] = s1 * psi[0] + c1 * psi[1]
    psi_new[2] = c1 * psi[2] - s1 * psi[3]  # |10⟩ ↔ |11⟩
    psi_new[3] = s1 * psi[2] + c1 * psi[3]
    psi = psi_new

    return float(np.real(psi.conj() @ (H @ psi)))


# ---------------------------------------------------------------------------
# OOVQESolver
# ---------------------------------------------------------------------------

class OOVQESolver:
    """Orbital-optimised VQE solver.

    Uses an alternating minimisation strategy:

    1. **Inner VQE**: minimise ⟨ψ(θ)|H(κ)|ψ(θ)⟩ over circuit parameters θ
       by gradient descent.
    2. **Orbital gradient step**: update orbital parameters κ by one step of
       gradient descent on the orbital rotation manifold.

    Parameters
    ----------
    h_core : array-like, shape (n_orb, n_orb)
        One-electron integrals (core Hamiltonian) in the initial MO basis.
    n_orb : int
        Number of spatial orbitals.
    n_circuit_params : int
        Number of parametric circuit (ansatz) angles.
    energy_fn : callable or None
        ``energy_fn(h_rotated, theta) -> float`` computing the VQE energy
        for given rotated one-electron integrals and circuit parameters.
        If None, the built-in 2-orbital JW model is used.
    max_iterations : int
        Maximum number of macro (OO) iterations.
    inner_steps : int
        Number of circuit gradient steps per macro iteration.
    lr_circuit : float
        Learning rate for circuit parameter updates.
    lr_orbital : float
        Learning rate for orbital rotation parameter updates.
    conv_tol : float
        Convergence threshold: |ΔE| < conv_tol terminates the loop.
    eps : float
        Finite-difference step for gradient estimates.
    """

    def __init__(
        self,
        h_core: np.ndarray,
        n_orb: int,
        n_circuit_params: int = 2,
        energy_fn: Optional[Callable] = None,
        max_iterations: int = 30,
        inner_steps: int = 20,
        lr_circuit: float = 0.05,
        lr_orbital: float = 0.02,
        conv_tol: float = 1e-5,
        eps: float = 1e-4,
    ) -> None:
        h = np.asarray(h_core, dtype=float)
        if h.ndim != 2 or h.shape[0] != h.shape[1]:
            raise OOVQEError("h_core must be a square 2-D array.")
        if n_orb < 2:
            raise OOVQEError("n_orb must be at least 2.")
        if h.shape[0] != n_orb:
            raise OOVQEError("h_core shape must be (n_orb, n_orb).")
        if n_circuit_params < 1:
            raise OOVQEError("n_circuit_params must be at least 1.")
        if max_iterations < 1:
            raise OOVQEError("max_iterations must be at least 1.")
        if inner_steps < 1:
            raise OOVQEError("inner_steps must be at least 1.")
        if conv_tol <= 0:
            raise OOVQEError("conv_tol must be positive.")
        if eps <= 0:
            raise OOVQEError("eps must be positive.")

        self.h_core = h.copy()
        self.n_orb = n_orb
        self.n_circuit_params = n_circuit_params
        self.max_iterations = max_iterations
        self.inner_steps = inner_steps
        self.lr_circuit = lr_circuit
        self.lr_orbital = lr_orbital
        self.conv_tol = conv_tol
        self.eps = eps

        if energy_fn is not None:
            self._energy_fn = energy_fn
        elif n_orb == 2 and n_circuit_params <= 2:
            self._energy_fn = _jw_energy_2orb
        else:
            # Generic fallback: diagonal approximation
            self._energy_fn = self._default_energy_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        init_circuit_params: Optional[np.ndarray] = None,
        init_orbital_params: Optional[np.ndarray] = None,
        seed: int = 0,
    ) -> OOVQEResult:
        """Run the OO-VQE optimisation.

        Parameters
        ----------
        init_circuit_params : array-like or None
            Initial circuit parameters θ.  If None, initialised to zeros.
        init_orbital_params : array-like or None
            Initial orbital rotation parameters κ.  If None, initialised to
            zeros (no rotation).
        seed : int
            RNG seed for random initialisation fallback.

        Returns
        -------
        OOVQEResult
        """
        rng = np.random.default_rng(seed)
        n_kappa = _n_kappa(self.n_orb)

        if init_circuit_params is not None:
            theta = np.asarray(init_circuit_params, dtype=float).ravel()
            if theta.shape != (self.n_circuit_params,):
                raise OOVQEError(
                    f"init_circuit_params must have shape "
                    f"({self.n_circuit_params},)."
                )
        else:
            theta = np.zeros(self.n_circuit_params)

        if init_orbital_params is not None:
            kappa = np.asarray(init_orbital_params, dtype=float).ravel()
            if kappa.shape != (n_kappa,):
                raise OOVQEError(
                    f"init_orbital_params must have shape ({n_kappa},)."
                )
        else:
            kappa = np.zeros(n_kappa)

        result = OOVQEResult()
        prev_energy = float("inf")

        for it in range(self.max_iterations):
            # Compute rotated integrals
            U = _orbital_rotation(kappa, self.n_orb)
            h_rot = _rotate_integrals(self.h_core, U)

            # Inner VQE: gradient descent over θ
            theta = self._inner_vqe(h_rot, theta)

            # Compute energy
            energy = float(self._energy_fn(h_rot, theta))
            result.energies.append(energy)
            result.n_iterations = it + 1

            # Check convergence
            if abs(energy - prev_energy) < self.conv_tol:
                result.converged = True
                break
            prev_energy = energy

            # Orbital gradient step
            kappa = self._orbital_gradient_step(kappa, theta)

        result.circuit_params = theta.copy()
        result.orbital_params = kappa.copy()
        return result

    # ------------------------------------------------------------------
    # Inner VQE (circuit gradient descent)
    # ------------------------------------------------------------------

    def _inner_vqe(
        self, h_rot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Gradient-descent minimisation of E(θ) = energy_fn(h_rot, θ)."""
        theta = theta.copy()
        for _ in range(self.inner_steps):
            grad = self._circuit_gradient(h_rot, theta)
            theta = theta - self.lr_circuit * grad
        return theta

    def _circuit_gradient(
        self, h_rot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Numerical gradient ∂E/∂θᵢ via central differences."""
        grad = np.zeros_like(theta)
        h = self.eps
        for i in range(len(theta)):
            th_p = theta.copy(); th_p[i] += h
            th_m = theta.copy(); th_m[i] -= h
            grad[i] = (
                self._energy_fn(h_rot, th_p)
                - self._energy_fn(h_rot, th_m)
            ) / (2 * h)
        return grad

    # ------------------------------------------------------------------
    # Orbital gradient step
    # ------------------------------------------------------------------

    def _orbital_gradient_step(
        self, kappa: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """One gradient step on the orbital parameters κ."""
        grad = np.zeros_like(kappa)
        h = self.eps
        for i in range(len(kappa)):
            kappa_p = kappa.copy(); kappa_p[i] += h
            kappa_m = kappa.copy(); kappa_m[i] -= h
            U_p = _orbital_rotation(kappa_p, self.n_orb)
            U_m = _orbital_rotation(kappa_m, self.n_orb)
            h_p = _rotate_integrals(self.h_core, U_p)
            h_m = _rotate_integrals(self.h_core, U_m)
            E_p = float(self._energy_fn(h_p, theta))
            E_m = float(self._energy_fn(h_m, theta))
            grad[i] = (E_p - E_m) / (2 * h)
        return kappa - self.lr_orbital * grad

    # ------------------------------------------------------------------
    # Default energy function for n_orb > 2
    # ------------------------------------------------------------------

    def _default_energy_fn(
        self, h_rot: np.ndarray, theta: np.ndarray
    ) -> float:
        """Generic diagonal energy: Tr(h_rot * D(θ)) where D is a density.

        Models a single-Slater-determinant energy with occupation numbers
        {cos²(θᵢ/2)} for the first n_circuit_params orbitals.
        """
        n = min(len(theta), self.n_orb)
        energy = 0.0
        for i in range(n):
            occ = math.cos(float(theta[i]) / 2) ** 2
            energy += float(h_rot[i, i]) * occ
        # Remaining occupied orbitals (if n < n_orb, use RHF occupation)
        for i in range(n, self.n_orb):
            energy += float(h_rot[i, i])
        return energy

    def orbital_gradient(
        self, kappa: Optional[np.ndarray] = None,
        theta: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return the orbital gradient ∂E/∂κ at current (κ, θ) values.

        Convenience method for external inspection.

        Parameters
        ----------
        kappa : array-like or None
            Orbital parameters.  Defaults to zeros.
        theta : array-like or None
            Circuit parameters.  Defaults to zeros.
        """
        n_kappa = _n_kappa(self.n_orb)
        if kappa is None:
            kappa = np.zeros(n_kappa)
        if theta is None:
            theta = np.zeros(self.n_circuit_params)
        kappa = np.asarray(kappa, dtype=float)
        theta = np.asarray(theta, dtype=float)
        return self._orbital_gradient_step(kappa, theta) - kappa  # = -lr * grad

    def __repr__(self) -> str:
        return (
            f"OOVQESolver(n_orb={self.n_orb}, "
            f"n_circuit_params={self.n_circuit_params}, "
            f"max_iterations={self.max_iterations})"
        )
