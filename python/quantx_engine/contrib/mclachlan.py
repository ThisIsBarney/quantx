"""M122: McLachlan variational quantum dynamics.

Implements time evolution of parametrised quantum states via the
McLachlan variational principle (MVP), which minimises

    ‖(iδ/δt - H)|ψ(θ)⟩‖²

over the velocity dθ/dt of the variational parameters.  This yields the
linear system

    M(θ) θ̇ = V(θ)

where M_ij = Re⟨∂_i ψ|∂_j ψ⟩ is the quantum geometric tensor (real part)
and V_i = Re⟨∂_i ψ|(-iH)|ψ⟩ is the force vector.

Both real-time (Schrödinger) and imaginary-time evolution are supported.

Classes
-------
:class:`McLachlanResult`
    Container for one time-evolution trajectory.
:class:`McLachlanSolver`
    Variational-parameter ODE solver (Euler / RK4 integrators).

References
----------
.. [1] McLachlan, A. D. (1964). A variational solution of the time-dependent
       Schrodinger equation. *Mol. Phys.* 8, 39–44.
       https://doi.org/10.1080/00268976400100041

.. [2] Endo, S. et al. (2020). Variational quantum simulation of general
       processes. *Phys. Rev. Lett.* 125, 010501.
       https://doi.org/10.1103/PhysRevLett.125.010501

.. [3] Yuan, X. et al. (2019). Theory of variational quantum simulation.
       *Quantum* 3, 191. https://doi.org/10.22331/q-2019-10-07-191
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Sequence

import numpy as np

__all__ = [
    "McLachlanError",
    "McLachlanResult",
    "McLachlanSolver",
]

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class McLachlanError(Exception):
    """Raised for invalid McLachlan-solver operations."""


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class McLachlanResult:
    """Trajectory from a McLachlan variational simulation.

    Attributes
    ----------
    times : ndarray, shape (n_steps+1,)
        Time points.
    params : ndarray, shape (n_steps+1, n_params)
        Variational parameters at each time.
    energies : ndarray, shape (n_steps+1,)
        ⟨H⟩ at each time.
    norms : ndarray, shape (n_steps+1,)
        ‖|ψ(θ)⟩‖ at each time.
    converged : bool
        True if the linear solve remained well-conditioned throughout.
    """
    times: np.ndarray
    params: np.ndarray
    energies: np.ndarray
    norms: np.ndarray
    converged: bool = True

    @property
    def final_params(self) -> np.ndarray:
        """Variational parameters at the final time."""
        return self.params[-1]

    @property
    def final_energy(self) -> float:
        """Energy ⟨H⟩ at the final time."""
        return float(self.energies[-1])

    @property
    def n_steps(self) -> int:
        """Number of time steps taken."""
        return len(self.times) - 1

    @property
    def energy_change(self) -> float:
        """Difference between final and initial energy."""
        return float(self.energies[-1] - self.energies[0])


# ---------------------------------------------------------------------------
# McLachlanSolver
# ---------------------------------------------------------------------------

# Type alias: circuit function θ → |ψ(θ)⟩ ∈ ℂ^{2^n}
_AnsatzFn = Callable[[np.ndarray], np.ndarray]
# Hamiltonian as matrix or callable H|ψ⟩ → ndarray
_HFn = Callable[[np.ndarray], np.ndarray]


class McLachlanSolver:
    """Time-evolve a parametrised ansatz under the McLachlan variational principle.

    Parameters
    ----------
    ansatz : callable
        Function ``ansatz(params) -> state_vector`` that maps variational
        parameters to the corresponding normalised quantum state.
    hamiltonian : array-like or callable
        Hamiltonian.  Either a 2-D complex array (dense matrix) or a callable
        ``H(state) -> ndarray`` representing the action H|ψ⟩.
    n_params : int
        Number of variational parameters.
    dt : float
        Time step.
    evolution : {"real", "imaginary"}
        "real": unitary Schrödinger evolution under e^{-iHt}.
        "imaginary": non-unitary cooling under e^{-Ht} (find ground state).
    integrator : {"euler", "rk4"}
        ODE integrator.
    reg : float
        Regularisation added to the diagonal of the metric tensor M to avoid
        singular linear systems.
    eps : float
        Finite-difference step for computing parameter gradients ∂|ψ⟩/∂θᵢ.
    """

    def __init__(
        self,
        ansatz: _AnsatzFn,
        hamiltonian: np.ndarray | _HFn,
        n_params: int,
        dt: float = 0.01,
        evolution: Literal["real", "imaginary"] = "real",
        integrator: Literal["euler", "rk4"] = "euler",
        reg: float = 1e-6,
        eps: float = 1e-5,
    ) -> None:
        if not callable(ansatz):
            raise McLachlanError("ansatz must be callable.")
        if n_params < 1:
            raise McLachlanError("n_params must be at least 1.")
        if dt <= 0:
            raise McLachlanError("dt must be positive.")
        if evolution not in ("real", "imaginary"):
            raise McLachlanError("evolution must be 'real' or 'imaginary'.")
        if integrator not in ("euler", "rk4"):
            raise McLachlanError("integrator must be 'euler' or 'rk4'.")
        if reg < 0:
            raise McLachlanError("reg must be non-negative.")
        if eps <= 0:
            raise McLachlanError("eps must be positive.")

        self.ansatz = ansatz
        self.n_params = n_params
        self.dt = dt
        self.evolution = evolution
        self.integrator = integrator
        self.reg = reg
        self.eps = eps

        # Wrap Hamiltonian
        if callable(hamiltonian) and not isinstance(hamiltonian, np.ndarray):
            self._H_fn: _HFn = hamiltonian
        else:
            H_mat = np.asarray(hamiltonian, dtype=complex)
            if H_mat.ndim != 2:
                raise McLachlanError("Hamiltonian array must be 2-D.")
            self._H_fn = lambda psi, _H=H_mat: _H @ psi  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evolve(
        self,
        init_params: np.ndarray,
        t_final: float,
        observables: Optional[list[np.ndarray | _HFn]] = None,
    ) -> McLachlanResult:
        """Evolve the ansatz from *t=0* to *t=t_final*.

        Parameters
        ----------
        init_params : array-like, shape (n_params,)
            Initial variational parameters.
        t_final : float
            Total evolution time.
        observables : list or None
            Extra observables to track (not stored in result, but checked
            for usage).  Unused in the current implementation but accepted
            for API compatibility.

        Returns
        -------
        McLachlanResult
        """
        theta = np.asarray(init_params, dtype=float).ravel()
        if theta.shape != (self.n_params,):
            raise McLachlanError(
                f"init_params must have shape ({self.n_params},); "
                f"got {theta.shape}."
            )
        if t_final <= 0:
            raise McLachlanError("t_final must be positive.")

        # Use ceil so that dt_actual ≤ configured dt (round can overshoot by ~1.5×)
        n_steps = max(1, math.ceil(t_final / self.dt))
        dt_actual = t_final / n_steps

        times = np.zeros(n_steps + 1)
        params_traj = np.zeros((n_steps + 1, self.n_params))
        energies = np.zeros(n_steps + 1)
        norms = np.zeros(n_steps + 1)
        converged = True

        params_traj[0] = theta.copy()
        psi0 = self._psi(theta)
        energies[0] = self._energy(theta, psi0)
        norms[0] = float(np.linalg.norm(psi0))

        for step in range(n_steps):
            if self.integrator == "euler":
                theta_new, ok = self._euler_step(theta, dt_actual)
            else:
                theta_new, ok = self._rk4_step(theta, dt_actual)
            if not ok:
                converged = False
            theta = theta_new
            times[step + 1] = (step + 1) * dt_actual
            params_traj[step + 1] = theta.copy()
            psi = self._psi(theta)
            energies[step + 1] = self._energy(theta, psi)
            norms[step + 1] = float(np.linalg.norm(psi))

        return McLachlanResult(
            times=times,
            params=params_traj,
            energies=energies,
            norms=norms,
            converged=converged,
        )

    def ground_state_energy(
        self,
        init_params: np.ndarray,
        t_final: float = 5.0,
    ) -> float:
        """Estimate the ground-state energy via imaginary-time evolution.

        Convenience wrapper: temporarily sets *evolution* to ``"imaginary"``,
        runs the trajectory, and returns the final energy.

        Parameters
        ----------
        init_params : array-like
            Starting parameters.
        t_final : float
            Total imaginary time.
        """
        old_evo = self.evolution
        self.evolution = "imaginary"
        try:
            result = self.evolve(init_params, t_final)
        finally:
            self.evolution = old_evo
        return result.final_energy

    # ------------------------------------------------------------------
    # State and energy helpers
    # ------------------------------------------------------------------

    def _psi(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate the ansatz state at parameters *theta*."""
        psi = np.asarray(self.ansatz(theta), dtype=complex).ravel()
        norm = float(np.linalg.norm(psi))
        if norm < 1e-14:
            raise McLachlanError("Ansatz returned a near-zero state vector.")
        return psi / norm

    def _energy(self, theta: np.ndarray, psi: Optional[np.ndarray] = None) -> float:
        """Compute ⟨ψ(θ)|H|ψ(θ)⟩ (real part)."""
        if psi is None:
            psi = self._psi(theta)
        return float(np.real(psi.conj() @ self._H_fn(psi)))

    # ------------------------------------------------------------------
    # Gradient (finite difference)
    # ------------------------------------------------------------------

    def _grad_psi(self, theta: np.ndarray) -> np.ndarray:
        """Compute ∂|ψ⟩/∂θᵢ for all i via central differences.

        Returns
        -------
        ndarray, shape (n_params, dim)
        """
        psi0 = self._psi(theta)
        dim = len(psi0)
        grads = np.zeros((self.n_params, dim), dtype=complex)
        eps = self.eps
        for i in range(self.n_params):
            th_p = theta.copy(); th_p[i] += eps
            th_m = theta.copy(); th_m[i] -= eps
            # Differentiate the *normalised* state to keep dynamics scale-invariant
            psi_p = self._psi(th_p)
            psi_m = self._psi(th_m)
            grads[i] = (psi_p - psi_m) / (2 * eps)
        return grads

    # ------------------------------------------------------------------
    # McLachlan metric and force
    # ------------------------------------------------------------------

    def _metric_and_force(
        self, theta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (M, V) for the McLachlan linear system M θ̇ = V.

        For real-time evolution:
            M_ij = Re⟨∂_i ψ|∂_j ψ⟩  (quantum geometric tensor, real part)
            V_i  = Re⟨∂_i ψ|(-iH)|ψ⟩ = Im⟨∂_i ψ|H|ψ⟩

        For imaginary-time evolution:
            M_ij = Re⟨∂_i ψ|∂_j ψ⟩
            V_i  = -Re⟨∂_i ψ|H|ψ⟩
        """
        psi = self._psi(theta)
        grads = self._grad_psi(theta)  # (n_params, dim)
        Hpsi = self._H_fn(psi)

        p = self.n_params
        M = np.zeros((p, p), dtype=float)
        for i in range(p):
            for j in range(i, p):
                val = float(np.real(grads[i].conj() @ grads[j]))
                M[i, j] = M[j, i] = val

        # Regularise
        M += self.reg * np.eye(p)

        V = np.zeros(p, dtype=float)
        for i in range(p):
            if self.evolution == "real":
                # Re⟨∂_i ψ|(-iH)|ψ⟩ = Im⟨∂_i ψ|H|ψ⟩
                V[i] = float(np.imag(grads[i].conj() @ Hpsi))
            else:
                # imaginary time: -Re⟨∂_i ψ|H|ψ⟩
                V[i] = -float(np.real(grads[i].conj() @ Hpsi))

        return M, V

    def _theta_dot(self, theta: np.ndarray) -> tuple[np.ndarray, bool]:
        """Solve M θ̇ = V and return (θ̇, converged)."""
        M, V = self._metric_and_force(theta)
        try:
            tdot = np.linalg.solve(M, V)
            ok = True
        except np.linalg.LinAlgError:
            tdot = np.zeros(self.n_params)
            ok = False
        return tdot, ok

    # ------------------------------------------------------------------
    # Integrators
    # ------------------------------------------------------------------

    def _euler_step(
        self, theta: np.ndarray, dt: float
    ) -> tuple[np.ndarray, bool]:
        tdot, ok = self._theta_dot(theta)
        return theta + dt * tdot, ok

    def _rk4_step(
        self, theta: np.ndarray, dt: float
    ) -> tuple[np.ndarray, bool]:
        k1, ok1 = self._theta_dot(theta)
        k2, ok2 = self._theta_dot(theta + 0.5 * dt * k1)
        k3, ok3 = self._theta_dot(theta + 0.5 * dt * k2)
        k4, ok4 = self._theta_dot(theta + dt * k3)
        tdot = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        ok = ok1 and ok2 and ok3 and ok4
        return theta + dt * tdot, ok

    def __repr__(self) -> str:
        return (
            f"McLachlanSolver(n_params={self.n_params}, dt={self.dt}, "
            f"evolution={self.evolution!r}, integrator={self.integrator!r})"
        )
