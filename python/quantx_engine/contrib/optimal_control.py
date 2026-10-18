"""Quantum Optimal Control — M84.

Implements two pulse-shaping algorithms:

- :class:`GRAPESolver` — Gradient Ascent Pulse Engineering (Khaneja et al. 2005).
  Computes analytical gradients of the gate fidelity with respect to piecewise-
  constant control amplitudes and performs gradient ascent.

- :class:`CRABSolver` — Chopped Random Basis (Doria et al. 2011).
  Parameterises each control as a sum of random Fourier harmonics and optimises
  the scalar coefficients with Nelder-Mead (gradient-free).

Both solvers work with pure-NumPy exact density-matrix / propagator arithmetic.
Matrix exponentials are computed via eigendecomposition of the (Hermitian)
total Hamiltonian at each time step, giving exact propagators for any step size.

Quick reference::

    import numpy as np
    from quantx_engine.contrib.optimal_control import GRAPESolver

    # Drive a single qubit from |0⟩ to |1⟩ using an X-rotation
    X = np.array([[0,1],[1,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)
    solver = GRAPESolver(
        H_drift=0.5 * Z,       # free precession along Z
        H_controls=[X],        # one control field along X
        U_target=np.array([[0,1],[1,0]], dtype=complex),  # X gate
        n_steps=20,
        total_time=math.pi,
    )
    result = solver.run(n_iterations=200, learning_rate=0.5)
    print(f"Fidelity: {result.fidelity_history[-1]:.6f}")   # → ≈ 1.0

References
----------
.. [1] Khaneja, N. et al. (2005). Optimal control of coupled spin dynamics:
       Design of NMR pulse sequences by gradient ascent algorithms.
       *J. Magn. Reson.* 172, 296–305.
       https://doi.org/10.1016/j.jmr.2004.11.004

.. [2] Caneva, T. et al. (2011). Chopped random-basis quantum optimization.
       *Phys. Rev. A* 84, 022326.
       https://doi.org/10.1103/PhysRevA.84.022326

.. [3] de Fouquieres, P. et al. (2011). Second order gradient ascent pulse
       engineering. *J. Magn. Reson.* 212, 412–417.
       https://doi.org/10.1016/j.jmr.2011.07.023

Physical validation
-------------------
GRAPE gradient for a single-qubit pi rotation: analytic gradient of
F = |<target|U>|^2 with respect to pulse amplitudes. The gradient has
magnitude ~ |target - U| * |dU/dparam|. For a Gaussian ansatz, dF/da_k
= 2*Re(<target|dU/da_k>) [Khaneja2005].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

__all__ = [
    "OCError",
    "GRAPEResult",
    "GRAPESolver",
    "CRABResult",
    "CRABSolver",
]


class OCError(ValueError):
    """Raised for invalid inputs to optimal control solvers."""


def _require_numpy() -> None:
    if not _HAS_NUMPY:  # pragma: no cover
        raise OCError("NumPy is required for optimal control.")


# ---------------------------------------------------------------------------
# Matrix exponential (Hermitian matrices via eigendecomposition)
# ---------------------------------------------------------------------------

def _expm_hermitian(H: "np.ndarray") -> "np.ndarray":
    """Exact matrix exponential of a Hermitian matrix H: returns e^H.

    Uses eigendecomposition: H = V Λ V†  →  e^H = V e^Λ V†.
    Exact for any step size; no Padé approximation required.
    """
    evals, evecs = np.linalg.eigh(H)
    return (evecs * np.exp(evals)) @ evecs.conj().T


def _prop(H_total: "np.ndarray", dt: float) -> "np.ndarray":
    """Compute the propagator U = exp(-i H dt) for a Hermitian H.

    Decomposes the *Hermitian* H (not the anti-Hermitian −iHdt) so that
    ``np.linalg.eigh`` receives a valid input and the phase factors are
    applied analytically: U = V diag(e^{−i dt λ_k}) V†.
    """
    evals, evecs = np.linalg.eigh(H_total)
    return (evecs * np.exp(-1j * dt * evals)) @ evecs.conj().T


# ---------------------------------------------------------------------------
# Gate fidelity (phase-insensitive)
# ---------------------------------------------------------------------------

def _fidelity(X: "np.ndarray", d: int) -> float:
    """Phase-insensitive fidelity Φ = |Tr(X)|² / d² where X = U†_target U_prop."""
    return float(abs(np.trace(X)) ** 2) / (d * d)


# ---------------------------------------------------------------------------
# GRAPESolver
# ---------------------------------------------------------------------------

@dataclass
class GRAPEResult:
    """Result of a GRAPE optimisation run.

    Attributes
    ----------
    controls:
        Final control amplitudes, shape ``(n_controls, n_steps)``.
    fidelity_history:
        Gate fidelity at each iteration.
    converged:
        True if final fidelity >= ``fidelity_threshold``.
    n_iterations:
        Number of iterations performed.
    """
    controls: "np.ndarray"
    fidelity_history: "list[float]"
    converged: bool
    n_iterations: int


class GRAPESolver:
    """GRAPE — Gradient Ascent Pulse Engineering.

    Maximises the gate fidelity:

        Φ = |Tr(U†_target U_prop)|² / d²

    with respect to piecewise-constant control amplitudes ``u[j, k]``
    (j = control index, k = time step index).

    The gradient is computed analytically:

        ∂Φ/∂u_j[k] = (2 dt / d²) Im(IP* · Tr(A[k+1] H_j F[k+1]))

    where ``F[k+1] = U_k … U_0`` (forward propagator through step k),
    ``A[k+1] = U†_target · U_{n-1} … U_{k+1}`` (backward propagator using
    *forward* unitaries, NOT their adjoints),
    and ``IP = Tr(U†_target U_prop)``.

    Parameters
    ----------
    H_drift:
        Time-independent drift Hamiltonian, shape ``(d, d)``, Hermitian.
    H_controls:
        List of control Hamiltonians H_j, each shape ``(d, d)``, Hermitian.
    U_target:
        Target unitary, shape ``(d, d)``.
    n_steps:
        Number of piecewise-constant time steps.
    total_time:
        Total evolution time (same units as Hamiltonian).
    seed:
        Optional seed for initial random controls.
    """

    def __init__(
        self,
        H_drift: "np.ndarray",
        H_controls: "list[np.ndarray]",
        U_target: "np.ndarray",
        n_steps: int = 20,
        total_time: float = math.pi,
        seed: "int | None" = None,
    ) -> None:
        _require_numpy()
        H_drift = np.asarray(H_drift, dtype=complex)
        if H_drift.ndim != 2 or H_drift.shape[0] != H_drift.shape[1]:
            raise OCError("H_drift must be a square matrix.")
        d = H_drift.shape[0]
        H_controls = [np.asarray(H, dtype=complex) for H in H_controls]
        for j, H in enumerate(H_controls):
            if H.shape != (d, d):
                raise OCError(
                    f"H_controls[{j}] must have shape ({d}, {d}), got {H.shape}."
                )
        U_target = np.asarray(U_target, dtype=complex)
        if U_target.shape != (d, d):
            raise OCError(
                f"U_target must have shape ({d}, {d}), got {U_target.shape}."
            )
        if n_steps < 1:
            raise OCError(f"n_steps must be >= 1, got {n_steps}.")
        if total_time <= 0.0:
            raise OCError(f"total_time must be > 0, got {total_time}.")
        if not H_controls:
            raise OCError("H_controls must not be empty.")

        self.H_drift = H_drift
        self.H_controls = H_controls
        self.U_target = U_target
        self.n_steps = n_steps
        self.total_time = total_time
        self.seed = seed
        self._d = d
        self._dt = total_time / n_steps
        self._n_controls = len(H_controls)

    def run(
        self,
        n_iterations: int = 200,
        learning_rate: float = 0.5,
        fidelity_threshold: float = 0.999,
        initial_controls: "np.ndarray | None" = None,
    ) -> GRAPEResult:
        """Run GRAPE optimisation.

        Parameters
        ----------
        n_iterations:
            Maximum number of gradient steps.
        learning_rate:
            Step size for gradient ascent.
        fidelity_threshold:
            Stop early when fidelity >= this value.
        initial_controls:
            Optional initial controls of shape ``(n_controls, n_steps)``.
            Initialised to small random values if None.

        Returns
        -------
        GRAPEResult
        """
        _require_numpy()
        if n_iterations < 1:
            raise OCError("n_iterations must be >= 1.")
        if learning_rate <= 0.0:
            raise OCError("learning_rate must be > 0.")
        if not (0.0 < fidelity_threshold <= 1.0):
            raise OCError("fidelity_threshold must be in (0, 1].")

        n = self.n_steps
        nc = self._n_controls
        d = self._d
        dt = self._dt
        rng = np.random.default_rng(self.seed)

        if initial_controls is not None:
            u = np.array(initial_controls, dtype=float)
            if u.shape != (nc, n):
                raise OCError(
                    f"initial_controls must have shape ({nc}, {n}), got {u.shape}."
                )
        else:
            # Initialise near the natural π/2-rotation scale so the total
            # propagator is not stuck near I (which would give IP≈0 for
            # traceless targets such as X and kill the gradient).
            _u_mean = math.pi / (2.0 * self.total_time)
            u = rng.standard_normal((nc, n)) * 0.3 + _u_mean

        fidelity_history: "list[float]" = []
        converged = False

        for _iter in range(n_iterations):
            # --- Forward pass ---
            # Build propagator U_k for each step and accumulate F[k]
            H_totals = [
                self.H_drift + sum(u[j, k] * self.H_controls[j] for j in range(nc))
                for k in range(n)
            ]
            props = [_prop(H_totals[k], dt) for k in range(n)]  # U_k

            # F[k] = U_k...U_1; F[-1] = I (using list, F_list[k] = U_k F_{k-1})
            F = [None] * (n + 1)  # F[0] = I, F[k] = props[k-1] @ F[k-1]
            F[0] = np.eye(d, dtype=complex)
            for k in range(1, n + 1):
                F[k] = props[k - 1] @ F[k - 1]

            U_prop = F[n]
            X = self.U_target.conj().T @ U_prop  # U†_target U_prop
            IP = np.trace(X)  # inner product scalar
            fid = float(abs(IP) ** 2) / (d * d)
            fidelity_history.append(fid)

            if fid >= fidelity_threshold:
                converged = True
                break

            # --- Backward pass ---
            # A[k] = U†_target · U_{n-1} · … · U_k  (products of FORWARD unitaries)
            # Recursion: A[n] = U†_target,  A[k] = A[k+1] @ props[k]  (multiply by U_k)
            A = [None] * (n + 1)
            A[n] = self.U_target.conj().T
            for k in range(n - 1, -1, -1):
                A[k] = A[k + 1] @ props[k]

            # --- Gradient computation ---
            # ∂Φ/∂u_j[k] = (2 dt / d²) Im( IP* · Tr(A[k+1] H_j F[k+1]) )
            # By trace cyclicity Tr(A H F) = Tr(F A H), so we pre-compute
            # FA = F[k+1] @ A[k+1] and evaluate Tr(FA @ H_j) efficiently.
            grad = np.zeros((nc, n), dtype=float)
            ip_conj = np.conj(IP)
            factor = 2.0 * dt / (d * d)
            for k in range(n):
                FA = F[k + 1] @ A[k + 1]  # Tr(A[k+1] H_j F[k+1]) = Tr(F[k+1] A[k+1] H_j)
                for j in range(nc):
                    inner = np.trace(FA @ self.H_controls[j])
                    grad[j, k] = factor * float(np.imag(ip_conj * inner))

            # --- Update controls ---
            u += learning_rate * grad

        return GRAPEResult(
            controls=u.copy(),
            fidelity_history=fidelity_history,
            converged=converged,
            n_iterations=len(fidelity_history),
        )


# ---------------------------------------------------------------------------
# CRABSolver
# ---------------------------------------------------------------------------

@dataclass
class CRABResult:
    """Result of a CRAB optimisation run.

    Attributes
    ----------
    controls:
        Final control amplitudes, shape ``(n_controls, n_steps)``, evaluated
        at the piecewise-constant time grid.
    fidelity_history:
        Fidelity evaluated at each Nelder-Mead function call (sparse).
    converged:
        True if final fidelity >= ``fidelity_threshold``.
    coefficients:
        Fourier coefficients optimised by Nelder-Mead,
        shape ``(n_controls, n_harmonics, 2)`` (cos/sin per harmonic).
    """
    controls: "np.ndarray"
    fidelity_history: "list[float]"
    converged: bool
    coefficients: "np.ndarray"


class CRABSolver:
    """CRAB — Chopped Random Basis optimal control.

    Represents each control field as a sum of n_harmonics random-frequency
    Fourier components:

        u_j(t) = ∑_k  (a_k cos(ω_k t) + b_k sin(ω_k t))

    where {ω_k} are drawn randomly near integer multiples of 2π/T.
    The scalar coefficients {a_k, b_k} are optimised by Nelder-Mead.

    Parameters
    ----------
    H_drift, H_controls, U_target, n_steps, total_time, seed:
        Same as :class:`GRAPESolver`.
    n_harmonics:
        Number of Fourier harmonics per control field.
    """

    def __init__(
        self,
        H_drift: "np.ndarray",
        H_controls: "list[np.ndarray]",
        U_target: "np.ndarray",
        n_steps: int = 20,
        total_time: float = math.pi,
        n_harmonics: int = 4,
        seed: "int | None" = None,
    ) -> None:
        _require_numpy()
        H_drift = np.asarray(H_drift, dtype=complex)
        if H_drift.ndim != 2 or H_drift.shape[0] != H_drift.shape[1]:
            raise OCError("H_drift must be a square matrix.")
        d = H_drift.shape[0]
        H_controls = [np.asarray(H, dtype=complex) for H in H_controls]
        for j, H in enumerate(H_controls):
            if H.shape != (d, d):
                raise OCError(
                    f"H_controls[{j}] must have shape ({d}, {d}), got {H.shape}."
                )
        U_target = np.asarray(U_target, dtype=complex)
        if U_target.shape != (d, d):
            raise OCError(
                f"U_target must have shape ({d}, {d}), got {U_target.shape}."
            )
        if n_steps < 1:
            raise OCError(f"n_steps must be >= 1, got {n_steps}.")
        if total_time <= 0.0:
            raise OCError(f"total_time must be > 0, got {total_time}.")
        if n_harmonics < 1:
            raise OCError(f"n_harmonics must be >= 1, got {n_harmonics}.")
        if not H_controls:
            raise OCError("H_controls must not be empty.")

        self.H_drift = H_drift
        self.H_controls = H_controls
        self.U_target = U_target
        self.n_steps = n_steps
        self.total_time = total_time
        self.n_harmonics = n_harmonics
        self.seed = seed
        self._d = d
        self._dt = total_time / n_steps
        self._n_controls = len(H_controls)

    def _build_controls(
        self,
        coeffs: "np.ndarray",
        t_grid: "np.ndarray",
        freqs: "np.ndarray",
    ) -> "np.ndarray":
        """Build control amplitudes u[j, k] from Fourier coefficients.

        Parameters
        ----------
        coeffs:
            Shape ``(n_controls, n_harmonics, 2)`` — cos/sin amplitudes.
        t_grid:
            Time sample points, shape ``(n_steps,)``.
        freqs:
            Random frequencies, shape ``(n_controls, n_harmonics)``.

        Returns
        -------
        Control array of shape ``(n_controls, n_steps)``.
        """
        nc = self._n_controls
        n = self.n_steps
        u = np.zeros((nc, n), dtype=float)
        for j in range(nc):
            for m in range(self.n_harmonics):
                omega = freqs[j, m]
                u[j] += coeffs[j, m, 0] * np.cos(omega * t_grid)
                u[j] += coeffs[j, m, 1] * np.sin(omega * t_grid)
        return u

    def _eval_fidelity(
        self,
        u: "np.ndarray",
    ) -> float:
        """Compute gate fidelity for a given control array u[j, k]."""
        n = self.n_steps
        nc = self._n_controls
        d = self._d
        dt = self._dt
        F = np.eye(d, dtype=complex)
        for k in range(n):
            H_k = self.H_drift + sum(
                u[j, k] * self.H_controls[j] for j in range(nc)
            )
            F = _prop(H_k, dt) @ F
        X = self.U_target.conj().T @ F
        return _fidelity(X, d)

    def run(
        self,
        n_iterations: int = 500,
        fidelity_threshold: float = 0.999,
    ) -> CRABResult:
        """Run CRAB optimisation using Nelder-Mead.

        Parameters
        ----------
        n_iterations:
            Maximum number of Nelder-Mead function evaluations.
        fidelity_threshold:
            Stop early when fidelity >= this value.

        Returns
        -------
        CRABResult
        """
        _require_numpy()
        if n_iterations < 1:
            raise OCError("n_iterations must be >= 1.")
        if not (0.0 < fidelity_threshold <= 1.0):
            raise OCError("fidelity_threshold must be in (0, 1].")

        n = self.n_steps
        nc = self._n_controls
        nh = self.n_harmonics
        T = self.total_time
        dt = self._dt

        rng = np.random.default_rng(self.seed)

        # Random frequencies: integer multiples of 2π/T with small random offset
        base_freqs = (
            rng.integers(1, nh + 2, size=(nc, nh)).astype(float) * (2 * math.pi / T)
            + rng.standard_normal((nc, nh)) * (math.pi / T)
        )

        # Time grid (midpoints of each step)
        t_grid = np.arange(n, dtype=float) * dt + dt / 2.0

        fidelity_history: "list[float]" = []

        # Nelder-Mead via a simple pure-NumPy implementation
        # Parameter vector: flatten coeffs (nc * nh * 2 scalars)
        n_params = nc * nh * 2

        def objective(x: "np.ndarray") -> float:
            coeffs = x.reshape(nc, nh, 2)
            u = self._build_controls(coeffs, t_grid, base_freqs)
            fid = self._eval_fidelity(u)
            fidelity_history.append(fid)
            return -fid  # minimise negative fidelity

        x0 = rng.standard_normal(n_params) * 0.05

        # Simple Nelder-Mead (pure NumPy, no scipy dependency).
        # stop_val = -fidelity_threshold because we minimise -fidelity.
        x_opt = _nelder_mead(
            objective, x0,
            max_iter=n_iterations,
            stop_val=-fidelity_threshold,
        )

        coeffs_opt = x_opt.reshape(nc, nh, 2)
        u_opt = self._build_controls(coeffs_opt, t_grid, base_freqs)
        final_fid = self._eval_fidelity(u_opt)
        fidelity_history.append(final_fid)

        return CRABResult(
            controls=u_opt,
            fidelity_history=fidelity_history,
            converged=final_fid >= fidelity_threshold,
            coefficients=coeffs_opt,
        )


# ---------------------------------------------------------------------------
# Pure-NumPy Nelder-Mead minimiser
# ---------------------------------------------------------------------------

def _nelder_mead(
    func: "callable",
    x0: "np.ndarray",
    max_iter: int = 500,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
    tol: float = 1e-6,
    stop_val: float = -float("inf"),
) -> "np.ndarray":
    """Minimise func(x) via the Nelder-Mead simplex algorithm.

    Parameters
    ----------
    func:
        Scalar objective function.
    x0:
        Initial parameter vector.
    max_iter:
        Maximum number of function evaluations.
    alpha, gamma, rho, sigma:
        Standard Nelder-Mead reflection/expansion/contraction/shrink factors.
    tol:
        Convergence tolerance on simplex diameter.
    stop_val:
        Stop early (success) when ``func(x) <= stop_val``.  The default
        ``-inf`` effectively disables early stopping.

    Returns
    -------
    Best parameter vector found.
    """
    n = len(x0)
    # Initialise simplex
    simplex = [x0.copy()]
    f0 = func(x0)
    f_vals = [f0]
    if f0 <= stop_val:
        return x0.copy()
    step = max(0.05 * np.max(np.abs(x0)), 0.05)
    for i in range(n):
        x = x0.copy()
        x[i] += step
        fi = func(x)
        simplex.append(x)
        f_vals.append(fi)
        if fi <= stop_val:
            return x.copy()

    n_evals = n + 1

    while n_evals < max_iter:
        # Sort
        order = np.argsort(f_vals)
        simplex = [simplex[i] for i in order]
        f_vals = [f_vals[i] for i in order]

        # Early-stop: best vertex already meets threshold
        if f_vals[0] <= stop_val:
            break

        # Check convergence
        diam = max(np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, n + 1))
        if diam < tol:
            break

        # Centroid of best n points
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = func(x_r)
        n_evals += 1
        if f_r <= stop_val:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            break
        if n_evals >= max_iter:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            break

        if f_vals[0] <= f_r < f_vals[-2]:
            simplex[-1] = x_r
            f_vals[-1] = f_r
            continue

        if f_r < f_vals[0]:
            # Expansion
            x_e = centroid + gamma * (x_r - centroid)
            f_e = func(x_e)
            n_evals += 1
            if f_e < f_r:
                simplex[-1] = x_e
                f_vals[-1] = f_e
            else:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            if min(f_e, f_r) <= stop_val:
                break
            continue

        # Contraction
        if f_r < f_vals[-1]:
            x_c = centroid + rho * (x_r - centroid)
        else:
            x_c = centroid + rho * (simplex[-1] - centroid)
        f_c = func(x_c)
        n_evals += 1
        if f_c < f_vals[-1]:
            simplex[-1] = x_c
            f_vals[-1] = f_c
            if f_c <= stop_val:
                break
            continue

        # Shrink
        new_simplex = [simplex[0]]
        new_fvals = [f_vals[0]]
        for i in range(1, n + 1):
            x_s = simplex[0] + sigma * (simplex[i] - simplex[0])
            f_s = func(x_s)
            n_evals += 1
            new_simplex.append(x_s)
            new_fvals.append(f_s)
            if f_s <= stop_val or n_evals >= max_iter:
                break
        simplex = new_simplex
        f_vals = new_fvals

    # Return best found
    best_idx = int(np.argmin(f_vals))
    return simplex[best_idx]
