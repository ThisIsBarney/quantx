"""M115: QAOA with CVaR objective — combinatorial optimisation.

Provides a complete QAOA pipeline for combinatorial optimisation problems:

* **CVaR-QAOA** (:class:`CVaRQAOA`) — replaces the standard ⟨H⟩ objective with
  the Conditional Value-at-Risk CVaR_α, which focuses training on the best-α
  fraction of measurement outcomes and significantly improves convergence.

* **MaxCut** (:class:`MaxCutProblem`) — builds the Ising cost Hamiltonian for
  graph max-cut and evaluates approximation ratios.

* **TSP** (:class:`TSPProblem`) — encodes the Traveling Salesman Problem as a
  QUBO/Ising Hamiltonian using the standard one-hot city-position encoding.

* **Parameter landscape** (:class:`QAOALandscape`) — scans a (γ, β) grid at
  depth p=1 and returns the full energy surface for visualisation and analysis.

* **Warm-start initialisation** (:func:`warm_start_qaoa`) — returns informed
  initial parameter guesses via several strategies.

Classes
-------
CVaRQAOAError
QAOAResult
MaxCutProblem
TSPProblem
CVaRQAOA
QAOALandscapeResult
QAOALandscape

Functions
---------
warm_start_qaoa

References
----------
.. [1] Farhi, E., Goldstone, J. & Gutmann, S. (2014). A quantum approximate
       optimization algorithm. arXiv:1411.4028.
       https://doi.org/10.48550/arXiv.1411.4028

.. [2] Bravyi, S. et al. (2020). Obstacles to variational quantum optimization
       from symmetry protection. *Phys. Rev. Lett.* 125, 260505.
       https://doi.org/10.1103/PhysRevLett.125.260505

.. [3] Wurtz, J. & Love, P. (2021). MaxCut quantum approximate optimization
       algorithm performance guarantees for p>1. *Phys. Rev. A* 103, 042612.
       https://doi.org/10.1103/PhysRevA.103.042612

Physical validation
-------------------
QAOA p=1 on MaxCut for a 3-regular graph: approximation ratio >= 0.6924
[Farhi2014]. CVaR (alpha=0.25) improves the effective expected value by
focusing on the best alpha-fraction of samples.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .chemistry import PauliHamiltonian, PauliTerm

__all__ = [
    "CVaRQAOAError",
    "QAOAResult",
    "MaxCutProblem",
    "TSPProblem",
    "CVaRQAOA",
    "QAOALandscapeResult",
    "QAOALandscape",
    "warm_start_qaoa",
]


class CVaRQAOAError(ValueError):
    """Raised for invalid CVaR-QAOA requests."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_cost_phase(psi: np.ndarray, cost_diag: np.ndarray, gamma: float) -> np.ndarray:
    """Apply e^{-iγH_C}; H_C diagonal → phase rotation per basis state."""
    return psi * np.exp(-1j * gamma * cost_diag)


def _apply_mixer(psi: np.ndarray, beta: float, n_qubits: int) -> np.ndarray:
    """Apply ⊗_i e^{-iβX_i} (standard QAOA mixer) to state psi.

    e^{-iβX} = [[cos β, -i sin β], [-i sin β, cos β]]
    """
    cb = np.cos(beta)
    isb = -1j * np.sin(beta)
    Rx = np.array([[cb, isb], [isb, cb]], dtype=complex)
    dim = 1 << n_qubits
    state = psi.reshape([2] * n_qubits)
    for q in range(n_qubits):
        state = np.tensordot(Rx, state, axes=[[1], [q]])
        state = np.moveaxis(state, 0, q)
    return state.reshape(dim)


def _qaoa_statevector(cost_diag: np.ndarray, params: np.ndarray,
                      n_qubits: int) -> np.ndarray:
    """Compute QAOA statevector for given (gamma, beta) parameters.

    Parameters
    ----------
    cost_diag : np.ndarray, shape (2**n,)
        Cost values C(x) for each computational basis state |x⟩.
    params : np.ndarray, shape (2*p,)
        Concatenated [γ_1, …, γ_p, β_1, …, β_p].
    n_qubits : int

    Returns
    -------
    np.ndarray, shape (2**n,) complex
    """
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]
    dim = 1 << n_qubits

    # Initial state: uniform superposition |+⟩^⊗n
    psi = np.ones(dim, dtype=complex) / math.sqrt(dim)

    for layer in range(p):
        psi = _apply_cost_phase(psi, cost_diag, float(gammas[layer]))
        psi = _apply_mixer(psi, float(betas[layer]), n_qubits)

    return psi


def _cost_diagonal(cost_fn, n_qubits: int) -> np.ndarray:
    """Evaluate cost_fn for all 2**n basis states."""
    dim = 1 << n_qubits
    return np.array([cost_fn(x) for x in range(dim)], dtype=float)


def _cvar(psi: np.ndarray, cost_diag: np.ndarray, alpha: float) -> float:
    """CVaR_α objective: expected cost conditioned on lowest-α fraction.

    For minimisation: lower CVaR is better.
    """
    probs = np.abs(psi) ** 2
    idx = np.argsort(cost_diag)
    sorted_costs = cost_diag[idx]
    sorted_probs = probs[idx]

    cum_prob = 0.0
    cvar_sum = 0.0
    for cost, prob in zip(sorted_costs, sorted_probs):
        remaining = alpha - cum_prob
        if remaining <= 0.0:
            break
        contrib = min(prob, remaining)
        cvar_sum += cost * contrib
        cum_prob += contrib

    return float(cvar_sum / alpha) if alpha > 1e-15 else float(sorted_costs[0])


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class QAOAResult:
    """Result of a :class:`CVaRQAOA` optimisation.

    Attributes
    ----------
    optimal_params : np.ndarray, shape (2*p,)
        Optimised [γ_1, …, γ_p, β_1, …, β_p].
    optimal_value : float
        CVaR_α value at the optimal parameters.
    expectation_value : float
        ⟨H_C⟩ at the optimal parameters (standard expectation).
    approximation_ratio : float | None
        Ratio ⟨H_C⟩ / exact_optimum if known; else None.
    n_layers : int
        QAOA depth p.
    alpha : float
        CVaR level used.
    n_qubits : int
    """
    optimal_params: np.ndarray
    optimal_value: float
    expectation_value: float
    approximation_ratio: Optional[float]
    n_layers: int
    alpha: float
    n_qubits: int

    def to_dict(self) -> dict:
        return {
            "optimal_params": self.optimal_params.tolist(),
            "optimal_value": self.optimal_value,
            "expectation_value": self.expectation_value,
            "approximation_ratio": self.approximation_ratio,
            "n_layers": self.n_layers,
            "alpha": self.alpha,
            "n_qubits": self.n_qubits,
        }


# ── MaxCutProblem ─────────────────────────────────────────────────────────────

class MaxCutProblem:
    """Graph Maximum Cut problem.

    The cost function is C(x) = Σ_{(i,j)∈E} w_{ij} (1 − z_i z_j)/2
    where z_i = 1 − 2x_i (z_i=+1 if qubit i is 0, z_i=−1 if qubit i is 1).

    Parameters
    ----------
    n_nodes : int
        Number of graph nodes (qubits).
    edges : list[tuple[int, int]] | list[tuple[int, int, float]]
        Edge list.  Each entry is (u, v) or (u, v, weight); default weight=1.
    """

    def __init__(
        self,
        n_nodes: int,
        edges: list,
    ) -> None:
        if n_nodes < 1:
            raise CVaRQAOAError(f"n_nodes must be ≥ 1, got {n_nodes}")
        self.n_nodes = n_nodes
        self._edges: list[tuple[int, int, float]] = []
        for e in edges:
            if len(e) == 2:
                u, v = int(e[0]), int(e[1])
                w = 1.0
            else:
                u, v, w = int(e[0]), int(e[1]), float(e[2])
            if not (0 <= u < n_nodes and 0 <= v < n_nodes):
                raise CVaRQAOAError(f"Edge ({u},{v}) out of range for n_nodes={n_nodes}")
            self._edges.append((u, v, w))

    @property
    def edges(self) -> list[tuple[int, int, float]]:
        return list(self._edges)

    def cut_value(self, x: int) -> float:
        """Evaluate cut value for bitstring x (integer representation)."""
        total = 0.0
        for u, v, w in self._edges:
            bit_u = (x >> u) & 1
            bit_v = (x >> v) & 1
            if bit_u != bit_v:
                total += w
        return total

    def cost_diagonal(self) -> np.ndarray:
        """Compute C(x) = −cut_value(x) for all 2^n basis states.

        Negative because QAOA minimises; maximising cut ↔ minimising −cut.
        """
        dim = 1 << self.n_nodes
        return np.array([-self.cut_value(x) for x in range(dim)], dtype=float)

    def cost_hamiltonian(self) -> PauliHamiltonian:
        """Build the Ising cost Hamiltonian H_C = −Σ w_{ij}(1−Z_iZ_j)/2."""
        terms = []
        for u, v, w in self._edges:
            # −w/2 * (I − Z_u Z_v) = −w/2 + (w/2) Z_u Z_v
            terms.append(PauliTerm(coeff=complex(-w / 2), ops={}))
            terms.append(PauliTerm(coeff=complex(w / 2), ops={u: "Z", v: "Z"}))
        if not terms:
            terms.append(PauliTerm(coeff=0j, ops={}))
        return PauliHamiltonian(terms=terms, n_qubits=self.n_nodes)

    def exact_max_cut(self) -> tuple[float, int]:
        """Brute-force exact max-cut value and the achieving bitstring.

        Returns
        -------
        (max_cut_value, bitstring)
        """
        dim = 1 << self.n_nodes
        best_val = -1.0
        best_x = 0
        for x in range(dim):
            v = self.cut_value(x)
            if v > best_val:
                best_val = v
                best_x = x
        return best_val, best_x

    def approximation_ratio(self, x: int) -> float:
        """Approximation ratio cut_value(x) / exact_max_cut."""
        exact, _ = self.exact_max_cut()
        if exact == 0:
            return 1.0
        return self.cut_value(x) / exact


# ── TSPProblem ────────────────────────────────────────────────────────────────

class TSPProblem:
    """Traveling Salesman Problem encoded as an Ising Hamiltonian.

    Uses a one-hot encoding: qubit x_{i,p} = 1 iff city i is visited at
    position p.  n² qubits total for n cities.

    Parameters
    ----------
    distances : np.ndarray, shape (n, n)
        Symmetric distance matrix (distances[i,j] = distance from city i to j).
    penalty : float, default 10.0
        Penalty coefficient for constraint violations.
    """

    def __init__(self, distances: np.ndarray, penalty: float = 10.0) -> None:
        dist = np.asarray(distances, dtype=float)
        if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
            raise CVaRQAOAError("distances must be a square 2-D array")
        if dist.shape[0] < 2:
            raise CVaRQAOAError("TSP requires at least 2 cities")
        self._dist = dist
        self.n_cities = dist.shape[0]
        self.penalty = penalty
        self.n_qubits = self.n_cities ** 2

    def _qubit(self, city: int, position: int) -> int:
        """Qubit index for (city, position) pair."""
        return city * self.n_cities + position

    def tour_length(self, tour: list[int]) -> float:
        """Compute total tour length for a city permutation.

        Parameters
        ----------
        tour : list[int]
            Sequence of city indices, e.g. [0, 2, 1] for 3 cities.
        """
        n = self.n_cities
        if len(tour) != n or set(tour) != set(range(n)):
            raise CVaRQAOAError("tour must be a permutation of 0..n_cities-1")
        total = 0.0
        for k in range(n):
            total += self._dist[tour[k], tour[(k + 1) % n]]
        return total

    def decode_bitstring(self, x: int) -> Optional[list[int]]:
        """Decode a bitstring to a tour if the assignment is valid.

        Returns
        -------
        list[int] | None
            Tour order if the bitstring is a valid one-hot assignment;
            ``None`` otherwise.
        """
        n = self.n_cities
        tour = [-1] * n
        for city in range(n):
            count = 0
            for pos in range(n):
                if (x >> self._qubit(city, pos)) & 1:
                    count += 1
                    tour[pos] = city
            if count != 1:
                return None
        # Check each position has exactly one city
        if -1 in tour or len(set(tour)) != n:
            return None
        return tour

    def cost_diagonal(self) -> np.ndarray:
        """QUBO cost for all 2^(n²) basis states."""
        dim = 1 << self.n_qubits
        n = self.n_cities
        costs = np.zeros(dim, dtype=float)

        for x in range(dim):
            # Distance term
            dist_cost = 0.0
            for city_i in range(n):
                for city_j in range(n):
                    if city_i == city_j:
                        continue
                    for pos in range(n):
                        q_ip = self._qubit(city_i, pos)
                        q_jnext = self._qubit(city_j, (pos + 1) % n)
                        if (x >> q_ip) & 1 and (x >> q_jnext) & 1:
                            dist_cost += self._dist[city_i, city_j]

            # Constraint penalty: each city visited exactly once
            city_penalty = 0.0
            for city in range(n):
                s = sum((x >> self._qubit(city, pos)) & 1 for pos in range(n))
                city_penalty += (s - 1) ** 2

            # Each position has exactly one city
            pos_penalty = 0.0
            for pos in range(n):
                s = sum((x >> self._qubit(city, pos)) & 1 for city in range(n))
                pos_penalty += (s - 1) ** 2

            costs[x] = dist_cost + self.penalty * (city_penalty + pos_penalty)

        return costs

    def cost_hamiltonian(self) -> PauliHamiltonian:
        """Build simplified QUBO Hamiltonian (diagonal Z terms only)."""
        n = self.n_cities
        nq = self.n_qubits
        terms = []

        # Distance terms: x_{i,p} * x_{j,p+1} * d_{ij}
        # (1/4)(I - Z_{ip})(I - Z_{j,p+1}) * d_{ij}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                for pos in range(n):
                    q1 = self._qubit(i, pos)
                    q2 = self._qubit(j, (pos + 1) % n)
                    d = self._dist[i, j]
                    # (I - Z_q1)(I - Z_q2)/4 → d/4 * (I - Z1 - Z2 + Z1Z2)
                    terms.append(PauliTerm(coeff=complex(d / 4), ops={}))
                    terms.append(PauliTerm(coeff=complex(-d / 4), ops={q1: "Z"}))
                    terms.append(PauliTerm(coeff=complex(-d / 4), ops={q2: "Z"}))
                    terms.append(PauliTerm(coeff=complex(d / 4), ops={q1: "Z", q2: "Z"}))

        # Penalty terms: Σ_i (Σ_p x_{ip} - 1)²
        A = self.penalty
        for i in range(n):
            qs = [self._qubit(i, pos) for pos in range(n)]
            _add_one_hot_penalty(terms, qs, A)
        for pos in range(n):
            qs = [self._qubit(i, pos) for i in range(n)]
            _add_one_hot_penalty(terms, qs, A)

        if not terms:
            terms.append(PauliTerm(coeff=0j, ops={}))
        return PauliHamiltonian(terms=terms, n_qubits=nq)


def _add_one_hot_penalty(
    terms: list[PauliTerm], qubits: list[int], A: float
) -> None:
    """Add penalty terms for (Σ_q x_q - 1)² using x_q = (I - Z_q)/2."""
    # (Σ x_q - 1)² = Σ x_q² + 2 Σ_{p<q} x_p x_q - 2 Σ x_q + 1
    # x_q² = x_q, so: (n-2) Σ x_q + 2 Σ_{p<q} x_p x_q + 1
    # substitute x_q = (I - Z_q)/2
    n = len(qubits)
    # Constant: A(1 - n/2 + n(n-1)/4 * ...)
    # Easier to just expand directly:
    # A * (Σ_q (I-Z_q)/2 - 1)²
    # = A * (n/2 - Σ Z_q/2 - 1)²
    # = A * (  (n/2-1)² - (n/2-1)Σ Z_q + (1/4)(Σ Z_q)²  )
    c = n / 2 - 1
    # Constant term
    terms.append(PauliTerm(coeff=complex(A * (c * c + n / 4)), ops={}))
    # Linear Z terms
    for q in qubits:
        terms.append(PauliTerm(coeff=complex(A * (-c - 1 / 2 + n / 4 * 2 / n)), ops={q: "Z"}))
    # ZZ cross terms
    for idx_p in range(n):
        for idx_q in range(idx_p + 1, n):
            terms.append(PauliTerm(coeff=complex(A * 0.5),
                                   ops={qubits[idx_p]: "Z", qubits[idx_q]: "Z"}))


# ── CVaRQAOA ─────────────────────────────────────────────────────────────────

class CVaRQAOA:
    """QAOA with Conditional Value-at-Risk (CVaR) objective.

    Replaces the standard expectation-value objective ⟨H_C⟩ with
    CVaR_α = E[C(x) | C(x) ≤ VaR_α], which focuses training on the
    best-α fraction of measurement outcomes.

    Parameters
    ----------
    cost_diagonal : np.ndarray, shape (2**n,)
        Pre-computed cost C(x) for every computational basis state.
    n_qubits : int
    n_layers : int, default 1
        QAOA depth p.
    alpha : float, default 0.25
        CVaR confidence level ∈ (0, 1].  alpha=1 recovers standard QAOA.
    exact_optimum : float | None
        Known exact optimum for computing approximation ratio.
    """

    def __init__(
        self,
        cost_diagonal: np.ndarray,
        n_qubits: int,
        n_layers: int = 1,
        alpha: float = 0.25,
        exact_optimum: Optional[float] = None,
    ) -> None:
        cost_diagonal = np.asarray(cost_diagonal, dtype=float)
        if cost_diagonal.shape != (1 << n_qubits,):
            raise CVaRQAOAError(
                f"cost_diagonal must have shape ({1 << n_qubits},) for "
                f"{n_qubits} qubits, got {cost_diagonal.shape}"
            )
        if n_layers < 1:
            raise CVaRQAOAError(f"n_layers must be ≥ 1, got {n_layers}")
        if not (0.0 < alpha <= 1.0):
            raise CVaRQAOAError(f"alpha must be in (0, 1], got {alpha}")

        self.cost_diagonal = cost_diagonal
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.alpha = alpha
        self.exact_optimum = exact_optimum

    def objective(self, params: np.ndarray) -> float:
        """Compute CVaR_α at given parameters.

        Parameters
        ----------
        params : np.ndarray, shape (2*p,)
            [γ_1, …, γ_p, β_1, …, β_p]
        """
        psi = _qaoa_statevector(self.cost_diagonal, params, self.n_qubits)
        return _cvar(psi, self.cost_diagonal, self.alpha)

    def expectation(self, params: np.ndarray) -> float:
        """Compute ⟨H_C⟩ (standard expectation value) at given parameters."""
        psi = _qaoa_statevector(self.cost_diagonal, params, self.n_qubits)
        probs = np.abs(psi) ** 2
        return float(np.dot(probs, self.cost_diagonal))

    def optimize(
        self,
        initial_params: Optional[np.ndarray] = None,
        method: str = "COBYLA",
        max_iter: int = 200,
        seed: Optional[int] = None,
    ) -> QAOAResult:
        """Optimise CVaR objective over QAOA parameters.

        Parameters
        ----------
        initial_params : np.ndarray | None
            Initial [γ, β] parameters.  If None, uses random initialisation.
        method : str
            Optimiser name passed to ``scipy.optimize.minimize``.
        max_iter : int
            Maximum number of objective evaluations.
        seed : int | None

        Returns
        -------
        QAOAResult
        """
        try:
            from scipy.optimize import minimize  # type: ignore
        except ImportError:
            # Fallback: simple random restart grid search
            return self._optimize_grid(initial_params, seed)

        rng = np.random.default_rng(seed)
        p = self.n_layers
        if initial_params is None:
            initial_params = rng.uniform(0, math.pi, 2 * p)
        initial_params = np.asarray(initial_params, dtype=float)

        res = minimize(
            self.objective,
            initial_params,
            method=method,
            options={"maxiter": max_iter, "rhobeg": 0.3},
        )
        opt_params = res.x
        opt_val = float(res.fun)
        exp_val = self.expectation(opt_params)

        approx = None
        if self.exact_optimum is not None and self.exact_optimum != 0:
            approx = float(exp_val / self.exact_optimum)

        return QAOAResult(
            optimal_params=opt_params,
            optimal_value=opt_val,
            expectation_value=exp_val,
            approximation_ratio=approx,
            n_layers=p,
            alpha=self.alpha,
            n_qubits=self.n_qubits,
        )

    def _optimize_grid(
        self,
        initial_params: Optional[np.ndarray],
        seed: Optional[int],
    ) -> QAOAResult:
        """Fallback grid-search optimiser (no scipy required)."""
        rng = np.random.default_rng(seed)
        p = self.n_layers
        n_restarts = 20
        best_val = float("inf")
        best_params = rng.uniform(0, math.pi, 2 * p)

        for _ in range(n_restarts):
            params = rng.uniform(0, math.pi, 2 * p)
            val = self.objective(params)
            if val < best_val:
                best_val = val
                best_params = params.copy()

        if initial_params is not None:
            val = self.objective(initial_params)
            if val < best_val:
                best_val = val
                best_params = initial_params.copy()

        # Simple coordinate descent
        params = best_params.copy()
        step = 0.2
        for _ in range(100):
            improved = False
            for i in range(2 * p):
                for delta in [step, -step]:
                    p2 = params.copy()
                    p2[i] += delta
                    v = self.objective(p2)
                    if v < best_val:
                        best_val = v
                        best_params = p2.copy()
                        params = p2.copy()
                        improved = True
            if not improved:
                step *= 0.7
            if step < 1e-4:
                break

        exp_val = self.expectation(best_params)
        approx = None
        if self.exact_optimum is not None and self.exact_optimum != 0:
            approx = float(exp_val / self.exact_optimum)

        return QAOAResult(
            optimal_params=best_params,
            optimal_value=best_val,
            expectation_value=exp_val,
            approximation_ratio=approx,
            n_layers=p,
            alpha=self.alpha,
            n_qubits=self.n_qubits,
        )


# ── QAOALandscape ─────────────────────────────────────────────────────────────

@dataclass
class QAOALandscapeResult:
    """Result of :class:`QAOALandscape`.

    Attributes
    ----------
    gamma_values : np.ndarray, shape (n_gamma,)
    beta_values : np.ndarray, shape (n_beta,)
    energy_surface : np.ndarray, shape (n_gamma, n_beta)
        ⟨H_C⟩ at each (γ, β) point.
    cvar_surface : np.ndarray, shape (n_gamma, n_beta)
        CVaR_α at each (γ, β) point.
    optimal_gamma : float
        γ value minimising energy_surface.
    optimal_beta : float
        β value minimising energy_surface.
    alpha : float
    """
    gamma_values: np.ndarray
    beta_values: np.ndarray
    energy_surface: np.ndarray
    cvar_surface: np.ndarray
    optimal_gamma: float
    optimal_beta: float
    alpha: float

    def to_dict(self) -> dict:
        return {
            "gamma_values": self.gamma_values.tolist(),
            "beta_values": self.beta_values.tolist(),
            "energy_surface": self.energy_surface.tolist(),
            "cvar_surface": self.cvar_surface.tolist(),
            "optimal_gamma": self.optimal_gamma,
            "optimal_beta": self.optimal_beta,
            "alpha": self.alpha,
        }


class QAOALandscape:
    """Parameter landscape scanner for depth-1 QAOA.

    Parameters
    ----------
    cost_diagonal : np.ndarray, shape (2**n,)
        Cost values C(x) for every computational basis state.
    n_qubits : int
    alpha : float, default 0.25
        CVaR level for the CVaR surface.
    """

    def __init__(
        self,
        cost_diagonal: np.ndarray,
        n_qubits: int,
        alpha: float = 0.25,
    ) -> None:
        self.cost_diagonal = np.asarray(cost_diagonal, dtype=float)
        self.n_qubits = n_qubits
        self.alpha = alpha

    def compute(
        self,
        gamma_range: tuple[float, float] = (0.0, math.pi),
        beta_range: tuple[float, float] = (0.0, math.pi / 2),
        n_gamma: int = 20,
        n_beta: int = 20,
    ) -> QAOALandscapeResult:
        """Compute energy landscape over a (γ, β) grid at depth p=1.

        Parameters
        ----------
        gamma_range, beta_range : tuple[float, float]
            Range for the parameter sweep.
        n_gamma, n_beta : int
            Grid resolution.

        Returns
        -------
        QAOALandscapeResult
        """
        gammas = np.linspace(*gamma_range, n_gamma)
        betas = np.linspace(*beta_range, n_beta)
        n = self.n_qubits
        energy_surface = np.zeros((n_gamma, n_beta))
        cvar_surface = np.zeros((n_gamma, n_beta))

        for ig, gamma in enumerate(gammas):
            for ib, beta in enumerate(betas):
                params = np.array([gamma, beta])
                psi = _qaoa_statevector(self.cost_diagonal, params, n)
                probs = np.abs(psi) ** 2
                energy_surface[ig, ib] = float(np.dot(probs, self.cost_diagonal))
                cvar_surface[ig, ib] = _cvar(psi, self.cost_diagonal, self.alpha)

        # Find minimum
        min_idx = np.unravel_index(np.argmin(energy_surface), energy_surface.shape)
        opt_gamma = float(gammas[min_idx[0]])
        opt_beta = float(betas[min_idx[1]])

        return QAOALandscapeResult(
            gamma_values=gammas,
            beta_values=betas,
            energy_surface=energy_surface,
            cvar_surface=cvar_surface,
            optimal_gamma=opt_gamma,
            optimal_beta=opt_beta,
            alpha=self.alpha,
        )


# ── warm_start_qaoa ───────────────────────────────────────────────────────────

def warm_start_qaoa(
    n_layers: int,
    strategy: str = "random",
    seed: Optional[int] = None,
    prev_params: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return initial QAOA parameter guess using a given strategy.

    Parameters
    ----------
    n_layers : int
        QAOA depth p.
    strategy : {"random", "fixed", "interpolated", "tqa"}
        Initialisation strategy:

        * ``"random"`` — uniform random in [0, π].
        * ``"fixed"`` — γ_k = 0.1k/p, β_k = π/4 (1 − k/p).
        * ``"interpolated"`` — linearly interpolate ``prev_params``
          (depth p−1) to depth p.  Requires ``prev_params``.
        * ``"tqa"`` — Trotterised Quantum Annealing schedule.
    seed : int | None
        Random seed (used only by ``"random"``).
    prev_params : np.ndarray | None
        Previous depth's parameters (needed for ``"interpolated"``).

    Returns
    -------
    np.ndarray, shape (2*p,)
        Concatenated [γ_1, …, γ_p, β_1, …, β_p].
    """
    if n_layers < 1:
        raise CVaRQAOAError(f"n_layers must be ≥ 1, got {n_layers}")
    if strategy not in ("random", "fixed", "interpolated", "tqa"):
        raise CVaRQAOAError(
            f"strategy must be 'random', 'fixed', 'interpolated', or 'tqa', "
            f"got {strategy!r}"
        )

    p = n_layers

    if strategy == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(0, math.pi, 2 * p).astype(float)

    elif strategy == "fixed":
        ts = np.linspace(1 / p, 1.0, p)
        gammas = 0.1 * ts
        betas = (math.pi / 4) * (1.0 - ts)
        return np.concatenate([gammas, betas])

    elif strategy == "interpolated":
        if prev_params is None:
            raise CVaRQAOAError(
                "'interpolated' strategy requires prev_params from depth p-1"
            )
        prev = np.asarray(prev_params, dtype=float)
        prev_p = len(prev) // 2
        # Linearly interpolate from p_prev points to p points
        old_times = np.linspace(0, 1, max(prev_p, 1))
        new_times = np.linspace(0, 1, p)
        gammas = np.interp(new_times, old_times, prev[:prev_p])
        betas = np.interp(new_times, old_times, prev[prev_p:])
        return np.concatenate([gammas, betas])

    else:  # "tqa"
        ts = np.linspace(1 / p, 1.0, p)
        gammas = ts * (math.pi / 2)
        betas = (1.0 - ts) * (math.pi / 2)
        return np.concatenate([gammas, betas])
