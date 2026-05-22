"""Quantum walk algorithms for the QuantX engine (M100).

Two walk models are provided:

**DiscreteTimeWalk** — coined quantum walk on an undirected graph.
The walker lives in a joint coin ⊗ position Hilbert space.  One step
applies a coin operator (Hadamard, Grover, or custom) followed by a
conditional shift.  Efficient for cycle graphs, complete graphs, and
hypercubes; extensible to arbitrary adjacency matrices.

**ContinuousTimeQuantumWalk (CTQW)** — Hamiltonian is the graph
Laplacian (or adjacency matrix).  State evolves as
``|ψ(t)⟩ = exp(−iHt)|ψ₀⟩`` via Krylov exponentiation (no full
diagonalisation for large graphs).

**GroverWalk** — marks one or more target nodes and uses the
quantum walk to achieve quadratic speedup over classical random
search.  Returns the first step at which the marked-node
probability exceeds a threshold (hitting time).

Quick start::

    import numpy as np
    from quantx_engine.quantum_walk import DiscreteTimeWalk, ContinuousTimeQuantumWalk

    # Discrete walk on a 16-node cycle, 20 steps
    walk = DiscreteTimeWalk.cycle(n=16, coin="hadamard")
    result = walk.run(steps=20)
    print(result.position_probs)      # shape (16,)

    # Continuous walk on the same cycle
    ctqw = ContinuousTimeQuantumWalk.from_adjacency(walk.adjacency)
    result_c = ctqw.run(t=5.0)
    print(result_c.position_probs)

References
----------
.. [1] Aharonov, Y. et al. (1993). Quantum random walks.
       *Phys. Rev. A* 48, 1687.
       https://doi.org/10.1103/PhysRevA.48.1687

.. [2] Aharonov, D., Ambainis, A., Kempe, J. & Vazirani, U. (2001).
       Quantum walks on graphs. *STOC 2001*, 50–59.
       https://doi.org/10.1145/380752.380758

.. [3] Grover, L. K. (1996). A fast quantum mechanical algorithm for database
       search. *STOC 1996*, 212–219.
       https://doi.org/10.1145/237814.237866

Physical validation
-------------------
Continuous-time quantum walk on a cycle of N nodes: probability P(0->k,t) =
|J_k(2t)|^2 in the large-N limit (Bessel function), where J_k is the k-th
order Bessel function of the first kind [Aharonov1993].
Discrete-time Grover walk: quadratic speedup for finding a marked vertex
[Grover1996]; amplitude at the marked vertex grows as O(1/sqrt(N)) steps.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# ── Errors ────────────────────────────────────────────────────────────────────

class QuantumWalkError(ValueError):
    """Raised when walk construction or evolution parameters are invalid."""


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class WalkResult:
    """Outcome of a single quantum walk run.

    Attributes
    ----------
    position_probs : np.ndarray
        Probability of finding the walker at each node (shape ``(n_nodes,)``).
    state : np.ndarray
        Final (normalised) state vector in the full walk Hilbert space.
    steps_or_time : float
        Number of discrete steps taken, or total continuous time elapsed.
    mixing_time : int | None
        Smallest step / time at which ``|position_probs - uniform|_1 < eps``
        (set to ``None`` if not computed or never reached).
    metadata : dict
        Extra diagnostics (e.g. coin type, graph size).
    """
    position_probs: np.ndarray
    state: np.ndarray
    steps_or_time: float
    mixing_time: Optional[int]
    metadata: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.position_probs)

    def tvd_from_uniform(self) -> float:
        """Total variation distance from the uniform distribution."""
        uniform = np.ones(self.n_nodes) / self.n_nodes
        return 0.5 * float(np.sum(np.abs(self.position_probs - uniform)))


@dataclass
class GroverWalkResult:
    """Outcome of a Grover quantum walk search.

    Attributes
    ----------
    marked_nodes : list[int]
        Indices of the target nodes.
    hitting_time : int | None
        First step where marked-node probability exceeds *threshold*.
        ``None`` if the threshold was not reached within *max_steps*.
    max_marked_prob : float
        Maximum probability of landing on any marked node (over all steps).
    step_probs : list[float]
        Probability of being on a marked node at each step.
    position_probs : np.ndarray
        Position probability distribution at the step with highest
        marked-node probability.
    """
    marked_nodes: list[int]
    hitting_time: Optional[int]
    max_marked_prob: float
    step_probs: list[float]
    position_probs: np.ndarray


# ── Coin operators ─────────────────────────────────────────────────────────────

def _hadamard_coin(d: int) -> np.ndarray:
    """d-dimensional Hadamard (DFT) coin unitary."""
    if d == 2:
        return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    # General: d×d DFT matrix
    idx = np.arange(d)
    return np.exp(2j * math.pi * np.outer(idx, idx) / d) / math.sqrt(d)


def _grover_coin(d: int) -> np.ndarray:
    """Grover diffusion coin: C = 2/d * J - I (J = all-ones matrix)."""
    C = (2.0 / d) * np.ones((d, d), dtype=complex) - np.eye(d, dtype=complex)
    return C


def _identity_coin(d: int) -> np.ndarray:
    return np.eye(d, dtype=complex)


def _make_coin(coin: str | np.ndarray, d: int) -> np.ndarray:
    if isinstance(coin, np.ndarray):
        if coin.shape != (d, d):
            raise QuantumWalkError(
                f"Custom coin must be {d}×{d}, got {coin.shape}"
            )
        c = coin.astype(complex)
        eye = np.eye(d, dtype=complex)
        if not np.allclose(c.conj().T @ c, eye, atol=1e-10):
            raise QuantumWalkError(
                "Custom coin must be unitary: C†C ≠ I. "
                "Pass a unitary matrix or use 'hadamard', 'grover', 'identity'."
            )
        if not np.allclose(c @ c.conj().T, eye, atol=1e-10):
            raise QuantumWalkError(
                "Custom coin must be unitary: CC† ≠ I. "
                "Pass a unitary matrix or use 'hadamard', 'grover', 'identity'."
            )
        return c
    name = coin.lower()
    if name == "hadamard":
        return _hadamard_coin(d)
    if name in ("grover", "diffusion"):
        return _grover_coin(d)
    if name == "identity":
        return _identity_coin(d)
    raise QuantumWalkError(
        f"Unknown coin {coin!r}. Use 'hadamard', 'grover', 'identity', or an ndarray."
    )


# ── DiscreteTimeWalk ───────────────────────────────────────────────────────────

class DiscreteTimeWalk:
    """Coined discrete-time quantum walk on an undirected graph.

    The Hilbert space is ``coin ⊗ position`` with dimension
    ``(max_degree × n_nodes)``.  The walk operator is ``W = S · (C ⊗ I)``
    where *C* is the coin and *S* is the conditional shift.

    Parameters
    ----------
    adjacency : np.ndarray
        ``n×n`` symmetric integer adjacency matrix (0/1 entries).
    coin : str or np.ndarray
        Coin operator: ``'hadamard'`` (default), ``'grover'``, ``'identity'``,
        or a custom unitary ``d×d`` array where *d* = max degree.
    initial_pos : int
        Starting node (default 0).
    initial_coin_state : np.ndarray or None
        Initial coin state (length *d*).  Defaults to uniform superposition.
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        coin: str | np.ndarray = "hadamard",
        initial_pos: int = 0,
        initial_coin_state: Optional[np.ndarray] = None,
    ) -> None:
        adj = np.asarray(adjacency, dtype=int)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise QuantumWalkError("adjacency must be a square 2-D array")
        n = adj.shape[0]
        if not np.array_equal(adj, adj.T):
            raise QuantumWalkError("adjacency must be symmetric (undirected graph)")
        if not (0 <= initial_pos < n):
            raise QuantumWalkError(
                f"initial_pos {initial_pos} out of range [0, {n})"
            )

        self._adj = adj
        self._n = n
        # Build neighbour lists
        self._neighbours: list[list[int]] = [
            list(np.where(adj[i] > 0)[0]) for i in range(n)
        ]
        self._degrees = np.array([len(nb) for nb in self._neighbours], dtype=int)
        d = int(self._degrees.max()) if n > 1 else 1
        if d == 0:
            raise QuantumWalkError("Graph has no edges")
        # Coined quantum walk requires a regular graph so the flip-flop shift
        # is a well-defined permutation matrix.
        if self._degrees.min() != d:
            raise QuantumWalkError(
                f"DiscreteTimeWalk requires a regular graph "
                f"(all nodes must have the same degree). "
                f"Got degrees ranging from {int(self._degrees.min())} to {d}. "
                f"Use ContinuousTimeQuantumWalk for irregular graphs."
            )
        self._d = d  # coin dimension = degree

        self._coin_op = _make_coin(coin, d)

        # ── Initial state ──────────────────────────────────────────────────
        if initial_coin_state is None:
            c0 = np.ones(d, dtype=complex) / math.sqrt(d)
        else:
            c0 = np.asarray(initial_coin_state, dtype=complex)
            if c0.shape != (d,):
                raise QuantumWalkError(
                    f"initial_coin_state must have length {d}"
                )
            nm = np.linalg.norm(c0)
            if nm < 1e-12:
                raise QuantumWalkError("initial_coin_state has zero norm")
            c0 = c0 / nm

        # State vector: index (c * n + v) where c ∈ [0,d), v ∈ [0,n)
        self._state = np.zeros(d * n, dtype=complex)
        for c in range(d):
            self._state[c * n + initial_pos] = c0[c]
        # Snapshot of the initial state — never mutated, used by GroverWalk.search
        self._initial_state = self._state.copy()

        # Precompute coin ⊗ I  (d·n × d·n)
        self._coin_full = np.kron(self._coin_op, np.eye(n, dtype=complex))

        # Precompute shift operator
        self._shift = self._build_shift()

        # Walk operator W = S · (C ⊗ I)
        self._W = self._shift @ self._coin_full

    # ── Graph factory helpers ──────────────────────────────────────────────

    @classmethod
    def cycle(
        cls,
        n: int,
        coin: str | np.ndarray = "hadamard",
        initial_pos: int = 0,
    ) -> "DiscreteTimeWalk":
        """Cycle graph C_n."""
        if n < 3:
            raise QuantumWalkError("Cycle graph requires n ≥ 3")
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            adj[i, (i + 1) % n] = 1
            adj[i, (i - 1) % n] = 1
        return cls(adj, coin=coin, initial_pos=initial_pos)

    @classmethod
    def complete(
        cls,
        n: int,
        coin: str | np.ndarray = "grover",
        initial_pos: int = 0,
    ) -> "DiscreteTimeWalk":
        """Complete graph K_n."""
        if n < 2:
            raise QuantumWalkError("Complete graph requires n ≥ 2")
        adj = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
        return cls(adj, coin=coin, initial_pos=initial_pos)

    @classmethod
    def hypercube(
        cls,
        dim: int,
        coin: str | np.ndarray = "hadamard",
        initial_pos: int = 0,
    ) -> "DiscreteTimeWalk":
        """d-dimensional hypercube (2^d nodes, each with degree d)."""
        if dim < 1:
            raise QuantumWalkError("Hypercube dimension must be ≥ 1")
        n = 2 ** dim
        adj = np.zeros((n, n), dtype=int)
        for v in range(n):
            for k in range(dim):
                nb = v ^ (1 << k)  # flip bit k
                adj[v, nb] = 1
        return cls(adj, coin=coin, initial_pos=initial_pos)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _build_shift(self) -> np.ndarray:
        """Flip-flop conditional shift: |c, v⟩ → |c_rev, u⟩.

        u = neighbours[v][c] and c_rev is the index of v in neighbours[u].
        This produces a permutation matrix (hence unitary) for any regular
        undirected graph.
        """
        n, d = self._n, self._d
        S = np.zeros((d * n, d * n), dtype=complex)
        for v in range(n):
            nbs = self._neighbours[v]
            for c in range(d):
                u = nbs[c]
                # Reverse direction: index of v in neighbours[u]
                c_rev = self._neighbours[u].index(v)
                S[c_rev * n + u, c * n + v] = 1.0
        return S

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def adjacency(self) -> np.ndarray:
        return self._adj.copy()

    @property
    def n_nodes(self) -> int:
        return self._n

    @property
    def coin_dim(self) -> int:
        return self._d

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def position_probs(self) -> np.ndarray:
        """Marginal position probability distribution."""
        d, n = self._d, self._n
        probs = np.zeros(n)
        for c in range(d):
            probs += np.abs(self._state[c * n:(c + 1) * n]) ** 2
        return probs

    # ── Evolution ─────────────────────────────────────────────────────────

    def step(self) -> "DiscreteTimeWalk":
        """Apply one walk step W = S·(C⊗I) in-place.  Returns *self*."""
        self._state = self._W @ self._state
        return self

    def run(
        self,
        steps: int,
        track_mixing: bool = False,
        mixing_eps: float = 0.01,
    ) -> WalkResult:
        """Evolve for *steps* steps.

        Parameters
        ----------
        steps : int
            Number of walk steps to apply.
        track_mixing : bool
            If True, record the first step where TVD from uniform < *mixing_eps*.
        mixing_eps : float
            TVD threshold for mixing time detection.

        Returns
        -------
        WalkResult
        """
        if steps < 0:
            raise QuantumWalkError(f"steps must be non-negative, got {steps}")

        mixing_time: Optional[int] = None
        uniform = np.ones(self._n) / self._n

        for t in range(steps):
            self._state = self._W @ self._state
            if track_mixing and mixing_time is None:
                pp = self.position_probs()
                if 0.5 * np.sum(np.abs(pp - uniform)) < mixing_eps:
                    mixing_time = t + 1

        probs = self.position_probs()
        return WalkResult(
            position_probs=probs,
            state=self._state.copy(),
            steps_or_time=float(steps),
            mixing_time=mixing_time,
            metadata={
                "n_nodes": self._n,
                "coin_dim": self._d,
                "coin": "custom" if isinstance(self._coin_op, np.ndarray) else "ndarray",
            },
        )


# ── ContinuousTimeQuantumWalk ─────────────────────────────────────────────────

class ContinuousTimeQuantumWalk:
    """Continuous-time quantum walk (CTQW) on a graph.

    The Hamiltonian is the graph adjacency matrix *A* (or Laplacian *L = D − A*).
    State evolves as ``|ψ(t)⟩ = exp(−i H t)|ψ₀⟩`` computed via the
    Krylov-Arnoldi algorithm — no full diagonalisation for large graphs.

    Parameters
    ----------
    hamiltonian : np.ndarray
        ``n×n`` Hermitian matrix (adjacency or Laplacian).
    initial_state : np.ndarray or None
        Initial state (length *n*).  Defaults to ``|0⟩``.
    """

    def __init__(
        self,
        hamiltonian: np.ndarray,
        initial_state: Optional[np.ndarray] = None,
    ) -> None:
        H = np.asarray(hamiltonian, dtype=complex)
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise QuantumWalkError("Hamiltonian must be a square 2-D array")
        if not np.allclose(H, H.conj().T, atol=1e-10):
            raise QuantumWalkError("Hamiltonian must be Hermitian")
        self._H = H
        self._n = H.shape[0]

        if initial_state is None:
            psi0 = np.zeros(self._n, dtype=complex)
            psi0[0] = 1.0
        else:
            psi0 = np.asarray(initial_state, dtype=complex)
            if psi0.shape != (self._n,):
                raise QuantumWalkError(
                    f"initial_state must have length {self._n}"
                )
            nm = np.linalg.norm(psi0)
            if nm < 1e-12:
                raise QuantumWalkError("initial_state has zero norm")
            psi0 = psi0 / nm
        self._psi0 = psi0

    # ── Factory helpers ────────────────────────────────────────────────────

    @classmethod
    def from_adjacency(
        cls,
        adjacency: np.ndarray,
        use_laplacian: bool = False,
        initial_state: Optional[np.ndarray] = None,
    ) -> "ContinuousTimeQuantumWalk":
        """Build CTQW from an adjacency matrix (or Laplacian if *use_laplacian*)."""
        A = np.asarray(adjacency, dtype=float)
        if use_laplacian:
            D = np.diag(A.sum(axis=1))
            H = (D - A).astype(complex)
        else:
            H = A.astype(complex)
        return cls(H, initial_state=initial_state)

    @classmethod
    def cycle(
        cls,
        n: int,
        use_laplacian: bool = False,
        initial_node: int = 0,
    ) -> "ContinuousTimeQuantumWalk":
        """Cycle graph C_n."""
        if n < 3:
            raise QuantumWalkError("Cycle graph requires n ≥ 3")
        adj = np.zeros((n, n))
        for i in range(n):
            adj[i, (i + 1) % n] = 1
            adj[i, (i - 1) % n] = 1
        psi0 = np.zeros(n, dtype=complex)
        psi0[initial_node] = 1.0
        return cls.from_adjacency(adj, use_laplacian=use_laplacian, initial_state=psi0)

    @classmethod
    def complete(
        cls,
        n: int,
        use_laplacian: bool = False,
        initial_node: int = 0,
    ) -> "ContinuousTimeQuantumWalk":
        """Complete graph K_n."""
        adj = np.ones((n, n)) - np.eye(n)
        psi0 = np.zeros(n, dtype=complex)
        psi0[initial_node] = 1.0
        return cls.from_adjacency(adj, use_laplacian=use_laplacian, initial_state=psi0)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def n_nodes(self) -> int:
        return self._n

    # ── Evolution ─────────────────────────────────────────────────────────

    def run(
        self,
        t: float,
        krylov_dim: int = 30,
    ) -> WalkResult:
        """Evolve for time *t* using Krylov-Arnoldi exponentiation.

        Parameters
        ----------
        t : float
            Evolution time (non-negative).
        krylov_dim : int
            Krylov subspace dimension (capped at *n*).

        Returns
        -------
        WalkResult
        """
        if t < 0:
            raise QuantumWalkError(f"t must be non-negative, got {t}")

        psi_t = self._krylov_expm(self._H, self._psi0, t, krylov_dim)
        probs = np.abs(psi_t) ** 2

        return WalkResult(
            position_probs=probs,
            state=psi_t,
            steps_or_time=float(t),
            mixing_time=None,
            metadata={"n_nodes": self._n, "t": t},
        )

    def propagate(
        self,
        t_values: Sequence[float],
        krylov_dim: int = 30,
    ) -> list[WalkResult]:
        """Compute position distributions at each time in *t_values*.

        Parameters
        ----------
        t_values : sequence of float
            Sorted sequence of times (must be non-negative and non-decreasing).
        krylov_dim : int
            Krylov subspace dimension.

        Returns
        -------
        list[WalkResult]
            One :class:`WalkResult` per time value.
        """
        tv = list(t_values)
        if any(t < 0 for t in tv):
            raise QuantumWalkError("All t_values must be non-negative")

        results = []
        psi = self._psi0.copy()
        t_prev = 0.0
        for t in tv:
            dt = t - t_prev
            psi = self._krylov_expm(self._H, psi, dt, krylov_dim)
            probs = np.abs(psi) ** 2
            results.append(WalkResult(
                position_probs=probs,
                state=psi.copy(),
                steps_or_time=float(t),
                mixing_time=None,
                metadata={"n_nodes": self._n, "t": t},
            ))
            t_prev = t
        return results

    # ── Internal ───────────────────────────────────────────────────────────

    @staticmethod
    def _krylov_expm(
        H: np.ndarray,
        v: np.ndarray,
        t: float,
        m: int,
    ) -> np.ndarray:
        """Arnoldi-Krylov approximation to exp(−i H t)|v⟩.

        Falls back to exact scipy expm for small matrices.
        """
        n = len(v)
        if t == 0.0:
            return v.copy()
        m = min(m, n)

        # For small n, exact exponentiation is faster and more accurate
        if n <= 64:
            try:
                from scipy.linalg import expm as sp_expm
                U = sp_expm(-1j * t * H)
                return U @ v
            except ImportError:
                pass

        # Arnoldi iteration to build Krylov basis Q and Hessenberg H_m
        Q = np.zeros((n, m + 1), dtype=complex)
        H_m = np.zeros((m + 1, m), dtype=complex)
        Q[:, 0] = v / np.linalg.norm(v)
        breakdown = m
        for j in range(m):
            w = H @ Q[:, j]
            for i in range(j + 1):
                H_m[i, j] = np.dot(Q[:, i].conj(), w)
                w -= H_m[i, j] * Q[:, i]
            beta = np.linalg.norm(w)
            H_m[j + 1, j] = beta
            if beta < 1e-12:
                breakdown = j + 1
                break
            Q[:, j + 1] = w / beta

        dm = breakdown
        Hk = H_m[:dm, :dm]
        # exp(−i t Hk)
        try:
            from scipy.linalg import expm as sp_expm
            ek = sp_expm(-1j * t * Hk)
        except ImportError:
            evals, evecs = np.linalg.eigh(Hk)
            ek = (evecs * np.exp(-1j * t * evals)) @ evecs.conj().T

        e0 = np.zeros(dm, dtype=complex)
        e0[0] = np.linalg.norm(v)
        result = Q[:, :dm] @ (ek @ e0)
        return result


# ── GroverWalk ────────────────────────────────────────────────────────────────

class GroverWalk:
    """Quantum walk search using the Grover diffusion coin.

    Uses a discrete-time quantum walk with a modified oracle step that flips
    the phase of the coin state at marked nodes, biasing the walk toward them.

    Parameters
    ----------
    graph : DiscreteTimeWalk
        A pre-configured walk (typically on a cycle or complete graph with
        Grover coin).  The walk is reset to its initial state each call to
        :meth:`search`.
    marked_nodes : list[int]
        Indices of nodes to find.
    threshold : float
        Success probability threshold (default 0.5).
    """

    def __init__(
        self,
        graph: DiscreteTimeWalk,
        marked_nodes: Sequence[int],
        threshold: float = 0.5,
    ) -> None:
        for node in marked_nodes:
            if not (0 <= node < graph.n_nodes):
                raise QuantumWalkError(
                    f"Marked node {node} out of range [0, {graph.n_nodes})"
                )
        if not (0 < threshold <= 1):
            raise QuantumWalkError("threshold must be in (0, 1]")
        self._walk = graph
        self._marked = list(marked_nodes)
        self._threshold = threshold

        # Oracle: flip phase on marked nodes (acts on position subspace)
        n, d = graph.n_nodes, graph.coin_dim
        oracle_diag = np.ones(d * n, dtype=complex)
        for v in self._marked:
            for c in range(d):
                oracle_diag[c * n + v] = -1.0
        self._oracle = np.diag(oracle_diag)

    def search(self, max_steps: int = 1000) -> GroverWalkResult:
        """Run the Grover quantum walk search.

        Each step applies the oracle then the walk operator.

        Parameters
        ----------
        max_steps : int
            Maximum steps before giving up.

        Returns
        -------
        GroverWalkResult
        """
        # Build the Grover walk operator: W_G = W · Oracle
        W_G = self._walk._W @ self._oracle

        # Always start from the walk's construction-time initial state so that
        # prior run()/step() calls on the shared DiscreteTimeWalk do not affect
        # hitting-time measurements.
        state = self._walk._initial_state.copy()
        n, d = self._walk.n_nodes, self._walk.coin_dim
        step_probs: list[float] = []
        hitting_time: Optional[int] = None
        best_prob = 0.0
        best_probs = np.zeros(n)
        best_step = 0

        for step in range(1, max_steps + 1):
            state = W_G @ state
            # Position marginal
            pos_probs = np.zeros(n)
            for c in range(d):
                pos_probs += np.abs(state[c * n:(c + 1) * n]) ** 2
            marked_prob = float(np.sum(pos_probs[self._marked]))
            step_probs.append(marked_prob)

            if marked_prob > best_prob:
                best_prob = marked_prob
                best_probs = pos_probs.copy()
                best_step = step

            if hitting_time is None and marked_prob >= self._threshold:
                hitting_time = step

        return GroverWalkResult(
            marked_nodes=list(self._marked),
            hitting_time=hitting_time,
            max_marked_prob=best_prob,
            step_probs=step_probs,
            position_probs=best_probs,
        )


# ── Utility functions ──────────────────────────────────────────────────────────

def walk_spectrum(walk: DiscreteTimeWalk) -> np.ndarray:
    """Return the eigenvalues of the walk operator W.

    The quasi-energy spectrum of *W* determines the walk's spreading speed
    and mixing properties.  For a symmetric walk the eigenvalues lie on the
    unit circle.

    Returns
    -------
    np.ndarray
        Complex array of eigenvalues of W, shape ``(d · n,)``.
    """
    return np.linalg.eigvals(walk._W)


def ctqw_transfer_probability(
    walk: ContinuousTimeQuantumWalk,
    source: int,
    target: int,
    t: float,
) -> float:
    """Probability amplitude squared to transfer from *source* to *target* at time *t*.

    Parameters
    ----------
    walk : ContinuousTimeQuantumWalk
    source : int
        Initial node index.
    target : int
        Target node index.
    t : float
        Evolution time.

    Returns
    -------
    float
        Transfer probability |⟨target|e^{−iHt}|source⟩|².
    """
    n = walk.n_nodes
    if not (0 <= source < n and 0 <= target < n):
        raise QuantumWalkError(
            f"Node indices must be in [0, {n}): got source={source}, target={target}"
        )
    psi0 = np.zeros(n, dtype=complex)
    psi0[source] = 1.0
    result = walk.__class__(walk._H, initial_state=psi0).run(t)
    return float(result.position_probs[target])
