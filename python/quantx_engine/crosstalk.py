"""Crosstalk and ZZ coupling model — M90.

Models parasitic qubit–qubit interactions of the form

    H_ZZ = Σ_{(j,k)} ζ_{jk} / 2 · σz_j ⊗ σz_k

which accumulate phase during idle periods and degrade two-qubit gate
fidelity.  The module provides:

* :class:`CouplingGraph` — weighted graph of ZZ coupling strengths.
* :class:`CrosstalkModel` — builds the full idle Hamiltonian H₀ + H_ZZ and
  integrates it to compute idle crosstalk phase accumulation.
* :func:`zz_routing_cost` — SABRE-compatible edge-cost function that adds
  ZZ coupling weight to standard swap costs.
* :func:`echo_sequence` — convenience function that generates a
  :class:`~quantx_engine.pulse.PulseSchedule` with X–X refocusing echoes
  to cancel ZZ accumulation between two qubits.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .pulse import (
    ConstantWaveform,
    DriveChannel,
    PulseError,
    PulseResult,
    PulseSchedule,
    PulseSimulator,
    QubitSystem,
    pi_pulse_area,
    GaussianWaveform,
    _kron_op,
    _Z2,
    _X2,
    _Y2,
    _I2,
    _build_static_hamiltonian,
    _build_drive_ops,
    _build_jump_ops,
)


# ── Errors ───────────────────────────────────────────────────────────────────

class CrosstalkError(ValueError):
    """Raised on invalid crosstalk configuration."""


# ── CouplingGraph ────────────────────────────────────────────────────────────

@dataclass
class CouplingGraph:
    """Weighted undirected graph of qubit–qubit ZZ coupling strengths.

    Parameters
    ----------
    n_qubits : int
    edges : list of (int, int, float)
        Tuples ``(q0, q1, zeta)`` where *zeta* is the ZZ coupling strength
        in the same angular-frequency unit used elsewhere (e.g. rad/ns).
        Self-loops are rejected.
    """

    n_qubits: int
    edges: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise CrosstalkError("n_qubits must be >= 1")
        for q0, q1, zeta in self.edges:
            if q0 == q1:
                raise CrosstalkError(f"Self-loop not allowed: qubit {q0}")
            if not (0 <= q0 < self.n_qubits and 0 <= q1 < self.n_qubits):
                raise CrosstalkError(
                    f"Qubit indices ({q0}, {q1}) out of range [0, {self.n_qubits})"
                )

    # ------------------------------------------------------------------
    def zeta(self, q0: int, q1: int) -> float:
        """Return the ZZ coupling strength between qubits *q0* and *q1* (0 if absent)."""
        for a, b, z in self.edges:
            if (a == q0 and b == q1) or (a == q1 and b == q0):
                return float(z)
        return 0.0

    def neighbors(self, qubit: int) -> list:
        """Return list of (neighbor_qubit, zeta) pairs for *qubit*."""
        result = []
        for a, b, z in self.edges:
            if a == qubit:
                result.append((b, float(z)))
            elif b == qubit:
                result.append((a, float(z)))
        return result

    def adjacency_matrix(self) -> np.ndarray:
        """Return an (n_qubits × n_qubits) matrix of ZZ strengths."""
        mat = np.zeros((self.n_qubits, self.n_qubits))
        for a, b, z in self.edges:
            mat[a, b] = z
            mat[b, a] = z
        return mat

    def __repr__(self) -> str:
        return (
            f"CouplingGraph(n_qubits={self.n_qubits}, "
            f"n_edges={len(self.edges)})"
        )


# ── Build ZZ Hamiltonian ─────────────────────────────────────────────────────

def _build_zz_hamiltonian(graph: CouplingGraph) -> np.ndarray:
    """H_ZZ = Σ_{(j,k)} ζ_{jk}/2 · σz_j ⊗ σz_k."""
    n = graph.n_qubits
    d = 2 ** n
    H_zz = np.zeros((d, d), dtype=complex)
    for q0, q1, zeta in graph.edges:
        Zj = _kron_op(_Z2, q0, n)
        Zk = _kron_op(_Z2, q1, n)
        H_zz += (zeta / 2.0) * (Zj @ Zk)
    return H_zz


# ── CrosstalkModel ────────────────────────────────────────────────────────────

class CrosstalkModel:
    """Idle Hamiltonian evolution including ZZ crosstalk.

    Adds H_ZZ to the static rotating-frame Hamiltonian of a
    :class:`~quantx_engine.pulse.QubitSystem` and simulates idle evolution.

    Parameters
    ----------
    system : QubitSystem
    graph : CouplingGraph
        ZZ coupling strengths.  Must have the same ``n_qubits``.
    """

    def __init__(self, system: QubitSystem, graph: CouplingGraph) -> None:
        if system.n_qubits != graph.n_qubits:
            raise CrosstalkError(
                f"system.n_qubits ({system.n_qubits}) != graph.n_qubits ({graph.n_qubits})"
            )
        self.system = system
        self.graph = graph
        self._H_zz = _build_zz_hamiltonian(graph)

    # ------------------------------------------------------------------
    def zz_phase(self, q0: int, q1: int, idle_time: float) -> float:
        """Accumulated ZZ phase (rad) between *q0* and *q1* during an idle period.

        φ_ZZ = ζ_{q0,q1} · idle_time
        """
        return self.graph.zeta(q0, q1) * idle_time

    def idle_unitary(self, idle_time: float) -> np.ndarray:
        """Exact unitary U = exp(−i (H₀ + H_ZZ) · idle_time)."""
        n = self.system.n_qubits
        H0 = _build_static_hamiltonian(self.system)
        H_total = H0 + self._H_zz
        eigvals, eigvecs = np.linalg.eigh(H_total)
        phases = np.exp(-1j * eigvals * idle_time)
        return (eigvecs * phases) @ eigvecs.conj().T

    def evolve_idle(
        self,
        idle_time: float,
        initial_state: Optional[np.ndarray] = None,
    ) -> PulseResult:
        """Simulate idle evolution (no drive) for *idle_time*.

        Uses the exact matrix exponential rather than ODE integration for
        efficiency.

        Parameters
        ----------
        idle_time : float
        initial_state : array of shape (2^n,), optional
            Defaults to |0…0⟩.

        Returns
        -------
        PulseResult
            Populations and final state after idle evolution.
        """
        n = self.system.n_qubits
        d = 2 ** n
        if idle_time < 0:
            raise CrosstalkError("idle_time must be non-negative")

        if initial_state is None:
            psi0 = np.zeros(d, dtype=complex)
            psi0[0] = 1.0
        else:
            psi0 = np.asarray(initial_state, dtype=complex).copy()
            nm = np.linalg.norm(psi0)
            if nm < 1e-12:
                raise CrosstalkError("initial_state has zero norm")
            psi0 /= nm

        U = self.idle_unitary(idle_time)
        psi_f = U @ psi0
        psi_f /= np.linalg.norm(psi_f)

        pops = {
            format(i, f"0{n}b"): float(v)
            for i, v in enumerate(np.abs(psi_f) ** 2)
            if v > 1e-12
        }
        return PulseResult(
            final_state=psi_f,
            populations=pops,
            times=np.array([0.0, idle_time]),
            metadata={"n_qubits": n, "duration": idle_time, "noisy": False,
                      "zz_model": True},
        )

    def __repr__(self) -> str:
        return (
            f"CrosstalkModel(n_qubits={self.system.n_qubits}, "
            f"n_zz_pairs={len(self.graph.edges)})"
        )


# ── ZZ-aware routing cost ─────────────────────────────────────────────────────

def zz_routing_cost(
    graph: CouplingGraph,
    q0: int,
    q1: int,
    *,
    swap_base_cost: float = 1.0,
    zz_weight: float = 1.0,
) -> float:
    """SABRE-compatible edge cost that penalises high-ZZ pairs.

    Cost = swap_base_cost + zz_weight · |ζ_{q0,q1}|

    A router minimising this cost prefers to route through low-ZZ neighbours,
    reducing idle crosstalk accumulated during SWAP decomposition.

    Parameters
    ----------
    graph : CouplingGraph
    q0, q1 : int
        Candidate SWAP edge.
    swap_base_cost : float
        Fixed cost per SWAP (default 1.0).
    zz_weight : float
        Scaling factor for the ZZ penalty (default 1.0).

    Returns
    -------
    float
        Edge cost ≥ swap_base_cost.
    """
    return swap_base_cost + zz_weight * abs(graph.zeta(q0, q1))


def cheapest_path(
    graph: CouplingGraph,
    source: int,
    target: int,
    *,
    swap_base_cost: float = 1.0,
    zz_weight: float = 1.0,
) -> tuple:
    """Dijkstra shortest path minimising :func:`zz_routing_cost`.

    Returns
    -------
    (path, total_cost) : (list[int], float)
        *path* is the ordered list of qubit indices from *source* to *target*
        (inclusive).  Returns ``([], inf)`` if no path exists.
    """
    import heapq  # noqa: PLC0415

    dist: dict = {source: 0.0}
    prev: dict = {}
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf):
            continue
        if u == target:
            # Reconstruct path
            path = []
            cur = target
            while cur in prev:
                path.append(cur)
                cur = prev[cur]
            path.append(source)
            path.reverse()
            return path, dist[target]
        for v, zeta in graph.neighbors(u):
            cost = d + zz_routing_cost(graph, u, v,
                                        swap_base_cost=swap_base_cost,
                                        zz_weight=zz_weight)
            if cost < dist.get(v, math.inf):
                dist[v] = cost
                prev[v] = u
                heapq.heappush(pq, (cost, v))

    return [], math.inf


# ── Echo sequence ─────────────────────────────────────────────────────────────

def echo_sequence(
    q0: int,
    q1: int,
    idle_time: float,
    *,
    pi_duration: float = 20.0,
    pi_sigma: Optional[float] = None,
    use_gaussian: bool = False,
) -> PulseSchedule:
    """Build an X–X refocusing echo schedule to cancel ZZ accumulation.

    The sequence is:

        idle(T/2) – X(q0)·X(q1) – idle(T/2)

    The two X pulses at T/2 flip both qubits simultaneously, reversing the
    sign of the ZZ phase so that the net ZZ accumulation cancels to zero.

    Parameters
    ----------
    q0, q1 : int
        Qubit indices to echo.
    idle_time : float
        Total idle period to refocus.
    pi_duration : float
        Duration of the π-pulse (used as ``T`` for ConstantWaveform or
        ``duration`` for GaussianWaveform).
    pi_sigma : float, optional
        Gaussian σ.  Required when *use_gaussian* is True.
    use_gaussian : bool
        If True use :class:`~quantx_engine.pulse.GaussianWaveform`;
        otherwise use :class:`~quantx_engine.pulse.ConstantWaveform`.

    Returns
    -------
    PulseSchedule
        Schedule of total duration ``idle_time + pi_duration``.
    """
    if idle_time < 0:
        raise CrosstalkError("idle_time must be non-negative")
    if pi_duration <= 0:
        raise CrosstalkError("pi_duration must be positive")

    sched = PulseSchedule(name=f"echo_q{q0}_q{q1}")
    half = idle_time / 2.0

    if use_gaussian:
        if pi_sigma is None:
            pi_sigma = pi_duration / 6.0
        amp = pi_pulse_area(pi_sigma)
        wf: object = GaussianWaveform(pi_duration, amp, pi_sigma)
    else:
        amp = math.pi / pi_duration
        wf = ConstantWaveform(pi_duration, amp)

    # The π-pulses are applied after the first half-idle period
    t_pi = half
    sched.add(DriveChannel(q0), wf, t_start=t_pi)  # type: ignore[arg-type]
    sched.add(DriveChannel(q1), wf, t_start=t_pi)  # type: ignore[arg-type]

    # Pad with a zero-amplitude waveform to set the schedule total duration
    sched.add(
        DriveChannel(q0),
        ConstantWaveform(idle_time + pi_duration, 0.0),
        t_start=0.0,
    )
    return sched
