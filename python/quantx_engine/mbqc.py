"""M125: Measurement-based quantum computation (MBQC).

Implements the one-way quantum computation model (1WQC / cluster-state QC):

1. Prepare a *resource state* — a graph state |G⟩ on the vertices of a graph G.
2. Perform adaptive *single-qubit measurements* in chosen bases.
3. Read off the result from the remaining (output) qubits, correcting for
   the randomness of each measurement outcome using *feed-forward* (Pauli
   corrections).

Classes
-------
:class:`GraphState`
    Build and manipulate graph states from an adjacency specification.
:class:`MBQCPattern`
    Specify and simulate an MBQC computation pattern (measurement sequence,
    correction rules, input/output qubits).

References
----------
* Raussendorf, R. & Briegel, H. J. (2001). A one-way quantum computer.
  *Phys. Rev. Lett.* 86, 5188.
* Danos, V. et al. (2007). The measurement calculus.
  *J. ACM* 54, 8.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "MBQCError",
    "GraphState",
    "MBQCPattern",
]

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class MBQCError(Exception):
    """Raised for invalid MBQC operations."""


# ---------------------------------------------------------------------------
# Single-qubit gates
# ---------------------------------------------------------------------------

_H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def _CZ(n: int, i: int, j: int) -> np.ndarray:
    """Controlled-Z gate on qubits i,j of an n-qubit system."""
    dim = 1 << n
    U = np.eye(dim, dtype=complex)
    # CZ flips the phase when both qubits are |1⟩
    for k in range(dim):
        bit_i = (k >> (n - 1 - i)) & 1
        bit_j = (k >> (n - 1 - j)) & 1
        if bit_i and bit_j:
            U[k, k] = -1.0
    return U


def _apply_single(gate: np.ndarray, psi: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Apply a 2×2 gate to *qubit* in an *n*-qubit state vector."""
    ops = [gate if i == qubit else _I2 for i in range(n)]
    U = ops[0]
    for op in ops[1:]:
        U = np.kron(U, op)
    return U @ psi


# ---------------------------------------------------------------------------
# GraphState
# ---------------------------------------------------------------------------

class GraphState:
    """A graph state |G⟩ on n vertices.

    A graph state is prepared by:
    1. Initialising all qubits in |+⟩ = H|0⟩.
    2. Applying a CZ gate for each edge (i, j) in the graph.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (vertices).
    edges : sequence of (int, int)
        Edge list of the graph.
    """

    def __init__(
        self,
        n_qubits: int,
        edges: Sequence[tuple[int, int]] = (),
    ) -> None:
        if n_qubits < 1:
            raise MBQCError("n_qubits must be at least 1.")
        if n_qubits > 15:
            raise MBQCError(
                "GraphState supports at most 15 qubits (2^15 = 32768 amplitudes)."
            )
        self.n_qubits = n_qubits
        self.edges: list[tuple[int, int]] = []
        for e in edges:
            i, j = int(e[0]), int(e[1])
            if not (0 <= i < n_qubits and 0 <= j < n_qubits):
                raise MBQCError(
                    f"Edge ({i}, {j}) out of range for {n_qubits} qubits."
                )
            if i == j:
                raise MBQCError(f"Self-loop ({i}, {i}) not allowed.")
            self.edges.append((i, j))

        self._state = self._prepare()

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def _prepare(self) -> np.ndarray:
        """Build the graph state from scratch."""
        n = self.n_qubits
        dim = 1 << n
        # Start with |00…0⟩
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1.0

        # Apply H to all qubits → |++…+⟩
        for q in range(n):
            psi = _apply_single(_H, psi, q, n)

        # Apply CZ for each edge
        for i, j in self.edges:
            U_CZ = _CZ(n, i, j)
            psi = U_CZ @ psi

        return psi

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state_vector(self) -> np.ndarray:
        """Statevector of the graph state (copy)."""
        return self._state.copy()

    @property
    def n_edges(self) -> int:
        """Number of edges in the graph."""
        return len(self.edges)

    @property
    def adjacency_matrix(self) -> np.ndarray:
        """Binary adjacency matrix Γ ∈ {0,1}^{n×n}."""
        n = self.n_qubits
        A = np.zeros((n, n), dtype=np.uint8)
        for i, j in self.edges:
            A[i, j] = A[j, i] = 1
        return A

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def reduced_density_matrix(self, qubits: Sequence[int]) -> np.ndarray:
        """Partial trace over the complement of *qubits*.

        Parameters
        ----------
        qubits : sequence of int
            Indices of the qubits to keep.

        Returns
        -------
        ndarray, shape (2^k, 2^k)
            Reduced density matrix for the subsystem *qubits*.
        """
        qubits = sorted(set(int(q) for q in qubits))
        for q in qubits:
            if not (0 <= q < self.n_qubits):
                raise MBQCError(f"Qubit index {q} out of range.")
        k = len(qubits)
        n = self.n_qubits
        psi = self._state.reshape([2] * n)
        # Trace out qubits NOT in the subsystem
        trace_qubits = [q for q in range(n) if q not in qubits]
        rho = np.outer(psi.ravel(), psi.ravel().conj()).reshape([2] * (2 * n))
        for q in sorted(trace_qubits, reverse=True):
            rho = np.trace(rho, axis1=q, axis2=q + n)
            n -= 1
        rho = rho.reshape(1 << k, 1 << k)
        return rho

    def entanglement_entropy(self, partition: Sequence[int]) -> float:
        """Von Neumann entanglement entropy S(ρ_A) for bipartition *partition*.

        Parameters
        ----------
        partition : sequence of int
            Indices of qubits in subsystem A.
        """
        rho = self.reduced_density_matrix(partition)
        eigvals = np.linalg.eigvalsh(rho)
        eigvals = np.maximum(eigvals, 0.0)  # clip numerical noise
        S = -sum(v * math.log2(v) if v > 1e-14 else 0.0 for v in eigvals)
        return float(S)

    def fidelity_with(self, other: "GraphState") -> float:
        """State fidelity |⟨G|G'⟩|² with another GraphState."""
        if self.n_qubits != other.n_qubits:
            raise MBQCError("Cannot compute fidelity between states of different sizes.")
        return float(abs(self._state.conj() @ other._state) ** 2)

    def __repr__(self) -> str:
        return f"GraphState(n_qubits={self.n_qubits}, n_edges={self.n_edges})"

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def linear_cluster(cls, n: int) -> "GraphState":
        """Build a linear (1D) cluster state on n qubits."""
        if n < 2:
            raise MBQCError("Linear cluster requires at least 2 qubits.")
        edges = [(i, i + 1) for i in range(n - 1)]
        return cls(n, edges)

    @classmethod
    def grid_cluster(cls, rows: int, cols: int) -> "GraphState":
        """Build a 2D rectangular cluster state on rows × cols qubits."""
        n = rows * cols
        if n < 2:
            raise MBQCError("Grid cluster requires at least 2 qubits.")
        if n > 15:
            raise MBQCError("Grid cluster exceeds 15-qubit limit.")
        edges = []
        for r in range(rows):
            for c in range(cols):
                v = r * cols + c
                if c + 1 < cols:
                    edges.append((v, v + 1))
                if r + 1 < rows:
                    edges.append((v, v + cols))
        return cls(n, edges)


# ---------------------------------------------------------------------------
# MBQCPattern
# ---------------------------------------------------------------------------

@dataclass
class MeasurementResult:
    """Result of measuring one qubit in the MBQC computation.

    Attributes
    ----------
    qubit : int
        Index of the measured qubit.
    angle : float
        Measurement angle θ (basis: (X cos θ + Y sin θ) / 2).
    outcome : int
        Measurement outcome (0 or 1).
    corrected_angle : float
        Effective measurement angle after feed-forward corrections.
    """
    qubit: int
    angle: float
    outcome: int
    corrected_angle: float


class MBQCPattern:
    """Specify and simulate an MBQC computation pattern on a graph state.

    A pattern consists of:
    * An ordered sequence of measurements ``(qubit, angle)``.
    * Feed-forward corrections: each measurement may trigger Pauli X or Z
      corrections on other qubits conditioned on the measurement outcome.
    * Input and output qubit designations.

    Parameters
    ----------
    graph_state : GraphState
        The resource graph state.
    input_qubits : sequence of int
        Qubits that carry the input (not measured).
    output_qubits : sequence of int
        Qubits that carry the output (not measured).
    """

    def __init__(
        self,
        graph_state: GraphState,
        input_qubits: Sequence[int] = (),
        output_qubits: Sequence[int] = (),
    ) -> None:
        self.graph_state = graph_state
        self.n = graph_state.n_qubits
        self.input_qubits = list(input_qubits)
        self.output_qubits = list(output_qubits)

        # Validate
        all_q = set(range(self.n))
        for q in self.input_qubits + self.output_qubits:
            if q not in all_q:
                raise MBQCError(f"Qubit {q} out of range for {self.n}-qubit state.")

        # Internal state
        self._state: np.ndarray = graph_state.state_vector
        self._x_corrections: dict[int, list[int]] = {}  # q: [controls]
        self._z_corrections: dict[int, list[int]] = {}
        self._measurements: list[MeasurementResult] = []
        self._measured: set[int] = set()

    # ------------------------------------------------------------------
    # Pattern specification
    # ------------------------------------------------------------------

    def add_x_correction(self, target: int, depends_on: list[int]) -> None:
        """Register: if XOR of outcomes for *depends_on* qubits = 1, apply X to *target*.

        Parameters
        ----------
        target : int
            Qubit to apply X correction to.
        depends_on : list[int]
            Qubit indices whose outcomes control the correction.
        """
        if not (0 <= target < self.n):
            raise MBQCError(f"target qubit {target} out of range.")
        for q in depends_on:
            if not (0 <= q < self.n):
                raise MBQCError(
                    f"depends_on qubit {q} out of range (must be in [0, {self.n - 1}])."
                )
        self._x_corrections[target] = list(depends_on)

    def add_z_correction(self, target: int, depends_on: list[int]) -> None:
        """Register: if XOR of outcomes for *depends_on* qubits = 1, apply Z to *target*.

        Parameters
        ----------
        target : int
            Qubit to apply Z correction to.
        depends_on : list[int]
            Qubit indices whose outcomes control the correction.
        """
        if not (0 <= target < self.n):
            raise MBQCError(f"target qubit {target} out of range.")
        for q in depends_on:
            if not (0 <= q < self.n):
                raise MBQCError(
                    f"depends_on qubit {q} out of range (must be in [0, {self.n - 1}])."
                )
        self._z_corrections[target] = list(depends_on)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def measure(
        self,
        qubit: int,
        angle: float,
        seed: Optional[int] = None,
        outcome: Optional[int] = None,
    ) -> MeasurementResult:
        """Measure *qubit* in the basis (|+_θ⟩, |−_θ⟩).

        The measurement basis vectors are:
            |+_θ⟩ = (|0⟩ + e^{iθ}|1⟩)/√2
            |−_θ⟩ = (|0⟩ - e^{iθ}|1⟩)/√2

        Parameters
        ----------
        qubit : int
            Index of the qubit to measure.
        angle : float
            Measurement angle θ.
        seed : int or None
            RNG seed for random outcome sampling.
        outcome : int or None
            If given, force the measurement outcome (0 or 1) — useful for
            deterministic testing.

        Returns
        -------
        MeasurementResult
        """
        if qubit in self._measured:
            raise MBQCError(f"Qubit {qubit} has already been measured.")
        if qubit in self.output_qubits:
            raise MBQCError(f"Qubit {qubit} is an output qubit and cannot be measured.")

        # Compute corrected angle from feed-forward.
        # MBQC rule: X-dependencies invert the sign of the angle ((-1)^s_x),
        # then Z-dependencies shift by s_z * π.  Applying an additive shift
        # instead of a sign flip is incorrect for non-Pauli angles.
        s_x = self._xor_outcomes(self._x_corrections[qubit]) if qubit in self._x_corrections else 0
        s_z = self._xor_outcomes(self._z_corrections[qubit]) if qubit in self._z_corrections else 0
        corrected_angle = ((-1) ** s_x) * angle + s_z * math.pi

        # Build projectors onto |+_θ⟩ and |−_θ⟩
        phase = cmath.exp(1j * corrected_angle)
        proj_0 = np.array([1.0, phase], dtype=complex) / math.sqrt(2)   # |+_θ⟩
        proj_1 = np.array([1.0, -phase], dtype=complex) / math.sqrt(2)  # |−_θ⟩

        # Compute born probabilities by partial projection
        prob_0 = self._born_prob(qubit, proj_0)
        prob_1 = self._born_prob(qubit, proj_1)

        # Sample outcome
        if outcome is not None:
            m = int(outcome) % 2
        else:
            rng = np.random.default_rng(seed)
            total = prob_0 + prob_1
            m = 0 if (total < 1e-12 or rng.random() < prob_0 / total) else 1

        # Post-select state
        projector = proj_0 if m == 0 else proj_1
        self._postselect(qubit, projector)
        self._measured.add(qubit)

        result = MeasurementResult(
            qubit=qubit,
            angle=angle,
            outcome=m,
            corrected_angle=corrected_angle,
        )
        self._measurements.append(result)
        return result

    def run(
        self,
        measurement_sequence: Sequence[tuple],
        seed: Optional[int] = None,
    ) -> list[MeasurementResult]:
        """Run a full measurement sequence.

        Parameters
        ----------
        measurement_sequence : sequence of (qubit, angle) or (qubit, angle, forced_outcome)
            Ordered list of measurements to perform.
        seed : int or None
            RNG seed.

        Returns
        -------
        list[MeasurementResult]
        """
        results = []
        rng = np.random.default_rng(seed)
        for step in measurement_sequence:
            if len(step) == 2:
                q, angle = int(step[0]), float(step[1])
                forced = None
            else:
                q, angle, forced = int(step[0]), float(step[1]), int(step[2])
            seed_step = int(rng.integers(0, 2**31))
            r = self.measure(q, angle, seed=seed_step, outcome=forced)
            results.append(r)
        return results

    def output_state(self) -> np.ndarray:
        """Return the (unnormalised) state of the output qubits after all measurements.

        Returns
        -------
        ndarray
            Statevector restricted to the output qubits (re-indexed 0..k-1).
        """
        # If output_qubits were declared, restrict to them; otherwise fall back
        # to all unmeasured qubits (to avoid silently including ancilla qubits).
        n = self.n
        if self.output_qubits:
            remaining = sorted(self.output_qubits)
        else:
            remaining = sorted(q for q in range(n) if q not in self._measured)
        if not remaining:
            raise MBQCError("No output qubits available.")
        k = len(remaining)
        psi = self._state.reshape([2] * n)
        # Contract away measured qubits (they're in a definite post-selected state already)
        # Permute remaining qubits to the front
        order = remaining + [q for q in range(n) if q not in remaining]
        psi = np.transpose(psi, order)
        psi = psi.reshape(1 << k, -1)
        # Sum over all states of the traced qubits
        psi = psi[:, 0] if psi.shape[1] >= 1 else psi.ravel()
        norm = np.linalg.norm(psi)
        if norm > 1e-14:
            psi = psi / norm
        return psi

    def apply_corrections(self, outcomes: dict[int, int]) -> None:
        """Apply Pauli X/Z corrections to output qubits based on *outcomes*.

        Parameters
        ----------
        outcomes : dict[int, int]
            Measurement outcomes {qubit: outcome}.
        """
        for target, controls in self._x_corrections.items():
            if target not in self._measured:
                s = sum(outcomes.get(c, 0) for c in controls) % 2
                if s:
                    self._state = _apply_single(_X, self._state, target, self.n)
        for target, controls in self._z_corrections.items():
            if target not in self._measured:
                s = sum(outcomes.get(c, 0) for c in controls) % 2
                if s:
                    self._state = _apply_single(_Z, self._state, target, self.n)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _xor_outcomes(self, controls: list[int]) -> int:
        """XOR of outcomes for the given control qubits."""
        measured_outcomes = {r.qubit: r.outcome for r in self._measurements}
        return sum(measured_outcomes.get(c, 0) for c in controls) % 2

    def _born_prob(self, qubit: int, proj_vec: np.ndarray) -> float:
        """Probability of projecting *qubit* onto *proj_vec*."""
        n = self.n
        psi = self._state.reshape([2] * n)
        # Contract qubit dimension with proj_vec
        psi_proj = np.tensordot(proj_vec.conj(), psi, axes=([0], [qubit]))
        return float(np.real(np.sum(psi_proj.conj() * psi_proj)))

    def _postselect(self, qubit: int, proj_vec: np.ndarray) -> None:
        """Post-select the state on *qubit* projecting onto *proj_vec*."""
        n = self.n
        psi = self._state.reshape([2] * n)
        psi_proj = np.tensordot(proj_vec, psi, axes=([0], [qubit]))
        # psi_proj has shape [2]*(n-1) with qubit dimension removed
        # Insert back as a size-1 slice (qubit is now projected)
        psi_out = np.expand_dims(psi_proj, axis=qubit)
        # The projected qubit's state is |proj_vec⟩ * contracted amplitude
        # Actually: postselection sets qubit to |0⟩ conceptually, but we keep
        # full dimension for compatibility; amplitude lives in the proj_vec direction.
        new_psi = np.zeros([2] * n, dtype=complex)
        # Reconstruct: project onto proj_vec subspace
        slc0 = [slice(None)] * n
        slc1 = [slice(None)] * n
        slc0[qubit] = 0
        slc1[qubit] = 1
        a0 = proj_vec[0]
        a1 = proj_vec[1]
        new_psi[tuple(slc0)] = psi_proj * a0
        new_psi[tuple(slc1)] = psi_proj * a1
        self._state = new_psi.ravel()

    @property
    def measurement_outcomes(self) -> dict[int, int]:
        """Dictionary of {qubit: outcome} for all measured qubits."""
        return {r.qubit: r.outcome for r in self._measurements}

    @property
    def n_measured(self) -> int:
        """Number of qubits measured so far."""
        return len(self._measured)

    def __repr__(self) -> str:
        return (
            f"MBQCPattern(n_qubits={self.n}, "
            f"measured={self.n_measured}/{self.n}, "
            f"outputs={len(self.output_qubits)})"
        )
