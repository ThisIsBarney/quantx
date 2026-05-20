"""M120: Fault-tolerant T gate via magic state injection.

Implements four components for fault-tolerant T-gate synthesis:

* :class:`MagicStateFactory` — simulates 7-to-1 and 15-to-1 magic state
  distillation protocols to produce high-fidelity |T⟩ states.
* :class:`FTTGate` — gate-teleportation T gate consuming one |T⟩ magic state.
* :class:`FTCCXGate` — fault-tolerant Toffoli via the Clifford+T decomposition
  (7 T gates, Selinger 2013).
* :class:`TCountOptimizer` — T-count/T-depth analyser and algebraic reducer.

References
----------
* Bravyi, S. & Kitaev, A. (2005). Universal quantum computation with ideal
  Clifford gates and noisy ancillas. *Phys. Rev. A* 71, 022316.
* Selinger, P. (2013). Quantum circuits of T-depth one.
  *Phys. Rev. A* 87, 042302.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

__all__ = [
    "MagicStateError",
    "MagicStateFactory",
    "FTTGate",
    "FTCCXGate",
    "TCountOptimizer",
]

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class MagicStateError(Exception):
    """Raised for invalid magic-state distillation or FT-gate operations."""


# ---------------------------------------------------------------------------
# Shared gate constants
# ---------------------------------------------------------------------------

_T_PHASE = cmath.exp(1j * math.pi / 4)  # e^{iπ/4}

_I2: np.ndarray = np.eye(2, dtype=complex)
_H: np.ndarray = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
_X: np.ndarray = np.array([[0, 1], [1, 0]], dtype=complex)
_Z: np.ndarray = np.array([[1, 0], [0, -1]], dtype=complex)
_S: np.ndarray = np.array([[1, 0], [0, 1j]], dtype=complex)
_T_GATE: np.ndarray = np.diag([1.0, _T_PHASE]).astype(complex)
_T_DAG: np.ndarray = _T_GATE.conj().T

# |T⟩ = T|+⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2
_MAGIC_STATE_T: np.ndarray = np.array([1.0, _T_PHASE], dtype=complex) / math.sqrt(2)


def _t_state_fidelity(state: np.ndarray) -> float:
    """Return fidelity of *state* with the ideal |T⟩ magic state."""
    return float(abs(_MAGIC_STATE_T.conj() @ state) ** 2)


# ---------------------------------------------------------------------------
# MagicStateFactory
# ---------------------------------------------------------------------------

class MagicStateFactory:
    """Simulates magic state distillation for fault-tolerant T gates.

    Two protocols are supported:

    * **7-to-1** — consumes 7 noisy |T⟩ copies; uses the [[7,1,3]] Steane
      code to detect and discard corrupted states.
      Output error rate ≈ 35 ε³.

    * **15-to-1** — consumes 15 noisy copies; uses the [[15,1,3]] Reed-Muller
      code.  Output error rate ≈ 35 ε³ (with a smaller coefficient in practice).

    Parameters
    ----------
    protocol : {"7_to_1", "15_to_1"}
        Distillation protocol.
    input_error_rate : float
        Per-copy error rate of the noisy input |T⟩ states (must be in (0, 1)).
    """

    _PROTOCOLS: dict[str, dict] = {
        "7_to_1": {
            "overhead": 7,
            "poly_degree": 3,
            "coeff": 35.0,
            "correction_capability": 1,
        },
        "15_to_1": {
            "overhead": 15,
            "poly_degree": 3,
            "coeff": 35.0,
            "correction_capability": 2,
        },
    }

    def __init__(
        self,
        protocol: Literal["7_to_1", "15_to_1"] = "7_to_1",
        input_error_rate: float = 1e-2,
    ) -> None:
        if protocol not in self._PROTOCOLS:
            raise MagicStateError(
                f"Unknown protocol {protocol!r}.  Choose from "
                f"{list(self._PROTOCOLS)}."
            )
        if not (0.0 < input_error_rate < 1.0):
            raise MagicStateError("input_error_rate must be in (0, 1).")
        self.protocol = protocol
        self.input_error_rate = float(input_error_rate)
        self._spec = self._PROTOCOLS[protocol]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def overhead(self) -> int:
        """Number of noisy input copies consumed per distilled output."""
        return self._spec["overhead"]

    @property
    def threshold_error_rate(self) -> float:
        """Maximum input error rate below which distillation improves fidelity.

        Above this threshold the output is *worse* than the input (ε_out > ε_in).
        """
        c: float = self._spec["coeff"]
        d: int = self._spec["poly_degree"]
        # ε_in = ε_out  ↔  c * ε^d = ε  ↔  ε^{d-1} = 1/c
        return float((1.0 / c) ** (1.0 / (d - 1)))

    @property
    def output_error_rate(self) -> float:
        """Theoretical output error rate after one distillation round.

        Computed as ``coeff * ε_in ** poly_degree``.
        """
        return self._spec["coeff"] * (self.input_error_rate ** self._spec["poly_degree"])

    @property
    def distillation_factor(self) -> float:
        """Ratio ``input_error_rate / output_error_rate`` (improvement factor)."""
        out = self.output_error_rate
        return float("inf") if out == 0.0 else self.input_error_rate / out

    # ------------------------------------------------------------------
    # Simulation helpers
    # ------------------------------------------------------------------

    def _sample_noisy_magic_state(self, rng: np.random.Generator) -> np.ndarray:
        """Return one noisy |T⟩ copy under a depolarising error model."""
        if rng.random() < self.input_error_rate:
            err = int(rng.integers(0, 3))
            error_map = [
                np.array([_T_PHASE, 1.0], dtype=complex) / math.sqrt(2),  # X
                np.array([1.0, -_T_PHASE], dtype=complex) / math.sqrt(2),  # Z
                np.array([-_T_PHASE, 1.0], dtype=complex) / math.sqrt(2),  # Y
            ]
            return error_map[err]
        return _MAGIC_STATE_T.copy()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def distill_one(self, seed: Optional[int] = None) -> float:
        """Simulate one distillation run and return the output state fidelity.

        Returns
        -------
        float
            Fidelity of the distilled state with the ideal |T⟩.
        """
        rng = np.random.default_rng(seed)
        n = self.overhead
        noisy_states = [self._sample_noisy_magic_state(rng) for _ in range(n)]
        fidelities = [_t_state_fidelity(s) for s in noisy_states]

        # Count copies that appear corrupted
        n_bad = sum(1 for f in fidelities if f < 0.9)
        cap: int = self._spec["correction_capability"]

        if n_bad > cap:
            # Post-selection would discard this run; return theoretical output
            return 1.0 - self.output_error_rate

        # Average the Bloch vectors of the "good" copies
        good_states = [s for s, f in zip(noisy_states, fidelities) if f >= 0.9] or noisy_states
        avg = np.mean(good_states, axis=0)
        norm = np.linalg.norm(avg)
        if norm < 1e-12:
            return 0.0
        avg /= norm
        return min(1.0, float(_t_state_fidelity(avg)))

    def distill(self, n_states: int = 1, seed: Optional[int] = None) -> list[float]:
        """Simulate ``n_states`` independent distillation rounds.

        Returns
        -------
        list[float]
            Fidelity of each distilled output with the ideal |T⟩.
        """
        if n_states < 1:
            raise MagicStateError("n_states must be at least 1.")
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 2**31, size=n_states)
        return [self.distill_one(int(s)) for s in seeds]

    def __repr__(self) -> str:
        return (
            f"MagicStateFactory(protocol={self.protocol!r}, "
            f"input_error_rate={self.input_error_rate:.2e}, "
            f"output_error_rate={self.output_error_rate:.2e})"
        )


# ---------------------------------------------------------------------------
# FTTGate — fault-tolerant T gate via magic state injection
# ---------------------------------------------------------------------------

class FTTGate:
    """Fault-tolerant T gate via magic state injection (gate teleportation).

    Protocol
    --------
    1. Prepare ancilla in |T⟩ = (|0⟩ + e^{iπ/4}|1⟩)/√2.
    2. Apply CNOT (data qubit = control, ancilla = target).
    3. Measure ancilla in the Z basis.
    4. If outcome is |1⟩, apply S correction to the data qubit.

    The resulting operation applies T to the data qubit (up to global phase).

    Parameters
    ----------
    magic_state_fidelity : float
        Fidelity of the injected |T⟩ magic state with the ideal state.
        Controls the effective gate error rate.
    """

    def __init__(self, magic_state_fidelity: float = 1.0 - 1e-6) -> None:
        if not (0.0 < magic_state_fidelity <= 1.0):
            raise MagicStateError("magic_state_fidelity must be in (0, 1].")
        self.magic_state_fidelity = float(magic_state_fidelity)

    @property
    def error_rate(self) -> float:
        """Estimated gate error rate: ``1 - magic_state_fidelity``."""
        return 1.0 - self.magic_state_fidelity

    @property
    def circuit_depth(self) -> int:
        """Depth of the FT T-gate circuit (CNOT → measure → S-correction)."""
        return 3

    def apply(
        self,
        state_vector: np.ndarray,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, float]:
        """Apply the fault-tolerant T gate to a single-qubit state.

        Parameters
        ----------
        state_vector : array-like, shape (2,)
            Input qubit state |ψ⟩.
        seed : int or None
            RNG seed for the simulated ancilla measurement.

        Returns
        -------
        output_state : ndarray, shape (2,)
            Post-gate qubit state (normalised).
        success_prob : float
            Probability of the sampled ancilla measurement outcome.
        """
        sv = np.asarray(state_vector, dtype=complex).ravel()
        if sv.shape != (2,):
            raise MagicStateError(
                "state_vector must have shape (2,) for a single-qubit gate."
            )
        sv = sv / np.linalg.norm(sv)
        rng = np.random.default_rng(seed)

        # Prepare (possibly noisy) magic state ancilla
        anc = _MAGIC_STATE_T.copy()
        if rng.random() > self.magic_state_fidelity:
            err = int(rng.integers(0, 3))
            anc = [_X @ anc, _Z @ anc, (_X @ _Z) @ anc][err]

        # 2-qubit state: data (qubit 0) ⊗ ancilla (qubit 1)
        state_2q = np.kron(sv, anc)  # shape (4,)

        # CNOT: control = qubit 0, target = qubit 1
        _CNOT = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]],
            dtype=complex,
        )
        state_2q = _CNOT @ state_2q

        # Measure ancilla (qubit 1) in Z basis
        # Indices with ancilla=|1⟩ are 1 and 3
        prob_1 = float(abs(state_2q[1]) ** 2 + abs(state_2q[3]) ** 2)
        prob_0 = 1.0 - prob_1

        if prob_1 > 0 and rng.random() < prob_1:
            # Outcome |1⟩: keep amplitude components at indices 1, 3
            data_out = np.array([state_2q[1], state_2q[3]], dtype=complex)
            data_out /= math.sqrt(prob_1)
            data_out = _S @ data_out  # S correction
            success_prob = prob_1
        else:
            # Outcome |0⟩: keep amplitude components at indices 0, 2
            data_out = np.array([state_2q[0], state_2q[2]], dtype=complex)
            n = float(np.linalg.norm(data_out))
            if n > 1e-12:
                data_out /= n
            success_prob = prob_0

        return data_out, success_prob

    def __repr__(self) -> str:
        return f"FTTGate(magic_state_fidelity={self.magic_state_fidelity:.6f})"


# ---------------------------------------------------------------------------
# FTCCXGate — fault-tolerant Toffoli gate
# ---------------------------------------------------------------------------

# Selinger 2013 T-count-7 decomposition of CCX (qubits: 0=ctrl0, 1=ctrl1, 2=tgt)
# Each entry is (gate_name, *qubit_indices)
_CCX_SEQUENCE: list[tuple] = [
    ("H",    2),
    ("CNOT", 1, 2), ("TDG", 2),
    ("CNOT", 0, 2), ("T",   2),
    ("CNOT", 1, 2), ("TDG", 2),
    ("CNOT", 0, 2), ("T",   2),
    ("T",    1),
    ("H",    2),
    ("CNOT", 0, 1), ("T",   0), ("TDG", 1),
    ("CNOT", 0, 1),
]

_CCX_T_COUNT = 7


def _apply_1q(gate: np.ndarray, sv: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Apply a 2×2 *gate* to *qubit* in an *n*-qubit state vector."""
    ops = [gate if i == qubit else _I2 for i in range(n)]
    U = ops[0]
    for op in ops[1:]:
        U = np.kron(U, op)
    return U @ sv


def _apply_cnot(sv: np.ndarray, ctrl: int, tgt: int, n: int) -> np.ndarray:
    """Apply CNOT to an *n*-qubit state vector."""
    dim = 1 << n
    out = sv.copy()
    tgt_mask = 1 << (n - 1 - tgt)
    ctrl_mask = 1 << (n - 1 - ctrl)
    for i in range(dim):
        if i & ctrl_mask:
            j = i ^ tgt_mask
            if i < j:
                out[i], out[j] = sv[j], sv[i]
    return out


class FTCCXGate:
    """Fault-tolerant Toffoli gate using the Clifford+T decomposition.

    Uses the standard 7-T-gate decomposition from Selinger (2013).

    Parameters
    ----------
    t_gate_error_rate : float
        Per-T-gate error rate (independent depolarising model).
    """

    def __init__(self, t_gate_error_rate: float = 1e-4) -> None:
        if not (0.0 <= t_gate_error_rate < 1.0):
            raise MagicStateError("t_gate_error_rate must be in [0, 1).")
        self.t_gate_error_rate = float(t_gate_error_rate)

    @property
    def t_count(self) -> int:
        """T-count of the decomposition (7)."""
        return _CCX_T_COUNT

    @property
    def circuit_depth(self) -> int:
        """Total gate count of the Clifford+T sequence."""
        return len(_CCX_SEQUENCE)

    @property
    def error_rate(self) -> float:
        """Estimated gate error rate (union bound: T_count × ε_T)."""
        return min(1.0, self.t_count * self.t_gate_error_rate)

    def apply(
        self,
        state_vector: np.ndarray,
        seed: Optional[int] = None,
    ) -> tuple[np.ndarray, float]:
        """Apply the fault-tolerant CCX gate to a 3-qubit state.

        Parameters
        ----------
        state_vector : array-like, shape (8,)
            Input 3-qubit state.
        seed : int or None
            RNG seed for error simulation.

        Returns
        -------
        output_state : ndarray, shape (8,)
            Post-gate state (normalised).
        error_prob : float
            Estimated probability that at least one T-gate error occurred.
        """
        sv = np.asarray(state_vector, dtype=complex).ravel()
        if sv.shape != (8,):
            raise MagicStateError(
                "state_vector must have shape (8,) for a 3-qubit (CCX) gate."
            )
        sv = sv / np.linalg.norm(sv)
        rng = np.random.default_rng(seed)

        n = 3
        current = sv.copy()
        n_errors = 0

        for instr in _CCX_SEQUENCE:
            name = instr[0]
            if name == "H":
                current = _apply_1q(_H, current, instr[1], n)
            elif name == "CNOT":
                current = _apply_cnot(current, instr[1], instr[2], n)
            elif name == "T":
                q = instr[1]
                if rng.random() < self.t_gate_error_rate:
                    n_errors += 1
                    current = _apply_1q(_X, current, q, n)
                else:
                    current = _apply_1q(_T_GATE, current, q, n)
            elif name == "TDG":
                q = instr[1]
                if rng.random() < self.t_gate_error_rate:
                    n_errors += 1
                    current = _apply_1q(_X, current, q, n)
                else:
                    current = _apply_1q(_T_DAG, current, q, n)

        error_prob = float(min(1.0, n_errors * self.t_gate_error_rate))
        return current, error_prob

    def __repr__(self) -> str:
        return f"FTCCXGate(t_gate_error_rate={self.t_gate_error_rate:.2e})"


# ---------------------------------------------------------------------------
# TCountOptimizer
# ---------------------------------------------------------------------------

# T cost of each recognised gate name (upper-case)
_T_COST: dict[str, int] = {
    "I": 0, "H": 0, "X": 0, "Y": 0, "Z": 0,
    "CX": 0, "CNOT": 0, "CZ": 0, "SWAP": 0,
    "S": 2, "S†": 2, "SDG": 2,
    "T": 1, "T†": 1, "TDG": 1,
    "CCX": 7, "TOFFOLI": 7,
    "CSWAP": 7,
}

# Pairs of consecutive same-qubit gates that simplify
_PAIR_REDUCTIONS: dict[tuple[str, str], str] = {
    ("T",   "T"):   "S",
    ("T†",  "T†"):  "S†",
    ("TDG", "TDG"): "SDG",
    ("T",   "T†"):  "",    # → identity
    ("T†",  "T"):   "",
    ("T",   "TDG"): "",
    ("TDG", "T"):   "",
    ("S",   "S"):   "Z",
    ("S†",  "S†"):  "Z",
    ("SDG", "SDG"): "Z",
    ("S",   "S†"):  "",
    ("S†",  "S"):   "",
    ("S",   "SDG"): "",
    ("SDG", "S"):   "",
    ("Z",   "Z"):   "",    # Z² = I
    ("X",   "X"):   "",
    ("Y",   "Y"):   "",
    ("H",   "H"):   "",
}


@dataclass
class GateOp:
    """A single gate operation in a Clifford+T circuit."""

    gate: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, d: dict) -> "GateOp":
        gate = str(d.get("gate", "I")).upper()
        # Normalise gate aliases
        gate = {"T†": "T†", "TDG": "TDG", "S†": "S†", "SDG": "SDG"}.get(gate, gate)
        qubits: tuple[int, ...] = ()
        # Single-qubit index
        if "qubit" in d:
            qubits = (int(d["qubit"]),)
        # Two-qubit index
        if "control" in d:
            qubits = qubits + (int(d["control"]),)
        if "target" in d:
            qubits = qubits + (int(d["target"]),)
        # Multi-control
        if "controls" in d:
            qubits = tuple(int(x) for x in d["controls"]) + qubits
        if "qubits" in d:
            qubits = qubits + tuple(int(x) for x in d["qubits"])
        return cls(gate=gate, qubits=qubits)

    def to_dict(self) -> dict:
        d: dict = {"gate": self.gate}
        if len(self.qubits) == 1:
            d["qubit"] = self.qubits[0]
        elif len(self.qubits) == 2:
            d["control"] = self.qubits[0]
            d["target"] = self.qubits[1]
        elif len(self.qubits) >= 3:
            d["controls"] = list(self.qubits[:-1])
            d["target"] = self.qubits[-1]
        return d


class TCountOptimizer:
    """Analyses and minimises the T-count of a Clifford+T quantum circuit.

    The circuit is specified as a list of gate-operation dicts.  Each dict
    must contain a ``"gate"`` key; qubit indices are passed via ``"qubit"``,
    ``"control"``/``"target"``, or ``"controls"``/``"target"`` keys.

    Example::

        circuit = [
            {"gate": "H",    "qubit": 0},
            {"gate": "T",    "qubit": 0},
            {"gate": "CNOT", "control": 0, "target": 1},
            {"gate": "T†",   "qubit": 1},
            {"gate": "CCX",  "controls": [0, 1], "target": 2},
        ]

    Parameters
    ----------
    circuit : list[dict]
        Circuit as a list of gate-operation dicts.
    """

    def __init__(self, circuit: list[dict]) -> None:
        if not isinstance(circuit, list):
            raise MagicStateError("circuit must be a list of gate dicts.")
        self._raw = list(circuit)
        self._ops: list[GateOp] = [GateOp.from_dict(g) for g in circuit]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def t_count(self) -> int:
        """Total T-gate count (including gates that decompose into T gates)."""
        return sum(_T_COST.get(op.gate, 0) for op in self._ops)

    def t_depth(self) -> int:
        """T-depth: length of the longest T-gate path through the circuit.

        Computed by a single forward pass tracking the current T-layer per qubit.
        """
        if not self._ops:
            return 0
        qubit_layer: dict[int, int] = {}

        def _layer(q: int) -> int:
            return qubit_layer.get(q, 0)

        max_depth = 0
        for op in self._ops:
            t_cost = _T_COST.get(op.gate, 0)
            if not op.qubits:
                continue
            if t_cost > 0:
                start = max(_layer(q) for q in op.qubits)
                end = start + t_cost
                for q in op.qubits:
                    qubit_layer[q] = end
                max_depth = max(max_depth, end)
            elif len(op.qubits) > 1:
                # Multi-qubit Clifford gate: synchronise T-layers
                sync = max(_layer(q) for q in op.qubits)
                for q in op.qubits:
                    qubit_layer[q] = sync
        return max_depth

    def clifford_gates(self) -> list[GateOp]:
        """Return all non-T-gate (Clifford) operations."""
        return [op for op in self._ops if _T_COST.get(op.gate, 0) == 0]

    def t_gates(self) -> list[GateOp]:
        """Return all T and T†-gate operations."""
        return [op for op in self._ops if _T_COST.get(op.gate, 0) > 0]

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize(self) -> "TCountOptimizer":
        """Return a new :class:`TCountOptimizer` with a reduced T-count.

        Applies algebraic cancellations repeatedly until no further reduction
        is possible:

        * T·T → S
        * T†·T† → S†
        * T·T† → I  (and T†·T → I)
        * S·S → Z,  S†·S† → Z,  S·S† → I
        * Z·Z → I,  X·X → I,  Y·Y → I,  H·H → I

        Only consecutive operations on the *same single qubit* are reduced.

        Returns
        -------
        TCountOptimizer
            New optimiser wrapping the reduced circuit.
        """
        ops = list(self._ops)
        changed = True
        while changed:
            changed = False
            new_ops: list[GateOp] = []
            i = 0
            while i < len(ops):
                if i + 1 < len(ops):
                    a, b = ops[i], ops[i + 1]
                    if a.qubits == b.qubits and len(a.qubits) == 1:
                        result = _PAIR_REDUCTIONS.get((a.gate, b.gate))
                        if result is not None:
                            if result:
                                new_ops.append(GateOp(gate=result, qubits=a.qubits))
                            # else: both cancel → skip
                            i += 2
                            changed = True
                            continue
                new_ops.append(ops[i])
                i += 1
            ops = new_ops

        return TCountOptimizer([op.to_dict() for op in ops])

    # ------------------------------------------------------------------
    # Decompositions
    # ------------------------------------------------------------------

    _DECOMPOSITIONS: dict[str, list[dict]] = {
        "T": [{"gate": "T"}],
        "T†": [{"gate": "T†"}],
        "TDG": [{"gate": "T†"}],
        "S": [{"gate": "T"}, {"gate": "T"}],
        "S†": [{"gate": "T†"}, {"gate": "T†"}],
        "SDG": [{"gate": "T†"}, {"gate": "T†"}],
        "Z": [{"gate": "T"}, {"gate": "T"}, {"gate": "T"}, {"gate": "T"}],
        "CCX": [
            {"gate": "H",    "qubit": 2},
            {"gate": "CNOT", "control": 1, "target": 2},
            {"gate": "T†",   "qubit": 2},
            {"gate": "CNOT", "control": 0, "target": 2},
            {"gate": "T",    "qubit": 2},
            {"gate": "CNOT", "control": 1, "target": 2},
            {"gate": "T†",   "qubit": 2},
            {"gate": "CNOT", "control": 0, "target": 2},
            {"gate": "T",    "qubit": 2},
            {"gate": "T",    "qubit": 1},
            {"gate": "H",    "qubit": 2},
            {"gate": "CNOT", "control": 0, "target": 1},
            {"gate": "T",    "qubit": 0},
            {"gate": "T†",   "qubit": 1},
            {"gate": "CNOT", "control": 0, "target": 1},
        ],
    }

    def clifford_t_decomposition(self, gate_name: str) -> list[dict]:
        """Return a Clifford+T decomposition for a named gate.

        Parameters
        ----------
        gate_name : str
            Gate name (e.g. ``"CCX"``, ``"S"``, ``"T"``).  Case-insensitive.

        Returns
        -------
        list[dict]
            Sequence of primitive gate dicts in the same format as the circuit.

        Raises
        ------
        MagicStateError
            If no decomposition is available for *gate_name*.
        """
        key = gate_name.upper()
        if key not in self._DECOMPOSITIONS:
            raise MagicStateError(
                f"No Clifford+T decomposition available for {gate_name!r}.  "
                f"Available gates: {list(self._DECOMPOSITIONS)}."
            )
        return [dict(g) for g in self._DECOMPOSITIONS[key]]

    def __repr__(self) -> str:
        return (
            f"TCountOptimizer(n_gates={len(self._ops)}, "
            f"t_count={self.t_count()}, t_depth={self.t_depth()})"
        )
