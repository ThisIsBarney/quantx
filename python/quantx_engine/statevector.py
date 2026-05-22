"""Pure-Python dense statevector simulator (M160).

A self-contained ~400-line NumPy statevector simulator that works with zero
native dependencies.  It is the fallback backend used by
:class:`~quantx_engine.engine.Engine` when the native C++ library is absent,
and can also be used directly for rapid prototyping and testing.

Supports up to 25 qubits (2^25 amplitudes ≈ 256 MB complex128).

Standard gate set
-----------------
Single-qubit: H, X, Y, Z, S, T, Sdg, Tdg, Rx, Ry, Rz, P (phase), I
Two-qubit:    CNOT (CX), CY, CZ, SWAP, iSWAP, CH
Three-qubit:  CCX (Toffoli), CSWAP (Fredkin)

Usage
-----
>>> from quantx_engine.statevector import StatevectorSimulator
>>> sim = StatevectorSimulator(2)
>>> sim.h(0).cnot(0, 1)
>>> sv = sim.statevector()          # np.ndarray[complex128], shape (4,)
>>> probs = sim.probabilities()     # np.ndarray[float64],   shape (4,)
>>> counts = sim.sample(1024)       # Counter[str]  e.g. {"00": 512, "11": 512}
>>> e = sim.expectation("ZZ")       # float
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

_MAX_QUBITS: int = 25
_DTYPE = np.complex128

# ── Single-qubit gate matrices ────────────────────────────────────────────────

_I  = np.eye(2, dtype=_DTYPE)
_X  = np.array([[0, 1], [1, 0]], dtype=_DTYPE)
_Y  = np.array([[0, -1j], [1j, 0]], dtype=_DTYPE)
_Z  = np.array([[1, 0], [0, -1]], dtype=_DTYPE)
_H  = np.array([[1, 1], [1, -1]], dtype=_DTYPE) / math.sqrt(2)
_S  = np.array([[1, 0], [0, 1j]], dtype=_DTYPE)
_T  = np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=_DTYPE)
_Sdg = np.array([[1, 0], [0, -1j]], dtype=_DTYPE)
_Tdg = np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=_DTYPE)


def _rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=_DTYPE)


def _ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=_DTYPE)


def _rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * theta / 2), 0],
         [0, np.exp(1j * theta / 2)]],
        dtype=_DTYPE,
    )


def _phase(lam: float) -> np.ndarray:
    return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=_DTYPE)


def _u3(theta: float, phi: float, lam: float) -> np.ndarray:
    """General single-qubit SU(2) gate (OpenQASM U3)."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array(
        [[c, -np.exp(1j * lam) * s],
         [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
        dtype=_DTYPE,
    )


# ── Core application helpers ──────────────────────────────────────────────────

def _apply_single(state: np.ndarray, n: int, gate: np.ndarray, qubit: int) -> np.ndarray:
    """Apply a 2×2 gate to *qubit* in an n-qubit statevector."""
    # Reshape to tensor with each qubit as an axis of size 2.
    sv = state.reshape([2] * n)
    # Contract: sv[..., qubit, ...] = gate @ sv[..., qubit, ...]
    sv = np.tensordot(gate, sv, axes=[[1], [qubit]])
    # tensordot moves contracted axis to position 0; move it back.
    sv = np.moveaxis(sv, 0, qubit)
    return sv.reshape(2 ** n)


def _apply_controlled(state: np.ndarray, n: int, gate: np.ndarray,
                      control: int, target: int) -> np.ndarray:
    """Apply a controlled-U gate (control → target)."""
    sv = state.reshape([2] * n)
    # Select the |1⟩ slice of the control qubit.
    idx_ctrl = [slice(None)] * n
    idx_ctrl[control] = 1
    slice_ctrl = tuple(idx_ctrl)
    # Apply gate only to the target axis within that slice.
    sub = sv[slice_ctrl]               # shape (2,)^(n-1), target axis still present
    sub = np.tensordot(gate, sub, axes=[[1], [target if target < control else target - 1]])
    sub = np.moveaxis(sub, 0, target if target < control else target - 1)
    sv[slice_ctrl] = sub
    return sv.reshape(2 ** n)


def _apply_swap(state: np.ndarray, n: int, q0: int, q1: int) -> np.ndarray:
    """Swap two qubits."""
    sv = state.reshape([2] * n)
    sv = np.swapaxes(sv, q0, q1)
    return sv.reshape(2 ** n)


def _apply_toffoli(state: np.ndarray, n: int, c0: int, c1: int, t: int) -> np.ndarray:
    """CCX (Toffoli) gate: |c0=1, c1=1⟩ → X|t⟩."""
    sv = state.reshape([2] * n)
    idx = [slice(None)] * n
    idx[c0] = 1
    idx[c1] = 1
    s = tuple(idx)
    sub = sv[s]
    # Apply X to the target (now shifted by number of controls before it)
    adjusted_t = t - (1 if c0 < t else 0) - (1 if c1 < t else 0)
    sub = np.tensordot(_X, sub, axes=[[1], [adjusted_t]])
    sub = np.moveaxis(sub, 0, adjusted_t)
    sv[s] = sub
    return sv.reshape(2 ** n)


def _apply_cswap(state: np.ndarray, n: int, ctrl: int, q0: int, q1: int) -> np.ndarray:
    """CSWAP (Fredkin) gate: when control=|1⟩, swap q0 and q1."""
    sv = state.reshape([2] * n)
    idx = [slice(None)] * n
    idx[ctrl] = 1
    s = tuple(idx)
    sub = sv[s]
    # Adjust target indices to exclude the control axis
    def _adj(q: int) -> int:
        return q - (1 if ctrl < q else 0)
    sub = np.swapaxes(sub, _adj(q0), _adj(q1))
    sv[s] = sub
    return sv.reshape(2 ** n)


# ── Pauli expectation value ───────────────────────────────────────────────────

_PAULI_MATRIX: dict[str, np.ndarray] = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def _pauli_string_matrix(pauli: str, n: int, qubits: Sequence[int] | None = None) -> np.ndarray:
    """Build the full 2^n × 2^n matrix for a Pauli string.

    *pauli* is a string like ``"ZI"`` or ``"XIZ"`` (length == n, left = qubit 0).
    If *qubits* is given, *pauli* characters are mapped to those qubit indices.
    """
    if len(pauli) != n and qubits is None:
        raise ValueError(
            f"Pauli string length {len(pauli)} != n_qubits {n}. "
            "Pass qubits= to specify which qubits the string acts on."
        )
    mats: list[np.ndarray] = [_I] * n
    if qubits is not None:
        for p, q in zip(pauli, qubits):
            mats[q] = _PAULI_MATRIX[p.upper()]
    else:
        for i, p in enumerate(pauli):
            mats[i] = _PAULI_MATRIX[p.upper()]
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result.astype(_DTYPE)


# ── StatevectorSimulator ──────────────────────────────────────────────────────

class StatevectorSimulator:
    """Dense statevector simulator backed by NumPy.

    All gate methods return ``self`` so they can be chained::

        sim = StatevectorSimulator(2)
        sv  = sim.h(0).cnot(0, 1).statevector()

    Parameters
    ----------
    n_qubits : int
        Number of qubits (1–25).
    initial_state : int | np.ndarray | None
        - ``int`` (default 0): computational basis state |initial_state⟩.
        - ``np.ndarray`` of shape ``(2**n_qubits,)``: custom initial amplitudes
          (will be normalised).
        - ``None``: same as 0 (|0...0⟩).
    """

    def __init__(
        self,
        n_qubits: int,
        initial_state: int | np.ndarray | None = 0,
    ) -> None:
        if not 1 <= n_qubits <= _MAX_QUBITS:
            raise ValueError(
                f"n_qubits must be in [1, {_MAX_QUBITS}], got {n_qubits}."
            )
        self._n = n_qubits
        self._dim = 1 << n_qubits
        self._state: np.ndarray = self._init_state(initial_state)

    # ── Initialisation helpers ────────────────────────────────────────────────

    def _init_state(self, initial_state: int | np.ndarray | None) -> np.ndarray:
        if initial_state is None or isinstance(initial_state, int):
            basis = 0 if initial_state is None else int(initial_state)
            if not 0 <= basis < self._dim:
                raise ValueError(
                    f"initial_state {basis} out of range [0, {self._dim})."
                )
            sv = np.zeros(self._dim, dtype=_DTYPE)
            sv[basis] = 1.0
            return sv
        arr = np.asarray(initial_state, dtype=_DTYPE).flatten()
        if arr.shape[0] != self._dim:
            raise ValueError(
                f"initial_state array length {arr.shape[0]} != 2**{self._n} = {self._dim}."
            )
        norm = np.linalg.norm(arr)
        if norm < 1e-15:
            raise ValueError("initial_state array has zero norm.")
        return arr / norm

    def _check_qubit(self, q: int, name: str = "qubit") -> None:
        if not 0 <= q < self._n:
            raise ValueError(f"{name} index {q} out of range [0, {self._n}).")

    def _apply_1q(self, gate: np.ndarray, qubit: int) -> "StatevectorSimulator":
        self._check_qubit(qubit)
        self._state = _apply_single(self._state, self._n, gate, qubit)
        return self

    def _apply_2q_ctrl(self, gate: np.ndarray, control: int, target: int) -> "StatevectorSimulator":
        self._check_qubit(control, "control")
        self._check_qubit(target, "target")
        if control == target:
            raise ValueError("control and target qubits must differ.")
        self._state = _apply_controlled(self._state, self._n, gate, control, target)
        return self

    # ── Single-qubit gates ────────────────────────────────────────────────────

    def i(self, qubit: int) -> "StatevectorSimulator":
        """Identity gate (no-op; useful for explicit circuit notation)."""
        return self._apply_1q(_I, qubit)

    def x(self, qubit: int) -> "StatevectorSimulator":
        """Pauli-X (NOT) gate."""
        return self._apply_1q(_X, qubit)

    def y(self, qubit: int) -> "StatevectorSimulator":
        """Pauli-Y gate."""
        return self._apply_1q(_Y, qubit)

    def z(self, qubit: int) -> "StatevectorSimulator":
        """Pauli-Z gate."""
        return self._apply_1q(_Z, qubit)

    def h(self, qubit: int) -> "StatevectorSimulator":
        """Hadamard gate."""
        return self._apply_1q(_H, qubit)

    def s(self, qubit: int) -> "StatevectorSimulator":
        """S gate (√Z, phase 90°)."""
        return self._apply_1q(_S, qubit)

    def t(self, qubit: int) -> "StatevectorSimulator":
        """T gate (√S, phase 45°)."""
        return self._apply_1q(_T, qubit)

    def sdg(self, qubit: int) -> "StatevectorSimulator":
        """S† gate (conjugate of S)."""
        return self._apply_1q(_Sdg, qubit)

    def tdg(self, qubit: int) -> "StatevectorSimulator":
        """T† gate (conjugate of T)."""
        return self._apply_1q(_Tdg, qubit)

    def rx(self, theta: float, qubit: int) -> "StatevectorSimulator":
        """Rotation around X axis by angle *theta* (radians)."""
        return self._apply_1q(_rx(theta), qubit)

    def ry(self, theta: float, qubit: int) -> "StatevectorSimulator":
        """Rotation around Y axis by angle *theta* (radians)."""
        return self._apply_1q(_ry(theta), qubit)

    def rz(self, theta: float, qubit: int) -> "StatevectorSimulator":
        """Rotation around Z axis by angle *theta* (radians)."""
        return self._apply_1q(_rz(theta), qubit)

    def p(self, lam: float, qubit: int) -> "StatevectorSimulator":
        """Phase gate P(λ): |1⟩ → e^{iλ}|1⟩."""
        return self._apply_1q(_phase(lam), qubit)

    def u3(self, theta: float, phi: float, lam: float, qubit: int) -> "StatevectorSimulator":
        """General single-qubit U3(θ, φ, λ) gate (OpenQASM U3)."""
        return self._apply_1q(_u3(theta, phi, lam), qubit)

    def custom(self, matrix: np.ndarray, qubit: int) -> "StatevectorSimulator":
        """Apply an arbitrary 2×2 unitary gate to *qubit*."""
        m = np.asarray(matrix, dtype=_DTYPE)
        if m.shape != (2, 2):
            raise ValueError(f"custom gate matrix must be 2×2, got {m.shape}.")
        return self._apply_1q(m, qubit)

    # ── Two-qubit gates ───────────────────────────────────────────────────────

    def cnot(self, control: int, target: int) -> "StatevectorSimulator":
        """CNOT (CX) gate."""
        return self._apply_2q_ctrl(_X, control, target)

    def cx(self, control: int, target: int) -> "StatevectorSimulator":
        """Alias for cnot()."""
        return self.cnot(control, target)

    def cy(self, control: int, target: int) -> "StatevectorSimulator":
        """Controlled-Y gate."""
        return self._apply_2q_ctrl(_Y, control, target)

    def cz(self, control: int, target: int) -> "StatevectorSimulator":
        """Controlled-Z gate."""
        return self._apply_2q_ctrl(_Z, control, target)

    def ch(self, control: int, target: int) -> "StatevectorSimulator":
        """Controlled-H gate."""
        return self._apply_2q_ctrl(_H, control, target)

    def swap(self, q0: int, q1: int) -> "StatevectorSimulator":
        """SWAP gate."""
        self._check_qubit(q0, "q0")
        self._check_qubit(q1, "q1")
        if q0 == q1:
            raise ValueError("SWAP qubits must differ.")
        self._state = _apply_swap(self._state, self._n, q0, q1)
        return self

    def iswap(self, q0: int, q1: int) -> "StatevectorSimulator":
        """iSWAP gate: SWAP with imaginary phase on swapped amplitudes."""
        _iSWAP = np.array(
            [[1, 0, 0, 0],
             [0, 0, 1j, 0],
             [0, 1j, 0, 0],
             [0, 0, 0, 1]],
            dtype=_DTYPE,
        )
        return self._apply_2q_unitary(_iSWAP, q0, q1)

    def _apply_2q_unitary(self, gate4: np.ndarray, q0: int, q1: int) -> "StatevectorSimulator":
        """Apply an arbitrary 4×4 unitary on qubits q0 and q1."""
        self._check_qubit(q0, "q0")
        self._check_qubit(q1, "q1")
        if q0 == q1:
            raise ValueError("2-qubit gate qubits must differ.")
        n = self._n
        sv = self._state.reshape([2] * n)
        # Collect the (q0, q1) subspace
        gate = gate4.reshape(2, 2, 2, 2)  # [out0, out1, in0, in1]
        sv = np.tensordot(gate, sv, axes=[[2, 3], [q0, q1]])
        # tensordot result has shape (2, 2, ...) with out axes at positions 0,1
        sv = np.moveaxis(sv, [0, 1], [q0, q1])
        self._state = sv.reshape(self._dim)
        return self

    # ── Three-qubit gates ─────────────────────────────────────────────────────

    def ccx(self, c0: int, c1: int, target: int) -> "StatevectorSimulator":
        """Toffoli (CCX) gate."""
        for q, name in [(c0, "c0"), (c1, "c1"), (target, "target")]:
            self._check_qubit(q, name)
        if len({c0, c1, target}) != 3:
            raise ValueError("CCX requires three distinct qubits.")
        self._state = _apply_toffoli(self._state, self._n, c0, c1, target)
        return self

    def toffoli(self, c0: int, c1: int, target: int) -> "StatevectorSimulator":
        """Alias for ccx()."""
        return self.ccx(c0, c1, target)

    def cswap(self, ctrl: int, q0: int, q1: int) -> "StatevectorSimulator":
        """Fredkin (CSWAP) gate: swap q0 and q1 when ctrl=|1⟩."""
        for q, name in [(ctrl, "ctrl"), (q0, "q0"), (q1, "q1")]:
            self._check_qubit(q, name)
        if len({ctrl, q0, q1}) != 3:
            raise ValueError("CSWAP requires three distinct qubits.")
        self._state = _apply_cswap(self._state, self._n, ctrl, q0, q1)
        return self

    def fredkin(self, ctrl: int, q0: int, q1: int) -> "StatevectorSimulator":
        """Alias for cswap()."""
        return self.cswap(ctrl, q0, q1)

    # ── Measurement & observables ─────────────────────────────────────────────

    def statevector(self) -> np.ndarray:
        """Return a copy of the current state as a complex128 NumPy array.

        Returns
        -------
        np.ndarray, shape (2**n_qubits,), dtype complex128
        """
        return self._state.copy()

    def probabilities(self) -> np.ndarray:
        """Return the measurement probability of each computational basis state.

        Returns
        -------
        np.ndarray, shape (2**n_qubits,), dtype float64
            ``probs[i]`` = |⟨i|ψ⟩|².
        """
        return (self._state.conj() * self._state).real.astype(np.float64)

    def sample(self, n_shots: int, seed: int | None = None) -> Counter:
        """Sample computational basis measurements.

        Parameters
        ----------
        n_shots : int
            Number of measurement shots.
        seed : int | None
            RNG seed for reproducibility.

        Returns
        -------
        Counter[str]
            Mapping from bitstring (big-endian, qubit 0 = leftmost) to count.
        """
        if n_shots < 1:
            raise ValueError(f"n_shots must be >= 1, got {n_shots}.")
        rng = np.random.default_rng(seed)
        probs = self.probabilities()
        # Normalise to guard against floating-point drift
        probs /= probs.sum()
        indices = rng.choice(self._dim, size=n_shots, p=probs)
        fmt = f"{{:0{self._n}b}}"
        return Counter(fmt.format(i) for i in indices)

    def expectation(
        self,
        pauli: str,
        qubits: Sequence[int] | None = None,
    ) -> float:
        """Compute ⟨ψ|P|ψ⟩ for a Pauli string P.

        Parameters
        ----------
        pauli : str
            Pauli string, e.g. ``"ZZ"`` or ``"XIZ"``.
            - If *qubits* is ``None``: length must equal n_qubits; character i
              applies to qubit i.
            - If *qubits* is given: ``pauli[k]`` applies to ``qubits[k]``.
        qubits : sequence of int | None
            Qubit indices for each character in *pauli*.

        Returns
        -------
        float
            Real part of the expectation value (imaginary part is zero for
            Hermitian observables).
        """
        mat = _pauli_string_matrix(pauli.upper(), self._n, qubits)
        ev = self._state.conj() @ mat @ self._state
        return float(ev.real)

    # ── Utility ───────────────────────────────────────────────────────────────

    def reset(self, initial_state: int | np.ndarray | None = 0) -> "StatevectorSimulator":
        """Reset the simulator to a new initial state.

        Returns ``self`` so it can be chained.
        """
        self._state = self._init_state(initial_state)
        return self

    def copy(self) -> "StatevectorSimulator":
        """Return a deep copy of this simulator."""
        new = StatevectorSimulator.__new__(StatevectorSimulator)
        new._n = self._n
        new._dim = self._dim
        new._state = self._state.copy()
        return new

    @property
    def n_qubits(self) -> int:
        """Number of qubits."""
        return self._n

    @property
    def dim(self) -> int:
        """Hilbert space dimension (2**n_qubits)."""
        return self._dim

    def __repr__(self) -> str:
        return (
            f"StatevectorSimulator(n_qubits={self._n}, "
            f"dim={self._dim})"
        )

    def __str__(self) -> str:
        return f"StatevectorSimulator({self._n} qubits)"

    # ── Dict-based circuit execution (Engine bridge) ──────────────────────────

    @classmethod
    def from_circuit_dict(cls, circuit: dict) -> "StatevectorSimulator":
        """Build and execute a circuit from the native QuantX circuit dict.

        This bridges the ``Engine.run_circuit()`` schema to the simulator,
        enabling pure-Python fallback when the native library is absent.

        Parameters
        ----------
        circuit : dict
            QuantX native circuit dict with keys:
            ``qubits``, ``operations``, ``shots``, ``seed``.

        Returns
        -------
        StatevectorSimulator
            Simulator at the final state after all operations.
        """
        n = int(circuit.get("qubits", 1))
        seed = int(circuit.get("seed", 42))
        sim = cls(n)
        for op in circuit.get("operations", []):
            sim._apply_circuit_op(op)
        return sim

    def run_circuit_dict(self, circuit: dict) -> dict:
        """Execute a QuantX circuit dict and return a result dict.

        Returns
        -------
        dict
            Contains ``"ok"``, ``"state"`` (list of [re, im] pairs),
            ``"counts"`` (dict mapping bitstring → count), ``"backend"``.
        """
        n = int(circuit.get("qubits", 1))
        shots = int(circuit.get("shots", 0))
        seed = int(circuit.get("seed", 42))
        sim = StatevectorSimulator(n)
        for op in circuit.get("operations", []):
            sim._apply_circuit_op(op)
        sv = sim.statevector()
        counts: dict[str, int] = {}
        if shots > 0:
            counts = dict(sim.sample(shots, seed=seed))
        return {
            "ok": True,
            "backend": "pure_python_statevector",
            "state": [[c.real, c.imag] for c in sv],
            "counts": counts,
            "n_qubits": n,
        }

    def _apply_circuit_op(self, op: dict) -> None:
        """Apply a single circuit operation dict to this simulator."""
        name = str(op.get("name", "")).upper()
        targets: list[int] = [int(q) for q in op.get("targets", [])]
        controls: list[int] = [int(q) for q in op.get("controls", [])]
        params: list[float] = [float(p) for p in op.get("params", [])]

        _SINGLE: dict[str, np.ndarray] = {
            "H": _H, "X": _X, "Y": _Y, "Z": _Z,
            "S": _S, "T": _T, "SDG": _Sdg, "TDG": _Tdg, "I": _I,
            "IDENTITY": _I,
        }

        if name in _SINGLE:
            for t in targets:
                self._apply_1q(_SINGLE[name], t)
        elif name in ("RX", "U1X"):
            self._apply_1q(_rx(params[0]), targets[0])
        elif name in ("RY", "U1Y"):
            self._apply_1q(_ry(params[0]), targets[0])
        elif name in ("RZ", "U1Z"):
            self._apply_1q(_rz(params[0]), targets[0])
        elif name in ("P", "PHASE", "U1"):
            self._apply_1q(_phase(params[0]), targets[0])
        elif name in ("U3", "U"):
            self._apply_1q(_u3(params[0], params[1], params[2]), targets[0])
        elif name in ("CNOT", "CX"):
            c = controls[0] if controls else targets[0]
            t = targets[-1]
            self._apply_2q_ctrl(_X, c, t)
        elif name == "CY":
            c = controls[0] if controls else targets[0]
            t = targets[-1]
            self._apply_2q_ctrl(_Y, c, t)
        elif name == "CZ":
            c = controls[0] if controls else targets[0]
            t = targets[-1]
            self._apply_2q_ctrl(_Z, c, t)
        elif name == "CH":
            c = controls[0] if controls else targets[0]
            t = targets[-1]
            self._apply_2q_ctrl(_H, c, t)
        elif name == "SWAP":
            self.swap(targets[0], targets[1])
        elif name == "ISWAP":
            self.iswap(targets[0], targets[1])
        elif name in ("CCX", "TOFFOLI"):
            c0 = controls[0] if len(controls) >= 2 else targets[0]
            c1 = controls[1] if len(controls) >= 2 else targets[1]
            t = targets[-1]
            self.ccx(c0, c1, t)
        elif name in ("CSWAP", "FREDKIN"):
            c = controls[0] if controls else targets[0]
            self.cswap(c, targets[0] if controls else targets[1],
                       targets[1] if controls else targets[2])
        elif name in ("MEASURE", "M", "MEAS"):
            pass  # Non-destructive; handled by shots sampling
        elif name == "BARRIER":
            pass  # No-op in ideal simulation
        else:
            raise ValueError(
                f"Unknown gate: {name!r}.  "
                "Supported: H, X, Y, Z, S, T, Sdg, Tdg, Rx, Ry, Rz, P, U3, "
                "CNOT/CX, CY, CZ, CH, SWAP, iSWAP, CCX/Toffoli, CSWAP/Fredkin."
            )
