"""Dynamic circuits with mid-circuit measurement and classical feed-forward — M86.

Provides:

- :class:`DynamicCircuit`: circuit IR supporting mid-circuit measurement,
  classical-register feed-forward, qubit reset, and bounded repeat-until-success
  loops.
- :class:`DynamicSimulator`: statevector trajectory simulator — runs ``shots``
  independent trajectories, collapses the wavefunction on each measurement, and
  applies conditional gates according to the classical register state.
- :class:`DynamicResult`: result container with counts, quasi-probabilities, and
  classical-register snapshots.

Qubit/classical-bit ordering matches the :mod:`~quantx_engine.primitives`
module: qubit 0 = LSB of the statevector index; bitstrings are printed
MSB-first (qubit n-1 … qubit 0).

Quick reference::

    from quantx_engine.dynamic import DynamicCircuit, DynamicSimulator

    # Teleportation / feed-forward demo:
    # Prepare |+⟩ on qubit 0, entangle 0-1, measure 0 → conditionally correct 1.
    circ = (
        DynamicCircuit(n_qubits=2, n_cbits=1)
        .h(0)
        .cx(0, 1)
        .measure(0, 0)
        .if_bit(cbit=0, value=1, name="x", qubits=[1])
    )
    result = DynamicSimulator(shots=200, seed=0).run(circ)
    # qubit 1 should always end up in |+⟩ ≡ (|0⟩+|1⟩)/√2, so counts ≈ 50/50
    print(result.counts)

References
----------
.. [1] Corcoles, A. D. et al. (2021). Exploiting dynamic quantum circuits in
       a quantum algorithm with superconducting qubits.
       *Phys. Rev. Lett.* 127, 100501.
       https://doi.org/10.1103/PhysRevLett.127.100501

.. [2] Pino, J. M. et al. (2021). Demonstration of the trapped-ion quantum CCD
       computer architecture. *Nature* 592, 209.
       https://doi.org/10.1038/s41586-021-03318-4

.. [3] Aaronson, S. (2005). Quantum computing, postselection, and probabilistic
       polynomial-time. *Proc. R. Soc. A* 461, 3473.
       https://doi.org/10.1098/rspa.2005.1546
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

# Reuse gate machinery from primitives
from .primitives import _apply_gate_sv, _get_gate_matrix, _require_numpy as _req

__all__ = [
    "DynamicCircuitError",
    "DynamicCircuit",
    "DynamicResult",
    "DynamicSimulator",
]


# ===========================================================================
# Exception
# ===========================================================================

class DynamicCircuitError(ValueError):
    """Raised for invalid dynamic circuit inputs or simulation errors."""


def _require_numpy() -> None:
    if not _HAS_NUMPY:  # pragma: no cover
        raise DynamicCircuitError("NumPy is required for dynamic circuits.")


# ===========================================================================
# DynamicCircuit — circuit IR
# ===========================================================================

class DynamicCircuit:
    """Circuit supporting mid-circuit measurements and classical feed-forward.

    Operations are appended with fluent builder methods; the circuit is
    immutable once passed to :class:`DynamicSimulator`.

    Parameters
    ----------
    n_qubits:
        Number of quantum bits.
    n_cbits:
        Number of classical bits (defaults to ``n_qubits``).
    """

    def __init__(self, n_qubits: int, n_cbits: int | None = None) -> None:
        if n_qubits < 1:
            raise DynamicCircuitError(f"n_qubits must be >= 1, got {n_qubits}.")
        self.n_qubits: int = n_qubits
        self.n_cbits: int = n_cbits if n_cbits is not None else n_qubits
        self.operations: list[dict] = []

    # ── gate shortcuts ──────────────────────────────────────────────────────

    def add_gate(
        self,
        name: str,
        qubits: list[int],
        params: list | None = None,
    ) -> "DynamicCircuit":
        """Append a unitary gate operation."""
        self._check_qubits(qubits)
        self.operations.append({
            "type": "gate",
            "name": name,
            "qubits": list(qubits),
            "params": list(params or []),
        })
        return self

    def h(self, qubit: int) -> "DynamicCircuit":
        return self.add_gate("h", [qubit])

    def x(self, qubit: int) -> "DynamicCircuit":
        return self.add_gate("x", [qubit])

    def y(self, qubit: int) -> "DynamicCircuit":
        return self.add_gate("y", [qubit])

    def z(self, qubit: int) -> "DynamicCircuit":
        return self.add_gate("z", [qubit])

    def s(self, qubit: int) -> "DynamicCircuit":
        return self.add_gate("s", [qubit])

    def t(self, qubit: int) -> "DynamicCircuit":
        return self.add_gate("t", [qubit])

    def cx(self, control: int, target: int) -> "DynamicCircuit":
        return self.add_gate("cx", [control, target])

    def cz(self, control: int, target: int) -> "DynamicCircuit":
        return self.add_gate("cz", [control, target])

    def rx(self, theta: float, qubit: int) -> "DynamicCircuit":
        return self.add_gate("rx", [qubit], [theta])

    def ry(self, theta: float, qubit: int) -> "DynamicCircuit":
        return self.add_gate("ry", [qubit], [theta])

    def rz(self, theta: float, qubit: int) -> "DynamicCircuit":
        return self.add_gate("rz", [qubit], [theta])

    # ── measurement ─────────────────────────────────────────────────────────

    def measure(self, qubit: int, cbit: int) -> "DynamicCircuit":
        """Mid-circuit measurement: collapses qubit, stores outcome in cbit."""
        self._check_qubits([qubit])
        self._check_cbits([cbit])
        self.operations.append({
            "type": "measure",
            "qubit": qubit,
            "cbit": cbit,
        })
        return self

    def measure_all(self) -> "DynamicCircuit":
        """Append measurements on all qubits, storing in cbits 0…n_qubits-1."""
        for i in range(min(self.n_qubits, self.n_cbits)):
            self.measure(i, i)
        return self

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, qubit: int) -> "DynamicCircuit":
        """Reset qubit to |0⟩ (measure then conditionally flip)."""
        self._check_qubits([qubit])
        self.operations.append({"type": "reset", "qubit": qubit})
        return self

    # ── classical feed-forward ───────────────────────────────────────────────

    def if_bit(
        self,
        cbit: int,
        value: int,
        name: str,
        qubits: list[int],
        params: list | None = None,
    ) -> "DynamicCircuit":
        """Apply a gate conditionally on a classical bit value.

        The gate ``name`` is applied to ``qubits`` only when
        ``classical_register[cbit] == value``.

        Parameters
        ----------
        cbit:
            Index of the classical bit to test.
        value:
            Required bit value (0 or 1) to trigger the gate.
        name:
            Gate name (same as :meth:`add_gate`).
        qubits:
            Target qubits for the gate.
        params:
            Optional gate parameters.
        """
        self._check_cbits([cbit])
        self._check_qubits(qubits)
        if value not in (0, 1):
            raise DynamicCircuitError(
                f"if_bit value must be 0 or 1, got {value!r}."
            )
        self.operations.append({
            "type": "if",
            "cbit": cbit,
            "value": value,
            "gate": {"name": name, "qubits": list(qubits), "params": list(params or [])},
        })
        return self

    # ── repeat-until-success ─────────────────────────────────────────────────

    def repeat_until(
        self,
        success_cbit: int,
        success_value: int,
        body: list[dict],
        reset_qubits: list[int] | None = None,
        max_iterations: int = 100,
    ) -> "DynamicCircuit":
        """Bounded repeat-until-success (RUS) loop.

        Runs the operations in ``body`` repeatedly until
        ``classical_register[success_cbit] == success_value``, up to
        ``max_iterations`` attempts.  After each failed attempt, the qubits
        listed in ``reset_qubits`` are reset to |0⟩ before retrying.

        Parameters
        ----------
        success_cbit:
            Classical bit index checked for loop termination.
        success_value:
            Loop exits when ``cbits[success_cbit] == success_value``.
        body:
            List of operation dicts (same format as ``self.operations``).
        reset_qubits:
            Qubits to reset after a failed attempt (e.g. ancilla qubits).
        max_iterations:
            Safety cap on the number of retries (default 100).
        """
        self._check_cbits([success_cbit])
        if success_value not in (0, 1):
            raise DynamicCircuitError(
                f"success_value must be 0 or 1, got {success_value!r}."
            )
        if max_iterations < 1:
            raise DynamicCircuitError(
                f"max_iterations must be >= 1, got {max_iterations}."
            )
        self.operations.append({
            "type": "repeat_until",
            "success_cbit": success_cbit,
            "success_value": success_value,
            "body": list(body),
            "reset_qubits": list(reset_qubits or []),
            "max_iterations": max_iterations,
        })
        return self

    # ── helpers ──────────────────────────────────────────────────────────────

    def _check_qubits(self, qubits: list[int]) -> None:
        for q in qubits:
            if not (0 <= q < self.n_qubits):
                raise DynamicCircuitError(
                    f"Qubit index {q} out of range [0, {self.n_qubits})."
                )

    def _check_cbits(self, cbits: list[int]) -> None:
        for c in cbits:
            if not (0 <= c < self.n_cbits):
                raise DynamicCircuitError(
                    f"Classical bit index {c} out of range [0, {self.n_cbits})."
                )

    def __repr__(self) -> str:
        return (
            f"DynamicCircuit(n_qubits={self.n_qubits}, "
            f"n_cbits={self.n_cbits}, "
            f"n_ops={len(self.operations)})"
        )


# ===========================================================================
# DynamicResult
# ===========================================================================

@dataclass
class DynamicResult:
    """Result from :meth:`DynamicSimulator.run`.

    Attributes
    ----------
    counts:
        Dict mapping bitstring (MSB-first, length = n_cbits) to the number of
        shots that produced that classical register value.
    n_shots:
        Total number of simulated trajectories.
    cbits_history:
        List of classical register snapshots — one ``list[int]`` per shot.
        Useful for debugging conditional-gate behaviour.
    metadata:
        Simulation metadata dict.
    """
    counts: dict
    n_shots: int
    cbits_history: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def quasi_dists(self) -> dict:
        """Normalise counts to quasi-probabilities (count / n_shots)."""
        return {k: v / self.n_shots for k, v in self.counts.items()}

    def most_likely(self) -> str:
        """Return the bitstring with the highest count."""
        if not self.counts:
            raise DynamicCircuitError("No counts available.")
        return max(self.counts, key=lambda k: self.counts[k])


# ===========================================================================
# Low-level statevector measurement primitives
# ===========================================================================

def _measure_sv_inplace(
    sv: "np.ndarray",
    qubit: int,
    rng: "np.random.Generator",
) -> tuple[int, "np.ndarray"]:
    """Sample a measurement outcome and collapse the statevector.

    Parameters
    ----------
    sv:
        Statevector of length 2**n.
    qubit:
        Which qubit to measure (LSB = qubit 0).
    rng:
        Numpy random generator for reproducibility.

    Returns
    -------
    (outcome, collapsed_sv)
        ``outcome`` is 0 or 1; ``collapsed_sv`` is normalised.
    """
    d = sv.shape[0]
    probs = np.abs(sv) ** 2
    # Indices where bit `qubit` == 0
    mask0 = np.array([(j >> qubit) & 1 == 0 for j in range(d)])
    p0 = float(np.sum(probs[mask0]))
    p0 = max(0.0, min(1.0, p0))
    outcome = int(rng.choice(2, p=[p0, 1.0 - p0]))
    mask = mask0 if outcome == 0 else ~mask0
    sv_new = sv * mask
    norm = float(np.linalg.norm(sv_new))
    if norm > 1e-15:
        sv_new = sv_new / norm
    return outcome, sv_new


def _reset_sv_inplace(
    sv: "np.ndarray",
    qubit: int,
    rng: "np.random.Generator",
) -> "np.ndarray":
    """Reset qubit to |0⟩: measure, then apply X if outcome was 1."""
    outcome, sv = _measure_sv_inplace(sv, qubit, rng)
    if outcome == 1:
        X = _get_gate_matrix("x", None)
        sv = _apply_gate_sv(sv, X, [qubit])
    return sv


# ===========================================================================
# DynamicSimulator
# ===========================================================================

class DynamicSimulator:
    """Statevector trajectory simulator for dynamic circuits.

    Each shot is an independent wavefunction trajectory.  Mid-circuit
    measurements probabilistically collapse the wavefunction; conditional gates
    are applied based on the resulting classical register state.

    Parameters
    ----------
    noise_model:
        Optional :class:`~quantx_engine.noise.NoiseModel` instance.  When
        set, its channels act on every qubit after every gate, as on the
        native circuit backend (:func:`~quantx_engine.noise.native_channels`:
        depolarizing, amplitude damping, bit and phase flips, T1/T2 decay).
        Each shot is one trajectory, so the noise is sampled as random Pauli
        kicks and quantum jumps and every trajectory stays a pure state.
    shots:
        Number of independent trajectories to run.
    seed:
        Random seed for reproducibility.

    Example
    -------
    ::

        circ = DynamicCircuit(2, 2).h(0).cx(0, 1).measure_all()
        result = DynamicSimulator(shots=500, seed=0).run(circ)
        print(result.counts)   # {"00": ~250, "11": ~250}
    """

    def __init__(
        self,
        noise_model=None,
        shots: int = 1,
        seed: int | None = None,
    ) -> None:
        _require_numpy()
        if shots < 1:
            raise DynamicCircuitError(f"shots must be >= 1, got {shots}.")
        self.noise_model = noise_model
        self.shots = shots
        self.seed = seed

    def run(self, circuit: DynamicCircuit) -> DynamicResult:
        """Simulate a dynamic circuit for ``self.shots`` trajectories.

        Parameters
        ----------
        circuit:
            The :class:`DynamicCircuit` to simulate.

        Returns
        -------
        DynamicResult
        """
        _require_numpy()
        if not isinstance(circuit, DynamicCircuit):
            raise DynamicCircuitError(
                f"Expected DynamicCircuit, got {type(circuit).__name__}."
            )

        # The channels the native backend would apply, so a noise model means the same
        # thing here: every qubit after every gate (see noise.native_channels).
        from .noise import native_channels
        channels = native_channels(self.noise_model)

        rng = np.random.default_rng(self.seed)
        counts: dict[str, int] = {}
        cbits_history: list[list[int]] = []

        for _ in range(self.shots):
            sv, cbits = self._run_trajectory(
                circuit, rng, channels
            )
            key = _cbits_to_str(cbits, circuit.n_cbits)
            counts[key] = counts.get(key, 0) + 1
            cbits_history.append(list(cbits))

        return DynamicResult(
            counts=counts,
            n_shots=self.shots,
            cbits_history=cbits_history,
            metadata={
                "n_qubits": circuit.n_qubits,
                "n_cbits": circuit.n_cbits,
                "n_ops": len(circuit.operations),
                "shots": self.shots,
                "noise_channels": [list(c) for c in channels],
            },
        )

    # ── trajectory engine ────────────────────────────────────────────────────

    def _run_trajectory(
        self,
        circuit: DynamicCircuit,
        rng: "np.random.Generator",
        channels: list[tuple[str, float]],
    ) -> tuple["np.ndarray", list[int]]:
        """Run one shot of the circuit; return (final_sv, classical_bits)."""
        nq = circuit.n_qubits
        sv = np.zeros(1 << nq, dtype=complex)
        sv[0] = 1.0
        cbits = [0] * circuit.n_cbits

        def _exec(op: dict) -> None:
            nonlocal sv
            otype = op["type"]

            if otype == "gate":
                U = _get_gate_matrix(
                    op["name"],
                    op["params"] if op["params"] else None,
                )
                sv = _apply_gate_sv(sv, U, op["qubits"])
                if channels:
                    sv = _apply_noise_channels_sv(sv, channels, nq, rng)

            elif otype == "measure":
                outcome, sv = _measure_sv_inplace(sv, op["qubit"], rng)
                cbits[op["cbit"]] = outcome

            elif otype == "reset":
                sv = _reset_sv_inplace(sv, op["qubit"], rng)

            elif otype == "if":
                if cbits[op["cbit"]] == op["value"]:
                    gate = op["gate"]
                    U = _get_gate_matrix(
                        gate["name"],
                        gate["params"] if gate["params"] else None,
                    )
                    sv = _apply_gate_sv(sv, U, gate["qubits"])
                    if channels:
                        sv = _apply_noise_channels_sv(sv, channels, nq, rng)

            elif otype == "repeat_until":
                success_cbit  = op["success_cbit"]
                success_value = op["success_value"]
                body          = op["body"]
                reset_qubits  = op["reset_qubits"]
                max_iter      = op["max_iterations"]

                for _ in range(max_iter):
                    for body_op in body:
                        _exec(body_op)
                    if cbits[success_cbit] == success_value:
                        break
                    # Reset ancilla qubits and retry
                    for q in reset_qubits:
                        sv = _reset_sv_inplace(sv, q, rng)

        for op in circuit.operations:
            _exec(op)

        return sv, cbits


# ===========================================================================
# Noise: the native backend's channels, sampled per trajectory
# ===========================================================================

def _apply_noise_channels_sv(
    sv: "np.ndarray",
    channels: list[tuple[str, float]],
    n_qubits: int,
    rng: "np.random.Generator",
) -> "np.ndarray":
    """One trajectory's draw of the noise channels on every qubit, in the native backend's
    order (each qubit in turn, each channel in turn; see ``noise.native_channels``).

    Depolarizing p applies a uniformly random X, Y or Z with probability 3p/4, which
    averages to (1 - p)rho + p I/2. Amplitude damping gamma is a quantum jump: |1> decays
    to |0> with probability gamma times the qubit's |1> population. A flip applies X or Z
    with probability p.
    """
    from .primitives import _gate_matrices
    mats = _gate_matrices()
    paulis = [mats["x"], mats["y"], mats["z"]]
    for q in range(n_qubits):
        for kind, rate in channels:
            if kind == "depolarizing":
                if rng.random() < 0.75 * rate:
                    sv = _apply_gate_sv(sv, paulis[int(rng.integers(3))], [q])
            elif kind == "amplitude_damping":
                sv = _amplitude_damp_sv(sv, q, rate, rng)
            elif kind in ("bit_flip", "phase_flip"):
                if rng.random() < rate:
                    sv = _apply_gate_sv(sv, mats["x" if kind == "bit_flip" else "z"], [q])
    return sv


def _amplitude_damp_sv(
    sv: "np.ndarray",
    qubit: int,
    gamma: float,
    rng: "np.random.Generator",
) -> "np.ndarray":
    """Amplitude damping on one qubit, as a quantum jump (LSB = qubit 0)."""
    if gamma <= 0.0:
        return sv
    idx = np.arange(sv.shape[0])
    one = ((idx >> qubit) & 1).astype(bool)
    p1 = float(np.sum(np.abs(sv[one]) ** 2))
    if rng.random() < gamma * p1:
        new = np.zeros_like(sv)
        new[idx[one] ^ (1 << qubit)] = sv[one]          # |1> -> |0> on this qubit
    else:
        new = sv.copy()
        new[one] *= math.sqrt(max(1.0 - gamma, 0.0))  # the no-jump branch
    norm = float(np.linalg.norm(new))
    return new / norm if norm > 0.0 else sv


# ===========================================================================
# Helpers
# ===========================================================================

def _cbits_to_str(cbits: list[int], n_cbits: int) -> str:
    """Convert classical register list to MSB-first bitstring of length n_cbits."""
    s = "".join(str(cbits[i]) for i in range(n_cbits - 1, -1, -1))
    return s
