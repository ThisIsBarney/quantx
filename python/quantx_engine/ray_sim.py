"""M111: Distributed statevector simulation via Ray.

Implements a sharded statevector simulator that distributes the 2^n
amplitude array across multiple Ray actors (one per shard), enabling
simulation of large circuits across a multi-core machine or cluster.
Falls back transparently to single-process numpy when Ray is unavailable.

Sharding strategy
-----------------
The 2^n statevector is partitioned into ``num_shards`` contiguous blocks,
each of size ``shard_size = 2^n // num_shards``.  Shard *k* owns amplitudes
with indices in the range ``[k * shard_size, (k+1) * shard_size)``.

Gate classification
-------------------
For an n-qubit statevector split into 2^s shards (s = log2(num_shards)):

- **Local gates** (qubit index < n − s): both paired amplitudes lie within
  the same shard → each actor applies the gate independently in parallel.
- **Global gates** (qubit index >= n − s): paired amplitudes live in
  *different* shards → the two affected shard actors exchange data, apply
  the gate jointly, and write back.  In the CPU-fallback path the same
  exchange is performed in-process using numpy.

Classes
-------
DistSVError
    Exception for invalid operation requests.
DistSVResult
    Benchmark result dataclass.
DistributedSVSimulator
    Sharded statevector simulator.  Supports ``h``, ``s``, ``t``, ``x``,
    ``y``, ``z``, ``cx``, ``cz``, ``rx``, ``ry``, ``rz``, ``to_statevector``,
    ``measure_all``.

Functions
---------
is_ray_available()
    Returns True if Ray is importable and a cluster can be initialised.
dist_sv_benchmark(n_qubits, circuit_depth, num_shards, seed)
    Times a random circuit on both single-process and distributed paths,
    returning a ``DistSVResult``.

Quick start::

    from quantx_engine.ray_sim import DistributedSVSimulator

    sim = DistributedSVSimulator(n_qubits=20, num_shards=4)
    sim.h(0)
    for q in range(19):
        sim.cx(q, q + 1)
    sv = sim.to_statevector()        # numpy array of shape (2**20,)
    print(abs(sv[0])**2, abs(sv[-1])**2)  # ≈ 0.5 each (GHZ)
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "DistSVError",
    "DistSVResult",
    "DistributedSVSimulator",
    "is_ray_available",
    "dist_sv_benchmark",
]


# ── Optional Ray import ───────────────────────────────────────────────────────

def is_ray_available() -> bool:
    """Return True if Ray is installed and can be initialised."""
    try:
        import ray  # noqa: F401
        return True
    except ImportError:
        return False


# ── Exceptions ────────────────────────────────────────────────────────────────

class DistSVError(ValueError):
    """Raised for invalid distributed statevector operations."""


# ── Benchmark result ──────────────────────────────────────────────────────────

@dataclass
class DistSVResult:
    """Result of :func:`dist_sv_benchmark`.

    Attributes
    ----------
    n_qubits : int
    num_shards : int
    circuit_depth : int
    cpu_seconds : float
        Wall time for single-process numpy simulation.
    dist_seconds : float
        Wall time for distributed (or CPU-fallback) simulation.
    speedup : float
        ``cpu_seconds / dist_seconds``.
    fidelity : float
        |⟨ψ_cpu | ψ_dist⟩|² — should be ≈ 1.0 if both runs are correct.
    ray_available : bool
        Whether a live Ray cluster was used.
    """
    n_qubits: int
    num_shards: int
    circuit_depth: int
    cpu_seconds: float
    dist_seconds: float
    speedup: float
    fidelity: float
    ray_available: bool


# ── Gate matrices ─────────────────────────────────────────────────────────────

_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)


def _rx(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)


def _ry(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=np.complex128)


# ── Low-level sharded gate kernels ────────────────────────────────────────────

def _apply_single_local(shard: np.ndarray, gate: np.ndarray,
                        local_qubit: int) -> np.ndarray:
    """Apply a 2×2 gate to *local_qubit* within a shard array.

    Parameters
    ----------
    shard : np.ndarray, shape (shard_size,), dtype complex128
    gate : np.ndarray, shape (2,2)
    local_qubit : int   index within the shard (0 = LSB of shard index)

    Returns
    -------
    np.ndarray — updated shard (same object, mutated in-place, also returned).
    """
    sz = len(shard)
    n_local = int(math.log2(sz))
    # Qubit 0 is the LSB, so it occupies the *last* axis of the [2]*n_local
    # tensor (C-order reshape: axis 0 = MSB, axis n_local-1 = LSB).
    axis = n_local - 1 - local_qubit
    sv2 = shard.reshape([2] * n_local)
    sv2 = np.tensordot(gate, sv2, axes=[[1], [axis]])
    sv2 = np.moveaxis(sv2, 0, axis)
    shard[:] = sv2.reshape(sz)
    return shard


def _apply_global_pair(shard0: np.ndarray, shard1: np.ndarray,
                       gate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a 2×2 gate across a shard pair (shard0 = |q=0⟩, shard1 = |q=1⟩).

    The full gate is::

        out0 = gate[0,0]*shard0 + gate[0,1]*shard1
        out1 = gate[1,0]*shard0 + gate[1,1]*shard1

    Parameters
    ----------
    shard0, shard1 : np.ndarray, shape (shard_size,)
    gate : np.ndarray, shape (2,2)

    Returns
    -------
    (new_shard0, new_shard1)
    """
    new0 = gate[0, 0] * shard0 + gate[0, 1] * shard1
    new1 = gate[1, 0] * shard0 + gate[1, 1] * shard1
    return new0, new1


def _apply_cx_local(shard: np.ndarray, ctrl_local: int,
                    tgt_local: int) -> np.ndarray:
    """CNOT with both ctrl and target in the same shard (both local qubits)."""
    sz = len(shard)
    n_local = int(math.log2(sz))
    ctrl_axis = n_local - 1 - ctrl_local  # qubit 0 = LSB = last axis
    tgt_axis = n_local - 1 - tgt_local
    sv2 = shard.reshape([2] * n_local)
    # Swap amplitudes: (ctrl=1, tgt=0) ↔ (ctrl=1, tgt=1)
    idx0: list[object] = [slice(None)] * n_local
    idx1: list[object] = [slice(None)] * n_local
    idx0[ctrl_axis] = 1
    idx1[ctrl_axis] = 1
    idx0[tgt_axis] = 0
    idx1[tgt_axis] = 1
    sv2[tuple(idx0)], sv2[tuple(idx1)] = (
        sv2[tuple(idx1)].copy(), sv2[tuple(idx0)].copy()
    )
    shard[:] = sv2.reshape(sz)
    return shard


# ── Ray shard actor (only defined when Ray is importable) ─────────────────────

def _make_ray_actor_class():
    """Return a Ray remote actor class, or None if Ray is unavailable."""
    try:
        import ray

        @ray.remote
        class _ShardActor:
            """Remote actor holding one shard of the statevector."""

            def __init__(self, shard_id: int, state: np.ndarray) -> None:
                self._id = shard_id
                self._state = state.copy()

            def get(self) -> np.ndarray:
                return self._state.copy()

            def set(self, state: np.ndarray) -> None:
                self._state = state.copy()

            def apply_local(self, gate: np.ndarray, local_qubit: int) -> None:
                _apply_single_local(self._state, gate, local_qubit)

            def apply_cx_local(self, ctrl_local: int, tgt_local: int) -> None:
                _apply_cx_local(self._state, ctrl_local, tgt_local)

            def norm(self) -> float:
                return float(np.dot(self._state.conj(), self._state).real)

        return _ShardActor

    except ImportError:
        return None


# ── DistributedSVSimulator ────────────────────────────────────────────────────

class DistributedSVSimulator:
    """Sharded statevector simulator with optional Ray backend.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (state vector size 2^n_qubits).
    num_shards : int | None
        Number of shards.  Must be a power of 2 and divide 2^n_qubits.
        Defaults to ``min(4, 2**n_qubits)``.
    backend : {"auto", "ray", "cpu"}
        "auto"  → use Ray if available, else CPU.
        "ray"   → require Ray (raises DistSVError if unavailable).
        "cpu"   → always use CPU regardless of Ray installation.
    """

    def __init__(
        self,
        n_qubits: int,
        num_shards: int | None = None,
        backend: str = "auto",
    ) -> None:
        if n_qubits < 1:
            raise DistSVError("n_qubits must be ≥ 1")

        self._n = n_qubits
        size = 1 << n_qubits

        if num_shards is None:
            num_shards = min(4, size)
        if num_shards < 1 or (num_shards & (num_shards - 1)) != 0:
            raise DistSVError("num_shards must be a power of 2")
        if size % num_shards != 0:
            raise DistSVError(
                f"num_shards={num_shards} does not divide 2^n_qubits={size}"
            )

        self._num_shards = num_shards
        self._shard_size = size // num_shards
        # s = number of global qubit bits (partitioned across shards)
        self._s = int(math.log2(num_shards)) if num_shards > 1 else 0
        self._n_local = n_qubits - self._s   # bits within each shard

        # Resolve backend
        if backend == "ray":
            if not is_ray_available():
                raise DistSVError("Ray is not installed; use backend='cpu'")
            self._use_ray = True
        elif backend == "cpu":
            self._use_ray = False
        else:  # "auto"
            self._use_ray = is_ray_available()

        if self._use_ray:
            warnings.warn(
                "Ray backend requested but not yet initialised in this process; "
                "call ray.init() before using DistributedSVSimulator(backend='ray').",
                stacklevel=2,
            )

        # Initialise |0...0⟩
        sv = np.zeros(size, dtype=np.complex128)
        sv[0] = 1.0
        # Store as list of shard arrays (Ray actors in Ray mode)
        self._shards: list = self._build_shards(sv)

    # ── Shard management ─────────────────────────────────────────────────────

    def _build_shards(self, sv: np.ndarray) -> list:
        chunks = [sv[k * self._shard_size:(k + 1) * self._shard_size].copy()
                  for k in range(self._num_shards)]
        if self._use_ray:
            actor_cls = _make_ray_actor_class()
            if actor_cls is None:
                self._use_ray = False
                return chunks
            try:
                import ray
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                return [actor_cls.remote(k, chunks[k])
                        for k in range(self._num_shards)]
            except Exception:
                self._use_ray = False
                return chunks
        return chunks

    def _get_shard(self, k: int) -> np.ndarray:
        if self._use_ray:
            import ray
            return ray.get(self._shards[k].get.remote())
        return self._shards[k]

    def _set_shard(self, k: int, arr: np.ndarray) -> None:
        if self._use_ray:
            self._shards[k].set.remote(arr)
        else:
            self._shards[k][:] = arr

    # ── Gate dispatch ─────────────────────────────────────────────────────────

    def _classify_qubit(self, qubit: int) -> tuple[str, int]:
        """Return ('local', local_idx) or ('global', global_bit_idx)."""
        if qubit < self._n_local:
            return "local", qubit
        else:
            # global bit index within the shard-selector bits
            return "global", qubit - self._n_local

    def _apply_single(self, gate: np.ndarray, qubit: int) -> None:
        """Apply a single-qubit gate, dispatching local vs global."""
        self._check_qubit(qubit)
        kind, idx = self._classify_qubit(qubit)

        if kind == "local":
            # Each shard applies independently
            if self._use_ray:
                import ray
                ray.get([sh.apply_local.remote(gate, idx)
                         for sh in self._shards])
            else:
                for shard in self._shards:
                    _apply_single_local(shard, gate, idx)

        else:
            # Global gate: shard pairs determined by the global bit `idx`
            # Shard k and shard k XOR (1 << idx) are paired
            paired: set[int] = set()
            for k in range(self._num_shards):
                partner = k ^ (1 << idx)
                if k < partner and k not in paired:
                    paired.add(k)
                    s0 = self._get_shard(k)
                    s1 = self._get_shard(partner)
                    new0, new1 = _apply_global_pair(s0, s1, gate)
                    self._set_shard(k, new0)
                    self._set_shard(partner, new1)

    def _apply_cx(self, ctrl: int, tgt: int) -> None:
        """CNOT gate."""
        self._check_qubit(ctrl)
        self._check_qubit(tgt)
        if ctrl == tgt:
            raise DistSVError(f"CNOT ctrl and target must differ: got {ctrl}")

        ctrl_kind, ctrl_idx = self._classify_qubit(ctrl)
        tgt_kind, tgt_idx = self._classify_qubit(tgt)

        if ctrl_kind == "local" and tgt_kind == "local":
            # Both local: apply within each shard independently
            if self._use_ray:
                import ray
                ray.get([sh.apply_cx_local.remote(ctrl_idx, tgt_idx)
                         for sh in self._shards])
            else:
                for shard in self._shards:
                    _apply_cx_local(shard, ctrl_idx, tgt_idx)

        else:
            # At least one global qubit: gather full state, apply, scatter
            sv = self.to_statevector()
            sv = _cx_statevector(sv, ctrl, tgt, self._n)
            shards = [sv[k * self._shard_size:(k + 1) * self._shard_size].copy()
                      for k in range(self._num_shards)]
            for k, arr in enumerate(shards):
                self._set_shard(k, arr)

    # ── Public gate API ───────────────────────────────────────────────────────

    def h(self, qubit: int) -> "DistributedSVSimulator":
        self._apply_single(_H, qubit); return self

    def x(self, qubit: int) -> "DistributedSVSimulator":
        self._apply_single(_X, qubit); return self

    def y(self, qubit: int) -> "DistributedSVSimulator":
        self._apply_single(_Y, qubit); return self

    def z(self, qubit: int) -> "DistributedSVSimulator":
        self._apply_single(_Z, qubit); return self

    def s(self, qubit: int) -> "DistributedSVSimulator":
        self._apply_single(_S, qubit); return self

    def t(self, qubit: int) -> "DistributedSVSimulator":
        self._apply_single(_T, qubit); return self

    def rx(self, theta: float, qubit: int) -> "DistributedSVSimulator":
        self._apply_single(_rx(theta), qubit); return self

    def ry(self, theta: float, qubit: int) -> "DistributedSVSimulator":
        self._apply_single(_ry(theta), qubit); return self

    def rz(self, theta: float, qubit: int) -> "DistributedSVSimulator":
        self._apply_single(_rz(theta), qubit); return self

    def cx(self, ctrl: int, tgt: int) -> "DistributedSVSimulator":
        self._apply_cx(ctrl, tgt); return self

    def cz(self, q0: int, q1: int) -> "DistributedSVSimulator":
        self.h(q1); self.cx(q0, q1); self.h(q1); return self

    # ── State access ──────────────────────────────────────────────────────────

    def to_statevector(self) -> np.ndarray:
        """Return the full statevector as a numpy array of shape (2^n,)."""
        chunks = [self._get_shard(k) for k in range(self._num_shards)]
        return np.concatenate(chunks)

    def norm(self) -> float:
        """Return ‖ψ‖², should be 1.0 for a normalised state."""
        sv = self.to_statevector()
        return float(np.dot(sv.conj(), sv).real)

    def measure_all(self, seed: int | None = None) -> list[int]:
        """Destructive Z-basis measurement of all qubits.

        Returns
        -------
        list[int] of length n_qubits, each 0 or 1.
        """
        sv = self.to_statevector()
        probs = (sv.conj() * sv).real
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()

        rng = np.random.default_rng(seed)
        outcome_idx = int(rng.choice(len(probs), p=probs))
        outcomes = [(outcome_idx >> q) & 1 for q in range(self._n)]

        # Collapse state
        sv_new = np.zeros_like(sv)
        sv_new[outcome_idx] = 1.0
        shards = [sv_new[k * self._shard_size:(k + 1) * self._shard_size].copy()
                  for k in range(self._num_shards)]
        for k, arr in enumerate(shards):
            self._set_shard(k, arr)

        return outcomes

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def num_shards(self) -> int:
        return self._num_shards

    @property
    def backend(self) -> str:
        return "ray" if self._use_ray else "cpu"

    def _check_qubit(self, q: int) -> None:
        if not (0 <= q < self._n):
            raise DistSVError(f"Qubit {q} out of range [0, {self._n})")

    def __repr__(self) -> str:  # pragma: no cover
        return (f"DistributedSVSimulator(n_qubits={self._n}, "
                f"num_shards={self._num_shards}, backend={self.backend!r})")


# ── Helper: CNOT on full statevector ─────────────────────────────────────────

def _cx_statevector(sv: np.ndarray, ctrl: int, tgt: int, n: int) -> np.ndarray:
    """Apply CX(ctrl, tgt) to statevector *sv* of 2^n amplitudes."""
    sv2 = sv.reshape([2] * n)
    # Qubit q occupies axis (n-1-q) in the C-ordered [2]*n tensor.
    ctrl_axis = n - 1 - ctrl
    tgt_axis = n - 1 - tgt
    idx0: list[object] = [slice(None)] * n
    idx1: list[object] = [slice(None)] * n
    idx0[ctrl_axis] = 1; idx0[tgt_axis] = 0
    idx1[ctrl_axis] = 1; idx1[tgt_axis] = 1
    sv2[tuple(idx0)], sv2[tuple(idx1)] = (
        sv2[tuple(idx1)].copy(), sv2[tuple(idx0)].copy()
    )
    return sv2.reshape(2 ** n)


# ── Single-process reference simulator ───────────────────────────────────────

class _CPUSimulator:
    """Minimal single-process statevector simulator for benchmarking."""

    def __init__(self, n_qubits: int) -> None:
        self._n = n_qubits
        sv = np.zeros(1 << n_qubits, dtype=np.complex128)
        sv[0] = 1.0
        self._sv = sv

    def _apply_1q(self, gate: np.ndarray, q: int) -> None:
        n = self._n
        axis = n - 1 - q  # qubit 0 = LSB = last axis in C-ordered reshape
        sv2 = self._sv.reshape([2] * n)
        sv2 = np.tensordot(gate, sv2, axes=[[1], [axis]])
        self._sv = np.moveaxis(sv2, 0, axis).reshape(1 << n)

    def h(self, q: int) -> "_CPUSimulator":   self._apply_1q(_H, q); return self
    def x(self, q: int) -> "_CPUSimulator":   self._apply_1q(_X, q); return self
    def ry(self, t: float, q: int) -> "_CPUSimulator":
        self._apply_1q(_ry(t), q); return self
    def rz(self, t: float, q: int) -> "_CPUSimulator":
        self._apply_1q(_rz(t), q); return self

    def cx(self, ctrl: int, tgt: int) -> "_CPUSimulator":
        self._sv = _cx_statevector(self._sv, ctrl, tgt, self._n)
        return self

    def cz(self, q0: int, q1: int) -> "_CPUSimulator":
        self.h(q1); self.cx(q0, q1); self.h(q1); return self

    def to_statevector(self) -> np.ndarray:
        return self._sv.copy()


# ── Benchmark ─────────────────────────────────────────────────────────────────

def dist_sv_benchmark(
    n_qubits: int = 20,
    circuit_depth: int = 4,
    num_shards: int = 4,
    seed: int = 0,
    warmup: bool = True,
) -> DistSVResult:
    """Benchmark single-process vs distributed statevector simulation.

    Runs a random brickwork circuit (alternating Ry + CZ layers) and
    measures wall-clock time for both the CPU reference and the sharded
    ``DistributedSVSimulator``.

    Parameters
    ----------
    n_qubits : int, default 20
    circuit_depth : int, default 4
    num_shards : int, default 4
    seed : int, default 0
    warmup : bool, default True
        Run a tiny warm-up circuit before timing to avoid cold-start bias.

    Returns
    -------
    DistSVResult
    """
    rng = np.random.default_rng(seed)

    def _build_circuit_ops(n: int, depth: int) -> list[tuple]:
        """Return list of (op, args) for a random brickwork circuit."""
        ops: list[tuple] = []
        angles = rng.uniform(0, 2 * np.pi, (depth, n))
        for d in range(depth):
            for q in range(n):
                ops.append(("ry", (float(angles[d, q]), q)))
            for parity in (d % 2, (d + 1) % 2):
                for q in range(parity, n - 1, 2):
                    ops.append(("cz", (q, q + 1)))
        return ops

    def _run_cpu(ops: list[tuple]) -> tuple[np.ndarray, float]:
        sim = _CPUSimulator(n_qubits)
        t0 = time.perf_counter()
        for op, args in ops:
            getattr(sim, op)(*args)
        elapsed = time.perf_counter() - t0
        return sim.to_statevector(), elapsed

    def _run_dist(ops: list[tuple]) -> tuple[np.ndarray, float]:
        sim = DistributedSVSimulator(n_qubits, num_shards=num_shards, backend="cpu")
        t0 = time.perf_counter()
        for op, args in ops:
            getattr(sim, op)(*args)
        elapsed = time.perf_counter() - t0
        return sim.to_statevector(), elapsed

    ops = _build_circuit_ops(n_qubits, circuit_depth)

    if warmup:
        _run_cpu([("h", (0,)), ("cx", (0, 1))])
        _run_dist([("h", (0,)), ("cx", (0, 1))])

    sv_cpu, cpu_t = _run_cpu(ops)
    sv_dist, dist_t = _run_dist(ops)

    fidelity = float(abs(np.dot(sv_cpu.conj(), sv_dist)) ** 2)
    speedup = cpu_t / dist_t if dist_t > 0 else 1.0

    return DistSVResult(
        n_qubits=n_qubits,
        num_shards=num_shards,
        circuit_depth=circuit_depth,
        cpu_seconds=cpu_t,
        dist_seconds=dist_t,
        speedup=speedup,
        fidelity=fidelity,
        ray_available=is_ray_available(),
    )
