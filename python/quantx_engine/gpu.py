"""GPU acceleration backend for QuantX using CuPy.

All classes fall back to NumPy automatically when CuPy is not installed,
so this module is usable on CPU-only machines and CI runners.

Usage::

    from quantx_engine.gpu import GpuStatevector, GpuLanczos, gpu_info

    # Auto-selects GPU if CuPy available, falls back to numpy otherwise
    sv = GpuStatevector(n_qubits=10, backend="auto")
    sv.apply_gate("H", 0)
    sv.apply_gate("CNOT", [0, 1])
    probs = sv.probabilities()

    # GPU-accelerated Lanczos
    from quantx_engine.lanczos import SparseHamiltonian
    H = SparseHamiltonian.ising_1d(n_sites=14)
    solver = GpuLanczos(k=4, backend="auto")
    result = solver.solve(H)

    print(gpu_info())
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

__all__ = [
    "GpuError",
    "GpuBackend",
    "GpuStatevector",
    "GpuLanczos",
    "GpuBatchedCircuit",
    "gpu_info",
    "to_gpu",
    "to_cpu",
]


class GpuError(RuntimeError):
    """Raised for GPU backend errors or unsupported operations."""


@dataclass
class GpuBackend:
    name: str        # "cupy", "numpy", or "auto"
    _xp: Any = field(init=False, repr=False)

    def __post_init__(self):
        if self.name == "auto":
            self._xp = cp if _HAS_CUPY else np
            self.name = "cupy" if _HAS_CUPY else "numpy"
        elif self.name == "cupy":
            if not _HAS_CUPY:
                raise GpuError("CuPy is not installed. Install with: pip install cupy-cuda12x")
            self._xp = cp
        elif self.name == "numpy":
            self._xp = np
        else:
            raise GpuError(f"Unknown backend: {self.name!r}. Choose 'auto', 'cupy', or 'numpy'.")

    @property
    def xp(self):
        """Return the array module (cupy or numpy)."""
        return self._xp

    def array(self, data, dtype=None):
        return self._xp.array(data, dtype=dtype)

    def zeros(self, shape, dtype=complex):
        return self._xp.zeros(shape, dtype=dtype)

    def eye(self, n, dtype=complex):
        return self._xp.eye(n, dtype=dtype)

    def to_numpy(self, arr) -> "np.ndarray":
        """Move array to CPU numpy."""
        if _HAS_CUPY and isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def to_device(self, arr):
        """Move numpy array to this backend's device."""
        if self.name == "cupy":
            return cp.asarray(arr)
        return np.asarray(arr)

    def is_gpu(self) -> bool:
        return self.name == "cupy"

    def to_dict(self) -> dict:
        return {"backend": self.name, "gpu_available": _HAS_CUPY}


@dataclass
class GpuStatevector:
    n_qubits: int
    backend: str = "auto"
    dtype: str = "complex128"   # "complex128" (fp64) or "complex64" (fp32)
    _backend: GpuBackend = field(init=False, repr=False)
    _sv: Any = field(init=False, repr=False)    # shape (2**n_qubits,)
    _dtype: Any = field(init=False, repr=False) # resolved numpy dtype

    def __post_init__(self):
        if self.n_qubits < 1:
            raise GpuError("n_qubits must be >= 1")
        if self.dtype not in ("complex64", "complex128"):
            raise GpuError(
                f"dtype must be 'complex64' or 'complex128', got {self.dtype!r}"
            )
        self._backend = GpuBackend(self.backend)
        self._dtype = np.complex64 if self.dtype == "complex64" else np.complex128
        xp = self._backend.xp
        sv = xp.zeros(2**self.n_qubits, dtype=self._dtype)
        sv[0] = 1.0
        self._sv = sv

    def reset(self):
        """Reset to |0...0⟩."""
        xp = self._backend.xp
        self._sv = xp.zeros(2**self.n_qubits, dtype=self._dtype)
        self._sv[0] = 1.0

    def _gate_matrix(self, name: str, params=()) -> "np.ndarray":
        """Return 2x2 gate matrix as a numpy array (converted to device in apply_gate)."""
        if name in ("X", "PauliX"):
            return np.array([[0, 1], [1, 0]], dtype=complex)
        elif name in ("Y", "PauliY"):
            return np.array([[0, -1j], [1j, 0]], dtype=complex)
        elif name in ("Z", "PauliZ"):
            return np.diag([1.0 + 0j, -1.0 + 0j])
        elif name in ("H", "Hadamard"):
            return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        elif name == "S":
            return np.diag([1.0 + 0j, 1j])
        elif name == "T":
            return np.diag([1.0 + 0j, np.exp(1j * np.pi / 4)])
        elif name in ("I", "Identity"):
            return np.eye(2, dtype=complex)
        elif name == "RX":
            t = float(params[0])
            return np.array([[np.cos(t / 2), -1j * np.sin(t / 2)],
                             [-1j * np.sin(t / 2), np.cos(t / 2)]], dtype=complex)
        elif name == "RY":
            t = float(params[0])
            return np.array([[np.cos(t / 2), -np.sin(t / 2)],
                             [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)
        elif name == "RZ":
            t = float(params[0])
            return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])
        elif name == "PhaseShift":
            phi = float(params[0])
            return np.diag([1.0 + 0j, np.exp(1j * phi)])
        else:
            raise GpuError(f"Unknown single-qubit gate: {name!r}")

    def _gate_matrix_2q(self, name: str) -> "np.ndarray":
        if name == "CNOT":
            return np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
        elif name == "CZ":
            return np.diag([1.0 + 0j, 1, 1, -1])
        elif name == "SWAP":
            return np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
        else:
            raise GpuError(f"Unknown two-qubit gate: {name!r}")

    def apply_gate(self, name: str, qubits, params=()):
        """Apply a gate. qubits: int (single) or [ctrl, tgt] (two-qubit)."""
        xp = self._backend.xp
        n = self.n_qubits

        if isinstance(qubits, int):
            qubits = [qubits]

        if len(qubits) == 1:
            mat_np = self._gate_matrix(name, params)
            mat = xp.array(mat_np, dtype=self._dtype)
            q = qubits[0]
            sv = self._sv.reshape([2] * n)
            sv = xp.tensordot(mat, sv, axes=[[1], [q]])
            sv = xp.moveaxis(sv, 0, q)
            self._sv = sv.ravel()
        elif len(qubits) == 2:
            mat_np = self._gate_matrix_2q(name)
            mat = xp.array(mat_np, dtype=self._dtype).reshape(2, 2, 2, 2)
            q0, q1 = qubits[0], qubits[1]
            sv = self._sv.reshape([2] * n)
            sv = xp.tensordot(mat, sv, axes=[[2, 3], [q0, q1]])
            # tensordot result: axes [out0, out1, remaining...]
            # need to transpose back to natural qubit order
            remaining = [i for i in range(n) if i != q0 and i != q1]
            target_order = []
            r_idx = 2
            for i in range(n):
                if i == q0:
                    target_order.append(0)
                elif i == q1:
                    target_order.append(1)
                else:
                    target_order.append(r_idx)
                    r_idx += 1
            sv = xp.transpose(sv, target_order)
            self._sv = sv.ravel()
        else:
            raise GpuError(f"Gates on {len(qubits)} qubits not supported")

    def apply_unitary(self, U, qubits: list[int]):
        """Apply an arbitrary unitary matrix to the specified qubits."""
        xp = self._backend.xp
        n = self.n_qubits
        k = len(qubits)
        U_dev = xp.array(np.asarray(U), dtype=self._dtype).reshape([2] * k * 2)
        sv = self._sv.reshape([2] * n)
        sv = xp.tensordot(U_dev, sv, axes=[list(range(k, 2 * k)), qubits])
        # move axes back
        remaining = [i for i in range(n) if i not in qubits]
        target_order = [None] * n
        for new_ax, q in enumerate(qubits):
            target_order[q] = new_ax
        r_idx = k
        for q in remaining:
            target_order[q] = r_idx
            r_idx += 1
        sv = xp.transpose(sv, target_order)
        self._sv = sv.ravel()

    def state(self) -> "np.ndarray":
        """Return statevector as complex128 numpy array (upcast only if internal dtype is complex64)."""
        sv_cpu = self._backend.to_numpy(self._sv)
        return sv_cpu if sv_cpu.dtype == np.complex128 else sv_cpu.astype(np.complex128)

    def probabilities(self, qubits: list[int] | None = None) -> "np.ndarray":
        """Return measurement probabilities, optionally marginalizing over qubits."""
        xp = self._backend.xp
        probs = xp.abs(self._sv) ** 2
        if qubits is None:
            return self._backend.to_numpy(probs)
        n = self.n_qubits
        probs_t = probs.reshape([2] * n)
        axes_to_sum = [i for i in range(n) if i not in qubits]
        for ax in sorted(axes_to_sum, reverse=True):
            probs_t = probs_t.sum(axis=ax)
        # reorder to requested qubit order
        remaining_sorted = sorted(qubits)
        perm = [remaining_sorted.index(q) for q in qubits]
        probs_t = xp.transpose(probs_t, perm)
        return self._backend.to_numpy(probs_t.ravel())

    def norm(self) -> float:
        """Return ‖ψ‖."""
        xp = self._backend.xp
        return float(xp.linalg.norm(self._sv))

    def expectation_z(self, qubit: int) -> float:
        """⟨Z_qubit⟩ = Σ_s sign(s_qubit) |ψ_s|²."""
        xp = self._backend.xp
        n = self.n_qubits
        sv2 = (xp.abs(self._sv) ** 2).reshape([2] * n)
        idx0 = [slice(None)] * n
        idx0[qubit] = 0
        idx1 = [slice(None)] * n
        idx1[qubit] = 1
        p0 = float(xp.sum(sv2[tuple(idx0)]))
        p1 = float(xp.sum(sv2[tuple(idx1)]))
        return p0 - p1

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "backend": self._backend.name,
            "dtype": self.dtype,
            "hilbert_dim": 2 ** self.n_qubits,
            "norm": round(self.norm(), 8),
        }


@dataclass
class GpuLanczos:
    k: int = 6
    max_iter: int = 200
    tol: float = 1e-10
    backend: str = "auto"
    seed: int | None = None
    _backend: GpuBackend = field(init=False, repr=False)

    def __post_init__(self):
        self._backend = GpuBackend(self.backend)

    def solve(self, H) -> "EigenResult":
        """Solve using GPU arrays if available.

        H can be np.ndarray, SparseHamiltonian, or callable.
        On GPU, the matvec runs as GPU matmul (for dense) or
        a CPU→GPU→CPU round-trip (for SparseHamiltonian).
        """
        from .lanczos import EigenResult
        xp = self._backend.xp

        if isinstance(H, np.ndarray):
            # Move dense matrix to GPU
            H_dev = xp.array(H)

            def matvec(v):
                return H_dev @ v

            N = H.shape[0]
        elif hasattr(H, 'n') and hasattr(H, 'matvec'):
            N = H.n
            if self._backend.is_gpu():
                # Try cupy.sparse (cuSPARSE), fall back to dense GPU
                _gpu_matvec = None
                try:
                    import cupyx.scipy.sparse as csp
                    rows = [e[0] for e in H.entries]
                    cols = [e[1] for e in H.entries]
                    vals = [e[2] for e in H.entries]
                    if rows:
                        H_csr = csp.csr_matrix(
                            (cp.array(vals, dtype=complex),
                             (cp.array(rows, dtype=np.int32),
                              cp.array(cols, dtype=np.int32))),
                            shape=(N, N)
                        )
                        _gpu_matvec = lambda v: H_csr @ v
                except (ImportError, AttributeError):
                    pass
                if _gpu_matvec is None:
                    # Fall back: dense GPU matrix if available, else CPU roundtrip
                    if hasattr(H, 'to_dense'):
                        H_dense = cp.array(H.to_dense())
                        _gpu_matvec = lambda v: H_dense @ v
                    else:
                        # Operator only implements matvec — CPU roundtrip (no entries/to_dense)
                        def _cpu_roundtrip(v_gpu, _H=H):
                            v_cpu = cp.asnumpy(v_gpu)
                            w_cpu = _H.matvec(v_cpu)
                            return cp.asarray(w_cpu)
                        _gpu_matvec = _cpu_roundtrip
                matvec = _gpu_matvec
            else:
                matvec = H.matvec
        elif callable(H):
            raise GpuError(
                "GpuLanczos does not support a bare callable. "
                "Use solve_matvec(fn, n) to provide the dimension explicitly."
            )
        else:
            raise GpuError(f"Unsupported H type: {type(H)}")

        # Run Lanczos iteration on device arrays
        rng = np.random.default_rng(self.seed)
        v0_cpu = rng.standard_normal(N).astype(complex)
        v0_cpu /= np.linalg.norm(v0_cpu)
        v0 = xp.array(v0_cpu)

        k = min(self.k, N)
        max_iter = min(self.max_iter, N)

        V_list = []
        alpha = []
        beta = []

        v = v0 / xp.linalg.norm(v0)
        V_list.append(v)
        w = matvec(v)
        a = float(xp.real(xp.dot(xp.conj(v), w)))
        alpha.append(a)
        w = w - a * v

        converged = False
        for j in range(1, max_iter):
            b_val = float(xp.linalg.norm(w))
            if b_val < 1e-14:
                if len(V_list) >= k:
                    break
                w_cpu = rng.standard_normal(N).astype(complex)
                w = xp.array(w_cpu)
                for vi in V_list:
                    w = w - xp.dot(xp.conj(vi), w) * vi
                b_val = float(xp.linalg.norm(w))
                if b_val < 1e-14:
                    break
            v_new = w / b_val
            for vi in V_list:
                v_new = v_new - xp.dot(xp.conj(vi), v_new) * vi
            norm_new = float(xp.linalg.norm(v_new))
            if norm_new < 1e-14:
                break
            v_new = v_new / norm_new
            beta.append(b_val * norm_new)
            V_list.append(v_new)
            w = matvec(v_new)
            a = float(xp.real(xp.dot(xp.conj(v_new), w)))
            alpha.append(a)
            w = w - a * v_new - b_val * v
            v = v_new

            if j >= k:
                T = self._build_tridiagonal(alpha, beta)
                evals_T, evecs_T = np.linalg.eigh(T)
                ritz_res = abs(b_val) * np.abs(evecs_T[-1, :k])
                if np.all(ritz_res < self.tol):
                    converged = True
                    break

        T = self._build_tridiagonal(alpha, beta)
        evals_T, evecs_T = np.linalg.eigh(T)
        k_actual = min(k, len(evals_T))
        evals_k = evals_T[:k_actual]
        evecs_k_np = evecs_T[:, :k_actual]

        # Reconstruct eigenvectors: move back to CPU
        V_cpu = np.column_stack([self._backend.to_numpy(vi) for vi in V_list])
        full_evecs = V_cpu @ evecs_k_np

        residuals = np.zeros(k_actual)
        for i in range(k_actual):
            ev = full_evecs[:, i]
            if isinstance(H, np.ndarray):
                Hev = H @ ev
            elif hasattr(H, 'matvec'):
                Hev = H.matvec(ev)
            else:
                Hev = ev
            residuals[i] = float(np.linalg.norm(Hev - evals_k[i] * ev))

        return EigenResult(
            eigenvalues=evals_k,
            eigenvectors=full_evecs,
            n_iter=len(alpha),
            converged=converged,
            residuals=residuals,
        )

    @staticmethod
    def _build_tridiagonal(alpha, beta) -> np.ndarray:
        m = len(alpha)
        T = np.diag(alpha).astype(float)
        for i, b in enumerate(beta):
            T[i, i + 1] = b
            T[i + 1, i] = b
        return T

    def solve_matvec(self, fn, n: int) -> "EigenResult":
        """Solve with a callable matvec function and explicit dimension n.

        Addresses P2: supports matrix-free operators on GPU.
        """

        class _Wrap:
            def __init__(self_inner):
                self_inner.n = n

            def matvec(self_inner, v):
                return fn(v)

        return self.solve(_Wrap())

    def to_dict(self) -> dict:
        return {"k": self.k, "backend": self._backend.name, "max_iter": self.max_iter}


@dataclass
class GpuBatchedCircuit:
    """Run multiple circuits simultaneously on GPU — fully vectorized, no Python loops over B."""
    n_qubits: int
    n_circuits: int
    backend: str = "auto"
    dtype: str = "complex128"   # "complex128" (fp64) or "complex64" (fp32)
    _backend: GpuBackend = field(init=False, repr=False)
    _sv: Any = field(init=False, repr=False)    # shape (n_circuits, 2**n_qubits)
    _dtype: Any = field(init=False, repr=False) # resolved numpy dtype

    def __post_init__(self):
        if self.n_qubits < 1 or self.n_circuits < 1:
            raise GpuError("n_qubits and n_circuits must be >= 1")
        if self.dtype not in ("complex64", "complex128"):
            raise GpuError(
                f"dtype must be 'complex64' or 'complex128', got {self.dtype!r}"
            )
        self._backend = GpuBackend(self.backend)
        self._dtype = np.complex64 if self.dtype == "complex64" else np.complex128
        self.reset()

    def reset(self):
        xp = self._backend.xp
        sv = xp.zeros((self.n_circuits, 2**self.n_qubits), dtype=self._dtype)
        sv[:, 0] = 1.0
        self._sv = sv

    def _build_rotation_matrices(self, name: str, params_dev) -> Any:
        """Build (B, 2, 2) rotation matrices for per-circuit angles."""
        xp = self._backend.xp
        half = params_dev / 2.0
        c = xp.cos(half)  # (B,)
        s = xp.sin(half)  # (B,)
        z = xp.zeros_like(c)
        if name == "RX":
            row0 = xp.stack([c, -1j * s], axis=-1)    # (B, 2)
            row1 = xp.stack([-1j * s, c], axis=-1)
        elif name == "RY":
            row0 = xp.stack([c, -s], axis=-1)
            row1 = xp.stack([s,  c], axis=-1)
            row0 = row0.astype(complex)
            row1 = row1.astype(complex)
        elif name == "RZ":
            row0 = xp.stack([xp.exp(-1j * half), z], axis=-1)
            row1 = xp.stack([z, xp.exp(1j * half)], axis=-1)
        elif name == "PhaseShift":
            row0 = xp.stack([xp.ones_like(c), z], axis=-1)
            row1 = xp.stack([z, xp.exp(1j * params_dev)], axis=-1)
        else:
            raise GpuError(f"Per-circuit batched rotation not supported for gate {name!r}")
        return xp.stack([row0, row1], axis=1).astype(self._dtype)  # (B, 2, 2)

    def _apply_1q_same(self, mat_2x2, qubit: int):
        """Apply same (2,2) matrix to qubit in all B circuits — vectorized."""
        xp = self._backend.xp
        B, n = self.n_circuits, self.n_qubits
        q = qubit
        sv = self._sv.reshape([B] + [2]*n)
        result = xp.tensordot(mat_2x2, sv, axes=[[1], [q + 1]])
        result = xp.moveaxis(result, 0, q + 1)
        self._sv = result.reshape(B, 2**n)

    def _apply_1q_batched(self, mat_B22, qubit: int):
        """Apply per-circuit (B,2,2) matrices to qubit — vectorized."""
        xp = self._backend.xp
        B, n = self.n_circuits, self.n_qubits
        q = qubit
        sv = self._sv.reshape([B] + [2]*n)      # (B, 2, ..., 2)
        sv = xp.moveaxis(sv, q + 1, -1)          # qubit axis to last
        sv = sv.reshape(B, -1, 2)                 # (B, D, 2)
        sv = xp.matmul(mat_B22, sv.swapaxes(-1, -2)).swapaxes(-1, -2)  # (B, D, 2)
        sv = sv.reshape([B] + [2]*n)
        sv = xp.moveaxis(sv, -1, q + 1)
        self._sv = sv.reshape(B, 2**n)

    def _apply_2q_same(self, mat_4x4, q0: int, q1: int):
        """Apply same (4,4) two-qubit matrix to (q0,q1) in all B circuits — vectorized."""
        xp = self._backend.xp
        B, n = self.n_circuits, self.n_qubits
        mat = xp.array(mat_4x4, dtype=self._dtype).reshape(2, 2, 2, 2)
        sv = self._sv.reshape([B] + [2]*n)
        result = xp.tensordot(mat, sv, axes=[[2, 3], [q0 + 1, q1 + 1]])
        # result: (out0, out1, B, remaining_axes...)
        result = xp.moveaxis(result, 2, 0)  # → (B, out0, out1, remaining...)
        remaining = sorted(i for i in range(n) if i != q0 and i != q1)
        cur = [q0, q1] + remaining
        perm_q = [cur.index(i) for i in range(n)]
        perm = [0] + [p + 1 for p in perm_q]
        result = xp.transpose(result, perm)
        self._sv = result.reshape(B, 2**n)

    def apply_gate(self, name: str, qubit: int, params=()):
        """Apply gate to all B circuits.

        params: () for non-rotation gates, scalar/1-element for same angle,
                or array of length n_circuits for per-circuit angles.
        """
        xp = self._backend.xp
        B = self.n_circuits

        # Determine per-circuit vs broadcast
        if hasattr(params, '__len__'):
            n_p = len(params)
        else:
            n_p = -1  # bare scalar

        is_per_circuit = (n_p == B and B > 1)

        if is_per_circuit:
            params_dev = xp.asarray(params, dtype=float)
            mat = self._build_rotation_matrices(name, params_dev)  # (B, 2, 2)
            self._apply_1q_batched(mat, qubit)
        else:
            # Normalise scalar params to tuple for GpuStatevector._gate_matrix
            if n_p == -1:
                scalar = (float(params),)
            elif n_p == 1:
                scalar = (float(params[0]),)
            else:
                scalar = tuple(params)
            sv_tmp = GpuStatevector.__new__(GpuStatevector)
            sv_tmp.n_qubits = self.n_qubits
            sv_tmp._backend = self._backend
            mat_np = sv_tmp._gate_matrix(name, scalar)
            mat_dev = xp.array(mat_np, dtype=self._dtype)
            self._apply_1q_same(mat_dev, qubit)

    def apply_gate_all(self, name: str, qubit: int, params=()):
        """Alias for apply_gate — same gate applied to all circuits."""
        self.apply_gate(name, qubit, params)

    def apply_cnot(self, control: int, target: int):
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex)
        self._apply_2q_same(cnot, control, target)

    def apply_cz(self, q0: int, q1: int):
        cz = np.diag([1.0+0j, 1, 1, -1])
        self._apply_2q_same(cz, q0, q1)

    def apply_swap(self, q0: int, q1: int):
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        self._apply_2q_same(swap, q0, q1)

    def probabilities(self) -> "np.ndarray":
        """Return probability arrays for all circuits. Shape (n_circuits, 2**n_qubits)."""
        xp = self._backend.xp
        probs = xp.abs(self._sv)**2
        return self._backend.to_numpy(probs)  # (B, 2**n)

    def expectation_z(self, qubit: int) -> "np.ndarray":
        """Return ⟨Z_qubit⟩ for all circuits. Shape (n_circuits,)."""
        xp = self._backend.xp
        B, n = self.n_circuits, self.n_qubits
        probs = (xp.abs(self._sv)**2).reshape([B] + [2]*n)
        sum_axes = tuple(i + 1 for i in range(n) if i != qubit)
        p01 = probs.sum(axis=sum_axes)   # (B, 2)
        expvals = p01[:, 0] - p01[:, 1]  # (B,)
        return self._backend.to_numpy(expvals)

    def norms(self) -> "np.ndarray":
        """Return ||psi_b|| for each circuit b. Shape (B,)."""
        xp = self._backend.xp
        norms = xp.sqrt(xp.sum(xp.abs(self._sv)**2, axis=1))
        return self._backend.to_numpy(norms)

    def states(self) -> "np.ndarray":
        """Return all B statevectors as complex128 numpy array (upcast only if internal dtype is complex64)."""
        sv_cpu = self._backend.to_numpy(self._sv)
        return sv_cpu if sv_cpu.dtype == np.complex128 else sv_cpu.astype(np.complex128)

    def to_dict(self) -> dict:
        return {"n_qubits": self.n_qubits, "n_circuits": self.n_circuits,
                "backend": self._backend.name, "dtype": self.dtype}


def gpu_info() -> dict:
    """Return GPU availability and device information."""
    info = {
        "cupy_available": _HAS_CUPY,
        "numpy_version": np.__version__ if _HAS_NUMPY else None,
    }
    if _HAS_CUPY:
        try:
            info["cupy_version"] = cp.__version__
            info["n_devices"] = cp.cuda.runtime.getDeviceCount()
            device = cp.cuda.Device(0)
            props = cp.cuda.runtime.getDeviceProperties(device.id)
            info["device_name"] = (
                props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
            )
            mem = cp.cuda.Device(0).mem_info
            info["free_memory_gb"] = round(mem[0] / 1e9, 2)
            info["total_memory_gb"] = round(mem[1] / 1e9, 2)
        except Exception as e:
            info["error"] = str(e)
    return info


def to_gpu(arr: "np.ndarray"):
    """Move a numpy array to GPU. Returns cupy array if CuPy available, else numpy."""
    if _HAS_CUPY:
        return cp.asarray(arr)
    return np.asarray(arr)


def to_cpu(arr) -> "np.ndarray":
    """Move an array to CPU numpy."""
    if _HAS_CUPY and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)
