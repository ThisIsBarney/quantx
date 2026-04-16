"""Lanczos and Arnoldi sparse eigensolver for large quantum Hamiltonians.

Fills the gap for systems too large for dense O(n³) Jacobi diagonalization.
"""
from __future__ import annotations

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

from dataclasses import dataclass, field

__all__ = ["LanczosError", "LanczosSolver", "ArnoldiSolver", "SparseHamiltonian", "EigenResult"]


class LanczosError(ValueError):
    """Raised on invalid input or solver failure."""


@dataclass
class EigenResult:
    eigenvalues: "np.ndarray"   # shape (k,), sorted ascending
    eigenvectors: "np.ndarray"  # shape (N, k), columns are eigenvectors
    n_iter: int                 # number of Lanczos iterations used
    converged: bool
    residuals: "np.ndarray"     # shape (k,), ||Av - lambda*v|| for each eigenpair

    def to_dict(self) -> dict:
        return {
            "eigenvalues": self.eigenvalues.tolist(),
            "n_iter": self.n_iter,
            "converged": self.converged,
            "residuals": self.residuals.tolist(),
        }

    def ground_state_energy(self) -> float:
        return float(self.eigenvalues[0])

    def ground_state_vector(self) -> "np.ndarray":
        return self.eigenvectors[:, 0]

    def spectral_gap(self) -> float:
        if len(self.eigenvalues) < 2:
            raise LanczosError("Need k >= 2 eigenvalues to compute spectral gap")
        return float(self.eigenvalues[1] - self.eigenvalues[0])


@dataclass
class SparseHamiltonian:
    n: int                                          # dimension of Hilbert space
    entries: list  # list of (row, col, value) triples — upper+lower or full

    def __post_init__(self):
        if self.n < 1:
            raise LanczosError("n must be >= 1")

    def matvec(self, v: "np.ndarray") -> "np.ndarray":
        """Sparse matrix-vector product Hv."""
        result = np.zeros(self.n, dtype=complex)
        for row, col, val in self.entries:
            result[row] += val * v[col]
        return result

    def to_dense(self) -> "np.ndarray":
        """Convert to dense matrix (only for small n)."""
        H = np.zeros((self.n, self.n), dtype=complex)
        for row, col, val in self.entries:
            H[row, col] += val
        return H

    def diagonal(self) -> "np.ndarray":
        """Return diagonal elements."""
        d = np.zeros(self.n, dtype=complex)
        for row, col, val in self.entries:
            if row == col:
                d[row] += val
        return d

    @property
    def nnz(self) -> int:
        return len(self.entries)

    def to_dict(self) -> dict:
        return {"n": self.n, "nnz": self.nnz}

    @staticmethod
    def ising_1d(n_sites: int, J: float = 1.0, h: float = 0.5) -> "SparseHamiltonian":
        """1D transverse-field Ising H = -J sum ZiZi+1 - h sum Xi (periodic BC).

        Hilbert space: 2^n_sites, basis states |s0 s1 ... s_{n-1}>, si in {0,1}.
        State index: i = s0 + 2*s1 + 4*s2 + ...
        """
        dim = 1 << n_sites
        entries = []
        for state in range(dim):
            # ZZ terms: diagonal
            for site in range(n_sites):
                next_site = (site + 1) % n_sites
                si = (state >> site) & 1
                sj = (state >> next_site) & 1
                zi = 1 - 2 * si   # Z eigenvalue: +1 for |0>, -1 for |1>
                zj = 1 - 2 * sj
                entries.append((state, state, complex(-J * zi * zj)))
            # X terms: off-diagonal (flip bit)
            for site in range(n_sites):
                flipped = state ^ (1 << site)
                entries.append((state, flipped, complex(-h)))
        return SparseHamiltonian(n=dim, entries=entries)

    @staticmethod
    def heisenberg_1d(n_sites: int, J: float = 1.0, hz: float = 0.0) -> "SparseHamiltonian":
        """1D Heisenberg H = J sum (Xi Xi+1 + Yi Yi+1 + Zi Zi+1) + hz sum Zi (open BC).

        XX+YY = 2(S+S- + S-S+): flips adjacent antiparallel spins.
        ZZ: diagonal.
        For spin-1/2: S+|0>=|1>, S+|1>=0; S-|1>=|0>, S-|0>=0 (|0>=up, |1>=down).
        Xi Xi+1 + Yi Yi+1 = 2(S+i S-i+1 + S-i S+i+1) where S+/- = sigma+/-/2.
        Matrix element for antiparallel flip: J * 0.5.
        """
        dim = 1 << n_sites
        entries = []
        for state in range(dim):
            # ZZ and hz*Z terms
            for site in range(n_sites - 1):
                next_site = site + 1
                si = (state >> site) & 1
                sj = (state >> next_site) & 1
                zi = 1 - 2 * si
                zj = 1 - 2 * sj
                entries.append((state, state, complex(J * zi * zj)))
            for site in range(n_sites):
                si = (state >> site) & 1
                zi = 1 - 2 * si
                entries.append((state, state, complex(hz * zi)))
            # XX+YY: flip antiparallel adjacent pairs
            # Matrix element = J * 0.5 for each direction
            for site in range(n_sites - 1):
                next_site = site + 1
                si = (state >> site) & 1
                sj = (state >> next_site) & 1
                if si != sj:  # antiparallel: can flip
                    flipped = state ^ (1 << site) ^ (1 << next_site)
                    entries.append((state, flipped, complex(J * 0.5)))
        return SparseHamiltonian(n=dim, entries=entries)


def _build_tridiagonal(alpha: list, beta: list) -> "np.ndarray":
    """Build symmetric tridiagonal matrix from diagonal and sub-diagonal."""
    m = len(alpha)
    T = np.diag(np.array(alpha, dtype=float))
    for i, b in enumerate(beta):
        T[i, i + 1] = b
        T[i + 1, i] = b
    return T


@dataclass
class LanczosSolver:
    """Symmetric (Hermitian) Lanczos algorithm for finding k lowest eigenvalues."""
    k: int = 6            # number of eigenvalues to compute
    max_iter: int = 200   # maximum Lanczos iterations
    tol: float = 1e-10    # convergence tolerance on residuals
    seed: "int | None" = None

    def __post_init__(self):
        if self.k < 1:
            raise LanczosError("k must be >= 1")
        if self.max_iter < self.k:
            raise LanczosError("max_iter must be >= k")
        if self.tol <= 0:
            raise LanczosError("tol must be > 0")

    def solve(self, H, v0: "np.ndarray | None" = None) -> EigenResult:
        """Find k lowest eigenpairs of Hermitian H.

        H can be:
        - np.ndarray (dense, shape (N, N))
        - SparseHamiltonian
        - object with .n and .matvec attributes

        Algorithm: Lanczos with full re-orthogonalization (modified Gram-Schmidt).
        """
        if isinstance(H, np.ndarray):
            N = H.shape[0]
            matvec = lambda v: H @ v
        elif isinstance(H, SparseHamiltonian):
            N = H.n
            matvec = H.matvec
        elif callable(H):
            raise LanczosError(
                "For callable H, pass n= explicitly; use solve_matvec()"
            )
        elif hasattr(H, "n") and hasattr(H, "matvec"):
            N = H.n
            matvec = H.matvec
        else:
            raise LanczosError(f"Unsupported H type: {type(H)}")

        k = min(self.k, N)
        max_iter = min(self.max_iter, N)

        rng = np.random.default_rng(self.seed)
        if v0 is None:
            v0 = rng.standard_normal(N).astype(complex)
            v0 /= np.linalg.norm(v0)
        else:
            v0 = np.array(v0, dtype=complex)
            norm_v0 = np.linalg.norm(v0)
            if norm_v0 < 1e-14:
                raise LanczosError("Initial vector v0 has near-zero norm")
            v0 /= norm_v0

        # Lanczos iteration
        V = []      # Krylov basis vectors
        alpha = []  # diagonal of tridiagonal T
        beta = []   # sub/superdiagonal of T

        v = v0.copy()
        V.append(v)
        w = matvec(v)
        a = float(np.real(np.dot(v.conj(), w)))
        alpha.append(a)
        w = w - a * v

        converged = False
        for j in range(1, max_iter):
            b = float(np.linalg.norm(w))
            if b < 1e-14:
                # Invariant subspace found — restart with a random orthogonal vector
                # so we can find eigenvalues in other invariant subspaces
                if len(V) >= k:
                    break  # already have enough basis vectors
                w = rng.standard_normal(N).astype(complex)
                for vi in V:
                    w = w - np.dot(vi.conj(), w) * vi
                b = float(np.linalg.norm(w))
                if b < 1e-14:
                    break  # truly exhausted the space
            v_new = w / b

            # Full re-orthogonalization (Modified Gram-Schmidt)
            for vi in V:
                v_new = v_new - np.dot(vi.conj(), v_new) * vi
            norm = float(np.linalg.norm(v_new))
            if norm < 1e-14:
                V.append(v_new)
                alpha.append(0.0)
                break
            v_new /= norm
            # Effective recurrence coeff after re-orthogonalization
            beta.append(b * norm)
            V.append(v_new)

            w = matvec(v_new)
            a = float(np.real(np.dot(v_new.conj(), w)))
            alpha.append(a)
            w = w - a * v_new - b * v
            v = v_new

            # Check convergence: solve small tridiagonal eigenproblem
            if j >= k:
                T = _build_tridiagonal(alpha, beta)
                evals, evecs = np.linalg.eigh(T)
                # Ritz residuals: b_j * |last component of Ritz vector|
                ritz_residuals = abs(b) * np.abs(evecs[-1, :k])
                if np.all(ritz_residuals < self.tol):
                    converged = True
                    break

        # Final solve on tridiagonal
        T = _build_tridiagonal(alpha, beta)
        evals, evecs = np.linalg.eigh(T)

        # Take k lowest
        k_actual = min(k, len(evals))
        evals_k = evals[:k_actual]
        evecs_k = evecs[:, :k_actual]

        # Reconstruct full eigenvectors: v = V @ evecs_k
        V_mat = np.column_stack(V)  # (N, m) where m = len(V)
        full_evecs = V_mat @ evecs_k  # (N, k_actual)

        # Compute residuals ||Hv - lambda*v||
        residuals = np.zeros(k_actual)
        for i in range(k_actual):
            ev = full_evecs[:, i]
            Hev = matvec(ev)
            residuals[i] = float(np.linalg.norm(Hev - evals_k[i] * ev))

        return EigenResult(
            eigenvalues=evals_k,
            eigenvectors=full_evecs,
            n_iter=len(alpha),
            converged=converged,
            residuals=residuals,
        )

    def solve_matvec(self, matvec, n: int, v0: "np.ndarray | None" = None) -> EigenResult:
        """Solve using a callable matvec function and explicit dimension n."""

        class _Wrap:
            def __init__(self_inner):
                self_inner.n = n

            def matvec(self_inner, v):
                return matvec(v)

        return self.solve(_Wrap(), v0=v0)


@dataclass
class ArnoldiSolver:
    """Arnoldi algorithm for non-Hermitian matrices."""
    k: int = 6
    max_iter: int = 100
    tol: float = 1e-8
    seed: "int | None" = None

    def __post_init__(self):
        if self.k < 1:
            raise LanczosError("k must be >= 1")
        if self.max_iter < self.k:
            raise LanczosError("max_iter must be >= k")
        if self.tol <= 0:
            raise LanczosError("tol must be > 0")

    def solve(self, H, v0: "np.ndarray | None" = None) -> EigenResult:
        """Find k eigenvalues of smallest real part.

        H can be:
        - np.ndarray (dense, shape (N, N))
        - SparseHamiltonian
        """
        if isinstance(H, np.ndarray):
            N = H.shape[0]
            matvec = lambda v: H @ v
        elif isinstance(H, SparseHamiltonian):
            N = H.n
            matvec = H.matvec
        elif hasattr(H, "n") and hasattr(H, "matvec"):
            N = H.n
            matvec = H.matvec
        else:
            raise LanczosError(f"Unsupported H type: {type(H)}")

        m = min(self.max_iter, N)
        k = min(self.k, N)
        rng = np.random.default_rng(self.seed)

        if v0 is None:
            v0 = rng.standard_normal(N).astype(complex)
            v0 /= np.linalg.norm(v0)
        else:
            v0 = np.array(v0, dtype=complex)
            norm_v0 = np.linalg.norm(v0)
            if norm_v0 > 1e-14:
                v0 /= norm_v0

        Q = []                                      # Arnoldi basis
        H_mat = np.zeros((m + 1, m), dtype=complex)  # upper Hessenberg

        _norm_v0 = np.linalg.norm(v0)
        if _norm_v0 < 1e-14:
            raise LanczosError("Initial vector v0 has near-zero norm")
        q = v0 / _norm_v0
        Q.append(q)

        converged = False
        j_final = m
        for j in range(m):
            z = matvec(Q[j])
            for i in range(j + 1):
                H_mat[i, j] = np.dot(Q[i].conj(), z)
                z = z - H_mat[i, j] * Q[i]
            H_mat[j + 1, j] = np.linalg.norm(z)
            if H_mat[j + 1, j] < 1e-14:
                j_final = j + 1
                break
            Q.append(z / H_mat[j + 1, j])

            if j + 1 >= k:
                Hk = H_mat[: j + 1, : j + 1]
                evals_check = np.linalg.eigvals(Hk)
                evals_check = np.sort(evals_check.real) + 0j
                if j + 1 >= min(m, 2 * k):
                    converged = True
                    break
        else:
            j_final = len(Q)

        j_final = len(Q)
        Hk = H_mat[:j_final, :j_final]
        evals, evecs = np.linalg.eig(Hk)

        # Sort by real part, take k lowest
        idx = np.argsort(evals.real)[:k]
        evals_k = evals[idx]
        evecs_k = evecs[:, idx]  # (j_final, k)

        Q_mat = np.column_stack(Q)  # (N, j_final)
        full_evecs = Q_mat @ evecs_k  # (N, k)

        residuals = np.zeros(k)
        for i in range(k):
            ev = full_evecs[:, i]
            norm = np.linalg.norm(ev)
            if norm > 1e-14:
                ev_n = ev / norm
                Hev = matvec(ev_n)
                residuals[i] = float(np.linalg.norm(Hev - evals_k[i] * ev_n))

        return EigenResult(
            eigenvalues=evals_k,
            eigenvectors=full_evecs,
            n_iter=j_final,
            converged=converged,
            residuals=residuals,
        )
