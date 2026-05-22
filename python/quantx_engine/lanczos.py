"""Lanczos and Arnoldi sparse eigensolver for large quantum Hamiltonians.

Fills the gap for systems too large for dense O(n³) Jacobi diagonalization.
The Lanczos algorithm reduces an N×N symmetric matrix to a k×k tridiagonal
form in k matrix-vector products, making it practical for quantum Hamiltonians
with dim = 2^n where n is the system size.

References
----------
.. [1] Lanczos, C. (1950). An iteration method for the solution of the
       eigenvalue problem of linear differential and integral operators.
       *J. Res. Natl. Bur. Stand.* 45, 255–282.
       https://doi.org/10.6028/jres.045.026

.. [2] Cullum, J. K. & Willoughby, R. A. (1985). *Lanczos Algorithms for
       Large Symmetric Eigenvalue Computations*, Vol. 1.
       Birkhäuser. ISBN 978-0-8176-3058-4.

.. [3] Paige, C. C. (1971). The computation of eigenvalues and eigenvectors
       of very large sparse matrices. PhD thesis, University of London.

Physical validation
-------------------
The 4-site antiferromagnetic Heisenberg chain (H = ½ Σ σ_i·σ_{i+1}, open BC)
has ground-state energy E₀ = −3.2321 with our normalization, corresponding to
E₀/J = −1.6160254 in the standard S = σ/2 convention.
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

    @staticmethod
    def from_pauli_terms(
        n_sites: int,
        terms: "list[tuple[float | complex, str, list[int]]]",
    ) -> "SparseHamiltonian":
        """Build a SparseHamiltonian from a list of Pauli-string terms.

        Each term is a ``(coefficient, ops, sites)`` triple where:

        * ``coefficient`` — real or complex scalar prefactor.
        * ``ops`` — string of single-character Pauli labels, one per site
          (e.g. ``"ZZ"``, ``"XY"``, ``"Z"``).  Supported: ``'I'``, ``'X'``,
          ``'Y'``, ``'Z'``.
        * ``sites`` — list of qubit indices, same length as ``ops``.

        The Hamiltonian acts on a 2ⁿ-dimensional Hilbert space with the
        computational basis ordered as :math:`|s_0, s_1, \\ldots, s_{n-1}\\rangle`
        where :math:`s_i = (\\text{state} >> i) \\& 1`.

        Parameters
        ----------
        n_sites:
            Number of qubits.
        terms:
            List of (coefficient, ops, sites) tuples.

        Returns
        -------
        SparseHamiltonian

        Examples
        --------
        Reproduce a 3-site ZZ + transverse-field Ising model::

            H = SparseHamiltonian.from_pauli_terms(3, [
                (-1.0, "ZZ", [0, 1]),
                (-1.0, "ZZ", [1, 2]),
                (-0.5, "X",  [0]),
                (-0.5, "X",  [1]),
                (-0.5, "X",  [2]),
            ])
        """
        if n_sites < 1:
            raise LanczosError("n_sites must be >= 1")
        dim = 1 << n_sites
        entries = []
        for coeff, ops, sites in terms:
            if len(ops) != len(sites):
                raise LanczosError(
                    f"ops length ({len(ops)}) must match sites length ({len(sites)})"
                )
            for site in sites:
                if site < 0 or site >= n_sites:
                    raise LanczosError(
                        f"Site index {site} out of range [0, {n_sites - 1}]"
                    )
            if len(sites) != len(set(sites)):
                raise LanczosError(
                    f"Duplicate site indices in term: {sites}. "
                    "Each site may appear at most once per term."
                )
            for op in ops:
                if op not in ("I", "X", "Y", "Z"):
                    raise LanczosError(
                        f"Unknown Pauli operator '{op}'. Choose from: I, X, Y, Z"
                    )
            c = complex(coeff)   # avoid shadowing the loop variable
            for i in range(dim):
                j = i
                amplitude = c
                for op, site in zip(ops, sites):
                    bit = (i >> site) & 1  # bit value in original ket |i⟩
                    if op == "I":
                        pass
                    elif op == "X":
                        j ^= (1 << site)
                    elif op == "Y":
                        j ^= (1 << site)
                        amplitude *= 1j if bit == 0 else -1j
                    elif op == "Z":
                        amplitude *= 1 - 2 * bit   # +1 for |0⟩, -1 for |1⟩
                entries.append((j, i, amplitude))
        return SparseHamiltonian(n=dim, entries=entries)

    @staticmethod
    def xxz_chain(
        n_sites: int, J: float = 1.0, delta: float = 1.0, hz: float = 0.0
    ) -> "SparseHamiltonian":
        """1D XXZ spin chain (open BC).

        .. math::

            H = J \\sum_{i=0}^{n-2} \\left[
                \\frac{1}{2}(\\sigma^+_i \\sigma^-_{i+1} + \\text{h.c.})
                + \\Delta\\, Z_i Z_{i+1}
            \\right]
            + h_z \\sum_i Z_i

        where :math:`\\sigma^+ = \\begin{pmatrix}0&1\\\\0&0\\end{pmatrix}`,
        ``J`` is the overall exchange scale, and ``delta`` (Δ) is the
        anisotropy ratio.

        Special cases:

        * ``delta=1``:  isotropic Heisenberg antiferromagnet;
          ``xxz_chain(n, J=J, delta=1)`` equals ``heisenberg_1d(n, J=J)``
        * ``delta=0``:  XX model (free-fermion), no ZZ coupling
        * ``delta→∞``:  classical Ising limit

        Parameters
        ----------
        n_sites:
            Number of spins.  Must be >= 1.
        J:
            Overall exchange coupling (energy scale).
        delta:
            Anisotropy Δ = J_z / J_xy.
        hz:
            Uniform longitudinal field.
        """
        if n_sites < 1:
            raise LanczosError("n_sites must be >= 1")
        dim = 1 << n_sites
        entries = []
        for state in range(dim):
            for site in range(n_sites - 1):
                nxt = site + 1
                si = (state >> site) & 1
                sj = (state >> nxt) & 1
                zi = 1 - 2 * si
                zj = 1 - 2 * sj
                # ZZ term: J * delta * ZiZj
                if delta != 0.0:
                    entries.append((state, state, complex(J * delta * zi * zj)))
                # XY flip: J * 0.5 * (σ+σ- + h.c.)
                if si != sj:
                    flipped = state ^ (1 << site) ^ (1 << nxt)
                    entries.append((state, flipped, complex(J * 0.5)))
            # hz*Z term
            if hz != 0.0:
                for site in range(n_sites):
                    si = (state >> site) & 1
                    entries.append((state, state, complex(hz * (1 - 2 * si))))
        return SparseHamiltonian(n=dim, entries=entries)

    @staticmethod
    def heisenberg_2d(
        Lx: int, Ly: int, J: float = 1.0
    ) -> "SparseHamiltonian":
        """2D Heisenberg model on an Lx × Ly rectangular lattice (open BC).

        .. math::

            H = J \\sum_{\\langle ij \\rangle} \\left[
                Z_i Z_j + \\frac{1}{2}(\\sigma^+_i \\sigma^-_j + \\text{h.c.})
            \\right]

        Sites are labelled ``(x, y) → x + y·Lx``.  Bonds connect
        ``(x, y)–(x+1, y)`` (horizontal) and ``(x, y)–(x, y+1)`` (vertical).

        Parameters
        ----------
        Lx, Ly:
            Lattice dimensions.  Total sites = Lx * Ly ≤ 20 (dense backend).
        J:
            Exchange coupling.
        """
        if Lx < 1 or Ly < 1:
            raise LanczosError(
                f"Lx and Ly must each be >= 1, got Lx={Lx}, Ly={Ly}"
            )
        n_sites = Lx * Ly
        dim = 1 << n_sites

        # Collect all bonds (pairs of site indices)
        bonds: list[tuple[int, int]] = []
        for y in range(Ly):
            for x in range(Lx - 1):
                bonds.append((x + y * Lx, x + 1 + y * Lx))   # horizontal
        for y in range(Ly - 1):
            for x in range(Lx):
                bonds.append((x + y * Lx, x + (y + 1) * Lx))  # vertical

        entries = []
        for state in range(dim):
            for s_i, s_j in bonds:
                si = (state >> s_i) & 1
                sj = (state >> s_j) & 1
                zi = 1 - 2 * si
                zj = 1 - 2 * sj
                entries.append((state, state, complex(J * zi * zj)))
                if si != sj:
                    flipped = state ^ (1 << s_i) ^ (1 << s_j)
                    entries.append((state, flipped, complex(J * 0.5)))
        return SparseHamiltonian(n=dim, entries=entries)

    @staticmethod
    def bose_hubbard(
        L: int, t: float = 1.0, U: float = 4.0, max_occ: int = 4
    ) -> "SparseHamiltonian":
        """Bose-Hubbard model on a 1D chain (open BC).

        .. math::

            H = -t \\sum_i (a^\\dagger_i a_{i+1} + \\text{h.c.})
                + \\frac{U}{2} \\sum_i n_i(n_i - 1)

        Fock basis: each site holds 0 … max_occ bosons.  State index:

        .. math::

            \\text{idx} = \\sum_{i=0}^{L-1} n_i \\cdot M^i, \\quad M = \\text{max\\_occ} + 1

        Hilbert space dimension: ``(max_occ + 1)^L``.

        Parameters
        ----------
        L:
            Number of sites.
        t:
            Hopping amplitude.
        U:
            On-site interaction strength.
        max_occ:
            Maximum occupation per site (truncation).
        """
        if L < 1:
            raise LanczosError("L must be >= 1")
        if max_occ < 1:
            raise LanczosError("max_occ must be >= 1")
        M = max_occ + 1
        dim = M ** L
        powers = [M ** s for s in range(L)]   # precompute M^s

        def get_occ(state: int, site: int) -> int:
            return (state // powers[site]) % M

        def set_occ(state: int, site: int, n: int, old_n: int) -> int:
            return state + (n - old_n) * powers[site]

        entries = []
        for state in range(dim):
            for site in range(L - 1):
                ni = get_occ(state, site)
                nj = get_occ(state, site + 1)
                # a†_i a_{i+1}: destroy at site+1, create at site
                if nj > 0 and ni < max_occ:
                    s2 = set_occ(set_occ(state, site, ni + 1, ni), site + 1, nj - 1, nj)
                    amp = -t * ((ni + 1) * nj) ** 0.5
                    entries.append((s2, state, complex(amp)))
                # a†_{i+1} a_i (h.c.): destroy at site, create at site+1
                if ni > 0 and nj < max_occ:
                    s2 = set_occ(set_occ(state, site, ni - 1, ni), site + 1, nj + 1, nj)
                    amp = -t * (ni * (nj + 1)) ** 0.5
                    entries.append((s2, state, complex(amp)))
            # on-site interaction U/2 * n*(n-1)
            for site in range(L):
                ni = get_occ(state, site)
                if ni >= 2:
                    entries.append((state, state, complex(U / 2 * ni * (ni - 1))))
        return SparseHamiltonian(n=dim, entries=entries)

    @staticmethod
    def hubbard_1d(L: int, t: float = 1.0, U: float = 4.0) -> "SparseHamiltonian":
        """Single-band Fermi-Hubbard model on a 1D chain (open BC).

        .. math::

            H = -t \\sum_{i,\\sigma} (c^\\dagger_{i,\\sigma} c_{i+1,\\sigma} + \\text{h.c.})
                + U \\sum_i n_{i,\\uparrow} n_{i,\\downarrow}

        Encoding: spin-up occupations in the lower L bits (bit ``i`` = site ``i``
        spin-up), spin-down occupations in the upper L bits.
        State index: ``up_bits | (down_bits << L)``.
        Hilbert space dimension: ``4^L = 2^{2L}``.

        JW ordering — ``(\\uparrow_0, \\uparrow_1, \\ldots, \\downarrow_0, \\downarrow_1,
        \\ldots)`` — gives sign ``+1`` for nearest-neighbour intra-sector hopping.

        Parameters
        ----------
        L:
            Number of sites.
        t:
            Hopping amplitude.
        U:
            On-site Hubbard interaction.
        """
        if L < 1:
            raise LanczosError("L must be >= 1")
        up_mask = (1 << L) - 1
        dim = 1 << (2 * L)
        entries = []

        for state in range(dim):
            up = state & up_mask
            dn = state >> L

            # Spin-up hopping (JW sign = +1 for adjacent sites in up sector)
            for i in range(L - 1):
                ui  = (up >> i) & 1
                ui1 = (up >> (i + 1)) & 1
                if ui != ui1:  # one occupied, one empty → can hop
                    new_up = up ^ (1 << i) ^ (1 << (i + 1))
                    entries.append((new_up | (dn << L), state, complex(-t)))

            # Spin-down hopping (JW sign = +1 for adjacent sites in dn sector)
            for i in range(L - 1):
                di  = (dn >> i) & 1
                di1 = (dn >> (i + 1)) & 1
                if di != di1:
                    new_dn = dn ^ (1 << i) ^ (1 << (i + 1))
                    entries.append((up | (new_dn << L), state, complex(-t)))

            # Hubbard on-site interaction
            for i in range(L):
                if ((up >> i) & 1) and ((dn >> i) & 1):
                    entries.append((state, state, complex(U)))

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
