"""
Tensor Network methods for the QuantX engine.

Implements Matrix Product States (MPS), Matrix Product Operators (MPO),
Time-Evolving Block Decimation (TEBD), and Density Matrix Renormalization
Group (DMRG) in pure Python / NumPy.

Classes:
    MPS          — Matrix Product State representation of N-qubit states
    MPO          — Matrix Product Operator representation of Hamiltonians
    TEBD         — Trotterized time evolution under nearest-neighbor Hamiltonians
    DMRG         — Two-site DMRG ground-state search
    DMRGResult   — Result container returned by DMRG.run()

Usage::

    from quantx_engine.tensor_network import MPS, MPO, TEBD, DMRG

    # Build a 6-qubit |000000⟩ product state
    mps = MPS.product_state([0, 0, 0, 0, 0, 0])

    # Build transverse-field Ising Hamiltonian MPO
    mpo = MPO.ising(n=6, J=1.0, h=0.5)

    # Find ground state with DMRG
    dmrg = DMRG(mpo=mpo, max_chi=16, n_sweeps=10)
    result = dmrg.run()
    print(f"Ground state energy: {result.energy:.6f}")

    # Time-evolve with TEBD
    tebd = TEBD(mps=MPS.product_state([0]*6), max_chi=16)
    history = tebd.evolve_ising(J=1.0, h=0.5, dt=0.01, n_steps=100)

References
----------
.. [1] White, S. R. (1992). Density matrix formulation for quantum
       renormalization groups. *Phys. Rev. Lett.* 69, 2863.
       https://doi.org/10.1103/PhysRevLett.69.2863

.. [2] Vidal, G. (2003). Efficient classical simulation of slightly entangled
       quantum computations. *Phys. Rev. Lett.* 91, 147902.
       https://doi.org/10.1103/PhysRevLett.91.147902

.. [3] Schollwöck, U. (2011). The density-matrix renormalization group in the
       age of matrix product states. *Ann. Phys.* 326, 96–192.
       https://doi.org/10.1016/j.aop.2010.09.012

Physical validation
-------------------
GHZ state norm conservation: after n TEBD steps the state norm stays 1.0 to
within floating-point precision (verified in test_physics_validation_2.py).
DMRG ground-state energy for 6-site TFIM (J=1, h=0.5) converges within 1e-4
of the exact value from numpy.linalg.eigvalsh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

__all__ = [
    "TensorNetworkError",
    "MPS",
    "MPO",
    "TEBD",
    "DMRGResult",
    "DMRG",
]

# ── Pauli matrices (lazy; safe to import without numpy) ──────────────────────

if _HAS_NUMPY:
    _I2 = np.eye(2, dtype=complex)
    _X = np.array([[0, 1], [1, 0]], dtype=complex)
    _Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    _Z = np.diag([1.0, -1.0]).astype(complex)
else:  # pragma: no cover
    _I2 = _X = _Y = _Z = None  # type: ignore[assignment]


# ── Custom exception ───────────────────────────────────────────────────────────

class TensorNetworkError(ValueError):
    """Raised for invalid tensor network inputs or unsupported configurations."""


# ── MPS ───────────────────────────────────────────────────────────────────────

@dataclass
class MPS:
    """Matrix Product State: N-site state as rank-3 tensors.

    Each tensor ``A[i]`` has shape ``(chi_left, d, chi_right)`` where:

    * ``d``         = local Hilbert-space dimension (2 for qubits)
    * ``chi_left``  = 1 for the leftmost site
    * ``chi_right`` = 1 for the rightmost site

    Parameters
    ----------
    tensors:
        List of N numpy arrays with the shapes described above.
    d:
        Local Hilbert-space dimension (default 2).
    """

    tensors: list[np.ndarray]
    d: int = 2

    def __post_init__(self) -> None:
        if not _HAS_NUMPY:  # pragma: no cover
            raise TensorNetworkError("numpy is required for MPS")
        if len(self.tensors) == 0:
            raise TensorNetworkError("MPS must have at least one tensor")
        for i, t in enumerate(self.tensors):
            if t.ndim != 3:
                raise TensorNetworkError(
                    f"Tensor {i} must be rank-3, got shape {t.shape}"
                )
            if t.shape[1] != self.d:
                raise TensorNetworkError(
                    f"Tensor {i} physical dim {t.shape[1]} != d={self.d}"
                )
        if self.tensors[0].shape[0] != 1:
            raise TensorNetworkError(
                f"Left boundary tensor must have chi_left=1, got {self.tensors[0].shape[0]}"
            )
        if self.tensors[-1].shape[2] != 1:
            raise TensorNetworkError(
                f"Right boundary tensor must have chi_right=1, got {self.tensors[-1].shape[2]}"
            )
        for i in range(len(self.tensors) - 1):
            if self.tensors[i].shape[2] != self.tensors[i + 1].shape[0]:
                raise TensorNetworkError(
                    f"Bond dimension mismatch between tensors {i} and {i+1}: "
                    f"{self.tensors[i].shape[2]} != {self.tensors[i+1].shape[0]}"
                )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_sites(self) -> int:
        """Number of sites."""
        return len(self.tensors)

    @property
    def bond_dims(self) -> list[int]:
        """Internal bond dimensions [chi_01, chi_12, ..., chi_{N-2,N-1}]."""
        return [self.tensors[i].shape[2] for i in range(self.n_sites - 1)]

    # ── Norm ──────────────────────────────────────────────────────────────────

    def norm(self) -> float:
        """Compute ⟨ψ|ψ⟩^(1/2) via sequential left-to-right contraction."""
        # env: (chi_l_bra, chi_l_ket)
        env = np.ones((1, 1), dtype=complex)
        for A in self.tensors:
            # env[a,b], A[a,s,c], A*[b,s,d] -> new_env[c,d]
            tmp = np.tensordot(env, A.conj(), axes=([0], [0]))  # (b, s, c) from (a,b),(a,s,c)
            # tmp[b,s,c], A[b,s,d] -> env[c,d]
            env = np.tensordot(tmp, A, axes=([0, 1], [0, 1]))   # (c, d)
        return float(np.sqrt(np.abs(env[0, 0])))

    def normalize(self) -> "MPS":
        """Normalize in-place, return self."""
        n = self.norm()
        if n > 0:
            self.tensors[0] = self.tensors[0] / n
        return self

    # ── Statevector conversion ────────────────────────────────────────────────

    def to_statevector(self) -> np.ndarray:
        """Full statevector via sequential contraction. Only for small N."""
        # result shape: (d^k, chi_r)
        result = self.tensors[0].reshape(self.d, self.tensors[0].shape[2])
        for A in self.tensors[1:]:
            chi_l, d, chi_r = A.shape
            # result: (..., chi_l), A: (chi_l, d, chi_r) -> (..., d, chi_r)
            result = np.tensordot(result, A, axes=([-1], [0]))
            # reshape last two dims into one: (..., d, chi_r) -> (...*d, chi_r)
            new_shape = result.shape[:-2] + (result.shape[-2] * chi_r,)
            # Actually we want (...*d, chi_r) not (..., d*chi_r)
            # result.shape is (prev_dim, d, chi_r), reshape to (prev_dim*d, chi_r)
            result = result.reshape(-1, chi_r)
        return result.reshape(-1)

    # ── Overlap ───────────────────────────────────────────────────────────────

    def overlap(self, other: "MPS") -> complex:
        """⟨self|other⟩ via sequential left-to-right contraction."""
        if self.n_sites != other.n_sites:
            raise TensorNetworkError("MPS must have the same number of sites for overlap")
        env = np.ones((1, 1), dtype=complex)
        for A_bra, A_ket in zip(self.tensors, other.tensors):
            # env[a,b], A_bra*[a,s,c], A_ket[b,s,d] -> new_env[c,d]
            tmp = np.tensordot(env, A_bra.conj(), axes=([0], [0]))  # (b, s, c)
            env = np.tensordot(tmp, A_ket, axes=([0, 1], [0, 1]))   # (c, d)
        return complex(env[0, 0])

    # ── Truncation ────────────────────────────────────────────────────────────

    def truncate(self, max_chi: int) -> "MPS":
        """SVD-compress all bonds to max_chi, right-canonical form."""
        new_tensors = [t.copy() for t in self.tensors]
        N = len(new_tensors)

        for i in range(N - 1, 0, -1):
            A = new_tensors[i]
            chi_l, d, chi_r = A.shape
            M = A.reshape(chi_l, d * chi_r)
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            chi_new = min(max_chi, len(S))
            U = U[:, :chi_new]
            S = S[:chi_new]
            Vt = Vt[:chi_new, :]
            new_tensors[i] = Vt.reshape(chi_new, d, chi_r)
            # Absorb U*S into left neighbor
            # new_tensors[i-1]: (chi_ll, d_prev, chi_l), contract chi_l with U: (chi_l, chi_new)
            prev = new_tensors[i - 1]
            new_tensors[i - 1] = np.tensordot(prev, U * S[np.newaxis, :], axes=([-1], [0]))
            # Result: (chi_ll, d_prev, chi_new) — correct shape

        return MPS(tensors=new_tensors, d=self.d)

    # ── Static constructors ───────────────────────────────────────────────────

    @staticmethod
    def from_statevector(
        psi: np.ndarray, d: int = 2, max_chi: int | None = None
    ) -> "MPS":
        """Left-to-right SVD decomposition of a statevector into MPS."""
        psi = np.asarray(psi, dtype=complex)
        total_dim = psi.size
        N = round(math.log(total_dim, d))
        if d**N != total_dim:
            raise TensorNetworkError(
                f"Statevector size {total_dim} is not a power of d={d}"
            )

        tensors = []
        M = psi.reshape(1, total_dim)
        chi_l = 1

        for i in range(N - 1):
            M = M.reshape(chi_l * d, d ** (N - i - 1))
            U, S, Vt = np.linalg.svd(M, full_matrices=False)
            chi_new = len(S)
            if max_chi is not None:
                chi_new = min(chi_new, max_chi)
                U = U[:, :chi_new]
                S = S[:chi_new]
                Vt = Vt[:chi_new, :]
            tensors.append(U.reshape(chi_l, d, chi_new))
            M = S[:, np.newaxis] * Vt
            chi_l = chi_new

        # Last tensor
        tensors.append(M.reshape(chi_l, d, 1))
        return MPS(tensors=tensors, d=d)

    @staticmethod
    def product_state(bits: list[int], d: int = 2) -> "MPS":
        """Construct |b_0 b_1 ... b_{N-1}⟩ as MPS with chi=1 everywhere."""
        tensors = []
        for b in bits:
            if not 0 <= b < d:
                raise TensorNetworkError(f"Bit value {b} out of range for d={d}")
            vec = np.zeros(d, dtype=complex)
            vec[b] = 1.0
            tensors.append(vec.reshape(1, d, 1))
        return MPS(tensors=tensors, d=d)

    @staticmethod
    def random(n: int, d: int = 2, max_chi: int = 4, seed: int | None = None) -> "MPS":
        """Random MPS with bond dimensions up to max_chi, normalized."""
        rng = np.random.default_rng(seed)
        tensors = []
        for i in range(n):
            chi_l = 1 if i == 0 else min(max_chi, d**i)
            chi_r = 1 if i == n - 1 else min(max_chi, d ** (i + 1))
            A = rng.standard_normal((chi_l, d, chi_r)) + 1j * rng.standard_normal(
                (chi_l, d, chi_r)
            )
            tensors.append(A)
        mps = MPS(tensors=tensors, d=d)
        mps.normalize()
        return mps

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        bond_dims = self.bond_dims
        return {
            "n_sites": self.n_sites,
            "bond_dims": bond_dims,
            "d": self.d,
            "max_bond_dim": max(bond_dims) if bond_dims else 1,
        }


# ── MPO ───────────────────────────────────────────────────────────────────────

@dataclass
class MPO:
    """Matrix Product Operator: N-site operator as rank-4 tensors.

    Each tensor ``W[i]`` has shape ``(D_left, D_right, d, d)`` where
    physical indices are the last two axes.

    Parameters
    ----------
    tensors:
        List of N numpy arrays with shapes as above.
    d:
        Local Hilbert-space dimension (default 2).
    """

    tensors: list[np.ndarray]
    d: int = 2

    @property
    def n_sites(self) -> int:
        return len(self.tensors)

    # ── Apply MPO to MPS ──────────────────────────────────────────────────────

    def apply(self, mps: MPS, max_chi: int | None = None) -> MPS:
        """Apply MPO to MPS giving new MPS; optionally truncate."""
        if self.n_sites != mps.n_sites:
            raise TensorNetworkError("MPO and MPS must have the same number of sites")

        new_tensors = []
        for W, A in zip(self.tensors, mps.tensors):
            # W: (D_l, D_r, d_out, d_in)
            # A: (chi_l, d_in, chi_r)
            # Contract d_in: sum_s W[...,s] * A[...,s,...]
            # Result shape: (D_l, chi_l, d_out, D_r, chi_r)
            # B[D_l, chi_l, d_out, D_r, chi_r] = sum_s W[D_l, D_r, d_out, s] A[chi_l, s, chi_r]
            B = np.tensordot(W, A, axes=([-1], [1]))
            # W: (D_l, D_r, d_out, d_in=s), A: (chi_l, s, chi_r)
            # tensordot on last of W (d_in) and index 1 of A (d_in=s)
            # Result: (D_l, D_r, d_out, chi_l, chi_r)
            D_l, D_r, d_out, chi_l, chi_r = B.shape
            # Reorder to (D_l*chi_l, d_out, D_r*chi_r)
            B = B.transpose(0, 3, 2, 1, 4)  # (D_l, chi_l, d_out, D_r, chi_r)
            B = B.reshape(D_l * chi_l, d_out, D_r * chi_r)
            new_tensors.append(B)

        result = MPS(tensors=new_tensors, d=self.d)
        if max_chi is not None:
            result = result.truncate(max_chi)
        return result

    # ── Expectation value ─────────────────────────────────────────────────────

    def expectation(self, mps: MPS) -> complex:
        """Compute ⟨mps|MPO|mps⟩ via three-layer contraction."""
        if self.n_sites != mps.n_sites:
            raise TensorNetworkError("MPO and MPS must have the same number of sites")

        # env: (chi_bra, D_mpo, chi_ket)
        env = np.ones((1, 1, 1), dtype=complex)

        for W, A in zip(self.tensors, mps.tensors):
            # W: (D_l, D_r, d, d) — W[D_l, D_r, s_out, s_in]
            # A: (chi_l, d, chi_r) — same tensor for bra (conj) and ket
            # env: (a, e, c) = (chi_bra_l, D_l, chi_ket_l)
            #
            # new_env[b, f, d] = sum_{a,e,c,s,t}
            #   env[a,e,c] * A_bra*[a,s,b] * W[e,f,s,t] * A_ket[c,t,d]

            # Step 1: contract env[a,e,c] with A_bra*[a,s,b] -> T1[e,c,s,b]
            T1 = np.tensordot(env, A.conj(), axes=([0], [0]))
            # env: (a,e,c), A.conj(): (a,s,b) -> result: (e,c,s,b)

            # Step 2: contract T1[e,c,s,b] with W[e,f,s,t] -> T2[c,b,f,t]
            # axes: T1 axes (0,2) = (e,s) against W axes (0,2) = (e,s) NO wait
            # W: (D_l, D_r, d_out, d_in) where d_out goes to bra, d_in to ket
            # So W[e, f, s, t] where s is bra physical, t is ket physical
            # Contract: sum_{e,s} T1[e,c,s,b] * W[e,f,s,t]
            # T1 axes (0,2) = (e,s), W axes (0,2) = (e,s)  => correct!
            T2 = np.tensordot(T1, W, axes=([0, 2], [0, 2]))
            # T1: (e,c,s,b), W: (e,f,s,t) -> result: (c,b,f,t)

            # Step 3: contract T2[c,b,f,t] with A_ket[c,t,d] -> new_env[b,f,d]
            # sum_{c,t} T2[c,b,f,t] * A[c,t,d]
            env = np.tensordot(T2, A, axes=([0, 3], [0, 1]))
            # T2: (c,b,f,t), A: (c,t,d) -> result: (b,f,d)

        return complex(env[0, 0, 0])

    # ── Hamiltonian factories ─────────────────────────────────────────────────

    @staticmethod
    def ising(n: int, J: float = 1.0, h: float = 0.5) -> "MPO":
        """Transverse-field Ising MPO: H = -J Σ Z_i Z_{i+1} − h Σ X_i.

        Uses D=3 MPO with the standard W matrix::

            W = [[I,    0,    0  ],
                 [Z,    0,    0  ],
                 [-h*X, -J*Z, I  ]]

        Boundary: W_0 = last row (1×3), W_{N-1} = first column (3×1).
        """
        if n < 2:
            raise TensorNetworkError("Ising MPO requires n >= 2")

        I, X, Z = _I2, _X, _Z
        d = 2
        D = 3

        W_bulk = np.zeros((D, D, d, d), dtype=complex)
        W_bulk[0, 0] = I
        W_bulk[1, 0] = Z
        W_bulk[2, 0] = -h * X
        W_bulk[2, 1] = -J * Z
        W_bulk[2, 2] = I

        tensors = []
        for i in range(n):
            if i == 0:
                W = W_bulk[2:3, :, :, :]   # (1, 3, d, d)
            elif i == n - 1:
                W = W_bulk[:, 0:1, :, :]   # (3, 1, d, d)
            else:
                W = W_bulk.copy()
            tensors.append(W)

        return MPO(tensors=tensors, d=d)

    @staticmethod
    def heisenberg(n: int, J: float = 1.0, hz: float = 0.0) -> "MPO":
        """Heisenberg MPO: H = J Σ (XX+YY+ZZ) + hz Σ Z.

        Uses D=5 MPO.
        """
        if n < 2:
            raise TensorNetworkError("Heisenberg MPO requires n >= 2")

        I, X, Y, Z = _I2, _X, _Y, _Z
        d = 2
        D = 5

        W_bulk = np.zeros((D, D, d, d), dtype=complex)
        W_bulk[0, 0] = I
        W_bulk[1, 0] = X
        W_bulk[2, 0] = Y
        W_bulk[3, 0] = Z
        W_bulk[4, 0] = hz * Z
        W_bulk[4, 1] = J * X
        W_bulk[4, 2] = J * Y
        W_bulk[4, 3] = J * Z
        W_bulk[4, 4] = I

        tensors = []
        for i in range(n):
            if i == 0:
                W = W_bulk[4:5, :, :, :]
            elif i == n - 1:
                W = W_bulk[:, 0:1, :, :]
            else:
                W = W_bulk.copy()
            tensors.append(W)

        return MPO(tensors=tensors, d=d)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_sites": self.n_sites,
            "bond_dim": self.tensors[1].shape[0] if self.n_sites > 1 else 1,
            "d": self.d,
        }


# ── TEBD ──────────────────────────────────────────────────────────────────────

@dataclass
class TEBD:
    """Time-Evolving Block Decimation for nearest-neighbor Hamiltonians.

    Parameters
    ----------
    mps:
        Initial MPS state (modified in-place during evolution).
    max_chi:
        Maximum bond dimension kept after each gate application.
    svd_cutoff:
        Singular values below this threshold are discarded.
    """

    mps: MPS
    max_chi: int = 32
    svd_cutoff: float = 1e-12

    def _two_site_gate(self, gate: np.ndarray, site: int) -> None:
        """Apply 4×4 (or 2×2×2×2) unitary to sites (site, site+1), SVD-truncate."""
        gate = np.asarray(gate, dtype=complex)
        if gate.shape == (4, 4):
            U = gate.reshape(2, 2, 2, 2)
        elif gate.shape == (2, 2, 2, 2):
            U = gate
        else:
            raise TensorNetworkError(f"Gate must be (4,4) or (2,2,2,2), got {gate.shape}")

        A = self.mps.tensors[site]       # (chi_l, d, chi_m)
        B = self.mps.tensors[site + 1]   # (chi_m, d, chi_r)
        chi_l, d, chi_m = A.shape
        _, _, chi_r = B.shape

        # Contract A,B -> two-site tensor theta: (chi_l, d, d, chi_r)
        # theta[i,s,t,j] = sum_a A[i,s,a] B[a,t,j]
        theta = np.tensordot(A, B, axes=([-1], [0]))  # (chi_l, d, d, chi_r)

        # Apply gate U[a,b,s,t]: U acts on physical indices
        # theta'[i,a,b,j] = sum_{s,t} U[a,b,s,t] theta[i,s,t,j]
        # tensordot: theta (chi_l,d,d,chi_r) axes (1,2) with U (d,d,d,d) axes (2,3)
        theta = np.tensordot(theta, U, axes=([1, 2], [2, 3]))
        # Result: (chi_l, chi_r, d_out, d_out) -> need (chi_l, d, d, chi_r)
        theta = theta.transpose(0, 2, 3, 1)  # (chi_l, d_out, d_out, chi_r)

        # SVD: (chi_l*d, d*chi_r)
        theta = theta.reshape(chi_l * d, d * chi_r)
        Usv, S, Vt = np.linalg.svd(theta, full_matrices=False)

        keep = int(np.sum(S > self.svd_cutoff))
        keep = max(1, min(keep, self.max_chi))
        Usv = Usv[:, :keep]
        S = S[:keep]
        Vt = Vt[:keep, :]

        self.mps.tensors[site] = Usv.reshape(chi_l, d, keep)
        self.mps.tensors[site + 1] = (S[:, np.newaxis] * Vt).reshape(keep, d, chi_r)

    def evolve_ising(
        self, J: float, h: float, dt: float, n_steps: int
    ) -> list[dict[str, Any]]:
        """Second-order Suzuki-Trotter evolution under H = -J Σ ZZ − h Σ X.

        Uses: exp(-iH_X dt/2) · exp(-iH_ZZ dt) · exp(-iH_X dt/2) per step,
        with H_ZZ applied layer-by-layer (even/odd bonds).
        """
        N = self.mps.n_sites
        history: list[dict[str, Any]] = []

        def make_zz_gate(t: float) -> np.ndarray:
            """exp(i J t ZZ) — note H=-JZZ so time-evo is exp(-i*(-J)*t*ZZ)=exp(iJtZZ)."""
            ZZ = np.kron(_Z.real, _Z.real)
            return (np.cos(J * t) * np.eye(4) + 1j * np.sin(J * t) * ZZ).astype(complex)

        def make_x_gate(t: float) -> np.ndarray:
            """exp(i h t X) for single site."""
            return (np.cos(h * t) * _I2 + 1j * np.sin(h * t) * _X).astype(complex)

        def apply_zz_layer(parity: int, t: float) -> None:
            for i in range(parity, N - 1, 2):
                self._two_site_gate(make_zz_gate(t), i)

        def apply_x_all(t: float) -> None:
            Xg = make_x_gate(t)
            for i in range(N):
                A = self.mps.tensors[i]
                # Contract Xg[s,t] with A[chi_l, t, chi_r] over physical index
                self.mps.tensors[i] = np.tensordot(Xg, A, axes=([1], [1])).transpose(1, 0, 2)

        for step in range(n_steps):
            # 2nd-order Trotter: X(dt/2), ZZ_even(dt/2), ZZ_odd(dt), ZZ_even(dt/2), X(dt/2)
            apply_x_all(dt / 2)
            apply_zz_layer(0, dt / 2)
            apply_zz_layer(1, dt)
            apply_zz_layer(0, dt / 2)
            apply_x_all(dt / 2)

            nrm = self.mps.norm()
            max_bond = max(self.mps.bond_dims) if self.mps.bond_dims else 1
            history.append({
                "step": step + 1,
                "time": (step + 1) * dt,
                "norm": nrm,
                "max_bond": max_bond,
            })

        return history

    def magnetization_z(self, site: int) -> float:
        """⟨Z_i⟩ = ⟨ψ|Z_i|ψ⟩ via environment contraction."""
        mps = self.mps
        N = mps.n_sites

        # Build left environment up to (not including) site
        L = np.ones((1, 1), dtype=complex)  # (chi_bra, chi_ket)
        for i in range(site):
            A = mps.tensors[i]
            # L[a,b], A*[a,s,c], A[b,s,d] -> new_L[c,d]
            tmp = np.tensordot(L, A.conj(), axes=([0], [0]))   # (b,s,c)
            L = np.tensordot(tmp, A, axes=([0, 1], [0, 1]))    # (c,d)

        # Build right environment from (not including) site
        R = np.ones((1, 1), dtype=complex)  # (chi_bra, chi_ket)
        for i in range(N - 1, site, -1):
            A = mps.tensors[i]
            # R[c,d], A*[a,s,c], A[b,s,d] -> new_R[a,b]
            tmp = np.tensordot(A.conj(), R, axes=([-1], [0]))   # (a,s,d)
            R = np.tensordot(tmp, A, axes=([1, 2], [1, 2]))     # (a,b)

        A = mps.tensors[site]  # (chi_l, d, chi_r)

        # Numerator: L[a,b] A*[a,s,c] Z[s,t] A[b,t,d] R[c,d]
        # Step 1: L[a,b] A*[a,s,c] -> T1[b,s,c]
        T1 = np.tensordot(L, A.conj(), axes=([0], [0]))           # (b,s,c)
        # Step 2: T1[b,s,c] Z[s,t] -> T2[b,c,t]
        T2 = np.tensordot(T1, _Z, axes=([1], [0]))                # (b,c,t)
        # Step 3: T2[b,c,t] A[b,t,d] -> T3[c,d]
        T3 = np.tensordot(T2, A, axes=([0, 2], [0, 1]))           # (c,d)
        # Step 4: T3[c,d] R[c,d] -> scalar
        numerator = np.tensordot(T3, R, axes=([0, 1], [0, 1]))

        # Denominator: norm squared
        T1n = np.tensordot(L, A.conj(), axes=([0], [0]))          # (b,s,c)
        T3n = np.tensordot(T1n, A, axes=([0, 1], [0, 1]))         # (c,d)
        denominator = np.tensordot(T3n, R, axes=([0, 1], [0, 1]))

        if abs(denominator) < 1e-30:
            return 0.0
        return float(np.real(numerator / denominator))


# ── DMRG ──────────────────────────────────────────────────────────────────────

@dataclass
class DMRGResult:
    """Result container for DMRG.run()."""

    energy: float
    mps: MPS
    sweep_energies: list[float]
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy": self.energy,
            "converged": self.converged,
            "n_sweeps": len(self.sweep_energies),
            "sweep_energies": self.sweep_energies,
        }


@dataclass
class DMRG:
    """Two-site DMRG ground-state search for MPO Hamiltonians.

    Parameters
    ----------
    mpo:
        The Hamiltonian as an MPO.
    max_chi:
        Maximum bond dimension for the MPS ansatz.
    n_sweeps:
        Total number of sweeps.
    tol:
        Convergence threshold on energy change per sweep.
    seed:
        Random seed for the initial MPS.
    """

    mpo: MPO
    max_chi: int = 16
    n_sweeps: int = 10
    tol: float = 1e-8
    seed: int | None = None

    def run(self, initial_mps: MPS | None = None) -> DMRGResult:
        """Run DMRG and return the ground state energy and MPS."""
        N = self.mpo.n_sites
        d = self.mpo.d

        if initial_mps is not None and initial_mps.n_sites != N:
            raise TensorNetworkError(
                f"initial_mps has {initial_mps.n_sites} sites but MPO has {N} sites"
            )
        if initial_mps is None:
            mps = MPS.random(N, d=d, max_chi=2, seed=self.seed)
        else:
            mps = MPS(tensors=[t.copy() for t in initial_mps.tensors], d=d)

        # Right-canonicalize
        mps = _right_canonicalize(mps)

        sweep_energies: list[float] = []
        energy = float("inf")
        converged = False

        for sweep in range(self.n_sweeps):
            # Left-to-right sweep
            L_env = np.ones((1, 1, 1), dtype=complex)
            R_envs = _build_right_envs(mps, self.mpo)

            for i in range(N - 1):
                R_env = R_envs[i + 2]  # right env excludes both sites i and i+1
                energy, mps = _dmrg_update(mps, self.mpo, L_env, R_env, i, self.max_chi, "right")
                L_env = _grow_left_env(L_env, mps.tensors[i], self.mpo.tensors[i])

            # Right-to-left sweep
            R_env = np.ones((1, 1, 1), dtype=complex)
            for i in range(N - 2, -1, -1):
                L_env_i = _build_left_env_to(mps, self.mpo, i)
                energy, mps = _dmrg_update(mps, self.mpo, L_env_i, R_env, i, self.max_chi, "left")
                R_env = _grow_right_env(R_env, mps.tensors[i + 1], self.mpo.tensors[i + 1])

            sweep_energies.append(float(energy))

            if len(sweep_energies) >= 2:
                delta = abs(sweep_energies[-1] - sweep_energies[-2])
                if delta < self.tol:
                    converged = True
                    break

        return DMRGResult(
            energy=float(energy),
            mps=mps,
            sweep_energies=sweep_energies,
            converged=converged,
        )


# ── DMRG helper functions ─────────────────────────────────────────────────────

def _right_canonicalize(mps: MPS) -> MPS:
    """Return right-canonical form via right-to-left QR decomposition."""
    N = mps.n_sites
    tensors = [t.copy() for t in mps.tensors]
    for i in range(N - 1, 0, -1):
        A = tensors[i]
        chi_l, d, chi_r = A.shape
        M = A.reshape(chi_l, d * chi_r)
        Qt, Rt = np.linalg.qr(M.T)
        k = Qt.shape[1]
        tensors[i] = Qt.T.reshape(k, d, chi_r)
        # Absorb Rt.T into left neighbor: prev[chi_ll, d_prev, chi_l] @ Rt.T[chi_l, k]
        tensors[i - 1] = np.tensordot(tensors[i - 1], Rt.T, axes=([-1], [0]))
    return MPS(tensors=tensors, d=mps.d)


def _build_right_envs(mps: MPS, mpo: MPO) -> list[np.ndarray]:
    """Build all right environments R_envs[i] where R_envs[N] = identity."""
    N = mps.n_sites
    R_envs: list[np.ndarray | None] = [None] * (N + 1)
    R_envs[N] = np.ones((1, 1, 1), dtype=complex)
    for i in range(N - 1, 0, -1):
        R_envs[i] = _grow_right_env(R_envs[i + 1], mps.tensors[i], mpo.tensors[i])
    return R_envs


def _build_left_env_to(mps: MPS, mpo: MPO, site: int) -> np.ndarray:
    """Build left environment contracted up to (not including) site."""
    env = np.ones((1, 1, 1), dtype=complex)
    for i in range(site):
        env = _grow_left_env(env, mps.tensors[i], mpo.tensors[i])
    return env


def _grow_left_env(env: np.ndarray, A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Grow left environment by one site.

    env: (chi_bra_l, D_l, chi_ket_l)
    A:   (chi_l, d, chi_r)        — bra (conj) and ket use same tensor
    W:   (D_l, D_r, d_out, d_in)  — d_out for bra side, d_in for ket side

    Returns new_env: (chi_bra_r, D_r, chi_ket_r)
    """
    # new_env[b, f, p] = sum_{a,e,c,s,t} env[a,e,c] A*[a,s,b] W[e,f,s,t] A[c,t,p]

    # Step 1: contract env[a,e,c] with A_bra*[a,s,b]
    #   env axes (0,) = a, A*.conj axes (0,) = a  -> T1: (e,c,s,b)
    T1 = np.tensordot(env, A.conj(), axes=([0], [0]))  # (e,c,s,b)

    # Step 2: contract T1[e,c,s,b] with W[e,f,s,t]
    #   sum over (e,s): T1 axes (0,2), W axes (0,2)  -> T2: (c,b,f,t)
    T2 = np.tensordot(T1, W, axes=([0, 2], [0, 2]))    # (c,b,f,t)

    # Step 3: contract T2[c,b,f,t] with A_ket[c,t,p]
    #   sum over (c,t): T2 axes (0,3), A axes (0,1)  -> new_env: (b,f,p)
    new_env = np.tensordot(T2, A, axes=([0, 3], [0, 1]))  # (b,f,p)
    return new_env


def _grow_right_env(env: np.ndarray, A: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Grow right environment by one site.

    env: (chi_bra_r, D_r, chi_ket_r)
    A:   (chi_l, d, chi_r)
    W:   (D_l, D_r, d_out, d_in)

    Returns new_env: (chi_bra_l, D_l, chi_ket_l)
    """
    # new_env[a, e, c] = sum_{b,f,p,s,t} A*[a,s,b] W[e,f,s,t] env[b,f,p] A[c,t,p]

    # Step 1: contract env[b,f,p] with A_ket[c,t,p]
    #   sum over p: env axes (2,), A axes (2,)  -> T1: (b,f,c,t)
    T1 = np.tensordot(env, A, axes=([2], [2]))          # (b,f,c,t)

    # Step 2: contract T1[b,f,c,t] with W[e,f,s,t]
    #   sum over (f,t): T1 axes (1,3), W axes (1,3)  -> T2: (b,c,e,s)
    T2 = np.tensordot(T1, W, axes=([1, 3], [1, 3]))    # (b,c,e,s)

    # Step 3: contract T2[b,c,e,s] with A_bra*[a,s,b]
    #   sum over (b,s): T2 axes (0,3), A* axes (2,1)  -> new_env: (c,e,a)
    new_env = np.tensordot(T2, A.conj(), axes=([0, 3], [2, 1]))  # (c,e,a)
    # Reorder to (a, e, c)
    return new_env.transpose(2, 1, 0)


def _dmrg_update(
    mps: MPS,
    mpo: MPO,
    L_env: np.ndarray,
    R_env: np.ndarray,
    site: int,
    max_chi: int,
    direction: str,
    eps: float | None = None,
) -> tuple[float, MPS]:
    """Single two-site DMRG update at bond (site, site+1)."""
    A_l = mps.tensors[site]        # (chi_l, d, chi_m)
    A_r = mps.tensors[site + 1]   # (chi_m, d, chi_r)
    W_l = mpo.tensors[site]        # (D_l, D_m, d, d)
    W_r = mpo.tensors[site + 1]   # (D_m, D_r, d, d)

    chi_l = A_l.shape[0]
    d = A_l.shape[1]
    chi_r = A_r.shape[2]

    dim = chi_l * d * d * chi_r

    # Build H_eff[a,s,t,b, a',s',t',b'] = L[a,e,a'] W_l[e,f,s,s'] W_r[f,g,t,t'] R[b,g,b']
    # where the |v> index ordering is (a, s, t, b)

    # Step 1: combine W_l and W_r -> WW[e,g,s,s',t,t'] via sum over f
    WW = np.tensordot(W_l, W_r, axes=([1], [0]))
    # W_l: (D_l, D_m, d, d), W_r: (D_m, D_r, d, d) -> WW: (D_l, d, d, D_r, d, d)
    # WW[e, s, sp, g, t, tp] where sp=s', tp=t'
    # Reorder: (D_l, D_r, d, d, d, d) = (e, g, s, sp, t, tp)
    WW = WW.transpose(0, 3, 1, 2, 4, 5)  # (D_l, D_r, d_l_out, d_l_in, d_r_out, d_r_in)
    # WW[e, g, s, sp, t, tp]

    # Step 2: contract L[a,e,ap] with WW[e,g,s,sp,t,tp]
    #   sum over e: L axes (1,), WW axes (0,)  -> LW: (a, ap, g, s, sp, t, tp)
    LW = np.tensordot(L_env, WW, axes=([1], [0]))   # (a, ap, g, s, sp, t, tp)

    # Step 3: contract LW with R[b,g,bp]
    #   sum over g: LW axes (2,), R axes (1,)  -> LWR: (a, ap, s, sp, t, tp, b, bp)
    LWR = np.tensordot(LW, R_env, axes=([2], [1]))  # (a, ap, s, sp, t, tp, b, bp)

    # H_eff shape: indices are (a,s,t,b) for bra, (ap,sp,tp,bp) for ket
    # LWR currently: (a, ap, s, sp, t, tp, b, bp)
    # Reorder to (a, s, t, b, ap, sp, tp, bp)
    H_eff_full = LWR.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    H_eff_mat = H_eff_full.reshape(dim, dim)

    # Symmetrize
    H_eff_mat = (H_eff_mat + H_eff_mat.conj().T) / 2

    # Diagonalize
    try:
        evals, evecs = np.linalg.eigh(H_eff_mat)
    except np.linalg.LinAlgError:
        evals_c, evecs_c = np.linalg.eig(H_eff_mat)
        idx = np.argsort(np.real(evals_c))
        evals = np.real(evals_c[idx])
        evecs = evecs_c[:, idx]

    energy = float(np.real(evals[0]))
    gs = evecs[:, 0]

    # SVD decompose ground state
    theta = gs.reshape(chi_l * d, d * chi_r)
    U, S, Vt = np.linalg.svd(theta, full_matrices=False)

    chi_new = min(max_chi, len(S))
    # Apply eps-based adaptive truncation: discard σᵢ < eps·σ₀
    if eps is not None and len(S) > 0 and S[0] > 0:
        adaptive_chi = int(np.sum(S > eps * S[0]))
        chi_new = min(chi_new, max(1, adaptive_chi))
    U = U[:, :chi_new]
    S = S[:chi_new]
    Vt = Vt[:chi_new, :]

    tensors = [t.copy() for t in mps.tensors]
    if direction == "right":
        tensors[site] = U.reshape(chi_l, d, chi_new)
        tensors[site + 1] = (S[:, np.newaxis] * Vt).reshape(chi_new, d, chi_r)
    else:
        tensors[site] = (U * S[np.newaxis, :]).reshape(chi_l, d, chi_new)
        tensors[site + 1] = Vt.reshape(chi_new, d, chi_r)

    return energy, MPS(tensors=tensors, d=mps.d)
