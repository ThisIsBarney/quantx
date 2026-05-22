"""Time-Dependent DMRG (tDMRG) — M94.

Provides real- and imaginary-time MPS evolution under arbitrary MPO Hamiltonians
using the TEBD Trotter-gate machinery with DMRG-style bond-dimension control.

Algorithm
---------
Two-site Trotter gates are extracted from the MPO by contracting adjacent
tensors along the MPO bond index::

    h_{i,i+1}[s0,s1,t0,t1] = Σ_k  W_i[-1, k, s0, t0] · W_{i+1}[k, 0, s1, t1]

i.e., the left tensor contributes its "last row" (the one that starts building
up the operator from the left) and the right tensor its "first column" (the one
that finalises the action on the right).

For real-time evolution each gate is U = exp(−i h_bond dt).
For imaginary-time evolution each gate is Γ = exp(−h_bond dτ) followed by
re-normalisation.

Trotter order 2 (default)::

    U(dt) ≈ [Π_{even} U_bond(dt/2)] · [Π_{odd} U_bond(dt)] · [Π_{even} U_bond(dt/2)]

Entanglement entropy is the von Neumann entropy of the bipartition at each
internal bond after each sweep::

    S_i = −Σ_k λ_k² log λ_k²

where {λ_k} are the normalised singular values kept at bond i.

Classes
-------
* :class:`TDMRGError` — base exception.
* :class:`TDMRGResult` — output container.
* :class:`TDMRGSolver` — main solver.

Functions
---------
* :func:`real_time_evolve` — convenience wrapper.
* :func:`imaginary_time_evolve` — convenience wrapper.

References
----------
.. [1] White, S. R. (1992). Density matrix formulation for quantum
       renormalization groups. *Phys. Rev. Lett.* 69, 2863.
       https://doi.org/10.1103/PhysRevLett.69.2863

.. [2] White, S. R. (1993). Density-matrix algorithms for quantum
       renormalization groups. *Phys. Rev. B* 48, 10345.
       https://doi.org/10.1103/PhysRevB.48.10345

.. [3] Vidal, G. (2004). Efficient simulation of one-dimensional quantum
       many-body systems. *Phys. Rev. Lett.* 93, 040502.
       https://doi.org/10.1103/PhysRevLett.93.040502

Physical validation
-------------------
Imaginary-time tDMRG on the 4-site Heisenberg chain (H = Σ XX+YY+ZZ, open BC)
converges to E_0 = -6.4641016... within tolerance 1e-6 after sufficient steps,
matching exact diagonalisation (numpy.linalg.eigvalsh reference).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .tensor_network import MPS, MPO, TensorNetworkError


# ── Exception ─────────────────────────────────────────────────────────────────

class TDMRGError(ValueError):
    """Raised on invalid tDMRG inputs or convergence failures."""


# ── TDMRGResult ───────────────────────────────────────────────────────────────

@dataclass
class TDMRGResult:
    """Output of :class:`TDMRGSolver`.

    Attributes
    ----------
    times : ndarray, shape (n_steps,)
        Physical times (dt * step index).
    energies : ndarray, shape (n_steps,)
        MPO energy expectation value ⟨H⟩ at each step.
    entanglement_entropy : ndarray, shape (n_steps, n_sites-1)
        Von Neumann entropy S = −Σ λ² log λ² at each internal bond.
    bond_dims : list[list[int]]
        Bond dimensions at each time step; ``bond_dims[t][b]`` is the bond
        dimension at bond *b* after step *t*.
    norms : ndarray, shape (n_steps,)
        MPS norm at each step (should stay ≈ 1 for real-time, decreases for
        imaginary time before renormalisation).
    observables : dict[str, ndarray]
        Per-step expectation values for user-supplied observables.
        Keys are observable names; values have shape (n_steps,).
    mps_final : MPS
        State after the last time step.
    dt : float
        Time step used.
    imaginary : bool
        Whether imaginary-time evolution was performed.
    """

    times: np.ndarray
    energies: np.ndarray
    entanglement_entropy: np.ndarray   # shape (n_steps, n_sites-1)
    bond_dims: list[list[int]]
    norms: np.ndarray
    observables: dict[str, np.ndarray]
    mps_final: MPS
    dt: float
    imaginary: bool = False

    def max_entropy(self) -> float:
        """Maximum entanglement entropy across all bonds and times."""
        return float(np.max(self.entanglement_entropy))

    def entropy_at(self, time_idx: int = -1) -> np.ndarray:
        """Entanglement entropy profile at a given time step."""
        return self.entanglement_entropy[time_idx]

    def __repr__(self) -> str:
        n_steps = len(self.times)
        final_e = float(self.energies[-1]) if n_steps else float("nan")
        return (
            f"TDMRGResult(n_steps={n_steps}, dt={self.dt:.4g}, "
            f"final_energy={final_e:.6f}, "
            f"max_entropy={self.max_entropy():.4f}, "
            f"imaginary={self.imaginary})"
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _mpo_bond_hamiltonian(mpo: MPO, site: int) -> np.ndarray:
    """Extract the 4×4 two-site Hamiltonian h_{site, site+1} from the MPO.

    Contracts W_site[-1, k, :, :] ⊗ W_{site+1}[k, 0, :, :] over k.

    Returns
    -------
    ndarray, shape (d*d, d*d)  — Hermitian two-site operator.
    """
    Wi = mpo.tensors[site]       # (D_l_i, D_r_i, d, d)
    Wj = mpo.tensors[site + 1]  # (D_l_j, D_r_j, d, d)
    d = mpo.d

    # Last row of Wi: Wi[-1, k, s, t]  shape (D_r_i, d, d)
    Wi_row = Wi[-1]              # (D_r_i, d, d)
    # First col of Wj: Wj[k, 0, s, t] shape (D_l_j, d, d)
    Wj_col = Wj[:, 0]            # (D_l_j, d, d)

    # Sanity: D_r_i == D_l_j (bond index)
    D = Wi_row.shape[0]

    # h[s0, s1, t0, t1] = Σ_k Wi_row[k, s0, t0] * Wj_col[k, s1, t1]
    # Use einsum: k s0 t0, k s1 t1 -> s0 s1 t0 t1
    h = np.einsum("kst,kuv->sutv", Wi_row, Wj_col)  # (d, d, d, d)
    # Reshape to (d^2, d^2)
    h = h.reshape(d * d, d * d)

    # Symmetrise to correct numerical drift
    h = 0.5 * (h + h.conj().T)
    return h


def _gate_from_h(h: np.ndarray, dt: float, imaginary: bool = False) -> np.ndarray:
    """Compute the time-evolution gate for a two-site Hamiltonian.

    Parameters
    ----------
    h : ndarray, shape (d², d²) — Hermitian two-site Hamiltonian.
    dt : float — time step (positive).
    imaginary : bool — if True, use imaginary time (exp(−h dt)); else exp(−i h dt).

    Returns
    -------
    ndarray, shape (d², d²).
    """
    if imaginary:
        # exp(-h * dt)
        eigvals, eigvecs = np.linalg.eigh(h)
        return (eigvecs * np.exp(-eigvals * dt)) @ eigvecs.conj().T
    else:
        # exp(-i h dt) — Hermitian h → unitary
        eigvals, eigvecs = np.linalg.eigh(h)
        return (eigvecs * np.exp(-1j * eigvals * dt)) @ eigvecs.conj().T


def _apply_two_site_gate(
    mps: MPS,
    gate: np.ndarray,
    site: int,
    max_chi: int,
    svd_cutoff: float,
) -> np.ndarray:
    """Apply a (d², d²) gate to sites (site, site+1) with SVD truncation.

    Returns
    -------
    singular_values : ndarray
        Singular values *before* truncation (for entropy computation).
    """
    d = mps.d
    A = mps.tensors[site]       # (chi_l, d, chi_m)
    B = mps.tensors[site + 1]  # (chi_m, d, chi_r)
    chi_l, _, chi_m = A.shape
    _, _, chi_r = B.shape

    # Two-site tensor θ: (chi_l, d, d, chi_r)
    theta = np.tensordot(A, B, axes=([-1], [0]))

    # Apply gate U[s0,s1,t0,t1] -> U(d²,d²) reshaped to (d,d,d,d)
    U = gate.reshape(d, d, d, d)
    # theta'[l,a,b,r] = Σ_{st} U[a,b,s,t] theta[l,s,t,r]
    theta = np.tensordot(theta, U, axes=([1, 2], [2, 3]))
    theta = theta.transpose(0, 2, 3, 1)  # (chi_l, d, d, chi_r)

    # SVD: (chi_l * d, d * chi_r)
    theta = theta.reshape(chi_l * d, d * chi_r)
    Usv, S, Vt = np.linalg.svd(theta, full_matrices=False)

    # Truncation
    keep = int(np.sum(S > svd_cutoff))
    keep = max(1, min(keep, max_chi))

    mps.tensors[site] = Usv[:, :keep].reshape(chi_l, d, keep)
    mps.tensors[site + 1] = (S[:keep, np.newaxis] * Vt[:keep, :]).reshape(
        keep, d, chi_r
    )
    return S[:keep]  # truncated singular values — match the actual MPS state


def _entanglement_entropy(singular_values: np.ndarray) -> float:
    """Von Neumann entropy S = −Σ λ² log λ²  (λ normalised)."""
    s = singular_values[singular_values > 0]
    s2 = s ** 2
    norm = s2.sum()
    if norm <= 0:
        return 0.0
    p = s2 / norm
    return float(-np.sum(p * np.log(p + 1e-300)))


def _mps_energy(mps: MPS, mpo: MPO) -> float:
    """Compute ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩."""
    bra_ket = mpo.expectation(mps)
    norm2 = mps.norm() ** 2
    if norm2 < 1e-30:
        return float("nan")
    return float(np.real(bra_ket)) / norm2


def _single_site_expectation(mps: MPS, op: np.ndarray, site: int) -> complex:
    """Compute ⟨ψ| op_site |ψ⟩ / ⟨ψ|ψ⟩."""
    N = mps.n_sites
    # Left environment
    L = np.ones((1, 1), dtype=complex)
    for i in range(site):
        A = mps.tensors[i]
        tmp = np.tensordot(L, A.conj(), axes=([0], [0]))  # (b, s, c)
        L = np.tensordot(tmp, A, axes=([0, 1], [0, 1]))   # (c, d)

    # Right environment
    R = np.ones((1, 1), dtype=complex)
    for i in range(N - 1, site, -1):
        A = mps.tensors[i]
        tmp = np.tensordot(A.conj(), R, axes=([-1], [0]))  # (a, s, d)
        R = np.tensordot(tmp, A, axes=([1, 2], [1, 2]))    # (a, b)

    A = mps.tensors[site]  # (chi_l, d, chi_r)
    T1 = np.tensordot(L, A.conj(), axes=([0], [0]))    # (b, s, c)
    T2 = np.tensordot(T1, op, axes=([1], [0]))          # (b, c, t)
    T2 = T2.transpose(0, 2, 1)                          # (b, t, c)
    T3 = np.tensordot(T2, A, axes=([0, 1], [0, 1]))    # (c, d)
    num = np.tensordot(T3, R, axes=([0, 1], [0, 1]))   # scalar

    norm2 = mps.norm() ** 2
    return complex(num) / norm2 if norm2 > 1e-30 else 0j


# ── TDMRGSolver ───────────────────────────────────────────────────────────────

class TDMRGSolver:
    """Time-Dependent DMRG via MPO-derived TEBD gates.

    Parameters
    ----------
    mpo : MPO
        Hamiltonian in MPO form (supports :meth:`MPO.ising`,
        :meth:`MPO.heisenberg`, or any user-built MPO).
    dt : float
        Time step (positive; real or imaginary depending on ``imaginary``).
    n_steps : int
        Number of time steps.
    max_chi : int
        Maximum bond dimension kept after each gate application.
    svd_cutoff : float
        Singular values below this threshold are discarded.
    trotter_order : {1, 2}
        Trotter decomposition order.  Order 2 (default) uses symmetric
        even–odd–even splitting with half-steps on the even layer.
    imaginary : bool
        If ``True`` perform imaginary-time evolution (convergence to ground
        state); re-normalises the MPS after each full sweep.
    observables : dict[str, tuple[int, ndarray]] or None
        Optional user-defined single-site observables to track.  Each entry
        maps an observable name to ``(site, operator_matrix)``.
    """

    def __init__(
        self,
        mpo: MPO,
        dt: float,
        n_steps: int,
        max_chi: int = 32,
        svd_cutoff: float = 1e-12,
        trotter_order: int = 2,
        imaginary: bool = False,
        observables: dict[str, tuple[int, np.ndarray]] | None = None,
    ) -> None:
        if not isinstance(mpo, MPO):
            raise TDMRGError(f"Expected MPO, got {type(mpo).__name__}")
        if mpo.n_sites < 2:
            raise TDMRGError("tDMRG requires at least 2 sites")
        if dt <= 0:
            raise TDMRGError("dt must be positive")
        if n_steps < 1:
            raise TDMRGError("n_steps must be >= 1")
        if max_chi < 1:
            raise TDMRGError("max_chi must be >= 1")
        if trotter_order not in (1, 2):
            raise TDMRGError("trotter_order must be 1 or 2")

        self.mpo = mpo
        self.dt = dt
        self.n_steps = n_steps
        self.max_chi = max_chi
        self.svd_cutoff = svd_cutoff
        self.trotter_order = trotter_order
        self.imaginary = imaginary
        self.observables = observables or {}

        # Pre-compute bond Hamiltonians and Trotter gates
        self._h_bonds: list[np.ndarray] = []
        self._gates_full: list[np.ndarray] = []   # full-step gates
        self._gates_half: list[np.ndarray] = []   # half-step gates (order 2 only)

        n = mpo.n_sites
        for i in range(n - 1):
            h = _mpo_bond_hamiltonian(mpo, i)
            self._h_bonds.append(h)
            self._gates_full.append(_gate_from_h(h, dt, imaginary=imaginary))
            self._gates_half.append(_gate_from_h(h, dt / 2.0, imaginary=imaginary))

    # ── Sweep helpers ─────────────────────────────────────────────────────────

    def _sweep_layer(
        self,
        mps: MPS,
        parity: int,
        gates: list[np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Apply gates on all bonds with a given parity (even=0 or odd=1).

        Returns a dict {bond_index: singular_values} for the touched bonds.
        """
        n = mps.n_sites
        sv_dict: dict[int, np.ndarray] = {}
        for i in range(parity, n - 1, 2):
            sv = _apply_two_site_gate(
                mps, gates[i], i, self.max_chi, self.svd_cutoff
            )
            sv_dict[i] = sv
        return sv_dict

    def _full_trotter_step(
        self, mps: MPS
    ) -> dict[int, np.ndarray]:
        """Apply one full Trotter step; return {bond: sv_array}."""
        if self.trotter_order == 1:
            sv = {}
            for i in range(mps.n_sites - 1):
                sv[i] = _apply_two_site_gate(
                    mps, self._gates_full[i], i, self.max_chi, self.svd_cutoff
                )
            return sv

        # Second-order: even(half) → odd(full) → even(half)
        sv: dict[int, np.ndarray] = {}
        sv.update(self._sweep_layer(mps, parity=0, gates=self._gates_half))
        sv.update(self._sweep_layer(mps, parity=1, gates=self._gates_full))
        sv.update(self._sweep_layer(mps, parity=0, gates=self._gates_half))
        return sv

    # ── Main solver ───────────────────────────────────────────────────────────

    def run(self, mps_initial: MPS | None = None) -> TDMRGResult:
        """Run the tDMRG time evolution.

        Parameters
        ----------
        mps_initial : MPS, optional
            Starting state.  If ``None``, uses the all-zeros product state
            |0 0 … 0⟩ for real-time or a random MPS for imaginary-time.

        Returns
        -------
        TDMRGResult
        """
        n = self.mpo.n_sites

        if mps_initial is not None:
            if not isinstance(mps_initial, MPS):
                raise TDMRGError(
                    f"mps_initial must be an MPS, got {type(mps_initial).__name__}"
                )
            if mps_initial.n_sites != n:
                raise TDMRGError(
                    f"mps_initial has {mps_initial.n_sites} sites, "
                    f"but MPO has {n}."
                )
            # Work on a copy so the caller's state isn't mutated
            mps = MPS(
                tensors=[t.copy() for t in mps_initial.tensors],
                d=mps_initial.d,
            )
        else:
            if self.imaginary:
                # Random normalised starting state
                mps = MPS.random(n, d=self.mpo.d, max_chi=self.max_chi, seed=0)
                mps.normalize()
            else:
                mps = MPS.product_state([0] * n, d=self.mpo.d)

        # Storage for time-series data
        n_bonds = n - 1
        times = np.empty(self.n_steps)
        energies = np.empty(self.n_steps)
        entropies = np.empty((self.n_steps, n_bonds))
        bond_dims: list[list[int]] = []
        norms = np.empty(self.n_steps)
        obs_history: dict[str, list[complex]] = {k: [] for k in self.observables}

        for step in range(self.n_steps):
            sv_dict = self._full_trotter_step(mps)

            # Imaginary time: renormalise to prevent numerical underflow
            if self.imaginary:
                mps.normalize()

            t = (step + 1) * self.dt
            times[step] = t

            # Entanglement entropy at each bond
            for b in range(n_bonds):
                if b in sv_dict:
                    entropies[step, b] = _entanglement_entropy(sv_dict[b])
                elif step > 0:
                    entropies[step, b] = entropies[step - 1, b]
                else:
                    entropies[step, b] = 0.0

            bond_dims.append(list(mps.bond_dims))
            norms[step] = mps.norm()
            energies[step] = _mps_energy(mps, self.mpo)

            # User observables
            for name, (site, op) in self.observables.items():
                obs_history[name].append(
                    _single_site_expectation(mps, op, site)
                )

        obs_arrays: dict[str, np.ndarray] = {
            k: np.array(v) for k, v in obs_history.items()
        }

        return TDMRGResult(
            times=times,
            energies=energies,
            entanglement_entropy=entropies,
            bond_dims=bond_dims,
            norms=norms,
            observables=obs_arrays,
            mps_final=mps,
            dt=self.dt,
            imaginary=self.imaginary,
        )

    def __repr__(self) -> str:
        kind = "imaginary" if self.imaginary else "real"
        return (
            f"TDMRGSolver(n_sites={self.mpo.n_sites}, dt={self.dt:.4g}, "
            f"n_steps={self.n_steps}, max_chi={self.max_chi}, "
            f"trotter_order={self.trotter_order}, time={kind})"
        )


# ── Convenience functions ─────────────────────────────────────────────────────

def real_time_evolve(
    mpo: MPO,
    mps_initial: MPS | None = None,
    *,
    dt: float = 0.01,
    n_steps: int = 100,
    max_chi: int = 32,
    svd_cutoff: float = 1e-12,
    trotter_order: int = 2,
    observables: dict[str, tuple[int, np.ndarray]] | None = None,
) -> TDMRGResult:
    """Real-time MPS evolution |ψ(t)⟩ = e^{−iHt} |ψ(0)⟩.

    Shorthand for::

        TDMRGSolver(mpo, dt=dt, n_steps=n_steps, ..., imaginary=False).run(mps_initial)

    Parameters
    ----------
    mpo : MPO
        System Hamiltonian.
    mps_initial : MPS, optional
        Initial state (default: |0…0⟩).
    dt, n_steps, max_chi, svd_cutoff, trotter_order, observables:
        Forwarded to :class:`TDMRGSolver`.

    Returns
    -------
    TDMRGResult
    """
    solver = TDMRGSolver(
        mpo=mpo,
        dt=dt,
        n_steps=n_steps,
        max_chi=max_chi,
        svd_cutoff=svd_cutoff,
        trotter_order=trotter_order,
        imaginary=False,
        observables=observables,
    )
    return solver.run(mps_initial)


def imaginary_time_evolve(
    mpo: MPO,
    mps_initial: MPS | None = None,
    *,
    dtau: float = 0.05,
    n_steps: int = 200,
    max_chi: int = 32,
    svd_cutoff: float = 1e-12,
) -> TDMRGResult:
    """Ground-state search via imaginary-time evolution.

    Evolves |ψ⟩ → exp(−H τ) |ψ⟩ / ‖…‖ until the lowest eigenstate is
    selected.  The energy expectation value converges to E₀.

    Parameters
    ----------
    mpo : MPO
        System Hamiltonian.
    mps_initial : MPS, optional
        Starting state.  Defaults to a random normalised MPS.
    dtau : float
        Imaginary time step (positive).
    n_steps : int
        Number of imaginary-time steps.
    max_chi : int
        Maximum bond dimension.
    svd_cutoff : float
        SVD truncation threshold.

    Returns
    -------
    TDMRGResult
        ``result.energies[-1]`` approximates the ground-state energy.
    """
    solver = TDMRGSolver(
        mpo=mpo,
        dt=dtau,
        n_steps=n_steps,
        max_chi=max_chi,
        svd_cutoff=svd_cutoff,
        trotter_order=2,
        imaginary=True,
    )
    return solver.run(mps_initial)
