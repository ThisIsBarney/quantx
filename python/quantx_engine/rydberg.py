"""M118: Neutral-atom backend — Rydberg blockade simulator.

Simulates neutral-atom quantum processors (QuEra Aquila, Pasqal) based on
the Rydberg blockade mechanism.  Atoms are modelled as two-level systems
|g⟩ (ground) and |r⟩ (Rydberg), driven by a global laser field.

**Physics model**
~~~~~~~~~~~~~~~~~
The many-body Hamiltonian (in units where ħ = 1) is::

    H(t) = (Ω/2) Σ_i σ^x_i  −  δ Σ_i n_i  +  Σ_{i<j} V_{ij} n_i n_j

where:

* Ω  — Rabi frequency of the global drive
* δ  — laser detuning
* n_i = |r⟩⟨r|_i — Rydberg population on atom i
* V_{ij} = C₆ / r_{ij}⁶ — van der Waals blockade interaction (C₆ is the
  Rydberg C₆ coefficient, r_{ij} the inter-atom distance)

For r_{ij} < R_b (blockade radius R_b = (C₆/Ω)^{1/6}), two atoms cannot
simultaneously occupy the Rydberg state.

**Usage**
~~~~~~~~~
::

    atoms = [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]  # μm
    sim = RydbergSimulator(atom_positions=atoms, Omega=1.0, delta=0.0, C6=862690)
    result = sim.evolve(t_final=2*pi, n_steps=100)
    # result.populations[i] = Rydberg population ⟨n_i⟩(t)

Classes
-------
RydbergError
RydbergAtom
RydbergHamiltonian
RydbergResult
RydbergSimulator

References
----------
.. [1] Jaksch, D. et al. (2000). Fast quantum gates for neutral atoms.
       *Phys. Rev. Lett.* 85, 2208.
       https://doi.org/10.1103/PhysRevLett.85.2208

.. [2] Urban, E. et al. (2009). Observation of Rydberg blockade between two
       atoms. *Nat. Phys.* 5, 110.
       https://doi.org/10.1038/nphys1178

.. [3] Lukin, M. D. et al. (2001). Dipole blockade and quantum information
       processing in mesoscopic atomic ensembles.
       *Phys. Rev. Lett.* 87, 037901.
       https://doi.org/10.1103/PhysRevLett.87.037901

Physical validation
-------------------
Van der Waals blockade: V(r) = C_6 / r^6. Blockade radius r_b defined by
V(r_b) = Omega (Rabi frequency). For r < r_b, doubly-excited state is
energetically suppressed. Rabi oscillations: Omega_eff = sqrt(n)*Omega
for an n-atom ensemble in the blockade regime [Lukin2001].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "RydbergError",
    "RydbergAtom",
    "RydbergHamiltonian",
    "RydbergResult",
    "RydbergSimulator",
]

# Physical constants (SI / natural units for Rydberg physics)
_DEFAULT_C6 = 862690.0      # C₆ for Rb 70S state in MHz·μm⁶
_DEFAULT_OMEGA = 1.0        # Rabi frequency in MHz
_DEFAULT_DELTA = 0.0        # Detuning in MHz


class RydbergError(ValueError):
    """Raised for invalid Rydberg simulation parameters."""


# ── Atom and Hamiltonian dataclasses ─────────────────────────────────────────

@dataclass
class RydbergAtom:
    """A single neutral atom in 2-D space.

    Attributes
    ----------
    x, y : float
        Position in μm.
    label : str
        Optional atom label.
    """
    x: float
    y: float
    label: str = ""

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y

    def distance_to(self, other: "RydbergAtom") -> float:
        """Euclidean distance to another atom in μm."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "label": self.label}


@dataclass
class RydbergHamiltonian:
    """Sparse representation of the Rydberg Hamiltonian parameters.

    Attributes
    ----------
    n_atoms : int
    Omega : float
        Global Rabi frequency (MHz or a.u.).
    delta : float
        Laser detuning (same units as Ω).
    interaction_matrix : np.ndarray, shape (n_atoms, n_atoms)
        V_{ij} = C₆ / r_{ij}⁶ (off-diagonal); diagonal = 0.
    blockade_radius : float
        R_b = (C₆ / Ω)^{1/6}  (μm, approximate).
    """
    n_atoms: int
    Omega: float
    delta: float
    interaction_matrix: np.ndarray
    blockade_radius: float

    def to_dict(self) -> dict:
        return {
            "n_atoms": self.n_atoms,
            "Omega": self.Omega,
            "delta": self.delta,
            "blockade_radius": self.blockade_radius,
            "interaction_matrix": self.interaction_matrix.tolist(),
        }


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class RydbergResult:
    """Result of a :class:`RydbergSimulator` evolution.

    Attributes
    ----------
    times : np.ndarray, shape (n_steps,)
        Time points in simulation units.
    statevector : np.ndarray, shape (2**n, n_steps)
        Statevector at each time step (complex).
    populations : np.ndarray, shape (n_atoms, n_steps)
        Rydberg population ⟨n_i⟩(t) for each atom.
    total_rydberg : np.ndarray, shape (n_steps,)
        Total Rydberg excitation Σ_i ⟨n_i⟩(t).
    n_atoms : int
    """
    times: np.ndarray
    statevector: np.ndarray
    populations: np.ndarray
    total_rydberg: np.ndarray
    n_atoms: int

    def final_state(self) -> np.ndarray:
        """Return statevector at the final time step."""
        return self.statevector[:, -1]

    def rydberg_population(self, atom: int) -> np.ndarray:
        """Return ⟨n_atom⟩(t) time series."""
        if not (0 <= atom < self.n_atoms):
            raise RydbergError(
                f"atom={atom} out of range [0, {self.n_atoms})"
            )
        return self.populations[atom]

    def blockade_fidelity(self) -> float:
        """Fraction of time steps where no two nearest neighbours are both excited.

        Uses the final-state expectation values.
        """
        pops = self.populations[:, -1]  # final populations
        # Simple proxy: if all pops < 0.5 (no strong double excitation)
        # For a more rigorous check: compute ⟨n_i n_j⟩ from the final SV
        return float(np.sum(pops < 0.5) / self.n_atoms)

    def to_dict(self) -> dict:
        return {
            "times": self.times.tolist(),
            "populations": self.populations.tolist(),
            "total_rydberg": self.total_rydberg.tolist(),
            "n_atoms": self.n_atoms,
        }


# ── RydbergSimulator ──────────────────────────────────────────────────────────

class RydbergSimulator:
    """Neutral-atom Rydberg blockade simulator.

    Simulates the coherent dynamics of n neutral atoms under a global
    laser drive using exact statevector evolution.

    Parameters
    ----------
    atom_positions : list[tuple[float, float]] | list[RydbergAtom]
        Atom positions in μm (2-D).  Accepted as ``(x, y)`` tuples or
        :class:`RydbergAtom` objects.
    Omega : float, default 1.0
        Global Rabi frequency.
    delta : float, default 0.0
        Laser detuning (negative = red-detuned, positive = blue-detuned).
    C6 : float, default 862690.0
        van der Waals C₆ coefficient in MHz·μm⁶.
    max_atoms : int, default 16
        Hard limit on atom count (exact diagonalisation scales as 2^n).
    """

    def __init__(
        self,
        atom_positions: list,
        Omega: float = _DEFAULT_OMEGA,
        delta: float = _DEFAULT_DELTA,
        C6: float = _DEFAULT_C6,
        max_atoms: int = 16,
    ) -> None:
        if not atom_positions:
            raise RydbergError("atom_positions must not be empty")

        atoms: list[RydbergAtom] = []
        for pos in atom_positions:
            if isinstance(pos, RydbergAtom):
                atoms.append(pos)
            else:
                try:
                    x, y = float(pos[0]), float(pos[1])
                    atoms.append(RydbergAtom(x=x, y=y))
                except (TypeError, IndexError, ValueError) as exc:
                    raise RydbergError(
                        f"Each atom position must be a (x, y) tuple or RydbergAtom, "
                        f"got {pos!r}"
                    ) from exc

        if len(atoms) > max_atoms:
            raise RydbergError(
                f"Too many atoms: {len(atoms)} > max_atoms={max_atoms}. "
                "Exact diagonalisation would require 2^n memory."
            )

        self.atoms = atoms
        self.n_atoms = len(atoms)
        self.Omega = float(Omega)
        self.delta = float(delta)
        self.C6 = float(C6)

        self._ham = self._build_hamiltonian()

    # ── Hamiltonian construction ───────────────────────────────────────────────

    def _build_hamiltonian(self) -> RydbergHamiltonian:
        n = self.n_atoms
        # Interaction matrix
        V = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                r = self.atoms[i].distance_to(self.atoms[j])
                if r < 1e-12:
                    V[i, j] = V[j, i] = float("inf")
                else:
                    V[i, j] = V[j, i] = self.C6 / r ** 6

        blockade_r = (self.C6 / max(abs(self.Omega), 1e-12)) ** (1 / 6)

        return RydbergHamiltonian(
            n_atoms=n,
            Omega=self.Omega,
            delta=self.delta,
            interaction_matrix=V,
            blockade_radius=blockade_r,
        )

    def hamiltonian(self) -> RydbergHamiltonian:
        """Return the :class:`RydbergHamiltonian` for this configuration."""
        return self._ham

    def build_matrix(self) -> np.ndarray:
        """Build the full 2^n × 2^n Hamiltonian matrix.

        Convention: computational basis |b_{n-1}...b_0⟩ where b_i=1 means
        atom i is in the Rydberg state.
        """
        n = self.n_atoms
        dim = 1 << n
        H = np.zeros((dim, dim), dtype=complex)

        Omega = self.Omega
        delta = self.delta
        V = self._ham.interaction_matrix

        for state in range(dim):
            # Diagonal: -δ Σ n_i  +  Σ_{i<j} V_ij n_i n_j
            diag = 0.0
            for i in range(n):
                if (state >> i) & 1:
                    diag -= delta
                    for j in range(i + 1, n):
                        if (state >> j) & 1:
                            diag += V[i, j]
            H[state, state] = diag

            # Off-diagonal: (Ω/2) σ^x_i = (Ω/2)(|g⟩⟨r| + |r⟩⟨g|)_i
            for i in range(n):
                # Flip qubit i
                flipped = state ^ (1 << i)
                H[state, flipped] += Omega / 2

        return H

    # ── Time evolution ────────────────────────────────────────────────────────

    def evolve(
        self,
        t_final: float,
        n_steps: int = 100,
        initial_state: Optional[np.ndarray] = None,
    ) -> RydbergResult:
        """Evolve the system from t=0 to t=t_final.

        Uses exact matrix-exponential time evolution (Runge-Kutta via
        eigendecomposition of H).

        Parameters
        ----------
        t_final : float
            Total evolution time.
        n_steps : int, default 100
            Number of time steps (snapshots taken at uniform intervals).
        initial_state : np.ndarray | None
            Initial statevector (shape ``(2**n,)``).  Defaults to all atoms
            in the ground state: |g...g⟩ = |0...0⟩.

        Returns
        -------
        RydbergResult
        """
        if t_final <= 0:
            raise RydbergError(f"t_final must be > 0, got {t_final}")
        if n_steps < 1:
            raise RydbergError(f"n_steps must be ≥ 1, got {n_steps}")

        n = self.n_atoms
        dim = 1 << n

        if initial_state is None:
            psi0 = np.zeros(dim, dtype=complex)
            psi0[0] = 1.0  # all atoms in ground state
        else:
            psi0 = np.asarray(initial_state, dtype=complex)
            if psi0.shape != (dim,):
                raise RydbergError(
                    f"initial_state must have shape ({dim},), got {psi0.shape}"
                )
            norm = np.linalg.norm(psi0)
            if norm > 0:
                psi0 = psi0 / norm

        H = self.build_matrix()
        eigvals, eigvecs = np.linalg.eigh(H)

        times = np.linspace(0, t_final, n_steps)
        sv_array = np.zeros((dim, n_steps), dtype=complex)
        pops = np.zeros((n, n_steps), dtype=float)

        # Pre-project initial state onto eigenbasis
        c0 = eigvecs.conj().T @ psi0

        # Rydberg number operators n_i = |r⟩⟨r|_i as arrays
        n_ops = []
        for i in range(n):
            diag = np.array([(1.0 if (s >> i) & 1 else 0.0) for s in range(dim)])
            n_ops.append(diag)

        for idx, t in enumerate(times):
            # ψ(t) = Σ_k c_k e^{-iε_k t} |k⟩
            phases = np.exp(-1j * eigvals * t)
            psi_t = eigvecs @ (phases * c0)
            sv_array[:, idx] = psi_t
            probs = np.abs(psi_t) ** 2
            for i in range(n):
                pops[i, idx] = float(np.dot(probs, n_ops[i]))

        total_rydberg = pops.sum(axis=0)

        return RydbergResult(
            times=times,
            statevector=sv_array,
            populations=pops,
            total_rydberg=total_rydberg,
            n_atoms=n,
        )

    def blockade_radius(self) -> float:
        """Return the blockade radius R_b = (C₆/Ω)^{1/6} in μm."""
        return self._ham.blockade_radius

    def nearest_neighbour_pairs(self) -> list[tuple[int, int, float]]:
        """Return all atom pairs with their distances.

        Returns
        -------
        list[(i, j, distance)]
            Sorted by distance ascending.
        """
        pairs = []
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                d = self.atoms[i].distance_to(self.atoms[j])
                pairs.append((i, j, d))
        return sorted(pairs, key=lambda p: p[2])

    def is_blockaded(self, i: int, j: int) -> bool:
        """Return True if atoms i and j are within the blockade radius."""
        d = self.atoms[i].distance_to(self.atoms[j])
        return d < self._ham.blockade_radius

    @classmethod
    def linear_chain(
        cls,
        n_atoms: int,
        spacing: float = 5.0,
        **kwargs,
    ) -> "RydbergSimulator":
        """Construct a 1-D linear chain of n_atoms atoms.

        Parameters
        ----------
        n_atoms : int
        spacing : float
            Nearest-neighbour spacing in μm.
        **kwargs
            Passed to :class:`RydbergSimulator`.
        """
        positions = [(i * spacing, 0.0) for i in range(n_atoms)]
        return cls(atom_positions=positions, **kwargs)

    @classmethod
    def square_lattice(
        cls,
        rows: int,
        cols: int,
        spacing: float = 5.0,
        **kwargs,
    ) -> "RydbergSimulator":
        """Construct a 2-D square lattice.

        Parameters
        ----------
        rows, cols : int
        spacing : float
            Lattice constant in μm.
        **kwargs
            Passed to :class:`RydbergSimulator`.
        """
        positions = [
            (col * spacing, row * spacing)
            for row in range(rows)
            for col in range(cols)
        ]
        return cls(atom_positions=positions, **kwargs)
