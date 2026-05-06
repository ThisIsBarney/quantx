"""M110: Large-scale Clifford (stabilizer) simulation.

Implements the Aaronson-Gottesman binary symplectic tableau algorithm
[arXiv:quant-ph/0406196] which tracks an n-qubit stabilizer state in
O(n²) bits and applies every Clifford gate in O(n) time.  The algorithm
scales to 10 000+ qubits where state-vector simulation is completely
infeasible.

Classes
-------
StabilizerState
    Full Aaronson-Gottesman tableau: (2n) × (2n+1) boolean array.
    Initialised to |0...0⟩.

CliffordSimulator
    Gate-level interface wrapping ``StabilizerState``.  Supported gates::

        h(q)          Hadamard
        s(q)          Phase (S = √Z)
        sdg(q)        S†
        x(q)          Pauli-X
        y(q)          Pauli-Y
        z(q)          Pauli-Z
        cx(ctrl, tgt) CNOT
        cz(q0, q1)    CZ   (= H·CNOT·H)
        swap(q0, q1)  SWAP (= CX·CX†·CX)
        measure(q)    Single-qubit Z-measurement (returns 0 or 1,
                      collapses state randomly when outcome is
                      indeterminate)

StimCircuit
    Lightweight Stim-compatible circuit parser.  Reads the textual
    Stim circuit format and translates supported gates to
    ``CliffordSimulator`` calls::

        QUBIT_COORDS, TICK             skipped
        H, S, S_DAG, X, Y, Z          single-qubit Clifford
        CNOT, CX, CZ, SWAP            two-qubit Clifford
        M, MZ                         Z-measurement

Quick start::

    from quantx_engine.clifford_sim import CliffordSimulator, StimCircuit

    # 10 000-qubit GHZ circuit
    sim = CliffordSimulator(n_qubits=10_000)
    sim.h(0)
    for q in range(9_999):
        sim.cx(q, q + 1)

    # Measure all qubits — should all collapse to the same value
    outcomes = [sim.measure(q) for q in range(10_000)]
    assert len(set(outcomes)) == 1        # all 0s or all 1s

    # Stim circuit import
    stim_text = '''
    H 0
    CNOT 0 1
    M 0 1
    '''
    circ = StimCircuit.from_string(stim_text)
    results = circ.run()                  # returns list of measurement outcomes
"""
from __future__ import annotations

import re
import random
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CliffordError",
    "StabilizerState",
    "CliffordSimulator",
    "StimCircuit",
    "StimCircuitError",
]


# ── Exceptions ────────────────────────────────────────────────────────────────

class CliffordError(ValueError):
    """Raised for invalid Clifford simulator operations."""


class StimCircuitError(ValueError):
    """Raised for malformed or unsupported Stim circuit syntax."""


# ── Phase lookup table (AG04 Appendix A) ─────────────────────────────────────
# Index = x1*8 + z1*4 + x2*2 + z2  (each bit ∈ {0,1})
# Value = g ∈ {0,1,2,3} such that σ(x1,z1)·σ(x2,z2) = i^g · σ(x1⊕x2, z1⊕z2)
# Encoding: σ(0,0)=I, σ(1,0)=X, σ(0,1)=Z, σ(1,1)=Y
#
#  x1 z1 x2 z2  product       g
#   0  0  *  *  I·anything    0
#   1  0  0  0  X·I = X       0
#   1  0  0  1  X·Z = -iY     1  (i^1 = i? No: X·Z = -iY → coefficient is -i = i^3)
# Wait — let me be very careful here.
# Using the convention: i·Y = X·Z (since XZ = iY in standard Pauli algebra).
# So X·Z = iY → g=1? But iY means the product is i·Y, so g=1.
# Actually: X·Z = i·Y  (standard: σx σz = i σy) → g=1
# But σz σx = -i σy → g=3
# Let me re-derive from first principles:
#   X·X = I  → g=0
#   X·Y = iZ → g=1   (XY = X·iXZ = i·X²·Z = iZ, so g=1? wait: iZ not i·Y)
# Hmm, I need to use the Pauli product table more carefully.
# Using: X=(0,1;1,0), Y=(0,-i;i,0), Z=(1,0;0,-1)
# X·X = I    → coefficient 1 = i^0
# X·Y = X·(iXZ) = iX²Z = iZ → coefficient i = i^1, but result is Z not Y
# X·Z = XZ = iY (standard: σx σz = i σy)? Let's check: XZ = (0,1;1,0)(1,0;0,-1) = (0,-1;1,0) = iY?
#   iY = i(0,-i;i,0) = (0,1;-1,0)... that's not right.
#   Actually XZ = (0·1+1·0, 0·0+1·(-1); 1·1+0·0, 1·0+0·(-1)) = (0,-1;1,0)
#   And iY = i(0,-i;i,0) = (0,1;-1,0) ≠ XZ
#   ZX = (1·0+0·1, 1·1+0·0; 0·0+(-1)·1, 0·1+(-1)·0) = (0,1;-1,0) = iY
#   So XZ = -iY (not iY), and ZX = iY
#
# Let me use the standard result: for Pauli group (ignoring global phase issues),
# from AG04 directly:
# The function g(x1,z1,x2,z2) in [AG04] satisfies:
#   σ(x1,z1)·σ(x2,z2) = i^g(x1,z1,x2,z2) · σ(x1⊕x2, z1⊕z2)
# where σ(1,0)=X, σ(0,1)=Z, σ(1,1)=Y=iXZ.
# From AG04 Table II (Appendix A), the values are:
_G_LUT = np.array([
    # idx 0..3:  x1=0,z1=0 (I·*) → always 0
    0, 0, 0, 0,
    # idx 4..7:  x1=0,z1=1 (Z·*)
    # Z·I=Z(g=0), Z·Z=I(g=0), Z·X=-iY? let's use: Z·X=ZX=iY(g=1? or -iY g=3)
    # Careful: σ(0,1)=Z, σ(1,0)=X, σ(1,1)=Y
    # Z·X: result = σ(1,1)=Y, with phase g: ZX=iY means g=1?
    # Standard: ZX = (1,0;0,-1)(0,1;1,0)=(0,1;-1,0). Y=(0,-i;i,0). iY=(0,1;-1,0)=ZX ✓
    # So Z·X = iY → g=1... but wait we said XZ=-iY above? Let's recheck:
    # XZ=(0,1;1,0)(1,0;0,-1)=(0,-1;1,0). -iY=(-i)(0,-i;i,0)=(0,-1;1,0)=XZ ✓
    # So ZX=iY(g=1) and XZ=-iY(g=3). Good.
    # Z·Z=I(g=0), Z·X=iY(g=1→ no wait idx for Z·X: x1=0,z1=1,x2=1,z2=0 → idx=4+0+2+0=6
    # Z·Y: Z·(iXZ)=iZXZ=i(iY)Z=i²YZ=-YZ=-(Y·Z)... let's just compute:
    # Z·Y=(1,0;0,-1)(0,-i;i,0)=(0,-i;-i,0)=-i(0,1;1,0)=-iX → g=3
    # So for Z(x1=0,z1=1):
    # ·I(0,0)→ g=0, ·Z(0,1)→ g=0, ·X(1,0)→ g=1, ·Y(1,1)→ g=3
    # Wait: idx 4=(0,1,0,0)=Z·I, 5=(0,1,0,1)=Z·Z, 6=(0,1,1,0)=Z·X, 7=(0,1,1,1)=Z·Y
    0, 0, 1, 3,  # but above I said g(Z·X)=1... let me double-check the index mapping
    # idx 8..11: x1=1,z1=0 (X·*)
    # X·I=X(g=0), X·Z=-iY(g=3), X·X=I(g=0), X·Y=?
    # X·Y=(0,1;1,0)(0,-i;i,0)=(i,0;0,-i)=i(1,0;0,-1)=iZ → g=1
    # idx 8=(1,0,0,0)=X·I→0, 9=(1,0,0,1)=X·Z→3, 10=(1,0,1,0)=X·X→0, 11=(1,0,1,1)=X·Y→1
    0, 3, 0, 1,
    # idx 12..15: x1=1,z1=1 (Y·*)
    # Y=iXZ. Y·I=Y(g=0)
    # Y·Z=(iXZ)Z=iX(Z²)=iX → g=1? Y·Z: (0,-i;i,0)(1,0;0,-1)=(0,i;i,0)=i(0,1;1,0)=iX → g=1
    # Y·X=(iXZ)X=iX(ZX)=iX(iY)=-XY=-(iZ)=-iZ → g=3
    # Y·Y=(iXZ)(iXZ)=i²X(ZX)Z=-X(iY)Z=-iXYZ=-i(iXZ)Z=X(Z²)=X... wait
    # Simpler: Y²=(0,-i;i,0)²=((-i)(i),0;0,(i)(-i))=(1,0;0,1)=I → g=0? No:
    # Y·Y=I but Y=(0,-i;i,0), so Y²=((0)(-i)+(−i)(i),(0)(0)+(−i)(0));...
    # Y²[0,0]=0·0+(-i)·i=−i²=1, Y²[0,1]=0·0+(−i)·0=0, Y²[1,0]=i·0+0·i=0, Y²[1,1]=i·(−i)+0=1 → Y²=I
    # So Y·Y=I → g=0 if σ(1⊕1,1⊕1)=σ(0,0)=I. Check: i^0·I=I ✓. But Y·Y=(-i²)I=I...
    # Actually the coefficient: since Y·Y=I exactly, g=0.
    # But wait: in our encoding σ(1,1)=Y, so Y·Y=i^g·σ(0,0)=i^g·I
    # Since Y·Y = I (as matrices), i^g=1 → g=0.
    # Let me reconsider Y·X:
    # Y·X = (0,-i;i,0)(0,1;1,0)=(-i,0;0,i)=-i(1,0;0,-1)=-iZ → g=3? i^3=-i, so i^3·Z=-iZ ✓ → g=3
    # Hmm but earlier I said g=3 for Y·X. Let me redo Y·Z:
    # Y·Z = (0,-i;i,0)(1,0;0,-1)=(0,i;i,0)... wait: (0·1+(−i)·0, 0·0+(−i)·(−1); i·1+0·0, i·0+0·(−1))
    #      = (0, i; i, 0) = i(0,1;1,0) = iX → g=1
    # So idx 12=(1,1,0,0)=Y·I→0, 13=(1,1,0,1)=Y·Z→1, 14=(1,1,1,0)=Y·X→3, 15=(1,1,1,1)=Y·Y→2
    # Wait Y·Y: Y²=I, but what's i^g? i^g·σ(1⊕1,1⊕1)=i^g·I=I → i^g=1 → g=0.
    # But hmm: Y·Y = I exactly (since Y is Hermitian and unitary). So g=0.
    # But wait: I had g=2 originally in the old code. Let me recheck.
    # The issue is the definition of σ(1,1). Is it Y or iXZ?
    # In AG04: "We use the convention that σ(1,1) = iXZ = Y" — so σ(1,1)=Y.
    # With Y=iXZ: Y·Y = (iXZ)(iXZ) = i²·X·Z·X·Z = -·X·(ZX)·Z = -X·(iY)·Z = -iXYZ
    # = -i·X·(iXZ)·Z = -i²·X·X·Z·Z = -(−1)·1·1 = 1... OK Y·Y=I confirmed.
    # For the Pauli product in the stabilizer formalism, we need the exact phase.
    # i^g · σ(x1⊕x2, z1⊕z2) = σ(x1,z1) · σ(x2,z2)
    # Y·Y: left = Y, right = Y, product = I, and σ(0,0) = I
    # So i^g · I = I → g = 0.
    #
    # Hmm, but in many references g(Y,Y) = 2. Let me check more carefully.
    # In AG04 Appendix A (arxiv version), the table is defined differently —
    # they use a DIFFERENT convention where g is the phase exponent in the
    # product of the *unnormalized* Paulis.
    #
    # After careful review of AG04 eq. (A1):
    # g(x1,z1,x2,z2) = x2*z1*(1+2*z2+2*x1) - x1*z2*(1+2*z1+2*x2) (mod 4)
    # Let's verify a few:
    # Y·Y: (x1,z1,x2,z2)=(1,1,1,1):
    #   g = 1*1*(1+2+2) - 1*1*(1+2+2) = 5 - 5 = 0 mod 4 → g=0
    # X·Z: (1,0,0,1): g = 0*0*(1+2+2) - 1*1*(1+0+0) = 0 - 1 = -1 ≡ 3 mod 4 → g=3 ✓
    # Z·X: (0,1,1,0): g = 1*1*(1+0+0) - 0*0*(1+2+0) = 1 - 0 = 1 → g=1 ✓
    # X·Y: (1,0,1,1): g = 1*0*(1+2+2) - 1*1*(1+0+2) = 0 - 3 = -3 ≡ 1 mod 4 → g=1 ✓
    # Y·X: (1,1,1,0): g = 1*1*(1+0+2) - 1*0*(1+2+2) = 3 - 0 = 3 → g=3 ✓
    # Y·Z: (1,1,0,1): g = 0*1*(1+2+2) - 1*1*(1+2+0) = 0 - 3 = -3 ≡ 1 mod 4 → g=1 ✓
    # Z·Y: (0,1,1,1): g = 1*1*(1+2+0) - 0*1*(1+2+2) = 3 - 0 = 3 → g=3 ✓
    # Y·Y: g=0 as computed above.
    # So the correct LUT should have idx=15 → g=0.
    # But wait, Y·Y = I, so i^0 · I = I ✓. Makes sense.
    # My original code had (1,1,1,1) → g=2. That was WRONG.
    # Let me rebuild the whole LUT using the formula g = x2*z1*(1+2*z2+2*x1) - x1*z2*(1+2*z1+2*x2) mod 4:
    0, 1, 3, 0,  # (1,1,0,0)=Y·I→0, (1,1,0,1)=Y·Z→1, (1,1,1,0)=Y·X→3, (1,1,1,1)=Y·Y→0
], dtype=np.int32)


def _g(x1: int, z1: int, x2: int, z2: int) -> int:
    """Phase factor g s.t. σ(x1,z1)·σ(x2,z2) = i^g · σ(x1⊕x2, z1⊕z2).

    Uses AG04 formula (A1): g = x2*z1*(1+2*z2+2*x1) - x1*z2*(1+2*z1+2*x2)  mod 4.
    """
    return (x2 * z1 * (1 + 2 * z2 + 2 * x1) - x1 * z2 * (1 + 2 * z1 + 2 * x2)) % 4


# Rebuild LUT from the formula to ensure correctness
_G_LUT = np.array(
    [_g(x1, z1, x2, z2)
     for x1 in (0, 1) for z1 in (0, 1) for x2 in (0, 1) for z2 in (0, 1)],
    dtype=np.int32,
)
# Index layout: x1*8 + z1*4 + x2*2 + z2  (but above loop iterates x1,z1,x2,z2)
# Need to reorder for the correct index mapping x1*8+z1*4+x2*2+z2:
_G_LUT_ORDERED = np.zeros(16, dtype=np.int32)
for _x1 in range(2):
    for _z1 in range(2):
        for _x2 in range(2):
            for _z2 in range(2):
                _idx = _x1 * 8 + _z1 * 4 + _x2 * 2 + _z2
                _G_LUT_ORDERED[_idx] = _g(_x1, _z1, _x2, _z2)
_G_LUT = _G_LUT_ORDERED


# ── StabilizerState ───────────────────────────────────────────────────────────

class StabilizerState:
    """Aaronson-Gottesman binary symplectic tableau for n-qubit stabilizer states.

    Tableau layout (following [AG04] Section 4)
    -------------------------------------------
    Row indices 0..n-1   : destabilizer generators
    Row indices n..2n-1  : stabilizer generators
    Column layout per row: [x_0, …, x_{n-1}, z_0, …, z_{n-1}, r]
      where x_i, z_i ∈ {0,1} and r ∈ {0,1} is the phase (r=1 means −1 factor).

    All Clifford gates are O(n) vectorized numpy operations (no Python loops).

    Parameters
    ----------
    n_qubits : int
        Number of qubits.  The tableau shape is (2n) × (2n+1).
    """

    def __init__(self, n_qubits: int) -> None:
        if n_qubits < 1:
            raise CliffordError("n_qubits must be ≥ 1")
        n = n_qubits
        self.n = n
        # tableau[i, j] for i in 0..2n-1, j in 0..2n (last col = phase r)
        self._tab = np.zeros((2 * n, 2 * n + 1), dtype=np.uint8)
        # Initialise to |0...0⟩:
        #   destabiliser i: X_i  → x[i,i] = 1
        #   stabiliser i:   Z_i  → z[i,i] = 1  (row n+i)
        idx = np.arange(n)
        self._tab[idx, idx] = 1            # destabilisers: X_i
        self._tab[n + idx, n + idx] = 1    # stabilisers:   Z_i

    # ── Row multiply helper (AG04 rowsum) ─────────────────────────────────────

    def _rowmult(self, h: int, i: int) -> None:
        """In-place: row[h] ←  row[h] · row[i].  O(n) vectorized."""
        tab = self._tab
        n = self.n
        x1 = tab[h, :n].astype(np.int32)
        z1 = tab[h, n:2 * n].astype(np.int32)
        x2 = tab[i, :n].astype(np.int32)
        z2 = tab[i, n:2 * n].astype(np.int32)
        idx = x1 * 8 + z1 * 4 + x2 * 2 + z2
        g_sum = int(_G_LUT[idx].sum()) % 4
        tab[h, 2 * n] = (int(tab[h, 2 * n]) ^ int(tab[i, 2 * n]) ^
                         (1 if g_sum >= 2 else 0))
        tab[h, :2 * n] ^= tab[i, :2 * n]

    def _scratch_rowmult(self, scratch: np.ndarray, row: np.ndarray) -> None:
        """Multiply scratch Pauli row by *row* in-place.  O(n) vectorized."""
        n = self.n
        x1 = scratch[:n].astype(np.int32)
        z1 = scratch[n:2 * n].astype(np.int32)
        x2 = row[:n].astype(np.int32)
        z2 = row[n:2 * n].astype(np.int32)
        idx = x1 * 8 + z1 * 4 + x2 * 2 + z2
        g_sum = int(_G_LUT[idx].sum()) % 4
        scratch[2 * n] = (int(scratch[2 * n]) ^ int(row[2 * n]) ^
                          (1 if g_sum >= 2 else 0))
        scratch[:2 * n] ^= row[:2 * n]

    # ── Clifford gate implementations (AG04 Table 1, Figure 4) ───────────────
    # All are fully vectorized: single numpy operation on the whole column.

    def h(self, qubit: int) -> None:
        """Hadamard gate on *qubit*."""
        self._check_qubit(qubit)
        n, tab, a = self.n, self._tab, qubit
        tab[:, 2 * n] ^= tab[:, a] & tab[:, n + a]
        tmp = tab[:, a].copy()
        tab[:, a] = tab[:, n + a]
        tab[:, n + a] = tmp

    def s(self, qubit: int) -> None:
        """S gate (√Z) on *qubit*."""
        self._check_qubit(qubit)
        n, tab, a = self.n, self._tab, qubit
        tab[:, 2 * n] ^= tab[:, a] & tab[:, n + a]
        tab[:, n + a] ^= tab[:, a]

    def sdg(self, qubit: int) -> None:
        """S† gate on *qubit*."""
        self.s(qubit)
        self.s(qubit)
        self.s(qubit)

    def x(self, qubit: int) -> None:
        """Pauli-X gate on *qubit*."""
        self.h(qubit)
        self.s(qubit)
        self.s(qubit)
        self.h(qubit)

    def y(self, qubit: int) -> None:
        """Pauli-Y gate on *qubit*."""
        self.x(qubit)
        self.z(qubit)

    def z(self, qubit: int) -> None:
        """Pauli-Z gate on *qubit*."""
        self.s(qubit)
        self.s(qubit)

    def cx(self, ctrl: int, target: int) -> None:
        """CNOT gate."""
        if ctrl == target:
            raise CliffordError(f"CNOT ctrl and target must differ: got {ctrl}")
        self._check_qubit(ctrl)
        self._check_qubit(target)
        n, tab, a, b = self.n, self._tab, ctrl, target
        tab[:, 2 * n] ^= (tab[:, a] & tab[:, n + b] &
                           (tab[:, b] ^ tab[:, n + a] ^ np.uint8(1)))
        tab[:, b] ^= tab[:, a]
        tab[:, n + a] ^= tab[:, n + b]

    def cz(self, q0: int, q1: int) -> None:
        """CZ gate."""
        self.h(q1)
        self.cx(q0, q1)
        self.h(q1)

    def swap(self, q0: int, q1: int) -> None:
        """SWAP gate."""
        self.cx(q0, q1)
        self.cx(q1, q0)
        self.cx(q0, q1)

    # ── Measurement (AG04 Algorithm 1) ────────────────────────────────────────

    def measure(self, qubit: int, rng: random.Random) -> int:
        """Measure *qubit* in the Z basis, collapsing the state.

        Parameters
        ----------
        qubit : int
        rng : random.Random
            Caller-supplied RNG advanced once per indeterminate measurement.
            Pass a shared, persistent instance to get independent draws across
            multiple ``measure()`` calls.

        Returns
        -------
        int — 0 or 1.
        """
        self._check_qubit(qubit)
        n, tab, a = self.n, self._tab, qubit

        # Find any stabiliser generator anti-commuting with Z_a (x_{p,a}=1)
        stab_rows = np.where(tab[n:2 * n, a])[0]  # offsets within stabilisers

        if stab_rows.size > 0:
            # ── Random / indeterminate case ──────────────────────────────────
            p = int(stab_rows[0]) + n          # chosen stabiliser row index

            # For every other row (destab + stab) with x_a=1, rowmult by row p
            all_anticommute = np.where(tab[:, a])[0]
            for ii in all_anticommute:
                if ii != p:
                    self._rowmult(int(ii), p)

            # Set destabiliser p-n := current stabiliser p
            tab[p - n] = tab[p].copy()
            # Overwrite stabiliser p with ±Z_a (random phase from shared RNG)
            tab[p] = 0
            outcome = rng.randint(0, 1)
            tab[p, 2 * n] = np.uint8(outcome)
            tab[p, n + a] = np.uint8(1)
            return outcome

        else:
            # ── Determinate case ─────────────────────────────────────────────
            # For each destabilizer d_i with x_a=1, accumulate paired stabilizer
            scratch = np.zeros(2 * n + 1, dtype=np.uint8)
            destab_anticommute = np.where(tab[:n, a])[0]
            for i in destab_anticommute:
                self._scratch_rowmult(scratch, tab[int(i) + n])
            return int(scratch[2 * n])

    # ── Utilities ─────────────────────────────────────────────────────────────

    def copy(self) -> "StabilizerState":
        """Return a deep copy of this tableau."""
        new = StabilizerState.__new__(StabilizerState)
        new.n = self.n
        new._tab = self._tab.copy()
        return new

    def _check_qubit(self, q: int) -> None:
        if not (0 <= q < self.n):
            raise CliffordError(f"Qubit {q} out of range [0, {self.n})")

    @property
    def n_qubits(self) -> int:
        return self.n

    def __repr__(self) -> str:  # pragma: no cover
        return f"StabilizerState(n_qubits={self.n})"


# ── Convenience function (also exposed in __all__) ────────────────────────────

def _g_table(x1: int, z1: int, x2: int, z2: int) -> int:
    """Return g ∈ {0,1,2,3} s.t. σ(x1,z1)·σ(x2,z2) = i^g · σ(x1⊕x2, z1⊕z2)."""
    return int(_G_LUT[x1 * 8 + z1 * 4 + x2 * 2 + z2])


# ── CliffordSimulator ─────────────────────────────────────────────────────────

class CliffordSimulator:
    """Gate-level Clifford circuit simulator using the Aaronson-Gottesman tableau.

    All operations run in O(n) time; the full state fits in O(n²) bits,
    enabling 10 000+ qubit simulation on commodity hardware.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.  Initialised to |0...0⟩.
    seed : int | None
        Default RNG seed for indeterminate measurements.
    """

    def __init__(self, n_qubits: int, seed: int | None = None) -> None:
        self._state = StabilizerState(n_qubits)
        # One persistent RNG per simulator instance.  Seeded once at construction
        # so that successive measurements advance the same generator — independent
        # qubits in product superpositions get independent (uncorrelated) draws.
        self._rng: random.Random = random.Random(seed)
        self._measurement_outcomes: list[int] = []

    # ── Single-qubit gates ────────────────────────────────────────────────────

    def h(self, qubit: int) -> "CliffordSimulator":
        self._state.h(qubit); return self

    def s(self, qubit: int) -> "CliffordSimulator":
        self._state.s(qubit); return self

    def sdg(self, qubit: int) -> "CliffordSimulator":
        self._state.sdg(qubit); return self

    def x(self, qubit: int) -> "CliffordSimulator":
        self._state.x(qubit); return self

    def y(self, qubit: int) -> "CliffordSimulator":
        self._state.y(qubit); return self

    def z(self, qubit: int) -> "CliffordSimulator":
        self._state.z(qubit); return self

    # ── Two-qubit gates ───────────────────────────────────────────────────────

    def cx(self, ctrl: int, target: int) -> "CliffordSimulator":
        self._state.cx(ctrl, target); return self

    def cz(self, q0: int, q1: int) -> "CliffordSimulator":
        self._state.cz(q0, q1); return self

    def swap(self, q0: int, q1: int) -> "CliffordSimulator":
        self._state.swap(q0, q1); return self

    # ── Measurement ───────────────────────────────────────────────────────────

    def measure(self, qubit: int, seed: int | None = None) -> int:
        """Measure *qubit* in the Z basis.  Returns 0 or 1.

        Parameters
        ----------
        qubit : int
        seed : int | None
            If provided, the simulator's RNG is re-seeded with this value
            *before* this measurement.  Useful for reproducible individual
            measurements; for reproducible multi-qubit circuits, seed the
            simulator at construction time instead.
        """
        if seed is not None:
            self._rng.seed(seed)
        outcome = self._state.measure(qubit, rng=self._rng)
        self._measurement_outcomes.append(outcome)
        return outcome

    def measure_all(self, seed: int | None = None) -> list[int]:
        """Measure all qubits in order using the simulator's shared RNG.

        Parameters
        ----------
        seed : int | None
            If provided, re-seeds the simulator's RNG once before the first
            measurement (not once per qubit), so successive measurements get
            independent draws from the same seeded sequence.
        """
        if seed is not None:
            self._rng.seed(seed)
        return [self._state.measure(q, rng=self._rng) for q in range(self.n_qubits)]

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def n_qubits(self) -> int:
        return self._state.n

    @property
    def state(self) -> StabilizerState:
        return self._state

    @property
    def measurement_outcomes(self) -> list[int]:
        return list(self._measurement_outcomes)

    def __repr__(self) -> str:  # pragma: no cover
        return f"CliffordSimulator(n_qubits={self.n_qubits})"


# ── Stim circuit parser ───────────────────────────────────────────────────────

_STIM_SINGLE: dict[str, str | None] = {
    "H": "h", "H_XY": "h", "H_YZ": "h",
    "S": "s", "S_DAG": "sdg", "SQRT_Z": "s", "SQRT_Z_DAG": "sdg",
    "X": "x", "Y": "y", "Z": "z",
    "I": None,   # identity — no-op
}
_STIM_TWO: dict[str, str] = {
    "CNOT": "cx", "CX": "cx",
    "CZ": "cz",
    "SWAP": "swap",
}
_STIM_MEASURE: set[str] = {"M", "MZ", "MEASURE"}
_STIM_SKIP: set[str] = {
    "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE",
    "SHIFT_COORDS", "TICK", "REPEAT",
}


@dataclass
class StimCircuit:
    """Parsed Stim-format circuit ready to simulate with ``CliffordSimulator``.

    Attributes
    ----------
    n_qubits : int
        Inferred from the largest qubit index in the circuit.
    instructions : list[tuple]
        Sequence of ``(gate_name, [qubit_indices])``.
    """

    n_qubits: int
    instructions: list[tuple[str, list[int]]] = field(default_factory=list)

    @classmethod
    def from_string(cls, text: str) -> "StimCircuit":
        """Parse a Stim circuit from *text*."""
        instructions: list[tuple[str, list[int]]] = []
        max_qubit = 0

        for lineno, raw_line in enumerate(text.splitlines(), start=1):
            line = re.sub(r"#.*$", "", raw_line).strip()
            if not line:
                continue
            line = re.sub(r"\([^)]*\)", "", line)
            parts = line.split()
            if not parts:
                continue

            gate = parts[0].upper()
            try:
                operands = [int(p) for p in parts[1:]]
            except ValueError as exc:
                raise StimCircuitError(
                    f"Line {lineno}: non-integer qubit index in {raw_line!r}"
                ) from exc

            if gate in _STIM_SKIP:
                continue

            if gate in _STIM_SINGLE:
                if not operands:
                    raise StimCircuitError(
                        f"Line {lineno}: {gate} requires at least one qubit operand"
                    )
                for q in operands:
                    instructions.append((gate, [q]))
                    max_qubit = max(max_qubit, q)

            elif gate in _STIM_TWO:
                if len(operands) < 2 or len(operands) % 2 != 0:
                    raise StimCircuitError(
                        f"Line {lineno}: {gate} requires an even number of qubit operands"
                    )
                for k in range(0, len(operands), 2):
                    q0, q1 = operands[k], operands[k + 1]
                    instructions.append((gate, [q0, q1]))
                    max_qubit = max(max_qubit, q0, q1)

            elif gate in _STIM_MEASURE:
                if not operands:
                    raise StimCircuitError(
                        f"Line {lineno}: {gate} requires at least one qubit operand"
                    )
                for q in operands:
                    instructions.append((gate, [q]))
                    max_qubit = max(max_qubit, q)

            else:
                raise StimCircuitError(
                    f"Line {lineno}: unsupported Stim gate {gate!r}."
                )

        return cls(
            n_qubits=max_qubit + 1 if instructions else 1,
            instructions=instructions,
        )

    @classmethod
    def from_file(cls, path: str) -> "StimCircuit":
        """Parse a Stim circuit from *path*."""
        with open(path, encoding="utf-8") as fh:
            return cls.from_string(fh.read())

    def run(self, seed: int | None = None) -> list[int]:
        """Execute the circuit on a fresh :class:`CliffordSimulator`.

        Returns all measurement outcomes in order as a list of ints.
        """
        sim = CliffordSimulator(n_qubits=self.n_qubits, seed=seed)
        outcomes: list[int] = []

        for gate, qubits in self.instructions:
            if gate in _STIM_SINGLE:
                method_name = _STIM_SINGLE[gate]
                if method_name is not None:
                    getattr(sim, method_name)(qubits[0])
            elif gate in _STIM_TWO:
                method_name = _STIM_TWO[gate]
                getattr(sim, method_name)(qubits[0], qubits[1])
            elif gate in _STIM_MEASURE:
                # Drive through the simulator's shared RNG (no per-call seed).
                outcomes.append(sim.measure(qubits[0]))

        return outcomes
