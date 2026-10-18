"""Randomized Benchmarking and Quantum Volume (M81).

Provides a pure-NumPy implementation of:
    - StandardRB: 1-qubit randomized benchmarking using the full Clifford
      group; fits the exponential decay P(m) = A·p^m + B and extracts the
      Error Per Clifford (EPC).
    - QuantumVolume: IBM quantum-volume protocol for n qubits; evaluates
      the heavy-output-generation probability and returns the QV number.

Both solvers use exact density-matrix simulation with a global depolarizing
noise channel; no external framework is required.

Quick reference::

    from quantx_engine.contrib.rb_qv import StandardRB, QuantumVolume
    result = StandardRB(p_error=0.01).run([1, 5, 10, 20, 50], n_sequences=30)
    print(result.epc)          # ~ 0.01/2

    qv = QuantumVolume(p_error=0.005)
    qv_results = qv.run(max_qubits=3, n_trials=100)
    print(qv_results[-1].quantum_volume)  # largest passing QV

References
----------
.. [1] Knill, E. et al. (2008). Randomized benchmarking of quantum gates.
       *Phys. Rev. A* 77, 012307.
       https://doi.org/10.1103/PhysRevA.77.012307

.. [2] Cross, A. W. et al. (2019). Validating quantum computers using
       randomized model circuits. *Phys. Rev. A* 100, 032328.
       https://doi.org/10.1103/PhysRevA.100.032328

.. [3] Magesan, E., Gambetta, J. M. & Emerson, J. (2011). Scalable and
       robust randomized benchmarking of quantum processes.
       *Phys. Rev. Lett.* 106, 180504.
       https://doi.org/10.1103/PhysRevLett.106.180504

Physical validation
-------------------
Standard RB: EPC (error per Clifford gate) extracted from exponential decay
A + B*p^m, where p = 1 - EPC/(1 - 1/d^2), d = 2^n [Magesan2011].
Quantum Volume: QV = 2^m where m is the largest m s.t. the heavy-output
probability of m-qubit random circuits exceeds 2/3 [Cross2019].
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

__all__ = [
    "RBQVError",
    "RBResult",
    "StandardRB",
    "QVResult",
    "QuantumVolume",
]


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class RBQVError(ValueError):
    """Raised for invalid inputs to RB or QV experiments."""


def _require_numpy() -> None:
    if not _HAS_NUMPY:  # pragma: no cover
        raise RBQVError("NumPy is required for RB/QV experiments.")


# ---------------------------------------------------------------------------
# Single-qubit Clifford group (24 elements)
# ---------------------------------------------------------------------------

def _generate_cliffords_1q() -> "list[np.ndarray]":
    """Return all 24 single-qubit Clifford matrices (BFS from H and S).

    Two unitaries are considered distinct only if they differ by more than
    a global phase.
    """
    H = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2)
    S = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex)

    def _phase_key(U: "np.ndarray") -> tuple:
        """Phase-normalised fingerprint: divide by the largest-magnitude entry."""
        v = U.ravel()
        idx = int(np.argmax(np.abs(v)))
        ref = v[idx]
        if abs(ref) < 1e-12:
            return tuple(np.round(v, 8).tolist())
        normed = (v / ref)
        return tuple(np.round(normed.real, 8).tolist() + np.round(normed.imag, 8).tolist())

    result: "list[np.ndarray]" = [np.eye(2, dtype=complex)]
    seen: "set" = {_phase_key(np.eye(2, dtype=complex))}
    queue: "list[np.ndarray]" = [np.eye(2, dtype=complex)]

    while queue and len(result) < 24:
        U = queue.pop(0)
        for G in (H, S):
            V = G @ U
            fk = _phase_key(V)
            if fk not in seen:
                seen.add(fk)
                result.append(V)
                queue.append(V)

    return result[:24]


# Cache on first call
_CLIFFORDS_1Q: "list[np.ndarray] | None" = None


def _get_cliffords_1q() -> "list[np.ndarray]":
    global _CLIFFORDS_1Q
    if _CLIFFORDS_1Q is None:
        _CLIFFORDS_1Q = _generate_cliffords_1q()
    return _CLIFFORDS_1Q


# ---------------------------------------------------------------------------
# Haar-random unitary (for QV and multi-qubit RB)
# ---------------------------------------------------------------------------

def _haar_unitary(d: int, rng: "np.random.Generator") -> "np.ndarray":
    """Return a Haar-random unitary of dimension d."""
    Z = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    Q, R = np.linalg.qr(Z)
    diag = np.diag(R)
    phase = diag / np.abs(diag)
    return Q * phase  # correct QR phase to get Haar measure


# ---------------------------------------------------------------------------
# Density-matrix helpers
# ---------------------------------------------------------------------------

def _dm_apply_gate(
    rho: "np.ndarray",
    V: "np.ndarray",
) -> "np.ndarray":
    """Apply unitary V to density matrix: ρ ← V ρ V†."""
    return V @ rho @ V.conj().T


def _dm_depolarize(
    rho: "np.ndarray",
    p: float,
) -> "np.ndarray":
    """Apply global depolarizing channel: ρ ← (1−p)ρ + p/d I."""
    d = rho.shape[0]
    return (1.0 - p) * rho + p / d * np.eye(d, dtype=complex)


def _embed_gate(
    U: "np.ndarray",
    n: int,
    target_qubits: "tuple[int, ...]",
) -> "np.ndarray":
    """Embed a gate U acting on `target_qubits` into the full n-qubit space.

    Qubits are indexed 0 (MSB) … n-1 (LSB).

    Parameters
    ----------
    U:
        Gate matrix of shape ``(2^k, 2^k)`` where k = len(target_qubits).
    n:
        Total number of qubits.
    target_qubits:
        Tuple of qubit indices the gate acts on (in order from MSB to LSB
        of the gate's own index).
    """
    k = len(target_qubits)
    d = 1 << n
    V = np.zeros((d, d), dtype=complex)
    bits = list(range(n))  # all bit positions

    for row in range(d):
        for col in range(d):
            # Bits of row and col for each qubit (MSB = qubit 0)
            row_bits = [(row >> (n - 1 - q)) & 1 for q in range(n)]
            col_bits = [(col >> (n - 1 - q)) & 1 for q in range(n)]
            # Non-target qubits must match
            if any(
                row_bits[q] != col_bits[q]
                for q in range(n)
                if q not in target_qubits
            ):
                continue
            # Target qubit states → gate indices
            row_gate = sum(
                row_bits[target_qubits[i]] << (k - 1 - i) for i in range(k)
            )
            col_gate = sum(
                col_bits[target_qubits[i]] << (k - 1 - i) for i in range(k)
            )
            V[row, col] = U[row_gate, col_gate]

    return V


# ---------------------------------------------------------------------------
# Exponential fit
# ---------------------------------------------------------------------------

def _fit_rb(
    m_vals: "np.ndarray",
    p_obs: "np.ndarray",
) -> "tuple[float, float, float]":
    """Fit P(m) = A·p^m + B via 1-D grid search over p + linear regression.

    Returns
    -------
    p, A, B : floats
    """
    m_arr = np.asarray(m_vals, dtype=float)
    p_arr = np.asarray(p_obs, dtype=float)

    best: "tuple[float, float, float, float] | None" = None

    # Grid spans [0, 1] so edge cases (noiseless p≈1 or fully-depolarised p≈0)
    # are reachable; np.linalg.lstsq handles degenerate columns gracefully.
    for p_try in np.linspace(0.0, 1.0, 600):
        phi = np.column_stack([p_try ** m_arr, np.ones_like(m_arr)])
        sol, _, _, _ = np.linalg.lstsq(phi, p_arr, rcond=None)
        A_try, B_try = float(sol[0]), float(sol[1])
        residual = float(np.sum((A_try * p_try ** m_arr + B_try - p_arr) ** 2))
        if best is None or residual < best[0]:
            best = (residual, p_try, A_try, B_try)

    if best is None:  # unreachable in practice, but safer than assert
        raise RBQVError("_fit_rb: grid search produced no candidates.")
    return best[1], best[2], best[3]  # p, A, B


# ===========================================================================
# StandardRB
# ===========================================================================

@dataclass
class RBResult:
    """Result of a Randomized Benchmarking experiment.

    Attributes
    ----------
    sequence_lengths:
        Clifford sequence lengths m tested.
    survival_probs:
        Mean survival probability at each sequence length.
    p_decay:
        Exponential decay constant p from P(m) = A·p^m + B.
    A:
        Amplitude fit parameter.
    B:
        Offset fit parameter (≈ 1/d² for fully depolarised case).
    epc:
        Error Per Clifford = (d−1)(1−p)/d  (d = 2^n_qubits).
    n_qubits:
        Number of qubits used in the experiment.
    """
    sequence_lengths: "list[int]"
    survival_probs: "list[float]"
    p_decay: float
    A: float
    B: float
    epc: float
    n_qubits: int

    @property
    def r_squared(self) -> float:
        """Coefficient of determination R² for the exponential fit A·p^m + B."""
        import numpy as _np
        m_arr = _np.array(self.sequence_lengths, dtype=float)
        p_obs = _np.array(self.survival_probs)
        p_fit = self.A * self.p_decay ** m_arr + self.B
        ss_res = float(_np.sum((p_obs - p_fit) ** 2))
        ss_tot = float(_np.sum((p_obs - p_obs.mean()) ** 2))
        if ss_tot < 1e-14:
            return 1.0
        return max(0.0, 1.0 - ss_res / ss_tot)


class StandardRB:
    """Standard randomized benchmarking.

    Generates random Clifford sequences of lengths ``m``, appends the
    composite inverse, and estimates the survival probability via exact
    density-matrix simulation with a global depolarizing channel.

    Currently supports 1-qubit RB using the full 24-element Clifford group
    and n-qubit RB using Haar-random unitaries (approximate RB).

    Parameters
    ----------
    n_qubits:
        Number of qubits (1 for standard single-qubit RB).
    p_error:
        Depolarizing error probability per Clifford gate.  0 = noiseless.
    seed:
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        p_error: float = 0.001,
        seed: "int | None" = None,
        *,
        gate_error: "float | None" = None,
    ) -> None:
        _require_numpy()
        if gate_error is not None:
            p_error = gate_error  # alias
        if n_qubits < 1:
            raise RBQVError(f"n_qubits must be ≥ 1, got {n_qubits}.")
        if not (0.0 <= p_error < 1.0):
            raise RBQVError(f"p_error must be in [0, 1), got {p_error}.")
        self.n_qubits = n_qubits
        self.p_error = p_error
        self.seed = seed

    def run(
        self,
        m_values: "list[int] | None" = None,
        n_sequences: int = 30,
        *,
        sequence_lengths: "list[int] | None" = None,
    ) -> RBResult:
        """Run the RB experiment.

        Parameters
        ----------
        m_values:
            List of sequence lengths to evaluate.  Each must be ≥ 1.
        n_sequences:
            Number of random sequences to average at each length.

        Returns
        -------
        RBResult
        """
        _require_numpy()
        if sequence_lengths is not None and m_values is None:
            m_values = sequence_lengths  # alias
        if m_values is None:
            raise RBQVError("m_values (or sequence_lengths) must be provided.")
        m_values = list(m_values)
        if not m_values:
            raise RBQVError("m_values must not be empty.")
        if any(m < 1 for m in m_values):
            raise RBQVError("All sequence lengths must be ≥ 1.")
        if n_sequences < 1:
            raise RBQVError("n_sequences must be ≥ 1.")

        rng = np.random.default_rng(self.seed)
        n = self.n_qubits
        d = 1 << n
        use_clifford = (n == 1)
        if use_clifford:
            cliffords = _get_cliffords_1q()

        survival_probs: "list[float]" = []

        for m in m_values:
            total = 0.0
            for _ in range(n_sequences):
                # --- Build random sequence ---
                if use_clifford:
                    idx_seq = rng.integers(0, 24, size=m)
                    gates = [cliffords[i] for i in idx_seq]
                    # Inverse = product of conjugate transposes in reverse order
                    composite = np.eye(d, dtype=complex)
                    for G in gates:
                        composite = G @ composite
                    inv_gate = composite.conj().T
                else:
                    # Haar random unitaries for n > 1
                    gates = [_haar_unitary(d, rng) for _ in range(m)]
                    composite = np.eye(d, dtype=complex)
                    for G in gates:
                        composite = G @ composite
                    inv_gate = composite.conj().T

                # --- Density-matrix simulation ---
                rho = np.zeros((d, d), dtype=complex)
                rho[0, 0] = 1.0  # |0…0⟩⟨0…0|

                for G in gates:
                    rho = _dm_apply_gate(rho, G)
                    if self.p_error > 0.0:
                        rho = _dm_depolarize(rho, self.p_error)

                # Apply composite inverse (also subject to per-Clifford noise)
                rho = _dm_apply_gate(rho, inv_gate)
                if self.p_error > 0.0:
                    rho = _dm_depolarize(rho, self.p_error)

                # Survival probability = ⟨0|ρ|0⟩
                total += float(np.real(rho[0, 0]))

            survival_probs.append(total / n_sequences)

        # Fit P(m) = A·p^m + B
        m_arr = np.array(m_values, dtype=float)
        p_arr = np.array(survival_probs)
        p_fit, A_fit, B_fit = _fit_rb(m_arr, p_arr)

        # Error Per Clifford: EPC = (d-1)/d * (1 - p)
        epc = (d - 1) / d * (1.0 - p_fit)

        return RBResult(
            sequence_lengths=m_values,
            survival_probs=survival_probs,
            p_decay=float(p_fit),
            A=float(A_fit),
            B=float(B_fit),
            epc=float(epc),
            n_qubits=n,
        )


# ===========================================================================
# Quantum Volume
# ===========================================================================

@dataclass
class QVResult:
    """Result of a single-depth Quantum Volume experiment.

    Attributes
    ----------
    n_qubits:
        Number of qubits (= circuit depth).
    heavy_output_prob:
        Estimated heavy-output generation probability (mean over trials).
    passed:
        True if ``heavy_output_prob > 2/3``.
    quantum_volume:
        ``2**n_qubits`` if *passed*, else 0.
    n_trials:
        Number of random-circuit trials used.
    """
    n_qubits: int
    heavy_output_prob: float
    passed: bool
    quantum_volume: int
    n_trials: int

    @property
    def heavy_output_probability(self) -> float:
        """Alias for heavy_output_prob."""
        return self.heavy_output_prob


class QuantumVolume:
    """IBM Quantum Volume protocol.

    For each width n (qubits), runs n-qubit random circuits of depth n
    (alternating layers of Haar-random 2-qubit unitaries), computes the
    heavy-output generation probability (HOGP), and declares success if
    HOGP > 2/3.

    Parameters
    ----------
    p_error:
        Depolarizing error rate per 2-qubit gate.  0 = noiseless.
    seed:
        Optional random seed.
    """

    def __init__(
        self,
        p_error: float = 0.001,
        seed: "int | None" = None,
        *,
        n_qubits: "int | None" = None,
        gate_error: "float | None" = None,
    ) -> None:
        _require_numpy()
        if gate_error is not None:
            p_error = gate_error  # alias
        if not (0.0 <= p_error < 1.0):
            raise RBQVError(f"p_error must be in [0, 1), got {p_error}.")
        self.p_error = p_error
        self.seed = seed
        self._fixed_n_qubits = n_qubits  # if set, run() targets this width

    def _run_single(
        self,
        n: int,
        n_trials: int,
        rng: "np.random.Generator",
    ) -> QVResult:
        """Run QV experiment for n qubits, n depth."""
        d = 1 << n
        hogp_sum = 0.0

        for _ in range(n_trials):
            # --- Ideal circuit (noiseless) ---
            rho_ideal = np.zeros((d, d), dtype=complex)
            rho_ideal[0, 0] = 1.0

            # --- Noisy circuit ---
            rho_noisy = rho_ideal.copy()

            # n layers of random 2-qubit unitaries
            for _layer in range(n):
                # Random permutation of qubit pairs
                perm = rng.permutation(n)
                for k in range(0, n - 1, 2):
                    q1, q2 = int(perm[k]), int(perm[k + 1])
                    U2 = _haar_unitary(4, rng)
                    V = _embed_gate(U2, n, (q1, q2))
                    # Apply the *same* random gate to both ideal and noisy
                    rho_ideal = _dm_apply_gate(rho_ideal, V)
                    rho_noisy = _dm_apply_gate(rho_noisy, V)
                    if self.p_error > 0.0:
                        rho_noisy = _dm_depolarize(rho_noisy, self.p_error)

            # Ideal probabilities (diagonal of ρ_ideal)
            p_ideal = np.real(np.diag(rho_ideal))
            p_ideal = np.maximum(p_ideal, 0.0)

            # Heavy outputs: those with P_ideal > median(P_ideal)
            median_val = float(np.median(p_ideal))
            heavy_mask = p_ideal > median_val

            # Noisy probabilities
            p_noisy = np.real(np.diag(rho_noisy))
            p_noisy = np.maximum(p_noisy, 0.0)

            # Heavy output generation probability (HOGP)
            hogp_sum += float(np.sum(p_noisy[heavy_mask]))

        hogp = hogp_sum / n_trials
        passed = hogp > 2.0 / 3.0

        return QVResult(
            n_qubits=n,
            heavy_output_prob=float(hogp),
            passed=passed,
            quantum_volume=int(1 << n) if passed else 0,
            n_trials=n_trials,
        )

    def run(
        self,
        max_qubits: int = 4,
        n_trials: int = 100,
        *,
        n_circuits: "int | None" = None,
    ) -> "list[QVResult] | QVResult":
        """Run QV experiments for n = 1 … max_qubits.

        If the instance was constructed with ``n_qubits=k``, runs a single
        experiment at width k and returns a single :class:`QVResult`.
        Otherwise returns a list of results (one per width 1..max_qubits).

        Parameters
        ----------
        max_qubits:
            Maximum circuit width to test.
        n_trials:
            Number of random-circuit trials per width.

        Returns
        -------
        List of :class:`QVResult`, one per width.
        """
        _require_numpy()
        if n_circuits is not None:
            n_trials = n_circuits  # alias
        if n_trials < 1:
            raise RBQVError("n_trials must be ≥ 1.")

        rng = np.random.default_rng(self.seed)

        # Single-width mode: when n_qubits was specified at construction
        if self._fixed_n_qubits is not None:
            n = self._fixed_n_qubits
            if n < 1:
                raise RBQVError(f"n_qubits must be ≥ 1, got {n}.")
            return self._run_single(n, n_trials, rng)

        # Multi-width sweep mode
        if max_qubits < 1:
            raise RBQVError(f"max_qubits must be ≥ 1, got {max_qubits}.")
        results: "list[QVResult]" = []
        for n in range(1, max_qubits + 1):
            results.append(self._run_single(n, n_trials, rng))
        return results

    def quantum_volume_number(
        self,
        max_qubits: int = 4,
        n_trials: int = 100,
    ) -> int:
        """Return the quantum volume number (largest 2^n that passes).

        Returns 0 if no width passes the HOGP > 2/3 criterion.
        """
        results = self.run(max_qubits=max_qubits, n_trials=n_trials)
        qv = 0
        for r in results:
            if r.passed:
                qv = max(qv, r.quantum_volume)
        return qv
