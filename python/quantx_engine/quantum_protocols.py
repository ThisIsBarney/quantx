"""Quantum communication protocols — M97.

Implements four landmark quantum protocols built on the quantum network
infrastructure from M96 (:mod:`quantum_network`):

* **BB84** — Bennett–Brassard 1984 QKD with optional eavesdropping.
* **E91**  — Ekert 1991 entanglement-based QKD with CHSH Bell test.
* **Teleportation** — Quantum state teleportation over an entangled channel.
* **SuperDenseCoding** — Two classical bits per one qubit transmission.

All protocols are purely classical/statistical simulations of the relevant
quantum information flows; no wavefunction is stored.  Fidelity parameters
from :mod:`quantum_network` are propagated throughout.

Classes
-------
* :class:`ProtocolError` — base exception.
* :class:`BB84Result` — result of one BB84 key-exchange run.
* :class:`BB84Protocol` — BB84 key distribution simulator.
* :class:`E91Result` — result of one E91 run.
* :class:`E91Protocol` — E91 / Ekert-91 protocol simulator.
* :class:`TeleportationResult` — result of a teleportation run.
* :class:`TeleportationProtocol` — quantum teleportation simulator.
* :class:`DenseCodingResult` — result of a dense coding run.
* :class:`DenseCodingProtocol` — superdense coding simulator.

Functions
---------
* :func:`bb84_key_rate` — theoretical BB84 secure key fraction.
* :func:`chsh_inequality_value` — CHSH S-parameter from correlations.
* :func:`teleportation_fidelity` — fidelity of teleported state given channel F.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quantx_engine.quantum_network import (
    NetworkError,
    fidelity_after_dephasing,
    fidelity_after_swapping,
    simulate_repeater_chain,
    _binary_entropy,
    _secret_key_rate,
)


__all__ = [
    "ProtocolError",
    "BB84Result",
    "BB84Protocol",
    "E91Result",
    "E91Protocol",
    "TeleportationResult",
    "TeleportationProtocol",
    "DenseCodingResult",
    "DenseCodingProtocol",
    "bb84_key_rate",
    "chsh_inequality_value",
    "teleportation_fidelity",
]


# ── Exception ─────────────────────────────────────────────────────────────────

class ProtocolError(ValueError):
    """Raised on invalid protocol configuration or execution."""


# ── Utility: binary entropy / Pauli helpers ───────────────────────────────────

def bb84_key_rate(qber: float) -> float:
    """Asymptotic BB84 secure key rate per sifted bit.

    Uses the Shor–Preskill formula::

        r = 1 - 2·h₂(QBER)

    where h₂ is the binary Shannon entropy.  Returns 0 if QBER ≥ 11 %.

    Parameters
    ----------
    qber : float ∈ [0, 0.5]
        Quantum Bit Error Rate after sifting.

    Returns
    -------
    float ∈ [0, 1]
    """
    if not 0.0 <= qber <= 0.5:
        raise ProtocolError(f"qber must be in [0, 0.5], got {qber}")
    r = 1.0 - 2.0 * _binary_entropy(qber)
    return float(max(0.0, r))


def chsh_inequality_value(
    correlations: dict[tuple[int, int], float],
) -> float:
    """Compute the CHSH S-parameter from four correlation measurements.

    The CHSH inequality is S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|.
    Classical bound: |S| ≤ 2.  Tsirelson's quantum bound: |S| ≤ 2√2 ≈ 2.828.

    Parameters
    ----------
    correlations : dict mapping (alice_basis, bob_basis) → E ∈ [-1, 1]
        Must contain keys (0,0), (0,1), (1,0), (1,1).

    Returns
    -------
    float  (S-parameter)
    """
    required = {(0, 0), (0, 1), (1, 0), (1, 1)}
    if not required.issubset(correlations):
        raise ProtocolError(
            f"correlations must contain keys {required}, got {set(correlations)}"
        )
    E00 = correlations[(0, 0)]
    E01 = correlations[(0, 1)]
    E10 = correlations[(1, 0)]
    E11 = correlations[(1, 1)]
    S = E00 - E01 + E10 + E11
    return float(abs(S))


def teleportation_fidelity(channel_fidelity: float) -> float:
    """Average fidelity of teleported qubit state as function of channel F.

    For a Werner-state channel with Bell-pair fidelity F, the average
    teleportation fidelity over the Bloch sphere is::

        f_tele = (2F + 1) / 3

    This saturates at f = 1 for F = 1 (perfect channel) and gives
    f = 1/2 for F = 1/4 (maximally mixed channel, same as classical guess).

    Parameters
    ----------
    channel_fidelity : float ∈ [0, 1]

    Returns
    -------
    float ∈ [1/3, 1]
    """
    if not 0.0 <= channel_fidelity <= 1.0:
        raise ProtocolError(
            f"channel_fidelity must be in [0,1], got {channel_fidelity}"
        )
    return float(np.clip((2.0 * channel_fidelity + 1.0) / 3.0, 0.0, 1.0))


# ── BB84 Protocol ─────────────────────────────────────────────────────────────

@dataclass
class BB84Result:
    """Result of a BB84 key exchange simulation.

    Attributes
    ----------
    n_raw_bits : int
        Total bits sent by Alice.
    n_sifted_bits : int
        Bits remaining after basis reconciliation.
    n_secure_bits : int
        Bits remaining after privacy amplification / error correction.
    qber : float
        Quantum Bit Error Rate in the sifted key.
    secure_key_rate : float
        Fraction of sifted bits that become secure key (Shor–Preskill).
    eve_detected : bool
        Whether Eve's presence was statistically detected (QBER > threshold).
    raw_key_alice : list[int]
        Sifted key bits from Alice's perspective (before reconciliation).
    raw_key_bob : list[int]
        Sifted key bits from Bob's perspective.
    metadata : dict
    """
    n_raw_bits: int
    n_sifted_bits: int
    n_secure_bits: int
    qber: float
    secure_key_rate: float
    eve_detected: bool
    raw_key_alice: list[int]
    raw_key_bob: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"BB84Result(raw={self.n_raw_bits}, sifted={self.n_sifted_bits}, "
            f"secure={self.n_secure_bits}, QBER={self.qber:.3f}, "
            f"eve={self.eve_detected})"
        )


class BB84Protocol:
    """Simulate the BB84 quantum key distribution protocol.

    Alice encodes random bits in randomly chosen bases (rectilinear Z or
    diagonal X).  Bob measures in randomly chosen bases.  After classical
    sifting, only bits where bases matched are kept.  Noise (channel error
    rate) and an optional intercept-resend eavesdropper (Eve) are modelled
    statistically.

    Parameters
    ----------
    channel_error_rate : float ∈ [0, 0.5]
        Intrinsic QBER from channel noise (no Eve).
    eve_intercept_fraction : float ∈ [0, 1]
        Fraction of qubits that Eve intercepts and re-sends.
        Eve introduces an additional QBER of ``0.25 * eve_intercept_fraction``.
    detection_threshold : float
        QBER threshold above which Eve is considered detected (default: 0.11).
    seed : int or None
    """

    def __init__(
        self,
        channel_error_rate: float = 0.02,
        eve_intercept_fraction: float = 0.0,
        detection_threshold: float = 0.11,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= channel_error_rate <= 0.5:
            raise ProtocolError(
                f"channel_error_rate must be in [0, 0.5], got {channel_error_rate}"
            )
        if not 0.0 <= eve_intercept_fraction <= 1.0:
            raise ProtocolError(
                f"eve_intercept_fraction must be in [0, 1], got {eve_intercept_fraction}"
            )
        if not 0.0 <= detection_threshold <= 0.5:
            raise ProtocolError(
                f"detection_threshold must be in [0, 0.5], got {detection_threshold}"
            )
        self.channel_error_rate = channel_error_rate
        self.eve_intercept_fraction = eve_intercept_fraction
        self.detection_threshold = detection_threshold
        self._rng = np.random.default_rng(seed)

    def run(self, n_bits: int = 1000) -> BB84Result:
        """Execute one BB84 protocol run.

        Parameters
        ----------
        n_bits : int
            Number of raw qubits Alice sends.

        Returns
        -------
        BB84Result
        """
        if n_bits < 1:
            raise ProtocolError(f"n_bits must be >= 1, got {n_bits}")

        rng = self._rng
        # Alice: random bits and bases
        alice_bits = rng.integers(0, 2, size=n_bits)
        alice_bases = rng.integers(0, 2, size=n_bits)  # 0=Z, 1=X

        # Eve: intercept-resend attack on a fraction of qubits
        # Eve measures in a random basis; if she guesses wrong (prob 1/2),
        # she disturbs the state, introducing QBER of 1/2 * 1/2 = 0.25 per
        # intercepted qubit on the channel.
        eve_mask = rng.random(n_bits) < self.eve_intercept_fraction
        # Eve's measurement: random basis, records her result
        # After re-send in Alice's original encoding, if Bob's basis = Alice's
        # but Eve's basis ≠ Alice's: Bob has 50% chance of error → avg 25%
        # contribution to total QBER from Eve
        eve_qber_contribution = 0.25 * self.eve_intercept_fraction

        # Bob: random bases
        bob_bases = rng.integers(0, 2, size=n_bits)

        # Sifting: keep bits where Alice and Bob chose the same basis
        matching = alice_bases == bob_bases
        sifted_alice = alice_bits[matching]
        sifted_bob_bases = bob_bases[matching]  # same as alice_bases here
        _ = sifted_bob_bases

        # Bob's received bits: flip with channel_error_rate + Eve contribution
        total_error_rate = self.channel_error_rate + eve_qber_contribution
        total_error_rate = min(total_error_rate, 0.5)
        flip_mask = rng.random(len(sifted_alice)) < total_error_rate
        sifted_bob = sifted_alice.copy()
        sifted_bob[flip_mask] = 1 - sifted_bob[flip_mask]

        n_sifted = len(sifted_alice)
        # Measured QBER on a sample check (sample half for error estimation)
        n_check = max(1, n_sifted // 2)
        errors_in_check = int(np.sum(sifted_alice[:n_check] != sifted_bob[:n_check]))
        measured_qber = errors_in_check / n_check if n_check > 0 else 0.0

        # Secure key rate
        skr = bb84_key_rate(measured_qber)
        # Secure bits come from the remaining (non-check) portion
        n_key_bits = n_sifted - n_check
        n_secure = max(0, int(n_key_bits * skr))

        alice_key = sifted_alice[n_check:].tolist()
        bob_key = sifted_bob[n_check:].tolist()

        return BB84Result(
            n_raw_bits=n_bits,
            n_sifted_bits=n_sifted,
            n_secure_bits=n_secure,
            qber=measured_qber,
            secure_key_rate=skr,
            eve_detected=measured_qber > self.detection_threshold,
            raw_key_alice=alice_key,
            raw_key_bob=bob_key,
            metadata={
                "channel_error_rate": self.channel_error_rate,
                "eve_intercept_fraction": self.eve_intercept_fraction,
                "eve_qber_contribution": eve_qber_contribution,
                "detection_threshold": self.detection_threshold,
            },
        )


# ── E91 Protocol ──────────────────────────────────────────────────────────────

@dataclass
class E91Result:
    """Result of an E91 entanglement-based QKD run.

    Attributes
    ----------
    n_pairs : int
        Number of Bell pairs consumed.
    n_key_bits : int
        Number of key bits extracted.
    chsh_s : float
        Measured CHSH S-parameter (quantum: ≤ 2√2, classical: ≤ 2).
    bell_violation : bool
        True if S > 2 (quantum correlations confirmed).
    qber : float
        Bit error rate in the key subset.
    secure_key_rate : float
        Shor–Preskill secure key fraction.
    metadata : dict
    """
    n_pairs: int
    n_key_bits: int
    chsh_s: float
    bell_violation: bool
    qber: float
    secure_key_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"E91Result(pairs={self.n_pairs}, key_bits={self.n_key_bits}, "
            f"S={self.chsh_s:.4f}, bell_violation={self.bell_violation}, "
            f"QBER={self.qber:.3f})"
        )


class E91Protocol:
    """Simulate the Ekert-91 entanglement-based QKD protocol.

    Alice and Bob share Bell pairs from a source.  Each independently chooses
    among three measurement angles.  Pairs where they chose the same angle
    contribute to the key; pairs with different angles are used to compute
    the CHSH S-parameter as a security test.

    The channel is characterised by a Bell-pair fidelity *F*.  Under the
    Werner state model, the correlation function for ideal angles is::

        E(θ_A, θ_B) = -(2F − 1/2) cos(2(θ_A − θ_B))

    (derived from Werner state correlations for the singlet component).

    Parameters
    ----------
    bell_pair_fidelity : float ∈ [0.25, 1]
        Fidelity of the entangled pairs shared between Alice and Bob.
    channel_error_rate : float ∈ [0, 0.5]
        Additional bit-flip error on top of Werner-state correlations.
    seed : int or None
    """

    # Optimal CHSH angles for |Φ+⟩ where E(θ_A, θ_B) = p·cos(2(θ_A − θ_B)):
    #   Alice: 0, π/8, π/4
    #   Bob:   π/8, 3π/8, 5π/8
    # Key bits: both choose π/8 (alice_idx=1, bob_idx=0) — same angle.
    # CHSH test: (alice 0, bob 0/1) and (alice 2, bob 0/1)
    #   S = E(0,π/8) − E(0,3π/8) + E(π/4,π/8) + E(π/4,3π/8)
    #     = cos(π/4) + cos(3π/4) + cos(π/4) + cos(π/4) ... = 2√2 for p=1
    _ALICE_ANGLES = [0.0, math.pi / 8, math.pi / 4]
    _BOB_ANGLES = [math.pi / 8, 3 * math.pi / 8, 5 * math.pi / 8]

    def __init__(
        self,
        bell_pair_fidelity: float = 0.95,
        channel_error_rate: float = 0.01,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= bell_pair_fidelity <= 1.0:
            raise ProtocolError(
                f"bell_pair_fidelity must be in [0,1], got {bell_pair_fidelity}"
            )
        if not 0.0 <= channel_error_rate <= 0.5:
            raise ProtocolError(
                f"channel_error_rate must be in [0, 0.5], got {channel_error_rate}"
            )
        self.bell_pair_fidelity = bell_pair_fidelity
        self.channel_error_rate = channel_error_rate
        self._rng = np.random.default_rng(seed)

    def _correlation(self, theta_a: float, theta_b: float) -> float:
        """Werner-state correlation: E(θ_A, θ_B) = −(4F−1) cos(2(θ_A−θ_B)) / 3."""
        F = self.bell_pair_fidelity
        # Werner state singlet component gives: E = -(4F-1)/3 * cos(2 delta)
        # For F=1 (pure singlet |Φ-⟩ variant) E = -cos(2Δ)
        # Here we use |Φ+⟩ convention: E = +(4F-1)/3 * cos(2 delta)
        p_werner = (4.0 * F - 1.0) / 3.0
        delta = theta_a - theta_b
        return p_werner * math.cos(2.0 * delta)

    def run(self, n_pairs: int = 2000) -> E91Result:
        """Run the E91 protocol.

        Parameters
        ----------
        n_pairs : int
            Number of Bell pairs to consume.

        Returns
        -------
        E91Result
        """
        if n_pairs < 4:
            raise ProtocolError(f"n_pairs must be >= 4, got {n_pairs}")

        rng = self._rng
        alice_choice = rng.integers(0, 3, size=n_pairs)
        bob_choice = rng.integers(0, 3, size=n_pairs)

        alice_angles = np.array(self._ALICE_ANGLES)[alice_choice]
        bob_angles = np.array(self._BOB_ANGLES)[bob_choice]

        # Generate outcomes using the Werner-state correlation model
        # P(same result) = (1 - E(θ_A, θ_B)) / 2  for ±1 outcomes
        alice_outcomes = rng.integers(0, 2, size=n_pairs) * 2 - 1  # ±1
        # Bob's outcomes correlated with Alice's via E(θ_A, θ_B)
        bob_outcomes = np.empty(n_pairs, dtype=int)
        for i in range(n_pairs):
            E = self._correlation(alice_angles[i], bob_angles[i])
            # P(alice_outcome == bob_outcome) = (1 + E) / 2
            p_agree = (1.0 + E) / 2.0
            p_agree = float(np.clip(p_agree, 0.0, 1.0))
            if rng.random() < p_agree:
                bob_outcomes[i] = alice_outcomes[i]
            else:
                bob_outcomes[i] = -alice_outcomes[i]
            # Additional channel error
            if rng.random() < self.channel_error_rate:
                bob_outcomes[i] = -bob_outcomes[i]

        # Key bits: pairs where Alice chose index 1 (π/8) and Bob chose index 0 (π/8)
        # Both measure at the same angle π/8 → perfectly correlated outcomes for pure channel
        key_mask = (alice_choice == 1) & (bob_choice == 0)
        alice_key = ((alice_outcomes[key_mask] + 1) // 2).tolist()
        bob_key = ((bob_outcomes[key_mask] + 1) // 2).tolist()
        n_key = len(alice_key)

        # QBER in key subset
        if n_key > 0:
            errors = int(np.sum(np.array(alice_key) != np.array(bob_key)))
            qber = errors / n_key
        else:
            qber = 0.0

        # CHSH: use four angle combinations for optimal violation with |Φ+⟩
        # Map to our angle indices:
        #   (0,0): Alice=0 (0),    Bob=0 (π/8)  → E = cos(-π/4)  =  √2/2
        #   (0,1): Alice=0 (0),    Bob=1 (3π/8) → E = cos(-3π/4) = -√2/2
        #   (1,0): Alice=2 (π/4),  Bob=0 (π/8)  → E = cos(π/4)   =  √2/2
        #   (1,1): Alice=2 (π/4),  Bob=1 (3π/8) → E = cos(-π/4)  =  √2/2
        # S = E(0,0) - E(0,1) + E(1,0) + E(1,1) = 2√2 (Tsirelson bound)
        chsh_masks = {
            (0, 0): (alice_choice == 0) & (bob_choice == 0),
            (0, 1): (alice_choice == 0) & (bob_choice == 1),
            (1, 0): (alice_choice == 2) & (bob_choice == 0),
            (1, 1): (alice_choice == 2) & (bob_choice == 1),
        }
        chsh_angles = {
            (0, 0): (self._ALICE_ANGLES[0], self._BOB_ANGLES[0]),
            (0, 1): (self._ALICE_ANGLES[0], self._BOB_ANGLES[1]),
            (1, 0): (self._ALICE_ANGLES[2], self._BOB_ANGLES[0]),
            (1, 1): (self._ALICE_ANGLES[2], self._BOB_ANGLES[1]),
        }
        correlations: dict[tuple[int, int], float] = {}
        for key_pair, mask in chsh_masks.items():
            if np.sum(mask) > 0:
                a_sub = alice_outcomes[mask]
                b_sub = bob_outcomes[mask]
                correlations[key_pair] = float(np.mean(a_sub * b_sub))
            else:
                # Fallback: theoretical correlation scaled by the same noise
                # factor applied in the sampling loop.  Each channel_error_rate
                # flip negates Bob's outcome, attenuating correlations by
                # (1 − 2·e).  Without this, the fallback overestimates |S|.
                ta, tb = chsh_angles[key_pair]
                noise_scale = 1.0 - 2.0 * self.channel_error_rate
                correlations[key_pair] = noise_scale * self._correlation(ta, tb)

        S = chsh_inequality_value(correlations)
        bell_violation = S > 2.0

        # QBER can statistically exceed 0.5 with very low fidelity states;
        # clamp to [0, 0.5] before computing key rate (negative rate → 0).
        skr = bb84_key_rate(min(qber, 0.5))

        return E91Result(
            n_pairs=n_pairs,
            n_key_bits=n_key,
            chsh_s=S,
            bell_violation=bell_violation,
            qber=qber,
            secure_key_rate=skr,
            metadata={
                "bell_pair_fidelity": self.bell_pair_fidelity,
                "channel_error_rate": self.channel_error_rate,
                "n_key_pairs": n_key,
                "correlations": {str(k): v for k, v in correlations.items()},
            },
        )


# ── Quantum Teleportation ──────────────────────────────────────────────────────

@dataclass
class TeleportationResult:
    """Result of a quantum teleportation run.

    Attributes
    ----------
    n_teleported : int
        Number of states teleported.
    average_fidelity : float
        Average fidelity of the teleported states relative to the inputs.
    channel_fidelity : float
        Bell-pair fidelity of the shared channel.
    classical_bits_sent : int
        Total classical bits used for corrections (2 per teleportation).
    success_rate : float
        Fraction of teleportations that achieved fidelity > 2/3 (classical limit).
    metadata : dict
    """
    n_teleported: int
    average_fidelity: float
    channel_fidelity: float
    classical_bits_sent: int
    success_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"TeleportationResult(n={self.n_teleported}, "
            f"avg_fidelity={self.average_fidelity:.4f}, "
            f"channel_F={self.channel_fidelity:.4f})"
        )


class TeleportationProtocol:
    """Simulate quantum state teleportation.

    Alice and Bob share a Bell pair with fidelity *F*.  Alice performs a BSM
    on her qubit and the input state, communicates 2 classical bits to Bob,
    and Bob applies the corresponding Pauli correction.

    The average fidelity of the teleported state (over the Bloch sphere) is::

        f_tele = (2F + 1) / 3

    This exceeds the classical ``f = 2/3`` bound iff ``F > 1/2``.

    Parameters
    ----------
    channel_fidelity : float ∈ [0.25, 1]
        Bell-pair fidelity shared between Alice and Bob.
    bsm_fidelity : float ∈ (0, 1]
        Alice's BSM gate fidelity.
    seed : int or None
    """

    def __init__(
        self,
        channel_fidelity: float = 0.95,
        bsm_fidelity: float = 0.99,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= channel_fidelity <= 1.0:
            raise ProtocolError(
                f"channel_fidelity must be in [0,1], got {channel_fidelity}"
            )
        if not 0.0 < bsm_fidelity <= 1.0:
            raise ProtocolError(
                f"bsm_fidelity must be in (0,1], got {bsm_fidelity}"
            )
        self.channel_fidelity = channel_fidelity
        self.bsm_fidelity = bsm_fidelity
        self._rng = np.random.default_rng(seed)

    def run(self, n_states: int = 100) -> TeleportationResult:
        """Teleport n_states random qubit states.

        Parameters
        ----------
        n_states : int
            Number of qubit states to teleport.

        Returns
        -------
        TeleportationResult
        """
        if n_states < 1:
            raise ProtocolError(f"n_states must be >= 1, got {n_states}")

        # Effective channel fidelity after BSM imperfection
        # BSM imperfection depolarises the channel similarly to a gate error
        F_eff = fidelity_after_swapping(
            self.channel_fidelity, 1.0, self.bsm_fidelity
        )

        # Theoretical average fidelity (Werner-state teleportation channel)
        f_avg = teleportation_fidelity(F_eff)

        # Per-state stochastic model consistent with f_avg = (2F+1)/3 for all F.
        #
        # Treat each teleportation as a Bernoulli trial:
        #   - "success"  (prob f_avg)     → state transferred perfectly, f = 1
        #   - "failure"  (prob 1 − f_avg) → state completely lost,       f = 0
        #
        # E[f_i] = f_avg                               ✓ for all F ∈ [0, 1]
        # P(f > 2/3) = P(success) = f_avg              ✓ only when F > 0.5
        #
        # Unlike the previous {1, 0.5} model, this remains consistent with
        # theoretical_avg_fidelity even for F < 0.25 where the Werner
        # mixture parameter p = (4F−1)/3 is negative (unphysical depolarizer).
        rng = self._rng
        good_events = rng.random(n_states) < f_avg
        per_state_fidelities = np.where(good_events, 1.0, 0.0)
        avg_fidelity = float(np.mean(per_state_fidelities))

        # Classical limit is 2/3; count successes above it
        n_success = int(np.sum(per_state_fidelities > 2.0 / 3.0))
        success_rate = n_success / n_states

        return TeleportationResult(
            n_teleported=n_states,
            average_fidelity=avg_fidelity,
            channel_fidelity=self.channel_fidelity,
            classical_bits_sent=2 * n_states,
            success_rate=success_rate,
            metadata={
                "bsm_fidelity": self.bsm_fidelity,
                "effective_channel_fidelity": F_eff,
                "theoretical_avg_fidelity": f_avg,
                "classical_fidelity_limit": 2.0 / 3.0,
            },
        )


# ── Superdense Coding ─────────────────────────────────────────────────────────

@dataclass
class DenseCodingResult:
    """Result of a superdense coding simulation.

    Attributes
    ----------
    n_messages : int
        Number of 2-bit messages sent.
    n_bits_sent_quantum : int
        Number of qubits transmitted over the quantum channel.
    n_classical_bits_encoded : int
        Total classical bits encoded (2 per qubit).
    channel_capacity : float
        Effective capacity in bits/qubit (2 for perfect channel).
    error_rate : float
        Fraction of messages decoded incorrectly due to channel noise.
    metadata : dict
    """
    n_messages: int
    n_bits_sent_quantum: int
    n_classical_bits_encoded: int
    channel_capacity: float
    error_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"DenseCodingResult(msgs={self.n_messages}, "
            f"capacity={self.channel_capacity:.4f} bits/qubit, "
            f"error_rate={self.error_rate:.4f})"
        )


class DenseCodingProtocol:
    """Simulate superdense coding.

    Alice encodes 2 classical bits into one qubit by applying a Pauli
    operation to her half of a shared Bell pair, then sends the qubit to Bob.
    Bob performs a BSM to decode both bits.

    The channel capacity in the presence of Werner-state noise F is::

        C = 1 + (4F - 1)/3 · 1 + (1-(4F-1)/3) · 0
          = 1 + (4F-1)/3   [bits/qubit, approximate for Werner noise]

    More precisely, the Holevo capacity of the dense coding channel is::

        χ = h(F + (1-F)/4) - h((1-F)/4) * 3   [nats, Werner model]

    We use the simpler error-rate model: probability of decoding error from
    Werner noise is ``(1 - (4F-1)/3) * 3/4``.

    Parameters
    ----------
    channel_fidelity : float ∈ [0.25, 1]
        Bell-pair fidelity of the shared channel.
    seed : int or None
    """

    def __init__(
        self,
        channel_fidelity: float = 0.95,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= channel_fidelity <= 1.0:
            raise ProtocolError(
                f"channel_fidelity must be in [0,1], got {channel_fidelity}"
            )
        self.channel_fidelity = channel_fidelity
        self._rng = np.random.default_rng(seed)

    def _message_error_prob(self) -> float:
        """Probability that one 2-bit message is decoded incorrectly.

        Under Werner noise, the Bell state is disturbed with probability
        (1 - p_werner) where p_werner = (4F-1)/3.  Any non-identity Pauli
        component causes a decoding error.
        """
        p_werner = (4.0 * self.channel_fidelity - 1.0) / 3.0
        p_werner = float(np.clip(p_werner, 0.0, 1.0))
        # Prob of Pauli error (X, Y, or Z each with prob (1-p_werner)/4)
        p_error = 1.0 - p_werner  # total depolarising probability
        # If any Pauli fires on the encoded qubit: message incorrect
        # (simplified: ignores that some Pauli errors may map to same message)
        return float(np.clip(p_error * 3.0 / 4.0, 0.0, 1.0))

    def run(self, n_messages: int = 500) -> DenseCodingResult:
        """Run superdense coding for n_messages 2-bit messages.

        Parameters
        ----------
        n_messages : int
            Number of 2-bit messages to send.

        Returns
        -------
        DenseCodingResult
        """
        if n_messages < 1:
            raise ProtocolError(f"n_messages must be >= 1, got {n_messages}")

        rng = self._rng
        p_err = self._message_error_prob()

        # Generate random messages and simulate decoding errors
        messages_sent = rng.integers(0, 4, size=n_messages)  # 00, 01, 10, 11
        error_mask = rng.random(n_messages) < p_err
        messages_received = messages_sent.copy()
        # Errors map the message to a different (random) 2-bit value
        wrong_msgs = rng.integers(0, 4, size=n_messages)
        # Ensure wrong message != correct one
        same = wrong_msgs == messages_sent
        wrong_msgs[same] = (wrong_msgs[same] + 1) % 4
        messages_received[error_mask] = wrong_msgs[error_mask]

        n_errors = int(np.sum(messages_sent != messages_received))
        error_rate = n_errors / n_messages

        # Capacity: in bits per qubit
        # Perfect: 2 bits/qubit. With errors:
        # C ≈ 2 - h(p_err) - p_err * log2(3)  [channel capacity approx]
        if p_err <= 0:
            capacity = 2.0
        elif p_err >= 0.75:
            capacity = 0.0
        else:
            # Mutual information I(X;Y) for a symmetric 4-ary channel
            # with error prob p_err (maps to any other symbol uniformly)
            p_correct = 1.0 - p_err
            p_other = p_err / 3.0
            h_out = -p_correct * math.log2(p_correct) - 3.0 * p_other * math.log2(p_other) if p_other > 0 else 0.0
            h_cond = -(
                p_correct * math.log2(p_correct)
                + 3.0 * p_other * math.log2(p_other)
            ) if p_other > 0 else 0.0
            # I(X;Y) = H(Y) - H(Y|X) = log2(4) - h_cond
            capacity = float(max(0.0, 2.0 - h_cond))

        return DenseCodingResult(
            n_messages=n_messages,
            n_bits_sent_quantum=n_messages,  # one qubit per 2-bit message
            n_classical_bits_encoded=2 * n_messages,
            channel_capacity=capacity,
            error_rate=error_rate,
            metadata={
                "channel_fidelity": self.channel_fidelity,
                "theoretical_error_prob": p_err,
            },
        )
