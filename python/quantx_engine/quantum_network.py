"""Quantum network simulation — M96.

Models quantum communication networks with:

* :class:`QuantumNode` — a network endpoint holding quantum memories.
* :class:`QuantumChannel` — a lossy, noisy optical channel between two nodes.
* :class:`QuantumNetwork` — a graph of nodes and channels.
* :class:`BellPairGenerator` — generates (possibly imperfect) Bell pairs.
* :class:`EntanglementSwapper` — performs BSM-based entanglement swapping.
* :class:`QuantumRepeater` — single-hop repeater with entanglement swapping.
* :class:`NetworkResult` — output container.

Physical model
--------------
Channels are characterised by:

* **Transmission** η ∈ [0, 1] — probability that a photon arrives.
* **Dephasing rate** p_deph ∈ [0, 1] — per-qubit dephasing probability of
  stored memories.
* **Gate fidelity** F_gate ∈ [0, 1] — Bell-state-measurement (BSM) fidelity.

Bell pair quality is tracked as a fidelity *F* relative to the ideal singlet
|Φ⁺⟩ = (|00⟩ + |11⟩)/√2.  The fidelity of a Werner state ρ = F|Φ⁺⟩⟨Φ⁺| +
(1−F)/4 · I₄ satisfies:

* Generation:   F_gen = η/2 + (1−η)/4   (photon loss → mixed state).
* Dephasing:    F_deph = F(1 − p) + (1−F)(p/3) [isotropic approx].
* Swapping:     F_swap(A,B) = F_A·F_B + (1−F_A)(1−F_B)/3.
* Purification: F_pur(F) = F²/(F² + (1−F)²/3) [one round].

Classes
-------
* :class:`NetworkError` — base exception.
* :class:`QuantumNode` — network node / quantum memory register.
* :class:`QuantumChannel` — noisy lossy channel link.
* :class:`QuantumNetwork` — network topology.
* :class:`BellPairGenerator` — Bell-pair source (inline or heralded).
* :class:`EntanglementSwapper` — BSM-based entanglement swapping.
* :class:`QuantumRepeater` — one repeater hop.
* :class:`NetworkResult` — simulation output.

Functions
---------
* :func:`fidelity_after_dephasing` — Werner-state dephasing formula.
* :func:`fidelity_after_swapping` — entanglement swapping fidelity.
* :func:`fidelity_after_purification` — one-round DEJMPS purification.
* :func:`simulate_repeater_chain` — end-to-end fidelity for N-hop chain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


__all__ = [
    "NetworkError",
    "QuantumNode",
    "QuantumChannel",
    "QuantumNetwork",
    "BellPairGenerator",
    "EntanglementSwapper",
    "QuantumRepeater",
    "NetworkResult",
    "fidelity_after_dephasing",
    "fidelity_after_swapping",
    "fidelity_after_purification",
    "simulate_repeater_chain",
]


# ── Exception ─────────────────────────────────────────────────────────────────

class NetworkError(ValueError):
    """Raised on invalid quantum network configurations."""


# ── Werner state helpers ──────────────────────────────────────────────────────

def _werner_state(F: float) -> np.ndarray:
    """Build a 2-qubit Werner state with Bell-pair fidelity F.

    The state is parameterised so that ⟨Φ⁺|ρ|Φ⁺⟩ = F exactly::

        ρ = p|Φ⁺⟩⟨Φ⁺| + (1−p)/4 · I₄,   p = (4F − 1) / 3

    At F = 1  → ρ = |Φ⁺⟩⟨Φ⁺| (maximally entangled).
    At F = 1/4 → p = 0 → ρ = I₄/4 (maximally mixed).

    Parameters
    ----------
    F : float ∈ [1/4, 1]

    Returns
    -------
    ndarray, shape (4, 4)
    """
    phi_plus = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    proj = np.outer(phi_plus, phi_plus.conj())
    p = (4.0 * F - 1.0) / 3.0          # mixture weight ∈ [0, 1] for F ∈ [1/4, 1]
    p = float(np.clip(p, 0.0, 1.0))
    return p * proj + (1.0 - p) / 4.0 * np.eye(4, dtype=complex)


def _fidelity_to_werner(rho: np.ndarray) -> float:
    """Extract Werner fidelity F = ⟨Φ⁺|ρ|Φ⁺⟩."""
    phi_plus = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
    return float(np.real(phi_plus.conj() @ rho @ phi_plus))


def fidelity_after_dephasing(F: float, p_deph: float) -> float:
    """Werner fidelity after pair-level depolarising noise.

    Models a two-qubit depolarising channel that with probability p_deph
    replaces the Bell pair with the maximally mixed state, and with
    probability (1 − p_deph) leaves it unchanged::

        F' = (1 − p) F + p / 4

    This is the exact Werner-state map for isotropic two-qubit noise with
    depolarising parameter p ∈ [0, 1].  Key properties:

    * p = 0  → identity (F' = F).
    * p = 1  → fully depolarised (F' = 1/4).
    * F = 1/4 is a fixed point for all p.

    Parameters
    ----------
    F : float ∈ [0, 1]
        Input Werner fidelity.
    p_deph : float ∈ [0, 1]
        Two-qubit depolarising probability.

    Returns
    -------
    float ∈ [0, 1]
    """
    if not 0.0 <= F <= 1.0:
        raise NetworkError(f"F must be in [0,1], got {F}")
    if not 0.0 <= p_deph <= 1.0:
        raise NetworkError(f"p_deph must be in [0,1], got {p_deph}")
    F_new = (1.0 - p_deph) * F + p_deph / 4.0
    return float(np.clip(F_new, 0.0, 1.0))


def fidelity_after_swapping(F_A: float, F_B: float, F_gate: float = 1.0) -> float:
    """Fidelity after BSM-based entanglement swapping.

    Given two Werner states with fidelities F_A and F_B and a BSM with
    fidelity F_gate, the output Werner-state fidelity is::

        F_ideal = F_A · F_B + (1 − F_A)(1 − F_B) / 3
        F_swap  = F_gate · F_ideal + (1 − F_gate) / 4

    This is the exact formula derived from the Werner-state parameterisation
    p = (4F − 1)/3 and the multiplicative property p_out = p_A · p_B under
    a perfect BSM.  Key properties:

    * F_A = 1 → F_swap = F_B (swapping a perfect pair preserves the other).
    * F_A = F_B = 1/4 → F_swap = 1/4 (maximally mixed is a fixed point).

    Parameters
    ----------
    F_A, F_B : float ∈ [0, 1]
        Fidelities of the two Bell pairs to be swapped.
    F_gate : float ∈ [0, 1]
        BSM gate fidelity (1 = perfect BSM).

    Returns
    -------
    float ∈ [0, 1]
    """
    for name, val in [("F_A", F_A), ("F_B", F_B), ("F_gate", F_gate)]:
        if not 0.0 <= val <= 1.0:
            raise NetworkError(f"{name} must be in [0,1], got {val}")
    # Exact Werner-state ideal-swap formula: p_out = p_A * p_B
    # → F_out = (3 p_A p_B + 1) / 4 = F_A F_B + (1-F_A)(1-F_B) / 3
    F_ideal = F_A * F_B + (1.0 - F_A) * (1.0 - F_B) / 3.0
    # Imperfect BSM depolarises the output
    F_out = F_gate * F_ideal + (1.0 - F_gate) / 4.0
    return float(np.clip(F_out, 0.0, 1.0))


def fidelity_after_purification(F: float, n_rounds: int = 1) -> float:
    """Fidelity after n rounds of DEJMPS entanglement purification.

    Each round of DEJMPS doubles the Bell-state probability vs.
    depolarisation noise::

        F' = F² / (F² + (1−F)²/3)

    Parameters
    ----------
    F : float ∈ [0, 1]
        Input Werner fidelity.
    n_rounds : int
        Number of purification rounds (each round consumes one extra pair).

    Returns
    -------
    float ∈ [0, 1]
    """
    if not 0.0 <= F <= 1.0:
        raise NetworkError(f"F must be in [0,1], got {F}")
    if n_rounds < 0:
        raise NetworkError(f"n_rounds must be non-negative, got {n_rounds}")
    for _ in range(n_rounds):
        denom = F ** 2 + (1.0 - F) ** 2 / 3.0
        if denom < 1e-30:
            return F
        F = F ** 2 / denom
    return float(np.clip(F, 0.0, 1.0))


# ── QuantumNode ───────────────────────────────────────────────────────────────

class QuantumNode:
    """A network endpoint with quantum memory.

    Parameters
    ----------
    node_id : str
        Unique node identifier.
    n_memories : int
        Number of qubit memory slots.
    coherence_time : float
        T2 coherence time in seconds (used to compute dephasing per step).
    """

    def __init__(
        self,
        node_id: str,
        n_memories: int = 4,
        coherence_time: float = 1.0,
    ) -> None:
        if n_memories < 1:
            raise NetworkError("n_memories must be >= 1")
        if coherence_time <= 0:
            raise NetworkError("coherence_time must be positive")
        self.node_id = node_id
        self.n_memories = n_memories
        self.coherence_time = coherence_time
        # Stored entanglement: memory_slot -> (partner_node_id, fidelity)
        self._memories: dict[int, tuple[str, float]] = {}

    def store_bell_pair(
        self, slot: int, partner: str, fidelity: float
    ) -> None:
        """Store a Bell pair in a memory slot."""
        if not 0 <= slot < self.n_memories:
            raise NetworkError(
                f"Slot {slot} out of range [0, {self.n_memories})"
            )
        self._memories[slot] = (partner, fidelity)

    def retrieve_bell_pair(self, slot: int) -> tuple[str, float]:
        """Retrieve (partner, fidelity) from a memory slot."""
        if slot not in self._memories:
            raise NetworkError(f"No Bell pair stored in slot {slot}")
        return self._memories.pop(slot)

    def apply_dephasing(self, t_wait: float) -> None:
        """Apply dephasing to all stored Bell pairs after waiting t_wait seconds."""
        p_deph = 1.0 - math.exp(-t_wait / self.coherence_time)
        new_mems: dict[int, tuple[str, float]] = {}
        for slot, (partner, F) in self._memories.items():
            F_new = fidelity_after_dephasing(F, p_deph)
            new_mems[slot] = (partner, F_new)
        self._memories = new_mems

    def available_slots(self) -> list[int]:
        """List of occupied memory slots."""
        return list(self._memories.keys())

    @property
    def free_slots(self) -> int:
        """Number of free memory slots."""
        return self.n_memories - len(self._memories)

    def __repr__(self) -> str:
        return (
            f"QuantumNode(id={self.node_id!r}, "
            f"memories={self.n_memories}, "
            f"T2={self.coherence_time:.3g}s)"
        )


# ── QuantumChannel ────────────────────────────────────────────────────────────

@dataclass
class QuantumChannel:
    """A noisy, lossy quantum channel between two nodes.

    Parameters
    ----------
    node_a, node_b : str
        IDs of the two connected nodes.
    transmission : float ∈ (0, 1]
        Photon transmission probability η.
    dephasing : float ∈ [0, 1]
        Per-qubit dephasing probability applied to newly generated Bell pairs.
    length_km : float
        Physical fibre length in km (informational; used in loss model).
    attenuation_db_km : float
        Fibre attenuation in dB/km (default: 0.2 dB/km for telecom fibre).
    """

    node_a: str
    node_b: str
    transmission: float = 0.9
    dephasing: float = 0.01
    length_km: float = 0.0
    attenuation_db_km: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 < self.transmission <= 1.0:
            raise NetworkError(
                f"transmission must be in (0, 1], got {self.transmission}"
            )
        if not 0.0 <= self.dephasing <= 1.0:
            raise NetworkError(
                f"dephasing must be in [0, 1], got {self.dephasing}"
            )
        if self.length_km < 0:
            raise NetworkError("length_km must be non-negative")

    @classmethod
    def from_length(
        cls,
        node_a: str,
        node_b: str,
        length_km: float,
        attenuation_db_km: float = 0.2,
        dephasing: float = 0.01,
    ) -> "QuantumChannel":
        """Create channel with transmission computed from fibre loss."""
        loss_db = attenuation_db_km * length_km
        # η = 10^(-loss_dB/10) × detector efficiency (assume 1 here)
        eta = 10.0 ** (-loss_db / 10.0)
        eta = float(np.clip(eta, 1e-9, 1.0))
        return cls(
            node_a=node_a,
            node_b=node_b,
            transmission=eta,
            dephasing=dephasing,
            length_km=length_km,
            attenuation_db_km=attenuation_db_km,
        )

    def bell_pair_fidelity(self) -> float:
        """Fidelity of a Bell pair generated across this channel.

        Combines photon loss (reduces singlet fraction) and dephasing::

            F_gen = η/2 + (1 − η)/4

        followed by ``fidelity_after_dephasing(F_gen, self.dephasing)``.
        """
        # Loss model: with prob η both photons arrive → Bell state (fidelity 1)
        # with prob 1-η → maximally mixed (fidelity 0.25)
        F_gen = self.transmission / 2.0 + (1.0 - self.transmission) / 4.0
        # Actually the correct formula for single-photon loss:
        # if photon detected (prob η): state is |Φ+⟩, F=1
        # if not detected: project onto mixed state, F=1/4
        # overall F = η*1 + (1-η)*1/4 = 3η/4 + 1/4
        F_gen = 3.0 * self.transmission / 4.0 + 0.25
        return fidelity_after_dephasing(F_gen, self.dephasing)

    def __repr__(self) -> str:
        return (
            f"QuantumChannel({self.node_a!r}↔{self.node_b!r}, "
            f"η={self.transmission:.3f}, deph={self.dephasing:.3f})"
        )


# ── BellPairGenerator ─────────────────────────────────────────────────────────

class BellPairGenerator:
    """Generate Bell pairs across a QuantumChannel.

    Parameters
    ----------
    channel : QuantumChannel
    attempts_per_cycle : int
        Number of generation attempts per time slot.
    seed : int or None
        Random seed for stochastic simulations.
    """

    def __init__(
        self,
        channel: QuantumChannel,
        attempts_per_cycle: int = 1,
        seed: int | None = None,
    ) -> None:
        if not isinstance(channel, QuantumChannel):
            raise NetworkError(
                f"Expected QuantumChannel, got {type(channel).__name__}"
            )
        if attempts_per_cycle < 1:
            raise NetworkError("attempts_per_cycle must be >= 1")
        self.channel = channel
        self.attempts_per_cycle = attempts_per_cycle
        self._rng = np.random.default_rng(seed)

    def generate(self, n_pairs: int = 1) -> list[float]:
        """Generate Bell pairs; return list of achieved fidelities.

        Each pair attempt succeeds with probability ``channel.transmission``;
        the success count is already modelled by the binomial draw, so the
        returned fidelity is the *conditional* fidelity of a successfully
        transmitted pair — dephasing only, no additional transmission penalty.

        Parameters
        ----------
        n_pairs : int
            Number of pairs to attempt to generate.

        Returns
        -------
        list[float]
            Fidelities of successfully generated pairs (length ≤ n_pairs).
        """
        # Conditional fidelity: quality of a pair that *did* arrive successfully.
        # Transmission loss affects rate (binomial), not per-pair quality.
        fidelity = fidelity_after_dephasing(1.0, self.channel.dephasing)
        total_attempts = n_pairs * self.attempts_per_cycle
        successes = self._rng.binomial(total_attempts, self.channel.transmission)
        successes = min(successes, n_pairs)
        return [fidelity] * successes

    def expected_fidelity(self) -> float:
        """Expected fidelity of a successfully generated pair (conditional on success)."""
        return fidelity_after_dephasing(1.0, self.channel.dephasing)

    def expected_rate(self) -> float:
        """Expected number of pairs generated per cycle."""
        return self.channel.transmission * self.attempts_per_cycle

    def __repr__(self) -> str:
        return (
            f"BellPairGenerator(channel={self.channel!r}, "
            f"attempts={self.attempts_per_cycle})"
        )


# ── EntanglementSwapper ───────────────────────────────────────────────────────

class EntanglementSwapper:
    """BSM-based entanglement swapping at a relay node.

    Parameters
    ----------
    bsm_fidelity : float ∈ (0, 1]
        Fidelity of the Bell-state measurement gate.
    seed : int or None
    """

    def __init__(
        self,
        bsm_fidelity: float = 0.99,
        seed: int | None = None,
    ) -> None:
        if not 0.0 < bsm_fidelity <= 1.0:
            raise NetworkError(
                f"bsm_fidelity must be in (0, 1], got {bsm_fidelity}"
            )
        self.bsm_fidelity = bsm_fidelity
        self._rng = np.random.default_rng(seed)

    def swap(self, F_left: float, F_right: float) -> float:
        """Perform entanglement swapping.

        Parameters
        ----------
        F_left : float
            Fidelity of the left Bell pair.
        F_right : float
            Fidelity of the right Bell pair.

        Returns
        -------
        float
            Fidelity of the end-to-end Bell pair after swapping.
        """
        return fidelity_after_swapping(F_left, F_right, self.bsm_fidelity)

    def swap_deterministic(self, fidelities: list[float]) -> float:
        """Swap a chain of Bell pairs, returning the end-to-end fidelity."""
        if not fidelities:
            raise NetworkError("fidelities list must not be empty")
        F = fidelities[0]
        for F_next in fidelities[1:]:
            F = self.swap(F, F_next)
        return F

    def __repr__(self) -> str:
        return f"EntanglementSwapper(bsm_fidelity={self.bsm_fidelity:.4f})"


# ── QuantumNetwork ────────────────────────────────────────────────────────────

class QuantumNetwork:
    """Graph of QuantumNodes connected by QuantumChannels.

    Parameters
    ----------
    nodes : list[QuantumNode]
    channels : list[QuantumChannel]
    """

    def __init__(
        self,
        nodes: list[QuantumNode] | None = None,
        channels: list[QuantumChannel] | None = None,
    ) -> None:
        self._nodes: dict[str, QuantumNode] = {}
        self._channels: list[QuantumChannel] = []
        for node in (nodes or []):
            self.add_node(node)
        for ch in (channels or []):
            self.add_channel(ch)

    def add_node(self, node: QuantumNode) -> None:
        """Add a node to the network."""
        if not isinstance(node, QuantumNode):
            raise NetworkError(
                f"Expected QuantumNode, got {type(node).__name__}"
            )
        if node.node_id in self._nodes:
            raise NetworkError(f"Node {node.node_id!r} already in network")
        self._nodes[node.node_id] = node

    def add_channel(self, channel: QuantumChannel) -> None:
        """Add a channel between two existing nodes."""
        if not isinstance(channel, QuantumChannel):
            raise NetworkError(
                f"Expected QuantumChannel, got {type(channel).__name__}"
            )
        for nid in (channel.node_a, channel.node_b):
            if nid not in self._nodes:
                raise NetworkError(
                    f"Node {nid!r} not in network. Add node before channel."
                )
        self._channels.append(channel)

    def get_node(self, node_id: str) -> QuantumNode:
        if node_id not in self._nodes:
            raise NetworkError(f"Node {node_id!r} not found")
        return self._nodes[node_id]

    def get_channel(self, node_a: str, node_b: str) -> QuantumChannel:
        """Return the channel between two nodes."""
        for ch in self._channels:
            if (ch.node_a == node_a and ch.node_b == node_b) or \
               (ch.node_a == node_b and ch.node_b == node_a):
                return ch
        raise NetworkError(
            f"No channel between {node_a!r} and {node_b!r}"
        )

    @property
    def n_nodes(self) -> int:
        return len(self._nodes)

    @property
    def n_channels(self) -> int:
        return len(self._channels)

    @property
    def node_ids(self) -> list[str]:
        return list(self._nodes)

    def adjacency_fidelity(self) -> dict[tuple[str, str], float]:
        """Return {(a, b): F} for all direct channels."""
        return {
            (ch.node_a, ch.node_b): ch.bell_pair_fidelity()
            for ch in self._channels
        }

    def total_channel_length(self) -> float:
        """Total network fibre length in km."""
        return sum(ch.length_km for ch in self._channels)

    def __repr__(self) -> str:
        return (
            f"QuantumNetwork(n_nodes={self.n_nodes}, "
            f"n_channels={self.n_channels})"
        )


# ── QuantumRepeater ───────────────────────────────────────────────────────────

class QuantumRepeater:
    """Single quantum repeater node performing entanglement swapping.

    A repeater receives Bell pairs from both the left and right segments,
    performs a BSM to swap entanglement, and optionally purifies the output.

    Parameters
    ----------
    node_id : str
        Identifier of the relay node.
    bsm_fidelity : float
        BSM fidelity.
    purification_rounds : int
        Number of DEJMPS purification rounds applied to each incoming pair
        before swapping (0 = no purification).
    """

    def __init__(
        self,
        node_id: str,
        bsm_fidelity: float = 0.99,
        purification_rounds: int = 0,
    ) -> None:
        if not 0.0 < bsm_fidelity <= 1.0:
            raise NetworkError(f"bsm_fidelity must be in (0,1], got {bsm_fidelity}")
        if purification_rounds < 0:
            raise NetworkError("purification_rounds must be non-negative")
        self.node_id = node_id
        self.bsm_fidelity = bsm_fidelity
        self.purification_rounds = purification_rounds

    def process(self, F_left: float, F_right: float) -> float:
        """Process two incoming Bell pairs and return the swapped fidelity.

        Applies optional purification to each incoming pair, then swaps.
        """
        F_l = fidelity_after_purification(F_left, self.purification_rounds)
        F_r = fidelity_after_purification(F_right, self.purification_rounds)
        return fidelity_after_swapping(F_l, F_r, self.bsm_fidelity)

    def __repr__(self) -> str:
        return (
            f"QuantumRepeater(id={self.node_id!r}, "
            f"bsm_fidelity={self.bsm_fidelity:.4f}, "
            f"purification_rounds={self.purification_rounds})"
        )


# ── NetworkResult ─────────────────────────────────────────────────────────────

@dataclass
class NetworkResult:
    """Output of a quantum network simulation.

    Attributes
    ----------
    end_to_end_fidelity : float
        Final Bell-pair fidelity between Alice and Bob.
    hop_fidelities : list[float]
        Fidelity after each hop / repeater stage.
    total_length_km : float
        Total path length in km.
    n_hops : int
        Number of repeater hops.
    secret_key_rate : float
        Estimated asymptotic secret key rate per Bell pair (BB84 bound):
        R = 1 - h₂(e) where h₂ is binary entropy and e = (1-F)/2.
    metadata : dict
        Extra simulation parameters.
    """

    end_to_end_fidelity: float
    hop_fidelities: list[float]
    total_length_km: float
    n_hops: int
    secret_key_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_useful(self, threshold: float = 0.5) -> bool:
        """Return True if fidelity exceeds the classical threshold."""
        return self.end_to_end_fidelity > threshold

    def __repr__(self) -> str:
        return (
            f"NetworkResult(F={self.end_to_end_fidelity:.4f}, "
            f"n_hops={self.n_hops}, "
            f"length={self.total_length_km:.1f}km, "
            f"SKR={self.secret_key_rate:.4f})"
        )


# ── Utilities ─────────────────────────────────────────────────────────────────

def _binary_entropy(p: float) -> float:
    """Binary entropy h₂(p) = -p log₂p - (1-p) log₂(1-p)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def _secret_key_rate(F: float) -> float:
    """Asymptotic BB84 secret key rate: R = 1 - h₂(e) where e = (1-F)/2."""
    e = (1.0 - F) / 2.0
    R = 1.0 - _binary_entropy(e)
    return float(max(0.0, R))


# ── simulate_repeater_chain ───────────────────────────────────────────────────

def simulate_repeater_chain(
    n_hops: int,
    segment_length_km: float,
    attenuation_db_km: float = 0.2,
    dephasing_per_segment: float = 0.01,
    bsm_fidelity: float = 0.99,
    purification_rounds: int = 0,
    memory_coherence_time: float | None = None,
    t_segment_s: float | None = None,
) -> "NetworkResult":
    """Simulate an N-hop quantum repeater chain.

    Builds an inline chain: Alice — [repeater] — … — [repeater] — Bob.

    Parameters
    ----------
    n_hops : int
        Number of intermediate repeater nodes (segments = n_hops + 1).
        For a direct link (no repeaters), use ``n_hops=0``.
    segment_length_km : float
        Length of each fibre segment.
    attenuation_db_km : float
        Fibre loss in dB/km.
    dephasing_per_segment : float
        Depolarising probability per Bell pair per segment.
    bsm_fidelity : float
        BSM fidelity at each repeater.
    purification_rounds : int
        Rounds of DEJMPS purification applied to each segment's Bell pair.
    memory_coherence_time : float, optional
        T2 of quantum memories in seconds; if provided, dephasing from
        waiting is also computed using ``t_segment_s``.
    t_segment_s : float, optional
        Time to generate one Bell pair per segment in seconds.

    Returns
    -------
    NetworkResult
    """
    if n_hops < 0:
        raise NetworkError("n_hops must be non-negative")
    if segment_length_km <= 0:
        raise NetworkError("segment_length_km must be positive")
    if memory_coherence_time is not None and memory_coherence_time <= 0:
        raise NetworkError("memory_coherence_time must be positive")

    n_segments = n_hops + 1
    total_km = n_segments * segment_length_km

    # Channel per segment
    ch = QuantumChannel.from_length(
        "A", "B",
        length_km=segment_length_km,
        attenuation_db_km=attenuation_db_km,
        dephasing=dephasing_per_segment,
    )
    F_seg = ch.bell_pair_fidelity()

    # Optional: apply memory dephasing from waiting time
    if memory_coherence_time is not None and t_segment_s is not None:
        t_wait = t_segment_s * n_hops  # inner segments wait longer
        p_mem_deph = 1.0 - math.exp(-t_wait / memory_coherence_time)
        F_seg = fidelity_after_dephasing(F_seg, p_mem_deph)

    # Purification on each segment pair
    F_seg = fidelity_after_purification(F_seg, purification_rounds)

    # Build hop-by-hop fidelity list
    hop_fidelities: list[float] = [F_seg] * n_segments

    if n_hops == 0:
        F_final = F_seg
    else:
        # Swap through all repeaters
        F_current = F_seg
        for _ in range(n_hops):
            F_current = fidelity_after_swapping(F_current, F_seg, bsm_fidelity)
        F_final = F_current

    return NetworkResult(
        end_to_end_fidelity=F_final,
        hop_fidelities=hop_fidelities,
        total_length_km=total_km,
        n_hops=n_hops,
        secret_key_rate=_secret_key_rate(F_final),
        metadata={
            "segment_length_km": segment_length_km,
            "attenuation_db_km": attenuation_db_km,
            "bsm_fidelity": bsm_fidelity,
            "purification_rounds": purification_rounds,
            "F_per_segment": F_seg,
        },
    )
