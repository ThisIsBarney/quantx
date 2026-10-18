"""Tests for quantum_network.py — M96 quantum network simulation."""
from __future__ import annotations

import math
import unittest

import numpy as np

from quantx_engine.contrib.quantum_network import (
    NetworkError,
    QuantumNode,
    QuantumChannel,
    QuantumNetwork,
    BellPairGenerator,
    EntanglementSwapper,
    QuantumRepeater,
    NetworkResult,
    fidelity_after_dephasing,
    fidelity_after_swapping,
    fidelity_after_purification,
    simulate_repeater_chain,
    _werner_state,
    _fidelity_to_werner,
    _binary_entropy,
    _secret_key_rate,
)


# ─────────────────────────────────────────────────────────────────────────────
# Werner state helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestWernerState(unittest.TestCase):
    def test_trace_one(self):
        rho = _werner_state(0.8)
        self.assertAlmostEqual(np.trace(rho).real, 1.0, places=10)

    def test_hermitian(self):
        rho = _werner_state(0.7)
        self.assertTrue(np.allclose(rho, rho.conj().T, atol=1e-12))

    def test_fidelity_roundtrip(self):
        for F in [0.25, 0.5, 0.75, 1.0]:
            rho = _werner_state(F)
            F_back = _fidelity_to_werner(rho)
            self.assertAlmostEqual(F_back, F, places=10)

    def test_f_one_is_bell_state(self):
        rho = _werner_state(1.0)
        phi = np.array([1, 0, 0, 1], dtype=complex) / math.sqrt(2)
        proj = np.outer(phi.conj(), phi)
        self.assertTrue(np.allclose(rho, proj, atol=1e-12))

    def test_f_quarter_maximally_mixed(self):
        rho = _werner_state(0.25)
        self.assertTrue(np.allclose(rho, np.eye(4) / 4, atol=1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# fidelity_after_dephasing
# ─────────────────────────────────────────────────────────────────────────────

class TestFidelityAfterDephasing(unittest.TestCase):
    def test_no_dephasing_identity(self):
        F = 0.9
        self.assertAlmostEqual(fidelity_after_dephasing(F, 0.0), F, places=12)

    def test_full_dephasing_to_mixed(self):
        """p=1 → completely depolarised state → F = 0.25."""
        F_out = fidelity_after_dephasing(0.9, 1.0)
        self.assertAlmostEqual(F_out, 0.25, places=6)

    def test_decreases_with_dephasing(self):
        F = 0.9
        F_low = fidelity_after_dephasing(F, 0.1)
        F_high = fidelity_after_dephasing(F, 0.5)
        self.assertLess(F_high, F_low)
        self.assertLess(F_low, F)

    def test_result_in_0_1(self):
        for F in [0.3, 0.6, 0.9]:
            for p in [0.0, 0.1, 0.5, 1.0]:
                F_out = fidelity_after_dephasing(F, p)
                self.assertGreaterEqual(F_out, 0.0)
                self.assertLessEqual(F_out, 1.0)

    def test_invalid_F_raises(self):
        with self.assertRaises(NetworkError):
            fidelity_after_dephasing(1.5, 0.1)

    def test_invalid_p_raises(self):
        with self.assertRaises(NetworkError):
            fidelity_after_dephasing(0.9, -0.1)

    def test_f_25_stays_at_25(self):
        """Maximally mixed state is invariant under dephasing."""
        F_out = fidelity_after_dephasing(0.25, 0.5)
        self.assertAlmostEqual(F_out, 0.25, places=10)


# ─────────────────────────────────────────────────────────────────────────────
# fidelity_after_swapping
# ─────────────────────────────────────────────────────────────────────────────

class TestFidelityAfterSwapping(unittest.TestCase):
    def test_perfect_pairs_perfect_bsm(self):
        """Two perfect pairs + perfect BSM → perfect output."""
        F = fidelity_after_swapping(1.0, 1.0, 1.0)
        self.assertAlmostEqual(F, 1.0, places=8)

    def test_decreases_with_imperfect_bsm(self):
        F_ideal = fidelity_after_swapping(0.9, 0.9, 1.0)
        F_lossy = fidelity_after_swapping(0.9, 0.9, 0.9)
        self.assertLess(F_lossy, F_ideal)

    def test_symmetric_in_fa_fb(self):
        F1 = fidelity_after_swapping(0.8, 0.9, 0.99)
        F2 = fidelity_after_swapping(0.9, 0.8, 0.99)
        self.assertAlmostEqual(F1, F2, places=12)

    def test_result_in_0_1(self):
        for fa in [0.5, 0.7, 0.9]:
            for fb in [0.5, 0.8]:
                F = fidelity_after_swapping(fa, fb, 0.95)
                self.assertGreaterEqual(F, 0.0)
                self.assertLessEqual(F, 1.0)

    def test_invalid_fa_raises(self):
        with self.assertRaises(NetworkError):
            fidelity_after_swapping(1.2, 0.9, 1.0)

    def test_invalid_fgate_raises(self):
        with self.assertRaises(NetworkError):
            fidelity_after_swapping(0.9, 0.9, 1.1)

    def test_mixed_pairs_give_low_fidelity(self):
        """Swapping two nearly-mixed pairs → near-classical fidelity."""
        F = fidelity_after_swapping(0.26, 0.26, 0.99)
        self.assertLess(F, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
# fidelity_after_purification
# ─────────────────────────────────────────────────────────────────────────────

class TestFidelityAfterPurification(unittest.TestCase):
    def test_zero_rounds_identity(self):
        F = fidelity_after_purification(0.8, 0)
        self.assertAlmostEqual(F, 0.8, places=12)

    def test_one_round_increases_fidelity(self):
        F_in = 0.75
        F_out = fidelity_after_purification(F_in, 1)
        self.assertGreater(F_out, F_in)

    def test_two_rounds_better_than_one(self):
        F_in = 0.75
        F1 = fidelity_after_purification(F_in, 1)
        F2 = fidelity_after_purification(F_in, 2)
        self.assertGreater(F2, F1)

    def test_perfect_pair_unchanged(self):
        F = fidelity_after_purification(1.0, 5)
        self.assertAlmostEqual(F, 1.0, places=8)

    def test_result_in_0_1(self):
        for F_in in [0.5, 0.6, 0.8, 0.95]:
            F_out = fidelity_after_purification(F_in, 3)
            self.assertGreaterEqual(F_out, 0.0)
            self.assertLessEqual(F_out, 1.0)

    def test_invalid_F_raises(self):
        with self.assertRaises(NetworkError):
            fidelity_after_purification(-0.1, 1)

    def test_invalid_rounds_raises(self):
        with self.assertRaises(NetworkError):
            fidelity_after_purification(0.8, -1)

    def test_low_fidelity_not_improved_much(self):
        """F < 0.25 → below DEJMPS threshold, purification degrades fidelity."""
        # DEJMPS fixed point: F' < F iff F < 1/4
        F_in = 0.1
        F_out = fidelity_after_purification(F_in, 1)
        self.assertLess(F_out, F_in)


# ─────────────────────────────────────────────────────────────────────────────
# QuantumNode
# ─────────────────────────────────────────────────────────────────────────────

class TestQuantumNode(unittest.TestCase):
    def _node(self, n=4, T2=1.0):
        return QuantumNode(node_id="A", n_memories=n, coherence_time=T2)

    def test_construction(self):
        node = self._node()
        self.assertEqual(node.node_id, "A")
        self.assertEqual(node.n_memories, 4)

    def test_repr(self):
        node = self._node()
        r = repr(node)
        self.assertIn("A", r)

    def test_store_and_retrieve(self):
        node = self._node()
        node.store_bell_pair(0, "B", 0.9)
        partner, F = node.retrieve_bell_pair(0)
        self.assertEqual(partner, "B")
        self.assertAlmostEqual(F, 0.9, places=10)

    def test_retrieve_twice_raises(self):
        node = self._node()
        node.store_bell_pair(1, "B", 0.9)
        node.retrieve_bell_pair(1)
        with self.assertRaises(NetworkError):
            node.retrieve_bell_pair(1)

    def test_invalid_slot_raises(self):
        node = self._node(n=2)
        with self.assertRaises(NetworkError):
            node.store_bell_pair(5, "B", 0.9)

    def test_available_slots(self):
        node = self._node()
        node.store_bell_pair(0, "B", 0.9)
        node.store_bell_pair(2, "C", 0.8)
        self.assertIn(0, node.available_slots())
        self.assertIn(2, node.available_slots())
        self.assertNotIn(1, node.available_slots())

    def test_free_slots(self):
        node = self._node(n=3)
        self.assertEqual(node.free_slots, 3)
        node.store_bell_pair(0, "B", 0.9)
        self.assertEqual(node.free_slots, 2)

    def test_dephasing_reduces_fidelity(self):
        node = QuantumNode("A", n_memories=1, coherence_time=1.0)
        node.store_bell_pair(0, "B", 0.95)
        node.apply_dephasing(0.5)  # wait half a coherence time
        _, F = node.retrieve_bell_pair(0)
        self.assertLess(F, 0.95)
        self.assertGreater(F, 0.0)

    def test_zero_memories_raises(self):
        with self.assertRaises(NetworkError):
            QuantumNode("A", n_memories=0)

    def test_zero_coherence_time_raises(self):
        with self.assertRaises(NetworkError):
            QuantumNode("A", coherence_time=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# QuantumChannel
# ─────────────────────────────────────────────────────────────────────────────

class TestQuantumChannel(unittest.TestCase):
    def test_basic_construction(self):
        ch = QuantumChannel("A", "B", transmission=0.9)
        self.assertEqual(ch.node_a, "A")
        self.assertEqual(ch.node_b, "B")
        self.assertAlmostEqual(ch.transmission, 0.9)

    def test_repr(self):
        ch = QuantumChannel("A", "B", transmission=0.9)
        r = repr(ch)
        self.assertIn("A", r)
        self.assertIn("B", r)

    def test_zero_transmission_raises(self):
        with self.assertRaises(NetworkError):
            QuantumChannel("A", "B", transmission=0.0)

    def test_negative_dephasing_raises(self):
        with self.assertRaises(NetworkError):
            QuantumChannel("A", "B", dephasing=-0.1)

    def test_from_length_decreases_with_distance(self):
        ch_near = QuantumChannel.from_length("A", "B", length_km=1.0)
        ch_far = QuantumChannel.from_length("A", "B", length_km=100.0)
        self.assertGreater(ch_near.transmission, ch_far.transmission)

    def test_from_length_zero_km_near_perfect(self):
        ch = QuantumChannel.from_length("A", "B", length_km=0.0)
        self.assertAlmostEqual(ch.transmission, 1.0, places=6)

    def test_bell_pair_fidelity_in_0_1(self):
        for eta in [0.1, 0.5, 0.9, 1.0]:
            ch = QuantumChannel("A", "B", transmission=eta, dephasing=0.01)
            F = ch.bell_pair_fidelity()
            self.assertGreater(F, 0.0)
            self.assertLessEqual(F, 1.0)

    def test_bell_pair_fidelity_perfect_channel(self):
        ch = QuantumChannel("A", "B", transmission=1.0, dephasing=0.0)
        F = ch.bell_pair_fidelity()
        self.assertAlmostEqual(F, 1.0, places=6)

    def test_bell_pair_fidelity_decreases_with_loss(self):
        ch_good = QuantumChannel("A", "B", transmission=0.99)
        ch_bad = QuantumChannel("A", "B", transmission=0.5)
        self.assertGreater(ch_good.bell_pair_fidelity(), ch_bad.bell_pair_fidelity())


# ─────────────────────────────────────────────────────────────────────────────
# BellPairGenerator
# ─────────────────────────────────────────────────────────────────────────────

class TestBellPairGenerator(unittest.TestCase):
    def _gen(self, eta=0.9, attempts=1):
        ch = QuantumChannel("A", "B", transmission=eta)
        return BellPairGenerator(ch, attempts_per_cycle=attempts, seed=0)

    def test_construction(self):
        gen = self._gen()
        self.assertIsNotNone(gen)

    def test_repr(self):
        gen = self._gen()
        r = repr(gen)
        self.assertIn("QuantumChannel", r)

    def test_generate_returns_list(self):
        gen = self._gen()
        result = gen.generate(1)
        self.assertIsInstance(result, list)

    def test_generate_at_most_n(self):
        gen = self._gen(eta=1.0)
        result = gen.generate(5)
        self.assertLessEqual(len(result), 5)

    def test_generate_fidelities_in_0_1(self):
        gen = self._gen(eta=0.9)
        for F in gen.generate(10):
            self.assertGreater(F, 0.0)
            self.assertLessEqual(F, 1.0)

    def test_expected_fidelity(self):
        gen = self._gen(eta=0.9)
        F_exp = gen.expected_fidelity()
        self.assertGreater(F_exp, 0.5)

    def test_expected_rate(self):
        gen = self._gen(eta=0.8, attempts=2)
        self.assertAlmostEqual(gen.expected_rate(), 1.6, places=10)

    def test_non_channel_raises(self):
        with self.assertRaises(NetworkError):
            BellPairGenerator("not a channel")

    def test_zero_attempts_raises(self):
        ch = QuantumChannel("A", "B")
        with self.assertRaises(NetworkError):
            BellPairGenerator(ch, attempts_per_cycle=0)

    def test_high_eta_many_successes(self):
        gen = self._gen(eta=1.0)
        result = gen.generate(20)
        self.assertGreater(len(result), 0)


# ─────────────────────────────────────────────────────────────────────────────
# EntanglementSwapper
# ─────────────────────────────────────────────────────────────────────────────

class TestEntanglementSwapper(unittest.TestCase):
    def _swapper(self, F_bsm=0.99):
        return EntanglementSwapper(bsm_fidelity=F_bsm, seed=0)

    def test_construction(self):
        s = self._swapper()
        self.assertAlmostEqual(s.bsm_fidelity, 0.99)

    def test_repr(self):
        s = self._swapper()
        r = repr(s)
        self.assertIn("0.99", r)

    def test_swap_perfect(self):
        s = self._swapper(1.0)
        F = s.swap(1.0, 1.0)
        self.assertAlmostEqual(F, 1.0, places=6)

    def test_swap_result_in_0_1(self):
        s = self._swapper(0.99)
        F = s.swap(0.9, 0.85)
        self.assertGreaterEqual(F, 0.0)
        self.assertLessEqual(F, 1.0)

    def test_swap_chain_decreases_fidelity(self):
        s = self._swapper(0.99)
        F1 = s.swap(0.9, 0.9)      # 1 swap
        F2 = s.swap(F1, 0.9)       # 2 swaps
        self.assertLess(F2, F1)

    def test_swap_deterministic_chain(self):
        s = self._swapper(0.99)
        F = s.swap_deterministic([0.9, 0.9, 0.9])
        self.assertGreater(F, 0.0)
        self.assertLess(F, 0.9)

    def test_swap_deterministic_single_element(self):
        s = self._swapper()
        F = s.swap_deterministic([0.8])
        self.assertAlmostEqual(F, 0.8, places=12)

    def test_swap_deterministic_empty_raises(self):
        s = self._swapper()
        with self.assertRaises(NetworkError):
            s.swap_deterministic([])

    def test_zero_bsm_fidelity_raises(self):
        with self.assertRaises(NetworkError):
            EntanglementSwapper(bsm_fidelity=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# QuantumNetwork
# ─────────────────────────────────────────────────────────────────────────────

class TestQuantumNetwork(unittest.TestCase):
    def _build(self):
        n = QuantumNetwork()
        n.add_node(QuantumNode("Alice"))
        n.add_node(QuantumNode("Bob"))
        n.add_node(QuantumNode("Charlie"))
        n.add_channel(QuantumChannel("Alice", "Bob", transmission=0.9))
        n.add_channel(QuantumChannel("Bob", "Charlie", transmission=0.8))
        return n

    def test_construction(self):
        net = self._build()
        self.assertEqual(net.n_nodes, 3)
        self.assertEqual(net.n_channels, 2)

    def test_repr(self):
        net = self._build()
        r = repr(net)
        self.assertIn("n_nodes=3", r)
        self.assertIn("n_channels=2", r)

    def test_get_node(self):
        net = self._build()
        node = net.get_node("Alice")
        self.assertEqual(node.node_id, "Alice")

    def test_get_node_missing_raises(self):
        net = self._build()
        with self.assertRaises(NetworkError):
            net.get_node("Dave")

    def test_get_channel(self):
        net = self._build()
        ch = net.get_channel("Alice", "Bob")
        self.assertIn("Alice", (ch.node_a, ch.node_b))

    def test_get_channel_reversed(self):
        net = self._build()
        ch = net.get_channel("Bob", "Alice")
        self.assertIsNotNone(ch)

    def test_get_channel_missing_raises(self):
        net = self._build()
        with self.assertRaises(NetworkError):
            net.get_channel("Alice", "Charlie")

    def test_duplicate_node_raises(self):
        net = QuantumNetwork()
        net.add_node(QuantumNode("A"))
        with self.assertRaises(NetworkError):
            net.add_node(QuantumNode("A"))

    def test_channel_unknown_node_raises(self):
        net = QuantumNetwork()
        net.add_node(QuantumNode("A"))
        with self.assertRaises(NetworkError):
            net.add_channel(QuantumChannel("A", "B"))  # B not in network

    def test_node_ids(self):
        net = self._build()
        ids = net.node_ids
        self.assertIn("Alice", ids)
        self.assertIn("Bob", ids)

    def test_adjacency_fidelity(self):
        net = self._build()
        adj = net.adjacency_fidelity()
        self.assertIsInstance(adj, dict)
        self.assertEqual(len(adj), 2)

    def test_total_channel_length(self):
        net = QuantumNetwork()
        net.add_node(QuantumNode("A"))
        net.add_node(QuantumNode("B"))
        net.add_channel(
            QuantumChannel.from_length("A", "B", length_km=10.0)
        )
        self.assertAlmostEqual(net.total_channel_length(), 10.0)

    def test_non_node_add_raises(self):
        net = QuantumNetwork()
        with self.assertRaises(NetworkError):
            net.add_node("not a node")

    def test_constructor_with_lists(self):
        nodes = [QuantumNode("A"), QuantumNode("B")]
        channels = [QuantumChannel("A", "B")]
        net = QuantumNetwork(nodes=nodes, channels=channels)
        self.assertEqual(net.n_nodes, 2)
        self.assertEqual(net.n_channels, 1)


# ─────────────────────────────────────────────────────────────────────────────
# QuantumRepeater
# ─────────────────────────────────────────────────────────────────────────────

class TestQuantumRepeater(unittest.TestCase):
    def test_construction(self):
        r = QuantumRepeater("relay", bsm_fidelity=0.99)
        self.assertEqual(r.node_id, "relay")
        self.assertAlmostEqual(r.bsm_fidelity, 0.99)

    def test_repr(self):
        r = QuantumRepeater("relay")
        rep = repr(r)
        self.assertIn("relay", rep)

    def test_process_no_purification(self):
        r = QuantumRepeater("relay", bsm_fidelity=1.0, purification_rounds=0)
        F = r.process(0.9, 0.9)
        # Should equal fidelity_after_swapping(0.9, 0.9, 1.0)
        expected = fidelity_after_swapping(0.9, 0.9, 1.0)
        self.assertAlmostEqual(F, expected, places=10)

    def test_process_with_purification_better(self):
        r_pure = QuantumRepeater("r", bsm_fidelity=0.99, purification_rounds=1)
        r_plain = QuantumRepeater("r", bsm_fidelity=0.99, purification_rounds=0)
        F_pure = r_pure.process(0.75, 0.75)
        F_plain = r_plain.process(0.75, 0.75)
        self.assertGreater(F_pure, F_plain)

    def test_invalid_bsm_fidelity_raises(self):
        with self.assertRaises(NetworkError):
            QuantumRepeater("r", bsm_fidelity=0.0)

    def test_invalid_purification_rounds_raises(self):
        with self.assertRaises(NetworkError):
            QuantumRepeater("r", purification_rounds=-1)


# ─────────────────────────────────────────────────────────────────────────────
# NetworkResult
# ─────────────────────────────────────────────────────────────────────────────

class TestNetworkResult(unittest.TestCase):
    def _result(self, F=0.85):
        return NetworkResult(
            end_to_end_fidelity=F,
            hop_fidelities=[0.9, 0.9],
            total_length_km=100.0,
            n_hops=1,
            secret_key_rate=_secret_key_rate(F),
        )

    def test_repr(self):
        r = repr(self._result())
        self.assertIn("F=0.8500", r)

    def test_is_useful_above_threshold(self):
        self.assertTrue(self._result(F=0.7).is_useful(threshold=0.5))

    def test_is_useful_below_threshold(self):
        self.assertFalse(self._result(F=0.4).is_useful(threshold=0.5))

    def test_secret_key_rate_in_0_1(self):
        for F in [0.5, 0.75, 0.9, 1.0]:
            R = _secret_key_rate(F)
            self.assertGreaterEqual(R, 0.0)
            self.assertLessEqual(R, 1.0)

    def test_secret_key_rate_zero_at_classical(self):
        """At F=0 (QBER=0.5), BB84 SKR is 0; F=0.5 has positive rate ≈ 0.19."""
        # BB84 rate R = 1 - h2(e) = 0 when e = (1-F)/2 = 0.5, i.e. F = 0
        R_zero = _secret_key_rate(0.0)
        self.assertAlmostEqual(R_zero, 0.0, places=5)
        # At F=0.5, QBER = 0.25, rate is positive (~0.19)
        R_half = _secret_key_rate(0.5)
        self.assertGreater(R_half, 0.0)

    def test_secret_key_rate_one_at_perfect(self):
        R = _secret_key_rate(1.0)
        self.assertAlmostEqual(R, 1.0, places=5)


# ─────────────────────────────────────────────────────────────────────────────
# simulate_repeater_chain
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulateRepeaterChain(unittest.TestCase):
    def test_returns_network_result(self):
        r = simulate_repeater_chain(n_hops=1, segment_length_km=10.0)
        self.assertIsInstance(r, NetworkResult)

    def test_zero_hops_direct_link(self):
        r = simulate_repeater_chain(n_hops=0, segment_length_km=10.0)
        self.assertEqual(r.n_hops, 0)

    def test_fidelity_in_0_1(self):
        for n in [0, 1, 3, 5]:
            r = simulate_repeater_chain(n_hops=n, segment_length_km=10.0)
            self.assertGreater(r.end_to_end_fidelity, 0.0)
            self.assertLessEqual(r.end_to_end_fidelity, 1.0)

    def test_total_length_correct(self):
        r = simulate_repeater_chain(n_hops=3, segment_length_km=10.0)
        self.assertAlmostEqual(r.total_length_km, 40.0, places=6)

    def test_n_hops_stored(self):
        r = simulate_repeater_chain(n_hops=5, segment_length_km=5.0)
        self.assertEqual(r.n_hops, 5)

    def test_hop_fidelities_length(self):
        r = simulate_repeater_chain(n_hops=3, segment_length_km=10.0)
        # n_hops + 1 segments
        self.assertEqual(len(r.hop_fidelities), 4)

    def test_more_hops_lower_fidelity(self):
        r1 = simulate_repeater_chain(n_hops=1, segment_length_km=10.0)
        r5 = simulate_repeater_chain(n_hops=5, segment_length_km=10.0)
        self.assertGreater(r1.end_to_end_fidelity, r5.end_to_end_fidelity)

    def test_purification_improves_fidelity(self):
        r_no = simulate_repeater_chain(n_hops=2, segment_length_km=20.0,
                                       purification_rounds=0)
        r_pu = simulate_repeater_chain(n_hops=2, segment_length_km=20.0,
                                       purification_rounds=1)
        self.assertGreaterEqual(r_pu.end_to_end_fidelity,
                                r_no.end_to_end_fidelity - 1e-9)

    def test_secret_key_rate_in_0_1(self):
        r = simulate_repeater_chain(n_hops=1, segment_length_km=10.0)
        self.assertGreaterEqual(r.secret_key_rate, 0.0)
        self.assertLessEqual(r.secret_key_rate, 1.0)

    def test_metadata_contains_segment_length(self):
        r = simulate_repeater_chain(n_hops=1, segment_length_km=10.0)
        self.assertIn("segment_length_km", r.metadata)
        self.assertAlmostEqual(r.metadata["segment_length_km"], 10.0)

    def test_negative_hops_raises(self):
        with self.assertRaises(NetworkError):
            simulate_repeater_chain(n_hops=-1, segment_length_km=10.0)

    def test_zero_segment_length_raises(self):
        with self.assertRaises(NetworkError):
            simulate_repeater_chain(n_hops=1, segment_length_km=0.0)

    def test_with_memory_dephasing(self):
        r = simulate_repeater_chain(
            n_hops=2, segment_length_km=10.0,
            memory_coherence_time=1.0, t_segment_s=0.1
        )
        self.assertIsInstance(r, NetworkResult)
        self.assertGreater(r.end_to_end_fidelity, 0.0)

    def test_zero_coherence_time_raises(self):
        with self.assertRaises(NetworkError):
            simulate_repeater_chain(
                n_hops=1, segment_length_km=10.0,
                memory_coherence_time=0.0, t_segment_s=0.1
            )

    def test_perfect_channel_high_fidelity(self):
        """Zero-length, no dephasing, perfect BSM → F ≈ 1."""
        r = simulate_repeater_chain(
            n_hops=1, segment_length_km=0.001,
            attenuation_db_km=0.0, dephasing_per_segment=0.0, bsm_fidelity=1.0
        )
        self.assertGreater(r.end_to_end_fidelity, 0.9)

    def test_is_useful(self):
        r = simulate_repeater_chain(n_hops=0, segment_length_km=1.0,
                                    attenuation_db_km=0.0)
        self.assertTrue(r.is_useful())


if __name__ == "__main__":
    unittest.main()
