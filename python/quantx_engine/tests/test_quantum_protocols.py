"""Tests for quantum communication protocols — M97."""
from __future__ import annotations

import math
import pytest

from quantx_engine.quantum_protocols import (
    ProtocolError,
    BB84Result,
    BB84Protocol,
    E91Result,
    E91Protocol,
    TeleportationResult,
    TeleportationProtocol,
    DenseCodingResult,
    DenseCodingProtocol,
    bb84_key_rate,
    chsh_inequality_value,
    teleportation_fidelity,
)


# ── bb84_key_rate ──────────────────────────────────────────────────────────────

class TestBB84KeyRate:
    def test_zero_qber_gives_one(self):
        assert bb84_key_rate(0.0) == pytest.approx(1.0)

    def test_above_threshold_gives_zero(self):
        # At QBER = 0.11 (Shor-Preskill threshold), rate ≈ 0
        assert bb84_key_rate(0.11) >= 0.0
        assert bb84_key_rate(0.15) == 0.0

    def test_typical_qber(self):
        # QBER = 0.05 → rate = 1 - 2*h2(0.05) ≈ 0.427 (Shor-Preskill formula)
        r = bb84_key_rate(0.05)
        assert 0.3 < r < 0.6

    def test_monotone_decreasing(self):
        rates = [bb84_key_rate(q) for q in [0.0, 0.02, 0.05, 0.10, 0.15]]
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1]

    def test_invalid_qber_raises(self):
        with pytest.raises(ProtocolError):
            bb84_key_rate(-0.01)
        with pytest.raises(ProtocolError):
            bb84_key_rate(0.6)


# ── chsh_inequality_value ─────────────────────────────────────────────────────

class TestCHSHInequality:
    def test_classical_bound(self):
        # Classical correlations: S ≤ 2
        correlations = {(0, 0): 1.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 1.0}
        S = chsh_inequality_value(correlations)
        assert S <= 2.0 + 1e-9

    def test_quantum_violation(self):
        # Ideal quantum: E(a,b) = cos(θ_a - θ_b), optimal angles give S = 2√2
        # Approximate with known quantum values
        c = 1.0 / math.sqrt(2)
        correlations = {
            (0, 0): c,
            (0, 1): -c,
            (1, 0): c,
            (1, 1): c,
        }
        S = chsh_inequality_value(correlations)
        assert S == pytest.approx(4 * c, abs=1e-9)  # = 2√2

    def test_tsirelson_bound(self):
        # Optimal quantum correlations saturate S = 2√2 (Tsirelson bound)
        c = 1.0 / math.sqrt(2)
        correlations = {(0, 0): c, (0, 1): -c, (1, 0): c, (1, 1): c}
        S = chsh_inequality_value(correlations)
        assert S == pytest.approx(2 * math.sqrt(2), abs=1e-9)

    def test_missing_key_raises(self):
        with pytest.raises(ProtocolError):
            chsh_inequality_value({(0, 0): 1.0, (0, 1): 0.0})


# ── teleportation_fidelity ────────────────────────────────────────────────────

class TestTeleportationFidelity:
    def test_perfect_channel(self):
        assert teleportation_fidelity(1.0) == pytest.approx(1.0)

    def test_maximally_mixed(self):
        # F=0.25 (maximally mixed channel) → f = (0.5+1)/3 = 0.5
        assert teleportation_fidelity(0.25) == pytest.approx(0.5, abs=1e-9)

    def test_classical_threshold(self):
        # Classical guess fidelity = 2/3 achieved at F = 1/2
        assert teleportation_fidelity(0.5) == pytest.approx(2.0 / 3.0, abs=1e-9)

    def test_exceeds_classical_above_half(self):
        assert teleportation_fidelity(0.6) > 2.0 / 3.0

    def test_monotone_increasing(self):
        fids = [teleportation_fidelity(F) for F in [0.25, 0.5, 0.75, 1.0]]
        for i in range(len(fids) - 1):
            assert fids[i] < fids[i + 1]

    def test_invalid_fidelity_raises(self):
        with pytest.raises(ProtocolError):
            teleportation_fidelity(-0.1)
        with pytest.raises(ProtocolError):
            teleportation_fidelity(1.1)


# ── BB84Protocol ──────────────────────────────────────────────────────────────

class TestBB84Protocol:
    def test_basic_run_returns_result(self):
        proto = BB84Protocol(seed=42)
        result = proto.run(n_bits=1000)
        assert isinstance(result, BB84Result)

    def test_sifted_bits_roughly_half(self):
        proto = BB84Protocol(seed=42)
        result = proto.run(n_bits=2000)
        # Sifted ≈ n_bits / 2 (bases match 50% of the time)
        assert 700 < result.n_sifted_bits < 1300

    def test_no_eve_low_qber(self):
        proto = BB84Protocol(channel_error_rate=0.02, eve_intercept_fraction=0.0, seed=0)
        result = proto.run(n_bits=2000)
        assert result.qber < 0.08

    def test_eve_raises_qber(self):
        proto_no_eve = BB84Protocol(channel_error_rate=0.02, eve_intercept_fraction=0.0, seed=1)
        proto_with_eve = BB84Protocol(channel_error_rate=0.02, eve_intercept_fraction=1.0, seed=1)
        r_clean = proto_no_eve.run(2000)
        r_eve = proto_with_eve.run(2000)
        assert r_eve.qber > r_clean.qber

    def test_full_eve_detected(self):
        proto = BB84Protocol(channel_error_rate=0.0, eve_intercept_fraction=1.0, seed=7)
        result = proto.run(n_bits=3000)
        assert result.eve_detected

    def test_no_eve_not_detected(self):
        proto = BB84Protocol(channel_error_rate=0.01, eve_intercept_fraction=0.0, seed=7)
        result = proto.run(n_bits=3000)
        assert not result.eve_detected

    def test_secure_key_positive_without_eve(self):
        proto = BB84Protocol(channel_error_rate=0.02, eve_intercept_fraction=0.0, seed=5)
        result = proto.run(n_bits=2000)
        assert result.n_secure_bits > 0

    def test_key_lengths_consistent(self):
        proto = BB84Protocol(seed=10)
        result = proto.run(n_bits=1000)
        assert len(result.raw_key_alice) == len(result.raw_key_bob)

    def test_key_bits_binary(self):
        proto = BB84Protocol(seed=11)
        result = proto.run(n_bits=500)
        assert all(b in (0, 1) for b in result.raw_key_alice)

    def test_secure_key_rate_in_range(self):
        proto = BB84Protocol(channel_error_rate=0.03, seed=3)
        result = proto.run(2000)
        assert 0.0 <= result.secure_key_rate <= 1.0

    def test_invalid_n_bits(self):
        proto = BB84Protocol()
        with pytest.raises(ProtocolError):
            proto.run(n_bits=0)

    def test_invalid_channel_error_raises(self):
        with pytest.raises(ProtocolError):
            BB84Protocol(channel_error_rate=-0.1)
        with pytest.raises(ProtocolError):
            BB84Protocol(channel_error_rate=0.6)

    def test_invalid_eve_fraction_raises(self):
        with pytest.raises(ProtocolError):
            BB84Protocol(eve_intercept_fraction=1.5)

    def test_metadata_present(self):
        proto = BB84Protocol(seed=0)
        result = proto.run(100)
        assert "channel_error_rate" in result.metadata
        assert "eve_intercept_fraction" in result.metadata

    def test_repr(self):
        proto = BB84Protocol(seed=0)
        result = proto.run(200)
        r = repr(result)
        assert "BB84Result" in r
        assert "QBER" in r


# ── E91Protocol ───────────────────────────────────────────────────────────────

class TestE91Protocol:
    def test_basic_run_returns_result(self):
        proto = E91Protocol(bell_pair_fidelity=0.95, seed=0)
        result = proto.run(n_pairs=2000)
        assert isinstance(result, E91Result)

    def test_high_fidelity_gives_bell_violation(self):
        proto = E91Protocol(bell_pair_fidelity=0.95, channel_error_rate=0.0, seed=0)
        result = proto.run(n_pairs=5000)
        assert result.bell_violation
        assert result.chsh_s > 2.0

    def test_low_fidelity_no_violation(self):
        # Werner state with F=0.25 (maximally mixed) → S ≤ 2
        proto = E91Protocol(bell_pair_fidelity=0.25, channel_error_rate=0.0, seed=0)
        result = proto.run(n_pairs=5000)
        assert not result.bell_violation

    def test_chsh_within_tsirelson(self):
        proto = E91Protocol(bell_pair_fidelity=1.0, seed=1)
        result = proto.run(n_pairs=3000)
        assert result.chsh_s <= 2 * math.sqrt(2) + 0.1  # small statistical tolerance

    def test_key_bits_extracted(self):
        proto = E91Protocol(bell_pair_fidelity=0.9, seed=2)
        result = proto.run(n_pairs=3000)
        assert result.n_key_bits > 0

    def test_qber_in_range(self):
        proto = E91Protocol(bell_pair_fidelity=0.9, channel_error_rate=0.02, seed=3)
        result = proto.run(n_pairs=2000)
        assert 0.0 <= result.qber <= 0.5

    def test_secure_key_rate_in_range(self):
        proto = E91Protocol(bell_pair_fidelity=0.95, seed=4)
        result = proto.run(2000)
        assert 0.0 <= result.secure_key_rate <= 1.0

    def test_invalid_n_pairs_raises(self):
        proto = E91Protocol()
        with pytest.raises(ProtocolError):
            proto.run(n_pairs=2)

    def test_invalid_fidelity_raises(self):
        with pytest.raises(ProtocolError):
            E91Protocol(bell_pair_fidelity=1.5)

    def test_metadata_contains_fidelity(self):
        proto = E91Protocol(bell_pair_fidelity=0.9, seed=0)
        result = proto.run(1000)
        assert "bell_pair_fidelity" in result.metadata

    def test_repr(self):
        proto = E91Protocol(seed=0)
        result = proto.run(500)
        r = repr(result)
        assert "E91Result" in r
        assert "S=" in r


# ── TeleportationProtocol ─────────────────────────────────────────────────────

class TestTeleportationProtocol:
    def test_basic_run_returns_result(self):
        proto = TeleportationProtocol(channel_fidelity=0.95, seed=0)
        result = proto.run(n_states=100)
        assert isinstance(result, TeleportationResult)

    def test_perfect_channel_high_fidelity(self):
        proto = TeleportationProtocol(channel_fidelity=1.0, bsm_fidelity=1.0, seed=0)
        result = proto.run(n_states=500)
        assert result.average_fidelity > 0.9

    def test_fidelity_above_classical_for_good_channel(self):
        # F > 0.5 → avg fidelity > 2/3
        proto = TeleportationProtocol(channel_fidelity=0.9, bsm_fidelity=1.0, seed=1)
        result = proto.run(n_states=500)
        assert result.average_fidelity > 2.0 / 3.0

    def test_poor_channel_below_classical(self):
        # F = 0.25 (maximally mixed) → avg fidelity ≈ 0.5 < 2/3
        proto = TeleportationProtocol(channel_fidelity=0.25, bsm_fidelity=1.0, seed=2)
        result = proto.run(n_states=1000)
        assert result.average_fidelity < 2.0 / 3.0 + 0.05  # allow tolerance

    def test_classical_bits_sent(self):
        proto = TeleportationProtocol(seed=0)
        result = proto.run(n_states=50)
        assert result.classical_bits_sent == 100  # 2 bits per teleportation

    def test_n_teleported_correct(self):
        proto = TeleportationProtocol(seed=0)
        result = proto.run(n_states=200)
        assert result.n_teleported == 200

    def test_success_rate_in_range(self):
        proto = TeleportationProtocol(channel_fidelity=0.95, seed=5)
        result = proto.run(n_states=300)
        assert 0.0 <= result.success_rate <= 1.0

    def test_high_fidelity_channel_good_success_rate(self):
        proto = TeleportationProtocol(channel_fidelity=0.99, bsm_fidelity=0.99, seed=6)
        result = proto.run(n_states=500)
        assert result.success_rate > 0.6

    def test_metadata_contains_theoretical(self):
        proto = TeleportationProtocol(channel_fidelity=0.9, seed=0)
        result = proto.run(100)
        assert "theoretical_avg_fidelity" in result.metadata
        assert "classical_fidelity_limit" in result.metadata

    def test_invalid_n_states_raises(self):
        proto = TeleportationProtocol()
        with pytest.raises(ProtocolError):
            proto.run(n_states=0)

    def test_invalid_channel_fidelity_raises(self):
        with pytest.raises(ProtocolError):
            TeleportationProtocol(channel_fidelity=1.5)

    def test_invalid_bsm_fidelity_raises(self):
        with pytest.raises(ProtocolError):
            TeleportationProtocol(bsm_fidelity=0.0)

    def test_repr(self):
        proto = TeleportationProtocol(seed=0)
        result = proto.run(50)
        r = repr(result)
        assert "TeleportationResult" in r
        assert "avg_fidelity" in r


# ── DenseCodingProtocol ───────────────────────────────────────────────────────

class TestDenseCodingProtocol:
    def test_basic_run_returns_result(self):
        proto = DenseCodingProtocol(channel_fidelity=0.95, seed=0)
        result = proto.run(n_messages=500)
        assert isinstance(result, DenseCodingResult)

    def test_perfect_channel_near_zero_error(self):
        proto = DenseCodingProtocol(channel_fidelity=1.0, seed=0)
        result = proto.run(n_messages=1000)
        assert result.error_rate == pytest.approx(0.0, abs=0.01)

    def test_perfect_channel_capacity_two(self):
        proto = DenseCodingProtocol(channel_fidelity=1.0, seed=0)
        result = proto.run(n_messages=1000)
        assert result.channel_capacity == pytest.approx(2.0, abs=0.05)

    def test_noisy_channel_higher_error(self):
        proto_clean = DenseCodingProtocol(channel_fidelity=1.0, seed=1)
        proto_noisy = DenseCodingProtocol(channel_fidelity=0.5, seed=1)
        r_clean = proto_clean.run(1000)
        r_noisy = proto_noisy.run(1000)
        assert r_noisy.error_rate > r_clean.error_rate

    def test_bits_encoded_correct(self):
        proto = DenseCodingProtocol(seed=0)
        result = proto.run(n_messages=200)
        assert result.n_classical_bits_encoded == 400
        assert result.n_bits_sent_quantum == 200

    def test_n_messages_correct(self):
        proto = DenseCodingProtocol(seed=0)
        result = proto.run(n_messages=300)
        assert result.n_messages == 300

    def test_capacity_in_range(self):
        proto = DenseCodingProtocol(channel_fidelity=0.8, seed=2)
        result = proto.run(500)
        assert 0.0 <= result.channel_capacity <= 2.0

    def test_error_rate_in_range(self):
        proto = DenseCodingProtocol(channel_fidelity=0.75, seed=3)
        result = proto.run(500)
        assert 0.0 <= result.error_rate <= 1.0

    def test_metadata_present(self):
        proto = DenseCodingProtocol(channel_fidelity=0.9, seed=0)
        result = proto.run(100)
        assert "channel_fidelity" in result.metadata
        assert "theoretical_error_prob" in result.metadata

    def test_invalid_n_messages_raises(self):
        proto = DenseCodingProtocol()
        with pytest.raises(ProtocolError):
            proto.run(n_messages=0)

    def test_invalid_fidelity_raises(self):
        with pytest.raises(ProtocolError):
            DenseCodingProtocol(channel_fidelity=-0.1)

    def test_repr(self):
        proto = DenseCodingProtocol(seed=0)
        result = proto.run(100)
        r = repr(result)
        assert "DenseCodingResult" in r
        assert "capacity" in r


# ── Integration: BB84 over repeater chain ─────────────────────────────────────

class TestIntegration:
    def test_bb84_after_repeater_chain(self):
        """BB84 key rate is positive for a short direct link (0 hops, 5 km).

        Uses the one-way BB84 bound R = 1 − h₂((1−F)/2) already stored in
        NetworkResult.secret_key_rate to confirm the two modules interoperate.
        """
        from quantx_engine.quantum_network import simulate_repeater_chain
        net_result = simulate_repeater_chain(
            n_hops=0,
            segment_length_km=5,
            dephasing_per_segment=0.01,
        )
        # High-F direct link → non-zero key rate
        assert net_result.end_to_end_fidelity > 0.75
        assert net_result.secret_key_rate > 0.0
        # Shor-Preskill (two-way) rate should also be positive for low QBER
        qber = (1.0 - net_result.end_to_end_fidelity) / 2.0
        assert qber < 0.11  # below Shor-Preskill threshold
        assert bb84_key_rate(qber) > 0.0

    def test_teleportation_over_repeater_channel(self):
        """Teleportation fidelity using repeater-chain output fidelity."""
        from quantx_engine.quantum_network import simulate_repeater_chain
        net_result = simulate_repeater_chain(
            n_hops=1,
            segment_length_km=10,
            dephasing_per_segment=0.01,
        )
        f_tele = teleportation_fidelity(net_result.end_to_end_fidelity)
        assert f_tele > 2.0 / 3.0  # above classical limit

    def test_e91_detects_maximally_mixed_channel(self):
        """E91 with F=0.25 (maximally mixed) should not violate Bell inequality."""
        proto = E91Protocol(bell_pair_fidelity=0.26, channel_error_rate=0.0, seed=99)
        result = proto.run(n_pairs=5000)
        # Very low fidelity → no Bell violation
        assert result.chsh_s < 2.1  # small tolerance for sampling noise
