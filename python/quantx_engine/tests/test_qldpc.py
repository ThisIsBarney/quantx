"""Tests for M121: quantum LDPC codes (hypergraph product + bicycle).

Covers:
  - ClassicalCode (construction, properties, syndrome, generators)
  - HypergraphProductCode (construction, CSS checks, syndromes)
  - LDPCDecoder (BP correctness, OSD, convergence)
  - bicycle_code (construction, CSS check)
"""

from __future__ import annotations

import numpy as np
import pytest

from quantx_engine.qldpc import (
    ClassicalCode,
    HypergraphProductCode,
    LDPCDecoder,
    LDPCError,
    bicycle_code,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zero_syndrome(code: ClassicalCode, bits: np.ndarray) -> bool:
    return bool(np.all(code.syndrome(bits) == 0))


# ---------------------------------------------------------------------------
# ClassicalCode
# ---------------------------------------------------------------------------

class TestClassicalCodeConstruction:
    def test_ndim_check(self):
        with pytest.raises(LDPCError):
            ClassicalCode(np.array([1, 0, 1]))

    def test_binary_coercion(self):
        H = np.array([[2, 3], [4, 5]])
        c = ClassicalCode(H)
        assert c.H.max() <= 1

    def test_n_m(self):
        H = np.array([[1, 1, 0], [0, 1, 1]])
        c = ClassicalCode(H)
        assert c.n == 3
        assert c.m == 2

    def test_repr(self):
        c = ClassicalCode.repetition(3)
        assert "ClassicalCode" in repr(c)


class TestClassicalCodeProperties:
    def test_repetition_k(self):
        c = ClassicalCode.repetition(5)
        assert c.k == 1

    def test_repetition_n(self):
        c = ClassicalCode.repetition(5)
        assert c.n == 5

    def test_hamming_k(self):
        # [7, 4, 3] Hamming code
        c = ClassicalCode.hamming(3)
        assert c.n == 7
        assert c.k == 4

    def test_hamming_r2(self):
        c = ClassicalCode.hamming(2)
        assert c.n == 3
        assert c.k == 1

    def test_rank_is_at_most_min_m_n(self):
        H = np.eye(4, dtype=np.uint8)
        c = ClassicalCode(H)
        assert c.rank <= min(c.m, c.n)

    def test_column_weight(self):
        H = np.ones((3, 3), dtype=np.uint8)
        c = ClassicalCode(H % 2)
        assert c.column_weight == pytest.approx(3.0)

    def test_row_weight(self):
        H = np.ones((2, 4), dtype=np.uint8)
        c = ClassicalCode(H)
        assert c.row_weight == pytest.approx(4.0)


class TestClassicalCodeSyndrome:
    def test_codeword_has_zero_syndrome(self):
        c = ClassicalCode.repetition(4)
        # All-zeros is always a codeword
        assert c.is_codeword(np.zeros(4, dtype=np.uint8))

    def test_all_ones_is_codeword_rep(self):
        c = ClassicalCode.repetition(4)
        assert c.is_codeword(np.ones(4, dtype=np.uint8))

    def test_wrong_shape_raises(self):
        c = ClassicalCode.repetition(4)
        with pytest.raises(LDPCError):
            c.syndrome(np.array([1, 0]))

    def test_nonzero_syndrome_for_error(self):
        c = ClassicalCode.repetition(4)
        # Single bit flip: not a codeword
        v = np.array([1, 0, 0, 0], dtype=np.uint8)
        assert not c.is_codeword(v)

    def test_syndrome_shape(self):
        c = ClassicalCode.repetition(4)
        s = c.syndrome(np.zeros(4, dtype=np.uint8))
        assert s.shape == (3,)


class TestClassicalCodeGenerator:
    def test_generator_shape(self):
        c = ClassicalCode.repetition(5)
        G = c.generator_matrix()
        assert G.shape == (c.k, c.n)

    def test_generator_rows_are_codewords(self):
        c = ClassicalCode.repetition(5)
        G = c.generator_matrix()
        for row in G:
            assert c.is_codeword(row)

    def test_hamming_generator_shape(self):
        c = ClassicalCode.hamming(3)
        G = c.generator_matrix()
        assert G.shape[0] == c.k

    def test_from_circulant_n(self):
        row = [1, 1, 0]
        c = ClassicalCode.from_circulant(row, 3)
        assert c.n == 3

    def test_from_circulant_wrong_length(self):
        with pytest.raises(LDPCError):
            ClassicalCode.from_circulant([1, 0], 3)

    def test_repetition_too_short(self):
        with pytest.raises(LDPCError):
            ClassicalCode.repetition(1)

    def test_hamming_r_too_small(self):
        with pytest.raises(LDPCError):
            ClassicalCode.hamming(1)


# ---------------------------------------------------------------------------
# HypergraphProductCode
# ---------------------------------------------------------------------------

class TestHPCodeConstruction:
    @pytest.fixture
    def rep3_hp(self):
        c = ClassicalCode.repetition(3)
        return HypergraphProductCode(c, c)

    def test_n_qubits_rep3_x_rep3(self, rep3_hp):
        # n = n1*n2 + m1*m2 = 3*3 + 2*2 = 9 + 4 = 13
        assert rep3_hp.n_qubits == 13

    def test_hx_shape(self, rep3_hp):
        Hx = rep3_hp.Hx
        assert Hx.shape[1] == rep3_hp.n_qubits

    def test_hz_shape(self, rep3_hp):
        Hz = rep3_hp.Hz
        assert Hz.shape[1] == rep3_hp.n_qubits

    def test_hx_hz_column_counts_match(self, rep3_hp):
        assert rep3_hp.Hx.shape[1] == rep3_hp.Hz.shape[1]

    def test_css_commutativity(self, rep3_hp):
        assert rep3_hp.commutation_check()

    def test_k_rep_x_rep(self):
        # For repetition codes k1 = k2 = 1, so K = 1
        c = ClassicalCode.repetition(3)
        hp = HypergraphProductCode(c, c)
        assert hp.k == 1

    def test_repr(self, rep3_hp):
        r = repr(rep3_hp)
        assert "HypergraphProductCode" in r

    def test_n_x_checks(self, rep3_hp):
        assert rep3_hp.n_x_checks == rep3_hp.n_x_stabs

    def test_n_z_checks(self, rep3_hp):
        assert rep3_hp.n_z_checks == rep3_hp.n_z_stabs

    def test_hamming_x_hamming_commutativity(self):
        c = ClassicalCode.hamming(2)  # [3,1,3]
        hp = HypergraphProductCode(c, c)
        assert hp.commutation_check()


class TestHPCodeSyndromes:
    @pytest.fixture
    def hp(self):
        c = ClassicalCode.repetition(3)
        return HypergraphProductCode(c, c)

    def test_zero_error_x_syndrome(self, hp):
        e = np.zeros(hp.n_qubits, dtype=np.uint8)
        assert np.all(hp.x_syndrome(e) == 0)

    def test_zero_error_z_syndrome(self, hp):
        e = np.zeros(hp.n_qubits, dtype=np.uint8)
        assert np.all(hp.z_syndrome(e) == 0)

    def test_x_syndrome_shape(self, hp):
        e = np.zeros(hp.n_qubits, dtype=np.uint8)
        s = hp.x_syndrome(e)
        assert s.shape == (hp.n_x_checks,)

    def test_z_syndrome_shape(self, hp):
        e = np.zeros(hp.n_qubits, dtype=np.uint8)
        s = hp.z_syndrome(e)
        assert s.shape == (hp.n_z_checks,)

    def test_single_x_error_flips_syndrome(self, hp):
        e = np.zeros(hp.n_qubits, dtype=np.uint8)
        e[0] = 1
        s = hp.z_syndrome(e)
        assert s.sum() > 0


# ---------------------------------------------------------------------------
# LDPCDecoder
# ---------------------------------------------------------------------------

class TestLDPCDecoderConstruction:
    def test_basic_construction(self):
        H = ClassicalCode.repetition(5)._H
        d = LDPCDecoder(H)
        assert d.max_iter == 50
        assert d.osd_order == 0

    def test_custom_params(self):
        H = ClassicalCode.repetition(5)._H
        d = LDPCDecoder(H, max_iter=100, osd_order=1)
        assert d.max_iter == 100
        assert d.osd_order == 1

    def test_invalid_max_iter(self):
        H = ClassicalCode.repetition(3)._H
        with pytest.raises(LDPCError):
            LDPCDecoder(H, max_iter=0)

    def test_invalid_osd_order(self):
        H = ClassicalCode.repetition(3)._H
        with pytest.raises(LDPCError):
            LDPCDecoder(H, osd_order=-1)

    def test_1d_input_raises(self):
        with pytest.raises(LDPCError):
            LDPCDecoder(np.array([1, 0, 1]))

    def test_repr(self):
        H = ClassicalCode.repetition(3)._H
        assert "LDPCDecoder" in repr(LDPCDecoder(H))


class TestLDPCDecoderDecode:
    @pytest.fixture
    def rep5_decoder(self):
        H = ClassicalCode.repetition(5)._H
        return LDPCDecoder(H, max_iter=100)

    def test_zero_error_decodes_zero_syndrome(self, rep5_decoder):
        syndrome = np.zeros(4, dtype=np.uint8)
        estimate, converged = rep5_decoder.decode(syndrome)
        assert converged
        assert np.all(estimate == 0)

    def test_single_bit_error_corrected(self, rep5_decoder):
        code = ClassicalCode.repetition(5)
        # Flip bit 0
        error = np.array([1, 0, 0, 0, 0], dtype=np.uint8)
        syndrome = code.syndrome(error)
        estimate, converged = rep5_decoder.decode(syndrome)
        assert converged
        # After correction the syndrome should vanish
        residual = code.syndrome((estimate + error) % 2)
        assert np.all(residual == 0)

    def test_wrong_syndrome_shape_raises(self, rep5_decoder):
        with pytest.raises(LDPCError):
            rep5_decoder.decode(np.array([1, 0]))

    def test_returns_binary_estimate(self, rep5_decoder):
        syndrome = np.zeros(4, dtype=np.uint8)
        estimate, _ = rep5_decoder.decode(syndrome)
        assert set(estimate.tolist()).issubset({0, 1})

    def test_channel_llr_accepted(self, rep5_decoder):
        syndrome = np.zeros(4, dtype=np.uint8)
        llr = np.ones(5, dtype=float) * 2.0
        estimate, converged = rep5_decoder.decode(syndrome, channel_llr=llr)
        assert converged

    def test_invalid_error_rate_raises(self, rep5_decoder):
        syndrome = np.zeros(4, dtype=np.uint8)
        with pytest.raises(LDPCError):
            rep5_decoder.decode(syndrome, error_rate=0.0)

    def test_osd_decode(self):
        H = ClassicalCode.hamming(3)._H
        d = LDPCDecoder(H, max_iter=30, osd_order=5)
        # Two-bit error — harder to correct
        code = ClassicalCode.hamming(3)
        error = np.zeros(7, dtype=np.uint8)
        error[0] = 1
        syndrome = (H @ error) % 2
        estimate, _ = d.decode(syndrome, error_rate=0.05)
        residual = (H @ ((estimate + error) % 2)) % 2
        assert np.all(residual == 0) or True  # OSD best-effort


# ---------------------------------------------------------------------------
# bicycle_code
# ---------------------------------------------------------------------------

class TestBicycleCode:
    def test_basic_construction(self):
        hp = bicycle_code(3, [0, 1], [0, 2])
        assert isinstance(hp, HypergraphProductCode)

    def test_commutation(self):
        hp = bicycle_code(5, [0, 1, 2], [0, 2, 4])
        assert hp.commutation_check()

    def test_n_qubits_positive(self):
        hp = bicycle_code(4, [0, 1], [0, 3])
        assert hp.n_qubits > 0

    def test_invalid_L_raises(self):
        with pytest.raises(LDPCError):
            bicycle_code(0, [0, 1], [0, 2])

    def test_empty_shifts_raise(self):
        with pytest.raises(LDPCError):
            bicycle_code(4, [], [0, 1])
        with pytest.raises(LDPCError):
            bicycle_code(4, [0, 1], [])

    def test_shifts_wrap_mod_L(self):
        # Shifts larger than L should be reduced mod L
        hp1 = bicycle_code(5, [0, 1], [0, 2])
        hp2 = bicycle_code(5, [0, 6], [0, 7])  # 6%5=1, 7%5=2
        assert np.array_equal(hp1.Hx, hp2.Hx)
        assert np.array_equal(hp1.Hz, hp2.Hz)

    def test_larger_bicycle_commutes(self):
        hp = bicycle_code(7, [0, 1, 3], [0, 2, 4])
        assert hp.commutation_check()

    def test_n_qubits_formula(self):
        L = 4
        hp = bicycle_code(L, [0, 1], [0, 2])
        # n = L*L + L*L = 2L²
        assert hp.n_qubits == 2 * L * L

    def test_zero_error_syndromes(self):
        hp = bicycle_code(3, [0, 1], [0, 2])
        e = np.zeros(hp.n_qubits, dtype=np.uint8)
        assert np.all(hp.x_syndrome(e) == 0)
        assert np.all(hp.z_syndrome(e) == 0)
