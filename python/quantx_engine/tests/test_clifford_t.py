"""Tests for the Clifford+T fault-tolerant compilation module."""
from __future__ import annotations

import math
import pytest

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

pytestmark = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy required")

from quantx_engine.clifford_t import (
    GateSequence,
    CliffordGroup,
    SolovayKitaev,
    TCountOptimizer,
    CliffordTError,
    H_GATE,
    T_GATE,
    S_GATE,
    I_GATE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ry(theta: float) -> "np.ndarray":
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _matrices_close_up_to_phase(A: "np.ndarray", B: "np.ndarray", atol: float = 1e-8) -> bool:
    """Return True if A ≈ e^{iφ} B for some phase φ."""
    idx = int(np.argmax(np.abs(A)))
    i, j = divmod(idx, A.shape[1])
    b_val = B[i, j]
    if abs(b_val) < 1e-12:
        return bool(np.allclose(A, 0, atol=atol))
    phase = A[i, j] / b_val
    return bool(np.allclose(A, phase * B, atol=atol))


# ── GateSequence tests ────────────────────────────────────────────────────────

class TestGateSequence:
    def test_gate_sequence_t_count(self):
        seq = GateSequence(["H", "T", "H", "S"])
        assert seq.t_count() == 1

    def test_gate_sequence_total_count(self):
        seq = GateSequence(["H", "T", "H", "S"])
        assert seq.total_count() == 4

    def test_gate_sequence_to_matrix_identity(self):
        seq = GateSequence([])
        mat = seq.to_matrix()
        assert mat.shape == (2, 2)
        assert np.allclose(mat, np.eye(2, dtype=complex), atol=1e-10)

    def test_gate_sequence_to_matrix_H(self):
        seq = GateSequence(["H"])
        assert np.allclose(seq.to_matrix(), H_GATE, atol=1e-10)

    def test_gate_sequence_to_matrix_HH(self):
        seq = GateSequence(["H", "H"])
        assert np.allclose(seq.to_matrix(), I_GATE, atol=1e-10)

    def test_gate_sequence_inverse_HT(self):
        seq = GateSequence(["H", "T"])
        inv = seq.inverse()
        assert inv.gates == ["T†", "H"]

    def test_gate_sequence_inverse_t_count(self):
        seq = GateSequence(["H", "T", "S", "T†"])
        assert seq.inverse().t_count() == seq.t_count()

    def test_gate_sequence_to_dict(self):
        seq = GateSequence(["H", "T"])
        d = seq.to_dict()
        assert "gates" in d
        assert "t_count" in d
        assert "clifford_count" in d

    def test_gate_sequence_unknown_gate_raises(self):
        with pytest.raises(CliffordTError):
            GateSequence(["Q"]).to_matrix()

    def test_gate_sequence_append(self):
        seq = GateSequence(["H"]).append("T")
        assert seq.gates == ["H", "T"]

    def test_gate_sequence_prepend(self):
        seq = GateSequence(["H"]).prepend("T")
        assert seq.gates == ["T", "H"]

    def test_gate_sequence_clifford_count(self):
        seq = GateSequence(["H", "T", "S", "T†"])
        # H and S are Clifford; T and T† are not
        assert seq.clifford_count() == 2

    def test_gate_sequence_repr(self):
        seq = GateSequence(["H", "T"])
        r = repr(seq)
        assert "GateSequence" in r
        assert "t_count=1" in r


# ── CliffordGroup tests ───────────────────────────────────────────────────────

class TestCliffordGroup:
    def test_clifford_group_size(self):
        elements = CliffordGroup.generate()
        assert len(elements) == 24

    def test_clifford_group_identify_H(self):
        result = CliffordGroup.identify(H_GATE)
        assert result is not None

    def test_clifford_group_identify_H_matrix(self):
        result = CliffordGroup.identify(H_GATE)
        assert result is not None
        mat = result.to_matrix()
        assert _matrices_close_up_to_phase(mat, H_GATE)

    def test_clifford_group_identify_non_clifford(self):
        # T is not a Clifford gate
        result = CliffordGroup.identify(T_GATE)
        assert result is None

    def test_clifford_group_random(self):
        seq = CliffordGroup.random(seed=0)
        assert isinstance(seq, GateSequence)

    def test_clifford_group_all_unitary(self):
        """All 24 elements should be unitary."""
        for seq in CliffordGroup.generate():
            mat = seq.to_matrix()
            prod = mat.conj().T @ mat
            assert np.allclose(prod, np.eye(2), atol=1e-9)

    def test_clifford_group_identify_S(self):
        result = CliffordGroup.identify(S_GATE)
        assert result is not None
        assert _matrices_close_up_to_phase(result.to_matrix(), S_GATE)


# ── SolovayKitaev tests ───────────────────────────────────────────────────────

class TestSolovayKitaev:
    def test_sk_compile_clifford_exact(self):
        sk = SolovayKitaev(epsilon=0.01, max_depth=3)
        err = sk.approximation_error(H_GATE)
        assert err < 0.1

    def test_sk_compile_produces_gate_sequence(self):
        sk = SolovayKitaev(epsilon=0.01, max_depth=3)
        result = sk.compile(H_GATE)
        assert isinstance(result, GateSequence)

    def test_sk_compile_t_gate(self):
        sk = SolovayKitaev(epsilon=0.01, max_depth=3)
        err = sk.approximation_error(T_GATE)
        assert err < 0.1  # T is in the basis

    def test_sk_approximation_error_small(self):
        sk = SolovayKitaev(epsilon=0.01, max_depth=3)
        err = sk.approximation_error(H_GATE)
        assert err < 0.1

    def test_sk_invalid_epsilon(self):
        with pytest.raises(CliffordTError):
            SolovayKitaev(epsilon=0)

    def test_sk_invalid_epsilon_negative(self):
        with pytest.raises(CliffordTError):
            SolovayKitaev(epsilon=-0.5)

    def test_sk_compile_arbitrary_rotation(self):
        sk = SolovayKitaev(epsilon=0.01, max_depth=3)
        U = _ry(math.pi / 5)
        result = sk.compile(U)
        assert isinstance(result, GateSequence)

    def test_sk_build_basis_non_empty(self):
        sk = SolovayKitaev(epsilon=0.01, max_depth=2)
        assert len(sk._basis) > 0

    def test_sk_approximation_error_nonnegative(self):
        sk = SolovayKitaev(epsilon=0.01, max_depth=3)
        err = sk.approximation_error(_ry(math.pi / 7))
        assert err >= 0.0


# ── TCountOptimizer tests ─────────────────────────────────────────────────────

class TestTCountOptimizer:
    def test_optimizer_tt_to_s(self):
        opt = TCountOptimizer()
        result = opt.optimize(GateSequence(["T", "T"]))
        assert result.t_count() == 0

    def test_optimizer_t_tdag_cancel(self):
        opt = TCountOptimizer()
        result = opt.optimize(GateSequence(["T", "T†"]))
        assert result.t_count() == 0
        assert len(result.gates) == 0

    def test_optimizer_hh_cancel(self):
        opt = TCountOptimizer()
        result = opt.optimize(GateSequence(["H", "H"]))
        assert len(result.gates) == 0

    def test_optimizer_no_change_needed(self):
        opt = TCountOptimizer()
        seq = GateSequence(["H", "T", "H"])
        result = opt.optimize(seq)
        assert result.gates == ["H", "T", "H"]

    def test_optimizer_reduction_count(self):
        # TTTT → SS → Z; t_count goes to 0
        opt = TCountOptimizer()
        seq = GateSequence(["T", "T", "T", "T"])
        result = opt.optimize(seq)
        assert result.t_count() == 0

    def test_optimizer_to_dict(self):
        opt = TCountOptimizer()
        d = opt.to_dict()
        assert "optimizer" in d
        assert d["optimizer"] == "peephole"

    def test_optimizer_ss_to_z(self):
        opt = TCountOptimizer()
        result = opt.optimize(GateSequence(["S", "S"]))
        assert result.gates == ["Z"]

    def test_optimizer_zz_cancel(self):
        opt = TCountOptimizer()
        result = opt.optimize(GateSequence(["Z", "Z"]))
        assert len(result.gates) == 0

    def test_optimizer_t_count_reduction_method(self):
        opt = TCountOptimizer()
        seq = GateSequence(["T", "T"])
        assert opt.t_count_reduction(seq) == 2

    def test_optimizer_tdag_tdag_to_sdag(self):
        opt = TCountOptimizer()
        result = opt.optimize(GateSequence(["T†", "T†"]))
        assert result.t_count() == 0
        assert result.gates == ["S†"]
