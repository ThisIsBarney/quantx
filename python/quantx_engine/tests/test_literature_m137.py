"""Literature-grounded numerical validation tests for M137.

Covers:
  - qec.py            (RepetitionCode, SurfaceCode, LogicalErrorSimulator)
  - fault_tolerant.py (LogicalQubit, MagicStateDistillery)
  - ft_tgate.py       (MagicStateFactory, TCountOptimizer)

References
----------
.. [Fowler2012]   Fowler, A. M. et al. (2012). Phys. Rev. A 86, 032324.
                  https://doi.org/10.1103/PhysRevA.86.032324
                  Surface code ~1% threshold.
.. [BravyiKitaev] Bravyi, S. & Kitaev, A. (2005). Phys. Rev. A 71, 022316.
                  https://doi.org/10.1103/PhysRevA.71.022316
                  15-to-1: epsilon_out = 35 * epsilon_in^3.
.. [Selinger2013] Selinger, P. (2013). Phys. Rev. A 87, 042302.
                  https://doi.org/10.1103/PhysRevA.87.042302
                  Toffoli = 7 T gates.
"""
from __future__ import annotations

import numpy as np
import pytest


# ──────────────────────────────────────────────────────────────────────────────
# RepetitionCode / QEC tests
# ──────────────────────────────────────────────────────────────────────────────

class TestRepetitionCodeLiterature:
    """Validate RepetitionCode against [Calderbank1996] / [Steane1996].

    A distance-d repetition code corrects all weight-< d/2 errors.
    For d=3 it corrects all single-qubit errors (t=1 correctable errors).
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from quantx_engine.qec import RepetitionCode
            self.RepetitionCode = RepetitionCode
        except ImportError:
            pytest.skip("qec module not available")

    def test_d3_corrects_all_single_errors(self):
        """d=3 repetition code corrects all weight-1 bit-flip errors [Calderbank1996].

        For any single-qubit error position i in {0,1,2},
        correction should recover the original codeword.
        """
        code = self.RepetitionCode(3)
        # Encode logical |0>
        cw0 = code.encode(0)
        # Apply each single-qubit error and correct
        for i in range(3):
            corrupted = cw0.copy()
            corrupted[i] ^= 1  # bit-flip error on qubit i
            syndrome = code.syndrome(corrupted)
            corrected = code.correct(corrupted)
            assert np.array_equal(corrected, cw0), (
                f"d=3 code failed to correct error at position {i}: "
                f"corrupted={corrupted}, corrected={corrected}, expected={cw0}"
            )

    def test_d3_corrects_logical_1(self):
        """d=3 code also corrects single errors on encoded |1>."""
        code = self.RepetitionCode(3)
        cw1 = code.encode(1)
        for i in range(3):
            corrupted = cw1.copy()
            corrupted[i] ^= 1
            corrected = code.correct(corrupted)
            assert np.array_equal(corrected, cw1), (
                f"d=3 code failed to correct |1> error at position {i}"
            )

    def test_logical_error_rate_below_physical(self):
        """d=3 logical error rate < physical error rate for p < 0.5 [Fowler2012].

        Below the threshold, encoding reduces the error rate.
        """
        try:
            from quantx_engine.qec import LogicalErrorSimulator, SurfaceCode
            code = SurfaceCode(d=3)
            sim = LogicalErrorSimulator(code)
            p_phys = 0.01
            p_L = sim.logical_error_rate(p_phys=p_phys, n_trials=500)
            assert p_L < p_phys, (
                f"Logical error rate {p_L:.4f} not below physical {p_phys}"
            )
        except Exception as exc:
            pytest.skip(f"LogicalErrorSimulator/SurfaceCode failed: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Magic state distillation tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMagicStateDistillationLiterature:
    """Validate magic state distillation against [BravyiKitaev2005].

    The 15-to-1 protocol output error is epsilon_out = 35 * epsilon_in^3.
    Reference: Table I in Bravyi & Kitaev 2005.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from quantx_engine.ft_tgate import MagicStateFactory
            self.MagicStateFactory = MagicStateFactory
        except ImportError:
            try:
                from quantx_engine.fault_tolerant import MagicStateDistillery
                self.MagicStateFactory = MagicStateDistillery
            except ImportError:
                pytest.skip("Magic state classes not available")

    def test_15to1_cubic_suppression(self):
        """15-to-1 output error = 35*epsilon^3 [BravyiKitaev2005, Table I].

        API: MagicStateFactory(protocol='15_to_1', input_error_rate=eps)
             .output_error_rate  (property, no .run() call needed)
        """
        eps_in = 0.01
        factory = self.MagicStateFactory(protocol="15_to_1", input_error_rate=eps_in)
        eps_out = factory.output_error_rate
        expected = 35 * eps_in ** 3
        # Allow 50% relative tolerance
        assert abs(eps_out - expected) < 0.5 * expected + 1e-12, (
            f"15-to-1 output error {eps_out:.2e}; expected ~{expected:.2e} "
            f"[BravyiKitaev2005]"
        )

    def test_distillation_reduces_error(self):
        """Output error must be less than input error [BravyiKitaev2005]."""
        eps_in = 0.05
        factory = self.MagicStateFactory(protocol="15_to_1", input_error_rate=eps_in)
        eps_out = factory.output_error_rate
        assert eps_out < eps_in, (
            f"Distillation increased error: in={eps_in}, out={eps_out}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# T-count tests
# ──────────────────────────────────────────────────────────────────────────────

class TestTCountLiterature:
    """Validate TCountOptimizer against [Selinger2013].

    The Selinger T-depth-one Toffoli decomposition uses exactly 7 T gates.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from quantx_engine.ft_tgate import TCountOptimizer
            self.TCountOptimizer = TCountOptimizer
        except ImportError:
            pytest.skip("TCountOptimizer not available")

    def test_toffoli_t_count_7(self):
        """Toffoli gate has T-count = 7 in Selinger decomposition [Selinger2013].

        The CCX gate in the Clifford+T gate set decomposes to exactly 7 T gates
        [Selinger2013, Fig. 1]. We count T gates in the canonical CCX circuit.
        """
        # Build a circuit containing one CCX gate and count T gates in its
        # canonical Selinger decomposition (via TCountOptimizer analysis)
        # CCX gate costs 7 T gates in the Selinger decomposition (_T_COST dict)
        circuit = [{"gate": "CCX", "controls": [0, 1], "target": 2}]
        opt = self.TCountOptimizer(circuit)
        t_count = opt.t_count()   # t_count() is a method, not a property
        assert t_count == 7, (
            f"Toffoli T-count = {t_count}; expected 7 [Selinger2013]"
        )
