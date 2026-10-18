"""Tests for noise_char.py — M87 Noise Characterization Pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from quantx_engine.contrib.noise_char import (
    CharacterizedNoise,
    NoiseCharError,
    NoiseCharacterizer,
)
from quantx_engine.contrib.noise import DepolarizingNoise
from quantx_engine.contrib.rb_qv import RBResult, StandardRB
from quantx_engine.contrib.xeb import XEBResult, CrossEntropyBenchmarking
from quantx_engine.contrib.tomography import ProcessTomography


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def make_rb_result(epc: float, n: int = 5) -> RBResult:
    """Construct a minimal RBResult with the given EPC."""
    # p_decay consistent with EPC for 1-qubit: EPC = (d-1)/d * (1-p) => p = 1-2*EPC
    p_decay = 1.0 - 2.0 * epc
    lengths = list(range(1, n + 1))
    # A·p^m + B with A=0.5, B=0.5 (standard for 1-qubit starting in |0>)
    survival = [0.5 * p_decay ** m + 0.5 for m in lengths]
    return RBResult(
        sequence_lengths=lengths,
        survival_probs=survival,
        p_decay=p_decay,
        A=0.5,
        B=0.5,
        epc=epc,
        n_qubits=1,
    )


def make_xeb_result(
    per_layer_fidelity: float,
    depths: "list[int] | None" = None,
) -> XEBResult:
    """Construct an XEBResult matching F(d) = r^d."""
    if depths is None:
        depths = [1, 2, 4, 8]
    r = per_layer_fidelity
    fidelities = [r ** d for d in depths]
    return XEBResult(
        depths=depths,
        xeb_fidelities=fidelities,
        mean_fidelity=float(np.mean(fidelities)),
        n_qubits=2,
        n_trials=50,
    )


def identity_chi(n_qubits: int = 1) -> np.ndarray:
    """Chi matrix for identity unitary."""
    pt = ProcessTomography(n_qubits=n_qubits)
    d = 1 << n_qubits
    return pt.chi_matrix(np.eye(d, dtype=complex))


def x_gate_chi() -> np.ndarray:
    """Chi matrix for X gate."""
    pt = ProcessTomography(n_qubits=1)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    return pt.chi_matrix(X)


# ---------------------------------------------------------------------------
# CharacterizedNoise dataclass tests
# ---------------------------------------------------------------------------

class TestCharacterizedNoise:
    def test_fields_present(self):
        cn = CharacterizedNoise(
            depolarizing_prob=0.05,
            t1_estimate=2.0,
            t2_estimate=None,
            epc=0.025,
            xeb_fidelity=None,
            process_infidelity=None,
            confidence=1.0,
            sources=["rb"],
        )
        assert cn.depolarizing_prob == pytest.approx(0.05)
        assert cn.t1_estimate == pytest.approx(2.0)
        assert cn.t2_estimate is None
        assert cn.epc == pytest.approx(0.025)
        assert cn.xeb_fidelity is None
        assert cn.process_infidelity is None
        assert cn.confidence == pytest.approx(1.0)
        assert cn.sources == ["rb"]

    def test_to_noise_model_returns_depolarizing(self):
        cn = CharacterizedNoise(
            depolarizing_prob=0.03,
            t1_estimate=None,
            t2_estimate=None,
            epc=None,
            xeb_fidelity=None,
            process_infidelity=None,
            confidence=1.0,
            sources=["rb"],
        )
        nm = cn.to_noise_model()
        assert isinstance(nm, DepolarizingNoise)
        assert nm.prob == pytest.approx(0.03)

    def test_to_noise_model_zero_prob(self):
        cn = CharacterizedNoise(
            depolarizing_prob=0.0,
            t1_estimate=None,
            t2_estimate=None,
            epc=None,
            xeb_fidelity=None,
            process_infidelity=None,
            confidence=1.0,
            sources=[],
        )
        nm = cn.to_noise_model()
        assert nm.prob == pytest.approx(0.0)

    def test_sources_is_list(self):
        cn = CharacterizedNoise(
            depolarizing_prob=0.01,
            t1_estimate=None,
            t2_estimate=None,
            epc=None,
            xeb_fidelity=None,
            process_infidelity=None,
            confidence=0.8,
            sources=["rb", "xeb"],
        )
        assert isinstance(cn.sources, list)
        assert len(cn.sources) == 2


# ---------------------------------------------------------------------------
# NoiseCharacterizer constructor tests
# ---------------------------------------------------------------------------

class TestNoiseCharacterizerInit:
    def test_default_gate_time(self):
        nc = NoiseCharacterizer()
        assert nc.gate_time == pytest.approx(0.1)

    def test_custom_gate_time(self):
        nc = NoiseCharacterizer(gate_time=0.05)
        assert nc.gate_time == pytest.approx(0.05)

    def test_invalid_gate_time_zero(self):
        with pytest.raises(NoiseCharError, match="gate_time"):
            NoiseCharacterizer(gate_time=0.0)

    def test_invalid_gate_time_negative(self):
        with pytest.raises(NoiseCharError, match="gate_time"):
            NoiseCharacterizer(gate_time=-1.0)


# ---------------------------------------------------------------------------
# fit_from_rb tests
# ---------------------------------------------------------------------------

class TestFitFromRB:
    def test_known_epc_gives_correct_pdep(self):
        """p_dep = 2 * EPC for 1-qubit."""
        nc = NoiseCharacterizer(gate_time=0.1)
        rb = make_rb_result(epc=0.01)
        cn = nc.fit_from_rb(rb)
        assert cn.depolarizing_prob == pytest.approx(0.02, rel=1e-6)

    def test_epc_005_gives_pdep_01(self):
        nc = NoiseCharacterizer(gate_time=0.1)
        rb = make_rb_result(epc=0.05)
        cn = nc.fit_from_rb(rb)
        assert cn.depolarizing_prob == pytest.approx(0.10, rel=1e-6)

    def test_epc_zero_gives_zero_pdep(self):
        nc = NoiseCharacterizer(gate_time=0.1)
        rb = make_rb_result(epc=0.0)
        cn = nc.fit_from_rb(rb)
        assert cn.depolarizing_prob == pytest.approx(0.0, abs=1e-10)

    def test_epc_stored_in_result(self):
        nc = NoiseCharacterizer(gate_time=0.1)
        rb = make_rb_result(epc=0.02)
        cn = nc.fit_from_rb(rb)
        assert cn.epc == pytest.approx(0.02, rel=1e-6)

    def test_sources_contains_rb(self):
        nc = NoiseCharacterizer()
        cn = nc.fit_from_rb(make_rb_result(0.01))
        assert "rb" in cn.sources

    def test_xeb_fidelity_none_from_rb(self):
        nc = NoiseCharacterizer()
        cn = nc.fit_from_rb(make_rb_result(0.01))
        assert cn.xeb_fidelity is None

    def test_process_infidelity_none_from_rb(self):
        nc = NoiseCharacterizer()
        cn = nc.fit_from_rb(make_rb_result(0.01))
        assert cn.process_infidelity is None

    def test_t1_estimate_nonzero(self):
        """T1 estimate should be gate_time / p_dep."""
        nc = NoiseCharacterizer(gate_time=0.1)
        rb = make_rb_result(epc=0.05)  # p_dep = 0.10
        cn = nc.fit_from_rb(rb)
        assert cn.t1_estimate == pytest.approx(0.1 / 0.10, rel=1e-6)

    def test_t1_estimate_none_when_zero_noise(self):
        nc = NoiseCharacterizer(gate_time=0.1)
        rb = make_rb_result(epc=0.0)
        cn = nc.fit_from_rb(rb)
        assert cn.t1_estimate is None

    def test_pdep_clamped_to_one(self):
        """EPC=0.6 → 2*0.6 = 1.2, should be clamped to 1.0."""
        nc = NoiseCharacterizer()
        rb = make_rb_result(epc=0.6)
        cn = nc.fit_from_rb(rb)
        assert cn.depolarizing_prob <= 1.0

    def test_rb_result_missing_epc_raises(self):
        nc = NoiseCharacterizer()

        class BadResult:
            survival_probs = [0.9]
            sequence_lengths = [1]

        with pytest.raises(NoiseCharError):
            nc.fit_from_rb(BadResult())

    def test_rb_result_empty_survival_probs_raises(self):
        nc = NoiseCharacterizer()

        class EmptyResult:
            epc = 0.01
            survival_probs = []
            sequence_lengths = [1]

        with pytest.raises(NoiseCharError):
            nc.fit_from_rb(EmptyResult())

    def test_rb_result_empty_sequence_lengths_raises(self):
        nc = NoiseCharacterizer()

        class EmptyResult:
            epc = 0.01
            survival_probs = [0.9]
            sequence_lengths = []

        with pytest.raises(NoiseCharError):
            nc.fit_from_rb(EmptyResult())

    def test_confidence_is_one(self):
        nc = NoiseCharacterizer()
        cn = nc.fit_from_rb(make_rb_result(0.01))
        assert cn.confidence == pytest.approx(1.0)

    def test_real_standardrb_round_trip(self):
        """Run actual StandardRB, fit, and verify p_dep is in reasonable range."""
        rb_result = StandardRB(p_error=0.02, seed=42).run(
            [1, 5, 10, 20, 50], n_sequences=20
        )
        nc = NoiseCharacterizer(gate_time=0.1)
        cn = nc.fit_from_rb(rb_result)
        # p_dep = 2 * EPC; EPC should be near 0.01 for p_error=0.02
        assert 0.0 < cn.depolarizing_prob < 0.5
        nm = cn.to_noise_model()
        assert isinstance(nm, DepolarizingNoise)


# ---------------------------------------------------------------------------
# fit_from_xeb tests
# ---------------------------------------------------------------------------

class TestFitFromXEB:
    def test_known_per_layer_fidelity(self):
        """r=0.9 => p_dep ≈ 0.1."""
        nc = NoiseCharacterizer(gate_time=0.1)
        xeb = make_xeb_result(per_layer_fidelity=0.9, depths=[1, 2, 4, 8, 16])
        cn = nc.fit_from_xeb(xeb)
        assert cn.depolarizing_prob == pytest.approx(0.1, abs=0.01)

    def test_sources_contains_xeb(self):
        nc = NoiseCharacterizer()
        xeb = make_xeb_result(0.95)
        cn = nc.fit_from_xeb(xeb)
        assert "xeb" in cn.sources

    def test_epc_none_from_xeb(self):
        nc = NoiseCharacterizer()
        cn = nc.fit_from_xeb(make_xeb_result(0.9))
        assert cn.epc is None

    def test_xeb_fidelity_stored(self):
        nc = NoiseCharacterizer()
        xeb = make_xeb_result(0.9, depths=[1, 2, 4, 8])
        cn = nc.fit_from_xeb(xeb)
        assert cn.xeb_fidelity is not None
        assert 0.0 <= cn.xeb_fidelity <= 1.0

    def test_single_depth_raises(self):
        nc = NoiseCharacterizer()

        class SingleDepth:
            depths = [5]
            xeb_fidelities = [0.8]
            mean_fidelity = 0.8

        with pytest.raises(NoiseCharError, match="at least 2"):
            nc.fit_from_xeb(SingleDepth())

    def test_empty_depths_raises(self):
        nc = NoiseCharacterizer()

        class EmptyXEB:
            depths = []
            xeb_fidelities = []
            mean_fidelity = 0.0

        with pytest.raises(NoiseCharError):
            nc.fit_from_xeb(EmptyXEB())

    def test_mismatched_lengths_raises(self):
        nc = NoiseCharacterizer()

        class MismatchXEB:
            depths = [1, 2, 4]
            xeb_fidelities = [0.9, 0.8]
            mean_fidelity = 0.85

        with pytest.raises(NoiseCharError):
            nc.fit_from_xeb(MismatchXEB())

    def test_missing_attribute_raises(self):
        nc = NoiseCharacterizer()

        class BadXEB:
            depths = [1, 2]

        with pytest.raises(NoiseCharError):
            nc.fit_from_xeb(BadXEB())

    def test_pdep_in_unit_interval(self):
        nc = NoiseCharacterizer()
        xeb = make_xeb_result(0.5)
        cn = nc.fit_from_xeb(xeb)
        assert 0.0 <= cn.depolarizing_prob <= 1.0

    def test_real_xeb_round_trip(self):
        """Run actual CrossEntropyBenchmarking, fit, and check output is valid."""
        xeb_result = CrossEntropyBenchmarking(
            n_qubits=2, p_error=0.05, seed=7
        ).run(depths=[1, 2, 3, 4], n_trials=20)
        nc = NoiseCharacterizer(gate_time=0.1)
        cn = nc.fit_from_xeb(xeb_result)
        assert 0.0 <= cn.depolarizing_prob <= 1.0
        assert isinstance(cn.to_noise_model(), DepolarizingNoise)


# ---------------------------------------------------------------------------
# fit_from_process_tomography tests
# ---------------------------------------------------------------------------

class TestFitFromProcessTomography:
    def test_identity_chi_vs_itself_zero_infidelity(self):
        """Comparing identity process against itself => process_infidelity = 0."""
        chi = identity_chi(n_qubits=1)
        nc = NoiseCharacterizer()
        cn = nc.fit_from_process_tomography(chi, chi)
        assert cn.process_infidelity == pytest.approx(0.0, abs=1e-10)
        assert cn.depolarizing_prob == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.parametrize("p", [0.02, 0.1, 0.4])
    def test_depolarizing_channel_reads_back_its_own_p(self, p):
        """DepolarizingNoise(p), (1 - p)rho + p I/2, has the Pauli chi matrix
        diag(1 - 3p/4, p/4, p/4, p/4), so its process infidelity is 3p/4 and the fit must
        return p. (The old factor (d + 1)/d gave 9p/8: 0.1125 for p = 0.1.)"""
        target = identity_chi()
        chi = np.diag([1 - 0.75 * p, 0.25 * p, 0.25 * p, 0.25 * p]).astype(complex)
        assert np.trace(chi @ target.conj().T).real == pytest.approx(1 - 0.75 * p)
        cn = NoiseCharacterizer().fit_from_process_tomography(chi, target)
        assert cn.depolarizing_prob == pytest.approx(p, rel=1e-12)

    def test_sources_contains_process_tomography(self):
        chi = identity_chi()
        nc = NoiseCharacterizer()
        cn = nc.fit_from_process_tomography(chi, chi)
        assert "process_tomography" in cn.sources

    def test_epc_none_from_process_tomo(self):
        chi = identity_chi()
        nc = NoiseCharacterizer()
        cn = nc.fit_from_process_tomography(chi, chi)
        assert cn.epc is None

    def test_xeb_fidelity_none_from_process_tomo(self):
        chi = identity_chi()
        nc = NoiseCharacterizer()
        cn = nc.fit_from_process_tomography(chi, chi)
        assert cn.xeb_fidelity is None

    def test_process_infidelity_stored(self):
        chi_actual = identity_chi()
        chi_target = x_gate_chi()
        nc = NoiseCharacterizer()
        cn = nc.fit_from_process_tomography(chi_actual, chi_target)
        assert cn.process_infidelity is not None
        assert 0.0 <= cn.process_infidelity <= 1.0

    def test_default_target_is_identity(self):
        """Passing None target should compare against identity process."""
        chi_id = identity_chi()
        nc = NoiseCharacterizer()
        cn_with_target = nc.fit_from_process_tomography(chi_id, chi_id)
        cn_default_target = nc.fit_from_process_tomography(chi_id, None)
        assert cn_with_target.process_infidelity == pytest.approx(
            cn_default_target.process_infidelity, abs=1e-9
        )

    def test_non_square_chi_raises(self):
        nc = NoiseCharacterizer()
        bad_chi = np.ones((4, 3), dtype=complex)
        with pytest.raises(NoiseCharError):
            nc.fit_from_process_tomography(bad_chi)

    def test_wrong_chi_size_raises(self):
        """Chi matrix size must be 4^n; size=3 is invalid."""
        nc = NoiseCharacterizer()
        bad_chi = np.ones((3, 3), dtype=complex)
        with pytest.raises(NoiseCharError):
            nc.fit_from_process_tomography(bad_chi)

    def test_mismatched_target_shape_raises(self):
        nc = NoiseCharacterizer()
        chi4 = identity_chi(n_qubits=1)   # shape (4, 4)
        chi16 = identity_chi(n_qubits=2)  # shape (16, 16)
        with pytest.raises(NoiseCharError):
            nc.fit_from_process_tomography(chi4, chi16)

    def test_pdep_in_unit_interval(self):
        chi_actual = identity_chi()
        chi_target = x_gate_chi()
        nc = NoiseCharacterizer()
        cn = nc.fit_from_process_tomography(chi_actual, chi_target)
        assert 0.0 <= cn.depolarizing_prob <= 1.0


# ---------------------------------------------------------------------------
# fit (combined) tests
# ---------------------------------------------------------------------------

class TestFit:
    def test_no_sources_raises(self):
        nc = NoiseCharacterizer()
        with pytest.raises(NoiseCharError, match="at least one"):
            nc.fit()

    def test_rb_only(self):
        nc = NoiseCharacterizer()
        cn = nc.fit(rb_result=make_rb_result(0.02))
        assert cn.depolarizing_prob == pytest.approx(0.04, rel=1e-6)
        assert cn.sources == ["rb"]

    def test_xeb_only(self):
        nc = NoiseCharacterizer()
        xeb = make_xeb_result(0.9, depths=[1, 2, 4, 8])
        cn = nc.fit(xeb_result=xeb)
        assert 0.0 < cn.depolarizing_prob <= 1.0
        assert cn.sources == ["xeb"]

    def test_process_chi_only(self):
        nc = NoiseCharacterizer()
        chi = identity_chi()
        cn = nc.fit(process_chi=chi, target_chi=chi)
        assert cn.depolarizing_prob == pytest.approx(0.0, abs=1e-9)
        assert cn.sources == ["process_tomography"]

    def test_rb_and_xeb_combined(self):
        nc = NoiseCharacterizer()
        rb = make_rb_result(epc=0.05)         # p_dep = 0.10
        xeb = make_xeb_result(0.9, depths=[1, 2, 4, 8])  # p_dep ≈ 0.10
        cn = nc.fit(rb_result=rb, xeb_result=xeb)
        # Mean of two similar estimates
        assert cn.depolarizing_prob == pytest.approx(0.10, abs=0.02)
        assert "rb" in cn.sources
        assert "xeb" in cn.sources

    def test_all_sources_included(self):
        nc = NoiseCharacterizer()
        rb = make_rb_result(0.02)
        xeb = make_xeb_result(0.95, depths=[1, 2, 4, 8])
        chi = identity_chi()
        cn = nc.fit(rb_result=rb, xeb_result=xeb, process_chi=chi, target_chi=chi)
        assert set(cn.sources) == {"rb", "xeb", "process_tomography"}

    def test_confidence_high_when_sources_agree(self):
        """Two sources with very similar p_dep => high confidence."""
        nc = NoiseCharacterizer()
        rb = make_rb_result(epc=0.05)          # p_dep = 0.10
        xeb = make_xeb_result(0.9, depths=[1, 2, 4, 8, 16])  # p_dep ≈ 0.10
        cn = nc.fit(rb_result=rb, xeb_result=xeb)
        assert cn.confidence == pytest.approx(1.0)

    def test_confidence_between_zero_and_one(self):
        nc = NoiseCharacterizer()
        rb = make_rb_result(0.01)
        xeb = make_xeb_result(0.5, depths=[1, 2, 4, 8])
        cn = nc.fit(rb_result=rb, xeb_result=xeb)
        assert 0.0 <= cn.confidence <= 1.0

    def test_epc_propagated_from_rb(self):
        nc = NoiseCharacterizer()
        rb = make_rb_result(0.03)
        cn = nc.fit(rb_result=rb)
        assert cn.epc == pytest.approx(0.03, rel=1e-6)

    def test_xeb_fidelity_propagated(self):
        nc = NoiseCharacterizer()
        xeb = make_xeb_result(0.9, depths=[1, 2, 4, 8])
        cn = nc.fit(xeb_result=xeb)
        assert cn.xeb_fidelity is not None

    def test_process_infidelity_propagated(self):
        nc = NoiseCharacterizer()
        chi = identity_chi()
        cn = nc.fit(process_chi=chi, target_chi=chi)
        assert cn.process_infidelity is not None

    def test_depolarizing_prob_clamped(self):
        nc = NoiseCharacterizer()
        rb = make_rb_result(epc=0.6)  # would give 1.2 unclamped
        cn = nc.fit(rb_result=rb)
        assert 0.0 <= cn.depolarizing_prob <= 1.0

    def test_to_noise_model_from_fit(self):
        nc = NoiseCharacterizer()
        rb = make_rb_result(0.02)
        cn = nc.fit(rb_result=rb)
        nm = cn.to_noise_model()
        assert isinstance(nm, DepolarizingNoise)
        assert nm.prob == pytest.approx(cn.depolarizing_prob)
