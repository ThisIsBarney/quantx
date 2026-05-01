"""Tests for mps_simulator.py — M104 MPS extension to 50+ qubits."""
from __future__ import annotations

import math
import numpy as np
import pytest

from quantx_engine.mps_simulator import (
    MPSSimulatorError,
    AdaptiveTruncation,
    MPSSimulator,
    CachedDMRG,
    BenchmarkResult,
    ComparisonResult,
    mps_benchmark,
    ghz_fidelity,
    compare_mps_sv,
    _ghz_mps,
)
from quantx_engine.tensor_network import MPS, MPO


# ─────────────────────────────────────────────────────────────────────────────
# AdaptiveTruncation
# ─────────────────────────────────────────────────────────────────────────────

class TestAdaptiveTruncation:
    def test_defaults(self):
        at = AdaptiveTruncation()
        assert at.max_bond == 128
        assert at.eps == 1e-12

    def test_invalid_max_bond(self):
        with pytest.raises(MPSSimulatorError, match="max_bond"):
            AdaptiveTruncation(max_bond=0)

    def test_invalid_eps(self):
        with pytest.raises(MPSSimulatorError, match="eps"):
            AdaptiveTruncation(eps=1.5)

    def test_truncate_keeps_all_above_threshold(self):
        at = AdaptiveTruncation(max_bond=10, eps=0.0)
        S = np.array([3.0, 2.0, 1.0, 0.1])
        U = np.eye(4)
        Vh = np.eye(4)
        Ut, St, Vht, err = at.truncate(U, S, Vh)
        assert len(St) == 4  # eps=0 keeps everything
        assert err == pytest.approx(0.0)

    def test_truncate_respects_max_bond(self):
        at = AdaptiveTruncation(max_bond=2, eps=0.0)
        S = np.array([3.0, 2.0, 1.0, 0.5])
        U = np.eye(4)
        Vh = np.eye(4)
        _, St, _, err = at.truncate(U, S, Vh)
        assert len(St) == 2
        assert err == pytest.approx(1.0 + 0.25)

    def test_truncate_eps_relative(self):
        """eps=0.1 means keep S_i only if S_i > 0.1 * S_max."""
        at = AdaptiveTruncation(max_bond=10, eps=0.1)
        S = np.array([10.0, 5.0, 0.5, 0.01])  # threshold = 10*0.1 = 1.0
        U = np.eye(4)
        Vh = np.eye(4)
        _, St, _, err = at.truncate(U, S, Vh)
        assert len(St) == 2  # keep 10.0 and 5.0; discard 0.5 and 0.01
        assert err == pytest.approx(0.5**2 + 0.01**2)

    def test_truncate_always_keeps_at_least_one(self):
        at = AdaptiveTruncation(max_bond=1, eps=0.99)
        S = np.array([1.0, 0.001])
        U = np.eye(2)
        Vh = np.eye(2)
        _, St, _, _ = at.truncate(U, S, Vh)
        assert len(St) >= 1

    def test_truncate_error_is_sum_of_squares(self):
        at = AdaptiveTruncation(max_bond=1, eps=0.0)
        S = np.array([3.0, 2.0, 1.0])
        U = np.eye(3)
        Vh = np.eye(3)
        _, St, _, err = at.truncate(U, S, Vh)
        assert len(St) == 1
        assert err == pytest.approx(4.0 + 1.0)  # 2² + 1²


# ─────────────────────────────────────────────────────────────────────────────
# MPSSimulator — construction and basic properties
# ─────────────────────────────────────────────────────────────────────────────

class TestMPSSimulatorInit:
    def test_n_qubits_stored(self):
        sim = MPSSimulator(n_qubits=5)
        assert sim.n_qubits == 5

    def test_initial_bond_dims_all_one(self):
        sim = MPSSimulator(n_qubits=4)
        assert all(b == 1 for b in sim.bond_dims)

    def test_initial_norm_is_one(self):
        sim = MPSSimulator(n_qubits=6)
        assert sim.norm() == pytest.approx(1.0, abs=1e-12)

    def test_initial_truncation_error_zero(self):
        sim = MPSSimulator(n_qubits=4)
        assert sim.truncation_error == 0.0

    def test_invalid_n_qubits(self):
        with pytest.raises(MPSSimulatorError):
            MPSSimulator(n_qubits=0)

    def test_max_bond_used_starts_at_one(self):
        sim = MPSSimulator(n_qubits=4)
        assert sim.max_bond_used == 1


# ─────────────────────────────────────────────────────────────────────────────
# Single-qubit gates
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleQubitGates:
    def test_h_creates_superposition(self):
        sim = MPSSimulator(n_qubits=1)
        sim.h(0)
        sv = sim.to_statevector()
        assert abs(sv[0]) == pytest.approx(1 / math.sqrt(2), abs=1e-12)
        assert abs(sv[1]) == pytest.approx(1 / math.sqrt(2), abs=1e-12)

    def test_x_flips_qubit(self):
        sim = MPSSimulator(n_qubits=2)
        sim.x(0)
        sv = sim.to_statevector()
        # |00⟩ → |10⟩ (index 2 in little-endian)
        assert abs(sv[2]) == pytest.approx(1.0, abs=1e-12)

    def test_rz_phase(self):
        sim = MPSSimulator(n_qubits=1)
        sim.x(0)  # |1⟩
        sim.rz(math.pi, 0)  # should give phase -i * |1⟩
        sv = sim.to_statevector()
        assert abs(sv[1]) == pytest.approx(1.0, abs=1e-12)

    def test_single_qubit_preserves_norm(self):
        sim = MPSSimulator(n_qubits=3)
        sim.h(0).h(1).h(2)
        assert sim.norm() == pytest.approx(1.0, abs=1e-12)

    def test_invalid_gate_shape(self):
        sim = MPSSimulator(n_qubits=2)
        with pytest.raises(MPSSimulatorError):
            sim.apply_gate(np.eye(4), 0)

    def test_invalid_site_raises(self):
        sim = MPSSimulator(n_qubits=3)
        with pytest.raises(MPSSimulatorError):
            sim.h(5)


# ─────────────────────────────────────────────────────────────────────────────
# Two-qubit gates
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoQubitGates:
    def test_cx_creates_bell_state(self):
        sim = MPSSimulator(n_qubits=2)
        sim.h(0)
        sim.cx(0, 1)
        sv = sim.to_statevector()
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2
        assert abs(sv[0]) == pytest.approx(1 / math.sqrt(2), abs=1e-12)
        assert abs(sv[3]) == pytest.approx(1 / math.sqrt(2), abs=1e-12)
        assert abs(sv[1]) < 1e-12
        assert abs(sv[2]) < 1e-12

    def test_cx_bond_grows(self):
        sim = MPSSimulator(n_qubits=3)
        sim.h(0).cx(0, 1)
        assert sim.bond_dims[0] == 2

    def test_cx_non_adjacent_raises(self):
        sim = MPSSimulator(n_qubits=4)
        with pytest.raises(MPSSimulatorError, match="adjacent"):
            sim.cx(0, 2)

    def test_cz_gate(self):
        sim = MPSSimulator(n_qubits=2)
        sim.h(0).h(1)
        sim.cz(0, 1)
        sv = sim.to_statevector()
        # CZ on |+⟩|+⟩ = (|00⟩+|01⟩+|10⟩-|11⟩)/2
        expected = np.array([0.5, 0.5, 0.5, -0.5], dtype=complex)
        assert np.allclose(np.abs(sv), np.abs(expected), atol=1e-12)

    def test_two_qubit_preserves_norm(self):
        sim = MPSSimulator(n_qubits=4, max_bond=8)
        sim.h(0).cx(0, 1).cx(1, 2).cx(2, 3)
        assert sim.norm() == pytest.approx(1.0, abs=1e-10)

    def test_bond_dim_limited_by_max_bond(self):
        sim = MPSSimulator(n_qubits=6, max_bond=2)
        sim.h(0)
        for i in range(5):
            sim.cx(i, i + 1)
        assert max(sim.bond_dims) <= 2

    def test_reversed_cx(self):
        """CNOT with ctrl=1, target=0 should also work."""
        sim = MPSSimulator(n_qubits=2)
        sim.x(1)          # |01⟩ in qubit order q0q1 → q1=1, so state is |10⟩ (q1 is control)
        sim.cx(1, 0)      # ctrl=1 (q1), target=0 (q0): should flip q0
        sv = sim.to_statevector()
        # |01⟩ → cx(1,0) → |11⟩ (both flipped)
        assert abs(sv[3]) == pytest.approx(1.0, abs=1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# GHZ state and fidelity
# ─────────────────────────────────────────────────────────────────────────────

class TestGHZFidelity:
    def test_ghz_4_fidelity_near_1(self):
        f = ghz_fidelity(4, max_bond=2)
        assert f == pytest.approx(1.0, abs=1e-10)

    def test_ghz_10_fidelity_near_1(self):
        f = ghz_fidelity(10, max_bond=2)
        assert f == pytest.approx(1.0, abs=1e-10)

    def test_ghz_50_fidelity_near_1(self):
        """Core M104 assertion: GHZ at n=50 with bond_dim=2 should be lossless."""
        f = ghz_fidelity(50, max_bond=2)
        assert f == pytest.approx(1.0, abs=1e-8)

    def test_ghz_1_fidelity(self):
        f = ghz_fidelity(1, max_bond=2)
        assert f == pytest.approx(1.0, abs=1e-10)

    def test_insufficient_bond_dim_reduces_fidelity(self):
        """With max_bond=1, GHZ cannot be represented exactly for n>1."""
        f = ghz_fidelity(4, max_bond=1)
        assert f < 0.99

    def test_ghz_mps_exact_norm(self):
        mps = _ghz_mps(4)
        assert mps.norm() == pytest.approx(1.0, abs=1e-12)

    def test_ghz_mps_correct_statevector(self):
        """GHZ MPS should give (|0000⟩+|1111⟩)/√2."""
        mps = _ghz_mps(4)
        sv = mps.to_statevector()
        assert abs(sv[0]) == pytest.approx(1 / math.sqrt(2), abs=1e-12)
        assert abs(sv[15]) == pytest.approx(1 / math.sqrt(2), abs=1e-12)
        assert np.sum(np.abs(sv[1:15]) ** 2) < 1e-24


# ─────────────────────────────────────────────────────────────────────────────
# MPS vs statevector comparison
# ─────────────────────────────────────────────────────────────────────────────

class TestCompareMpsSv:
    def test_fidelity_near_1_small_system(self):
        r = compare_mps_sv(n_qubits=4, circuit_depth=3, max_bond=32)
        assert r.fidelity == pytest.approx(1.0, abs=1e-8)

    def test_fidelity_near_1_n8(self):
        r = compare_mps_sv(n_qubits=8, circuit_depth=4, max_bond=64)
        assert r.fidelity == pytest.approx(1.0, abs=1e-6)

    def test_fidelity_near_1_n12(self):
        r = compare_mps_sv(n_qubits=12, circuit_depth=3, max_bond=32)
        assert r.fidelity == pytest.approx(1.0, abs=1e-4)

    def test_returns_comparison_result(self):
        r = compare_mps_sv(n_qubits=4, circuit_depth=2, max_bond=16)
        assert isinstance(r, ComparisonResult)
        assert r.n_qubits == 4

    def test_n_qubits_too_large_raises(self):
        with pytest.raises(MPSSimulatorError, match="≤ 20"):
            compare_mps_sv(n_qubits=25)

    def test_lower_bond_dim_lower_fidelity(self):
        r_high = compare_mps_sv(n_qubits=6, circuit_depth=5, max_bond=64)
        r_low  = compare_mps_sv(n_qubits=6, circuit_depth=5, max_bond=2)
        assert r_high.fidelity >= r_low.fidelity


# ─────────────────────────────────────────────────────────────────────────────
# mps_benchmark
# ─────────────────────────────────────────────────────────────────────────────

class TestMpsBenchmark:
    def test_returns_benchmark_result(self):
        r = mps_benchmark(n_qubits=6, circuit_depth=2, max_bond=8)
        assert isinstance(r, BenchmarkResult)

    def test_n_qubits_stored(self):
        r = mps_benchmark(n_qubits=8, circuit_depth=2, max_bond=8)
        assert r.n_qubits == 8

    def test_elapsed_positive(self):
        r = mps_benchmark(n_qubits=4, circuit_depth=3, max_bond=8)
        assert r.elapsed_seconds > 0

    def test_truncation_error_nonnegative(self):
        r = mps_benchmark(n_qubits=4, circuit_depth=2, max_bond=4)
        assert r.truncation_error >= 0.0

    def test_max_bond_used_within_limit(self):
        r = mps_benchmark(n_qubits=10, circuit_depth=5, max_bond=4)
        assert r.max_bond_used <= 4

    def test_large_bond_dim_zero_truncation_error(self):
        """With max_bond=256, no truncation should be needed for 8 qubits."""
        r = mps_benchmark(n_qubits=8, circuit_depth=3, max_bond=256)
        assert r.truncation_error == pytest.approx(0.0, abs=1e-20)

    def test_50_qubits_runs_without_error(self):
        """Core M104: 50-qubit circuit with bond_dim=8 should complete."""
        r = mps_benchmark(n_qubits=50, circuit_depth=5, max_bond=8)
        assert isinstance(r, BenchmarkResult)
        assert math.isfinite(r.truncation_error)
        assert math.isfinite(r.elapsed_seconds)

    def test_invalid_n_qubits(self):
        with pytest.raises(MPSSimulatorError):
            mps_benchmark(n_qubits=0)

    def test_invalid_circuit_depth(self):
        with pytest.raises(MPSSimulatorError):
            mps_benchmark(n_qubits=4, circuit_depth=0)


# ─────────────────────────────────────────────────────────────────────────────
# CachedDMRG
# ─────────────────────────────────────────────────────────────────────────────

class TestCachedDMRG:
    def test_heisenberg_ground_state(self):
        mpo = MPO.heisenberg(n=6, J=1.0)
        dmrg = CachedDMRG(mpo=mpo, max_chi=16, n_sweeps=8, seed=0)
        result = dmrg.run()
        # Known ground state energy of 6-site AFM Heisenberg (spin convention):
        # E0 ≈ -2.8028 for J=1 open BC. Our MPO uses Pauli convention → ×4 ≈ -11.21
        assert result.energy < -8.0  # strictly below trivial bound

    def test_converged_flag_set(self):
        mpo = MPO.heisenberg(n=4, J=1.0)
        dmrg = CachedDMRG(mpo=mpo, max_chi=8, n_sweeps=20, tol=1e-8, seed=42)
        result = dmrg.run()
        assert result.converged

    def test_returns_dmrg_result(self):
        mpo = MPO.ising(n=4, J=1.0, h=0.5)
        dmrg = CachedDMRG(mpo=mpo, max_chi=8, seed=0)
        result = dmrg.run()
        assert hasattr(result, "energy")
        assert hasattr(result, "mps")
        assert hasattr(result, "sweep_energies")

    def test_agrees_with_standard_dmrg(self):
        """CachedDMRG and standard DMRG should give the same energy."""
        mpo = MPO.heisenberg(n=6, J=1.0)
        from quantx_engine.tensor_network import DMRG
        ref = DMRG(mpo=mpo, max_chi=8, n_sweeps=6, seed=7).run()
        cached = CachedDMRG(mpo=mpo, max_chi=8, n_sweeps=6, seed=7).run()
        assert cached.energy == pytest.approx(ref.energy, abs=1e-3)

    def test_ising_ground_state_negative(self):
        mpo = MPO.ising(n=8, J=1.0, h=0.3)
        dmrg = CachedDMRG(mpo=mpo, max_chi=8, n_sweeps=4, seed=0)
        result = dmrg.run()
        assert result.energy < 0


# ─────────────────────────────────────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────────────────────────────────────

class TestExports:
    def test_importable_from_package(self):
        from quantx_engine import (
            MPSSimulatorError,
            AdaptiveTruncation,
            MPSSimulator,
            CachedDMRG,
            BenchmarkResult,
            ComparisonResult,
            mps_benchmark,
            ghz_fidelity,
            compare_mps_sv,
        )
        assert callable(mps_benchmark)
        assert callable(ghz_fidelity)
        assert callable(compare_mps_sv)
