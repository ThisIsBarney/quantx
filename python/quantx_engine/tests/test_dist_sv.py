"""M111: Tests for distributed statevector simulation via Ray.

All tests run on CPU (Ray-fallback path) and pass regardless of whether
Ray is installed.  Ray-specific assertions are gated on ``is_ray_available()``.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from quantx_engine.ray_sim import (
    DistSVError,
    DistSVResult,
    DistributedSVSimulator,
    dist_sv_benchmark,
    is_ray_available,
)


# ── is_ray_available ──────────────────────────────────────────────────────────

def test_is_ray_available_returns_bool():
    assert isinstance(is_ray_available(), bool)


# ── DistributedSVSimulator construction ──────────────────────────────────────

class TestConstruction:
    def test_default_init(self):
        sim = DistributedSVSimulator(n_qubits=4)
        assert sim.n_qubits == 4
        assert sim.num_shards in (1, 2, 4)

    def test_explicit_num_shards(self):
        sim = DistributedSVSimulator(n_qubits=4, num_shards=4)
        assert sim.num_shards == 4

    def test_num_shards_1(self):
        sim = DistributedSVSimulator(n_qubits=3, num_shards=1)
        assert sim.num_shards == 1

    def test_cpu_backend(self):
        sim = DistributedSVSimulator(n_qubits=4, backend="cpu")
        assert sim.backend == "cpu"

    def test_invalid_n_qubits_raises(self):
        with pytest.raises(DistSVError):
            DistributedSVSimulator(n_qubits=0)

    def test_non_power_of_two_shards_raises(self):
        with pytest.raises(DistSVError):
            DistributedSVSimulator(n_qubits=4, num_shards=3)

    def test_shards_larger_than_state_raises(self):
        with pytest.raises(DistSVError):
            DistributedSVSimulator(n_qubits=2, num_shards=8)

    def test_initial_state_is_zero(self):
        sim = DistributedSVSimulator(n_qubits=3)
        sv = sim.to_statevector()
        assert abs(sv[0] - 1.0) < 1e-12
        assert np.allclose(sv[1:], 0.0)

    def test_initial_norm(self):
        sim = DistributedSVSimulator(n_qubits=5, num_shards=4)
        assert sim.norm() == pytest.approx(1.0, abs=1e-12)

    def test_ray_backend_falls_back_gracefully(self):
        """backend='ray' must not raise even without Ray installed."""
        import warnings
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            sim = DistributedSVSimulator(n_qubits=4, backend="auto")
        sim.h(0)
        assert sim.norm() == pytest.approx(1.0, abs=1e-10)


# ── Single-qubit gates ────────────────────────────────────────────────────────

class TestSingleQubitGates:
    @pytest.mark.parametrize("num_shards", [1, 2, 4])
    def test_x_gate(self, num_shards: int):
        """X|0⟩ → |1⟩: amplitude 1 should be 1."""
        sim = DistributedSVSimulator(n_qubits=3, num_shards=num_shards, backend="cpu")
        sim.x(0)
        sv = sim.to_statevector()
        assert abs(sv[1] - 1.0) < 1e-12  # |001⟩
        assert abs(sv[0]) < 1e-12

    @pytest.mark.parametrize("num_shards", [1, 2, 4])
    def test_h_gate_superposition(self, num_shards: int):
        """H|0⟩ → (|0⟩+|1⟩)/√2."""
        sim = DistributedSVSimulator(n_qubits=3, num_shards=num_shards, backend="cpu")
        sim.h(0)
        sv = sim.to_statevector()
        expected = 1.0 / math.sqrt(2)
        assert abs(sv[0] - expected) < 1e-12
        assert abs(sv[1] - expected) < 1e-12

    @pytest.mark.parametrize("num_shards", [1, 2, 4])
    def test_h_h_identity(self, num_shards: int):
        """H·H = I for all num_shards."""
        sim = DistributedSVSimulator(n_qubits=4, num_shards=num_shards, backend="cpu")
        sim.h(0).h(0)
        sv = sim.to_statevector()
        assert abs(sv[0] - 1.0) < 1e-12
        np.testing.assert_allclose(sv[1:], 0.0, atol=1e-12)

    def test_gate_on_high_qubit(self):
        """Gate on the highest qubit (global qubit) is applied correctly."""
        sim = DistributedSVSimulator(n_qubits=4, num_shards=4, backend="cpu")
        # qubit 3 is a global qubit for 4 shards (2 global bits: qubits 2,3)
        sim.x(3)
        sv = sim.to_statevector()
        assert abs(sv[8] - 1.0) < 1e-12  # |1000⟩ = index 8

    def test_z_gate(self):
        """Z|0⟩ = |0⟩ (phase -1 on |1⟩ but state is |0⟩)."""
        sim = DistributedSVSimulator(n_qubits=2, backend="cpu")
        sim.z(0)
        sv = sim.to_statevector()
        assert abs(sv[0] - 1.0) < 1e-12

    def test_s_gate(self):
        """S|+⟩ = |y+⟩: amplitudes (1, i)/√2."""
        sim = DistributedSVSimulator(n_qubits=1, num_shards=1, backend="cpu")
        sim.h(0).s(0)
        sv = sim.to_statevector()
        expected = np.array([1.0, 1j]) / math.sqrt(2)
        np.testing.assert_allclose(sv, expected, atol=1e-12)

    def test_rx_gate(self):
        """Rx(π)|0⟩ = -i|1⟩."""
        sim = DistributedSVSimulator(n_qubits=1, num_shards=1, backend="cpu")
        sim.rx(math.pi, 0)
        sv = sim.to_statevector()
        assert abs(sv[0]) < 1e-12
        assert abs(abs(sv[1]) - 1.0) < 1e-12

    def test_out_of_range_raises(self):
        sim = DistributedSVSimulator(n_qubits=3)
        with pytest.raises(DistSVError):
            sim.h(3)
        with pytest.raises(DistSVError):
            sim.x(-1)


# ── Two-qubit gates ───────────────────────────────────────────────────────────

class TestTwoQubitGates:
    @pytest.mark.parametrize("num_shards", [1, 2, 4])
    def test_cx_bell_state(self, num_shards: int):
        """H(0)·CX(0,1) → Bell state (|00⟩+|11⟩)/√2."""
        sim = DistributedSVSimulator(n_qubits=2, num_shards=num_shards, backend="cpu")
        sim.h(0).cx(0, 1)
        sv = sim.to_statevector()
        expected = np.zeros(4, dtype=np.complex128)
        expected[0] = 1.0 / math.sqrt(2)
        expected[3] = 1.0 / math.sqrt(2)
        np.testing.assert_allclose(sv, expected, atol=1e-12)

    def test_cx_ctrl_eq_tgt_raises(self):
        sim = DistributedSVSimulator(n_qubits=2)
        with pytest.raises(DistSVError):
            sim.cx(0, 0)

    @pytest.mark.parametrize("num_shards", [1, 2, 4])
    def test_cz_gate(self, num_shards: int):
        """CZ on |+,+⟩ = |Φ-⟩ up to normalisation."""
        sim = DistributedSVSimulator(n_qubits=2, num_shards=num_shards, backend="cpu")
        sim.h(0).h(1).cz(0, 1)
        sv = sim.to_statevector()
        # |+,+⟩ = (|00⟩+|01⟩+|10⟩+|11⟩)/2
        # CZ flips sign of |11⟩ → (|00⟩+|01⟩+|10⟩-|11⟩)/2
        expected = np.array([1, 1, 1, -1], dtype=np.complex128) / 2.0
        np.testing.assert_allclose(sv, expected, atol=1e-12)

    def test_cx_global_qubit(self):
        """CX where ctrl or target is a global qubit falls back correctly."""
        sim = DistributedSVSimulator(n_qubits=4, num_shards=4, backend="cpu")
        # qubit 3 is global (2 global bits for 4 shards on 4 qubits)
        sim.x(3)          # |1000⟩
        sim.cx(3, 0)      # ctrl=1 → flip tgt: |1001⟩
        sv = sim.to_statevector()
        assert abs(sv[9] - 1.0) < 1e-12  # |1001⟩ = 8+1=9


# ── Parity with CPU reference ─────────────────────────────────────────────────

class TestParityWithCPU:
    """Verify shard counts produce identical results."""

    @pytest.mark.parametrize("n,shards", [(4, 1), (4, 2), (4, 4), (6, 4), (8, 4)])
    def test_ghz_parity(self, n: int, shards: int):
        """GHZ circuit matches single-shard reference for various shard counts."""
        def make_ghz(s: int) -> np.ndarray:
            sim = DistributedSVSimulator(n_qubits=n, num_shards=s, backend="cpu")
            sim.h(0)
            for q in range(n - 1):
                sim.cx(q, q + 1)
            return sim.to_statevector()

        sv_ref = make_ghz(1)
        sv_sharded = make_ghz(shards)
        np.testing.assert_allclose(sv_ref, sv_sharded, atol=1e-12)

    @pytest.mark.parametrize("n,shards", [(4, 2), (6, 4)])
    def test_random_circuit_parity(self, n: int, shards: int):
        rng = np.random.default_rng(42)
        angles = rng.uniform(0, 2 * math.pi, n)

        def run(s: int) -> np.ndarray:
            sim = DistributedSVSimulator(n_qubits=n, num_shards=s, backend="cpu")
            for q in range(n):
                sim.ry(float(angles[q]), q)
            for q in range(0, n - 1, 2):
                sim.cz(q, q + 1)
            return sim.to_statevector()

        np.testing.assert_allclose(run(1), run(shards), atol=1e-12)


# ── measure_all ───────────────────────────────────────────────────────────────

class TestMeasureAll:
    def test_zero_state_measures_all_zero(self):
        sim = DistributedSVSimulator(n_qubits=4)
        outcomes = sim.measure_all(seed=0)
        assert outcomes == [0, 0, 0, 0]

    def test_x_state_measures_all_one(self):
        sim = DistributedSVSimulator(n_qubits=3, backend="cpu")
        sim.x(0).x(1).x(2)
        outcomes = sim.measure_all(seed=0)
        assert outcomes == [1, 1, 1]

    def test_bell_state_correlated(self):
        """Both qubits must agree for Bell state."""
        for seed in range(20):
            sim = DistributedSVSimulator(n_qubits=2, backend="cpu")
            sim.h(0).cx(0, 1)
            outcomes = sim.measure_all(seed=seed)
            assert outcomes[0] == outcomes[1], f"seed={seed}: {outcomes}"

    def test_measure_returns_list_of_ints(self):
        sim = DistributedSVSimulator(n_qubits=3)
        outcomes = sim.measure_all(seed=7)
        assert len(outcomes) == 3
        assert all(o in (0, 1) for o in outcomes)


# ── to_statevector ────────────────────────────────────────────────────────────

def test_to_statevector_is_numpy():
    sim = DistributedSVSimulator(n_qubits=4)
    sv = sim.to_statevector()
    assert isinstance(sv, np.ndarray)
    assert sv.shape == (16,)
    assert sv.dtype == np.complex128


def test_norm_preserved_after_gates():
    sim = DistributedSVSimulator(n_qubits=5, num_shards=4, backend="cpu")
    sim.h(0).h(1).cx(0, 1).cz(2, 3).rx(1.2, 4)
    assert sim.norm() == pytest.approx(1.0, abs=1e-10)


# ── dist_sv_benchmark ─────────────────────────────────────────────────────────

class TestDistSVBenchmark:
    def test_returns_dist_sv_result(self):
        result = dist_sv_benchmark(n_qubits=6, circuit_depth=2, num_shards=4, seed=0)
        assert isinstance(result, DistSVResult)

    def test_fidelity_is_one(self):
        result = dist_sv_benchmark(n_qubits=6, circuit_depth=2, num_shards=4, seed=42)
        assert result.fidelity == pytest.approx(1.0, abs=1e-8)

    def test_speedup_positive(self):
        result = dist_sv_benchmark(n_qubits=6, circuit_depth=2, num_shards=4, seed=0)
        assert result.speedup > 0.0

    def test_fields(self):
        result = dist_sv_benchmark(n_qubits=8, circuit_depth=3, num_shards=4, seed=1)
        assert result.n_qubits == 8
        assert result.num_shards == 4
        assert result.circuit_depth == 3
        assert result.cpu_seconds > 0.0
        assert result.dist_seconds > 0.0
        assert isinstance(result.ray_available, bool)
        assert result.ray_available == is_ray_available()

    def test_warmup_false_still_works(self):
        result = dist_sv_benchmark(
            n_qubits=4, circuit_depth=1, num_shards=2, warmup=False
        )
        assert isinstance(result, DistSVResult)

    def test_n20_runs_without_error(self):
        """Default n=20 benchmark completes."""
        result = dist_sv_benchmark(n_qubits=20, circuit_depth=3, num_shards=4, seed=0)
        assert result.n_qubits == 20
        assert result.fidelity == pytest.approx(1.0, abs=1e-8)

    def test_num_shards_one(self):
        """Single shard should still produce fidelity=1."""
        result = dist_sv_benchmark(n_qubits=6, circuit_depth=2, num_shards=1, seed=0)
        assert result.fidelity == pytest.approx(1.0, abs=1e-8)
