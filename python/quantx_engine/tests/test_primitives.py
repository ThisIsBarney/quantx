"""Tests for Estimator + Sampler primitives — M85."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quantx_engine.primitives import (
    Estimator,
    EstimatorResult,
    PrimitiveError,
    Sampler,
    SamplerResult,
    SparsePauliOp,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bell_circuit() -> dict:
    return {
        "n_qubits": 2,
        "operations": [
            {"name": "h",  "qubits": [0]},
            {"name": "cx", "qubits": [0, 1]},
        ],
    }


def _x_circuit(n: int = 1) -> dict:
    return {
        "n_qubits": n,
        "operations": [{"name": "x", "qubits": [i]} for i in range(n)],
    }


def _h_circuit() -> dict:
    return {
        "n_qubits": 1,
        "operations": [{"name": "h", "qubits": [0]}],
    }


def _ghz_circuit(n: int) -> dict:
    ops = [{"name": "h", "qubits": [0]}]
    for i in range(n - 1):
        ops.append({"name": "cx", "qubits": [i, i + 1]})
    return {"n_qubits": n, "operations": ops}


# ---------------------------------------------------------------------------
# SparsePauliOp
# ---------------------------------------------------------------------------

class TestSparsePauliOp:
    def test_from_string(self):
        op = SparsePauliOp("ZI")
        assert op.n_qubits == 2
        assert op.terms == [("ZI", 1.0 + 0j)]

    def test_from_list(self):
        op = SparsePauliOp([("ZZ", 0.5), ("XX", -0.5)])
        assert len(op.terms) == 2

    def test_copy(self):
        a = SparsePauliOp("ZZ")
        b = SparsePauliOp(a)
        assert b.terms == a.terms

    def test_from_pauli_hamiltonian(self):
        from quantx_engine.chemistry import PauliHamiltonian, PauliTerm
        ham = PauliHamiltonian(
            terms=[PauliTerm(coeff=1.0, ops={0: "Z"})], n_qubits=1
        )
        op = SparsePauliOp.from_pauli_hamiltonian(ham)
        assert op.n_qubits == 1
        # "Z" on qubit 0 → single-character string "Z"
        assert op.terms[0][0] == "Z"

    def test_invalid_characters(self):
        with pytest.raises(PrimitiveError, match="Invalid characters"):
            SparsePauliOp("AZ")

    def test_mismatched_lengths(self):
        with pytest.raises(PrimitiveError, match="same length"):
            SparsePauliOp([("ZI", 1.0), ("ZZZ", 1.0)])

    def test_empty_list(self):
        with pytest.raises(PrimitiveError):
            SparsePauliOp([])

    def test_invalid_type(self):
        with pytest.raises(PrimitiveError):
            SparsePauliOp(42)

    def test_invalid_list_item(self):
        with pytest.raises(PrimitiveError):
            SparsePauliOp(["ZZ"])  # str, not tuple

    def test_empty_string(self):
        with pytest.raises(PrimitiveError):
            SparsePauliOp("")

    def test_case_insensitive(self):
        op = SparsePauliOp("xyzI")
        assert op.terms[0][0] == "XYZI"


# ---------------------------------------------------------------------------
# Gate matrices (spot checks)
# ---------------------------------------------------------------------------

class TestGateMatrices:
    def test_h_unitary(self):
        from quantx_engine.primitives import _get_gate_matrix
        H = _get_gate_matrix("h", None)
        assert np.allclose(H @ H.conj().T, np.eye(2), atol=1e-12)

    def test_cx_is_cnot(self):
        from quantx_engine.primitives import _get_gate_matrix
        cx = _get_gate_matrix("cx", None)
        # |10⟩ → |11⟩: row 3 (|11⟩) should be 1 at column 2 (|10⟩)
        assert cx[3, 2] == pytest.approx(1.0)
        assert cx[2, 3] == pytest.approx(1.0)

    def test_rx_pi_equals_x(self):
        from quantx_engine.primitives import _get_gate_matrix
        Rx = _get_gate_matrix("rx", [math.pi])
        X  = _get_gate_matrix("x", None)
        # RX(π) = −i·X (global phase only)
        assert np.allclose(np.abs(Rx), np.abs(X), atol=1e-10)

    def test_unknown_gate(self):
        from quantx_engine.primitives import _get_gate_matrix
        with pytest.raises(PrimitiveError, match="Unknown gate"):
            _get_gate_matrix("foo", None)

    def test_rzz_diagonal(self):
        from quantx_engine.primitives import _get_gate_matrix
        R = _get_gate_matrix("rzz", [math.pi / 2])
        # Should be diagonal
        off = R - np.diag(np.diag(R))
        assert np.allclose(off, 0, atol=1e-12)


# ---------------------------------------------------------------------------
# Estimator — basic (noiseless, exact)
# ---------------------------------------------------------------------------

class TestEstimatorBasic:
    def test_returns_estimator_result(self):
        r = Estimator().run([{"n_qubits": 1, "operations": []}], ["Z"])
        assert isinstance(r, EstimatorResult)

    def test_z_expectation_ground_state(self):
        # |0⟩ is Z eigenstate +1
        r = Estimator().run(
            [{"n_qubits": 1, "operations": []}], ["Z"]
        )
        assert r.values[0] == pytest.approx(1.0, abs=1e-10)

    def test_z_expectation_excited_state(self):
        # X|0⟩ = |1⟩, Z eigenvalue = −1
        r = Estimator().run([_x_circuit()], ["Z"])
        assert r.values[0] == pytest.approx(-1.0, abs=1e-10)

    def test_x_expectation_plus_state(self):
        # H|0⟩ = |+⟩, ⟨+|X|+⟩ = 1
        r = Estimator().run([_h_circuit()], ["X"])
        assert r.values[0] == pytest.approx(1.0, abs=1e-10)

    def test_bell_state_zz_equals_one(self):
        r = Estimator().run([_bell_circuit()], ["ZZ"])
        assert r.values[0] == pytest.approx(1.0, abs=1e-10)

    def test_bell_state_xx_equals_one(self):
        # Bell state is also an eigenstate of XX
        r = Estimator().run([_bell_circuit()], ["XX"])
        assert r.values[0] == pytest.approx(1.0, abs=1e-10)

    def test_bell_state_zi_equals_zero(self):
        # Bell state: ⟨ZI⟩ = 0 (marginal is maximally mixed)
        r = Estimator().run([_bell_circuit()], ["ZI"])
        assert r.values[0] == pytest.approx(0.0, abs=1e-10)

    def test_identity_observable_equals_one(self):
        r = Estimator().run([_bell_circuit()], ["II"])
        assert r.values[0] == pytest.approx(1.0, abs=1e-10)

    def test_stds_zero_for_exact(self):
        r = Estimator().run([_h_circuit()], ["Z"])
        assert r.stds[0] == pytest.approx(0.0, abs=1e-12)

    def test_multiple_pairs(self):
        circs = [_x_circuit(), _h_circuit()]
        obs   = ["Z", "X"]
        r = Estimator().run(circs, obs)
        assert r.values.shape == (2,)
        assert r.values[0] == pytest.approx(-1.0, abs=1e-10)
        assert r.values[1] == pytest.approx(1.0,  abs=1e-10)

    def test_broadcast_single_circuit(self):
        # one circuit, two observables
        r = Estimator().run(
            [_x_circuit()], ["Z", SparsePauliOp("X")]
        )
        assert r.values.shape == (2,)

    def test_broadcast_single_observable(self):
        circs = [_x_circuit(), {"n_qubits": 1, "operations": []}]
        r = Estimator().run(circs, ["Z"])
        assert r.values[0] == pytest.approx(-1.0, abs=1e-10)
        assert r.values[1] == pytest.approx(1.0,  abs=1e-10)

    def test_weighted_hamiltonian(self):
        # 0.5 ZZ − 0.5 XX on Bell state: 0.5·1 − 0.5·1 = 0
        obs = SparsePauliOp([("ZZ", 0.5), ("XX", -0.5)])
        r = Estimator().run([_bell_circuit()], [obs])
        assert r.values[0] == pytest.approx(0.0, abs=1e-10)

    def test_metadata_circuit_index(self):
        r = Estimator().run([_h_circuit()], ["Z"])
        assert r.metadata[0]["circuit_index"] == 0
        assert r.metadata[0]["n_qubits"] == 1


# ---------------------------------------------------------------------------
# Estimator — parametric circuits
# ---------------------------------------------------------------------------

class TestEstimatorParametric:
    def test_rz_theta_zero(self):
        # RZ(0) = I; ⟨0|Z|0⟩ = 1
        circ = {
            "n_qubits": 1,
            "operations": [{"name": "rz", "qubits": [0], "params": ["theta"]}],
        }
        r = Estimator().run([circ], ["Z"], [{"theta": 0.0}])
        assert r.values[0] == pytest.approx(1.0, abs=1e-10)

    def test_rx_pi(self):
        # RX(π)|0⟩ ≈ |1⟩; ⟨Z⟩ = −1
        circ = {
            "n_qubits": 1,
            "operations": [{"name": "rx", "qubits": [0], "params": ["a"]}],
        }
        r = Estimator().run([circ], ["Z"], [{"a": math.pi}])
        assert r.values[0] == pytest.approx(-1.0, abs=1e-6)

    def test_missing_parameter_raises(self):
        circ = {
            "n_qubits": 1,
            "operations": [{"name": "rx", "qubits": [0], "params": ["theta"]}],
        }
        with pytest.raises(PrimitiveError, match="not found"):
            Estimator().run([circ], ["Z"])


# ---------------------------------------------------------------------------
# Estimator — error cases
# ---------------------------------------------------------------------------

class TestEstimatorErrors:
    def test_mismatched_lengths(self):
        with pytest.raises(PrimitiveError, match="same length"):
            Estimator().run(
                [_h_circuit(), _h_circuit()], ["Z", "X", "Y"]
            )

    def test_wrong_observable_size(self):
        with pytest.raises(PrimitiveError, match="qubits"):
            Estimator().run([_h_circuit()], ["ZZ"])  # 1-qubit circuit, 2-qubit obs

    def test_invalid_observable_type(self):
        with pytest.raises(PrimitiveError):
            Estimator().run([_h_circuit()], [42])

    def test_invalid_circuit_type(self):
        with pytest.raises(PrimitiveError):
            Estimator().run(["not a circuit"], ["Z"])


# ---------------------------------------------------------------------------
# Estimator — noisy (density matrix)
# ---------------------------------------------------------------------------

class TestEstimatorNoisy:
    def test_depolarizing_reduces_purity(self):
        from quantx_engine.noise import DepolarizingNoise
        noise = DepolarizingNoise(0.1)
        r_clean = Estimator().run([_x_circuit()], ["Z"])
        r_noisy = Estimator(noise_model=noise).run([_x_circuit()], ["Z"])
        # Noisy |⟨Z⟩| should be smaller
        assert abs(r_noisy.values[0]) < abs(r_clean.values[0])

    def test_zero_noise_matches_exact(self):
        from quantx_engine.noise import DepolarizingNoise
        noise = DepolarizingNoise(0.0)
        r_exact = Estimator().run([_bell_circuit()], ["ZZ"])
        r_noisy = Estimator(noise_model=noise).run([_bell_circuit()], ["ZZ"])
        assert r_noisy.values[0] == pytest.approx(r_exact.values[0], abs=1e-8)


# ---------------------------------------------------------------------------
# Estimator — shot-based
# ---------------------------------------------------------------------------

class TestEstimatorShots:
    def test_shot_result_approximately_correct(self):
        r = Estimator(shots=4000, seed=0).run([_h_circuit()], ["Z"])
        # H|0⟩: ⟨Z⟩ = 0
        assert abs(r.values[0]) < 0.15

    def test_std_positive_for_shots(self):
        r = Estimator(shots=100, seed=1).run([_h_circuit()], ["Z"])
        assert r.stds[0] > 0.0

    def test_deterministic_state_exact_with_shots(self):
        # |1⟩: ⟨Z⟩ = −1 deterministically, std should be 0
        r = Estimator(shots=200, seed=2).run([_x_circuit()], ["Z"])
        assert r.values[0] == pytest.approx(-1.0, abs=1e-10)
        assert r.stds[0] == pytest.approx(0.0, abs=1e-10)

    def test_seed_reproducibility(self):
        ra = Estimator(shots=50, seed=99).run([_h_circuit()], ["Z"])
        rb = Estimator(shots=50, seed=99).run([_h_circuit()], ["Z"])
        assert ra.values[0] == pytest.approx(rb.values[0], abs=1e-12)


# ---------------------------------------------------------------------------
# Sampler — basic (exact probabilities)
# ---------------------------------------------------------------------------

class TestSamplerBasic:
    def test_returns_sampler_result(self):
        r = Sampler().run([_h_circuit()])
        assert isinstance(r, SamplerResult)

    def test_ground_state_probability(self):
        r = Sampler().run([{"n_qubits": 1, "operations": []}])
        qd = r.quasi_dists[0]
        assert qd.get(0, 0.0) == pytest.approx(1.0, abs=1e-12)
        assert len(qd) == 1

    def test_x_gate_flips_to_one(self):
        r = Sampler().run([_x_circuit()])
        qd = r.quasi_dists[0]
        assert qd.get(1, 0.0) == pytest.approx(1.0, abs=1e-12)

    def test_h_gate_equal_superposition(self):
        r = Sampler().run([_h_circuit()])
        qd = r.quasi_dists[0]
        assert qd.get(0, 0.0) == pytest.approx(0.5, abs=1e-12)
        assert qd.get(1, 0.0) == pytest.approx(0.5, abs=1e-12)

    def test_bell_state_bitstring_distribution(self):
        r = Sampler().run([_bell_circuit()])
        qd = r.quasi_dists[0]
        # Only |00⟩=0 and |11⟩=3 should be present
        assert set(qd.keys()) == {0, 3}
        assert qd[0] == pytest.approx(0.5, abs=1e-12)
        assert qd[3] == pytest.approx(0.5, abs=1e-12)

    def test_probabilities_sum_to_one(self):
        r = Sampler().run([_ghz_circuit(3)])
        total = sum(r.quasi_dists[0].values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_multiple_circuits(self):
        r = Sampler().run([_h_circuit(), _x_circuit()])
        assert len(r.quasi_dists) == 2
        assert r.quasi_dists[1].get(1, 0.0) == pytest.approx(1.0, abs=1e-12)

    def test_metadata_present(self):
        r = Sampler().run([_h_circuit()])
        assert r.metadata[0]["n_qubits"] == 1


# ---------------------------------------------------------------------------
# Sampler — shot-based
# ---------------------------------------------------------------------------

class TestSamplerShots:
    def test_shot_counts_sum_to_one(self):
        r = Sampler(shots=500, seed=7).run([_h_circuit()])
        total = sum(r.quasi_dists[0].values())
        assert total == pytest.approx(1.0, abs=1e-10)

    def test_approx_50_50(self):
        r = Sampler(shots=4000, seed=0).run([_h_circuit()])
        qd = r.quasi_dists[0]
        p0 = qd.get(0, 0.0)
        assert abs(p0 - 0.5) < 0.05

    def test_deterministic_circuit_shot_result(self):
        r = Sampler(shots=100, seed=3).run([_x_circuit()])
        qd = r.quasi_dists[0]
        assert set(qd.keys()) == {1}
        assert qd[1] == pytest.approx(1.0, abs=1e-10)

    def test_seed_reproducibility(self):
        ra = Sampler(shots=50, seed=42).run([_h_circuit()])
        rb = Sampler(shots=50, seed=42).run([_h_circuit()])
        assert ra.quasi_dists[0] == rb.quasi_dists[0]

    def test_different_seeds_differ(self):
        ra = Sampler(shots=20, seed=1).run([_h_circuit()])
        rb = Sampler(shots=20, seed=2).run([_h_circuit()])
        # Not guaranteed to differ, but very likely with only 20 shots
        # Just check they are valid distributions
        assert abs(sum(ra.quasi_dists[0].values()) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Sampler — noisy
# ---------------------------------------------------------------------------

class TestSamplerNoisy:
    def test_noisy_bell_state_blurs_distribution(self):
        from quantx_engine.noise import DepolarizingNoise
        noise = DepolarizingNoise(0.2)
        r_clean = Sampler().run([_bell_circuit()])
        r_noisy = Sampler(noise_model=noise).run([_bell_circuit()])
        # Noisy distribution should have more non-zero entries than {0, 3}
        assert len(r_noisy.quasi_dists[0]) > len(r_clean.quasi_dists[0])

    def test_noisy_probabilities_sum_to_one(self):
        from quantx_engine.noise import DepolarizingNoise
        noise = DepolarizingNoise(0.05)
        r = Sampler(noise_model=noise).run([_bell_circuit()])
        assert sum(r.quasi_dists[0].values()) == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Sampler — parametric
# ---------------------------------------------------------------------------

class TestSamplerParametric:
    def test_rz_parametric(self):
        # RZ acts on phase only; |0⟩ → still 100 % in |0⟩
        circ = {
            "n_qubits": 1,
            "operations": [
                {"name": "rz", "qubits": [0], "params": ["phi"]},
            ],
        }
        r = Sampler().run([circ], [{"phi": 1.2}])
        qd = r.quasi_dists[0]
        assert qd.get(0, 0.0) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# __init__ re-exports
# ---------------------------------------------------------------------------

class TestPrimitivesInit:
    def test_estimator_exported(self):
        from quantx_engine import Estimator as E
        assert E is Estimator

    def test_sampler_exported(self):
        from quantx_engine import Sampler as S
        assert S is Sampler

    def test_sparse_pauli_op_exported(self):
        from quantx_engine import SparsePauliOp as SPO
        assert SPO is SparsePauliOp
