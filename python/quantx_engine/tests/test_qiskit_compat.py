"""M116: Tests for Qiskit 1.x Primitives V2 compatibility layer."""
from __future__ import annotations

import math
import pytest
import numpy as np

from quantx_engine.qiskit_compat import (
    QiskitCompatError,
    BitArray,
    DataBin,
    PubResult,
    PrimitiveResult,
    SamplerPUB,
    EstimatorPUB,
    QuantXSamplerV2,
    QuantXEstimatorV2,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _zero_state(n_qubits: int) -> np.ndarray:
    """Statevector |0...0⟩."""
    sv = np.zeros(1 << n_qubits, dtype=complex)
    sv[0] = 1.0
    return sv


def _plus_state(n_qubits: int) -> np.ndarray:
    """Statevector |+⟩^⊗n."""
    sv = np.ones(1 << n_qubits, dtype=complex)
    sv /= np.sqrt(len(sv))
    return sv


def _bell_state() -> np.ndarray:
    """Bell state (|00⟩ + |11⟩)/√2."""
    sv = np.zeros(4, dtype=complex)
    sv[0] = sv[3] = 1.0 / math.sqrt(2)
    return sv


def _pauli_z(n: int = 1) -> np.ndarray:
    """Z⊗n matrix."""
    Z = np.array([[1., 0.], [0., -1.]])
    mat = Z
    for _ in range(n - 1):
        mat = np.kron(mat, Z)
    return mat.astype(complex)


# ── BitArray ──────────────────────────────────────────────────────────────────

class TestBitArray:
    def test_basic_construction(self):
        data = np.zeros((10, 2), dtype=np.uint8)
        ba = BitArray(data, num_bits=2)
        assert ba.num_shots == 10
        assert ba.num_bits == 2

    def test_array_property(self):
        data = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        ba = BitArray(data, num_bits=2)
        np.testing.assert_array_equal(ba.array, data)

    def test_wrong_shape_raises(self):
        data = np.zeros((10, 3), dtype=np.uint8)
        with pytest.raises(QiskitCompatError):
            BitArray(data, num_bits=2)

    def test_1d_single_bit(self):
        data = np.array([0, 1, 0, 1], dtype=np.uint8)
        ba = BitArray(data, num_bits=1)
        assert ba.num_shots == 4
        assert ba.num_bits == 1

    def test_get_counts(self):
        data = np.array([[0, 0], [0, 0], [1, 1]], dtype=np.uint8)
        ba = BitArray(data, num_bits=2)
        counts = ba.get_counts()
        # '00' → 2 shots, '11' → 1 shot (qubit order: reversed, MSB first)
        assert isinstance(counts, dict)
        total = sum(counts.values())
        assert total == 3

    def test_get_int_counts(self):
        data = np.array([[0, 0], [0, 0], [1, 0]], dtype=np.uint8)
        ba = BitArray(data, num_bits=2)
        counts = ba.get_int_counts()
        assert counts.get(0, 0) == 2
        assert counts.get(1, 0) == 1

    def test_get_counts_sums_to_shots(self):
        rng = np.random.default_rng(0)
        data = rng.integers(0, 2, (100, 3), dtype=np.uint8)
        ba = BitArray(data, num_bits=3)
        assert sum(ba.get_counts().values()) == 100

    def test_repr(self):
        ba = BitArray(np.zeros((5, 3), dtype=np.uint8), num_bits=3)
        assert "BitArray" in repr(ba)
        assert "5" in repr(ba)

    def test_all_zeros(self):
        data = np.zeros((20, 4), dtype=np.uint8)
        ba = BitArray(data, num_bits=4)
        counts = ba.get_int_counts()
        assert counts == {0: 20}


# ── DataBin ───────────────────────────────────────────────────────────────────

class TestDataBin:
    def test_sampler_factory(self):
        data = np.zeros((10, 2), dtype=np.uint8)
        ba = BitArray(data, num_bits=2)
        db = DataBin.sampler(ba)
        assert hasattr(db, "meas")
        assert db.meas is ba

    def test_estimator_factory(self):
        evs = np.array([0.5, -0.3])
        stds = np.array([0.0, 0.0])
        db = DataBin.estimator(evs=evs, stds=stds)
        assert hasattr(db, "evs")
        assert hasattr(db, "stds")
        np.testing.assert_array_equal(db.evs, evs)

    def test_fields_list(self):
        db = DataBin.sampler(BitArray(np.zeros((5, 2), dtype=np.uint8), num_bits=2))
        assert "meas" in db.fields()

    def test_custom_fields(self):
        db = DataBin(counts={"00": 5}, shots=5)
        assert hasattr(db, "counts")

    def test_repr(self):
        db = DataBin.estimator(np.array([1.0]), np.array([0.0]))
        assert "DataBin" in repr(db)


# ── PubResult / PrimitiveResult ────────────────────────────────────────────────

class TestPrimitiveResult:
    def test_indexing(self):
        db = DataBin.estimator(np.array([1.0]), np.array([0.0]))
        pr = PrimitiveResult([PubResult(db)])
        assert isinstance(pr[0], PubResult)

    def test_len(self):
        dbs = [DataBin.estimator(np.array([float(i)]), np.array([0.0])) for i in range(3)]
        pr = PrimitiveResult([PubResult(d) for d in dbs])
        assert len(pr) == 3

    def test_iteration(self):
        db = DataBin.estimator(np.array([0.0]), np.array([0.0]))
        pr = PrimitiveResult([PubResult(db)] * 4)
        assert len(list(pr)) == 4

    def test_metadata_stored(self):
        pr = PrimitiveResult([], metadata={"run_id": "abc"})
        assert pr.metadata["run_id"] == "abc"

    def test_pub_result_metadata(self):
        db = DataBin.sampler(BitArray(np.zeros((5, 1), dtype=np.uint8), num_bits=1))
        pr = PubResult(db, metadata={"shots": 5})
        assert pr.metadata["shots"] == 5


# ── SamplerPUB ────────────────────────────────────────────────────────────────

class TestSamplerPUB:
    def test_circuit_only(self):
        sv = _zero_state(2)
        pub = SamplerPUB(sv)
        assert pub.circuit is sv
        assert pub.shots == 1024  # default
        assert pub.parameter_values is None

    def test_circuit_shots_tuple(self):
        sv = _zero_state(2)
        pub = SamplerPUB((sv, 512))
        assert pub.shots == 512

    def test_circuit_params_shots_tuple(self):
        sv = _zero_state(2)
        params = {"theta": 0.5}
        pub = SamplerPUB((sv, params, 256))
        assert pub.parameter_values == params
        assert pub.shots == 256

    def test_empty_raises(self):
        with pytest.raises(QiskitCompatError):
            SamplerPUB(())


# ── EstimatorPUB ──────────────────────────────────────────────────────────────

class TestEstimatorPUB:
    def test_circuit_obs_tuple(self):
        sv = _zero_state(1)
        obs = {"Z": 1.0}
        pub = EstimatorPUB((sv, obs))
        assert pub.circuit is sv
        assert pub.observables is obs

    def test_circuit_obs_params_precision(self):
        sv = _zero_state(1)
        obs = {"Z": 1.0}
        pub = EstimatorPUB((sv, obs, {}, 0.01))
        assert pub.precision == pytest.approx(0.01)

    def test_not_tuple_raises(self):
        with pytest.raises(QiskitCompatError):
            EstimatorPUB(_zero_state(1))

    def test_too_short_raises(self):
        with pytest.raises(QiskitCompatError):
            EstimatorPUB((_zero_state(1),))


# ── QuantXSamplerV2 construction ──────────────────────────────────────────────

class TestSamplerV2Construction:
    def test_default_shots(self):
        sampler = QuantXSamplerV2()
        assert sampler.default_shots == 1024

    def test_explicit_shots(self):
        sampler = QuantXSamplerV2(default_shots=512)
        assert sampler.default_shots == 512

    def test_zero_shots_raises(self):
        with pytest.raises(QiskitCompatError):
            QuantXSamplerV2(default_shots=0)

    def test_empty_pubs_raises(self):
        with pytest.raises(QiskitCompatError):
            QuantXSamplerV2().run([])


# ── QuantXSamplerV2 sampling ──────────────────────────────────────────────────

class TestSamplerV2Sampling:
    def test_returns_primitive_result(self):
        sampler = QuantXSamplerV2(seed=0)
        result = sampler.run([(_zero_state(2),)])
        assert isinstance(result, PrimitiveResult)

    def test_one_pub_one_result(self):
        sampler = QuantXSamplerV2(seed=0)
        result = sampler.run([_zero_state(2)])
        assert len(result) == 1

    def test_multiple_pubs(self):
        sampler = QuantXSamplerV2(seed=0)
        result = sampler.run([_zero_state(2), _plus_state(2)])
        assert len(result) == 2

    def test_data_has_meas(self):
        sampler = QuantXSamplerV2(seed=0)
        result = sampler.run([_zero_state(2)])
        assert hasattr(result[0].data, "meas")
        assert isinstance(result[0].data.meas, BitArray)

    def test_shots_count(self):
        sampler = QuantXSamplerV2(default_shots=500, seed=0)
        result = sampler.run([_zero_state(2)])
        assert result[0].data.meas.num_shots == 500

    def test_shots_override(self):
        sampler = QuantXSamplerV2(default_shots=1000, seed=0)
        result = sampler.run([(_zero_state(2), 200)])
        assert result[0].data.meas.num_shots == 200

    def test_global_shots_override(self):
        sampler = QuantXSamplerV2(default_shots=100, seed=0)
        result = sampler.run([_zero_state(2)], shots=300)
        assert result[0].data.meas.num_shots == 300

    def test_zero_state_always_zero(self):
        """Measuring |0⟩ should always yield '00'."""
        sampler = QuantXSamplerV2(default_shots=100, seed=42)
        result = sampler.run([_zero_state(2)])
        counts = result[0].data.meas.get_int_counts()
        assert counts == {0: 100}

    def test_plus_state_uniform(self):
        """Measuring |+⟩^⊗1 should give 0 and 1 roughly equally."""
        sampler = QuantXSamplerV2(default_shots=2000, seed=7)
        result = sampler.run([_plus_state(1)])
        counts = result[0].data.meas.get_int_counts()
        total = sum(counts.values())
        assert total == 2000
        # Both outcomes should appear
        assert len(counts) == 2

    def test_bell_state_correlated(self):
        """Bell state (|00⟩+|11⟩)/√2 should yield only 00 or 11."""
        sampler = QuantXSamplerV2(default_shots=500, seed=5)
        result = sampler.run([_bell_state()])
        counts = result[0].data.meas.get_int_counts()
        # Only integer 0 (00) and 3 (11) should appear
        assert set(counts.keys()).issubset({0, 3})

    def test_reproducible_with_seed(self):
        sv = _plus_state(3)
        r1 = QuantXSamplerV2(seed=99).run([sv])
        r2 = QuantXSamplerV2(seed=99).run([sv])
        np.testing.assert_array_equal(
            r1[0].data.meas.array, r2[0].data.meas.array
        )

    def test_num_bits_matches_n_qubits(self):
        sampler = QuantXSamplerV2(seed=0)
        for n in [1, 2, 3]:
            result = sampler.run([_zero_state(n)])
            assert result[0].data.meas.num_bits == n

    def test_metadata_stored(self):
        sampler = QuantXSamplerV2(seed=0)
        result = sampler.run([(_zero_state(2), 100)])
        assert "shots" in result[0].metadata or "n_qubits" in result[0].metadata


# ── QuantXEstimatorV2 construction ────────────────────────────────────────────

class TestEstimatorV2Construction:
    def test_default_precision(self):
        estimator = QuantXEstimatorV2()
        assert estimator.default_precision == 0.0

    def test_explicit_precision(self):
        estimator = QuantXEstimatorV2(default_precision=0.01)
        assert estimator.default_precision == 0.01

    def test_empty_pubs_raises(self):
        with pytest.raises(QiskitCompatError):
            QuantXEstimatorV2().run([])


# ── QuantXEstimatorV2 expectation values ──────────────────────────────────────

class TestEstimatorV2ExpectationValues:
    def test_returns_primitive_result(self):
        sv = _zero_state(1)
        obs = {"Z": 1.0}
        result = QuantXEstimatorV2().run([(sv, obs)])
        assert isinstance(result, PrimitiveResult)

    def test_data_has_evs_and_stds(self):
        sv = _zero_state(1)
        obs = {"Z": 1.0}
        result = QuantXEstimatorV2().run([(sv, obs)])
        assert hasattr(result[0].data, "evs")
        assert hasattr(result[0].data, "stds")

    def test_z_on_zero_state(self):
        """⟨0|Z|0⟩ = 1."""
        sv = _zero_state(1)
        # Z = diag(1, -1)
        obs = np.array([[1, 0], [0, -1]], dtype=complex)
        result = QuantXEstimatorV2().run([(sv, obs)])
        assert abs(result[0].data.evs[0] - 1.0) < 1e-10

    def test_z_on_one_state(self):
        """⟨1|Z|1⟩ = -1."""
        sv = np.array([0.0, 1.0], dtype=complex)
        obs = np.array([[1, 0], [0, -1]], dtype=complex)
        result = QuantXEstimatorV2().run([(sv, obs)])
        assert abs(result[0].data.evs[0] + 1.0) < 1e-10

    def test_x_on_plus_state(self):
        """⟨+|X|+⟩ = 1."""
        sv = _plus_state(1)
        obs = np.array([[0, 1], [1, 0]], dtype=complex)
        result = QuantXEstimatorV2().run([(sv, obs)])
        assert abs(result[0].data.evs[0] - 1.0) < 1e-10

    def test_multiple_observables(self):
        sv = _zero_state(2)
        obs_list = [
            np.kron(np.diag([1., -1.]), np.eye(2)),   # ZI
            np.kron(np.eye(2), np.diag([1., -1.])),   # IZ
        ]
        result = QuantXEstimatorV2().run([(sv, obs_list)])
        assert result[0].data.evs.shape == (2,)
        # |00⟩: ZI=+1, IZ=+1
        assert abs(result[0].data.evs[0] - 1.0) < 1e-10
        assert abs(result[0].data.evs[1] - 1.0) < 1e-10

    def test_stds_are_zero(self):
        """Exact statevector → zero variance."""
        sv = _zero_state(2)
        obs = np.eye(4, dtype=complex)  # identity → EV=1
        result = QuantXEstimatorV2().run([(sv, obs)])
        assert np.all(result[0].data.stds == 0.0)

    def test_pauli_dict_observable(self):
        """Dict observable: {'ZI': 1.0, 'IZ': 0.5} on |00⟩."""
        sv = _zero_state(2)
        obs = {"ZI": 1.0, "IZ": 0.5}
        result = QuantXEstimatorV2().run([(sv, obs)])
        # ZI|00⟩: Z⊗I on |00⟩ → ⟨ZI⟩ = +1; IZ: ⟨IZ⟩ = +1
        # combined: 1*1 + 0.5*1 = 1.5
        assert abs(result[0].data.evs[0] - 1.5) < 1e-10

    def test_bell_state_zz(self):
        """⟨Φ+|ZZ|Φ+⟩ = 1 (Bell state)."""
        sv = _bell_state()
        ZZ = np.kron(np.diag([1., -1.]), np.diag([1., -1.]))
        result = QuantXEstimatorV2().run([(sv, ZZ)])
        assert abs(result[0].data.evs[0] - 1.0) < 1e-10

    def test_multiple_pubs(self):
        sv0 = _zero_state(1)
        sv1 = np.array([0.0, 1.0], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        result = QuantXEstimatorV2().run([(sv0, Z), (sv1, Z)])
        assert len(result) == 2
        assert abs(result[0].data.evs[0] - 1.0) < 1e-10
        assert abs(result[1].data.evs[0] + 1.0) < 1e-10

    def test_unsupported_observable_raises(self):
        sv = _zero_state(1)
        with pytest.raises(QiskitCompatError):
            QuantXEstimatorV2().run([(sv, "invalid_obs")])

    def test_pauli_hamiltonian_observable(self):
        """PauliHamiltonian observable on |0⟩."""
        from quantx_engine.chemistry import PauliHamiltonian, PauliTerm
        ham = PauliHamiltonian(
            terms=[PauliTerm(coeff=-0.5+0j, ops={0: "Z"})],
            n_qubits=1,
        )
        sv = _zero_state(1)
        result = QuantXEstimatorV2().run([(sv, ham)])
        # ⟨0|(-0.5 Z)|0⟩ = -0.5
        assert abs(result[0].data.evs[0] + 0.5) < 1e-10

    def test_precision_stored_in_metadata(self):
        sv = _zero_state(1)
        obs = np.eye(2, dtype=complex)
        result = QuantXEstimatorV2().run([(sv, obs, {}, 0.02)])
        assert "precision" in result[0].metadata


# ── Codex review regression tests ─────────────────────────────────────────────

class TestQiskitCompatRegressions:
    """Regression tests for Codex P1/P2 review findings on M116."""

    def test_callable_circuit_receives_parameter_values_sampler(self):
        """P2: SamplerPUB callable must receive parameter_values from the PUB.

        Before the fix, circuit() was always called with no arguments, so
        parameterised factories silently ignored their PUB parameters.
        """
        received: dict = {}

        def circuit_factory(params):
            received["params"] = params
            # Return a simple |+⟩ statevector regardless of params
            sv = np.ones(2, dtype=complex) / math.sqrt(2)
            return sv

        sampler = QuantXSamplerV2(default_shots=100, seed=0)
        result = sampler.run([(circuit_factory, {"theta": 0.42}, 100)])
        # Factory must have been called with the parameter dict
        assert "params" in received
        assert received["params"] == {"theta": 0.42}
        # Result must be valid
        assert isinstance(result, PrimitiveResult)

    def test_callable_circuit_receives_parameter_values_estimator(self):
        """P2: EstimatorPUB callable must receive parameter_values from the PUB."""
        received: dict = {}

        def circuit_factory(params):
            received["params"] = params
            return _zero_state(1)

        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        estimator = QuantXEstimatorV2()
        result = estimator.run([(circuit_factory, Z, {"theta": 1.0})])
        assert "params" in received
        assert received["params"] == {"theta": 1.0}
        assert abs(result[0].data.evs[0] - 1.0) < 1e-10

    def test_callable_without_parameters_still_works(self):
        """Callables with no PUB parameters are still called with no arguments."""
        called: list[int] = []

        def circuit_factory():
            called.append(1)
            return _zero_state(1)

        sampler = QuantXSamplerV2(default_shots=50, seed=0)
        result = sampler.run([(circuit_factory, 50)])
        assert len(called) == 1  # factory was called exactly once
        assert isinstance(result, PrimitiveResult)

    def test_sparse_pauli_op_observable(self):
        """P2: SparsePauliOp must work as an estimator observable.

        Before the fix, observable.to_matrix() was called but SparsePauliOp
        has no such method; the AttributeError was swallowed and the call
        fell through to an 'Unsupported observable type' error.
        """
        try:
            from quantx_engine.primitives import SparsePauliOp
        except ImportError:
            pytest.skip("primitives module not available")

        # Build Z observable using SparsePauliOp
        obs = SparsePauliOp("Z")
        sv = _zero_state(1)
        estimator = QuantXEstimatorV2()
        result = estimator.run([(sv, obs)])
        # ⟨0|Z|0⟩ = +1
        assert abs(result[0].data.evs[0] - 1.0) < 1e-10

    def test_sparse_pauli_op_multi_term(self):
        """SparsePauliOp with duplicate Pauli strings must aggregate coefficients.

        0.5·Z + 0.5·Z = 1.0·Z; ⟨1|Z|1⟩ = -1.0.
        Before the fix, dict(observable.terms) would discard the second term,
        leaving only 0.5·Z and returning -0.5 instead of -1.0.
        """
        try:
            from quantx_engine.primitives import SparsePauliOp
        except ImportError:
            pytest.skip("primitives module not available")

        obs = SparsePauliOp([("Z", 0.5), ("Z", 0.5)])  # sums to Z
        sv = np.array([0.0, 1.0], dtype=complex)  # |1⟩
        estimator = QuantXEstimatorV2()
        result = estimator.run([(sv, obs)])
        # ⟨1|(0.5Z + 0.5Z)|1⟩ = ⟨1|Z|1⟩ = -1.0
        assert abs(result[0].data.evs[0] + 1.0) < 1e-10


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration:
    def test_sampler_counts_sum_to_shots(self):
        for n in [1, 2, 3]:
            sampler = QuantXSamplerV2(default_shots=500, seed=0)
            result = sampler.run([_plus_state(n)])
            counts = result[0].data.meas.get_counts()
            assert sum(counts.values()) == 500

    def test_estimator_single_qubit_rotation(self):
        """Verify ⟨Rz(θ)|Z|Rz(θ)⟩ = cos(θ) by sweeping θ."""
        estimator = QuantXEstimatorV2()
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        thetas = [0, math.pi / 4, math.pi / 2, math.pi]
        for theta in thetas:
            # Rz(θ)|0⟩ = cos(θ/2)|0⟩ + i sin(θ/2)|1⟩... wait
            # Actually e^{-iθZ/2}|0⟩ = [cos(θ/2), -i sin(θ/2)]
            # ⟨Z⟩ = cos²(θ/2) - sin²(θ/2) = cos(θ)
            c = np.cos(theta / 2)
            s = np.sin(theta / 2)
            sv = np.array([c, -1j * s], dtype=complex)
            result = estimator.run([(sv, Z)])
            expected = math.cos(theta)
            assert abs(result[0].data.evs[0] - expected) < 1e-10

    def test_sampler_then_estimator_same_state(self):
        """Both primitives on the same Bell state."""
        sv = _bell_state()
        sampler = QuantXSamplerV2(default_shots=1000, seed=42)
        estimator = QuantXEstimatorV2()

        s_result = sampler.run([sv])
        ZZ = np.kron(np.diag([1., -1.]), np.diag([1., -1.]))
        e_result = estimator.run([(sv, ZZ)])

        # Sampler: only 00 and 11 outcomes
        counts = s_result[0].data.meas.get_int_counts()
        assert set(counts.keys()).issubset({0, 3})
        # Estimator: ⟨ZZ⟩ = 1
        assert abs(e_result[0].data.evs[0] - 1.0) < 1e-10
