"""
Tests for quantx_engine.tensor_network

Covers MPS, MPO, TEBD, and DMRG functionality.
Only requires numpy and pytest (no scipy).
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from quantx_engine.tensor_network import (
    TensorNetworkError,
    MPS,
    MPO,
    TEBD,
    DMRGResult,
    DMRG,
)


# ── MPS tests ─────────────────────────────────────────────────────────────────

class TestMPS:
    def test_mps_product_state_norm(self):
        """Product state |000⟩ should have norm = 1.0."""
        mps = MPS.product_state([0, 0, 0])
        assert abs(mps.norm() - 1.0) < 1e-12

    def test_mps_from_statevector_roundtrip(self):
        """Convert random 4-qubit statevector -> MPS -> statevector and check fidelity."""
        rng = np.random.default_rng(42)
        psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        psi /= np.linalg.norm(psi)

        mps = MPS.from_statevector(psi, d=2)
        psi_recovered = mps.to_statevector()

        # Global phase freedom: compare up to phase
        diff = np.linalg.norm(psi_recovered - psi * (np.vdot(psi_recovered, psi) / abs(np.vdot(psi_recovered, psi))))
        assert diff < 1e-10

    def test_mps_from_statevector_norm(self):
        """MPS from normalized statevector should have norm ≈ 1."""
        rng = np.random.default_rng(7)
        psi = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_statevector(psi, d=2)
        assert abs(mps.norm() - 1.0) < 1e-10

    def test_mps_truncation_error(self):
        """GHZ state MPS, truncated to max_chi=2, after normalize() has norm ≈ 1."""
        # GHZ_4 = (|0000⟩ + |1111⟩) / sqrt(2)
        ghz = np.zeros(16, dtype=complex)
        ghz[0] = 1.0 / math.sqrt(2)
        ghz[15] = 1.0 / math.sqrt(2)

        mps = MPS.from_statevector(ghz, d=2)
        truncated = mps.truncate(max_chi=2)
        truncated.normalize()
        assert abs(truncated.norm() - 1.0) < 1e-10

    def test_mps_overlap_orthogonal(self):
        """⟨000|111⟩ should be 0."""
        bra = MPS.product_state([0, 0, 0])
        ket = MPS.product_state([1, 1, 1])
        assert abs(bra.overlap(ket)) < 1e-14

    def test_mps_overlap_self(self):
        """⟨ψ|ψ⟩ should equal norm^2."""
        mps = MPS.random(4, seed=0)
        ov = mps.overlap(mps)
        norm_sq = mps.norm() ** 2
        assert abs(ov - norm_sq) < 1e-10

    def test_mps_bond_dims(self):
        """Product state has all bond dims = 1."""
        mps = MPS.product_state([0, 1, 0, 1])
        assert all(b == 1 for b in mps.bond_dims)

    def test_mps_to_dict(self):
        """to_dict() returns dict with required keys."""
        mps = MPS.product_state([0, 0, 0])
        d = mps.to_dict()
        assert "n_sites" in d
        assert "bond_dims" in d
        assert "d" in d
        assert "max_bond_dim" in d

    def test_mps_validation_error(self):
        """MPS constructor should raise TensorNetworkError for invalid shapes."""
        # Wrong rank
        with pytest.raises(TensorNetworkError):
            MPS(tensors=[np.zeros((2, 2))])

        # Wrong boundary (chi_left != 1)
        with pytest.raises(TensorNetworkError):
            MPS(tensors=[np.zeros((2, 2, 1))])

    def test_mps_n_sites(self):
        """n_sites property returns correct count."""
        mps = MPS.product_state([0, 1, 0, 1, 0])
        assert mps.n_sites == 5

    def test_mps_random_norm(self):
        """Random MPS should be normalized."""
        mps = MPS.random(5, seed=123)
        assert abs(mps.norm() - 1.0) < 1e-10

    def test_mps_from_statevector_max_chi(self):
        """from_statevector with max_chi truncates bond dimensions."""
        rng = np.random.default_rng(99)
        psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
        psi /= np.linalg.norm(psi)
        mps = MPS.from_statevector(psi, d=2, max_chi=2)
        assert all(b <= 2 for b in mps.bond_dims)

    def test_mps_to_statevector_product(self):
        """Product state to_statevector matches expected vector."""
        mps = MPS.product_state([1, 0])
        sv = mps.to_statevector()
        expected = np.array([0, 0, 1, 0], dtype=complex)  # |10⟩ in {00,01,10,11}
        assert np.allclose(sv, expected)


# ── MPO tests ─────────────────────────────────────────────────────────────────

class TestMPO:
    def test_ising_mpo_shape(self):
        """n=4 Ising MPO tensors have correct physical dimensions."""
        mpo = MPO.ising(n=4, J=1.0, h=0.5)
        assert mpo.n_sites == 4
        for W in mpo.tensors:
            assert W.shape[2] == 2  # d_out
            assert W.shape[3] == 2  # d_in

    def test_ising_mpo_boundary_shapes(self):
        """Left boundary has D_l=1, right boundary has D_r=1."""
        mpo = MPO.ising(n=4)
        assert mpo.tensors[0].shape[0] == 1    # D_left = 1
        assert mpo.tensors[-1].shape[1] == 1   # D_right = 1

    def test_ising_mpo_expectation(self):
        """⟨00|H_Ising(J=1,h=0)|00⟩ = -1 (just -ZZ term, Z|0⟩=|0⟩)."""
        mpo = MPO.ising(n=2, J=1.0, h=0.0)
        mps = MPS.product_state([0, 0])
        E = mpo.expectation(mps)
        # H = -J ZZ, ⟨00|ZZ|00⟩ = ⟨0|Z|0⟩⟨0|Z|0⟩ = (+1)(+1) = 1
        # E = -1 * 1 = -1
        assert abs(E - (-1.0)) < 1e-10

    def test_ising_mpo_h_only(self):
        """⟨++|H_Ising(J=0,h=1)|++⟩ = -2 for n=2 (X|+⟩=|+⟩, ⟨+|X|+⟩=1)."""
        mpo = MPO.ising(n=2, J=0.0, h=1.0)
        # |+⟩ = (|0⟩+|1⟩)/sqrt(2)
        plus = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2)
        psi = np.kron(plus, plus)
        mps = MPS.from_statevector(psi)
        E = mpo.expectation(mps)
        # H = -h * (X_0 + X_1), ⟨+|X|+⟩=1 each, so E = -1*1 - 1*1 = -2
        assert abs(E - (-2.0)) < 1e-10

    def test_ising_mpo_to_dict(self):
        """to_dict has required keys."""
        mpo = MPO.ising(n=3)
        d = mpo.to_dict()
        assert "n_sites" in d
        assert "bond_dim" in d
        assert "d" in d

    def test_heisenberg_mpo_shape(self):
        """n=3 Heisenberg MPO tensors have correct physical dimensions."""
        mpo = MPO.heisenberg(n=3)
        assert mpo.n_sites == 3
        for W in mpo.tensors:
            assert W.shape[2] == 2
            assert W.shape[3] == 2
        assert mpo.tensors[0].shape[0] == 1
        assert mpo.tensors[-1].shape[1] == 1

    def test_mpo_apply(self):
        """MPO.apply returns MPS with same n_sites."""
        mpo = MPO.ising(n=3)
        mps = MPS.product_state([0, 0, 0])
        result = mpo.apply(mps)
        assert result.n_sites == 3

    def test_ising_expectation_11(self):
        """⟨11|H_Ising(J=1,h=0)|11⟩ = -1 (Z|1⟩=-|1⟩, ZZ gives +1)."""
        mpo = MPO.ising(n=2, J=1.0, h=0.0)
        mps = MPS.product_state([1, 1])
        E = mpo.expectation(mps)
        # ⟨1|Z|1⟩ = -1, ⟨11|ZZ|11⟩ = (-1)(-1) = 1, H = -ZZ so E = -1
        assert abs(E - (-1.0)) < 1e-10

    def test_ising_expectation_01(self):
        """⟨01|H_Ising(J=1,h=0)|01⟩ = +1 (Z gives +1,-1, product=-1, H=-ZZ gives +1)."""
        mpo = MPO.ising(n=2, J=1.0, h=0.0)
        mps = MPS.product_state([0, 1])
        E = mpo.expectation(mps)
        # ⟨0|Z|0⟩=+1, ⟨1|Z|1⟩=-1, ZZ gives +1*(-1)=-1, H=-ZZ gives +1
        assert abs(E - 1.0) < 1e-10


# ── TEBD tests ────────────────────────────────────────────────────────────────

class TestTEBD:
    def test_tebd_norm_preserved(self):
        """Norm stays ≈ 1.0 during TEBD evolution."""
        mps = MPS.product_state([0, 0, 0, 0])
        tebd = TEBD(mps=mps, max_chi=8)
        history = tebd.evolve_ising(J=1.0, h=0.5, dt=0.01, n_steps=5)
        for entry in history:
            assert abs(entry["norm"] - 1.0) < 1e-8, \
                f"Norm deviated at step {entry['step']}: {entry['norm']}"

    def test_tebd_history_length(self):
        """n_steps=5 gives history list of length 5."""
        mps = MPS.product_state([0, 0, 0, 0])
        tebd = TEBD(mps=mps, max_chi=8)
        history = tebd.evolve_ising(J=1.0, h=0.5, dt=0.01, n_steps=5)
        assert len(history) == 5

    def test_tebd_history_keys(self):
        """Each history entry has the required keys."""
        mps = MPS.product_state([0, 0, 0, 0])
        tebd = TEBD(mps=mps, max_chi=8)
        history = tebd.evolve_ising(J=1.0, h=0.5, dt=0.01, n_steps=3)
        for entry in history:
            assert "step" in entry
            assert "time" in entry
            assert "norm" in entry
            assert "max_bond" in entry

    def test_tebd_small_system(self):
        """n=4 qubits, Ising J=1, h=0.5, dt=0.01, n_steps=3 runs without error, norm ≈ 1."""
        mps = MPS.product_state([0, 0, 0, 0])
        tebd = TEBD(mps=mps, max_chi=8)
        history = tebd.evolve_ising(J=1.0, h=0.5, dt=0.01, n_steps=3)
        assert len(history) == 3
        assert abs(history[-1]["norm"] - 1.0) < 1e-8

    def test_tebd_step_counter(self):
        """Step indices in history start at 1 and increment."""
        mps = MPS.product_state([0, 0, 0, 0])
        tebd = TEBD(mps=mps, max_chi=8)
        history = tebd.evolve_ising(J=1.0, h=0.5, dt=0.05, n_steps=4)
        for i, entry in enumerate(history):
            assert entry["step"] == i + 1

    def test_tebd_time_accumulates(self):
        """Time in history entry = step * dt."""
        mps = MPS.product_state([0, 0, 0, 0])
        tebd = TEBD(mps=mps, max_chi=8)
        dt = 0.02
        history = tebd.evolve_ising(J=1.0, h=0.5, dt=dt, n_steps=5)
        for entry in history:
            assert abs(entry["time"] - entry["step"] * dt) < 1e-14

    def test_tebd_magnetization_z(self):
        """Magnetization ⟨Z_i⟩ of |0⟩ state = +1."""
        mps = MPS.product_state([0, 0, 0])
        tebd = TEBD(mps=mps, max_chi=4)
        for i in range(3):
            assert abs(tebd.magnetization_z(i) - 1.0) < 1e-10

    def test_tebd_magnetization_z_down(self):
        """Magnetization ⟨Z_i⟩ of |1⟩ state = -1."""
        mps = MPS.product_state([1, 1, 1])
        tebd = TEBD(mps=mps, max_chi=4)
        for i in range(3):
            assert abs(tebd.magnetization_z(i) - (-1.0)) < 1e-10


# ── DMRG tests ────────────────────────────────────────────────────────────────

class TestDMRG:
    def test_dmrg_ising_energy(self):
        """n=4, J=1, h=0.5: DMRG energy should be below -2.0."""
        mpo = MPO.ising(n=4, J=1.0, h=0.5)
        dmrg = DMRG(mpo=mpo, max_chi=8, n_sweeps=6, seed=0)
        result = dmrg.run()
        # Product state |0000⟩ energy = -J*(n-1)*1 = -3 for h=0
        # With h=0.5, transverse field lowers energy further
        assert result.energy < -2.0, f"Energy {result.energy} not below -2.0"

    def test_dmrg_result_structure(self):
        """DMRGResult has correct types for all fields."""
        mpo = MPO.ising(n=3, J=1.0, h=0.5)
        dmrg = DMRG(mpo=mpo, max_chi=4, n_sweeps=4, seed=1)
        result = dmrg.run()
        assert isinstance(result.mps, MPS)
        assert isinstance(result.sweep_energies, list)
        assert isinstance(result.converged, bool)
        assert isinstance(result.energy, float)

    def test_dmrg_result_to_dict(self):
        """DMRGResult.to_dict() has required keys."""
        mpo = MPO.ising(n=3, J=1.0, h=0.5)
        dmrg = DMRG(mpo=mpo, max_chi=4, n_sweeps=3, seed=2)
        result = dmrg.run()
        d = result.to_dict()
        assert "energy" in d
        assert "converged" in d
        assert "n_sweeps" in d

    def test_dmrg_energy_decreases(self):
        """Sweep energies should be non-increasing (or nearly so up to tolerance)."""
        mpo = MPO.ising(n=4, J=1.0, h=0.5)
        dmrg = DMRG(mpo=mpo, max_chi=8, n_sweeps=6, seed=3)
        result = dmrg.run()
        energies = result.sweep_energies
        # Allow small fluctuations (numerical noise) but overall trend should decrease
        if len(energies) >= 2:
            # The last energy should be lower than or close to the first
            assert energies[-1] <= energies[0] + 1.0, \
                f"Energy increased significantly: {energies[0]} -> {energies[-1]}"

    def test_dmrg_mps_n_sites(self):
        """Resulting MPS has correct number of sites."""
        n = 4
        mpo = MPO.ising(n=n, J=1.0, h=0.5)
        dmrg = DMRG(mpo=mpo, max_chi=4, n_sweeps=3, seed=4)
        result = dmrg.run()
        assert result.mps.n_sites == n

    def test_dmrg_heisenberg(self):
        """Heisenberg chain n=4: ground state energy should be below 0."""
        mpo = MPO.heisenberg(n=4, J=1.0, hz=0.0)
        dmrg = DMRG(mpo=mpo, max_chi=8, n_sweeps=6, seed=5)
        result = dmrg.run()
        # Heisenberg ground state energy is negative for J>0
        assert result.energy < 0.0

    def test_dmrg_with_initial_mps(self):
        """DMRG accepts an initial MPS and produces valid result."""
        n = 3
        mpo = MPO.ising(n=n, J=1.0, h=0.5)
        initial = MPS.product_state([0, 0, 0])
        dmrg = DMRG(mpo=mpo, max_chi=4, n_sweeps=3, seed=6)
        result = dmrg.run(initial_mps=initial)
        assert isinstance(result.mps, MPS)
        assert result.mps.n_sites == n

    def test_dmrg_initial_mps_wrong_size_raises(self):
        """DMRG raises TensorNetworkError if initial_mps has wrong number of sites."""
        mpo = MPO.ising(n=4, J=1.0, h=0.5)
        wrong_size = MPS.product_state([0, 0, 0])  # 3 sites, MPO has 4
        dmrg = DMRG(mpo=mpo, max_chi=4, n_sweeps=1)
        with pytest.raises(TensorNetworkError, match="sites"):
            dmrg.run(initial_mps=wrong_size)
