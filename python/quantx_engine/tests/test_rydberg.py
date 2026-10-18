"""M118: Tests for Rydberg neutral-atom simulator."""
from __future__ import annotations

import math
import pytest
import numpy as np

from quantx_engine.contrib.rydberg import (
    RydbergError,
    RydbergAtom,
    RydbergHamiltonian,
    RydbergResult,
    RydbergSimulator,
)


# ── RydbergAtom ───────────────────────────────────────────────────────────────

class TestRydbergAtom:
    def test_basic_creation(self):
        atom = RydbergAtom(x=1.0, y=2.0)
        assert atom.x == 1.0
        assert atom.y == 2.0

    def test_position_property(self):
        atom = RydbergAtom(x=3.0, y=4.0)
        assert atom.position == (3.0, 4.0)

    def test_distance(self):
        a = RydbergAtom(0.0, 0.0)
        b = RydbergAtom(3.0, 4.0)
        assert abs(a.distance_to(b) - 5.0) < 1e-10

    def test_distance_self(self):
        a = RydbergAtom(1.0, 2.0)
        assert a.distance_to(a) == 0.0

    def test_label_default_empty(self):
        a = RydbergAtom(0.0, 0.0)
        assert a.label == ""

    def test_label_stored(self):
        a = RydbergAtom(0.0, 0.0, label="atom_0")
        assert a.label == "atom_0"

    def test_to_dict(self):
        d = RydbergAtom(1.5, 2.5, label="a").to_dict()
        assert d["x"] == 1.5
        assert d["y"] == 2.5
        assert d["label"] == "a"


# ── RydbergSimulator construction ─────────────────────────────────────────────

class TestRydbergSimulatorConstruction:
    def test_from_tuple_positions(self):
        sim = RydbergSimulator([(0.0, 0.0), (5.0, 0.0)])
        assert sim.n_atoms == 2

    def test_from_atom_objects(self):
        atoms = [RydbergAtom(0.0, 0.0), RydbergAtom(5.0, 0.0)]
        sim = RydbergSimulator(atoms)
        assert sim.n_atoms == 2

    def test_single_atom(self):
        sim = RydbergSimulator([(0.0, 0.0)])
        assert sim.n_atoms == 1

    def test_empty_positions_raises(self):
        with pytest.raises(RydbergError):
            RydbergSimulator([])

    def test_invalid_position_raises(self):
        with pytest.raises(RydbergError):
            RydbergSimulator(["not_a_position"])

    def test_too_many_atoms_raises(self):
        with pytest.raises(RydbergError):
            RydbergSimulator([(i, 0.0) for i in range(20)], max_atoms=10)

    def test_default_params(self):
        sim = RydbergSimulator([(0.0, 0.0)])
        assert sim.Omega == 1.0
        assert sim.delta == 0.0

    def test_explicit_params(self):
        sim = RydbergSimulator([(0.0, 0.0)], Omega=2.0, delta=1.0, C6=100.0)
        assert sim.Omega == 2.0
        assert sim.delta == 1.0
        assert sim.C6 == 100.0


# ── RydbergHamiltonian ────────────────────────────────────────────────────────

class TestRydbergHamiltonian:
    def test_hamiltonian_returns_object(self):
        sim = RydbergSimulator([(0.0, 0.0), (5.0, 0.0)])
        ham = sim.hamiltonian()
        assert isinstance(ham, RydbergHamiltonian)

    def test_n_atoms_stored(self):
        sim = RydbergSimulator([(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)])
        ham = sim.hamiltonian()
        assert ham.n_atoms == 3

    def test_interaction_matrix_shape(self):
        sim = RydbergSimulator([(0.0, 0.0), (5.0, 0.0)])
        ham = sim.hamiltonian()
        assert ham.interaction_matrix.shape == (2, 2)

    def test_interaction_symmetric(self):
        sim = RydbergSimulator([(0.0, 0.0), (5.0, 0.0)])
        V = sim.hamiltonian().interaction_matrix
        np.testing.assert_allclose(V, V.T)

    def test_interaction_diagonal_zero(self):
        sim = RydbergSimulator([(0.0, 0.0), (5.0, 0.0)])
        V = sim.hamiltonian().interaction_matrix
        assert V[0, 0] == 0.0
        assert V[1, 1] == 0.0

    def test_interaction_decays_with_distance(self):
        """V ∝ 1/r⁶; larger r → smaller V."""
        sim1 = RydbergSimulator([(0.0, 0.0), (3.0, 0.0)], C6=100.0)
        sim2 = RydbergSimulator([(0.0, 0.0), (6.0, 0.0)], C6=100.0)
        V1 = sim1.hamiltonian().interaction_matrix[0, 1]
        V2 = sim2.hamiltonian().interaction_matrix[0, 1]
        assert V1 > V2

    def test_blockade_radius(self):
        Omega = 2.0
        C6 = 64.0  # R_b = (64/2)^(1/6) = 32^(1/6) ≈ 1.78
        sim = RydbergSimulator([(0.0, 0.0)], Omega=Omega, C6=C6)
        R_b = sim.blockade_radius()
        expected = (C6 / Omega) ** (1 / 6)
        assert abs(R_b - expected) < 1e-10

    def test_to_dict(self):
        sim = RydbergSimulator([(0.0, 0.0), (5.0, 0.0)])
        d = sim.hamiltonian().to_dict()
        assert "n_atoms" in d
        assert "Omega" in d
        assert "interaction_matrix" in d


# ── Build matrix ──────────────────────────────────────────────────────────────

class TestBuildMatrix:
    def test_single_atom_matrix(self):
        """Single atom: H = (Ω/2)X − δ|1⟩⟨1|."""
        sim = RydbergSimulator([(0.0, 0.0)], Omega=1.0, delta=0.5)
        H = sim.build_matrix()
        assert H.shape == (2, 2)
        # H[0,1] = H[1,0] = Ω/2 = 0.5
        assert abs(H[0, 1] - 0.5) < 1e-10
        assert abs(H[1, 0] - 0.5) < 1e-10
        # H[1,1] = -δ = -0.5 (Rydberg state has energy -δ)
        assert abs(H[1, 1] + 0.5) < 1e-10
        assert abs(H[0, 0]) < 1e-10

    def test_two_atom_matrix_shape(self):
        sim = RydbergSimulator([(0.0, 0.0), (5.0, 0.0)])
        H = sim.build_matrix()
        assert H.shape == (4, 4)

    def test_matrix_is_hermitian(self):
        sim = RydbergSimulator([(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)])
        H = sim.build_matrix()
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_matrix_is_real_for_real_omega(self):
        """With real Ω and no complex phases, H should be real."""
        sim = RydbergSimulator([(0.0, 0.0), (5.0, 0.0)], Omega=1.0)
        H = sim.build_matrix()
        np.testing.assert_allclose(H.imag, 0.0, atol=1e-12)

    def test_two_distant_atoms_no_interaction(self):
        """Very distant atoms: interaction V≈0, block structure is ⊗."""
        C6 = 1.0
        r = 1000.0  # V = 1/r^6 ≈ 10^-18 ≈ 0
        sim = RydbergSimulator([(0.0, 0.0), (r, 0.0)], Omega=1.0, delta=0.0, C6=C6)
        H = sim.build_matrix()
        # H[3,3] = V_01 ≈ 0 (both atoms Rydberg: 11)
        assert abs(H[3, 3]) < 1e-10

    def test_blockade_energy_large_for_close_atoms(self):
        """V >> Ω for atoms within blockade radius → large diagonal for |11⟩."""
        sim = RydbergSimulator(
            [(0.0, 0.0), (0.1, 0.0)],  # very close atoms
            Omega=1.0, delta=0.0, C6=862690.0
        )
        H = sim.build_matrix()
        # H[3,3] = V_01 ≈ C6 / 0.1^6 >> 1
        assert H[3, 3].real > 100.0


# ── RydbergResult ─────────────────────────────────────────────────────────────

class TestRydbergResult:
    def _run(self, n_atoms=2, n_steps=20) -> RydbergResult:
        positions = [(i * 5.0, 0.0) for i in range(n_atoms)]
        sim = RydbergSimulator(positions, Omega=1.0, delta=0.0, C6=862690.0)
        return sim.evolve(t_final=2 * math.pi, n_steps=n_steps)

    def test_returns_rydberg_result(self):
        assert isinstance(self._run(), RydbergResult)

    def test_times_shape(self):
        result = self._run(n_steps=20)
        assert result.times.shape == (20,)

    def test_times_starts_at_zero(self):
        result = self._run()
        assert result.times[0] == 0.0

    def test_statevector_shape(self):
        result = self._run(n_atoms=2, n_steps=15)
        assert result.statevector.shape == (4, 15)

    def test_populations_shape(self):
        result = self._run(n_atoms=3, n_steps=10)
        assert result.populations.shape == (3, 10)

    def test_populations_in_range(self):
        result = self._run()
        assert np.all(result.populations >= -1e-10)
        assert np.all(result.populations <= 1.0 + 1e-10)

    def test_norm_conserved(self):
        """Statevector norm should remain 1 throughout evolution."""
        result = self._run()
        for t_idx in range(result.statevector.shape[1]):
            norm = np.linalg.norm(result.statevector[:, t_idx])
            assert abs(norm - 1.0) < 1e-8

    def test_total_rydberg_shape(self):
        result = self._run()
        assert result.total_rydberg.shape == result.times.shape

    def test_final_state(self):
        result = self._run(n_steps=10)
        fs = result.final_state()
        assert fs.shape == (4,)

    def test_rydberg_population_atom_0(self):
        result = self._run()
        pop = result.rydberg_population(0)
        assert pop.shape == result.times.shape

    def test_rydberg_population_out_of_range_raises(self):
        result = self._run(n_atoms=2)
        with pytest.raises(RydbergError):
            result.rydberg_population(5)

    def test_to_dict(self):
        result = self._run()
        d = result.to_dict()
        assert "times" in d
        assert "populations" in d
        assert "total_rydberg" in d
        assert "n_atoms" in d


# ── Time evolution physics ────────────────────────────────────────────────────

class TestEvolutionPhysics:
    def test_single_atom_rabi_oscillation(self):
        """Single atom without detuning: Rabi oscillation from |g⟩ to |r⟩."""
        Omega = 1.0
        sim = RydbergSimulator([(0.0, 0.0)], Omega=Omega, delta=0.0, C6=1.0)
        T_rabi = math.pi / Omega  # π-pulse time: g→r
        result = sim.evolve(t_final=T_rabi, n_steps=50)
        # At t=π/Ω, atom should be in |r⟩ with population ≈ 1
        pop_final = result.populations[0, -1]
        assert abs(pop_final - 1.0) < 0.05

    def test_single_atom_2pi_returns_to_ground(self):
        """At t=2π/Ω, atom returns to |g⟩."""
        Omega = 1.0
        T_2pi = 2 * math.pi / Omega
        sim = RydbergSimulator([(0.0, 0.0)], Omega=Omega, delta=0.0, C6=1.0)
        result = sim.evolve(t_final=T_2pi, n_steps=100)
        pop_final = result.populations[0, -1]
        assert abs(pop_final) < 0.05

    def test_zero_time_raises(self):
        sim = RydbergSimulator([(0.0, 0.0)])
        with pytest.raises(RydbergError):
            sim.evolve(t_final=0.0)

    def test_negative_time_raises(self):
        sim = RydbergSimulator([(0.0, 0.0)])
        with pytest.raises(RydbergError):
            sim.evolve(t_final=-1.0)

    def test_zero_steps_raises(self):
        sim = RydbergSimulator([(0.0, 0.0)])
        with pytest.raises(RydbergError):
            sim.evolve(t_final=1.0, n_steps=0)

    def test_custom_initial_state(self):
        """Evolve from |r⟩ (not |g⟩)."""
        sim = RydbergSimulator([(0.0, 0.0)], Omega=1.0)
        psi0 = np.array([0.0, 1.0], dtype=complex)  # |r⟩
        result = sim.evolve(t_final=0.01, n_steps=5, initial_state=psi0)
        assert abs(result.populations[0, 0] - 1.0) < 0.01

    def test_wrong_initial_state_shape_raises(self):
        sim = RydbergSimulator([(0.0, 0.0)])
        with pytest.raises(RydbergError):
            sim.evolve(t_final=1.0, initial_state=np.array([1.0, 0.0, 0.0]))

    def test_two_blockaded_atoms(self):
        """Very close atoms: Rydberg population suppressed due to blockade."""
        # Very strong blockade: atoms 0.1 μm apart
        sim = RydbergSimulator(
            [(0.0, 0.0), (0.1, 0.0)],
            Omega=1.0, delta=0.0, C6=862690.0
        )
        result = sim.evolve(t_final=5.0, n_steps=50)
        # Total Rydberg excitation should stay below 1 (blockade prevents double excitation)
        max_total = np.max(result.total_rydberg)
        assert max_total < 1.5  # at most ~1 excitation (W state)

    def test_distant_atoms_independent(self):
        """Very distant atoms: each oscillates independently."""
        r = 500.0  # V ≈ 0
        Omega = 1.0
        T_rabi = math.pi / Omega
        sim = RydbergSimulator(
            [(0.0, 0.0), (r, 0.0)],
            Omega=Omega, delta=0.0, C6=862690.0
        )
        result = sim.evolve(t_final=T_rabi, n_steps=60)
        # Both atoms should reach ~1 Rydberg population (total ≈ 2)
        total_final = result.total_rydberg[-1]
        assert abs(total_final - 2.0) < 0.1


# ── Factory methods ───────────────────────────────────────────────────────────

class TestFactoryMethods:
    def test_linear_chain(self):
        sim = RydbergSimulator.linear_chain(n_atoms=4, spacing=5.0)
        assert sim.n_atoms == 4
        # Check positions
        for i, atom in enumerate(sim.atoms):
            assert abs(atom.x - i * 5.0) < 1e-10
            assert abs(atom.y) < 1e-10

    def test_square_lattice(self):
        sim = RydbergSimulator.square_lattice(rows=2, cols=3, spacing=4.0)
        assert sim.n_atoms == 6

    def test_square_lattice_spacing(self):
        sim = RydbergSimulator.square_lattice(rows=2, cols=2, spacing=5.0)
        # atom (0,0) and (1,0) should be 5 μm apart
        d = sim.atoms[0].distance_to(sim.atoms[1])
        assert abs(d - 5.0) < 1e-10

    def test_factory_with_kwargs(self):
        sim = RydbergSimulator.linear_chain(3, spacing=6.0, Omega=2.0, delta=-1.0)
        assert sim.Omega == 2.0
        assert sim.delta == -1.0


# ── Blockade helpers ──────────────────────────────────────────────────────────

class TestBlockadeHelpers:
    def test_is_blockaded_close(self):
        # R_b = (C6/Omega)^(1/6); for C6=1, Omega=1: R_b=1
        sim = RydbergSimulator([(0.0, 0.0), (0.5, 0.0)], Omega=1.0, C6=1.0)
        assert sim.is_blockaded(0, 1)

    def test_is_not_blockaded_far(self):
        sim = RydbergSimulator([(0.0, 0.0), (10.0, 0.0)], Omega=1.0, C6=1.0)
        assert not sim.is_blockaded(0, 1)

    def test_nearest_neighbour_pairs_sorted(self):
        sim = RydbergSimulator.linear_chain(3, spacing=5.0)
        pairs = sim.nearest_neighbour_pairs()
        distances = [p[2] for p in pairs]
        assert distances == sorted(distances)

    def test_nearest_neighbour_pairs_count(self):
        # 3 atoms → 3 pairs
        sim = RydbergSimulator.linear_chain(3, spacing=5.0)
        pairs = sim.nearest_neighbour_pairs()
        assert len(pairs) == 3


# ── Integration ───────────────────────────────────────────────────────────────

class TestIntegration:
    def test_linear_chain_evolve(self):
        sim = RydbergSimulator.linear_chain(n_atoms=3, spacing=5.0, Omega=1.0)
        result = sim.evolve(t_final=2 * math.pi, n_steps=30)
        assert result.n_atoms == 3
        assert result.populations.shape == (3, 30)

    def test_blockade_fidelity_close_atoms(self):
        """Close atoms: blockade → ≤1 excitation → higher fidelity measure."""
        sim = RydbergSimulator(
            [(0.0, 0.0), (0.1, 0.0)], Omega=1.0, delta=0.0, C6=862690.0
        )
        result = sim.evolve(t_final=3.0, n_steps=20)
        fid = result.blockade_fidelity()
        assert 0.0 <= fid <= 1.0

    def test_square_lattice_evolve(self):
        sim = RydbergSimulator.square_lattice(rows=2, cols=2, spacing=5.0, Omega=1.0)
        result = sim.evolve(t_final=1.0, n_steps=10)
        assert result.n_atoms == 4
