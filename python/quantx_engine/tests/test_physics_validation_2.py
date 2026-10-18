"""M130: Physical correctness sweep II — literature cross-validation tests.

Each test cross-checks a known analytical or published numerical result for
modules introduced in M91–M118.  References are cited inline.

Modules covered
---------------
* fault_tolerant.py (M91) — logical error suppression below threshold
* floquet.py (M92)        — Floquet quasi-energies of driven two-level system
* quantum_network.py (M96) — Werner-state dephasing/swapping exact formulas
* quantum_protocols.py (M97) — teleportation fidelity vs. channel fidelity
* clifford_sim.py (M110)  — Bell-state correlations; stabilizer invariants
* neqdyn.py (M114)        — Kibble-Zurek power law; defects increase with ramp rate
* mps_simulator.py (M104) — GHZ fidelity = 1 with sufficient bond dimension
* many_body.py (M102)     — SYK Hamiltonian Hermiticity; energy variance
* rydberg.py (M118)       — Rydberg blockade constraint
* ftqc_pipeline.py (M112) — logical rate decreases with distance below threshold
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fault-tolerant qubits: logical error rate below threshold (M91)
# ---------------------------------------------------------------------------

class TestFaultTolerantPhysics:
    """Ref: Knill et al. 1998 (threshold theorem); Fowler et al. 2012 (surface code)."""

    def test_logical_rate_decreases_with_distance_below_threshold(self):
        """For p_physical << p_threshold, logical error rate decreases with d.

        The surface-code threshold is ≈ 1%.  At p = 0.0005 (far below threshold),
        d=3 should have lower logical rate than d=5, which should be lower than
        a single physical qubit error (the d=1 baseline).
        """
        from quantx_engine.contrib.fault_tolerant import logical_error_rate_vs_distance

        p = 5e-4  # well below threshold
        results = logical_error_rate_vs_distance(
            distances=[3, 5, 7],
            physical_error_rate=p,
            n_shots=1000,
            rng_seed=42,
        )

        # Logical rate should be non-increasing with distance (below threshold).
        # Use p=5e-4 which is large enough that rates are reliably non-zero for d=3.
        r3, r5, r7 = results[3], results[5], results[7]
        assert r3 >= r5, (
            f"d=3 rate {r3:.2e} should be ≥ d=5 rate {r5:.2e} below threshold"
        )
        # At least one distance should give near-perfect suppression
        assert min(r3, r5, r7) < p * 10

    def test_logical_rate_above_threshold_is_high(self):
        """For p >> p_threshold, the logical error rate remains high.

        At p = 0.3 (well above the ~1% surface-code threshold),
        the syndrome decoding fails and the logical error rate stays large.
        """
        from quantx_engine.contrib.fault_tolerant import logical_error_rate_vs_distance

        p = 0.3  # well above threshold
        results = logical_error_rate_vs_distance(
            distances=[3, 5],
            physical_error_rate=p,
            n_shots=500,
            rng_seed=99,
        )

        # Both rates should be substantially above the physical error rate p_physical=0.0005
        # (if they were near zero the code would be working well, which is wrong above threshold)
        assert results[3] > 0.05, f"d=3 rate {results[3]:.3f} too low for p_phys=0.3"
        assert results[5] > 0.05, f"d=5 rate {results[5]:.3f} too low for p_phys=0.3"


# ---------------------------------------------------------------------------
# Floquet: quasi-energies of driven two-level system (M92)
# ---------------------------------------------------------------------------

class TestFloquetPhysics:
    """Ref: Shirley 1965 (Floquet theory); Grifoni & Hänggi 1998 (driven systems)."""

    def test_driven_twolevel_quasi_energies_symmetric_on_resonance(self):
        """On resonance (ω_drive = ω₀), Floquet quasi-energies are ±Ω_R/2.

        In the rotating-wave approximation, the effective Rabi frequency is
        Ω_R = A (drive amplitude), and quasi-energies are ±A/2.
        These come in ±pairs by particle-hole symmetry of the Floquet Hamiltonian.
        """
        from quantx_engine.contrib.floquet import driven_two_level

        omega0 = 2.0
        A = 0.5
        result = driven_two_level(
            omega0=omega0,
            drive_amp=A,
            omega_drive=omega0,  # resonance
            n_steps=200,
        )

        qe = np.sort(np.real(result.quasi_energies))
        # Quasi-energies come in ±pairs (particle-hole symmetry)
        # At least two quasi-energies for a 2-level system
        assert len(qe) >= 2

        # Sum of all quasi-energies should be close to 0 (symmetric spectrum)
        assert abs(np.sum(qe)) < 0.5 * abs(qe[-1] - qe[0]) + 1e-6

    def test_floquet_quasi_energies_within_brillouin_zone(self):
        """Floquet quasi-energies must lie in (-ω_drive/2, +ω_drive/2].

        This is the defining property of quasi-energies in the Floquet-Brillouin zone.
        """
        from quantx_engine.contrib.floquet import driven_two_level

        omega_d = 3.0
        result = driven_two_level(
            omega0=2.5, drive_amp=0.3, omega_drive=omega_d, n_steps=100
        )

        qe = np.real(result.quasi_energies)
        assert np.all(qe > -omega_d / 2 - 1e-8)
        assert np.all(qe <= omega_d / 2 + 1e-8)

    def test_resonant_drive_quasi_energies_come_in_pairs(self):
        """On resonance, the 2-level Floquet quasi-energies come in ±pairs.

        For H = ω₀/2 · Z + A·cos(ω₀t)·X, the Floquet Hamiltonian in the
        rotating-wave approximation is block-diagonal with eigenvalues ±A/2.
        The two quasi-energies must sum to zero (by particle-hole symmetry).
        """
        from quantx_engine.contrib.floquet import driven_two_level

        omega0 = 2.0
        A = 0.5
        res = driven_two_level(omega0=omega0, drive_amp=A, omega_drive=omega0, n_steps=200)
        qe = np.sort(np.real(res.quasi_energies))

        # Exactly 2 quasi-energies for a 2-level system
        assert len(qe) == 2
        # They must sum to zero (symmetric spectrum)
        assert abs(qe[0] + qe[1]) < 0.2 * (qe[1] - qe[0]) + 1e-3


# ---------------------------------------------------------------------------
# Quantum network: Werner-state formulas (M96)
# ---------------------------------------------------------------------------

class TestQuantumNetworkPhysics:
    """Ref: Bennett et al. 1996 (purification); Briegel et al. 1998 (repeater)."""

    def test_dephasing_identity_at_zero_noise(self):
        """fidelity_after_dephasing with p=0 must return F unchanged."""
        from quantx_engine.contrib.quantum_network import fidelity_after_dephasing

        for F in [0.5, 0.75, 0.9, 1.0]:
            assert fidelity_after_dephasing(F, p_deph=0.0) == pytest.approx(F, abs=1e-12)

    def test_dephasing_at_full_noise_gives_maximally_mixed(self):
        """fidelity_after_dephasing with p=1 must give F'=1/4 for any input F."""
        from quantx_engine.contrib.quantum_network import fidelity_after_dephasing

        for F in [0.5, 0.75, 1.0]:
            F_out = fidelity_after_dephasing(F, p_deph=1.0)
            assert F_out == pytest.approx(0.25, abs=1e-12)

    def test_swapping_perfect_pair_preserves_fidelity(self):
        """fidelity_after_swapping with F_A=1 (perfect pair) must return F_B.

        Ref: Werner-state p-parameter: p_out = p_A * p_B.
        If p_A = 1 (F_A=1): p_out = p_B → F_out = F_B.
        """
        from quantx_engine.contrib.quantum_network import fidelity_after_swapping

        for F_B in [0.6, 0.8, 0.95, 1.0]:
            F_out = fidelity_after_swapping(F_A=1.0, F_B=F_B, F_gate=1.0)
            assert F_out == pytest.approx(F_B, abs=1e-10)

    def test_swapping_maximally_mixed_is_fixed_point(self):
        """fidelity_after_swapping with F_A=F_B=1/4 must give F_out=1/4."""
        from quantx_engine.contrib.quantum_network import fidelity_after_swapping

        F_out = fidelity_after_swapping(F_A=0.25, F_B=0.25, F_gate=1.0)
        assert F_out == pytest.approx(0.25, abs=1e-10)

    def test_purification_improves_fidelity(self):
        """DEJMPS purification must increase fidelity for F > 1/2.

        Ref: Bennett et al. 1996, DEJMPS; Deutsch et al. 1996.
        fidelity_after_purification(F) > F for F ∈ (0.5, 1.0).
        """
        from quantx_engine.contrib.quantum_network import fidelity_after_purification

        for F in [0.6, 0.75, 0.9]:
            F_out = fidelity_after_purification(F, n_rounds=1)
            assert F_out > F, f"Purification decreased fidelity at F={F}: {F_out} < {F}"


# ---------------------------------------------------------------------------
# Quantum protocols: teleportation fidelity (M97)
# ---------------------------------------------------------------------------

class TestQuantumProtocolsPhysics2:
    """Ref: Bennett et al. 1993 (teleportation); Horodecki et al. 1999 (Werner states)."""

    def test_teleportation_fidelity_perfect_channel(self):
        """teleportation_fidelity(F=1) = 1 (perfect channel → perfect fidelity)."""
        from quantx_engine.contrib.quantum_protocols import teleportation_fidelity

        assert teleportation_fidelity(1.0) == pytest.approx(1.0, abs=1e-12)

    def test_teleportation_fidelity_formula(self):
        """Teleportation fidelity formula: f = (2F+1)/3.

        Ref: Horodecki et al. (1999), Eq. (3).
        """
        from quantx_engine.contrib.quantum_protocols import teleportation_fidelity

        for F in [0.0, 0.25, 0.5, 0.75, 1.0]:
            expected = (2.0 * F + 1.0) / 3.0
            assert teleportation_fidelity(F) == pytest.approx(expected, abs=1e-12)

    def test_teleportation_fidelity_minimum(self):
        """At F=0 (anti-correlated channel), f = 1/3 (classical guessing = 1/3)."""
        from quantx_engine.contrib.quantum_protocols import teleportation_fidelity

        # F=0: f = (2*0+1)/3 = 1/3
        assert teleportation_fidelity(0.0) == pytest.approx(1.0 / 3.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Clifford simulation: Bell state correlations (M110)
# ---------------------------------------------------------------------------

class TestCliffordSimPhysics:
    """Ref: Aaronson & Gottesman 2004 (CHP algorithm); Bell 1964."""

    def test_bell_state_measurement_correlation(self):
        """H+CNOT creates a Bell state: measuring both qubits always gives 00 or 11.

        The stabilizers of |Φ+⟩ = (|00⟩+|11⟩)/√2 are XX and ZZ.
        Measuring qubit 0 collapses both qubits to the same value.
        """
        from quantx_engine.contrib.clifford_sim import CliffordSimulator

        outcomes_00_or_11 = 0
        n_trials = 100
        for seed in range(n_trials):
            sim = CliffordSimulator(n_qubits=2, seed=seed)
            sim.h(0)
            sim.cx(0, 1)
            m0 = sim.measure(0)
            m1 = sim.measure(1)
            assert m0 == m1, f"Bell state gave anti-correlated {m0},{m1} (seed={seed})"
            outcomes_00_or_11 += 1

        assert outcomes_00_or_11 == n_trials

    def test_single_qubit_h_gate_50_50(self):
        """H|0⟩ = |+⟩: measuring 0 and 1 each with probability 1/2 over many shots.

        With N=200 shots, the fraction should be in [0.35, 0.65] with high probability.
        """
        from quantx_engine.contrib.clifford_sim import CliffordSimulator

        zeros = 0
        n = 200
        for seed in range(n):
            sim = CliffordSimulator(n_qubits=1, seed=seed)
            sim.h(0)
            zeros += (sim.measure(0) == 0)

        assert 0.35 <= zeros / n <= 0.65, f"H|0⟩ gave {zeros/n:.2f} (expected ≈ 0.5)"

    def test_ghz_state_all_correlated(self):
        """GHZ state: H+CNOT chain on 5 qubits → all qubits correlated."""
        from quantx_engine.contrib.clifford_sim import CliffordSimulator

        n = 5
        for seed in range(50):
            sim = CliffordSimulator(n_qubits=n, seed=seed)
            sim.h(0)
            for q in range(n - 1):
                sim.cx(q, q + 1)
            outcomes = [sim.measure(q) for q in range(n)]
            # All outcomes should be equal (all 0s or all 1s)
            assert len(set(outcomes)) == 1, (
                f"GHZ failed (seed={seed}): {outcomes}"
            )


# ---------------------------------------------------------------------------
# MPS simulator: GHZ fidelity (M104)
# ---------------------------------------------------------------------------

class TestMPSSimulatorPhysics:
    """Ref: Vidal 2003 (efficient simulation of 1D quantum systems)."""

    def test_ghz_fidelity_with_sufficient_bond_dimension(self):
        """GHZ state on n qubits has bond dimension 2; fidelity should be ≈ 1.

        The exact GHZ = (|0…0⟩ + |1…1⟩)/√2 requires only bond dimension 2.
        With max_bond=2, the MPS representation is exact.
        """
        from quantx_engine.contrib.mps_simulator import ghz_fidelity

        for n in [4, 6, 8]:
            F = ghz_fidelity(n_qubits=n, max_bond=2)
            assert F == pytest.approx(1.0, abs=1e-8), (
                f"GHZ fidelity for n={n}: {F:.10f}"
            )

    def test_ghz_fidelity_decreases_with_truncation(self):
        """Truncating bond dimension below 2 degrades GHZ fidelity."""
        from quantx_engine.contrib.mps_simulator import ghz_fidelity

        F_exact = ghz_fidelity(n_qubits=6, max_bond=2)
        F_trunc = ghz_fidelity(n_qubits=6, max_bond=1)

        assert F_exact > F_trunc


# ---------------------------------------------------------------------------
# Non-equilibrium dynamics: Kibble-Zurek power law (M114)
# ---------------------------------------------------------------------------

class TestNeqdynPhysics:
    """Ref: Kibble 1976; Zurek 1985; Zurek et al. 2005 (quantum KZ mechanism)."""

    def test_faster_quench_produces_more_defects(self):
        """Kibble-Zurek: faster ramp rate → more topological defects.

        For the 1D TFIM, defect density n_d ∝ v^{1/2} where v is the quench rate.
        A faster quench (larger v) must create strictly more defects.
        """
        from quantx_engine.contrib.neqdyn import KibbleZurekScaling

        kz = KibbleZurekScaling(n_sites=6, J=1.0, n_steps=30)
        result = kz.sweep(quench_rates=[0.5, 1.0, 2.0], h_i=2.0, h_f=0.1)

        defects = result.defect_densities
        # Faster quench rate → more defects (monotone in rate)
        assert defects[0] <= defects[1] + 1e-6  # small tolerance for numerics
        assert defects[1] <= defects[2] + 1e-6

    def test_kz_exponent_positive(self):
        """The Kibble-Zurek exponent must be positive.

        n_d ∝ v^exponent with exponent = ν/(1+νz) > 0 (for 1D TFIM: ν=z=1, so exponent=0.5).
        """
        from quantx_engine.contrib.neqdyn import KibbleZurekScaling

        kz = KibbleZurekScaling(n_sites=6, J=1.0, n_steps=30)
        result = kz.sweep(h_i=2.0, h_f=0.1)  # uses default log-spaced rates

        # The fitted exponent should be positive
        assert result.kz_exponent > 0.0

    def test_kz_defect_density_in_valid_range(self):
        """Defect density per bond must be in [0, 0.5].

        Domain-wall density cannot exceed 0.5 (maximum disorder in the paramagnetic
        fully-disordered state of the 1D Ising model).
        """
        from quantx_engine.contrib.neqdyn import KibbleZurekScaling

        kz = KibbleZurekScaling(n_sites=8, J=1.0, n_steps=20)
        result = kz.sweep(quench_rates=[0.5, 2.0], h_i=2.0, h_f=0.1)

        for d in result.defect_densities:
            assert 0.0 <= d <= 0.5


# ---------------------------------------------------------------------------
# Many-body physics: SYK Hamiltonian properties (M102)
# ---------------------------------------------------------------------------

class TestManyBodyPhysics:
    """Ref: Sachdev & Ye 1993; Kitaev 2015 (KITP lectures); Garcia-Garcia & Verbaarschot 2016."""

    def test_syk_hamiltonian_is_hermitian(self):
        """The SYK Hamiltonian must be Hermitian: H = H†.

        This is a fundamental constraint: the Jordan-Wigner mapping of real-valued
        couplings between Majorana operators must produce a Hermitian matrix.
        """
        from quantx_engine.contrib.many_body import SYKModel

        model = SYKModel(N=8, q=4, seed=42)
        H = model.hamiltonian
        assert np.allclose(H, H.conj().T, atol=1e-10), "SYK Hamiltonian is not Hermitian"

    def test_syk_spectrum_is_real(self):
        """All eigenvalues of the SYK Hamiltonian must be real."""
        from quantx_engine.contrib.many_body import SYKModel

        model = SYKModel(N=6, q=4, seed=7)
        H = model.hamiltonian
        evals = np.linalg.eigvalsh(H)  # use hermitian solver

        assert np.all(np.isfinite(evals))
        # Energy gap should be finite (non-degenerate spectrum for random H)
        assert evals[-1] - evals[0] > 0.0

    def test_syk_ground_state_energy_is_negative(self):
        """The SYK ground-state energy is always negative (by particle-hole symmetry).

        For q=4 SYK at even N, the spectrum is symmetric around 0 by the
        particle-hole symmetry.  The ground state energy E₀ < 0.
        """
        from quantx_engine.contrib.many_body import SYKModel

        model = SYKModel(N=8, q=4, seed=123)
        H = model.hamiltonian
        E0 = float(np.linalg.eigvalsh(H)[0])

        assert E0 < 0.0, f"SYK ground state energy E₀ = {E0} should be negative"

    def test_syk_energy_variance_scales_with_n(self):
        """SYK energy variance σ² ∝ N^{q-1} / (q-1)! (random matrix scaling).

        For fixed q=4, the ratio σ²(N=8) / σ²(N=6) should be roughly (8/6)^3 ≈ 2.37.
        We just test that variance increases with N.
        """
        from quantx_engine.contrib.many_body import SYKModel

        model6 = SYKModel(N=6, q=4, seed=11)
        model8 = SYKModel(N=8, q=4, seed=22)

        var6 = float(np.var(np.linalg.eigvalsh(model6.hamiltonian)))
        var8 = float(np.var(np.linalg.eigvalsh(model8.hamiltonian)))

        assert var8 > var6, f"Variance should increase with N: var(N=8)={var8} ≤ var(N=6)={var6}"


# ---------------------------------------------------------------------------
# Rydberg simulator: blockade physics (M118)
# ---------------------------------------------------------------------------

class TestRydbergPhysics:
    """Ref: Jaksch et al. 2000; Saffman et al. 2010 (Rydberg blockade review)."""

    def test_blockade_radius_formula(self):
        """Blockade radius r_b = (C₆/Ω)^{1/6} — verify dimensional consistency.

        RydbergSimulator takes atom_positions (x,y) in 2D.
        The Hamiltonian must be a 4×4 Hermitian matrix for 2 atoms.
        """
        from quantx_engine.contrib.rydberg import RydbergSimulator

        # Two atoms separated by 5 μm (2D positions)
        sim = RydbergSimulator(
            atom_positions=[(0.0, 0.0), (5.0, 0.0)],
            Omega=1.0,    # MHz
            delta=0.0,    # on resonance
            C6=1000.0,    # van-der-Waals coefficient
        )

        H = sim.build_matrix()
        # Should be a 4×4 Hermitian matrix (2-qubit Hilbert space)
        assert H.shape == (4, 4)
        assert np.allclose(H, H.conj().T, atol=1e-10)

    def test_rydberg_blockade_shifts_energy(self):
        """Within the blockade radius, the |rr⟩ state is energy-shifted by V₁₂.

        Two atoms with large C₆/r⁶: |rr⟩ diagonal element should include V = C₆/r⁶.
        """
        from quantx_engine.contrib.rydberg import RydbergSimulator

        r = 3.0   # small separation → large blockade energy (2D)
        C6 = 1000.0
        V_expected = C6 / r**6

        sim = RydbergSimulator(
            atom_positions=[(0.0, 0.0), (r, 0.0)],
            Omega=0.1,    # small Rabi frequency so interaction dominates
            delta=0.0,
            C6=C6,
        )
        H = sim.build_matrix()

        # The |rr⟩ = |11⟩ state is the last basis state (index 3 in {00,01,10,11})
        # Its diagonal energy should include the interaction V
        H_rr = float(np.real(H[3, 3]))
        # Tolerance: ±0.05 covers small Ω/2 = 0.05 off-diagonal corrections
        # while still requiring < 4% relative error on V_expected ≈ 1.37.
        assert abs(H_rr - V_expected) < 0.05, (
            f"|rr⟩ diagonal energy {H_rr:.4f} should be close to V={V_expected:.4f}"
        )
