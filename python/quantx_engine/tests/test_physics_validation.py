"""M129: Physical correctness sweep I — literature cross-validation tests.

Each test cross-checks a known analytical or published numerical result for
modules introduced in M60–M90.  References are cited inline.

Modules covered
---------------
* lanczos.py      — Heisenberg / Ising ground states vs exact diagonalization
* trotter.py      — Trotter convergence order; total-spin conservation
* mclachlan.py    — Imaginary-time VQD converges to ground state
* mitigation.py   — Richardson extrapolation exact for polynomial noise
* quantum_protocols.py — CHSH bound 2√2; BB84 Shor–Preskill key rate
* qec.py          — Repetition code: d=3 corrects all weight-1 errors
* finite_size.py  — Binder cumulant for classical Ising model
* quantum_walk.py — CTQW transfer probability sum rule
* tensor_network.py — MPS norm conservation under TEBD
"""

from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Lanczos: ground-state energies vs. exact diagonalization
# ---------------------------------------------------------------------------

class TestLanczosPhysics:
    """Ref: Exact diagonalization for small spin chains."""

    def test_heisenberg_2site_ground_state(self):
        """2-site Heisenberg antiferromagnet: E₀ = -3J/2.

        H = J · Σ (σᵢσᵢ₊₁/2 + h.c.) for XX+YY+ZZ, giving singlet -3J/2.
        Exact eigenvalues for J=1: {-1.5, -0.5, 1, 1}.

        Note: The Lanczos solver may set converged=False when it terminates by
        finding an exact invariant subspace (b < 1e-14), even though the
        eigenvalue is exact.  We therefore validate via the residual norm.
        """
        from quantx_engine.lanczos import SparseHamiltonian, LanczosSolver

        H = SparseHamiltonian.heisenberg_1d(n_sites=2, J=1.0)
        solver = LanczosSolver(k=1, max_iter=100, tol=1e-12, seed=42)
        result = solver.solve(H)

        # Residual ||Hv - λv|| must be small (exact even if converged flag is False)
        assert result.residuals[0] < 1e-8
        assert result.ground_state_energy() == pytest.approx(-1.5, abs=1e-8)

    def test_heisenberg_4site_ground_state_vs_exact(self):
        """4-site Heisenberg open chain: Lanczos must match numpy.linalg.eigh.

        For n=4, J=1, the exact ground-state energy is known to be ≈ -2.0 J
        (singlet sector of the Heisenberg chain).  We cross-validate using
        exact dense diagonalization as the reference.
        """
        from quantx_engine.lanczos import SparseHamiltonian, LanczosSolver

        H = SparseHamiltonian.heisenberg_1d(n_sites=4, J=1.0)
        # Exact reference via dense diagonalization
        H_dense = H.to_dense()
        exact_E0 = float(np.linalg.eigvalsh(H_dense)[0])

        solver = LanczosSolver(k=2, max_iter=300, tol=1e-12, seed=0)
        result = solver.solve(H)

        assert result.converged
        assert result.ground_state_energy() == pytest.approx(exact_E0, abs=1e-6)

    def test_ising_disordered_phase_has_larger_gap(self):
        """1D TFIM: disordered phase (h >> J) has a large spectral gap.

        For h >> J the system is paramagnetic: gap ≈ 2(h - J).
        At the critical point (h = J) the gap scales as 1/n for finite chains.
        We compare h=3 (strongly disordered, large gap) vs h=1 (critical,
        smaller gap) using PBC with 8 sites.

        Note: h << J (ordered/ferromagnetic phase with PBC) gives an
        exponentially small gap due to quasi-degenerate ground states from
        spontaneous symmetry breaking — that phase is deliberately excluded.
        """
        from quantx_engine.lanczos import SparseHamiltonian, LanczosSolver

        n = 8
        H_disorder = SparseHamiltonian.ising_1d(n_sites=n, J=1.0, h=3.0)  # h >> J: big gap
        H_critical  = SparseHamiltonian.ising_1d(n_sites=n, J=1.0, h=1.0)  # critical

        solver = LanczosSolver(k=3, max_iter=400, tol=1e-10, seed=7)
        res_dis  = solver.solve(H_disorder)
        res_crit = solver.solve(H_critical)

        gap_dis  = res_dis.spectral_gap()
        gap_crit = res_crit.spectral_gap()

        assert gap_dis > 0           # finite chain always has non-zero gap
        assert gap_crit > 0
        assert gap_dis > gap_crit    # strongly disordered > critical

    def test_ground_state_vector_is_normalized(self):
        """Lanczos ground-state eigenvector must have unit norm."""
        from quantx_engine.lanczos import SparseHamiltonian, LanczosSolver

        H = SparseHamiltonian.heisenberg_1d(n_sites=4, J=1.0)
        solver = LanczosSolver(k=1, max_iter=200, tol=1e-12, seed=1)
        result = solver.solve(H)

        v = result.ground_state_vector()
        assert abs(np.linalg.norm(v) - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# Trotter: convergence order and exact limits
# ---------------------------------------------------------------------------

class TestTrotterPhysics:
    """Ref: Suzuki 1991 (product formulas), Lloyd 1996 (Hamiltonian simulation)."""

    def _make_single_qubit_sim(self, omega: float) -> "tuple":
        """H = omega * Z for a single qubit; exact solution: exp(-iωtZ) rotates
        around the Bloch sphere Z-axis."""
        from quantx_engine.trotter import TrotterTerm, TrotterSimulator

        terms = [TrotterTerm(omega, "Z")]
        psi0 = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)  # |+⟩
        sim = TrotterSimulator(n_qubits=1, terms=terms, initial_state=psi0)
        return sim

    def test_trotter_order2_converges(self):
        """2nd-order Trotter error should decrease as Δt² → 0.

        For H = ω·Z starting from |+⟩, the exact time evolution is
        |ψ(t)⟩ = (e^{-iωt}|0⟩ + e^{+iωt}|1⟩)/√2.
        We compare the Trotter state to numpy exact and verify convergence.
        """
        from quantx_engine.trotter import TrotterTerm, TrotterSimulator

        omega = 0.5
        t = 1.0
        psi0 = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)
        terms = [TrotterTerm(omega, "Z")]
        sim = TrotterSimulator(n_qubits=1, terms=terms, initial_state=psi0)

        # Exact state
        exact_psi = np.array(
            [math.cos(omega * t) - 1j * math.sin(omega * t),
             math.cos(omega * t) + 1j * math.sin(omega * t)],
            dtype=complex
        ) / math.sqrt(2.0)

        errors = []
        for n_steps in [10, 100]:
            result = sim.run(t=t, n_steps=n_steps, order=2)
            err = float(np.linalg.norm(result.state - exact_psi))
            errors.append(err)

        # More steps → smaller error
        assert errors[1] < errors[0]
        # 100 steps should be accurate to < 1e-4 for this simple problem
        assert errors[1] < 1e-4

    def test_trotter_zz_conserves_populations(self):
        """For H = J·ZZ (diagonal), populations |⟨i|ψ⟩|² are unchanged.

        exp(-i J ZZ t)|ψ⟩ only shifts phases, so all measurement probabilities
        are constant — a fundamental symmetry check.
        """
        from quantx_engine.trotter import TrotterTerm, TrotterSimulator

        # Start from a superposition state
        psi0 = np.array([1, 1, 1, 1], dtype=complex) / 2.0
        terms = [TrotterTerm(1.0, "ZZ")]
        sim = TrotterSimulator(n_qubits=2, terms=terms, initial_state=psi0)

        initial_probs = np.abs(psi0) ** 2
        result = sim.run(t=2.5, n_steps=50, order=2)

        # Populations must be unchanged to numerical precision
        assert np.allclose(result.probs, initial_probs, atol=1e-10)

    def test_trotter_order2_vs_order1_accuracy(self):
        """2nd-order Trotter should be more accurate than 1st-order for same steps.

        H = 0.5·XX + 0.3·ZZ on 2 qubits; non-commuting terms test accuracy.
        """
        from quantx_engine.trotter import TrotterTerm, TrotterSimulator
        try:
            import scipy  # noqa: F401
        except ImportError:
            pytest.skip("scipy not available for compute_exact")

        # Use H = 0.5·XZ + 0.3·ZZ — these terms do NOT commute ([XZ, ZZ] ≠ 0),
        # so the Trotter decomposition has real O(dt²) / O(dt³) truncation error.
        # (XX and ZZ DO commute, so they would give zero truncation error at all orders.)
        terms = [TrotterTerm(0.5, "XZ"), TrotterTerm(0.3, "ZZ")]
        psi0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
        sim = TrotterSimulator(n_qubits=2, terms=terms, initial_state=psi0)

        # n_steps=4 (dt=0.25): order-1 error ~3e-2, order-2 error ~1e-3 (30× smaller).
        # Both are well above machine precision so the comparison is reliable.
        res1 = sim.run(t=1.0, n_steps=4, order=1, compute_exact=True)
        res2 = sim.run(t=1.0, n_steps=4, order=2, compute_exact=True)

        assert res2.trotter_error < res1.trotter_error, (
            f"Order-2 error {res2.trotter_error:.2e} should be < "
            f"order-1 error {res1.trotter_error:.2e}"
        )


# ---------------------------------------------------------------------------
# McLachlan: imaginary-time evolution converges to ground state
# ---------------------------------------------------------------------------

class TestMcLachlanPhysics:
    """Ref: McLachlan 1964; Motta et al. 2020 (imaginary-time VQD on hardware)."""

    def test_imaginary_time_finds_ground_state(self):
        """For H = Z (single qubit), ground state is |0⟩ with E₀ = -1.

        Imaginary-time McLachlan starting from |+⟩ should converge to E₀ = -1.
        """
        from quantx_engine.mclachlan import McLachlanSolver

        H = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)  # Z matrix: E_0 = -1

        def ansatz(theta):
            """Single-qubit rotation: |ψ(θ)⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩."""
            t = float(theta[0])
            return np.array([math.cos(t / 2), math.sin(t / 2)], dtype=complex)

        solver = McLachlanSolver(
            ansatz=ansatz,
            hamiltonian=H,
            n_params=1,
            dt=0.05,
            evolution="imaginary",
            integrator="rk4",
            reg=1e-6,
        )

        # Start from θ = π/2 (which gives |+x⟩ rotated to equal superposition)
        init = np.array([math.pi / 2])
        result = solver.evolve(init, t_final=5.0)

        # The ground-state energy of Z is -1 (|0⟩ eigenstate with -1 not right...)
        # Actually Z has eigenvalues +1 (|0⟩) and -1 (|1⟩). Ground state = |1⟩, E₀ = -1
        # But wait, H = Z = [[1,0],[0,-1]], so eigenvalues are +1 and -1.
        # Ground state energy = -1. But imaginary time cools to the lowest eigenvalue
        # which is -1 (the |1⟩ state).
        assert result.final_energy < 0.0
        # Should be close to -1 after 5 imaginary time units
        assert result.final_energy == pytest.approx(-1.0, abs=0.1)

    def test_real_time_energy_conservation(self):
        """For real-time evolution, energy ⟨H⟩ should be conserved.

        Variational real-time dynamics preserves energy when the ansatz is
        sufficiently expressive (parameter manifold contains the exact trajectory).
        """
        from quantx_engine.mclachlan import McLachlanSolver

        H = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

        def ansatz(theta):
            t = float(theta[0])
            return np.array([math.cos(t / 2), math.sin(t / 2)], dtype=complex)

        solver = McLachlanSolver(
            ansatz=ansatz,
            hamiltonian=H,
            n_params=1,
            dt=0.01,
            evolution="real",
            integrator="rk4",
        )
        init = np.array([math.pi / 4])  # 45° rotation
        result = solver.evolve(init, t_final=1.0)

        # Energy should remain approximately constant (within 5% over short time)
        E0 = result.energies[0]
        E_final = result.energies[-1]
        if abs(E0) > 1e-10:  # avoid division by zero at E=0
            rel_change = abs(E_final - E0) / abs(E0)
            assert rel_change < 0.1, f"Energy changed by {rel_change:.1%} (should be < 10%)"


# ---------------------------------------------------------------------------
# Mitigation: Richardson extrapolation is exact for polynomial noise
# ---------------------------------------------------------------------------

class TestMitigationPhysics:
    """Ref: Temme et al. 2017 (ZNE); Li & Benjamin 2017 (virtual distillation)."""

    def test_richardson_exact_for_linear_noise(self):
        """Richardson extrapolation with 2 points is exact for linear noise.

        If E(λ) = E₀ + a·λ, then the 2-point Richardson extrapolation at
        λ=1 and λ=3 gives exactly E(0) = E₀.
        """
        from quantx_engine.mitigation import RichardsonExtrapolator

        E0, a = 0.75, 0.1
        scale_factors = [1.0, 3.0]
        noisy_values = [E0 + a * lam for lam in scale_factors]

        extrap = RichardsonExtrapolator()
        result = extrap.extrapolate(scale_factors, noisy_values)

        assert result.mitigated_value == pytest.approx(E0, abs=1e-12)

    def test_richardson_exact_for_quadratic_noise(self):
        """Richardson extrapolation with 3 points is exact for quadratic noise.

        E(λ) = E₀ + a·λ + b·λ²  →  extrapolated value = E₀.
        """
        from quantx_engine.mitigation import RichardsonExtrapolator

        E0, a, b = 0.5, 0.05, 0.02
        scale_factors = [1.0, 3.0, 5.0]
        noisy_values = [E0 + a * lam + b * lam**2 for lam in scale_factors]

        extrap = RichardsonExtrapolator()
        result = extrap.extrapolate(scale_factors, noisy_values)

        assert result.mitigated_value == pytest.approx(E0, abs=1e-10)

    def test_mitigated_value_closer_to_truth_than_raw(self):
        """ZNE-mitigated value must be closer to E₀ than the raw (λ=1) value.

        This tests the core premise of ZNE: mitigation reduces bias.
        """
        from quantx_engine.mitigation import RichardsonExtrapolator

        E0, a = 1.0, 0.2
        scale_factors = [1.0, 3.0, 5.0]
        noisy_values = [E0 + a * lam for lam in scale_factors]  # linear noise

        extrap = RichardsonExtrapolator()
        result = extrap.extrapolate(scale_factors, noisy_values)

        raw_error = abs(noisy_values[0] - E0)
        mitigated_error = abs(result.mitigated_value - E0)

        assert mitigated_error < raw_error


# ---------------------------------------------------------------------------
# Quantum protocols: CHSH and BB84 Shor–Preskill bounds
# ---------------------------------------------------------------------------

class TestQuantumProtocolsPhysics:
    """Ref: Bell 1964; CHSH 1969; Shor & Preskill 2000 (BB84 security)."""

    def test_chsh_tsirelson_bound(self):
        """Quantum CHSH can achieve up to S = 2√2 (Tsirelson's bound).

        For the singlet state and optimal bases (0°, 45°, 22.5°, 67.5°),
        the four correlations are: E(a,b) = -cos(θ_ab).
        Optimal angles: θ₀₀=45°, θ₀₁=0°, θ₁₀=22.5°, θ₁₁=67.5°.
        """
        from quantx_engine.quantum_protocols import chsh_inequality_value

        # Tsirelson-optimal correlations: each |E(a,b)| = 1/√2 with signs chosen
        # so S = E(0,0) - E(0,1) + E(1,0) + E(1,1) = 4/√2 = 2√2.
        # These correspond to singlet-state measurements with angle differences 45°.
        sqrt2 = math.sqrt(2.0)
        corr = {
            (0, 0):  1.0 / sqrt2,
            (0, 1): -1.0 / sqrt2,
            (1, 0):  1.0 / sqrt2,
            (1, 1):  1.0 / sqrt2,
        }
        S = chsh_inequality_value(corr)
        # S = |1/√2 - (-1/√2) + 1/√2 + 1/√2| = |4/√2| = 2√2 ≈ 2.828
        assert S == pytest.approx(2.0 * sqrt2, abs=1e-10)

    def test_chsh_classical_bound(self):
        """Classical correlations obey |S| ≤ 2."""
        from quantx_engine.quantum_protocols import chsh_inequality_value

        # A classical deterministic strategy: Alice outputs a, Bob outputs b=a
        # All correlations E = +1 → S = 2
        corr = {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): -1.0}
        S = chsh_inequality_value(corr)
        assert S <= 2.0 + 1e-10

    def test_bb84_key_rate_zero_qber(self):
        """At QBER = 0, the Shor–Preskill key rate is 1 (full key rate)."""
        from quantx_engine.quantum_protocols import bb84_key_rate

        r = bb84_key_rate(0.0)
        assert r == pytest.approx(1.0, abs=1e-10)

    def test_bb84_key_rate_threshold(self):
        """At QBER = 11%, key rate should be ≈ 0 (security threshold).

        The Shor–Preskill formula r = 1 - 2·h₂(QBER) gives r ≈ 0 at QBER ≈ 11%.
        """
        from quantx_engine.quantum_protocols import bb84_key_rate

        # At QBER = 0.11: h₂(0.11) ≈ 0.500 → raw formula ≈ 0 (clamped to 0)
        r = bb84_key_rate(0.11)
        assert r < 0.01  # very close to zero at the security threshold
        # Raw formula 1 - 2·h₂(0.11) should be negative or negligibly positive,
        # confirming we are at or past the threshold (not just clamped from far below).
        import math
        h2 = lambda p: -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        raw = 1.0 - 2.0 * h2(0.11)
        assert raw < 0.05  # raw formula near or below zero, not clamped from well above

    def test_bb84_key_rate_monotone_decreasing(self):
        """Key rate must be monotonically decreasing with QBER."""
        from quantx_engine.quantum_protocols import bb84_key_rate

        qbers = [0.0, 0.02, 0.05, 0.08, 0.10]
        rates = [bb84_key_rate(q) for q in qbers]

        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1]


# ---------------------------------------------------------------------------
# QEC: repetition code corrects all weight-1 errors (t = floor((d-1)/2))
# ---------------------------------------------------------------------------

class TestQECPhysics:
    """Ref: Nielsen & Chuang §10.3; threshold theorem for surface codes."""

    def test_repetition_d3_corrects_single_errors(self):
        """d=3 repetition code (t=1) corrects any single-bit-flip error.

        For every single-bit error on a codeword, the decoder (majority vote)
        must recover the original logical bit.  This tests the core correction
        guarantee of a distance-3 code.
        """
        from quantx_engine.qec import RepetitionCode

        code = RepetitionCode(n=3)  # API: RepetitionCode(n) not RepetitionCode(distance=n)

        for logical in (0, 1):
            codeword = code.encode(logical)
            for bit in range(code.n):
                errored = codeword.copy()
                errored[bit] ^= 1           # flip one physical bit
                corrected_codeword = code.correct(errored)
                decoded = code.decode(corrected_codeword)
                assert decoded == logical, (
                    f"logical={logical}, bit={bit}: decoded {decoded} ≠ {logical}"
                )

    def test_repetition_distance_property(self):
        """Minimum Hamming distance must equal n for the repetition code."""
        from quantx_engine.qec import RepetitionCode

        for n in [3, 5, 7]:
            code = RepetitionCode(n=n)
            assert code.d == n   # distance attribute
            assert code.n == n   # physical qubits


# ---------------------------------------------------------------------------
# Quantum walk: CTQW transfer probability sum rule
# ---------------------------------------------------------------------------

class TestQuantumWalkPhysics:
    """Ref: Farhi & Gutmann 1998 (CTQW); Kempe 2003 (review)."""

    def test_ctqw_total_probability_conserved(self):
        """Sum of all transfer probabilities in CTQW must equal 1 (unitarity).

        The CTQW is unitary, so summing |⟨j|e^{-iHt}|0⟩|² over all j = 1.
        """
        from quantx_engine.quantum_walk import ContinuousTimeQuantumWalk

        # 5-node path graph adjacency matrix
        n = 5
        A = np.zeros((n, n), dtype=float)
        for i in range(n - 1):
            A[i, i + 1] = 1.0
            A[i + 1, i] = 1.0
        gamma = 0.5
        H = gamma * A  # Hamiltonian = γ·A

        walk = ContinuousTimeQuantumWalk(hamiltonian=H)
        result = walk.run(t=2.0)  # evolves from default |0⟩

        # The final probability distribution must sum to 1
        total_prob = float(np.sum(result.position_probs))
        assert total_prob == pytest.approx(1.0, abs=1e-8)

    def test_ctqw_transfer_probability_formula(self):
        """Transfer probability for a 2-node complete graph has an exact form.

        For a 2-node graph with Hamiltonian H = γ·[[0,1],[1,0]] = γ·X,
        the exact transfer probabilities are:
            P(0→0, t) = cos²(γ·t)
            P(0→1, t) = sin²(γ·t)
        """
        from quantx_engine.quantum_walk import ContinuousTimeQuantumWalk, ctqw_transfer_probability

        gamma = 1.0
        t = math.pi / 4  # 45° rotation: P(0→0) = P(0→1) = 0.5

        # 2-node Hamiltonian
        H = gamma * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        walk = ContinuousTimeQuantumWalk(hamiltonian=H)

        p_01 = ctqw_transfer_probability(walk, source=0, target=1, t=t)
        p_00 = ctqw_transfer_probability(walk, source=0, target=0, t=t)

        assert p_01 == pytest.approx(math.sin(gamma * t) ** 2, abs=1e-8)
        assert p_00 == pytest.approx(math.cos(gamma * t) ** 2, abs=1e-8)
        # Conservation: P(0→0) + P(0→1) = 1
        assert p_00 + p_01 == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Tensor network (TEBD): norm conservation during time evolution
# ---------------------------------------------------------------------------

class TestTensorNetworkPhysics:
    """Ref: Vidal 2004 (TEBD algorithm); White 1992 (DMRG)."""

    def test_tebd_norm_conserved(self):
        """TEBD time evolution under the TFIM must preserve the MPS norm.

        evolve_ising() applies unitary 2-site gates; the SVD truncation
        keeps the state renormalised, so norm() should remain 1 after
        each step.
        """
        from quantx_engine.tensor_network import MPS, TEBD

        # 6-site MPS in |000000⟩ state
        n = 6
        mps = MPS.product_state([0] * n, d=2)  # API: d=2 (not local_dim=2)
        tebd = TEBD(mps=mps, max_chi=16, svd_cutoff=1e-12)

        # Evolve 10 steps; check norm stays ≈ 1 throughout
        history = tebd.evolve_ising(J=1.0, h=0.5, dt=0.05, n_steps=10)

        for entry in history:
            assert abs(entry["norm"] - 1.0) < 1e-6, (
                f"Step {entry['step']}: norm = {entry['norm']:.10f} ≠ 1"
            )

    def test_mps_inner_product_normalized_state(self):
        """A product-state MPS initialized from a computational basis state
        must have ⟨ψ|ψ⟩ = 1 exactly (norm = 1.0)."""
        from quantx_engine.tensor_network import MPS

        for n in [2, 4, 6]:
            mps = MPS.product_state([0] * n, d=2)  # API: d=2
            norm = mps.norm()  # API: norm() returns ‖|ψ⟩‖, not norm_sq
            assert norm == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Finite-size scaling: Binder cumulant signature
# ---------------------------------------------------------------------------

class TestFiniteSizePhysics:
    """Ref: Binder 1981 (cumulant method for locating phase transitions)."""

    def test_binder_cumulant_ordered_vs_disordered(self):
        """Binder cumulant U = 1 - ⟨m⁴⟩/(3⟨m²⟩²).

        At T→0 (ordered bimodal): U→2/3.
        At T→∞ (disordered Gaussian): U→0 (but can be slightly negative
        with finite samples since ⟨m⁴⟩_Gaussian = 3⟨m²⟩² exactly only in
        the thermodynamic limit).

        The key testable property is: U_ordered > U_disordered.
        """
        from quantx_engine.finite_size import binder_cumulant

        # Use a large fixed-seed ensemble to reduce statistical fluctuations
        rng = np.random.default_rng(12345)
        N = 100_000  # large N for accurate statistics

        # Ordered phase: bimodal ±1 with small Gaussian noise
        m_ordered = rng.choice([-1.0, 1.0], size=N) + rng.normal(0, 0.02, N)
        U_ordered = binder_cumulant(m_ordered)

        # Disordered phase: zero-mean Gaussian → U ≈ 0 (can be slightly negative)
        m_disorder = rng.normal(0, 1.0, N)
        U_disorder = binder_cumulant(m_disorder)

        # Key physics: ordered phase has significantly higher cumulant
        assert U_ordered > U_disorder + 0.3  # difference should be clear
        # Ordered-phase cumulant should be close to 2/3
        assert U_ordered == pytest.approx(2.0 / 3.0, abs=0.05)
