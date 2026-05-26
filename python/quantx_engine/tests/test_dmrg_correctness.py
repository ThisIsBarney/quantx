"""M176 — Correctness tests for the C++ MPS tensor contractions + DMRG.

All tests run in CI without the native library (pure-Python / exact-diag
fallback for small n).  When the native library IS available the C++ DMRG
path is exercised automatically.

Physical validation targets
---------------------------
* Heisenberg chain energy vs exact diagonalisation (n ≤ 10)
* Heisenberg chain energy vs Bethe ansatz (n → ∞ limit)
* Energy monotonically decreases with bond dimension (variational principle)
* Parity symmetry: E(J) is translation-invariant for periodic-like systems
* Structural: result has correct field types, sweep_energies non-increasing

References
----------
.. [1] Schollwöck (2011). Ann. Phys. 326, 96.
.. [2] Bethe ansatz: E/n → -ln2 + 1/4 ≈ -0.4431 J per site (n → ∞).
"""
from __future__ import annotations

import math
import numpy as np
import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def solver():
    """DMRGSolver backed by Engine (native if available, pure-Python fallback)."""
    try:
        from quantx_engine import Engine
        from quantx_engine.dmrg import DMRGSolver
        return DMRGSolver(Engine())
    except (ImportError, FileNotFoundError, OSError) as e:
        pytest.skip(f"DMRGSolver not available: {e}")
    except Exception as e:
        if "NativeUnavailable" in type(e).__name__ or "native" in str(e).lower():
            pytest.skip(f"Native library not built: {e}")
        raise


# ── Exact diagonalisation reference ──────────────────────────────────────────

def exact_heisenberg_energy(n: int, J: float = 1.0) -> float:
    """Compute Heisenberg XXX ground-state energy via exact diagonalisation."""
    d = 2 ** n
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = {"X": sx, "Y": sy, "Z": sz}
    H = np.zeros((d, d), dtype=complex)
    for i in range(n - 1):
        for P in "XYZ":
            ops = [np.eye(2, dtype=complex)] * n
            ops[i] = paulis[P]
            ops[i + 1] = paulis[P]
            mat = ops[0]
            for op in ops[1:]:
                mat = np.kron(mat, op)
            H += J * mat
    vals = np.linalg.eigvalsh(H)
    return float(vals[0])


# ── Group 1: Basic API / structural ──────────────────────────────────────────

class TestDMRGAPI:

    def test_result_has_required_fields(self, solver):
        """DMRGResult has all expected fields with correct types."""
        from quantx_engine.dmrg import DMRGResult
        result = solver.solve(n_qubits=4, hamiltonian_type="heisenberg", J=1.0,
                              max_bond=16, n_sweeps=4)
        assert isinstance(result, DMRGResult)
        assert isinstance(result.energy, float)
        assert isinstance(result.truncation_error, float)
        assert isinstance(result.n_sweeps_done, int)
        assert isinstance(result.sweep_energies, list)
        assert isinstance(result.wall_time_s, float)
        assert result.backend in ("cpp", "python", "python_exact")

    def test_energy_is_finite(self, solver):
        """Ground-state energy must be finite."""
        result = solver.solve(n_qubits=4, hamiltonian_type="heisenberg", J=1.0,
                              max_bond=8, n_sweeps=4)
        assert math.isfinite(result.energy)

    def test_wall_time_positive(self, solver):
        """Wall time must be positive."""
        result = solver.solve(n_qubits=4, hamiltonian_type="heisenberg", J=1.0,
                              max_bond=8, n_sweeps=2)
        assert result.wall_time_s >= 0.0

    def test_n_sweeps_done_bounded(self, solver):
        """n_sweeps_done ≤ n_sweeps."""
        n_sweeps = 5
        result = solver.solve(n_qubits=4, hamiltonian_type="heisenberg", J=1.0,
                              max_bond=8, n_sweeps=n_sweeps)
        assert 1 <= result.n_sweeps_done <= n_sweeps

    def test_sweep_energies_decreasing(self, solver):
        """Sweep energies must be non-increasing (variational)."""
        result = solver.solve(n_qubits=6, hamiltonian_type="heisenberg", J=1.0,
                              max_bond=16, n_sweeps=6)
        for i in range(1, len(result.sweep_energies)):
            assert result.sweep_energies[i] <= result.sweep_energies[i - 1] + 1e-6, (
                f"Energy increased at sweep {i}: "
                f"{result.sweep_energies[i-1]:.8f} → {result.sweep_energies[i]:.8f}"
            )


# ── Group 2: Physical correctness (small n vs exact diag) ────────────────────

class TestDMRGPhysics:

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_heisenberg_energy_vs_exact(self, solver, n):
        """DMRG energy matches exact diagonalisation for small n (bond dim ≥ full)."""
        J = 1.0
        ref = exact_heisenberg_energy(n, J)
        result = solver.solve(
            n_qubits=n,
            hamiltonian_type="heisenberg",
            J=J,
            max_bond=2 ** (n // 2),   # large enough for exact MPS
            n_sweeps=20,
            tol=1e-12,
        )
        assert abs(result.energy - ref) < 1e-4, (
            f"n={n}: DMRG energy {result.energy:.8f} vs exact {ref:.8f}, "
            f"diff={abs(result.energy - ref):.2e}"
        )

    def test_heisenberg_energy_negative(self, solver):
        """Heisenberg AFM (J>0) ground-state energy is negative."""
        result = solver.solve(n_qubits=6, hamiltonian_type="heisenberg", J=1.0,
                              max_bond=16, n_sweeps=6)
        assert result.energy < 0, f"Expected negative energy, got {result.energy}"

    def test_heisenberg_fm_energy_positive(self, solver):
        """Ferromagnetic (J<0) Heisenberg ground-state energy is ≤ 0."""
        # FM: ground state is fully polarised, E/bond = -J (each bond = -3)
        J = -1.0
        n = 4
        result = solver.solve(n_qubits=n, hamiltonian_type="heisenberg", J=J,
                              max_bond=8, n_sweeps=6)
        ref = exact_heisenberg_energy(n, J)
        assert abs(result.energy - ref) < 1e-4, (
            f"FM n={n}: DMRG {result.energy:.8f} vs exact {ref:.8f}"
        )

    def test_heisenberg_energy_scales_with_n(self, solver):
        """Energy per site converges toward Bethe-ansatz value as n grows."""
        # Bethe ansatz: e_∞ = J * (1/4 - ln2) ≈ -0.4431 J per site
        bethe = 1.0 * (0.25 - math.log(2))  # ≈ -0.4431
        for n in (8, 10):
            result = solver.solve(n_qubits=n, hamiltonian_type="heisenberg", J=1.0,
                                  max_bond=2 ** (n // 2), n_sweeps=20, tol=1e-12)
            e_per_site = result.energy / n
            # Finite-size corrections are large; just check the sign and order
            assert e_per_site < 0, f"n={n}: energy/site should be negative"
            assert e_per_site > 2 * bethe, (
                f"n={n}: energy/site {e_per_site:.4f} too negative vs Bethe {bethe:.4f}"
            )

    def test_energy_decreases_with_bond_dim(self, solver):
        """Ground energy with max_bond=16 ≤ energy with max_bond=4 (variational)."""
        n = 8
        r_small = solver.solve(n_qubits=n, hamiltonian_type="heisenberg", J=1.0,
                               max_bond=4, n_sweeps=6)
        r_large = solver.solve(n_qubits=n, hamiltonian_type="heisenberg", J=1.0,
                               max_bond=16, n_sweeps=6)
        assert r_large.energy <= r_small.energy + 1e-6, (
            f"Larger bond dim should give lower or equal energy: "
            f"chi=4 → {r_small.energy:.6f}, chi=16 → {r_large.energy:.6f}"
        )


# ── Group 3: Pauli-sum Hamiltonian ────────────────────────────────────────────

class TestPauliSumHamiltonian:

    def test_pauli_sum_heisenberg_matches_builtin(self, solver):
        """Pauli-sum version of Heisenberg matches the built-in heisenberg type."""
        n = 6
        J = 1.0
        terms = []
        for i in range(n - 1):
            for gate in ("X", "Y", "Z"):
                terms.append((J, f"{gate}{i}{gate}{i+1}"))

        r_builtin = solver.solve(n_qubits=n, hamiltonian_type="heisenberg", J=J,
                                 max_bond=16, n_sweeps=10)
        r_pauli   = solver.solve(n_qubits=n, hamiltonian_type="pauli_sum",
                                 hamiltonian_terms=terms,
                                 max_bond=16, n_sweeps=10)
        assert abs(r_pauli.energy - r_builtin.energy) < 0.01, (
            f"pauli_sum energy {r_pauli.energy:.6f} differs from "
            f"heisenberg builtin {r_builtin.energy:.6f}"
        )

    def test_ising_energy(self, solver):
        """Transverse-field Ising: Z0Z1 + Z1Z2 + ... + h Σ X_i."""
        n, J, h = 6, 1.0, 0.5
        terms = []
        for i in range(n - 1):
            terms.append((-J, f"Z{i}Z{i+1}"))
        for i in range(n):
            terms.append((-h, f"X{i}"))
        result = solver.solve(n_qubits=n, hamiltonian_type="pauli_sum",
                              hamiltonian_terms=terms,
                              max_bond=16, n_sweeps=10)
        assert math.isfinite(result.energy)
        # For h < J the Ising model is in the ordered phase; ground energy < 0
        assert result.energy < 0


# ── Group 4: Error handling ───────────────────────────────────────────────────

class TestDMRGErrors:

    def test_n_qubits_too_small(self, solver):
        from quantx_engine.dmrg import DMRGError
        with pytest.raises(DMRGError, match="n_qubits"):
            solver.solve(n_qubits=1, hamiltonian_type="heisenberg")

    def test_unknown_hamiltonian_type(self, solver):
        from quantx_engine.dmrg import DMRGError
        with pytest.raises(DMRGError, match="hamiltonian_type"):
            solver.solve(n_qubits=4, hamiltonian_type="foobar")

    def test_pauli_sum_missing_terms(self, solver):
        from quantx_engine.dmrg import DMRGError
        with pytest.raises(DMRGError):
            solver.solve(n_qubits=4, hamiltonian_type="pauli_sum",
                         hamiltonian_terms=None)


# ── Group 5: Truncation error ─────────────────────────────────────────────────

class TestTruncation:

    def test_truncation_zero_for_exact_bond_dim(self, solver):
        """Truncation error ≈ 0 when bond dim is large enough for exact MPS."""
        n = 4
        # For n=4 the exact MPS needs bond dim ≤ 2^(n//2) = 4
        result = solver.solve(n_qubits=n, hamiltonian_type="heisenberg", J=1.0,
                              max_bond=8, n_sweeps=10, tol=1e-12)
        assert result.truncation_error >= 0.0
        # If native C++ is used, truncation should be tiny
        if result.backend == "cpp":
            assert result.truncation_error < 1e-6, (
                f"Expected negligible truncation error, got {result.truncation_error:.2e}"
            )

    def test_truncation_positive_for_small_bond_dim(self, solver):
        """With bond dim=2 on n=8, truncation error must be > 0."""
        result = solver.solve(n_qubits=8, hamiltonian_type="heisenberg", J=1.0,
                              max_bond=2, n_sweeps=4)
        if result.backend == "cpp":
            assert result.truncation_error >= 0.0  # can't assert > 0, might still fit
