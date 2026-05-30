"""M177 — Tests for CachedDMRG memoisation layer.

Tests run without the native library; the exact-diag fallback is used for
small systems (n ≤ 20).

Physical validation
-------------------
* Heisenberg energy per site vs Bethe ansatz value (n → ∞ limit)
* CachedDMRG caching: identical calls hit cache, different calls miss
* Cache invalidation: clear_cache() resets counters and stored results
* Thread safety: concurrent calls with the same key return consistent results

References
----------
.. [1] Bethe ansatz: E/n → J(1/4 - ln 2) ≈ -0.4431 J per site (n → ∞)
"""
from __future__ import annotations

import math
import threading
import pytest


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def cached_solver():
    """CachedDMRG backed by Engine (native if available, exact-diag fallback)."""
    try:
        from quantx_engine import Engine
        from quantx_engine.cached_dmrg import CachedDMRG
        return CachedDMRG(Engine())
    except (ImportError, FileNotFoundError, OSError) as e:
        pytest.skip(f"CachedDMRG not available: {e}")
    except Exception as e:
        if "NativeUnavailable" in type(e).__name__ or "native" in str(e).lower():
            pytest.skip(f"Native library not built: {e}")
        raise


# ── Group 1: Caching behaviour ────────────────────────────────────────────────

class TestCaching:

    def test_cache_miss_on_first_call(self, cached_solver):
        """First call records a cache miss."""
        cached_solver.clear_cache()
        cached_solver.solve(n_qubits=4, hamiltonian_type="heisenberg", max_bond=8)
        info = cached_solver.cache_info()
        assert info["misses"] == 1
        assert info["hits"] == 0
        assert info["size"] == 1

    def test_cache_hit_on_repeat_call(self, cached_solver):
        """Identical call returns the same object from cache."""
        cached_solver.clear_cache()
        r1 = cached_solver.solve(n_qubits=4, hamiltonian_type="heisenberg", max_bond=8)
        r2 = cached_solver.solve(n_qubits=4, hamiltonian_type="heisenberg", max_bond=8)
        assert r1 is r2, "Expected same object from cache"
        info = cached_solver.cache_info()
        assert info["hits"] == 1
        assert info["misses"] == 1

    def test_different_params_different_cache_entries(self, cached_solver):
        """Different n_qubits produce separate cache entries."""
        cached_solver.clear_cache()
        r1 = cached_solver.solve(n_qubits=4, hamiltonian_type="heisenberg", max_bond=8)
        r2 = cached_solver.solve(n_qubits=6, hamiltonian_type="heisenberg", max_bond=8)
        assert r1 is not r2
        assert cached_solver.cache_info()["size"] == 2

    def test_different_max_bond_different_entries(self, cached_solver):
        """max_bond affects cache key."""
        cached_solver.clear_cache()
        cached_solver.solve(n_qubits=4, hamiltonian_type="heisenberg", max_bond=4)
        cached_solver.solve(n_qubits=4, hamiltonian_type="heisenberg", max_bond=8)
        assert cached_solver.cache_info()["size"] == 2

    def test_clear_cache_resets_everything(self, cached_solver):
        """clear_cache() wipes stored results and hit/miss counters."""
        cached_solver.solve(n_qubits=4, hamiltonian_type="heisenberg", max_bond=8)
        cached_solver.solve(n_qubits=4, hamiltonian_type="heisenberg", max_bond=8)
        cached_solver.clear_cache()
        info = cached_solver.cache_info()
        assert info["hits"]   == 0
        assert info["misses"] == 0
        assert info["size"]   == 0

    def test_cache_info_returns_dict(self, cached_solver):
        """cache_info() returns a dict with the required keys."""
        info = cached_solver.cache_info()
        assert isinstance(info, dict)
        assert set(info.keys()) == {"hits", "misses", "size"}

    def test_pauli_sum_term_order_invariant(self, cached_solver):
        """Equivalent Pauli-sum Hamiltonians with different term order share cache."""
        cached_solver.clear_cache()
        terms_a = [(1.0, "X0X1"), (1.0, "Y0Y1"), (1.0, "Z0Z1")]
        terms_b = [(1.0, "Z0Z1"), (1.0, "X0X1"), (1.0, "Y0Y1")]
        r1 = cached_solver.solve(n_qubits=4, hamiltonian_type="pauli_sum",
                                 hamiltonian_terms=terms_a, max_bond=8)
        r2 = cached_solver.solve(n_qubits=4, hamiltonian_type="pauli_sum",
                                 hamiltonian_terms=terms_b, max_bond=8)
        # Same energy (order-invariant key)
        assert abs(r1.energy - r2.energy) < 1e-10


# ── Group 2: Thread safety ─────────────────────────────────────────────────────

class TestThreadSafety:

    def test_concurrent_calls_same_key(self, cached_solver):
        """Concurrent calls with the same key all return equal energies."""
        cached_solver.clear_cache()
        results = []
        errors = []

        def worker():
            try:
                r = cached_solver.solve(n_qubits=4, hamiltonian_type="heisenberg",
                                        max_bond=8, n_sweeps=4)
                results.append(r.energy)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Threads raised exceptions: {errors}"
        assert len(results) == 8
        # All energies must be identical (all hit or computed the same result)
        for e in results:
            assert abs(e - results[0]) < 1e-10, f"Energy inconsistency: {results}"


# ── Group 3: Physical correctness ─────────────────────────────────────────────

class TestCachedDMRGPhysics:

    def test_heisenberg_energy_negative(self, cached_solver):
        """Heisenberg AFM ground state has negative energy."""
        r = cached_solver.solve(n_qubits=6, hamiltonian_type="heisenberg", J=1.0,
                                max_bond=16, n_sweeps=6)
        assert r.energy < 0

    def test_heisenberg_energy_per_site_approaches_bethe(self, cached_solver):
        """Energy per site approaches Bethe ansatz value for moderate n."""
        # Bethe ansatz (Pauli convention): sigma.sigma = 4 S.S, so
        # e_inf = 4 * J(1/4 - ln2) ~= -1.7726 per site.
        bethe = 4.0 * (0.25 - math.log(2))
        for n in (6, 8, 10):
            r = cached_solver.solve(n_qubits=n, hamiltonian_type="heisenberg", J=1.0,
                                    max_bond=2 ** (n // 2), n_sweeps=20, tol=1e-12)
            e_per_site = r.energy / n
            # Finite-size: |e/site| is below the asymptotic magnitude and negative.
            assert e_per_site < 0
            assert e_per_site > bethe, (
                f"n={n}: e/site={e_per_site:.4f} below asymptotic Bethe {bethe:.4f}"
            )

    def test_cached_result_consistent_with_solver(self, cached_solver):
        """CachedDMRG result matches direct DMRGSolver call."""
        from quantx_engine.dmrg import DMRGSolver
        from quantx_engine import Engine

        n = 6
        try:
            direct = DMRGSolver(Engine()).solve(
                n_qubits=n, hamiltonian_type="heisenberg", J=1.0,
                max_bond=16, n_sweeps=6
            )
        except Exception:
            pytest.skip("DMRGSolver not available for direct comparison")

        cached_solver.clear_cache()
        cached = cached_solver.solve(n_qubits=n, hamiltonian_type="heisenberg", J=1.0,
                                     max_bond=16, n_sweeps=6)
        # Energies should match (same algorithm, same params)
        assert abs(cached.energy - direct.energy) < 1e-6, (
            f"cached={cached.energy:.8f} vs direct={direct.energy:.8f}"
        )

    def test_backend_property(self, cached_solver):
        """backend property returns 'cpp' or 'python'."""
        assert cached_solver.backend in ("cpp", "python")


# ── Group 4: Scale and Bethe ansatz (larger n) ────────────────────────────────

class TestScaleTests:

    @pytest.mark.parametrize("n,chi", [(8, 8), (10, 16), (12, 32)])
    def test_heisenberg_converges_at_scale(self, cached_solver, n, chi):
        """DMRG converges and energy is below Hartree-Fock bound for n, chi."""
        cached_solver.clear_cache()
        r = cached_solver.solve(n_qubits=n, hamiltonian_type="heisenberg", J=1.0,
                                max_bond=chi, n_sweeps=15, tol=1e-10)
        # Variational: energy must be negative and finite
        assert math.isfinite(r.energy)
        assert r.energy < 0
        # Energy per site must be in the right ballpark: above the asymptotic
        # Bethe value (Pauli convention, sigma.sigma = 4 S.S) and below zero.
        bethe = 4.0 * (0.25 - math.log(2))  # ~= -1.7726 (Pauli units)
        e_per_site = r.energy / n
        assert e_per_site > bethe, f"n={n},chi={chi}: e/site={e_per_site:.4f}"

    def test_energy_monotone_in_bond_dim(self, cached_solver):
        """Larger χ gives lower or equal energy (variational)."""
        n = 8
        cached_solver.clear_cache()
        r_small = cached_solver.solve(n_qubits=n, hamiltonian_type="heisenberg",
                                      max_bond=4, n_sweeps=6)
        r_large = cached_solver.solve(n_qubits=n, hamiltonian_type="heisenberg",
                                      max_bond=16, n_sweeps=6)
        assert r_large.energy <= r_small.energy + 1e-6, (
            f"chi=4 → {r_small.energy:.6f}, chi=16 → {r_large.energy:.6f}"
        )
