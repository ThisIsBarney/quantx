"""M177 — CachedDMRG: memoised wrapper around DMRGSolver.

Calling ``DMRGSolver.solve()`` repeatedly with the same problem parameters
re-runs the full DMRG sweep from scratch.  ``CachedDMRG`` adds a thin cache
layer so that identical problem/option pairs return the stored result
immediately without calling the solver again.

Cache key
---------
The cache key is built from all parameters that affect the result:
  (n_qubits, hamiltonian_type, J, sorted_terms, max_bond, cutoff, n_sweeps, tol)
The ``hamiltonian_terms`` list is sorted and rounded to 12 decimal places
before hashing to ensure key stability across equivalent representations.

Usage::

    from quantx_engine import Engine
    from quantx_engine.cached_dmrg import CachedDMRG

    dmrg = CachedDMRG(Engine())

    # First call — runs the solver
    r1 = dmrg.solve(n_qubits=20, hamiltonian_type="heisenberg", max_bond=32)
    # Second call with identical arguments — returned from cache instantly
    r2 = dmrg.solve(n_qubits=20, hamiltonian_type="heisenberg", max_bond=32)
    assert r1 is r2              # same object

    dmrg.cache_info()            # {'hits': 1, 'misses': 1, 'size': 1}
    dmrg.clear_cache()           # discard all cached results

Notes
-----
* The cache is per-instance, not shared across ``CachedDMRG`` objects.
* The cache is in-memory only; it is not persisted to disk.
* Thread safety: the cache is protected by a simple lock, so concurrent
  calls with the same key from multiple threads are serialised.
"""
from __future__ import annotations

import hashlib
import json
import threading
from typing import Optional

from quantx_engine.dmrg import DMRGResult, DMRGSolver

__all__ = ["CachedDMRG"]


def _make_cache_key(
    n_qubits: int,
    hamiltonian_type: str,
    J: float,
    hamiltonian_terms: list,
    max_bond: int,
    cutoff: float,
    n_sweeps: int,
    tol: float,
) -> str:
    """Build a stable string key for the given problem configuration."""
    # Normalise terms: sort by pauli label, round coefficients to 12 dp
    norm_terms = sorted(
        [(round(float(c), 12), str(p)) for c, p in hamiltonian_terms],
        key=lambda x: x[1],
    )
    payload = {
        "n": n_qubits,
        "type": hamiltonian_type,
        "J": round(float(J), 12),
        "terms": norm_terms,
        "max_bond": max_bond,
        "cutoff": cutoff,
        "n_sweeps": n_sweeps,
        "tol": tol,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


class CachedDMRG:
    """Memoised ground-state DMRG solver.

    Parameters
    ----------
    engine : quantx_engine.Engine
        QuantX engine instance forwarded to the underlying ``DMRGSolver``.

    Examples
    --------
    >>> from quantx_engine import Engine
    >>> from quantx_engine.cached_dmrg import CachedDMRG
    >>> dmrg = CachedDMRG(Engine())
    >>> r = dmrg.solve(n_qubits=8, hamiltonian_type="heisenberg", max_bond=16)
    >>> r.energy  # varies by system
    -9.189...
    """

    def __init__(self, engine) -> None:
        self._solver = DMRGSolver(engine)
        self._cache: dict[str, DMRGResult] = {}
        self._inflight: dict[str, threading.Event] = {}  # per-key in-flight guard
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    # ── Public API ──────────────────────────────────────────────────────────

    def solve(
        self,
        n_qubits: int,
        *,
        hamiltonian_type: str = "pauli_sum",
        J: float = 1.0,
        hamiltonian_terms: Optional[list] = None,
        max_bond: int = 64,
        cutoff: float = 1e-10,
        n_sweeps: int = 10,
        tol: float = 1e-10,
    ) -> DMRGResult:
        """Run DMRG (or return cached result).

        All parameters are identical to :meth:`DMRGSolver.solve`.

        Concurrent calls with the same key are deduplicated: only the first
        thread runs the solver; subsequent threads wait on a per-key
        ``threading.Event`` and return the single cached result once it is
        ready.

        Returns
        -------
        DMRGResult
            Cached result if the same parameters were used before; otherwise
            the freshly computed result (which is then stored in the cache).
        """
        terms = list(hamiltonian_terms) if hamiltonian_terms is not None else []
        key = _make_cache_key(
            n_qubits, hamiltonian_type, J, terms,
            max_bond, cutoff, n_sweeps, tol,
        )

        # ── Fast path: already cached ──────────────────────────────────────
        with self._lock:
            if key in self._cache:
                self._hits += 1
                return self._cache[key]

            # ── Register or wait on in-flight guard ───────────────────────
            if key in self._inflight:
                # Another thread is already solving — wait for it.
                wait_event = self._inflight[key]
            else:
                # We are the solver thread for this key.
                wait_event = None
                self._inflight[key] = threading.Event()

        if wait_event is not None:
            wait_event.wait()
            with self._lock:
                self._hits += 1
                return self._cache[key]

        # ── Solver thread: compute and populate cache ──────────────────────
        try:
            result = self._solver.solve(
                n_qubits,
                hamiltonian_type=hamiltonian_type,
                J=J,
                hamiltonian_terms=terms or None,
                max_bond=max_bond,
                cutoff=cutoff,
                n_sweeps=n_sweeps,
                tol=tol,
            )
            with self._lock:
                self._cache[key] = result
                self._misses += 1
                done_event = self._inflight.pop(key)
            done_event.set()   # wake all waiting threads
            return result
        except Exception:
            # On failure: remove in-flight guard and wake waiters so they
            # don't block forever (they will re-raise their own exceptions
            # or get a KeyError from the cache — handled by re-entry).
            with self._lock:
                done_event = self._inflight.pop(key, None)
            if done_event is not None:
                done_event.set()
            raise

    def cache_info(self) -> dict:
        """Return cache statistics.

        Returns
        -------
        dict
            ``{"hits": int, "misses": int, "size": int}``
        """
        with self._lock:
            return {
                "hits":   self._hits,
                "misses": self._misses,
                "size":   len(self._cache),
            }

    def clear_cache(self) -> None:
        """Discard all cached results and reset hit/miss counters."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    # Expose solver properties for convenience
    @property
    def backend(self) -> str:
        """``"cpp"`` if native library is loaded, else ``"python"``."""
        return "cpp" if self._solver._lib is not None else "python"
