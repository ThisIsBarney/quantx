"""G1: examples/quickstart.py against exact references computed here, without the engine.

The quick start runs three engine features in about twenty lines and asserts each against a
stored exact value. Here the three exact values are recomputed (two exact diagonalisations
and a state vector) and the example's own results are compared with them, so neither the
engine nor the stored values can drift unseen.
"""
from __future__ import annotations

import os
import pathlib
import runpy

import numpy as np
import pytest

EXAMPLE = pathlib.Path(__file__).resolve().parents[3] / "examples" / "quickstart.py"


def _native_or_skip():
    # Where the native library is supposed to be present (CI's native job sets
    # QX_REQUIRE_NATIVE=1), a load failure must fail the gate, not quietly skip it.
    unavailable = pytest.fail if os.environ.get("QX_REQUIRE_NATIVE") == "1" else pytest.skip
    try:
        from quantx_engine.engine import Engine
        engine = Engine()
    except Exception as exc:  # QuantXNativeUnavailable, or a missing dependency
        unavailable(f"native library unavailable: {exc}")
    for fn in ("qx_dmrg_sweep", "qx_thermal", "qx_hva_circuit"):
        if not hasattr(engine._lib, fn):
            unavailable(f"native library has no {fn}")


@pytest.mark.skipif(not EXAMPLE.exists(), reason="examples/ is not part of an installed package")
def test_quickstart_matches_exact_references():
    _native_or_skip()
    out = runpy.run_path(str(EXAMPLE), run_name="quickstart")    # its own asserts run too

    # The Hubbard ground state against an exact diagonalisation of its sector.
    from quantx_engine.tests.test_models_ed import _hubbard_sector_ed
    assert abs(out["gs"].energy - _hubbard_sector_ed(8, 1.0, 4.0, 0.0, 4, 4)) < 1e-9

    # S(k, omega, T): exact correlations of the whole Hilbert space through the same transform.
    from quantx_engine.tests.test_thermal_ed import _Gibbs, _spin_chain
    from quantx_engine.thermal import ThermalStructureFactor
    sf = out["sf"]
    H, Sz, _ = _spin_chain(8, 1, 1.0, 1.0)
    g = _Gibbs(H, sf.beta)
    C = np.stack([g.correlation(s, Sz[sf.source], sf.times) for s in Sz], axis=1)
    exact = ThermalStructureFactor(beta=sf.beta, times=sf.times, correlation=C, source=sf.source,
                                   energy=g.mean(H), max_bond=0,
                                   bond_dims=np.zeros(len(sf.times), dtype=int), wall_time_s=0.0)
    assert np.max(np.abs(out["S"] - exact.spectrum()[2])) < 1e-3

    # The HVA circuit against a state vector of the same circuit.
    from quantx_engine import hva
    from quantx_engine.tests.test_hva_statevector import _reference
    ladder, r = out["ladder"], out["r"]
    e_sv, _ = _reference(ladder, hva.layout(ladder).groups, out["angles"], 3, 3, "product",
                         r.init_config)
    assert abs(r.energy - e_sv) < 1e-9
