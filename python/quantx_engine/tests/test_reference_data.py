"""The exact references committed under docs/ are what they claim to be.

docs/hubbard_lieb_wu.csv carries the Lieb-Wu ground-state energy of the half-filled
Hubbard chain, the exact answer its DMRG rows are compared with. Here it is recomputed
independently of the C++ generator (src/hubbard_showcase.cpp), by adaptive quadrature one
Bessel half-period at a time, and the committed column must match to the 8 digits it
prints. The trapezoid rule that first made it was off in the 7th digit (1.8e-7 to 7.3e-7),
and nothing pinned it.
"""
from __future__ import annotations

import csv
import math
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
LIEB_WU_CSV = ROOT / "docs" / "hubbard_lieb_wu.csv"


def lieb_wu_energy(U: float) -> float:
    """e(U) = -4 int_0^inf J0(w) J1(w) / (w (1 + e^{wU/2})) dw, the half-filled chain at t = 1."""
    import scipy.integrate as si
    import scipy.special as ss

    def f(w):
        return ss.j0(w) * ss.j1(w) * ss.expit(-w * U / 2) / w

    w_end = 100.0 / U + 50.0          # past it the Fermi factor is below e^{-50}
    edges = [k * math.pi for k in range(int(w_end / math.pi) + 2)]
    return -4.0 * sum(si.quad(f, a, b, epsabs=1e-15, epsrel=1e-13)[0]
                      for a, b in zip(edges, edges[1:]))


@pytest.mark.skipif(not LIEB_WU_CSV.exists(), reason="docs/ is not part of an installed package")
def test_the_lieb_wu_column_is_exact():
    rows = list(csv.DictReader(LIEB_WU_CSV.open(encoding="utf-8")))
    assert rows
    for U in sorted({float(r["U"]) for r in rows}):
        exact = lieb_wu_energy(U)
        for r in rows:
            if float(r["U"]) == U:
                # 8 printed decimals: 5e-9 of rounding, and nothing more.
                assert abs(float(r["e_liebwu"]) - exact) < 6e-9, (U, r["e_liebwu"], exact)


def test_lieb_wu_energy_matches_strong_coupling():
    # The independent check of the formula's normalisation: at strong coupling
    # e(U) = -4 ln 2 / U + 9 zeta(3) / U^3 + O(U^-5) (the Heisenberg chain with J = 4/U,
    # and its first correction). The residual is measured at 7.6e-7 and 2.4e-8 at
    # U = 40 and 80, falling by 2^5 per doubling.
    zeta3 = 1.2020569031595942
    for U, tol in ((40.0, 1e-6), (80.0, 4e-8)):
        series = -4 * math.log(2) / U + 9 * zeta3 / U ** 3
        assert abs(lieb_wu_energy(U) - series) < tol, U
