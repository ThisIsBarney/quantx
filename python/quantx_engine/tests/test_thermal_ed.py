"""M6 ED gate: the finite-temperature engine against exact diagonalisation.

quantx_engine.thermal holds the Gibbs state of a chain as a purified (ancilla) state on
the native engine. Here every quantity it returns is checked against a dense exact
diagonalisation of the whole Hilbert space, built independently of the engine: the
Gibbs state over every S^z sector (for Hubbard, every particle number too) at the same
beta, its exact real-time evolution, and the operators of quantx_engine.models (spins
S = sigma/2, the S^z current, and for Hubbard Jordan-Wigner fermions with the particle
current) built here from scratch. A slip in the conversions from the engine's Pauli
conventions is off by a factor of 2 or 4.

The chains are small enough that the bond dimension holds the purified state whole, so
nothing is truncated and what remains is the second-order Trotter error of the cooling
step dbeta and the real-time step dt. The tolerances are set from that error at the
steps used, and the comparison is repeated at half the steps, where the error has to
fall by close to 4: an error of convention does not move with the step.

The spectra are gated separately, without the engine: the Fourier transforms of exact
correlations against the Lehmann sums they approximate.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from quantx_engine import thermal
from quantx_engine.models import (HubbardChain, HubbardLattice, J1J2Chain, Spin1Chain,
                                  XXZChain)
from quantx_engine.thermal import (ThermalCurrentCorrelation, ThermalError,
                                   ThermalStructureFactor)

# Each model with a bond dimension that holds its purified state whole (the largest
# cut of the 2L-site chain is d^L), the Trotter tolerances at the default steps
# dbeta = dt = 0.05, and the scan.
MODELS = {
    "xxz": (XXZChain(L=6, J=1.3, delta=0.7), 64),
    "spin1": (Spin1Chain(L=4, J=0.8), 81),
    "hubbard": (HubbardChain(L=4, t=1.1, U=3.0, mu=0.9), 256),
}
BETA_STATICS = 2.0
BETA_DYNAMICS, T_MAX = 1.0, 2.0
EXACT = dict(cutoff=1e-14)


def _native_or_skip():
    # Where the native library is supposed to be present (CI's native job sets
    # QX_REQUIRE_NATIVE=1), a load failure must fail the gate, not quietly skip it.
    unavailable = pytest.fail if os.environ.get("QX_REQUIRE_NATIVE") == "1" else pytest.skip
    try:
        from quantx_engine.engine import Engine
        engine = Engine()
    except Exception as exc:  # QuantXNativeUnavailable, or a missing dependency
        unavailable(f"native library unavailable: {exc}")
    if not hasattr(engine._lib, "qx_thermal"):
        unavailable("native library has no qx_thermal")
    return engine


@pytest.fixture(scope="module")
def engine():
    return _native_or_skip()


# ── Exact diagonalisation, independent of the engine ─────────────────────────

def _embed(op, j, L):
    d = op.shape[0]
    return np.kron(np.kron(np.eye(d ** j), op), np.eye(d ** (L - j - 1)))


def _spin_chain(L, two_s, J, delta):
    """H, the S^z_j and the S^z current of the open spin-S chain, on the full space.

    H = J sum_i (Sx Sx + Sy Sy + delta Sz Sz) and, from dS^z_l/dt = j_{l-1} - j_l, the
    current J = sum_l (i J / 2) (S+_l S-_{l+1} - h.c.).
    """
    S = two_s / 2
    ms = [S - k for k in range(two_s + 1)]
    sp = np.zeros((two_s + 1, two_s + 1))
    for k in range(1, two_s + 1):                  # level k - 1 is m + 1
        sp[k - 1, k] = math.sqrt(S * (S + 1) - ms[k] * (ms[k] + 1))
    Sz = [_embed(np.diag(ms), j, L) for j in range(L)]
    Sp = [_embed(sp, j, L) for j in range(L)]
    Sm = [o.T for o in Sp]
    H = sum(J * (0.5 * (Sp[i] @ Sm[i + 1] + Sm[i] @ Sp[i + 1]) + delta * Sz[i] @ Sz[i + 1])
            for i in range(L - 1))
    cur = sum(0.5j * J * (Sp[i] @ Sm[i + 1] - Sm[i] @ Sp[i + 1]) for i in range(L - 1))
    return H, Sz, cur


def _hubbard_chain(L, t, U, mu):
    """H, n_j, S^z_j and the particle current of the open Hubbard chain, on the full space.

    Jordan-Wigner fermions on the 2L modes (all up, then all down), so the signs are the
    physical ones in any geometry. H is that of HubbardChain, and the current is
    J = sum_{l,s} i t (c+_{l+1 s} c_{l s} - h.c.), from dn_l/dt = j_{l-1} - j_l.
    """
    modes = 2 * L
    a, z = np.array([[0.0, 1.0], [0.0, 0.0]]), np.diag([1.0, -1.0])
    c = []
    for k in range(modes):
        op = np.eye(1)
        for m in range(modes):
            op = np.kron(op, z if m < k else a if m == k else np.eye(2))
        c.append(op)
    n = [ck.T @ ck for ck in c]
    H, cur = 0.0, 0.0
    for s in (0, 1):
        for l in range(L - 1):
            fwd = c[s * L + l + 1].T @ c[s * L + l]          # c+_{l+1} c_l
            H = H - t * (fwd + fwd.T)
            cur = cur + 1j * t * (fwd - fwd.T)
    for l in range(L):
        H = H + U * n[l] @ n[L + l] - mu * (n[l] + n[L + l])
    dens = [n[l] + n[L + l] for l in range(L)]
    sz = [0.5 * (n[l] - n[L + l]) for l in range(L)]
    return H, dens, sz, cur


def _ed(model):
    """(H, O_j, S^z_j, J): O is what ThermalStatics.local holds (S^z, or n for Hubbard)."""
    if isinstance(model, XXZChain):
        H, sz, cur = _spin_chain(model.L, 1, model.J, model.delta)
        return H, sz, sz, cur
    if isinstance(model, Spin1Chain):
        H, sz, cur = _spin_chain(model.L, 2, model.J, 1.0)
        return H, sz, sz, cur
    return _hubbard_chain(model.L, model.t, model.U, model.mu)


class _Gibbs:
    """rho = e^{-beta H} / Z, with exact expectation values and real-time correlations."""

    def __init__(self, H, beta):
        self.E, self.V = np.linalg.eigh(H)
        w = np.exp(-beta * (self.E - self.E[0]))
        self.p = w / w.sum()

    def _eig(self, A):
        return self.V.conj().T @ A @ self.V

    def mean(self, A):
        return float(np.real(np.sum(self.p * np.diag(self._eig(A)))))

    def weights(self, A, B):
        """w_mn = p_m A_mn B_nm, so <A(t) B> = sum_mn w_mn e^{-i (E_n - E_m) t}."""
        return self.p[:, None] * self._eig(A) * self._eig(B).T

    def correlation(self, A, B, times):
        """<A(t) B> with A(t) = e^{iHt} A e^{-iHt}, at each time."""
        w = self.weights(A, B)
        dE = self.E[None, :] - self.E[:, None]               # E_n - E_m
        return np.array([np.sum(w * np.exp(-1j * dE * t)) for t in times])


def _gaussian(x, eta):
    """The Fourier transform of the time window exp(-(eta t)^2 / 2) over all t."""
    return math.sqrt(2 * math.pi) / eta * np.exp(-0.5 * (x / eta) ** 2)


# ── The gates ───────────────────────────────────────────────────────────────────

def _statics_errors(model, chi, engine, dbeta):
    st = thermal.statics(model, beta=BETA_STATICS, dbeta=dbeta, max_bond=chi, **EXACT,
                         engine=engine)
    H, O, _, _ = _ed(model)
    c, L = st.center, model.L
    M = sum(O)
    err = {"energy": 0.0, "local": 0.0, "correlation": 0.0, "susceptibility": 0.0}
    for k, b in enumerate(st.betas):
        g = _Gibbs(H, b)
        ed_local = np.array([g.mean(o) for o in O])
        ed_corr = np.array([g.mean(O[c] @ o) for o in O])
        ed_chi = b * (g.mean(M @ M) - g.mean(M) ** 2) / L
        err["energy"] = max(err["energy"], abs(st.energy[k] - g.mean(H)))
        err["local"] = max(err["local"], np.max(np.abs(st.local[k] - ed_local)))
        err["correlation"] = max(err["correlation"], np.max(np.abs(st.correlation[k] - ed_corr)))
        err["susceptibility"] = max(err["susceptibility"], abs(st.susceptibility[k] - ed_chi))
    return st, err


def _dynamics_errors(model, chi, engine, step):
    H, _, Sz, cur = _ed(model)
    g = _Gibbs(H, BETA_DYNAMICS)
    sf = thermal.structure_factor(model, beta=BETA_DYNAMICS, t_max=T_MAX, dt=step,
                                  dbeta=step, max_bond=chi, **EXACT, engine=engine)
    ed_sf = np.stack([g.correlation(s, Sz[sf.source], sf.times) for s in Sz], axis=1)
    cc = thermal.current_correlation(model, beta=BETA_DYNAMICS, t_max=T_MAX, dt=step,
                                     dbeta=step, max_bond=chi, **EXACT, engine=engine)
    ed_cc = g.correlation(cur, cur, cc.times)
    err = {"energy": max(abs(sf.energy - g.mean(H)), abs(cc.energy - g.mean(H))),
           "structure_factor": float(np.max(np.abs(sf.correlation - ed_sf))),
           "current": float(np.max(np.abs(cc.correlation - ed_cc)))}
    return sf, cc, err


# At dbeta = dt = STEP every |engine - ED| is below TROTTER_TOL (the largest, the
# Hubbard current, is 4.2e-4 of a C(0) of 3.8; a slip of convention is 0.1 or more),
# and at half the steps it falls by at least 3 (second order gives 4; measured 3.9 to
# 4.0). Errors below ORDER_FLOOR are rounding and the SVD cutoff, with no order to see.
STEP = 0.05
TROTTER_TOL = 1e-3
ORDER_FLOOR = 1e-7


def _check_trotter(name, err, half):
    for q, e in err.items():
        assert e < TROTTER_TOL, f"{name} {q}: |engine - ED| = {e:.2e}"
        if e > ORDER_FLOOR:
            assert half[q] < e / 3, (f"{name} {q}: {e:.2e} at dt = {STEP} but {half[q]:.2e} "
                                     f"at {STEP / 2}: not a Trotter error")


@pytest.mark.parametrize("name", list(MODELS))
def test_statics_match_ed(engine, name):
    model, chi = MODELS[name]
    st, err = _statics_errors(model, chi, engine, STEP)
    assert st.observable == ("n" if name == "hubbard" else "Sz")
    assert st.betas[0] == 0.0 and st.betas[-1] == pytest.approx(BETA_STATICS)
    assert st.local.shape == st.correlation.shape == (len(st.betas), model.L)
    assert st.center == model.L // 2 and st.max_bond <= chi
    _, half = _statics_errors(model, chi, engine, STEP / 2)
    _check_trotter(name, err, half)


@pytest.mark.parametrize("name", list(MODELS))
def test_dynamics_match_ed(engine, name):
    model, chi = MODELS[name]
    sf, cc, err = _dynamics_errors(model, chi, engine, STEP)
    n_t = round(T_MAX / STEP) + 1
    assert sf.correlation.shape == (n_t, model.L) and cc.correlation.shape == (n_t,)
    assert sf.times[-1] == pytest.approx(T_MAX) and cc.n_sites == model.L
    assert sf.bond_dims.shape == (n_t,) and sf.max_bond <= chi and cc.max_bond <= chi
    _, _, half = _dynamics_errors(model, chi, engine, STEP / 2)
    _check_trotter(name, err, half)


def test_grids_end_at_the_requested_beta_and_time(engine):
    # 0.07 / 0.05 is not a whole number of steps: the ratio is rounded up and the step
    # shrinks, so neither grid overshoots (rounding the count alone ran to t = 0.10).
    st = thermal.statics(XXZChain(L=4), beta=0.07, dbeta=0.05, engine=engine)
    np.testing.assert_allclose(st.betas, [0.0, 0.035, 0.07], atol=1e-12)
    for run in (thermal.structure_factor, thermal.current_correlation):
        r = run(XXZChain(L=4), beta=0.5, t_max=0.07, dt=0.05, engine=engine)
        np.testing.assert_allclose(r.times, [0.0, 0.035, 0.07], atol=1e-12)
    r = thermal.structure_factor(XXZChain(L=4), beta=0.5, t_max=0.0, engine=engine)
    np.testing.assert_allclose(r.times, [0.0])


def test_statics_stride_and_no_susceptibility(engine):
    model, chi = MODELS["xxz"]
    full = thermal.statics(model, beta=1.0, max_bond=chi, engine=engine)
    st = thermal.statics(model, beta=1.0, max_bond=chi, stride=4, susceptibility=False,
                         engine=engine)
    assert st.betas[-1] == pytest.approx(1.0) and len(st.betas) < len(full.betas)
    assert not np.any(st.susceptibility)
    np.testing.assert_allclose(st.energy[-1], full.energy[-1], atol=1e-12)
    np.testing.assert_allclose(st.correlation[-1], full.correlation[-1], atol=1e-12)


# ── The transforms, against Lehmann sums (no engine) ─────────────────────────

TIMES = np.arange(0.0, 10.0 + 1e-9, 0.05)
ETA = 6.0 / TIMES[-1]     # the window leaves e^{-18} at the last time


def test_spectrum_is_the_lehmann_sum():
    """S(k, omega) of exact correlations is the broadened Lehmann sum.

    Five sites, so the source sits at the centre: with the chain's reflection and a real
    H, C_k(-t) = conj(C_k(t)) holds exactly, and the windowed transform is exactly
    sum_mn w_mn G(omega - (E_n - E_m)), G the transform of the window.
    """
    L, beta = 5, 0.8
    H, Sz, _ = _spin_chain(L, 1, 1.0, 0.6)
    g, c = _Gibbs(H, beta), L // 2
    C = np.stack([g.correlation(s, Sz[c], TIMES) for s in Sz], axis=1)
    sf = ThermalStructureFactor(beta=beta, times=TIMES, correlation=C, source=c,
                                energy=g.mean(H), max_bond=0,
                                bond_dims=np.zeros(len(TIMES), dtype=int), wall_time_s=0.0)
    ks, omegas, S = sf.spectrum(n_omega=241, omega_max=6.0, eta=ETA)
    assert S.shape == (L, 241) and np.allclose(ks, 2 * np.pi * np.arange(L) / L)
    dE = g.E[None, :] - g.E[:, None]
    for i, k in enumerate(ks):
        w = g.weights(sum(np.exp(-1j * k * (j - c)) * Sz[j] for j in range(L)), Sz[c])
        ref = np.array([np.sum(w * _gaussian(x - dE, ETA)) for x in omegas])
        assert np.max(np.abs(ref.imag)) < 1e-12
        np.testing.assert_allclose(S[i], ref.real, rtol=0, atol=1e-8)
    # Absorption above zero: at beta > 0 the positive frequencies carry more weight.
    local = S.mean(axis=0)
    assert np.sum(local[omegas > 0]) > np.sum(local[omegas < 0])


def test_transform_is_the_trapezoid_rule():
    """The transform is the trapezoid rule over [-t_max, t_max], ends at half weight.

    At the default window the last time still carries e^{-4.5}, so a rule that counted the
    ends in full would be off by that much of the last sample.
    """
    rng = np.random.default_rng(7)
    times = np.arange(0.0, 3.0 + 1e-9, 0.05)
    C = (rng.normal(size=len(times)) + 1j * rng.normal(size=len(times)))[:, None]
    sf = ThermalStructureFactor(beta=1.0, times=times, correlation=C, source=0, energy=0.0,
                                max_bond=0, bond_dims=np.zeros(len(times), dtype=int),
                                wall_time_s=0.0)
    _, omegas, S = sf.spectrum(n_omega=61, omega_max=6.0)
    t_all = np.concatenate([-times[:0:-1], times])
    C_all = np.concatenate([np.conj(C[:0:-1, 0]), C[:, 0]])
    window = np.exp(-0.5 * (3.0 / times[-1] * t_all) ** 2)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz    # numpy < 2 has only trapz
    ref = [trapezoid(np.exp(1j * w * t_all) * window * C_all, t_all).real for w in omegas]
    np.testing.assert_allclose(S[0], ref, rtol=0, atol=1e-12)
    single = ThermalStructureFactor(beta=1.0, times=times[:1], correlation=C[:1], source=0,
                                    energy=0.0, max_bond=0, bond_dims=np.zeros(1, dtype=int),
                                    wall_time_s=0.0)
    with pytest.raises(ThermalError, match="two times"):
        single.spectrum()


def test_conductivity_is_the_kubo_formula():
    """Re sigma(omega) = (1 - e^{-beta omega}) / (2 L omega) sum_mn p_m |J_mn|^2 G(...)."""
    L, beta = 6, 0.7
    H, _, cur = _spin_chain(L, 1, 1.0, 0.5)
    g = _Gibbs(H, beta)
    cc = ThermalCurrentCorrelation(beta=beta, times=TIMES,
                                   correlation=g.correlation(cur, cur, TIMES), n_sites=L,
                                   energy=g.mean(H), max_bond=0,
                                   bond_dims=np.zeros(len(TIMES), dtype=int), wall_time_s=0.0)
    omegas, sigma = cc.conductivity(n_omega=241, omega_max=6.0, eta=ETA)
    assert omegas[0] == 0.0 and omegas[-1] == pytest.approx(6.0)
    w, dE = g.weights(cur, cur), g.E[None, :] - g.E[:, None]
    ref = []
    for x in omegas:
        f = beta if abs(x) < 1e-9 else -math.expm1(-beta * x) / x
        ref.append(f * np.real(np.sum(w * _gaussian(x - dE, ETA))) / (2 * L))
    np.testing.assert_allclose(sigma, ref, rtol=0, atol=1e-8)
    assert np.all(sigma > -1e-12)
    # The default grid runs to the Nyquist frequency. Below zero the Kubo factor grows as
    # e^{beta |omega|} against a transform that is only ringing there: 1e19 for a Hubbard
    # chain at beta = 1, which is why the grid starts at zero.
    w_def, s_def = cc.conductivity()
    assert w_def[0] == 0.0 and np.max(np.abs(s_def)) < 2 * np.max(sigma)


# ── Refusals (no engine) ─────────────────────────────────────────────────────

@pytest.mark.parametrize("model", [J1J2Chain(L=6, J2=0.5), HubbardLattice(Lx=3, W=2)])
def test_models_without_a_purified_form_are_refused(model):
    for run in (thermal.statics, thermal.structure_factor, thermal.current_correlation):
        with pytest.raises(ThermalError, match="no purified form"):
            run(model, beta=1.0)


@pytest.mark.parametrize("kwargs", [
    dict(beta=0.0), dict(beta=-1.0), dict(beta=float("nan")), dict(beta="1"),
    dict(beta=True), dict(beta=1.0, dbeta=0.0), dict(beta=1.0, max_bond=0),
    dict(beta=1.0, max_bond=8.0), dict(beta=1.0, stride=0), dict(beta=1.0, stride=True)])
def test_bad_statics_arguments_are_refused(kwargs):
    with pytest.raises(ThermalError):
        thermal.statics(XXZChain(L=4), **kwargs)


@pytest.mark.parametrize("kwargs", [
    dict(beta=1.0, dt=0.0), dict(beta=1.0, dt=-0.1), dict(beta=1.0, t_max=-1.0),
    dict(beta=1.0, t_max=float("inf")), dict(beta=0.0)])
def test_bad_dynamics_arguments_are_refused(kwargs):
    for run in (thermal.structure_factor, thermal.current_correlation):
        with pytest.raises(ThermalError):
            run(XXZChain(L=4), **kwargs)
