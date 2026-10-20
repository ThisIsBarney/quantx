"""M13 gate: the adjoint gradient of HVA circuits, against a state vector and against
central differences.

``hva.circuit(..., gradient=True)`` returns dE/dtheta for every angle from one pass back
through the circuit. The checks:

- Against the exact gradient of a state-vector simulation written independently of the
  engine: the Fock-space simulator of test_hva_statevector, with the adjoint formula in
  dense linear algebra. With ``svd="direct"`` the circuit's own rounding is ~1e-14, so
  this shows the method's precision. It is gated to 1e-10 from product starts (measured:
  up to 1.2e-11 on gradients of size 3 to 7) and to 1e-6 from free starts, which carry
  their DMRG preparation's residual (measured 1.4e-7 to 2.4e-7).
- The default route, ``svd="gram"``, within its rounding. Its splits go through the Gram
  matrix, which leaves Schmidt values below sqrt(eps) ~ 1e-8 imprecise, so a circuit
  whose bonds fill up, like the chain here, carries a rounding noise of up to ~1e-10 in
  its energy. Measured: the chain's energy 6e-12 off on Windows, 1.2e-10 on Linux; its
  gradient 3.5e-9 off on Windows. It is gated to 1e-8 and 1e-6.
- Against central differences of the engine's own energy, to 1e-8 (the done-when):
  the five-point stencil at h = 5e-4, whose truncation error is h^4/30 times the fifth
  derivative, with ``svd="direct"``. On the Gram route, the chain's rounding noise,
  divided by h, would cap it at a few 1e-8 at any step size.
"""
from __future__ import annotations

import numpy as np
import pytest

from quantx_engine.models import HubbardChain, HubbardLattice
from quantx_engine.tests.test_hva_statevector import Fock, _angles, _native_or_skip

TOL = 1e-10         # energy and gradient against the state vector: product start, direct
FREE_TOL = 1e-6     # from a free start, which carries its DMRG preparation's residual
GRAM_E_TOL = 1e-8   # the default (Gram) route's energy, within its rounding
GRAM_TOL = 1e-6     # and its gradient
FD_TOL = 1e-8       # against central differences: the M13 done-when
H_FD = 5e-4


@pytest.fixture(scope="module")
def engine():
    return _native_or_skip()


def _lattice(model):
    if isinstance(model, HubbardLattice):
        return model
    return HubbardLattice(Lx=model.L, W=1, t=model.t, U=model.U, mu=model.mu)


def _exact_gradient(lat, groups, angles, n_up, n_dn, init, config=None):
    """Energy and dE/dtheta of the circuit, by state vector.

    The circuit is U_K ... U_1 |psi_0>, each U_k = exp(-i theta_k G_k): a layer's bond
    groups, then its on-site term. Then dE/dtheta_k = 2 Im <lambda_k| G_k |psi_k>, with
    psi_k the state after U_k and lambda_k = U_{k+1}^dag ... U_K^dag H psi_K."""
    fock = Fock(lat.L, n_up, n_dn)
    gens = [fock.hop(g, lat.t) for g in groups] + [np.diag(fock.onsite(lat.U, lat.mu))]
    if init == "product":
        v = fock.product(config)
    else:
        w, V = np.linalg.eigh(sum(gens[:-1]))
        v = V[:, 0].astype(complex)
    eig = [np.linalg.eigh(G) for G in gens]
    states, steps = [], []
    for layer in angles:
        for k, theta in enumerate(layer):
            w, V = eig[k]
            v = V @ (np.exp(-1j * theta * w) * (V.conj().T @ v))
            states.append(v)
            steps.append((k, theta))
    H = sum(gens)
    energy = float(np.real(v.conj() @ H @ v))
    lam = H @ v
    grad = []
    for psi, (k, theta) in zip(reversed(states), reversed(steps)):
        grad.append(2 * np.imag(lam.conj() @ (gens[k] @ psi)))
        w, V = eig[k]
        lam = V @ (np.exp(1j * theta * w) * (V.conj().T @ lam))
    per = len(angles[0])
    grad = grad[::-1]
    return energy, [grad[i:i + per] for i in range(0, len(grad), per)]


CASES = [
    # model, (n_up, n_dn), init, layers
    (HubbardLattice(Lx=6, W=1, U=4.0), (3, 3), "product", 3),                    # chain
    (HubbardLattice(Lx=3, W=2, U=4.0), (4, 2), "product", 2),                    # ladder
    (HubbardLattice(Lx=2, W=3, U=2.0, mu=0.3), (3, 3), "product", 2),            # strip
    (HubbardLattice(Lx=2, W=3, U=4.0, y_periodic=True), (3, 3), "product", 3),   # cylinder
    (HubbardChain(L=5, U=2.0), (3, 2), "free", 2),                               # odd chain
    (HubbardLattice(Lx=3, W=2, U=4.0), (3, 3), "free", 2),                       # ladder
]
IDS = ["chain", "ladder", "strip", "cylinder", "free-chain", "free-ladder"]


def _case(engine, model, sector, init, layers, svd):
    from quantx_engine import hva
    lay = hva.layout(model, engine=engine)
    angles = _angles(layers, lay.angles_per_layer, seed=7 * layers + sum(sector))
    zero = (len(angles) - 1, lay.angles_per_layer // 2)
    angles[zero[0]][zero[1]] = 0.0      # a zero angle skips its gates, not its derivative
    r = hva.circuit(model, angles, n_up=sector[0], n_dn=sector[1], init=init,
                    max_bond=256, cutoff=1e-14, entropy=False, gradient=True, svd=svd,
                    engine=engine)
    energy, grad = _exact_gradient(_lattice(model), lay.groups, angles, *sector, init,
                                   r.init_config)
    return r, energy, grad, zero


@pytest.mark.parametrize("model, sector, init, layers", CASES, ids=IDS)
def test_gradient_matches_the_state_vector(engine, model, sector, init, layers):
    r, energy, grad, zero = _case(engine, model, sector, init, layers, "direct")
    tol = TOL if init == "product" else FREE_TOL
    assert abs(r.energy - energy) < tol
    assert np.allclose(r.gradient, grad, atol=tol, rtol=0), (r.gradient, grad)
    assert abs(grad[zero[0]][zero[1]]) > 1e-3       # the skipped group's derivative
    assert r.gradient_return_error < 1e-12


@pytest.mark.parametrize("model, sector, init, layers", CASES[:4], ids=IDS[:4])
def test_the_default_route_stays_within_its_rounding(engine, model, sector, init, layers):
    r, energy, grad, _ = _case(engine, model, sector, init, layers, "gram")
    assert abs(r.energy - energy) < GRAM_E_TOL
    assert np.allclose(r.gradient, grad, atol=GRAM_TOL, rtol=0), (r.gradient, grad)


@pytest.mark.parametrize("model, sector, layers",
                         [(HubbardLattice(Lx=6, W=1, U=4.0), (3, 3), 2),
                          (HubbardLattice(Lx=3, W=2, U=4.0), (3, 3), 2),
                          (HubbardLattice(Lx=2, W=3, U=4.0, y_periodic=True), (3, 3), 2)],
                         ids=["chain", "ladder", "cylinder"])
def test_gradient_matches_central_differences(engine, model, sector, layers):
    """The done-when: every component against the five-point central difference."""
    from quantx_engine import hva
    lay = hva.layout(model, engine=engine)
    angles = np.array(_angles(layers, lay.angles_per_layer, seed=3))
    kw = dict(n_up=sector[0], n_dn=sector[1], init="product", max_bond=256, cutoff=1e-14,
              entropy=False, svd="direct", engine=engine)
    r = hva.circuit(model, angles.tolist(), gradient=True, **kw)

    def energy(a):
        return hva.circuit(model, a.tolist(), **kw).energy

    fd = np.zeros_like(angles)
    for idx in np.ndindex(*angles.shape):
        e = {}
        for m in (-2, -1, 1, 2):
            a = angles.copy()
            a[idx] += m * H_FD
            e[m] = energy(a)
        fd[idx] = (e[-2] - 8 * e[-1] + 8 * e[1] - e[2]) / (12 * H_FD)
    assert np.abs(np.array(r.gradient) - fd).max() < FD_TOL, (r.gradient, fd)


def test_the_direct_route_is_smooth(engine):
    """On a chain whose bonds fill up, second differences of the energy at a step of 1e-6
    are f'' h^2 ~ 1e-11 plus the rounding: about 5e-11 of it on the Gram route, 1e-14 on
    the direct one (measured)."""
    from quantx_engine import hva
    model = HubbardLattice(Lx=6, W=1, U=4.0)
    angles = np.array(_angles(2, hva.layout(model, engine=engine).angles_per_layer, seed=3))
    es = []
    for m in range(12):
        a = angles.copy()
        a[0, 1] += m * 1e-6
        es.append(hva.circuit(model, a.tolist(), init="product", max_bond=256, cutoff=1e-14,
                              entropy=False, svd="direct", engine=engine).energy)
    d2 = np.diff(es, 2)
    assert np.std(d2 - d2.mean()) < 1e-12


def test_the_gradient_leaves_the_energy_alone(engine):
    from quantx_engine import hva
    model = HubbardLattice(Lx=3, W=2, U=4.0)
    angles = _angles(2, hva.layout(model, engine=engine).angles_per_layer, seed=1)
    a = hva.circuit(model, angles, init="product", engine=engine)
    b = hva.circuit(model, angles, init="product", gradient=True, engine=engine)
    assert a.energy == b.energy
    assert a.gradient is None and a.gradient_return_error is None
    assert a.gradient_truncation_error is None
    assert np.array(b.gradient).shape == np.array(angles).shape
    with pytest.raises(hva.HVAError, match="svd must be"):
        hva.circuit(model, angles, init="product", svd="qr", engine=engine)


def test_an_optimizer_takes_the_gradient(engine):
    """L-BFGS-B on the energy and its adjoint gradient: a 2-layer ladder circuit from a
    product start goes downhill to a point where the gradient vanishes."""
    from scipy.optimize import minimize

    from quantx_engine import hva
    model = HubbardLattice(Lx=2, W=2, U=4.0)
    per = hva.layout(model, engine=engine).angles_per_layer

    def f(x):
        r = hva.circuit(model, x.reshape(2, per).tolist(), init="product", cutoff=1e-14,
                        entropy=False, gradient=True, engine=engine)
        return r.energy, np.ravel(r.gradient)

    x0 = np.full(2 * per, 0.1)
    res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"gtol": 1e-8})
    # measured: -0.233 to -0.846 in 44 steps, the largest gradient component 3e-6 there
    # (a product start cannot reach the singlet ground state, -2.103)
    assert res.success and res.fun < f(x0)[0] - 0.5
    assert np.abs(f(res.x)[1]).max() < 1e-4


def test_a_truncated_round_trip_says_so(engine):
    """At a bond dimension below the state's, the backward pass cannot return to the start
    exactly, and gradient_return_error reports it."""
    from quantx_engine import hva
    model = HubbardLattice(Lx=8, W=1, U=4.0)
    angles = _angles(3, hva.layout(model, engine=engine).angles_per_layer, seed=5)
    exact = hva.circuit(model, angles, init="product", max_bond=256, cutoff=1e-14,
                        entropy=False, gradient=True, engine=engine)
    cut = hva.circuit(model, angles, init="product", max_bond=8, cutoff=1e-14,
                      entropy=False, gradient=True, engine=engine)
    assert exact.gradient_return_error < 1e-12 and exact.gradient_truncation_error < 1e-12
    assert cut.truncation_error > 1e-6 and cut.gradient_return_error > 1e-6
    assert cut.gradient_truncation_error > 1e-6
