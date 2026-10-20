"""M13 gate: the adjoint gradient of HVA circuits, against a state vector and against
central differences.

``hva.circuit(..., gradient=True)`` returns dE/dtheta for every angle from one pass back
through the circuit. It is checked two ways:

- against the exact gradient of a state-vector simulation written independently of the
  engine (the Fock-space simulator of test_hva_statevector, with the adjoint formula in
  dense linear algebra), to 1e-8 from product starts. Measured on gradients of size 3
  to 7: 1e-13 to 2e-12 on the ladder, the strip and the cylinder, 3.5e-9 on the chain,
  whose bonds fill up (see below); the energies agree to 1e-11;
- against central differences of the engine's own energies, to 1e-8: the five-point
  stencil at h = 5e-4, whose truncation error is h^4/30 times the fifth derivative.

The central differences run on the ladder and the cylinder, whose energies are smooth in
the angles to 1e-14. A chain whose bonds fill up carries a rounding noise of about 3e-11
in its energy: the circuit's SVDs go through the Gram matrix, which leaves Schmidt
values below sqrt(eps) ~ 1e-8 imprecise, and as an angle moves they shift. Divided by
h, that noise limits a five-point difference there to a few 1e-8 at any step size. The
same rounding is what puts the chain's adjoint gradient 3.5e-9 off the state vector.
"""
from __future__ import annotations

import numpy as np
import pytest

from quantx_engine.models import HubbardChain, HubbardLattice
from quantx_engine.tests.test_hva_statevector import Fock, _angles, _native_or_skip

TOL = 1e-10         # the energy against the state vector, from a product start
GRAD_TOL = 1e-8     # the gradient against the state vector, from a product start
FREE_TOL = 1e-6     # from a free start, which carries its DMRG preparation's residual
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


@pytest.mark.parametrize("model, sector, init, layers", CASES,
                         ids=["chain", "ladder", "strip", "cylinder", "free-chain",
                              "free-ladder"])
def test_gradient_matches_the_state_vector(engine, model, sector, init, layers):
    from quantx_engine import hva
    lay = hva.layout(model, engine=engine)
    angles = _angles(layers, lay.angles_per_layer, seed=7 * layers + sum(sector))
    zero = (len(angles) - 1, lay.angles_per_layer // 2)
    angles[zero[0]][zero[1]] = 0.0      # a zero angle skips its gates, not its derivative
    r = hva.circuit(model, angles, n_up=sector[0], n_dn=sector[1], init=init,
                    max_bond=256, cutoff=1e-14, entropy=False, gradient=True,
                    engine=engine)
    energy, grad = _exact_gradient(_lattice(model), lay.groups, angles, *sector, init,
                                   r.init_config)
    tol, gtol = (TOL, GRAD_TOL) if init == "product" else (FREE_TOL, FREE_TOL)
    assert abs(r.energy - energy) < tol
    assert np.allclose(r.gradient, grad, atol=gtol, rtol=0), (r.gradient, grad)
    assert abs(grad[zero[0]][zero[1]]) > 1e-3       # the skipped group's derivative
    assert r.gradient_return_error < 1e-12


@pytest.mark.parametrize("model, sector, layers",
                         [(HubbardLattice(Lx=3, W=2, U=4.0), (3, 3), 2),
                          (HubbardLattice(Lx=2, W=3, U=4.0, y_periodic=True), (3, 3), 2)],
                         ids=["ladder", "cylinder"])
def test_gradient_matches_central_differences(engine, model, sector, layers):
    """The done-when: every component against the five-point central difference."""
    from quantx_engine import hva
    lay = hva.layout(model, engine=engine)
    angles = np.array(_angles(layers, lay.angles_per_layer, seed=3))
    kw = dict(n_up=sector[0], n_dn=sector[1], init="product", max_bond=256, cutoff=1e-14,
              entropy=False, engine=engine)
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
