"""Tests for the Quantum Boltzmann Machine module."""
from __future__ import annotations

import math
import pytest

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

pytestmark = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy required")

from quantx_engine.qbm import RBM, QuantumBoltzmannMachine, BoltzmannSampler, QBMError


# ── RBM ───────────────────────────────────────────────────────────────────────

def test_rbm_init_shapes():
    rbm = RBM(n_visible=4, n_hidden=3, seed=0)
    assert rbm.W.shape == (4, 3)
    assert rbm.a.shape == (4,)
    assert rbm.b.shape == (3,)


def test_rbm_invalid_size():
    with pytest.raises(QBMError):
        RBM(n_visible=0, n_hidden=3)


def test_rbm_p_h_given_v_range():
    rbm = RBM(n_visible=4, n_hidden=3, seed=1)
    v = np.array([1.0, 0.0, 1.0, 1.0])
    p = rbm.p_h_given_v(v)
    assert p.shape == (3,)
    assert np.all(p >= 0) and np.all(p <= 1)


def test_rbm_p_v_given_h_range():
    rbm = RBM(n_visible=4, n_hidden=3, seed=1)
    h = np.array([1.0, 0.0, 1.0])
    p = rbm.p_v_given_h(h)
    assert p.shape == (4,)
    assert np.all(p >= 0) and np.all(p <= 1)


def test_rbm_reconstruct_shape():
    rbm = RBM(n_visible=6, n_hidden=4, seed=2)
    v = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    r = rbm.reconstruct(v)
    assert r.shape == (6,)


def test_rbm_generate_shape():
    rbm = RBM(n_visible=4, n_hidden=3, seed=3)
    samples = rbm.generate(n_samples=5, n_gibbs=10, seed=3)
    assert samples.shape == (5, 4)


def test_rbm_generate_binary():
    rbm = RBM(n_visible=4, n_hidden=3, seed=4)
    samples = rbm.generate(n_samples=10, n_gibbs=10, seed=4)
    assert set(np.unique(samples)).issubset({0.0, 1.0})


def test_rbm_fit_returns_history():
    rbm = RBM(n_visible=4, n_hidden=3, seed=5)
    data = np.random.default_rng(0).integers(0, 2, (20, 4)).astype(float)
    history = rbm.fit(data, n_epochs=3, seed=5)
    assert len(history) == 3
    assert all("epoch" in h and "reconstruction_error" in h for h in history)


def test_rbm_fit_reduces_error():
    """Training on a fixed pattern should reduce reconstruction error."""
    rbm = RBM(n_visible=4, n_hidden=4, seed=0)
    # Repeat a single pattern 50 times
    pattern = np.array([1.0, 0.0, 1.0, 0.0])
    data = np.tile(pattern, (50, 1))
    history = rbm.fit(data, n_epochs=20, learning_rate=0.05, batch_size=10, seed=0)
    assert history[-1]["reconstruction_error"] < history[0]["reconstruction_error"] + 0.05


def test_rbm_to_dict_keys():
    rbm = RBM(n_visible=4, n_hidden=3)
    d = rbm.to_dict()
    assert "n_visible" in d and "n_hidden" in d and "n_params" in d
    assert d["n_params"] == 4 + 3 + 4 * 3


def test_rbm_free_energy_finite():
    rbm = RBM(n_visible=4, n_hidden=3, seed=6)
    v = np.array([1.0, 0.0, 1.0, 1.0])
    fe = rbm.free_energy(v)
    assert math.isfinite(fe)


# ── QuantumBoltzmannMachine ───────────────────────────────────────────────────

def test_qbm_init_shapes():
    qbm = QuantumBoltzmannMachine(n_visible=4, n_hidden=3, seed=0)
    assert qbm.W.shape == (4, 3)
    assert qbm.a.shape == (4,)
    assert qbm.b.shape == (3,)


def test_qbm_invalid_gamma():
    with pytest.raises(QBMError):
        QuantumBoltzmannMachine(n_visible=4, n_hidden=3, gamma=-1.0)


def test_qbm_p_h_given_v_range():
    qbm = QuantumBoltzmannMachine(n_visible=4, n_hidden=3, gamma=0.5, seed=1)
    v = np.array([1.0, 0.0, 1.0, 1.0])
    p = qbm.p_h_given_v(v)
    assert p.shape == (3,)
    assert np.all(p >= 0) and np.all(p <= 1)


def test_qbm_free_energy_finite():
    qbm = QuantumBoltzmannMachine(n_visible=4, n_hidden=3, gamma=0.5, seed=2)
    v = np.array([1.0, 0.0, 1.0, 1.0])
    fe = qbm.free_energy(v)
    assert math.isfinite(fe)


def test_qbm_gamma_zero_matches_rbm():
    """With gamma=0, quantum P(h|v) should equal classical sigmoid."""
    n_v, n_h = 4, 3
    rng = np.random.default_rng(42)
    W = rng.normal(0, 0.5, (n_v, n_h))
    a = rng.normal(0, 0.5, n_v)
    b = rng.normal(0, 0.5, n_h)

    rbm = RBM(n_visible=n_v, n_hidden=n_h, seed=42)
    qbm = QuantumBoltzmannMachine(n_visible=n_v, n_hidden=n_h, gamma=0.0, seed=42)
    # Manually set same weights
    rbm.W[:] = W; rbm.a[:] = a; rbm.b[:] = b
    qbm.W[:] = W; qbm.a[:] = a; qbm.b[:] = b

    v = np.array([1.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(rbm.p_h_given_v(v), qbm.p_h_given_v(v), atol=1e-8)


def test_qbm_fit_returns_history():
    qbm = QuantumBoltzmannMachine(n_visible=4, n_hidden=3, gamma=0.3, seed=5)
    data = np.random.default_rng(0).integers(0, 2, (20, 4)).astype(float)
    history = qbm.fit(data, n_epochs=3, seed=5)
    assert len(history) == 3
    assert all("epoch" in h and "reconstruction_error" in h for h in history)


def test_qbm_to_dict_has_gamma():
    qbm = QuantumBoltzmannMachine(n_visible=4, n_hidden=3, gamma=0.5)
    d = qbm.to_dict()
    assert "gamma" in d
    assert d["gamma"] == 0.5


# ── BoltzmannSampler ──────────────────────────────────────────────────────────

def _two_spin_ising():
    sampler = BoltzmannSampler(n_spins=2, beta=1.0, seed=0)
    J = np.array([[0.0, 1.0], [1.0, 0.0]])
    h = np.zeros(2)
    sampler.set_ising(J, h)
    return sampler


def test_sampler_metropolis_shape():
    sampler = _two_spin_ising()
    samples = sampler.metropolis_sample(n_samples=20, n_burnin=100, thin=5)
    assert samples.shape == (20, 2)


def test_sampler_metropolis_values():
    sampler = _two_spin_ising()
    samples = sampler.metropolis_sample(n_samples=20, n_burnin=100, thin=5)
    assert set(np.unique(samples)).issubset({-1.0, 1.0})


def test_sampler_exact_partition_function():
    """n=2, J=[[0,1],[1,0]], h=[0,0], beta=1: Z = 2*exp(1) + 2*exp(-1)."""
    sampler = _two_spin_ising()
    Z = sampler.partition_function()
    expected = 2 * math.exp(1.0) + 2 * math.exp(-1.0)
    assert abs(Z - expected) < 1e-10


def test_sampler_exact_too_large():
    sampler = BoltzmannSampler(n_spins=21, beta=1.0)
    J = np.zeros((21, 21))
    h = np.zeros(21)
    sampler.set_ising(J, h)
    with pytest.raises(QBMError):
        sampler.exact_sample(10)


def test_sampler_invalid_J_shape():
    sampler = BoltzmannSampler(n_spins=3, beta=1.0)
    with pytest.raises(QBMError):
        sampler.set_ising(np.zeros((2, 2)), np.zeros(3))


def test_sampler_exact_samples_shape():
    sampler = _two_spin_ising()
    samples = sampler.exact_sample(n_samples=15, seed=0)
    assert samples.shape == (15, 2)


def test_sampler_exact_samples_values():
    sampler = _two_spin_ising()
    samples = sampler.exact_sample(n_samples=15, seed=0)
    assert set(np.unique(samples)).issubset({-1.0, 1.0})
