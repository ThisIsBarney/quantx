"""Quantum Boltzmann Machine for generative quantum machine learning.
References
----------
.. [1] Benedetti, M. et al. (2016). Estimation of effective temperatures in
       quantum annealers for sampling applications: A case study with possible
       applications in deep learning. *Phys. Rev. A* 94, 022308.
       https://doi.org/10.1103/PhysRevA.94.022308

.. [2] Kieferova, M. & Wiebe, N. (2017). Tomography and generative training
       with quantum Boltzmann machines. *Phys. Rev. A* 96, 062327.
       https://doi.org/10.1103/PhysRevA.96.062327

.. [3] Amin, M. H. et al. (2018). Quantum Boltzmann machine.
       *Phys. Rev. X* 8, 021050.
       https://doi.org/10.1103/PhysRevX.8.021050
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

__all__ = ["QBMError", "RBM", "QuantumBoltzmannMachine", "BoltzmannSampler"]


class QBMError(ValueError):
    """Raised for invalid QBM inputs or unsupported configurations."""


@dataclass
class RBM:
    n_visible: int
    n_hidden: int
    seed: int | None = None
    # These are set in __post_init__:
    W: "np.ndarray" = field(init=False, repr=False)
    a: "np.ndarray" = field(init=False, repr=False)  # visible biases
    b: "np.ndarray" = field(init=False, repr=False)  # hidden biases

    def __post_init__(self):
        if self.n_visible < 1 or self.n_hidden < 1:
            raise QBMError("n_visible and n_hidden must be >= 1")
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, 0.01, (self.n_visible, self.n_hidden))
        self.a = np.zeros(self.n_visible)
        self.b = np.zeros(self.n_hidden)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def p_h_given_v(self, v):
        return self._sigmoid(self.b + v @ self.W)  # shape (n_hidden,)

    def p_v_given_h(self, h):
        return self._sigmoid(self.a + h @ self.W.T)  # shape (n_visible,)

    def sample_h(self, v, rng):
        p = self.p_h_given_v(v)
        return (rng.random(self.n_hidden) < p).astype(float)

    def sample_v(self, h, rng):
        p = self.p_v_given_h(h)
        return (rng.random(self.n_visible) < p).astype(float)

    def free_energy(self, v):
        x = self.b + v @ self.W  # (n_hidden,)
        return -float(v @ self.a) - float(np.sum(np.log1p(np.exp(np.clip(x, -500, 500)))))

    def reconstruct(self, v):
        h_prob = self.p_h_given_v(v)
        return self.p_v_given_h(h_prob)

    def generate(self, n_samples, n_gibbs=100, seed=None):
        rng = np.random.default_rng(seed)
        results = []
        for _ in range(n_samples):
            v = (rng.random(self.n_visible) < 0.5).astype(float)
            for _ in range(n_gibbs):
                h = self.sample_h(v, rng)
                v = self.sample_v(h, rng)
            results.append(v)
        return np.array(results)

    def fit(self, data, n_epochs=10, learning_rate=0.01, k=1, batch_size=10, seed=None):
        if data.ndim != 2 or data.shape[1] != self.n_visible:
            raise QBMError(f"data must have shape (N, {self.n_visible})")
        rng = np.random.default_rng(seed)
        history = []
        N = len(data)
        for epoch in range(n_epochs):
            idx = rng.permutation(N)
            data_shuffled = data[idx]
            recon_errors = []
            for start in range(0, N, batch_size):
                batch = data_shuffled[start:start + batch_size]
                m = len(batch)
                # Positive phase
                h_pos = np.array([self.p_h_given_v(v) for v in batch])  # (m, n_hidden)
                pos_W = batch.T @ h_pos / m
                pos_a = batch.mean(axis=0)
                pos_b = h_pos.mean(axis=0)
                # Negative phase: k Gibbs steps
                v_neg = batch.copy()
                for _ in range(k):
                    h_neg_sample = np.array([self.sample_h(v, rng) for v in v_neg])
                    v_neg = np.array([self.sample_v(h, rng) for h in h_neg_sample])
                h_neg = np.array([self.p_h_given_v(v) for v in v_neg])
                neg_W = v_neg.T @ h_neg / m
                neg_a = v_neg.mean(axis=0)
                neg_b = h_neg.mean(axis=0)
                # Updates
                self.W += learning_rate * (pos_W - neg_W)
                self.a += learning_rate * (pos_a - neg_a)
                self.b += learning_rate * (pos_b - neg_b)
                # Reconstruction error for this batch
                recon = np.array([self.reconstruct(v) for v in batch])
                recon_errors.append(float(np.mean((batch - recon) ** 2)))
            history.append({"epoch": epoch, "reconstruction_error": float(np.mean(recon_errors))})
        return history

    def to_dict(self):
        return {
            "n_visible": self.n_visible,
            "n_hidden": self.n_hidden,
            "n_params": self.n_visible + self.n_hidden + self.n_visible * self.n_hidden,
        }


@dataclass
class QuantumBoltzmannMachine:
    n_visible: int
    n_hidden: int
    gamma: float = 0.5
    seed: int | None = None
    W: "np.ndarray" = field(init=False, repr=False)
    a: "np.ndarray" = field(init=False, repr=False)
    b: "np.ndarray" = field(init=False, repr=False)

    def __post_init__(self):
        if self.n_visible < 1 or self.n_hidden < 1:
            raise QBMError("n_visible and n_hidden must be >= 1")
        if self.gamma < 0:
            raise QBMError("gamma must be >= 0")
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0, 0.01, (self.n_visible, self.n_hidden))
        self.a = np.zeros(self.n_visible)
        self.b = np.zeros(self.n_hidden)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _quantum_p_h(self, x):
        """Quantum P(h=1) for effective field vector x."""
        if self.gamma == 0.0:
            return self._sigmoid(x)
        eff = np.sqrt(x**2 + self.gamma**2)
        return 0.5 * (1.0 + x / eff * np.tanh(eff))

    def p_h_given_v(self, v):
        x = self.b + v @ self.W
        return self._quantum_p_h(x)

    def p_v_given_h(self, h):
        return self._sigmoid(self.a + h @ self.W.T)

    def sample_h(self, v, rng):
        p = self.p_h_given_v(v)
        return (rng.random(self.n_hidden) < p).astype(float)

    def sample_v(self, h, rng):
        p = self.p_v_given_h(h)
        return (rng.random(self.n_visible) < p).astype(float)

    def free_energy(self, v):
        x = self.b + v @ self.W
        eff = np.sqrt(x**2 + self.gamma**2)
        return -float(v @ self.a) - float(np.sum(np.log(2.0 * np.cosh(eff))))

    def reconstruct(self, v):
        h_prob = self.p_h_given_v(v)
        return self.p_v_given_h(h_prob)

    def generate(self, n_samples, n_gibbs=100, seed=None):
        rng = np.random.default_rng(seed)
        results = []
        for _ in range(n_samples):
            v = (rng.random(self.n_visible) < 0.5).astype(float)
            for _ in range(n_gibbs):
                h = self.sample_h(v, rng)
                v = self.sample_v(h, rng)
            results.append(v)
        return np.array(results)

    def fit(self, data, n_epochs=10, learning_rate=0.01, k=1, batch_size=10, seed=None):
        if data.ndim != 2 or data.shape[1] != self.n_visible:
            raise QBMError(f"data must have shape (N, {self.n_visible})")
        rng = np.random.default_rng(seed)
        history = []
        N = len(data)
        for epoch in range(n_epochs):
            idx = rng.permutation(N)
            data_shuffled = data[idx]
            recon_errors = []
            for start in range(0, N, batch_size):
                batch = data_shuffled[start:start + batch_size]
                m = len(batch)
                h_pos = np.array([self.p_h_given_v(v) for v in batch])
                pos_W = batch.T @ h_pos / m
                pos_a = batch.mean(axis=0)
                pos_b = h_pos.mean(axis=0)
                v_neg = batch.copy()
                for _ in range(k):
                    h_neg_sample = np.array([self.sample_h(v, rng) for v in v_neg])
                    v_neg = np.array([self.sample_v(h, rng) for h in h_neg_sample])
                h_neg = np.array([self.p_h_given_v(v) for v in v_neg])
                neg_W = v_neg.T @ h_neg / m
                neg_a = v_neg.mean(axis=0)
                neg_b = h_neg.mean(axis=0)
                self.W += learning_rate * (pos_W - neg_W)
                self.a += learning_rate * (pos_a - neg_a)
                self.b += learning_rate * (pos_b - neg_b)
                recon = np.array([self.reconstruct(v) for v in batch])
                recon_errors.append(float(np.mean((batch - recon) ** 2)))
            history.append({"epoch": epoch, "reconstruction_error": float(np.mean(recon_errors))})
        return history

    def to_dict(self):
        return {
            "n_visible": self.n_visible,
            "n_hidden": self.n_hidden,
            "gamma": self.gamma,
            "n_params": self.n_visible + self.n_hidden + self.n_visible * self.n_hidden,
        }


@dataclass
class BoltzmannSampler:
    n_spins: int
    beta: float = 1.0
    seed: int | None = None
    _J: "np.ndarray | None" = field(default=None, init=False, repr=False)
    _h: "np.ndarray | None" = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.n_spins < 1:
            raise QBMError("n_spins must be >= 1")
        if self.beta <= 0:
            raise QBMError("beta must be > 0")

    def set_ising(self, J, h):
        J = np.asarray(J, dtype=float)
        h = np.asarray(h, dtype=float)
        if J.shape != (self.n_spins, self.n_spins):
            raise QBMError(f"J must have shape ({self.n_spins}, {self.n_spins})")
        if h.shape != (self.n_spins,):
            raise QBMError(f"h must have shape ({self.n_spins},)")
        self._J = J
        self._h = h

    def _energy(self, s):
        return -0.5 * float(s @ self._J @ s) - float(self._h @ s)

    def metropolis_sample(self, n_samples, n_burnin=1000, thin=10):
        if self._J is None:
            raise QBMError("Call set_ising() first")
        rng = np.random.default_rng(self.seed)
        s = rng.choice([-1.0, 1.0], size=self.n_spins)
        E = self._energy(s)
        total = n_burnin + n_samples * thin
        samples = []
        count = 0
        for step in range(total):
            i = rng.integers(0, self.n_spins)
            s_new = s.copy()
            s_new[i] *= -1
            E_new = self._energy(s_new)
            dE = E_new - E
            if dE < 0 or rng.random() < np.exp(-self.beta * dE):
                s = s_new
                E = E_new
            if step >= n_burnin and (step - n_burnin) % thin == 0:
                samples.append(s.copy())
        return np.array(samples)

    def exact_sample(self, n_samples, seed=None):
        if self.n_spins > 20:
            raise QBMError("exact_sample only supported for n_spins <= 20")
        if self._J is None:
            raise QBMError("Call set_ising() first")
        rng = np.random.default_rng(seed)
        n = self.n_spins
        configs = np.array([[(((i >> j) & 1) * 2 - 1) for j in range(n)] for i in range(2**n)], dtype=float)
        energies = np.array([self._energy(s) for s in configs])
        log_weights = -self.beta * energies
        log_weights -= log_weights.max()
        weights = np.exp(log_weights)
        weights /= weights.sum()
        idx = rng.choice(2**n, size=n_samples, p=weights)
        return configs[idx]

    def partition_function(self):
        if self.n_spins > 20:
            raise QBMError("partition_function only supported for n_spins <= 20")
        if self._J is None:
            raise QBMError("Call set_ising() first")
        n = self.n_spins
        configs = np.array([[(((i >> j) & 1) * 2 - 1) for j in range(n)] for i in range(2**n)], dtype=float)
        energies = np.array([self._energy(s) for s in configs])
        return float(np.sum(np.exp(-self.beta * energies)))

    def to_dict(self):
        return {"n_spins": self.n_spins, "beta": self.beta}
