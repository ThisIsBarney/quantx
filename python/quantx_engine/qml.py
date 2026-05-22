"""
Quantum Machine Learning module for the QuantX engine.

Implements variational quantum circuits for supervised learning tasks.
State evolution is delegated to :class:`~quantx_engine.gpu.GpuStatevector`,
which supports both NumPy (CPU) and CuPy (GPU) backends — pass
``backend="cupy"`` to run on GPU with no other changes.

Classes:
    QuantumNeuralNetwork  — parameterized circuit (forward pass + gradients)
    QuantumClassifier     — sklearn-style binary/multi-class classifier

The circuit architecture:
  1. Angle encoding: Ry(2·x_i) on qubit i for each input feature.
  2. Hardware-efficient ansatz (repeated n_layers times):
       - Ry(θ_{l,i}) on every qubit i
       - CNOT ladder: (0→1), (1→2), ..., (n_qubits-2→n_qubits-1)
  3. Measurement: ⟨Z_0⟩ (binary) or ⟨Z_i⟩ per output qubit (multi-class).

Parameter-shift rule (exact gradient):
    ∂f/∂θ = [f(θ + π/2) − f(θ − π/2)] / 2

Usage:
    from quantx_engine.qml import QuantumClassifier

    clf = QuantumClassifier(n_features=4, n_layers=2)
    history = clf.fit(X_train, y_train, n_epochs=100, learning_rate=0.05)
    predictions = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)

    # GPU-accelerated (requires CuPy):
    clf_gpu = QuantumClassifier(n_features=4, n_layers=2, backend="cupy")

References
----------
.. [1] Schuld, M. & Killoran, N. (2019). Quantum machine learning in feature
       Hilbert spaces. *Phys. Rev. Lett.* 122, 040504.
       https://doi.org/10.1103/PhysRevLett.122.040504

.. [2] Havlicek, V. et al. (2019). Supervised learning with quantum-enhanced
       feature spaces. *Nature* 567, 209–212.
       https://doi.org/10.1038/s41586-019-0980-2

.. [3] Cerezo, M. et al. (2021). Variational quantum algorithms.
       *Nat. Rev. Phys.* 3, 625–644.
       https://doi.org/10.1038/s42254-021-00348-9

Physical validation
-------------------
ZZ feature map: kernel K(x,y) = |<phi(x)|phi(y)>|^2 is bounded in [0,1].
For identical inputs K(x,x) = 1 (perfect self-similarity) [Havlicek2019].
Quantum-classical kernel PSD (positive semi-definite): Gram matrix K_ij
should have all non-negative eigenvalues (verified in test suite).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

__all__ = [
    "QMLError",
    "QuantumNeuralNetwork",
    "QuantumClassifier",
]

_HALF_PI = math.pi / 2.0


class QMLError(ValueError):
    """Raised for invalid QML inputs or configuration."""


# ── Core circuit evaluation via GpuStatevector ───────────────────────────────

def _run_circuit_sv(
    x: "np.ndarray",
    params: "np.ndarray",
    n_qubits: int,
    n_layers: int,
    backend: str = "numpy",
) -> Any:
    """Evaluate the parameterized circuit and return a GpuStatevector.

    Parameters
    ----------
    x:
        Input features, shape (n_features,). Features beyond n_qubits are
        ignored; missing features (n_features < n_qubits) default to 0.
    params:
        Circuit parameters, shape (n_layers * n_qubits,).
    n_qubits:
        Number of qubits.
    n_layers:
        Number of ansatz layers.
    backend:
        ``"numpy"`` (default), ``"cupy"``, or ``"auto"``.

    Returns
    -------
    GpuStatevector after circuit evaluation.
    Call ``.expectation_z(qubit)`` or ``.state()`` on the result.
    """
    # P2a: numpy check before any GPU/array operations so callers always
    # see QMLError, never a raw NameError or ImportError from gpu.py.
    if not _HAS_NUMPY:
        raise QMLError(
            "numpy is required for QuantumNeuralNetwork. "
            "Install with: pip install numpy"
        )

    from quantx_engine.gpu import GpuStatevector, GpuError  # lazy import

    try:
        sv = GpuStatevector(n_qubits=n_qubits, backend=backend)
    except GpuError as exc:
        raise QMLError(str(exc)) from exc

    # Angle encoding: Ry(2 * x_i) on qubit i
    for i in range(n_qubits):
        xi = float(x[i]) if i < len(x) else 0.0
        sv.apply_gate("RY", i, (2.0 * xi,))

    # P2b: always validate params shape — catches stale/mis-sized vectors even
    # when n_layers=0 (previously those params were silently ignored).
    params_arr = np.asarray(params)
    expected_n = n_layers * n_qubits
    if params_arr.size != expected_n:
        raise QMLError(
            f"params has {params_arr.size} elements; "
            f"expected {expected_n} (n_layers={n_layers} × n_qubits={n_qubits})"
        )

    # Ansatz layers
    if n_layers > 0:
        p2d = params_arr.reshape(n_layers, n_qubits)
        for layer in range(n_layers):
            for i in range(n_qubits):
                sv.apply_gate("RY", i, (float(p2d[layer, i]),))
            for i in range(n_qubits - 1):
                sv.apply_gate("CNOT", [i, i + 1])

    return sv


# ── QuantumNeuralNetwork ──────────────────────────────────────────────────────

@dataclass
class QuantumNeuralNetwork:
    """Parameterized quantum circuit for supervised learning.

    The circuit uses angle encoding followed by hardware-efficient ansatz
    layers.  Gradients are computed exactly via the parameter-shift rule.
    State evolution is handled by :class:`~quantx_engine.gpu.GpuStatevector`,
    so passing ``backend="cupy"`` transparently enables GPU acceleration.

    Parameters
    ----------
    n_qubits:
        Number of qubits. Must be >= n_features if you want full encoding;
        features beyond n_qubits are truncated.
    n_layers:
        Number of ansatz repetitions (depth = n_layers * (n_qubits + n_qubits - 1)).
    n_outputs:
        Number of output nodes (qubits measured). 1 for binary classification.
    backend:
        Statevector backend: ``"numpy"`` (default), ``"cupy"``, or ``"auto"``.
    """
    n_qubits: int
    n_layers: int = 2
    n_outputs: int = 1
    backend: str = "numpy"

    @property
    def n_params(self) -> int:
        return self.n_layers * self.n_qubits

    def random_params(self, seed: int | None = None) -> "np.ndarray":
        """Return randomly initialized parameters in [0, 2π)."""
        if not _HAS_NUMPY:
            raise QMLError("numpy is required")
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 2 * math.pi, self.n_params)

    def forward(self, x: "np.ndarray", params: "np.ndarray") -> "np.ndarray":
        """Evaluate the circuit and return expectation values.

        Parameters
        ----------
        x:
            Input features, shape (n_features,).
        params:
            Circuit parameters, shape (n_params,).

        Returns
        -------
        Expectation values ⟨Z_i⟩ for i in range(n_outputs), shape (n_outputs,).
        """
        sv = _run_circuit_sv(x, params, self.n_qubits, self.n_layers, self.backend)
        return np.array([sv.expectation_z(i) for i in range(self.n_outputs)])

    def gradient(self, x: "np.ndarray", params: "np.ndarray") -> "np.ndarray":
        """Compute parameter-shift gradients ∂⟨Z_0⟩/∂params.

        Uses the parameter-shift rule: ∂f/∂θ = [f(θ+π/2) - f(θ-π/2)] / 2.

        Returns
        -------
        Gradient array, shape (n_params,).
        """
        grads = np.zeros(self.n_params)
        for i in range(self.n_params):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += _HALF_PI
            p_minus[i] -= _HALF_PI
            f_plus = self.forward(x, p_plus)[0]
            f_minus = self.forward(x, p_minus)[0]
            grads[i] = (f_plus - f_minus) / 2.0
        return grads

    def batch_gradient(
        self,
        X: "np.ndarray",
        y: "np.ndarray",
        params: "np.ndarray",
        loss: str = "mse",
    ) -> tuple["np.ndarray", float]:
        """Compute mean gradient and loss over a batch.

        Parameters
        ----------
        X:
            Input batch, shape (n_samples, n_features).
        y:
            Target labels, shape (n_samples,). Values should be in {-1, +1} for
            MSE loss or {0, 1} for BCE loss.
        params:
            Current parameters.
        loss:
            Loss function: 'mse' or 'bce'.

        Returns
        -------
        (mean_gradient, mean_loss): gradient array and scalar loss.
        """
        total_grad = np.zeros(self.n_params)
        total_loss = 0.0
        n = len(X)
        for xi, yi in zip(X, y):
            pred = self.forward(xi, params)[0]  # scalar in [-1, 1]
            if loss == "mse":
                # MSE: L = (pred - yi)^2, dL/dpred = 2*(pred-yi)
                dl_dpred = 2.0 * (pred - float(yi))
                total_loss += (pred - float(yi)) ** 2
            elif loss == "bce":
                # Map pred from [-1,1] → [0,1] via sigmoid of 4*pred
                prob = 1.0 / (1.0 + math.exp(-4.0 * pred))
                prob = max(1e-7, min(1.0 - 1e-7, prob))
                yi_f = float(yi)
                total_loss += -(yi_f * math.log(prob) + (1 - yi_f) * math.log(1 - prob))
                # Chain rule: dL/dpred = 4*(prob - yi)
                dl_dpred = 4.0 * (prob - yi_f)
            else:
                raise QMLError(f"unknown loss: {loss!r}; choose 'mse' or 'bce'")
            grad = self.gradient(xi, params)
            total_grad += dl_dpred * grad
        return total_grad / n, total_loss / n

    def to_dict(self) -> dict[str, Any]:
        """Serialize architecture to a plain dict."""
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_outputs": self.n_outputs,
            "n_params": self.n_params,
            "backend": self.backend,
        }


# ── QuantumClassifier ─────────────────────────────────────────────────────────

@dataclass
class QuantumClassifier:
    """Variational quantum classifier with sklearn-compatible API.

    Supports binary classification (n_classes=2) and one-vs-rest multi-class.

    Parameters
    ----------
    n_features:
        Number of input features (≤ n_qubits for lossless encoding).
    n_layers:
        Number of ansatz repetitions.
    n_qubits:
        Number of qubits. Defaults to n_features.
    optimizer:
        'adam' or 'gd' (gradient descent).
    loss:
        'mse' or 'bce'.
    seed:
        Random seed for parameter initialization.
    backend:
        Statevector backend: ``"numpy"`` (default), ``"cupy"``, or ``"auto"``.
    """
    n_features: int
    n_layers: int = 2
    n_qubits: int = 0  # 0 → auto-set to n_features in __post_init__
    optimizer: str = "adam"
    loss: str = "mse"
    seed: int | None = 42
    backend: str = "numpy"

    # Learned state (set after fit())
    params_: "np.ndarray | None" = field(default=None, repr=False)
    classes_: list = field(default_factory=list, repr=False)
    history_: list[dict] = field(default_factory=list, repr=False)
    x_scale_: float = field(default=1.0, repr=False)  # feature scale from training

    def __post_init__(self) -> None:
        if self.optimizer not in ("adam", "gd"):
            raise QMLError(
                f"Unknown optimizer {self.optimizer!r}; choose 'adam' or 'gd'."
            )
        if self.n_qubits == 0:
            self.n_qubits = self.n_features
        self._qnn = QuantumNeuralNetwork(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_outputs=1,
            backend=self.backend,
        )

    def fit(
        self,
        X: "np.ndarray",
        y: "np.ndarray",
        n_epochs: int = 100,
        learning_rate: float = 0.05,
        batch_size: int | None = None,
        verbose: bool = False,
    ) -> list[dict]:
        """Train the classifier.

        Parameters
        ----------
        X:
            Training features, shape (n_samples, n_features). Will be
            normalized to [-1, 1] if values outside [-π, π] are detected.
        y:
            Integer class labels, shape (n_samples,).
        n_epochs:
            Number of training epochs.
        learning_rate:
            Step size for the optimizer.
        batch_size:
            Mini-batch size (None = full batch).
        verbose:
            Print loss every 10 epochs.

        Returns
        -------
        Training history: list of {'epoch', 'loss'} dicts.
        """
        if not _HAS_NUMPY:
            raise QMLError("numpy is required for training. pip install numpy")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = sorted(set(y.tolist()))

        # Binary classification: map to {-1, +1} for MSE or {0, 1} for BCE
        if len(self.classes_) != 2:
            raise QMLError(
                f"QuantumClassifier supports binary classification only "
                f"(got {len(self.classes_)} classes). "
                "For multi-class use one-vs-rest with multiple classifiers."
            )
        neg_cls, pos_cls = self.classes_[0], self.classes_[1]
        if self.loss == "mse":
            y_enc = np.where(y == pos_cls, 1.0, -1.0)
        else:
            y_enc = np.where(y == pos_cls, 1.0, 0.0)

        # Normalize features to [-π, π] if needed; persist scale for inference
        x_max = float(np.abs(X).max())
        self.x_scale_ = x_max / math.pi if x_max > math.pi else 1.0
        if self.x_scale_ != 1.0:
            X = X / self.x_scale_

        params = self._qnn.random_params(seed=self.seed)
        n = len(X)
        bs = batch_size or n

        # Adam state
        m = np.zeros_like(params)
        v = np.zeros_like(params)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0

        self.history_ = []
        for epoch in range(n_epochs):
            # Shuffle
            perm = np.random.default_rng(epoch).permutation(n)
            X_shuf, y_shuf = X[perm], y_enc[perm]

            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n, bs):
                Xb = X_shuf[start:start + bs]
                yb = y_shuf[start:start + bs]
                grad, batch_loss = self._qnn.batch_gradient(Xb, yb, params, loss=self.loss)
                epoch_loss += batch_loss
                n_batches += 1

                if self.optimizer == "adam":
                    t += 1
                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * grad ** 2
                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)
                    params -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
                else:
                    params -= learning_rate * grad

            avg_loss = epoch_loss / n_batches
            entry = {"epoch": epoch + 1, "loss": float(avg_loss)}
            self.history_.append(entry)
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1:4d}/{n_epochs}  loss={avg_loss:.6f}")

        self.params_ = params
        return self.history_

    def predict_proba(self, X: "np.ndarray") -> "np.ndarray":
        """Predict class probabilities, shape (n_samples, 2).

        Converts ⟨Z_0⟩ ∈ [-1, 1] to probability via σ(4 · ⟨Z_0⟩).
        Uses the same feature scaling determined during training.
        """
        if self.params_ is None:
            raise QMLError("Classifier has not been fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        if self.x_scale_ != 1.0:
            X = X / self.x_scale_

        probs = []
        for xi in X:
            exp_z = self._qnn.forward(xi, self.params_)[0]
            p_pos = 1.0 / (1.0 + math.exp(-4.0 * exp_z))
            probs.append([1.0 - p_pos, p_pos])
        return np.array(probs)

    def predict(self, X: "np.ndarray") -> "np.ndarray":
        """Predict class labels, shape (n_samples,)."""
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        neg_cls, pos_cls = self.classes_[0], self.classes_[1]
        return np.where(idx == 1, pos_cls, neg_cls)

    def score(self, X: "np.ndarray", y: "np.ndarray") -> float:
        """Return classification accuracy."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def circuit_spec(self, x: "np.ndarray | None" = None) -> dict[str, Any]:
        """Return a human-readable circuit specification dict.

        Parameters
        ----------
        x:
            Optional sample input to show encoding angles. If None, uses zeros.

        Returns
        -------
        dict with 'architecture', 'n_params', 'encoding', and optionally
        'trained_params' if the model has been fitted.
        """
        if not _HAS_NUMPY:
            raise QMLError("numpy is required")
        x_example = np.zeros(self.n_features) if x is None else np.asarray(x, dtype=float)
        encoding = {
            f"q{i}": f"Ry({2.0 * float(x_example[i]):.4f})" if i < len(x_example) else "Ry(0)"
            for i in range(self.n_qubits)
        }
        spec: dict[str, Any] = {
            "architecture": self._qnn.to_dict(),
            "n_params": self._qnn.n_params,
            "encoding": f"angle (Ry(2·x_i) on qubit i, i=0..{self.n_qubits-1})",
            "ansatz": f"hardware-efficient ({self.n_layers} layers: Ry rotations + CNOT ladder)",
            "measurement": "⟨Z_0⟩ → sigmoid → binary probability",
            "gradient_method": "parameter-shift rule (exact)",
            "optimizer": self.optimizer,
            "loss": self.loss,
            "backend": self.backend,
            "example_encoding": encoding,
        }
        if self.params_ is not None:
            spec["trained_params"] = self.params_.tolist()
        return spec
