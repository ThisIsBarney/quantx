"""Tests for the quantum machine learning module."""
from __future__ import annotations

import math
import pytest

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

pytestmark = pytest.mark.skipif(not _HAS_NUMPY, reason="numpy required")

from quantx_engine.contrib.qml import QuantumNeuralNetwork, QuantumClassifier, QMLError


# ── QuantumNeuralNetwork ──────────────────────────────────────────────────────

def test_qnn_n_params():
    qnn = QuantumNeuralNetwork(n_qubits=3, n_layers=2)
    assert qnn.n_params == 6  # 2 layers × 3 qubits


def test_qnn_forward_shape():
    qnn = QuantumNeuralNetwork(n_qubits=2, n_layers=1, n_outputs=1)
    x = np.array([0.5, -0.3])
    params = np.zeros(qnn.n_params)
    out = qnn.forward(x, params)
    assert out.shape == (1,)
    assert -1.0 <= float(out[0]) <= 1.0


def test_qnn_forward_deterministic():
    qnn = QuantumNeuralNetwork(n_qubits=2, n_layers=1)
    x = np.array([0.1, 0.2])
    params = np.array([0.3, 0.7])
    out1 = qnn.forward(x, params)
    out2 = qnn.forward(x, params)
    np.testing.assert_array_almost_equal(out1, out2)


def test_qnn_zero_params_identity():
    """With zero params + zero input, ⟨Z_0⟩ should be +1 (all qubits in |0⟩)."""
    qnn = QuantumNeuralNetwork(n_qubits=2, n_layers=2)
    x = np.zeros(2)
    params = np.zeros(qnn.n_params)
    out = qnn.forward(x, params)
    assert abs(float(out[0]) - 1.0) < 1e-6


def test_qnn_pi_encoding_flips():
    """Encoding x=π/2 applies Ry(π) = X gate, flipping qubit 0 to |1⟩."""
    qnn = QuantumNeuralNetwork(n_qubits=1, n_layers=0)
    x = np.array([math.pi / 2])
    params = np.zeros(0)
    out = qnn.forward(x, params)
    # Ry(π) rotates |0⟩ → |1⟩, so ⟨Z⟩ ≈ -1
    assert abs(float(out[0]) + 1.0) < 1e-6


def test_qnn_gradient_shape():
    qnn = QuantumNeuralNetwork(n_qubits=2, n_layers=2)
    x = np.array([0.3, -0.1])
    params = qnn.random_params(seed=0)
    grad = qnn.gradient(x, params)
    assert grad.shape == (qnn.n_params,)


def test_qnn_parameter_shift_numerical():
    """Parameter-shift gradient should match finite-difference gradient."""
    qnn = QuantumNeuralNetwork(n_qubits=2, n_layers=1)
    x = np.array([0.4, -0.2])
    params = np.array([0.8, 1.2])
    eps = 1e-4

    ps_grad = qnn.gradient(x, params)
    fd_grad = np.zeros_like(params)
    for i in range(len(params)):
        p_plus = params.copy(); p_plus[i] += eps
        p_minus = params.copy(); p_minus[i] -= eps
        fd_grad[i] = (qnn.forward(x, p_plus)[0] - qnn.forward(x, p_minus)[0]) / (2 * eps)

    np.testing.assert_allclose(ps_grad, fd_grad, atol=1e-4)


def test_qnn_random_params_range():
    qnn = QuantumNeuralNetwork(n_qubits=3, n_layers=2)
    params = qnn.random_params(seed=42)
    assert params.shape == (qnn.n_params,)
    assert np.all(params >= 0) and np.all(params <= 2 * math.pi)


# ── QuantumClassifier ─────────────────────────────────────────────────────────

def _make_xor_dataset(n=40):
    """Simple linearly-separable dataset (not XOR, just two Gaussian blobs)."""
    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=-0.8, scale=0.3, size=(n // 2, 2))
    X1 = rng.normal(loc=+0.8, scale=0.3, size=(n // 2, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def test_classifier_fit_returns_history():
    X, y = _make_xor_dataset()
    clf = QuantumClassifier(n_features=2, n_layers=1, seed=0)
    history = clf.fit(X, y, n_epochs=5)
    assert len(history) == 5
    assert all("epoch" in h and "loss" in h for h in history)


def test_classifier_predict_shape():
    X, y = _make_xor_dataset()
    clf = QuantumClassifier(n_features=2, n_layers=1, seed=0)
    clf.fit(X, y, n_epochs=5)
    preds = clf.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds.tolist()).issubset({0, 1})


def test_classifier_predict_proba_shape():
    X, y = _make_xor_dataset()
    clf = QuantumClassifier(n_features=2, n_layers=1, seed=0)
    clf.fit(X, y, n_epochs=5)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    # Each row must sum to ~1
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-6)


def test_classifier_score():
    """After training, accuracy should beat random chance on a separable problem."""
    X, y = _make_xor_dataset(n=60)
    clf = QuantumClassifier(n_features=2, n_layers=2, seed=1)
    clf.fit(X, y, n_epochs=30, learning_rate=0.05)
    acc = clf.score(X, y)
    assert acc > 0.5, f"Expected accuracy > 0.5 on separable data, got {acc:.3f}"


def test_classifier_loss_decreases():
    """Training loss should generally decrease over epochs."""
    X, y = _make_xor_dataset()
    clf = QuantumClassifier(n_features=2, n_layers=1, loss="bce", seed=2)
    history = clf.fit(X, y, n_epochs=20, learning_rate=0.05)
    first_loss = history[0]["loss"]
    last_loss = history[-1]["loss"]
    # Should decrease at least a little; allow generous margin
    assert last_loss <= first_loss + 0.1, (
        f"Loss did not decrease: {first_loss:.4f} → {last_loss:.4f}"
    )


def test_classifier_not_fitted_raises():
    clf = QuantumClassifier(n_features=2)
    with pytest.raises(QMLError, match="not been fitted"):
        clf.predict(np.array([[0.1, 0.2]]))


def test_classifier_multiclass_raises():
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    y = np.array([0, 1, 2])
    clf = QuantumClassifier(n_features=2)
    with pytest.raises(QMLError, match="binary classification"):
        clf.fit(X, y, n_epochs=1)


def test_classifier_circuit_spec():
    X, y = _make_xor_dataset()
    clf = QuantumClassifier(n_features=2, n_layers=1, seed=0)
    clf.fit(X, y, n_epochs=2)
    spec = clf.circuit_spec(x=np.array([0.5, -0.3]))
    assert "architecture" in spec
    assert "trained_params" in spec
    assert spec["gradient_method"] == "parameter-shift rule (exact)"


def test_classifier_adam_vs_gd():
    """Both optimizers should produce a valid model."""
    X, y = _make_xor_dataset()
    clf_adam = QuantumClassifier(n_features=2, n_layers=1, optimizer="adam", seed=0)
    clf_gd = QuantumClassifier(n_features=2, n_layers=1, optimizer="gd", seed=0)
    clf_adam.fit(X, y, n_epochs=5)
    clf_gd.fit(X, y, n_epochs=5)
    assert clf_adam.params_ is not None
    assert clf_gd.params_ is not None
    # Parameters should differ (different update rules)
    assert not np.allclose(clf_adam.params_, clf_gd.params_)


def test_classifier_invalid_optimizer_raises():
    with pytest.raises(QMLError, match="optimizer"):
        QuantumClassifier(n_features=2, optimizer="sgd")


def test_classifier_consistent_inference_scale():
    """Inference with a single sample should give the same result as batch inference."""
    X, y = _make_xor_dataset()
    clf = QuantumClassifier(n_features=2, n_layers=1, seed=0)
    clf.fit(X, y, n_epochs=5)
    # Single sample vs batch — probabilities must be identical
    single = clf.predict_proba(X[0:1])
    batch = clf.predict_proba(X[:5])
    np.testing.assert_allclose(single[0], batch[0], atol=1e-8)


# ── GpuStatevector delegation tests ──────────────────────────────────────────

def test_qnn_uses_gpu_statevector_not_hand_rolled():
    """_run_circuit_sv must return a GpuStatevector, not a raw numpy array."""
    from quantx_engine.contrib.gpu import GpuStatevector
    from quantx_engine.contrib.qml import _run_circuit_sv
    sv = _run_circuit_sv(np.zeros(2), np.zeros(2), n_qubits=2, n_layers=1)
    assert isinstance(sv, GpuStatevector)


def test_qnn_no_private_apply_single():
    """The old hand-rolled helpers must no longer exist in the module."""
    import quantx_engine.contrib.qml as qml_mod
    for name in ("_ry", "_apply_single", "_apply_cnot", "_run_circuit", "_expectation_z"):
        assert not hasattr(qml_mod, name), f"{name!r} still present in qml module"


def test_qnn_backend_default_is_numpy():
    """Default backend is 'numpy'."""
    qnn = QuantumNeuralNetwork(n_qubits=2, n_layers=1)
    assert qnn.backend == "numpy"


def test_qnn_backend_numpy_matches_forward():
    """Explicit backend='numpy' gives same result as default."""
    qnn1 = QuantumNeuralNetwork(n_qubits=2, n_layers=1)
    qnn2 = QuantumNeuralNetwork(n_qubits=2, n_layers=1, backend="numpy")
    x = np.array([0.3, -0.1])
    params = np.array([0.5, 0.8])
    np.testing.assert_allclose(qnn1.forward(x, params), qnn2.forward(x, params), atol=1e-12)


def test_qnn_to_dict_includes_backend():
    """to_dict() must include the backend key."""
    qnn = QuantumNeuralNetwork(n_qubits=2, n_layers=1, backend="numpy")
    d = qnn.to_dict()
    assert "backend" in d
    assert d["backend"] == "numpy"


def test_classifier_backend_propagated_to_qnn():
    """QuantumClassifier propagates backend to its internal QNN."""
    clf = QuantumClassifier(n_features=2, backend="numpy")
    assert clf._qnn.backend == "numpy"


def test_classifier_circuit_spec_includes_backend():
    """circuit_spec() must include the backend key."""
    clf = QuantumClassifier(n_features=2, n_layers=1, seed=0)
    spec = clf.circuit_spec()
    assert "backend" in spec


def test_qnn_gpu_statevector_agrees_with_cpu():
    """numpy and (numpy-fallback) 'auto' backends produce the same output."""
    x = np.array([0.4, 0.7])
    params = np.array([1.1, 0.3])
    qnn_np = QuantumNeuralNetwork(n_qubits=2, n_layers=1, backend="numpy")
    qnn_auto = QuantumNeuralNetwork(n_qubits=2, n_layers=1, backend="auto")
    np.testing.assert_allclose(
        qnn_np.forward(x, params),
        qnn_auto.forward(x, params),
        atol=1e-10,
    )


def test_run_circuit_sv_wrong_param_size_raises():
    """P2b: stale/mis-sized params raise QMLError even when n_layers=0."""
    from quantx_engine.contrib.qml import _run_circuit_sv
    # n_layers=0 expects 0 params — passing 2 should raise
    with pytest.raises(QMLError, match="params has 2 elements"):
        _run_circuit_sv(np.zeros(2), np.array([0.1, 0.2]), n_qubits=2, n_layers=0)


def test_run_circuit_sv_wrong_param_size_nonzero_layers():
    """P2b: mis-sized params also raise for n_layers > 0."""
    from quantx_engine.contrib.qml import _run_circuit_sv
    # 2 qubits × 1 layer = 2 params expected; passing 3 should raise
    with pytest.raises(QMLError, match="params has 3 elements"):
        _run_circuit_sv(np.zeros(2), np.array([0.1, 0.2, 0.3]), n_qubits=2, n_layers=1)


def test_run_circuit_sv_zero_layers_empty_params_ok():
    """P2b: n_layers=0 with empty params is valid (pure encoding circuit)."""
    from quantx_engine.contrib.qml import _run_circuit_sv
    sv = _run_circuit_sv(np.array([0.5, 0.3]), np.zeros(0), n_qubits=2, n_layers=0)
    assert abs(sv.norm() - 1.0) < 1e-10
