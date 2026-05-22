"""Quantum kernel methods and Quantum Support Vector Machine (M80).

Implements a ZZ-entanglement circuit feature map (IBM Qiskit convention),
the fidelity quantum kernel K(x,x') = |⟨φ(x)|φ(x')⟩|², kernel target
alignment metrics, and a soft-margin QSVM trained with a simplified SMO.

All computations are pure NumPy; no external quantum-computing or ML
framework is required.

Quick reference::

    from quantx_engine.quantum_kernel import ZZFeatureMap, QSVM
    import numpy as np

    fm  = ZZFeatureMap(n_qubits=2, n_reps=2)
    clf = QSVM(feature_map=fm, C=1.0)
    X   = np.array([[0.1, 0.2], [0.8, 0.9], [0.2, 0.1], [0.9, 0.8]])
    y   = np.array([-1.0, 1.0, -1.0, 1.0])
    res = clf.fit(X, y)
    print(clf.score(X, y))   # 1.0

References
----------
.. [1] Havlicek, V. et al. (2019). Supervised learning with quantum-enhanced
       feature spaces. *Nature* 567, 209.
       https://doi.org/10.1038/s41586-019-0980-2

.. [2] Schuld, M. & Killoran, N. (2019). Quantum machine learning in feature
       Hilbert spaces. *Phys. Rev. Lett.* 122, 040504.
       https://doi.org/10.1103/PhysRevLett.122.040504

.. [3] Kubler, J. et al. (2021). An inductive bias for distances: Neural nets
       that respect the geometry of the data.
       *NeurIPS 2021*. https://doi.org/10.48550/arXiv.2103.09316
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

__all__ = [
    "QKernelError",
    "QuantumFeatureMap",
    "ZZFeatureMap",
    "FidelityQuantumKernel",
    "target_alignment",
    "frobenius_alignment",
    "QSVMResult",
    "QSVM",
]


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class QKernelError(ValueError):
    """Raised for invalid inputs to quantum kernel methods."""


def _require_numpy() -> None:
    if not _HAS_NUMPY:  # pragma: no cover
        raise QKernelError("NumPy is required for quantum kernel methods.")


# ---------------------------------------------------------------------------
# Statevector gate helpers
# ---------------------------------------------------------------------------

def _apply_single_qubit(
    psi: "np.ndarray",
    n: int,
    qubit: int,
    U: "np.ndarray",
) -> "np.ndarray":
    """Apply a 2×2 gate U to qubit ``qubit`` of a statevector.

    Qubits are indexed 0 (MSB) … n-1 (LSB), consistent with QuantX
    convention.  Basis state integer index k encodes bit (n-1-qubit) for
    qubit ``qubit``.
    """
    bit = n - 1 - qubit
    mask = 1 << bit
    dim = 1 << n
    all_idx = np.arange(dim)
    idx0 = all_idx[all_idx & mask == 0]   # qubit in |0⟩
    idx1 = idx0 | mask                    # qubit in |1⟩
    result = psi.copy()
    result[idx0] = U[0, 0] * psi[idx0] + U[0, 1] * psi[idx1]
    result[idx1] = U[1, 0] * psi[idx0] + U[1, 1] * psi[idx1]
    return result


def _apply_cx(
    psi: "np.ndarray",
    n: int,
    ctrl: int,
    tgt: int,
) -> "np.ndarray":
    """Apply CX(ctrl, tgt) to a statevector."""
    ctrl_bit = n - 1 - ctrl
    tgt_bit  = n - 1 - tgt
    dim = 1 << n
    all_idx = np.arange(dim)
    ctrl_mask = 1 << ctrl_bit
    tgt_mask  = 1 << tgt_bit
    # Indices where ctrl = |1⟩ and tgt = |0⟩ (swap with ctrl=1, tgt=1)
    c1t0 = all_idx[(all_idx & ctrl_mask != 0) & (all_idx & tgt_mask == 0)]
    c1t1 = c1t0 | tgt_mask
    result = psi.copy()
    result[c1t0] = psi[c1t1]
    result[c1t1] = psi[c1t0]
    return result


def _rz(theta: float) -> "np.ndarray":
    """Rz(θ) = diag(e^{−iθ/2}, e^{+iθ/2})."""
    return np.array(
        [[math.cos(theta / 2) - 1j * math.sin(theta / 2), 0.0],
         [0.0, math.cos(theta / 2) + 1j * math.sin(theta / 2)]],
        dtype=complex,
    )


_H_GATE: "np.ndarray" = (
    np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)
)


# ---------------------------------------------------------------------------
# Quantum Feature Maps
# ---------------------------------------------------------------------------

class QuantumFeatureMap:
    """Abstract base class for quantum feature maps.

    A feature map φ sends a classical vector x ∈ ℝⁿ to a normalised
    quantum statevector |φ(x)⟩ ∈ ℂ^(2^n_qubits).
    """

    @property
    def n_qubits(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def statevector(self, x: "np.ndarray") -> "np.ndarray":  # pragma: no cover
        """Return |φ(x)⟩ as a complex array of length 2^n_qubits."""
        raise NotImplementedError


class ZZFeatureMap(QuantumFeatureMap):
    """ZZ-entanglement circuit feature map (IBM Qiskit convention).

    Each of *n_reps* repetitions applies:

    1. Hadamard on every qubit: H^⊗n
    2. Phase encoding: Rz(2 x_i) on qubit i for all i
    3. Pairwise ZZ interactions for all ordered pairs (i, j), i < j:

       CX_{i,j} → Rz_j(2(π − x_i)(π − x_j)) → CX_{i,j}

    This realises the unitary U_φ(x) such that
    |φ(x)⟩ = U_φ(x)|0…0⟩.

    Parameters
    ----------
    n_qubits:
        Number of qubits (must equal the dimension of the input vectors).
    n_reps:
        Number of encoding layers.
    """

    def __init__(self, n_qubits: int, n_reps: int = 2) -> None:
        _require_numpy()
        if n_qubits < 1:
            raise QKernelError(f"n_qubits must be ≥ 1, got {n_qubits}")
        if n_reps < 1:
            raise QKernelError(f"n_reps must be ≥ 1, got {n_reps}")
        self._n_qubits = n_qubits
        self._n_reps = n_reps
        self._pairs = [
            (i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)
        ]

    @property
    def n_qubits(self) -> int:
        return self._n_qubits

    @property
    def n_features(self) -> int:
        """Alias for :attr:`n_qubits`."""
        return self._n_qubits

    @property
    def n_reps(self) -> int:
        return self._n_reps

    def statevector(self, x: "np.ndarray") -> "np.ndarray":
        """Compute the feature-map state |φ(x)⟩.

        Parameters
        ----------
        x:
            Feature vector of length ``n_qubits``.

        Returns
        -------
        Complex numpy array of shape ``(2**n_qubits,)``, unit-normalised.

        Raises
        ------
        QKernelError
            If ``len(x) != n_qubits``.
        """
        _require_numpy()
        x = np.asarray(x, dtype=float).ravel()
        if len(x) != self._n_qubits:
            raise QKernelError(
                f"ZZFeatureMap(n_qubits={self._n_qubits}) expects input of "
                f"length {self._n_qubits}, got {len(x)}."
            )
        n = self._n_qubits
        psi = np.zeros(1 << n, dtype=complex)
        psi[0] = 1.0  # |0…0⟩

        for _ in range(self._n_reps):
            # 1. H on every qubit
            for i in range(n):
                psi = _apply_single_qubit(psi, n, i, _H_GATE)
            # 2. Rz(2 x_i) on qubit i
            for i in range(n):
                psi = _apply_single_qubit(psi, n, i, _rz(2.0 * x[i]))
            # 3. Pairwise ZZ interactions
            for i, j in self._pairs:
                theta = 2.0 * (math.pi - x[i]) * (math.pi - x[j])
                psi = _apply_cx(psi, n, i, j)
                psi = _apply_single_qubit(psi, n, j, _rz(theta))
                psi = _apply_cx(psi, n, i, j)

        # Normalise (guards against floating-point drift)
        norm = float(np.linalg.norm(psi))
        return psi / norm if norm > 1e-15 else psi


# ---------------------------------------------------------------------------
# Fidelity quantum kernel
# ---------------------------------------------------------------------------

class FidelityQuantumKernel:
    """Fidelity (overlap) quantum kernel.

    K(x, x') = |⟨φ(x)|φ(x')⟩|²

    The diagonal satisfies K(x, x) = 1 for any unit-normalised feature map.
    The kernel matrix is positive semi-definite by construction.

    Parameters
    ----------
    feature_map:
        A :class:`QuantumFeatureMap` instance.
    """

    def __init__(self, feature_map: QuantumFeatureMap) -> None:
        _require_numpy()
        self.feature_map = feature_map

    def evaluate(self, x1: "np.ndarray", x2: "np.ndarray") -> float:
        """Evaluate K(x1, x2).

        Returns
        -------
        float in [0, 1].
        """
        psi1 = self.feature_map.statevector(np.asarray(x1, dtype=float))
        psi2 = self.feature_map.statevector(np.asarray(x2, dtype=float))
        return float(abs(np.dot(psi1.conj(), psi2)) ** 2)

    def matrix(
        self,
        X: "np.ndarray",
        Y: "np.ndarray | None" = None,
    ) -> "np.ndarray":
        """Compute the kernel matrix K[i, j] = evaluate(X[i], Y[j]).

        Parameters
        ----------
        X:
            Array of shape ``(n_x, n_features)``.
        Y:
            Array of shape ``(n_y, n_features)``.  If ``None``, computes
            the symmetric self-kernel of X (exploiting K[i,j] = K[j,i]).

        Returns
        -------
        numpy.ndarray of shape ``(n_x, n_y)`` (or ``(n_x, n_x)`` when
        Y is None), entries in [0, 1].
        """
        _require_numpy()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        symmetric = Y is None
        if symmetric:
            Y = X
        else:
            Y = np.atleast_2d(np.asarray(Y, dtype=float))

        n_x, n_y = len(X), len(Y)

        # Precompute statevectors (avoids recomputing when symmetric)
        svecs_x = [self.feature_map.statevector(X[i]) for i in range(n_x)]
        svecs_y = svecs_x if symmetric else [
            self.feature_map.statevector(Y[j]) for j in range(n_y)
        ]

        K = np.zeros((n_x, n_y))
        for i in range(n_x):
            j_start = i if symmetric else 0
            for j in range(j_start, n_y):
                val = float(abs(np.dot(svecs_x[i].conj(), svecs_y[j])) ** 2)
                K[i, j] = val
                if symmetric and i != j:
                    K[j, i] = val

        return K


# ---------------------------------------------------------------------------
# Kernel alignment metrics
# ---------------------------------------------------------------------------

def target_alignment(K: "np.ndarray", y: "np.ndarray") -> float:
    """Kernel-Target Alignment (KTA) between kernel K and labels y.

    Measures how well K separates the two classes::

        KTA = ⟨K, yy^T⟩_F / (n · ‖K‖_F)

    where ⟨·,·⟩_F is the Frobenius inner product and y ∈ {−1, +1}.
    KTA = 1 when K = yy^T (perfect alignment).

    Parameters
    ----------
    K:
        Kernel matrix of shape ``(n, n)``.
    y:
        Binary labels ±1, shape ``(n,)``.

    Returns
    -------
    float in [−1, 1] (0 for trivial/zero kernel).
    """
    _require_numpy()
    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    yy = np.outer(y, y)
    num = float(np.sum(K * yy))
    denom = n * float(np.linalg.norm(K, "fro"))
    return num / denom if denom > 1e-12 else 0.0


def frobenius_alignment(K1: "np.ndarray", K2: "np.ndarray") -> float:
    """Cosine similarity between two kernel matrices in Frobenius norm.

    Returns ⟨K1, K2⟩_F / (‖K1‖_F · ‖K2‖_F).

    Parameters
    ----------
    K1, K2:
        Kernel matrices of the same shape.

    Returns
    -------
    float in [−1, 1] (0 if either matrix is zero).
    """
    _require_numpy()
    K1 = np.asarray(K1, dtype=float)
    K2 = np.asarray(K2, dtype=float)
    if K1.shape != K2.shape:
        raise QKernelError(
            f"K1 and K2 must have the same shape, got {K1.shape} vs {K2.shape}."
        )
    num = float(np.sum(K1 * K2))
    denom = float(np.linalg.norm(K1, "fro") * np.linalg.norm(K2, "fro"))
    return num / denom if denom > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# QSVM — dual solver (simplified SMO)
# ---------------------------------------------------------------------------

def _solve_smo(
    K: "np.ndarray",
    y: "np.ndarray",
    C: float,
    tol: float = 1e-3,
    max_passes: int = 100,
) -> "tuple[np.ndarray, float]":
    """Simplified SMO for soft-margin kernel SVM.

    Solves the dual:
        max  Σ α_i − ½ Σ_ij α_i α_j y_i y_j K_ij
        s.t. 0 ≤ α_i ≤ C,   Σ_i α_i y_i = 0

    Returns
    -------
    alpha : np.ndarray of shape (n,)
    bias  : float
    """
    n = len(y)
    alpha = np.zeros(n)
    b = 0.0

    def _recompute_errors() -> "np.ndarray":
        return K @ (alpha * y) + b - y

    passes = 0
    E = _recompute_errors()

    while passes < max_passes:
        num_changed = 0

        for i in range(n):
            Ei = E[i]
            yi = y[i]

            # Check KKT violation
            if not ((yi * Ei < -tol and alpha[i] < C) or
                    (yi * Ei > tol and alpha[i] > 0.0)):
                continue

            # Select j: index with maximum |Ei − Ej|, excluding i
            diffs = np.abs(Ei - E)
            diffs[i] = -1.0
            j = int(np.argmax(diffs))
            Ej = E[j]

            ai_old, aj_old = alpha[i], alpha[j]
            b_old = b

            # Box bounds for alpha_j
            if y[i] == y[j]:
                L = max(0.0, ai_old + aj_old - C)
                H = min(C, ai_old + aj_old)
            else:
                L = max(0.0, aj_old - ai_old)
                H = min(C, C + aj_old - ai_old)
            if H - L < 1e-10:
                continue

            eta = K[i, i] + K[j, j] - 2.0 * K[i, j]
            if eta < 1e-10:
                continue

            # Update alpha_j (clipped)
            alpha[j] = float(np.clip(aj_old + y[j] * (Ei - Ej) / eta, L, H))
            if abs(alpha[j] - aj_old) < 1e-5:
                continue

            # Update alpha_i (preserves Σ α_i y_i = 0)
            alpha[i] = ai_old + y[i] * y[j] * (aj_old - alpha[j])

            # Update bias
            b1 = (b_old - Ei
                  - y[i] * (alpha[i] - ai_old) * K[i, i]
                  - y[j] * (alpha[j] - aj_old) * K[i, j])
            b2 = (b_old - Ej
                  - y[i] * (alpha[i] - ai_old) * K[i, j]
                  - y[j] * (alpha[j] - aj_old) * K[j, j])
            if 0.0 < alpha[i] < C:
                b = b1
            elif 0.0 < alpha[j] < C:
                b = b2
            else:
                b = 0.5 * (b1 + b2)

            # Incrementally update cached errors
            d_i = (alpha[i] - ai_old) * y[i]
            d_j = (alpha[j] - aj_old) * y[j]
            E += d_i * K[i] + d_j * K[j] + (b - b_old)

            num_changed += 1

        passes = 0 if num_changed > 0 else passes + 1

    return alpha, b


# ---------------------------------------------------------------------------
# QSVM
# ---------------------------------------------------------------------------

@dataclass
class QSVMResult:
    """Result of fitting a QSVM classifier.

    Attributes
    ----------
    support_vector_indices:
        Indices into the training set of support vectors (α_i > 0).
    alphas:
        Dual coefficients for every training sample.
    bias:
        Decision boundary offset b.
    n_support:
        Number of support vectors.
    train_accuracy:
        Fraction of correctly classified training samples.
    kernel_matrix:
        Symmetric training kernel matrix K (n_train × n_train).
    """
    support_vector_indices: "list[int]"
    alphas: "list[float]"
    bias: float
    n_support: int
    train_accuracy: float
    kernel_matrix: "np.ndarray"


class QSVM:
    """Quantum Support Vector Machine for binary classification.

    Trains a soft-margin SVM using the fidelity quantum kernel::

        K(x, x') = |⟨φ(x)|φ(x')⟩|²

    The dual QP is solved with a simplified Sequential Minimal
    Optimisation (SMO) algorithm.

    Parameters
    ----------
    feature_map:
        A :class:`QuantumFeatureMap` instance.  Exactly one of
        *feature_map* or *kernel* must be supplied.
    kernel:
        A pre-built :class:`FidelityQuantumKernel`.  Takes precedence
        over *feature_map* when both are given.
    C:
        Soft-margin regularisation parameter (must be > 0).  Larger C
        allows fewer margin violations.

    Examples
    --------
    >>> fm  = ZZFeatureMap(n_qubits=2, n_reps=1)
    >>> clf = QSVM(feature_map=fm, C=1.0)
    >>> X   = np.array([[0.1, 0.2], [0.9, 0.8]])
    >>> y   = np.array([-1.0, 1.0])
    >>> result = clf.fit(X, y)
    >>> clf.predict(np.array([[0.15, 0.15]]))[0] in {-1.0, 1.0}
    True
    """

    def __init__(
        self,
        feature_map: "QuantumFeatureMap | None" = None,
        kernel: "FidelityQuantumKernel | None" = None,
        C: float = 1.0,
    ) -> None:
        _require_numpy()
        if kernel is not None:
            self.kernel: FidelityQuantumKernel = kernel
        elif feature_map is not None:
            self.kernel = FidelityQuantumKernel(feature_map)
        else:
            raise QKernelError(
                "Either feature_map or kernel must be provided to QSVM."
            )
        if C <= 0:
            raise QKernelError(f"C must be > 0, got {C}.")
        self.C = C
        self._alpha: "np.ndarray | None" = None
        self._bias: float = 0.0
        self._X_train: "np.ndarray | None" = None
        self._y_train: "np.ndarray | None" = None

    def fit(
        self,
        X: "np.ndarray",
        y: "np.ndarray",
        tol: float = 1e-3,
        max_passes: int = 100,
    ) -> QSVMResult:
        """Fit the QSVM on training data.

        Parameters
        ----------
        X:
            Training features, shape ``(n_samples, n_features)``.
        y:
            Binary labels ∈ {−1, +1}, shape ``(n_samples,)``.
        tol:
            KKT tolerance for the SMO solver.
        max_passes:
            Maximum consecutive passes without an α update.

        Returns
        -------
        QSVMResult
        """
        _require_numpy()
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()  # flatten (n,1) → (n,)
        if X.shape[0] != len(y):
            raise QKernelError(
                f"X has {X.shape[0]} rows but y has {len(y)} entries."
            )
        unique = np.unique(y)
        if set(unique) != {-1.0, 1.0}:
            raise QKernelError(
                f"Labels must be exactly {{−1, +1}}, got {set(unique)}."
            )

        K = self.kernel.matrix(X)
        self._alpha, self._bias = _solve_smo(K, y, self.C, tol, max_passes)
        self._X_train = X
        self._y_train = y

        sv_idx = list(np.where(self._alpha > 1e-5)[0])
        y_pred = np.sign(K @ (self._alpha * y) + self._bias)
        y_pred[y_pred == 0] = 1.0  # tie-break
        train_acc = float(np.mean(y_pred == y))

        return QSVMResult(
            support_vector_indices=sv_idx,
            alphas=self._alpha.tolist(),
            bias=self._bias,
            n_support=len(sv_idx),
            train_accuracy=train_acc,
            kernel_matrix=K,
        )

    def decision_function(self, X: "np.ndarray") -> "np.ndarray":
        """Return the signed distance to the decision boundary.

        Parameters
        ----------
        X:
            Test features, shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray of shape ``(n_samples,)``.

        Raises
        ------
        QKernelError
            If the classifier has not been fitted yet.
        """
        if self._alpha is None or self._X_train is None:
            raise QKernelError("Call fit() before predict() or decision_function().")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        K_test = self.kernel.matrix(X, self._X_train)
        return K_test @ (self._alpha * self._y_train) + self._bias

    def predict(self, X: "np.ndarray") -> "np.ndarray":
        """Predict class labels for test samples.

        Parameters
        ----------
        X:
            Test features, shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray of ±1.0, shape ``(n_samples,)``.
        """
        df = self.decision_function(X)
        preds = np.sign(df)
        preds[preds == 0.0] = 1.0  # tie-break to positive class
        return preds

    def score(self, X: "np.ndarray", y: "np.ndarray") -> float:
        """Compute classification accuracy.

        Parameters
        ----------
        X:
            Test features.
        y:
            True labels ±1.

        Returns
        -------
        float in [0, 1].
        """
        y = np.asarray(y, dtype=float).ravel()  # flatten (n,1) → (n,)
        return float(np.mean(self.predict(X) == y))
