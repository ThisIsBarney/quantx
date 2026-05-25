"""Quantum error mitigation for the QuantX engine.

Provides two complementary mitigation techniques:

**Zero-Noise Extrapolation (ZNE)**  — run circuits at artificially amplified
noise levels (via *gate folding*) then extrapolate the results back to zero
noise using Richardson, polynomial, or exponential fitting.

**Probabilistic Error Cancellation (PEC)** — decompose ideal gates as a
quasi-probability distribution over noisy implementations; sample circuits
with associated ±weights and take a weighted mean to obtain an unbiased
estimator of the noiseless expectation value.

Both techniques operate through a user-supplied *executor* callable::

    executor(ops: list[dict]) -> float

where each op dict has the shape ``{"name": str, "qubits": list[int],
"params": list[float]}``.  A convenience factory :func:`make_sv_executor`
builds one from :class:`~quantx_engine.gpu.GpuStatevector`.

Quick-start example::

    from quantx_engine.mitigation import (
        zne_mitigate, pec_mitigate, make_sv_executor,
        RichardsonExtrapolator,
    )

    # Build a statevector executor: run bell-state circuit, measure <Z0>
    executor = make_sv_executor(n_qubits=2, observable_fn=lambda sv: sv.expectation_z(0))

    bell_ops = [
        {"name": "H",    "qubits": [0], "params": []},
        {"name": "CNOT", "qubits": [0, 1], "params": []},
    ]

    # ZNE with Richardson extrapolation at scale factors 1, 3, 5
    result = zne_mitigate(bell_ops, executor, scale_factors=[1, 3, 5])
    print(result.mitigated_value)

    # PEC with 2% depolarizing noise per gate, 500 samples
    result = pec_mitigate(bell_ops, noisy_executor, noise_prob=0.02, n_samples=500)
    print(result.mitigated_value, "±", result.std_error)

References
----------
.. [1] Temme, K., Bravyi, S. & Gambetta, J. M. (2017). Error mitigation for
       short-depth quantum circuits. *Phys. Rev. Lett.* 119, 180509.
       https://doi.org/10.1103/PhysRevLett.119.180509

.. [2] Li, Y. & Benjamin, S. C. (2017). Efficient variational quantum
       simulator incorporating active error minimization.
       *Phys. Rev. X* 7, 021050.
       https://doi.org/10.1103/PhysRevX.7.021050

.. [3] Kandala, A. et al. (2019). Error mitigation extends the computational
       reach of a noisy quantum processor. *Nature* 567, 491.
       https://doi.org/10.1038/s41586-019-1040-7

Physical validation
-------------------
ZNE with Richardson extrapolation: for polynomial noise N(lambda) = N_0*(1+c*lambda),
the zero-noise estimate is exact at degree 1 [Temme2017, Eq. 3].
For a depolarising noise model, the unbiased estimator has zero variance at
the noise-extrapolation point (verified analytically).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

__all__ = [
    "MitigationError",
    # Gate folding
    "fold_gates",
    # Extrapolators
    "Extrapolator",
    "ExtrapolationResult",
    "RichardsonExtrapolation",
    "RichardsonExtrapolator",
    "PolyExtrapolator",
    "ExpExtrapolator",
    # ZNE
    "ZNEResult",
    "zne_mitigate",
    # PEC
    "QuasiProbabilityDecomposition",
    "PECResult",
    "pec_mitigate",
    # Utilities
    "make_sv_executor",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class MitigationError(ValueError):
    """Raised for invalid mitigation parameters or unsupported configurations."""


# ---------------------------------------------------------------------------
# Gate-operation helpers
# ---------------------------------------------------------------------------

# Self-inverse single-qubit gates (G† = G)
_SELF_INVERSE_1Q = frozenset({"X", "Y", "Z", "H", "PauliX", "PauliY", "PauliZ", "Hadamard", "I", "Identity"})
# Self-inverse two-qubit gates
_SELF_INVERSE_2Q = frozenset({"CNOT", "CX", "CZ", "SWAP"})


def _invert_op(op: dict) -> dict:
    """Return the inverse of a gate operation.

    Supports all gates recognized by :class:`~quantx_engine.gpu.GpuStatevector`.
    For gates without a named inverse (e.g. parametric rotations) the inverse is
    computed by negating the rotation angle.

    Parameters
    ----------
    op:
        Gate operation dict ``{"name": str, "qubits": list[int], "params": list}``.

    Returns
    -------
    dict
        New operation dict representing the inverse gate.
    """
    name = op["name"]
    qubits = op["qubits"]
    params = list(op.get("params", []))

    if name in _SELF_INVERSE_1Q or name in _SELF_INVERSE_2Q:
        return {"name": name, "qubits": qubits, "params": params}
    if name == "S":
        # S = PhaseShift(π/2) = diag([1, i]).  S† = diag([1, -i]) = PhaseShift(-π/2).
        # Using RZ(-π/2) would introduce a global phase exp(-iπ/4), so S·RZ(-π/2) ≠ I.
        # PhaseShift(φ) = diag([1, exp(iφ)]) has no global-phase issue: S·S† = I exactly.
        return {"name": "PhaseShift", "qubits": qubits, "params": [-math.pi / 2]}
    if name == "T":
        # T = PhaseShift(π/4) = diag([1, exp(iπ/4)]).  T† = PhaseShift(-π/4).
        return {"name": "PhaseShift", "qubits": qubits, "params": [-math.pi / 4]}
    if name in ("RX", "RY", "RZ", "PhaseShift"):
        if not params:
            raise MitigationError(f"Gate {name!r} requires a parameter but params is empty")
        return {"name": name, "qubits": qubits, "params": [-params[0]] + params[1:]}
    raise MitigationError(
        f"Cannot compute inverse for gate {name!r}. "
        "Add it to _SELF_INVERSE_1Q/_SELF_INVERSE_2Q or handle explicitly."
    )


# ---------------------------------------------------------------------------
# Gate folding (noise amplification for ZNE)
# ---------------------------------------------------------------------------

def fold_gates(ops: list[dict], scale_factor: float) -> list[dict]:
    """Amplify circuit noise by gate-level folding.

    Each gate G is replaced by G·(G†·G)^k to inflate the effective noise
    by an integer factor (2k+1).  For non-integer scale factors the *last*
    ``tail`` gates receive one additional fold so that the total gate count
    matches ``round(scale_factor * len(ops))``.

    Parameters
    ----------
    ops:
        List of gate operation dicts ``{"name": str, "qubits": list[int],
        "params": list[float]}``.
    scale_factor:
        Desired noise amplification factor.  Must be ≥ 1.  Values of the
        form 2k+1 (odd integers) are exact; other values use partial folding.

    Returns
    -------
    list[dict]
        Folded gate sequence.  Length ≈ ``scale_factor * len(ops)``.

    Examples
    --------
    >>> ops = [{"name": "RY", "qubits": [0], "params": [0.5]}]
    >>> fold_gates(ops, 3)  # G -> G·G†·G
    [{"name": "RY", ...}, {"name": "RY", ..., "params": [-0.5]}, {"name": "RY", ...}]
    """
    if scale_factor < 1.0:
        raise MitigationError(f"scale_factor must be >= 1, got {scale_factor}")
    if not ops:
        return []

    n = len(ops)
    # Number of full G†G repeats each gate receives
    num_folds = int((scale_factor - 1) // 2)
    # Fractional remainder → how many tail gates get one extra fold
    remainder = scale_factor - 1 - 2 * num_folds  # in [0, 2)
    tail_count = round(remainder * n / 2)

    result: list[dict] = []
    for i, op in enumerate(ops):
        result.append(op)
        extra = 1 if (n - tail_count <= i < n) else 0
        for _ in range(num_folds + extra):
            result.append(_invert_op(op))
            result.append(op)

    return result


# ---------------------------------------------------------------------------
# Extrapolators
# ---------------------------------------------------------------------------

@dataclass
class ExtrapolationResult:
    """Raw output from an extrapolator fit."""
    mitigated_value: float
    """Extrapolated expectation value at noise scale = 0."""
    coefficients: list[float]
    """Fit coefficients (interpretation depends on extrapolator)."""


class Extrapolator(ABC):
    """Abstract base class for ZNE extrapolators."""

    @abstractmethod
    def extrapolate(
        self,
        scale_factors: list[float],
        noisy_values: list[float],
    ) -> ExtrapolationResult:
        """Fit the noise-vs-value curve and return the zero-noise estimate.

        Parameters
        ----------
        scale_factors:
            Noise scale factors (x-axis).  Must have len ≥ 2.
        noisy_values:
            Measured expectation values at each scale factor (y-axis).

        Returns
        -------
        ExtrapolationResult
        """

    def _validate(self, scale_factors, noisy_values):
        if len(scale_factors) != len(noisy_values):
            raise MitigationError(
                f"scale_factors and noisy_values must have the same length, "
                f"got {len(scale_factors)} and {len(noisy_values)}"
            )
        if len(scale_factors) < 2:
            raise MitigationError(
                f"At least 2 data points are required for extrapolation, "
                f"got {len(scale_factors)}"
            )
        if len(set(scale_factors)) != len(scale_factors):
            raise MitigationError(
                "scale_factors must be unique; duplicate values cause division by zero "
                f"in Richardson extrapolation. Got: {scale_factors}"
            )


class RichardsonExtrapolator(Extrapolator):
    """Richardson / Lagrange extrapolation to zero noise.

    Fits the unique polynomial of degree ``len(scale_factors) - 1`` passing
    through all data points and evaluates it at x = 0 using the closed-form
    Lagrange formula.  Exact for polynomial noise models of matching degree.

    This is the most commonly used ZNE extrapolator: it requires no
    hyperparameters and works with as few as 2 points.
    """

    def extrapolate(self, scale_factors, noisy_values):
        self._validate(scale_factors, noisy_values)
        xs = list(scale_factors)
        ys = list(noisy_values)
        n = len(xs)

        # Lagrange interpolation at x=0:  y(0) = Σ_i y_i * Π_{j≠i} (-x_j) / (x_i - x_j)
        total = 0.0
        coeffs: list[float] = []
        for i in range(n):
            num = 1.0
            den = 1.0
            for j in range(n):
                if j == i:
                    continue
                num *= -xs[j]          # (0 - x_j)
                den *= xs[i] - xs[j]
            c = num / den
            coeffs.append(c)
            total += c * ys[i]

        return ExtrapolationResult(mitigated_value=total, coefficients=coeffs)


class PolyExtrapolator(Extrapolator):
    """Polynomial least-squares extrapolation to zero noise.

    Fits a polynomial of the specified ``degree`` to all data points via
    :func:`numpy.polyfit` (least squares) and evaluates it at x = 0.

    Parameters
    ----------
    degree:
        Degree of the polynomial.  Must be ≥ 1 and < len(scale_factors).
        Use ``degree=1`` for linear extrapolation (fastest, fewest points
        needed) and higher degrees for more flexible fits.
    """

    def __init__(self, degree: int = 1):
        if degree < 1:
            raise MitigationError(f"degree must be >= 1, got {degree}")
        self.degree = degree

    def extrapolate(self, scale_factors, noisy_values):
        if not _HAS_NUMPY:  # pragma: no cover
            raise MitigationError("PolyExtrapolator requires numpy")
        self._validate(scale_factors, noisy_values)
        if self.degree >= len(scale_factors):
            raise MitigationError(
                f"Polynomial degree {self.degree} requires at least "
                f"{self.degree + 1} data points, got {len(scale_factors)}"
            )
        xs = np.array(scale_factors, dtype=float)
        ys = np.array(noisy_values, dtype=float)
        coeffs = np.polyfit(xs, ys, self.degree)   # highest degree first
        mitigated = float(np.polyval(coeffs, 0.0))
        return ExtrapolationResult(
            mitigated_value=mitigated,
            coefficients=list(float(c) for c in coeffs),
        )


class ExpExtrapolator(Extrapolator):
    """Exponential extrapolation to zero noise.

    Fits the model y = A + B·exp(C·x) to the data and returns A + B (the
    value at x = 0) as the mitigated estimate.  Requires at least 3 data
    points and ``scipy.optimize.curve_fit``.

    Parameters
    ----------
    p0:
        Initial guess ``(A, B, C)`` for the curve-fit.  Defaults to
        ``(0.0, 1.0, -0.5)`` which works well for decaying expectation values.

    Raises
    ------
    MitigationError
        If scipy is not installed or the fit fails to converge.
    """

    def __init__(self, p0: tuple[float, float, float] = (0.0, 1.0, -0.5)):
        self.p0 = p0

    def extrapolate(self, scale_factors, noisy_values):
        if not _HAS_NUMPY:  # pragma: no cover
            raise MitigationError("ExpExtrapolator requires numpy")
        self._validate(scale_factors, noisy_values)
        if len(scale_factors) < 3:
            raise MitigationError(
                "ExpExtrapolator requires at least 3 data points for a 3-parameter fit"
            )
        try:
            from scipy.optimize import curve_fit  # type: ignore
        except ImportError as exc:
            raise MitigationError(
                "ExpExtrapolator requires scipy.  Install with: pip install scipy"
            ) from exc

        xs = list(float(x) for x in scale_factors)
        ys = list(float(y) for y in noisy_values)

        def _model(x, A, B, C):
            return A + B * np.exp(C * np.array(x))

        try:
            popt, _ = curve_fit(_model, xs, ys, p0=self.p0, maxfev=10_000)
        except RuntimeError as exc:
            raise MitigationError(f"Exponential fit failed to converge: {exc}") from exc

        A, B, C = popt
        mitigated = float(A + B)   # = _model(0, A, B, C)
        return ExtrapolationResult(mitigated_value=mitigated, coefficients=[A, B, C])


# ---------------------------------------------------------------------------
# ZNE
# ---------------------------------------------------------------------------

@dataclass
class ZNEResult:
    """Result of a zero-noise extrapolation run.

    Attributes
    ----------
    scale_factors:
        Noise scale factors used in the experiment.
    noisy_values:
        Measured (noisy) expectation values at each scale factor.
    mitigated_value:
        Extrapolated expectation value at zero noise.
    extrapolator_name:
        Class name of the extrapolator used.
    extrapolation_coefficients:
        Fit coefficients returned by the extrapolator (interpretation
        depends on extrapolator type).
    """
    scale_factors: list[float]
    noisy_values: list[float]
    mitigated_value: float
    extrapolator_name: str
    extrapolation_coefficients: list[float]

    def __repr__(self) -> str:  # pragma: no cover
        pairs = ", ".join(
            f"{sf:.2g}→{v:.4g}" for sf, v in zip(self.scale_factors, self.noisy_values)
        )
        return (
            f"ZNEResult(mitigated={self.mitigated_value:.6g}, "
            f"extrapolator={self.extrapolator_name}, "
            f"data=[{pairs}])"
        )


def zne_mitigate(
    ops: list[dict],
    executor: Callable[[list[dict]], float],
    scale_factors: list[float] | None = None,
    extrapolator: Extrapolator | None = None,
) -> ZNEResult:
    """Run zero-noise extrapolation and return the mitigated expectation value.

    The circuit ``ops`` is run at each noise level (via gate folding) and the
    noisy results are extrapolated back to zero noise.

    Parameters
    ----------
    ops:
        List of gate operation dicts ``{"name": str, "qubits": list[int],
        "params": list[float]}``.
    executor:
        Callable ``(ops: list[dict]) -> float`` that runs the circuit and
        returns an expectation value.  Build one with :func:`make_sv_executor`.
    scale_factors:
        Noise scale factors to evaluate.  Default: ``[1, 3, 5]``.
        Must all be ≥ 1; values of the form 2k+1 (odd integers) produce
        exact gate folding, while non-integer values use partial folding.
    extrapolator:
        Extrapolation strategy.  Default: :class:`RichardsonExtrapolator`.

    Returns
    -------
    ZNEResult

    Examples
    --------
    >>> def my_executor(ops):
    ...     sv = GpuStatevector(n_qubits=2)
    ...     for op in ops:
    ...         sv.apply_gate(op["name"], op["qubits"], op.get("params", ()))
    ...     return sv.expectation_z(0)
    >>> result = zne_mitigate(bell_ops, my_executor, scale_factors=[1, 3, 5])
    >>> print(result.mitigated_value)
    """
    if scale_factors is None:
        scale_factors = [1.0, 3.0, 5.0]
    if extrapolator is None:
        extrapolator = RichardsonExtrapolator()

    for sf in scale_factors:
        if sf < 1.0:
            raise MitigationError(f"All scale_factors must be >= 1, got {sf}")

    noisy_values: list[float] = []
    for sf in scale_factors:
        folded = fold_gates(ops, sf)
        val = executor(folded)
        noisy_values.append(float(val))

    ext_result = extrapolator.extrapolate(scale_factors, noisy_values)
    return ZNEResult(
        scale_factors=list(float(sf) for sf in scale_factors),
        noisy_values=noisy_values,
        mitigated_value=ext_result.mitigated_value,
        extrapolator_name=type(extrapolator).__name__,
        extrapolation_coefficients=ext_result.coefficients,
    )


# ---------------------------------------------------------------------------
# PEC
# ---------------------------------------------------------------------------

@dataclass
class QuasiProbabilityDecomposition:
    """Quasi-probability decomposition (QPD) of an ideal single-qubit gate.

    Represents the ideal gate as a signed sum of noisy implementations::

        G_ideal = Σ_i q_i * (P_i · G_noisy)

    where q_i can be negative (hence "quasi-probability") and P_i is a Pauli
    operator prepended before the (noisy) gate.

    The **one-norm** γ = Σ_i |q_i| > 1 quantifies the sampling overhead; the
    total number of circuit samples required to achieve standard error ε
    scales as O(γ^{2m} / ε²) for an m-gate circuit.

    Attributes
    ----------
    basis:
        List of ``(pauli_name, sign, sample_prob)`` tuples where:
        - ``pauli_name`` ∈ {``"I"``, ``"X"``, ``"Y"``, ``"Z"``},
        - ``sign`` = +1.0 or −1.0,
        - ``sample_prob`` = |q_i| / γ  (positive, sums to 1).
    gamma:
        One-norm of the quasi-probability distribution.
    """
    basis: list[tuple[str, float, float]]  # (pauli_name, sign, sample_prob)
    gamma: float

    @classmethod
    def from_depolarizing(cls, noise_prob: float) -> "QuasiProbabilityDecomposition":
        """Build the QPD for a gate subject to single-qubit depolarizing noise.

        Models the noisy channel as::

            Λ_p(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)

        and inverts it to recover the ideal gate.  The resulting quasi-probability
        weights are:

        - q_I = (1 − p/3) / (1 − 4p/3)   [positive]
        - q_X = q_Y = q_Z = −(p/3) / (1 − 4p/3)  [negative for p > 0]
        - γ = (1 + 2p/3) / (1 − 4p/3)

        Parameters
        ----------
        noise_prob:
            Depolarizing probability p per gate, in [0, 3/4).

        Raises
        ------
        MitigationError
            If ``noise_prob`` is out of range.
        """
        p = float(noise_prob)
        if not (0.0 <= p < 0.75):
            raise MitigationError(
                f"noise_prob must be in [0, 3/4) for PEC decomposition, got {p:.4g}"
            )
        if p == 0.0:
            return cls(basis=[("I", 1.0, 1.0)], gamma=1.0)

        denom = 1.0 - 4.0 * p / 3.0   # > 0 for p < 3/4
        # Derivation: require Σ q_k · (Λ_p ∘ T_k) = I and trace-preservation (Σ q_k = 1).
        # Solution: q_I = (1 - p/3) / denom,  q_{X,Y,Z} = -(p/3) / denom.
        # γ = |q_I| + 3·|q_pauli| = (1 - p/3 + p) / denom = (1 + 2p/3) / denom.
        q_I = (1.0 - p / 3.0) / denom
        q_pauli = -(p / 3.0) / denom   # negative for p > 0
        gamma = q_I + 3.0 * abs(q_pauli)   # = (1 + 2p/3) / (1 - 4p/3)

        # Normalised sampling probabilities |q_i| / γ
        prob_I = q_I / gamma
        prob_pauli = abs(q_pauli) / gamma

        basis = [
            ("I", +1.0, prob_I),
            ("X", -1.0, prob_pauli),
            ("Y", -1.0, prob_pauli),
            ("Z", -1.0, prob_pauli),
        ]
        return cls(basis=basis, gamma=gamma)

    def sample(self, rng: "np.random.Generator") -> tuple[float, str]:
        """Draw one sample from the quasi-probability distribution.

        Parameters
        ----------
        rng:
            NumPy random generator instance.

        Returns
        -------
        (sign, pauli_name):
            ``sign`` ∈ {+1.0, −1.0} and ``pauli_name`` ∈ {``"I"``, ``"X"``,
            ``"Y"``, ``"Z"``}.
        """
        probs = [b[2] for b in self.basis]
        # Renormalize to handle floating-point rounding
        total = sum(probs)
        probs = [p / total for p in probs]
        idx = int(rng.choice(len(self.basis), p=probs))
        _, sign, _ = self.basis[idx]
        name = self.basis[idx][0]
        return sign, name

    def __repr__(self) -> str:  # pragma: no cover
        terms = ", ".join(
            f"{s:+.3g}·{name}" for name, s, _ in self.basis
        )
        return f"QuasiProbabilityDecomposition(γ={self.gamma:.4g}, [{terms}])"


@dataclass
class PECResult:
    """Result of a probabilistic error cancellation run.

    Attributes
    ----------
    mitigated_value:
        Unbiased estimate of the noiseless expectation value.
    std_error:
        Standard error of the mean across the ``n_samples`` samples.
    n_samples:
        Number of Monte Carlo samples used.
    gamma:
        One-norm of the single-gate QPD.  The total sampling overhead
        scales as γ^{2m} for an m-gate circuit.
    raw_estimates:
        Individual weighted estimates (length = n_samples).  Useful for
        computing confidence intervals or diagnosing convergence.
    """
    mitigated_value: float
    std_error: float
    n_samples: int
    gamma: float
    raw_estimates: list[float] = field(default_factory=list, repr=False)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PECResult(mitigated={self.mitigated_value:.6g} ± {self.std_error:.4g}, "
            f"n_samples={self.n_samples}, γ_total={self.gamma:.4g})"
        )


def pec_mitigate(
    ops: list[dict],
    noisy_executor: Callable[[list[dict]], float],
    noise_prob: float,
    n_samples: int = 100,
    seed: int | None = None,
) -> PECResult:
    """Run probabilistic error cancellation and return the mitigated expectation.

    For each of the ``n_samples`` Monte Carlo trials, an independent Pauli
    twirl is sampled for every single-qubit gate in ``ops`` and prepended to
    it.  The trial result is multiplied by a weight of γ^m · Π sign_i and
    the weighted mean over all trials converges to the noiseless expectation
    value.

    Two-qubit gates are passed through unmodified (no twirl applied).

    Parameters
    ----------
    ops:
        List of gate operation dicts ``{"name": str, "qubits": list[int],
        "params": list[float]}``.
    noisy_executor:
        Callable ``(ops: list[dict]) -> float`` that runs the circuit *with
        noise* (e.g. backed by a noisy simulator or real hardware) and returns
        an expectation value.
    noise_prob:
        Single-qubit depolarizing noise probability per gate, in [0, 3/4).
        Used to build the quasi-probability decomposition.
    n_samples:
        Number of Monte Carlo samples.  Larger values reduce statistical
        error but increase runtime.  Standard error ∝ 1/√n_samples.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    PECResult

    Notes
    -----
    The statistical error of PEC scales as O(γ^m / √n_samples) where γ is
    the one-norm of the QPD and m is the number of single-qubit gates.  For
    deep circuits this overhead becomes prohibitive, making PEC suitable
    mainly for shallow circuits on near-term devices.
    """
    if not _HAS_NUMPY:  # pragma: no cover
        raise MitigationError("pec_mitigate requires numpy")
    if n_samples < 1:
        raise MitigationError(f"n_samples must be >= 1, got {n_samples}")

    qpd = QuasiProbabilityDecomposition.from_depolarizing(noise_prob)
    rng = np.random.default_rng(seed)

    # Count single-qubit gates to compute total gamma factor
    n_1q_gates = sum(1 for op in ops if len(op.get("qubits", [])) == 1)
    total_gamma = qpd.gamma ** n_1q_gates

    raw_estimates: list[float] = []

    for _ in range(n_samples):
        twirled: list[dict] = []
        total_sign = 1.0

        for op in ops:
            qubits = op.get("qubits", [])
            if len(qubits) == 1:
                sign, pauli = qpd.sample(rng)
                total_sign *= sign
                if pauli != "I":
                    # Insert Pauli twirl BEFORE the (noisy) gate
                    twirled.append({"name": pauli, "qubits": qubits, "params": []})
            twirled.append(op)

        E = float(noisy_executor(twirled))
        raw_estimates.append(total_gamma * total_sign * E)

    arr = np.array(raw_estimates, dtype=float)
    mitigated = float(np.mean(arr))
    std_err = float(np.std(arr, ddof=1) / math.sqrt(n_samples)) if n_samples > 1 else 0.0

    return PECResult(
        mitigated_value=mitigated,
        std_error=std_err,
        n_samples=n_samples,
        gamma=total_gamma,
        raw_estimates=raw_estimates,
    )


# ---------------------------------------------------------------------------
# Utility: statevector executor factory
# ---------------------------------------------------------------------------

def make_sv_executor(
    n_qubits: int,
    observable_fn: Callable[["Any"], float],
    backend: str = "numpy",
    dtype: str = "complex128",
) -> Callable[[list[dict]], float]:
    """Create a noiseless statevector executor for use with ZNE.

    The returned callable initialises a fresh :class:`~quantx_engine.gpu.GpuStatevector`
    for each circuit run, applies all gates, and returns
    ``observable_fn(sv)``.

    Parameters
    ----------
    n_qubits:
        Number of qubits.
    observable_fn:
        Function that takes a :class:`~quantx_engine.gpu.GpuStatevector` and
        returns a float.  Common choices:
        ``lambda sv: sv.expectation_z(0)`` or
        ``lambda sv: float(sv.probabilities([0])[1])``.
    backend:
        ``"numpy"`` (default), ``"cupy"``, or ``"auto"``.
    dtype:
        ``"complex128"`` (default, fp64) or ``"complex64"`` (fp32).

    Returns
    -------
    Callable[[list[dict]], float]
        An executor compatible with :func:`zne_mitigate`.

    Examples
    --------
    >>> executor = make_sv_executor(2, lambda sv: sv.expectation_z(0))
    >>> result = zne_mitigate(ops, executor, scale_factors=[1, 3, 5])
    """
    from quantx_engine.gpu import GpuStatevector, GpuError  # local import to avoid circular deps

    def executor(ops: list[dict]) -> float:
        try:
            sv = GpuStatevector(n_qubits=n_qubits, backend=backend, dtype=dtype)
        except GpuError as exc:
            raise MitigationError(str(exc)) from exc
        for op in ops:
            sv.apply_gate(op["name"], op.get("qubits", []), op.get("params", ()))
        return float(observable_fn(sv))

    return executor


# ---------------------------------------------------------------------------
# Convenience wrapper: RichardsonExtrapolation
# ---------------------------------------------------------------------------

class RichardsonExtrapolation:
    """Simple Richardson extrapolation to zero noise.

    Parameters
    ----------
    scale_factors : list of float
        Noise scale factors λ at which measurements are taken.

    Examples
    --------
    >>> extrap = RichardsonExtrapolation(scale_factors=[1.0, 2.0])
    >>> result = extrap.extrapolate([0.8, 0.6])  # returns float
    """

    def __init__(self, scale_factors: list) -> None:
        self._scale_factors = list(scale_factors)
        self._impl = RichardsonExtrapolator()

    def extrapolate(self, noisy_values: list) -> float:
        """Extrapolate to zero noise.

        Parameters
        ----------
        noisy_values : list of float
            Measured expectation values at the corresponding scale factors.

        Returns
        -------
        float
            Zero-noise extrapolated value.
        """
        result = self._impl.extrapolate(self._scale_factors, noisy_values)
        return float(result.mitigated_value)


# Alias
ZNEMitigator = ZNEResult
