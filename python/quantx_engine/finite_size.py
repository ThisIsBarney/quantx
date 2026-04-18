"""Finite-size scaling and quantum phase-transition analysis (M74).

Three main components:

**binder_cumulant** — compute the Binder cumulant U = 1 − ⟨m⁴⟩/(3⟨m²⟩²)
from order-parameter samples.  U → 0 in the disordered phase (Gaussian
fluctuations) and U → 2/3 in the ordered phase (bimodal distribution).

**entanglement_scaling** — compute the bipartite von Neumann entropy S(A)
for all subsystem sizes A = 1, …, L−1 via SVD of the reshaped state vector,
and fit the 1D CFT formula S(A) = (c/3) ln[(L/π) sin(πA/L)] + const to
extract the central charge *c*.

**FiniteSizeScaling** — locate the critical coupling J_c from Binder cumulant
data at multiple system sizes L:

1. ``binder_crossings()`` finds parameter values where adjacent-size curves
   intersect (linear interpolation).
2. ``fit_critical_point()`` averages those crossings into a single estimate
   of J_c ± σ.
3. ``data_collapse(jc, nu)`` measures how well data from different L collapse
   onto a single scaling curve when plotted against (J − J_c) L^{1/ν}.

Quick start::

    import numpy as np
    from quantx_engine.finite_size import FiniteSizeScaling, binder_cumulant

    # Simulated Binder cumulants near J_c = 1.0
    params = np.linspace(0.7, 1.3, 30)
    sizes  = [8, 16, 32, 64]
    nu     = 1.0  # correlation-length exponent
    J_c    = 1.0
    def logistic(x): return (2/3) / (1 + np.exp(-4 * x))
    binder = np.array([logistic((params - J_c) * L**(1/nu)) for L in sizes])

    fss    = FiniteSizeScaling(sizes, params, binder)
    result = fss.fit_critical_point()
    print(f"J_c = {result.critical_point:.4f} ± {result.critical_point_std:.4f}")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

__all__ = [
    "FiniteSizeError",
    "ScalingResult",
    "EntanglementResult",
    "FiniteSizeScaling",
    "binder_cumulant",
    "entanglement_scaling",
]


# ---------------------------------------------------------------------------
# Exceptions and result types
# ---------------------------------------------------------------------------

class FiniteSizeError(ValueError):
    """Raised for invalid finite-size scaling parameters."""


@dataclass
class ScalingResult:
    """Result of a finite-size scaling analysis.

    Attributes
    ----------
    critical_point:
        Best estimate of the critical coupling J_c (mean of pairwise
        Binder cumulant crossings).
    critical_point_std:
        Standard deviation of the individual crossing estimates.  Small
        when curves cross at a well-defined point; larger when there is
        significant spread (e.g. slow convergence or small system sizes).
    binder_crossings:
        List of individual pairwise crossing values used to compute the
        mean and standard deviation.
    raw:
        Original data passed to :class:`FiniteSizeScaling` (sizes, params,
        binder array) as plain Python objects.
    """
    critical_point: float
    critical_point_std: float
    binder_crossings: "list[float]"
    raw: "dict[str, Any]" = field(default_factory=dict, repr=False)


@dataclass
class EntanglementResult:
    """Result of a bipartite entanglement entropy calculation.

    Attributes
    ----------
    subsystem_sizes:
        Integer array A = 1, 2, …, L−1 (number of sites in the subsystem).
    entropy:
        Von Neumann entropy S(A) = −Tr(ρ_A ln ρ_A) for each A.
    central_charge:
        Central charge *c* extracted from a least-squares fit of the 1D
        CFT formula S(A) = (c/3) ln[(L/π) sin(πA/L)] + const.
        ``None`` when L < 4 (not enough points for a reliable fit).
    fit_residual:
        Mean squared residual of the CFT fit (lower = better).  ``None``
        when the fit was not attempted.
    """
    subsystem_sizes: "np.ndarray"
    entropy: "np.ndarray"
    central_charge: "float | None" = None
    fit_residual: "float | None" = None

    @property
    def max_entropy(self) -> float:
        """Maximum entropy over all bipartitions."""
        return float(np.max(self.entropy))


# ---------------------------------------------------------------------------
# binder_cumulant
# ---------------------------------------------------------------------------

def binder_cumulant(samples: "np.ndarray") -> float:
    """Compute the Binder cumulant U = 1 − ⟨m⁴⟩ / (3⟨m²⟩²).

    The Binder cumulant is a dimensionless ratio that distinguishes phases
    without needing the thermodynamic limit:

    * U → 0   disordered phase (Gaussian fluctuations: ⟨m⁴⟩ = 3⟨m²⟩²)
    * U → 2/3 ordered phase (bimodal: |m| ≈ const)

    At the critical point, Binder cumulant curves for different system sizes
    *L* cross at a universal value U* ∈ (0, 2/3).

    .. note::
       This function uses **raw moments** ⟨m²⟩ and ⟨m⁴⟩ (the standard
       definition in the condensed-matter literature).  These are *not*
       centred cumulants.  Passing samples with a non-zero mean correctly
       gives U ≈ 2/3 for a biased ordered phase (e.g. ferromagnet in an
       external field), which is the physically correct result.

    Parameters
    ----------
    samples:
        1D (or ravelled) array of order-parameter values.  At least 2
        samples are required.

    Returns
    -------
    float
        Binder cumulant U.  Theoretically in [0, 2/3], but may be slightly
        outside this range due to finite sampling.

    Raises
    ------
    FiniteSizeError
        If fewer than 2 samples are provided or ⟨m²⟩ ≈ 0.
    """
    if not _HAS_NUMPY:
        raise FiniteSizeError("numpy is required for binder_cumulant")
    arr = np.asarray(samples, dtype=float).ravel()
    if arr.size < 2:
        raise FiniteSizeError(
            f"binder_cumulant requires at least 2 samples, got {arr.size}"
        )
    m2 = float(np.mean(arr ** 2))
    m4 = float(np.mean(arr ** 4))
    if m2 < 1e-14:
        raise FiniteSizeError(
            "⟨m²⟩ ≈ 0; the order parameter is identically zero and the "
            "Binder cumulant is not defined."
        )
    return 1.0 - m4 / (3.0 * m2 ** 2)


# ---------------------------------------------------------------------------
# entanglement_scaling
# ---------------------------------------------------------------------------

def entanglement_scaling(
    psi: "np.ndarray",
    L: int,
    d: int = 2,
) -> EntanglementResult:
    """Compute bipartite von Neumann entropy S(A) for all bipartitions.

    Reshapes the length-d^L state vector as a (d^A, d^{L−A}) matrix and
    computes the von Neumann entropy

        S(A) = −Tr(ρ_A ln ρ_A) = −∑_i λ_i ln λ_i,

    where λ_i = σ_i² are the eigenvalues of the reduced density matrix ρ_A
    obtained from the singular values σ_i of the reshaped matrix.

    For a 1D critical system described by a CFT with central charge *c*, the
    entropy satisfies

        S(A) = (c/3) ln[(L/π) sin(πA/L)] + s₀.

    A least-squares fit of this formula (for L ≥ 4) returns the central
    charge.

    Parameters
    ----------
    psi:
        State vector, shape (d^L,).  Must be exactly d^L elements.
        Need not be normalised (normalisation is applied internally).

        **Storage convention**: the state is indexed in standard
        tensor-product (row-major / C) order, i.e.
        ``psi[i] = ⟨s_0 s_1 … s_{L-1}|ψ⟩`` where site 0 is the
        most-significant index.  Equivalently,
        ``index = s_0·d^{L-1} + s_1·d^{L-2} + … + s_{L-1}``.
        This is the natural NumPy ordering and the convention used
        throughout QuantX.

    L:
        Number of lattice sites.
    d:
        Local Hilbert-space dimension (default 2 for qubits/spin-1/2).

    Returns
    -------
    EntanglementResult

    Raises
    ------
    FiniteSizeError
        If ``len(psi) != d^L`` or ``psi`` has near-zero norm.
    """
    if not _HAS_NUMPY:
        raise FiniteSizeError("numpy is required for entanglement_scaling")
    if L < 1:
        raise FiniteSizeError(f"L must be >= 1, got {L}")
    if d < 2:
        raise FiniteSizeError(f"d must be >= 2, got {d}")

    psi_arr = np.asarray(psi, dtype=complex).ravel()
    expected_dim = d ** L
    if psi_arr.size != expected_dim:
        raise FiniteSizeError(
            f"psi has length {psi_arr.size}, expected d^L = {d}^{L} = {expected_dim}"
        )
    norm = np.linalg.norm(psi_arr)
    if norm < 1e-14:
        raise FiniteSizeError("psi has near-zero norm")
    psi_arr = psi_arr / norm

    sizes = np.arange(1, L, dtype=int)
    entropy = np.zeros(len(sizes), dtype=float)

    for i, A in enumerate(sizes):
        dA = d ** int(A)
        dB = d ** (L - int(A))
        mat = psi_arr.reshape(dA, dB)
        sv = np.linalg.svd(mat, compute_uv=False)
        prob = sv ** 2
        # Filter numerically-zero singular values to avoid log(0)
        prob = prob[prob > 1e-15]
        entropy[i] = float(-np.sum(prob * np.log(prob)))

    # ── CFT fit for central charge ────────────────────────────────────────
    central_charge: "float | None" = None
    fit_residual: "float | None" = None

    if L >= 4:
        # CFT argument: ξ(A) = (L/π) sin(πA/L)
        xi = (L / np.pi) * np.sin(np.pi * sizes.astype(float) / L)
        log_xi = np.log(np.maximum(xi, 1e-14))
        # Linear system: S = (c/3) * log_xi + s0  ⟶  Ax = b
        A_mat = np.column_stack([log_xi, np.ones(len(log_xi))])
        try:
            coeffs, residuals_sq, *_ = np.linalg.lstsq(A_mat, entropy, rcond=None)
            central_charge = float(3.0 * coeffs[0])
            if len(residuals_sq) > 0 and len(sizes) > 0:
                fit_residual = float(residuals_sq[0] / len(sizes))
            else:
                pred = A_mat @ coeffs
                fit_residual = float(np.mean((entropy - pred) ** 2))
        except np.linalg.LinAlgError:
            pass

    return EntanglementResult(
        subsystem_sizes=sizes,
        entropy=entropy,
        central_charge=central_charge,
        fit_residual=fit_residual,
    )


# ---------------------------------------------------------------------------
# FiniteSizeScaling
# ---------------------------------------------------------------------------

class FiniteSizeScaling:
    """Finite-size scaling analysis for quantum phase transitions.

    Identifies the critical coupling J_c where Binder cumulant curves
    for different system sizes L cross, using linear interpolation of
    adjacent-size pairs.

    Parameters
    ----------
    sizes:
        System sizes L, e.g. ``[8, 16, 32, 64]``.  At least 2 required.
    params:
        Coupling/parameter values (x-axis), e.g. ``np.linspace(0.7, 1.3, 30)``.
        At least 2 required.
    binder:
        2D array of shape ``(len(sizes), len(params))`` containing the
        Binder cumulant for each (L, J) pair.

    Raises
    ------
    FiniteSizeError
        On invalid shapes or fewer than 2 sizes/params.
    """

    def __init__(
        self,
        sizes: "list[int]",
        params: "list[float] | np.ndarray",
        binder: "np.ndarray",
    ):
        if not _HAS_NUMPY:
            raise FiniteSizeError("numpy is required for FiniteSizeScaling")
        if len(sizes) < 2:
            raise FiniteSizeError(
                f"FiniteSizeScaling requires at least 2 system sizes, got {len(sizes)}"
            )
        params_arr = np.asarray(params, dtype=float)
        if len(params_arr) < 2:
            raise FiniteSizeError(
                f"FiniteSizeScaling requires at least 2 parameter values, "
                f"got {len(params_arr)}"
            )
        binder_arr = np.asarray(binder, dtype=float)
        if binder_arr.shape != (len(sizes), len(params_arr)):
            raise FiniteSizeError(
                f"binder array shape {binder_arr.shape} does not match "
                f"(len(sizes)={len(sizes)}, len(params)={len(params_arr)})"
            )
        self.sizes = list(sizes)
        self.params = params_arr
        self.binder = binder_arr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_crossing(
        self, u1: "np.ndarray", u2: "np.ndarray"
    ) -> "float | None":
        """Return the parameter value where two 1D curves cross (linear interp).

        Returns ``None`` if no sign change is found in u1 − u2.
        """
        diff = u1 - u2
        for i in range(len(diff) - 1):
            d0, d1 = float(diff[i]), float(diff[i + 1])
            if d0 * d1 < 0:
                # Strict sign change: genuine crossing between grid points i and i+1.
                # Linear interpolation: 0 = d0 + t*(d1-d0)  →  t = -d0/(d1-d0)
                t = -d0 / (d1 - d0)
                return float(self.params[i] + t * (self.params[i + 1] - self.params[i]))
            if d1 == 0.0:
                # Curves meet exactly at grid point i+1.
                if i + 2 < len(diff):
                    # Only a genuine crossing if the segment *after* i+1 has
                    # the opposite sign to the segment *before* i+1.
                    d2 = float(diff[i + 2])
                    if d0 * d2 < 0:
                        return float(self.params[i + 1])
                # If i+1 is the last point there is no segment after it, so
                # we cannot determine whether this is a crossing or a touch;
                # conservatively skip it.
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def binder_crossings(self) -> "list[float]":
        """Find crossing points of consecutive Binder cumulant curves.

        For each adjacent pair (L_i, L_{i+1}) finds the parameter value
        where U_{L_i}(J) = U_{L_{i+1}}(J) using linear interpolation.

        Returns
        -------
        list[float]
            Crossing parameter values, one per consecutive pair that has a
            crossing in the supplied range.  May be empty.
        """
        crossings = []
        for i in range(len(self.sizes) - 1):
            cross = self._find_crossing(self.binder[i], self.binder[i + 1])
            if cross is not None:
                crossings.append(cross)
        return crossings

    def fit_critical_point(self) -> ScalingResult:
        """Estimate the critical coupling from Binder cumulant crossings.

        Returns
        -------
        ScalingResult
            ``critical_point`` = mean of crossing estimates;
            ``critical_point_std`` = standard deviation (0.0 if only one
            crossing is found).

        Raises
        ------
        FiniteSizeError
            If no crossings are found in the supplied parameter range.
        """
        crossings = self.binder_crossings()
        if not crossings:
            raise FiniteSizeError(
                "No Binder cumulant crossings found in the given parameter "
                "range. Try extending the range or using more system sizes."
            )
        arr = np.array(crossings, dtype=float)
        return ScalingResult(
            critical_point=float(arr.mean()),
            critical_point_std=float(arr.std()),
            binder_crossings=crossings,
            raw={
                "sizes": self.sizes,
                "params": self.params.tolist(),
                "binder": self.binder.tolist(),
            },
        )

    def data_collapse(
        self,
        jc: float,
        nu: float,
        observable: "np.ndarray | None" = None,
        beta_over_nu: float = 0.0,
    ) -> float:
        """Measure data-collapse quality for given (J_c, ν).

        Rescales the parameter axis as x̃ = (J − J_c) · L^{1/ν} and (if
        ``observable`` and ``beta_over_nu`` are provided) the observable
        axis as Ỹ = O · L^{β/ν}, then computes the mean-squared distance
        between adjacent rescaled curves (interpolated onto a common grid).

        A perfect collapse gives cost = 0.

        Parameters
        ----------
        jc:
            Trial critical coupling.
        nu:
            Trial correlation-length exponent (must be > 0).
        observable:
            2D array (n_sizes, n_params) of order-parameter values.
            If ``None``, the Binder cumulant is used (and β/ν = 0).
        beta_over_nu:
            β/ν exponent for y-axis rescaling.  Use 0 for Binder cumulant.

        Returns
        -------
        float
            Mean squared deviation averaged over **all N(N−1)/2 unique
            pairs** of rescaled curves (not just adjacent pairs).  Using
            all pairs makes the metric more robust when sizes span a wide
            range.  Lower is better.  Returns ``float('inf')`` if no
            pair has an overlapping rescaled-x region with at least 2
            points; this can happen when ``jc`` is far outside the
            supplied parameter range or ``nu`` is very small.

        Raises
        ------
        FiniteSizeError
            If ``nu <= 0`` or ``observable`` has wrong shape.
        """
        if nu <= 0.0:
            raise FiniteSizeError(f"nu must be > 0, got {nu}")
        if observable is None:
            obs = self.binder
        else:
            obs = np.asarray(observable, dtype=float)
            if obs.shape != (len(self.sizes), len(self.params)):
                raise FiniteSizeError(
                    f"observable shape {obs.shape} must be "
                    f"({len(self.sizes)}, {len(self.params)})"
                )

        total_cost = 0.0
        count = 0
        for i in range(len(self.sizes)):
            L_i = self.sizes[i]
            x_i = (self.params - jc) * L_i ** (1.0 / nu)
            y_i = obs[i] * L_i ** beta_over_nu
            for j in range(i + 1, len(self.sizes)):
                L_j = self.sizes[j]
                x_j = (self.params - jc) * L_j ** (1.0 / nu)
                y_j = obs[j] * L_j ** beta_over_nu
                # Restrict to overlapping x-range
                x_lo = max(float(x_i.min()), float(x_j.min()))
                x_hi = min(float(x_i.max()), float(x_j.max()))
                mask = (x_j >= x_lo) & (x_j <= x_hi)
                if mask.sum() < 2:
                    continue
                # Interpolate curve i onto curve j's x-grid within overlap
                y_i_on_j = np.interp(x_j[mask], x_i, y_i)
                diff = y_i_on_j - y_j[mask]
                total_cost += float(np.mean(diff ** 2))
                count += 1

        return total_cost / count if count > 0 else float("inf")
