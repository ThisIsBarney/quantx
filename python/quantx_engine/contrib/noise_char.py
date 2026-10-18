"""Noise Characterization Pipeline — M87.

Fits a parametric noise model from RB, XEB, and/or process-tomography
benchmark data.  All fitting is pure NumPy; no SciPy is required.

Quick reference::

    from quantx_engine.contrib.noise_char import NoiseCharacterizer
    from quantx_engine.contrib.rb_qv import StandardRB

    rb_result = StandardRB(p_error=0.02, seed=0).run([1, 5, 10, 20, 50])
    nc = NoiseCharacterizer(gate_time=0.1)
    cn = nc.fit_from_rb(rb_result)
    print(cn.depolarizing_prob)   # ≈ 0.02 (2 × the error per Clifford)
    noise = cn.to_noise_model()   # DepolarizingNoise instance

References
----------
.. [1] Magesan, E. et al. (2012). Efficient measurement of quantum gate error
       by interleaved randomized benchmarking.
       *Phys. Rev. Lett.* 109, 080505.
       https://doi.org/10.1103/PhysRevLett.109.080505

.. [2] Chuang, I. L. & Nielsen, M. A. (1997). Prescription for experimental
       determination of the dynamics of a quantum black box.
       *J. Mod. Opt.* 44, 2455.
       https://doi.org/10.1080/09500349708231894

.. [3] Blume-Kohout, R. et al. (2017). Demonstration of qubit operations below
       a rigorous fault tolerance threshold with gate set tomography.
       *Nat. Commun.* 8, 14485. https://doi.org/10.1038/ncomms14485
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

from .noise import DepolarizingNoise

__all__ = [
    "NoiseCharError",
    "CharacterizedNoise",
    "NoiseCharacterizer",
]


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class NoiseCharError(ValueError):
    """Raised for invalid inputs to noise characterization routines."""


def _require_numpy() -> None:
    if not _HAS_NUMPY:  # pragma: no cover
        raise NoiseCharError("NumPy is required for noise characterization.")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CharacterizedNoise:
    """Result of noise characterization.

    Attributes
    ----------
    depolarizing_prob:
        Per-gate depolarizing probability, clamped to [0, 1].
    t1_estimate:
        Estimated T1 in gate-time units.  ``None`` if not available.
    t2_estimate:
        Estimated T2 in gate-time units.  ``None`` if not available.
    epc:
        Error Per Clifford from RB.  ``None`` if RB was not used.
    xeb_fidelity:
        Mean XEB fidelity.  ``None`` if XEB was not used.
    process_infidelity:
        1 - process_fidelity from process tomography.  ``None`` if tomography
        was not used.
    confidence:
        Heuristic confidence in the fit, in [0, 1].
    sources:
        List of data sources used (e.g. ``['rb', 'xeb']``).
    """

    depolarizing_prob: float
    t1_estimate: "float | None"
    t2_estimate: "float | None"
    epc: "float | None"
    xeb_fidelity: "float | None"
    process_infidelity: "float | None"
    confidence: float
    sources: "list[str]"

    def to_noise_model(self) -> DepolarizingNoise:
        """Convert to :class:`~quantx_engine.contrib.noise.DepolarizingNoise`.

        Returns a ``DepolarizingNoise`` instance suitable for use with the
        ``Estimator``, ``Sampler``, and ``DynamicSimulator`` primitives.
        """
        return DepolarizingNoise(self.depolarizing_prob)


# ---------------------------------------------------------------------------
# NoiseCharacterizer
# ---------------------------------------------------------------------------

class NoiseCharacterizer:
    """Fit a parametric noise model from experimental benchmark data.

    Supports fitting from:

    * :meth:`fit_from_rb`   — Randomized Benchmarking results
    * :meth:`fit_from_xeb`  — Cross-Entropy Benchmarking results
    * :meth:`fit_from_process_tomography` — process chi matrices
    * :meth:`fit`           — combine all available sources

    Parameters
    ----------
    gate_time:
        Physical gate duration in microseconds, used when estimating T1/T2
        relaxation times from the fitted depolarizing probability.
    """

    def __init__(self, gate_time: float = 0.1) -> None:
        _require_numpy()
        if gate_time <= 0.0:
            raise NoiseCharError(
                f"gate_time must be positive, got {gate_time}."
            )
        self.gate_time = float(gate_time)

    # ------------------------------------------------------------------
    # fit_from_rb
    # ------------------------------------------------------------------

    def fit_from_rb(self, rb_result) -> CharacterizedNoise:
        """Extract a noise model from a Randomized Benchmarking result.

        Uses the single-qubit approximation::

            p_dep = 2 * EPC

        which is exact for a 1-qubit depolarizing channel where
        ``EPC = (d-1)/d * (1-p)`` with d=2.

        Parameters
        ----------
        rb_result:
            An :class:`~quantx_engine.contrib.rb_qv.RBResult` instance.

        Returns
        -------
        CharacterizedNoise
        """
        _require_numpy()
        self._validate_rb_result(rb_result)

        epc = float(rb_result.epc)
        if not (0.0 <= epc <= 1.0):
            raise NoiseCharError(
                f"rb_result.epc must be in [0, 1], got {epc}."
            )

        # Standard single-qubit approximation: p_dep = 2 * EPC
        p_dep = float(np.clip(2.0 * epc, 0.0, 1.0))

        # Heuristic T1 estimate: p_dep ≈ gate_time / T1 for small p_dep
        t1_est: "float | None" = None
        if p_dep > 0.0:
            t1_est = self.gate_time / p_dep

        return CharacterizedNoise(
            depolarizing_prob=p_dep,
            t1_estimate=t1_est,
            t2_estimate=None,
            epc=epc,
            xeb_fidelity=None,
            process_infidelity=None,
            confidence=1.0,
            sources=["rb"],
        )

    # ------------------------------------------------------------------
    # fit_from_xeb
    # ------------------------------------------------------------------

    def fit_from_xeb(self, xeb_result) -> CharacterizedNoise:
        """Extract a noise model from a Cross-Entropy Benchmarking result.

        Fits the exponential decay model ``F_XEB(d) = A * r^d`` via
        log-linear regression (``np.polyfit``) and estimates::

            p_dep ≈ 1 - r

        Parameters
        ----------
        xeb_result:
            An :class:`~quantx_engine.contrib.xeb.XEBResult` instance.

        Returns
        -------
        CharacterizedNoise
        """
        _require_numpy()
        self._validate_xeb_result(xeb_result)

        depths = np.array(xeb_result.depths, dtype=float)
        fidelities = np.array(xeb_result.xeb_fidelities, dtype=float)

        if len(depths) < 2:
            raise NoiseCharError(
                "fit_from_xeb requires at least 2 depth points for regression."
            )

        # Clamp fidelities to a small positive value before taking log
        fidelities_clamped = np.maximum(fidelities, 1e-10)
        log_fid = np.log(fidelities_clamped)

        # Fit log(F) = log(A) + d * log(r)  =>  polyfit degree-1
        coeffs = np.polyfit(depths, log_fid, 1)
        log_r = float(coeffs[0])  # slope = log(r)

        r = float(np.exp(log_r))
        r = float(np.clip(r, 0.0, 1.0))

        p_dep = float(np.clip(1.0 - r, 0.0, 1.0))
        mean_fidelity = float(np.mean(fidelities))

        t1_est: "float | None" = None
        if p_dep > 0.0:
            t1_est = self.gate_time / p_dep

        return CharacterizedNoise(
            depolarizing_prob=p_dep,
            t1_estimate=t1_est,
            t2_estimate=None,
            epc=None,
            xeb_fidelity=mean_fidelity,
            process_infidelity=None,
            confidence=1.0,
            sources=["xeb"],
        )

    # ------------------------------------------------------------------
    # fit_from_process_tomography
    # ------------------------------------------------------------------

    def fit_from_process_tomography(
        self,
        process_result,
        process_target_result=None,
    ) -> CharacterizedNoise:
        """Extract a noise model from process tomography chi matrices.

        Uses the relation between process infidelity and the depolarizing
        probability for an n-qubit system (d = 2^n)::

            p_dep = d**2 / (d**2 - 1) * process_infidelity

        Parameters
        ----------
        process_result:
            Either a ``ProcessTomography`` instance (whose :meth:`chi_matrix`
            will be used with a provided target), or a ``numpy.ndarray``
            representing a chi matrix directly.
        process_target_result:
            Target chi matrix as ``numpy.ndarray``, or ``None``.  If
            ``process_result`` is a ``numpy.ndarray``, this argument is the
            target chi matrix used to compute process infidelity.  Passing
            ``None`` assumes the target is an ideal (identity-like) process
            and uses ``process_infidelity = 1 - Tr(chi)``.

        Returns
        -------
        CharacterizedNoise
        """
        _require_numpy()

        chi_actual, chi_target = self._resolve_chi_matrices(
            process_result, process_target_result
        )

        # process_fidelity = Tr(chi_actual @ chi_target†)
        f_proc = float(np.real(np.trace(chi_actual @ chi_target.conj().T)))
        f_proc = float(np.clip(f_proc, 0.0, 1.0))
        process_infidelity = 1.0 - f_proc

        # Determine n_qubits from chi matrix size: chi is (4^n x 4^n)
        n_ops = chi_actual.shape[0]  # 4^n
        n_qubits = int(round(math.log2(n_ops) / 2))
        d = 1 << n_qubits  # 2^n

        # The depolarizing channel (1 - p)rho + p I/d has process fidelity
        # 1 - p (d^2 - 1)/d^2, so p = d^2/(d^2 - 1) * process infidelity. (This used
        # (d + 1)/d, the factor that links average-gate and process infidelity: it read
        # DepolarizingNoise(0.1) back as 0.1125.)
        p_dep = float(np.clip(d * d / (d * d - 1) * process_infidelity, 0.0, 1.0))

        t1_est: "float | None" = None
        if p_dep > 0.0:
            t1_est = self.gate_time / p_dep

        return CharacterizedNoise(
            depolarizing_prob=p_dep,
            t1_estimate=t1_est,
            t2_estimate=None,
            epc=None,
            xeb_fidelity=None,
            process_infidelity=float(process_infidelity),
            confidence=1.0,
            sources=["process_tomography"],
        )

    # ------------------------------------------------------------------
    # fit (combine multiple sources)
    # ------------------------------------------------------------------

    def fit(
        self,
        rb_result=None,
        xeb_result=None,
        process_chi=None,
        target_chi=None,
    ) -> CharacterizedNoise:
        """Combine all available data sources into a single noise estimate.

        If multiple sources are provided, the final ``depolarizing_prob`` is
        a weighted average of per-source estimates (uniform weights).
        The ``confidence`` is the fraction of sources whose estimates agree
        within a factor of 2 of the mean estimate.

        At least one source must be provided.

        Parameters
        ----------
        rb_result:
            Optional :class:`~quantx_engine.contrib.rb_qv.RBResult`.
        xeb_result:
            Optional :class:`~quantx_engine.contrib.xeb.XEBResult`.
        process_chi:
            Optional chi matrix (``numpy.ndarray``) from process tomography.
        target_chi:
            Optional target chi matrix (``numpy.ndarray``).

        Returns
        -------
        CharacterizedNoise
        """
        _require_numpy()

        estimates: "list[float]" = []
        sources: "list[str]" = []
        epc: "float | None" = None
        xeb_fidelity: "float | None" = None
        process_infidelity: "float | None" = None

        if rb_result is not None:
            cn_rb = self.fit_from_rb(rb_result)
            estimates.append(cn_rb.depolarizing_prob)
            sources.append("rb")
            epc = cn_rb.epc

        if xeb_result is not None:
            cn_xeb = self.fit_from_xeb(xeb_result)
            estimates.append(cn_xeb.depolarizing_prob)
            sources.append("xeb")
            xeb_fidelity = cn_xeb.xeb_fidelity

        if process_chi is not None:
            cn_proc = self.fit_from_process_tomography(process_chi, target_chi)
            estimates.append(cn_proc.depolarizing_prob)
            sources.append("process_tomography")
            process_infidelity = cn_proc.process_infidelity

        if not estimates:
            raise NoiseCharError(
                "fit() requires at least one of: rb_result, xeb_result, "
                "process_chi."
            )

        p_arr = np.array(estimates, dtype=float)
        p_mean = float(np.mean(p_arr))
        p_dep = float(np.clip(p_mean, 0.0, 1.0))

        # Confidence = fraction of estimates within 2x of the mean
        if p_mean > 0.0:
            ratios = p_arr / p_mean
            n_agree = int(np.sum((ratios >= 0.5) & (ratios <= 2.0)))
        else:
            # All estimates are zero — perfect agreement
            n_agree = len(estimates)
        confidence = float(n_agree) / float(len(estimates))

        t1_est: "float | None" = None
        if p_dep > 0.0:
            t1_est = self.gate_time / p_dep

        return CharacterizedNoise(
            depolarizing_prob=p_dep,
            t1_estimate=t1_est,
            t2_estimate=None,
            epc=epc,
            xeb_fidelity=xeb_fidelity,
            process_infidelity=process_infidelity,
            confidence=confidence,
            sources=sources,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_rb_result(rb_result) -> None:
        """Raise NoiseCharError if rb_result is not a valid RBResult-like object."""
        required = ("epc", "survival_probs", "sequence_lengths")
        for attr in required:
            if not hasattr(rb_result, attr):
                raise NoiseCharError(
                    f"rb_result must have attribute '{attr}'. "
                    f"Got type {type(rb_result).__name__}."
                )
        if not rb_result.survival_probs:
            raise NoiseCharError(
                "rb_result.survival_probs must not be empty."
            )
        if not rb_result.sequence_lengths:
            raise NoiseCharError(
                "rb_result.sequence_lengths must not be empty."
            )

    @staticmethod
    def _validate_xeb_result(xeb_result) -> None:
        """Raise NoiseCharError if xeb_result is not a valid XEBResult-like object."""
        required = ("depths", "xeb_fidelities", "mean_fidelity")
        for attr in required:
            if not hasattr(xeb_result, attr):
                raise NoiseCharError(
                    f"xeb_result must have attribute '{attr}'. "
                    f"Got type {type(xeb_result).__name__}."
                )
        if not xeb_result.depths:
            raise NoiseCharError("xeb_result.depths must not be empty.")
        if not xeb_result.xeb_fidelities:
            raise NoiseCharError(
                "xeb_result.xeb_fidelities must not be empty."
            )
        if len(xeb_result.depths) != len(xeb_result.xeb_fidelities):
            raise NoiseCharError(
                "xeb_result.depths and xeb_result.xeb_fidelities must have "
                "the same length."
            )

    @staticmethod
    def _resolve_chi_matrices(
        process_result,
        process_target_result,
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Return (chi_actual, chi_target) as numpy arrays.

        Accepts either:
        - A numpy array for ``process_result`` (chi matrix directly), with an
          optional numpy array for ``process_target_result``.
        - Any object that is not an ndarray is rejected with NoiseCharError.
        """
        if not _HAS_NUMPY:  # pragma: no cover
            raise NoiseCharError("NumPy is required.")

        # Resolve actual chi
        chi_actual = np.asarray(process_result, dtype=complex)
        if chi_actual.ndim != 2 or chi_actual.shape[0] != chi_actual.shape[1]:
            raise NoiseCharError(
                f"process_result must be a square 2-D array, "
                f"got shape {chi_actual.shape}."
            )
        n_ops = chi_actual.shape[0]
        # Check that n_ops is a power of 4 (= 4^n for some n >= 1)
        log4 = math.log2(n_ops) / 2.0
        if abs(log4 - round(log4)) > 1e-9 or round(log4) < 1:
            raise NoiseCharError(
                f"Chi matrix size must be 4^n for n >= 1, got {n_ops}."
            )

        # Resolve target chi
        if process_target_result is None:
            # Default target: diagonal matrix with 1/n_ops on the diagonal
            # (identity process in Pauli basis: chi = I/n_ops^2 normalised
            # to have Tr = 1, equivalent to identity unitary).
            # For identity unitary U=I: c_m = Tr(P_m I)/d = Tr(P_m)/d.
            # Tr(I) = d, Tr(other Paulis) = 0  => c_I = 1, others = 0
            # => chi = outer([1, 0, ...], [1, 0, ...]) = diag([1,0,...])
            chi_target = np.zeros((n_ops, n_ops), dtype=complex)
            chi_target[0, 0] = 1.0
        else:
            chi_target = np.asarray(process_target_result, dtype=complex)
            if chi_target.shape != chi_actual.shape:
                raise NoiseCharError(
                    f"process_target_result shape {chi_target.shape} does not "
                    f"match process_result shape {chi_actual.shape}."
                )

        return chi_actual, chi_target
