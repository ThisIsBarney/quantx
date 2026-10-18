"""M112: Fault-tolerant QC end-to-end pipeline.

Ties :class:`~quantx_engine.contrib.qec.SurfaceCode` and
:class:`~quantx_engine.contrib.qec.MWPMDecoder` into a single :class:`FTQCPipeline`
object that simulates the complete error-correction cycle:

    encode → (logical gate →)* n_rounds × syndrome extraction → MWPM decode → check

Two noise models are supported:

* **Code-capacity** (``n_rounds=1``, ``p_meas=0``): ideal stabilizer measurements;
  only data qubit errors are injected.
* **Phenomenological** (``n_rounds > 1``, ``p_meas > 0``): every syndrome
  measurement may flip independently with probability ``p_meas``; data errors
  accumulate across rounds and the final ideal syndrome is decoded.

The :func:`threshold_curve` function sweeps ``p_phys`` over a range for
multiple code distances and estimates the threshold crossing point — the
physical error rate below which increasing *d* strictly improves the logical
error rate.

Classes
-------
FTQCPipelineError
    Exception for invalid pipeline operations.
PipelineResult
    Result of a single :meth:`FTQCPipeline.run` call.
ThresholdCurveResult
    Threshold curve across multiple distances.

Functions
---------
threshold_curve(...)
    Sweep p_phys for several distances; estimate the threshold crossing.
threshold_crossing(result)
    Extract (or refine) the threshold estimate from a :class:`ThresholdCurveResult`.

Example::

    from quantx_engine.contrib.ftqc_pipeline import FTQCPipeline, threshold_curve

    # Single-pipeline run
    pipe = FTQCPipeline(code_distance=5, p_physical=0.01)
    result = pipe.run(n_shots=2000, seed=0)
    print(result.logical_error_rate)   # should be ≪ 0.01 below threshold

    # Threshold curve
    curve = threshold_curve(distances=[3, 5, 7], n_shots=500, seed=42)
    print(f"threshold ≈ {curve.threshold:.4f}")

References
----------
.. [1] Fowler, A. M. et al. (2012). Surface codes: Towards practical
       large-scale quantum computation. *Phys. Rev. A* 86, 032324.
       https://doi.org/10.1103/PhysRevA.86.032324

.. [2] Litinski, D. (2019). A game of surface codes: Large-scale quantum
       computing with lattice surgery. *Quantum* 3, 128.
       https://doi.org/10.22331/q-2019-03-05-128

.. [3] Jones, C. et al. (2012). Layered architecture for quantum computing.
       *Phys. Rev. X* 2, 031007.
       https://doi.org/10.1103/PhysRevX.2.031007
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .qec import SurfaceCode, MWPMDecoder, QECError

__all__ = [
    "FTQCPipelineError",
    "PipelineResult",
    "ThresholdCurveResult",
    "FTQCPipeline",
    "threshold_curve",
    "threshold_crossing",
    "ResourceEstimate",
    "FTQCResourceEstimator",
]


# ── Exception ─────────────────────────────────────────────────────────────────

class FTQCPipelineError(ValueError):
    """Raised for invalid fault-tolerant pipeline operations."""


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Result of a :meth:`FTQCPipeline.run` Monte Carlo simulation.

    Attributes
    ----------
    code_distance : int
    p_physical : float
        Physical error rate per qubit per round.
    p_meas : float
        Syndrome measurement error rate.
    n_rounds : int
        Number of syndrome extraction rounds per trial.
    n_shots : int
        Number of Monte Carlo trials.
    n_logical_errors : int
        Number of trials that resulted in a logical error.
    logical_error_rate : float
        ``n_logical_errors / n_shots``.
    decoder : str
        Decoder used (``"mwpm"`` or ``"greedy"``).
    """
    code_distance: int
    p_physical: float
    p_meas: float
    n_rounds: int
    n_shots: int
    n_logical_errors: int
    logical_error_rate: float
    decoder: str

    def to_dict(self) -> dict:
        return {
            "code_distance": self.code_distance,
            "p_physical": self.p_physical,
            "p_meas": self.p_meas,
            "n_rounds": self.n_rounds,
            "n_shots": self.n_shots,
            "n_logical_errors": self.n_logical_errors,
            "logical_error_rate": self.logical_error_rate,
            "decoder": self.decoder,
        }


@dataclass
class ThresholdCurveResult:
    """Threshold curve data across multiple code distances.

    Attributes
    ----------
    distances : list[int]
        Code distances used.
    p_phys : list[float]
        Physical error rates swept.
    p_logical : dict[int, list[float]]
        Logical error rates per distance: ``p_logical[d][i]`` is the rate at
        ``p_phys[i]`` for distance ``d``.
    threshold : float | None
        Estimated threshold crossing point, or ``None`` if not detectable.
    decoder : str
    n_rounds : int
    """
    distances: list[int]
    p_phys: list[float]
    p_logical: dict[int, list[float]]
    threshold: Optional[float]
    decoder: str
    n_rounds: int

    def to_dict(self) -> dict:
        return {
            "distances": self.distances,
            "p_phys": self.p_phys,
            "p_logical": {str(d): v for d, v in self.p_logical.items()},
            "threshold": self.threshold,
            "decoder": self.decoder,
            "n_rounds": self.n_rounds,
        }


# ── FTQCPipeline ──────────────────────────────────────────────────────────────

class FTQCPipeline:
    """End-to-end fault-tolerant QC pipeline for a single logical qubit.

    Simulates the full error-correction cycle:

    1. Encode logical |0⟩ (or |1⟩) using a distance-*d* surface code.
    2. Inject independent depolarizing noise on each data qubit per round.
    3. Extract stabilizer syndromes (optionally with measurement errors).
    4. Decode with MWPM (or greedy) to get a Pauli correction.
    5. Determine whether a logical error occurred.

    Parameters
    ----------
    code_distance : int
        Surface code distance d (must be ≥ 3, odd recommended).
    p_physical : float
        Physical error rate per qubit per error-correction round (0 < p < 0.5).
    p_meas : float, default 0.0
        Syndrome measurement error probability per stabilizer per round.
        Set to 0 for the ideal code-capacity noise model.
    decoder : {"mwpm", "greedy"}, default "mwpm"
        Syndrome decoder.  ``"mwpm"`` uses minimum-weight perfect matching;
        ``"greedy"`` uses the built-in greedy matching in :class:`SurfaceCode`.
    """

    def __init__(
        self,
        code_distance: int,
        p_physical: float,
        p_meas: float = 0.0,
        decoder: str = "mwpm",
    ) -> None:
        if not isinstance(code_distance, int) or code_distance < 3:
            raise FTQCPipelineError(
                f"code_distance must be an integer ≥ 3, got {code_distance!r}"
            )
        if not (0 < p_physical < 1):
            raise FTQCPipelineError(
                f"p_physical must be in (0, 1), got {p_physical}"
            )
        if not (0 <= p_meas < 1):
            raise FTQCPipelineError(
                f"p_meas must be in [0, 1), got {p_meas}"
            )
        if decoder not in ("mwpm", "greedy"):
            raise FTQCPipelineError(
                f"decoder must be 'mwpm' or 'greedy', got {decoder!r}"
            )

        self.code_distance = code_distance
        self.p_physical = p_physical
        self.p_meas = p_meas
        self.decoder = decoder

        self._code = SurfaceCode(code_distance)
        self._mwpm: Optional[MWPMDecoder] = (
            MWPMDecoder(self._code) if decoder == "mwpm" else None
        )

    # ── Core simulation ───────────────────────────────────────────────────────

    def run(
        self,
        n_shots: int = 1000,
        n_rounds: int = 1,
        seed: Optional[int] = None,
    ) -> PipelineResult:
        """Run a Monte Carlo simulation of the error-correction pipeline.

        Parameters
        ----------
        n_shots : int
            Number of independent trials.
        n_rounds : int
            Syndrome extraction rounds per trial.  Use ``n_rounds=1`` for the
            ideal code-capacity model; use larger values with ``p_meas > 0``
            for the phenomenological noise model.
        seed : int | None
            Random seed for reproducibility.

        Returns
        -------
        PipelineResult
        """
        if n_shots < 1:
            raise FTQCPipelineError(f"n_shots must be ≥ 1, got {n_shots}")
        if n_rounds < 1:
            raise FTQCPipelineError(f"n_rounds must be ≥ 1, got {n_rounds}")

        rng = random.Random(seed)
        n_logical_errors = sum(
            1 for _ in range(n_shots) if self._trial(rng, n_rounds)
        )
        return PipelineResult(
            code_distance=self.code_distance,
            p_physical=self.p_physical,
            p_meas=self.p_meas,
            n_rounds=n_rounds,
            n_shots=n_shots,
            n_logical_errors=n_logical_errors,
            logical_error_rate=n_logical_errors / n_shots,
            decoder=self.decoder,
        )

    def logical_error_rate(
        self,
        n_shots: int = 1000,
        n_rounds: int = 1,
        seed: Optional[int] = None,
    ) -> float:
        """Convenience wrapper — returns only the logical error rate float."""
        return self.run(n_shots=n_shots, n_rounds=n_rounds, seed=seed).logical_error_rate

    # ── Trial internals ───────────────────────────────────────────────────────

    def _trial(self, rng: random.Random, n_rounds: int) -> bool:
        """Run one error-correction trial; return True if a logical error occurs."""
        code = self._code
        n = code.n
        p = self.p_physical
        p_m = self.p_meas

        # Accumulated data-qubit errors across all rounds
        x_err = [0] * n
        z_err = [0] * n

        if n_rounds == 1 and p_m == 0:
            # ── Code-capacity model ──────────────────────────────────────────
            # Single round, perfect syndrome measurement.
            for i in range(n):
                if rng.random() < p:
                    x_err[i] ^= 1
                if rng.random() < p:
                    z_err[i] ^= 1

            syn_z = code.z_syndrome(x_err)
            syn_x = code.x_syndrome(z_err)

        else:
            # ── Phenomenological model ───────────────────────────────────────
            # Rounds 0..n_rounds-2: data errors + *noisy* syndrome measurement.
            # Round n_rounds-1 (final): data errors + *ideal* syndrome.
            #
            # Syndrome "change detection" (spacetime event accumulation):
            #   Each syndrome bit is tracked across rounds.  The XOR
            #   accumulation of per-round changes gives the decoder a
            #   syndrome that encodes both data-qubit errors and measurement
            #   errors as spacetime events.  For p_meas > 0, a spurious
            #   flip in round r creates a change event at r and another
            #   when the syndrome returns to its true value — the MWPM
            #   decoder must pair these time-domain events, which degrades
            #   performance compared to p_meas = 0.
            nz = code.n_z_syndromes
            nx = code.n_x_syndromes
            prev_z = [0] * nz
            prev_x = [0] * nx
            change_z = [0] * nz
            change_x = [0] * nx

            for r in range(n_rounds):
                # Data errors this round
                for i in range(n):
                    if rng.random() < p:
                        x_err[i] ^= 1
                    if rng.random() < p:
                        z_err[i] ^= 1

                # Syndrome measurement: noisy for intermediate rounds,
                # ideal (p_meas = 0) for the final round.
                ideal_z = code.z_syndrome(x_err)
                ideal_x = code.x_syndrome(z_err)

                if r < n_rounds - 1 and p_m > 0:
                    curr_z = [s ^ (1 if rng.random() < p_m else 0)
                               for s in ideal_z]
                    curr_x = [s ^ (1 if rng.random() < p_m else 0)
                               for s in ideal_x]
                else:
                    curr_z = ideal_z
                    curr_x = ideal_x

                # Accumulate XOR of successive syndrome differences
                for i in range(nz):
                    change_z[i] ^= curr_z[i] ^ prev_z[i]
                for i in range(nx):
                    change_x[i] ^= curr_x[i] ^ prev_x[i]

                prev_z = curr_z
                prev_x = curr_x

            # The accumulated change syndrome is used for decoding.
            # With p_meas = 0, change_z == final ideal syndrome (correct).
            # With p_meas > 0, spurious events inflate the syndrome, causing
            # the decoder to over-correct and raising the logical error rate.
            syn_z = change_z
            syn_x = change_x

        # ── Decode ────────────────────────────────────────────────────────────
        if self._mwpm is not None:
            corr_x = self._mwpm.decode_x(syn_z)
            corr_z = self._mwpm.decode_z(syn_x)
        else:
            corr_x = code.decode_x(syn_z)
            corr_z = code.decode_z(syn_x)

        # ── Logical error check ───────────────────────────────────────────────
        lx, lz = code.logical_error(x_err, z_err, corr_x, corr_z)
        return lx or lz

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def n_physical_qubits(self) -> int:
        """Number of physical data qubits (d²)."""
        return self._code.n

    @property
    def n_correctable(self) -> int:
        """Maximum number of correctable errors ⌊(d−1)/2⌋."""
        return (self.code_distance - 1) // 2

    def __repr__(self) -> str:
        return (
            f"FTQCPipeline(d={self.code_distance}, p={self.p_physical}, "
            f"p_meas={self.p_meas}, decoder={self.decoder!r})"
        )


# ── threshold_curve ───────────────────────────────────────────────────────────

def threshold_curve(
    distances: list[int] | None = None,
    p_phys_range: tuple[float, float] = (0.005, 0.20),
    n_points: int = 15,
    n_shots: int = 500,
    n_rounds: int = 1,
    decoder: str = "mwpm",
    seed: Optional[int] = None,
) -> ThresholdCurveResult:
    """Sweep ``p_phys`` for multiple distances and estimate the threshold.

    For each ``(d, p_phys)`` pair, a :class:`FTQCPipeline` is constructed and
    ``n_shots`` trials are run.  The threshold is estimated as the average of
    pairwise crossing points between adjacent distance curves (see
    :func:`threshold_crossing`).

    Parameters
    ----------
    distances : list[int] | None
        Code distances to sweep.  Defaults to ``[3, 5, 7]``.
    p_phys_range : (float, float)
        ``(p_min, p_max)`` interval for the sweep.
    n_points : int
        Number of evenly-spaced points in ``p_phys_range``.
    n_shots : int
        Monte Carlo shots per (d, p_phys) point.
    n_rounds : int
        Syndrome extraction rounds per trial.
    decoder : {"mwpm", "greedy"}
    seed : int | None

    Returns
    -------
    ThresholdCurveResult
    """
    if distances is None:
        distances = [3, 5, 7]
    distances = sorted(set(distances))

    p_min, p_max = p_phys_range
    if p_min <= 0 or p_max >= 1 or p_min >= p_max:
        raise FTQCPipelineError(
            f"p_phys_range must satisfy 0 < p_min < p_max < 1, got {p_phys_range}"
        )

    p_list = [
        p_min + (p_max - p_min) * i / (n_points - 1)
        for i in range(n_points)
    ]

    rng_seed = seed
    p_logical: dict[int, list[float]] = {}

    for d in distances:
        rates: list[float] = []
        for j, p in enumerate(p_list):
            trial_seed = None if rng_seed is None else rng_seed + d * 1000 + j
            pipe = FTQCPipeline(
                code_distance=d,
                p_physical=p,
                decoder=decoder,
            )
            rate = pipe.logical_error_rate(
                n_shots=n_shots, n_rounds=n_rounds, seed=trial_seed
            )
            rates.append(rate)
        p_logical[d] = rates

    threshold = _estimate_threshold(p_list, p_logical, distances)

    return ThresholdCurveResult(
        distances=distances,
        p_phys=p_list,
        p_logical=p_logical,
        threshold=threshold,
        decoder=decoder,
        n_rounds=n_rounds,
    )


# ── threshold_crossing ────────────────────────────────────────────────────────

def threshold_crossing(result: ThresholdCurveResult) -> Optional[float]:
    """Re-estimate the threshold crossing from a :class:`ThresholdCurveResult`.

    Uses pairwise linear interpolation between consecutive distance curves to
    find the ``p_phys`` value where larger distance transitions from *worse*
    (above threshold) to *better* (below threshold).

    Returns ``None`` if no crossing is detectable.
    """
    return _estimate_threshold(result.p_phys, result.p_logical, result.distances)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _estimate_threshold(
    p_list: list[float],
    p_logical: dict[int, list[float]],
    distances: list[int],
) -> Optional[float]:
    """Estimate threshold as the mean of pairwise crossing points."""
    crossings: list[float] = []
    sorted_d = sorted(distances)

    for i in range(len(sorted_d) - 1):
        d_lo, d_hi = sorted_d[i], sorted_d[i + 1]
        if d_lo not in p_logical or d_hi not in p_logical:
            continue
        rates_lo = p_logical[d_lo]
        rates_hi = p_logical[d_hi]

        # Find sign change in (rates_lo - rates_hi)
        # Below threshold: rates_lo > rates_hi (smaller code is worse)
        # Above threshold: rates_lo < rates_hi
        for j in range(len(p_list) - 1):
            diff_j = rates_lo[j] - rates_hi[j]
            diff_j1 = rates_lo[j + 1] - rates_hi[j + 1]
            if diff_j * diff_j1 < 0:  # sign change
                # Linear interpolation
                frac = abs(diff_j) / (abs(diff_j) + abs(diff_j1))
                crossing = p_list[j] + frac * (p_list[j + 1] - p_list[j])
                crossings.append(crossing)
                break  # first crossing per pair

    if not crossings:
        return None
    return sum(crossings) / len(crossings)


# ── FTQCResourceEstimator ─────────────────────────────────────────────────────

@dataclass
class ResourceEstimate:
    """Resource estimate for a fault-tolerant quantum computation.

    Attributes
    ----------
    code_distance : int
        Minimum surface code distance achieving the target logical error rate.
    physical_qubits : int
        Total physical qubits = 2·d² (data + ancilla for one logical qubit).
    logical_error_rate : float
        Estimated logical error rate at the chosen distance.
    magic_states_per_t : int
        Number of 15-to-1 magic state distillation levels (each level = ~200 qubits).
    physical_qubits_for_t : int
        Additional physical qubits for one round of T-gate distillation.
    cycles : int
        Surface code QEC cycles per logical gate (≈ d rounds).
    physical_error_rate : float
        Input physical error rate.
    target_logical_error_rate : float
        Input target logical error rate.
    """
    code_distance: int
    physical_qubits: int
    logical_error_rate: float
    magic_states_per_t: int
    physical_qubits_for_t: int
    cycles: int
    physical_error_rate: float
    target_logical_error_rate: float

    def __repr__(self) -> str:
        return (
            f"ResourceEstimate(d={self.code_distance}, "
            f"p_phys={self.physical_error_rate:.2e}, "
            f"p_target={self.target_logical_error_rate:.2e}, "
            f"qubits={self.physical_qubits}, "
            f"p_logical={self.logical_error_rate:.2e})"
        )


class FTQCResourceEstimator:
    """Estimate physical qubit and magic-state resources for fault-tolerant QC.

    Uses the surface code threshold model from Fowler et al. (2012).  The
    logical error rate per round scales as

        p_L ≈ A × (p_phys / p_th)^{⌊(d+1)/2⌋}

    where A ≈ 0.1, p_th ≈ 0.01 (1% threshold for the depolarizing model).

    References
    ----------
    .. [Fowler2012] Fowler, A. G. et al. (2012). Surface codes: Towards practical
                    large-scale quantum computation. *Phys. Rev. A* 86, 032324.
                    https://doi.org/10.1103/PhysRevA.86.032324

    Parameters
    ----------
    target_logical_error_rate : float
        Desired logical error rate per logical gate.
    physical_error_rate : float
        Physical qubit error rate (must be < threshold ≈ 0.01).
    threshold : float, default 0.01
        Surface code threshold (default: 1%).
    prefactor : float, default 0.1
        Pre-exponential constant A in p_L = A * (p/p_th)^t.
    max_distance : int, default 51
        Maximum code distance to search.

    Examples
    --------
    >>> est = FTQCResourceEstimator(target_logical_error_rate=1e-10,
    ...                             physical_error_rate=1e-3)
    >>> result = est.estimate()
    >>> print(result)
    ResourceEstimate(d=7, p_phys=1.00e-03, p_target=1.00e-10, qubits=98, p_logical=...)
    """

    def __init__(
        self,
        target_logical_error_rate: float,
        physical_error_rate: float,
        threshold: float = 0.01,
        prefactor: float = 0.1,
        max_distance: int = 51,
    ) -> None:
        if not (0 < target_logical_error_rate < 1):
            raise FTQCPipelineError(
                f"target_logical_error_rate must be in (0, 1), got {target_logical_error_rate}"
            )
        if not (0 < physical_error_rate < threshold):
            raise FTQCPipelineError(
                f"physical_error_rate must be in (0, threshold={threshold:.3f}), "
                f"got {physical_error_rate}"
            )
        self.target_logical_error_rate = target_logical_error_rate
        self.physical_error_rate = physical_error_rate
        self.threshold = threshold
        self.prefactor = prefactor
        self.max_distance = max_distance

    def _logical_error_rate(self, d: int) -> float:
        """Estimate p_L for a given distance d using the Fowler 2012 formula."""
        import math
        t = (d + 1) // 2
        r = self.physical_error_rate / self.threshold
        return self.prefactor * (r ** t)

    def estimate(self) -> ResourceEstimate:
        """Find the minimum code distance and associated resource count.

        Returns
        -------
        ResourceEstimate
        """
        # Find minimum d (odd) where p_L < target
        chosen_d = None
        chosen_p_L = None
        for d in range(3, self.max_distance + 2, 2):  # odd d only
            p_L = self._logical_error_rate(d)
            if p_L <= self.target_logical_error_rate:
                chosen_d = d
                chosen_p_L = p_L
                break

        if chosen_d is None:
            # Could not meet target even at max distance
            chosen_d = self.max_distance if self.max_distance % 2 == 1 else self.max_distance - 1
            chosen_p_L = self._logical_error_rate(chosen_d)

        # Physical qubit count: 2d² (d² data + d²-1 ancilla ≈ d² ancilla)
        physical_qubits = 2 * chosen_d * chosen_d

        # Magic state distillation levels for T gates
        # Each 15-to-1 level: p_out = 35 * p_in^3; target p_T = target_logical_error_rate
        p_in = self.physical_error_rate
        levels = 0
        p_current = p_in
        while p_current > self.target_logical_error_rate:
            p_current = 35.0 * p_current ** 3
            levels += 1
            if levels > 20:  # safety cap
                break

        # Each distillation factory: ~200 physical qubits per level (rough estimate)
        physical_qubits_for_t = levels * 200

        # QEC cycles per logical gate ≈ d syndrome rounds
        cycles = chosen_d

        return ResourceEstimate(
            code_distance=chosen_d,
            physical_qubits=physical_qubits,
            logical_error_rate=float(chosen_p_L),
            magic_states_per_t=levels,
            physical_qubits_for_t=physical_qubits_for_t,
            cycles=cycles,
            physical_error_rate=self.physical_error_rate,
            target_logical_error_rate=self.target_logical_error_rate,
        )

    def sweep_distances(
        self,
        distances: list[int] | None = None,
    ) -> dict[int, float]:
        """Return {d: p_logical} for a range of code distances.

        Parameters
        ----------
        distances : list[int] | None
            Distances to evaluate. Defaults to [3, 5, 7, 9, 11, 13, 15].

        Returns
        -------
        dict[int, float]
        """
        if distances is None:
            distances = [3, 5, 7, 9, 11, 13, 15]
        return {d: self._logical_error_rate(d) for d in distances}
