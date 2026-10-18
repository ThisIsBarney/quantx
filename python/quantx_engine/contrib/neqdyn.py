"""M114: Non-equilibrium quantum dynamics.

Three complementary tools for studying systems far from equilibrium:

**Keldysh Green's functions** (:class:`KeldyshSolver`)
    Computes the single-particle retarded Green's function
    G^R(t) = −iθ(t)⟨{c(t), c†(0)}⟩ and the spectral function
    A(ω) = −(1/π) Im G^R(ω + iη) for a quadratic Hamiltonian.

**Entanglement dynamics** (:class:`EntanglementDynamics`)
    Tracks bipartite von Neumann entropy S_A(t) = −Tr[ρ_A ln ρ_A] and
    mutual information I(A:B) during exact real-time evolution of an
    n-qubit state.

**Kibble-Zurek scaling** (:class:`KibbleZurekScaling`)
    Measures defect density after quenching through the critical point of
    the 1D transverse-field Ising model at different ramp rates and fits
    the power-law exponent predicted by the Kibble-Zurek mechanism.

Classes
-------
NonEquilError
KeldyshResult
KeldyshSolver
EntanglementDynamicsResult
EntanglementDynamics
KibbleZurekResult
KibbleZurekScaling

References
----------
.. [1] Keldysh, L. V. (1965). Diagram technique for nonequilibrium processes.
       *Sov. Phys. JETP* 20, 1018. (Translation of Zh. Eksp. Teor. Fiz. 47, 1515)

.. [2] Kibble, T. W. B. (1976). Topology of cosmic domains and strings.
       *J. Phys. A* 9, 1387.
       https://doi.org/10.1088/0305-4470/9/8/029

.. [3] Zurek, W. H. (1985). Cosmological experiments in superfluid helium?
       *Nature* 317, 505.
       https://doi.org/10.1038/317505a0

Physical validation
-------------------
Kibble-Zurek mechanism for 1D TFIM: defect density xi ~ tau_Q^{-nu/(1+nu*z)}
For 1D TFIM: nu=1, z=1 => xi ~ tau_Q^{-0.5}. Power law exponent = -0.5.
Verified by linear regression on log(xi) vs log(tau_Q) in test suite.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "NonEquilError",
    "KeldyshResult",
    "KeldyshSolver",
    "EntanglementDynamicsResult",
    "EntanglementDynamics",
    "KibbleZurekResult",
    "KibbleZurekScaling",
]


class NonEquilError(ValueError):
    """Raised for invalid non-equilibrium dynamics requests."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _expm_H(H: np.ndarray, t: float) -> np.ndarray:
    """Matrix exponential exp(-i H t) via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(H)
    phases = np.exp(-1j * eigvals * t)
    return (eigvecs * phases) @ eigvecs.conj().T


def _von_neumann_entropy(rho: np.ndarray) -> float:
    """Von Neumann entropy S = -Tr[ρ ln ρ]."""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-15]
    return float(-np.sum(eigvals * np.log(eigvals)))


def _reduced_density_matrix(psi: np.ndarray, n_qubits: int,
                             subsystem: list[int]) -> np.ndarray:
    """Compute ρ_A = Tr_B[|ψ⟩⟨ψ|] by partial trace."""
    n_a = len(subsystem)
    n_b = n_qubits - n_a
    # Reorder: subsystem A first, B second
    all_qubits = list(range(n_qubits))
    b_qubits = [q for q in all_qubits if q not in subsystem]
    perm = subsystem + b_qubits
    state = psi.reshape([2] * n_qubits)
    state = np.transpose(state, perm)
    state = state.reshape(2**n_a, 2**n_b)
    rho_a = state @ state.conj().T
    return rho_a


# ── KeldyshSolver ─────────────────────────────────────────────────────────────

@dataclass
class KeldyshResult:
    """Result of a :class:`KeldyshSolver` computation.

    Attributes
    ----------
    times : np.ndarray
        Real-time grid used for the Green's function.
    G_ret : np.ndarray, shape (n_times,)
        Retarded Green's function G^R(t) in the time domain.
    spectral_function : np.ndarray, shape (n_freqs,)
        Spectral function A(ω) = −(1/π) Im G^R(ω + iη).
    frequencies : np.ndarray, shape (n_freqs,)
        Frequency axis corresponding to spectral_function.
    site : int
        Site index for which G^R was computed.
    """
    times: np.ndarray
    G_ret: np.ndarray
    spectral_function: np.ndarray
    frequencies: np.ndarray
    site: int

    def to_dict(self) -> dict:
        return {
            "times": self.times.tolist(),
            "G_ret_real": self.G_ret.real.tolist(),
            "G_ret_imag": self.G_ret.imag.tolist(),
            "spectral_function": self.spectral_function.tolist(),
            "frequencies": self.frequencies.tolist(),
            "site": self.site,
        }


class KeldyshSolver:
    """Keldysh non-equilibrium Green's function for a quadratic Hamiltonian.

    Computes the single-particle retarded Green's function on the real-time
    axis using exact eigendecomposition, then Fourier-transforms to obtain
    the spectral function.

    Parameters
    ----------
    H_matrix : np.ndarray, shape (n_sites, n_sites)
        Single-particle Hamiltonian matrix (Hermitian).
    site : int, default 0
        Site index for the Green's function G^R_{site,site}(t).
    broadening : float, default 0.05
        Lorentzian broadening η added to the frequency-domain denominator.
    """

    def __init__(
        self,
        H_matrix: np.ndarray,
        site: int = 0,
        broadening: float = 0.05,
    ) -> None:
        H_matrix = np.asarray(H_matrix, dtype=complex)
        if H_matrix.ndim != 2 or H_matrix.shape[0] != H_matrix.shape[1]:
            raise NonEquilError("H_matrix must be a square 2-D array")
        if not (0 <= site < H_matrix.shape[0]):
            raise NonEquilError(
                f"site={site} out of range [0, {H_matrix.shape[0]})"
            )
        self._H = H_matrix
        self.site = site
        self.broadening = broadening

    def solve(
        self,
        t_max: float = 10.0,
        n_times: int = 200,
        n_freqs: int = 400,
    ) -> KeldyshResult:
        """Compute G^R(t) and the spectral function A(ω).

        Parameters
        ----------
        t_max : float
            Maximum time for the Green's function.
        n_times : int
            Number of time points.
        n_freqs : int
            Number of frequency points for A(ω).

        Returns
        -------
        KeldyshResult
        """
        times = np.linspace(0, t_max, n_times)
        eigvals, eigvecs = np.linalg.eigh(self._H)
        n = self._H.shape[0]
        s = self.site

        # G^R_{s,s}(t) = -iθ(t) Σ_k |⟨s|k⟩|² exp(-iε_k t)
        weights = np.abs(eigvecs[s, :])**2  # |⟨s|k⟩|²
        G_ret = np.zeros(n_times, dtype=complex)
        for i, t in enumerate(times):
            if t > 0:
                G_ret[i] = -1j * np.sum(weights * np.exp(-1j * eigvals * t))

        # Spectral function: A(ω) = Σ_k |⟨s|k⟩|² δ(ω - ε_k) (Lorentzian broadened)
        e_min = eigvals.min() - 3 * self.broadening
        e_max = eigvals.max() + 3 * self.broadening
        freqs = np.linspace(e_min, e_max, n_freqs)
        eta = self.broadening
        spectral = np.zeros(n_freqs)
        for k, (ek, wk) in enumerate(zip(eigvals, weights)):
            spectral += wk * (eta / math.pi) / ((freqs - ek)**2 + eta**2)

        return KeldyshResult(
            times=times,
            G_ret=G_ret,
            spectral_function=spectral,
            frequencies=freqs,
            site=s,
        )


# ── EntanglementDynamics ──────────────────────────────────────────────────────

@dataclass
class EntanglementDynamicsResult:
    """Result of :class:`EntanglementDynamics`.

    Attributes
    ----------
    times : list[float]
    entropy : list[float]
        Bipartite von Neumann entropy S_A(t) at each time step.
    mutual_info : list[float]
        Mutual information I(A:B)(t).
    subsystem_a : list[int]
        Qubit indices in subsystem A.
    """
    times: list[float]
    entropy: list[float]
    mutual_info: list[float]
    subsystem_a: list[int]

    def to_dict(self) -> dict:
        return {
            "times": self.times,
            "entropy": self.entropy,
            "mutual_info": self.mutual_info,
            "subsystem_a": self.subsystem_a,
        }


class EntanglementDynamics:
    """Track bipartite entanglement entropy during real-time evolution.

    Uses exact matrix-exponential evolution of an n-qubit state.

    Parameters
    ----------
    H : np.ndarray, shape (2**n, 2**n)
        Full many-body Hamiltonian matrix.
    n_qubits : int
        Number of qubits.
    subsystem_a : list[int] | None
        Qubit indices for subsystem A.  Defaults to the left half.
    """

    def __init__(
        self,
        H: np.ndarray,
        n_qubits: int,
        subsystem_a: Optional[list[int]] = None,
    ) -> None:
        H = np.asarray(H, dtype=complex)
        expected = 2**n_qubits
        if H.shape != (expected, expected):
            raise NonEquilError(
                f"H must have shape ({expected},{expected}) for {n_qubits} qubits, "
                f"got {H.shape}"
            )
        self._H = H
        self.n_qubits = n_qubits
        if subsystem_a is None:
            subsystem_a = list(range(n_qubits // 2))
        self.subsystem_a = subsystem_a

    def evolve(
        self,
        initial_state: np.ndarray,
        times: list[float],
    ) -> EntanglementDynamicsResult:
        """Evolve an initial state and track entanglement entropy.

        Parameters
        ----------
        initial_state : np.ndarray, shape (2**n,)
            Normalised initial state vector.
        times : list[float]
            Time points at which to compute entropy.

        Returns
        -------
        EntanglementDynamicsResult
        """
        psi0 = np.asarray(initial_state, dtype=complex)
        psi0 /= np.linalg.norm(psi0)
        n = self.n_qubits
        sub = self.subsystem_a
        b_sub = [q for q in range(n) if q not in sub]

        # Pre-compute eigenbasis for fast exponentiation
        eigvals, eigvecs = np.linalg.eigh(self._H)

        entropies: list[float] = []
        mutual_infos: list[float] = []

        for t in times:
            phases = np.exp(-1j * eigvals * t)
            psi_t = eigvecs @ (phases * (eigvecs.conj().T @ psi0))
            psi_t /= np.linalg.norm(psi_t)

            rho_a = _reduced_density_matrix(psi_t, n, sub)
            s_a = _von_neumann_entropy(rho_a)

            # Mutual information: I(A:B) = S_A + S_B - S_AB
            # For a pure state |ψ⟩, S_AB = 0 and S_A = S_B
            mutual_info = 2 * s_a  # I = S_A + S_B = 2 S_A (pure state)

            entropies.append(s_a)
            mutual_infos.append(mutual_info)

        return EntanglementDynamicsResult(
            times=list(times),
            entropy=entropies,
            mutual_info=mutual_infos,
            subsystem_a=self.subsystem_a,
        )


# ── KibbleZurekScaling ────────────────────────────────────────────────────────

@dataclass
class KibbleZurekResult:
    """Result of :class:`KibbleZurekScaling`.

    Attributes
    ----------
    quench_rates : list[float]
        Ramp rates v = dh/dt used in the quench.
    defect_densities : list[float]
        Average defect (domain-wall) density after each quench.
    kz_exponent : float
        Fitted Kibble-Zurek exponent ν/(1+zν) from log-log slope.
    kz_prefactor : float
        Fitted prefactor in n_def ∝ v^exponent.
    """
    quench_rates: list[float]
    defect_densities: list[float]
    kz_exponent: float
    kz_prefactor: float

    @property
    def exponent(self) -> float:
        """Alias for kz_exponent."""
        return self.kz_exponent

    def to_dict(self) -> dict:
        return {
            "quench_rates": self.quench_rates,
            "defect_densities": self.defect_densities,
            "kz_exponent": self.kz_exponent,
            "kz_prefactor": self.kz_prefactor,
        }


class KibbleZurekScaling:
    """Kibble-Zurek scaling across the 1D transverse-field Ising critical point.

    Linearly ramps the transverse field h from h_i > h_c to h_f < h_c
    (quenching through the critical point h_c = 1) at rate v = (h_i − h_f)/T,
    then measures the density of domain walls (defects) created.

    Parameters
    ----------
    n_sites : int
        Number of sites (qubits) in the 1D chain.
    J : float, default 1.0
        Ising coupling strength.
    n_steps : int, default 50
        Number of discrete Trotter steps for the quench.
    """

    def __init__(
        self,
        n_sites: int,
        J: float = 1.0,
        n_steps: int = 50,
        *,
        model: "Optional[str]" = None,
    ) -> None:
        if n_sites < 2:
            raise NonEquilError(f"n_sites must be ≥ 2, got {n_sites}")
        self.n_sites = n_sites
        self.J = J
        self.n_steps = n_steps
        # ``model`` kwarg accepted for API compatibility ("tfim_1d" is default)
        if model is not None and model not in ("tfim_1d",):
            raise NonEquilError(
                f"Unknown model {model!r}; supported models: 'tfim_1d'"
            )

    def _build_ising_H(self, h: float) -> np.ndarray:
        """Build 1D TFIM Hamiltonian at field h."""
        n = self.n_sites
        dim = 1 << n
        H = np.zeros((dim, dim), dtype=float)

        Z = np.array([[1., 0.], [0., -1.]])
        X = np.array([[0., 1.], [1., 0.]])
        I = np.eye(2)

        def kron_op(op: np.ndarray, site: int) -> np.ndarray:
            ops = [I] * n
            ops[site] = op
            result = ops[0]
            for o in ops[1:]:
                result = np.kron(result, o)
            return result

        # ZZ couplings
        for i in range(n - 1):
            H -= self.J * kron_op(Z, i) @ kron_op(Z, i + 1).T
        # Transverse field
        for i in range(n):
            H -= h * kron_op(X, i)

        return H

    def _defect_density(self, psi: np.ndarray) -> float:
        """Measure domain-wall density ⟨(1 - Z_i Z_{i+1})/2⟩ per bond."""
        n = self.n_sites
        Z = np.array([[1., 0.], [0., -1.]])
        I = np.eye(2)

        def zz(i: int) -> np.ndarray:
            ops = [I] * n
            ops[i] = Z
            ops[i + 1] = Z
            result = ops[0]
            for o in ops[1:]:
                result = np.kron(result, o)
            return result

        total = 0.0
        for i in range(n - 1):
            zz_i = zz(i)
            zz_exp = float(np.real(psi.conj() @ zz_i @ psi))
            total += (1.0 - zz_exp) / 2.0
        return total / (n - 1)

    def _run_quench(self, v: float, h_i: float = 2.0, h_f: float = 0.1) -> float:
        """Quench from h_i to h_f at rate v; return defect density."""
        T = (h_i - h_f) / v
        dt = T / self.n_steps

        # Initial ground state at h_i (paramagnetic)
        H0 = self._build_ising_H(h_i)
        eigvals, eigvecs = np.linalg.eigh(H0)
        psi = eigvecs[:, 0].astype(complex)

        for step in range(self.n_steps):
            h_now = h_i - v * (step + 0.5) * dt
            H_now = self._build_ising_H(h_now)
            U = _expm_H(H_now, dt)
            psi = U @ psi
            psi /= np.linalg.norm(psi)

        return self._defect_density(psi)

    def run(
        self,
        tau_Q_values: "Optional[list[float]]" = None,
        seed: "Optional[int]" = None,
        h_i: float = 2.0,
        h_f: float = 0.1,
    ) -> KibbleZurekResult:
        """Alias for :meth:`sweep` with ``tau_Q_values`` → ``quench_rates``.

        Parameters
        ----------
        tau_Q_values : list[float] | None
            Quench time scales (τ_Q).  The quench rate is v = Δh / τ_Q.
            If None, defaults to a log-spaced range.
        seed : int | None
            Ignored (kept for API compatibility).
        """
        quench_rates: "Optional[list[float]]" = None
        if tau_Q_values is not None:
            dh = h_i - h_f
            quench_rates = [dh / tq for tq in tau_Q_values]
        return self.sweep(quench_rates=quench_rates, h_i=h_i, h_f=h_f)

    def sweep(
        self,
        quench_rates: Optional[list[float]] = None,
        h_i: float = 2.0,
        h_f: float = 0.1,
    ) -> KibbleZurekResult:
        """Run the KZ scaling experiment for multiple quench rates.

        Parameters
        ----------
        quench_rates : list[float] | None
            Ramp rates v = Δh/T.  Defaults to a log-spaced range.
        h_i, h_f : float
            Initial and final transverse field values.

        Returns
        -------
        KibbleZurekResult
        """
        if quench_rates is None:
            quench_rates = list(np.logspace(-1, 1, 8))

        defects = [self._run_quench(v, h_i, h_f) for v in quench_rates]

        # Fit KZ power law: log(n_def) = exponent * log(v) + log(prefactor)
        log_v = np.log(quench_rates)
        log_n = np.log(np.maximum(defects, 1e-12))
        exponent, log_pref = float("nan"), float("nan")
        if len(quench_rates) >= 2:
            coeffs = np.polyfit(log_v, log_n, 1)
            exponent = float(coeffs[0])
            log_pref = float(coeffs[1])

        return KibbleZurekResult(
            quench_rates=list(quench_rates),
            defect_densities=defects,
            kz_exponent=exponent,
            kz_prefactor=float(np.exp(log_pref)) if not math.isnan(log_pref) else float("nan"),
        )
