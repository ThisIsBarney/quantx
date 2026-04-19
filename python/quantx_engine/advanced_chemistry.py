"""Advanced quantum chemistry algorithms (M79).

Implements UCCSD, ADAPT-VQE, QAOA, and QITE as pure-Python/NumPy exact
statevector solvers.  All methods operate on the full 2ⁿ-dimensional Hilbert
space (feasible for n ≤ 16) and build on the :class:`PauliHamiltonian`
representation from :mod:`quantx_engine.chemistry`.

Quick reference::

    from quantx_engine.chemistry import MolecularVQE
    from quantx_engine.advanced_chemistry import UCCSDSolver, QITESolver

    problem = MolecularVQE.h2()
    H_mat = build_hamiltonian_matrix(problem.hamiltonian)

    # UCCSD
    solver = UCCSDSolver(n_electrons=2, n_qubits=4)
    result = solver.run(H_mat)
    print(result.energy)

    # QITE
    qite = QITESolver(n_qubits=4)
    res  = qite.run(H_mat, delta_tau=0.05, n_steps=80)
    print(res.final_energy)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    pass

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

from .chemistry import PauliHamiltonian, ChemistryError

__all__ = [
    "AdvChemError",
    "build_hamiltonian_matrix",
    "hf_state",
    "UCCSDResult",
    "UCCSDSolver",
    "ADAPTVQEResult",
    "ADAPTVQESolver",
    "QAOAResult",
    "QAOASolver",
    "QITEResult",
    "QITESolver",
]


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class AdvChemError(ChemistryError):
    """Raised for invalid inputs to the advanced chemistry solvers."""


# ---------------------------------------------------------------------------
# Pauli matrix primitives
# ---------------------------------------------------------------------------

def _require_numpy() -> None:
    if not _HAS_NUMPY:  # pragma: no cover
        raise AdvChemError("NumPy is required for advanced chemistry solvers.")


def _kron_list(ops: "list") -> "np.ndarray":
    """Left-to-right Kronecker product of a list of 2×2 matrices."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def _pauli_term_matrix(ops_dict: "dict[int, str]", n_qubits: int) -> "np.ndarray":
    """Build the 2ⁿ × 2ⁿ matrix for a single Pauli-string term.

    Qubit 0 is the most significant (leftmost) tensor factor.
    """
    _I = np.eye(2, dtype=complex)
    _X = np.array([[0, 1], [1, 0]], dtype=complex)
    _Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    _Z = np.array([[1, 0], [0, -1]], dtype=complex)
    _MAP = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}

    factors = [_MAP[ops_dict.get(q, "I")] for q in range(n_qubits)]
    return _kron_list(factors)


def build_hamiltonian_matrix(ham: PauliHamiltonian) -> "np.ndarray":
    """Build the full 2ⁿ × 2ⁿ Hermitian matrix of a :class:`PauliHamiltonian`.

    Parameters
    ----------
    ham:
        A :class:`~quantx_engine.chemistry.PauliHamiltonian` instance.

    Returns
    -------
    numpy.ndarray of shape ``(2**n, 2**n)``, dtype ``complex128``.
    """
    _require_numpy()
    n = ham.n_qubits
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=complex)
    for term in ham.terms:
        H += term.coeff * _pauli_term_matrix(term.ops, n)
    return H


# ---------------------------------------------------------------------------
# Fermionic operators (Jordan-Wigner)
# ---------------------------------------------------------------------------

def _sigma_plus() -> "np.ndarray":
    """σ₊ = |1⟩⟨0| = [[0,1],[0,0]]."""
    return np.array([[0, 1], [0, 0]], dtype=complex)


def _sigma_minus() -> "np.ndarray":
    """σ₋ = |0⟩⟨1| = [[0,0],[1,0]]."""
    return np.array([[0, 0], [1, 0]], dtype=complex)


def _fermi_op(p: int, n_qubits: int, creation: bool) -> "np.ndarray":
    """Jordan-Wigner creation (creation=True) or annihilation operator for site p.

    a†_p = Z_0 ⊗ … ⊗ Z_{p-1} ⊗ σ₊ ⊗ I_{p+1} ⊗ … ⊗ I_{N-1}
    """
    _I = np.eye(2, dtype=complex)
    _Z = np.array([[1, 0], [0, -1]], dtype=complex)
    op = _sigma_plus() if creation else _sigma_minus()
    factors = [_Z if k < p else (op if k == p else _I) for k in range(n_qubits)]
    return _kron_list(factors)


def _excitation_antiherm(
    occ: "tuple[int, ...]",
    virt: "tuple[int, ...]",
    n_qubits: int,
) -> "np.ndarray":
    """Return the anti-Hermitian operator K = (T - T†) for an excitation.

    For singles (|occ|=1, |virt|=1):
        K = a†_v a_o − a†_o a_v

    For doubles (|occ|=2, |virt|=2):
        K = a†_v1 a†_v2 a_o2 a_o1 − h.c.
        (preserves particle number; occ and virt ordered ascending)

    Returns
    -------
    Anti-Hermitian matrix K = −K†.
    """
    if len(occ) == 1 and len(virt) == 1:
        o, v = occ[0], virt[0]
        adag_v = _fermi_op(v, n_qubits, creation=True)
        a_o    = _fermi_op(o, n_qubits, creation=False)
        T = adag_v @ a_o
        return T - T.conj().T
    elif len(occ) == 2 and len(virt) == 2:
        o1, o2 = sorted(occ)
        v1, v2 = sorted(virt)
        adag_v1 = _fermi_op(v1, n_qubits, creation=True)
        adag_v2 = _fermi_op(v2, n_qubits, creation=True)
        a_o2    = _fermi_op(o2, n_qubits, creation=False)
        a_o1    = _fermi_op(o1, n_qubits, creation=False)
        T = adag_v1 @ adag_v2 @ a_o2 @ a_o1
        return T - T.conj().T
    else:
        raise AdvChemError(
            f"Only single and double excitations supported, "
            f"got occ={occ}, virt={virt}"
        )


# ---------------------------------------------------------------------------
# State utilities
# ---------------------------------------------------------------------------

def hf_state(n_electrons: int, n_qubits: int) -> "np.ndarray":
    """Return the Hartree-Fock reference state vector.

    The HF state occupies the *n_electrons* lowest-indexed spin-orbitals.
    Qubit 0 is the most significant (leftmost) tensor factor, so the
    occupied qubits 0..n_electrons-1 give index ``2ⁿ − 2^(n-n_e) = ... =
    integer with the top n_e bits set``.

    Parameters
    ----------
    n_electrons:
        Number of electrons (occupied spin-orbitals).
    n_qubits:
        Total number of spin-orbitals / qubits.

    Returns
    -------
    numpy.ndarray of shape ``(2**n_qubits,)``, dtype ``complex128``.
    """
    _require_numpy()
    dim = 1 << n_qubits
    psi = np.zeros(dim, dtype=complex)
    # Qubits 0..n_e-1 are |1⟩, rest |0⟩.  In MSB-first ordering:
    # index = Σ_{k=0}^{n_e-1} 2^(n-1-k)
    idx = sum(1 << (n_qubits - 1 - k) for k in range(n_electrons))
    psi[idx] = 1.0
    return psi


def _expm_antiherm(A: "np.ndarray") -> "np.ndarray":
    """Compute expm(A) for an anti-Hermitian matrix A = iH (H Hermitian).

    Uses eigendecomposition:  A = iU Λ U†  ⟹  expm(A) = U diag(exp(iλ)) U†.
    """
    # A = iH ⟹ H = -iA; H must be Hermitian
    H = -1j * A  # Hermitian
    # Symmetrise to guard against numerical drift
    H = 0.5 * (H + H.conj().T)
    vals, vecs = np.linalg.eigh(H)
    phase = np.exp(1j * vals)
    return (vecs * phase) @ vecs.conj().T


def _energy(psi: "np.ndarray", H: "np.ndarray") -> float:
    """⟨ψ|H|ψ⟩ (real part)."""
    return float(np.real(psi.conj() @ H @ psi))


def _normalize(psi: "np.ndarray") -> "np.ndarray":
    return psi / np.linalg.norm(psi)


# ---------------------------------------------------------------------------
# Simple numerical optimiser (gradient-free, Nelder-Mead / coordinate descent)
# ---------------------------------------------------------------------------

def _minimize_energy(
    energy_fn: "callable",
    params0: "np.ndarray",
    n_iter: int,
    step: float = 0.1,
) -> "tuple[float, np.ndarray]":
    """Coordinate descent optimizer for the energy function.

    Parameters
    ----------
    energy_fn:
        energy_fn(params) -> float.
    params0:
        Initial parameter vector.
    n_iter:
        Number of full coordinate sweeps.
    step:
        Initial step size for line search.

    Returns
    -------
    (best_energy, best_params)
    """
    p = params0.copy()
    E = energy_fn(p)
    step_sizes = np.full(len(p), step)

    for _sweep in range(n_iter):
        improved = False
        for k in range(len(p)):
            # Try step up
            p_up = p.copy(); p_up[k] += step_sizes[k]
            E_up = energy_fn(p_up)
            # Try step down
            p_dn = p.copy(); p_dn[k] -= step_sizes[k]
            E_dn = energy_fn(p_dn)
            best = min((E, p), (E_up, p_up), (E_dn, p_dn), key=lambda t: t[0])
            if best[0] < E - 1e-12:
                E, p = best
                step_sizes[k] = min(step_sizes[k] * 1.1, 1.0)
                improved = True
            else:
                step_sizes[k] *= 0.5
        if not improved and np.max(step_sizes) < 1e-8:
            break

    return E, p


# ===========================================================================
# UCCSD
# ===========================================================================

@dataclass
class UCCSDResult:
    """Result of a UCCSD optimization.

    Attributes
    ----------
    energy:
        Final variational energy ⟨ψ|H|ψ⟩.
    params:
        Optimized UCCSD amplitudes (singles then doubles).
    n_params:
        Total number of variational parameters.
    n_singles:
        Number of single-excitation parameters.
    n_doubles:
        Number of double-excitation parameters.
    converged:
        True if the optimizer reached a stationary point.
    """
    energy: float
    params: "list[float]"
    n_params: int
    n_singles: int
    n_doubles: int
    converged: bool = False


class UCCSDSolver:
    """Unitary Coupled Cluster Singles and Doubles solver.

    Builds the UCCSD ansatz by:

    1. Generating all single-excitation generators
       ``K_single = a†_v a_o − h.c.``
       and double-excitation generators
       ``K_double = a†_v₁ a†_v₂ a_o₂ a_o₁ − h.c.``
       in the Jordan-Wigner representation.

    2. Forming the combined anti-Hermitian operator
       ``A(θ) = Σ θ_s K_s + Σ θ_d K_d``.

    3. Preparing the UCCSD state
       ``|ψ(θ)⟩ = expm(A(θ)) |HF⟩``
       via exact matrix exponentiation.

    4. Minimizing ``⟨ψ(θ)|H|ψ(θ)⟩`` over ``θ`` using coordinate descent.

    Parameters
    ----------
    n_electrons:
        Number of electrons (occupied spin-orbitals in HF reference).
    n_qubits:
        Total number of spin-orbitals / qubits.

    Examples
    --------
    >>> from quantx_engine.chemistry import MolecularVQE
    >>> from quantx_engine.advanced_chemistry import UCCSDSolver, build_hamiltonian_matrix
    >>> problem = MolecularVQE.h2()
    >>> H = build_hamiltonian_matrix(problem.hamiltonian)
    >>> result = UCCSDSolver(n_electrons=2, n_qubits=4).run(H)
    >>> abs(result.energy - problem.fci_energy) < 0.01
    True
    """

    def __init__(self, n_electrons: int, n_qubits: int) -> None:
        _require_numpy()
        if n_electrons < 1 or n_electrons >= n_qubits:
            raise AdvChemError(
                f"n_electrons must be in [1, n_qubits-1], "
                f"got {n_electrons} with n_qubits={n_qubits}"
            )
        self.n_electrons = n_electrons
        self.n_qubits = n_qubits
        self._occ = list(range(n_electrons))
        self._virt = list(range(n_electrons, n_qubits))
        self._generators = self._build_generators()

    # ------------------------------------------------------------------

    def _build_generators(self) -> "list[np.ndarray]":
        """Build all single and double excitation anti-Hermitian generators."""
        gens = []
        occ, virt = self._occ, self._virt
        # Singles
        for o in occ:
            for v in virt:
                gens.append(_excitation_antiherm((o,), (v,), self.n_qubits))
        # Doubles
        for i, o1 in enumerate(occ):
            for o2 in occ[i + 1:]:
                for j, v1 in enumerate(virt):
                    for v2 in virt[j + 1:]:
                        gens.append(
                            _excitation_antiherm((o1, o2), (v1, v2), self.n_qubits)
                        )
        return gens

    @property
    def n_singles(self) -> int:
        return len(self._occ) * len(self._virt)

    @property
    def n_doubles(self) -> int:
        no, nv = len(self._occ), len(self._virt)
        return no * (no - 1) // 2 * nv * (nv - 1) // 2

    @property
    def n_params(self) -> int:
        return len(self._generators)

    def _uccsd_state(self, params: "np.ndarray") -> "np.ndarray":
        """Evaluate |ψ_UCCSD(θ)⟩."""
        A = sum(t * K for t, K in zip(params, self._generators))
        U = _expm_antiherm(A)
        psi0 = hf_state(self.n_electrons, self.n_qubits)
        return _normalize(U @ psi0)

    def run(
        self,
        H_matrix: "np.ndarray",
        n_iter: int = 80,
        step: float = 0.2,
    ) -> UCCSDResult:
        """Optimize the UCCSD energy.

        Parameters
        ----------
        H_matrix:
            Full Hamiltonian matrix from :func:`build_hamiltonian_matrix`.
        n_iter:
            Number of coordinate descent sweeps.
        step:
            Initial step size for the optimizer.

        Returns
        -------
        UCCSDResult
        """
        params = np.zeros(self.n_params)

        def energy_fn(p: "np.ndarray") -> float:
            return _energy(self._uccsd_state(p), H_matrix)

        E_final, p_final = _minimize_energy(energy_fn, params, n_iter, step)

        # Check convergence: gradient norm < tolerance
        eps = 1e-5
        grad = np.zeros_like(p_final)
        for k in range(len(p_final)):
            p_p = p_final.copy(); p_p[k] += eps
            p_m = p_final.copy(); p_m[k] -= eps
            grad[k] = (energy_fn(p_p) - energy_fn(p_m)) / (2 * eps)
        converged = float(np.linalg.norm(grad)) < 1e-4

        return UCCSDResult(
            energy=E_final,
            params=p_final.tolist(),
            n_params=self.n_params,
            n_singles=self.n_singles,
            n_doubles=self.n_doubles,
            converged=converged,
        )


# ===========================================================================
# ADAPT-VQE
# ===========================================================================

@dataclass
class ADAPTVQEResult:
    """Result of an ADAPT-VQE run.

    Attributes
    ----------
    energy:
        Final variational energy.
    n_operators:
        Number of operators added to the ansatz.
    gradient_norm:
        L2 norm of the energy gradient over the operator pool at convergence.
    operator_indices:
        Indices of operators chosen from the pool, in order.
    params:
        Optimized parameters for each chosen operator.
    """
    energy: float
    n_operators: int
    gradient_norm: float
    operator_indices: "list[int]"
    params: "list[float]"


class ADAPTVQESolver:
    """Adaptive derivative-assembled pseudo-Trotter VQE (ADAPT-VQE).

    At each iteration, finds the operator from the pool with the largest
    energy gradient and appends it to the ansatz with a new parameter.
    Stops when the maximum absolute gradient falls below *threshold* or
    *max_operators* is reached.

    Parameters
    ----------
    n_electrons:
        Number of electrons in the HF reference.
    n_qubits:
        Total number of qubits.
    pool:
        List of anti-Hermitian generator matrices.  If ``None``, uses
        the UCCSD pool (all singles + doubles).
    """

    def __init__(
        self,
        n_electrons: int,
        n_qubits: int,
        pool: "list[np.ndarray] | None" = None,
    ) -> None:
        _require_numpy()
        self.n_electrons = n_electrons
        self.n_qubits = n_qubits
        if pool is None:
            # Reuse UCCSDSolver to get the standard singles+doubles pool
            solver = UCCSDSolver(n_electrons, n_qubits)
            self._pool = solver._generators
        else:
            self._pool = pool

    def run(
        self,
        H_matrix: "np.ndarray",
        threshold: float = 1e-4,
        max_operators: int = 20,
        n_iter_per_step: int = 40,
        step: float = 0.2,
    ) -> ADAPTVQEResult:
        """Run ADAPT-VQE.

        Parameters
        ----------
        H_matrix:
            Full Hamiltonian matrix.
        threshold:
            Stop when max |∂E/∂θ_k| < threshold.
        max_operators:
            Maximum number of operators to add.
        n_iter_per_step:
            Optimizer iterations after each operator is added.
        step:
            Initial step size for the optimizer.

        Returns
        -------
        ADAPTVQEResult
        """
        psi = hf_state(self.n_electrons, self.n_qubits)
        chosen_ops: "list[int]" = []
        params: "list[float]" = []
        grad_norm = float("inf")

        for _step in range(max_operators):
            # Compute gradient for each pool operator
            grads = np.zeros(len(self._pool))
            for k, K in enumerate(self._pool):
                # ∂E/∂θ_k = ⟨ψ|[H, K]|ψ⟩ (exact gradient at θ_k=0)
                comm = H_matrix @ K - K @ H_matrix
                grads[k] = float(np.real(psi.conj() @ comm @ psi))

            grad_norm = float(np.max(np.abs(grads)))
            if grad_norm < threshold:
                break

            # Pick the operator with the largest gradient magnitude
            best_k = int(np.argmax(np.abs(grads)))
            chosen_ops.append(best_k)
            params.append(0.0)

            # Optimise all current parameters
            def energy_fn(p: "np.ndarray") -> float:
                state = hf_state(self.n_electrons, self.n_qubits)
                for param_i, op_idx in zip(p, chosen_ops):
                    K = self._pool[op_idx]
                    U = _expm_antiherm(param_i * K)
                    state = _normalize(U @ state)
                return _energy(state, H_matrix)

            E, p_opt = _minimize_energy(
                energy_fn, np.array(params), n_iter_per_step, step
            )
            params = p_opt.tolist()

            # Update state with optimised parameters
            psi = hf_state(self.n_electrons, self.n_qubits)
            for param_i, op_idx in zip(params, chosen_ops):
                K = self._pool[op_idx]
                psi = _normalize(_expm_antiherm(param_i * K) @ psi)

        final_energy = _energy(psi, H_matrix)
        return ADAPTVQEResult(
            energy=final_energy,
            n_operators=len(chosen_ops),
            gradient_norm=grad_norm,
            operator_indices=chosen_ops,
            params=params,
        )


# ===========================================================================
# QAOA
# ===========================================================================

@dataclass
class QAOAResult:
    """Result of a QAOA optimization.

    Attributes
    ----------
    energy:
        Best variational energy ⟨ψ(γ,β)|H_C|ψ(γ,β)⟩.
    gamma:
        Optimized cost Hamiltonian rotation angles.
    beta:
        Optimized mixer Hamiltonian rotation angles.
    p:
        Number of QAOA layers used.
    """
    energy: float
    gamma: "list[float]"
    beta: "list[float]"
    p: int


class QAOASolver:
    """Quantum Approximate Optimization Algorithm.

    Prepares the p-layer QAOA state::

        |ψ(γ,β)⟩ = ∏_{l=1}^{p} exp(-iβ_l H_B) exp(-iγ_l H_C) |+⟩^⊗n

    where H_B = Σ_i X_i is the mixer Hamiltonian and H_C is the cost
    Hamiltonian.  Optimizes ⟨ψ(γ,β)|H_C|ψ(γ,β)⟩ over the 2p parameters.

    Parameters
    ----------
    n_qubits:
        Number of qubits.
    """

    def __init__(self, n_qubits: int) -> None:
        _require_numpy()
        self.n_qubits = n_qubits
        self._H_mixer = self._build_mixer()

    def _build_mixer(self) -> "np.ndarray":
        """Build the standard transverse-field mixer H_B = Σ_i X_i."""
        _I = np.eye(2, dtype=complex)
        _X = np.array([[0, 1], [1, 0]], dtype=complex)
        n = self.n_qubits
        H_B = np.zeros((1 << n, 1 << n), dtype=complex)
        for i in range(n):
            factors = [(_X if k == i else _I) for k in range(n)]
            H_B += _kron_list(factors)
        return H_B

    def _initial_state(self) -> "np.ndarray":
        """Equal superposition |+⟩^⊗n = H^⊗n |0⟩^⊗n."""
        n = self.n_qubits
        psi = np.ones(1 << n, dtype=complex) / math.sqrt(1 << n)
        return psi

    def _qaoa_state(
        self, gamma: "np.ndarray", beta: "np.ndarray", H_cost: "np.ndarray"
    ) -> "np.ndarray":
        psi = self._initial_state()
        for g, b in zip(gamma, beta):
            # exp(-iγ H_C): H_C diagonal in computational basis ≡ eigendecomp
            vals_c, vecs_c = np.linalg.eigh(H_cost)
            psi = (vecs_c * np.exp(-1j * g * vals_c)) @ (vecs_c.conj().T @ psi)
            # exp(-iβ H_B)
            vals_b, vecs_b = np.linalg.eigh(self._H_mixer)
            psi = (vecs_b * np.exp(-1j * b * vals_b)) @ (vecs_b.conj().T @ psi)
        return psi

    def run(
        self,
        cost_ham: PauliHamiltonian,
        p: int = 1,
        n_iter: int = 60,
        step: float = 0.3,
        seed: "int | None" = None,
    ) -> QAOAResult:
        """Optimize QAOA parameters.

        Parameters
        ----------
        cost_ham:
            The cost Hamiltonian to minimize.
        p:
            Number of QAOA layers.
        n_iter:
            Coordinate descent iterations.
        step:
            Initial step size.
        seed:
            Random seed for initial parameter perturbation.

        Returns
        -------
        QAOAResult
        """
        _require_numpy()
        H_cost = build_hamiltonian_matrix(cost_ham)
        rng = np.random.default_rng(seed)
        params = rng.uniform(-0.3, 0.3, 2 * p)  # [γ1,..,γp, β1,..,βp]

        def energy_fn(params: "np.ndarray") -> float:
            g = params[:p]
            b = params[p:]
            psi = self._qaoa_state(g, b, H_cost)
            return _energy(psi, H_cost)

        E_final, p_final = _minimize_energy(energy_fn, params, n_iter, step)
        gamma = p_final[:p].tolist()
        beta  = p_final[p:].tolist()

        return QAOAResult(energy=E_final, gamma=gamma, beta=beta, p=p)


# ===========================================================================
# QITE
# ===========================================================================

@dataclass
class QITEResult:
    """Result of a Quantum Imaginary Time Evolution run.

    Attributes
    ----------
    energies:
        Energy at each imaginary-time step.
    final_energy:
        Energy at the last step (approximation to ground-state energy).
    tau_values:
        Imaginary-time values corresponding to each energy point.
    """
    energies: "list[float]"
    final_energy: float
    tau_values: "list[float]"

    @property
    def converged(self) -> bool:
        """True if the energy change in the last 5 steps was < 1e-6."""
        if len(self.energies) < 5:
            return False
        return abs(self.energies[-1] - self.energies[-5]) < 1e-6


class QITESolver:
    """Quantum Imaginary Time Evolution (QITE).

    Evolves an initial state under::

        |ψ(τ + Δτ)⟩ ∝ (I − ΔτH) |ψ(τ)⟩

    which converges to the ground state of H as τ → ∞.

    Parameters
    ----------
    n_qubits:
        Number of qubits.

    Examples
    --------
    >>> H = build_hamiltonian_matrix(MolecularVQE.h2().hamiltonian)
    >>> result = QITESolver(n_qubits=4).run(H, delta_tau=0.05, n_steps=100)
    >>> result.final_energy  # close to FCI energy -1.1373
    """

    def __init__(self, n_qubits: int) -> None:
        _require_numpy()
        self.n_qubits = n_qubits

    def run(
        self,
        H_matrix: "np.ndarray",
        delta_tau: float = 0.05,
        n_steps: int = 100,
        initial_state: "np.ndarray | None" = None,
        n_electrons: "int | None" = None,
        tol: float = 1e-8,
    ) -> QITEResult:
        """Run imaginary time evolution.

        Parameters
        ----------
        H_matrix:
            Full Hamiltonian matrix from :func:`build_hamiltonian_matrix`.
        delta_tau:
            Imaginary time step size Δτ.
        n_steps:
            Maximum number of steps.
        initial_state:
            Initial state vector.  If ``None``, uses the equal superposition
            (or the HF state if *n_electrons* is given).
        n_electrons:
            If set and *initial_state* is None, start from the HF state.
        tol:
            Stop early if |ΔE| < tol for 10 consecutive steps.

        Returns
        -------
        QITEResult
        """
        if initial_state is not None:
            psi = _normalize(np.array(initial_state, dtype=complex))
        elif n_electrons is not None:
            psi = hf_state(n_electrons, self.n_qubits)
        else:
            # Equal superposition
            dim = 1 << self.n_qubits
            psi = np.ones(dim, dtype=complex) / math.sqrt(dim)

        I_mat = np.eye(H_matrix.shape[0], dtype=complex)
        M = I_mat - delta_tau * H_matrix

        energies: "list[float]" = []
        tau_vals: "list[float]" = []
        stagnant = 0

        for step in range(n_steps):
            E = _energy(psi, H_matrix)
            energies.append(E)
            tau_vals.append(step * delta_tau)

            # Early stopping check
            if step > 0 and abs(energies[-1] - energies[-2]) < tol:
                stagnant += 1
                if stagnant >= 10:
                    break
            else:
                stagnant = 0

            # Update: psi ← (I − Δτ H)|ψ⟩ / norm
            psi = _normalize(M @ psi)

        return QITEResult(
            energies=energies,
            final_energy=energies[-1] if energies else float("nan"),
            tau_values=tau_vals,
        )
