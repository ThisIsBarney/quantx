"""Circuit Templates Library — M88.

Provides a collection of standard parameterized quantum circuit ansätze and
fully-fixed circuit templates, all returning circuit dicts compatible with the
:class:`~quantx_engine.primitives.Estimator` and
:class:`~quantx_engine.primitives.Sampler` primitives.

Quick reference::

    from quantx_engine.templates import hardware_efficient_ansatz, ghz_circuit

    # Parameterized HEA on 4 qubits, 2 layers
    tmpl = hardware_efficient_ansatz(4, n_layers=2)
    bound = tmpl.bind([0.1] * tmpl.n_parameters)
    # bound is a plain circuit dict ready for Estimator/Sampler

    # Fixed GHZ circuit (no parameters)
    ghz = ghz_circuit(3)
    circuit_dict = ghz.to_circuit()

References
----------
.. [1] Kandala, A. et al. (2017). Hardware-efficient variational quantum
       eigensolver for small molecules and quantum magnets.
       *Nature* 549, 242. https://doi.org/10.1038/nature23879
       (Hardware-efficient ansatz template)

.. [2] Farhi, E., Goldstone, J. & Gutmann, S. (2014). A quantum approximate
       optimization algorithm. arXiv:1411.4028.
       https://doi.org/10.48550/arXiv.1411.4028

.. [3] Romero, J. et al. (2018). Strategies for quantum computing molecular
       energies using the unitary coupled cluster ansatz.
       *Quantum Sci. Technol.* 4, 014008.
       https://doi.org/10.1088/2058-9565/aad3e4
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Union

__all__ = [
    "TemplateError",
    "CircuitTemplate",
    "hardware_efficient_ansatz",
    "iqp_circuit",
    "qaoa_circuit",
    "ghz_circuit",
    "quantum_fourier_transform",
    "uccsd_ansatz",
]


# ===========================================================================
# Exception
# ===========================================================================

class TemplateError(ValueError):
    """Raised for invalid inputs to circuit template factory functions."""


# ===========================================================================
# CircuitTemplate dataclass
# ===========================================================================

@dataclass
class CircuitTemplate:
    """A parameterized (or fixed) circuit with metadata.

    Attributes
    ----------
    n_qubits:
        Number of qubits in the circuit.
    operations:
        List of gate-op dicts (may contain string parameter names).
    parameter_names:
        Ordered list of symbolic parameter name strings.
    name:
        Human-readable name of the template.
    description:
        Optional longer description.
    """
    n_qubits: int
    operations: list
    parameter_names: list
    name: str
    description: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        """Number of free (symbolic) parameters."""
        return len(self.parameter_names)

    # ------------------------------------------------------------------
    # to_circuit
    # ------------------------------------------------------------------

    def to_circuit(self) -> dict:
        """Return the raw (possibly symbolic) circuit dict.

        Returns
        -------
        dict
            ``{"n_qubits": ..., "operations": [...]}`` with string params
            left as symbolic names.
        """
        return {"n_qubits": self.n_qubits, "operations": list(self.operations)}

    # ------------------------------------------------------------------
    # bind
    # ------------------------------------------------------------------

    def bind(self, values: Union[list, dict]) -> dict:
        """Return a concrete circuit dict with all symbolic params bound.

        Parameters
        ----------
        values:
            - If a **list**: positionally bind ``parameter_names[i]`` to
              ``values[i]``.  Length must equal :attr:`n_parameters`.
            - If a **dict**: map from parameter name to float value.  All
              symbolic names must be present.

        Returns
        -------
        dict
            ``{"n_qubits": ..., "operations": [...]}`` with every string
            param replaced by its float value.  Ready for
            :meth:`~quantx_engine.primitives.Estimator.run`.

        Raises
        ------
        TemplateError
            If list length doesn't match, or a required key is missing.
        """
        if isinstance(values, (list, tuple)):
            if len(values) != self.n_parameters:
                raise TemplateError(
                    f"Expected {self.n_parameters} parameter values, "
                    f"got {len(values)}."
                )
            pv: dict = dict(zip(self.parameter_names, [float(v) for v in values]))
        elif isinstance(values, dict):
            pv = {k: float(v) for k, v in values.items()}
            missing = [n for n in self.parameter_names if n not in pv]
            if missing:
                raise TemplateError(
                    f"Missing parameter values for: {missing}."
                )
        else:
            raise TemplateError(
                f"values must be a list or dict, got {type(values).__name__}."
            )

        bound_ops = []
        for op in self.operations:
            new_op = dict(op)
            if "params" in op:
                new_params = []
                for p in op["params"]:
                    if isinstance(p, str):
                        if p not in pv:
                            raise TemplateError(
                                f"Parameter {p!r} not found in provided values."
                            )
                        new_params.append(pv[p])
                    else:
                        new_params.append(float(p))
                new_op = dict(op)
                new_op["params"] = new_params
            bound_ops.append(new_op)

        return {"n_qubits": self.n_qubits, "operations": bound_ops}


# ===========================================================================
# Helpers
# ===========================================================================

def _check_n_qubits(n_qubits: int, min_val: int = 1) -> None:
    if not isinstance(n_qubits, int) or n_qubits < min_val:
        raise TemplateError(
            f"n_qubits must be an integer >= {min_val}, got {n_qubits!r}."
        )


def _check_positive_int(val: int, name: str) -> None:
    if not isinstance(val, int) or val < 1:
        raise TemplateError(
            f"{name} must be a positive integer >= 1, got {val!r}."
        )


# ===========================================================================
# Hardware-Efficient Ansatz (HEA)
# ===========================================================================

def hardware_efficient_ansatz(
    n_qubits: int,
    n_layers: int = 1,
    rotation: str = "ry",
) -> CircuitTemplate:
    """Hardware-efficient ansatz (HEA).

    Structure per layer:
    1. Single-qubit rotation (rx/ry/rz) on every qubit.
    2. Entangling CNOT ladder: CX(0,1), CX(1,2), …, CX(n-2, n-1).

    Repeated ``n_layers`` times, followed by a final rotation layer.

    Total parameters: ``n_qubits * (n_layers + 1)``.

    Parameters
    ----------
    n_qubits:
        Number of qubits (>= 1).
    n_layers:
        Number of rotation + entangling layers (>= 1).
    rotation:
        Single-qubit rotation gate: ``"rx"``, ``"ry"``, or ``"rz"``.

    Returns
    -------
    CircuitTemplate
    """
    _check_n_qubits(n_qubits)
    _check_positive_int(n_layers, "n_layers")
    rotation = rotation.lower()
    if rotation not in ("rx", "ry", "rz"):
        raise TemplateError(
            f"rotation must be 'rx', 'ry', or 'rz', got {rotation!r}."
        )

    ops: list[dict] = []
    param_names: list[str] = []
    param_idx = 0

    for layer in range(n_layers):
        # Rotation layer
        for q in range(n_qubits):
            pname = f"theta_{layer}_{q}"
            param_names.append(pname)
            ops.append({"name": rotation, "qubits": [q], "params": [pname]})
        # Entangling CNOT ladder
        for q in range(n_qubits - 1):
            ops.append({"name": "cx", "qubits": [q, q + 1]})

    # Final rotation layer
    for q in range(n_qubits):
        pname = f"theta_{n_layers}_{q}"
        param_names.append(pname)
        ops.append({"name": rotation, "qubits": [q], "params": [pname]})

    assert len(param_names) == n_qubits * (n_layers + 1)

    return CircuitTemplate(
        n_qubits=n_qubits,
        operations=ops,
        parameter_names=param_names,
        name="HardwareEfficientAnsatz",
        description=(
            f"{n_layers}-layer hardware-efficient ansatz with {rotation} rotations "
            f"on {n_qubits} qubits."
        ),
    )


# ===========================================================================
# IQP Circuit
# ===========================================================================

def iqp_circuit(
    n_qubits: int,
    depth: int = 2,
    seed: int | None = None,
) -> CircuitTemplate:
    """Instantaneous Quantum Polynomial (IQP) circuit.

    Structure (repeated ``depth`` times):
    1. H on every qubit.
    2. Diagonal phase layer: RZZ(theta_i_j) on random qubit pairs, then
       RZ(phi_i) on each qubit.

    Useful for quantum advantage demonstrations.

    Parameters
    ----------
    n_qubits:
        Number of qubits (>= 2).
    depth:
        Number of H + diagonal-phase repetitions (>= 1).
    seed:
        Random seed for choosing which qubit pairs get RZZ interactions.
        Makes the structure reproducible.

    Returns
    -------
    CircuitTemplate
    """
    _check_n_qubits(n_qubits, min_val=2)
    _check_positive_int(depth, "depth")

    rng = random.Random(seed)
    ops: list[dict] = []
    param_names: list[str] = []

    # Enumerate all pairs (i,j) with i < j deterministically for reproducibility
    all_pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

    for d in range(depth):
        # Hadamard layer
        for q in range(n_qubits):
            ops.append({"name": "h", "qubits": [q]})

        # Diagonal ZZ phase layer: choose a random subset of pairs
        n_pairs = rng.randint(1, max(1, len(all_pairs)))
        chosen_pairs = rng.sample(all_pairs, min(n_pairs, len(all_pairs)))
        for i, (qi, qj) in enumerate(chosen_pairs):
            pname = f"theta_{d}_{i}"
            param_names.append(pname)
            ops.append({"name": "rzz", "qubits": [qi, qj], "params": [pname]})

        # Single-qubit diagonal Z phase layer
        for q in range(n_qubits):
            pname = f"phi_{d}_{q}"
            param_names.append(pname)
            ops.append({"name": "rz", "qubits": [q], "params": [pname]})

    return CircuitTemplate(
        n_qubits=n_qubits,
        operations=ops,
        parameter_names=param_names,
        name="IQPCircuit",
        description=(
            f"Depth-{depth} IQP circuit on {n_qubits} qubits."
        ),
    )


# ===========================================================================
# QAOA Circuit
# ===========================================================================

def qaoa_circuit(n_qubits: int, p_layers: int = 1) -> CircuitTemplate:
    """QAOA ansatz for MaxCut on a ring graph.

    Alternating cost (ZZ Ising) and mixer (RX) layers.

    Cost layer: RZZ(2*gamma_k, i, i+1) for all ring edges (i, i+1 mod n).
    Mixer layer: RX(2*beta_k, i) for all qubits.

    Total parameters: ``2 * p_layers`` (gamma_0…gamma_{p-1}, beta_0…beta_{p-1}).

    Parameters
    ----------
    n_qubits:
        Number of qubits (>= 2).
    p_layers:
        Number of QAOA rounds (>= 1).

    Returns
    -------
    CircuitTemplate
    """
    _check_n_qubits(n_qubits, min_val=2)
    _check_positive_int(p_layers, "p_layers")

    ops: list[dict] = []
    param_names: list[str] = []

    # Initial superposition
    for q in range(n_qubits):
        ops.append({"name": "h", "qubits": [q]})

    # Ring edges: (0,1), (1,2), ..., (n-2, n-1), (n-1, 0)
    edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

    for p in range(p_layers):
        gamma_name = f"gamma_{p}"
        beta_name = f"beta_{p}"
        param_names.append(gamma_name)
        param_names.append(beta_name)

        # Cost layer: RZZ(2*gamma, i, j) for each edge
        # We use a fixed numeric factor by embedding 2* into the param binding.
        # Since we can't do arithmetic in the param name, we implement the
        # scaling by emitting RZZ ops with params=[gamma_name] and document
        # that the QAOA convention uses RZZ(gamma) for the cost Hamiltonian
        # (i.e. the factor-of-2 is absorbed into gamma's meaning).
        # Alternatively, we use a helper approach: store param as-is and the
        # user passes gamma = actual_gamma.
        #
        # To faithfully implement RZZ(2*gamma), we decompose:
        #   RZZ(2*gamma) via CNOT + RZ(2*gamma) + CNOT
        # But since primitives.py supports rzz directly, we use it and note
        # the user should pass the full angle.
        for (qi, qj) in edges:
            ops.append({"name": "rzz", "qubits": [qi, qj], "params": [gamma_name]})

        # Mixer layer: RX(2*beta) for each qubit
        # Same convention: store beta_name, user passes 2*beta directly.
        for q in range(n_qubits):
            ops.append({"name": "rx", "qubits": [q], "params": [beta_name]})

    assert len(param_names) == 2 * p_layers

    return CircuitTemplate(
        n_qubits=n_qubits,
        operations=ops,
        parameter_names=param_names,
        name="QAOACircuit",
        description=(
            f"{p_layers}-layer QAOA on {n_qubits}-qubit ring MaxCut. "
            "Parameters: gamma_0..gamma_{p-1} (cost), beta_0..beta_{p-1} (mixer)."
        ),
    )


# ===========================================================================
# GHZ Circuit
# ===========================================================================

def ghz_circuit(n_qubits: int) -> CircuitTemplate:
    """GHZ state preparation circuit.

    Applies H on qubit 0 followed by a CNOT ladder:
    CX(0,1), CX(1,2), …, CX(n-2, n-1).

    No free parameters.

    Parameters
    ----------
    n_qubits:
        Number of qubits (>= 2).

    Returns
    -------
    CircuitTemplate
    """
    _check_n_qubits(n_qubits, min_val=2)

    ops: list[dict] = [{"name": "h", "qubits": [0]}]
    for i in range(n_qubits - 1):
        ops.append({"name": "cx", "qubits": [i, i + 1]})

    return CircuitTemplate(
        n_qubits=n_qubits,
        operations=ops,
        parameter_names=[],
        name="GHZCircuit",
        description=f"GHZ state preparation on {n_qubits} qubits.",
    )


# ===========================================================================
# Quantum Fourier Transform
# ===========================================================================

def quantum_fourier_transform(
    n_qubits: int,
    inverse: bool = False,
) -> CircuitTemplate:
    """Quantum Fourier Transform (QFT) or its inverse on n qubits.

    Implements the standard recursive QFT decomposition:
    H gate + controlled-phase gates + SWAP network.

    Controlled-Rz decomposition (no native CRZ gate in the simulator):

    .. code-block:: text

        CRZ(theta, ctrl, tgt):
            RZ(theta/2, tgt)
            CX(ctrl, tgt)
            RZ(-theta/2, tgt)
            CX(ctrl, tgt)

    No free parameters (fully fixed circuit).

    Parameters
    ----------
    n_qubits:
        Number of qubits (>= 1).
    inverse:
        If ``True``, return the inverse QFT (IQFT).

    Returns
    -------
    CircuitTemplate
    """
    _check_n_qubits(n_qubits)

    def crz_ops(theta: float, ctrl: int, tgt: int) -> list[dict]:
        """Decompose CRZ(theta, ctrl, tgt) into RZ + CX gates."""
        return [
            {"name": "rz", "qubits": [tgt],  "params": [theta / 2]},
            {"name": "cx", "qubits": [ctrl, tgt]},
            {"name": "rz", "qubits": [tgt],  "params": [-theta / 2]},
            {"name": "cx", "qubits": [ctrl, tgt]},
        ]

    def qft_ops(n: int) -> list[dict]:
        """Build forward QFT ops on qubits 0..n-1."""
        ops: list[dict] = []
        for j in range(n):
            ops.append({"name": "h", "qubits": [j]})
            for k in range(j + 1, n):
                # Controlled-R_k: phase = 2*pi / 2^(k-j+1)
                angle = 2 * math.pi / (2 ** (k - j + 1))
                ops.extend(crz_ops(angle, j, k))
        # Bit-reversal SWAP network
        for i in range(n // 2):
            ops.append({"name": "swap", "qubits": [i, n - 1 - i]})
        return ops

    if not inverse:
        ops = qft_ops(n_qubits)
        name = "QFT"
        description = f"Quantum Fourier Transform on {n_qubits} qubits."
    else:
        # Inverse QFT = reverse + conjugate of forward QFT
        # Equivalent to reversing the forward circuit and negating angles
        forward_ops = qft_ops(n_qubits)
        rev_ops: list[dict] = []
        for op in reversed(forward_ops):
            new_op = dict(op)
            if "params" in op and op["params"]:
                new_op["params"] = [-p for p in op["params"]]
            rev_ops.append(new_op)
        ops = rev_ops
        name = "IQFT"
        description = f"Inverse Quantum Fourier Transform on {n_qubits} qubits."

    return CircuitTemplate(
        n_qubits=n_qubits,
        operations=ops,
        parameter_names=[],
        name=name,
        description=description,
    )


# ===========================================================================
# UCCSD-inspired Ansatz
# ===========================================================================

def uccsd_ansatz(n_electrons: int, n_qubits: int) -> CircuitTemplate:
    """Minimal UCCSD-inspired ansatz for quantum chemistry.

    Structure:
    1. **Hartree-Fock reference**: X gates on the first ``n_electrons`` qubits
       (occupying the lowest orbitals).
    2. **Single excitations**: For each pair (p, q) with p < n_electrons and
       q >= n_electrons, a parameterized RY rotation that mixes occupied and
       virtual orbitals.
    3. **Double excitations** (simplified): For each pair of single-excitation
       pairs ((p, q), (r, s)) taken in order, a CNOT + RZ + CNOT sequence on
       qubits q and s (the virtual orbitals).

    This is a simplified approximation suitable for testing/demonstration,
    not a rigorous UCCSD implementation.

    Parameters
    ----------
    n_electrons:
        Number of electrons (occupied orbitals). Must satisfy
        ``1 <= n_electrons < n_qubits``.
    n_qubits:
        Total number of spin-orbital modes (qubits). Must be >= 2.

    Returns
    -------
    CircuitTemplate
    """
    _check_n_qubits(n_qubits, min_val=2)
    if not isinstance(n_electrons, int) or n_electrons < 1:
        raise TemplateError(
            f"n_electrons must be a positive integer, got {n_electrons!r}."
        )
    if n_electrons >= n_qubits:
        raise TemplateError(
            f"n_electrons ({n_electrons}) must be < n_qubits ({n_qubits}) "
            "to have at least one virtual orbital."
        )

    ops: list[dict] = []
    param_names: list[str] = []

    # 1. Hartree-Fock reference state
    for q in range(n_electrons):
        ops.append({"name": "x", "qubits": [q]})

    # Occupied orbitals: 0..n_electrons-1
    # Virtual  orbitals: n_electrons..n_qubits-1
    occupied = list(range(n_electrons))
    virtual  = list(range(n_electrons, n_qubits))

    # 2. Single excitations
    singles: list[tuple[int, int]] = []
    for p in occupied:
        for q in virtual:
            pname = f"theta_s_{p}_{q}"
            param_names.append(pname)
            ops.append({"name": "ry", "qubits": [q], "params": [pname]})
            singles.append((p, q))

    # 3. Double excitations (simplified)
    for idx in range(0, len(singles) - 1, 2):
        (p, q) = singles[idx]
        (r, s) = singles[idx + 1]
        pname = f"theta_d_{idx}"
        param_names.append(pname)
        # Simplified double: CNOT(q, s) + RZ(theta, s) + CNOT(q, s)
        ops.append({"name": "cx", "qubits": [q, s]})
        ops.append({"name": "rz", "qubits": [s], "params": [pname]})
        ops.append({"name": "cx", "qubits": [q, s]})

    return CircuitTemplate(
        n_qubits=n_qubits,
        operations=ops,
        parameter_names=param_names,
        name="UCCSDansatz",
        description=(
            f"UCCSD-inspired ansatz: {n_electrons} electrons in "
            f"{n_qubits} spin-orbital modes."
        ),
    )
