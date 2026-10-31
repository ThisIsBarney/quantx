"""PennyLane devices on the native engine (M15).

``quantx.statevector`` runs a circuit on the engine's state vector, up to 25 qubits::

    import pennylane as qml

    dev = qml.device("quantx.statevector", wires=4)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Z(1))

    qml.grad(circuit)(qml.numpy.array(0.3, requires_grad=True))

The engine evolves the state; what is measured on it is PennyLane's own code, so every
measurement and shot setting behaves as on ``default.qubit``. A gate on up to four wires
runs as its matrix, and a larger one is decomposed if it has a decomposition. Gradients are
adjoint by default, both passes in the engine: one pass back through the circuit, however
many parameters it has. Where the adjoint method does not apply (finite shots, a variance)
``diff_method="best"`` falls back to parameter shift.

``quantx.mps`` runs one on the engine's block-sparse matrix-product state, at any number of
wires, with the entanglement as its limit rather than the width::

    dev = qml.device("quantx.mps", wires=100, max_bond=256)

    @qml.qnode(dev)
    def circuit(x):
        qml.Hadamard(0)
        for w in range(99):
            qml.CRX(x, wires=[w, w + 1])
        return qml.expval(qml.Z(50) @ qml.Z(51))

``max_bond`` caps the bond dimension, and is unlimited by default, so a circuit is exact
unless it is told otherwise: set it for a circuit that entangles much, and raise it until the
answer stops moving. ``qml.Tracker`` records each run's ``truncation_sum``, ``max_bond`` and
``conserved``, whether the charge below was used.
Gates on more than two wires are decomposed, and a pair that is not adjacent is brought
together by SWAPs, each a truncating update, so a chain's order of wires is the cheap one.

When every gate conserves the number of qubits in ``|1>`` (leading X gates and a BasisState
only set where the count starts), the state runs with that number as an exact charge: it is
conserved at any truncation, and each tensor splits into blocks. ``conserve="auto"``, the
default, finds this by itself; True refuses a circuit that does not conserve it, False never
uses it. It looks at the gates as they run, so a gate on more than two wires counts by its
decomposition: a DoubleExcitation conserves the count, the CNOTs it decomposes into do not.

Expectation values of anything with a Pauli form, variances, samples and shots run in the
engine at any width, as do the probabilities of a few wires. A measurement with no form on an
MPS (the state itself, a reduced density matrix, an entropy) reads the amplitudes up to 20
qubits and is refused beyond. Gradients are by parameter shift.

On both devices mid-circuit measurements are deferred, and need a free wire if the measured
wire is reused, and noise channels are refused: these are pure-state simulators. The devices
need the native library, which the wheels carry; there is no pure-Python fallback. They need
PennyLane 0.45 or later, which needs Python 3.11.
"""
from __future__ import annotations

import json
import math
from ctypes import (POINTER, c_char_p, c_double, c_longlong, c_ubyte, c_ulonglong, c_void_p,
                    cast)
from dataclasses import replace

import numpy as np
import pennylane as qp
from pennylane.devices import Device, ExecutionConfig
from pennylane.devices.default_qubit import (
    ALL_DQ_GATES,
    accepted_analytic_measurement,
    accepted_sample_measurement,
    adjoint_observables,
    adjoint_ops,
    adjoint_state_measurements,
)
from pennylane.devices.modifiers import simulator_tracking, single_tape_support
from pennylane.devices.preprocess import (
    decompose,
    device_resolve_dynamic_wires,
    no_sampling,
    null_postprocessing,
    validate_adjoint_trainable_params,
    validate_device_wires,
    validate_measurements,
    validate_observables,
)
from pennylane.devices.qubit import adjoint_jacobian, apply_operation, measure
from pennylane.devices.qubit.simulate import measure_final_state
from pennylane.exceptions import DeviceError
from pennylane.math import Interface
from pennylane.measurements import ExpectationMP, ProbabilityMP, StateMP, VarianceMP
from pennylane.operation import StatePrepBase, operation_derivative
from pennylane.transforms import broadcast_expand, defer_measurements, split_non_commuting
from pennylane.transforms.core import CompilePipeline

from quantx_engine import __version__ as _VERSION
from quantx_engine.engine import Engine

__all__ = ["StatevectorDevice", "MPSDevice"]

# The gates the engine has a kernel for. Everything else runs as its matrix.
_ONE_QUBIT = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Hadamard": "H", "S": "S", "T": "T",
              "RX": "RX", "RY": "RY", "RZ": "RZ"}
_CONTROLLED = {"CNOT", "CZ"}
_SELF_INVERSE = {"PauliX", "PauliY", "PauliZ", "Hadamard", "CNOT", "CZ", "SWAP"}
_ROTATIONS = {"RX", "RY", "RZ"}
# These do nothing to a measurement: Identity and Snapshot leave the state alone, and a global
# phase cancels in every <bra|ket> of the backward pass.
_NOT_GATES = {"Identity", "Snapshot", "GlobalPhase"}

# The most wires a gate runs on as its matrix when it could be decomposed instead. Two to the
# power of this is the matrix's side, so it is 4^k in the request and 2^k work per amplitude,
# against a DoubleExcitation's 14 CNOTs and 14 rotations.
_MAX_MATRIX_WIRES = 4


def _bind(lib, name: str, argtypes: list):
    """One native entry point with its C types. A library that lacks it is refused."""
    try:
        fn = getattr(lib, name)
    except AttributeError:
        raise DeviceError(f"the loaded native library does not export {name}; it is older "
                          "than this Python package") from None
    fn.argtypes = argtypes
    fn.restype = c_void_p
    return fn


def _doubles(a: np.ndarray):
    return a.ctypes.data_as(POINTER(c_double))


def _pairs(m) -> list:
    """A matrix's entries, row-major, as re, im, re, im, ..."""
    m = np.asarray(m, dtype=np.complex128).ravel()
    return np.column_stack((m.real, m.imag)).ravel().tolist()


def _qubits(op, n: int) -> list:
    """PennyLane's wire 0 is the high bit of an amplitude's index and the engine's qubit 0 the
    low bit, so wire w is qubit n - 1 - w and the two number amplitudes the same way."""
    return [n - 1 - int(w) for w in op.wires]


def _angle(op):
    """A rotation's real angle, or None if it has an imaginary part. The adjoint pipeline casts
    angles to complex; one that is truly complex takes the matrix route, where the engine
    refuses it as not unitary."""
    angle = np.real_if_close(op.data[0])
    return None if np.iscomplexobj(angle) else float(angle)


def _matrix_gate(m, qubits: list) -> dict:
    return {"name": "UNITARY", "targets": qubits, "params": _pairs(m)}


def _gate(op, n: int) -> dict:
    """The engine's form of ``op``."""
    name, qubits = op.name, _qubits(op, n)
    if name in _CONTROLLED:
        return {"name": name, "controls": qubits[:1], "targets": qubits[1:]}
    if name == "SWAP":
        return {"name": name, "targets": qubits}
    if name in _ONE_QUBIT:
        if not op.data:
            return {"name": _ONE_QUBIT[name], "targets": qubits}
        angle = _angle(op)
        if angle is not None:
            return {"name": _ONE_QUBIT[name], "targets": qubits, "params": [angle]}
    return _matrix_gate(op.matrix(), qubits)


def _inverse_gate(op, n: int) -> dict:
    """The engine's form of the adjoint of ``op``."""
    if op.name in _SELF_INVERSE:
        return _gate(op, n)
    if op.name in _ROTATIONS:
        angle = _angle(op)
        if angle is not None:
            return {"name": op.name, "targets": _qubits(op, n), "params": [-angle]}
    return _matrix_gate(np.conj(np.asarray(op.matrix())).T, _qubits(op, n))


def _derivative(op):
    """dU/dtheta of a one-parameter gate. PennyLane's generator machinery gives it for any gate,
    at a tenth of a millisecond a call; a layer of rotations is most of a gradient's calls."""
    if op.name in _ROTATIONS:
        angle = _angle(op)
        if angle is not None:
            c, s = 0.5 * math.cos(angle / 2), 0.5 * math.sin(angle / 2)
            if op.name == "RX":
                return ((-s, -1j * c), (-1j * c, -s))
            if op.name == "RY":
                return ((-s, -c), (c, -s))
            return ((complex(-s, -c), 0), (0, complex(-s, c)))
    return operation_derivative(op)


def _supported(op) -> bool:
    """Whether ``op`` runs as it is. Otherwise ``decompose`` takes it apart."""
    if op.name in _NOT_GATES:
        return True
    if op.__class__.__name__[:3] == "Pow" and any(qp.math.requires_grad(d) for d in op.data):
        return False
    if not op.has_matrix:
        return False
    return len(op.wires) <= _MAX_MATRIX_WIRES or not op.has_decomposition


def _backward_steps(circuit, n: int) -> list:
    """The backward pass of ``circuit``: its gates last to first, each with its derivative if
    its parameter is being differentiated.

    Parameters are numbered through the whole circuit, so an op's place in the numbering is
    found going forward: every op counts all its parameters, frozen or not. (The adjoint
    pipeline leaves a frozen Rot whole, with its three.)"""
    columns = {p: c for c, p in enumerate(circuit.trainable_params)}
    index = sum(op.num_params for op in circuit.operations[:circuit.num_preps])
    steps = []
    for op in circuit.operations[circuit.num_preps:]:
        if op.name not in _NOT_GATES:
            step = {"inverse": _inverse_gate(op, n)}
            if op.num_params == 1 and index in columns:
                step.update(targets=_qubits(op, n), derivative=_pairs(_derivative(op)),
                            column=columns[index])
            steps.append(step)
        index += op.num_params
    steps.reverse()
    return steps


def _jacobian_shape(jac: np.ndarray):
    """PennyLane's nesting of a Jacobian: squeezed, with tuples for what is left."""
    jac = np.squeeze(jac)
    if jac.ndim == 0:
        return np.array(jac)
    if jac.ndim == 1:
        return tuple(np.array(j) for j in jac)
    return tuple(tuple(np.array(j_) for j_ in j) for j in jac)


@simulator_tracking
@single_tape_support
class StatevectorDevice(Device):
    """State-vector simulation on the native engine.

    Args:
        wires (int or sequence): the device's wires, at most 25, or None to take them from
            each circuit.
        shots: the default shots, as on any PennyLane device.
        seed: seeds the sampling of finite-shot measurements, as on ``default.qubit``; the
            state evolution is deterministic. ``"global"`` draws it from NumPy's global state.
    """

    name = "quantx.statevector"
    short_name = "quantx.statevector"
    version = _VERSION
    author = "QuantX"

    def __init__(self, wires=None, shots=None, seed="global"):
        super().__init__(wires=wires, shots=shots)
        lib = Engine()._lib
        pointer = POINTER(c_double)
        self._forward = _bind(lib, "qx_run_circuit_state",
                              [c_char_p, c_char_p, pointer, pointer, c_ulonglong])
        self._backward = _bind(lib, "qx_circuit_adjoint",
                               [c_char_p, pointer, pointer, c_ulonglong, c_ulonglong, pointer,
                                c_ulonglong])
        self._free = lib.qx_free_string
        seed = np.random.randint(0, high=10_000_000) if isinstance(seed, str) else seed
        self._rng = np.random.default_rng(seed)

    def _call(self, fn, *args) -> None:
        """Call an entry point that writes to buffers, and raise what it reports."""
        ptr = fn(*args)
        if not ptr:
            raise DeviceError("the native engine returned no result")
        try:
            out = json.loads(cast(ptr, c_char_p).value)
        finally:
            self._free(ptr)
        if not out.get("ok", False):
            raise DeviceError(out.get("error_message", out.get("error", "engine failure")))

    # ---- execution ---------------------------------------------------------------------

    def _final_state(self, circuit) -> np.ndarray:
        """The state after ``circuit``, on wires 0..n-1, as a tensor of shape (2,) * n."""
        n = circuit.num_wires
        ops = circuit.operations
        initial = None
        if ops and isinstance(ops[0], StatePrepBase):
            initial = np.asarray(ops[0].state_vector(wire_order=list(range(n))),
                                 dtype=np.complex128).ravel()
            ops = ops[1:]
        gates = [_gate(op, n) for op in ops if op.name not in _NOT_GATES]
        state = np.empty(2 ** n, dtype=np.complex128)
        if gates:
            request = json.dumps({"qubits": n, "operations": gates}).encode()
            start = None if initial is None else _doubles(initial)
            self._call(self._forward, request, b"{}", start, _doubles(state), state.size)
        elif initial is None:               # the engine refuses an empty circuit
            state[:] = 0
            state[0] = 1
        else:
            state[:] = initial
        for op in ops:
            if op.name == "GlobalPhase":
                state *= np.exp(-1j * op.data[0])
        return state.reshape((2,) * n)

    def _simulate(self, circuit):
        circuit = circuit.map_to_standard_wires()
        return measure_final_state(circuit, self._final_state(circuit), False, rng=self._rng)

    def execute(self, circuits, execution_config: ExecutionConfig | None = None):
        return tuple(self._simulate(c) for c in circuits)

    # ---- derivatives -------------------------------------------------------------------

    def _jacobian(self, circuit, state: np.ndarray):
        """d<O_r>/dtheta_c from the final ``state``, by the engine's backward pass."""
        if isinstance(circuit.measurements[0], StateMP):
            return adjoint_jacobian(circuit)        # the Jacobian of the state itself
        n = circuit.num_wires
        observables = circuit.observables
        bras = np.empty((len(observables), 2 ** n), dtype=np.complex128)
        for r, obs in enumerate(observables):
            bras[r] = 2 * np.asarray(apply_operation(obs, state)).ravel()
        jac = np.empty((len(observables), len(circuit.trainable_params)))
        request = json.dumps({"qubits": n, "steps": _backward_steps(circuit, n)}).encode()
        self._call(self._backward, request, _doubles(state), _doubles(bras), 2 ** n,
                   len(observables), _doubles(jac), jac.shape[1])
        return _jacobian_shape(jac)

    def _add_adjoint_transforms(self, program) -> None:
        name = f"adjoint + {self.name}"
        program.add_transform(no_sampling, name=name)
        program.add_transform(decompose, stopping_condition=adjoint_ops, device_wires=self.wires,
                              target_gates=ALL_DQ_GATES, name=name,
                              skip_initial_state_prep=False)
        program.add_transform(validate_observables, adjoint_observables, name=name)
        program.add_transform(validate_measurements, name=name)
        program.add_transform(adjoint_state_measurements, device_vjp=False)
        program.add_transform(broadcast_expand)
        program.add_transform(validate_adjoint_trainable_params)

    def supports_derivatives(self, execution_config=None, circuit=None) -> bool:
        if execution_config is None:
            return True
        if execution_config.gradient_method not in {"adjoint", "best"}:
            return False
        if execution_config.derivative_order != 1:
            return False
        if circuit is None:
            return True
        # Whether the pipeline that would run takes this circuit. Anything it raises means it
        # does not, and "best" then falls back to parameter shift: PennyLane's transforms have
        # no one exception for that (an SProd left by a decomposition raises an IndexError).
        program = self._common_transforms()
        self._add_adjoint_transforms(program)
        try:
            program((circuit,))
        except Exception:  # pylint: disable=broad-except
            return False
        return True

    def compute_derivatives(self, circuits, execution_config: ExecutionConfig | None = None):
        out = []
        for c in circuits:
            c = c.map_to_standard_wires()
            out.append(self._jacobian(c, self._final_state(c)))
        return tuple(out)

    def execute_and_compute_derivatives(self, circuits,
                                        execution_config: ExecutionConfig | None = None):
        results, jacobians = [], []
        for c in circuits:
            c = c.map_to_standard_wires()
            state = self._final_state(c)
            jacobians.append(self._jacobian(c, state))
            results.append(measure_final_state(c, state, False, rng=self._rng))
        return tuple(results), tuple(jacobians)

    # ---- preprocessing -----------------------------------------------------------------

    def setup_execution_config(self, config=None, circuit=None) -> ExecutionConfig:
        config = config or ExecutionConfig()
        if config.mcm_config.mcm_method not in {None, "deferred"}:
            raise DeviceError(f"{self.name} supports the deferred measurement principle only, "
                              f"not {config.mcm_config.mcm_method}")
        for option in config.device_options:
            raise DeviceError(f"device option {option} not present on {self}")
        method = "adjoint" if config.gradient_method == "best" else config.gradient_method
        updated = {"gradient_method": method,
                   "mcm_config": replace(config.mcm_config, mcm_method="deferred")}
        if config.use_device_gradient is None:
            updated["use_device_gradient"] = method == "adjoint"
        if config.use_device_jacobian_product is None:
            updated["use_device_jacobian_product"] = False
        if config.grad_on_execution is None:
            updated["grad_on_execution"] = method == "adjoint"
        return replace(config, **updated)

    def _common_transforms(self) -> CompilePipeline:
        program = CompilePipeline()
        program.add_transform(defer_measurements, allow_postselect=False)
        program.add_transform(decompose, stopping_condition=_supported, device_wires=self.wires,
                              target_gates=ALL_DQ_GATES, name=self.name)
        program.add_transform(device_resolve_dynamic_wires, wires=self.wires, allow_resets=False)
        program.add_transform(validate_device_wires, self.wires, name=self.name)
        program.add_transform(validate_measurements,
                              analytic_measurements=accepted_analytic_measurement,
                              sample_measurements=accepted_sample_measurement, name=self.name)
        program.add_transform(broadcast_expand)
        return program

    def preprocess_transforms(self, execution_config: ExecutionConfig | None = None):
        config = execution_config or ExecutionConfig()
        program = self._common_transforms()
        # "best" resolves to adjoint on every call, differentiated or not, and those transforms
        # rewrite a circuit for the gradient. With NumPy parameters there is none to take.
        if config.gradient_method == "adjoint" and config.interface != Interface.NUMPY:
            self._add_adjoint_transforms(program)
        return program


# ---- the MPS device ------------------------------------------------------------------------

# Up to here a measurement with no form on an MPS reads the state's amplitudes.
_MAX_DENSE_QUBITS = 20
# The most wires a Hermitian observable is written in Pauli strings for: 4^k of them.
_MAX_PAULI_WIRES = 6
# The most wires whose probabilities are read past _MAX_DENSE_QUBITS: 2^k strings.
_MAX_PROBS_WIRES = 16
# The engine refuses a charge-changing entry above 1e-10; this is the bar for not trying.
_CHARGE_TOL = 1e-12
_POPULATION = np.array([0, 1, 1, 2])
_MIXES_CHARGE = _POPULATION[:, None] != _POPULATION[None, :]


def _conserves(m: np.ndarray) -> bool:
    """Whether a one- or two-qubit gate conserves the number of |1>s."""
    if m.shape == (2, 2):
        return max(abs(m[0, 1]), abs(m[1, 0])) <= _CHARGE_TOL
    return float(np.max(np.abs(m[_MIXES_CHARGE]))) <= _CHARGE_TOL


def _two_wires(op) -> bool:
    """Whether ``op`` runs on the MPS as it is. Otherwise ``decompose`` takes it apart."""
    if op.name in _NOT_GATES:
        return True
    if op.__class__.__name__[:3] == "Pow" and any(qp.math.requires_grad(d) for d in op.data):
        return False
    return op.has_matrix and len(op.wires) <= 2


def _sentence(obs):
    """``obs`` as Pauli strings, or None if it has no form the engine can read."""
    rep = obs.pauli_rep
    if rep is not None:
        return rep
    if obs.has_matrix and len(obs.wires) <= _MAX_PAULI_WIRES:
        return qp.pauli_decompose(qp.matrix(obs), wire_order=obs.wires, pauli=True)
    return None


@qp.transform
def _split_for_sampling(tape):
    """Samples read one basis at a time. An analytic expectation needs no splitting."""
    if not tape.shots:
        return (tape,), null_postprocessing
    return split_non_commuting(tape, grouping_strategy="qwc")


@qp.transform
def _rotate_for_sampling(tape):
    """Bits are read in the computational basis, so the circuit ends with what rotates it into
    the observables' eigenbasis. The observables stay as they are: process_samples numbers an
    observable's eigenvalues the way its diagonalizing gates number the basis."""
    if not tape.shots:
        return (tape,), null_postprocessing
    return (tape.copy(operations=tape.operations + tape.diagonalizing_gates),), null_postprocessing


def _decomposition(op) -> list:
    """How ``decompose`` takes ``op`` apart. A gate with a matrix and no decomposition of its own
    (a DoubleExcitationPlus, say) goes through QubitUnitary's, which handles any size."""
    if op.has_matrix and not op.has_decomposition:
        return [qp.QubitUnitary(op.matrix(), wires=op.wires)]
    return op.decomposition()


@simulator_tracking
@single_tape_support
class MPSDevice(Device):
    """Matrix-product-state simulation on the native engine.

    Args:
        wires (int or sequence): the device's wires in chain order, or None to take them from
            each circuit.
        shots: the default shots, as on any PennyLane device.
        seed: seeds the sampling of finite-shot measurements. ``"global"`` draws it from
            NumPy's global state.
        max_bond (int): the largest bond dimension, or None for no limit: exact, at whatever
            bond dimension the circuit needs.
        cutoff (float): Schmidt values below this fraction of a two-site tensor's norm are
            dropped at every update.
        conserve: "auto" uses the number of ``|1>``s as an exact charge when every gate
            conserves it; True refuses a circuit that does not, and False never uses it.
        svd (str): "direct", or "gram", a faster SVD by the eigensolve of the Gram matrix, which
            resolves Schmidt values only down to the square root of machine precision.
    """

    name = "quantx.mps"
    short_name = "quantx.mps"
    version = _VERSION
    author = "QuantX"

    def __init__(self, wires=None, shots=None, seed="global", max_bond=None, cutoff=1e-14,
                 conserve="auto", svd="direct"):
        super().__init__(wires=wires, shots=shots)
        if max_bond is not None and (isinstance(max_bond, bool) or int(max_bond) != max_bond
                                     or max_bond < 1):
            raise DeviceError(f"max_bond must be a positive integer or None, got {max_bond!r}")
        if not cutoff >= 0:
            raise DeviceError(f"cutoff must be >= 0, got {cutoff!r}")
        if conserve not in ("auto", True, False):
            raise DeviceError(f'conserve must be "auto", True or False, got {conserve!r}')
        if svd not in ("direct", "gram"):
            raise DeviceError(f'svd must be "direct" or "gram", got {svd!r}')
        self._max_bond, self._cutoff, self._conserve = max_bond, float(cutoff), conserve
        self._direct = svd == "direct"
        lib = Engine()._lib
        self._mps = _bind(lib, "qx_mps", [c_char_p])
        self._mps.restype = c_char_p
        self._sample = _bind(lib, "qx_mps_sample",
                             [c_longlong, c_ulonglong, c_ulonglong, POINTER(c_ubyte)])
        self._sample.restype = c_char_p
        self._amplitudes = _bind(lib, "qx_mps_amplitudes",
                                 [c_longlong, POINTER(c_double), c_ulonglong])
        self._amplitudes.restype = c_char_p
        seed = np.random.randint(0, high=10_000_000) if isinstance(seed, str) else seed
        self._rng = np.random.default_rng(seed)

    def _call(self, request: dict) -> dict:
        return self._check(self._mps(json.dumps(request).encode()))

    @staticmethod
    def _check(raw: bytes) -> dict:
        out = json.loads(raw)
        if not out.get("ok", False):
            raise DeviceError(out.get("error", "engine failure"))
        return out

    # ---- execution ---------------------------------------------------------------------

    def _prepare(self, circuit) -> tuple[int, complex]:
        """A handle to the state after ``circuit``, and the global phase its gates carry."""
        n = circuit.num_wires
        bits = [0] * n
        touched = set()
        gates = []
        phase = 1.0 + 0.0j
        for op in circuit.operations:
            if op.name == "GlobalPhase":
                phase *= np.exp(-1j * op.data[0])
            elif op.name not in _NOT_GATES:
                wires = [int(w) for w in op.wires]
                # An X on a wire nothing has touched only moves where the count starts.
                if op.name == "PauliX" and wires[0] not in touched:
                    bits[wires[0]] ^= 1
                    continue
                touched.update(wires)
                gates.append((wires, np.asarray(op.matrix(), dtype=np.complex128)))
        conserve = self._conserve
        if conserve == "auto":
            conserve = all(_conserves(m) for _, m in gates)
        handle = self._call({"op": "qubits", "n_qubits": n, "bits": bits,
                             "conserve": bool(conserve)})["handle"]
        try:
            stats = {"max_bond": 1, "truncation_sum": 0.0}
            if gates:
                # No limit is a bond of 2^(n/2), which no state exceeds.
                chi = self._max_bond or 2 ** min(n // 2 + 1, 30)
                stats = self._call({"op": "gates", "handle": handle,
                                    "gates": [{"q": w, "u": _pairs(m)} for w, m in gates],
                                    "max_bond": int(chi), "cutoff": self._cutoff,
                                    "direct": self._direct})
        except Exception:
            self._call({"op": "free", "handle": handle})
            raise
        if self.tracker.active:
            self.tracker.update(max_bond=stats["max_bond"],
                                truncation_sum=stats["truncation_sum"],
                                conserved=bool(conserve))
        return handle, phase

    def _paulis(self, handle: int, sentence) -> float:
        """<sentence> on the state, normalized."""
        terms, coeffs = [], []
        for word, coeff in sentence.items():
            terms.append([[int(w), p] for w, p in sorted(word.items())])
            coeffs.append(complex(coeff).real)
        if not terms:
            return 0.0
        values = self._call({"op": "pauli", "handle": handle, "terms": terms})["values"]
        return float(np.dot(coeffs, values))

    def _dense_state(self, handle: int, n: int, phase: complex, what) -> np.ndarray:
        if n > _MAX_DENSE_QUBITS:
            raise DeviceError(f"{what} has no form on an MPS, and reading the amplitudes of "
                              f"{n} qubits would take 2^{n} of them: {self.name} does it up "
                              f"to {_MAX_DENSE_QUBITS} qubits")
        state = np.empty(2 ** n, dtype=np.complex128)
        self._check(self._amplitudes(handle, _doubles(state), state.size))
        state *= phase / np.linalg.norm(state)
        return state.reshape((2,) * n)

    def _probabilities(self, handle: int, wires: list) -> np.ndarray:
        """The distribution of ``wires``, from its Z strings: P(b) = 2^-k sum_z (-1)^(b.z) <Z^z>."""
        k = len(wires)
        if k > _MAX_PROBS_WIRES:
            raise DeviceError(f"the probabilities of {k} wires of a large state are 2^{k} "
                              f"strings: {self.name} does it for {_MAX_PROBS_WIRES}")
        terms = [[[w, "Z"] for w in sorted(wires[j] for j in range(k) if (z >> (k - 1 - j)) & 1)]
                 for z in range(2 ** k)]
        p = np.array(self._call({"op": "pauli", "handle": handle, "terms": terms})["values"])
        p = p.reshape((2,) * k)
        for axis in range(k):                       # a Walsh-Hadamard transform
            p = np.stack((p.take(0, axis) + p.take(1, axis),
                          p.take(0, axis) - p.take(1, axis)), axis=axis)
        return np.maximum(p.ravel() / 2 ** k, 0.0)

    def _measure(self, mp, handle: int, n: int, phase: complex):
        # With no observable, it is of a mid-circuit measurement's value, kept on its wires.
        if isinstance(mp, (ExpectationMP, VarianceMP)) and mp.obs is not None:
            sentence = _sentence(mp.obs)
            if sentence is not None:
                mean = self._paulis(handle, sentence)
                if isinstance(mp, ExpectationMP):
                    return np.array(mean)
                return np.array(self._paulis(handle, sentence @ sentence) - mean ** 2)
        if isinstance(mp, ProbabilityMP) and mp.obs is None and n > _MAX_DENSE_QUBITS:
            return self._probabilities(handle, [int(w) for w in mp.wires])
        return measure(mp, self._dense_state(handle, n, phase, type(mp).__name__), False)

    def _sampled(self, circuit, handle: int):
        n = circuit.num_wires
        shots = circuit.shots
        samples = np.empty((shots.total_shots, n), dtype=np.uint8)
        self._check(self._sample(handle, shots.total_shots, int(self._rng.integers(2 ** 63)),
                                 samples.ctypes.data_as(POINTER(c_ubyte))))
        samples = samples.astype(np.int64)
        # Column j is wire j. circuit.wires is in order of first use, which is not that.
        order = qp.wires.Wires(range(n))
        results = []
        for lower, upper in shots.bins():
            piece = samples[lower:upper]
            results.append(tuple(mp.process_samples(piece, wire_order=order)
                                 for mp in circuit.measurements))
        if len(circuit.measurements) == 1:
            results = [r[0] for r in results]
        return tuple(results) if shots.has_partitioned_shots else results[0]

    def _on_chain(self, circuit):
        """``circuit`` on wires 0..n-1, with the wires it uses in the device's order.

        On an MPS that order is the chain, and what is cheap depends on it. PennyLane's
        map_to_standard_wires numbers wires by first use, which would put the first two wires
        a circuit touches side by side whatever the device says. Unused wires are left out."""
        used = circuit.wires
        if self.wires is not None:
            chain = [w for w in self.wires if w in used]
        else:
            try:
                chain = sorted(used)
            except TypeError:           # labels of mixed types have no order but first use
                chain = list(used)
        wire_map = {w: i for i, w in enumerate(chain)}
        if all(w == i for w, i in wire_map.items()):
            return circuit
        tapes, fn = qp.map_wires(circuit, wire_map)
        return fn(tapes)

    def _simulate(self, circuit):
        circuit = self._on_chain(circuit)
        n = circuit.num_wires
        if n == 0:          # measurements of the identity, on no wires at all
            return measure_final_state(circuit, np.array(1.0 + 0.0j), False, rng=self._rng)
        handle, phase = self._prepare(circuit)
        try:
            if circuit.shots:
                return self._sampled(circuit, handle)
            results = tuple(self._measure(mp, handle, n, phase) for mp in circuit.measurements)
            return results[0] if len(results) == 1 else results
        finally:
            self._call({"op": "free", "handle": handle})

    def execute(self, circuits, execution_config: ExecutionConfig | None = None):
        return tuple(self._simulate(c) for c in circuits)

    # ---- preprocessing -----------------------------------------------------------------

    def setup_execution_config(self, config=None, circuit=None) -> ExecutionConfig:
        config = config or ExecutionConfig()
        if config.mcm_config.mcm_method not in {None, "deferred"}:
            raise DeviceError(f"{self.name} supports the deferred measurement principle only, "
                              f"not {config.mcm_config.mcm_method}")
        for option in config.device_options:
            raise DeviceError(f"device option {option} not present on {self}")
        return replace(config, mcm_config=replace(config.mcm_config, mcm_method="deferred"))

    def preprocess_transforms(self, execution_config: ExecutionConfig | None = None):
        program = CompilePipeline()
        program.add_transform(defer_measurements, allow_postselect=False)
        program.add_transform(_split_for_sampling)
        program.add_transform(_rotate_for_sampling)
        # skip_initial_state_prep=False: an MPS has no amplitudes to start from. A BasisState
        # comes out as X gates, which _prepare folds into the initial bits.
        program.add_transform(decompose, stopping_condition=_two_wires, device_wires=self.wires,
                              decomposer=_decomposition, target_gates=ALL_DQ_GATES,
                              name=self.name, skip_initial_state_prep=False)
        program.add_transform(device_resolve_dynamic_wires, wires=self.wires, allow_resets=False)
        program.add_transform(validate_device_wires, self.wires, name=self.name)
        program.add_transform(validate_measurements,
                              analytic_measurements=accepted_analytic_measurement,
                              sample_measurements=accepted_sample_measurement, name=self.name)
        program.add_transform(broadcast_expand)
        return program
