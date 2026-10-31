"""M15 gate: PennyLane circuits on the engine's state vector.

Two references, for two layers:

* the engine: ``qx_run_circuit_state``'s UNITARY gate and start state against a state vector
  written here (its own einsum, its own qubit order), so that nothing it shares with
  PennyLane can hide a mistake in the kernels.
* the device: every PennyLane qubit gate, measurement and gradient against ``default.qubit``,
  which is what the device claims to behave as. PennyLane's own device suite and its demos
  run elsewhere (``test_pennylane_demos.py``, and the native CI job).
"""
from __future__ import annotations

import json
import math
import os
import zlib
from ctypes import POINTER, c_char_p, c_double, c_ulonglong, c_void_p, cast

import numpy as np
import pytest

qml = pytest.importorskip("pennylane", minversion="0.45")

from pennylane.exceptions import DeviceError, WireError  # noqa: E402

TOL = 1e-12


@pytest.fixture(scope="module")
def device_class():
    try:
        from quantx_engine.pennylane_device import StatevectorDevice
        StatevectorDevice(wires=1)
    except Exception as e:      # CI's native job sets QX_REQUIRE_NATIVE=1: there it must load
        unavailable = pytest.fail if os.environ.get("QX_REQUIRE_NATIVE") == "1" else pytest.skip
        unavailable(f"native engine unavailable: {e}")
    return StatevectorDevice


@pytest.fixture(scope="module")
def ref():
    return qml.device("default.qubit")


def _random_state(rng, n):
    s = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return s / np.linalg.norm(s)


def _random_unitary(rng, dim):
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


# ── the engine, against a state vector written here ──────────────────────────


def _run(lib, circuit, initial=None, size=None):
    fn = lib.qx_run_circuit_state
    fn.argtypes = [c_char_p, c_char_p, POINTER(c_double), POINTER(c_double), c_ulonglong]
    fn.restype = c_void_p
    n = circuit["qubits"]
    state = np.full(2 ** n, 7.0 + 0j)       # a filler no run can leave behind unnoticed
    start = None
    if initial is not None:
        initial = np.asarray(initial, dtype=np.complex128)
        start = initial.ctypes.data_as(POINTER(c_double))
    ptr = fn(json.dumps(circuit).encode(), b"{}", start, state.ctypes.data_as(POINTER(c_double)),
             state.size if size is None else size)
    try:
        return json.loads(cast(ptr, c_char_p).value), state
    finally:
        lib.qx_free_string(ptr)


def _unitary_op(u, targets):
    m = np.asarray(u, dtype=np.complex128).ravel()
    return {"name": "UNITARY", "targets": list(targets),
            "params": np.column_stack((m.real, m.imag)).ravel().tolist()}


def _einsum_apply(state, u, targets, n):
    """U on qubits ``targets`` (targets[0] the matrix's high bit); qubit q is bit q of an index."""
    k = len(targets)
    t = state.reshape((2,) * n)                 # axis a is qubit n - 1 - a
    axes = [n - 1 - q for q in targets]
    t = np.tensordot(u.reshape((2,) * (2 * k)), t, axes=(list(range(k, 2 * k)), axes))
    return np.moveaxis(t, list(range(k)), axes).ravel()


@pytest.fixture(scope="module")
def lib(device_class):
    from quantx_engine.engine import Engine
    return Engine()._lib


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_engine_unitary_kernel_vs_einsum(lib, k):
    rng = np.random.default_rng(100 + k)
    n = 6
    for _ in range(12):
        psi = _random_state(rng, n)
        u = _random_unitary(rng, 2 ** k)
        targets = [int(q) for q in rng.permutation(n)[:k]]
        out, state = _run(lib, {"qubits": n, "operations": [_unitary_op(u, targets)]}, psi)
        assert out["ok"], out
        assert np.max(np.abs(state - _einsum_apply(psi, u, targets, n))) < TOL


def test_engine_kernels_past_the_parallel_threshold(lib):
    # 2^14 amplitudes is where the kernels start to thread: every kind against einsum there.
    rng = np.random.default_rng(1)
    n = 15
    psi = _random_state(rng, n)
    u2, u3 = _random_unitary(rng, 4), _random_unitary(rng, 8)
    h = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    cx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])     # control high
    sw = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    cz = np.diag([1, 1, 1, -1])
    ops = [{"name": "H", "targets": [14]}, {"name": "CNOT", "controls": [3], "targets": [11]},
           {"name": "CNOT", "controls": [12], "targets": [0]},
           {"name": "CZ", "controls": [13], "targets": [2]},
           {"name": "SWAP", "targets": [1, 9]}, _unitary_op(u2, [14, 0]), _unitary_op(u3, [7, 2, 13])]
    want = psi
    for u, targets in ((h, [14]), (cx, [3, 11]), (cx, [12, 0]), (cz, [13, 2]), (sw, [1, 9]),
                       (u2, [14, 0]), (u3, [7, 2, 13])):
        want = _einsum_apply(want, u, targets, n)
    out, state = _run(lib, {"qubits": n, "operations": ops}, psi)
    assert out["ok"], out
    assert np.max(np.abs(state - want)) < 1e-11


def test_engine_starts_from_zero_without_an_initial_state(lib):
    out, state = _run(lib, {"qubits": 3, "operations": [{"name": "X", "targets": [1]}]})
    assert out["ok"], out
    want = np.zeros(8, complex)
    want[2] = 1                                 # qubit 1 is bit 1
    assert np.array_equal(state, want)


@pytest.mark.parametrize("op, message", [
    (_unitary_op(2 * np.eye(2), [0]), "not unitary"),
    (_unitary_op(np.eye(4), [0, 0]), "repeats"),
    (_unitary_op(np.eye(2), [5]), "out-of-range"),
    ({"name": "UNITARY", "targets": [0], "params": [1.0, 0.0]}, "params"),
    ({"name": "UNITARY", "targets": [], "params": []}, "targets"),
    ({"name": "NOPE", "targets": [0]}, "unsupported gate"),
])
def test_engine_refuses_a_bad_gate_and_leaves_the_buffer_alone(lib, op, message):
    out, state = _run(lib, {"qubits": 2, "operations": [op]})
    assert not out["ok"]
    assert message in out["error_message"]
    assert np.all(state == 7.0)


def test_engine_refuses_a_buffer_of_the_wrong_size(lib):
    out, state = _run(lib, {"qubits": 2, "operations": [{"name": "X", "targets": [0]}]}, size=3)
    assert not out["ok"]
    assert "2^qubits" in out["error_message"]
    assert np.all(state == 7.0)


def test_engine_noise_and_measurement_paths_use_the_start_state(lib):
    # A MEASURE on a start state of |1>: the outcome must come from it, not from |0>.
    start = np.array([0, 0, 1, 0], complex)     # qubit 1 up
    out, state = _run(lib, {"qubits": 2, "clbits": 1, "shots": 16, "operations": [
        {"name": "MEASURE", "targets": [1], "controls": [0]}]}, start)
    assert out["ok"], out
    assert out["measurement_results"] == [1]
    assert out["counts"] == {"10": 16}


def _adjoint(lib, request, ket, bras, columns):
    fn = lib.qx_circuit_adjoint
    pointer = POINTER(c_double)
    fn.argtypes = [c_char_p, pointer, pointer, c_ulonglong, c_ulonglong, pointer, c_ulonglong]
    fn.restype = c_void_p
    bras = np.asarray(bras, dtype=np.complex128)
    jac = np.full((bras.shape[0], columns), 7.0)
    ptr = fn(json.dumps(request).encode(), ket.ctypes.data_as(pointer),
             bras.ctypes.data_as(pointer), ket.size, bras.shape[0], jac.ctypes.data_as(pointer),
             columns)
    try:
        return json.loads(cast(ptr, c_char_p).value), jac
    finally:
        lib.qx_free_string(ptr)


def _pairs(m):
    m = np.asarray(m, dtype=np.complex128).ravel()
    return np.column_stack((m.real, m.imag)).ravel().tolist()


def test_engine_adjoint_vs_central_differences(lib):
    # A circuit of e^{-i theta H} gates on random Hermitian H, written here: the engine gets
    # each gate's inverse and derivative, and central differences of a state vector built
    # with this file's einsum are the reference. Two observables, so two rows.
    rng = np.random.default_rng(21)
    n = 5
    gates = []
    for k in (1, 2, 3, 2, 1, 2):
        d = 2 ** k
        a = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        w, v = np.linalg.eigh(a + a.conj().T)
        gates.append(([int(q) for q in rng.permutation(n)[:k]], w, v))
    fixed = _random_unitary(rng, 4)
    psi0 = _random_state(rng, n)
    theta = rng.uniform(-1, 1, size=len(gates))
    obs = [(_random_unitary(rng, 4), [0, 3]), (_random_unitary(rng, 4), [1, 4])]
    obs = [((u + u.conj().T) / 2, t) for u, t in obs]

    def unitary(g, t):
        _, w, v = g
        return (v * np.exp(-1j * t * w)) @ v.conj().T

    def forward(angles):
        s = psi0
        for g, t in zip(gates, angles):
            s = _einsum_apply(s, unitary(g, t), g[0], n)
        return _einsum_apply(s, fixed, [2, 0], n)   # a gate with no parameter, last

    def expect(s):
        return np.array([np.vdot(s, _einsum_apply(s, o, t, n)).real for o, t in obs])

    ket = forward(theta)
    bras = [2 * _einsum_apply(ket, o, t, n) for o, t in obs]
    steps = [{"inverse": _unitary_op(fixed.conj().T, [2, 0])}]
    for c in reversed(range(len(gates))):
        g = gates[c]
        u = unitary(g, theta[c])
        du = (g[2] * (-1j * g[1] * np.exp(-1j * theta[c] * g[1]))) @ g[2].conj().T
        steps.append({"inverse": _unitary_op(u.conj().T, g[0]), "targets": g[0],
                      "derivative": _pairs(du), "column": c})
    out, jac = _adjoint(lib, {"qubits": n, "steps": steps}, ket, bras, len(gates))
    assert out["ok"], out
    h = 1e-5
    fd = np.zeros_like(jac)
    for c in range(len(gates)):
        up, down = theta.copy(), theta.copy()
        up[c] += h
        down[c] -= h
        fd[:, c] = (expect(forward(up)) - expect(forward(down))) / (2 * h)
    assert np.max(np.abs(jac - fd)) < 1e-8


@pytest.mark.parametrize("step, message", [
    ({"inverse": {"name": "X", "targets": [0]}, "targets": [0],
      "derivative": _pairs(np.eye(2)), "column": 5}, "column"),
    ({"inverse": {"name": "X", "targets": [0]}, "targets": [0, 1],
      "derivative": _pairs(np.eye(2)), "column": 0}, "derivative"),
    ({"inverse": {"name": "NOPE", "targets": [0]}}, "unsupported gate"),
    ({"targets": [0]}, "inverse"),
])
def test_engine_adjoint_refuses_a_bad_step_and_leaves_the_jacobian_alone(lib, step, message):
    ket = np.array([1, 0, 0, 0], complex)
    out, jac = _adjoint(lib, {"qubits": 2, "steps": [step]}, ket, [ket], 1)
    assert not out["ok"]
    assert message in out["error_message"] or message in out.get("error", "")
    if message != "unsupported gate":       # that one is found on the walk, past the zeroing
        assert np.all(jac == 7.0)


def test_engine_adjoint_refuses_buffers_of_the_wrong_size(lib):
    ket = np.array([1, 0, 0], complex)
    out, _ = _adjoint(lib, {"qubits": 2, "steps": []}, ket, [ket], 1)
    assert not out["ok"]
    assert "2^qubits" in out["error_message"]


# ── every gate, against default.qubit ────────────────────────────────────────

GATES = [
    (qml.PauliX, 1, 0), (qml.PauliY, 1, 0), (qml.PauliZ, 1, 0), (qml.Hadamard, 1, 0), (qml.S, 1, 0),
    (qml.T, 1, 0), (qml.SX, 1, 0), (qml.RX, 1, 1), (qml.RY, 1, 1), (qml.RZ, 1, 1),
    (qml.PhaseShift, 1, 1), (qml.Rot, 1, 3), (qml.U1, 1, 1), (qml.U2, 1, 2), (qml.U3, 1, 3),
    (qml.CNOT, 2, 0), (qml.CZ, 2, 0), (qml.CY, 2, 0), (qml.CH, 2, 0), (qml.SWAP, 2, 0),
    (qml.ISWAP, 2, 0), (qml.SISWAP, 2, 0), (qml.ECR, 2, 0), (qml.CRX, 2, 1), (qml.CRY, 2, 1),
    (qml.CRZ, 2, 1), (qml.CRot, 2, 3), (qml.ControlledPhaseShift, 2, 1), (qml.IsingXX, 2, 1),
    (qml.IsingYY, 2, 1), (qml.IsingZZ, 2, 1), (qml.IsingXY, 2, 1), (qml.PSWAP, 2, 1),
    (qml.SingleExcitation, 2, 1), (qml.SingleExcitationPlus, 2, 1),
    (qml.SingleExcitationMinus, 2, 1), (qml.FermionicSWAP, 2, 1), (qml.Toffoli, 3, 0),
    (qml.CSWAP, 3, 0), (qml.CCZ, 3, 0), (qml.DoubleExcitation, 4, 1),
    (qml.DoubleExcitationPlus, 4, 1), (qml.DoubleExcitationMinus, 4, 1),
    (qml.OrbitalRotation, 4, 1), (qml.MultiRZ, 3, 1), (qml.MultiRZ, 5, 1),
]


@pytest.mark.parametrize("cls, k, p", GATES, ids=lambda g: getattr(g, "__name__", None))
def test_gate_vs_default_qubit(device_class, ref, cls, k, p):
    n = 5
    rng = np.random.default_rng(zlib.crc32(f"{cls.__name__}{k}".encode()))
    dev = device_class(wires=n)
    for _ in range(4):
        psi = _random_state(rng, n)
        params = rng.uniform(-3, 3, size=p)
        wires = [int(w) for w in rng.permutation(n)[:k]]

        def circuit():
            qml.StatePrep(psi, wires=range(n))
            cls(*params, wires=wires if k > 1 else wires[0])
            return qml.state()

        got = qml.QNode(circuit, dev)()
        assert np.max(np.abs(got - qml.QNode(circuit, ref)())) < TOL


def test_large_gate_without_a_decomposition_runs_as_its_matrix(device_class):
    rng = np.random.default_rng(5)
    u = _random_unitary(rng, 2 ** 5)

    def circuit():
        qml.Hadamard(2)
        qml.QubitUnitary(u, wires=[4, 0, 2, 1, 5])
        return qml.state()

    dev, ref = device_class(wires=6), qml.device("default.qubit", wires=6)
    assert np.max(np.abs(qml.QNode(circuit, dev)() - qml.QNode(circuit, ref)())) < TOL


def test_symbolic_ops(device_class, ref):
    def circuit(x):
        qml.Hadamard(0)
        qml.adjoint(qml.S(1))
        qml.ctrl(qml.RY, control=[0, 1], control_values=[1, 0])(x, wires=3)
        qml.pow(qml.T(2), 3)
        qml.exp(qml.X(0) @ qml.Z(2), 0.3j)
        qml.GlobalPhase(0.7)
        qml.ctrl(qml.GlobalPhase(0.2), control=1)
        qml.Identity(3)
        return qml.state()

    dev = device_class(wires=4)
    assert np.max(np.abs(qml.QNode(circuit, dev)(0.4) - qml.QNode(circuit, ref)(0.4))) < TOL


# ── what is prepared ─────────────────────────────────────────────────────────


def test_empty_circuit(device_class):
    def circuit():
        return qml.state()

    dev, ref = device_class(wires=3), qml.device("default.qubit", wires=3)
    assert np.array_equal(qml.QNode(circuit, dev)(), qml.QNode(circuit, ref)())


@pytest.mark.parametrize("wires", [[0, 1, 2], [2, 0], [1]])
def test_state_prep_on_some_wires(device_class, wires):
    rng = np.random.default_rng(len(wires))
    psi = _random_state(rng, len(wires))

    def circuit():
        qml.StatePrep(psi, wires=wires)
        return qml.state()

    dev, ref = device_class(wires=3), qml.device("default.qubit", wires=3)
    assert np.max(np.abs(qml.QNode(circuit, dev)() - qml.QNode(circuit, ref)())) < TOL


def test_basis_state_first_and_mid_circuit(device_class, ref):
    def circuit():
        qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
        qml.Hadamard(1)
        qml.CNOT([1, 2])
        return qml.state()

    def late():
        qml.Hadamard(3)
        qml.BasisState(np.array([1, 0]), wires=[0, 1])
        qml.Hadamard(1)
        return qml.state()

    dev = device_class(wires=4)
    full = qml.device("default.qubit", wires=4)
    assert np.max(np.abs(qml.QNode(circuit, dev)() - qml.QNode(circuit, full)())) < TOL
    assert np.max(np.abs(qml.QNode(late, dev)() - qml.QNode(late, full)())) < TOL


def test_wire_labels(device_class):
    labels = ["a", 7, "q2", -1]

    def circuit(x):
        qml.Hadamard("q2")
        qml.CRX(x, wires=["q2", 7])
        qml.SWAP(wires=[7, -1])
        return qml.probs(wires=[-1, "a"]), qml.expval(qml.Z("q2") @ qml.Z(-1))

    dev, ref = device_class(wires=labels), qml.device("default.qubit", wires=labels)
    for a, b in zip(qml.QNode(circuit, dev)(0.8), qml.QNode(circuit, ref)(0.8)):
        assert np.allclose(a, b, atol=TOL)


def test_wires_taken_from_the_circuit(device_class, ref):
    def circuit():
        qml.Hadamard(1)
        qml.CNOT([1, 4])
        return qml.expval(qml.Z(1) @ qml.Z(4)), qml.probs(wires=[4])

    dev = device_class()
    for a, b in zip(qml.QNode(circuit, dev)(), qml.QNode(circuit, ref)()):
        assert np.allclose(a, b, atol=TOL)


# ── what is measured ─────────────────────────────────────────────────────────


def test_analytic_measurements(device_class):
    def circuit(x):
        qml.Hadamard(0)
        qml.CRX(x, wires=[0, 2])
        qml.IsingXY(0.3, wires=[1, 2])
        qml.CNOT([2, 3])
        return (qml.expval(qml.Z(0) @ qml.X(3)), qml.var(qml.Y(1)), qml.probs(wires=[2, 0]),
                qml.expval(qml.Hamiltonian([0.5, -2.0], [qml.Z(0), qml.Y(1) @ qml.X(2)])),
                qml.var(qml.Hermitian(np.array([[1, 0.5j], [-0.5j, 2]]), wires=2)),
                qml.density_matrix(wires=[1]), qml.vn_entropy(wires=[0]),
                qml.mutual_info(wires0=[0], wires1=[3]), qml.purity(wires=[0, 1]),
                qml.expval(qml.Projector([1, 0], wires=[0, 3])))

    dev, ref = device_class(wires=4), qml.device("default.qubit", wires=4)
    for a, b in zip(qml.QNode(circuit, dev)(0.71), qml.QNode(circuit, ref)(0.71)):
        assert np.allclose(a, b, atol=1e-11)


def test_shots_follow_the_born_rule(device_class):
    dev = device_class(wires=2, seed=11)

    @qml.set_shots(40000)
    @qml.qnode(dev)
    def circuit():
        qml.RY(2 * math.acos(math.sqrt(0.2)), wires=0)     # P(|0>) = 0.2
        qml.CNOT([0, 1])
        return qml.counts(), qml.expval(qml.Z(0) @ qml.Z(1)), qml.expval(qml.Z(0))

    counts, zz, z = circuit()
    assert set(counts) == {"00", "11"}
    sigma = math.sqrt(0.2 * 0.8 / 40000)
    assert abs(counts["00"] / 40000 - 0.2) < 5 * sigma
    assert zz == 1.0
    assert abs(z - (0.2 - 0.8)) < 10 * sigma


def test_seed_repeats_samples_and_shot_vectors_partition(device_class):
    def circuit():
        qml.Hadamard(0)
        qml.CNOT([0, 1])
        return qml.sample(wires=[0, 1])

    def draw(seed):
        return qml.set_shots(qml.QNode(circuit, device_class(wires=2, seed=seed)), 200)()

    assert np.array_equal(draw(4), draw(4))
    assert not np.array_equal(draw(4), draw(5))
    a, b = qml.set_shots(qml.QNode(circuit, device_class(wires=2, seed=1)), (50, 150))()
    assert a.shape == (50, 2) and b.shape == (150, 2)


def test_mid_circuit_measurement_is_deferred(device_class):
    def circuit(x):
        qml.RY(x, wires=0)
        m = qml.measure(0)
        qml.cond(m, qml.PauliX)(wires=1)
        return qml.expval(qml.Z(1)), qml.probs(op=m)

    dev, ref = device_class(wires=3), qml.device("default.qubit", wires=3)
    for a, b in zip(qml.QNode(circuit, dev)(0.9),
                    qml.QNode(circuit, ref, mcm_method="deferred")(0.9)):
        assert np.allclose(a, b, atol=TOL)


# ── derivatives ──────────────────────────────────────────────────────────────


def test_best_gradient_is_adjoint_and_matches_parameter_shift(device_class, ref):
    rng = np.random.default_rng(3)
    w = qml.numpy.array(rng.uniform(-1, 1, size=(2, 4, 3)), requires_grad=True)

    def circuit(w):
        qml.StronglyEntanglingLayers(w, wires=range(4))
        return qml.expval(qml.Z(0) @ qml.Z(3))

    dev = device_class(wires=4)
    node = qml.QNode(circuit, dev)
    with qml.Tracker(dev) as tracker:
        got = qml.grad(node)(w)
    assert tracker.totals["executions"] == 1           # one pass, not two per parameter
    assert tracker.totals["derivatives"] == 1
    want = qml.grad(qml.QNode(circuit, ref, diff_method="parameter-shift"))(w)
    assert np.max(np.abs(got - want)) < 1e-10


def test_jacobian_of_several_outputs_and_a_frozen_argument(device_class, ref):
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.RY(y, wires=1)
        qml.CNOT([0, 1])
        qml.CRZ(x, wires=[1, 2])
        return qml.expval(qml.Z(0)), qml.expval(qml.Y(1) @ qml.X(2))

    def jac(node):
        return qml.jacobian(lambda x, y: qml.math.stack(node(x, y)))(x, y)

    x = qml.numpy.array(0.4, requires_grad=True)
    y = qml.numpy.array(0.9, requires_grad=False)
    got = jac(qml.QNode(circuit, device_class(wires=3)))
    want = jac(qml.QNode(circuit, ref, diff_method="parameter-shift"))
    assert np.allclose(got, want, atol=1e-10)


def test_frozen_gate_of_several_parameters_before_a_trainable_one(device_class, ref):
    # The adjoint pipeline leaves a frozen Rot whole, with three parameters to a single op.
    # PennyLane 0.45.1's own adjoint_jacobian counts it as none, so the RX before it reads a
    # gradient of exactly 0; parameter shift and backprop give the value asserted here.
    def circuit(x, y):
        qml.RX(x, wires=0)
        qml.Rot(0.3, 0.8, 1.1, wires=1)
        qml.CNOT([0, 1])
        qml.RY(y, wires=1)
        return qml.expval(qml.Z(1))

    x = qml.numpy.array(0.4, requires_grad=True)
    y = qml.numpy.array(0.9, requires_grad=True)
    got = qml.grad(qml.QNode(circuit, device_class(wires=2), diff_method="adjoint"))(x, y)
    want = qml.grad(qml.QNode(circuit, ref, diff_method="parameter-shift"))(x, y)
    assert abs(want[0]) > 0.1
    assert np.allclose(got, want, atol=1e-10)


@pytest.mark.parametrize("seed", range(6))
def test_adjoint_on_random_circuits_vs_parameter_shift(device_class, ref, seed):
    # Gates with generators, multi-parameter gates (trainable, so decomposed), frozen gates of
    # every size, a repeated argument, and a Hamiltonian with terms on several wires.
    rng = np.random.default_rng(seed)
    n = 4
    picks = rng.integers(0, 7, size=14)
    wires = [[int(w) for w in rng.permutation(n)[:3]] for _ in picks]
    frozen = rng.uniform(-2, 2, size=(14, 3))
    H = qml.Hamiltonian([0.7, -0.4, 1.1], [qml.Z(0) @ qml.Y(2), qml.X(1), qml.Z(3) @ qml.X(0)])

    def circuit(t):
        qml.Hadamard(0)
        for i, (p, w) in enumerate(zip(picks, wires)):
            x = t[i % t.shape[0]]
            if p == 0:
                qml.RX(x, wires=w[0])
            elif p == 1:
                qml.IsingXY(x, wires=w[:2])
            elif p == 2:
                qml.Rot(x, 0.4 * x, -x, wires=w[0])
            elif p == 3:
                qml.CRY(x, wires=w[:2])
            elif p == 4:
                qml.Rot(*frozen[i], wires=w[0])
            elif p == 5:
                qml.CNOT(wires=w[:2])
            else:
                qml.SingleExcitation(x, wires=w[:2])
        return qml.expval(H)

    t = qml.numpy.array(rng.uniform(-1, 1, size=5), requires_grad=True)
    dev = device_class(wires=n)
    got = qml.grad(qml.QNode(circuit, dev, diff_method="adjoint"))(t)
    want = qml.grad(qml.QNode(circuit, qml.device("default.qubit", wires=n),
                              diff_method="backprop"))(t)
    assert np.max(np.abs(got - want)) < 1e-10


def test_parameter_shift_on_the_device(device_class, ref):
    def circuit(x):
        qml.RX(x, wires=0)
        qml.IsingZZ(x, wires=[0, 1])
        qml.RY(0.2, wires=1)
        return qml.expval(qml.X(1))

    x = qml.numpy.array(0.6, requires_grad=True)
    dev = device_class(wires=2)
    got = qml.grad(qml.QNode(circuit, dev, diff_method="parameter-shift"))(x)
    assert abs(got - qml.grad(qml.QNode(circuit, ref))(x)) < 1e-10


def test_finite_shots_fall_back_to_parameter_shift(device_class):
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.Z(0))

    x = qml.numpy.array(0.6, requires_grad=True)
    dev = device_class(wires=1, seed=2)
    node = qml.set_shots(qml.QNode(circuit, dev), 40000)
    assert abs(qml.grad(node)(x) + math.sin(0.6)) < 0.03


def test_backprop_is_refused(device_class):
    def circuit(x):
        qml.RX(x, wires=0)
        return qml.expval(qml.Z(0))

    with pytest.raises(qml.exceptions.QuantumFunctionError, match="backprop"):
        qml.QNode(circuit, device_class(wires=1), diff_method="backprop")(0.1)


# ── what is refused ──────────────────────────────────────────────────────────


def test_a_noise_channel_is_refused(device_class):
    def circuit():
        qml.DepolarizingChannel(0.1, wires=0)
        return qml.expval(qml.Z(0))

    with pytest.raises(DeviceError, match="DepolarizingChannel"):
        qml.QNode(circuit, device_class(wires=1))()


def test_too_many_qubits_is_refused(device_class):
    def circuit():
        for w in range(26):
            qml.Hadamard(w)
        return qml.expval(qml.Z(0))

    with pytest.raises(DeviceError, match="26"):
        qml.QNode(circuit, device_class(wires=26))()


def test_a_wire_off_the_device_is_refused(device_class):
    def circuit():
        qml.Hadamard(5)
        return qml.expval(qml.Z(5))

    with pytest.raises(WireError):
        qml.QNode(circuit, device_class(wires=2))()


def test_postselection_and_other_mcm_methods_are_refused(device_class):
    def circuit():
        qml.Hadamard(0)
        qml.measure(0, postselect=1)
        return qml.expval(qml.Z(0))

    with pytest.raises(ValueError, match="[Pp]ostselect"):
        qml.QNode(circuit, device_class(wires=2))()
    with pytest.raises(DeviceError, match="deferred"):
        qml.QNode(circuit, device_class(wires=2), mcm_method="tree-traversal")()


# ── the entry point and the old path ─────────────────────────────────────────


def test_former_path_is_the_same_class(device_class):
    from quantx_engine.contrib.pennylane_plugin import QuantXDevice
    assert QuantXDevice is device_class


def test_entry_point_targets_the_device(device_class):
    # Read from the source of truth: the package's metadata may be of another install.
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert ('"quantx.statevector" = "quantx_engine.pennylane_device:StatevectorDevice"'
            in text)
