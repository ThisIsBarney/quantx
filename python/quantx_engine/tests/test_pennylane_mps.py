"""M15 gate: PennyLane circuits on the engine's matrix-product state.

Two references, for two layers:

* the engine: the ``qx_mps`` qubit operations against a state vector written here (its own
  einsum, its own qubit order), with and without the charge, so that nothing shared with
  PennyLane can hide a mistake in the kernels.
* the device: gates, measurements and shots against ``default.qubit`` at an exact bond, and
  what only an MPS can do against answers known in closed form, past where a state vector
  reaches. PennyLane's own device suite runs in the native CI job.
"""
from __future__ import annotations

import json
import math
import os
import zlib
from ctypes import POINTER, c_char_p, c_double, c_longlong, c_ubyte, c_ulonglong

import numpy as np
import pytest

qml = pytest.importorskip("pennylane", minversion="0.45")

from pennylane.exceptions import DeviceError  # noqa: E402

TOL = 1e-11


@pytest.fixture(scope="module")
def device_class():
    try:
        from quantx_engine.pennylane_device import MPSDevice
        MPSDevice(wires=1)
    except Exception as e:      # CI's native job sets QX_REQUIRE_NATIVE=1: there it must load
        unavailable = pytest.fail if os.environ.get("QX_REQUIRE_NATIVE") == "1" else pytest.skip
        unavailable(f"native engine unavailable: {e}")
    return MPSDevice


@pytest.fixture(scope="module")
def lib(device_class):
    from quantx_engine.engine import Engine
    lib = Engine()._lib
    lib.qx_mps.argtypes = [c_char_p]
    lib.qx_mps.restype = c_char_p
    lib.qx_mps_amplitudes.argtypes = [c_longlong, POINTER(c_double), c_ulonglong]
    lib.qx_mps_amplitudes.restype = c_char_p
    lib.qx_mps_sample.argtypes = [c_longlong, c_ulonglong, c_ulonglong, POINTER(c_ubyte)]
    lib.qx_mps_sample.restype = c_char_p
    return lib


def _ask(lib, **request):
    return json.loads(lib.qx_mps(json.dumps(request).encode()))


def _ok(lib, **request):
    out = _ask(lib, **request)
    assert out["ok"], out
    return out


def _pairs(m):
    m = np.asarray(m, dtype=np.complex128).ravel()
    return np.column_stack((m.real, m.imag)).ravel().tolist()


def _amplitudes(lib, handle, n):
    a = np.empty(2 ** n, dtype=np.complex128)
    out = json.loads(lib.qx_mps_amplitudes(handle, a.ctypes.data_as(POINTER(c_double)), a.size))
    assert out["ok"], out
    return a


def _random_unitary(rng, dim):
    q, r = np.linalg.qr(rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _random_conserving(rng):
    """A two-qubit gate that conserves the number of |1>s: phases on |00>, |11>, and a block."""
    u = np.zeros((4, 4), complex)
    u[0, 0], u[3, 3] = np.exp(1j * rng.uniform(-3, 3, size=2))
    u[np.ix_([1, 2], [1, 2])] = _random_unitary(rng, 2)
    return u


def _apply(state, u, qubits, n):
    """u on ``qubits`` (qubits[0] its high bit) of a state whose site 0 is the index's high bit."""
    k = len(qubits)
    t = np.tensordot(u.reshape((2,) * (2 * k)), state.reshape((2,) * n),
                     axes=(list(range(k, 2 * k)), list(qubits)))
    return np.moveaxis(t, list(range(k)), list(qubits)).ravel()


PAULI = {"X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]]),
         "Z": np.diag([1.0 + 0j, -1.0])}


def _pauli_value(state, term, n):
    s = state
    for site, letter in term:
        s = _apply(s, PAULI[letter], [site], n)
    return float(np.vdot(state, s).real)


def _basis(bits):
    index = int("".join(str(b) for b in bits), 2)
    s = np.zeros(2 ** len(bits), complex)
    s[index] = 1
    return s


# ── the engine, against a state vector written here ──────────────────────────


def test_engine_gates_vs_state_vector(lib):
    # Any gate on any pair: adjacent, far apart and in either order, on a state with no charges.
    rng = np.random.default_rng(1)
    n = 7
    for _ in range(4):
        bits = [int(b) for b in rng.integers(0, 2, size=n)]
        handle = _ok(lib, op="qubits", n_qubits=n, bits=bits)["handle"]
        want = _basis(bits)
        gates = []
        for _ in range(30):
            if rng.random() < 0.3:
                q, u = [int(rng.integers(n))], _random_unitary(rng, 2)
            else:
                q, u = [int(w) for w in rng.permutation(n)[:2]], _random_unitary(rng, 4)
            gates.append({"q": q, "u": _pairs(u)})
            want = _apply(want, u, q, n)
        _ok(lib, op="gates", handle=handle, gates=gates, max_bond=64, cutoff=0.0)
        got = _amplitudes(lib, handle, n)
        _ok(lib, op="free", handle=handle)
        assert np.max(np.abs(got - want)) < TOL


def test_engine_charge_conserving_gates_vs_state_vector(lib):
    rng = np.random.default_rng(2)
    n = 8
    bits = [1, 0] * (n // 2)
    handle = _ok(lib, op="qubits", n_qubits=n, bits=bits, conserve=True)["handle"]
    want = _basis(bits)
    gates = []
    for _ in range(40):
        if rng.random() < 0.2:
            q, u = [int(rng.integers(n))], np.diag(np.exp(1j * rng.uniform(-3, 3, size=2)))
        else:
            q, u = [int(w) for w in rng.permutation(n)[:2]], _random_conserving(rng)
        gates.append({"q": q, "u": _pairs(u)})
        want = _apply(want, u, q, n)
    _ok(lib, op="gates", handle=handle, gates=gates, max_bond=64, cutoff=0.0)
    assert np.max(np.abs(_amplitudes(lib, handle, n) - want)) < TOL
    info = _ok(lib, op="info", handle=handle)
    assert (n - info["charge"]) // 2 == sum(bits)         # the charge counts +1 per |0>, -1 per |1>
    _ok(lib, op="free", handle=handle)


@pytest.mark.parametrize("conserve", [False, True])
def test_engine_pauli_strings_vs_state_vector(lib, conserve):
    # X and Y move a charged state's bra and ket apart; the strings must still come out right.
    rng = np.random.default_rng(3 + conserve)
    n = 7
    bits = [1, 0, 1, 0, 1, 0, 0]
    handle = _ok(lib, op="qubits", n_qubits=n, bits=bits, conserve=conserve)["handle"]
    want = _basis(bits)
    gates = []
    for _ in range(25):
        q = sorted(int(w) for w in rng.permutation(n)[:2])
        u = _random_conserving(rng) if conserve else _random_unitary(rng, 4)
        gates.append({"q": q, "u": _pairs(u)})
        want = _apply(want, u, q, n)
    _ok(lib, op="gates", handle=handle, gates=gates, max_bond=64, cutoff=0.0)
    terms = [[]]
    for _ in range(40):
        k = int(rng.integers(1, 5))
        sites = sorted(int(w) for w in rng.permutation(n)[:k])
        terms.append([[s, "XYZ"[int(rng.integers(3))]] for s in sites])
    values = _ok(lib, op="pauli", handle=handle, terms=terms)["values"]
    _ok(lib, op="free", handle=handle)
    assert abs(values[0] - 1) < TOL                     # the empty string is the identity
    for term, value in zip(terms, values):
        assert abs(value - _pauli_value(want, term, n)) < TOL


@pytest.mark.parametrize("conserve", [False, True])
def test_engine_sampling_follows_the_born_rule(lib, conserve):
    rng = np.random.default_rng(5)
    n = 4
    bits = [1, 0, 1, 0]
    handle = _ok(lib, op="qubits", n_qubits=n, bits=bits, conserve=conserve)["handle"]
    want = _basis(bits)
    gates = []
    for _ in range(10):
        q = [int(w) for w in rng.permutation(n)[:2]]
        u = _random_conserving(rng)
        gates.append({"q": q, "u": _pairs(u)})
        want = _apply(want, u, q, n)
    _ok(lib, op="gates", handle=handle, gates=gates, max_bond=16, cutoff=0.0)
    shots = 40000
    s = np.empty((shots, n), dtype=np.uint8)
    out = json.loads(lib.qx_mps_sample(handle, shots, 9, s.ctypes.data_as(POINTER(c_ubyte))))
    assert out["ok"], out
    again = np.empty((shots, n), dtype=np.uint8)
    lib.qx_mps_sample(handle, shots, 9, again.ctypes.data_as(POINTER(c_ubyte)))
    other = np.empty((shots, n), dtype=np.uint8)
    lib.qx_mps_sample(handle, shots, 10, other.ctypes.data_as(POINTER(c_ubyte)))
    _ok(lib, op="free", handle=handle)
    assert np.array_equal(s, again)
    assert not np.array_equal(s, other)
    assert np.all(np.sum(s, axis=1) == sum(bits))       # these gates conserve it either way
    index = s.astype(np.int64) @ (2 ** np.arange(n - 1, -1, -1))
    freq = np.bincount(index, minlength=2 ** n) / shots
    p = np.abs(want) ** 2
    sigma = np.sqrt(np.maximum(p * (1 - p), 1e-12) / shots)
    assert np.max(np.abs(freq - p) / np.maximum(sigma, 1e-3)) < 5


def test_engine_truncation_is_reported_and_the_cap_holds(lib):
    rng = np.random.default_rng(6)
    n = 10
    handle = _ok(lib, op="qubits", n_qubits=n)["handle"]
    gates = []
    for layer in range(10):
        for w in range(layer % 2, n - 1, 2):
            gates.append({"q": [w, w + 1], "u": _pairs(_random_unitary(rng, 4))})
    r = _ok(lib, op="gates", handle=handle, gates=gates, max_bond=6, cutoff=0.0)
    assert r["max_bond"] == 6
    assert 0 < r["truncation_max"] <= r["truncation_sum"]
    info = _ok(lib, op="info", handle=handle)
    assert max(info["bonds"]) == 6
    assert info["norm2"] < 1
    # An expectation divides by the norm truncation left: the identity is still 1.
    assert abs(_ok(lib, op="pauli", handle=handle, terms=[[]])["values"][0] - 1) < TOL
    _ok(lib, op="free", handle=handle)


def test_engine_routed_gates_count_every_swap(lib):
    # A long-range gate on a chain at a tight bond truncates at each swap, not only at the
    # gate: the reported sum must exceed the largest single discard.
    rng = np.random.default_rng(7)
    n = 8
    handle = _ok(lib, op="qubits", n_qubits=n)["handle"]
    gates = [{"q": [w, w + 1], "u": _pairs(_random_unitary(rng, 4))} for w in range(n - 1)]
    gates += [{"q": [0, n - 1], "u": _pairs(_random_unitary(rng, 4))}]
    r = _ok(lib, op="gates", handle=handle, gates=gates, max_bond=4, cutoff=0.0)
    _ok(lib, op="free", handle=handle)
    assert r["truncation_sum"] > r["truncation_max"] > 0


@pytest.mark.parametrize("request_, message", [
    ({"op": "qubits", "n_qubits": 0}, "at least 1"),
    ({"op": "qubits", "n_qubits": 2, "bits": [0]}, "one bit per qubit"),
    ({"op": "qubits", "n_qubits": 2, "bits": [0, 2]}, "0 or 1"),
    ({"op": "qubits", "n_qubits": 2, "nope": 1}, "unknown key"),
    ({"op": "gates", "handle": 999999, "gates": []}, "no live state"),
])
def test_engine_refuses_bad_requests(lib, request_, message):
    out = _ask(lib, **request_)
    assert not out["ok"]
    assert message in out["error"]


@pytest.mark.parametrize("gate, message", [
    ({"q": [0], "u": _pairs(2 * np.eye(2))}, "not unitary"),
    ({"q": [0, 0], "u": _pairs(np.eye(4))}, "repeated"),
    ({"q": [0, 5], "u": _pairs(np.eye(4))}, "out of range"),
    ({"q": [0], "u": _pairs(np.eye(4))}, "matrix"),
    ({"q": [0, 1, 2], "u": []}, "one qubit or two"),
])
def test_engine_refuses_a_bad_gate_and_leaves_the_state_alone(lib, gate, message):
    handle = _ok(lib, op="qubits", n_qubits=3, bits=[1, 0, 0])["handle"]
    x = np.array([[0, 1], [1, 0]])
    out = _ask(lib, op="gates", handle=handle, gates=[{"q": [1], "u": _pairs(x)}, gate])
    assert not out["ok"]
    assert message in out["error"]
    assert np.array_equal(_amplitudes(lib, handle, 3), _basis([1, 0, 0]))   # the X did not run
    _ok(lib, op="free", handle=handle)


def test_engine_refuses_a_gate_that_changes_a_conserved_charge(lib):
    handle = _ok(lib, op="qubits", n_qubits=2, conserve=True)["handle"]
    out = _ask(lib, op="gates", handle=handle,
               gates=[{"q": [0], "u": _pairs([[0, 1], [1, 0]])}])
    assert not out["ok"]
    assert "does not conserve" in out["error"]
    _ok(lib, op="free", handle=handle)


def test_engine_drops_what_a_conserving_gate_has_between_charges(lib):
    # The check passes a gate as conserving with charge-changing entries up to 1e-10, and
    # RX(2 pi) has 1e-16 of them. They join column groups of different charges, so of
    # different widths, and must be dropped: the state is then, bit for bit, the one the
    # same gate with them zeroed makes.
    rng = np.random.default_rng(8)
    n = 6
    bits = [1, 0] * (n // 2)
    handle = _ok(lib, op="qubits", n_qubits=n, bits=bits, conserve=True)["handle"]
    want = _basis(bits)
    gates = []
    for _ in range(20):
        q, u = [int(w) for w in rng.permutation(n)[:2]], _random_conserving(rng)
        gates.append({"q": q, "u": _pairs(u)})
        want = _apply(want, u, q, n)
    _ok(lib, op="gates", handle=handle, gates=gates, max_bond=64, cutoff=0.0)
    other = _ok(lib, op="copy", handle=handle)["handle"]
    rx = qml.RX.compute_matrix(2 * np.pi)
    assert 0 < abs(rx[0, 1]) < 1e-10
    for h, u in ((handle, rx), (other, np.diag(np.diag(rx)))):
        _ok(lib, op="gates", handle=h, gates=[{"q": [w], "u": _pairs(u)} for w in range(n)],
            max_bond=64, cutoff=0.0)
    got, exact = _amplitudes(lib, handle, n), _amplitudes(lib, other, n)
    _ok(lib, op="free", handle=handle)
    _ok(lib, op="free", handle=other)
    assert np.array_equal(got, exact)
    assert np.max(np.abs(got - want)) < TOL           # (-1)^6: the tail is the identity


@pytest.mark.parametrize("terms, message", [
    ([[[9, "Z"]]], "site 9"),
    ([[[1, "Z"], [0, "X"]]], "increase"),
    ([[[0, "Q"]]], "letters"),
])
def test_engine_refuses_a_bad_pauli_string(lib, terms, message):
    handle = _ok(lib, op="qubits", n_qubits=2)["handle"]
    out = _ask(lib, op="pauli", handle=handle, terms=terms)
    _ok(lib, op="free", handle=handle)
    assert not out["ok"]
    assert message in out["error"]


def test_engine_refuses_a_buffer_of_the_wrong_size(lib):
    handle = _ok(lib, op="qubits", n_qubits=2)["handle"]
    a = np.empty(3, dtype=np.complex128)
    out = json.loads(lib.qx_mps_amplitudes(handle, a.ctypes.data_as(POINTER(c_double)), 3))
    _ok(lib, op="free", handle=handle)
    assert not out["ok"]
    assert "2^n" in out["error"]


def test_engine_a_circuit_state_is_not_a_model_state(lib):
    # Operations on model states must refuse a qubit state, not read it as something else.
    handle = _ok(lib, op="qubits", n_qubits=4)["handle"]
    model = {"type": "xxz", "n_sites": 4, "J": 1.0, "delta": 1.0}
    out = _ask(lib, op="energy", handle=handle, model=model)
    _ok(lib, op="free", handle=handle)
    assert not out["ok"]


# ── the device, against default.qubit ────────────────────────────────────────

GATES = [
    (qml.PauliX, 1, 0), (qml.PauliY, 1, 0), (qml.PauliZ, 1, 0), (qml.Hadamard, 1, 0), (qml.S, 1, 0),
    (qml.T, 1, 0), (qml.SX, 1, 0), (qml.RX, 1, 1), (qml.RY, 1, 1), (qml.RZ, 1, 1),
    (qml.PhaseShift, 1, 1), (qml.Rot, 1, 3), (qml.U3, 1, 3), (qml.CNOT, 2, 0), (qml.CZ, 2, 0),
    (qml.CY, 2, 0), (qml.SWAP, 2, 0), (qml.ISWAP, 2, 0), (qml.CRX, 2, 1), (qml.CRot, 2, 3),
    (qml.IsingXX, 2, 1), (qml.IsingXY, 2, 1), (qml.IsingZZ, 2, 1), (qml.PSWAP, 2, 1),
    (qml.SingleExcitation, 2, 1), (qml.FermionicSWAP, 2, 1), (qml.Toffoli, 3, 0),
    (qml.CSWAP, 3, 0), (qml.DoubleExcitation, 4, 1), (qml.DoubleExcitationPlus, 4, 1),
    (qml.OrbitalRotation, 4, 1), (qml.MultiRZ, 4, 1),
]


@pytest.mark.parametrize("cls, k, p", GATES, ids=lambda g: getattr(g, "__name__", None))
def test_gate_vs_default_qubit(device_class, cls, k, p):
    n = 5
    rng = np.random.default_rng(zlib.crc32(f"{cls.__name__}{k}".encode()))
    psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    psi /= np.linalg.norm(psi)
    params = rng.uniform(-3, 3, size=p)
    wires = [int(w) for w in rng.permutation(n)[:k]]

    def circuit():
        qml.StatePrep(psi, wires=range(n))
        cls(*params, wires=wires if k > 1 else wires[0])
        return qml.state()

    got = qml.QNode(circuit, device_class(wires=n))()
    want = qml.QNode(circuit, qml.device("default.qubit", wires=n))()
    assert np.max(np.abs(got - want)) < 1e-10


def test_measurements_vs_default_qubit(device_class):
    def circuit(x):
        qml.Hadamard(0)
        qml.CRX(x, wires=[0, 2])
        qml.IsingXY(0.3, wires=[1, 2])
        qml.CNOT([2, 3])
        qml.GlobalPhase(0.4)
        return (qml.expval(qml.Z(0) @ qml.X(3)), qml.var(qml.Y(1)), qml.probs(wires=[2, 0]),
                qml.expval(qml.Hamiltonian([0.5, -2.0], [qml.Z(0), qml.Y(1) @ qml.X(2)])),
                qml.var(qml.Hamiltonian([0.5, -2.0], [qml.Z(0), qml.Y(1) @ qml.X(2)])),
                qml.var(qml.Hermitian(np.array([[1, 0.5j], [-0.5j, 2]]), wires=2)),
                qml.density_matrix(wires=[1]), qml.vn_entropy(wires=[0]),
                qml.mutual_info(wires0=[0], wires1=[3]), qml.purity(wires=[0, 1]),
                qml.expval(qml.Projector([1, 0], wires=[0, 3])), qml.state())

    dev, ref = device_class(wires=4), qml.device("default.qubit", wires=4)
    for a, b in zip(qml.QNode(circuit, dev)(0.71), qml.QNode(circuit, ref)(0.71)):
        assert np.allclose(a, b, atol=TOL)


def test_wire_labels_and_wires_from_the_circuit(device_class):
    labels = ["a", 7, "q2", -1]

    def circuit(x):
        qml.Hadamard("q2")
        qml.CRX(x, wires=["q2", 7])
        qml.SWAP(wires=[7, -1])
        return qml.probs(wires=[-1, "a"]), qml.expval(qml.Z("q2") @ qml.Z(-1))

    ref = qml.device("default.qubit", wires=labels)
    for dev in (device_class(wires=labels), device_class()):
        for a, b in zip(qml.QNode(circuit, dev)(0.8), qml.QNode(circuit, ref)(0.8)):
            assert np.allclose(a, b, atol=TOL)


def test_the_chain_is_the_device_order_not_the_order_of_first_use(device_class):
    # A CNOT ladder along the device's chain stays at bond 2 whatever order it is written in.
    # Numbered by first use, the wires would land in another order and the bond would grow.
    n = 12
    chain = list(np.random.default_rng(1).permutation(n))
    order = [int(w) for w in chain]

    def circuit():
        qml.Hadamard(order[0])
        for i in reversed(range(n - 1)):            # last link first: first use is scrambled
            qml.CNOT(wires=[order[i], order[i + 1]])
        return qml.expval(qml.Z(order[0]))

    dev = device_class(wires=order)
    with qml.Tracker(dev) as tracker:
        qml.QNode(circuit, dev)()
    assert tracker.history["max_bond"][-1] == 2


def test_unused_wires_cost_nothing(device_class):
    # A wire the circuit never touches is not simulated: the amplitudes' limit is on what is used.
    def circuit():
        qml.Hadamard(40)
        qml.CNOT([40, 7])
        return qml.density_matrix(wires=[7])

    rho = qml.QNode(circuit, device_class(wires=60))()
    assert np.max(np.abs(rho - np.eye(2) / 2)) < TOL


def test_basis_state_and_state_prep_on_some_wires(device_class):
    rng = np.random.default_rng(8)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)

    def circuit():
        qml.BasisState(np.array([1, 0]), wires=[3, 1])
        qml.StatePrep(psi, wires=[2, 0])
        qml.CNOT([0, 3])
        return qml.state()

    dev, ref = device_class(wires=4), qml.device("default.qubit", wires=4)
    assert np.max(np.abs(qml.QNode(circuit, dev)() - qml.QNode(circuit, ref)())) < TOL


def test_empty_circuit_and_a_measurement_on_no_wires(device_class):
    def empty():
        return qml.state()

    def nothing():
        return qml.expval(qml.s_prod(2.0, qml.Identity()))

    assert np.array_equal(qml.QNode(empty, device_class(wires=2))(), [1, 0, 0, 0])
    assert qml.QNode(nothing, device_class())() == 2


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


def test_expectation_of_a_mid_circuit_measurement(device_class):
    # Its measurement process has a value in place of an observable.
    def circuit(x):
        qml.RY(x, wires=0)
        m = qml.measure(0)
        return qml.expval(m), qml.var(m)

    dev, ref = device_class(wires=2), qml.device("default.qubit", wires=2)
    for a, b in zip(qml.QNode(circuit, dev)(0.9),
                    qml.QNode(circuit, ref, mcm_method="deferred")(0.9)):
        assert np.allclose(a, b, atol=TOL)


def test_parameter_shift_gradient(device_class):
    def circuit(w):
        qml.StronglyEntanglingLayers(w, wires=range(4))
        return qml.expval(qml.Z(0) @ qml.X(3))

    rng = np.random.default_rng(1)
    w = qml.numpy.array(rng.uniform(-1, 1, size=(2, 4, 3)), requires_grad=True)
    got = qml.grad(qml.QNode(circuit, device_class(wires=4)))(w)
    want = qml.grad(qml.QNode(circuit, qml.device("default.qubit", wires=4)))(w)
    assert np.max(np.abs(got - want)) < 1e-9


# ── shots ────────────────────────────────────────────────────────────────────


def test_shots_on_every_observable_kind(device_class):
    # Non-commuting observables split; a Hermitian's eigenvalues must keep their order through
    # the rotation into its eigenbasis (they did not, when it was diagonalized as a transform).
    a = np.array([[1, 2], [2, -4.0]])

    def circuit(x):
        qml.RX(x, wires=0)
        qml.CNOT([0, 1])
        qml.RY(0.4, wires=2)
        return (qml.expval(qml.X(2)), qml.expval(qml.Z(0) @ qml.Z(1)), qml.var(qml.Y(0)),
                qml.expval(qml.Hamiltonian([0.5, 2.0], [qml.Z(0), qml.X(2)])),
                qml.var(qml.Hermitian(a, wires=2)), qml.expval(qml.Hermitian(a, wires=2)),
                qml.probs(wires=[1, 0]))

    n = 200000
    node = qml.set_shots(qml.QNode(circuit, device_class(wires=3, seed=3)), n)
    want = qml.QNode(circuit, qml.device("default.qubit", wires=3))(0.7)
    for got, ref in zip(node(0.7), want):
        assert np.allclose(got, ref, atol=0.03)


def test_samples_on_a_wire_subset_keep_their_order(device_class):
    def circuit():
        qml.PauliX(2)
        qml.CNOT([2, 0])
        return qml.sample(wires=[2, 1, 0])

    s = qml.set_shots(qml.QNode(circuit, device_class(wires=3, seed=1)), 10)()
    assert np.array_equal(s, np.tile([1, 0, 1], (10, 1)))


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


# ── past a state vector ──────────────────────────────────────────────────────


def test_ghz_on_a_hundred_qubits(device_class):
    n = 100

    def ghz():
        qml.Hadamard(0)
        for w in range(n - 1):
            qml.CNOT(wires=[w, w + 1])

    def circuit():
        ghz()
        return (qml.expval(qml.Z(0) @ qml.Z(n - 1)), qml.expval(qml.Z(50)),
                qml.probs(wires=[0, 50, n - 1]),
                qml.expval(qml.prod(*[qml.X(w) for w in range(n)])))

    def sampled():
        ghz()
        return qml.sample(wires=range(n))

    dev = device_class(wires=n)
    with qml.Tracker(dev) as tracker:
        zz, z, p, xs = qml.QNode(circuit, dev)()
    assert tracker.history["max_bond"][-1] == 2
    assert abs(zz - 1) < TOL and abs(z) < TOL and abs(xs - 1) < TOL
    want = np.zeros(8)
    want[0] = want[7] = 0.5
    assert np.max(np.abs(p - want)) < TOL
    s = qml.set_shots(qml.QNode(sampled, device_class(wires=n, seed=2)), 500)()
    assert np.all(s == s[:, :1])
    assert abs(s[:, 0].mean() - 0.5) < 5 * math.sqrt(0.25 / 500)


def test_a_long_range_gate_on_a_product_state_needs_no_bond(device_class):
    # CZ between the ends of 60 qubits, on |+...+>: <X_0 Z_59> = 1 by the stabilizer, whatever
    # the swaps between them do, and they must not cost a truncation.
    n = 60

    def circuit():
        for w in range(n):
            qml.Hadamard(w)
        qml.CZ(wires=[0, n - 1])
        return qml.expval(qml.X(0) @ qml.Z(n - 1)), qml.expval(qml.X(30))

    dev = device_class(wires=n, max_bond=8)
    with qml.Tracker(dev) as tracker:
        stabilizer, x = qml.QNode(circuit, dev)()
    assert abs(stabilizer - 1) < TOL and abs(x - 1) < TOL
    assert tracker.history["truncation_sum"][-1] < 1e-25


def test_truncation_converges_with_the_bond(device_class):
    n = 10

    def circuit():
        for layer in range(8):
            for w in range(layer % 2, n - 1, 2):
                qml.IsingXX(0.7, wires=[w, w + 1])
                qml.CRY(1.1, wires=[w, w + 1])
                qml.RX(0.5 + 0.1 * w, wires=w)
                qml.RZ(0.7, wires=w + 1)
        return qml.expval(qml.Z(4) @ qml.Z(5))

    exact = qml.QNode(circuit, qml.device("default.qubit", wires=n))()
    errors, weights = [], []
    for chi in (4, 8, 16, 32):
        dev = device_class(wires=n, max_bond=chi)
        with qml.Tracker(dev) as tracker:
            errors.append(abs(qml.QNode(circuit, dev)() - exact))
        weights.append(tracker.history["truncation_sum"][-1])
    assert weights[0] > weights[1] > weights[2] > 1e-8 > 1e-25 > weights[3]
    assert errors[0] > errors[1] > errors[2] > 1e-8 > 1e-13 > errors[3]


def test_probabilities_of_few_wires_on_many(device_class):
    # Past the amplitudes' limit they come from Z strings. Independent bits, so they multiply.
    n = 30
    angles = np.linspace(0.2, 2.0, 3)

    def circuit():
        for w, a in zip([3, 11, 25], angles):
            qml.RY(a, wires=w)
        for w in range(n):
            qml.Identity(w)
        qml.CZ(wires=[0, 29])
        return qml.probs(wires=[25, 3, 11])

    p = qml.QNode(circuit, device_class(wires=n))()
    one = [np.array([math.cos(a / 2) ** 2, math.sin(a / 2) ** 2]) for a in angles]
    want = np.kron(np.kron(one[2], one[0]), one[1])
    assert np.max(np.abs(p - want)) < TOL


def test_what_has_no_form_on_an_mps_is_refused_past_the_amplitudes(device_class):
    n = 21

    def state():
        qml.Hadamard(0)
        for w in range(n - 1):
            qml.CNOT(wires=[w, w + 1])
        return qml.state()

    def many_probs():
        for w in range(30):
            qml.Hadamard(w)
        return qml.probs(wires=range(17))

    with pytest.raises(DeviceError, match="20 qubits"):
        qml.QNode(state, device_class(wires=n))()
    with pytest.raises(DeviceError, match="2\\^17"):
        qml.QNode(many_probs, device_class(wires=30))()


# ── the charge ───────────────────────────────────────────────────────────────


def _hopping(n, layers=8):
    def circuit():
        qml.BasisState(np.array([1, 0] * (n // 2)), wires=range(n))
        for layer in range(layers):
            for w in range(layer % 2, n - 1, 2):
                qml.SingleExcitation(0.6, wires=[w, w + 1])
        return [qml.expval(qml.Z(i)) for i in range(n)]
    return circuit


def _conserved(dev, circuit):
    with qml.Tracker(dev) as tracker:
        qml.QNode(circuit, dev)()
    return tracker.history["conserved"][-1]


@pytest.mark.parametrize("build, expected", [
    (lambda: qml.SingleExcitation(0.3, wires=[0, 1]), True),
    (lambda: (qml.PauliX(0), qml.PauliX(2), qml.IsingXY(0.2, wires=[0, 2])), True),
    (lambda: (qml.RZ(0.3, wires=0), qml.CZ([0, 1]), qml.SWAP([1, 2])), True),
    (lambda: (qml.PauliX(0), qml.IsingXY(0.2, wires=[0, 1]), qml.PauliX(0)), False),
    (lambda: qml.RX(0.3, wires=0), False),
    (lambda: qml.RX(2 * np.pi, wires=0), True),
    (lambda: qml.CNOT([0, 1]), False),
], ids=["excitation", "leading-x", "phases", "late-x", "rx", "rx-2pi", "cnot"])
def test_auto_finds_when_the_charge_is_conserved(device_class, build, expected):
    def circuit():
        build()
        return qml.expval(qml.Z(0))

    assert _conserved(device_class(wires=3), circuit) is expected


def test_the_charge_can_be_forced_either_way(device_class):
    def conserving():
        qml.SingleExcitation(0.3, wires=[0, 1])
        return qml.expval(qml.Z(0))

    def not_conserving():
        qml.RX(0.3, wires=0)
        return qml.expval(qml.Z(0))

    assert _conserved(device_class(wires=2, conserve=False), conserving) is False
    assert _conserved(device_class(wires=2, conserve=True), conserving) is True
    with pytest.raises(DeviceError, match="does not conserve"):
        qml.QNode(not_conserving, device_class(wires=2, conserve=True))()


def test_the_charge_is_exact_at_a_hard_truncation(device_class):
    n = 20
    z = np.array(qml.QNode(_hopping(n), device_class(wires=n, max_bond=8))())
    assert abs(z.sum()) < 1e-12                  # as many |0>s as |1>s, at any truncation


def test_the_two_paths_agree_where_both_are_exact(device_class):
    n = 10
    a = qml.QNode(_hopping(n), device_class(wires=n, conserve=True))()
    b = qml.QNode(_hopping(n), device_class(wires=n, conserve=False))()
    c = qml.QNode(_hopping(n), qml.device("default.qubit", wires=n))()
    assert np.max(np.abs(np.array(a) - np.array(c))) < 1e-10
    assert np.max(np.abs(np.array(b) - np.array(c))) < 1e-10


# ── options ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kwargs, message", [
    ({"max_bond": 0}, "max_bond"),
    ({"max_bond": 2.5}, "max_bond"),
    ({"cutoff": -1.0}, "cutoff"),
    ({"conserve": "yes"}, "conserve"),
    ({"svd": "qr"}, "svd"),
])
def test_bad_options_are_refused(device_class, kwargs, message):
    with pytest.raises(DeviceError, match=message):
        device_class(wires=2, **kwargs)


def test_gram_svd_agrees_at_an_exact_bond(device_class):
    n = 6

    def circuit():
        for layer in range(4):
            for w in range(layer % 2, n - 1, 2):
                qml.CRX(0.4 + 0.1 * layer, wires=[w, w + 1])
                qml.RY(0.3, wires=w)
        return qml.expval(qml.Z(2) @ qml.X(3))

    a = qml.QNode(circuit, device_class(wires=n, svd="gram"))()
    b = qml.QNode(circuit, device_class(wires=n, svd="direct"))()
    assert abs(a - b) < 1e-9


def test_entry_point_targets_the_device(device_class):
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[2]
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"quantx.mps" = "quantx_engine.pennylane_device:MPSDevice"' in text


def test_a_freed_state_is_not_kept(device_class, lib):
    # Every run frees what it made, also when a measurement is refused.
    before = _ok(lib, op="live")["handles"]

    def circuit():
        qml.Hadamard(0)
        return qml.expval(qml.Z(0))

    qml.QNode(circuit, device_class(wires=2))()

    def refused():
        qml.Hadamard(0)
        return qml.state()

    with pytest.raises(DeviceError):
        qml.QNode(refused, device_class(wires=21))()
    assert _ok(lib, op="live")["handles"] == before
