"""M15 gate: PennyLane's own demos on ``quantx.statevector`` and ``quantx.mps``.

The circuits and optimisers of the demos at pennylane.ai/qml/demos, in the small: qubit
rotation, variational eigensolver, QAOA, a variational classifier, Grover's search, phase
estimation and the templates a demo builds on. Each runs on both devices and on
``default.qubit`` from the same start, and must reach the same numbers; where a demo has a
known answer it must reach that too. The demo on simulating circuits with tensor networks
runs on ``quantx.mps``, against the state vector where one reaches.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

qml = pytest.importorskip("pennylane", minversion="0.45")
pnp = qml.numpy

TOL = 1e-9


def _load(name):
    try:
        import quantx_engine.pennylane_device as devices
        cls = getattr(devices, name)
        cls(wires=1)
    except Exception as e:      # CI's native job sets QX_REQUIRE_NATIVE=1: there it must load
        unavailable = pytest.fail if os.environ.get("QX_REQUIRE_NATIVE") == "1" else pytest.skip
        unavailable(f"native engine unavailable: {e}")
    return cls


@pytest.fixture(scope="module", params=["StatevectorDevice", "MPSDevice"],
                ids=["statevector", "mps"])
def device(request):
    return _load(request.param)


@pytest.fixture(scope="module")
def mps():
    return _load("MPSDevice")


def _both(device, circuit, wires, **kwargs):
    """The same circuit as a QNode on each device, differentiated by the same method."""
    return (qml.QNode(circuit, device(wires=wires), **kwargs),
            qml.QNode(circuit, qml.device("default.qubit", wires=wires), **kwargs))


def test_qubit_rotation(device):
    # Basic tutorial: qubit rotation. Gradient descent turns |0> to |1>, ending at <Z> = -1.
    @qml.qnode(device(wires=1))
    def circuit(params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)
        return qml.expval(qml.Z(0))

    params = pnp.array([0.011, 0.012], requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    for _ in range(100):
        params = opt.step(circuit, params)
    assert circuit(params) < -0.9999
    assert abs(float(params[0]) - 0.0) < 1e-1 or abs(abs(float(params[1])) - math.pi) < 1e-1


def test_variational_eigensolver_h2(device):
    # A brief overview of VQE: H2 in STO-3G from PennyLane's own quantum chemistry, one
    # DoubleExcitation on the Hartree-Fock state. The exact (FCI) energy at this geometry is
    # the Hamiltonian's lowest eigenvalue.
    symbols = ["H", "H"]
    coordinates = pnp.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]], requires_grad=False)
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
    assert qubits == 4
    exact = float(np.min(np.linalg.eigvalsh(qml.matrix(H, wire_order=range(4)))))
    hf = qml.qchem.hf_state(2, qubits)

    def cost(theta):
        qml.BasisState(hf, wires=range(qubits))
        qml.DoubleExcitation(theta, wires=[0, 1, 2, 3])
        return qml.expval(H)

    node, ref = _both(device, cost, 4)
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = pnp.array(0.0, requires_grad=True)
    theta_ref = pnp.array(0.0, requires_grad=True)
    for _ in range(40):
        theta = opt.step(node, theta)
        theta_ref = opt.step(ref, theta_ref)
    assert abs(float(theta) - float(theta_ref)) < 1e-9
    assert abs(float(node(theta)) - exact) < 1e-6
    assert abs(exact - (-1.13619)) < 1e-5         # the demo's value, at 0.7 Angstrom


def test_qaoa_maxcut(device):
    # Intro to QAOA: MaxCut on a four-cycle, whose optimum cuts all four edges.
    import networkx as nx
    graph = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0)])
    cost_h, mixer_h = qml.qaoa.maxcut(graph)
    wires = 4

    def layer(gamma, alpha):
        qml.qaoa.cost_layer(gamma, cost_h)
        qml.qaoa.mixer_layer(alpha, mixer_h)

    def cost(params):
        for w in range(wires):
            qml.Hadamard(w)
        qml.layer(layer, 2, params[0], params[1])
        return qml.expval(cost_h)

    node, ref = _both(device, cost, wires)
    opt, opt_ref = qml.AdagradOptimizer(stepsize=0.4), qml.AdagradOptimizer(stepsize=0.4)
    params = pnp.array([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)
    params_ref = pnp.array([[0.5, 0.5], [0.5, 0.5]], requires_grad=True)
    for _ in range(30):
        params = opt.step(node, params)
        params_ref = opt_ref.step(ref, params_ref)
    assert np.max(np.abs(params - params_ref)) < 1e-9
    assert float(node(params)) < -3.9        # the cut of four edges, at <C> = -4


def test_variational_classifier(device):
    # Variational classifier: the parity of four bits, on a few steps of Nesterov momentum.
    def layer(w):
        for i in range(4):
            qml.Rot(*w[i], wires=i)
        for i in range(4):
            qml.CNOT(wires=[i, (i + 1) % 4])

    def circuit(weights, x):
        qml.BasisState(x, wires=range(4))
        for w in weights:
            layer(w)
        return qml.expval(qml.Z(0))

    node, ref = _both(device, circuit, 4)
    X = pnp.array([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 0, 1, 1]], requires_grad=False)
    Y = pnp.array([-1.0, 1.0, -1.0, 1.0], requires_grad=False)

    def cost(q, w, bias):
        loss = 0.0
        for x, y in zip(X, Y):
            loss = loss + (y - (q(w, x) + bias)) ** 2
        return loss / len(X)

    rng = np.random.default_rng(0)
    start = pnp.array(0.01 * rng.normal(size=(2, 4, 3)), requires_grad=True)
    results = []
    for q in (node, ref):
        w, bias = start.copy(), pnp.array(0.0, requires_grad=True)
        opt = qml.NesterovMomentumOptimizer(0.5)
        for _ in range(6):
            (w, bias), _ = opt.step_and_cost(lambda a, b: cost(q, a, b), w, bias)
        results.append((w, bias))
    assert np.max(np.abs(results[0][0] - results[1][0])) < TOL
    assert abs(float(results[0][1]) - float(results[1][1])) < TOL


def test_strongly_entangling_gradient(device):
    # Quantum gradients: the same Jacobian as backprop, by whichever method the device has.
    def circuit(w):
        qml.StronglyEntanglingLayers(w, wires=range(4))
        return qml.expval(qml.Z(0) @ qml.X(3))

    rng = np.random.default_rng(1)
    w = pnp.array(rng.uniform(-1, 1, size=(3, 4, 3)), requires_grad=True)
    node, ref = _both(device, circuit, 4)
    assert np.max(np.abs(qml.grad(node)(w) - qml.grad(ref)(w))) < 1e-10


def test_grover_search(device):
    # Grover's search: three qubits, one marked state, two iterations: P(marked) = 0.945.
    wires = list(range(3))
    marked = [1, 0, 1]

    def circuit():
        for w in wires:
            qml.Hadamard(w)
        for _ in range(2):
            qml.FlipSign(marked, wires=wires)
            qml.GroverOperator(wires=wires)
        return qml.probs(wires=wires)

    probs = qml.QNode(circuit, device(wires=3))()
    assert abs(probs[int("101", 2)] - 0.9453125) < 1e-12


def test_quantum_phase_estimation(device):
    # QPE reads the phase of a T gate on |1>, 1/8, exactly in three estimation qubits.
    def circuit():
        qml.PauliX(0)
        qml.QuantumPhaseEstimation(qml.T(0), estimation_wires=[1, 2, 3])
        return qml.probs(wires=[1, 2, 3])

    probs = qml.QNode(circuit, device(wires=4))()
    assert abs(probs[1] - 1.0) < 1e-10


@pytest.mark.parametrize("build", [
    lambda: (qml.AngleEmbedding([0.1, 0.2, 0.3], wires=range(3), rotation="Y"),
             qml.BasicEntanglerLayers([[0.4, 0.5, 0.6]], wires=range(3))),
    lambda: qml.AmplitudeEmbedding(np.arange(1, 9.0), wires=range(3), normalize=True),
    lambda: qml.IQPEmbedding([0.1, 0.2, 0.3], wires=range(3), n_repeats=2),
    lambda: (qml.PauliX(0), qml.QFT(wires=range(3))),
    lambda: qml.RandomLayers(np.array([[0.3, 0.8, 1.2, 0.1]]), wires=range(3), seed=4),
    lambda: qml.TrotterProduct(qml.X(0) @ qml.X(1) + 0.5 * qml.Z(2), time=0.7, n=3, order=2),
    lambda: qml.ArbitraryUnitary(np.arange(63) / 30.0, wires=range(3)),
    lambda: qml.FermionicDoubleExcitation(0.4, wires1=[0, 1], wires2=[2, 3]),
    lambda: qml.UCCSD(np.array([0.2, 0.1, 0.3]), wires=range(4), s_wires=[[0, 1, 2], [1, 2, 3]],
                      d_wires=[[[0, 1], [2, 3]]], init_state=np.array([1, 1, 0, 0])),
    lambda: qml.kUpCCGSD(np.arange(6.0).reshape(1, 6) / 9, wires=range(4), k=1, delta_sz=0,
                         init_state=np.array([1, 1, 0, 0])),
], ids=["embedding", "amplitudes", "iqp", "qft", "random", "trotter", "arbitrary",
        "fermionic", "uccsd", "kupccgsd"])
def test_templates(device, build):
    def circuit():
        build()
        return qml.state()

    def wires_of():
        tape = qml.tape.make_qscript(circuit)()
        return len(tape.operations and tape.wires) or 1

    n = max(wires_of(), 3)
    node, ref = _both(device, circuit, n)
    assert np.max(np.abs(node() - ref())) < 1e-11


def _tensor_network_demo(n, theta=0.5, phi=0.1):
    # The circuit of PennyLane's demo on simulating quantum circuits with tensor networks.
    def circuit():
        for q in range(n - 4):
            qml.X(wires=q)
            qml.RX(theta, wires=q + 1)
            qml.CNOT(wires=[q, q + 1])
            qml.DoubleExcitation(phi, wires=[q, q + 1, q + 3, q + 4])
            qml.CSWAP(wires=[q + 1, q + 3, q + 4])
            qml.RY(theta, wires=q + 1)
            qml.Toffoli(wires=[q + 1, q + 3, q + 4])
        return [qml.expval(qml.Z(0)),
                qml.expval(qml.Hamiltonian([0.1, 0.2], [qml.Z(0), qml.Y(1)])),
                qml.expval(qml.Y(n - 1))]
    return circuit


def test_tensor_network_demo_is_exact_where_a_state_vector_reaches(mps):
    n = 12
    circuit = _tensor_network_demo(n)
    got = qml.QNode(circuit, mps(wires=n))()
    want = qml.QNode(circuit, qml.device("default.qubit", wires=n))()
    assert np.max(np.abs(np.array(got) - np.array(want))) < 1e-10


def test_tensor_network_demo_at_fifty_qubits_converges_with_the_bond(mps):
    # The demo's own size and bond, 50. No state vector reaches it, so the check is that a
    # doubled bond, past what the demo uses, does not move the answer beyond the demo's cap.
    n = 50
    circuit = _tensor_network_demo(n)
    a = np.array(qml.QNode(circuit, mps(wires=n, max_bond=50))())
    b = np.array(qml.QNode(circuit, mps(wires=n, max_bond=100))())
    assert np.max(np.abs(a - b)) < 1e-3
