"""M14 gate: CUDA-Q kernels on the "quantx" target against CUDA-Q's own state vector.

The target is an NVQIR simulator (src/cudaq/quantx_simulator.cpp) that runs a kernel on the
engine's block-sparse MPS. Every check runs the same kernel on "qpp-cpu", CUDA-Q's CPU state
vector, so nothing the target shares with the engine can hide a mistake in it: amplitudes,
exact expectation values and sampled distributions, for circuits that conserve the number of
qubits in |1> (where the target uses it as an exact charge, which QUANTX_VERBOSE reports) and
for ones that do not. What only an MPS reaches is checked against closed forms.

It needs CUDA-Q with the target installed (``cmake -DQX_ENABLE_CUDAQ=ON`` then ``cmake
--install``), so Linux; CI's CUDA-Q job sets QX_REQUIRE_CUDAQ=1, and there a missing target
fails instead of skipping.
"""
from __future__ import annotations

import math
import os
import re

import numpy as np
import pytest

cudaq = pytest.importorskip("cudaq")
from cudaq import spin  # noqa: E402

TOL = 1e-10
REFERENCE = "qpp-cpu"


@pytest.fixture(scope="module", autouse=True)
def installed():
    if not cudaq.has_target("quantx"):
        unavailable = pytest.fail if os.environ.get("QX_REQUIRE_CUDAQ") == "1" else pytest.skip
        unavailable("the CUDA-Q target 'quantx' is not installed")
    yield
    cudaq.reset_target()


@pytest.fixture(autouse=True)
def options(monkeypatch):
    for name in ("QUANTX_MAX_BOND", "QUANTX_CUTOFF", "QUANTX_CONSERVE", "QUANTX_SVD",
                 "QUANTX_VERBOSE"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def on(target, run, *args, **kwargs):
    cudaq.set_target(target)
    try:
        return run(*args, **kwargs)
    finally:
        cudaq.reset_target()


def state(target, kernel, *args):
    return np.array(on(target, cudaq.get_state, kernel, *args), dtype=complex)


def charge_reports(capfd, options, run):
    """Run with QUANTX_VERBOSE=1, and return each state's 'used' or 'not used'."""
    options.setenv("QUANTX_VERBOSE", "1")
    capfd.readouterr()
    run()
    return re.findall(r"the charge (used|not used)", capfd.readouterr().err)


# ── kernels ──────────────────────────────────────────────────────────────────


@cudaq.kernel
def ghz(n: int):
    q = cudaq.qvector(n)
    h(q[0])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])


@cudaq.kernel
def scrambled(n: int, angles: list[float]):
    # Every one- and two-qubit gate CUDA-Q has, on near and far pairs: nothing conserved.
    q = cudaq.qvector(n)
    k = 0
    for layer in range(3):
        for i in range(n):
            rx(angles[k], q[i])
            ry(angles[k + 1], q[i])
            rz(angles[k + 2], q[i])
            k += 3
        for i in range(0, n - 1, 2):
            x.ctrl(q[i], q[i + 1])
        for i in range(1, n - 1, 2):
            z.ctrl(q[i], q[i + 1])
        h(q[0])
        s(q[1])
        t(q[2])
        swap(q[0], q[n - 1])
        u3(angles[k], angles[k + 1], angles[k + 2], q[n - 2])
        r1(angles[k + 3], q[n - 1])
        k += 4
        ry.ctrl(angles[k], q[n - 1], q[1])
        k += 1


@cudaq.kernel
def givens_network(n: int, angles: list[float]):
    # A particle-number-conserving circuit, written as CUDA-Q writes such things: each Givens
    # rotation is CNOT, a controlled Ry and CNOT, none of which conserves the number alone.
    q = cudaq.qvector(n)
    for i in range(0, n, 2):
        x(q[i])
    k = 0
    for layer in range(4):
        for i in range(layer % 2, n - 1, 2):
            x.ctrl(q[i + 1], q[i])
            ry.ctrl(angles[k], q[i], q[i + 1])
            x.ctrl(q[i + 1], q[i])
            rz(angles[k + 1], q[i])
            k += 2
    z.ctrl(q[0], q[n - 1])
    swap(q[1], q[n - 2])


@cudaq.kernel
def exp_pauli_givens(theta: float):
    # The same rotation as exp(i theta (XY - YX)): two exp_paulis, each decomposed by CUDA-Q
    # into basis changes, a CNOT ladder and an Rz.
    q = cudaq.qvector(2)
    x(q[0])
    exp_pauli(theta, q, "XY")
    exp_pauli(-theta, q, "YX")


@cudaq.kernel
def many_controls():
    q = cudaq.qvector(6)
    h(q[0])
    h(q[1])
    h(q[2])
    ry(0.7, q[4])
    x.ctrl([q[0], q[1]], q[3])                 # Toffoli
    z.ctrl([q[0], q[1], q[2]], q[4])           # three controls
    rx.ctrl(0.9, [q[0], q[2], q[3], q[4]], q[5])
    swap.ctrl(q[0], q[4], q[5])                # Fredkin
    x.ctrl([q[1], q[2]], q[0])


@cudaq.kernel
def measured_and_reset() -> int:
    # q0 = a, measured at once; q2 gets a, then b = q1 after the reset (P(1) = sin^2 0.2):
    # the result is a + 2 b + 4 (a xor b).
    q = cudaq.qvector(3)
    h(q[0])
    ry(1.1, q[1])
    if mz(q[0]):
        x(q[2])
    reset(q[1])
    ry(0.4, q[1])
    x.ctrl(q[1], q[2])
    total = 0
    if mz(q[0]):
        total += 1
    if mz(q[1]):
        total += 2
    if mz(q[2]):
        total += 4
    return total


@cudaq.kernel
def late_h() -> int:
    # The state is made, charged, at the measurement the kernel branches on; the H after it
    # cannot keep the charge, so the log is replayed without it.
    q = cudaq.qvector(3)
    x(q[0])
    x.ctrl(q[1], q[0])
    ry.ctrl(0.8, q[0], q[1])
    x.ctrl(q[1], q[0])
    if mz(q[2]):
        x(q[0])
    h(q[2])
    x.ctrl(q[2], q[1])
    total = 0
    if mz(q[0]):
        total += 1
    if mz(q[1]):
        total += 2
    if mz(q[2]):
        total += 4
    return total


@cudaq.kernel
def hea(n: int, angles: list[float]):
    q = cudaq.qvector(n)
    for i in range(n):
        ry(angles[i], q[i])
    for i in range(n - 1):
        x.ctrl(q[i], q[i + 1])
    for i in range(n):
        ry(angles[n + i], q[i])


def scrambled_angles(rng, n):
    return list(rng.uniform(-math.pi, math.pi, size=3 * (3 * n + 5)))


def heisenberg(n):
    h = 0.3 * spin.z(0) - 0.2 * spin.i(0)
    for i in range(n - 1):
        h += spin.x(i) * spin.x(i + 1) + spin.y(i) * spin.y(i + 1) + 0.5 * spin.z(i) * spin.z(i + 1)
    return h


def number(n):
    op = 0.5 * (spin.i(0) - spin.z(0))
    for i in range(1, n):
        op += 0.5 * (spin.i(i) - spin.z(i))
    return op


# ── amplitudes ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_ghz_state(n):
    got = state("quantx", ghz, n)
    assert np.max(np.abs(got - state(REFERENCE, ghz, n))) < TOL


def test_scrambled_circuit_vs_state_vector():
    n = 7
    angles = scrambled_angles(np.random.default_rng(1), n)
    got = state("quantx", scrambled, n, angles)
    assert np.max(np.abs(got - state(REFERENCE, scrambled, n, angles))) < TOL


def test_givens_network_vs_state_vector_and_uses_the_charge(capfd, options):
    rng = np.random.default_rng(2)
    n = 10
    angles = list(rng.uniform(-math.pi, math.pi, size=40))
    want = state(REFERENCE, givens_network, n, angles)
    got = None

    def run():
        nonlocal got
        got = state("quantx", givens_network, n, angles)

    assert charge_reports(capfd, options, run) == ["used"]
    assert np.max(np.abs(got - want)) < TOL


def test_exp_pauli_pairs_fuse_into_a_conserving_gate(capfd, options):
    want = state(REFERENCE, exp_pauli_givens, 0.37)
    got = None

    def run():
        nonlocal got
        got = state("quantx", exp_pauli_givens, 0.37)

    assert charge_reports(capfd, options, run) == ["used"]
    assert np.max(np.abs(got - want)) < TOL


def test_gates_with_many_controls_and_a_controlled_swap():
    got = state("quantx", many_controls)
    assert np.max(np.abs(got - state(REFERENCE, many_controls))) < TOL


def test_custom_two_qubit_operation_in_msb_order():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    u, _ = np.linalg.qr(a)
    cudaq.register_operation("quantx_test_u", u.ravel())

    @cudaq.kernel
    def custom():
        q = cudaq.qvector(3)
        h(q[0])
        ry(0.3, q[2])
        quantx_test_u(q[2], q[0])

    got = state("quantx", custom)
    assert np.max(np.abs(got - state(REFERENCE, custom))) < TOL


@cudaq.kernel
def grows() -> int:
    # Two qubits allocated after a measurement the kernel branches on: every bit agrees with
    # it, so the result is 0 or 15.
    a = cudaq.qvector(2)
    h(a[0])
    x.ctrl(a[0], a[1])
    first = mz(a[0])
    b = cudaq.qvector(2)
    if first:
        x(b[1])
    x.ctrl(a[1], b[0])
    total = 0
    if mz(a[1]):
        total += 1
    if mz(b[0]):
        total += 2
    if mz(b[1]):
        total += 4
    if first:
        total += 8
    return total


def test_qubits_allocated_after_the_first_measurement():
    cudaq.set_random_seed(7)
    results = on("quantx", cudaq.run, grows, shots_count=400)
    assert len(results) == 400
    assert set(results) == {0, 15}
    assert 140 < results.count(0) < 260


# ── expectation values ───────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [4, 9])
def test_observe_heisenberg_vs_state_vector(n):
    rng = np.random.default_rng(10 + n)
    angles = list(rng.uniform(-math.pi, math.pi, size=2 * n))
    h = heisenberg(n)
    got = on("quantx", cudaq.observe, hea, h, n, angles).expectation()
    want = on(REFERENCE, cudaq.observe, hea, h, n, angles).expectation()
    assert abs(got - want) < TOL


def test_exact_observe_of_xx_and_yy_terms_keeps_the_charge(capfd, options):
    # The engine evaluates every term at once. Basis changes for X and Y would put an H on the
    # charged state, and the kernel would be replayed without the charge.
    n = 8
    angles = list(np.random.default_rng(13).uniform(-math.pi, math.pi, size=4 * n))
    h = heisenberg(n)
    want = on(REFERENCE, cudaq.observe, givens_network, h, n, angles).expectation()
    got = None

    def run():
        nonlocal got
        got = on("quantx", cudaq.observe, givens_network, h, n, angles).expectation()

    assert charge_reports(capfd, options, run) == ["used"]
    assert abs(got - want) < TOL


def test_observe_terms_one_by_one():
    n = 5
    angles = list(np.linspace(0.1, 1.2, 2 * n))
    h = heisenberg(n)
    got = on("quantx", cudaq.observe, hea, h, n, angles)
    want = on(REFERENCE, cudaq.observe, hea, h, n, angles)
    for term in h:
        if term.is_identity():
            continue
        assert abs(got.expectation(term) - want.expectation(term)) < TOL


def test_observe_at_shots_matches_within_its_error():
    n = 4
    angles = list(np.linspace(-0.8, 0.9, 2 * n))
    h = heisenberg(n)
    exact = on(REFERENCE, cudaq.observe, hea, h, n, angles).expectation()
    cudaq.set_random_seed(11)
    shots = 20000
    got = on("quantx", cudaq.observe, hea, h, n, angles, shots_count=shots).expectation()
    terms = sum(1 for t in h if not t.is_identity())
    assert abs(got - exact) < 5 * math.sqrt(terms) * 1.5 / math.sqrt(shots)


def test_number_is_exact_on_a_wide_conserving_circuit(capfd, options):
    # Past a state vector: 48 qubits, half of them |1>, at a bond cap that truncates.
    n = 48
    angles = list(np.random.default_rng(4).uniform(-math.pi, math.pi, size=4 * n))
    options.setenv("QUANTX_MAX_BOND", "16")
    value = None

    def run():
        nonlocal value
        value = on("quantx", cudaq.observe, givens_network, number(n), n, angles).expectation()

    assert charge_reports(capfd, options, run) == ["used"]
    assert abs(value - n // 2) < 1e-10        # conserved at any truncation


# ── sampling ─────────────────────────────────────────────────────────────────


def total_variation(a, b, shots):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) / shots


def counts_of(result):
    return {k: v for k, v in result.items()}


def test_sampled_distribution_vs_state_vector():
    n = 5
    angles = scrambled_angles(np.random.default_rng(5), n)
    p = np.abs(state(REFERENCE, scrambled, n, angles)) ** 2
    shots = 40000
    cudaq.set_random_seed(5)
    got = counts_of(on("quantx", cudaq.sample, scrambled, n, angles, shots_count=shots))
    for bits, count in got.items():
        # CUDA-Q's bitstrings put qubit 0 first; its state vector puts qubit 0 on bit 0.
        index = int(bits[::-1], 2)
        sigma = math.sqrt(max(p[index] * (1 - p[index]), 1e-12) / shots)
        assert abs(count / shots - p[index]) < 5 * max(sigma, 1e-3)
    assert sum(got.values()) == shots


def test_ghz_on_a_hundred_qubits():
    cudaq.set_random_seed(6)
    got = counts_of(on("quantx", cudaq.sample, ghz, 100, shots_count=500))
    assert set(got) == {"0" * 100, "1" * 100}
    assert 150 < got["0" * 100] < 350


def test_mid_circuit_measurement_and_reset_against_their_closed_form():
    shots = 6000
    cudaq.set_random_seed(8)
    results = on("quantx", cudaq.run, measured_and_reset, shots_count=shots)
    c2, s2 = math.cos(0.2) ** 2, math.sin(0.2) ** 2
    p = {0: c2 / 2, 5: c2 / 2, 6: s2 / 2, 3: s2 / 2}       # a + 2 b + 4 (a xor b)
    assert set(results) <= set(p)
    for value, prob in p.items():
        sigma = math.sqrt(prob * (1 - prob) / shots)
        assert abs(results.count(value) / shots - prob) < 5 * sigma


def test_seed_repeats_samples():
    # set_random_seed seeds the simulator of the target that is set when it is called.
    def draw(seed):
        cudaq.set_target("quantx")
        try:
            cudaq.set_random_seed(seed)
            return counts_of(cudaq.sample(ghz, 6, shots_count=300))
        finally:
            cudaq.reset_target()

    assert draw(9) == draw(9)
    assert draw(9) != draw(10)


# ── options ──────────────────────────────────────────────────────────────────


def test_conserve_options(capfd, options):
    angles = list(np.linspace(0.2, 1.4, 16))
    options.setenv("QUANTX_CONSERVE", "0")
    assert charge_reports(capfd, options, lambda: state("quantx", givens_network, 4, angles)) \
        == ["not used"]
    options.setenv("QUANTX_CONSERVE", "1")
    with pytest.raises(RuntimeError, match="QUANTX_CONSERVE=1"):
        state("quantx", ghz, 3)
    assert np.max(np.abs(state("quantx", givens_network, 4, angles)
                         - state(REFERENCE, givens_network, 4, angles))) < TOL


def test_a_gate_after_the_charge_was_taken_replays_without_it(capfd, options):
    cudaq.set_random_seed(10)
    results = None

    def run():
        nonlocal results
        results = on("quantx", cudaq.run, late_h, shots_count=50)

    reports = charge_reports(capfd, options, run)
    assert len(reports) == 50 and set(reports) == {"not used"}
    cudaq.set_random_seed(10)
    want = on(REFERENCE, cudaq.run, late_h, shots_count=4000)
    got = on("quantx", cudaq.run, late_h, shots_count=4000)
    for value in set(want) | set(got):
        p = want.count(value) / 4000
        assert abs(got.count(value) / 4000 - p) < 5 * math.sqrt(max(p * (1 - p), 1e-4) / 4000)


def test_bad_options_are_refused(options):
    options.setenv("QUANTX_CONSERVE", "maybe")
    with pytest.raises(RuntimeError, match="QUANTX_CONSERVE"):
        state("quantx", ghz, 2)
    options.setenv("QUANTX_CONSERVE", "auto")
    options.setenv("QUANTX_MAX_BOND", "0")
    with pytest.raises(RuntimeError, match="QUANTX_MAX_BOND"):
        state("quantx", ghz, 2)


def test_truncation_shows_in_a_scrambled_state(options):
    n = 10
    angles = scrambled_angles(np.random.default_rng(12), n)
    exact = state(REFERENCE, scrambled, n, angles)
    options.setenv("QUANTX_MAX_BOND", "2")
    capped = state("quantx", scrambled, n, angles)
    assert abs(np.vdot(exact, capped)) < 0.999   # a bond of 2 cannot hold it
    options.delenv("QUANTX_MAX_BOND")
    assert np.max(np.abs(state("quantx", scrambled, n, angles) - exact)) < TOL
