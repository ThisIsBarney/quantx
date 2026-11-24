"""M14: the examples of CUDA-Q's documentation on the "quantx" target, against qpp-cpu.

Each is the circuit CUDA-Q's documentation gives for it, run on both targets: Bernstein-Vazirani,
phase estimation, Grover's search, QAOA for MaxCut, the variational eigensolver on the deuteron
and with the library's UCCSD kernel, and a spin chain evolved by Trotter steps of exp_pauli. The
last conserves the magnetisation term by term only after fusion (an exp_pauli of XX alone does
not), and at 40 qubits, past the state vector, the target keeps it exactly at a bond cap.
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


def same_state(kernel, *args):
    got, want = state("quantx", kernel, *args), state(REFERENCE, kernel, *args)
    return np.max(np.abs(got - want))


# ── Bernstein-Vazirani ───────────────────────────────────────────────────────


@cudaq.kernel
def bernstein_vazirani(secret: list[int]):
    n = len(secret)
    q = cudaq.qvector(n)
    aux = cudaq.qubit()
    x(aux)
    h(aux)
    h(q)
    for i in range(n):
        if secret[i] == 1:
            x.ctrl(q[i], aux)
    h(q)
    mz(q)


def test_bernstein_vazirani_finds_the_secret():
    secret = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    cudaq.set_random_seed(1)
    counts = on("quantx", cudaq.sample, bernstein_vazirani, secret, shots_count=50)
    assert {k: v for k, v in counts.items()} == {"".join(map(str, secret)): 50}


# ── phase estimation ─────────────────────────────────────────────────────────


@cudaq.kernel
def phase_estimation(n: int, phase: float):
    # The eigenvalue exp(2 pi i phase) of R1 on |1>, read into n counting qubits.
    counting = cudaq.qvector(n)
    target = cudaq.qubit()
    x(target)
    h(counting)
    for i in range(n):
        for _ in range(2 ** i):
            r1.ctrl(2.0 * math.pi * phase, counting[i], target)
    # the inverse quantum Fourier transform
    for i in range(n // 2):
        swap(counting[i], counting[n - i - 1])
    for i in range(n):
        for j in range(i):
            r1.ctrl(-math.pi / 2 ** (i - j), counting[j], counting[i])
        h(counting[i])


def test_phase_estimation():
    assert same_state(phase_estimation, 4, 0.3125) < TOL      # 5/16: exact in 4 bits
    assert same_state(phase_estimation, 4, 0.3) < TOL         # not


# ── Grover ───────────────────────────────────────────────────────────────────


@cudaq.kernel
def grover(marked: list[int], rounds: int):
    n = len(marked)
    q = cudaq.qvector(n)
    h(q)
    for _ in range(rounds):
        # oracle: a phase on |marked>
        for i in range(n):
            if marked[i] == 0:
                x(q[i])
        z.ctrl(q.front(n - 1), q[n - 1])
        for i in range(n):
            if marked[i] == 0:
                x(q[i])
        # diffusion
        h(q)
        x(q)
        z.ctrl(q.front(n - 1), q[n - 1])
        x(q)
        h(q)


def test_grover():
    marked = [1, 0, 1, 1]
    assert same_state(grover, marked, 3) < TOL
    p = np.abs(state("quantx", grover, marked, 3)) ** 2
    index = sum(b << k for k, b in enumerate(marked))       # qubit k is bit k
    assert p[index] > 0.95


# ── QAOA for MaxCut ──────────────────────────────────────────────────────────

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (3, 4)]


@cudaq.kernel
def qaoa(n: int, us: list[int], vs: list[int], gammas: list[float], betas: list[float]):
    q = cudaq.qvector(n)
    h(q)
    for layer in range(len(gammas)):
        for e in range(len(us)):
            x.ctrl(q[us[e]], q[vs[e]])
            rz(2.0 * gammas[layer], q[vs[e]])
            x.ctrl(q[us[e]], q[vs[e]])
        for i in range(n):
            rx(2.0 * betas[layer], q[i])


def maxcut(edges):
    h = 0.0 * spin.i(0)
    for u, v in edges:
        h += 0.5 * spin.z(u) * spin.z(v) - 0.5 * spin.i(u)
    return h


def test_qaoa_maxcut_energy():
    us, vs = [u for u, _ in EDGES], [v for _, v in EDGES]
    gammas, betas = [0.4, 0.8, 1.1], [0.7, 0.35, 0.2]
    args = (5, us, vs, gammas, betas)
    got = on("quantx", cudaq.observe, qaoa, maxcut(EDGES), *args).expectation()
    want = on(REFERENCE, cudaq.observe, qaoa, maxcut(EDGES), *args).expectation()
    assert abs(got - want) < TOL


# ── variational eigensolvers ─────────────────────────────────────────────────

DEUTERON = (5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(0) * spin.y(1)
            + 0.21829 * spin.z(0) - 6.125 * spin.z(1))


@cudaq.kernel
def deuteron_ansatz(theta: float):
    q = cudaq.qvector(2)
    x(q[0])
    ry(theta, q[1])
    x.ctrl(q[1], q[0])


def test_vqe_on_the_deuteron():
    for theta in (-0.3, 0.59, 1.7):
        got = on("quantx", cudaq.observe, deuteron_ansatz, DEUTERON, theta).expectation()
        want = on(REFERENCE, cudaq.observe, deuteron_ansatz, DEUTERON, theta).expectation()
        assert abs(got - want) < TOL
    # the documented minimum, near theta = 0.59: -1.748
    assert abs(on("quantx", cudaq.observe, deuteron_ansatz, DEUTERON, 0.594).expectation()
               + 1.7487) < 1e-3


@cudaq.kernel
def uccsd_ansatz(thetas: list[float]):
    q = cudaq.qvector(4)
    x(q[0])
    x(q[1])
    cudaq.kernels.uccsd(q, thetas, 2, 4)


def test_uccsd_library_kernel():
    count = cudaq.kernels.uccsd_num_parameters(2, 4)
    thetas = list(np.linspace(-0.4, 0.6, count))
    assert same_state(uccsd_ansatz, thetas) < TOL
    h = 0.0 * spin.i(0)
    for i in range(3):
        h += spin.x(i) * spin.x(i + 1) + spin.y(i) * spin.y(i + 1) + 0.7 * spin.z(i) * spin.z(i + 1)
    got = on("quantx", cudaq.observe, uccsd_ansatz, h, thetas).expectation()
    want = on(REFERENCE, cudaq.observe, uccsd_ansatz, h, thetas).expectation()
    assert abs(got - want) < TOL


# ── a spin chain by Trotter steps of exp_pauli ───────────────────────────────


def heisenberg_words(n, delta):
    words, coeffs = [], []
    for i in range(n - 1):
        for p, c in (("X", 1.0), ("Y", 1.0), ("Z", delta)):
            words.append(cudaq.pauli_word("I" * i + p + p + "I" * (n - i - 2)))
            coeffs.append(c)
    return words, coeffs


@cudaq.kernel
def trotter(n: int, steps: int, dt: float, words: list[cudaq.pauli_word], coeffs: list[float]):
    q = cudaq.qvector(n)
    for i in range(0, n, 2):
        x(q[i])
    for _ in range(steps):
        for k in range(len(words)):
            exp_pauli(-coeffs[k] * dt, q, words[k])


def magnetisation(n):
    m = spin.z(0)
    for i in range(1, n):
        m += spin.z(i)
    return m


def charge_reports(capfd, options, run):
    options.setenv("QUANTX_VERBOSE", "1")
    capfd.readouterr()
    run()
    return re.findall(r"the charge (used|not used)", capfd.readouterr().err)


def test_trotterised_chain_vs_state_vector(capfd, options):
    n = 10
    words, coeffs = heisenberg_words(n, 0.6)
    args = (n, 4, 0.1, words, coeffs)
    want = state(REFERENCE, trotter, *args)
    got = None

    def run():
        nonlocal got
        got = state("quantx", trotter, *args)

    assert charge_reports(capfd, options, run) == ["used"]   # XX then YY: one conserving gate
    assert np.max(np.abs(got - want)) < TOL


def test_trotterised_chain_keeps_its_magnetisation_past_the_state_vector(capfd, options):
    n = 40
    words, coeffs = heisenberg_words(n, 0.6)
    options.setenv("QUANTX_MAX_BOND", "24")
    value = None

    def run():
        nonlocal value
        value = on("quantx", cudaq.observe, trotter, magnetisation(n),
                   n, 6, 0.15, words, coeffs).expectation()

    assert charge_reports(capfd, options, run) == ["used"]
    assert abs(value) < 1e-10        # the Neel start has zero magnetisation, at any truncation
