"""M16 gate: charge-conserving qubit circuits, fused from OpenQASM and run on the native engine.

Both layers are checked against a state-vector simulator written here, independently of the
module (its own gate matrices, its own qubit order):

* the fusion (no native library): random circuits of transpiled patterns -- an XY hop
  written as an XX and a YY rotation, neither of which conserves the number of |1>s alone,
  with a phase between them that makes it asymmetric; a ZZ phase conjugated by X gates;
  SWAPs; lone phases -- are read by ``from_qasm`` and fused
  into one- and two-qubit gates. The fused circuit must give the source circuit's state, at
  the end and at every position ``measure_points`` maps.
* the engine (native): the fused circuit runs on the block-sparse MPS. At a bond large
  enough to be exact, <Z_i> and the norm must match the state vector to 1e-10; truncated
  hard, the state loses weight but every state it can hold has sum_i <Z_i> = n - 2 * ones.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from quantx_engine import u1_circuit as uc

TOL = 1e-10

# ── an independent state vector: axis i of a (2,)*n tensor is qubit i ─────────────────────
_H = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
_X = np.array([[0, 1], [1, 0]])
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def _rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def _rx(t):
    return np.array([[math.cos(t / 2), -1j * math.sin(t / 2)],
                     [-1j * math.sin(t / 2), math.cos(t / 2)]])


_MATS = {"h": lambda: _H, "x": lambda: _X, "rz": _rz, "rx": _rx,
         "cx": lambda: _CX, "swap": lambda: _SWAP}


def apply(psi, u, qs):
    """``u`` (its first qubit the high bit) on qubits ``qs`` of the tensor ``psi``."""
    k = len(qs)
    out = np.tensordot(u.reshape((2,) * (2 * k)), psi, axes=(list(range(k, 2 * k)), qs))
    return np.moveaxis(out, list(range(k)), qs)


def basis(bits):
    psi = np.zeros((2,) * len(bits), dtype=complex)
    psi[tuple(bits)] = 1
    return psi


def z_of(psi):
    p = np.abs(psi) ** 2
    n = psi.ndim
    return np.array([(p.sum(axis=tuple(j for j in range(n) if j != i)) * [1, -1]).sum()
                     for i in range(n)])


def run_source(n, ops, upto=None):
    """The source circuit, op by op from |0...0>."""
    psi = basis([0] * n)
    for name, par, qs in ops[:upto]:
        psi = apply(psi, _MATS[name](*par), qs)
    return psi


def run_fused(circ, upto=None):
    psi = basis(circ.initial)
    for g in circ.gates[:upto]:
        psi = apply(psi, g.u, [g.q] if g.arity == 1 else [g.q, g.q + 1])
    return psi


def same_state(a, b):
    """Equal up to a global phase."""
    return abs(abs(np.vdot(a, b)) - 1) < TOL and abs(np.vdot(a, a) - 1) < TOL


# ── random circuits of transpiled patterns ────────────────────────────────────────────────

def hop(a, b, t, s=0.0):
    """exp(-i t/2 YY) D exp(-i t/2 XX), each rotation through CX RZ CX.

    With D = 1 it is the XY hop exp(-i t/2 (XX + YY)). With D = exp(-i s/2 (Z_a - Z_b)) it
    still conserves the number of |1>s, but is neither symmetric under exchanging its qubits
    nor equal to its transpose: a gate put on the wrong qubit order, or sent transposed,
    then changes the state.
    """
    xx = [("h", (), [a]), ("h", (), [b]), ("cx", (), [a, b]), ("rz", (t,), [b]),
          ("cx", (), [a, b]), ("h", (), [a]), ("h", (), [b])]
    twist = [("rz", (s,), [a]), ("rz", (-s,), [b])] if s else []
    yy = [("rx", (math.pi / 2,), [a]), ("rx", (math.pi / 2,), [b]), ("cx", (), [a, b]),
          ("rz", (t,), [b]), ("cx", (), [a, b]), ("rx", (-math.pi / 2,), [a]),
          ("rx", (-math.pi / 2,), [b])]
    return xx + twist + yy


def zz_flipped(a, b, t):
    """X_a exp(-i t/2 ZZ) X_a: a ZZ phase of the opposite sign."""
    return [("x", (), [a]), ("cx", (), [a, b]), ("rz", (t,), [b]), ("cx", (), [a, b]),
            ("x", (), [a])]


def random_circuit(n, n_patterns, seed, prepare=True):
    rng = np.random.default_rng(seed)
    ops = [("x", (), [q]) for q in range(n) if prepare and rng.random() < 0.5]
    bounds = [len(ops)]                       # source positions between patterns
    for _ in range(n_patterns):
        q = int(rng.integers(n - 1))
        a, b = (q, q + 1) if rng.random() < 0.5 else (q + 1, q)
        t = float(rng.uniform(-math.pi, math.pi))
        # not an X U X first: its X would read as one of the preparation X gates
        kind = rng.integers(4) if len(bounds) > 1 else 0
        s = float(rng.uniform(-math.pi, math.pi))
        ops += (hop(a, b, t, s) if kind == 0 else zz_flipped(a, b, t) if kind == 1
                else [("swap", (), [a, b])] if kind == 2 else [("rz", (t,), [a])])
        bounds.append(len(ops))
    return ops, bounds


def qasm(n, ops):
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];", f"creg c[{n}];"]
    for name, par, qs in ops:
        if name == "rx":          # angles of pi/2 as an expression, as transpilers write them
            p = "(pi/2)" if par[0] > 0 else "(-pi/2)"
        else:
            p = f"({par[0]!r})" if par else ""
        lines.append(f"{name}{p} " + ",".join(f"q[{q}]" for q in qs) + ";")
    return "\n".join(lines) + "\n"


@pytest.mark.parametrize("seed", range(6))
def test_fused_circuit_gives_the_source_state(seed):
    n = 6
    ops, bounds = random_circuit(n, 30, seed)
    circ = uc.from_qasm(qasm(n, ops))
    assert circ.n_qubits == n and circ.source_gates == len(ops)
    assert all(g.arity == 1 or g.u.shape == (4, 4) for g in circ.gates)
    assert same_state(run_fused(circ), run_source(n, ops))
    # every pattern boundary maps to a fused position holding the same state
    for c, k in zip(bounds, uc.measure_points(circ, bounds)):
        assert same_state(run_fused(circ, k), run_source(n, ops, c)), c


def test_fused_gates_conserve_the_charge_exactly():
    ops, _ = random_circuit(6, 40, 7)
    circ = uc.from_qasm(qasm(6, ops))
    for g in circ.gates:
        ones = np.array([bin(i).count("1") for i in range(g.u.shape[0])])
        assert np.all(g.u[ones[:, None] != ones[None, :]] == 0)
        assert np.allclose(g.u.conj().T @ g.u, np.eye(g.u.shape[0]), atol=TOL)


def test_preparation_x_gates_become_the_bitstring():
    # interleaved with other gates, the way a transpiler may emit them
    ops = [("x", (), [0]), ("swap", (), [0, 1]), ("x", (), [3]), ("rz", (0.3,), [1]),
           ("x", (), [4])] + hop(3, 4, 0.7) + hop(2, 3, 1.1)
    circ = uc.from_qasm(qasm(5, ops))
    assert circ.initial == (1, 0, 0, 1, 1) and circ.ones == 3 and circ.initial_end == 5
    assert same_state(run_fused(circ), run_source(5, ops))


def test_an_xux_cluster_on_an_untouched_qubit_still_fuses():
    # The X opening zz_flipped acts on a qubit nothing has touched: taking it into the
    # bitstring would leave the closing X alone, so only the leading X gates are taken.
    ops = [("x", (), [0])] + hop(0, 1, 0.9) + zz_flipped(2, 1, 0.4)
    circ = uc.from_qasm(qasm(3, ops))
    assert circ.initial == (1, 0, 0) and circ.initial_end == 1
    assert same_state(run_fused(circ), run_source(3, ops))


def test_an_xux_cluster_right_after_the_preparation_is_refused():
    # Its opening X directly follows the preparation X gates: both readings take it as
    # preparation, so the circuit is refused -- never fused into a different circuit.
    ops = [("x", (), [0])] + zz_flipped(2, 1, 0.4) + hop(0, 1, 0.9)
    with pytest.raises(uc.U1CircuitError, match="2 X gates that open the circuit"):
        uc.from_qasm(qasm(3, ops))


def test_a_position_inside_a_fused_gate_is_refused():
    ops = [("x", (), [0])] + hop(0, 1, 0.5)
    circ = uc.from_qasm(qasm(2, ops))
    assert uc.measure_points(circ, [1, len(ops)]) == [0, len(circ.gates)]
    with pytest.raises(uc.U1CircuitError, match="splits a fused gate"):
        uc.measure_points(circ, [4])           # after h h cx
    with pytest.raises(uc.U1CircuitError, match="before the X gates"):
        uc.measure_points(circ, [0])
    with pytest.raises(uc.U1CircuitError, match="outside"):
        uc.measure_points(circ, [len(ops) + 1])


@pytest.mark.parametrize("text, match", [
    ("qreg q[3];\nh q[0];\ncx q[0],q[2];\nh q[0];\n", "not one qubit or two adjacent"),
    ("qreg q[2];\nh q[0];\n", "do not conserve"),
    ("qreg q[2];\nh q[0];\ncx q[0],q[1];\n", "do not conserve"),
    ("qreg q[2];\nmeasure q[0] -> c[0];\n", "must be unitary"),
    ("qreg q[2];\nccx q[0],q[1],q[0];\n", "unsupported gate"),
    ("qreg q[2];\nrz(__import__('os')) q[0];\n", "cannot read"),
    ("qreg q[2];\nrz(os.system) q[0];\n", "unsupported gate parameter"),
    ("qreg q[2];\nrz(9**9) q[0];\n", "unsupported gate parameter"),
    ("qreg q[2];\nrz(1/0) q[0];\n", "unsupported gate parameter"),
    ("qreg q[2];\nrz q[0];\n", "takes 1 parameter"),
    ("qreg q[2];\ncx q[0],q[0];\n", "same qubit twice"),
    ("qreg q[2];\nh q[2];\n", "past the register"),
    ("qreg q[2];\nh r[0];\n", "other than the qreg"),
    ("h q[0];\nqreg q[2];\n", "before the qreg"),
    ("OPENQASM 2.0;\n", "no qreg"),
])
def test_unfusable_or_unreadable_circuits_are_refused(text, match):
    with pytest.raises(uc.U1CircuitError, match=match):
        uc.from_qasm(text)


def test_gate_parameters_are_arithmetic_on_pi():
    circ = uc.from_qasm("qreg q[1];\nrz(-pi/4) q[0];\nrz(2*pi/3 + 0.5) q[0];\nu1(1e-3) q[0];\n")
    phases = [np.angle(g.u[1, 1] / g.u[0, 0]) for g in circ.gates]
    assert np.allclose(phases, [-math.pi / 4, 2 * math.pi / 3 + 0.5, 1e-3])


def test_from_gates_refuses_a_gate_that_changes_the_charge():
    with pytest.raises(uc.U1CircuitError, match="does not conserve"):
        uc.from_gates(2, [(0, _CX)])
    with pytest.raises(uc.U1CircuitError, match="one bit"):
        uc.from_gates(2, [], initial=[1])


# ── the engine ────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    unavailable = pytest.fail if os.environ.get("QX_REQUIRE_NATIVE") == "1" else pytest.skip
    try:
        from quantx_engine.engine import Engine
        eng = Engine()
    except Exception as exc:  # QuantXNativeUnavailable, or a missing dependency
        unavailable(f"native library unavailable: {exc}")
    if not hasattr(eng._lib, "qx_u1_circuit"):
        unavailable("native library has no qx_u1_circuit")
    return eng


@pytest.mark.parametrize("seed", range(3))
def test_engine_matches_the_state_vector(engine, seed):
    n = 8
    ops, bounds = random_circuit(n, 40, 100 + seed)
    circ = uc.from_qasm(qasm(n, ops))
    srcs = bounds[::8] + [bounds[-1]]
    when = uc.measure_points(circ, srcs)
    source_of = dict(zip(when, srcs))
    r = uc.run(circ, max_bond=64, measure_after=when, engine=engine)
    assert r.ones == circ.ones and [m.after for m in r.measurements] == sorted(source_of)
    for m in r.measurements:
        assert np.allclose(m.z, z_of(run_source(n, ops, source_of[m.after])), atol=TOL, rtol=0)
        assert abs(m.norm2 - 1) < TOL


def test_engine_runs_generic_gates(engine):
    # Each two-qubit gate a phase on |00> and on |11> and a random 2x2 unitary on
    # {|01>, |10>}: no symmetry under transposing it or exchanging its qubits to hide behind.
    n, rng = 7, np.random.default_rng(5)
    gates, psi = [], basis([1, 0, 0, 1, 1, 0, 1])
    for _ in range(50):
        if rng.random() < 0.25:
            q = int(rng.integers(n))
            u = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, 2)))
            psi = apply(psi, u, [q])
        else:
            q = int(rng.integers(n - 1))
            m = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            u = np.diag(np.exp(1j * rng.uniform(0, 2 * np.pi, 4)))
            u[1:3, 1:3] = np.linalg.qr(m)[0]
            psi = apply(psi, u, [q, q + 1])
        gates.append((q, u))
    circ = uc.from_gates(n, gates, initial=[1, 0, 0, 1, 1, 0, 1])
    r = uc.run(circ, max_bond=64, engine=engine)
    assert np.allclose(r.measurements[-1].z, z_of(psi), atol=TOL, rtol=0)
    assert r.ones == 4


def test_truncation_keeps_the_charge(engine):
    n = 8
    ops, _ = random_circuit(n, 60, 11)
    circ = uc.from_qasm(qasm(n, ops))
    r = uc.run(circ, max_bond=2, engine=engine)
    m = r.measurements[-1]
    assert r.max_bond <= 2 and r.truncation_sum > 1e-6 and m.norm2 < 1
    assert r.ones == circ.ones and abs(m.z.sum() - (n - 2 * circ.ones)) < 1e-12


def test_engine_refuses_bad_input(engine):
    circ = uc.from_gates(4, [(1, np.eye(4))])
    with pytest.raises(uc.U1CircuitError, match="measure_after"):
        uc.run(circ, measure_after=[2], engine=engine)
    with pytest.raises(uc.U1CircuitError, match="max_bond"):
        uc.run(circ, max_bond=0, engine=engine)
    with pytest.raises(uc.U1CircuitError, match="integers"):
        uc.run(circ, max_bond=2.5, engine=engine)
