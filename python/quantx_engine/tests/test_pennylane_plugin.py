"""Tests for the PennyLane QuantX device plugin."""
from __future__ import annotations

import math
import pytest

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import pennylane as qml
    _HAS_PENNYLANE = True
except ImportError:
    _HAS_PENNYLANE = False

pytestmark = pytest.mark.skipif(
    not (_HAS_NUMPY and _HAS_PENNYLANE),
    reason="numpy and pennylane required",
)

from quantx_engine.pennylane_plugin import QuantXDevice, QuantXPluginError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dev(n: int) -> "QuantXDevice":
    """Create a QuantXDevice with *n* wires directly (no entry-point needed)."""
    return QuantXDevice(wires=n)


# ---------------------------------------------------------------------------
# Basic device creation
# ---------------------------------------------------------------------------


def test_device_creation():
    dev = _dev(2)
    assert dev is not None


def test_device_wires():
    dev = _dev(4)
    assert len(dev.wires) == 4


# ---------------------------------------------------------------------------
# Single-qubit expectation values
# ---------------------------------------------------------------------------


def test_pauli_z_on_zero():
    """⟨Z⟩ on |0⟩ = +1."""
    dev = _dev(1)

    @qml.qnode(dev)
    def circuit():
        return qml.expval(qml.PauliZ(0))

    assert abs(float(circuit()) - 1.0) < 1e-10


def test_pauli_z_after_x():
    """⟨Z⟩ on |1⟩ = -1."""
    dev = _dev(1)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=0)
        return qml.expval(qml.PauliZ(0))

    assert abs(float(circuit()) + 1.0) < 1e-10


def test_hadamard_creates_superposition():
    """⟨Z⟩ on H|0⟩ = 0."""
    dev = _dev(1)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    assert abs(float(circuit())) < 1e-10


def test_rx_rotation():
    """⟨Z⟩ after RX(θ) = cos(θ)."""
    dev = _dev(1)

    @qml.qnode(dev)
    def circuit(theta):
        qml.RX(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    theta = 0.7
    assert abs(float(circuit(theta)) - math.cos(theta)) < 1e-10


def test_ry_rotation():
    """⟨Z⟩ after RY(θ) = cos(θ)."""
    dev = _dev(1)

    @qml.qnode(dev)
    def circuit(theta):
        qml.RY(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    theta = 1.2
    assert abs(float(circuit(theta)) - math.cos(theta)) < 1e-10


def test_rz_phase_gate():
    """⟨X⟩ on RZ(θ)|+⟩ = cos(θ)."""
    dev = _dev(1)

    @qml.qnode(dev)
    def circuit(theta):
        qml.Hadamard(wires=0)
        qml.RZ(theta, wires=0)
        return qml.expval(qml.PauliX(0))

    theta = 0.9
    assert abs(float(circuit(theta)) - math.cos(theta)) < 1e-10


# ---------------------------------------------------------------------------
# Two-qubit gates
# ---------------------------------------------------------------------------


def test_cnot_entanglement():
    """CNOT(|+⟩⊗|0⟩) creates Bell state; ⟨Z₀Z₁⟩ = 1."""
    dev = _dev(2)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    assert abs(float(circuit()) - 1.0) < 1e-10


def test_bell_state_z0_expectation():
    """In Bell state (|00⟩+|11⟩)/√2, ⟨Z₀⟩ = 0."""
    dev = _dev(2)

    @qml.qnode(dev)
    def circuit():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    assert abs(float(circuit())) < 1e-10


def test_cz_gate():
    """CZ flips phase of |11⟩; ⟨Z₀Z₁⟩ for the resulting state is +1."""
    dev = _dev(2)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=0)
        qml.PauliX(wires=1)
        qml.CZ(wires=[0, 1])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    # CZ|11⟩ = -|11⟩; ⟨Z₀⟩ = -1, ⟨Z₁⟩ = -1 → ⟨Z₀Z₁⟩ = +1
    assert abs(float(circuit()) - 1.0) < 1e-10


def test_swap_gate():
    """SWAP(|10⟩) = |01⟩; qubit 1 ends in |1⟩ → ⟨Z₁⟩ = -1."""
    dev = _dev(2)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=0)   # |10⟩
        qml.SWAP(wires=[0, 1])  # → |01⟩
        return qml.expval(qml.PauliZ(1))

    assert abs(float(circuit()) + 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------


def test_two_qubit_state_probs():
    """Product state |10⟩: P(|10⟩) = 1."""
    dev = _dev(2)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=0)
        return qml.probs(wires=[0, 1])

    probs = circuit()
    # Wire 0 flipped → |10⟩ in wire-0,wire-1 order → binary index 2
    assert abs(float(probs[2]) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Multi-qubit circuits
# ---------------------------------------------------------------------------


def test_three_qubit_ghz():
    """GHZ state (|000⟩+|111⟩)/√2 on 3 qubits.

    * ⟨Z₀Z₁Z₂⟩ = 0 because |000⟩ contributes +1 and |111⟩ contributes
      (−1)³ = −1, so the average is 0.
    * ⟨Z₀Z₁⟩ = 1 because both basis states give (±1)(±1) = +1.
    """
    dev = _dev(3)

    @qml.qnode(dev)
    def circuit_zzz():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2))

    assert abs(float(circuit_zzz()) - 0.0) < 1e-10

    @qml.qnode(dev)
    def circuit_zz():
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    assert abs(float(circuit_zz()) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# State property
# ---------------------------------------------------------------------------


def test_state_property():
    """qml.state() returns a length-2 statevector for 1-qubit circuit."""
    dev = _dev(1)

    @qml.qnode(dev)
    def circuit():
        return qml.state()

    sv = circuit()
    assert sv.shape == (2,)
    assert abs(float(abs(sv[0]) ** 2) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_stub_without_pennylane():
    """QuantXPluginError is a subclass of RuntimeError."""
    assert issubclass(QuantXPluginError, RuntimeError)


def test_probs_reversed_wire_order():
    """probs(wires=[1,0]) should be in [1,0] index order, not [0,1]."""
    dev = QuantXDevice(wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.PauliX(wires=0)   # |10⟩
        return qml.probs(wires=[1, 0])  # reversed: inner bit is wire-0

    probs = circuit()
    # With wires=[1,0]: outcome "01" (wire1=0, wire0=1) → index 1 → prob=1
    assert abs(float(probs[1]) - 1.0) < 1e-10, f"Expected probs[1]=1, got {probs}"
