"""PennyLane device plugin for QuantX.

Usage::

    import pennylane as qml
    from quantx_engine.contrib.pennylane_plugin import QuantXDevice

    dev = QuantXDevice(wires=3)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    print(circuit(0.5))

For proper ``qml.device("quantx.statevector", ...)`` support the package must
be installed so that its entry-point is visible to PennyLane::

    [project.entry-points."pennylane.plugins"]
    "quantx.statevector" = "quantx_engine.contrib.pennylane_plugin:QuantXDevice"

References
----------
.. [1] Bergholm, V. et al. (2022). PennyLane: Automatic differentiation of
       hybrid quantum-classical computations. arXiv:1811.04968.
       https://doi.org/10.48550/arXiv.1811.04968

.. [2] Schuld, M. et al. (2019). Evaluating analytic gradients on quantum
       hardware. *Phys. Rev. A* 99, 032331.
       https://doi.org/10.1103/PhysRevA.99.032331
"""
from __future__ import annotations

__all__ = ["QuantXDevice", "QuantXPluginError"]


class QuantXPluginError(RuntimeError):
    """Raised for unsupported operations in the QuantX PennyLane device."""


try:
    import pennylane as qml
    from pennylane import numpy as pnp  # noqa: F401 – re-exported for callers
    # QubitDevice moved to pennylane.devices in PL ≥ 0.36; fall back gracefully.
    try:
        _QubitDevice = qml.QubitDevice  # type: ignore[attr-defined]
    except AttributeError:
        from pennylane.devices import QubitDevice as _QubitDevice  # type: ignore[assignment]
    _HAS_PENNYLANE = True
except ImportError:
    _HAS_PENNYLANE = False
    _QubitDevice = object  # dummy so the class body can be parsed

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


if _HAS_PENNYLANE and _HAS_NUMPY:
    class QuantXDevice(_QubitDevice):  # type: ignore[valid-type,misc]
        """QuantX statevector device for PennyLane.

        Implements the legacy ``QubitDevice`` interface for maximum
        compatibility across PennyLane ≥ 0.30.
        """

        name = "QuantX Statevector"
        short_name = "quantx.statevector"
        pennylane_requires = ">=0.30.0"
        version = "0.1.0"
        author = "QuantX"

        # Supported operations
        operations = [
            "PauliX", "PauliY", "PauliZ",
            "Hadamard", "S", "T",
            "RX", "RY", "RZ", "PhaseShift",
            "CNOT", "CZ", "SWAP",
            "Rot", "U1", "U2", "U3",
            "Identity",
        ]

        # Supported observables
        observables = [
            "PauliX", "PauliY", "PauliZ",
            "Hadamard", "Identity", "Hermitian",
            # Composite / tensor-product observables (PL ≥ 0.36)
            "Prod", "Sum", "SProd", "LinearCombination",
        ]

        def __init__(self, wires, shots=None, **kwargs):
            super().__init__(wires=wires, shots=shots, **kwargs)
            self._statevector = None

        # ------------------------------------------------------------------
        # Core simulation
        # ------------------------------------------------------------------

        @classmethod
        def capabilities(cls):
            """Declare device capabilities including statevector access."""
            caps = super().capabilities().copy()
            caps.update(returns_state=True)
            return caps

        def apply(self, operations, rotations=None, **kwargs):
            """Apply a list of PennyLane operations to the statevector.

            *rotations* contains the diagonalising gates that PennyLane
            appends when computing expectation values of non-diagonal
            observables (e.g. PauliX, PauliY).  We apply them after the
            main circuit operations so that ``probability()`` is evaluated
            in the correct eigenbasis.
            """
            n = len(self.wires)
            sv = np.zeros(2 ** n, dtype=complex)
            sv[0] = 1.0
            self._statevector = sv

            for op in operations:
                self._apply_operation(op)

            if rotations:
                for op in rotations:
                    self._apply_operation(op)

        def _apply_operation(self, op):
            """Dispatch a single gate to the appropriate kernel."""
            n = len(self.wires)
            name = op.name
            wires = [self.wires.index(w) for w in op.wires]
            params = op.parameters

            if len(wires) == 1:
                mat = self._gate_matrix(name, params)
                self._apply_single_qubit(mat, wires[0], n)
            elif len(wires) == 2:
                mat = self._gate_matrix_2q(name, params)
                self._apply_two_qubit(mat, wires[0], wires[1], n)
            else:
                raise QuantXPluginError(
                    f"Gate {name} on {len(wires)} qubits is not supported"
                )

        # ------------------------------------------------------------------
        # Gate matrices
        # ------------------------------------------------------------------

        def _gate_matrix(self, name: str, params) -> np.ndarray:
            """Return 2×2 unitary for a single-qubit gate."""
            if name in ("PauliX", "X"):
                return np.array([[0, 1], [1, 0]], dtype=complex)
            elif name in ("PauliY", "Y"):
                return np.array([[0, -1j], [1j, 0]], dtype=complex)
            elif name in ("PauliZ", "Z"):
                return np.diag([1.0 + 0j, -1.0 + 0j])
            elif name in ("Hadamard", "H"):
                return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            elif name == "S":
                return np.diag([1.0 + 0j, 1j])
            elif name == "T":
                return np.diag([1.0 + 0j, np.exp(1j * np.pi / 4)])
            elif name == "Identity":
                return np.eye(2, dtype=complex)
            elif name == "RX":
                theta = float(params[0])
                c, s = np.cos(theta / 2), np.sin(theta / 2)
                return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)
            elif name == "RY":
                theta = float(params[0])
                c, s = np.cos(theta / 2), np.sin(theta / 2)
                return np.array([[c, -s], [s, c]], dtype=complex)
            elif name == "RZ":
                theta = float(params[0])
                return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])
            elif name in ("PhaseShift", "U1"):
                phi = float(params[0])
                return np.diag([1.0 + 0j, np.exp(1j * phi)])
            elif name == "Rot":
                phi, theta, omega = (
                    float(params[0]), float(params[1]), float(params[2])
                )
                rz_phi = np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)])
                ry = np.array(
                    [
                        [np.cos(theta / 2), -np.sin(theta / 2)],
                        [np.sin(theta / 2), np.cos(theta / 2)],
                    ],
                    dtype=complex,
                )
                rz_omega = np.diag(
                    [np.exp(-1j * omega / 2), np.exp(1j * omega / 2)]
                )
                return rz_phi @ ry @ rz_omega
            elif name == "U2":
                phi, lam = float(params[0]), float(params[1])
                return (
                    np.array(
                        [
                            [1, -np.exp(1j * lam)],
                            [np.exp(1j * phi), np.exp(1j * (phi + lam))],
                        ],
                        dtype=complex,
                    )
                    / np.sqrt(2)
                )
            elif name == "U3":
                theta, phi, lam = (
                    float(params[0]), float(params[1]), float(params[2])
                )
                return np.array(
                    [
                        [
                            np.cos(theta / 2),
                            -np.exp(1j * lam) * np.sin(theta / 2),
                        ],
                        [
                            np.exp(1j * phi) * np.sin(theta / 2),
                            np.exp(1j * (phi + lam)) * np.cos(theta / 2),
                        ],
                    ],
                    dtype=complex,
                )
            else:
                raise QuantXPluginError(f"Unsupported single-qubit gate: {name}")

        def _gate_matrix_2q(self, name: str, params) -> np.ndarray:
            """Return 4×4 unitary for a two-qubit gate."""
            if name == "CNOT":
                return np.array(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                    dtype=complex,
                )
            elif name == "CZ":
                return np.diag([1.0 + 0j, 1, 1, -1])
            elif name == "SWAP":
                return np.array(
                    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                    dtype=complex,
                )
            else:
                raise QuantXPluginError(f"Unsupported two-qubit gate: {name}")

        # ------------------------------------------------------------------
        # Statevector application kernels
        # ------------------------------------------------------------------

        def _apply_single_qubit(self, mat: np.ndarray, qubit: int, n: int):
            """Apply 2×2 matrix to *qubit* in the statevector."""
            sv = self._statevector.reshape([2] * n)
            sv = np.tensordot(mat, sv, axes=[[1], [qubit]])
            # tensordot places the contracted axis at position 0; restore it.
            sv = np.moveaxis(sv, 0, qubit)
            self._statevector = sv.ravel()

        def _apply_two_qubit(
            self, mat: np.ndarray, q0: int, q1: int, n: int
        ):
            """Apply 4×4 matrix to the (q0, q1) pair in the statevector."""
            sv = self._statevector.reshape([2] * n)
            mat4 = mat.reshape(2, 2, 2, 2)  # (out0, out1, in0, in1)
            sv = np.tensordot(mat4, sv, axes=[[2, 3], [q0, q1]])
            # After tensordot:
            #   axes 0,1 → out0, out1
            #   axes 2..n-1 → the n-2 remaining qubit axes (in their original order,
            #                  excluding q0 and q1)
            # We need to transpose back so that qubit i sits at axis i.
            target_order = []
            r_idx = 2  # pointer into the "remaining" block
            for i in range(n):
                if i == q0:
                    target_order.append(0)
                elif i == q1:
                    target_order.append(1)
                else:
                    target_order.append(r_idx)
                    r_idx += 1
            sv = np.transpose(sv, target_order)
            self._statevector = sv.ravel()

        # ------------------------------------------------------------------
        # Observable / measurement interface
        # ------------------------------------------------------------------

        @property
        def state(self):
            """Return the current statevector."""
            return self._statevector

        def analytic_probability(self, wires=None):
            """Return the probability distribution over measurement outcomes."""
            if self._statevector is None:
                raise QuantXPluginError("No circuit has been applied yet.")
            probs = np.abs(self._statevector) ** 2
            if wires is None or list(wires) == list(self.wires):
                return probs
            # Marginalise over qubits not in *wires*.
            n = len(self.wires)
            wire_indices = [self.wires.index(w) for w in wires]
            sv2 = probs.reshape([2] * n)
            axes_to_sum = sorted(
                [i for i in range(n) if i not in wire_indices], reverse=True
            )
            for ax in axes_to_sum:
                sv2 = sv2.sum(axis=ax)
            # After summing, remaining axes are in device-wire order.
            # Transpose to match the requested wires order.
            remaining_device_axes = sorted(wire_indices)
            perm = [remaining_device_axes.index(wi) for wi in wire_indices]
            sv2 = np.transpose(sv2, perm)
            return sv2.ravel()

else:
    # ---------------------------------------------------------------------------
    # Stub: PennyLane or NumPy not available
    # ---------------------------------------------------------------------------
    class QuantXDevice:  # type: ignore[no-redef]
        """Stub – install pennylane to use the QuantX PennyLane device."""

        def __init__(self, *args, **kwargs):
            raise QuantXPluginError(
                "PennyLane is not installed. Install it with:\n"
                "    pip install pennylane"
            )


# ---------------------------------------------------------------------------
# Auto-registration with PennyLane's plugin registry
# ---------------------------------------------------------------------------

def _register_device() -> None:
    """Register ``quantx.statevector`` with PennyLane's device registry.

    This is a best-effort registration for the in-process case.  For full
    ``qml.device("quantx.statevector", ...)`` support across processes the
    package must be installed with its entry-point declared in
    ``pyproject.toml``::

        [project.entry-points."pennylane.plugins"]
        "quantx.statevector" = "quantx_engine.contrib.pennylane_plugin:QuantXDevice"
    """
    if not _HAS_PENNYLANE:
        return
    try:
        registry = getattr(qml, "plugin_devices", None)
        if registry is not None and "quantx.statevector" not in registry:
            registry["quantx.statevector"] = QuantXDevice
    except Exception:
        # Defensive: registry shape may change across PL versions.
        pass


_register_device()
