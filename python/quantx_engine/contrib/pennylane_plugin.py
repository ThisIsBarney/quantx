"""The PennyLane device's former home: use :mod:`quantx_engine.pennylane_device`.

``qml.device("quantx.statevector", wires=n)`` is the way to reach the device. It used to be a
pure-Python simulator defined here; since M15 it runs on the native engine, and
:class:`QuantXDevice` is the same class as
:class:`~quantx_engine.pennylane_device.StatevectorDevice`.
"""
from __future__ import annotations

__all__ = ["QuantXDevice", "QuantXPluginError"]


class QuantXPluginError(RuntimeError):
    """Raised when the device cannot be built because PennyLane is not installed."""


try:
    from quantx_engine.pennylane_device import StatevectorDevice as QuantXDevice
except ImportError as _exc:
    _reason = str(_exc)

    class QuantXDevice:  # type: ignore[no-redef]
        """Stub: the device needs PennyLane 0.45 or later."""

        def __init__(self, *args, **kwargs):
            raise QuantXPluginError(
                "The QuantX device needs PennyLane 0.45 or later (and so Python 3.11). "
                f"Install it with: pip install \"pennylane>=0.45\". ({_reason})"
            )
