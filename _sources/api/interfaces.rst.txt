Interfaces & Integrations
==========================

Modules providing interoperability with external frameworks and tools.

PennyLane Plugin
-----------------

The devices are part of the engine now, as ``qml.device("quantx.statevector", wires=n)``
and ``qml.device("quantx.mps", wires=n)``: see :mod:`quantx_engine.pennylane_device`. This
module keeps the former import path of the first.

.. automodule:: quantx_engine.contrib.pennylane_plugin
   :members:
   :show-inheritance:

Qiskit V2 Primitives
---------------------

.. automodule:: quantx_engine.contrib.primitives
   :members:
   :show-inheritance:

Jupyter Widgets
----------------

Interactive visualisation widgets for Jupyter notebooks (requires ``ipywidgets``).

.. automodule:: quantx_engine.contrib.jupyter_widgets
   :members:
   :show-inheritance:
