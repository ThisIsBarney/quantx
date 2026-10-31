Interfaces & Integrations
==========================

Modules providing interoperability with external frameworks and tools.

PennyLane Plugin
-----------------

Register QuantX as a PennyLane device (requires ``quantx-engine[pennylane]``):

.. code-block:: python

   import pennylane as qml

   dev = qml.device("quantx.statevector", wires=4)

   @qml.qnode(dev)
   def circuit():
       qml.Hadamard(0)
       qml.CNOT(wires=[0, 1])
       return qml.expval(qml.PauliZ(0))

   print(circuit())

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
