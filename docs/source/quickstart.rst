Quick Start
===========

This page shows five common workflows to get you started.

1. Gate-model circuit simulation
---------------------------------

.. code-block:: python

   from quantx_engine.engine import Engine

   engine = Engine()                 # needs QUANTX_LIB_PATH set

   circuit = {
       "qubits": 2,
       "gates": [
           {"gate": "H",    "target": 0},
           {"gate": "CNOT", "control": 0, "target": 1},
       ],
   }
   result = engine.run_circuit(circuit)
   print(result["counts"])           # e.g. {"00": 512, "11": 512}

2. Lanczos eigensolver
-----------------------

Find the ground-state energy of a 1D Heisenberg chain:

.. code-block:: python

   import numpy as np
   from quantx_engine.lanczos import LanczosSolver

   # 2-site Heisenberg: H = J*(XX + YY + ZZ), J=1
   # exact E₀ = -3/2
   H = np.array([
       [-1,  0,  0,  0],
       [ 0,  1, -2,  0],
       [ 0, -2,  1,  0],
       [ 0,  0,  0, -1],
   ], dtype=float) * 0.5

   solver = LanczosSolver(H, num_eigenvalues=1, max_iterations=100, tolerance=1e-10)
   result = solver.solve()
   print(f"E₀ = {result.eigenvalues[0]:.6f}")   # -1.500000

3. Variational Quantum Eigensolver (VQE)
-----------------------------------------

.. code-block:: python

   from quantx_engine.advanced_chemistry import VQE, MolecularHamiltonian

   mol = MolecularHamiltonian(n_qubits=4, terms=[
       (0.5, "ZI"), (-0.5, "IZ"), (0.25, "XX"), (0.25, "YY"),
   ])
   vqe = VQE(mol, ansatz="hardware_efficient", layers=2)
   result = vqe.run(max_iterations=100)
   print(f"VQE energy: {result.energy:.4f}")

4. Quantum error correction
-----------------------------

Test a distance-3 repetition code:

.. code-block:: python

   from quantx_engine.qec import RepetitionCode

   code = RepetitionCode(n=3)
   # Encode logical |0>
   logical = [0]
   encoded = code.encode(logical)         # [0, 0, 0]
   # Flip the middle qubit
   noisy = list(encoded)
   noisy[1] ^= 1                          # [0, 1, 0]
   # Decode
   syndrome = code.syndrome(noisy)
   corrected = code.correct(syndrome, noisy)
   decoded = code.decode(corrected)
   assert decoded == logical, "Correction failed!"
   print("Error corrected successfully")

5. Quantum network simulation
------------------------------

Simulate entanglement distribution over a three-node chain:

.. code-block:: python

   from quantx_engine.quantum_network import QuantumNetwork

   net = QuantumNetwork()
   net.add_node("Alice")
   net.add_node("Bob")
   net.add_node("Charlie")
   net.add_link("Alice", "Bob",     fidelity=0.95, distance_km=10)
   net.add_link("Bob",   "Charlie", fidelity=0.92, distance_km=15)

   # Swap entanglement through Bob
   fidelity = net.entanglement_swap("Alice", "Bob", "Charlie")
   print(f"Alice-Charlie fidelity after swap: {fidelity:.3f}")

Next steps
----------

* Browse the :doc:`API Reference <api/index>` for all 130+ modules.
* Explore the REST API docs at :doc:`rest_api`.
* Read the :doc:`changelog` for what changed in v0.2.0.
