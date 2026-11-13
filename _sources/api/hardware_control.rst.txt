Hardware & Control
===================

Modules for pulse-level simulation, optimal control, and hardware-specific
noise characterisation.

Pulse-Level Simulation
-----------------------

Drive pulse scheduling, drag pulses, frame tracking.

.. automodule:: quantx_engine.contrib.pulse
   :members:
   :show-inheritance:

Optimal Control (GRAPE / CRAB)
-------------------------------

.. automodule:: quantx_engine.contrib.optimal_control
   :members:
   :show-inheritance:

Dynamical Decoupling
---------------------

XY-4, CPMG, Uhrig DD sequences.

.. automodule:: quantx_engine.contrib.dyndecouple
   :members:
   :show-inheritance:

Crosstalk Characterisation
---------------------------

.. automodule:: quantx_engine.contrib.crosstalk
   :members:
   :show-inheritance:

GPU Acceleration
-----------------

CuPy-backed statevector simulation (requires ``quantx-engine[gpu]``).

.. automodule:: quantx_engine.contrib.gpu
   :members:
   :show-inheritance:

Distributed Simulation (Ray)
-----------------------------

Multi-node statevector simulation with Ray (requires ``quantx-engine[distributed]``).

.. automodule:: quantx_engine.contrib.ray_sim
   :members:
   :show-inheritance:

Floquet Theory
---------------

Stroboscopic evolution, quasi-energy bands, Floquet Hamiltonian.
See :doc:`advanced_physics` for full API.
