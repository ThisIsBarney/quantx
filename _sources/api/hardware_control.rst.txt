Hardware & Control
===================

Modules for pulse-level simulation, optimal control, and hardware-specific
noise characterisation.

Pulse-Level Simulation
-----------------------

Drive pulse scheduling, drag pulses, frame tracking.

.. automodule:: quantx_engine.pulse
   :members:
   :show-inheritance:

Optimal Control (GRAPE / CRAB)
-------------------------------

.. automodule:: quantx_engine.optimal_control
   :members:
   :show-inheritance:

Dynamical Decoupling
---------------------

XY-4, CPMG, Uhrig DD sequences.

.. automodule:: quantx_engine.dyndecouple
   :members:
   :show-inheritance:

Crosstalk Characterisation
---------------------------

.. automodule:: quantx_engine.crosstalk
   :members:
   :show-inheritance:

GPU Acceleration
-----------------

CuPy-backed statevector simulation (requires ``quantx-engine[gpu]``).

.. automodule:: quantx_engine.gpu
   :members:
   :show-inheritance:

Distributed Simulation (Ray)
-----------------------------

Multi-node statevector simulation with Ray (requires ``quantx-engine[distributed]``).

.. automodule:: quantx_engine.ray_sim
   :members:
   :show-inheritance:

Floquet Theory
---------------

Stroboscopic evolution, quasi-energy bands, Floquet Hamiltonian.
See :doc:`advanced_physics` for full API.
