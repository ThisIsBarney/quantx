Experimental modules
====================

.. warning::

   Everything on this page is experimental. Its API may change, the engine's gates do
   not cover it, and the pure-Python modules may move to a separate
   ``quantx_engine.contrib`` namespace. Most of these modules do not use the native
   engine. For the engine, see the :doc:`engine API <engine>`.

The modules are grouped by topic.

.. toctree::
   :maxdepth: 1

   core
   circuit_gate
   many_body
   chemistry
   qec
   algorithms
   networking
   hardware_control
   ml_characterization
   advanced_physics
   interfaces

----

Module Index
------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantx_engine
   quantx_engine.engine
   quantx_engine.lanczos
   quantx_engine.trotter
   quantx_engine.mclachlan
   quantx_engine.noise
   quantx_engine.mitigation
   quantx_engine.lindblad
   quantx_engine.adapters
   quantx_engine.transpiler
   quantx_engine.qasm
   quantx_engine.templates
   quantx_engine.hardware
   quantx_engine.ibm
   quantx_engine.many_body
   quantx_engine.finite_size
   quantx_engine.mps_simulator
   quantx_engine.tensor_network
   quantx_engine.tdmrg
   quantx_engine.spectral
   quantx_engine.chemistry
   quantx_engine.advanced_chemistry
   quantx_engine.oovqe
   quantx_engine.noise_vqe
   quantx_engine.qaoa_optim
   quantx_engine.qec
   quantx_engine.fault_tolerant
   quantx_engine.ft_tgate
   quantx_engine.clifford_sim
   quantx_engine.clifford_t
   quantx_engine.ftqc_pipeline
   quantx_engine.qldpc
   quantx_engine.bosonic
   quantx_engine.quantum_walk
   quantx_engine.qsp_qsvt
   quantx_engine.excited_state
   quantx_engine.dynamic
   quantx_engine.sweep
   quantx_engine.mbqc
   quantx_engine.quantum_network
   quantx_engine.quantum_protocols
   quantx_engine.pulse
   quantx_engine.optimal_control
   quantx_engine.dyndecouple
   quantx_engine.crosstalk
   quantx_engine.gpu
   quantx_engine.ray_sim
   quantx_engine.qml
   quantx_engine.qbm
   quantx_engine.quantum_kernel
   quantx_engine.rb_qv
   quantx_engine.noise_char
   quantx_engine.xeb
   quantx_engine.tomography
   quantx_engine.floquet
   quantx_engine.neqdyn
   quantx_engine.rydberg
   quantx_engine.pennylane_plugin
   quantx_engine.primitives
   quantx_engine.jupyter_widgets
   quantx_engine.circuit
   quantx_engine.statevector
