Experimental modules
====================

.. warning::

   Everything on this page is experimental, and lives in ``quantx_engine.contrib``. Its
   API may change, and the engine's gates do not cover it. The old import paths, such as
   ``import quantx_engine.qec`` or ``from quantx_engine import SurfaceCode``, still work
   until 0.4.0, with a ``DeprecationWarning``. Most of these modules do not use the
   native engine. For the engine, see the :doc:`engine API <engine>`.

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
   quantx_engine.contrib.lanczos
   quantx_engine.contrib.trotter
   quantx_engine.contrib.mclachlan
   quantx_engine.contrib.noise
   quantx_engine.contrib.mitigation
   quantx_engine.contrib.lindblad
   quantx_engine.contrib.adapters
   quantx_engine.contrib.transpiler
   quantx_engine.contrib.qasm
   quantx_engine.contrib.templates
   quantx_engine.contrib.hardware
   quantx_engine.contrib.ibm
   quantx_engine.contrib.many_body
   quantx_engine.contrib.finite_size
   quantx_engine.contrib.mps_simulator
   quantx_engine.contrib.tensor_network
   quantx_engine.contrib.tdmrg
   quantx_engine.contrib.spectral
   quantx_engine.contrib.chemistry
   quantx_engine.contrib.advanced_chemistry
   quantx_engine.contrib.oovqe
   quantx_engine.contrib.noise_vqe
   quantx_engine.contrib.qaoa_optim
   quantx_engine.contrib.qec
   quantx_engine.contrib.fault_tolerant
   quantx_engine.contrib.ft_tgate
   quantx_engine.contrib.clifford_sim
   quantx_engine.contrib.clifford_t
   quantx_engine.contrib.ftqc_pipeline
   quantx_engine.contrib.qldpc
   quantx_engine.contrib.bosonic
   quantx_engine.contrib.quantum_walk
   quantx_engine.contrib.qsp_qsvt
   quantx_engine.contrib.excited_state
   quantx_engine.contrib.dynamic
   quantx_engine.contrib.sweep
   quantx_engine.contrib.mbqc
   quantx_engine.contrib.quantum_network
   quantx_engine.contrib.quantum_protocols
   quantx_engine.contrib.pulse
   quantx_engine.contrib.optimal_control
   quantx_engine.contrib.dyndecouple
   quantx_engine.contrib.crosstalk
   quantx_engine.contrib.gpu
   quantx_engine.contrib.ray_sim
   quantx_engine.contrib.qml
   quantx_engine.contrib.qbm
   quantx_engine.contrib.quantum_kernel
   quantx_engine.contrib.rb_qv
   quantx_engine.contrib.noise_char
   quantx_engine.contrib.xeb
   quantx_engine.contrib.tomography
   quantx_engine.contrib.floquet
   quantx_engine.contrib.neqdyn
   quantx_engine.contrib.rydberg
   quantx_engine.contrib.pennylane_plugin
   quantx_engine.contrib.primitives
   quantx_engine.contrib.jupyter_widgets
   quantx_engine.contrib.circuit
   quantx_engine.contrib.statevector
