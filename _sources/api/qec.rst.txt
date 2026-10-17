Quantum Error Correction
=========================

Modules implementing quantum error correcting codes, fault-tolerant
architectures, logical qubit encodings, and bosonic codes.

Stabilizer Codes
-----------------

.. automodule:: quantx_engine.qec
   :members:
   :show-inheritance:

Surface Code & Fault-Tolerant Thresholds
------------------------------------------

.. automodule:: quantx_engine.fault_tolerant
   :members:
   :show-inheritance:

Fault-Tolerant T-Gate Synthesis
---------------------------------

Magic state distillation, T-gate synthesis via resource states.

.. automodule:: quantx_engine.ft_tgate
   :members:
   :show-inheritance:

Clifford Simulator
-------------------

Stabilizer (Clifford) circuit simulator using the tableau representation.

.. automodule:: quantx_engine.clifford_sim
   :members:
   :show-inheritance:

Clifford + T Gate Set
----------------------

.. automodule:: quantx_engine.clifford_t
   :members:
   :show-inheritance:

Fault-Tolerant QC Pipeline
----------------------------

End-to-end FTQC pipeline: logical encoding → magic-state distillation → decoding.

.. automodule:: quantx_engine.ftqc_pipeline
   :members:
   :show-inheritance:

Quantum LDPC Codes (M119)
--------------------------

Bicycle codes, Tanner graph construction, belief-propagation decoder.

.. automodule:: quantx_engine.qldpc
   :members:
   :show-inheritance:

Bosonic Qubits (M120)
----------------------

Cat qubits, GKP codes, Fock-space operators, bosonic channel noise.

.. automodule:: quantx_engine.bosonic
   :members:
   :show-inheritance:

Measurement-Based QC (MBQC)
-----------------------------

Graph states, cluster state preparation, one-way computation.

.. automodule:: quantx_engine.mbqc
   :members:
   :show-inheritance:
