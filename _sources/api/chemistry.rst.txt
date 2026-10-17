Chemistry & Optimization
=========================

Variational quantum eigensolver, molecular Hamiltonians, and combinatorial
optimization via QAOA.

Quantum Chemistry
------------------

Molecular Hamiltonians, second-quantized operators, VQE driver.

.. automodule:: quantx_engine.chemistry
   :members:
   :show-inheritance:

Advanced Chemistry
-------------------

Noise-aware VQE, excited-state solver, CASSCF, multi-reference methods.

.. automodule:: quantx_engine.advanced_chemistry
   :members:
   :show-inheritance:

OO-VQE (Orbital-Optimised VQE)
--------------------------------

.. automodule:: quantx_engine.oovqe
   :members:
   :show-inheritance:

Noise-Aware VQE
----------------

VQE with device-level noise characterisation and mitigation loops.

.. automodule:: quantx_engine.noise_vqe
   :members:
   :show-inheritance:

CVaR-QAOA Optimizer
--------------------

Conditional Value-at-Risk QAOA for combinatorial problems (MaxCut, TSP).

.. note::

   The class exported as ``CVaRQAOAResult`` in ``quantx_engine`` is the result
   dataclass from this module (renamed to avoid a clash with
   ``advanced_chemistry.QAOAResult``).

.. automodule:: quantx_engine.qaoa_optim
   :members:
   :show-inheritance:

Excited-State Methods
---------------------

.. automodule:: quantx_engine.excited_state
   :members:
   :show-inheritance:
