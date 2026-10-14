QuantX
======

**QuantX** is a tensor-network engine for quantum many-body physics. It runs
block-sparse MPS algorithms (DMRG, TDVP and fermionic HVA circuits) that conserve U(1),
U(1)×U(1) and SU(2) charges exactly. The core is C++20, with a Python API.

These pages document what the engine does from Python and how each capability is
checked. Everything else in the package is marked experimental.

What is verified, and against what
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Capability
     - Python
     - Checked against
   * - Ground states of the XXZ, J1-J2, spin-1 Heisenberg and Fermi-Hubbard chains, in a
       chosen symmetry sector
     - :func:`~quantx_engine.dmrg.ground_state` with the models of
       :mod:`quantx_engine.models`
     - exact diagonalisation, written independently of the engine, to 1e-10
   * - Hubbard ground states on ladders, strips and cylinders
     - ``ground_state(HubbardLattice(...))``
     - exact diagonalisation on a ladder and a cylinder, where the fermion signs are
       physical, to 1e-10
   * - Fermionic HVA (VQE) circuits on chains, ladders, strips and cylinders: energy,
       entanglement entropy, truncation
     - :func:`~quantx_engine.hva.circuit`, :func:`~quantx_engine.hva.layout`
     - an independent state vector: energy and entropies to 1e-10 from a product start;
       to 1e-6 from the default U = 0 start (about 1e-8 measured)

These checks run in CI on a Linux build of the engine (``test_models_ed.py`` and
``test_hva_statevector.py`` in ``python/quantx_engine/tests/``).

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/engine
   api/index

.. toctree::
   :maxdepth: 1
   :caption: More

   rest_api
   changelog

The project's `roadmap <https://github.com/ThisIsBarney/quantx/blob/master/ROADMAP.md>`_
says where it is going, and the
`contributing guide <https://github.com/ThisIsBarney/quantx/blob/master/CONTRIBUTING.md>`_
says how to take part. QuantX is released under the MIT License.
