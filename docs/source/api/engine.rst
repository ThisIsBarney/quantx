Engine API
==========

These modules drive the native engine and are held to its gates: each capability is
tested against a reference computed independently of the engine (see the
:doc:`front page <../index>`). They need the native library, which comes with the
platform wheels (or set ``QUANTX_LIB_PATH``). Without it, each raises its module's
error, naming what is missing.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Module
     - What it does
   * - :mod:`quantx_engine.models`
     - The lattice models: XXZ, J1-J2 and spin-1 chains, the Hubbard chain, and Hubbard
       on ladders, strips and cylinders. Each also builds its symmetry sector.
   * - :mod:`quantx_engine.dmrg`
     - :func:`~quantx_engine.dmrg.ground_state`: DMRG in one symmetry sector. The older
       :class:`~quantx_engine.dmrg.DMRGSolver` keeps the Pauli convention and adds the
       SU(2) path for Heisenberg chains.
   * - :mod:`quantx_engine.hva`
     - Fermionic HVA circuits on the Hubbard lattices: energy, entanglement entropies,
       truncation.
   * - :mod:`quantx_engine.tdvp`
     - Real- and imaginary-time evolution and Loschmidt echoes of the XXZ chain.
   * - :mod:`quantx_engine.structure_factor`
     - The zero-temperature dynamical structure factor S(k, ω) and the current-current
       correlator of the XXZ chain.

.. autosummary::
   :toctree: generated
   :nosignatures:

   quantx_engine.models
   quantx_engine.dmrg
   quantx_engine.hva
   quantx_engine.tdvp
   quantx_engine.structure_factor
