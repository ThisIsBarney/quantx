Engine API
==========

These modules drive the native engine. They need the native library, which comes with
the platform wheels (or set ``QUANTX_LIB_PATH``); without it, each raises its module's
error, naming what is missing.

:func:`~quantx_engine.dmrg.ground_state` with :mod:`quantx_engine.models`,
:mod:`quantx_engine.hva`, :mod:`quantx_engine.thermal`, :mod:`quantx_engine.u1_circuit` and
:mod:`quantx_engine.mps` are gated: each capability is tested against a reference computed independently of the
engine (see the :doc:`front page <../index>`).
:mod:`quantx_engine.tdvp`, :mod:`quantx_engine.structure_factor` and the SU(2) path of
:class:`~quantx_engine.dmrg.DMRGSolver` are validated in the C++ tests but have no
Python gate yet.

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
   * - :mod:`quantx_engine.thermal`
     - Finite temperature by purification, for the XXZ, spin-1 and Hubbard chains: the
       statics, the dynamical structure factor S(k, ω, T) and the optical conductivity
       σ(ω, T).
   * - :mod:`quantx_engine.u1_circuit`
     - Qubit circuits that conserve the number of qubits in ``|1>``: OpenQASM fused into
       one- and two-qubit gates that conserve it, run on the block-sparse MPS with that
       number as an exact charge; <Z_i> at chosen points of the circuit.
   * - :mod:`quantx_engine.mps`
     - Matrix-product states that stay in the engine between calls: a DMRG ground state
       or a product state, one-site diagonal operators, TDVP in real or imaginary time
       (under any model of the same local space), overlaps, matrix elements and
       energies, chained without the state passing through Python.
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
   quantx_engine.thermal
   quantx_engine.u1_circuit
   quantx_engine.mps
   quantx_engine.tdvp
   quantx_engine.structure_factor
