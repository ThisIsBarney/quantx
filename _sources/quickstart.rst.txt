Quick start
===========

Ground states
-------------

Build a model, then solve it in a symmetry sector:

.. code-block:: python

   from quantx_engine.models import HubbardChain, XXZChain
   from quantx_engine.dmrg import ground_state

   gs = ground_state(HubbardChain(L=16, U=4.0), n_up=8, n_dn=8, max_bond=256)
   gs.energy, gs.truncation_error, gs.sector

   gs = ground_state(XXZChain(L=21, delta=0.5), sz=0.5)   # an odd chain, S^z = 1/2

The models are :class:`~quantx_engine.models.XXZChain`,
:class:`~quantx_engine.models.J1J2Chain`, :class:`~quantx_engine.models.Spin1Chain`,
:class:`~quantx_engine.models.HubbardChain` and
:class:`~quantx_engine.models.HubbardLattice` (ladders, strips and cylinders).

Fermionic circuits
------------------

.. code-block:: python

   from quantx_engine.models import HubbardLattice
   from quantx_engine import hva

   cyl = HubbardLattice(Lx=4, W=3, U=4.0, y_periodic=True)
   angles = [[0.1] * hva.layout(cyl).angles_per_layer] * 3      # 3 layers
   r = hva.circuit(cyl, angles, n_up=6, n_dn=6, max_bond=256)   # ~10 s
   r.energy, r.entropy_max, r.truncation_error, r.init_energy_error

This cylinder already truncates at ``max_bond=256``: ``truncation_error`` is about 5e-4,
and the energy moves by 3e-3 at ``max_bond=1024``. Raise ``max_bond`` until it stops
moving.

Finite temperature
------------------

.. code-block:: python

   from quantx_engine import thermal
   from quantx_engine.models import XXZChain

   sf = thermal.structure_factor(XXZChain(L=12), beta=1.0, t_max=4.0)   # ~10 s
   k, omega, S = sf.spectrum()     # S(k, omega, T), both signs of omega

The ensemble is grand canonical. :mod:`quantx_engine.thermal` also gives the statics and the
optical conductivity.

All three, checked
------------------

``examples/quickstart.py`` in the repository runs a Hubbard ground state, a
finite-temperature spectral function and an HVA circuit, and asserts each against an exact
answer computed without the engine. CI runs it on every wheel it builds.

.. literalinclude:: ../../examples/quickstart.py
   :language: python
   :start-at: from quantx_engine import hva

Conventions
-----------

* The spin models use physical spin operators S = σ/2 throughout.
* Every chain is open.
* Hubbard is H = −t Σ (c†c + h.c.) + U Σ n↑n↓ − μ Σ n.
* A solver works in one symmetry sector: total S\ :sup:`z`, or the numbers of up and
  down electrons.
* An unreachable sector, or an unknown argument, is an error rather than a silent
  default.

Reading the results
-------------------

* ``truncation_error`` is the weight the bond-dimension cap discarded. For
  ``ground_state`` it is summed over the last sweep; for ``hva.circuit`` it is the
  largest single truncation.
* An HVA circuit started from the U = 0 ground state (``init="free"``, the default)
  inherits that state's DMRG error. ``init_energy_error`` is the start's energy against
  the exact U = 0 energy. A large value flags a bad start, but a small one does not
  bound the circuit's error: the energy is stationary at the ground state, so a start
  right to 1e-14 in energy can still carry ~1e-8 into the circuit. For the full
  precision, start from ``init="product"``.

Next steps
----------

* The :doc:`engine API <api/engine>` documents every function and result field.
* The experimental modules are listed in the :doc:`API reference <api/index>`.
