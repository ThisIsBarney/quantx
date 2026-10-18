"""Experimental modules.

Their API may change, and the engine's checks do not cover them. Most are pure Python and
do not use the native engine: VQE and quantum chemistry, error correction and decoders,
noise models and mitigation, quantum walks, bosonic codes, an IBM Quantum connector, a
PennyLane device and more. A few are older interfaces to parts of the engine
(``lindblad``, ``cached_dmrg``). They moved here from ``quantx_engine`` in 0.2.0; the old
paths warn, and go in 0.4.0. They are frozen: new features go to the engine API.

Import a module directly::

    from quantx_engine.contrib.qec import SurfaceCode
"""
