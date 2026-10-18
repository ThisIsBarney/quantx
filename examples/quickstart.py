"""Quick start: three things the engine does, each checked against its exact answer.

With the engine installed this takes a few seconds on a desktop CPU:

    python examples/quickstart.py

It needs quantx-engine 0.2.0 or later, for `quantx_engine.thermal`.

Each assertion's value is an exact reference computed without the engine, by exact
diagonalisation or a state vector; python/quantx_engine/tests/test_quickstart.py
recomputes all three and runs this file against them.
"""
from quantx_engine import hva, thermal
from quantx_engine.dmrg import ground_state
from quantx_engine.models import HubbardChain, HubbardLattice, XXZChain

# A Hubbard ground state: 8 sites, U = 4, 4 up and 4 down electrons.
gs = ground_state(HubbardChain(L=8, U=4.0), n_up=4, n_dn=4)
print(f"Hubbard ground-state energy  {gs.energy:.10f}")
assert abs(gs.energy - -4.2358069991) < 1e-9            # exact diagonalisation

# A finite-temperature spectral function: S(k, omega, T) of the Heisenberg chain at T = 1.
sf = thermal.structure_factor(XXZChain(L=8), beta=1.0, t_max=4.0, max_bond=256)
k, omega, S = sf.spectrum()
print(f"Peak of S(k, omega, T)       {S.max():.6f}")
assert abs(S.max() - 0.795005) < 1e-3                    # exact diagonalisation

# An HVA circuit: two layers on a 3 x 2 Hubbard ladder, from a product state.
ladder = HubbardLattice(Lx=3, W=2, U=4.0)
angles = [[0.1] * hva.layout(ladder).angles_per_layer] * 2
r = hva.circuit(ladder, angles, n_up=3, n_dn=3, init="product")
print(f"HVA circuit energy           {r.energy:.10f}")
assert abs(r.energy - -0.3501424393) < 1e-9              # state vector
