# Engine notebooks

These run the native engine against exact references, and assert the agreement:

| Notebook | What it checks |
|---|---|
| [10_hubbard_ground_state.ipynb](10_hubbard_ground_state.ipynb) | Hubbard chain ground states against an exact diagonalisation at 8 sites (to 1e-10), and against the Lieb–Wu solution of the infinite chain: the energy per site and the Mott gap |
| [11_finite_temperature.ipynb](11_finite_temperature.ipynb) | `quantx_engine.thermal` against an exact diagonalisation of the whole Hilbert space: the energy, the susceptibility and a dynamical correlation; then S(k, ω, T) and σ(ω, T) of a longer chain |

They need the native engine, which the `pip install quantx-engine` wheels carry on Linux,
macOS and Windows, plus matplotlib and scipy (`pip install -r notebooks/requirements.txt`).
Each runs in about a minute. CI runs them in the Linux native job, against the library it
builds; the notebooks in the parent folder do not need the engine.
