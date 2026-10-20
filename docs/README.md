# Documents in `docs/`

The Python API is documented on the documentation site built from `docs/source/`. This
directory also holds the engine's technical reports, with their figures and data, and
older documents kept as a record. The tables say which is which. Where a report and a
record disagree, the report wins.

## Technical reports

Each one documents a part of the engine and how it was validated. They are current
unless a later report says otherwise.

| Report | Topic |
|---|---|
| [dmrg_block_sparse_report.md](dmrg_block_sparse_report.md) ([中文](dmrg_block_sparse_report_zh.md)) | Sz-conserving block-sparse DMRG: correctness fixes and performance |
| [dmrg_su2_benchmark_report.md](dmrg_su2_benchmark_report.md) | U(1) and SU(2) ground-state solvers against TeNPy |
| [dmrg_tdvp_report.md](dmrg_tdvp_report.md) | TDVP time evolution |
| [time_evolution_status.md](time_evolution_status.md) | A correction to an earlier time-evolution benchmark, and the honest status |
| [gpu_acceleration_report.md](gpu_acceleration_report.md) | The cuSOLVER GPU path (phases A to C) |
| [finite_temperature.md](finite_temperature.md) | Finite-temperature spectroscopy by purification |
| [universal_mpo.md](universal_mpo.md) | Any charge-conserving model on the block-sparse engine |
| [referee_su2_hadron.md](referee_su2_hadron.md) | Whether the engine can referee the Quantum Advantage Tracker's SU(2) hadron instance (M8), and the bond dimension its observable needs: n_f(20) = 0.1169 ± 0.0022 at χ = 1600 on QuantX's U(1) MPS, with the charge exact (M16) |
| [showcase_google2023.md](showcase_google2023.md) | Reproducing a surface-code result (experimental modules) |

The `.csv` and `.svg` files beside them are the data and figures these reports cite.

## Plans and records (not current)

These are kept for the history. Some describe the project as it was before it was
refocused on the engine, in October 2026; [ROADMAP.md](../ROADMAP.md) is the current
plan, and [CONTRIBUTING.md](../CONTRIBUTING.md) the current development guide.

| Document | What it was |
|---|---|
| [gpu_acceleration_plan.md](gpu_acceleration_plan.md), [finite_temperature_plan.md](finite_temperature_plan.md) | The plans those two reports carried out |
| [add_your_hamiltonian.md](add_your_hamiltonian.md) | A guide to adding a pure-Python solver. New models now go through the engine ([universal_mpo.md](universal_mpo.md)), with a gate ([CONTRIBUTING.md](../CONTRIBUTING.md)) |
| [positioning.md](positioning.md) | An earlier positioning, superseded by the README and the roadmap |
| [USAGE.md](USAGE.md) | A usage guide to the older JSON entry points (the CLI and `Engine.run`) |
| [QUANTUM_COMPUTING_BACKEND.md](QUANTUM_COMPUTING_BACKEND.md), [ML_LLM_INTEGRATION.md](ML_LLM_INTEGRATION.md) | Early design notes; the machine-learning and LLM directions are paused |
| [ENGINEERING_STANDARDS.md](ENGINEERING_STANDARDS.md), [TESTING.md](TESTING.md), [DEPENDENCIES.md](DEPENDENCIES.md) | Early process notes. Third-party licenses are now in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) |
