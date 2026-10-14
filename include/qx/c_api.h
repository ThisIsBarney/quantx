#pragma once

#ifdef _WIN32
#ifdef QX_CAPI_EXPORTS
#define QX_API __declspec(dllexport)
#else
#define QX_API __declspec(dllimport)
#endif
#else
#define QX_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Run Hamiltonian simulation from JSON inputs and return heap-allocated JSON output.
// Error payloads are append-only JSON objects with ok/error_kind/error_code/error_message/error.
QX_API const char* qx_run(const char* problem_json, const char* options_json);
// Explain backend choice from JSON inputs and return heap-allocated JSON output.
QX_API const char* qx_explain_backend_choice(const char* problem_json, const char* options_json);
// Run circuit simulation from JSON inputs and return heap-allocated JSON output.
// Error payloads are append-only JSON objects with ok/error_kind/error_code/error_message/error.
QX_API const char* qx_run_circuit(const char* circuit_json, const char* options_json);
// Query gate-model capability metadata and return heap-allocated JSON output.
QX_API const char* qx_circuit_capabilities(void);
// Validate problem JSON payload and return JSON status.
// Error payloads are append-only JSON objects with ok/error_kind/error_code/error_message/error.
QX_API const char* qx_validate_problem(const char* problem_json);
// Validate circuit JSON payload and return JSON status.
// Error payloads are append-only JSON objects with ok/error_kind/error_code/error_message/error.
QX_API const char* qx_validate_circuit(const char* circuit_json);
// Free strings returned by qx_run/qx_explain_backend_choice/qx_run_circuit.
QX_API void qx_free_string(const char* ptr);
// Return engine semantic version string (no ownership transfer required).
QX_API const char* qx_version(void);
// Run a named specialized physics solver from JSON options and return heap-allocated JSON output.
// solver: solver name (e.g., "vqe", "toric", "hubbard", ...).  Use qx_list_solvers() for full list.
QX_API const char* qx_run_solver(const char* solver, const char* opts_json);
// Return a heap-allocated JSON array of all available solver names.
QX_API const char* qx_list_solvers(void);

// ── M173: Sparse (matrix-free) Lindblad solver ───────────────────────────────
// Supports up to 16 qubits for time evolution (vs 10 in the dense backend).
// Steady-state solver supports up to 12 qubits (explicit sparse Liouvillian).
//
// problem_json fields:
//   n_qubits        : int
//   hamiltonian_terms: [{coeff: float, pauli: "XZI..."}]
//   jump_operators  : [{type: "amplitude_damping"|"dephasing"|"depolarizing",
//                       qubit: int, rate: float}]
//   total_time      : float  (evolve only)
//   dt              : float  (evolve only)
//
// Returns thread-local JSON string (valid until next call on same thread).

// Time evolution: returns lindblad_frames + final_purity + final_entropy.
QX_API const char* qx_lindblad_sparse_evolve(const char* problem_json,
                                              const char* options_json);

// Steady-state density matrix: returns populations + purity + entropy.
QX_API const char* qx_lindblad_sparse_steady(const char* problem_json);

// ── M176: C++ MPS tensor contractions + two-site DMRG ────────────────────────
// Finds the ground state of a 1D open chain via two-site DMRG. The symmetric models
// run on the block-sparse engine in a fixed U(1) sector; pauli_sum runs dense.
//
// problem_json fields:
//   n_sites           : int     (chain length; the legacy key n_qubits is also read)
//   type              : "heisenberg" | "xxz" | "j1j2" | "spin1_heisenberg" | "hubbard"
//                       | "pauli_sum" (the default); any other type is an error
//   Hamiltonians, open boundaries, in each builder's own convention:
//     heisenberg, xxz   J sum_i [X_i X_{i+1} + Y_i Y_{i+1} + delta Z_i Z_{i+1}]   (Pauli
//                       matrices, so 4x the S = sigma/2 convention); J, delta default 1
//     j1j2              sum_i [J1 h_{i,i+1} + J2 h_{i,i+2}], h = XX + YY + delta ZZ
//                       (Pauli); J1 = 1, J2 = 0, delta = 1
//     spin1_heisenberg  J sum_i S_i . S_{i+1} with spin-1 operators; J = 1
//     hubbard           -t sum_{i,s} (c+_{i s} c_{i+1 s} + h.c.) + U sum_i n_{i up} n_{i dn}
//                       - mu sum_i n_i; t = 1, U = 0, mu = 0
//     hubbard_lattice   the same Hamiltonian on a width-W lattice of Lx columns: fields
//                       Lx, W (1 = chain, 2 = ladder), y_periodic (W >= 3: cylinder).
//                       Sites are column-major, site = W*x + y; n_sites is W*Lx
//   sector            : object, symmetric models only. Spin models: {sz2: 2 Sz_total},
//                       default the lowest |Sz| (0, or 1 for an odd spin-1/2 chain).
//                       hubbard: {n_up, n_dn}, default half filling, n_up = ceil(n/2).
//                       The symmetric models need n_sites >= 3 (heisenberg at n = 2
//                       falls back to the dense solver, which has no sector).
//   terms             : [{coeff: float, pauli: "X0X1"}]  (pauli_sum)
//   hamiltonian_terms : [[coeff, "pauli"]]               (alternative key)
//
// options_json fields:
//   max_bond  : int    (default 64)
//   cutoff    : float  (SVD truncation weight cutoff, default 1e-10)
//   n_sweeps  : int    (max sweeps, default 10)
//   tol       : float  (energy convergence tolerance, default 1e-10)
//   method    : "u1" (default) | "su2" (heisenberg only; field spin = 0.5, 1, ...)
//
// Returns thread-local JSON: {ok, energy, truncation_error, n_sweeps_done,
//                             wall_time_s, sweep_energies[], n_sites, method,
//                             sector (symmetric models)}. method names the solver
//                             that ran: "u1", "u1xu1" (hubbard: N and 2Sz), "su2",
//                             or "dense".
QX_API const char* qx_dmrg_sweep(const char* problem_json,
                                  const char* options_json);

// Compute <0...0|H|0...0> for the all-zero product state.
// Returns thread-local JSON: {ok, energy}
QX_API const char* qx_mps_energy(const char* problem_json);

// The four entries below were exported from the library but missing from this header, so a
// C caller could only reach them by declaring the symbols itself. All take
// (problem_json, options_json), return thread-local JSON with "ok", and support the
// Heisenberg/XXZ chain (problem type "heisenberg", fields n_qubits, J, delta).

// Real- or imaginary-time evolution from a Neel state (block TEBD/TDVP2 by default; falls
// back to non-block TDVP for n_qubits < 3). Options: dt, n_steps, chi_max, imag, block.
QX_API const char* qx_tdvp_evolve(const char* problem_json, const char* options_json);
// Ground-state dynamical structure factor data. Block-sparse only: n_qubits >= 3.
QX_API const char* qx_spectral(const char* problem_json, const char* options_json);
// Loschmidt echo after a quench to delta_f. Block-sparse only: n_qubits >= 3.
QX_API const char* qx_loschmidt(const char* problem_json, const char* options_json);
// Ground-state current-current correlator C(t) = <psi0|J e^{-i(H-E0)t} J|psi0>, whose FFT
// is the zero-temperature spin conductivity. (The finite-temperature sigma(omega,T) of the
// purification track is not exposed yet.) Block-sparse only: n_qubits >= 3.
QX_API const char* qx_conductivity(const char* problem_json, const char* options_json);

// ── M5: fermionic HVA circuits on the Hubbard lattice ────────────────────────
// The Hamiltonian Variational Ansatz, simulated exactly (up to max_bond) on the
// block-sparse engine. Each layer applies exp(-i theta_g H_g) for every bond group g,
// then exp(-i theta_U H_onsite). H_g sums the hopping terms of bonds that share no site,
// so it factorises into two-site gates with no Trotter error. The groups depend on the
// geometry; qx_hva_layout lists them.
//
// problem_json: the hubbard_lattice fields of qx_dmrg_sweep (Lx, W, y_periodic, t, U, mu,
//   sector {n_up, n_dn}), plus
//   angles : [[theta_group_0, ..., theta_group_last, theta_onsite], ...], one list per layer
// options_json:
//   max_bond : int    (default 256; also caps the free-state preparation, at no more
//              than the 4^(n/2) an n-site chain can use)
//   cutoff   : float  (default 1e-12; the circuit's truncation -- the preparation uses
//              its own 1e-12)
//   init     : "free" (default: the U = 0 ground state, prepared by DMRG; refused where it
//              is degenerate) | "product" (a product state in the sector)
//   entropy  : bool   (default true; the entropies cost one extra lossless sweep)
// Returns thread-local JSON: {ok, energy, entropy_max, entropy_mid (null for odd W*Lx),
//   truncation_error (the circuit's LARGEST single discarded weight), max_bond,
//   layer_bonds[], wall_time_s, sector, init,
//   init "free": init_energy, init_energy_exact (the U = 0 ground-state energy from the
//     single-particle levels), init_energy_error (their difference: how far the DMRG
//     preparation stopped short. A large value means a bad start; a small one does not
//     bound the circuit's error, since the energy is stationary at the ground state
//     while the circuit inherits the state's error at first order), and
//     init_truncation_error (the SUM over the preparation's last sweep),
//   init "product": init_config, the local state per site, 0 empty, 1 up, 2 down,
//     3 double, with |up dn> = c+_up c+_dn |0>}.
QX_API const char* qx_hva_circuit(const char* problem_json, const char* options_json);

// The bond groups of the lattice circuit in application order:
// {ok, n_sites, groups: [[[i, j], ...], ...], angles_per_layer}.
QX_API const char* qx_hva_layout(const char* problem_json);

#ifdef __cplusplus
}
#endif
