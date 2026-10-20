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
//   gradient : bool   (default false; M13) also return dE/dtheta for every angle, by the
//              adjoint method: one pass back through the circuit, carrying the state
//              and H|psi>. Its cost does not grow with the number of angles (see
//              benchmarks/hva_gradient_cost.py)
// Returns thread-local JSON: {ok, energy, entropy_max, entropy_mid (null for odd W*Lx),
//   gradient (with the option: laid out as the angles), gradient_truncation_error (the
//     largest single weight discarded computing it: H psi's compression to max_bond and
//     the backward pass's truncations), gradient_return_error (1 - |<psi_0|psi>| for the
//     state the backward pass brings back to the start: 0 without truncation),
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

// ── M6: finite temperature by ancilla purification ───────────────────────────
// The Gibbs state rho = e^{-beta H}/Z of an open chain, held as the pure state
// e^{-beta H/2}|I> on a chain of 2*n_sites (each physical site paired with an ancilla;
// |I> is the beta = 0 state), cooled by imaginary-time block TEBD (second order).
// The ensemble is grand canonical: every Sz (and, for hubbard, every N) sector, with
// hubbard's mu fixing the filling.
//
// problem_json: the qx_dmrg_sweep fields for one of
//   heisenberg, xxz   J, delta (Pauli convention)
//   spin1_heisenberg  J (spin-1 S)
//   hubbard           t, U, mu
// with n_sites >= 3. Other types are refused: no purified MPO exists for them.
//
// options_json:
//   quantity  : "statics" | "structure_factor" | "current"   (required)
//   beta      : float > 0. statics: the largest beta of the scan; else the temperature 1/T
//   dbeta     : float > 0 (default 0.05). The cooling step in beta; beta/dbeta is rounded
//               up to a whole number of steps, so the step shrinks to end exactly at beta
//   max_bond  : int (default 128), cutoff: float (default 1e-10)
//   statics only:
//     stride         : int (default 1). Measure every stride-th step, and at the end
//     susceptibility : bool (default true; costs n_sites overlaps per measurement)
//   structure_factor and current:
//     dt         : float > 0 (default 0.05), t_max: float >= 0 (default 5). t_max/dt is
//                  rounded up the same way, so the time grid ends exactly at t_max
//     disentangle: bool (default true). The Karrasch-Bardarson-Moore backward ancilla
//                  evolution: exact without truncation, and it slows the bond growth
//
// Observables are in the engine's units. O_j is the site's charge: 2 Sz_j for the spin
// chains (the Pauli Z for spin-1/2), n_j for hubbard.
// Returns thread-local JSON with ok, quantity, n_sites, max_bond, wall_time_s, and
//   statics: betas[], energy[] (<H>, whole chain), O[beta][j] (<O_j>), OO[beta][j]
//     (<O_c O_j>, c = n_sites/2), susceptibility[] (beta Var(sum_j O_j) / n_sites; zeros
//     without "susceptibility"), center (c)
//   structure_factor: times[], C_re[t][j], C_im[t][j], C_{j,c}(t) = <S_j(t) S_c(0)>_beta
//     with S = 2 Sz (hubbard too: the spin structure factor), source (c), beta, energy,
//     bond_t[]
//   current: times[], C_re[t], C_im[t], C(t) = <J(t) J(0)>_beta, beta, energy, bond_t[],
//     with J = sum_l i J (s+_l s-_{l+1} - h.c.) for the spins, J the coupling as sent and
//     s+ the raising operator (|up><dn| for spin-1/2, S+ for spin-1), and
//     J = sum_{l,s} i t (c+_{l s} c_{l+1 s} - h.c.) for hubbard
QX_API const char* qx_thermal(const char* problem_json, const char* options_json);

// ── M16: qubit circuits that conserve the number of |1>s ─────────────────────
// One-qubit phases and nearest-neighbour two-qubit gates on the block-sparse MPS. Every
// bond carries the number of qubits in |1> to its left as an exact charge, so the total
// is conserved at any truncation. Qubit |0> is Z = +1.
//
// problem_json:
//   n_qubits      : int >= 2                                              (required)
//   initial       : [0|1, ...], one bit per qubit (default all 0)
//   gates         : [{q, u}, ...], in order. u is a row-major matrix of [re, im] entries:
//                   4 entries, a phase on qubit q (it must be diagonal), or 16, a gate on
//                   qubits (q, q+1) in the basis |00>, |01>, |10>, |11> with qubit q the
//                   high bit. Each must be unitary and charge-conserving to 1e-10, or the
//                   call fails naming it.
//   measure_after : [int, ...] (default [len(gates)]): measure after this many gates, each
//                   count once and in increasing order; the whole circuit runs either way
// options_json:
//   max_bond : int (default 256), cutoff: float (default 1e-10; the relative discarded
//              norm, as everywhere in the engine)
// Unknown keys are refused. Returns thread-local JSON: {ok, n_qubits, ones (the conserved
//   number of |1>s, read off the MPS), measurements: [{after, z[] (<Z_i>, normalized),
//   norm2 (the squared norm: the weight the truncations left), max_bond}], max_bond,
//   truncation_sum, truncation_max, wall_time_s}.
QX_API const char* qx_u1_circuit(const char* problem_json, const char* options_json);

// ── M11: matrix-product states that stay in the engine ──────────────────────────
// A block-sparse MPS lives in the library between calls, under an integer handle, so a
// caller chains a ground state, one-site operators, time evolution and measurements
// without the state passing through JSON. One request per call, {"op": ..., ...}:
//   ground_state {model, max_bond?, cutoff?, n_sweeps?, tol?}  -> {handle, energy,
//                 truncation_error, n_sweeps_done, sweep_energies, sector, method, n_sites}
//   product      {model, config: [local state per site]}       -> {handle}
//   copy         {handle}                                     -> {handle}
//   apply        {handle, site, diag | matrix, string?}       -> {norm2}   (in place; not
//                 renormalized; refused if it annihilates the state)
//   evolve       {handle, model, dt, n_steps, imaginary?, max_bond?, cutoff?}
//                 -> {max_bond, wall_time_s}   (TDVP2, in place; exp(-i H dt) per step, or
//                 exp(-H dt) renormalized when imaginary)
//   overlap      {bra, ket}                                   -> {re, im}  (<bra|ket>)
//   expect       {bra, ket?, diag | matrix, string?}          -> {re: [..], im: [..]}
//                 (<bra| P_0 .. P_{j-1} O_j |ket> for every site j; ket defaults to bra)
// A one-site operator O is its real diagonal ("diag", one entry per local state), or its
// row-major d x d matrix ("matrix", entries <s'|O|s> with row s' the result, each a number
// or an [re, im] pair), optionally with a "string": a diagonal of phases P applied on every
// site left of O, such as the Jordan-Wigner parity (-1)^n of a fermion operator. A matrix
// may change the charge, by one amount for all its entries: apply moves the state to that
// sector, and expect is zero unless charge(bra) = charge(ket) + the shift.
//   energy       {handle, model}                              -> {energy}  (<H>/<psi|psi>)
//   info         {handle}  -> {n_sites, local_dim, bonds, norm2, charge}
//   free         {handle}
//   live         {}        -> {handles}  (how many states are alive)
// A model is the object qx_dmrg_sweep takes as its problem (type, n_sites, couplings,
// sector), for a block-sparse type; evolve and energy may take another model of the same
// length and local space than the state's (a quench). Local states are in the engine's
// order: spin 1/2 (up, down), spin 1 (+1, 0, -1), Hubbard (empty, up, down, double).
// Unknown keys and handles are refused. Returns thread-local JSON with "ok".
QX_API const char* qx_mps(const char* request_json);

#ifdef __cplusplus
}
#endif
