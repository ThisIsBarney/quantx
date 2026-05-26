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

#ifdef __cplusplus
}
#endif
