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
QX_API const char* qx_run(const char* problem_json, const char* options_json);
// Explain backend choice from JSON inputs and return heap-allocated JSON output.
QX_API const char* qx_explain_backend_choice(const char* problem_json, const char* options_json);
// Run circuit simulation from JSON inputs and return heap-allocated JSON output.
QX_API const char* qx_run_circuit(const char* circuit_json, const char* options_json);
// Free strings returned by qx_run/qx_explain_backend_choice/qx_run_circuit.
QX_API void qx_free_string(const char* ptr);

#ifdef __cplusplus
}
#endif
