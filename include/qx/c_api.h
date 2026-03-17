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

QX_API const char* qx_run(const char* problem_json, const char* options_json);
QX_API const char* qx_explain_backend_choice(const char* problem_json, const char* options_json);
QX_API const char* qx_run_circuit(const char* circuit_json, const char* options_json);
QX_API void qx_free_string(const char* ptr);

#ifdef __cplusplus
}
#endif
