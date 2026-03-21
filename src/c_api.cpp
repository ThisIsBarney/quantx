#include "qx/c_api.h"

#include <cstring>
#include <exception>
#include <string>

#include "qx/engine.hpp"
#include "qx/json_utils.hpp"
#include "qx/version.hpp"

#include <nlohmann/json.hpp>

namespace {

// Copy std::string payload to C-ABI heap string.
const char* copy_to_cstr(const std::string& s) {
  char* out = new char[s.size() + 1];
  std::memcpy(out, s.c_str(), s.size() + 1);
  return out;
}

// Wrap exception information into a stable JSON error object.
std::string wrap_error(const std::string& kind, const std::string& code, const std::string& message) {
  nlohmann::json j;
  j["ok"] = false;
  j["error_kind"] = kind;
  j["error_code"] = code;
  j["error_message"] = message;
  j["error"] = message;
  return j.dump();
}

}  // namespace

// C entrypoint for Hamiltonian simulation.
extern "C" const char* qx_run(const char* problem_json, const char* options_json) {
  try {
    const std::string p = problem_json ? problem_json : "{}";
    const std::string o = options_json ? options_json : "{}";
    auto result = qx::run(qx::problem_from_json(p), qx::options_from_json(o));
    return copy_to_cstr(qx::result_to_json(result));
  } catch (const qx::CircuitError& ex) {
    return copy_to_cstr(wrap_error("circuit", ex.code(), ex.what()));
  } catch (const std::exception& ex) {
    return copy_to_cstr(wrap_error("runtime", "QX_RUNTIME_ERROR", ex.what()));
  } catch (...) {
    return copy_to_cstr(wrap_error("unknown", "QX_UNKNOWN_ERROR", "unknown exception"));
  }
}

// C entrypoint for backend-selection explanation.
extern "C" const char* qx_explain_backend_choice(const char* problem_json, const char* options_json) {
  try {
    const std::string p = problem_json ? problem_json : "{}";
    const std::string o = options_json ? options_json : "{}";
    auto info = qx::explain_backend_choice(qx::problem_from_json(p), qx::options_from_json(o));
    return copy_to_cstr(qx::backend_info_to_json(info));
  } catch (const qx::CircuitError& ex) {
    return copy_to_cstr(wrap_error("circuit", ex.code(), ex.what()));
  } catch (const std::exception& ex) {
    return copy_to_cstr(wrap_error("runtime", "QX_RUNTIME_ERROR", ex.what()));
  } catch (...) {
    return copy_to_cstr(wrap_error("unknown", "QX_UNKNOWN_ERROR", "unknown exception"));
  }
}

// C entrypoint for circuit execution.
extern "C" const char* qx_run_circuit(const char* circuit_json, const char* options_json) {
  try {
    const std::string c = circuit_json ? circuit_json : "{}";
    const std::string o = options_json ? options_json : "{}";
    auto result = qx::run_circuit(qx::circuit_from_json(c), qx::options_from_json(o));
    return copy_to_cstr(qx::circuit_result_to_json(result));
  } catch (const qx::CircuitError& ex) {
    return copy_to_cstr(wrap_error("circuit", ex.code(), ex.what()));
  } catch (const std::exception& ex) {
    return copy_to_cstr(wrap_error("runtime", "QX_RUNTIME_ERROR", ex.what()));
  } catch (...) {
    return copy_to_cstr(wrap_error("unknown", "QX_UNKNOWN_ERROR", "unknown exception"));
  }
}

// C entrypoint for circuit capability introspection.
extern "C" const char* qx_circuit_capabilities(void) {
  try {
    return copy_to_cstr(qx::circuit_capabilities_to_json(qx::circuit_backend_capabilities()));
  } catch (const qx::CircuitError& ex) {
    return copy_to_cstr(wrap_error("circuit", ex.code(), ex.what()));
  } catch (const std::exception& ex) {
    return copy_to_cstr(wrap_error("runtime", "QX_RUNTIME_ERROR", ex.what()));
  } catch (...) {
    return copy_to_cstr(wrap_error("unknown", "QX_UNKNOWN_ERROR", "unknown exception"));
  }
}

// C entrypoint for problem JSON validation.
extern "C" const char* qx_validate_problem(const char* problem_json) {
  try {
    const std::string p = problem_json ? problem_json : "{}";
    (void)qx::problem_from_json(p);
    return copy_to_cstr("{\"ok\":true,\"validated\":\"problem\"}");
  } catch (const qx::CircuitError& ex) {
    return copy_to_cstr(wrap_error("circuit", ex.code(), ex.what()));
  } catch (const std::exception& ex) {
    return copy_to_cstr(wrap_error("runtime", "QX_RUNTIME_ERROR", ex.what()));
  } catch (...) {
    return copy_to_cstr(wrap_error("unknown", "QX_UNKNOWN_ERROR", "unknown exception"));
  }
}

// C entrypoint for circuit JSON validation.
extern "C" const char* qx_validate_circuit(const char* circuit_json) {
  try {
    const std::string c = circuit_json ? circuit_json : "{}";
    (void)qx::circuit_from_json(c);
    return copy_to_cstr("{\"ok\":true,\"validated\":\"circuit\"}");
  } catch (const qx::CircuitError& ex) {
    return copy_to_cstr(wrap_error("circuit", ex.code(), ex.what()));
  } catch (const std::exception& ex) {
    return copy_to_cstr(wrap_error("runtime", "QX_RUNTIME_ERROR", ex.what()));
  } catch (...) {
    return copy_to_cstr(wrap_error("unknown", "QX_UNKNOWN_ERROR", "unknown exception"));
  }
}

// Free C strings returned by C API calls.
extern "C" void qx_free_string(const char* ptr) {
  delete[] ptr;
}

// Return engine semantic version string for integration diagnostics.
extern "C" const char* qx_version(void) {
  return qx::version_string().c_str();
}
