#include "qx/c_api.h"

#include <cstring>
#include <exception>
#include <string>

#include "qx/engine.hpp"
#include "qx/json_utils.hpp"

namespace {

// Copy std::string payload to C-ABI heap string.
const char* copy_to_cstr(const std::string& s) {
  char* out = new char[s.size() + 1];
  std::memcpy(out, s.c_str(), s.size() + 1);
  return out;
}

// Wrap exception message into a minimal JSON error object.
std::string wrap_error(const std::string& message) {
  return std::string("{\"ok\":false,\"error\":\"") + message + "\"}";
}

}  // namespace

// C entrypoint for Hamiltonian simulation.
extern "C" const char* qx_run(const char* problem_json, const char* options_json) {
  try {
    const std::string p = problem_json ? problem_json : "{}";
    const std::string o = options_json ? options_json : "{}";
    auto result = qx::run(qx::problem_from_json(p), qx::options_from_json(o));
    return copy_to_cstr(qx::result_to_json(result));
  } catch (const std::exception& ex) {
    return copy_to_cstr(wrap_error(ex.what()));
  } catch (...) {
    return copy_to_cstr(wrap_error("unknown exception"));
  }
}

// C entrypoint for backend-selection explanation.
extern "C" const char* qx_explain_backend_choice(const char* problem_json, const char* options_json) {
  try {
    const std::string p = problem_json ? problem_json : "{}";
    const std::string o = options_json ? options_json : "{}";
    auto info = qx::explain_backend_choice(qx::problem_from_json(p), qx::options_from_json(o));
    return copy_to_cstr(qx::backend_info_to_json(info));
  } catch (const std::exception& ex) {
    return copy_to_cstr(wrap_error(ex.what()));
  } catch (...) {
    return copy_to_cstr(wrap_error("unknown exception"));
  }
}

// C entrypoint for circuit execution.
extern "C" const char* qx_run_circuit(const char* circuit_json, const char* options_json) {
  try {
    const std::string c = circuit_json ? circuit_json : "{}";
    const std::string o = options_json ? options_json : "{}";
    auto result = qx::run_circuit(qx::circuit_from_json(c), qx::options_from_json(o));
    return copy_to_cstr(qx::circuit_result_to_json(result));
  } catch (const std::exception& ex) {
    return copy_to_cstr(wrap_error(ex.what()));
  } catch (...) {
    return copy_to_cstr(wrap_error("unknown exception"));
  }
}

// Free C strings returned by C API calls.
extern "C" void qx_free_string(const char* ptr) {
  delete[] ptr;
}
