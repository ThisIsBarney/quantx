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

// C entrypoint for unified solver dispatch.
extern "C" const char* qx_run_solver(const char* solver, const char* opts_json) {
  try {
    const std::string s = solver ? solver : "";
    const std::string o = opts_json ? opts_json : "{}";
    if (s == "vqe")              return copy_to_cstr(qx::vqe_result_to_json(qx::run_vqe(qx::vqe_options_from_json(o))));
    if (s == "qpe")              return copy_to_cstr(qx::qpe_result_to_json(qx::run_qpe(qx::qpe_options_from_json(o))));
    if (s == "qec")              return copy_to_cstr(qx::qec_result_to_json(qx::run_qec(qx::qec_options_from_json(o))));
    if (s == "qfield")           return copy_to_cstr(qx::qfield_result_to_json(qx::run_qfield(qx::qfield_options_from_json(o))));
    if (s == "phi4")             return copy_to_cstr(qx::phi4_result_to_json(qx::run_phi4(qx::phi4_options_from_json(o))));
    if (s == "hubbard")          return copy_to_cstr(qx::hubbard_result_to_json(qx::run_hubbard(qx::hubbard_options_from_json(o))));
    if (s == "floquet")          return copy_to_cstr(qx::floquet_result_to_json(qx::run_floquet(qx::floquet_options_from_json(o))));
    if (s == "heisenberg")       return copy_to_cstr(qx::heisenberg_result_to_json(qx::run_heisenberg(qx::heisenberg_options_from_json(o))));
    if (s == "bose_hubbard")     return copy_to_cstr(qx::bose_hubbard_result_to_json(qx::run_bose_hubbard(qx::bose_hubbard_options_from_json(o))));
    if (s == "tfi")              return copy_to_cstr(qx::tfi_result_to_json(qx::run_tfi(qx::tfi_options_from_json(o))));
    if (s == "ssh")              return copy_to_cstr(qx::ssh_result_to_json(qx::run_ssh(qx::ssh_options_from_json(o))));
    if (s == "kitaev")           return copy_to_cstr(qx::kitaev_result_to_json(qx::run_kitaev(qx::kitaev_options_from_json(o))));
    if (s == "anderson")         return copy_to_cstr(qx::anderson_result_to_json(qx::run_anderson(qx::anderson_options_from_json(o))));
    if (s == "chern")            return copy_to_cstr(qx::chern_result_to_json(qx::run_chern(qx::chern_options_from_json(o))));
    if (s == "hofstadter")       return copy_to_cstr(qx::hofstadter_result_to_json(qx::run_hofstadter(qx::hofstadter_options_from_json(o))));
    if (s == "mbl")              return copy_to_cstr(qx::mbl_result_to_json(qx::run_mbl(qx::mbl_options_from_json(o))));
    if (s == "spectral")         return copy_to_cstr(qx::spectral_result_to_json(qx::run_spectral(qx::spectral_options_from_json(o))));
    if (s == "tenfold")          return copy_to_cstr(qx::tenfold_result_to_json(qx::run_tenfold(qx::tenfold_options_from_json(o))));
    if (s == "second_chern")     return copy_to_cstr(qx::second_chern_result_to_json(qx::run_second_chern(qx::second_chern_options_from_json(o))));
    if (s == "wilson_loop")      return copy_to_cstr(qx::wilson_loop_result_to_json(qx::run_wilson_loop(qx::wilson_loop_options_from_json(o))));
    if (s == "edge_states")      return copy_to_cstr(qx::edge_states_result_to_json(qx::run_edge_states(qx::edge_states_options_from_json(o))));
    if (s == "haldane")          return copy_to_cstr(qx::haldane_result_to_json(qx::run_haldane(qx::haldane_options_from_json(o))));
    if (s == "weyl")             return copy_to_cstr(qx::weyl_result_to_json(qx::run_weyl(qx::weyl_options_from_json(o))));
    if (s == "fu_kane")          return copy_to_cstr(qx::fu_kane_result_to_json(qx::run_fu_kane(qx::fu_kane_options_from_json(o))));
    if (s == "lutchyn_oreg")     return copy_to_cstr(qx::lutchyn_oreg_result_to_json(qx::run_lutchyn_oreg(qx::lutchyn_oreg_options_from_json(o))));
    if (s == "kitaev_honeycomb") return copy_to_cstr(qx::kitaev_hc_result_to_json(qx::run_kitaev_honeycomb(qx::kitaev_hc_options_from_json(o))));
    if (s == "bbh")              return copy_to_cstr(qx::bbh_result_to_json(qx::run_bbh(qx::bbh_options_from_json(o))));
    if (s == "floquet_topo")     return copy_to_cstr(qx::floquet_topo_result_to_json(qx::run_floquet_topo(qx::floquet_topo_options_from_json(o))));
    if (s == "non_herm")         return copy_to_cstr(qx::non_herm_result_to_json(qx::run_non_herm(qx::non_herm_options_from_json(o))));
    if (s == "aah")              return copy_to_cstr(qx::aah_result_to_json(qx::run_aah(qx::aah_options_from_json(o))));
    if (s == "syk")              return copy_to_cstr(qx::syk_result_to_json(qx::run_syk(qx::syk_options_from_json(o))));
    if (s == "kondo")            return copy_to_cstr(qx::kondo_result_to_json(qx::run_kondo(qx::kondo_options_from_json(o))));
    if (s == "luttinger")        return copy_to_cstr(qx::ll_result_to_json(qx::run_luttinger(qx::ll_options_from_json(o))));
    if (s == "toric")            return copy_to_cstr(qx::toric_result_to_json(qx::run_toric(qx::toric_options_from_json(o))));
    if (s == "mipt")             return copy_to_cstr(qx::mipt_result_to_json(qx::run_mipt(qx::mipt_options_from_json(o))));
    if (s == "rg")               return copy_to_cstr(qx::rg_result_to_json(qx::run_rg(qx::rg_options_from_json(o))));
    if (s == "entanglement")     return copy_to_cstr(qx::ent_result_to_json(qx::run_entanglement(qx::ent_options_from_json(o))));
    return copy_to_cstr(wrap_error("runtime", "QX_UNKNOWN_SOLVER", "unknown solver: " + s));
  } catch (const qx::CircuitError& ex) {
    return copy_to_cstr(wrap_error("circuit", ex.code(), ex.what()));
  } catch (const std::exception& ex) {
    return copy_to_cstr(wrap_error("runtime", "QX_RUNTIME_ERROR", ex.what()));
  } catch (...) {
    return copy_to_cstr(wrap_error("unknown", "QX_UNKNOWN_ERROR", "unknown exception"));
  }
}

// C entrypoint returning JSON array of all named solver identifiers.
extern "C" const char* qx_list_solvers(void) {
  try {
    nlohmann::json j = nlohmann::json::array({
        "vqe", "qpe", "qec", "qfield", "phi4", "hubbard", "floquet",
        "heisenberg", "bose_hubbard", "tfi", "ssh", "kitaev", "anderson",
        "chern", "hofstadter", "mbl", "spectral", "tenfold", "second_chern",
        "wilson_loop", "edge_states", "haldane", "weyl", "fu_kane",
        "lutchyn_oreg", "kitaev_honeycomb", "bbh", "floquet_topo",
        "non_herm", "aah", "syk", "kondo", "luttinger", "toric",
        "mipt", "rg", "entanglement"
    });
    return copy_to_cstr(j.dump());
  } catch (...) {
    return copy_to_cstr(wrap_error("unknown", "QX_UNKNOWN_ERROR", "unknown exception"));
  }
}
