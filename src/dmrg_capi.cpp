#include "qx/c_api.h"
#include "dmrg_internal.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// Bring internal types into scope
using namespace qx_dmrg;

namespace {

static thread_local std::string tl_dmrg_buf;

static const char* dmrg_error(const std::string& msg) {
    nlj out;
    out["ok"] = false;
    out["error"] = msg;
    tl_dmrg_buf = out.dump();
    return tl_dmrg_buf.c_str();
}

// Parse Hamiltonian from JSON into MPO
static std::vector<MPOSite> parse_ham(const nlj& prob, int n) {
    std::string type = prob.value("type", "pauli_sum");
    if (type == "heisenberg") {
        double J = prob.value("J", 1.0);
        return build_heisenberg_mpo(n, J);
    }
    // pauli_sum
    std::vector<std::pair<double, std::string>> terms;
    if (prob.contains("terms")) {
        for (const auto& t : prob["terms"]) {
            double c  = t.value("coeff", 1.0);
            std::string p = t.value("pauli", "I");
            terms.emplace_back(c, p);
        }
    } else if (prob.contains("hamiltonian_terms")) {
        for (const auto& t : prob["hamiltonian_terms"]) {
            double c = t[0].get<double>();
            std::string p = t[1].get<std::string>();
            terms.emplace_back(c, p);
        }
    }
    return build_pauli_sum_mpo(n, terms);
}

}  // namespace

extern "C" {

QX_API const char* qx_dmrg_sweep(const char* problem_json,
                                  const char* options_json) {
    try {
        nlj prob  = nlj::parse(problem_json  ? problem_json  : "{}");
        nlj opts  = nlj::parse(options_json  ? options_json  : "{}");

        int  n        = prob.value("n_qubits",  8);
        int  max_bond = opts.value("max_bond",  64);
        double cutoff = opts.value("cutoff",    1e-10);
        int  n_sweeps = opts.value("n_sweeps",  10);
        double tol    = opts.value("tol",       1e-10);

        if (n < 2)        return dmrg_error("n_qubits must be >= 2");
        if (n > 200)      return dmrg_error("n_qubits exceeds max (200)");
        if (n_sweeps < 1) return dmrg_error("n_sweeps must be >= 1");

        auto mpo = parse_ham(prob, n);
        std::string type = prob.value("type", "pauli_sum");
        DMRGResult res = (type == "heisenberg")
            ? run_dmrg_blocks(mpo, n, max_bond, cutoff, n_sweeps, tol)
            : run_dmrg(mpo, n, max_bond, cutoff, n_sweeps, tol);

        nlj out;
        out["ok"]               = true;
        out["energy"]           = res.energy;
        out["truncation_error"] = res.truncation_error;
        out["n_sweeps_done"]    = res.n_sweeps_done;
        out["wall_time_s"]      = res.wall_time_s;
        out["sweep_energies"]   = res.sweep_energies;

        tl_dmrg_buf = out.dump();
        return tl_dmrg_buf.c_str();
    } catch (const std::exception& e) {
        return dmrg_error(e.what());
    }
}

QX_API const char* qx_mps_energy(const char* problem_json) {
    try {
        nlj prob = nlj::parse(problem_json ? problem_json : "{}");
        int n    = prob.value("n_qubits", 8);
        auto mpo = parse_ham(prob, n);
        MPS mps  = init_zero_state(n);
        double E = compute_energy(mps, mpo);
        nlj out;
        out["ok"]     = true;
        out["energy"] = E;
        tl_dmrg_buf = out.dump();
        return tl_dmrg_buf.c_str();
    } catch (const std::exception& e) {
        return dmrg_error(e.what());
    }
}

}  // extern "C"
