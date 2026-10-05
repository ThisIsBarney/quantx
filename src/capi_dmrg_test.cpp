// Contract test for the exported DMRG / dynamics entry points, called the way a consumer
// calls them: through the public header and the shared library (quantx.dll /
// libquantx.so), never by compiling the sources in.
//
// The block-sparse engine refuses chains shorter than 3 by design -- derive_mpo_sym needs a
// bulk MPO site (qx_mpo_sym_test asserts the throw). The entry points used to accept
// n_qubits >= 2 and hand that internal refusal straight back to the caller, and
// qx_dmrg_test, still asking the block path for n=2, died in std::terminate: 0xC0000409 on
// Windows, indistinguishable from a real stack overrun. This pins down what each entry
// point does now for a 2-site chain, and that n=4 still takes the fast path.
#include "qx/c_api.h"

#include <cmath>
#include <cstdio>
#include <string>

#include <nlohmann/json.hpp>

using nlohmann::json;

static int failures = 0;

static void check(bool ok, const char* what, const json& got) {
    std::printf("  %-58s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) {
        ++failures;
        std::printf("      got: %s\n", got.dump().substr(0, 300).c_str());
    }
}

static json call(const char* (*fn)(const char*, const char*), const json& prob, const json& opts) {
    return json::parse(fn(prob.dump().c_str(), opts.dump().c_str()));
}

static json heis(int n) { return {{"type", "heisenberg"}, {"n_qubits", n}, {"J", 1.0}, {"delta", 1.0}}; }

static bool says_needs_3(const json& r) {
    const std::string e = r.value("error", r.value("error_message", std::string()));
    return !r.value("ok", true) && e.find("n_qubits must be >= 3") != std::string::npos;
}

int main() {
    // H = sum (XX + YY + ZZ) in Pauli matrices: the 2-site singlet sits at -3, the open
    // 4-site chain at -3 - 2 sqrt(3), and the Neel state has <H> = -(n-1).
    const double e2 = -3.0, e4 = -3.0 - 2.0 * std::sqrt(3.0);

    std::printf("qx_dmrg_sweep\n");
    json r = call(qx_dmrg_sweep, heis(2), json::object());
    check(r.value("ok", false) && r.value("method", "") == std::string("dense") &&
          std::abs(r.value("energy", 0.0) - e2) < 1e-10,
          "n=2: dense solver, singlet energy -3", r);
    r = call(qx_dmrg_sweep, heis(4), json::object());
    check(r.value("ok", false) && r.value("method", "") == std::string("u1") &&
          std::abs(r.value("energy", 0.0) - e4) < 1e-8,
          "n=4: block-sparse u1 solver, -3 - 2 sqrt(3)", r);

    std::printf("qx_tdvp_evolve (real time from the Neel state)\n");
    const json evo = {{"n_steps", 4}, {"chi_max", 16}, {"dt", 0.05}};
    r = call(qx_tdvp_evolve, heis(2), evo);
    // unitary evolution conserves <H>: the Neel value -1 must survive the fallback path
    check(r.value("ok", false) && r.value("method", "").find("block") == std::string::npos &&
          std::abs(r.value("energy", 0.0) - (-1.0)) < 1e-8,
          "n=2: falls back to non-block TDVP, <H> stays -1", r);
    r = call(qx_tdvp_evolve, heis(4), evo);
    check(r.value("ok", false) && r.value("method", "").find("block") != std::string::npos,
          "n=4: block-sparse path", r);

    const json dyn = {{"n_steps", 4}, {"chi_max", 16}, {"n_sweeps", 4}};
    struct Entry { const char* name; const char* (*fn)(const char*, const char*); };
    const Entry block_only[] = {
        {"qx_spectral", qx_spectral},
        {"qx_loschmidt", qx_loschmidt},
        {"qx_conductivity", qx_conductivity},
    };
    for (const Entry& e : block_only) {
        std::printf("%s (block-sparse only)\n", e.name);
        r = call(e.fn, heis(2), dyn);
        check(says_needs_3(r), "n=2: refused in the caller's terms (n_qubits >= 3)", r);
        r = call(e.fn, heis(4), dyn);
        check(r.value("ok", false), "n=4: runs", r);
    }

    std::printf("\n%s\n", failures ? "CAPI DMRG TEST: FAIL" : "CAPI DMRG TEST: ALL OK");
    return failures ? 1 : 0;
}
