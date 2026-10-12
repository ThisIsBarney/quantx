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

    // ---- symmetric models and sectors (M4) ------------------------------------
    // Each model runs in a U(1) sector the caller names, from a product state in that
    // sector. Every odd Heisenberg chain used to fail: the Neel start sits at 2Sz = +1
    // while the engine targeted 0 ("sector missing from the bond-space guess").
    std::printf("\nqx_dmrg_sweep: symmetric models and sectors\n");
    const json tight = {{"max_bond", 128}, {"n_sweeps", 40}, {"tol", 1e-13}, {"cutoff", 1e-14}};
    auto energy_of = [&](const json& prob) { return call(qx_dmrg_sweep, prob, tight); };
    {
        // Open 5-site chain, Pauli convention: 4 x the S.S ground energy -1.92788625... (ED).
        json r = energy_of(heis(5));
        check(r.value("ok", false) && r["sector"] == json{{"sz2", 1}} &&
              std::abs(r.value("energy", 0.0) - (-7.711545013271978)) < 1e-9,
              "heisenberg n=5: runs in 2Sz=+1, E matches ED", r);
        json down = heis(5); down["sector"] = {{"sz2", -1}};
        json rd = energy_of(down);
        check(rd.value("ok", false) &&
              std::abs(rd.value("energy", 0.0) - r.value("energy", 1.0)) < 1e-9,
              "heisenberg n=5: E(2Sz=-1) == E(2Sz=+1)", rd);
    }
    {
        // Majumdar-Ghosh, J2 = J1/2: exact dimer product, E = -3L/8 in S.S = -12 at L=8
        // in the Pauli convention.
        json r = energy_of({{"type", "j1j2"}, {"n_sites", 8}, {"J1", 1.0}, {"J2", 0.5}});
        check(r.value("ok", false) && std::abs(r.value("energy", 0.0) + 12.0) < 1e-9,
              "j1j2 Majumdar-Ghosh n=8: E = -12", r);
    }
    {
        // Free fermions: 4 up (the full band, sum 0) + the 3 lowest down levels of
        // -2 cos(k pi/5), which sum to -(1 + sqrt 5)/2.
        json r = energy_of({{"type", "hubbard"}, {"n_sites", 4}, {"U", 0.0},
                            {"sector", {{"n_up", 4}, {"n_dn", 3}}}});
        check(r.value("ok", false) && r.value("method", "") == std::string("u1xu1") &&
              std::abs(r.value("energy", 0.0) + (1.0 + std::sqrt(5.0)) / 2.0) < 1e-9,
              "hubbard U=0 (4 up, 3 dn) on 4 sites: E = -(1+sqrt5)/2", r);
        json a = energy_of({{"type", "hubbard"}, {"n_sites", 5}, {"U", 2.0},
                            {"sector", {{"n_up", 2}, {"n_dn", 1}}}});
        json b = energy_of({{"type", "hubbard"}, {"n_sites", 5}, {"U", 2.0},
                            {"sector", {{"n_up", 1}, {"n_dn", 2}}}});
        check(a.value("ok", false) && b.value("ok", false) &&
              std::abs(a.value("energy", 0.0) - b.value("energy", 1.0)) < 1e-9,
              "hubbard: E(n_up, n_dn) == E(n_dn, n_up)", b);
    }
    {
        json r = energy_of({{"type", "spin1_heisenberg"}, {"n_sites", 6}});
        check(r.value("ok", false) && r["sector"] == json{{"sz2", 0}} &&
              r.value("energy", 0.0) < -7.0, "spin1_heisenberg n=6: runs in 2Sz=0", r);
    }
    auto expect_error = [&](const json& prob, const json& opts, const char* text,
                            const char* what) {
        json r = call(qx_dmrg_sweep, prob, opts);
        check(!r.value("ok", true) &&
              r.value("error", std::string()).find(text) != std::string::npos, what, r);
    };
    expect_error({{"type", "xxz"}, {"n_sites", 8}, {"sector", {{"sz2", 1}}}}, tight,
                 "not reachable", "xxz: an unreachable sector is refused");
    expect_error({{"type", "hubbard"}, {"n_sites", 2}}, tight, "n_sites must be >= 3",
                 "hubbard n=2: refused (no dense fallback for a sector)");
    {
        json p = heis(6); p["sector"] = {{"sz2", 2}};
        json o = tight; o["method"] = "su2";
        expect_error(p, o, "a sector needs the block-sparse",
                     "a sector with method su2 is refused, not ignored");
    }
    expect_error({{"type", "ising"}, {"n_sites", 6}}, tight, "unknown type",
                 "an unknown type is an error (it used to run as an empty pauli_sum)");
    // A misspelled or fractional sector key used to leave the default sector in force.
    expect_error({{"type", "xxz"}, {"n_sites", 8}, {"sector", {{"sz", 1}}}}, tight,
                 "unknown sector key 'sz'", "a misspelled sector key is refused");
    expect_error({{"type", "xxz"}, {"n_sites", 8}, {"sector", {{"sz2", 2.5}}}}, tight,
                 "must be an integer", "a fractional sector value is refused");
    {
        // The start state used to alternate spins "while both remain", piling a spin
        // imbalance's excess at the right end: a path the bond-space guess pruned, so
        // this sector failed below max_bond 91.
        json r = call(qx_dmrg_sweep,
                      {{"type", "hubbard"}, {"n_sites", 32}, {"U", 4.0},
                       {"sector", {{"n_up", 23}, {"n_dn", 9}}}},
                      {{"max_bond", 64}, {"n_sweeps", 2}, {"tol", 1e-6}});
        check(r.value("ok", false), "hubbard L=32 (23 up, 9 dn) runs at max_bond 64", r);
    }
    {
        // The other entry points read n_sites too (they used to read only n_qubits, so a
        // 2-site request silently ran the default 8 sites).
        json r = call(qx_spectral, {{"type", "heisenberg"}, {"n_sites", 2}}, json::object());
        check(says_needs_3(r), "qx_spectral reads n_sites (n_sites=2 is refused)", r);
    }

    // ---- the Hubbard lattice and HVA circuits (M5) --------------------------------
    std::printf("\nhubbard_lattice and HVA circuits\n");
    {
        // Two independent builders for one Hamiltonian: the W = 1 lattice (site-split,
        // period-W MPO) must reproduce the chain MPO's ground-state energy.
        json lat = {{"type", "hubbard_lattice"}, {"Lx", 6}, {"W", 1}, {"U", 4.0},
                    {"sector", {{"n_up", 3}, {"n_dn", 2}}}};
        json chn = {{"type", "hubbard"}, {"n_sites", 6}, {"U", 4.0},
                    {"sector", {{"n_up", 3}, {"n_dn", 2}}}};
        json a = energy_of(lat), b = energy_of(chn);
        check(a.value("ok", false) && b.value("ok", false) &&
              std::abs(a.value("energy", 0.0) - b.value("energy", 1.0)) < 1e-9,
              "hubbard_lattice W=1 == hubbard chain (two builders)", a);
    }
    {
        json r = json::parse(qx_hva_layout(json({{"Lx", 6}, {"W", 1}}).dump().c_str()));
        check(r.value("ok", false) && r["groups"].size() == 2 &&
              r.value("angles_per_layer", 0) == 3,
              "qx_hva_layout chain: 2 groups (even, odd bonds), 3 angles per layer", r);
    }
    auto hva_error = [&](const json& prob, const json& opts, const char* text, const char* what) {
        json r = call(qx_hva_circuit, prob, opts);
        check(!r.value("ok", true) &&
              r.value("error", std::string()).find(text) != std::string::npos, what, r);
    };
    {
        // W = 3 cylinder: transverse levels -2, 1, 1, so at half filling the Fermi level
        // sits inside the degenerate pair and the U = 0 start is not unique.
        json cyl = {{"Lx", 2}, {"W", 3}, {"y_periodic", true}, {"U", 4.0},
                    {"sector", {{"n_up", 3}, {"n_dn", 3}}}};
        json lay = json::parse(qx_hva_layout(cyl.dump().c_str()));
        cyl["angles"] = json::array({std::vector<double>(lay.value("angles_per_layer", 1), 0.1)});
        hva_error(cyl, {{"init", "free"}}, "degenerate",
                  "hva: a degenerate U=0 start is refused");
        json r = call(qx_hva_circuit, cyl, {{"init", "product"}});
        check(r.value("ok", false) && r.contains("init_config"),
              "hva: the same cylinder runs from a product start", r);
        cyl["angles"] = json::array({json::array({0.1, 0.2})});
        hva_error(cyl, {{"init", "product"}}, "one angle per bond group",
                  "hva: a layer of the wrong length is refused");
    }
    hva_error({{"Lx", 3}, {"W", 2}, {"y_periodic", true}, {"angles", json::array()}},
              json::object(), "y_periodic needs W >= 3",
              "hva: y_periodic below W=3 is refused");

    std::printf("\n%s\n", failures ? "CAPI DMRG TEST: FAIL" : "CAPI DMRG TEST: ALL OK");
    return failures ? 1 : 0;
}
