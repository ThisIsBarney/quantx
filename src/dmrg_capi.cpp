#include "qx/c_api.h"
#include "dmrg_internal.hpp"
#include "dmrg_su2_run.hpp"      // SU(2)-symmetric Heisenberg DMRG (header-only)
#include "tdvp.hpp"              // TDVP MPS time evolution (header-only)

#include <chrono>
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
        double J     = prob.value("J", 1.0);
        double delta = prob.value("delta", 1.0);   // XXZ anisotropy (1 = Heisenberg)
        return build_xxz_mpo(n, J, delta);
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

        std::string type   = prob.value("type", "pauli_sum");
        std::string method = opts.value("method", "u1");  // "u1" (Sz blocks) | "su2"

        // SU(2)-symmetric path: Heisenberg only. max_bond is the multiplet cap;
        // the energy is in the same J*sum sigma.sigma convention as the U(1) path.
        if (method == "su2") {
            if (type != "heisenberg")
                return dmrg_error("method 'su2' supports only type 'heisenberg'");
            double J  = prob.value("J", 1.0);
            int seed  = opts.value("seed", 7);
            double S  = prob.value("spin", 0.5);     // physical site spin (0.5, 1, ...)
            int tS    = (int)std::lround(2.0 * S);
            if (tS < 1) return dmrg_error("spin must be >= 1/2");
            auto t0   = std::chrono::steady_clock::now();
            double E  = qx_su2::run_su2_dmrg(n, J, max_bond, n_sweeps, (unsigned)seed, tS);
            double wt = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
            nlj out;
            out["ok"]               = true;
            out["energy"]           = E;
            out["truncation_error"] = 0.0;
            out["n_sweeps_done"]    = n_sweeps;
            out["wall_time_s"]      = wt;
            out["sweep_energies"]   = nlj::array();
            out["method"]           = "su2";
            out["spin"]             = S;
            tl_dmrg_buf = out.dump();
            return tl_dmrg_buf.c_str();
        }

        auto mpo = parse_ham(prob, n);
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
        out["method"]           = "u1";

        tl_dmrg_buf = out.dump();
        return tl_dmrg_buf.c_str();
    } catch (const std::exception& e) {
        return dmrg_error(e.what());
    }
}

QX_API const char* qx_tdvp_evolve(const char* problem_json,
                                  const char* options_json) {
    try {
        nlj prob = nlj::parse(problem_json ? problem_json : "{}");
        nlj opts = nlj::parse(options_json ? options_json : "{}");
        int    n        = prob.value("n_qubits", 8);
        double J        = prob.value("J", 1.0);
        double delta    = prob.value("delta", 1.0);   // XXZ anisotropy (1 = Heisenberg)
        std::string type = prob.value("type", "heisenberg");
        if (type != "heisenberg") return dmrg_error("tdvp supports only type 'heisenberg'/'xxz'");
        if (n < 2)  return dmrg_error("n_qubits must be >= 2");

        double dt       = opts.value("dt", 0.05);
        int    n_steps  = opts.value("n_steps", 20);
        int    chi_max  = opts.value("chi_max", 64);
        bool   imag     = opts.value("imag", false);
        bool   two_site = opts.value("two_site", true);
        std::string init = opts.value("initial", "neel");
        // Block-sparse (Sz-symmetric) TDVP2 is the default fast path (Neel init).
        bool   block    = opts.value("block", true) && two_site && init == "neel";

        auto W = qx_dmrg::build_xxz_mpo(n, J, delta);

        if (block) {
            // Block evolution engine: "tebd" (Trotter gates, fast NN -- default) or
            // "tdvp2" (Krylov, needed for general/long-range MPOs).
            std::string engine = opts.value("engine", "tebd");
            int stride = opts.value("measure_stride", 1);
            auto r = (engine == "tdvp2")
                ? qx_dmrg::run_tdvp2_blocks(W, n, dt, n_steps, chi_max, 1e-12, imag)
                : qx_dmrg::run_tebd2_blocks(W, n, dt, n_steps, chi_max, 1e-12, imag, stride, J, delta);
            nlj out;
            out["ok"]            = true;
            out["energy"]        = r.energy;
            out["times"]         = r.times;
            out["staggered_mag"] = r.stag;
            out["max_bond"]      = r.max_bond;
            out["wall_time_s"]   = r.wall_time_s;
            out["method"]        = (engine == "tdvp2") ? "tdvp2-block" : "tebd2-block";
            tl_dmrg_buf = out.dump();
            return tl_dmrg_buf.c_str();
        }
        std::vector<qx_tdvp::Mat> A = qx_tdvp::neel_state(n);
        if (init == "up") for (int i=0;i<n;++i) { A[i]=qx_tdvp::Mat::Zero(2,1); A[i](0,0)=1; }
        qx_tdvp::right_canonicalize(A);
        if (imag) { double nz=std::sqrt(qx_tdvp::mps_norm2(A)); if (nz>0) A[0]/=nz; }

        auto stagmag = [&]() {
            auto sz = qx_tdvp::mps_sz(A);
            double m = 0; for (int i=0;i<n;++i) m += ((i%2==0)?1.0:-1.0) * sz[i];
            return m / n;
        };
        std::vector<double> times, stag;
        times.push_back(0.0); stag.push_back(stagmag());

        auto t0 = std::chrono::steady_clock::now();
        for (int step=0; step<n_steps; ++step) {
            if (two_site) qx_tdvp::tdvp2_step(A, W, dt, imag, chi_max, 1e-12);
            else          qx_tdvp::tdvp_step (A, W, dt, imag);
            times.push_back((step+1)*dt);
            stag.push_back(stagmag());
        }
        double wt = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();

        int max_bond = 0; for (auto& a : A) max_bond = std::max(max_bond, (int)a.cols());
        nlj out;
        out["ok"]              = true;
        out["energy"]          = qx_tdvp::mps_energy(A, W);
        out["sz"]              = qx_tdvp::mps_sz(A);
        out["times"]           = times;
        out["staggered_mag"]   = stag;
        out["max_bond"]        = max_bond;
        out["wall_time_s"]     = wt;
        out["method"]          = two_site ? "tdvp2" : "tdvp1";
        tl_dmrg_buf = out.dump();
        return tl_dmrg_buf.c_str();
    } catch (const std::exception& e) {
        return dmrg_error(e.what());
    }
}

QX_API const char* qx_spectral(const char* problem_json, const char* options_json) {
    try {
        nlj prob = nlj::parse(problem_json ? problem_json : "{}");
        nlj opts = nlj::parse(options_json ? options_json : "{}");
        int    n  = prob.value("n_qubits", 8);
        double J  = prob.value("J", 1.0);
        double delta = prob.value("delta", 1.0);   // XXZ anisotropy (1 = Heisenberg)
        std::string type = prob.value("type", "heisenberg");
        if (type != "heisenberg") return dmrg_error("spectral supports only type 'heisenberg'/'xxz'");
        if (n < 2) return dmrg_error("n_qubits must be >= 2");

        double dt      = opts.value("dt", 0.05);
        int    n_steps = opts.value("n_steps", 200);
        int    chi_max = opts.value("chi_max", 128);
        double cutoff  = opts.value("cutoff", 1e-10);
        int    n_sweeps = opts.value("n_sweeps", 40);

        auto W = qx_dmrg::build_xxz_mpo(n, J, delta);
        auto r = qx_dmrg::run_spectral_blocks(W, n, dt, n_steps, chi_max, cutoff, n_sweeps, J, delta);
        nlj out;
        out["ok"]          = true;
        out["times"]       = r.times;
        out["l0"]          = r.l0;
        out["n"]           = r.n;
        out["E0"]          = r.E0;
        out["max_bond"]    = r.max_bond;
        out["wall_time_s"] = r.wall_time_s;
        out["C_re"]        = r.C_re;     // C_re[t][j]
        out["C_im"]        = r.C_im;
        tl_dmrg_buf = out.dump();
        return tl_dmrg_buf.c_str();
    } catch (const std::exception& e) {
        return dmrg_error(e.what());
    }
}

QX_API const char* qx_loschmidt(const char* problem_json, const char* options_json) {
    try {
        nlj prob = nlj::parse(problem_json ? problem_json : "{}");
        nlj opts = nlj::parse(options_json ? options_json : "{}");
        int    n  = prob.value("n_qubits", 8);
        double J  = prob.value("J", 1.0);
        double delta = prob.value("delta", 1.0);     // initial XXZ anisotropy
        std::string type = prob.value("type", "heisenberg");
        if (type != "heisenberg") return dmrg_error("loschmidt supports only type 'heisenberg'/'xxz'");
        if (n < 2) return dmrg_error("n_qubits must be >= 2");

        double delta_f = opts.value("delta_f", 1.0);  // quench: evolve under XXZ(J, delta_f)
        double dt      = opts.value("dt", 0.05);
        int    n_steps = opts.value("n_steps", 200);
        int    chi_max = opts.value("chi_max", 128);
        double cutoff  = opts.value("cutoff", 1e-10);
        int    n_sweeps = opts.value("n_sweeps", 40);

        auto Wi = qx_dmrg::build_xxz_mpo(n, J, delta);     // initial Hamiltonian
        auto r  = qx_dmrg::run_loschmidt_blocks(Wi, n, dt, n_steps, chi_max, cutoff,
                                                n_sweeps, J, delta_f);
        nlj out;
        out["ok"]          = true;
        out["times"]       = r.times;
        out["L"]           = r.L;
        out["rate"]        = r.rate;
        out["E0"]          = r.E0;
        out["n"]           = r.n;
        out["max_bond"]    = r.max_bond;
        out["wall_time_s"] = r.wall_time_s;
        out["delta_i"]     = delta;
        out["delta_f"]     = delta_f;
        tl_dmrg_buf = out.dump();
        return tl_dmrg_buf.c_str();
    } catch (const std::exception& e) {
        return dmrg_error(e.what());
    }
}

QX_API const char* qx_conductivity(const char* problem_json, const char* options_json) {
    try {
        nlj prob = nlj::parse(problem_json ? problem_json : "{}");
        nlj opts = nlj::parse(options_json ? options_json : "{}");
        int    n  = prob.value("n_qubits", 8);
        double J  = prob.value("J", 1.0);
        double delta = prob.value("delta", 1.0);
        std::string type = prob.value("type", "heisenberg");
        if (type != "heisenberg") return dmrg_error("conductivity supports only type 'heisenberg'/'xxz'");
        if (n < 2) return dmrg_error("n_qubits must be >= 2");

        double dt      = opts.value("dt", 0.0125);
        int    n_steps = opts.value("n_steps", 400);
        int    chi_max = opts.value("chi_max", 128);
        double cutoff  = opts.value("cutoff", 1e-10);
        int    n_sweeps = opts.value("n_sweeps", 40);

        auto W = qx_dmrg::build_xxz_mpo(n, J, delta);
        auto r = qx_dmrg::run_conductivity_blocks(W, n, dt, n_steps, chi_max, cutoff,
                                                  n_sweeps, J, delta);
        nlj out;
        out["ok"]          = true;
        out["times"]       = r.times;
        out["C_re"]        = r.C_re;
        out["C_im"]        = r.C_im;
        out["E0"]          = r.E0;
        out["n"]           = r.n;
        out["max_bond"]    = r.max_bond;
        out["wall_time_s"] = r.wall_time_s;
        out["delta"]       = delta;
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
        MPS mps  = init_zero_state(n, mpo[0].d);
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
