#include "qx/c_api.h"
#include "dmrg_internal.hpp"
#include "dmrg_su2_run.hpp"      // SU(2)-symmetric Heisenberg DMRG (header-only)
#include "tdvp.hpp"              // TDVP MPS time evolution (header-only)

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
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

// Chain length: "n_sites", or the legacy "n_qubits" (default 8).
static int read_n(const nlj& prob) {
    if (prob.contains("n_sites")) return prob["n_sites"].get<int>();
    return prob.value("n_qubits", 8);
}

// ---- Models the block-sparse engine runs -----------------------------------
// Each resolves to its MPO, the U(1) sector to target, and a product state in that
// sector to start from. Conventions are the MPO builders' own:
//   heisenberg, xxz   J sum_i [X_i X_{i+1} + Y_i Y_{i+1} + delta Z_i Z_{i+1}]        (Pauli)
//   j1j2              sum_i [J1 h_{i,i+1} + J2 h_{i,i+2}],  h = XX + YY + delta ZZ   (Pauli)
//   spin1_heisenberg  J sum_i S_i . S_{i+1}                                  (spin-1 S)
//   hubbard           -t sum_{i,s} (c+_{i s} c_{i+1 s} + h.c.) + U sum_i n_up n_dn - mu sum_i n_i
//   hubbard_lattice   the same on a width-W lattice: Lx columns of W sites, site-split
//                     (one site per orbital), column-major i = W*x + y; W = 1 is the
//                     chain, W = 2 the ladder, y_periodic (W >= 3) makes it a cylinder
// Spin sectors are "sz2" = 2 Sz_total; the Hubbard ones are "n_up" and "n_dn".
static bool is_block_model(const std::string& type) {
    return type == "heisenberg" || type == "xxz" || type == "j1j2" ||
           type == "spin1_heisenberg" || type == "hubbard" || type == "hubbard_lattice";
}

struct Lattice { int Lx = 0, W = 1; bool y_periodic = false; };

static Lattice read_lattice(const nlj& prob) {
    Lattice g;
    g.Lx = prob.value("Lx", 0);
    g.W = prob.value("W", 1);
    g.y_periodic = prob.value("y_periodic", false);
    if (g.Lx < 1 || g.W < 1)
        throw std::runtime_error("hubbard_lattice: need Lx >= 1 and W >= 1");
    // In 64 bits: Lx = 2^30 + 1 with W = 4 used to wrap W*Lx round to 4 and run.
    if ((long long)g.Lx * g.W > 200)
        throw std::runtime_error("hubbard_lattice: W*Lx exceeds max (200)");
    if (g.y_periodic && g.W < 3)
        throw std::runtime_error("hubbard_lattice: y_periodic needs W >= 3 (at W = 2 the "
                                 "wrap would double every rung bond)");
    return g;
}

// Chain length of a problem: W*Lx for the lattice (checked against n_sites if given),
// otherwise "n_sites" / the legacy "n_qubits".
static int problem_n(const nlj& prob) {
    if (prob.value("type", std::string()) != "hubbard_lattice") return read_n(prob);
    const Lattice g = read_lattice(prob);
    const int n = g.W * g.Lx;
    if (prob.contains("n_sites") && prob["n_sites"].get<int>() != n)
        throw std::runtime_error("hubbard_lattice: n_sites must equal W*Lx (" +
                                 std::to_string(n) + ")");
    return n;
}

// The (n_up, n_dn) sector of a Hubbard problem; default half filling, lowest |Sz|.
static void check_sector_keys(const nlj& sec, const std::vector<std::string>& allowed,
                              const std::string& type);

static void hubbard_sector(const nlj& prob, int n, int& n_up, int& n_dn) {
    const nlj sec = prob.contains("sector") ? prob["sector"] : nlj::object();
    check_sector_keys(sec, {"n_up", "n_dn"}, prob.value("type", std::string("hubbard")));
    n_up = sec.value("n_up", (n + 1) / 2);
    n_dn = sec.value("n_dn", n / 2);
    if (n_up < 0 || n_up > n || n_dn < 0 || n_dn > n)
        throw std::runtime_error("hubbard sector: need 0 <= n_up, n_dn <= n_sites (got n_up=" +
            std::to_string(n_up) + ", n_dn=" + std::to_string(n_dn) + ")");
}

static std::vector<MPOSite> build_block_model_mpo(const nlj& prob, const std::string& type, int n) {
    if (type == "hubbard_lattice") {
        const Lattice g = read_lattice(prob);
        return build_hubbard_lattice_split_mpo(g.Lx, g.W, prob.value("t", 1.0),
                                               prob.value("U", 0.0), prob.value("mu", 0.0),
                                               g.y_periodic);
    }
    if (type == "heisenberg" || type == "xxz")
        return build_xxz_mpo(n, prob.value("J", 1.0), prob.value("delta", 1.0));
    if (type == "j1j2")
        return build_j1j2_mpo(n, prob.value("J1", 1.0), prob.value("J2", 0.0),
                              prob.value("delta", 1.0));
    if (type == "spin1_heisenberg")
        return build_spin1_heisenberg_mpo(n, prob.value("J", 1.0));
    // hubbard: always the packed two-U(1) charge (N and 2Sz), the fine-block path.
    return build_hubbard_mpo(n, prob.value("t", 1.0), prob.value("U", 0.0),
                             /*two_u1=*/true, prob.value("mu", 0.0));
}

// Pick k of the m slots [0, m), spread evenly (k <= m).
static std::vector<int> spread(int k, int m) {
    std::vector<int> out;
    for (int j = 0; j < k; ++j) out.push_back((int)((j + 0.5) * m / k));
    return out;
}

// Product state of a spin-S chain (local state l has 2Sz = 2S - 2l) whose levels sum
// to level_sum, i.e. whose 2 Sz_total is 2S n - 2 level_sum. Starts from the Neel
// alternation of the extreme states (the engine's own default start) and moves evenly
// spread sites one level at a time until the sum is right. Staying close to Neel keeps
// every bond's accumulated charge typical; the engine's bond-space guess needs that
// ("sector missing from the bond-space guess" otherwise).
static std::vector<int> spin_config(int n, int d, int level_sum) {
    std::vector<int> cfg(n);
    int cur = 0;
    for (int i = 0; i < n; ++i) { cfg[i] = (i % 2 == 0) ? 0 : d - 1; cur += cfg[i]; }
    int need = level_sum - cur;
    while (need != 0) {
        const int dir = need > 0 ? +1 : -1;
        std::vector<int> cand;
        for (int i = 0; i < n; ++i)
            if (dir > 0 ? cfg[i] < d - 1 : cfg[i] > 0) cand.push_back(i);
        if (cand.empty()) throw std::runtime_error("spin_config: sector unreachable");
        const int k = std::min(std::abs(need), (int)cand.size());
        for (int j : spread(k, (int)cand.size())) cfg[cand[j]] += dir;
        need -= dir * k;
    }
    return cfg;
}

// Hubbard product state with n_up up and n_dn down electrons (local states 0 empty,
// 1 up, 2 dn, 3 double): doubly occupied sites only when N > n, singly occupied ones
// spread evenly, and the up spins spread evenly among those. Half filling gives
// |dn up dn up ...>. The spins used to alternate "while both remain", which piled a
// spin imbalance's excess at the right end: an atypical path that the bond-space guess
// pruned, so e.g. L = 32 with (23 up, 9 dn) failed below max_bond 91.
static std::vector<int> hubbard_config(int n, int n_up, int n_dn) {
    const int D = std::max(0, n_up + n_dn - n);        // doubly occupied sites
    const int S = n_up + n_dn - 2 * D;                 // singly occupied sites
    std::vector<int> occ(n, 0);
    for (int i : spread(D, n)) occ[i] = 2;
    std::vector<int> free_sites;
    for (int i = 0; i < n; ++i) if (occ[i] == 0) free_sites.push_back(i);
    for (int j : spread(S, (int)free_sites.size())) occ[free_sites[j]] = 1;
    std::vector<int> cfg(n, 0), singles;
    for (int i = 0; i < n; ++i) {
        if (occ[i] == 2) cfg[i] = 3;
        if (occ[i] == 1) { cfg[i] = 2; singles.push_back(i); }   // down unless chosen below
    }
    for (int j : spread(n_up - D, (int)singles.size())) cfg[singles[j]] = 1;
    return cfg;
}

// A sector names exactly the model's conserved charges, as integers. Anything else is an
// error: a misspelled key ({"sz": 1}) used to leave the default sector in force, silently,
// and {"sz2": 2.7} was truncated to 2.
static void check_sector_keys(const nlj& sec, const std::vector<std::string>& allowed,
                              const std::string& type) {
    if (!sec.is_object()) throw std::runtime_error("sector must be an object");
    for (auto it = sec.begin(); it != sec.end(); ++it) {
        if (std::find(allowed.begin(), allowed.end(), it.key()) == allowed.end()) {
            std::string names;
            for (const auto& a : allowed) names += (names.empty() ? "" : ", ") + a;
            throw std::runtime_error("unknown sector key '" + it.key() + "' for type '" +
                                     type + "' (expected " + names + ")");
        }
        if (!it.value().is_number_integer())
            throw std::runtime_error("sector." + it.key() + " must be an integer");
    }
}

struct BlockModel {
    std::vector<MPOSite> mpo;
    int total_charge = 0;
    std::vector<int> init_config;
    nlj sector;
    const char* method = "u1";
};

static BlockModel resolve_block_model(const nlj& prob, const std::string& type, int n) {
    BlockModel m;
    m.mpo = build_block_model_mpo(prob, type, n);
    const nlj sec = prob.contains("sector") ? prob["sector"] : nlj::object();
    if (type == "hubbard" || type == "hubbard_lattice") {
        int n_up, n_dn;
        hubbard_sector(prob, n, n_up, n_dn);
        m.total_charge = pack_q2(n_up + n_dn, n_up - n_dn);
        m.init_config  = hubbard_config(n, n_up, n_dn);
        m.sector       = {{"n_up", n_up}, {"n_dn", n_dn}};
        m.method       = "u1xu1";
        return m;
    }
    check_sector_keys(sec, {"sz2"}, type);
    const int two_s = (type == "spin1_heisenberg") ? 2 : 1;   // 2S per site
    const int d = two_s + 1;
    const int sz2 = sec.value("sz2", (two_s * n) % 2);        // default: lowest |Sz|
    if (std::abs(sz2) > two_s * n || (two_s * n - sz2) % 2 != 0)
        throw std::runtime_error("sector sz2=" + std::to_string(sz2) + " (2 Sz_total) is not "
            "reachable on " + std::to_string(n) + " spin-" + (two_s == 1 ? "1/2" : "1") +
            " sites: need |sz2| <= " + std::to_string(two_s * n) +
            (two_s == 1 ? " with the parity of n_sites" : " and even"));
    m.total_charge = sz2;
    m.init_config  = spin_config(n, d, (two_s * n - sz2) / 2);
    m.sector       = {{"sz2", sz2}};
    return m;
}

// Parse Hamiltonian from JSON into MPO
static std::vector<MPOSite> parse_ham(const nlj& prob, int n) {
    std::string type = prob.value("type", "pauli_sum");
    if (is_block_model(type)) return build_block_model_mpo(prob, type, n);
    if (type != "pauli_sum")
        throw std::runtime_error("unknown type '" + type + "' (expected heisenberg, xxz, j1j2, "
                                 "spin1_heisenberg, hubbard, hubbard_lattice, or pauli_sum)");
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

// ---- HVA circuits on the Hubbard lattice (M5) ---------------------------------

// The circuit's bond groups as JSON, [[[i, j], ...], ...], in the order it applies them.
static nlj groups_json(const std::vector<std::vector<std::pair<int,int>>>& groups) {
    nlj out = nlj::array();
    for (const auto& g : groups) {
        nlj gj = nlj::array();
        for (const auto& e : g) gj.push_back({e.first, e.second});
        out.push_back(gj);
    }
    return out;
}

// At U = 0 the ground state of a sector is one Slater determinant per species, unique
// unless a species' Fermi level falls inside a degenerate single-particle shell (the
// +-k_y pairs of a cylinder, say). DMRG would then return an arbitrary member of the
// degenerate space, and a circuit started from it would not be reproducible, so refuse.
// Otherwise return the exact U = 0 (and mu = 0) ground-state energy of the sector, the
// sum of the lowest n_up and n_dn single-particle levels, so the prepared state's
// error can be REPORTED: a DMRG start stops when its energy stops changing, which on a
// long chain with a small gap left it 7.7e-7 off -- and a circuit on it 2e-4 off --
// with both truncation fields at 0. The degeneracy test scales with t itself (t = 0 is
// all-degenerate and refused; t = 1e-12 has gaps of that size and is not).
static double free_ground_energy(
        int n, double t, const std::vector<std::vector<std::pair<int,int>>>& groups,
        int n_up, int n_dn) {
    Eigen::MatrixXd h = Eigen::MatrixXd::Zero(n, n);
    for (const auto& g : groups)
        for (const auto& e : g) { h(e.first, e.second) -= t; h(e.second, e.first) -= t; }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(h);
    const Eigen::VectorXd& eps = es.eigenvalues();   // ascending
    double e0 = 0;
    for (int k : {n_up, n_dn}) {
        if (k > 0 && k < n && std::abs(eps(k) - eps(k - 1)) <= 1e-9 * std::abs(t))
            throw std::runtime_error("init 'free': the U = 0 ground state is degenerate at this "
                "filling (a Fermi level inside a degenerate single-particle shell), so it is no "
                "unique starting state; choose another filling, or init 'product'");
        e0 += eps.head(k).sum();
    }
    return e0;
}

// ---- MPS handles (M11): the registry -----------------------------------------------------
// States live here between calls, under integer handles. No pointer crosses the C API, so a
// stale or made-up handle is refused, never dereferenced; and an operation holds its own
// reference, so a concurrent free cannot pull a state out from under it.
std::mutex g_mps_mu;
std::map<long long, std::shared_ptr<MPSState>> g_mps;
long long g_mps_next = 1;

long long mps_put(std::shared_ptr<MPSState> s) {
    std::lock_guard<std::mutex> lk(g_mps_mu);
    const long long h = g_mps_next++;
    g_mps[h] = std::move(s);
    return h;
}

std::shared_ptr<MPSState> mps_get(const nlj& req, const char* key) {
    if (!req.contains(key) || !req[key].is_number_integer())
        throw std::runtime_error(std::string("mps: '") + key + "' must be a state handle");
    const long long h = req[key].get<long long>();
    std::lock_guard<std::mutex> lk(g_mps_mu);
    auto it = g_mps.find(h);
    if (it == g_mps.end())
        throw std::runtime_error("mps: no live state has handle " + std::to_string(h));
    return it->second;
}

void mps_keys(const nlj& req, std::initializer_list<const char*> allowed) {
    for (auto it = req.begin(); it != req.end(); ++it) {
        bool ok = it.key() == "op";
        for (const char* a : allowed) ok = ok || it.key() == a;
        if (!ok)
            throw std::runtime_error("mps " + req.value("op", std::string()) + ": unknown key '" +
                                     it.key() + "'");
    }
}

// The block-sparse model of a request's "model" object, as qx_dmrg_sweep reads a problem.
BlockModel mps_model(const nlj& req) {
    if (!req.contains("model") || !req["model"].is_object())
        throw std::runtime_error("mps " + req.value("op", std::string()) +
                                 ": 'model' must be a model object, as qx_dmrg_sweep takes");
    const nlj& m = req["model"];
    const std::string type = m.value("type", std::string());
    if (!is_block_model(type))
        throw std::runtime_error("mps: model type '" + type + "' has no block-sparse form");
    const int n = problem_n(m);
    if (n < 3) throw std::runtime_error("mps: n_sites must be >= 3 (block-sparse solver)");
    if (n > 400) throw std::runtime_error("mps: n_sites exceeds max (400)");
    return resolve_block_model(m, type, n);
}

std::vector<double> mps_real_diag(const nlj& req) {
    if (!req.contains("diag") || !req["diag"].is_array())
        throw std::runtime_error("mps " + req.value("op", std::string()) +
                                 ": 'diag' must be the operator's diagonal on the local space");
    std::vector<double> w;
    for (const auto& x : req["diag"]) {
        if (!x.is_number()) throw std::runtime_error("mps: 'diag' entries must be real numbers");
        w.push_back(x.get<double>());
    }
    return w;
}

// A list of complex numbers, each a number or an [re, im] pair.
std::vector<std::complex<double>> mps_complex_entries(const nlj& req, const char* key) {
    const std::string what = "mps " + req.value("op", std::string()) + ": '" + key + "'";
    if (!req[key].is_array())
        throw std::runtime_error(what + " must be a list of numbers or [re, im] pairs");
    std::vector<std::complex<double>> v;
    for (const auto& x : req[key]) {
        if (x.is_number())
            v.emplace_back(x.get<double>(), 0.0);
        else if (x.is_array() && x.size() == 2 && x[0].is_number() && x[1].is_number())
            v.emplace_back(x[0].get<double>(), x[1].get<double>());
        else
            throw std::runtime_error(what + " entries must be numbers or [re, im] pairs");
    }
    return v;
}

// A request's one-site operator: its real diagonal ("diag"), or its row-major matrix
// ("matrix") with an optional diagonal "string" on the sites left of it.
struct MPSOperator {
    bool diag = true;
    std::vector<double> w;
    std::vector<std::complex<double>> m, string;
    bool has_string = false;
    const std::vector<std::complex<double>>* string_or_null() const {
        return has_string ? &string : nullptr;
    }
};

MPSOperator mps_operator(const nlj& req) {
    const std::string what = "mps " + req.value("op", std::string());
    if (req.contains("diag") == req.contains("matrix"))
        throw std::runtime_error(what + ": give the operator as its 'diag' or its 'matrix', "
                                 "one of the two");
    MPSOperator o;
    if (req.contains("diag")) {
        if (req.contains("string"))
            throw std::runtime_error(what + ": a 'string' goes with a 'matrix'");
        o.w = mps_real_diag(req);
        return o;
    }
    o.diag = false;
    o.m = mps_complex_entries(req, "matrix");
    if (req.contains("string")) {
        o.has_string = true;
        o.string = mps_complex_entries(req, "string");
    }
    return o;
}

nlj mps_complex_list(const std::vector<std::complex<double>>& v) {
    nlj re = nlj::array(), im = nlj::array();
    for (const auto& z : v) { re.push_back(z.real()); im.push_back(z.imag()); }
    return {{"re", re}, {"im", im}};
}

}  // namespace

extern "C" {

QX_API const char* qx_dmrg_sweep(const char* problem_json,
                                  const char* options_json) {
    try {
        nlj prob  = nlj::parse(problem_json  ? problem_json  : "{}");
        nlj opts  = nlj::parse(options_json  ? options_json  : "{}");

        int  n        = problem_n(prob);
        int  max_bond = opts.value("max_bond",  64);
        double cutoff = opts.value("cutoff",    1e-10);
        int  n_sweeps = opts.value("n_sweeps",  10);
        double tol    = opts.value("tol",       1e-10);

        if (n < 2)        return dmrg_error("n_qubits must be >= 2");
        if (n > 200)      return dmrg_error("n_qubits exceeds max (200)");
        if (n_sweeps < 1) return dmrg_error("n_sweeps must be >= 1");
        if (max_bond < 1) return dmrg_error("max_bond must be >= 1");

        std::string type   = prob.value("type", "pauli_sum");
        std::string method = opts.value("method", "u1");  // "u1" (Sz blocks) | "su2"

        // SU(2)-symmetric path: Heisenberg only. max_bond is the multiplet cap;
        // the energy is in the same J*sum sigma.sigma convention as the U(1) path.
        // Neither the SU(2) path nor the dense fallback can target a sector; refuse one
        // rather than silently return the ground state of some other sector.
        const bool sector_capable = method != "su2" && is_block_model(type) && n >= 3;
        if (prob.contains("sector") && !sector_capable)
            return dmrg_error("a sector needs the block-sparse U(1) solver: a symmetric "
                              "model type, method 'u1', and n_sites >= 3");

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
            out["n_sites"]          = n;
            tl_dmrg_buf = out.dump();
            return tl_dmrg_buf.c_str();
        }

        // The block-sparse engine derives its symmetry data from a bulk MPO site, so it
        // refuses chains shorter than 3 (derive_mpo_sym's contract, tested in
        // qx_mpo_sym_test). A 2-site Heisenberg chain is still a valid question: answer
        // it with the dense solver rather than returning an internal error. The other
        // models target a symmetry sector, which the dense solver cannot, so they refuse.
        // Report which solver actually ran -- this used to say "u1" for every call.
        nlj out;
        DMRGResult res;
        if (is_block_model(type) && !(type == "heisenberg" && n < 3)) {
            if (n < 3)
                return dmrg_error("n_sites must be >= 3 for type '" + type +
                                  "' (block-sparse solver)");
            BlockModel m = resolve_block_model(prob, type, n);
            res = run_dmrg_blocks(m.mpo, n, max_bond, cutoff, n_sweeps, tol,
                                  m.total_charge, &m.init_config);
            out["method"] = m.method;
            out["sector"] = m.sector;
        } else {
            res = run_dmrg(parse_ham(prob, n), n, max_bond, cutoff, n_sweeps, tol);
            out["method"] = "dense";
        }

        out["ok"]               = true;
        out["energy"]           = res.energy;
        out["truncation_error"] = res.truncation_error;
        out["n_sweeps_done"]    = res.n_sweeps_done;
        out["wall_time_s"]      = res.wall_time_s;
        out["sweep_energies"]   = res.sweep_energies;
        out["n_sites"]          = n;

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
        int    n        = read_n(prob);
        double J        = prob.value("J", 1.0);
        double delta    = prob.value("delta", 1.0);   // XXZ anisotropy (1 = Heisenberg)
        std::string type = prob.value("type", "heisenberg");
        if (type != "heisenberg" && type != "xxz")
            return dmrg_error("tdvp supports only type 'heisenberg'/'xxz'");
        if (n < 2)  return dmrg_error("n_qubits must be >= 2");

        double dt       = opts.value("dt", 0.05);
        int    n_steps  = opts.value("n_steps", 20);
        int    chi_max  = opts.value("chi_max", 64);
        bool   imag     = opts.value("imag", false);
        bool   two_site = opts.value("two_site", true);
        std::string init = opts.value("initial", "neel");
        // Block-sparse (Sz-symmetric) TDVP2 is the default fast path (Neel init). It needs
        // n >= 3 (a bulk MPO site for the symmetry derivation); the non-block TDVP below
        // handles n = 2 fine, so fall back to it instead of failing.
        bool   block    = opts.value("block", true) && two_site && init == "neel" && n >= 3;

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
        int    n  = read_n(prob);
        double J  = prob.value("J", 1.0);
        double delta = prob.value("delta", 1.0);   // XXZ anisotropy (1 = Heisenberg)
        std::string type = prob.value("type", "heisenberg");
        if (type != "heisenberg" && type != "xxz")
            return dmrg_error("spectral supports only type 'heisenberg'/'xxz'");
        // Block-sparse only (no dense fallback here), and the block engine needs a bulk MPO
        // site. Say so in the caller's terms rather than surfacing derive_mpo_sym's.
        if (n < 3) return dmrg_error("n_qubits must be >= 3 (block-sparse solver)");

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
        int    n  = read_n(prob);
        double J  = prob.value("J", 1.0);
        double delta = prob.value("delta", 1.0);     // initial XXZ anisotropy
        std::string type = prob.value("type", "heisenberg");
        if (type != "heisenberg" && type != "xxz")
            return dmrg_error("loschmidt supports only type 'heisenberg'/'xxz'");
        // Block-sparse only (no dense fallback here), and the block engine needs a bulk MPO
        // site. Say so in the caller's terms rather than surfacing derive_mpo_sym's.
        if (n < 3) return dmrg_error("n_qubits must be >= 3 (block-sparse solver)");

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
        int    n  = read_n(prob);
        double J  = prob.value("J", 1.0);
        double delta = prob.value("delta", 1.0);
        std::string type = prob.value("type", "heisenberg");
        if (type != "heisenberg" && type != "xxz")
            return dmrg_error("conductivity supports only type 'heisenberg'/'xxz'");
        // Block-sparse only (no dense fallback here), and the block engine needs a bulk MPO
        // site. Say so in the caller's terms rather than surfacing derive_mpo_sym's.
        if (n < 3) return dmrg_error("n_qubits must be >= 3 (block-sparse solver)");

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

// M6: finite temperature (see c_api.h). The purification entry points select the model
// by its local dimension d, with the (J, delta) arguments read as (t, U) for hubbard.
QX_API const char* qx_thermal(const char* problem_json, const char* options_json) {
    try {
        nlj prob = nlj::parse(problem_json ? problem_json : "{}");
        nlj opts = nlj::parse(options_json ? options_json : "{}");
        const std::string type = prob.value("type", "");
        int d = 0;
        double J = 1.0, delta = 1.0, mu = 0.0;
        if (type == "heisenberg" || type == "xxz") {
            d = 2; J = prob.value("J", 1.0); delta = prob.value("delta", 1.0);
        } else if (type == "spin1_heisenberg") {
            d = 3; J = prob.value("J", 1.0);
        } else if (type == "hubbard") {
            d = 4; J = prob.value("t", 1.0); delta = prob.value("U", 0.0);
            mu = prob.value("mu", 0.0);
        } else {
            return dmrg_error("thermal supports the heisenberg/xxz, spin1_heisenberg and "
                              "hubbard chains; got type '" + type + "'");
        }
        const int n = read_n(prob);
        if (n < 3) return dmrg_error("thermal needs n_sites >= 3 (block-sparse solver)");

        if (!opts.contains("quantity"))
            return dmrg_error("thermal needs options.quantity: statics, structure_factor "
                              "or current");
        const std::string quantity = opts["quantity"].get<std::string>();
        const double beta = opts.value("beta", 0.0);
        const double dbeta = opts.value("dbeta", 0.05);
        const int max_bond = opts.value("max_bond", 128);
        const double cutoff = opts.value("cutoff", 1e-10);
        if (!(beta > 0.0)) return dmrg_error("thermal needs beta > 0");
        if (!(dbeta > 0.0)) return dmrg_error("thermal needs dbeta > 0");
        if (max_bond < 1) return dmrg_error("thermal needs max_bond >= 1");
        // A whole number of steps that ends exactly at the end of the span, none longer
        // than the step asked for: the ratio rounded up (1e-9 absorbs the rounding, so
        // 5 / 0.05 stays 100 steps).
        auto steps_to = [](double span, double step) {
            return (int)std::ceil(span / step - 1e-9);
        };
        if (beta / dbeta > 1e7) return dmrg_error("thermal: beta / dbeta exceeds 1e7 steps");
        const int n_cool = std::max(1, steps_to(beta, dbeta));

        nlj out;
        out["ok"] = true;
        out["quantity"] = quantity;
        out["n_sites"] = n;
        if (quantity == "statics") {
            const int stride = opts.value("stride", 1);
            const bool with_chi = opts.value("susceptibility", true);
            auto r = qx_dmrg::run_thermal_blocks(n, J, delta, beta / n_cool, n_cool, max_bond,
                                                 cutoff, stride, with_chi, d, mu);
            out["betas"] = r.betas;
            out["energy"] = r.E;
            out["O"] = r.sz;
            out["OO"] = r.szsz;
            out["susceptibility"] = r.chi_zz;
            out["center"] = r.c;
            out["max_bond"] = r.max_bond;
            out["wall_time_s"] = r.wall_time_s;
        } else if (quantity == "structure_factor" || quantity == "current") {
            const double dt = opts.value("dt", 0.05);
            const double t_max = opts.value("t_max", 5.0);
            const bool disentangle = opts.value("disentangle", true);
            if (!(dt > 0.0)) return dmrg_error("thermal needs dt > 0");
            if (!(t_max >= 0.0)) return dmrg_error("thermal needs t_max >= 0");
            if (t_max / dt > 1e7) return dmrg_error("thermal: t_max / dt exceeds 1e7 steps");
            const int n_steps = (t_max > 0.0) ? std::max(1, steps_to(t_max, dt)) : 0;
            const double dt_run = (n_steps > 0) ? t_max / n_steps : dt;
            if (quantity == "structure_factor") {
                auto r = qx_dmrg::run_spectral_thermal_blocks(n, J, delta, beta, n_cool, dt_run,
                                                              n_steps, max_bond, cutoff,
                                                              disentangle, d, mu);
                out["times"] = r.times;
                out["C_re"] = r.C_re;
                out["C_im"] = r.C_im;
                out["source"] = r.l0;
                out["beta"] = r.beta;
                out["energy"] = r.E;
                out["bond_t"] = r.bond_t;
                out["max_bond"] = r.max_bond;
                out["wall_time_s"] = r.wall_time_s;
            } else {
                auto r = qx_dmrg::run_conductivity_thermal_blocks(n, J, delta, beta, n_cool, dt_run,
                                                                  n_steps, max_bond, cutoff,
                                                                  disentangle, d, mu);
                out["times"] = r.times;
                out["C_re"] = r.C_re;
                out["C_im"] = r.C_im;
                out["beta"] = r.beta;
                out["energy"] = r.E;
                out["bond_t"] = r.bond_t;
                out["max_bond"] = r.max_bond;
                out["wall_time_s"] = r.wall_time_s;
            }
        } else {
            return dmrg_error("thermal: unknown quantity '" + quantity +
                              "' (statics, structure_factor or current)");
        }
        tl_dmrg_buf = out.dump();
        return tl_dmrg_buf.c_str();
    } catch (const std::exception& e) {
        return dmrg_error(e.what());
    }
}

QX_API const char* qx_mps_energy(const char* problem_json) {
    try {
        nlj prob = nlj::parse(problem_json ? problem_json : "{}");
        int n    = problem_n(prob);
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

QX_API const char* qx_hva_layout(const char* problem_json) {
    try {
        nlj prob = nlj::parse(problem_json ? problem_json : "{}");
        const Lattice g = read_lattice(prob);
        auto groups = hva_lattice_group_bonds(g.Lx, g.W, g.y_periodic);
        nlj out;
        out["ok"]               = true;
        out["n_sites"]          = g.W * g.Lx;
        out["groups"]           = groups_json(groups);
        out["angles_per_layer"] = (int)groups.size() + 1;
        tl_dmrg_buf = out.dump();
        return tl_dmrg_buf.c_str();
    } catch (const std::exception& e) {
        return dmrg_error(e.what());
    }
}

QX_API const char* qx_hva_circuit(const char* problem_json, const char* options_json) {
    try {
        nlj prob = nlj::parse(problem_json ? problem_json : "{}");
        nlj opts = nlj::parse(options_json ? options_json : "{}");
        const Lattice g = read_lattice(prob);
        const int n = g.W * g.Lx;
        if (n < 3)   return dmrg_error("hva: need W*Lx >= 3 (block-sparse solver)");
        if (n > 200) return dmrg_error("hva: W*Lx exceeds max (200)");
        const double t = prob.value("t", 1.0), U = prob.value("U", 0.0),
                     mu = prob.value("mu", 0.0);
        int n_up, n_dn;
        hubbard_sector(prob, n, n_up, n_dn);
        const int total = pack_q2(n_up + n_dn, n_up - n_dn);
        const int chi = opts.value("max_bond", 256);
        const double cutoff = opts.value("cutoff", 1e-12);
        const std::string init = opts.value("init", std::string("free"));
        const bool want_entropy = opts.value("entropy", true);
        if (opts.contains("gradient") && !opts["gradient"].is_boolean())
            return dmrg_error("hva: 'gradient' must be true or false");
        const bool want_gradient = opts.value("gradient", false);
        const std::string svd = opts.value("svd", std::string("gram"));
        if (svd != "gram" && svd != "direct")
            return dmrg_error("hva: 'svd' must be 'gram' (the default) or 'direct'");
        if (chi < 1) return dmrg_error("max_bond must be >= 1");
        if (init != "free" && init != "product")
            return dmrg_error("init must be 'free' (the U = 0 ground state) or 'product'");
        if (!prob.contains("angles") || !prob["angles"].is_array())
            return dmrg_error("hva: 'angles' must be a list of layers, each a list of "
                              "angles (one per bond group, then one for the on-site term)");
        std::vector<std::vector<double>> angles;
        for (const auto& lay : prob["angles"]) angles.push_back(lay.get<std::vector<double>>());

        auto groups = hva_lattice_group_bonds(g.Lx, g.W, g.y_periodic);
        auto lat = build_hubbard_lattice_split_mpo(g.Lx, g.W, t, U, mu, g.y_periodic);
        auto hop = build_hubbard_mpo(n, t, 0.0, /*two_u1=*/true);   // the bond generator
        std::vector<int> cfg = hubbard_config(n, n_up, n_dn);

        nlj out;
        std::shared_ptr<HVAInitState> psi0;
        if (init == "free") {
            const double e_exact = free_ground_energy(n, t, groups, n_up, n_dn);
            // Prepare at no more than the largest bond an n-site chain can have (4^(n/2)).
            // The preparation's bond-space guess fills every bond to its max_bond, so at
            // the circuit's max_bond it cost O(max_bond^3) at ANY size: 117 s for 6 sites
            // at max_bond 4096, where 64 is already exact.
            long long cap = 1;
            for (int k = 0; k < n / 2 && cap < chi; ++k) cap *= 4;
            const int chi_prep = (int)std::min<long long>(chi, cap);
            DMRGResult prep;
            psi0 = hva_prepare_init(build_hubbard_lattice_split_mpo(g.Lx, g.W, t, 0.0, 0.0,
                                                                    g.y_periodic),
                                    n, chi_prep, total, &cfg, &prep);
            out["init_energy"]           = prep.energy;
            out["init_energy_exact"]     = e_exact;
            out["init_energy_error"]     = prep.energy - e_exact;
            out["init_truncation_error"] = prep.truncation_error;   // sum over its last sweep
        } else {
            out["init_config"] = cfg;   // local states: 0 empty, 1 up, 2 down, 3 double
        }
        HVAGradient grad;
        HVAResult r = run_hva_lattice(lat, hop, g.Lx, g.W, g.y_periodic, angles, chi,
                                      cutoff, total, &cfg, psi0, want_entropy,
                                      want_gradient ? &grad : nullptr, svd == "direct");
        out["ok"]               = true;
        out["energy"]           = r.energy;
        if (want_gradient) {
            out["gradient"]                  = grad.grad;   // dE/dtheta, laid out as the angles
            out["gradient_truncation_error"] = grad.trunc_err;
            out["gradient_return_error"]     = grad.return_error;
        }
        // Entropies of the final state (natural log), from a lossless centre walk: the
        // largest over all cuts, and the balanced cut's (none when W*Lx is odd).
        out["entropy_max"]      = want_entropy ? nlj(r.S_max) : nlj(nullptr);
        out["entropy_mid"]      = (want_entropy && n % 2 == 0) ? nlj(r.S_mid) : nlj(nullptr);
        out["truncation_error"] = r.trunc_err;    // largest single discarded weight
        out["max_bond"]         = r.max_bond;
        out["layer_bonds"]      = r.layer_bond;
        out["wall_time_s"]      = r.wall_s;
        out["sector"]           = {{"n_up", n_up}, {"n_dn", n_dn}};
        out["init"]             = init;
        tl_dmrg_buf = out.dump();
        return tl_dmrg_buf.c_str();
    } catch (const std::exception& e) {
        return dmrg_error(e.what());
    }
}

QX_API const char* qx_u1_circuit(const char* problem_json, const char* options_json) {
    try {
        nlj prob = nlj::parse(problem_json ? problem_json : "{}");
        nlj opts = nlj::parse(options_json ? options_json : "{}");
        for (auto it = prob.begin(); it != prob.end(); ++it)
            if (it.key() != "n_qubits" && it.key() != "initial" && it.key() != "gates" &&
                it.key() != "measure_after")
                return dmrg_error("u1_circuit: unknown problem key '" + it.key() + "'");
        for (auto it = opts.begin(); it != opts.end(); ++it)
            if (it.key() != "max_bond" && it.key() != "cutoff")
                return dmrg_error("u1_circuit: unknown option '" + it.key() + "'");
        if (!prob.contains("n_qubits")) return dmrg_error("u1_circuit: 'n_qubits' is required");
        const int n = prob["n_qubits"].get<int>();
        if (n > 4000) return dmrg_error("u1_circuit: n_qubits exceeds max (4000)");
        std::vector<int> bits(std::max(n, 0), 0);
        if (prob.contains("initial")) bits = prob["initial"].get<std::vector<int>>();
        if (!prob.contains("gates") || !prob["gates"].is_array())
            return dmrg_error("u1_circuit: 'gates' must be a list of {q, u}");
        std::vector<U1Gate> gates;
        gates.reserve(prob["gates"].size());
        for (const auto& g : prob["gates"]) {
            U1Gate ug;
            ug.q = g.at("q").get<int>();
            for (const auto& z : g.at("u")) {
                if (!z.is_array() || z.size() != 2)
                    throw std::runtime_error("u1_circuit: each matrix entry is [re, im]");
                ug.u.emplace_back(z[0].get<double>(), z[1].get<double>());
            }
            if (ug.u.size() == 4) ug.arity = 1;
            else if (ug.u.size() == 16) ug.arity = 2;
            else throw std::runtime_error("u1_circuit: a gate has 4 entries (one qubit) or 16 "
                                          "(qubits q, q+1)");
            gates.push_back(std::move(ug));
        }
        std::vector<int> when = {(int)gates.size()};
        if (prob.contains("measure_after")) when = prob["measure_after"].get<std::vector<int>>();
        const int chi = opts.value("max_bond", 256);
        const double cutoff = opts.value("cutoff", 1e-10);

        U1CircuitResult r = run_u1_circuit(n, bits, gates, when, chi, cutoff);
        nlj out;
        out["ok"] = true;
        out["n_qubits"] = n;
        out["ones"] = r.ones;
        nlj ms = nlj::array();
        for (const auto& m : r.measurements)
            ms.push_back({{"after", m.after}, {"z", m.z}, {"norm2", m.norm2},
                          {"max_bond", m.max_bond}});
        out["measurements"] = ms;
        out["max_bond"] = r.max_bond;
        out["truncation_sum"] = r.trunc_sum;
        out["truncation_max"] = r.trunc_max;
        out["wall_time_s"] = r.wall_s;
        tl_dmrg_buf = out.dump();
        return tl_dmrg_buf.c_str();
    } catch (const std::exception& e) {
        return dmrg_error(e.what());
    }
}

QX_API const char* qx_mps(const char* request_json) {
    try {
        nlj req = nlj::parse(request_json ? request_json : "{}");
        const std::string op = req.value("op", std::string());
        nlj out;
        out["ok"] = true;
        if (op == "ground_state") {
            mps_keys(req, {"model", "max_bond", "cutoff", "n_sweeps", "tol"});
            BlockModel m = mps_model(req);
            const int n = (int)m.mpo.size();
            const int chi = req.value("max_bond", 64);
            const double cutoff = req.value("cutoff", 1e-10);
            const int n_sweeps = req.value("n_sweeps", 10);
            const double tol = req.value("tol", 1e-10);
            if (chi < 1) return dmrg_error("mps ground_state: max_bond must be >= 1");
            if (n_sweeps < 1) return dmrg_error("mps ground_state: n_sweeps must be >= 1");
            DMRGResult r;
            auto st = mps_ground_state(m.mpo, n, chi, cutoff, n_sweeps, tol, m.total_charge,
                                       &m.init_config, &r);
            out["handle"] = mps_put(st);
            out["energy"] = r.energy;
            out["truncation_error"] = r.truncation_error;
            out["n_sweeps_done"] = r.n_sweeps_done;
            out["sweep_energies"] = r.sweep_energies;
            out["wall_time_s"] = r.wall_time_s;
            out["sector"] = m.sector;
            out["method"] = m.method;
            out["n_sites"] = n;
        } else if (op == "product") {
            mps_keys(req, {"model", "config"});
            BlockModel m = mps_model(req);
            if (!req.contains("config") || !req["config"].is_array())
                return dmrg_error("mps product: 'config' must list one local state per site");
            const auto config = req["config"].get<std::vector<int>>();
            out["handle"] = mps_put(mps_product(m.mpo, (int)m.mpo.size(), config));
        } else if (op == "copy") {
            mps_keys(req, {"handle"});
            out["handle"] = mps_put(mps_copy(*mps_get(req, "handle")));
        } else if (op == "apply") {
            mps_keys(req, {"handle", "site", "diag", "matrix", "string"});
            auto st = mps_get(req, "handle");
            if (!req.contains("site") || !req["site"].is_number_integer())
                return dmrg_error("mps apply: 'site' must be an integer");
            const int site = req["site"].get<int>();
            const MPSOperator o = mps_operator(req);
            if (o.diag) {
                std::vector<std::complex<double>> diag(o.w.begin(), o.w.end());
                out["norm2"] = mps_apply_diag(*st, site, diag);
            } else {
                out["norm2"] = mps_apply_local(*st, site, o.m, o.string_or_null());
            }
        } else if (op == "evolve") {
            mps_keys(req, {"handle", "model", "dt", "n_steps", "imaginary", "max_bond", "cutoff"});
            auto st = mps_get(req, "handle");
            BlockModel m = mps_model(req);
            MPSEvolveStats e = mps_evolve(*st, m.mpo, req.value("dt", 0.0),
                                          req.value("n_steps", 1), req.value("imaginary", false),
                                          req.value("max_bond", 64), req.value("cutoff", 1e-10));
            out["max_bond"] = e.max_bond;
            out["wall_time_s"] = e.wall_s;
        } else if (op == "overlap") {
            mps_keys(req, {"bra", "ket"});
            const auto z = mps_overlap(*mps_get(req, "bra"), *mps_get(req, "ket"));
            out["re"] = z.real();
            out["im"] = z.imag();
        } else if (op == "expect") {
            mps_keys(req, {"bra", "ket", "diag", "matrix", "string"});
            auto bra = mps_get(req, "bra");
            auto ket = req.contains("ket") ? mps_get(req, "ket") : bra;
            const MPSOperator o = mps_operator(req);
            nlj v = mps_complex_list(o.diag ? mps_expect_diag(*bra, *ket, o.w)
                                            : mps_expect_local(*bra, *ket, o.m,
                                                               o.string_or_null()));
            out["re"] = v["re"];
            out["im"] = v["im"];
        } else if (op == "energy") {
            mps_keys(req, {"handle", "model"});
            auto st = mps_get(req, "handle");
            out["energy"] = mps_energy(*st, mps_model(req).mpo);
        } else if (op == "info") {
            mps_keys(req, {"handle"});
            MPSInfo i = mps_info(*mps_get(req, "handle"));
            out["n_sites"] = i.n;
            out["local_dim"] = i.d;
            out["bonds"] = i.bonds;
            out["norm2"] = i.norm2;
            out["charge"] = i.charge;
        } else if (op == "free") {
            mps_keys(req, {"handle"});
            mps_get(req, "handle");                         // refuses an unknown handle
            std::lock_guard<std::mutex> lk(g_mps_mu);
            g_mps.erase(req["handle"].get<long long>());
        } else if (op == "live") {
            mps_keys(req, {});
            std::lock_guard<std::mutex> lk(g_mps_mu);
            out["handles"] = (long long)g_mps.size();
        } else {
            return dmrg_error("mps: unknown op '" + op + "' (expected ground_state, product, "
                              "copy, apply, evolve, overlap, expect, energy, info, free or live)");
        }
        tl_dmrg_buf = out.dump();
        return tl_dmrg_buf.c_str();
    } catch (const std::exception& e) {
        return dmrg_error(e.what());
    }
}

}  // extern "C"
