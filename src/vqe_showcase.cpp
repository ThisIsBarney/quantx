// Fermionic VQE on the block-sparse engine: how deep must a Hamiltonian Variational
// Ansatz circuit be, and is the state it prepares classically simulable?
//
// The circuit (see run_hva_circuit): p layers of
//     e^{-i th_U H_U} e^{-i th_odd H_odd} e^{-i th_even H_even}
// on the 1D Hubbard chain at half filling. Every layer conserves N and 2Sz, so the
// U(1)xU(1) block machinery simulates the circuit EXACTLY (validated against a dense
// state-vector simulator in qx_hubbard_test, section HVA CIRCUIT) at sizes far past
// current quantum hardware -- 2n qubits for an n-site chain.
//
// For each (n, p) we optimise the 3p angles (layer-wise warm start -> simplex ->
// Adam on central-difference gradients) and report
//   * E_VQE(p) vs the exact DMRG ground state  -> how deep the ansatz must be;
//   * the bond dimension the circuit actually reached -> the VQE state's entanglement,
//     i.e. whether a classical MPS can follow the circuit at all.
// The second number is the one that decides whether running this ansatz on hardware
// could ever beat a laptop: an ansatz whose states stay low-chi is classically
// simulable no matter how many qubits it uses.
//
// Usage: qx_vqe_showcase          -> the (n, p) scan, writes docs/vqe_hubbard.csv
//        QX_VQE="n p chi U iters" -> a single point (used by the scan driver/tests)
//        QX_VQE_ONLY=A|B|D|E|F    -> run just that part (C runs with B)
//        QX_VQE_QUICK=1           -> shrink every part to a smoke test
//        QX_VQE_BP=N              -> part F: N random-angle samples per point (default
//                                    400; the size axis uses max(50, N/3))
//        QX_VQE_BPBUDGET=SEC      -> part F size axis: skip a point whose projected
//                                    sample cost exceeds SEC (default 900), printing
//                                    the measured per-evaluation time and the reason
//        QX_VQE_BPCHI=CHI         -> part F size axis: skip a point whose LOSSLESS chi
//                                    exceeds CHI (default 1024, i.e. n <= 10). Checked
//                                    before the timing probe, because reaching the probe
//                                    means preparing a state at that chi first
//        QX_SWAPCOST="W Lx chi"   -> time one x-hop layer of the swap network
//        QX_LATCHK=1              -> cross-check the W=1 lattice driver against the
//                                    ED-validated chain driver
#include "dmrg_internal.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <vector>

using namespace qx_dmrg;

// ---- Nelder-Mead simplex (derivative-free; 3p parameters, smooth objective) ------
// VQE angle optimisation has no cheap exact gradient here (a parameter-shift rule
// needs the same circuit run twice per angle anyway), and 3p is small, so a simplex
// method is both simpler and competitive. Standard coefficients.
struct NMResult { std::vector<double> x; double f = 0; int nfev = 0; };

static NMResult nelder_mead(const std::function<double(const std::vector<double>&)>& f,
                            std::vector<double> x0, double step, int max_fev,
                            double ftol = 1e-9) {
    const int dim = (int)x0.size();
    std::vector<std::vector<double>> S(dim + 1, x0);
    std::vector<double> F(dim + 1);
    for (int i = 0; i < dim; ++i) S[i + 1][i] += step;
    int nfev = 0;
    for (int i = 0; i <= dim; ++i) { F[i] = f(S[i]); ++nfev; }
    auto order = [&]() {
        std::vector<int> idx(dim + 1);
        for (int i = 0; i <= dim; ++i) idx[i] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return F[a] < F[b]; });
        std::vector<std::vector<double>> S2(dim + 1);
        std::vector<double> F2(dim + 1);
        for (int i = 0; i <= dim; ++i) { S2[i] = S[idx[i]]; F2[i] = F[idx[i]]; }
        S.swap(S2); F.swap(F2);
    };
    order();
    while (nfev < max_fev && std::abs(F[dim] - F[0]) > ftol * (std::abs(F[0]) + ftol)) {
        std::vector<double> c(dim, 0.0);                       // centroid of the best dim
        for (int i = 0; i < dim; ++i)
            for (int k = 0; k < dim; ++k) c[k] += S[i][k] / dim;
        auto comb = [&](double a) {                            // c + a (c - worst)
            std::vector<double> y(dim);
            for (int k = 0; k < dim; ++k) y[k] = c[k] + a * (c[k] - S[dim][k]);
            return y;
        };
        std::vector<double> xr = comb(1.0);                    // reflect
        double fr = f(xr); ++nfev;
        if (fr < F[0]) {
            std::vector<double> xe = comb(2.0);                // expand
            double fe = f(xe); ++nfev;
            if (fe < fr) { S[dim] = xe; F[dim] = fe; } else { S[dim] = xr; F[dim] = fr; }
        } else if (fr < F[dim - 1]) {
            S[dim] = xr; F[dim] = fr;
        } else {
            std::vector<double> xc = comb(fr < F[dim] ? 0.5 : -0.5);   // contract
            double fc = f(xc); ++nfev;
            if (fc < std::min(fr, F[dim])) { S[dim] = xc; F[dim] = fc; }
            else {                                             // shrink
                for (int i = 1; i <= dim; ++i) {
                    for (int k = 0; k < dim; ++k) S[i][k] = S[0][k] + 0.5 * (S[i][k] - S[0][k]);
                    F[i] = f(S[i]); ++nfev;
                }
            }
        }
        order();
    }
    return {S[0], F[0], nfev};
}

// ---- Adam with central-difference gradients ---------------------------------
// The simplex stalls in 3p dimensions (measured at p=6: a 6x larger budget only
// halved the error), which would make the reported "depth needed" an artefact of the
// optimiser rather than a property of the ansatz. A gradient step costs 6p circuit
// evaluations here -- the parameter-shift rule does not apply, since the HVA
// generators are not single Paulis -- and central differences at h=1e-4 are accurate
// to ~1e-8, far below the energy differences that matter.
static double adam_min(const std::function<double(const std::vector<double>&)>& f,
                       std::vector<double>& x, int n_steps, double lr,
                       int* nfev = nullptr) {
    const int dim = (int)x.size();
    const double b1 = 0.9, b2 = 0.999, eps = 1e-8, h = 1e-4;
    std::vector<double> m(dim, 0.0), v(dim, 0.0), g(dim, 0.0);
    double fx = f(x); if (nfev) ++*nfev;
    std::vector<double> best_x = x; double best_f = fx;
    for (int step = 1; step <= n_steps; ++step) {
        for (int k = 0; k < dim; ++k) {
            double x0 = x[k];
            x[k] = x0 + h; double fp = f(x);
            x[k] = x0 - h; double fm = f(x);
            x[k] = x0;
            g[k] = (fp - fm) / (2 * h);
            if (nfev) *nfev += 2;
        }
        double gn = 0; for (double gk : g) gn += gk * gk;
        if (std::sqrt(gn) < 1e-10) break;
        for (int k = 0; k < dim; ++k) {
            m[k] = b1 * m[k] + (1 - b1) * g[k];
            v[k] = b2 * v[k] + (1 - b2) * g[k] * g[k];
            double mh = m[k] / (1 - std::pow(b1, step));
            double vh = v[k] / (1 - std::pow(b2, step));
            x[k] -= lr * mh / (std::sqrt(vh) + eps);
        }
        fx = f(x); if (nfev) ++*nfev;
        if (fx < best_f) { best_f = fx; best_x = x; }
    }
    x = best_x;
    return best_f;
}

// Standard error of a variance estimate, from the sample's OWN fourth moment:
//   Var[s^2] = (mu4 - mu2^2) / N.
// The familiar s^2*sqrt(2/N) is this formula with mu4 = 3 mu2^2 substituted in -- i.e.
// it ASSUMES a Gaussian. Gradient distributions in this setting need not be, and the
// Gaussian form then UNDERSTATES the error, which is the direction that promotes noise
// to a "significant" difference. Measure the moments instead of assuming them.
static double var_stderr(double s1, double s2, double s3, double s4, int N) {
    const double m1 = s1 / N, m2 = s2 / N, m3 = s3 / N, m4 = s4 / N;
    const double mu2 = m2 - m1 * m1;
    const double mu4 = m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1 * m1 * m1 * m1;
    const double v = (mu4 - mu2 * mu2) / N;
    return v > 0 ? std::sqrt(v) : 0.0;
}

static std::vector<std::array<double,3>> to_layers(const std::vector<double>& x) {
    std::vector<std::array<double,3>> a(x.size() / 3);
    for (size_t l = 0; l < a.size(); ++l) a[l] = {x[3*l], x[3*l+1], x[3*l+2]};
    return a;
}

// One VQE point: optimise the 3p angles of a depth-p HVA circuit on the n-site chain.
struct VQEPoint {
    double E_vqe = 0, E_exact = 0, err = 0, err_site = 0;
    int max_bond = 0, nfev = 0, p = 0, n = 0;
    double wall = 0, S_max = 0, trunc = 0;
};

// prev_best (optional, in/out): the optimal angles at depth p-1, used to WARM-START
// depth p. Layer-wise growth is the standard HVA recipe and it matters a lot: random
// multi-start in 3p dimensions gets stuck (on the n=6 chain it plateaued ~35% above
// the ground state no matter how large the iteration budget), because a depth-p
// circuit that ignores its extra layers is already a local minimum. Seeding the new
// layer at zero starts exactly from the depth-(p-1) optimum, which can only improve.
static VQEPoint vqe_point(int n, int p, int chi, double U, int max_fev, double t = 1.0,
                          std::vector<double>* prev_best = nullptr) {
    VQEPoint out; out.n = n; out.p = p;
    auto mpo = build_hubbard_mpo(n, t, U, /*two_u1*/true);
    // |psi_0> = the NON-INTERACTING (U=0) ground state, prepared by DMRG. This is the
    // standard HVA state preparation and it is structurally required here: every HVA
    // generator commutes with S^2, so <S^2> is conserved along the whole circuit. The
    // Neel product state has <S^2> = 3 (n=6), not 0, so no depth and no optimiser
    // budget can reach the singlet ground state from it -- measured as a ~35% energy
    // plateau that persists even at U=0, where the exact answer is a free-fermion
    // determinant. The U=0 ground state is a closed-shell singlet, so it starts in
    // the right sector.
    auto mpo0 = build_hubbard_mpo(n, t, 0.0, /*two_u1*/true);
    std::vector<int> cfg(n);
    for (int i = 0; i < n; ++i) cfg[i] = (i % 2 == 0) ? 1 : 2;   // DMRG seed only
    const int target = pack_q2(n, 0);                            // half filling, Sz = 0

    auto t0 = std::chrono::steady_clock::now();
    // prepared ONCE for the whole optimisation (thousands of circuit runs)
    auto psi0 = hva_prepare_init(mpo0, n, chi, target, &cfg);
    int nfev_total = 0, bond_seen = 0;
    auto energy_of = [&](const std::vector<double>& x) {
        HVAResult r = run_hva_circuit(mpo, n, to_layers(x), chi, 1e-12, target, &cfg,
                                      psi0, /*want_entropy*/false);
        ++nfev_total;
        bond_seen = std::max(bond_seen, r.max_bond);
        return r.energy;
    };
    std::mt19937 rng(12345 + 17 * n + 101 * p);
    std::uniform_real_distribution<double> uni(-1.5, 1.5);
    double best = 1e300; std::vector<double> best_x;
    auto try_start = [&](std::vector<double> x0, int budget) {
        NMResult r = nelder_mead(energy_of, x0, 0.3, budget);
        if (r.f < best) { best = r.f; best_x = r.x; }
    };
    // 1) warm start from depth p-1 (new layer = identity), the workhorse
    if (prev_best && (int)prev_best->size() == 3 * (p - 1)) {
        std::vector<double> x0 = *prev_best;
        x0.insert(x0.end(), {0.0, 0.0, 0.0});
        try_start(x0, max_fev / 2);
    }
    // 2) the small-angle (near-identity) point the adiabatic argument motivates
    std::vector<double> xs(3 * p);
    for (int k = 0; k < 3*p; ++k) xs[k] = 0.15 * ((k % 3 == 2) ? -1.0 : 1.0);
    try_start(xs, max_fev / 4);
    // 3) random restarts
    const int n_rand = 3;
    for (int s = 0; s < n_rand; ++s) {
        std::vector<double> x0(3 * p);
        for (int k = 0; k < 3*p; ++k) x0[k] = uni(rng);
        try_start(x0, max_fev / (4 * n_rand));
    }
    // 4) gradient polish (Adam): the simplex gets into the right basin, Adam takes it
    // to the bottom. Two learning rates, coarse then fine.
    {
        int steps = std::max(60, max_fev / (12 * p));
        adam_min(energy_of, best_x, steps, 0.05);
        best = adam_min(energy_of, best_x, steps, 0.01);
    }
    {   // one final run at the optimum, to record the state's entanglement
        HVAResult rb = run_hva_circuit(mpo, n, to_layers(best_x), chi, 1e-12,
                                       target, &cfg, psi0);
        out.S_max = rb.S_max; out.trunc = rb.trunc_err;
    }
    out.E_vqe = best;
    out.nfev = nfev_total;
    out.max_bond = bond_seen;
    out.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (prev_best) *prev_best = best_x;                    // carry forward to depth p+1

    // the exact reference on the same chain (DMRG is exact here to ~1e-10)
    DMRGResult gs = run_dmrg_blocks(mpo, n, 512, 1e-12, 80, 1e-12, target, &cfg);
    out.E_exact = gs.energy;
    out.err = out.E_vqe - out.E_exact;
    out.err_site = out.err / n;
    return out;
}

int main(int argc, char** argv) {
    // QX_SWAPCOST="W Lx chi" -> time ONE x-hop layer of a width-W cylinder through the
    // fermionic-swap network, on a state prepared at that chi. Used to size the width
    // scan from a measurement instead of an estimate.
    if (const char* sc = std::getenv("QX_SWAPCOST")) {
        int W = 3, Lx = 4, chi = 512;
        std::sscanf(sc, "%d %d %d", &W, &Lx, &chi);
        const int n = W * Lx, target = pack_q2(n, 0);
        // W=2 has no wrap (the builder rejects a width-2 cylinder), so derive it rather
        // than assuming a cylinder -- `true` here threw for the ladder width
        auto mpo  = build_hubbard_lattice_split_mpo(Lx, W, 1.0, 4.0, 0.0, W > 2);
        auto hop  = build_hubbard_mpo(n, 1.0, 0.0, true);   // adjacent-hop gate source
        std::vector<int> cfg(n);
        for (int i = 0; i < n; ++i) cfg[i] = (i % 2 == 0) ? 1 : 2;
        auto t0 = std::chrono::steady_clock::now();
        double e = hva_lattice_layer_cost(mpo, hop, n, W, Lx, chi, target, cfg);
        double w = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0).count();
        std::printf("SWAPCOST W=%d Lx=%d n=%d chi=%d : one x-layer = %.2f s  "
                    "(worst trunc %.1e)\n", W, Lx, n, chi, w, e);
        return 0;
    }

    // QX_LATCHK=1 -> at W=1 the lattice driver IS the chain driver: same bonds, same
    // odd/even grouping, no swaps. run_hva is ED-validated, so any disagreement here
    // localises a lattice-driver bug without waiting on the full test suite.
    if (std::getenv("QX_LATCHK")) {
        const int n = 6, target = pack_q2(n, 0);
        auto mpo = build_hubbard_lattice_split_mpo(n, 1, 1.0, 4.0, 0.0, true);
        auto chain = build_hubbard_mpo(n, 1.0, 4.0, true);
        auto hop = build_hubbard_mpo(n, 1.0, 0.0, true);
        std::vector<int> cfg(n);
        for (int i = 0; i < n; ++i) cfg[i] = (i % 2 == 0) ? 1 : 2;
        // groups are {even bonds, odd bonds} -- the same order run_hva_circuit uses
        std::vector<std::vector<double>> ang = {{0.13, 0.26, 0.39}, {-0.06, -0.19, -0.32}};
        std::vector<std::array<double,3>> ac = {{0.13, 0.26, 0.39}, {-0.06, -0.19, -0.32}};
        double e_lat = run_hva_lattice(mpo, hop, n, 1, true, ang, 4096, 1e-14,
                                       target, &cfg, nullptr, false).energy;
        double e_chn = run_hva_circuit(chain, n, ac, 4096, 1e-14, target, &cfg,
                                       nullptr, false).energy;
        // the two DMRG references the circuit energies should be read against -- at W=1
        // the lattice and chain MPOs are the same Hamiltonian, so these must agree too
        double e_gs_lat = run_dmrg_blocks(mpo, n, 256, 1e-12, 60, 1e-12, target, &cfg).energy;
        double e_gs_chn = run_dmrg_blocks(chain, n, 256, 1e-12, 60, 1e-12, target, &cfg).energy;
        std::printf("LATCHK n=6 W=1: circuit lattice=%.10f chain=%.10f  d=%.1e\n",
                    e_lat, e_chn, std::abs(e_lat - e_chn));
        std::printf("               gs      lattice=%.10f chain=%.10f  d=%.1e\n",
                    e_gs_lat, e_gs_chn, std::abs(e_gs_lat - e_gs_chn));
        // isolate: one non-zero angle at a time, single layer
        const char* nm[3] = {"even", "odd ", "onsite"};
        for (int k = 0; k < 3; ++k) {
            std::vector<double> one(3, 0.0); one[k] = 0.37;
            std::array<double,3> onec = {one[0], one[1], one[2]};
            double a = run_hva_lattice(mpo, hop, n, 1, true, {one}, 4096, 1e-14,
                                       target, &cfg, nullptr, false).energy;
            double b = run_hva_circuit(chain, n, {onec}, 4096, 1e-14, target, &cfg,
                                       nullptr, false).energy;
            std::printf("LATCHK   only %s : lattice=%.10f chain=%.10f  d=%.1e\n",
                        nm[k], a, b, std::abs(a - b));
        }
        return 0;
    }

    if (const char* v = std::getenv("QX_VQE")) {          // single point
        int n = 8, p = 2, chi = 256, iters = 800; double U = 4.0;
        std::sscanf(v, "%d %d %d %lf %d", &n, &p, &chi, &U, &iters);
        // grow the depth layer by layer so the requested p is warm-started, exactly
        // as the scan does (a depth-p run from scratch gets stuck -- see vqe_point)
        std::vector<double> carry;
        VQEPoint r;
        for (int q = 1; q <= p; ++q) r = vqe_point(n, q, chi, U, iters, 1.0, &carry);
        std::printf("VQE n=%d (%d qubits) p=%d chi=%d U=%.2f : E_vqe=%.10f "
                    "E_exact=%.10f err=%.3e err/site=%.3e chi_used=%d S_max=%.4f "
                    "trunc=%.1e nfev=%d (%.1fs)\n",
                    n, 2*n, p, chi, U, r.E_vqe, r.E_exact, r.err, r.err_site,
                    r.max_bond, r.S_max, r.trunc, r.nfev, r.wall);
        return 0;
    }

    // ================================================================= PART A ====
    // How deep must the circuit be? Needs the OPTIMISER (thousands of circuit runs
    // per point), so it is restricted to sizes where the simulation is also EXACT:
    // an n-site chain needs chi = 4^(n/2) for a lossless MPS, i.e. 64 at n=6 and 256
    // at n=8. Beyond that the optimiser would be optimising a TRUNCATED circuit --
    // still a valid variational upper bound, but no longer the ansatz's own energy,
    // and 100x more expensive per point. (Measured the hard way: an n=16 point cost
    // ~4 h and sat pinned at the chi cap the whole time.)
    // QX_VQE_QUICK=1 shrinks both parts to a few seconds (smoke test of the whole
    // pipeline, not a physics run).
    const bool quick = std::getenv("QX_VQE_QUICK") != nullptr;
    const double U = 4.0;
    const int iters = quick ? 200 : 2500;
    const std::vector<int> sizes  = quick ? std::vector<int>{4, 6}
                                          : std::vector<int>{4, 6, 8};
    const std::vector<int> depths = quick ? std::vector<int>{1, 2}
                                          : std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
    const int chi_b = quick ? 128 : 4096;
    // State preparation gets its OWN, much smaller chi. |psi_0> is the U=0 ground
    // state -- a Slater determinant whose largest bond entropy grows only like
    // (1/3) log n, so a couple of hundred states is already lossless for it. Passing
    // the circuit's chi budget here instead cost hours: at n=32 the complex-scalar
    // DMRG ran 60 sweeps with chi=4096 allowed (11.6 GB resident) to prepare a state
    // that needs a fraction of that.
    const int chi_prep = quick ? 64 : 256;
    const double cutoff_b = 1e-10;
    // QX_VQE_ONLY=D reruns just the ladder study: parts A/B/C are unaffected by the
    // ladder work and their CSVs are already committed, so redoing them would cost
    // ~2 h for identical numbers.
    const char* only = std::getenv("QX_VQE_ONLY");
    // NOTE strchr(set, NUL) returns a pointer to the terminator, so an EMPTY value
    // would pass validation and then match no part -- the exact silent no-op this
    // check exists to prevent. Only the first character is examined; C runs with B.
    if (only && (*only == char(0) || !std::strchr("AaBbDdEeFf", *only))) {
        std::fprintf(stderr, "QX_VQE_ONLY must be A, B, D, E or F (got '%s')\n", only);
        return 1;
    }
    auto want = [&](char c) { return !only || *only == c || *only == (c ^ 32); };
    std::map<int, std::vector<double>> angles_ref;   // part A -> part B
    // Part B consumes the per-depth optima part A produces, so asking for B alone
    // would otherwise print headers and no rows. B implies A.
    if (want('A') || want('B')) {
    std::printf("# PART A -- VQE depth convergence, 1D Hubbard, U=%.1ft, half filling\n", U);
    std::printf("# (exact simulation: chi is at the full-Hilbert-space bound 4^(n/2))\n");
    std::printf("# n qubits p E_vqe E_exact err err_site chi_used S_max trunc nfev wall_s\n");
    FILE* f = std::fopen("docs/vqe_hubbard.csv", "w");
    if (f) std::fprintf(f, "n,qubits,p,E_vqe,E_exact,err,err_site,chi_used,S_max,trunc,nfev,wall_s\n");
    // angles_ref (declared above, so part B can see it) holds the optimum at EACH
    // depth from the largest exactly-simulable size. Storing per depth matters: the
    // first p layers of a depth-8 optimum are NOT the depth-p optimum (using them
    // made the p=1 transfer land above the starting state).
    for (int n : sizes) {
        int chi_exact = 1; for (int k = 0; k < n / 2; ++k) chi_exact *= 4;
        std::vector<double> carry;              // depth-(p-1) optimum, warm-starts p
        for (int p : depths) {
            VQEPoint r = vqe_point(n, p, chi_exact, U, iters, 1.0, &carry);
            if (n == sizes.back()) angles_ref[p] = carry;   // depth-p optimum
            std::printf("%d %d %d %.10f %.10f %.4e %.4e %d %.6f %.1e %d %.1f\n",
                        n, 2*n, p, r.E_vqe, r.E_exact, r.err, r.err_site,
                        r.max_bond, r.S_max, r.trunc, r.nfev, r.wall);
            std::fflush(stdout);
            if (f) {
                std::fprintf(f, "%d,%d,%d,%.10f,%.10f,%.6e,%.6e,%d,%.6f,%.3e,%d,%.1f\n",
                             n, 2*n, p, r.E_vqe, r.E_exact, r.err, r.err_site,
                             r.max_bond, r.S_max, r.trunc, r.nfev, r.wall);
                std::fflush(f);
            }
        }

    }
    if (f) std::fclose(f);

    }   // end of part A

    if (want('B')) {
    // ================================================================= PART B ====
    // Is the state the circuit prepares classically simulable? This needs NO
    // optimiser -- one circuit run per (n, p) at FIXED angles -- so it is thousands
    // of times cheaper and can be pushed to sizes far past anything optimisable, let
    // alone past quantum hardware. Angles are transferred from the part-A optimum at
    // the largest exactly-simulable size: HVA optimal angles approximate an adiabatic
    // schedule and are only weakly size-dependent, so the transferred circuit is a
    // faithful stand-in for "the VQE circuit" at larger n. (Stated as a transfer, not
    // claimed to be each size's own optimum.)
    // The metric is the largest bond entropy S, not the bond dimension: chi is capped by
    // whatever we can afford, S is not. An MPS needs chi ~ e^S, so S saturating with
    // depth means the ansatz stays classically cheap at ANY qubit count.
    const std::vector<int> sizes_b = quick ? std::vector<int>{6, 8}
                                           : std::vector<int>{8, 12, 16, 24, 32};
    std::printf("\n# PART B -- entanglement of the prepared state (angles transferred "
                "from the n=%d optimum)\n", sizes.back());
    std::printf("# E_circ is the TRANSFERRED circuit's energy (not re-optimised at this\n"
                "# size); frac = captured correlation energy (E_psi0 - E_circ)/(E_psi0 -\n"
                "# E_exact), so the reader can judge how representative the transfer is.\n");
    std::printf("# n qubits p S_max chi_used trunc_err E_circ E_exact frac wall_s\n");
    FILE* g = std::fopen("docs/vqe_entanglement.csv", "w");
    if (g) std::fprintf(g, "n,qubits,p,S_max,chi_used,trunc_err,E_circ,E_exact,frac,wall_s\n");
    for (int n : sizes_b) {
        auto mpo  = build_hubbard_mpo(n, 1.0, U, true);
        auto mpo0 = build_hubbard_mpo(n, 1.0, 0.0, true);
        std::vector<int> cfg(n);
        for (int i = 0; i < n; ++i) cfg[i] = (i % 2 == 0) ? 1 : 2;
        const int target = pack_q2(n, 0);
        auto psi0 = hva_prepare_init(mpo0, n, chi_prep, target, &cfg);
        // context for the transfer: the exact ground state and the starting energy
        double E_exact = run_dmrg_blocks(mpo, n, 512, 1e-12, 60, 1e-11,
                                         target, &cfg).energy;
        double E_psi0 = run_hva_circuit(mpo, n, {}, chi_b, cutoff_b,
                                        target, &cfg, psi0).energy;
        for (int p : depths) {
            auto it = angles_ref.find(p);
            if (it == angles_ref.end()) break;
            const std::vector<double>& x = it->second;
            auto t0 = std::chrono::steady_clock::now();
            HVAResult r = run_hva_circuit(mpo, n, to_layers(x), chi_b, cutoff_b,
                                          target, &cfg, psi0);
            double w = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t0).count();
            double frac = (E_psi0 - E_exact) != 0.0
                        ? (E_psi0 - r.energy) / (E_psi0 - E_exact) : 0.0;
            std::printf("%d %d %d %.6f %d %.2e %.10f %.10f %.4f %.1f\n",
                        n, 2*n, p, r.S_max, r.max_bond, r.trunc_err, r.energy,
                        E_exact, frac, w);
            std::fflush(stdout);
            if (g) {
                std::fprintf(g, "%d,%d,%d,%.6f,%d,%.3e,%.10f,%.10f,%.4f,%.1f\n",
                             n, 2*n, p, r.S_max, r.max_bond, r.trunc_err, r.energy,
                             E_exact, frac, w);
                std::fflush(g);
            }
        }
    }
    if (g) std::fclose(g);

    // ================================================================= PART C ====
    // Does the angle transfer actually stand up? Part B's whole premise is that the
    // depth-p optimum found at the largest exactly-simulable size is a faithful
    // stand-in for the VQE circuit at larger n. Here that assumption is TESTED at one
    // size: re-optimise the transferred angles in place (a short Adam run -- they are
    // already a good warm start) and compare both the energy and, crucially, the
    // largest bond entropy. If S is essentially unchanged, the entanglement conclusion
    // does not depend on the transfer; if it moves a lot, Part B's S values are only
    // indicative and the doc must say so.
    const int n_c = quick ? 6 : 12;
    const int adam_steps = quick ? 3 : 25;
    const std::vector<int> depths_c = quick ? std::vector<int>{2}
                                            : std::vector<int>{2, 4, 6, 8};
    std::printf("\n# PART C -- transfer check at n=%d: transferred angles vs a short "
                "re-optimisation\n", n_c);
    std::printf("# n p S_transfer S_reopt dS E_transfer E_reopt E_exact wall_s\n");
    FILE* h = std::fopen("docs/vqe_transfer_check.csv", "w");
    if (h) std::fprintf(h, "n,p,S_transfer,S_reopt,dS,E_transfer,E_reopt,E_exact,wall_s\n");
    {
        auto mpo  = build_hubbard_mpo(n_c, 1.0, U, true);
        auto mpo0 = build_hubbard_mpo(n_c, 1.0, 0.0, true);
        std::vector<int> cfg(n_c);
        for (int i = 0; i < n_c; ++i) cfg[i] = (i % 2 == 0) ? 1 : 2;
        const int target = pack_q2(n_c, 0);
        auto psi0 = hva_prepare_init(mpo0, n_c, chi_prep, target, &cfg);
        double E_exact = run_dmrg_blocks(mpo, n_c, 512, 1e-12, 60, 1e-11,
                                         target, &cfg).energy;
        for (int p : depths_c) {
            auto it = angles_ref.find(p);
            if (it == angles_ref.end()) continue;
            auto t0 = std::chrono::steady_clock::now();
            HVAResult rt = run_hva_circuit(mpo, n_c, to_layers(it->second), chi_b,
                                           cutoff_b, target, &cfg, psi0);
            std::vector<double> x = it->second;
            auto energy_of = [&](const std::vector<double>& y) {
                return run_hva_circuit(mpo, n_c, to_layers(y), chi_b, cutoff_b,
                                       target, &cfg, psi0, false).energy;
            };
            double E_re = adam_min(energy_of, x, adam_steps, 0.02);
            HVAResult rr = run_hva_circuit(mpo, n_c, to_layers(x), chi_b, cutoff_b,
                                           target, &cfg, psi0);
            double w = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t0).count();
            std::printf("%d %d %.6f %.6f %+.6f %.10f %.10f %.10f %.1f\n",
                        n_c, p, rt.S_max, rr.S_max, rr.S_max - rt.S_max,
                        rt.energy, E_re, E_exact, w);
            std::fflush(stdout);
            if (h) {
                std::fprintf(h, "%d,%d,%.6f,%.6f,%+.6f,%.10f,%.10f,%.10f,%.1f\n",
                             n_c, p, rt.S_max, rr.S_max, rr.S_max - rt.S_max,
                             rt.energy, E_re, E_exact, w);
                std::fflush(h);
            }
        }
    }
    if (h) std::fclose(h);
    }   // end of parts B/C

    if (want('D')) {
    // ================================================================= PART D ====
    // Geometry at MATCHED qubit count: does a quasi-2D system entangle more than a
    // chain? The 2-leg Hubbard ladder is rung-fused, so each MPS site is a whole rung
    // (d=16) and the chain is nearest-neighbour again -- and its on-site block carries
    // the RUNG HOPPING, making this the first VQE layer here that is a general
    // one-site unitary rather than a free phase.
    //
    // Each geometry uses ITS OWN optimised angles, at sizes where the simulation is
    // lossless by construction (chi = 16^(Lx/2) for the ladder). Two protocols were
    // tried and rejected first, and both failures are the reason this one is small:
    //   * transferring the Lx=4 optimum upward -- the ladder optimum at 4 rungs is
    //     far from converged (61.5% of the correlation energy at the depths reached),
    //     i.e. it is overfitted to that size,
    //     and at Lx >= 6 it gave frac < 0 (the circuit ended ABOVE the state it
    //     started from) with S inflated to 4.9-6.2 purely by scrambling;
    //   * a fixed adiabatic ramp -- at the depths a VQE actually uses, tau/p is far
    //     too coarse to be adiabatic, and the circuit HEATS the state (frac ~ -4).
    // So the comparison is deliberately restricted to exactly-simulable, properly
    // optimised points rather than being large and meaningless.
    // Lx >= 3: derive_mpo_sym needs a square bulk MPO site, so a 2-rung ladder throws.
    const std::vector<int> lx_d = quick ? std::vector<int>{3}
                                        : std::vector<int>{3, 4};
    std::printf("\n# PART D -- 2-leg ladder (rung-fused d=16) at its OWN optimum\n");
    std::printf("# compare against the chain at matched qubits (part A/B: 2n qubits)\n");
    std::printf("# Lx qubits p E_vqe E_exact frac S_max chi_used trunc nfev wall_s\n");
    std::fflush(stdout);
    FILE* q = std::fopen("docs/vqe_ladder.csv", "w");
    if (q) std::fprintf(q, "Lx,qubits,p,E_vqe,E_exact,frac,S_max,chi_used,trunc,nfev,wall_s\n");
    for (int Lx : lx_d) {
        const int nm = 2 * Lx, target = pack_q2(nm, 0);
        auto mpo  = build_hubbard_ladder_mpo(Lx, 1.0, 1.0, U);
        auto mpo0 = build_hubbard_ladder_mpo(Lx, 1.0, 1.0, 0.0);
        std::vector<int> cfg(Lx);
        for (int x = 0; x < Lx; ++x) cfg[x] = (x % 2 == 0) ? (4*1 + 2) : (4*2 + 1);
        int chi_ex = 1; for (int k = 0; k < Lx / 2; ++k) chi_ex *= 16;   // lossless
        auto psi0 = hva_prepare_init(mpo0, Lx, chi_ex, target, &cfg);
        double E_exact = run_dmrg_blocks(mpo, Lx, 512, 1e-12, 60, 1e-11,
                                         target, &cfg).energy;
        double E_psi0 = run_hva_circuit(mpo, Lx, {}, chi_ex, 1e-12,
                                        target, &cfg, psi0).energy;
        std::vector<double> carry;
        for (int p : depths) {
            auto t0 = std::chrono::steady_clock::now();
            int nfev = 0;
            auto energy_of = [&](const std::vector<double>& x) {
                ++nfev;
                return run_hva_circuit(mpo, Lx, to_layers(x), chi_ex, 1e-12,
                                       target, &cfg, psi0, false).energy;
            };
            // SAME search protocol as part A (warm start + small-angle start + three
            // random restarts + Adam polish). Anything weaker here would show up as
            // "the ladder converges worse" -- the very claim this part makes -- so
            // the two geometries must be searched identically.
            std::mt19937 rng(9871 + 31 * Lx + 7 * p);
            std::uniform_real_distribution<double> uni(-1.5, 1.5);
            double best = 1e300; std::vector<double> best_x;
            auto try_start = [&](std::vector<double> x0, int budget) {
                NMResult r0 = nelder_mead(energy_of, x0, 0.3, budget);
                if (r0.f < best) { best = r0.f; best_x = r0.x; }
            };
            { std::vector<double> x0 = carry;               // layer-wise warm start
              x0.insert(x0.end(), {0.0, 0.0, 0.0});
              try_start(x0, iters / 2); }
            { std::vector<double> xs(3 * p);                // near-identity start
              for (int k = 0; k < 3*p; ++k) xs[k] = 0.15 * ((k % 3 == 2) ? -1.0 : 1.0);
              try_start(xs, iters / 4); }
            for (int r = 0; r < 3; ++r) {                   // random restarts
                std::vector<double> x0(3 * p);
                for (int k = 0; k < 3*p; ++k) x0[k] = uni(rng);
                try_start(x0, iters / 12);
            }
            carry = best_x;
            int st = std::max(60, iters / (12 * p));
            adam_min(energy_of, carry, st, 0.05);
            double E = adam_min(energy_of, carry, st, 0.01);
            HVAResult rb = run_hva_circuit(mpo, Lx, to_layers(carry), chi_ex, 1e-12,
                                           target, &cfg, psi0);
            double w = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t0).count();
            double frac = (E_psi0 - E_exact) != 0.0
                        ? (E_psi0 - E) / (E_psi0 - E_exact) : 0.0;
            std::printf("%d %d %d %.10f %.10f %.4f %.6f %d %.1e %d %.1f\n",
                        Lx, 4*Lx, p, E, E_exact, frac, rb.S_max, rb.max_bond,
                        rb.trunc_err, nfev, w);
            std::fflush(stdout);
            if (q) {
                std::fprintf(q, "%d,%d,%d,%.10f,%.10f,%.4f,%.6f,%d,%.3e,%d,%.1f\n",
                             Lx, 4*Lx, p, E, E_exact, frac, rb.S_max, rb.max_bond,
                             rb.trunc_err, nfev, w);
                std::fflush(q);
            }
        }
    }
    if (q) std::fclose(q);
    }   // end of part D

    if (want('E')) {
    // ================================================================= PART E ====
    // The width scan, done so the three geometries are actually comparable.
    //
    // Part D compared a rung-fused ladder (d=16, cuts between rungs) against a chain
    // (d=4, cuts between sites). Those cuts are not the same object, so an entropy
    // difference between them is partly a difference of representation. Here all three
    // widths run in the SAME site-split representation through the SAME driver:
    //
    //   W=1 Lx=6  chain        W=2 Lx=3  ladder        W=3 Lx=2  cylinder
    //
    // n = W*Lx = 6 sites and 12 qubits for every one; d = 4 for every one; the
    // balanced cut is bond 2 (3|3 sites = 6|6 qubits) for every one; chi = 4^3 = 64
    // is lossless for every one. The only thing that varies is the connectivity graph
    // -- which is exactly the variable under study.
    //
    // Every geometry is optimised at its OWN optimum with an identical protocol
    // (warm start + small-angle start + 3 random restarts + two Adam refinements).
    // The wider geometries get MORE angles per layer, which favours them; a worse
    // energy at equal depth is therefore the stronger claim.
    //
    // Small on purpose. Measured with QX_SWAPCOST, not guessed -- one x-hop layer:
    //   W=3 Lx=2 chi=64   0.08 s      (this scan)
    //   W=4 Lx=2 chi=256  0.59 s
    //   W=3 Lx=4 chi=256  5.76 s      W=3 Lx=4 chi=512  10.65 s
    // A VQE needs thousands of circuit evaluations per point and each evaluation is
    // many layers, so an optimised scan at 24+ qubits is tens of hours PER POINT on
    // this box. Hence 12 qubits, where all three widths are also exactly simulable.
    const int n_e = 6, target_e = pack_q2(n_e, 0);
    int chi_e = 1; for (int k = 0; k < n_e / 2; ++k) chi_e *= 4;      // 64, lossless
    const int widths[3] = {1, 2, 3};
    std::vector<int> cfg_e(n_e);
    for (int i = 0; i < n_e; ++i) cfg_e[i] = (i % 2 == 0) ? 1 : 2;
    std::printf("\n# PART E -- width scan, all three geometries site-split at 12 qubits\n");
    std::printf("# W Lx qubits nbond ngroup nangle p E_vqe E_exact frac S_mid S_max chi trunc nfev wall_s\n");
    std::fflush(stdout);
    FILE* fe = std::fopen("docs/vqe_width.csv", "w");
    if (fe) std::fprintf(fe, "W,Lx,qubits,nbond,ngroup,nangle,p,E_vqe,E_exact,frac,S_mid,S_max,chi,trunc,nfev,wall_s\n");
    for (int wi = 0; wi < 3; ++wi) {
        const int W_e = widths[wi], Lx_e = n_e / W_e;
        // the y-wrap exists only for W>2: at W=2 it would duplicate the rung (the MPO
        // builder rejects it outright) and at W=1 there is no y direction at all
        const bool yp = (W_e > 2);
        auto mpo_e  = build_hubbard_lattice_split_mpo(Lx_e, W_e, 1.0, U, 0.0, yp);
        auto mpo0_e = build_hubbard_lattice_split_mpo(Lx_e, W_e, 1.0, 0.0, 0.0, yp);
        auto hop_e  = build_hubbard_mpo(n_e, 1.0, 0.0, true);
        const int ng_e = hva_lattice_groups(Lx_e, W_e, yp);       // bond groups + on-site
        const int nb_e = hva_lattice_nbonds(Lx_e, W_e, yp);
        auto psi0 = hva_prepare_init(mpo0_e, n_e, chi_e, target_e, &cfg_e);
        double E_exact = run_dmrg_blocks(mpo_e, n_e, 512, 1e-12, 60, 1e-11,
                                         target_e, &cfg_e).energy;
        double E_psi0 = run_hva_lattice(mpo_e, hop_e, Lx_e, W_e, yp, {}, chi_e,
                                        1e-12, target_e, &cfg_e, psi0, false).energy;
        std::vector<double> carry;
        for (int p : depths) {
            auto t0 = std::chrono::steady_clock::now();
            int nfev = 0;
            auto to_layers_e = [&](const std::vector<double>& x) {
                std::vector<std::vector<double>> a(x.size() / ng_e);
                for (size_t l = 0; l < a.size(); ++l)
                    a[l].assign(x.begin() + l * ng_e, x.begin() + (l + 1) * ng_e);
                return a;
            };
            auto energy_of = [&](const std::vector<double>& x) {
                ++nfev;
                return run_hva_lattice(mpo_e, hop_e, Lx_e, W_e, yp, to_layers_e(x),
                                       chi_e, 1e-12, target_e, &cfg_e, psi0, false).energy;
            };
            // IDENTICAL search protocol to parts A and D
            std::mt19937 rng(4421 + 13 * p + 101 * W_e);
            std::uniform_real_distribution<double> uni(-1.5, 1.5);
            double best = 1e300; std::vector<double> best_x;
            auto try_start = [&](std::vector<double> x0, int budget) {
                NMResult r0 = nelder_mead(energy_of, x0, 0.3, budget);
                if (r0.f < best) { best = r0.f; best_x = r0.x; }
            };
            { std::vector<double> x0 = carry;
              x0.insert(x0.end(), ng_e, 0.0);
              try_start(x0, iters / 2); }
            { std::vector<double> xs(ng_e * p);
              for (int k = 0; k < ng_e * p; ++k)
                  xs[k] = 0.15 * ((k % ng_e == ng_e - 1) ? -1.0 : 1.0);
              try_start(xs, iters / 4); }
            for (int r = 0; r < 3; ++r) {
                std::vector<double> x0(ng_e * p);
                for (int k = 0; k < ng_e * p; ++k) x0[k] = uni(rng);
                try_start(x0, iters / 12);
            }
            carry = best_x;
            int st = std::max(60, iters / (12 * p));
            adam_min(energy_of, carry, st, 0.05);
            double E = adam_min(energy_of, carry, st, 0.01);
            HVAResult rb = run_hva_lattice(mpo_e, hop_e, Lx_e, W_e, yp,
                                           to_layers_e(carry), chi_e, 1e-12,
                                           target_e, &cfg_e, psi0, true);
            double w = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t0).count();
            double frac = (E_psi0 - E_exact) != 0.0
                        ? (E_psi0 - E) / (E_psi0 - E_exact) : 0.0;
            std::printf("%d %d %d %d %d %d %d %.10f %.10f %.4f %.6f %.6f %d %.1e %d %.1f\n",
                        W_e, Lx_e, 2 * n_e, nb_e, ng_e - 1, ng_e * p, p, E, E_exact,
                        frac, rb.S_mid, rb.S_max, rb.max_bond, rb.trunc_err, nfev, w);
            std::fflush(stdout);
            if (fe) {
                std::fprintf(fe, "%d,%d,%d,%d,%d,%d,%d,%.10f,%.10f,%.4f,%.6f,%.6f,%d,%.3e,%d,%.1f\n",
                             W_e, Lx_e, 2 * n_e, nb_e, ng_e - 1, ng_e * p, p, E, E_exact,
                             frac, rb.S_mid, rb.S_max, rb.max_bond, rb.trunc_err, nfev, w);
                std::fflush(fe);
            }
        }
    }
    if (fe) std::fclose(fe);
    }   // end of part E

    if (want('F')) {
    // ================================================================= PART F ====
    // Barren plateaus vs WIDTH -- the question part E's result actually raises.
    //
    // Part E measured that a wider lattice reaches a worse energy at equal depth. That
    // is consistent with two different stories: the ansatz cannot REPRESENT the state
    // (expressibility), or it can but the landscape is too flat to FIND it
    // (trainability). They have opposite implications -- the first says add layers, the
    // second says adding layers will not help -- and part E cannot tell them apart. The
    // standard trainability diagnostic is the variance of a cost gradient over random
    // parameter draws: a barren plateau is Var[dE/dtheta] decaying exponentially.
    //
    // THE INSTRUMENT COMES FIRST. A barren plateau IS a vanishing gradient, so a study
    // that cannot separate a small gradient from its own numerical noise measures
    // nothing. There is no parameter-shift rule here (the HVA generators are not single
    // Paulis), so the gradient is a central difference carrying an h^2 truncation error
    // plus an O(eps/h) roundoff error. Stage 1 measures that error directly by
    // Richardson extrapolation: g_rich = (4 g(h/2) - g(h))/3 cancels the h^2 term, so
    // |g(h) - g_rich| estimates the error in g(h) while assuming nothing about the
    // function. Stage 2 reports every row's spread as a multiple of that floor, and a
    // row that is not well clear of it is measuring the difference engine, not physics.
    //
    // The energy itself needs no new validation: the circuit is already pinned against
    // a dense state-vector reference at 1e-12 for all three widths (qx_hubbard_test
    // section r), and the gradient is a finite difference OF that energy.
    //
    // NORMALISATION. The three geometries have 5, 7 and 9 bonds, so dE/dtheta is not
    // comparable across them -- a wider lattice has a bigger H and would show a bigger
    // gradient for reasons that have nothing to do with the landscape. Everything is
    // therefore reported for the DIMENSIONLESS cost frac = (E0 - E)/(E0 - E_exact),
    // the same correlation-energy fraction part E ranks the widths by, whose gradient
    // is (dE/dtheta)/(E0 - E_exact). The raw dE/dtheta is emitted alongside it so the
    // normalisation can be checked rather than trusted.
    const int n_f = 6, target_f = pack_q2(n_f, 0);
    int chi_f = 1; for (int k = 0; k < n_f / 2; ++k) chi_f *= 4;      // 64, lossless
    int nsamp = quick ? 24 : 400;
    if (const char* s = std::getenv("QX_VQE_BP")) {
        int v = 0; std::sscanf(s, "%d", &v);
        // A variance needs two samples, and its fourth-moment error estimate several.
        // QX_VQE_BP=1 used to be accepted: stage 2 then wrote zero for every variance
        // and standard error, analyze_vqe_barren.py divided by them -- and the run had
        // already overwritten the committed CSVs. Refuse before writing anything.
        if (v < 10) {
            std::fprintf(stderr, "QX_VQE_BP=%s: need at least 10 samples per point for a "
                                 "variance and its fourth-moment error; nothing written\n", s);
            std::exit(2);
        }
        nsamp = v;
    }
    std::vector<int> cfg_f(n_f);
    for (int i = 0; i < n_f; ++i) cfg_f[i] = (i % 2 == 0) ? 1 : 2;
    auto hop_f = build_hubbard_mpo(n_f, 1.0, 0.0, true);

    std::printf("\n# PART F -- gradient variance vs width (barren-plateau diagnostic)\n");
    std::fflush(stdout);

    // ---- stage 1: what is this instrument's noise floor? ----------------------
    // One representative point per width, angles from the same distribution the scan
    // uses. Report g(h) and the Richardson error estimate across five decades of h;
    // the useful h is where the error estimate bottoms out.
    std::printf("# calibration: central-difference error vs step size (Richardson)\n");
    std::printf("# CAL W h grad err_est\n");
    double floor_abs = 0.0;
    const double h_f = 1e-5;
    for (int W_f = 1; W_f <= 3; ++W_f) {
        const int Lx_f = n_f / W_f;
        const bool yp = (W_f > 2);
        auto mpo_f  = build_hubbard_lattice_split_mpo(Lx_f, W_f, 1.0, U, 0.0, yp);
        auto mpo0_f = build_hubbard_lattice_split_mpo(Lx_f, W_f, 1.0, 0.0, 0.0, yp);
        const int ng_f = hva_lattice_groups(Lx_f, W_f, yp);
        auto psi0_f = hva_prepare_init(mpo0_f, n_f, chi_f, target_f, &cfg_f);
        const int p_cal = 4, dim = ng_f * p_cal, kmid = (p_cal / 2) * ng_f;
        // THE CONFOUND CHECK. theta_g generates exp(-i theta_g H_g) with H_g the SUM
        // over the bonds in group g, so ||H_g|| grows with the group's SIZE. If group
        // 0 holds a different number of bonds at different widths, a gradient gap
        // between geometries is partly that and nothing to do with the landscape --
        // the same trap as comparing entropies across different kinds of cut. Emit
        // the group sizes so the normalisation can be checked rather than assumed.
        {
            auto gb = hva_lattice_group_bonds(Lx_f, W_f, yp);
            std::printf("GRP %d groups=%d sizes=", W_f, (int)gb.size());
            for (size_t gi = 0; gi < gb.size(); ++gi)
                std::printf("%s%d", gi ? "/" : "", (int)gb[gi].size());
            std::printf(" differentiated=group0(%d bonds)\n", (int)gb[0].size());
        }
        auto energy_of = [&](const std::vector<double>& x) {
            std::vector<std::vector<double>> a(x.size() / ng_f);
            for (size_t l = 0; l < a.size(); ++l)
                a[l].assign(x.begin() + l * ng_f, x.begin() + (l + 1) * ng_f);
            return run_hva_lattice(mpo_f, hop_f, Lx_f, W_f, yp, a, chi_f, 1e-12,
                                   target_f, &cfg_f, psi0_f, false).energy;
        };
        std::mt19937 rng(20260818u + (unsigned)W_f);
        std::uniform_real_distribution<double> UU(-3.14159265358979323846,
                                                   3.14159265358979323846);
        std::vector<double> x(dim);
        for (int k = 0; k < dim; ++k) x[k] = UU(rng);
        auto grad_at = [&](double h) {
            double x0 = x[kmid];
            x[kmid] = x0 + h; double fp = energy_of(x);
            x[kmid] = x0 - h; double fm = energy_of(x);
            x[kmid] = x0;
            return (fp - fm) / (2 * h);
        };
        const double hs[6] = {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7};
        for (int hi = 0; hi < 6; ++hi) {
            double h = hs[hi];
            double g1 = grad_at(h), g2 = grad_at(h / 2);
            double err = std::abs(g1 - (4 * g2 - g1) / 3.0);   // |g(h) - g_richardson|
            std::printf("CAL %d %.0e %+.12f %.2e\n", W_f, h, g1, err);
            if (h == h_f) floor_abs = std::max(floor_abs, err);
        }
        std::fflush(stdout);
    }
    std::printf("# noise floor (worst |g(h) - g_rich| at h=%.0e over widths) = %.2e\n",
                h_f, floor_abs);
    std::fflush(stdout);

    // ---- stage 2: the scan ----------------------------------------------------
    // theta ~ U(-pi, pi) independently per angle -- the standard barren-plateau setup,
    // and the honest one here: it is the distribution a VQE with no warm start actually
    // initialises from. The differentiated parameter is the FIRST BOND-GROUP angle of
    // the MIDDLE layer, which exists at every width (the widths differ in how many
    // groups they have, not in whether they have a group 0) and sits where the circuit
    // is deepest on both sides.
    std::printf("# W Lx qubits nbond ngroup nb_g0 p nangle var_frac se_frac std_frac var_raw "
                "mean_frac floor snr nsamp wall_s\n");
    std::fflush(stdout);
    FILE* ff = std::fopen("docs/vqe_barren.csv", "w");
    if (ff) std::fprintf(ff, "W,Lx,qubits,nbond,ngroup,nb_g0,p,nangle,var_frac,se_frac,std_frac,"
                             "var_raw,mean_frac,floor,snr,nsamp,wall_s\n");
    std::vector<int> bp_depths;
    if (quick) { bp_depths = {2, 4}; }
    else       { bp_depths = {2, 4, 6, 8, 10, 12}; }
    for (int W_f = 1; W_f <= 3; ++W_f) {
        const int Lx_f = n_f / W_f;
        const bool yp = (W_f > 2);
        auto mpo_f  = build_hubbard_lattice_split_mpo(Lx_f, W_f, 1.0, U, 0.0, yp);
        auto mpo0_f = build_hubbard_lattice_split_mpo(Lx_f, W_f, 1.0, 0.0, 0.0, yp);
        const int ng_f = hva_lattice_groups(Lx_f, W_f, yp);
        const int nb_f = hva_lattice_nbonds(Lx_f, W_f, yp);
        const int nb_g0 = (int)hva_lattice_group_bonds(Lx_f, W_f, yp)[0].size();
        auto psi0_f = hva_prepare_init(mpo0_f, n_f, chi_f, target_f, &cfg_f);
        double E_exact = run_dmrg_blocks(mpo_f, n_f, 512, 1e-12, 60, 1e-11,
                                         target_f, &cfg_f).energy;
        double E_psi0 = run_hva_lattice(mpo_f, hop_f, Lx_f, W_f, yp, {}, chi_f,
                                        1e-12, target_f, &cfg_f, psi0_f, false).energy;
        const double corr = E_psi0 - E_exact;          // the frac denominator
        auto energy_of = [&](const std::vector<double>& x) {
            std::vector<std::vector<double>> a(x.size() / ng_f);
            for (size_t l = 0; l < a.size(); ++l)
                a[l].assign(x.begin() + l * ng_f, x.begin() + (l + 1) * ng_f);
            return run_hva_lattice(mpo_f, hop_f, Lx_f, W_f, yp, a, chi_f, 1e-12,
                                   target_f, &cfg_f, psi0_f, false).energy;
        };
        for (size_t pi_ = 0; pi_ < bp_depths.size(); ++pi_) {
            const int p = bp_depths[pi_];
            auto t0 = std::chrono::steady_clock::now();
            const int dim = ng_f * p, kmid = (p / 2) * ng_f;
            // fixed seed per (W,p): the comparison is between geometries, so sampling
            // noise should not be an extra uncontrolled difference between them
            std::mt19937 rng(20260818u + 1000u * (unsigned)W_f + (unsigned)p);
            std::uniform_real_distribution<double> UU(-3.14159265358979323846,
                                                       3.14159265358979323846);
            double s1 = 0, s2 = 0, s3 = 0, s4 = 0;   // raw moments -> real error bar
            std::vector<double> x(dim);
            for (int t = 0; t < nsamp; ++t) {
                for (int k = 0; k < dim; ++k) x[k] = UU(rng);
                double x0 = x[kmid];
                x[kmid] = x0 + h_f; double fp = energy_of(x);
                x[kmid] = x0 - h_f; double fm = energy_of(x);
                double g = (fp - fm) / (2 * h_f);
                s1 += g; s2 += g * g; s3 += g * g * g; s4 += g * g * g * g;
            }
            const double mean_raw = s1 / nsamp;
            const double var_raw  = std::max(0.0, s2 / nsamp - mean_raw * mean_raw);
            const double var_frac = var_raw / (corr * corr);
            const double std_frac = std::sqrt(var_frac);
            const double se_frac = var_stderr(s1, s2, s3, s4, nsamp) / (corr * corr);
            // how far the measured gradient SPREAD sits above the instrument's own
            // error -- the number that decides whether a row is physics or noise
            const double snr = floor_abs > 0 ? std::sqrt(var_raw) / floor_abs : 0.0;
            double w = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t0).count();
            std::printf("%d %d %d %d %d %d %d %d %.6e %.6e %.6e %.6e %+.6e %.2e %.1f %d %.1f\n",
                        W_f, Lx_f, 2 * n_f, nb_f, ng_f - 1, nb_g0, p, dim,
                        var_frac, se_frac, std_frac, var_raw, mean_raw / corr, floor_abs,
                        snr, nsamp, w);
            std::fflush(stdout);
            if (ff) {
                std::fprintf(ff, "%d,%d,%d,%d,%d,%d,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.3e,%.1f,%d,%.1f\n",
                             W_f, Lx_f, 2 * n_f, nb_f, ng_f - 1, nb_g0, p, dim,
                             var_frac, se_frac, std_frac, var_raw, mean_raw / corr,
                             floor_abs, snr, nsamp, w);
                std::fflush(ff);
            }
        }
    }
    if (ff) std::fclose(ff);

    // ---- stage 3: the SIZE axis ------------------------------------------------
    // Stage 2 compares geometries at ONE size, and that is not enough to say "barren
    // plateau": a barren plateau is defined by the variance decaying EXPONENTIALLY in
    // system size, and at fixed n a constant factor and e^{-cn} evaluated at that n are
    // the same number. So this stage holds the depth fixed and grows n, per width, and
    // reports the decay rate -- which is the quantity the comparison is actually about.
    //
    // COST IS MEASURED, NOT ASSUMED. chi for a lossless simulation is 4^(n/2), so the
    // cost per circuit evaluation grows steeply and the affordable n differs per width
    // (the cylinder pays for range-W swaps on top). Each point therefore TIMES one
    // evaluation first and skips itself, loudly and with the projected cost, if the
    // sample would not finish inside the budget. A point missing from the CSV should be
    // a recorded decision, not a silent hole.
    const double budget_s = [&]{
        const char* b = std::getenv("QX_VQE_BPBUDGET");
        double v = 0; if (b) std::sscanf(b, "%lf", &v);
        return v > 0 ? v : 900.0;
    }();
    const int nsamp_n = std::max(50, nsamp / 3);   // ~8% SE on a variance at 300
    const int chi_cap = [&]{
        const char* c = std::getenv("QX_VQE_BPCHI");
        int v = 0; if (c) std::sscanf(c, "%d", &v);
        return v > 0 ? v : 1024;      // n<=10; 4^6=4096 at n=12 is not simulable here
    }();
    const int p_n = 8;
    std::printf("\n# size axis at p=%d, budget %.0f s/point, %d samples\n",
                p_n, budget_s, nsamp_n);
    std::printf("# W n Lx qubits chi nb_g0 p var_frac se_frac std_frac var_raw floor snr "
                "nsamp t_eval wall_s\n");
    std::fflush(stdout);
    FILE* fn = std::fopen("docs/vqe_barren_size.csv", "w");
    if (fn) std::fprintf(fn, "W,n,Lx,qubits,chi,nb_g0,p,var_frac,se_frac,std_frac,var_raw,"
                             "floor,snr,nsamp,t_eval,wall_s\n");
    for (int W_n = 1; W_n <= 3; ++W_n) {
        for (int n_n = 4; n_n <= 12; ++n_n) {
            if (n_n % W_n != 0) continue;              // the lattice must be rectangular
            const int Lx_n = n_n / W_n;
            if (Lx_n < 2) continue;                    // need at least one x-hop
            const bool yp = (W_n > 2);
            int chi_n = 1; for (int k = 0; k < n_n / 2; ++k) chi_n *= 4;   // lossless
            // The chi gate has to come BEFORE the timing probe, not after it: the
            // probe measures one circuit evaluation, but reaching it requires
            // hva_prepare_init to converge a DMRG ground state at this chi first.
            // At n=12 that is chi=4096 with d=4 -- ~1 GB per tensor -- so the
            // "measure the cost" step would itself hang before reporting anything.
            if (chi_n > chi_cap) {
                std::printf("SKIP W=%d n=%d (%d qubits): lossless chi=%d exceeds the chi cap %d; state preparation alone is out of reach on this box\n",
                            W_n, n_n, 2 * n_n, chi_n, chi_cap);
                std::fflush(stdout);
                continue;
            }
            std::vector<int> cfg_n(n_n);
            for (int i = 0; i < n_n; ++i) cfg_n[i] = (i % 2 == 0) ? 1 : 2;
            // half filling at ODD n carries 2Sz = 1, not 0 -- take the target from the
            // configuration rather than assuming a balanced sector, or n=9 (the only
            // size between 6 and 12 the cylinder admits) would be unreachable
            auto mpo_n  = build_hubbard_lattice_split_mpo(Lx_n, W_n, 1.0, U, 0.0, yp);
            auto mpo0_n = build_hubbard_lattice_split_mpo(Lx_n, W_n, 1.0, 0.0, 0.0, yp);
            auto hop_n  = build_hubbard_mpo(n_n, 1.0, 0.0, true);
            const std::vector<int>& qp = mpo_n[0].phys_charge;
            int target_n = 0;
            for (int i = 0; i < n_n; ++i) target_n += qp[cfg_n[i]];
            const int ng_n = hva_lattice_groups(Lx_n, W_n, yp);
            const int nb_g0n = (int)hva_lattice_group_bonds(Lx_n, W_n, yp)[0].size();
            auto psi0_n = hva_prepare_init(mpo0_n, n_n, chi_n, target_n, &cfg_n);
            auto energy_of = [&](const std::vector<double>& x) {
                std::vector<std::vector<double>> a(x.size() / ng_n);
                for (size_t l = 0; l < a.size(); ++l)
                    a[l].assign(x.begin() + l * ng_n, x.begin() + (l + 1) * ng_n);
                return run_hva_lattice(mpo_n, hop_n, Lx_n, W_n, yp, a, chi_n, 1e-12,
                                       target_n, &cfg_n, psi0_n, false).energy;
            };
            const int dim = ng_n * p_n, kmid = (p_n / 2) * ng_n;
            std::mt19937 rng(20260818u + 7000u * (unsigned)W_n + (unsigned)n_n);
            std::uniform_real_distribution<double> UU(-3.14159265358979323846,
                                                       3.14159265358979323846);
            std::vector<double> x(dim);
            for (int k = 0; k < dim; ++k) x[k] = UU(rng);
            auto tc = std::chrono::steady_clock::now();
            volatile double e_probe = energy_of(x);
            (void)e_probe;
            double t_eval = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - tc).count();
            double projected = 2.0 * nsamp_n * t_eval;
            if (projected > budget_s) {
                std::printf("SKIP W=%d n=%d (%d qubits, chi=%d): one evaluation %.2f s "
                            "-> %.0f s projected, over the %.0f s budget\n",
                            W_n, n_n, 2 * n_n, chi_n, t_eval, projected, budget_s);
                std::fflush(stdout);
                continue;
            }
            double E_exact = run_dmrg_blocks(mpo_n, n_n, 512, 1e-12, 60, 1e-11,
                                             target_n, &cfg_n).energy;
            double E_psi0 = run_hva_lattice(mpo_n, hop_n, Lx_n, W_n, yp, {}, chi_n,
                                            1e-12, target_n, &cfg_n, psi0_n, false).energy;
            const double corr = E_psi0 - E_exact;
            auto t0 = std::chrono::steady_clock::now();
            double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
            for (int t = 0; t < nsamp_n; ++t) {
                for (int k = 0; k < dim; ++k) x[k] = UU(rng);
                double x0 = x[kmid];
                x[kmid] = x0 + h_f; double fp = energy_of(x);
                x[kmid] = x0 - h_f; double fm = energy_of(x);
                double g = (fp - fm) / (2 * h_f);
                s1 += g; s2 += g * g; s3 += g * g * g; s4 += g * g * g * g;
            }
            const double mean_raw = s1 / nsamp_n;
            const double var_raw  = std::max(0.0, s2 / nsamp_n - mean_raw * mean_raw);
            const double var_frac = var_raw / (corr * corr);
            const double se_frac = var_stderr(s1, s2, s3, s4, nsamp_n) / (corr * corr);
            const double snr = floor_abs > 0 ? std::sqrt(var_raw) / floor_abs : 0.0;
            double w = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - t0).count();
            std::printf("%d %d %d %d %d %d %d %.6e %.6e %.6e %.6e %.2e %.1f %d %.3f %.1f\n",
                        W_n, n_n, Lx_n, 2 * n_n, chi_n, nb_g0n, p_n, var_frac, se_frac,
                        std::sqrt(var_frac), var_raw, floor_abs, snr, nsamp_n,
                        t_eval, w);
            std::fflush(stdout);
            if (fn) {
                std::fprintf(fn, "%d,%d,%d,%d,%d,%d,%d,%.6e,%.6e,%.6e,%.6e,%.3e,%.1f,%d,%.4f,%.1f\n",
                             W_n, n_n, Lx_n, 2 * n_n, chi_n, nb_g0n, p_n, var_frac, se_frac,
                             std::sqrt(var_frac), var_raw, floor_abs, snr, nsamp_n,
                             t_eval, w);
                std::fflush(fn);
            }
        }
    }
    if (fn) std::fclose(fn);
    }   // end of part F
    return 0;
}
