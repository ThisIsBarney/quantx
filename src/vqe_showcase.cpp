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
    if (only && (*only == char(0) || !std::strchr("AaBbDd", *only))) {
        std::fprintf(stderr, "QX_VQE_ONLY must be A, B or D (got '%s')\n", only);
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
    return 0;
}
