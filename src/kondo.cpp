// src/kondo.cpp  —  M48: Kondo Effect / Single-Impurity Anderson Model (SIAM)
//
// H = eps_d*(n_up + n_dn) + U*n_up*n_dn
//   + sum_k eps_k*(c†_{k↑}c_{k↑} + c†_{k↓}c_{k↓})
//   + V * sum_{k,σ} (d†_σ c_{kσ} + c†_{kσ} d_σ)
//
// Geometry: "star" — impurity (site 0) coupled to N_bath bath levels (sites 1..N_bath).
// Bath energies: eps_k = (k - (N_bath-1)/2) * dW,  dW = bandwidth/(N_bath-1).
//
// Diagonalisation: exact full Jacobi in the half-filled sector
//   N_up = N_dn = N_sites/2,  N_sites = N_bath+1 (N_bath must be odd for N_sites even).
//   Default N_bath=5 → N_sites=6, dim = C(6,3)^2 = 400.
//
// Observables (from ground-state eigenvector):
//   n_imp      — impurity occupation ⟨n_↑+n_↓⟩_imp
//   m2_imp     — local moment ⟨S_z^{imp}²⟩ = ⟨(n_↑-n_↓)²/4⟩  (0=singlet, 1/4=doublet)
//   E_gap      — excitation gap E_1 - E_0  (proxy for Kondo temperature)
//   spin_corr  — ⟨S_z^{imp}·S_z^k⟩ for each bath site (Kondo cloud screening)
//   T_K        — analytical estimate √(ΓU/2) exp(-π|ε_d|(|ε_d|+U)/(ΓU))
//
// Scenarios: "kondo" (V sweep) | "occupancy" (ε_d sweep) | "both"
//
// Kondo temperature analytic estimate (Haldane, 1978):
//   Γ = π * ρ_0 * V^2  ≈  N_bath * V^2 / bandwidth  (flat DOS approximation)
//   T_K = sqrt(Γ*U/2) * exp(-π * |ε_d| * (|ε_d|+U) / (Γ*U))
//   Valid in Kondo regime: -U < ε_d < 0,  Γ << U.

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
#include <chrono>
#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES  // MSVC: expose M_PI in <cmath>
#endif
#include <cmath>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace qx {

// ── Portable bit-count helper ────────────────────────────────────────────────
static int popcount_bits(unsigned v) {
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    return static_cast<int>(((v + (v >> 4)) & 0x0f0f0f0fu) * 0x01010101u >> 24);
}

// ── Fock-space utilities ─────────────────────────────────────────────────────

// Enumerate all n_occ-particle states in n_sites-site Hilbert space.
static std::vector<int> enum_sector(int n_sites, int n_occ) {
    std::vector<int> states;
    for (int s = 0; s < (1 << n_sites); ++s)
        if (popcount_bits(static_cast<unsigned>(s)) == n_occ)
            states.push_back(s);
    return states;
}

// Lookup table: occupation integer → index in sector (−1 if absent).
static std::vector<int> make_lut(const std::vector<int>& states, int n_sites) {
    std::vector<int> lut(1 << n_sites, -1);
    for (int i = 0; i < static_cast<int>(states.size()); ++i)
        lut[states[i]] = i;
    return lut;
}

// ── Hamiltonian construction ─────────────────────────────────────────────────

// Build SIAM Hamiltonian (real symmetric) in the (n_up, n_dn) sector.
// Returns flattened dim×dim row-major matrix (dim = |up_states|*|dn_states|).
static std::vector<double> build_siam(
    const std::vector<int>& up_states,
    const std::vector<int>& dn_states,
    const std::vector<int>& up_lut,
    const std::vector<int>& dn_lut,
    int n_bath,
    double eps_d, double U, double V, double bandwidth)
{
    const int du  = static_cast<int>(up_states.size());
    const int dd  = static_cast<int>(dn_states.size());
    const int dim = du * dd;
    std::vector<double> H(static_cast<size_t>(dim) * dim, 0.0);

    // Bath energies: eps_k for bath level k (k=0..n_bath-1, site index = k+1)
    const double dW = (n_bath > 1) ? bandwidth / static_cast<double>(n_bath - 1) : 0.0;
    auto bath_eps = [&](int k) -> double {
        return (static_cast<double>(k) - static_cast<double>(n_bath - 1) * 0.5) * dW;
    };

    for (int iu = 0; iu < du; ++iu) {
        for (int id = 0; id < dd; ++id) {
            const int row = iu * dd + id;
            const int au  = up_states[iu];
            const int ad  = dn_states[id];

            // ── Diagonal terms ────────────────────────────────────────────
            const int nu0 = (au >> 0) & 1;  // spin-up impurity occupation
            const int nd0 = (ad >> 0) & 1;  // spin-dn impurity occupation
            double diag = eps_d * (nu0 + nd0) + U * nu0 * nd0;
            for (int k = 0; k < n_bath; ++k) {
                const int s = k + 1;
                diag += bath_eps(k) * (((au >> s) & 1) + ((ad >> s) & 1));
            }
            H[static_cast<size_t>(row) * dim + row] = diag;

            // ── Hybridisation hoppings: V*(d†_σ c_{kσ} + h.c.) ──────────
            // For σ=up: hop between site 0 (impurity) and site s = k+1 (bath)
            for (int k = 0; k < n_bath; ++k) {
                const int s    = k + 1;
                // JW string for hopping between sites 0 and s (bits strictly between them)
                const int mask = (1 << s) - 2;  // bits 1..s-1

                // ── spin-up hopping ───────────────────────────────────────
                // c†_0 c_s: requires bit s set and bit 0 clear in au
                if (((au >> s) & 1) && !((au >> 0) & 1)) {
                    const int sgn_u = (popcount_bits(static_cast<unsigned>(au & mask)) & 1) ? -1 : 1;
                    const int au2   = (au ^ (1 << s)) | 1;  // clear s, set 0
                    const int iu2   = up_lut[au2];
                    if (iu2 >= 0) {
                        const int col = iu2 * dd + id;
                        H[static_cast<size_t>(row) * dim + col] += V * sgn_u;
                        H[static_cast<size_t>(col) * dim + row] += V * sgn_u;  // h.c.
                    }
                }
                // (c†_s c_0 is the h.c. already added above)

                // ── spin-dn hopping ───────────────────────────────────────
                // c†_0 c_s (down): requires bit s set and bit 0 clear in ad
                if (((ad >> s) & 1) && !((ad >> 0) & 1)) {
                    const int sgn_d = (popcount_bits(static_cast<unsigned>(ad & mask)) & 1) ? -1 : 1;
                    const int ad2   = (ad ^ (1 << s)) | 1;
                    const int id2   = dn_lut[ad2];
                    if (id2 >= 0) {
                        const int col = iu * dd + id2;
                        H[static_cast<size_t>(row) * dim + col] += V * sgn_d;
                        H[static_cast<size_t>(col) * dim + row] += V * sgn_d;
                    }
                }
            }
        }
    }
    return H;
}

// ── Jacobi full-diagonalisation (symmetric, with eigenvectors) ──────────────
static void jacobi_full(std::vector<double>& A, std::vector<double>& evecs, int n) {
    // Initialise eigenvector matrix to identity.
    evecs.assign(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) evecs[static_cast<size_t>(i) * n + i] = 1.0;

    const int max_sweeps = 100;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                off += A[static_cast<size_t>(i) * n + j] * A[static_cast<size_t>(i) * n + j];
        if (off < 1e-26) break;

        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = A[static_cast<size_t>(p) * n + q];
                if (std::abs(apq) < 1e-15) continue;
                const double app = A[static_cast<size_t>(p) * n + p];
                const double aqq = A[static_cast<size_t>(q) * n + q];
                const double theta = 0.5 * (aqq - app) / apq;
                const double t_val = (theta >= 0.0 ? 1.0 : -1.0)
                                   / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t_val * t_val + 1.0);
                const double s = t_val * c;

                // Update A
                A[static_cast<size_t>(p) * n + p] = app - t_val * apq;
                A[static_cast<size_t>(q) * n + q] = aqq + t_val * apq;
                A[static_cast<size_t>(p) * n + q] = 0.0;
                A[static_cast<size_t>(q) * n + p] = 0.0;
                for (int r = 0; r < n; ++r) {
                    if (r == p || r == q) continue;
                    const double arp = A[static_cast<size_t>(r) * n + p];
                    const double arq = A[static_cast<size_t>(r) * n + q];
                    A[static_cast<size_t>(r) * n + p] = c * arp - s * arq;
                    A[static_cast<size_t>(p) * n + r] = A[static_cast<size_t>(r) * n + p];
                    A[static_cast<size_t>(r) * n + q] = s * arp + c * arq;
                    A[static_cast<size_t>(q) * n + r] = A[static_cast<size_t>(r) * n + q];
                }
                // Update eigenvectors
                for (int r = 0; r < n; ++r) {
                    const double vrp = evecs[static_cast<size_t>(r) * n + p];
                    const double vrq = evecs[static_cast<size_t>(r) * n + q];
                    evecs[static_cast<size_t>(r) * n + p] = c * vrp - s * vrq;
                    evecs[static_cast<size_t>(r) * n + q] = s * vrp + c * vrq;
                }
            }
        }
    }
}

// Jacobi eigenvalues-only (faster — no eigenvector tracking).
static void jacobi_evals_only(std::vector<double>& A, int n) {
    const int max_sweeps = 100;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                off += A[static_cast<size_t>(i) * n + j] * A[static_cast<size_t>(i) * n + j];
        if (off < 1e-26) break;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = A[static_cast<size_t>(p) * n + q];
                if (std::abs(apq) < 1e-15) continue;
                const double app = A[static_cast<size_t>(p) * n + p];
                const double aqq = A[static_cast<size_t>(q) * n + q];
                const double theta = 0.5 * (aqq - app) / apq;
                const double t_val = (theta >= 0.0 ? 1.0 : -1.0)
                                   / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t_val * t_val + 1.0);
                const double s = t_val * c;
                A[static_cast<size_t>(p) * n + p] = app - t_val * apq;
                A[static_cast<size_t>(q) * n + q] = aqq + t_val * apq;
                A[static_cast<size_t>(p) * n + q] = 0.0;
                A[static_cast<size_t>(q) * n + p] = 0.0;
                for (int r = 0; r < n; ++r) {
                    if (r == p || r == q) continue;
                    const double arp = A[static_cast<size_t>(r) * n + p];
                    const double arq = A[static_cast<size_t>(r) * n + q];
                    A[static_cast<size_t>(r) * n + p] = c * arp - s * arq;
                    A[static_cast<size_t>(p) * n + r] = A[static_cast<size_t>(r) * n + p];
                    A[static_cast<size_t>(r) * n + q] = s * arp + c * arq;
                    A[static_cast<size_t>(q) * n + r] = A[static_cast<size_t>(r) * n + q];
                }
            }
        }
    }
}

// ── Observables from ground-state eigenvector ───────────────────────────────

// Compute impurity occupation, local moment, and bath spin-spin correlations.
static void compute_gs_obs(
    const std::vector<double>& gs_vec,  // ground-state eigenvector (column 0 of evecs)
    const std::vector<int>& up_states,
    const std::vector<int>& dn_states,
    int n_bath,
    double& n_imp_out,
    double& m2_imp_out,
    std::vector<double>& spin_corr_out)
{
    const int dd = static_cast<int>(dn_states.size());
    double n_imp = 0.0, m2_imp = 0.0;
    spin_corr_out.assign(n_bath, 0.0);

    for (int iu = 0; iu < static_cast<int>(up_states.size()); ++iu) {
        for (int id = 0; id < dd; ++id) {
            const int idx = iu * dd + id;
            const double c2 = gs_vec[idx] * gs_vec[idx];
            if (c2 < 1e-20) continue;
            const int au = up_states[iu], ad = dn_states[id];
            const double nu0 = static_cast<double>((au >> 0) & 1);
            const double nd0 = static_cast<double>((ad >> 0) & 1);
            n_imp  += c2 * (nu0 + nd0);
            const double sz_imp = (nu0 - nd0) * 0.5;
            m2_imp += c2 * sz_imp * sz_imp;
            for (int k = 0; k < n_bath; ++k) {
                const int s = k + 1;
                const double nu_s = static_cast<double>((au >> s) & 1);
                const double nd_s = static_cast<double>((ad >> s) & 1);
                const double sz_bath = (nu_s - nd_s) * 0.5;
                spin_corr_out[k] += c2 * sz_imp * sz_bath;
            }
        }
    }
    n_imp_out  = n_imp;
    m2_imp_out = m2_imp;
}

// ── Kondo temperature analytic estimate ────────────────────────────────────
// Haldane (1978): T_K = sqrt(Γ*U/2) * exp(-π*|ε_d|*(|ε_d|+U)/(Γ*U))
// Valid in Kondo regime: -U < ε_d < 0, Γ << U.
static double kondo_temp(double eps_d, double U, double V,
                          double bandwidth, int n_bath)
{
    if (V <= 0.0 || U <= 0.0) return 0.0;
    // Flat-DOS approximation: ρ_0 ≈ N_bath / bandwidth
    const double rho0 = (bandwidth > 0.0) ? static_cast<double>(n_bath) / bandwidth : 0.0;
    const double Gamma = M_PI * rho0 * V * V;
    if (Gamma < 1e-14 || Gamma * U < 1e-14) return 0.0;
    const double ed_abs = std::abs(eps_d);
    const double exponent = -M_PI * ed_abs * (ed_abs + U) / (Gamma * U);
    if (exponent < -100.0) return 0.0;  // underflow guard
    return std::sqrt(Gamma * U * 0.5) * std::exp(exponent);
}

// ── run_kondo ────────────────────────────────────────────────────────────────

KondoResult run_kondo(const KondoOptions& opts) {
    // Validation
    if (opts.N_bath < 1 || opts.N_bath % 2 == 0)
        throw std::runtime_error("Kondo: N_bath must be a positive odd integer (for even N_sites = N_bath+1)");
    if (opts.N_bath > 7)
        throw std::runtime_error("Kondo: N_bath > 7 makes dim > C(8,4)^2 = 4900 "
                                 "and the dense dim*dim matrices would exceed available memory; use N_bath <= 7");
    if (opts.U < 0.0)
        throw std::runtime_error("Kondo: U must be >= 0");
    if (opts.bandwidth <= 0.0)
        throw std::runtime_error("Kondo: bandwidth must be positive");
    if (opts.n_V < 1)
        throw std::runtime_error("Kondo: n_V must be >= 1");
    if (opts.n_eps < 1)
        throw std::runtime_error("Kondo: n_eps must be >= 1");
    const std::string& sc = opts.scenario;
    if (sc != "kondo" && sc != "occupancy" && sc != "both")
        throw std::runtime_error(
            "Kondo: unknown scenario \"" + sc + "\"; valid: kondo, occupancy, both");

    auto t0 = std::chrono::steady_clock::now();

    const int n_bath  = opts.N_bath;
    const int n_sites = n_bath + 1;
    const int n_half  = n_sites / 2;  // = (N_bath+1)/2, integer since N_sites is even

    // Enumerate half-filling sector
    const auto up_states = enum_sector(n_sites, n_half);
    const auto dn_states = enum_sector(n_sites, n_half);
    const auto up_lut    = make_lut(up_states, n_sites);
    const auto dn_lut    = make_lut(dn_states, n_sites);
    const int dim = static_cast<int>(up_states.size() * dn_states.size());

    KondoResult res;
    res.scenario  = sc;
    res.N_bath    = n_bath;
    res.eps_d     = opts.eps_d;
    res.U         = opts.U;
    res.V         = opts.V;
    res.bandwidth = opts.bandwidth;
    res.dim       = dim;
    res.TK_analytic = kondo_temp(opts.eps_d, opts.U, opts.V, opts.bandwidth, n_bath);

    // ── Fixed-parameter ground state ─────────────────────────────────────────
    {
        auto H   = build_siam(up_states, dn_states, up_lut, dn_lut,
                              n_bath, opts.eps_d, opts.U, opts.V, opts.bandwidth);
        std::vector<double> evecs;
        jacobi_full(H, evecs, dim);

        // Extract eigenvalues (diagonal of H after Jacobi)
        std::vector<double> evals(dim);
        for (int i = 0; i < dim; ++i)
            evals[i] = H[static_cast<size_t>(i) * dim + i];

        // Sort by eigenvalue to get ground state
        std::vector<int> idx(dim);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b){ return evals[a] < evals[b]; });

        res.E_ground = evals[idx[0]];
        res.E_gap    = (dim > 1) ? (evals[idx[1]] - evals[idx[0]]) : 0.0;

        // Ground state eigenvector (column idx[0] of evecs, stored as rows)
        std::vector<double> gs_vec(dim);
        for (int r = 0; r < dim; ++r)
            gs_vec[r] = evecs[static_cast<size_t>(r) * dim + idx[0]];

        compute_gs_obs(gs_vec, up_states, dn_states, n_bath,
                       res.n_imp, res.m2_imp, res.spin_corr);
    }

    // ── Kondo sweep: V from 0 to V_max ───────────────────────────────────────
    if (sc == "kondo" || sc == "both") {
        const int nV = opts.n_V;
        res.V_vals.resize(nV);
        res.gap_V.resize(nV);
        res.m2_V.resize(nV);

        for (int vi = 0; vi < nV; ++vi) {
            const double Vv = (nV > 1) ? opts.V_max * vi / static_cast<double>(nV - 1)
                                        : opts.V_max;
            res.V_vals[vi] = Vv;

            auto H = build_siam(up_states, dn_states, up_lut, dn_lut,
                                n_bath, opts.eps_d, opts.U, Vv, opts.bandwidth);
            std::vector<double> evecs;
            jacobi_full(H, evecs, dim);

            std::vector<double> evals(dim);
            for (int i = 0; i < dim; ++i)
                evals[i] = H[static_cast<size_t>(i) * dim + i];
            std::vector<int> idx(dim);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b){ return evals[a] < evals[b]; });

            res.gap_V[vi] = (dim > 1) ? (evals[idx[1]] - evals[idx[0]]) : 0.0;

            std::vector<double> gs_vec(dim);
            for (int r = 0; r < dim; ++r)
                gs_vec[r] = evecs[static_cast<size_t>(r) * dim + idx[0]];

            double n_imp_v = 0.0, m2_imp_v = 0.0;
            std::vector<double> sc_dummy;
            compute_gs_obs(gs_vec, up_states, dn_states, n_bath,
                           n_imp_v, m2_imp_v, sc_dummy);
            res.m2_V[vi] = m2_imp_v;
        }
    }

    // ── Occupancy sweep: ε_d from eps_min to eps_max ──────────────────────────
    if (sc == "occupancy" || sc == "both") {
        const int ne = opts.n_eps;
        res.eps_vals.resize(ne);
        res.n_d_eps.resize(ne);
        res.m2_eps.resize(ne);

        for (int ei = 0; ei < ne; ++ei) {
            const double ed = opts.eps_min + (opts.eps_max - opts.eps_min)
                              * static_cast<double>(ei) / static_cast<double>(ne > 1 ? ne - 1 : 1);
            res.eps_vals[ei] = ed;

            auto H = build_siam(up_states, dn_states, up_lut, dn_lut,
                                n_bath, ed, opts.U, opts.V, opts.bandwidth);
            std::vector<double> evecs;
            jacobi_full(H, evecs, dim);

            std::vector<double> evals(dim);
            for (int i = 0; i < dim; ++i)
                evals[i] = H[static_cast<size_t>(i) * dim + i];
            std::vector<int> idx(dim);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b){ return evals[a] < evals[b]; });

            std::vector<double> gs_vec(dim);
            for (int r = 0; r < dim; ++r)
                gs_vec[r] = evecs[static_cast<size_t>(r) * dim + idx[0]];

            double n_imp_e = 0.0, m2_imp_e = 0.0;
            std::vector<double> sc_dummy;
            compute_gs_obs(gs_vec, up_states, dn_states, n_bath,
                           n_imp_e, m2_imp_e, sc_dummy);
            res.n_d_eps[ei] = n_imp_e;
            res.m2_eps[ei]  = m2_imp_e;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.ok      = true;
    res.message = "ok";
    return res;
}

}  // namespace qx
