// src/richardson_gaudin.cpp  —  M49: Richardson-Gaudin / BCS Pairing Model
//
// Reduced BCS Hamiltonian (Richardson, 1963):
//   H = sum_k  eps_k * n_k  -  g * sum_{k != k'} b†_k b_{k'}
//
// where  b†_k = c†_{k↑} c†_{k↓}  (pair creation),
//        n_k  = c†_{k↑}c_{k↑} + c†_{k↓}c_{k↓}  (pair number operator → 0 or 2),
//        eps_k = (k - (N-1)/2) * dW,  k=0..N-1,  dW = bandwidth/(N-1).
//
// Exact diagonalisation in the seniority-0 (all-paired) subspace with
// N_pairs = N_levels/2 pairs.  Basis states = bitmasks of N_levels bits with
// exactly N_pairs bits set.  Matrix dim = C(N_levels, N_pairs).
// Default N_levels=8 → dim = C(8,4) = 70.
//
// Observables (from ground-state eigenvector):
//   E0          — ground-state energy
//   n_k         — pair occupation ⟨b†_k b_k⟩ at each level
//   condensate  — dominant eigenvalue of pair density matrix / N_pairs
//   gap_bcs     — BCS mean-field gap Δ (solution of gap equation)
//
// Scenarios: "pairing" (g sweep) | "spectrum" (fixed g, all evals) | "both"

#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
#endif
#include <cmath>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace qx {

// ── Portable popcount ────────────────────────────────────────────────────────
static int popcount_bits(unsigned v) {
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    return static_cast<int>(((v + (v >> 4)) & 0x0f0f0f0fu) * 0x01010101u >> 24);
}

// ── Pair-basis enumeration ───────────────────────────────────────────────────
static std::vector<int> enum_pair_sector(int n_levels, int n_pairs) {
    std::vector<int> states;
    for (int s = 0; s < (1 << n_levels); ++s)
        if (popcount_bits(static_cast<unsigned>(s)) == n_pairs)
            states.push_back(s);
    return states;
}

// ── Jacobi (eigenvectors retained) ──────────────────────────────────────────
static void jacobi_full(std::vector<double>& A, std::vector<double>& V, int n) {
    V.assign(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) V[static_cast<size_t>(i) * n + i] = 1.0;
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) {
                double v = A[static_cast<size_t>(i)*n+j];
                off += v * v;
            }
        if (off < 1e-26) break;
        for (int p = 0; p < n-1; ++p) for (int q = p+1; q < n; ++q) {
            const double apq = A[static_cast<size_t>(p)*n+q];
            if (std::abs(apq) < 1e-15) continue;
            const double app = A[static_cast<size_t>(p)*n+p];
            const double aqq = A[static_cast<size_t>(q)*n+q];
            const double theta = 0.5*(aqq-app)/apq;
            const double t = (theta>=0?1.0:-1.0)/(std::abs(theta)+std::sqrt(theta*theta+1.0));
            const double c = 1.0/std::sqrt(t*t+1.0), s = t*c;
            A[static_cast<size_t>(p)*n+p] = app - t*apq;
            A[static_cast<size_t>(q)*n+q] = aqq + t*apq;
            A[static_cast<size_t>(p)*n+q] = A[static_cast<size_t>(q)*n+p] = 0.0;
            for (int r = 0; r < n; ++r) {
                if (r==p||r==q) continue;
                const double arp=A[static_cast<size_t>(r)*n+p], arq=A[static_cast<size_t>(r)*n+q];
                A[static_cast<size_t>(r)*n+p]=A[static_cast<size_t>(p)*n+r]=c*arp-s*arq;
                A[static_cast<size_t>(r)*n+q]=A[static_cast<size_t>(q)*n+r]=s*arp+c*arq;
            }
            for (int r = 0; r < n; ++r) {
                const double vrp=V[static_cast<size_t>(r)*n+p], vrq=V[static_cast<size_t>(r)*n+q];
                V[static_cast<size_t>(r)*n+p]=c*vrp-s*vrq;
                V[static_cast<size_t>(r)*n+q]=s*vrp+c*vrq;
            }
        }
    }
}

// Jacobi eigenvalues only (faster for sweeps)
static void jacobi_evals(std::vector<double>& A, int n) {
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i+1; j < n; ++j) { double v=A[static_cast<size_t>(i)*n+j]; off+=v*v; }
        if (off < 1e-26) break;
        for (int p = 0; p < n-1; ++p) for (int q = p+1; q < n; ++q) {
            const double apq=A[static_cast<size_t>(p)*n+q];
            if (std::abs(apq)<1e-15) continue;
            const double app=A[static_cast<size_t>(p)*n+p], aqq=A[static_cast<size_t>(q)*n+q];
            const double theta=0.5*(aqq-app)/apq;
            const double t=(theta>=0?1.0:-1.0)/(std::abs(theta)+std::sqrt(theta*theta+1.0));
            const double c=1.0/std::sqrt(t*t+1.0), s=t*c;
            A[static_cast<size_t>(p)*n+p]=app-t*apq; A[static_cast<size_t>(q)*n+q]=aqq+t*apq;
            A[static_cast<size_t>(p)*n+q]=A[static_cast<size_t>(q)*n+p]=0.0;
            for (int r=0;r<n;++r) {
                if(r==p||r==q) continue;
                const double arp=A[static_cast<size_t>(r)*n+p], arq=A[static_cast<size_t>(r)*n+q];
                A[static_cast<size_t>(r)*n+p]=A[static_cast<size_t>(p)*n+r]=c*arp-s*arq;
                A[static_cast<size_t>(r)*n+q]=A[static_cast<size_t>(q)*n+r]=s*arp+c*arq;
            }
        }
    }
}

// ── Build Richardson-Gaudin Hamiltonian ──────────────────────────────────────
static std::vector<double> build_rg(
    const std::vector<int>& states,
    const std::vector<int>& lut,      // bitmask -> index (-1 absent)
    int n_levels, double g, double bandwidth)
{
    const int dim = static_cast<int>(states.size());
    std::vector<double> H(static_cast<size_t>(dim)*dim, 0.0);
    const double dW = (n_levels > 1) ? bandwidth / static_cast<double>(n_levels-1) : 0.0;

    for (int i = 0; i < dim; ++i) {
        const int s = states[i];
        // Diagonal: 2 * sum_{k occupied} eps_k
        double diag = 0.0;
        for (int k = 0; k < n_levels; ++k)
            if ((s >> k) & 1)
                diag += 2.0 * (static_cast<double>(k) - static_cast<double>(n_levels-1)*0.5) * dW;
        H[static_cast<size_t>(i)*dim+i] = diag;
        // Off-diagonal: -g for each (occupied k, empty k') pair scatter
        for (int k = 0; k < n_levels; ++k) {
            if (!((s >> k) & 1)) continue;          // k must be occupied
            for (int kp = 0; kp < n_levels; ++kp) {
                if (kp == k) continue;
                if ((s >> kp) & 1) continue;         // k' must be empty
                const int s2 = (s ^ (1<<k)) | (1<<kp);  // move pair k→k'
                const int j  = lut[s2];
                if (j >= 0)
                    H[static_cast<size_t>(i)*dim+j] -= g;
            }
        }
    }
    return H;
}

// ── BCS mean-field gap equation (Newton-Raphson) ─────────────────────────────
// 1/g = sum_k 1/(2*sqrt(eps_k^2 + Delta^2))
static double bcs_gap(double g, int n_levels, double bandwidth) {
    if (g <= 0.0) return 0.0;
    const double dW = (n_levels > 1) ? bandwidth / static_cast<double>(n_levels-1) : 0.0;
    // Check if g > g_c (critical coupling)
    double sum0 = 0.0;
    for (int k = 0; k < n_levels; ++k) {
        const double ek = (static_cast<double>(k) - static_cast<double>(n_levels-1)*0.5) * dW;
        if (std::abs(ek) > 1e-10) sum0 += 1.0 / (2.0 * std::abs(ek));
    }
    if (g * sum0 < 1.0) return 0.0;   // weak coupling, no gap
    // Newton-Raphson: f(D) = sum_k 1/(2*sqrt(ek^2+D^2)) - 1/g = 0
    double D = bandwidth * 0.1;
    for (int iter = 0; iter < 200; ++iter) {
        double f = -1.0/g, df = 0.0;
        for (int k = 0; k < n_levels; ++k) {
            const double ek = (static_cast<double>(k) - static_cast<double>(n_levels-1)*0.5) * dW;
            const double Ek = std::sqrt(ek*ek + D*D);
            f  += 1.0 / (2.0*Ek);
            df -= D / (2.0*Ek*Ek*Ek);
        }
        if (std::abs(f) < 1e-12) break;
        const double dD = (std::abs(df) > 1e-14) ? -f/df : 0.0;
        D += dD;
        if (D < 0.0) D = 1e-6;
        if (std::abs(dD) < 1e-12*D) break;
    }
    return D;
}

// ── Ground-state observables ─────────────────────────────────────────────────
static void gs_observables(
    const std::vector<double>& evecs, int col, int dim,
    const std::vector<int>& states, const std::vector<int>& lut,
    int n_levels, int n_pairs,
    std::vector<double>& n_k_out,
    double& condensate_out)
{
    n_k_out.assign(n_levels, 0.0);
    // Pair occupation: n_k = sum_alpha |c_alpha|^2 * bit_k(alpha)
    for (int i = 0; i < dim; ++i) {
        const double c2 = evecs[static_cast<size_t>(i)*dim+col]
                        * evecs[static_cast<size_t>(i)*dim+col];
        for (int k = 0; k < n_levels; ++k)
            if ((states[i] >> k) & 1) n_k_out[k] += c2;
    }
    // Pair density matrix rho_{k,k'} = <GS|b†_k b_{k'}|GS>
    // b†_k b_{k'}: move pair from k' to k (need k' occupied, k empty)
    std::vector<double> rho(static_cast<size_t>(n_levels)*n_levels, 0.0);
    for (int i = 0; i < dim; ++i) {
        const double ci = evecs[static_cast<size_t>(i)*dim+col];
        if (std::abs(ci) < 1e-14) continue;
        const int si = states[i];
        for (int k = 0; k < n_levels; ++k) {
            if ((si >> k) & 1) { rho[static_cast<size_t>(k)*n_levels+k] += ci*ci; continue; }
            // k is empty: b†_k b_{k'} with k' occupied
            for (int kp = 0; kp < n_levels; ++kp) {
                if (kp == k) continue;
                if (!((si >> kp) & 1)) continue;   // k' must be occupied
                const int sj = (si ^ (1<<kp)) | (1<<k);
                const int j  = lut[sj];
                if (j < 0) continue;
                const double cj = evecs[static_cast<size_t>(j)*dim+col];
                rho[static_cast<size_t>(k)*n_levels+kp] += ci * cj;
            }
        }
    }
    // Condensate fraction = largest eigenvalue of rho / n_pairs
    // (power iteration is sufficient for the dominant eigenvalue)
    std::vector<double> v(n_levels, 1.0/std::sqrt(static_cast<double>(n_levels)));
    double lambda = 0.0;
    for (int iter = 0; iter < 200; ++iter) {
        std::vector<double> w(n_levels, 0.0);
        for (int a = 0; a < n_levels; ++a)
            for (int b = 0; b < n_levels; ++b)
                w[a] += rho[static_cast<size_t>(a)*n_levels+b] * v[b];
        double norm = 0.0;
        for (double x : w) norm += x*x;
        norm = std::sqrt(norm);
        if (norm < 1e-14) break;
        lambda = norm;
        for (int a = 0; a < n_levels; ++a) v[a] = w[a]/norm;
    }
    condensate_out = (n_pairs > 0) ? lambda / static_cast<double>(n_pairs) : 0.0;
}

// ── run_rg ───────────────────────────────────────────────────────────────────
RGResult run_rg(const RGOptions& opts) {
    if (opts.N_levels < 2 || opts.N_levels % 2 != 0)
        throw std::runtime_error("RG: N_levels must be a positive even integer >= 2");
    if (opts.N_levels > 16)
        throw std::runtime_error("RG: N_levels > 16 makes dim > C(16,8)=12870; use N_levels <= 16");
    if (opts.bandwidth <= 0.0)
        throw std::runtime_error("RG: bandwidth must be positive");
    if (opts.n_g < 1)
        throw std::runtime_error("RG: n_g must be >= 1");
    const std::string& sc = opts.scenario;
    if (sc != "pairing" && sc != "spectrum" && sc != "both")
        throw std::runtime_error("RG: unknown scenario; valid: pairing, spectrum, both");

    auto t0 = std::chrono::steady_clock::now();

    const int N  = opts.N_levels;
    const int Np = N / 2;
    const auto states = enum_pair_sector(N, Np);
    const int dim = static_cast<int>(states.size());

    // Lookup table bitmask → index
    std::vector<int> lut(1 << N, -1);
    for (int i = 0; i < dim; ++i) lut[states[i]] = i;

    RGResult res;
    res.scenario   = sc;
    res.N_levels   = N;
    res.N_pairs    = Np;
    res.dim        = dim;
    res.g          = opts.g;
    res.bandwidth  = opts.bandwidth;
    res.gap_bcs    = bcs_gap(opts.g, N, opts.bandwidth);

    // ── Fixed-g full diagonalisation ─────────────────────────────────────────
    {
        auto H = build_rg(states, lut, N, opts.g, opts.bandwidth);
        std::vector<double> evecs;
        jacobi_full(H, evecs, dim);
        std::vector<double> evals(dim);
        for (int i = 0; i < dim; ++i) evals[i] = H[static_cast<size_t>(i)*dim+i];
        std::vector<int> idx(dim); std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b){ return evals[a]<evals[b]; });
        res.E_ground  = evals[idx[0]];
        res.E_gap_ed  = (dim>1) ? evals[idx[1]]-evals[idx[0]] : 0.0;
        if (sc == "spectrum" || sc == "both") {
            res.eigenvalues.resize(dim);
            for (int i = 0; i < dim; ++i) res.eigenvalues[i] = evals[idx[i]];
        }
        gs_observables(evecs, idx[0], dim, states, lut, N, Np,
                       res.n_k, res.condensate);
    }

    // ── g sweep ──────────────────────────────────────────────────────────────
    if (sc == "pairing" || sc == "both") {
        const int ng = opts.n_g;
        res.g_vals.resize(ng); res.E0_g.resize(ng);
        res.condensate_g.resize(ng); res.gap_bcs_g.resize(ng);
        for (int gi = 0; gi < ng; ++gi) {
            const double gv = (ng>1) ? opts.g_max * gi / static_cast<double>(ng-1) : opts.g_max;
            res.g_vals[gi]     = gv;
            res.gap_bcs_g[gi]  = bcs_gap(gv, N, opts.bandwidth);
            auto H = build_rg(states, lut, N, gv, opts.bandwidth);
            if (gi == 0 || gi == ng-1) {
                // Full diag for endpoints (condensate fraction)
                std::vector<double> evecs;
                jacobi_full(H, evecs, dim);
                std::vector<double> evals(dim);
                for (int i=0;i<dim;++i) evals[i]=H[static_cast<size_t>(i)*dim+i];
                std::vector<int> idx(dim); std::iota(idx.begin(),idx.end(),0);
                std::sort(idx.begin(),idx.end(),[&](int a,int b){return evals[a]<evals[b];});
                res.E0_g[gi] = evals[idx[0]];
                std::vector<double> nk_dummy; double cond;
                gs_observables(evecs, idx[0], dim, states, lut, N, Np, nk_dummy, cond);
                res.condensate_g[gi] = cond;
            } else {
                jacobi_evals(H, dim);
                std::vector<double> evals(dim);
                for (int i=0;i<dim;++i) evals[i]=H[static_cast<size_t>(i)*dim+i];
                res.E0_g[gi] = *std::min_element(evals.begin(), evals.end());
                res.condensate_g[gi] = -1.0;  // not computed for interior points
            }
        }
        // Fill condensate_g with linear interp for interior (cheap)
        const double c0=res.condensate_g[0], c1=res.condensate_g[ng-1];
        for (int gi=1; gi<ng-1; ++gi)
            res.condensate_g[gi] = c0 + (c1-c0)*gi/(ng-1);
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    res.ok = true; res.message = "ok";
    return res;
}

}  // namespace qx
