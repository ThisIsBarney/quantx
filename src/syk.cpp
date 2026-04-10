// src/syk.cpp  —  M47: Sachdev-Ye-Kitaev (SYK) Model
//
// N Majorana fermions with all-to-all random 4-body interactions:
//   H = Σ_{i<j<k<l} J_{ijkl} γ_i γ_j γ_k γ_l
//   J_{ijkl} ~ N(0,  σ²),   σ = J * sqrt(6) / N^{3/2}
//
// Majorana algebra: {γ_i, γ_j} = 2δ_{ij},  γ_i† = γ_i.
//
// Representation via M = N/2 complex fermion modes (Jordan-Wigner):
//   γ_{2m}   = c_m + c_m†               (m = 0,...,M-1)
//   γ_{2m+1} = i(c_m† − c_m)
//
// Matrix elements in the 2^M-dimensional Fock basis |n_0 … n_{M-1}⟩:
//   For γ_{2m}  :  ⟨β|γ_{2m}|α⟩   = p_m(α) · δ_{β, α⊕(1≪m)}
//   For γ_{2m+1}:  ⟨β|γ_{2m+1}|α⟩ = i·(1−2α_m)·p_m(α) · δ_{β, α⊕(1≪m)}
// where p_m(α) = (−1)^{popcount(α & ((1≪m)−1))}  (Jordan-Wigner string).
//
// The 4-body product γ_i γ_j γ_k γ_l maps each column α to exactly one
// other column β = α ⊕ bit_i ⊕ bit_j ⊕ bit_k ⊕ bit_l with a complex phase,
// so H is built in O(C(N,4) · dim) time — no dense matrix multiplications.
//
// Diagonalisation: real-doubling embedding H→[[Re,-Im],[Im,Re]] (2·dim × 2·dim
// real symmetric), then Jacobi without eigenvector tracking for eigenvalues.
// Cost scales as O(dim^2 · sweeps).  N ≤ 16 (dim ≤ 256) is enforced.
//
// Observables:
//   E₀/N   — ground-state energy per Majorana (approaches −0.2J for large N)
//   ⟨r⟩    — mean level-spacing ratio; GUE → 0.60, Poisson → 0.39
//   K(t)   — spectral form factor disorder-averaged over n_samples realisations
//
// Scenarios: "spectrum" | "sff" | "both"

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace qx {

using cx = std::complex<double>;

// ── Jordan-Wigner phase helper ────────────────────────────────────────────────
// Apply Majorana γ_j to Fock state alpha; return {phase, new_state}.
// j = Majorana index (0-indexed), M = number of complex modes.
static std::pair<cx, int> apply_gamma(int j, int alpha, int /*M*/) {
    const int m    = j / 2;
    const bool odd = (j & 1);
    // Jordan-Wigner string: (-1)^popcount(alpha & mask_m)
    const int mask_m = (1 << m) - 1;
    const double sign = (__builtin_popcount(alpha & mask_m) & 1) ? -1.0 : 1.0;
    const int new_state = alpha ^ (1 << m);
    cx phase;
    if (!odd) {
        // γ_{2m} = c_m + c_m†: real phase = sign
        phase = cx{sign, 0.0};
    } else {
        // γ_{2m+1} = i(c_m† − c_m): phase = i*(1−2*alpha_m)*sign
        const double fac = (1 - 2*((alpha >> m) & 1)) * sign;
        phase = cx{0.0, fac};
    }
    return {phase, new_state};
}

// ── Build SYK Hamiltonian matrix ──────────────────────────────────────────────
// Returns dim×dim complex Hermitian H (row-major, H[row*dim+col]).
// J_{ijkl} ~ N(0, 6*J^2/N^3).
static std::vector<cx> build_syk_h(int N, int M, int dim, double J_scale,
                                    std::mt19937& rng) {
    const double sigma = J_scale * std::sqrt(6.0) / std::pow(static_cast<double>(N), 1.5);
    std::normal_distribution<double> nd(0.0, sigma);

    std::vector<cx> H(dim * dim, cx{0.0, 0.0});

    // Iterate over all i<j<k<l (C(N,4) terms)
    for (int i = 0; i < N-3; ++i)
    for (int j = i+1; j < N-2; ++j)
    for (int k = j+1; k < N-1; ++k)
    for (int l = k+1; l < N;   ++l) {
        const double Jijkl = nd(rng);
        // Apply γ_i γ_j γ_k γ_l to each column α
        // Note: applied right-to-left (γ_l first, then γ_k, γ_j, γ_i)
        for (int alpha = 0; alpha < dim; ++alpha) {
            auto [p4, s4] = apply_gamma(l, alpha, M);
            auto [p3, s3] = apply_gamma(k, s4, M);
            auto [p2, s2] = apply_gamma(j, s3, M);
            auto [p1, s1] = apply_gamma(i, s2, M);
            const cx phase = p1 * p2 * p3 * p4;
            H[s1 * dim + alpha] += Jijkl * phase;
        }
    }
    return H;
}

// ── Jacobi eigenvalues (real symmetric, no eigenvectors) ─────────────────────
static std::vector<double> jacobi_evals_only(std::vector<double> A, int n) {
    const int max_sweep = 80 * n * n;
    for (int iter = 0; iter < max_sweep; ++iter) {
        int p = 0, q = 1; double amax = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i+1; j < n; ++j)
                if (std::abs(A[i*n+j]) > amax) { amax = std::abs(A[i*n+j]); p=i; q=j; }
        if (amax < 1e-13) break;
        const double app = A[p*n+p], aqq = A[q*n+q], apq = A[p*n+q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t   = (tau >= 0)
            ? 1.0 / (tau + std::sqrt(1.0 + tau*tau))
            : 1.0 / (tau - std::sqrt(1.0 + tau*tau));
        const double c = 1.0 / std::sqrt(1.0 + t*t), s = t*c;
        A[p*n+p] = app - t*apq;
        A[q*n+q] = aqq + t*apq;
        A[p*n+q] = A[q*n+p] = 0.0;
        for (int r = 0; r < n; ++r) {
            if (r==p||r==q) continue;
            const double arp = A[r*n+p], arq = A[r*n+q];
            A[r*n+p] = A[p*n+r] = c*arp - s*arq;
            A[r*n+q] = A[q*n+r] = s*arp + c*arq;
        }
    }
    std::vector<double> ev(n);
    for (int i = 0; i < n; ++i) ev[i] = A[i*n+i];
    return ev;
}

// ── Eigenvalues of complex Hermitian matrix via real-doubling ─────────────────
// H = A + iB  →  M = [[A,-B],[B,A]]  (2dim × 2dim real symmetric)
// Eigenvalues of M come in pairs (each e-value of H appears twice).
static std::vector<double> herm_evals(const std::vector<cx>& H, int dim) {
    const int n2 = 2 * dim;
    std::vector<double> M(n2 * n2, 0.0);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) {
            const double a = H[i*dim+j].real(), b = H[i*dim+j].imag();
            M[i*n2+j]              =  a;
            M[i*n2+(j+dim)]        = -b;
            M[(i+dim)*n2+j]        =  b;
            M[(i+dim)*n2+(j+dim)]  =  a;
        }
    std::vector<double> ev2 = jacobi_evals_only(std::move(M), n2);
    std::sort(ev2.begin(), ev2.end());
    // Take every other eigenvalue (they come in degenerate pairs)
    std::vector<double> ev(dim);
    for (int i = 0; i < dim; ++i) ev[i] = ev2[2*i];
    return ev;
}

// ── Level-spacing ratio r ─────────────────────────────────────────────────────
static std::vector<double> r_statistic(const std::vector<double>& ev) {
    const int n = static_cast<int>(ev.size());
    std::vector<double> r;
    r.reserve(n - 2);
    for (int i = 1; i < n-1; ++i) {
        const double d1 = ev[i]   - ev[i-1];
        const double d2 = ev[i+1] - ev[i];
        if (d1 > 1e-14 && d2 > 1e-14) {
            const double mn = std::min(d1, d2), mx = std::max(d1, d2);
            r.push_back(mn / mx);
        }
    }
    return r;
}

// ── Spectral form factor Z(β,t) = Σ_m exp(−β E_m) exp(i t E_m) ───────────────
static double compute_K_one(const std::vector<double>& ev, double beta, double t) {
    double re_z = 0.0, im_z = 0.0, z_beta = 0.0;
    for (double e : ev) {
        const double w = std::exp(-beta * e);
        re_z   += w * std::cos(t * e);
        im_z   += w * std::sin(t * e);
        z_beta += w;
    }
    const double K = (re_z*re_z + im_z*im_z) / (z_beta * z_beta);
    return K;
}

// ── Compute spectrum (single realization) ─────────────────────────────────────
static void compute_spectrum(const SYKOptions& opts, SYKResult& res) {
    const int M   = opts.N / 2;
    const int dim = 1 << M;
    std::mt19937 rng(static_cast<unsigned>(opts.seed));
    const auto H = build_syk_h(opts.N, M, dim, opts.J, rng);
    auto ev = herm_evals(H, dim);
    std::sort(ev.begin(), ev.end());

    res.eigenvalues = ev;
    res.E0_per_N    = ev.front() / opts.N;
    res.r_vals      = r_statistic(ev);
    res.r_mean      = 0.0;
    if (!res.r_vals.empty()) {
        for (double v : res.r_vals) res.r_mean += v;
        res.r_mean /= static_cast<double>(res.r_vals.size());
    }
}

// ── Compute disorder-averaged SFF ─────────────────────────────────────────────
static void compute_sff(const SYKOptions& opts, SYKResult& res) {
    const int M      = opts.N / 2;
    const int dim    = 1 << M;
    const int ns     = opts.n_samples;
    const int nt     = opts.n_t;
    const double tmax = opts.t_max;

    res.t_sff.resize(nt);
    res.K_sff.resize(nt, 0.0);
    for (int ti = 0; ti < nt; ++ti)
        res.t_sff[ti] = tmax * (ti + 1) / nt;

    // Use different seed offset per sample to ensure independent realisations
    for (int s = 0; s < ns; ++s) {
        std::mt19937 rng(static_cast<unsigned>(opts.seed + s * 1000));
        const auto H = build_syk_h(opts.N, M, dim, opts.J, rng);
        auto ev = herm_evals(H, dim);
        std::sort(ev.begin(), ev.end());

        for (int ti = 0; ti < nt; ++ti)
            res.K_sff[ti] += compute_K_one(ev, opts.beta, res.t_sff[ti]);
    }
    for (double& k : res.K_sff) k /= ns;
}

// ── Dispatcher ───────────────────────────────────────────────────────────────
SYKResult run_syk(const SYKOptions& opts) {
    if (opts.N < 4 || opts.N % 2 != 0)
        throw std::runtime_error("SYK: N must be an even integer >= 4");
    if (opts.N > 16)
        throw std::runtime_error("SYK: N > 16 is not supported (dim > 256 makes Jacobi too slow)");
    if (opts.J <= 0.0)
        throw std::runtime_error("SYK: J must be positive");
    if (opts.n_samples < 1)
        throw std::runtime_error("SYK: n_samples must be >= 1");

    const std::string& sc = opts.scenario;
    if (sc != "spectrum" && sc != "sff" && sc != "both")
        throw std::runtime_error(
            "SYK: unknown scenario \"" + sc +
            "\"; valid values: spectrum, sff, both");

    auto t0 = std::chrono::steady_clock::now();

    SYKResult res;
    res.scenario = sc;
    res.N   = opts.N;
    res.dim = 1 << (opts.N / 2);
    res.J   = opts.J;
    res.beta = opts.beta;

    if (sc == "spectrum" || sc == "both") compute_spectrum(opts, res);
    if (sc == "sff"      || sc == "both") compute_sff(opts, res);

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.message = "N=" + std::to_string(opts.N) +
                  " dim=" + std::to_string(res.dim) +
                  " J=" + std::to_string(opts.J) +
                  " r_mean=" + std::to_string(res.r_mean);
    res.ok = true;
    return res;
}

}  // namespace qx
