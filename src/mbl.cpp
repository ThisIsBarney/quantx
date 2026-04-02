// src/mbl.cpp — M32: Many-Body Localization (disordered XXZ Heisenberg chain)
//
// H = J * sum_i [ 1/2 * (S+_i S-_{i+1} + S-_i S+_{i+1}) + Delta * Sz_i Sz_{i+1} ]
//   + sum_i h_i Sz_i
//
// h_i ~ Uniform[-W, W]  (disorder strength W)
// Clean limit W=0: Heisenberg chain (integrable, ETH-violating but not MBL)
// MBL transition at W_c ~3.7 for J=Delta=1 (XXX Heisenberg, open BC, N~14)
//
// Diagnostics:
//   Level-spacing ratio <r>  => GOE ~0.530 (ergodic), Poisson ~0.386 (MBL)
//   Half-chain entanglement S_EE => volume law (ergodic) vs area law (MBL)
//   IPR = sum|psi_i|^4        => delocalized (IPR~1/dim) vs localized (IPR~1)
//
// Implementation:
//   Full exact diagonalization in the Sz=0 sector (dim = C(N, N/2))
//   Householder tridiagonalization + Wilkinson QR (with eigenvectors for single)
//   S_EE via reduced density matrix rho_A = C*C^T at bipartition N/2
//   Open boundary conditions; N <= 14 recommended (dim <= 3432)

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace qx {
namespace {

using MatR = std::vector<double>;

// ─────────────────────────────────────────────────────────────────────────────
// A.  Sz=0 basis enumeration (Gosper's hack)
//     bit i = 0 => up (Sz = +1/2),  bit i = 1 => down (Sz = -1/2)
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<int> sz0_basis(int N) {
    int n_down = N / 2;
    std::vector<int> basis;
    basis.reserve(1024);
    int s = (1 << n_down) - 1;
    int lim = 1 << N;
    while (s < lim) {
        basis.push_back(s);
        int c = s & (-s);
        int r = s + c;
        s = (((r ^ s) >> 2) / c) | r;
    }
    return basis;
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  Build dense MBL Hamiltonian in Sz=0 sector (row-major, real symmetric)
// ─────────────────────────────────────────────────────────────────────────────

static MatR build_mbl_ham(int N, double J, double delta,
                           const std::vector<double>& h,
                           const std::vector<int>&  basis,
                           std::vector<int>&        idx_map) {
    int dim = static_cast<int>(basis.size());
    idx_map.assign(1 << N, -1);
    for (int i = 0; i < dim; ++i) idx_map[basis[i]] = i;

    MatR H(static_cast<std::size_t>(dim) * dim, 0.0);

    for (int i = 0; i < dim; ++i) {
        int s = basis[i];
        // Diagonal: Delta*J * sum Sz_i Sz_{i+1} + sum h_i Sz_i
        double diag = 0.0;
        for (int site = 0; site < N - 1; ++site) {
            int bi   = (s >> site    ) & 1;
            int bi1  = (s >> (site+1)) & 1;
            // Sz_i Sz_{i+1} = (1-2*bi)/2 * (1-2*bi1)/2
            diag += J * delta * (1 - 2*bi) * (1 - 2*bi1) * 0.25;
        }
        for (int site = 0; site < N; ++site) {
            int bi = (s >> site) & 1;
            diag += h[site] * (1 - 2*bi) * 0.5;  // h_i * Sz_i
        }
        H[i * dim + i] = diag;

        // Off-diagonal: 1/2 * J * (S+_i S-_{i+1} + S-_i S+_{i+1}) for each bond
        for (int site = 0; site < N - 1; ++site) {
            int bi   = (s >> site    ) & 1;
            int bi1  = (s >> (site+1)) & 1;
            if (bi == bi1) continue;  // same spin => no flip
            int mask = (1 << site) | (1 << (site + 1));
            int sp   = s ^ mask;
            int j    = idx_map[sp];
            if (j > i) {
                double elem = J * 0.5;
                H[i * dim + j] = elem;
                H[j * dim + i] = elem;
            }
        }
    }
    return H;
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  Householder tridiagonalization (real symmetric, optional eigvec)
// ─────────────────────────────────────────────────────────────────────────────

static void householder_tridiag(const MatR& A, std::vector<double>& alpha,
                                 std::vector<double>& beta, int n,
                                 MatR* V_out = nullptr) {
    alpha.resize(n); beta.resize(n > 1 ? n-1 : 0);
    MatR Q = A;
    MatR V;
    if (V_out) {
        V.assign(static_cast<std::size_t>(n)*n, 0.0);
        for (int i = 0; i < n; ++i) V[i*n+i] = 1.0;
    }
    for (int k = 0; k < n - 2; ++k) {
        double norm2 = 0.0;
        for (int i = k+1; i < n; ++i) norm2 += Q[i*n+k]*Q[i*n+k];
        double norm = std::sqrt(norm2);
        if (norm < 1e-15) continue;
        std::vector<double> v(n, 0.0);
        for (int i = k+1; i < n; ++i) v[i] = Q[i*n+k];
        v[k+1] += (v[k+1] >= 0.0 ? 1.0 : -1.0) * norm;
        double vn2 = 0.0;
        for (int i = k+1; i < n; ++i) vn2 += v[i]*v[i];
        if (vn2 < 1e-30) continue;
        std::vector<double> p(n, 0.0);
        for (int r = 0; r < n; ++r)
            for (int c = k+1; c < n; ++c) p[r] += Q[r*n+c]*v[c];
        for (int r = 0; r < n; ++r) p[r] /= (vn2/2.0);
        double vTp = 0.0;
        for (int i = k+1; i < n; ++i) vTp += v[i]*p[i];
        std::vector<double> w(n);
        for (int i = 0; i < n; ++i) w[i] = p[i] - (vTp/vn2)*v[i];
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c)
                Q[r*n+c] -= v[r]*w[c] + w[r]*v[c];
        if (V_out) {
            std::vector<double> Vv(n, 0.0);
            for (int row = 0; row < n; ++row)
                for (int j = k+1; j < n; ++j) Vv[row] += V[row*n+j]*v[j];
            for (int row = 0; row < n; ++row)
                for (int c2 = k+1; c2 < n; ++c2)
                    V[row*n+c2] -= (2.0/vn2)*Vv[row]*v[c2];
        }
    }
    for (int i = 0; i < n; ++i) alpha[i] = Q[i*n+i];
    for (int i = 0; i < n-1; ++i) beta[i] = Q[(i+1)*n+i];
    if (V_out) *V_out = std::move(V);
}

// ─────────────────────────────────────────────────────────────────────────────
// D.  Wilkinson QR — eigenvalues only (fast path for sweep)
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<double> tridiag_evals(std::vector<double> alpha,
                                          std::vector<double> beta) {
    int n = static_cast<int>(alpha.size());
    for (int end = n-1; end > 0; ) {
        for (int i = 0; i < end; ++i)
            if (std::abs(beta[i]) < 1e-14*(std::abs(alpha[i])+std::abs(alpha[i+1])))
                beta[i] = 0.0;
        int start = end;
        while (start > 0 && beta[start-1] != 0.0) --start;
        if (start == end) { --end; continue; }
        double d  = (alpha[end-1]-alpha[end])*0.5;
        double sg = (d >= 0.0) ? 1.0 : -1.0;
        double mu = alpha[end] - sg*beta[end-1]*beta[end-1]/
                    (std::abs(d)+std::sqrt(d*d+beta[end-1]*beta[end-1]));
        double x = alpha[start]-mu, z = beta[start];
        for (int i = start; i < end; ++i) {
            double rr = std::sqrt(x*x+z*z);
            double c = x/rr, s2 = z/rr;
            if (i > start) beta[i-1] = rr;
            double a0=alpha[i],a1=alpha[i+1],b=beta[i];
            alpha[i]  =c*c*a0+2*c*s2*b+s2*s2*a1;
            alpha[i+1]=s2*s2*a0-2*c*s2*b+c*c*a1;
            beta[i]   =c*s2*(a1-a0)+(c*c-s2*s2)*b;
            if (i < end-1) { x=beta[i]; z=-s2*beta[i+1]; beta[i+1]*=c; }
        }
    }
    std::sort(alpha.begin(), alpha.end());
    return alpha;
}

// ─────────────────────────────────────────────────────────────────────────────
// E.  Wilkinson QR — eigenvalues + eigenvectors (single scenario)
//     Q_out columns are eigenvectors of the tridiagonal T (sorted by eigenvalue)
// ─────────────────────────────────────────────────────────────────────────────

static void tridiag_eig_full(std::vector<double> alpha, std::vector<double> beta,
                              std::vector<double>& evals, MatR& Q_out) {
    int n = static_cast<int>(alpha.size());
    Q_out.assign(static_cast<std::size_t>(n)*n, 0.0);
    for (int i = 0; i < n; ++i) Q_out[i*n+i] = 1.0;

    for (int end = n-1; end > 0; ) {
        for (int i = 0; i < end; ++i)
            if (std::abs(beta[i]) < 1e-14*(std::abs(alpha[i])+std::abs(alpha[i+1])))
                beta[i] = 0.0;
        int start = end;
        while (start > 0 && beta[start-1] != 0.0) --start;
        if (start == end) { --end; continue; }
        double d  = (alpha[end-1]-alpha[end])*0.5;
        double sg = (d >= 0.0) ? 1.0 : -1.0;
        double mu = alpha[end] - sg*beta[end-1]*beta[end-1]/
                    (std::abs(d)+std::sqrt(d*d+beta[end-1]*beta[end-1]));
        double x = alpha[start]-mu, z = beta[start];
        for (int i = start; i < end; ++i) {
            double rr = std::sqrt(x*x+z*z);
            if (rr < 1e-15) break;
            double c = x/rr, s2 = z/rr;
            if (i > start) beta[i-1] = rr;
            double a0=alpha[i],a1=alpha[i+1],b=beta[i];
            alpha[i]  =c*c*a0+2*c*s2*b+s2*s2*a1;
            alpha[i+1]=s2*s2*a0-2*c*s2*b+c*c*a1;
            beta[i]   =c*s2*(a1-a0)+(c*c-s2*s2)*b;
            if (i < end-1) { x=beta[i]; z=-s2*beta[i+1]; beta[i+1]*=c; }
            for (int row = 0; row < n; ++row) {
                double qi  = Q_out[row*n+i];
                double qi1 = Q_out[row*n+i+1];
                Q_out[row*n+i]   =  c*qi + s2*qi1;
                Q_out[row*n+i+1] = -s2*qi + c*qi1;
            }
        }
    }
    std::vector<int> ord(n);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b){ return alpha[a] < alpha[b]; });
    evals.resize(n);
    MatR Qs(static_cast<std::size_t>(n)*n);
    for (int j = 0; j < n; ++j) {
        evals[j] = alpha[ord[j]];
        for (int row = 0; row < n; ++row)
            Qs[row*n+j] = Q_out[row*n+ord[j]];
    }
    Q_out = std::move(Qs);
}

// ─────────────────────────────────────────────────────────────────────────────
// F.  Half-chain entanglement entropy via reduced density matrix rho_A = C*C^T
//     Bipartition: A = sites 0..NA-1, B = sites NA..N-1
// ─────────────────────────────────────────────────────────────────────────────

static double entanglement_entropy(const std::vector<double>& psi,
                                   const std::vector<int>& basis,
                                   int N) {
    int NA    = N / 2;
    int dim_L = 1 << NA;
    int dim_R = 1 << (N - NA);

    std::vector<double> C(static_cast<std::size_t>(dim_L) * dim_R, 0.0);
    for (std::size_t i = 0; i < basis.size(); ++i) {
        int s  = basis[i];
        int sL = s & (dim_L - 1);
        int sR = s >> NA;
        C[sL * dim_R + sR] = psi[i];
    }

    // rho_A = C * C^T  (dim_L x dim_L)
    MatR rho(static_cast<std::size_t>(dim_L) * dim_L, 0.0);
    for (int a = 0; a < dim_L; ++a)
        for (int b = 0; b <= a; ++b) {
            double sum = 0.0;
            for (int c = 0; c < dim_R; ++c)
                sum += C[a * dim_R + c] * C[b * dim_R + c];
            rho[a * dim_L + b] = rho[b * dim_L + a] = sum;
        }

    std::vector<double> ralpha, rbeta;
    householder_tridiag(rho, ralpha, rbeta, dim_L);
    auto rvals = tridiag_evals(ralpha, rbeta);

    double S = 0.0;
    for (double lam : rvals)
        if (lam > 1e-15) S -= lam * std::log(lam);
    return S;
}

// ─────────────────────────────────────────────────────────────────────────────
// G.  Level-spacing ratio r_n = min(d_n, d_{n+1}) / max(d_n, d_{n+1})
// ─────────────────────────────────────────────────────────────────────────────

static double level_spacing_ratio(const std::vector<double>& evals) {
    int n = static_cast<int>(evals.size());
    if (n < 3) return 0.0;
    double sum = 0.0; int cnt = 0;
    for (int i = 0; i + 2 <= n - 1; ++i) {
        double d0 = evals[i+1] - evals[i];
        double d1 = evals[i+2] - evals[i+1];
        if (d0 < 1e-15 || d1 < 1e-15) continue;
        sum += std::min(d0, d1) / std::max(d0, d1);
        ++cnt;
    }
    return (cnt > 0) ? sum / cnt : 0.0;
}

static std::vector<double> level_spacing_ratios(const std::vector<double>& evals) {
    int n = static_cast<int>(evals.size());
    std::vector<double> rv;
    rv.reserve(n);
    for (int i = 0; i + 2 <= n - 1; ++i) {
        double d0 = evals[i+1] - evals[i];
        double d1 = evals[i+2] - evals[i+1];
        if (d0 < 1e-15 || d1 < 1e-15) continue;
        rv.push_back(std::min(d0,d1)/std::max(d0,d1));
    }
    return rv;
}

// ─────────────────────────────────────────────────────────────────────────────
// H.  IPR  sum|psi_i|^4  (expects a column of Q_total, length dim)
// ─────────────────────────────────────────────────────────────────────────────

static double ipr_of(const MatR& Q_total, int col, int dim) {
    double sum = 0.0;
    for (int row = 0; row < dim; ++row) {
        double x = Q_total[row * dim + col];
        sum += x * x * x * x;
    }
    return sum;
}

// ─────────────────────────────────────────────────────────────────────────────
// I.  Single disorder realization (full eigensolver)
// ─────────────────────────────────────────────────────────────────────────────

static void single_realization(int N, double J, double delta, double W,
                                unsigned int seed,
                                std::vector<double>& evals_out,
                                std::vector<double>& entropies_out,
                                std::vector<double>& iprs_out,
                                std::vector<double>& r_vals_out,
                                double& r_mean_out,
                                double& entropy_mean_out,
                                double& ipr_mean_out) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(-W, W);

    std::vector<double> h(N);
    for (auto& hi : h) hi = uni(rng);

    auto basis = sz0_basis(N);
    int dim = static_cast<int>(basis.size());

    std::vector<int> idx_map;
    MatR H = build_mbl_ham(N, J, delta, h, basis, idx_map);

    std::vector<double> alpha, beta;
    MatR V;
    householder_tridiag(H, alpha, beta, dim, &V);

    std::vector<double> evals;
    MatR Q_trid;
    tridiag_eig_full(alpha, beta, evals, Q_trid);

    // Eigenvectors of original H: V * Q_trid (dim x dim)
    MatR Q_total(static_cast<std::size_t>(dim) * dim, 0.0);
    for (int row = 0; row < dim; ++row)
        for (int col = 0; col < dim; ++col)
            for (int k = 0; k < dim; ++k)
                Q_total[row*dim+col] += V[row*dim+k] * Q_trid[k*dim+col];

    evals_out = evals;

    r_vals_out    = level_spacing_ratios(evals);
    r_mean_out    = level_spacing_ratio(evals);

    // S_EE and IPR for middle half of spectrum
    int lo = dim / 4, hi = 3 * dim / 4;
    entropies_out.clear(); iprs_out.clear();
    for (int col = lo; col < hi; ++col) {
        std::vector<double> psi(dim);
        for (int row = 0; row < dim; ++row) psi[row] = Q_total[row*dim+col];
        entropies_out.push_back(entanglement_entropy(psi, basis, N));
        iprs_out.push_back(ipr_of(Q_total, col, dim));
    }
    entropy_mean_out = 0.0;
    ipr_mean_out     = 0.0;
    if (!entropies_out.empty()) {
        for (double s : entropies_out) entropy_mean_out += s;
        entropy_mean_out /= static_cast<double>(entropies_out.size());
        for (double ip : iprs_out) ipr_mean_out += ip;
        ipr_mean_out /= static_cast<double>(iprs_out.size());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// J.  Sweep over W (eigenvalues + S_EE for small dim)
// ─────────────────────────────────────────────────────────────────────────────

static MBLSweepPoint sweep_point(int N, double J, double delta, double W,
                                  int n_samples, unsigned int base_seed) {
    auto basis = sz0_basis(N);
    int dim = static_cast<int>(basis.size());

    double sum_r = 0.0, sum_S = 0.0, sum_ipr = 0.0;

    for (int sample = 0; sample < n_samples; ++sample) {
        std::mt19937 rng(base_seed + static_cast<unsigned>(sample));
        std::uniform_real_distribution<double> uni(-W, W);
        std::vector<double> h(N);
        for (auto& hi : h) hi = uni(rng);

        std::vector<int> idx_map;
        MatR H = build_mbl_ham(N, J, delta, h, basis, idx_map);

        if (dim <= 500) {
            // Full diagonalization: get eigenvectors for S_EE
            MatR V2;
            std::vector<double> alpha2, beta2;
            householder_tridiag(H, alpha2, beta2, dim, &V2);
            std::vector<double> evals2;
            MatR Q_trid;
            tridiag_eig_full(alpha2, beta2, evals2, Q_trid);

            sum_r += level_spacing_ratio(evals2);

            MatR Q_total(static_cast<std::size_t>(dim)*dim, 0.0);
            for (int row = 0; row < dim; ++row)
                for (int col = 0; col < dim; ++col)
                    for (int k = 0; k < dim; ++k)
                        Q_total[row*dim+col] += V2[row*dim+k]*Q_trid[k*dim+col];

            int lo = dim/4, hi2 = 3*dim/4;
            double sS = 0.0, sIPR = 0.0; int cnt = 0;
            int step = std::max(1, (hi2-lo)/8);
            for (int col = lo; col < hi2; col += step) {
                std::vector<double> psi(dim);
                for (int row = 0; row < dim; ++row) psi[row] = Q_total[row*dim+col];
                sS   += entanglement_entropy(psi, basis, N);
                sIPR += ipr_of(Q_total, col, dim);
                ++cnt;
            }
            if (cnt > 0) { sum_S += sS/cnt; sum_ipr += sIPR/cnt; }
        } else {
            // Eigenvalues only (fast path for larger dim)
            std::vector<double> alpha, beta;
            householder_tridiag(H, alpha, beta, dim);
            auto evals = tridiag_evals(alpha, beta);
            sum_r += level_spacing_ratio(evals);
        }
    }

    MBLSweepPoint pt;
    pt.W            = W;
    pt.r_mean       = sum_r / n_samples;
    pt.entropy_mean = (dim <= 500) ? sum_S / n_samples : 0.0;
    pt.ipr_mean     = (dim <= 500) ? sum_ipr / n_samples : 0.0;
    return pt;
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// K.  Public entry point
// ─────────────────────────────────────────────────────────────────────────────

MBLResult run_mbl(const MBLOptions& opts) {
    auto t0 = std::chrono::steady_clock::now();

    if (opts.n_sites < 2 || opts.n_sites > 16 || opts.n_sites % 2 != 0)
        throw std::runtime_error("MBL: n_sites must be even and in [2, 16]");
    if (opts.J <= 0.0)
        throw std::runtime_error("MBL: J must be positive");
    if (opts.W < 0.0)
        throw std::runtime_error("MBL: W must be non-negative");
    if (opts.n_samples < 1)
        throw std::runtime_error("MBL: n_samples must be >= 1");

    MBLResult res;
    res.n_sites  = opts.n_sites;
    res.J        = opts.J;
    res.delta    = opts.delta;
    res.W        = opts.W;
    res.scenario = opts.scenario;

    if (opts.scenario == "single") {
        single_realization(opts.n_sites, opts.J, opts.delta, opts.W, opts.seed,
                           res.evals, res.entropies, res.iprs, res.r_vals,
                           res.r_mean, res.entropy_mean, res.ipr_mean);
    } else if (opts.scenario == "sweep") {
        if (opts.n_sweep < 2)
            throw std::runtime_error("MBL: n_sweep must be >= 2 for sweep scenario");
        res.sweep.reserve(opts.n_sweep);
        for (int k = 0; k < opts.n_sweep; ++k) {
            double Wk = opts.W_min + (opts.W_max - opts.W_min) * k / (opts.n_sweep - 1);
            res.sweep.push_back(sweep_point(opts.n_sites, opts.J, opts.delta, Wk,
                                            opts.n_samples, opts.seed + static_cast<unsigned>(k) * 1000u));
        }
    } else {
        throw std::runtime_error("MBL: unknown scenario '" + opts.scenario + "'");
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    res.ok = true;
    return res;
}

}  // namespace qx
