// src/spectral.cpp -- M33: Spectral Functions & Imaginary-Time Evolution
//
// Computes the dynamical structure factor S(k,omega) of the XXZ Heisenberg chain
// via exact diagonalization (Lehmann representation) and the imaginary-time
// projection e^{-tau H}|psi> that converges to the ground state.
//
// Physics:
//   H = J * sum_i [ Delta * Sz_i Sz_{i+1} + 1/2 * (S+_i S-_{i+1} + h.c.) ]
//       - h * sum_i Sz_i
//
// Dynamical structure factor (Lehmann representation):
//   S(k, omega) = sum_n |<n| S^z_k |GS>|^2 * eta/pi / [(omega-(E_n-E_0))^2 + eta^2]
//   where S^z_k = (1/sqrt(N)) * sum_j exp(i k j) * Sz_j
//
// For the isotropic Heisenberg chain (Delta=1, h=0, J>0):
//   The exact 2-spinon continuum is bounded by the des Cloizeaux-Pearson
//   lower edge  omega_lower(k) = (pi/2) |J sin k|
//   and the upper edge.  S(k,omega) shows the dominant magnon weight there.
//
// Imaginary-time evolution:
//   Starting from |psi(0)> = sum_n c_n |n>, tau-evolution gives
//   |psi(tau)> ~ exp(-tau H)|psi(0)> / norm = sum_n c_n exp(-tau E_n)|n> / norm
//   <E>(tau)  = sum_n |c_n|^2 E_n exp(-2 tau (E_n - E_0)) / Z(tau)
//   which converges to E_0 exponentially in tau (Wick rotation it -> tau).
//
// Scenarios:
//   spectral    -- S(k,omega) heatmap
//   imagtime    -- E(tau) and fidelity vs tau
//   both        -- both computations

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace qx {
namespace {

using MatR = std::vector<double>;

// -----------------------------------------------------------------------------
// A.  Sz=0 basis (same Gosper's-hack enumeration as mbl.cpp)
//     bit i = 0 -> up (Sz = +1/2), bit i = 1 -> down (Sz = -1/2)
// -----------------------------------------------------------------------------

static std::vector<int> sz0_basis(int N) {
    int n_down = N / 2;
    std::vector<int> basis;
    basis.reserve(512);
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

// -----------------------------------------------------------------------------
// B.  Build dense XXZ Hamiltonian in Sz=0 sector (open BC, real symmetric)
// -----------------------------------------------------------------------------

static MatR build_xxz(int N, double J, double delta, double h,
                       const std::vector<int>& basis,
                       std::vector<int>&       idx_map) {
    int dim = static_cast<int>(basis.size());
    idx_map.assign(1 << N, -1);
    for (int i = 0; i < dim; ++i) idx_map[basis[i]] = i;

    MatR H(static_cast<std::size_t>(dim) * dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        int s = basis[i];
        double diag = 0.0;
        for (int site = 0; site < N - 1; ++site) {
            int bi  = (s >> site    ) & 1;
            int bi1 = (s >> (site+1)) & 1;
            diag += J * delta * (1-2*bi) * (1-2*bi1) * 0.25;
        }
        for (int site = 0; site < N; ++site) {
            int bi = (s >> site) & 1;
            diag += -h * (1-2*bi) * 0.5;
        }
        H[i*dim+i] = diag;

        for (int site = 0; site < N - 1; ++site) {
            int bi  = (s >> site    ) & 1;
            int bi1 = (s >> (site+1)) & 1;
            if (bi == bi1) continue;
            int mask = (1 << site) | (1 << (site+1));
            int sp   = s ^ mask;
            int j    = idx_map[sp];
            if (j > i) {
                H[i*dim+j] = H[j*dim+i] = J * 0.5;
            }
        }
    }
    return H;
}

// -----------------------------------------------------------------------------
// C.  Householder tridiagonalization (real symmetric, optional eigvec)
// -----------------------------------------------------------------------------

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
    for (int k = 0; k < n-2; ++k) {
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

// -----------------------------------------------------------------------------
// D.  Wilkinson QR with eigenvector accumulation
// -----------------------------------------------------------------------------

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
            double r = std::sqrt(x*x+z*z);
            if (r < 1e-15) break;
            double c = x/r, s = z/r;
            if (i > start) beta[i-1] = r;
            double a0=alpha[i], a1=alpha[i+1], b=beta[i];
            alpha[i]  = c*c*a0 + 2*c*s*b + s*s*a1;
            alpha[i+1]= s*s*a0 - 2*c*s*b + c*c*a1;
            beta[i]   = c*s*(a1-a0) + (c*c-s*s)*b;
            if (i < end-1) { x=beta[i]; z=-s*beta[i+1]; beta[i+1]*=c; }
            for (int row = 0; row < n; ++row) {
                double qi  = Q_out[row*n+i];
                double qi1 = Q_out[row*n+i+1];
                Q_out[row*n+i]   =  c*qi + s*qi1;
                Q_out[row*n+i+1] = -s*qi + c*qi1;
            }
        }
    }
    // Sort by eigenvalue
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

// -----------------------------------------------------------------------------
// E.  Apply S^z_k to a state vector
//     S^z_k = (1/sqrt(N)) * sum_j exp(i k j) * Sz_j
//     Acting on a real state |psi> in the Sz=0 sector gives a *complex* vector,
//     but we only need Re and Im separately for the overlap |<n|S^z_k|0>|^2.
//
//     <n| S^z_k |0> = (1/sqrt(N)) sum_j exp(i k j) * <n| Sz_j |0>
//     where <n| Sz_j |0> = sum_s <n|s> * sz_j(s) * <s|0>
//                        = sum_s Q[s,n] * sz_j(s) * Q[s,0]
//
//     Result (complex): amp_k[n] = sum_s Q[s,n] * (1/sqrt(N) * sum_j exp(ikj)*sz_j(s)) * Q[s,0]
//                                = sum_s Q[s,n] * Vk[s] * Q[s,0]
//     where Vk[s] = (1/sqrt(N)) * sum_j exp(ikj) * sz_j(s)
//
//     Since sz_j(s) is real, Vk[s] = Vk_re[s] + i*Vk_im[s].
//     |amp_k[n]|^2 = (sum_s Vk_re[s]*Q[s,0]*Q[s,n])^2 + (sum_s Vk_im[s]*Q[s,0]*Q[s,n])^2
// -----------------------------------------------------------------------------

static double spectral_weight(int n_val, const MatR& Q_total, int dim, int N,
                                double k,
                                const std::vector<int>& basis) {
    // Ground state is column 0; excited state is column n_val
    double re_sum = 0.0, im_sum = 0.0;
    double inv_sqrtN = 1.0 / std::sqrt(static_cast<double>(N));
    for (int s_idx = 0; s_idx < dim; ++s_idx) {
        int s = basis[s_idx];
        // Compute Vk[s] = (1/sqrt(N)) sum_j exp(ikj) * sz_j(s)
        double vr = 0.0, vi = 0.0;
        for (int j = 0; j < N; ++j) {
            double sz_j = (((s >> j) & 1) == 0) ? 0.5 : -0.5;
            vr += std::cos(k * j) * sz_j;
            vi += std::sin(k * j) * sz_j;
        }
        vr *= inv_sqrtN;
        vi *= inv_sqrtN;
        double psi0_s = Q_total[s_idx * dim + 0];     // GS amplitude at s
        double psin_s = Q_total[s_idx * dim + n_val];  // excited state at s
        re_sum += vr * psi0_s * psin_s;
        im_sum += vi * psi0_s * psin_s;
    }
    return re_sum * re_sum + im_sum * im_sum;
}

// Lorentzian broadening: L(x, eta) = eta / (pi * (x^2 + eta^2))
inline double lorentzian(double x, double eta) {
    return eta / (M_PI * (x * x + eta * eta));
}

// -----------------------------------------------------------------------------
// F.  Compute full S(k, omega) heatmap
// -----------------------------------------------------------------------------

static void compute_spectral(int N, double J, double delta, double h,
                              int n_k, int n_omega,
                              double omega_min, double omega_max, double eta,
                              std::vector<double>& k_vals,
                              std::vector<double>& omega_vals,
                              std::vector<double>& spectral_out,
                              double& E_ground) {
    auto basis = sz0_basis(N);
    int dim = static_cast<int>(basis.size());
    std::vector<int> idx_map;
    MatR H = build_xxz(N, J, delta, h, basis, idx_map);

    // Full diagonalization
    std::vector<double> alpha, beta;
    MatR V;
    householder_tridiag(H, alpha, beta, dim, &V);
    std::vector<double> evals;
    MatR Q_trid;
    tridiag_eig_full(alpha, beta, evals, Q_trid);

    // Q_total = V * Q_trid  (eigenvectors of original H, columns)
    MatR Q_total(static_cast<std::size_t>(dim)*dim, 0.0);
    for (int row = 0; row < dim; ++row)
        for (int col = 0; col < dim; ++col)
            for (int kk = 0; kk < dim; ++kk)
                Q_total[row*dim+col] += V[row*dim+kk] * Q_trid[kk*dim+col];

    E_ground = evals[0];  // already sorted, evals[0] is the minimum

    // Build k and omega grids
    k_vals.resize(n_k);
    for (int ik = 0; ik < n_k; ++ik)
        k_vals[ik] = 2.0 * M_PI * ik / (n_k - 1);

    omega_vals.resize(n_omega);
    double dw = (omega_max - omega_min) / (n_omega - 1);
    for (int iw = 0; iw < n_omega; ++iw)
        omega_vals[iw] = omega_min + dw * iw;

    spectral_out.assign(static_cast<std::size_t>(n_k) * n_omega, 0.0);

    for (int ik = 0; ik < n_k; ++ik) {
        double k = k_vals[ik];
        // Pre-compute spectral weights |<n|S^z_k|0>|^2 for all n
        std::vector<double> weights(dim, 0.0);
        for (int n = 1; n < dim; ++n)  // n=0 is GS; S(k,0->0) vanishes for S^z_k in Sz=0 sector
            weights[n] = spectral_weight(n, Q_total, dim, N, k, basis);

        // Convolve with Lorentzian on omega grid
        for (int iw = 0; iw < n_omega; ++iw) {
            double om = omega_vals[iw];
            double val = 0.0;
            for (int n = 1; n < dim; ++n) {
                double dE = evals[n] - evals[0];
                val += weights[n] * lorentzian(om - dE, eta);
            }
            spectral_out[ik * n_omega + iw] = val;
        }
    }
}

// -----------------------------------------------------------------------------
// G.  Imaginary-time projection  E(tau) and fidelity(tau)
//     Given full spectrum {E_n} and coefficients c_n = <n|psi_0> of a random
//     initial state (uniform c_n = 1/sqrt(dim) for simplicity):
//     <E>(tau) = sum_n c_n^2 E_n exp(-2 tau (E_n-E_0)) / Z(tau)
//     fidelity = c_0^2 exp(-2 tau (E_0-E_0)) / Z(tau) = c_0^2 / Z(tau) * Z(tau)
//              = c_0^2 exp(0) / Z(tau) = c_0^2 / Z_norm
// -----------------------------------------------------------------------------

static void compute_imagtime(const std::vector<double>& evals,
                              double tau_max, int n_tau,
                              std::vector<double>& tau_vals,
                              std::vector<double>& energy_tau,
                              std::vector<double>& fidelity_tau) {
    int dim = static_cast<int>(evals.size());
    double E0 = evals[0];

    // Uniform initial state: c_n^2 = 1/dim
    double c0_sq = 1.0 / dim;

    tau_vals.resize(n_tau);
    energy_tau.resize(n_tau);
    fidelity_tau.resize(n_tau);

    for (int it = 0; it < n_tau; ++it) {
        double tau = tau_max * it / (n_tau - 1);
        tau_vals[it] = tau;

        // Z(tau) = sum_n c_n^2 * exp(-2*tau*(E_n - E_0))
        double Z = 0.0, E_avg = 0.0;
        for (int n = 0; n < dim; ++n) {
            double w = c0_sq * std::exp(-2.0 * tau * (evals[n] - E0));
            Z     += w;
            E_avg += w * evals[n];
        }
        energy_tau[it]  = E_avg / Z;
        fidelity_tau[it] = c0_sq / Z;   // fidelity = weight of GS / Z_norm
    }
}

// -----------------------------------------------------------------------------
// H.  Eigenvalues-only path (fast, for imagtime without spectral)
// -----------------------------------------------------------------------------

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
            double r = std::sqrt(x*x+z*z);
            double c = x/r, s = z/r;
            if (i > start) beta[i-1] = r;
            double a0=alpha[i], a1=alpha[i+1], b=beta[i];
            alpha[i]  =c*c*a0+2*c*s*b+s*s*a1;
            alpha[i+1]=s*s*a0-2*c*s*b+c*c*a1;
            beta[i]   =c*s*(a1-a0)+(c*c-s*s)*b;
            if (i < end-1) { x=beta[i]; z=-s*beta[i+1]; beta[i+1]*=c; }
        }
    }
    std::sort(alpha.begin(), alpha.end());
    return alpha;
}

}  // anonymous namespace

// =============================================================================
// I.  Public entry point
// =============================================================================

SpectralResult run_spectral(const SpectralOptions& opts) {
    auto t0 = std::chrono::steady_clock::now();

    if (opts.n_sites < 2 || opts.n_sites > 16 || opts.n_sites % 2 != 0)
        throw std::runtime_error("Spectral: n_sites must be even and in [2, 16]");
    if (opts.J == 0.0)
        throw std::runtime_error("Spectral: J must be non-zero");
    if (opts.h != 0.0)
        throw std::runtime_error(
            "Spectral: h != 0 has no effect in the Sz=0 sector — the uniform "
            "Zeeman term h*sum_i Sz_i vanishes identically on all Sz=0 basis "
            "states.  Set h = 0.0 (default) or implement a multi-sector solver.");
    if (opts.n_k < 2)
        throw std::runtime_error("Spectral: n_k must be >= 2");
    if (opts.n_omega < 2)
        throw std::runtime_error("Spectral: n_omega must be >= 2");
    if (opts.eta <= 0.0)
        throw std::runtime_error("Spectral: eta must be positive");
    if (opts.n_tau < 2)
        throw std::runtime_error("Spectral: n_tau must be >= 2");

    SpectralResult res;
    res.n_sites  = opts.n_sites;
    res.J        = opts.J;
    res.delta    = opts.delta;
    res.h        = opts.h;
    res.scenario = opts.scenario;

    bool do_spectral = (opts.scenario == "spectral" || opts.scenario == "both");
    bool do_imagtime = (opts.scenario == "imagtime" || opts.scenario == "both");

    if (!do_spectral && !do_imagtime)
        throw std::runtime_error("Spectral: unknown scenario '" + opts.scenario + "'");

    if (do_spectral) {
        compute_spectral(opts.n_sites, opts.J, opts.delta, opts.h,
                         opts.n_k, opts.n_omega,
                         opts.omega_min, opts.omega_max, opts.eta,
                         res.k_vals, res.omega_vals, res.spectral, res.E_ground);
    }

    if (do_imagtime) {
        // If we already have eigenvalues from spectral, reuse them;
        // otherwise compute eigenvalues only (fast path)
        if (!do_spectral) {
            auto basis = sz0_basis(opts.n_sites);
            int dim = static_cast<int>(basis.size());
            std::vector<int> idx_map;
            MatR H = build_xxz(opts.n_sites, opts.J, opts.delta, opts.h,
                                basis, idx_map);
            std::vector<double> alpha, beta;
            householder_tridiag(H, alpha, beta, dim);
            auto evals = tridiag_evals(alpha, beta);
            res.E_ground = evals[0];
            compute_imagtime(evals, opts.tau_max, opts.n_tau,
                             res.tau_vals, res.energy_tau, res.fidelity_tau);
        } else {
            // Build eigenvalues from existing spectral data: need to recompute
            // (E_ground already set; for imagtime we need all evals)
            auto basis = sz0_basis(opts.n_sites);
            int dim = static_cast<int>(basis.size());
            std::vector<int> idx_map;
            MatR H = build_xxz(opts.n_sites, opts.J, opts.delta, opts.h,
                                basis, idx_map);
            std::vector<double> alpha, beta;
            householder_tridiag(H, alpha, beta, dim);
            auto evals = tridiag_evals(alpha, beta);
            compute_imagtime(evals, opts.tau_max, opts.n_tau,
                             res.tau_vals, res.energy_tau, res.fidelity_tau);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    res.ok = true;
    return res;
}

}  // namespace qx
