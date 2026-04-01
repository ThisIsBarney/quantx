// src/tfi.cpp — M29-G: 1D Transverse-Field Ising Model
//
//   H = -J Σ_{i=0}^{N-2} Z_i Z_{i+1}  -  h Σ_{i=0}^{N-1} X_i
//
// Quantum phase transition at h_c = J (ferromagnetic ↔ paramagnetic).
// Written in the Z-basis {|↓⟩,|↑⟩}^⊗N with open boundary conditions.
//
// The Hamiltonian is Z₂-symmetric (global spin flip Π = Π X_i commutes with H).
// We build the full 2^N × 2^N matrix and do dense exact diagonalization
// via Householder tridiagonalization + Wilkinson QR (same as phi4.cpp).
// For N ≤ 18 this is fast enough; default N=10.
//
// Observables:
//   E₀, gap Δ = E₁ - E₀
//   Order parameter |⟨Z_0⟩| (non-zero in FM phase, zero in PM phase)
//   Spin-spin correlator ⟨Z_0 Z_r⟩ for r = 0..N-1
//   Correlation length ξ extracted from exponential fit of |⟨Z_0 Z_r⟩|
//
// Scenarios:
//   ground_state  — single (J, h) ground state + observables
//   sweep         — scan h from h_min to h_max, track order_param + gap
//   dynamics      — linear Kibble-Zurek ramp h(t) = h_quench - (h_quench-h)*t/T
//                   with Arnoldi-Krylov time propagation

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace qx {
namespace {

using cx   = std::complex<double>;
using MatR = std::vector<double>;
using MatC = std::vector<cx>;

// ─────────────────────────────────────────────────────────────────────────────
// A.  Dense symmetric matrix × vector  (real)
// ─────────────────────────────────────────────────────────────────────────────

static void sym_matvec(const MatR& A, const std::vector<double>& x,
                       std::vector<double>& y, int n) {
    y.assign(n, 0.0);
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            y[r] += A[r * n + c] * x[c];
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  Hamiltonian builder  (full 2^N × 2^N dense matrix)
// ─────────────────────────────────────────────────────────────────────────────

// Bit convention: bit i = 0 → spin up (↑), bit i = 1 → spin down (↓).
//   Z_i |s⟩ = ±|s⟩   where +1 for bit=0, −1 for bit=1
//   X_i flips bit i

static MatR build_tfi_matrix(int N, double J, double h) {
    const int dim = 1 << N;
    MatR H(static_cast<std::size_t>(dim) * dim, 0.0);

    auto idx = [&](int s) { return static_cast<std::size_t>(s); };

    for (int s = 0; s < dim; ++s) {
        // Diagonal: -J Σ Z_i Z_{i+1}
        double diag = 0.0;
        for (int i = 0; i < N - 1; ++i) {
            int zi = ((s >> i) & 1) ? -1 : 1;
            int zj = ((s >> (i + 1)) & 1) ? -1 : 1;
            diag += -J * zi * zj;
        }
        H[idx(s) * dim + idx(s)] += diag;

        // Off-diagonal: -h X_i = -h (flip bit i)
        for (int i = 0; i < N; ++i) {
            int s2 = s ^ (1 << i);
            H[idx(s) * dim + idx(s2)] += -h;
        }
    }
    return H;
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  Householder tridiagonalization (real symmetric)
// ─────────────────────────────────────────────────────────────────────────────

static void householder_tridiag_real(const MatR& A, std::vector<double>& alpha,
                                      std::vector<double>& beta, int n) {
    alpha.resize(n); beta.resize(n - 1);
    MatR Q = A; // copy; we will overwrite with tridiagonal form in-place via similarity

    for (int k = 0; k < n - 2; ++k) {
        // Form Householder vector for column k below diagonal
        double norm2 = 0.0;
        for (int i = k + 1; i < n; ++i) norm2 += Q[i * n + k] * Q[i * n + k];
        double norm = std::sqrt(norm2);
        if (norm < 1e-15) continue;

        std::vector<double> v(n, 0.0);
        for (int i = k + 1; i < n; ++i) v[i] = Q[i * n + k];
        v[k + 1] += (v[k + 1] >= 0.0 ? 1.0 : -1.0) * norm;
        double vnorm2 = 0.0;
        for (int i = k + 1; i < n; ++i) vnorm2 += v[i] * v[i];
        if (vnorm2 < 1e-30) continue;

        // P = I - 2 v v^T / (v^T v)
        // Q ← P Q P  (similarity)
        // Compute p = Q v / vnorm2
        std::vector<double> p(n, 0.0);
        for (int r = 0; r < n; ++r)
            for (int c = k + 1; c < n; ++c)
                p[r] += Q[r * n + c] * v[c];
        for (int r = 0; r < n; ++r) p[r] /= (vnorm2 / 2.0);

        double vTpv = 0.0;
        for (int i = k + 1; i < n; ++i) vTpv += v[i] * p[i];
        std::vector<double> w(n);
        for (int i = 0; i < n; ++i) w[i] = p[i] - (vTpv / vnorm2) * v[i];

        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c)
                Q[r * n + c] -= v[r] * w[c] + w[r] * v[c];
    }

    for (int i = 0; i < n; ++i) alpha[i] = Q[i * n + i];
    for (int i = 0; i < n - 1; ++i) beta[i] = Q[(i + 1) * n + i];
}

// ─────────────────────────────────────────────────────────────────────────────
// D.  Wilkinson-shift implicit QR for symmetric tridiagonal
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<double> tridiag_eig(std::vector<double> alpha,
                                        std::vector<double> beta,
                                        MatR* evecs = nullptr) {
    int n = static_cast<int>(alpha.size());
    MatR Q;
    if (evecs) { Q.assign(static_cast<std::size_t>(n) * n, 0.0); for (int i = 0; i < n; ++i) Q[i * n + i] = 1.0; }

    for (int end = n - 1; end > 0; ) {
        // Deflate small off-diagonals
        for (int i = 0; i < end; ++i)
            if (std::abs(beta[i]) < 1e-14 * (std::abs(alpha[i]) + std::abs(alpha[i + 1])))
                beta[i] = 0.0;

        int start = end;
        while (start > 0 && beta[start - 1] != 0.0) --start;

        if (start == end) { --end; continue; }

        // Wilkinson shift
        double d  = (alpha[end - 1] - alpha[end]) * 0.5;
        double sg = (d >= 0.0) ? 1.0 : -1.0;
        double mu = alpha[end] - sg * beta[end - 1] * beta[end - 1] /
                    (std::abs(d) + std::sqrt(d * d + beta[end - 1] * beta[end - 1]));

        double x = alpha[start] - mu;
        double z = beta[start];

        for (int i = start; i < end; ++i) {
            double r = std::sqrt(x * x + z * z);
            double c = x / r, s = z / r;

            if (i > start) beta[i - 1] = r;
            double a0 = alpha[i], a1 = alpha[i + 1], b = beta[i];
            alpha[i]     = c * c * a0 + 2 * c * s * b + s * s * a1;
            alpha[i + 1] = s * s * a0 - 2 * c * s * b + c * c * a1;
            beta[i]      = c * s * (a1 - a0) + (c * c - s * s) * b;

            if (i < end - 1) { x = beta[i]; z = -s * beta[i + 1]; beta[i + 1] *= c; }

            if (evecs) {
                for (int row = 0; row < n; ++row) {
                    double qi  = Q[row * n + i];
                    double qi1 = Q[row * n + i + 1];
                    Q[row * n + i]     = c * qi + s * qi1;
                    Q[row * n + i + 1] = -s * qi + c * qi1;
                }
            }
        }
    }

    if (evecs) *evecs = Q;

    // Sort ascending
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return alpha[a] < alpha[b]; });
    std::vector<double> sorted(n);
    for (int i = 0; i < n; ++i) sorted[i] = alpha[idx[i]];
    if (evecs) {
        MatR Qsorted(static_cast<std::size_t>(n) * n);
        for (int col = 0; col < n; ++col)
            for (int row = 0; row < n; ++row)
                Qsorted[row * n + col] = (*evecs)[row * n + idx[col]];
        *evecs = Qsorted;
    }
    return sorted;
}

// ─────────────────────────────────────────────────────────────────────────────
// E.  Dense exact diagonalization wrapper
// ─────────────────────────────────────────────────────────────────────────────

struct EigenDecomp {
    std::vector<double> evals;   // all eigenvalues, sorted ascending
    MatR                evecs;   // columns = eigenvectors (row-major storage)
};

static EigenDecomp dense_eig(const MatR& H, int n, bool want_vecs = false) {
    std::vector<double> alpha, beta;
    householder_tridiag_real(H, alpha, beta, n);
    EigenDecomp ed;
    if (want_vecs) {
        ed.evals = tridiag_eig(alpha, beta, &ed.evecs);
    } else {
        ed.evals = tridiag_eig(alpha, beta);
    }
    return ed;
}

// ─────────────────────────────────────────────────────────────────────────────
// F.  Observables  (ground-state vector required)
// ─────────────────────────────────────────────────────────────────────────────

// ⟨Z_i⟩ from ground state vector
static double expect_Z(const std::vector<double>& psi, int N, int site) {
    int dim = 1 << N;
    double val = 0.0;
    for (int s = 0; s < dim; ++s) {
        int zi = ((s >> site) & 1) ? -1 : 1;
        val += zi * psi[s] * psi[s];
    }
    return val;
}

// ⟨Z_i Z_j⟩ from ground state vector
static double expect_ZiZj(const std::vector<double>& psi, int N, int i, int j) {
    int dim = 1 << N;
    double val = 0.0;
    for (int s = 0; s < dim; ++s) {
        int zi = ((s >> i) & 1) ? -1 : 1;
        int zj = ((s >> j) & 1) ? -1 : 1;
        val += zi * zj * psi[s] * psi[s];
    }
    return val;
}

// Correlation length: fit |C(r)| ~ A exp(-r/ξ) for r ≥ 1
static double correlation_length(const std::vector<double>& corr) {
    int N = static_cast<int>(corr.size());
    // Log-linear regression on r = 1..N/2-1 where |C(r)| > 1e-12
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0; int cnt = 0;
    for (int r = 1; r < N / 2; ++r) {
        double c = std::abs(corr[r]);
        if (c < 1e-12) continue;
        double lc = std::log(c);
        sx  += r;  sy  += lc;
        sxx += r * r; sxy += r * lc;
        ++cnt;
    }
    if (cnt < 2) return 0.0;
    double slope = (cnt * sxy - sx * sy) / (cnt * sxx - sx * sx);
    return (slope < -1e-12) ? -1.0 / slope : 0.0;
}

// Bipartite entropy: half-chain cut, using Schmidt decomposition via SVD of the
// reshaped state matrix (2^(N/2) x 2^(N/2)).  We use a simple power-iteration
// method: build ρ_A = Tr_B |ψ⟩⟨ψ| and diagonalize.
static double entropy_half_chain(const std::vector<double>& psi, int N) {
    int dA = 1 << (N / 2), dB = 1 << (N - N / 2);
    // Build reduced density matrix ρ_A[a1, a2] = Σ_b ψ[a1*dB+b] ψ[a2*dB+b]
    MatR rhoA(static_cast<std::size_t>(dA) * dA, 0.0);
    for (int a1 = 0; a1 < dA; ++a1)
        for (int a2 = 0; a2 < dA; ++a2)
            for (int b = 0; b < dB; ++b)
                rhoA[a1 * dA + a2] += psi[a1 * dB + b] * psi[a2 * dB + b];
    auto evals = dense_eig(rhoA, dA).evals;
    double S = 0.0;
    for (double ev : evals) if (ev > 1e-14) S -= ev * std::log(ev);
    return S;
}

// ─────────────────────────────────────────────────────────────────────────────
// G.  Arnoldi-Krylov time propagation  exp(-i dt H)|ψ⟩
// ─────────────────────────────────────────────────────────────────────────────

// Dense Taylor + squaring on small m×m matrix
static MatC dense_expm_cx(const MatC& A, int m) {
    // Scale-and-squaring: pick s so ||A/s|| ≤ 1
    double norm = 0.0;
    for (auto& a : A) norm = std::max(norm, std::abs(a));
    int s = 1;
    while (norm / s > 1.0 && s < 1024) s *= 2;

    // Taylor series exp(A/s)
    MatC E(static_cast<std::size_t>(m) * m, cx{0, 0});
    for (int i = 0; i < m; ++i) E[i * m + i] = cx{1, 0};
    MatC T = E;
    for (int k = 1; k <= 20; ++k) {
        MatC T2(static_cast<std::size_t>(m) * m, cx{0, 0});
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < m; ++c) {
                cx acc{0, 0};
                for (int l = 0; l < m; ++l)
                    acc += T[r * m + l] * A[l * m + c] / cx{(double)s, 0};
                T2[r * m + c] = acc;
            }
        T = T2;
        for (int i = 0; i < m * m; ++i) E[i] += T[i] / cx{(double)k, 0};
        if (norm / s / k < 1e-14) break;
    }

    // Squaring: E^s via repeated squaring (linear accumulation)
    if (s > 1) {
        MatC E_orig = E;
        for (int sq = 1; sq < s; ++sq) {
            MatC E2(static_cast<std::size_t>(m) * m, cx{0, 0});
            for (int r = 0; r < m; ++r)
                for (int c = 0; c < m; ++c) {
                    cx acc{0, 0};
                    for (int l = 0; l < m; ++l)
                        acc += E[r * m + l] * E_orig[l * m + c];
                    E2[r * m + c] = acc;
                }
            E = E2;
        }
    }
    return E;
}

static void arnoldi_step(
    const std::function<void(const std::vector<cx>&, std::vector<cx>&)>& Hmul,
    std::vector<std::vector<cx>>& Q,
    MatC& Hm, int m_kry, int N) {

    Q.resize(m_kry + 1, std::vector<cx>(N, cx{0, 0}));
    Hm.assign(static_cast<std::size_t>(m_kry) * m_kry, cx{0, 0});

    Hmul(Q[0], Q[1]);  // q1 = H q0
    cx h00{0, 0};
    for (int i = 0; i < N; ++i) h00 += std::conj(Q[0][i]) * Q[1][i];
    Hm[0] = h00;
    for (int i = 0; i < N; ++i) Q[1][i] -= h00 * Q[0][i];
    double norm = 0.0;
    for (int i = 0; i < N; ++i) norm += std::norm(Q[1][i]);
    norm = std::sqrt(norm);
    if (norm > 1e-14) for (int i = 0; i < N; ++i) Q[1][i] /= norm;
    if (m_kry > 1) Hm[m_kry] = cx{norm, 0};  // H[1,0]

    for (int j = 1; j < m_kry; ++j) {
        Hmul(Q[j], Q[j + 1]);
        for (int i = 0; i <= j; ++i) {
            cx hij{0, 0};
            for (int k = 0; k < N; ++k) hij += std::conj(Q[i][k]) * Q[j + 1][k];
            Hm[i * m_kry + j] = hij;
            for (int k = 0; k < N; ++k) Q[j + 1][k] -= hij * Q[i][k];
        }
        double bn = 0.0;
        for (int k = 0; k < N; ++k) bn += std::norm(Q[j + 1][k]);
        bn = std::sqrt(bn);
        if (j < m_kry - 1) Hm[(j + 1) * m_kry + j] = cx{bn, 0};
        if (bn > 1e-14) for (int k = 0; k < N; ++k) Q[j + 1][k] /= bn;
    }
}

// Apply exp(-i dt H)|ψ⟩ using m-step Arnoldi
static void krylov_expm(
    const std::function<void(const std::vector<cx>&, std::vector<cx>&)>& Hmul,
    std::vector<cx>& psi, double dt, int m_kry) {

    int N = static_cast<int>(psi.size());
    m_kry = std::min(m_kry, N);

    // Normalize
    double nrm = 0.0;
    for (auto& p : psi) nrm += std::norm(p);
    nrm = std::sqrt(nrm);
    for (auto& p : psi) p /= nrm;

    std::vector<std::vector<cx>> Q;
    Q.resize(1, psi);
    MatC Hm;
    arnoldi_step(Hmul, Q, Hm, m_kry, N);

    // Build -i dt Hm and exponentiate
    MatC A(static_cast<std::size_t>(m_kry) * m_kry);
    for (int i = 0; i < m_kry * m_kry; ++i) A[i] = cx{0, -dt} * Hm[i];
    MatC Em = dense_expm_cx(A, m_kry);

    // psi_new = Σ_j (Σ_k Q_k Em[k,0]) * nrm
    std::vector<cx> psi_new(N, cx{0, 0});
    for (int k = 0; k < m_kry; ++k) {
        cx coeff = Em[k * m_kry + 0] * cx{nrm, 0};
        for (int i = 0; i < N; ++i) psi_new[i] += coeff * Q[k][i];
    }
    psi = psi_new;
}

// ─────────────────────────────────────────────────────────────────────────────
// H.  Core solver
// ─────────────────────────────────────────────────────────────────────────────

struct TFIGround {
    double E0, E1;
    std::vector<double> psi0;  // ground-state vector
};

static TFIGround solve_tfi_ground(int N, double J, double h) {
    int dim = 1 << N;
    MatR H = build_tfi_matrix(N, J, h);
    EigenDecomp ed = dense_eig(H, dim, /*want_vecs=*/true);

    TFIGround g;
    g.E0 = ed.evals[0];
    g.E1 = (dim > 1) ? ed.evals[1] : ed.evals[0];
    g.psi0.resize(dim);
    for (int i = 0; i < dim; ++i) g.psi0[i] = ed.evecs[i * dim + 0];
    return g;
}

// Compute all TFI observables from a ground-state solution
static void fill_tfi_observables(int N, const TFIGround& g,
                                   double& order_param,
                                   double& corr_len,
                                   std::vector<double>& spin_corr) {
    order_param = std::abs(expect_Z(g.psi0, N, 0));
    spin_corr.resize(N);
    for (int r = 0; r < N; ++r)
        spin_corr[r] = expect_ZiZj(g.psi0, N, 0, r);
    corr_len = correlation_length(spin_corr);
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// I.  Public entry point
// ─────────────────────────────────────────────────────────────────────────────

TFIResult run_tfi(const TFIOptions& opts) {
    auto t0 = std::chrono::steady_clock::now();

    TFIResult res;
    res.n_sites = opts.n_sites;
    res.J = opts.J; res.h = opts.h;

    if (opts.n_sites < 2 || opts.n_sites > 22)
        throw std::runtime_error("TFI: n_sites must be in [2, 22]");
    if (opts.J <= 0.0)
        throw std::runtime_error("TFI: J must be positive");
    if (opts.h < 0.0)
        throw std::runtime_error("TFI: h must be non-negative");

    int N = opts.n_sites;

    if (opts.scenario == "ground_state" || opts.scenario == "dynamics") {
        TFIGround g = solve_tfi_ground(N, opts.J, opts.h);
        res.ground_energy = g.E0;
        res.energy_gap    = g.E1 - g.E0;
        fill_tfi_observables(N, g, res.order_param, res.corr_len, res.spin_corr);
    }

    if (opts.scenario == "sweep") {
        if (opts.n_sweep < 2)
            throw std::runtime_error("TFI: n_sweep must be >= 2");
        res.sweep.reserve(opts.n_sweep);
        for (int k = 0; k < opts.n_sweep; ++k) {
            double hk = opts.h_min + (opts.h_max - opts.h_min) * k / (opts.n_sweep - 1);
            TFIGround g = solve_tfi_ground(N, opts.J, hk);
            TFISweepPoint pt;
            pt.h = hk;
            pt.ground_energy = g.E0;
            pt.energy_gap    = g.E1 - g.E0;
            std::vector<double> sc;
            fill_tfi_observables(N, g, pt.order_param, pt.corr_len, sc);
            res.sweep.push_back(pt);
        }
        // Also fill single-point result at opts.h
        TFIGround g0 = solve_tfi_ground(N, opts.J, opts.h);
        res.ground_energy = g0.E0;
        res.energy_gap    = g0.E1 - g0.E0;
        fill_tfi_observables(N, g0, res.order_param, res.corr_len, res.spin_corr);
    }

    if (opts.scenario == "dynamics") {
        if (opts.n_steps < 1)
            throw std::runtime_error("TFI: n_steps must be >= 1");
        if (opts.dt <= 0.0)
            throw std::runtime_error("TFI: dt must be positive");

        int dim = 1 << N;
        double T_ramp = opts.dt * opts.n_steps;

        // Initial state: ground state at h_quench
        TFIGround g_init = solve_tfi_ground(N, opts.J, opts.h_quench);
        std::vector<cx> psi(dim);
        for (int i = 0; i < dim; ++i) psi[i] = cx{g_init.psi0[i], 0.0};

        // Precomputed static Hamiltonian at h_quench for matvec
        // We rebuild the matrix at each step since h changes
        res.frames.reserve(opts.n_steps + 1);

        auto make_matvec = [&](double h_now) {
            MatR H = build_tfi_matrix(N, opts.J, h_now);
            return [H, dim](const std::vector<cx>& x, std::vector<cx>& y) {
                y.assign(dim, cx{0, 0});
                for (int r = 0; r < dim; ++r)
                    for (int c = 0; c < dim; ++c)
                        y[r] += H[r * dim + c] * x[c];
            };
        };

        int m_kry = std::min(20, dim);

        for (int step = 0; step <= opts.n_steps; ++step) {
            double t = step * opts.dt;
            double h_now = opts.h_quench - (opts.h_quench - opts.h) * t / T_ramp;

            // Record frame
            TFIFrame fr;
            fr.time = t;
            fr.h_now = h_now;

            // ⟨Z_0⟩ order parameter from current state
            double op = 0.0;
            for (int s = 0; s < dim; ++s) {
                int zi = ((s) & 1) ? -1 : 1;
                op += zi * std::norm(psi[s]);
            }
            fr.order_param = std::abs(op);

            // Energy ⟨ψ|H(t)|ψ⟩
            MatR H = build_tfi_matrix(N, opts.J, h_now);
            double en = 0.0;
            for (int r = 0; r < dim; ++r) {
                cx row_sum{0, 0};
                for (int c = 0; c < dim; ++c) row_sum += H[r * dim + c] * psi[c];
                en += (std::conj(psi[r]) * row_sum).real();
            }
            fr.energy = en;

            // Bipartite entropy (half-chain, real part of psi since H is real)
            std::vector<double> psi_re(dim);
            for (int i = 0; i < dim; ++i) psi_re[i] = psi[i].real();
            // Renormalize real part (Krylov keeps psi complex but mostly real)
            double nr2 = 0.0; for (auto v : psi_re) nr2 += v * v;
            if (nr2 > 1e-10) { double nr = std::sqrt(nr2); for (auto& v : psi_re) v /= nr; }
            fr.entropy = entropy_half_chain(psi_re, N);

            res.frames.push_back(fr);

            if (step < opts.n_steps) {
                auto mv = make_matvec(h_now + 0.5 * (opts.h - opts.h_quench) / opts.n_steps);
                krylov_expm(mv, psi, opts.dt, m_kry);
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.ok = true;
    return res;
}

}  // namespace qx
