// src/hofstadter.cpp — M31-L: Hofstadter Butterfly
//
// The Hofstadter butterfly (1976) arises from a 2D square-lattice tight-binding
// model in a perpendicular magnetic field.  When the flux per plaquette is a
// rational number φ = p/q, the magnetic unit cell is q sites tall and the
// spectrum splits into exactly q bands — producing the famous fractal structure
// when plotted as allowed energies vs φ ∈ [0,1].
//
// Peierls substitution (Landau gauge, flux per plaquette φ = p/q):
//   Hopping along x: t_{j → j+1 in x} = t  (no phase in Landau gauge)
//   Hopping along y: t_{j → j+1 in y at column n} = t · exp(i 2π φ n)
//
// Bloch Hamiltonian H(kx, ky) is q×q (magnetic BZ: kx ∈ [0, 2π/q], ky ∈ [0, 2π]):
//   H[n,n]   = −2t cos(2π φ n − ky)
//   H[n,n±1] = −t  (nearest-neighbour x-hopping; ±1 with periodic BC)
//   Corner:  H[0, q−1] = −t exp(−i q kx),  H[q−1, 0] = h.c.
//
// At kx = 0: H is real symmetric (tridiagonal + periodic corners).
// At kx ≠ 0: H is complex Hermitian — represented as a 2q×2q real symmetric
//            matrix via the standard real doubling H_real = [[Re H, −Im H],[Im H, Re H]].
//            Eigenvalues appear in degenerate pairs; we keep only q unique values.
//
// Scenarios:
//   butterfly — for all rationals p/q (gcd=1, q ≤ q_max), sample n_ky ky values
//               at kx = 0 and collect (φ, E) pairs → fractal scatter plot
//   bands     — for fixed φ = p/q (nearest rational), scan kx ∈ [0, 2π/q]
//               with ky = 0 and return q bands vs kx → TKNN band structure

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
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

// ─────────────────────────────────────────────────────────────────────────────
// A.  Wilkinson QR tridiagonal eigensolver (eigenvalues only)
//     Identical to the one in chern.cpp — kept local to avoid linkage issues.
// ─────────────────────────────────────────────────────────────────────────────

static std::vector<double> tridiag_evals(std::vector<double> alpha,
                                          std::vector<double> beta) {
    int n = static_cast<int>(alpha.size());
    for (int end = n - 1; end > 0; ) {
        for (int i = 0; i < end; ++i)
            if (std::abs(beta[i]) < 1e-14 * (std::abs(alpha[i]) + std::abs(alpha[i+1])))
                beta[i] = 0.0;
        int start = end;
        while (start > 0 && beta[start-1] != 0.0) --start;
        if (start == end) { --end; continue; }
        double d  = (alpha[end-1] - alpha[end]) * 0.5;
        double sg = (d >= 0.0) ? 1.0 : -1.0;
        double mu = alpha[end] - sg * beta[end-1] * beta[end-1] /
                    (std::abs(d) + std::sqrt(d * d + beta[end-1] * beta[end-1]));
        double x = alpha[start] - mu, z = beta[start];
        for (int i = start; i < end; ++i) {
            double r = std::sqrt(x * x + z * z);
            double c = x / r, s = z / r;
            if (i > start) beta[i-1] = r;
            double a0 = alpha[i], a1 = alpha[i+1], b = beta[i];
            alpha[i]   = c*c*a0 + 2*c*s*b + s*s*a1;
            alpha[i+1] = s*s*a0 - 2*c*s*b + c*c*a1;
            beta[i]    = c*s*(a1-a0) + (c*c-s*s)*b;
            if (i < end-1) { x = beta[i]; z = -s * beta[i+1]; beta[i+1] *= c; }
        }
    }
    std::sort(alpha.begin(), alpha.end());
    return alpha;
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  Householder tridiagonalisation (for the real-doubling approach)
// ─────────────────────────────────────────────────────────────────────────────

static void householder_tridiag(const MatR& A, std::vector<double>& alpha,
                                std::vector<double>& beta, int n) {
    alpha.resize(n); beta.resize(n - 1);
    MatR Q = A;
    for (int k = 0; k < n - 2; ++k) {
        double norm2 = 0.0;
        for (int i = k + 1; i < n; ++i) norm2 += Q[i * n + k] * Q[i * n + k];
        double norm = std::sqrt(norm2);
        if (norm < 1e-15) continue;
        std::vector<double> v(n, 0.0);
        for (int i = k + 1; i < n; ++i) v[i] = Q[i * n + k];
        v[k+1] += (v[k+1] >= 0.0 ? 1.0 : -1.0) * norm;
        double vn2 = 0.0;
        for (int i = k + 1; i < n; ++i) vn2 += v[i] * v[i];
        if (vn2 < 1e-30) continue;
        std::vector<double> p(n, 0.0);
        for (int r = 0; r < n; ++r)
            for (int c = k+1; c < n; ++c) p[r] += Q[r*n+c] * v[c];
        for (int r = 0; r < n; ++r) p[r] /= (vn2 / 2.0);
        double vTp = 0.0;
        for (int i = k+1; i < n; ++i) vTp += v[i] * p[i];
        std::vector<double> w(n);
        for (int i = 0; i < n; ++i) w[i] = p[i] - (vTp / vn2) * v[i];
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c)
                Q[r*n+c] -= v[r]*w[c] + w[r]*v[c];
    }
    for (int i = 0; i < n; ++i) alpha[i] = Q[i * n + i];
    for (int i = 0; i < n - 1; ++i) beta[i] = Q[(i+1)*n+i];
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  Harper matrix builders
// ─────────────────────────────────────────────────────────────────────────────

// GCD helper
static int gcd(int a, int b) { return b == 0 ? a : gcd(b, a % b); }

// Build q×q Harper matrix at kx=0, given ky (real symmetric, has corner elements).
// H[n,n]   = −2t cos(2π (p/q) n − ky)
// H[n,n+1] = H[n+1,n] = −t  (for n=0…q−2)
// H[0,q-1] = H[q-1,0] = −t  (periodic BC at kx=0)
static MatR build_harper_real(int p, int q, double ky, double t) {
    MatR H(static_cast<std::size_t>(q) * q, 0.0);
    double phi = static_cast<double>(p) / q;
    for (int n = 0; n < q; ++n) {
        H[n * q + n] = -2.0 * t * std::cos(2.0 * M_PI * phi * n - ky);
    }
    for (int n = 0; n < q - 1; ++n) {
        H[n * q + (n+1)] = -t;
        H[(n+1) * q + n] = -t;
    }
    // Periodic corners
    H[0 * q + (q-1)] = -t;
    H[(q-1) * q + 0] = -t;
    return H;
}

// Build 2q×2q real-symmetric matrix from complex q×q Harper at given kx, ky.
// H_real = [[Re H, −Im H], [Im H, Re H]]
// Only the off-diagonal corner elements have imaginary parts (Peierls phase).
// At kx=0 the corner phase is 1 (real) → Im corners = 0 → H_real is block-diagonal
// and we just get degenerate pairs. For kx≠0 the corners pick up phase e^{iqkx}.
static MatR build_harper_doubled(int p, int q, double kx, double ky, double t) {
    int dim = 2 * q;
    MatR H(static_cast<std::size_t>(dim) * dim, 0.0);

    double phi  = static_cast<double>(p) / q;
    // Re(H) part: upper-left and lower-right q×q blocks
    // Im(H) part: upper-right (−) and lower-left (+) q×q blocks
    // Diagonal: −2t cos(...), real → only in Re block
    for (int n = 0; n < q; ++n) {
        double d = -2.0 * t * std::cos(2.0 * M_PI * phi * n - ky);
        H[n * dim + n]           = d;   // Re block
        H[(n+q)*dim + (n+q)]     = d;   // Re block (lower-right)
    }
    // Interior hoppings: real → appear in Re block only
    for (int n = 0; n < q-1; ++n) {
        H[n * dim + (n+1)] = -t;  H[(n+1)*dim + n] = -t;        // UL Re block
        H[(n+q)*dim + (n+1+q)] = -t;  H[(n+1+q)*dim + (n+q)] = -t;  // LR Re block
    }
    // Corner hoppings: H[0,q−1] = −t e^{−iqkx}, H[q−1,0] = −t e^{+iqkx}
    double phase_re = -t * std::cos(q * kx);
    double phase_im = -t * std::sin(q * kx);   // Im part of −t e^{−iqkx}
    // H_real layout: [[Re H, −Im H],[Im H, Re H]]
    // Upper-left (Re H) block: corner = phase_re
    H[0 * dim + (q-1)] = phase_re;
    H[(q-1)*dim + 0]   = phase_re;
    // Upper-right (−Im H) block: [0, q-1+q]=−phase_im, [0+q, q-1]=+phase_im, etc.
    H[0     * dim + (q-1+q)] = -(-phase_im);   // −(−Im H[0,q-1]) = Im(−t e^{−iqkx}) … wait
    // Let's be careful: H_real = [[A, -B],[B, A]] where A=Re H, B=Im H
    // H[0,q-1] (upper-right in complex H) = phase_re + i*phase_im'
    // phase_im' = Im(−t e^{−iqkx}) = t sin(qkx)  (minus sign from −t, minus from e^{-i})
    double im_corner = t * std::sin(q * kx);   // Im(−t e^{−iqkx}) = t sin(qkx)
    // Upper-right block (−B): row 0, col q−1+q  and row q−1, col 0+q
    H[0 * dim + (q-1+q)]   = -im_corner;      // −Im[H[0,q-1]] = −t sin(qkx) … typo check
    H[(q-1)*dim + (0+q)]   = -im_corner;      // symmetric
    // Lower-left block (+B): row 0+q, col q-1  and row q-1+q, col 0
    H[(0+q)*dim + (q-1)]   = +im_corner;
    H[(q-1+q)*dim + 0]     = +im_corner;
    // Lower-right Re block corners
    H[(0+q)*dim + (q-1+q)] = phase_re;
    H[(q-1+q)*dim + (0+q)] = phase_re;
    return H;
}

// ─────────────────────────────────────────────────────────────────────────────
// D.  Butterfly scan: collect (phi, eigenvalue) pairs
// ─────────────────────────────────────────────────────────────────────────────

static void butterfly_scan(int q_max, int n_ky, double t,
                            std::vector<double>& phi_out,
                            std::vector<double>& eval_out) {
    phi_out.clear(); eval_out.clear();
    // Reserve estimated capacity
    phi_out.reserve(static_cast<std::size_t>(q_max * q_max * n_ky / 2));
    eval_out.reserve(phi_out.capacity());

    for (int q = 1; q <= q_max; ++q) {
        for (int p = 1; p < q; ++p) {
            if (gcd(p, q) != 1) continue;   // only coprime fractions
            double phi = static_cast<double>(p) / q;
            for (int iky = 0; iky < n_ky; ++iky) {
                double ky = 2.0 * M_PI * iky / n_ky;
                MatR H   = build_harper_real(p, q, ky, t);
                std::vector<double> alpha, beta;
                householder_tridiag(H, alpha, beta, q);
                auto evals = tridiag_evals(alpha, beta);
                for (double e : evals) {
                    phi_out.push_back(phi);
                    eval_out.push_back(e);
                }
            }
            // Also include phi=0 and phi=1 boundaries once
        }
    }
    // Add phi=0 (trivial: 1 band) and phi=1 (= phi=0 by periodicity)
    // Just add the single cosine band at phi=0
    for (int iky = 0; iky < n_ky; ++iky) {
        double ky = 2.0 * M_PI * iky / n_ky;
        phi_out.push_back(0.0);
        eval_out.push_back(-2.0 * t * std::cos(ky));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E.  Fixed-flux band structure: kx scan at ky = 0
// ─────────────────────────────────────────────────────────────────────────────

static void fixed_phi_bands(int p, int q, int Nk, double t,
                             std::vector<double>& kx_vals,
                             std::vector<std::vector<double>>& bands) {
    kx_vals.resize(Nk);
    bands.resize(Nk, std::vector<double>(q));

    double kx_max = 2.0 * M_PI / q;  // magnetic BZ in kx direction
    for (int ik = 0; ik < Nk; ++ik) {
        double kx = kx_max * ik / (Nk - 1);
        kx_vals[ik] = kx;

        if (std::abs(kx) < 1e-12) {
            // kx=0: real symmetric, use Householder + tridiag
            MatR H = build_harper_real(p, q, 0.0, t);
            std::vector<double> alpha, beta;
            householder_tridiag(H, alpha, beta, q);
            bands[ik] = tridiag_evals(alpha, beta);
        } else {
            // kx≠0: complex → real doubling (2q×2q)
            MatR H2 = build_harper_doubled(p, q, kx, 0.0, t);
            std::vector<double> alpha, beta;
            householder_tridiag(H2, alpha, beta, 2 * q);
            auto ev2 = tridiag_evals(alpha, beta);
            // Degenerate pairs: take every other eigenvalue (0,2,4,...)
            bands[ik].resize(q);
            for (int n = 0; n < q; ++n) bands[ik][n] = ev2[2 * n];
        }
    }
}

// Nearest rational p/q to a given double phi, with q ≤ q_max
static std::pair<int,int> nearest_rational(double phi, int q_max) {
    int best_p = 1, best_q = 4;
    double best_err = 1e30;
    for (int q = 1; q <= q_max; ++q) {
        int p = static_cast<int>(std::round(phi * q));
        p = std::max(1, std::min(p, q - 1));
        if (gcd(p, q) != 1) continue;
        double err = std::abs(static_cast<double>(p) / q - phi);
        if (err < best_err) { best_err = err; best_p = p; best_q = q; }
    }
    return {best_p, best_q};
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// F.  Public entry point
// ─────────────────────────────────────────────────────────────────────────────

HofstadterResult run_hofstadter(const HofstadterOptions& opts) {
    auto t0 = std::chrono::steady_clock::now();

    if (opts.t <= 0.0)
        throw std::runtime_error("Hofstadter: t must be positive");
    if (opts.q_max < 2 || opts.q_max > 60)
        throw std::runtime_error("Hofstadter: q_max must be in [2, 60]");
    if (opts.n_k < 2)
        throw std::runtime_error("Hofstadter: n_k must be >= 2");

    HofstadterResult res;

    if (opts.scenario == "butterfly") {
        butterfly_scan(opts.q_max, opts.n_ky, opts.t,
                       res.phi_vals, res.evals);
    } else if (opts.scenario == "bands") {
        auto [p, q] = nearest_rational(opts.phi, opts.q_max);
        res.phi_used = static_cast<double>(p) / q;
        res.n_bands  = q;
        fixed_phi_bands(p, q, opts.n_k, opts.t, res.kx_vals, res.bands);
    } else {
        throw std::runtime_error("Hofstadter: unknown scenario '" + opts.scenario + "'");
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.ok = true;
    return res;
}

}  // namespace qx
