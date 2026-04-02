// src/chern.cpp — M31-K: 2D Chern Insulator (Qi-Wu-Zhang model)
//
// The QWZ model (Qi, Wu, Zhang 2006) is the simplest two-band model on a 2D
// square lattice that realises a quantised Hall conductance without an external
// magnetic field (a "Chern insulator" or quantum anomalous Hall insulator):
//
//   H(k) = sin(kx) σx + sin(ky) σy + [m + cos(kx) + cos(ky)] σz
//
// Topological phase diagram (FHS Chern number of lower band):
//   |C| = 1  for  −2 < m < 0  (C = −1 by FHS convention)
//   |C| = 1  for   0 < m < 2  (C = +1 by FHS convention, opposite chirality)
//   C  = 0  for  |m| > 2  (trivial)
// QPTs occur at m = 0, ±2: gap closes at (π,0)+(0,π), (0,0), (π,π) resp.
// The two topological phases have opposite edge-state chiralities.
//
// The Chern number is computed numerically via the Fukui-Hatsugai-Suzuki (FHS)
// lattice method (JPSJ 2005) on an n_k × n_k Brillouin-zone mesh.  This
// approach is gauge-invariant and gives exact integers even for coarse meshes.
//
// Edge states: strip geometry (Ny sites open in y, periodic in x).  The strip
// Hamiltonian H_strip(kx) is 2Ny × 2Ny real symmetric — solved with the same
// Householder + Wilkinson QR eigensolver used in the rest of the physics track.
//
// Scenarios:
//   bands   — Berry curvature map + Chern number on n_k × n_k BZ grid
//   edge    — strip spectrum E(kx) showing chiral edge modes
//   sweep   — scan m ∈ [m_min, m_max], record gap and Chern number

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <numeric>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace qx {
namespace {

// ─────────────────────────────────────────────────────────────────────────────
// A.  k-space QWZ Hamiltonian
//
//   H(k) = [[dz,   dx−idy],  dz = m + cos(kx) + cos(ky)
//            [dx+idy, −dz]]  dx = sin(kx),  dy = sin(ky)
//
// Lower-band (negative) eigenvector computed analytically.
// Upper-band energy E+ = +|d|.
// ─────────────────────────────────────────────────────────────────────────────

using cx   = std::complex<double>;
using Vec2 = std::array<cx, 2>;

inline double band_energy(double dz, double dx, double dy) {
    return std::sqrt(dz * dz + dx * dx + dy * dy);
}

// Analytic lower-band eigenvector (energy −E).
// Two complementary gauges avoid the gauge-singularity at either pole of the
// Bloch sphere; we switch gauges where |dz + E| is small.
static Vec2 lower_evec(double dz, double dx, double dy) {
    double E = band_energy(dz, dx, dy);
    if (E < 1e-15) return {cx(1.0, 0.0), cx(0.0, 0.0)};
    double denom = E + dz;
    cx c = cx(dx, -dy);              // dx − i dy
    if (std::abs(denom) > 1e-8 * E) {
        // Standard gauge: |u−⟩ ∝ [−(dx−idy)/(E+dz), 1]
        cx v0  = -c / denom;
        double nrm = std::sqrt(std::norm(v0) + 1.0);
        return {v0 / nrm, cx(1.0 / nrm, 0.0)};
    } else {
        // Alternate gauge (near south pole): |u−⟩ ∝ [1, (dx+idy)/(E−dz)]
        cx v1  = std::conj(c) / (E - dz);
        double nrm = std::sqrt(1.0 + std::norm(v1));
        return {cx(1.0 / nrm, 0.0), v1 / nrm};
    }
}

// Hermitian inner product ⟨a|b⟩
static cx dot2(const Vec2& a, const Vec2& b) {
    return std::conj(a[0]) * b[0] + std::conj(a[1]) * b[1];
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  FHS Chern number + Berry curvature on n_k × n_k BZ mesh
// ─────────────────────────────────────────────────────────────────────────────

static void compute_berry(int Nk, double m, double t,
                          std::vector<double>& lower, std::vector<double>& upper,
                          std::vector<double>& berry, double& chern, double& gap) {
    int N2 = Nk * Nk;
    lower.resize(N2);
    upper.resize(N2);
    berry.resize(N2, 0.0);

    double min_gap = 1e30;

    // Pre-compute eigenvectors at each k-point (index ix*Nk + iy)
    // Note: we use indices 0 … Nk (inclusive) with wrap for FHS plaquettes,
    // but store only 0 … Nk−1; the extra index wraps via modulo.
    auto idx  = [&](int ix, int iy) { return (ix % Nk) * Nk + (iy % Nk); };
    auto kval = [&](int i) { return 2.0 * M_PI * i / Nk; };

    // Fill energy arrays
    for (int ix = 0; ix < Nk; ++ix) {
        for (int iy = 0; iy < Nk; ++iy) {
            double kx = kval(ix), ky = kval(iy);
            double dz = m + t * (std::cos(kx) + std::cos(ky));
            double dx = t * std::sin(kx), dy = t * std::sin(ky);
            double E  = band_energy(dz, dx, dy);
            lower[idx(ix, iy)] = -E;
            upper[idx(ix, iy)] = +E;
            if (2.0 * E < min_gap) min_gap = 2.0 * E;
        }
    }
    gap   = (min_gap < 1e29) ? min_gap : 0.0;
    chern = 0.0;

    // FHS plaquette summation
    // F(plaquette at ix,iy) = log[ U1(ix,iy) * U2(ix+1,iy) * conj(U1(ix,iy+1)) * conj(U2(ix,iy)) ]
    // where U_μ(k) = ⟨u(k)|u(k+eμ)⟩ / |⟨u(k)|u(k+eμ)⟩|
    for (int ix = 0; ix < Nk; ++ix) {
        for (int iy = 0; iy < Nk; ++iy) {
            auto evec = [&](int jx, int jy) -> Vec2 {
                double kx = kval(jx % Nk), ky = kval(jy % Nk);
                return lower_evec(m + t * (cos(kx) + cos(ky)),
                                  t * sin(kx), t * sin(ky));
            };
            Vec2 u00 = evec(ix,   iy  );
            Vec2 u10 = evec(ix+1, iy  );
            Vec2 u01 = evec(ix,   iy+1);
            Vec2 u11 = evec(ix+1, iy+1);

            auto link = [](cx z) -> cx {
                double r = std::abs(z);
                return (r > 1e-15) ? z / r : cx(1.0, 0.0);
            };
            cx U1_00 = link(dot2(u00, u10));
            cx U2_10 = link(dot2(u10, u11));
            cx U1_01 = link(dot2(u01, u11));   // U_x at (ix, iy+1)
            cx U2_00 = link(dot2(u00, u01));   // U_y at (ix, iy)

            cx F = std::log(U1_00 * U2_10 * std::conj(U1_01) * std::conj(U2_00));
            double omega = F.imag();
            berry[idx(ix, iy)] = omega;
            chern += omega;
        }
    }
    chern /= (2.0 * M_PI);
    // Normalise berry curvature to units of 2π / (BZ area)
    double dA = (2.0 * M_PI / Nk) * (2.0 * M_PI / Nk);
    for (auto& b : berry) b /= dA;  // Ω in rad/k² units
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  Strip geometry eigensolver (real symmetric, existing Householder + QR)
//     Open BC in y (Ny sites), periodic in x → Bloch Hamiltonian H(kx)
//
// The QWZ real-space Hamiltonian in the strip basis |j,σ⟩ (site j, orbital σ):
//   On-site:   H_onsite(kx) = (m + cos(kx)) σz + sin(kx) σx   [real symmetric]
//   y-hopping: t_y  = σz/2 − iσy/2 = [[½,−½],[½,−½]]          [real, NOT symmetric]
//              t_y† = σz/2 + iσy/2 = [[½, ½],[−½,−½]]
//
// The full (2Ny)×(2Ny) block matrix is real symmetric:
//   H_strip[j,j]     = H_onsite(kx)   (2×2 diagonal block)
//   H_strip[j,j+1]   = t_y             (off-diagonal)
//   H_strip[j+1,j]   = t_y^T           (transpose, = (t_y†)^T)
// ─────────────────────────────────────────────────────────────────────────────

using MatR = std::vector<double>;

static MatR build_strip(int Ny, double m, double kx, double t) {
    int dim = 2 * Ny;
    MatR H(static_cast<std::size_t>(dim) * dim, 0.0);

    // On-site block: (m + t*cos(kx)) σz + t*sin(kx) σx
    double c = m + t * std::cos(kx), s = t * std::sin(kx);
    for (int j = 0; j < Ny; ++j) {
        int b = j * 2;
        H[b * dim + b]           +=  c;   // σz[0,0]
        H[(b+1)*dim + b+1]       += -c;   // σz[1,1]
        H[b * dim + b+1]         +=  s;   // σx[0,1]
        H[(b+1)*dim + b]         +=  s;   // σx[1,0]
    }
    // y-hopping: t * [[½,−½],[½,−½]] at (j,j+1); transpose at (j+1,j)
    double th = 0.5 * t;
    for (int j = 0; j < Ny - 1; ++j) {
        int b = j * 2, bp = (j + 1) * 2;
        // t_y  at rows=b,b+1, cols=bp,bp+1
        H[b    * dim + bp  ] += +th;
        H[b    * dim + bp+1] += -th;
        H[(b+1)*dim + bp  ] += +th;
        H[(b+1)*dim + bp+1] += -th;
        // t_y^T at rows=bp,bp+1, cols=b,b+1
        H[bp    * dim + b  ] += +th;
        H[bp    * dim + b+1] += +th;
        H[(bp+1)*dim + b   ] += -th;
        H[(bp+1)*dim + b+1 ] += -th;
    }
    return H;
}

// Householder tridiagonalisation with optional reflector accumulation
static void householder_tridiag(const MatR& A, std::vector<double>& alpha,
                                std::vector<double>& beta, int n,
                                MatR* V_out = nullptr) {
    alpha.resize(n); beta.resize(n - 1);
    MatR Q = A;
    MatR V;
    if (V_out) {
        V.assign(static_cast<std::size_t>(n) * n, 0.0);
        for (int i = 0; i < n; ++i) V[i * n + i] = 1.0;
    }
    for (int k = 0; k < n - 2; ++k) {
        double norm2 = 0.0;
        for (int i = k + 1; i < n; ++i) norm2 += Q[i * n + k] * Q[i * n + k];
        double norm = std::sqrt(norm2);
        if (norm < 1e-15) continue;
        std::vector<double> v(n, 0.0);
        for (int i = k + 1; i < n; ++i) v[i] = Q[i * n + k];
        v[k + 1] += (v[k + 1] >= 0.0 ? 1.0 : -1.0) * norm;
        double vn2 = 0.0;
        for (int i = k + 1; i < n; ++i) vn2 += v[i] * v[i];
        if (vn2 < 1e-30) continue;
        std::vector<double> p(n, 0.0);
        for (int r = 0; r < n; ++r)
            for (int c = k + 1; c < n; ++c) p[r] += Q[r * n + c] * v[c];
        for (int r = 0; r < n; ++r) p[r] /= (vn2 / 2.0);
        double vTp = 0.0;
        for (int i = k + 1; i < n; ++i) vTp += v[i] * p[i];
        std::vector<double> w(n);
        for (int i = 0; i < n; ++i) w[i] = p[i] - (vTp / vn2) * v[i];
        for (int r = 0; r < n; ++r)
            for (int c = 0; c < n; ++c)
                Q[r * n + c] -= v[r] * w[c] + w[r] * v[c];
        if (V_out) {
            std::vector<double> Vv(n, 0.0);
            for (int row = 0; row < n; ++row)
                for (int j = k + 1; j < n; ++j) Vv[row] += V[row * n + j] * v[j];
            for (int row = 0; row < n; ++row)
                for (int c2 = k + 1; c2 < n; ++c2)
                    V[row * n + c2] -= (2.0 / vn2) * Vv[row] * v[c2];
        }
    }
    for (int i = 0; i < n; ++i) alpha[i] = Q[i * n + i];
    for (int i = 0; i < n - 1; ++i) beta[i] = Q[(i + 1) * n + i];
    if (V_out) *V_out = V;
}

// Wilkinson QR tridiagonal eigensolver (eigenvalues only)
static std::vector<double> tridiag_evals(std::vector<double> alpha,
                                          std::vector<double> beta) {
    int n = static_cast<int>(alpha.size());
    for (int end = n - 1; end > 0; ) {
        for (int i = 0; i < end; ++i)
            if (std::abs(beta[i]) < 1e-14 * (std::abs(alpha[i]) + std::abs(alpha[i+1])))
                beta[i] = 0.0;
        int start = end;
        while (start > 0 && beta[start - 1] != 0.0) --start;
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
// D.  Strip spectrum over kx scan
// ─────────────────────────────────────────────────────────────────────────────

static void solve_strip_scan(int Ny, int Nk, double m, double t,
                              std::vector<double>& kx_vals,
                              std::vector<std::vector<double>>& spectra) {
    int dim = 2 * Ny;
    kx_vals.resize(Nk);
    spectra.resize(Nk);
    for (int ik = 0; ik < Nk; ++ik) {
        double kx    = 2.0 * M_PI * ik / (Nk - 1);
        kx_vals[ik]  = kx / (2.0 * M_PI);    // store as fraction of 2π
        MatR H       = build_strip(Ny, m, kx, t);
        std::vector<double> alpha, beta;
        householder_tridiag(H, alpha, beta, dim);
        spectra[ik]  = tridiag_evals(alpha, beta);
    }
}

}  // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// E.  Public entry point
// ─────────────────────────────────────────────────────────────────────────────

ChernResult run_chern(const ChernOptions& opts) {
    auto t0 = std::chrono::steady_clock::now();

    if (opts.n_sites < 2 || opts.n_sites > 200)
        throw std::runtime_error("Chern: n_sites must be in [2, 200]");
    if (opts.t <= 0.0)
        throw std::runtime_error("Chern: t must be positive");
    if (opts.n_k < 4)
        throw std::runtime_error("Chern: n_k must be >= 4");

    ChernResult res;
    res.n_sites  = opts.n_sites;
    res.m        = opts.m;
    res.t        = opts.t;
    res.scenario = opts.scenario;

    auto do_bands = [&](double m_val) {
        compute_berry(opts.n_k, m_val, opts.t,
                      res.lower_band, res.upper_band, res.berry_curv,
                      res.chern, res.gap);
    };

    if (opts.scenario == "bands") {
        do_bands(opts.m);
    }

    if (opts.scenario == "edge") {
        solve_strip_scan(opts.n_sites, opts.n_k, opts.m, opts.t,
                         res.kx_vals, res.edge_spectrum);
        // Also compute Chern number for the badge
        compute_berry(opts.n_k, opts.m, opts.t,
                      res.lower_band, res.upper_band, res.berry_curv,
                      res.chern, res.gap);
    }

    if (opts.scenario == "sweep") {
        if (opts.n_sweep < 2)
            throw std::runtime_error("Chern: n_sweep must be >= 2");
        res.sweep.reserve(opts.n_sweep);
        for (int k = 0; k < opts.n_sweep; ++k) {
            double mk = opts.m_min + (opts.m_max - opts.m_min) * k / (opts.n_sweep - 1);
            ChernResult tmp;
            compute_berry(opts.n_k, mk, opts.t,
                          tmp.lower_band, tmp.upper_band, tmp.berry_curv,
                          tmp.chern, tmp.gap);
            ChernSweepPoint pt;
            pt.m     = mk;
            pt.gap   = tmp.gap;
            pt.chern = std::round(tmp.chern);
            res.sweep.push_back(pt);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    res.ok = true;
    return res;
}

}  // namespace qx
