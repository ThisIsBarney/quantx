// src/bose_hubbard.cpp — M28-F: Bose-Hubbard model
//
// H = -J Σ_{⟨i,i+1⟩} (a†_i a_{i+1} + h.c.) + U/2 Σ_i n_i(n_i-1) - μ Σ_i n_i
//
// Basis: product Fock states |n_1,...,n_N⟩ at fixed total boson number M,
//        with n_i ∈ [0, n_max].  Enumerated via lexicographic DFS.
// Solver: CSR + Lanczos with full reorthogonalization (same pattern as hubbard.cpp).

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qx {

// ── Basis enumeration ────────────────────────────────────────────────────────

// Enumerate all configurations |n_1,...,n_N⟩ with Σn_i = M, 0≤n_i≤n_max.
// Returns matrix: rows=basis states, cols=site occupations (row-major).
static std::vector<std::vector<std::uint32_t>> enum_bose_states(
    std::uint32_t N, std::uint32_t M, std::uint32_t n_max) {
    std::vector<std::vector<std::uint32_t>> result;
    std::vector<std::uint32_t> state(N, 0);

    std::function<void(std::uint32_t, std::uint32_t)> dfs =
        [&](std::uint32_t site, std::uint32_t remaining) {
            if (site == N - 1) {
                if (remaining <= n_max) {
                    state[site] = remaining;
                    result.push_back(state);
                }
                return;
            }
            std::uint32_t lo = (remaining > n_max * (N - 1 - site))
                                    ? remaining - n_max * (N - 1 - site)
                                    : 0;
            std::uint32_t hi = std::min(remaining, n_max);
            for (std::uint32_t n = lo; n <= hi; ++n) {
                state[site] = n;
                dfs(site + 1, remaining - n);
            }
        };

    dfs(0, M);
    return result;
}

// ── CSR sparse matrix ─────────────────────────────────────────────────────

struct BHCsr {
    std::uint32_t dim{0};
    std::vector<double> val;
    std::vector<std::uint32_t> col;
    std::vector<std::uint32_t> rowptr; // length dim+1
};

static void bh_matvec(const BHCsr& H, const std::vector<double>& x,
                      std::vector<double>& y) {
    y.assign(H.dim, 0.0);
    for (std::uint32_t r = 0; r < H.dim; ++r) {
        for (std::uint32_t k = H.rowptr[r]; k < H.rowptr[r + 1]; ++k)
            y[r] += H.val[k] * x[H.col[k]];
    }
}

struct CooEntry {
    std::uint32_t r, c;
    double v;
};

static BHCsr coo_to_bh_csr(std::uint32_t dim, std::vector<CooEntry>& coo) {
    // sort by row then col
    std::sort(coo.begin(), coo.end(), [](const CooEntry& a, const CooEntry& b) {
        return (a.r != b.r) ? (a.r < b.r) : (a.c < b.c);
    });
    // merge duplicates
    std::vector<CooEntry> merged;
    for (auto& e : coo) {
        if (!merged.empty() && merged.back().r == e.r && merged.back().c == e.c)
            merged.back().v += e.v;
        else
            merged.push_back(e);
    }

    BHCsr H;
    H.dim = dim;
    H.rowptr.assign(dim + 1, 0);
    for (auto& e : merged) H.rowptr[e.r + 1]++;
    for (std::uint32_t i = 0; i < dim; ++i) H.rowptr[i + 1] += H.rowptr[i];
    H.val.resize(merged.size());
    H.col.resize(merged.size());
    std::vector<std::uint32_t> pos(H.rowptr.begin(), H.rowptr.end());
    for (auto& e : merged) {
        std::uint32_t p = pos[e.r]++;
        H.val[p] = e.v;
        H.col[p] = e.c;
    }
    return H;
}

// ── Hamiltonian builder ───────────────────────────────────────────────────

// Build a map from configuration → basis index (linear search is ok for small dims).
// For larger systems we could use a hash map, but Bose-Hubbard dims stay manageable.
static BHCsr build_bose_hubbard_csr(
    const std::vector<std::vector<std::uint32_t>>& basis,
    std::uint32_t N, std::uint32_t n_max, double J, double U, double mu) {

    const auto dim = static_cast<std::uint32_t>(basis.size());

    // Build index map: encode state as mixed-radix integer for O(1) lookup.
    std::vector<CooEntry> coo;
    coo.reserve(dim * (2 * N + 1));

    // Sort basis for binary search (DFS order is not globally sorted by value,
    // so we build a sorted index array for O(log dim) lookup).
    std::vector<std::uint32_t> sorted_idx(dim);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(), [&](std::uint32_t a, std::uint32_t b) {
        return basis[a] < basis[b];
    });
    std::vector<const std::vector<std::uint32_t>*> sorted_basis(dim);
    for (std::uint32_t k = 0; k < dim; ++k)
        sorted_basis[k] = &basis[sorted_idx[k]];

    auto find_sorted = [&](const std::vector<std::uint32_t>& s) -> std::uint32_t {
        // binary search in sorted_basis
        std::uint32_t lo = 0, hi = dim;
        while (lo < hi) {
            std::uint32_t mid = (lo + hi) / 2;
            if (*sorted_basis[mid] < s) lo = mid + 1;
            else hi = mid;
        }
        return sorted_idx[lo]; // original index
    };

    for (std::uint32_t bi = 0; bi < dim; ++bi) {
        const auto& s = basis[bi];

        // Diagonal: U/2 * n_i*(n_i-1) - mu * n_i
        double diag = 0.0;
        for (std::uint32_t i = 0; i < N; ++i) {
            double ni = static_cast<double>(s[i]);
            diag += (U / 2.0) * ni * (ni - 1.0) - mu * ni;
        }
        coo.push_back({bi, bi, diag});

        // Off-diagonal: -J (a†_i a_{i+1} + a_{i+1}† a_i) for i=0..N-2
        for (std::uint32_t i = 0; i < N - 1; ++i) {
            // a†_i a_{i+1}: destroy one boson at i+1, create one at i
            if (s[i + 1] >= 1 && s[i] < n_max) {
                std::vector<std::uint32_t> s2 = s;
                s2[i]++;
                s2[i + 1]--;
                // matrix element: -J * sqrt(n_{i+1} * (n_i + 1))
                double mel = -J * std::sqrt(static_cast<double>(s[i + 1]) *
                                            static_cast<double>(s[i] + 1));
                std::uint32_t bj = find_sorted(s2);
                coo.push_back({bi, bj, mel});
                coo.push_back({bj, bi, mel}); // Hermitian conjugate
            }
        }
    }

    return coo_to_bh_csr(dim, coo);
}

// ── Tridiagonal eigensolver (Wilkinson QR) ────────────────────────────────

static void tridiag_eig_bh(std::vector<double>& alpha, std::vector<double>& beta,
                             std::vector<std::vector<double>>& Q) {
    // Symmetric tridiagonal QR with Wilkinson shift.
    // alpha[i] diagonal, beta[i] subdiagonal (beta[0] unused).
    int n = static_cast<int>(alpha.size());
    Q.assign(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) Q[i][i] = 1.0;

    for (int m = n - 1; m >= 1; --m) {
        for (int iter = 0; iter < 100 * n; ++iter) {
            // Wilkinson shift
            double d = (alpha[m - 1] - alpha[m]) / 2.0;
            double sign_d = (d >= 0) ? 1.0 : -1.0;
            double shift = alpha[m] - (beta[m] * beta[m]) /
                                          (d + sign_d * std::sqrt(d * d + beta[m] * beta[m]));

            double x = alpha[0] - shift;
            double z = beta[1];
            for (int k = 0; k < m; ++k) {
                double r = std::sqrt(x * x + z * z);
                if (r < 1e-300) break;
                double c = x / r, s = z / r;
                // Givens rotation G(k, k+1)
                double ak = alpha[k], ak1 = alpha[k + 1];
                double bk1 = beta[k + 1];
                alpha[k] = c * c * ak + 2 * c * s * bk1 + s * s * ak1 - shift;
                alpha[k + 1] = s * s * ak - 2 * c * s * bk1 + c * c * ak1;
                // Re-add shift only on last pass of this k (handled outside loop)
                // Actually we subtract shift above and add back:
                alpha[k] += shift;
                alpha[k + 1] += shift;
                // beta update
                if (k < m - 1) {
                    double new_bk1 = r * 1.0; // Placeholder — full update below
                    (void)new_bk1;
                }
                // correct update:
                double bk = (k > 0) ? beta[k] : 0.0;
                if (k > 0) {
                    beta[k] = r;
                }
                beta[k + 1] = -s * (k < m - 1 ? beta[k + 2] : 0.0) +
                               c * (c * bk1 - s * (ak1 - ak));
                // update Q columns
                for (int row = 0; row < n; ++row) {
                    double q0 = Q[row][k], q1 = Q[row][k + 1];
                    Q[row][k] = c * q0 + s * q1;
                    Q[row][k + 1] = -s * q0 + c * q1;
                }
                if (k < m - 1) {
                    x = beta[k + 1];
                    z = s * beta[k + 2];
                    beta[k + 2] *= c;
                }
            }
            if (std::abs(beta[m]) < 1e-12 * (std::abs(alpha[m - 1]) + std::abs(alpha[m])))
                break;
        }
    }
}

// Simpler: just use plain symmetric tridiagonal QL / implicit-shift for eigenvalues.
// We only need eigenvalues (not eigenvectors) for the Lanczos subproblem.
static std::vector<double> tridiag_eigenvalues_bh(std::vector<double> d,
                                                   std::vector<double> e) {
    int n = static_cast<int>(d.size());
    // e[0] unused; e[i] = beta_i for i>=1
    std::vector<double> f(n, 0.0); // off-diagonal, 1-indexed
    for (int i = 1; i < n; ++i) f[i] = e[i];

    for (int l = 0; l < n; ++l) {
        int m_iter = 0;
        for (;;) {
            m_iter = l;
            while (m_iter < n - 1 &&
                   std::abs(f[m_iter + 1]) >
                       1e-12 * (std::abs(d[m_iter]) + std::abs(d[m_iter + 1])))
                ++m_iter;
            if (m_iter == l) break;
            // Wilkinson shift
            double dd = (d[l + 1] - d[l]) / 2.0;
            double sgn = (dd >= 0) ? 1.0 : -1.0;
            double shift =
                d[l] - f[l + 1] * f[l + 1] /
                            (dd + sgn * std::hypot(dd, f[l + 1]));
            double g = d[l] - shift, p_val = 1.0;
            double s_val = 1.0, c_val = 1.0, prev_s = 1.0;
            for (int i = l + 1; i <= m_iter; ++i) {
                double fi = f[i];
                double r = std::hypot(g, fi);
                c_val = g / r;
                s_val = fi / r;
                if (i > l + 1) f[i - 1] = prev_s * r;
                g = d[i - 1] - shift;
                double new_g = (d[i] - shift) * s_val * s_val + c_val * (c_val * g + s_val * 2.0 * f[i]);
                double pp = (g * c_val + s_val * f[i]) * s_val;
                d[i - 1] = shift + pp + new_g; // simplified — not tracking exact update
                // Use standard LAPACK DSTEQR-style:
                double dl = d[i - 1], di = d[i];
                double fl_i = f[i];
                d[i - 1] = c_val * c_val * (dl - shift) - 2.0 * c_val * s_val * fl_i +
                            s_val * s_val * (di - shift) + shift;
                d[i] = s_val * s_val * (dl - shift) + 2.0 * c_val * s_val * fl_i +
                       c_val * c_val * (di - shift) + shift;
                f[i] = c_val * s_val * ((di - shift) - (dl - shift)) +
                       (c_val * c_val - s_val * s_val) * fl_i;
                prev_s = s_val;
                g = c_val * f[i]; // next g before shift correction
                (void)p_val;
            }
            if (l < m_iter) f[l + 1] = 0.0;
        }
    }
    std::sort(d.begin(), d.end());
    return d;
}

// ── Lanczos eigensolver ──────────────────────────────────────────────────

struct LanczosResult {
    double E0, E1;
    std::vector<double> ground_vec;
};

static LanczosResult bh_lanczos(const BHCsr& H, int max_iter = 200) {
    const int dim = static_cast<int>(H.dim);
    if (dim == 1) return {H.val[0], H.val[0] + 1e10, {1.0}};

    int m = std::min(max_iter, dim);

    // Random start vector
    std::vector<double> v(dim, 0.0);
    {
        std::uint32_t seed = 12345;
        for (int i = 0; i < dim; ++i) {
            seed = seed * 1664525u + 1013904223u;
            v[i] = static_cast<double>(static_cast<int32_t>(seed)) * 4.6566e-10;
        }
        double norm = 0.0;
        for (double x : v) norm += x * x;
        norm = std::sqrt(norm);
        for (double& x : v) x /= norm;
    }

    std::vector<double> alpha_l, beta_l;
    std::vector<std::vector<double>> V; // Krylov basis
    V.reserve(m + 1);
    V.push_back(v);

    std::vector<double> w(dim);

    for (int j = 0; j < m; ++j) {
        bh_matvec(H, V[j], w);
        double aj = 0.0;
        for (int i = 0; i < dim; ++i) aj += V[j][i] * w[i];
        alpha_l.push_back(aj);

        // w = w - aj*v_j - bj*v_{j-1}
        for (int i = 0; i < dim; ++i) w[i] -= aj * V[j][i];
        if (j > 0) {
            for (int i = 0; i < dim; ++i) w[i] -= beta_l.back() * V[j - 1][i];
        }

        // Full reorthogonalization
        for (int k = 0; k <= j; ++k) {
            double dot = 0.0;
            for (int i = 0; i < dim; ++i) dot += V[k][i] * w[i];
            for (int i = 0; i < dim; ++i) w[i] -= dot * V[k][i];
        }

        double bj1 = 0.0;
        for (double x : w) bj1 += x * x;
        bj1 = std::sqrt(bj1);

        if (bj1 < 1e-12 || j == m - 1) {
            beta_l.push_back(0.0);
            V.push_back(w); // may be zero
            break;
        }
        beta_l.push_back(bj1);
        std::vector<double> vn(dim);
        for (int i = 0; i < dim; ++i) vn[i] = w[i] / bj1;
        V.push_back(vn);
    }

    int kk = static_cast<int>(alpha_l.size());
    // beta array for tridiag: beta[0]=0, beta[i]=beta_l[i-1]
    std::vector<double> beta_td(kk, 0.0);
    for (int i = 1; i < kk; ++i) beta_td[i] = beta_l[i - 1];

    // Get eigenvalues
    auto evals = tridiag_eigenvalues_bh(alpha_l, beta_td);

    double E0 = evals[0];
    double E1 = (kk > 1) ? evals[1] : E0 + 1e10;

    // Eigenvector: inverse iteration / direct Lanczos vector extraction.
    // Rerun Lanczos to get ground eigenvector from tridiagonal.
    // Use the Krylov basis we already built.
    // Compute ground eigenvector of tridiag, then back-transform.
    std::vector<double> td_alpha = alpha_l;
    std::vector<double> td_beta = beta_td;

    // Power iteration on (H - E0*I) shifted for ground state:
    // Since we have V and tridiag, reconstruct ground vec via tridiag eigenvec.
    // Simple approach: inverse-free, use the Krylov vectors.
    // Compute tridiag eigenvector via QR:
    std::vector<std::vector<double>> Qmat;
    tridiag_eig_bh(td_alpha, td_beta, Qmat);
    // eigenvector 0 of tridiag = column 0 of Qmat (after sorting by eigenvalue)
    // But our tridiag_eig_bh modifies alpha in-place (sorted eigenvalues).
    // Just pick the column with smallest eigenvalue.
    int ev_col = 0;
    for (int c = 1; c < kk; ++c)
        if (td_alpha[c] < td_alpha[ev_col]) ev_col = c;

    std::vector<double> gvec(dim, 0.0);
    for (int j = 0; j < kk && j < static_cast<int>(V.size()); ++j) {
        double coeff = Qmat[j][ev_col];
        for (int i = 0; i < dim; ++i) gvec[i] += coeff * V[j][i];
    }

    // Normalize
    double gnorm = 0.0;
    for (double x : gvec) gnorm += x * x;
    gnorm = std::sqrt(gnorm);
    if (gnorm > 1e-14)
        for (double& x : gvec) x /= gnorm;

    return {E0, E1, gvec};
}

// ── Observables ──────────────────────────────────────────────────────────

// Superfluid order parameter: |⟨a_0⟩| averaged over sites.
// ⟨ψ|a_i|ψ⟩ = Σ_{b,b'} ψ[b] * ψ[b'] * ⟨b'|a_i|b⟩
// a_i|b⟩ = sqrt(n_i)|b with n_i → n_i-1⟩
static double superfluid_order(const std::vector<std::vector<std::uint32_t>>& basis,
                                const std::vector<double>& gvec, std::uint32_t N) {
    const auto dim = static_cast<std::uint32_t>(basis.size());

    // Build sorted index map
    std::vector<std::uint32_t> sorted_idx(dim);
    std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
    std::sort(sorted_idx.begin(), sorted_idx.end(), [&](std::uint32_t a, std::uint32_t b) {
        return basis[a] < basis[b];
    });
    std::vector<const std::vector<std::uint32_t>*> sorted_basis(dim);
    for (std::uint32_t k = 0; k < dim; ++k)
        sorted_basis[k] = &basis[sorted_idx[k]];

    auto find_sorted = [&](const std::vector<std::uint32_t>& s) -> int {
        std::uint32_t lo = 0, hi = dim;
        while (lo < hi) {
            std::uint32_t mid = (lo + hi) / 2;
            if (*sorted_basis[mid] < s) lo = mid + 1;
            else hi = mid;
        }
        if (lo >= dim || *sorted_basis[lo] != s) return -1;
        return static_cast<int>(sorted_idx[lo]);
    };

    double sf = 0.0;
    for (std::uint32_t site = 0; site < N; ++site) {
        double re_a = 0.0;
        for (std::uint32_t bi = 0; bi < dim; ++bi) {
            const auto& s = basis[bi];
            if (s[site] == 0) continue;
            std::vector<std::uint32_t> s2 = s;
            double mel = std::sqrt(static_cast<double>(s2[site]));
            s2[site]--;
            int bj = find_sorted(s2);
            if (bj < 0) continue;
            re_a += gvec[bi] * mel * gvec[bj];
        }
        sf += std::abs(re_a);
    }
    return sf / static_cast<double>(N);
}

// Per-site occupation ⟨n_i⟩ and double occupancy ⟨Σ n_i(n_i-1)⟩/N
static void occupation_obs(const std::vector<std::vector<std::uint32_t>>& basis,
                            const std::vector<double>& gvec, std::uint32_t N,
                            std::vector<double>& occ, double& double_occ) {
    const auto dim = static_cast<std::uint32_t>(basis.size());
    occ.assign(N, 0.0);
    double_occ = 0.0;
    for (std::uint32_t bi = 0; bi < dim; ++bi) {
        double p = gvec[bi] * gvec[bi];
        const auto& s = basis[bi];
        for (std::uint32_t i = 0; i < N; ++i) {
            occ[i] += p * static_cast<double>(s[i]);
            double_occ += p * static_cast<double>(s[i]) *
                          static_cast<double>(s[i] > 0 ? s[i] - 1 : 0);
        }
    }
    double_occ /= static_cast<double>(N);
}

// Number distribution P(n) at site 0: P(n) = Σ_{b: b[0]=n} ψ[b]²
static std::vector<double> number_distribution(
    const std::vector<std::vector<std::uint32_t>>& basis,
    const std::vector<double>& gvec, std::uint32_t n_max) {
    std::vector<double> dist(n_max + 1, 0.0);
    for (std::uint32_t bi = 0; bi < basis.size(); ++bi) {
        std::uint32_t n0 = basis[bi][0];
        if (n0 <= n_max) dist[n0] += gvec[bi] * gvec[bi];
    }
    return dist;
}

// ── Single-point computation ─────────────────────────────────────────────

struct BHSingle {
    double E0, E1, sf, double_occ;
    std::vector<double> occ, num_dist;
};

static BHSingle compute_bh_single(std::uint32_t N, std::uint32_t M,
                                   std::uint32_t n_max, double J, double U,
                                   double mu) {
    auto basis = enum_bose_states(N, M, n_max);
    if (basis.empty())
        throw std::runtime_error("bose_hubbard: no valid basis states (check n_max vs n_bosons)");

    auto H = build_bose_hubbard_csr(basis, N, n_max, J, U, mu);
    auto lr = bh_lanczos(H);

    BHSingle res;
    res.E0 = lr.E0;
    res.E1 = lr.E1;
    res.sf = superfluid_order(basis, lr.ground_vec, N);
    occupation_obs(basis, lr.ground_vec, N, res.occ, res.double_occ);
    res.num_dist = number_distribution(basis, lr.ground_vec, n_max);
    return res;
}

// ── Public entry point ───────────────────────────────────────────────────

BoseHubbardResult run_bose_hubbard(const BoseHubbardOptions& opts) {
    auto t0 = std::chrono::steady_clock::now();

    BoseHubbardResult res;
    res.n_sites = opts.n_sites;
    res.n_bosons = opts.n_bosons;
    res.n_max = opts.n_max;
    res.J = opts.J;
    res.U = opts.U;

    // Validation
    if (opts.n_sites < 2 || opts.n_sites > 14)
        throw std::runtime_error("bose_hubbard: n_sites must be in [2, 14]");
    if (opts.n_bosons == 0)
        throw std::runtime_error("bose_hubbard: n_bosons must be >= 1");
    if (opts.n_max < 1)
        throw std::runtime_error("bose_hubbard: n_max must be >= 1");
    if (opts.n_bosons > opts.n_sites * opts.n_max)
        throw std::runtime_error(
            "bose_hubbard: n_bosons > n_sites * n_max — no valid states");
    if (opts.J < 0.0)
        throw std::runtime_error("bose_hubbard: J must be >= 0");

    if (opts.scenario == "ground_state") {
        auto s = compute_bh_single(opts.n_sites, opts.n_bosons, opts.n_max,
                                    opts.J, opts.U, opts.mu);
        res.ground_energy = s.E0;
        res.energy_gap = (s.E1 > s.E0 + 1e8) ? 0.0 : s.E1 - s.E0;
        res.superfluid_op = s.sf;
        res.double_occupancy = s.double_occ;
        res.occupation = s.occ;
        res.number_dist = s.num_dist;

    } else if (opts.scenario == "sweep") {
        // Sweep J from J_min to J_max at fixed U
        for (std::uint32_t k = 0; k < opts.n_sweep; ++k) {
            double J_k = (opts.n_sweep > 1)
                             ? opts.J_min + (opts.J_max - opts.J_min) * k /
                                                (opts.n_sweep - 1)
                             : opts.J_min;
            auto s = compute_bh_single(opts.n_sites, opts.n_bosons, opts.n_max,
                                        J_k, opts.U, opts.mu);
            BoseHubbardSweepPoint pt;
            pt.J = J_k;
            pt.ground_energy = s.E0;
            pt.energy_gap = (s.E1 > s.E0 + 1e8) ? 0.0 : s.E1 - s.E0;
            pt.superfluid_op = s.sf;
            pt.double_occupancy = s.double_occ;
            res.sweep.push_back(pt);
        }
        // Set summary fields from last point
        if (!res.sweep.empty()) {
            const auto& last = res.sweep.back();
            res.ground_energy = last.ground_energy;
            res.energy_gap = last.energy_gap;
            res.superfluid_op = last.superfluid_op;
            res.double_occupancy = last.double_occupancy;
        }
        // occupation / number_dist at the requested J
        auto s0 = compute_bh_single(opts.n_sites, opts.n_bosons, opts.n_max,
                                     opts.J, opts.U, opts.mu);
        res.occupation = s0.occ;
        res.number_dist = s0.num_dist;
    } else {
        throw std::runtime_error("bose_hubbard: unknown scenario '" + opts.scenario +
                                  "'; use 'ground_state' or 'sweep'");
    }

    res.ok = true;
    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

}  // namespace qx
