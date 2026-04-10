// src/luttinger.cpp  —  M52: Luttinger Liquid & Bosonization
//
// XXZ spin-1/2 chain at half-filling (S_z=0 sector):
//   H = J Σ_i [1/2(S+_i S-_{i+1} + h.c.) + Δ Sz_i Sz_{i+1}]  (PBC)
//
// For |Δ| < 1 the model is in the Tomonaga-Luttinger liquid phase.
// Bethe-ansatz result: Luttinger parameter K(Δ) = π / (2 arccos(-Δ))
//   K > 1 (Δ < 0):  dominant superfluid correlations
//   K = 1 (Δ = 0):  free-fermion XX chain
//   K < 1 (Δ > 0):  dominant CDW (Néel) correlations
//   Δ → ±1:         BKT transitions to gapped phases
//
// Key observables:
//   E_0, E_gap  (sector S_z=0, PBC)
//   Spin structure factor S(q) = (1/L) Σ_{ij} e^{iq(i-j)} ⟨Sz_i Sz_j⟩
//   K_analytic(Δ) = π / (2 arccos(-Δ))
//
// Scenarios: "delta" | "structure" | "both"

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

// ── Portable popcount ─────────────────────────────────────────────────────────
static int popcount_bits_ll(unsigned v) {
    v = v - ((v >> 1) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
    return static_cast<int>(((v + (v >> 4)) & 0x0f0f0f0fu) * 0x01010101u >> 24);
}

// ── Jacobi diagonalisation (eigenvalues only, in-place) ───────────────────────
static void jacobi_evals_ll(std::vector<double>& A, int n) {
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i+1; j < n; ++j) { double v = A[i*n+j]; off += v*v; }
        if (off < 1e-26) break;
        for (int p = 0; p < n-1; ++p) for (int q = p+1; q < n; ++q) {
            const double apq = A[p*n+q]; if (std::abs(apq) < 1e-15) continue;
            const double app = A[p*n+p], aqq = A[q*n+q];
            const double theta = 0.5*(aqq-app)/apq;
            const double t = (theta>=0?1.0:-1.0)/(std::abs(theta)+std::sqrt(theta*theta+1.0));
            const double c = 1.0/std::sqrt(t*t+1.0), s = t*c;
            A[p*n+p]=app-t*apq; A[q*n+q]=aqq+t*apq; A[p*n+q]=A[q*n+p]=0.0;
            for (int r=0; r<n; ++r) {
                if(r==p||r==q) continue;
                const double arp=A[r*n+p], arq=A[r*n+q];
                A[r*n+p]=A[p*n+r]=c*arp-s*arq;
                A[r*n+q]=A[q*n+r]=s*arp+c*arq;
            }
        }
    }
}

// Jacobi with eigenvectors
static void jacobi_full_ll(std::vector<double>& A, std::vector<double>& V, int n) {
    V.assign(static_cast<size_t>(n)*n, 0.0);
    for (int i=0; i<n; ++i) V[i*n+i]=1.0;
    for (int sweep=0; sweep<100; ++sweep) {
        double off=0.0;
        for (int i=0;i<n;++i) for (int j=i+1;j<n;++j) { double v=A[i*n+j]; off+=v*v; }
        if (off<1e-26) break;
        for (int p=0;p<n-1;++p) for (int q=p+1;q<n;++q) {
            const double apq=A[p*n+q]; if(std::abs(apq)<1e-15) continue;
            const double app=A[p*n+p], aqq=A[q*n+q];
            const double theta=0.5*(aqq-app)/apq;
            const double t=(theta>=0?1.0:-1.0)/(std::abs(theta)+std::sqrt(theta*theta+1.0));
            const double c=1.0/std::sqrt(t*t+1.0), s=t*c;
            A[p*n+p]=app-t*apq; A[q*n+q]=aqq+t*apq; A[p*n+q]=A[q*n+p]=0.0;
            for (int r=0;r<n;++r) {
                if(r==p||r==q) continue;
                const double arp=A[r*n+p], arq=A[r*n+q];
                A[r*n+p]=A[p*n+r]=c*arp-s*arq;
                A[r*n+q]=A[q*n+r]=s*arp+c*arq;
            }
            for (int r=0;r<n;++r) {
                const double vrp=V[r*n+p], vrq=V[r*n+q];
                V[r*n+p]=c*vrp-s*vrq; V[r*n+q]=s*vrp+c*vrq;
            }
        }
    }
}

// ── Enumerate S_z=0 basis states (half-filling) ──────────────────────────────
static std::vector<int> enum_sz0(int L) {
    std::vector<int> states;
    states.reserve(512);
    for (int s = 0; s < (1 << L); ++s)
        if (popcount_bits_ll(static_cast<unsigned>(s)) == L / 2)
            states.push_back(s);
    return states;
}

// ── Build XXZ Hamiltonian in S_z=0 sector ────────────────────────────────────
static std::vector<double> build_xxz(
    const std::vector<int>& states,
    const std::vector<int>& lut,
    int L, double J, double Delta)
{
    const int dim = static_cast<int>(states.size());
    std::vector<double> H(static_cast<size_t>(dim)*dim, 0.0);
    for (int a = 0; a < dim; ++a) {
        const int sa = states[a];
        // Diagonal: J*Delta * Σ_i Sz_i Sz_{i+1} = J*Delta/4 Σ_i (1-2*b_i)(1-2*b_{i+1})
        double diag = 0.0;
        for (int i = 0; i < L; ++i) {
            const int j = (i + 1) % L;  // PBC
            const int bi = (sa >> i) & 1;
            const int bj = (sa >> j) & 1;
            diag += J * Delta * 0.25 * (1 - 2*bi) * (1 - 2*bj);
        }
        H[a*dim+a] = diag;
        // Off-diagonal: J/2 * (S+_i S-_{i+1} + h.c.)
        // S+_i S-_{i+1}: bit_i = 1→0, bit_{i+1} = 0→1  (σ_i up, σ_{i+1} down → flip)
        // In our convention: |0⟩ = spin up (bit=0), |1⟩ = spin down (bit=1)
        // Wait: let's define bit=1 → spin up, bit=0 → spin down
        // S+ raises spin (0→1), S- lowers spin (1→0)
        // S+_i S-_{i+1}: bit_i = 0→1, bit_{i+1} = 1→0 (i.e., sb has bit_i=0, bit_{i+1}=1)
        for (int i = 0; i < L; ++i) {
            const int j = (i + 1) % L;
            const int bi = (sa >> i) & 1;
            const int bj = (sa >> j) & 1;
            if (bi == 0 && bj == 1) {
                // Flip: set bit i, clear bit j → new state
                const int sb = (sa | (1 << i)) & ~(1 << j);
                const int b = lut[sb];
                if (b >= 0) H[a*dim+b] += J * 0.5;
            }
            if (bi == 1 && bj == 0) {
                // Flip: clear bit i, set bit j
                const int sb = (sa & ~(1 << i)) | (1 << j);
                const int b = lut[sb];
                if (b >= 0) H[a*dim+b] += J * 0.5;
            }
        }
    }
    return H;
}

// ── Spin structure factor S(q) from ground-state vector ──────────────────────
static std::vector<double> structure_factor(
    const std::vector<double>& psi,
    const std::vector<int>& states,
    int L,
    const std::vector<double>& q_vals)
{
    const int dim = static_cast<int>(states.size());
    // Compute equal-time correlator C(r) = (1/L) Σ_i ⟨Sz_i Sz_{i+r}⟩
    std::vector<double> Cr(L, 0.0);
    for (int a = 0; a < dim; ++a) {
        const double w = psi[a] * psi[a];
        const int sa = states[a];
        for (int i = 0; i < L; ++i) {
            const double szi = 0.5 * (1 - 2 * ((sa >> i) & 1));
            for (int r = 0; r < L; ++r) {
                const int j = (i + r) % L;
                const double szj = 0.5 * (1 - 2 * ((sa >> j) & 1));
                Cr[r] += w * szi * szj;
            }
        }
    }
    for (auto& c : Cr) c /= L;
    // S(q) = Σ_r e^{iqr} C(r) = Σ_r cos(qr) C(r)  (real for PBC)
    std::vector<double> Sq(q_vals.size());
    for (std::size_t qi = 0; qi < q_vals.size(); ++qi) {
        double s = 0.0;
        for (int r = 0; r < L; ++r)
            s += std::cos(q_vals[qi] * r) * Cr[r];
        Sq[qi] = s;
    }
    return Sq;
}

// ── run_luttinger ─────────────────────────────────────────────────────────────
LLResult run_luttinger(const LLOptions& opts) {
    if (opts.L < 4 || opts.L % 2 != 0)
        throw std::runtime_error("LL: L must be an even integer >= 4");
    if (opts.L > 14)
        throw std::runtime_error("LL: L must be <= 14 (dim=C(14,7)=3432 is the safe dense-ED limit)");
    if (opts.n_delta < 1)
        throw std::runtime_error("LL: n_delta must be >= 1");
    const std::string& sc = opts.scenario;
    if (sc != "delta" && sc != "structure" && sc != "both")
        throw std::runtime_error("LL: unknown scenario; valid: delta, structure, both");

    auto t0 = std::chrono::steady_clock::now();
    LLResult res;
    res.scenario = sc; res.L = opts.L; res.Delta = opts.Delta; res.J = opts.J;

    const auto states = enum_sz0(opts.L);
    const int dim = static_cast<int>(states.size());

    // Precompute LUT
    std::vector<int> lut(1 << opts.L, -1);
    for (int i = 0; i < dim; ++i) lut[states[i]] = i;

    // q grid for structure factor
    const int nq = opts.L / 2 + 1;
    res.q_vals.resize(nq);
    for (int k = 0; k < nq; ++k)
        res.q_vals[k] = 2.0 * M_PI * k / opts.L;

    // ── Structure factor at fixed Delta ─────────────────────────────────────
    if (sc == "structure" || sc == "both") {
        auto H = build_xxz(states, lut, opts.L, opts.J, opts.Delta);
        std::vector<double> evecs;
        jacobi_full_ll(H, evecs, dim);
        // Find ground state
        std::vector<double> evals(dim);
        for (int i = 0; i < dim; ++i) evals[i] = H[i*dim+i];
        const int gs = static_cast<int>(
            std::min_element(evals.begin(), evals.end()) - evals.begin());
        res.E0 = evals[gs];
        // Second distinct eigenvalue for gap
        std::vector<double> sorted_e = evals;
        std::sort(sorted_e.begin(), sorted_e.end());
        res.E_gap = (sorted_e.size() > 1) ? sorted_e[1] - sorted_e[0] : 0.0;
        // Extract GS eigenvector
        std::vector<double> psi(dim);
        for (int r = 0; r < dim; ++r) psi[r] = evecs[r*dim+gs];
        res.Sq = structure_factor(psi, states, opts.L, res.q_vals);
        // Analytic K
        const double acos_arg = std::max(-1.0, std::min(1.0, -opts.Delta));
        res.K_analytic = (std::abs(opts.Delta) < 0.999)
            ? M_PI / (2.0 * std::acos(acos_arg))
            : (opts.Delta > 0 ? 0.5 : 5.0);  // saturate: FM→0.5, AFM→5.0 (matches sweep)
    }

    // ── Delta sweep ──────────────────────────────────────────────────────────
    if (sc == "delta" || sc == "both") {
        const int nd = opts.n_delta;
        res.delta_vals.resize(nd); res.E0_delta.resize(nd);
        res.gap_delta.resize(nd); res.K_delta.resize(nd);
        res.Sq_pi_delta.resize(nd);
        const std::vector<double> qpi_v = {M_PI};
        for (int di = 0; di < nd; ++di) {
            const double dv = (nd > 1)
                ? opts.delta_min + (opts.delta_max - opts.delta_min) * di / static_cast<double>(nd-1)
                : opts.Delta;
            res.delta_vals[di] = dv;
            auto Hd = build_xxz(states, lut, opts.L, opts.J, dv);
            std::vector<double> evecs_d;
            jacobi_full_ll(Hd, evecs_d, dim);
            std::vector<double> evd(dim);
            for (int i = 0; i < dim; ++i) evd[i] = Hd[i*dim+i];
            const int gs_d = static_cast<int>(
                std::min_element(evd.begin(), evd.end()) - evd.begin());
            res.E0_delta[di] = evd[gs_d];
            // First excitation energy (smallest eigenvalue above gs)
            std::vector<double> sorted_evd = evd;
            std::sort(sorted_evd.begin(), sorted_evd.end());
            res.gap_delta[di] = (dim > 1) ? sorted_evd[1] - sorted_evd[0] : 0.0;
            // Analytic K
            const double acos_arg = std::max(-1.0, std::min(1.0, -dv));
            res.K_delta[di] = (std::abs(dv) < 0.999)
                ? M_PI / (2.0 * std::acos(acos_arg))
                : (dv > 0 ? 0.5 : 5.0);
            // S(q=π) from GS eigenvector
            std::vector<double> psi_d(dim);
            for (int r = 0; r < dim; ++r) psi_d[r] = evecs_d[r*dim+gs_d];
            auto Sq_pi_v = structure_factor(psi_d, states, opts.L, qpi_v);
            res.Sq_pi_delta[di] = Sq_pi_v[0];
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    res.ok = true; res.message = "ok";
    return res;
}

}  // namespace qx
