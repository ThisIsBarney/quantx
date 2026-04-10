// src/mipt.cpp  —  M51: Measurement-Induced Phase Transitions (MIPT)
//
// 1D random quantum circuit: brickwork Haar-random 2-qubit gates +
// single-qubit projective measurements at rate p.
//
// State: full state vector |ψ⟩ ∈ (C²)^⊗L,  L ≤ 12 (dim = 4096)
//
// Circuit structure per step:
//   Even layer: apply random U ∈ U(4) to pairs (0,1),(2,3),...
//   Odd layer:  apply random U ∈ U(4) to pairs (1,2),(3,4),...
//   Measure:    each qubit independently with probability p →
//               collapse |ψ⟩ to eigenstate of σ_z^i
//
// Entanglement entropy S(L/2): SVD of ψ reshaped as 2^(L/2)×2^(L/2).
//
// Phases (Haar-random 2-qubit gates, 1+1D circuit):
//   p < p_c ≈ 0.16:  volume law  S ~ (L/2) ln 2
//   p > p_c:          area law    S ~ const
//   p ≈ p_c:          critical    S ~ (c/3) ln L  with c ≈ 0.5–1
//
// Scenarios: "sweep" | "scaling" | "both"

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
#include <complex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace qx {

using Cx = std::complex<double>;
using State = std::vector<Cx>;

// ── Haar-random 4×4 unitary via Gram-Schmidt on random complex vectors ────────
// Returns flat row-major 4×4 complex matrix U[16].
static std::vector<Cx> haar_u4(std::mt19937& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);
    // Generate 4 random complex vectors
    std::vector<Cx> cols(16);
    for (auto& c : cols) c = Cx(nd(rng), nd(rng));
    // Modified Gram-Schmidt
    for (int j = 0; j < 4; ++j) {
        // Subtract projections onto previous columns
        for (int k = 0; k < j; ++k) {
            Cx dot = Cx(0);
            for (int r = 0; r < 4; ++r) dot += std::conj(cols[r*4+k]) * cols[r*4+j];
            for (int r = 0; r < 4; ++r) cols[r*4+j] -= dot * cols[r*4+k];
        }
        // Normalize
        double nrm = 0.0;
        for (int r = 0; r < 4; ++r) nrm += std::norm(cols[r*4+j]);
        nrm = std::sqrt(nrm);
        if (nrm < 1e-14) {
            // Degenerate (extremely rare): return identity for this column
            for (int r = 0; r < 4; ++r) cols[r*4+j] = (r == j) ? Cx(1) : Cx(0);
        } else {
            for (int r = 0; r < 4; ++r) cols[r*4+j] /= nrm;
        }
    }
    // Transpose to row-major (U[i][j] = cols[j*4+i] → U[i*4+j])
    std::vector<Cx> U(16);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            U[i*4+j] = cols[j*4+i];
    return U;
}

// ── Apply 4×4 unitary U to sites (q0, q0+1) of L-qubit state ─────────────────
// Qubit ordering: bit 0 = least significant (site 0).
static void apply_u4(State& psi, int q0, int L, const std::vector<Cx>& U) {
    const int dim = 1 << L;
    // Iterate over all basis states; group by the rest of the bits (mask out q0,q0+1)
    for (int rest = 0; rest < (dim >> 2); ++rest) {
        // Reconstruct the 4 full indices by inserting 0b?? at positions q0,q0+1
        // Bits above q0+1: shift up by 2
        // rest bits: [high bits | low bits]  where low = bits 0..q0-1, high = bits q0..
        int lo = rest & ((1 << q0) - 1);
        int hi = rest >> q0;
        int base = lo | (hi << (q0 + 2));
        int idx[4];
        idx[0] = base | (0 << q0);
        idx[1] = base | (1 << q0);
        idx[2] = base | (2 << q0);
        idx[3] = base | (3 << q0);
        Cx v0 = psi[idx[0]], v1 = psi[idx[1]], v2 = psi[idx[2]], v3 = psi[idx[3]];
        for (int r = 0; r < 4; ++r) {
            psi[idx[r]] = U[r*4+0]*v0 + U[r*4+1]*v1 + U[r*4+2]*v2 + U[r*4+3]*v3;
        }
    }
}

// ── Projective measurement of qubit q with probability p ─────────────────────
static void maybe_measure(State& psi, int q, int L, double p, std::mt19937& rng) {
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    if (ud(rng) >= p) return;  // no measurement this step

    // Compute probability of outcome |1⟩ on qubit q
    double p1 = 0.0;
    for (int s = 0; s < (1 << L); ++s)
        if ((s >> q) & 1) p1 += std::norm(psi[s]);

    const int outcome = (ud(rng) < p1) ? 1 : 0;
    const double nrm = std::sqrt(outcome ? p1 : (1.0 - p1));
    if (nrm < 1e-14) return;  // shouldn't happen

    const int q_mask = 1 << q;
    for (int s = 0; s < (1 << L); ++s) {
        if (((s >> q) & 1) != outcome)
            psi[s] = Cx(0);
        else
            psi[s] /= nrm;
    }
}

// ── Entanglement entropy at half-cut via singular values of reshaped matrix ───
static double half_entropy(const State& psi, int L) {
    const int dA = 1 << (L / 2);
    const int dB = 1 << (L - L / 2);
    // M[a,b] = psi[a * dB + b]  but we need to reorder: site 0..L/2-1 form A
    // With qubit indexing: bit 0..L/2-1 form A, bit L/2..L-1 form B
    // psi[s]: s = (s_B << (L/2)) | s_A  (bit q0 = bit 0 → site 0)
    std::vector<double> MTM(static_cast<size_t>(dA)*dA, 0.0);
    // M_{a,b} = psi[b*dA + a]  (b = B bits, a = A bits)
    // M†M_{a,a'} = Σ_b M*_{b,a} M_{b,a'} = Σ_b conj(psi[b*dA+a]) * psi[b*dA+a']
    for (int a = 0; a < dA; ++a)
        for (int a2 = 0; a2 < dA; ++a2) {
            Cx sum = 0;
            for (int b = 0; b < dB; ++b)
                sum += std::conj(psi[b*dA+a]) * psi[b*dA+a2];
            MTM[a*dA+a2] = sum.real();
        }
    // Eigenvalues of M†M = squared singular values
    // Use simple power-iteration-free approach: just sum diagonal after "Jacobi"
    // For small dA (≤ 64), inline Jacobi
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < dA; ++i)
            for (int j = i+1; j < dA; ++j) { double v = MTM[i*dA+j]; off += v*v; }
        if (off < 1e-24) break;
        for (int p2 = 0; p2 < dA-1; ++p2) for (int q2 = p2+1; q2 < dA; ++q2) {
            const double apq = MTM[p2*dA+q2]; if (std::abs(apq) < 1e-15) continue;
            const double app = MTM[p2*dA+p2], aqq = MTM[q2*dA+q2];
            const double theta = 0.5*(aqq-app)/apq;
            const double t = (theta>=0?1.0:-1.0)/(std::abs(theta)+std::sqrt(theta*theta+1.0));
            const double c = 1.0/std::sqrt(t*t+1.0), s = t*c;
            MTM[p2*dA+p2] = app-t*apq; MTM[q2*dA+q2] = aqq+t*apq; MTM[p2*dA+q2]=MTM[q2*dA+p2]=0.0;
            for (int r = 0; r < dA; ++r) {
                if (r==p2||r==q2) continue;
                const double arp=MTM[r*dA+p2], arq=MTM[r*dA+q2];
                MTM[r*dA+p2]=MTM[p2*dA+r]=c*arp-s*arq;
                MTM[r*dA+q2]=MTM[q2*dA+r]=s*arp+c*arq;
            }
        }
    }
    double S = 0.0;
    for (int i = 0; i < dA; ++i) {
        const double sv2 = std::max(0.0, MTM[i*dA+i]);
        if (sv2 > 1e-14) S -= sv2 * std::log(sv2);
    }
    return S;
}

// ── Run one circuit trajectory ────────────────────────────────────────────────
static double run_circuit(int L, double p, int T, int seed) {
    std::mt19937 rng(static_cast<unsigned>(seed));
    const int dim = 1 << L;
    // Initial state: Neel-like product state |0101...⟩
    State psi(dim, Cx(0));
    int s0 = 0;
    for (int i = 1; i < L; i += 2) s0 |= (1 << i);  // bits 1,3,5... set
    psi[s0] = Cx(1.0);

    for (int t = 0; t < T; ++t) {
        // Even layer
        for (int q = 0; q + 1 < L; q += 2) {
            auto U = haar_u4(rng);
            apply_u4(psi, q, L, U);
        }
        // Odd layer
        for (int q = 1; q + 1 < L; q += 2) {
            auto U = haar_u4(rng);
            apply_u4(psi, q, L, U);
        }
        // Measurements
        for (int q = 0; q < L; ++q)
            maybe_measure(psi, q, L, p, rng);
    }
    return half_entropy(psi, L);
}

// ── run_mipt ─────────────────────────────────────────────────────────────────
MIPTResult run_mipt(const MIPTOptions& opts) {
    if (opts.L < 2 || opts.L % 2 != 0)
        throw std::runtime_error("MIPT: L must be a positive even integer >= 2");
    if (opts.L > 12)
        throw std::runtime_error("MIPT: L must be <= 12 for full state-vector simulation");
    if (opts.n_p < 1)
        throw std::runtime_error("MIPT: n_p must be >= 1");
    if (opts.n_samples < 1)
        throw std::runtime_error("MIPT: n_samples must be >= 1");
    const std::string& sc = opts.scenario;
    if (sc != "sweep" && sc != "scaling" && sc != "both")
        throw std::runtime_error("MIPT: unknown scenario; valid: sweep, scaling, both");

    auto t0 = std::chrono::steady_clock::now();
    MIPTResult res;
    res.scenario = sc; res.L = opts.L; res.p = opts.p;
    const int T = (opts.T_steps > 0) ? opts.T_steps : std::max(4 * opts.L, 40);

    // ── p sweep at fixed L ────────────────────────────────────────────────────
    if (sc == "sweep" || sc == "both") {
        const int np = opts.n_p;
        res.p_vals.resize(np); res.S_mean.resize(np); res.S_std.resize(np);
        for (int pi = 0; pi < np; ++pi) {
            const double pv = (np > 1) ? opts.p_max * pi / static_cast<double>(np-1) : opts.p_max;
            res.p_vals[pi] = pv;
            double sum = 0.0, sum2 = 0.0;
            for (int s = 0; s < opts.n_samples; ++s) {
                const double sv = run_circuit(opts.L, pv, T, opts.seed + s * 997 + pi * 7);
                sum += sv; sum2 += sv * sv;
            }
            const double mean = sum / opts.n_samples;
            const double var  = sum2 / opts.n_samples - mean * mean;
            res.S_mean[pi] = mean;
            res.S_std[pi]  = std::sqrt(std::max(0.0, var));
        }
        // Estimate p_c as location of max |dS/dp|
        if (np >= 3) {
            int best_i = 1;
            double best_slope = 0.0;
            for (int pi = 1; pi + 1 < np; ++pi) {
                const double ds = std::abs(res.S_mean[pi+1] - res.S_mean[pi-1]);
                if (ds > best_slope) { best_slope = ds; best_i = pi; }
            }
            res.p_c_est = res.p_vals[best_i];
        }
    }

    // ── Scaling: S(L/2) vs L at p = p_c ─────────────────────────────────────
    if (sc == "scaling" || sc == "both") {
        // Use estimated p_c or opts.p
        const double p_scale = (sc == "both" && !res.p_vals.empty()) ? res.p_c_est : opts.p;
        for (int Lv = 4; Lv <= opts.L_max; Lv += 2) {
            if (Lv > 12) break;
            const int Tv = std::max(4 * Lv, 40);
            double sum = 0.0;
            for (int s = 0; s < opts.n_samples; ++s)
                sum += run_circuit(Lv, p_scale, Tv, opts.seed + s * 997 + 10000 + Lv);
            res.L_vals_sc.push_back(Lv);
            res.S_at_pc.push_back(sum / opts.n_samples);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    res.ok = true; res.message = "ok";
    return res;
}

}  // namespace qx
