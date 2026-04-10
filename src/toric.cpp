// src/toric.cpp  —  M53: Toric Code / Surface Code
//
// Kitaev's toric code on an L×L torus (L=2 or 3).
//
// Layout: 2L² qubits on edges of the L×L torus.
//   Horizontal: h_{r,c} → qubit r*L + c,    r,c = 0..L-1
//   Vertical:   v_{r,c} → qubit L² + r*L + c
//
// Stabilizers:
//   Star A_v = ⊗_{e∈star(v)} X_e  (vertex stabilizer, 4-body X)
//   Plaquette B_p = ⊗_{e∈∂p} Z_e  (face stabilizer, 4-body Z)
//
// Hamiltonian:
//   H = -J_A Σ_v A_v  -  J_B Σ_p B_p  -  h Σ_i X_i  -  g Σ_i Z_i
//
// Topological phase (h=g=0):
//   Ground state is in +1 eigenspace of all A_v and B_p.
//   On the torus: 4-fold degeneracy (logical X̄_1, X̄_2, Z̄_1, Z̄_2).
//   Anyonic gap: Δ_e = 2J_A,  Δ_m = 2J_B.
//
// Phase diagram: h sweep (X perturbation) at J_A=J_B=1 drives a
//   topological-to-trivial quantum phase transition at h_c ≈ 0.33 J.
//
// Scenarios: "spectrum" | "perturb" | "both"

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

// ── Jacobi diagonalisation (eigenvalues + eigenvectors, in-place) ─────────────
static void jacobi_full_tc(std::vector<double>& A, std::vector<double>& V, int n) {
    V.assign(static_cast<size_t>(n)*n, 0.0);
    for (int i=0; i<n; ++i) V[i*n+i]=1.0;
    for (int sweep=0; sweep<200; ++sweep) {
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

// Eigenvalues only
static void jacobi_evals_tc(std::vector<double>& A, int n) {
    for (int sweep=0; sweep<200; ++sweep) {
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
        }
    }
}

// ── Toric code Hamiltonian builder ────────────────────────────────────────────
// N_q = 2*L*L qubits, dim = 2^N_q (L=2: N_q=8, dim=256)
static std::vector<double> build_toric(int L, double JA, double JB, double h, double g) {
    const int Nq = 2 * L * L;
    const int dim = 1 << Nq;
    std::vector<double> H(static_cast<size_t>(dim)*dim, 0.0);

    // Helper: qubit index
    auto hq = [&](int r, int c) { return r*L + c; };            // horizontal
    auto vq = [&](int r, int c) { return L*L + r*L + c; };      // vertical

    // Star operators A_v = X_{h(r,c)} X_{h(r,(c-1)%L)} X_{v(r,c)} X_{v((r-1)%L,c)}
    // Each A_v flips 4 bits → off-diagonal
    for (int r = 0; r < L; ++r) for (int c = 0; c < L; ++c) {
        int q0 = hq(r, c);
        int q1 = hq(r, (c-1+L)%L);
        int q2 = vq(r, c);
        int q3 = vq((r-1+L)%L, c);
        int flip = (1<<q0) | (1<<q1) | (1<<q2) | (1<<q3);
        for (int s = 0; s < dim; ++s) {
            int t = s ^ flip;
            H[s*dim+t] -= JA;
        }
    }

    // Plaquette operators B_p = Z_{h(r,c)} Z_{h((r+1)%L,c)} Z_{v(r,c)} Z_{v(r,(c+1)%L)}
    // Each B_p is diagonal: (-1)^{sum of 4 bits}
    for (int r = 0; r < L; ++r) for (int c = 0; c < L; ++c) {
        int q0 = hq(r, c);
        int q1 = hq((r+1)%L, c);
        int q2 = vq(r, c);
        int q3 = vq(r, (c+1)%L);
        for (int s = 0; s < dim; ++s) {
            int bits = ((s>>q0)&1) + ((s>>q1)&1) + ((s>>q2)&1) + ((s>>q3)&1);
            H[s*dim+s] -= JB * (bits%2==0 ? 1.0 : -1.0);
        }
    }

    // Transverse-field perturbations
    if (std::abs(h) > 1e-14) {
        // h Σ X_i  (off-diagonal: flip bit i)
        for (int i = 0; i < Nq; ++i)
            for (int s = 0; s < dim; ++s)
                H[s*dim + (s^(1<<i))] -= h;
    }
    if (std::abs(g) > 1e-14) {
        // g Σ Z_i  (diagonal: (-1)^{bit i})
        for (int i = 0; i < Nq; ++i)
            for (int s = 0; s < dim; ++s)
                H[s*dim+s] -= g * (((s>>i)&1) ? -1.0 : 1.0);
    }

    return H;
}

// ── Entanglement entropy of half the qubits from ground state ─────────────────
static double half_entropy_tc(const std::vector<double>& psi, int Nq) {
    const int dA = 1 << (Nq / 2);
    const int dB = 1 << (Nq - Nq / 2);
    std::vector<double> MTM(static_cast<size_t>(dA)*dA, 0.0);
    // M_{a,b} = psi[b*dA + a]  (low bits = A, high bits = B)
    for (int a = 0; a < dA; ++a)
        for (int a2 = 0; a2 < dA; ++a2) {
            double sum = 0.0;
            for (int b = 0; b < dB; ++b)
                sum += psi[b*dA+a] * psi[b*dA+a2];
            MTM[a*dA+a2] = sum;
        }
    jacobi_evals_tc(MTM, dA);
    double S = 0.0;
    for (int i = 0; i < dA; ++i) {
        const double sv2 = std::max(0.0, MTM[i*dA+i]);
        if (sv2 > 1e-14) S -= sv2 * std::log(sv2);
    }
    return S;
}

// ── run_toric ─────────────────────────────────────────────────────────────────
ToricResult run_toric(const ToricOptions& opts) {
    if (opts.L != 2)
        throw std::runtime_error("Toric: L must be 2 (L=3 needs 2^18=262144 dim → ~549 GB)");
    const std::string& sc = opts.scenario;
    if (sc != "spectrum" && sc != "perturb" && sc != "both")
        throw std::runtime_error("Toric: unknown scenario; valid: spectrum, perturb, both");

    auto t0 = std::chrono::steady_clock::now();
    ToricResult res;
    res.scenario = sc; res.L = opts.L; res.JA = opts.JA; res.JB = opts.JB;

    const int Nq = 2 * opts.L * opts.L;
    const int dim = 1 << Nq;

    // ── Spectrum at fixed parameters ─────────────────────────────────────────
    if (sc == "spectrum" || sc == "both") {
        auto H = build_toric(opts.L, opts.JA, opts.JB, opts.h, opts.g);
        std::vector<double> evecs;
        jacobi_full_tc(H, evecs, dim);
        std::vector<double> evals(dim);
        for (int i = 0; i < dim; ++i) evals[i] = H[i*dim+i];
        // Find GS index in the UNSORTED diagonal BEFORE sorting evals
        const int gs_idx = static_cast<int>(
            std::min_element(evals.begin(), evals.end()) - evals.begin());
        std::sort(evals.begin(), evals.end());
        res.eigenvalues = std::vector<double>(evals.begin(), evals.begin() + std::min(dim, 20));
        res.E0 = evals[0];
        // Count degeneracy of ground state (within tolerance)
        int degen = 0;
        const double tol = 1e-8;
        for (double e : evals) if (std::abs(e - res.E0) < tol) ++degen;
        res.gs_degeneracy = degen;
        res.anyonic_gap = (dim > degen) ? evals[degen] - res.E0 : 0.0;
        // Ground state entanglement entropy: use gs_idx directly
        {
            std::vector<double> psi(dim);
            for (int r = 0; r < dim; ++r) psi[r] = evecs[r*dim+gs_idx];
            res.entropy_gs = half_entropy_tc(psi, Nq);
        }
    }

    // ── Perturbation sweep: h_x from 0 to h_max ──────────────────────────────
    if (sc == "perturb" || sc == "both") {
        const int nh = opts.n_h;
        res.h_vals.resize(nh); res.gap_h.resize(nh); res.entropy_h.resize(nh);
        for (int hi = 0; hi < nh; ++hi) {
            const double hv = (nh > 1) ? opts.h_max * hi / static_cast<double>(nh-1) : opts.h_max;
            res.h_vals[hi] = hv;
            auto Hh = build_toric(opts.L, opts.JA, opts.JB, hv, 0.0);
            std::vector<double> evecs_h;
            jacobi_full_tc(Hh, evecs_h, dim);
            std::vector<double> evh(dim);
            for (int i = 0; i < dim; ++i) evh[i] = Hh[i*dim+i];
            std::sort(evh.begin(), evh.end());
            const double e0h = evh[0];
            // Count degeneracy
            int dh = 0;
            for (double e : evh) if (std::abs(e - e0h) < 1e-8) ++dh;
            res.gap_h[hi] = (dim > dh) ? evh[dh] - e0h : 0.0;
            // GS entropy
            for (int i = 0; i < dim; ++i) {
                if (std::abs(Hh[i*dim+i] - e0h) < 1e-8) {
                    std::vector<double> psi(dim);
                    for (int r = 0; r < dim; ++r) psi[r] = evecs_h[r*dim+i];
                    res.entropy_h[hi] = half_entropy_tc(psi, Nq);
                    break;
                }
            }
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    res.ok = true; res.message = "ok";
    return res;
}

}  // namespace qx
