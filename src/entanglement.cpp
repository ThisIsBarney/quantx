// src/entanglement.cpp  —  M50: Bipartite Entanglement Entropy & Area Law
//
// Two complementary approaches:
//
// 1. Free-fermion XX chain (correlation-matrix method, O(L^3), any L):
//    H = -J sum_i (c†_i c_{i+1} + h.c.)  (PBC, half-filling)
//    Single-particle eigenstates: phi_k(j) = exp(2πi k j/L)/sqrt(L)
//    Fill L/2 lowest levels; correlation matrix C_{ij} = sum_{k filled} phi_k(i)*conj(phi_k(j)).
//    For subsystem A=[0,l): eigenvalues {lambda_a} of C_A = C[0:l,0:l],
//    S_l = -sum_a [ lambda_a ln(lambda_a) + (1-lambda_a) ln(1-lambda_a) ].
//    CFT prediction (c=1): S(l) = (1/3) ln(sin(π l/L)) + const.
//
// 2. Transverse-field Ising chain (full ED, L <= 14):
//    H = -J sum_i sigma_z_i sigma_z_{i+1} - h sum_i sigma_x_i  (OBC)
//    Ground state via full Jacobi; bipartite cut at position l via SVD of
//    reshaped state vector M[2^l, 2^(L-l)].
//    Critical point h=J=1: S ~ (c/3) ln(l), c=1/2.
//    Area law for h != 1.
//
// Scenarios: "xx" | "tfi" | "both"

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

// ── Entropy from eigenvalues of reduced correlation matrix (free-fermion) ────
static double entropy_from_lambda(const std::vector<double>& lam) {
    double S = 0.0;
    for (double x : lam) {
        const double eps = 1e-14;
        const double xl  = std::max(eps, std::min(1.0-eps, x));
        S -= xl * std::log(xl) + (1.0-xl) * std::log(1.0-xl);
    }
    return S;
}

// ── Jacobi eigenvalues (symmetric, in-place) ─────────────────────────────────
static void jacobi_evals(std::vector<double>& A, int n) {
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i+1; j < n; ++j) { double v=A[i*n+j]; off+=v*v; }
        if (off < 1e-26) break;
        for (int p = 0; p < n-1; ++p) for (int q = p+1; q < n; ++q) {
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

// Jacobi with eigenvectors
static void jacobi_full(std::vector<double>& A, std::vector<double>& V, int n) {
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

// ── XX chain: S(l) via correlation matrix ────────────────────────────────────
static std::vector<double> xx_entropy_profile(int L, double J) {
    // Fill correlation matrix C_{ij} = (1/L) sum_{k: E_k <= E_F} exp(2pi*i*(i-j)*k/L)
    // E_k = -2J cos(2 pi k/L).  Half-filling: L/2 states with lowest energy.
    const int Nfill = L / 2;
    // Sort k by energy
    std::vector<int> ks(L);
    std::iota(ks.begin(), ks.end(), 0);
    std::sort(ks.begin(), ks.end(), [&](int a, int b){
        return std::cos(2*M_PI*a/L) > std::cos(2*M_PI*b/L); // -2J*cos: lowest = largest cos
    });
    // Build real correlation matrix: C_{ij} = (1/L) sum_{k filled} exp(2pi i (i-j) k/L)
    // For real symmetric result, note C_{ij} = C_{ji} and C real because we fill conjugate pairs
    std::vector<double> C(static_cast<size_t>(L)*L, 0.0);
    for (int fi = 0; fi < Nfill; ++fi) {
        const int k = ks[fi];
        for (int i = 0; i < L; ++i)
            for (int j = 0; j < L; ++j)
                C[i*L+j] += std::cos(2*M_PI*k*(i-j)/static_cast<double>(L)) / L;
    }
    // For each cut l = 1,...,L/2: eigenvalues of C[0:l,0:l]
    std::vector<double> S_l;
    S_l.reserve(L/2);
    for (int l = 1; l <= L/2; ++l) {
        std::vector<double> CA(static_cast<size_t>(l)*l);
        for (int i=0; i<l; ++i)
            for (int j=0; j<l; ++j)
                CA[i*l+j] = C[i*L+j];
        jacobi_evals(CA, l);
        std::vector<double> lam(l);
        for (int i=0; i<l; ++i) lam[i] = std::max(0.0, std::min(1.0, CA[i*l+i]));
        S_l.push_back(entropy_from_lambda(lam));
    }
    return S_l;
}

// ── TFI chain: ground state via Jacobi then entropy via SVD of state ─────────
// sigma_z|0>=|0>, sigma_z|1>=-|1>, sigma_x|0>=|1>, sigma_x|1>=|0>
static double tfi_entropy(int L, double J, double h, int cut) {
    const int dim = 1 << L;
    // Build H in {|0⟩,|1⟩}^L basis, |s⟩ = sigma_z basis: sigma_z_i = 1-2*bit_i(s)
    std::vector<double> H(static_cast<size_t>(dim)*dim, 0.0);
    for (int s = 0; s < dim; ++s) {
        // ZZ couplings (OBC)
        double diag = 0.0;
        for (int i = 0; i < L-1; ++i) {
            const int zi = 1-2*((s>>i)&1), zi1 = 1-2*((s>>(i+1))&1);
            diag += -J * zi * zi1;
        }
        H[s*dim+s] = diag;
        // Transverse field sigma_x: flip bit i
        for (int i = 0; i < L; ++i) {
            const int sp = s ^ (1<<i);
            H[s*dim+sp] -= h;
        }
    }
    std::vector<double> evecs;
    jacobi_full(H, evecs, dim);
    // Find ground state column
    std::vector<double> evals(dim);
    for (int i=0; i<dim; ++i) evals[i]=H[i*dim+i];
    const int gs = static_cast<int>(std::min_element(evals.begin(),evals.end())-evals.begin());
    // Extract GS vector
    std::vector<double> psi(dim);
    for (int r=0; r<dim; ++r) psi[r] = evecs[r*dim+gs];
    // Reshape to matrix M[2^cut, 2^(L-cut)]
    const int dA = 1<<cut, dB = 1<<(L-cut);
    // M_{a,b} = psi[a * dB + b]
    // SVD via M^T M (faster for smaller dimension)
    // Compute correlation matrix C = M M^T (dA x dA) if dA <= dB, else M^T M
    if (dA <= dB) {
        std::vector<double> MMT(static_cast<size_t>(dA)*dA, 0.0);
        for (int a=0;a<dA;++a) for (int a2=0;a2<dA;++a2)
            for (int b=0;b<dB;++b)
                MMT[a*dA+a2] += psi[a*dB+b]*psi[a2*dB+b];
        jacobi_evals(MMT, dA);
        std::vector<double> sv2(dA);
        for (int i=0;i<dA;++i) sv2[i] = std::max(0.0, MMT[i*dA+i]);
        double S = 0.0;
        for (double x : sv2) if (x > 1e-14) S -= x * std::log(x);
        return S;
    } else {
        std::vector<double> MTM(static_cast<size_t>(dB)*dB, 0.0);
        for (int b=0;b<dB;++b) for (int b2=0;b2<dB;++b2)
            for (int a=0;a<dA;++a)
                MTM[b*dB+b2] += psi[a*dB+b]*psi[a*dB+b2];
        jacobi_evals(MTM, dB);
        std::vector<double> sv2(dB);
        for (int i=0;i<dB;++i) sv2[i] = std::max(0.0, MTM[i*dB+i]);
        double S = 0.0;
        for (double x : sv2) if (x > 1e-14) S -= x * std::log(x);
        return S;
    }
}

// ── run_entanglement ──────────────────────────────────────────────────────────
EntResult run_entanglement(const EntOptions& opts) {
    if (opts.L < 2 || opts.L % 2 != 0)
        throw std::runtime_error("Ent: L must be a positive even integer >= 2");
    if (opts.L_max < opts.L)
        throw std::runtime_error("Ent: L_max must be >= L");
    if (opts.n_h < 1)
        throw std::runtime_error("Ent: n_h must be >= 1");
    const std::string& sc = opts.scenario;
    if (sc != "xx" && sc != "tfi" && sc != "both")
        throw std::runtime_error("Ent: unknown scenario; valid: xx, tfi, both");

    auto t0 = std::chrono::steady_clock::now();
    EntResult res;
    res.scenario = sc; res.L = opts.L; res.J = opts.J; res.h = opts.h;

    // ── XX chain ─────────────────────────────────────────────────────────────
    if (sc == "xx" || sc == "both") {
        // Profile at fixed L
        const auto Sl = xx_entropy_profile(opts.L, opts.J);
        res.l_vals.resize(Sl.size());
        std::iota(res.l_vals.begin(), res.l_vals.end(), 1);
        res.S_l = Sl;

        // Scaling: S(L/2) vs ln(L) for L = 4,6,...,L_max
        for (int Lv = 4; Lv <= opts.L_max; Lv += 2) {
            const auto Slv = xx_entropy_profile(Lv, opts.J);
            res.L_vals.push_back(Lv);
            res.ln_L.push_back(std::log(static_cast<double>(Lv)));
            res.S_Lhalf.push_back(Slv.back());
        }
        // Fit central charge: S = c/3 * ln(L) + const → linear fit
        if (res.L_vals.size() >= 2) {
            const int nf = static_cast<int>(res.L_vals.size());
            double sx=0,sy=0,sxx=0,sxy=0;
            for (int i=0;i<nf;++i) {
                sx+=res.ln_L[i]; sy+=res.S_Lhalf[i];
                sxx+=res.ln_L[i]*res.ln_L[i]; sxy+=res.ln_L[i]*res.S_Lhalf[i];
            }
            const double denom = nf*sxx-sx*sx;
            res.c_fit = (denom>1e-14) ? 3.0*(nf*sxy-sx*sy)/denom : 0.0;
        }
    }

    // ── TFI h sweep ───────────────────────────────────────────────────────────
    if (sc == "tfi" || sc == "both") {
        const int Ltfi = std::min(opts.L, 8);   // cap at L=8 (2^8=256) for Jacobi speed
        const int nh = opts.n_h;
        res.h_vals.resize(nh); res.S_h.resize(nh);
        for (int hi = 0; hi < nh; ++hi) {
            const double hv = (nh>1) ? opts.h_max * hi / static_cast<double>(nh-1) : opts.h_max;
            res.h_vals[hi] = hv;
            res.S_h[hi]    = tfi_entropy(Ltfi, opts.J, hv, Ltfi/2);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    res.ok = true; res.message = "ok";
    return res;
}

}  // namespace qx
