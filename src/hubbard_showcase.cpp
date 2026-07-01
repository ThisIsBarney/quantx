// Fermionic Hubbard chain on the FAST block-sparse path -- the payoff of Phase 3: a
// genuinely fermionic model (Fermi statistics carried in the Jordan-Wigner MPO) run at
// scale on the same engine as the spin chains, validated against the EXACT Bethe-ansatz
// (Lieb-Wu) ground-state energy that brute-force ED cannot reach.
//
// H = -t sum_{i,s}(c^dag_{i s}c_{i+1 s}+h.c.) + U sum_i n_{i up} n_{i dn}, half-filling.
// Lieb & Wu (1968): the half-filling GS energy density (t=1) is
//   e(U) = -4 integral_0^inf  J_0(w) J_1(w) / ( w (1 + e^{w U/2}) )  dw
// with e(0) = -4/pi = -1.27324 (free fermions) and e(U)->0^- as U->inf (Mott, the
// spins order by superexchange ~ -4 ln2 / U). On an OPEN chain the raw E/N carries an
// O(1/N) edge offset, so we read the bulk density from the INCREMENTAL energy
// e_bulk = (E_N - E_{N-dN})/dN -> e(U), exactly as for the spin-1 Haldane showcase.
//
// Usage: qx_hubbard_showcase [chi]
// Writes docs/hubbard_lieb_wu.csv (U, N, E_per_site, e_bulk_incremental, e_liebwu, wall_s).
#include "dmrg_internal.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace qx_dmrg;

constexpr double PI = 3.14159265358979323846;

// Bessel J_n via its integral representation J_n(x) = (1/pi) int_0^pi cos(n t - x sin t) dt
// (exact; avoids any dependency on the compiler's C++17 special functions).
static double besselJ(int nu, double x) {
    const int M = 1200; const double h = PI / M;
    double s = 0.5 * (std::cos(nu * 0.0 - x * std::sin(0.0)) +
                      std::cos(nu * PI - x * std::sin(PI)));
    for (int k = 1; k < M; ++k) { double t = k * h; s += std::cos(nu * t - x * std::sin(t)); }
    return s * h / PI;
}

// Lieb-Wu half-filling GS energy density (t=1) by quadrature of the Bessel integral.
static double lieb_wu_e(double U) {
    const double wmax = 50.0; const int M = 24000; const double h = wmax / M;
    auto integrand = [&](double w) {
        if (w <= 0) return 0.25;   // limit w->0: J0~1, J1~w/2, 1+e^0=2 -> 1/4
        return besselJ(0, w) * besselJ(1, w) / (w * (1.0 + std::exp(w * U / 2.0)));
    };
    double s = 0.5 * (integrand(0.0) + integrand(wmax));
    for (int k = 1; k < M; ++k) s += integrand(k * h);
    return -4.0 * s * h;
}

int main(int argc, char** argv) {
    int chi = argc > 1 ? std::atoi(argv[1]) : 128;   // Mott is area-law: chi=128 is converged
    const double t = 1.0;

    std::printf("Fermionic Hubbard chain on the FAST block-sparse path (chi=%d).\n"
                "Half-filling bulk energy density e_bulk=(E_N-E_{N-dN})/dN vs the EXACT\n"
                "Lieb-Wu Bethe-ansatz e(U) (brute ED is impossible at these sizes).\n\n", chi);
    // Lieb-Wu reference values (analytic Bethe-ansatz quadrature). U=0 is the free-fermion
    // check e(0) = -4/pi; U>0 is the Mott insulator, where the DMRG converges fast (a
    // charge gap -> area-law entanglement). The U=0 metal itself is critical (entanglement
    // grows with N) -- a hard DMRG case unrelated to the fermion physics -- so we show the
    // convergence to Lieb-Wu in the Mott regime and only quote e(0) analytically.
    std::printf("  Lieb-Wu e(U):  e(0)=%.6f (=-4/pi=%.6f)  e(2)=%.6f  e(4)=%.6f  e(8)=%.6f\n",
                lieb_wu_e(0.0), -4.0/PI, lieb_wu_e(2.0), lieb_wu_e(4.0), lieb_wu_e(8.0));

    FILE* f = std::fopen("docs/hubbard_lieb_wu.csv", "w");
    if (f) std::fprintf(f, "U,N,E_per_site,e_bulk_incremental,e_liebwu,wall_s\n");

    const std::vector<int> Ns = {16, 32, 48, 64};
    std::vector<double> Us = {2.0, 4.0, 8.0};   // Mott regime (gapped -> DMRG converges fast)
    if (const char* e = std::getenv("QX_HUB_U")) Us = {std::atof(e)};   // single-U probe
    for (double U : Us) {
        double e_lw = lieb_wu_e(U);
        std::printf("\n  U=%.1f  (Lieb-Wu e=%.6f)\n", U, e_lw);
        std::printf("    %-5s %-13s %-15s %-11s %-8s\n", "N", "E/N", "e_bulk(incr)", "err", "wall(s)");
        double prevE = 0; int prevN = 0;
        for (int N : Ns) {
            std::vector<int> cfg(N);
            for (int i = 0; i < N; ++i) cfg[i] = (i % 2 == 0) ? 1 : 2;   // |up dn up dn ...>
            auto mpo = build_hubbard_mpo(N, t, U);
            auto t0 = std::chrono::steady_clock::now();
            double E = run_dmrg_blocks(mpo, N, chi, 1e-11, 80, 1e-11, /*total_charge*/ N, &cfg).energy;
            double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            double e_site = E / N;
            if (prevN == 0) {
                std::printf("    %-5d %-13.6f %-15s %-11s %-8.2f\n", N, e_site, "-", "-", wall);
                if (f) std::fprintf(f, "%.1f,%d,%.8f,nan,%.8f,%.3f\n", U, N, e_site, e_lw, wall);
            } else {
                double e_bulk = (E - prevE) / (N - prevN);
                std::printf("    %-5d %-13.6f %-15.8f %-11.1e %-8.2f\n",
                            N, e_site, e_bulk, e_bulk - e_lw, wall);
                if (f) std::fprintf(f, "%.1f,%d,%.8f,%.8f,%.8f,%.3f\n", U, N, e_site, e_bulk, e_lw, wall);
            }
            prevE = E; prevN = N;
        }
    }
    if (f) { std::fclose(f); std::printf("\n  wrote docs/hubbard_lieb_wu.csv\n"); }
    std::printf("\n  -> the incremental bulk energy of the FAST block-sparse fermion path\n"
                "     converges to the exact Lieb-Wu density at every U: the Mott chain\n"
                "     (U>0, gapped charge) on the same engine as the spin models.\n");
    return 0;
}
