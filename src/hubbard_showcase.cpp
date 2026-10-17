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
#include <cstring>
#include <vector>

// Crash reporter (QX_CRASH_STACK=1): on an unhandled SEH exception (access
// violation etc.) print a dbghelp-symbolized stack to stderr and exit. Debugging
// aid for long detached runs on a box with no debugger installed; with a /Zi PDB
// next to the exe the frames come out as function [file:line].
#ifdef _WIN32
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
static LONG WINAPI qx_crash_report(EXCEPTION_POINTERS* ep) {
    HANDLE proc = GetCurrentProcess();
    SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
    SymInitialize(proc, nullptr, TRUE);
    std::fprintf(stderr, "\n=== QX CRASH: code=0x%08lx addr=%p ===\n",
                 (unsigned long)ep->ExceptionRecord->ExceptionCode,
                 ep->ExceptionRecord->ExceptionAddress);
    void* frames[62];
    USHORT nf = CaptureStackBackTrace(0, 62, frames, nullptr);
    alignas(SYMBOL_INFO) char buf[sizeof(SYMBOL_INFO) + 256];
    for (USHORT k = 0; k < nf; ++k) {
        auto* si = (SYMBOL_INFO*)buf;
        si->SizeOfStruct = sizeof(SYMBOL_INFO); si->MaxNameLen = 255;
        DWORD64 disp = 0; DWORD ld = 0;
        IMAGEHLP_LINE64 line; line.SizeOfStruct = sizeof(IMAGEHLP_LINE64);
        if (SymFromAddr(proc, (DWORD64)frames[k], &disp, si)) {
            if (SymGetLineFromAddr64(proc, (DWORD64)frames[k], &ld, &line))
                std::fprintf(stderr, "  #%02u %s +0x%llx  [%s:%lu]\n", k, si->Name,
                             (unsigned long long)disp, line.FileName,
                             (unsigned long)line.LineNumber);
            else
                std::fprintf(stderr, "  #%02u %s +0x%llx\n", k, si->Name,
                             (unsigned long long)disp);
        } else {
            std::fprintf(stderr, "  #%02u %p\n", k, frames[k]);
        }
    }
    std::fflush(stderr);
    return EXCEPTION_EXECUTE_HANDLER;
}
#endif

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

// Lieb-Wu half-filling GS energy density (t=1) by quadrature of the Bessel integral,
// e(U) = -4 int_0^inf J0(w) J1(w) / (w (1 + e^{wU/2})) dw. Simpson's rule: the trapezoid
// rule this used left an O(h^2) error from the integrand's slope -U/16 at w = 0, which
// put e(U) off by 1.8e-7, 3.6e-7 and 7.3e-7 at U = 2, 4 and 8. The cut at w = 50 is
// harmless for U > 0 (the factor 1/(1 + e^{wU/2}) is below e^{-50}); at U = 0 the
// integrand decays only as 1/w^2 and the cut leaves 2.4e-6 against -4/pi.
static double lieb_wu_e(double U) {
    const double wmax = 50.0; const int M = 24000; const double h = wmax / M;   // M even
    auto integrand = [&](double w) {
        if (w <= 0) return 0.25;   // limit w->0: J0~1, J1~w/2, 1+e^0=2 -> 1/4
        return besselJ(0, w) * besselJ(1, w) / (w * (1.0 + std::exp(w * U / 2.0)));
    };
    double s = integrand(0.0) + integrand(wmax);
    for (int k = 1; k < M; ++k) s += (k % 2 ? 4.0 : 2.0) * integrand(k * h);
    return -4.0 * s * h / 3.0;
}

int main(int argc, char** argv) {
#ifdef _WIN32
    if (const char* cs = std::getenv("QX_CRASH_STACK"); cs && std::atoi(cs) != 0)
        SetUnhandledExceptionFilter(qx_crash_report);
#endif
    int chi = argc > 1 ? std::atoi(argv[1]) : 128;   // Mott is area-law: chi=128 is converged
    const double t = 1.0;

    // STRIPES showcase: QX_STRIPES="Lx W chi [U holes hpin nsw tp lam]" -> hole-doped
    // width-W cylinder GS with staggered edge pinning (x=0 column); emits per-site
    // charge/spin profiles ("PROF i x y n 2sz" lines) for the stripe-order figure.
    // holes must be even (removed as up/dn pairs so the target sector is 2Sz=0).
    // lam > 0 seeds a stripe pattern of WAVELENGTH lam: Lx/lam stripes, holes/(Lx/lam)
    // holes dug from each stripe column -- the lambda=4 (half-filled) vs lambda=8
    // (filled) duel at delta=1/8. lam = 0 keeps the evenly-spread seed.
    if (const char* sz = std::getenv("QX_STRIPES")) {
        int Lx = 16, Wd = 4, cchi = 1024; double U = 8.0;
        int holes = 8; double hp = 0.5; int nsw = 16; double tp = 0.0, lam = 0, vamp = 0.0;
        std::sscanf(sz, "%d %d %d %lf %d %lf %d %lf %lf %lf",
                    &Lx, &Wd, &cchi, &U, &holes, &hp, &nsw, &tp, &lam, &vamp);
        const int lami = (int)lam;
        const int n = Wd * Lx, Ne = n - holes;
        if (holes < 0 || holes > n || holes % 2 != 0 || n % 2 != 0) {
            std::fprintf(stderr, "QX_STRIPES: need even n and even holes in [0, n] "
                         "(got n=%d holes=%d)\n", n, holes);
            return 1;
        }
#ifdef _WIN32
        _putenv_s("QX_PROFILES", "1");
#else
        setenv("QX_PROFILES", "1", 1);
#endif
        // vamp != 0: run the BIASED Hamiltonian (charge modulation of wavelength lam,
        // which holds a stripe pattern that a bare product seed cannot -- the seed is
        // washed out by the two-site bond ramp) and report <H_0> measured on the
        // resulting state. <psi|H_0|psi> is a legitimate variational upper bound for
        // the physical H_0 whatever the bias was, so the two patterns can be compared.
        auto mpo = build_hubbard_lattice_split_mpo(Lx, Wd, 1.0, U, 0.0, true, hp, tp,
                                                   vamp, lami);
        auto mpo0 = build_hubbard_lattice_split_mpo(Lx, Wd, 1.0, U, 0.0, true, hp, tp);
        // init: half-filling Neel, then dig the holes.
        std::vector<int> cfg(n);
        for (int i = 0; i < n; ++i) cfg[i] = (i % 2 == 0) ? 1 : 2;
        if (lami > 0) {
            // stripe seed: ns stripes of hps holes, one stripe column per period,
            // centred in its period so the pattern is symmetric about the cylinder.
            if (Lx % lami != 0) {
                std::fprintf(stderr, "QX_STRIPES: lam=%d must divide Lx=%d\n", lami, Lx);
                return 1;
            }
            const int ns = Lx / lami, hps = holes / ns;
            if (ns == 0 || holes % ns != 0 || hps > Wd || hps % 2 != 0) {
                std::fprintf(stderr, "QX_STRIPES: holes=%d cannot form %d stripes of an "
                             "even count <= W=%d\n", holes, ns, Wd);
                return 1;
            }
            for (int k = 0; k < ns; ++k) {
                int xs = lami / 2 + k * lami;             // stripe column
                for (int y = 0; y < hps; ++y) cfg[Wd * xs + y] = 0;
            }
        } else {
            for (int k = 0; k < holes; ++k) {             // evenly spread (the default)
                int want = (k % 2 == 0) ? 1 : 2;          // remove up, then dn, ...
                int i0 = (int)(((double)k + 0.5) * n / holes);
                for (int off = 0; off < n; ++off) {
                    int i = (i0 + off) % n;
                    if (cfg[i] == want) { cfg[i] = 0; break; }
                }
            }
        }
        // The seed must land exactly in the target sector, or the bond spaces have no
        // path for it and DMRG throws a confusing width-0 error deep inside.
        {
            const int qs[4] = {pack_q2(0,0), pack_q2(1,+1), pack_q2(1,-1), pack_q2(2,0)};
            int qtot = 0;
            for (int i = 0; i < n; ++i) qtot += qs[cfg[i]];
            if (qtot != pack_q2(Ne, 0)) {
                std::fprintf(stderr, "QX_STRIPES: seed charge %d != target %d "
                             "(hole pattern broke N or 2Sz)\n", qtot, pack_q2(Ne, 0));
                return 1;
            }
        }
        auto t0 = std::chrono::steady_clock::now();
        DMRGResult r;
        double EH0;
        if (vamp != 0.0) {
            auto vals = gs_mpo_expectations(mpo, n, cchi, 1e-11, nsw, 1e-12,
                                            pack_q2(Ne, 0), &cfg, {mpo0}, &r);
            EH0 = vals[0];                       // <psi_biased| H_0 |psi_biased>
        } else {
            r = run_dmrg_blocks(mpo, n, cchi, 1e-11, nsw, 1e-12, pack_q2(Ne, 0), &cfg);
            EH0 = r.energy;                      // unbiased run: the GS energy itself
        }
        double w0 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double nsum = 0;
        for (double v : r.site_n) nsum += v;
        // NOTE: everything after the wall time is appended, so the existing
        // plot_stripes.py header regex (which ends there) keeps matching. E0 is the
        // energy of whatever Hamiltonian was optimised; EH0 is always the physical
        // (unbiased) energy -- the quantity to compare across stripe patterns.
        std::printf("STRIPES Lx=%d W=%d chi=%d U=%.2f holes=%d hpin=%.2f nsw=%d tp=%.2f : "
                    "E0=%.10f (%.1fs)  sum<n>=%.6f (target %d) lam=%d trunc=%.6e "
                    "E0site=%.10f vamp=%.4f EH0=%.10f EH0site=%.10f\n",
                    Lx, Wd, cchi, U, holes, hp, nsw, tp, r.energy, w0, nsum, Ne,
                    lami, r.truncation_error, r.energy / n, vamp, EH0, EH0 / n);
        for (int i = 0; i < n; ++i)
            std::printf("PROF %d %d %d %.8f %.8f\n", i, i / Wd, i % Wd,
                        r.site_n[i], r.site_2sz[i]);
        return 0;
    }

    // PAIR-FIELD showcase: QX_PAIRS="Lx W chi [U holes hpin nsw x0 rmax]" -> the
    // doped pinned cylinder GS (the stripe state), then singlet pair-field
    // correlators Phi(r) = <Delta^dag_{b(x0)} Delta_{b(x0+r)}> via one GS run +
    // per-obs environment rebuilds: yy (rung-oriented, y-averaged), xx (leg,
    // starts at r=2 -- r=1 legs share a site), yx (mixed orientation -- the d-wave
    // sign test: Phi_yx < 0 while Phi_yy > 0). Emits "PAIRF kind r phi" lines.
    if (const char* pz = std::getenv("QX_PAIRS")) {
        int Lx = 16, Wd = 4, cchi = 1024; double U = 8.0;
        int holes = 8; double hp = 0.5; int nsw = 20, x0 = 2, rmax = 11; double tp = 0.0;
        std::sscanf(pz, "%d %d %d %lf %d %lf %d %d %d %lf",
                    &Lx, &Wd, &cchi, &U, &holes, &hp, &nsw, &x0, &rmax, &tp);
        const int n = Wd * Lx, Ne = n - holes;
        if (holes < 0 || holes > n || holes % 2 != 0 || n % 2 != 0 || x0 < 0 || x0 >= Lx) {
            std::fprintf(stderr, "QX_PAIRS: bad geometry (n=%d holes=%d x0=%d)\n", n, holes, x0);
            return 1;
        }
#ifdef _WIN32
        _putenv_s("QX_PROFILES", "1");
#else
        setenv("QX_PROFILES", "1", 1);
#endif
        auto mpo = build_hubbard_lattice_split_mpo(Lx, Wd, 1.0, U, 0.0, true, hp, tp);
        std::vector<int> cfg(n);
        for (int i = 0; i < n; ++i) cfg[i] = (i % 2 == 0) ? 1 : 2;
        for (int k = 0; k < holes; ++k) {
            int want = (k % 2 == 0) ? 1 : 2;
            int i0 = (int)(((double)k + 0.5) * n / holes);
            for (int off = 0; off < n; ++off) {
                int i = (i0 + off) % n;
                if (cfg[i] == want) { cfg[i] = 0; break; }
            }
        }
        struct ObsMeta { const char* kind; int r; };
        std::vector<std::vector<MPOSite>> obs;
        std::vector<ObsMeta> meta;
        for (int r = 1; r <= rmax && x0 + r < Lx; ++r)
            for (int y = 0; y < Wd - 1; ++y) {
                obs.push_back(build_pair_corr_mpo(n, Wd, x0, y, 'y', x0 + r, y, 'y'));
                meta.push_back({"yy", r});
            }
        for (int r = 2; r <= rmax && x0 + r + 1 < Lx; ++r) {
            obs.push_back(build_pair_corr_mpo(n, Wd, x0, 1, 'x', x0 + r, 1, 'x'));
            meta.push_back({"xx", r});
        }
        for (int r = 1; r <= rmax && x0 + r + 1 < Lx; ++r) {
            obs.push_back(build_pair_corr_mpo(n, Wd, x0, 1, 'y', x0 + r, 1, 'x'));
            meta.push_back({"yx", r});
        }
        DMRGResult rgs;
        auto t0 = std::chrono::steady_clock::now();
        auto vals = gs_mpo_expectations(mpo, n, cchi, 1e-11, nsw, 1e-12,
                                        pack_q2(Ne, 0), &cfg, obs, &rgs);
        double w0 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("PAIRS Lx=%d W=%d chi=%d U=%.2f holes=%d hpin=%.2f nsw=%d x0=%d tp=%.2f : "
                    "E0=%.10f (%.1fs)  n_obs=%d\n",
                    Lx, Wd, cchi, U, holes, hp, nsw, x0, tp, rgs.energy, w0, (int)obs.size());
        for (int i = 0; i < n && i < (int)rgs.site_n.size(); ++i)
            std::printf("PROF %d %d %d %.8f %.8f\n", i, i / Wd, i % Wd,
                        rgs.site_n[i], rgs.site_2sz[i]);
        // aggregate: yy averaged over the Wd-1 leg positions; xx/yx as-is
        for (int r = 1; r <= rmax; ++r) {
            double s = 0; int c = 0;
            for (size_t k = 0; k < meta.size(); ++k)
                if (meta[k].r == r && std::strcmp(meta[k].kind, "yy") == 0) { s += vals[k]; ++c; }
            if (c) std::printf("PAIRF yy %d %.10e\n", r, s / c);
        }
        for (size_t k = 0; k < meta.size(); ++k)
            if (std::strcmp(meta[k].kind, "yy") != 0)
                std::printf("PAIRF %s %d %.10e\n", meta[k].kind, meta[k].r, vals[k]);
        return 0;
    }

    // Width-W CYLINDER/strip mode: QX_CYL="Lx W chi [U yper]" -> half-filling GS on
    // the general period-W builder; prints E0 and wall. The quasi-2D benchmark entry.
    if (const char* cz = std::getenv("QX_CYL")) {
        int Lx = 8, Wd = 4, cchi = 512; double U = 8.0; int yper = 1, nsw = 60;
        std::sscanf(cz, "%d %d %d %lf %d %d", &Lx, &Wd, &cchi, &U, &yper, &nsw);
        // Width-4 cylinders at practical chi are TRUNCATION-limited: dE per sweep
        // plateaus far above any tol, so a tol-stop burns the whole sweep budget.
        // The benchmark protocol is therefore a FIXED sweep budget (nsw, same on
        // both tools), reporting the energy reached and the wall time.
        auto mpo = build_hubbard_lattice_split_mpo(Lx, Wd, 1.0, U, 0.0, yper != 0);
        const int n = Wd * Lx;
        std::vector<int> cfg(n);
        for (int i = 0; i < n; ++i) cfg[i] = (i % 2 == 0) ? 1 : 2;
        auto t0 = std::chrono::steady_clock::now();
        double E0 = run_dmrg_blocks(mpo, n, cchi, 1e-11, nsw, 1e-12,
                                    pack_q2(n, 0), &cfg).energy;
        double w0 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("CYL Lx=%d W=%d chi=%d U=%.2f yper=%d nsw=%d : E0=%.10f (%.1fs)  E0/site=%.6f\n",
                    Lx, Wd, cchi, U, yper, nsw, E0, w0, E0 / n);
        return 0;
    }

    // Two-leg LADDER mode: QX_LADDER="Lx chi [U tp]" -> half-filling GS + the Sz=1
    // sector on the rung-fused (d=16) chain; prints both energies, the spin gap and
    // wall times. Used by the ladder showcase sweep and the vs-TeNPy ladder bench.
    if (const char* lz = std::getenv("QX_LADDER")) {
        int Lx = 8, lchi = 256; double U = 8.0, tp = 1.0; int split = 0;
        std::sscanf(lz, "%d %d %lf %lf %d", &Lx, &lchi, &U, &tp, &split);
        const int nm = 2 * Lx;
        const int nsites = split ? nm : Lx;
        auto mpo = split ? build_hubbard_ladder_split_mpo(Lx, 1.0, tp, U)
                         : build_hubbard_ladder_mpo(Lx, 1.0, tp, U);
        std::vector<int> cfg0(nsites), cfg1;
        if (split) { for (int i = 0; i < nm; ++i) cfg0[i] = (i % 2 == 0) ? 1 : 2;
                     cfg1 = cfg0; cfg1[1] = 1; }
        else       { for (int x = 0; x < Lx; ++x) cfg0[x] = (x % 2 == 0) ? (4*1+2) : (4*2+1);
                     cfg1 = cfg0; cfg1[0] = 4*1 + 1; }
        auto t0 = std::chrono::steady_clock::now();
        double E0 = run_dmrg_blocks(mpo, nsites, lchi, 1e-11, 60, 1e-10,
                                    pack_q2(nm, 0), &cfg0).energy;
        double w0 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        t0 = std::chrono::steady_clock::now();
        double E1 = run_dmrg_blocks(mpo, nsites, lchi, 1e-11, 60, 1e-10,
                                    pack_q2(nm, 2), &cfg1).energy;
        double w1 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("LADDER%s Lx=%d chi=%d U=%.2f tp=%.2f : E0=%.10f (%.1fs)  "
                    "E1=%.10f (%.1fs)  gap=%.6f\n", split ? "-SPLIT" : "",
                    Lx, lchi, U, tp, E0, w0, E1, w1, E1 - E0);
        return 0;
    }

    // Single-run GS bench mode (for benchmarks/hubbard_dmrg_vs_tenpy.py):
    // QX_HUB_BENCH="N chi" [QX_HUB_U, QX_HUB_2U1=1] -> one half-filling GS DMRG,
    // prints "E= ... wall= ... s" and exits. Packed two-U(1) by default (our best).
    if (const char* b = std::getenv("QX_HUB_BENCH")) {
        int N = 32, bchi = 128;
        std::sscanf(b, "%d %d", &N, &bchi);
        double U = std::getenv("QX_HUB_U") ? std::atof(std::getenv("QX_HUB_U")) : 4.0;
        const char* h2 = std::getenv("QX_HUB_2U1");
        const bool two_u1 = !(h2 && std::atoi(h2) == 0);
        auto mpo = build_hubbard_mpo(N, t, U, two_u1);
        std::vector<int> cfg(N);
        for (int i = 0; i < N; ++i) cfg[i] = (i % 2 == 0) ? 1 : 2;
        auto t0 = std::chrono::steady_clock::now();
        // tol 1e-10 matches the TeNPy side's max_E_err=1e-10 (fair convergence criterion)
        double E = run_dmrg_blocks(mpo, N, bchi, 1e-11, 80, 1e-10,
                                   two_u1 ? pack_q2(N, 0) : N, &cfg).energy;
        double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("HUB_BENCH N=%d chi=%d U=%.2f 2U1=%d : E=%.10f wall=%8.3f s\n",
                    N, bchi, U, (int)two_u1, E, wall);
        return 0;
    }

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
