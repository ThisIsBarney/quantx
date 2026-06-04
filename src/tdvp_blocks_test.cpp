// Validation of block-sparse (Sz-symmetric) TDVP2 vs exact diagonalization:
// real-time Neel quench staggered magnetization, and energy conservation.
#include "dmrg_internal.hpp"
#include <Eigen/Dense>
#include <cstdio>
#include <complex>
#include <vector>

using namespace qx_dmrg;
using Cm = Eigen::MatrixXcd;
using Cv = Eigen::VectorXcd;
using Cx = std::complex<double>;

static int failures = 0;
static void check(const char* name, double err, double tol) {
    bool ok = err < tol;
    if (!ok) ++failures;
    printf("  %-44s err=%.3e  %s\n", name, err, ok ? "OK" : "FAIL");
}

static Cm kron(const Cm& A, const Cm& B) {
    Cm C(A.rows()*B.rows(), A.cols()*B.cols());
    for (int i=0;i<A.rows();++i) for (int j=0;j<A.cols();++j)
        C.block(i*B.rows(), j*B.cols(), B.rows(), B.cols()) = A(i,j)*B;
    return C;
}

int main() {
    const int n = 6;
    Cm X(2,2); X<<0,1,1,0;
    Cm Y(2,2); Y<<Cx(0,0),Cx(0,-1),Cx(0,1),Cx(0,0);
    Cm Z(2,2); Z<<1,0,0,-1;
    Cm I2 = Cm::Identity(2,2);
    auto emb=[&](const Cm& P,int i){ Cm m(1,1); m<<1; for(int k=0;k<n;++k) m=kron(m,(k==i)?P:I2); return m; };
    int d=1<<n;
    Cm H=Cm::Zero(d,d);
    for(int i=0;i+1<n;++i) for(const Cm* P:{&X,&Y,&Z}) H+=emb(*P,i)*emb(*P,i+1);
    Eigen::SelfAdjointEigenSolver<Cm> es(H);
    Cv psi0=Cv::Zero(d); int idx=0; for(int i=0;i<n;++i) idx=idx*2+(i%2==0?0:1); psi0(idx)=1;
    std::vector<Cm> Zs; for(int i=0;i<n;++i) Zs.push_back(emb(Z,i));
    auto stag_exact=[&](double t){ Cv c=es.eigenvectors().adjoint()*psi0;
        for(int k=0;k<d;++k) c(k)*=std::exp(Cx(0,-t)*es.eigenvalues()(k));
        Cv ps=es.eigenvectors()*c;
        double m=0; for(int i=0;i<n;++i) m+=((i%2==0)?1.0:-1.0)*std::real(ps.dot(Zs[i]*ps)); return m/n; };

    auto mpo = build_heisenberg_mpo(n, 1.0);
    double dt=0.02; int steps=25;
    auto r = run_tdvp2_blocks(mpo, n, dt, steps, /*chi_max*/16, 1e-12, /*imag*/false);

    printf("Block-sparse TDVP2 Neel quench (n=%d):\n", n);
    double maxerr=0;
    for (size_t k=0;k<r.times.size();++k) maxerr=std::max(maxerr, std::abs(r.stag[k]-stag_exact(r.times[k])));
    check("staggered magnetization vs exact", maxerr, 1e-3);
    // energy conservation: Neel energy under sigma.sigma = -(n-1)
    check("energy conserved (= -(n-1))", std::abs(r.energy - (-(n-1.0))), 1e-6);
    printf("  max bond reached: %d\n", r.max_bond);

    // imaginary-time -> ground state
    auto g = run_tdvp2_blocks(mpo, n, 0.1, 200, 16, 1e-12, /*imag*/true);
    check("imag-time -> ground vs exact", std::abs(g.energy - es.eigenvalues()(0)), 1e-5);

    // Block-sparse TEBD2 (Trotter gates): same Neel quench vs exact, and imag->ground
    {
        // TEBD has O(dt^2) Trotter error (unlike Krylov-exact TDVP2), so tolerances
        // reflect the second-order Suzuki-Trotter accuracy, not machine precision.
        auto rt = run_tebd2_blocks(mpo, n, dt, steps, /*chi_max*/16, 1e-12, /*imag*/false);
        double me = 0;
        for (size_t k=0;k<rt.times.size();++k) me=std::max(me, std::abs(rt.stag[k]-stag_exact(rt.times[k])));
        check("TEBD2 staggered magnetization vs exact", me, 1e-3);
        check("TEBD2 energy conserved (Trotter ~dt^2)", std::abs(rt.energy - (-(n-1.0))), 1e-3);
        auto gt = run_tebd2_blocks(mpo, n, 0.05, 400, 16, 1e-12, /*imag*/true);
        check("TEBD2 imag-time -> ground vs exact", std::abs(gt.energy - es.eigenvalues()(0)), 1e-4);
    }

    // Dynamical structure factor: C_{j,l0}(t) = <psi0| sz_j e^{-i(H-E0)t} sz_l0 |psi0>
    // from the block engine vs exact diagonalization.
    printf("Dynamical S(k,w) correlations C_{j,l0}(t) vs exact:\n");
    {
        int l0 = n / 2;
        Cv psi0 = es.eigenvectors().col(0);
        double E0 = es.eigenvalues()(0);
        Cv phi0 = Zs[l0] * psi0;                         // sz_l0 |psi0>
        auto C_exact = [&](int j, double t) -> Cx {
            Cv c = es.eigenvectors().adjoint() * phi0;
            for (int k = 0; k < d; ++k) c(k) *= std::exp(Cx(0,-(es.eigenvalues()(k)-E0))*t);
            Cv phit = es.eigenvectors() * c;             // e^{-i(H-E0)t} sz_l0 |psi0>
            return psi0.dot(Zs[j] * phit);
        };
        auto sp = run_spectral_blocks(mpo, n, /*dt*/0.05, /*n_steps*/20, /*chi*/32, 1e-12, 40);
        double maxerr = 0;
        for (size_t ti = 0; ti < sp.times.size(); ++ti)
            for (int j = 0; j < n; ++j) {
                Cx ce = C_exact(j, sp.times[ti]);
                Cx cb(sp.C_re[ti][j], sp.C_im[ti][j]);
                maxerr = std::max(maxerr, std::abs(ce - cb));
            }
        // evolution is TEBD (O(dt^2) Trotter); dt=0.05 over 20 steps -> ~1e-2 accuracy
        check("C_{j,l0}(t) vs exact (all j,t, TEBD Trotter)", maxerr, 1.5e-2);
        printf("  E0=%.6f exact=%.6f  l0=%d  max_bond=%d  %.3fs\n",
               sp.E0, E0, sp.l0, sp.max_bond, sp.wall_time_s);
    }

    // XXZ generalization: H = sum (XX + YY + delta ZZ). Validate the parameterized
    // TEBD gate + MPO at delta != 1 (gapless XY-like and gapped Ising-like) vs ED.
    printf("XXZ (delta != 1) TEBD + ground vs exact:\n");
    for (double delta : {0.5, 1.5}) {
        Cm Hd = Cm::Zero(d, d);
        for (int i=0;i+1<n;++i) {
            Hd += emb(X,i)*emb(X,i+1);
            Hd += emb(Y,i)*emb(Y,i+1);
            Hd += delta * (emb(Z,i)*emb(Z,i+1));
        }
        Eigen::SelfAdjointEigenSolver<Cm> esd(Hd);
        auto mpo = build_xxz_mpo(n, 1.0, delta);
        // imaginary-time TEBD cooling -> ground energy
        auto g = run_tebd2_blocks(mpo, n, 0.05, 400, 32, 1e-12, /*imag*/true, 0, 1.0, delta);
        char nm[64];
        std::snprintf(nm, sizeof nm, "XXZ delta=%.1f imag->ground", delta);
        check(nm, std::abs(g.energy - esd.eigenvalues()(0)), 2e-3);
        // real-time Neel quench: energy is conserved (Neel energy = -delta*(n-1))
        auto q = run_tebd2_blocks(mpo, n, 0.02, 25, 16, 1e-12, /*imag*/false, 1, 1.0, delta);
        std::snprintf(nm, sizeof nm, "XXZ delta=%.1f real-time energy cons", delta);
        check(nm, std::abs(q.energy - (-delta*(n-1.0))), 1e-3);
        printf("  delta=%.1f  E_gs=%.6f exact=%.6f\n", delta, g.energy, esd.eigenvalues()(0));
    }

    // Loschmidt echo: quench from the XXZ(delta_i) ground state, evolve under
    // XXZ(delta_f); L(t) = |<psi0|psi(t)>|^2 vs exact diagonalization.
    printf("Loschmidt echo (quench delta 0.5 -> 1.5) vs exact:\n");
    {
        double di = 0.5, df = 1.5;
        Cm Hi = Cm::Zero(d,d), Hf = Cm::Zero(d,d);
        for (int i=0;i+1<n;++i) {
            Cm xy = emb(X,i)*emb(X,i+1) + emb(Y,i)*emb(Y,i+1);
            Cm zz = emb(Z,i)*emb(Z,i+1);
            Hi += xy + di*zz;  Hf += xy + df*zz;
        }
        Eigen::SelfAdjointEigenSolver<Cm> esi(Hi), esf(Hf);
        Cv psi0 = esi.eigenvectors().col(0);
        auto L_exact = [&](double t){
            Cv c = esf.eigenvectors().adjoint() * psi0;
            for (int k=0;k<d;++k) c(k) *= std::exp(Cx(0,-t)*esf.eigenvalues()(k));
            Cv psit = esf.eigenvectors() * c;
            return std::norm(psi0.dot(psit));
        };
        auto mpo_i = build_xxz_mpo(n, 1.0, di);
        auto lo = run_loschmidt_blocks(mpo_i, n, 0.05, 30, 32, 1e-12, 40, 1.0, df);
        double maxerr = 0;
        for (size_t k=0;k<lo.times.size();++k)
            maxerr = std::max(maxerr, std::abs(lo.L[k] - L_exact(lo.times[k])));
        check("Loschmidt L(t) vs exact (all t, TEBD Trotter)", maxerr, 5e-3);
        printf("  L(0)=%.4f (should be 1)  max_bond=%d  %.3fs\n", lo.L[0], lo.max_bond, lo.wall_time_s);
    }

    // Spin conductivity: current-current correlator C(t) = <psi0|J e^{-i(H-E0)t}J|psi0>
    // vs ED. Validates the new block MPO x MPS apply (phi = J|psi0>).
    printf("Spin conductivity C(t)=<J(t)J(0)> vs exact:\n");
    {
        double dl = 1.0, Jc = 1.0;
        Cm sp(2,2); sp<<0,1,0,0;  Cm sm(2,2); sm<<0,0,1,0;
        Cm Hc = Cm::Zero(d,d);
        for (int i=0;i+1<n;++i)
            Hc += emb(X,i)*emb(X,i+1) + emb(Y,i)*emb(Y,i+1) + dl*(emb(Z,i)*emb(Z,i+1));
        Eigen::SelfAdjointEigenSolver<Cm> esc(Hc);
        Cv psi0 = esc.eigenvectors().col(0); double E0c = esc.eigenvalues()(0);
        Cm Jop = Cm::Zero(d,d);          // total spin current
        for (int i=0;i+1<n;++i)
            Jop += Cx(0,Jc)*(emb(sp,i)*emb(sm,i+1) - emb(sm,i)*emb(sp,i+1));
        Cv phi = Jop * psi0;
        auto C_exact = [&](double t){
            Cv c = esc.eigenvectors().adjoint() * phi;
            Cx s=0; for (int k=0;k<d;++k) s += std::norm(c(k)) * std::exp(Cx(0,-(esc.eigenvalues()(k)-E0c))*t);
            return s;
        };
        auto mpo = build_xxz_mpo(n, Jc, dl);
        // current state J|psi0> has broad energy support -> faster dynamics, so a
        // small dt is needed to keep the O(dt^2) Trotter error down (it scales dt^2).
        auto cd = run_conductivity_blocks(mpo, n, 0.0125, 120, 32, 1e-12, 40, Jc, dl);
        double maxerr = 0;
        for (size_t k=0;k<cd.times.size();++k)
            maxerr = std::max(maxerr, std::abs(C_exact(cd.times[k]) - Cx(cd.C_re[k], cd.C_im[k])));
        check("conductivity C(t) vs exact (all t, TEBD Trotter)", maxerr, 5e-3);
        printf("  C(0)=%.4f (=<J^2>, exact=%.4f)  max_bond=%d  %.3fs\n",
               cd.C_re[0], C_exact(0.0).real(), cd.max_bond, cd.wall_time_s);
    }

    printf("\n%s\n", failures==0 ? "ALL PASSED" : "SOME FAILED");
    return failures==0 ? 0 : 1;
}
