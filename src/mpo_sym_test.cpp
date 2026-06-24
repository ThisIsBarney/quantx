// Universal-MPO phase 1a: validate the per-bond Sz-charge derivation
// (derive_mpo_sym) and the new J1-J2 model MPO.
//
// derive_mpo_sym is the foundation of running arbitrary Sz-conserving
// Hamiltonians on the fast block-sparse engine: it recovers, from the operator
// structure alone, the per-bond-index charge array the env / H_eff kernels need
// (currently hard-coded to the Heisenberg HB_DELTA={0,+2,-2,0,0}). This test
// proves it (a) reproduces HB_DELTA for the Heisenberg / XXZ / current MPOs,
// (b) yields the correct wider array for the 8-wide J1-J2 MPO, (c) rejects a
// non-Sz-conserving (transverse-field Ising) MPO, and (d) that the J1-J2 MPO
// itself is correct -- its dense-DMRG ground state matches exact diagonalization
// (the reference the phase-1b block path must then reproduce).
#include "dmrg_internal.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace qx_dmrg;

static bool check_charges(const std::vector<MPOSite>& mpo,
                          const std::vector<int>& expect, const char* name) {
    MPOSym sym = derive_mpo_sym(mpo);
    bool ok = (sym.chi == (int)expect.size()) && (sym.delta == expect);
    std::printf("  %-14s chi=%d delta=[", name, sym.chi);
    for (size_t i = 0; i < sym.delta.size(); ++i)
        std::printf("%s%d", i ? "," : "", sym.delta[i]);
    std::printf("]  %s\n", ok ? "OK" : "FAIL");
    if (!ok) {
        std::printf("    expected chi=%d [", (int)expect.size());
        for (size_t i = 0; i < expect.size(); ++i) std::printf("%s%d", i ? "," : "", expect[i]);
        std::printf("]\n");
    }
    return ok;
}

// dense Kron helpers for the J1-J2 ED reference
static Eigen::MatrixXcd kron(const Eigen::MatrixXcd& A, const Eigen::MatrixXcd& B) {
    Eigen::MatrixXcd K(A.rows()*B.rows(), A.cols()*B.cols());
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            K.block(i*B.rows(), j*B.cols(), B.rows(), B.cols()) = A(i,j)*B;
    return K;
}
static Eigen::MatrixXcd site_op(int n, int s, const Mat2& o) {
    Eigen::MatrixXcd O = Eigen::MatrixXcd::Identity(1,1);
    for (int i = 0; i < n; ++i) O = kron(O, (i==s)? Eigen::MatrixXcd(o) : Eigen::MatrixXcd(Mat2::Identity()));
    return O;
}

// general d-dim single-site operator embedding (d x d op at `site`, identity else)
static Eigen::MatrixXcd site_opd(int n, int site, const Eigen::MatrixXcd& o, int d) {
    Eigen::MatrixXcd O = Eigen::MatrixXcd::Identity(1, 1);
    for (int i = 0; i < n; ++i)
        O = kron(O, (i == site) ? o : Eigen::MatrixXcd(Eigen::MatrixXcd::Identity(d, d)));
    return O;
}

// Contract an MPO chain into the full d^n x d^n operator (boundary: site 0 row,
// last site col 0). Validates the MPO without needing a (not-yet-generalized for
// d>2) DMRG path.
static Eigen::MatrixXcd contract_mpo(const std::vector<qx_dmrg::MPOSite>& mpo) {
    std::vector<Eigen::MatrixXcd> partial;
    for (int a = 0; a < mpo[0].chi_r; ++a) partial.push_back(mpo[0].W(0, a));
    for (size_t i = 1; i < mpo.size(); ++i) {
        std::vector<Eigen::MatrixXcd> next(mpo[i].chi_r);
        for (int b = 0; b < mpo[i].chi_r; ++b) {
            Eigen::MatrixXcd acc; bool first = true;
            for (int a = 0; a < mpo[i].chi_l; ++a) {
                Eigen::MatrixXcd term = kron(partial[a], mpo[i].W(a, b));
                if (first) { acc = term; first = false; } else acc += term;
            }
            next[b] = acc;
        }
        partial = std::move(next);
    }
    return partial[0];
}

int main() {
    int fails = 0;
    std::printf("== derive_mpo_sym charge derivation ==\n");
    // Heisenberg / XXZ / current all share the 5-wide {0,+2,-2,0,0} structure.
    fails += !check_charges(build_xxz_mpo(6, 1.0, 1.0), {0,2,-2,0,0}, "Heisenberg");
    fails += !check_charges(build_xxz_mpo(6, 1.0, 0.5), {0,2,-2,0,0}, "XXZ(d=0.5)");
    fails += !check_charges(build_current_mpo(6, 1.0),   {0,2,-2,0,0}, "current");
    // J1-J2: 8-wide, each operator has a range-1 and a range-2 (delayed) state.
    fails += !check_charges(build_j1j2_mpo(6, 1.0, 0.5, 1.0), {0,2,2,-2,-2,0,0,0}, "J1-J2");
    // Spin-1 (d=3): same 5-wide BOND charges as spin-1/2 (S+- still shift Sz by 1),
    // but the local dimension and operators differ -- the first d>2 model.
    fails += !check_charges(build_spin1_heisenberg_mpo(6, 1.0), {0,2,-2,0,0}, "spin-1");

    // Arbitrary local dimension (phase 2a): the generalized MPOSite (d, phys_charge,
    // d x d ops) holds the spin-1 (d=3) MPO and it encodes the correct Hamiltonian.
    std::printf("== spin-1 (d=3) MPO reconstructs H = J sum S.S ==\n");
    {
        const int n = 4; const double J = 1.0; const double r2 = std::sqrt(2.0);
        Eigen::MatrixXcd Sp = Eigen::MatrixXcd::Zero(3,3); Sp(0,1)=r2; Sp(1,2)=r2;
        Eigen::MatrixXcd Sm = Eigen::MatrixXcd::Zero(3,3); Sm(1,0)=r2; Sm(2,1)=r2;
        Eigen::MatrixXcd Sz = Eigen::MatrixXcd::Zero(3,3); Sz(0,0)=1; Sz(2,2)=-1;
        Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(81, 81);   // 3^4
        for (int i = 0; i < n-1; ++i)
            H += J * (0.5*(site_opd(n,i,Sp,3)*site_opd(n,i+1,Sm,3)
                         + site_opd(n,i,Sm,3)*site_opd(n,i+1,Sp,3))
                    + site_opd(n,i,Sz,3)*site_opd(n,i+1,Sz,3));
        Eigen::MatrixXcd H_mpo = contract_mpo(build_spin1_heisenberg_mpo(n, J));
        double err = (H - H_mpo).cwiseAbs().maxCoeff();
        bool ok = err < 1e-12;
        // ED ground energy (sanity: the spin-1 Heisenberg GS is gapped/Haldane)
        double E_ed = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd>(H_mpo).eigenvalues()(0);
        std::printf("  N=%d: ||H_MPO - H_direct||=%.2e  E_GS=%.6f  %s\n",
                    n, err, E_ed, ok ? "OK" : "FAIL");
        fails += !ok;
    }

    // Phase 2b/2c: the dense AND block-sparse DMRG paths now run arbitrary local
    // dimension d. The spin-1 (d=3) Heisenberg ground state must match exact
    // diagonalization on both (the gapped Haldane phase). For N=4,6 the bond stays
    // below truncation so DMRG is effectively exact.
    std::printf("== spin-1 (d=3) DMRG vs ED (dense + block) ==\n");
    {
        const double r2 = std::sqrt(2.0);
        Eigen::MatrixXcd Sp = Eigen::MatrixXcd::Zero(3,3); Sp(0,1)=r2; Sp(1,2)=r2;
        Eigen::MatrixXcd Sm = Eigen::MatrixXcd::Zero(3,3); Sm(1,0)=r2; Sm(2,1)=r2;
        Eigen::MatrixXcd Sz = Eigen::MatrixXcd::Zero(3,3); Sz(0,0)=1; Sz(2,2)=-1;
        auto spin1_ed = [&](int n) {
            int dim = 1; for (int k=0;k<n;++k) dim*=3;
            Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(dim, dim);
            for (int i = 0; i < n-1; ++i)
                H += 0.5*(site_opd(n,i,Sp,3)*site_opd(n,i+1,Sm,3)
                        + site_opd(n,i,Sm,3)*site_opd(n,i+1,Sp,3))
                   + site_opd(n,i,Sz,3)*site_opd(n,i+1,Sz,3);
            return Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd>(H).eigenvalues()(0);
        };
        for (int n : {4, 6}) {
            double E_ed = spin1_ed(n);
            auto mpo = build_spin1_heisenberg_mpo(n, 1.0);
            double E_dense = run_dmrg(mpo, n, 64, 1e-11, 80, 1e-11).energy;
            double E_block = run_dmrg_blocks(mpo, n, 64, 1e-11, 80, 1e-11).energy;
            double eD = std::abs(E_dense - E_ed), eB = std::abs(E_block - E_ed);
            bool ok = eD < 1e-6 && eB < 1e-6;
            std::printf("  N=%d : E(ED)=%.8f  dense err=%.2e  block err=%.2e  %s\n",
                        n, E_ed, eD, eB, ok ? "OK" : "FAIL");
            fails += !ok;
        }
    }

    // A non-Sz-conserving MPO (transverse-field Ising bond H=sum X_i X_{i+1})
    // must be rejected: the X completion bond carries both +2 and -2.
    std::printf("== rejects non-Sz-conserving MPO ==\n");
    {
        Mat2 X = pauli('X'), I = pauli('I');
        std::array<std::array<Mat2,3>,3> W{};
        for (auto& r : W) for (auto& e : r) e.setZero();
        W[0][0]=I; W[2][2]=I; W[2][1]=X; W[1][0]=X;   // start=2, done=0
        int n=4; std::vector<MPOSite> tfim(n);
        for (int i=0;i<n;++i){ int cl=(i==0)?1:3, cr=(i==n-1)?1:3;
            tfim[i].chi_l=cl; tfim[i].chi_r=cr; tfim[i].ops.assign(cl*cr, Mat2::Zero());
            for(int a=0;a<cl;++a)for(int b=0;b<cr;++b){int ar=(i==0)?2:a,br=(i==n-1)?0:b; tfim[i].W(a,b)=W[ar][br];}}
        bool threw=false;
        try { derive_mpo_sym(tfim); } catch (const std::exception&) { threw=true; }
        std::printf("  TFIM bond  threw=%d  %s\n", (int)threw, threw?"OK":"FAIL");
        fails += !threw;
    }
    // A chain too short to have a square bulk site (n=2) must be rejected, not
    // read out of bounds.
    {
        bool threw=false;
        try { derive_mpo_sym(build_xxz_mpo(2, 1.0, 1.0)); } catch (const std::exception&) { threw=true; }
        std::printf("  n=2 (no bulk) threw=%d  %s\n", (int)threw, threw?"OK":"FAIL");
        fails += !threw;
    }
    // Phase 2c-dynamics: block TDVP2 now runs d>2. Spin-1 (d=3) checks: real-time
    // TDVP2 conserves energy (the Neel product state has bond energy <S^z S^z> = -1,
    // so <H> = -(N-1)); imaginary-time TDVP2 cools the Neel start to the GS.
    {
        const int N = 6; const double E_neel = -(N - 1);
        auto s1 = build_spin1_heisenberg_mpo(N, 1.0);
        double E_gs = run_dmrg_blocks(s1, N, 48, 1e-11, 80, 1e-11).energy;   // == ED
        double E_rt = run_tdvp2_blocks(s1, N, 0.05, 8, 24, 1e-12, /*imag*/false).energy;
        double E_it = run_tdvp2_blocks(s1, N, 0.1, 400, 24, 1e-12, /*imag*/true).energy;
        bool okrt = std::abs(E_rt - E_neel) < 1e-6;
        bool okit = std::abs(E_it - E_gs)   < 1e-3;
        std::printf("  spin-1 TDVP2 real-time <H>=%.6f (=%.0f) %s ; imag->GS=%.6f (ED %.6f) %s\n",
                    E_rt, E_neel, okrt?"OK":"FAIL", E_it, E_gs, okit?"OK":"FAIL");
        fails += !(okrt && okit);
    }
    // Spin-1 (d=3) TEBD: the gate is built from the MPO's 2-site bond Hamiltonian
    // (exp(-dt h)). Imaginary-time cooling from the Neel start reaches the GS (with
    // a small Trotter error ~dt^2).
    {
        const int N = 6;
        auto s1 = build_spin1_heisenberg_mpo(N, 1.0);
        double E_gs = run_dmrg_blocks(s1, N, 48, 1e-11, 80, 1e-11).energy;   // == ED
        double E_te = run_tebd2_blocks(s1, N, 0.02, 1000, 24, 1e-12, /*imag*/true, 0, 1.0, 1.0).energy;
        bool ok = std::abs(E_te - E_gs) < 2e-3;
        std::printf("  spin-1 TEBD imag->GS=%.6f (ED %.6f)  %s\n",
                    E_te, E_gs, ok ? "OK" : "FAIL");
        fails += !ok;
    }

    // Spin-1 (d=3) finite temperature via ancilla purification: cool the beta=0
    // maximally-entangled state with the d>2 swap-and-carry gates, and the thermal
    // energy <H>_beta must match the exact-diagonalization Gibbs average
    // sum_i E_i e^{-beta E_i} / sum_i e^{-beta E_i}.
    std::printf("== spin-1 (d=3) finite-T via purification vs ED Gibbs ==\n");
    {
        const int n = 4; const double beta = 1.0;
        const double r2 = std::sqrt(2.0);
        Eigen::MatrixXcd Sp = Eigen::MatrixXcd::Zero(3,3); Sp(0,1)=r2; Sp(1,2)=r2;
        Eigen::MatrixXcd Sm = Eigen::MatrixXcd::Zero(3,3); Sm(1,0)=r2; Sm(2,1)=r2;
        Eigen::MatrixXcd Sz = Eigen::MatrixXcd::Zero(3,3); Sz(0,0)=1; Sz(2,2)=-1;
        Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(81, 81);
        for (int i = 0; i < n-1; ++i)
            H += 0.5*(site_opd(n,i,Sp,3)*site_opd(n,i+1,Sm,3)
                    + site_opd(n,i,Sm,3)*site_opd(n,i+1,Sp,3))
               + site_opd(n,i,Sz,3)*site_opd(n,i+1,Sz,3);
        Eigen::VectorXd ev = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd>(H).eigenvalues();
        double num = 0, den = 0, e0 = ev(0);   // shift by e0 for numerical stability
        for (int i = 0; i < ev.size(); ++i) { double w = std::exp(-beta*(ev(i)-e0)); num += ev(i)*w; den += w; }
        double E_ed = num / den;
        // purified block thermal: dbeta=0.05 -> 20 steps reach beta=1.0
        auto th = run_thermal_blocks(n, 1.0, 1.0, 0.05, 20, 64, 1e-12, 1, false, /*d*/3);
        double E_th = th.E.back();
        double err = std::abs(E_th - E_ed);
        bool ok = err < 5e-3;
        std::printf("  N=%d beta=%.1f : <H>_ED=%.6f  purified=%.6f  err=%.1e  %s\n",
                    n, beta, E_ed, E_th, err, ok ? "OK" : "FAIL");
        fails += !ok;
    }

    // J1-J2 ground state: exact diagonalization vs both the dense DMRG (MPO-
    // agnostic) and -- the phase-1b payoff -- the FAST block-sparse path, which
    // now consumes the derived 8-wide charge array instead of the hard-coded
    // Heisenberg HB_DELTA. Both must match ED.
    std::printf("== J1-J2 ground state: ED vs dense DMRG vs block DMRG ==\n");
    auto j1j2_check = [&](int n, double J1, double J2, double d) {
        Mat2 Xp=pauli('X'), Yp=pauli('Y'), Zp=pauli('Z');
        Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(1<<n, 1<<n);
        for (int i=0;i<n;++i)
            for (int r=1;r<=2;++r){ int j=i+r; if(j>=n) continue; double Jr=(r==1)?J1:J2;
                H += Jr*(site_op(n,i,Xp)*site_op(n,j,Xp) + site_op(n,i,Yp)*site_op(n,j,Yp)
                       + d*site_op(n,i,Zp)*site_op(n,j,Zp)); }
        double E_ed = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd>(H).eigenvalues()(0);
        auto mpo = build_j1j2_mpo(n, J1, J2, d);
        double E_dense = run_dmrg(mpo, n, 96, 1e-11, 80, 1e-11).energy;
        double E_block = run_dmrg_blocks(mpo, n, 96, 1e-11, 80, 1e-11).energy;
        double eD = std::abs(E_dense - E_ed), eB = std::abs(E_block - E_ed);
        bool ok = eD < 1e-6 && eB < 1e-6;
        std::printf("  N=%d J2=%.2f d=%.2f : E(ED)=%.8f  dense err=%.2e  block err=%.2e  %s\n",
                    n, J2, d, E_ed, eD, eB, ok ? "OK" : "FAIL");
        fails += !ok;
    };
    j1j2_check(8,  1.0, 0.5, 1.0);
    j1j2_check(10, 1.0, 0.4, 0.8);   // different J2 / anisotropy / size

    // Majumdar-Ghosh degeneracy regression: at J2=J1/2 the open-chain ground state
    // is the exact dimer product (E/N = -1.5). Under truncation the Gram-trick SVD
    // squares the condition number and, at this degenerate Schmidt spectrum,
    // amplified the ~1e-15 FP differences of multi-threaded GEMMs into a
    // non-orthonormal (non-canonical) state -> a NON-VARIATIONAL energy below the
    // true GS, varying run-to-run. The direct (BDCSVD) SVD on the DMRG path keeps
    // the state canonical: the energy must equal -1.5 and be deterministic. (This
    // test runs multi-threaded, so it exercises exactly the failing condition;
    // chi=192 > the level where the Gram trick used to break.)
    std::printf("== Majumdar-Ghosh degeneracy (stable-SVD regression, multi-threaded) ==\n");
    {
        auto mpo = build_j1j2_mpo(24, 1.0, 0.5, 1.0);
        double e1 = run_dmrg_blocks(mpo, 24, 192, 1e-11, 50, 1e-11).energy / 24;
        double e2 = run_dmrg_blocks(mpo, 24, 192, 1e-11, 50, 1e-11).energy / 24;
        bool ok = std::abs(e1 - (-1.5)) < 1e-5 && e1 == e2;   // exact GS, deterministic, variational
        std::printf("  N=24 chi=192: E/N=%.8f (exact -1.5)  deterministic=%d  %s\n",
                    e1, (int)(e1 == e2), ok ? "OK" : "FAIL");
        fails += !ok;
    }

    std::printf("%s\n", fails ? "MPO SYM TEST: FAIL" : "MPO SYM TEST: ALL OK");
    return fails ? 1 : 0;
}
