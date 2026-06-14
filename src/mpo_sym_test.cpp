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

int main() {
    int fails = 0;
    std::printf("== derive_mpo_sym charge derivation ==\n");
    // Heisenberg / XXZ / current all share the 5-wide {0,+2,-2,0,0} structure.
    fails += !check_charges(build_xxz_mpo(6, 1.0, 1.0), {0,2,-2,0,0}, "Heisenberg");
    fails += !check_charges(build_xxz_mpo(6, 1.0, 0.5), {0,2,-2,0,0}, "XXZ(d=0.5)");
    fails += !check_charges(build_current_mpo(6, 1.0),   {0,2,-2,0,0}, "current");
    // J1-J2: 8-wide, each operator has a range-1 and a range-2 (delayed) state.
    fails += !check_charges(build_j1j2_mpo(6, 1.0, 0.5, 1.0), {0,2,2,-2,-2,0,0,0}, "J1-J2");

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
