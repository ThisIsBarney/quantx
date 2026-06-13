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

    // J1-J2 MPO correctness: dense DMRG vs exact diagonalization (N=8).
    std::printf("== J1-J2 MPO: dense DMRG vs ED (N=8) ==\n");
    {
        const int n=8; const double J1=1.0, J2=0.5, d=1.0;
        Mat2 Xp=pauli('X'), Yp=pauli('Y'), Zp=pauli('Z');
        Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(1<<n, 1<<n);
        for (int i=0;i<n;++i){
            for (int r=1;r<=2;++r){ int j=i+r; if(j>=n) continue; double Jr=(r==1)?J1:J2;
                H += Jr*(site_op(n,i,Xp)*site_op(n,j,Xp) + site_op(n,i,Yp)*site_op(n,j,Yp)
                       + d*site_op(n,i,Zp)*site_op(n,j,Zp)); } }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
        double E_ed = es.eigenvalues()(0);
        auto r = run_dmrg(build_j1j2_mpo(n, J1, J2, d), n, /*chi*/64, /*cutoff*/1e-10,
                          /*sweeps*/60, /*tol*/1e-10);
        double err = std::abs(r.energy - E_ed);
        bool ok = err < 1e-6;
        std::printf("  E(dense DMRG)=%.8f  E(ED)=%.8f  err=%.2e  %s\n",
                    r.energy, E_ed, err, ok?"OK":"FAIL");
        fails += !ok;
    }

    std::printf("%s\n", fails ? "MPO SYM TEST: FAIL" : "MPO SYM TEST: ALL OK");
    return fails ? 1 : 0;
}
