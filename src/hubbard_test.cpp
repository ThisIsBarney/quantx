// Phase 3a: the fermionic Hubbard chain (d=4) on the universal-MPO engine.
//
// The block/DMRG engine is bosonic (it only knows a U(1) charge, no anticommutation).
// The Fermi statistics are carried entirely by the MPO via Jordan-Wigner: every
// operator that travels along an MPO bond is dressed with the on-site parity
// F = (-1)^{n_up+n_dn}, and the on-site annihilators carry the intra-site up-before-dn
// sign. This test proves the construction is correct by comparing against an EXPLICIT
// Jordan-Wigner exact diagonalization (fermion operators built as
// c_{i s} = (F ⊗ ... ⊗ F ⊗ a_s ⊗ I ⊗ ... ), the unambiguous ground truth):
//   (a) contract_mpo(build_hubbard_mpo) == the explicit fermionic H  (signs are right),
//   (b) derive_mpo_sym reads the particle-number U(1) charges cleanly,
//   (c) the dense DMRG ground state == ED (the engine runs the fermionic MPO),
//   (d) the 2-site value matches the analytic E = (U - sqrt(U^2+16 t^2))/2.
#include "dmrg_internal.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <map>
#include <tuple>
#include <vector>

using namespace qx_dmrg;
using Cxd = std::complex<double>;
using MatC = Eigen::MatrixXcd;

static MatC kron(const MatC& A, const MatC& B) {
    MatC K(A.rows()*B.rows(), A.cols()*B.cols());
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            K.block(i*B.rows(), j*B.cols(), B.rows(), B.cols()) = A(i,j)*B;
    return K;
}

// Contract an MPO chain into the full d^n x d^n operator (boundary: site 0 row 0,
// last site col 0) -- plain (bosonic) kron, so it reproduces the fermionic H only
// because the MPO already carries the Jordan-Wigner signs.
static MatC contract_mpo(const std::vector<MPOSite>& mpo) {
    std::vector<MatC> partial;
    for (int a = 0; a < mpo[0].chi_r; ++a) partial.push_back(mpo[0].W(0, a));
    for (size_t i = 1; i < mpo.size(); ++i) {
        std::vector<MatC> next(mpo[i].chi_r);
        for (int b = 0; b < mpo[i].chi_r; ++b) {
            MatC acc; bool first = true;
            for (int a = 0; a < mpo[i].chi_l; ++a) {
                MatC term = kron(partial[a], mpo[i].W(a, b));
                if (first) { acc = term; first = false; } else acc += term;
            }
            next[b] = acc;
        }
        partial = std::move(next);
    }
    return partial[0];
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);   // unbuffered: last line printed = where a hang is
    int fails = 0;

    // On-site (d=4) operators, basis {|0>,|up>,|dn>,|up dn>}, |up dn>=c^dag_up c^dag_dn|0>.
    MatC I4 = MatC::Identity(4,4);
    MatC F  = MatC::Zero(4,4); F(0,0)=1; F(1,1)=-1; F(2,2)=-1; F(3,3)=1;
    MatC Cup = MatC::Zero(4,4); Cup(0,1)=1; Cup(2,3)=1;
    MatC Cdn = MatC::Zero(4,4); Cdn(0,2)=1; Cdn(1,3)=-1;   // on-site JW sign
    MatC Nu = MatC::Zero(4,4); Nu(1,1)=1; Nu(3,3)=1;
    MatC Nd = MatC::Zero(4,4); Nd(2,2)=1; Nd(3,3)=1;

    // Explicit Jordan-Wigner fermion operator: c_{i s} = F ⊗ ... ⊗ F ⊗ a_s ⊗ I ⊗ ...
    // (F on every site left of i). a_up=Cup, a_dn=Cdn already carry the intra-site sign.
    auto ferm = [&](int n, int i, const MatC& a) {
        MatC O = MatC::Identity(1,1);
        for (int k = 0; k < n; ++k)
            O = kron(O, k < i ? F : (k == i ? a : I4));
        return O;
    };
    // Explicit fermionic Hubbard Hamiltonian (the ground truth). Memoised: the 4^n
    // matrices (n=6 -> 4096x4096) are reused by the dense and the half-filling checks.
    static std::map<std::tuple<int,long long,long long>, MatC> H_cache;
    auto hubbard_ed_H = [&](int n, double t, double U) -> const MatC& {
        auto key = std::make_tuple(n, std::llround(t*1e6), std::llround(U*1e6));
        if (auto it = H_cache.find(key); it != H_cache.end()) return it->second;
        int dim = 1; for (int k=0;k<n;++k) dim *= 4;
        MatC H = MatC::Zero(dim, dim);
        for (int i = 0; i < n-1; ++i) {
            MatC hop = ferm(n,i,Cup.adjoint())*ferm(n,i+1,Cup)
                     + ferm(n,i,Cdn.adjoint())*ferm(n,i+1,Cdn);
            H += -t * (hop + hop.adjoint());
        }
        for (int i = 0; i < n; ++i)
            H += U * (ferm(n,i,Nu) * ferm(n,i,Nd));   // on-site U n_up n_dn (diagonal)
        return H_cache.emplace(key, std::move(H)).first->second;
    };
    // ED restricted to the half-filling sector (total particle number N = n): the block
    // DMRG conserves only N (single U(1)), so its half-filling GS must match the lowest
    // eigenvalue of H projected onto the N=n subspace (all Sz within it).
    auto hubbard_ed_halffill = [&](int n, double t, double U) {
        const MatC& H = hubbard_ed_H(n, t, U);
        int dim = (int)H.rows();
        // Total particle number of basis state k = sum over sites of the local n
        // ({0,1,1,2} for {|0>,|up>,|dn>,|up dn>}). Computed directly from the base-4
        // digits -- NOT via ferm(), whose Jordan-Wigner F-strings would corrupt this
        // (even/bosonic) count. Select the half-filling (N=n) subspace.
        const int nof[4] = {0, 1, 1, 2};
        std::vector<int> idx;
        for (int k = 0; k < dim; ++k) {
            int Nt = 0, kk = k;
            for (int i = 0; i < n; ++i) { Nt += nof[kk % 4]; kk /= 4; }
            if (Nt == n) idx.push_back(k);
        }
        int m = (int)idx.size();
        MatC Hs(m, m);
        for (int a = 0; a < m; ++a) for (int b = 0; b < m; ++b) Hs(a,b) = H(idx[a], idx[b]);
        return Eigen::SelfAdjointEigenSolver<MatC>(Hs).eigenvalues()(0);
    };

    // (a) MPO reconstructs the fermionic H exactly, and (d) the 2-site analytic value.
    std::printf("== Hubbard (d=4) MPO reconstructs the Jordan-Wigner H ==\n");
    for (auto [t, U] : std::vector<std::pair<double,double>>{{1.0,0.0},{1.0,4.0},{1.0,8.0},{0.7,2.5}}) {
        for (int n : {2, 3, 4}) {
            MatC H_ed  = hubbard_ed_H(n, t, U);
            MatC H_mpo = contract_mpo(build_hubbard_mpo(n, t, U));
            double err = (H_ed - H_mpo).cwiseAbs().maxCoeff();
            bool ok = err < 1e-12;
            double E0 = Eigen::SelfAdjointEigenSolver<MatC>(H_ed).eigenvalues()(0);
            std::printf("  t=%.1f U=%.1f N=%d : ||H_MPO-H_JW||=%.1e  E_GS=%.6f  %s",
                        t, U, n, err, E0, ok ? "OK" : "FAIL");
            if (n == 2) {   // analytic 2-site GLOBAL ground state: the half-filling singlet
                // (U-sqrt(U^2+16t^2))/2 for U<3t, else the single-electron bonding state -t
                // (repulsion pushes the global minimum off half-filling past U=3t).
                double Ean = std::min(0.5 * (U - std::sqrt(U*U + 16.0*t*t)), -t);
                bool oka = std::abs(E0 - Ean) < 1e-9;
                std::printf("  [2-site analytic %.6f %s]", Ean, oka ? "OK" : "FAIL");
                fails += !oka;
            }
            std::printf("\n");
            fails += !ok;
        }
    }

    // (b) derive_mpo_sym reads the particle-number U(1) charges (must not throw).
    std::printf("== Hubbard MPO is N-conserving (derive_mpo_sym) ==\n");
    {
        bool ok = true; std::vector<int> delta;
        try { auto sym = derive_mpo_sym(build_hubbard_mpo(6, 1.0, 4.0)); delta = sym.delta; }
        catch (const std::exception& e) { ok = false; std::printf("  threw: %s\n", e.what()); }
        std::printf("  chi=%d delta=[", (int)delta.size());
        for (size_t i=0;i<delta.size();++i) std::printf("%s%d", i?",":"", delta[i]);
        std::printf("]  %s\n", ok ? "OK" : "FAIL");
        fails += !ok;
    }

    // (c) the dense DMRG (arbitrary-d) ground state matches ED -- the engine runs the
    // fermionic MPO. (Dense DMRG finds the global GS, so no particle-number targeting
    // is needed here; the block-sparse path with N sectors is the next phase.)
    std::printf("== Hubbard dense DMRG vs ED ==\n");
    for (auto [t, U] : std::vector<std::pair<double,double>>{{1.0,0.0},{1.0,4.0},{1.0,8.0}}) {
        for (int n : {2, 4}) {   // n<=4: brute ED (dense 4^n operator products) stays cheap
            double E_ed = Eigen::SelfAdjointEigenSolver<MatC>(hubbard_ed_H(n,t,U)).eigenvalues()(0);
            auto mpo = build_hubbard_mpo(n, t, U);
            double E_dense = run_dmrg(mpo, n, 128, 1e-11, 120, 1e-11).energy;
            double err = std::abs(E_dense - E_ed);
            bool ok = err < 1e-6;
            std::printf("  t=%.1f U=%.1f N=%d : E(ED)=%.8f  dense err=%.2e  %s\n",
                        t, U, n, E_ed, err, ok ? "OK" : "FAIL");
            fails += !ok;
        }
    }

    // (e) THE PAYOFF: the fast block-sparse DMRG runs the fermionic Hubbard MPO at
    // half-filling. The block engine conserves the particle-number U(1) (phys_charge
    // = {0,1,1,2}); we target the N=n sector and start from the singly-occupied
    // |up dn up dn ...> product (SDW-like, close to the repulsive-Hubbard GS).
    std::printf("== Hubbard FAST block-sparse DMRG at half-filling (N=n) vs sector-ED ==\n");
    for (auto [t, U] : std::vector<std::pair<double,double>>{{1.0,0.0},{1.0,4.0},{1.0,8.0}}) {
        for (int n : {4}) {   // block path needs n>=3; n=4 exercises sectors/SDW-init/JW signs
            double E_sec = hubbard_ed_halffill(n, t, U);
            std::vector<int> cfg(n);
            for (int i = 0; i < n; ++i) cfg[i] = (i % 2 == 0) ? 1 : 2;   // |up dn up dn ...>
            auto mpo = build_hubbard_mpo(n, t, U);
            double E_blk = run_dmrg_blocks(mpo, n, 128, 1e-11, 120, 1e-11,
                                           /*total_charge*/ n, &cfg).energy;
            double err = std::abs(E_blk - E_sec);
            bool ok = err < 1e-6;
            std::printf("  t=%.1f U=%.1f N=%d(half) : E(sector-ED)=%.8f  block err=%.2e  %s\n",
                        t, U, n, E_sec, err, ok ? "OK" : "FAIL");
            fails += !ok;
        }
    }

    // (f) DYNAMICS on the fast block path: real-time TDVP2 from the charge-density-wave
    // |0,up dn,0,up dn,...> (the default Neel of charges {0,2} for Hubbard = half-filling).
    // The staggered charge s(t) = (1/n) sum_i (-1)^i <n_i> melts as the doublons spread;
    // it must track ED e^{-iHt}. Real-time also conserves energy; imaginary time cools the
    // CDW to the half-filling ground state. The Fermi statistics ride in the same MPO, so
    // TDVP2 (which applies the MPO's H_eff) runs the fermionic dynamics unchanged.
    std::printf("== Hubbard DYNAMICS: block TDVP2 (real-time vs ED + imag->GS) ==\n");
    const int nof[4] = {0, 1, 1, 2};
    for (auto [t, U] : std::vector<std::pair<double,double>>{{1.0,4.0},{1.0,8.0}}) {
        const int n = 4;
        auto mpo = build_hubbard_mpo(n, t, U);
        const MatC& H = hubbard_ed_H(n, t, U);
        int dim = (int)H.rows();
        Eigen::SelfAdjointEigenSolver<MatC> es(H);
        Eigen::VectorXd Eev = es.eigenvalues(); MatC V = es.eigenvectors();
        // CDW basis index (site 0 = most significant base-4 digit) and each state's
        // staggered particle number sum_i (-1)^i n_i.
        int cdw = 0; for (int i = 0; i < n; ++i) cdw = cdw*4 + (i%2==0 ? 0 : 3);
        std::vector<double> stagN(dim, 0.0);
        for (int k = 0; k < dim; ++k) { int kk = k;
            for (int i = n-1; i >= 0; --i) { stagN[k] += ((i%2==0)?1.0:-1.0) * nof[kk%4]; kk /= 4; } }
        Eigen::VectorXcd amp = V.adjoint().col(cdw);   // <eigenstate | CDW>
        auto stag_ed = [&](double tt) {
            Eigen::VectorXcd psit = Eigen::VectorXcd::Zero(dim);
            for (int b = 0; b < dim; ++b) psit += std::exp(Cxd(0,-Eev[b]*tt)) * amp[b] * V.col(b);
            double s = 0; for (int k = 0; k < dim; ++k) s += std::norm(psit[k]) * stagN[k];
            return s / n;
        };
        // real-time TDVP2 (dt small, chi=128 -> no truncation at n=4 -> Trotter-limited)
        const double dt = 0.05; const int nst = 12;
        auto rt = run_tdvp2_blocks(mpo, n, dt, nst, 128, 1e-12, /*imag*/false);
        double smax = 0;
        for (size_t k = 0; k < rt.stag.size(); ++k)
            smax = std::max(smax, std::abs(rt.stag[k] - stag_ed(rt.times[k])));
        double E_init = 2.0 * U;                 // <CDW|H|CDW> = U * (#doubly-occ) = 2U (n=4)
        bool ok_rt = smax < 5e-3 && std::abs(rt.energy - E_init) < 1e-3;
        // imaginary-time TDVP2 cools the CDW to the half-filling GS
        double E_gs = hubbard_ed_halffill(n, t, U);
        double E_it = run_tdvp2_blocks(mpo, n, 0.1, 400, 32, 1e-12, /*imag*/true).energy;
        bool ok_it = std::abs(E_it - E_gs) < 1e-4;
        std::printf("  t=%.1f U=%.1f N=%d : real-t max|s_block-s_ED|=%.1e  E_cons=%.4f(=%.1f) %s ;"
                    " imag->GS=%.6f (ED %.6f) %s\n",
                    t, U, n, smax, rt.energy, E_init, ok_rt?"OK":"FAIL", E_it, E_gs, ok_it?"OK":"FAIL");
        fails += !(ok_rt && ok_it);
    }

    std::printf("%s\n", fails ? "HUBBARD TEST: FAIL" : "HUBBARD TEST: ALL OK");
    return fails ? 1 : 0;
}
