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
#include <cstdlib>
#include <map>
#include <tuple>
#include <vector>

using namespace qx_dmrg;

// Set/clear an environment variable (portable): run_dmrg_blocks reads
// QX_DMRG_NO_SINGLESITE per call, so the test can toggle the hybrid path.
static void set_env(const char* k, const char* v) {
#ifdef _WIN32
    _putenv_s(k, v ? v : "");   // empty value removes the variable on Windows
#else
    if (v) setenv(k, v, 1); else unsetenv(k);
#endif
}
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
            // Packed two-U(1) (N and 2Sz conserved): same Hamiltonian, finer blocks.
            // The half-filling GS is a singlet, so targeting pack(n, 0) must reproduce
            // the same sector-ED energy through entirely different block structures.
            auto mpo2 = build_hubbard_mpo(n, t, U, /*two_u1*/true);
            double E_2u1 = run_dmrg_blocks(mpo2, n, 128, 1e-11, 120, 1e-11,
                                           pack_q2(n, 0), &cfg).energy;
            double err2 = std::abs(E_2u1 - E_sec);
            bool ok = err < 1e-6 && err2 < 1e-6;
            std::printf("  t=%.1f U=%.1f N=%d(half) : E(sector-ED)=%.8f  block err=%.2e  "
                        "2U1 err=%.2e  %s\n",
                        t, U, n, E_sec, err, err2, ok ? "OK" : "FAIL");
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

        // TEBD (Trotter gates from the MPO): the first model whose gate needs the
        // on-site term (U n_up n_dn = W(start,done)) split with boundary weights --
        // 1/2 on interior bonds, 1 at the chain ends (blind carry-channel summing
        // would double-count U on interior sites). Real-time tracks ED to Trotter
        // accuracy O(dt^2); imaginary-time cools the CDW to the half-filling GS.
        auto te = run_tebd2_blocks(mpo, n, dt, nst, 128, 1e-12, /*imag*/false, 1, 1.0, 1.0);
        double tmax = 0;
        for (size_t k = 0; k < te.stag.size(); ++k)
            tmax = std::max(tmax, std::abs(te.stag[k] - stag_ed(te.times[k])));
        bool ok_te = tmax < 5e-3 && std::abs(te.energy - E_init) < 2e-2;
        double E_tit = run_tebd2_blocks(mpo, n, 0.02, 1000, 24, 1e-12, /*imag*/true, 0, 1.0, 1.0).energy;
        bool ok_tit = std::abs(E_tit - E_gs) < 2e-3;
        std::printf("            TEBD : real-t max|s-s_ED|=%.1e  E=%.4f(=%.1f) %s ;"
                    " imag->GS=%.6f (ED %.6f) %s\n",
                    tmax, te.energy, E_init, ok_te?"OK":"FAIL", E_tit, E_gs, ok_tit?"OK":"FAIL");
        fails += !(ok_te && ok_tit);

        // Packed two-U(1) dynamics: identical physics through the finer (N,2Sz) block
        // structure. The CDW start and the staggered <n_i> observable are symmetry-
        // agnostic (obs_of_charge decodes the packed N), so both TDVP2 and TEBD must
        // reproduce the same ED time series; imag-time TEBD must reach the same GS.
        auto mpo2 = build_hubbard_mpo(n, t, U, /*two_u1*/true);
        auto rt2 = run_tdvp2_blocks(mpo2, n, dt, nst, 128, 1e-12, /*imag*/false);
        double smax2 = 0;
        for (size_t k = 0; k < rt2.stag.size(); ++k)
            smax2 = std::max(smax2, std::abs(rt2.stag[k] - stag_ed(rt2.times[k])));
        auto te2 = run_tebd2_blocks(mpo2, n, dt, nst, 128, 1e-12, /*imag*/false, 1, 1.0, 1.0);
        double tmax2 = 0;
        for (size_t k = 0; k < te2.stag.size(); ++k)
            tmax2 = std::max(tmax2, std::abs(te2.stag[k] - stag_ed(te2.times[k])));
        double E_tit2 = run_tebd2_blocks(mpo2, n, 0.02, 1000, 24, 1e-12, /*imag*/true, 0, 1.0, 1.0).energy;
        bool ok2 = smax2 < 5e-3 && tmax2 < 5e-3 && std::abs(rt2.energy - E_init) < 1e-3
                   && std::abs(E_tit2 - E_gs) < 2e-3;
        std::printf("            2U1  : TDVP2 max|s-s_ED|=%.1e  TEBD=%.1e  E=%.4f(=%.1f) ;"
                    " TEBD imag->GS=%.6f  %s\n",
                    smax2, tmax2, rt2.energy, E_init, E_tit2, ok2?"OK":"FAIL");
        fails += !ok2;

        // T=0 dynamical structure factors on the packed path: the spin and charge
        // correlators C^W_{j,l0}(t) = <gs| W_j e^{-i(H-E0)t} W_{l0} |gs> for diagonal
        // W (spin 2Sz = {0,+1,-1,0}; charge fluctuation n-1 = {-1,0,0,+1}) -- the raw
        // data of S_spin(k,w) / S_charge(k,w), i.e. spin-charge separation. ED
        // reference restricted to the half-filling sector (diagonal W stays in it).
        {
            const int l0 = n / 2;
            std::vector<int> idx;
            for (int k = 0; k < dim; ++k) { int Nt=0, kk=k;
                for (int i2=0;i2<n;++i2){ Nt += nof[kk%4]; kk/=4; } if (Nt==n) idx.push_back(k); }
            int m = (int)idx.size();
            MatC Hs(m, m);
            for (int a=0;a<m;++a) for (int b=0;b<m;++b) Hs(a,b) = H(idx[a], idx[b]);
            Eigen::SelfAdjointEigenSolver<MatC> ess(Hs);
            Eigen::VectorXd Es = ess.eigenvalues(); MatC Vs = ess.eigenvectors();
            // site-j digit of sector basis state a (site 0 = most significant digit)
            auto digit = [&](int a, int j){ int kk = idx[a];
                for (int p = n-1; p > j; --p) kk /= 4; return kk % 4; };
            auto C_ed = [&](const std::vector<double>& w, int j, double tt) {
                // <gs|W_j e^{-i(H-E0)t} W_l0|gs> via the sector eigenbasis
                Cxd c = 0;
                for (int b = 0; b < m; ++b) {
                    Cxd amp_j = 0, amp_l = 0;
                    for (int a = 0; a < m; ++a) {
                        amp_j += std::conj(Vs(a,0)) * w[digit(a,j)]  * Vs(a,b);
                        amp_l += std::conj(Vs(a,b)) * w[digit(a,l0)] * Vs(a,0);
                    }
                    c += amp_j * std::exp(Cxd(0,-(Es(b)-Es(0))*tt)) * amp_l;
                }
                return c;
            };
            const std::vector<double> w_spin  = {0.0, +1.0, -1.0, 0.0};   // 2Sz
            const std::vector<double> w_chg   = {-1.0, 0.0, 0.0, +1.0};   // n - 1
            std::vector<int> cfgS(n);
            for (int i2 = 0; i2 < n; ++i2) cfgS[i2] = (i2 % 2 == 0) ? 1 : 2;
            double werr = 0;
            for (auto* w : {&w_spin, &w_chg}) {
                auto sp = run_spectral_blocks(mpo2, n, dt, nst, 128, 1e-12, 60, 1.0, 1.0,
                                              pack_q2(n, 0), &cfgS, w);
                for (size_t k = 0; k < sp.times.size(); ++k)
                    for (int j = 0; j < n; ++j) {
                        Cxd ced = C_ed(*w, j, sp.times[k]);
                        werr = std::max(werr, std::abs(ced - Cxd(sp.C_re[k][j], sp.C_im[k][j])));
                    }
                if (std::getenv("QX_SPEC_DIAG")) {   // where does it diverge: t=0 or later?
                    for (int k : {0, 1}) for (int j : {l0, l0+1}) {
                        Cxd ced = C_ed(*w, j, sp.times[k]);
                        std::printf("      [diag w0=%g t=%.2f j=%d: block=(%.5f,%.5f) ED=(%.5f,%.5f)]\n",
                                    (*w)[0], sp.times[k], j, sp.C_re[k][j], sp.C_im[k][j],
                                    ced.real(), ced.imag());
                    }
                    // third opinion on the t=0 statics: diagonal expectation straight
                    // from the sector-ED ground state (no time evolution, no engine)
                    for (int j : {l0, l0+1}) {
                        double c0 = 0;
                        for (int a = 0; a < m; ++a)
                            c0 += std::norm(Vs(a,0)) * (*w)[digit(a,j)] * (*w)[digit(a,l0)];
                        std::printf("      [diag  diagexp t=0 j=%d: ED-diag=%.5f]\n", j, c0);
                    }
                    // sum rule: sum_j C(j,l0;0) = <(sum_j W_j) W_l0> = 0 exactly (conserved)
                    double srB = 0, srE = 0;
                    for (int j = 0; j < n; ++j) { srB += sp.C_re[0][j];
                        srE += C_ed(*w, j, 0.0).real(); }
                    // and the SINGLE-U(1) path on the same physics
                    auto sp1 = run_spectral_blocks(mpo, n, dt, 1, 128, 1e-12, 60, 1.0, 1.0,
                                                   n, &cfgS, w);
                    std::printf("      [diag  sumrule block=%.2e ED=%.2e | 1U1 t=0 j=%d: %.5f]\n",
                                srB, srE, l0, sp1.C_re[0][l0]);
                    std::printf("      [diag  E0(spectral)=%.12f  E_gs(sector-ED)=%.12f  "
                                "gaps Es1-Es0=%.3e Es2-Es0=%.3e]\n",
                                sp.E0, Es(0), Es(1)-Es(0), Es(2)-Es(0));
                }
            }
            bool okw = werr < 2e-3;
            std::printf("            S(k,w): max|C_spin/chg - C_ED| (t<=%.1f) = %.1e  %s\n",
                        nst*dt, werr, okw ? "OK" : "FAIL");
            fails += !okw;
        }
    }

    // (g) single-site GS refinement at total_charge != 0 (the #171 regression, now
    // root-caused to the non-injective-qphys bug in apply_heff_1site). At n=4 the
    // two-site warm-up converges before single-site engages, so exercise it at N=12
    // (truncation + single-site both active; the old bug corrupted this run to ~0).
    // No ED at this size -- the check is hybrid (warm-up -> single-site) vs pure
    // two-site: same variational optimum, and a sane (non-product-state) energy.
    std::printf("== Hubbard single-site GS refinement at N=12 (hybrid vs two-site) ==\n");
    {
        const int n = 12; const double t = 1.0, U = 4.0;
        std::vector<int> cfg(n);
        for (int i = 0; i < n; ++i) cfg[i] = (i % 2 == 0) ? 1 : 2;
        auto mpo = build_hubbard_mpo(n, t, U);
        double E_hybrid = run_dmrg_blocks(mpo, n, 64, 1e-11, 30, 1e-11, n, &cfg).energy;
        set_env("QX_DMRG_NO_SINGLESITE", "1");
        double E_twosite = run_dmrg_blocks(mpo, n, 64, 1e-11, 30, 1e-11, n, &cfg).energy;
        set_env("QX_DMRG_NO_SINGLESITE", nullptr);
        double diff = std::abs(E_hybrid - E_twosite);
        // Tolerance 1e-5, not 1e-7: the DMRG3S mixer changes the single-site
        // convergence PATH (both energies are legitimate variational values on the
        // way to the same fixed point; they differ at the 1e-7 level within this
        // sweep budget). The check still catches the catastrophic regression class
        // (the old total_charge bug left hybrid near 0, a 6.5-sized error).
        bool ok = diff < 1e-5 && E_hybrid < -6.0;   // old bug: hybrid ended near 0
        std::printf("  N=%d U=%.1f : hybrid=%.8f  two-site=%.8f  diff=%.1e  %s\n",
                    n, U, E_hybrid, E_twosite, diff, ok ? "OK" : "FAIL");
        fails += !ok;
    }

    // (h) FINITE TEMPERATURE via ancilla purification: cool the beta=0 thermofield
    // double with the fermionic (JW-in-the-MPO) swap-and-carry gates. The JW string
    // threads the physical modes only (ancillas are bosonic spectators: identity MPO
    // carries + plain SWAP -- the cooling never exchanges two physical modes), so if
    // any fermion sign were misplaced, <H>_beta would miss the exact grand-canonical
    // Gibbs average sum_i E_i e^{-beta E_i} / Z over ALL 4^n states (mu = 0).
    std::printf("== Hubbard finite-T via purification vs ED Gibbs (grand canonical) ==\n");
    {
        const int np = 3; const double t = 1.0, U = 4.0, beta = 1.0;
        const MatC& H = hubbard_ed_H(np, t, U);
        Eigen::VectorXd ev = Eigen::SelfAdjointEigenSolver<MatC>(H).eigenvalues();
        double e0 = ev(0), num = 0, den = 0;
        for (int i = 0; i < ev.size(); ++i) {
            double w = std::exp(-beta * (ev(i) - e0)); num += ev(i) * w; den += w; }
        double E_ed = num / den;
        // (J, delta) -> (t, U) convention for d=4; dbeta=0.05 -> 20 steps reach beta=1
        auto th = run_thermal_blocks(np, t, U, 0.05, 20, 64, 1e-12, 1, false, /*d*/4);
        double E_th = th.E.back();
        double err = std::abs(E_th - E_ed);
        // beta=0 exact checks: E(0) = Tr(H)/4^n = U*np/4 (hopping traceless);
        // physical RDM = I/4 per site -> <n_c n_j> = 1.5 (diag), <n>^2 = 1 (off-diag).
        double E0_exact = U * np / 4.0;
        bool ok0 = std::abs(th.E.front() - E0_exact) < 1e-8;
        const auto& nn0 = th.szsz.front();
        int cc = np / 2;
        double offmax = 0;
        for (int p = 0; p < np; ++p) if (p != cc) offmax = std::max(offmax, std::abs(nn0[p] - 1.0));
        bool ok1 = std::abs(nn0[cc] - 1.5) < 1e-6 && offmax < 1e-6;
        bool ok2 = err < 5e-3;
        std::printf("  np=%d U=%.1f : E(0)=%.6f (=%.2f) %s ; beta=0 <n_c n_j> diag=%.4f(=1.5) "
                    "off|max-1|=%.1e %s ; <H>_b1=%.6f ED=%.6f err=%.1e %s\n",
                    np, U, th.E.front(), E0_exact, ok0?"OK":"FAIL",
                    nn0[cc], offmax, ok1?"OK":"FAIL", E_th, E_ed, err, ok2?"OK":"FAIL");
        fails += !(ok0 && ok1 && ok2);
    }

    // (i) FINITE-T DYNAMICS, F1 + F2: the purified real-time correlators vs ED.
    // F1: C_{j,l0}(t;T) = Tr(rho_beta (2Sz_j)(t) (2Sz_l0)) -- the finite-T spin
    // structure factor's raw data. F2: C(t;T) = <J(t)J(0)>_beta with the fermionic
    // CHARGE current J = it sum_s (c^dag_{i s} c_{i+1 s} - h.c.) (JW form, so a
    // misplaced parity factor in the current MPO could not match ED).
    std::printf("== Hubbard finite-T F1 S(k,w) + F2 <J(t)J(0)> vs ED (grand canonical) ==\n");
    {
        const int np = 3, l0v = np / 2; const double t = 1.0, U = 4.0, beta = 0.5, dt = 0.1;
        const int nst = 5;
        const MatC& H = hubbard_ed_H(np, t, U);
        int dim = (int)H.rows();
        Eigen::SelfAdjointEigenSolver<MatC> es(H);
        Eigen::VectorXd E = es.eigenvalues(); MatC V = es.eigenvectors();
        double e0 = E(0), Zt = 0;
        for (int a = 0; a < dim; ++a) Zt += std::exp(-beta * (E(a) - e0));
        // plain (no JW string -- diagonal/even operators) site embedding
        MatC I4e = MatC::Identity(4,4);
        auto emb = [&](int site, const MatC& o) {
            MatC O = MatC::Identity(1,1);
            for (int k = 0; k < np; ++k) O = kron(O, k == site ? o : I4e);
            return O;
        };
        // F1 reference: M_j = V^dag (2Sz_j) V; C(j,t) = sum_ab w_a e^{i(Ea-Eb)t} Mj(a,b) Ml0(b,a)
        MatC Sz2 = MatC::Zero(4,4); Sz2(1,1) = 1; Sz2(2,2) = -1;
        auto Mop = [&](int j){ return MatC(V.adjoint() * emb(j, Sz2) * V); };
        MatC Ml0 = Mop(l0v);
        auto C1_ed = [&](int j, double tt) {
            MatC Mj = Mop(j); Cxd c = 0;
            for (int a = 0; a < dim; ++a) for (int b = 0; b < dim; ++b)
                c += std::exp(-beta*(E(a)-e0)) *
                     std::exp(Cxd(0,(E(a)-E(b))*tt)) * Mj(a,b) * Ml0(b,a);
            return c / Zt;
        };
        auto r1 = run_spectral_thermal_blocks(np, t, U, beta, 30, dt, nst, 64, 1e-10,
                                              /*disentangle*/true, /*d*/4);
        double e1max = 0;
        for (int s = 0; s <= nst; ++s) for (int j = 0; j < np; ++j) {
            Cxd ced = C1_ed(j, s*dt), cbl(r1.C_re[s][j], r1.C_im[s][j]);
            e1max = std::max(e1max, std::abs(ced - cbl));
        }
        bool okf1 = e1max < 2e-2;
        // F2 reference: the fermionic charge current via explicit JW operators
        MatC Jop = MatC::Zero(dim, dim);
        for (int l = 0; l < np - 1; ++l) {
            MatC hop = ferm(np,l,Cup.adjoint())*ferm(np,l+1,Cup)
                     + ferm(np,l,Cdn.adjoint())*ferm(np,l+1,Cdn);
            Jop += Cxd(0, t) * (hop - MatC(hop.adjoint()));
        }
        MatC Jm = MatC(V.adjoint() * Jop * V);
        auto C2_ed = [&](double tt) {
            Cxd c = 0;
            for (int a = 0; a < dim; ++a) for (int b = 0; b < dim; ++b)
                c += std::exp(-beta*(E(a)-e0)) *
                     std::exp(Cxd(0,(E(a)-E(b))*tt)) * Jm(a,b) * Jm(b,a);
            return c / Zt;
        };
        auto r2 = run_conductivity_thermal_blocks(np, t, U, beta, 30, dt, nst, 96, 1e-10,
                                                  /*disentangle*/true, /*d*/4);
        double e2max = 0;
        for (int s = 0; s <= nst; ++s) {
            Cxd ced = C2_ed(s*dt), cbl(r2.C_re[s], r2.C_im[s]);
            e2max = std::max(e2max, std::abs(ced - cbl));
        }
        bool okf2 = e2max < 3e-2;
        std::printf("  np=%d U=%.1f beta=%.1f : F1 max|C-C_ED|=%.2e %s ; "
                    "F2 C(0)=%.4f (ED %.4f) max|C-C_ED|=%.2e %s\n",
                    np, U, beta, e1max, okf1?"OK":"FAIL",
                    r2.C_re[0], std::real(C2_ed(0.0)), e2max, okf2?"OK":"FAIL");
        fails += !(okf1 && okf2);
    }

    // (j) TWO-LEG LADDER via rung fusion (d=16) -- the first quasi-2D model. The
    // fused-rung MPO must reproduce the explicit Jordan-Wigner ladder Hamiltonian
    // (JW order (x,a),(x,b),(x+1,a),...; leg hops have a string over the intervening
    // mode), and the fast block DMRG must hit the half-filling sector-ED energy AND
    // the Sz=1 sector (the spin gap) through the d=16 packed charges.
    std::printf("== Hubbard 2-leg LADDER (rung-fused d=16) vs explicit-JW ED ==\n");
    {
        const int Lx = 3, nm = 2 * Lx;   // 6 fermion modes -> 4^6 = 4096 ED
        const double t = 1.0, tp = 0.9, U = 4.0;
        // explicit-JW ladder H on the mode chain (a,b interleaved): bonds =
        // rungs (2x, 2x+1) with tp and legs (i, i+2) with t.
        int dim = 1; for (int k = 0; k < nm; ++k) dim *= 4;
        MatC H = MatC::Zero(dim, dim);
        auto hop = [&](int i, int j, double amp) {   // -amp (c^dag_i c_j + h.c.), i<j
            MatC h = ferm(nm,i,Cup.adjoint())*ferm(nm,j,Cup)
                   + ferm(nm,i,Cdn.adjoint())*ferm(nm,j,Cdn);
            H += -amp * (h + MatC(h.adjoint()));
        };
        for (int x = 0; x < Lx; ++x) hop(2*x, 2*x + 1, tp);          // rungs
        for (int x = 0; x < Lx - 1; ++x) { hop(2*x, 2*x + 2, t);     // leg a
                                           hop(2*x + 1, 2*x + 3, t); }  // leg b
        for (int i = 0; i < nm; ++i) H += U * (ferm(nm,i,Nu) * ferm(nm,i,Nd));
        auto mpo = build_hubbard_ladder_mpo(Lx, t, tp, U);
        MatC H_mpo = contract_mpo(mpo);   // 16^3 = 4096, same basis/order as the mode chain
        double herr = (H - H_mpo).cwiseAbs().maxCoeff();
        bool okh = herr < 1e-12;
        // sector-ED: lowest energy at (N = nm, 2Sz = 0) and (N = nm, 2Sz = 2)
        auto sector_e0 = [&](int Nt_want, int Sz_want) {
            const int z4l[4] = {0, 1, -1, 0};
            std::vector<int> idx;
            for (int k = 0; k < dim; ++k) { int Nt=0, Zt=0, kk=k;
                for (int i = 0; i < nm; ++i) { Nt += nof[kk%4]; Zt += z4l[kk%4]; kk /= 4; }
                if (Nt == Nt_want && Zt == Sz_want) idx.push_back(k); }
            int m = (int)idx.size();
            MatC Hs(m, m);
            for (int a2=0;a2<m;++a2) for (int b2=0;b2<m;++b2) Hs(a2,b2) = H(idx[a2], idx[b2]);
            return Eigen::SelfAdjointEigenSolver<MatC>(Hs).eigenvalues()(0);
        };
        double E0_ed = sector_e0(nm, 0), E1_ed = sector_e0(nm, 2);
        // fast block DMRG on the fused chain: half filling (N=nm, Sz=0) and Sz=1
        std::vector<int> cfg0(Lx), cfg1(Lx);
        for (int x = 0; x < Lx; ++x) cfg0[x] = (x % 2 == 0) ? (4*1 + 2) : (4*2 + 1);  // (up,dn)/(dn,up)
        cfg1 = cfg0; cfg1[0] = 4*1 + 1;                                   // one rung (up,up): Sz=1
        double E0_blk = run_dmrg_blocks(mpo, Lx, 256, 1e-11, 120, 1e-11,
                                        pack_q2(nm, 0), &cfg0).energy;
        double E1_blk = run_dmrg_blocks(mpo, Lx, 256, 1e-11, 120, 1e-11,
                                        pack_q2(nm, 2), &cfg1).energy;
        double e0 = std::abs(E0_blk - E0_ed), e1 = std::abs(E1_blk - E1_ed);
        bool ok0 = e0 < 1e-6, ok1 = e1 < 1e-6;
        std::printf("  Lx=%d tp=%.1f U=%.1f : ||H_MPO-H_JW||=%.1e %s ; E0(half,Sz=0)=%.8f "
                    "blk err=%.1e %s ; E(Sz=1)=%.8f blk err=%.1e %s ; gap=%.4f\n",
                    Lx, tp, U, herr, okh?"OK":"FAIL", E0_ed, e0, ok0?"OK":"FAIL",
                    E1_ed, e1, ok1?"OK":"FAIL", E1_ed - E0_ed);
        fails += !(okh && ok0 && ok1);

        // SITE-SPLIT formulation of the same ladder: d=4 on 2*Lx sites, PERIOD-2 MPO
        // (leg hops range-2 with F carries; derive_mpo_sym reads both sublattices).
        // Same basis/ordering as the mode-chain ED -> direct comparison; the block
        // DMRG must reproduce the same sector energies through this second route.
        auto mpo2 = build_hubbard_ladder_split_mpo(Lx, t, tp, U);
        MatC H_mpo2 = contract_mpo(mpo2);
        double herr2 = (H - H_mpo2).cwiseAbs().maxCoeff();
        bool okh2 = herr2 < 1e-12;
        std::vector<int> scfg0(nm), scfg1(nm);
        for (int i = 0; i < nm; ++i) scfg0[i] = (i % 2 == 0) ? 1 : 2;   // |up dn up dn ...>
        scfg1 = scfg0; scfg1[1] = 1;                                    // two ups -> Sz=1
        double E0s = run_dmrg_blocks(mpo2, nm, 256, 1e-11, 120, 1e-11,
                                     pack_q2(nm, 0), &scfg0).energy;
        double E1s = run_dmrg_blocks(mpo2, nm, 256, 1e-11, 120, 1e-11,
                                     pack_q2(nm, 2), &scfg1).energy;
        double es0 = std::abs(E0s - E0_ed), es1 = std::abs(E1s - E1_ed);
        bool oks = es0 < 1e-6 && es1 < 1e-6;
        std::printf("  SPLIT (period-2 d=4): ||H_MPO-H_JW||=%.1e %s ; blk errs %.1e / %.1e %s\n",
                    herr2, okh2?"OK":"FAIL", es0, es1, oks?"OK":"FAIL");
        fails += !(okh2 && oks);

        // (k) The GENERAL width-W builder. W=2 open must equal the ladder Hamiltonian
        // exactly (self-check against the ED-validated ladder H at tp=t). W=3 strip
        // and CYLINDER vs fresh 4096-dim explicit-JW ED (column-major bond lists);
        // the cylinder wrap is the first range-(W-1) F-carry chain.
        {
            auto gen2 = build_hubbard_lattice_split_mpo(Lx, 2, t, U);   // isotropic: tp = t
            // reference: the ladder JW-ED rebuilt at tp = t
            MatC Ht = MatC::Zero(dim, dim);
            auto hop2 = [&](int i, int j, double amp) {
                MatC h = ferm(nm,i,Cup.adjoint())*ferm(nm,j,Cup)
                       + ferm(nm,i,Cdn.adjoint())*ferm(nm,j,Cdn);
                Ht += -amp * (h + MatC(h.adjoint()));
            };
            for (int x = 0; x < Lx; ++x) hop2(2*x, 2*x + 1, t);
            for (int x = 0; x < Lx - 1; ++x) { hop2(2*x, 2*x + 2, t); hop2(2*x + 1, 2*x + 3, t); }
            for (int i = 0; i < nm; ++i) Ht += U * (ferm(nm,i,Nu) * ferm(nm,i,Nd));
            double eg2 = (Ht - contract_mpo(gen2)).cwiseAbs().maxCoeff();
            // W=3, Lx=2 (6 modes): strip + cylinder
            const int W3 = 3, L3 = 2, n3 = 6;
            auto bonds_H = [&](bool cyl) {
                MatC Hh = MatC::Zero(dim, dim);
                auto hp = [&](int i, int j) {
                    MatC h = ferm(n3,i,Cup.adjoint())*ferm(n3,j,Cup)
                           + ferm(n3,i,Cdn.adjoint())*ferm(n3,j,Cdn);
                    Hh += -t * (h + MatC(h.adjoint()));
                };
                for (int x = 0; x < L3; ++x) {
                    for (int y = 0; y < W3 - 1; ++y) hp(W3*x + y, W3*x + y + 1);
                    if (cyl) hp(W3*x, W3*x + W3 - 1);
                }
                for (int x = 0; x < L3 - 1; ++x)
                    for (int y = 0; y < W3; ++y) hp(W3*x + y, W3*(x+1) + y);
                for (int i = 0; i < n3; ++i) Hh += U * (ferm(n3,i,Nu) * ferm(n3,i,Nd));
                return Hh;
            };
            double eg3o = (bonds_H(false) -
                           contract_mpo(build_hubbard_lattice_split_mpo(L3, W3, t, U))).cwiseAbs().maxCoeff();
            MatC H3c = bonds_H(true);
            auto cyl3 = build_hubbard_lattice_split_mpo(L3, W3, t, U, 0.0, /*y_periodic*/true);
            double eg3c = (H3c - contract_mpo(cyl3)).cwiseAbs().maxCoeff();
            // block DMRG on the W=3 cylinder, half filling
            const int z4l[4] = {0, 1, -1, 0};
            std::vector<int> idx;
            for (int k = 0; k < dim; ++k) { int Nt=0, Zt=0, kk=k;
                for (int i = 0; i < n3; ++i) { Nt += nof[kk%4]; Zt += z4l[kk%4]; kk /= 4; }
                if (Nt == n3 && Zt == 0) idx.push_back(k); }
            int m3 = (int)idx.size();
            MatC Hs3(m3, m3);
            for (int a2=0;a2<m3;++a2) for (int b2=0;b2<m3;++b2) Hs3(a2,b2) = H3c(idx[a2], idx[b2]);
            double E3_ed = Eigen::SelfAdjointEigenSolver<MatC>(Hs3).eigenvalues()(0);
            std::vector<int> c3(n3);
            for (int i = 0; i < n3; ++i) c3[i] = (i % 2 == 0) ? 1 : 2;
            double E3_blk = run_dmrg_blocks(cyl3, n3, 256, 1e-11, 120, 1e-11,
                                            pack_q2(n3, 0), &c3).energy;
            double e3 = std::abs(E3_blk - E3_ed);
            bool okg = eg2 < 1e-12 && eg3o < 1e-12 && eg3c < 1e-12 && e3 < 1e-6;
            std::printf("  WIDTH-W: ||W=2gen-ladder||=%.1e ; W=3 strip=%.1e cyl=%.1e ; "
                        "cyl blk err=%.1e  %s\n", eg2, eg3o, eg3c, e3, okg?"OK":"FAIL");
            fails += !okg;

            // (l) STRIPE MACHINERY: edge pinning + hole doping + per-site profiles,
            // the full chain vs ED on the W=3 Lx=2 cylinder. hpin adds
            // -hpin (-1)^y Sz_{0,y} on the x=0 column (Sz-diagonal: both charges
            // conserved); doped sector N=4, 2Sz=0 (2 holes); QX_PROFILES=1 makes
            // run_dmrg_blocks emit per-site (<n_i>, <2Sz_i>), compared against the
            // ED ground-vector expectation values site by site.
            {
                const double hpin = 0.4, U6 = 6.0;
                auto cylp = build_hubbard_lattice_split_mpo(L3, W3, t, U6, 0.0, true, hpin);
                // ED reference: rebuild the U=6 cylinder H + the pinning term
                MatC Hp = MatC::Zero(dim, dim);
                {
                    auto hp = [&](int i, int j) {
                        MatC h = ferm(n3,i,Cup.adjoint())*ferm(n3,j,Cup)
                               + ferm(n3,i,Cdn.adjoint())*ferm(n3,j,Cdn);
                        Hp += -t * (h + MatC(h.adjoint()));
                    };
                    for (int x = 0; x < L3; ++x) {
                        for (int y = 0; y < W3 - 1; ++y) hp(W3*x + y, W3*x + y + 1);
                        hp(W3*x, W3*x + W3 - 1);
                    }
                    for (int x = 0; x < L3 - 1; ++x)
                        for (int y = 0; y < W3; ++y) hp(W3*x + y, W3*(x+1) + y);
                    for (int i = 0; i < n3; ++i) Hp += U6 * (ferm(n3,i,Nu) * ferm(n3,i,Nd));
                    // NO ferm() here: Sz is parity-even, a bare local embedding
                    // (ferm's JW string would be uncancelled for a lone operator)
                    MatC Szl = 0.5 * (Nu - Nd);
                    auto locop = [&](int i) {
                        MatC O = MatC::Identity(1,1);
                        for (int k = 0; k < n3; ++k) O = kron(O, k == i ? Szl : I4);
                        return O;
                    };
                    for (int y = 0; y < W3; ++y)
                        Hp += (-hpin * ((y % 2 == 0) ? 1.0 : -1.0)) * locop(y);
                }
                double herrp = (Hp - contract_mpo(cylp)).cwiseAbs().maxCoeff();
                // doped sector N=4, 2Sz=0: ED energy, gap, and per-site profiles
                std::vector<int> idp;
                for (int k = 0; k < dim; ++k) { int Nt=0, Zt=0, kk=k;
                    for (int i = 0; i < n3; ++i) { Nt += nof[kk%4]; Zt += z4l[kk%4]; kk /= 4; }
                    if (Nt == 4 && Zt == 0) idp.push_back(k); }
                int mp = (int)idp.size();
                MatC Hsp(mp, mp);
                for (int a2=0;a2<mp;++a2) for (int b2=0;b2<mp;++b2) Hsp(a2,b2) = Hp(idp[a2], idp[b2]);
                Eigen::SelfAdjointEigenSolver<MatC> esp(Hsp);
                double Ep_ed = esp.eigenvalues()(0);
                double gap_ed = esp.eigenvalues()(1) - Ep_ed;   // pinning must lift degeneracy
                std::vector<double> n_ed(n3, 0.0), z_ed(n3, 0.0);
                for (int a2 = 0; a2 < mp; ++a2) {
                    double w = std::norm(esp.eigenvectors()(a2, 0));
                    int kk = idp[a2];
                    // kron builds site 0 leftmost = MOST significant digit, so the
                    // %4 decode walks sites n3-1 down to 0 (total sums in (k) are
                    // order-blind; per-site profiles are not)
                    for (int i = n3 - 1; i >= 0; --i) {
                        n_ed[i] += w * nof[kk%4]; z_ed[i] += w * z4l[kk%4]; kk /= 4; }
                }
                set_env("QX_PROFILES", "1");
                std::vector<int> cp = {1, 2, 0, 1, 2, 0};      // N=4, 2Sz=0 seed
                DMRGResult rp = run_dmrg_blocks(cylp, n3, 256, 1e-11, 120, 1e-11,
                                                pack_q2(4, 0), &cp);
                set_env("QX_PROFILES", nullptr);
                double ep = std::abs(rp.energy - Ep_ed), eprof = 0;
                for (int i = 0; i < n3; ++i) {
                    eprof = std::max(eprof, std::abs(rp.site_n[i] - n_ed[i]));
                    eprof = std::max(eprof, std::abs(rp.site_2sz[i] - z_ed[i]));
                }
                bool okp = herrp < 1e-12 && gap_ed > 1e-8 && ep < 1e-8 && eprof < 1e-6;
                std::printf("  STRIPE-MACH: ||H_pin||=%.1e ; gap=%.2e ; doped blk E err=%.1e ; "
                            "profile err=%.1e  %s\n", herrp, gap_ed, ep, eprof, okp?"OK":"FAIL");
                fails += !okp;

                // (m) PAIR-FIELD correlator MPO vs ED on the same pinned doped
                // cylinder. Three bond geometries exercise the y and x orientations,
                // the intra-pair JW string (x-bond: F carries across W-1 sites) and
                // the parity-even between-pair stretch. Checks: ||O_MPO - O_ED|| == 0
                // (contract vs explicit ferm products) and <GS|O|GS> from
                // gs_mpo_expectations == the ED sector ground-vector expectation.
                {
                    auto delta_ed = [&](int x, int y, char o) {   // ferm-built Delta^dag_b
                        int a = W3*x + y, b = (o=='y') ? a+1 : a+W3;
                        return MatC(( ferm(n3,a,MatC(Cup.adjoint()))*ferm(n3,b,MatC(Cdn.adjoint()))
                                    - ferm(n3,a,MatC(Cdn.adjoint()))*ferm(n3,b,MatC(Cup.adjoint())) )
                                    / std::sqrt(2.0));
                    };
                    struct BP { int x1,y1; char o1; int x2,y2; char o2; };
                    const BP bps[3] = { {0,0,'y', 1,0,'y'},     // y-y, r=1
                                        {0,1,'y', 1,0,'y'},     // y-y cross-leg
                                        {0,0,'x', 1,1,'y'} };   // x-y (strings + mixed orient)
                    double e_mpo = 0, e_gs = 0;
                    std::vector<std::vector<MPOSite>> obs;
                    std::vector<double> phi_ed;
                    for (auto& bp : bps) {
                        auto omp = build_pair_corr_mpo(n3, W3, bp.x1, bp.y1, bp.o1,
                                                       bp.x2, bp.y2, bp.o2);
                        MatC O_ed = delta_ed(bp.x1,bp.y1,bp.o1)
                                  * delta_ed(bp.x2,bp.y2,bp.o2).adjoint();
                        e_mpo = std::max(e_mpo, (O_ed - contract_mpo(omp)).cwiseAbs().maxCoeff());
                        obs.push_back(std::move(omp));
                        Cxd acc = 0;
                        for (int a2 = 0; a2 < mp; ++a2)
                            for (int b2 = 0; b2 < mp; ++b2)
                                acc += std::conj(esp.eigenvectors()(a2,0))
                                     * O_ed(idp[a2], idp[b2]) * esp.eigenvectors()(b2,0);
                        phi_ed.push_back(acc.real());
                    }
                    double Egs = 0;
                    auto phi = gs_mpo_expectations(cylp, n3, 256, 1e-11, 120, 1e-11,
                                                   pack_q2(4,0), &cp, obs, &Egs);
                    for (size_t k = 0; k < phi.size(); ++k)
                        e_gs = std::max(e_gs, std::abs(phi[k] - phi_ed[k]));
                    bool okpf = e_mpo < 1e-12 && e_gs < 1e-7 && std::abs(Egs - Ep_ed) < 1e-8;
                    std::printf("  PAIR-FIELD: ||O_MPO-O_ED||=%.1e ; <GS|O|GS> err=%.1e "
                                "(3 geoms, |Phi|~%.1e)  %s\n",
                                e_mpo, e_gs, std::abs(phi_ed[0]), okpf?"OK":"FAIL");
                    fails += !okpf;
                }
            }
        }
    }

    std::printf("%s\n", fails ? "HUBBARD TEST: FAIL" : "HUBBARD TEST: ALL OK");
    return fails ? 1 : 0;
}
