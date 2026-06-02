#pragma once
// Time-dependent variational principle (TDVP) for MPS time evolution.
//
// Stage 1: the Krylov (Lanczos) action of a matrix exponential,
//   exp(tau * A) v,   A Hermitian (supplied as a matvec), tau complex.
// This is the local time-step kernel of TDVP: tau = -i*dt for real-time
// evolution, tau = -dt for imaginary-time (cooling toward the ground state).

#include "dmrg_internal.hpp"     // qx_dmrg::MPOSite (2x2 MPO), build_heisenberg_mpo
#include <Eigen/Dense>
#include <complex>
#include <functional>
#include <vector>
#include <array>
#include <cmath>

namespace qx_tdvp {

using Cx   = std::complex<double>;
using Vec  = Eigen::VectorXcd;
using Mat  = Eigen::MatrixXcd;
using Mat2 = Eigen::Matrix2cd;
using MPOSite = qx_dmrg::MPOSite;

// exp(tau * A) v for Hermitian A given as matvec(.). Adaptive Lanczos with full
// reorthogonalization; stops when the estimate stops changing or the Krylov
// space becomes invariant.
inline Vec expmv(Cx tau, const std::function<Vec(const Vec&)>& matvec,
                 const Vec& v, int m_max = 40, double tol = 1e-12) {
    const double beta0 = v.norm();
    if (beta0 < 1e-300) return v;

    std::vector<Vec> V;
    V.reserve(m_max + 1);
    V.push_back(v / beta0);
    std::vector<double> alpha, beta;     // tridiagonal (real for Hermitian A)

    Vec w = matvec(V[0]);
    alpha.push_back(std::real(V[0].dot(w)));
    w -= alpha[0] * V[0];

    // exp(tau T_m) e_0 -> coefficients in the Krylov basis (length m)
    auto krylov_coeffs = [&]() -> Vec {
        int m = (int)alpha.size();
        Eigen::MatrixXd T = Eigen::MatrixXd::Zero(m, m);
        for (int i = 0; i < m; ++i) {
            T(i, i) = alpha[i];
            if (i + 1 < m) { T(i, i+1) = beta[i]; T(i+1, i) = beta[i]; }
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
        const Eigen::VectorXd& d = es.eigenvalues();
        const Eigen::MatrixXd& Q = es.eigenvectors();
        Vec c(m);
        for (int k = 0; k < m; ++k) c(k) = std::exp(tau * d(k)) * Q(0, k);
        return Q.cast<Cx>() * c;         // length-m coefficients
    };
    auto assemble = [&](const Vec& y) -> Vec {
        Vec r = Vec::Zero(v.size());
        for (int i = 0; i < (int)y.size(); ++i) r += y(i) * V[i];
        return beta0 * r;
    };

    Vec result = assemble(krylov_coeffs());
    for (int k = 1; k <= m_max; ++k) {
        double b = w.norm();
        if (b < 1e-14) break;            // invariant Krylov subspace
        beta.push_back(b);
        V.push_back(w / b);
        w = matvec(V[k]);
        double ak = std::real(V[k].dot(w));
        alpha.push_back(ak);
        w -= ak * V[k] + b * V[k-1];
        for (auto& vv : V) w -= vv.dot(w) * vv;   // full reorthogonalization

        Vec result_new = assemble(krylov_coeffs());
        double diff = (result_new - result).norm();
        result.swap(result_new);
        if (diff < tol * std::max(1.0, result.norm())) break;
    }
    return result;
}

// ===== Stage 2: MPS environments + one/zero-site effective Hamiltonians =====
//
// Conventions (match the dense DMRG path):
//   - site tensor A: Mat of shape (chi_L*2, chi_R), row = s*chi_L + a  (s phys).
//   - environment L/R: at(bra, mpo, ket).

struct Env {
    int chi1 = 0, chi_w = 0, chi2 = 0;
    std::vector<Cx> data;
    Env() = default;
    Env(int c1, int cw, int c2) : chi1(c1), chi_w(cw), chi2(c2),
        data((size_t)c1 * cw * c2, Cx(0)) {}
    Cx&       at(int m1, int a, int m2)       { return data[((size_t)m1*chi_w + a)*chi2 + m2]; }
    const Cx& at(int m1, int a, int m2) const { return data[((size_t)m1*chi_w + a)*chi2 + m2]; }
};

inline Env left_boundary()  { Env L(1,1,1); L.at(0,0,0) = 1; return L; }
inline Env right_boundary() { Env R(1,1,1); R.at(0,0,0) = 1; return R; }

// L_new[m2,b,m2'] = sum conj(A[s,m1,m2]) W(a,b)(s,sp) L[m1,a,m1'] A[sp,m1',m2']
inline Env update_left(const Env& L, const Mat& A, const MPOSite& W) {
    const int chi_new = A.cols(), chi_old = A.rows()/2, chi_wl = W.chi_l, chi_wr = W.chi_r;
    Env Ln(chi_new, chi_wr, chi_new);
    Mat L_flat(chi_old, chi_wl*chi_old);
    for (int m1=0;m1<chi_old;++m1) for (int a=0;a<chi_wl;++a) for (int m1p=0;m1p<chi_old;++m1p)
        L_flat(m1, a*chi_old+m1p) = L.at(m1,a,m1p);
    std::array<Mat,2> T1;
    for (int s=0;s<2;++s) T1[s] = A.block(s*chi_old,0,chi_old,chi_new).conjugate().transpose() * L_flat;
    for (int a=0;a<chi_wl;++a) for (int b=0;b<chi_wr;++b) {
        const Mat2& Wab = W.W(a,b);
        for (int s=0;s<2;++s) {
            Mat T1as = T1[s].block(0,a*chi_old,chi_new,chi_old);
            for (int sp=0;sp<2;++sp) {
                Cx w = Wab(s,sp); if (std::abs(w)<1e-15) continue;
                Mat c = w * (T1as * A.block(sp*chi_old,0,chi_old,chi_new));
                for (int m2=0;m2<chi_new;++m2) for (int m2p=0;m2p<chi_new;++m2p)
                    Ln.at(m2,b,m2p) += c(m2,m2p);
            }
        }
    }
    return Ln;
}

inline Env update_right(const Env& R, const Mat& A, const MPOSite& W) {
    const int chi_r = A.cols(), chi_l = A.rows()/2, chi_wl = W.chi_l, chi_wr = W.chi_r;
    Env Rn(chi_l, chi_wl, chi_l);
    Mat RT(chi_r, chi_wr*chi_r);
    for (int m2=0;m2<chi_r;++m2) for (int b=0;b<chi_wr;++b) for (int m2p=0;m2p<chi_r;++m2p)
        RT(m2p, b*chi_r+m2) = R.at(m2,b,m2p);
    std::array<Mat,2> T1;
    for (int sp=0;sp<2;++sp) T1[sp] = A.block(sp*chi_l,0,chi_l,chi_r).conjugate() * RT;
    for (int a=0;a<chi_wl;++a) for (int b=0;b<chi_wr;++b) {
        const Mat2& Wab = W.W(a,b);
        for (int s=0;s<2;++s) {
            Mat A_s = A.block(s*chi_l,0,chi_l,chi_r);
            for (int sp=0;sp<2;++sp) {
                Cx w = Wab(sp,s); if (std::abs(w)<1e-15) continue;
                Mat c = w * (A_s * T1[sp].block(0,b*chi_r,chi_l,chi_r).transpose());
                for (int m1=0;m1<chi_l;++m1) for (int m1p=0;m1p<chi_l;++m1p)
                    Rn.at(m1,a,m1p) += c(m1,m1p);
            }
        }
    }
    return Rn;
}

// one-site effective Hamiltonian. NOTE the index orders: update_left gives
// L.at(bra, mpo, ket) while update_right gives R.at(ket, mpo, bra).
//   (H1 X)[sp,a',b'] = sum L[a',wl,a] W(wl,wr)(sp,s) X[s,a,b] R[b,wr,b']
inline Mat apply_h1(const Env& L, const MPOSite& W, const Env& R, const Mat& X) {
    const int chiL = X.rows()/2, chiR = X.cols(), Dwl = W.chi_l, Dwr = W.chi_r;
    Mat Y = Mat::Zero(2*chiL, chiR);
    for (int wl=0; wl<Dwl; ++wl) {
        Mat Lwl(L.chi1, L.chi2);                       // (bra a', ket a)
        for (int ap=0;ap<L.chi1;++ap) for (int a=0;a<L.chi2;++a) Lwl(ap,a) = L.at(ap,wl,a);
        for (int wr=0; wr<Dwr; ++wr) {
            const Mat2& Wm = W.W(wl,wr); if (Wm.isZero(1e-15)) continue;
            Mat Rwr(R.chi1, R.chi2);                    // (ket b, bra b')
            for (int b=0;b<R.chi1;++b) for (int bp=0;bp<R.chi2;++bp) Rwr(b,bp) = R.at(b,wr,bp);
            for (int s=0;s<2;++s) {
                Mat LXsR = Lwl * X.block(s*chiL,0,chiL,chiR) * Rwr;   // (a', b')
                for (int sp=0;sp<2;++sp) {
                    Cx w = Wm(sp,s); if (std::abs(w)<1e-15) continue;
                    Y.block(sp*chiL,0,chiL,chiR) += w * LXsR;
                }
            }
        }
    }
    return Y;
}

// zero-site (bond) effective Hamiltonian:  (H0 C)[a',b'] = sum_w L[a',w,a] C[a,b] R[b,w,b']
inline Mat apply_h0(const Env& L, const Env& R, const Mat& C) {
    Mat Y = Mat::Zero(L.chi1, R.chi2);
    for (int w=0; w<L.chi_w; ++w) {
        Mat Lw(L.chi1,L.chi2), Rw(R.chi1,R.chi2);
        for (int ap=0;ap<L.chi1;++ap) for (int a=0;a<L.chi2;++a) Lw(ap,a) = L.at(ap,w,a);   // (bra,ket)
        for (int b=0;b<R.chi1;++b) for (int bp=0;bp<R.chi2;++bp) Rw(b,bp) = R.at(b,w,bp);    // (ket,bra)
        Y += Lw * C * Rw;
    }
    return Y;
}

// ----- canonicalization (QR/LQ on a site, carrying the bond matrix) ----------
// left-canonicalize A=(chiL*2,chiR): A<-Q (left-canonical), return R (k x chiR).
inline Mat qr_left(Mat& A) {
    Eigen::HouseholderQR<Mat> qr(A);
    int k = std::min<int>(A.rows(), A.cols());
    Mat Q = qr.householderQ() * Mat::Identity(A.rows(), k);
    Mat R = Mat(qr.matrixQR().topLeftCorner(k, A.cols()).template triangularView<Eigen::Upper>());
    A = Q;
    return R;
}
// absorb a left bond matrix R (k x chiL) into the left bond of A=(chiL*2,chiR).
inline void absorb_left(const Mat& R, Mat& A) {
    int chi = A.rows()/2, chiR = A.cols(), k = R.rows();
    Mat An(2*k, chiR);
    for (int s=0;s<2;++s) An.block(s*k,0,k,chiR) = R * A.block(s*chi,0,chi,chiR);
    A = An;
}
// right-canonicalize A=(chiL*2,chiR): A<-Q (right-canonical), return L (chiL x k).
inline Mat lq_right(Mat& A) {
    int chiL = A.rows()/2, chiR = A.cols();
    Mat Mr(chiL, 2*chiR);
    for (int s=0;s<2;++s) for (int a=0;a<chiL;++a) for (int b=0;b<chiR;++b)
        Mr(a, s*chiR+b) = A(s*chiL+a, b);
    Eigen::HouseholderQR<Mat> qr(Mr.transpose());
    int k = std::min<int>(Mr.rows(), Mr.cols());
    Mat Qt = qr.householderQ() * Mat::Identity(Mr.cols(), k);          // (2chiR x k)
    Mat Rt = Mat(qr.matrixQR().topLeftCorner(k, Mr.rows()).template triangularView<Eigen::Upper>());
    Mat Q = Qt.transpose();                                           // (k x 2chiR) right-canonical
    Mat An(2*k, chiR);
    for (int s=0;s<2;++s) for (int a=0;a<k;++a) for (int b=0;b<chiR;++b)
        An(s*k+a, b) = Q(a, s*chiR+b);
    A = An;
    return Rt.transpose();                                            // L (chiL x k)
}
// absorb a right bond matrix L (chiR x k) into the right bond of A=(chiL*2,chiR).
inline void absorb_right(Mat& A, const Mat& L) {
    int chi = A.rows()/2, chiR = A.cols(), k = L.cols();
    Mat An(2*chi, k);
    for (int s=0;s<2;++s) An.block(s*chi,0,chi,k) = A.block(s*chi,0,chi,chiR) * L;
    A = An;
}

// ===== Stage 3-4: TDVP1 integrator + real/imaginary-time drivers ============

inline double mps_norm2(const std::vector<Mat>& A) {
    Mat O(1,1); O << Cx(1,0);
    for (auto& Ai : A) {
        int chiL = Ai.rows()/2, chiR = Ai.cols();
        Mat On = Mat::Zero(chiR, chiR);
        for (int s=0;s<2;++s) {
            Mat As = Ai.block(s*chiL,0,chiL,chiR);
            On += As.adjoint() * O * As;
        }
        O = On;
    }
    return std::real(O(0,0));
}

inline double mps_energy(const std::vector<Mat>& A, const std::vector<MPOSite>& W) {
    Env L = left_boundary();
    for (int i=0;i<(int)A.size();++i) L = update_left(L, A[i], W[i]);
    return std::real(L.at(0,0,0)) / mps_norm2(A);
}

inline void right_canonicalize(std::vector<Mat>& A) {
    for (int i=(int)A.size()-1;i>=1;--i) { Mat L = lq_right(A[i]); absorb_right(A[i-1], L); }
}

// Neel product state |up dn up dn ...> as a bond-dimension-1 MPS.
inline std::vector<Mat> neel_state(int n) {
    std::vector<Mat> A(n);
    for (int i=0;i<n;++i) {
        A[i] = Mat::Zero(2,1);
        A[i]((i%2==0)?0:1, 0) = 1;   // even site up, odd site down
    }
    return A;
}

// one symmetric second-order TDVP1 step of size dt. The MPS must enter in
// right-canonical form (centre at site 0) and leaves so. imag=true does
// imaginary-time (e^{-H dt}, renormalized) for ground-state / cooling.
inline void tdvp_step(std::vector<Mat>& A, const std::vector<MPOSite>& W,
                      double dt, bool imag) {
    int N = (int)A.size();
    Cx fwd = imag ? Cx(-dt/2,0) : Cx(0,-dt/2);   // site forward (half step)
    Cx bwd = imag ? Cx(+dt/2,0) : Cx(0,+dt/2);   // bond backward (half step)

    std::vector<Env> Renv(N+1), Lenv(N+1);
    Renv[N] = right_boundary();
    for (int i=N-1;i>=1;--i) Renv[i] = update_right(Renv[i+1], A[i], W[i]);
    Lenv[0] = left_boundary();

    auto evolve_site = [&](int i, const Env& L, const Env& R) {
        int r = (int)A[i].rows(), c = (int)A[i].cols();
        auto h1 = [&](const Vec& v) -> Vec {
            Mat X = Eigen::Map<const Mat>(v.data(), r, c);
            Mat Y = apply_h1(L, W[i], R, X);
            return Eigen::Map<Vec>(Y.data(), Y.size());
        };
        Vec av = Eigen::Map<Vec>(A[i].data(), A[i].size());
        Vec ev = expmv(fwd, h1, av);
        A[i] = Eigen::Map<Mat>(ev.data(), r, c);
    };
    auto evolve_bond = [&](Mat& C, const Env& L, const Env& R) {
        int r = (int)C.rows(), c = (int)C.cols();
        auto h0 = [&](const Vec& v) -> Vec {
            Mat Cm = Eigen::Map<const Mat>(v.data(), r, c);
            Mat Y = apply_h0(L, R, Cm);
            return Eigen::Map<Vec>(Y.data(), Y.size());
        };
        Vec cv = Eigen::Map<Vec>(C.data(), C.size());
        Vec ce = expmv(bwd, h0, cv);
        C = Eigen::Map<Mat>(ce.data(), r, c);
    };

    // left -> right
    for (int i=0;i<N;++i) {
        evolve_site(i, Lenv[i], Renv[i+1]);
        if (i < N-1) {
            Mat C = qr_left(A[i]);
            Lenv[i+1] = update_left(Lenv[i], A[i], W[i]);
            evolve_bond(C, Lenv[i+1], Renv[i+1]);
            absorb_left(C, A[i+1]);
        }
    }
    // right -> left
    for (int i=N-1;i>=0;--i) {
        evolve_site(i, Lenv[i], Renv[i+1]);
        if (i > 0) {
            Mat C = lq_right(A[i]);
            Renv[i] = update_right(Renv[i+1], A[i], W[i]);
            evolve_bond(C, Lenv[i], Renv[i]);
            absorb_right(A[i-1], C);
        }
    }
    if (imag) { double nz = std::sqrt(mps_norm2(A)); if (nz>0) A[0] /= nz; }
}

// evolve for n_steps of size dt; returns the final energy. The MPS is
// right-canonicalized on entry.
inline double run_tdvp(std::vector<Mat>& A, const std::vector<MPOSite>& W,
                       double dt, int n_steps, bool imag) {
    right_canonicalize(A);
    if (imag) { double nz = std::sqrt(mps_norm2(A)); if (nz>0) A[0] /= nz; }
    for (int step=0; step<n_steps; ++step) tdvp_step(A, W, dt, imag);
    return mps_energy(A, W);
}

} // namespace qx_tdvp
