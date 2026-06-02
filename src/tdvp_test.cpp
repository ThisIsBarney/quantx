// Validation of the TDVP Krylov exponential (stage 1): exp(tau*A) v via Lanczos
// vs the dense matrix exponential (eigendecomposition) for random Hermitian A,
// real-time (tau = -i dt) and imaginary-time (tau = -dt).
#include "tdvp.hpp"
#include <cstdio>
#include <random>

using namespace qx_tdvp;

static int failures = 0;
static void check(const char* name, double err, double tol = 1e-9) {
    bool ok = err < tol && std::isfinite(err);
    if (!ok) ++failures;
    printf("  %-44s err=%.3e  %s\n", name, err, ok ? "OK" : "FAIL");
}

// ---- Stage 2: environments + one/zero-site H_eff vs dense ED ----------------
static Mat kron(const Mat& A, const Mat& B) {
    Mat C(A.rows()*B.rows(), A.cols()*B.cols());
    for (int i=0;i<A.rows();++i) for (int j=0;j<A.cols();++j)
        C.block(i*B.rows(), j*B.cols(), B.rows(), B.cols()) = A(i,j)*B;
    return C;
}
static Mat dense_heisenberg(int n, double J) {   // H = J sum sigma_i.sigma_j (OBC)
    Mat X(2,2); X<<0,1,1,0;
    Mat Y(2,2); Y<<Cx(0,0),Cx(0,-1),Cx(0,1),Cx(0,0);
    Mat Z(2,2); Z<<1,0,0,-1;
    Mat I2 = Mat::Identity(2,2);
    int d = 1<<n;
    Mat H = Mat::Zero(d,d);
    for (int i=0;i+1<n;++i) for (const Mat* P : {&X,&Y,&Z}) {
        Mat t(1,1); t<<1;
        for (int k=0;k<n;++k) t = kron(t, (k==i||k==i+1) ? *P : I2);
        H += J*t;
    }
    return H;
}
static Vec mps_to_sv(const std::vector<Mat>& A) {
    Mat psi(1,1); psi<<1;
    for (auto& Ai : A) {
        int chiL = (int)psi.cols(), chiR = (int)Ai.cols(), oldr = (int)psi.rows();
        Mat np(oldr*2, chiR);
        for (int s=0;s<2;++s) {
            Mat prod = psi * Ai.block(s*chiL,0,chiL,chiR);
            for (int r=0;r<oldr;++r) np.row(r*2+s) = prod.row(r);
        }
        psi = np;
    }
    return psi.col(0);
}
static double rfrob(const Mat& A, const Mat& B) {
    return std::real((A.array().conjugate() * B.array()).sum());
}
static void stage2(std::mt19937& rng) {
    std::uniform_real_distribution<double> d(-1,1);
    const int n = 4;
    auto W = qx_dmrg::build_heisenberg_mpo(n, 1.0);
    std::vector<int> chi = {1, 2, 4, 2, 1};   // bond dims
    std::vector<Mat> A(n);
    for (int i=0;i<n;++i) {
        A[i] = Mat(chi[i]*2, chi[i+1]);
        for (int r=0;r<A[i].rows();++r) for (int c=0;c<A[i].cols();++c) A[i](r,c)=Cx(d(rng),d(rng));
    }
    // dense reference energy
    Vec sv = mps_to_sv(A);
    Mat H = dense_heisenberg(n, 1.0);
    double Eref = std::real(sv.dot(H*sv)) / std::real(sv.dot(sv));

    // full-environment energy on the ORIGINAL (non-canonical) MPS
    {
        Env Lf = left_boundary();
        for (int i=0;i<n;++i) Lf = update_left(Lf, A[i], W[i]);
        double Eenv = std::real(Lf.at(0,0,0)) / std::real(sv.dot(sv));
        check("full-env (left) energy vs dense", std::abs(Eenv - Eref), 1e-9);

        Env Rf = right_boundary();
        for (int i=n-1;i>=0;--i) Rf = update_right(Rf, A[i], W[i]);
        double Eenv_r = std::real(Rf.at(0,0,0)) / std::real(sv.dot(sv));
        check("full-env (right) energy vs dense", std::abs(Eenv_r - Eref), 1e-9);
    }

    // center-canonical at site c: left-canon 0..c-1, right-canon c+1..n-1
    const int c = 2;
    for (int i=0;i<c;++i) { Mat R = qr_left(A[i]); absorb_left(R, A[i+1]); }
    for (int i=n-1;i>c;--i) { Mat L = lq_right(A[i]); absorb_right(A[i-1], L); }

    // canonicalization must preserve the state (up to overall scale)
    Vec sv2 = mps_to_sv(A);
    Cx ov = sv.dot(sv2);
    double fid = std::norm(ov) / (std::real(sv.dot(sv)) * std::real(sv2.dot(sv2)));
    check("canonicalization preserves state", std::abs(fid - 1.0), 1e-9);

    Env L = left_boundary();
    for (int i=0;i<c;++i) L = update_left(L, A[i], W[i]);
    Env R = right_boundary();
    for (int i=n-1;i>c;--i) R = update_right(R, A[i], W[i]);

    // one-site energy
    Mat HA = apply_h1(L, W[c], R, A[c]);
    double E1 = rfrob(A[c], HA) / rfrob(A[c], A[c]);
    check("one-site H_eff energy vs dense", std::abs(E1 - Eref), 1e-9);

    // zero-site: push center to the bond between c and c+1
    Mat C = qr_left(A[c]);            // A[c] now left-canonical, C is the bond
    Env L2 = update_left(L, A[c], W[c]);
    Mat HC = apply_h0(L2, R, C);
    double E0 = rfrob(C, HC) / rfrob(C, C);
    check("zero-site H_eff energy vs dense", std::abs(E0 - Eref), 1e-9);
}

// ---- Stage 3-4: TDVP1 real- and imaginary-time vs exact --------------------
static std::vector<Mat> random_mps(int n, const std::vector<int>& chi, std::mt19937& rng) {
    std::uniform_real_distribution<double> d(-1,1);
    std::vector<Mat> A(n);
    for (int i=0;i<n;++i) {
        A[i] = Mat(chi[i]*2, chi[i+1]);
        for (int r=0;r<A[i].rows();++r) for (int c=0;c<A[i].cols();++c) A[i](r,c)=Cx(d(rng),d(rng));
    }
    return A;
}
static void stage34(std::mt19937& rng) {
    const int n = 4;
    auto W = qx_dmrg::build_heisenberg_mpo(n, 1.0);
    std::vector<int> chi = {1,2,4,2,1};        // full bond dims -> TDVP1 is exact
    Mat H = dense_heisenberg(n, 1.0);
    Eigen::SelfAdjointEigenSolver<Mat> es(H);

    // real-time: compare to exact exp(-iHt)
    {
        std::vector<Mat> A = random_mps(n, chi, rng);
        Vec psi0 = mps_to_sv(A); psi0 /= psi0.norm();
        double Einit = std::real(psi0.dot(H*psi0));
        double dt = 0.1; int steps = 5; double t = dt*steps;
        Vec cc = es.eigenvectors().adjoint() * psi0;
        for (int k=0;k<cc.size();++k) cc(k) *= std::exp(Cx(0,-t)*es.eigenvalues()(k));
        Vec psit = es.eigenvectors() * cc;

        double Efin = run_tdvp(A, W, dt, steps, false);
        Vec psf = mps_to_sv(A); psf /= psf.norm();
        double fid = std::norm(psit.dot(psf));
        check("real-time TDVP1 fidelity vs exact (full bond)", std::abs(fid-1.0), 1e-8);
        check("real-time energy conserved", std::abs(Efin - Einit), 1e-8);
    }
    // imaginary-time: converge to the ground state
    {
        std::vector<Mat> B = random_mps(n, chi, rng);
        double Eg = run_tdvp(B, W, 0.1, 400, true);
        check("imag-time TDVP1 -> ground (full bond)", std::abs(Eg - es.eigenvalues()(0)), 1e-5);
    }
}

// ---- Stage 5: TDVP2 quench from a product state vs exact -------------------
static void stage5() {
    const int n = 6;
    auto W = qx_dmrg::build_heisenberg_mpo(n, 1.0);
    Mat H = dense_heisenberg(n, 1.0);
    Eigen::SelfAdjointEigenSolver<Mat> es(H);

    std::vector<Mat> A = neel_state(n);
    Vec psi0 = mps_to_sv(A);                 // |up dn up dn ...>, norm 1
    double E0 = std::real(psi0.dot(H*psi0));

    double dt = 0.02; int steps = 50; double t = dt*steps;   // t = 1.0
    double Ef = run_tdvp2(A, W, dt, steps, /*imag*/false, /*chi_max*/16, 1e-12);

    // energy conservation (no truncation: chi_max >= 8 = full bond for n=6)
    check("TDVP2 quench energy conserved", std::abs(Ef - E0), 1e-6);

    // fidelity vs exact exp(-iHt)|Neel>
    Vec cc = es.eigenvectors().adjoint() * psi0;
    for (int k=0;k<cc.size();++k) cc(k) *= std::exp(Cx(0,-t)*es.eigenvalues()(k));
    Vec psit = es.eigenvectors() * cc;
    Vec psf = mps_to_sv(A); psf /= psf.norm();
    double fid = std::norm(psit.dot(psf)) / std::real(psit.dot(psit));
    check("TDVP2 quench fidelity vs exact", std::abs(fid-1.0), 1e-3);

    // entanglement actually grew (middle bond > 1)
    int mid_bond = (int)A[n/2-1].cols();
    if (mid_bond <= 1) { ++failures; printf("  TDVP2 bond growth                            FAIL (bond=%d)\n", mid_bond); }
    else printf("  TDVP2 bond growth                            mid bond=%d  OK\n", mid_bond);
}

int main() {
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> d(-1, 1);
    const int dim = 40;

    // random Hermitian A
    Eigen::MatrixXcd M(dim, dim);
    for (int i = 0; i < dim; ++i)
        for (int j = 0; j < dim; ++j) M(i, j) = Cx(d(rng), d(rng));
    Eigen::MatrixXcd A = 0.5 * (M + M.adjoint());

    Vec v(dim);
    for (int i = 0; i < dim; ++i) v(i) = Cx(d(rng), d(rng));

    auto matvec = [&](const Vec& x) -> Vec { return A * x; };

    // dense reference exp(tau A) v
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(A);
    auto ref = [&](Cx tau) -> Vec {
        Vec c = es.eigenvectors().adjoint() * v;
        for (int k = 0; k < dim; ++k) c(k) *= std::exp(tau * es.eigenvalues()(k));
        return es.eigenvectors() * c;
    };

    printf("TDVP Krylov exponential exp(tau*A)v vs dense:\n");
    for (double dt : {0.05, 0.2, 0.5}) {
        Cx tau_rt(0, -dt);                 // real time
        double e_rt = (expmv(tau_rt, matvec, v) - ref(tau_rt)).norm() / ref(tau_rt).norm();
        char buf[64]; snprintf(buf, sizeof buf, "real-time dt=%.2f", dt);
        check(buf, e_rt);

        Cx tau_it(-dt, 0);                 // imaginary time
        double e_it = (expmv(tau_it, matvec, v) - ref(tau_it)).norm() / ref(tau_it).norm();
        snprintf(buf, sizeof buf, "imag-time dt=%.2f", dt);
        check(buf, e_it);
    }

    // norm preservation for real-time on a Hermitian generator
    Cx tau(0, -0.37);
    double n0 = v.norm(), n1 = expmv(tau, matvec, v).norm();
    check("real-time norm preserved", std::abs(n1 - n0) / n0, 1e-9);

    stage2(rng);
    stage34(rng);
    stage5();

    printf("\n%s\n", failures == 0 ? "ALL PASSED" : "SOME FAILED");
    return failures == 0 ? 0 : 1;
}
