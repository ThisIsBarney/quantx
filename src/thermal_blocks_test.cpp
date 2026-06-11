// Finite-temperature purification (F0) validation against dense ED.
//
// run_thermal_blocks cools the interleaved purified chain (2N sites: physical at
// even positions, ancilla at odd) by imaginary-time swap-and-carry block-TEBD2,
// and measures E(beta), <Z_j>, <Z_c Z_j>, and chi_zz on the physical sites.
// This test compares every checkpoint against exact thermal averages
//   <O>_beta = Tr(O e^{-beta H}) / Tr(e^{-beta H})
// from dense exact diagonalization of the *physical* XXZ chain (N=8 -> 256x256),
// for the isotropic (delta=1) and anisotropic (delta=0.5) points.
//
// What each check pins down:
//   beta=0 exactness     -> purified_init builds the infinite-T Gibbs state
//                           (E=0, <Z_j>=0, <Z_c Z_j>=delta_cj to machine precision:
//                           no Trotter, no truncation at beta=0)
//   E(beta) curve        -> the purified MPO (ancilla pass-through) + bt_energy
//                           and the cooling gate sequence (swap-and-carry layout
//                           restoration; a single misplaced swap scrambles E)
//   <Z_c Z_j>(beta)      -> overlap-env correlation measurement on the doubled chain
//   chi_zz(beta)         -> the all-pairs Var(sum Z)/N susceptibility path
//
// Tolerances are Trotter-dominated (O(dbeta^2) per unit beta at 2nd order;
// truncation is negligible at chi=256 >= 2^N for N=8).
#include "dmrg_internal.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace qx_dmrg;

static Mat kron(const Mat& A, const Mat& B) {
    Mat K(A.rows() * B.rows(), A.cols() * B.cols());
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < A.cols(); ++j)
            K.block(i * B.rows(), j * B.cols(), B.rows(), B.cols()) = A(i, j) * B;
    return K;
}

static Mat site_op(int n, int site, const Mat2& o) {
    Mat O = Mat::Identity(1, 1);
    for (int i = 0; i < n; ++i)
        O = kron(O, (i == site) ? Mat(o) : Mat(Mat2::Identity()));
    return O;
}

// Dense ED thermal reference: precomputes diag(V^dag O V) per operator so each
// checkpoint is a dot product with the Boltzmann weights.
struct EDThermal {
    Eigen::VectorXd w;   // eigenvalues (shifted weights handled in weights())
    Mat V;

    explicit EDThermal(const Mat& H) {
        Eigen::SelfAdjointEigenSolver<Mat> es(H);
        w = es.eigenvalues();
        V = es.eigenvectors();
    }
    Eigen::VectorXd diag(const Mat& O) const {
        return (V.adjoint() * O * V).diagonal().real();
    }
    Eigen::VectorXd weights(double beta) const {
        Eigen::VectorXd p = (-beta * (w.array() - w.minCoeff())).exp();
        return p / p.sum();
    }
};

struct Tols { double E, sz, szsz, chi; };

static int check_case(int n, double J, double delta, double dbeta, int n_steps,
                      int stride, int chi_max, const Tols& tol) {
    std::printf("=== XXZ N=%d J=%.2f delta=%.2f : dbeta=%.3f, %d steps (beta to %.2f), "
                "chi=%d ===\n", n, J, delta, dbeta, n_steps, dbeta * n_steps, chi_max);

    // -- dense ED reference on the physical chain --
    const int dim = 1 << n;
    Mat H = Mat::Zero(dim, dim);
    Mat2 X = pauli('X'), Y = pauli('Y'), Z = pauli('Z');
    for (int i = 0; i < n - 1; ++i)
        H += J * (site_op(n, i, X) * site_op(n, i + 1, X)
                + site_op(n, i, Y) * site_op(n, i + 1, Y)
                + delta * site_op(n, i, Z) * site_op(n, i + 1, Z));
    EDThermal ed(H);

    const int c = n / 2;
    std::vector<Eigen::VectorXd> dZ(n), dZcZ(n);
    Mat Zc = site_op(n, c, Z), M = Mat::Zero(dim, dim);
    for (int j = 0; j < n; ++j) {
        Mat Zj = site_op(n, j, Z);
        dZ[j]   = ed.diag(Zj);
        dZcZ[j] = ed.diag(Zc * Zj);
        M += Zj;
    }
    Eigen::VectorXd dM = ed.diag(M), dM2 = ed.diag(Mat(M * M));

    // -- purified MPS cooling --
    auto r = run_thermal_blocks(n, J, delta, dbeta, n_steps, chi_max,
                                /*cutoff*/1e-12, /*measure_stride*/stride,
                                /*with_chi*/true);
    std::printf("  cooled in %.2fs, max_bond=%d, c=%d, %d checkpoints\n",
                r.wall_time_s, r.max_bond, r.c, (int)r.betas.size());
    if (r.c != c) { std::printf("  FAIL: source site %d != %d\n", r.c, c); return 1; }

    int fails = 0;
    double worstE = 0, worstSz = 0, worstZZ = 0, worstChi = 0;
    std::printf("  %-6s %-12s %-9s | %-9s %-9s %-9s %-9s\n",
                "beta", "E(MPS)", "E(ED)", "errE", "errSz", "errZcZ", "errChi");
    for (size_t k = 0; k < r.betas.size(); ++k) {
        double beta = r.betas[k];
        Eigen::VectorXd p = ed.weights(beta);
        double E_ed = p.dot(ed.w);
        double errE = std::abs(r.E[k] - E_ed);
        double errSz = 0, errZZ = 0;
        for (int j = 0; j < n; ++j) {
            errSz = std::max(errSz, std::abs(r.sz[k][j]   - p.dot(dZ[j])));
            errZZ = std::max(errZZ, std::abs(r.szsz[k][j] - p.dot(dZcZ[j])));
        }
        double m1 = p.dot(dM);
        double chi_ed = beta * (p.dot(dM2) - m1 * m1) / n;
        double errChi = std::abs(r.chi_zz[k] - chi_ed);
        std::printf("  %-6.2f %-12.6f %-9.4f | %-9.2e %-9.2e %-9.2e %-9.2e\n",
                    beta, r.E[k], E_ed, errE, errSz, errZZ, errChi);

        // beta=0 must be exact (no Trotter, no truncation): machine precision.
        double tE   = (beta == 0.0) ? 1e-10 : tol.E;
        double tSz  = (beta == 0.0) ? 1e-10 : tol.sz;
        double tZZ  = (beta == 0.0) ? 1e-10 : tol.szsz;
        double tChi = (beta == 0.0) ? 1e-10 : tol.chi;
        if (errE > tE || errSz > tSz || errZZ > tZZ || errChi > tChi) {
            std::printf("    ^ FAIL (tol E=%.1e sz=%.1e zz=%.1e chi=%.1e)\n",
                        tE, tSz, tZZ, tChi);
            ++fails;
        }
        worstE = std::max(worstE, errE);     worstSz  = std::max(worstSz, errSz);
        worstZZ = std::max(worstZZ, errZZ);  worstChi = std::max(worstChi, errChi);
    }
    std::printf("  worst: errE=%.2e errSz=%.2e errZcZ=%.2e errChi=%.2e -> %s\n\n",
                worstE, worstSz, worstZZ, worstChi, fails ? "FAIL" : "OK");
    return fails;
}

int main() {
    int fails = 0;
    // Tolerances: ~10x the observed Trotter-dominated errors (worst measured:
    // errE=8.7e-4, errZcZ=1.7e-4, errChi=1.9e-4), so a regression in any path
    // -- init, MPO, gate sequence, measurement -- trips the assert while honest
    // Trotter noise does not. <Z_j> = 0 by spin-flip symmetry at every beta;
    // CPU reaches ~6e-12 but the GPU Gram-SVD (QX_GPU_MIN_BLK=1) carries ~2e-9
    // orthogonality noise, so 1e-7 keeps the symmetry check meaningful (any real
    // symmetry breaking is O(0.01..1)) on both backends.
    // Isotropic Heisenberg, beta to 2.0. chi=256 = 2^8 >= exact purified rank.
    fails += check_case(/*n*/8, /*J*/1.0, /*delta*/1.0, /*dbeta*/0.05,
                        /*n_steps*/40, /*stride*/8, /*chi*/256,
                        Tols{/*E*/5e-3, /*sz*/1e-7, /*szsz*/2e-3, /*chi*/2e-3});
    // Anisotropic (delta=0.5) wiring of the gate + purified MPO, beta to 1.0.
    fails += check_case(/*n*/8, /*J*/1.0, /*delta*/0.5, /*dbeta*/0.05,
                        /*n_steps*/20, /*stride*/5, /*chi*/256,
                        Tols{/*E*/5e-3, /*sz*/1e-7, /*szsz*/2e-3, /*chi*/2e-3});
    std::printf("%s\n", fails ? "THERMAL BLOCKS TEST: FAIL" : "THERMAL BLOCKS TEST: ALL OK");
    return fails ? 1 : 0;
}
