// Phase A unit check (docs/gpu_acceleration_plan.md, validation step 2):
// random complex m x n blocks of several shapes, verify the GPU Gram-trick SVD
//   - reconstructs M:        || U diag(sv) Vᴴ - M ||
//   - has orthonormal U,V:   || UᴴU - I ||, || VᴴV - I ||
//   - matches the reference singular values from Eigen's BDCSVD
#include "gpu_svd.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <complex>
#include <cstdio>
#include <random>

using Cx  = std::complex<double>;
using Cm  = Eigen::MatrixXcd;
using Rv  = Eigen::VectorXd;

static int failures = 0;
static void check(const char* name, double err, double tol) {
    bool ok = err < tol;
    if (!ok) ++failures;
    std::printf("  %-46s err=%.3e  %s\n", name, err, ok ? "OK" : "FAIL");
}

static void test_shape(int m, int n, std::mt19937_64& rng) {
    std::normal_distribution<double> g(0.0, 1.0);
    Cm M(m, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) M(i, j) = Cx(g(rng), g(rng));

    const int k = std::min(m, n);
    Cm U(m, k), V(n, k);
    Rv sv(k);
    bool ok = qx_gpu::thin_svd_gram_gpu_z(
        reinterpret_cast<const Cx*>(M.data()), m, n,
        reinterpret_cast<Cx*>(U.data()), reinterpret_cast<Cx*>(V.data()), sv.data());
    if (!ok) { std::printf("  [%dx%d] thin_svd_gram_gpu_z returned false\n", m, n); ++failures; return; }

    // reconstruction
    Cm recon = U * sv.asDiagonal() * V.adjoint();
    double rec_err = (recon - M).norm() / M.norm();

    // orthonormality
    Cm Ik = Cm::Identity(k, k);
    double uerr = (U.adjoint() * U - Ik).norm();
    double verr = (V.adjoint() * V - Ik).norm();

    // descending order
    double order_ok = 0;
    for (int j = 1; j < k; ++j) order_ok = std::max(order_ok, std::max(0.0, sv(j) - sv(j-1)));

    // reference singular values (Eigen BDCSVD)
    Eigen::BDCSVD<Cm> ref(M);
    Rv refsv = ref.singularValues();   // descending, length k
    double sv_err = (sv - refsv).cwiseAbs().maxCoeff() / (refsv(0) > 0 ? refsv(0) : 1.0);

    std::printf(" [%dx%d] k=%d:\n", m, n, k);
    check("reconstruction ||UΣVᴴ-M||/||M||", rec_err, 1e-8);
    check("U orthonormal ||UᴴU-I||",          uerr,    1e-9);
    check("V orthonormal ||VᴴV-I||",          verr,    1e-9);
    check("sv descending",                     order_ok, 1e-14);
    check("sv vs Eigen BDCSVD (rel)",          sv_err,  1e-8);
}

int main() {
    if (!qx_gpu::gpu_available()) {
        std::printf("No usable CUDA device (gpu_available()=false); skipping.\n");
        return 0;   // not a failure on a CPU-only box
    }
    std::printf("GPU Gram-SVD unit check (thin_svd_gram_gpu_z):\n");
    std::mt19937_64 rng(12345);
    test_shape(64, 64, rng);     // square
    test_shape(80, 120, rng);    // m < n  (G = M Mᴴ)
    test_shape(120, 80, rng);    // m > n  (G = Mᴴ M)
    test_shape(200, 200, rng);   // larger square (dominant-block scale)
    test_shape(96, 300, rng);    // tall-skinny transposed
    std::printf("\n%s\n", failures == 0 ? "ALL PASSED" : "SOME FAILED");
    return failures == 0 ? 0 : 1;
}
