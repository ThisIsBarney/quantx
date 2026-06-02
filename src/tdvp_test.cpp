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

    printf("\n%s\n", failures == 0 ? "ALL PASSED" : "SOME FAILED");
    return failures == 0 ? 0 : 1;
}
