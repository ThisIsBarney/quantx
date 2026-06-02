#pragma once
// Time-dependent variational principle (TDVP) for MPS time evolution.
//
// Stage 1: the Krylov (Lanczos) action of a matrix exponential,
//   exp(tau * A) v,   A Hermitian (supplied as a matvec), tau complex.
// This is the local time-step kernel of TDVP: tau = -i*dt for real-time
// evolution, tau = -dt for imaginary-time (cooling toward the ground state).

#include <Eigen/Dense>
#include <complex>
#include <functional>
#include <vector>
#include <cmath>

namespace qx_tdvp {

using Cx  = std::complex<double>;
using Vec = Eigen::VectorXcd;

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

} // namespace qx_tdvp
