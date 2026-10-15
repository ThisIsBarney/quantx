#include "robust_svd.hpp"

#include <Eigen/SVD>

#include <algorithm>
#include <limits>

namespace qx {

template <class MatT>
ThinSVD<MatT> robust_thin_bdcsvd(const MatT& X) {
  ThinSVD<MatT> out;
  Eigen::BDCSVD<MatT> bdc(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (bdc.info() == Eigen::Success) {
    out.U = bdc.matrixU();
    out.V = bdc.matrixV();
    out.s = bdc.singularValues();
    return out;
  }
  Eigen::JacobiSVD<MatT> jac(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (jac.info() != Eigen::Success) {
    // A NaN or an Inf in X: there is no factorisation to give. NaNs of the right shapes make
    // that show downstream; the solvers' own outputs are uninitialised and can look finite.
    // (No exception: the callers run inside OpenMP loops, which an exception cannot leave.)
    using Real = typename Eigen::NumTraits<typename MatT::Scalar>::Real;
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const Eigen::Index k = std::min(X.rows(), X.cols());
    out.U = MatT::Constant(X.rows(), k, typename MatT::Scalar(nan));
    out.V = MatT::Constant(X.cols(), k, typename MatT::Scalar(nan));
    out.s.setConstant(k, nan);
    return out;
  }
  out.U = jac.matrixU();
  out.V = jac.matrixV();
  out.s = jac.singularValues();
  return out;
}

template ThinSVD<Eigen::MatrixXd> robust_thin_bdcsvd(const Eigen::MatrixXd&);
template ThinSVD<Eigen::MatrixXcd> robust_thin_bdcsvd(const Eigen::MatrixXcd&);

}  // namespace qx
