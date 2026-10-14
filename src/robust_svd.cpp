#include "robust_svd.hpp"

#include <Eigen/SVD>

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
  out.U = jac.matrixU();
  out.V = jac.matrixV();
  out.s = jac.singularValues();
  return out;
}

template ThinSVD<Eigen::MatrixXd> robust_thin_bdcsvd(const Eigen::MatrixXd&);
template ThinSVD<Eigen::MatrixXcd> robust_thin_bdcsvd(const Eigen::MatrixXcd&);

}  // namespace qx
