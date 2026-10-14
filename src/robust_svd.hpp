#pragma once
// A thin SVD that does not trust Eigen's divide-and-conquer blindly.
//
// Eigen's BDCSVD is the fast direct SVD for large blocks, but its deflation step can give up
// on rank-deficient or degenerate matrices: Eigen 3.4.0 read out of bounds there
// (perm(l-1) at l == 0 in perturbCol0, Eigen issue #1723). With the fix from Eigen 3.4.1
// that the build applies (cmake/eigen_bdcsvd_fixes.cmake), it sets info() = NumericalIssue
// instead. The two-sided JacobiSVD has no deflation step, so it recomputes the
// factorisation whenever BDCSVD reports anything but success.
//
// The function is compiled once, in robust_svd.cpp (the qx_svd library, which every target
// linking qx_eigen gets), for Eigen::MatrixXd and Eigen::MatrixXcd. That also keeps the two
// SVDs out of the large DMRG translation units, where MSVC 19.50 (VS 2026) crashed (C1001)
// on some compilations with the function inline.
#include <Eigen/Core>

namespace qx {

template <class MatT>
struct ThinSVD {
  MatT U;
  MatT V;
  Eigen::Matrix<typename Eigen::NumTraits<typename MatT::Scalar>::Real, Eigen::Dynamic, 1> s;
};

template <class MatT>
ThinSVD<MatT> robust_thin_bdcsvd(const MatT& X);

extern template ThinSVD<Eigen::MatrixXd> robust_thin_bdcsvd(const Eigen::MatrixXd&);
extern template ThinSVD<Eigen::MatrixXcd> robust_thin_bdcsvd(const Eigen::MatrixXcd&);

}  // namespace qx
