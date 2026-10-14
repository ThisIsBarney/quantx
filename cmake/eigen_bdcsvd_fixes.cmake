# Two faults in Eigen 3.4.0's BDCSVD (Eigen/src/SVD/BDCSVD.h), both fixed in Eigen 3.4.1:
#
# - perturbCol0 reads perm(l-1) at l == 0, out of bounds, on some rank-deficient blocks (Eigen
#   issue #1723). 3.4.1 stops there and sets info() = NumericalIssue.
# - The total-deflation test compares col0, not |col0|, with the threshold, so a negative entry
#   counts as zero.
#
# The engine stays on Eigen 3.4.0: with MSVC 19.50 (Visual Studio 2026), 3.4.1's headers made
# cl.exe crash with C1001 on a few percent of the large DMRG compilations, which a full build
# almost always hit. qx_fix_eigen_bdcsvd() applies the two fixes, verbatim from 3.4.1, to the
# Eigen the build downloaded. They are Eigen's code, under the MPL-2.0 like the file they change.
#
# An Eigen outside the build tree (FETCHCONTENT_SOURCE_DIR_EIGEN) is never modified; if it lacks
# the fixes, configuring warns.
function(qx_fix_eigen_bdcsvd eigen_dir)
  set(file "${eigen_dir}/Eigen/src/SVD/BDCSVD.h")
  if(NOT EXISTS "${file}")
    message(FATAL_ERROR "Eigen's BDCSVD.h is not at ${file}")
  endif()
  file(READ "${file}" text)

  set(perm_old "Index j = i<k ? i : perm(l-1);")
  # As in Eigen 3.4.1, at the indentation of the line it replaces.
  set(perm_new "// Avoid index out of bounds.
          // Will end up setting zhat(k) = 0.
          if (i >= k && l == 0) {
            m_info = NumericalIssue;
            prod = 0;
            break;
          }
          Index j = i<k ? i : l > 0 ? perm(l-1) : i;")
  set(defl_old "bool total_deflation = (col0.tail(length-1).array()<considerZero).all();")
  set(defl_new "const bool total_deflation = (col0.tail(length-1).array().abs()<considerZero).all();")

  string(FIND "${text}" "l > 0 ? perm(l-1) : i" perm_fixed)
  string(FIND "${text}" ".array().abs()<considerZero).all();" defl_fixed)
  if(perm_fixed GREATER -1 AND defl_fixed GREATER -1)
    return()  # Eigen 3.4.1 or later, or already fixed
  endif()

  cmake_path(IS_PREFIX CMAKE_BINARY_DIR "${file}" NORMALIZE in_build_tree)
  if(NOT in_build_tree)
    message(WARNING "The Eigen at ${eigen_dir} lacks two BDCSVD fixes from Eigen 3.4.1 (an "
                    "out-of-bounds read on rank-deficient blocks, and a total-deflation test "
                    "that ignores the sign). It is outside the build tree, so it is left as it is.")
    return()
  endif()

  string(FIND "${text}" "${perm_old}" perm_at)
  string(FIND "${text}" "${defl_old}" defl_at)
  if(perm_at EQUAL -1 OR defl_at EQUAL -1)
    message(FATAL_ERROR "${file} is neither Eigen 3.4.0's BDCSVD.h nor a fixed one; "
                        "the BDCSVD fixes cannot be applied")
  endif()
  string(REPLACE "${perm_old}" "${perm_new}" text "${text}")
  string(REPLACE "${defl_old}" "${defl_new}" text "${text}")
  file(WRITE "${file}" "${text}")
  message(STATUS "Eigen: applied the two BDCSVD fixes from Eigen 3.4.1 to ${file}")
endfunction()
