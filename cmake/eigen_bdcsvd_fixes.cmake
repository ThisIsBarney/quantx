# Two faults in Eigen 3.4.0's BDCSVD (Eigen/src/SVD/BDCSVD.h), both fixed in Eigen 3.4.1:
#
# - perturbCol0 reads perm(l-1) at l == 0, out of bounds, on some rank-deficient blocks (Eigen
#   issue #1723). 3.4.1 stops there and sets info() = NumericalIssue.
# - The total-deflation test compares col0, not |col0|, with the threshold, so a negative entry
#   counts as zero.
#
# The engine stays on Eigen 3.4.0, the version it was validated and benchmarked with, and takes
# only these two fixes rather than the rest of 3.4.1 (some 5,600 changed lines). On the
# development machine, MSVC 19.50 (Visual Studio 2026) also crashed (C1001) more often with
# 3.4.1's headers, though it crashes now and then with 3.4.0's too. qx_fix_eigen_bdcsvd()
# applies the two fixes, verbatim from 3.4.1, to the Eigen the build uses. They are Eigen's code,
# under the MPL-2.0 like the file they change.
#
# qx_fix_eigen_bdcsvd(<eigen dir> <out var>) sets <out var> to the Eigen directory to compile
# against. An Eigen inside the build tree (a FetchContent download) is fixed in place. One outside
# it (FETCHCONTENT_SOURCE_DIR_EIGEN, FETCHCONTENT_BASE_DIR) is not ours to modify, so its Eigen/
# and unsupported/ directories are copied into the build tree and the copy is fixed. An Eigen that
# already has the fixes (3.4.1 or later) is used as it is.
function(qx_fix_eigen_bdcsvd eigen_dir out_var)
  set(${out_var} "${eigen_dir}" PARENT_SCOPE)
  set(rel "Eigen/src/SVD/BDCSVD.h")
  if(NOT EXISTS "${eigen_dir}/${rel}")
    message(FATAL_ERROR "Eigen's BDCSVD.h is not at ${eigen_dir}/${rel}")
  endif()
  file(READ "${eigen_dir}/${rel}" text)

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
    return()  # Eigen 3.4.1 or later, or fixed already
  endif()
  string(FIND "${text}" "${perm_old}" perm_at)
  string(FIND "${text}" "${defl_old}" defl_at)
  if(perm_at EQUAL -1 OR defl_at EQUAL -1)
    message(WARNING "${eigen_dir}/${rel} is neither Eigen 3.4.0's BDCSVD.h nor one with the "
                    "fixes from 3.4.1, so the fixes cannot be applied; it is used as it is.")
    return()
  endif()

  # Compare real paths: the same directory can be spelled with an 8.3 short name or another
  # letter case on Windows.
  file(REAL_PATH "${eigen_dir}" real_eigen)
  file(REAL_PATH "${CMAKE_BINARY_DIR}" real_bin)
  cmake_path(IS_PREFIX real_bin "${real_eigen}" NORMALIZE in_build_tree)
  if(NOT in_build_tree)
    set(copy "${CMAKE_BINARY_DIR}/eigen-bdcsvd-fixed")
    set(dirs "${eigen_dir}/Eigen")
    if(EXISTS "${eigen_dir}/unsupported")
      list(APPEND dirs "${eigen_dir}/unsupported")
    endif()
    file(COPY ${dirs} DESTINATION "${copy}")
    message(STATUS "Eigen: ${eigen_dir} is outside the build tree; compiling against a copy "
                   "with the BDCSVD fixes, ${copy}")
    set(eigen_dir "${copy}")
    set(${out_var} "${copy}" PARENT_SCOPE)
  endif()

  string(REPLACE "${perm_old}" "${perm_new}" text "${text}")
  string(REPLACE "${defl_old}" "${defl_new}" text "${text}")
  file(WRITE "${eigen_dir}/${rel}" "${text}")
  message(STATUS "Eigen: applied the two BDCSVD fixes from Eigen 3.4.1 to ${eigen_dir}/${rel}")
endfunction()
