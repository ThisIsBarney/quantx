cmake_minimum_required(VERSION 3.20)
project(quantx_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# dmrg_blocks.cpp is a large scalar-templated translation unit; in Release (and
# more so once the GPU path adds instantiations) it exceeds the COFF section limit.
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif()

option(QX_ENABLE_OPENMP "Enable OpenMP parallel backend" ON)
option(QX_ENABLE_MPI "Enable MPI features" OFF)
option(QX_ENABLE_PYBIND "Build pybind11 bindings" OFF)
option(QX_ENABLE_CUDA "Enable CUDA/cuFFT support when the CUDA toolkit is available" OFF)

find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

find_package(spdlog CONFIG QUIET)
if(NOT spdlog_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    spdlog
    URL https://github.com/gabime/spdlog/archive/refs/tags/v1.14.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(spdlog)
endif()

find_package(CLI11 CONFIG QUIET)
if(NOT CLI11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    CLI11
    URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(CLI11)
endif()

include(FetchContent)
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
  endif()
  FetchContent_Populate(eigen)
endif()
add_library(qx_eigen INTERFACE)
target_include_directories(qx_eigen INTERFACE ${eigen_SOURCE_DIR})

# Optional: link OpenBLAS/LAPACKE so the large-block SVD uses LAPACK zgesdd
# instead of Eigen's built-in Gram eigensolve (Phase 0, docs/gpu_acceleration_plan.md).
# Configure with: -DQX_ENABLE_BLAS=ON -DQX_BLAS_INCLUDE_DIR=<dir with lapacke.h>
#                 -DQX_BLAS_LIB=<path to openblas import lib>
option(QX_ENABLE_BLAS "Link OpenBLAS/LAPACKE for the large-block SVD path" OFF)
add_library(qx_blas INTERFACE)
if(QX_ENABLE_BLAS)
  if(NOT QX_BLAS_INCLUDE_DIR OR NOT QX_BLAS_LIB)
    message(FATAL_ERROR "QX_ENABLE_BLAS=ON requires -DQX_BLAS_INCLUDE_DIR and -DQX_BLAS_LIB")
  endif()
  target_include_directories(qx_blas INTERFACE ${QX_BLAS_INCLUDE_DIR})
  target_link_libraries(qx_blas INTERFACE ${QX_BLAS_LIB})
  target_compile_definitions(qx_blas INTERFACE QX_HAS_LAPACKE=1 EIGEN_USE_BLAS=1)
  message(STATUS "QX_ENABLE_BLAS: linking ${QX_BLAS_LIB}")
endif()

# Optional: GPU (cuSOLVER/cuBLAS) Gram-SVD path for the block-sparse TEBD dynamics
# (Phase A, docs/gpu_acceleration_plan.md). Off by default; the CPU build is
# unaffected. Builds src/gpu_svd.cu with nvcc into a small static lib (qx_gpu) and
# defines QX_USE_CUDA on the targets that need it.
#   Configure with: -DQX_USE_CUDA=ON [-DCMAKE_CUDA_ARCHITECTURES=89]
option(QX_USE_CUDA "Build the GPU (cuSOLVER) Gram-SVD path for the block TEBD dynamics (Phase A)" OFF)
if(QX_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)  # RTX 4090 (Ada); 80=A100, 90=H100
  endif()
  add_library(qx_gpu STATIC src/gpu_svd.cu)
  set_target_properties(qx_gpu PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(qx_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(qx_gpu PUBLIC CUDA::cusolver CUDA::cublas CUDA::cudart)
  message(STATUS "QX_USE_CUDA: GPU SVD enabled (CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})")
endif()

if(QX_ENABLE_OPENMP)
  find_package(OpenMP)
endif()

if(QX_ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()

if(QX_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(NOT CUDAToolkit_FOUND)
    message(STATUS "CUDA toolkit not found; building Eigen fallback only.")
  endif()
endif()

add_library(qx_core
  src/backend_registry.cpp
  src/runtime.cpp
  src/scheduler.cpp
  src/logging.cpp
  src/version.cpp
  src/fft_provider.cpp
  src/statevector_backend.cpp
  src/mps_backend.cpp
  src/continuous2d_backend.cpp
  src/continuous3d_backend.cpp
  src/lindblad_backend.cpp
  src/engine.cpp
  src/vqe.cpp
  src/qpe.cpp
  src/qec.cpp
  src/qfield.cpp
  src/phi4.cpp
  src/hubbard.cpp
  src/floquet.cpp
  src/heisenberg.cpp
  src/bose_hubbard.cpp
  src/tfi.cpp
  src/ssh.cpp
  src/kitaev.cpp
  src/anderson.cpp
  src/chern.cpp
  src/hofstadter.cpp
  src/mbl.cpp
  src/spectral.cpp
  src/tenfold.cpp
  src/second_chern.cpp
  src/wilson_loop.cpp
  src/edge_states.cpp
  src/haldane.cpp
  src/weyl.cpp
  src/fu_kane.cpp
  src/lutchyn_oreg.cpp
  src/kitaev_honeycomb.cpp
  src/bbh.cpp
  src/floquet_topo.cpp
  src/non_herm.cpp
  src/aah.cpp
  src/syk.cpp
  src/kondo.cpp
  src/luttinger.cpp
  src/toric.cpp
  src/mipt.cpp
  src/richardson_gaudin.cpp
  src/entanglement.cpp
  src/json_utils.cpp
)
set_target_properties(qx_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(qx_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(qx_core PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(qx_core PUBLIC qx_eigen)
if(TARGET spdlog::spdlog_header_only)
  target_link_libraries(qx_core PUBLIC spdlog::spdlog_header_only)
else()
  target_link_libraries(qx_core PUBLIC spdlog::spdlog)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(qx_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(qx_core PUBLIC QX_HAS_OPENMP=1)
endif()

if(QX_ENABLE_MPI)
  target_link_libraries(qx_core PUBLIC MPI::MPI_CXX)
  target_compile_definitions(qx_core PUBLIC QX_HAS_MPI=1)
endif()

if(QX_ENABLE_CUDA AND CUDAToolkit_FOUND)
  target_compile_definitions(qx_core PRIVATE QX_HAS_CUDA=1 QX_HAS_CUFFT=1)
  target_link_libraries(qx_core PUBLIC CUDA::cufft CUDA::cudart)
endif()

if(WIN32)
  target_compile_definitions(qx_core PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

function(qx_set_config_output_dirs target_name)
  set_target_properties(${target_name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
  )
endfunction()

# lindblad_sparse.cpp and dmrg_capi.cpp define QX_API symbols directly.
# They must be compiled as part of the SHARED library (qx_capi), not the
# static qx_core, so that QX_CAPI_EXPORTS=1 causes dllexport to take effect
# and their symbols appear in the exported symbol table of the DLL.
add_library(qx_capi SHARED
  src/c_api.cpp
  src/lindblad_sparse.cpp
  src/dmrg_mpo.cpp
  src/dmrg_dense.cpp
  src/dmrg_blocks.cpp
  src/dmrg_capi.cpp
)
target_link_libraries(qx_capi PRIVATE qx_core)
target_link_libraries(qx_capi PRIVATE qx_blas)
target_include_directories(qx_capi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(qx_capi PRIVATE QX_CAPI_EXPORTS=1)
if(QX_USE_CUDA)
  target_link_libraries(qx_capi PRIVATE qx_gpu)
  target_compile_definitions(qx_capi PRIVATE QX_USE_CUDA=1)
  target_include_directories(qx_capi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
endif()
set_target_properties(qx_capi PROPERTIES OUTPUT_NAME quantx)
qx_set_config_output_dirs(qx_capi)

# ── Wheel install target (M158) ───────────────────────────────────────────────
# When QX_WHEEL_LIB_DIR is set (by build_native_for_wheel.py or CI), cmake
# --install will copy the compiled library directly into the Python package
# directory so it gets bundled into the wheel.
if(DEFINED QX_WHEEL_LIB_DIR)
  install(
    TARGETS qx_capi
    LIBRARY DESTINATION "${QX_WHEEL_LIB_DIR}"
    RUNTIME DESTINATION "${QX_WHEEL_LIB_DIR}"
  )
  message(STATUS "qx_capi wheel install dir: ${QX_WHEEL_LIB_DIR}")
endif()

add_executable(qx_smoke tests/test_engine.cpp)
target_link_libraries(qx_smoke PRIVATE qx_core)
if(NOT WIN32)
  target_link_libraries(qx_smoke PRIVATE dl)
endif()
qx_set_config_output_dirs(qx_smoke)

add_executable(qx_fft_smoke tests/test_fft_provider.cpp)
target_link_libraries(qx_fft_smoke PRIVATE qx_core)
qx_set_config_output_dirs(qx_fft_smoke)

add_executable(qx_cli src/qx_cli.cpp)
target_link_libraries(qx_cli PRIVATE qx_core)
if(TARGET CLI11::CLI11)
  target_link_libraries(qx_cli PRIVATE CLI11::CLI11)
else()
  target_link_libraries(qx_cli PRIVATE CLI11)
endif()
qx_set_config_output_dirs(qx_cli)

add_executable(qx_dmrg_test
  src/dmrg_test.cpp
  src/dmrg_mpo.cpp
  src/dmrg_dense.cpp
  src/dmrg_blocks.cpp
)
target_link_libraries(qx_dmrg_test PRIVATE qx_eigen nlohmann_json::nlohmann_json)
if(OpenMP_CXX_FOUND)
  target_link_libraries(qx_dmrg_test PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(qx_dmrg_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
qx_set_config_output_dirs(qx_dmrg_test)

# Large-block SVD-backend regression test: exercises the >=96-block CPU eigensolve
# (Eigen vs LAPACK) that the ED suite (tiny blocks) never reaches. Links qx_blas so the
# LAPACKE path can be validated via QX_LAPACKE_MIN.
add_executable(qx_large_block_test
  src/large_block_test.cpp
  src/dmrg_mpo.cpp
  src/dmrg_dense.cpp
  src/dmrg_blocks.cpp
)
target_link_libraries(qx_large_block_test PRIVATE qx_eigen nlohmann_json::nlohmann_json qx_blas)
if(OpenMP_CXX_FOUND)
  target_link_libraries(qx_large_block_test PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(qx_large_block_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
qx_set_config_output_dirs(qx_large_block_test)
# n=32 chi=512 reaches ~256-wide central blocks (>=96), exercising the large-block
# eigensolve the ED suite never reaches; assert the known ground-state energy.
add_test(NAME qx_large_block_test COMMAND qx_large_block_test 32 512 -55.989262 1e-3)

add_executable(qx_dmrg_bench
  src/dmrg_bench.cpp
  src/dmrg_mpo.cpp
  src/dmrg_dense.cpp
  src/dmrg_blocks.cpp
)
target_link_libraries(qx_dmrg_bench PRIVATE qx_eigen nlohmann_json::nlohmann_json)
if(OpenMP_CXX_FOUND)
  target_link_libraries(qx_dmrg_bench PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(qx_dmrg_bench PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
qx_set_config_output_dirs(qx_dmrg_bench)

add_executable(qx_su2_test src/su2_test.cpp)
target_include_directories(qx_su2_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
qx_set_config_output_dirs(qx_su2_test)

add_executable(qx_su2_mpo_test src/dmrg_su2_test.cpp)
target_include_directories(qx_su2_mpo_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
qx_set_config_output_dirs(qx_su2_mpo_test)

add_executable(qx_su2_env_test src/dmrg_su2_env_test.cpp)
target_link_libraries(qx_su2_env_test PRIVATE qx_eigen)
target_include_directories(qx_su2_env_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
qx_set_config_output_dirs(qx_su2_env_test)

add_executable(qx_su2_heff_test src/dmrg_su2_heff_test.cpp)
target_link_libraries(qx_su2_heff_test PRIVATE qx_eigen)
target_include_directories(qx_su2_heff_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
qx_set_config_output_dirs(qx_su2_heff_test)

add_executable(qx_su2_run_test src/dmrg_su2_run_test.cpp)
target_link_libraries(qx_su2_run_test PRIVATE qx_eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(qx_su2_run_test PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(qx_su2_run_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
qx_set_config_output_dirs(qx_su2_run_test)

add_executable(qx_su2_bench src/dmrg_su2_bench.cpp)
target_link_libraries(qx_su2_bench PRIVATE qx_eigen)
if(OpenMP_CXX_FOUND)
  target_link_libraries(qx_su2_bench PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(qx_su2_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
qx_set_config_output_dirs(qx_su2_bench)

add_executable(qx_tdvp_test src/tdvp_test.cpp src/dmrg_mpo.cpp)
target_link_libraries(qx_tdvp_test PRIVATE qx_eigen nlohmann_json::nlohmann_json)
target_include_directories(qx_tdvp_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
qx_set_config_output_dirs(qx_tdvp_test)

add_executable(qx_tdvp_blocks_test src/tdvp_blocks_test.cpp
  src/dmrg_mpo.cpp src/dmrg_dense.cpp src/dmrg_blocks.cpp)
target_link_libraries(qx_tdvp_blocks_test PRIVATE qx_eigen nlohmann_json::nlohmann_json qx_blas)
if(OpenMP_CXX_FOUND)
  target_link_libraries(qx_tdvp_blocks_test PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(qx_tdvp_blocks_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(QX_USE_CUDA)
  target_link_libraries(qx_tdvp_blocks_test PRIVATE qx_gpu)
  target_compile_definitions(qx_tdvp_blocks_test PRIVATE QX_USE_CUDA=1)
endif()
qx_set_config_output_dirs(qx_tdvp_blocks_test)

add_executable(qx_thermal_blocks_test src/thermal_blocks_test.cpp
  src/dmrg_mpo.cpp src/dmrg_dense.cpp src/dmrg_blocks.cpp)
target_link_libraries(qx_thermal_blocks_test PRIVATE qx_eigen nlohmann_json::nlohmann_json qx_blas)
if(OpenMP_CXX_FOUND)
  target_link_libraries(qx_thermal_blocks_test PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(qx_thermal_blocks_test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(QX_USE_CUDA)
  target_link_libraries(qx_thermal_blocks_test PRIVATE qx_gpu)
  target_compile_definitions(qx_thermal_blocks_test PRIVATE QX_USE_CUDA=1)
endif()
qx_set_config_output_dirs(qx_thermal_blocks_test)

# Phase A unit check: GPU Gram-SVD vs reconstruction/orthonormality (CUDA only).
if(QX_USE_CUDA)
  add_executable(qx_gpu_svd_test src/gpu_svd_test.cpp)
  target_link_libraries(qx_gpu_svd_test PRIVATE qx_gpu qx_eigen)
  target_include_directories(qx_gpu_svd_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  qx_set_config_output_dirs(qx_gpu_svd_test)
  add_test(NAME qx_gpu_svd_test COMMAND qx_gpu_svd_test)

  # Phase A benchmarks: kernel SVD scaling, and end-to-end TEBD step time.
  add_executable(qx_gpu_svd_bench src/gpu_svd_bench.cpp)
  target_link_libraries(qx_gpu_svd_bench PRIVATE qx_gpu qx_eigen)
  target_include_directories(qx_gpu_svd_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  qx_set_config_output_dirs(qx_gpu_svd_bench)

  add_executable(qx_tebd_bench src/tebd_bench.cpp
    src/dmrg_mpo.cpp src/dmrg_dense.cpp src/dmrg_blocks.cpp)
  target_link_libraries(qx_tebd_bench PRIVATE qx_eigen nlohmann_json::nlohmann_json qx_gpu qx_blas)
  target_compile_definitions(qx_tebd_bench PRIVATE QX_USE_CUDA=1)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(qx_tebd_bench PRIVATE OpenMP::OpenMP_CXX)
  endif()
  target_include_directories(qx_tebd_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
  qx_set_config_output_dirs(qx_tebd_bench)

  add_executable(qx_obs_bench src/obs_bench.cpp
    src/dmrg_mpo.cpp src/dmrg_dense.cpp src/dmrg_blocks.cpp)
  target_link_libraries(qx_obs_bench PRIVATE qx_eigen nlohmann_json::nlohmann_json qx_gpu qx_blas)
  target_compile_definitions(qx_obs_bench PRIVATE QX_USE_CUDA=1)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(qx_obs_bench PRIVATE OpenMP::OpenMP_CXX)
  endif()
  target_include_directories(qx_obs_bench PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
  qx_set_config_output_dirs(qx_obs_bench)
endif()

add_executable(qx_su2_vs_u1
  src/dmrg_su2_vs_u1.cpp
  src/dmrg_mpo.cpp
  src/dmrg_dense.cpp
  src/dmrg_blocks.cpp
)
target_link_libraries(qx_su2_vs_u1 PRIVATE qx_eigen nlohmann_json::nlohmann_json)
if(OpenMP_CXX_FOUND)
  target_link_libraries(qx_su2_vs_u1 PRIVATE OpenMP::OpenMP_CXX)
endif()
target_include_directories(qx_su2_vs_u1 PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)
qx_set_config_output_dirs(qx_su2_vs_u1)

enable_testing()
add_test(NAME qx_smoke COMMAND qx_smoke)
add_test(NAME qx_su2_test COMMAND qx_su2_test)
add_test(NAME qx_su2_mpo_test COMMAND qx_su2_mpo_test)
add_test(NAME qx_su2_env_test COMMAND qx_su2_env_test)
add_test(NAME qx_su2_heff_test COMMAND qx_su2_heff_test)
add_test(NAME qx_su2_run_test COMMAND qx_su2_run_test)
add_test(NAME qx_tdvp_test COMMAND qx_tdvp_test)
add_test(NAME qx_tdvp_blocks_test COMMAND qx_tdvp_blocks_test)
add_test(NAME qx_thermal_blocks_test COMMAND qx_thermal_blocks_test)
add_test(NAME qx_fft_smoke COMMAND qx_fft_smoke)
add_test(NAME qx_dmrg_test COMMAND qx_dmrg_test)

if(QX_ENABLE_PYBIND)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_quantx_py bindings/python/pybind_module.cpp)
  target_link_libraries(_quantx_py PRIVATE qx_core)
endif()
