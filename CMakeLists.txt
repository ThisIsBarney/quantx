cmake_minimum_required(VERSION 3.20)
project(quantx_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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
  src/toric.cpp
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

add_library(qx_capi SHARED src/c_api.cpp)
target_link_libraries(qx_capi PRIVATE qx_core)
target_include_directories(qx_capi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(qx_capi PRIVATE QX_CAPI_EXPORTS=1)
set_target_properties(qx_capi PROPERTIES OUTPUT_NAME quantx)
qx_set_config_output_dirs(qx_capi)

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

enable_testing()
add_test(NAME qx_smoke COMMAND qx_smoke)
add_test(NAME qx_fft_smoke COMMAND qx_fft_smoke)

if(QX_ENABLE_PYBIND)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_quantx_py bindings/python/pybind_module.cpp)
  target_link_libraries(_quantx_py PRIVATE qx_core)
endif()
