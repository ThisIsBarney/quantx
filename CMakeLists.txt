cmake_minimum_required(VERSION 3.20)
project(quantx_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QX_ENABLE_OPENMP "Enable OpenMP parallel backend" ON)
option(QX_ENABLE_MPI "Enable MPI features" OFF)
option(QX_ENABLE_PYBIND "Build pybind11 bindings" OFF)

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

add_library(qx_core
  src/backend_registry.cpp
  src/runtime.cpp
  src/scheduler.cpp
  src/logging.cpp
  src/statevector_backend.cpp
  src/mps_backend.cpp
  src/engine.cpp
  src/json_utils.cpp
)

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

if(WIN32)
  target_compile_definitions(qx_core PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

add_library(qx_capi SHARED src/c_api.cpp)
target_link_libraries(qx_capi PRIVATE qx_core)
target_include_directories(qx_capi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(qx_capi PRIVATE QX_CAPI_EXPORTS=1)
set_target_properties(qx_capi PROPERTIES OUTPUT_NAME quantx)

add_executable(qx_smoke tests/test_engine.cpp)
target_link_libraries(qx_smoke PRIVATE qx_core)

add_executable(qx_cli src/qx_cli.cpp)
target_link_libraries(qx_cli PRIVATE qx_core)
if(TARGET CLI11::CLI11)
  target_link_libraries(qx_cli PRIVATE CLI11::CLI11)
else()
  target_link_libraries(qx_cli PRIVATE CLI11)
endif()

enable_testing()
add_test(NAME qx_smoke COMMAND qx_smoke)

if(QX_ENABLE_PYBIND)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_quantx_py bindings/python/pybind_module.cpp)
  target_link_libraries(_quantx_py PRIVATE qx_core)
endif()
