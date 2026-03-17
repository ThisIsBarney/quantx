cmake_minimum_required(VERSION 3.20)
project(quantx_engine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QX_ENABLE_OPENMP "Enable OpenMP parallel backend" ON)
option(QX_ENABLE_MPI "Enable MPI features" OFF)
option(QX_ENABLE_PYBIND "Build pybind11 bindings" OFF)

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
  src/statevector_backend.cpp
  src/mps_backend.cpp
  src/engine.cpp
  src/json_utils.cpp
)

target_include_directories(qx_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

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

enable_testing()
add_test(NAME qx_smoke COMMAND qx_smoke)

if(QX_ENABLE_PYBIND)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_quantx_py bindings/python/pybind_module.cpp)
  target_link_libraries(_quantx_py PRIVATE qx_core)
endif()
