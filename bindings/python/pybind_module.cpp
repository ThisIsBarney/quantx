#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "qx/engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_quantx_py, m) {
  m.doc() = "QuantX Python bindings";

  py::class_<qx::QuantumProblem>(m, "QuantumProblem")
      .def(py::init<>())
      .def_readwrite("qubits", &qx::QuantumProblem::qubits)
      .def_readwrite("target_error", &qx::QuantumProblem::target_error)
      .def_readwrite("noisy", &qx::QuantumProblem::noisy)
      .def_readwrite("expected_circuit_depth", &qx::QuantumProblem::expected_circuit_depth);

  py::class_<qx::RunOptions>(m, "RunOptions")
      .def(py::init<>())
      .def_readwrite("use_gpu", &qx::RunOptions::use_gpu)
      .def_readwrite("allow_mpi", &qx::RunOptions::allow_mpi)
      .def_readwrite("enable_checkpoint", &qx::RunOptions::enable_checkpoint)
      .def_readwrite("checkpoint_path", &qx::RunOptions::checkpoint_path);

  m.def("run_cpp", &qx::run);
  m.def("explain_cpp", &qx::explain_backend_choice);
}
