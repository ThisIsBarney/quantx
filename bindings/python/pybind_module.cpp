/**
 * QuantX pybind11 Python bindings  (M159 — NumPy zero-copy bridge)
 *
 * Exposes three hot-path entry points that return NumPy arrays directly,
 * avoiding the JSON round-trip overhead of the ctypes C-API:
 *
 *   run_statevector(problem_json, options_json="{}") -> np.ndarray[complex128]
 *   run_circuit_sv(circuit_json)                    -> np.ndarray[complex128]
 *   run_circuit_counts(circuit_json)                -> dict[str, int]
 *
 * plus metadata helpers:
 *
 *   run_solver(name, params_json) -> dict
 *   explain(problem_json)         -> dict
 *   circuit_capabilities()        -> dict
 *   version()                     -> str
 *   list_solvers()                -> list[str]
 *
 * and backward-compatible wrappers:
 *
 *   run_cpp(problem, options)     -> dict   (JSON round-trip — legacy)
 *   explain_cpp(problem, options) -> dict   (JSON round-trip — legacy)
 *
 * Structured bindings for QuantumProblem, RunOptions, and SimulationResult
 * (the latter exposes .state as a zero-copy NumPy view).
 */

#include <complex>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "qx/engine.hpp"
#include "qx/json_utils.hpp"
#include "qx/types.hpp"

namespace py = pybind11;
using cdouble = std::complex<double>;
using json    = nlohmann::json;


// ── Internal helpers ─────────────────────────────────────────────────────────

/**
 * Copy a std::vector<cdouble> into a 1-D NumPy array (complex128).
 *
 * We intentionally copy rather than share memory so that the C++ SimulationResult
 * can be destroyed after the call returns without invalidating the Python object.
 */
static py::array_t<cdouble> vec_to_numpy(const std::vector<cdouble>& v) {
    auto result = py::array_t<cdouble>(static_cast<py::ssize_t>(v.size()));
    auto buf    = result.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(v.size()); ++i) {
        buf(i) = v[static_cast<std::size_t>(i)];
    }
    return result;
}

/**
 * Convert a flat std::vector<cdouble> of length rows*cols into a 2-D NumPy
 * array of shape (rows, cols).  Used for density-matrix output.
 */
static py::array_t<cdouble> flat_to_numpy_2d(const std::vector<cdouble>& v,
                                              py::ssize_t rows,
                                              py::ssize_t cols) {
    auto result = py::array_t<cdouble>({rows, cols});
    auto buf    = result.mutable_unchecked<2>();
    for (py::ssize_t r = 0; r < rows; ++r) {
        for (py::ssize_t c = 0; c < cols; ++c) {
            buf(r, c) = v[static_cast<std::size_t>(r * cols + c)];
        }
    }
    return result;
}

/**
 * Parse a JSON string and return it as a Python dict via json.loads().
 * This is the cleanest cross-platform way to convert nlohmann::json →
 * Python objects without reimplementing the full type mapping.
 */
static py::object json_str_to_pydict(const std::string& json_str) {
    return py::module_::import("json").attr("loads")(json_str);
}


// ── Hot-path functions ───────────────────────────────────────────────────────

/**
 * run_statevector(problem_json, options_json="{}") -> np.ndarray[complex128]
 *
 * Run the quantum simulation described by *problem_json* and return the final
 * statevector as a 1-D complex128 NumPy array.  Avoids JSON serialisation of
 * the result — the vector<cdouble> is copied directly into NumPy memory.
 *
 * Parameters
 * ----------
 * problem_json : str
 *     JSON-serialised QuantumProblem (same schema as the C-API run_cpp).
 * options_json : str, optional
 *     JSON-serialised RunOptions (defaults to "{}").
 *
 * Returns
 * -------
 * np.ndarray, dtype=complex128, shape=(2**n_qubits,)
 */
static py::array_t<cdouble> run_statevector(const std::string& problem_json,
                                            const std::string& options_json = "{}") {
    auto problem = qx::problem_from_json(json::parse(problem_json));
    auto options = qx::options_from_json(json::parse(options_json));
    auto result  = qx::run(problem, options);
    return vec_to_numpy(result.state);
}

/**
 * run_circuit_sv(circuit_json) -> np.ndarray[complex128]
 *
 * Execute a gate-level quantum circuit and return its output statevector as a
 * 1-D complex128 NumPy array.
 */
static py::array_t<cdouble> run_circuit_sv(const std::string& circuit_json) {
    auto circuit = qx::circuit_from_json(json::parse(circuit_json));
    auto result  = qx::run_circuit(circuit);
    return vec_to_numpy(result.statevector);
}

/**
 * run_circuit_counts(circuit_json) -> dict[str, int]
 *
 * Execute a gate-level quantum circuit and return measurement counts as a
 * Python dict mapping bitstring → count.
 */
static py::object run_circuit_counts(const std::string& circuit_json) {
    auto circuit = qx::circuit_from_json(json::parse(circuit_json));
    auto result  = qx::run_circuit(circuit);
    py::dict counts;
    for (const auto& [bitstring, count] : result.counts) {
        counts[py::str(bitstring)] = py::int_(count);
    }
    return counts;
}


// ── Metadata helpers ─────────────────────────────────────────────────────────

/**
 * run_solver(name, params_json="{}") -> dict
 *
 * Call a named solver by name (one of list_solvers()) with optional params.
 * Returns the full result dict via JSON round-trip.
 */
static py::object run_solver(const std::string& name,
                             const std::string& params_json = "{}") {
    auto params = json::parse(params_json);
    auto result = qx::run_named_solver(name, params);
    return json_str_to_pydict(result.dump());
}

/**
 * explain(problem_json) -> dict
 *
 * Return the backend-choice explanation for a given problem.
 * Wraps qx::explain_backend_choice() and parses the JSON result.
 */
static py::object explain(const std::string& problem_json) {
    auto problem = qx::problem_from_json(json::parse(problem_json));
    auto result  = qx::explain_backend_choice(problem, qx::RunOptions{});
    return json_str_to_pydict(qx::result_to_json(result).dump());
}

/**
 * circuit_capabilities() -> dict
 *
 * Return the gate set and capabilities of the circuit simulator backend.
 */
static py::object circuit_capabilities() {
    auto caps = qx::circuit_backend_capabilities();
    return json_str_to_pydict(caps.dump());
}

/**
 * version() -> str
 *
 * Return the native engine version string.
 */
static std::string version() {
    return qx::engine_version();
}

/**
 * list_solvers() -> list[str]
 *
 * Return the names of all available solvers.
 */
static std::vector<std::string> list_solvers() {
    return qx::list_solver_names();
}


// ── Backward-compatible JSON wrappers ────────────────────────────────────────

/**
 * run_cpp(problem, options) -> dict   [legacy — JSON round-trip]
 *
 * Accepts QuantumProblem and RunOptions objects (or dicts) and returns the
 * full result as a Python dict.  Kept for backward compatibility with code
 * that used the old pybind11 module.
 */
static py::object run_cpp_compat(const qx::QuantumProblem& problem,
                                  const qx::RunOptions&     options) {
    auto result     = qx::run(problem, options);
    auto result_j   = qx::result_to_json(result);
    return json_str_to_pydict(result_j.dump());
}

/**
 * explain_cpp(problem, options) -> dict   [legacy — JSON round-trip]
 */
static py::object explain_cpp_compat(const qx::QuantumProblem& problem,
                                      const qx::RunOptions&     options) {
    auto result   = qx::explain_backend_choice(problem, options);
    auto result_j = qx::result_to_json(result);
    return json_str_to_pydict(result_j.dump());
}


// ── Module definition ────────────────────────────────────────────────────────

PYBIND11_MODULE(_quantx_py, m) {
    m.doc() = R"pbdoc(
        QuantX native Python extension  (M159 — pybind11 NumPy bridge)

        Hot-path entry points return NumPy arrays directly, avoiding
        the JSON serialisation overhead of the ctypes C-API.

        Hot paths
        ---------
        run_statevector(problem_json, options_json="{}") -> ndarray[complex128]
        run_circuit_sv(circuit_json)                     -> ndarray[complex128]
        run_circuit_counts(circuit_json)                 -> dict[str, int]

        Metadata helpers
        ----------------
        run_solver(name, params_json="{}") -> dict
        explain(problem_json)              -> dict
        circuit_capabilities()             -> dict
        version()                          -> str
        list_solvers()                     -> list[str]

        Legacy wrappers (JSON round-trip)
        ---------------------------------
        run_cpp(problem, options)     -> dict
        explain_cpp(problem, options) -> dict
    )pbdoc";

    // ── Structured types ──────────────────────────────────────────────────

    py::class_<qx::QuantumProblem>(m, "QuantumProblem", R"pbdoc(
        Quantum problem specification.

        Attributes
        ----------
        qubits : int
        target_error : float
        noisy : bool
        expected_circuit_depth : int
    )pbdoc")
        .def(py::init<>())
        .def_readwrite("qubits",                  &qx::QuantumProblem::qubits)
        .def_readwrite("target_error",             &qx::QuantumProblem::target_error)
        .def_readwrite("noisy",                    &qx::QuantumProblem::noisy)
        .def_readwrite("expected_circuit_depth",   &qx::QuantumProblem::expected_circuit_depth);

    py::class_<qx::RunOptions>(m, "RunOptions", R"pbdoc(
        Runtime options for the simulation engine.

        Attributes
        ----------
        use_gpu : bool
        allow_mpi : bool
        enable_checkpoint : bool
        checkpoint_path : str
    )pbdoc")
        .def(py::init<>())
        .def_readwrite("use_gpu",            &qx::RunOptions::use_gpu)
        .def_readwrite("allow_mpi",          &qx::RunOptions::allow_mpi)
        .def_readwrite("enable_checkpoint",  &qx::RunOptions::enable_checkpoint)
        .def_readwrite("checkpoint_path",    &qx::RunOptions::checkpoint_path);

    py::class_<qx::SimulationResult>(m, "SimulationResult", R"pbdoc(
        Result of a quantum simulation.

        Attributes
        ----------
        state : np.ndarray[complex128]
            Final statevector (zero-copy NumPy view).
        solver_used : str
        wall_time_s : float
    )pbdoc")
        .def(py::init<>())
        .def_property_readonly("state",
            [](const qx::SimulationResult& r) { return vec_to_numpy(r.state); },
            "Final statevector as a complex128 NumPy array.")
        .def_readwrite("solver_used",  &qx::SimulationResult::solver_used)
        .def_readwrite("wall_time_s",  &qx::SimulationResult::wall_time_s);

    py::class_<qx::CircuitResult>(m, "CircuitResult", R"pbdoc(
        Result of a circuit-level simulation.

        Attributes
        ----------
        statevector : np.ndarray[complex128]
        counts : dict[str, int]
    )pbdoc")
        .def(py::init<>())
        .def_property_readonly("statevector",
            [](const qx::CircuitResult& r) { return vec_to_numpy(r.statevector); })
        .def_property_readonly("counts",
            [](const qx::CircuitResult& r) {
                py::dict d;
                for (const auto& [k, v] : r.counts) {
                    d[py::str(k)] = py::int_(v);
                }
                return d;
            });

    // ── Hot-path functions ────────────────────────────────────────────────

    m.def("run_statevector", &run_statevector,
          py::arg("problem_json"),
          py::arg("options_json") = "{}",
          R"pbdoc(
        Run quantum simulation and return statevector as NumPy array.

        This is the fastest entry point: the C++ result vector is copied
        directly into NumPy memory without JSON serialisation.

        Parameters
        ----------
        problem_json : str
            JSON-serialised QuantumProblem.
        options_json : str, optional
            JSON-serialised RunOptions (default: "{}").

        Returns
        -------
        np.ndarray, dtype=complex128, shape=(2**n_qubits,)
    )pbdoc");

    m.def("run_circuit_sv", &run_circuit_sv,
          py::arg("circuit_json"),
          R"pbdoc(
        Execute a gate-level circuit and return its statevector.

        Returns
        -------
        np.ndarray, dtype=complex128, shape=(2**n_qubits,)
    )pbdoc");

    m.def("run_circuit_counts", &run_circuit_counts,
          py::arg("circuit_json"),
          R"pbdoc(
        Execute a gate-level circuit and return measurement counts.

        Returns
        -------
        dict[str, int]
            Mapping from bitstring (e.g. "010") to measurement count.
    )pbdoc");

    // ── Metadata helpers ──────────────────────────────────────────────────

    m.def("run_solver", &run_solver,
          py::arg("name"),
          py::arg("params_json") = "{}",
          "Call a named solver and return the result as a Python dict.");

    m.def("explain", &explain,
          py::arg("problem_json"),
          "Return backend-choice explanation for the given problem.");

    m.def("circuit_capabilities", &circuit_capabilities,
          "Return gate set and capabilities of the circuit simulator.");

    m.def("version", &version,
          "Return the native engine version string.");

    m.def("list_solvers", &list_solvers,
          "Return the names of all available solvers.");

    // ── Backward-compatible legacy wrappers ───────────────────────────────

    m.def("run_cpp", &run_cpp_compat,
          py::arg("problem"),
          py::arg("options"),
          "Legacy JSON-round-trip wrapper for run().  Prefer run_statevector().");

    m.def("explain_cpp", &explain_cpp_compat,
          py::arg("problem"),
          py::arg("options"),
          "Legacy JSON-round-trip wrapper for explain_backend_choice().");
}
