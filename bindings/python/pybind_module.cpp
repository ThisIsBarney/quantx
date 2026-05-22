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
 *   explain(problem_json)     -> dict
 *   circuit_capabilities()    -> dict
 *   version()                 -> str
 *   list_solvers()            -> list[str]
 *
 * and backward-compatible wrappers:
 *
 *   run_cpp(problem, options)     -> dict   (JSON round-trip — legacy)
 *   explain_cpp(problem, options) -> dict   (JSON round-trip — legacy)
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
#include "qx/version.hpp"

namespace py = pybind11;
using cdouble = std::complex<double>;


// ── Internal helpers ─────────────────────────────────────────────────────────

/**
 * Copy a std::vector<cdouble> into a 1-D NumPy array (complex128).
 *
 * We intentionally copy rather than share memory so that the C++ result object
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
 * Parse a JSON string and return it as a Python dict via json.loads().
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
 */
static py::array_t<cdouble> run_statevector(const std::string& problem_json,
                                            const std::string& options_json = "{}") {
    const auto problem = qx::problem_from_json(problem_json);
    const auto options = qx::options_from_json(options_json);
    const auto result  = qx::run(problem, options);
    return vec_to_numpy(result.state);
}

/**
 * run_circuit_sv(circuit_json) -> np.ndarray[complex128]
 *
 * Execute a gate-level quantum circuit and return its output statevector as a
 * 1-D complex128 NumPy array.
 *
 * CircuitResult::state holds the final statevector (see include/qx/types.hpp).
 */
static py::array_t<cdouble> run_circuit_sv(const std::string& circuit_json) {
    const auto circuit = qx::circuit_from_json(circuit_json);
    const auto result  = qx::run_circuit(circuit, qx::RunOptions{});
    return vec_to_numpy(result.state);   // CircuitResult::state, not ::statevector
}

/**
 * run_circuit_counts(circuit_json) -> dict[str, int]
 *
 * Execute a gate-level quantum circuit and return measurement counts as a
 * Python dict mapping bitstring → count.
 */
static py::object run_circuit_counts(const std::string& circuit_json) {
    const auto circuit = qx::circuit_from_json(circuit_json);
    const auto result  = qx::run_circuit(circuit, qx::RunOptions{});
    py::dict counts;
    for (const auto& [bitstring, count] : result.counts) {
        counts[py::str(bitstring)] = py::int_(static_cast<long long>(count));
    }
    return counts;
}


// ── Metadata helpers ─────────────────────────────────────────────────────────

/**
 * explain(problem_json) -> dict
 *
 * Return the backend-choice explanation for a given problem.
 * Uses backend_info_to_json() for serialisation (BackendInfo, not SimulationResult).
 */
static py::object explain(const std::string& problem_json) {
    const auto problem = qx::problem_from_json(problem_json);
    const auto info    = qx::explain_backend_choice(problem, qx::RunOptions{});
    return json_str_to_pydict(qx::backend_info_to_json(info));
}

/**
 * circuit_capabilities() -> dict
 *
 * Return the gate set and capabilities of the circuit simulator backend.
 */
static py::object circuit_capabilities() {
    const auto caps = qx::circuit_backend_capabilities();
    return json_str_to_pydict(qx::circuit_capabilities_to_json(caps));
}

/**
 * version() -> str
 *
 * Return the native engine version string.
 * Uses qx::version_string() declared in include/qx/version.hpp.
 */
static std::string version() {
    return qx::version_string();
}

/**
 * list_solvers() -> list[str]
 *
 * Return the names of all 37 available physics solvers.
 * This list mirrors the one in src/c_api.cpp::qx_list_solvers().
 */
static std::vector<std::string> list_solvers() {
    return {
        "vqe", "qpe", "qec", "qfield", "phi4", "hubbard", "floquet",
        "heisenberg", "bose_hubbard", "tfi", "ssh", "kitaev", "anderson",
        "chern", "hofstadter", "mbl", "spectral", "tenfold", "second_chern",
        "wilson_loop", "edge_states", "haldane", "weyl", "fu_kane",
        "lutchyn_oreg", "kitaev_honeycomb", "bbh", "floquet_topo",
        "non_herm", "aah", "syk", "kondo", "luttinger", "toric",
        "mipt", "rg", "entanglement"
    };
}


// ── Backward-compatible JSON wrappers ────────────────────────────────────────

/**
 * run_cpp(problem, options) -> dict   [legacy — JSON round-trip]
 *
 * Accepts QuantumProblem and RunOptions objects and returns the full result as
 * a Python dict.  Kept for backward compatibility.
 */
static py::object run_cpp_compat(const qx::QuantumProblem& problem,
                                  const qx::RunOptions&     options) {
    const auto result = qx::run(problem, options);
    return json_str_to_pydict(qx::result_to_json(result));
}

/**
 * explain_cpp(problem, options) -> dict   [legacy — JSON round-trip]
 */
static py::object explain_cpp_compat(const qx::QuantumProblem& problem,
                                      const qx::RunOptions&     options) {
    const auto info = qx::explain_backend_choice(problem, options);
    return json_str_to_pydict(qx::backend_info_to_json(info));
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
        explain(problem_json)   -> dict
        circuit_capabilities()  -> dict
        version()               -> str
        list_solvers()          -> list[str]

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
        Result of a Hamiltonian simulation.

        Attributes
        ----------
        state : np.ndarray[complex128]
            Final statevector (copied to NumPy on access).
        backend : str
        elapsed_ms : float
    )pbdoc")
        .def(py::init<>())
        .def_property_readonly("state",
            [](const qx::SimulationResult& r) { return vec_to_numpy(r.state); },
            "Final statevector as a complex128 NumPy array.")
        .def_readwrite("backend",     &qx::SimulationResult::backend)
        .def_readwrite("elapsed_ms",  &qx::SimulationResult::elapsed_ms);

    py::class_<qx::CircuitResult>(m, "CircuitResult", R"pbdoc(
        Result of a circuit-level simulation.

        Attributes
        ----------
        state : np.ndarray[complex128]
            Output statevector (CircuitResult::state, copied to NumPy on access).
        counts : dict[str, int]
            Measurement counts (bitstring → count).
        ok : bool
        elapsed_ms : float
    )pbdoc")
        .def(py::init<>())
        .def_property_readonly("state",
            [](const qx::CircuitResult& r) { return vec_to_numpy(r.state); },
            "Output statevector as a complex128 NumPy array.")
        .def_property_readonly("counts",
            [](const qx::CircuitResult& r) {
                py::dict d;
                for (const auto& [k, v] : r.counts) {
                    d[py::str(k)] = py::int_(static_cast<long long>(v));
                }
                return d;
            })
        .def_readwrite("ok",          &qx::CircuitResult::ok)
        .def_readwrite("elapsed_ms",  &qx::CircuitResult::elapsed_ms);

    // ── Hot-path functions ────────────────────────────────────────────────

    m.def("run_statevector", &run_statevector,
          py::arg("problem_json"),
          py::arg("options_json") = "{}",
          R"pbdoc(
        Run quantum simulation and return statevector as NumPy array.

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

        Uses CircuitResult::state (the output statevector field).

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

    m.def("explain", &explain,
          py::arg("problem_json"),
          "Return backend-choice explanation for the given problem as a dict.");

    m.def("circuit_capabilities", &circuit_capabilities,
          "Return gate set and capabilities of the circuit simulator as a dict.");

    m.def("version", &version,
          "Return the native engine version string (from qx::version_string()).");

    m.def("list_solvers", &list_solvers,
          "Return the names of all 37 available physics solvers.");

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
