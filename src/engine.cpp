#include "qx/engine.hpp"

#include <filesystem>
#include <stdexcept>

#include "qx/backend.hpp"
#include "qx/backends.hpp"
#include "qx/runtime.hpp"
#include "qx/scheduler.hpp"

namespace qx {
namespace {

void ensure_builtin_backends() {
  static bool initialized = false;
  if (initialized) {
    return;
  }
  register_builtin_backends();
  initialized = true;
}

}  // namespace

void register_builtin_backends() {
  auto& registry = BackendRegistry::instance();
  registry.register_backend("statevector", &make_statevector_backend, make_statevector_backend()->capabilities());
  registry.register_backend("mps", &make_mps_backend, make_mps_backend()->capabilities());
}

BackendInfo explain_backend_choice(const QuantumProblem& problem, const RunOptions& options) {
  ensure_builtin_backends();
  return explain_backend_choice_internal(problem, options);
}

SimulationResult run(const QuantumProblem& problem, const RunOptions& options) {
  ensure_builtin_backends();

  SimulationResult result;
  RuntimeContext runtime(options);
  runtime.emit_event("runtime_initialized",
                     {{"threads", static_cast<double>(runtime.effective_threads())},
                      {"mpi_ranks", static_cast<double>(runtime.effective_mpi_ranks())}});

  const std::string backend_name = choose_backend(problem, options);
  auto backend = BackendRegistry::instance().create(backend_name);

  if (options.enable_checkpoint) {
    std::filesystem::create_directories(options.checkpoint_path);
    backend->restore(options.checkpoint_path);
    runtime.emit_event("checkpoint_restore_attempt", {{"success", 1.0}});
  }

  backend->initialize(problem, options);
  runtime.emit_event("backend_initialized", {{"qubits", static_cast<double>(problem.qubits)}});
  backend->evolve(problem, options);
  runtime.emit_event("evolution_complete", {{"total_time", problem.evolution.total_time}});

  result.ok = true;
  result.backend = backend->name();
  result.observables = backend->measure(problem);
  result.state = backend->current_state();
  result.events = runtime.events();
  result.message = "Simulation completed";

  if (options.enable_checkpoint) {
    backend->checkpoint(options.checkpoint_path);
    result.events.push_back(Event{"checkpoint_saved", {{"enabled", 1.0}}});
  }

  return result;
}

}  // namespace qx
