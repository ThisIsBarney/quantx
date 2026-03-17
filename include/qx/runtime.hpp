#pragma once

#include <string>
#include <vector>

#include "qx/types.hpp"

namespace qx {

// Lightweight runtime helper for resource resolution and event collection.
class RuntimeContext {
 public:
  // Resolve effective runtime resources from user options and host defaults.
  explicit RuntimeContext(const RunOptions& options);

  // Append a runtime event with numeric metrics.
  void emit_event(const std::string& name, const std::map<std::string, double>& metrics);
  // Return captured runtime events in chronological order.
  const std::vector<Event>& events() const;
  // Effective CPU thread count used by this run.
  std::uint32_t effective_threads() const;
  // Effective MPI rank count used by this run.
  std::uint32_t effective_mpi_ranks() const;

 private:
  RunOptions options_;
  std::vector<Event> events_;
  std::uint32_t threads_{1};
  std::uint32_t mpi_ranks_{1};
};

}  // namespace qx
