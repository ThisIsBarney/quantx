#pragma once

#include <string>
#include <vector>

#include "qx/types.hpp"

namespace qx {

class RuntimeContext {
 public:
  explicit RuntimeContext(const RunOptions& options);

  void emit_event(const std::string& name, const std::map<std::string, double>& metrics);
  const std::vector<Event>& events() const;
  std::uint32_t effective_threads() const;
  std::uint32_t effective_mpi_ranks() const;

 private:
  RunOptions options_;
  std::vector<Event> events_;
  std::uint32_t threads_{1};
  std::uint32_t mpi_ranks_{1};
};

}  // namespace qx
