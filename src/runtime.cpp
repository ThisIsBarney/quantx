#include "qx/runtime.hpp"

#include <thread>

namespace qx {

RuntimeContext::RuntimeContext(const RunOptions& options) : options_(options) {
  const std::uint32_t hw = static_cast<std::uint32_t>(std::thread::hardware_concurrency());
  threads_ = options.resources.max_threads > 0 ? options.resources.max_threads : (hw == 0 ? 1 : hw);
  mpi_ranks_ = options.resources.mpi_ranks > 0 ? options.resources.mpi_ranks : 1;
}

void RuntimeContext::emit_event(const std::string& name, const std::map<std::string, double>& metrics) {
  events_.push_back(Event{name, metrics});
}

const std::vector<Event>& RuntimeContext::events() const {
  return events_;
}

std::uint32_t RuntimeContext::effective_threads() const {
  return threads_;
}

std::uint32_t RuntimeContext::effective_mpi_ranks() const {
  return mpi_ranks_;
}

}  // namespace qx
