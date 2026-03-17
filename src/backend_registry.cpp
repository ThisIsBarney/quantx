#include "qx/backend.hpp"

#include <stdexcept>

namespace qx {

// Return global backend registry singleton.
BackendRegistry& BackendRegistry::instance() {
  static BackendRegistry registry;
  return registry;
}

// Register or replace backend record by name.
void BackendRegistry::register_backend(
    const std::string& name, BackendFactory factory, BackendCapabilities capabilities) {
  records_[name] = Record{std::move(factory), capabilities};
}

// List registered backend names.
std::vector<std::string> BackendRegistry::names() const {
  std::vector<std::string> out;
  out.reserve(records_.size());
  for (const auto& [name, _] : records_) {
    out.push_back(name);
  }
  return out;
}

// Create backend instance by name.
std::unique_ptr<IBackend> BackendRegistry::create(const std::string& name) const {
  const auto it = records_.find(name);
  if (it == records_.end()) {
    throw std::runtime_error("Backend not found: " + name);
  }
  return it->second.factory();
}

// Return static capabilities for a registered backend.
BackendCapabilities BackendRegistry::capabilities_for(const std::string& name) const {
  const auto it = records_.find(name);
  if (it == records_.end()) {
    throw std::runtime_error("Backend not found: " + name);
  }
  return it->second.capabilities;
}

}  // namespace qx
