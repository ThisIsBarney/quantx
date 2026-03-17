#include "qx/backend.hpp"

#include <stdexcept>

namespace qx {

BackendRegistry& BackendRegistry::instance() {
  static BackendRegistry registry;
  return registry;
}

void BackendRegistry::register_backend(
    const std::string& name, BackendFactory factory, BackendCapabilities capabilities) {
  records_[name] = Record{std::move(factory), capabilities};
}

std::vector<std::string> BackendRegistry::names() const {
  std::vector<std::string> out;
  out.reserve(records_.size());
  for (const auto& [name, _] : records_) {
    out.push_back(name);
  }
  return out;
}

std::unique_ptr<IBackend> BackendRegistry::create(const std::string& name) const {
  const auto it = records_.find(name);
  if (it == records_.end()) {
    throw std::runtime_error("Backend not found: " + name);
  }
  return it->second.factory();
}

BackendCapabilities BackendRegistry::capabilities_for(const std::string& name) const {
  const auto it = records_.find(name);
  if (it == records_.end()) {
    throw std::runtime_error("Backend not found: " + name);
  }
  return it->second.capabilities;
}

}  // namespace qx
