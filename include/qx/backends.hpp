#pragma once

#include <memory>

#include "qx/backend.hpp"

namespace qx {

// Factory for dense statevector backend.
std::unique_ptr<IBackend> make_statevector_backend();
// Factory for tensor-network (MPS) backend.
std::unique_ptr<IBackend> make_mps_backend();

}  // namespace qx
