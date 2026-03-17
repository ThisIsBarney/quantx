#pragma once

#include <memory>

#include "qx/backend.hpp"

namespace qx {

std::unique_ptr<IBackend> make_statevector_backend();
std::unique_ptr<IBackend> make_mps_backend();

}  // namespace qx
