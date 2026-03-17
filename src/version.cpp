#include "qx/version.hpp"

namespace qx {

// Return semantic version string for this build.
const std::string& version_string() {
  static const std::string kVersion = "0.1.0-dev";
  return kVersion;
}

}  // namespace qx
