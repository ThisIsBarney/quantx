#include "qx/version.hpp"

namespace qx {

// Return semantic version string for this build.
const std::string& version_string() {
  // QX_VERSION_STRING comes from CMake, which reads python/quantx_engine/__init__.py --
  // the single place the version is written. No default: a build that does not pass it
  // should fail to compile, not report a made-up version.
  static const std::string kVersion = QX_VERSION_STRING;
  return kVersion;
}

}  // namespace qx
