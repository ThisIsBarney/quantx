#pragma once

#include <string>

namespace qx::log {

// Initialize project logger singleton with default sink/pattern/level.
void init_logging();

// Set logger verbosity from string token (trace/debug/info/warn/error/off).
void set_log_level(const std::string& level);

// Emit debug-level message.
void debug(const std::string& message);

// Emit info-level message.
void info(const std::string& message);

// Emit warn-level message.
void warn(const std::string& message);

// Emit error-level message.
void error(const std::string& message);

}  // namespace qx::log
