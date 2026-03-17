#include "qx/logging.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <mutex>
#include <string>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace qx::log {
namespace {

// Return lower-cased copy for level parsing.
std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// Map textual level to spdlog level enum.
spdlog::level::level_enum parse_level(const std::string& level) {
  const std::string l = to_lower(level);
  if (l == "trace") return spdlog::level::trace;
  if (l == "debug") return spdlog::level::debug;
  if (l == "info") return spdlog::level::info;
  if (l == "warn" || l == "warning") return spdlog::level::warn;
  if (l == "error") return spdlog::level::err;
  if (l == "off") return spdlog::level::off;
  return spdlog::level::info;
}

// Create or fetch the singleton logger instance.
std::shared_ptr<spdlog::logger> get_logger() {
  static std::once_flag once;
  static std::shared_ptr<spdlog::logger> logger;
  std::call_once(once, []() {
    logger = spdlog::get("qx");
    if (!logger) {
      // Use stderr sink so JSON result streams on stdout remain machine-parseable.
      logger = spdlog::stderr_color_mt("qx");
    }
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
    logger->set_level(spdlog::level::warn);
    logger->flush_on(spdlog::level::warn);
  });
  return logger;
}

}  // namespace

// Initialize logger singleton.
void init_logging() {
  (void)get_logger();
}

// Configure global logger level.
void set_log_level(const std::string& level) {
  get_logger()->set_level(parse_level(level));
}

// Write debug message.
void debug(const std::string& message) {
  get_logger()->debug("{}", message);
}

// Write info message.
void info(const std::string& message) {
  get_logger()->info("{}", message);
}

// Write warning message.
void warn(const std::string& message) {
  get_logger()->warn("{}", message);
}

// Write error message.
void error(const std::string& message) {
  get_logger()->error("{}", message);
}

}  // namespace qx::log
