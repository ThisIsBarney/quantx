#pragma once

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#ifdef _MSC_VER
#include <crtdbg.h>
#endif

namespace qx::test {

// Route MSVC CRT diagnostics to stderr so smoke tests fail in the console instead of opening modal popups.
inline void configure_console_diagnostics() {
#ifdef _MSC_VER
#ifdef _DEBUG
  _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
  _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
  _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
#endif
#endif
}

// Print a caught std::exception to stderr and return the smoke-test failure code.
inline int report_exception(const std::exception& ex) {
  std::cerr << "EXCEPTION: " << ex.what() << "\n";
  return 2;
}

// Print an unknown failure to stderr and return the smoke-test failure code.
inline int report_unknown_exception() {
  std::cerr << "EXCEPTION: unknown\n";
  return 2;
}

}  // namespace qx::test
