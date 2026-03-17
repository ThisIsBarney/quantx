#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <CLI/CLI.hpp>

#include "qx/engine.hpp"
#include "qx/json_utils.hpp"
#include "qx/logging.hpp"
#include "qx/version.hpp"

namespace {

// Read full text file.
std::string read_text_file(const std::string& path) {
  std::ifstream is(path, std::ios::binary);
  if (!is.good()) {
    throw std::runtime_error("failed to open file: " + path);
  }
  std::ostringstream os;
  os << is.rdbuf();
  return os.str();
}

// Write full text file.
void write_text_file(const std::string& path, const std::string& data) {
  std::ofstream os(path, std::ios::binary);
  if (!os.good()) {
    throw std::runtime_error("failed to write file: " + path);
  }
  os << data;
}

}  // namespace

// CLI entrypoint for problem/circuit execution.
int main(int argc, char** argv) {
  try {
    qx::log::init_logging();

    std::string problem_path;
    std::string circuit_path;
    std::string options_path;
    std::string out_path;
    std::string log_level{"warn"};
    bool explain = false;
    bool show_version = false;

    CLI::App app{"QuantX CLI"};
    app.add_option("--problem", problem_path, "problem json path");
    app.add_option("--circuit", circuit_path, "circuit json path");
    app.add_option("--options", options_path, "options json path");
    app.add_option("--out", out_path, "output json path");
    app.add_option("--log-level", log_level, "trace|debug|info|warn|error|off")
        ->check(CLI::IsMember({"trace", "debug", "info", "warn", "error", "off"}, CLI::ignore_case));
    app.add_flag("--explain", explain, "explain backend selection (problem mode only)");
    app.add_flag("--version", show_version, "print engine version and exit");

    try {
      app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
      return app.exit(e);
    }

    qx::log::set_log_level(log_level);

    if (show_version) {
      std::cout << qx::version_string() << "\n";
      return 0;
    }

    if (problem_path.empty() && circuit_path.empty()) {
      throw std::runtime_error("missing required --problem or --circuit");
    }
    if (!problem_path.empty() && !circuit_path.empty()) {
      throw std::runtime_error("use either --problem or --circuit, not both");
    }

    const std::string options_json = options_path.empty() ? "{}" : read_text_file(options_path);
    const qx::RunOptions options = qx::options_from_json(options_json);
    qx::log::debug("options loaded");

    std::string output_json;
    if (!problem_path.empty()) {
      const std::string problem_json = read_text_file(problem_path);
      const qx::QuantumProblem problem = qx::problem_from_json(problem_json);
      qx::log::info("loaded problem file: " + problem_path);
      if (explain) {
        output_json = qx::backend_info_to_json(qx::explain_backend_choice(problem, options));
      } else {
        output_json = qx::result_to_json(qx::run(problem, options));
      }
    } else {
      if (explain) {
        throw std::runtime_error("--explain currently supports --problem mode only");
      }
      const std::string circuit_json = read_text_file(circuit_path);
      const qx::QuantumCircuit circuit = qx::circuit_from_json(circuit_json);
      qx::log::info("loaded circuit file: " + circuit_path);
      output_json = qx::circuit_result_to_json(qx::run_circuit(circuit, options));
    }

    if (out_path.empty()) {
      std::cout << output_json << "\n";
    } else {
      write_text_file(out_path, output_json);
      std::cout << "wrote: " << out_path << "\n";
    }
    return 0;
  } catch (const std::exception& ex) {
    qx::log::error(std::string("qx_cli error: ") + ex.what());
    std::cerr << "qx_cli error: " << ex.what() << "\n";
    return 1;
  }
}
