#include <fstream>
#include <cstddef>
#include <cstdint>
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
    bool validate_only = false;
    bool export_continuous_frames = false;
    bool export_wigner = false;
    std::uint32_t export_frame_stride = 1U;
    std::size_t export_points_max = 2500U;
    double export_density_floor_ratio = 0.08;

    CLI::App app{"QuantX CLI"};
    app.add_option("--problem", problem_path, "problem json path");
    app.add_option("--circuit", circuit_path, "circuit json path");
    app.add_option("--options", options_path, "options json path");
    app.add_option("--out", out_path, "output json path");
    app.add_option("--log-level", log_level, "trace|debug|info|warn|error|off")
        ->check(CLI::IsMember({"trace", "debug", "info", "warn", "error", "off"}, CLI::ignore_case));
    app.add_flag("--explain", explain, "explain backend selection (problem mode only)");
    app.add_flag("--version", show_version, "print engine version and exit");
    app.add_flag("--validate-only", validate_only, "validate input json and exit without execution");
    app.add_flag("--export-continuous-frames", export_continuous_frames,
                 "include native continuous-frame payloads in continuous results");
    app.add_flag("--export-wigner", export_wigner,
                 "include Wigner quasi-probability payloads when continuous-frame exports are enabled");
    app.add_option("--export-frame-stride", export_frame_stride,
                   "sample continuous export frames every N steps (minimum 1)");
    app.add_option("--export-points-max", export_points_max,
                   "maximum sparse points emitted by 3D continuous export payloads");
    app.add_option("--export-density-floor-ratio", export_density_floor_ratio,
                   "minimum density ratio used when sampling sparse 3D export points");

    try {
      app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
      return app.exit(e);
    }

    if (export_frame_stride == 0U) {
      throw std::runtime_error("--export-frame-stride must be at least 1");
    }
    if (export_density_floor_ratio < 0.0 || export_density_floor_ratio > 1.0) {
      throw std::runtime_error("--export-density-floor-ratio must be between 0 and 1");
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
    qx::RunOptions options = qx::options_from_json(options_json);
    if (export_continuous_frames) {
      options.export_continuous_frames = true;
    }
    if (export_wigner) {
      options.export_wigner = true;
    }
    options.export_frame_stride = export_frame_stride;
    options.export_points_max = export_points_max;
    options.export_density_floor_ratio = export_density_floor_ratio;
    qx::log::debug("options loaded");

    std::string output_json;
    if (!problem_path.empty()) {
      const std::string problem_json = read_text_file(problem_path);
      const qx::QuantumProblem problem = qx::problem_from_json(problem_json);
      qx::log::info("loaded problem file: " + problem_path);
      if (validate_only) {
        output_json = std::string("{\"ok\":true,\"validated\":\"problem\"}");
      } else if (explain) {
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
      if (validate_only) {
        output_json = std::string("{\"ok\":true,\"validated\":\"circuit\"}");
      } else {
        output_json = qx::circuit_result_to_json(qx::run_circuit(circuit, options));
      }
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
