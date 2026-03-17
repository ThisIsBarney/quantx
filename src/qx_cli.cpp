#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "qx/engine.hpp"
#include "qx/json_utils.hpp"

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

// Print CLI usage help.
void print_usage() {
  std::cout
      << "Usage:\n"
      << "  qx_cli --problem <problem.json> [--options <options.json>] [--out <result.json>]\n"
      << "  qx_cli --circuit <circuit.json> [--options <options.json>] [--out <result.json>]\n"
      << "  qx_cli --problem <problem.json> [--options <options.json>] --explain [--out <explain.json>]\n";
}

}  // namespace

// CLI entrypoint for problem/circuit execution.
int main(int argc, char** argv) {
  try {
    std::string problem_path;
    std::string circuit_path;
    std::string options_path;
    std::string out_path;
    bool explain = false;

    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--problem" && i + 1 < argc) {
        problem_path = argv[++i];
      } else if (arg == "--circuit" && i + 1 < argc) {
        circuit_path = argv[++i];
      } else if (arg == "--options" && i + 1 < argc) {
        options_path = argv[++i];
      } else if (arg == "--out" && i + 1 < argc) {
        out_path = argv[++i];
      } else if (arg == "--explain") {
        explain = true;
      } else if (arg == "--help" || arg == "-h") {
        print_usage();
        return 0;
      } else {
        std::cerr << "Unknown arg: " << arg << "\n";
        print_usage();
        return 2;
      }
    }

    if (problem_path.empty() && circuit_path.empty()) {
      std::cerr << "Missing required --problem or --circuit\n";
      print_usage();
      return 2;
    }
    if (!problem_path.empty() && !circuit_path.empty()) {
      std::cerr << "Use either --problem or --circuit, not both\n";
      print_usage();
      return 2;
    }

    const std::string options_json = options_path.empty() ? "{}" : read_text_file(options_path);
    const qx::RunOptions options = qx::options_from_json(options_json);

    std::string output_json;
    if (!problem_path.empty()) {
      const std::string problem_json = read_text_file(problem_path);
      const qx::QuantumProblem problem = qx::problem_from_json(problem_json);
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
    std::cerr << "qx_cli error: " << ex.what() << "\n";
    return 1;
  }
}
