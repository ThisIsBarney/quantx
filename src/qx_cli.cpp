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
    std::string vqe_path;
    std::string qpe_path;
    std::string qec_path;
    std::string qfield_path;
    std::string phi4_path;
    std::string hubbard_path;
    std::string floquet_path;
    std::string heisenberg_path;
    std::string bose_hubbard_path;
    std::string tfi_path;
    std::string ssh_path;
    std::string kitaev_path;
    std::string anderson_path;
    std::string chern_path;
    std::string hofstadter_path;
    std::string mbl_path;
    std::string spectral_path;
    std::string tenfold_path;
    std::string second_chern_path;
    std::string options_path;
    std::string out_path;
    std::string log_level{"warn"};
    bool explain = false;
    bool show_version = false;
    bool validate_only = false;
    bool export_continuous_frames = false;
    bool export_wigner = false;
    std::uint32_t entanglement_cut = 0U;
    std::uint32_t export_frame_stride = 1U;
    std::size_t export_points_max = 2500U;
    double export_density_floor_ratio = 0.08;

    CLI::App app{"QuantX CLI"};
    app.add_option("--problem", problem_path, "problem json path");
    app.add_option("--circuit", circuit_path, "circuit json path");
    app.add_option("--vqe", vqe_path, "VQE/QAOA options json path (run variational optimization)");
    app.add_option("--qpe", qpe_path, "QPE options json path (run Quantum Phase Estimation)");
    app.add_option("--qec", qec_path, "QEC options json path (run Quantum Error Correction Monte Carlo)");
    app.add_option("--qfield", qfield_path, "QField options json path (run Schwinger model / lattice QFT simulation)");
    app.add_option("--phi4", phi4_path, "phi4 options json path (run phi^4 scalar QFT simulation)");
    app.add_option("--hubbard", hubbard_path, "hubbard options json path (run 1D Hubbard model simulation)");
    app.add_option("--floquet", floquet_path, "floquet options json path (run Floquet kicked-Ising simulation)");
    app.add_option("--heisenberg", heisenberg_path, "heisenberg options json path (run 1D XXZ Heisenberg spin chain simulation)");
    app.add_option("--bose-hubbard", bose_hubbard_path, "bose-hubbard options json path (run Bose-Hubbard model simulation)");
    app.add_option("--tfi", tfi_path, "tfi options json path (run transverse-field Ising model simulation)");
    app.add_option("--ssh", ssh_path, "ssh options json path (run SSH topological insulator simulation)");
    app.add_option("--kitaev", kitaev_path, "kitaev options json path (run Kitaev chain p-wave superconductor simulation)");
    app.add_option("--anderson", anderson_path, "anderson options json path (run Anderson localization simulation)");
    app.add_option("--chern", chern_path, "chern options json path (run 2D Chern insulator / QWZ model simulation)");
    app.add_option("--hofstadter", hofstadter_path, "hofstadter options json path (run Hofstadter butterfly simulation)");
    app.add_option("--mbl", mbl_path, "mbl options json path (run Many-Body Localization disordered XXZ chain simulation)");
    app.add_option("--spectral", spectral_path, "spectral options json path (run dynamical structure factor S(k,omega) simulation)");
    app.add_option("--tenfold", tenfold_path, "tenfold options json path (run 10-fold way Altland-Zirnbauer topological classification)");
    app.add_option("--second-chern", second_chern_path, "second-chern options json path (run second Chern number / 4D Quantum Hall Effect)");
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
    app.add_option("--entanglement-cut", entanglement_cut,
                   "bipartite cut qubit index n_a: subsystem A=[0,n_a), B=[n_a,n); 0 disables");
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

    if (problem_path.empty() && circuit_path.empty() && vqe_path.empty() &&
        qpe_path.empty() && qec_path.empty() && qfield_path.empty() &&
        phi4_path.empty() && hubbard_path.empty() && floquet_path.empty() &&
        heisenberg_path.empty() && bose_hubbard_path.empty() &&
        tfi_path.empty() && ssh_path.empty() &&
        kitaev_path.empty() && anderson_path.empty() &&
        chern_path.empty() && hofstadter_path.empty() && mbl_path.empty() &&
        spectral_path.empty() && tenfold_path.empty() && second_chern_path.empty()) {
      throw std::runtime_error(
          "missing required --problem, --circuit, --vqe, --qpe, --qec, --qfield, --phi4, "
          "--hubbard, --floquet, --heisenberg, --bose-hubbard, --tfi, --ssh, --kitaev, "
          "--anderson, --chern, --hofstadter, --mbl, --spectral, --tenfold, or --second-chern");
    }
    {
      int modes = (!problem_path.empty()      ? 1 : 0) +
                  (!circuit_path.empty()      ? 1 : 0) +
                  (!vqe_path.empty()          ? 1 : 0) +
                  (!qpe_path.empty()          ? 1 : 0) +
                  (!qec_path.empty()          ? 1 : 0) +
                  (!qfield_path.empty()       ? 1 : 0) +
                  (!phi4_path.empty()         ? 1 : 0) +
                  (!hubbard_path.empty()      ? 1 : 0) +
                  (!floquet_path.empty()      ? 1 : 0) +
                  (!heisenberg_path.empty()   ? 1 : 0) +
                  (!bose_hubbard_path.empty() ? 1 : 0) +
                  (!tfi_path.empty()          ? 1 : 0) +
                  (!ssh_path.empty()          ? 1 : 0) +
                  (!kitaev_path.empty()       ? 1 : 0) +
                  (!anderson_path.empty()     ? 1 : 0) +
                  (!chern_path.empty()        ? 1 : 0) +
                  (!hofstadter_path.empty()   ? 1 : 0) +
                  (!mbl_path.empty()          ? 1 : 0) +
                  (!spectral_path.empty()     ? 1 : 0) +
                  (!tenfold_path.empty()      ? 1 : 0) +
                  (!second_chern_path.empty() ? 1 : 0);
      if (modes > 1) {
        throw std::runtime_error(
            "use exactly one of --problem, --circuit, --vqe, --qpe, --qec, --qfield, --phi4, "
            "--hubbard, --floquet, --heisenberg, --bose-hubbard, --tfi, --ssh, --kitaev, "
            "--anderson, --chern, --hofstadter, --mbl, --spectral, --tenfold, or --second-chern");
      }
    }

    const std::string options_json = options_path.empty() ? "{}" : read_text_file(options_path);
    qx::RunOptions options = qx::options_from_json(options_json);
    if (export_continuous_frames) {
      options.export_continuous_frames = true;
    }
    if (export_wigner) {
      options.export_wigner = true;
    }
    if (entanglement_cut > 0U) {
      options.entanglement_cut = entanglement_cut;
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
    } else if (!circuit_path.empty()) {
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
    } else if (!vqe_path.empty()) {
      // VQE mode.
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --vqe mode");
      }
      const std::string vqe_json = read_text_file(vqe_path);
      const qx::VqeOptions vqe_opts = qx::vqe_options_from_json(vqe_json);
      qx::log::info("loaded vqe options file: " + vqe_path);
      output_json = qx::vqe_result_to_json(qx::run_vqe(vqe_opts));
    } else if (!qpe_path.empty()) {
      // QPE mode.
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --qpe mode");
      }
      const std::string qpe_json = read_text_file(qpe_path);
      const qx::QpeOptions qpe_opts = qx::qpe_options_from_json(qpe_json);
      qx::log::info("loaded qpe options file: " + qpe_path);
      output_json = qx::qpe_result_to_json(qx::run_qpe(qpe_opts));
    } else if (!qfield_path.empty()) {
      // QField (Schwinger model) mode.
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --qfield mode");
      }
      const std::string qfield_json = read_text_file(qfield_path);
      const qx::QfieldOptions qfield_opts = qx::qfield_options_from_json(qfield_json);
      qx::log::info("loaded qfield options file: " + qfield_path);
      output_json = qx::qfield_result_to_json(qx::run_qfield(qfield_opts));
    } else if (!phi4_path.empty()) {
      // φ⁴ scalar QFT mode.
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --phi4 mode");
      }
      const std::string phi4_json = read_text_file(phi4_path);
      const qx::Phi4Options phi4_opts = qx::phi4_options_from_json(phi4_json);
      qx::log::info("loaded phi4 options file: " + phi4_path);
      output_json = qx::phi4_result_to_json(qx::run_phi4(phi4_opts));
    } else if (!hubbard_path.empty()) {
      // 1D Hubbard model mode.
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --hubbard mode");
      }
      const std::string hubbard_json = read_text_file(hubbard_path);
      const qx::HubbardOptions hubbard_opts = qx::hubbard_options_from_json(hubbard_json);
      qx::log::info("loaded hubbard options file: " + hubbard_path);
      output_json = qx::hubbard_result_to_json(qx::run_hubbard(hubbard_opts));
    } else if (!floquet_path.empty()) {
      // Floquet kicked-Ising mode.
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --floquet mode");
      }
      const std::string floquet_json = read_text_file(floquet_path);
      const qx::FloquetOptions floquet_opts = qx::floquet_options_from_json(floquet_json);
      qx::log::info("loaded floquet options file: " + floquet_path);
      output_json = qx::floquet_result_to_json(qx::run_floquet(floquet_opts));
    } else if (!heisenberg_path.empty()) {
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --heisenberg mode");
      }
      const std::string heisenberg_json = read_text_file(heisenberg_path);
      const qx::HeisenbergOptions heisenberg_opts = qx::heisenberg_options_from_json(heisenberg_json);
      qx::log::info("loaded heisenberg options file: " + heisenberg_path);
      output_json = qx::heisenberg_result_to_json(qx::run_heisenberg(heisenberg_opts));
    } else if (!bose_hubbard_path.empty()) {
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --bose-hubbard mode");
      }
      const std::string bose_hubbard_json = read_text_file(bose_hubbard_path);
      const qx::BoseHubbardOptions bose_hubbard_opts = qx::bose_hubbard_options_from_json(bose_hubbard_json);
      qx::log::info("loaded bose-hubbard options file: " + bose_hubbard_path);
      output_json = qx::bose_hubbard_result_to_json(qx::run_bose_hubbard(bose_hubbard_opts));
    } else if (!tfi_path.empty()) {
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --tfi mode");
      }
      const std::string tfi_json = read_text_file(tfi_path);
      const qx::TFIOptions tfi_opts = qx::tfi_options_from_json(tfi_json);
      qx::log::info("loaded tfi options file: " + tfi_path);
      output_json = qx::tfi_result_to_json(qx::run_tfi(tfi_opts));
    } else if (!ssh_path.empty()) {
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --ssh mode");
      }
      const std::string ssh_json = read_text_file(ssh_path);
      const qx::SSHOptions ssh_opts = qx::ssh_options_from_json(ssh_json);
      qx::log::info("loaded ssh options file: " + ssh_path);
      output_json = qx::ssh_result_to_json(qx::run_ssh(ssh_opts));
    } else if (!kitaev_path.empty()) {
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --kitaev mode");
      }
      const std::string kitaev_json = read_text_file(kitaev_path);
      const qx::KitaevOptions kitaev_opts = qx::kitaev_options_from_json(kitaev_json);
      qx::log::info("loaded kitaev options file: " + kitaev_path);
      output_json = qx::kitaev_result_to_json(qx::run_kitaev(kitaev_opts));
    } else if (!anderson_path.empty()) {
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --anderson mode");
      }
      const std::string anderson_json = read_text_file(anderson_path);
      const qx::AndersonOptions anderson_opts = qx::anderson_options_from_json(anderson_json);
      qx::log::info("loaded anderson options file: " + anderson_path);
      output_json = qx::anderson_result_to_json(qx::run_anderson(anderson_opts));
    } else if (!chern_path.empty()) {
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --chern mode");
      }
      const std::string chern_json = read_text_file(chern_path);
      const qx::ChernOptions chern_opts = qx::chern_options_from_json(chern_json);
      qx::log::info("loaded chern options file: " + chern_path);
      output_json = qx::chern_result_to_json(qx::run_chern(chern_opts));
    } else if (!hofstadter_path.empty()) {
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --hofstadter mode");
      }
      const std::string hof_json = read_text_file(hofstadter_path);
      const qx::HofstadterOptions hof_opts = qx::hofstadter_options_from_json(hof_json);
      qx::log::info("loaded hofstadter options file: " + hofstadter_path);
      output_json = qx::hofstadter_result_to_json(qx::run_hofstadter(hof_opts));
    } else if (!mbl_path.empty()) {
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --mbl mode");
      }
      const std::string mbl_json = read_text_file(mbl_path);
      const qx::MBLOptions mbl_opts = qx::mbl_options_from_json(mbl_json);
      qx::log::info("loaded mbl options file: " + mbl_path);
      output_json = qx::mbl_result_to_json(qx::run_mbl(mbl_opts));
    } else if (!spectral_path.empty()) {
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --spectral mode");
      }
      const std::string spectral_json = read_text_file(spectral_path);
      const qx::SpectralOptions spectral_opts = qx::spectral_options_from_json(spectral_json);
      qx::log::info("loaded spectral options file: " + spectral_path);
      output_json = qx::spectral_result_to_json(qx::run_spectral(spectral_opts));
    } else if (!tenfold_path.empty()) {
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --tenfold mode");
      }
      const std::string tenfold_json = read_text_file(tenfold_path);
      const qx::TenfoldOptions tenfold_opts = qx::tenfold_options_from_json(tenfold_json);
      qx::log::info("loaded tenfold options file: " + tenfold_path);
      output_json = qx::tenfold_result_to_json(qx::run_tenfold(tenfold_opts));
    } else if (!second_chern_path.empty()) {
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --second-chern mode");
      }
      const std::string sc_json = read_text_file(second_chern_path);
      const qx::SecondChernOptions sc_opts = qx::second_chern_options_from_json(sc_json);
      qx::log::info("loaded second-chern options file: " + second_chern_path);
      output_json = qx::second_chern_result_to_json(qx::run_second_chern(sc_opts));
    } else {
      // QEC mode.
      if (explain || validate_only) {
        throw std::runtime_error("--explain and --validate-only are not supported in --qec mode");
      }
      const std::string qec_json = read_text_file(qec_path);
      const qx::QecOptions qec_opts = qx::qec_options_from_json(qec_json);
      qx::log::info("loaded qec options file: " + qec_path);
      output_json = qx::qec_result_to_json(qx::run_qec(qec_opts));
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
