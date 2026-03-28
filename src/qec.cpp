// M22: Quantum Error Correction — bit_flip_3, phase_flip_3, shor_9
//
// Classical Monte Carlo simulation of three QEC codes:
//   bit_flip_3   — 3-qubit repetition code for X errors (syndromes: Z0Z1, Z1Z2)
//   phase_flip_3 — 3-qubit repetition code for Z errors (syndromes: X0X1, X1X2)
//   shor_9       — 9-qubit Shor code for arbitrary single-qubit Pauli errors
//                  (6 Z-type + 2 X-type stabilizer syndromes)
//
// Each trial:
//   1. Sample independent Pauli errors at physical_error_rate.
//   2. Compute syndrome analytically from the error pattern.
//   3. Apply classical minimum-weight correction via lookup table.
//   4. Check whether the residual error is a logical error.
//
// Threshold curve: sweep over sweep_error_rates and report (p_phys, p_log_mc, p_log_theory).
// Theoretical formulas:
//   Repetition-3: p_log = 3p²(1-p) + p³
//   Shor-9 (approx): two-level repetition → p_log ≈ theory_rep3(theory_rep3(p))

#include "qx/engine.hpp"

#include <chrono>
#include <cmath>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qx {
namespace {

// ── Theoretical formulas ─────────────────────────────────────────────────────

double theory_rep3(double p) {
  // Uncorrectable: 2 or 3 qubits have the relevant error.
  return 3.0 * p * p * (1.0 - p) + p * p * p;
}

double theory_for(const std::string& code, const std::string& error_type, double p) {
  if (code == "shor_9") {
    const double p_x = (error_type == "depolarizing") ? (2.0 * p / 3.0)
                     : (error_type == "X"             ? p : 0.0);
    const double p_z = (error_type == "depolarizing") ? (2.0 * p / 3.0)
                     : (error_type == "Z"             ? p : 0.0);
    const double p_bf = (p_x > 0.0) ? 1.0 - std::pow(1.0 - theory_rep3(p_x), 3.0) : 0.0;
    double p_pf = 0.0;
    if (p_z > 0.0) {
      const double p_z_odd = 3.0 * p_z * (1.0 - p_z) * (1.0 - p_z) + p_z * p_z * p_z;
      p_pf = theory_rep3(p_z_odd);
    }
    return p_bf + p_pf - p_bf * p_pf;
  }
  return theory_rep3(p);
}

// ── 3-qubit bit-flip decoder ──────────────────────────────────────────────────
// Syndromes measure Z0Z1 (s0) and Z1Z2 (s1) parity.
// Syndrome → correction qubit index (-1 = no correction).
int decode_bit_flip_3(bool e[3]) {
  const int s0 = static_cast<int>(e[0]) ^ static_cast<int>(e[1]);
  const int s1 = static_cast<int>(e[1]) ^ static_cast<int>(e[2]);
  if (s0 == 1 && s1 == 0) return 0;
  if (s0 == 1 && s1 == 1) return 1;
  if (s0 == 0 && s1 == 1) return 2;
  return -1;  // no error detected
}

// Returns: (syndrome_string, correction_string, succeeded)
struct TrialResult3 {
  int s0, s1;
  int correction;
  bool succeeded;
};

TrialResult3 run_trial_bit_flip(bool errX[3]) {
  const int s0 = static_cast<int>(errX[0]) ^ static_cast<int>(errX[1]);
  const int s1 = static_cast<int>(errX[1]) ^ static_cast<int>(errX[2]);
  int c = -1;
  if (s0 == 1 && s1 == 0) c = 0;
  else if (s0 == 1 && s1 == 1) c = 1;
  else if (s0 == 0 && s1 == 1) c = 2;
  bool res[3] = {errX[0], errX[1], errX[2]};
  if (c >= 0) res[c] = !res[c];
  const int cnt = static_cast<int>(res[0]) + static_cast<int>(res[1]) + static_cast<int>(res[2]);
  return {s0, s1, c, cnt < 2};
}

TrialResult3 run_trial_phase_flip(bool errZ[3]) {
  // Phase-flip code in Hadamard basis — same syndrome table for Z errors.
  return run_trial_bit_flip(errZ);
}

// ── 9-qubit Shor decoder ──────────────────────────────────────────────────────
// Stabilizers:
//   Z-type (bit-flip): Z0Z1, Z1Z2, Z3Z4, Z4Z5, Z6Z7, Z7Z8  (detect X errors per group)
//   X-type (phase-flip): X0X1X2X3X4X5, X3X4X5X6X7X8        (detect Z errors across groups)
//
// Phase syndrome calculation:
//   ps0 = (Z parity of group 0) XOR (Z parity of group 1)   [X0..X5 stabilizer]
//   ps1 = (Z parity of group 1) XOR (Z parity of group 2)   [X3..X8 stabilizer]
//
// Logical error:
//   X logical: any group has >=2 uncorrected X (majority-vote failure within group)
//   Z logical: >=2 groups have odd Z parity after phase correction

struct TrialResult9 {
  int syndrome[8];  // [s00,s01, s10,s11, s20,s21, ps0,ps1]
  bool succeeded;
};

TrialResult9 run_trial_shor(bool errX[9], bool errZ[9]) {
  // ── Bit-flip syndromes (Z-type) ──────────────────────────────────────────
  int bf[6];
  for (int g = 0; g < 3; g++) {
    const int b = 3 * g;
    bf[2 * g]     = static_cast<int>(errX[b])     ^ static_cast<int>(errX[b + 1]);
    bf[2 * g + 1] = static_cast<int>(errX[b + 1]) ^ static_cast<int>(errX[b + 2]);
  }

  // ── Phase-flip syndromes (X-type) ────────────────────────────────────────
  int gzp[3];
  for (int g = 0; g < 3; g++) {
    const int b = 3 * g;
    gzp[g] = (static_cast<int>(errZ[b]) + static_cast<int>(errZ[b + 1]) +
               static_cast<int>(errZ[b + 2])) & 1;
  }
  const int ps0 = gzp[0] ^ gzp[1];
  const int ps1 = gzp[1] ^ gzp[2];

  // ── Collect syndrome ─────────────────────────────────────────────────────
  int syn[8] = {bf[0], bf[1], bf[2], bf[3], bf[4], bf[5], ps0, ps1};

  // ── Apply bit-flip correction per group ──────────────────────────────────
  bool resX[9];
  for (int i = 0; i < 9; i++) resX[i] = errX[i];
  for (int g = 0; g < 3; g++) {
    const int b = 3 * g;
    int c = -1;
    const int s0 = static_cast<int>(errX[b])     ^ static_cast<int>(errX[b + 1]);
    const int s1 = static_cast<int>(errX[b + 1]) ^ static_cast<int>(errX[b + 2]);
    if (s0 == 1 && s1 == 0) c = b;
    else if (s0 == 1 && s1 == 1) c = b + 1;
    else if (s0 == 0 && s1 == 1) c = b + 2;
    if (c >= 0) resX[c] = !resX[c];
  }

  // ── Apply phase-flip correction (Z on first qubit of identified group) ───
  bool resZ[9];
  for (int i = 0; i < 9; i++) resZ[i] = errZ[i];
  {
    int cg = -1;
    if (ps0 == 1 && ps1 == 0) cg = 0;
    else if (ps0 == 1 && ps1 == 1) cg = 1;
    else if (ps0 == 0 && ps1 == 1) cg = 2;
    if (cg >= 0) resZ[3 * cg] = !resZ[3 * cg];
  }

  // ── Logical error check ──────────────────────────────────────────────────
  bool bf_error = false;
  for (int g = 0; g < 3; g++) {
    const int b = 3 * g;
    const int cnt = static_cast<int>(resX[b]) + static_cast<int>(resX[b + 1]) +
                    static_cast<int>(resX[b + 2]);
    if (cnt >= 2) { bf_error = true; break; }
  }
  int n_parity_groups = 0;
  for (int g = 0; g < 3; g++) {
    const int b = 3 * g;
    const int p3 = (static_cast<int>(resZ[b]) + static_cast<int>(resZ[b + 1]) +
                    static_cast<int>(resZ[b + 2])) & 1;
    n_parity_groups += p3;
  }
  const bool pf_error = (n_parity_groups >= 2);

  TrialResult9 out;
  for (int i = 0; i < 8; i++) out.syndrome[i] = syn[i];
  out.succeeded = (!bf_error && !pf_error);
  return out;
}

// ── Syndrome string helpers ───────────────────────────────────────────────────

std::string syndrome_str3(int s0, int s1) {
  return std::string(1, '0' + s0) + std::string(1, '0' + s1);
}

std::string syndrome_str9(const int syn[8]) {
  std::string s;
  for (int i = 0; i < 8; i++) s += static_cast<char>('0' + syn[i]);
  return s;
}

std::string error_str3X(bool e[3]) {
  std::string s;
  for (int q = 0; q < 3; q++) if (e[q]) s += "X" + std::to_string(q) + " ";
  return s.empty() ? "I" : s.substr(0, s.size() - 1);
}

std::string error_str3Z(bool e[3]) {
  std::string s;
  for (int q = 0; q < 3; q++) if (e[q]) s += "Z" + std::to_string(q) + " ";
  return s.empty() ? "I" : s.substr(0, s.size() - 1);
}

std::string error_str9(bool eX[9], bool eZ[9]) {
  std::string s;
  for (int q = 0; q < 9; q++) {
    if (eX[q] && eZ[q]) s += "Y" + std::to_string(q) + " ";
    else if (eX[q]) s += "X" + std::to_string(q) + " ";
    else if (eZ[q]) s += "Z" + std::to_string(q) + " ";
  }
  return s.empty() ? "I" : s.substr(0, s.size() - 1);
}

// ── Monte Carlo runner ────────────────────────────────────────────────────────

double run_mc(const QecOptions& opts, std::mt19937_64& rng,
              std::uint32_t n_trials_override,
              std::vector<QecFrame>* frames_out) {
  const double p = opts.physical_error_rate;
  const std::uint32_t n = n_trials_override;
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::bernoulli_distribution bernp(p);

  std::uint32_t successes = 0;

  for (std::uint32_t t = 0; t < n; t++) {
    bool errX[9] = {};
    bool errZ[9] = {};
    std::uint32_t n_phys = (opts.code == "shor_9") ? 9u : 3u;

    if (opts.error_type == "X") {
      for (std::uint32_t q = 0; q < n_phys; q++) errX[q] = bernp(rng);
    } else if (opts.error_type == "Z") {
      for (std::uint32_t q = 0; q < n_phys; q++) errZ[q] = bernp(rng);
    } else {
      // Depolarizing: X, Z, or Y each with prob p/3; I with prob 1-p.
      for (std::uint32_t q = 0; q < n_phys; q++) {
        const double r = uni(rng);
        if (r < p / 3.0) errX[q] = true;
        else if (r < 2.0 * p / 3.0) errZ[q] = true;
        else if (r < p) { errX[q] = true; errZ[q] = true; }
      }
    }

    bool ok = false;
    std::string syn_s, err_s, cor_s;

    if (opts.code == "bit_flip_3") {
      auto res = run_trial_bit_flip(errX);
      ok = res.succeeded;
      syn_s = syndrome_str3(res.s0, res.s1);
      err_s = error_str3X(errX);
      cor_s = (res.correction >= 0) ? ("X" + std::to_string(res.correction)) : "I";
    } else if (opts.code == "phase_flip_3") {
      auto res = run_trial_phase_flip(errZ);
      ok = res.succeeded;
      syn_s = syndrome_str3(res.s0, res.s1);
      err_s = error_str3Z(errZ);
      cor_s = (res.correction >= 0) ? ("Z" + std::to_string(res.correction)) : "I";
    } else {
      auto res = run_trial_shor(errX, errZ);
      ok = res.succeeded;
      syn_s = syndrome_str9(res.syndrome);
      err_s = error_str9(errX, errZ);
      cor_s = "decoded";
    }

    if (ok) successes++;

    if (frames_out && t < opts.n_frames_max) {
      QecFrame fr;
      fr.trial = t;
      fr.errors_injected = err_s;
      fr.syndrome = syn_s;
      fr.correction = cor_s;
      fr.succeeded = ok;
      frames_out->push_back(fr);
    }
  }

  return 1.0 - static_cast<double>(successes) / static_cast<double>(n);
}

}  // namespace

// ── Public entry point ────────────────────────────────────────────────────────

QecResult run_qec(const QecOptions& opts) {
  const auto t0 = std::chrono::steady_clock::now();

  QecResult result;
  result.code = opts.code;

  if (opts.code == "bit_flip_3") {
    result.n_physical = 3;
    result.n_syndrome = 2;
  } else if (opts.code == "phase_flip_3") {
    result.n_physical = 3;
    result.n_syndrome = 2;
  } else if (opts.code == "shor_9") {
    result.n_physical = 9;
    result.n_syndrome = 8;
  } else {
    result.ok = false;
    result.message = "unknown QEC code: " + opts.code +
                     " (use bit_flip_3 | phase_flip_3 | shor_9)";
    return result;
  }

  // Validate code/noise compatibility to avoid silently reporting invalid
  // logical-error rates for unsupported code/error-type combinations.
  if (opts.code == "bit_flip_3" && opts.error_type == "Z") {
    result.ok = false;
    result.message = "unsupported QEC configuration: bit_flip_3 cannot decode pure Z noise; "
                     "use error_type X or depolarizing, or switch to phase_flip_3 for Z noise";
    return result;
  }
  if (opts.code == "phase_flip_3" && opts.error_type == "X") {
    result.ok = false;
    result.message = "unsupported QEC configuration: phase_flip_3 cannot decode pure X noise; "
                     "use error_type Z or depolarizing, or switch to bit_flip_3 for X noise";
    return result;
  }

  std::mt19937_64 rng(opts.seed);

  // Main Monte Carlo run.
  result.logical_error_rate = run_mc(opts, rng, opts.n_trials, &result.frames);
  result.logical_error_rate_theory = theory_for(opts.code, opts.error_type, opts.physical_error_rate);

  // Threshold sweep.
  if (!opts.sweep_error_rates.empty()) {
    std::mt19937_64 sweep_rng(opts.seed + 99991ULL);
    for (double p_sweep : opts.sweep_error_rates) {
      QecOptions sweep_opts = opts;
      sweep_opts.physical_error_rate = p_sweep;
      sweep_opts.n_trials = std::max(opts.n_trials / 4u, 500u);
      sweep_opts.n_frames_max = 0;
      const double p_log_mc = run_mc(sweep_opts, sweep_rng, sweep_opts.n_trials, nullptr);
      const double p_log_th = theory_for(opts.code, opts.error_type, p_sweep);
      result.threshold_curve.push_back({p_sweep, p_log_mc, p_log_th});
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return result;
}

}  // namespace qx
