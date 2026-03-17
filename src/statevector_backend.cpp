#include "qx/backends.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <fstream>
#include <limits>
#include <regex>
#include <string>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace qx {
namespace {

using Complex = std::complex<double>;
using Hessenberg = std::vector<std::vector<Complex>>;

// Parsed and precompiled Pauli term for fast Hamiltonian application.
struct ParsedTerm {
  std::vector<char> paulis{};
  std::vector<std::uint32_t> qubits{};
  double coefficient{0.0};
  std::size_t flip_mask{0};
  std::size_t yz_sign_mask{0};
  Complex y_global_phase{1.0, 0.0};
};

// Parse Hamiltonian Pauli strings into bitmask-based representation.
std::vector<ParsedTerm> parse_terms(const QuantumProblem& problem) {
  std::vector<ParsedTerm> out;
  if (problem.hamiltonian_terms.empty()) {
    out.push_back(ParsedTerm{{'Z'}, {0}, 1.0});
    return out;
  }

  std::regex token("([XYZxyz])(\\d+)");
  for (const auto& t : problem.hamiltonian_terms) {
    if (t.label.empty()) {
      continue;
    }
    ParsedTerm parsed;
    parsed.coefficient = t.coefficient;
    std::unordered_set<std::uint32_t> seen_qubits;
    auto begin = std::sregex_iterator(t.label.begin(), t.label.end(), token);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
      const char p = static_cast<char>(std::toupper(static_cast<unsigned char>((*it)[1].str()[0])));
      const std::uint32_t q = static_cast<std::uint32_t>(std::stoul((*it)[2].str()));
      if (seen_qubits.find(q) != seen_qubits.end()) {
        parsed.paulis.clear();
        parsed.qubits.clear();
        break;
      }
      seen_qubits.insert(q);
      parsed.paulis.push_back(p);
      parsed.qubits.push_back(q);
    }
    if (parsed.paulis.empty()) {
      continue;
    }
    std::size_t y_count = 0;
    for (std::size_t i = 0; i < parsed.paulis.size(); ++i) {
      const char p = parsed.paulis[i];
      const std::uint32_t q = parsed.qubits[i];
      if (q >= 63) {
        parsed.paulis.clear();
        parsed.qubits.clear();
        break;
      }
      const std::size_t m = static_cast<std::size_t>(1ULL << q);
      if (p == 'X') {
        parsed.flip_mask |= m;
      } else if (p == 'Y') {
        parsed.flip_mask |= m;
        parsed.yz_sign_mask |= m;
        ++y_count;
      } else if (p == 'Z') {
        parsed.yz_sign_mask |= m;
      } else {
        parsed.paulis.clear();
        parsed.qubits.clear();
        break;
      }
    }
    if (parsed.paulis.empty()) {
      continue;
    }
    switch (y_count % 4U) {
      case 0:
        parsed.y_global_phase = Complex{1.0, 0.0};
        break;
      case 1:
        parsed.y_global_phase = Complex{0.0, 1.0};
        break;
      case 2:
        parsed.y_global_phase = Complex{-1.0, 0.0};
        break;
      default:
        parsed.y_global_phase = Complex{0.0, -1.0};
        break;
    }
    out.push_back(parsed);
  }
  if (out.empty()) {
    out.push_back(ParsedTerm{{'Z'}, {0}, 1.0});
  }
  return out;
}

// Apply one parsed Pauli term to psi and accumulate into out.
void apply_pauli_term(const std::vector<Complex>& psi, std::vector<Complex>& out, const ParsedTerm& term) {
  if (psi.empty()) {
    return;
  }
  const std::size_t n = psi.size();
  if (term.paulis.empty()) {
    return;
  }

  for (std::size_t src = 0; src < n; ++src) {
    const std::size_t dst = src ^ term.flip_mask;
    if (dst >= n) continue;
    const std::size_t sign_bits = src & term.yz_sign_mask;
    const bool odd = (std::popcount(sign_bits) & 1U) != 0U;
    const double sign = odd ? -1.0 : 1.0;
    const Complex phase = term.y_global_phase * sign;
    out[dst] += term.coefficient * phase * psi[src];
  }
}

// Apply full Hamiltonian H to vector psi and write to out.
void apply_hamiltonian(
    const std::vector<Complex>& psi, std::vector<Complex>& out, const std::vector<ParsedTerm>& terms) {
  std::fill(out.begin(), out.end(), Complex{0.0, 0.0});
  for (const auto& term : terms) {
    apply_pauli_term(psi, out, term);
  }
}

// Compute squared L2 norm.
double norm_sq(const std::vector<Complex>& v) {
  double n = 0.0;
  for (const auto& x : v) {
    n += std::norm(x);
  }
  return n;
}

// Normalize vector in-place.
void normalize(std::vector<Complex>& v) {
  const double n = std::sqrt(std::max(0.0, norm_sq(v)));
  if (n <= 0.0) {
    return;
  }
  for (auto& x : v) {
    x /= n;
  }
}

// Conjugate inner product <a|b>.
Complex dot(const std::vector<Complex>& a, const std::vector<Complex>& b) {
  Complex d{0.0, 0.0};
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    d += std::conj(a[i]) * b[i];
  }
  return d;
}

// BLAS-like y += alpha * x.
void axpy(std::vector<Complex>& y, Complex alpha, const std::vector<Complex>& x) {
  const std::size_t n = std::min(y.size(), x.size());
  for (std::size_t i = 0; i < n; ++i) {
    y[i] += alpha * x[i];
  }
}

// Convert Hessenberg matrix to Eigen matrix for robust matrix functions.
Eigen::MatrixXcd to_eigen_matrix(const Hessenberg& h, std::size_t k) {
  Eigen::MatrixXcd out = Eigen::MatrixXcd::Zero(static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(k));
  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < k; ++j) {
      out(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = h[i][j];
    }
  }
  return out;
}

// Build Arnoldi basis and Hessenberg matrix for Krylov time evolution.
std::size_t arnoldi(
    const std::vector<Complex>& v0,
    std::size_t m,
    const std::vector<ParsedTerm>& terms,
    std::vector<std::vector<Complex>>& V,
    Hessenberg& H) {
  const std::size_t n = v0.size();
  V.clear();
  V.reserve(m + 1);
  V.push_back(v0);
  H.assign(m + 1, std::vector<Complex>(m, Complex{0.0, 0.0}));

  std::vector<Complex> w(n, Complex{0.0, 0.0});
  std::size_t used = 1;
  for (std::size_t j = 0; j < m; ++j) {
    apply_hamiltonian(V[j], w, terms);
    for (std::size_t i = 0; i <= j; ++i) {
      H[i][j] = dot(V[i], w);
      axpy(w, -H[i][j], V[i]);
    }
    const double beta = std::sqrt(std::max(0.0, norm_sq(w)));
    H[j + 1][j] = Complex{beta, 0.0};
    used = j + 1;
    if (beta < 1e-13) {
      break;
    }
    for (auto& x : w) {
      x /= beta;
    }
    V.push_back(w);
    used = j + 2;
  }
  return std::min<std::size_t>(m, used);
}

// Dense statevector backend with trotter and Krylov propagators.
class StatevectorBackend final : public IBackend {
 public:
  // Return backend name.
  std::string name() const override { return "statevector"; }

  // Return static statevector backend capability metadata.
  BackendCapabilities capabilities() const override {
    BackendCapabilities c;
    c.kind = BackendKind::Statevector;
    c.supports_gpu = true;
    c.supports_mpi = true;
    c.supports_noisy = false;
    c.max_qubits_recommended = 34;
    c.equation_scope = "Schrodinger";
    return c;
  }

  // Initialize dense state and parsed Hamiltonian terms.
  void initialize(const QuantumProblem& problem, const RunOptions&) override {
    std::size_t n = static_cast<std::size_t>(1ULL << problem.qubits);
    if (problem.qubits >= 26) {
      // Limit memory footprint for scaffold-mode runs.
      n = static_cast<std::size_t>(1ULL << 20);
    }
    state_.assign(n, {0.0, 0.0});
    state_[0] = {1.0, 0.0};
    terms_ = parse_terms(problem);
  }

  // Evolve state by selected method ("krylov" or trotter-like product formula).
  void evolve(const QuantumProblem& problem, const RunOptions&) override {
    const std::size_t steps = static_cast<std::size_t>(problem.evolution.total_time / problem.evolution.dt);
    if (steps == 0 || state_.empty()) {
      return;
    }
    std::vector<Complex> tmp(state_.size(), Complex{0.0, 0.0});
    std::vector<Complex> hpsi(state_.size(), Complex{0.0, 0.0});
    const std::string method = problem.evolution.method;

    for (std::size_t s = 0; s < steps; ++s) {
      if (method == "krylov") {
        std::vector<Complex> v0 = state_;
        const double psi_norm = std::sqrt(std::max(0.0, norm_sq(v0)));
        if (psi_norm < 1e-15) {
          continue;
        }
        for (auto& x : v0) {
          x /= psi_norm;
        }

        const std::size_t max_dim =
            std::max<std::size_t>(1, std::min<std::size_t>(problem.evolution.krylov_dim, static_cast<std::size_t>(32)));
        std::vector<std::vector<Complex>> V;
        Hessenberg Hfull;
        const std::size_t k = arnoldi(v0, max_dim, terms_, V, Hfull);
        if (k == 0 || V.empty()) {
          continue;
        }

        // Reuse Eigen's tested matrix exponential instead of maintaining custom approximants.
        const Eigen::MatrixXcd Hk = to_eigen_matrix(Hfull, k);
        const Eigen::MatrixXcd A = Complex{0.0, -problem.evolution.dt} * Hk;
        const Eigen::MatrixXcd expA = A.exp();

        Eigen::VectorXcd e1 = Eigen::VectorXcd::Zero(static_cast<Eigen::Index>(k));
        e1(0) = Complex{1.0, 0.0};
        const Eigen::VectorXcd y = expA * e1;

        std::fill(tmp.begin(), tmp.end(), Complex{0.0, 0.0});
        for (std::size_t j = 0; j < k; ++j) {
          axpy(tmp, y(static_cast<Eigen::Index>(j)), V[j]);
        }
        for (auto& x : tmp) {
          x *= psi_norm;
        }
        state_.swap(tmp);
        normalize(state_);
      } else {
        for (const auto& term : terms_) {
          std::fill(hpsi.begin(), hpsi.end(), Complex{0.0, 0.0});
          apply_pauli_term(state_, hpsi, term);
          const double theta = problem.evolution.dt * term.coefficient;
          const Complex c = Complex{std::cos(theta), 0.0};
          const Complex s = Complex{0.0, -std::sin(theta)};
          for (std::size_t i = 0; i < state_.size(); ++i) {
            tmp[i] = c * state_[i] + s * hpsi[i];
          }
          state_.swap(tmp);
        }
      }
    }
  }

  // Return basic normalization/population observables.
  std::map<std::string, double> measure(const QuantumProblem&) override {
    double norm = 0.0;
    for (const auto& a : state_) {
      norm += std::norm(a);
    }
    const double p0 = std::norm(state_.empty() ? Complex(0.0, 0.0) : state_[0]);
    const double p1 = state_.size() > 1 ? std::norm(state_[1]) : 0.0;
    const double plast = state_.empty() ? 0.0 : std::norm(state_.back());
    return {{"norm", norm}, {"population_0", p0}, {"population_1", p1}, {"population_last", plast}};
  }

  // Persist dense state to binary checkpoint file.
  void checkpoint(const std::string& path) override {
    std::ofstream os(path + "/statevector.chk", std::ios::binary);
    const std::size_t size = state_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size > 0) {
      os.write(reinterpret_cast<const char*>(state_.data()),
               static_cast<std::streamsize>(size * sizeof(std::complex<double>)));
    }
  }

  // Restore dense state from binary checkpoint file if present.
  void restore(const std::string& path) override {
    std::ifstream is(path + "/statevector.chk", std::ios::binary);
    if (!is.good()) {
      return;
    }
    std::size_t size = 0;
    is.read(reinterpret_cast<char*>(&size), sizeof(size));
    state_.resize(size);
    if (size > 0) {
      is.read(reinterpret_cast<char*>(state_.data()),
              static_cast<std::streamsize>(size * sizeof(std::complex<double>)));
    }
  }

  // Expose current dense statevector.
  std::vector<std::complex<double>> current_state() const override { return state_; }

 private:
  std::vector<Complex> state_;
  std::vector<ParsedTerm> terms_;
};

}  // namespace

// Construct statevector backend instance.
std::unique_ptr<IBackend> make_statevector_backend() {
  return std::make_unique<StatevectorBackend>();
}

}  // namespace qx
