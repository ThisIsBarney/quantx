#include "qx/backends.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

namespace qx {
namespace {

using Complex = std::complex<double>;

struct ParsedTerm {
  std::vector<char> paulis{};
  std::vector<std::uint32_t> qubits{};
  double coefficient{0.0};
};

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
    auto begin = std::sregex_iterator(t.label.begin(), t.label.end(), token);
    auto end = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
      const char p = static_cast<char>(std::toupper(static_cast<unsigned char>((*it)[1].str()[0])));
      const std::uint32_t q = static_cast<std::uint32_t>(std::stoul((*it)[2].str()));
      parsed.paulis.push_back(p);
      parsed.qubits.push_back(q);
    }
    if (parsed.paulis.empty()) {
      continue;
    }
    out.push_back(parsed);
  }
  if (out.empty()) {
    out.push_back(ParsedTerm{{'Z'}, {0}, 1.0});
  }
  return out;
}

void apply_pauli_term(const std::vector<Complex>& psi, std::vector<Complex>& out, const ParsedTerm& term) {
  if (psi.empty()) {
    return;
  }
  const std::size_t n = psi.size();
  const std::size_t k = term.paulis.size();
  if (k == 0 || k != term.qubits.size()) {
    return;
  }

  for (std::size_t src = 0; src < n; ++src) {
    std::size_t dst = src;
    Complex phase{1.0, 0.0};
    bool valid = true;

    for (std::size_t idx = 0; idx < k; ++idx) {
      const std::uint32_t q = term.qubits[idx];
      const std::size_t mask = static_cast<std::size_t>(1ULL << q);
      if (mask >= n) {
        valid = false;
        break;
      }
      const bool bit = (dst & mask) != 0;
      const char p = term.paulis[idx];
      if (p == 'X') {
        dst ^= mask;
      } else if (p == 'Y') {
        phase *= bit ? Complex{0.0, -1.0} : Complex{0.0, 1.0};
        dst ^= mask;
      } else if (p == 'Z') {
        phase *= bit ? Complex{-1.0, 0.0} : Complex{1.0, 0.0};
      } else {
        valid = false;
        break;
      }
    }

    if (valid) {
      out[dst] += term.coefficient * phase * psi[src];
    }
  }
}

void apply_hamiltonian(
    const std::vector<Complex>& psi, std::vector<Complex>& out, const std::vector<ParsedTerm>& terms) {
  std::fill(out.begin(), out.end(), Complex{0.0, 0.0});
  for (const auto& term : terms) {
    apply_pauli_term(psi, out, term);
  }
}

double norm_sq(const std::vector<Complex>& v) {
  double n = 0.0;
  for (const auto& x : v) {
    n += std::norm(x);
  }
  return n;
}

void normalize(std::vector<Complex>& v) {
  const double n = std::sqrt(std::max(0.0, norm_sq(v)));
  if (n <= 0.0) {
    return;
  }
  for (auto& x : v) {
    x /= n;
  }
}

Complex dot(const std::vector<Complex>& a, const std::vector<Complex>& b) {
  Complex d{0.0, 0.0};
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i) {
    d += std::conj(a[i]) * b[i];
  }
  return d;
}

void axpy(std::vector<Complex>& y, Complex alpha, const std::vector<Complex>& x) {
  const std::size_t n = std::min(y.size(), x.size());
  for (std::size_t i = 0; i < n; ++i) {
    y[i] += alpha * x[i];
  }
}

class StatevectorBackend final : public IBackend {
 public:
  std::string name() const override { return "statevector"; }

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
        apply_hamiltonian(state_, hpsi, terms_);
        std::vector<Complex> v1 = state_;
        normalize(v1);

        Complex alpha = dot(v1, hpsi);
        std::vector<Complex> w = hpsi;
        axpy(w, -alpha, v1);
        const double beta = std::sqrt(std::max(0.0, norm_sq(w)));

        if (beta < 1e-14) {
          const Complex phase = std::exp(Complex{0.0, -problem.evolution.dt} * alpha);
          for (std::size_t i = 0; i < state_.size(); ++i) {
            state_[i] = phase * v1[i];
          }
          continue;
        }

        for (auto& x : w) {
          x /= beta;
        }
        std::vector<Complex> hw(state_.size(), Complex{0.0, 0.0});
        apply_hamiltonian(w, hw, terms_);
        Complex gamma = dot(w, hw);

        const Complex tr = alpha + gamma;
        const Complex det = alpha * gamma - Complex{beta * beta, 0.0};
        const Complex disc = std::sqrt(tr * tr / Complex{4.0, 0.0} - det);
        const Complex l1 = tr / Complex{2.0, 0.0} + disc;
        const Complex l2 = tr / Complex{2.0, 0.0} - disc;

        auto coeff = [&](Complex lambda) {
          return std::exp(Complex{0.0, -problem.evolution.dt} * lambda);
        };
        const Complex e1 = coeff(l1);
        const Complex e2 = coeff(l2);

        const Complex denom = (l1 - l2);
        Complex c1{0.0, 0.0};
        Complex c2{0.0, 0.0};
        if (std::abs(denom) > 1e-14) {
          c1 = ((alpha - l2) * e1 - (alpha - l1) * e2) / denom;
          c2 = Complex{beta, 0.0} * (e1 - e2) / denom;
        } else {
          c1 = std::exp(Complex{0.0, -problem.evolution.dt} * alpha);
        }

        for (std::size_t i = 0; i < state_.size(); ++i) {
          tmp[i] = c1 * v1[i] + c2 * w[i];
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

  void checkpoint(const std::string& path) override {
    std::ofstream os(path + "/statevector.chk", std::ios::binary);
    const std::size_t size = state_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size > 0) {
      os.write(reinterpret_cast<const char*>(state_.data()),
               static_cast<std::streamsize>(size * sizeof(std::complex<double>)));
    }
  }

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

  std::vector<std::complex<double>> current_state() const override { return state_; }

 private:
  std::vector<Complex> state_;
  std::vector<ParsedTerm> terms_;
};

}  // namespace

std::unique_ptr<IBackend> make_statevector_backend() {
  return std::make_unique<StatevectorBackend>();
}

}  // namespace qx
