#include "qx/backends.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <limits>
#include <regex>
#include <string>
#include <vector>

namespace qx {
namespace {

using Complex = std::complex<double>;
using CMatrix = std::vector<std::vector<Complex>>;

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

CMatrix eye(std::size_t n) {
  CMatrix I(n, std::vector<Complex>(n, Complex{0.0, 0.0}));
  for (std::size_t i = 0; i < n; ++i) {
    I[i][i] = Complex{1.0, 0.0};
  }
  return I;
}

double matrix_one_norm(const CMatrix& A) {
  if (A.empty()) return 0.0;
  const std::size_t n = A.size();
  double max_col = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    double col_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      col_sum += std::abs(A[i][j]);
    }
    max_col = std::max(max_col, col_sum);
  }
  return max_col;
}

CMatrix mat_mul(const CMatrix& A, const CMatrix& B) {
  const std::size_t n = A.size();
  CMatrix C(n, std::vector<Complex>(n, Complex{0.0, 0.0}));
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t k = 0; k < n; ++k) {
      const Complex aik = A[i][k];
      if (std::abs(aik) < 1e-15) continue;
      for (std::size_t j = 0; j < n; ++j) {
        C[i][j] += aik * B[k][j];
      }
    }
  }
  return C;
}

CMatrix mat_add(const CMatrix& A, const CMatrix& B) {
  const std::size_t n = A.size();
  CMatrix C = A;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      C[i][j] += B[i][j];
    }
  }
  return C;
}

CMatrix mat_scale(const CMatrix& A, Complex alpha) {
  CMatrix C = A;
  for (auto& row : C) {
    for (auto& x : row) {
      x *= alpha;
    }
  }
  return C;
}

CMatrix mat_exp_taylor_scaled(const CMatrix& A) {
  const std::size_t n = A.size();
  if (n == 0) return {};
  if (n == 1) return CMatrix{{std::exp(A[0][0])}};

  const double n1 = matrix_one_norm(A);
  int s = 0;
  if (n1 > 0.5) {
    s = static_cast<int>(std::ceil(std::log2(n1 / 0.5)));
  }
  const double scale = std::ldexp(1.0, s);
  const CMatrix As = mat_scale(A, Complex{1.0 / scale, 0.0});

  CMatrix term = eye(n);
  CMatrix result = eye(n);
  for (int k = 1; k <= 40; ++k) {
    term = mat_mul(term, As);
    const double invk = 1.0 / static_cast<double>(k);
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        term[i][j] *= invk;
        result[i][j] += term[i][j];
      }
    }
    if (matrix_one_norm(term) < 1e-13) {
      break;
    }
  }

  for (int i = 0; i < s; ++i) {
    result = mat_mul(result, result);
  }
  return result;
}

std::vector<Complex> mat_vec_mul(const CMatrix& A, const std::vector<Complex>& x) {
  const std::size_t n = A.size();
  std::vector<Complex> y(n, Complex{0.0, 0.0});
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      y[i] += A[i][j] * x[j];
    }
  }
  return y;
}

std::size_t arnoldi(
    const std::vector<Complex>& v0,
    std::size_t m,
    const std::vector<ParsedTerm>& terms,
    std::vector<std::vector<Complex>>& V,
    CMatrix& H) {
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
        CMatrix Hfull;
        const std::size_t k = arnoldi(v0, max_dim, terms_, V, Hfull);
        if (k == 0 || V.empty()) {
          continue;
        }

        CMatrix Hk(k, std::vector<Complex>(k, Complex{0.0, 0.0}));
        for (std::size_t i = 0; i < k; ++i) {
          for (std::size_t j = 0; j < k; ++j) {
            Hk[i][j] = Hfull[i][j];
          }
        }

        const CMatrix A = mat_scale(Hk, Complex{0.0, -problem.evolution.dt});
        const CMatrix expA = mat_exp_taylor_scaled(A);

        std::vector<Complex> e1(k, Complex{0.0, 0.0});
        e1[0] = Complex{1.0, 0.0};
        const std::vector<Complex> y = mat_vec_mul(expA, e1);

        std::fill(tmp.begin(), tmp.end(), Complex{0.0, 0.0});
        for (std::size_t j = 0; j < k; ++j) {
          axpy(tmp, y[j], V[j]);
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
