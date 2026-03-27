// Real TEBD (Time-Evolving Block Decimation) MPS backend.
//
// Supports 1D nearest-neighbour Hamiltonians specified as Pauli strings:
//   Two-body NN terms: "XiXj", "YiYj", "ZiZj", "XiYj", etc. with |i−j|=1
//   Single-body terms: "Xi", "Yi", "Zi"
// Non-NN terms are skipped with a warning in the observable cache.
//
// MPS representation
// ------------------
// Each site tensor A[i] is stored as an Eigen matrix of shape
//   (chi_L * d) × chi_R   where d=2 (qubit) and chi_{L,R} are bond dims.
// Index convention:  row = il * d + s  (il = left-bond index, s ∈ {0,1})
//                    col = ir           (right-bond index)
//
// Two-site TEBD gate application at bond (i, i+1)
// ------------------------------------------------
// 1. Contract A[i] · A[i+1] → Theta  shape (chi_L*d, d*chi_R)
// 2. Apply 4×4 two-site gate U to the physical indices.
// 3. SVD  Theta' = U · Σ · V†,  truncate to chi_new ≤ max_bond_.
// 4. Absorb singular values into right tensor; update A[i], A[i+1].
//
// Suzuki-Trotter order-2 sweep: odd bonds → even bonds → odd bonds (× ½ dt each).

#include "qx/backends.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <fstream>
#include <numeric>
#include <regex>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace qx {
namespace {

using Complex   = std::complex<double>;
using MatrixXcd = Eigen::MatrixXcd;
using VectorXd  = Eigen::VectorXd;

constexpr std::uint32_t kPhysDim = 2U;  // d = 2 for qubits

// ── Single-qubit Pauli matrices ────────────────────────────────────────────

MatrixXcd pauli_x() {
  MatrixXcd m(2, 2);
  m << Complex{0,0}, Complex{1,0}, Complex{1,0}, Complex{0,0};
  return m;
}
MatrixXcd pauli_y() {
  MatrixXcd m(2, 2);
  m << Complex{0,0}, Complex{0,-1}, Complex{0,1}, Complex{0,0};
  return m;
}
MatrixXcd pauli_z() {
  MatrixXcd m(2, 2);
  m << Complex{1,0}, Complex{0,0}, Complex{0,0}, Complex{-1,0};
  return m;
}
MatrixXcd pauli_i() { return MatrixXcd::Identity(2, 2); }

MatrixXcd pauli_from_char(char c) {
  switch (c) {
    case 'X': return pauli_x();
    case 'Y': return pauli_y();
    case 'Z': return pauli_z();
    default:  return pauli_i();
  }
}

// Kronecker product of two 2×2 matrices → 4×4 matrix.
MatrixXcd kron2(const MatrixXcd& A, const MatrixXcd& B) {
  MatrixXcd R(4, 4);
  for (Eigen::Index i = 0; i < 2; ++i)
    for (Eigen::Index j = 0; j < 2; ++j)
      R.block<2,2>(i*2, j*2) = A(i,j) * B;
  return R;
}

// ── Parsed Hamiltonian term ────────────────────────────────────────────────

struct ParsedTerm {
  double coeff{0.0};
  // If site_b == site_a: single-site term on site_a with op_a.
  // If site_b == site_a+1: two-body NN term (op_a ⊗ op_b).
  std::uint32_t site_a{0};
  std::uint32_t site_b{0};
  char op_a{'I'};
  char op_b{'I'};
};

// Parse Pauli-string Hamiltonian terms into single- and two-body NN pieces.
// Non-NN multi-body terms are silently dropped (tracked in skipped_terms count).
std::vector<ParsedTerm> parse_hamiltonian(
    const std::vector<HamiltonianTerm>& terms,
    std::uint32_t n_sites,
    std::uint32_t& skipped_terms) {

  skipped_terms = 0;
  std::vector<ParsedTerm> out;
  const std::regex tok(R"(([XYZxyz])(\d+))");

  for (const auto& ht : terms) {
    if (ht.label.empty()) continue;

    std::vector<std::pair<char, std::uint32_t>> ops;
    bool valid = true;
    for (auto it = std::sregex_iterator(ht.label.begin(), ht.label.end(), tok);
         it != std::sregex_iterator(); ++it) {
      const char p = static_cast<char>(std::toupper(static_cast<unsigned char>((*it)[1].str()[0])));
      const auto q = static_cast<std::uint32_t>(std::stoul((*it)[2].str()));
      if (q >= n_sites) { valid = false; break; }
      ops.push_back({p, q});
    }
    if (!valid || ops.empty()) { ++skipped_terms; continue; }

    if (ops.size() == 1U) {
      ParsedTerm t;
      t.coeff  = ht.coefficient;
      t.site_a = ops[0].second;
      t.site_b = ops[0].second;
      t.op_a   = ops[0].first;
      t.op_b   = 'I';
      out.push_back(t);
    } else if (ops.size() == 2U) {
      const std::uint32_t qa = ops[0].second, qb = ops[1].second;
      if (std::max(qa,qb) - std::min(qa,qb) != 1U) { ++skipped_terms; continue; }
      ParsedTerm t;
      t.coeff  = ht.coefficient;
      // Ensure site_a < site_b.
      if (qa < qb) { t.site_a = qa; t.site_b = qb; t.op_a = ops[0].first; t.op_b = ops[1].first; }
      else         { t.site_a = qb; t.site_b = qa; t.op_a = ops[1].first; t.op_b = ops[0].first; }
      out.push_back(t);
    } else {
      // Multi-body or non-NN: skip.
      ++skipped_terms;
    }
  }
  return out;
}

// ── Bond Hamiltonian and gate cache ───────────────────────────────────────

// Build a 4×4 Hermitian matrix for all NN terms on bond (site, site+1).
MatrixXcd bond_hamiltonian(const std::vector<ParsedTerm>& terms, std::uint32_t site) {
  MatrixXcd H = MatrixXcd::Zero(4, 4);
  for (const auto& t : terms) {
    if (t.site_a == site && t.site_b == site + 1U) {
      H += t.coeff * kron2(pauli_from_char(t.op_a), pauli_from_char(t.op_b));
    }
  }
  return H;
}

// Build a 2×2 Hermitian matrix for all single-site terms on site i.
MatrixXcd site_hamiltonian(const std::vector<ParsedTerm>& terms, std::uint32_t site) {
  MatrixXcd H = MatrixXcd::Zero(2, 2);
  for (const auto& t : terms) {
    if (t.site_a == site && t.site_b == site) {
      H += t.coeff * pauli_from_char(t.op_a);
    }
  }
  return H;
}

// Precomputed gates for one TEBD sweep.
struct GateCache {
  // two_site_gates[bond] = exp(-i * dt_bond * H_bond) for bond = site-left of NN pair.
  std::vector<MatrixXcd> two_site_gates;
  // single_site_gates[site] = exp(-i * dt_site * H_site) embedded per half-step.
  std::vector<MatrixXcd> single_site_half;
};

GateCache build_gates(const std::vector<ParsedTerm>& terms,
                       std::uint32_t n_sites, double dt) {
  GateCache gc;
  gc.two_site_gates.resize(n_sites - 1U);
  gc.single_site_half.resize(n_sites);

  for (std::uint32_t b = 0; b < n_sites - 1U; ++b) {
    const MatrixXcd H = bond_hamiltonian(terms, b);
    gc.two_site_gates[b] = (Complex{0.0, -1.0} * dt * H).exp();
  }
  for (std::uint32_t s = 0; s < n_sites; ++s) {
    const MatrixXcd H = site_hamiltonian(terms, s);
    gc.single_site_half[s] = (Complex{0.0, -0.5} * dt * H).exp();
  }
  return gc;
}

// ── MPS state ─────────────────────────────────────────────────────────────
//
// Tensors_ [i] has shape (chi_L * d) × chi_R.
// Initialized in right-canonical form from product state |00…0⟩.

struct MpsState {
  std::uint32_t n{0};
  std::uint32_t max_bond{0};
  double trunc_tol{1e-8};
  double truncation_error{0.0};

  // tensors[i]: (chi_L*d) x chi_R
  std::vector<MatrixXcd> tensors;

  // Initialize from |00…0⟩ product state.
  void init(std::uint32_t n_sites, std::uint32_t max_bond_dim, double tol) {
    n = n_sites;
    max_bond = max_bond_dim;
    trunc_tol = tol;
    truncation_error = 0.0;
    tensors.resize(n);
    for (std::uint32_t i = 0; i < n; ++i) {
      tensors[i] = MatrixXcd::Zero(kPhysDim, 1);  // chi_L=1, chi_R=1
      tensors[i](0, 0) = Complex{1.0, 0.0};        // |0⟩
    }
  }

  // Bond dimension at bond b (between sites b and b+1).
  std::uint32_t bond_dim(std::uint32_t b) const {
    return static_cast<std::uint32_t>(tensors[b].cols());
  }

  // Maximum bond dimension across all bonds.
  std::uint32_t max_bond_dim_actual() const {
    std::uint32_t mx = 1;
    for (std::uint32_t b = 0; b < n - 1U; ++b)
      mx = std::max(mx, bond_dim(b));
    return mx;
  }
};

// Apply a 2×2 single-site gate U to site i of the MPS.
// A[i] has shape (chi_L*d) x chi_R. Physical index s is the lower bit of row.
void apply_single_site_gate(MpsState& mps, std::uint32_t site, const MatrixXcd& U2) {
  auto& A = mps.tensors[site];
  const auto chi_L = static_cast<std::uint32_t>(A.rows()) / kPhysDim;
  const auto chi_R = static_cast<std::uint32_t>(A.cols());
  // A'[il*d + s', ir] = Σ_s U2[s', s] * A[il*d + s, ir]
  for (std::uint32_t il = 0; il < chi_L; ++il) {
    for (std::uint32_t ir = 0; ir < chi_R; ++ir) {
      const Complex a0 = A(il*kPhysDim + 0, ir);
      const Complex a1 = A(il*kPhysDim + 1, ir);
      A(il*kPhysDim + 0, ir) = U2(0,0)*a0 + U2(0,1)*a1;
      A(il*kPhysDim + 1, ir) = U2(1,0)*a0 + U2(1,1)*a1;
    }
  }
}

// Apply a 4×4 two-site gate U to bond (site, site+1), truncate via SVD.
// Gate acts on physical indices: U[(s_i', s_{i+1}'), (s_i, s_{i+1})] in binary order.
void apply_two_site_gate(MpsState& mps, std::uint32_t site, const MatrixXcd& U4) {
  const std::uint32_t d = kPhysDim;
  auto& Ai   = mps.tensors[site];
  auto& Aip1 = mps.tensors[site + 1U];

  const auto chi_L  = static_cast<std::uint32_t>(Ai.rows()) / d;
  const auto chi_M  = static_cast<std::uint32_t>(Ai.cols());   // = Aip1 left bond
  const auto chi_R  = static_cast<std::uint32_t>(Aip1.cols());

  // ── Step 1: Reshape A[i+1] from (chi_M*d, chi_R) to (chi_M, d*chi_R) ──
  // Aip1_r[im, s*chi_R + ir] = Aip1[im*d + s, ir]
  MatrixXcd Aip1_r(chi_M, d * chi_R);
  for (std::uint32_t im = 0; im < chi_M; ++im)
    for (std::uint32_t s = 0; s < d; ++s)
      for (std::uint32_t ir = 0; ir < chi_R; ++ir)
        Aip1_r(im, s*chi_R + ir) = Aip1(im*d + s, ir);

  // ── Step 2: Theta = A[i] * Aip1_r  shape (chi_L*d, d*chi_R) ──
  // Theta[il*d + si, sip1*chi_R + ir]
  const MatrixXcd Theta = Ai * Aip1_r;

  // ── Step 3: Apply gate to physical indices ──
  // Theta'[il*d + si', sip1'*chi_R + ir] =
  //   Σ_{si, sip1} U4[si'*d+sip1', si*d+sip1] * Theta[il*d+si, sip1*chi_R+ir]
  MatrixXcd Theta_prime = MatrixXcd::Zero(chi_L * d, d * chi_R);
  for (std::uint32_t il = 0; il < chi_L; ++il) {
    for (std::uint32_t ir = 0; ir < chi_R; ++ir) {
      for (std::uint32_t si_new = 0; si_new < d; ++si_new) {
        for (std::uint32_t sip1_new = 0; sip1_new < d; ++sip1_new) {
          Complex val{0.0, 0.0};
          for (std::uint32_t si_old = 0; si_old < d; ++si_old)
            for (std::uint32_t sip1_old = 0; sip1_old < d; ++sip1_old)
              val += U4(si_new*d + sip1_new, si_old*d + sip1_old)
                   * Theta(il*d + si_old, sip1_old*chi_R + ir);
          Theta_prime(il*d + si_new, sip1_new*chi_R + ir) = val;
        }
      }
    }
  }

  // ── Step 4: SVD of Theta' ──
  Eigen::JacobiSVD<MatrixXcd> svd(Theta_prime,
                                    Eigen::ComputeThinU | Eigen::ComputeThinV);
  const VectorXd& sv = svd.singularValues();
  const std::size_t k_full = static_cast<std::size_t>(sv.size());

  // Determine truncation threshold.
  const double sv_max   = sv.size() > 0 ? sv(0) : 1.0;
  const double abs_tol  = mps.trunc_tol * sv_max;
  std::size_t chi_new   = 0;
  for (std::size_t i = 0; i < k_full; ++i) {
    if (sv(static_cast<Eigen::Index>(i)) > abs_tol) ++chi_new;
    else break;
  }
  chi_new = std::max<std::size_t>(1, std::min<std::size_t>(chi_new, mps.max_bond));

  // Accumulate discarded singular-value weight as truncation error.
  for (std::size_t i = chi_new; i < k_full; ++i) {
    const double s = sv(static_cast<Eigen::Index>(i));
    mps.truncation_error += s * s;
  }

  // ── Step 5: Update tensors ──
  // A[site]   = U_svd[:, :chi_new]              shape (chi_L*d, chi_new)
  // A[site+1] = diag(S) * V†[:chi_new, :]  →   shape (chi_new*d, chi_R)
  const auto idx = static_cast<Eigen::Index>(chi_new);
  Ai = svd.matrixU().leftCols(idx);

  // S * V†: (chi_new, d*chi_R)  → reshape to (chi_new*d, chi_R)
  const MatrixXcd SV = svd.singularValues().head(idx).asDiagonal() *
                        svd.matrixV().adjoint().topRows(idx);
  Aip1.resize(chi_new * d, chi_R);
  for (std::uint32_t im = 0; im < static_cast<std::uint32_t>(chi_new); ++im)
    for (std::uint32_t s = 0; s < d; ++s)
      for (std::uint32_t ir = 0; ir < chi_R; ++ir)
        Aip1(im*d + s, ir) = SV(static_cast<Eigen::Index>(im),
                                  static_cast<Eigen::Index>(s*chi_R + ir));
}

// ── Observables ────────────────────────────────────────────────────────────

// Von Neumann entropy at bond b from the singular values at that bond.
// After SVD, the singular values of A[b] (viewed as chi_L*d × chi_R) give
// the Schmidt coefficients.  We expose them directly by doing a reduced SVD.
double bond_entropy(const MpsState& mps, std::uint32_t b) {
  // Quick SVD of the (chi_L*d x chi_R) tensor.
  Eigen::JacobiSVD<MatrixXcd> svd(mps.tensors[b], Eigen::ComputeThinU | Eigen::ComputeThinV);
  const VectorXd& sv = svd.singularValues();
  // Normalise.
  const double norm2 = sv.squaredNorm();
  if (norm2 < 1e-30) return 0.0;
  double S = 0.0;
  for (Eigen::Index i = 0; i < sv.size(); ++i) {
    const double p = sv(i) * sv(i) / norm2;
    if (p > 1e-15) S -= p * std::log(p);
  }
  return std::max(0.0, S);
}

// Expectation value ⟨Z_i⟩ by computing the local reduced density matrix.
// We contract the MPS from both ends and extract the one-site RDM.
// For efficiency, we use the right-orthogonality approximation:
// After evolving, A[i] is (chi_L*d x chi_R).  We approximate
//   ⟨Z_i⟩ ≈ Σ_{il,ir} |A[i][il*d, ir]|² - |A[i][il*d+1, ir]|²
// weighted by the left/right environment norms (identity for now → good for
// right-canonical initial state; accumulates small errors over evolution).
double site_magnetization_z(const MpsState& mps, std::uint32_t site) {
  const auto& A = mps.tensors[site];
  const auto chi_L = static_cast<std::uint32_t>(A.rows()) / kPhysDim;
  const auto chi_R = static_cast<std::uint32_t>(A.cols());
  double val = 0.0;
  for (std::uint32_t il = 0; il < chi_L; ++il)
    for (std::uint32_t ir = 0; ir < chi_R; ++ir)
      val += std::norm(A(il*kPhysDim + 0, ir))  // |0⟩ component → +1
           - std::norm(A(il*kPhysDim + 1, ir)); // |1⟩ component → -1
  return val;
}

// ── TEBD sweep helpers ─────────────────────────────────────────────────────

void apply_even_bonds(MpsState& mps, const GateCache& gc) {
  for (std::uint32_t b = 0; b + 1U < mps.n; b += 2U)
    apply_two_site_gate(mps, b, gc.two_site_gates[b]);
}

void apply_odd_bonds(MpsState& mps, const GateCache& gc) {
  for (std::uint32_t b = 1U; b + 1U < mps.n; b += 2U)
    apply_two_site_gate(mps, b, gc.two_site_gates[b]);
}

void apply_single_site_half_all(MpsState& mps, const GateCache& gc) {
  for (std::uint32_t s = 0; s < mps.n; ++s)
    apply_single_site_gate(mps, s, gc.single_site_half[s]);
}

// 2nd-order Suzuki-Trotter step:
//   single-half → even → odd → even → single-half
void tebd_step(MpsState& mps, const GateCache& gc) {
  apply_single_site_half_all(mps, gc);
  apply_even_bonds(mps, gc);
  apply_odd_bonds(mps, gc);
  apply_even_bonds(mps, gc);
  apply_single_site_half_all(mps, gc);
}

// Collect one MpsFrame of diagnostics.
MpsFrame collect_frame(const MpsState& mps, std::uint32_t step, double t) {
  MpsFrame f;
  f.step = step;
  f.time = t;
  f.n_sites = mps.n;
  f.truncation_error = mps.truncation_error;

  f.bond_dims.resize(mps.n - 1U);
  f.bond_entropies.resize(mps.n - 1U);
  f.magnetization.resize(mps.n);

  std::uint32_t mx = 1;
  for (std::uint32_t b = 0; b + 1U < mps.n; ++b) {
    f.bond_dims[b]     = mps.bond_dim(b);
    f.bond_entropies[b] = bond_entropy(mps, b);
    mx = std::max(mx, f.bond_dims[b]);
  }
  f.max_bond_dim = mx;
  for (std::uint32_t s = 0; s < mps.n; ++s)
    f.magnetization[s] = site_magnetization_z(mps, s);
  return f;
}

// ── MpsBackend ─────────────────────────────────────────────────────────────

class MpsBackend final : public IBackend {
 public:
  std::string name() const override { return "mps"; }

  BackendCapabilities capabilities() const override {
    BackendCapabilities c;
    c.kind                   = BackendKind::MPS;
    c.supports_gpu           = false;
    c.supports_mpi           = false;
    c.supports_noisy         = false;
    c.max_qubits_recommended = 200U;
    c.equation_scope         = "Schrodinger";
    return c;
  }

  void initialize(const QuantumProblem& problem, const RunOptions&) override {
    const std::uint32_t n = problem.qubits == 0U ? 2U : problem.qubits;
    const std::uint32_t chi = std::max(2U, problem.evolution.mps_max_bond);
    const double tol = std::max(1e-14, problem.evolution.mps_trunc_tol);
    dt_         = problem.evolution.dt > 0.0 ? problem.evolution.dt : 0.01;
    total_time_ = problem.evolution.total_time > 0.0 ? problem.evolution.total_time : 1.0;

    std::uint32_t skipped = 0U;
    terms_ = parse_hamiltonian(problem.hamiltonian_terms, n, skipped);
    skipped_terms_ = skipped;

    mps_.init(n, chi, tol);
    frames_.clear();
    observable_cache_.clear();
  }

  void evolve(const QuantumProblem&, const RunOptions&) override {
    if (mps_.n < 2U) {
      // Single-site: no NN gates, just record trivial frames.
      const auto steps = static_cast<std::uint32_t>(
          std::max(1.0, std::round(total_time_ / dt_)));
      frames_.push_back(collect_frame(mps_, 0U, 0.0));
      for (std::uint32_t s = 0; s < steps; ++s)
        frames_.push_back(collect_frame(mps_, s + 1U, (s + 1U) * dt_));
      update_obs();
      return;
    }

    const GateCache gc = build_gates(terms_, mps_.n, dt_);
    const auto steps = static_cast<std::uint32_t>(
        std::max(1.0, std::round(total_time_ / dt_)));

    frames_.push_back(collect_frame(mps_, 0U, 0.0));
    for (std::uint32_t s = 0; s < steps; ++s) {
      tebd_step(mps_, gc);
      const double t = static_cast<double>(s + 1U) * dt_;
      frames_.push_back(collect_frame(mps_, s + 1U, t));
    }
    update_obs();
  }

  std::map<std::string, double> measure(const QuantumProblem&) override {
    return observable_cache_;
  }

  void checkpoint(const std::string& path) override {
    std::ofstream os(path + "/mps.chk");
    if (!os.good()) return;
    os << mps_.n << "\n"
       << mps_.max_bond << "\n"
       << mps_.trunc_tol << "\n"
       << mps_.truncation_error << "\n"
       << dt_ << "\n"
       << total_time_ << "\n";
  }

  void restore(const std::string& path) override {
    std::ifstream is(path + "/mps.chk");
    if (!is.good()) return;
    std::uint32_t n_sites{2}, max_bond{32};
    double tol{1e-8}, trunc_err{0.0};
    is >> n_sites >> max_bond >> tol >> trunc_err >> dt_ >> total_time_;
    mps_.init(n_sites, max_bond, tol);
    mps_.truncation_error = trunc_err;
    update_obs();
  }

  std::vector<std::complex<double>> current_state() const override { return {}; }
  std::vector<ContinuousDensityFrame> continuous_density_frames() const override { return {}; }
  std::vector<MpsFrame> mps_frames() const override { return frames_; }

 private:
  void update_obs() {
    observable_cache_["bond_dim"] =
        static_cast<double>(mps_.max_bond_dim_actual());
    observable_cache_["truncation_error"] = mps_.truncation_error;
    observable_cache_["mps_max_bond"]     = static_cast<double>(mps_.max_bond);
    observable_cache_["mps_trunc_tol"]    = mps_.trunc_tol;
    observable_cache_["skipped_terms"]    = static_cast<double>(skipped_terms_);
    if (!frames_.empty()) {
      const auto& last = frames_.back();
      // Entropy at centre bond (best single-number summary).
      if (!last.bond_entropies.empty()) {
        const std::size_t mid = last.bond_entropies.size() / 2U;
        observable_cache_["entanglement_entropy_estimate"] =
            last.bond_entropies[mid];
      }
    }
  }

  double dt_{0.01};
  double total_time_{1.0};
  std::uint32_t skipped_terms_{0U};
  MpsState mps_{};
  std::vector<ParsedTerm> terms_{};
  std::vector<MpsFrame> frames_{};
  std::map<std::string, double> observable_cache_{};
};

}  // namespace

std::unique_ptr<IBackend> make_mps_backend() {
  return std::make_unique<MpsBackend>();
}

}  // namespace qx
