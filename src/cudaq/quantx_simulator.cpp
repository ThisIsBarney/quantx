// The CUDA-Q target "quantx" (M14): an NVQIR circuit simulator that runs a kernel on QuantX's
// block-sparse MPS, through the engine's C API (qx_mps, qx_mps_sample, qx_mps_amplitudes).
//
// Gates are kept until a result is needed, then fused, greedily, into clusters of at most two
// qubits, each closing once its product conserves the number of qubits in |1>. When every
// cluster closes, the state runs with that number as an exact charge, block by block; when one
// cannot, it runs on the same kernels with no charge. CUDA-Q hands a simulator its gates already
// decomposed (exp_pauli into H, Rx, CNOT and Rz; a Givens rotation into CNOTs and rotations), and
// the fusion is what recovers the symmetry. X gates on qubits in a known basis state move the
// starting bitstring (or, once the state exists, apply |1><0| or |0><1|), as do measurements and
// resets, so preparation, mid-circuit measurement and reset keep the charge.
//
// Options, by environment variable:
//   QUANTX_MAX_BOND  the bond cap (default: none, so a run is exact unless told otherwise)
//   QUANTX_CUTOFF    the discarded weight allowed per split (default 1e-14)
//   QUANTX_CONSERVE  auto (default), 1 to refuse a kernel that cannot use the charge, or 0
//   QUANTX_SVD       direct (default) or gram
//   QUANTX_VERBOSE   1 to report, on stderr, whether each state used the charge, its largest
//                    bond and the weight its truncations discarded

#include "nvqir/CircuitSimulator.h"
#include "qx/c_api.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace nvqir {
namespace {

using cplx = std::complex<double>;
using json = nlohmann::json;

// A cluster closes once its charge-changing entries are this small, and they are then zeroed.
// The engine refuses a gate whose charge-changing entries exceed 1e-10.
constexpr double kChargeTol = 1e-12;
// The most qubits whose amplitudes get_state reads: 2^24 of them, 256 MB.
constexpr std::size_t kMaxDenseQubits = 24;
// The most controls a gate on one target may carry: its decomposition doubles, and more,
// with each one (53 two-qubit gates at four, 485 at six).
constexpr std::size_t kMaxControls = 8;

// ---- small dense matrices ------------------------------------------------------------------

struct Mat {
  std::size_t n = 0;     // n x n
  std::vector<cplx> a;   // row-major
  Mat() = default;
  explicit Mat(std::size_t n) : n(n), a(n * n) {}
  static Mat identity(std::size_t n) {
    Mat m(n);
    for (std::size_t i = 0; i < n; ++i)
      m(i, i) = 1;
    return m;
  }
  cplx &operator()(std::size_t r, std::size_t c) { return a[r * n + c]; }
  cplx operator()(std::size_t r, std::size_t c) const { return a[r * n + c]; }
};

Mat operator*(const Mat &x, const Mat &y) {
  Mat out(x.n);
  for (std::size_t r = 0; r < x.n; ++r)
    for (std::size_t k = 0; k < x.n; ++k)
      if (x(r, k) != cplx(0))
        for (std::size_t c = 0; c < x.n; ++c)
          out(r, c) += x(r, k) * y(k, c);
  return out;
}

Mat adjoint(const Mat &x) {
  Mat out(x.n);
  for (std::size_t r = 0; r < x.n; ++r)
    for (std::size_t c = 0; c < x.n; ++c)
      out(c, r) = std::conj(x(r, c));
  return out;
}

Mat pauliX() {
  Mat x(2);
  x(0, 1) = x(1, 0) = 1;
  return x;
}

// The gate controlled by its high bit: diag(1, 1, u).
Mat controlled(const Mat &u) {
  Mat out = Mat::identity(4);
  for (std::size_t r = 0; r < 2; ++r)
    for (std::size_t c = 0; c < 2; ++c)
      out(2 + r, 2 + c) = u(r, c);
  return out;
}

// A square root of a 2x2 unitary: (U + s) / t with s^2 = det U and t^2 = tr U + 2 s, the branch
// of s taken that keeps t away from zero.
Mat sqrtUnitary(const Mat &u) {
  const cplx det = u(0, 0) * u(1, 1) - u(0, 1) * u(1, 0);
  const cplx tr = u(0, 0) + u(1, 1);
  cplx s = std::sqrt(det);
  if (std::abs(tr + 2.0 * s) < std::abs(tr - 2.0 * s))
    s = -s;
  const cplx t = std::sqrt(tr + 2.0 * s);
  Mat v(2);
  for (std::size_t r = 0; r < 2; ++r)
    for (std::size_t c = 0; c < 2; ++c)
      v(r, c) = (u(r, c) + (r == c ? s : cplx(0))) / t;
  return v;
}

// The largest entry between basis states that differ in their number of |1>s.
double chargeLeak(const Mat &u) {
  double leak = 0;
  for (std::size_t r = 0; r < u.n; ++r)
    for (std::size_t c = 0; c < u.n; ++c)
      if (std::popcount(r) != std::popcount(c))
        leak = std::max(leak, std::abs(u(r, c)));
  return leak;
}

void zeroLeak(Mat &u) {
  for (std::size_t r = 0; r < u.n; ++r)
    for (std::size_t c = 0; c < u.n; ++c)
      if (std::popcount(r) != std::popcount(c))
        u(r, c) = 0;
}

// ---- gates ---------------------------------------------------------------------------------

// A gate on one or two qubits, q[0] the high bit of its matrix's index.
struct Gate {
  std::vector<std::size_t> q;
  Mat u;
};

bool isX(const Gate &g) {
  return g.q.size() == 1 && g.u(0, 0) == cplx(0) && g.u(1, 1) == cplx(0) &&
         g.u(0, 1) == cplx(1) && g.u(1, 0) == cplx(1);
}

bool isSwap(const Mat &u) {
  if (u.n != 4)
    return false;
  for (std::size_t r = 0; r < 4; ++r)
    for (std::size_t c = 0; c < 4; ++c) {
      const bool one = (r == c && (r == 0 || r == 3)) || (r == 1 && c == 2) || (r == 2 && c == 1);
      if (u(r, c) != cplx(one ? 1 : 0))
        return false;
    }
  return true;
}

// u, on qubits `from` (from[0] the high bit), as a matrix on `to`, which holds them all.
Mat embed(const Mat &u, const std::vector<std::size_t> &from,
          const std::vector<std::size_t> &to) {
  const std::size_t k = to.size(), dim = std::size_t(1) << k;
  std::vector<std::size_t> shift(from.size());
  std::size_t mask = 0;
  for (std::size_t i = 0; i < from.size(); ++i) {
    const auto at = std::size_t(std::find(to.begin(), to.end(), from[i]) - to.begin());
    shift[i] = k - 1 - at;
    mask |= std::size_t(1) << shift[i];
  }
  auto sub = [&](std::size_t x) {
    std::size_t s = 0;
    for (std::size_t i = 0; i < from.size(); ++i)
      s = (s << 1) | ((x >> shift[i]) & 1);
    return s;
  };
  Mat out(dim);
  for (std::size_t r = 0; r < dim; ++r)
    for (std::size_t c = 0; c < dim; ++c)
      if ((r & ~mask) == (c & ~mask))
        out(r, c) = u(sub(r), sub(c));
  return out;
}

// U on `target` controlled by every qubit of `controls`, as gates on one and two qubits:
// Barenco et al. (1995), lemma 7.1, with V^2 = U,
//   C^k(U) = CV(c_k, t) . C^(k-1)X(c_1..c_k-1; c_k) . CV^+(c_k, t) . C^(k-1)X . C^(k-1)V(t),
// in the order they act.
void decomposeControlled(std::vector<Gate> &out, const Mat &u,
                         const std::vector<std::size_t> &controls, std::size_t target) {
  if (controls.empty()) {
    out.push_back({{target}, u});
    return;
  }
  if (controls.size() == 1) {
    out.push_back({{controls[0], target}, controlled(u)});
    return;
  }
  const Mat v = sqrtUnitary(u);
  const std::size_t last = controls.back();
  const std::vector<std::size_t> rest(controls.begin(), controls.end() - 1);
  out.push_back({{last, target}, controlled(v)});
  decomposeControlled(out, pauliX(), rest, last);
  out.push_back({{last, target}, controlled(adjoint(v))});
  decomposeControlled(out, pauliX(), rest, last);
  decomposeControlled(out, v, rest, target);
}

// A CUDA-Q gate task as gates on one and two qubits. Its matrix is row-major with targets[0]
// the high bit (the simulator asks for MSB ordering), and acts when every control is |1>.
template <typename Task>
std::vector<Gate> gatesOf(const Task &task) {
  const auto &controls = task.controls;
  const auto &targets = task.targets;
  const std::size_t dim = std::size_t(1) << targets.size();
  if (targets.empty() || targets.size() > 2 || task.matrix.size() != dim * dim)
    throw std::runtime_error("quantx: gate '" + task.operationName + "' acts on " +
                             std::to_string(targets.size()) +
                             " target qubits; the target runs gates on one or two");
  if (controls.size() > kMaxControls)
    throw std::runtime_error("quantx: gate '" + task.operationName + "' has " +
                             std::to_string(controls.size()) + " controls; the target "
                             "decomposes gates with up to " + std::to_string(kMaxControls));
  Mat u(dim);
  for (std::size_t r = 0; r < dim; ++r)
    for (std::size_t c = 0; c < dim; ++c)
      u(r, c) = cplx(task.matrix[r * dim + c]);
  std::vector<Gate> out;
  if (targets.size() == 1) {
    decomposeControlled(out, u, controls, targets[0]);
  } else if (controls.empty()) {
    out.push_back({{targets[0], targets[1]}, u});
  } else if (isSwap(u)) {
    // a controlled SWAP: CNOT(b -> a), then X on b controlled by the controls and a, then the
    // CNOT again
    out.push_back({{targets[1], targets[0]}, controlled(pauliX())});
    std::vector<std::size_t> c = controls;
    c.push_back(targets[0]);
    decomposeControlled(out, pauliX(), c, targets[1]);
    out.push_back({{targets[1], targets[0]}, controlled(pauliX())});
  } else {
    throw std::runtime_error("quantx: gate '" + task.operationName + "' is a two-qubit gate "
                             "with controls; the target runs a controlled SWAP, and controlled "
                             "gates on one qubit");
  }
  return out;
}

// ---- the operation log ---------------------------------------------------------------------

struct Op {
  enum Kind { gate, flip, measure, reset } kind = gate;
  Gate g;              // gate
  std::size_t q = 0;   // flip, measure, reset
  int bit = 0;         // flip: the qubit's value before; measure and reset: the outcome
};

// `ops` (gates and flips) with the gates fused into clusters of at most two qubits that each
// conserve the number of |1>s; nothing if a cluster outgrows two qubits or one is still open
// at the end. A flip acts on a qubit in a known basis state, so on none that an open cluster
// holds, and commutes with them all.
std::optional<std::vector<Op>> fuse(const std::vector<Op> &ops) {
  struct Cluster {
    std::vector<std::size_t> q;
    Mat u;
  };
  std::map<std::size_t, int> openOf;   // qubit -> the open cluster that holds it
  std::map<int, Cluster> open;
  std::vector<Op> out;
  int next = 0;
  for (const Op &op : ops) {
    if (op.kind != Op::gate) {
      out.push_back(op);
      continue;
    }
    const Gate &g = op.g;
    std::vector<int> ids;
    for (auto q : g.q) {
      auto it = openOf.find(q);
      if (it != openOf.end() && std::find(ids.begin(), ids.end(), it->second) == ids.end())
        ids.push_back(it->second);
    }
    int id;
    if (ids.empty()) {
      id = next++;
      open[id] = Cluster{g.q, Mat::identity(std::size_t(1) << g.q.size())};
    } else {
      id = ids[0];
      Cluster &a = open[id];
      for (std::size_t k = 1; k < ids.size(); ++k) {   // clusters on disjoint qubits commute
        Cluster b = std::move(open[ids[k]]);
        open.erase(ids[k]);
        std::vector<std::size_t> q = a.q;
        q.insert(q.end(), b.q.begin(), b.q.end());
        if (q.size() > 2)
          return std::nullopt;
        a.u = embed(b.u, b.q, q) * embed(a.u, a.q, q);
        a.q = q;
      }
      for (auto q : g.q)
        if (std::find(a.q.begin(), a.q.end(), q) == a.q.end()) {
          std::vector<std::size_t> grown = a.q;
          grown.push_back(q);
          if (grown.size() > 2)
            return std::nullopt;
          a.u = embed(a.u, a.q, grown);
          a.q = grown;
        }
    }
    Cluster &c = open[id];
    c.u = embed(g.u, g.q, c.q) * c.u;
    for (auto q : c.q)
      openOf[q] = id;
    if (chargeLeak(c.u) <= kChargeTol) {
      Op fused;
      fused.g = Gate{c.q, c.u};
      zeroLeak(fused.g.u);
      out.push_back(std::move(fused));
      for (auto q : c.q)
        openOf.erase(q);
      open.erase(id);
    }
  }
  if (!open.empty())
    return std::nullopt;
  return out;
}

// `ops` with their gates fused into clusters of at most two qubits whatever they conserve: a
// cluster closes when a gate would take it to a third qubit, or at a flip of one of its qubits.
// A fused gate is one truncating update where its gates were several (a CNOT, an Rz and a CNOT,
// two). Products of gates that conserve the number of |1>s conserve it exactly too, so this
// also merges what fuse() returns.
std::vector<Op> fuseBySize(const std::vector<Op> &ops) {
  struct Cluster {
    std::vector<std::size_t> q;
    Mat u;
  };
  std::map<std::size_t, int> openOf;   // qubit -> the open cluster that holds it
  std::map<int, Cluster> open;
  std::vector<Op> out;
  int next = 0;
  auto close = [&](int id) {
    Op fused;
    fused.g = Gate{open[id].q, open[id].u};
    out.push_back(std::move(fused));
    for (auto q : open[id].q)
      openOf.erase(q);
    open.erase(id);
  };
  for (const Op &op : ops) {
    if (op.kind != Op::gate) {
      if (auto it = openOf.find(op.q); it != openOf.end())
        close(it->second);
      out.push_back(op);
      continue;
    }
    const Gate &g = op.g;
    std::vector<int> ids;
    std::vector<std::size_t> qs = g.q;
    for (auto q : g.q) {
      auto it = openOf.find(q);
      if (it == openOf.end() || std::find(ids.begin(), ids.end(), it->second) != ids.end())
        continue;
      ids.push_back(it->second);
      for (auto p : open[it->second].q)
        if (std::find(qs.begin(), qs.end(), p) == qs.end())
          qs.push_back(p);
    }
    if (qs.size() > 2) {   // the gate goes to a cluster of its own
      for (int id : ids)
        close(id);
      ids.clear();
    }
    int id;
    if (ids.empty()) {
      id = next++;
      open[id] = Cluster{g.q, Mat::identity(std::size_t(1) << g.q.size())};
    } else {
      id = ids[0];
      Cluster &a = open[id];
      for (std::size_t k = 1; k < ids.size(); ++k) {   // clusters on disjoint qubits commute
        Cluster b = std::move(open[ids[k]]);
        open.erase(ids[k]);
        std::vector<std::size_t> q = a.q;
        q.insert(q.end(), b.q.begin(), b.q.end());
        a.u = embed(b.u, b.q, q) * embed(a.u, a.q, q);
        a.q = q;
      }
      for (auto q : g.q)
        if (std::find(a.q.begin(), a.q.end(), q) == a.q.end()) {
          std::vector<std::size_t> grown = a.q;
          grown.push_back(q);
          a.u = embed(a.u, a.q, grown);
          a.q = grown;
        }
    }
    Cluster &c = open[id];
    c.u = embed(g.u, g.q, c.q) * c.u;
    for (auto q : c.q)
      openOf[q] = id;
  }
  while (!open.empty())   // disjoint, so in any order
    close(open.begin()->first);
  return out;
}

json engine(const json &request) {
  const json out = json::parse(qx_mps(request.dump().c_str()));
  if (!out.value("ok", false))
    throw std::runtime_error("quantx: " + out.value("error", std::string("engine failure")));
  return out;
}

void check(const char *raw) {
  const json out = json::parse(raw);
  if (!out.value("ok", false))
    throw std::runtime_error("quantx: " + out.value("error", std::string("engine failure")));
}

json flatPairs(const Mat &u) {   // "gates": re, im, re, im, ...
  json out = json::array();
  for (const auto &z : u.a) {
    out.push_back(z.real());
    out.push_back(z.imag());
  }
  return out;
}

const char *env(const char *name) {
  const char *v = std::getenv(name);
  return v && *v ? v : nullptr;
}

} // namespace

// ---- get_state -----------------------------------------------------------------------------

// The amplitudes, in CUDA-Q's order: qubit k is bit k of an index.
struct QuantXState : public cudaq::SimulationState {
  std::vector<cplx> amp;

  explicit QuantXState(std::vector<cplx> a) : amp(std::move(a)) {}

  std::size_t getNumQubits() const override { return std::countr_zero(amp.size()); }

  std::complex<double> overlap(const cudaq::SimulationState &other) override {
    if (other.getNumTensors() != 1 || other.getTensor().extents != getTensor().extents ||
        other.getPrecision() != precision::fp64)
      throw std::runtime_error("[quantx-state] overlap: the other state is not a vector of "
                               "the same size in double precision");
    const auto *o = reinterpret_cast<const cplx *>(other.getTensor().data);
    cplx s = 0;
    for (std::size_t i = 0; i < amp.size(); ++i)
      s += amp[i] * std::conj(o[i]);
    return std::abs(s);
  }

  std::complex<double> getAmplitude(const std::vector<int> &basisState) override {
    if (basisState.size() != getNumQubits())
      throw std::runtime_error("[quantx-state] getAmplitude: expected " +
                               std::to_string(getNumQubits()) + " bits, got " +
                               std::to_string(basisState.size()));
    std::size_t idx = 0;
    for (std::size_t k = 0; k < basisState.size(); ++k) {
      if (basisState[k] != 0 && basisState[k] != 1)
        throw std::runtime_error("[quantx-state] getAmplitude: bits are 0 or 1");
      idx |= std::size_t(basisState[k]) << k;
    }
    return amp[idx];
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    if (tensorIdx != 0)
      throw std::runtime_error("[quantx-state] the state is one tensor");
    return Tensor{const_cast<cplx *>(amp.data()), {amp.size()}, getPrecision()};
  }

  std::vector<Tensor> getTensors() const override { return {getTensor()}; }

  std::size_t getNumTensors() const override { return 1; }

  std::complex<double> operator()(std::size_t tensorIdx,
                                  const std::vector<std::size_t> &indices) override {
    if (tensorIdx != 0 || indices.size() != 1 || indices[0] >= amp.size())
      throw std::runtime_error("[quantx-state] invalid element");
    return amp[indices[0]];
  }

  void dump(std::ostream &os) const override {
    for (const auto &z : amp)
      os << z << "\n";
  }

  precision getPrecision() const override { return precision::fp64; }

  void destroyState() override { std::vector<cplx>().swap(amp); }

  void toHost(std::complex<double> *out, std::size_t numElements) const override {
    std::copy_n(amp.begin(), std::min(numElements, amp.size()), out);
  }

protected:
  std::unique_ptr<cudaq::SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    if (!ptr || size == 0)
      throw std::runtime_error("[quantx-state] invalid null pointer or zero size");
    const auto *p = reinterpret_cast<const cplx *>(ptr);
    return std::make_unique<QuantXState>(std::vector<cplx>(p, p + size));
  }
};

// ---- the simulator -------------------------------------------------------------------------

class QuantXSimulator : public nvqir::CircuitSimulatorBase<double> {
  enum class Conserve { automatic, always, never };

  int maxBond = 0;   // 0: no cap
  double cutoff = 1e-14;
  Conserve conserve = Conserve::automatic;
  bool direct = true;
  bool verbose = false;

  std::size_t nQubits = 0;
  std::vector<int> bits;      // the starting bitstring
  std::vector<int> known;     // a qubit's value while it is in a basis state, else -1
  std::vector<Op> pending;    // since the state was last brought up to date
  std::vector<Op> log;        // all that the state holds, for a replay
  long long handle = -1;
  bool charged = false;
  bool stale = false;         // qubits were added since the state was made
  int bondReached = 1;
  double truncationSum = 0;
  std::mt19937_64 rng{std::random_device{}()};

  int bond() const {
    if (maxBond > 0)
      return maxBond;
    return 1 << std::min<std::size_t>(nQubits / 2 + 1, 30);   // no state exceeds 2^(n/2)
  }

  void freeState() {
    if (handle < 0)
      return;
    if (verbose)
      std::cerr << "quantx: " << nQubits << " qubits, the charge " << (charged ? "used" : "not used")
                << ", bond " << bondReached << ", truncation " << truncationSum << std::endl;
    engine({{"op", "free"}, {"handle", handle}});
    handle = -1;
  }

  void clearCircuit() {
    freeState();
    bits.assign(nQubits, 0);
    known.assign(nQubits, 0);
    pending.clear();
    log.clear();
    charged = stale = false;
    bondReached = 1;
    truncationSum = 0;
  }

  // One-site operators, applied as the engine's "apply": real diagonals, or a matrix that may
  // move the charge by one amount.
  void applyDiag(std::size_t q, double d0, double d1) {
    engine({{"op", "apply"}, {"handle", handle}, {"site", q}, {"diag", {d0, d1}}});
  }
  void applyMatrix(std::size_t q, int row, int col) {   // |row><col|
    json m = json::array();
    for (int r = 0; r < 2; ++r)
      for (int c = 0; c < 2; ++c)
        m.push_back({r == row && c == col ? 1.0 : 0.0, 0.0});
    engine({{"op", "apply"}, {"handle", handle}, {"site", q}, {"matrix", m}});
  }

  double probabilityOfOne(std::size_t q) {
    const json r = engine({{"op", "pauli"}, {"handle", handle},
                           {"terms", json::array({json::array({json::array({q, "Z"})})})}});
    return std::clamp((1.0 - r["values"][0].get<double>()) / 2.0, 0.0, 1.0);
  }

  // Measurement's projector, renormalised by the probability of the outcome (p1: of |1>).
  void project(std::size_t q, int one, double p1) {
    const double p = one ? p1 : 1.0 - p1;
    if (!(p > 0))
      throw std::runtime_error("quantx: a measurement outcome of probability 0");
    applyDiag(q, one ? 0.0 : 1.0 / std::sqrt(p), one ? 1.0 / std::sqrt(p) : 0.0);
  }

  // Gates and flips, in order, on the state as it is.
  void applyRun(const std::vector<Op> &ops) {
    json gates = json::array();
    auto sendGates = [&] {
      if (gates.empty())
        return;
      const json r = engine({{"op", "gates"}, {"handle", handle}, {"gates", gates},
                             {"max_bond", bond()}, {"cutoff", cutoff}, {"direct", direct}});
      bondReached = std::max(bondReached, r["max_bond"].get<int>());
      truncationSum += r["truncation_sum"].get<double>();
      gates = json::array();
    };
    for (const Op &op : ops) {
      if (op.kind == Op::gate) {
        gates.push_back({{"q", op.g.q}, {"u", flatPairs(op.g.u)}});
      } else if (op.kind == Op::flip) {
        sendGates();
        applyMatrix(op.q, op.bit ? 0 : 1, op.bit ? 1 : 0);   // |0> -> |1>, or back
      }
    }
    sendGates();
  }

  // A run of gates and flips as the state's representation takes it: with the charge, fused
  // into conserving gates (which the caller has checked it can) and those merged; without, the
  // gates merged as they come.
  void applyRunAs(const std::vector<Op> &ops) {
    if (!charged) {
      applyRun(fuseBySize(ops));
      return;
    }
    auto fused = fuse(ops);
    if (!fused)
      throw std::logic_error("quantx: a run that was checked to fuse did not");
    applyRun(fuseBySize(*fused));
  }

  // Whether every run of the log, between its measurements and resets, fuses.
  static bool fusesThroughout(const std::vector<Op> &ops) {
    std::vector<Op> run;
    for (const Op &op : ops) {
      if (op.kind == Op::measure || op.kind == Op::reset) {
        if (!fuse(run))
          return false;
        run.clear();
      } else {
        run.push_back(op);
      }
    }
    return fuse(run).has_value();
  }

  // The state made again from the starting bits, with the whole log.
  void replay() {
    if (handle >= 0) {
      engine({{"op", "free"}, {"handle", handle}});
      handle = -1;
    }
    handle = engine({{"op", "qubits"}, {"n_qubits", nQubits}, {"bits", bits},
                     {"conserve", charged}})["handle"].get<long long>();
    bondReached = 1;
    truncationSum = 0;
    std::vector<Op> run;
    for (const Op &op : log) {
      if (op.kind == Op::measure || op.kind == Op::reset) {
        applyRunAs(run);
        run.clear();
        project(op.q, op.bit, probabilityOfOne(op.q));
        if (op.kind == Op::reset && op.bit)
          applyMatrix(op.q, 0, 1);   // |1> -> |0>
      } else {
        run.push_back(op);
      }
    }
    applyRunAs(run);
    stale = false;
  }

  void movePendingToLog() {
    log.insert(log.end(), pending.begin(), pending.end());
    pending.clear();
  }

  [[noreturn]] static void refuseUncharged() {
    throw std::runtime_error("quantx: QUANTX_CONSERVE=1, and the kernel's gates do not fuse "
                             "into ones that conserve the number of |1>s");
  }

  // The state brought up to date with every gate given so far. A state that carries the
  // charge keeps it while the new gates fuse; once they do not, the log is replayed without it,
  // and the kernel runs on from there on the same kernels with no charge.
  void bringUpToDate() {
    if (nQubits == 0)
      throw std::runtime_error("quantx: the kernel has no qubits");
    if (handle >= 0 && !stale) {
      if (pending.empty())
        return;
      if (!charged) {
        applyRun(fuseBySize(pending));
        movePendingToLog();
        return;
      }
      if (auto fused = fuse(pending)) {
        applyRun(fuseBySize(*fused));
        movePendingToLog();
        return;
      }
      if (conserve == Conserve::always)
        refuseUncharged();
      movePendingToLog();
      charged = false;
      replay();
      return;
    }
    // A first state, or one that qubits were added to: made from the whole log, with the
    // charge if every run of it fuses.
    movePendingToLog();
    if (handle < 0 || charged) {
      const bool can = conserve != Conserve::never && fusesThroughout(log);
      if (!can && conserve == Conserve::always)
        refuseUncharged();
      charged = can;
    }
    replay();
  }

  // Every kernel starts from the defaults: the simulator outlives a kernel, and a variable
  // unset since the last one must not keep its value.
  void readOptions() {
    maxBond = 0;
    cutoff = 1e-14;
    conserve = Conserve::automatic;
    direct = true;
    verbose = false;
    if (const char *v = env("QUANTX_MAX_BOND")) {
      maxBond = std::stoi(v);
      if (maxBond < 1)
        throw std::runtime_error("quantx: QUANTX_MAX_BOND must be a positive integer");
    }
    if (const char *v = env("QUANTX_CUTOFF")) {
      cutoff = std::stod(v);
      if (!(cutoff >= 0))
        throw std::runtime_error("quantx: QUANTX_CUTOFF must be >= 0");
    }
    if (const char *v = env("QUANTX_CONSERVE")) {
      const std::string s = v;
      if (s == "auto")
        conserve = Conserve::automatic;
      else if (s == "1" || s == "true")
        conserve = Conserve::always;
      else if (s == "0" || s == "false")
        conserve = Conserve::never;
      else
        throw std::runtime_error("quantx: QUANTX_CONSERVE must be auto, 1 or 0");
    }
    if (const char *v = env("QUANTX_SVD")) {
      const std::string s = v;
      if (s != "direct" && s != "gram")
        throw std::runtime_error("quantx: QUANTX_SVD must be direct or gram");
      direct = s == "direct";
    }
    if (const char *v = env("QUANTX_VERBOSE"))
      verbose = std::string(v) != "0";
  }

  // A failure leaves no state behind: CUDA-Q does not deallocate a kernel that threw while it
  // was being finalised, and the next kernel would start on this one's qubits and options.
  template <typename F>
  auto guarded(F &&f) -> decltype(f()) {
    try {
      return f();
    } catch (...) {
      deallocateState();
      throw;
    }
  }

protected:
  // The base class sizes a state vector, and refuses past 63 qubits; an MPS has no such size.
  std::size_t calculateStateDim(const std::size_t numQubits) override { return numQubits; }

  void addQubitToState() override { addQubitsToState(1); }

  void addQubitsToState(std::size_t count, const void *state = nullptr) override {
    guarded([&] {
      if (state != nullptr)
        throw std::runtime_error("quantx: a kernel cannot start from a state vector on this "
                                 "target; prepare the state with gates");
      if (count == 0)
        return;
      if (nQubits == 0)
        readOptions();   // per kernel, so a change between kernels takes effect
      nQubits += count;
      bits.resize(nQubits, 0);
      known.resize(nQubits, 0);
      if (handle >= 0)
        stale = true;   // the new qubits start in |0>, after everything the log holds
    });
  }

  void addQubitsToState(const cudaq::SimulationState &) override {
    throw std::runtime_error("quantx: a kernel cannot start from a state on this target; "
                             "prepare the state with gates");
  }

  void deallocateStateImpl() override {
    clearCircuit();
    nQubits = 0;
    bits.clear();
    known.clear();
  }

  void applyGate(const GateApplicationTask &task) override {
    guarded([&] { bufferGate(task); });
  }

  void bufferGate(const GateApplicationTask &task) {
    for (Gate &g : gatesOf(task)) {
      const std::size_t q = g.q[0];
      if (isX(g) && known[q] >= 0) {
        if (handle < 0 && log.empty())
          bits[q] ^= 1;   // before the state exists, an X moves the starting bitstring
        else
          pending.push_back(Op{Op::flip, {}, q, known[q]});
        known[q] ^= 1;
        continue;
      }
      for (auto t : g.q)
        known[t] = -1;
      Op op;
      op.g = std::move(g);
      pending.push_back(std::move(op));
    }
  }

  // An outcome drawn from the state, and its projector applied.
  int drawAndProject(std::size_t q) {
    const double p1 = probabilityOfOne(q);
    const int one = std::uniform_real_distribution<double>(0.0, 1.0)(rng) < p1 ? 1 : 0;
    project(q, one, p1);
    return one;
  }

  bool measureQubit(const std::size_t qubitIdx) override {
    return guarded([&] {
      bringUpToDate();
      const int one = drawAndProject(qubitIdx);
      log.push_back(Op{Op::measure, {}, qubitIdx, one});
      known[qubitIdx] = one;
      return one == 1;
    });
  }

  QubitOrdering getQubitOrdering() const override { return QubitOrdering::msb; }

public:
  QuantXSimulator() { summaryData.name = name(); }

  void setRandomSeed(std::size_t seed) override { rng.seed(seed); }

  void setToZeroState() override {
    flushPendingQubits();
    clearCircuit();
  }

  void resetQubit(const std::size_t qubitIdx) override {
    guarded([&] {
      flushGateQueue();
      flushAnySamplingTasks();
      bringUpToDate();
      const int one = drawAndProject(qubitIdx);
      if (one)
        applyMatrix(qubitIdx, 0, 1);   // |1> -> |0>
      log.push_back(Op{Op::reset, {}, qubitIdx, one});
      known[qubitIdx] = 0;
    });
  }

  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits, const int shots,
                                bool includeSequentialData = true) override {
    return guarded([&] { return sampleImpl(qubits, shots, includeSequentialData); });
  }

  cudaq::ExecutionResult sampleImpl(const std::vector<std::size_t> &qubits, const int shots,
                                    bool includeSequentialData) {
    bringUpToDate();
    if (shots < 1) {
      // The exact <Z...Z> on the qubits, which is how a basis-changed term is evaluated.
      json term = json::array();
      std::vector<std::size_t> sorted = qubits;
      std::sort(sorted.begin(), sorted.end());
      sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
      for (auto q : sorted)
        term.push_back(json::array({q, "Z"}));
      const json r = engine({{"op", "pauli"}, {"handle", handle},
                             {"terms", json::array({term})}});
      return cudaq::ExecutionResult{{}, r["values"][0].get<double>()};
    }
    std::vector<unsigned char> samples(std::size_t(shots) * nQubits);
    const unsigned long long seed =
        std::uniform_int_distribution<unsigned long long>()(rng);
    check(qx_mps_sample(handle, (unsigned long long)shots, seed, samples.data()));
    std::map<std::string, std::size_t> counts;
    std::string bitstring(qubits.size(), '0');
    for (std::size_t s = 0; s < std::size_t(shots); ++s) {
      for (std::size_t i = 0; i < qubits.size(); ++i)
        bitstring[i] = samples[s * nQubits + qubits[i]] ? '1' : '0';
      ++counts[bitstring];
    }
    cudaq::ExecutionResult result;
    double expectation = 0;
    for (auto &[b, count] : counts) {
      const double p = count / double(shots);
      expectation += cudaq::sample_result::has_even_parity(b) ? p : -p;
      if (includeSequentialData)
        result.appendResult(b, count);
      else
        result.counts[b] += count;
    }
    result.expectationValue = expectation;
    return result;
  }

  bool canHandleObserve() override {
    // Shots: the base class samples each term in its own basis. Exactly (shots 0, as Python
    // sets them, or -1): the engine's Pauli strings, all terms at once, unless
    // CUDAQ_OBSERVE_FROM_SAMPLING=1 asks otherwise. The basis changes would also cost the
    // charge: an H on a charged state is replayed without it.
    auto *ctx = cudaq::getExecutionContext();
    if (ctx && ctx->shots != 0 && ctx->shots != static_cast<std::size_t>(-1))
      return false;
    return !shouldObserveFromSampling(false);
  }

  cudaq::observe_result observe(const cudaq::spin_op &op) override {
    return guarded([&] { return observeImpl(op); });
  }

  cudaq::observe_result observeImpl(const cudaq::spin_op &op) {
    flushGateQueue();
    bringUpToDate();
    json terms = json::array();
    std::vector<std::pair<std::string, double>> ids;   // each term's id and real coefficient
    double sum = 0;
    for (const auto &term : op) {
      const double coeff = term.evaluate_coefficient().real();
      if (term.is_identity()) {
        sum += coeff;
        continue;
      }
      std::vector<std::pair<std::size_t, char>> letters;
      for (const auto &p : term) {
        const auto pauli = p.as_pauli();
        if (pauli == cudaq::pauli::I)
          continue;
        letters.emplace_back(p.target(), pauli == cudaq::pauli::X   ? 'X'
                                         : pauli == cudaq::pauli::Y ? 'Y'
                                                                    : 'Z');
      }
      std::sort(letters.begin(), letters.end());
      json string = json::array();
      for (auto &[q, c] : letters)
        string.push_back(json::array({q, std::string(1, c)}));
      terms.push_back(string);
      ids.emplace_back(term.get_term_id(), coeff);
    }
    std::vector<cudaq::ExecutionResult> results;
    if (!terms.empty()) {
      const json values = engine({{"op", "pauli"}, {"handle", handle}, {"terms", terms}})["values"];
      for (std::size_t k = 0; k < ids.size(); ++k) {
        const double v = values[k].get<double>();
        sum += ids[k].second * v;
        results.emplace_back(cudaq::CountsDictionary{}, ids[k].first, v);
      }
    }
    return cudaq::observe_result(sum, op, cudaq::sample_result(sum, results));
  }

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    return guarded([&] { return stateImpl(); });
  }

  std::unique_ptr<cudaq::SimulationState> stateImpl() {
    flushGateQueue();
    if (nQubits > kMaxDenseQubits)
      throw std::runtime_error("quantx: get_state reads the amplitudes, 2^" +
                               std::to_string(nQubits) + " of them; this target does it up "
                               "to " + std::to_string(kMaxDenseQubits) +
                               " qubits (observe and sample work at any width)");
    bringUpToDate();
    const std::size_t dim = std::size_t(1) << nQubits;
    std::vector<cplx> engineOrder(dim);
    check(qx_mps_amplitudes(handle, reinterpret_cast<double *>(engineOrder.data()), dim));
    // The engine puts qubit 0 on the high bit, CUDA-Q on the low one.
    std::vector<cplx> amp(dim);
    double norm2 = 0;
    for (std::size_t x = 0; x < dim; ++x) {
      std::size_t y = 0;
      for (std::size_t k = 0; k < nQubits; ++k)
        y |= ((x >> (nQubits - 1 - k)) & 1) << k;
      amp[y] = engineOrder[x];
      norm2 += std::norm(engineOrder[x]);
    }
    const double scale = norm2 > 0 ? 1.0 / std::sqrt(norm2) : 1.0;   // truncation's lost weight
    for (auto &z : amp)
      z *= scale;
    return std::make_unique<QuantXState>(std::move(amp));
  }

  std::unique_ptr<cudaq::SimulationState>
  createStateFromData(const cudaq::state_data &data) override {
    return QuantXState({}).createFromData(data);
  }

  std::string name() const override { return "quantx"; }
  NVQIR_SIMULATOR_CLONE_IMPL(QuantXSimulator)
};

} // namespace nvqir

NVQIR_REGISTER_SIMULATOR(nvqir::QuantXSimulator, quantx)
