// src/tenfold.cpp -- M34: 10-Fold Way Topological Classification Engine
//
// Implements the Altland-Zirnbauer (AZ) symmetry classification of free-fermion
// Hamiltonians and numerically computes topological invariants for three
// canonical models:
//
//   SSH chain   (class BDI, d=1): winding number W in Z
//   QWZ model   (class A,   d=2): Chern number C in Z  (FHS algorithm)
//   Kitaev chain(class D,   d=1): Z2 invariant Q = sign(Pf(0)*Pf(pi))
//
// AZ Periodic Table (relevant portion, d=1,2,3):
//   Class  | d=1 | d=2 | d=3 |  T^2 | C^2 | S
//   -------|-----|-----|-----|------|-----|---
//   A      |  0  |  Z  |  0  |   0  |  0  | 0
//   AI     |  0  |  0  |  0  |  +1  |  0  | 0
//   AII    |  0  | Z2  | Z2  |  -1  |  0  | 0
//   AIII   |  Z  |  0  |  Z  |   0  |  0  | 1
//   BDI    |  Z  |  0  |  0  |  +1  | +1  | 1
//   CII    | 2Z  |  0  |  Z  |  -1  | -1  | 1
//   D      | Z2  |  Z  |  0  |   0  | +1  | 0
//   DIII   | Z2  | Z2  |  Z  |  -1  | +1  | 1
//   C      |  0  |  Z  |  0  |   0  | -1  | 0
//   CI     |  0  |  0  |  Z  |  +1  | -1  | 1
//
// Scenarios:
//   classify      -- look up invariant type from AZ table (no numerics)
//   invariant     -- compute numerical invariant for a model at given params
//   phase_diagram -- sweep a parameter and show invariant vs parameter

#include "qx/engine.hpp"
#include "qx/types.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace qx {
namespace {

using cx = std::complex<double>;

// ─────────────────────────────────────────────────────────────────────────────
// A.  Altland-Zirnbauer periodic table lookup
// ─────────────────────────────────────────────────────────────────────────────

struct AZEntry {
    const char* az_class;
    int          dim;
    const char*  inv_type;   // "Z", "Z2", "2Z", "0"
    int          T_sq;       // T^2:  0=absent, +1, -1
    int          C_sq;       // C^2:  0=absent, +1, -1
    int          S;          // S:    0=absent, 1=present
};

static const AZEntry AZ_TABLE[] = {
    // class   dim  inv   T   C  S
    {"A",    1,  "0",   0,  0, 0},
    {"A",    2,  "Z",   0,  0, 0},
    {"A",    3,  "0",   0,  0, 0},
    {"AI",   1,  "0",  +1,  0, 0},
    {"AI",   2,  "0",  +1,  0, 0},
    {"AI",   3,  "0",  +1,  0, 0},
    {"AII",  1,  "0",  -1,  0, 0},
    {"AII",  2,  "Z2", -1,  0, 0},
    {"AII",  3,  "Z2", -1,  0, 0},
    {"AIII", 1,  "Z",   0,  0, 1},
    {"AIII", 2,  "0",   0,  0, 1},
    {"AIII", 3,  "Z",   0,  0, 1},
    {"BDI",  1,  "Z",  +1, +1, 1},
    {"BDI",  2,  "0",  +1, +1, 1},
    {"BDI",  3,  "0",  +1, +1, 1},
    {"CII",  1,  "2Z", -1, -1, 1},
    {"CII",  2,  "0",  -1, -1, 1},
    {"CII",  3,  "Z",  -1, -1, 1},
    {"D",    1,  "Z2",  0, +1, 0},
    {"D",    2,  "Z",   0, +1, 0},
    {"D",    3,  "0",   0, +1, 0},
    {"DIII", 1,  "Z2", -1, +1, 1},
    {"DIII", 2,  "Z2", -1, +1, 1},
    {"DIII", 3,  "Z",  -1, +1, 1},
    {"C",    1,  "0",   0, -1, 0},
    {"C",    2,  "Z",   0, -1, 0},
    {"C",    3,  "0",   0, -1, 0},
    {"CI",   1,  "0",  +1, -1, 1},
    {"CI",   2,  "0",  +1, -1, 1},
    {"CI",   3,  "Z",  +1, -1, 1},
};
static constexpr int AZ_TABLE_SIZE = static_cast<int>(sizeof(AZ_TABLE) / sizeof(AZ_TABLE[0]));

static const AZEntry* lookup_az(const std::string& cls, int dim) {
    for (int i = 0; i < AZ_TABLE_SIZE; ++i) {
        if (AZ_TABLE[i].az_class == cls && AZ_TABLE[i].dim == dim)
            return &AZ_TABLE[i];
    }
    return nullptr;
}

static std::string lookup_invariant_type(const std::string& cls, int dim) {
    const AZEntry* e = lookup_az(cls, dim);
    if (!e) return "?";
    return e->inv_type;
}

// Return all valid class names for input validation.
static bool valid_az_class(const std::string& cls) {
    static const char* CLASSES[] = {
        "A","AI","AII","AIII","BDI","CII","D","DIII","C","CI"
    };
    for (auto* c : CLASSES) if (cls == c) return true;
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// B.  SSH winding number  (class BDI / AIII, d=1)
//
//   H(k) = d_x(k) sigma_x + d_y(k) sigma_y
//   d_x(k) = t1 + t2*cos(k),   d_y(k) = t2*sin(k)
//
//   W = (1/2pi) * argument accumulated by d(k) = d_x + i*d_y as k: 0 -> 2pi
//   Computed via Wilson-loop product  W_loop = product_k  d*(k)/|d(k)| * d(k+dk)/|d(k+dk)|
//   W = round( Im log W_loop / (-pi) )  (sign matches CW vs CCW convention)
// ─────────────────────────────────────────────────────────────────────────────

static int ssh_winding(double t1, double t2, int n_k) {
    cx W_loop{1.0, 0.0};
    for (int m = 0; m < n_k; ++m) {
        double k0 = 2.0 * M_PI * m       / n_k;
        double k1 = 2.0 * M_PI * (m + 1) / n_k;
        cx d0{t1 + t2 * std::cos(k0), t2 * std::sin(k0)};
        cx d1{t1 + t2 * std::cos(k1), t2 * std::sin(k1)};
        double r0 = std::abs(d0), r1 = std::abs(d1);
        if (r0 < 1e-13 || r1 < 1e-13) return 0;  // gap closes -> ill-defined
        W_loop *= (std::conj(d0) / r0) * (d1 / r1);
    }
    // Normalise accumulated phase to unit circle before taking arg
    double phase = std::arg(W_loop);
    return static_cast<int>(std::round(phase / (-M_PI)));
}

// Build SSH band dispersion E±(k) for visualisation.
static void ssh_bands(double t1, double t2, int n_k,
                      std::vector<double>& k_out,
                      std::vector<double>& e_plus,
                      std::vector<double>& e_minus) {
    k_out.resize(n_k);
    e_plus.resize(n_k);
    e_minus.resize(n_k);
    for (int m = 0; m < n_k; ++m) {
        double k = 2.0 * M_PI * m / (n_k - 1);
        double dx = t1 + t2 * std::cos(k);
        double dy = t2 * std::sin(k);
        double E  = std::sqrt(dx * dx + dy * dy);
        k_out[m]  = k;
        e_plus[m] = +E;
        e_minus[m]= -E;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// C.  QWZ Chern number  (class A, d=2)
//     Fukui-Hatsugai-Suzuki lattice gauge field method.
//
//   H(k) = t*sin(kx)*sx + t*sin(ky)*sy + (m + t*(cos(kx)+cos(ky)))*sz
//   Returns rounded Chern integer.
// ─────────────────────────────────────────────────────────────────────────────

// Lower-band eigenvector of the 2x2 QWZ Hamiltonian.
static std::array<cx,2> qwz_evec(double kx, double ky, double m, double t) {
    double dx = t * std::sin(kx);
    double dy = t * std::sin(ky);
    double dz = m + t * (std::cos(kx) + std::cos(ky));
    double r  = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (r < 1e-15) return {cx{1.0,0.0}, cx{0.0,0.0}};
    // Lower eigenvalue: E = -r;  state from (H+r*I)|v>=0
    double norm = std::sqrt(2.0 * r * (r + dz));
    if (norm < 1e-15) return {cx{0.0,0.0}, cx{1.0,0.0}};
    return {cx{(dx + cx{0.0,1.0}*dy) / norm},
            cx{-(r + dz) / norm}};
}

// Hermitian inner product a†·b
static cx inner(const std::array<cx,2>& a, const std::array<cx,2>& b) {
    return std::conj(a[0])*b[0] + std::conj(a[1])*b[1];
}

static int qwz_chern(int n_k, double m, double t) {
    double dk = 2.0 * M_PI / n_k;
    double chern = 0.0;
    for (int ix = 0; ix < n_k; ++ix) {
        for (int iy = 0; iy < n_k; ++iy) {
            double kx = -M_PI + ix * dk;
            double ky = -M_PI + iy * dk;
            auto u00 = qwz_evec(kx,      ky,      m, t);
            auto u10 = qwz_evec(kx + dk, ky,      m, t);
            auto u11 = qwz_evec(kx + dk, ky + dk, m, t);
            auto u01 = qwz_evec(kx,      ky + dk, m, t);
            cx F = inner(u00, u10) * inner(u10, u11) *
                   inner(u11, u01) * inner(u01, u00);
            double r = std::abs(F);
            if (r > 1e-15) F /= r;
            chern += std::arg(F);
        }
    }
    return static_cast<int>(std::round(chern / (2.0 * M_PI)));
}

// ─────────────────────────────────────────────────────────────────────────────
// D.  Kitaev Z2 invariant  (class D, d=1)
//
//   Q = sign( Pf(H_BdG(k=0)) * Pf(H_BdG(k=pi)) )
//   For the 1D Kitaev BdG Hamiltonian in the chiral/Majorana basis:
//     d(k=0)  = -(mu + 2t)
//     d(k=pi) = -(mu - 2t)
//   Pf of 2x2 skew-sym = off-diagonal element = d(k).
//   Q = sign( d(0)*d(pi) );  Q=-1 -> topological (invariant_z2=1).
// ─────────────────────────────────────────────────────────────────────────────

static int kitaev_z2(double t_hop, double mu) {
    double pf0  = -(mu + 2.0 * t_hop);
    double pfpi = -(mu - 2.0 * t_hop);
    // Handle exact gap-closing points
    if (std::abs(pf0) < 1e-14 || std::abs(pfpi) < 1e-14) return 0;
    double Q = (pf0 > 0 ? 1.0 : -1.0) * (pfpi > 0 ? 1.0 : -1.0);
    return (Q < 0) ? 1 : 0;  // 1 = topological, 0 = trivial
}

// ─────────────────────────────────────────────────────────────────────────────
// E.  Phase label helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::string phase_label(const std::string& inv_type, double val, int z2) {
    if (inv_type == "0") return "Trivial";
    if (inv_type == "Z2") return z2 ? "Topological (Z2=1)" : "Trivial (Z2=0)";
    // Z or 2Z
    int w = static_cast<int>(std::round(val));
    if (w == 0) return "Trivial (W=0)";
    return "Topological (W=" + std::to_string(w) + ")";
}

// ─────────────────────────────────────────────────────────────────────────────
// F.  Scenario implementations
// ─────────────────────────────────────────────────────────────────────────────

static TenfoldResult run_classify(const TenfoldOptions& opts) {
    TenfoldResult res;
    res.az_class       = opts.az_class;
    res.dimension      = opts.dimension;
    res.scenario       = opts.scenario;
    res.model          = opts.model;
    res.invariant_type = lookup_invariant_type(opts.az_class, opts.dimension);
    res.phase_label    = (res.invariant_type == "0")
                            ? "Trivial — no topological invariant"
                            : "Potentially topological (" + res.invariant_type + ")";
    res.message = "AZ class " + opts.az_class + " in d=" +
                  std::to_string(opts.dimension) + " has invariant type " +
                  res.invariant_type;

    // Echo symmetry indicators for this class
    const AZEntry* e = lookup_az(opts.az_class, opts.dimension);
    if (e) {
        res.message += "  (T^2=" + std::to_string(e->T_sq) +
                       ", C^2=" + std::to_string(e->C_sq) +
                       ", S="   + std::to_string(e->S) + ")";
    }
    res.ok = true;
    return res;
}

static TenfoldResult run_invariant(const TenfoldOptions& opts) {
    TenfoldResult res;
    res.az_class       = opts.az_class;
    res.dimension      = opts.dimension;
    res.scenario       = opts.scenario;
    res.model          = opts.model;
    res.invariant_type = lookup_invariant_type(opts.az_class, opts.dimension);

    if (opts.model == "ssh") {
        int W = ssh_winding(opts.t1, opts.t2, opts.n_k);
        res.invariant_value = static_cast<double>(W);
        res.invariant_z2    = 0;
        res.phase_label     = phase_label(res.invariant_type, res.invariant_value, 0);
        // Also attach band data
        ssh_bands(opts.t1, opts.t2, opts.n_k,
                  res.k_bands, res.e_plus, res.e_minus);
    } else if (opts.model == "qwz") {
        int C = qwz_chern(opts.n_k, opts.m, opts.t);
        res.invariant_value = static_cast<double>(C);
        res.invariant_z2    = 0;
        res.phase_label     = phase_label(res.invariant_type, res.invariant_value, 0);
    } else if (opts.model == "kitaev") {
        int q = kitaev_z2(opts.t_hop, opts.mu);
        res.invariant_value = static_cast<double>(q);
        res.invariant_z2    = q;
        res.phase_label     = phase_label(res.invariant_type, res.invariant_value, q);
    } else {
        throw std::runtime_error("tenfold: unknown model '" + opts.model +
                                 "' — use ssh | qwz | kitaev");
    }

    res.message = "model=" + opts.model + ", invariant=" +
                  res.invariant_type + ", value=" +
                  std::to_string(static_cast<int>(std::round(res.invariant_value)));
    res.ok = true;
    return res;
}

static TenfoldResult run_phase_diagram(const TenfoldOptions& opts) {
    TenfoldResult res;
    res.az_class       = opts.az_class;
    res.dimension      = opts.dimension;
    res.scenario       = opts.scenario;
    res.model          = opts.model;
    res.invariant_type = lookup_invariant_type(opts.az_class, opts.dimension);

    int n = opts.n_sweep;
    if (n < 2) throw std::runtime_error("tenfold: n_sweep must be >= 2");

    res.sweep_vals.resize(n);
    res.sweep_invariant.resize(n);
    res.sweep_z2.resize(n);
    res.sweep_phase.resize(n);

    for (int k = 0; k < n; ++k) {
        double p = opts.param_min +
                   (opts.param_max - opts.param_min) * k / (n - 1);
        res.sweep_vals[k] = p;

        // Build a local options copy with the swept parameter substituted
        TenfoldOptions lo = opts;
        lo.scenario = "invariant";
        if      (opts.sweep_param == "m")  lo.m   = p;
        else if (opts.sweep_param == "t1") lo.t1  = p;
        else if (opts.sweep_param == "t2") lo.t2  = p;
        else if (opts.sweep_param == "mu") lo.mu  = p;
        else
            throw std::runtime_error("tenfold: sweep_param must be m|t1|t2|mu");

        TenfoldResult pt = run_invariant(lo);
        res.sweep_invariant[k] = pt.invariant_value;
        res.sweep_z2[k]        = pt.invariant_z2;
        res.sweep_phase[k]     = pt.phase_label;
    }

    // Use the last point's single-value fields as summary
    if (n > 0) {
        int mid = n / 2;
        res.invariant_value = res.sweep_invariant[mid];
        res.invariant_z2    = res.sweep_z2[mid];
        res.phase_label     = res.sweep_phase[mid];
    }

    res.message = "phase diagram: " + opts.model + " sweeping " +
                  opts.sweep_param + " in [" +
                  std::to_string(opts.param_min) + ", " +
                  std::to_string(opts.param_max) + "]";
    res.ok = true;
    return res;
}

}  // namespace (anonymous)

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

TenfoldResult run_tenfold(const TenfoldOptions& opts) {
    auto t0 = std::chrono::steady_clock::now();

    // Validate inputs
    if (!valid_az_class(opts.az_class))
        throw std::runtime_error(
            "tenfold: unknown AZ class '" + opts.az_class +
            "' — use A|AI|AII|AIII|BDI|CII|D|DIII|C|CI");
    if (opts.dimension < 1 || opts.dimension > 3)
        throw std::runtime_error("tenfold: dimension must be 1, 2, or 3");
    if (opts.n_k < 4)
        throw std::runtime_error("tenfold: n_k must be >= 4");

    TenfoldResult res;
    if (opts.scenario == "classify")
        res = run_classify(opts);
    else if (opts.scenario == "invariant")
        res = run_invariant(opts);
    else if (opts.scenario == "phase_diagram")
        res = run_phase_diagram(opts);
    else
        throw std::runtime_error(
            "tenfold: unknown scenario '" + opts.scenario +
            "' — use classify | invariant | phase_diagram");

    auto t1 = std::chrono::steady_clock::now();
    res.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

}  // namespace qx
