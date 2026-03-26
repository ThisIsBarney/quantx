#include "qx/backends.hpp"
#include "qx/fft_provider.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qx {
namespace {

using Complex = std::complex<double>;

// Checkpoint payload cached before initialize() applies it.
struct PendingCheckpoint2D {
  bool valid{false};
  std::uint32_t nx{0};
  std::uint32_t ny{0};
  double dx{0.0};
  double dy{0.0};
  double dt{0.0};
  std::uint32_t total_steps{0};
  std::uint32_t completed_steps{0};
  std::string boundary;
  std::vector<Complex> state;
};

// Convert boundary token into a compact checkpoint code.
std::uint32_t boundary_to_code(const std::string& boundary) {
  if (boundary == "absorbing") {
    return 1U;
  }
  if (boundary == "dirichlet") {
    return 2U;
  }
  return 0U;
}

// Convert checkpoint code back into a boundary token.
std::string boundary_from_code(std::uint32_t code) {
  if (code == 1U) {
    return "absorbing";
  }
  if (code == 2U) {
    return "dirichlet";
  }
  return "periodic";
}

// Check whether a checkpoint header matches the 2D continuous magic prefix.
bool is_checkpoint_magic(const char magic[4]) {
  return magic[0] == 'Q' && magic[1] == 'X' && magic[2] == '2' && magic[3] == 'D';
}

class Continuous2DBackend final : public IBackend {
 public:
  std::string name() const override { return "continuous2d"; }

  BackendCapabilities capabilities() const override {
    BackendCapabilities c;
    c.kind = BackendKind::Continuous2D;
    c.supports_gpu = false;
    c.supports_mpi = false;
    c.supports_noisy = false;
    c.max_qubits_recommended = 0;
    c.equation_scope = "Schrodinger,KleinGordon";
    return c;
  }

  void initialize(const QuantumProblem& problem, const RunOptions& options) override {
    if (problem.kind != ProblemKind::ContinuousSpace2D) {
      throw std::runtime_error("continuous2d backend requires ProblemKind::ContinuousSpace2D");
    }

    const auto& c = problem.continuous_2d;
    if (c.nx == 0 || c.ny == 0) {
      throw std::runtime_error("continuous2d grid resolution must be positive");
    }
    if (c.dx <= 0.0 || c.dy <= 0.0 || c.dt <= 0.0 || c.steps == 0) {
      throw std::runtime_error("continuous2d dx/dy/dt/steps must be positive");
    }
    if (c.boundary != "periodic" && c.boundary != "absorbing" && c.boundary != "dirichlet") {
      throw std::runtime_error("continuous2d backend supports boundary in {periodic,absorbing,dirichlet}");
    }

    nx_ = c.nx;
    ny_ = c.ny;
    dx_ = c.dx;
    dy_ = c.dy;
    dt_ = c.dt;
    steps_ = c.steps;
    boundary_ = c.boundary;
    dispersion_ = c.dispersion;
    mass_ = (c.mass > 0.0) ? c.mass : 1.0;
    speed_of_light_ = (c.speed_of_light > 0.0) ? c.speed_of_light : 1.0;
    equation_ = problem.equation;
    potential_kind_ = c.potential_kind;
    driving_config_ = c.driving;
    has_driving_ = (c.driving.type != "none" && c.driving.amplitude != 0.0);
    completed_steps_ = 0;
    export_frame_payloads_ = options.export_continuous_frames;
    export_frame_stride_ = std::max<std::uint32_t>(1U, options.export_frame_stride);
    export_wigner_ = options.export_wigner;
    wigner_frames_.clear();
    frames_.clear();
    absorbing_mask_.assign(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_), 1.0);

    fft_provider_ = nullptr;
    if (options.use_gpu) {
      try {
        fft_provider_ = create_fft_provider(FFTProviderKind::CuFFT);
      } catch (const std::exception&) {
        fft_provider_ = create_fft_provider(FFTProviderKind::Eigen);
      }
    } else {
      fft_provider_ = create_fft_provider(FFTProviderKind::Auto);
    }

    const std::size_t n = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_);
    state_.assign(n, Complex{0.0, 0.0});
    kbuf_.assign(n, Complex{0.0, 0.0});
    kpibuf_.assign(n, Complex{0.0, 0.0});
    pi_state_.assign(n, Complex{0.0, 0.0});
    dirac_s1_.assign(n, Complex{0.0, 0.0});
    potential_.assign(n, 0.0);
    potential_phase_.assign(n, Complex{1.0, 0.0});
    potential_half_phase_.assign(n, Complex{1.0, 0.0});
    kinetic_half_phase_.assign(n, Complex{1.0, 0.0});
    kg_pot_c_.assign(n, 1.0);
    kg_pot_sc_.assign(n, 0.0);
    kg_pot_os_.assign(n, 0.0);
    kg_kin_c_.assign(n, 1.0);
    kg_kin_sc_.assign(n, 0.0);
    kg_kin_os_.assign(n, 0.0);

    if (!c.potential_grid.empty()) {
      if (c.potential_grid.size() != n) {
        throw std::runtime_error("continuous2d potential_grid size must equal nx*ny");
      }
      potential_ = c.potential_grid;
    } else {
      build_analytic_potential(c.potential_kind, c.potential_params);
    }

    // Build spatial driving profile (precomputed once at initialization).
    driving_spatial_.assign(n, 1.0);
    if (has_driving_) {
      const std::string& sp = driving_config_.spatial_profile;
      if (sp == "dipole_x") {
        for (std::uint32_t y = 0; y < ny_; ++y)
          for (std::uint32_t x = 0; x < nx_; ++x)
            driving_spatial_[idx(x, y)] = x_coord(x);
      } else if (sp == "dipole_y") {
        for (std::uint32_t y = 0; y < ny_; ++y)
          for (std::uint32_t x = 0; x < nx_; ++x)
            driving_spatial_[idx(x, y)] = y_coord(y);
      }
      // "uniform" or default: driving_spatial_ stays 1.0
    }

    initialize_wavepacket(c.initial_state, c.potential_params);
    precompute_phases();
    if (equation_ == EquationType::KleinGordon) {
      precompute_kg_phases();
      initialize_pi_state(c.initial_state);
    }
    // dirac_s1_ is already zero-initialized; ψ₀ = state_ gets the Gaussian wavepacket.
    precompute_boundary_mask(c.potential_params);

    const bool restored = apply_pending_checkpoint();
    if (!restored && boundary_ == "dirichlet") {
      enforce_dirichlet_boundary();
    }

    update_observables();
    record_frame();
    record_wigner_frame();
  }

  void evolve(const QuantumProblem&, const RunOptions&) override {
    if (equation_ == EquationType::KleinGordon) {
      evolve_kg();
      return;
    }
    if (equation_ == EquationType::Dirac) {
      evolve_dirac();
      return;
    }
    for (std::uint32_t s = completed_steps_; s < steps_; ++s) {
      const double t = static_cast<double>(s) * dt_;
      apply_half_kinetic();
      if (has_driving_) {
        apply_potential_driven(t + 0.5 * dt_);
      } else {
        apply_potential();
      }
      apply_half_kinetic();
      if (boundary_ == "absorbing") {
        apply_absorbing_boundary();
      } else if (boundary_ == "dirichlet") {
        enforce_dirichlet_boundary();
      }
      if (boundary_ != "absorbing") {
        normalize_state();
      }
      completed_steps_ = s + 1;
      update_observables();
      record_frame();
      record_wigner_frame();
    }
  }

  void evolve_kg() {
    for (std::uint32_t s = completed_steps_; s < steps_; ++s) {
      const double t = static_cast<double>(s) * dt_;
      if (has_driving_) {
        recompute_kg_potential_phases(t);
      }
      apply_half_potential_kg();
      apply_full_kinetic_kg();
      apply_half_potential_kg();
      if (boundary_ == "absorbing") {
        apply_absorbing_boundary();
        for (std::size_t i = 0; i < pi_state_.size(); ++i) {
          pi_state_[i] *= absorbing_mask_[i];
        }
      } else if (boundary_ == "dirichlet") {
        enforce_dirichlet_boundary();
        std::fill(pi_state_.begin(), pi_state_.end(), Complex{0.0, 0.0});
      }
      completed_steps_ = s + 1;
      update_observables();
      record_frame();
      record_wigner_frame();
    }
  }

  std::map<std::string, double> measure(const QuantumProblem&) override { return observables_; }

  void checkpoint(const std::string& path) override {
    std::ofstream os(path + "/continuous2d.chk", std::ios::binary);
    if (!os.good()) {
      return;
    }
    const char magic[4] = {'Q', 'X', '2', 'D'};
    const std::uint32_t version = 2U;
    const std::uint32_t boundary_code = boundary_to_code(boundary_);
    os.write(magic, sizeof(magic));
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));
    os.write(reinterpret_cast<const char*>(&nx_), sizeof(nx_));
    os.write(reinterpret_cast<const char*>(&ny_), sizeof(ny_));
    os.write(reinterpret_cast<const char*>(&dx_), sizeof(dx_));
    os.write(reinterpret_cast<const char*>(&dy_), sizeof(dy_));
    os.write(reinterpret_cast<const char*>(&dt_), sizeof(dt_));
    os.write(reinterpret_cast<const char*>(&steps_), sizeof(steps_));
    os.write(reinterpret_cast<const char*>(&completed_steps_), sizeof(completed_steps_));
    os.write(reinterpret_cast<const char*>(&boundary_code), sizeof(boundary_code));
    const std::size_t size = state_.size();
    os.write(reinterpret_cast<const char*>(&size), sizeof(size));
    if (size > 0) {
      os.write(reinterpret_cast<const char*>(state_.data()),
               static_cast<std::streamsize>(size * sizeof(std::complex<double>)));
    }
  }

  void restore(const std::string& path) override {
    std::ifstream is(path + "/continuous2d.chk", std::ios::binary);
    if (!is.good()) {
      return;
    }

    PendingCheckpoint2D pending;
    char magic[4] = {};
    is.read(magic, sizeof(magic));
    if (!is.good()) {
      return;
    }
    if (is_checkpoint_magic(magic)) {
      std::uint32_t version = 0U;
      std::uint32_t boundary_code = 0U;
      std::size_t size = 0;
      is.read(reinterpret_cast<char*>(&version), sizeof(version));
      is.read(reinterpret_cast<char*>(&pending.nx), sizeof(pending.nx));
      is.read(reinterpret_cast<char*>(&pending.ny), sizeof(pending.ny));
      is.read(reinterpret_cast<char*>(&pending.dx), sizeof(pending.dx));
      is.read(reinterpret_cast<char*>(&pending.dy), sizeof(pending.dy));
      is.read(reinterpret_cast<char*>(&pending.dt), sizeof(pending.dt));
      is.read(reinterpret_cast<char*>(&pending.total_steps), sizeof(pending.total_steps));
      is.read(reinterpret_cast<char*>(&pending.completed_steps), sizeof(pending.completed_steps));
      is.read(reinterpret_cast<char*>(&boundary_code), sizeof(boundary_code));
      is.read(reinterpret_cast<char*>(&size), sizeof(size));
      pending.boundary = boundary_from_code(boundary_code);
      pending.state.resize(size);
      if (size > 0) {
        is.read(reinterpret_cast<char*>(pending.state.data()),
                static_cast<std::streamsize>(size * sizeof(std::complex<double>)));
      }
      pending.valid = is.good();
    } else {
      is.clear();
      is.seekg(0, std::ios::beg);
      std::uint32_t nx = 0;
      std::uint32_t ny = 0;
      double dx = 0.0;
      double dy = 0.0;
      double dt = 0.0;
      std::uint32_t steps = 0;
      std::size_t size = 0;
      is.read(reinterpret_cast<char*>(&nx), sizeof(nx));
      is.read(reinterpret_cast<char*>(&ny), sizeof(ny));
      is.read(reinterpret_cast<char*>(&dx), sizeof(dx));
      is.read(reinterpret_cast<char*>(&dy), sizeof(dy));
      is.read(reinterpret_cast<char*>(&dt), sizeof(dt));
      is.read(reinterpret_cast<char*>(&steps), sizeof(steps));
      is.read(reinterpret_cast<char*>(&size), sizeof(size));
      pending.nx = nx;
      pending.ny = ny;
      pending.dx = dx;
      pending.dy = dy;
      pending.dt = dt;
      pending.total_steps = steps;
      pending.completed_steps = steps;
      pending.boundary = "periodic";
      pending.state.resize(size);
      if (size > 0) {
        is.read(reinterpret_cast<char*>(pending.state.data()),
                static_cast<std::streamsize>(size * sizeof(std::complex<double>)));
      }
      pending.valid = is.good();
    }

    if (!pending.valid || pending.state.empty()) {
      return;
    }

    pending.state.shrink_to_fit();
    pending_checkpoint_ = std::move(pending);
    has_pending_checkpoint_ = true;
  }

  std::vector<std::complex<double>> current_state() const override { return state_; }

  std::vector<ContinuousDensityFrame> continuous_density_frames() const override { return frames_; }

  std::vector<WignerFrame> wigner_frames() const override { return wigner_frames_; }

 private:
  std::size_t idx(std::uint32_t x, std::uint32_t y) const {
    return static_cast<std::size_t>(y) * static_cast<std::size_t>(nx_) + static_cast<std::size_t>(x);
  }

  double x_coord(std::uint32_t x) const {
    return (static_cast<double>(x) - 0.5 * static_cast<double>(nx_)) * dx_;
  }

  double y_coord(std::uint32_t y) const {
    return (static_cast<double>(y) - 0.5 * static_cast<double>(ny_)) * dy_;
  }

  void build_analytic_potential(const std::string& kind, const std::map<std::string, double>& params) {
    const auto get_param = [&params](const std::string& key, double def) {
      const auto it = params.find(key);
      return it == params.end() ? def : it->second;
    };

    if (kind.empty() || kind == "none") {
      std::fill(potential_.begin(), potential_.end(), 0.0);
      return;
    }

    const double x0 = get_param("x0", 0.0);
    const double y0 = get_param("y0", 0.0);

    if (kind == "harmonic") {
      const double kx = get_param("kx", 1.0);
      const double ky = get_param("ky", 1.0);
      for (std::uint32_t y = 0; y < ny_; ++y) {
        const double yy = y_coord(y) - y0;
        for (std::uint32_t x = 0; x < nx_; ++x) {
          const double xx = x_coord(x) - x0;
          potential_[idx(x, y)] = 0.5 * (kx * xx * xx + ky * yy * yy);
        }
      }
      return;
    }

    if (kind == "barrier") {
      const double height = get_param("height", 1.0);
      const double width_x = std::max(0.0, get_param("width_x", dx_ * 4.0));
      const double width_y = std::max(0.0, get_param("width_y", dy_ * static_cast<double>(ny_)));
      for (std::uint32_t y = 0; y < ny_; ++y) {
        const double yy = std::abs(y_coord(y) - y0);
        for (std::uint32_t x = 0; x < nx_; ++x) {
          const double xx = std::abs(x_coord(x) - x0);
          potential_[idx(x, y)] = (xx <= 0.5 * width_x && yy <= 0.5 * width_y) ? height : 0.0;
        }
      }
      return;
    }

    if (kind == "well") {
      const double depth = std::abs(get_param("depth", 1.0));
      const double width_x = std::max(0.0, get_param("width_x", dx_ * 8.0));
      const double width_y = std::max(0.0, get_param("width_y", dy_ * 8.0));
      for (std::uint32_t y = 0; y < ny_; ++y) {
        const double yy = std::abs(y_coord(y) - y0);
        for (std::uint32_t x = 0; x < nx_; ++x) {
          const double xx = std::abs(x_coord(x) - x0);
          potential_[idx(x, y)] = (xx <= 0.5 * width_x && yy <= 0.5 * width_y) ? -depth : 0.0;
        }
      }
      return;
    }

    std::fill(potential_.begin(), potential_.end(), 0.0);
  }

  void initialize_wavepacket(const ContinuousInitialStateConfig& initial_state,
                             const std::map<std::string, double>& legacy_params) {
    const auto& params = initial_state.params.empty() ? legacy_params : initial_state.params;
    const auto get_param = [&params](const std::string& key, double def) {
      const auto it = params.find(key);
      return it == params.end() ? def : it->second;
    };

    const std::string kind = initial_state.kind.empty() ? "gaussian" : initial_state.kind;
    if (kind == "grid") {
      if (!initial_state.grid.empty()) {
        if (initial_state.grid.size() != state_.size()) {
          throw std::runtime_error("continuous2d initial_state.grid size must equal nx*ny");
        }
        for (std::size_t i = 0; i < state_.size(); ++i) {
          state_[i] = Complex{initial_state.grid[i], 0.0};
        }
      } else {
        std::fill(state_.begin(), state_.end(), Complex{0.0, 0.0});
      }
      normalize_state();
      return;
    }

    const double psi_x0 = get_param("psi_x0", -0.2 * static_cast<double>(nx_) * dx_);
    const double psi_y0 = get_param("psi_y0", 0.0);
    const double psi_sigma = std::max(1e-6, get_param("psi_sigma", 0.1 * std::min(nx_ * dx_, ny_ * dy_)));
    const double psi_kx = get_param("psi_kx", 3.0);
    const double psi_ky = get_param("psi_ky", 0.0);

    for (std::uint32_t y = 0; y < ny_; ++y) {
      const double yy = y_coord(y);
      for (std::uint32_t x = 0; x < nx_; ++x) {
        const double xx = x_coord(x);
        const double rx = xx - psi_x0;
        const double ry = yy - psi_y0;
        const double amp = std::exp(-(rx * rx + ry * ry) / (4.0 * psi_sigma * psi_sigma));
        const double phase = psi_kx * xx + psi_ky * yy;
        state_[idx(x, y)] = amp * std::exp(Complex{0.0, phase});
      }
    }

    normalize_state();
  }

  // Build the absorbing boundary attenuation mask from potential parameters.
  void precompute_boundary_mask(const std::map<std::string, double>& params) {
    if (boundary_ != "absorbing") {
      return;
    }
    const auto get_param = [&params](const std::string& key, double def) {
      const auto it = params.find(key);
      return it == params.end() ? def : it->second;
    };

    const double width_cells = std::max(1.0, get_param("absorb_width", 6.0));
    const double strength = std::max(0.0, get_param("absorb_strength", 3.0));

    for (std::uint32_t y = 0; y < ny_; ++y) {
      const double dy_edge = std::min<double>(y, ny_ - 1U - y);
      for (std::uint32_t x = 0; x < nx_; ++x) {
        const double dx_edge = std::min<double>(x, nx_ - 1U - x);
        const double edge_dist = std::min(dx_edge, dy_edge);
        if (edge_dist >= width_cells) {
          absorbing_mask_[idx(x, y)] = 1.0;
          continue;
        }
        const double t = (width_cells - edge_dist) / width_cells;
        const double damp = std::exp(-strength * t * t * dt_);
        absorbing_mask_[idx(x, y)] = std::clamp(damp, 0.0, 1.0);
      }
    }
  }

  // Apply the absorbing boundary attenuation to the current wavefunction.
  void apply_absorbing_boundary() {
    if (boundary_ != "absorbing") {
      return;
    }
    for (std::size_t i = 0; i < state_.size(); ++i) {
      state_[i] *= absorbing_mask_[i];
    }
  }

  // Zero the outer grid cells for Dirichlet boundary conditions.
  void enforce_dirichlet_boundary() {
    if (boundary_ != "dirichlet") {
      return;
    }
    for (std::uint32_t y = 0; y < ny_; ++y) {
      state_[idx(0, y)] = Complex{0.0, 0.0};
      state_[idx(nx_ - 1U, y)] = Complex{0.0, 0.0};
    }
    for (std::uint32_t x = 0; x < nx_; ++x) {
      state_[idx(x, 0)] = Complex{0.0, 0.0};
      state_[idx(x, ny_ - 1U)] = Complex{0.0, 0.0};
    }
  }

  // Apply a cached checkpoint once the live grid has been initialized.
  bool apply_pending_checkpoint() {
    if (!has_pending_checkpoint_ || !pending_checkpoint_.valid) {
      return false;
    }
    if (pending_checkpoint_.nx != nx_ || pending_checkpoint_.ny != ny_) {
      return false;
    }
    if (std::abs(pending_checkpoint_.dx - dx_) > 1e-15 || std::abs(pending_checkpoint_.dy - dy_) > 1e-15 ||
        std::abs(pending_checkpoint_.dt - dt_) > 1e-15) {
      return false;
    }
    if (!pending_checkpoint_.boundary.empty() && pending_checkpoint_.boundary != boundary_) {
      return false;
    }
    if (pending_checkpoint_.state.size() != state_.size()) {
      return false;
    }
    state_ = pending_checkpoint_.state;
    completed_steps_ = std::min(pending_checkpoint_.completed_steps, steps_);
    has_pending_checkpoint_ = false;
    pending_checkpoint_ = PendingCheckpoint2D{};
    return true;
  }

  // Capture a compact density-frame metadata record for export.
  // ── Wigner quasi-probability and transport observables ─────────────────────

  // Compute 1D Wigner W(x_j, p_k) from the y=y_mid row (periodic wrap, hbar=1).
  //
  // Algorithm:
  //   a'_j[l] = psi[(j+l)%N] * conj(psi[(j-l)%N]) * (-1)^l
  //   W[j, k]  = (dx/pi) * Re{ FFT(a'_j)[k] }
  // Momentum axis: p_k = (k - N/2) * pi/(N*dx)  for k = 0..N-1.
  // Normalization: sum_k W[j,k] * pi/(N*dx) = |psi(x_j, y_mid)|^2.
  void compute_wigner_row(std::vector<double>& out) {
    static constexpr double kPi = 3.14159265358979323846;
    const std::uint32_t N = nx_;
    const std::uint32_t y_mid = ny_ / 2U;
    out.assign(static_cast<std::size_t>(N) * N, 0.0);
    const double scale = dx_ / kPi;

    std::vector<Complex> a_prime(N);
    std::vector<Complex> w_fft(N);

    for (std::uint32_t j = 0; j < N; ++j) {
      for (std::uint32_t l = 0; l < N; ++l) {
        const std::size_t iplus  = idx((j + l) % N, y_mid);
        const std::size_t iminus = idx((j + N - l) % N, y_mid);
        const double sign = (l % 2U == 0U) ? 1.0 : -1.0;
        a_prime[l] = state_[iplus] * std::conj(state_[iminus]) * sign;
      }
      fft_provider_->forward(w_fft, a_prime);
      for (std::uint32_t k = 0; k < N; ++k) {
        out[static_cast<std::size_t>(j) * N + k] = scale * w_fft[k].real();
      }
    }
  }

  // Compute probability current J_x(x,y) and J_y(x,y) via spectral differentiation.
  //   J = (1/m) * Im{ psi* * (-i * grad * psi) }  (hbar=1)
  void compute_probability_current(std::vector<double>& Jx_out, std::vector<double>& Jy_out) {
    static constexpr double kTwoPi = 6.28318530717958647692;
    const double lx = static_cast<double>(nx_) * dx_;
    const double ly = static_cast<double>(ny_) * dy_;
    const double inv_mass = 1.0 / mass_;
    const std::size_t n = state_.size();

    std::vector<Complex> psi_k(n);
    fft2_forward(state_, psi_k);

    std::vector<Complex> dx_k(n), dy_k(n);
    for (std::uint32_t y = 0; y < ny_; ++y) {
      const int ky_idx = (y <= ny_ / 2U) ? static_cast<int>(y) :
                          static_cast<int>(y) - static_cast<int>(ny_);
      const double ky = kTwoPi * static_cast<double>(ky_idx) / ly;
      for (std::uint32_t x = 0; x < nx_; ++x) {
        const int kx_idx = (x <= nx_ / 2U) ? static_cast<int>(x) :
                            static_cast<int>(x) - static_cast<int>(nx_);
        const double kx = kTwoPi * static_cast<double>(kx_idx) / lx;
        const std::size_t i = idx(x, y);
        dx_k[i] = Complex{0.0, kx} * psi_k[i];
        dy_k[i] = Complex{0.0, ky} * psi_k[i];
      }
    }

    std::vector<Complex> dpsi_dx(n), dpsi_dy(n);
    fft2_inverse(dx_k, dpsi_dx);
    fft2_inverse(dy_k, dpsi_dy);

    Jx_out.resize(n);
    Jy_out.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      Jx_out[i] = inv_mass * (std::conj(state_[i]) * dpsi_dx[i]).imag();
      Jy_out[i] = inv_mass * (std::conj(state_[i]) * dpsi_dy[i]).imag();
    }
  }

  // Compute momentum-space expectation values from |psi_tilde(k)|^2.
  void compute_momentum_observables(double& px_mean_out, double& py_mean_out,
                                     double& ke_out) {
    static constexpr double kTwoPi = 6.28318530717958647692;
    const double lx = static_cast<double>(nx_) * dx_;
    const double ly = static_cast<double>(ny_) * dy_;

    std::vector<Complex> psi_k(state_.size());
    fft2_forward(state_, psi_k);

    double norm_k = 0.0, px_sum = 0.0, py_sum = 0.0, p2_sum = 0.0;
    for (std::uint32_t y = 0; y < ny_; ++y) {
      const int ky_idx = (y <= ny_ / 2U) ? static_cast<int>(y) :
                          static_cast<int>(y) - static_cast<int>(ny_);
      const double ky = kTwoPi * static_cast<double>(ky_idx) / ly;
      for (std::uint32_t x = 0; x < nx_; ++x) {
        const int kx_idx = (x <= nx_ / 2U) ? static_cast<int>(x) :
                            static_cast<int>(x) - static_cast<int>(nx_);
        const double kx = kTwoPi * static_cast<double>(kx_idx) / lx;
        const double w = std::norm(psi_k[idx(x, y)]);
        norm_k  += w;
        px_sum  += kx * w;
        py_sum  += ky * w;
        p2_sum  += (kx * kx + ky * ky) * w;
      }
    }
    const double inv_nk = (norm_k > 1e-30) ? 1.0 / norm_k : 0.0;
    px_mean_out = px_sum * inv_nk;
    py_mean_out = py_sum * inv_nk;
    ke_out      = p2_sum * inv_nk / (2.0 * mass_);
  }

  // Assemble and store one Wigner frame on stride-aligned export steps.
  void record_wigner_frame() {
    if (!export_wigner_ || !export_frame_payloads_ || !should_export_payload_frame()) {
      return;
    }
    // Skip non-Schrodinger equations (Wigner is well-defined for scalar wavefunctions).
    if (equation_ != EquationType::Schrodinger) { return; }

    static constexpr double kPi = 3.14159265358979323846;
    WignerFrame wf;
    wf.step = completed_steps_;
    wf.time = static_cast<double>(completed_steps_) * dt_;
    wf.nx   = nx_;
    wf.np   = nx_;
    wf.x_min = x_coord(0);
    wf.x_max = x_coord(nx_ - 1U);
    wf.p_min = -0.5 * kPi / dx_;
    wf.p_max = (0.5 * static_cast<double>(nx_) - 1.0) * kPi /
               (static_cast<double>(nx_) * dx_);

    compute_wigner_row(wf.wigner_flat);
    compute_probability_current(wf.current_x_flat, wf.current_y_flat);

    const std::uint32_t x_left  = nx_ / 4U;
    const std::uint32_t x_right = (3U * nx_) / 4U;
    double t_sum = 0.0, r_sum = 0.0;
    for (std::uint32_t y = 0; y < ny_; ++y) {
      t_sum += wf.current_x_flat[idx(x_right, y)] * dy_;
      r_sum += wf.current_x_flat[idx(x_left,  y)] * dy_;
    }
    wf.transmission = t_sum;
    wf.reflection   = -r_sum;

    compute_momentum_observables(wf.px_mean, wf.py_mean, wf.kinetic_energy);
    wigner_frames_.push_back(std::move(wf));
  }

  void record_frame() {
    ContinuousDensityFrame frame;
    frame.step = completed_steps_;
    frame.time = static_cast<double>(completed_steps_) * dt_;
    frame.nx = nx_;
    frame.ny = ny_;
    frame.nz = 0;
    frame.boundary = boundary_;
    const auto get_obs = [this](const std::string& key) {
      const auto it = observables_.find(key);
      return it == observables_.end() ? 0.0 : it->second;
    };
    frame.norm = get_obs("norm");
    frame.x_mean = get_obs("x_mean");
    frame.y_mean = get_obs("y_mean");
    frame.z_mean = get_obs("z_mean");
    frame.sigma_x = get_obs("sigma_x");
    frame.sigma_y = get_obs("sigma_y");
    frame.sigma_z = get_obs("sigma_z");
    frame.x_min = x_coord(0);
    frame.x_max = x_coord(nx_ - 1U);
    frame.y_min = y_coord(0);
    frame.y_max = y_coord(ny_ - 1U);
    frame.z_min = 0.0;
    frame.z_max = 0.0;
    frame.potential_kind = potential_kind_;
    frame.driving_amplitude = has_driving_ ? driving_envelope(frame.time) : 0.0;
    // Export the dense payload only on stride-aligned frames so the wrapper can sample
    // a stable subset without changing the full metadata timeline.
    if (export_frame_payloads_ && should_export_payload_frame()) {
      frame.density_xy_flat.resize(state_.size());
      if (equation_ == EquationType::Dirac) {
        for (std::size_t i = 0; i < state_.size(); ++i) {
          frame.density_xy_flat[i] = std::norm(state_[i]) + std::norm(dirac_s1_[i]);
        }
      } else {
        for (std::size_t i = 0; i < state_.size(); ++i) {
          frame.density_xy_flat[i] = std::norm(state_[i]);
        }
      }
      // Phase of the primary spinor component: arg(ψ₀)
      frame.phase_xy_flat.resize(state_.size());
      for (std::size_t i = 0; i < state_.size(); ++i) {
        frame.phase_xy_flat[i] = std::arg(state_[i]);
      }
      // Per-component densities for Dirac equation
      if (equation_ == EquationType::Dirac) {
        frame.dirac_s0_xy_flat.resize(state_.size());
        frame.dirac_s1_xy_flat.resize(state_.size());
        for (std::size_t i = 0; i < state_.size(); ++i) {
          frame.dirac_s0_xy_flat[i] = std::norm(state_[i]);
          frame.dirac_s1_xy_flat[i] = std::norm(dirac_s1_[i]);
        }
      }
    }
    frames_.push_back(frame);
  }

  // Decide whether the current frame should carry a dense payload in the public export.
  bool should_export_payload_frame() const {
    return frames_.empty() || completed_steps_ == steps_ || (completed_steps_ % export_frame_stride_ == 0U);
  }

  void precompute_phases() {
    const double two_pi = 6.28318530717958647692;
    const double lx = static_cast<double>(nx_) * dx_;
    const double ly = static_cast<double>(ny_) * dy_;
    const bool relativistic = (dispersion_ == "relativistic");
    const double mc2 = mass_ * speed_of_light_ * speed_of_light_;
    const double c2 = speed_of_light_ * speed_of_light_;

    for (std::uint32_t y = 0; y < ny_; ++y) {
      const int ky_index = (y <= ny_ / 2) ? static_cast<int>(y) : static_cast<int>(y) - static_cast<int>(ny_);
      const double ky = two_pi * static_cast<double>(ky_index) / ly;
      for (std::uint32_t x = 0; x < nx_; ++x) {
        const int kx_index = (x <= nx_ / 2) ? static_cast<int>(x) : static_cast<int>(x) - static_cast<int>(nx_);
        const double kx = two_pi * static_cast<double>(kx_index) / lx;
        const double k2 = kx * kx + ky * ky;
        const std::size_t i = idx(x, y);
        double E_k;
        if (relativistic) {
          // Relativistic kinetic energy: E_k = sqrt(k²c² + m²c⁴) - mc²  (ℏ=1)
          E_k = std::sqrt(k2 * c2 + mc2 * mc2) - mc2;
        } else {
          // Non-relativistic: E_k = k²/(2m)  (ℏ=1)
          E_k = k2 / (2.0 * mass_);
        }
        kinetic_half_phase_[i] = std::exp(Complex{0.0, -0.5 * dt_ * E_k});
        potential_phase_[i] = std::exp(Complex{0.0, -dt_ * potential_[i]});
        potential_half_phase_[i] = std::exp(Complex{0.0, -0.5 * dt_ * potential_[i]});
      }
    }
  }

  void precompute_kg_phases() {
    const double two_pi = 6.28318530717958647692;
    const double lx = static_cast<double>(nx_) * dx_;
    const double ly = static_cast<double>(ny_) * dy_;
    const double half_dt = 0.5 * dt_;

    // Potential half-step arrays (position space): ω_V = sqrt(m² + max(0, V))
    for (std::size_t i = 0; i < potential_.size(); ++i) {
      const double omega_v = std::sqrt(mass_ * mass_ + std::max(0.0, potential_[i]));
      kg_pot_c_[i]  = std::cos(omega_v * half_dt);
      kg_pot_sc_[i] = std::sin(omega_v * half_dt) / omega_v;
      kg_pot_os_[i] = omega_v * std::sin(omega_v * half_dt);
    }

    // Kinetic full-step arrays (Fourier space): ω_k = |k|
    for (std::uint32_t y = 0; y < ny_; ++y) {
      const int ky_index = (y <= ny_ / 2) ? static_cast<int>(y) : static_cast<int>(y) - static_cast<int>(ny_);
      const double ky = two_pi * static_cast<double>(ky_index) / ly;
      for (std::uint32_t x = 0; x < nx_; ++x) {
        const int kx_index = (x <= nx_ / 2) ? static_cast<int>(x) : static_cast<int>(x) - static_cast<int>(nx_);
        const double kx = two_pi * static_cast<double>(kx_index) / lx;
        const double omega_k = std::sqrt(kx * kx + ky * ky);
        const std::size_t i = idx(x, y);
        kg_kin_c_[i]  = std::cos(omega_k * dt_);
        kg_kin_sc_[i] = (omega_k > 1e-14) ? std::sin(omega_k * dt_) / omega_k : dt_;
        kg_kin_os_[i] = omega_k * std::sin(omega_k * dt_);
      }
    }
  }

  void initialize_pi_state(const ContinuousInitialStateConfig& ic) {
    if (ic.pi_kind == "positive_frequency") {
      // π̃_k = -i · ω_k · φ̃_k  where ω_k = sqrt(k² + m²)
      // This selects the positive-energy branch of the full KG dispersion.
      const double two_pi = 6.28318530717958647692;
      const double lx = static_cast<double>(nx_) * dx_;
      const double ly = static_cast<double>(ny_) * dy_;
      fft2_forward(state_, kbuf_);
      for (std::uint32_t y = 0; y < ny_; ++y) {
        const int ky_index = (y <= ny_ / 2) ? static_cast<int>(y) : static_cast<int>(y) - static_cast<int>(ny_);
        const double ky = two_pi * static_cast<double>(ky_index) / ly;
        for (std::uint32_t x = 0; x < nx_; ++x) {
          const int kx_index = (x <= nx_ / 2) ? static_cast<int>(x) : static_cast<int>(x) - static_cast<int>(nx_);
          const double kx = two_pi * static_cast<double>(kx_index) / lx;
          const double k2 = kx * kx + ky * ky;
          const double omega_k = std::sqrt(k2 + mass_ * mass_);
          kpibuf_[idx(x, y)] = Complex{0.0, -omega_k} * kbuf_[idx(x, y)];
        }
      }
      fft2_inverse(kpibuf_, pi_state_);
    }
    // else pi_kind == "zero": pi_state_ already zeroed in initialize()
  }

  void apply_half_potential_kg() {
    for (std::size_t i = 0; i < state_.size(); ++i) {
      const Complex phi = state_[i];
      const Complex pi  = pi_state_[i];
      state_[i]    = phi * kg_pot_c_[i]  + pi * kg_pot_sc_[i];
      pi_state_[i] = -phi * kg_pot_os_[i] + pi * kg_pot_c_[i];
    }
  }

  void apply_full_kinetic_kg() {
    fft2_forward(state_, kbuf_);
    fft2_forward(pi_state_, kpibuf_);
    for (std::size_t i = 0; i < kbuf_.size(); ++i) {
      const Complex phi_k = kbuf_[i];
      const Complex pi_k  = kpibuf_[i];
      kbuf_[i]   = phi_k * kg_kin_c_[i]  + pi_k * kg_kin_sc_[i];
      kpibuf_[i] = -phi_k * kg_kin_os_[i] + pi_k * kg_kin_c_[i];
    }
    fft2_inverse(kbuf_, state_);
    fft2_inverse(kpibuf_, pi_state_);
  }

  // ── Dirac solver (2-component spinor, 2+1 D) ──────────────────────────────
  // H_D(k) = kx·σx + ky·σy + m·σz  →  U(k,dt) = cos(ω·dt)·I - i·sin(ω·dt)/ω · H_D(k)

  void evolve_dirac() {
    for (std::uint32_t s = completed_steps_; s < steps_; ++s) {
      const double t = static_cast<double>(s) * dt_;
      if (has_driving_) {
        apply_half_potential_dirac_driven(t);
      } else {
        apply_half_potential_dirac();
      }
      apply_full_kinetic_dirac_2d();
      if (has_driving_) {
        apply_half_potential_dirac_driven(t + dt_);
      } else {
        apply_half_potential_dirac();
      }
      if (boundary_ == "absorbing") {
        apply_absorbing_boundary();
        for (std::size_t i = 0; i < dirac_s1_.size(); ++i) {
          dirac_s1_[i] *= absorbing_mask_[i];
        }
      } else if (boundary_ == "dirichlet") {
        enforce_dirichlet_boundary();
        std::fill(dirac_s1_.begin(), dirac_s1_.end(), Complex{0.0, 0.0});
      }
      completed_steps_ = s + 1;
      update_observables();
      record_frame();
      record_wigner_frame();
    }
  }

  void apply_half_potential_dirac() {
    for (std::size_t i = 0; i < state_.size(); ++i) {
      state_[i]   *= potential_half_phase_[i];
      dirac_s1_[i] *= potential_half_phase_[i];
    }
  }

  // ── Time-dependent driving helpers ─────────────────────────────────────────

  // Returns the scalar envelope multiplier at time t (before spatial weighting).
  double driving_envelope(double t) const {
    const double A = driving_config_.amplitude;
    const double f = driving_config_.frequency;
    const double ph = driving_config_.phase;
    const double tc = driving_config_.t_center;
    const double tw = std::max(1e-30, driving_config_.t_width);
    if (driving_config_.type == "sine") {
      return A * std::cos(f * t + ph);
    }
    if (driving_config_.type == "gaussian_pulse") {
      const double dt_c = t - tc;
      return A * std::exp(-0.5 * dt_c * dt_c / (tw * tw)) * std::cos(f * t + ph);
    }
    if (driving_config_.type == "chirp") {
      // chirp: ω(t) = frequency + t_width * t  (t_width doubles as chirp rate α)
      return A * std::cos((f + 0.5 * tw * t) * t + ph);
    }
    return 0.0;
  }

  // Full-step potential with time-dependent driving (Schrödinger).
  void apply_potential_driven(double t) {
    const double drive = driving_envelope(t);
    for (std::size_t i = 0; i < state_.size(); ++i) {
      const double V_total = potential_[i] + drive * driving_spatial_[i];
      state_[i] *= std::exp(Complex{0.0, -dt_ * V_total});
    }
  }

  // Half-step potential with time-dependent driving (Dirac).
  void apply_half_potential_dirac_driven(double t) {
    const double drive = driving_envelope(t);
    for (std::size_t i = 0; i < state_.size(); ++i) {
      const double V_total = potential_[i] + drive * driving_spatial_[i];
      const Complex hp = std::exp(Complex{0.0, -0.5 * dt_ * V_total});
      state_[i]    *= hp;
      dirac_s1_[i] *= hp;
    }
  }

  // Recompute Klein-Gordon potential matrices for the current time step.
  void recompute_kg_potential_phases(double t) {
    const double drive = driving_envelope(t + 0.5 * dt_);
    const double half_dt = 0.5 * dt_;
    for (std::size_t i = 0; i < potential_.size(); ++i) {
      const double V_total = potential_[i] + drive * driving_spatial_[i];
      const double omega_v = std::sqrt(mass_ * mass_ + std::max(0.0, V_total));
      kg_pot_c_[i]  = std::cos(omega_v * half_dt);
      kg_pot_sc_[i] = std::sin(omega_v * half_dt) / omega_v;
      kg_pot_os_[i] = omega_v * std::sin(omega_v * half_dt);
    }
  }

  void apply_full_kinetic_dirac_2d() {
    const double two_pi = 6.28318530717958647692;
    const double lx = static_cast<double>(nx_) * dx_;
    const double ly = static_cast<double>(ny_) * dy_;
    fft2_forward(state_,    kbuf_);
    fft2_forward(dirac_s1_, kpibuf_);
    for (std::uint32_t y = 0; y < ny_; ++y) {
      const int ky_idx = (y <= ny_ / 2) ? static_cast<int>(y) : static_cast<int>(y) - static_cast<int>(ny_);
      const double ky = two_pi * static_cast<double>(ky_idx) / ly;
      for (std::uint32_t x = 0; x < nx_; ++x) {
        const int kx_idx = (x <= nx_ / 2) ? static_cast<int>(x) : static_cast<int>(x) - static_cast<int>(nx_);
        const double kx = two_pi * static_cast<double>(kx_idx) / lx;
        const double omega = std::sqrt(kx * kx + ky * ky + mass_ * mass_);
        const double c = std::cos(omega * dt_);
        const double s = (omega > 1e-14) ? std::sin(omega * dt_) / omega : dt_;
        // U = c·I - i·s·H_D, H_D(k) = kx·σx + ky·σy + m·σz
        // U₀₀ = c - i·s·m,  U₀₁ = -i·s·(kx - i·ky) = {-s·ky, -s·kx}
        // U₁₀ = -i·s·(kx + i·ky) = {s·ky, -s·kx},  U₁₁ = c + i·s·m
        const Complex u00 = Complex{c, -s * mass_};
        const Complex u01 = Complex{-s * ky, -s * kx};
        const Complex u10 = Complex{ s * ky, -s * kx};
        const Complex u11 = Complex{c,  s * mass_};
        const std::size_t i = idx(x, y);
        const Complex p0 = kbuf_[i];
        const Complex p1 = kpibuf_[i];
        kbuf_[i]   = u00 * p0 + u01 * p1;
        kpibuf_[i] = u10 * p0 + u11 * p1;
      }
    }
    fft2_inverse(kbuf_,    state_);
    fft2_inverse(kpibuf_,  dirac_s1_);
  }

  void normalize_state() {
    double sum = 0.0;
    for (const auto& a : state_) {
      sum += std::norm(a);
    }
    const double cell = dx_ * dy_;
    const double norm = std::sqrt(std::max(1e-30, sum * cell));
    for (auto& a : state_) {
      a /= norm;
    }
  }

  void fft2_forward(const std::vector<Complex>& in, std::vector<Complex>& out) {
    out = in;
    std::vector<Complex> line;
    std::vector<Complex> transformed;

    line.resize(nx_);
    for (std::uint32_t y = 0; y < ny_; ++y) {
      for (std::uint32_t x = 0; x < nx_; ++x) {
        line[x] = out[idx(x, y)];
      }
      fft_provider_->forward(transformed, line);
      line.swap(transformed);
      for (std::uint32_t x = 0; x < nx_; ++x) {
        out[idx(x, y)] = line[x];
      }
    }

    line.resize(ny_);
    for (std::uint32_t x = 0; x < nx_; ++x) {
      for (std::uint32_t y = 0; y < ny_; ++y) {
        line[y] = out[idx(x, y)];
      }
      fft_provider_->forward(transformed, line);
      line.swap(transformed);
      for (std::uint32_t y = 0; y < ny_; ++y) {
        out[idx(x, y)] = line[y];
      }
    }
  }

  void fft2_inverse(const std::vector<Complex>& in, std::vector<Complex>& out) {
    out = in;
    std::vector<Complex> line;
    std::vector<Complex> transformed;

    line.resize(nx_);
    for (std::uint32_t y = 0; y < ny_; ++y) {
      for (std::uint32_t x = 0; x < nx_; ++x) {
        line[x] = out[idx(x, y)];
      }
      fft_provider_->inverse(transformed, line);
      line.swap(transformed);
      for (std::uint32_t x = 0; x < nx_; ++x) {
        out[idx(x, y)] = line[x];
      }
    }

    line.resize(ny_);
    for (std::uint32_t x = 0; x < nx_; ++x) {
      for (std::uint32_t y = 0; y < ny_; ++y) {
        line[y] = out[idx(x, y)];
      }
      fft_provider_->inverse(transformed, line);
      line.swap(transformed);
      for (std::uint32_t y = 0; y < ny_; ++y) {
        out[idx(x, y)] = line[y];
      }
    }
  }

  void apply_half_kinetic() {
    fft2_forward(state_, kbuf_);
    for (std::size_t i = 0; i < kbuf_.size(); ++i) {
      kbuf_[i] *= kinetic_half_phase_[i];
    }
    fft2_inverse(kbuf_, state_);
  }

  void apply_potential() {
    for (std::size_t i = 0; i < state_.size(); ++i) {
      state_[i] *= potential_phase_[i];
    }
  }

  void update_observables() {
    const double cell = dx_ * dy_;
    double norm = 0.0;
    double x_mean = 0.0;
    double y_mean = 0.0;
    double x2_mean = 0.0;
    double y2_mean = 0.0;

    const bool is_dirac = (equation_ == EquationType::Dirac);
    for (std::uint32_t y = 0; y < ny_; ++y) {
      const double yy = y_coord(y);
      for (std::uint32_t x = 0; x < nx_; ++x) {
        const double xx = x_coord(x);
        const std::size_t ii = idx(x, y);
        double density = std::norm(state_[ii]);
        if (is_dirac) density += std::norm(dirac_s1_[ii]);
        const double p = density * cell;
        norm += p;
        x_mean += xx * p;
        y_mean += yy * p;
        x2_mean += xx * xx * p;
        y2_mean += yy * yy * p;
      }
    }

    const double inv_norm = norm > 1e-30 ? 1.0 / norm : 0.0;
    x_mean *= inv_norm;
    y_mean *= inv_norm;
    x2_mean *= inv_norm;
    y2_mean *= inv_norm;

    const double var_x = std::max(0.0, x2_mean - x_mean * x_mean);
    const double var_y = std::max(0.0, y2_mean - y_mean * y_mean);

    observables_["norm"] = norm;
    observables_["x_mean"] = x_mean;
    observables_["y_mean"] = y_mean;
    observables_["x2_mean"] = x2_mean;
    observables_["y2_mean"] = y2_mean;
    observables_["sigma_x"] = std::sqrt(var_x);
    observables_["sigma_y"] = std::sqrt(var_y);
    observables_["nx"] = static_cast<double>(nx_);
    observables_["ny"] = static_cast<double>(ny_);
    observables_["fft_provider_eigen"] = (fft_provider_ && fft_provider_->name() == "eigen") ? 1.0 : 0.0;
    observables_["fft_provider_cufft"] = (fft_provider_ && fft_provider_->name() == "cufft") ? 1.0 : 0.0;
    observables_["boundary_periodic"] = boundary_ == "periodic" ? 1.0 : 0.0;
    observables_["boundary_absorbing"] = boundary_ == "absorbing" ? 1.0 : 0.0;
    observables_["boundary_dirichlet"] = boundary_ == "dirichlet" ? 1.0 : 0.0;
    observables_["dispersion_relativistic"] = (dispersion_ == "relativistic") ? 1.0 : 0.0;
    observables_["mass"] = mass_;
    observables_["speed_of_light"] = speed_of_light_;
    observables_["equation_klein_gordon"] = (equation_ == EquationType::KleinGordon) ? 1.0 : 0.0;
    observables_["equation_dirac"] = (equation_ == EquationType::Dirac) ? 1.0 : 0.0;
    if (equation_ == EquationType::Dirac) {
      // Spin-z expectation: <σz> = integral of (|ψ₀|² - |ψ₁|²) dx / norm
      double spin_z_num = 0.0;
      for (std::size_t i = 0; i < state_.size(); ++i) {
        spin_z_num += (std::norm(state_[i]) - std::norm(dirac_s1_[i])) * cell;
      }
      observables_["dirac_spin_z"] = (norm > 1e-30) ? spin_z_num / norm : 0.0;
      observables_["dirac_norm_up"] = [&] {
        double s = 0.0;
        for (const auto& a : state_) s += std::norm(a) * cell;
        return s;
      }();
      observables_["dirac_norm_down"] = [&] {
        double s = 0.0;
        for (const auto& a : dirac_s1_) s += std::norm(a) * cell;
        return s;
      }();
    }
    if (equation_ == EquationType::KleinGordon) {
      // Conserved KG charge: Q = integral of i(φ*·π - π*·φ) dx = 2·integral of Im(φ*·π) dx
      double charge = 0.0;
      double pi_norm = 0.0;
      for (std::size_t i = 0; i < state_.size(); ++i) {
        charge += 2.0 * (std::conj(state_[i]) * pi_state_[i]).imag() * cell;
        pi_norm += std::norm(pi_state_[i]) * cell;
      }
      observables_["kg_charge"] = charge;
      observables_["kg_pi_norm"] = pi_norm;
    }
  }

 private:
  std::uint32_t nx_{0};
  std::uint32_t ny_{0};
  double dx_{0.0};
  double dy_{0.0};
  double dt_{0.0};
  std::uint32_t steps_{0};
  std::uint32_t completed_steps_{0};
  std::string boundary_{"periodic"};
  std::string dispersion_{"nonrelativistic"};
  double mass_{1.0};
  double speed_of_light_{1.0};
  EquationType equation_{EquationType::Schrodinger};
  std::string potential_kind_{"none"};
  DrivingConfig driving_config_{};
  bool has_driving_{false};
  std::vector<double> driving_spatial_;  // precomputed spatial profile
  bool export_frame_payloads_{false};
  std::uint32_t export_frame_stride_{1};

  std::vector<Complex> state_;
  std::vector<Complex> pi_state_;   // KG conjugate momentum  π = ∂φ/∂t
  std::vector<Complex> dirac_s1_;   // Dirac spinor component ψ₁ (down)
  std::vector<Complex> kbuf_;
  std::vector<Complex> kpibuf_;     // k-space buffer shared by KG-π and Dirac-ψ₁
  std::vector<double> potential_;
  std::vector<double> absorbing_mask_;
  std::vector<Complex> potential_phase_;       // exp(-i·dt·V)   — used by Schrödinger
  std::vector<Complex> potential_half_phase_;  // exp(-i·dt/2·V) — used by Dirac
  std::vector<Complex> kinetic_half_phase_;
  // Klein-Gordon split-step precomputed arrays.
  std::vector<double> kg_pot_c_;   // cos(ω_V · dt/2)
  std::vector<double> kg_pot_sc_;  // sin(ω_V · dt/2) / ω_V
  std::vector<double> kg_pot_os_;  // ω_V · sin(ω_V · dt/2)
  std::vector<double> kg_kin_c_;   // cos(|k| · dt)
  std::vector<double> kg_kin_sc_;  // sin(|k| · dt) / |k|  (dt at k=0)
  std::vector<double> kg_kin_os_;  // |k| · sin(|k| · dt)  (0 at k=0)
  std::map<std::string, double> observables_;
  std::vector<ContinuousDensityFrame> frames_;
  bool export_wigner_{false};
  std::vector<WignerFrame> wigner_frames_;
  PendingCheckpoint2D pending_checkpoint_{};
  bool has_pending_checkpoint_{false};
  std::unique_ptr<IFFTProvider> fft_provider_;
};

}  // namespace

std::unique_ptr<IBackend> make_continuous2d_backend() {
  return std::make_unique<Continuous2DBackend>();
}

}  // namespace qx
