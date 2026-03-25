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
    c.equation_scope = "Schrodinger";
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
    potential_kind_ = c.potential_kind;
    completed_steps_ = 0;
    export_frame_payloads_ = options.export_continuous_frames;
    export_frame_stride_ = std::max<std::uint32_t>(1U, options.export_frame_stride);
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
    potential_.assign(n, 0.0);
    potential_phase_.assign(n, Complex{1.0, 0.0});
    kinetic_half_phase_.assign(n, Complex{1.0, 0.0});

    if (!c.potential_grid.empty()) {
      if (c.potential_grid.size() != n) {
        throw std::runtime_error("continuous2d potential_grid size must equal nx*ny");
      }
      potential_ = c.potential_grid;
    } else {
      build_analytic_potential(c.potential_kind, c.potential_params);
    }

    initialize_wavepacket(c.initial_state, c.potential_params);
    precompute_phases();
    precompute_boundary_mask(c.potential_params);

    const bool restored = apply_pending_checkpoint();
    if (!restored && boundary_ == "dirichlet") {
      enforce_dirichlet_boundary();
    }

    update_observables();
    record_frame();
  }

  void evolve(const QuantumProblem&, const RunOptions&) override {
    for (std::uint32_t s = completed_steps_; s < steps_; ++s) {
      apply_half_kinetic();
      apply_potential();
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
    // Export the dense payload only on stride-aligned frames so the wrapper can sample
    // a stable subset without changing the full metadata timeline.
    if (export_frame_payloads_ && should_export_payload_frame()) {
      frame.density_xy_flat.resize(state_.size());
      for (std::size_t i = 0; i < state_.size(); ++i) {
        frame.density_xy_flat[i] = std::norm(state_[i]);
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
      }
    }
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

    for (std::uint32_t y = 0; y < ny_; ++y) {
      const double yy = y_coord(y);
      for (std::uint32_t x = 0; x < nx_; ++x) {
        const double xx = x_coord(x);
        const double p = std::norm(state_[idx(x, y)]) * cell;
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
  std::string potential_kind_{"none"};
  bool export_frame_payloads_{false};
  std::uint32_t export_frame_stride_{1};

  std::vector<Complex> state_;
  std::vector<Complex> kbuf_;
  std::vector<double> potential_;
  std::vector<double> absorbing_mask_;
  std::vector<Complex> potential_phase_;
  std::vector<Complex> kinetic_half_phase_;
  std::map<std::string, double> observables_;
  std::vector<ContinuousDensityFrame> frames_;
  PendingCheckpoint2D pending_checkpoint_{};
  bool has_pending_checkpoint_{false};
  std::unique_ptr<IFFTProvider> fft_provider_;
};

}  // namespace

std::unique_ptr<IBackend> make_continuous2d_backend() {
  return std::make_unique<Continuous2DBackend>();
}

}  // namespace qx
