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
struct PendingCheckpoint3D {
  bool valid{false};
  std::uint32_t nx{0};
  std::uint32_t ny{0};
  std::uint32_t nz{0};
  double dx{0.0};
  double dy{0.0};
  double dz{0.0};
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

// Check whether a checkpoint header matches the 3D continuous magic prefix.
bool is_checkpoint_magic(const char magic[4]) {
  return magic[0] == 'Q' && magic[1] == 'X' && magic[2] == '3' && magic[3] == 'D';
}

class Continuous3DBackend final : public IBackend {
 public:
  std::string name() const override { return "continuous3d"; }

  BackendCapabilities capabilities() const override {
    BackendCapabilities c;
    c.kind = BackendKind::Continuous3D;
    c.supports_gpu = false;
    c.supports_mpi = false;
    c.supports_noisy = false;
    c.max_qubits_recommended = 0;
    c.equation_scope = "Schrodinger";
    return c;
  }

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

    for (std::uint32_t z = 0; z < nz_; ++z) {
      const double dz_edge = std::min<double>(z, nz_ - 1U - z);
      for (std::uint32_t y = 0; y < ny_; ++y) {
        const double dy_edge = std::min<double>(y, ny_ - 1U - y);
        for (std::uint32_t x = 0; x < nx_; ++x) {
          const double dx_edge = std::min<double>(x, nx_ - 1U - x);
          const double edge_dist = std::min({dx_edge, dy_edge, dz_edge});
          if (edge_dist >= width_cells) {
            absorbing_mask_[idx(x, y, z)] = 1.0;
            continue;
          }
          const double t = (width_cells - edge_dist) / width_cells;
          const double damp = std::exp(-strength * t * t * dt_);
          absorbing_mask_[idx(x, y, z)] = std::clamp(damp, 0.0, 1.0);
        }
      }
    }
  }

  void apply_absorbing_boundary() {
    if (boundary_ != "absorbing") {
      return;
    }
    for (std::size_t i = 0; i < state_.size(); ++i) {
      state_[i] *= absorbing_mask_[i];
    }
  }

  void enforce_dirichlet_boundary() {
    if (boundary_ != "dirichlet") {
      return;
    }
    for (std::uint32_t z = 0; z < nz_; ++z) {
      for (std::uint32_t y = 0; y < ny_; ++y) {
        state_[idx(0, y, z)] = Complex{0.0, 0.0};
        state_[idx(nx_ - 1U, y, z)] = Complex{0.0, 0.0};
      }
    }
    for (std::uint32_t z = 0; z < nz_; ++z) {
      for (std::uint32_t x = 0; x < nx_; ++x) {
        state_[idx(x, 0, z)] = Complex{0.0, 0.0};
        state_[idx(x, ny_ - 1U, z)] = Complex{0.0, 0.0};
      }
    }
    for (std::uint32_t y = 0; y < ny_; ++y) {
      for (std::uint32_t x = 0; x < nx_; ++x) {
        state_[idx(x, y, 0)] = Complex{0.0, 0.0};
        state_[idx(x, y, nz_ - 1U)] = Complex{0.0, 0.0};
      }
    }
  }

  void initialize(const QuantumProblem& problem, const RunOptions& options) override {
    if (problem.kind != ProblemKind::ContinuousSpace3D) {
      throw std::runtime_error("continuous3d backend requires ProblemKind::ContinuousSpace3D");
    }

    const auto& c = problem.continuous_3d;
    if (c.nx == 0 || c.ny == 0 || c.nz == 0) {
      throw std::runtime_error("continuous3d grid resolution must be positive");
    }
    if (c.dx <= 0.0 || c.dy <= 0.0 || c.dz <= 0.0 || c.dt <= 0.0 || c.steps == 0) {
      throw std::runtime_error("continuous3d dx/dy/dz/dt/steps must be positive");
    }
    if (c.boundary != "periodic" && c.boundary != "absorbing" && c.boundary != "dirichlet") {
      throw std::runtime_error("continuous3d backend supports boundary in {periodic,absorbing,dirichlet}");
    }

    nx_ = c.nx;
    ny_ = c.ny;
    nz_ = c.nz;
    dx_ = c.dx;
    dy_ = c.dy;
    dz_ = c.dz;
    dt_ = c.dt;
    steps_ = c.steps;
    boundary_ = c.boundary;
    potential_kind_ = c.potential_kind;
    completed_steps_ = 0;
    export_frame_payloads_ = options.export_continuous_frames;
    frames_.clear();
    absorbing_mask_.assign(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_) * static_cast<std::size_t>(nz_),
                           1.0);

    if (options.use_gpu) {
      try {
        fft_provider_ = create_fft_provider(FFTProviderKind::CuFFT);
      } catch (const std::exception&) {
        fft_provider_ = create_fft_provider(FFTProviderKind::Eigen);
      }
    } else {
      fft_provider_ = create_fft_provider(FFTProviderKind::Auto);
    }

    const std::size_t n = static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_) * static_cast<std::size_t>(nz_);
    state_.assign(n, Complex{0.0, 0.0});
    kbuf_.assign(n, Complex{0.0, 0.0});
    potential_.assign(n, 0.0);
    potential_phase_.assign(n, Complex{1.0, 0.0});
    kinetic_half_phase_.assign(n, Complex{1.0, 0.0});
    absorbing_mask_.assign(n, 1.0);

    if (!c.potential_grid.empty()) {
      if (c.potential_grid.size() != n) {
        throw std::runtime_error("continuous3d potential_grid size must equal nx*ny*nz");
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
      apply_absorbing_boundary();
      enforce_dirichlet_boundary();
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
    std::ofstream os(path + "/continuous3d.chk", std::ios::binary);
    if (!os.good()) {
      return;
    }
    const char magic[4] = {'Q', 'X', '3', 'D'};
    const std::uint32_t version = 2U;
    const std::uint32_t boundary_code = boundary_to_code(boundary_);
    os.write(magic, sizeof(magic));
    os.write(reinterpret_cast<const char*>(&version), sizeof(version));
    os.write(reinterpret_cast<const char*>(&nx_), sizeof(nx_));
    os.write(reinterpret_cast<const char*>(&ny_), sizeof(ny_));
    os.write(reinterpret_cast<const char*>(&nz_), sizeof(nz_));
    os.write(reinterpret_cast<const char*>(&dx_), sizeof(dx_));
    os.write(reinterpret_cast<const char*>(&dy_), sizeof(dy_));
    os.write(reinterpret_cast<const char*>(&dz_), sizeof(dz_));
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
    std::ifstream is(path + "/continuous3d.chk", std::ios::binary);
    if (!is.good()) {
      return;
    }
    PendingCheckpoint3D pending;
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
      is.read(reinterpret_cast<char*>(&pending.nz), sizeof(pending.nz));
      is.read(reinterpret_cast<char*>(&pending.dx), sizeof(pending.dx));
      is.read(reinterpret_cast<char*>(&pending.dy), sizeof(pending.dy));
      is.read(reinterpret_cast<char*>(&pending.dz), sizeof(pending.dz));
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
      std::uint32_t nz = 0;
      double dx = 0.0;
      double dy = 0.0;
      double dz = 0.0;
      double dt = 0.0;
      std::uint32_t steps = 0;
      std::size_t size = 0;
      is.read(reinterpret_cast<char*>(&nx), sizeof(nx));
      is.read(reinterpret_cast<char*>(&ny), sizeof(ny));
      is.read(reinterpret_cast<char*>(&nz), sizeof(nz));
      is.read(reinterpret_cast<char*>(&dx), sizeof(dx));
      is.read(reinterpret_cast<char*>(&dy), sizeof(dy));
      is.read(reinterpret_cast<char*>(&dz), sizeof(dz));
      is.read(reinterpret_cast<char*>(&dt), sizeof(dt));
      is.read(reinterpret_cast<char*>(&steps), sizeof(steps));
      is.read(reinterpret_cast<char*>(&size), sizeof(size));
      pending.nx = nx;
      pending.ny = ny;
      pending.nz = nz;
      pending.dx = dx;
      pending.dy = dy;
      pending.dz = dz;
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
  std::size_t idx(std::uint32_t x, std::uint32_t y, std::uint32_t z) const {
    return (static_cast<std::size_t>(z) * static_cast<std::size_t>(ny_) + static_cast<std::size_t>(y)) *
               static_cast<std::size_t>(nx_) +
           static_cast<std::size_t>(x);
  }

  double x_coord(std::uint32_t x) const {
    return (static_cast<double>(x) - 0.5 * static_cast<double>(nx_)) * dx_;
  }

  double y_coord(std::uint32_t y) const {
    return (static_cast<double>(y) - 0.5 * static_cast<double>(ny_)) * dy_;
  }

  double z_coord(std::uint32_t z) const {
    return (static_cast<double>(z) - 0.5 * static_cast<double>(nz_)) * dz_;
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
    const double z0 = get_param("z0", 0.0);

    if (kind == "harmonic") {
      const double kx = get_param("kx", 1.0);
      const double ky = get_param("ky", 1.0);
      const double kz = get_param("kz", 1.0);
      for (std::uint32_t z = 0; z < nz_; ++z) {
        const double zz = z_coord(z) - z0;
        for (std::uint32_t y = 0; y < ny_; ++y) {
          const double yy = y_coord(y) - y0;
          for (std::uint32_t x = 0; x < nx_; ++x) {
            const double xx = x_coord(x) - x0;
            potential_[idx(x, y, z)] = 0.5 * (kx * xx * xx + ky * yy * yy + kz * zz * zz);
          }
        }
      }
      return;
    }

    if (kind == "barrier") {
      const double height = get_param("height", 1.0);
      const double width_x = std::max(0.0, get_param("width_x", dx_ * 4.0));
      const double width_y = std::max(0.0, get_param("width_y", dy_ * static_cast<double>(ny_)));
      const double width_z = std::max(0.0, get_param("width_z", dz_ * static_cast<double>(nz_)));
      for (std::uint32_t z = 0; z < nz_; ++z) {
        const double zz = std::abs(z_coord(z) - z0);
        for (std::uint32_t y = 0; y < ny_; ++y) {
          const double yy = std::abs(y_coord(y) - y0);
          for (std::uint32_t x = 0; x < nx_; ++x) {
            const double xx = std::abs(x_coord(x) - x0);
            potential_[idx(x, y, z)] = (xx <= 0.5 * width_x && yy <= 0.5 * width_y && zz <= 0.5 * width_z) ? height : 0.0;
          }
        }
      }
      return;
    }

    if (kind == "well") {
      const double depth = std::abs(get_param("depth", 1.0));
      const double width_x = std::max(0.0, get_param("width_x", dx_ * 8.0));
      const double width_y = std::max(0.0, get_param("width_y", dy_ * 8.0));
      const double width_z = std::max(0.0, get_param("width_z", dz_ * 8.0));
      for (std::uint32_t z = 0; z < nz_; ++z) {
        const double zz = std::abs(z_coord(z) - z0);
        for (std::uint32_t y = 0; y < ny_; ++y) {
          const double yy = std::abs(y_coord(y) - y0);
          for (std::uint32_t x = 0; x < nx_; ++x) {
            const double xx = std::abs(x_coord(x) - x0);
            potential_[idx(x, y, z)] = (xx <= 0.5 * width_x && yy <= 0.5 * width_y && zz <= 0.5 * width_z) ? -depth : 0.0;
          }
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
          throw std::runtime_error("continuous3d initial_state.grid size must equal nx*ny*nz");
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
    const double psi_z0 = get_param("psi_z0", 0.0);
    const double psi_sigma =
        std::max(1e-6, get_param("psi_sigma", 0.1 * std::min(nx_ * dx_, std::min(ny_ * dy_, nz_ * dz_))));
    const double psi_kx = get_param("psi_kx", 3.0);
    const double psi_ky = get_param("psi_ky", 0.0);
    const double psi_kz = get_param("psi_kz", 0.0);

    for (std::uint32_t z = 0; z < nz_; ++z) {
      const double zz = z_coord(z);
      for (std::uint32_t y = 0; y < ny_; ++y) {
        const double yy = y_coord(y);
        for (std::uint32_t x = 0; x < nx_; ++x) {
          const double xx = x_coord(x);
          const double rx = xx - psi_x0;
          const double ry = yy - psi_y0;
          const double rz = zz - psi_z0;
          const double amp = std::exp(-(rx * rx + ry * ry + rz * rz) / (4.0 * psi_sigma * psi_sigma));
          const double phase = psi_kx * xx + psi_ky * yy + psi_kz * zz;
          state_[idx(x, y, z)] = amp * std::exp(Complex{0.0, phase});
        }
      }
    }

    normalize_state();
  }

  // Apply a cached checkpoint once the live grid has been initialized.
  bool apply_pending_checkpoint() {
    if (!has_pending_checkpoint_ || !pending_checkpoint_.valid) {
      return false;
    }
    if (pending_checkpoint_.nx != nx_ || pending_checkpoint_.ny != ny_ || pending_checkpoint_.nz != nz_) {
      return false;
    }
    if (std::abs(pending_checkpoint_.dx - dx_) > 1e-15 || std::abs(pending_checkpoint_.dy - dy_) > 1e-15 ||
        std::abs(pending_checkpoint_.dz - dz_) > 1e-15 || std::abs(pending_checkpoint_.dt - dt_) > 1e-15) {
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
    pending_checkpoint_ = PendingCheckpoint3D{};
    return true;
  }

  // Capture a compact metadata record for the current density frame.
  void record_frame() {
    ContinuousDensityFrame frame;
    frame.step = completed_steps_;
    frame.time = static_cast<double>(completed_steps_) * dt_;
    frame.nx = nx_;
    frame.ny = ny_;
    frame.nz = nz_;
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
    frame.z_min = z_coord(0);
    frame.z_max = z_coord(nz_ - 1U);
    frame.potential_kind = potential_kind_;
    if (export_frame_payloads_) {
      const std::uint32_t z_mid = nz_ / 2U;
      const std::uint32_t y_mid = ny_ / 2U;
      const std::uint32_t x_mid = nx_ / 2U;
      frame.density_xy_flat.reserve(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(ny_));
      frame.density_xz_flat.reserve(static_cast<std::size_t>(nx_) * static_cast<std::size_t>(nz_));
      frame.density_yz_flat.reserve(static_cast<std::size_t>(ny_) * static_cast<std::size_t>(nz_));

      double dens_max = 0.0;
      for (const auto& a : state_) {
        dens_max = std::max(dens_max, std::norm(a));
      }

      for (std::uint32_t y = 0; y < ny_; ++y) {
        for (std::uint32_t x = 0; x < nx_; ++x) {
          frame.density_xy_flat.push_back(std::norm(state_[idx(x, y, z_mid)]));
        }
      }
      for (std::uint32_t z = 0; z < nz_; ++z) {
        for (std::uint32_t x = 0; x < nx_; ++x) {
          frame.density_xz_flat.push_back(std::norm(state_[idx(x, y_mid, z)]));
        }
      }
      for (std::uint32_t z = 0; z < nz_; ++z) {
        for (std::uint32_t y = 0; y < ny_; ++y) {
          frame.density_yz_flat.push_back(std::norm(state_[idx(x_mid, y, z)]));
        }
      }

      // Export sparse volumetric points for the standalone 3D player.
      const double floor = dens_max * 0.08;
      const std::size_t max_points = 2500;
      struct PointRow {
        double x;
        double y;
        double z;
        double d;
      };
      std::vector<PointRow> points;
      points.reserve(std::min<std::size_t>(max_points * 2U, state_.size()));
      for (std::uint32_t z = 0; z < nz_; ++z) {
        const double zz = z_coord(z);
        for (std::uint32_t y = 0; y < ny_; ++y) {
          const double yy = y_coord(y);
          for (std::uint32_t x = 0; x < nx_; ++x) {
            const double d = std::norm(state_[idx(x, y, z)]);
            if (d < floor) {
              continue;
            }
            points.push_back(PointRow{x_coord(x), yy, zz, d});
          }
        }
      }
      if (points.size() > max_points) {
        std::nth_element(
            points.begin(),
            points.begin() + static_cast<std::ptrdiff_t>(max_points),
            points.end(),
            [](const PointRow& a, const PointRow& b) { return a.d > b.d; });
        points.resize(max_points);
      }
      frame.points_xyzd_flat.reserve(points.size() * 4U);
      for (const auto& p : points) {
        frame.points_xyzd_flat.push_back(p.x);
        frame.points_xyzd_flat.push_back(p.y);
        frame.points_xyzd_flat.push_back(p.z);
        frame.points_xyzd_flat.push_back(p.d);
      }
    }
    frames_.push_back(frame);
  }

  void precompute_phases() {
    const double two_pi = 6.28318530717958647692;
    const double lx = static_cast<double>(nx_) * dx_;
    const double ly = static_cast<double>(ny_) * dy_;
    const double lz = static_cast<double>(nz_) * dz_;

    for (std::uint32_t z = 0; z < nz_; ++z) {
      const int kz_index = (z <= nz_ / 2) ? static_cast<int>(z) : static_cast<int>(z) - static_cast<int>(nz_);
      const double kz = two_pi * static_cast<double>(kz_index) / lz;
      for (std::uint32_t y = 0; y < ny_; ++y) {
        const int ky_index = (y <= ny_ / 2) ? static_cast<int>(y) : static_cast<int>(y) - static_cast<int>(ny_);
        const double ky = two_pi * static_cast<double>(ky_index) / ly;
        for (std::uint32_t x = 0; x < nx_; ++x) {
          const int kx_index = (x <= nx_ / 2) ? static_cast<int>(x) : static_cast<int>(x) - static_cast<int>(nx_);
          const double kx = two_pi * static_cast<double>(kx_index) / lx;
          const double k2 = kx * kx + ky * ky + kz * kz;
          const std::size_t i = idx(x, y, z);
          kinetic_half_phase_[i] = std::exp(Complex{0.0, -0.25 * dt_ * k2});
          potential_phase_[i] = std::exp(Complex{0.0, -dt_ * potential_[i]});
        }
      }
    }
  }

  void normalize_state() {
    double sum = 0.0;
    for (const auto& a : state_) {
      sum += std::norm(a);
    }
    const double cell = dx_ * dy_ * dz_;
    const double norm = std::sqrt(std::max(1e-30, sum * cell));
    for (auto& a : state_) {
      a /= norm;
    }
  }

  void fft3_forward(const std::vector<Complex>& in, std::vector<Complex>& out) {
    out = in;
    std::vector<Complex> line;
    std::vector<Complex> transformed;

    line.resize(nx_);
    for (std::uint32_t z = 0; z < nz_; ++z) {
      for (std::uint32_t y = 0; y < ny_; ++y) {
        for (std::uint32_t x = 0; x < nx_; ++x) {
          line[x] = out[idx(x, y, z)];
        }
        fft_provider_->forward(transformed, line);
        line.swap(transformed);
        for (std::uint32_t x = 0; x < nx_; ++x) {
          out[idx(x, y, z)] = line[x];
        }
      }
    }

    line.resize(ny_);
    for (std::uint32_t z = 0; z < nz_; ++z) {
      for (std::uint32_t x = 0; x < nx_; ++x) {
        for (std::uint32_t y = 0; y < ny_; ++y) {
          line[y] = out[idx(x, y, z)];
        }
        fft_provider_->forward(transformed, line);
        line.swap(transformed);
        for (std::uint32_t y = 0; y < ny_; ++y) {
          out[idx(x, y, z)] = line[y];
        }
      }
    }

    line.resize(nz_);
    for (std::uint32_t y = 0; y < ny_; ++y) {
      for (std::uint32_t x = 0; x < nx_; ++x) {
        for (std::uint32_t z = 0; z < nz_; ++z) {
          line[z] = out[idx(x, y, z)];
        }
        fft_provider_->forward(transformed, line);
        line.swap(transformed);
        for (std::uint32_t z = 0; z < nz_; ++z) {
          out[idx(x, y, z)] = line[z];
        }
      }
    }
  }

  void fft3_inverse(const std::vector<Complex>& in, std::vector<Complex>& out) {
    out = in;
    std::vector<Complex> line;
    std::vector<Complex> transformed;

    line.resize(nx_);
    for (std::uint32_t z = 0; z < nz_; ++z) {
      for (std::uint32_t y = 0; y < ny_; ++y) {
        for (std::uint32_t x = 0; x < nx_; ++x) {
          line[x] = out[idx(x, y, z)];
        }
        fft_provider_->inverse(transformed, line);
        line.swap(transformed);
        for (std::uint32_t x = 0; x < nx_; ++x) {
          out[idx(x, y, z)] = line[x];
        }
      }
    }

    line.resize(ny_);
    for (std::uint32_t z = 0; z < nz_; ++z) {
      for (std::uint32_t x = 0; x < nx_; ++x) {
        for (std::uint32_t y = 0; y < ny_; ++y) {
          line[y] = out[idx(x, y, z)];
        }
        fft_provider_->inverse(transformed, line);
        line.swap(transformed);
        for (std::uint32_t y = 0; y < ny_; ++y) {
          out[idx(x, y, z)] = line[y];
        }
      }
    }

    line.resize(nz_);
    for (std::uint32_t y = 0; y < ny_; ++y) {
      for (std::uint32_t x = 0; x < nx_; ++x) {
        for (std::uint32_t z = 0; z < nz_; ++z) {
          line[z] = out[idx(x, y, z)];
        }
        fft_provider_->inverse(transformed, line);
        line.swap(transformed);
        for (std::uint32_t z = 0; z < nz_; ++z) {
          out[idx(x, y, z)] = line[z];
        }
      }
    }
  }

  void apply_half_kinetic() {
    fft3_forward(state_, kbuf_);
    for (std::size_t i = 0; i < kbuf_.size(); ++i) {
      kbuf_[i] *= kinetic_half_phase_[i];
    }
    fft3_inverse(kbuf_, state_);
  }

  void apply_potential() {
    for (std::size_t i = 0; i < state_.size(); ++i) {
      state_[i] *= potential_phase_[i];
    }
  }

  void update_observables() {
    const double cell = dx_ * dy_ * dz_;
    double norm = 0.0;
    double x_mean = 0.0;
    double y_mean = 0.0;
    double z_mean = 0.0;
    double x2_mean = 0.0;
    double y2_mean = 0.0;
    double z2_mean = 0.0;

    for (std::uint32_t z = 0; z < nz_; ++z) {
      const double zz = z_coord(z);
      for (std::uint32_t y = 0; y < ny_; ++y) {
        const double yy = y_coord(y);
        for (std::uint32_t x = 0; x < nx_; ++x) {
          const double xx = x_coord(x);
          const double p = std::norm(state_[idx(x, y, z)]) * cell;
          norm += p;
          x_mean += xx * p;
          y_mean += yy * p;
          z_mean += zz * p;
          x2_mean += xx * xx * p;
          y2_mean += yy * yy * p;
          z2_mean += zz * zz * p;
        }
      }
    }

    const double inv_norm = norm > 1e-30 ? 1.0 / norm : 0.0;
    x_mean *= inv_norm;
    y_mean *= inv_norm;
    z_mean *= inv_norm;
    x2_mean *= inv_norm;
    y2_mean *= inv_norm;
    z2_mean *= inv_norm;

    const double var_x = std::max(0.0, x2_mean - x_mean * x_mean);
    const double var_y = std::max(0.0, y2_mean - y_mean * y_mean);
    const double var_z = std::max(0.0, z2_mean - z_mean * z_mean);

    observables_["norm"] = norm;
    observables_["x_mean"] = x_mean;
    observables_["y_mean"] = y_mean;
    observables_["z_mean"] = z_mean;
    observables_["x2_mean"] = x2_mean;
    observables_["y2_mean"] = y2_mean;
    observables_["z2_mean"] = z2_mean;
    observables_["sigma_x"] = std::sqrt(var_x);
    observables_["sigma_y"] = std::sqrt(var_y);
    observables_["sigma_z"] = std::sqrt(var_z);
    observables_["nx"] = static_cast<double>(nx_);
    observables_["ny"] = static_cast<double>(ny_);
    observables_["nz"] = static_cast<double>(nz_);
    observables_["fft_provider_eigen"] = (fft_provider_ && fft_provider_->name() == "eigen") ? 1.0 : 0.0;
    observables_["fft_provider_cufft"] = (fft_provider_ && fft_provider_->name() == "cufft") ? 1.0 : 0.0;
    observables_["boundary_periodic"] = boundary_ == "periodic" ? 1.0 : 0.0;
    observables_["boundary_absorbing"] = boundary_ == "absorbing" ? 1.0 : 0.0;
    observables_["boundary_dirichlet"] = boundary_ == "dirichlet" ? 1.0 : 0.0;
  }

 private:
  std::uint32_t nx_{0};
  std::uint32_t ny_{0};
  std::uint32_t nz_{0};
  double dx_{0.0};
  double dy_{0.0};
  double dz_{0.0};
  double dt_{0.0};
  std::uint32_t steps_{0};
  std::uint32_t completed_steps_{0};
  std::string boundary_{"periodic"};
  std::string potential_kind_{"none"};
  bool export_frame_payloads_{false};

  std::vector<Complex> state_;
  std::vector<Complex> kbuf_;
  std::vector<double> potential_;
  std::vector<double> absorbing_mask_;
  std::vector<Complex> potential_phase_;
  std::vector<Complex> kinetic_half_phase_;
  std::map<std::string, double> observables_;
  std::vector<ContinuousDensityFrame> frames_;
  PendingCheckpoint3D pending_checkpoint_{};
  bool has_pending_checkpoint_{false};
  std::unique_ptr<IFFTProvider> fft_provider_;
};

}  // namespace

std::unique_ptr<IBackend> make_continuous3d_backend() {
  return std::make_unique<Continuous3DBackend>();
}

}  // namespace qx
