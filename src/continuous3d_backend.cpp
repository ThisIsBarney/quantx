#include "qx/backends.hpp"
#include "qx/fft_provider.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace qx {
namespace {

using Complex = std::complex<double>;

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
    if (c.boundary != "periodic") {
      throw std::runtime_error("continuous3d backend currently supports boundary=periodic only");
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

    if (!c.potential_grid.empty()) {
      if (c.potential_grid.size() != n) {
        throw std::runtime_error("continuous3d potential_grid size must equal nx*ny*nz");
      }
      potential_ = c.potential_grid;
    } else {
      build_analytic_potential(c.potential_kind, c.potential_params);
    }

    initialize_wavepacket(c.potential_params);
    precompute_phases();
    update_observables();
  }

  void evolve(const QuantumProblem&, const RunOptions&) override {
    for (std::uint32_t s = 0; s < steps_; ++s) {
      apply_half_kinetic();
      apply_potential();
      apply_half_kinetic();
      normalize_state();
    }
    update_observables();
  }

  std::map<std::string, double> measure(const QuantumProblem&) override { return observables_; }

  void checkpoint(const std::string& path) override {
    std::ofstream os(path + "/continuous3d.chk", std::ios::binary);
    if (!os.good()) {
      return;
    }
    os.write(reinterpret_cast<const char*>(&nx_), sizeof(nx_));
    os.write(reinterpret_cast<const char*>(&ny_), sizeof(ny_));
    os.write(reinterpret_cast<const char*>(&nz_), sizeof(nz_));
    os.write(reinterpret_cast<const char*>(&dx_), sizeof(dx_));
    os.write(reinterpret_cast<const char*>(&dy_), sizeof(dy_));
    os.write(reinterpret_cast<const char*>(&dz_), sizeof(dz_));
    os.write(reinterpret_cast<const char*>(&dt_), sizeof(dt_));
    os.write(reinterpret_cast<const char*>(&steps_), sizeof(steps_));
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

    if (!is.good() || size == 0) {
      return;
    }

    if (nx == nx_ && ny == ny_ && nz == nz_ && std::abs(dx - dx_) < 1e-15 && std::abs(dy - dy_) < 1e-15 &&
        std::abs(dz - dz_) < 1e-15 && std::abs(dt - dt_) < 1e-15) {
      state_.resize(size);
      is.read(reinterpret_cast<char*>(state_.data()),
              static_cast<std::streamsize>(size * sizeof(std::complex<double>)));
      if (is.good()) {
        update_observables();
      }
    }
  }

  std::vector<std::complex<double>> current_state() const override { return state_; }

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

  void initialize_wavepacket(const std::map<std::string, double>& params) {
    const auto get_param = [&params](const std::string& key, double def) {
      const auto it = params.find(key);
      return it == params.end() ? def : it->second;
    };

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
  std::string boundary_{"periodic"};

  std::vector<Complex> state_;
  std::vector<Complex> kbuf_;
  std::vector<double> potential_;
  std::vector<Complex> potential_phase_;
  std::vector<Complex> kinetic_half_phase_;
  std::map<std::string, double> observables_;
  std::unique_ptr<IFFTProvider> fft_provider_;
};

}  // namespace

std::unique_ptr<IBackend> make_continuous3d_backend() {
  return std::make_unique<Continuous3DBackend>();
}

}  // namespace qx
