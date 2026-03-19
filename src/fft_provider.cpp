#include "qx/fft_provider.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <unsupported/Eigen/FFT>

#if defined(QX_HAS_CUDA)
#include <cuda_runtime.h>
#include <cufft.h>
#endif

namespace qx {
namespace {

using Complex = std::complex<double>;

// Build a readable runtime error message from a CUDA status code.
std::runtime_error make_cuda_error(const char* api, int code) {
  return std::runtime_error(std::string(api) + " failed with code " + std::to_string(code));
}

// Build a readable runtime error message from a cuFFT status code.
std::runtime_error make_cufft_error(const char* api, int code) {
  return std::runtime_error(std::string(api) + " failed with code " + std::to_string(code));
}

// Report a CUDA call failure as an exception.
void check_cuda(int code, const char* api) {
#if defined(QX_HAS_CUDA)
  if (code != cudaSuccess) {
    throw make_cuda_error(api, code);
  }
#else
  (void)code;
  (void)api;
#endif
}

// Report a cuFFT call failure as an exception.
void check_cufft(int code, const char* api) {
#if defined(QX_HAS_CUDA)
  if (code != CUFFT_SUCCESS) {
    throw make_cufft_error(api, code);
  }
#else
  (void)code;
  (void)api;
#endif
}

// Probe whether the current runtime can create and execute a tiny cuFFT plan.
bool probe_cufft_runtime() {
#if defined(QX_HAS_CUDA)
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return false;
  }

  cufftHandle plan = 0;
  cufftDoubleComplex* device_buffer = nullptr;
  constexpr int kProbeSize = 4;
  const std::size_t bytes = static_cast<std::size_t>(kProbeSize) * sizeof(cufftDoubleComplex);
  cufftDoubleComplex host_input[kProbeSize] = {
      {1.0, 0.0},
      {0.0, 0.0},
      {0.0, 0.0},
      {0.0, 0.0},
  };
  cufftDoubleComplex host_output[kProbeSize] = {};

  try {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_buffer), bytes), "cudaMalloc");
    check_cuda(cudaMemcpy(device_buffer, host_input, bytes, cudaMemcpyHostToDevice), "cudaMemcpyHostToDevice");
    check_cufft(cufftPlan1d(&plan, kProbeSize, CUFFT_Z2Z, 1), "cufftPlan1d");
    check_cufft(cufftExecZ2Z(plan, device_buffer, device_buffer, CUFFT_FORWARD), "cufftExecZ2Z");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check_cuda(cudaMemcpy(host_output, device_buffer, bytes, cudaMemcpyDeviceToHost), "cudaMemcpyDeviceToHost");
  } catch (...) {
    if (plan != 0) {
      cufftDestroy(plan);
    }
    if (device_buffer != nullptr) {
      cudaFree(device_buffer);
    }
    return false;
  }

  if (plan != 0) {
    cufftDestroy(plan);
  }
  if (device_buffer != nullptr) {
    cudaFree(device_buffer);
  }
  return std::isfinite(host_output[0].x) && std::isfinite(host_output[0].y);
#else
  return false;
#endif
}

// Dense Eigen-backed FFT provider used as the portable default.
class EigenFFTProvider final : public IFFTProvider {
 public:
  // Return the stable provider name for diagnostics.
  std::string name() const override { return "eigen"; }

  // Return the stable observability marker name for Eigen fallback paths.
  std::string marker_name() const override { return "fft_provider_eigen"; }

  // Compute a forward FFT using Eigen's portable implementation.
  void forward(std::vector<Complex>& out, const std::vector<Complex>& in) override {
    fft_.fwd(out, in);
  }

  // Compute an inverse FFT using Eigen's portable implementation.
  void inverse(std::vector<Complex>& out, const std::vector<Complex>& in) override {
    fft_.inv(out, in);
  }

 private:
  Eigen::FFT<double> fft_;
};

#if defined(QX_HAS_CUDA)
// cuFFT-backed provider that falls back to Eigen when initialization or a call fails.
class CuFFTProvider final : public IFFTProvider {
 public:
  // Build the provider with an eager Eigen fallback ready for any failure.
  CuFFTProvider() : fallback_(std::make_unique<EigenFFTProvider>()) {}

  // Release any CUDA resources owned by the provider.
  ~CuFFTProvider() override { destroy_plan(); }

  // Return the active provider name, or Eigen if the CUDA path has been disabled.
  std::string name() const override { return active_cufft_ ? "cufft" : fallback_->name(); }

  // Return the active observability marker name.
  std::string marker_name() const override {
    return active_cufft_ ? "fft_provider_cufft" : fallback_->marker_name();
  }

  // Compute the forward FFT using cuFFT when available, otherwise Eigen.
  void forward(std::vector<Complex>& out, const std::vector<Complex>& in) override {
    if (!transform_with_cufft(out, in, false)) {
      fallback_->forward(out, in);
    }
  }

  // Compute the inverse FFT using cuFFT when available, otherwise Eigen.
  void inverse(std::vector<Complex>& out, const std::vector<Complex>& in) override {
    if (!transform_with_cufft(out, in, true)) {
      fallback_->inverse(out, in);
    }
  }

 private:
  // Reset CUDA state and permanently switch this provider to Eigen fallback.
  void disable_cufft() {
    destroy_plan();
    active_cufft_ = false;
  }

  // Destroy any cached cuFFT plan and associated device buffer.
  void destroy_plan() {
    if (device_buffer_ != nullptr) {
      cudaFree(device_buffer_);
      device_buffer_ = nullptr;
    }
    if (plan_ != 0) {
      cufftDestroy(plan_);
      plan_ = 0;
    }
    planned_size_ = 0;
  }

  // Ensure the cached plan matches the requested transform length.
  bool ensure_plan(std::size_t n) {
    if (!active_cufft_) {
      return false;
    }
    if (n == 0) {
      return true;
    }
    if (n == planned_size_ && plan_ != 0 && device_buffer_ != nullptr) {
      return true;
    }

    destroy_plan();

    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      return false;
    }

    const int n_int = static_cast<int>(n);
    const std::size_t bytes = n * sizeof(cufftDoubleComplex);
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&device_buffer_), bytes), "cudaMalloc");
    check_cufft(cufftPlan1d(&plan_, n_int, CUFFT_Z2Z, 1), "cufftPlan1d");
    planned_size_ = n;
    return true;
  }

  // Execute one FFT direction with cuFFT and fall back on the first failure.
  bool transform_with_cufft(std::vector<Complex>& out, const std::vector<Complex>& in, bool inverse) {
    if (!active_cufft_) {
      return false;
    }

    try {
      if (!ensure_plan(in.size())) {
        disable_cufft();
        return false;
      }

      out.resize(in.size());
      if (in.empty()) {
        return true;
      }

      host_buffer_.resize(in.size());
      for (std::size_t i = 0; i < in.size(); ++i) {
        host_buffer_[i].x = in[i].real();
        host_buffer_[i].y = in[i].imag();
      }

      const std::size_t bytes = in.size() * sizeof(cufftDoubleComplex);
      check_cuda(cudaMemcpy(device_buffer_, host_buffer_.data(), bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpyHostToDevice");
      check_cufft(cufftExecZ2Z(plan_, device_buffer_, device_buffer_, inverse ? CUFFT_INVERSE : CUFFT_FORWARD),
                  "cufftExecZ2Z");
      check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
      check_cuda(cudaMemcpy(host_buffer_.data(), device_buffer_, bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpyDeviceToHost");

      const double scale = inverse ? 1.0 / static_cast<double>(in.size()) : 1.0;
      for (std::size_t i = 0; i < in.size(); ++i) {
        out[i] = Complex{host_buffer_[i].x * scale, host_buffer_[i].y * scale};
      }
      return true;
    } catch (const std::exception&) {
      disable_cufft();
      return false;
    }
  }

  std::unique_ptr<EigenFFTProvider> fallback_;
  bool active_cufft_{true};
  std::size_t planned_size_{0};
  cufftHandle plan_{0};
  cufftDoubleComplex* device_buffer_{nullptr};
  std::vector<cufftDoubleComplex> host_buffer_;
};
#endif

}  // namespace

// Report whether a real cuFFT backend is usable on this build and machine.
bool cufft_backend_available() {
#if defined(QX_HAS_CUDA)
  static const bool available = probe_cufft_runtime();
  return available;
#else
  return false;
#endif
}

// Create a provider and use Eigen whenever cuFFT is unavailable or unsupported.
std::unique_ptr<IFFTProvider> create_fft_provider(FFTProviderKind kind) {
  if (kind == FFTProviderKind::CuFFT) {
#if defined(QX_HAS_CUDA)
    if (cufft_backend_available()) {
      return std::make_unique<CuFFTProvider>();
    }
#endif
    return std::make_unique<EigenFFTProvider>();
  }

  if (kind == FFTProviderKind::Auto || kind == FFTProviderKind::Eigen) {
    return std::make_unique<EigenFFTProvider>();
  }

  return std::make_unique<EigenFFTProvider>();
}

}  // namespace qx
