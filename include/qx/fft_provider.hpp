#pragma once

#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace qx {

// FFT provider selection requested by continuous-space solvers.
enum class FFTProviderKind {
  Auto,
  Eigen,
  CuFFT
};

// Abstract 1D FFT provider used by continuous-space backends.
class IFFTProvider {
 public:
  // Virtual destructor for safe polymorphic cleanup.
  virtual ~IFFTProvider() = default;
  // Stable provider name used by runtime diagnostics.
  virtual std::string name() const = 0;
  // Stable marker name used by observability layers.
  virtual std::string marker_name() const { return "fft_provider_eigen"; }
  // Compute the forward FFT of one complex vector.
  virtual void forward(std::vector<std::complex<double>>& out,
                       const std::vector<std::complex<double>>& in) = 0;
  // Compute the inverse FFT of one complex vector.
  virtual void inverse(std::vector<std::complex<double>>& out,
                       const std::vector<std::complex<double>>& in) = 0;
};

// Report whether a true cuFFT path is usable in the current build/runtime.
bool cufft_backend_available();
// Create an FFT provider with safe fallback to Eigen when needed.
std::unique_ptr<IFFTProvider> create_fft_provider(FFTProviderKind kind);

}  // namespace qx
