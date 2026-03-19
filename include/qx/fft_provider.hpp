#pragma once

#include <complex>
#include <memory>
#include <string>
#include <vector>

namespace qx {

enum class FFTProviderKind {
  Auto,
  Eigen,
  CuFFT
};

class IFFTProvider {
 public:
  virtual ~IFFTProvider() = default;
  virtual std::string name() const = 0;
  virtual void forward(std::vector<std::complex<double>>& out,
                       const std::vector<std::complex<double>>& in) = 0;
  virtual void inverse(std::vector<std::complex<double>>& out,
                       const std::vector<std::complex<double>>& in) = 0;
};

std::unique_ptr<IFFTProvider> create_fft_provider(FFTProviderKind kind);

}  // namespace qx
