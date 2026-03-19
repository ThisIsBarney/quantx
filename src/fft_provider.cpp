#include "qx/fft_provider.hpp"

#include <unsupported/Eigen/FFT>

namespace qx {
namespace {

class EigenFFTProvider final : public IFFTProvider {
 public:
  std::string name() const override { return "eigen"; }

  void forward(std::vector<std::complex<double>>& out,
               const std::vector<std::complex<double>>& in) override {
    fft_.fwd(out, in);
  }

  void inverse(std::vector<std::complex<double>>& out,
               const std::vector<std::complex<double>>& in) override {
    fft_.inv(out, in);
  }

 private:
  Eigen::FFT<double> fft_;
};

class CuFFTProvider final : public IFFTProvider {
 public:
  std::string name() const override { return "cufft"; }

  void forward(std::vector<std::complex<double>>& out,
               const std::vector<std::complex<double>>& in) override {
    fft_.fwd(out, in);
  }

  void inverse(std::vector<std::complex<double>>& out,
               const std::vector<std::complex<double>>& in) override {
    fft_.inv(out, in);
  }

 private:
  Eigen::FFT<double> fft_;
};

}  // namespace

std::unique_ptr<IFFTProvider> create_fft_provider(FFTProviderKind kind) {
  if (kind == FFTProviderKind::CuFFT) {
    return std::make_unique<CuFFTProvider>();
  }
  if (kind == FFTProviderKind::Auto || kind == FFTProviderKind::Eigen) {
    return std::make_unique<EigenFFTProvider>();
  }
  return std::make_unique<EigenFFTProvider>();
}

}  // namespace qx
