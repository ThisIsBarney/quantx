#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "qx/fft_provider.hpp"
#include "test_support.hpp"

namespace {

using Complex = std::complex<double>;

// Minimal assertion helper for the FFT provider smoke test.
void expect(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    std::exit(1);
  }
}

// Measure the maximum absolute difference between two complex vectors.
double max_abs_diff(const std::vector<Complex>& a, const std::vector<Complex>& b) {
  if (a.size() != b.size()) {
    return std::numeric_limits<double>::infinity();
  }
  double max_diff = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
  }
  return max_diff;
}

// Verify that a provider round-trips a small complex signal.
void verify_round_trip(qx::IFFTProvider& provider) {
  const std::vector<Complex> input = {
      {1.0, 0.0},
      {0.5, -1.0},
      {-0.25, 0.75},
      {2.0, 0.125},
  };

  std::vector<Complex> spectrum;
  std::vector<Complex> output;
  provider.forward(spectrum, input);
  provider.inverse(output, spectrum);

  expect(output.size() == input.size(), "FFT round-trip should preserve size");
  expect(max_abs_diff(input, output) < 1e-9, "FFT round-trip should preserve values");
}

}  // namespace

// Smoke test for FFT provider selection and fallback semantics.
int main() {
  qx::test::configure_console_diagnostics();
  try {
    const auto eigen = qx::create_fft_provider(qx::FFTProviderKind::Eigen);
    expect(eigen != nullptr, "Eigen provider should construct");
    expect(eigen->name() == "eigen", "Eigen provider should report eigen name");
    expect(eigen->marker_name() == "fft_provider_eigen", "Eigen provider should report eigen marker");
    verify_round_trip(*eigen);

    const auto auto_provider = qx::create_fft_provider(qx::FFTProviderKind::Auto);
    expect(auto_provider != nullptr, "Auto provider should construct");
    expect(auto_provider->name() == "eigen", "Auto provider should keep the portable Eigen path");
    expect(auto_provider->marker_name() == "fft_provider_eigen", "Auto provider should report eigen marker");
    verify_round_trip(*auto_provider);

    const auto requested_cufft = qx::create_fft_provider(qx::FFTProviderKind::CuFFT);
    expect(requested_cufft != nullptr, "CuFFT request should construct");
    verify_round_trip(*requested_cufft);

    if (qx::cufft_backend_available()) {
      expect(requested_cufft->name() == "cufft", "Available cuFFT backend should stay on cufft");
      expect(requested_cufft->marker_name() == "fft_provider_cufft",
             "Available cuFFT backend should report cufft marker");
    } else {
      expect(requested_cufft->name() == "eigen", "Unavailable cuFFT backend should fall back to eigen");
      expect(requested_cufft->marker_name() == "fft_provider_eigen",
             "Unavailable cuFFT backend should report eigen marker");
    }

    std::cout << "qx_fft_smoke passed\n";
    return 0;
  } catch (const std::exception& ex) {
    return qx::test::report_exception(ex);
  } catch (...) {
    return qx::test::report_unknown_exception();
  }
}
