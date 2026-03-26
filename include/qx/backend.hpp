#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "qx/types.hpp"

namespace qx {

// Backend interface shared by statevector and tensor-network implementations.
class IBackend {
 public:
  // Virtual destructor for polymorphic backend cleanup.
  virtual ~IBackend() = default;
  // Stable backend identifier (for logs/selection results).
  virtual std::string name() const = 0;
  // Static backend capability descriptor.
  virtual BackendCapabilities capabilities() const = 0;
  // Initialize backend state before time evolution.
  virtual void initialize(const QuantumProblem& problem, const RunOptions& options) = 0;
  // Execute time evolution for the configured problem.
  virtual void evolve(const QuantumProblem& problem, const RunOptions& options) = 0;
  // Return observables/diagnostics collected after evolution.
  virtual std::map<std::string, double> measure(const QuantumProblem& problem) = 0;
  // Persist backend state for resumable execution.
  virtual void checkpoint(const std::string& path) = 0;
  // Restore backend state from checkpoint directory if present.
  virtual void restore(const std::string& path) = 0;
  // Return dense state if available; empty for compressed backends.
  virtual std::vector<std::complex<double>> current_state() const = 0;
  // Return continuous density frame metadata; empty for non-continuous backends.
  virtual std::vector<ContinuousDensityFrame> continuous_density_frames() const = 0;
  // Return Lindblad density-matrix frame metadata; empty for non-Lindblad backends.
  virtual std::vector<LindbladFrame> lindblad_frames() const { return {}; }
};

using BackendFactory = std::function<std::unique_ptr<IBackend>()>;

// Process-wide backend registry with plugin-friendly registration API.
class BackendRegistry {
 public:
  // Access singleton registry instance.
  static BackendRegistry& instance();
  // Register backend factory + capabilities under a unique name.
  void register_backend(const std::string& name, BackendFactory factory, BackendCapabilities capabilities);
  // Return all registered backend names.
  std::vector<std::string> names() const;
  // Construct backend instance by name.
  std::unique_ptr<IBackend> create(const std::string& name) const;
  // Query static capabilities for a backend by name.
  BackendCapabilities capabilities_for(const std::string& name) const;

 private:
  // Internal registry record.
  struct Record {
    BackendFactory factory;
    BackendCapabilities capabilities;
  };
  std::map<std::string, Record> records_;
};

// Register built-in backends (statevector/mps).
void register_builtin_backends();

}  // namespace qx
