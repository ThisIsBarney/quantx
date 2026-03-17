#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "qx/types.hpp"

namespace qx {

class IBackend {
 public:
  virtual ~IBackend() = default;
  virtual std::string name() const = 0;
  virtual BackendCapabilities capabilities() const = 0;
  virtual void initialize(const QuantumProblem& problem, const RunOptions& options) = 0;
  virtual void evolve(const QuantumProblem& problem, const RunOptions& options) = 0;
  virtual std::map<std::string, double> measure(const QuantumProblem& problem) = 0;
  virtual void checkpoint(const std::string& path) = 0;
  virtual void restore(const std::string& path) = 0;
  virtual std::vector<std::complex<double>> current_state() const = 0;
};

using BackendFactory = std::function<std::unique_ptr<IBackend>()>;

class BackendRegistry {
 public:
  static BackendRegistry& instance();
  void register_backend(const std::string& name, BackendFactory factory, BackendCapabilities capabilities);
  std::vector<std::string> names() const;
  std::unique_ptr<IBackend> create(const std::string& name) const;
  BackendCapabilities capabilities_for(const std::string& name) const;

 private:
  struct Record {
    BackendFactory factory;
    BackendCapabilities capabilities;
  };
  std::map<std::string, Record> records_;
};

void register_builtin_backends();

}  // namespace qx
