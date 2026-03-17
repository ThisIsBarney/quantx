from quantx_engine import Engine


def main():
  engine = Engine()
  problem = {
      "qubits": 12,
      "equation": "Schrodinger",
      "total_time": 1.0,
      "dt": 0.01,
      "method": "trotter",
      "target_error": 1e-8,
      "expected_circuit_depth": 32,
      "noisy": False,
  }
  options = {
      "use_gpu": True,
      "allow_mpi": True,
      "enable_checkpoint": False,
      "max_threads": 8,
      "mpi_ranks": 1,
  }
  print("explain:", engine.explain(problem, options))
  print("run:", engine.run(problem, options))

  rabi = {
      "qubits": 1,
      "equation": "Schrodinger",
      "total_time": 1.57079632679,
      "dt": 0.01,
      "method": "krylov",
      "hamiltonian_terms": [{"label": "X0", "coefficient": 1.0}],
  }
  print("rabi:", engine.run(rabi, {"enable_checkpoint": False})["observables"])

  xx = {
      "qubits": 2,
      "equation": "Schrodinger",
      "total_time": 1.57079632679,
      "dt": 0.005,
      "method": "trotter",
      "hamiltonian_terms": [{"label": "X0X1", "coefficient": 1.0}],
  }
  print("xx:", engine.run(xx, {"enable_checkpoint": False})["observables"])


if __name__ == "__main__":
  main()
