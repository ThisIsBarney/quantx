"""M14 from C++: programs compiled by ``nvq++ --target quantx``, against ``--target qpp-cpu``.

nvq++ takes a target from outside CUDA-Q's catalogue only as a compiled description, which the
build makes with cudaq-target-db-gen. Both ship with CUDA-Q's C++ toolchain (its installer and its
container), not with the pip packages, so these tests run where nvq++ is on PATH and skip
elsewhere. CI runs them in CUDA-Q's container (scripts/cudaq_cpp_check.sh) with
QX_REQUIRE_NVQPP=1, where a missing nvq++ or target fails instead.

One program holds the kernels and is compiled once per target; its arguments pick what it prints.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

import numpy as np
import pytest

NVQPP = shutil.which("nvq++")
REQUIRED = os.environ.get("QX_REQUIRE_NVQPP") == "1"
REFERENCE = "qpp-cpu"
TOL = 1e-10

if NVQPP is None:
    if REQUIRED:
        raise RuntimeError("QX_REQUIRE_NVQPP=1, and nvq++ is not on PATH")
    pytest.skip("nvq++, CUDA-Q's C++ compiler, is not on PATH", allow_module_level=True)

PROGRAM = r"""
#include <cudaq.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct ghz {
  void operator()(int n) __qpu__ {
    cudaq::qvector q(n);
    h(q[0]);
    for (int i = 0; i + 1 < n; i++)
      x<cudaq::ctrl>(q[i], q[i + 1]);
    mz(q);
  }
};

// Every kind of gate, on near and far pairs, so that nothing is conserved.
struct scrambled {
  void operator()(int n) __qpu__ {
    cudaq::qvector q(n);
    for (int i = 0; i < n; i++) {
      h(q[i]);
      rz(0.3 * (i + 1), q[i]);
    }
    for (int i = 0; i + 1 < n; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
      ry(0.7 + 0.1 * i, q[i + 1]);
    }
    x<cudaq::ctrl>(q[0], q[n - 1]);
    s(q[1]);
    t(q[2]);
    r1(0.4, q[3]);
    swap(q[0], q[2]);
    rx(1.1, q[n - 1]);
    x<cudaq::ctrl>(q[0], q[1], q[2]);
  }
};

// A Neel state evolved by Trotter steps, one exp_pauli per term of the Hamiltonian, as CUDA-Q's
// documentation writes it. An XX or a YY term alone does not conserve the number of qubits in
// |1>; the target fuses each bond's terms into a gate that does.
struct trotter {
  void operator()(int n, int steps, std::vector<double> &coefficients,
                  std::vector<cudaq::pauli_word> &words, double dt) __qpu__ {
    cudaq::qvector q(n);
    for (int i = 1; i < n; i += 2)
      x(q[i]);
    for (int k = 0; k < steps; k++)
      for (std::size_t i = 0; i < coefficients.size(); i++)
        cudaq::exp_pauli(coefficients[i] * dt, q, words[i]);
  }
};

static cudaq::spin_op xxz(int n, double delta) {
  auto h = cudaq::spin_op::empty();
  for (int i = 0; i + 1 < n; i++) {
    h += cudaq::spin::x(i) * cudaq::spin::x(i + 1);
    h += cudaq::spin::y(i) * cudaq::spin::y(i + 1);
    h += delta * cudaq::spin::z(i) * cudaq::spin::z(i + 1);
  }
  return h;
}

static cudaq::spin_op number(int n) {
  auto op = cudaq::spin_op::empty();
  for (int i = 0; i < n; i++)
    op += 0.5 * (cudaq::spin::i(i) - cudaq::spin::z(i));
  return op;
}

int main(int argc, char **argv) {
  const std::string what = argc > 1 ? argv[1] : "";
  const int n = argc > 2 ? std::atoi(argv[2]) : 6;
  const int steps = argc > 3 ? std::atoi(argv[3]) : 4;
  auto h = xxz(n, 0.5);
  std::vector<double> coefficients;
  std::vector<cudaq::pauli_word> words;
  for (const auto &term : h) {
    coefficients.push_back(term.evaluate_coefficient().real());
    words.push_back(term.get_pauli_word(n));
  }
  const double dt = 0.1;
  if (what == "scrambled" || what == "trotter") {
    // The amplitudes, "re im" a line, in CUDA-Q's order.
    auto state = what == "scrambled"
                     ? cudaq::get_state(scrambled{}, n)
                     : cudaq::get_state(trotter{}, n, steps, coefficients, words, dt);
    for (std::size_t i = 0; i < (std::size_t{1} << n); i++)
      std::printf("%.17g %.17g\n", state[i].real(), state[i].imag());
  } else if (what == "observe") {
    // The energy, and the number of qubits in |1>, after the steps.
    const double energy =
        cudaq::observe(trotter{}, h, n, steps, coefficients, words, dt).expectation();
    const double count =
        cudaq::observe(trotter{}, number(n), n, steps, coefficients, words, dt).expectation();
    std::printf("%.17g %.17g\n", energy, count);
  } else if (what == "sample") {
    cudaq::set_random_seed(13);
    auto counts = cudaq::sample(2000, ghz{}, n);
    for (const auto &[bits, count] : counts)
      std::printf("%s %zu\n", bits.c_str(), count);
  } else {
    std::fprintf(stderr, "usage: %s scrambled|trotter|observe|sample n [steps]\n", argv[0]);
    return 2;
  }
  return 0;
}
"""


@pytest.fixture(scope="module")
def program(tmp_path_factory):
    """The program, compiled for quantx and for the reference."""
    where = tmp_path_factory.mktemp("cudaq_cpp")
    source = where / "kernels.cpp"
    source.write_text(PROGRAM)
    built = {}
    for target in ("quantx", REFERENCE):
        exe = where / f"kernels-{target}"
        done = subprocess.run([NVQPP, "--target", target, str(source), "-o", str(exe)],
                              capture_output=True, text=True, timeout=900)
        if done.returncode != 0:
            pytest.fail(f"nvq++ --target {target} failed:\n{done.stdout}\n{done.stderr}")
        # The target declares the CUDA-Q it was built for, and under another one nvq++ warns.
        assert "was built for CUDA-Q" not in done.stdout + done.stderr, done.stdout + done.stderr
        built[target] = exe
    return built


def run(program, target, *args, **options):
    """Run the program for target with QUANTX_* options; return its stdout and stderr."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("QUANTX_")}
    env.update({f"QUANTX_{k.upper()}": str(v) for k, v in options.items()})
    done = subprocess.run([str(program[target]), *map(str, args)], capture_output=True,
                          text=True, env=env, timeout=900)
    assert done.returncode == 0, f"{target} {args}: {done.stderr}"
    return done.stdout, done.stderr


def amplitudes(text):
    values = np.array([[float(x) for x in line.split()] for line in text.splitlines()])
    return values[:, 0] + 1j * values[:, 1]


def charge_reports(stderr):
    return re.findall(r"the charge (used|not used)", stderr)


def test_nvqpp_lists_the_target():
    done = subprocess.run([NVQPP, "--list-targets"], capture_output=True, text=True, timeout=120)
    assert "quantx" in done.stdout.split(), done.stdout + done.stderr


@pytest.mark.parametrize("kernel, n", [("scrambled", 6), ("trotter", 8)])
def test_state_matches_the_reference(program, kernel, n):
    got, err = run(program, "quantx", kernel, n, 4, verbose=1)
    want, _ = run(program, REFERENCE, kernel, n, 4)
    got, want = amplitudes(got), amplitudes(want)
    assert got.shape == want.shape == (2 ** n,)
    assert np.max(np.abs(got - want)) < TOL
    # And it ran on quantx: the Trotter steps with the charge, the scrambled circuit without.
    assert charge_reports(err) == ["used" if kernel == "trotter" else "not used"]


def test_observe_matches_the_reference_and_keeps_the_charge(program):
    got, err = run(program, "quantx", "observe", 10, 5, verbose=1)
    want, _ = run(program, REFERENCE, "observe", 10, 5)
    (energy, count), (energy_ref, count_ref) = map(float, got.split()), map(float, want.split())
    assert abs(energy - energy_ref) < TOL
    assert abs(count - count_ref) < TOL and abs(count - 5) < TOL
    assert set(charge_reports(err)) == {"used"}


def test_number_exact_at_a_bond_cap_past_the_state_vector(program):
    # 32 qubits under a cap of 8: the number of qubits in |1> stays 16 exactly, by the charge.
    out, err = run(program, "quantx", "observe", 32, 10, max_bond=8, verbose=1)
    energy, count = map(float, out.split())
    assert abs(count - 16) < TOL
    assert set(charge_reports(err)) == {"used"}
    discarded = [float(x) for x in re.findall(r"truncation ([0-9.eE+-]+)", err)]
    assert discarded and max(discarded) > 1e-8      # the cap binds


@pytest.mark.parametrize("n", [20, 64])
def test_ghz_samples(program, n):
    out, _ = run(program, "quantx", "sample", n)
    counts = {bits: int(c) for bits, c in (line.split() for line in out.splitlines())}
    assert set(counts) <= {"0" * n, "1" * n} and sum(counts.values()) == 2000
    assert abs(counts.get("0" * n, 0) - 1000) < 5 * np.sqrt(500)
    if n <= 20:
        out, _ = run(program, REFERENCE, "sample", n)
        assert {line.split()[0] for line in out.splitlines()} == set(counts)
