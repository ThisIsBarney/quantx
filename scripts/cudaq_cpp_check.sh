#!/usr/bin/env bash
# M14 in CUDA-Q's container: build the CUDA-Q target "quantx", install it into the CUDA-Q there,
# and run python/quantx_engine/tests/test_cudaq_cpp.py, which compiles programs with
# nvq++ --target quantx and --target qpp-cpu and compares what they print. nvq++ is CUDA-Q's C++
# toolchain, which the pip packages leave out; the container has it. The container's Python is
# the same CUDA-Q installation, so test_cudaq_target.py and test_cudaq_examples.py run there too,
# on the compiled description of the target alone. CI's CUDA-Q job runs this; by hand, from the
# repository's root:
#
#   docker run --rm --user root -v "$PWD":/src -w /src \
#     nvcr.io/nvidia/quantum/cuda-quantum:cu12-0.17.0 scripts/cudaq_cpp_check.sh
#
# (the container's entry point runs its command with bash). The build goes to /tmp, outside the
# checkout. QX_JOBS sets the build's parallelism (default: every core).
set -euo pipefail

cudaq="${CUDA_QUANTUM_PATH:-/opt/nvidia/cudaq}"
export PATH="$cudaq/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
nvq++ --version

export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q --no-install-recommends cmake ninja-build g++ git ca-certificates
# numpy and scipy for the quantx_engine package the tests sit in; a Python that refuses a
# system-wide install (PEP 668) gets it anyway: the container is thrown away.
python3 -m pip install -q numpy scipy pytest \
  || python3 -m pip install -q --break-system-packages numpy scipy pytest

nvqir_config="$(find "$cudaq" -name NVQIRConfig.cmake | head -n1)"
test -n "$nvqir_config"
cmake -S . -B /tmp/quantx-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DQX_ENABLE_CUDA=OFF \
  -DQX_ENABLE_CUDAQ=ON \
  -DNVQIR_DIR="$(dirname "$nvqir_config")"
cmake --build /tmp/quantx-build --target nvqir-quantx --parallel "${QX_JOBS:-$(nproc)}"
cmake --install /tmp/quantx-build
ls -l "$cudaq/plugins/quantx/targets"

QX_REQUIRE_NVQPP=1 python3 -m pytest -p no:cacheprovider \
  python/quantx_engine/tests/test_cudaq_cpp.py
QX_REQUIRE_CUDAQ=1 python3 -m pytest -p no:cacheprovider \
  python/quantx_engine/tests/test_cudaq_target.py \
  python/quantx_engine/tests/test_cudaq_examples.py
