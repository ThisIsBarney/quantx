Installation
============

.. code-block:: bash

   pip install quantx-engine

QuantX needs Python 3.10 or later. The wheels carry the native engine for:

* Linux x86_64 (``manylinux_2_28``), with OpenMP;
* macOS 10.15 or later, on arm64 and x86_64, without OpenMP;
* Windows AMD64.

One wheel per platform serves every CPython 3. The wheels are CPU-only. Check that the
engine loaded:

.. code-block:: python

   import quantx_engine
   quantx_engine.backend_info()["native"]     # True with one of the wheels above

On any other platform pip builds the sdist instead, a pure-Python package without the
engine. ``backend_info()["native"]`` is then ``False``, and the engine functions say so
when called: :func:`~quantx_engine.dmrg.ground_state` raises
:class:`~quantx_engine.dmrg.DMRGError` and :mod:`quantx_engine.hva` raises
:class:`~quantx_engine.hva.HVAError`, each asking for a platform wheel or
``QUANTX_LIB_PATH``. The experimental modules work either way.

:mod:`quantx_engine.thermal` came after 0.1.0, the release on PyPI. Until the next
release it needs a build from source (below).

Optional extras
---------------

These serve the experimental modules.

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Extra
     - What it adds
     - Install command
   * - ``pennylane``
     - the :class:`~quantx_engine.contrib.pennylane_plugin.QuantXDevice` plugin for PennyLane
     - ``pip install "quantx-engine[pennylane]"``
   * - ``gpu``
     - CuPy (CUDA 12), for the GPU statevector of :mod:`quantx_engine.contrib.gpu`
     - ``pip install "quantx-engine[gpu]"``
   * - ``distributed``
     - Ray, for distributed runs
     - ``pip install "quantx-engine[distributed]"``
   * - ``all``
     - all of the above
     - ``pip install "quantx-engine[all]"``

Building from source
--------------------

The engine builds with CMake 3.20 or later and a C++20 compiler. On Linux and macOS:

.. code-block:: bash

   git clone https://github.com/ThisIsBarney/quantx.git
   cd quantx
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel
   pip install -e "python/[dev]"

Then point Python at the library, on Linux:

.. code-block:: bash

   export QUANTX_LIB_PATH="$PWD/build/Release/libquantx.so"

and on macOS:

.. code-block:: bash

   export QUANTX_LIB_PATH="$PWD/build/Release/libquantx.dylib"

On Windows, with Visual Studio 2022 (64-bit, which a 64-bit Python needs):

.. code-block:: powershell

   git clone https://github.com/ThisIsBarney/quantx.git
   cd quantx
   cmake -S . -B build -A x64
   cmake --build build --config Release --parallel
   pip install -e "python/[dev]"
   $env:QUANTX_LIB_PATH = "$PWD\build\Release\quantx.dll"

``-DQX_ENABLE_BLAS=ON`` links OpenBLAS/LAPACKE for faster large-block SVDs. It also
needs ``-DQX_BLAS_INCLUDE_DIR=<the directory with lapacke.h>`` and
``-DQX_BLAS_LIB=<the OpenBLAS library>``; configuring stops with an error without
them. ``-DQX_USE_CUDA=ON`` builds the cuSOLVER GPU path (it needs the CUDA toolkit;
``-DCMAKE_CUDA_ARCHITECTURES`` defaults to 89). The
`contributing guide <https://github.com/ThisIsBarney/quantx/blob/master/CONTRIBUTING.md>`_
covers the platforms and how to run the tests.
