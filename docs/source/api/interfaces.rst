Interfaces & Integrations
==========================

Modules providing interoperability with external frameworks and tools.

PennyLane Plugin
-----------------

The devices are part of the engine now, as ``qml.device("quantx.statevector", wires=n)``
and ``qml.device("quantx.mps", wires=n)``: see :mod:`quantx_engine.pennylane_device`. This
module keeps the former import path of the first.

.. automodule:: quantx_engine.contrib.pennylane_plugin
   :members:
   :show-inheritance:

CUDA-Q Target
-------------

``quantx`` is a CUDA-Q target: an NVQIR circuit simulator (``src/cudaq/``) that runs a CUDA-Q
kernel on the engine's block-sparse MPS. A Python program selects it as any other target::

    cudaq.set_target("quantx")

or, unchanged, with ``CUDAQ_DEFAULT_SIMULATOR=quantx`` in its environment, or ``--target
quantx`` on its command line when it calls ``cudaq.parse_args()``.

It is built against an existing CUDA-Q installation, so far tested on Linux (from pip,
``cudaq-core`` carries the headers and CMake files a simulator builds against), and installed
into it::

    cmake -S . -B build -DQX_ENABLE_CUDAQ=ON -DNVQIR_DIR=<cudaq>/lib/cmake/nvqir
    cmake --build build --target nvqir-quantx
    cmake --install build

The install puts ``libnvqir-quantx`` and the ``libquantx`` it calls beside CUDA-Q's own
simulators, the one place CUDA-Q 0.17 loads a simulator from, and ``quantx.yml`` into
``plugins/quantx/targets`` beside them. From pip, ``NVQIR_DIR`` is ``lib/cmake/nvqir`` under
``site-packages/cudaq_core/cudaq/core``. The target declares the CUDA-Q version it was built
for, which CMake takes from the CUDA-Q that Python imports (``QX_CUDAQ_VERSION`` overrides it);
under another version CUDA-Q warns, as it does for any plugin.

The gates of a kernel are kept until a result is needed, then fused, greedily, into clusters of
at most two qubits that each conserve the number of qubits in ``|1>``. When every cluster closes,
that number is an exact charge: it holds at any truncation, and the tensors split into blocks.
CUDA-Q hands a simulator its gates already decomposed (``exp_pauli`` into basis changes, a CNOT
ladder and an Rz), and the fusion is what recovers the symmetry of, say, a Givens rotation
written that way. When a cluster cannot close within two qubits, the kernel runs on the same
kernels with no charge. X gates on qubits in a known basis state, measurements and resets keep
the charge.

``get_state`` reads the amplitudes, up to 24 qubits; ``observe`` evaluates every term of a spin
operator as a Pauli string in the engine, all at once, and ``sample`` and ``run`` (which CUDA-Q
requires of a kernel that branches on a measurement) draw by perfect sampling, at any width. A gate on one target may have up to eight controls (decomposed into two-qubit gates),
and a SWAP any number; a two-qubit gate with controls, a gate on three or more targets, an
initial state vector and noise models are not supported. The options are environment variables:

``QUANTX_MAX_BOND``
    the bond cap (default: none, so a kernel runs exactly unless told otherwise)
``QUANTX_CUTOFF``
    the weight a split may discard (default ``1e-14``)
``QUANTX_CONSERVE``
    ``auto`` (default); ``1`` refuses a kernel that cannot keep the charge; ``0`` never uses it
``QUANTX_SVD``
    ``direct`` (default) or ``gram``, the faster and less accurate route
``QUANTX_VERBOSE``
    ``1`` reports, on stderr, whether each state used the charge, its largest bond and the
    weight its truncations discarded

``test_cudaq_target.py`` and ``test_cudaq_examples.py`` (in ``python/quantx_engine/tests/``)
check the target against CUDA-Q's own CPU state vector, ``qpp-cpu``, in CI: amplitudes and exact
expectation values to 1e-10, sampled distributions, mid-circuit measurement and reset, gates
with many controls, and the examples of CUDA-Q's documentation; at 40 to 100 qubits, answers
known in closed form.

Qiskit V2 Primitives
---------------------

.. automodule:: quantx_engine.contrib.primitives
   :members:
   :show-inheritance:

Jupyter Widgets
----------------

Interactive visualisation widgets for Jupyter notebooks (requires ``ipywidgets``).

.. automodule:: quantx_engine.contrib.jupyter_widgets
   :members:
   :show-inheritance:
