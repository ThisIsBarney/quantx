Installation
============

Requirements
------------

* Python 3.10 or newer
* NumPy ≥ 1.24

Pure-Python install (no C++ build required)
-------------------------------------------

.. code-block:: bash

   pip install quantx-engine

This installs all pure-Python modules (130+ physics modules).  The native C++
engine (``Engine`` class) is **optional** — all other modules work without it.

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Extra
     - What it adds
     - Install command
   * - ``pennylane``
     - :class:`~quantx_engine.pennylane_plugin.QuantXDevice` plugin for PennyLane
     - ``pip install quantx-engine[pennylane]``
   * - ``gpu``
     - GPU-accelerated statevector via CuPy (CUDA 12)
     - ``pip install quantx-engine[gpu]``
   * - ``distributed``
     - Distributed simulation via Ray
     - ``pip install quantx-engine[distributed]``
   * - ``all``
     - All optional deps
     - ``pip install quantx-engine[all]``

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/ThisIsBarney/quantx.git
   cd quantx/python
   pip install -e ".[dev]"
   pytest                     # runs the full test suite

Native C++ Engine (optional)
-----------------------------

The :class:`~quantx_engine.Engine` class wraps a native C++ shared library that
implements 37 physics solvers.  To use it, set the environment variable
``QUANTX_LIB_PATH`` to the compiled ``libquantx.so`` / ``quantx.dll`` path:

.. code-block:: bash

   export QUANTX_LIB_PATH=/path/to/libquantx.so
   python -c "from quantx_engine import Engine; e = Engine(); print(e.version())"

If the library cannot be found, a
:class:`~quantx_engine.QuantXNativeUnavailable` exception is raised.
All pure-Python modules are unaffected.
