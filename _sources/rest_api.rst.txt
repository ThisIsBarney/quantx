REST API
========

QuantX Engine ships a FastAPI REST server that exposes the C++ solver backend and
most Python modules as HTTP endpoints.

Starting the server
--------------------

.. code-block:: bash

   cd quantx/
   uvicorn server.main:app --reload --port 8000

Interactive Swagger UI is available at http://localhost:8000/docs.

Authentication
--------------

When the ``QUANTX_API_KEY`` environment variable is set, all ``/jobs/*``
endpoints require a bearer token:

.. code-block:: http

   Authorization: Bearer <api-key>

Obtain a token via ``POST /auth/token`` (see ``/docs`` for the schema).

Core endpoints
--------------

Meta
~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 15 50

   * - Method
     - Path
     - Description
   * - GET
     - ``/health``
     - Server health, DB and cache status
   * - GET
     - ``/ready``
     - Kubernetes readiness probe (503 if backend not connected)
   * - GET
     - ``/version``
     - SDK version + native library version/availability

Solvers
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 20 50

   * - Method
     - Path
     - Description
   * - GET
     - ``/solvers``
     - List all 37 named C++ physics solvers
   * - POST
     - ``/solve/{solver}``
     - Run a named solver; body is solver-specific JSON options
   * - POST
     - ``/run``
     - Legacy Hamiltonian simulation (problem/options schema)

Circuit simulation
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 25 45

   * - Method
     - Path
     - Description
   * - POST
     - ``/circuit``
     - Run a gate-model circuit (JSON circuit schema)
   * - GET
     - ``/circuit/capabilities``
     - Gate set metadata
   * - POST
     - ``/circuit/validate``
     - Validate a circuit payload without running it
   * - POST
     - ``/problem/validate``
     - Validate a problem payload

Async job queue (M103)
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 25 45

   * - Method
     - Path
     - Description
   * - POST
     - ``/jobs``
     - Submit a solver job (returns ``job_id``)
   * - GET
     - ``/jobs/{job_id}``
     - Poll job status (pending / running / done / failed)
   * - DELETE
     - ``/jobs/{job_id}``
     - Cancel a pending job
   * - GET
     - ``/cache/{cache_key}``
     - Retrieve a cached result by SHA-256 key

Quantum network (M96)
~~~~~~~~~~~~~~~~~~~~~~

``/network/*`` — see ``/docs`` for full schema.

Quantum protocols (M97)
~~~~~~~~~~~~~~~~~~~~~~~~

``/protocols/*`` — BB84, teleportation, entanglement swapping, superdense coding.

Pulse-level simulation (M89)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``/pulse/*`` — pulse schedule submission and waveform results.

Advanced solvers (M91–M95)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``/advanced/*`` — Floquet, TD-DMRG, noise-aware VQE, fault-tolerant T-gate synthesis.

Observability (M107)
~~~~~~~~~~~~~~~~~~~~~

``/metrics`` — Prometheus-format scrape endpoint.
