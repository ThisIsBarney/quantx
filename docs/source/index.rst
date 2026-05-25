QuantX Engine Documentation
============================

**QuantX Engine** is a high-performance quantum simulation SDK that exposes 130+ physics
modules — from gate-model circuits and VQE to many-body Hamiltonians, fault-tolerant
architectures, quantum networks, and machine learning.

.. image:: https://img.shields.io/pypi/v/quantx-engine
   :target: https://pypi.org/project/quantx-engine/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/python-3.10%2B-blue
   :alt: Python 3.10+

.. image:: https://img.shields.io/badge/license-MIT-green
   :alt: MIT License

.. image:: https://github.com/ThisIsBarney/quantx/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/ThisIsBarney/quantx/actions/workflows/docs.yml
   :alt: Docs

----

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart
   changelog

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: REST API

   rest_api

----

Features at a Glance
---------------------

.. rubric:: Core Simulation

* :class:`~quantx_engine.Engine` — C++ physics solver dispatcher (37 named solvers)
* :class:`~quantx_engine.lanczos.LanczosSolver` — sparse Hamiltonian eigensolver
* :class:`~quantx_engine.trotter.TrotterSimulator` — product-formula time evolution
* :class:`~quantx_engine.mps_simulator.MPSSimulator` — matrix-product-state simulator (TEBD)

.. rubric:: Quantum Error Correction

* :class:`~quantx_engine.qec.RepetitionCode` — classical repetition code
* :class:`~quantx_engine.fault_tolerant.SurfaceCode` — surface code threshold analysis
* :class:`~quantx_engine.qldpc.QLDPCCode` — quantum LDPC / bicycle codes (M119)
* :class:`~quantx_engine.bosonic.BosonicQubit` — bosonic cat/GKP qubits (M120)

.. rubric:: Algorithms

* :class:`~quantx_engine.qaoa_optim.CVaRQAOA` — CVaR-QAOA optimizer
* :class:`~quantx_engine.advanced_chemistry.VQE` — noise-aware variational eigensolver
* :class:`~quantx_engine.qsp_qsvt.QSPCircuit` — quantum signal processing / QSVT
* :class:`~quantx_engine.quantum_walk.ContinuousTimeQuantumWalk` — CTQW simulator

.. rubric:: Quantum Networking

* :class:`~quantx_engine.quantum_network.QuantumNetwork` — entanglement-based network
* :class:`~quantx_engine.quantum_protocols.BB84Protocol` — BB84 key distribution
* :class:`~quantx_engine.quantum_protocols.QuantumTeleportation` — teleportation with Werner states

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
