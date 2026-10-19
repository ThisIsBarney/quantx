"""QuantX: a tensor-network engine for quantum many-body physics.

The engine's Python API:

* :mod:`quantx_engine.models`: the lattice models;
* :mod:`quantx_engine.dmrg`: ground states, by :func:`~quantx_engine.dmrg.ground_state`;
* :mod:`quantx_engine.hva`: fermionic HVA circuits;
* :mod:`quantx_engine.thermal`: finite temperature;
* :mod:`quantx_engine.u1_circuit`: charge-conserving qubit circuits, read from OpenQASM;
* :mod:`quantx_engine.tdvp` and :mod:`quantx_engine.structure_factor`: zero-temperature
  dynamics of the XXZ chain;
* :class:`Engine`: the native library itself.

Everything else is experimental and lives in :mod:`quantx_engine.contrib`. The old paths,
such as ``import quantx_engine.qec`` or ``from quantx_engine import SurfaceCode``, still
work until 0.4.0, with a ``DeprecationWarning``.
"""

import importlib as _importlib
import importlib.abc as _abc
import importlib.util as _util
import sys as _sys
import warnings as _warnings

# The only place the version is written: pyproject.toml reads it from here.
__version__ = "0.2.0"


def backend_info() -> dict:
    """Return a capability summary for the active QuantX backend.

    Checks whether the native C++ shared library is loaded (requires
    ``QUANTX_LIB_PATH`` to point to a compiled ``libquantx.so`` /
    ``quantx.dll``), and returns a dict describing available features.

    Returns
    -------
    dict
        Keys:

        ``native`` : bool
            ``True`` when the native C++ engine is available and loaded.
        ``gpu`` : bool
            ``True`` when the native engine was built with CUDA/OpenCL support.
        ``version`` : str
            Version string (native engine version when available, else
            ``"pure-python-<__version__>"``).
        ``solvers`` : list[str]
            Named physics solvers exposed by the native library (empty for
            pure-Python mode).
        ``max_qubits`` : int | None
            Practical qubit limit reported by the native circuit backend, or
            ``None`` in pure-Python mode.

    Examples
    --------
    >>> from quantx_engine import backend_info
    >>> info = backend_info()
    >>> info["native"]
    False  # unless QUANTX_LIB_PATH is set and the library is compiled
    """
    try:
        from quantx_engine.engine import Engine
        engine = Engine()
        try:
            version_str = engine.version()
        except Exception:
            version_str = __version__
        try:
            solvers_raw = engine.list_solvers()
        except Exception:
            solvers_raw = []
        try:
            cap = engine.circuit_capabilities()
            gpu = cap.get("gpu", False)
            max_qubits = cap.get("max_qubits", None)
        except Exception:
            gpu = False
            max_qubits = None
        return {
            "native": True,
            "gpu": bool(gpu),
            "version": str(version_str),
            "solvers": list(solvers_raw),
            "max_qubits": max_qubits,
        }
    except Exception:
        return {
            "native": False,
            "gpu": False,
            "version": f"pure-python-{__version__}",
            "solvers": [],
            "max_qubits": None,
        }


from .engine import CircuitError, Engine, QuantXNativeUnavailable  # noqa: E402

__all__ = ["__version__", "backend_info", "Engine", "CircuitError", "QuantXNativeUnavailable"]


# ── The old paths of the experimental modules ───────────────────────────────────────────
# They moved to quantx_engine.contrib in 0.2.0. Until _REMOVED_IN, `import quantx_engine.qec`
# gives quantx_engine.contrib.qec itself (the same module object), and the names this
# package used to export resolve to the contrib objects; each warns once per call site.

_REMOVED_IN = "0.4.0"

_MOVED_MODULES = frozenset((
    "adapters advanced_chemistry bosonic cached_dmrg chemistry circuit clifford_sim clifford_t "
    "crosstalk dynamic dyndecouple excited_state fault_tolerant finite_size floquet ft_tgate "
    "ftqc_pipeline gpu hardware ibm jupyter_widgets lanczos lindblad many_body mbqc mclachlan "
    "mitigation mps_simulator neqdyn noise noise_char noise_vqe oovqe optimal_control "
    "pennylane_plugin primitives pulse qaoa_optim qasm qbm qec qiskit_compat qldpc qml qsp_qsvt "
    "quantum_channel quantum_kernel quantum_network quantum_protocols quantum_walk ray_sim rb_qv "
    "rydberg spectral statevector sweep tdmrg templates tensor_network tomography transpiler "
    "trotter xeb "
).split())

# The names this package exported before 0.2.0, by the contrib module that defines each.
_OLD_EXPORTS = {
    "adapters": "AdapterError to_cirq from_cirq to_braket from_braket",
    "advanced_chemistry": (
        "AdvChemError build_hamiltonian_matrix hf_state UCCSDResult UCCSDSolver ADAPTVQEResult "
        "ADAPTVQESolver QAOAResult QAOASolver QITEResult QITESolver"
    ),
    "bosonic": "BosonicError CatNoiseModel CatQubit GKPState GKPDecoder",
    "chemistry": "PauliTerm PauliHamiltonian jordan_wigner MolecularVQE ChemistryError",
    "circuit": "QuantumCircuit CircuitResult Symbol",
    "clifford_sim": "CliffordError StabilizerState CliffordSimulator StimCircuit StimCircuitError",
    "clifford_t": "GateSequence CliffordGroup SolovayKitaev TCountOptimizer CliffordTError",
    "crosstalk": (
        "CrosstalkError CouplingGraph CrosstalkModel zz_routing_cost cheapest_path echo_sequence"
    ),
    "dynamic": "DynamicCircuitError DynamicCircuit DynamicResult DynamicSimulator",
    "dyndecouple": "DDError DDSequence DynamicalDecoupling QuantumMemory",
    "excited_state": (
        "ExcitedStateError ExcitedStateResult ExcitedStateVQE ConicalIntersectionResult "
        "detect_conical_intersections pyscf_to_pauli_hamiltonian"
    ),
    "fault_tolerant": (
        "FTError LogicalQubit MagicStateDistillery DistillationResult FaultTolerantCircuit "
        "logical_error_rate_vs_distance t_gate_cost"
    ),
    "finite_size": (
        "FiniteSizeError ScalingResult EntanglementResult FiniteSizeScaling binder_cumulant "
        "entanglement_scaling"
    ),
    "floquet": (
        "FloquetError FloquetSystem FloquetResult BlochFloquetSolver floquet_spectrum "
        "floquet_effective_hamiltonian driven_two_level"
    ),
    "ft_tgate": "MagicStateError MagicStateFactory FTTGate FTCCXGate",
    "ftqc_pipeline": (
        "FTQCPipelineError PipelineResult ThresholdCurveResult FTQCPipeline threshold_curve "
        "threshold_crossing ResourceEstimate FTQCResourceEstimator"
    ),
    "gpu": (
        "GpuBackend GpuStatevector GpuLanczos GpuBatchedCircuit GpuError gpu_info to_gpu to_cpu"
    ),
    "hardware": (
        "HardwareError BackendCalibration HardwareJobResult HardwareJob IBMHardwareBackend "
        "IonQBackend GoogleBackend calibration_to_noise_model available_backends"
    ),
    "ibm": "IBMQuantumBackend IBMConnectorError circuit_to_qasm3",
    "jupyter_widgets": (
        "WidgetError CircuitWidgetState CircuitWidget BlochSphereWidgetState BlochSphereWidget "
        "JobMonitorWidgetState JobMonitorWidget"
    ),
    "lanczos": "LanczosSolver ArnoldiSolver SparseHamiltonian EigenResult LanczosError",
    "lindblad": (
        "LindbladError JumpOperator LindbladFrame EntanglementTrace LindbladResult LindbladSolver"
    ),
    "many_body": (
        "ManyBodyError OTOCResult OTOCSweepResult SYKResult SYKModel DMRG2DResult SnakeMapper "
        "compute_otoc otoc_sweep syk_hamiltonian dmrg_2d"
    ),
    "mbqc": "MBQCError GraphState MBQCPattern",
    "mclachlan": "McLachlanError McLachlanResult McLachlanSolver",
    "mitigation": (
        "MitigationError fold_gates Extrapolator RichardsonExtrapolator PolyExtrapolator "
        "ExpExtrapolator ZNEResult zne_mitigate QuasiProbabilityDecomposition PECResult "
        "pec_mitigate make_sv_executor"
    ),
    "mps_simulator": (
        "MPSSimulatorError AdaptiveTruncation MPSSimulator CachedDMRG BenchmarkResult "
        "ComparisonResult GpuBenchmarkResult mps_benchmark ghz_fidelity compare_mps_sv "
        "gpu_mps_benchmark get_backend_module is_gpu_available"
    ),
    "neqdyn": (
        "NonEquilError KeldyshResult KeldyshSolver EntanglementDynamicsResult "
        "EntanglementDynamics KibbleZurekResult KibbleZurekScaling"
    ),
    "noise": (
        "NoiseModel DepolarizingNoise AmplitudeDampingNoise ThermalRelaxationNoise ReadoutNoise "
        "CompositeNoise NoiseModelError"
    ),
    "noise_char": "NoiseCharError CharacterizedNoise NoiseCharacterizer",
    "noise_vqe": (
        "NoiseVQEError SPSAOptimizer NoiseAwareVQEResult NoiseAwareVQE build_hea_circuit "
        "hardware_efficient_vqe n_hea_params"
    ),
    "oovqe": "OOVQEError OOVQEResult OOVQESolver",
    "optimal_control": "OCError GRAPEResult GRAPESolver CRABResult CRABSolver",
    "pennylane_plugin": "QuantXDevice QuantXPluginError",
    "primitives": "PrimitiveError SparsePauliOp EstimatorResult SamplerResult Estimator Sampler",
    "pulse": (
        "PulseError DriveChannel MeasureChannel ControlChannel ConstantWaveform GaussianWaveform "
        "DRAGWaveform FlatTopWaveform ArbitraryWaveform PulseSchedule PulseInstruction "
        "QubitSystem PulseResult PulseSimulator pi_pulse_area"
    ),
    "qaoa_optim": (
        "CVaRQAOAError MaxCutProblem TSPProblem CVaRQAOA QAOALandscapeResult QAOALandscape "
        "warm_start_qaoa"
    ),
    "qasm": "ParseError parse_qasm_to_circuit",
    "qbm": "RBM QuantumBoltzmannMachine BoltzmannSampler QBMError",
    "qec": "RepetitionCode SurfaceCode LogicalErrorSimulator ThresholdResult QECError MWPMDecoder",
    "qiskit_compat": (
        "QiskitCompatError BitArray DataBin PubResult PrimitiveResult SamplerPUB EstimatorPUB "
        "QuantXSamplerV2 QuantXEstimatorV2"
    ),
    "qldpc": "LDPCError ClassicalCode HypergraphProductCode LDPCDecoder bicycle_code",
    "qml": "QuantumNeuralNetwork QuantumClassifier QMLError",
    "qsp_qsvt": "QSPError QSPAnglesFinder QSVTCircuit",
    "quantum_channel": "ChannelError SteadyStateSolver",
    "quantum_kernel": (
        "QKernelError QuantumFeatureMap ZZFeatureMap FidelityQuantumKernel target_alignment "
        "frobenius_alignment QSVMResult QSVM"
    ),
    "quantum_network": (
        "QuantumChannel NetworkError QuantumNode QuantumNetwork BellPairGenerator "
        "EntanglementSwapper QuantumRepeater NetworkResult fidelity_after_dephasing "
        "fidelity_after_swapping fidelity_after_purification simulate_repeater_chain"
    ),
    "quantum_protocols": (
        "ProtocolError BB84Result BB84Protocol E91Result E91Protocol TeleportationResult "
        "TeleportationProtocol DenseCodingResult DenseCodingProtocol bb84_key_rate "
        "chsh_inequality_value teleportation_fidelity"
    ),
    "quantum_walk": (
        "QuantumWalkError WalkResult GroverWalkResult DiscreteTimeWalk ContinuousTimeQuantumWalk "
        "GroverWalk walk_spectrum ctqw_transfer_probability"
    ),
    "ray_sim": (
        "DistSVError DistSVResult DistributedSVSimulator is_ray_available dist_sv_benchmark"
    ),
    "rb_qv": "RBQVError RBResult StandardRB QVResult QuantumVolume",
    "rydberg": "RydbergError RydbergAtom RydbergHamiltonian RydbergResult RydbergSimulator",
    "spectral": "SpectralError SpectralResult KrylovResult KrylovEvolver",
    "statevector": "StatevectorSimulator",
    "sweep": "ParameterSweep SweepResult SweepError grid_sweep list_sweep",
    "tdmrg": "TDMRGError TDMRGResult TDMRGSolver real_time_evolve imaginary_time_evolve",
    "templates": (
        "TemplateError CircuitTemplate hardware_efficient_ansatz iqp_circuit qaoa_circuit "
        "ghz_circuit quantum_fourier_transform uccsd_ansatz"
    ),
    "tensor_network": "MPS MPO TEBD DMRGResult DMRG TensorNetworkError",
    "tomography": "TomographyError StateTomography ProcessTomography",
    "transpiler": "TranspilerError GateOp CouplingMap TranspileResult HardwareTranspiler",
    "trotter": (
        "TrotterError TrotterTerm TrotterResult TrotterSimulator heisenberg_trotter ising_trotter"
    ),
    "xeb": "XEBError XEBResult CrossEntropyBenchmarking",
}
_MOVED_NAMES = {name: (module, name) for module, names in _OLD_EXPORTS.items()
                for name in names.split()}
_MOVED_NAMES.update({
    "CVaRQAOAResult": ("qaoa_optim", "QAOAResult"),
})


def _moved(old, new):
    _warnings.warn(f"{old} is now {new}: it is experimental, and the old name goes in "
                   f"{_REMOVED_IN}.", DeprecationWarning, stacklevel=3)


def __getattr__(name):
    # `from quantx_engine import X` looks X up twice, first through hasattr() in the import
    # system's _handle_fromlist; only the second lookup, the import itself, warns.
    quiet = _sys._getframe(1).f_code.co_name == "_handle_fromlist"
    if name in _MOVED_MODULES:
        if not quiet:
            _moved(f"{__name__}.{name}", f"{__name__}.contrib.{name}")
        module = _importlib.import_module(f"{__name__}.contrib.{name}")
        _sys.modules.setdefault(f"{__name__}.{name}", module)
        return module
    if name in _MOVED_NAMES:
        module, attr = _MOVED_NAMES[name]
        if not quiet:
            _moved(f"{__name__}.{name}", f"{__name__}.contrib.{module}.{attr}")
        return getattr(_importlib.import_module(f"{__name__}.contrib.{module}"), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _OldPaths(_abc.MetaPathFinder, _abc.Loader):
    """Imports quantx_engine.<m>, for a module that moved, as quantx_engine.contrib.<m>."""

    _quantx_old_paths = True

    def __init__(self):
        self._specs = {}

    def find_spec(self, fullname, path=None, target=None):
        parent, _, name = fullname.rpartition(".")
        if parent == __name__ and name in _MOVED_MODULES:
            return _util.spec_from_loader(fullname, self)
        return None

    def create_module(self, spec):
        new = f"{__name__}.contrib.{spec.name.rpartition('.')[2]}"
        _moved(spec.name, new)
        module = _importlib.import_module(new)
        self._specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module):
        # module_from_spec() pointed the contrib module's __spec__ at the old name.
        module.__spec__ = self._specs.pop(module.__spec__.name)


# First in line: an editable install of an older checkout must not answer for these names.
if not any(getattr(f, "_quantx_old_paths", False) for f in _sys.meta_path):
    _sys.meta_path.insert(0, _OldPaths())
