"""Public Python SDK exports for QuantX."""

__version__ = "0.2.0"

from .fault_tolerant import (
    FTError,
    LogicalQubit,
    MagicStateDistillery,
    DistillationResult,
    FaultTolerantCircuit,
    logical_error_rate_vs_distance,
    t_gate_cost,
)
from .crosstalk import (
    CrosstalkError,
    CouplingGraph,
    CrosstalkModel,
    zz_routing_cost,
    cheapest_path,
    echo_sequence,
)
from .pulse import (
    PulseError,
    DriveChannel,
    MeasureChannel,
    ControlChannel,
    ConstantWaveform,
    GaussianWaveform,
    DRAGWaveform,
    FlatTopWaveform,
    ArbitraryWaveform,
    PulseSchedule,
    PulseInstruction,
    QubitSystem,
    PulseResult,
    PulseSimulator,
    pi_pulse_area,
)
from .primitives import (
    PrimitiveError,
    SparsePauliOp,
    EstimatorResult,
    SamplerResult,
    Estimator,
    Sampler,
)
from .dynamic import (
    DynamicCircuitError,
    DynamicCircuit,
    DynamicResult,
    DynamicSimulator,
)
from .engine import CircuitError, Engine, QuantXNativeUnavailable
from .qasm import ParseError, parse_qasm_to_circuit
from .noise import (
    NoiseModel,
    DepolarizingNoise,
    AmplitudeDampingNoise,
    ThermalRelaxationNoise,
    ReadoutNoise,
    CompositeNoise,
    NoiseModelError,
)
from .ibm import IBMQuantumBackend, IBMConnectorError, circuit_to_qasm3
from .chemistry import (
    PauliTerm, PauliHamiltonian, jordan_wigner, MolecularVQE, ChemistryError,
)
from .advanced_chemistry import (
    AdvChemError,
    build_hamiltonian_matrix,
    hf_state,
    UCCSDResult,
    UCCSDSolver,
    ADAPTVQEResult,
    ADAPTVQESolver,
    QAOAResult,
    QAOASolver,
    QITEResult,
    QITESolver,
)
from .quantum_kernel import (
    QKernelError,
    QuantumFeatureMap,
    ZZFeatureMap,
    FidelityQuantumKernel,
    target_alignment,
    frobenius_alignment,
    QSVMResult,
    QSVM,
)
from .rb_qv import (
    RBQVError,
    RBResult,
    StandardRB,
    QVResult,
    QuantumVolume,
)
from .qml import QuantumNeuralNetwork, QuantumClassifier, QMLError
from .qec import RepetitionCode, SurfaceCode, LogicalErrorSimulator, ThresholdResult, QECError, MWPMDecoder
from .tensor_network import MPS, MPO, TEBD, DMRGResult, DMRG, TensorNetworkError
from .qbm import RBM, QuantumBoltzmannMachine, BoltzmannSampler, QBMError
from .clifford_t import GateSequence, CliffordGroup, SolovayKitaev, TCountOptimizer, CliffordTError
from .pennylane_plugin import QuantXDevice, QuantXPluginError
from .lanczos import LanczosSolver, ArnoldiSolver, SparseHamiltonian, EigenResult, LanczosError
from .sweep import ParameterSweep, SweepResult, SweepError, grid_sweep, list_sweep
from .gpu import GpuBackend, GpuStatevector, GpuLanczos, GpuBatchedCircuit, GpuError, gpu_info, to_gpu, to_cpu
from .lindblad import (
    LindbladError,
    JumpOperator,
    LindbladFrame,
    EntanglementTrace,
    LindbladResult,
    LindbladSolver,
)
from .spectral import (
    SpectralError,
    SpectralResult,
    KrylovResult,
    KrylovEvolver,
)
from .quantum_channel import (
    ChannelError,
    QuantumChannel,
    SteadyStateSolver,
)
from .finite_size import (
    FiniteSizeError,
    ScalingResult,
    EntanglementResult,
    FiniteSizeScaling,
    binder_cumulant,
    entanglement_scaling,
)
from .mitigation import (
    MitigationError,
    fold_gates,
    Extrapolator,
    RichardsonExtrapolator,
    PolyExtrapolator,
    ExpExtrapolator,
    ZNEResult,
    zne_mitigate,
    QuasiProbabilityDecomposition,
    PECResult,
    pec_mitigate,
    make_sv_executor,
)
from .transpiler import (
    TranspilerError,
    GateOp,
    CouplingMap,
    TranspileResult,
    HardwareTranspiler,
)
from .adapters import (
    AdapterError,
    to_cirq,
    from_cirq,
    to_braket,
    from_braket,
)
from .xeb import (
    XEBError,
    XEBResult,
    CrossEntropyBenchmarking,
)
from .tomography import (
    TomographyError,
    StateTomography,
    ProcessTomography,
)
from .optimal_control import (
    OCError,
    GRAPEResult,
    GRAPESolver,
    CRABResult,
    CRABSolver,
)
from .templates import (
    TemplateError,
    CircuitTemplate,
    hardware_efficient_ansatz,
    iqp_circuit,
    qaoa_circuit,
    ghz_circuit,
    quantum_fourier_transform,
    uccsd_ansatz,
)
from .noise_char import (
    NoiseCharError,
    CharacterizedNoise,
    NoiseCharacterizer,
)
from .floquet import (
    FloquetError,
    FloquetSystem,
    FloquetResult,
    BlochFloquetSolver,
    floquet_spectrum,
    floquet_effective_hamiltonian,
    driven_two_level,
)
from .hardware import (
    HardwareError,
    BackendCalibration,
    HardwareJobResult,
    HardwareJob,
    IBMHardwareBackend,
    IonQBackend,
    GoogleBackend,
    calibration_to_noise_model,
    available_backends,
)
from .tdmrg import (
    TDMRGError,
    TDMRGResult,
    TDMRGSolver,
    real_time_evolve,
    imaginary_time_evolve,
)
from .quantum_network import (
    NetworkError,
    QuantumNode,
    QuantumChannel,
    QuantumNetwork,
    BellPairGenerator,
    EntanglementSwapper,
    QuantumRepeater,
    NetworkResult,
    fidelity_after_dephasing,
    fidelity_after_swapping,
    fidelity_after_purification,
    simulate_repeater_chain,
)
from .quantum_protocols import (
    ProtocolError,
    BB84Result,
    BB84Protocol,
    E91Result,
    E91Protocol,
    TeleportationResult,
    TeleportationProtocol,
    DenseCodingResult,
    DenseCodingProtocol,
    bb84_key_rate,
    chsh_inequality_value,
    teleportation_fidelity,
)
from .noise_vqe import (
    NoiseVQEError,
    SPSAOptimizer,
    NoiseAwareVQEResult,
    NoiseAwareVQE,
    build_hea_circuit,
    hardware_efficient_vqe,
    n_hea_params,
)
from .quantum_walk import (
    QuantumWalkError,
    WalkResult,
    GroverWalkResult,
    DiscreteTimeWalk,
    ContinuousTimeQuantumWalk,
    GroverWalk,
    walk_spectrum,
    ctqw_transfer_probability,
)
from .trotter import (
    TrotterError,
    TrotterTerm,
    TrotterResult,
    TrotterSimulator,
    heisenberg_trotter,
    ising_trotter,
)
from .many_body import (
    ManyBodyError,
    OTOCResult,
    OTOCSweepResult,
    SYKResult,
    SYKModel,
    DMRG2DResult,
    SnakeMapper,
    compute_otoc,
    otoc_sweep,
    syk_hamiltonian,
    dmrg_2d,
)
from .mps_simulator import (
    MPSSimulatorError,
    AdaptiveTruncation,
    MPSSimulator,
    CachedDMRG,
    BenchmarkResult,
    ComparisonResult,
    GpuBenchmarkResult,       # M109
    mps_benchmark,
    ghz_fidelity,
    compare_mps_sv,
    gpu_mps_benchmark,        # M109
    get_backend_module,       # M109
    is_gpu_available,         # M109
)
from .clifford_sim import (   # M110
    CliffordError,
    StabilizerState,
    CliffordSimulator,
    StimCircuit,
    StimCircuitError,
)
from .ray_sim import (        # M111
    DistSVError,
    DistSVResult,
    DistributedSVSimulator,
    is_ray_available,
    dist_sv_benchmark,
)
from .ftqc_pipeline import (  # M112
    FTQCPipelineError,
    PipelineResult,
    ThresholdCurveResult,
    FTQCPipeline,
    threshold_curve,
    threshold_crossing,
)
from .excited_state import (  # M113
    ExcitedStateError,
    ExcitedStateResult,
    ExcitedStateVQE,
    ConicalIntersectionResult,
    detect_conical_intersections,
    pyscf_to_pauli_hamiltonian,
)
from .neqdyn import (  # M114
    NonEquilError,
    KeldyshResult,
    KeldyshSolver,
    EntanglementDynamicsResult,
    EntanglementDynamics,
    KibbleZurekResult,
    KibbleZurekScaling,
)
from .qaoa_optim import (  # M115
    CVaRQAOAError,
    QAOAResult as CVaRQAOAResult,  # renamed to avoid clash with advanced_chemistry.QAOAResult
    MaxCutProblem,
    TSPProblem,
    CVaRQAOA,
    QAOALandscapeResult,
    QAOALandscape,
    warm_start_qaoa,
)
from .qiskit_compat import (  # M116
    QiskitCompatError,
    BitArray,
    DataBin,
    PubResult,
    PrimitiveResult,
    SamplerPUB,
    EstimatorPUB,
    QuantXSamplerV2,
    QuantXEstimatorV2,
)
from .jupyter_widgets import (  # M117
    WidgetError,
    CircuitWidgetState,
    CircuitWidget,
    BlochSphereWidgetState,
    BlochSphereWidget,
    JobMonitorWidgetState,
    JobMonitorWidget,
)
from .rydberg import (  # M118
    RydbergError,
    RydbergAtom,
    RydbergHamiltonian,
    RydbergResult,
    RydbergSimulator,
)
from .bosonic import (  # M119
    BosonicError,
    CatNoiseModel,
    CatQubit,
    GKPState,
    GKPDecoder,
)
from .dyndecouple import (  # M126
    DDError,
    DDSequence,
    DynamicalDecoupling,
    QuantumMemory,
)
from .mbqc import (  # M125
    MBQCError,
    GraphState,
    MBQCPattern,
)
from .oovqe import (  # M124
    OOVQEError,
    OOVQEResult,
    OOVQESolver,
)
from .qsp_qsvt import (  # M123
    QSPError,
    QSPAnglesFinder,
    QSVTCircuit,
)
from .mclachlan import (  # M122
    McLachlanError,
    McLachlanResult,
    McLachlanSolver,
)
from .qldpc import (  # M121
    LDPCError,
    ClassicalCode,
    HypergraphProductCode,
    LDPCDecoder,
    bicycle_code,
)
from .ft_tgate import (  # M120
    MagicStateError,
    MagicStateFactory,
    FTTGate,
    FTCCXGate,
    # TCountOptimizer lives in ft_tgate but is not re-exported here to
    # avoid shadowing the existing quantx_engine.clifford_t.TCountOptimizer.
    # Import it directly: from quantx_engine.ft_tgate import TCountOptimizer
)

__all__ = [
    # Fault-tolerant logical qubits (M91)
    "FTError",
    "LogicalQubit",
    "MagicStateDistillery",
    "DistillationResult",
    "FaultTolerantCircuit",
    "logical_error_rate_vs_distance",
    "t_gate_cost",
    # Crosstalk + ZZ coupling model (M90)
    "CrosstalkError",
    "CouplingGraph",
    "CrosstalkModel",
    "zz_routing_cost",
    "cheapest_path",
    "echo_sequence",
    # Pulse-level simulator (M89)
    "PulseError",
    "DriveChannel",
    "MeasureChannel",
    "ControlChannel",
    "ConstantWaveform",
    "GaussianWaveform",
    "DRAGWaveform",
    "FlatTopWaveform",
    "ArbitraryWaveform",
    "PulseSchedule",
    "PulseInstruction",
    "QubitSystem",
    "PulseResult",
    "PulseSimulator",
    "pi_pulse_area",
    "__version__",
    "CircuitError",
    "Engine",
    "QuantXNativeUnavailable",
    "ParseError",
    "parse_qasm_to_circuit",
    "NoiseModel",
    "DepolarizingNoise",
    "AmplitudeDampingNoise",
    "ThermalRelaxationNoise",
    "ReadoutNoise",
    "CompositeNoise",
    "NoiseModelError",
    "IBMQuantumBackend",
    "IBMConnectorError",
    "circuit_to_qasm3",
    "PauliTerm",
    "PauliHamiltonian",
    "jordan_wigner",
    "MolecularVQE",
    "ChemistryError",
    # Advanced chemistry solvers (M79)
    "AdvChemError",
    "build_hamiltonian_matrix",
    "hf_state",
    "UCCSDResult",
    "UCCSDSolver",
    "ADAPTVQEResult",
    "ADAPTVQESolver",
    "QAOAResult",
    "QAOASolver",
    "QITEResult",
    "QITESolver",
    # Quantum kernel methods + QSVM (M80)
    "QKernelError",
    "QuantumFeatureMap",
    "ZZFeatureMap",
    "FidelityQuantumKernel",
    "target_alignment",
    "frobenius_alignment",
    "QSVMResult",
    "QSVM",
    # Randomized Benchmarking + Quantum Volume (M81)
    "RBQVError",
    "RBResult",
    "StandardRB",
    "QVResult",
    "QuantumVolume",
    "QuantumNeuralNetwork",
    "QuantumClassifier",
    "QMLError",
    "RepetitionCode",
    "SurfaceCode",
    "LogicalErrorSimulator",
    "ThresholdResult",
    "QECError",
    "MWPMDecoder",
    "MPS",
    "MPO",
    "TEBD",
    "DMRGResult",
    "DMRG",
    "TensorNetworkError",
    "RBM",
    "QuantumBoltzmannMachine",
    "BoltzmannSampler",
    "QBMError",
    "GateSequence",
    "CliffordGroup",
    "SolovayKitaev",
    "TCountOptimizer",
    "CliffordTError",
    "QuantXDevice",
    "QuantXPluginError",
    "LanczosSolver",
    "ArnoldiSolver",
    "SparseHamiltonian",
    "EigenResult",
    "LanczosError",
    "ParameterSweep",
    "SweepResult",
    "SweepError",
    "grid_sweep",
    "list_sweep",
    "GpuBackend",
    "GpuStatevector",
    "GpuLanczos",
    "GpuBatchedCircuit",
    "GpuError",
    "gpu_info",
    "to_gpu",
    "to_cpu",
    # Quantum channels + steady-state solver (M75)
    "ChannelError",
    "QuantumChannel",
    "SteadyStateSolver",
    # Finite-size scaling + quantum phase transitions (M74)
    "FiniteSizeError",
    "ScalingResult",
    "EntanglementResult",
    "FiniteSizeScaling",
    "binder_cumulant",
    "entanglement_scaling",
    # Spectral functions + Krylov time evolution (M73)
    "SpectralError",
    "SpectralResult",
    "KrylovResult",
    "KrylovEvolver",
    # Lindblad open-system solver (M71)
    "LindbladError",
    "JumpOperator",
    "LindbladFrame",
    "EntanglementTrace",
    "LindbladResult",
    "LindbladSolver",
    # Error mitigation (M70)
    "MitigationError",
    "fold_gates",
    "Extrapolator",
    "RichardsonExtrapolator",
    "PolyExtrapolator",
    "ExpExtrapolator",
    "ZNEResult",
    "zne_mitigate",
    "QuasiProbabilityDecomposition",
    "PECResult",
    "pec_mitigate",
    "make_sv_executor",
    # Hardware-aware transpilation (M76)
    "TranspilerError",
    "GateOp",
    "CouplingMap",
    "TranspileResult",
    "HardwareTranspiler",
    # Cirq + Braket adapters (M77)
    "AdapterError",
    "to_cirq",
    "from_cirq",
    "to_braket",
    "from_braket",
    # Cross-entropy benchmarking (M82)
    "XEBError",
    "XEBResult",
    "CrossEntropyBenchmarking",
    # State and process tomography (M83)
    "TomographyError",
    "StateTomography",
    "ProcessTomography",
    # Quantum optimal control — GRAPE and CRAB (M84)
    "OCError",
    "GRAPEResult",
    "GRAPESolver",
    "CRABResult",
    "CRABSolver",
    # Estimator + Sampler primitives (M85)
    "PrimitiveError",
    "SparsePauliOp",
    "EstimatorResult",
    "SamplerResult",
    "Estimator",
    "Sampler",
    # Dynamic circuits (M86)
    "DynamicCircuitError",
    "DynamicCircuit",
    "DynamicResult",
    "DynamicSimulator",
    # Circuit Templates Library (M88)
    "TemplateError",
    "CircuitTemplate",
    "hardware_efficient_ansatz",
    "iqp_circuit",
    "qaoa_circuit",
    "ghz_circuit",
    "quantum_fourier_transform",
    "uccsd_ansatz",
    # Noise characterization pipeline (M87)
    "NoiseCharError",
    "CharacterizedNoise",
    "NoiseCharacterizer",
    # Floquet engineering + Bloch-Floquet band structure (M92)
    "FloquetError",
    "FloquetSystem",
    "FloquetResult",
    "BlochFloquetSolver",
    "floquet_spectrum",
    "floquet_effective_hamiltonian",
    "driven_two_level",
    # Quantum communication protocols (M97)
    "ProtocolError",
    "BB84Result",
    "BB84Protocol",
    "E91Result",
    "E91Protocol",
    "TeleportationResult",
    "TeleportationProtocol",
    "DenseCodingResult",
    "DenseCodingProtocol",
    "bb84_key_rate",
    "chsh_inequality_value",
    "teleportation_fidelity",
    # Quantum network simulation (M96)
    "NetworkError",
    "QuantumNode",
    "QuantumChannel",
    "QuantumNetwork",
    "BellPairGenerator",
    "EntanglementSwapper",
    "QuantumRepeater",
    "NetworkResult",
    "fidelity_after_dephasing",
    "fidelity_after_swapping",
    "fidelity_after_purification",
    "simulate_repeater_chain",
    # Noise-aware VQE + SPSA optimizer (M95)
    "NoiseVQEError",
    "SPSAOptimizer",
    "NoiseAwareVQEResult",
    "NoiseAwareVQE",
    "build_hea_circuit",
    "hardware_efficient_vqe",
    "n_hea_params",
    # Time-dependent DMRG (M94)
    "TDMRGError",
    "TDMRGResult",
    "TDMRGSolver",
    "real_time_evolve",
    "imaginary_time_evolve",
    # Real hardware backend integration (M93)
    "HardwareError",
    "BackendCalibration",
    "HardwareJobResult",
    "HardwareJob",
    "IBMHardwareBackend",
    "IonQBackend",
    "GoogleBackend",
    "calibration_to_noise_model",
    "available_backends",
    # Quantum walk algorithms (M100)
    "QuantumWalkError",
    "WalkResult",
    "GroverWalkResult",
    "DiscreteTimeWalk",
    "ContinuousTimeQuantumWalk",
    "GroverWalk",
    "walk_spectrum",
    "ctqw_transfer_probability",
    # Suzuki-Trotter Hamiltonian simulation (M101)
    "TrotterError",
    "TrotterTerm",
    "TrotterResult",
    "TrotterSimulator",
    "heisenberg_trotter",
    "ising_trotter",
    # Many-body physics enhancements (M102)
    "ManyBodyError",
    "OTOCResult",
    "OTOCSweepResult",
    "SYKResult",
    "SYKModel",
    "DMRG2DResult",
    "SnakeMapper",
    "compute_otoc",
    "otoc_sweep",
    "syk_hamiltonian",
    "dmrg_2d",
    # MPS simulator extension (M104)
    "MPSSimulatorError",
    "AdaptiveTruncation",
    "MPSSimulator",
    "CachedDMRG",
    "BenchmarkResult",
    "ComparisonResult",
    "mps_benchmark",
    "ghz_fidelity",
    "compare_mps_sv",
    # GPU-accelerated MPS (M109)
    "GpuBenchmarkResult",
    "gpu_mps_benchmark",
    "get_backend_module",
    "is_gpu_available",
    # Large-scale Clifford simulation (M110)
    "CliffordError",
    "StabilizerState",
    "CliffordSimulator",
    "StimCircuit",
    "StimCircuitError",
    # Distributed statevector via Ray (M111)
    "DistSVError",
    "DistSVResult",
    "DistributedSVSimulator",
    "is_ray_available",
    "dist_sv_benchmark",
    # Fault-tolerant QC end-to-end pipeline (M112)
    "FTQCPipelineError",
    "PipelineResult",
    "ThresholdCurveResult",
    "FTQCPipeline",
    "threshold_curve",
    "threshold_crossing",
    # Excited-state quantum chemistry (M113)
    "ExcitedStateError",
    "ExcitedStateResult",
    "ExcitedStateVQE",
    "ConicalIntersectionResult",
    "detect_conical_intersections",
    "pyscf_to_pauli_hamiltonian",
    # Non-equilibrium quantum dynamics (M114)
    "NonEquilError",
    "KeldyshResult",
    "KeldyshSolver",
    "EntanglementDynamicsResult",
    "EntanglementDynamics",
    "KibbleZurekResult",
    "KibbleZurekScaling",
    # CVaR-QAOA combinatorial optimisation (M115)
    "CVaRQAOAError",
    "CVaRQAOAResult",  # quantx_engine.qaoa_optim.QAOAResult aliased to avoid clash
    "MaxCutProblem",
    "TSPProblem",
    "CVaRQAOA",
    "QAOALandscapeResult",
    "QAOALandscape",
    "warm_start_qaoa",
    # Qiskit 1.x Primitives V2 compatibility (M116)
    "QiskitCompatError",
    "BitArray",
    "DataBin",
    "PubResult",
    "PrimitiveResult",
    "SamplerPUB",
    "EstimatorPUB",
    "QuantXSamplerV2",
    "QuantXEstimatorV2",
    # Jupyter interactive widgets (M117)
    "WidgetError",
    "CircuitWidgetState",
    "CircuitWidget",
    "BlochSphereWidgetState",
    "BlochSphereWidget",
    "JobMonitorWidgetState",
    "JobMonitorWidget",
    # Neutral-atom Rydberg blockade simulator (M118)
    "RydbergError",
    "RydbergAtom",
    "RydbergHamiltonian",
    "RydbergResult",
    "RydbergSimulator",
    # Bosonic codes — cat qubits + GKP (M119)
    "BosonicError",
    "CatNoiseModel",
    "CatQubit",
    "GKPState",
    "GKPDecoder",
    # Fault-tolerant T gate via magic state injection (M120)
    "MagicStateError",
    "MagicStateFactory",
    "FTTGate",
    "FTCCXGate",
    # Quantum memory + dynamical decoupling (M126)
    "DDError",
    "DDSequence",
    "DynamicalDecoupling",
    "QuantumMemory",
    # Measurement-based quantum computation (M125)
    "MBQCError",
    "GraphState",
    "MBQCPattern",
    # Orbital-optimised VQE (M124)
    "OOVQEError",
    "OOVQEResult",
    "OOVQESolver",
    # Quantum signal processing + QSVT (M123)
    "QSPError",
    "QSPAnglesFinder",
    "QSVTCircuit",
    # McLachlan variational quantum dynamics (M122)
    "McLachlanError",
    "McLachlanResult",
    "McLachlanSolver",
    # Quantum LDPC codes — hypergraph product + bicycle (M121)
    "LDPCError",
    "ClassicalCode",
    "HypergraphProductCode",
    "LDPCDecoder",
    "bicycle_code",
]
