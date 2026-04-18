"""Public Python SDK exports for QuantX."""

from .engine import CircuitError, Engine
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
from .qml import QuantumNeuralNetwork, QuantumClassifier, QMLError
from .qec import RepetitionCode, SurfaceCode, LogicalErrorSimulator, ThresholdResult, QECError
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

__all__ = [
    "CircuitError",
    "Engine",
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
    "QuantumNeuralNetwork",
    "QuantumClassifier",
    "QMLError",
    "RepetitionCode",
    "SurfaceCode",
    "LogicalErrorSimulator",
    "ThresholdResult",
    "QECError",
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
]
