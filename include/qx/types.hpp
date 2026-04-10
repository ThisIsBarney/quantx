#pragma once

#include <cstddef>
#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace qx {

// High-level simulation problem category.
enum class ProblemKind {
  Discrete,
  ContinuousSpace2D,
  ContinuousSpace3D
};

// Equation family selector for present/future solver tracks.
enum class EquationType {
  Schrodinger,
  Lindblad,
  Dirac,
  KleinGordon
};

// Backend categories currently supported by the runtime.
enum class BackendKind {
  Statevector,
  MPS,
  Continuous2D,
  Continuous3D
};

// Runtime resource budget and hard limits.
struct ResourceConstraints {
  // 0 means "runtime decides"; non-zero values are hard limits from caller.
  std::uint32_t max_threads{0};
  std::uint32_t max_gpus{0};
  std::uint32_t mpi_ranks{1};
  std::uint64_t max_memory_mb{0};
};

// Time-evolution algorithm and numeric controls.
struct EvolutionConfig {
  // Time-domain evolution settings shared by statevector and tensor-network backends.
  double total_time{1.0};
  double dt{0.01};
  // statevector: trotter/krylov; mps: tebd/tdvp.
  std::string method{"trotter"};
  // Krylov subspace dimension for Arnoldi-based short-time propagation.
  std::uint32_t krylov_dim{6};
  // MPS controls: cap bond growth and control truncation accumulation.
  std::uint32_t mps_max_bond{1024};
  double mps_trunc_tol{1e-8};
};

// One Pauli-string Hamiltonian term.
struct HamiltonianTerm {
  std::string label;
  double coefficient{0.0};
};

// Node in operator graph representation reserved for advanced equations.
struct OperatorNode {
  std::string id;
  std::string op_type;
  std::vector<std::string> inputs;
  std::map<std::string, double> params;
};

// Graph-structured operator container for v2+ solvers.
struct OperatorGraph {
  // Reserved for v2+ operator-centric formulations (Dirac/Klein-Gordon etc.).
  std::vector<OperatorNode> nodes;
};

// Initial condition descriptor for continuous-space solvers.
struct ContinuousInitialStateConfig {
  // Initial-state token, e.g. "gaussian" or "grid".
  std::string kind{"gaussian"};
  // Analytic parameters used by named initial states.
  std::map<std::string, double> params{};
  // Optional flattened amplitude grid used by grid-based initial states.
  std::vector<double> grid{};
  // Initial condition for the conjugate momentum field π = ∂φ/∂t (Klein-Gordon only).
  // "zero":               π(x,0) = 0 (default — stationary wavepacket).
  // "positive_frequency": π̃_k = -i·ω_k·φ̃_k, where ω_k = sqrt(k² + m²).
  //                       Selects the positive-energy branch of the KG dispersion.
  std::string pi_kind{"zero"};
};

// Time-dependent driving field configuration for continuous-space solvers.
// The driving potential adds V_drive(r,t) = amplitude * envelope(t) * spatial_profile(r)
// on top of the static potential at each split-step half/full step.
//
// Driving types:
//   "none"           — no driving (default; fast path using precomputed phases)
//   "sine"           — envelope(t) = cos(frequency * t + phase)
//   "gaussian_pulse" — envelope(t) = exp(-(t-t_center)²/(2·t_width²)) * cos(frequency·t + phase)
//   "chirp"          — envelope(t) = cos((frequency + 0.5·chirp_rate·t)·t + phase)
//
// Spatial profile:
//   "uniform" — spatial_profile(r) = 1  (homogeneous field)
//   "dipole_x" — spatial_profile(r) = x  (x-polarized dipole coupling)
//   "dipole_y" — spatial_profile(r) = y  (y-polarized dipole coupling)
struct DrivingConfig {
  std::string type{"none"};
  double amplitude{0.0};
  double frequency{1.0};       // ω in rad/time-unit
  double phase{0.0};           // initial phase offset (radians)
  double t_center{0.0};        // gaussian_pulse: center time
  double t_width{1.0};         // gaussian_pulse: σ; chirp: linear sweep rate α
  std::string spatial_profile{"uniform"};
};

// Continuous 2D spatial simulation configuration.
struct ContinuousSpace2DConfig {
  // Uniform Cartesian grid resolution.
  std::uint32_t nx{128};
  std::uint32_t ny{128};
  // Spatial step size.
  double dx{0.1};
  double dy{0.1};
  // Time integration controls.
  double dt{1e-3};
  std::uint32_t steps{100};
  // Boundary mode token (e.g. periodic/dirichlet/absorbing).
  std::string boundary{"periodic"};
  // Dispersion relation for the kinetic energy operator.
  // "nonrelativistic": E_k = k²/(2m)  (default, ℏ=1 units)
  // "relativistic":    E_k = √(k²c² + m²c⁴) - mc²  (ℏ=1 units)
  std::string dispersion{"nonrelativistic"};
  // Particle mass in natural units (ℏ=1). Default 1.0.
  double mass{1.0};
  // Speed of light in simulation units. Only used when dispersion="relativistic". Default 1.0.
  double speed_of_light{1.0};
  // Potential descriptor.
  std::string potential_kind{"none"};
  std::map<std::string, double> potential_params{};
  // Optional flattened potential grid (row-major, size nx*ny).
  std::vector<double> potential_grid{};
  // Time-dependent driving field (default: none).
  DrivingConfig driving{};
  // Initial condition descriptor for the wavefunction.
  ContinuousInitialStateConfig initial_state{};
};

// Continuous 3D spatial simulation configuration.
struct ContinuousSpace3DConfig {
  // Uniform Cartesian grid resolution.
  std::uint32_t nx{64};
  std::uint32_t ny{64};
  std::uint32_t nz{64};
  // Spatial step size.
  double dx{0.1};
  double dy{0.1};
  double dz{0.1};
  // Time integration controls.
  double dt{1e-3};
  std::uint32_t steps{100};
  // Boundary mode token (e.g. periodic/dirichlet/absorbing).
  std::string boundary{"periodic"};
  // Dispersion relation for the kinetic energy operator.
  // "nonrelativistic": E_k = k²/(2m)  (default, ℏ=1 units)
  // "relativistic":    E_k = √(k²c² + m²c⁴) - mc²  (ℏ=1 units)
  std::string dispersion{"nonrelativistic"};
  // Particle mass in natural units (ℏ=1). Default 1.0.
  double mass{1.0};
  // Speed of light in simulation units. Only used when dispersion="relativistic". Default 1.0.
  double speed_of_light{1.0};
  // Potential descriptor.
  std::string potential_kind{"none"};
  std::map<std::string, double> potential_params{};
  // Optional flattened potential grid (row-major, size nx*ny*nz).
  std::vector<double> potential_grid{};
  // Time-dependent driving field (default: none).
  DrivingConfig driving{};
  // Initial condition descriptor for the wavefunction.
  ContinuousInitialStateConfig initial_state{};
};

// Metadata snapshot for a continuous-density animation frame.
struct ContinuousDensityFrame {
  // Frame index within the continuous evolution.
  std::uint32_t step{0};
  // Physical time associated with the frame.
  double time{0.0};
  // Grid dimensions used to render the frame.
  std::uint32_t nx{0};
  std::uint32_t ny{0};
  std::uint32_t nz{0};
  // Boundary token used by the simulation.
  std::string boundary{"periodic"};
  // Integrated norm and low-order moments for diagnostics.
  double norm{0.0};
  double x_mean{0.0};
  double y_mean{0.0};
  double z_mean{0.0};
  double sigma_x{0.0};
  double sigma_y{0.0};
  double sigma_z{0.0};
  // Coordinate bounds for the exported frame payload.
  double x_min{0.0};
  double x_max{0.0};
  double y_min{0.0};
  double y_max{0.0};
  double z_min{0.0};
  double z_max{0.0};
  // Potential tag used to generate this frame.
  std::string potential_kind{"none"};
  // Instantaneous driving field amplitude at this frame's time (0 when no driving).
  double driving_amplitude{0.0};
  // Optional flattened slice payloads.
  // 2D: density_xy_flat has nx*ny entries.
  // 3D: density_xy_flat has nx*ny entries at z-mid.
  //     density_xz_flat has nx*nz entries at y-mid.
  //     density_yz_flat has ny*nz entries at x-mid.
  std::vector<double> density_xy_flat{};
  std::vector<double> density_xz_flat{};
  std::vector<double> density_yz_flat{};
  // Optional point-cloud payload encoded as flat [x,y,z,density] tuples.
  std::vector<double> points_xyzd_flat{};
  // Optional phase payload: arg(ψ₀) in radians.
  // 2D: nx*ny entries. 3D: slice payloads at respective midplanes.
  std::vector<double> phase_xy_flat{};
  std::vector<double> phase_xz_flat{};
  std::vector<double> phase_yz_flat{};
  // Optional per-spinor-component density slices for Dirac equation (nx*ny each).
  std::vector<double> dirac_s0_xy_flat{};
  std::vector<double> dirac_s1_xy_flat{};
  // Optional full 3D voxel grid for marching cubes (nx*ny*nz, x-fastest indexing).
  // Only emitted when RunOptions::export_volume_grid is true and grid is small enough.
  std::vector<double> density_xyz_flat{};
};

// Stable contract metadata for the continuous-export wrapper.
struct ContinuousExportConfig {
  // Frozen schema version for append-only compatibility.
  std::uint32_t schema_version{1};
  // Frame sampling stride used when building the export wrapper.
  std::uint32_t export_frame_stride{1};
  // Upper bound on sparse point-cloud tuples emitted by 3D payloads.
  std::size_t export_points_max{2500};
  // Density threshold ratio used to filter sparse 3D points.
  double export_density_floor_ratio{0.08};
};

// One recorded frame from a Lindblad density-matrix evolution.
struct LindbladFrame {
  // Step index.
  std::uint32_t step{0};
  // Physical time.
  double time{0.0};
  // Purity Tr(ρ²) — 1.0 = pure state, 1/2^n = maximally mixed.
  double purity{1.0};
  // Von Neumann entropy -Tr(ρ log ρ) in nats.
  double entropy{0.0};
  // Diagonal population of each computational basis state |i⟩.
  std::vector<double> populations{};
  // Maximum off-diagonal |ρ_ij| for i≠j — quantifies coherence.
  double coherence{0.0};
};

// One recorded frame from a Wigner quasi-probability and transport analysis.
// Computed from the 2D Schrödinger wavefunction ψ(x, y) at the y=y_mid slice.
struct WignerFrame {
  // Step index and physical time.
  std::uint32_t step{0};
  double time{0.0};
  // Wigner grid dimensions: nx x-positions × np momentum points (np == nx).
  std::uint32_t nx{0};
  std::uint32_t np{0};
  // Spatial and momentum axis bounds for the Wigner array.
  double x_min{0.0};
  double x_max{0.0};
  // Momentum range:  p_k = (k - N/2) * pi/(N*dx),  so
  //   p_min = -pi/(2*dx),  p_max approx +pi/(2*dx).
  double p_min{0.0};
  double p_max{0.0};
  // W(x_j, p_k) at y=y_mid, row-major [j*np + k].
  // The Wigner function can be negative (quantum interference indicator).
  std::vector<double> wigner_flat{};
  // Probability current density J_x(x,y) and J_y(x,y) via spectral differentiation.
  // Row-major [y*nx + x] (same layout as state_), size nx*ny.
  std::vector<double> current_x_flat{};
  std::vector<double> current_y_flat{};
  // Integrated J_x at fixed x-planes for transport coefficient estimates (hbar=1 units):
  //   transmission = sum_y  J_x(x_right, y) * dy   (x_right = 3/4 * Lx)
  //   reflection   = -sum_y J_x(x_left,  y) * dy   (x_left  = 1/4 * Lx)
  double transmission{0.0};
  double reflection{0.0};
  // Momentum-space expectation values (mass-independent, units of hbar/length).
  double px_mean{0.0};
  double py_mean{0.0};
  // Kinetic energy <(px^2 + py^2)/(2m)> in energy units.
  double kinetic_energy{0.0};
};

// One recorded diagnostics frame from a real TEBD/MPS time evolution.
// Emitted by MpsBackend each exported step.
struct MpsFrame {
  // Step index and physical time.
  std::uint32_t step{0};
  double time{0.0};
  // Number of sites (qubits) in the MPS.
  std::uint32_t n_sites{0};
  // Maximum bond dimension observed across all bonds at this step.
  std::uint32_t max_bond_dim{0};
  // Cumulative truncation error (sum of discarded singular-value weight).
  double truncation_error{0.0};
  // Bond dimension χ_j at each of the (n_sites−1) inter-site bonds.
  std::vector<std::uint32_t> bond_dims{};
  // Bipartite von Neumann entropy S_j = −Σ σ²_k log σ²_k at each bond.
  std::vector<double> bond_entropies{};
  // Single-site magnetisation ⟨Z_i⟩ for each site.
  std::vector<double> magnetization{};
};

// One recorded frame from a bipartite entanglement analysis.
// Computed from a pure statevector (Schmidt decomposition via SVD) or a
// Lindblad density matrix (partial trace + eigendecomposition).
struct EntanglementFrame {
  // Step index and physical time.
  std::uint32_t step{0};
  double time{0.0};
  // Bipartite cut: subsystem A = qubits [0, n_a), subsystem B = [n_a, n).
  std::uint32_t n_a{0};
  std::uint32_t n_b{0};
  // Von Neumann entanglement entropy S(ρ_A) = -Tr(ρ_A log ρ_A) in nats.
  double entropy_a{0.0};
  // Mutual information I(A:B) = S_A + S_B − S_total (0 for pure states since S=S_A+S_B).
  double mutual_info{0.0};
  // Concurrence C ∈ [0,1] for 2-qubit systems (pure: 2σ₀σ₁; mixed: Wootters trace).
  // Set to -1.0 when the system is not 2 qubits.
  double concurrence{-1.0};
  // Logarithmic negativity E_N = log2(‖ρ^{T_A}‖₁) — non-zero only for entangled mixed states.
  double log_negativity{0.0};
  // Schmidt coefficients (singular values), largest first.
  std::vector<double> schmidt_values{};
};

// Canonical simulation problem description used across API boundaries.
struct QuantumProblem {
  ProblemKind kind{ProblemKind::Discrete};
  std::uint32_t qubits{1};
  EquationType equation{EquationType::Schrodinger};
  EvolutionConfig evolution{};
  ContinuousSpace2DConfig continuous_2d{};
  ContinuousSpace3DConfig continuous_3d{};
  std::vector<HamiltonianTerm> hamiltonian_terms{};
  OperatorGraph operator_graph{};
  std::vector<std::string> observables{"norm"};
  double target_error{1e-8};
  bool noisy{false};
  // Hint for scheduler/backend heuristics, not a strict execution bound.
  std::uint32_t expected_circuit_depth{0};
};

// Runtime execution options independent of physical model.
struct RunOptions {
  bool use_gpu{false};
  bool allow_mpi{false};
  bool enable_checkpoint{true};
  // When enabled, continuous backends export frame payloads for visualization.
  bool export_continuous_frames{false};
  // When enabled, 3D backends export the full nx*ny*nz density grid per frame
  // (for marching cubes isosurface rendering). Guarded by export_volume_max_voxels.
  bool export_volume_grid{false};
  std::size_t export_volume_max_voxels{131072};  // 128k voxel limit
  // Append-only controls for the stable continuous-export wrapper.
  std::uint32_t export_frame_stride{1};
  std::size_t export_points_max{2500};
  double export_density_floor_ratio{0.08};
  // When true, the 2D continuous backend computes and exports Wigner quasi-probability
  // frames (W(x,p), probability current, and transport coefficients).
  // Guarded separately because the Wigner computation is O(nx^2 log nx) per frame.
  bool export_wigner{false};
  // When > 0, statevector and Lindblad backends compute bipartite entanglement measures
  // each step, splitting qubit register at [0, entanglement_cut) | [entanglement_cut, n).
  std::uint32_t entanglement_cut{0};
  std::string checkpoint_path{"./checkpoints"};
  ResourceConstraints resources{};
};

// One gate operation in circuit IR.
struct CircuitOp {
  std::string name;
  std::vector<std::uint32_t> targets{};
  std::vector<std::uint32_t> controls{};
  std::vector<double> params{};
  // Optional named parameter references resolved against QuantumCircuit::parameter_map.
  // param_names[i] names the binding for params slot i; empty string means use params[i] directly.
  std::vector<std::string> param_names{};
};

// Quantum circuit input for gate-model simulation path.
struct QuantumCircuit {
  std::uint32_t qubits{1};
  std::vector<CircuitOp> operations{};
  std::vector<std::string> observables{};
  // 0 means exact expectation only; >0 enables sampling statistics.
  std::uint32_t shots{0};
  // Optional explicit noise-model selector for adapter payloads.
  // "none" keeps legacy explicit probability fields in force.
  std::string noise_model{"none"};
  // Optional user-supplied parameters for the selected noise model.
  std::map<std::string, double> noise_params{};
  double depolarizing_prob{0.0};
  double amplitude_damping_gamma{0.0};
  std::uint64_t seed{42};
  // Classical register width. 0 means no classical register; MEASURE ops write here by index.
  std::uint32_t clbits{0};
  // Named parameter bindings resolved at circuit execution time by CircuitOp::param_names.
  std::map<std::string, double> parameter_map{};
};

// Circuit execution output including exact and shot-based estimators.
struct CircuitResult {
  bool ok{true};
  std::string backend;
  double elapsed_ms{0.0};
  std::vector<std::complex<double>> state{};
  std::map<std::string, std::uint64_t> counts{};
  std::map<std::string, double> expectation_exact{};
  std::map<std::string, double> expectation_shot{};
  std::map<std::string, double> estimation_error{};
  std::map<std::string, double> shot_variance{};
  std::map<std::string, double> shot_stderr{};
  std::map<std::string, double> error_summary{};
  // Classical register outcomes from MEASURE operations (in clbit index order).
  std::vector<std::uint32_t> measurement_results{};
  std::string message;
};

// Runtime event record with metric payload.
struct Event {
  std::string name;
  std::map<std::string, double> metrics;
};

// One iteration snapshot from a VQE/QAOA optimizer run.
struct VqeFrame {
  std::uint32_t iteration{0};
  double energy{0.0};
  double gradient_norm{0.0};
  bool accepted{true};
  std::vector<double> params{};
};

// Options for variational (VQE/QAOA) optimization.
struct VqeOptions {
  // Number of qubits in the ansatz.
  std::uint32_t n_qubits{2};
  // Ansatz type: "hardware_efficient" | "qaoa".
  std::string ansatz_type{"hardware_efficient"};
  // Number of ansatz repetitions (HEA layers) or QAOA rounds.
  std::uint32_t n_layers{1};
  // Optimizer: "adam" | "gradient_descent".
  std::string optimizer{"adam"};
  // Step size / Adam learning rate.
  double learning_rate{0.1};
  // Adam β₁, β₂, ε hyper-parameters.
  double adam_beta1{0.9};
  double adam_beta2{0.999};
  double adam_epsilon{1e-8};
  // Termination conditions.
  std::uint32_t max_iterations{200};
  double convergence_tol{1e-6};
  // Cost Hamiltonian: VQE minimises ⟨H⟩ = Σ c_k ⟨P_k⟩.
  std::vector<HamiltonianTerm> cost_hamiltonian{};
  // User-supplied initial parameters (random if empty).
  std::vector<double> initial_params{};
  // Graph edges for QAOA MaxCut; each pair (i,j) is an undirected edge.
  std::vector<std::pair<std::uint32_t, std::uint32_t>> qaoa_edges{};
  // RNG seed for random initial parameters.
  std::uint64_t seed{42};
};

// Options for Quantum Phase Estimation.
struct QpeOptions {
  // Number of ancilla (precision) qubits; resolution = 1/2^n_ancilla.
  std::uint32_t n_ancilla{4};
  // Number of target qubits (must match the unitary).
  std::uint32_t n_target{1};
  // Unitary specification: "phase_gate" | "hamiltonian".
  //   "phase_gate"   — single-qubit phase gate P(phase_angle): |1⟩ eigenvalue e^{i*phase_angle}
  //   "hamiltonian"  — U = exp(-i*H*evolution_time) built from hamiltonian_terms
  std::string unitary_type{"phase_gate"};
  // For "phase_gate": angle in radians, eigenvalue e^{i*phase_angle} on |1⟩.
  double phase_angle{0.0};
  // For "hamiltonian": Pauli-string terms and evolution time.
  std::vector<HamiltonianTerm> hamiltonian_terms{};
  double evolution_time{1.0};
  // Number of shot-based measurement samples.
  std::uint32_t n_shots{4096};
  // RNG seed.
  std::uint64_t seed{42};
};

// ── M22: Quantum Error Correction ────────────────────────────────────────────

// One trial record from a QEC Monte Carlo run.
struct QecFrame {
  std::uint32_t trial{0};
  // Description of errors injected, e.g. "X0 X2" or "Z1".  "I" = no error.
  std::string errors_injected{};
  // Syndrome measurement outcomes as a binary string, e.g. "10" or "00011001".
  std::string syndrome{};
  // Correction applied by the classical decoder, e.g. "X1" or "I" or "decoded".
  std::string correction{};
  // true if the logical qubit was recovered correctly after correction.
  bool succeeded{false};
};

// One point on the logical-error-rate vs physical-error-rate threshold curve.
struct QecThresholdPoint {
  double p_physical{0.0};
  double p_logical_mc{0.0};
  double p_logical_theory{0.0};
};

// Options for a Quantum Error Correction simulation run.
struct QecOptions {
  // Code to simulate: "bit_flip_3" | "phase_flip_3" | "shor_9".
  std::string code{"bit_flip_3"};
  // Error type injected per qubit: "X" | "Z" | "depolarizing".
  std::string error_type{"X"};
  // Independent probability that each physical qubit suffers an error.
  double physical_error_rate{0.05};
  // Number of Monte Carlo trials.
  std::uint32_t n_trials{2000};
  // Maximum number of per-trial detail frames to record (first n frames).
  std::uint32_t n_frames_max{64};
  // Physical error rates to sweep for the threshold curve; empty = skip sweep.
  std::vector<double> sweep_error_rates{};
  // RNG seed.
  std::uint64_t seed{42};
};

// Result of a Quantum Error Correction simulation run.
struct QecResult {
  bool ok{true};
  std::string code{};
  // Number of physical qubits used by the code.
  std::uint32_t n_physical{0};
  // Number of syndrome ancilla qubits.
  std::uint32_t n_syndrome{0};
  // Empirical logical error rate from n_trials Monte Carlo trials.
  double logical_error_rate{0.0};
  // Theoretical logical error rate for the given physical_error_rate.
  double logical_error_rate_theory{0.0};
  // Per-trial detail records (first n_frames_max trials).
  std::vector<QecFrame> frames{};
  // Threshold curve: one entry per sweep_error_rate value.
  std::vector<QecThresholdPoint> threshold_curve{};
  double elapsed_ms{0.0};
  std::string message{};
};

// Result of a Quantum Phase Estimation run.
struct QpeResult {
  bool ok{true};
  std::uint32_t n_ancilla{0};
  std::uint32_t n_target{0};
  // Phase estimate in [0, 1): bin with highest count divided by 2^n_ancilla.
  double estimated_phase{0.0};
  // Known true phase (set by run_qpe when the ground truth is available).
  double true_phase{-1.0};
  // Per-bin measurement counts; length = 2^n_ancilla.
  std::vector<std::uint64_t> measurement_counts{};
  // Per-bin phase probabilities (exact, from statevector).
  std::vector<double> phase_probs{};
  double elapsed_ms{0.0};
  std::string message;
};

// ── M24: φ⁴ Scalar Quantum Field Theory ───────────────────────────────────

// One point in the parameter sweep (m² scan) for the φ⁴ phase transition.
struct Phi4SweepPoint {
  double m2{0.0};
  double ground_energy{0.0};
  // Order parameter: √(Σ⟨φᵢ²⟩/N) — grows from near-zero in symmetric phase.
  double order_parameter{0.0};
  // Energy gap Δ = E₁ - E₀ — closes at the quantum phase transition.
  double energy_gap{0.0};
};

// Options for the φ⁴ QFT simulation.
struct Phi4Options {
  // Number of lattice sites.
  std::uint32_t n_sites{4};
  // Bare mass squared m² (negative → Mexican-hat potential → SSB).
  double m2{1.0};
  // Quartic coupling λ in λ/4 · φ⁴.
  double lambda_coupling{1.0};
  // Nearest-neighbour hopping κ (spatial kinetic term).
  double kappa{1.0};
  // Local truncated Fock space dimension (number of oscillator levels).
  std::uint32_t d_local{8};
  // Scenario: "symmetric", "broken", "sweep", or "dynamics"
  std::string scenario{"symmetric"};
  // Number of m² values in the sweep.
  std::uint32_t n_sweep{20};
  // Sweep range for m².
  double m2_min{-3.0};
  double m2_max{2.0};
  // Tiny explicit symmetry-breaking field ε·Σφᵢ used to select a vacuum.
  double epsilon{0.05};
  std::uint64_t seed{42};
  // ── Dynamics (quantum quench) — used when scenario == "dynamics" ───────────
  // m² for the pre-quench ground-state preparation.
  double m2_initial{1.0};
  // m² for the post-quench Hamiltonian that drives the time evolution.
  double m2_final{-1.5};
  // Time step size for Krylov propagation.
  double dt{0.05};
  // Number of time steps to record.
  std::uint32_t n_steps{80};
};

// One time-step frame from a φ⁴ quantum-quench dynamics run.
struct Phi4DynamicsFrame {
  // Physical time t = step * dt.
  double time{0.0};
  // Per-site field expectation ⟨φᵢ(t)⟩ (length n_sites).
  std::vector<double> phi_expect{};
  // Per-site occupation ⟨nᵢ(t)⟩ (length n_sites).
  std::vector<double> occupation{};
  // Spatial-average order parameter √(Σ⟨φᵢ²⟩/N).
  double order_param{0.0};
  // Bipartite von Neumann entropy S = -Tr(ρ_A log ρ_A), cut at N/2.
  double entropy{0.0};
  // Instantaneous energy ⟨ψ(t)|H_f|ψ(t)⟩.
  double energy{0.0};
};

// Result of a φ⁴ QFT simulation.
struct Phi4Result {
  bool ok{true};
  std::uint32_t n_sites{0};
  std::uint32_t d_local{0};
  double m2{0.0};
  double lambda_coupling{0.0};
  // Ground state energy E₀.
  double ground_energy{0.0};
  // Energy gap Δ = E₁ - E₀.
  double energy_gap{0.0};
  // Order parameter √(Σ⟨φᵢ²⟩/N).
  double order_parameter{0.0};
  // Per-site expectation values.
  std::vector<double> phi_expectation{};   // ⟨φᵢ⟩
  std::vector<double> phi2_expectation{};  // ⟨φᵢ²⟩
  std::vector<double> occupation{};        // ⟨nᵢ⟩
  // Discretised marginal probability distribution p(φ₀) at site 0.
  std::vector<double> field_distribution{};
  // m² sweep results.
  std::vector<Phi4SweepPoint> sweep{};
  // Real-time dynamics frames from quantum quench (empty unless scenario == "dynamics").
  std::vector<Phi4DynamicsFrame> dynamics{};
  double elapsed_ms{0.0};
  std::string message;
};

// ── M28-E: XXZ Heisenberg Spin Chain ─────────────────────────────────────

// One point in the Jz/J parameter sweep (anisotropy scan).
struct HeisenbergSweepPoint {
  double Jz{0.0};           // z-axis exchange (Jz = J → isotropic; Jz = 0 → XX; Jz → ∞ → Ising)
  double ground_energy{0.0};
  double energy_gap{0.0};   // spin gap (0 in gapless phase)
  double Sz_total{0.0};     // total Sz of ground state
  double structure_pi{0.0}; // spin structure factor S(π) = (1/N) Σ_r (-1)^r ⟨S₀·S_r⟩
};

// Options for the 1D XXZ Heisenberg chain.
struct HeisenbergOptions {
  std::uint32_t n_sites{8};
  double J{1.0};     // transverse exchange (XX + YY part scaled by J/2)
  double Jz{1.0};    // longitudinal exchange (ZZ part); Jz = J → isotropic Heisenberg
  double h{0.0};     // uniform longitudinal field h Σ Z_i
  // "ground_state" | "sweep" | "dynamics"
  std::string scenario{"ground_state"};
  std::uint32_t n_sweep{20};
  double Jz_min{-2.0};
  double Jz_max{3.0};
  // Dynamics (quench from h≠0 to h=0):
  double h_initial{2.0};
  double dt{0.05};
  std::uint32_t n_steps{60};
};

// One dynamics frame from a Heisenberg quench.
struct HeisenbergFrame {
  double time{0.0};
  double Sz_total{0.0};        // ⟨Σ Z_i⟩/2
  double energy{0.0};          // ⟨H_final⟩
  double entropy{0.0};         // bipartite entropy at N/2
  std::vector<double> Sz_site{}; // ⟨Z_i⟩/2 per site
};

// Result of a Heisenberg chain simulation.
struct HeisenbergResult {
  bool ok{true};
  std::uint32_t n_sites{0};
  double J{0.0};
  double Jz{0.0};
  double h{0.0};
  double ground_energy{0.0};
  double energy_gap{0.0};
  double structure_pi{0.0};    // S(π) antiferromagnetic structure factor
  std::vector<double> spin_corr{};  // ⟨S₀·S_r⟩ = ¼⟨X₀X_r + Y₀Y_r + Z₀Z_r⟩
  std::vector<double> Sz_site{};    // ⟨Z_i⟩/2 per site in ground state
  std::vector<HeisenbergSweepPoint> sweep{};
  std::vector<HeisenbergFrame> frames{};
  double elapsed_ms{0.0};
  std::string message;
};

// ── M28-F: Bose-Hubbard Model ─────────────────────────────────────────────

// One point in the J/U parameter sweep.
struct BoseHubbardSweepPoint {
  double J{0.0};              // hopping
  double superfluid_op{0.0}; // |⟨a_0⟩| superfluid order parameter
  double ground_energy{0.0};
  double energy_gap{0.0};
  double double_occupancy{0.0}; // ⟨Σ n_i(n_i-1)⟩/N
};

// Options for the Bose-Hubbard model.
struct BoseHubbardOptions {
  std::uint32_t n_sites{4};
  std::uint32_t n_bosons{4};     // total boson number (default = n_sites)
  std::uint32_t n_max{4};        // max occupation per site
  double J{0.1};                 // hopping amplitude
  double U{1.0};                 // on-site interaction U/2 n(n-1)
  double mu{0.0};                // chemical potential
  // "ground_state" | "sweep"
  std::string scenario{"ground_state"};
  std::uint32_t n_sweep{20};
  double J_min{0.0};
  double J_max{0.5};
};

// Result of a Bose-Hubbard simulation.
struct BoseHubbardResult {
  bool ok{true};
  std::uint32_t n_sites{0};
  std::uint32_t n_bosons{0};
  std::uint32_t n_max{0};
  double J{0.0};
  double U{0.0};
  double ground_energy{0.0};
  double energy_gap{0.0};
  double superfluid_op{0.0};   // |⟨a_0⟩| averaged over sites
  double double_occupancy{0.0};
  std::vector<double> occupation{};   // ⟨n_i⟩ per site
  std::vector<double> number_dist{};  // P(n) marginal at site 0
  std::vector<BoseHubbardSweepPoint> sweep{};
  double elapsed_ms{0.0};
  std::string message;
};

// ── M29-G: Transverse-Field Ising Model ──────────────────────────────────

// One point in the transverse-field sweep h/J.
struct TFISweepPoint {
  double h{0.0};
  double ground_energy{0.0};
  double energy_gap{0.0};
  double order_param{0.0};    // |⟨Z_0⟩| (ferromagnetic order parameter)
  double corr_len{0.0};       // exponential correlation length from ⟨Z_0 Z_r⟩
};

// One Kibble-Zurek dynamics frame.
struct TFIFrame {
  double time{0.0};
  double h_now{0.0};
  double order_param{0.0};
  double energy{0.0};
  double entropy{0.0};
};

struct TFIOptions {
  int n_sites{10};  // must be ≤ 14 (dense exact-diag; 2^14×2^14 ≈ 2 GB)
  double J{1.0};        // ZZ coupling
  double h{0.5};        // transverse field (QPT at h = J)
  std::string scenario{"ground_state"};  // ground_state | sweep | dynamics
  int n_sweep{30};
  double h_min{0.0};
  double h_max{2.0};
  // dynamics: linear ramp h_quench → h over n_steps steps of dt
  double h_quench{2.0};
  double dt{0.05};
  int n_steps{60};
};

struct TFIResult {
  bool ok{true};
  int n_sites{0};
  double J{0.0};
  double h{0.0};
  double ground_energy{0.0};
  double energy_gap{0.0};
  double order_param{0.0};
  double corr_len{0.0};
  std::vector<double> spin_corr{};     // ⟨Z_0 Z_r⟩, r = 0..N-1
  std::vector<TFISweepPoint> sweep{};
  std::vector<TFIFrame>      frames{};
  double elapsed_ms{0.0};
  std::string message;
};

// ── M29-H: SSH Model (Su-Schrieffer-Heeger) ──────────────────────────────

// One point in the dimerization parameter sweep.
struct SSHSweepPoint {
  double delta{0.0};      // dimerization: t1 = t(1-δ), t2 = t(1+δ)
  double gap{0.0};
  double zak_phase{0.0};  // 0 or π
  double edge_weight{0.0};// weight of mid-gap modes on boundary sites
};

struct SSHOptions {
  int n_cells{10};            // number of unit cells → 2*n_cells sites
  double t{1.0};              // hopping scale
  double delta{0.5};          // dimerization: t1=t(1-δ), t2=t(1+δ)
  std::string scenario{"spectrum"};  // spectrum | sweep
  int n_sweep{30};
  double delta_min{-0.9};
  double delta_max{0.9};
};

struct SSHResult {
  bool ok{true};
  int n_cells{0};
  double t{0.0};
  double delta{0.0};
  std::vector<double> energies{};      // all 2*n_cells single-particle energies
  double gap{0.0};
  double zak_phase{0.0};              // Berry phase (0 or π)
  std::vector<double> edge_weight{};  // |ψ|² per site for left edge mode
  std::vector<SSHSweepPoint> sweep{};
  double elapsed_ms{0.0};
  std::string message;
};

// ── M30-I: Kitaev Chain (p-wave topological superconductor) ──────────────

// One point in the chemical-potential sweep.
struct KitaevSweepPoint {
  double mu{0.0};         // chemical potential (topo phase: |μ| < 2t)
  double gap{0.0};        // quasiparticle gap
  double topo_inv{0.0};   // topological invariant Q ∈ {-1, 0, +1} (sign of Pfaffian)
  double majorana_weight{0.0}; // |ψ|² of near-zero mode on boundary sites
};

struct KitaevOptions {
  int n_sites{20};
  double t{1.0};      // hopping
  double delta{1.0};  // p-wave pairing amplitude
  double mu{0.0};     // chemical potential (QPT at |μ| = 2t)
  std::string scenario{"spectrum"};  // spectrum | sweep
  int n_sweep{40};
  double mu_min{-4.0};
  double mu_max{4.0};
};

struct KitaevResult {
  bool ok{true};
  int n_sites{0};
  double t{0.0};
  double delta{0.0};
  double mu{0.0};
  std::vector<double> energies{};   // all 2N BdG quasi-energies
  double gap{0.0};
  double topo_inv{0.0};             // Q ∈ {-1, 0, +1}
  std::vector<double> majorana_weight{};  // |u+v|² per site of lowest mode
  std::vector<KitaevSweepPoint> sweep{};
  double elapsed_ms{0.0};
  std::string message;
};

// ── M30-J: Anderson Localization ─────────────────────────────────────────

// One disorder-averaged sample point.
struct AndersonSweepPoint {
  double W{0.0};          // disorder strength
  double mean_ipr{0.0};   // mean inverse participation ratio
  double mean_r{0.0};     // mean level-spacing ratio (Poisson ≈ 0.39, GOE ≈ 0.53)
  double loc_length{0.0}; // localization length from IPR ~ 1/ξ (1D: ξ ~ (W/2t)^{-2})
};

struct AndersonOptions {
  int n_sites{100};
  double t{1.0};      // hopping amplitude
  double W{2.0};      // disorder strength (uniform random in [-W/2, W/2])
  int n_samples{200}; // disorder realizations to average
  std::string scenario{"sweep"};  // single | sweep
  int n_sweep{20};
  double W_min{0.1};
  double W_max{8.0};
  unsigned int seed{42};
};

struct AndersonResult {
  bool ok{true};
  int n_sites{0};
  double t{0.0};
  double W{0.0};
  double mean_ipr{0.0};
  double mean_r{0.0};
  double loc_length{0.0};
  std::vector<double> ipr_spectrum{};  // IPR per eigenstate (one sample)
  std::vector<double> energies{};      // eigenvalues (one sample)
  std::vector<AndersonSweepPoint> sweep{};
  double elapsed_ms{0.0};
  std::string message;
};

// ── M34: 10-Fold Way Topological Classification (Altland-Zirnbauer) ─────────
//
// Classifies free-fermion Hamiltonians by T (time-reversal), C (particle-hole),
// and S = T*C (chiral) symmetries, yielding Z, Z2, 2Z, or trivial invariants.
// Numerically computes invariants for SSH (BDI d=1), QWZ (A d=2), Kitaev (D d=1).

struct TenfoldOptions {
  std::string az_class{"A"};      // symmetry class: A|AI|AII|AIII|BDI|CII|D|DIII|C|CI
  int         dimension{2};       // spatial dimension: 1, 2, or 3
  std::string model{"qwz"};       // model: ssh | qwz | kitaev
  std::string scenario{"classify"}; // classify | invariant | phase_diagram
  // SSH parameters (class BDI/AIII, d=1)
  double t1{1.0};                 // intra-cell hopping
  double t2{0.5};                 // inter-cell hopping
  // QWZ parameters (class A, d=2)
  double m{1.0};                  // mass / band inversion parameter
  double t{1.0};                  // hopping amplitude
  // Kitaev parameters (class D, d=1)
  double t_hop{1.0};              // nearest-neighbour hopping
  double mu{0.0};                 // chemical potential
  // Numerical
  int n_k{100};                   // k-grid points for invariant computation
  // Phase diagram sweep
  double param_min{-3.0};
  double param_max{3.0};
  int    n_sweep{60};
  std::string sweep_param{"m"};   // which parameter to sweep: m | t1 | t2 | mu
};

struct TenfoldResult {
  bool ok{false};
  std::string az_class;
  int         dimension{2};
  std::string invariant_type;       // "Z", "Z2", "2Z", "0"
  std::string scenario;
  std::string model;
  double      invariant_value{0.0}; // winding (Z/2Z) or Chern (Z)
  int         invariant_z2{0};      // 0 or 1 for Z2 classes
  std::string phase_label;          // "Topological (W=1)" | "Trivial (W=0)" etc.
  // Band data (SSH: dispersion for visualisation)
  std::vector<double> k_bands;      // k grid for SSH bands
  std::vector<double> e_plus;       // upper band E+(k)
  std::vector<double> e_minus;      // lower band E-(k)
  // Phase diagram
  std::vector<double>      sweep_vals;
  std::vector<double>      sweep_invariant;  // invariant value per sweep point
  std::vector<int>         sweep_z2;         // Z2 value per sweep point
  std::vector<std::string> sweep_phase;      // phase label per sweep point
  double elapsed_ms{0.0};
  std::string message;
};

// ── M37: Edge States / Surface Spectrum (BHZ strip geometry) ──────────────────
//
// Computes the energy band structure of the BHZ model in a strip geometry:
// y finite (N_y sites, open BCs), x periodic (kx a good quantum number).
// For each kx: diagonalize (4*N_y)×(4*N_y) block-tridiagonal Hamiltonian.
// Topological edge states (Z₂=1, 0<M/B<4) appear as in-gap bands colored
// by edge localization weight (red=edge, gray=bulk).

struct EdgeStatesOptions {
  double M{1.0};          // BHZ mass parameter
  double A{1.0};          // BHZ velocity (off-diagonal coupling)
  double B{1.0};          // BHZ curvature (quadratic k term)
  double C{0.0};          // particle-hole asymmetry C
  double D{0.0};          // particle-hole asymmetry D
  int    n_kx{60};        // number of kx grid points in [-π, π]
  int    n_y{20};         // number of sites in y direction (strip width)
  std::string scenario{"spectrum"};   // spectrum | phase_diagram
  std::string sweep_param{"M"};
  double param_min{-1.0};
  double param_max{5.0};
  int    n_sweep{40};
};

struct EdgeStatesResult {
  bool ok{false};
  double M{0.0}, A{0.0}, B{0.0};
  int    n_y{0};
  int    n_bands{0};       // = 4 * n_y
  std::string scenario;
  int    z2{0};
  std::string phase_label;
  // Band spectrum (flattened, size = n_kx * n_bands)
  std::vector<double> kx_vals;    // length n_kx
  std::vector<double> energies;   // [kx_idx * n_bands + band_idx]
  std::vector<double> edge_weight; // same shape, 0=bulk 1=edge
  // Phase diagram
  std::vector<double>      sweep_vals;
  std::vector<int>         sweep_z2;
  std::vector<std::string> sweep_phase;
  double elapsed_ms{0.0};
  std::string message;
};

// ── M39: Weyl Semimetal & Surface Fermi Arcs ──────────────────────────────
//
// Minimal 2-band cubic-lattice Weyl Hamiltonian:
// H(k) = sin(kx)σx + sin(ky)σy + (m - cos(kx) - cos(ky) - cos(kz))σz
// Weyl nodes at kx=ky=0, cos(kz)=m-2 (exist for 1<m<3).
// Chern number on kx-ky slices: C=+1 between nodes, 0 outside.
// Surface Fermi arcs on z-slab connect projected Weyl nodes in surface BZ.

struct WeylOptions {
  double m_param{2.0};     // controls Weyl node separation; nodes at kz=±arccos(m-2)
  int    n_k{20};          // kx-ky grid resolution for Chern/arc computations
  int    n_z{20};          // slab thickness (unit cells in z, open BCs)
  int    n_kz{40};         // number of kz values for Chern sweep
  double fermi_energy{0.0}; // Fermi level for arc detection
  std::string scenario{"fermi_arc"};  // "fermi_arc" | "chern_sweep" | "both"
  double param_min{1.1};
  double param_max{2.9};
  int    n_sweep{30};
};

struct WeylResult {
  bool ok{false};
  double m_param{0.0};
  int    n_k{0}, n_z{0};
  std::string scenario;
  // Weyl node positions
  double node_kz_pos{0.0};   // +arccos(m-2), chirality +1
  double node_kz_neg{0.0};   // -arccos(m-2), chirality -1
  // Chern number vs kz (on kx-ky slices)
  std::vector<double> kz_vals;
  std::vector<int>    chern_kz;
  // Surface Fermi arc: (kx, ky, weight) on n_k × n_k grid
  std::vector<double> arc_kx;       // n_k² flattened
  std::vector<double> arc_ky;       // n_k²
  std::vector<double> arc_weight;   // n_k², edge probability of state at Fermi energy
  std::vector<double> arc_energy;   // n_k², energy of state closest to Fermi level
  int    n_arc_pts{0};              // = n_k²
  // Bulk Weyl cone spectrum (kz scan at kx=ky=0)
  std::vector<double> cone_kz;
  std::vector<double> cone_E_plus;   // upper band E+(0,0,kz)
  std::vector<double> cone_E_minus;  // lower band E-(0,0,kz)
  // m_param sweep (Weyl node kz positions vs m)
  std::vector<double>      sweep_vals;
  std::vector<double>      sweep_node_kz;   // arccos(m-2) for each m, NaN if no nodes
  std::vector<std::string> sweep_phase;
  double elapsed_ms{0.0};
  std::string message;
};

// ── M40: Fu-Kane Superconductor — Majorana Zero Modes ────────────────────────
//
// BdG Hamiltonian for the surface of a 3D topological insulator proximitized
// with an s-wave superconductor, plus a Zeeman exchange field M:
//
//   H_BdG(k) = [h(k),  iΔσ_y]    h(k) = v_F(kx σ_y − ky σ_x) + M σ_z − μ I
//              [−iΔσ_y*, −h^T(−k)]
//
// Phase transition at M_c = √(μ² + Δ²): topological (C=1) for M < M_c.
// Chiral Majorana edge mode crosses E=0 on the strip spectrum in the topo phase.
struct FuKaneOptions {
  double v_F{1.0};      // Fermi velocity of the TI surface Dirac cone
  double M_val{0.5};    // Zeeman exchange field (drives topo transition at M_c=√(μ²+Δ²))
  double mu{0.0};       // chemical potential
  double Delta{0.3};    // s-wave SC proximity gap
  int    n_k{40};       // kx-ky grid resolution for Chern / phase diagram
  int    n_y{30};       // strip width (sites along y, open BCs) for edge-state calc
  int    n_kx{80};      // kx points for band structure / strip spectrum
  int    n_M{40};       // number of M values for phase diagram sweep
  int    n_mu{40};      // number of μ values for phase diagram sweep
  double M_max{2.0};    // |M| range for phase diagram
  double mu_max{2.0};   // |μ| range for phase diagram
  // "bulk" | "edge" | "phase_diagram" | "both"
  std::string scenario{"bulk"};
};

struct FuKaneResult {
  bool ok{false};
  std::string scenario;
  // Parameters
  double v_F{0.0}, M_val{0.0}, mu{0.0}, Delta{0.0};
  // Phase label at the given (M, μ, Δ)
  int    chern{0};
  std::string phase_label;
  double gap{0.0};           // min quasiparticle gap across BZ
  double M_critical{0.0};    // √(μ² + Δ²)
  // BdG bulk band structure (kx scan at ky=0)
  std::vector<double> kx_bulk;   // n_kx values in [−π, π]
  std::vector<double> E0, E1, E2, E3;  // four BdG branches (sorted)
  // Strip edge-state spectrum (kx scan, n_y-site strip with open BCs in y)
  std::vector<double> kx_edge;       // n_kx values
  std::vector<double> E_edge;        // flattened [n_kx × 4*n_y], all eigenvalues
  int    n_edge_evals{0};            // eigenvalues per kx = 4 * n_y
  // Phase diagram: Chern number on (M, μ) grid
  std::vector<double> pd_M_vals;     // n_M values in [0, M_max]
  std::vector<double> pd_mu_vals;    // n_mu values in [−mu_max, mu_max]
  std::vector<int>    pd_chern;      // flattened [n_M × n_mu]
  double elapsed_ms{0.0};
  std::string message;
};

// ── M41: Lutchyn-Oreg Majorana Nanowire ──────────────────────────────────────
//
// 1D semiconductor nanowire with Rashba spin-orbit coupling, proximity-induced
// s-wave SC gap, and Zeeman field V_z (applied via magnetic field along wire).
// BdG Hamiltonian (lattice, tight-binding):
//   H(k) = (2t(1−cos k) − μ) τ_z + α sin(k) σ_y τ_z + V_z σ_x + Δ τ_x
// Phase transition: V_z,c = √(μ² + Δ²)
//   V_z < V_z,c → trivial SC (Z₂ = +1)
//   V_z > V_z,c → topological SC (Z₂ = −1), Majorana zero modes at wire ends
// Scenarios: "bulk" | "wire" | "phase_diagram" | "both"

struct LutchynOregOptions {
  double t{1.0};       // hopping amplitude (sets bandwidth 4t)
  double alpha{0.5};   // Rashba spin-orbit coupling
  double mu{0.0};      // chemical potential
  double Delta{0.25};  // s-wave SC proximity gap
  double V_z{0.0};     // Zeeman field (drives topological transition at V_z,c = √(μ²+Δ²))
  int    n_k{200};     // k-grid points for bulk BdG band structure
  int    n_sites{80};  // wire length (sites, open BCs) for real-space Majorana calc
  int    n_Vz{60};     // V_z points for phase diagram sweep
  int    n_mu{60};     // μ points for phase diagram sweep
  double Vz_max{2.0};  // V_z range for phase diagram
  double mu_max{2.0};  // |μ| range for phase diagram
  std::string scenario{"bulk"};  // "bulk" | "wire" | "phase_diagram" | "both"
};

struct LutchynOregResult {
  bool ok{false};
  std::string scenario;
  // Operating-point parameters
  double t{0.0}, alpha{0.0}, mu{0.0}, Delta{0.0}, V_z{0.0};
  // Phase at this (V_z, μ, Δ)
  int    z2{0};               // topological invariant: +1 trivial, -1 topological
  std::string phase_label;
  double gap{0.0};            // min |E| over BZ
  double V_z_critical{0.0};  // √(μ² + Δ²)
  // Bulk BdG band structure (4 bands vs k ∈ [−π, π])
  std::vector<double> k_bulk;
  std::vector<double> E0, E1, E2, E3;  // sorted branches
  // Real-space wire spectrum: eigenvalues and edge LDOS
  std::vector<double> E_wire;          // 4*n_sites eigenvalues (sorted)
  std::vector<double> ldos_left;       // LDOS at left end vs E  (same length)
  std::vector<double> ldos_right;      // LDOS at right end vs E (same length)
  // Majorana oscillations: gap vs wire length L
  std::vector<int>    osc_L;           // wire lengths
  std::vector<double> osc_gap;         // min |E| at each length
  // Phase diagram: Z₂(V_z, μ) on 2D grid
  std::vector<double> pd_Vz_vals;      // n_Vz values in [0, Vz_max]
  std::vector<double> pd_mu_vals;      // n_mu values in [−mu_max, mu_max]
  std::vector<int>    pd_z2;           // flattened [n_Vz × n_mu]
  double elapsed_ms{0.0};
  std::string message;
};

// ── M42: Kitaev Honeycomb Model ───────────────────────────────────────────────
//
// Exactly solvable spin-1/2 model on the honeycomb lattice:
//   H = -Jx Σ_x σ_i^x σ_j^x  -Jy Σ_y σ_i^y σ_j^y  -Jz Σ_z σ_i^z σ_j^z
//       -κ Σ_NNN ν_ij c_i c_j   (magnetic field → effective NNN term)
//
// Solved via Majorana representation in zero-flux sector.
// 2-band Bloch Hamiltonian:
//   H(k) = [[m(k),  if(k)],
//            [-if*(k), -m(k)]]
//   f(k) = Jz + Jx e^{i k·a1} + Jy e^{i k·a2},  a1=(1,0), a2=(1/2,√3/2)
//   m(k) = 2κ(sin(k·a1)+sin(k·a2)+sin(k·(a2-a1)))   [NNN from field]
//
// Phase diagram (on Jx+Jy+Jz=1 simplex):
//   A_x phase: Jx > Jy+Jz  (gapped)
//   A_y phase: Jy > Jx+Jz  (gapped)
//   A_z phase: Jz > Jx+Jy  (gapped)
//   B phase:   triangle inequality satisfied  (gapless Dirac; gapped Chern C=±1 with κ≠0)
//
// Scenarios: "spectrum" | "phase_diagram" | "chern" | "both"

struct KitaevHcOptions {
  double Jx{1.0};    // x-bond coupling
  double Jy{1.0};    // y-bond coupling
  double Jz{1.0};    // z-bond coupling
  double kappa{0.0}; // effective NNN coupling from magnetic field h (∝ hx·hy·hz/J²)
  int    n_k{60};    // BZ grid per reciprocal direction (n_k² k-points total)
  int    n_pd{80};   // simplex grid resolution for phase diagram
  std::string scenario{"spectrum"};  // "spectrum"|"phase_diagram"|"chern"|"both"
};

struct KitaevHcResult {
  bool ok{false};
  std::string scenario;
  double Jx{0}, Jy{0}, Jz{0}, kappa{0};
  // Phase classification at given (Jx,Jy,Jz,κ)
  std::string phase_label;  // "A_x"|"A_y"|"A_z"|"B (gapless)"|"B (topo, C=±1)"
  double gap{0.0};          // min E+(k) over BZ
  int    chern{0};          // Chern number (0 in A phases, ±1 in B+κ phase)
  // Band structure: kx,ky on BZ grid, E±(k)
  std::vector<double> kx_bz;   // n_k² points
  std::vector<double> ky_bz;
  std::vector<double> E_plus;  // upper band
  std::vector<double> E_minus; // lower band
  // Phase diagram: phase label at each simplex point
  // Parametrize simplex as (Jx,Jy,Jz) with Jx+Jy+Jz=1, Jx,Jy,Jz≥0
  // Store as ternary grid: pd_Jx[i], pd_Jy[i], pd_Jz[i], pd_phase[i]
  std::vector<double>      pd_Jx, pd_Jy, pd_Jz;
  std::vector<std::string> pd_phase;
  std::vector<double>      pd_gap;
  double elapsed_ms{0.0};
  std::string message;
};

// ── M43: BBH Quadrupole Insulator (Higher-Order Topological Insulator) ───────
//
// Benalcazar-Bernevig-Hughes (BBH) model on a 2D square lattice.
// 4-band model with pi-flux per plaquette; quantised bulk quadrupole moment
// q_xy = 1/2 (mod 1) characterises the higher-order topological phase.
//
// Bulk Hamiltonian (4-band, momentum space):
//   H(k) = (gamma_x + lambda_x cos kx) Gamma_4
//         + lambda_x sin(kx) Gamma_3
//         + (gamma_y + lambda_y cos ky) Gamma_2
//         + lambda_y sin(ky) Gamma_1
//
//   Gamma_1..4 are 4x4 Dirac matrices: Gamma_i = sigma_i ⊗ sigma_z etc.
//   See BBH Phys.Rev.B 96, 245115 (2017).
//
// Topological condition: |gamma_x/lambda_x| < 1 AND |gamma_y/lambda_y| < 1
//   → quadrupole phase with 4 in-gap corner states at E≈0 on finite lattice
//
// Scenarios: "spectrum" | "corner" | "phase_diagram" | "both"

struct BBHOptions {
  double gamma_x{0.5};   // intra-cell hopping x
  double lambda_x{1.0};  // inter-cell hopping x
  double gamma_y{0.5};   // intra-cell hopping y
  double lambda_y{1.0};  // inter-cell hopping y
  int    n_k{60};        // k-grid resolution for bulk spectrum
  int    n_sites{8};     // linear size of finite lattice (n_sites x n_sites unit cells)
  int    n_pd{60};       // grid resolution per axis for phase diagram
  std::string scenario{"both"};  // "spectrum"|"corner"|"phase_diagram"|"both"
};

struct BBHResult {
  bool ok{false};
  std::string scenario;
  double gamma_x{0}, lambda_x{0}, gamma_y{0}, lambda_y{0};
  std::string phase_label;  // "Trivial" or "Quadrupole (q_xy=1/2)"
  double gap{0.0};          // bulk gap
  // Bulk band structure: kx,ky over BZ, 4 bands
  std::vector<double> kx_bulk;   // n_k^2 points
  std::vector<double> ky_bulk;
  std::vector<double> E0, E1, E2, E3;  // bands sorted ascending
  // Finite lattice open-BC spectrum
  std::vector<double> E_open;    // all eigenvalues sorted
  std::vector<double> ldos_corner; // LDOS on 4 corner sites (summed near-zero modes)
  // Phase diagram: quadrupole invariant over (gamma_x/lambda_x, gamma_y/lambda_y) grid
  std::vector<double> pd_gx_over_lx;   // grid x-axis: gamma_x/lambda_x
  std::vector<double> pd_gy_over_ly;   // grid y-axis: gamma_y/lambda_y
  std::vector<int>    pd_topo;         // 1=topological, 0=trivial
  double elapsed_ms{0.0};
  std::string message;
};

// ── M44: Floquet Topological Insulator ───────────────────────────────────────
//
// Periodically driven 2D tight-binding model (square lattice) with a 5-step
// drive protocol (Rudner et al., Phys. Rev. X 3, 031005, 2013).
//
// Each period T is split into 5 equal steps; in step i the nearest-neighbour
// hopping along direction i is turned on with amplitude J_i.
// The Floquet operator (stroboscopic evolution over one period) is:
//   U(T) = U_5 U_4 U_3 U_2 U_1
//
// Quasienergy spectrum ε ∈ (-π/T, π/T].  Two topological invariants:
//   W_0  = winding number in the 0-gap  (ε≈0)
//   W_pi = winding number in the π-gap  (ε≈±π/T)
//
// Anomalous Floquet phase: all Chern numbers = 0 but W_pi = ±1.
//
// Scenarios: "spectrum" | "edge" | "winding" | "both"

struct FloquetTopoOptions {
  double J{1.0};        // hopping amplitude (uniform across all steps)
  double T{1.0};        // drive period (sets energy scale ε∈(-π/T,π/T])
  int    n_k{60};       // k-grid per direction for bulk quasienergy spectrum
  int    n_y{40};       // strip width (unit cells) for edge-state calculation
  int    n_J{50};       // number of J values for winding-number sweep
  double J_max{2.5};    // maximum J for winding-number sweep
  std::string scenario{"both"};  // "spectrum"|"edge"|"winding"|"both"
};

struct FloquetTopoResult {
  bool ok{false};
  std::string scenario;
  double J{0}, T{0};
  int W_0{0}, W_pi{0};          // winding numbers in 0-gap and pi-gap
  std::string phase_label;       // e.g. "Anomalous Floquet (W_pi=1)"
  // Bulk quasienergy: kx,ky → two quasienergies per k-point
  std::vector<double> kx_bulk, ky_bulk;
  std::vector<double> eps_minus, eps_plus;  // lower/upper quasienergy band
  // Edge-state spectrum: kx → quasienergies of strip (2 n_y bands)
  std::vector<double> kx_edge;
  std::vector<double> eps_edge;  // n_k * 2*n_y values (all bands)
  // Winding sweep: J values → W_pi(J)
  std::vector<double> sweep_J;
  std::vector<int>    sweep_W_pi;
  double elapsed_ms{0.0};
  std::string message;
};

// ── M36: Wilson Loop & Wannier Charge Centers (BHZ Z₂ Topological Insulator) ─
//
// Computes the Wilson loop spectrum (Wannier Charge Centers, WCC) for the
// Bernevig-Hughes-Zhang (BHZ) 2D topological insulator model.
// W(ky) = Π_{kx} Q(kx,ky)† Q(kx+Δkx,ky)  [2×2 product of overlaps]
// WCC θᵢ(ky) = -arg(λᵢ(W)) / 2π ∈ [0,1)
// Z₂ = parity of WCC crossings of θ=0.5 as ky sweeps [0, π].
// Phase diagram: topological when 0 < M/B < 4 (gap closes at M=0, M=4B).

struct WilsonLoopOptions {
  double M{1.0};          // BHZ mass parameter (band-inversion strength)
  double A{1.0};          // BHZ velocity parameter (off-diagonal coupling)
  double B{1.0};          // BHZ curvature parameter (quadratic k term)
  double C{0.0};          // particle-hole asymmetry (set 0 for pure TI)
  double D{0.0};          // particle-hole asymmetry (set 0 for pure TI)
  int    n_kx{40};        // k-grid points along kx for Wilson loop product
  int    n_ky{40};        // k-grid points along ky for WCC spectrum
  std::string scenario{"wcc"};       // wcc | phase_diagram
  std::string sweep_param{"M"};      // parameter to sweep (phase_diagram)
  double param_min{-1.0};
  double param_max{5.0};
  int    n_sweep{40};
};

struct WilsonLoopResult {
  bool ok{false};
  double M{0.0}, A{0.0}, B{0.0};
  std::string scenario;
  int    z2{0};            // Z₂ invariant (0 = trivial, 1 = topological)
  std::string phase_label; // "Topological (Z₂=1)" | "Trivial (Z₂=0)"
  double gap_gamma{0.0};   // energy gap at Γ = (0,0)
  double gap_m_pt{0.0};    // energy gap at M = (π,π)
  // WCC spectrum
  std::vector<double> ky_vals;  // ky grid [0, π]
  std::vector<double> wcc1;     // lower WCC θ₁(ky) ∈ [0,1)
  std::vector<double> wcc2;     // upper WCC θ₂(ky) ∈ [0,1)
  // Phase diagram sweep
  std::vector<double>      sweep_vals;
  std::vector<int>         sweep_z2;
  std::vector<std::string> sweep_phase;
  double elapsed_ms{0.0};
  std::string message;
};

// ── M35: Second Chern Number & 4D Quantum Hall Effect ─────────────────────
//
// Computes the second Chern number C₂ of the Zhang-Hu 4D QHE model
// H(k) = Σ dₐ(k) Γₐ  (a=1..5, Γ matrices = tensor products of Paulis).
// Uses the 4D Abelian Fukui-Hatsugai-Suzuki lattice-gauge algorithm.
// Phase diagram: C₂ = 0 (|m|>4), ±1 (2<|m|<4), ±3 (0<|m|<2).

struct SecondChernOptions {
  double m{1.0};               // mass / band-inversion parameter
  double t{1.0};               // hopping amplitude
  int    n_k{6};               // k-grid points per dimension (6^4 = 1296 cubes)
  std::string scenario{"invariant"};  // invariant | phase_diagram
  double param_min{-5.0};
  double param_max{5.0};
  int    n_sweep{40};
  std::string sweep_param{"m"};  // only "m" supported currently
};

struct SecondChernResult {
  bool ok{false};
  double m{0.0}, t{0.0};
  std::string scenario;
  int    second_chern{0};
  std::string phase_label;     // "C₂=+3 (4D QH)" | "C₂=±1" | "Trivial (C₂=0)"
  std::vector<double>      sweep_vals;
  std::vector<int>         sweep_c2;
  std::vector<std::string> sweep_phase;
  double elapsed_ms{0.0};
  std::string message;
};

// ── M38: Haldane Model (Quantum Anomalous Hall Effect) ─────────────────────
//
// 2-band tight-binding model on honeycomb lattice with complex NNN hopping.
// H(k) = h0 I₂ + hx σx + hy σy + hz σz
// Chern number C₁ via Fukui-Hatsugai-Suzuki lattice gauge formula on BZ grid.
// Phase diagram: C₁=±1 when |M/t2| < 3√3|sin(φ)|, trivial otherwise.

struct HaldaneOptions {
  double t1{1.0};           // NN hopping amplitude
  double t2{0.1};           // NNN hopping magnitude
  double phi{1.5707963};    // NNN phase φ (Haldane flux); π/2 maximizes gap
  double M_val{0.0};        // staggered on-site potential (inversion breaking)
  int    n_k{40};           // k-grid resolution per dimension (n_k² points in BZ)
  int    n_y{20};           // strip width (unit cells, for ribbon spectrum)
  std::string scenario{"phase_diagram"};  // "chern" | "phase_diagram" | "spectrum"
  std::string sweep_param{"M"};  // "M" | "phi"
  double param_min{-6.0};        // sweep range min (in units of t2)
  double param_max{6.0};         // sweep range max
  int    n_sweep{60};
};

struct HaldaneResult {
  bool ok{false};
  double t1{0.0}, t2{0.0}, phi{0.0}, M_val{0.0};
  int    n_k{0};
  std::string scenario;
  int    chern{0};               // first Chern number C₁
  std::string phase_label;       // "QAH (C₁=+1)" | "QAH (C₁=-1)" | "Trivial (C₁=0)"
  // Berry curvature map over BZ (n_k × n_k, flattened row-major)
  std::vector<double> kx_bz;    // n_k² kx coordinates (Cartesian)
  std::vector<double> ky_bz;    // n_k² ky coordinates
  std::vector<double> berry_curv; // n_k² Berry curvature values (sum = 2π·C₁)
  // Ribbon spectrum (kx ∈ [-π,π], 2*n_y bands)
  std::vector<double> ribbon_kx;
  std::vector<double> ribbon_energies;   // length n_kx_ribbon * 2*n_y
  std::vector<double> ribbon_edge_weight;
  int    n_ribbon_bands{0};
  // Phase diagram sweep
  std::vector<double>      sweep_vals;
  std::vector<int>         sweep_chern;
  std::vector<std::string> sweep_phase;
  double elapsed_ms{0.0};
  std::string message;
};

// ── M33: Spectral Functions & Imaginary-Time Evolution ───────────────────
//
// Dynamical structure factor S(k,omega) = sum_n |<n|S^z_k|0>|^2 delta(omega-(E_n-E_0))
// with Lorentzian broadening eta.  Imaginary-time projection e^{-tau H}|psi> -> |GS>.

struct SpectralOptions {
  int    n_sites{12};        // chain length (N, Sz=0 sector; N even)
  double J{1.0};             // Heisenberg exchange coupling
  double delta{1.0};         // XXZ anisotropy (1=Heisenberg, 0=XX, <0=ferromagnet)
  double h{0.0};             // uniform longitudinal field
  std::string scenario{"spectral"};  // spectral | imagtime | both
  // Spectral function parameters
  int    n_k{20};            // k-space sample points in [0, 2pi]
  int    n_omega{200};       // omega grid points
  double omega_min{-4.0};
  double omega_max{4.0};
  double eta{0.15};          // Lorentzian broadening half-width
  // Imaginary-time parameters
  double tau_max{4.0};
  int    n_tau{60};
};

struct SpectralResult {
  bool   ok{false};
  int    n_sites{0};
  double J{0.0}, delta{0.0}, h{0.0};
  std::string scenario;
  double E_ground{0.0};              // ground-state energy per site
  // Spectral function  S(k, omega): n_k x n_omega row-major (k outer, omega inner)
  std::vector<double> k_vals;        // k in [0, 2*pi], length n_k
  std::vector<double> omega_vals;    // omega grid, length n_omega
  std::vector<double> spectral;      // S(k,omega), flattened n_k*n_omega
  // Imaginary-time evolution
  std::vector<double> tau_vals;      // tau grid
  std::vector<double> energy_tau;    // <E>(tau) converging to E_0
  std::vector<double> fidelity_tau;  // |<GS|psi(tau)>|^2 converging to 1
  double elapsed_ms{0.0};
  std::string message;
};

// ── M32: Many-Body Localization (disordered XXZ chain) ────────────────────

struct MBLSweepPoint {
  double W{0.0};
  double r_mean{0.0};       // mean level-spacing ratio <r>
  double entropy_mean{0.0}; // mean half-chain entanglement entropy <S_EE>
  double ipr_mean{0.0};     // mean IPR of middle-spectrum eigenstates
};

struct MBLOptions {
  int    n_sites{10};        // chain length (N even, Sz=0 sector)
  double J{1.0};             // Heisenberg coupling
  double delta{1.0};         // XXZ anisotropy (1 = Heisenberg, 0 = XX)
  double W{3.0};             // disorder strength at single point
  std::string scenario{"single"};  // single | sweep
  double W_min{0.5};
  double W_max{8.0};
  int    n_sweep{16};        // number of W values
  int    n_samples{10};      // disorder realizations per W (sweep)
  unsigned int seed{42};
};

struct MBLResult {
  bool   ok{false};
  int    n_sites{0};
  double J{0.0}, delta{0.0}, W{0.0};
  std::string scenario;
  // single-realization data
  std::vector<double> evals;       // sorted eigenvalues in Sz=0 sector
  std::vector<double> entropies;   // S_EE per eigenstate (middle half of spectrum)
  std::vector<double> iprs;        // IPR per eigenstate (middle half)
  std::vector<double> r_vals;      // individual r_n values
  double entropy_mean{0.0};
  double r_mean{0.0};
  double ipr_mean{0.0};
  // sweep data
  std::vector<MBLSweepPoint> sweep;
  double elapsed_ms{0.0};
  std::string message;
};

// ── M31-K: 2D Chern Insulator (Qi-Wu-Zhang model) ────────────────────────
//
// H(k) = sin(kx)σx + sin(ky)σy + (m + cos(kx) + cos(ky))σz
// Topological phases:  C = +1 for -2 < m < 0
//                      C = -1 for -4 < m < -2
//                      C =  0 for m > 0 or m < -4

struct ChernSweepPoint {
  double m{0.0};       // mass parameter
  double gap{0.0};     // bulk band gap
  double chern{0.0};   // Chern number C ∈ {-1, 0, +1} (near-integer after FHS)
};

struct ChernOptions {
  int n_sites{20};     // Ny for strip-geometry (edge states)
  double m{-1.0};      // mass parameter (QPT at m = 0, -2, -4)
  double t{1.0};       // hopping amplitude (scales all off-diagonal terms)
  std::string scenario{"bands"};  // bands | edge | sweep
  int n_k{40};         // BZ k-grid: n_k × n_k for Chern, n_k kx points for edge
  int n_sweep{30};
  double m_min{-5.0};
  double m_max{1.0};
};

struct ChernResult {
  bool ok{false};
  int n_sites{0};
  double m{0.0};
  double t{0.0};
  std::string scenario;
  // bands scenario: Berry curvature + band energies on n_k×n_k BZ grid (row-major kx→ky)
  std::vector<double> lower_band{};   // E−(kx,ky) flattened [n_k*n_k]
  std::vector<double> upper_band{};   // E+(kx,ky)
  std::vector<double> berry_curv{};   // Ω(kx,ky) Berry curvature of lower band
  double chern{0.0};                  // rounded Chern number
  double gap{0.0};                    // minimum bulk band gap
  // edge scenario: spectrum of a finite strip vs kx
  std::vector<double> kx_vals{};               // kx / (2π) ∈ [0,1]
  std::vector<std::vector<double>> edge_spectrum{};  // [n_k][2*n_sites] energies
  // sweep: scan m from m_min to m_max
  std::vector<ChernSweepPoint> sweep{};
  double elapsed_ms{0.0};
  std::string message;
};

// ── M31-L: Hofstadter Butterfly ───────────────────────────────────────────
//
// 2D square lattice in perpendicular magnetic field B, rational flux φ = p/q
// per plaquette (Peierls substitution, Landau gauge).
// Harper equation gives q bands for each rational flux.
// Butterfly: allowed energies vs φ ∈ [0,1] → fractal spectrum.
// TKNN band Chern numbers via FHS on 2D magnetic BZ.

struct HofstadterOptions {
  double t{1.0};
  std::string scenario{"butterfly"};  // butterfly | bands
  double phi{0.25};   // for bands scenario (e.g. 1/4, 1/3, …)
  int q_max{24};      // max denominator scanned for butterfly
  int n_k{40};        // kx points for bands scenario
  int n_ky{10};       // ky samples per flux value (butterfly density)
};

struct HofstadterResult {
  bool ok{false};
  // butterfly: sparse (phi, energy) pairs
  std::vector<double> phi_vals{};  // one entry per eigenvalue across all φ
  std::vector<double> evals{};     // eigenvalue (same length as phi_vals)
  // bands: kx-resolved spectrum for a fixed rational φ
  double phi_used{0.0};            // actual φ used (nearest rational p/q)
  int n_bands{0};                  // = q for the chosen flux
  std::vector<double> kx_vals{};   // kx ∈ [0, 2π]
  std::vector<std::vector<double>> bands{};  // bands[ik][band_index]
  double elapsed_ms{0.0};
  std::string message;
};

// ── M27-C: 1D Hubbard Model ───────────────────────────────────────────────

// One point in the U/t parameter sweep.
struct HubbardSweepPoint {
  double U{0.0};
  double ground_energy{0.0};
  double double_occupancy{0.0};    // D = ⟨Σ n_{i↑}n_{i↓}⟩/N
  double spin_structure_pi{0.0};   // S(k=π) antiferromagnetic structure factor
  double energy_gap{0.0};          // E₁ − E₀ within same particle-number sector
};

// Options for the 1D Hubbard model simulation.
struct HubbardOptions {
  std::uint32_t n_sites{6};
  double t_hop{1.0};     // hopping amplitude t
  double U{4.0};         // on-site Coulomb repulsion U
  std::uint32_t n_up{0}; // spin-up electrons (0 = n_sites/2)
  std::uint32_t n_dn{0}; // spin-down electrons (0 = n_sites/2)
  // "ground_state" | "sweep"
  std::string scenario{"ground_state"};
  std::uint32_t n_sweep{20};
  double U_min{0.0};
  double U_max{12.0};
};

// Result of a Hubbard model simulation.
struct HubbardResult {
  bool ok{true};
  std::uint32_t n_sites{0};
  std::uint32_t n_up{0};
  std::uint32_t n_dn{0};
  double t_hop{0.0};
  double U{0.0};
  double ground_energy{0.0};
  double energy_gap{0.0};
  double double_occupancy{0.0};
  double spin_structure_pi{0.0};
  std::vector<double> spin_corr{};      // ⟨S^z_0 S^z_r⟩, r = 0..N−1
  std::vector<double> momentum_dist{};  // n(k) = n↑(k)+n↓(k), k = 0..N−1
  std::vector<HubbardSweepPoint> sweep{};
  double elapsed_ms{0.0};
  std::string message;
};

// ── M27-D: Floquet Periodic Driving ───────────────────────────────────────

// One stroboscopic snapshot at t = n·T.
struct FloquetFrame {
  std::uint32_t period{0};
  double time{0.0};
  double magnetization_z{0.0};  // ⟨Σ Z_i⟩/N
  double energy{0.0};           // ⟨H₀⟩
  double entropy{0.0};          // bipartite von Neumann entropy at N/2 cut
};

// Options for the Floquet kicked-Ising simulation.
struct FloquetOptions {
  std::uint32_t n_sites{8};
  double J{1.0};           // Ising ZZ coupling
  double h_z{0.809};       // longitudinal field (breaks integrability when ≠ 0)
  double drive_amp{1.5};   // kick amplitude h_x = A (effective X rotation per period)
  double drive_omega{4.5}; // drive frequency ω; period T = 2π/ω
  std::uint32_t n_periods{40};
  // "dynamics" | "spectrum" | "heating"
  std::string scenario{"dynamics"};
  std::uint32_t n_scan{25};          // number of amplitude points in spectrum scan
  double scan_amp_max{3.0};          // max h_x for spectrum scan
};

// Result of a Floquet simulation.
struct FloquetResult {
  bool ok{true};
  std::uint32_t n_sites{0};
  double J{0.0};
  double h_z{0.0};
  double drive_amp{0.0};
  double drive_omega{0.0};
  std::vector<FloquetFrame> frames{};  // stroboscopic dynamics
  std::vector<double> quasi_energies{};  // sorted quasi-energies in [−π/T, π/T)
  double mean_gap_ratio{0.0};            // level spacing gap ratio r (Poisson≈0.39, GUE≈0.60)
  // Spectrum scan: drive_amp values and corresponding quasi-energy arrays
  std::vector<double> scan_amps{};
  std::vector<std::vector<double>> scan_qe{};  // [n_scan][dim]
  double elapsed_ms{0.0};
  std::string message;
};

// ── M23: Lattice Schwinger model (1+1D QED) ────────────────────────────────

// One time-step snapshot of the Schwinger model state.
struct SchwingerFrame {
  double time{0.0};
  // Per-site charge density q_n = <(σz_n + (-1)^n)/2> (particle - antiparticle).
  std::vector<double> charge_density{};
  // Per-link electric field <L_l> (length n_sites-1).
  std::vector<double> electric_field{};
  // Chiral condensate Σ = -(1/N) Σ_n (-1)^n <σz_n>.
  double chiral_condensate{0.0};
  // Total energy <H>.
  double energy{0.0};
  // Total particle number (sum of positive charges).
  double particle_number{0.0};
};

// Options for the Schwinger model simulation.
struct QfieldOptions {
  // Number of lattice sites (must be even; default 8).
  std::uint32_t n_sites{8};
  // Bare fermion mass in units of hopping J (can be negative).
  double mass{0.5};
  // Gauge coupling strength g² in units of J.
  double coupling{0.3};
  // Background electric field α (uniform, integer or half-integer).
  double background{0.0};
  // Scenario: "vacuum", "string", "dynamics"
  std::string scenario{"vacuum"};
  // Time step for Trotter evolution.
  double dt{0.1};
  // Number of time steps.
  std::uint32_t n_steps{50};
  // Number of mass values to sweep for ground state observables.
  std::uint32_t n_mass_sweep{20};
  // Random seed.
  std::uint64_t seed{42};
};

// Result of a Schwinger model simulation run.
struct QfieldResult {
  bool ok{true};
  std::uint32_t n_sites{0};
  double mass{0.0};
  double coupling{0.0};
  // Ground state energy.
  double ground_energy{0.0};
  // Ground state chiral condensate.
  double chiral_condensate{0.0};
  // Time evolution frames (empty for vacuum scenario).
  std::vector<SchwingerFrame> frames{};
  // Mass sweep: list of {mass, ground_energy, chiral_condensate} triplets.
  std::vector<std::array<double, 3>> mass_sweep{};
  // String energy vs separation (for string scenario).
  std::vector<std::array<double, 2>> string_energies{};
  double elapsed_ms{0.0};
  std::string message;
};

// Summary of a completed VQE/QAOA optimization run.
struct VqeResult {
  bool ok{true};
  bool converged{false};
  std::uint32_t n_iterations{0};
  double optimal_energy{0.0};
  std::vector<double> optimal_params{};
  std::vector<VqeFrame> frames{};
  std::string message;
  double elapsed_ms{0.0};
};

// Hamiltonian simulation output.
struct SimulationResult {
  bool ok{true};
  std::string backend;
  double elapsed_ms{0.0};
  std::vector<std::complex<double>> state{};
  std::map<std::string, double> observables{};
  std::vector<ContinuousDensityFrame> continuous_density_frames{};
  std::vector<MpsFrame> mps_frames{};
  std::vector<LindbladFrame> lindblad_frames{};
  std::vector<WignerFrame> wigner_frames{};
  std::vector<EntanglementFrame> entanglement_frames{};
  ContinuousExportConfig continuous_export{};
  std::vector<Event> events{};
  std::string message;
};

// Scheduler explanation artifact for backend selection.
struct BackendInfo {
  std::string selected_backend;
  std::vector<std::string> candidates;
  std::vector<std::string> reasons;
};

// Static capability descriptor for each backend.
struct BackendCapabilities {
  BackendKind kind{BackendKind::Statevector};
  bool supports_gpu{false};
  bool supports_mpi{false};
  bool supports_noisy{false};
  std::uint32_t max_qubits_recommended{0};
  std::string equation_scope{"Schrodinger"};
};

// Static capability descriptor for the gate-model execution path.
struct CircuitCapabilities {
  // Native backend used by the circuit path today.
  std::string backend{"statevector"};
  // Gate vocabulary accepted by the current circuit executor.
  std::vector<std::string> supported_gates{};
  // Noise kinds accepted by the current circuit executor.
  std::vector<std::string> supported_noise_models{};
  // Practical hard limit enforced by the statevector implementation.
  std::uint32_t max_qubits{0};
  // Practical sampling limit exposed to callers for planning.
  std::uint32_t max_shots{0};
  // Framework-adapter hints.
  bool supports_qiskit_like{true};
  bool supports_cirq_like{true};
};

// ── M45: Non-Hermitian Topology ────────────────────────────────────────────────
//
// Two canonical non-Hermitian topological phenomena:
//
// 1. Hatano-Nelson skin effect (asymmetric hopping chain):
//    H = Σ_n J_R c†_{n+1}c_n + J_L c†_n c_{n+1},  J_R=J+γ/2, J_L=J-γ/2
//    PBC: E(k) = 2J cos(k) + iγ sin(k)  (complex ellipse, winding W=sign(γ))
//    OBC: E_n = 2√(J_R J_L) cos(nπ/(N+1))  — all real, skin-localized eigenstates
//
// 2. Exceptional point (2×2 non-Hermitian, off-diagonal gain-loss):
//    H(δ,γ) = [[δ, J+γ],[J-γ, -δ]]
//    E = ±√(δ² + J² - γ²);  EP at δ=0, γ=J (eigenvalues and eigenvectors coalesce)
//
// Scenarios: "skin" | "ep" | "both"

struct NonHermOptions {
  double J{1.0};         // base hopping amplitude
  double gamma{0.5};     // asymmetry / non-Hermitian gain-loss parameter
  double delta{0.0};     // detuning for EP scenario (on-site energy imbalance)
  int    n_sites{60};    // OBC chain length
  int    n_k{100};       // k-grid for PBC spectrum
  int    n_gamma{80};    // points for gamma-sweep
  double gamma_max{2.0}; // maximum gamma for sweeps
  std::string scenario{"both"};  // "skin"|"ep"|"both"
};

struct NonHermResult {
  bool ok{false};
  std::string scenario;
  double J{0}, gamma{0}, delta{0};
  int    W{0};               // skin winding number (PBC, reference energy = 0)
  std::string phase_label;   // e.g. "Skin (W=+1)" or "Trivial (W=0)"
  // Skin effect — PBC complex spectrum (one band, n_k points)
  std::vector<double> k_pbc;
  std::vector<double> e_re_pbc, e_im_pbc;
  // Skin effect — OBC real spectrum (n_sites eigenvalues)
  std::vector<double> e_re_obc;
  // Skin localization — |ψ_r|² profile of the most localized eigenstate
  std::vector<double> skin_site;     // site indices 1..N
  std::vector<double> skin_density;  // |ψ_r|² (normalized)
  // Winding-number sweep — W vs gamma (n_gamma points)
  std::vector<double> sweep_gamma;
  std::vector<int>    sweep_W;
  // Exceptional point — E vs gamma at delta=0 (n_gamma points, two branches ±)
  std::vector<double> ep_gamma;
  std::vector<double> ep_e_re_plus,  ep_e_im_plus;   // + branch
  std::vector<double> ep_e_re_minus, ep_e_im_minus;  // - branch
  double elapsed_ms{0.0};
  std::string message;
};

// ── M46: Aubry-André-Harper (AAH) Quasicrystal ────────────────────────────────
//
// 1D tight-binding chain with quasiperiodic on-site potential:
//   H = Σ_n t(c†_{n+1}c_n + h.c.) + 2V cos(2πβn + φ) c†_n c_n
//
// Self-dual at V = t (for irrational β): metal-insulator transition.
//   V < t: all states extended (metal,  IPR → 0 as N → ∞)
//   V = t: critical states (multifractal, IPR ~ N^{-D₂})
//   V > t: all states localised (insulator, IPR = O(1))
//
// Lyapunov exponent (localization length inverse) for V > t:
//   γ = ln(V/t)  (same for all energies — consequence of Aubry self-duality)
//
// β = golden ratio (√5−1)/2 ≈ 0.6180 (default): quasiperiodic (irrational).
//
// Scenarios: "spectrum" | "phase" | "both"
//   spectrum: eigenvalues + IPR at fixed V
//   phase:    eigenvalue spectrum as V sweeps 0 → V_max  (fractal butterfly)

struct AAHOptions {
  double t{1.0};        // hopping amplitude (energy scale)
  double V{0.5};        // quasiperiodic potential strength
  double beta{0.0};     // frequency: 0 = golden ratio (√5-1)/2
  double phi{0.0};      // phase offset (usually 0)
  int    n_sites{89};   // chain length (Fibonacci number works well for golden-ratio β)
  int    n_V{60};       // V values for phase-diagram sweep
  double V_max{2.0};    // maximum V for sweep
  std::string scenario{"both"};  // "spectrum"|"phase"|"both"
};

struct AAHResult {
  bool ok{false};
  std::string scenario;
  double t{0}, V{0}, beta_used{0};
  // Fixed-V spectrum and IPR
  std::vector<double> eigenvalues;  // n_sites sorted energies
  std::vector<double> ipr;         // inverse participation ratio per eigenstate
  double mean_ipr{0.0};            // average IPR (localization diagnostic)
  double lyapunov{0.0};            // analytic Lyapunov exponent = max(0, ln(V/t))
  std::string phase_label;         // "Metal (V<t)", "Critical (V=t)", "Insulator (V>t)"
  // Phase diagram: spectrum vs V  (n_V * n_sites values, row-major: outer=V, inner=E)
  std::vector<double> pd_V;        // n_V V values
  std::vector<double> pd_E;        // n_V * n_sites eigenvalues (flattened)
  std::vector<double> pd_ipr_mean; // mean IPR at each V (n_V values)
  double elapsed_ms{0.0};
  std::string message;
};

}  // namespace qx
