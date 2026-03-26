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

// Hamiltonian simulation output.
struct SimulationResult {
  bool ok{true};
  std::string backend;
  double elapsed_ms{0.0};
  std::vector<std::complex<double>> state{};
  std::map<std::string, double> observables{};
  std::vector<ContinuousDensityFrame> continuous_density_frames{};
  std::vector<LindbladFrame> lindblad_frames{};
  std::vector<WignerFrame> wigner_frames{};
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

}  // namespace qx
