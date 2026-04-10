#!/usr/bin/env python3
"""Generate animated HTML visualizations for QuantX simulations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from viz.common import (
    _user_cli_error, _make_engine, _animated_page,
    _resolve_continuous_payload, _attach_continuous_metadata,
    _require_list, _reshape_flat_to_rows, _reshape_flat_points,
    _normalize_engine_export_payload,
    _load_continuous_export_file, _extract_continuous_export_from_result,
    _continuous_problem_2d_request, _continuous_problem_2d_driven_request,
    _continuous_problem_3d_request,
)
from viz.continuous import (
    _spatial_wavepacket_frames, _spatial_wavepacket_page,
    _spatial_wavepacket_2d_frames, _spatial_wavepacket_2d_page,
    _spatial_wavepacket_2d_dirac_frames, _driven_wavepacket_2d_page,
    _spatial_wavepacket_3d_frames, _spatial_wavepacket_3d_page,
    _dirac_spinor_page,
)
from viz.open_systems import (
    _lindblad_frames, _lindblad_page,
    _mps_frames, _mps_page,
    _entanglement_frames, _entanglement_page,
    _wigner_frames, _wigner_page,
    _rabi_data, _mps_data,
)
from viz.algorithms import (
    _qec_frames, _qec_page,
    _qpe_frames, _qpe_page,
    _vqe_frames, _vqe_page,
)
from viz.qft import (
    _phi4_dynamics_frames, _phi4_frames, _phi4_page, _phi4_dynamics_page,
    _schwinger_frames, _schwinger_fallback_dynamics, _schwinger_page,
)
from viz.condensed import (
    _dashboard_page,
    _ssh_frames, _ssh_page,
    _kitaev_frames, _kitaev_page,
    _anderson_frames, _anderson_page,
    _chern_frames, _chern_page,
    _hofstadter_frames, _hofstadter_page,
    _mbl_frames, _mbl_page,
    _spectral_frames, _spectral_page,
    _tenfold_frames, _tenfold_page,
)
from viz.manybody import (
    _hubbard_frames, _hubbard_page,
    _floquet_frames, _floquet_page,
    _heisenberg_frames, _heisenberg_page,
    _bose_hubbard_frames, _bose_hubbard_page,
    _tfi_frames, _tfi_page,
)
from viz.topology import (
    _wilson_loop_frames, _wilson_loop_page,
    _second_chern_frames, _second_chern_page,
    _edge_states_frames, _edge_states_page,
    _weyl_frames, _weyl_page,
    _haldane_frames, _haldane_page,
    _fu_kane_frames, _fu_kane_page,
    _lutchyn_oreg_frames, _lutchyn_oreg_page,
    _kitaev_honeycomb_frames, _kitaev_honeycomb_page,
    _bbh_frames, _bbh_page,
    _floquet_topo_frames, _floquet_topo_page,
    _non_herm_frames, _non_herm_page,
    _aah_frames, _aah_page,
    _syk_frames, _syk_page,
    _kondo_frames, _kondo_page,
    _mipt_frames, _mipt_page,
)


CLI_EXAMPLES = """Examples:
  python scripts/visualize_simulation.py --scenario spatial2d --solver-mode python
  python scripts/visualize_simulation.py --scenario spatial2d --solver-mode engine --continuous-export-json out/export.json --strict-export-schema
  python scripts/visualize_simulation.py --scenario spatial3d --continuous-export-json out/export.json --spatial3d-export-json
"""

def main() -> int:
  """Parse CLI arguments, generate requested visualizations, and write them to disk."""
  parser = argparse.ArgumentParser(
      description="Generate QuantX visualization HTML.",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=CLI_EXAMPLES,
  )
  parser.add_argument("--scenario", choices=["rabi", "mps", "mps_ising", "mps_xx", "spatial", "spatial2d", "spatial3d", "dirac2d", "driven_barrier", "bloch", "gaussian_pulse", "lindblad", "wigner_free", "wigner_barrier", "entanglement_rabi", "entanglement_decay", "vqe_ising", "vqe_qaoa", "qpe_phase", "qpe_hamiltonian", "qec_bit_flip", "qec_phase_flip", "qec_shor", "schwinger_vacuum", "schwinger_string", "schwinger_dynamics", "phi4_symmetric", "phi4_broken", "phi4_sweep", "phi4_dynamics", "hubbard", "floquet", "heisenberg", "bose_hubbard", "tfi", "ssh", "kitaev", "anderson", "chern", "hofstadter", "mbl", "spectral", "tenfold", "second_chern", "wilson_loop", "edge_states", "haldane", "weyl", "fu_kane", "lutchyn_oreg", "kitaev_honeycomb", "bbh", "floquet_topo", "non_herm", "aah", "syk", "kondo", "mipt", "all"], default="all")
  parser.add_argument(
      "--solver-mode",
      choices=["auto", "engine", "python"],
      default="auto",
      help="How continuous spatial pages should source their frame data.",
  )
  parser.add_argument(
      "--continuous-export-json",
      default="",
      help="Optional engine-export JSON file to render directly instead of solving again.",
  )
  parser.add_argument(
      "--strict-export-schema",
      action="store_true",
      help="Fail immediately when the continuous export schema_version is missing, below 1, or newer than supported.",
  )
  parser.add_argument("--spatial-potential", choices=["barrier", "free", "harmonic", "well"], default="barrier")
  parser.add_argument("--spatial2d-potential", choices=["barrier", "free", "harmonic", "well"], default="barrier")
  parser.add_argument("--spatial3d-potential", choices=["barrier", "free", "harmonic", "well"], default="barrier")
  parser.add_argument("--spatial3d-boundary", choices=["periodic", "absorbing", "dirichlet"], default="periodic")
  parser.add_argument("--spatial3d-export-json", action="store_true")
  parser.add_argument("--spatial3d-json-path", default="")
  parser.add_argument("--dirac2d-potential", choices=["barrier", "free", "harmonic", "well"], default="barrier")
  parser.add_argument("--dirac2d-mass", type=float, default=1.0)
  parser.add_argument("--lindblad-scenario", choices=["amplitude_damping", "dephasing", "depolarizing", "rabi_decay"], default="amplitude_damping")
  parser.add_argument("--lindblad-qubits", type=int, default=1)
  parser.add_argument("--samples-lindblad-t", type=int, default=80)
  parser.add_argument("--samples-wigner-t", type=int, default=50)
  parser.add_argument("--samples-entanglement-t", type=int, default=80)
  parser.add_argument("--samples-mps-t", type=int, default=60)
  parser.add_argument("--mps-sites", type=int, default=12)
  parser.add_argument("--mps-max-bond", type=int, default=16)
  parser.add_argument("--vqe-max-iter", type=int, default=120, help="max VQE/QAOA optimizer iterations")
  parser.add_argument("--vqe-layers", type=int, default=2, help="number of HEA layers / QAOA rounds")
  parser.add_argument("--vqe-qubits", type=int, default=4, help="number of qubits for VQE scenarios")
  parser.add_argument("--qpe-ancilla", type=int, default=6, help="number of ancilla (precision) qubits for QPE")
  parser.add_argument("--qpe-shots", type=int, default=4096, help="number of measurement shots for QPE")
  parser.add_argument("--qpe-phase", type=float, default=0.375, help="true phase φ ∈ [0,1) for qpe_phase scenario (phase_angle = 2π·φ)")
  parser.add_argument("--qpe-sweep-max", type=int, default=10, help="maximum n_ancilla in precision sweep (1..N)")
  parser.add_argument("--qec-error-rate", type=float, default=0.05, help="physical error rate per qubit for QEC scenarios")
  parser.add_argument("--qec-trials", type=int, default=2000, help="number of Monte Carlo trials for QEC scenarios")
  parser.add_argument("--qec-sweep", action="store_true", default=True, help="include threshold curve sweep in QEC output")
  parser.add_argument("--qec-no-sweep", dest="qec_sweep", action="store_false", help="disable QEC threshold sweep")
  parser.add_argument("--schwinger-sites", type=int, default=8, help="number of lattice sites for Schwinger model (even, ≤14)")
  parser.add_argument("--schwinger-mass", type=float, default=0.5, help="bare fermion mass m in units of J")
  parser.add_argument("--schwinger-coupling", type=float, default=0.3, help="gauge coupling g² in units of J")
  parser.add_argument("--schwinger-background", type=float, default=0.5, help="background electric field α for dynamics quench")
  parser.add_argument("--schwinger-steps", type=int, default=60, help="number of time steps for dynamics scenario")
  parser.add_argument("--schwinger-dt", type=float, default=0.08, help="time step dt for Schwinger dynamics")
  parser.add_argument("--schwinger-mass-sweep", type=int, default=30, help="number of mass values in chiral condensate sweep")
  parser.add_argument("--phi4-sites", type=int, default=4, help="number of lattice sites for phi4 (2-5)")
  parser.add_argument("--phi4-m2", type=float, default=1.0, help="bare mass squared m² for phi4 (negative → broken phase)")
  parser.add_argument("--phi4-lambda", type=float, default=1.0, help="quartic coupling λ in λ/4·φ⁴")
  parser.add_argument("--phi4-kappa", type=float, default=1.0, help="nearest-neighbour hopping κ")
  parser.add_argument("--phi4-d-local", type=int, default=10, help="local Fock space truncation (oscillator levels)")
  parser.add_argument("--phi4-sweep-points", type=int, default=25, help="number of m² points in phi4 phase transition sweep")
  parser.add_argument("--phi4-dynamics-m2-initial", type=float, default=1.0, help="pre-quench mass squared for phi4_dynamics")
  parser.add_argument("--phi4-dynamics-m2-final",   type=float, default=-1.5, help="post-quench mass squared for phi4_dynamics")
  parser.add_argument("--phi4-dynamics-steps",      type=int,   default=80,  help="number of time steps for phi4_dynamics")
  parser.add_argument("--phi4-dynamics-dt",         type=float, default=0.05, help="time step dt for phi4_dynamics")
  parser.add_argument("--hubbard-sites",   type=int,   default=6,    help="number of lattice sites for Hubbard (2-12)")
  parser.add_argument("--hubbard-t",       type=float, default=1.0,  help="hopping amplitude t")
  parser.add_argument("--hubbard-U",       type=float, default=4.0,  help="on-site Coulomb repulsion U")
  parser.add_argument("--hubbard-n-up",    type=int,   default=0,    help="spin-up electrons (0 = half-filling)")
  parser.add_argument("--hubbard-n-dn",    type=int,   default=0,    help="spin-down electrons (0 = half-filling)")
  parser.add_argument("--hubbard-sweep-points", type=int, default=20, help="number of U/t points in sweep")
  parser.add_argument("--hubbard-U-max",   type=float, default=12.0, help="maximum U for Hubbard sweep")
  parser.add_argument("--floquet-sites",   type=int,   default=8,    help="number of qubits for Floquet kicked-Ising")
  parser.add_argument("--floquet-J",       type=float, default=1.0,  help="Ising ZZ coupling J")
  parser.add_argument("--floquet-hz",      type=float, default=0.809, help="longitudinal field h_z")
  parser.add_argument("--floquet-amp",     type=float, default=1.5,  help="kick amplitude h_x")
  parser.add_argument("--floquet-omega",   type=float, default=4.5,  help="drive frequency ω")
  parser.add_argument("--floquet-periods", type=int,   default=40,   help="number of Floquet periods to simulate")
  parser.add_argument("--floquet-scenario", choices=["dynamics", "spectrum"], default="dynamics")
  parser.add_argument("--heisenberg-sites",   type=int,   default=8,    help="number of sites for Heisenberg chain (2-12)")
  parser.add_argument("--heisenberg-J",       type=float, default=1.0,  help="transverse exchange coupling J")
  parser.add_argument("--heisenberg-Jz",      type=float, default=1.0,  help="longitudinal exchange Jz")
  parser.add_argument("--heisenberg-h",       type=float, default=0.0,  help="uniform longitudinal field h")
  parser.add_argument("--heisenberg-sweep-points", type=int, default=20, help="number of Jz points in sweep")
  parser.add_argument("--heisenberg-Jz-min",  type=float, default=-2.0, help="minimum Jz for sweep")
  parser.add_argument("--heisenberg-Jz-max",  type=float, default=3.0,  help="maximum Jz for sweep")
  parser.add_argument("--bose-hubbard-sites",   type=int,   default=4,   help="number of sites for Bose-Hubbard (2-8)")
  parser.add_argument("--bose-hubbard-bosons",  type=int,   default=4,   help="total boson number")
  parser.add_argument("--bose-hubbard-nmax",    type=int,   default=4,   help="max occupation per site")
  parser.add_argument("--bose-hubbard-J",       type=float, default=0.1, help="hopping amplitude J")
  parser.add_argument("--bose-hubbard-U",       type=float, default=1.0, help="on-site interaction U")
  parser.add_argument("--bose-hubbard-mu",      type=float, default=0.0, help="chemical potential mu")
  parser.add_argument("--bose-hubbard-sweep-points", type=int, default=20, help="number of J points in sweep")
  parser.add_argument("--bose-hubbard-J-max",   type=float, default=0.5, help="maximum J for Bose-Hubbard sweep")
  parser.add_argument("--tfi-sites",        type=int,   default=10,   help="number of sites for TFI (2-18)")
  parser.add_argument("--tfi-J",            type=float, default=1.0,  help="ZZ coupling J for TFI")
  parser.add_argument("--tfi-h",            type=float, default=0.5,  help="transverse field h (QPT at h=J)")
  parser.add_argument("--tfi-sweep-points", type=int,   default=30,   help="number of h values in TFI sweep")
  parser.add_argument("--tfi-h-min",        type=float, default=0.0,  help="minimum h for TFI sweep")
  parser.add_argument("--tfi-h-max",        type=float, default=2.0,  help="maximum h for TFI sweep")
  parser.add_argument("--ssh-cells",        type=int,   default=10,   help="number of unit cells for SSH (2-200)")
  parser.add_argument("--ssh-t",            type=float, default=1.0,  help="hopping scale t for SSH")
  parser.add_argument("--ssh-delta",        type=float, default=0.5,  help="dimerization δ (trivial if δ<0, topo if δ>0)")
  parser.add_argument("--ssh-sweep-points", type=int,   default=30,   help="number of δ values in SSH sweep")
  parser.add_argument("--ssh-delta-min",    type=float, default=-0.9, help="minimum δ for SSH sweep")
  parser.add_argument("--ssh-delta-max",    type=float, default=0.9,  help="maximum δ for SSH sweep")
  parser.add_argument("--kitaev-sites",    type=int,   default=20,   help="number of sites for Kitaev chain (2-200)")
  parser.add_argument("--kitaev-t",        type=float, default=1.0,  help="hopping t for Kitaev chain")
  parser.add_argument("--kitaev-delta",    type=float, default=1.0,  help="pairing amplitude Δ")
  parser.add_argument("--kitaev-mu",       type=float, default=0.0,  help="chemical potential μ (topo: |μ|<2t)")
  parser.add_argument("--kitaev-sweep-points", type=int, default=40, help="number of μ values in Kitaev sweep")
  parser.add_argument("--kitaev-mu-min",   type=float, default=-4.0, help="minimum μ for Kitaev sweep")
  parser.add_argument("--kitaev-mu-max",   type=float, default=4.0,  help="maximum μ for Kitaev sweep")
  parser.add_argument("--anderson-sites",   type=int,   default=100, help="number of sites for Anderson chain (4-2000)")
  parser.add_argument("--anderson-t",       type=float, default=1.0, help="hopping t for Anderson chain")
  parser.add_argument("--anderson-W",       type=float, default=2.0, help="disorder strength W")
  parser.add_argument("--anderson-samples", type=int,   default=50,  help="disorder realisations to average")
  parser.add_argument("--anderson-sweep-points", type=int, default=20, help="number of W values in Anderson sweep")
  parser.add_argument("--anderson-W-min",   type=float, default=0.1, help="minimum W for Anderson sweep")
  parser.add_argument("--anderson-W-max",   type=float, default=8.0, help="maximum W for Anderson sweep")
  # M31-K: 2D Chern insulator (QWZ model)
  parser.add_argument("--chern-sites",    type=int,   default=20,   help="strip width Ny for edge states")
  parser.add_argument("--chern-m",        type=float, default=-1.0, help="mass parameter (QPT at m=0,±2)")
  parser.add_argument("--chern-t",        type=float, default=1.0,  help="hopping t")
  parser.add_argument("--chern-nk",       type=int,   default=40,   help="BZ k-grid points (n_k × n_k)")
  parser.add_argument("--chern-sweep-points", type=int, default=30, help="number of m values in sweep")
  parser.add_argument("--chern-m-min",    type=float, default=-5.0, help="minimum m for Chern sweep")
  parser.add_argument("--chern-m-max",    type=float, default=1.0,  help="maximum m for Chern sweep")
  # M31-L: Hofstadter butterfly
  parser.add_argument("--hofstadter-t",   type=float, default=1.0,  help="hopping amplitude")
  parser.add_argument("--hofstadter-q-max", type=int, default=24,   help="max denominator for butterfly")
  parser.add_argument("--hofstadter-phi", type=float, default=0.25, help="flux for fixed-phi bands scenario")
  parser.add_argument("--hofstadter-nk",  type=int,   default=40,   help="k-points for bands scenario")
  parser.add_argument("--hofstadter-nky", type=int,   default=12,   help="ky samples per flux value")
  parser.add_argument("--mbl-sites",    type=int,   default=10,   help="number of sites for MBL chain (even, 2-16)")
  parser.add_argument("--mbl-J",        type=float, default=1.0,  help="Heisenberg coupling J")
  parser.add_argument("--mbl-delta",    type=float, default=1.0,  help="XXZ anisotropy delta")
  parser.add_argument("--mbl-W",        type=float, default=3.0,  help="disorder strength W (single)")
  parser.add_argument("--mbl-scenario", choices=["single", "sweep"], default="single", help="single | sweep")
  parser.add_argument("--mbl-W-min",    type=float, default=0.5,  help="minimum W for MBL sweep")
  parser.add_argument("--mbl-W-max",    type=float, default=8.0,  help="maximum W for MBL sweep")
  parser.add_argument("--mbl-sweep-points", type=int, default=16, help="number of W values in MBL sweep")
  parser.add_argument("--mbl-samples",  type=int,   default=10,   help="disorder realizations per W (sweep)")
  parser.add_argument("--mbl-seed",     type=int,   default=42,   help="random seed for MBL")
  parser.add_argument("--spectral-sites",   type=int,   default=12,   help="number of sites for spectral XXZ chain (even, 2-16)")
  parser.add_argument("--spectral-J",       type=float, default=1.0,  help="Heisenberg coupling J for spectral")
  parser.add_argument("--spectral-delta",   type=float, default=1.0,  help="XXZ anisotropy delta for spectral")
  parser.add_argument("--spectral-h",       type=float, default=0.0,  help="external field h for spectral")
  parser.add_argument("--spectral-scenario", choices=["spectral", "imagtime", "both"], default="both", help="spectral | imagtime | both")
  parser.add_argument("--spectral-nk",      type=int,   default=20,   help="number of k points in S(k,omega)")
  parser.add_argument("--spectral-nomega",  type=int,   default=200,  help="number of omega points in S(k,omega)")
  parser.add_argument("--spectral-omega-min", type=float, default=-4.0, help="minimum omega")
  parser.add_argument("--spectral-omega-max", type=float, default=4.0,  help="maximum omega")
  parser.add_argument("--spectral-eta",     type=float, default=0.15, help="Lorentzian broadening eta")
  parser.add_argument("--spectral-tau-max", type=float, default=4.0,  help="maximum imaginary time tau")
  parser.add_argument("--spectral-ntau",    type=int,   default=60,   help="number of tau points")
  parser.add_argument("--tenfold-class",      default="BDI",  help="AZ symmetry class (A|AI|AII|AIII|BDI|CII|D|DIII|C|CI)")
  parser.add_argument("--tenfold-dim",        type=int,   default=1,    help="spatial dimension 1|2|3")
  parser.add_argument("--tenfold-model",      default="ssh",  help="model: ssh | qwz | kitaev")
  parser.add_argument("--tenfold-scenario",   choices=["classify", "invariant", "phase_diagram"], default="phase_diagram", help="classify | invariant | phase_diagram")
  parser.add_argument("--tenfold-t1",         type=float, default=1.0,  help="SSH intra-cell hopping t1")
  parser.add_argument("--tenfold-t2",         type=float, default=0.5,  help="SSH inter-cell hopping t2")
  parser.add_argument("--tenfold-m",          type=float, default=1.0,  help="QWZ mass parameter m")
  parser.add_argument("--tenfold-t",          type=float, default=1.0,  help="QWZ hopping t")
  parser.add_argument("--tenfold-t-hop",      type=float, default=1.0,  help="Kitaev hopping t_hop")
  parser.add_argument("--tenfold-mu",         type=float, default=0.0,  help="Kitaev chemical potential mu")
  parser.add_argument("--tenfold-nk",         type=int,   default=100,  help="k-grid points for invariant")
  parser.add_argument("--tenfold-sweep-param", default="t2", help="parameter to sweep: t1|t2|m|mu")
  parser.add_argument("--tenfold-param-min",  type=float, default=-2.0, help="sweep parameter minimum")
  parser.add_argument("--tenfold-param-max",  type=float, default=2.0,  help="sweep parameter maximum")
  parser.add_argument("--tenfold-n-sweep",    type=int,   default=80,   help="number of sweep points")
  parser.add_argument("--wilson-loop-M",           type=float, default=1.0,  help="BHZ mass parameter M (band-inversion strength)")
  parser.add_argument("--wilson-loop-A",           type=float, default=1.0,  help="BHZ velocity parameter A")
  parser.add_argument("--wilson-loop-B",           type=float, default=1.0,  help="BHZ curvature parameter B")
  parser.add_argument("--wilson-loop-C",           type=float, default=0.0,  help="BHZ particle-hole asymmetry C")
  parser.add_argument("--wilson-loop-D",           type=float, default=0.0,  help="BHZ particle-hole asymmetry D")
  parser.add_argument("--wilson-loop-nkx",         type=int,   default=40,   help="k-grid points along kx for Wilson loop")
  parser.add_argument("--wilson-loop-nky",         type=int,   default=40,   help="k-grid points along ky for WCC spectrum")
  parser.add_argument("--wilson-loop-sweep-param", default="M",              help="parameter to sweep in phase_diagram (only M supported)")
  parser.add_argument("--wilson-loop-param-min",   type=float, default=-1.0, help="sweep parameter minimum")
  parser.add_argument("--wilson-loop-param-max",   type=float, default=5.0,  help="sweep parameter maximum")
  parser.add_argument("--wilson-loop-n-sweep",     type=int,   default=40,   help="number of sweep points")
  parser.add_argument("--second-chern-m",         type=float, default=1.0,  help="Zhang-Hu mass parameter m")
  parser.add_argument("--second-chern-t",         type=float, default=1.0,  help="Zhang-Hu hopping t")
  parser.add_argument("--second-chern-nk",        type=int,   default=6,    help="k-grid points per dimension (nk^4 hypercubes)")
  parser.add_argument("--second-chern-sweep-param", default="m",            help="parameter to sweep: m | t")
  parser.add_argument("--second-chern-param-min", type=float, default=-5.0, help="sweep parameter minimum")
  parser.add_argument("--second-chern-param-max", type=float, default=5.0,  help="sweep parameter maximum")
  parser.add_argument("--second-chern-n-sweep",   type=int,   default=40,   help="number of sweep points")
  parser.add_argument("--edge-states-M",           type=float, default=1.0,  help="BHZ mass parameter M (band-inversion strength)")
  parser.add_argument("--edge-states-A",           type=float, default=1.0,  help="BHZ velocity parameter A")
  parser.add_argument("--edge-states-B",           type=float, default=1.0,  help="BHZ curvature parameter B")
  parser.add_argument("--edge-states-C",           type=float, default=0.0,  help="BHZ particle-hole asymmetry C")
  parser.add_argument("--edge-states-D",           type=float, default=0.0,  help="BHZ particle-hole asymmetry D")
  parser.add_argument("--edge-states-nkx",         type=int,   default=60,   help="k-grid points along kx for strip spectrum")
  parser.add_argument("--edge-states-ny",          type=int,   default=20,   help="strip width (sites along y, open BCs)")
  parser.add_argument("--edge-states-sweep-param", default="M",              help="parameter to sweep in phase_diagram (only M supported)")
  parser.add_argument("--edge-states-param-min",   type=float, default=-1.0, help="sweep parameter minimum")
  parser.add_argument("--edge-states-param-max",   type=float, default=5.0,  help="sweep parameter maximum")
  parser.add_argument("--edge-states-n-sweep",     type=int,   default=40,   help="number of sweep points")
  # M38: Haldane Model (QAH)
  parser.add_argument("--haldane-t1",          type=float, default=1.0,       help="Haldane NN hopping t1")
  parser.add_argument("--haldane-t2",          type=float, default=0.1,       help="Haldane NNN hopping magnitude t2")
  parser.add_argument("--haldane-phi",         type=float, default=1.5707963, help="Haldane NNN flux phase phi (radians)")
  parser.add_argument("--haldane-M",           type=float, default=0.0,       help="Haldane staggered on-site potential M")
  parser.add_argument("--haldane-nk",          type=int,   default=40,        help="BZ k-grid resolution n_k (n_k^2 points)")
  parser.add_argument("--haldane-sweep-param", default="M",                   help="parameter to sweep: M | phi")
  parser.add_argument("--haldane-param-min",   type=float, default=-6.0,      help="sweep parameter minimum")
  parser.add_argument("--haldane-param-max",   type=float, default=6.0,       help="sweep parameter maximum")
  parser.add_argument("--haldane-n-sweep",     type=int,   default=60,        help="number of sweep points")
  # M39: Weyl Semimetal & Surface Fermi Arcs
  parser.add_argument("--weyl-m",             type=float, default=2.0,  help="Weyl mass parameter m (nodes at kz=+-arccos(m-2) for 1<m<3)")
  parser.add_argument("--weyl-nk",            type=int,   default=20,   help="kx-ky grid resolution for Chern/arc computations")
  parser.add_argument("--weyl-nz",            type=int,   default=20,   help="slab thickness (unit cells in z, open BCs)")
  parser.add_argument("--weyl-nkz",           type=int,   default=40,   help="number of kz values for Chern sweep")
  parser.add_argument("--weyl-fermi-energy",  type=float, default=0.0,  help="Fermi level for arc detection")
  parser.add_argument("--weyl-param-min",     type=float, default=1.1,  help="minimum m for node sweep")
  parser.add_argument("--weyl-param-max",     type=float, default=2.9,  help="maximum m for node sweep")
  parser.add_argument("--weyl-n-sweep",       type=int,   default=30,   help="number of m sweep points for node diagram")
  # M40: Fu-Kane superconductor
  parser.add_argument("--fu-kane-vF",     type=float, default=1.0,  help="Fu-Kane Fermi velocity v_F")
  parser.add_argument("--fu-kane-M",      type=float, default=0.5,  help="Fu-Kane Zeeman exchange field M")
  parser.add_argument("--fu-kane-mu",     type=float, default=0.0,  help="Fu-Kane chemical potential mu")
  parser.add_argument("--fu-kane-Delta",  type=float, default=0.3,  help="Fu-Kane s-wave SC gap Delta")
  parser.add_argument("--fu-kane-nk",     type=int,   default=30,   help="k-grid for Chern number")
  parser.add_argument("--fu-kane-ny",     type=int,   default=24,   help="strip width (sites in y)")
  parser.add_argument("--fu-kane-nkx",    type=int,   default=80,   help="kx points for band / strip spectrum")
  # M41: Lutchyn-Oreg Majorana nanowire
  parser.add_argument("--lo-t",           type=float, default=1.0,   help="Lutchyn-Oreg hopping t")
  parser.add_argument("--lo-alpha",       type=float, default=0.5,   help="Lutchyn-Oreg Rashba coupling alpha")
  parser.add_argument("--lo-mu",          type=float, default=0.0,   help="Lutchyn-Oreg chemical potential mu")
  parser.add_argument("--lo-Delta",       type=float, default=0.25,  help="Lutchyn-Oreg SC gap Delta")
  parser.add_argument("--lo-Vz",          type=float, default=0.0,   help="Lutchyn-Oreg Zeeman field V_z")
  parser.add_argument("--lo-nk",          type=int,   default=200,   help="k-grid points for bulk BdG")
  parser.add_argument("--lo-nsites",      type=int,   default=60,    help="wire length (sites, open BCs)")
  # ── M42: Kitaev honeycomb ──────────────────────────────────────────────
  parser.add_argument("--kh-Jx",    type=float, default=1.0,  help="Kitaev honeycomb Jx coupling")
  parser.add_argument("--kh-Jy",    type=float, default=1.0,  help="Kitaev honeycomb Jy coupling")
  parser.add_argument("--kh-Jz",    type=float, default=1.0,  help="Kitaev honeycomb Jz coupling")
  parser.add_argument("--kh-kappa", type=float, default=0.3,  help="Kitaev NNN coupling κ (breaks TRS, opens gap)")
  parser.add_argument("--kh-nk",    type=int,   default=60,   help="BZ grid resolution for honeycomb")
  parser.add_argument("--kh-npd",   type=int,   default=80,   help="phase diagram simplex sample density")
  # ── M43: BBH quadrupole insulator ─────────────────────────────────────
  parser.add_argument("--bbh-gamma-x",  type=float, default=0.5, help="BBH intra-cell hopping gamma_x")
  parser.add_argument("--bbh-lambda-x", type=float, default=1.0, help="BBH inter-cell hopping lambda_x")
  parser.add_argument("--bbh-gamma-y",  type=float, default=0.5, help="BBH intra-cell hopping gamma_y")
  parser.add_argument("--bbh-lambda-y", type=float, default=1.0, help="BBH inter-cell hopping lambda_y")
  parser.add_argument("--bbh-nk",       type=int,   default=40,  help="BZ grid resolution for BBH bulk")
  parser.add_argument("--bbh-nsites",   type=int,   default=6,   help="linear size of finite lattice (n x n unit cells)")
  parser.add_argument("--bbh-npd",      type=int,   default=60,  help="phase diagram grid density")
  # ── M44: Floquet topological insulator ───────────────────────────────────
  parser.add_argument("--ft-J",     type=float, default=1.0,  help="Floquet hopping amplitude J")
  parser.add_argument("--ft-T",     type=float, default=1.0,  help="Floquet drive period T")
  parser.add_argument("--ft-nk",    type=int,   default=50,   help="k-grid resolution for Floquet bulk")
  parser.add_argument("--ft-ny",    type=int,   default=30,   help="strip width n_y for Floquet edge states")
  parser.add_argument("--ft-nJ",    type=int,   default=50,   help="winding sweep J-grid points")
  parser.add_argument("--ft-Jmax",  type=float, default=2.5,  help="max J for winding sweep")
  # ── M45: Non-Hermitian topology ──────────────────────────────────────────
  parser.add_argument("--nh-J",       type=float, default=1.0,  help="Non-Herm base hopping J")
  parser.add_argument("--nh-gamma",   type=float, default=0.5,  help="Non-Herm asymmetry gamma")
  parser.add_argument("--nh-nsites",  type=int,   default=60,   help="OBC chain length")
  parser.add_argument("--nh-nk",      type=int,   default=100,  help="k-grid for PBC spectrum")
  parser.add_argument("--nh-ngamma",  type=int,   default=80,   help="gamma sweep points")
  parser.add_argument("--nh-gammax",  type=float, default=2.0,  help="max gamma for sweeps")
  # ── M46: Aubry-André-Harper quasicrystal ─────────────────────────────────
  parser.add_argument("--aah-t",       type=float, default=1.0,  help="AAH hopping amplitude t")
  parser.add_argument("--aah-V",       type=float, default=0.5,  help="AAH quasiperiodic potential V")
  parser.add_argument("--aah-nsites",  type=int,   default=89,   help="AAH chain length")
  parser.add_argument("--aah-nV",      type=int,   default=60,   help="V sweep points for phase diagram")
  parser.add_argument("--aah-Vmax",    type=float, default=2.0,  help="max V for phase diagram")
  # ── M47: Sachdev-Ye-Kitaev model ─────────────────────────────────────────
  parser.add_argument("--syk-N",        type=int,   default=12,   help="SYK number of Majorana fermions (even, <=16)")
  parser.add_argument("--syk-J",        type=float, default=1.0,  help="SYK coupling scale J")
  parser.add_argument("--syk-beta",     type=float, default=1.0,  help="SYK inverse temperature beta")
  parser.add_argument("--syk-samples",  type=int,   default=20,   help="SYK disorder samples for SFF")
  parser.add_argument("--syk-nt",       type=int,   default=100,  help="SYK number of time points for SFF")
  parser.add_argument("--syk-tmax",     type=float, default=30.0, help="SYK max time for SFF")
  # ── M48: Kondo / SIAM ─────────────────────────────────────────────────────
  parser.add_argument("--kondo-Nbath",  type=int,   default=5,    help="Kondo bath levels (odd)")
  parser.add_argument("--kondo-epsd",   type=float, default=-1.5, help="Kondo impurity level eps_d")
  parser.add_argument("--kondo-U",      type=float, default=3.0,  help="Kondo Coulomb U")
  parser.add_argument("--kondo-V",      type=float, default=0.3,  help="Kondo hybridisation V")
  parser.add_argument("--kondo-bw",     type=float, default=4.0,  help="Kondo bath bandwidth")
  parser.add_argument("--kondo-nV",     type=int,   default=30,   help="Kondo V sweep points")
  parser.add_argument("--kondo-Vmax",   type=float, default=0.8,  help="Kondo max V")
  parser.add_argument("--kondo-neps",   type=int,   default=40,   help="Kondo eps_d sweep points")
  # ── M51: MIPT ───────────────────────────────────────────────────────────────
  parser.add_argument("--mipt-L",        type=int,   default=8,    help="MIPT chain length (even, <=12)")
  parser.add_argument("--mipt-p",        type=float, default=0.16, help="MIPT measurement rate for scaling")
  parser.add_argument("--mipt-np",       type=int,   default=20,   help="MIPT p sweep points")
  parser.add_argument("--mipt-pmax",     type=float, default=0.5,  help="MIPT max p for sweep")
  parser.add_argument("--mipt-nsamples", type=int,   default=10,   help="MIPT disorder realizations")
  parser.add_argument("--mipt-Lmax",     type=int,   default=10,   help="MIPT max L for scaling")
  parser.add_argument("--mipt-seed",     type=int,   default=42,   help="MIPT RNG seed")
  parser.add_argument("--out-dir", default="out/visualizations")
  parser.add_argument(
      "--dashboard", action="store_true", default=True,
      help="(--scenario all) Generate quantum_dashboard.html aggregating all scenarios.",
  )
  parser.add_argument(
      "--no-dashboard", dest="dashboard", action="store_false",
      help="Skip dashboard generation even when --scenario all is used.",
  )
  parser.add_argument("--samples-rabi", type=int, default=80)
  parser.add_argument("--samples-mps", type=int, default=60)
  parser.add_argument("--samples-spatial-t", type=int, default=120)
  parser.add_argument("--samples-spatial-x", type=int, default=220)
  parser.add_argument("--samples-spatial2d-t", type=int, default=90)
  parser.add_argument("--samples-spatial2d-x", type=int, default=72)
  parser.add_argument("--samples-spatial2d-y", type=int, default=72)
  parser.add_argument("--samples-spatial3d-t", type=int, default=48)
  parser.add_argument("--samples-spatial3d-x", type=int, default=32)
  parser.add_argument("--samples-spatial3d-y", type=int, default=32)
  parser.add_argument("--samples-spatial3d-z", type=int, default=32)
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[1]
  out_dir = root / args.out_dir
  out_dir.mkdir(parents=True, exist_ok=True)
  engine = _make_engine(root) if args.scenario in {"rabi", "mps", "spatial", "spatial2d", "spatial3d", "all"} or args.solver_mode not in {"python", None} else None  # lindblad is python-only
  generated = []
  export_json_path = root / args.continuous_export_json if args.continuous_export_json else None

  if args.scenario in {"rabi", "all"}:
    if engine is None:
      raise RuntimeError("rabi visualization requires the native engine library")
    sim, ref = _rabi_data(engine, args.samples_rabi)
    charts = [
        {
            "title": "Rabi Oscillation: Population(|1>) vs Time",
            "x_label": "time",
            "y_label": "population_1",
            "series": [
                {"name": "simulation", "color": "#0b84f3", "points": sim},
                {"name": "analytic sin^2(t)", "color": "#f39c12", "points": ref},
            ],
        }
    ]
    path = out_dir / "rabi_population.html"
    path.write_text(_animated_page("QuantX Animation - Rabi", charts), encoding="utf-8")
    generated.append(path)

  if args.scenario in {"mps", "all"}:
    if engine is None:
      raise RuntimeError("mps visualization requires the native engine library")
    bond, trunc = _mps_data(engine, args.samples_mps)
    charts = [
        {
            "title": "MPS Bond Dimension vs Time",
            "x_label": "time",
            "y_label": "bond_dim",
            "series": [{"name": "bond_dim", "color": "#2e8b57", "points": bond}],
        },
        {
            "title": "MPS Truncation Error vs Time",
            "x_label": "time",
            "y_label": "truncation_error",
            "series": [{"name": "truncation_error", "color": "#c0392b", "points": trunc}],
        },
    ]
    path = out_dir / "mps_diagnostics.html"
    path.write_text(_animated_page("QuantX Animation - MPS", charts), encoding="utf-8")
    generated.append(path)

  if args.scenario in {"spatial", "all"}:
    spatial_data = _spatial_wavepacket_frames(
        args.samples_spatial_t, args.samples_spatial_x, args.spatial_potential, source="python-reference"
    )
    path = out_dir / f"quantum_spatial_wavepacket_{args.spatial_potential}.html"
    path.write_text(_spatial_wavepacket_page(spatial_data), encoding="utf-8")
    generated.append(path)

  if args.scenario in {"spatial2d", "all"}:
    spatial2d_problem = _continuous_problem_2d_request(
        args.samples_spatial2d_t, args.samples_spatial2d_x, args.samples_spatial2d_y, args.spatial2d_potential
    )
    spatial2d = _resolve_continuous_payload(
        engine,
        spatial2d_problem,
        args.solver_mode,
        lambda: _spatial_wavepacket_2d_frames(
            args.samples_spatial2d_t, args.samples_spatial2d_x, args.samples_spatial2d_y, args.spatial2d_potential
        ),
        export_json_path,
        args.strict_export_schema,
    )
    path = out_dir / f"quantum_spatial2d_wavepacket_{args.spatial2d_potential}.html"
    path.write_text(_spatial_wavepacket_2d_page(spatial2d), encoding="utf-8")
    generated.append(path)

  if args.scenario in {"spatial3d", "all"}:
    spatial3d_problem = _continuous_problem_3d_request(
        args.samples_spatial3d_t,
        args.samples_spatial3d_x,
        args.samples_spatial3d_y,
        args.samples_spatial3d_z,
        args.spatial3d_potential,
        args.spatial3d_boundary,
    )
    spatial3d = _resolve_continuous_payload(
        engine,
        spatial3d_problem,
        args.solver_mode,
        lambda: _spatial_wavepacket_3d_frames(
            args.samples_spatial3d_t,
            args.samples_spatial3d_x,
            args.samples_spatial3d_y,
            args.samples_spatial3d_z,
            args.spatial3d_potential,
            args.spatial3d_boundary,
        ),
        export_json_path,
        args.strict_export_schema,
    )
    path = out_dir / f"quantum_spatial3d_wavepacket_{args.spatial3d_potential}_{args.spatial3d_boundary}.html"
    path.write_text(_spatial_wavepacket_3d_page(spatial3d), encoding="utf-8")
    generated.append(path)
    if args.spatial3d_export_json:
      if args.spatial3d_json_path:
        json_path = root / args.spatial3d_json_path
      else:
        json_path = out_dir / f"quantum_spatial3d_wavepacket_{args.spatial3d_potential}_{args.spatial3d_boundary}.json"
      json_path.parent.mkdir(parents=True, exist_ok=True)
      json_path.write_text(json.dumps(spatial3d, ensure_ascii=False), encoding="utf-8")
      generated.append(json_path)

  if args.scenario in {"dirac2d", "all"}:
    dirac2d_data = _spatial_wavepacket_2d_dirac_frames(
        args.samples_spatial2d_t,
        args.samples_spatial2d_x,
        args.samples_spatial2d_y,
        args.dirac2d_potential,
        args.dirac2d_mass,
    )
    path = out_dir / f"quantum_dirac2d_{args.dirac2d_potential}.html"
    path.write_text(_dirac_spinor_page(dirac2d_data), encoding="utf-8")
    generated.append(path)

  _DRIVEN_SCENARIOS: dict[str, tuple[str, str, dict]] = {
      # scenario_key -> (potential_type, output_filename_stem, driving_config)
      "driven_barrier": (
          "barrier",
          "quantum_driven_barrier",
          {"type": "sine", "amplitude": 0.4, "frequency": 1.8, "spatial_profile": "uniform"},
      ),
      "bloch": (
          "free",
          "quantum_bloch_oscillation",
          {"type": "sine", "amplitude": 0.35, "frequency": 2.5, "spatial_profile": "dipole_x"},
      ),
      "gaussian_pulse": (
          "harmonic",
          "quantum_gaussian_pulse",
          {
              "type": "gaussian_pulse",
              "amplitude": 0.6,
              "frequency": 0.55,
              "t_center": 1.2,
              "t_width": 0.4,
              "spatial_profile": "dipole_x",
          },
      ),
  }
  for scenario_key, (pot_type, stem, drv_cfg) in _DRIVEN_SCENARIOS.items():
    if args.scenario not in {scenario_key, "all"}:
      continue
    driven_problem = _continuous_problem_2d_driven_request(
        args.samples_spatial2d_t, args.samples_spatial2d_x, args.samples_spatial2d_y,
        pot_type, drv_cfg,
    )
    driven_data = _resolve_continuous_payload(
        engine,
        driven_problem,
        args.solver_mode,
        lambda pt=pot_type, dc=drv_cfg: _spatial_wavepacket_2d_frames(
            args.samples_spatial2d_t, args.samples_spatial2d_x, args.samples_spatial2d_y,
            pt, driving=dc,
        ),
        export_json_path,
        args.strict_export_schema,
    )
    # Ensure the driving config is preserved in the payload for the page template.
    if "driving" not in driven_data:
      driven_data["driving"] = drv_cfg
    if "has_driving" not in driven_data:
      driven_data["has_driving"] = True
    path = out_dir / f"{stem}.html"
    path.write_text(_driven_wavepacket_2d_page(driven_data), encoding="utf-8")
    generated.append(path)

  if args.scenario in {"lindblad", "all"}:
    if args.lindblad_scenario == "amplitude_damping":
      h_terms = [{"label": "X0", "coefficient": 0.5}]
      c_ops = [{"label": "AmplitudeDamping0", "coefficient": 0.8}]
    elif args.lindblad_scenario == "dephasing":
      h_terms = [{"label": "X0", "coefficient": 1.0}]
      c_ops = [{"label": "Dephasing0", "coefficient": 1.0}]
    elif args.lindblad_scenario == "depolarizing":
      h_terms = [{"label": "Z0", "coefficient": 0.5}]
      c_ops = [{"label": "Depolarizing0", "coefficient": 0.5}]
    else:  # rabi_decay
      h_terms = [{"label": "X0", "coefficient": 1.0}, {"label": "Z0", "coefficient": -0.5}]
      c_ops = [{"label": "AmplitudeDamping0", "coefficient": 0.3}]
    lindblad_data = _lindblad_frames(
        args.lindblad_qubits, h_terms, c_ops,
        total_time=2.0, dt=0.02,
        samples=args.samples_lindblad_t,
    )
    path = out_dir / f"quantum_lindblad_{args.lindblad_scenario}.html"
    path.write_text(_lindblad_page(lindblad_data), encoding="utf-8")
    generated.append(path)

  _WIGNER_SCENARIOS: dict[str, str] = {
      "wigner_free":    "free",
      "wigner_barrier": "barrier",
  }
  for scenario_key, pot_type in _WIGNER_SCENARIOS.items():
    if args.scenario not in {scenario_key, "all"}:
      continue
    nx_w = args.samples_spatial2d_x
    ny_w = args.samples_spatial2d_y
    wigner_data = _wigner_frames(
        nx=nx_w,
        ny=ny_w,
        total_time=6.0,
        dt=0.05,
        potential_type=pot_type,
        samples=args.samples_wigner_t,
    )
    path = out_dir / f"quantum_wigner_{scenario_key}.html"
    path.write_text(_wigner_page(wigner_data), encoding="utf-8")
    generated.append(path)

  # M19 — MPS/TEBD scenarios
  # mps_ising: transverse-field Ising model H = -Σ Z_i Z_{i+1} - g Σ X_i
  if args.scenario in {"mps_ising", "all"}:
    n = args.mps_sites
    ising_terms = (
        [{"label": f"Z{i}Z{i+1}", "coefficient": -1.0} for i in range(n - 1)]
        + [{"label": f"X{i}", "coefficient": -0.5} for i in range(n)]
    )
    mps_ising_data = _mps_frames(
        n_sites=n,
        hamiltonian_terms=ising_terms,
        total_time=4.0, dt=0.05,
        max_bond=args.mps_max_bond,
        samples=args.samples_mps_t,
    )
    path = out_dir / "quantum_mps_ising.html"
    path.write_text(_mps_page(mps_ising_data), encoding="utf-8")
    generated.append(path)

  # mps_xx: XX model H = Σ X_i X_{i+1} — immediately entangles |00…0⟩ (XX|00⟩=|11⟩≠0)
  if args.scenario in {"mps_xx", "all"}:
    n = args.mps_sites
    xx_terms = [{"label": f"X{i}X{i+1}", "coefficient": 1.0} for i in range(n - 1)]
    mps_xx_data = _mps_frames(
        n_sites=n,
        hamiltonian_terms=xx_terms,
        total_time=3.0, dt=0.04,
        max_bond=args.mps_max_bond,
        samples=args.samples_mps_t,
    )
    path = out_dir / "quantum_mps_xx.html"
    path.write_text(_mps_page(mps_xx_data), encoding="utf-8")
    generated.append(path)

  # M18 — Entanglement scenarios
  # entanglement_rabi: transverse-field Ising model — X0+X1 drives, Z0Z1 creates entanglement
  if args.scenario in {"entanglement_rabi", "all"}:
    h_terms_rabi = [
        {"label": "X0", "coefficient": 0.5},
        {"label": "X1", "coefficient": 0.5},
        {"label": "Z0Z1", "coefficient": 1.0},
    ]
    ent_rabi_data = _entanglement_frames(
        n_qubits=2,
        hamiltonian_terms=h_terms_rabi,
        collapse_ops=[],
        n_a=1,
        total_time=math.pi * 2,
        dt=0.02,
        samples=args.samples_entanglement_t,
    )
    path = out_dir / "quantum_entanglement_rabi.html"
    path.write_text(_entanglement_page(ent_rabi_data), encoding="utf-8")
    generated.append(path)

  # entanglement_decay: 2-qubit Heisenberg XX + amplitude damping on qubit 1
  if args.scenario in {"entanglement_decay", "all"}:
    h_terms_decay = [
        {"label": "X0X1", "coefficient": 1.0},
        {"label": "Z0",   "coefficient": 0.2},
        {"label": "Z1",   "coefficient": 0.2},
    ]
    c_ops_decay = [{"label": "AmplitudeDamping1", "coefficient": 0.5}]
    ent_decay_data = _entanglement_frames(
        n_qubits=2,
        hamiltonian_terms=h_terms_decay,
        collapse_ops=c_ops_decay,
        n_a=1,
        total_time=6.0,
        dt=0.02,
        samples=args.samples_entanglement_t,
    )
    path = out_dir / "quantum_entanglement_decay.html"
    path.write_text(_entanglement_page(ent_decay_data), encoding="utf-8")
    generated.append(path)

  # ── M20: VQE / QAOA variational algorithms ───────────────────────────────────

  # vqe_ising: VQE on the transverse-field Ising model
  # H = -Σ Z_iZ_{i+1} - 0.5 Σ X_i  (4 qubits, HEA ansatz)
  if args.scenario in {"vqe_ising", "all"}:
    n = args.vqe_qubits
    ising_ham = (
        [{"label": f"Z{i}Z{i+1}", "coefficient": -1.0} for i in range(n - 1)]
        + [{"label": f"X{i}", "coefficient": -0.5} for i in range(n)]
    )
    vqe_ising_data = _vqe_frames(
        n_qubits=n,
        cost_hamiltonian=ising_ham,
        ansatz_type="hardware_efficient",
        n_layers=args.vqe_layers,
        optimizer="adam",
        max_iterations=args.vqe_max_iter,
        learning_rate=0.12,
    )
    path = out_dir / "quantum_vqe_ising.html"
    path.write_text(_vqe_page(vqe_ising_data), encoding="utf-8")
    generated.append(path)

  # vqe_qaoa: QAOA MaxCut on a 4-node cycle graph
  if args.scenario in {"vqe_qaoa", "all"}:
    n = args.vqe_qubits
    # 4-node cycle: 0-1-2-3-0
    edges = [(i, (i + 1) % n) for i in range(n)]
    # Cost Hamiltonian for MaxCut: minimize Σ Z_iZ_j (equivalent to maximising cut)
    qaoa_ham = [{"label": f"Z{i}Z{j}", "coefficient": 1.0} for (i, j) in edges]
    vqe_qaoa_data = _vqe_frames(
        n_qubits=n,
        cost_hamiltonian=qaoa_ham,
        ansatz_type="qaoa",
        n_layers=args.vqe_layers,
        optimizer="adam",
        max_iterations=args.vqe_max_iter,
        learning_rate=0.15,
    )
    # Attach edges so _vqe_page can rebuild the final state.
    vqe_qaoa_data["qaoa_edges"] = edges
    path = out_dir / "quantum_vqe_qaoa.html"
    path.write_text(_vqe_page(vqe_qaoa_data), encoding="utf-8")
    generated.append(path)

  # ── M21 — QPE scenarios ──────────────────────────────────────────────────────
  # qpe_phase: single-qubit phase gate P(2π·φ) with exact phase φ = 3/8 = 0.375
  if args.scenario in {"qpe_phase", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("qpe_phase skipped: numpy not available", file=sys.stderr)
    else:
      phi = args.qpe_phase  # e.g. 0.375 → 3/8 → exact representation with n_ancilla >= 3
      phase_angle = 2.0 * 3.141592653589793 * phi
      # Single-qubit phase gate U = diag(1, e^{i·phase_angle})
      U_phase = np.array([[1.0, 0.0], [0.0, np.exp(1j * phase_angle)]], dtype=complex)
      # Eigenstate |1⟩ of P(θ) with eigenvalue e^{iθ}
      eigenstate_phase = np.array([0.0, 1.0], dtype=complex)
      sweep_ns = list(range(1, min(args.qpe_sweep_max, 13) + 1))
      qpe_phase_data = _qpe_frames(
          n_ancilla=args.qpe_ancilla,
          unitary=U_phase,
          eigenstate=eigenstate_phase,
          n_shots=args.qpe_shots,
          true_phase=phi,
          sweep_ancilla=sweep_ns,
          seed=42,
      )
      path = out_dir / "quantum_qpe_phase.html"
      path.write_text(_qpe_page(qpe_phase_data, title="QPE — Phase Gate"), encoding="utf-8")
      generated.append(path)

  # qpe_hamiltonian: Heisenberg ZZ Hamiltonian H = Z₀Z₁ → U = exp(-iHt), t = π/8
  # Eigenvalues of Z₀Z₁: ±1 → U eigenvalues e^{∓iπ/8} → phases ∓1/16 mod 1
  if args.scenario in {"qpe_hamiltonian", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("qpe_hamiltonian skipped: numpy not available", file=sys.stderr)
    else:
      t_evo = 3.141592653589793 / 8.0  # π/8
      # H = Z₀ ⊗ Z₁ = diag(1, -1, -1, 1) in computational basis.
      H_zz = np.diag([1.0, -1.0, -1.0, 1.0])
      # U = exp(-i H t) = diag(e^{-it}, e^{it}, e^{it}, e^{-it})
      U_ham = np.diag(np.exp(-1j * t_evo * np.diag(H_zz)))
      # Eigenstate: |01⟩ (index 1) — eigenvalue of H_zz is -1 → phase = t/(2π)
      eigenstate_ham = np.zeros(4, dtype=complex)
      eigenstate_ham[1] = 1.0
      true_phase_ham = t_evo / (2.0 * 3.141592653589793)  # ≈ 0.0625 = 1/16
      sweep_ns = list(range(1, min(args.qpe_sweep_max, 13) + 1))
      qpe_ham_data = _qpe_frames(
          n_ancilla=args.qpe_ancilla,
          unitary=U_ham,
          eigenstate=eigenstate_ham,
          n_shots=args.qpe_shots,
          true_phase=true_phase_ham,
          sweep_ancilla=sweep_ns,
          seed=42,
      )
      path = out_dir / "quantum_qpe_hamiltonian.html"
      path.write_text(_qpe_page(qpe_ham_data, title="QPE — ZZ Hamiltonian"), encoding="utf-8")
      generated.append(path)

  # ── M22 — QEC scenarios ──────────────────────────────────────────────────────
  # Build the threshold sweep points (log-spaced from 0.01 to 0.45)
  _qec_sweep_pts: list[float] | None = None
  if args.qec_sweep:
    _qec_sweep_pts = [round(0.01 + 0.44 * i / 19, 4) for i in range(20)]

  # qec_bit_flip: 3-qubit bit-flip repetition code with X errors
  if args.scenario in {"qec_bit_flip", "all"}:
    qec_bf_data = _qec_frames(
        code="bit_flip_3",
        physical_error_rate=args.qec_error_rate,
        error_type="X",
        n_trials=args.qec_trials,
        n_detail_trials=64,
        sweep_error_rates=_qec_sweep_pts,
        seed=42,
    )
    path = out_dir / "quantum_qec_bit_flip.html"
    path.write_text(_qec_page(qec_bf_data, title="QEC — 3-Qubit Bit-Flip Code"), encoding="utf-8")
    generated.append(path)

  # qec_phase_flip: 3-qubit phase-flip code with Z errors
  if args.scenario in {"qec_phase_flip", "all"}:
    qec_pf_data = _qec_frames(
        code="phase_flip_3",
        physical_error_rate=args.qec_error_rate,
        error_type="Z",
        n_trials=args.qec_trials,
        n_detail_trials=64,
        sweep_error_rates=_qec_sweep_pts,
        seed=42,
    )
    path = out_dir / "quantum_qec_phase_flip.html"
    path.write_text(_qec_page(qec_pf_data, title="QEC — 3-Qubit Phase-Flip Code"), encoding="utf-8")
    generated.append(path)

  # qec_shor: 9-qubit Shor code with depolarizing noise
  if args.scenario in {"qec_shor", "all"}:
    qec_shor_data = _qec_frames(
        code="shor_9",
        physical_error_rate=args.qec_error_rate,
        error_type="depolarizing",
        n_trials=args.qec_trials,
        n_detail_trials=64,
        sweep_error_rates=_qec_sweep_pts,
        seed=42,
    )
    path = out_dir / "quantum_qec_shor.html"
    path.write_text(_qec_page(qec_shor_data, title="QEC — 9-Qubit Shor Code"), encoding="utf-8")
    generated.append(path)

  # ── M23 — Schwinger model scenarios ─────────────────────────────────────────
  # schwinger_vacuum: ground state observables + chiral condensate mass sweep
  if args.scenario in {"schwinger_vacuum", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("schwinger_vacuum skipped: numpy not available", file=sys.stderr)
    else:
      vac_data = _schwinger_frames(
          n_sites=args.schwinger_sites,
          mass=args.schwinger_mass,
          coupling=args.schwinger_coupling,
          background=0.0,
          scenario="vacuum",
          n_mass_sweep=args.schwinger_mass_sweep,
      )
      path = out_dir / "quantum_schwinger_vacuum.html"
      path.write_text(_schwinger_page(vac_data, title="Schwinger Model — Vacuum"), encoding="utf-8")
      generated.append(path)

  # schwinger_string: string tension E(R) vs quark–antiquark separation
  if args.scenario in {"schwinger_string", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("schwinger_string skipped: numpy not available", file=sys.stderr)
    else:
      str_data = _schwinger_frames(
          n_sites=args.schwinger_sites,
          mass=args.schwinger_mass,
          coupling=args.schwinger_coupling,
          background=0.0,
          scenario="string",
          n_mass_sweep=args.schwinger_mass_sweep,
      )
      path = out_dir / "quantum_schwinger_string.html"
      path.write_text(_schwinger_page(str_data, title="Schwinger Model — String Tension"), encoding="utf-8")
      generated.append(path)

  # schwinger_dynamics: pair production under background field quench
  if args.scenario in {"schwinger_dynamics", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
      # Try scipy for exact expm; fall back to Trotter.
      try:
        import scipy.linalg as _scipy_linalg  # noqa: PLC0415
        _has_scipy = True
      except ImportError:
        _has_scipy = False
    except ImportError:
      print("schwinger_dynamics skipped: numpy not available", file=sys.stderr)
    else:
      # Use smaller N for dynamics (exact matrix exp is O(dim³)).
      n_dyn = min(args.schwinger_sites, 6)
      if not _has_scipy:
        # Fallback: eigendecompose once → fast time evolution.
        dyn_frames = _schwinger_fallback_dynamics(
            n_sites=n_dyn,
            mass=args.schwinger_mass,
            coupling=args.schwinger_coupling,
            background_quench=args.schwinger_background,
            dt=args.schwinger_dt,
            n_steps=args.schwinger_steps,
        )
        dyn_data = _schwinger_frames(
            n_sites=n_dyn,
            mass=args.schwinger_mass,
            coupling=args.schwinger_coupling,
            background=0.0,
            scenario="vacuum",
            n_mass_sweep=args.schwinger_mass_sweep,
        )
        dyn_data["scenario"] = "dynamics"
        dyn_data["frames"] = dyn_frames
        dyn_data["background"] = args.schwinger_background
      else:
        dyn_data = _schwinger_frames(
            n_sites=n_dyn,
            mass=args.schwinger_mass,
            coupling=args.schwinger_coupling,
            background=args.schwinger_background,
            scenario="dynamics",
            dt=args.schwinger_dt,
            n_steps=args.schwinger_steps,
            n_mass_sweep=args.schwinger_mass_sweep,
        )
      path = out_dir / "quantum_schwinger_dynamics.html"
      path.write_text(_schwinger_page(dyn_data, title="Schwinger Model — Pair Production Dynamics"), encoding="utf-8")
      generated.append(path)

  # ── M24 — φ⁴ Scalar QFT scenarios ───────────────────────────────────────────
  _phi4_common = dict(
      n_sites=args.phi4_sites,
      lam=args.phi4_lambda,
      kappa=args.phi4_kappa,
      d_local=args.phi4_d_local,
      n_sweep=args.phi4_sweep_points,
  )

  # phi4_symmetric: m² > 0, single-well potential, Z₂ symmetric vacuum
  if args.scenario in {"phi4_symmetric", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("phi4_symmetric skipped: numpy not available", file=sys.stderr)
    else:
      sym_data = _phi4_frames(m2=max(args.phi4_m2, 0.5), scenario="symmetric", **_phi4_common)
      path = out_dir / "quantum_phi4_symmetric.html"
      path.write_text(_phi4_page(sym_data, title="φ⁴ QFT — Symmetric Phase"), encoding="utf-8")
      generated.append(path)

  # phi4_broken: m² < 0, Mexican-hat potential, Z₂ spontaneously broken
  if args.scenario in {"phi4_broken", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("phi4_broken skipped: numpy not available", file=sys.stderr)
    else:
      brk_data = _phi4_frames(m2=min(args.phi4_m2, -1.5), scenario="broken", **_phi4_common)
      path = out_dir / "quantum_phi4_broken.html"
      path.write_text(_phi4_page(brk_data, title="φ⁴ QFT — Broken Phase (SSB)"), encoding="utf-8")
      generated.append(path)

  # phi4_sweep: scan m² across the quantum phase transition, track order parameter + gap
  if args.scenario in {"phi4_sweep", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("phi4_sweep skipped: numpy not available", file=sys.stderr)
    else:
      swp_data = _phi4_frames(m2=args.phi4_m2, scenario="sweep", **_phi4_common)
      path = out_dir / "quantum_phi4_sweep.html"
      path.write_text(_phi4_page(swp_data, title="φ⁴ QFT — Phase Transition Sweep"), encoding="utf-8")
      generated.append(path)

  if args.scenario in {"phi4_dynamics", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("phi4_dynamics skipped: numpy not available", file=sys.stderr)
    else:
      dyn_data = _phi4_dynamics_frames(
          n_sites=args.phi4_sites,
          d_local=args.phi4_d_local,
          lam=args.phi4_lambda,
          kappa=args.phi4_kappa,
          m2_initial=args.phi4_dynamics_m2_initial,
          m2_final=args.phi4_dynamics_m2_final,
          n_steps=args.phi4_dynamics_steps,
          dt=args.phi4_dynamics_dt,
      )
      path = out_dir / "quantum_phi4_dynamics.html"
      path.write_text(_phi4_dynamics_page(dyn_data, title="φ⁴ QFT — Quantum Quench Dynamics"), encoding="utf-8")
      generated.append(path)

  # ── M27-C: Hubbard Model ──────────────────────────────────────────────────
  if args.scenario in {"hubbard", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("hubbard skipped: numpy not available", file=sys.stderr)
    else:
      hub_data = _hubbard_frames(
          n_sites=args.hubbard_sites,
          t_hop=args.hubbard_t,
          U=args.hubbard_U,
          n_up=args.hubbard_n_up,
          n_dn=args.hubbard_n_dn,
          scenario="sweep",
          n_sweep=args.hubbard_sweep_points,
          U_max=args.hubbard_U_max,
      )
      path = out_dir / "quantum_hubbard.html"
      path.write_text(_hubbard_page(hub_data, title="1D Hubbard Model — Mott Insulator"), encoding="utf-8")
      generated.append(path)

  # ── M27-D: Floquet Kicked-Ising ──────────────────────────────────────────
  if args.scenario in {"floquet", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("floquet skipped: numpy not available", file=sys.stderr)
    else:
      flo_data = _floquet_frames(
          n_sites=args.floquet_sites,
          J=args.floquet_J,
          h_z=args.floquet_hz,
          drive_amp=args.floquet_amp,
          drive_omega=args.floquet_omega,
          n_periods=args.floquet_periods,
          scenario=args.floquet_scenario,
      )
      path = out_dir / "quantum_floquet.html"
      path.write_text(_floquet_page(flo_data, title="Floquet 周期驱动 — Kicked-Ising"), encoding="utf-8")
      generated.append(path)

  # ── M28-E: XXZ Heisenberg chain ──────────────────────────────────────────
  if args.scenario in {"heisenberg", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("heisenberg skipped: numpy not available", file=sys.stderr)
    else:
      heis_data = _heisenberg_frames(
          n_sites=args.heisenberg_sites,
          J=args.heisenberg_J,
          Jz=args.heisenberg_Jz,
          h=args.heisenberg_h,
          scenario="sweep",
          n_sweep=args.heisenberg_sweep_points,
          Jz_min=args.heisenberg_Jz_min,
          Jz_max=args.heisenberg_Jz_max,
      )
      path = out_dir / "quantum_heisenberg.html"
      path.write_text(_heisenberg_page(heis_data, title="1D XXZ Heisenberg Chain"), encoding="utf-8")
      generated.append(path)

  # ── M28-F: Bose-Hubbard model ─────────────────────────────────────────────
  if args.scenario in {"bose_hubbard", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("bose_hubbard skipped: numpy not available", file=sys.stderr)
    else:
      bh_data = _bose_hubbard_frames(
          n_sites=args.bose_hubbard_sites,
          n_bosons=args.bose_hubbard_bosons,
          n_max=args.bose_hubbard_nmax,
          J=args.bose_hubbard_J,
          U=args.bose_hubbard_U,
          mu=args.bose_hubbard_mu,
          scenario="sweep",
          n_sweep=args.bose_hubbard_sweep_points,
          J_min=0.0,
          J_max=args.bose_hubbard_J_max,
      )
      path = out_dir / "quantum_bose_hubbard.html"
      path.write_text(_bose_hubbard_page(bh_data, title="Bose-Hubbard Model — Mott/Superfluid"), encoding="utf-8")
      generated.append(path)

  # ── M29-G: Transverse-Field Ising model ──────────────────────────────────
  if args.scenario in {"tfi", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("tfi skipped: numpy not available", file=sys.stderr)
    else:
      tfi_data = _tfi_frames(
          n_sites=args.tfi_sites,
          J=args.tfi_J,
          h=args.tfi_h,
          scenario="sweep",
          n_sweep=args.tfi_sweep_points,
          h_min=args.tfi_h_min,
          h_max=args.tfi_h_max,
      )
      path = out_dir / "quantum_tfi.html"
      path.write_text(_tfi_page(tfi_data, title="Transverse-Field Ising — QPT at h=J"), encoding="utf-8")
      generated.append(path)

  # ── M29-H: SSH model ──────────────────────────────────────────────────────
  if args.scenario in {"ssh", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("ssh skipped: numpy not available", file=sys.stderr)
    else:
      ssh_data = _ssh_frames(
          n_cells=args.ssh_cells,
          t=args.ssh_t,
          delta=args.ssh_delta,
          scenario="sweep",
          n_sweep=args.ssh_sweep_points,
          delta_min=args.ssh_delta_min,
          delta_max=args.ssh_delta_max,
      )
      path = out_dir / "quantum_ssh.html"
      path.write_text(_ssh_page(ssh_data, title="SSH Model — Bulk-Boundary Correspondence"), encoding="utf-8")
      generated.append(path)

  # ── M30-I: Kitaev chain ───────────────────────────────────────────────────
  if args.scenario in {"kitaev", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("kitaev skipped: numpy not available", file=sys.stderr)
    else:
      kitaev_data = _kitaev_frames(
          n_sites=args.kitaev_sites,
          t=args.kitaev_t,
          delta=args.kitaev_delta,
          mu=args.kitaev_mu,
          scenario="sweep",
          n_sweep=args.kitaev_sweep_points,
          mu_min=args.kitaev_mu_min,
          mu_max=args.kitaev_mu_max,
      )
      path = out_dir / "quantum_kitaev.html"
      path.write_text(_kitaev_page(kitaev_data, title="Kitaev Chain — Majorana Zero Modes"), encoding="utf-8")
      generated.append(path)

  # ── M30-J: Anderson localization ──────────────────────────────────────────
  if args.scenario in {"anderson", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("anderson skipped: numpy not available", file=sys.stderr)
    else:
      anderson_data = _anderson_frames(
          n_sites=args.anderson_sites,
          t=args.anderson_t,
          W=args.anderson_W,
          n_samples=args.anderson_samples,
          scenario="sweep",
          n_sweep=args.anderson_sweep_points,
          W_min=args.anderson_W_min,
          W_max=args.anderson_W_max,
      )
      path = out_dir / "quantum_anderson.html"
      path.write_text(_anderson_page(anderson_data, title="Anderson Localization — 1D Disorder"), encoding="utf-8")
      generated.append(path)

  # ── M31-K: 2D Chern Insulator ─────────────────────────────────────────────
  if args.scenario in {"chern", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("chern skipped: numpy not available", file=sys.stderr)
    else:
      chern_data = _chern_frames(
          n_sites=args.chern_sites,
          m=args.chern_m,
          t=args.chern_t,
          n_k=args.chern_nk,
          n_sweep=args.chern_sweep_points,
          m_min=args.chern_m_min,
          m_max=args.chern_m_max,
      )
      path = out_dir / "quantum_chern.html"
      path.write_text(_chern_page(chern_data, title="2D Chern Insulator — Qi-Wu-Zhang Model"), encoding="utf-8")
      generated.append(path)

  # ── M31-L: Hofstadter Butterfly ───────────────────────────────────────────
  if args.scenario in {"hofstadter", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("hofstadter skipped: numpy not available", file=sys.stderr)
    else:
      hof_data = _hofstadter_frames(
          t=args.hofstadter_t,
          q_max=args.hofstadter_q_max,
          phi=args.hofstadter_phi,
          n_k=args.hofstadter_nk,
          n_ky=args.hofstadter_nky,
      )
      path = out_dir / "quantum_hofstadter.html"
      path.write_text(_hofstadter_page(hof_data, title="Hofstadter Butterfly — Fractal Landau Levels"), encoding="utf-8")
      generated.append(path)

  # ── M32: Many-Body Localization ───────────────────────────────────────────
  if args.scenario in {"mbl", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("mbl skipped: numpy not available", file=sys.stderr)
    else:
      mbl_data = _mbl_frames(
          n_sites=args.mbl_sites,
          J=args.mbl_J,
          delta=args.mbl_delta,
          W=args.mbl_W,
          scenario=args.mbl_scenario,
          W_min=args.mbl_W_min,
          W_max=args.mbl_W_max,
          n_sweep=args.mbl_sweep_points,
          n_samples=args.mbl_samples,
          seed=args.mbl_seed,
      )
      path = out_dir / "quantum_mbl.html"
      path.write_text(_mbl_page(mbl_data, title="Many-Body Localization — Disordered XXZ Chain"), encoding="utf-8")
      generated.append(path)

  # ── M33: Spectral Functions ───────────────────────────────────────────────
  if args.scenario in {"spectral", "all"}:
    try:
      import numpy as np  # noqa: PLC0415
    except ImportError:
      print("spectral skipped: numpy not available", file=sys.stderr)
    else:
      spectral_data = _spectral_frames(
          n_sites=args.spectral_sites,
          J=args.spectral_J,
          delta=args.spectral_delta,
          h=args.spectral_h,
          scenario=args.spectral_scenario,
          n_k=args.spectral_nk,
          n_omega=args.spectral_nomega,
          omega_min=args.spectral_omega_min,
          omega_max=args.spectral_omega_max,
          eta=args.spectral_eta,
          tau_max=args.spectral_tau_max,
          n_tau=args.spectral_ntau,
      )
      path = out_dir / "quantum_spectral.html"
      path.write_text(_spectral_page(spectral_data, title="Spectral Functions — S(k,\u03c9) & Imaginary Time"), encoding="utf-8")
      generated.append(path)

  # ── M34: 10-Fold Way Topological Classification ──────────────────────────
  if args.scenario in {"tenfold", "all"}:
    tenfold_data = _tenfold_frames(
        az_class=args.tenfold_class,
        dimension=args.tenfold_dim,
        model=args.tenfold_model,
        scenario=args.tenfold_scenario,
        t1=args.tenfold_t1,
        t2=args.tenfold_t2,
        m=args.tenfold_m,
        t=args.tenfold_t,
        t_hop=args.tenfold_t_hop,
        mu=args.tenfold_mu,
        n_k=args.tenfold_nk,
        sweep_param=args.tenfold_sweep_param,
        param_min=args.tenfold_param_min,
        param_max=args.tenfold_param_max,
        n_sweep=args.tenfold_n_sweep,
    )
    path = out_dir / "quantum_tenfold.html"
    path.write_text(_tenfold_page(tenfold_data, title="10-Fold Way — Altland-Zirnbauer Topological Classification"), encoding="utf-8")
    generated.append(path)

  # ── M36: Wilson Loop & Wannier Charge Centers ─────────────────────────────
  if args.scenario in {"wilson_loop", "all"}:
    wl_data = _wilson_loop_frames(
        M=args.wilson_loop_M,
        A=args.wilson_loop_A,
        B=args.wilson_loop_B,
        C=args.wilson_loop_C,
        D=args.wilson_loop_D,
        n_kx=args.wilson_loop_nkx,
        n_ky=args.wilson_loop_nky,
        scenario="phase_diagram",
        sweep_param=args.wilson_loop_sweep_param,
        param_min=args.wilson_loop_param_min,
        param_max=args.wilson_loop_param_max,
        n_sweep=args.wilson_loop_n_sweep,
        source="python-reference",
    )
    path = out_dir / "quantum_wilson_loop.html"
    path.write_text(_wilson_loop_page(wl_data, title="Wilson Loop & Wannier Charge Centers — BHZ Z₂ Topological Insulator"), encoding="utf-8")
    generated.append(path)

  # ── M35: Second Chern Number & 4D Quantum Hall Effect ─────────────────────
  if args.scenario in {"second_chern", "all"}:
    sc_data = _second_chern_frames(
        m=args.second_chern_m,
        t=args.second_chern_t,
        n_k=args.second_chern_nk,
        scenario="phase_diagram",
        sweep_param=args.second_chern_sweep_param,
        param_min=args.second_chern_param_min,
        param_max=args.second_chern_param_max,
        n_sweep=args.second_chern_n_sweep,
        source="python-reference",
    )
    path = out_dir / "quantum_second_chern.html"
    path.write_text(_second_chern_page(sc_data, title="Second Chern Number — 4D Quantum Hall Effect"), encoding="utf-8")
    generated.append(path)

  # ── M37: Edge States / Surface Spectrum ───────────────────────────────────
  if args.scenario in {"edge_states", "all"}:
    try:
      import numpy as np  # noqa: PLC0415, F401
    except ImportError:
      print("edge_states skipped: numpy not available", file=sys.stderr)
    else:
      es_data = _edge_states_frames(
          M=args.edge_states_M,
          A=args.edge_states_A,
          B=args.edge_states_B,
          C=args.edge_states_C,
          D=args.edge_states_D,
          n_kx=args.edge_states_nkx,
          n_y=args.edge_states_ny,
          scenario="spectrum",
          sweep_param=args.edge_states_sweep_param,
          param_min=args.edge_states_param_min,
          param_max=args.edge_states_param_max,
          n_sweep=args.edge_states_n_sweep,
          source="python-reference",
      )
      path = out_dir / "quantum_edge_states.html"
      path.write_text(_edge_states_page(es_data, title="Edge States / Surface Spectrum — BHZ Topological Insulator"), encoding="utf-8")
      generated.append(path)

  # ── M38: Haldane Model (Quantum Anomalous Hall Effect) ─────────────────────
  if args.scenario in {"haldane", "all"}:
    try:
      import numpy as np  # noqa: PLC0415, F401
    except ImportError:
      print("haldane skipped: numpy not available", file=sys.stderr)
    else:
      h_data = _haldane_frames(
          t1=args.haldane_t1,
          t2=args.haldane_t2,
          phi=args.haldane_phi,
          M_val=args.haldane_M,
          n_k=args.haldane_nk,
          scenario="phase_diagram",
          sweep_param=args.haldane_sweep_param,
          param_min=args.haldane_param_min,
          param_max=args.haldane_param_max,
          n_sweep=args.haldane_n_sweep,
          source="python-reference",
      )
      path = out_dir / "quantum_haldane.html"
      path.write_text(_haldane_page(h_data, title="Haldane Model — Quantum Anomalous Hall Effect"), encoding="utf-8")
      generated.append(path)

  # ── M39: Weyl Semimetal & Surface Fermi Arcs ──────────────────────────────
  if args.scenario in {"weyl", "all"}:
    try:
      import numpy as np  # noqa: PLC0415, F401
    except ImportError:
      print("weyl skipped: numpy not available", file=sys.stderr)
    else:
      weyl_data = _weyl_frames(
          m_param=args.weyl_m,
          n_k=args.weyl_nk,
          n_z=args.weyl_nz,
          n_kz=args.weyl_nkz,
          fermi_energy=args.weyl_fermi_energy,
          scenario="fermi_arc",
          param_min=args.weyl_param_min,
          param_max=args.weyl_param_max,
          n_sweep=args.weyl_n_sweep,
          source="python-reference",
      )
      path = out_dir / "quantum_weyl.html"
      path.write_text(_weyl_page(weyl_data, title="Weyl Semimetal & Surface Fermi Arcs"), encoding="utf-8")
      path = out_dir / "quantum_weyl.html"
      path.write_text(_weyl_page(weyl_data, title="Weyl Semimetal & Surface Fermi Arcs"), encoding="utf-8")
      generated.append(path)

  # ── M40: Fu-Kane Superconductor — Majorana Zero Modes ────────────────────
  if args.scenario in {"fu_kane", "all"}:
    try:
      import numpy as np  # noqa: PLC0415, F401
    except ImportError:
      print("fu_kane skipped: numpy not available", file=sys.stderr)
    else:
      fk_data = _fu_kane_frames(
          v_F=args.fu_kane_vF,
          M_val=args.fu_kane_M,
          mu=args.fu_kane_mu,
          Delta=args.fu_kane_Delta,
          n_k=args.fu_kane_nk,
          n_y=args.fu_kane_ny,
          n_kx=args.fu_kane_nkx,
          scenario="both",
          source="python-reference",
      )
      path = out_dir / "quantum_fu_kane.html"
      path.write_text(_fu_kane_page(fk_data, title="Fu-Kane Superconductor \u2014 Majorana Zero Modes"), encoding="utf-8")
      generated.append(path)

  # ── M41: Lutchyn-Oreg Majorana Nanowire ──────────────────────────────────
  if args.scenario in {"lutchyn_oreg", "all"}:
    try:
      import numpy as np  # noqa: PLC0415, F401
    except ImportError:
      print("lutchyn_oreg skipped: numpy not available", file=sys.stderr)
    else:
      lo_data = _lutchyn_oreg_frames(
          t=args.lo_t,
          alpha=args.lo_alpha,
          mu=args.lo_mu,
          Delta=args.lo_Delta,
          V_z=args.lo_Vz,
          n_k=args.lo_nk,
          n_sites=args.lo_nsites,
          scenario="both",
          source="python-reference",
      )
      path = out_dir / "quantum_lutchyn_oreg.html"
      path.write_text(_lutchyn_oreg_page(lo_data, title="Lutchyn-Oreg Majorana Nanowire"), encoding="utf-8")
      generated.append(path)

  # ── M42: Kitaev honeycomb ──────────────────────────────────────────────────
  if args.scenario in {"kitaev_honeycomb", "all"}:
    try:
      import numpy as np  # noqa: PLC0415, F401
    except ImportError:
      print("kitaev_honeycomb skipped: numpy not available", file=sys.stderr)
    else:
      kh_data = _kitaev_honeycomb_frames(
          Jx=args.kh_Jx,
          Jy=args.kh_Jy,
          Jz=args.kh_Jz,
          kappa=args.kh_kappa,
          n_k=args.kh_nk,
          n_pd=args.kh_npd,
          scenario="both",
          source="python-reference",
      )
      path = out_dir / "quantum_kitaev_honeycomb.html"
      path.write_text(_kitaev_honeycomb_page(kh_data), encoding="utf-8")
      generated.append(path)

  # ── M43: BBH quadrupole insulator ──────────────────────────────────────────
  if args.scenario in {"bbh", "all"}:
    try:
      import numpy as np  # noqa: PLC0415, F401
    except ImportError:
      print("bbh skipped: numpy not available", file=sys.stderr)
    else:
      bbh_data = _bbh_frames(
          gamma_x=args.bbh_gamma_x,
          lambda_x=args.bbh_lambda_x,
          gamma_y=args.bbh_gamma_y,
          lambda_y=args.bbh_lambda_y,
          n_k=args.bbh_nk,
          n_sites=args.bbh_nsites,
          n_pd=args.bbh_npd,
          scenario="both",
          source="python-reference",
      )
      path = out_dir / "quantum_bbh.html"
      path.write_text(_bbh_page(bbh_data), encoding="utf-8")
      generated.append(path)

  # ── M44: Floquet topological insulator ─────────────────────────────────────
  if args.scenario in {"floquet_topo", "all"}:
    try:
      import numpy as np  # noqa: PLC0415, F401
    except ImportError:
      print("floquet_topo skipped: numpy not available", file=sys.stderr)
    else:
      ft_data = _floquet_topo_frames(
          J=args.ft_J,
          T=args.ft_T,
          n_k=args.ft_nk,
          n_y=args.ft_ny,
          n_J=args.ft_nJ,
          J_max=args.ft_Jmax,
          scenario="both",
          source="python-reference",
      )
      path = out_dir / "quantum_floquet_topo.html"
      path.write_text(_floquet_topo_page(ft_data), encoding="utf-8")
      generated.append(path)

  if args.scenario in {"non_herm", "all"}:
    try:
      import numpy as np  # noqa: PLC0415, F401
    except ImportError:
      print("non_herm skipped: numpy not available", file=sys.stderr)
    else:
      nh_data = _non_herm_frames(
          J=args.nh_J,
          gamma=args.nh_gamma,
          n_sites=args.nh_nsites,
          n_k=args.nh_nk,
          n_gamma=args.nh_ngamma,
          gamma_max=args.nh_gammax,
          scenario="both",
          source="python-reference",
      )
      path = out_dir / "quantum_non_herm.html"
      path.write_text(_non_herm_page(nh_data), encoding="utf-8")
      generated.append(path)

  if args.scenario in {"aah", "all"}:
    try:
      import numpy as np  # noqa: PLC0415, F401
    except ImportError:
      print("aah skipped: numpy not available", file=sys.stderr)
    else:
      aah_data = _aah_frames(
          t=args.aah_t,
          V=args.aah_V,
          n_sites=args.aah_nsites,
          n_V=args.aah_nV,
          V_max=args.aah_Vmax,
          scenario="both",
          source="python-reference",
      )
      path = out_dir / "quantum_aah.html"
      path.write_text(_aah_page(aah_data), encoding="utf-8")
      generated.append(path)

  if args.scenario in {"syk", "all"}:
    try:
      import numpy as np  # noqa: PLC0415, F401
    except ImportError:
      print("syk skipped: numpy not available", file=sys.stderr)
    else:
      syk_data = _syk_frames(
          N=args.syk_N,
          J=args.syk_J,
          beta=args.syk_beta,
          n_samples=args.syk_samples,
          n_t=args.syk_nt,
          t_max=args.syk_tmax,
          scenario="both",
          source="python-reference",
      )
      path = out_dir / "quantum_syk.html"
      path.write_text(_syk_page(syk_data), encoding="utf-8")
      generated.append(path)

  if args.scenario in {"kondo", "all"}:
    try:
      import numpy as np  # noqa: PLC0415, F401
    except ImportError:
      print("kondo skipped: numpy not available", file=sys.stderr)
    else:
      kondo_data = _kondo_frames(
          N_bath=args.kondo_Nbath,
          eps_d=args.kondo_epsd,
          U=args.kondo_U,
          V=args.kondo_V,
          bandwidth=args.kondo_bw,
          n_V=args.kondo_nV,
          V_max=args.kondo_Vmax,
          n_eps=args.kondo_neps,
          scenario="both",
          source="python-reference",
      )
      path = out_dir / "quantum_kondo.html"
      path.write_text(_kondo_page(kondo_data), encoding="utf-8")
      generated.append(path)

  if args.scenario in {"mipt", "all"}:
    try:
      import numpy as np  # noqa: PLC0415, F401
    except ImportError:
      print("mipt skipped: numpy not available", file=sys.stderr)
    else:
      mipt_data = _mipt_frames(
          L=args.mipt_L,
          p=args.mipt_p,
          n_p=args.mipt_np,
          p_max=args.mipt_pmax,
          n_samples=args.mipt_nsamples,
          L_max=args.mipt_Lmax,
          seed=args.mipt_seed,
          scenario="both",
          source="python-reference",
      )
      path = out_dir / "quantum_mipt.html"
      path.write_text(_mipt_page(mipt_data), encoding="utf-8")
      generated.append(path)

  # ── M25: Unified Dashboard ─────────────────────────────────────────────────
  # ── M25: Unified Dashboard ─────────────────────────────────────────────────
  # When all scenarios were generated and --dashboard is active, build a single
  # quantum_dashboard.html that embeds every scenario page via iframe srcdoc.
  if args.scenario == "all" and args.dashboard:
    # Map logical scenario keys → file stems used when writing individual pages.
    _DASH_STEM_MAP = {
        "rabi":               "rabi_population",
        "mps":                "mps_diagnostics",
        "spatial":            f"quantum_spatial_wavepacket_{args.spatial_potential}",
        "spatial2d":          f"quantum_spatial2d_wavepacket_{args.spatial2d_potential}",
        "spatial3d":          f"quantum_spatial3d_wavepacket_{args.spatial3d_potential}_{args.spatial3d_boundary}",
        "driven_barrier":     "quantum_driven_barrier",
        "bloch":              "quantum_bloch_oscillation",
        "gaussian_pulse":     "quantum_gaussian_pulse",
        "dirac2d":            f"quantum_dirac2d_{args.dirac2d_potential}",
        "lindblad":           f"quantum_lindblad_{args.lindblad_scenario}",
        "wigner_free":        "quantum_wigner_wigner_free",
        "wigner_barrier":     "quantum_wigner_wigner_barrier",
        "mps_ising":          "quantum_mps_ising",
        "mps_xx":             "quantum_mps_xx",
        "entanglement_rabi":  "quantum_entanglement_rabi",
        "entanglement_decay": "quantum_entanglement_decay",
        "vqe_ising":          "quantum_vqe_ising",
        "vqe_qaoa":           "quantum_vqe_qaoa",
        "qpe_phase":          "quantum_qpe_phase",
        "qpe_hamiltonian":    "quantum_qpe_hamiltonian",
        "qec_bit_flip":       "quantum_qec_bit_flip",
        "qec_phase_flip":     "quantum_qec_phase_flip",
        "qec_shor":           "quantum_qec_shor",
        "schwinger_vacuum":   "quantum_schwinger_vacuum",
        "schwinger_string":   "quantum_schwinger_string",
        "schwinger_dynamics": "quantum_schwinger_dynamics",
        "phi4_symmetric":     "quantum_phi4_symmetric",
        "phi4_broken":        "quantum_phi4_broken",
        "phi4_sweep":         "quantum_phi4_sweep",
        "phi4_dynamics":      "quantum_phi4_dynamics",
        "hubbard":            "quantum_hubbard",
        "floquet":            "quantum_floquet",
        "heisenberg":         "quantum_heisenberg",
        "bose_hubbard":       "quantum_bose_hubbard",
        "tfi":                "quantum_tfi",
        "ssh":                "quantum_ssh",
        "kitaev":             "quantum_kitaev",
        "anderson":           "quantum_anderson",
        "chern":              "quantum_chern",
        "hofstadter":         "quantum_hofstadter",
        "mbl":                "quantum_mbl",
        "spectral":           "quantum_spectral",
        "tenfold":            "quantum_tenfold",
        "second_chern":       "quantum_second_chern",
        "wilson_loop":        "quantum_wilson_loop",
        "edge_states":        "quantum_edge_states",
        "haldane":            "quantum_haldane",
        "weyl":               "quantum_weyl",
        "fu_kane":            "quantum_fu_kane",
        "lutchyn_oreg":       "quantum_lutchyn_oreg",
        "kitaev_honeycomb":   "quantum_kitaev_honeycomb",
        "bbh":                "quantum_bbh",
        "floquet_topo":       "quantum_floquet_topo",
        "non_herm":           "quantum_non_herm",
        "aah":                "quantum_aah",
        "syk":                "quantum_syk",
        "kondo":              "quantum_kondo",
        "mipt":               "quantum_mipt",
    }
    dash_pages: dict = {}
    for key, stem in _DASH_STEM_MAP.items():
      candidate = out_dir / f"{stem}.html"
      if candidate.exists():
        dash_pages[key] = candidate.read_text(encoding="utf-8")
    if dash_pages:
      dash_path = out_dir / "quantum_dashboard.html"
      dash_path.write_text(_dashboard_page(dash_pages), encoding="utf-8")
      generated.append(dash_path)

  for p in generated:
    print(f"wrote visualization: {p}")
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except (FileNotFoundError, RuntimeError) as ex:
    print(f"error: {ex}", file=sys.stderr)
    raise SystemExit(1)
