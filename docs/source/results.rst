Results
=======

Everything below is produced by code in this repository, and every figure has the data
file it was rendered from next to it. Where an exact answer exists — an analytic solution,
an independent exact diagonalisation, a dense state-vector simulation — the comparison is
stated as a number, not as "good agreement".

Two conventions are worth knowing before reading any of it. **Lossless** means the bond
dimension χ was at or above the full Hilbert-space bound, so the result carries no
truncation error at all and an agreement figure measures the method rather than the
cutoff. And where the engine is compared against itself at a different χ, that is a
convergence check, not a validation — those are labelled as such.

Against exact answers
---------------------

**The Hubbard chain against the Lieb–Wu integral equations.** The 1D Fermi-Hubbard model
is exactly solvable, so the bulk energy per site is known analytically rather than
numerically. Extrapolating the engine's incremental bulk energy at N = 64 and comparing
against the Lieb–Wu result:

.. list-table::
   :header-rows: 1

   * - U/t
     - engine, bulk extrapolation
     - Lieb–Wu exact
     - difference
   * - 2
     - −0.84429042
     - −0.84437452
     - 8.4e-5
   * - 4
     - −0.57368361
     - −0.57372973
     - 4.6e-5
   * - 8
     - −0.32750449
     - −0.32753126
     - 2.7e-5

.. image:: ../hubbard_lieb_wu.svg
   :alt: Hubbard chain energy per site versus the Lieb-Wu exact solution
   :width: 100%

Data: ``docs/hubbard_lieb_wu.csv``. The residual is finite-size extrapolation error, not
truncation: it shrinks with U because the chain is less correlated at strong coupling.

**The spin-1 Heisenberg chain and the Haldane phase.** The gapped Haldane ground state is
the standard hard case for a spin chain, and its energy density is known to nine digits.
By N = 96 the engine reproduces −1.401484039 J per site to about **2e-8**, and the string
order parameter saturates while the spin-spin correlation decays exponentially — the two
signatures that distinguish the Haldane phase from a trivial gapped one.

.. image:: ../spin1_haldane.svg
   :alt: Spin-1 Heisenberg energy per site converging to the Haldane value
   :width: 100%

.. image:: ../spin1_string_order.svg
   :alt: String order parameter saturating in the Haldane phase
   :width: 100%

Data: ``docs/spin1_haldane.csv``, ``docs/spin1_string_order.csv``.

**The J1-J2 chain's dimerisation transition.** At J2/J1 = 1/2 the Majumdar-Ghosh point has
an exactly known dimerised ground state. The measured dimerisation order parameter is
8.2e-6 per site at J2 = 0.55 and rises through the transition, so the engine resolves an
order parameter five orders of magnitude below the energy scale.

.. image:: ../j1j2_dimerization.svg
   :alt: J1-J2 dimerisation order parameter across the transition
   :width: 100%

Data: ``docs/j1j2_dimerization.csv``.

Fermionic VQE circuits
----------------------

The engine simulates Hamiltonian Variational Ansatz circuits **exactly**: every layer
conserves particle number and spin, so the U(1)×U(1) block structure is an advantage for
circuits rather than a restriction. At the sizes below χ is at the full Hilbert-space
bound, so there is no truncation anywhere in these numbers, and the circuit is pinned
against an independent dense state-vector simulator to 1e-12.

**What width costs a fermionic VQE.** Chain, ladder and cylinder at *the same* 12 qubits,
the same site-split representation, the same balanced cut and the same search protocol —
only the connectivity graph changes. At depth 8 the correlation energy captured falls
98.5 % → 95.7 % → 90.7 % as the lattice widens, and the ordering holds at every depth
from 2 to 12. The wider lattices get **more** variational parameters per layer (4 against
the chain's 3), so a worse energy with more freedom is the stronger form of the result.

.. image:: ../vqe_width.svg
   :alt: VQE energy error and entanglement entropy versus lattice width
   :width: 100%

Data: ``docs/vqe_width.csv``. The entropy half of this comparison does **not** order by
width — the figure's own caption says so, because it is generated from the data. At 12
qubits a cut that splits the qubits evenly slices through a ladder rung but falls on a
clean column boundary for the cylinder, so the two are not the same kind of cut. That
limitation is a property of the size, not of the measurement.

**Trainability versus width.** The same three geometries, now measuring the variance of a
cost gradient over random parameter draws — the standard barren-plateau diagnostic. The
cylinder's landscape is flatter than the chain's by a depth-independent factor: ×0.30–0.35
on the dimensionless cost and ×0.15–0.18 on the raw gradient, at every depth, with a
measured 2σ of ±0.05. The differentiated generator holds three bonds at every width and a
colouring group is a matching, so the comparison is not a normalisation artefact.

.. image:: ../vqe_barren.svg
   :alt: Gradient variance versus circuit depth for three lattice widths
   :width: 100%

Data: ``docs/vqe_barren.csv``, ``docs/vqe_barren_size.csv``. The instrument is calibrated
before it is used: Richardson extrapolation puts the finite-difference noise floor at
1.7e-9 and every plotted point sits at least 1.8e8 × above it, which is why the floor is
drawn on the same axes. This figure does **not** show a barren plateau — that term means
exponential decay in system size, and the accompanying size scan cannot establish one
either way. The documentation says why at length.

Finite temperature and dynamics
-------------------------------

Purification carries the engine to finite temperature, where it computes dynamical
response functions rather than just static energies: the dynamical spin structure factor
S(k, ω, T) and the optical conductivity σ(ω, T). Both are validated against exact
diagonalisation correlators before any spectrum is plotted.

.. image:: ../hubbard_spin_charge_skw.svg
   :alt: Spin and charge structure factors of the Hubbard chain
   :width: 100%

.. image:: ../conductivity_sigma_wT.svg
   :alt: Optical conductivity versus frequency and temperature
   :width: 100%

Data: ``docs/skw_hubbard_b2.0.csv``, ``docs/cond_hubbard_correlator.csv``.

Two-dimensional Hubbard physics
-------------------------------

On strips and cylinders the engine reproduces the stripe order that makes the doped
Hubbard model hard, and — with a diagonal hopping t′ switched on — lets stripe and pairing
tendencies be compared on the same footing.

.. image:: ../hubbard_stripes.svg
   :alt: Charge and spin stripes in the doped Hubbard model on a strip
   :width: 100%

.. image:: ../hubbard_stripe_duel.svg
   :alt: Stripe versus pairing tendency with diagonal hopping
   :width: 100%

Data: ``docs/hubbard_stripes.csv``, ``docs/hubbard_stripe_duel.csv``,
``docs/hubbard_ttprime.csv``.

Performance
-----------

The honest comparison is against a mature library on the same problem, not against the
engine's own CPU path. Against single-threaded TeNPy with MKL, on TEBD at N = 32,
min-of-2 runs:

.. list-table::
   :header-rows: 1

   * - χ
     - 256
     - 512
     - 1024
   * - QuantX (CUDA + OpenBLAS) vs TeNPy (MKL)
     - 0.91×
     - 1.12×
     - **2.17×** (2.74× with FP32)

.. image:: ../gpu_margin_vs_chi.svg
   :alt: GPU speed-up margin versus bond dimension
   :width: 100%

The margin grows with χ because that is where the SVD dominates: the bare GPU SVD kernel
is 45× the CPU one at χ = 1024. Below χ ≈ 512 the engine is at parity or slower, and that
is reported rather than omitted. At N = 64, χ = 1024 the figure is 2.22× (2.66× FP32),
consistent with the N = 32 measurement. Timings on this machine are thermally noisy; the
quoted numbers are minima over repeats.

How these numbers are kept honest
---------------------------------

Three review rounds on one pull request each found a figure in the prose that no longer
matched the data it came from — stale after a re-run, or mistyped. Remembering to check
did not work, so it is enforced instead:

* ``benchmarks/check_vqe_doc.py`` derives every quoted VQE figure from the committed CSVs
  and asserts the rendered string appears in the documentation. It runs in CI as
  ``qx_vqe_doc_figures``.
* It also asserts the *shape* of the data, not only the digits — that an ordering holds at
  every depth, that a decay rate is not constant, that a committed data row could have
  been produced by the documented command. When a re-run overturns a conclusion, the
  check demands the prose be rewritten rather than the numbers retuned.
* Figure captions are generated from the data files, so a figure states what its numbers
  support even where that contradicts the surrounding text. The width figure does exactly
  that.

The intended failure mode is a loud one. A claim that drifts from its data should break a
test, not survive into a release.
