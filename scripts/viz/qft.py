from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _phi4_dynamics_frames(
    n_sites: int = 4,
    d_local: int = 6,
    lam: float = 1.0,
    kappa: float = 1.0,
    m2_initial: float = 1.0,
    m2_final: float = -1.5,
    n_steps: int = 80,
    dt: float = 0.05,
    epsilon: float = 0.0,
) -> dict:
  """Compute φ⁴ quantum quench dynamics via exact time evolution (NumPy/SciPy reference)."""
  import numpy as np

  d = d_local
  N = n_sites

  # Local operators (same as in _phi4_frames)
  a_loc  = np.diag(np.sqrt(np.arange(1, d, dtype=float)), 1)
  adag   = a_loc.T
  phi_loc  = (a_loc + adag) / np.sqrt(2.0)
  pi2_loc  = np.real(-0.5 * (a_loc - adag) @ (a_loc - adag))
  phi2_loc = phi_loc @ phi_loc
  phi4_loc = phi2_loc @ phi2_loc
  n_op_loc = adag @ a_loc  # occupation number operator

  def kron_single(op, site):
    """Embed a d×d operator at `site` into the full d^N Hilbert space."""
    ops = [np.eye(d)] * N
    ops[site] = op
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  def kron_pair(op1, s1, op2, s2):
    """Embed op1⊗op2 at sites s1,s2 into the full space."""
    ops = [np.eye(d)] * N
    ops[s1] = op1
    ops[s2] = op2
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  def build_H(m2_val: float) -> "np.ndarray":
    dim = d ** N
    H = np.zeros((dim, dim), dtype=float)
    for i in range(N):
      H += 0.5 * kron_single(pi2_loc,  i)
      H += (0.5 * m2_val) * kron_single(phi2_loc, i)
      H += (0.25 * lam)   * kron_single(phi4_loc, i)
      H += epsilon        * kron_single(phi_loc,  i)
    for i in range(N - 1):
      H -= kappa * kron_pair(phi_loc, i, phi_loc, i + 1)
    return H

  def site_expect(psi, op, site):
    """Compute Re<psi|O_site|psi> without building the full kron."""
    # Reshape psi as d^site × d × d^(N-site-1)
    left  = d ** site
    right = d ** (N - site - 1)
    v = psi.reshape(left, d, right)
    Ov = np.tensordot(op, v, axes=([1], [1])).transpose(1, 0, 2)
    return float(np.real(np.vdot(psi, Ov.reshape(-1))))

  H_init  = build_H(m2_initial)
  H_final = build_H(m2_final)

  # Ground state of pre-quench Hamiltonian
  try:
    import numpy.linalg as nla
    vals_i, vecs_i = nla.eigh(H_init)
    psi0 = vecs_i[:, 0].copy()
  except Exception:
    psi0 = np.ones(d**N, dtype=float) / np.sqrt(d**N)

  # Post-quench gap for reference
  try:
    vals_f, _ = np.linalg.eigh(H_final)
    delta_E_final = float(vals_f[1] - vals_f[0]) if len(vals_f) > 1 else 0.0
    E0_initial = float(vals_i[0])
  except Exception:
    delta_E_final = 0.0
    E0_initial = 0.0

  # Time-evolution operator (try scipy, fall back to Taylor Krylov)
  try:
    from scipy.linalg import expm as _scipy_expm
    U_step = _scipy_expm(-1j * dt * H_final)
    def evolve(psi_cx):
      return U_step @ psi_cx
  except ImportError:
    # Fallback: small-m Arnoldi Krylov (m=15)
    def evolve(psi_cx):
      m = min(15, len(psi_cx))
      V = []
      b = float(np.linalg.norm(psi_cx))
      v = psi_cx / b
      H_m = np.zeros((m, m), dtype=complex)
      for j in range(m):
        V.append(v)
        w = H_final @ v
        for i in range(j + 1):
          H_m[i, j] = np.vdot(V[i], w)
          w = w - H_m[i, j] * V[i]
        nrm = float(np.linalg.norm(w))
        if nrm < 1e-14:
          m = j + 1
          break
        if j < m - 1:
          H_m[j + 1, j] = nrm
          v = w / nrm
      Vm = np.column_stack(V[:m])
      exp_Hm = np.array([[np.exp(-1j * dt * ev) for ev in np.linalg.eigvalsh(H_m[:m, :m])]])
      evals, evecs = np.linalg.eigh(H_m[:m, :m])
      exp_Hm_mat = evecs @ np.diag(np.exp(-1j * dt * evals)) @ evecs.conj().T
      e1 = np.zeros(m, dtype=complex); e1[0] = b
      phi_m = exp_Hm_mat @ e1
      return Vm @ phi_m

  def entropy_from_psi(psi_cx):
    dim_A = d ** (N // 2)
    dim_B = d ** (N - N // 2)
    M = psi_cx.reshape(dim_A, dim_B)
    sv = np.linalg.svd(M, compute_uv=False)
    probs = sv ** 2
    probs = probs[probs > 1e-14]
    return float(-np.sum(probs * np.log(probs))) if len(probs) > 0 else 0.0

  psi = psi0.astype(complex).copy()
  frames = []
  for step in range(n_steps + 1):
    t = step * dt
    phi_exp  = [site_expect(psi, phi_loc,  i) for i in range(N)]
    n_exp    = [site_expect(psi, n_op_loc, i) for i in range(N)]
    phi2_exp = [site_expect(psi, phi2_loc, i) for i in range(N)]
    order    = float(np.sqrt(max(0.0, sum(phi2_exp) / N)))
    S_ent    = entropy_from_psi(psi)
    Hpsi     = H_final @ psi
    energy   = float(np.real(np.vdot(psi, Hpsi)))
    frames.append({
        "time":        round(t, 6),
        "phi_expect":  [round(float(np.real(x)), 8) for x in phi_exp],
        "occupation":  [round(float(np.real(x)), 8) for x in n_exp],
        "order_param": round(order, 8),
        "entropy":     round(S_ent, 8),
        "energy":      round(energy, 8),
    })
    if step < n_steps:
      psi = evolve(psi)
      nrm = float(np.linalg.norm(psi))
      if nrm > 1e-14:
        psi /= nrm

  return {
      "n_sites":       N,
      "d_local":       d,
      "m2_initial":    m2_initial,
      "m2_final":      m2_final,
      "lam":           lam,
      "kappa":         kappa,
      "dt":            dt,
      "n_steps":       n_steps,
      "E0_initial":    round(E0_initial, 8),
      "delta_E_final": round(delta_E_final, 8),
      "frames":        frames,
  }


def _phi4_frames(
    n_sites: int = 4,
    m2: float = 1.0,
    lam: float = 1.0,
    kappa: float = 1.0,
    d_local: int = 10,
    scenario: str = "symmetric",
    n_sweep: int = 25,
    m2_min: float = -3.0,
    m2_max: float = 2.5,
    epsilon: float = 0.05,
    seed: int = 42,
) -> dict:
  """φ⁴ scalar QFT on a 1D lattice via truncated Fock space exact diagonalization.

  Hamiltonian (natural units, lattice spacing a=1):

    H = Σᵢ [½πᵢ² + ½m²φᵢ² + (λ/4)φᵢ⁴] − κ Σ⟨ij⟩ φᵢφⱼ

  Local operators in truncated oscillator basis (d_local levels):
    φᵢ = (âᵢ + â†ᵢ)/√2,   πᵢ = i(â†ᵢ − âᵢ)/√2

  A small explicit symmetry-breaking field ε·Σφᵢ is added to select a vacuum
  in the broken phase (m² < 0), making ⟨φ⟩ non-zero there.

  Scenarios:
    "symmetric"  — m² > 0 ground state + spectral gap
    "broken"     — m² < 0 ground state showing SSB
    "sweep"      — scan m² from m2_max → m2_min tracking order parameter + gap
  """
  try:
    import numpy as np
    from numpy.linalg import eigh
  except ImportError as ex:
    raise RuntimeError("phi4 visualization requires numpy") from ex

  N = n_sites
  d = d_local

  # ── Local Fock-space operators ─────────────────────────────────────────────
  # â: lowering, (â)_{n-1,n} = √n
  a_loc = np.diag(np.sqrt(np.arange(1, d, dtype=float)), 1)   # d×d, upper triangular
  adag  = a_loc.T
  phi_loc  = (a_loc + adag) / np.sqrt(2.0)                   # position-like field
  pi_loc   = 1j * (adag - a_loc) / np.sqrt(2.0)              # momentum-like field
  pi2_loc  = np.real(-0.5 * (a_loc - adag) @ (a_loc - adag)) # ½π² per site
  phi2_loc = phi_loc @ phi_loc
  phi4_loc = phi2_loc @ phi2_loc
  n_op_loc = np.diag(np.arange(d, dtype=float))               # number operator N̂

  I_loc = np.eye(d, dtype=float)

  # ── Tensor product helpers ─────────────────────────────────────────────────
  def kron_single(op: "np.ndarray", site: int) -> "np.ndarray":
    """I ⊗ ... ⊗ op (at site) ⊗ ... ⊗ I, total N sites."""
    ops = [I_loc] * N
    ops[site] = op
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  def kron_pair(op1: "np.ndarray", s1: int, op2: "np.ndarray", s2: int) -> "np.ndarray":
    ops = [I_loc] * N
    ops[s1] = op1
    ops[s2] = op2
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  # ── Hamiltonian builder ────────────────────────────────────────────────────
  def build_H(m2_val: float, eps: float = epsilon) -> "np.ndarray":
    dim = d ** N
    H = np.zeros((dim, dim), dtype=float)
    for i in range(N):
      H += 0.5 * kron_single(pi2_loc,      i)         # ½πᵢ²
      H += (0.5 * m2_val) * kron_single(phi2_loc,  i)   # ½m²φᵢ²
      H += (0.25 * lam)   * kron_single(phi4_loc,  i)   # λ/4 φᵢ⁴
      H += eps             * kron_single(phi_loc,   i)   # explicit Z₂ breaking
    for i in range(N - 1):
      H -= kappa * kron_pair(phi_loc, i, phi_loc, i + 1)  # −κ φᵢφᵢ₊₁
    return H

  # ── Solve: get lowest 3 eigenpairs ────────────────────────────────────────
  def solve(m2_val: float, eps: float = epsilon) -> "tuple":
    H = build_H(m2_val, eps)
    # eigh returns eigenvalues in ascending order.
    vals, vecs = eigh(H)
    return vals[:3], vecs[:, :3]   # (energies, states)

  # ── Observables from a state vector ───────────────────────────────────────
  def site_expect(psi: "np.ndarray", op_loc: "np.ndarray", site: int) -> float:
    """⟨ψ|op_site|ψ⟩ using tensor structure (avoid full kron for speed)."""
    d_n = d ** N
    d_before = d ** site
    d_after  = d ** (N - site - 1)
    psi_r = psi.reshape(d_before, d, d_after)
    # Contract op over the site index.
    psi_op = np.einsum("ij,ajb->aib", op_loc, psi_r)
    return float(np.real(np.dot(psi.ravel(), psi_op.ravel())))

  def all_site_obs(psi: "np.ndarray") -> dict:
    phi_exp  = [site_expect(psi, phi_loc,  i) for i in range(N)]
    phi2_exp = [site_expect(psi, phi2_loc, i) for i in range(N)]
    n_exp    = [site_expect(psi, n_op_loc, i) for i in range(N)]
    op_val   = float(np.sqrt(max(0.0, sum(phi2_exp) / N)))
    return {
        "phi_expectation":  phi_exp,
        "phi2_expectation": phi2_exp,
        "occupation":       n_exp,
        "order_parameter":  op_val,
        "phi_mean":         float(np.mean(phi_exp)),
    }

  # ── Marginal field distribution p(φ₀) at site 0 ───────────────────────────
  def field_distribution(psi: "np.ndarray", n_bins: int = 60) -> "tuple":
    """Return (phi_vals, probs) for the marginal distribution at site 0."""
    # psi has shape (d^N,). Reshape to (d, d^{N-1}).
    d_rest = d ** (N - 1)
    psi_r = psi.reshape(d, d_rest)  # index 0 = site 0 Fock level
    # Marginal amplitudes in the φ basis: ψ_φ(φ₀) = Σ_{n₀} ψ_{n₀} φ_{n₀}(φ₀).
    # Use a discrete grid for φ and evaluate in the harmonic oscillator basis.
    phi_max = 4.0 * np.sqrt(float(d))
    phi_vals = np.linspace(-phi_max, phi_max, n_bins)
    # Harmonic oscillator wave functions ψ_n(φ) = H_n(φ) e^{-φ²/2} / normalization.
    # We use the recurrence relation for Hermite polynomials.
    # ψ_0(φ) = π^{-1/4} exp(-φ²/2)
    # ψ_n(φ) = (a†)^n / √(n!) ψ_0
    # Numerically stable via recurrence: ψ_n = (√2 φ ψ_{n-1} - √(n-1) ψ_{n-2}) / √n
    psi_phi = np.zeros((d, n_bins), dtype=float)
    psi_phi[0] = np.pi ** (-0.25) * np.exp(-0.5 * phi_vals ** 2)
    if d > 1:
      psi_phi[1] = np.sqrt(2.0) * phi_vals * psi_phi[0]
    for n in range(2, d):
      psi_phi[n] = (np.sqrt(2.0) * phi_vals * psi_phi[n-1] -
                    np.sqrt(float(n-1)) * psi_phi[n-2]) / np.sqrt(float(n))
    # Marginal density: ρ(φ₀) = Σ_{rest} |Σ_n0 ψ_{n₀,rest} ψ_{HO,n₀}(φ₀)|²
    # = |Σ_n0 c_{n0} ψ_n(φ₀)|² where c_{n₀} = Σ_{rest} ψ_{n₀,rest}
    # Reduce over rest sites: c_{n₀} = √(sum_{rest} |ψ_{n₀,rest}|²) × phase
    # For probability: ρ(φ₀) = Σ_n₀ Σ_m₀ (ρ_{n₀m₀}) ψ_n(φ₀) ψ_m(φ₀)
    # where ρ_{n₀m₀} = Σ_{rest} ψ*_{n₀,rest} ψ_{m₀,rest}  (reduced density matrix)
    rho_local = psi_r @ psi_r.T   # (d, d) reduced density matrix (real for real psi)
    prob = np.einsum("ij,ix,jx->x", rho_local, psi_phi, psi_phi)
    prob = np.real(prob)
    prob = np.clip(prob, 0.0, None)
    # Normalise (approximate: discrete grid, may not integrate to 1 perfectly).
    dv = phi_vals[1] - phi_vals[0]
    prob /= max(prob.sum() * dv, 1e-12)
    return phi_vals.tolist(), prob.tolist()

  # ── Main computation ───────────────────────────────────────────────────────
  vals, vecs = solve(m2)
  E0, E1, E2 = float(vals[0]), float(vals[1]), float(vals[2])
  psi0 = vecs[:, 0]
  obs  = all_site_obs(psi0)
  phi_vals, probs = field_distribution(psi0)

  # ── Phase transition sweep ─────────────────────────────────────────────────
  sweep = []
  if scenario == "sweep" and n_sweep > 0:
    for i in range(n_sweep):
      m2_i = m2_max - (m2_max - m2_min) * i / max(1, n_sweep - 1)
      v_i, vec_i = solve(m2_i)
      psi_i = vec_i[:, 0]
      obs_i = all_site_obs(psi_i)
      sweep.append({
          "m2":             m2_i,
          "ground_energy":  float(v_i[0]),
          "order_parameter": obs_i["order_parameter"],
          "energy_gap":     float(v_i[1] - v_i[0]),
          "phi_mean":       obs_i["phi_mean"],
      })

  # ── Classical potential for visualisation ─────────────────────────────────
  phi_cl = np.linspace(-4.0, 4.0, 200)
  V_cl   = (0.5 * m2 * phi_cl ** 2 + 0.25 * lam * phi_cl ** 4).tolist()
  phi_cl = phi_cl.tolist()

  return {
      "n_sites":           N,
      "d_local":           d,
      "m2":                m2,
      "lam":               lam,
      "kappa":             kappa,
      "scenario":          scenario,
      "ground_energy":     E0,
      "first_excited":     E1,
      "second_excited":    E2,
      "energy_gap":        E1 - E0,
      "order_parameter":   obs["order_parameter"],
      "phi_expectation":   obs["phi_expectation"],
      "phi2_expectation":  obs["phi2_expectation"],
      "occupation":        obs["occupation"],
      "phi_mean":          obs["phi_mean"],
      "field_phi_vals":    phi_vals,
      "field_probs":       probs,
      "classical_phi":     phi_cl,
      "classical_V":       V_cl,
      "sweep":             sweep,
  }


def _phi4_page(data: dict, title: str = "φ⁴ QFT") -> str:
  """Generate a self-contained HTML page for the φ⁴ QFT results."""
  import json as _json

  data_js = _json.dumps(data, ensure_ascii=False)
  m2_val  = data.get("m2", 1.0)
  scenario = data.get("scenario", "symmetric")

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --accent:#58a6ff;
           --green:#3fb950; --orange:#d29922; --red:#f85149; --purple:#bc8cff;
           --yellow:#e3b341; --muted:#8b949e; --text:#e6edf3; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
          min-height:100vh; padding:24px; }}
  h1 {{ font-size:1.4rem; font-weight:600; color:var(--accent); margin-bottom:4px; }}
  .subtitle {{ color:var(--muted); font-size:.85rem; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; max-width:1100px; margin:0 auto; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; }}
  .panel.wide {{ grid-column:span 2; }}
  .panel h2 {{ font-size:.95rem; color:var(--muted); margin-bottom:12px; font-weight:500; }}
  canvas {{ display:block; width:100%; }}
  .stat-row {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px; max-width:1100px;
               margin-left:auto; margin-right:auto; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:8px 14px; }}
  .stat-label {{ font-size:.75rem; color:var(--muted); margin-bottom:2px; }}
  .stat-value {{ font-size:1.05rem; font-weight:600; color:var(--accent); }}
  .tag {{ display:inline-block; font-size:.75rem; padding:2px 8px; border-radius:4px;
          background:#21262d; border:1px solid var(--border); color:var(--muted);
          margin-right:6px; margin-bottom:12px; }}
  .phase-badge {{ display:inline-block; font-size:.8rem; padding:3px 10px; border-radius:12px;
                  font-weight:600; margin-bottom:16px; }}
  .phase-sym {{ background:rgba(88,166,255,.15); border:1px solid #58a6ff; color:#58a6ff; }}
  .phase-brk {{ background:rgba(240,81,73,.15); border:1px solid #f85149; color:#f85149; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">φ⁴ scalar field theory on a 1D lattice · truncated Fock space · exact diagonalization</p>
<div>
  <span class="tag">N={data.get('n_sites',4)} sites</span>
  <span class="tag">d={data.get('d_local',10)} levels</span>
  <span class="tag">m²={data.get('m2',1.0)}</span>
  <span class="tag">λ={data.get('lam',1.0)}</span>
  <span class="tag">κ={data.get('kappa',1.0)}</span>
</div>
<div id="phaseBadge"></div>

<div class="stat-row" id="stats"></div>

<div class="grid">
  <div class="panel">
    <h2>Classical Potential V(φ) = ½m²φ² + ¼λφ⁴</h2>
    <canvas id="cvPot" height="230"></canvas>
  </div>
  <div class="panel">
    <h2>Ground State Field Distribution p(φ₀)</h2>
    <canvas id="cvDist" height="230"></canvas>
  </div>
  <div class="panel">
    <h2>Site Profile: ⟨φᵢ⟩ and √⟨φᵢ²⟩ per Lattice Site</h2>
    <canvas id="cvSite" height="230"></canvas>
  </div>
  <div class="panel">
    <h2>Mean Occupation ⟨nᵢ⟩ per Site (quantum fluctuations)</h2>
    <canvas id="cvOcc" height="230"></canvas>
  </div>
  <div class="panel wide" id="sweepPanel" style="display:none">
    <h2>Phase Transition: Order Parameter √⟨φ²⟩ and Energy Gap Δ vs m²</h2>
    <canvas id="cvSweep" height="240"></canvas>
  </div>
</div>

<script>
const DATA = {data_js};

// ── Stats ─────────────────────────────────────────────────────────────────────
(function() {{
  const s = document.getElementById('stats');
  function stat(label, val, color) {{
    const d = document.createElement('div');
    d.className = 'stat';
    d.innerHTML = `<div class="stat-label">${{label}}</div>
      <div class="stat-value" style="${{color ? 'color:'+color : ''}}">${{val}}</div>`;
    s.appendChild(d);
  }}
  stat('Ground energy E₀', DATA.ground_energy.toFixed(5));
  stat('First excited E₁', DATA.first_excited.toFixed(5));
  stat('Energy gap Δ', DATA.energy_gap.toFixed(5),
       DATA.energy_gap < 0.05 ? '#f85149' : '#3fb950');
  stat('Order parameter √⟨φ²⟩', DATA.order_parameter.toFixed(5),
       DATA.order_parameter > 0.3 ? '#f85149' : '#58a6ff');
  stat('⟨φ⟩ (mean)', DATA.phi_mean.toFixed(5));

  // Phase badge.
  const badge = document.getElementById('phaseBadge');
  const broken = DATA.order_parameter > 0.3 && DATA.m2 < 0;
  badge.innerHTML = broken
    ? '<span class="phase-badge phase-brk">⚠ Broken Phase — Z₂ symmetry spontaneously broken</span>'
    : '<span class="phase-badge phase-sym">✓ Symmetric Phase — Z₂ preserved</span>';
}})();

// ── Canvas helpers ─────────────────────────────────────────────────────────────
function setup(id, h) {{
  const cv = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  cv.width  = (cv.offsetWidth  || 500) * dpr;
  cv.height = (cv.offsetHeight || h  ) * dpr;
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.fillStyle = '#161b22'; ctx.fillRect(0, 0, cv.offsetWidth||500, cv.offsetHeight||h);
  return {{ctx, W: cv.offsetWidth||500, H: cv.offsetHeight||h}};
}}

function axes(ctx, pad, W, H) {{
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H-pad.b);
  ctx.lineTo(W-pad.r, H-pad.b); ctx.stroke();
}}

function grid(ctx, pad, W, H) {{
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {{
    const y = pad.t + (H-pad.t-pad.b) * i / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W-pad.r, y); ctx.stroke();
  }}
}}

// ── Classical potential ────────────────────────────────────────────────────────
function drawPot() {{
  const {{ctx, W, H}} = setup('cvPot', 230);
  const pad = {{l:52, r:12, t:16, b:28}};
  const xs  = DATA.classical_phi;
  const ys  = DATA.classical_V;
  if (!xs || !xs.length) return;
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  // Clip to show interesting part of potential.
  const ymax_vis = Math.min(Math.max(...ys), 8.0);
  const ymin_vis = Math.min(...ys.filter(y=>y<=ymax_vis), 0);
  function xOf(v) {{ return pad.l + (v - xs[0]) / (xs[xs.length-1]-xs[0]) * cw; }}
  function yOf(v) {{ return H - pad.b - (Math.min(v,ymax_vis) - ymin_vis) / (ymax_vis - ymin_vis + 1e-9) * ch; }}

  // Fill under curve.
  ctx.fillStyle = 'rgba(88,166,255,0.07)';
  ctx.beginPath();
  ctx.moveTo(xOf(xs[0]), H - pad.b);
  for (let i = 0; i < xs.length; i++) ctx.lineTo(xOf(xs[i]), yOf(ys[i]));
  ctx.lineTo(xOf(xs[xs.length-1]), H - pad.b); ctx.closePath(); ctx.fill();

  // Curve.
  ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 2; ctx.beginPath();
  let first = true;
  for (let i = 0; i < xs.length; i++) {{
    const y = ys[i];
    if (y > ymax_vis + 0.5) {{ first = true; continue; }}
    if (first) {{ ctx.moveTo(xOf(xs[i]), yOf(y)); first = false; }}
    else ctx.lineTo(xOf(xs[i]), yOf(y));
  }}
  ctx.stroke();

  // Zero line.
  const y0 = yOf(0);
  if (y0 >= pad.t && y0 <= H - pad.b) {{
    ctx.strokeStyle = '#30363d'; ctx.lineWidth = 0.8;
    ctx.beginPath(); ctx.moveTo(pad.l, y0); ctx.lineTo(W-pad.r, y0); ctx.stroke();
  }}

  // Minima markers.
  const m2 = DATA.m2, lam = DATA.lam;
  if (m2 < 0 && lam > 0) {{
    const phi_min = Math.sqrt(-m2 / (2*lam));  // classical minimum ±φ* = ±√(-m²/2λ)
    [-phi_min, phi_min].forEach(pm => {{
      if (Math.abs(pm) <= Math.abs(xs[xs.length-1])) {{
        const V_min = 0.5*m2*pm*pm + 0.25*lam*pm*pm*pm*pm;
        ctx.fillStyle = '#f85149';
        ctx.beginPath(); ctx.arc(xOf(pm), yOf(V_min), 5, 0, 2*Math.PI); ctx.fill();
      }}
    }});
    ctx.fillStyle = '#f85149'; ctx.font = '10px system-ui'; ctx.textAlign = 'left';
    ctx.fillText('±φ* = ±' + phi_min.toFixed(2), pad.l+4, pad.t+12);
  }}

  axes(ctx, pad, W, H);
  // x axis labels.
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  [-3,-2,-1,0,1,2,3].forEach(v => {{
    if (v >= xs[0] && v <= xs[xs.length-1])
      ctx.fillText(v, xOf(v), H-pad.b+14);
  }});
  ctx.fillText('φ', pad.l+cw/2, H-4);
  // y labels.
  ctx.textAlign = 'right';
  [ymin_vis, (ymin_vis+ymax_vis)/2, ymax_vis].forEach(v => {{
    ctx.fillText(v.toFixed(2), pad.l-4, yOf(v)+3);
  }});
}}

// ── Field distribution p(φ₀) ──────────────────────────────────────────────────
function drawDist() {{
  const {{ctx, W, H}} = setup('cvDist', 230);
  const pad = {{l:52, r:12, t:16, b:28}};
  const xs = DATA.field_phi_vals;
  const ps = DATA.field_probs;
  if (!xs || !xs.length) return;
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  const maxP = Math.max(...ps, 1e-9);
  function xOf(v) {{ return pad.l + (v-xs[0])/(xs[xs.length-1]-xs[0])*cw; }}
  function yOf(p) {{ return H - pad.b - (p/maxP)*ch; }}

  // Fill.
  ctx.fillStyle = 'rgba(188,140,255,0.15)';
  ctx.beginPath(); ctx.moveTo(xOf(xs[0]), H-pad.b);
  for (let i=0;i<xs.length;i++) ctx.lineTo(xOf(xs[i]), yOf(ps[i]));
  ctx.lineTo(xOf(xs[xs.length-1]), H-pad.b); ctx.closePath(); ctx.fill();

  // Line.
  ctx.strokeStyle = '#bc8cff'; ctx.lineWidth = 2; ctx.beginPath();
  for (let i=0;i<xs.length;i++) {{
    if(i===0) ctx.moveTo(xOf(xs[i]),yOf(ps[i]));
    else ctx.lineTo(xOf(xs[i]),yOf(ps[i]));
  }}
  ctx.stroke();

  // Classical minima.
  const m2=DATA.m2, lam=DATA.lam;
  if (m2<0 && lam>0) {{
    const pm=Math.sqrt(-m2/(2*lam));
    [pm,-pm].forEach(v => {{
      if (v>=xs[0]&&v<=xs[xs.length-1]) {{
        ctx.strokeStyle='#f85149'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
        ctx.beginPath(); ctx.moveTo(xOf(v),pad.t); ctx.lineTo(xOf(v),H-pad.b); ctx.stroke();
        ctx.setLineDash([]);
      }}
    }});
  }}

  axes(ctx, pad, W, H);
  ctx.fillStyle='#8b949e'; ctx.font='10px system-ui'; ctx.textAlign='center';
  [-3,-2,-1,0,1,2,3].forEach(v => {{
    if(v>=xs[0]&&v<=xs[xs.length-1]) ctx.fillText(v, xOf(v), H-pad.b+14);
  }});
  ctx.fillText('φ₀', pad.l+cw/2, H-4);
  ctx.textAlign='right';
  [0, maxP/2, maxP].forEach(v => ctx.fillText(v.toFixed(3), pad.l-4, yOf(v)+3));
  ctx.fillStyle='#bc8cff'; ctx.textAlign='left';
  ctx.fillText('p(φ₀) = ⟨φ₀|ρ_local|φ₀⟩', pad.l+4, pad.t+12);
}}

// ── Site profile ───────────────────────────────────────────────────────────────
function drawSite() {{
  const {{ctx, W, H}} = setup('cvSite', 230);
  const pad = {{l:52, r:12, t:16, b:28}};
  const phi  = DATA.phi_expectation  || [];
  const phi2 = DATA.phi2_expectation || [];
  const N = phi.length;
  if (!N) return;
  const cw = W-pad.l-pad.r, ch = H-pad.t-pad.b;
  const maxV = Math.max(...phi.map(Math.abs), ...phi2.map(Math.sqrt), 0.05);
  function xOf(i) {{ return pad.l + (i+0.5)*(cw/N); }}
  function yOf(v) {{ return H-pad.b-(v+maxV)/(2*maxV)*ch; }}

  // Zero line.
  ctx.strokeStyle='#30363d'; ctx.lineWidth=0.8;
  ctx.beginPath(); ctx.moveTo(pad.l,yOf(0)); ctx.lineTo(W-pad.r,yOf(0)); ctx.stroke();

  axes(ctx, pad, W, H); grid(ctx, pad, W, H);

  const bw = Math.max(4, cw/N*0.35);
  for (let i=0;i<N;i++) {{
    const x=xOf(i);
    // √⟨φ²⟩ bar (background).
    const sq=Math.sqrt(phi2[i]||0);
    ctx.fillStyle='rgba(88,166,255,0.3)';
    const bh_sq=sq/(2*maxV)*ch;
    ctx.fillRect(x-bw*0.8, H-pad.b-bh_sq, bw*1.6, bh_sq);
    // ⟨φ⟩ bar (foreground).
    const v=phi[i]||0;
    ctx.fillStyle=v>=0?'#3fb950':'#f85149';
    const bh=Math.abs(v)/(2*maxV)*ch;
    ctx.fillRect(x-bw/2, v>=0?H-pad.b-bh:H-pad.b, bw, bh);
    // Label.
    ctx.fillStyle='#8b949e'; ctx.font='9px system-ui'; ctx.textAlign='center';
    ctx.fillText(i, x, H-pad.b+12);
  }}
  ctx.textAlign='right'; ctx.fillStyle='#8b949e'; ctx.font='10px system-ui';
  [-maxV,0,maxV].forEach(v => ctx.fillText(v.toFixed(2), pad.l-4, yOf(v)+3));
  ctx.textAlign='left';
  ctx.fillStyle='rgba(88,166,255,0.6)'; ctx.fillRect(pad.l+4, pad.t+2, 10, 8);
  ctx.fillStyle='#8b949e'; ctx.fillText(' √⟨φ²⟩', pad.l+16, pad.t+10);
  ctx.fillStyle='#3fb950'; ctx.fillRect(pad.l+4, pad.t+14, 10, 8);
  ctx.fillText(' ⟨φ⟩', pad.l+16, pad.t+22);
}}

// ── Occupation ─────────────────────────────────────────────────────────────────
function drawOcc() {{
  const {{ctx, W, H}} = setup('cvOcc', 230);
  const pad = {{l:52, r:12, t:16, b:28}};
  const occ = DATA.occupation || [];
  const N = occ.length;
  if (!N) return;
  const cw = W-pad.l-pad.r, ch = H-pad.t-pad.b;
  const maxV = Math.max(...occ, 0.1);
  function xOf(i) {{ return pad.l+(i+0.5)*(cw/N); }}
  function yOf(v) {{ return H-pad.b-v/maxV*ch; }}
  axes(ctx, pad, W, H); grid(ctx, pad, W, H);
  const bw = Math.max(4, cw/N*0.5);
  for (let i=0;i<N;i++) {{
    const x=xOf(i);
    const bh=(occ[i]||0)/maxV*ch;
    ctx.fillStyle='#d29922';
    ctx.fillRect(x-bw/2, H-pad.b-bh, bw, bh);
    ctx.fillStyle='#8b949e'; ctx.font='9px system-ui'; ctx.textAlign='center';
    ctx.fillText(i, x, H-pad.b+12);
    ctx.fillStyle='#e6edf3'; ctx.font='9px system-ui';
    ctx.fillText((occ[i]||0).toFixed(2), x, H-pad.b-bh-4);
  }}
  ctx.textAlign='right'; ctx.fillStyle='#8b949e'; ctx.font='10px system-ui';
  [0,maxV/2,maxV].forEach(v=>ctx.fillText(v.toFixed(2),pad.l-4,yOf(v)+3));
  ctx.fillStyle='#d29922'; ctx.textAlign='left'; ctx.font='10px system-ui';
  ctx.fillText('⟨n̂ᵢ⟩ (quanta per site)', pad.l+4, pad.t+12);
}}

// ── Phase transition sweep ─────────────────────────────────────────────────────
function drawSweep() {{
  const sweep = DATA.sweep || [];
  if (!sweep.length) return;
  document.getElementById('sweepPanel').style.display='';
  const {{ctx, W, H}} = setup('cvSweep', 240);
  const pad = {{l:56, r:60, t:20, b:36}};
  const m2s = sweep.map(s=>s.m2);
  const ops  = sweep.map(s=>s.order_parameter);
  const gaps = sweep.map(s=>s.energy_gap);
  const minM=Math.min(...m2s), maxM=Math.max(...m2s);
  const maxOP=Math.max(...ops, 0.01), maxGap=Math.max(...gaps, 0.01);
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  function xOf(m) {{ return pad.l+(m-minM)/(maxM-minM+1e-9)*cw; }}
  function yOfOP(v) {{ return H-pad.b-v/maxOP*ch; }}
  function yOfG(v)  {{ return H-pad.b-v/maxGap*ch; }}

  axes(ctx, pad, W, H); grid(ctx, pad, W, H);

  // Vertical line at m²=0.
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(xOf(0),pad.t); ctx.lineTo(xOf(0),H-pad.b); ctx.stroke();

  // Order parameter (left axis, blue).
  ctx.strokeStyle='#58a6ff'; ctx.lineWidth=2.5; ctx.beginPath();
  for (let i=0;i<m2s.length;i++) {{
    if(i===0) ctx.moveTo(xOf(m2s[i]),yOfOP(ops[i]));
    else ctx.lineTo(xOf(m2s[i]),yOfOP(ops[i]));
  }}
  ctx.stroke();
  // Dots.
  ops.forEach((op,i) => {{
    ctx.fillStyle='#58a6ff';
    ctx.beginPath(); ctx.arc(xOf(m2s[i]),yOfOP(op),3,0,2*Math.PI); ctx.fill();
  }});

  // Energy gap (right axis, orange).
  ctx.strokeStyle='#d29922'; ctx.lineWidth=2; ctx.beginPath();
  for (let i=0;i<m2s.length;i++) {{
    if(i===0) ctx.moveTo(xOf(m2s[i]),yOfG(gaps[i]));
    else ctx.lineTo(xOf(m2s[i]),yOfG(gaps[i]));
  }}
  ctx.stroke();

  // x axis labels.
  ctx.fillStyle='#8b949e'; ctx.font='10px system-ui'; ctx.textAlign='center';
  [-3,-2,-1,0,1,2].forEach(v => {{
    if(v>=minM&&v<=maxM) ctx.fillText(v.toFixed(0),xOf(v),H-pad.b+14);
  }});
  ctx.fillText('bare mass squared m²', pad.l+cw/2, H-4);

  // Left y labels (order parameter).
  ctx.fillStyle='#58a6ff'; ctx.textAlign='right';
  [0,maxOP/2,maxOP].forEach(v=>ctx.fillText(v.toFixed(2),pad.l-4,yOfOP(v)+3));
  ctx.save(); ctx.translate(14,pad.t+ch/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign='center'; ctx.fillText('√⟨φ²⟩ (order param)', 0, 0); ctx.restore();

  // Right y labels (gap).
  ctx.fillStyle='#d29922'; ctx.textAlign='left';
  [0,maxGap/2,maxGap].forEach(v=>ctx.fillText(v.toFixed(2),W-pad.r+4,yOfG(v)+3));
  ctx.save(); ctx.translate(W-12,pad.t+ch/2); ctx.rotate(Math.PI/2);
  ctx.textAlign='center'; ctx.fillText('gap Δ=E₁−E₀', 0, 0); ctx.restore();

  // Legend.
  ctx.textAlign='left'; ctx.font='11px system-ui';
  ctx.fillStyle='#58a6ff'; ctx.fillText('order param √⟨φ²⟩', pad.l+8, pad.t+14);
  ctx.fillStyle='#d29922'; ctx.fillText('energy gap Δ', pad.l+8, pad.t+28);
  ctx.fillStyle='#8b949e'; ctx.font='9px system-ui';
  ctx.fillText('← symmetric', xOf(0)+4, pad.t+14);
  ctx.textAlign='right'; ctx.fillText('broken →', xOf(0)-4, pad.t+14);
}}

window.addEventListener('load', () => {{
  drawPot(); drawDist(); drawSite(); drawOcc(); drawSweep();
}});
window.addEventListener('resize', () => {{
  drawPot(); drawDist(); drawSite(); drawOcc(); drawSweep();
}});
</script>
</body>
</html>"""
  return html


def _phi4_dynamics_page(data: dict, title: str = "φ⁴ QFT — Quantum Quench Dynamics") -> str:
  """Generate a self-contained HTML page for the φ⁴ quantum quench dynamics."""
  import json as _json

  data_js = _json.dumps(data, ensure_ascii=False)
  N        = data.get("n_sites", 4)
  d        = data.get("d_local", 6)
  m2i      = data.get("m2_initial", 1.0)
  m2f      = data.get("m2_final", -1.5)
  n_steps  = data.get("n_steps", 80)
  dt_val   = data.get("dt", 0.05)
  frames   = data.get("frames", [])
  final_op = frames[-1]["order_param"] if frames else 0.0
  final_S  = frames[-1]["entropy"]     if frames else 0.0

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --accent:#36cfc9;
           --green:#3fb950; --orange:#d29922; --red:#f85149; --muted:#8b949e; --text:#e6edf3; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
          min-height:100vh; padding:24px; }}
  h1 {{ font-size:1.4rem; font-weight:600; color:var(--accent); margin-bottom:4px; }}
  .subtitle {{ color:var(--muted); font-size:.85rem; margin-bottom:16px; }}
  .tags {{ margin-bottom:16px; }}
  .tag {{ display:inline-block; font-size:.75rem; padding:2px 8px; border-radius:4px;
          background:#21262d; border:1px solid var(--border); color:var(--muted);
          margin-right:6px; margin-bottom:4px; }}
  .stat-row {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; max-width:1100px;
               margin-left:auto; margin-right:auto; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:8px 14px; }}
  .stat-label {{ font-size:.75rem; color:var(--muted); margin-bottom:2px; }}
  .stat-value {{ font-size:1.05rem; font-weight:600; color:var(--accent); }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; max-width:1100px; margin:0 auto; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; }}
  .panel h2 {{ font-size:.92rem; color:var(--muted); margin-bottom:10px; font-weight:500; }}
  canvas {{ display:block; width:100%; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">φ⁴ scalar QFT — quantum quench dynamics · exact time evolution · truncated Fock space</p>
<div class="tags">
  <span class="tag">N={N} sites</span>
  <span class="tag">d={d} levels</span>
  <span class="tag">m²_initial={m2i}</span>
  <span class="tag">m²_final={m2f}</span>
  <span class="tag">n_steps={n_steps}</span>
  <span class="tag">dt={dt_val}</span>
</div>
<div class="stat-row">
  <div class="stat"><div class="stat-label">final order param √⟨φ²⟩</div>
    <div class="stat-value">{final_op:.6f}</div></div>
  <div class="stat"><div class="stat-label">final entanglement S</div>
    <div class="stat-value">{final_S:.6f}</div></div>
  <div class="stat"><div class="stat-label">quench direction</div>
    <div class="stat-value">{"sym→broken" if m2f < 0 else "broken→sym"}</div></div>
</div>
<div class="grid">
  <div class="panel"><h2>Order Parameter √⟨φ²⟩(t)</h2><canvas id="cvOrder" height="220"></canvas></div>
  <div class="panel"><h2>Entanglement Entropy S(t)</h2><canvas id="cvEntropy" height="220"></canvas></div>
  <div class="panel"><h2>⟨φᵢ(t)⟩ Heatmap (blue=−, white=0, red=+)</h2><canvas id="cvPhiHeat" height="220"></canvas></div>
  <div class="panel"><h2>⟨nᵢ(t)⟩ Heatmap (white=0, orange=high)</h2><canvas id="cvOccHeat" height="220"></canvas></div>
</div>
<script>
const DATA = {data_js};

function setup(id, h) {{
  const cv = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  const W = cv.parentElement.clientWidth - 32;
  const H = h;
  cv.width  = W * dpr;
  cv.height = H * dpr;
  cv.style.width  = W + 'px';
  cv.style.height = H + 'px';
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  return {{ctx, W, H}};
}}

function axes(ctx, pad, W, H) {{
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H-pad.b);
  ctx.lineTo(W-pad.r, H-pad.b);
  ctx.stroke();
}}

function grid(ctx, pad, W, H, nv, nh) {{
  ctx.strokeStyle='#21262d'; ctx.lineWidth=0.5;
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  for (let i=1; i<=(nh||4); i++) {{
    const y=pad.t+i*ch/(nh||4);
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(W-pad.r,y); ctx.stroke();
  }}
  for (let i=1; i<=(nv||5); i++) {{
    const x=pad.l+i*cw/(nv||5);
    ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,H-pad.b); ctx.stroke();
  }}
}}

function drawLineChart(id, h, ys, color, labelY, labelX) {{
  const {{ctx, W, H}} = setup(id, h);
  const pad = {{l:52, r:12, t:16, b:32}};
  if (!ys.length) return;
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const n=ys.length;
  const minY=Math.min(...ys), maxY=Math.max(...ys);
  const rng=Math.max(maxY-minY, 1e-9);
  const pad_y=rng*0.08;
  const lo=minY-pad_y, hi=maxY+pad_y;
  function xOf(i) {{ return pad.l+i/(n-1||1)*cw; }}
  function yOf(v) {{ return H-pad.b-(v-lo)/(hi-lo)*ch; }}

  axes(ctx, pad, W, H); grid(ctx, pad, W, H, 5, 4);

  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
  for (let i=0;i<n;i++) {{
    if(i===0) ctx.moveTo(xOf(i),yOf(ys[i]));
    else ctx.lineTo(xOf(i),yOf(ys[i]));
  }}
  ctx.stroke();

  // Axis labels
  ctx.fillStyle='#8b949e'; ctx.font='10px system-ui'; ctx.textAlign='center';
  const frames=DATA.frames||[];
  const tMax=frames.length?frames[frames.length-1].time:n*DATA.dt;
  [0,0.25,0.5,0.75,1.0].forEach(f=>{{
    const t=(tMax*f).toFixed(1);
    ctx.fillText(t, pad.l+f*cw, H-pad.b+14);
  }});
  ctx.fillText(labelX||'time', pad.l+cw/2, H-2);
  ctx.textAlign='right'; ctx.font='9px system-ui';
  [lo,(lo+hi)/2,hi].forEach(v=>ctx.fillText(v.toFixed(3),pad.l-4,yOf(v)+3));
  ctx.save(); ctx.translate(12,pad.t+ch/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign='center'; ctx.fillStyle=color; ctx.font='10px system-ui';
  ctx.fillText(labelY||'', 0, 0); ctx.restore();
}}

function lerpColor(t, c0, c1) {{
  // t in [0,1], c0/c1 as [r,g,b]
  return [
    Math.round(c0[0]+(c1[0]-c0[0])*t),
    Math.round(c0[1]+(c1[1]-c0[1])*t),
    Math.round(c0[2]+(c1[2]-c0[2])*t),
  ];
}}

function drawHeatmap(id, h, getData, colorFn) {{
  const {{ctx, W, H}} = setup(id, h);
  const pad = {{l:36, r:12, t:16, b:28}};
  const frames=DATA.frames||[];
  const N=DATA.n_sites||4;
  if (!frames.length) return;
  const cw=W-pad.l-pad.r, ch=H-pad.t-pad.b;
  const nT=frames.length, nS=N;
  const cellW=cw/nT, cellH=ch/nS;
  for (let ti=0;ti<nT;ti++) {{
    const row=getData(frames[ti]);
    for (let si=0;si<nS;si++) {{
      const v=row[si]||0;
      ctx.fillStyle=colorFn(v);
      ctx.fillRect(pad.l+ti*cellW, pad.t+si*cellH, Math.ceil(cellW)+1, Math.ceil(cellH)+1);
    }}
  }}
  // Axes
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath();
  ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.lineTo(W-pad.r,H-pad.b);
  ctx.stroke();
  // Labels
  ctx.fillStyle='#8b949e'; ctx.font='9px system-ui'; ctx.textAlign='center';
  const tMax=frames[frames.length-1].time;
  [0,0.5,1.0].forEach(f=>{{
    ctx.fillText((tMax*f).toFixed(1), pad.l+f*cw, H-pad.b+12);
  }});
  ctx.fillText('time', pad.l+cw/2, H-2);
  ctx.textAlign='right';
  for (let si=0;si<nS;si++) {{
    ctx.fillText(si, pad.l-3, pad.t+(si+0.6)*cellH);
  }}
}}

function phiColor(v) {{
  // blue (#1e90ff) → white → red (#ff4500)
  const mx=0.5;
  const t=Math.min(Math.abs(v)/mx, 1.0);
  if (v<0) {{
    const c=lerpColor(t,[255,255,255],[30,144,255]);
    return `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;
  }} else {{
    const c=lerpColor(t,[255,255,255],[255,69,0]);
    return `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;
  }}
}}

function occColor(v) {{
  // white → orange (#ff8c00)
  const mx=3.0;
  const t=Math.min(Math.max(v,0)/mx, 1.0);
  const c=lerpColor(t,[255,255,255],[255,140,0]);
  return `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;
}}

window.addEventListener('load', () => {{
  const frames=DATA.frames||[];
  const orders  = frames.map(f=>f.order_param);
  const entropies = frames.map(f=>f.entropy);
  drawLineChart('cvOrder',  220, orders,   '#36cfc9', '√⟨φ²⟩', 'time');
  drawLineChart('cvEntropy',220, entropies,'#3fb950', 'S_ent',  'time');
  drawHeatmap('cvPhiHeat', 220, f=>f.phi_expect, phiColor);
  drawHeatmap('cvOccHeat', 220, f=>f.occupation, occColor);
}});
window.addEventListener('resize', () => {{
  const frames=DATA.frames||[];
  const orders    = frames.map(f=>f.order_param);
  const entropies = frames.map(f=>f.entropy);
  drawLineChart('cvOrder',  220, orders,   '#36cfc9', '√⟨φ²⟩', 'time');
  drawLineChart('cvEntropy',220, entropies,'#3fb950', 'S_ent',  'time');
  drawHeatmap('cvPhiHeat', 220, f=>f.phi_expect, phiColor);
  drawHeatmap('cvOccHeat', 220, f=>f.occupation, occColor);
}});
</script>
</body>
</html>"""
  return html


# ── M23: Schwinger Model (1+1D QED) Lattice Quantum Field Theory ──────────────

def _schwinger_frames(
    n_sites: int = 8,
    mass: float = 0.5,
    coupling: float = 0.3,
    background: float = 0.0,
    scenario: str = "vacuum",
    dt: float = 0.08,
    n_steps: int = 60,
    n_mass_sweep: int = 30,
    seed: int = 42,
) -> dict:
  """Schwinger model (1+1D QED) via exact diagonalization.

  Kogut-Susskind staggered Hamiltonian (gauge field integrated out, open BCs):

    H = -½ Σ_n (σ+_n σ-_{n+1} + h.c.)        [hopping / kinetic]
      + (m/2) Σ_n (-1)^n σz_n                 [staggered mass]
      + (g²/2) Σ_{l=0}^{N-2} L_l²             [electric field energy]

  where L_l = background + Σ_{k=0}^{l} (σz_k + (-1)^k)/2  via Gauss's law.

  Scenarios:
    "vacuum"    — ground state observables + mass sweep (chiral condensate)
    "string"    — string tension: E(R) vs quark separation R
    "dynamics"  — time evolution from a pair-production initial state under
                  sudden quench of background electric field
  """
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("Schwinger model visualization requires numpy") from ex

  rng = np.random.default_rng(seed)
  N = n_sites
  dim = 1 << N

  # ── Gate primitives ──────────────────────────────────────────────────────
  sz = np.array([[1, 0], [0, -1]], dtype=complex)
  sp = np.array([[0, 1], [0, 0]], dtype=complex)
  sm = np.array([[0, 0], [1, 0]], dtype=complex)
  I2 = np.eye(2, dtype=complex)

  def kron_op(op: "np.ndarray", site: int) -> "np.ndarray":
    ops = [I2] * N
    ops[site] = op
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  def kron_2site(op1: "np.ndarray", op2: "np.ndarray", s1: int, s2: int) -> "np.ndarray":
    ops = [I2] * N
    ops[s1] = op1
    ops[s2] = op2
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  # ── Hamiltonian builder ───────────────────────────────────────────────────
  def build_H(m: float, g2: float, alpha: float) -> "np.ndarray":
    H = np.zeros((dim, dim), dtype=complex)
    # Hopping: -½ Σ (σ+_n σ-_{n+1} + h.c.)
    for n in range(N - 1):
      H -= 0.5 * (kron_2site(sp, sm, n, n + 1) + kron_2site(sm, sp, n, n + 1))
    # Staggered mass: (m/2) Σ (-1)^n σz_n
    for n in range(N):
      H += (m / 2.0) * ((-1) ** n) * kron_op(sz, n)
    # Electric field energy: (g²/2) Σ_{l} L_l²
    # L_l = alpha*I + Σ_{k≤l} (σz_k + (-1)^k) / 2
    for l in range(N - 1):
      # Constant part of L_l.
      c_l = alpha + sum(0.5 * ((-1) ** k) for k in range(l + 1))
      # Operator part of L_l.
      L_op = c_l * np.eye(dim, dtype=complex)
      for k in range(l + 1):
        L_op += 0.5 * kron_op(sz, k)
      H += (g2 / 2.0) * (L_op @ L_op)
    return H

  # ── Ground state via numpy eigh (exact for small N) ──────────────────────
  def ground_state(H: "np.ndarray") -> "tuple[float, np.ndarray]":
    # For dim ≤ 4096 numpy eigh is fast enough.
    vals, vecs = np.linalg.eigh(H)
    return float(np.real(vals[0])), vecs[:, 0]

  # ── Observable extraction ─────────────────────────────────────────────────
  def observables(psi: "np.ndarray", alpha: float) -> dict:
    prob = np.abs(psi) ** 2
    # <σz_n> per site.
    sz_exp = np.zeros(N)
    for n in range(N):
      Sz_n = kron_op(sz, n)
      sz_exp[n] = float(np.real(psi.conj() @ Sz_n @ psi))
    # Charge density Q_n = (σz_n + (-1)^n) / 2.
    charge = np.array([(sz_exp[n] + (-1) ** n) / 2.0 for n in range(N)])
    # Electric field per link: <L_l> = alpha + Σ_{k≤l} <Q_k>.
    elec = np.zeros(N - 1)
    cum = 0.0
    for l in range(N - 1):
      cum += charge[l]
      elec[l] = alpha + cum
    # Chiral condensate: Σ = -(1/N) Σ (-1)^n <σz_n>.
    chiral = -float(np.sum([(-1) ** n * sz_exp[n] for n in range(N)])) / N
    particle_number = float(np.sum(np.maximum(charge, 0.0)))
    return {
        "charge_density": charge.tolist(),
        "electric_field": elec.tolist(),
        "chiral_condensate": chiral,
        "particle_number": particle_number,
    }

  # ── Main scenario dispatch ────────────────────────────────────────────────
  # In dynamics mode, prepare |psi0> in a true pre-quench vacuum (alpha=0)
  # and then evolve under H(alpha_quench). Non-dynamics scenarios continue to
  # use the user-provided background directly.
  alpha0 = 0.0 if scenario == "dynamics" else background
  H0 = build_H(mass, coupling, alpha0)
  E0, psi0 = ground_state(H0)
  obs0 = observables(psi0, alpha0)

  # Mass sweep: chiral condensate and ground energy vs m ∈ [-2, 2].
  mass_sweep = []
  if n_mass_sweep > 0:
    m_vals = np.linspace(-2.0, 2.0, n_mass_sweep)
    for m_val in m_vals:
      Hi = build_H(float(m_val), coupling, background)
      Ei, psi_i = ground_state(Hi)
      obs_i = observables(psi_i, background)
      mass_sweep.append({
          "mass": float(m_val),
          "ground_energy": float(Ei),
          "chiral_condensate": obs_i["chiral_condensate"],
      })

  # String tension: energy vs quark–antiquark separation R.
  string_data = []
  if scenario == "string":
    # Create string state: insert a quark–antiquark pair by setting occupation
    # pattern and measure energy relative to vacuum as a function of R.
    # Physically: place a static charge +1 at site 0, -1 at site R.
    # Achieved by applying σ+_0 σ-_R |vacuum⟩ and measuring <H> - E0.
    # For the string scenario we vary R and compute E_string(R) = <psi_string|H|psi_string> - E0.
    for R in range(1, N // 2 + 1):
      # String state: apply σ+_0 σ-_R to vacuum.
      op_string = kron_2site(sp, sm, 0, R) if R < N else kron_op(sp, 0)
      psi_str = op_string @ psi0
      norm = np.linalg.norm(psi_str)
      if norm < 1e-10:
        # Vacuum already has this structure — use σ-_0 σ+_R instead.
        op_string = kron_2site(sm, sp, 0, R) if R < N else kron_op(sm, 0)
        psi_str = op_string @ psi0
        norm = np.linalg.norm(psi_str)
      if norm < 1e-10:
        continue
      psi_str /= norm
      E_str = float(np.real(psi_str.conj() @ H0 @ psi_str))
      string_data.append({"separation": R, "string_energy": E_str - E0})

  # Dynamics: time evolution from vacuum under sudden background quench.
  frames = []
  if scenario == "dynamics":
    alpha_quench = background if abs(background) > 1e-9 else 0.5
    H_quench = build_H(mass, coupling, alpha_quench)
    # Time evolution via matrix exponential (exact for small N).
    psi = psi0.copy()
    U_step = None  # Lazy: compute once.
    for step in range(n_steps + 1):
      t = step * dt
      obs = observables(psi, alpha_quench)
      E = float(np.real(psi.conj() @ H_quench @ psi))
      frames.append({
          "time": round(t, 6),
          "energy": E,
          **obs,
      })
      if step < n_steps:
        if U_step is None:
          from scipy.linalg import expm  # type: ignore
          U_step = expm(-1j * dt * H_quench)
        psi = U_step @ psi

  return {
      "n_sites": N,
      "mass": mass,
      "coupling": coupling,
      "background": background,
      "scenario": scenario,
      "ground_energy": E0,
      "ground_chiral_condensate": obs0["chiral_condensate"],
      "ground_charge_density": obs0["charge_density"],
      "ground_electric_field": obs0["electric_field"],
      "mass_sweep": mass_sweep,
      "string_data": string_data,
      "frames": frames,
  }


def _schwinger_fallback_dynamics(
    n_sites: int,
    mass: float,
    coupling: float,
    background_quench: float,
    dt: float,
    n_steps: int,
) -> "list[dict]":
  """Trotter-based fallback for dynamics when scipy is unavailable."""
  try:
    import numpy as np
  except ImportError:
    return []

  N = n_sites
  I2 = np.eye(2, dtype=complex)
  sz = np.array([[1, 0], [0, -1]], dtype=complex)
  sp = np.array([[0, 1], [0, 0]], dtype=complex)
  sm = np.array([[0, 0], [1, 0]], dtype=complex)

  def kron_op(op, site):
    ops = [I2] * N
    ops[site] = op
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  def kron_2site(op1, op2, s1, s2):
    ops = [I2] * N
    ops[s1] = op1
    ops[s2] = op2
    result = ops[0]
    for o in ops[1:]:
      result = np.kron(result, o)
    return result

  # Build Hamiltonian terms as a list for first-order Trotter.
  # Hopping pairs.
  hop_terms = []
  for n in range(N - 1):
    H_hop = -0.5 * (kron_2site(sp, sm, n, n + 1) + kron_2site(sm, sp, n, n + 1))
    hop_terms.append(H_hop)
  # Diagonal terms (mass + electric).
  diag_terms = []
  for n in range(N):
    diag_terms.append((mass / 2.0) * ((-1) ** n) * kron_op(sz, n))
  for l in range(N - 1):
    c_l = background_quench + sum(0.5 * ((-1) ** k) for k in range(l + 1))
    L_op = c_l * np.eye(1 << N, dtype=complex)
    for k in range(l + 1):
      L_op += 0.5 * kron_op(sz, k)
    diag_terms.append((coupling / 2.0) * (L_op @ L_op))

  # Combine all terms.
  H_full = sum(hop_terms) + sum(diag_terms)
  # Approximate Trotter: exp(-iHdt) ≈ exp(-iH_hop dt/2) exp(-iH_diag dt) exp(-iH_hop dt/2).
  # For simplicity just use full matrix: eigendecompose.
  vals, vecs = np.linalg.eigh(H_full)
  # exp(-iHt) psi = vecs @ diag(exp(-i*vals*t)) @ vecs^† @ psi.
  # Get ground state.
  psi = vecs[:, 0].copy()
  dim = 1 << N

  frames = []
  for step in range(n_steps + 1):
    t = step * dt
    coeffs = np.exp(-1j * vals * t)
    psi_t = vecs @ (coeffs * (vecs.conj().T @ psi))
    charge = np.array(
        [(float(np.real(psi_t.conj() @ kron_op(sz, n) @ psi_t)) + (-1) ** n) / 2.0
         for n in range(N)]
    )
    elec = np.zeros(N - 1)
    cum = 0.0
    for l in range(N - 1):
      cum += charge[l]
      elec[l] = background_quench + cum
    chiral = -float(np.sum([(-1) ** n *
        float(np.real(psi_t.conj() @ kron_op(sz, n) @ psi_t))
        for n in range(N)])) / N
    E = float(np.real(psi_t.conj() @ H_full @ psi_t))
    frames.append({
        "time": round(t, 6),
        "energy": E,
        "charge_density": charge.tolist(),
        "electric_field": elec.tolist(),
        "chiral_condensate": chiral,
        "particle_number": float(np.sum(np.maximum(charge, 0.0))),
    })
  return frames


def _schwinger_page(data: dict, title: str = "Schwinger Model") -> str:
  """Generate a self-contained HTML page for Schwinger model results."""
  import json as _json

  data_js = _json.dumps(data, ensure_ascii=False)
  scenario = data.get("scenario", "vacuum")

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --accent:#58a6ff;
           --green:#3fb950; --orange:#d29922; --red:#f85149; --purple:#bc8cff;
           --muted:#8b949e; --text:#e6edf3; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
          min-height:100vh; padding:24px; }}
  h1 {{ font-size:1.4rem; font-weight:600; color:var(--accent); margin-bottom:4px; }}
  .subtitle {{ color:var(--muted); font-size:.85rem; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; max-width:1100px; margin:0 auto; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:16px; }}
  .panel.wide {{ grid-column:span 2; }}
  .panel h2 {{ font-size:.95rem; color:var(--muted); margin-bottom:12px; font-weight:500; }}
  canvas {{ display:block; width:100%; }}
  .stat-row {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px; max-width:1100px; margin-left:auto; margin-right:auto; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:8px; padding:8px 14px; }}
  .stat-label {{ font-size:.75rem; color:var(--muted); margin-bottom:2px; }}
  .stat-value {{ font-size:1.05rem; font-weight:600; color:var(--accent); }}
  .tag {{ display:inline-block; font-size:.75rem; padding:2px 8px; border-radius:4px;
          background:#21262d; border:1px solid var(--border); color:var(--muted);
          margin-right:6px; margin-bottom:12px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">Kogut-Susskind staggered Hamiltonian · 1+1D QED · exact diagonalization</p>
<div>
  <span class="tag">N={data.get('n_sites',8)} sites</span>
  <span class="tag">m={data.get('mass',0.5)}</span>
  <span class="tag">g²={data.get('coupling',0.3)}</span>
  <span class="tag">α={data.get('background',0.0)}</span>
  <span class="tag">scenario={data.get('scenario','vacuum')}</span>
</div>

<div class="stat-row" id="stats"></div>

<div class="grid">
  <div class="panel">
    <h2>Ground State Charge Density Q_n = ⟨(σz_n + (−1)ⁿ)/2⟩</h2>
    <canvas id="cvCharge" height="220"></canvas>
  </div>
  <div class="panel">
    <h2>Ground State Electric Field ⟨L_l⟩ per Link</h2>
    <canvas id="cvElec" height="220"></canvas>
  </div>
  <div class="panel wide" id="sweepPanel">
    <h2>Chiral Condensate ⟨ψ̄ψ⟩ vs Bare Mass m  (order parameter)</h2>
    <canvas id="cvSweep" height="220"></canvas>
  </div>
  <div class="panel wide" id="stringPanel" style="display:none">
    <h2>String Energy E(R) − E_vac vs Quark Separation R (confinement)</h2>
    <canvas id="cvString" height="220"></canvas>
  </div>
  <div class="panel wide" id="dynPanel" style="display:none">
    <h2>Pair Production Dynamics: Particle Number and Chiral Condensate vs Time</h2>
    <canvas id="cvDyn" height="240"></canvas>
  </div>
  <div class="panel wide" id="heatPanel" style="display:none">
    <h2>Charge Density Heatmap: Q_n(t) vs Site n and Time t</h2>
    <canvas id="cvHeat" height="240"></canvas>
  </div>
</div>

<script>
const DATA = {data_js};

// ── Stats ─────────────────────────────────────────────────────────────────────
(function() {{
  const s = document.getElementById('stats');
  function stat(label, val) {{
    const d = document.createElement('div');
    d.className = 'stat';
    d.innerHTML = `<div class="stat-label">${{label}}</div><div class="stat-value">${{val}}</div>`;
    s.appendChild(d);
  }}
  stat('Ground energy', DATA.ground_energy.toFixed(5));
  stat('Chiral condensate', DATA.ground_chiral_condensate.toFixed(5));
  const ne = DATA.ground_charge_density.filter(q=>q>0.01).length;
  stat('Occupied sites', ne);
  if (DATA.string_data && DATA.string_data.length) {{
    const slope = DATA.string_data.length > 1
      ? (DATA.string_data[DATA.string_data.length-1].string_energy - DATA.string_data[0].string_energy)
        / (DATA.string_data[DATA.string_data.length-1].separation - DATA.string_data[0].separation)
      : DATA.string_data[0].string_energy;
    stat('String tension σ', slope.toFixed(4));
  }}
  if (DATA.frames && DATA.frames.length > 1) {{
    const last = DATA.frames[DATA.frames.length-1];
    stat('Final ⟨n_pairs⟩', last.particle_number.toFixed(3));
  }}
}})();

// ── Drawing helpers ────────────────────────────────────────────────────────────
function setupCanvas(id, h) {{
  const cv = document.getElementById(id);
  const dpr = window.devicePixelRatio || 1;
  cv.width  = (cv.offsetWidth  || 500) * dpr;
  cv.height = (cv.offsetHeight || h  ) * dpr;
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.fillStyle = '#161b22'; ctx.fillRect(0,0,cv.offsetWidth||500, cv.offsetHeight||h);
  return {{ctx, W:cv.offsetWidth||500, H:cv.offsetHeight||h}};
}}

function drawAxes(ctx, pad, W, H) {{
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H-pad.b);
  ctx.lineTo(W-pad.r, H-pad.b); ctx.stroke();
}}

function yTicks(ctx, pad, W, H, minV, maxV, nTick) {{
  const ch = H - pad.t - pad.b;
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
  for (let i = 0; i <= nTick; i++) {{
    const v = minV + (maxV - minV) * i / nTick;
    const y = H - pad.b - (v - minV) / (maxV - minV + 1e-14) * ch;
    ctx.fillText(v.toFixed(3), pad.l - 4, y + 3);
    ctx.strokeStyle = '#21262d'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W-pad.r, y); ctx.stroke();
  }}
}}

// ── Charge density bar chart ───────────────────────────────────────────────────
function drawCharge() {{
  const {{ctx, W, H}} = setupCanvas('cvCharge', 220);
  const pad = {{l:52, r:10, t:12, b:28}};
  const data = DATA.ground_charge_density;
  const N = data.length;
  const maxV = Math.max(...data.map(Math.abs), 0.05);
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  const zero_y = H - pad.b - (0 - (-maxV)) / (2*maxV) * ch;
  drawAxes(ctx, pad, W, H);
  // Zero line.
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, zero_y); ctx.lineTo(W-pad.r, zero_y); ctx.stroke();
  const bw = Math.max(2, cw / N - 1);
  for (let i = 0; i < N; i++) {{
    const x = pad.l + i * (cw / N);
    const bh = Math.abs(data[i]) / (2*maxV) * ch;
    ctx.fillStyle = data[i] >= 0 ? '#3fb950' : '#f85149';
    const y = data[i] >= 0 ? zero_y - bh : zero_y;
    ctx.fillRect(x + 1, y, bw, bh);
    // Site label.
    ctx.fillStyle = '#8b949e'; ctx.font = '9px system-ui'; ctx.textAlign = 'center';
    ctx.fillText(i, x + bw/2, H - pad.b + 12);
  }}
  // Y ticks.
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
  [-maxV, 0, maxV].forEach(v => {{
    const y = H - pad.b - (v - (-maxV)) / (2*maxV) * ch;
    ctx.fillText(v.toFixed(2), pad.l - 4, y + 3);
  }});
  // Legend.
  ctx.fillStyle = '#3fb950'; ctx.font = '10px system-ui'; ctx.textAlign = 'left';
  ctx.fillText('particle (Q>0)', pad.l + 4, pad.t + 12);
  ctx.fillStyle = '#f85149';
  ctx.fillText('antiparticle (Q<0)', pad.l + 4, pad.t + 24);
}}

// ── Electric field per link ────────────────────────────────────────────────────
function drawElec() {{
  const {{ctx, W, H}} = setupCanvas('cvElec', 220);
  const pad = {{l:52, r:10, t:12, b:28}};
  const data = DATA.ground_electric_field;
  const Nl = data.length;
  const minV = Math.min(...data, -0.05), maxV = Math.max(...data, 0.05);
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  drawAxes(ctx, pad, W, H);
  const bw = Math.max(2, cw / Nl - 2);
  for (let i = 0; i < Nl; i++) {{
    const x = pad.l + i * (cw / Nl);
    const bh = (data[i] - minV) / (maxV - minV + 1e-14) * ch;
    ctx.fillStyle = '#58a6ff';
    ctx.fillRect(x + 1, H - pad.b - bh, bw, bh);
    ctx.fillStyle = '#8b949e'; ctx.font = '9px system-ui'; ctx.textAlign = 'center';
    ctx.fillText(i + '–' + (i+1), x + bw/2, H - pad.b + 12);
  }}
  yTicks(ctx, pad, W, H, minV, maxV, 4);
}}

// ── Chiral condensate vs mass sweep ───────────────────────────────────────────
function drawSweep() {{
  const sweep = DATA.mass_sweep || [];
  if (!sweep.length) return;
  const {{ctx, W, H}} = setupCanvas('cvSweep', 220);
  const pad = {{l:56, r:16, t:16, b:32}};
  const ms = sweep.map(s=>s.mass);
  const cs = sweep.map(s=>s.chiral_condensate);
  const minC = Math.min(...cs), maxC = Math.max(...cs);
  const minM = Math.min(...ms), maxM = Math.max(...ms);
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  drawAxes(ctx, pad, W, H);
  function xOf(m) {{ return pad.l + (m - minM) / (maxM - minM) * cw; }}
  function yOf(c) {{ return H - pad.b - (c - minC) / (maxC - minC + 1e-14) * ch; }}
  // Fill area under curve.
  ctx.fillStyle = 'rgba(88,166,255,0.12)';
  ctx.beginPath(); ctx.moveTo(xOf(ms[0]), H - pad.b);
  for (let i = 0; i < ms.length; i++) ctx.lineTo(xOf(ms[i]), yOf(cs[i]));
  ctx.lineTo(xOf(ms[ms.length-1]), H - pad.b); ctx.closePath(); ctx.fill();
  // Line.
  ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i < ms.length; i++) {{
    if (i === 0) ctx.moveTo(xOf(ms[i]), yOf(cs[i]));
    else ctx.lineTo(xOf(ms[i]), yOf(cs[i]));
  }}
  ctx.stroke();
  // Current mass marker.
  const mCur = DATA.mass;
  if (mCur >= minM && mCur <= maxM) {{
    ctx.strokeStyle = '#d29922'; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
    const x = xOf(mCur);
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#d29922'; ctx.font = '10px system-ui'; ctx.textAlign = 'left';
    ctx.fillText('m=' + mCur, x + 3, pad.t + 12);
  }}
  // Zero condensate line.
  const y0 = yOf(0);
  if (y0 >= pad.t && y0 <= H - pad.b) {{
    ctx.strokeStyle = '#30363d'; ctx.lineWidth = 0.5; ctx.setLineDash([]);
    ctx.beginPath(); ctx.moveTo(pad.l, y0); ctx.lineTo(W-pad.r, y0); ctx.stroke();
  }}
  yTicks(ctx, pad, W, H, minC, maxC, 4);
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  [minM, 0, maxM].forEach(v => {{
    ctx.fillText(v.toFixed(1), xOf(v), H - pad.b + 14);
  }});
  ctx.fillText('bare mass m/J', pad.l + cw/2, H - 4);
  ctx.save(); ctx.translate(14, pad.t + ch/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign = 'center'; ctx.fillText('⟨ψ̄ψ⟩', 0, 0); ctx.restore();
}}

// ── String energy vs separation ────────────────────────────────────────────────
function drawString() {{
  const sdata = DATA.string_data || [];
  if (!sdata.length) return;
  document.getElementById('stringPanel').style.display = '';
  const {{ctx, W, H}} = setupCanvas('cvString', 220);
  const pad = {{l:56, r:16, t:16, b:32}};
  const Rs = sdata.map(s=>s.separation);
  const Es = sdata.map(s=>s.string_energy);
  const minR = Math.min(...Rs), maxR = Math.max(...Rs);
  const minE = 0, maxE = Math.max(...Es, 0.01);
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  function xOf(r) {{ return pad.l + (r - minR) / Math.max(maxR - minR, 1) * cw; }}
  function yOf(e) {{ return H - pad.b - (e - minE) / (maxE - minE + 1e-14) * ch; }}
  drawAxes(ctx, pad, W, H);
  // Linear confinement reference: E = σ·R, σ = (g²/2).
  const sigma = DATA.coupling / 2.0;
  ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 1; ctx.setLineDash([4,3]);
  ctx.beginPath();
  ctx.moveTo(xOf(minR), yOf(sigma * minR));
  ctx.lineTo(xOf(maxR), yOf(sigma * maxR));
  ctx.stroke(); ctx.setLineDash([]);
  // Actual string energies.
  ctx.strokeStyle = '#bc8cff'; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i < Rs.length; i++) {{
    if (i === 0) ctx.moveTo(xOf(Rs[i]), yOf(Es[i]));
    else ctx.lineTo(xOf(Rs[i]), yOf(Es[i]));
  }}
  ctx.stroke();
  for (let i = 0; i < Rs.length; i++) {{
    ctx.fillStyle = '#bc8cff';
    ctx.beginPath(); ctx.arc(xOf(Rs[i]), yOf(Es[i]), 4, 0, 2*Math.PI); ctx.fill();
  }}
  yTicks(ctx, pad, W, H, minE, maxE, 4);
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  Rs.forEach(r => ctx.fillText(r, xOf(r), H - pad.b + 14));
  ctx.fillText('separation R (sites)', pad.l + cw/2, H - 4);
  // Legend.
  ctx.textAlign = 'left'; ctx.font = '10px system-ui';
  ctx.fillStyle = '#3fb950'; ctx.fillText('σ·R (linear)', pad.l+8, pad.t+12);
  ctx.fillStyle = '#bc8cff'; ctx.fillText('E_string(R)', pad.l+8, pad.t+24);
}}

// ── Dynamics: particle number + chiral condensate vs time ─────────────────────
function drawDyn() {{
  const frames = DATA.frames || [];
  if (!frames.length) return;
  document.getElementById('dynPanel').style.display = '';
  document.getElementById('heatPanel').style.display = '';
  const {{ctx, W, H}} = setupCanvas('cvDyn', 240);
  const pad = {{l:56, r:60, t:16, b:32}};
  const ts = frames.map(f=>f.time);
  const ns = frames.map(f=>f.particle_number);
  const cs_dyn = frames.map(f=>f.chiral_condensate);
  const minT = Math.min(...ts), maxT = Math.max(...ts);
  const maxN = Math.max(...ns, 0.01);
  const minC = Math.min(...cs_dyn), maxC = Math.max(...cs_dyn);
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  function xOf(t) {{ return pad.l + (t - minT) / (maxT - minT + 1e-14) * cw; }}
  function yOfN(n) {{ return H - pad.b - (n / maxN) * ch; }}
  function yOfC(c) {{ return H - pad.b - (c - minC) / (maxC - minC + 1e-14) * ch; }}
  drawAxes(ctx, pad, W, H);
  // Grid lines.
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 0.5;
  for (let t = 0; t <= 4; t++) {{
    const y = pad.t + ch * t / 4;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W-pad.r, y); ctx.stroke();
  }}
  // Particle number.
  ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i < ts.length; i++) {{
    if (i === 0) ctx.moveTo(xOf(ts[i]), yOfN(ns[i]));
    else ctx.lineTo(xOf(ts[i]), yOfN(ns[i]));
  }}
  ctx.stroke();
  // Chiral condensate (right axis).
  ctx.strokeStyle = '#d29922'; ctx.lineWidth = 2; ctx.beginPath();
  for (let i = 0; i < ts.length; i++) {{
    if (i === 0) ctx.moveTo(xOf(ts[i]), yOfC(cs_dyn[i]));
    else ctx.lineTo(xOf(ts[i]), yOfC(cs_dyn[i]));
  }}
  ctx.stroke();
  // Left y-axis labels (particle number).
  ctx.fillStyle = '#3fb950'; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
  for (let t = 0; t <= 4; t++) {{
    const n = maxN * t / 4;
    ctx.fillText(n.toFixed(2), pad.l - 4, H - pad.b - n/maxN * ch + 3);
  }}
  // Right y-axis labels (chiral condensate).
  ctx.fillStyle = '#d29922'; ctx.textAlign = 'left';
  for (let t = 0; t <= 4; t++) {{
    const c = minC + (maxC - minC) * t / 4;
    ctx.fillText(c.toFixed(3), W - pad.r + 4, H - pad.b - (c-minC)/(maxC-minC+1e-14)*ch + 3);
  }}
  // x ticks.
  ctx.fillStyle = '#8b949e'; ctx.textAlign = 'center'; ctx.font = '10px system-ui';
  for (let t = 0; t <= 5; t++) {{
    const tv = minT + (maxT - minT) * t / 5;
    ctx.fillText(tv.toFixed(1), xOf(tv), H - pad.b + 14);
  }}
  ctx.fillText('time t (J·t/ℏ)', pad.l + cw/2, H - 4);
  // Legend.
  ctx.textAlign = 'left'; ctx.font = '11px system-ui';
  ctx.fillStyle = '#3fb950'; ctx.fillText('particle number ⟨N⟩', pad.l + 8, pad.t + 14);
  ctx.fillStyle = '#d29922'; ctx.fillText('chiral condensate ⟨ψ̄ψ⟩', pad.l + 8, pad.t + 28);
}}

// ── Charge density heatmap ─────────────────────────────────────────────────────
function drawHeat() {{
  const frames = DATA.frames || [];
  if (!frames.length) return;
  const {{ctx, W, H}} = setupCanvas('cvHeat', 240);
  const pad = {{l:52, r:16, t:12, b:32}};
  const Nframes = frames.length;
  const Nsites = (frames[0].charge_density || []).length;
  if (!Nsites) return;
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  const cellW = cw / Nframes, cellH = ch / Nsites;
  // Colour scale: negative (red) → zero (dark) → positive (green).
  const allQ = frames.flatMap(f => f.charge_density);
  const qMax = Math.max(...allQ.map(Math.abs), 0.01);
  function qColor(q) {{
    const t = q / qMax;  // [-1, 1]
    if (t >= 0) {{
      const v = Math.round(t * 200);
      return `rgb(0,${{63 + v}},0)`;
    }} else {{
      const v = Math.round(-t * 200);
      return `rgb(${{63 + v}},0,0)`;
    }}
  }}
  for (let fi = 0; fi < Nframes; fi++) {{
    const charges = frames[fi].charge_density || [];
    for (let si = 0; si < Nsites; si++) {{
      const x = pad.l + fi * cellW;
      const y = pad.t + (Nsites - 1 - si) * cellH;  // site 0 at bottom
      ctx.fillStyle = qColor(charges[si] || 0);
      ctx.fillRect(x, y, Math.ceil(cellW), Math.ceil(cellH));
    }}
  }}
  // Axes.
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H-pad.b);
  ctx.lineTo(W-pad.r, H-pad.b); ctx.stroke();
  // x labels.
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  const tStep = Math.max(1, Math.floor(Nframes / 6));
  for (let fi = 0; fi < Nframes; fi += tStep) {{
    ctx.fillText(frames[fi].time.toFixed(1), pad.l + fi * cellW, H - pad.b + 14);
  }}
  ctx.fillText('time →', pad.l + cw/2, H - 4);
  // y labels.
  ctx.textAlign = 'right';
  for (let si = 0; si < Nsites; si += 2) {{
    const y = pad.t + (Nsites - 1 - si) * cellH + cellH/2;
    ctx.fillText(si, pad.l - 4, y + 3);
  }}
  // Colour bar legend.
  ctx.fillStyle = '#f85149'; ctx.fillRect(W-pad.r-40, pad.t, 12, 10);
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'left';
  ctx.fillText('Q<0', W-pad.r-26, pad.t+9);
  ctx.fillStyle = '#3fb950'; ctx.fillRect(W-pad.r-40, pad.t+14, 12, 10);
  ctx.fillText('Q>0', W-pad.r-26, pad.t+23);
}}

window.addEventListener('load', () => {{
  drawCharge();
  drawElec();
  drawSweep();
  drawString();
  drawDyn();
  drawHeat();
}});
window.addEventListener('resize', () => {{
  drawCharge();
  drawElec();
  drawSweep();
  drawString();
  drawDyn();
  drawHeat();
}});
</script>
</body>
</html>"""
  return html


