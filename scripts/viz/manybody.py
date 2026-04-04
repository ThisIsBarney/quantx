from __future__ import annotations

import json
import math
import sys
from pathlib import Path


# ── M27-C: 1D Hubbard Model ───────────────────────────────────────────────────

def _hubbard_frames(
    n_sites: int = 6,
    t_hop: float = 1.0,
    U: float = 4.0,
    n_up: int = 0,
    n_dn: int = 0,
    scenario: str = "sweep",
    n_sweep: int = 20,
    U_min: float = 0.0,
    U_max: float = 12.0,
) -> dict:
  """Compute 1D Hubbard model via exact diagonalization (NumPy reference)."""
  import numpy as np
  from itertools import combinations

  N = n_sites
  if n_up == 0: n_up = N // 2
  if n_dn == 0: n_dn = N // 2

  def enum_states(n_bits, n_ones):
    return [sum(1 << b for b in combo) for combo in combinations(range(n_bits), n_ones)]

  def jw_sign(state, i, j):
    lo, hi = min(i, j), max(i, j)
    if hi - lo <= 1: return 1
    bits_between = sum(1 for b in range(lo + 1, hi) if (state >> b) & 1)
    return (-1) ** bits_between

  def build_H(U_val):
    up_st = enum_states(N, n_up)
    dn_st = enum_states(N, n_dn)
    up_map = {s: i for i, s in enumerate(up_st)}
    dn_map = {s: i for i, s in enumerate(dn_st)}
    dim = len(up_st) * len(dn_st)
    H = np.zeros((dim, dim), dtype=float)
    for iu, us in enumerate(up_st):
      for id_, ds in enumerate(dn_st):
        idx = iu * len(dn_st) + id_
        # diagonal: U * double-occupancy
        docc = bin(us & ds).count('1')
        if docc: H[idx, idx] += U_val * docc
        # spin-up hopping
        for site in range(N - 1):
          b0, b1 = 1 << site, 1 << (site + 1)
          if (us & b0) and not (us & b1):
            us2 = (us ^ b0) | b1
            if us2 in up_map:
              iu2 = up_map[us2]; idx2 = iu2 * len(dn_st) + id_
              sg = jw_sign(us, site, site + 1)
              H[idx2, idx] -= t_hop * sg
              H[idx, idx2] -= t_hop * sg
        # spin-down hopping
        for site in range(N - 1):
          b0, b1 = 1 << site, 1 << (site + 1)
          if (ds & b0) and not (ds & b1):
            ds2 = (ds ^ b0) | b1
            if ds2 in dn_map:
              id2 = dn_map[ds2]; idx2 = iu * len(dn_st) + id2
              sg = jw_sign(ds, site, site + 1)
              H[idx2, idx] -= t_hop * sg
              H[idx, idx2] -= t_hop * sg
    return H, up_st, dn_st, up_map, dn_map

  def compute_obs(psi, up_st, dn_st, up_map, dn_map):
    n_dn_loc = len(dn_st)
    docc = 0.0
    sz_corr = np.zeros(N)
    rho = np.zeros((N, N), dtype=float)
    for iu, us in enumerate(up_st):
      for id_, ds in enumerate(dn_st):
        idx = iu * n_dn_loc + id_
        p = psi[idx] ** 2
        docc += p * bin(us & ds).count('1')
        sz0 = 0.5 * (((us >> 0) & 1) - ((ds >> 0) & 1))
        for r in range(N):
          szr = 0.5 * (((us >> r) & 1) - ((ds >> r) & 1))
          sz_corr[r] += p * sz0 * szr
        # one-body density matrix (diagonal part)
        for s in range(N):
          if (us >> s) & 1: rho[s, s] += p
          if (ds >> s) & 1: rho[s, s] += p
    # off-diagonal rho (spin-up)
    for iu, us in enumerate(up_st):
      for id_, ds in enumerate(dn_st):
        idx = iu * n_dn_loc + id_
        pv = psi[idx]
        for s in range(N):
          if not (us >> s) & 1: continue
          for r in range(N):
            if r == s or (us >> r) & 1: continue
            us2 = (us & ~(1 << s)) | (1 << r)
            if us2 in up_map:
              idx2 = up_map[us2] * n_dn_loc + id_
              sg = jw_sign(us, s, r)
              rho[r, s] += psi[idx2] * pv * sg
    # off-diagonal rho (spin-down)
    for iu, us in enumerate(up_st):
      for id_, ds in enumerate(dn_st):
        idx = iu * n_dn_loc + id_
        pv = psi[idx]
        for s in range(N):
          if not (ds >> s) & 1: continue
          for r in range(N):
            if r == s or (ds >> r) & 1: continue
            ds2 = (ds & ~(1 << s)) | (1 << r)
            if ds2 in dn_map:
              idx2 = iu * n_dn_loc + dn_map[ds2]
              sg = jw_sign(ds, s, r)
              rho[r, s] += psi[idx2] * pv * sg
    n_k = np.array([
        sum(np.cos(2 * np.pi * m * (r - s) / N) * rho[r, s]
            for r in range(N) for s in range(N)) / N
        for m in range(N)
    ])
    s_pi = sum(((-1) ** r) * sz_corr[r] for r in range(N)) / N
    return {
        "double_occupancy": float(docc / N),
        "spin_structure_pi": float(s_pi),
        "spin_corr": sz_corr.tolist(),
        "momentum_dist": n_k.tolist(),
    }

  def solve_one(U_val):
    H, up_st, dn_st, up_map, dn_map = build_H(U_val)
    evals, evecs = np.linalg.eigh(H)
    psi0 = evecs[:, 0]
    E0 = float(evals[0])
    E1 = float(evals[1]) if len(evals) > 1 else E0
    obs = compute_obs(psi0, up_st, dn_st, up_map, dn_map)
    return E0, E1, obs

  sweep_pts = []
  if scenario == "sweep":
    for i in range(n_sweep):
      U_val = U_min + (U_max - U_min) * i / max(n_sweep - 1, 1)
      E0, E1, obs = solve_one(U_val)
      sweep_pts.append({
          "U": float(U_val),
          "ground_energy": E0,
          "energy_gap": E1 - E0,
          "double_occupancy": obs["double_occupancy"],
          "spin_structure_pi": obs["spin_structure_pi"],
      })

  E0, E1, obs = solve_one(U)
  return {
      "n_sites": N,
      "n_up": n_up,
      "n_dn": n_dn,
      "t_hop": t_hop,
      "U": U,
      "ground_energy": E0,
      "energy_gap": E1 - E0,
      "double_occupancy": obs["double_occupancy"],
      "spin_structure_pi": obs["spin_structure_pi"],
      "spin_corr": obs["spin_corr"],
      "momentum_dist": obs["momentum_dist"],
      "sweep": sweep_pts,
      "scenario": scenario,
  }


def _hubbard_page(data: dict, title: str = "1D Hubbard Model") -> str:
  """Generate self-contained HTML for the 1D Hubbard model."""
  import json as json_mod
  data_json = json_mod.dumps(data)
  N = data.get("n_sites", 6)

  return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif;padding:20px}}
h2{{font-size:1.2rem;font-weight:600;color:#58a6ff;margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:960px;margin:0 auto}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
canvas{{display:block;width:100%;height:240px}}
.info{{font-size:0.82rem;color:#8b949e;margin-top:8px}}
.param{{display:flex;gap:24px;flex-wrap:wrap;font-size:0.85rem;color:#8b949e;margin-bottom:16px;max-width:960px;margin-left:auto;margin-right:auto}}
.param span{{color:#58a6ff;font-weight:600}}
</style>
</head>
<body>
<h1 style="font-size:1.4rem;color:#58a6ff;margin-bottom:12px">{title}</h1>
<div class="param">
  <div>N = <span id="pN"></span> sites</div>
  <div>N↑ = <span id="pNup"></span>, N↓ = <span id="pNdn"></span></div>
  <div>t = <span id="pt"></span></div>
  <div>U/t = <span id="pUt"></span></div>
  <div>E₀ = <span id="pE0"></span></div>
  <div>Δ = <span id="pGap"></span></div>
  <div>D = <span id="pDocc"></span></div>
  <div>S(π) = <span id="pSpi"></span></div>
</div>
<div class="grid">
  <div class="card"><h2>Spin Correlation ⟨S⁰ᶻ Sʳᶻ⟩</h2><canvas id="cSpinCorr"></canvas>
    <div class="info">Antiferromagnetic: alternating sign pattern at large U/t</div></div>
  <div class="card"><h2>Momentum Distribution n(k)</h2><canvas id="cMomDist"></canvas>
    <div class="info">Fermi step discontinuity at k_F for non-interacting; smeared at large U</div></div>
  <div class="card"><h2>Double Occupancy vs U/t</h2><canvas id="cDocc"></canvas>
    <div class="info">D → 0 as U/t → ∞ (Mott insulator)</div></div>
  <div class="card"><h2>S(π) Spin Structure Factor vs U/t</h2><canvas id="cSpi"></canvas>
    <div class="info">Antiferromagnetic order grows with U/t</div></div>
</div>
<script>
const DATA = {data_json};

// Populate params
document.getElementById("pN").textContent    = DATA.n_sites;
document.getElementById("pNup").textContent  = DATA.n_up;
document.getElementById("pNdn").textContent  = DATA.n_dn;
document.getElementById("pt").textContent    = DATA.t_hop.toFixed(2);
document.getElementById("pUt").textContent   = (DATA.U / DATA.t_hop).toFixed(2);
document.getElementById("pE0").textContent   = DATA.ground_energy.toFixed(4);
document.getElementById("pGap").textContent  = DATA.energy_gap.toFixed(4);
document.getElementById("pDocc").textContent = DATA.double_occupancy.toFixed(4);
document.getElementById("pSpi").textContent  = DATA.spin_structure_pi.toFixed(4);

function barChart(canvasId, labels, values, colorFn, yLabel) {{
  var cv = document.getElementById(canvasId);
  var w = cv.offsetWidth || 400; var h = cv.offsetHeight || 240;
  cv.width = w; cv.height = h;
  var ctx = cv.getContext("2d");
  var pad = 40;
  var bw = Math.max(1, (w - 2*pad) / values.length - 2);
  var vmax = Math.max(...values.map(Math.abs)) * 1.2 || 1;
  var vmin = Math.min(0, ...values) * 1.2;
  var range = vmax - vmin;
  var zero_y = h - pad - (-vmin / range) * (h - 2*pad);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, h-pad); ctx.lineTo(w-pad, h-pad); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad, zero_y); ctx.lineTo(w-pad, zero_y);
  ctx.strokeStyle = "#444c56"; ctx.stroke();
  for (var i = 0; i < values.length; i++) {{
    var x = pad + i * ((w - 2*pad) / values.length) + 1;
    var bh = Math.abs(values[i]) / range * (h - 2*pad);
    var y = values[i] >= 0 ? zero_y - bh : zero_y;
    ctx.fillStyle = colorFn(i, values[i]);
    ctx.fillRect(x, y, bw, bh);
    if (labels && labels[i] !== undefined) {{
      ctx.fillStyle = "#8b949e"; ctx.font = "10px monospace";
      ctx.textAlign = "center";
      ctx.fillText(labels[i], x + bw/2, h - pad + 12);
    }}
  }}
  if (yLabel) {{
    ctx.fillStyle = "#8b949e"; ctx.font = "11px sans-serif"; ctx.textAlign = "left";
    ctx.fillText(yLabel, 4, 14);
  }}
}}

// Spin correlation
var scLabels = DATA.spin_corr.map(function(_, i) {{ return "r=" + i; }});
barChart("cSpinCorr", scLabels, DATA.spin_corr,
  function(i, v) {{ return v >= 0 ? "#388bfd" : "#f47067"; }}, "⟨S⁰ᶻSʳᶻ⟩");

// Momentum distribution
var N = DATA.n_sites;
var kLabels = DATA.momentum_dist.map(function(_, m) {{ return (m <= N/2) ? m : m-N; }});
barChart("cMomDist", kLabels.map(function(k) {{ return "k=" + k; }}), DATA.momentum_dist,
  function() {{ return "#57ab5a"; }}, "n(k)");

// Sweep charts
if (DATA.sweep && DATA.sweep.length > 0) {{
  var sweepU   = DATA.sweep.map(function(p) {{ return p.U / DATA.t_hop; }});
  var sweepD   = DATA.sweep.map(function(p) {{ return p.double_occupancy; }});
  var sweepSpi = DATA.sweep.map(function(p) {{ return p.spin_structure_pi; }});

  function lineChart(canvasId, xs, ys, color, yLabel) {{
    var cv = document.getElementById(canvasId);
    var w = cv.offsetWidth || 400; var h = cv.offsetHeight || 240;
    cv.width = w; cv.height = h;
    var ctx = cv.getContext("2d");
    var pad = 44;
    var xmin = Math.min(...xs), xmax = Math.max(...xs);
    var ymin = Math.min(0, ...ys), ymax = Math.max(...ys) * 1.1 || 1;
    var xr = xmax - xmin || 1; var yr = ymax - ymin || 1;
    function px(x) {{ return pad + (x - xmin)/xr * (w - 2*pad); }}
    function py(y) {{ return h - pad - (y - ymin)/yr * (h - 2*pad); }}
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
    // axes
    ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, h-pad);
    ctx.lineTo(w-pad, h-pad); ctx.stroke();
    // grid
    for (var yi = 0; yi <= 4; yi++) {{
      var gy = py(ymin + yr * yi/4);
      ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(w-pad, gy);
      ctx.strokeStyle = "#21262d"; ctx.stroke();
      ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "right";
      ctx.fillText((ymin + yr * yi/4).toFixed(2), pad - 3, gy + 3);
    }}
    // line
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
    for (var i = 0; i < xs.length; i++) {{
      i === 0 ? ctx.moveTo(px(xs[i]), py(ys[i])) : ctx.lineTo(px(xs[i]), py(ys[i]));
    }}
    ctx.stroke();
    // x ticks
    for (var xi = 0; xi <= 4; xi++) {{
      var gx = px(xmin + xr * xi/4);
      ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "center";
      ctx.fillText((xmin + xr * xi/4).toFixed(1), gx, h - pad + 12);
    }}
    ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "left";
    ctx.fillText(yLabel, 4, 14);
    ctx.fillText("U/t", w/2, h - 4);
  }}

  lineChart("cDocc", sweepU, sweepD, "#e3b341", "Double Occupancy D");
  lineChart("cSpi", sweepU, sweepSpi, "#f47067", "S(π)");
}}
</script>
</body>
</html>"""


# ── M27-D: Floquet Kicked-Ising ───────────────────────────────────────────────

def _floquet_frames(
    n_sites: int = 8,
    J: float = 1.0,
    h_z: float = 0.809,
    drive_amp: float = 1.5,
    drive_omega: float = 4.5,
    n_periods: int = 40,
    scenario: str = "dynamics",
    n_scan: int = 20,
    scan_amp_max: float = 3.0,
) -> dict:
  """Compute Floquet kicked-Ising dynamics (NumPy reference)."""
  import numpy as np

  N = n_sites
  dim = 2 ** N
  PI = np.pi
  T = 2.0 * PI / drive_omega
  halfT = T / 2.0

  # Build H₀ diagonal (Z basis)
  h0_diag = np.zeros(dim)
  for s in range(dim):
    e = 0.0
    for i in range(N - 1):
      zi = 1 if (s >> i) & 1 else -1
      zj = 1 if (s >> (i+1)) & 1 else -1
      e -= J * zi * zj
    for i in range(N):
      zi = 1 if (s >> i) & 1 else -1
      e -= h_z * zi
    h0_diag[s] = e

  def apply_exp_h0(psi, tau):
    return psi * np.exp(-1j * tau * h0_diag)

  def apply_kick(psi, hx):
    c = np.cos(hx); ms = -1j * np.sin(hx)
    out = psi.copy()
    for i in range(N):
      bit = 1 << i
      # Pair each |s⟩ with |s ^ bit⟩
      s_arr = np.arange(dim)
      mask = (s_arr & bit) == 0
      lo = s_arr[mask]; hi = lo | bit
      a, b = out[lo].copy(), out[hi].copy()
      out[lo] = c * a + ms * b
      out[hi] = ms * a + c * b
    return out

  def apply_period(psi):
    psi = apply_exp_h0(psi, halfT)
    psi = apply_kick(psi, drive_amp)
    psi = apply_exp_h0(psi, halfT)
    return psi

  def entropy(psi):
    dimA = 2 ** (N // 2); dimB = 2 ** (N - N // 2)
    M = psi.reshape(dimA, dimB)
    sv = np.linalg.svd(M, compute_uv=False)
    lam = sv ** 2
    lam = lam[lam > 1e-14]
    lam /= lam.sum()
    return float(-np.sum(lam * np.log(lam)))

  frames = []
  # Néel state |↑↓↑↓…⟩
  neel = sum(1 << i for i in range(0, N, 2))
  psi = np.zeros(dim, dtype=complex); psi[neel] = 1.0

  def record(period, t, psi_r):
    sz_ops = np.array([1 if ((s >> i) & 1) else -1
                       for i in range(N) for s in range(dim)]).reshape(N, dim)
    prob = np.abs(psi_r) ** 2
    mz = float(np.sum(prob[np.newaxis, :] * sz_ops) / (N * dim))
    mz = float(np.dot(prob, sum(
        np.array([1 if (s >> i) & 1 else -1 for s in range(dim)]) for i in range(N)
    )) / N)
    en = float(np.real(np.dot(np.conj(psi_r), h0_diag * psi_r)))
    s = entropy(psi_r)
    frames.append({"period": int(period), "time": float(t),
                   "magnetization_z": float(mz), "energy": en, "entropy": s})

  record(0, 0.0, psi)
  for p in range(1, n_periods + 1):
    psi = apply_period(psi)
    record(p, p * T, psi)

  # Quasi-energy spectrum
  quasi_energies = []
  mean_gap_ratio = 0.0
  if dim <= 256:
    U_mat = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
      e = np.zeros(dim, dtype=complex); e[col] = 1.0
      e = apply_period(e)
      U_mat[:, col] = e
    evals = np.linalg.eigvals(U_mat)
    qe = sorted(float(np.angle(ev)) / T for ev in evals)
    quasi_energies = qe
    if len(qe) >= 3:
      diffs = np.diff(qe)
      r_vals = [min(diffs[i], diffs[i+1]) / max(diffs[i], diffs[i+1])
                for i in range(len(diffs)-1)
                if diffs[i] > 1e-14 and diffs[i+1] > 1e-14]
      mean_gap_ratio = float(np.mean(r_vals)) if r_vals else 0.0

  # Amplitude scan
  scan_amps = []
  scan_qe = []
  if scenario == "spectrum" and dim <= 64:
    for si in range(n_scan):
      amp = scan_amp_max * si / max(n_scan - 1, 1)
      scan_amps.append(float(amp))
      U_s = np.zeros((dim, dim), dtype=complex)
      for col in range(dim):
        e = np.zeros(dim, dtype=complex); e[col] = 1.0
        e = apply_exp_h0(e, halfT)
        e = apply_kick(e, amp)
        e = apply_exp_h0(e, halfT)
        U_s[:, col] = e
      evs = np.linalg.eigvals(U_s)
      scan_qe.append(sorted(float(np.angle(ev)) / T for ev in evs))

  return {
      "n_sites": N, "J": J, "h_z": h_z,
      "drive_amp": drive_amp, "drive_omega": drive_omega,
      "T": T, "frames": frames,
      "quasi_energies": quasi_energies,
      "mean_gap_ratio": mean_gap_ratio,
      "scan_amps": scan_amps,
      "scan_qe": scan_qe,
  }


def _floquet_page(data: dict, title: str = "Floquet 周期驱动") -> str:
  """Generate self-contained HTML for the Floquet kicked-Ising simulation."""
  import json as json_mod
  data_json = json_mod.dumps(data)

  return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif;padding:20px}}
h2{{font-size:1.1rem;font-weight:600;color:#58a6ff;margin-bottom:10px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:980px;margin:0 auto}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
canvas{{display:block;width:100%;height:220px}}
.info{{font-size:0.8rem;color:#8b949e;margin-top:6px}}
.param{{display:flex;gap:20px;flex-wrap:wrap;font-size:0.84rem;color:#8b949e;margin-bottom:14px;max-width:980px;margin-left:auto;margin-right:auto}}
.param span{{color:#58a6ff;font-weight:600}}
.badge{{display:inline-block;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:600}}
.badge-green{{background:#033a16;color:#3fb950}}
.badge-red{{background:#3d1a1a;color:#f47067}}
</style>
</head>
<body>
<h1 style="font-size:1.4rem;color:#58a6ff;margin-bottom:12px">{title}</h1>
<div class="param">
  <div>N = <span id="pN"></span> qubits</div>
  <div>J = <span id="pJ"></span></div>
  <div>h_z = <span id="pHz"></span></div>
  <div>h_x (kick) = <span id="pHx"></span></div>
  <div>ω = <span id="pOmega"></span></div>
  <div>T = <span id="pT"></span></div>
  <div>Gap ratio r = <span id="pGapR"></span></div>
  <div id="phaseBadge"></div>
</div>
<div class="grid">
  <div class="card"><h2>Magnetization ⟨m_z(t)⟩</h2><canvas id="cMz"></canvas>
    <div class="info">Stroboscopic snapshots at integer multiples of period T</div></div>
  <div class="card"><h2>Bipartite Entanglement Entropy</h2><canvas id="cEnt"></canvas>
    <div class="info">Logarithmic growth → thermalization; bounded → MBL / prethermal</div></div>
  <div class="card"><h2>⟨H₀⟩ Energy Absorption</h2><canvas id="cEn"></canvas>
    <div class="info">Heating to infinite temperature in ergodic phase</div></div>
  <div class="card"><h2>Quasi-energy Spectrum (level spacing)</h2><canvas id="cQE"></canvas>
    <div class="info">Poisson ≈ 0.39 (MBL/integrable); GUE ≈ 0.60 (ergodic/chaotic)</div></div>
</div>
<script>
const DATA = {data_json};

document.getElementById("pN").textContent     = DATA.n_sites;
document.getElementById("pJ").textContent     = DATA.J.toFixed(2);
document.getElementById("pHz").textContent    = DATA.h_z.toFixed(3);
document.getElementById("pHx").textContent    = DATA.drive_amp.toFixed(3);
document.getElementById("pOmega").textContent = DATA.drive_omega.toFixed(3);
document.getElementById("pT").textContent     = (DATA.T || 0).toFixed(3);
var gr = DATA.mean_gap_ratio || 0;
document.getElementById("pGapR").textContent  = gr.toFixed(3);
var badge = document.getElementById("phaseBadge");
if (gr > 0) {{
  if (gr > 0.52) {{
    badge.innerHTML = '<span class="badge badge-red">Ergodic / Chaotic</span>';
  }} else {{
    badge.innerHTML = '<span class="badge badge-green">MBL / Prethermal</span>';
  }}
}}

function lineChart(canvasId, xs, ys, color, xLabel, yLabel, ymin_hint) {{
  var cv = document.getElementById(canvasId);
  var w = cv.offsetWidth || 420; var h = cv.offsetHeight || 220;
  cv.width = w; cv.height = h;
  var ctx = cv.getContext("2d");
  var pad = {{l:50, r:12, t:10, b:32}};
  var xmin = Math.min(...xs), xmax = Math.max(...xs);
  var ymin = ymin_hint !== undefined ? ymin_hint : Math.min(...ys);
  var ymax = Math.max(...ys);
  if (ymax === ymin) {{ ymax += 0.001; }}
  var ypad = (ymax - ymin) * 0.08;
  ymin -= ypad; ymax += ypad;
  var xr = xmax - xmin || 1; var yr = ymax - ymin;
  function px(x) {{ return pad.l + (x - xmin)/xr * (w - pad.l - pad.r); }}
  function py(y) {{ return h - pad.b - (y - ymin)/yr * (h - pad.t - pad.b); }}
  ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#21262d"; ctx.lineWidth = 1;
  for (var gi = 0; gi <= 4; gi++) {{
    var gy = py(ymin + yr * gi / 4);
    ctx.beginPath(); ctx.moveTo(pad.l, gy); ctx.lineTo(w - pad.r, gy); ctx.stroke();
    ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "right";
    ctx.fillText((ymin + yr * gi/4).toFixed(2), pad.l - 3, gy + 3);
  }}
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b);
  ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
  for (var i = 0; i < xs.length; i++) {{
    i === 0 ? ctx.moveTo(px(xs[i]), py(ys[i])) : ctx.lineTo(px(xs[i]), py(ys[i]));
  }}
  ctx.stroke();
  // x ticks
  for (var xi = 0; xi <= 5; xi++) {{
    var gx = px(xmin + xr * xi / 5);
    ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "center";
    ctx.fillText((xmin + xr * xi/5).toFixed(1), gx, h - pad.b + 12);
  }}
  ctx.fillStyle = "#6e7681"; ctx.font = "10px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(xLabel, w/2, h - 2);
  ctx.save(); ctx.translate(12, h/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign = "center"; ctx.fillText(yLabel, 0, 0); ctx.restore();
}}

var ts = DATA.frames.map(function(f) {{ return f.period; }});
var mz = DATA.frames.map(function(f) {{ return f.magnetization_z; }});
var en = DATA.frames.map(function(f) {{ return f.energy; }});
var sv = DATA.frames.map(function(f) {{ return f.entropy; }});

lineChart("cMz", ts, mz, "#58a6ff", "Period n", "⟨m_z⟩");
lineChart("cEnt", ts, sv, "#3fb950", "Period n", "S(n)", 0);
lineChart("cEn", ts, en, "#e3b341", "Period n", "⟨H₀⟩");

// Quasi-energy spectrum: plot level spacings as histogram
(function() {{
  var qe = DATA.quasi_energies;
  if (!qe || qe.length < 4) {{
    var cv = document.getElementById("cQE");
    cv.width = cv.offsetWidth || 420; cv.height = cv.offsetHeight || 220;
    var ctx = cv.getContext("2d");
    ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = "#8b949e"; ctx.font = "13px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("Spectrum: N too large for full diag", cv.width/2, cv.height/2);
    return;
  }}
  // Compute level spacings
  var sorted = qe.slice().sort(function(a,b) {{ return a-b; }});
  var spacings = [];
  for (var i = 0; i < sorted.length - 1; i++)
    spacings.push(sorted[i+1] - sorted[i]);
  var mean_s = spacings.reduce(function(a,b){{return a+b;}}, 0) / spacings.length;
  if (mean_s < 1e-14) mean_s = 1;
  var r_norm = spacings.map(function(s) {{ return s / mean_s; }});

  // Histogram
  var nbins = 20; var binmax = 3.0;
  var bins = new Array(nbins).fill(0);
  r_norm.forEach(function(s) {{
    var bi = Math.min(nbins - 1, Math.floor(s / binmax * nbins));
    bins[bi]++;
  }});
  var bmax = Math.max(...bins);

  var cv = document.getElementById("cQE");
  var w = cv.offsetWidth || 420; var h = cv.offsetHeight || 220;
  cv.width = w; cv.height = h;
  var ctx = cv.getContext("2d");
  var pad = 40; var bw = (w - 2*pad) / nbins;
  ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
  for (var i = 0; i < nbins; i++) {{
    var bh = bins[i] / bmax * (h - 2*pad);
    ctx.fillStyle = "#388bfd";
    ctx.fillRect(pad + i*bw + 1, h - pad - bh, bw - 2, bh);
  }}
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, h-pad); ctx.lineTo(w-pad, h-pad); ctx.stroke();
  // Reference lines: Poisson 0.39, GUE 0.60
  var r_poisson = 0.39, r_gue = 0.60;
  function grx(r) {{ return pad + r / binmax * (w - 2*pad); }}
  ctx.strokeStyle = "#3fb950"; ctx.lineWidth = 1.5; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(grx(r_poisson), pad); ctx.lineTo(grx(r_poisson), h-pad); ctx.stroke();
  ctx.strokeStyle = "#f47067";
  ctx.beginPath(); ctx.moveTo(grx(r_gue), pad); ctx.lineTo(grx(r_gue), h-pad); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#3fb950"; ctx.font = "9px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("P", grx(r_poisson), pad - 3);
  ctx.fillStyle = "#f47067";
  ctx.fillText("GUE", grx(r_gue), pad - 3);
  ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "center";
  for (var xi = 0; xi <= 3; xi++) {{
    ctx.fillText((binmax * xi / 3).toFixed(1), pad + (w-2*pad)*xi/3, h-pad+12);
  }}
  ctx.fillText("r = gap ratio", w/2, h-2);
}})();
</script>
</body>
</html>"""


# ── M28-E: XXZ Heisenberg spin chain ─────────────────────────────────────────

def _heisenberg_frames(
    n_sites: int = 8,
    J: float = 1.0,
    Jz: float = 1.0,
    h: float = 0.0,
    scenario: str = "sweep",
    n_sweep: int = 20,
    Jz_min: float = -2.0,
    Jz_max: float = 3.0,
) -> dict:
  """Compute 1D XXZ Heisenberg chain via exact diagonalization (NumPy reference)."""
  import numpy as np
  from itertools import combinations

  N = n_sites
  n_down = N // 2  # Sz = 0 sector

  def spin_states(n_bits, nd):
    return [sum(1 << b for b in combo) for combo in combinations(range(n_bits), nd)]

  def build_H(Jz_val):
    states = spin_states(N, n_down)
    s_map = {s: i for i, s in enumerate(states)}
    dim = len(states)
    H = np.zeros((dim, dim), dtype=float)
    for bi, s in enumerate(states):
      # ZZ diagonal + longitudinal field
      for i in range(N - 1):
        sz_i = 0.5 if not ((s >> i) & 1) else -0.5
        sz_j = 0.5 if not ((s >> (i+1)) & 1) else -0.5
        H[bi, bi] += Jz_val / 4.0 * (2*sz_i) * (2*sz_j)  # Jz/4 * ZiZj
      if h != 0.0:
        for i in range(N):
          sz_i = 0.5 if not ((s >> i) & 1) else -0.5
          H[bi, bi] -= h / 2.0 * (2*sz_i)
      # XX + YY (= S+S- + S-S+) off-diagonal
      for i in range(N - 1):
        # S+_i S-_{i+1}: site i up (0), site i+1 down (1)
        i_up = not ((s >> i) & 1)
        j_down = (s >> (i+1)) & 1
        if i_up and j_down:
          s2 = s | (1 << i)
          s2 = s2 & ~(1 << (i+1))
          if s2 in s_map:
            H[bi, s_map[s2]] += J / 2.0
            H[s_map[s2], bi] += J / 2.0
    return H, states

  def spin_corr(states, psi):
    corr = np.zeros(N)
    s_map = {s: i for i, s in enumerate(states)}
    for bi, s in enumerate(states):
      p = psi[bi] ** 2
      sz0 = 0.5 if not (s & 1) else -0.5
      for r in range(N):
        # ZZ part
        sz_r = 0.5 if not ((s >> r) & 1) else -0.5
        corr[r] += p * sz0 * sz_r
    # XX+YY contributions via off-diagonal
    for bi, s in enumerate(states):
      for r in range(1, N):
        i_up = not (s & 1)
        j_down = (s >> r) & 1
        if i_up and j_down:
          s2 = s | 1
          s2 = s2 & ~(1 << r)
          if s2 in s_map:
            bj = s_map[s2]
            corr[r] += 0.5 * psi[bi] * psi[bj]  # XX part
    return corr

  if scenario == "sweep":
    sweep = []
    for k in range(n_sweep):
      Jz_k = Jz_min + (Jz_max - Jz_min) * k / max(n_sweep - 1, 1)
      H, states = build_H(Jz_k)
      evals, evecs = np.linalg.eigh(H)
      E0, E1 = evals[0], evals[1] if len(evals) > 1 else evals[0]
      psi = evecs[:, 0]
      corr = spin_corr(states, psi)
      spi = sum((-1)**r * corr[r] for r in range(N)) / N
      Sz_total = sum(
          (0.5 if not ((s >> i) & 1) else -0.5) * psi[bi]**2
          for bi, s in enumerate(states)
          for i in range(N)
      )
      sweep.append({
          "Jz": Jz_k,
          "ground_energy": float(E0),
          "energy_gap": float(E1 - E0),
          "structure_pi": float(spi),
          "Sz_total": float(Sz_total),
      })
    # Single-point at requested Jz
    H0, states0 = build_H(Jz)
    evals0, evecs0 = np.linalg.eigh(H0)
    psi0 = evecs0[:, 0]
    corr0 = spin_corr(states0, psi0)
    spi0 = sum((-1)**r * corr0[r] for r in range(N)) / N
    return {
        "n_sites": N, "J": J, "Jz": Jz, "h": h,
        "ground_energy": float(evals0[0]),
        "energy_gap": float(evals0[1] - evals0[0]) if len(evals0) > 1 else 0.0,
        "structure_pi": float(spi0),
        "spin_corr": corr0.tolist(),
        "sweep": sweep,
    }
  else:  # ground_state
    H0, states0 = build_H(Jz)
    evals0, evecs0 = np.linalg.eigh(H0)
    psi0 = evecs0[:, 0]
    corr0 = spin_corr(states0, psi0)
    spi0 = sum((-1)**r * corr0[r] for r in range(N)) / N
    return {
        "n_sites": N, "J": J, "Jz": Jz, "h": h,
        "ground_energy": float(evals0[0]),
        "energy_gap": float(evals0[1] - evals0[0]) if len(evals0) > 1 else 0.0,
        "structure_pi": float(spi0),
        "spin_corr": corr0.tolist(),
        "sweep": [],
    }


def _heisenberg_page(data: dict, title: str = "XXZ Heisenberg Chain") -> str:
  """Generate self-contained HTML for the 1D XXZ Heisenberg model."""
  import json as json_mod
  data_json = json_mod.dumps(data)
  N = data.get("n_sites", 8)

  return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif;padding:20px}}
h2{{font-size:1.2rem;font-weight:600;color:#58a6ff;margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:960px;margin:0 auto}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
canvas{{display:block;width:100%;height:240px}}
.info{{font-size:0.82rem;color:#8b949e;margin-top:8px}}
.param{{display:flex;gap:24px;flex-wrap:wrap;font-size:0.85rem;color:#8b949e;margin-bottom:16px;max-width:960px;margin-left:auto;margin-right:auto}}
.param span{{color:#58a6ff;font-weight:600}}
</style>
</head>
<body>
<h1 style="font-size:1.4rem;color:#58a6ff;margin-bottom:12px">{title}</h1>
<div class="param">
  <div>N = <span id="pN"></span> sites</div>
  <div>J = <span id="pJ"></span></div>
  <div>Jz = <span id="pJz"></span></div>
  <div>E₀ = <span id="pE0"></span></div>
  <div>Δ = <span id="pGap"></span></div>
  <div>S(π) = <span id="pSpi"></span></div>
</div>
<div class="grid">
  <div class="card"><h2>Spin-Spin Correlation ⟨S₀·Sᵣ⟩</h2><canvas id="cSpinCorr"></canvas>
    <div class="info">Antiferromagnetic: alternating for Jz &gt; 0; ferromagnetic for Jz &lt; 0</div></div>
  <div class="card"><h2>S(π) Spin Structure Factor vs Jz</h2><canvas id="cSpi"></canvas>
    <div class="info">Peaks in antiferromagnetic phase (Jz &gt; −J); vanishes in ferromagnetic phase</div></div>
  <div class="card"><h2>Energy Gap Δ vs Jz</h2><canvas id="cGap"></canvas>
    <div class="info">Gapless for |Jz| ≤ J (XXZ critical line); gapped outside</div></div>
  <div class="card"><h2>Ground Energy E₀ vs Jz</h2><canvas id="cEnergy"></canvas>
    <div class="info">Ground energy decreases with Jz in the antiferromagnetic regime</div></div>
</div>
<script>
const DATA = {data_json};

document.getElementById("pN").textContent   = DATA.n_sites;
document.getElementById("pJ").textContent   = DATA.J.toFixed(3);
document.getElementById("pJz").textContent  = DATA.Jz.toFixed(3);
document.getElementById("pE0").textContent  = DATA.ground_energy.toFixed(4);
document.getElementById("pGap").textContent = DATA.energy_gap.toFixed(4);
document.getElementById("pSpi").textContent = DATA.structure_pi.toFixed(4);

function barChart(canvasId, labels, values, colorFn, yLabel) {{
  var cv = document.getElementById(canvasId);
  var w = cv.offsetWidth || 400; var h = cv.offsetHeight || 240;
  cv.width = w; cv.height = h;
  var ctx = cv.getContext("2d");
  var pad = 44;
  var bw = Math.max(1, (w - 2*pad) / values.length - 2);
  var vmax = Math.max(...values.map(Math.abs)) * 1.2 || 1;
  var vmin = Math.min(0, ...values) * 1.2;
  var range = Math.max(vmax - vmin, 1e-9);
  var zero_y = h - pad - (-vmin / range) * (h - 2*pad);
  ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, h-pad); ctx.lineTo(w-pad, h-pad); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad, zero_y); ctx.lineTo(w-pad, zero_y);
  ctx.strokeStyle = "#444c56"; ctx.stroke();
  for (var i = 0; i < values.length; i++) {{
    var x = pad + i * ((w - 2*pad) / values.length) + 1;
    var bh = Math.abs(values[i]) / range * (h - 2*pad);
    var y = values[i] >= 0 ? zero_y - bh : zero_y;
    ctx.fillStyle = colorFn(i, values[i]);
    ctx.fillRect(x, y, bw, bh);
    if (labels && labels[i] !== undefined) {{
      ctx.fillStyle = "#8b949e"; ctx.font = "10px monospace"; ctx.textAlign = "center";
      ctx.fillText(labels[i], x + bw/2, h - pad + 12);
    }}
  }}
  if (yLabel) {{
    ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "left";
    ctx.fillText(yLabel, 4, 14);
  }}
}}

function lineChart(canvasId, xs, ys, color, yLabel, xLabel) {{
  var cv = document.getElementById(canvasId);
  var w = cv.offsetWidth || 400; var h = cv.offsetHeight || 240;
  cv.width = w; cv.height = h;
  var ctx = cv.getContext("2d");
  var pad = 44;
  var xmin = Math.min(...xs), xmax = Math.max(...xs);
  var ymin = Math.min(...ys), ymax = Math.max(...ys);
  var xr = Math.max(xmax - xmin, 1e-9); var yr = Math.max(ymax - ymin, 1e-9) * 1.1;
  ymin -= yr * 0.05; ymax = ymin + yr;
  function px(x) {{ return pad + (x - xmin)/xr * (w - 2*pad); }}
  function py(y) {{ return h - pad - (y - ymin)/yr * (h - 2*pad); }}
  ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, h-pad); ctx.lineTo(w-pad, h-pad); ctx.stroke();
  for (var yi = 0; yi <= 4; yi++) {{
    var gv = ymin + yr * yi/4;
    var gy = py(gv);
    ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(w-pad, gy);
    ctx.strokeStyle = "#21262d"; ctx.stroke();
    ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "right";
    ctx.fillText(gv.toFixed(2), pad - 3, gy + 3);
  }}
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
  for (var i = 0; i < xs.length; i++) {{
    i === 0 ? ctx.moveTo(px(xs[i]), py(ys[i])) : ctx.lineTo(px(xs[i]), py(ys[i]));
  }}
  ctx.stroke();
  for (var xi = 0; xi <= 4; xi++) {{
    var gx = px(xmin + xr * xi/4);
    ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "center";
    ctx.fillText((xmin + xr * xi/4).toFixed(1), gx, h - pad + 12);
  }}
  if (yLabel) {{ ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "left"; ctx.fillText(yLabel, 4, 14); }}
  if (xLabel) {{ ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "center"; ctx.fillText(xLabel, w/2, h - 4); }}
}}

// Spin-spin correlation bar chart
var scLabels = DATA.spin_corr.map(function(_, i) {{ return "r=" + i; }});
barChart("cSpinCorr", scLabels, DATA.spin_corr,
  function(i, v) {{ return v >= 0 ? "#388bfd" : "#f47067"; }}, "⟨S₀·Sᵣ⟩");

// Sweep line charts
if (DATA.sweep && DATA.sweep.length > 0) {{
  var swJz  = DATA.sweep.map(function(p) {{ return p.Jz; }});
  var swSpi = DATA.sweep.map(function(p) {{ return p.structure_pi; }});
  var swGap = DATA.sweep.map(function(p) {{ return p.energy_gap; }});
  var swE0  = DATA.sweep.map(function(p) {{ return p.ground_energy; }});
  lineChart("cSpi",    swJz, swSpi, "#f47067", "S(π)",  "Jz/J");
  lineChart("cGap",    swJz, swGap, "#e3b341", "Gap Δ", "Jz/J");
  lineChart("cEnergy", swJz, swE0,  "#57ab5a", "E₀",    "Jz/J");
}}
</script>
</body>
</html>"""


# ── M28-F: Bose-Hubbard model ─────────────────────────────────────────────────

def _bose_hubbard_frames(
    n_sites: int = 4,
    n_bosons: int = 4,
    n_max: int = 4,
    J: float = 0.1,
    U: float = 1.0,
    mu: float = 0.0,
    scenario: str = "sweep",
    n_sweep: int = 20,
    J_min: float = 0.0,
    J_max: float = 0.5,
) -> dict:
  """Compute Bose-Hubbard model via exact diagonalization (NumPy reference)."""
  import numpy as np

  N = n_sites

  def enum_states(n_sites_loc, n_bosons_loc, n_max_loc):
    result = []
    state = [0] * n_sites_loc
    def dfs(site, remaining):
      if site == n_sites_loc - 1:
        if remaining <= n_max_loc:
          state[site] = remaining
          result.append(tuple(state))
        return
      lo = max(0, remaining - n_max_loc * (n_sites_loc - 1 - site))
      hi = min(remaining, n_max_loc)
      for n in range(lo, hi + 1):
        state[site] = n
        dfs(site + 1, remaining - n)
    dfs(0, n_bosons_loc)
    return sorted(result)

  def build_H(J_val):
    states = enum_states(N, n_bosons, n_max)
    s_map = {s: i for i, s in enumerate(states)}
    dim = len(states)
    H = np.zeros((dim, dim), dtype=float)
    for bi, s in enumerate(states):
      diag = sum(U/2 * s[i]*(s[i]-1) - mu*s[i] for i in range(N))
      H[bi, bi] = diag
      for i in range(N - 1):
        if s[i+1] >= 1:
          s2 = list(s); s2[i] += 1; s2[i+1] -= 1
          t = tuple(s2)
          if t in s_map:
            mel = -J_val * np.sqrt(float(s[i+1]) * float(s[i]+1))
            bj = s_map[t]
            H[bi, bj] += mel
            H[bj, bi] += mel
    return H, states

  def compute_obs(states, psi):
    occ = np.zeros(N)
    double_occ = 0.0
    num_dist = np.zeros(n_max + 1)
    for bi, s in enumerate(states):
      p = psi[bi] ** 2
      for i in range(N):
        occ[i] += p * s[i]
        double_occ += p * s[i] * (s[i]-1)
      if s[0] <= n_max:
        num_dist[s[0]] += p
    double_occ /= N
    return occ, double_occ, num_dist

  def superfluid_order(states, psi):
    s_map = {s: i for i, s in enumerate(states)}
    sf = 0.0
    for site in range(N):
      re_a = 0.0
      for bi, s in enumerate(states):
        if s[site] == 0: continue
        s2 = list(s); mel = np.sqrt(float(s2[site])); s2[site] -= 1; t = tuple(s2)
        if t in s_map:
          re_a += psi[bi] * mel * psi[s_map[t]]
      sf += abs(re_a)
    return sf / N

  if scenario == "sweep":
    sweep = []
    for k in range(n_sweep):
      J_k = J_min + (J_max - J_min) * k / max(n_sweep - 1, 1)
      H, states = build_H(J_k)
      evals, evecs = np.linalg.eigh(H)
      psi = evecs[:, 0]
      _, docc, _ = compute_obs(states, psi)
      sf = superfluid_order(states, psi)
      E1 = evals[1] if len(evals) > 1 else evals[0]
      sweep.append({
          "J": float(J_k),
          "ground_energy": float(evals[0]),
          "energy_gap": float(E1 - evals[0]),
          "superfluid_op": float(sf),
          "double_occupancy": float(docc),
      })
    # Single-point at requested J
    H0, states0 = build_H(J)
    evals0, evecs0 = np.linalg.eigh(H0)
    psi0 = evecs0[:, 0]
    occ0, docc0, ndist0 = compute_obs(states0, psi0)
    sf0 = superfluid_order(states0, psi0)
    return {
        "n_sites": N, "n_bosons": n_bosons, "n_max": n_max,
        "J": J, "U": U, "mu": mu,
        "ground_energy": float(evals0[0]),
        "energy_gap": float(evals0[1] - evals0[0]) if len(evals0) > 1 else 0.0,
        "superfluid_op": float(sf0),
        "double_occupancy": float(docc0),
        "occupation": occ0.tolist(),
        "number_dist": ndist0.tolist(),
        "sweep": sweep,
    }
  else:  # ground_state
    H0, states0 = build_H(J)
    evals0, evecs0 = np.linalg.eigh(H0)
    psi0 = evecs0[:, 0]
    occ0, docc0, ndist0 = compute_obs(states0, psi0)
    sf0 = superfluid_order(states0, psi0)
    return {
        "n_sites": N, "n_bosons": n_bosons, "n_max": n_max,
        "J": J, "U": U, "mu": mu,
        "ground_energy": float(evals0[0]),
        "energy_gap": float(evals0[1] - evals0[0]) if len(evals0) > 1 else 0.0,
        "superfluid_op": float(sf0),
        "double_occupancy": float(docc0),
        "occupation": occ0.tolist(),
        "number_dist": ndist0.tolist(),
        "sweep": [],
    }


def _bose_hubbard_page(data: dict, title: str = "Bose-Hubbard Model") -> str:
  """Generate self-contained HTML for the Bose-Hubbard model."""
  import json as json_mod
  data_json = json_mod.dumps(data)

  return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:'Segoe UI',system-ui,sans-serif;padding:20px}}
h2{{font-size:1.2rem;font-weight:600;color:#58a6ff;margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:960px;margin:0 auto}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
canvas{{display:block;width:100%;height:240px}}
.info{{font-size:0.82rem;color:#8b949e;margin-top:8px}}
.param{{display:flex;gap:24px;flex-wrap:wrap;font-size:0.85rem;color:#8b949e;margin-bottom:16px;max-width:960px;margin-left:auto;margin-right:auto}}
.param span{{color:#58a6ff;font-weight:600}}
</style>
</head>
<body>
<h1 style="font-size:1.4rem;color:#58a6ff;margin-bottom:12px">{title}</h1>
<div class="param">
  <div>N = <span id="pN"></span> sites</div>
  <div>M = <span id="pM"></span> bosons</div>
  <div>n_max = <span id="pNmax"></span></div>
  <div>J = <span id="pJ"></span></div>
  <div>U = <span id="pU"></span></div>
  <div>J/U = <span id="pJU"></span></div>
  <div>E₀ = <span id="pE0"></span></div>
  <div>Δ = <span id="pGap"></span></div>
  <div>|⟨a⟩| = <span id="pSF"></span></div>
  <div>D = <span id="pDocc"></span></div>
</div>
<div class="grid">
  <div class="card"><h2>Site Occupation ⟨nᵢ⟩</h2><canvas id="cOcc"></canvas>
    <div class="info">Uniform in superfluid phase; integer-locked in Mott phase</div></div>
  <div class="card"><h2>Number Distribution P(n) at site 0</h2><canvas id="cNdist"></canvas>
    <div class="info">Sharp Fock peak in Mott phase; broad distribution in superfluid phase</div></div>
  <div class="card"><h2>Superfluid Order Parameter |⟨a⟩| vs J/U</h2><canvas id="cSF"></canvas>
    <div class="info">|⟨a⟩| = 0 in Mott insulator; grows above critical J/U</div></div>
  <div class="card"><h2>Energy Gap Δ vs J/U</h2><canvas id="cGap"></canvas>
    <div class="info">Gap closes at Mott-superfluid transition</div></div>
</div>
<script>
const DATA = {data_json};

document.getElementById("pN").textContent    = DATA.n_sites;
document.getElementById("pM").textContent    = DATA.n_bosons;
document.getElementById("pNmax").textContent = DATA.n_max;
document.getElementById("pJ").textContent    = DATA.J.toFixed(3);
document.getElementById("pU").textContent    = DATA.U.toFixed(3);
document.getElementById("pJU").textContent   = (DATA.U > 0 ? DATA.J / DATA.U : 0).toFixed(3);
document.getElementById("pE0").textContent   = DATA.ground_energy.toFixed(4);
document.getElementById("pGap").textContent  = DATA.energy_gap.toFixed(4);
document.getElementById("pSF").textContent   = DATA.superfluid_op.toFixed(4);
document.getElementById("pDocc").textContent = DATA.double_occupancy.toFixed(4);

function barChart(canvasId, labels, values, colorFn, yLabel) {{
  var cv = document.getElementById(canvasId);
  var w = cv.offsetWidth || 400; var h = cv.offsetHeight || 240;
  cv.width = w; cv.height = h;
  var ctx = cv.getContext("2d");
  var pad = 44;
  var bw = Math.max(1, (w - 2*pad) / Math.max(values.length,1) - 2);
  var vmax = Math.max(...values.map(Math.abs)) * 1.2 || 1;
  var range = Math.max(vmax, 1e-9);
  ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, h-pad); ctx.lineTo(w-pad, h-pad); ctx.stroke();
  for (var i = 0; i < values.length; i++) {{
    var x = pad + i * ((w - 2*pad) / values.length) + 1;
    var bh = values[i] / range * (h - 2*pad);
    ctx.fillStyle = colorFn(i, values[i]);
    ctx.fillRect(x, h - pad - bh, bw, bh);
    if (labels && labels[i] !== undefined) {{
      ctx.fillStyle = "#8b949e"; ctx.font = "10px monospace"; ctx.textAlign = "center";
      ctx.fillText(labels[i], x + bw/2, h - pad + 12);
    }}
  }}
  if (yLabel) {{
    ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "left";
    ctx.fillText(yLabel, 4, 14);
  }}
}}

function lineChart(canvasId, xs, ys, color, yLabel, xLabel) {{
  var cv = document.getElementById(canvasId);
  var w = cv.offsetWidth || 400; var h = cv.offsetHeight || 240;
  cv.width = w; cv.height = h;
  var ctx = cv.getContext("2d");
  var pad = 44;
  var xmin = Math.min(...xs), xmax = Math.max(...xs);
  var ymin = Math.min(0, ...ys), ymax = Math.max(...ys) * 1.1 || 1;
  var xr = Math.max(xmax - xmin, 1e-9); var yr = Math.max(ymax - ymin, 1e-9);
  function px(x) {{ return pad + (x - xmin)/xr * (w - 2*pad); }}
  function py(y) {{ return h - pad - (y - ymin)/yr * (h - 2*pad); }}
  ctx.fillStyle = "#1c2128"; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, h-pad); ctx.lineTo(w-pad, h-pad); ctx.stroke();
  for (var yi = 0; yi <= 4; yi++) {{
    var gv = ymin + yr * yi/4; var gy = py(gv);
    ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(w-pad, gy);
    ctx.strokeStyle = "#21262d"; ctx.stroke();
    ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "right";
    ctx.fillText(gv.toFixed(3), pad - 3, gy + 3);
  }}
  ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 2;
  for (var i = 0; i < xs.length; i++) {{
    i === 0 ? ctx.moveTo(px(xs[i]), py(ys[i])) : ctx.lineTo(px(xs[i]), py(ys[i]));
  }}
  ctx.stroke();
  for (var xi = 0; xi <= 4; xi++) {{
    var gx = px(xmin + xr * xi/4);
    ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace"; ctx.textAlign = "center";
    ctx.fillText((xmin + xr * xi/4).toFixed(3), gx, h - pad + 12);
  }}
  if (yLabel) {{ ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "left"; ctx.fillText(yLabel, 4, 14); }}
  if (xLabel) {{ ctx.fillStyle = "#8b949e"; ctx.font = "10px sans-serif"; ctx.textAlign = "center"; ctx.fillText(xLabel, w/2, h - 4); }}
}}

// Site occupation
var occLabels = DATA.occupation.map(function(_, i) {{ return "i=" + i; }});
barChart("cOcc", occLabels, DATA.occupation, function() {{ return "#57ab5a"; }}, "⟨nᵢ⟩");

// Number distribution
var nLabels = DATA.number_dist.map(function(_, n) {{ return "n=" + n; }});
barChart("cNdist", nLabels, DATA.number_dist, function(i, v) {{ return "#bc8cff"; }}, "P(n)");

// Sweep charts
if (DATA.sweep && DATA.sweep.length > 0) {{
  var swJU = DATA.sweep.map(function(p) {{ return DATA.U > 0 ? p.J / DATA.U : p.J; }});
  var swSF = DATA.sweep.map(function(p) {{ return p.superfluid_op; }});
  var swGap = DATA.sweep.map(function(p) {{ return p.energy_gap; }});
  lineChart("cSF",  swJU, swSF,  "#388bfd", "|⟨a⟩|",  "J/U");
  lineChart("cGap", swJU, swGap, "#e3b341", "Gap Δ",   "J/U");
}}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# M29-G: Transverse-Field Ising  —  Python reference solver + page
# ─────────────────────────────────────────────────────────────────────────────

def _tfi_frames(
    n_sites: int = 10,
    J: float = 1.0,
    h: float = 0.5,
    scenario: str = "sweep",
    n_sweep: int = 30,
    h_min: float = 0.0,
    h_max: float = 2.0,
) -> dict:
    """NumPy exact-diag TFI reference.  Returns dict with sweep + single-point data."""
    import numpy as np  # noqa: PLC0415

    N = n_sites

    def build_h(J_: float, h_: float) -> "np.ndarray":
        dim = 1 << N
        H = np.zeros((dim, dim), dtype=float)
        for s in range(dim):
            # -J Z_i Z_{i+1}
            diag = 0.0
            for i in range(N - 1):
                zi = -1 if (s >> i) & 1 else 1
                zj = -1 if (s >> (i + 1)) & 1 else 1
                diag += -J_ * zi * zj
            H[s, s] = diag
            # -h X_i
            for i in range(N):
                s2 = s ^ (1 << i)
                H[s, s2] -= h_
        return H

    def solve_point(h_: float):
        H = build_h(J, h_)
        evals, evecs = np.linalg.eigh(H)
        psi0 = evecs[:, 0]
        E0, E1 = evals[0], evals[1] if len(evals) > 1 else evals[0]
        # Order parameter |<Z_0>|
        dim = 1 << N
        Z0 = np.array([-1 if s & 1 else 1 for s in range(dim)], dtype=float)
        op = abs(float(psi0 @ (Z0 * psi0)))
        # Spin-spin corr <Z_0 Z_r>
        corr = []
        for r in range(N):
            Zr = np.array([-1 if (s >> r) & 1 else 1 for s in range(dim)], dtype=float)
            corr.append(float(psi0 @ (Z0 * Zr * psi0)))
        # Correlation length
        xi = 0.0
        xs, ys = [], []
        for r in range(1, N // 2):
            c = abs(corr[r])
            if c > 1e-12:
                xs.append(r)
                ys.append(np.log(c))
        if len(xs) >= 2:
            slope, _ = np.polyfit(xs, ys, 1)
            xi = -1.0 / slope if slope < -1e-12 else 0.0
        return E0, E1 - E0, op, xi, corr

    E0, gap, op, xi, spin_corr = solve_point(h)

    sweep = []
    for k in range(n_sweep):
        hk = h_min + (h_max - h_min) * k / (n_sweep - 1)
        e0k, gapk, opk, xik, _ = solve_point(hk)
        sweep.append({"h": hk, "ground_energy": e0k, "energy_gap": gapk,
                      "order_param": opk, "corr_len": xik})

    return {
        "n_sites": N, "J": J, "h": h,
        "ground_energy": E0, "energy_gap": gap,
        "order_param": op, "corr_len": xi,
        "spin_corr": spin_corr,
        "sweep": sweep,
    }


def _tfi_page(data: dict, title: str = "Transverse-Field Ising Model") -> str:
    """4-panel TFI page: spin-spin corr, order param vs h, gap vs h, corr length vs h."""
    import json as _json  # noqa: PLC0415

    jdata = _json.dumps(data)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body{{margin:0;background:#0d1117;color:#e6edf3;font-family:monospace;}}
  h1{{text-align:center;padding:16px;font-size:1.1rem;color:#388bfd;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;}}
  canvas{{width:100%;height:220px;display:block;}}
  .label{{font-size:0.7rem;color:#8b949e;text-align:center;margin-top:4px;}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="grid">
  <div class="panel"><canvas id="cCorr"></canvas><div class="label">Spin-spin correlator ⟨Z₀Zᵣ⟩</div></div>
  <div class="panel"><canvas id="cOP"></canvas><div class="label">Order parameter |⟨Z₀⟩| vs h/J</div></div>
  <div class="panel"><canvas id="cGap"></canvas><div class="label">Energy gap Δ vs h/J (closes at h=J)</div></div>
  <div class="panel"><canvas id="cXi"></canvas><div class="label">Correlation length ξ vs h/J</div></div>
</div>
<script>
var DATA = {jdata};

function lineChart(id, xs, ys, color, ylabel, xlabel) {{
  var cv = document.getElementById(id);
  if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var xmin = Math.min.apply(null,xs), xmax = Math.max.apply(null,xs);
  var ymin = Math.min.apply(null,ys), ymax = Math.max.apply(null,ys);
  var xr = xmax-xmin||1, yr = ymax-ymin||1;
  ctx.strokeStyle = "#21262d"; ctx.lineWidth = 1;
  for (var gi=0; gi<5; gi++) {{
    var gy = pad + (H-2*pad)*gi/4;
    ctx.beginPath(); ctx.moveTo(pad,gy); ctx.lineTo(W-pad,gy); ctx.stroke();
  }}
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
  for (var i=0; i<xs.length; i++) {{
    var px = pad + (xs[i]-xmin)/xr*(W-2*pad);
    var py = H-pad - (ys[i]-ymin)/yr*(H-2*pad);
    if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace";
  ctx.fillText(ylabel,4,14);
  ctx.fillText(xlabel,W-pad,H-4);
  ctx.fillText(ymin.toFixed(3),2,H-pad);
  ctx.fillText(ymax.toFixed(3),2,pad+4);
}}

function barChart(id, labels, vals, colorFn, ylabel) {{
  var cv = document.getElementById(id);
  if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d");
  var W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var n = vals.length;
  var ymax = Math.max.apply(null, vals.map(Math.abs));
  if (ymax === 0) ymax = 1;
  var bw = (W - 2*pad) / n;
  for (var i=0; i<n; i++) {{
    var v = vals[i];
    var bh = Math.abs(v)/ymax*(H-2*pad);
    var bx = pad + i*bw + bw*0.1;
    var by = v >= 0 ? H-pad-bh : H-pad;
    ctx.fillStyle = colorFn(i, v);
    ctx.fillRect(bx, by, bw*0.8, bh);
    if (n <= 16) {{
      ctx.fillStyle="#8b949e"; ctx.font="9px monospace";
      ctx.fillText(labels[i], bx, H-pad+12);
    }}
  }}
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace";
  ctx.fillText(ylabel,4,14);
  // Zero line
  ctx.strokeStyle="#444"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
}}

// Panel 1: spin-spin correlator
var rLabels = DATA.spin_corr.map(function(_,r){{ return "r="+r; }});
barChart("cCorr", rLabels, DATA.spin_corr,
  function(i,v) {{ return v >= 0 ? "#57ab5a" : "#e3b341"; }}, "⟨Z₀Zᵣ⟩");

// Panels 2-4: sweeps
if (DATA.sweep && DATA.sweep.length > 0) {{
  var swH  = DATA.sweep.map(function(p) {{ return p.h / DATA.J; }});
  var swOP = DATA.sweep.map(function(p) {{ return p.order_param; }});
  var swGp = DATA.sweep.map(function(p) {{ return p.energy_gap; }});
  var swXi = DATA.sweep.map(function(p) {{ return p.corr_len; }});
  lineChart("cOP",  swH, swOP, "#57ab5a", "|⟨Z₀⟩|", "h/J");
  lineChart("cGap", swH, swGp, "#e3b341", "Gap Δ",   "h/J");
  lineChart("cXi",  swH, swXi, "#bc8cff", "ξ",       "h/J");
}}
</script>
</body>
</html>"""


