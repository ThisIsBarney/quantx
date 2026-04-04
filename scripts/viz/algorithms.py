from __future__ import annotations

import json
import math
import sys
from pathlib import Path


# ── M22: Quantum Error Correction ─────────────────────────────────────────────

def _qec_frames(
    code: str = "bit_flip_3",
    physical_error_rate: float = 0.05,
    error_type: str = "X",
    n_trials: int = 2000,
    n_detail_trials: int = 64,
    sweep_error_rates: list[float] | None = None,
    seed: int = 42,
) -> dict:
  """Monte Carlo QEC reference simulator.

  Three codes supported:
    bit_flip_3   — 3-qubit repetition code, corrects any single X error.
                   Syndromes: Z₀Z₁ (s0), Z₁Z₂ (s1).
    phase_flip_3 — 3-qubit phase-flip code, corrects any single Z error.
                   Same syndrome table as bit_flip_3 but for Z errors.
    shor_9       — 9-qubit Shor code, corrects any single-qubit Pauli error.
                   Syndromes: 6 Z-type (bit-flip per group) + 2 X-type (phase across groups).

  Theoretical logical error rates:
    Repetition-3: p_log = 3p²(1−p) + p³
    Shor-9 (approx composite): p_log ≈ theory_rep3(theory_rep3(p))

  Returns dict with keys:
    code, n_physical, n_syndrome, physical_error_rate, error_type, n_trials,
    logical_error_rate, logical_error_rate_theory, frames, syndrome_histogram,
    threshold_curve.
  """
  import random as _rnd

  rng = _rnd.Random(seed)

  CODE_PARAMS = {
      "bit_flip_3":   {"n_physical": 3, "n_syndrome": 2},
      "phase_flip_3": {"n_physical": 3, "n_syndrome": 2},
      "shor_9":       {"n_physical": 9, "n_syndrome": 8},
  }
  if code not in CODE_PARAMS:
    raise ValueError(f"Unknown QEC code: {code!r}. Choose bit_flip_3 | phase_flip_3 | shor_9")
  n_physical = CODE_PARAMS[code]["n_physical"]
  n_syndrome = CODE_PARAMS[code]["n_syndrome"]

  # ── Theoretical formulas ────────────────────────────────────────────────────
  def _theory_rep3(p: float) -> float:
    return 3 * p * p * (1 - p) + p * p * p

  def _theory(p: float) -> float:
    if code == "shor_9":
      # Effective per-qubit error rates for each Pauli component.
      p_x = (2 * p / 3) if error_type == "depolarizing" else (p if error_type in ("X",) else 0.0)
      p_z = (2 * p / 3) if error_type == "depolarizing" else (p if error_type in ("Z",) else 0.0)
      # Bit-flip logical: at least one group fails its 3-qubit bit-flip code.
      p_bf = 1.0 - (1.0 - _theory_rep3(p_x)) ** 3 if p_x > 0 else 0.0
      # Phase-flip logical: outer 3-qubit code fails; group effective Z = odd-parity count.
      if p_z > 0:
        p_z_odd = 3 * p_z * (1 - p_z) ** 2 + p_z ** 3
        p_pf = _theory_rep3(p_z_odd)
      else:
        p_pf = 0.0
      return p_bf + p_pf - p_bf * p_pf
    return _theory_rep3(p)

  # ── Single-trial helpers ─────────────────────────────────────────────────────
  def _lookup3(e0: bool, e1: bool, e2: bool) -> tuple[int, int, int]:
    """Z0Z1 / Z1Z2 parity syndrome + correction qubit for repetition codes."""
    s0 = e0 ^ e1
    s1 = e1 ^ e2
    c = {(False, False): -1, (True, False): 0, (True, True): 1, (False, True): 2}[(s0, s1)]
    return int(s0), int(s1), c

  def _trial_bit_flip(p: float) -> tuple[str, str, str, bool]:
    e = [rng.random() < p for _ in range(3)]
    s0, s1, c = _lookup3(e[0], e[1], e[2])
    res = e[:]
    if c >= 0:
      res[c] = not res[c]
    ok = sum(res) < 2
    err = "".join(f"X{q}" for q in range(3) if e[q]) or "I"
    cor = f"X{c}" if c >= 0 else "I"
    return err, f"{s0}{s1}", cor, ok

  def _trial_phase_flip(p: float) -> tuple[str, str, str, bool]:
    e = [rng.random() < p for _ in range(3)]
    s0, s1, c = _lookup3(e[0], e[1], e[2])
    res = e[:]
    if c >= 0:
      res[c] = not res[c]
    ok = sum(res) < 2
    err = "".join(f"Z{q}" for q in range(3) if e[q]) or "I"
    cor = f"Z{c}" if c >= 0 else "I"
    return err, f"{s0}{s1}", cor, ok

  def _trial_shor(p: float) -> tuple[str, str, str, bool]:
    # Sample errors
    eX = [False] * 9
    eZ = [False] * 9
    if error_type == "X":
      eX = [rng.random() < p for _ in range(9)]
    elif error_type == "Z":
      eZ = [rng.random() < p for _ in range(9)]
    else:  # depolarizing
      for q in range(9):
        r = rng.random()
        if r < p / 3:
          eX[q] = True
        elif r < 2 * p / 3:
          eZ[q] = True
        elif r < p:
          eX[q] = True
          eZ[q] = True

    # Bit-flip syndromes (Z-stabilizers, detect X per group)
    bf_syn = []
    for g in range(3):
      b = 3 * g
      bf_syn.append(eX[b] ^ eX[b + 1])
      bf_syn.append(eX[b + 1] ^ eX[b + 2])

    # Phase-flip syndromes (X-stabilizers, detect Z across groups)
    gzp = [sum(int(eZ[3 * g + k]) for k in range(3)) % 2 for g in range(3)]
    ps0 = gzp[0] ^ gzp[1]
    ps1 = gzp[1] ^ gzp[2]
    syndrome_str = "".join(map(str, bf_syn + [ps0, ps1]))

    # Bit-flip correction per group
    resX = eX[:]
    for g in range(3):
      b = 3 * g
      _, _, c = _lookup3(eX[b], eX[b + 1], eX[b + 2])
      if c >= 0:
        resX[b + c] = not resX[b + c]

    # Phase-flip correction (Z on first qubit of identified group)
    resZ = eZ[:]
    pf_lookup = {(False, False): -1, (True, False): 0, (True, True): 1, (False, True): 2}
    cg = pf_lookup[(bool(ps0), bool(ps1))]
    if cg >= 0:
      resZ[3 * cg] = not resZ[3 * cg]

    # Logical error check
    bf_err = any(sum(int(resX[3 * g + k]) for k in range(3)) >= 2 for g in range(3))
    pf_err = sum(sum(int(resZ[3 * g + k]) for k in range(3)) % 2 for g in range(3)) >= 2
    ok = not bf_err and not pf_err

    parts_x = [f"X{q}" for q in range(9) if eX[q]]
    parts_z = [f"Z{q}" for q in range(9) if eZ[q]]
    err = " ".join(parts_x + parts_z) or "I"
    return err, syndrome_str, "decoded", ok

  # ── Monte Carlo runner ───────────────────────────────────────────────────────
  def _run_mc(p_phys: float, n: int, collect_detail: bool) -> tuple[float, list, dict]:
    successes = 0
    frames: list[dict] = []
    syn_hist: dict[str, dict] = {}

    for t in range(n):
      if code == "bit_flip_3":
        err_s, syn_s, cor_s, ok = _trial_bit_flip(p_phys)
      elif code == "phase_flip_3":
        err_s, syn_s, cor_s, ok = _trial_phase_flip(p_phys)
      else:
        err_s, syn_s, cor_s, ok = _trial_shor(p_phys)

      if ok:
        successes += 1
      if collect_detail and t < n_detail_trials:
        frames.append({"trial": t, "error": err_s, "syndrome": syn_s,
                        "correction": cor_s, "succeeded": ok})
      if syn_s not in syn_hist:
        syn_hist[syn_s] = {"count": 0, "successes": 0}
      syn_hist[syn_s]["count"] += 1
      if ok:
        syn_hist[syn_s]["successes"] += 1

    p_log = 1.0 - successes / n if n > 0 else 0.0
    return p_log, frames, syn_hist

  p_log, frames, syn_hist = _run_mc(physical_error_rate, n_trials, collect_detail=True)

  # ── Threshold sweep ─────────────────────────────────────────────────────────
  threshold_curve: list[dict] = []
  if sweep_error_rates:
    sweep_n = max(500, n_trials // 4)
    for p_sw in sweep_error_rates:
      p_log_mc, _, _ = _run_mc(p_sw, sweep_n, collect_detail=False)
      threshold_curve.append({
          "p_physical": p_sw,
          "p_logical_mc": p_log_mc,
          "p_logical_theory": _theory(p_sw),
      })

  return {
      "code": code,
      "n_physical": n_physical,
      "n_syndrome": n_syndrome,
      "physical_error_rate": physical_error_rate,
      "error_type": error_type,
      "n_trials": n_trials,
      "n_detail_trials": min(n_detail_trials, n_trials),
      "logical_error_rate": p_log,
      "logical_error_rate_theory": _theory(physical_error_rate),
      "frames": frames,
      "syndrome_histogram": syn_hist,
      "threshold_curve": threshold_curve,
  }


def _qec_page(data: dict, title: str = "QEC") -> str:
  """Generate a self-contained HTML page for QEC Monte Carlo results."""
  import json as _json

  data_js = _json.dumps(data, ensure_ascii=False)
  code_label = {
      "bit_flip_3":   "3-Qubit Bit-Flip Code",
      "phase_flip_3": "3-Qubit Phase-Flip Code",
      "shor_9":       "9-Qubit Shor Code",
  }.get(data.get("code", ""), data.get("code", ""))

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --accent:#58a6ff;
           --green:#3fb950; --red:#f85149; --orange:#d29922; --muted:#8b949e;
           --text:#e6edf3; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
          min-height:100vh; padding:24px; }}
  h1 {{ font-size:1.4rem; font-weight:600; color:var(--accent); margin-bottom:4px; }}
  .subtitle {{ color:var(--muted); font-size:.85rem; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; max-width:1140px; margin:0 auto; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px;
            padding:16px; }}
  .panel.wide {{ grid-column:span 2; }}
  .panel h2 {{ font-size:.95rem; color:var(--muted); margin-bottom:12px; font-weight:500; }}
  canvas {{ display:block; width:100%; }}
  .stat-row {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:18px;
               max-width:1140px; margin-left:auto; margin-right:auto; margin-bottom:18px; }}
  .stat {{ background:var(--panel); border:1px solid var(--border); border-radius:8px;
           padding:8px 14px; min-width:120px; }}
  .stat-label {{ font-size:.75rem; color:var(--muted); margin-bottom:2px; }}
  .stat-value {{ font-size:1.1rem; font-weight:600; color:var(--accent); }}
  .stat-value.good {{ color:var(--green); }}
  .stat-value.warn {{ color:var(--orange); }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">{code_label} — Monte Carlo reference simulation</p>

<div class="stat-row" id="stats"></div>

<div class="grid">
  <div class="panel">
    <h2>Trial Grid (first {min(data.get('n_detail_trials', 64), len(data.get('frames', [])))} trials)</h2>
    <canvas id="cvGrid" height="200"></canvas>
  </div>
  <div class="panel">
    <h2>Syndrome Success Rate</h2>
    <canvas id="cvSyndrome" height="200"></canvas>
  </div>
  <div class="panel wide" id="thresholdPanel" style="display:none">
    <h2>Logical Error Rate vs Physical Error Rate (Threshold Curve)</h2>
    <canvas id="cvThreshold" height="240"></canvas>
  </div>
</div>

<script>
const DATA = {data_js};

// ── Stats ─────────────────────────────────────────────────────────────────────
(function() {{
  const s = document.getElementById('stats');
  function stat(label, val, cls) {{
    const d = document.createElement('div'); d.className = 'stat';
    d.innerHTML = `<div class="stat-label">${{label}}</div>
                   <div class="stat-value ${{cls||''}}">${{val}}</div>`;
    s.appendChild(d);
  }}
  stat('Code', DATA.code.replace(/_/g,' '));
  stat('n_physical', DATA.n_physical);
  stat('n_syndrome', DATA.n_syndrome);
  stat('p_physical', DATA.physical_error_rate.toFixed(3));
  stat('Error type', DATA.error_type);
  stat('n_trials', DATA.n_trials);
  const ler = DATA.logical_error_rate;
  const cls = ler < DATA.physical_error_rate ? 'good' : 'warn';
  stat('p_logical (MC)', ler.toFixed(4), cls);
  stat('p_logical (theory)', DATA.logical_error_rate_theory.toFixed(4));
  const gain = DATA.physical_error_rate > 0
    ? (DATA.physical_error_rate / Math.max(ler, 1e-9)).toFixed(1) + '×'
    : '—';
  stat('Error reduction', gain, ler < DATA.physical_error_rate ? 'good' : 'warn');
}})();

// ── Trial grid ─────────────────────────────────────────────────────────────────
function drawGrid() {{
  const cv = document.getElementById('cvGrid');
  const dpr = window.devicePixelRatio||1;
  cv.width  = (cv.offsetWidth||500)*dpr;
  cv.height = (cv.offsetHeight||200)*dpr;
  const ctx = cv.getContext('2d'); ctx.scale(dpr,dpr);
  const W = cv.offsetWidth||500, H = cv.offsetHeight||200;
  ctx.fillStyle='#161b22'; ctx.fillRect(0,0,W,H);

  const frames = DATA.frames||[];
  if (!frames.length) return;
  const n = frames.length;
  const cols = Math.ceil(Math.sqrt(n * W / H));
  const rows = Math.ceil(n / cols);
  const cellW = (W-4) / cols, cellH = (H-4) / rows;
  const pad = 1;

  for (let i=0; i<n; i++) {{
    const f = frames[i];
    const col = i % cols, row = Math.floor(i/cols);
    const x = 2 + col*cellW, y = 2 + row*cellH;
    const hasErr = f.error !== 'I';
    const ok = f.succeeded;
    ctx.fillStyle = ok ? (hasErr ? '#2ea04326' : '#3fb95033') : '#f8514966';
    if (!ok) ctx.fillStyle = '#f85149cc';
    else if (hasErr) ctx.fillStyle = '#3fb95088';
    else ctx.fillStyle = '#30363d';
    ctx.fillRect(x+pad, y+pad, cellW-2*pad, cellH-2*pad);
  }}

  // Legend
  ctx.font = '10px system-ui'; ctx.textAlign='left';
  const ly = H-6;
  const legend = [['#3fb95088','Corrected'], ['#f85149cc','Logical error'], ['#30363d','No error']];
  let lx = 4;
  for (const [c, lab] of legend) {{
    ctx.fillStyle=c; ctx.fillRect(lx, ly-8, 10, 10);
    ctx.fillStyle='#8b949e'; ctx.fillText(' '+lab, lx+12, ly);
    lx += ctx.measureText(' '+lab).width + 20;
  }}
}}

// ── Syndrome success-rate bar chart ────────────────────────────────────────────
function drawSyndrome() {{
  const cv = document.getElementById('cvSyndrome');
  const dpr = window.devicePixelRatio||1;
  cv.width  = (cv.offsetWidth||500)*dpr;
  cv.height = (cv.offsetHeight||200)*dpr;
  const ctx = cv.getContext('2d'); ctx.scale(dpr,dpr);
  const W = cv.offsetWidth||500, H = cv.offsetHeight||200;
  ctx.fillStyle='#161b22'; ctx.fillRect(0,0,W,H);

  const hist = DATA.syndrome_histogram||{{}};
  const entries = Object.entries(hist).sort((a,b)=>b[1].count-a[1].count);
  if (!entries.length) return;

  const pad = {{l:40,r:10,t:12,b:36}};
  const cw = W-pad.l-pad.r, ch = H-pad.t-pad.b;
  const n = entries.length;
  const barW = Math.max(1, cw/n - 2);

  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch);
  ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();

  // Y gridlines at 25%, 50%, 75%, 100%
  ctx.fillStyle='#8b949e'; ctx.font='9px system-ui'; ctx.textAlign='right';
  for (let t=0; t<=4; t++) {{
    const y = pad.t + ch - (t/4)*ch;
    ctx.fillText((t*25)+'%', pad.l-3, y+3);
    ctx.strokeStyle='#21262d'; ctx.lineWidth=0.5;
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+cw,y); ctx.stroke();
  }}

  for (let i=0; i<n; i++) {{
    const [syn, stats] = entries[i];
    const rate = stats.count>0 ? stats.successes/stats.count : 0;
    const x = pad.l + i*(cw/n);
    const bh = rate*ch;
    ctx.fillStyle = rate > 0.9 ? '#3fb950' : rate > 0.5 ? '#d29922' : '#f85149';
    ctx.fillRect(x+1, pad.t+ch-bh, barW, bh);
    // Label
    ctx.fillStyle='#8b949e'; ctx.font='8px system-ui'; ctx.textAlign='center';
    if (barW > 12) ctx.fillText(syn, x+barW/2+1, pad.t+ch+12);
  }}
  ctx.fillStyle='#8b949e'; ctx.font='10px system-ui'; ctx.textAlign='center';
  ctx.fillText('Syndrome pattern', pad.l+cw/2, H-2);
}}

// ── Threshold curve ────────────────────────────────────────────────────────────
function drawThreshold() {{
  const tc = DATA.threshold_curve||[];
  if (!tc.length) return;
  document.getElementById('thresholdPanel').style.display='';
  const cv = document.getElementById('cvThreshold');
  const dpr = window.devicePixelRatio||1;
  cv.width  = (cv.offsetWidth||900)*dpr;
  cv.height = (cv.offsetHeight||240)*dpr;
  const ctx = cv.getContext('2d'); ctx.scale(dpr,dpr);
  const W = cv.offsetWidth||900, H = cv.offsetHeight||240;
  ctx.fillStyle='#161b22'; ctx.fillRect(0,0,W,H);

  const pad = {{l:52,r:20,t:14,b:40}};
  const cw = W-pad.l-pad.r, ch = H-pad.t-pad.b;

  const pMax = Math.max(...tc.map(p=>p.p_physical), 0.5);
  function xOf(p) {{ return pad.l + (p/pMax)*cw; }}
  function yOf(p) {{ return pad.t + ch - (p/pMax)*ch; }}

  // Axes
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ch);
  ctx.lineTo(pad.l+cw,pad.t+ch); ctx.stroke();

  // Grid + ticks
  ctx.fillStyle='#8b949e'; ctx.font='10px system-ui';
  for (let t=0; t<=4; t++) {{
    const p = t/4*pMax;
    const y = yOf(p), x = xOf(p);
    ctx.textAlign='right'; ctx.fillText(p.toFixed(2), pad.l-4, y+3);
    ctx.strokeStyle='#21262d'; ctx.lineWidth=0.5;
    ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+cw,y); ctx.stroke();
    ctx.textAlign='center'; ctx.fillText(p.toFixed(2), x, pad.t+ch+14);
    ctx.beginPath(); ctx.moveTo(x,pad.t); ctx.lineTo(x,pad.t+ch); ctx.stroke();
  }}

  // Diagonal p_log = p_phys (no improvement)
  ctx.strokeStyle='#30363d66'; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(xOf(0),yOf(0)); ctx.lineTo(xOf(pMax),yOf(pMax)); ctx.stroke();
  ctx.setLineDash([]);

  // Theory curve
  ctx.strokeStyle='#58a6ff'; ctx.lineWidth=2;
  ctx.beginPath();
  for (let i=0; i<tc.length; i++) {{
    const px = xOf(tc[i].p_physical), py = yOf(tc[i].p_logical_theory);
    if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();

  // MC dots
  ctx.fillStyle='#d29922';
  for (const pt of tc) {{
    ctx.beginPath(); ctx.arc(xOf(pt.p_physical), yOf(pt.p_logical_mc), 4, 0, 2*Math.PI); ctx.fill();
  }}
  // MC line
  ctx.strokeStyle='#d29922'; ctx.lineWidth=1.5;
  ctx.beginPath();
  for (let i=0; i<tc.length; i++) {{
    const px = xOf(tc[i].p_physical), py = yOf(tc[i].p_logical_mc);
    if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();

  // Labels
  ctx.fillStyle='#8b949e'; ctx.font='10px system-ui'; ctx.textAlign='center';
  ctx.fillText('Physical error rate p', pad.l+cw/2, H-4);
  ctx.save(); ctx.translate(13,pad.t+ch/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign='center'; ctx.fillText('Logical error rate', 0, 0); ctx.restore();

  // Legend
  ctx.textAlign='left'; ctx.font='11px system-ui';
  ctx.strokeStyle='#58a6ff'; ctx.lineWidth=2; ctx.beginPath();
  ctx.moveTo(pad.l+4, pad.t+10); ctx.lineTo(pad.l+22, pad.t+10); ctx.stroke();
  ctx.fillStyle='#e6edf3'; ctx.fillText(' Theory', pad.l+24, pad.t+14);
  ctx.fillStyle='#d29922'; ctx.beginPath(); ctx.arc(pad.l+4+100+9, pad.t+10, 4, 0, 2*Math.PI); ctx.fill();
  ctx.fillStyle='#e6edf3'; ctx.fillText(' Monte Carlo', pad.l+4+100+15, pad.t+14);
  ctx.strokeStyle='#30363d66'; ctx.lineWidth=1; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(pad.l+4+230, pad.t+10); ctx.lineTo(pad.l+22+230, pad.t+10); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#e6edf3'; ctx.fillText(' No correction', pad.l+24+230, pad.t+14);
}}

window.addEventListener('load', () => {{ drawGrid(); drawSyndrome(); drawThreshold(); }});
window.addEventListener('resize', () => {{ drawGrid(); drawSyndrome(); drawThreshold(); }});
</script>
</body>
</html>"""
  return html


# ── M24: φ⁴ Scalar Quantum Field Theory ───────────────────────────────────────


# ── M21: Quantum Phase Estimation ─────────────────────────────────────────────

def _qpe_frames(
    n_ancilla: int,
    unitary: "np.ndarray",        # 2^n_target × 2^n_target unitary matrix
    eigenstate: "np.ndarray",     # 2^n_target initial target state (normalized)
    n_shots: int = 4096,
    true_phase: float | None = None,
    sweep_ancilla: list[int] | None = None,
    seed: int = 42,
) -> dict:
  """Full-matrix QPE reference solver using numpy FFT.

  Algorithm:
    1. Construct the (2^n_ancilla × 2^n_target) state |+⟩^n_anc ⊗ |eigenstate⟩
    2. Apply controlled-U^{2^k} for k = 0..n_ancilla-1 to build the phase-kickback state
    3. Apply inverse QFT on ancilla register via np.fft.fft / sqrt(N) (NOT ifft — sign convention)
    4. Marginalise over target register → phase probability distribution over 2^n_ancilla bins
    5. Sample n_shots from exact probabilities

  Returns dict with keys: n_ancilla, n_target, phase_probs, shot_counts, estimated_phase,
    true_phase, sweep (list of {n_ancilla, estimated_phase, error} for each ancilla in sweep_ancilla).
  """
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("QPE visualization requires numpy") from ex

  rng = np.random.default_rng(seed)

  dim_tgt = unitary.shape[0]
  n_target = int(round(np.log2(dim_tgt)))
  dim_anc = 1 << n_ancilla

  def run_qpe_exact(n_anc: int) -> "np.ndarray":
    """Return exact phase probabilities (length 2^n_anc) for given n_anc."""
    d_anc = 1 << n_anc
    # Build ancilla state: uniform superposition (|+⟩^n_anc) via equal-amplitude vector of length d_anc.
    # Combined state is d_anc × dim_tgt matrix: row a = amplitude for ancilla=a.
    state_mat = np.zeros((d_anc, dim_tgt), dtype=complex)
    # All ancilla basis states get equal amplitude times eigenstate.
    for a in range(d_anc):
      state_mat[a, :] = eigenstate / np.sqrt(d_anc)

    # Apply controlled-U^{2^k}: ancilla qubit k (in binary, MSB = qubit 0) controls U^{2^k}.
    # For each basis state |a⟩ of ancilla (integer a), each ancilla qubit k contributes a phase
    # if bit k of a is 1 (qubit k measured as 1 means U^{2^k} was applied).
    # The ancilla basis state |a⟩ accumulates phase e^{i * phase_angle * a} if eigenvalue is e^{i*phi}
    # and U|ψ⟩ = e^{i*phi}|ψ⟩.
    # Build U^{2^k} powers iteratively.
    u_power = unitary.copy()  # U^{2^0} = U
    for k in range(n_anc):
      # Ancilla qubit k controls U^{2^k}: bit k of integer a determines whether to apply u_power.
      # This gives phase e^{i2πφ·a} for |a⟩ (since Σ_k bit_k(a)*2^k = a).
      for a in range(d_anc):
        if (a >> k) & 1:
          state_mat[a, :] = u_power @ state_mat[a, :]
      u_power = u_power @ u_power  # U^{2^{k+1}}

    # Inverse QFT on ancilla via fft (NOT ifft — QFT uses e^{+i2πjk/N} = fft convention).
    # QFT†|a⟩ = (1/√N) Σ_k e^{-i2πak/N} |k⟩  →  fft along axis 0 divided by √N.
    state_mat = np.fft.fft(state_mat, axis=0) / np.sqrt(d_anc)

    # Marginalise over target register: prob[a] = Σ_t |state_mat[a,t]|^2
    probs = np.sum(np.abs(state_mat) ** 2, axis=1)
    # Normalise (should already be ~1).
    probs = np.real(probs)
    probs = np.clip(probs, 0.0, None)
    probs /= probs.sum()
    return probs

  # Main QPE run.
  phase_probs = run_qpe_exact(n_ancilla)

  # Shot sampling.
  shot_counts = [0] * dim_anc
  if n_shots > 0:
    samples = rng.choice(dim_anc, size=n_shots, p=phase_probs)
    for s in samples:
      shot_counts[int(s)] += 1

  peak_bin = int(np.argmax(phase_probs))
  estimated_phase = peak_bin / dim_anc

  # Sweep over ancilla counts if requested.
  sweep = []
  if sweep_ancilla:
    for n_anc_s in sweep_ancilla:
      if n_anc_s < 1:
        continue
      probs_s = run_qpe_exact(n_anc_s)
      d_s = 1 << n_anc_s
      peak_s = int(np.argmax(probs_s))
      est_s = peak_s / d_s
      entry: dict = {"n_ancilla": n_anc_s, "estimated_phase": est_s}
      if true_phase is not None:
        err = abs(est_s - true_phase)
        # Handle wrap-around.
        err = min(err, 1.0 - err)
        entry["error"] = err
      sweep.append(entry)

  return {
      "n_ancilla": n_ancilla,
      "n_target": n_target,
      "phase_probs": phase_probs.tolist(),
      "shot_counts": shot_counts,
      "estimated_phase": estimated_phase,
      "true_phase": true_phase,
      "sweep": sweep,
      "n_shots": n_shots,
  }


def _qpe_page(data: dict, title: str = "QPE") -> str:
  """Generate a self-contained HTML page for QPE results."""
  import json as _json

  data_js = _json.dumps(data, ensure_ascii=False)
  true_phase = data.get("true_phase")
  true_phase_js = "null" if true_phase is None else str(true_phase)

  html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  :root {{ --bg:#0d1117; --panel:#161b22; --border:#30363d; --accent:#58a6ff;
           --green:#3fb950; --orange:#d29922; --muted:#8b949e; --text:#e6edf3; }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',system-ui,sans-serif;
          min-height:100vh; padding:24px; }}
  h1 {{ font-size:1.4rem; font-weight:600; color:var(--accent); margin-bottom:4px; }}
  .subtitle {{ color:var(--muted); font-size:.85rem; margin-bottom:20px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; max-width:1100px; margin:0 auto; }}
  .panel {{ background:var(--panel); border:1px solid var(--border); border-radius:10px;
            padding:16px; }}
  .panel.wide {{ grid-column:span 2; }}
  .panel h2 {{ font-size:.95rem; color:var(--muted); margin-bottom:12px; font-weight:500; }}
  canvas {{ display:block; width:100%; }}
  .stat-row {{ display:flex; gap:24px; flex-wrap:wrap; margin-bottom:16px; }}
  .stat {{ background:var(--bg); border:1px solid var(--border); border-radius:8px;
           padding:8px 14px; }}
  .stat-label {{ font-size:.75rem; color:var(--muted); margin-bottom:2px; }}
  .stat-value {{ font-size:1.1rem; font-weight:600; color:var(--accent); }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">Quantum Phase Estimation — Python Reference Solver</p>

<div class="stat-row" id="stats"></div>

<div class="grid">
  <div class="panel">
    <h2>Exact Phase Probability Distribution</h2>
    <canvas id="cvPhase" height="260"></canvas>
  </div>
  <div class="panel">
    <h2>Shot Histogram (n={data.get('n_shots', 0)} samples)</h2>
    <canvas id="cvShots" height="260"></canvas>
  </div>
  <div class="panel wide" id="sweepPanel" style="display:none">
    <h2>Estimation Error vs n_ancilla (precision bits)</h2>
    <canvas id="cvPrecision" height="220"></canvas>
  </div>
</div>

<script>
const DATA = {data_js};
const TRUE_PHASE = {true_phase_js};

// ── Stats row ──────────────────────────────────────────────────────────────────
(function() {{
  const s = document.getElementById('stats');
  function stat(label, val) {{
    const d = document.createElement('div');
    d.className = 'stat';
    d.innerHTML = `<div class="stat-label">${{label}}</div><div class="stat-value">${{val}}</div>`;
    s.appendChild(d);
  }}
  stat('n_ancilla', DATA.n_ancilla);
  stat('n_target', DATA.n_target);
  stat('Estimated φ', DATA.estimated_phase.toFixed(6));
  if (TRUE_PHASE !== null) stat('True φ', TRUE_PHASE.toFixed(6));
  if (TRUE_PHASE !== null) {{
    let err = Math.abs(DATA.estimated_phase - TRUE_PHASE);
    err = Math.min(err, 1 - err);
    stat('|Error|', err.toExponential(3));
  }}
  stat('n_shots', DATA.n_shots);
}})();

// ── Helpers ────────────────────────────────────────────────────────────────────
function drawBarChart(canvasId, values, labels, opts) {{
  const cv = document.getElementById(canvasId);
  const dpr = window.devicePixelRatio || 1;
  cv.width  = cv.offsetWidth  * dpr || 500 * dpr;
  cv.height = cv.offsetHeight * dpr || 260 * dpr;
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = cv.offsetWidth || 500, H = cv.offsetHeight || 260;
  ctx.fillStyle = '#161b22'; ctx.fillRect(0, 0, W, H);

  const pad = {{ l:48, r:16, t:12, b:36 }};
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  const n = values.length;
  if (!n) return;
  const maxV = Math.max(...values, 1e-9);
  const barW = Math.max(1, cw / n - 1);

  for (let i = 0; i < n; i++) {{
    const x = pad.l + i * (cw / n);
    const bh = (values[i] / maxV) * ch;
    let color = opts.barColor || '#58a6ff';
    if (opts.highlightEst && i === opts.peakBin) color = '#d29922';
    if (opts.highlightTrue && TRUE_PHASE !== null) {{
      const trueBin = Math.round(TRUE_PHASE * n) % n;
      if (i === trueBin) color = '#3fb950';
    }}
    ctx.fillStyle = color;
    ctx.fillRect(x + 1, pad.t + ch - bh, barW, bh);
  }}

  // Axes.
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t + ch);
  ctx.lineTo(pad.l + cw, pad.t + ch); ctx.stroke();

  // Y ticks.
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
  for (let t = 0; t <= 4; t++) {{
    const y = pad.t + ch - (t / 4) * ch;
    ctx.fillText((maxV * t / 4).toFixed(3), pad.l - 4, y + 3);
    ctx.strokeStyle = '#21262d';
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + cw, y); ctx.stroke();
  }}

  // X labels — show every Nth.
  ctx.fillStyle = '#8b949e'; ctx.textAlign = 'center';
  const step = Math.max(1, Math.floor(n / 8));
  for (let i = 0; i < n; i += step) {{
    const x = pad.l + i * (cw / n) + barW / 2;
    ctx.fillText(i, x, pad.t + ch + 14);
  }}

  // Legend.
  if (opts.highlightEst || opts.highlightTrue) {{
    ctx.textAlign = 'left'; ctx.font = '11px system-ui';
    let lx = pad.l + 4, ly = pad.t + 14;
    if (opts.highlightEst) {{
      ctx.fillStyle = '#d29922'; ctx.fillRect(lx, ly - 8, 10, 10);
      ctx.fillStyle = '#e6edf3'; ctx.fillText(' Estimated peak', lx + 12, ly);
      lx += 130;
    }}
    if (opts.highlightTrue && TRUE_PHASE !== null) {{
      ctx.fillStyle = '#3fb950'; ctx.fillRect(lx, ly - 8, 10, 10);
      ctx.fillStyle = '#e6edf3'; ctx.fillText(' True phase bin', lx + 12, ly);
    }}
  }}
}}

// ── Phase probability chart ────────────────────────────────────────────────────
function drawPhase() {{
  const probs = DATA.phase_probs;
  const peak = probs.indexOf(Math.max(...probs));
  drawBarChart('cvPhase', probs, null, {{
    barColor: '#58a6ff',
    highlightEst: true, peakBin: peak,
    highlightTrue: true,
  }});
}}

// ── Shot histogram ─────────────────────────────────────────────────────────────
function drawShots() {{
  const counts = DATA.shot_counts;
  const peak = counts.indexOf(Math.max(...counts));
  drawBarChart('cvShots', counts, null, {{
    barColor: '#58a6ff',
    highlightEst: true, peakBin: peak,
    highlightTrue: true,
  }});
}}

// ── Precision vs n_ancilla sweep ───────────────────────────────────────────────
function drawSweep() {{
  const sweep = DATA.sweep || [];
  if (!sweep.length || sweep[0].error === undefined) return;
  document.getElementById('sweepPanel').style.display = '';
  const cv = document.getElementById('cvPrecision');
  const dpr = window.devicePixelRatio || 1;
  cv.width  = cv.offsetWidth  * dpr || 900 * dpr;
  cv.height = cv.offsetHeight * dpr || 220 * dpr;
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = cv.offsetWidth || 900, H = cv.offsetHeight || 220;
  ctx.fillStyle = '#161b22'; ctx.fillRect(0, 0, W, H);

  const pad = {{ l:60, r:20, t:16, b:36 }};
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;

  // Theoretical curve: error ≤ 1/(2^n_anc).
  const ns = sweep.map(s => s.n_ancilla);
  const errs = sweep.map(s => s.error);
  const maxN = Math.max(...ns), minN = Math.min(...ns);
  const maxErr = Math.max(...errs, 0.01);

  function xOf(n) {{ return pad.l + (n - minN) / Math.max(maxN - minN, 1) * cw; }}
  function yOf(e) {{ return pad.t + ch - Math.log(e / maxErr) / Math.log(0.001 / maxErr) * ch; }}
  // Simple linear-y fallback.
  function yLin(e) {{ return pad.t + ch - (e / maxErr) * ch; }}

  // Theoretical 1/2^n line.
  ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
  ctx.beginPath();
  for (let i = 0; i < ns.length; i++) {{
    const th = 1.0 / (1 << ns[i]);
    const x = xOf(ns[i]), y = yLin(Math.min(th, maxErr));
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  ctx.stroke(); ctx.setLineDash([]);

  // Actual errors.
  ctx.strokeStyle = '#d29922'; ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < ns.length; i++) {{
    const x = xOf(ns[i]), y = yLin(errs[i]);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }}
  ctx.stroke();
  for (let i = 0; i < ns.length; i++) {{
    ctx.fillStyle = '#d29922';
    ctx.beginPath(); ctx.arc(xOf(ns[i]), yLin(errs[i]), 4, 0, 2*Math.PI); ctx.fill();
  }}

  // Axes & labels.
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, pad.t+ch);
  ctx.lineTo(pad.l+cw, pad.t+ch); ctx.stroke();
  ctx.fillStyle = '#8b949e'; ctx.font = '10px system-ui'; ctx.textAlign = 'center';
  for (let i = 0; i < ns.length; i++) {{
    ctx.fillText(ns[i], xOf(ns[i]), pad.t+ch+14);
  }}
  ctx.fillText('n_ancilla (precision bits)', pad.l + cw/2, H-4);
  ctx.textAlign = 'right';
  for (let t = 0; t <= 4; t++) {{
    const e = maxErr * (1 - t/4);
    ctx.fillText(e.toExponential(1), pad.l-4, yLin(e)+3);
  }}
  ctx.save(); ctx.translate(14, pad.t+ch/2); ctx.rotate(-Math.PI/2);
  ctx.textAlign = 'center'; ctx.fillText('|error|', 0, 0); ctx.restore();

  // Legend.
  ctx.textAlign = 'left'; ctx.font = '11px system-ui';
  ctx.fillStyle = '#d29922'; ctx.fillRect(pad.l+8, pad.t+4, 12, 3);
  ctx.fillStyle = '#e6edf3'; ctx.fillText(' Actual error', pad.l+22, pad.t+9);
  ctx.fillStyle = '#3fb950'; ctx.fillRect(pad.l+8+110, pad.t+4, 12, 3);
  ctx.fillStyle = '#e6edf3'; ctx.fillText(' 1/2ⁿ bound', pad.l+22+110, pad.t+9);
}}

window.addEventListener('load', () => {{
  drawPhase();
  drawShots();
  drawSweep();
}});
window.addEventListener('resize', () => {{
  drawPhase();
  drawShots();
  drawSweep();
}});
</script>
</body>
</html>"""
  return html


# ── M20: VQE / QAOA variational optimizer ─────────────────────────────────────

def _vqe_frames(
    n_qubits: int,
    cost_hamiltonian: list[dict],
    ansatz_type: str = "hardware_efficient",
    n_layers: int = 2,
    optimizer: str = "adam",
    max_iterations: int = 120,
    learning_rate: float = 0.1,
    initial_params: list[float] | None = None,
    seed: int = 42,
) -> dict:
  """Numpy VQE reference solver.

  Returns a dict with keys:
    frames        list of {iteration, energy, gradient_norm, params}
    optimal_energy  float
    optimal_params  list[float]
    converged       bool
    n_qubits        int
    ansatz_type     str
    n_layers        int
    n_params        int
  """
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("VQE visualization requires numpy") from ex

  # ── Gate matrices ──────────────────────────────────────────────────────────
  I2 = np.eye(2, dtype=complex)
  X  = np.array([[0, 1], [1, 0]], dtype=complex)
  Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
  Z  = np.array([[1, 0], [0, -1]], dtype=complex)
  H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
  CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=complex).reshape(4, 4)

  PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}

  def kron_n(ops: list) -> np.ndarray:
    """Tensor-product of a list of 2×2 matrices → 2^n × 2^n."""
    result = ops[0]
    for op in ops[1:]:
      result = np.kron(result, op)
    return result

  def pauli_string_matrix(label: str, n: int) -> np.ndarray:
    """Build 2^n × 2^n matrix for a Pauli string like 'X0Z1'."""
    import re
    ops = ["I"] * n
    for m in re.finditer(r"([XYZI])(\d+)", label.upper()):
      p, q = m.group(1), int(m.group(2))
      if q < n:
        ops[q] = p
    return kron_n([PAULI[p] for p in ops])

  # ── Build cost Hamiltonian matrix ──────────────────────────────────────────
  dim = 1 << n_qubits
  H_cost = np.zeros((dim, dim), dtype=complex)
  for term in cost_hamiltonian:
    label = term.get("label", "")
    coeff = float(term.get("coefficient", 0.0))
    if not label or label.upper() == "I":
      H_cost += coeff * np.eye(dim, dtype=complex)
    else:
      H_cost += coeff * pauli_string_matrix(label, n_qubits)

  # ── Ansatz circuits ────────────────────────────────────────────────────────
  def ry_mat(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

  def rx_mat(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

  def rz_mat(theta: float) -> np.ndarray:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)

  def single_qubit_op(gate_2x2: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """Embed a 2×2 gate on `qubit` into the n-qubit space."""
    ops = [I2] * n
    ops[qubit] = gate_2x2
    return kron_n(ops)

  def two_qubit_cnot(ctrl: int, targ: int, n: int) -> np.ndarray:
    """Exact CNOT(ctrl, targ) in n-qubit space."""
    dim_n = 1 << n
    mat = np.eye(dim_n, dtype=complex)
    for idx in range(dim_n):
      if (idx >> (n - 1 - ctrl)) & 1:
        flipped = idx ^ (1 << (n - 1 - targ))
        mat[:, idx] = 0
        mat[flipped, idx] = 1.0
    return mat

  def build_hea_state(params: np.ndarray) -> np.ndarray:
    """Hardware-efficient ansatz: n_layers × (RY on each qubit + CNOT chain) + final RY."""
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    p = 0
    for _ in range(n_layers):
      for q in range(n_qubits):
        state = single_qubit_op(ry_mat(params[p]), q, n_qubits) @ state
        p += 1
      for q in range(n_qubits - 1):
        state = two_qubit_cnot(q, q + 1, n_qubits) @ state
    for q in range(n_qubits):
      state = single_qubit_op(ry_mat(params[p]), q, n_qubits) @ state
      p += 1
    return state

  def build_qaoa_state(params: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    """QAOA: |+⟩^n + p rounds of cost + mixer."""
    state = np.zeros(dim, dtype=complex)
    state[0] = 1.0
    # Prepare |+⟩^n.
    for q in range(n_qubits):
      state = single_qubit_op(H_gate, q, n_qubits) @ state
    for layer in range(n_layers):
      gamma = params[2 * layer]
      beta  = params[2 * layer + 1]
      # Cost unitary: CNOT·RZ(γ)·CNOT per edge → exp(-iγ/2 Z_iZ_j).
      # This keeps the gate in the standard form exp(-iθG/2) with θ=γ, G=ZiZj,
      # so the parameter-shift rule ∂E/∂γ = (E(γ+π/2)-E(γ-π/2))/2 is valid.
      for (qi, qj) in edges:
        cx = two_qubit_cnot(qi, qj, n_qubits)
        state = cx @ state
        state = single_qubit_op(rz_mat(gamma), qj, n_qubits) @ state
        state = cx @ state
      # Mixer: RX(β) on each qubit → exp(-iβ/2 Xi), standard form with θ=β.
      for q in range(n_qubits):
        state = single_qubit_op(rx_mat(beta), q, n_qubits) @ state
    return state

  # Detect QAOA mode from cost_hamiltonian structure.
  is_qaoa = (ansatz_type == "qaoa")
  # Extract QAOA edges from ZiZj terms in cost_hamiltonian.
  import re as _re
  qaoa_edges: list[tuple[int, int]] = []
  if is_qaoa:
    for term in cost_hamiltonian:
      lbl = term.get("label", "")
      m = _re.match(r"^Z(\d+)Z(\d+)$", lbl.upper())
      if m:
        qaoa_edges.append((int(m.group(1)), int(m.group(2))))

  if is_qaoa:
    n_params = 2 * n_layers
  else:
    n_params = (n_layers + 1) * n_qubits

  def build_state(params: np.ndarray) -> np.ndarray:
    if is_qaoa:
      return build_qaoa_state(params, qaoa_edges)
    return build_hea_state(params)

  def eval_energy(params: np.ndarray) -> float:
    psi = build_state(params)
    return float(np.real(psi.conj() @ H_cost @ psi))

  # ── Gradient ───────────────────────────────────────────────────────────────
  # HEA: parameter-shift rule (each param appears in exactly one gate).
  # QAOA: parameter-shift is invalid for shared parameters (γ/β appear across
  #       all edge/qubit gates in the same round).  Use central finite differences.
  if is_qaoa:
    FD_EPS = 1e-3  # small finite-difference step

    def gradient(params: np.ndarray) -> np.ndarray:
      grad = np.zeros(n_params)
      for k in range(n_params):
        tp = params.copy(); tp[k] += FD_EPS
        tm = params.copy(); tm[k] -= FD_EPS
        grad[k] = (eval_energy(tp) - eval_energy(tm)) / (2.0 * FD_EPS)
      return grad
  else:
    def gradient(params: np.ndarray) -> np.ndarray:
      shift = np.pi / 2
      grad = np.zeros(n_params)
      for k in range(n_params):
        tp = params.copy(); tp[k] += shift
        tm = params.copy(); tm[k] -= shift
        grad[k] = (eval_energy(tp) - eval_energy(tm)) / 2.0
      return grad

  # ── Adam optimizer ─────────────────────────────────────────────────────────
  rng = np.random.default_rng(seed)
  if initial_params is not None:
    params = np.array(initial_params, dtype=float)
  elif is_qaoa:
    # QAOA: initialize γ ∈ [0, π/2] and β ∈ [0, π/4] to avoid barren plateaus.
    # Interleaved layout: [γ₁, β₁, γ₂, β₂, ...]
    params = np.zeros(n_params)
    for layer in range(n_layers):
      params[2 * layer]     = rng.uniform(0.05, np.pi / 2)   # γ
      params[2 * layer + 1] = rng.uniform(0.05, np.pi / 4)   # β
  else:
    params = rng.uniform(0, 2 * np.pi, n_params)

  b1, b2, eps = 0.9, 0.999, 1e-8
  m_adam = np.zeros(n_params)
  v_adam = np.zeros(n_params)
  t_adam = 0

  frames = []
  converged = False
  prev_energy = eval_energy(params)

  for it in range(max_iterations):
    energy = eval_energy(params)
    grad = gradient(params)
    gnorm = float(np.linalg.norm(grad))

    frames.append({
      "iteration": it,
      "energy": float(energy),
      "gradient_norm": gnorm,
      "params": params.tolist(),
    })

    if gnorm < 1e-5 or (it > 0 and abs(energy - prev_energy) < 1e-7):
      converged = True
      break

    if optimizer == "gradient_descent":
      params = params - learning_rate * grad
    else:
      # Adam
      t_adam += 1
      m_adam = b1 * m_adam + (1 - b1) * grad
      v_adam = b2 * v_adam + (1 - b2) * grad ** 2
      m_hat = m_adam / (1 - b1 ** t_adam)
      v_hat = v_adam / (1 - b2 ** t_adam)
      params = params - learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    prev_energy = energy

  # Final energy.
  final_energy = eval_energy(params)
  frames.append({
    "iteration": len(frames),
    "energy": float(final_energy),
    "gradient_norm": 0.0,
    "params": params.tolist(),
  })

  return {
    "frames": frames,
    "optimal_energy": float(final_energy),
    "optimal_params": params.tolist(),
    "converged": converged,
    "n_qubits": n_qubits,
    "ansatz_type": ansatz_type,
    "n_layers": n_layers,
    "n_params": n_params,
  }


def _vqe_page(data: dict) -> str:
  """HTML page with VQE/QAOA optimization visualization:
    - Energy convergence curve
    - Gradient norm convergence curve
    - Parameter trajectory heatmap
    - Final-state probability distribution
  """
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("VQE page requires numpy") from ex

  frames = data["frames"]
  optimal_energy = data["optimal_energy"]
  converged = data["converged"]
  n_params = data["n_params"]
  ansatz_type = data["ansatz_type"]
  n_qubits = data["n_qubits"]
  n_layers = data["n_layers"]

  iterations   = [f["iteration"] for f in frames]
  energies     = [f["energy"]    for f in frames]
  grad_norms   = [f["gradient_norm"] for f in frames]
  params_mat   = [f["params"]    for f in frames]  # list of lists

  # Rebuild final state for probability distribution.
  import re as _re
  I2 = np.eye(2, dtype=complex)
  X  = np.array([[0, 1], [1, 0]], dtype=complex)
  Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
  Z  = np.array([[1, 0], [0, -1]], dtype=complex)
  H_gate = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
  PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}
  dim = 1 << n_qubits

  def kron_n(ops):
    r = ops[0]
    for o in ops[1:]: r = np.kron(r, o)
    return r

  def ry_mat(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

  def rx_mat(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)

  def rz_mat(theta):
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]).astype(complex)

  def sq(gate, q, n):
    ops = [I2] * n
    ops[q] = gate
    return kron_n(ops)

  def cx_mat(ctrl, targ, n):
    d = 1 << n
    mat = np.eye(d, dtype=complex)
    for idx in range(d):
      if (idx >> (n - 1 - ctrl)) & 1:
        fl = idx ^ (1 << (n - 1 - targ))
        mat[:, idx] = 0
        mat[fl, idx] = 1.0
    return mat

  best_params = np.array(data["optimal_params"])
  is_qaoa = (ansatz_type == "qaoa")

  if is_qaoa:
    # Extract edges from cost_hamiltonian if available (stored in data).
    qaoa_edges = data.get("qaoa_edges", [])
    state = np.zeros(dim, dtype=complex); state[0] = 1.0
    for q in range(n_qubits): state = sq(H_gate, q, n_qubits) @ state
    for layer in range(n_layers):
      gamma = best_params[2 * layer]
      beta  = best_params[2 * layer + 1]
      for (qi, qj) in qaoa_edges:
        cx = cx_mat(qi, qj, n_qubits)
        state = cx @ state
        state = sq(rz_mat(gamma), qj, n_qubits) @ state
        state = cx @ state
      for q in range(n_qubits):
        state = sq(rx_mat(beta), q, n_qubits) @ state
  else:
    state = np.zeros(dim, dtype=complex); state[0] = 1.0
    p = 0
    for _ in range(n_layers):
      for q in range(n_qubits):
        state = sq(ry_mat(best_params[p]), q, n_qubits) @ state; p += 1
      for q in range(n_qubits - 1):
        state = cx_mat(q, q + 1, n_qubits) @ state
    for q in range(n_qubits):
      state = sq(ry_mat(best_params[p]), q, n_qubits) @ state; p += 1

  probs = np.abs(state) ** 2
  bitstrings = [format(i, f"0{n_qubits}b") for i in range(dim)]

  # ── Build JSON data blobs ─────────────────────────────────────────────────
  import json as _json
  energy_series  = _json.dumps(list(zip(iterations, energies)))
  grad_series    = _json.dumps(list(zip(iterations, grad_norms)))
  params_json    = _json.dumps(params_mat)
  probs_json     = _json.dumps(list(zip(bitstrings, [float(p) for p in probs])))
  param_labels   = _json.dumps([f"θ{k}" for k in range(n_params)])
  iter_labels    = _json.dumps(iterations)

  title_suffix = "QAOA" if is_qaoa else "VQE"
  conv_str = "converged" if converged else "max iterations"
  status_line = f"optimal energy = {optimal_energy:.6f} ({conv_str})"

  return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>QuantX {title_suffix} Optimization</title>
<style>
  body{{font-family:sans-serif;background:#0a0a14;color:#e0e0f0;margin:0;padding:20px;}}
  h1{{color:#a0c8ff;text-align:center;font-size:1.4em;margin-bottom:4px;}}
  .subtitle{{text-align:center;color:#7090b0;font-size:0.9em;margin-bottom:20px;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:18px;max-width:1100px;margin:0 auto;}}
  .panel{{background:#111827;border:1px solid #1e3050;border-radius:8px;padding:14px;}}
  canvas{{width:100%;}}
  h2{{color:#60a8e0;font-size:1em;margin:0 0 10px;}}
</style>
</head>
<body>
<h1>QuantX — {title_suffix} Optimization ({n_qubits} qubits, {n_layers} layers)</h1>
<div class="subtitle">{status_line}</div>
<div class="grid">
  <div class="panel">
    <h2>Energy Convergence</h2>
    <canvas id="cvEnergy" height="220"></canvas>
  </div>
  <div class="panel">
    <h2>Gradient Norm</h2>
    <canvas id="cvGrad" height="220"></canvas>
  </div>
  <div class="panel">
    <h2>Parameter Trajectory Heatmap (θ vs iteration)</h2>
    <canvas id="cvParams" height="220"></canvas>
  </div>
  <div class="panel">
    <h2>Final State Probability Distribution</h2>
    <canvas id="cvProbs" height="220"></canvas>
  </div>
</div>
<script>
const energySeries  = {energy_series};
const gradSeries    = {grad_series};
const paramsMat     = {params_json};
const probsData     = {probs_json};
const paramLabels   = {param_labels};
const iterLabels    = {iter_labels};

// ── Utilities ──────────────────────────────────────────────────────────────
function linspace(a, b, n) {{
  if (n < 2) return [a];
  return Array.from({{length: n}}, (_, i) => a + i * (b - a) / (n - 1));
}}

function drawLineChart(cvId, series, xlabel, ylabel, color) {{
  const cv = document.getElementById(cvId);
  const W = cv.clientWidth, H = 220;
  cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d");
  const pad = {{l:50, r:15, t:10, b:36}};
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0, 0, W, H);
  if (!series.length) return;
  const xs = series.map(p => p[0]), ys = series.map(p => p[1]);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys) - 0.05 * (Math.max(...ys) - Math.min(...ys) + 1e-9);
  const yMax = Math.max(...ys) + 0.05 * (Math.max(...ys) - Math.min(...ys) + 1e-9);
  const tx = x => pad.l + (x - xMin) / (xMax - xMin + 1e-9) * pw;
  const ty = y => pad.t + ph - (y - yMin) / (yMax - yMin + 1e-9) * ph;
  ctx.strokeStyle = "#1e3050"; ctx.lineWidth = 1;
  [0.0, 0.25, 0.5, 0.75, 1.0].forEach(r => {{
    const x = pad.l + r * pw, y = pad.t + r * ph;
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + ph); ctx.stroke();
  }});
  ctx.strokeStyle = color; ctx.lineWidth = 2;
  ctx.beginPath();
  series.forEach(([x, y], i) => i === 0 ? ctx.moveTo(tx(x), ty(y)) : ctx.lineTo(tx(x), ty(y)));
  ctx.stroke();
  // Axis labels.
  ctx.fillStyle = "#7090b0"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
  ctx.fillText(xlabel, pad.l + pw / 2, H - 4);
  ctx.save(); ctx.translate(12, pad.t + ph / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText(ylabel, 0, 0); ctx.restore();
  ctx.textAlign = "right";
  [yMin, yMax].forEach(v => ctx.fillText(v.toFixed(3), pad.l - 4, ty(v) + 4));
}}

function drawParamHeatmap(cvId) {{
  const cv = document.getElementById(cvId);
  const W = cv.clientWidth, H = 220;
  cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0, 0, W, H);
  const pad = {{l:50, r:15, t:10, b:36}};
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const nIter = paramsMat.length, nP = paramsMat[0].length;
  if (!nIter || !nP) return;
  // Flatten to find global min/max for color scaling.
  const allVals = paramsMat.flat();
  const vMin = Math.min(...allVals), vMax = Math.max(...allVals);
  const cellW = pw / nIter, cellH = ph / nP;
  for (let i = 0; i < nIter; ++i) {{
    for (let p = 0; p < nP; ++p) {{
      const t = (paramsMat[i][p] - vMin) / (vMax - vMin + 1e-9);
      const r = Math.round(20 + t * 180), g = Math.round(60 + t * 80), b = Math.round(180 - t * 80);
      ctx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
      ctx.fillRect(pad.l + i * cellW, pad.t + p * cellH, cellW + 1, cellH + 1);
    }}
  }}
  ctx.fillStyle = "#7090b0"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("iteration", pad.l + pw / 2, H - 4);
  ctx.save(); ctx.translate(12, pad.t + ph / 2); ctx.rotate(-Math.PI / 2);
  ctx.fillText("parameter", 0, 0); ctx.restore();
}}

function drawProbBars(cvId) {{
  const cv = document.getElementById(cvId);
  const W = cv.clientWidth, H = 220;
  cv.width = W; cv.height = H;
  const ctx = cv.getContext("2d");
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0, 0, W, H);
  const pad = {{l:30, r:10, t:10, b:42}};
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const n = probsData.length;
  const barW = pw / n - 2;
  const pMax = Math.max(...probsData.map(d => d[1]));
  probsData.forEach(([bs, prob], i) => {{
    const barH = ph * prob / (pMax + 1e-9);
    const x = pad.l + i * (pw / n) + 1;
    const y = pad.t + ph - barH;
    const t = prob / (pMax + 1e-9);
    ctx.fillStyle = `rgb(${{Math.round(30 + t * 100)}}, ${{Math.round(100 + t * 100)}}, ${{Math.round(200 + t * 55)}})`;
    ctx.fillRect(x, y, barW, barH);
    if (n <= 16) {{
      ctx.fillStyle = "#7090b0"; ctx.font = "9px sans-serif";
      ctx.textAlign = "center"; ctx.save();
      ctx.translate(x + barW / 2, pad.t + ph + 2);
      ctx.rotate(-Math.PI / 3);
      ctx.fillText("|" + bs + "⟩", 0, 0);
      ctx.restore();
    }}
  }});
  ctx.fillStyle = "#7090b0"; ctx.font = "11px sans-serif"; ctx.textAlign = "center";
  ctx.fillText("basis state", pad.l + pw / 2, H - 2);
}}

window.addEventListener("load", () => {{
  drawLineChart("cvEnergy", energySeries, "iteration", "⟨H⟩", "#4ca3dd");
  drawLineChart("cvGrad",   gradSeries,   "iteration", "|∇E|", "#e87040");
  drawParamHeatmap("cvParams");
  drawProbBars("cvProbs");
}});
</script>
</body>
</html>"""


