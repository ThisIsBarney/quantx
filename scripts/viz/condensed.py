from __future__ import annotations

import json
import math
import sys
from pathlib import Path


# ── M25: Unified Dashboard ────────────────────────────────────────────────────

def _dashboard_page(pages: dict) -> str:  # noqa: PLR0912
  """Single-page unified dashboard embedding all QuantX scenario visualizations.

  Args:
      pages: Mapping of scenario-key (str) → full HTML string for that scenario.

  Returns:
      Self-contained HTML string for the dashboard.
  """
  CATEGORY_DEF = [
      ("🧮 量子基础", [
          ("rabi",               "Rabi 振荡"),
          ("entanglement_rabi",  "纠缠演化 (Rabi)"),
          ("entanglement_decay", "纠缠衰减"),
      ]),
      ("🌊 空间波函数", [
          ("spatial",        "1D 波包"),
          ("spatial2d",      "2D 波包"),
          ("spatial3d",      "3D 波包"),
          ("driven_barrier", "驱动势垒"),
          ("bloch",          "Bloch 振荡"),
          ("gaussian_pulse", "高斯脉冲"),
      ]),
      ("⚛️ 相对论性", [
          ("dirac2d", "2D Dirac 旋量"),
      ]),
      ("📊 相空间", [
          ("wigner_free",    "Wigner 自由"),
          ("wigner_barrier", "Wigner 势垒"),
      ]),
      ("🔗 开放量子系统", [
          ("lindblad", "Lindblad 主方程"),
      ]),
      ("🕸️ 张量网络", [
          ("mps",       "MPS 诊断"),
          ("mps_ising", "MPS Ising"),
          ("mps_xx",    "MPS XX"),
      ]),
      ("🔬 量子算法", [
          ("vqe_ising",       "VQE (Ising)"),
          ("vqe_qaoa",        "QAOA MaxCut"),
          ("qpe_phase",       "QPE 相位门"),
          ("qpe_hamiltonian", "QPE 哈密顿量"),
          ("qec_bit_flip",    "QEC 比特翻转"),
          ("qec_phase_flip",  "QEC 相位翻转"),
          ("qec_shor",        "QEC Shor 码"),
      ]),
      ("🌌 量子场论", [
          ("phi4_symmetric",     "φ⁴ 对称相"),
          ("phi4_broken",        "φ⁴ 破缺相"),
          ("phi4_sweep",         "φ⁴ 相变扫描"),
          ("phi4_dynamics",      "φ⁴ 量子淬火动力学"),
          ("schwinger_vacuum",   "Schwinger 真空"),
          ("hubbard",            "Hubbard Mott 绝缘体"),
          ("floquet",            "Floquet 周期驱动"),
          ("heisenberg",         "XXZ Heisenberg 链"),
          ("bose_hubbard",       "Bose-Hubbard 超流/Mott"),
          ("tfi",                "横场 Ising 量子相变"),
          ("ssh",                "SSH 拓扑绝缘体"),
          ("kitaev",             "Kitaev 链 Majorana 模式"),
          ("anderson",           "Anderson 局域化"),
          ("chern",              "2D Chern 绝缘体 (QWZ)"),
          ("hofstadter",         "Hofstadter 蝴蝶"),
          ("mbl",                "MBL 多体局域化 (XXZ+disorder)"),
          ("spectral",           "Spectral S(k,ω) 动力学结构因子"),
          ("tenfold",            "10-Fold Way AZ 拓扑分类"),
          ("second_chern",       "Second Chern Number 4D QHE"),
          ("wilson_loop",        "Wilson Loop WCC BHZ Z₂ TI"),
          ("edge_states",        "Edge States / Surface Spectrum BHZ"),
          ("haldane",            "Haldane Model QAH C₁=±1"),
          ("weyl",               "Weyl Semimetal Fermi Arcs"),
          ("fu_kane",            "Fu-Kane SC Majorana C=1"),
          ("lutchyn_oreg",       "Lutchyn-Oreg Majorana Wire Z\u2082"),
          ("kitaev_honeycomb",   "Kitaev Honeycomb Majorana Z\u2082"),
          ("bbh",                "BBH Quadrupole Insulator HOTI"),
          ("floquet_topo",       "Floquet Topological Anomalous Edge"),
          ("non_herm",           "Non-Hermitian Skin Effect EP"),
          ("aah",                "Aubry-Andre-Harper Quasicrystal MIT"),
          ("syk",                "Sachdev-Ye-Kitaev Quantum Chaos SFF"),
          ("kondo",              "Kondo Effect SIAM Impurity Screening"),
          ("rg",                 "Richardson-Gaudin BCS Pairing Condensate"),
          ("ent",                "Bipartite Entanglement Entropy Area Law"),
          ("schwinger_string",   "Schwinger 弦张力"),
          ("schwinger_dynamics", "Schwinger 对产生"),
      ]),
  ]

  CAT_ICONS = {
      "🧮 量子基础":    "🧮",
      "🌊 空间波函数":  "🌊",
      "⚛️ 相对论性":    "⚛️",
      "📊 相空间":      "📊",
      "🔗 开放量子系统":"🔗",
      "🕸️ 张量网络":    "🕸️",
      "🔬 量子算法":    "🔬",
      "🌌 量子场论":    "🌌",
  }

  # Build sidebar HTML and lookup maps.
  nav_html = ""
  key_to_label: dict = {}
  key_to_cat: dict = {}
  key_to_icon: dict = {}
  first_key = None
  available_keys: list = []

  for cat_name, items in CATEGORY_DEF:
    icon = CAT_ICONS.get(cat_name, "📊")
    items_html = ""
    for key, label in items:
      if key not in pages:
        continue
      if first_key is None:
        first_key = key
      available_keys.append(key)
      key_to_label[key] = label
      key_to_cat[key] = cat_name
      key_to_icon[key] = icon
      items_html += (
          f'<button class="nav-item" data-key="{key}" '
          f'data-cat="{cat_name}" data-label="{label}" data-icon="{icon}">'
          f'{label}</button>\n'
      )
    if items_html:
      nav_html += (
          f'<div class="nav-group">'
          f'<div class="nav-cat">{cat_name}</div>'
          f'{items_html}</div>\n'
      )

  def _safe_json_for_script(data: object) -> str:
    # NOTE:
    #   This JSON is inlined into an HTML <script> tag below. Raw "</script>"
    #   inside scenario HTML would terminate the script element early during HTML
    #   parsing, even if it appears inside a JavaScript string literal. Escaping
    #   "<" prevents accidental closing tags while preserving JSON semantics.
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")

  total = len(available_keys)
  pages_json  = _safe_json_for_script(pages)
  labels_json = _safe_json_for_script(key_to_label)
  cats_json   = _safe_json_for_script(key_to_cat)
  icons_json  = _safe_json_for_script(key_to_icon)
  first_json  = _safe_json_for_script(first_key or "")

  return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>QuantX Dashboard</title>
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;overflow:hidden;background:#0d1117;color:#e6edf3;
  font-family:'Segoe UI',system-ui,-apple-system,sans-serif;font-size:14px}}
.shell{{display:flex;height:100vh}}
/* ── Sidebar ────────────────────────────────────── */
.sidebar{{
  width:220px;min-width:170px;flex-shrink:0;
  background:#161b22;border-right:1px solid #30363d;
  display:flex;flex-direction:column;overflow:hidden
}}
.sidebar-header{{
  padding:13px 13px 9px;border-bottom:1px solid #30363d;flex-shrink:0
}}
.logo{{font-size:16px;font-weight:700;color:#58a6ff;letter-spacing:.4px;
  display:flex;align-items:center;gap:6px}}
.logo-badge{{
  background:#21262d;border:1px solid #30363d;border-radius:9px;
  padding:1px 7px;font-size:10px;color:#8b949e;font-weight:400
}}
.logo-sub{{font-size:10px;color:#6e7681;margin-top:2px;margin-bottom:9px}}
.search{{
  width:100%;padding:5px 9px;background:#21262d;border:1px solid #30363d;
  border-radius:6px;color:#e6edf3;font-size:12px;outline:none;
  transition:border-color .15s
}}
.search::placeholder{{color:#484f58}}
.search:focus{{border-color:#58a6ff}}
/* ── Nav scroll ──────────────────────────────────── */
.nav-scroll{{flex:1;overflow-y:auto;padding:6px 0}}
.nav-scroll::-webkit-scrollbar{{width:4px}}
.nav-scroll::-webkit-scrollbar-thumb{{background:#30363d;border-radius:2px}}
.nav-group{{margin-bottom:2px}}
.nav-cat{{
  padding:9px 13px 3px;font-size:10px;font-weight:600;
  color:#6e7681;text-transform:uppercase;letter-spacing:.9px;
  user-select:none
}}
.nav-item{{
  display:block;width:100%;padding:5px 13px 5px 19px;
  background:none;border:none;border-left:2px solid transparent;
  color:#c9d1d9;font-size:12.5px;text-align:left;cursor:pointer;
  transition:background .1s,color .1s;line-height:1.45
}}
.nav-item:hover{{background:#21262d;color:#e6edf3}}
.nav-item.active{{
  background:#1c2d3d;color:#58a6ff;
  border-left-color:#58a6ff;font-weight:500
}}
.nav-item.hidden{{display:none!important}}
.nav-group.all-hidden{{display:none}}
/* ── Viewport ────────────────────────────────────── */
.viewport{{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}}
.vbar{{
  height:40px;flex-shrink:0;background:#161b22;border-bottom:1px solid #30363d;
  display:flex;align-items:center;gap:7px;padding:0 13px
}}
.vbar-icon{{font-size:15px;flex-shrink:0;line-height:1}}
.vbar-title{{font-weight:600;font-size:13px;color:#e6edf3;flex-shrink:0}}
.vbar-cat{{
  font-size:10.5px;color:#8b949e;padding:1px 7px;
  background:#21262d;border:1px solid #30363d;border-radius:9px;flex-shrink:0
}}
.vbar-sep{{flex:1}}
.vbar-hint{{font-size:11px;color:#484f58}}
.btn-newtab{{
  background:#21262d;border:1px solid #30363d;border-radius:6px;
  color:#8b949e;cursor:pointer;padding:3px 9px;font-size:11.5px;
  transition:background .1s,color .1s;flex-shrink:0
}}
.btn-newtab:hover{{background:#2d333b;color:#e6edf3}}
/* ── Main frame / empty state ────────────────────── */
.empty{{
  flex:1;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:12px;color:#484f58
}}
.empty-icon{{font-size:52px;opacity:.35}}
.empty-msg{{font-size:13px}}
.empty-hint{{font-size:11px;color:#30363d}}
.main-frame{{
  flex:1;border:none;width:100%;height:100%;background:#0d1117;display:block
}}
.main-frame.off{{display:none}}
</style>
</head>
<body>
<div class="shell">
  <nav class="sidebar">
    <div class="sidebar-header">
      <div class="logo">QuantX <span class="logo-badge">{total}</span></div>
      <div class="logo-sub">Visualization Dashboard</div>
      <input class="search" id="search" type="text"
             placeholder="搜索场景… (按 / 聚焦)" autocomplete="off"/>
    </div>
    <div class="nav-scroll" id="nav-scroll">
{nav_html}
    </div>
  </nav>
  <div class="viewport">
    <div class="vbar">
      <span class="vbar-icon" id="vbar-icon">⚛️</span>
      <span class="vbar-title" id="vbar-title">QuantX Dashboard</span>
      <span class="vbar-cat"  id="vbar-cat" style="display:none"></span>
      <span class="vbar-sep"></span>
      <span class="vbar-hint" id="vbar-hint">{total} 个场景 · ↑↓ 切换 · / 搜索</span>
      <button class="btn-newtab" id="btn-newtab"
              title="在新标签页中打开当前场景" style="display:none">⧉ 新标签页</button>
    </div>
    <div class="empty" id="empty">
      <span class="empty-icon">⚛️</span>
      <span class="empty-msg">从左侧选择一个场景开始可视化</span>
      <span class="empty-hint">↑ ↓ 键盘导航 · / 快速搜索</span>
    </div>
    <iframe class="main-frame off" id="main-frame"
            sandbox="allow-scripts"
            title="场景可视化"></iframe>
  </div>
</div>
<script>
(function(){{
"use strict";
var PAGES  = {pages_json};
var LABELS = {labels_json};
var CATS   = {cats_json};
var ICONS  = {icons_json};
var current = null;

var frame    = document.getElementById("main-frame");
var empty    = document.getElementById("empty");
var vbarIcon = document.getElementById("vbar-icon");
var vbarTitle= document.getElementById("vbar-title");
var vbarCat  = document.getElementById("vbar-cat");
var vbarHint = document.getElementById("vbar-hint");
var btnNew   = document.getElementById("btn-newtab");
var search   = document.getElementById("search");

function allItems() {{
  return Array.from(document.querySelectorAll(".nav-item"));
}}

function switchTo(key) {{
  if (!PAGES[key] || key === current) return;
  current = key;
  frame.srcdoc = PAGES[key];
  frame.classList.remove("off");
  empty.style.display = "none";
  vbarIcon.textContent  = ICONS[key]  || "📊";
  vbarTitle.textContent = LABELS[key] || key;
  var cat = CATS[key] || "";
  if (cat) {{ vbarCat.textContent = cat; vbarCat.style.display = ""; }}
  else      {{ vbarCat.style.display = "none"; }}
  btnNew.style.display = "";
  vbarHint.style.display = "none";
  allItems().forEach(function(b) {{
    b.classList.toggle("active", b.dataset.key === key);
  }});
  var active = document.querySelector(".nav-item.active");
  if (active) active.scrollIntoView({{block:"nearest",behavior:"smooth"}});
}}

// Open current page in a new browser tab via Blob URL.
btnNew.addEventListener("click", function() {{
  if (!current || !PAGES[current]) return;
  var blob = new Blob([PAGES[current]], {{type:"text/html;charset=utf-8"}});
  var url  = URL.createObjectURL(blob);
  var tab  = window.open(url, "_blank");
  if (!tab) alert("请允许弹出窗口以在新标签页中打开场景。");
  setTimeout(function() {{ URL.revokeObjectURL(url); }}, 15000);
}});

// Sidebar click.
document.querySelectorAll(".nav-item").forEach(function(btn) {{
  btn.addEventListener("click", function() {{ switchTo(btn.dataset.key); }});
}});

// Keyboard navigation.
document.addEventListener("keydown", function(e) {{
  if (document.activeElement === search) {{
    if (e.key === "Escape") {{ search.value = ""; filterNav(""); search.blur(); }}
    return;
  }}
  if (e.key === "/" || e.key === "f") {{
    e.preventDefault(); search.focus(); return;
  }}
  var visible = allItems().filter(function(b) {{
    return !b.classList.contains("hidden");
  }});
  var idx = visible.findIndex(function(b) {{ return b.dataset.key === current; }});
  if (e.key === "ArrowDown") {{
    e.preventDefault();
    if (idx < visible.length - 1) switchTo(visible[idx + 1].dataset.key);
  }} else if (e.key === "ArrowUp") {{
    e.preventDefault();
    if (idx > 0) switchTo(visible[idx - 1].dataset.key);
  }}
}});

// Search / filter sidebar items.
function filterNav(q) {{
  q = q.trim().toLowerCase();
  document.querySelectorAll(".nav-group").forEach(function(grp) {{
    var any = false;
    grp.querySelectorAll(".nav-item").forEach(function(btn) {{
      var match = !q ||
        (btn.dataset.label || "").toLowerCase().includes(q) ||
        (btn.dataset.key   || "").toLowerCase().includes(q) ||
        (btn.dataset.cat   || "").toLowerCase().includes(q);
      btn.classList.toggle("hidden", !match);
      if (match) any = true;
    }});
    grp.classList.toggle("all-hidden", !any);
  }});
}}
search.addEventListener("input", function() {{ filterNav(search.value); }});

// Load the first available scenario on start.
switchTo({first_json});
}})();
</script>
</body>
</html>"""



# ─────────────────────────────────────────────────────────────────────────────
# M29-H: SSH Model  —  Python reference solver + page
# ─────────────────────────────────────────────────────────────────────────────

def _ssh_frames(
    n_cells: int = 10,
    t: float = 1.0,
    delta: float = 0.5,
    scenario: str = "sweep",
    n_sweep: int = 30,
    delta_min: float = -0.9,
    delta_max: float = 0.9,
) -> dict:
    """NumPy SSH reference solver.  Returns spectrum + Zak phase + sweep."""
    import numpy as np  # noqa: PLC0415

    N = 2 * n_cells

    def build_h(delta_: float) -> "np.ndarray":
        t1 = t * (1.0 - delta_)
        t2 = t * (1.0 + delta_)
        H = np.zeros((N, N), dtype=float)
        for i in range(N - 1):
            tij = t1 if i % 2 == 0 else t2
            H[i, i + 1] = -tij
            H[i + 1, i] = -tij
        return H

    def zak_phase(delta_: float) -> float:
        """Discrete Wilson loop in k-space for occupied band."""
        nk = n_cells
        t1 = t * (1.0 - delta_)
        t2 = t * (1.0 + delta_)
        W = 1.0 + 0.0j
        u_prev = None
        u_first = None
        for m in range(nk):
            k = 2.0 * np.pi * m / nk
            dk = -t1 + (-t2) * np.exp(1j * k)
            absdk = abs(dk)
            if absdk < 1e-14:
                u = np.array([1.0, 0.0], dtype=complex)
            else:
                # lower eigenvector: [d*, -|d|]/sqrt(2|d|)  (unnormalized d*/|d|, -1)
                phase = dk / absdk
                u = np.array([np.conj(phase), -1.0], dtype=complex)
                u /= np.linalg.norm(u)
            if u_prev is not None:
                ov = np.dot(np.conj(u_prev), u)
                if abs(ov) > 1e-14:
                    W *= ov / abs(ov)
            if m == 0:
                u_first = u
            u_prev = u
        # Close the loop
        if u_first is not None and u_prev is not None:
            ov = np.dot(np.conj(u_prev), u_first)
            if abs(ov) > 1e-14:
                W *= ov / abs(ov)
        zak = -np.angle(W)
        return float(np.pi if abs(zak) > np.pi / 2 else 0.0)

    def solve_point(delta_: float) -> dict:
        H = build_h(delta_)
        evals, evecs = np.linalg.eigh(H)
        # Bulk gap: edge states in the topological phase have |E| exponentially
        # small (~(t2/t1)^{-n_cells}), while bulk bands start at ~min(t1, t2).
        # Use 0.3*min(t1, t2) to separate edge states from bulk.
        t1_abs = abs(t * (1.0 - delta_))
        t2_abs = abs(t * (1.0 + delta_))
        edge_thr = max(0.3 * min(t1_abs, t2_abs), 1e-10)
        neg = evals[evals < -edge_thr]
        pos = evals[evals >  edge_thr]
        gap = float(pos.min() - neg.max()) if len(neg) and len(pos) else 0.0
        # Edge mode: any eigenvalue with |E| < edge_thr is an in-gap (edge) state
        mid_idx = np.where(np.abs(evals) < edge_thr)[0]
        ew = np.zeros(N)
        if len(mid_idx):
            best = mid_idx[np.argmin(np.abs(evals[mid_idx]))]
            ew = evecs[:, best] ** 2
        zp = zak_phase(delta_)
        return {"energies": evals.tolist(), "gap": gap, "zak_phase": zp, "edge_weight": ew.tolist()}

    sol = solve_point(delta)

    sweep = []
    for k in range(n_sweep):
        dk = delta_min + (delta_max - delta_min) * k / (n_sweep - 1)
        if abs(dk) >= 1.0:
            continue
        s = solve_point(dk)
        ew_boundary = s["edge_weight"][0] + s["edge_weight"][1] if len(s["edge_weight"]) >= 2 else 0.0
        sweep.append({"delta": dk, "gap": s["gap"], "zak_phase": s["zak_phase"],
                      "edge_weight": ew_boundary})

    return {
        "n_cells": n_cells, "t": t, "delta": delta,
        "energies": sol["energies"],
        "gap": sol["gap"],
        "zak_phase": sol["zak_phase"],
        "edge_weight": sol["edge_weight"],
        "sweep": sweep,
    }


def _ssh_page(data: dict, title: str = "SSH Topological Insulator") -> str:
    """4-panel SSH page: spectrum, edge weight profile, gap vs δ, Zak phase vs δ."""
    import json as _json  # noqa: PLC0415

    jdata = _json.dumps(data)
    zak_label = "π (topological)" if abs(data.get("zak_phase", 0)) > 1.0 else "0 (trivial)"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body{{margin:0;background:#0d1117;color:#e6edf3;font-family:monospace;}}
  h1{{text-align:center;padding:16px;font-size:1.1rem;color:#388bfd;}}
  .subtitle{{text-align:center;font-size:0.8rem;color:#8b949e;margin-bottom:8px;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;}}
  canvas{{width:100%;height:220px;display:block;}}
  .label{{font-size:0.7rem;color:#8b949e;text-align:center;margin-top:4px;}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:bold;}}
  .topo{{background:#1c3a2c;color:#57ab5a;border:1px solid #57ab5a;}}
  .trivial{{background:#2c1c1c;color:#e3b341;border:1px solid #e3b341;}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">
  δ = {data.get('delta', 0):.3f} &nbsp;|&nbsp;
  t = {data.get('t', 1):.2f} &nbsp;|&nbsp;
  gap = {data.get('gap', 0):.4f} &nbsp;|&nbsp;
  Zak phase = <span class="badge {'topo' if abs(data.get('zak_phase',0)) > 1.0 else 'trivial'}">{zak_label}</span>
</div>
<div class="grid">
  <div class="panel"><canvas id="cSpec"></canvas><div class="label">Single-particle spectrum (all bands)</div></div>
  <div class="panel"><canvas id="cEdge"></canvas><div class="label">Edge mode |ψ|² per site</div></div>
  <div class="panel"><canvas id="cGap"></canvas><div class="label">Band gap vs δ</div></div>
  <div class="panel"><canvas id="cZak"></canvas><div class="label">Zak phase vs δ (0 or π)</div></div>
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
    var by = H-pad-bh;
    ctx.fillStyle = colorFn(i, v);
    ctx.fillRect(bx, by, bw*0.8, bh);
    if (n <= 24) {{
      ctx.fillStyle="#8b949e"; ctx.font="9px monospace";
      ctx.fillText(i, bx, H-pad+12);
    }}
  }}
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace"; ctx.fillText(ylabel,4,14);
}}

// Panel 1: Spectrum as bar chart (index vs energy)
var eIdx = DATA.energies.map(function(_,i){{ return ""+i; }});
barChart("cSpec", eIdx, DATA.energies,
  function(i,v) {{
    var mid = DATA.energies.length/2;
    if (Math.abs(v) < DATA.gap/4) return "#e3b341";  // mid-gap = gold
    return v < 0 ? "#388bfd" : "#bc8cff";
  }}, "E");

// Panel 2: Edge mode |ψ|² per site
var siteLabels = DATA.edge_weight.map(function(_,i){{ return ""+i; }});
barChart("cEdge", siteLabels, DATA.edge_weight,
  function(i,v) {{ return i < 2 || i >= DATA.edge_weight.length-2 ? "#57ab5a" : "#30363d"; }},
  "|ψ|²");

// Panels 3-4: Sweeps
if (DATA.sweep && DATA.sweep.length > 0) {{
  var swD  = DATA.sweep.map(function(p) {{ return p.delta; }});
  var swGp = DATA.sweep.map(function(p) {{ return p.gap; }});
  var swZp = DATA.sweep.map(function(p) {{ return p.zak_phase; }});
  lineChart("cGap", swD, swGp, "#e3b341", "Gap Δ", "δ");
  lineChart("cZak", swD, swZp, "#57ab5a", "θ_Zak", "δ");
}}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# M30-I: Kitaev Chain  —  Python reference solver + page
# ─────────────────────────────────────────────────────────────────────────────

def _kitaev_frames(
    n_sites: int = 20,
    t: float = 1.0,
    delta: float = 1.0,
    mu: float = 0.0,
    scenario: str = "sweep",
    n_sweep: int = 40,
    mu_min: float = -4.0,
    mu_max: float = 4.0,
) -> dict:
    """NumPy BdG reference for Kitaev chain.  Topo phase: |μ| < 2t."""
    import numpy as np  # noqa: PLC0415

    N = n_sites

    def build_bdg(mu_: float) -> "np.ndarray":
        dim = 2 * N
        H = np.zeros((dim, dim), dtype=float)
        # On-site: -μ/2 in (c,c) block, +μ/2 in (c†,c†) block
        for i in range(N):
            H[i, i]         = -mu_ / 2.0
            H[i + N, i + N] =  mu_ / 2.0
        for i in range(N - 1):
            # Hopping
            H[i, i + 1] += -t / 2.0; H[i + 1, i] += -t / 2.0
            H[i + N, i + 1 + N] += t / 2.0; H[i + 1 + N, i + N] += t / 2.0
            # Pairing
            H[i, i + 1 + N]     += +delta / 2.0
            H[i + 1 + N, i]     += +delta / 2.0
            H[i + 1, i + N]     += -delta / 2.0
            H[i + N, i + 1]     += -delta / 2.0
        return H

    def solve_point(mu_: float) -> dict:
        H = build_bdg(mu_)
        evals, evecs = np.linalg.eigh(H)
        # Threshold: bulk scale, not derived from min eigenvalue (P1 fix)
        thr = max(0.3 * min(abs(t), abs(delta)), 1e-8)
        # Gap: smallest bulk quasi-energy (above threshold, not Majorana split)
        pos = evals[evals > thr]
        gap = float(pos.min()) if len(pos) else 0.0
        # Topological invariant from near-zero modes
        n_zero = int(np.sum(np.abs(evals) < thr))
        topo_inv = -1.0 if n_zero > 0 else 1.0
        # Majorana weight: collect ALL near-zero mode indices.
        # In the topological phase there are 2 (±E degenerate pair at ~0).
        # A single eigenvector from a degenerate pair has arbitrary mixing
        # and yields all-zero u+v.  Combine both modes into symmetric (+)
        # and antisymmetric (-) Majorana operators and pick the one with
        # higher left-boundary weight (site 0).
        zero_inds = np.where(np.abs(evals) < thr)[0]
        if len(zero_inds) >= 2:
            idx1, idx2 = zero_inds[0], zero_inds[1]
            u1, v1 = evecs[:N, idx1], evecs[N:, idx1]
            u2, v2 = evecs[:N, idx2], evecs[N:, idx2]
            majA = ((u1 + u2) + (v1 + v2)) ** 2
            majB = ((u1 - u2) + (v1 - v2)) ** 2
            nA, nB = majA.sum(), majB.sum()
            if nA > 1e-14:
                majA /= nA
            if nB > 1e-14:
                majB /= nB
            # Pick the combination with higher weight on either boundary (site 0 or N-1)
            bndA = max(majA[0], majA[N - 1])
            bndB = max(majB[0], majB[N - 1])
            majorana = majA if bndA >= bndB else majB
        else:
            zero_idx = int(np.argmin(np.abs(evals)))
            evec = evecs[:, zero_idx]
            u, v = evec[:N], evec[N:]
            majorana = (u + v) ** 2
            norm = majorana.sum()
            if norm > 1e-14:
                majorana /= norm
        return {"gap": gap, "topo_inv": topo_inv, "majorana": majorana.tolist()}

    sol0 = solve_point(mu)
    H0 = build_bdg(mu)
    evals0, _ = np.linalg.eigh(H0)

    sweep = []
    for k in range(n_sweep):
        muk = mu_min + (mu_max - mu_min) * k / (n_sweep - 1)
        s = solve_point(muk)
        # Take the larger of the two boundary weights (left or right end)
        maj = s["majorana"]
        bdy_l = maj[0] + maj[1] if len(maj) >= 2 else 0.0
        bdy_r = maj[-1] + maj[-2] if len(maj) >= 2 else 0.0
        bdy = max(bdy_l, bdy_r)
        sweep.append({"mu": muk, "gap": s["gap"], "topo_inv": s["topo_inv"],
                      "majorana_weight": bdy})

    return {
        "n_sites": N, "t": t, "delta": delta, "mu": mu,
        "energies": evals0.tolist(),
        "gap": sol0["gap"],
        "topo_inv": sol0["topo_inv"],
        "majorana_weight": sol0["majorana"],
        "sweep": sweep,
    }


def _kitaev_page(data: dict, title: str = "Kitaev Chain") -> str:
    """4-panel Kitaev page: BdG spectrum, Majorana weight, gap vs μ, topo_inv vs μ."""
    import json as _json  # noqa: PLC0415

    jdata = _json.dumps(data)
    topo_label = "Topological (Q=-1)" if data.get("topo_inv", 1) < 0 else "Trivial (Q=+1)"
    topo_cls   = "topo" if data.get("topo_inv", 1) < 0 else "trivial"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body{{margin:0;background:#0d1117;color:#e6edf3;font-family:monospace;}}
  h1{{text-align:center;padding:16px;font-size:1.1rem;color:#388bfd;}}
  .subtitle{{text-align:center;font-size:0.8rem;color:#8b949e;margin-bottom:8px;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;}}
  canvas{{width:100%;height:220px;display:block;}}
  .label{{font-size:0.7rem;color:#8b949e;text-align:center;margin-top:4px;}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.75rem;font-weight:bold;}}
  .topo{{background:#1c3a2c;color:#57ab5a;border:1px solid #57ab5a;}}
  .trivial{{background:#2c1c1c;color:#e3b341;border:1px solid #e3b341;}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">
  μ = {data.get('mu', 0):.2f} &nbsp;|&nbsp; t = {data.get('t', 1):.2f} &nbsp;|&nbsp;
  Δ = {data.get('delta', 1):.2f} &nbsp;|&nbsp;
  Phase: <span class="badge {topo_cls}">{topo_label}</span>
  &nbsp;(QPT at |μ| = 2t = {2*data.get('t',1):.2f})
</div>
<div class="grid">
  <div class="panel"><canvas id="cSpec"></canvas><div class="label">BdG quasi-energy spectrum</div></div>
  <div class="panel"><canvas id="cMaj"></canvas><div class="label">Majorana weight |u+v|² per site</div></div>
  <div class="panel"><canvas id="cGap"></canvas><div class="label">Quasiparticle gap vs μ/t</div></div>
  <div class="panel"><canvas id="cTopo"></canvas><div class="label">Topological invariant Q vs μ/t</div></div>
</div>
<script>
var DATA = {jdata};

function lineChart(id, xs, ys, color, ylabel, xlabel) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var xmin=Math.min.apply(null,xs), xmax=Math.max.apply(null,xs);
  var ymin=Math.min.apply(null,ys), ymax=Math.max.apply(null,ys);
  var xr=xmax-xmin||1, yr=ymax-ymin||1;
  ctx.strokeStyle="#21262d"; ctx.lineWidth=1;
  for(var gi=0;gi<5;gi++){{var gy=pad+(H-2*pad)*gi/4;ctx.beginPath();ctx.moveTo(pad,gy);ctx.lineTo(W-pad,gy);ctx.stroke();}}
  // QPT markers at μ=±2t
  var twoT = 2 * DATA.t;
  [[twoT,true],[-twoT,true]].forEach(function(m){{
    var px=pad+(m[0]-xmin)/xr*(W-2*pad);
    ctx.strokeStyle="#444"; ctx.lineWidth=1; ctx.setLineDash([4,3]);
    ctx.beginPath(); ctx.moveTo(px,pad); ctx.lineTo(px,H-pad); ctx.stroke();
    ctx.setLineDash([]);
  }});
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
  for(var i=0;i<xs.length;i++){{
    var px=pad+(xs[i]-xmin)/xr*(W-2*pad), py=H-pad-(ys[i]-ymin)/yr*(H-2*pad);
    if(i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace";
  ctx.fillText(ylabel,4,14); ctx.fillText(xlabel,W-pad,H-4);
  ctx.fillText(ymin.toFixed(3),2,H-pad); ctx.fillText(ymax.toFixed(3),2,pad+4);
}}

function barChart(id, vals, colorFn, ylabel) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var n = vals.length, ymax = Math.max.apply(null, vals.map(Math.abs)); if(ymax===0)ymax=1;
  var bw = (W-2*pad)/n;
  for(var i=0;i<n;i++){{
    var v=vals[i], bh=Math.abs(v)/ymax*(H-2*pad), bx=pad+i*bw+bw*0.1, by=H-pad-bh;
    ctx.fillStyle=colorFn(i,v); ctx.fillRect(bx,by,bw*0.8,bh);
  }}
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace"; ctx.fillText(ylabel,4,14);
  ctx.strokeStyle="#444"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
}}

// Panel 1: BdG spectrum (bar by eigenvalue index, colour by sign)
barChart("cSpec", DATA.energies, function(i,v){{return v>0?"#bc8cff":"#388bfd";}}, "E");

// Panel 2: Majorana weight
barChart("cMaj", DATA.majorana_weight,
  function(i,v){{return i<2||i>=DATA.majorana_weight.length-2?"#57ab5a":"#30363d";}},
  "|u+v|²");

// Panels 3-4: sweep
if(DATA.sweep && DATA.sweep.length>0){{
  var swM  = DATA.sweep.map(function(p){{return p.mu/DATA.t;}});
  var swGp = DATA.sweep.map(function(p){{return p.gap;}});
  var swQ  = DATA.sweep.map(function(p){{return p.topo_inv;}});
  lineChart("cGap",  swM, swGp, "#e3b341", "Gap",  "μ/t");
  lineChart("cTopo", swM, swQ,  "#57ab5a", "Q", "μ/t");
}}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# M30-J: Anderson Localization  —  Python reference solver + page
# ─────────────────────────────────────────────────────────────────────────────

def _anderson_frames(
    n_sites: int = 100,
    t: float = 1.0,
    W: float = 2.0,
    n_samples: int = 50,
    scenario: str = "sweep",
    n_sweep: int = 20,
    W_min: float = 0.1,
    W_max: float = 8.0,
    seed: int = 42,
) -> dict:
    """NumPy Anderson localization reference."""
    import numpy as np  # noqa: PLC0415

    N = n_sites
    rng = np.random.default_rng(seed)

    def build_h(W_: float, seed_: int) -> "np.ndarray":
        r = np.random.default_rng(seed_)
        eps = r.uniform(-W_ / 2.0, W_ / 2.0, N)
        H = np.diag(eps) - t * (np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1))
        return H

    def gap_ratio(evals: "np.ndarray") -> float:
        d = np.diff(evals)
        d = d[d > 1e-14]
        if len(d) < 2:
            return 0.0
        r = np.minimum(d[:-1], d[1:]) / np.maximum(d[:-1], d[1:])
        return float(r.mean())

    def solve_sample(W_: float, seed_: int) -> dict:
        H = build_h(W_, seed_)
        evals, evecs = np.linalg.eigh(H)
        ipr = (evecs ** 4).sum(axis=0)  # IPR per eigenstate
        mean_ipr = float(ipr.mean())
        mean_r = gap_ratio(evals)
        return {"evals": evals.tolist(), "ipr": ipr.tolist(),
                "mean_ipr": mean_ipr, "mean_r": mean_r}

    # Single sample at (W, seed)
    s0 = solve_sample(W, seed)
    loc_len = min(float(1.0 / s0["mean_ipr"]) if s0["mean_ipr"] > 1e-14 else N, N)

    sweep = []
    for k in range(n_sweep):
        Wk = W_min + (W_max - W_min) * k / (n_sweep - 1)
        sum_ipr, sum_r = 0.0, 0.0
        for samp in range(n_samples):
            s = solve_sample(Wk, int(rng.integers(0, 2**31)))
            sum_ipr += s["mean_ipr"]
            sum_r   += s["mean_r"]
        avg_ipr = sum_ipr / n_samples
        avg_r   = sum_r / n_samples
        xi = min(1.0 / avg_ipr if avg_ipr > 1e-14 else N, N)
        sweep.append({"W": Wk, "mean_ipr": avg_ipr, "mean_r": avg_r, "loc_length": xi})

    return {
        "n_sites": N, "t": t, "W": W,
        "energies": s0["evals"],
        "ipr_spectrum": s0["ipr"],
        "mean_ipr": s0["mean_ipr"],
        "mean_r": s0["mean_r"],
        "loc_length": loc_len,
        "sweep": sweep,
    }


def _anderson_page(data: dict, title: str = "Anderson Localization") -> str:
    """4-panel Anderson page: IPR spectrum, IPR vs W, ⟨r⟩ vs W, ξ vs W."""
    import json as _json  # noqa: PLC0415

    jdata = _json.dumps(data)
    r_val = data.get("mean_r", 0)
    r_label = f"⟨r⟩ = {r_val:.3f} ({'localised ~Poisson' if r_val < 0.45 else 'ergodic ~GOE'})"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body{{margin:0;background:#0d1117;color:#e6edf3;font-family:monospace;}}
  h1{{text-align:center;padding:16px;font-size:1.1rem;color:#388bfd;}}
  .subtitle{{text-align:center;font-size:0.8rem;color:#8b949e;margin-bottom:8px;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;}}
  canvas{{width:100%;height:220px;display:block;}}
  .label{{font-size:0.7rem;color:#8b949e;text-align:center;margin-top:4px;}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">
  N={data.get('n_sites',100)}, t={data.get('t',1):.2f}, W={data.get('W',2):.2f} &nbsp;|&nbsp;
  ξ ≈ {data.get('loc_length',0):.1f} &nbsp;|&nbsp; {r_label}
</div>
<div class="grid">
  <div class="panel"><canvas id="cIPRs"></canvas><div class="label">IPR per eigenstate (single sample)</div></div>
  <div class="panel"><canvas id="cIPRw"></canvas><div class="label">⟨IPR⟩ vs disorder W/t</div></div>
  <div class="panel"><canvas id="cR"></canvas><div class="label">Level-spacing ratio ⟨r⟩ vs W/t (Poisson≈0.39, GOE≈0.53)</div></div>
  <div class="panel"><canvas id="cXi"></canvas><div class="label">Localization length ξ vs W/t</div></div>
</div>
<script>
var DATA = {jdata};

function lineChart(id, xs, ys, color, ylabel, xlabel, refs) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var xmin=Math.min.apply(null,xs), xmax=Math.max.apply(null,xs);
  var allY = ys.slice(); if(refs) refs.forEach(function(r){{allY.push(r.v);}});
  var ymin=Math.min.apply(null,allY), ymax=Math.max.apply(null,allY);
  var xr=xmax-xmin||1, yr=ymax-ymin||1;
  ctx.strokeStyle="#21262d"; ctx.lineWidth=1;
  for(var gi=0;gi<5;gi++){{var gy=pad+(H-2*pad)*gi/4;ctx.beginPath();ctx.moveTo(pad,gy);ctx.lineTo(W-pad,gy);ctx.stroke();}}
  if(refs) refs.forEach(function(r){{
    var py=H-pad-(r.v-ymin)/yr*(H-2*pad);
    ctx.strokeStyle=r.c; ctx.lineWidth=1; ctx.setLineDash([5,3]);
    ctx.beginPath(); ctx.moveTo(pad,py); ctx.lineTo(W-pad,py); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=r.c; ctx.font="9px monospace"; ctx.fillText(r.label,W-pad-4,py-3);
  }});
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
  for(var i=0;i<xs.length;i++){{
    var px=pad+(xs[i]-xmin)/xr*(W-2*pad), py=H-pad-(ys[i]-ymin)/yr*(H-2*pad);
    if(i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace";
  ctx.fillText(ylabel,4,14); ctx.fillText(xlabel,W-pad,H-4);
}}

function barChart(id, vals, colorFn, ylabel) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var n = vals.length, ymax = Math.max.apply(null, vals); if(ymax===0)ymax=1;
  var bw = Math.max(1,(W-2*pad)/n);
  for(var i=0;i<n;i++){{
    var bh=vals[i]/ymax*(H-2*pad), bx=pad+i*bw, by=H-pad-bh;
    ctx.fillStyle=colorFn(i,vals[i]); ctx.fillRect(bx,by,Math.max(1,bw-1),bh);
  }}
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace"; ctx.fillText(ylabel,4,14);
}}

// Panel 1: IPR per state (scatter via bars)
barChart("cIPRs", DATA.ipr_spectrum,
  function(i,v){{return v > 1/DATA.n_sites * 5 ? "#e3b341" : "#388bfd";}}, "IPR");

if(DATA.sweep && DATA.sweep.length>0){{
  var swW  = DATA.sweep.map(function(p){{return p.W/DATA.t;}});
  var swI  = DATA.sweep.map(function(p){{return p.mean_ipr;}});
  var swR  = DATA.sweep.map(function(p){{return p.mean_r;}});
  var swXi = DATA.sweep.map(function(p){{return p.loc_length;}});
  lineChart("cIPRw", swW, swI, "#bc8cff", "⟨IPR⟩", "W/t", null);
  lineChart("cR",  swW, swR, "#57ab5a", "⟨r⟩", "W/t",
    [{{v:0.386,c:"#e3b341",label:"Poisson"}},{{v:0.530,c:"#388bfd",label:"GOE"}}]);
  lineChart("cXi", swW, swXi, "#388bfd", "ξ", "W/t", null);
}}
</script>
</body>
</html>"""


def _chern_frames(
    n_sites: int = 20,
    m: float = -1.0,
    t: float = 1.0,
    n_k: int = 40,
    n_sweep: int = 30,
    m_min: float = -5.0,
    m_max: float = 1.0,
) -> dict:
    """Compute 2D Chern insulator data: Berry curvature, edge states, Chern sweep."""
    import numpy as np  # noqa: PLC0415

    N = n_sites

    def lower_evec(dz, dx, dy):
        E = np.sqrt(dz**2 + dx**2 + dy**2)
        if E < 1e-15:
            return np.array([1.0 + 0j, 0.0 + 0j])
        denom = E + dz
        c = dx - 1j * dy
        if abs(denom) > 1e-8 * E:
            v0 = -c / denom
            nrm = np.sqrt(abs(v0)**2 + 1.0)
            return np.array([v0 / nrm, 1.0 / nrm])
        else:
            v1 = np.conj(c) / (E - dz)
            nrm = np.sqrt(1.0 + abs(v1)**2)
            return np.array([1.0 / nrm, v1 / nrm])

    def chern_fhs(nk, mv, tv=1.0):
        """FHS Chern number on nk×nk BZ grid."""
        # Pre-compute eigenvectors at (nk+1)×(nk+1) for plaquette corners
        evecs = {}
        for ix in range(nk + 1):
            for iy in range(nk + 1):
                kx, ky = 2*np.pi*(ix % nk)/nk, 2*np.pi*(iy % nk)/nk
                evecs[(ix, iy)] = lower_evec(mv + tv * (np.cos(kx) + np.cos(ky)),
                                              tv * np.sin(kx), tv * np.sin(ky))
        C, berry = 0.0, np.zeros(nk * nk)
        for ix in range(nk):
            for iy in range(nk):
                u00, u10, u01, u11 = (evecs[(ix, iy)], evecs[(ix+1, iy)],
                                       evecs[(ix, iy+1)], evecs[(ix+1, iy+1)])
                def link(a, b):
                    z = np.conj(a) @ b
                    r = abs(z)
                    return z / r if r > 1e-15 else 1.0 + 0j
                U1_00 = link(u00, u10)
                U2_10 = link(u10, u11)
                U1_01 = link(u01, u11)
                U2_00 = link(u00, u01)
                F = np.log(U1_00 * U2_10 * np.conj(U1_01) * np.conj(U2_00))
                omega = F.imag
                berry[ix * nk + iy] = omega
                C += omega
        return C / (2 * np.pi), berry

    def solve_strip(ny, mv, kx, tv=1.0):
        """Build 2ny×2ny real symmetric strip Hamiltonian at given kx and diagonalise."""
        dim = 2 * ny
        H = np.zeros((dim, dim))
        c_, s_ = mv + tv * np.cos(kx), tv * np.sin(kx)
        th = 0.5 * tv
        for j in range(ny):
            b = j * 2
            H[b, b] += c_;   H[b+1, b+1] += -c_
            H[b, b+1] += s_; H[b+1, b] += s_
        for j in range(ny - 1):
            b, bp = j*2, (j+1)*2
            H[b, bp] += th;    H[b, bp+1] += -th
            H[b+1, bp] += th;  H[b+1, bp+1] += -th
            H[bp, b] += th;    H[bp, b+1] += th
            H[bp+1, b] += -th; H[bp+1, b+1] += -th
        return np.linalg.eigvalsh(H).tolist()

    # -- bands scenario data
    chern_val, berry = chern_fhs(n_k, m, t)
    kgrid = 2 * np.pi * np.arange(n_k) / n_k
    KX, KY = np.meshgrid(kgrid, kgrid, indexing='ij')
    dz = m + t * (np.cos(KX) + np.cos(KY))
    dx, dy = t * np.sin(KX), t * np.sin(KY)
    E = np.sqrt(dz**2 + dx**2 + dy**2)
    lower_band = (-E).flatten().tolist()
    upper_band = (+E).flatten().tolist()
    gap = float(2 * E.min())

    # -- edge scenario: strip spectrum vs kx
    kx_vals = [i / (n_k - 1) for i in range(n_k)]    # fraction of 2π
    edge_spectrum = [solve_strip(N, m, 2*np.pi*k, t) for k in kx_vals]

    # -- sweep: Chern # and gap vs m
    sweep = []
    for k in range(n_sweep):
        mk = m_min + (m_max - m_min) * k / (n_sweep - 1)
        cv, _ = chern_fhs(n_k, mk, t)
        kgM = np.meshgrid(kgrid, kgrid, indexing='ij')
        dz_ = mk + t * (np.cos(kgM[0]) + np.cos(kgM[1]))
        Ek = np.sqrt(dz_**2 + (t*np.sin(kgM[0]))**2 + (t*np.sin(kgM[1]))**2)
        sweep.append({"m": mk, "gap": float(2 * Ek.min()), "chern": round(cv)})

    return {
        "n_sites": N, "m": m, "t": t,
        "n_k": n_k, "lower_band": lower_band, "upper_band": upper_band,
        "berry_curv": berry.tolist(), "chern": round(chern_val), "gap": gap,
        "kx_vals": kx_vals, "edge_spectrum": edge_spectrum,
        "sweep": sweep,
    }


def _chern_page(data: dict, title: str = "2D Chern Insulator") -> str:
    """4-panel Chern page: Berry curvature heatmap, edge state spectrum, gap and C vs m."""
    import json as _json  # noqa: PLC0415

    jdata = _json.dumps(data)
    C = data.get("chern", 0)
    topo_label = f"C = {C:+d} ({'Topological' if C != 0 else 'Trivial'})"
    topo_cls   = "topo" if C != 0 else "trivial"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body{{margin:0;background:#0d1117;color:#e6edf3;font-family:monospace;}}
  h1{{text-align:center;padding:16px;font-size:1.1rem;color:#388bfd;}}
  .subtitle{{text-align:center;font-size:0.8rem;color:#8b949e;margin-bottom:8px;}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:4px;font-size:0.8rem;}}
  .topo{{background:#1b4332;color:#57ab5a;border:1px solid #57ab5a;}}
  .trivial{{background:#3d1f0d;color:#e3b341;border:1px solid #e3b341;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;}}
  canvas{{width:100%;height:220px;display:block;}}
  .label{{font-size:0.7rem;color:#8b949e;text-align:center;margin-top:4px;}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">
  m={data.get('m',0):.2f}, t={data.get('t',1):.2f}, N_strip={data.get('n_sites',20)} &nbsp;|&nbsp;
  gap={data.get('gap',0):.4f} &nbsp;|&nbsp;
  Phase: <span class="badge {topo_cls}">{topo_label}</span>
  &nbsp;(QPTs at m = 0, ±2)
</div>
<div class="grid">
  <div class="panel"><canvas id="cBerry"></canvas><div class="label">Berry curvature Ω(kx,ky) [lower band]</div></div>
  <div class="panel"><canvas id="cEdge"></canvas><div class="label">Strip spectrum E(kx) — chiral edge states</div></div>
  <div class="panel"><canvas id="cGap"></canvas><div class="label">Band gap vs mass m</div></div>
  <div class="panel"><canvas id="cChern"></canvas><div class="label">Chern number C vs mass m</div></div>
</div>
<script>
var DATA = {jdata};

function heatmap(id, vals, nk) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  var vmin = Math.min.apply(null,vals), vmax = Math.max.apply(null,vals);
  var vr = vmax - vmin || 1;
  var cw = W/nk, ch = H/nk;
  for (var ix=0;ix<nk;ix++) for (var iy=0;iy<nk;iy++) {{
    var t = (vals[ix*nk+iy]-vmin)/vr;
    // diverging colormap: blue (negative) → white → red (positive)
    var r,g,b;
    if (t < 0.5) {{ r=Math.round(2*t*255); g=Math.round(2*t*255); b=255; }}
    else {{ r=255; g=Math.round(2*(1-t)*255); b=Math.round(2*(1-t)*255); }}
    ctx.fillStyle="rgb("+r+","+g+","+b+")";
    ctx.fillRect(ix*cw, H - (iy+1)*ch, cw+1, ch+1);
  }}
  ctx.fillStyle="#e6edf3"; ctx.font="9px monospace";
  ctx.fillText("kx→", W-28, H-2); ctx.fillText("↑ky", 2, 12);
}}

function scatter(id, xs, ys, color) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 36;
  ctx.fillStyle="#0d1117"; ctx.fillRect(0,0,W,H);
  var xmin=Math.min.apply(null,xs), xmax=Math.max.apply(null,xs)||1;
  var ymin=Math.min.apply(null,ys), ymax=Math.max.apply(null,ys)||1;
  var xr=xmax-xmin||1, yr=ymax-ymin||1;
  ctx.strokeStyle="#21262d"; ctx.lineWidth=1;
  for(var gi=0;gi<5;gi++){{var gy=pad+(H-2*pad)*gi/4;ctx.beginPath();ctx.moveTo(pad,gy);ctx.lineTo(W-pad,gy);ctx.stroke();}}
  ctx.fillStyle=color;
  for(var i=0;i<xs.length;i++) {{
    var px=pad+(xs[i]-xmin)/xr*(W-2*pad), py=H-pad-(ys[i]-ymin)/yr*(H-2*pad);
    ctx.fillRect(px-1, py-1, 2, 2);
  }}
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace";
  ctx.fillText("E",4,14); ctx.fillText("kx/2π",W-40,H-4);
}}

function lineChart(id, xs, ys, color, ylabel, xlabel, refs, xrefs) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle="#0d1117"; ctx.fillRect(0,0,W,H);
  var xmin=Math.min.apply(null,xs), xmax=Math.max.apply(null,xs);
  var allY=ys.slice(); if(refs) refs.forEach(function(r){{allY.push(r.v);}});
  var ymin=Math.min.apply(null,allY), ymax=Math.max.apply(null,allY);
  var xr=xmax-xmin||1, yr=ymax-ymin||1;
  function gpx(x){{return pad+(x-xmin)/xr*(W-2*pad);}}
  function gpy(y){{return H-pad-(y-ymin)/yr*(H-2*pad);}}
  ctx.strokeStyle="#21262d"; ctx.lineWidth=1;
  for(var gi=0;gi<5;gi++){{var gy=pad+(H-2*pad)*gi/4;ctx.beginPath();ctx.moveTo(pad,gy);ctx.lineTo(W-pad,gy);ctx.stroke();}}
  if(xrefs) xrefs.forEach(function(r){{
    var gx=gpx(r.v);
    ctx.strokeStyle=r.c||"#e3b341"; ctx.lineWidth=1; ctx.setLineDash([4,3]);
    ctx.beginPath();ctx.moveTo(gx,pad);ctx.lineTo(gx,H-pad);ctx.stroke(); ctx.setLineDash([]);
    if(r.label){{ctx.fillStyle=r.c||"#e3b341"; ctx.font="9px monospace"; ctx.fillText(r.label,gx+2,pad+10);}}
  }});
  if(refs) refs.forEach(function(r){{
    var py=gpy(r.v);
    ctx.strokeStyle=r.c; ctx.lineWidth=1; ctx.setLineDash([4,3]);
    ctx.beginPath();ctx.moveTo(pad,py);ctx.lineTo(W-pad,py);ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle=r.c; ctx.font="9px monospace"; ctx.fillText(r.label,W-pad-4,py-3);
  }});
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
  for(var i=0;i<xs.length;i++){{
    var px=gpx(xs[i]), py=gpy(ys[i]);
    if(i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace";
  ctx.fillText(ylabel,4,14); ctx.fillText(xlabel,W-pad,H-4);
}}

// Panel 1: Berry curvature heatmap
heatmap("cBerry", DATA.berry_curv, DATA.n_k);

// Panel 2: edge spectrum (scatter kx vs E)
if (DATA.edge_spectrum && DATA.edge_spectrum.length > 0) {{
  var ex=[], ey=[];
  DATA.edge_spectrum.forEach(function(sp,i){{
    sp.forEach(function(e){{ ex.push(DATA.kx_vals[i]); ey.push(e); }});
  }});
  scatter("cEdge", ex, ey, "#388bfd");
}}

// Panels 3-4: sweep
if (DATA.sweep && DATA.sweep.length > 0) {{
  var swM  = DATA.sweep.map(function(p){{return p.m;}});
  var swG  = DATA.sweep.map(function(p){{return p.gap;}});
  var swC  = DATA.sweep.map(function(p){{return p.chern;}});
  var qptX=[{{v:0,c:"#e3b341",label:"m=0"}},{{v:2,c:"#e3b341",label:"m=+2"}},{{v:-2,c:"#e3b341",label:"m=−2"}}];
  lineChart("cGap",   swM, swG, "#bc8cff", "gap", "m", null, qptX);
  lineChart("cChern", swM, swC, "#57ab5a", "C",   "m",
    [{{v:0,c:"#30363d",label:"C=0"}},{{v:1,c:"#388bfd",label:"C=+1"}},{{v:-1,c:"#bc8cff",label:"C=−1"}}],
    qptX);
}}
</script>
</body>
</html>"""


def _hofstadter_frames(
    t: float = 1.0,
    q_max: int = 24,
    phi: float = 0.25,
    n_k: int = 40,
    n_ky: int = 12,
) -> dict:
    """Compute Hofstadter butterfly data and fixed-flux band structure."""
    import math  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    def gcd(a, b):
        return a if b == 0 else gcd(b, a % b)

    def harper_evals(p, q, ky):
        """q×q real symmetric Harper matrix at kx=0, return eigenvalues."""
        phi_ = p / q
        H = np.zeros((q, q))
        for n in range(q):
            H[n, n] = -2 * t * np.cos(2 * np.pi * phi_ * n - ky)
        for n in range(q - 1):
            H[n, n+1] = H[n+1, n] = -t
        H[0, q-1] = H[q-1, 0] = -t
        return np.linalg.eigvalsh(H).tolist()

    # Butterfly: scan all rationals p/q with q <= q_max
    phi_vals, evals = [], []
    for q in range(1, q_max + 1):
        for p in range(1, q):
            if gcd(p, q) != 1:
                continue
            phi_ = p / q
            for iky in range(n_ky):
                ky = 2 * np.pi * iky / n_ky
                for e in harper_evals(p, q, ky):
                    phi_vals.append(phi_)
                    evals.append(e)
    # Add phi=0 trivial band
    for iky in range(n_ky):
        ky = 2 * np.pi * iky / n_ky
        phi_vals.append(0.0)
        evals.append(-2 * t * np.cos(ky))

    # Fixed-flux bands: find nearest rational p/q to given phi
    best_p, best_q, best_err = 1, 4, 1e30
    for q in range(1, q_max + 1):
        p = max(1, min(round(phi * q), q - 1))
        if gcd(p, q) != 1:
            continue
        err = abs(p / q - phi)
        if err < best_err:
            best_err, best_p, best_q = err, p, q

    phi_used = best_p / best_q
    kx_max = 2 * np.pi / best_q
    kx_vals = [kx_max * i / (n_k - 1) for i in range(n_k)]
    bands_data = []
    for kx in kx_vals:
        phi_ = best_p / best_q
        H = np.zeros((best_q, best_q))
        for n in range(best_q):
            H[n, n] = -2 * t * np.cos(2 * np.pi * phi_ * n)
        for n in range(best_q - 1):
            H[n, n+1] = H[n+1, n] = -t
        # Corner with Peierls phase e^{i*q*kx}
        phase_re = -t * np.cos(best_q * kx)
        phase_im = -t * np.sin(best_q * kx)
        # Real doubling for complex corner: [[A,-B],[B,A]]
        if abs(phase_im) < 1e-12:
            H[0, best_q-1] = H[best_q-1, 0] = phase_re
            evals_k = np.linalg.eigvalsh(H).tolist()
        else:
            dim2 = 2 * best_q
            HR = np.zeros((dim2, dim2))
            HR[:best_q, :best_q] = H.copy()
            HR[best_q:, best_q:] = H.copy()
            # Fix corner entries
            HR[0, best_q-1]          = phase_re
            HR[best_q-1, 0]          = phase_re
            HR[best_q, 2*best_q-1]   = phase_re
            HR[2*best_q-1, best_q]   = phase_re
            im_c = t * np.sin(best_q * kx)
            HR[0, best_q + best_q-1]        = -im_c
            HR[best_q-1, best_q + 0]        = -im_c
            HR[best_q + 0, best_q-1]        = +im_c
            HR[best_q + best_q-1, 0]        = +im_c
            evals2 = np.linalg.eigvalsh(HR)
            evals_k = [evals2[2*n] for n in range(best_q)]
        bands_data.append(evals_k)

    return {
        "t": t, "q_max": q_max, "phi": phi, "phi_used": phi_used,
        "n_bands": best_q,
        "phi_vals": phi_vals, "evals": evals,
        "kx_vals": [k / kx_max for k in kx_vals],   # normalised to [0,1]
        "bands": bands_data,
    }


def _hofstadter_page(data: dict, title: str = "Hofstadter Butterfly") -> str:
    """2-panel Hofstadter page: butterfly scatter and fixed-flux band structure."""
    import json as _json  # noqa: PLC0415

    jdata = _json.dumps(data)
    phi_used = data.get("phi_used", 0.25)
    n_bands  = data.get("n_bands", 4)
    # Express phi_used as fraction string
    from fractions import Fraction  # noqa: PLC0415
    frac = Fraction(phi_used).limit_denominator(60)
    phi_str = f"{frac.numerator}/{frac.denominator}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body{{margin:0;background:#0d1117;color:#e6edf3;font-family:monospace;}}
  h1{{text-align:center;padding:16px;font-size:1.1rem;color:#388bfd;}}
  .subtitle{{text-align:center;font-size:0.8rem;color:#8b949e;margin-bottom:8px;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;}}
  canvas{{width:100%;height:340px;display:block;}}
  .label{{font-size:0.7rem;color:#8b949e;text-align:center;margin-top:4px;}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">
  Hofstadter 1976 · t={data.get('t',1):.1f} · q_max={data.get('q_max',24)}
  &nbsp;|&nbsp; Fixed flux φ = {phi_str} ({n_bands} bands shown right)
</div>
<div class="grid">
  <div class="panel"><canvas id="cButterfly"></canvas>
    <div class="label">Allowed energies E vs flux φ = p/q (fractal butterfly)</div></div>
  <div class="panel"><canvas id="cBands"></canvas>
    <div class="label">Band structure at φ = {phi_str} (magnetic BZ kx ∈ [0, 2π/q])</div></div>
</div>
<script>
var DATA = {jdata};

function scatter2(id, xs, ys, dotSize) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 340;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 36;
  ctx.fillStyle="#0d1117"; ctx.fillRect(0,0,W,H);
  var xmin=0, xmax=1, ymin=Math.min.apply(null,ys), ymax=Math.max.apply(null,ys);
  var xr=1, yr=ymax-ymin||1;
  ctx.strokeStyle="#21262d"; ctx.lineWidth=1;
  for(var gi=0;gi<5;gi++){{var gy=pad+(H-2*pad)*gi/4;ctx.beginPath();ctx.moveTo(pad,gy);ctx.lineTo(W-pad,gy);ctx.stroke();}}
  var imgData = ctx.getImageData(0,0,W,H);
  var d = imgData.data;
  for(var i=0;i<xs.length;i++){{
    var px=Math.round(pad+(xs[i]-xmin)/xr*(W-2*pad));
    var py=Math.round(H-pad-(ys[i]-ymin)/yr*(H-2*pad));
    if(px>=0&&px<W&&py>=0&&py<H){{
      var idx=4*(py*W+px); d[idx]=56;d[idx+1]=139;d[idx+2]=253;d[idx+3]=180;
    }}
  }}
  ctx.putImageData(imgData,0,0);
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace";
  ctx.fillText("E",4,14); ctx.fillText("φ",W-pad,H-4);
}}

function multiline(id, kxs, bandsArr, nBands) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 340;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle="#0d1117"; ctx.fillRect(0,0,W,H);
  var xmin=kxs[0], xmax=kxs[kxs.length-1];
  var allE=[]; bandsArr.forEach(function(row){{row.forEach(function(e){{allE.push(e);}});}});
  var ymin=Math.min.apply(null,allE), ymax=Math.max.apply(null,allE);
  var xr=xmax-xmin||1, yr=ymax-ymin||1;
  ctx.strokeStyle="#21262d"; ctx.lineWidth=1;
  for(var gi=0;gi<5;gi++){{var gy=pad+(H-2*pad)*gi/4;ctx.beginPath();ctx.moveTo(pad,gy);ctx.lineTo(W-pad,gy);ctx.stroke();}}
  var palette=["#388bfd","#57ab5a","#bc8cff","#e3b341","#f47067","#79c0ff","#56d364","#d2a8ff"];
  for(var b=0;b<nBands;b++){{
    ctx.strokeStyle=palette[b%palette.length]; ctx.lineWidth=2; ctx.beginPath();
    for(var i=0;i<kxs.length;i++){{
      var px=pad+(kxs[i]-xmin)/xr*(W-2*pad);
      var py=H-pad-(bandsArr[i][b]-ymin)/yr*(H-2*pad);
      if(i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
    }}
    ctx.stroke();
  }}
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace";
  ctx.fillText("E",4,14); ctx.fillText("kx/kmax",W-52,H-4);
}}

scatter2("cButterfly", DATA.phi_vals, DATA.evals, 1);
if(DATA.bands && DATA.bands.length > 0)
  multiline("cBands", DATA.kx_vals, DATA.bands, DATA.n_bands);
</script>
</body>
</html>"""


def _mbl_frames(
    n_sites: int = 10,
    J: float = 1.0,
    delta: float = 1.0,
    W: float = 3.0,
    scenario: str = "single",
    W_min: float = 0.5,
    W_max: float = 8.0,
    n_sweep: int = 16,
    n_samples: int = 10,
    seed: int = 42,
) -> dict:
    """NumPy MBL reference implementation: disordered XXZ Heisenberg chain."""
    import numpy as np  # noqa: PLC0415

    N = n_sites
    rng = np.random.default_rng(seed)

    def sz0_basis(n: int):
        """Enumerate Sz=0 basis states (Gosper's hack)."""
        n_down = n // 2
        s = (1 << n_down) - 1
        lim = 1 << n
        states = []
        while s < lim:
            states.append(s)
            c = s & (-s)
            r = s + c
            s = (((r ^ s) >> 2) // c) | r
        return np.array(states, dtype=np.int32)

    def build_ham(h_disorder):
        """Build dense MBL Hamiltonian in Sz=0 sector."""
        basis = sz0_basis(N)
        dim = len(basis)
        idx_map = {int(basis[i]): i for i in range(dim)}
        H = np.zeros((dim, dim))
        for i in range(dim):
            s = int(basis[i])
            diag = 0.0
            for site in range(N - 1):
                bi  = (s >> site    ) & 1
                bi1 = (s >> (site+1)) & 1
                diag += J * delta * (1 - 2*bi) * (1 - 2*bi1) * 0.25
            for site in range(N):
                bi = (s >> site) & 1
                diag += h_disorder[site] * (1 - 2*bi) * 0.5
            H[i, i] = diag
            for site in range(N - 1):
                bi  = (s >> site    ) & 1
                bi1 = (s >> (site+1)) & 1
                if bi == bi1:
                    continue
                mask = (1 << site) | (1 << (site + 1))
                sp = s ^ mask
                j = idx_map.get(sp, -1)
                if j > i:
                    H[i, j] = H[j, i] = J * 0.5
        return H, basis

    def entanglement_entropy(psi, basis_arr):
        """Half-chain S_EE via reduced density matrix."""
        NA = N // 2
        dim_L = 1 << NA
        dim_R = 1 << (N - NA)
        C = np.zeros((dim_L, dim_R))
        for idx, s in enumerate(basis_arr):
            sL = int(s) & (dim_L - 1)
            sR = int(s) >> NA
            C[sL, sR] = psi[idx]
        rho = C @ C.T
        evals_rho = np.linalg.eigvalsh(rho)
        evals_rho = evals_rho[evals_rho > 1e-15]
        return float(-np.sum(evals_rho * np.log(evals_rho)))

    def level_spacing_ratio(evals_sorted):
        """Mean level-spacing ratio <r>."""
        d = np.diff(evals_sorted)
        d = d[d > 1e-15]
        if len(d) < 2:
            return 0.0
        r = np.minimum(d[:-1], d[1:]) / np.maximum(d[:-1], d[1:])
        return float(r.mean())

    def level_spacing_ratios_all(evals_sorted):
        """Individual r_n values."""
        d = np.diff(evals_sorted)
        mask = (d[:-1] > 1e-15) & (d[1:] > 1e-15)
        d0, d1 = d[:-1][mask], d[1:][mask]
        return (np.minimum(d0, d1) / np.maximum(d0, d1)).tolist()

    def solve_single(h_disorder):
        """Full diagonalization for one disorder realization."""
        H, basis = build_ham(h_disorder)
        dim = len(basis)
        evals, evecs = np.linalg.eigh(H)
        lo, hi = dim // 4, 3 * dim // 4
        entropies = [entanglement_entropy(evecs[:, col], basis) for col in range(lo, hi)]
        iprs = [(evecs[:, col] ** 4).sum() for col in range(lo, hi)]
        r_vals = level_spacing_ratios_all(evals)
        r_mean = level_spacing_ratio(evals)
        ent_mean = float(np.mean(entropies)) if entropies else 0.0
        ipr_mean = float(np.mean(iprs)) if iprs else 0.0
        return {
            "evals": evals.tolist(),
            "entropies": entropies,
            "iprs": iprs,
            "r_vals": r_vals,
            "r_mean": r_mean,
            "entropy_mean": ent_mean,
            "ipr_mean": ipr_mean,
        }

    # Single realization
    h0 = rng.uniform(-W, W, N)
    single = solve_single(h0)
    phase_label = "Ergodic / ETH" if single["r_mean"] > 0.46 else "MBL / Localized"

    sweep = []
    if scenario == "sweep":
        for k in range(n_sweep):
            Wk = W_min + (W_max - W_min) * k / (n_sweep - 1)
            sum_r, sum_S, sum_ipr = 0.0, 0.0, 0.0
            for samp in range(n_samples):
                h_s = np.random.default_rng(int(rng.integers(0, 2**31))).uniform(-Wk, Wk, N)
                H_s, basis_s = build_ham(h_s)
                dim = len(basis_s)
                evals_s, evecs_s = np.linalg.eigh(H_s)
                sum_r += level_spacing_ratio(evals_s)
                lo, hi = dim // 4, 3 * dim // 4
                step = max(1, (hi - lo) // 8)
                samp_S, samp_ipr, cnt = 0.0, 0.0, 0
                for col in range(lo, hi, step):
                    samp_S   += entanglement_entropy(evecs_s[:, col], basis_s)
                    samp_ipr += float((evecs_s[:, col] ** 4).sum())
                    cnt += 1
                if cnt > 0:
                    sum_S   += samp_S   / cnt
                    sum_ipr += samp_ipr / cnt
            sweep.append({
                "W": float(Wk),
                "r_mean": float(sum_r / n_samples),
                "entropy_mean": float(sum_S / n_samples),
                "ipr_mean": float(sum_ipr / n_samples),
            })

    return {
        "n_sites": N,
        "J": J,
        "delta": delta,
        "W": W,
        "scenario": scenario,
        "evals": single["evals"],
        "entropies": single["entropies"],
        "iprs": single["iprs"],
        "r_vals": single["r_vals"],
        "r_mean": single["r_mean"],
        "entropy_mean": single["entropy_mean"],
        "ipr_mean": single["ipr_mean"],
        "phase_label": phase_label,
        "sweep": sweep,
    }


def _mbl_page(data: dict, title: str = "Many-Body Localization") -> str:
    """4-panel MBL page: energy+IPR, r histogram, S_EE, sweep diagnostics."""
    import json as _json  # noqa: PLC0415

    jdata = _json.dumps(data)
    r_val = data.get("r_mean", 0.0)
    phase = data.get("phase_label", "MBL / Localized")
    ent   = data.get("entropy_mean", 0.0)
    ipr   = data.get("ipr_mean", 0.0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body{{margin:0;background:#0d1117;color:#e6edf3;font-family:monospace;}}
  h1{{text-align:center;padding:16px;font-size:1.1rem;color:#388bfd;}}
  .subtitle{{text-align:center;font-size:0.8rem;color:#8b949e;margin-bottom:8px;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;}}
  canvas{{width:100%;height:220px;display:block;}}
  .label{{font-size:0.7rem;color:#8b949e;text-align:center;margin-top:4px;}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:4px;font-size:0.75rem;
          background:#1f3a5f;color:#79c0ff;border:1px solid #388bfd;margin:4px;}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">
  N={data.get('n_sites',10)}, J={data.get('J',1):.2f}, &Delta;={data.get('delta',1):.2f},
  W={data.get('W',3):.2f} &nbsp;|&nbsp;
  <b>&langle;r&rangle;</b>={r_val:.3f} &nbsp;|&nbsp;
  <b>&langle;S_EE&rangle;</b>={ent:.3f} &nbsp;|&nbsp;
  <b>&langle;IPR&rangle;</b>={ipr:.4f}
  &nbsp;&nbsp;<span class="badge">{phase}</span>
</div>
<div class="grid">
  <div class="panel"><canvas id="cIPR"></canvas>
    <div class="label">IPR per eigenstate (middle half) — high IPR = localized</div></div>
  <div class="panel"><canvas id="cRhist"></canvas>
    <div class="label">Level-spacing ratio r histogram (GOE=0.530, Poisson=0.386)</div></div>
  <div class="panel"><canvas id="cEnt"></canvas>
    <div class="label">Half-chain entanglement S_EE per eigenstate (middle half)</div></div>
  <div class="panel"><canvas id="cSweep"></canvas>
    <div class="label">&langle;r&rangle; vs disorder W — transition ergodic&rarr;MBL</div></div>
</div>
<script>
var DATA = {jdata};

function lineChart(id, xs, ys, color, ylabel, xlabel, refs) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var xmin=Math.min.apply(null,xs), xmax=Math.max.apply(null,xs);
  var allY = ys.slice(); if(refs) refs.forEach(function(r){{allY.push(r.v);}});
  var ymin=Math.min.apply(null,allY), ymax=Math.max.apply(null,allY);
  var xr=xmax-xmin||1, yr=ymax-ymin||1;
  ctx.strokeStyle="#21262d"; ctx.lineWidth=1;
  for(var gi=0;gi<5;gi++){{var gy=pad+(H-2*pad)*gi/4;ctx.beginPath();ctx.moveTo(pad,gy);ctx.lineTo(W-pad,gy);ctx.stroke();}}
  if(refs) refs.forEach(function(r){{
    var py=H-pad-(r.v-ymin)/yr*(H-2*pad);
    ctx.strokeStyle=r.c; ctx.lineWidth=1; ctx.setLineDash([5,3]);
    ctx.beginPath(); ctx.moveTo(pad,py); ctx.lineTo(W-pad,py); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=r.c; ctx.font="9px monospace"; ctx.fillText(r.label,W-pad-4,py-3);
  }});
  ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath();
  for(var i=0;i<xs.length;i++){{
    var px=pad+(xs[i]-xmin)/xr*(W-2*pad), py=H-pad-(ys[i]-ymin)/yr*(H-2*pad);
    if(i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace";
  ctx.fillText(ylabel,4,14); ctx.fillText(xlabel,W-pad,H-4);
}}

function barChart(id, vals, colorFn, ylabel) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var n = vals.length, ymax = Math.max.apply(null, vals); if(ymax===0)ymax=1;
  var bw = Math.max(1,(W-2*pad)/n);
  for(var i=0;i<n;i++){{
    var bh=vals[i]/ymax*(H-2*pad), bx=pad+i*bw, by=H-pad-bh;
    ctx.fillStyle=colorFn(i,vals[i]); ctx.fillRect(bx,by,Math.max(1,bw-1),bh);
  }}
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace"; ctx.fillText(ylabel,4,14);
}}

function rHistogram(id, rvals) {{
  var cv = document.getElementById(id); if (!cv) return;
  cv.width = cv.offsetWidth || 400; cv.height = 220;
  var ctx = cv.getContext("2d"), W = cv.width, H = cv.height, pad = 40;
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0,0,W,H);
  var nbins = 20;
  var counts = new Array(nbins).fill(0);
  for(var i=0;i<rvals.length;i++){{
    var b = Math.min(nbins-1, Math.floor(rvals[i]*nbins));
    counts[b]++;
  }}
  var ymax = Math.max.apply(null, counts); if(ymax===0)ymax=1;
  var bw = (W-2*pad)/nbins;
  for(var b=0;b<nbins;b++){{
    var bh=counts[b]/ymax*(H-2*pad), bx=pad+b*bw, by=H-pad-bh;
    ctx.fillStyle="#57ab5a"; ctx.fillRect(bx,by,Math.max(1,bw-1),bh);
  }}
  // GOE and Poisson reference lines
  var refs2=[{{v:0.530,c:"#388bfd",label:"GOE"}},{{v:0.386,c:"#e3b341",label:"Poisson"}}];
  refs2.forEach(function(r){{
    var rx=pad+r.v*(W-2*pad);
    ctx.strokeStyle=r.c; ctx.lineWidth=1.5; ctx.setLineDash([5,3]);
    ctx.beginPath(); ctx.moveTo(rx,pad); ctx.lineTo(rx,H-pad); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle=r.c; ctx.font="9px monospace"; ctx.fillText(r.label,rx+2,pad+12);
  }});
  ctx.fillStyle="#8b949e"; ctx.font="10px monospace";
  ctx.fillText("count",4,14); ctx.fillText("r",W-pad,H-4);
}}

// Panel 1: IPR per eigenstate
if(DATA.iprs && DATA.iprs.length>0){{
  var dim=DATA.evals.length, ipr_th=1.0/(dim||1);
  barChart("cIPR", DATA.iprs,
    function(i,v){{return v > ipr_th*5 ? "#e3b341" : "#388bfd";}}, "IPR");
}}

// Panel 2: r histogram
if(DATA.r_vals && DATA.r_vals.length>0)
  rHistogram("cRhist", DATA.r_vals);

// Panel 3: S_EE per eigenstate
if(DATA.entropies && DATA.entropies.length>0){{
  var xs3=[]; for(var i=0;i<DATA.entropies.length;i++) xs3.push(i);
  lineChart("cEnt", xs3, DATA.entropies, "#bc8cff", "S_EE", "state idx", null);
}}

// Panel 4: <r> vs W sweep
if(DATA.sweep && DATA.sweep.length>0){{
  var swW=DATA.sweep.map(function(p){{return p.W;}});
  var swR=DATA.sweep.map(function(p){{return p.r_mean;}});
  lineChart("cSweep", swW, swR, "#57ab5a", "<r>", "W",
    [{{v:0.386,c:"#e3b341",label:"Poisson"}},{{v:0.530,c:"#388bfd",label:"GOE"}}]);
}}
</script>
</body>
</html>"""


def _spectral_frames(
    n_sites: int = 12,
    J: float = 1.0,
    delta: float = 1.0,
    h: float = 0.0,
    scenario: str = "both",
    n_k: int = 20,
    n_omega: int = 200,
    omega_min: float = -4.0,
    omega_max: float = 4.0,
    eta: float = 0.15,
    tau_max: float = 4.0,
    n_tau: int = 60,
) -> dict:
    """NumPy reference: dynamical structure factor S(k,omega) and imaginary-time convergence."""
    import numpy as np  # noqa: PLC0415

    N = n_sites
    if N < 2 or N % 2 != 0:
        raise ValueError("n_sites must be even and >= 2")
    # h has no effect in the Sz=0 sector (sum_i Sz_i = 0 identically).
    if h != 0.0:
        raise ValueError(
            "h != 0 has no effect in the Sz=0 sector — the Zeeman term "
            "h*sum_i Sz_i vanishes identically on all Sz=0 basis states. "
            "Set h=0.0 (default)."
        )

    do_spectral = scenario in {"spectral", "both"}
    do_imagtime = scenario in {"imagtime", "both"}

    # ── Sz=0 basis (Gosper's hack) ───────────────────────────────────────────
    def sz0_basis(n: int):
        n_down = n // 2
        s = (1 << n_down) - 1
        lim = 1 << n
        states = []
        while s < lim:
            states.append(s)
            c = s & (-s)
            r = s + c
            s = (((r ^ s) >> 2) // c) | r
        return np.array(states, dtype=np.int64)

    basis = sz0_basis(N)
    dim = len(basis)
    idx_map: dict = {int(s): i for i, s in enumerate(basis)}

    # ── Build XXZ Hamiltonian (h=0 enforced above) ───────────────────────────
    H = np.zeros((dim, dim), dtype=np.float64)
    for s_idx, s in enumerate(basis):
        diag = 0.0
        for j in range(N - 1):
            bj  = (s >> j)      & 1
            bj1 = (s >> (j + 1)) & 1
            sz_j   = 0.5 - bj
            sz_j1  = 0.5 - bj1
            diag  += J * delta * sz_j * sz_j1
            if bj != bj1:
                t_state = s ^ (1 << j) ^ (1 << (j + 1))
                t_idx = idx_map.get(int(t_state))
                if t_idx is not None:
                    H[s_idx, t_idx] += 0.5 * J
        H[s_idx, s_idx] += diag

    # ── Diagonalize — eigenvectors only if needed ────────────────────────────
    if do_spectral:
        evals, evecs = np.linalg.eigh(H)
        psi0 = evecs[:, 0]   # ground state
    else:
        evals = np.linalg.eigvalsh(H)
        evecs = None
        psi0 = None
    E0 = evals[0]

    # ── S(k,omega) — dynamical structure factor ──────────────────────────────
    k_vals = [2.0 * np.pi * ik / N for ik in range(N)]
    omega_vals = list(np.linspace(omega_min, omega_max, n_omega))
    spectral_flat: list[float] = []

    if do_spectral:
        inv_sqrtN = 1.0 / np.sqrt(N)
        lorentz_pi = eta / np.pi
        dE = evals - E0

        for k in k_vals[:n_k]:
            # Build S^z_k |GS> in the Sz=0 subspace
            skz_gs_re = np.zeros(dim)
            skz_gs_im = np.zeros(dim)
            for s_idx, s in enumerate(basis):
                for j in range(N):
                    sz_j = 0.5 - ((s >> j) & 1)
                    phase_k = k * j
                    ck = np.cos(phase_k) * sz_j * inv_sqrtN
                    sk = np.sin(phase_k) * sz_j * inv_sqrtN
                    skz_gs_re[s_idx] += ck * psi0[s_idx]
                    skz_gs_im[s_idx] += sk * psi0[s_idx]

            # Project onto eigenstates: amp_n = <n|S^z_k|GS>
            amps_re = evecs.T @ skz_gs_re   # shape (dim,)
            amps_im = evecs.T @ skz_gs_im
            weights = amps_re**2 + amps_im**2  # |<n|S^z_k|GS>|^2

            row: list[float] = []
            for w in omega_vals:
                val = float(np.sum(weights * lorentz_pi / ((w - dE)**2 + eta**2)))
                row.append(val)
            spectral_flat.extend(row)

    # ── Imaginary-time evolution ─────────────────────────────────────────────
    tau_vals_np = np.linspace(0, tau_max, n_tau)
    energy_tau: list[float] = []
    fidelity_tau: list[float] = []

    if do_imagtime:
        # Use first basis state as initial state to show non-trivial convergence.
        # c_n^2 = |<n|s_0>|^2 — overlaps with all eigenstates.
        if evecs is not None:
            c_sq = evecs[0, :]**2
        else:
            # eigenvectors not computed yet; compute them now just for imagtime
            _, evecs_it = np.linalg.eigh(H)
            c_sq = evecs_it[0, :]**2
        for tau in tau_vals_np:
            weights_tau = c_sq * np.exp(-2.0 * tau * (evals - E0))
            Z = float(np.sum(weights_tau))
            if Z < 1e-300:
                energy_tau.append(float(E0))
                fidelity_tau.append(1.0)
            else:
                E_tau = float(np.sum(evals * weights_tau) / Z)
                F_tau = float(weights_tau[0] / Z)
                energy_tau.append(E_tau)
                fidelity_tau.append(F_tau)

    return {
        "ok": True,
        "n_sites": N,
        "J": J,
        "delta": delta,
        "h": h,
        "scenario": scenario,
        "E_ground": float(E0),
        "k_vals": k_vals[:n_k],
        "omega_vals": omega_vals,
        "spectral": spectral_flat,
        "n_k": min(n_k, N),
        "n_omega": n_omega,
        "tau_vals": [float(t) for t in tau_vals_np],
        "energy_tau": energy_tau,
        "fidelity_tau": fidelity_tau,
        "source": "python-reference",
    }


def _spectral_page(data: dict, title: str = "Spectral Functions — S(k,ω) & Imaginary Time") -> str:
    """2-panel HTML page: S(k,ω) heatmap with DCP dispersion + imaginary-time convergence."""
    import json as _json  # noqa: PLC0415

    jdata = _json.dumps(data)
    E0    = data.get("E_ground", 0.0)
    N     = data.get("n_sites", 12)
    n_k   = data.get("n_k", 20)
    n_om  = data.get("n_omega", 200)
    J_val = data.get("J", 1.0)
    d_val = data.get("delta", 1.0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body{{margin:0;background:#0d1117;color:#e6edf3;font-family:monospace;}}
  h1{{text-align:center;padding:16px;font-size:1.1rem;color:#388bfd;}}
  .subtitle{{text-align:center;font-size:0.8rem;color:#8b949e;margin-bottom:8px;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;}}
  canvas{{display:block;}}
  .label{{font-size:0.7rem;color:#8b949e;text-align:center;margin-top:4px;}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">
  N={N}, J={J_val:.2f}, &Delta;={d_val:.2f} &nbsp;|&nbsp;
  E<sub>0</sub>={E0:.4f} &nbsp;|&nbsp;
  n<sub>k</sub>={n_k}, n<sub>&omega;</sub>={n_om}
</div>
<div class="grid">
  <div class="panel">
    <canvas id="cSpectral" width="480" height="360"></canvas>
    <div class="label">S(k,&omega;) — dynamical structure factor (des&nbsp;Cloizeaux&ndash;Pearson dispersion dashed)</div>
  </div>
  <div class="panel">
    <canvas id="cImagTime" width="480" height="360"></canvas>
    <div class="label">Imaginary-time convergence &lang;E&rang;(&tau;)&nbsp;&rarr;&nbsp;E<sub>0</sub></div>
  </div>
</div>
<script>
var DATA = {jdata};

// ── Viridis-like colormap ────────────────────────────────────────────────────
function viridisCmap(t) {{
  t = Math.max(0, Math.min(1, t));
  var r = Math.round(255*(0.267 + t*(0.004 + t*(1.623 + t*(-2.233 + t*(1.342))))));
  var g = Math.round(255*(0.005 + t*(0.361 + t*(0.850 + t*(-0.971 + t*(0.412))))));
  var b = Math.round(255*(0.329 + t*(1.098 + t*(-1.468 + t*(1.151 + t*(-0.502))))));
  return [Math.max(0,Math.min(255,r)), Math.max(0,Math.min(255,g)), Math.max(0,Math.min(255,b))];
}}

// ── Panel 1: S(k,omega) heatmap ──────────────────────────────────────────────
(function() {{
  var cv = document.getElementById("cSpectral"); if (!cv) return;
  var nk = DATA.n_k, nom = DATA.n_omega;
  var spec = DATA.spectral;
  var smax = 0;
  for (var i = 0; i < spec.length; i++) if (spec[i] > smax) smax = spec[i];
  if (smax === 0) smax = 1;

  var W = cv.width, H = cv.height;
  var pad = 40;
  var ctx = cv.getContext("2d");
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0, 0, W, H);

  var pw = Math.max(1, (W - 2*pad) / nom);
  var ph = Math.max(1, (H - 2*pad) / nk);

  for (var ki = 0; ki < nk; ki++) {{
    for (var wi = 0; wi < nom; wi++) {{
      var val = spec[ki * nom + wi] / smax;
      var rgb = viridisCmap(Math.pow(val, 0.5));
      ctx.fillStyle = "rgb(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")";
      var px = pad + wi * pw;
      var py = H - pad - (ki + 1) * ph;
      ctx.fillRect(px, py, pw + 0.5, ph + 0.5);
    }}
  }}

  // DCP dispersion overlay: omega_DCP(k) = (pi/2) * |sin(k)|  (for J=1, delta=1)
  var J = DATA.J || 1.0;
  var omMin = DATA.omega_vals[0], omMax = DATA.omega_vals[nom - 1];
  var omRange = omMax - omMin || 1;
  var kArr = DATA.k_vals;
  ctx.strokeStyle = "rgba(255,200,50,0.85)"; ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  for (var ki2 = 0; ki2 < kArr.length; ki2++) {{
    var dcp = J * Math.PI * 0.5 * Math.abs(Math.sin(kArr[ki2]));
    var px2 = pad + (dcp - omMin) / omRange * (W - 2*pad);
    var py2 = H - pad - (ki2 + 0.5) * ph;
    if (ki2 === 0) ctx.moveTo(px2, py2); else ctx.lineTo(px2, py2);
  }}
  ctx.stroke();
  // Lower boundary: omega = J*pi/2*|sin(k)| (same for XXZ at delta=1)
  ctx.setLineDash([]);

  // Axes labels
  ctx.fillStyle = "#8b949e"; ctx.font = "10px monospace";
  ctx.fillText("k", 4, pad - 4);
  ctx.fillText("\u03c9", W - pad + 2, H - pad + 12);
  ctx.fillText(omMin.toFixed(1), pad, H - pad + 12);
  ctx.fillText(omMax.toFixed(1), W - pad - 18, H - pad + 12);
  ctx.fillText("0", pad - 10, H - pad);
  ctx.fillText("2\u03c0", pad - 22, pad + 8);
}})();

// ── Panel 2: imaginary-time convergence ──────────────────────────────────────
(function() {{
  var cv = document.getElementById("cImagTime"); if (!cv) return;
  var tau = DATA.tau_vals, Etau = DATA.energy_tau, F = DATA.fidelity_tau;
  if (!tau || tau.length === 0) return;
  var W = cv.width, H = cv.height, pad = 44;
  var ctx = cv.getContext("2d");
  ctx.fillStyle = "#0d1117"; ctx.fillRect(0, 0, W, H);

  // Draw <E>(tau)
  var emin = Math.min.apply(null, Etau), emax = Math.max.apply(null, Etau);
  var de = emax - emin || 1;
  var tmax = tau[tau.length - 1] || 1;

  ctx.strokeStyle = "#21262d"; ctx.lineWidth = 1;
  for (var gi = 0; gi < 5; gi++) {{
    var gy = pad + (H - 2*pad) * gi / 4;
    ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(W - pad, gy); ctx.stroke();
  }}

  // E0 reference line
  var E0 = DATA.E_ground;
  var e0py = H - pad - (E0 - emin) / de * (H - 2*pad);
  ctx.strokeStyle = "#e3b341"; ctx.lineWidth = 1; ctx.setLineDash([5, 3]);
  ctx.beginPath(); ctx.moveTo(pad, e0py); ctx.lineTo(W - pad, e0py); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#e3b341"; ctx.font = "9px monospace"; ctx.fillText("E\u2080", W - pad - 14, e0py - 3);

  // <E>(tau) curve
  ctx.strokeStyle = "#58a6ff"; ctx.lineWidth = 2; ctx.beginPath();
  for (var i = 0; i < tau.length; i++) {{
    var px = pad + tau[i] / tmax * (W - 2*pad);
    var py = H - pad - (Etau[i] - emin) / de * (H - 2*pad);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }}
  ctx.stroke();

  // Fidelity on right axis (green)
  var fmax = Math.max.apply(null, F) || 1;
  ctx.strokeStyle = "#57ab5a"; ctx.lineWidth = 1.5; ctx.beginPath();
  for (var i2 = 0; i2 < tau.length; i2++) {{
    var px2 = pad + tau[i2] / tmax * (W - 2*pad);
    var py2 = H - pad - F[i2] / fmax * (H - 2*pad);
    if (i2 === 0) ctx.moveTo(px2, py2); else ctx.lineTo(px2, py2);
  }}
  ctx.stroke();

  ctx.fillStyle = "#8b949e"; ctx.font = "10px monospace";
  ctx.fillText("\u27e8E\u27e9(\u03c4)", 4, 14);
  ctx.fillStyle = "#57ab5a";
  ctx.fillText("fidelity", 4, 26);
  ctx.fillStyle = "#8b949e";
  ctx.fillText("\u03c4", W - pad + 2, H - pad + 12);
  ctx.fillText("0", pad - 10, H - pad + 12);
  ctx.fillText(tmax.toFixed(1), W - pad - 16, H - pad + 12);
}})();
</script>
</body>
</html>"""


def _tenfold_frames(
    az_class: str = "BDI",
    dimension: int = 1,
    model: str = "ssh",
    scenario: str = "phase_diagram",
    t1: float = 1.0,
    t2: float = 0.5,
    m: float = 1.0,
    t: float = 1.0,
    t_hop: float = 1.0,
    mu: float = 0.0,
    n_k: int = 100,
    sweep_param: str = "t2",
    param_min: float = -2.0,
    param_max: float = 2.0,
    n_sweep: int = 80,
) -> dict:
    """NumPy reference: AZ classification table + topological invariants."""
    import numpy as np  # noqa: PLC0415

    # ── AZ periodic table (static) ──────────────────────────────────────────
    AZ_TABLE = [
        # (class, dim, inv_type, T2, C2, S)
        ("A",    1, "0",  0,  0, 0), ("A",    2, "Z",  0,  0, 0), ("A",    3, "0",  0,  0, 0),
        ("AI",   1, "0", +1,  0, 0), ("AI",   2, "0", +1,  0, 0), ("AI",   3, "0", +1,  0, 0),
        ("AII",  1, "0", -1,  0, 0), ("AII",  2, "Z2",-1,  0, 0), ("AII",  3, "Z2",-1,  0, 0),
        ("AIII", 1, "Z",  0,  0, 1), ("AIII", 2, "0",  0,  0, 1), ("AIII", 3, "Z",  0,  0, 1),
        ("BDI",  1, "Z", +1, +1, 1), ("BDI",  2, "0", +1, +1, 1), ("BDI",  3, "0", +1, +1, 1),
        ("CII",  1, "2Z",-1, -1, 1), ("CII",  2, "0", -1, -1, 1), ("CII",  3, "Z", -1, -1, 1),
        ("D",    1, "Z2", 0, +1, 0), ("D",    2, "Z",  0, +1, 0), ("D",    3, "0",  0, +1, 0),
        ("DIII", 1, "Z2",-1, +1, 1), ("DIII", 2, "Z2",-1, +1, 1), ("DIII", 3, "Z", -1, +1, 1),
        ("C",    1, "0",  0, -1, 0), ("C",    2, "Z",  0, -1, 0), ("C",    3, "0",  0, -1, 0),
        ("CI",   1, "0", +1, -1, 1), ("CI",   2, "0", +1, -1, 1), ("CI",   3, "Z", +1, -1, 1),
    ]
    inv_lookup: dict = {(r[0], r[1]): r[2] for r in AZ_TABLE}

    def ssh_winding(t1_: float, t2_: float, nk: int) -> int:
        # Accumulate phase increments arg(conj(d0)*d1); each step is O(2pi/nk).
        # The product-of-unit-links formula (conj(d0)/r0)*(d1/r1) telescopes
        # to 1 for any periodic d(k) and gives W=0 always -- do not use it.
        total_phase = 0.0
        for m_ in range(nk):
            k0 = 2 * np.pi * m_ / nk
            k1 = 2 * np.pi * (m_ + 1) / nk
            d0 = complex(t1_ + t2_ * np.cos(k0), t2_ * np.sin(k0))
            d1 = complex(t1_ + t2_ * np.cos(k1), t2_ * np.sin(k1))
            r0, r1 = abs(d0), abs(d1)
            if r0 < 1e-13 or r1 < 1e-13:
                return 0
            total_phase += np.angle(d0.conjugate() * d1)
        return int(round(total_phase / (2 * np.pi)))

    def qwz_chern(m_: float, t_: float, nk: int) -> int:
        dk = 2 * np.pi / nk
        chern = 0.0
        for ix in range(nk):
            for iy in range(nk):
                kx = -np.pi + ix * dk
                ky = -np.pi + iy * dk
                def evec(kx_, ky_):
                    dx = t_ * np.sin(kx_)
                    dy = t_ * np.sin(ky_)
                    dz = m_ + t_ * (np.cos(kx_) + np.cos(ky_))
                    r = np.sqrt(dx**2 + dy**2 + dz**2)
                    if r < 1e-15:
                        return np.array([1.0+0j, 0.0+0j])
                    n2 = np.sqrt(2 * r * (r + dz))
                    if n2 < 1e-15:
                        # South pole (dz=-r, dx=dy=0): lower eigenvector of -r*sz is [1,0]
                        return np.array([1.0+0j, 0.0+0j])
                    return np.array([(dx + 1j*dy)/n2, -(r+dz)/n2])
                u00 = evec(kx,    ky)
                u10 = evec(kx+dk, ky)
                u11 = evec(kx+dk, ky+dk)
                u01 = evec(kx,    ky+dk)
                F = (np.vdot(u00, u10) * np.vdot(u10, u11) *
                     np.vdot(u11, u01) * np.vdot(u01, u00))
                r = abs(F)
                if r > 1e-15:
                    F /= r
                chern += np.angle(F)
        return int(round(chern / (2 * np.pi)))

    def kitaev_z2(t_: float, mu_: float) -> int:
        pf0  = -(mu_ + 2 * t_)
        pfpi = -(mu_ - 2 * t_)
        if abs(pf0) < 1e-14 or abs(pfpi) < 1e-14:
            return 0
        Q = (1.0 if pf0 > 0 else -1.0) * (1.0 if pfpi > 0 else -1.0)
        return 1 if Q < 0 else 0

    def compute_invariant(model_: str, t1_: float, t2_: float,
                           m_: float, t_: float, t_hop_: float, mu_: float,
                           nk: int) -> tuple[float, int]:
        if model_ == "ssh":
            w = ssh_winding(t1_, t2_, nk)
            return float(w), 0
        elif model_ == "qwz":
            c = qwz_chern(m_, t_, nk)
            return float(c), 0
        elif model_ == "kitaev":
            q = kitaev_z2(t_hop_, mu_)
            return float(q), q
        return 0.0, 0

    inv_type = inv_lookup.get((az_class, dimension), "?")

    # Single-point invariant
    inv_val, inv_z2 = compute_invariant(model, t1, t2, m, t, t_hop, mu, n_k)

    # SSH band dispersion
    k_arr = np.linspace(0, 2 * np.pi, n_k)
    e_plus_arr  = [float(np.sqrt((t1 + t2*np.cos(k))**2 + (t2*np.sin(k))**2)) for k in k_arr]
    e_minus_arr = [-v for v in e_plus_arr]

    # Phase diagram sweep
    sweep_vals: list[float] = []
    sweep_inv: list[float] = []
    sweep_z2_list: list[int] = []

    if scenario == "phase_diagram":
        for k_idx in range(n_sweep):
            p = param_min + (param_max - param_min) * k_idx / max(n_sweep - 1, 1)
            sweep_vals.append(p)
            t1_ = t1; t2_ = t2; m_ = m; t_ = t; th_ = t_hop; mu_ = mu
            if   sweep_param == "t1": t1_ = p
            elif sweep_param == "t2": t2_ = p
            elif sweep_param == "m":  m_  = p
            elif sweep_param == "mu": mu_ = p
            sv, sz = compute_invariant(model, t1_, t2_, m_, t_, th_, mu_, n_k)
            sweep_inv.append(sv)
            sweep_z2_list.append(sz)

    # Build AZ table as 10x3 grid for the visualiser
    az_classes = ["A","AI","AII","AIII","BDI","CII","D","DIII","C","CI"]
    table_data = []
    for cls in az_classes:
        row = []
        for d in [1, 2, 3]:
            row.append(inv_lookup.get((cls, d), "?"))
        table_data.append({"class": cls, "d1": row[0], "d2": row[1], "d3": row[2]})

    return {
        "ok": True,
        "az_class": az_class,
        "dimension": dimension,
        "model": model,
        "scenario": scenario,
        "invariant_type": inv_type,
        "invariant_value": inv_val,
        "invariant_z2": inv_z2,
        "t1": t1, "t2": t2, "m": m, "t": t, "t_hop": t_hop, "mu": mu,
        "k_bands": [float(k) for k in k_arr],
        "e_plus": e_plus_arr,
        "e_minus": e_minus_arr,
        "sweep_vals": sweep_vals,
        "sweep_invariant": sweep_inv,
        "sweep_z2": sweep_z2_list,
        "sweep_param": sweep_param,
        "table_data": table_data,
    }


def _tenfold_page(data: dict, title: str = "10-Fold Way — Topological Classification") -> str:
    """3-panel HTML: AZ table, band dispersion, phase diagram."""
    import json as _json  # noqa: PLC0415

    jdata     = _json.dumps(data)
    cls       = data.get("az_class", "BDI")
    dim       = data.get("dimension", 1)
    inv_type  = data.get("invariant_type", "?")
    inv_val   = data.get("invariant_value", 0.0)
    model     = data.get("model", "ssh")
    sweep_p   = data.get("sweep_param", "t2")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  body{{margin:0;background:#0d1117;color:#e6edf3;font-family:monospace;}}
  h1{{text-align:center;padding:16px;font-size:1.1rem;color:#388bfd;}}
  .subtitle{{text-align:center;font-size:0.8rem;color:#8b949e;margin-bottom:8px;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px;}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;}}
  .panel-wide{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px;
               grid-column:1/-1;}}
  canvas{{display:block;width:100%;}}
  .label{{font-size:0.7rem;color:#8b949e;text-align:center;margin-top:4px;}}
  table{{border-collapse:collapse;width:100%;font-size:0.75rem;}}
  th,td{{border:1px solid #30363d;padding:4px 8px;text-align:center;}}
  th{{background:#161b22;color:#8b949e;}}
  td.Z{{color:#58a6ff;}} td.Z2{{color:#e3b341;}} td.twoZ{{color:#bc8cff;}}
  td.triv{{color:#484f58;}}
  td.active{{border:2px solid #57ab5a !important;background:#1c2e1c;color:#57ab5a;font-weight:bold;}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">
  Class <b>{cls}</b>, d={dim} &nbsp;|&nbsp;
  Invariant: <b>{inv_type}</b> &nbsp;|&nbsp;
  Model: <b>{model}</b> &nbsp;|&nbsp;
  Value: <b>{inv_val:.0f}</b>
</div>
<div class="grid">
  <div class="panel">
    <div id="azTable"></div>
    <div class="label">Altland-Zirnbauer periodic table — highlighted cell = selected (class, dim)</div>
  </div>
  <div class="panel">
    <canvas id="cBands" height="260"></canvas>
    <div class="label">Band dispersion E\u00b1(k) for {model} model (gap = topologically relevant)</div>
  </div>
  <div class="panel-wide">
    <canvas id="cPhase" height="200"></canvas>
    <div class="label">Topological invariant vs {sweep_p} — integer steps mark phase transitions</div>
  </div>
</div>
<script>
var DATA = {jdata};

// ── Panel 1: AZ Table ────────────────────────────────────────────────────────
(function() {{
  var rows = DATA.table_data;
  var selCls = DATA.az_class, selDim = DATA.dimension;
  var html = '<table><tr><th>Class</th><th>d=1</th><th>d=2</th><th>d=3</th></tr>';
  rows.forEach(function(r) {{
    html += '<tr>';
    html += '<td style="color:#e6edf3;font-weight:bold">' + r.class + '</td>';
    [['d1',1],['d2',2],['d3',3]].forEach(function(dc) {{
      var v = r[dc[0]], d = dc[1];
      var cls2 = v==='Z' ? 'Z' : v==='Z2' ? 'Z2' : v==='2Z' ? 'twoZ' : 'triv';
      var active = (r.class===selCls && d===selDim) ? ' active' : '';
      var disp = v==='0' ? '\u2014' : v;
      html += '<td class="' + cls2 + active + '">' + disp + '</td>';
    }});
    html += '</tr>';
  }});
  html += '</table>';
  document.getElementById('azTable').innerHTML = html;
}})();

// ── Panel 2: Band dispersion ─────────────────────────────────────────────────
(function() {{
  var cv = document.getElementById('cBands'); if (!cv) return;
  cv.width = cv.offsetWidth || 460; cv.height = 260;
  var ctx = cv.getContext('2d'), W = cv.width, H = cv.height, pad = 36;
  ctx.fillStyle = '#0d1117'; ctx.fillRect(0,0,W,H);
  var ks = DATA.k_bands, ep = DATA.e_plus, em = DATA.e_minus;
  if (!ks || ks.length === 0) return;
  var kmax = ks[ks.length-1] || 1;
  var emax = Math.max.apply(null, ep), emin = Math.min.apply(null, em);
  var er = emax - emin || 1;
  // Grid
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
  for (var gi=0; gi<5; gi++) {{
    var gy = pad + (H-2*pad)*gi/4;
    ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(W-pad, gy); ctx.stroke();
  }}
  // E=0 line
  var z0 = H - pad - (0-emin)/er*(H-2*pad);
  ctx.strokeStyle = '#484f58'; ctx.lineWidth = 1; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(pad,z0); ctx.lineTo(W-pad,z0); ctx.stroke();
  ctx.setLineDash([]);
  // Upper band
  ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 2; ctx.beginPath();
  for (var i=0; i<ks.length; i++) {{
    var px = pad + ks[i]/kmax*(W-2*pad);
    var py = H - pad - (ep[i]-emin)/er*(H-2*pad);
    if (i===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
  }}
  ctx.stroke();
  // Lower band
  ctx.strokeStyle = '#f47067'; ctx.lineWidth = 2; ctx.beginPath();
  for (var i2=0; i2<ks.length; i2++) {{
    var px2 = pad + ks[i2]/kmax*(W-2*pad);
    var py2 = H - pad - (em[i2]-emin)/er*(H-2*pad);
    if (i2===0) ctx.moveTo(px2,py2); else ctx.lineTo(px2,py2);
  }}
  ctx.stroke();
  ctx.fillStyle = '#8b949e'; ctx.font = '10px monospace';
  ctx.fillText('E', 4, 14); ctx.fillText('k', W-pad+2, H-pad+12);
  ctx.fillText('0', pad-20, z0+4);
}})();

// ── Panel 3: Phase diagram ───────────────────────────────────────────────────
(function() {{
  var cv = document.getElementById('cPhase'); if (!cv) return;
  cv.width = cv.offsetWidth || 920; cv.height = 200;
  var ctx = cv.getContext('2d'), W = cv.width, H = cv.height, pad = 44;
  ctx.fillStyle = '#0d1117'; ctx.fillRect(0,0,W,H);
  var sv = DATA.sweep_vals, si = DATA.sweep_invariant;
  if (!sv || sv.length === 0) return;
  var smin = sv[0], smax = sv[sv.length-1], sr = smax-smin||1;
  var imin = Math.min.apply(null,si)-0.5, imax = Math.max.apply(null,si)+0.5;
  var ir = imax-imin||1;
  // Grid
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
  for (var gi=0; gi<5; gi++) {{
    var gy = pad + (H-2*pad)*gi/4;
    ctx.beginPath(); ctx.moveTo(pad,gy); ctx.lineTo(W-pad,gy); ctx.stroke();
  }}
  // W=0 reference
  var z0 = H - pad - (0-imin)/ir*(H-2*pad);
  ctx.strokeStyle = '#484f58'; ctx.setLineDash([4,3]); ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,z0); ctx.lineTo(W-pad,z0); ctx.stroke();
  ctx.setLineDash([]);
  // Step-function invariant
  ctx.strokeStyle = '#57ab5a'; ctx.lineWidth = 2.5; ctx.beginPath();
  for (var i=0; i<sv.length; i++) {{
    var px = pad + (sv[i]-smin)/sr*(W-2*pad);
    var py = H - pad - (si[i]-imin)/ir*(H-2*pad);
    if (i===0) ctx.moveTo(px,py);
    else {{
      // Step: horizontal then vertical
      var ppx = pad + (sv[i-1]-smin)/sr*(W-2*pad);
      ctx.lineTo((ppx+px)/2, H-pad-(si[i-1]-imin)/ir*(H-2*pad));
      ctx.lineTo((ppx+px)/2, py);
      ctx.lineTo(px,py);
    }}
  }}
  ctx.stroke();
  // Transition markers
  for (var i2=1; i2<sv.length; i2++) {{
    if (Math.round(si[i2]) !== Math.round(si[i2-1])) {{
      var mx = pad + ((sv[i2]+sv[i2-1])/2-smin)/sr*(W-2*pad);
      ctx.strokeStyle = '#e3b341'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(mx,pad); ctx.lineTo(mx,H-pad); ctx.stroke();
      ctx.setLineDash([]);
    }}
  }}
  ctx.fillStyle = '#8b949e'; ctx.font = '10px monospace';
  ctx.fillText('W', 4, 14);
  ctx.fillText(DATA.sweep_param, W-pad+2, H-pad+12);
  ctx.fillText(smin.toFixed(1), pad, H-pad+12);
  ctx.fillText(smax.toFixed(1), W-pad-18, H-pad+12);
}})();
</script>
</body>
</html>"""


