from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _wilson_loop_frames(
    M: float = 1.0,
    A: float = 1.0,
    B: float = 1.0,
    C: float = 0.0,
    D: float = 0.0,
    n_kx: int = 40,
    n_ky: int = 40,
    scenario: str = "phase_diagram",
    sweep_param: str = "M",
    param_min: float = -1.0,
    param_max: float = 5.0,
    n_sweep: int = 40,
    source: str = "python-reference",
) -> dict:
    """NumPy reference: Wilson loop / Wannier Charge Centers for BHZ model."""
    import numpy as np  # noqa: PLC0415

    # ── BHZ 4×4 Hamiltonian (lattice) ────────────────────────────────────
    # Γ₁=σx⊗σx, Γ₂=σx⊗σy, Γ₅=σz⊗I
    # H = ε I₄ + d₁Γ₁ + d₂Γ₂ + d₅Γ₅
    # ε = C - 2D(2-cos kx-cos ky)
    # d₁ = A sin kx,  d₂ = A sin ky
    # d₅ = M - 2B(2-cos kx-cos ky)
    def bhz_ham(kx, ky):
        eps = C - 2*D*(2 - np.cos(kx) - np.cos(ky))
        d1  = A * np.sin(kx)
        d2  = A * np.sin(ky)
        d5  = M - 2*B*(2 - np.cos(kx) - np.cos(ky))
        # Explicit 4×4:
        # H[0][0]=ε+d5, H[0][3]=d1-id2
        # H[1][1]=ε+d5, H[1][2]=d1+id2
        # H[2][1]=d1-id2, H[2][2]=ε-d5
        # H[3][0]=d1+id2, H[3][3]=ε-d5
        H = np.zeros((4, 4), dtype=complex)
        H[0, 0] = eps + d5;  H[0, 3] = d1 - 1j*d2
        H[1, 1] = eps + d5;  H[1, 2] = d1 + 1j*d2
        H[2, 1] = d1 - 1j*d2; H[2, 2] = eps - d5
        H[3, 0] = d1 + 1j*d2; H[3, 3] = eps - d5
        return H

    def lower_evecs(kx, ky):
        """Return 4×2 lower eigenvector matrix."""
        H = bhz_ham(kx, ky)
        evals, evecs = np.linalg.eigh(H)
        # eigh returns sorted eigenvalues; lower 2 are indices 0,1
        return evecs[:, :2]

    def wilson_wcc(ky_val):
        """Compute 2 WCC values for a fixed ky."""
        kxs = 2 * np.pi * np.arange(n_kx) / n_kx
        W = np.eye(2, dtype=complex)
        for ix in range(n_kx):
            kx0 = kxs[ix]
            kx1 = 2 * np.pi * ((ix + 1) % n_kx) / n_kx
            Q0 = lower_evecs(kx0, ky_val)
            Q1 = lower_evecs(kx1, ky_val)
            Ov = Q0.conj().T @ Q1   # 2×2 overlap
            W = W @ Ov
        evals = np.linalg.eigvals(W)
        wcc = sorted((-np.angle(ev) / (2 * np.pi)) % 1.0 for ev in evals)
        return wcc  # list of 2 values in [0,1)

    def z2_from_wcc_grid(wcc_grid):
        """Count θ=0.5 crossings over BOTH WCC branches with continuity tracking.

        Sorting at each step loses branch connectivity: when the two WCC bands
        wind past each other, sorting swaps their labels, creating false crossings
        and missing real ones.  We instead assign each new pair of WCC values to
        the existing tracked branches by minimising total circular distance, then
        count sign changes of (branch - 0.5) on each branch independently.
        """
        n = len(wcc_grid)
        if n < 2:
            return 0

        def circ_dist(a, b):
            d = abs(a - b)
            return 1.0 - d if d > 0.5 else d

        track = list(wcc_grid[0])   # [branch0, branch1]
        crossings = 0

        for i in range(1, n):
            c0, c1 = wcc_grid[i][0], wcc_grid[i][1]
            d_straight = circ_dist(c0, track[0]) + circ_dist(c1, track[1])
            d_crossed  = circ_dist(c1, track[0]) + circ_dist(c0, track[1])

            if d_straight <= d_crossed:
                nxt = [c0, c1]
            else:
                nxt = [c1, c0]

            for b in range(2):
                w0 = track[b] - 0.5
                w1 = nxt[b]   - 0.5
                if w0 * w1 < 0:
                    crossings += 1

            track = nxt

        return crossings % 2

    def compute_wcc_spectrum(M_val):
        """Full WCC spectrum: ky in [0, π]."""
        ky_vals = np.linspace(0, np.pi, n_ky).tolist()
        wcc_grid = []
        for ky in ky_vals:
            # Temporarily patch closure M → M_val
            nonlocal M
            M_save = M
            M = M_val
            w = wilson_wcc(ky)
            M = M_save
            wcc_grid.append(w)
        z2 = z2_from_wcc_grid(wcc_grid)
        return ky_vals, [w[0] for w in wcc_grid], [w[1] for w in wcc_grid], z2

    def gap_at(kx, ky, M_val):
        nonlocal M
        M_save = M
        M = M_val
        H = bhz_ham(kx, ky)
        M = M_save
        evals = np.linalg.eigvalsh(H)
        return float(abs(evals[2] - evals[1]))

    # ── Full WCC spectrum at nominal M ───────────────────────────────────
    ky_vals, wcc1, wcc2, z2 = compute_wcc_spectrum(M)

    phase_label = f"Topological (Z\u2082=1)" if z2 == 1 else f"Trivial (Z\u2082=0)"

    # ── Phase diagram sweep ───────────────────────────────────────────────
    if sweep_param != "M":
        raise ValueError(f"_wilson_loop_frames: unsupported sweep_param {sweep_param!r} — use 'M'")
    sweep_vals = np.linspace(param_min, param_max, n_sweep).tolist()
    sweep_z2 = []
    sweep_phase = []
    for Mp in sweep_vals:
        # Use coarser grid for sweep
        ky_s = np.linspace(0, np.pi, max(8, n_ky // 4)).tolist()
        wcc_g = []
        for ky in ky_s:
            nonlocal_M_save = M
            M_tmp = Mp
            # direct compute without closure patch
            kxs = 2 * np.pi * np.arange(n_kx) / n_kx
            W = np.eye(2, dtype=complex)
            for ix in range(n_kx):
                kx0 = kxs[ix]
                kx1 = 2 * np.pi * ((ix + 1) % n_kx) / n_kx
                eps0 = C - 2*D*(2 - np.cos(kx0) - np.cos(ky))
                d1_0 = A * np.sin(kx0);  d2_0 = A * np.sin(ky)
                d5_0 = M_tmp - 2*B*(2 - np.cos(kx0) - np.cos(ky))
                H0 = np.zeros((4,4),dtype=complex)
                H0[0,0]=eps0+d5_0; H0[0,3]=d1_0-1j*d2_0
                H0[1,1]=eps0+d5_0; H0[1,2]=d1_0+1j*d2_0
                H0[2,1]=d1_0-1j*d2_0; H0[2,2]=eps0-d5_0
                H0[3,0]=d1_0+1j*d2_0; H0[3,3]=eps0-d5_0
                eps1 = C - 2*D*(2 - np.cos(kx1) - np.cos(ky))
                d1_1 = A * np.sin(kx1);  d2_1 = A * np.sin(ky)
                d5_1 = M_tmp - 2*B*(2 - np.cos(kx1) - np.cos(ky))
                H1 = np.zeros((4,4),dtype=complex)
                H1[0,0]=eps1+d5_1; H1[0,3]=d1_1-1j*d2_1
                H1[1,1]=eps1+d5_1; H1[1,2]=d1_1+1j*d2_1
                H1[2,1]=d1_1-1j*d2_1; H1[2,2]=eps1-d5_1
                H1[3,0]=d1_1+1j*d2_1; H1[3,3]=eps1-d5_1
                _, ev0 = np.linalg.eigh(H0); Q0 = ev0[:,:2]
                _, ev1 = np.linalg.eigh(H1); Q1 = ev1[:,:2]
                W = W @ (Q0.conj().T @ Q1)
            evals = np.linalg.eigvals(W)
            wcc_g.append(sorted((-np.angle(ev)/(2*np.pi)) % 1.0 for ev in evals))
        z2_sweep = z2_from_wcc_grid(wcc_g)
        sweep_z2.append(z2_sweep)
        sweep_phase.append("Topological" if z2_sweep == 1 else "Trivial")

    return {
        "M": M, "A": A, "B": B, "C": C, "D": D,
        "n_kx": n_kx, "n_ky": n_ky,
        "scenario": scenario,
        "z2": z2,
        "phase_label": phase_label,
        "gap_gamma": gap_at(0.0, 0.0, M),
        "gap_m_pt":  gap_at(np.pi, np.pi, M),
        "ky_vals": ky_vals,
        "wcc1": wcc1,
        "wcc2": wcc2,
        "sweep_param": sweep_param,
        "param_min": param_min,
        "param_max": param_max,
        "sweep_vals": sweep_vals,
        "sweep_z2": sweep_z2,
        "sweep_phase": sweep_phase,
        "source": source,
    }


def _wilson_loop_page(data: dict, title: str = "Wilson Loop & Wannier Charge Centers — BHZ Z₂ Topological Insulator") -> str:
    """HTML page: WCC spectrum + Z₂ phase diagram for the BHZ model."""
    import json as _json  # noqa: PLC0415
    data_json = _json.dumps(data, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  :root {{
    --bg:#0d1117; --surface:#161b22; --border:#30363d;
    --text:#e6edf3; --muted:#8b949e;
    --accent:#58a6ff; --topo:#3fb950; --trivial:#8b949e;
    --wcc1:#f78166; --wcc2:#d2a8ff;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
        display:flex;flex-direction:column;align-items:center;padding:32px 16px;gap:24px}}
  h1{{font-size:1.3rem;color:var(--accent);text-align:center}}
  .subtitle{{color:var(--muted);font-size:0.9rem;text-align:center}}
  .card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;
         padding:20px;width:100%;max-width:860px}}
  .card h2{{font-size:1rem;color:var(--muted);margin-bottom:14px}}
  .info-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
  .info-box{{background:var(--bg);border:1px solid var(--border);border-radius:6px;
             padding:12px;text-align:center}}
  .info-box .label{{font-size:0.75rem;color:var(--muted);margin-bottom:6px}}
  .info-box .value{{font-size:1.4rem;font-weight:700}}
  .topo{{color:var(--topo)}} .trivial{{color:var(--trivial)}}
  canvas{{display:block;width:100%;cursor:crosshair}}
  .legend{{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;font-size:0.8rem;color:var(--muted)}}
  .dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:4px}}
  .footer{{color:var(--muted);font-size:0.75rem;text-align:center}}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">BHZ model &nbsp;·&nbsp; H(k)=εI+d₁Γ₁+d₂Γ₂+d₅Γ₅ &nbsp;·&nbsp; Wilson loop W(ky)=∏Q†Q′ &nbsp;·&nbsp; Z₂=parity of WCC crossings</p>

<div class="card">
  <h2>Parameters &amp; Z₂ Invariant</h2>
  <div class="info-grid" id="infoGrid"></div>
</div>

<div class="card">
  <h2>Wannier Charge Centers θ(ky) — Wilson loop spectrum along kx, vs ky ∈ [0, π]</h2>
  <canvas id="wccPlot" height="300"></canvas>
  <div class="legend">
    <span><span class="dot" style="background:var(--wcc1)"></span>θ₁(ky) lower WCC</span>
    <span><span class="dot" style="background:var(--wcc2)"></span>θ₂(ky) upper WCC</span>
    <span style="border-left:1px dashed #484f58;padding-left:12px">Dashed line: θ=0.5 (crossing reference)</span>
  </div>
</div>

<div class="card">
  <h2>Z₂ Phase Diagram — Z₂ vs M &nbsp;(topological when 0&lt;M/B&lt;4, gap closes at M=0 and M=4B)</h2>
  <canvas id="phasePlot" height="200"></canvas>
  <div class="legend">
    <span><span class="dot" style="background:var(--topo)"></span>Topological (Z₂=1)</span>
    <span><span class="dot" style="background:var(--trivial)"></span>Trivial (Z₂=0)</span>
  </div>
</div>

<p class="footer">Source: {data.get("source","python-reference")} &nbsp;·&nbsp; n_kx={data.get("n_kx",40)} &nbsp;·&nbsp; n_ky={data.get("n_ky",40)}</p>

<script>
(function(){{
  const DATA = {data_json};

  // ── Info grid ────────────────────────────────────────────────────────────
  const grid = document.getElementById('infoGrid');
  const isTopo = DATA.z2 === 1;
  [
    ['M', DATA.M?.toFixed(2)],
    ['A', DATA.A?.toFixed(2)],
    ['B', DATA.B?.toFixed(2)],
    ['Z₂', DATA.z2],
    ['Phase', DATA.phase_label],
    ['Gap Γ', DATA.gap_gamma?.toFixed(3)],
    ['Gap M', DATA.gap_m_pt?.toFixed(3)],
  ].forEach(([lbl,val])=>{{
    const box=document.createElement('div'); box.className='info-box';
    const hi=(lbl==='Z₂'&&val===1)||(lbl==='Phase'&&String(val).includes('Topo'));
    box.innerHTML=`<div class="label">${{lbl}}</div><div class="value ${{hi?'topo':'trivial'}}">${{val}}</div>`;
    grid.appendChild(box);
  }});

  // ── WCC spectrum plot ────────────────────────────────────────────────────
  const wccCanvas = document.getElementById('wccPlot');
  function drawWCC(){{
    const dpr=window.devicePixelRatio||1;
    const W=wccCanvas.clientWidth, H=parseInt(wccCanvas.getAttribute('height'));
    wccCanvas.width=W*dpr; wccCanvas.height=H*dpr;
    const ctx=wccCanvas.getContext('2d'); ctx.scale(dpr,dpr);
    const pad={{l:50,r:20,t:20,b:40}};
    const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;
    const ky=DATA.ky_vals, w1=DATA.wcc1, w2=DATA.wcc2;
    if(!ky||ky.length===0)return;
    const kyMin=ky[0], kyMax=ky[ky.length-1];
    function sx(v){{return pad.l+(v-kyMin)/(kyMax-kyMin)*pw;}}
    function sy(v){{return pad.t+(1-v)*ph;}}  // v in [0,1)

    ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);

    // Grid
    ctx.strokeStyle='#21262d'; ctx.lineWidth=1;
    [0,0.25,0.5,0.75,1].forEach(v=>{{
      ctx.beginPath(); ctx.moveTo(pad.l,sy(v)); ctx.lineTo(W-pad.r,sy(v)); ctx.stroke();
    }});

    // θ=0.5 reference line
    ctx.strokeStyle='#484f58'; ctx.lineWidth=1; ctx.setLineDash([5,4]);
    ctx.beginPath(); ctx.moveTo(pad.l,sy(0.5)); ctx.lineTo(W-pad.r,sy(0.5)); ctx.stroke();
    ctx.setLineDash([]);

    // WCC curves
    function drawCurve(arr, color){{
      ctx.strokeStyle=color; ctx.lineWidth=2;
      ctx.beginPath();
      arr.forEach((v,i)=>{{
        const x=sx(ky[i]), y=sy(v);
        i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
      }});
      ctx.stroke();
    }}
    drawCurve(w1,'#f78166');
    drawCurve(w2,'#d2a8ff');

    // Axes labels
    ctx.fillStyle='#8b949e'; ctx.font='11px monospace'; ctx.textAlign='right';
    [0,0.5,1].forEach(v=>{{ ctx.fillText(v.toFixed(1),pad.l-4,sy(v)+4); }});
    ctx.textAlign='center';
    [0, Math.PI/2, Math.PI].forEach(v=>{{
      ctx.fillText(v===0?'0':v===Math.PI?'π':'π/2', sx(v), H-pad.b+14);
    }});
    ctx.fillStyle='#8b949e'; ctx.font='10px monospace';
    ctx.textAlign='left'; ctx.fillText('θ',2,pad.t+12);
    ctx.textAlign='center'; ctx.fillText('ky',W/2,H-4);
  }}
  drawWCC();

  // ── Phase diagram ────────────────────────────────────────────────────────
  const phCanvas=document.getElementById('phasePlot');
  function drawPhase(){{
    const dpr=window.devicePixelRatio||1;
    const W=phCanvas.clientWidth, H=parseInt(phCanvas.getAttribute('height'));
    phCanvas.width=W*dpr; phCanvas.height=H*dpr;
    const ctx=phCanvas.getContext('2d'); ctx.scale(dpr,dpr);
    const pad={{l:50,r:20,t:20,b:40}};
    const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;
    const sv=DATA.sweep_vals, sz=DATA.sweep_z2;
    if(!sv||sv.length===0)return;
    const smin=sv[0], smax=sv[sv.length-1];
    function sx(v){{return pad.l+(v-smin)/(smax-smin)*pw;}}
    function sy(z){{return pad.t+(1-z)*ph;}}

    ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);

    // Filled regions
    for(let i=0;i<sv.length-1;i++){{
      const c=sz[i]===1?'#3fb95033':'#8b949e22';
      ctx.fillStyle=c;
      ctx.fillRect(sx(sv[i]),pad.t,sx(sv[i+1])-sx(sv[i]),ph);
    }}

    // Step-function Z₂ line
    ctx.lineWidth=2.5;
    for(let i=0;i<sv.length-1;i++){{
      ctx.strokeStyle=sz[i]===1?'#3fb950':'#8b949e';
      ctx.beginPath(); ctx.moveTo(sx(sv[i]),sy(sz[i])); ctx.lineTo(sx(sv[i+1]),sy(sz[i])); ctx.stroke();
    }}

    // Transition markers: M=0 and M=4B
    ctx.strokeStyle='#484f58'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
    [0, 4*DATA.B].forEach(mx=>{{
      if(mx>=smin&&mx<=smax){{
        ctx.beginPath(); ctx.moveTo(sx(mx),pad.t); ctx.lineTo(sx(mx),H-pad.b); ctx.stroke();
      }}
    }});
    ctx.setLineDash([]);

    // Axes
    ctx.strokeStyle='#484f58'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l,sy(0)); ctx.lineTo(W-pad.r,sy(0)); ctx.stroke();

    ctx.fillStyle='#8b949e'; ctx.font='11px monospace'; ctx.textAlign='right';
    ctx.fillText('0',pad.l-4,sy(0)+4); ctx.fillText('1',pad.l-4,sy(1)+4);
    ctx.textAlign='center';
    [smin,(smin+smax)/2,smax].forEach(v=>{{ ctx.fillText(v.toFixed(1),sx(v),H-pad.b+14); }});
    [0,4*DATA.B].forEach(v=>{{ if(v>=smin&&v<=smax) ctx.fillText(v===0?'0':`4B=${{(4*DATA.B).toFixed(1)}}`,sx(v),H-pad.b+14); }});
    ctx.fillStyle='#8b949e'; ctx.font='10px monospace';
    ctx.textAlign='left'; ctx.fillText('Z₂',2,pad.t+12);
    ctx.textAlign='center'; ctx.fillText('M',W/2,H-4);
  }}
  drawPhase();
  window.addEventListener('resize',()=>{{drawWCC();drawPhase();}});
}})();
</script>
</body>
</html>"""


def _second_chern_frames(
    m: float = 1.0,
    t: float = 1.0,
    n_k: int = 6,
    scenario: str = "phase_diagram",
    sweep_param: str = "m",
    param_min: float = -5.0,
    param_max: float = 5.0,
    n_sweep: int = 40,
    source: str = "python-reference",
) -> dict:
    """NumPy reference: second Chern number for the Zhang-Hu 4D QHE model."""
    import numpy as np  # noqa: PLC0415

    # ── 4×4 Hamiltonian ────────────────────────────────────────────────────
    # Gamma matrices: Γ₁=σ₁⊗σ₁, Γ₂=σ₁⊗σ₂, Γ₃=σ₁⊗σ₃, Γ₄=σ₂⊗I, Γ₅=σ₃⊗I
    def build_ham(d1, d2, d3, d4, d5):
        return np.array([
            [d5,          0,            d3 - 1j*d4,  d1 - 1j*d2],
            [0,           d5,           d1 + 1j*d2, -d3 - 1j*d4],
            [d3 + 1j*d4,  d1 - 1j*d2, -d5,          0          ],
            [d1 + 1j*d2, -d3 + 1j*d4,  0,           -d5        ],
        ], dtype=complex)

    def lower_basis(m_val, t_val, k):
        """Return 4×2 orthonormal lower-band eigenvector matrix Q via projector + GS."""
        k1, k2, k3, k4 = k
        d1 = np.sin(k1)
        d2 = np.sin(k2)
        d3 = np.sin(k3)
        d4 = np.sin(k4)
        d5 = m_val + t_val * (np.cos(k1) + np.cos(k2) + np.cos(k3) + np.cos(k4))
        r = np.sqrt(d1**2 + d2**2 + d3**2 + d4**2 + d5**2)
        if r < 1e-14:
            return np.eye(4, 2, dtype=complex)
        H = build_ham(d1, d2, d3, d4, d5)
        P = (np.eye(4, dtype=complex) - H / r) / 2.0
        # Gram-Schmidt on P's columns
        basis = []
        for col in range(4):
            v = P[:, col].copy()
            for b in basis:
                v -= np.dot(b.conj(), v) * b
            nrm = np.linalg.norm(v)
            if nrm > 1e-10:
                basis.append(v / nrm)
                if len(basis) == 2:
                    break
        while len(basis) < 2:
            basis.append(np.zeros(4, dtype=complex))
        return np.column_stack(basis)  # 4×2

    def link_mat(Q0, Q1):
        """2×2 link matrix Q0†Q1, normalized so |det|=1."""
        M = Q0.conj().T @ Q1  # 2×2
        d = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        r = abs(d)
        if r > 1e-14:
            M = M / np.sqrt(r)
        return M

    def mat2_log(F):
        """Matrix log of an approximately unitary 2×2 matrix.
        Decomposes F = e^(iφ/2) W (W ∈ SU(2)), log(W) = (α/sinα)(W - cosα I)."""
        det = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
        phi = np.angle(det)
        W = F * np.exp(-0.5j * phi)
        cos_a = np.clip(0.5 * (W[0, 0] + W[1, 1]).real, -1.0, 1.0)
        alpha = np.arccos(cos_a)
        c = alpha / np.sin(alpha) if abs(np.sin(alpha)) > 1e-10 else 1.0
        logW = c * (W - cos_a * np.eye(2, dtype=complex))
        logW[0, 0] += 0.5j * phi
        logW[1, 1] += 0.5j * phi
        return logW

    def compute_c2(m_val, t_val, nk):
        N1 = nk + 1
        ks = 2 * np.pi * np.arange(nk) / nk
        grid = {}
        for i1 in range(N1):
            for i2 in range(N1):
                for i3 in range(N1):
                    for i4 in range(N1):
                        k = (ks[i1 % nk], ks[i2 % nk], ks[i3 % nk], ks[i4 % nk])
                        grid[(i1, i2, i3, i4)] = lower_basis(m_val, t_val, k)
        # Precompute 2×2 link matrices (4 directions)
        U = {}
        steps = [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]
        for i1 in range(nk):
            for i2 in range(nk):
                for i3 in range(nk):
                    for i4 in range(nk):
                        p = (i1, i2, i3, i4)
                        Q0 = grid[p]
                        for mu, s in enumerate(steps):
                            pn = (p[0]+s[0], p[1]+s[1], p[2]+s[2], p[3]+s[3])
                            U[(mu,) + p] = link_mat(Q0, grid[pn])
        # Non-Abelian plaquette and field strength
        def plaq_mat(mu, nu, p):
            i1, i2, i3, i4 = p
            sm, sn = steps[mu], steps[nu]
            pm = ((i1+sm[0])%nk, (i2+sm[1])%nk, (i3+sm[2])%nk, (i4+sm[3])%nk)
            pn = ((i1+sn[0])%nk, (i2+sn[1])%nk, (i3+sn[2])%nk, (i4+sn[3])%nk)
            return U[(mu,)+p] @ U[(nu,)+pm] @ U[(mu,)+pn].conj().T @ U[(nu,)+p].conj().T
        # C₂ = (1/8π²) Σ Re Tr(f₁₂f₃₄ - f₁₃f₂₄ + f₁₄f₂₃)
        c2_sum = 0.0
        for i1 in range(nk):
            for i2 in range(nk):
                for i3 in range(nk):
                    for i4 in range(nk):
                        p = (i1, i2, i3, i4)
                        f12 = mat2_log(plaq_mat(0, 1, p))
                        f34 = mat2_log(plaq_mat(2, 3, p))
                        f13 = mat2_log(plaq_mat(0, 2, p))
                        f24 = mat2_log(plaq_mat(1, 3, p))
                        f14 = mat2_log(plaq_mat(0, 3, p))
                        f23 = mat2_log(plaq_mat(1, 2, p))
                        c2_sum += (np.trace(f12 @ f34) - np.trace(f13 @ f24)
                                   + np.trace(f14 @ f23)).real
        return int(round(c2_sum / (8 * np.pi**2)))

    def phase_label(c2):
        if c2 == 0:
            return "Trivial (C₂=0)"
        return f"4D QHE (C₂={c2:+d})"

    # ── sweep ──────────────────────────────────────────────────────────────
    if sweep_param not in ("m", "t"):
        raise ValueError(
            f"_second_chern_frames: unsupported sweep_param {sweep_param!r} — use 'm' or 't'"
        )
    sweep_vals = np.linspace(param_min, param_max, n_sweep).tolist()
    sweep_c2 = []
    sweep_phase = []
    for p in sweep_vals:
        m_k = p if sweep_param == "m" else m
        t_k = p if sweep_param == "t" else t
        c2 = compute_c2(m_k, t_k, n_k) if t_k != 0 else 0
        sweep_c2.append(int(c2))
        sweep_phase.append(phase_label(c2))

    c2_mid = sweep_c2[n_sweep // 2] if n_sweep > 0 else compute_c2(m, t, n_k)

    return {
        "m": m,
        "t": t,
        "n_k": n_k,
        "scenario": scenario,
        "sweep_param": sweep_param,
        "param_min": param_min,
        "param_max": param_max,
        "second_chern": c2_mid,
        "phase_label": phase_label(c2_mid),
        "sweep_vals": sweep_vals,
        "sweep_c2": sweep_c2,
        "sweep_phase": sweep_phase,
        "source": source,
    }


def _second_chern_page(data: dict, title: str = "Second Chern Number — 4D Quantum Hall Effect") -> str:
    """Generate an HTML page for the second Chern number / 4D QHE phase diagram."""
    import json as _json  # noqa: PLC0415

    data_json = _json.dumps(data, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e;
    --accent: #58a6ff; --green: #3fb950; --red: #f85149;
    --topo: #d29922; --trivial: #8b949e;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif;
          display: flex; flex-direction: column; align-items: center; padding: 32px 16px; gap: 24px; }}
  h1 {{ font-size: 1.3rem; color: var(--accent); text-align: center; }}
  .subtitle {{ color: var(--muted); font-size: 0.9rem; text-align: center; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
           padding: 20px; width: 100%; max-width: 860px; }}
  .card h2 {{ font-size: 1rem; color: var(--muted); margin-bottom: 14px; }}
  .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
  .info-box {{ background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
               padding: 12px; text-align: center; }}
  .info-box .label {{ font-size: 0.75rem; color: var(--muted); margin-bottom: 6px; }}
  .info-box .value {{ font-size: 1.4rem; font-weight: 700; }}
  .topo {{ color: var(--topo); }}
  .trivial {{ color: var(--trivial); }}
  canvas {{ display: block; width: 100%; cursor: crosshair; }}
  .phase-legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 10px; font-size: 0.8rem; color: var(--muted); }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 4px; }}
  .footer {{ color: var(--muted); font-size: 0.75rem; text-align: center; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">Zhang-Hu model &nbsp;·&nbsp; H(k) = Σ dₐ(k) Γₐ &nbsp;·&nbsp; 4D lattice Fukui-Hatsugai-Suzuki</p>

<div class="card">
  <h2>Model Parameters &amp; Invariant</h2>
  <div class="info-grid" id="infoGrid"></div>
</div>

<div class="card">
  <h2>Phase Diagram: C₂ vs m &nbsp;(gap closes at m = 0, ±2, ±4)</h2>
  <canvas id="phasePlot" height="280"></canvas>
  <div class="phase-legend">
    <span><span class="legend-dot" style="background:#d29922"></span>|C₂|=3 &nbsp;(0&lt;|m|&lt;2)</span>
    <span><span class="legend-dot" style="background:#58a6ff"></span>|C₂|=1 &nbsp;(2&lt;|m|&lt;4)</span>
    <span><span class="legend-dot" style="background:#8b949e"></span>Trivial (C₂=0)</span>
  </div>
</div>

<div class="card">
  <h2>Phase Classification Rules</h2>
  <table style="width:100%;border-collapse:collapse;font-size:0.85rem;font-family:monospace">
    <thead><tr style="color:var(--muted);border-bottom:1px solid var(--border)">
      <th style="text-align:left;padding:6px">m range</th>
      <th style="text-align:center;padding:6px">C₂</th>
      <th style="text-align:left;padding:6px">Phase</th>
    </tr></thead>
    <tbody id="phaseTable"></tbody>
  </table>
</div>

<p class="footer">Source: {data.get("source","python-reference")} &nbsp;·&nbsp; k-grid: {data.get("n_k",6)}⁴ &nbsp;·&nbsp; n_sweep: {len(data.get("sweep_vals",[]))}</p>

<script>
(function() {{
  const DATA = {data_json};
  const C2_COLOR = {{
    0: '#8b949e', 1: '#58a6ff', '-1': '#58a6ff',
    3: '#d29922', '-3': '#d29922',
  }};
  function c2Color(c2) {{
    return C2_COLOR[c2] || (Math.abs(c2) >= 2 ? '#d29922' : '#58a6ff');
  }}

  // ── Info grid ───────────────────────────────────────────────────────────
  const grid = document.getElementById('infoGrid');
  const items = [
    ['m', DATA.m.toFixed(2)],
    ['t', DATA.t.toFixed(2)],
    ['k-grid (per dim)', DATA.n_k],
    ['C₂', DATA.second_chern],
    ['Phase', DATA.phase_label],
  ];
  items.forEach(([lbl, val]) => {{
    const box = document.createElement('div');
    box.className = 'info-box';
    const isTopo = (lbl === 'C₂' && val !== 0) || (lbl === 'Phase' && String(val).includes('QHE'));
    box.innerHTML = `<div class="label">${{lbl}}</div>`
                  + `<div class="value ${{isTopo ? 'topo' : 'trivial'}}">${{val}}</div>`;
    grid.appendChild(box);
  }});

  // ── Phase table ─────────────────────────────────────────────────────────
  const rows = [
    ['|m| > 4', 0, 'Trivial'],
    ['2 < |m| < 4', '±1', '4D QHE (weak)'],
    ['0 < |m| < 2', '±3', '4D QHE (strong)'],
  ];
  const tbody = document.getElementById('phaseTable');
  rows.forEach(([range, c2, phase]) => {{
    const tr = document.createElement('tr');
    tr.style.borderBottom = '1px solid #30363d';
    const color = c2 === 0 ? '#8b949e' : (String(c2).includes('3') ? '#d29922' : '#58a6ff');
    tr.innerHTML = `<td style="padding:6px">${{range}}</td>`
                 + `<td style="padding:6px;text-align:center;color:${{color}};font-weight:700">${{c2}}</td>`
                 + `<td style="padding:6px;color:var(--muted)">${{phase}}</td>`;
    tbody.appendChild(tr);
  }});

  // ── Phase diagram plot ───────────────────────────────────────────────────
  const canvas = document.getElementById('phasePlot');
  function draw() {{
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth, H = parseInt(canvas.getAttribute('height'));
    canvas.width = W * dpr; canvas.height = H * dpr;
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);

    const pad = {{l:50,r:20,t:20,b:40}};
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
    const sv = DATA.sweep_vals, sc = DATA.sweep_c2;
    if (!sv || sv.length === 0) return;
    const smin = sv[0], smax = sv[sv.length-1];
    const allC2 = sc.map(v => Math.abs(v));
    const maxC2 = Math.max(...allC2, 1);

    function sx(v) {{ return pad.l + (v - smin) / (smax - smin) * pw; }}
    function sy(v) {{ return pad.t + ph - (v / maxC2) * ph; }}

    // Background
    ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
    [-3,-2,-1,0,1,2,3].forEach(v => {{
      if (Math.abs(v) > maxC2) return;
      ctx.beginPath(); ctx.moveTo(pad.l, sy(v)); ctx.lineTo(W-pad.r, sy(v)); ctx.stroke();
    }});

    // Transition markers (m = 0, ±2, ±4)
    ctx.strokeStyle = '#30363d'; ctx.setLineDash([4,4]);
    [-4,-2,0,2,4].forEach(mx => {{
      if (mx < smin || mx > smax) return;
      ctx.beginPath(); ctx.moveTo(sx(mx), pad.t); ctx.lineTo(sx(mx), H-pad.b); ctx.stroke();
    }});
    ctx.setLineDash([]);

    // Filled region
    if (sv.length > 1) {{
      for (let i = 0; i < sv.length - 1; i++) {{
        const v0 = sc[i], v1 = sc[i+1];
        const c = c2Color(v0);
        if (v0 === 0) continue;
        ctx.fillStyle = c + '33';
        ctx.fillRect(sx(sv[i]), pad.t, sx(sv[i+1])-sx(sv[i]), ph);
      }}
    }}

    // Step-function line
    ctx.lineWidth = 2.5;
    for (let i = 0; i < sv.length - 1; i++) {{
      ctx.strokeStyle = c2Color(sc[i]);
      ctx.beginPath();
      ctx.moveTo(sx(sv[i]), sy(sc[i]));
      ctx.lineTo(sx(sv[i+1]), sy(sc[i]));
      ctx.stroke();
    }}
    // Last segment
    if (sv.length > 0) {{
      ctx.strokeStyle = c2Color(sc[sc.length-1]);
      ctx.beginPath();
      ctx.moveTo(sx(sv[sv.length-2]||sv[0]), sy(sc[sc.length-1]));
      ctx.lineTo(sx(sv[sv.length-1]), sy(sc[sc.length-1]));
      ctx.stroke();
    }}

    // Axes
    ctx.strokeStyle = '#484f58'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H-pad.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l, sy(0)); ctx.lineTo(W-pad.r, sy(0)); ctx.stroke();

    // Labels
    ctx.fillStyle = '#8b949e'; ctx.font = '11px monospace';
    ctx.textAlign = 'right';
    [-3,-1,1,3].forEach(v => {{
      if (Math.abs(v) <= maxC2)
        ctx.fillText(v, pad.l-4, sy(v)+4);
    }});
    ctx.textAlign = 'center';
    [smin, (smin+smax)/2, smax].forEach(v => {{
      ctx.fillText(v.toFixed(1), sx(v), H-pad.b+14);
    }});
    [-4,-2,0,2,4].forEach(v => {{
      if (v >= smin && v <= smax)
        ctx.fillText(v.toFixed(0), sx(v), H-pad.b+14);
    }});
    ctx.fillStyle = '#8b949e'; ctx.font = '10px monospace';
    ctx.textAlign = 'left';
    ctx.fillText('C₂', 2, pad.t+12);
    ctx.textAlign = 'center';
    ctx.fillText(DATA.sweep_param || 'm', W/2, H-4);
  }}
  draw();
  window.addEventListener('resize', draw);
}})();
</script>
</body>
</html>"""


def _edge_states_frames(
    M: float = 1.0,
    A: float = 1.0,
    B: float = 1.0,
    C: float = 0.0,
    D: float = 0.0,
    n_kx: int = 60,
    n_y: int = 20,
    scenario: str = "spectrum",
    sweep_param: str = "M",
    param_min: float = -1.0,
    param_max: float = 5.0,
    n_sweep: int = 40,
    source: str = "python-reference",
) -> dict:
    """NumPy reference: BHZ strip edge-state spectrum and Z₂ phase diagram.

    Strip geometry: y-direction finite (n_y sites, open BCs), x periodic.
    On-site block  H_on(kx) and hopping H_hop(-DI + BΓ₅ - iA/2 Γ₂).
    Γ₁=σx⊗σx, Γ₂=σx⊗σy, Γ₅=σz⊗I  (same conventions as Wilson-loop solver).
    """
    import numpy as np  # noqa: PLC0415

    N = n_y            # lattice sites along y
    DIM = 4 * N        # Hilbert-space dimension of strip

    # ── BHZ on-site 4×4 block H_on(kx) ──────────────────────────────────────
    def onsite(kx, M_val):
        # On-site contribution from kx (ky-dependent terms handled by hopping blocks)
        eps = C - 2*D*(2 - np.cos(kx))
        d1  = A * np.sin(kx)
        d5  = M_val - 2*B*(2 - np.cos(kx))
        H = np.zeros((4, 4), dtype=complex)
        H[0, 0] = eps + d5;   H[0, 3] = d1
        H[1, 1] = eps + d5;   H[1, 2] = d1
        H[2, 1] = d1;          H[2, 2] = eps - d5
        H[3, 0] = d1;          H[3, 3] = eps - d5
        return H

    # ── y-hopping 4×4 block (forward, +y direction) ──────────────────────────
    # H_hop = -D I + B Γ₅ - (iA/2) Γ₂
    # Γ₅ = diag(1,1,-1,-1); Γ₂ = σx⊗σy = [[0,0,-i,0],[0,0,0,i],[i,0,0,0],[0,-i,0,0]]
    def hop(_M_val=None):  # noqa: ARG001  (M unused — hop doesn't depend on M)
        # H_hop = -D I + B Γ₅ - (iA/2) Γ₂
        # Γ₂ = σx⊗σy: (0,3)=-i, (1,2)=+i, (2,1)=-i, (3,0)=+i
        # -(iA/2)*(-i)=-A/2; -(iA/2)*(+i)=+A/2
        H = np.zeros((4, 4), dtype=complex)
        H[0, 0] = -D + B;   H[1, 1] = -D + B
        H[2, 2] = -D - B;   H[3, 3] = -D - B
        H[0, 3] = -A / 2
        H[1, 2] = +A / 2
        H[2, 1] = -A / 2
        H[3, 0] = +A / 2
        return H

    # ── Build (4N)×(4N) strip Hamiltonian at kx ───────────────────────────────
    def build_strip(kx, M_val):
        H = np.zeros((DIM, DIM), dtype=complex)
        Hon = onsite(kx, M_val)
        Hop = hop(M_val)
        for i in range(N):
            r0 = 4 * i
            H[r0:r0+4, r0:r0+4] += Hon
            if i + 1 < N:
                r1 = 4 * (i + 1)
                H[r0:r0+4, r1:r1+4] += Hop
                H[r1:r1+4, r0:r0+4] += Hop.conj().T
        return H

    def edge_weight(eigvecs, col):
        """Probability on first/last 2 sites (8 components out of 4N)."""
        v = eigvecs[:, col]
        left  = float(np.sum(np.abs(v[:8])**2))
        right = float(np.sum(np.abs(v[-8:])**2))
        return min(1.0, left + right)

    # ── Wilson loop / WCC on strip (quick Z₂ from bulk) ──────────────────────
    def lower_evecs_bulk(kx, ky, M_val):
        eps = C - 2*D*(2 - np.cos(kx) - np.cos(ky))
        d1  = A * np.sin(kx)
        d2  = A * np.sin(ky)
        d5  = M_val - 2*B*(2 - np.cos(kx) - np.cos(ky))
        H = np.zeros((4, 4), dtype=complex)
        H[0, 0] = eps + d5;  H[0, 3] = d1 - 1j*d2
        H[1, 1] = eps + d5;  H[1, 2] = d1 + 1j*d2
        H[2, 1] = d1 - 1j*d2; H[2, 2] = eps - d5
        H[3, 0] = d1 + 1j*d2; H[3, 3] = eps - d5
        evals, evecs = np.linalg.eigh(H)
        return evecs[:, :2]

    def wcc_at_ky(ky, n_kx_wl, M_val):
        kxs = 2 * np.pi * np.arange(n_kx_wl) / n_kx_wl
        W = np.eye(2, dtype=complex)
        for ix in range(n_kx_wl):
            Q0 = lower_evecs_bulk(kxs[ix], ky, M_val)
            Q1 = lower_evecs_bulk(2*np.pi*((ix+1)%n_kx_wl)/n_kx_wl, ky, M_val)
            W = W @ (Q0.conj().T @ Q1)
        evals = np.linalg.eigvals(W)
        return sorted((-np.angle(ev)/(2*np.pi)) % 1.0 for ev in evals)

    def z2_bulk(M_val, n_kx_wl=20, n_ky_wl=20):
        """Branch-continuity Z₂ from bulk Wilson loop."""
        ky_grid = np.linspace(0, np.pi, n_ky_wl)
        wcc_grid = [wcc_at_ky(ky, n_kx_wl, M_val) for ky in ky_grid]

        def circ_dist(a, b):
            d = abs(a - b)
            return 1.0 - d if d > 0.5 else d

        track = list(wcc_grid[0])
        crossings = 0
        for i in range(1, len(wcc_grid)):
            c0, c1 = wcc_grid[i]
            ds = circ_dist(c0, track[0]) + circ_dist(c1, track[1])
            dc = circ_dist(c1, track[0]) + circ_dist(c0, track[1])
            nxt = [c0, c1] if ds <= dc else [c1, c0]
            for b in range(2):
                if (track[b] - 0.5) * (nxt[b] - 0.5) < 0:
                    crossings += 1
            track = nxt
        return crossings % 2

    # ── Spectrum at nominal M ──────────────────────────────────────────────────
    kx_vals = np.linspace(-np.pi, np.pi, n_kx).tolist()
    all_energies: list = []
    all_weights: list = []
    for kx in kx_vals:
        H = build_strip(kx, M)
        evals, evecs = np.linalg.eigh(H)
        for b in range(DIM):
            all_energies.append(float(evals[b]))
            all_weights.append(edge_weight(evecs, b))

    z2 = z2_bulk(M)
    phase_label = f"Topological (Z\u2082=1)" if z2 == 1 else f"Trivial (Z\u2082=0)"

    # ── Phase diagram sweep ───────────────────────────────────────────────────
    if sweep_param not in ("M",):
        raise ValueError(f"_edge_states_frames: unsupported sweep_param {sweep_param!r} — use 'M'")

    sweep_vals = np.linspace(param_min, param_max, n_sweep).tolist()
    sweep_z2: list = []
    sweep_phase: list = []
    for Mp in sweep_vals:
        z2_s = z2_bulk(Mp, n_kx_wl=16, n_ky_wl=16)
        sweep_z2.append(z2_s)
        sweep_phase.append("Topological" if z2_s == 1 else "Trivial")

    return {
        "M": M, "A": A, "B": B, "C": C, "D": D,
        "n_kx": n_kx, "n_y": n_y,
        "scenario": scenario,
        "z2": z2,
        "phase_label": phase_label,
        "kx_vals": kx_vals,
        "n_bands": DIM,
        "energies": all_energies,
        "edge_weight": all_weights,
        "sweep_param": sweep_param,
        "param_min": param_min,
        "param_max": param_max,
        "sweep_vals": sweep_vals,
        "sweep_z2": sweep_z2,
        "sweep_phase": sweep_phase,
        "source": source,
    }


def _edge_states_page(data: dict, title: str = "Edge States / Surface Spectrum — BHZ Topological Insulator") -> str:
    """HTML page: kx-energy scatter colored by edge weight + Z₂ phase diagram."""
    import json as _json  # noqa: PLC0415
    data_json = _json.dumps(data, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  :root {{
    --bg:#0d1117; --surface:#161b22; --border:#30363d;
    --text:#e6edf3; --muted:#8b949e;
    --accent:#58a6ff; --topo:#3fb950; --trivial:#8b949e;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
        display:flex;flex-direction:column;align-items:center;padding:32px 16px;gap:24px}}
  h1{{font-size:1.3rem;color:var(--accent);text-align:center}}
  .subtitle{{color:var(--muted);font-size:0.9rem;text-align:center}}
  .card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;
         padding:20px;width:100%;max-width:900px}}
  .card h2{{font-size:1rem;color:var(--muted);margin-bottom:14px}}
  .info-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
  .info-box{{background:var(--bg);border:1px solid var(--border);border-radius:6px;
             padding:12px;text-align:center}}
  .info-box .label{{font-size:0.75rem;color:var(--muted);margin-bottom:6px}}
  .info-box .value{{font-size:1.4rem;font-weight:700}}
  .topo{{color:var(--topo)}} .trivial{{color:var(--trivial)}}
  canvas{{display:block;width:100%;cursor:crosshair}}
  .legend{{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;font-size:0.8rem;color:var(--muted)}}
  .cb-row{{display:flex;align-items:stretch;gap:8px}}
  .cb-row canvas:first-child{{flex:1}}
  .cb-col{{display:flex;flex-direction:column;justify-content:space-between;
           font-size:10px;color:var(--muted);padding:20px 0 40px}}
  .footer{{color:var(--muted);font-size:0.75rem;text-align:center}}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">BHZ strip &nbsp;·&nbsp; N_y={data.get("n_y",20)} sites (open BCs) &nbsp;·&nbsp; color = edge-state probability &nbsp;·&nbsp; teal = bulk, red = edge</p>

<div class="card">
  <h2>Parameters &amp; Z₂ Invariant</h2>
  <div class="info-grid" id="infoGrid"></div>
</div>

<div class="card">
  <h2>Band Structure E(kx) — colored by edge probability (bulk=teal → edge=red)</h2>
  <div class="cb-row">
    <canvas id="specPlot" height="400"></canvas>
    <canvas id="colorbar" width="18" height="400" style="flex:none"></canvas>
    <div class="cb-col" id="cbLabels"><span>edge</span><span>0.5</span><span>bulk</span></div>
  </div>
  <div class="legend">
    <span style="color:#f85149">■ Edge state (high boundary probability)</span>
    <span style="color:#39d353">■ Mid-gap bridge</span>
    <span style="color:#58a6ff">■ Bulk band</span>
  </div>
</div>

<div class="card">
  <h2>Z₂ Phase Diagram — Z₂ vs M &nbsp;(topological: 0&lt;M/B&lt;4, gap closes at M=0 and M=4B)</h2>
  <canvas id="phasePlot" height="200"></canvas>
  <div class="legend">
    <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--topo);margin-right:4px"></span>Topological (Z₂=1)</span>
    <span><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--trivial);margin-right:4px"></span>Trivial (Z₂=0)</span>
  </div>
</div>

<p class="footer">Source: {data.get("source","python-reference")} &nbsp;·&nbsp; n_kx={data.get("n_kx",60)} &nbsp;·&nbsp; n_y={data.get("n_y",20)}</p>

<script>
(function(){{
  const DATA = {data_json};

  // ── Edge-weight colormap: bulk (teal) → edge (red) ──────────────────────
  function edgeColor(w) {{
    // w in [0,1]: 0=bulk (teal #58a6ff), 0.5=mid (#39d353), 1=edge (#f85149)
    const t = Math.max(0, Math.min(1, w));
    if (t < 0.5) {{
      const s = t * 2;   // 0→1
      const r = Math.round(88  + (57  - 88)  * s);  // 88→57
      const g = Math.round(166 + (211 - 166) * s);  // 166→211
      const b = Math.round(255 + (83  - 255) * s);  // 255→83
      return `rgb(${{r}},${{g}},${{b}})`;
    }} else {{
      const s = (t - 0.5) * 2;  // 0→1
      const r = Math.round(57  + (248 - 57)  * s);  // 57→248
      const g = Math.round(211 + (81  - 211) * s);  // 211→81
      const b = Math.round(83  + (73  - 83)  * s);  // 83→73
      return `rgb(${{r}},${{g}},${{b}})`;
    }}
  }}

  // ── Info grid ────────────────────────────────────────────────────────────
  const grid = document.getElementById('infoGrid');
  const isTopo = DATA.z2 === 1;
  [
    ['M', DATA.M?.toFixed(2)],
    ['A', DATA.A?.toFixed(2)],
    ['B', DATA.B?.toFixed(2)],
    ['N_y', DATA.n_y],
    ['Z₂', DATA.z2],
    ['Phase', DATA.phase_label],
  ].forEach(([lbl,val])=>{{
    const box=document.createElement('div'); box.className='info-box';
    const hi=(lbl==='Z₂'&&val===1)||(lbl==='Phase'&&String(val).includes('Topo'));
    box.innerHTML=`<div class="label">${{lbl}}</div><div class="value ${{hi?'topo':'trivial'}}">${{val}}</div>`;
    grid.appendChild(box);
  }});

  // ── Colorbar ────────────────────────────────────────────────────────────
  function drawColorbar() {{
    const cb = document.getElementById('colorbar');
    const ctx2 = cb.getContext('2d');
    const h = cb.height;
    for (let y = 0; y < h; y++) {{
      const w = 1 - y / h;   // top=1(edge), bottom=0(bulk)
      ctx2.fillStyle = edgeColor(w);
      ctx2.fillRect(0, y, cb.width, 1);
    }}
  }}
  drawColorbar();

  // ── Spectrum scatter plot ────────────────────────────────────────────────
  const specCanvas = document.getElementById('specPlot');
  function drawSpec() {{
    const dpr = window.devicePixelRatio || 1;
    const W = specCanvas.clientWidth, H = parseInt(specCanvas.getAttribute('height'));
    specCanvas.width = W*dpr; specCanvas.height = H*dpr;
    const ctx = specCanvas.getContext('2d'); ctx.scale(dpr, dpr);
    const pad = {{l:55,r:10,t:20,b:40}};
    const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;

    const kx = DATA.kx_vals;
    const en = DATA.energies;
    const ew = DATA.edge_weight;
    const nb = DATA.n_bands;
    if (!kx || kx.length === 0 || !en) return;

    const enMin = Math.min(...en), enMax = Math.max(...en);
    const kxMin = kx[0], kxMax = kx[kx.length-1];

    function sx(v) {{ return pad.l + (v - kxMin) / (kxMax - kxMin) * pw; }}
    function sy(v) {{ return pad.t + (1 - (v - enMin) / (enMax - enMin)) * ph; }}

    ctx.fillStyle = '#0d1117'; ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
    for (let k = 0; k <= 4; k++) {{
      const v = enMin + k * (enMax - enMin) / 4;
      ctx.beginPath(); ctx.moveTo(pad.l, sy(v)); ctx.lineTo(W-pad.r, sy(v)); ctx.stroke();
    }}
    // E=0 line
    if (enMin < 0 && enMax > 0) {{
      ctx.strokeStyle = '#484f58'; ctx.setLineDash([5,4]);
      ctx.beginPath(); ctx.moveTo(pad.l, sy(0)); ctx.lineTo(W-pad.r, sy(0)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.strokeStyle = '#21262d';
    }}

    // Scatter: one dot per (kx_i, band_b)
    const nk = kx.length;
    for (let i = 0; i < nk; i++) {{
      for (let b = 0; b < nb; b++) {{
        const idx = i * nb + b;
        ctx.fillStyle = edgeColor(ew[idx]);
        ctx.fillRect(sx(kx[i])-0.8, sy(en[idx])-0.8, 1.6, 1.6);
      }}
    }}

    // Axes
    ctx.strokeStyle = '#484f58'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l,H-pad.b); ctx.lineTo(W-pad.r,H-pad.b); ctx.stroke();

    ctx.fillStyle = '#8b949e'; ctx.font = '11px monospace'; ctx.textAlign = 'right';
    for (let k = 0; k <= 4; k++) {{
      const v = enMin + k*(enMax-enMin)/4;
      ctx.fillText(v.toFixed(1), pad.l-4, sy(v)+4);
    }}
    ctx.textAlign = 'center';
    [-Math.PI, 0, Math.PI].forEach(v => {{
      ctx.fillText(v===0?'0':v>0?'π':'−π', sx(v), H-pad.b+14);
    }});
    ctx.fillStyle = '#8b949e'; ctx.font = '10px monospace';
    ctx.textAlign = 'left'; ctx.fillText('E', 4, pad.t+12);
    ctx.textAlign = 'center'; ctx.fillText('kx', W/2, H-4);
  }}
  drawSpec();

  // ── Phase diagram ────────────────────────────────────────────────────────
  const phCanvas = document.getElementById('phasePlot');
  function drawPhase() {{
    const dpr = window.devicePixelRatio||1;
    const W = phCanvas.clientWidth, H = parseInt(phCanvas.getAttribute('height'));
    phCanvas.width = W*dpr; phCanvas.height = H*dpr;
    const ctx = phCanvas.getContext('2d'); ctx.scale(dpr,dpr);
    const pad = {{l:50,r:20,t:20,b:40}};
    const pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;
    const sv = DATA.sweep_vals, sz = DATA.sweep_z2;
    if (!sv||sv.length===0) return;
    const smin=sv[0], smax=sv[sv.length-1];
    function sx(v) {{ return pad.l+(v-smin)/(smax-smin)*pw; }}
    function sy(z) {{ return pad.t+(1-z)*ph; }}

    ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);

    // Filled regions
    for (let i=0;i<sv.length-1;i++) {{
      ctx.fillStyle = sz[i]===1?'#3fb95033':'#8b949e22';
      ctx.fillRect(sx(sv[i]),pad.t,sx(sv[i+1])-sx(sv[i]),ph);
    }}

    // Step-function Z₂ line
    ctx.lineWidth=2.5;
    for (let i=0;i<sv.length-1;i++) {{
      ctx.strokeStyle=sz[i]===1?'#3fb950':'#8b949e';
      ctx.beginPath(); ctx.moveTo(sx(sv[i]),sy(sz[i])); ctx.lineTo(sx(sv[i+1]),sy(sz[i])); ctx.stroke();
    }}

    // Transition markers M=0 and M=4B
    ctx.strokeStyle='#484f58'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
    [0, 4*DATA.B].forEach(mx=>{{
      if (mx>=smin&&mx<=smax) {{
        ctx.beginPath(); ctx.moveTo(sx(mx),pad.t); ctx.lineTo(sx(mx),H-pad.b); ctx.stroke();
      }}
    }});
    ctx.setLineDash([]);

    ctx.strokeStyle='#484f58'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l,sy(0)); ctx.lineTo(W-pad.r,sy(0)); ctx.stroke();

    ctx.fillStyle='#8b949e'; ctx.font='11px monospace'; ctx.textAlign='right';
    ctx.fillText('0',pad.l-4,sy(0)+4); ctx.fillText('1',pad.l-4,sy(1)+4);
    ctx.textAlign='center';
    [smin,(smin+smax)/2,smax].forEach(v=>{{ ctx.fillText(v.toFixed(1),sx(v),H-pad.b+14); }});
    [0,4*DATA.B].forEach(v=>{{ if(v>=smin&&v<=smax) ctx.fillText(v===0?'0':`4B=${{(4*DATA.B).toFixed(1)}}`,sx(v),H-pad.b+14); }});
    ctx.fillStyle='#8b949e'; ctx.font='10px monospace';
    ctx.textAlign='left'; ctx.fillText('Z₂',2,pad.t+12);
    ctx.textAlign='center'; ctx.fillText('M',W/2,H-4);
  }}
  drawPhase();
  window.addEventListener('resize',()=>{{drawSpec();drawPhase();}});
}})();
</script>
</body>
</html>"""


def _haldane_frames(
    t1: float = 1.0,
    t2: float = 0.1,
    phi: float = 1.5707963,
    M_val: float = 0.0,
    n_k: int = 40,
    scenario: str = "phase_diagram",
    sweep_param: str = "M",
    param_min: float = -6.0,
    param_max: float = 6.0,
    n_sweep: int = 60,
    source: str = "python-reference",
) -> dict:
    """NumPy reference: Haldane Model / Quantum Anomalous Hall Effect.

    Computes the Chern number C1 via the Fukui-Hatsugai-Suzuki lattice gauge
    formula, Berry curvature map over the BZ, and a phase diagram sweep.
    """
    import numpy as np  # noqa: PLC0415

    sqrt3 = np.sqrt(3.0)

    # NN vectors A->B
    delta = np.array([
        [0.0,       1.0],
        [sqrt3/2,  -0.5],
        [-sqrt3/2, -0.5],
    ])
    # NNN vectors (A->A CCW)
    b_nnn = np.array([
        [sqrt3,    0.0],
        [sqrt3/2,  1.5],
        [-sqrt3/2, 1.5],
    ])
    # Reciprocal lattice vectors
    G1 = np.array([2*np.pi/sqrt3, -2*np.pi/3])
    G2 = np.array([0.0,            4*np.pi/3])

    def _haldane_ham(kx, ky, _t1, _t2, _phi, _M):
        k = np.array([kx, ky])
        f = sum(np.exp(1j * float(k @ d)) for d in delta)
        g_cos = sum(float(np.cos(k @ b)) for b in b_nnn)
        g_sin = sum(float(np.sin(k @ b)) for b in b_nnn)
        h0 = 2 * _t2 * np.cos(_phi) * g_cos
        hz = _M - 2 * _t2 * np.sin(_phi) * g_sin
        hx = _t1 * f.real
        hy = _t1 * f.imag
        H = np.array([[h0 + hz, hx - 1j*hy],
                      [hx + 1j*hy, h0 - hz]], dtype=complex)
        return H

    def _lower_evec(kx, ky, _t1, _t2, _phi, _M):
        H = _haldane_ham(kx, ky, _t1, _t2, _phi, _M)
        _, vecs = np.linalg.eigh(H)
        return vecs[:, 0]  # eigh sorts ascending -> index 0 is lower

    def _link(v1, v2):
        z = np.vdot(v1, v2)
        a = abs(z)
        return (z / a) if a > 1e-12 else complex(1.0)

    def _compute_chern(_t1, _t2, _phi, _M, _nk):
        kpts = np.zeros((_nk, _nk, 2))
        vecs = np.zeros((_nk, _nk, 2), dtype=complex)
        for i in range(_nk):
            for j in range(_nk):
                kpts[i, j] = (i/_nk)*G1 + (j/_nk)*G2
                vecs[i, j] = _lower_evec(kpts[i, j, 0], kpts[i, j, 1],
                                         _t1, _t2, _phi, _M)

        bc = np.zeros(_nk * _nk)
        kx_out = []
        ky_out = []
        F_sum = 0.0
        for i in range(_nk):
            for j in range(_nk):
                ip = (i + 1) % _nk
                jp = (j + 1) % _nk
                U1  = _link(vecs[i, j],  vecs[ip, j])
                U2  = _link(vecs[i, j],  vecs[i,  jp])
                U1p = _link(vecs[i, jp], vecs[ip, jp])
                U2p = _link(vecs[ip, j], vecs[ip, jp])
                F = np.angle(U1 * U2p * np.conj(U1p) * np.conj(U2))
                bc[i*_nk + j] = float(F)
                kx_out.append(float(kpts[i, j, 0]))
                ky_out.append(float(kpts[i, j, 1]))
                F_sum += float(F)
        C1 = int(round(F_sum / (2*np.pi)))
        return C1, kx_out, ky_out, bc.tolist()

    def _phase_label_h(C1):
        if C1 > 0:
            return f"QAH (C\u2081=+{C1})"
        elif C1 < 0:
            return f"QAH (C\u2081={C1})"
        return "Trivial (C\u2081=0)"

    # Compute Chern + Berry curvature at reference parameters
    C1, kx_bz, ky_bz, berry_curv = _compute_chern(t1, t2, phi, M_val, n_k)

    # Phase diagram sweep
    sweep_vals = []
    sweep_chern = []
    sweep_phase = []
    if n_sweep >= 2:
        for s in range(n_sweep):
            v = param_min + s * (param_max - param_min) / (n_sweep - 1)
            sweep_vals.append(float(v))
            if sweep_param == "M":
                C1s, _, _, _ = _compute_chern(t1, t2, phi, v, n_k)
            else:
                C1s, _, _, _ = _compute_chern(t1, t2, v, M_val, n_k)
            sweep_chern.append(C1s)
            sweep_phase.append(_phase_label_h(C1s))

    return {
        "t1": t1,
        "t2": t2,
        "phi": phi,
        "M": M_val,
        "n_k": n_k,
        "scenario": scenario,
        "sweep_param": sweep_param,
        "chern": C1,
        "phase_label": _phase_label_h(C1),
        "kx_bz": kx_bz,
        "ky_bz": ky_bz,
        "berry_curv": berry_curv,
        "sweep_vals": sweep_vals,
        "sweep_chern": sweep_chern,
        "sweep_phase": sweep_phase,
        "source": source,
    }


def _haldane_page(data: dict, title: str = "Haldane Model \u2014 Quantum Anomalous Hall Effect") -> str:
    """Render HTML page: info grid + Berry curvature BZ scatter + phase diagram."""
    import json as _json  # noqa: PLC0415

    data_js = _json.dumps(data)
    sweep_param_label = data.get("sweep_param", "M")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#c9d1d9;font-family:'Segoe UI',monospace,sans-serif;padding:16px}}
h1{{font-size:1.15rem;color:#58a6ff;margin-bottom:12px}}
.info-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px;margin-bottom:14px}}
.info-card{{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:8px 10px}}
.info-card .lbl{{font-size:0.68rem;color:#8b949e;margin-bottom:2px}}
.info-card .val{{font-size:1.05rem;color:#e6edf3;font-weight:600}}
.plots{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px}}
canvas{{width:100%;background:#161b22;border:1px solid #30363d;border-radius:6px;display:block}}
@media(max-width:700px){{.plots{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="info-grid" id="infoGrid"></div>
<div class="plots">
  <canvas id="bzPlot" height="340"></canvas>
  <canvas id="phasePlot" height="340"></canvas>
</div>
<script>
(function(){{
  const DATA = {data_js};

  // ── Info grid ──────────────────────────────────────────────────────────
  const grid = document.getElementById('infoGrid');
  const fields = [
    ['t\u2081',  DATA.t1.toFixed(3)],
    ['t\u2082',  DATA.t2.toFixed(4)],
    ['\u03c6',   DATA.phi.toFixed(4)],
    ['M',        DATA.M.toFixed(3)],
    ['C\u2081',  String(DATA.chern)],
    ['Phase',    DATA.phase_label],
    ['n_k',      String(DATA.n_k)],
    ['Source',   DATA.source || 'python-reference'],
  ];
  fields.forEach(([l,v])=>{{
    const d=document.createElement('div'); d.className='info-card';
    d.innerHTML='<div class="lbl">'+l+'</div><div class="val">'+v+'</div>';
    grid.appendChild(d);
  }});

  // ── Berry curvature BZ scatter ─────────────────────────────────────────
  const bzCanvas = document.getElementById('bzPlot');
  function drawBZ(){{
    const dpr = window.devicePixelRatio||1;
    const W = bzCanvas.clientWidth;
    const H = parseInt(bzCanvas.getAttribute('height'));
    bzCanvas.width  = W*dpr;
    bzCanvas.height = H*dpr;
    const ctx = bzCanvas.getContext('2d');
    ctx.scale(dpr,dpr);
    const pad={{l:50,r:24,t:32,b:40}};
    const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;

    ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);

    const kx=DATA.kx_bz, ky=DATA.ky_bz, bc=DATA.berry_curv;
    if(!kx||kx.length===0) return;

    const kxMin=Math.min(...kx), kxMax=Math.max(...kx);
    const kyMin=Math.min(...ky), kyMax=Math.max(...ky);
    let bcMax=0; bc.forEach(v=>{{if(Math.abs(v)>bcMax) bcMax=Math.abs(v);}});
    if(bcMax<1e-14) bcMax=1.0;

    function ssx(v){{return pad.l+(v-kxMin)/(kxMax-kxMin+1e-12)*pw;}}
    function ssy(v){{return pad.t+(1-(v-kyMin)/(kyMax-kyMin+1e-12))*ph;}}

    function bcColor(v){{
      // Diverging red–white–blue: positive→red, zero→white, negative→blue
      const t=Math.max(-1,Math.min(1,v/bcMax));
      if(t>=0){{
        const c=Math.round(255*(1-t));
        return 'rgb(255,'+c+','+c+')';
      }}
      const c=Math.round(255*(1+t));
      return 'rgb('+c+','+c+',255)';
    }}

    const dotR = Math.max(1.5, pw/DATA.n_k*0.45);
    for(let i=0;i<kx.length;i++){{
      ctx.fillStyle=bcColor(bc[i]);
      ctx.beginPath();
      ctx.arc(ssx(kx[i]),ssy(ky[i]),dotR,0,2*Math.PI);
      ctx.fill();
    }}

    ctx.strokeStyle='#484f58'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l,H-pad.b); ctx.lineTo(W-pad.r,H-pad.b); ctx.stroke();

    ctx.fillStyle='#8b949e'; ctx.font='10px monospace'; ctx.textAlign='center';
    ctx.fillText('kx', W/2, H-4);
    ctx.save(); ctx.translate(12,H/2); ctx.rotate(-Math.PI/2);
    ctx.fillText('ky',0,0); ctx.restore();

    ctx.fillStyle='#c9d1d9'; ctx.font='11px monospace'; ctx.textAlign='center';
    ctx.fillText('Berry Curvature \u03a9(k) over BZ', W/2, pad.t-10);

    // Color legend bar
    const lx=W-pad.r-58, ly=pad.t+4, lw=50, lh=10;
    const grad=ctx.createLinearGradient(lx,ly,lx+lw,ly);
    grad.addColorStop(0,'rgb(0,0,255)');
    grad.addColorStop(0.5,'rgb(0,0,0)');
    grad.addColorStop(1,'rgb(255,0,0)');
    ctx.fillStyle=grad; ctx.fillRect(lx,ly,lw,lh);
    ctx.strokeStyle='#484f58'; ctx.strokeRect(lx,ly,lw,lh);
    ctx.fillStyle='#8b949e'; ctx.font='9px monospace';
    ctx.textAlign='right'; ctx.fillText('-',lx-2,ly+lh);
    ctx.textAlign='left';  ctx.fillText('+',lx+lw+2,ly+lh);
  }}
  drawBZ();

  // ── Phase diagram ──────────────────────────────────────────────────────
  const phCanvas = document.getElementById('phasePlot');
  function drawPhase(){{
    const dpr = window.devicePixelRatio||1;
    const W = phCanvas.clientWidth;
    const H = parseInt(phCanvas.getAttribute('height'));
    phCanvas.width  = W*dpr;
    phCanvas.height = H*dpr;
    const ctx = phCanvas.getContext('2d');
    ctx.scale(dpr,dpr);
    const pad={{l:56,r:20,t:32,b:40}};
    const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;

    ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);

    const sv=DATA.sweep_vals, sc=DATA.sweep_chern;
    if(!sv||sv.length===0) return;

    const smin=sv[0], smax=sv[sv.length-1];
    let cmin=0,cmax=0;
    sc.forEach(v=>{{if(v<cmin)cmin=v;if(v>cmax)cmax=v;}});
    const ymin=cmin-0.6, ymax=cmax+0.6;

    function ssx(v){{return pad.l+(v-smin)/(smax-smin+1e-12)*pw;}}
    function ssy(v){{return pad.t+(1-(v-ymin)/(ymax-ymin+1e-12))*ph;}}

    for(let i=0;i<sv.length-1;i++){{
      let col='#8b949e22';
      if(sc[i]>0) col='#3fb95033';
      else if(sc[i]<0) col='#da363333';
      ctx.fillStyle=col;
      ctx.fillRect(ssx(sv[i]),pad.t,ssx(sv[i+1])-ssx(sv[i]),ph);
    }}

    ctx.lineWidth=2.5;
    for(let i=0;i<sv.length-1;i++){{
      let col='#8b949e';
      if(sc[i]>0) col='#3fb950';
      else if(sc[i]<0) col='#f85149';
      ctx.strokeStyle=col;
      ctx.beginPath(); ctx.moveTo(ssx(sv[i]),ssy(sc[i])); ctx.lineTo(ssx(sv[i+1]),ssy(sc[i])); ctx.stroke();
    }}

    if(ymin<0&&ymax>0){{
      ctx.strokeStyle='#484f58'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
      ctx.beginPath(); ctx.moveTo(pad.l,ssy(0)); ctx.lineTo(W-pad.r,ssy(0)); ctx.stroke();
      ctx.setLineDash([]);
    }}

    ctx.strokeStyle='#484f58'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(pad.l,H-pad.b); ctx.lineTo(W-pad.r,H-pad.b); ctx.stroke();

    ctx.fillStyle='#8b949e'; ctx.font='11px monospace'; ctx.textAlign='right';
    const yTicks=[...new Set([cmin,0,cmax])];
    yTicks.forEach(v=>{{ ctx.fillText(String(v),pad.l-4,ssy(v)+4); }});
    ctx.textAlign='center';
    [smin,(smin+smax)/2,smax].forEach(v=>{{ ctx.fillText(v.toFixed(2),ssx(v),H-pad.b+14); }});
    ctx.fillStyle='#8b949e'; ctx.font='10px monospace';
    ctx.textAlign='left'; ctx.fillText('C\u2081',2,pad.t+12);
    ctx.textAlign='center'; ctx.fillText('{sweep_param_label}',W/2,H-4);
    ctx.fillStyle='#c9d1d9'; ctx.font='11px monospace';
    ctx.fillText('Phase Diagram: C\u2081 vs {sweep_param_label}',W/2,pad.t-10);
  }}
  drawPhase();

  window.addEventListener('resize',()=>{{drawBZ();drawPhase();}});
}})();
</script>
</body>
</html>"""


def _weyl_frames(
    m_param: float = 2.0,
    n_k: int = 20,
    n_z: int = 20,
    n_kz: int = 40,
    fermi_energy: float = 0.0,
    scenario: str = "fermi_arc",
    param_min: float = 1.1,
    param_max: float = 2.9,
    n_sweep: int = 30,
    source: str = "python-reference",
) -> dict:
    """NumPy reference: Weyl semimetal Hamiltonian, Chern numbers, and surface Fermi arcs."""
    import numpy as np  # noqa: PLC0415

    def weyl_ham(kx, ky, kz):
        dx = np.sin(kx)
        dy = np.sin(ky)
        dz = m_param - np.cos(kx) - np.cos(ky) - np.cos(kz)
        return np.array([[dz, dx - 1j * dy], [dx + 1j * dy, -dz]])

    def lower_evec(kx, ky, kz):
        H = weyl_ham(kx, ky, kz)
        evals, evecs = np.linalg.eigh(H)
        return evecs[:, 0]

    # Chern number on kx-ky slice at fixed kz (FHS lattice gauge)
    def chern_slice(kz):
        kpts = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
        vecs = [[lower_evec(kx, ky, kz) for ky in kpts] for kx in kpts]

        def link(v1, v2):
            z = np.vdot(v1, v2)
            return z / abs(z) if abs(z) > 1e-12 else 1.0

        F_sum = 0.0
        for i in range(n_k):
            for j in range(n_k):
                ip = (i + 1) % n_k
                jp = (j + 1) % n_k
                U1  = link(vecs[i][j],  vecs[ip][j])
                U2  = link(vecs[i][j],  vecs[i][jp])
                U1p = link(vecs[i][jp], vecs[ip][jp])
                U2p = link(vecs[ip][j], vecs[ip][jp])
                F_sum += np.angle(U1 * U2p * np.conj(U1p) * np.conj(U2))
        return int(round(F_sum / (2 * np.pi)))

    # Weyl node positions
    val = m_param - 2.0
    if abs(val) < 1.0:
        node_kz = float(np.arccos(val))
    else:
        node_kz = float("nan")

    # Bulk Weyl cone: E+(0,0,kz)
    kz_line = np.linspace(-np.pi, np.pi, 80).tolist()
    cone_E_plus = []
    cone_E_minus = []
    for kz in kz_line:
        dz = m_param - 1.0 - 1.0 - np.cos(kz)  # kx=ky=0 => cos(0)=1 each
        e = float(np.sqrt(dz ** 2))
        cone_E_plus.append(e)
        cone_E_minus.append(-e)

    # Chern sweep over kz
    kz_vals = np.linspace(-np.pi, np.pi, n_kz).tolist()
    chern_kz = [chern_slice(kz) for kz in kz_vals]

    # Surface Fermi arc: slab spectrum
    kx_arc = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
    ky_arc = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
    arc_kx: list = []
    arc_ky: list = []
    arc_weight: list = []
    arc_energy: list = []

    def build_slab(kx, ky):
        H = np.zeros((2 * n_z, 2 * n_z), dtype=complex)
        dx = np.sin(kx)
        dy = np.sin(ky)
        dz_on = m_param - np.cos(kx) - np.cos(ky)
        H_on  = np.array([[dz_on, dx - 1j * dy], [dx + 1j * dy, -dz_on]])
        H_hop = np.array([[-0.5, 0.0], [0.0, 0.5]])  # -(1/2)*sigma_z
        for iz in range(n_z):
            H[2 * iz: 2 * iz + 2, 2 * iz: 2 * iz + 2] = H_on
            if iz + 1 < n_z:
                H[2 * iz: 2 * iz + 2,     2 * (iz + 1): 2 * (iz + 1) + 2] = H_hop
                H[2 * (iz + 1): 2 * (iz + 1) + 2, 2 * iz: 2 * iz + 2]    = H_hop.conj().T
        return H

    for kx in kx_arc:
        for ky in ky_arc:
            H_slab = build_slab(kx, ky)
            evals, evecs = np.linalg.eigh(H_slab)
            idx = int(np.argmin(np.abs(evals - fermi_energy)))
            v = evecs[:, idx]
            w = float(np.sum(np.abs(v[:2]) ** 2) + np.sum(np.abs(v[-2:]) ** 2))
            arc_kx.append(float(kx))
            arc_ky.append(float(ky))
            arc_weight.append(min(w, 1.0))
            arc_energy.append(float(evals[idx]))

    # m sweep: node kz position vs m
    sweep_vals_list = np.linspace(param_min, param_max, n_sweep).tolist()
    sweep_node_kz: list = []
    sweep_phase: list = []
    for m_s in sweep_vals_list:
        vs = m_s - 2.0
        if abs(vs) < 1.0:
            sweep_node_kz.append(float(np.arccos(vs)))
            sweep_phase.append("Weyl semimetal")
        else:
            sweep_node_kz.append(float("nan"))
            sweep_phase.append("Trivial/Insulator")

    return {
        "m":             m_param,
        "n_k":           n_k,
        "n_z":           n_z,
        "n_kz":          n_kz,
        "scenario":      scenario,
        "node_kz_pos":   node_kz if not (node_kz != node_kz) else None,
        "node_kz_neg":   -node_kz if not (node_kz != node_kz) else None,
        "kz_vals":       kz_vals,
        "chern_kz":      chern_kz,
        "arc_kx":        arc_kx,
        "arc_ky":        arc_ky,
        "arc_weight":    arc_weight,
        "arc_energy":    arc_energy,
        "n_arc_pts":     n_k * n_k,
        "cone_kz":       kz_line,
        "cone_E_plus":   cone_E_plus,
        "cone_E_minus":  cone_E_minus,
        "sweep_vals":    sweep_vals_list,
        "sweep_node_kz": sweep_node_kz,
        "sweep_phase":   sweep_phase,
        "source":        source,
    }


def _weyl_page(data: dict, title: str = "Weyl Semimetal & Surface Fermi Arcs") -> str:
    """HTML page: Surface Fermi arc + Chern vs kz + bulk cone + node diagram."""
    import json as _json  # noqa: PLC0415
    data_json = _json.dumps(data, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
  :root {{
    --bg:#0d1117; --surface:#161b22; --border:#30363d;
    --text:#e6edf3; --muted:#8b949e;
    --accent:#58a6ff; --topo:#3fb950; --trivial:#8b949e;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;
        display:flex;flex-direction:column;align-items:center;padding:32px 16px;gap:24px}}
  h1{{font-size:1.3rem;color:var(--accent);text-align:center}}
  .subtitle{{color:var(--muted);font-size:0.9rem;text-align:center}}
  .card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;
         padding:20px;width:100%;max-width:900px}}
  .card h2{{font-size:1rem;color:var(--muted);margin-bottom:14px}}
  .info-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}}
  .info-box{{background:var(--bg);border:1px solid var(--border);border-radius:6px;
             padding:12px;text-align:center}}
  .info-box .label{{font-size:0.75rem;color:var(--muted);margin-bottom:6px}}
  .info-box .value{{font-size:1.4rem;font-weight:700}}
  .weyl{{color:#f78166}} .trivial{{color:var(--trivial)}}
  canvas{{display:block;width:100%;cursor:crosshair}}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">H(k) = sin(k&#x2093;)&#x3C3;&#x2093; + sin(k&#x1D67;)&#x3C3;&#x1D67; + (m&minus;cos k&#x2093;&minus;cos k&#x1D67;&minus;cos k&#x1D69;)&#x3C3;&#x1D69; &nbsp;|&nbsp; Weyl nodes at k&#x1D69;=&plusmn;arccos(m&minus;2)</p>

<div class="card">
<h2>Parameters</h2>
<div class="info-grid" id="infoGrid"></div>
</div>

<div class="card">
<h2>Surface Fermi Arc (E&#x2248;0 edge weight, k&#x2093;&ndash;k&#x1D67; slab BZ)</h2>
<canvas id="arcPlot" height="420"></canvas>
</div>

<div class="card">
<h2>Chern Number C&#x2081;(k&#x1D69;) on k&#x2093;&ndash;k&#x1D67; Slices</h2>
<canvas id="chernPlot" height="260"></canvas>
</div>

<div class="card">
<h2>Bulk Weyl Cone E(0,0,k&#x1D69;)</h2>
<canvas id="conePlot" height="260"></canvas>
</div>

<div class="card">
<h2>Weyl Node k&#x1D69; Position vs m (Node Diagram)</h2>
<canvas id="nodePlot" height="260"></canvas>
</div>

<script>
(function(){{
const DATA = {data_json};

// ── Info grid ─────────────────────────────────────────────────────────────
const ig = document.getElementById('infoGrid');
function ib(label,value,cls='') {{
  return `<div class="info-box"><div class="label">${{label}}</div><div class="value ${{cls}}">${{value}}</div></div>`;
}}
const hasNodes = DATA.node_kz_pos !== null && !isNaN(DATA.node_kz_pos);
const phaseLabel = hasNodes ? 'Weyl' : 'Trivial';
const phaseCls   = hasNodes ? 'weyl' : 'trivial';
ig.innerHTML =
  ib('m',DATA.m.toFixed(3)) +
  ib('n_k',DATA.n_k) +
  ib('n_z',DATA.n_z) +
  ib('Node k\u2080 +',hasNodes?DATA.node_kz_pos.toFixed(3):'—') +
  ib('Node k\u2080 −',hasNodes?DATA.node_kz_neg.toFixed(3):'—') +
  ib('Phase',phaseLabel,phaseCls);

// ── Colour scale: teal (bulk) → red (edge) ─────────────────────────────
function weightColour(w) {{
  const r = Math.round(w*220+(1-w)*30);
  const g = Math.round((1-w)*180+w*50);
  const b = Math.round((1-w)*200+w*50);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

// ── Surface Fermi arc plot ────────────────────────────────────────────────
const arcCanvas = document.getElementById('arcPlot');
function drawArc() {{
  const dpr = window.devicePixelRatio||1;
  const W = arcCanvas.clientWidth;
  const H = parseInt(arcCanvas.getAttribute('height'));
  arcCanvas.width  = W*dpr;
  arcCanvas.height = H*dpr;
  const ctx = arcCanvas.getContext('2d'); ctx.scale(dpr,dpr);
  const pad = {{l:50,r:20,t:20,b:40}};
  const pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;
  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);

  const kmin=-Math.PI, kmax=Math.PI;
  function sx(v){{ return pad.l+(v-kmin)/(kmax-kmin)*pw; }}
  function sy(v){{ return pad.t+(kmax-v)/(kmax-kmin)*ph; }}

  const kx=DATA.arc_kx, ky=DATA.arc_ky, wt=DATA.arc_weight;
  if(!kx||kx.length===0) return;
  const nk = DATA.n_k;
  const sz = Math.max(2, Math.floor(pw/nk)-1);
  for(let i=0;i<kx.length;i++) {{
    ctx.fillStyle = weightColour(wt[i]);
    ctx.fillRect(sx(kx[i])-sz/2, sy(ky[i])-sz/2, sz, sz);
  }}

  // Mark projected Weyl nodes (both project to kx=ky=0)
  if(hasNodes) {{
    ctx.strokeStyle='#f78166'; ctx.lineWidth=2;
    const px=sx(0), py=sy(0);
    ctx.beginPath(); ctx.arc(px,py,6,0,2*Math.PI); ctx.stroke();
  }}

  // Axes
  ctx.strokeStyle='#484f58'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.l,H-pad.b); ctx.lineTo(W-pad.r,H-pad.b); ctx.stroke();
  ctx.fillStyle='#8b949e'; ctx.font='11px monospace';
  ctx.textAlign='center';
  ['-π','0','π'].forEach((lbl,ii)=>{{
    const v=[-Math.PI,0,Math.PI][ii];
    ctx.fillText(lbl,sx(v),H-pad.b+14);
  }});
  ctx.textAlign='right';
  ['-π','0','π'].forEach((lbl,ii)=>{{
    const v=[-Math.PI,0,Math.PI][ii];
    ctx.fillText(lbl,pad.l-4,sy(v)+4);
  }});
  ctx.textAlign='center'; ctx.fillText('kₓ',W/2,H-4);
  ctx.save(); ctx.translate(12,H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('k\u1d67',0,0); ctx.restore();

  // Colour legend
  const lgx=W-pad.r-120, lgy=pad.t+10;
  const grad=ctx.createLinearGradient(lgx,0,lgx+100,0);
  grad.addColorStop(0,weightColour(0));
  grad.addColorStop(1,weightColour(1));
  ctx.fillStyle=grad; ctx.fillRect(lgx,lgy,100,10);
  ctx.fillStyle='#8b949e'; ctx.font='9px monospace'; ctx.textAlign='left';
  ctx.fillText('bulk',lgx,lgy+22); ctx.textAlign='right';
  ctx.fillText('edge',lgx+100,lgy+22);
}}
drawArc();

// ── Chern number vs kz ────────────────────────────────────────────────────
const chCanvas = document.getElementById('chernPlot');
function drawChern() {{
  const dpr = window.devicePixelRatio||1;
  const W = chCanvas.clientWidth, H = parseInt(chCanvas.getAttribute('height'));
  chCanvas.width = W*dpr; chCanvas.height = H*dpr;
  const ctx = chCanvas.getContext('2d'); ctx.scale(dpr,dpr);
  const pad = {{l:50,r:20,t:20,b:40}};
  const pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;
  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);

  const kz=DATA.kz_vals, ch=DATA.chern_kz;
  if(!kz||kz.length===0) return;
  const kmin=kz[0], kmax=kz[kz.length-1];
  function sx(v){{ return pad.l+(v-kmin)/(kmax-kmin)*pw; }}
  function sy(c){{ return pad.t+(1-c)*ph/2; }}

  function chernColor(c, alpha) {{
    if(c===1)  return alpha?'#3fb95033':'#3fb950';
    if(c===-1) return alpha?'#58a6ff33':'#58a6ff';
    return alpha?'#8b949e22':'#8b949e';
  }}

  // Filled regions
  for(let i=0;i<kz.length-1;i++) {{
    ctx.fillStyle = chernColor(ch[i], true);
    ctx.fillRect(sx(kz[i]),pad.t,sx(kz[i+1])-sx(kz[i]),ph);
  }}

  // Step function
  ctx.lineWidth=2.5;
  for(let i=0;i<kz.length-1;i++) {{
    ctx.strokeStyle = chernColor(ch[i], false);
    ctx.beginPath(); ctx.moveTo(sx(kz[i]),sy(ch[i])); ctx.lineTo(sx(kz[i+1]),sy(ch[i])); ctx.stroke();
  }}

  // Vertical lines at Weyl node positions
  if(hasNodes) {{
    ctx.strokeStyle='#f78166'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
    [DATA.node_kz_pos, DATA.node_kz_neg].forEach(kzn=>{{
      if(kzn>=kmin&&kzn<=kmax) {{
        ctx.beginPath(); ctx.moveTo(sx(kzn),pad.t); ctx.lineTo(sx(kzn),H-pad.b); ctx.stroke();
      }}
    }});
    ctx.setLineDash([]);
  }}

  ctx.strokeStyle='#484f58'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.l,sy(0)); ctx.lineTo(W-pad.r,sy(0)); ctx.stroke();
  ctx.fillStyle='#8b949e'; ctx.font='11px monospace'; ctx.textAlign='right';
  ctx.fillText('0',pad.l-4,sy(0)+4); ctx.fillText('1',pad.l-4,sy(1)+4); ctx.fillText('-1',pad.l-4,sy(-1)+4);
  ctx.textAlign='center';
  ['-π','0','π'].forEach((lbl,ii)=>{{ const v=[-Math.PI,0,Math.PI][ii]; ctx.fillText(lbl,sx(v),H-pad.b+14); }});
  ctx.fillStyle='#8b949e'; ctx.font='10px monospace';
  ctx.textAlign='left'; ctx.fillText('C\u2081',2,pad.t+12);
  ctx.textAlign='center'; ctx.fillText('k\u1d69',W/2,H-4);
}}
drawChern();

// ── Bulk Weyl cone ────────────────────────────────────────────────────────
const coneCanvas = document.getElementById('conePlot');
function drawCone() {{
  const dpr = window.devicePixelRatio||1;
  const W = coneCanvas.clientWidth, H = parseInt(coneCanvas.getAttribute('height'));
  coneCanvas.width = W*dpr; coneCanvas.height = H*dpr;
  const ctx = coneCanvas.getContext('2d'); ctx.scale(dpr,dpr);
  const pad = {{l:50,r:20,t:20,b:40}};
  const pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;
  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);

  const kz=DATA.cone_kz, ep=DATA.cone_E_plus, em=DATA.cone_E_minus;
  if(!kz||kz.length===0) return;
  const kmin=kz[0], kmax=kz[kz.length-1];
  const allE=ep.concat(em);
  const emin=Math.min(...allE), emax=Math.max(...allE);
  const espan=emax-emin||1;
  function sx(v){{ return pad.l+(v-kmin)/(kmax-kmin)*pw; }}
  function sy(e){{ return pad.t+(1-(e-emin)/espan)*ph; }}

  function drawBand(arr,col) {{
    ctx.strokeStyle=col; ctx.lineWidth=2; ctx.beginPath();
    arr.forEach((e,i)=>{{ if(i===0) ctx.moveTo(sx(kz[i]),sy(e)); else ctx.lineTo(sx(kz[i]),sy(e)); }});
    ctx.stroke();
  }}
  drawBand(ep,'#58a6ff');
  drawBand(em,'#f78166');

  // Vertical lines at Weyl nodes
  if(hasNodes) {{
    ctx.strokeStyle='#8b949e'; ctx.lineWidth=1; ctx.setLineDash([4,4]);
    [DATA.node_kz_pos, DATA.node_kz_neg].forEach(kzn=>{{
      ctx.beginPath(); ctx.moveTo(sx(kzn),pad.t); ctx.lineTo(sx(kzn),H-pad.b); ctx.stroke();
    }});
    ctx.setLineDash([]);
  }}

  ctx.strokeStyle='#484f58'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,sy(0)); ctx.lineTo(W-pad.r,sy(0)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.stroke();
  ctx.fillStyle='#8b949e'; ctx.font='11px monospace'; ctx.textAlign='right';
  [emin,0,emax].forEach(e=>{{ ctx.fillText(e.toFixed(1),pad.l-4,sy(e)+4); }});
  ctx.textAlign='center';
  ['-π','0','π'].forEach((lbl,ii)=>{{ const v=[-Math.PI,0,Math.PI][ii]; ctx.fillText(lbl,sx(v),H-pad.b+14); }});
  ctx.textAlign='left'; ctx.fillStyle='#8b949e'; ctx.font='10px monospace'; ctx.fillText('E',2,pad.t+12);
  ctx.textAlign='center'; ctx.fillText('k\u1d69',W/2,H-4);
}}
drawCone();

// ── Node diagram: kz0 vs m ────────────────────────────────────────────────
const nodeCanvas = document.getElementById('nodePlot');
function drawNode() {{
  const dpr = window.devicePixelRatio||1;
  const W = nodeCanvas.clientWidth, H = parseInt(nodeCanvas.getAttribute('height'));
  nodeCanvas.width = W*dpr; nodeCanvas.height = H*dpr;
  const ctx = nodeCanvas.getContext('2d'); ctx.scale(dpr,dpr);
  const pad = {{l:50,r:20,t:20,b:40}};
  const pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;
  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);

  const sv=DATA.sweep_vals, sk=DATA.sweep_node_kz;
  if(!sv||sv.length===0) return;
  const smin=sv[0], smax=sv[sv.length-1];
  const ymin=-Math.PI, ymax=Math.PI;
  function sx(v){{ return pad.l+(v-smin)/(smax-smin)*pw; }}
  function sy(v){{ return pad.t+(ymax-v)/(ymax-ymin)*ph; }}

  // Two branches: +kz0 and -kz0
  ctx.lineWidth=2.5;
  const posPts = [], negPts = [];
  for(let i=0;i<sv.length;i++) {{
    if(!isNaN(sk[i])) {{ posPts.push([sv[i],sk[i]]); negPts.push([sv[i],-sk[i]]); }}
  }}
  function drawPts(pts,col) {{
    if(!pts.length) return;
    ctx.strokeStyle=col; ctx.beginPath();
    pts.forEach(([x,y],i)=>{{ if(i===0) ctx.moveTo(sx(x),sy(y)); else ctx.lineTo(sx(x),sy(y)); }});
    ctx.stroke();
  }}
  drawPts(posPts,'#f78166');
  drawPts(negPts,'#58a6ff');

  // Zero line
  ctx.strokeStyle='#484f58'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,sy(0)); ctx.lineTo(W-pad.r,sy(0)); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b); ctx.stroke();

  ctx.fillStyle='#8b949e'; ctx.font='11px monospace'; ctx.textAlign='right';
  ['-π','0','π'].forEach((lbl,ii)=>{{
    const v=[-Math.PI,0,Math.PI][ii]; ctx.fillText(lbl,pad.l-4,sy(v)+4);
  }});
  ctx.textAlign='center';
  [smin,(smin+smax)/2,smax].forEach(v=>{{ ctx.fillText(v.toFixed(1),sx(v),H-pad.b+14); }});
  ctx.fillStyle='#8b949e'; ctx.font='10px monospace';
  ctx.textAlign='left'; ctx.fillText('k\u1d69\u2080',2,pad.t+12);
  ctx.textAlign='center'; ctx.fillText('m',W/2,H-4);
}}
drawNode();

window.addEventListener('resize',()=>{{drawArc();drawChern();drawCone();drawNode();}});
}})();
</script>
</body>
</html>"""


def _fu_kane_frames(
    v_F: float = 1.0,
    M_val: float = 0.5,
    mu: float = 0.0,
    Delta: float = 0.3,
    n_k: int = 30,
    n_y: int = 24,
    n_kx: int = 80,
    scenario: str = "both",
    source: str = "python-reference",
) -> dict:
    """NumPy reference: Fu-Kane superconductor BdG spectrum and chiral Majorana edge mode."""
    import numpy as np  # noqa: PLC0415

    def fu_kane_ham(kx, ky):
        h01 = -v_F * (ky + 1j * kx)
        h10 =  v_F * (1j * kx - ky)
        return np.array([
            [M_val - mu,  h01,         0,                         Delta],
            [h10,        -M_val - mu, -Delta,                     0    ],
            [0,          -Delta,       mu - M_val, -v_F * (ky - 1j * kx)],
            [Delta,       0,          -v_F * (ky + 1j * kx),  M_val + mu],
        ], dtype=complex)

    def lower_evecs(kx, ky):
        _, vecs = np.linalg.eigh(fu_kane_ham(kx, ky))
        return vecs[:, :2]

    def chern_fk():
        kpts = np.linspace(-np.pi, np.pi, n_k, endpoint=False)
        Q = [[lower_evecs(kx, ky) for ky in kpts] for kx in kpts]

        def det_link(i0, j0, i1, j1):
            ov = Q[i0 % n_k][j0 % n_k].conj().T @ Q[i1 % n_k][j1 % n_k]
            d = np.linalg.det(ov)
            return d / abs(d) if abs(d) > 1e-14 else 1.0

        F = 0.0
        for i in range(n_k):
            for j in range(n_k):
                U1  = det_link(i,   j,   i+1, j  )
                U2  = det_link(i,   j,   i,   j+1)
                U1p = det_link(i,   j+1, i+1, j+1)
                U2p = det_link(i+1, j,   i+1, j+1)
                F  += np.angle(U1 * U2p * np.conj(U1p) * np.conj(U2))
        return int(round(F / (2 * np.pi)))

    M_c = float(np.sqrt(mu ** 2 + Delta ** 2))
    chern = chern_fk()
    phase_label = "Topological SC (C=1)" if chern != 0 else "Trivial SC (C=0)"

    kx_vals = np.linspace(-np.pi, np.pi, n_kx)
    bands = np.array([np.sort(np.linalg.eigvalsh(fu_kane_ham(kx, 0.0))) for kx in kx_vals])
    gap = float(np.min(np.abs(bands)))

    ivF2 = 1j * v_F * 0.5
    T = np.zeros((4, 4), dtype=complex)
    T[0, 1] = ivF2;  T[1, 0] = ivF2
    T[2, 3] = -ivF2; T[3, 2] = -ivF2

    N = 4 * n_y
    E_strip = np.zeros((n_kx, N))
    for i, kx in enumerate(kx_vals):
        H_big = np.zeros((N, N), dtype=complex)
        H4 = fu_kane_ham(kx, 0.0)
        for j in range(n_y):
            H_big[4*j:4*j+4, 4*j:4*j+4] += H4
        for j in range(n_y - 1):
            H_big[4*(j+1):4*(j+1)+4, 4*j:4*j+4] += T
            H_big[4*j:4*j+4, 4*(j+1):4*(j+1)+4] += T.conj().T
        E_strip[i] = np.sort(np.linalg.eigvalsh(H_big))

    return {
        "source": source,
        "v_F": v_F,
        "M_val": M_val,
        "mu": mu,
        "Delta": Delta,
        "M_critical": M_c,
        "chern": chern,
        "phase_label": phase_label,
        "gap": gap,
        "kx_bulk": kx_vals.tolist(),
        "E0": bands[:, 0].tolist(),
        "E1": bands[:, 1].tolist(),
        "E2": bands[:, 2].tolist(),
        "E3": bands[:, 3].tolist(),
        "kx_edge": kx_vals.tolist(),
        "E_edge": E_strip.flatten().tolist(),
        "n_edge_evals": N,
    }


def _fu_kane_page(data: dict, title: str = "Fu-Kane Superconductor \u2014 Majorana Zero Modes") -> str:
    """Render a 3-panel HTML page: BdG bands, strip edge spectrum, phase indicator."""
    import json as _json  # noqa: PLC0415
    DATA_JSON = _json.dumps(data)
    Mc_fmt = f"{data.get('M_critical', 0):.3f}"

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;color:#c9d1d9;font-family:'Segoe UI',system-ui,sans-serif;padding:20px}}
  h1{{font-size:1.15rem;font-weight:600;color:#e6edf3;margin-bottom:18px}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:1100px}}
  .card{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:16px}}
  .card h2{{font-size:0.85rem;color:#8b949e;margin-bottom:10px;text-transform:uppercase;letter-spacing:.05em}}
  canvas{{display:block;width:100%;cursor:crosshair}}
  .info-row{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;align-items:center}}
  .badge{{padding:4px 12px;border-radius:20px;font-size:0.8rem;font-weight:600}}
  .topo{{background:#1a3a1a;color:#3fb950;border:1px solid #3fb950}}
  .trivial{{background:#2a1a1a;color:#f85149;border:1px solid #f85149}}
  .param-box{{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:6px 12px;font-size:0.75rem;color:#8b949e}}
  .param-box .val{{font-size:1rem;font-weight:700;color:#e6edf3}}
  .legend{{display:flex;gap:16px;margin-top:8px;font-size:0.75rem;color:#8b949e;align-items:center}}
  .dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:middle}}
  .line{{display:inline-block;width:20px;height:3px;border-radius:1px;vertical-align:middle;margin-right:4px}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="info-row" id="infoRow"></div>

<div class="grid">
  <div class="card">
    <h2>BdG Bulk Spectrum (k<sub>y</sub>=0)</h2>
    <canvas id="bulkCanvas" height="260"></canvas>
    <div class="legend">
      <span><span class="line" style="background:#58a6ff"></span>E\u2080,E\u2083 outer bands</span>
      <span><span class="line" style="background:#f0883e"></span>E\u2081,E\u2082 inner bands</span>
    </div>
  </div>
  <div class="card">
    <h2>Strip Spectrum \u2014 Chiral Majorana Edge Mode</h2>
    <canvas id="edgeCanvas" height="260"></canvas>
    <div class="legend">
      <span><span class="dot" style="background:#8b949e"></span>bulk states</span>
      <span><span class="dot" style="background:#f78166"></span>in-gap (|E|&lt;\u0394/2)</span>
    </div>
  </div>
  <div class="card" style="grid-column:1/-1">
    <h2>Topological Phase Boundary \u2014 M<sub>c</sub> = \u221a(\u03bc\u00b2+\u0394\u00b2) = {Mc_fmt}</h2>
    <canvas id="phaseCanvas" height="120"></canvas>
  </div>
</div>

<script>
const DATA = {DATA_JSON};
const M_c = DATA.M_critical;
const topological = DATA.chern !== 0;

// Info row
(function(){{
  const ir = document.getElementById('infoRow');
  const badge = topological
    ? '<span class="badge topo">\u25cf ' + DATA.phase_label + '</span>'
    : '<span class="badge trivial">\u25cf ' + DATA.phase_label + '</span>';
  const params = [
    ['M', DATA.M_val.toFixed(3)],['M\u2099', M_c.toFixed(3)],
    ['\u03bc', DATA.mu.toFixed(3)],['\u0394', DATA.Delta.toFixed(3)],
    ['v\u1da0', DATA.v_F.toFixed(2)],['gap', DATA.gap.toFixed(4)],
  ];
  ir.innerHTML = badge + params.map(([l,v])=>
    `<div class="param-box"><div>${{l}}</div><div class="val">${{v}}</div></div>`
  ).join('');
}})();

// ── Shared helpers ────────────────────────────────────────────────────────
function setupCanvas(id, H){{
  const c=document.getElementById(id);
  const dpr=window.devicePixelRatio||1;
  const W=c.clientWidth;
  c.width=W*dpr; c.height=H*dpr;
  const ctx=c.getContext('2d'); ctx.scale(dpr,dpr);
  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);
  return {{ctx,W,H}};
}}

// ── Panel 1: BdG bulk bands ───────────────────────────────────────────────
(function(){{
  const {{ctx,W,H}}=setupCanvas('bulkCanvas',260);
  const pad={{l:46,r:14,t:14,b:34}};
  const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;
  const kx=DATA.kx_bulk; if(!kx||!kx.length) return;
  const kmin=kx[0],kmax=kx[kx.length-1];
  let ymax=0;
  [DATA.E0,DATA.E1,DATA.E2,DATA.E3].forEach(b=>b.forEach(v=>{{ymax=Math.max(ymax,Math.abs(v));}}));
  ymax*=1.08;
  function sx(v){{return pad.l+(v-kmin)/(kmax-kmin)*pw;}}
  function sy(v){{return pad.t+(1-v/ymax)*ph/2;}}

  ctx.strokeStyle='#30363d';ctx.lineWidth=1;ctx.setLineDash([4,4]);
  ctx.beginPath();ctx.moveTo(pad.l,sy(0));ctx.lineTo(W-pad.r,sy(0));ctx.stroke();
  ctx.setLineDash([]);

  const cols=['#58a6ff','#f0883e','#f0883e','#58a6ff'];
  [DATA.E0,DATA.E1,DATA.E2,DATA.E3].forEach((b,bi)=>{{
    ctx.strokeStyle=cols[bi];ctx.lineWidth=1.8;ctx.beginPath();
    b.forEach((v,i)=>{{i===0?ctx.moveTo(sx(kx[i]),sy(v)):ctx.lineTo(sx(kx[i]),sy(v));}});
    ctx.stroke();
  }});

  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,H-pad.b);ctx.stroke();
  ctx.fillStyle='#8b949e';ctx.font='10px monospace';ctx.textAlign='right';
  [ymax,0,-ymax].forEach(v=>ctx.fillText(v.toFixed(1),pad.l-4,sy(v)+4));
  ctx.textAlign='center';
  ['\u2212\u03c0','0','\u03c0'].forEach((l,i)=>{{
    ctx.fillText(l,sx([-Math.PI,0,Math.PI][i]),H-pad.b+14);
  }});
  ctx.fillText('k\u2093',W/2,H-4);
}})();

// ── Panel 2: Strip edge spectrum ──────────────────────────────────────────
(function(){{
  const {{ctx,W,H}}=setupCanvas('edgeCanvas',260);
  const pad={{l:46,r:14,t:14,b:34}};
  const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;
  const kx=DATA.kx_edge, Ee=DATA.E_edge, ne=DATA.n_edge_evals;
  if(!kx||!Ee) return;
  const nkx=kx.length;
  const kmin=kx[0],kmax=kx[kx.length-1];
  let ymax=0;
  for(let v of Ee) ymax=Math.max(ymax,Math.abs(v));
  ymax=Math.min(ymax*1.05,3.5*Math.abs(DATA.Delta)+0.2);
  function sx(v){{return pad.l+(v-kmin)/(kmax-kmin)*pw;}}
  function sy(v){{return pad.t+(1-v/ymax)*ph/2;}}

  ctx.strokeStyle='#30363d';ctx.lineWidth=1;ctx.setLineDash([4,4]);
  ctx.beginPath();ctx.moveTo(pad.l,sy(0));ctx.lineTo(W-pad.r,sy(0));ctx.stroke();
  ctx.strokeStyle='#58a6ff44';
  [DATA.Delta,-DATA.Delta].forEach(v=>{{
    ctx.beginPath();ctx.moveTo(pad.l,sy(v));ctx.lineTo(W-pad.r,sy(v));ctx.stroke();
  }});
  ctx.setLineDash([]);

  const thr=Math.abs(DATA.Delta)*0.55;
  for(let ik=0;ik<nkx;ik++){{
    for(let b=0;b<ne;b++){{
      const e=Ee[ik*ne+b];
      if(Math.abs(e)>ymax) continue;
      ctx.fillStyle=Math.abs(e)<thr?'#f78166':'#8b949e55';
      ctx.fillRect(sx(kx[ik])-0.8,sy(e)-0.8,1.6,1.6);
    }}
  }}

  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,H-pad.b);ctx.stroke();
  ctx.fillStyle='#8b949e';ctx.font='10px monospace';ctx.textAlign='right';
  [ymax,0,-ymax].forEach(v=>ctx.fillText(v.toFixed(2),pad.l-4,sy(v)+4));
  ctx.textAlign='center';
  ['\u2212\u03c0','0','\u03c0'].forEach((l,i)=>{{
    ctx.fillText(l,sx([-Math.PI,0,Math.PI][i]),H-pad.b+14);
  }});
  ctx.fillText('k\u2093',W/2,H-4);
}})();

// ── Panel 3: Phase indicator bar ──────────────────────────────────────────
(function(){{
  const {{ctx,W,H}}=setupCanvas('phaseCanvas',120);
  const pad={{l:60,r:30,t:34,b:36}};
  const pw=W-pad.l-pad.r;
  const Mshow=Math.max(DATA.M_val,M_c)*1.45+0.4;
  function sx(v){{return pad.l+(v/Mshow)*pw;}}
  const barY=pad.t, barH=26;

  // Topo region
  const gradT=ctx.createLinearGradient(pad.l,0,sx(M_c),0);
  gradT.addColorStop(0,'#1a3a1a');gradT.addColorStop(1,'#2a4a2a');
  ctx.fillStyle=gradT;ctx.fillRect(pad.l,barY,sx(M_c)-pad.l,barH);
  ctx.fillStyle='#161b22';ctx.fillRect(sx(M_c),barY,W-pad.r-sx(M_c),barH);
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.strokeRect(pad.l,barY,pw,barH);

  // M_c
  ctx.strokeStyle='#f0883e';ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(sx(M_c),barY-6);ctx.lineTo(sx(M_c),barY+barH+6);ctx.stroke();
  ctx.fillStyle='#f0883e';ctx.font='10px monospace';ctx.textAlign='center';
  ctx.fillText('M\u2099='+M_c.toFixed(3),sx(M_c),barY-9);

  // Current M
  ctx.strokeStyle=topological?'#3fb950':'#f85149';ctx.lineWidth=3;
  ctx.beginPath();ctx.moveTo(sx(DATA.M_val),barY);ctx.lineTo(sx(DATA.M_val),barY+barH);ctx.stroke();
  ctx.fillStyle=topological?'#3fb950':'#f85149';ctx.font='bold 10px monospace';ctx.textAlign='center';
  ctx.fillText('M='+DATA.M_val.toFixed(3),sx(DATA.M_val),barY+barH+14);

  ctx.font='10px monospace';ctx.textAlign='center';
  ctx.fillStyle='#3fb950';
  ctx.fillText('Topological  C=1',pad.l+(sx(M_c)-pad.l)/2,barY+barH/2+4);
  ctx.fillStyle='#8b949e';
  ctx.fillText('Trivial  C=0',sx(M_c)+(W-pad.r-sx(M_c))/2,barY+barH/2+4);

  [0,M_c,Mshow*0.9].forEach(v=>{{
    ctx.fillStyle='#8b949e';ctx.font='9px monospace';ctx.textAlign='center';
    ctx.fillText(v.toFixed(2),sx(v),barY+barH+26);
    ctx.strokeStyle='#484f58';ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(sx(v),barY+barH);ctx.lineTo(sx(v),barY+barH+5);ctx.stroke();
  }});
  ctx.fillStyle='#8b949e';ctx.font='10px monospace';ctx.textAlign='center';
  ctx.fillText('M',W/2,H-4);
}})();
</script>
</body>
</html>"""


def _lutchyn_oreg_frames(
    t: float = 1.0,
    alpha: float = 0.5,
    mu: float = 0.0,
    Delta: float = 0.25,
    V_z: float = 0.0,
    n_k: int = 200,
    n_sites: int = 60,
    n_Vz: int = 50,
    n_mu: int = 50,
    Vz_max: float = 2.0,
    mu_max: float = 2.0,
    scenario: str = "bulk",
    source: str = "python-reference",
) -> dict:
    """NumPy reference: Lutchyn-Oreg Majorana nanowire BdG computation."""
    import numpy as np  # noqa: PLC0415

    def lo_ham_k(k):
        eps = 2*t*(1 - np.cos(k)) - mu
        sk  = np.sin(k)
        # 4×4 BdG in {e↑, e↓, h↓, -h↑} basis
        H = np.zeros((4, 4), dtype=complex)
        # electron sector
        H[0, 0] = eps;  H[1, 1] = eps
        H[0, 1] = V_z - 1j*alpha*sk;  H[1, 0] = V_z + 1j*alpha*sk
        # pairing
        H[0, 3] =  Delta;  H[3, 0] =  Delta
        H[1, 2] = -Delta;  H[2, 1] = -Delta
        # hole sector
        H[2, 2] = -eps;  H[3, 3] = -eps
        H[2, 3] = -V_z + 1j*alpha*sk;  H[3, 2] = -V_z - 1j*alpha*sk
        return H

    def compute_z2():
        pf0  = V_z**2 - mu**2 - Delta**2
        pfpi = V_z**2 - (mu - 4*t)**2 - Delta**2
        return -1 if pf0 * pfpi < 0 else +1

    z2 = compute_z2()
    Vzc = np.sqrt(mu**2 + Delta**2)

    # Bulk bands
    k_vals = np.linspace(-np.pi, np.pi, n_k)
    bands  = np.zeros((n_k, 4))
    for i, k in enumerate(k_vals):
        ev = np.linalg.eigvalsh(lo_ham_k(k))
        bands[i] = np.sort(ev)
    gap = float(np.min(np.abs(bands)))

    result: dict = {
        "source":      source,
        "scenario":    scenario,
        "t": t, "alpha": alpha, "mu": mu, "Delta": Delta, "V_z": V_z,
        "z2":          z2,
        "phase_label": f"Topological SC (Z\u2082=-1)" if z2 < 0 else "Trivial SC (Z\u2082=+1)",
        "gap":         gap,
        "V_z_critical": float(Vzc),
        "k_bulk":  k_vals.tolist(),
        "E0": bands[:, 0].tolist(),
        "E1": bands[:, 1].tolist(),
        "E2": bands[:, 2].tolist(),
        "E3": bands[:, 3].tolist(),
    }

    if scenario in ("wire", "both"):
        N = 4 * n_sites
        H_big = np.zeros((N, N), dtype=complex)
        eps0 = 2*t - mu
        ah   = alpha / 2.0
        for j in range(n_sites):
            b = 4 * j
            # onsite electron
            H_big[b, b] = eps0;       H_big[b+1, b+1] = eps0
            H_big[b, b+1] = V_z;      H_big[b+1, b] = V_z
            # pairing
            H_big[b, b+3]   =  Delta; H_big[b+3, b]   =  Delta
            H_big[b+1, b+2] = -Delta; H_big[b+2, b+1] = -Delta
            # onsite hole
            H_big[b+2, b+2] = -eps0;  H_big[b+3, b+3] = -eps0
            H_big[b+2, b+3] = -V_z;   H_big[b+3, b+2] = -V_z
            if j + 1 < n_sites:
                b2 = 4 * (j + 1)
                # electron hop
                H_big[b, b2]     = -t;   H_big[b2, b]     = -t
                H_big[b+1, b2+1] = -t;   H_big[b2+1, b+1] = -t
                H_big[b, b2+1]   = -ah;  H_big[b2+1, b]   = -ah
                H_big[b+1, b2]   =  ah;  H_big[b2, b+1]   =  ah
                # hole hop (sign of t flipped, same spin structure)
                H_big[b+2, b2+2] =  t;   H_big[b2+2, b+2] =  t
                H_big[b+3, b2+3] =  t;   H_big[b2+3, b+3] =  t
                H_big[b+2, b2+3] = -ah;  H_big[b2+3, b+2] = -ah
                H_big[b+3, b2+2] =  ah;  H_big[b2+2, b+3] =  ah

        ev_w, evec_w = np.linalg.eigh(H_big)
        # LDOS at ends (first 2 and last 2 sites)
        n_edge = min(2, n_sites)
        ldos_l = np.zeros(N)
        ldos_r = np.zeros(N)
        for n in range(N):
            for j in range(n_edge):
                ldos_l[n] += float(np.sum(np.abs(evec_w[4*j:4*j+4, n])**2))
            for j in range(n_sites - n_edge, n_sites):
                ldos_r[n] += float(np.sum(np.abs(evec_w[4*j:4*j+4, n])**2))

        # Majorana oscillations
        osc_L   = list(range(10, n_sites + 1, 5))
        osc_gap = []
        for L in osc_L:
            Nl = 4 * L
            Hl = H_big[:Nl, :Nl].copy()
            evl = np.linalg.eigvalsh(Hl)
            osc_gap.append(float(np.min(np.abs(evl))))

        result["E_wire"]    = ev_w.tolist()
        result["ldos_left"] = ldos_l.tolist()
        result["ldos_right"]= ldos_r.tolist()
        result["osc_L"]     = osc_L
        result["osc_gap"]   = osc_gap

    if scenario == "phase_diagram":
        Vz_arr = np.linspace(0, Vz_max, n_Vz)
        mu_arr = np.linspace(-mu_max, mu_max, n_mu)
        pd_z2  = np.zeros((n_Vz, n_mu), dtype=int)
        for iVz, Vzi in enumerate(Vz_arr):
            for imu, mui in enumerate(mu_arr):
                pf0  = Vzi**2 - mui**2 - Delta**2
                pfpi = Vzi**2 - (mui - 4*t)**2 - Delta**2
                pd_z2[iVz, imu] = -1 if pf0 * pfpi < 0 else +1
        result["pd_Vz_vals"] = Vz_arr.tolist()
        result["pd_mu_vals"] = mu_arr.tolist()
        result["pd_z2"]      = pd_z2.flatten().tolist()

    return result


def _lutchyn_oreg_page(
    data: dict,
    title: str = "Lutchyn-Oreg Majorana Nanowire",
) -> str:
    """Render a 3-panel HTML page: BdG bulk bands, wire spectrum + LDOS, Majorana oscillations."""
    import json as _json  # noqa: PLC0415
    DATA_JSON = _json.dumps(data)
    Vzc_fmt = f"{data.get('V_z_critical', 0):.3f}"
    z2      = data.get("z2", 1)
    phase   = data.get("phase_label", "")
    phase_color = "#3fb950" if z2 < 0 else "#8b949e"

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  :root{{--bg:#0d1117;--surface:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;
        --topo:#3fb950;--trivial:#6e7681}}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',sans-serif;padding:20px;min-height:100vh}}
  h1{{font-size:1.2rem;margin-bottom:4px}}
  .subtitle{{color:var(--muted);font-size:0.85rem;margin-bottom:20px}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:1100px}}
  .card{{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:20px;width:100%}}
  .card h2{{font-size:1rem;color:var(--muted);margin-bottom:14px}}
  .wide{{grid-column:1/-1}}
  .info-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:16px}}
  .info-box{{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:10px;text-align:center}}
  .info-box .label{{font-size:0.75rem;color:var(--muted);margin-bottom:4px}}
  .info-box .value{{font-size:1.3rem;font-weight:700}}
  .topo{{color:var(--topo)}} .trivial{{color:var(--trivial)}}
  canvas{{display:block;width:100%;cursor:crosshair}}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="subtitle">1D Rashba nanowire + s-wave proximity SC + Zeeman field &mdash; Majorana end modes</p>

<div class="grid">
  <div class="card wide">
    <div class="info-grid">
      <div class="info-box"><div class="label">Phase</div>
        <div class="value" style="font-size:1rem;color:{phase_color}">{phase}</div></div>
      <div class="info-box"><div class="label">V&#8239;_z / V&#8239;_z,c</div>
        <div class="value">{data.get('V_z',0):.3f} / {Vzc_fmt}</div></div>
      <div class="info-box"><div class="label">Bulk gap</div>
        <div class="value">{data.get('gap',0):.4f}</div></div>
      <div class="info-box"><div class="label">Z&#8322; invariant</div>
        <div class="value" style="color:{phase_color}">{z2:+d}</div></div>
      <div class="info-box"><div class="label">&Delta; (SC gap)</div>
        <div class="value">{data.get('Delta',0):.3f}</div></div>
      <div class="info-box"><div class="label">&alpha; (Rashba)</div>
        <div class="value">{data.get('alpha',0):.3f}</div></div>
    </div>
  </div>

  <!-- BdG bulk bands -->
  <div class="card">
    <h2>BdG Bulk Band Structure</h2>
    <canvas id="bulkPlot" height="280"></canvas>
  </div>

  <!-- Wire spectrum + LDOS -->
  <div class="card">
    <h2>Wire Spectrum &amp; End LDOS</h2>
    <canvas id="wirePlot" height="280"></canvas>
  </div>

  <!-- Majorana oscillations -->
  <div class="card wide">
    <h2>Majorana Gap Oscillations vs Wire Length</h2>
    <canvas id="oscPlot" height="200"></canvas>
  </div>
</div>

<script>
(function(){{
const DATA = {DATA_JSON};
const Vzc  = DATA.V_z_critical || 0;
const isTop = DATA.z2 < 0;

// ── Helper ────────────────────────────────────────────────────────────────
function drawAxes(ctx,pad,W,H,xlabels,ylabels){{
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,H-pad.b);
  ctx.moveTo(pad.l,H-pad.b); ctx.lineTo(W-pad.r,H-pad.b); ctx.stroke();
  ctx.fillStyle='#8b949e'; ctx.font='11px monospace'; ctx.textAlign='center';
  xlabels.forEach(([v,lbl])=>{{ ctx.fillText(lbl,v,H-pad.b+14); }});
  ctx.textAlign='right';
  ylabels.forEach(([v,lbl])=>{{ ctx.fillText(lbl,pad.l-4,v+4); }});
}}

// ── Bulk BdG bands ────────────────────────────────────────────────────────
const bulkCanvas = document.getElementById('bulkPlot');
function drawBulk(){{
  const dpr=window.devicePixelRatio||1;
  const W=bulkCanvas.clientWidth, H=parseInt(bulkCanvas.getAttribute('height'));
  bulkCanvas.width=W*dpr; bulkCanvas.height=H*dpr;
  const ctx=bulkCanvas.getContext('2d'); ctx.scale(dpr,dpr);
  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);
  const pad={{l:50,r:15,t:15,b:35}};
  const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;

  const k=DATA.k_bulk||[], E0=DATA.E0||[], E1=DATA.E1||[], E2=DATA.E2||[], E3=DATA.E3||[];
  if(!k.length) return;

  const allE=[...E0,...E1,...E2,...E3];
  const kmin=k[0], kmax=k[k.length-1];
  const Emin=Math.min(...allE)*1.1, Emax=Math.max(...allE)*1.1;
  const sx=v=>pad.l+(v-kmin)/(kmax-kmin)*pw;
  const sy=e=>pad.t+(1-(e-Emin)/(Emax-Emin))*ph;

  // Shaded gap region
  if(isTop){{ ctx.fillStyle='rgba(63,185,80,0.07)'; ctx.fillRect(pad.l,pad.t,pw,ph); }}

  // Zero line
  ctx.strokeStyle='#21262d'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(pad.l,sy(0)); ctx.lineTo(W-pad.r,sy(0)); ctx.stroke();
  ctx.setLineDash([]);

  // Bands
  const colors=['#58a6ff','#79c0ff','#58a6ff','#79c0ff'];
  [E0,E1,E2,E3].forEach((band,bi)=>{{
    ctx.strokeStyle=colors[bi]; ctx.lineWidth=1.5;
    ctx.beginPath();
    band.forEach((e,i)=>{{ i===0?ctx.moveTo(sx(k[i]),sy(e)):ctx.lineTo(sx(k[i]),sy(e)); }});
    ctx.stroke();
  }});

  drawAxes(ctx,pad,W,H,
    [[sx(-Math.PI),'-π'],[sx(0),'0'],[sx(Math.PI),'π']],
    [[sy(0),'0']]
  );
  ctx.fillStyle='#8b949e'; ctx.font='10px monospace';
  ctx.textAlign='left'; ctx.fillText('E',2,pad.t+12);
  ctx.textAlign='center'; ctx.fillText('k',W/2,H-4);
}}
drawBulk();

// ── Wire spectrum + LDOS ──────────────────────────────────────────────────
const wireCanvas = document.getElementById('wirePlot');
function drawWire(){{
  const dpr=window.devicePixelRatio||1;
  const W=wireCanvas.clientWidth, H=parseInt(wireCanvas.getAttribute('height'));
  wireCanvas.width=W*dpr; wireCanvas.height=H*dpr;
  const ctx=wireCanvas.getContext('2d'); ctx.scale(dpr,dpr);
  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);
  const pad={{l:50,r:15,t:15,b:35}};
  const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;

  const Ew=DATA.E_wire||[], ldosL=DATA.ldos_left||[], ldosR=DATA.ldos_right||[];
  if(!Ew.length){{ ctx.fillStyle='#8b949e'; ctx.font='13px monospace';
    ctx.textAlign='center'; ctx.fillText('(run with scenario "wire" or "both")',W/2,H/2); return; }}

  const N=Ew.length;
  const Emin=Math.min(...Ew)*1.1, Emax=Math.max(...Ew)*1.1;
  const wmax=Math.max(...ldosL,...ldosR)||1;
  // x-axis: eigenstate index; y-axis: energy
  const sx=i=>pad.l+i/N*pw;
  const sy=e=>pad.t+(1-(e-Emin)/(Emax-Emin))*ph;

  // Zero line
  ctx.strokeStyle='#21262d'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(pad.l,sy(0)); ctx.lineTo(W-pad.r,sy(0)); ctx.stroke();
  ctx.setLineDash([]);

  // Scatter: colour by LDOS weight at ends
  for(let n=0;n<N;n++){{
    const w=Math.min(1,(ldosL[n]+ldosR[n])/wmax);
    const r=Math.round(63+(255-63)*w), gb=Math.round(185*(1-w));
    ctx.fillStyle=`rgb(${{r}},${{gb}},${{gb}})`;
    ctx.fillRect(sx(n)-1, sy(Ew[n])-1, 2, 2);
  }}

  ctx.fillStyle='#8b949e'; ctx.font='10px monospace';
  ctx.textAlign='left'; ctx.fillText('E',2,pad.t+12);
  ctx.textAlign='center'; ctx.fillText('eigenstate index',W/2,H-4);
  // Zero label
  ctx.fillStyle='#484f58'; ctx.textAlign='right'; ctx.fillText('0',pad.l-4,sy(0)+4);
}}
drawWire();

// ── Majorana oscillations ─────────────────────────────────────────────────
const oscCanvas = document.getElementById('oscPlot');
function drawOsc(){{
  const dpr=window.devicePixelRatio||1;
  const W=oscCanvas.clientWidth, H=parseInt(oscCanvas.getAttribute('height'));
  oscCanvas.width=W*dpr; oscCanvas.height=H*dpr;
  const ctx=oscCanvas.getContext('2d'); ctx.scale(dpr,dpr);
  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);
  const pad={{l:55,r:15,t:15,b:35}};
  const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;

  const L=DATA.osc_L||[], G=DATA.osc_gap||[];
  if(!L.length){{ ctx.fillStyle='#8b949e'; ctx.font='13px monospace';
    ctx.textAlign='center'; ctx.fillText('(run with scenario "wire" or "both")',W/2,H/2); return; }}

  const Lmin=L[0], Lmax=L[L.length-1];
  const Gmax=Math.max(...G)||1;
  const sx=v=>pad.l+(v-Lmin)/(Lmax-Lmin)*pw;
  const sy=g=>pad.t+(1-g/Gmax)*ph;

  // Shaded background
  ctx.fillStyle='rgba(30,40,30,0.5)'; ctx.fillRect(pad.l,pad.t,pw,ph);

  // Line
  const col=isTop?'#3fb950':'#8b949e';
  ctx.strokeStyle=col; ctx.lineWidth=2;
  ctx.beginPath();
  L.forEach((v,i)=>{{ i===0?ctx.moveTo(sx(v),sy(G[i])):ctx.lineTo(sx(v),sy(G[i])); }});
  ctx.stroke();
  // Dots
  ctx.fillStyle=col;
  L.forEach((v,i)=>{{ ctx.beginPath(); ctx.arc(sx(v),sy(G[i]),3,0,2*Math.PI); ctx.fill(); }});

  drawAxes(ctx,pad,W,H,
    [[sx(Lmin),String(Lmin)],[sx(Math.round((Lmin+Lmax)/2)),String(Math.round((Lmin+Lmax)/2))],[sx(Lmax),String(Lmax)]],
    [[sy(Gmax),Gmax.toFixed(3)],[sy(0),'0']]
  );
  ctx.fillStyle='#8b949e'; ctx.font='10px monospace';
  ctx.textAlign='left'; ctx.fillText('gap',2,pad.t+12);
  ctx.textAlign='center'; ctx.fillText('wire length (sites)',W/2,H-4);
}}
drawOsc();

window.addEventListener('resize',()=>{{drawBulk();drawWire();drawOsc();}});
}})();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# M42  Kitaev honeycomb model
# ─────────────────────────────────────────────────────────────────────────────

def _kitaev_honeycomb_frames(
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    kappa: float = 0.3,
    n_k: int = 60,
    n_pd: int = 80,
    scenario: str = "both",
    source: str = "python-reference",
) -> dict:
    """NumPy reference: Kitaev honeycomb model spectrum + ternary phase diagram."""
    import numpy as np  # noqa: PLC0415

    PI = math.pi

    # Honeycomb reciprocal lattice vectors (a=1 unit cell)
    # b1 = 2π(1, -1/√3),  b2 = 2π(0, 2/√3)
    b1 = np.array([2 * PI, -2 * PI / math.sqrt(3)])
    b2 = np.array([0.0,     4 * PI / math.sqrt(3)])

    def kitaev_ham(kx: float, ky: float) -> np.ndarray:
        """2×2 Kitaev honeycomb Bloch Hamiltonian.

        a1 = (1, 0),  a2 = (1/2, sqrt(3)/2)
        f(k) = Jz + Jx*exp(i k·a1) + Jy*exp(i k·a2)
        m(k) = 2κ[sin(k·a1) + sin(k·a2) + sin(k·(a2-a1))]
        """
        sqrt3_half = math.sqrt(3) * 0.5
        dot_a1 = kx                           # k · a1 = kx
        dot_a2 = kx * 0.5 + ky * sqrt3_half  # k · a2 = kx/2 + ky*√3/2
        fk = Jz + Jx * np.exp(1j * dot_a1) + Jy * np.exp(1j * dot_a2)
        mk = 2.0 * kappa * (
            math.sin(dot_a1) + math.sin(dot_a2) + math.sin(dot_a2 - dot_a1)
        )
        H = np.array([[ mk,       1j * fk ],
                      [-1j * np.conj(fk), -mk]], dtype=complex)
        return H

    def classify_phase(jx: float, jy: float, jz: float) -> str:
        """Classify by triangle inequality (gapped/gapless phases)."""
        vals = sorted([jx, jy, jz])
        if vals[2] > vals[0] + vals[1]:
            # Gapped A phases
            if jx >= jy and jx >= jz:
                return "A_x"
            elif jy >= jx and jy >= jz:
                return "A_y"
            else:
                return "A_z"
        else:
            return "B"

    out: dict = {"source": source, "Jx": Jx, "Jy": Jy, "Jz": Jz, "kappa": kappa}

    # ── Spectrum over honeycomb BZ ──────────────────────────────────────────
    if scenario in ("spectrum", "chern", "both"):
        kx_list, ky_list, ep_list, em_list = [], [], [], []
        for si in range(n_k):
            for ti in range(n_k):
                s = si / n_k
                t = ti / n_k
                kv = s * b1 + t * b2
                H = kitaev_ham(kv[0], kv[1])
                evals = np.linalg.eigvalsh(H)
                kx_list.append(float(kv[0]))
                ky_list.append(float(kv[1]))
                em_list.append(float(evals[0]))
                ep_list.append(float(evals[1]))

        gap = float(min(ep_list[i] - em_list[i] for i in range(len(ep_list))))
        out.update({
            "kx_bz": kx_list, "ky_bz": ky_list,
            "E_plus": ep_list, "E_minus": em_list,
            "gap": gap,
        })

        # FHS Chern number on lower band
        chern = 0
        if kappa != 0.0:
            F_sum = 0.0
            for si in range(n_k):
                for ti in range(n_k):
                    def evec_lower(s2, t2):
                        kv2 = (s2 / n_k) * b1 + (t2 / n_k) * b2
                        H2 = kitaev_ham(kv2[0], kv2[1])
                        _, vecs = np.linalg.eigh(H2)
                        return vecs[:, 0]

                    u00 = evec_lower(si,   ti)
                    u10 = evec_lower(si+1, ti)
                    u01 = evec_lower(si,   ti+1)
                    u11 = evec_lower(si+1, ti+1)

                    U1  = np.vdot(u00, u10); U1  /= abs(U1)  if abs(U1)>1e-12 else 1.0
                    U2  = np.vdot(u10, u11); U2  /= abs(U2)  if abs(U2)>1e-12 else 1.0
                    U1p = np.vdot(u01, u11); U1p /= abs(U1p) if abs(U1p)>1e-12 else 1.0
                    U2p = np.vdot(u00, u01); U2p /= abs(U2p) if abs(U2p)>1e-12 else 1.0

                    F_sum += math.log((U1 * U2 * np.conj(U1p) * np.conj(U2p)).real
                                       + 1j * (U1 * U2 * np.conj(U1p) * np.conj(U2p)).imag).imag

            chern = int(round(F_sum / (2 * PI)))
        out["chern"] = chern
        out["phase_label"] = classify_phase(Jx, Jy, Jz) + (f" C={chern:+d}" if kappa != 0.0 else "")

    # ── Ternary phase diagram ───────────────────────────────────────────────
    if scenario in ("phase_diagram", "both"):
        pd_jx, pd_jy, pd_jz, pd_gap, pd_phase = [], [], [], [], []
        # Sample uniform simplex: Jx+Jy+Jz=1, Jx,Jy,Jz≥0
        for si in range(n_pd + 1):
            for ti in range(n_pd + 1 - si):
                ui = n_pd - si - ti
                jx_s = si / n_pd
                jy_s = ti / n_pd
                jz_s = ui / n_pd
                # Quick gap estimate at Γ point
                H = np.array([[0, 1j*(jx_s + jy_s + jz_s)],
                               [-1j*(jx_s + jy_s + jz_s), 0]], dtype=complex)
                evals = np.linalg.eigvalsh(H)
                gap_s = float(evals[1] - evals[0])
                pd_jx.append(jx_s); pd_jy.append(jy_s); pd_jz.append(jz_s)
                pd_gap.append(gap_s)
                pd_phase.append(classify_phase(jx_s, jy_s, jz_s))

        out.update({
            "pd_Jx": pd_jx, "pd_Jy": pd_jy, "pd_Jz": pd_jz,
            "pd_gap": pd_gap, "pd_phase": pd_phase,
        })

    return out


def _kitaev_honeycomb_page(data: dict) -> str:
    """Render Kitaev honeycomb results as a self-contained HTML page."""
    payload = json.dumps(data, separators=(",", ":"))
    phase_label = data.get("phase_label", "")
    chern = data.get("chern", 0)
    gap = data.get("gap", 0.0)
    Jx = data.get("Jx", 1.0); Jy = data.get("Jy", 1.0); Jz = data.get("Jz", 1.0)
    kappa = data.get("kappa", 0.0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Kitaev Honeycomb — QuantX</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:16px}}
  h2{{color:#e6edf3;font-size:1.1rem;margin-bottom:12px}}
  .info{{display:flex;gap:24px;flex-wrap:wrap;margin-bottom:14px;font-size:.82rem;color:#8b949e}}
  .info span b{{color:#c9d1d9}}
  .panels{{display:flex;gap:16px;flex-wrap:wrap}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}}
  canvas{{display:block}}
</style>
</head>
<body>
<h2>Kitaev Honeycomb Model</h2>
<div class="info">
  <span><b>Jx</b> = {Jx:.3f}</span>
  <span><b>Jy</b> = {Jy:.3f}</span>
  <span><b>Jz</b> = {Jz:.3f}</span>
  <span><b>κ</b> = {kappa:.3f}</span>
  <span><b>Phase</b>: {phase_label}</span>
  <span><b>Gap</b>: {gap:.4f}</span>
  <span><b>Chern</b>: {chern:+d}</span>
</div>
<div class="panels">
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Majorana Band Structure (BZ scatter)</div>
    <canvas id="cBand" width="480" height="380"></canvas>
  </div>
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Phase Diagram (Jx+Jy+Jz=1 simplex)</div>
    <canvas id="cPhase" width="420" height="380"></canvas>
  </div>
</div>
<script>
(function(){{
'use strict';
const DATA={payload};

// ── colour helpers ────────────────────────────────────────────────────────
function lerp3(a,b,t){{return[a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1]),a[2]+t*(b[2]-a[2])];}}
function energyColor(e,emin,emax){{
  const t=Math.max(0,Math.min(1,(e-emin)/(emax-emin||1)));
  // blue→white→red
  const c=t<0.5?lerp3([88,166,255],[200,200,200],t*2):lerp3([200,200,200],[248,81,73],(t-0.5)*2);
  return `rgb(${{Math.round(c[0])}},${{Math.round(c[1])}},${{Math.round(c[2])}})`;
}}
const PHASE_COLOR={{"A_x":"#e8b86d","A_y":"#58a6ff","A_z":"#3fb950","B":"#bc8cff"}};

// ── Band scatter ──────────────────────────────────────────────────────────
function drawBand(){{
  const cv=document.getElementById('cBand');
  const ctx=cv.getContext('2d');
  const W=cv.width, H=cv.height;
  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);
  const pad={{l:55,r:20,t:20,b:40}};
  const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;

  const kx=DATA.kx_bz||[], ep=DATA.E_plus||[], em=DATA.E_minus||[];
  if(!kx.length){{
    ctx.fillStyle='#8b949e';ctx.font='12px monospace';ctx.textAlign='center';
    ctx.fillText('(run with scenario "spectrum" or "both")',W/2,H/2); return;
  }}

  const kmin=Math.min(...kx), kmax=Math.max(...kx);
  const emin=Math.min(...em), emax=Math.max(...ep);
  const sx=v=>pad.l+(v-kmin)/(kmax-kmin||1)*pw;
  const sy=e=>pad.t+(1-(e-emin)/(emax-emin||1))*ph;

  // Draw lower band dots
  for(let i=0;i<kx.length;i++){{
    ctx.fillStyle=energyColor(em[i],emin,emax);
    ctx.fillRect(Math.round(sx(kx[i])),Math.round(sy(em[i])),2,2);
  }}
  // Draw upper band dots (lighter)
  for(let i=0;i<kx.length;i++){{
    ctx.fillStyle=energyColor(ep[i],emin,emax);
    ctx.fillRect(Math.round(sx(kx[i])),Math.round(sy(ep[i])),2,2);
  }}

  // Axes
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+ph); ctx.lineTo(pad.l+pw,pad.t+ph); ctx.stroke();
  ctx.fillStyle='#8b949e'; ctx.font='10px monospace';
  ctx.textAlign='center'; ctx.fillText('kx',W/2,H-4);
  ctx.save(); ctx.translate(12,H/2); ctx.rotate(-Math.PI/2);
  ctx.fillText('E',0,0); ctx.restore();
  // E=0 line
  if(emin<0 && emax>0){{
    ctx.strokeStyle='rgba(255,255,255,0.15)'; ctx.setLineDash([3,3]);
    ctx.beginPath(); ctx.moveTo(pad.l,sy(0)); ctx.lineTo(pad.l+pw,sy(0)); ctx.stroke();
    ctx.setLineDash([]);
  }}
  // Gap annotation
  const gap=DATA.gap||0;
  ctx.fillStyle='#8b949e'; ctx.font='10px monospace';
  ctx.textAlign='left';
  ctx.fillText(`gap=${{gap.toFixed(4)}}  C=${{DATA.chern>=0?'+':''}}${{DATA.chern}}`,pad.l+4,pad.t+12);
}}

// ── Ternary phase diagram ─────────────────────────────────────────────────
function drawPhase(){{
  const cv=document.getElementById('cPhase');
  const ctx=cv.getContext('2d');
  const W=cv.width, H=cv.height;
  ctx.fillStyle='#0d1117'; ctx.fillRect(0,0,W,H);

  const pjx=DATA.pd_Jx||[], pjy=DATA.pd_Jy||[], pjz=DATA.pd_Jz||[], ph=DATA.pd_phase||[];
  if(!pjx.length){{
    ctx.fillStyle='#8b949e';ctx.font='12px monospace';ctx.textAlign='center';
    ctx.fillText('(run with scenario "phase_diagram" or "both")',W/2,H/2); return;
  }}

  // Ternary → cartesian: equilateral triangle
  // Jx at bottom-left, Jy at bottom-right, Jz at top
  const pad=30;
  const triW=W-2*pad, triH=H-2*pad-20;
  // Vertices
  const vx={{x:pad, y:pad+triH}};
  const vy={{x:pad+triW, y:pad+triH}};
  const vz={{x:pad+triW/2, y:pad}};
  function ternaryToXY(jx,jy,jz){{
    return{{
      x: jx*vx.x + jy*vy.x + jz*vz.x,
      y: jx*vx.y + jy*vy.y + jz*vz.y,
    }};
  }}

  // Draw dots
  const R=4;
  for(let i=0;i<pjx.length;i++){{
    const p=ternaryToXY(pjx[i],pjy[i],pjz[i]);
    ctx.fillStyle=PHASE_COLOR[ph[i]]||'#8b949e';
    ctx.beginPath(); ctx.arc(p.x,p.y,R,0,2*Math.PI); ctx.fill();
  }}

  // Triangle border
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1.5;
  ctx.beginPath();
  ctx.moveTo(vx.x,vx.y); ctx.lineTo(vy.x,vy.y);
  ctx.lineTo(vz.x,vz.y); ctx.closePath(); ctx.stroke();

  // Axis labels
  ctx.fillStyle='#8b949e'; ctx.font='11px monospace';
  ctx.textAlign='center';
  ctx.fillText('Jx=1',vx.x,vx.y+14);
  ctx.fillText('Jy=1',vy.x,vy.y+14);
  ctx.fillText('Jz=1',vz.x,vz.y-6);

  // Legend
  const legend=[['A_x','#e8b86d'],['A_y','#58a6ff'],['A_z','#3fb950'],['B','#bc8cff']];
  legend.forEach(([lbl,col],i)=>{{
    ctx.fillStyle=col; ctx.fillRect(pad+i*80,H-16,10,10);
    ctx.fillStyle='#8b949e'; ctx.textAlign='left';
    ctx.fillText(lbl,pad+i*80+13,H-6);
  }});

  // Current point marker
  const cp=ternaryToXY(DATA.Jx/(DATA.Jx+DATA.Jy+DATA.Jz),DATA.Jy/(DATA.Jx+DATA.Jy+DATA.Jz),DATA.Jz/(DATA.Jx+DATA.Jy+DATA.Jz));
  ctx.strokeStyle='#ffffff'; ctx.lineWidth=1.5;
  ctx.beginPath();
  ctx.arc(cp.x,cp.y,7,0,2*Math.PI); ctx.stroke();
}}

drawBand();
drawPhase();
window.addEventListener('resize',()=>{{drawBand();drawPhase();}});
}})();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# M43  BBH Quadrupole Insulator (Higher-Order Topological Insulator)
# ─────────────────────────────────────────────────────────────────────────────

def _bbh_frames(
    gamma_x: float = 0.5,
    lambda_x: float = 1.0,
    gamma_y: float = 0.5,
    lambda_y: float = 1.0,
    n_k: int = 60,
    n_sites: int = 8,
    n_pd: int = 60,
    scenario: str = "both",
    source: str = "python-reference",
) -> dict:
    """NumPy reference: BBH quadrupole insulator bulk bands + corner states."""
    import numpy as np  # noqa: PLC0415

    PI = math.pi

    # ── Gamma matrices (4×4) ───────────────────────────────────────────────
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    s0 = np.eye(2, dtype=complex)

    G1 = np.kron(sy, sx)   # sigma_y ⊗ sigma_x
    G2 = np.kron(sx, s0)   # sigma_x ⊗ I
    G3 = np.kron(sy, sy)   # sigma_y ⊗ sigma_y
    G4 = np.kron(sy, sz)   # sigma_y ⊗ sigma_z

    def bbh_ham(kx: float, ky: float) -> np.ndarray:
        c1 = lambda_y * math.sin(ky)
        c2 = gamma_y  + lambda_y * math.cos(ky)
        c3 = lambda_x * math.sin(kx)
        c4 = gamma_x  + lambda_x * math.cos(kx)
        return c1*G1 + c2*G2 + c3*G3 + c4*G4

    out: dict = {
        "source": source,
        "gamma_x": gamma_x, "lambda_x": lambda_x,
        "gamma_y": gamma_y, "lambda_y": lambda_y,
    }
    topo = abs(gamma_x / lambda_x) < 1.0 and abs(gamma_y / lambda_y) < 1.0
    out["phase_label"] = "Quadrupole (q_xy=1/2)" if topo else "Trivial"

    # ── Bulk spectrum ─────────────────────────────────────────────────────
    if scenario in ("spectrum", "both"):
        kx_list, ky_list = [], []
        E0_list, E1_list, E2_list, E3_list = [], [], [], []
        for si in range(n_k):
            for ti in range(n_k):
                kx = 2*PI*si/n_k - PI
                ky = 2*PI*ti/n_k - PI
                H = bbh_ham(kx, ky)
                evals = np.linalg.eigvalsh(H)
                kx_list.append(float(kx)); ky_list.append(float(ky))
                E0_list.append(float(evals[0])); E1_list.append(float(evals[1]))
                E2_list.append(float(evals[2])); E3_list.append(float(evals[3]))
        gap = float(min(E2_list[i] - E1_list[i] for i in range(len(E0_list))))
        out.update({
            "kx_bulk": kx_list, "ky_bulk": ky_list,
            "E0": E0_list, "E1": E1_list, "E2": E2_list, "E3": E3_list,
            "gap": gap,
        })

    # ── Finite-lattice open-BC spectrum ───────────────────────────────────
    if scenario in ("corner", "both"):
        Ns = n_sites
        Norb = 4
        dim = Ns * Ns * Norb

        def orb_idx(ix: int, iy: int, o: int) -> int:
            return (iy * Ns + ix) * Norb + o

        H_full = np.zeros((dim, dim), dtype=complex)

        def add_block(ix: int, iy: int, jx: int, jy: int,
                      c: complex, G: np.ndarray) -> None:
            if not (0 <= ix < Ns and 0 <= iy < Ns): return
            if not (0 <= jx < Ns and 0 <= jy < Ns): return
            for a in range(Norb):
                for b in range(Norb):
                    H_full[orb_idx(ix,iy,a), orb_idx(jx,jy,b)] += c * G[a,b]

        for iy in range(Ns):
            for ix in range(Ns):
                # On-site
                add_block(ix, iy, ix, iy, gamma_x, G4)
                add_block(ix, iy, ix, iy, gamma_y, G2)
                # x-direction hops
                add_block(ix, iy, ix+1, iy, lambda_x * 0.5,     G4)
                add_block(ix, iy, ix+1, iy, -1j * lambda_x*0.5, G3)
                add_block(ix, iy, ix-1, iy, lambda_x * 0.5,     G4)
                add_block(ix, iy, ix-1, iy,  1j * lambda_x*0.5, G3)
                # y-direction hops
                add_block(ix, iy, ix, iy+1, lambda_y * 0.5,     G2)
                add_block(ix, iy, ix, iy+1, -1j * lambda_y*0.5, G1)
                add_block(ix, iy, ix, iy-1, lambda_y * 0.5,     G2)
                add_block(ix, iy, ix, iy-1,  1j * lambda_y*0.5, G1)

        evals = np.linalg.eigvalsh(H_full)
        # Sort 8 closest to zero
        near_zero = sorted(evals, key=abs)[:8]
        out["E_open"] = [float(e) for e in sorted(evals)]
        out["ldos_corner"] = [float(e) for e in near_zero]

    # ── Phase diagram ─────────────────────────────────────────────────────
    if scenario in ("phase_diagram", "both"):
        n = n_pd
        pd_rx, pd_ry, pd_topo = [], [], []
        for i in range(n):
            for j in range(n):
                rx = -1.5 + 3.0 * i / (n - 1)
                ry = -1.5 + 3.0 * j / (n - 1)
                pd_rx.append(float(rx)); pd_ry.append(float(ry))
                pd_topo.append(1 if (abs(rx) < 1.0 and abs(ry) < 1.0) else 0)
        out.update({"pd_gx_over_lx": pd_rx, "pd_gy_over_ly": pd_ry, "pd_topo": pd_topo})

    return out


def _bbh_page(data: dict) -> str:
    """Render BBH quadrupole insulator results as a self-contained HTML page."""
    payload = json.dumps(data, separators=(",", ":"))
    phase_label = data.get("phase_label", "")
    gap = data.get("gap", 0.0)
    gx = data.get("gamma_x", 0.5); lx = data.get("lambda_x", 1.0)
    gy = data.get("gamma_y", 0.5); ly = data.get("lambda_y", 1.0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>BBH Quadrupole Insulator — QuantX</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:16px}}
  h2{{color:#e6edf3;font-size:1.1rem;margin-bottom:12px}}
  .info{{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:14px;font-size:.82rem;color:#8b949e}}
  .info span b{{color:#c9d1d9}}
  .panels{{display:flex;gap:16px;flex-wrap:wrap}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}}
  canvas{{display:block}}
</style>
</head>
<body>
<h2>BBH Quadrupole Insulator (Higher-Order TI)</h2>
<div class="info">
  <span><b>gx</b>={gx:.3f}</span><span><b>lx</b>={lx:.3f}</span>
  <span><b>gy</b>={gy:.3f}</span><span><b>ly</b>={ly:.3f}</span>
  <span><b>gx/lx</b>={gx/lx:.3f}</span><span><b>gy/ly</b>={gy/ly:.3f}</span>
  <span><b>Phase</b>: {phase_label}</span>
  <span><b>Bulk gap</b>: {gap:.4f}</span>
</div>
<div class="panels">
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Open-BC Spectrum (corner states near E=0)</div>
    <canvas id="cOpen" width="420" height="320"></canvas>
  </div>
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Phase Diagram (gx/lx vs gy/ly)</div>
    <canvas id="cPhase" width="380" height="320"></canvas>
  </div>
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Bulk Bands (E vs kx)</div>
    <canvas id="cBulk" width="380" height="320"></canvas>
  </div>
</div>
<script>
(function(){{
'use strict';
const DATA={payload};

function drawAxes(ctx,pad,W,H,xTicks,yTicks){{
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,pad.t+(H-pad.t-pad.b));
  ctx.lineTo(pad.l+(W-pad.l-pad.r),pad.t+(H-pad.t-pad.b));ctx.stroke();
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';ctx.textAlign='center';
  xTicks.forEach(([x,lbl])=>ctx.fillText(lbl,x,H-pad.b+10));
  ctx.textAlign='right';
  yTicks.forEach(([y,lbl])=>ctx.fillText(lbl,pad.l-3,y+3));
}}

// ── Open-BC spectrum ─────────────────────────────────────────────────────
function drawOpen(){{
  const cv=document.getElementById('cOpen');
  const ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height;
  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
  const pad={{l:50,r:15,t:15,b:30}};
  const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;
  const ev=DATA.E_open||[];
  if(!ev.length){{ctx.fillStyle='#8b949e';ctx.font='12px monospace';ctx.textAlign='center';
    ctx.fillText('(run with scenario "corner" or "both")',W/2,H/2);return;}}
  const emin=ev[0], emax=ev[ev.length-1];
  const sy=e=>pad.t+(1-(e-emin)/(emax-emin||1))*ph;
  // Draw all eigenvalues as horizontal bars at index x
  const n=ev.length;
  const sx=i=>pad.l+i/(n-1)*pw;
  // Color: near zero → bright, others → dim
  ev.forEach((e,i)=>{{
    const norm=Math.abs(e)/(Math.max(Math.abs(emin),Math.abs(emax))||1);
    ctx.fillStyle=norm<0.15?'#f78166':'#8b949e';
    ctx.fillRect(Math.round(sx(i))-1,Math.round(sy(e))-1,2,2);
  }});
  // Zero line
  if(emin<0&&emax>0){{
    ctx.strokeStyle='rgba(255,255,255,0.2)';ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(pad.l,sy(0));ctx.lineTo(pad.l+pw,sy(0));ctx.stroke();
    ctx.setLineDash([]);
  }}
  drawAxes(ctx,pad,W,H,
    [[pad.l,'0'],[pad.l+pw/2,String(Math.round(n/2))],[pad.l+pw,String(n)]],
    [[sy(emax),emax.toFixed(2)],[sy(0),'0'],[sy(emin),emin.toFixed(2)]]
  );
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';ctx.textAlign='center';
  ctx.fillText('state index',W/2,H-2);
  ctx.save();ctx.translate(12,H/2);ctx.rotate(-Math.PI/2);ctx.fillText('E',0,0);ctx.restore();
  // Annotation
  const nc=DATA.ldos_corner||[];
  const nMid=nc.filter(e=>Math.abs(e)<0.15).length;
  ctx.fillStyle='#f78166';ctx.font='10px monospace';ctx.textAlign='left';
  ctx.fillText('corner states (red): '+nMid+' near E=0',pad.l+4,pad.t+12);
}}

// ── Phase diagram ────────────────────────────────────────────────────────
function drawPhase(){{
  const cv=document.getElementById('cPhase');
  const ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height;
  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
  const pad={{l:40,r:15,t:15,b:35}};
  const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;
  const rx=DATA.pd_gx_over_lx||[],ry=DATA.pd_gy_over_ly||[],tp=DATA.pd_topo||[];
  if(!rx.length){{ctx.fillStyle='#8b949e';ctx.font='12px monospace';ctx.textAlign='center';
    ctx.fillText('(run with scenario "phase_diagram" or "both")',W/2,H/2);return;}}
  const xmin=Math.min(...rx),xmax=Math.max(...rx);
  const ymin=Math.min(...ry),ymax=Math.max(...ry);
  const sx=v=>pad.l+(v-xmin)/(xmax-xmin)*pw;
  const sy=v=>pad.t+(1-(v-ymin)/(ymax-ymin))*ph;
  for(let i=0;i<rx.length;i++){{
    ctx.fillStyle=tp[i]?'#3fb950':'#21262d';
    ctx.fillRect(Math.round(sx(rx[i]))-2,Math.round(sy(ry[i]))-2,4,4);
  }}
  // Current point marker
  const cx2=sx(DATA.gamma_x/DATA.lambda_x), cy2=sy(DATA.gamma_y/DATA.lambda_y);
  ctx.strokeStyle='#fff';ctx.lineWidth=1.5;
  ctx.beginPath();ctx.arc(cx2,cy2,7,0,2*Math.PI);ctx.stroke();
  // Unit square (topo region)
  ctx.strokeStyle='rgba(255,255,255,0.15)';ctx.setLineDash([4,4]);
  ctx.strokeRect(sx(-1),sy(1),sx(1)-sx(-1),sy(-1)-sy(1));
  ctx.setLineDash([]);
  drawAxes(ctx,pad,W,H,
    [[sx(-1),'-1'],[sx(0),'0'],[sx(1),'1']],
    [[sy(1),'1'],[sy(0),'0'],[sy(-1),'-1']]
  );
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';ctx.textAlign='center';
  ctx.fillText('gx/lx',W/2,H-4);
  ctx.save();ctx.translate(10,H/2);ctx.rotate(-Math.PI/2);ctx.fillText('gy/ly',0,0);ctx.restore();
  ctx.fillStyle='#3fb950';ctx.fillRect(pad.l+2,pad.t+2,8,8);
  ctx.fillStyle='#8b949e';ctx.textAlign='left';ctx.fillText('Topo',pad.l+13,pad.t+9);
}}

// ── Bulk bands ────────────────────────────────────────────────────────────
function drawBulk(){{
  const cv=document.getElementById('cBulk');
  const ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height;
  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
  const pad={{l:45,r:15,t:15,b:30}};
  const pw=W-pad.l-pad.r, ph=H-pad.t-pad.b;
  const kx=DATA.kx_bulk||[],nk2=Math.round(Math.sqrt(kx.length||1));
  const E1=DATA.E1||[],E2=DATA.E2||[];
  if(!kx.length){{ctx.fillStyle='#8b949e';ctx.font='12px monospace';ctx.textAlign='center';
    ctx.fillText('(run with scenario "spectrum" or "both")',W/2,H/2);return;}}
  // Show kx slice at ky≈0 (ti = nk/2)
  const ti0=Math.floor(nk2/2);
  const e1s=[],e2s=[],kxs=[];
  for(let si=0;si<nk2;si++){{
    const idx=si*nk2+ti0;
    kxs.push(kx[idx]);e1s.push(E1[idx]);e2s.push(E2[idx]);
  }}
  const kmin=Math.min(...kxs),kmax=Math.max(...kxs);
  const emin=Math.min(...e1s),emax=Math.max(...e2s);
  const sxf=v=>pad.l+(v-kmin)/(kmax-kmin||1)*pw;
  const syf=e=>pad.t+(1-(e-emin)/(emax-emin||1))*ph;
  ctx.strokeStyle='#58a6ff';ctx.lineWidth=1.5;
  ctx.beginPath();kxs.forEach((v,i)=>i===0?ctx.moveTo(sxf(v),syf(e1s[i])):ctx.lineTo(sxf(v),syf(e1s[i])));ctx.stroke();
  ctx.strokeStyle='#3fb950';ctx.lineWidth=1.5;
  ctx.beginPath();kxs.forEach((v,i)=>i===0?ctx.moveTo(sxf(v),syf(e2s[i])):ctx.lineTo(sxf(v),syf(e2s[i])));ctx.stroke();
  if(emin<0&&emax>0){{ctx.strokeStyle='rgba(255,255,255,0.15)';ctx.setLineDash([3,3]);
    ctx.beginPath();ctx.moveTo(pad.l,syf(0));ctx.lineTo(pad.l+pw,syf(0));ctx.stroke();ctx.setLineDash([]);}}
  drawAxes(ctx,pad,W,H,
    [[sxf(-Math.PI),'-π'],[sxf(0),'0'],[sxf(Math.PI),'π']],
    [[syf(emax),emax.toFixed(2)],[syf(0),'0'],[syf(emin),emin.toFixed(2)]]
  );
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';ctx.textAlign='center';
  ctx.fillText('kx (ky=0)',W/2,H-2);
  ctx.fillStyle='#8b949e';ctx.font='10px monospace';ctx.textAlign='left';
  ctx.fillText('gap='+((DATA.gap||0).toFixed(4)),pad.l+4,pad.t+12);
}}

drawOpen();drawPhase();drawBulk();
window.addEventListener('resize',()=>{{drawOpen();drawPhase();drawBulk();}});
}})();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# M44  Floquet Topological Insulator
# ─────────────────────────────────────────────────────────────────────────────

def _floquet_topo_frames(
    J: float = 1.0,
    T: float = 1.0,
    n_k: int = 50,
    n_y: int = 30,
    n_J: int = 50,
    J_max: float = 2.5,
    scenario: str = "both",
    source: str = "python-reference",
) -> dict:
    """NumPy reference: Floquet topological insulator (5-step drive protocol)."""
    import numpy as np  # noqa: PLC0415

    PI = math.pi
    dt = T / 5.0

    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    s0 = np.eye(2, dtype=complex)

    def step_U(kphase: float, t: float) -> np.ndarray:
        """exp(-i t J (cos(kphase) sx - sin(kphase) sy))"""
        f = complex(J * math.cos(kphase), J * math.sin(kphase))
        H = f * (sx + 1j*sy) * 0.5 + f.conjugate() * (sx - 1j*sy) * 0.5
        # H = Re(f)*sx + Im(f)*sy  ← actually H = [[0,f],[f*,0]] for σx/σy combo
        # = J*cos*sx - J*sin*sy
        H2 = np.array([[0, f], [f.conjugate(), 0]], dtype=complex)
        # wait, let me be explicit:
        # H = J(cos kphase σx - sin kphase σy)
        # = [[0, J*cos + i*J*sin], [J*cos - i*J*sin, 0]]
        # = [[0, J e^{i kphase}], [J e^{-i kphase}, 0]]
        # eigenvalues ±J → exp(-i t H) = cos(tJ) I - i sin(tJ) H/J
        E = J if J > 1e-15 else 1e-15
        c = math.cos(t * E)
        s = math.sin(t * E) / E
        return c * s0 - 1j * s * np.array([[0, J*math.cos(kphase) + 1j*J*math.sin(kphase)],
                                             [J*math.cos(kphase) - 1j*J*math.sin(kphase), 0]])

    def floquet_U_bulk(kx: float, ky: float) -> np.ndarray:
        U1 = step_U(kx,  dt)  # +x hop
        U2 = step_U(ky,  dt)  # +y hop
        U3 = step_U(-kx, dt)  # -x hop
        U4 = step_U(-ky, dt)  # -y hop
        U5 = s0.copy()         # trivial step
        return U5 @ U4 @ U3 @ U2 @ U1

    def quasi_energy(U: np.ndarray) -> tuple:
        tr = (U[0,0] + U[1,1]).real * 0.5
        theta = math.acos(max(-1.0, min(1.0, tr)))
        return -theta / T, theta / T

    def winding_W_pi(J_val: float) -> int:
        phase_total = 0.0
        nk_w = n_k
        prev = None
        for i in range(nk_w + 1):
            kx = 2*PI * i / nk_w - PI
            U1t = step_U(kx,  dt) if J_val > 1e-15 else s0.copy()
            # rebuild with J_val
            def step_Jv(kp: float) -> np.ndarray:
                e = J_val if J_val > 1e-15 else 1e-15
                c2 = math.cos(dt * e); s2 = math.sin(dt * e) / e
                return c2 * s0 - 1j * s2 * np.array(
                    [[0, J_val*math.cos(kp) + 1j*J_val*math.sin(kp)],
                     [J_val*math.cos(kp) - 1j*J_val*math.sin(kp), 0]])
            U = step_Jv(-0.0) @ step_Jv(-0.0) @ step_Jv(-kx) @ step_Jv(0.0) @ step_Jv(kx)
            val = U[0, 1]
            if prev is not None:
                phase_total += float(np.angle(val * np.conj(prev)))
            prev = val
        return int(round(phase_total / (2 * PI)))

    out: dict = {"source": source, "J": J, "T": T}

    # Bulk spectrum
    if scenario in ("spectrum", "both"):
        kx_list, ky_list, em_list, ep_list = [], [], [], []
        for si in range(n_k):
            for ti in range(n_k):
                kx = 2*PI*si/n_k - PI
                ky = 2*PI*ti/n_k - PI
                U = floquet_U_bulk(kx, ky)
                em, ep = quasi_energy(U)
                kx_list.append(float(kx)); ky_list.append(float(ky))
                em_list.append(em); ep_list.append(ep)
        out.update({"kx_bulk": kx_list, "ky_bulk": ky_list,
                    "eps_minus": em_list, "eps_plus": ep_list})

    # Edge spectrum (strip)
    if scenario in ("edge", "both"):
        Ny = n_y; dim = 2 * Ny
        kx_edge, eps_edge = [], []
        for si in range(n_k):
            kx = 2*PI*si/n_k - PI
            kx_edge.append(float(kx))
            # Build dim×dim strip Floquet operator
            Ustrip = np.eye(dim, dtype=complex)
            # Step 1: x-hops, block diagonal
            U1b = step_U(kx, dt)
            S1 = np.zeros((dim, dim), dtype=complex)
            for iy in range(Ny):
                S1[2*iy:2*iy+2, 2*iy:2*iy+2] = U1b
            Ustrip = S1 @ Ustrip
            # Step 2: +y hops, pairs (2iy+1, 2iy+2) for iy=0,2,...
            S2 = np.eye(dim, dtype=complex)
            for iy in range(0, Ny-1, 2):
                blk = step_U(0.0, dt)  # σx block (kphase=0)
                bA = 2*iy+1; aB = 2*(iy+1)
                S2[bA, bA] = blk[0,0]; S2[bA, aB] = blk[0,1]
                S2[aB, bA] = blk[1,0]; S2[aB, aB] = blk[1,1]
            Ustrip = S2 @ Ustrip
            # Step 3: x-hops (-kx)
            U3b = step_U(-kx, dt)
            S3 = np.zeros((dim, dim), dtype=complex)
            for iy in range(Ny):
                S3[2*iy:2*iy+2, 2*iy:2*iy+2] = U3b
            Ustrip = S3 @ Ustrip
            # Step 4: -y hops, pairs (2iy+1, 2iy+2) for iy=1,3,...
            S4 = np.eye(dim, dtype=complex)
            for iy in range(1, Ny-1, 2):
                blk = step_U(0.0, dt)
                bB = 2*iy+1; aA = 2*(iy+1)
                S4[bB, bB] = blk[0,0]; S4[bB, aA] = blk[0,1]
                S4[aA, bB] = blk[1,0]; S4[aA, aA] = blk[1,1]
            Ustrip = S4 @ Ustrip
            # Step 5: identity
            # Quasienergies via joint diagonalisation of Msin and Mcos.
            # Msin and Mcos commute (same unitary eigenbasis), so eigvecs of
            # Msin simultaneously diagonalise Mcos; atan2 covers the full
            # range (-π/T, π/T] including π-gap edge modes.
            Msin = ((Ustrip - Ustrip.conj().T) / (2j)).real
            Mcos = ((Ustrip + Ustrip.conj().T) / 2.0).real
            evals_sin, V = np.linalg.eigh(Msin)
            evals_cos_diag = np.diag(V.T @ Mcos @ V).real
            eps_row = [float(np.arctan2(s, c) / T)
                       for s, c in zip(evals_sin, evals_cos_diag)]
            eps_edge.extend(sorted(eps_row))
        out.update({"kx_edge": kx_edge, "eps_edge": eps_edge})

    # Winding sweep
    if scenario in ("winding", "both"):
        w_pi = winding_W_pi(J)
        out["W_pi"] = w_pi
        out["W_0"]  = 0
        out["phase_label"] = (f"Anomalous Floquet (W_pi={w_pi:+d})"
                              if w_pi != 0 else "Trivial Floquet (W_pi=0)")
        sweep_J_list, sweep_W_list = [], []
        if n_J >= 2:
            for i in range(n_J):
                Jv = J_max * i / (n_J - 1)
                sweep_J_list.append(float(Jv))
                sweep_W_list.append(winding_W_pi(Jv))
        else:
            sweep_J_list = [float(J_max)]
            sweep_W_list = [winding_W_pi(J_max)]
        out.update({"sweep_J": sweep_J_list, "sweep_W_pi": sweep_W_list})
    else:
        out.setdefault("W_pi", 0)
        out.setdefault("phase_label", "Floquet")

    return out


def _floquet_topo_page(data: dict) -> str:
    """Render Floquet topological insulator results as a self-contained HTML page."""
    payload = json.dumps(data, separators=(",", ":"))
    phase_label = data.get("phase_label", "")
    J_val = data.get("J", 1.0)
    T_val = data.get("T", 1.0)
    W_pi = data.get("W_pi", 0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Floquet Topological Insulator — QuantX</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:16px}}
  h2{{color:#e6edf3;font-size:1.1rem;margin-bottom:12px}}
  .info{{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:14px;font-size:.82rem;color:#8b949e}}
  .info span b{{color:#c9d1d9}}
  .panels{{display:flex;gap:16px;flex-wrap:wrap}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}}
  canvas{{display:block}}
</style>
</head>
<body>
<h2>Floquet Topological Insulator (5-step drive protocol)</h2>
<div class="info">
  <span><b>J</b>={J_val:.3f}</span>
  <span><b>T</b>={T_val:.3f}</span>
  <span><b>W_pi</b>={W_pi:+d}</span>
  <span><b>Phase</b>: {phase_label}</span>
</div>
<div class="panels">
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Edge Spectrum (strip, open y)</div>
    <canvas id="cEdge" width="440" height="340"></canvas>
  </div>
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Winding Number W_pi vs J</div>
    <canvas id="cWind" width="380" height="340"></canvas>
  </div>
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Bulk Quasienergy (kx slice, ky=0)</div>
    <canvas id="cBulk" width="380" height="340"></canvas>
  </div>
</div>
<script>
(function(){{
'use strict';
const DATA={payload};
const PI=Math.PI;

function drawAxes(ctx,pad,W,H,xTicks,yTicks){{
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad.l,pad.t);ctx.lineTo(pad.l,H-pad.b);
  ctx.lineTo(W-pad.r,H-pad.b);ctx.stroke();
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';
  ctx.textAlign='center';xTicks.forEach(([x,l])=>ctx.fillText(l,x,H-pad.b+10));
  ctx.textAlign='right';yTicks.forEach(([y,l])=>ctx.fillText(l,pad.l-3,y+3));
}}

// ── Edge spectrum ──────────────────────────────────────────────────────────
function drawEdge(){{
  const cv=document.getElementById('cEdge');
  const ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height;
  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
  const pad={{l:50,r:15,t:15,b:30}};
  const pw=W-pad.l-pad.r,ph=H-pad.t-pad.b;
  const kx=DATA.kx_edge||[],eps=DATA.eps_edge||[];
  if(!kx.length){{ctx.fillStyle='#8b949e';ctx.font='12px monospace';ctx.textAlign='center';
    ctx.fillText('(run with scenario "edge" or "both")',W/2,H/2);return;}}
  const nk=kx.length, nb=eps.length/nk;
  const T2=DATA.T||1;
  const emax=PI/T2;
  const sxf=v=>pad.l+(v+PI)/(2*PI)*pw;
  const syf=e=>pad.t+(1-(e+emax)/(2*emax))*ph;
  // Draw all bands as grey dots
  ctx.fillStyle='#21262d';
  for(let si=0;si<nk;si++)for(let b=0;b<nb;b++){{
    const e=eps[si*nb+b];
    ctx.fillRect(Math.round(sxf(kx[si]))-1,Math.round(syf(e))-1,2,2);
  }}
  // Highlight near-zero (edge states in 0-gap) and near ±π/T (π-gap)
  const thr=0.15*emax;
  ctx.fillStyle='#f78166';
  for(let si=0;si<nk;si++)for(let b=0;b<nb;b++){{
    const e=eps[si*nb+b];
    if(Math.abs(e)<thr||Math.abs(Math.abs(e)-emax)<thr)
      ctx.fillRect(Math.round(sxf(kx[si]))-1,Math.round(syf(e))-1,2,2);
  }}
  // ±π/T lines
  ctx.strokeStyle='rgba(255,255,255,0.1)';ctx.setLineDash([3,3]);
  [0, PI/T2, -PI/T2].forEach(e=>{{
    ctx.beginPath();ctx.moveTo(pad.l,syf(e));ctx.lineTo(pad.l+pw,syf(e));ctx.stroke();
  }});
  ctx.setLineDash([]);
  drawAxes(ctx,pad,W,H,
    [[sxf(-PI),'-π'],[sxf(0),'0'],[sxf(PI),'π']],
    [[syf(emax),'π/T'],[syf(0),'0'],[syf(-emax),'-π/T']]
  );
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';ctx.textAlign='center';
  ctx.fillText('kx',W/2,H-2);
  ctx.fillStyle='#f78166';ctx.font='9px monospace';ctx.textAlign='left';
  ctx.fillText('edge states (red)',pad.l+4,pad.t+10);
}}

// ── Winding-number plot ────────────────────────────────────────────────────
function drawWind(){{
  const cv=document.getElementById('cWind');
  const ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height;
  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
  const pad={{l:40,r:15,t:15,b:30}};
  const pw=W-pad.l-pad.r,ph=H-pad.t-pad.b;
  const sJ=DATA.sweep_J||[],sW=DATA.sweep_W_pi||[];
  if(!sJ.length){{ctx.fillStyle='#8b949e';ctx.font='12px monospace';ctx.textAlign='center';
    ctx.fillText('(run with scenario "winding" or "both")',W/2,H/2);return;}}
  const Jmin=0,Jmax=sJ[sJ.length-1]||2.5;
  const Wmin=Math.min(-1,...sW),Wmax=Math.max(1,...sW);
  const sxf=v=>pad.l+(v-Jmin)/(Jmax-Jmin||1)*pw;
  const syf=v=>pad.t+(1-(v-Wmin)/(Wmax-Wmin||1))*ph;
  // Colour-coded step plot
  ctx.lineWidth=2;
  let prev=null;
  sJ.forEach((j,i)=>{{
    const w=sW[i];
    ctx.strokeStyle=w!==0?'#3fb950':'#8b949e';
    if(prev!==null){{
      ctx.beginPath();ctx.moveTo(sxf(sJ[i-1]),syf(prev));ctx.lineTo(sxf(j),syf(w));ctx.stroke();
    }}
    prev=w;
  }});
  // W=0 line
  ctx.strokeStyle='rgba(255,255,255,0.1)';ctx.setLineDash([3,3]);
  ctx.beginPath();ctx.moveTo(pad.l,syf(0));ctx.lineTo(pad.l+pw,syf(0));ctx.stroke();
  ctx.setLineDash([]);
  // Current J marker
  const Jcur=DATA.J||1;
  const Wcur=DATA.W_pi||0;
  ctx.fillStyle='#fff';
  ctx.beginPath();ctx.arc(sxf(Jcur),syf(Wcur),5,0,2*Math.PI);ctx.fill();
  drawAxes(ctx,pad,W,H,
    [[sxf(0),'0'],[sxf(Jmax/2),(Jmax/2).toFixed(1)],[sxf(Jmax),Jmax.toFixed(1)]],
    [[syf(Wmax),String(Wmax)],[syf(0),'0'],[syf(Wmin),String(Wmin)]]
  );
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';ctx.textAlign='center';
  ctx.fillText('J',W/2,H-2);
  ctx.save();ctx.translate(10,H/2);ctx.rotate(-Math.PI/2);ctx.fillText('W_pi',0,0);ctx.restore();
}}

// ── Bulk quasienergy (kx slice at ky≈0) ────────────────────────────────────
function drawBulk(){{
  const cv=document.getElementById('cBulk');
  const ctx=cv.getContext('2d');
  const W=cv.width,H=cv.height;
  ctx.fillStyle='#0d1117';ctx.fillRect(0,0,W,H);
  const pad={{l:50,r:15,t:15,b:30}};
  const pw=W-pad.l-pad.r,ph=H-pad.t-pad.b;
  const kx=DATA.kx_bulk||[];
  const nk2=Math.round(Math.sqrt(kx.length||1));
  const em=DATA.eps_minus||[],ep=DATA.eps_plus||[];
  if(!kx.length){{ctx.fillStyle='#8b949e';ctx.font='12px monospace';ctx.textAlign='center';
    ctx.fillText('(run with scenario "spectrum" or "both")',W/2,H/2);return;}}
  const ti0=Math.floor(nk2/2);
  const kxs=[],e1s=[],e2s=[];
  for(let si=0;si<nk2;si++){{
    const idx=si*nk2+ti0;
    kxs.push(kx[idx]);e1s.push(em[idx]);e2s.push(ep[idx]);
  }}
  const T2=DATA.T||1, emax=PI/T2;
  const sxf=v=>pad.l+(v+PI)/(2*PI)*pw;
  const syf=e=>pad.t+(1-(e+emax)/(2*emax))*ph;
  ctx.strokeStyle='#58a6ff';ctx.lineWidth=1.5;
  ctx.beginPath();kxs.forEach((v,i)=>i===0?ctx.moveTo(sxf(v),syf(e1s[i])):ctx.lineTo(sxf(v),syf(e1s[i])));ctx.stroke();
  ctx.strokeStyle='#3fb950';ctx.lineWidth=1.5;
  ctx.beginPath();kxs.forEach((v,i)=>i===0?ctx.moveTo(sxf(v),syf(e2s[i])):ctx.lineTo(sxf(v),syf(e2s[i])));ctx.stroke();
  // Gap lines
  ctx.strokeStyle='rgba(255,255,255,0.1)';ctx.setLineDash([3,3]);
  [0, PI/T2, -PI/T2].forEach(e=>{{ctx.beginPath();ctx.moveTo(pad.l,syf(e));ctx.lineTo(pad.l+pw,syf(e));ctx.stroke();}});
  ctx.setLineDash([]);
  drawAxes(ctx,pad,W,H,
    [[sxf(-PI),'-π'],[sxf(0),'0'],[sxf(PI),'π']],
    [[syf(emax),'π/T'],[syf(0),'0'],[syf(-emax),'-π/T']]
  );
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';ctx.textAlign='center';
  ctx.fillText('kx (ky=0)',W/2,H-2);
}}

drawEdge();drawWind();drawBulk();
window.addEventListener('resize',()=>{{drawEdge();drawWind();drawBulk();}});
}})();
</script>
</body>
</html>"""


# M45  Non-Hermitian Topology
# ─────────────────────────────────────────────────────────────────────────────

def _non_herm_frames(
    J: float = 1.0,
    gamma: float = 0.5,
    delta: float = 0.0,
    n_sites: int = 60,
    n_k: int = 100,
    n_gamma: int = 80,
    gamma_max: float = 2.0,
    scenario: str = "both",
    source: str = "python-reference",
) -> dict:
    """NumPy reference: Non-Hermitian topology (skin effect + exceptional points)."""
    import numpy as np  # noqa: PLC0415

    PI = math.pi
    J_R = J + gamma / 2.0
    J_L = J - gamma / 2.0

    def pbc_spectrum(J_val: float, gam: float) -> tuple:
        """E(k) = 2J cos(k) + i*gamma*sin(k) for k in [-pi, pi)."""
        k = np.linspace(-PI, PI, n_k, endpoint=False)
        e_re = 2.0 * J_val * np.cos(k)
        e_im = gam * np.sin(k)
        return k.tolist(), e_re.tolist(), e_im.tolist()

    def obc_spectrum(J_val: float, gam: float, N: int) -> list:
        """E_n = 2*sqrt(J_R*J_L)*cos(n*pi/(N+1)) for n=1..N (all real)."""
        jr = J_val + gam / 2.0
        jl = J_val - gam / 2.0
        e0 = 2.0 * math.sqrt(max(jr * jl, 0.0))
        ns = np.arange(1, N + 1)
        return sorted((e0 * np.cos(ns * PI / (N + 1))).tolist())

    def skin_density_profile(J_val: float, gam: float, N: int) -> tuple:
        """|psi_r|^2 for middle eigenstate, using stable log-space computation."""
        jr = J_val + gam / 2.0
        jl = J_val - gam / 2.0
        n_mid = N // 2 + 1
        theta = n_mid * PI / (N + 1)
        r = np.arange(1, N + 1, dtype=float)
        log_ratio = math.log(abs(jr / jl)) if abs(jl) > 1e-14 else 30.0
        log_rho = r * log_ratio + 2.0 * np.log(np.maximum(1e-300, np.abs(np.sin(theta * r))))
        log_rho -= log_rho.max()
        rho = np.exp(log_rho)
        rho /= rho.sum()
        return r.tolist(), rho.tolist()

    def winding(J_val: float, gam: float) -> int:
        """Winding number of E(k) around E=0.
        Special-case gamma=0 (Hermitian): contour is real, W=0.
        Without this guard, E(k) passes through the origin, producing
        a spurious ±2pi phase jump that rounds to W=±1."""
        if abs(gam) < 1e-12:
            return 0
        k = np.linspace(0, 2*PI, n_k, endpoint=False)
        e_re = 2.0 * J_val * np.cos(k)
        e_im = gam * np.sin(k)
        E = e_re + 1j * e_im
        total = float(np.sum(np.angle(E * np.conj(np.roll(E, -1)))))
        return int(round(total / (2 * PI)))

    def ep_sweep(J_val: float, delta_val: float) -> tuple:
        """E(gamma) = ±sqrt(delta^2 + J^2 - gamma^2); EP at gamma=sqrt(delta^2+J^2)."""
        gams = np.linspace(0.0, gamma_max, n_gamma)
        disc = delta_val**2 + J_val**2 - gams**2
        re = np.where(disc >= 0, np.sqrt(np.maximum(disc, 0.0)), 0.0)
        im = np.where(disc < 0, np.sqrt(np.maximum(-disc, 0.0)), 0.0)
        return (gams.tolist(),
                re.tolist(), im.tolist(),
                (-re).tolist(), (-im).tolist())

    out: dict = {"source": source, "J": J, "gamma": gamma, "delta": delta}

    if scenario in ("skin", "both"):
        k_pbc, e_re_pbc, e_im_pbc = pbc_spectrum(J, gamma)
        e_re_obc = obc_spectrum(J, gamma, n_sites)
        sites, density = skin_density_profile(J, gamma, n_sites)
        W = winding(J, gamma)
        # Winding sweep
        gams = np.linspace(0.0, gamma_max, n_gamma)
        sweep_W = [winding(J, float(g)) for g in gams]
        out.update({
            "k_pbc": k_pbc, "e_re_pbc": e_re_pbc, "e_im_pbc": e_im_pbc,
            "e_re_obc": e_re_obc,
            "skin_site": sites, "skin_density": density,
            "W": W,
            "sweep_gamma": gams.tolist(), "sweep_W": sweep_W,
            "phase_label": f"Skin (W={W:+d})" if W != 0 else "Trivial (W=0)",
        })

    if scenario in ("ep", "both"):
        ep_gams, ep_re_p, ep_im_p, ep_re_m, ep_im_m = ep_sweep(J, delta)
        out.update({
            "ep_gamma": ep_gams,
            "ep_e_re_plus": ep_re_p, "ep_e_im_plus": ep_im_p,
            "ep_e_re_minus": ep_re_m, "ep_e_im_minus": ep_im_m,
        })
        if "phase_label" not in out:
            out["phase_label"] = ("Below EP" if abs(gamma) < J else "Above EP")

    return out


def _non_herm_page(data: dict) -> str:
    """Render non-Hermitian topology results as a self-contained HTML page."""
    payload = json.dumps(data, separators=(",", ":"))
    J_val    = data.get("J", 1.0)
    gam_val  = data.get("gamma", 0.5)
    W_val    = data.get("W", 0)
    phase    = data.get("phase_label", "")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Non-Hermitian Topology — QuantX</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:16px}}
  h2{{color:#e6edf3;font-size:1.1rem;margin-bottom:12px}}
  .info{{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:14px;font-size:.82rem;color:#8b949e}}
  .info span b{{color:#c9d1d9}}
  .panels{{display:flex;gap:16px;flex-wrap:wrap}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}}
  canvas{{display:block}}
</style>
</head>
<body>
<h2>Non-Hermitian Topology: Skin Effect &amp; Exceptional Points</h2>
<div class="info">
  <span><b>J</b>={J_val:.3f}</span>
  <span><b>&#947;</b>={gam_val:.3f}</span>
  <span><b>W</b>={W_val:+d}</span>
  <span><b>Phase</b>: {phase}</span>
</div>
<div class="panels">
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Complex Energy Plane (PBC ellipse + OBC eigenvalues)</div>
    <canvas id="cComplex" width="400" height="360"></canvas>
  </div>
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Skin Density |&#968;|&#178; vs Site</div>
    <canvas id="cSkin" width="360" height="360"></canvas>
  </div>
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Exceptional Point: E vs &#947; (&#948;=0)</div>
    <canvas id="cEP" width="360" height="360"></canvas>
  </div>
</div>
<script>
(function(){{
const DATA={payload};
const W={W_val};const J={J_val};const DELTA={data.get("delta",0.0)};
const EP_GAMMA=Math.sqrt(DELTA*DELTA+J*J);

// ── Panel 1: Complex energy plane ──────────────────────────────────────────
function drawComplex(){{
  const cv=document.getElementById('cComplex');
  const W2=cv.width,H2=cv.height,pad=44;
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,W2,H2);
  const re_pbc=DATA.e_re_pbc||[],im_pbc=DATA.e_im_pbc||[];
  const re_obc=DATA.e_re_obc||[];
  if(re_pbc.length===0)return;
  // Ranges
  var rmin=Infinity,rmax=-Infinity,imin=Infinity,imax=-Infinity;
  for(var i=0;i<re_pbc.length;i++){{
    if(re_pbc[i]<rmin)rmin=re_pbc[i]; if(re_pbc[i]>rmax)rmax=re_pbc[i];
    if(im_pbc[i]<imin)imin=im_pbc[i]; if(im_pbc[i]>imax)imax=im_pbc[i];
  }}
  // Add 15% margin
  var dr=(rmax-rmin)||1,di=(imax-imin)||1;
  rmin-=dr*0.15; rmax+=dr*0.15; imin-=di*0.25; imax+=di*0.25;
  const toX=r=>(pad+(r-rmin)/(rmax-rmin)*(W2-2*pad));
  const toY=i=>(H2-pad-(i-imin)/(imax-imin)*(H2-2*pad));
  // Axes
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  const x0=toX(0),y0=toY(0);
  ctx.beginPath(); ctx.moveTo(pad,y0); ctx.lineTo(W2-pad,y0); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x0,pad); ctx.lineTo(x0,H2-pad); ctx.stroke();
  ctx.fillStyle='#8b949e'; ctx.font='9px monospace';
  ctx.fillText('Re(E)',W2-pad-20,y0-4);
  ctx.fillText('Im(E)',x0+4,pad+10);
  // PBC ellipse (gradient k→hue)
  ctx.lineWidth=1.5;
  for(var i=0;i<re_pbc.length-1;i++){{
    const hue=Math.round(240*i/re_pbc.length);
    ctx.strokeStyle=`hsl(${{hue}},80%,55%)`;
    ctx.beginPath();
    ctx.moveTo(toX(re_pbc[i]),toY(im_pbc[i]));
    ctx.lineTo(toX(re_pbc[i+1]),toY(im_pbc[i+1]));
    ctx.stroke();
  }}
  // OBC eigenvalues (dots on real axis)
  ctx.fillStyle='#f0883e';
  for(var i=0;i<re_obc.length;i++){{
    ctx.beginPath(); ctx.arc(toX(re_obc[i]),toY(0),2,0,2*Math.PI); ctx.fill();
  }}
  // Origin dot
  ctx.fillStyle='#ff7b72'; ctx.beginPath(); ctx.arc(x0,y0,4,0,2*Math.PI); ctx.fill();
  ctx.fillStyle='#c9d1d9'; ctx.font='9px monospace';
  ctx.fillText('PBC',pad+4,pad+12);
  ctx.fillStyle='#f0883e'; ctx.fillText('OBC',pad+4,pad+22);
}}

// ── Panel 2: Skin density ─────────────────────────────────────────────────
function drawSkin(){{
  const cv=document.getElementById('cSkin');
  const W2=cv.width,H2=cv.height,pad=44;
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,W2,H2);
  const sites=DATA.skin_site||[],dens=DATA.skin_density||[];
  if(sites.length===0)return;
  const N=sites.length;
  const dmax=Math.max(...dens)||1;
  const toX=r=>(pad+(r-1)/(N-1)*(W2-2*pad));
  const toY=d=>(H2-pad-d/dmax*(H2-2*pad));
  // Fill
  ctx.fillStyle='rgba(56,139,253,0.15)';
  ctx.beginPath(); ctx.moveTo(toX(1),H2-pad);
  for(var i=0;i<N;i++) ctx.lineTo(toX(sites[i]),toY(dens[i]));
  ctx.lineTo(toX(N),H2-pad); ctx.closePath(); ctx.fill();
  // Line
  ctx.strokeStyle='#388bfd'; ctx.lineWidth=1.5;
  ctx.beginPath();
  for(var i=0;i<N;i++){{
    if(i===0)ctx.moveTo(toX(sites[i]),toY(dens[i]));
    else ctx.lineTo(toX(sites[i]),toY(dens[i]));
  }}
  ctx.stroke();
  // Axes
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,H2-pad); ctx.lineTo(W2-pad,H2-pad); ctx.stroke();
  ctx.fillStyle='#8b949e'; ctx.font='9px monospace';
  ctx.fillText('Site',W2/2-10,H2-8);
  ctx.fillText('|psi|^2',4,pad+4);
  ctx.fillText('1',pad-6,H2-pad+12);
  ctx.fillText(String(N),W2-pad-8,H2-pad+12);
}}

// ── Panel 3: Exceptional point ────────────────────────────────────────────
function drawEP(){{
  const cv=document.getElementById('cEP');
  const W2=cv.width,H2=cv.height,pad=44;
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,W2,H2);
  const gams=DATA.ep_gamma||[];
  const rep=DATA.ep_e_re_plus||[],imp=DATA.ep_e_im_plus||[];
  const rem=DATA.ep_e_re_minus||[],imm=DATA.ep_e_im_minus||[];
  if(gams.length===0)return;
  const N=gams.length;
  const gmax=gams[N-1]||2;
  // Collect all values for range
  const allvals=[...rep,...rem,...imp,...imm];
  var vmin=Math.min(...allvals),vmax=Math.max(...allvals);
  if(vmax-vmin<1e-10){{vmin-=0.1;vmax+=0.1;}}
  vmin-=(vmax-vmin)*0.08; vmax+=(vmax-vmin)*0.08;
  const toX=g=>(pad+g/gmax*(W2-2*pad));
  const toY=v=>(H2-pad-(v-vmin)/(vmax-vmin)*(H2-2*pad));
  // Zero axis
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,toY(0)); ctx.lineTo(W2-pad,toY(0)); ctx.stroke();
  // EP vertical line at gamma=sqrt(delta^2+J^2)
  const xep=toX(EP_GAMMA);
  ctx.strokeStyle='#e3b341'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(xep,pad); ctx.lineTo(xep,H2-pad); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#e3b341'; ctx.font='9px monospace';
  ctx.fillText('EP',xep+2,pad+10);
  // Re(E) branches — solid
  function drawLine(vals, color){{
    ctx.strokeStyle=color; ctx.lineWidth=1.5;
    ctx.beginPath();
    for(var i=0;i<N;i++){{
      if(i===0)ctx.moveTo(toX(gams[i]),toY(vals[i]));
      else ctx.lineTo(toX(gams[i]),toY(vals[i]));
    }}
    ctx.stroke();
  }}
  drawLine(rep,'#79c0ff');   // Re(E+)
  drawLine(rem,'#79c0ff');   // Re(E-)
  // Im(E) branches — dashed
  ctx.setLineDash([4,3]);
  drawLine(imp,'#ff7b72');   // Im(E+)
  drawLine(imm,'#ff7b72');   // Im(E-)
  ctx.setLineDash([]);
  // Axes
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,H2-pad); ctx.lineTo(W2-pad,H2-pad); ctx.stroke();
  ctx.fillStyle='#8b949e'; ctx.font='9px monospace';
  ctx.fillText('gamma',W2/2-14,H2-8);
  ctx.fillText('E',4,pad+4);
  ctx.fillText('0',pad-10,H2-pad+4);
  ctx.fillText(gmax.toFixed(1),W2-pad-16,H2-pad+12);
  ctx.fillStyle='#79c0ff'; ctx.fillText('Re(E)',W2-pad-34,pad+12);
  ctx.fillStyle='#ff7b72'; ctx.fillText('Im(E)',W2-pad-34,pad+24);
}}

drawComplex();drawSkin();drawEP();
window.addEventListener('resize',()=>{{drawComplex();drawSkin();drawEP();}});
}})();
</script>
</body>
</html>"""


# M46  Aubry-André-Harper Quasicrystal
# ─────────────────────────────────────────────────────────────────────────────

def _aah_frames(
    t: float = 1.0,
    V: float = 0.5,
    beta: float = 0.0,
    phi: float = 0.0,
    n_sites: int = 89,
    n_V: int = 60,
    V_max: float = 2.0,
    scenario: str = "both",
    source: str = "python-reference",
) -> dict:
    """NumPy reference: Aubry-André-Harper quasicrystal (metal-insulator transition)."""
    import numpy as np  # noqa: PLC0415

    GOLDEN = (math.sqrt(5) - 1) / 2  # ≈ 0.6180339887
    beta_used = GOLDEN if beta == 0.0 else beta

    def build_H(V_val: float) -> np.ndarray:
        n = np.arange(n_sites, dtype=float)
        diag = 2.0 * V_val * np.cos(2 * math.pi * beta_used * n + phi)
        off  = np.full(n_sites - 1, t)
        return np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)

    def ipr_from_evecs(evecs: np.ndarray) -> np.ndarray:
        """IPR = sum_n |psi_n|^4 for each eigenstate (column)."""
        return np.sum(evecs**4, axis=0)

    out: dict = {"source": source, "t": t, "V": V, "beta_used": beta_used}

    if scenario in ("spectrum", "both"):
        H = build_H(V)
        evals, evecs = np.linalg.eigh(H)
        ipr = ipr_from_evecs(evecs)
        mean_ipr = float(np.mean(ipr))
        lyapunov = float(max(0.0, math.log(V / t))) if t > 0 else 0.0
        eps = 0.05
        if abs(V - t) < eps * t:
            phase_label = "Critical (V=t)"
        elif V < t:
            phase_label = "Metal (V<t)"
        else:
            phase_label = "Insulator (V>t)"
        out.update({
            "eigenvalues": evals.tolist(),
            "ipr": ipr.tolist(),
            "mean_ipr": mean_ipr,
            "lyapunov": lyapunov,
            "phase_label": phase_label,
        })

    if scenario in ("phase", "both"):
        pd_V, pd_E, pd_ipr_mean = [], [], []
        Vs = np.linspace(0.0, V_max, n_V) if n_V > 1 else np.array([V_max])
        for Vv in Vs:
            H = build_H(float(Vv))
            evals, evecs = np.linalg.eigh(H)
            ipr = ipr_from_evecs(evecs)
            pd_V.append(float(Vv))
            pd_E.extend(sorted(evals.tolist()))
            pd_ipr_mean.append(float(np.mean(ipr)))
        out.update({
            "pd_V": pd_V,
            "pd_E": pd_E,
            "pd_ipr_mean": pd_ipr_mean,
        })
        if "phase_label" not in out:
            out["phase_label"] = "AAH phase diagram"

    return out


def _aah_page(data: dict) -> str:
    """Render AAH quasicrystal results as a self-contained HTML page."""
    payload = json.dumps(data, separators=(",", ":"))
    t_val   = data.get("t", 1.0)
    V_val   = data.get("V", 0.5)
    beta_v  = data.get("beta_used", 0.618)
    phase   = data.get("phase_label", "")
    lya     = data.get("lyapunov", 0.0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Aubry-André-Harper Quasicrystal — QuantX</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:16px}}
  h2{{color:#e6edf3;font-size:1.1rem;margin-bottom:12px}}
  .info{{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:14px;font-size:.82rem;color:#8b949e}}
  .info span b{{color:#c9d1d9}}
  .panels{{display:flex;gap:16px;flex-wrap:wrap}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}}
  canvas{{display:block}}
</style>
</head>
<body>
<h2>Aubry-André-Harper Quasicrystal: Metal-Insulator Transition</h2>
<div class="info">
  <span><b>t</b>={t_val:.3f}</span>
  <span><b>V</b>={V_val:.3f}</span>
  <span><b>&#946;</b>={beta_v:.4f}</span>
  <span><b>&#947;</b>={lya:.3f}</span>
  <span><b>Phase</b>: {phase}</span>
</div>
<div class="panels">
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Eigenvalue Spectrum (fixed V, coloured by IPR)</div>
    <canvas id="cSpec" width="400" height="340"></canvas>
  </div>
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">IPR vs Eigenstate Index</div>
    <canvas id="cIPR" width="360" height="340"></canvas>
  </div>
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Fractal Butterfly: Spectrum vs V</div>
    <canvas id="cButt" width="400" height="340"></canvas>
  </div>
</div>
<script>
(function(){{
const DATA={payload};
const t={t_val};const V_cur={V_val};

// ── Panel 1: Spectrum coloured by IPR ────────────────────────────────────────
function drawSpec(){{
  const cv=document.getElementById('cSpec');
  const W=cv.width,H=cv.height,pad=44;
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,H);
  const evals=DATA.eigenvalues||[],ipr=DATA.ipr||[];
  if(evals.length===0)return;
  const N=evals.length;
  const emin=Math.min(...evals),emax=Math.max(...evals);
  const er=emax-emin||1;
  const imax=Math.max(...ipr)||1;
  const toX=i=>(pad+(i/N)*(W-2*pad));
  const toY=e=>(H-pad-(e-emin)/er*(H-2*pad));
  // Axes
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
  // Dots coloured by IPR (low=blue=extended, high=red=localised)
  for(var i=0;i<N;i++){{
    const ipr_n=ipr[i]/imax;
    const r=Math.round(255*ipr_n),b=Math.round(255*(1-ipr_n));
    ctx.fillStyle=`rgb(${{r}},80,${{b}})`;
    ctx.beginPath(); ctx.arc(toX(i),toY(evals[i]),2.5,0,2*Math.PI); ctx.fill();
  }}
  // Critical line at V=t
  const ycrit=H/2;
  ctx.fillStyle='#8b949e'; ctx.font='9px monospace';
  ctx.fillText('E',4,pad+4);
  ctx.fillText('index',W/2-12,H-8);
  ctx.fillStyle='#388bfd'; ctx.fillText('IPR low=extended',pad+4,pad+12);
  ctx.fillStyle='#ff7b72'; ctx.fillText('IPR high=localised',pad+4,pad+22);
}}

// ── Panel 2: IPR vs eigenstate index ─────────────────────────────────────────
function drawIPR(){{
  const cv=document.getElementById('cIPR');
  const W=cv.width,H=cv.height,pad=44;
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,H);
  const ipr=DATA.ipr||[];
  if(ipr.length===0)return;
  const N=ipr.length;
  const imax=Math.max(...ipr)||1;
  const toX=i=>(pad+(i/N)*(W-2*pad));
  const toY=v=>(H-pad-(v/imax)*(H-2*pad));
  // Fill
  ctx.fillStyle='rgba(240,136,62,0.15)';
  ctx.beginPath(); ctx.moveTo(toX(0),H-pad);
  for(var i=0;i<N;i++) ctx.lineTo(toX(i),toY(ipr[i]));
  ctx.lineTo(toX(N-1),H-pad); ctx.closePath(); ctx.fill();
  ctx.strokeStyle='#f0883e'; ctx.lineWidth=1;
  ctx.beginPath();
  for(var i=0;i<N;i++){{
    if(i===0) ctx.moveTo(toX(i),toY(ipr[i]));
    else ctx.lineTo(toX(i),toY(ipr[i]));
  }}
  ctx.stroke();
  // Mean IPR line
  const mean=DATA.mean_ipr||0;
  ctx.strokeStyle='#e3b341'; ctx.lineWidth=1; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(pad,toY(mean)); ctx.lineTo(W-pad,toY(mean)); ctx.stroke();
  ctx.setLineDash([]);
  // Axes
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
  ctx.fillStyle='#8b949e'; ctx.font='9px monospace';
  ctx.fillText('IPR',4,pad+4);
  ctx.fillText('index',W/2-12,H-8);
  ctx.fillStyle='#e3b341';
  ctx.fillText('mean='+mean.toFixed(3),pad+4,pad+12);
}}

// ── Panel 3: Fractal butterfly (spectrum vs V) ────────────────────────────────
function drawButt(){{
  const cv=document.getElementById('cButt');
  const W=cv.width,H=cv.height,pad=44;
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,H);
  const pdV=DATA.pd_V||[],pdE=DATA.pd_E||[];
  if(pdV.length===0)return;
  const nV=pdV.length;
  const N=pdE.length/nV|0;
  const vmax=pdV[nV-1]||2;
  // Range of E
  var emin=Infinity,emax=-Infinity;
  for(var i=0;i<pdE.length;i++){{if(pdE[i]<emin)emin=pdE[i];if(pdE[i]>emax)emax=pdE[i];}}
  const er=emax-emin||1;
  const toX=v=>(pad+v/vmax*(W-2*pad));
  const toY=e=>(H-pad-(e-emin)/er*(H-2*pad));
  // Draw dots
  ctx.fillStyle='rgba(56,139,253,0.7)';
  for(var vi=0;vi<nV;vi++){{
    const x=toX(pdV[vi]);
    for(var i=0;i<N;i++){{
      ctx.fillRect(x,toY(pdE[vi*N+i]),1,1);
    }}
  }}
  // MIT line at V=t
  const xmit=toX(t);
  ctx.strokeStyle='#e3b341'; ctx.lineWidth=1; ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(xmit,pad); ctx.lineTo(xmit,H-pad); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle='#e3b341'; ctx.font='9px monospace';
  ctx.fillText('MIT',xmit+2,pad+10);
  // Current V marker
  const xcur=toX(V_cur);
  ctx.strokeStyle='#ff7b72'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(xcur,pad); ctx.lineTo(xcur,H-pad); ctx.stroke();
  // Axes
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad,pad); ctx.lineTo(pad,H-pad); ctx.lineTo(W-pad,H-pad); ctx.stroke();
  ctx.fillStyle='#8b949e'; ctx.font='9px monospace';
  ctx.fillText('E',4,pad+4);
  ctx.fillText('V',W/2-4,H-8);
  ctx.fillText('0',pad-8,H-pad+12);
  ctx.fillText(vmax.toFixed(1),W-pad-12,H-pad+12);
}}

drawSpec();drawIPR();drawButt();
window.addEventListener('resize',()=>{{drawSpec();drawIPR();drawButt();}});
}})();
</script>
</body>
</html>"""


def _syk_frames(
    N: int = 12,
    J: float = 1.0,
    beta: float = 1.0,
    n_samples: int = 20,
    n_t: int = 100,
    t_max: float = 30.0,
    seed: int = 42,
    scenario: str = "both",
    source: str = "python-reference",
) -> dict:
    """NumPy reference: Sachdev-Ye-Kitaev model (quantum chaos, SFF)."""
    import numpy as np  # noqa: PLC0415

    M = N // 2
    dim = 1 << M
    sigma = J * math.sqrt(6.0) / N**1.5

    def apply_gamma(j: int, alpha: int):
        m = j // 2
        odd = j & 1
        mask_m = (1 << m) - 1
        sign = -1.0 if bin(alpha & mask_m).count("1") & 1 else 1.0
        new_state = alpha ^ (1 << m)
        if not odd:
            phase = complex(sign, 0.0)
        else:
            fac = (1 - 2 * ((alpha >> m) & 1)) * sign
            phase = complex(0.0, fac)
        return phase, new_state

    def build_H(seed_local: int):
        rng_l = np.random.default_rng(seed_local)
        H = np.zeros((dim, dim), dtype=complex)
        for i in range(N - 3):
            for j in range(i + 1, N - 2):
                for k in range(j + 1, N - 1):
                    for l in range(k + 1, N):
                        Jijkl = rng_l.normal(0.0, sigma)
                        for alpha in range(dim):
                            p4, s4 = apply_gamma(l, alpha)
                            p3, s3 = apply_gamma(k, s4)
                            p2, s2 = apply_gamma(j, s3)
                            p1, s1 = apply_gamma(i, s2)
                            H[s1, alpha] += Jijkl * (p1 * p2 * p3 * p4)
        return H

    def r_stat(ev):
        r = []
        for i in range(1, len(ev) - 1):
            d1, d2 = float(ev[i] - ev[i - 1]), float(ev[i + 1] - ev[i])
            if d1 > 1e-14 and d2 > 1e-14:
                r.append(min(d1, d2) / max(d1, d2))
        return r

    out: dict = {"source": source, "N": N, "dim": dim, "J": J, "beta": beta}

    if scenario in ("spectrum", "both"):
        H = build_H(seed)
        ev = np.sort(np.linalg.eigvalsh(H))
        r_vals = r_stat(ev)
        r_mean = float(np.mean(r_vals)) if r_vals else 0.0
        out.update({
            "eigenvalues": ev.tolist(),
            "E0_per_N": float(ev[0]) / N,
            "r_vals": r_vals,
            "r_mean": r_mean,
        })

    if scenario in ("sff", "both"):
        t_grid = [t_max * (ti + 1) / n_t for ti in range(n_t)]
        K_sff = [0.0] * n_t
        for s in range(n_samples):
            H_s = build_H(seed + s * 1000)
            ev_s = np.sort(np.linalg.eigvalsh(H_s))
            w = np.exp(-beta * ev_s)
            z_beta = float(np.sum(w))
            if z_beta == 0.0:
                continue
            for ti, t_val in enumerate(t_grid):
                re_z = float(np.dot(w, np.cos(t_val * ev_s)))
                im_z = float(np.dot(w, np.sin(t_val * ev_s)))
                K_sff[ti] += (re_z**2 + im_z**2) / (z_beta**2)
        K_sff = [k / n_samples for k in K_sff]
        out.update({"t_sff": t_grid, "K_sff": K_sff})

    return out


def _syk_page(data: dict) -> str:
    """Render SYK model results as a self-contained HTML page."""
    payload = json.dumps(data, separators=(",", ":"))
    N_val   = data.get("N", 12)
    dim_val = data.get("dim", 64)
    J_val   = data.get("J", 1.0)
    beta_v  = data.get("beta", 1.0)
    r_mean  = data.get("r_mean", 0.0)
    E0_N    = data.get("E0_per_N", 0.0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Sachdev-Ye-Kitaev Model \u2014 QuantX</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:16px}}
  h2{{color:#e6edf3;font-size:1.1rem;margin-bottom:12px}}
  .info{{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:14px;font-size:.82rem;color:#8b949e}}
  .info span b{{color:#c9d1d9}}
  .panels{{display:flex;gap:16px;flex-wrap:wrap}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}}
  canvas{{display:block}}
</style>
</head>
<body>
<h2>Sachdev-Ye-Kitaev Model: Quantum Chaos &amp; Spectral Statistics</h2>
<div class="info">
  <span><b>N</b>={N_val}</span>
  <span><b>dim</b>={dim_val}</span>
  <span><b>J</b>={J_val:.3f}</span>
  <span><b>&#946;</b>={beta_v:.2f}</span>
  <span><b>&#x27e8;r&#x27e9;</b>={r_mean:.4f}</span>
  <span><b>E&#x2080;/N</b>={E0_N:.4f}</span>
</div>
<div class="panels">
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Eigenvalue Spectrum (coloured by index)</div>
    <canvas id="cSpec" width="380" height="340"></canvas>
  </div>
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Level-Spacing Ratio &#x27e8;r&#x27e9; vs GUE/Poisson</div>
    <canvas id="cHist" width="360" height="340"></canvas>
  </div>
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Spectral Form Factor K(t) &#x2014; disorder average</div>
    <canvas id="cSFF" width="400" height="340"></canvas>
  </div>
</div>
<script>
(function(){{
const DATA={payload};

function drawSpec(){{
  const cv=document.getElementById('cSpec');
  if(!cv)return;
  const W=cv.width,H=cv.height,pad=44;
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,H);
  const evals=DATA.eigenvalues||[];
  if(!evals.length){{ctx.fillStyle='#8b949e';ctx.font='11px monospace';ctx.fillText('(spectrum not computed)',pad,H/2);return;}}
  const n=evals.length,emin=evals[0],emax=evals[n-1],er=emax-emin||1;
  const toX=i=>(pad+(i/(n-1||1))*(W-2*pad));
  const toY=e=>(H-pad-(e-emin)/er*(H-2*pad));
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(pad,H-pad);ctx.lineTo(W-pad,H-pad);ctx.stroke();
  for(let i=0;i<n;i++){{
    const t01=i/(n-1||1);
    const r=Math.round(255*Math.min(1,2*t01));
    const b=Math.round(255*Math.min(1,2*(1-t01)));
    const g=Math.round(255*(1-Math.abs(2*t01-1)));
    ctx.fillStyle=`rgb(${{r}},${{g}},${{b}})`;
    ctx.fillRect(toX(i)-1,toY(evals[i])-1,2,2);
  }}
  ctx.fillStyle='#e3b341';ctx.font='9px monospace';
  ctx.fillText('E\u2080/N='+(DATA.E0_per_N||0).toFixed(3),pad+2,pad+12);
  ctx.fillStyle='#8b949e';ctx.fillText('E',4,pad+4);ctx.fillText('index',W/2-12,H-8);
}}

function drawHist(){{
  const cv=document.getElementById('cHist');
  if(!cv)return;
  const W=cv.width,H=cv.height,pad=44;
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,H);
  const r_vals=DATA.r_vals||[];
  if(!r_vals.length){{ctx.fillStyle='#8b949e';ctx.font='11px monospace';ctx.fillText('(spectrum not computed)',pad,H/2);return;}}
  const nBins=20,bins=new Array(nBins).fill(0);
  for(const rv of r_vals){{const bi=Math.min(nBins-1,Math.floor(rv*nBins));bins[bi]++;}}
  const bmax=Math.max(...bins)||1;
  const bw=(W-2*pad)/nBins-1;
  const toX=b=>(pad+b/nBins*(W-2*pad));
  const toY=v=>(H-pad-(v/bmax)*(H-2*pad));
  ctx.fillStyle='rgba(56,139,253,0.7)';
  for(let b=0;b<nBins;b++){{const x=toX(b);ctx.fillRect(x,toY(bins[b]),bw,H-pad-toY(bins[b]));}}
  const xGUE=toX(0.60*nBins);
  ctx.strokeStyle='#f0883e';ctx.lineWidth=1.5;ctx.setLineDash([4,3]);
  ctx.beginPath();ctx.moveTo(xGUE,pad);ctx.lineTo(xGUE,H-pad);ctx.stroke();
  const xPoi=toX(0.39*nBins);
  ctx.strokeStyle='#3fb950';ctx.lineWidth=1.5;
  ctx.beginPath();ctx.moveTo(xPoi,pad);ctx.lineTo(xPoi,H-pad);ctx.stroke();
  ctx.setLineDash([]);
  const r_mean=DATA.r_mean||0,xm=toX(r_mean*nBins);
  ctx.strokeStyle='#e3b341';ctx.lineWidth=2;
  ctx.beginPath();ctx.moveTo(xm,pad);ctx.lineTo(xm,H-pad);ctx.stroke();
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(pad,H-pad);ctx.lineTo(W-pad,H-pad);ctx.stroke();
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';
  ctx.fillText('count',4,pad+4);ctx.fillText('r',W/2-4,H-8);
  ctx.fillText('0',pad-8,H-pad+12);ctx.fillText('1',W-pad-4,H-pad+12);
  ctx.fillStyle='#f0883e';ctx.fillText('GUE 0.60',pad+4,pad+12);
  ctx.fillStyle='#3fb950';ctx.fillText('Poisson 0.39',pad+4,pad+22);
  ctx.fillStyle='#e3b341';ctx.fillText('\u27e8r\u27e9='+r_mean.toFixed(3),pad+4,pad+32);
}}

function drawSFF(){{
  const cv=document.getElementById('cSFF');
  if(!cv)return;
  const W=cv.width,H=cv.height,pad=48;
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,H);
  const t_sff=DATA.t_sff||[],K_sff=DATA.K_sff||[];
  if(!t_sff.length){{ctx.fillStyle='#8b949e';ctx.font='11px monospace';ctx.fillText('(SFF not computed)',pad,H/2);return;}}
  const n=t_sff.length,tmin=t_sff[0],tmax=t_sff[n-1];
  const logT0=Math.log10(tmin),logT1=Math.log10(tmax);
  let kmin=Infinity,kmax=-Infinity;
  for(let ki=0;ki<n;ki++){{if(K_sff[ki]>0){{kmin=Math.min(kmin,K_sff[ki]);kmax=Math.max(kmax,K_sff[ki]);}}}}
  if(!isFinite(kmin)){{kmin=1e-4;kmax=1;}}
  const logK0=Math.log10(kmin*0.7),logK1=Math.log10(kmax*2);
  const toX=t=>(pad+(Math.log10(t)-logT0)/(logT1-logT0)*(W-2*pad));
  const toY=k=>(k>0?H-pad-(Math.log10(k)-logK0)/(logK1-logK0)*(H-2*pad):H-pad);
  ctx.strokeStyle='#21262d';ctx.lineWidth=1;
  for(let d=Math.ceil(logK0);d<=Math.floor(logK1);d++){{
    const y=H-pad-(d-logK0)/(logK1-logK0)*(H-2*pad);
    ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(W-pad,y);ctx.stroke();
  }}
  ctx.strokeStyle='#388bfd';ctx.lineWidth=1.5;
  ctx.beginPath();let first=true;
  for(let i=0;i<n;i++){{
    if(K_sff[i]<=0)continue;
    const x=toX(t_sff[i]),y=toY(K_sff[i]);
    if(first){{ctx.moveTo(x,y);first=false;}}else ctx.lineTo(x,y);
  }}
  ctx.stroke();
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(pad,H-pad);ctx.lineTo(W-pad,H-pad);ctx.stroke();
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';
  ctx.fillText('K(t)',4,pad+4);ctx.fillText('t (log)',W/2-12,H-8);
  ctx.fillText(tmin.toFixed(1),pad-2,H-pad+12);ctx.fillText(tmax.toFixed(0),W-pad-16,H-pad+12);
  ctx.fillText('dip\u2192ramp\u2192plateau',pad+4,pad+24);
}}

drawSpec();drawHist();drawSFF();
window.addEventListener('resize',()=>{{drawSpec();drawHist();drawSFF();}});
}})();
</script>
</body>
</html>"""


def _kondo_frames(
    N_bath: int = 5,
    eps_d: float = -1.5,
    U: float = 3.0,
    V: float = 0.3,
    bandwidth: float = 4.0,
    n_V: int = 30,
    V_max: float = 0.8,
    n_eps: int = 40,
    eps_min: float = -5.0,
    eps_max: float = 2.0,
    scenario: str = "both",
    source: str = "python-reference",
) -> dict:
    """NumPy reference: Single-Impurity Anderson Model (SIAM) / Kondo effect."""
    import numpy as np  # noqa: PLC0415
    from itertools import combinations  # noqa: PLC0415

    n_sites = N_bath + 1
    n_half = n_sites // 2  # half-filling: n_up = n_dn

    # Enumerate n_occ-particle states in n_sites-site space
    def enum_sector(n_occ: int) -> list[int]:
        return [sum(1 << b for b in combo)
                for combo in combinations(range(n_sites), n_occ)]

    # Jordan-Wigner sign for hopping c†_i c_j (0 < j, i = 0)
    def hopping_sign(state: int, i: int, j: int) -> int:
        lo, hi = min(i, j), max(i, j)
        mask = ((1 << hi) - (1 << (lo + 1)))
        return -1 if bin(state & mask).count("1") & 1 else 1

    def build_H(eps_d_v: float, V_v: float) -> np.ndarray:
        up_s = enum_sector(n_half)
        dn_s = enum_sector(n_half)
        up_lut = {s: i for i, s in enumerate(up_s)}
        dn_lut = {s: i for i, s in enumerate(dn_s)}
        du, dd = len(up_s), len(dn_s)
        dim = du * dd
        H = np.zeros((dim, dim))
        dW = bandwidth / (N_bath - 1) if N_bath > 1 else 0.0
        bath_eps = [(k - (N_bath - 1) * 0.5) * dW for k in range(N_bath)]

        for iu, au in enumerate(up_s):
            for id_, ad in enumerate(dn_s):
                row = iu * dd + id_
                # Diagonal
                nu0, nd0 = (au >> 0) & 1, (ad >> 0) & 1
                diag = eps_d_v * (nu0 + nd0) + U * nu0 * nd0
                for k in range(N_bath):
                    s = k + 1
                    diag += bath_eps[k] * (((au >> s) & 1) + ((ad >> s) & 1))
                H[row, row] = diag
                # Hybridisation: c†_0 c_s (spin-up)
                for k in range(N_bath):
                    s = k + 1
                    # spin-up
                    if ((au >> s) & 1) and not ((au >> 0) & 1):
                        sgn = hopping_sign(au, 0, s)
                        au2 = (au ^ (1 << s)) | 1
                        if au2 in up_lut:
                            col = up_lut[au2] * dd + id_
                            H[row, col] += V_v * sgn
                            H[col, row] += V_v * sgn
                    # spin-dn
                    if ((ad >> s) & 1) and not ((ad >> 0) & 1):
                        sgn = hopping_sign(ad, 0, s)
                        ad2 = (ad ^ (1 << s)) | 1
                        if ad2 in dn_lut:
                            col = iu * dd + dn_lut[ad2]
                            H[row, col] += V_v * sgn
                            H[col, row] += V_v * sgn
        return H, up_s, dn_s, du, dd

    def gs_obs(ev, evecs, up_s, dn_s, dd):
        idx0 = int(np.argmin(ev))
        gs = evecs[:, idx0]
        du = len(up_s)
        n_imp = m2_imp = 0.0
        spin_corr = [0.0] * N_bath
        for iu, au in enumerate(up_s):
            for id_, ad in enumerate(dn_s):
                c2 = gs[iu * dd + id_] ** 2
                if c2 < 1e-20:
                    continue
                nu0 = (au >> 0) & 1
                nd0 = (ad >> 0) & 1
                n_imp += c2 * (nu0 + nd0)
                sz_imp = (nu0 - nd0) * 0.5
                m2_imp += c2 * sz_imp ** 2
                for k in range(N_bath):
                    s = k + 1
                    sz_b = ((au >> s) & 1) - ((ad >> s) & 1)
                    spin_corr[k] += c2 * sz_imp * sz_b * 0.5
        return n_imp, m2_imp, spin_corr

    # Analytical Kondo temperature
    rho0 = N_bath / bandwidth if bandwidth > 0 else 0.0
    Gamma = math.pi * rho0 * V ** 2
    if Gamma > 1e-12 and U > 0:
        ed_abs = abs(eps_d)
        exp_arg = -math.pi * ed_abs * (ed_abs + U) / (Gamma * U)
        TK = math.sqrt(Gamma * U * 0.5) * math.exp(max(exp_arg, -100.0))
    else:
        TK = 0.0

    out: dict = {
        "source": source, "N_bath": N_bath, "eps_d": eps_d,
        "U": U, "V": V, "bandwidth": bandwidth, "TK_analytic": TK,
    }

    # Fixed-parameter ground state
    H, up_s, dn_s, du, dd = build_H(eps_d, V)
    ev, evecs = np.linalg.eigh(H)
    idx0 = int(np.argmin(ev))
    E_ground = float(ev[idx0])
    E_gap = float(ev[idx0 + 1] - ev[idx0]) if len(ev) > idx0 + 1 else 0.0
    n_imp, m2_imp, spin_corr = gs_obs(ev, evecs, up_s, dn_s, dd)
    out.update({
        "E_ground": E_ground, "E_gap": E_gap,
        "n_imp": n_imp, "m2_imp": m2_imp, "spin_corr": spin_corr,
    })

    if scenario in ("kondo", "both"):
        V_vals, gap_V, m2_V = [], [], []
        for vi in range(n_V):
            Vv = V_max * vi / (n_V - 1) if n_V > 1 else V_max
            H_v, up_sv, dn_sv, _, dd_v = build_H(eps_d, Vv)
            ev_v, evecs_v = np.linalg.eigh(H_v)
            i0 = int(np.argmin(ev_v))
            gap_v = float(ev_v[i0 + 1] - ev_v[i0]) if len(ev_v) > i0 + 1 else 0.0
            _, m2_v, _ = gs_obs(ev_v, evecs_v, up_sv, dn_sv, dd_v)
            V_vals.append(float(Vv))
            gap_V.append(gap_v)
            m2_V.append(float(m2_v))
        out.update({"V_vals": V_vals, "gap_V": gap_V, "m2_V": m2_V})

    if scenario in ("occupancy", "both"):
        eps_vals, n_d_eps, m2_eps = [], [], []
        for ei in range(n_eps):
            ed = eps_min + (eps_max - eps_min) * ei / (n_eps - 1 if n_eps > 1 else 1)
            H_e, up_se, dn_se, _, dd_e = build_H(ed, V)
            ev_e, evecs_e = np.linalg.eigh(H_e)
            n_e, m2_e, _ = gs_obs(ev_e, evecs_e, up_se, dn_se, dd_e)
            eps_vals.append(float(ed))
            n_d_eps.append(float(n_e))
            m2_eps.append(float(m2_e))
        out.update({"eps_vals": eps_vals, "n_d_eps": n_d_eps, "m2_eps": m2_eps})

    return out


def _kondo_page(data: dict) -> str:
    """Render Kondo/SIAM results as a self-contained HTML page."""
    payload = json.dumps(data, separators=(",", ":"))
    N_bath  = data.get("N_bath", 5)
    eps_d   = data.get("eps_d", -1.5)
    U_val   = data.get("U", 3.0)
    V_val   = data.get("V", 0.3)
    TK      = data.get("TK_analytic", 0.0)
    n_imp   = data.get("n_imp", 0.0)
    m2_imp  = data.get("m2_imp", 0.0)
    E_gap   = data.get("E_gap", 0.0)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Kondo Effect / SIAM \u2014 QuantX</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0d1117;color:#c9d1d9;font-family:monospace;padding:16px}}
  h2{{color:#e6edf3;font-size:1.1rem;margin-bottom:12px}}
  .info{{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:14px;font-size:.82rem;color:#8b949e}}
  .info span b{{color:#c9d1d9}}
  .panels{{display:flex;gap:16px;flex-wrap:wrap}}
  .panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px}}
  canvas{{display:block}}
</style>
</head>
<body>
<h2>Kondo Effect \u2014 Single-Impurity Anderson Model</h2>
<div class="info">
  <span><b>N_bath</b>={N_bath}</span>
  <span><b>&#949;_d</b>={eps_d:.2f}</span>
  <span><b>U</b>={U_val:.2f}</span>
  <span><b>V</b>={V_val:.3f}</span>
  <span><b>T_K</b>={TK:.4g}</span>
  <span><b>n_imp</b>={n_imp:.3f}</span>
  <span><b>&#60;S_z&#178;&#62;</b>={m2_imp:.4f}</span>
  <span><b>&#916;E</b>={E_gap:.4g}</span>
</div>
<div class="panels">
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">&#60;S_z&#178;&#62; &amp; Gap vs V (Kondo screening)</div>
    <canvas id="cKondo" width="380" height="340"></canvas>
  </div>
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Impurity Occupation n_d vs &#949;_d</div>
    <canvas id="cOcc" width="360" height="340"></canvas>
  </div>
  <div class="panel">
    <div style="color:#8b949e;font-size:.78rem;margin-bottom:6px">Kondo Cloud: &#60;S_z&#94;imp&#183;S_z&#94;k&#62; per bath site</div>
    <canvas id="cCloud" width="320" height="340"></canvas>
  </div>
</div>
<script>
(function(){{
const DATA={payload};

function drawKondo(){{
  const cv=document.getElementById('cKondo');
  if(!cv)return;
  const W=cv.width,H=cv.height,pad=48;
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,H);
  const Vv=DATA.V_vals||[],gV=DATA.gap_V||[],m2V=DATA.m2_V||[];
  if(!Vv.length){{ctx.fillStyle='#8b949e';ctx.font='11px monospace';ctx.fillText('(kondo sweep not computed)',pad,H/2);return;}}
  const n=Vv.length,vmax=Vv[n-1]||1;
  const gmax=Math.max(...gV,1e-9);
  const toX=v=>(pad+v/vmax*(W-2*pad));
  const toYg=g=>(H-pad-(g/gmax)*(H-2*pad));
  const toYm=m=>(H-pad-(m/0.25)*(H-2*pad));
  // Gap line (blue)
  ctx.strokeStyle='#388bfd';ctx.lineWidth=1.5;
  ctx.beginPath();
  for(let i=0;i<n;i++){{if(i===0)ctx.moveTo(toX(Vv[i]),toYg(gV[i]));else ctx.lineTo(toX(Vv[i]),toYg(gV[i]));}}
  ctx.stroke();
  // m2 line (orange)
  ctx.strokeStyle='#f0883e';ctx.lineWidth=1.5;
  ctx.beginPath();
  for(let i=0;i<n;i++){{if(i===0)ctx.moveTo(toX(Vv[i]),toYm(m2V[i]));else ctx.lineTo(toX(Vv[i]),toYm(m2V[i]));}}
  ctx.stroke();
  // Axes
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(pad,H-pad);ctx.lineTo(W-pad,H-pad);ctx.stroke();
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';
  ctx.fillText('0',pad-8,H-pad+12);ctx.fillText(vmax.toFixed(2),W-pad-18,H-pad+12);
  ctx.fillText('V',W/2-4,H-8);
  ctx.fillStyle='#388bfd';ctx.fillText('\u0394E (gap)',pad+4,pad+12);
  ctx.fillStyle='#f0883e';ctx.fillText('\u27e8S_z\u00b2\u27e9 (moment)',pad+4,pad+22);
  ctx.fillStyle='#3fb950';ctx.font='8px monospace';
  const TK=DATA.TK_analytic||0;
  ctx.fillText('T_K\u2248'+TK.toFixed(4),pad+4,pad+32);
}}

function drawOcc(){{
  const cv=document.getElementById('cOcc');
  if(!cv)return;
  const W=cv.width,H=cv.height,pad=48;
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,H);
  const ev=DATA.eps_vals||[],nd=DATA.n_d_eps||[],m2=DATA.m2_eps||[];
  if(!ev.length){{ctx.fillStyle='#8b949e';ctx.font='11px monospace';ctx.fillText('(occupancy sweep not computed)',pad,H/2);return;}}
  const n=ev.length,emin=ev[0],emax=ev[n-1],er=emax-emin||1;
  const toX=e=>(pad+(e-emin)/er*(W-2*pad));
  const toY=v=>(H-pad-v/2*(H-2*pad));
  const toYm=m=>(H-pad-m/0.25*(H-2*pad));
  // n_d curve (blue)
  ctx.strokeStyle='#388bfd';ctx.lineWidth=1.5;
  ctx.beginPath();
  for(let i=0;i<n;i++){{if(i===0)ctx.moveTo(toX(ev[i]),toY(nd[i]));else ctx.lineTo(toX(ev[i]),toY(nd[i]));}}
  ctx.stroke();
  // m2 curve (orange)
  ctx.strokeStyle='#f0883e';ctx.lineWidth=1.5;
  ctx.beginPath();
  for(let i=0;i<n;i++){{if(i===0)ctx.moveTo(toX(ev[i]),toYm(m2[i]));else ctx.lineTo(toX(ev[i]),toYm(m2[i]));}}
  ctx.stroke();
  // Kondo plateau at n_d=1 (dashed)
  ctx.strokeStyle='#e3b341';ctx.lineWidth=1;ctx.setLineDash([3,3]);
  ctx.beginPath();ctx.moveTo(pad,toY(1));ctx.lineTo(W-pad,toY(1));ctx.stroke();
  ctx.setLineDash([]);
  // Axes
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(pad,H-pad);ctx.lineTo(W-pad,H-pad);ctx.stroke();
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';
  ctx.fillText('\u03b5_d',W/2-8,H-8);
  ctx.fillText(emin.toFixed(1),pad-4,H-pad+12);ctx.fillText(emax.toFixed(1),W-pad-12,H-pad+12);
  ctx.fillStyle='#388bfd';ctx.fillText('n_d',pad+4,pad+12);
  ctx.fillStyle='#f0883e';ctx.fillText('\u27e8S_z\u00b2\u27e9',pad+4,pad+22);
  ctx.fillStyle='#e3b341';ctx.fillText('n_d=1 (Kondo)',pad+4,toY(1)-4);
}}

function drawCloud(){{
  const cv=document.getElementById('cCloud');
  if(!cv)return;
  const W=cv.width,H=cv.height,pad=44;
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,W,H);
  const sc=DATA.spin_corr||[];
  if(!sc.length){{ctx.fillStyle='#8b949e';ctx.font='11px monospace';ctx.fillText('(spin corr not computed)',pad,H/2);return;}}
  const n=sc.length;
  const vmax=Math.max(Math.max(...sc.map(Math.abs)),1e-9);
  const bw=(W-2*pad)/n-2;
  const toX=i=>(pad+i/n*(W-2*pad));
  const midY=H/2;
  const scale=(H/2-pad-4)/vmax;
  // Zero line
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad,midY);ctx.lineTo(W-pad,midY);ctx.stroke();
  for(let i=0;i<n;i++){{
    const x=toX(i),h=Math.abs(sc[i])*scale;
    ctx.fillStyle=sc[i]<0?'rgba(56,139,253,0.8)':'rgba(240,136,62,0.8)';
    if(sc[i]<0)ctx.fillRect(x,midY,bw,h);
    else ctx.fillRect(x,midY-h,bw,h);
  }}
  // Axes
  ctx.strokeStyle='#30363d';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(pad,pad);ctx.lineTo(pad,H-pad);ctx.lineTo(W-pad,H-pad);ctx.stroke();
  ctx.fillStyle='#8b949e';ctx.font='9px monospace';
  ctx.fillText('\u27e8S_z^imp\u00b7S_z^k\u27e9',4,pad+4);
  ctx.fillText('bath site k',W/2-24,H-8);
  ctx.fillStyle='#388bfd';ctx.fillText('neg=antiferro',pad+2,H-pad-6);
}}

drawKondo();drawOcc();drawCloud();
window.addEventListener('resize',()=>{{drawKondo();drawOcc();drawCloud();}});
}})();
</script>
</body>
</html>"""
