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
