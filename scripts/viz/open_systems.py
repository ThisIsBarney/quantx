from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from .common import _make_engine, _animated_page, _attach_continuous_metadata, _user_cli_error


def _lindblad_frames(
    n_qubits: int,
    hamiltonian_terms: list[dict],
    collapse_ops: list[dict],
    total_time: float = 2.0,
    dt: float = 0.01,
    samples: int = 80,
    source: str = "python-reference",
) -> dict:
    """Simulate Lindblad master equation and return animation frame data.

    hamiltonian_terms: list of {"label": "X0Z1", "coefficient": 1.0}
    collapse_ops: list of {"label": "AmplitudeDamping0", "coefficient": gamma}
    Returns a dict with keys: n_qubits, frames, source, total_time
    """
    try:
      import numpy as np
    except ImportError as ex:
      raise RuntimeError("Lindblad visualization requires numpy") from ex

    dim = 1 << n_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0

    # Build Hamiltonian matrix
    H = np.zeros((dim, dim), dtype=complex)
    import re
    token_re = re.compile(r'([XYZxyz])(\d+)')

    def _pauli_x():
        return np.array([[0, 1], [1, 0]], dtype=complex)
    def _pauli_y():
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    def _pauli_z():
        return np.array([[1, 0], [0, -1]], dtype=complex)
    def _sigma_minus():
        return np.array([[0, 1], [0, 0]], dtype=complex)

    def _embed(op2x2, q, n):
        """Embed 2x2 op at qubit q (qubit 0 = LSB) in 2^n space."""
        full_dim = 1 << n
        q_mask = 1 << q
        result = np.zeros((full_dim, full_dim), dtype=complex)
        for col in range(full_dim):
            c_q = (col >> q) & 1
            col_rest = col & ~q_mask
            for row in range(full_dim):
                if (row & ~q_mask) == col_rest:
                    r_q = (row >> q) & 1
                    result[row, col] = op2x2[r_q, c_q]
        return result

    def _build_H(terms, n):
        full_dim = 1 << n
        H = np.zeros((full_dim, full_dim), dtype=complex)
        for t in terms:
            label = t.get("label", "")
            coeff = t.get("coefficient", 0.0)
            if any(label.startswith(p) for p in ("AmplitudeDamping", "Dephasing", "Depolarizing")):
                continue
            if not label:
                continue
            flip_mask = 0
            yz_mask = 0
            y_count = 0
            valid = True
            for m in token_re.finditer(label):
                p = m.group(1).upper()
                q = int(m.group(2))
                if q >= n or q >= 63:
                    valid = False; break
                bit = 1 << q
                if p == 'X':
                    flip_mask |= bit
                elif p == 'Y':
                    flip_mask |= bit; yz_mask |= bit; y_count += 1
                elif p == 'Z':
                    yz_mask |= bit
                else:
                    valid = False; break
            if not valid:
                continue
            phases = [1+0j, 0+1j, -1+0j, 0-1j]
            global_phase = phases[y_count % 4]
            for src in range(full_dim):
                dst = src ^ flip_mask
                sign_bits = bin(src & yz_mask).count('1')
                sign = -1.0 if sign_bits % 2 else 1.0
                H[dst, src] += coeff * global_phase * sign
        return H

    def _build_collapse_ops(ops_spec, n):
        Lk = []
        for op in ops_spec:
            label = op.get("label", "")
            gamma = max(0.0, op.get("coefficient", 0.0))
            sqrt_gamma = math.sqrt(gamma)
            if label.startswith("AmplitudeDamping"):
                q = int(label[len("AmplitudeDamping"):])
                if q < n:
                    L = sqrt_gamma * _embed(_sigma_minus(), q, n)
                    Lk.append(L)
            elif label.startswith("Dephasing"):
                q = int(label[len("Dephasing"):])
                if q < n:
                    L = (sqrt_gamma * 0.5) * _embed(_pauli_z(), q, n)
                    Lk.append(L)
            elif label.startswith("Depolarizing"):
                q = int(label[len("Depolarizing"):])
                if q < n:
                    sqrt_g3 = math.sqrt(gamma / 3.0) if gamma > 0 else 0.0
                    for op2x2 in [_pauli_x(), _pauli_y(), _pauli_z()]:
                        Lk.append(sqrt_g3 * _embed(op2x2, q, n))
        return Lk

    H_mat = _build_H(hamiltonian_terms + collapse_ops, n_qubits)
    Lk_list = _build_collapse_ops(hamiltonian_terms + collapse_ops, n_qubits)
    LdagL_list = [L.conj().T @ L for L in Lk_list]

    def lindblad_rhs(rho):
        drho = -1j * (H_mat @ rho - rho @ H_mat)
        for L, LdagL in zip(Lk_list, LdagL_list):
            drho += L @ rho @ L.conj().T - 0.5 * (LdagL @ rho + rho @ LdagL)
        return drho

    def rk4_step(rho, dt):
        k1 = lindblad_rhs(rho)
        k2 = lindblad_rhs(rho + 0.5 * dt * k1)
        k3 = lindblad_rhs(rho + 0.5 * dt * k2)
        k4 = lindblad_rhs(rho + dt * k3)
        rho_new = rho + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return 0.5 * (rho_new + rho_new.conj().T)  # symmetrise

    def record_frame(rho, step, t):
        pops = [max(0.0, rho[i, i].real) for i in range(dim)]
        purity = max(0.0, (rho @ rho).trace().real)
        # entropy from eigenvalues
        evals = np.maximum(0.0, np.linalg.eigvalsh(rho))
        entropy = float(-np.sum(evals * np.log(np.where(evals > 1e-15, evals, 1.0)) * (evals > 1e-15)))
        off_diag = [abs(rho[i, j]) for i in range(dim) for j in range(dim) if i != j]
        coherence = max(off_diag) if off_diag else 0.0
        return {
            "step": step,
            "t": round(t, 10),
            "purity": round(purity, 10),
            "entropy": round(entropy, 10),
            "populations": [round(p, 10) for p in pops],
            "coherence": round(coherence, 10),
        }

    # Compute step stride for output frames
    total_steps = max(1, round(total_time / dt))
    stride = max(1, total_steps // max(1, samples - 1))

    frames = [record_frame(rho, 0, 0.0)]
    for s in range(total_steps):
        rho = rk4_step(rho, dt)
        t = (s + 1) * dt
        if (s + 1) % stride == 0 or s + 1 == total_steps:
            frames.append(record_frame(rho, s + 1, t))

    return {
        "n_qubits": n_qubits,
        "dim": dim,
        "source": source,
        "total_time": total_time,
        "frames": frames,
        "hamiltonian_terms": hamiltonian_terms,
        "collapse_ops": collapse_ops,
    }


def _lindblad_page(data: dict) -> str:
    """Render an animated HTML page for Lindblad density-matrix evolution."""
    data_json = json.dumps(data, ensure_ascii=False)
    n_qubits = data.get("n_qubits", 1)
    dim = data.get("dim", 1 << n_qubits)
    state_labels = [f"|{i:0{n_qubits}b}\u27e9" for i in range(dim)]
    labels_json = json.dumps(state_labels)

    # Generate a distinct color for each basis state
    colors = ["#0b84f3", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
              "#1abc9c", "#e67e22", "#34495e"]
    colors_json = json.dumps([colors[i % len(colors)] for i in range(dim)])

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>QuantX \u2014 Lindblad Open Quantum Systems</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; background: #0f1117; color: #e0e0e0;
            margin: 0; padding: 16px; }}
    h1 {{ font-size: 1.4em; margin-bottom: 4px; color: #90caf9; }}
    .subtitle {{ color: #888; font-size: 0.9em; margin-bottom: 16px; }}
    .panels {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; max-width: 1100px; }}
    .panel {{ background: #1a1d27; border-radius: 8px; padding: 12px; }}
    .panel h2 {{ font-size: 1em; margin: 0 0 8px; color: #b0c4de; }}
    canvas {{ width: 100%; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin: 12px 0; }}
    button {{ padding: 4px 14px; background: #263040; border: 1px solid #4a90d9;
             color: #e0e0e0; border-radius: 4px; cursor: pointer; }}
    button:hover {{ background: #2d4060; }}
    input[type=range] {{ flex: 1; accent-color: #4a90d9; }}
    .info {{ font-size: 0.82em; color: #888; margin-top: 4px; }}
    .legend {{ font-size: 0.82em; margin-top: 6px; }}
    .legend span {{ margin-right: 12px; }}
  </style>
</head>
<body>
  <h1>Lindblad \u2014 Open Quantum System Evolution</h1>
  <p class="subtitle">source: {data.get("source","python-reference")} &nbsp;|&nbsp;
     qubits: {n_qubits} &nbsp;|&nbsp;
     dim: {dim}\xd7{dim} density matrix</p>
  <div class="controls">
    <button id="play">\u25b6 Play</button>
    <button id="pause">\u23f8 Pause</button>
    <input type="range" id="slider" min="1" max="1" value="1"/>
    <span id="label" class="info">frame 1</span>
  </div>
  <div class="panels">
    <div class="panel">
      <h2>Populations \u03c1\u1d62\u1d62</h2>
      <canvas id="cv_pop" height="240"></canvas>
      <div id="leg_pop" class="legend"></div>
    </div>
    <div class="panel">
      <h2>Purity Tr(\u03c1\u00b2) &amp; Coherence max|\u03c1\u1d62\u2c7c|</h2>
      <canvas id="cv_pur" height="240"></canvas>
    </div>
    <div class="panel">
      <h2>Von Neumann Entropy \u2212Tr(\u03c1 log \u03c1)</h2>
      <canvas id="cv_ent" height="200"></canvas>
    </div>
    <div class="panel">
      <h2>Density Matrix \u03c1 (real part at current frame)</h2>
      <canvas id="cv_rho" height="200"></canvas>
    </div>
  </div>
  <script>
  const DATA = {data_json};
  const STATE_LABELS = {labels_json};
  const COLORS = {colors_json};
  const DIM = DATA.dim;
  const frames = DATA.frames;

  const slider = document.getElementById("slider");
  slider.max = String(frames.length);

  // Build population legend
  const legPop = document.getElementById("leg_pop");
  COLORS.forEach((c, i) => {{
    const sp = document.createElement("span");
    sp.innerHTML = `<span style="color:${{c}}">&#9632;</span> ${{STATE_LABELS[i]}}`;
    legPop.appendChild(sp);
  }});

  // \u2500 Generic line chart helper \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  function drawLineChart(canvasId, series, currentFrame) {{
    const cv = document.getElementById(canvasId);
    const W = cv.offsetWidth || 500, H = cv.height;
    cv.width = W;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#1a1d27";
    ctx.fillRect(0, 0, W, H);

    const L = 48, R = 12, T = 8, B = 32;
    const plotW = W - L - R, plotH = H - T - B;

    // Find ranges
    let yMin = Infinity, yMax = -Infinity;
    for (const s of series) {{
      for (const [, y] of s.points) {{
        yMin = Math.min(yMin, y); yMax = Math.max(yMax, y);
      }}
    }}
    if (!isFinite(yMin)) yMin = 0;
    if (!isFinite(yMax)) yMax = 1;
    if (Math.abs(yMax - yMin) < 1e-10) {{ yMin -= 0.1; yMax += 0.1; }}
    const xMin = frames[0].t, xMax = frames[frames.length-1].t;
    const sx = x => L + (x - xMin) / (xMax - xMin) * plotW;
    const sy = y => T + (1 - (y - yMin) / (yMax - yMin)) * plotH;

    // Axes
    ctx.strokeStyle = "#444";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(L, T); ctx.lineTo(L, T + plotH); ctx.lineTo(L + plotW, T + plotH); ctx.stroke();
    ctx.fillStyle = "#888"; ctx.font = "10px sans-serif"; ctx.textAlign = "right";
    for (let i = 0; i <= 4; i++) {{
      const v = yMin + (yMax - yMin) * i / 4;
      const py = sy(v);
      ctx.fillText(v.toFixed(2), L - 4, py + 3);
      ctx.strokeStyle = "#2a2a3a"; ctx.beginPath(); ctx.moveTo(L, py); ctx.lineTo(L + plotW, py); ctx.stroke();
    }}
    ctx.textAlign = "center";
    for (let i = 0; i <= 4; i++) {{
      const v = xMin + (xMax - xMin) * i / 4;
      ctx.fillText(v.toFixed(2), sx(v), T + plotH + 16);
    }}

    // Series
    for (const s of series) {{
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      s.points.forEach(([x, y], i) => {{
        if (i === 0) ctx.moveTo(sx(x), sy(y));
        else ctx.lineTo(sx(x), sy(y));
      }});
      ctx.stroke();
    }}

    // Current time marker
    const t_cur = frames[currentFrame - 1].t;
    ctx.strokeStyle = "#ffffff44";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(sx(t_cur), T); ctx.lineTo(sx(t_cur), T + plotH); ctx.stroke();
    ctx.setLineDash([]);
  }}

  // \u2500 Density matrix heatmap \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
  function drawRhoHeatmap(frameIdx) {{
    const cv = document.getElementById("cv_rho");
    const W = cv.offsetWidth || 300;
    cv.width = W;
    const H = cv.height;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#1a1d27";
    ctx.fillRect(0, 0, W, H);

    const f = frames[frameIdx - 1];
    const pops = f.populations;
    if (!pops) return;
    const cellW = Math.floor(W / DIM), cellH = Math.floor(H / DIM);

    // We only have populations (diagonal). Show diagonal bar chart as heatmap substitute.
    const maxPop = Math.max(...pops, 1e-10);
    for (let i = 0; i < DIM; ++i) {{
      const v = pops[i] / maxPop;
      const g = Math.round(v * 200);
      ctx.fillStyle = `rgb(${{Math.round(v*80)}},${{Math.round(v*160)}},255)`;
      const x = i * cellW + 2;
      const barH = Math.round(v * (H - 20));
      ctx.fillRect(x, H - barH - 10, Math.max(1, cellW - 4), barH);
      ctx.fillStyle = "#888";
      ctx.font = "9px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(STATE_LABELS[i], x + (cellW - 4)/2, H - 2);
    }}
  }}

  function render(frameIdx) {{
    const popSeries = COLORS.map((c, i) => ({{
      color: c,
      points: frames.map(f => [f.t, f.populations[i] || 0]),
    }}));
    drawLineChart("cv_pop", popSeries, frameIdx);
    drawLineChart("cv_pur", [
      {{ color: "#4fc3f7", points: frames.map(f => [f.t, f.purity]) }},
      {{ color: "#ffb74d", points: frames.map(f => [f.t, f.coherence]) }},
    ], frameIdx);
    drawLineChart("cv_ent", [
      {{ color: "#a5d6a7", points: frames.map(f => [f.t, f.entropy]) }},
    ], frameIdx);
    drawRhoHeatmap(frameIdx);

    const f = frames[frameIdx - 1];
    document.getElementById("label").textContent =
      `frame ${{frameIdx}}/${{frames.length}}, t=${{f.t.toFixed(4)}}, ` +
      `purity=${{f.purity.toFixed(4)}}, entropy=${{f.entropy.toFixed(4)}}`;
    slider.value = String(frameIdx);
  }}

  let timer = null;
  function doPlay() {{
    if (timer) return;
    timer = setInterval(() => {{
      let v = Number(slider.value);
      v = v >= frames.length ? 1 : v + 1;
      render(v);
    }}, 60);
  }}
  function doPause() {{ if (!timer) return; clearInterval(timer); timer = null; }}
  document.getElementById("play").onclick = doPlay;
  document.getElementById("pause").onclick = doPause;
  slider.oninput = () => render(Number(slider.value));
  render(1);
  </script>
</body>
</html>"""


def _mps_frames(
    n_sites: int,
    hamiltonian_terms: list[dict],
    total_time: float = 6.0,
    dt: float = 0.05,
    max_bond: int = 32,
    trunc_tol: float = 1e-8,
    samples: int = 60,
    source: str = "python-reference",
) -> dict:
    """Numpy TEBD reference solver for 1D nearest-neighbour Hamiltonians.

    Supports Pauli-string terms with |i-j|=1 (NN two-body) and single-site.
    Returns bond entropies, bond dimensions, and site magnetisation over time.
    """
    try:
        import numpy as np
    except ImportError as ex:
        raise RuntimeError("MPS visualization requires numpy") from ex

    import re
    d = 2  # qubit physical dimension
    tok = re.compile(r'([XYZxyz])(\d+)')

    def _px(): return np.array([[0,1],[1,0]], dtype=complex)
    def _py(): return np.array([[0,-1j],[1j,0]], dtype=complex)
    def _pz(): return np.array([[1,0],[0,-1]], dtype=complex)
    def _pi(): return np.eye(2, dtype=complex)
    def _p(c):
        c = c.upper()
        return _px() if c=='X' else _py() if c=='Y' else _pz() if c=='Z' else _pi()

    def kron2(A, B): return np.kron(A, B)

    # Parse Hamiltonian into NN 2-body and single-site terms.
    two_body: dict[int, np.ndarray] = {}   # bond b → 4×4 Hermitian
    single_body: dict[int, np.ndarray] = {}  # site → 2×2 Hermitian
    for ht in hamiltonian_terms:
        label, coeff = ht.get("label",""), ht.get("coefficient", 0.0)
        ops = [(m.group(1).upper(), int(m.group(2))) for m in tok.finditer(label)]
        if not ops: continue
        if len(ops) == 1:
            s, q = ops[0]
            if 0 <= q < n_sites:
                single_body[q] = single_body.get(q, np.zeros((2,2), complex)) + coeff * _p(s)
        elif len(ops) == 2:
            (s1,q1),(s2,q2) = ops
            if abs(q1-q2) == 1 and 0 <= q1 < n_sites and 0 <= q2 < n_sites:
                b = min(q1,q2)
                op = kron2(_p(s1 if q1<q2 else s2), _p(s2 if q1<q2 else s1))
                two_body[b] = two_body.get(b, np.zeros((4,4), complex)) + coeff * op

    # Build gates for 2nd-order Suzuki-Trotter:
    # even bonds use half-steps at start/end, odd bonds use full step in middle.
    def _expm_small(H):
        vals, vecs = np.linalg.eigh(H)
        return (vecs * np.exp(-1j * vals)) @ vecs.conj().T

    even_half_gates = {b: _expm_small(0.5 * dt * H) for b, H in two_body.items() if b % 2 == 0}
    odd_full_gates = {b: _expm_small(dt * H) for b, H in two_body.items() if b % 2 == 1}
    # Single-site half-step gates for 2nd-order ST (½ dt each end of sweep).
    single_half = {s: _expm_small(0.5 * dt * H) for s, H in single_body.items()}

    # MPS state: list of site tensors A[i] shape (chi_L*d, chi_R).
    tensors = [np.zeros((d, 1), dtype=complex) for _ in range(n_sites)]
    for t in tensors: t[0, 0] = 1.0  # |0⟩
    schmidt = [np.array([1.0], dtype=float) for _ in range(max(0, n_sites - 1))]

    def bond_dim(i): return tensors[i].shape[1]

    def apply_1site(s, U2):
        A = tensors[s]
        chi_L = A.shape[0] // d
        chi_R = A.shape[1]
        for il in range(chi_L):
            for ir in range(chi_R):
                a0, a1 = A[il*d, ir], A[il*d+1, ir]
                A[il*d,   ir] = U2[0,0]*a0 + U2[0,1]*a1
                A[il*d+1, ir] = U2[1,0]*a0 + U2[1,1]*a1

    def apply_2site(b, U4):
        Ai, Aip1 = tensors[b], tensors[b+1]
        chi_L = Ai.shape[0] // d
        chi_M = Ai.shape[1]
        chi_R = Aip1.shape[1]
        # Reshape Aip1: (chi_M*d, chi_R) → (chi_M, d*chi_R)
        Aip1_r = np.zeros((chi_M, d*chi_R), complex)
        for im in range(chi_M):
            for s in range(d):
                Aip1_r[im, s*chi_R:(s+1)*chi_R] = Aip1[im*d+s, :]
        Theta = Ai @ Aip1_r  # (chi_L*d, d*chi_R)
        # Apply gate.
        Tp = np.zeros_like(Theta)
        for il in range(chi_L):
            for ir in range(chi_R):
                for s1n in range(d):
                    for s2n in range(d):
                        v = complex(0)
                        for s1o in range(d):
                            for s2o in range(d):
                                v += U4[s1n*d+s2n, s1o*d+s2o] * Theta[il*d+s1o, s2o*chi_R+ir]
                        Tp[il*d+s1n, s2n*chi_R+ir] = v
        # SVD and truncate.
        U, sv, Vh = np.linalg.svd(Tp, full_matrices=False)
        abs_tol = trunc_tol * sv[0] if sv.size > 0 else 1e-30
        chi_new = max(1, min(max_bond, int(np.sum(sv > abs_tol))))
        schmidt[b] = sv[:chi_new].copy()
        tensors[b]   = U[:, :chi_new]
        SV = np.diag(sv[:chi_new]) @ Vh[:chi_new, :]  # (chi_new, d*chi_R)
        Aip1_new = np.zeros((chi_new*d, chi_R), complex)
        for im in range(chi_new):
            for s in range(d):
                Aip1_new[im*d+s, :] = SV[im, s*chi_R:(s+1)*chi_R]
        tensors[b+1] = Aip1_new

    def tebd_step():
        for s, U in single_half.items(): apply_1site(s, U)
        for b in range(0, n_sites-1, 2):
            if b in even_half_gates: apply_2site(b, even_half_gates[b])
        for b in range(1, n_sites-1, 2):
            if b in odd_full_gates: apply_2site(b, odd_full_gates[b])
        for b in range(0, n_sites-1, 2):
            if b in even_half_gates: apply_2site(b, even_half_gates[b])
        for s, U in single_half.items(): apply_1site(s, U)

    def bond_entropy(b):
        sv = schmidt[b]
        norm2 = np.sum(sv**2)
        if norm2 < 1e-30: return 0.0
        p = sv**2 / norm2
        p = p[p > 1e-15]
        return float(-np.sum(p * np.log(p)))

    def magnetization_z(s):
        A = tensors[s]
        chi_L = A.shape[0] // d
        chi_R = A.shape[1]
        val = 0.0
        for il in range(chi_L):
            for ir in range(chi_R):
                val += abs(A[il*d, ir])**2 - abs(A[il*d+1, ir])**2
        return float(val)

    def collect():
        bds = [int(bond_dim(i)) for i in range(n_sites-1)]
        bes = [round(bond_entropy(b), 10) for b in range(n_sites-1)]
        mag = [round(magnetization_z(s), 10) for s in range(n_sites)]
        return {"bond_dims": bds, "bond_entropies": bes, "magnetization": mag,
                "max_bond_dim": max(bds) if bds else 1}

    total_steps = max(1, round(total_time / dt))
    stride = max(1, total_steps // max(1, samples - 1))

    frames = [{"step": 0, "t": 0.0, **collect()}]
    for s in range(total_steps):
        tebd_step()
        if (s+1) % stride == 0 or s+1 == total_steps:
            frames.append({"step": s+1, "t": round((s+1)*dt, 10), **collect()})

    return {
        "n_sites": n_sites, "max_bond": max_bond, "total_time": total_time,
        "source": source, "frames": frames,
        "hamiltonian_terms": hamiltonian_terms,
    }


def _mps_page(data: dict) -> str:
    """Render an animated HTML page for MPS/TEBD bond entropies and magnetization."""
    data_json = json.dumps(data, ensure_ascii=False)
    n_sites = data.get("n_sites", 4)
    title = f"MPS TEBD — {n_sites}-site chain"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif;padding:20px}}
h1{{font-size:1.1rem;font-weight:600;margin-bottom:12px;color:#58a6ff}}
.meta{{font-size:0.78rem;color:#8b949e;margin-bottom:16px}}
.controls{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}}
button{{background:#21262d;border:1px solid #30363d;color:#e6edf3;
        padding:6px 14px;border-radius:6px;cursor:pointer;font-size:0.82rem}}
button:hover{{background:#30363d}}
input[type=range]{{width:220px;accent-color:#58a6ff}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:980px}}
.panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}}
.panel h2{{font-size:0.85rem;font-weight:600;color:#8b949e;margin-bottom:10px}}
canvas{{display:block}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">TEBD 2nd-order Suzuki-Trotter &nbsp;|&nbsp; {n_sites} sites</div>
<div class="controls">
  <button id="btnPlay">&#9654; Play</button>
  <label style="font-size:.8rem;color:#8b949e">Step: <input type="range" id="slider" min="1" max="1" value="1"></label>
  <span id="tLabel" style="font-size:.8rem;color:#58a6ff"></span>
</div>
<div class="grid">
  <div class="panel">
    <h2>Bond Entanglement Entropy S(bond j)</h2>
    <canvas id="cvEntropy" width="440" height="200"></canvas>
  </div>
  <div class="panel">
    <h2>Site Magnetisation ⟨Z_i⟩</h2>
    <canvas id="cvMag" width="440" height="200"></canvas>
  </div>
  <div class="panel">
    <h2>Bond Entropy vs Time (heatmap)</h2>
    <canvas id="cvHeatEntropy" width="440" height="200"></canvas>
  </div>
  <div class="panel">
    <h2>Magnetisation vs Time (heatmap)</h2>
    <canvas id="cvHeatMag" width="440" height="200"></canvas>
  </div>
</div>
<script>
const DATA = {data_json};
const frames = DATA.frames;
const N = frames.length;
const nSites = DATA.n_sites;
const nBonds = nSites - 1;

const slider = document.getElementById("slider");
slider.max = N;

// ── Colour helpers ──────────────────────────────────────────────────────────
function lerp(a,b,t){{return a+(b-a)*t;}}
function heatColour(v, vMin, vMax, palette){{
  const t=vMax>vMin?(v-vMin)/(vMax-vMin):0.5;
  const ti=Math.min(palette.length-2,Math.floor(t*(palette.length-1)));
  const tf=t*(palette.length-1)-ti;
  const c0=palette[ti],c1=palette[ti+1];
  return `rgb(${{Math.round(lerp(c0[0],c1[0],tf))}},${{Math.round(lerp(c0[1],c1[1],tf))}},${{Math.round(lerp(c0[2],c1[2],tf))}})`;
}}
const HOT  =[[13,17,23],[30,80,160],[88,166,255],[63,185,80],[255,220,80],[255,120,40],[255,40,40]];
const COOL =[[13,17,23],[30,30,120],[60,60,200],[80,150,220],[200,200,255]];
const RWB  =[[0,80,200],[255,255,255],[200,0,0]];

// ── Static heatmaps ─────────────────────────────────────────────────────────
function drawHeatmap(canvasId, nCols, seriesKey, vMin, vMax, palette){{
  const cv=document.getElementById(canvasId);
  const W=cv.width, H=cv.height;
  const ctx=cv.getContext("2d");
  ctx.clearRect(0,0,W,H);
  const cellW=W/N, cellH=H/nCols;
  frames.forEach((f,fi)=>{{
    const vals=f[seriesKey]||[];
    for(let j=0;j<nCols;j++){{
      ctx.fillStyle=heatColour(vals[j]??0,vMin,vMax,palette);
      ctx.fillRect(fi*cellW,j*cellH,Math.ceil(cellW),Math.ceil(cellH));
    }}
  }});
}}

const maxEnt=Math.max(...frames.flatMap(f=>f.bond_entropies||[0]),0.01);
const maxMag=1.0;
drawHeatmap("cvHeatEntropy", nBonds, "bond_entropies", 0, maxEnt, HOT);
drawHeatmap("cvHeatMag",    nSites, "magnetization",  -1, 1,      RWB);

// ── Bar charts ──────────────────────────────────────────────────────────────
function drawBars(canvasId, vals, vMin, vMax, colour, label){{
  const cv=document.getElementById(canvasId);
  const W=cv.width,H=cv.height;
  const ctx=cv.getContext("2d");
  const pad={{l:36,r:8,t:8,b:28}};
  const w=W-pad.l-pad.r, h=H-pad.t-pad.b;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle="#30363d"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+h); ctx.lineTo(pad.l+w,pad.t+h); ctx.stroke();
  const range=vMax-vMin||1;
  ctx.fillStyle="#8b949e"; ctx.font="9px system-ui"; ctx.textAlign="right";
  for(let i=0;i<=4;i++){{
    const v=vMin+range*i/4;
    const y=pad.t+h-(v-vMin)/range*h;
    ctx.fillText(v.toFixed(2),pad.l-2,y+3);
    ctx.strokeStyle="#21262d"; ctx.beginPath(); ctx.moveTo(pad.l,y); ctx.lineTo(pad.l+w,y); ctx.stroke();
  }}
  if(!vals||!vals.length)return;
  const bw=w/vals.length*0.8;
  const gap=w/vals.length;
  vals.forEach((v,i)=>{{
    const x=pad.l+i*gap+(gap-bw)/2;
    const y0=pad.t+h-(0-vMin)/range*h;
    const y=pad.t+h-(v-vMin)/range*h;
    ctx.fillStyle=colour;
    ctx.fillRect(x,Math.min(y,y0),bw,Math.abs(y-y0)+0.5);
  }});
  ctx.fillStyle=colour; ctx.font="9px system-ui"; ctx.textAlign="left";
  ctx.fillText(label, pad.l+2, pad.t+12);
}}

// Cursor overlay on heatmap.
function drawCursor(canvasId, fi){{
  const cv=document.getElementById(canvasId);
  const ctx=cv.getContext("2d");
  const x=cv.width*fi/N;
  ctx.strokeStyle="rgba(255,255,255,0.5)"; ctx.lineWidth=1.5; ctx.setLineDash([4,3]);
  ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,cv.height); ctx.stroke();
  ctx.setLineDash([]);
}}

function render(idx){{
  const f=frames[idx-1];
  document.getElementById("tLabel").textContent=
    `t=${{f.t?.toFixed(3)??0}}  χ_max=${{f.max_bond_dim??1}}`;
  drawBars("cvEntropy", f.bond_entropies||[], 0, maxEnt, "#58a6ff", "S (nats)");
  drawBars("cvMag",     f.magnetization||[], -1, 1,      "#f0883e", "⟨Z_i⟩");
  // Redraw heatmaps and cursor.
  drawHeatmap("cvHeatEntropy", nBonds, "bond_entropies", 0, maxEnt, HOT);
  drawHeatmap("cvHeatMag",    nSites, "magnetization",  -1, 1,      RWB);
  drawCursor("cvHeatEntropy", idx-1);
  drawCursor("cvHeatMag",     idx-1);
}}
render(1);
slider.addEventListener("input",()=>render(+slider.value));

let timer=null;
document.getElementById("btnPlay").addEventListener("click",function(){{
  if(timer){{clearInterval(timer);timer=null;this.textContent="▶ Play";return;}}
  this.textContent="⏸ Pause";
  let idx=+slider.value;
  timer=setInterval(()=>{{
    idx=idx>=N?1:idx+1; slider.value=idx; render(idx);
    if(idx>=N){{clearInterval(timer);timer=null;document.getElementById("btnPlay").textContent="▶ Play";}}
  }},100);
}});
</script>
</body>
</html>"""


def _entanglement_frames(
    n_qubits: int,
    hamiltonian_terms: list[dict],
    collapse_ops: list[dict],
    n_a: int,
    total_time: float = 4.0,
    dt: float = 0.02,
    samples: int = 80,
    source: str = "python-reference",
) -> dict:
    """Compute bipartite entanglement measures via numpy.

    For pure-state time evolution (no collapse ops), uses exact matrix
    exponentiation and Schmidt decomposition via SVD.  For mixed-state runs
    (collapse_ops non-empty) uses Lindblad RK4 then partial-trace + SVD.

    Returns a dict with keys: n_qubits, n_a, n_b, frames, source, total_time.
    Each frame: step, t, entropy_a, mutual_info, concurrence, log_negativity,
                schmidt_values.
    """
    try:
      import numpy as np
    except ImportError as ex:
      raise RuntimeError("Entanglement visualization requires numpy") from ex

    import re
    token_re = re.compile(r'([XYZxyz])(\d+)')

    dim = 1 << n_qubits
    n_b = n_qubits - n_a
    da = 1 << n_a
    db = 1 << n_b

    # ── Hamiltonian builder (replicates _lindblad_frames logic) ─────────────
    def _pauli_x():
        return np.array([[0, 1], [1, 0]], dtype=complex)
    def _pauli_y():
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    def _pauli_z():
        return np.array([[1, 0], [0, -1]], dtype=complex)
    def _sigma_minus():
        return np.array([[0, 1], [0, 0]], dtype=complex)

    def _embed(op2x2, q, n):
        full_dim = 1 << n
        q_mask = 1 << q
        result = np.zeros((full_dim, full_dim), dtype=complex)
        for col in range(full_dim):
            c_q = (col >> q) & 1
            col_rest = col & ~q_mask
            for row in range(full_dim):
                if (row & ~q_mask) == col_rest:
                    r_q = (row >> q) & 1
                    result[row, col] = op2x2[r_q, c_q]
        return result

    def _build_H(terms, n):
        H = np.zeros((1 << n, 1 << n), dtype=complex)
        for t in terms:
            label = t.get("label", "")
            coeff = t.get("coefficient", 0.0)
            if any(label.startswith(p) for p in ("AmplitudeDamping", "Dephasing", "Depolarizing")):
                continue
            if not label:
                continue
            flip_mask = yz_mask = y_count = 0
            valid = True
            for m in token_re.finditer(label):
                p = m.group(1).upper()
                q = int(m.group(2))
                if q >= n or q >= 63:
                    valid = False; break
                bit = 1 << q
                if p == 'X':
                    flip_mask |= bit
                elif p == 'Y':
                    flip_mask |= bit; yz_mask |= bit; y_count += 1
                elif p == 'Z':
                    yz_mask |= bit
                else:
                    valid = False; break
            if not valid:
                continue
            global_phase = [1+0j, 0+1j, -1+0j, 0-1j][y_count % 4]
            for src in range(1 << n):
                dst = src ^ flip_mask
                sign = -1.0 if bin(src & yz_mask).count('1') % 2 else 1.0
                H[dst, src] += coeff * global_phase * sign
        return H

    def _build_collapse_ops(ops_spec, n):
        Lk = []
        for op in ops_spec:
            label = op.get("label", "")
            gamma = max(0.0, op.get("coefficient", 0.0))
            sg = math.sqrt(gamma)
            if label.startswith("AmplitudeDamping"):
                q = int(label[len("AmplitudeDamping"):])
                if q < n:
                    Lk.append(sg * _embed(_sigma_minus(), q, n))
            elif label.startswith("Dephasing"):
                q = int(label[len("Dephasing"):])
                if q < n:
                    Lk.append((sg * 0.5) * _embed(_pauli_z(), q, n))
            elif label.startswith("Depolarizing"):
                q = int(label[len("Depolarizing"):])
                if q < n:
                    sg3 = math.sqrt(gamma / 3.0) if gamma > 0 else 0.0
                    for op2x2 in [_pauli_x(), _pauli_y(), _pauli_z()]:
                        Lk.append(sg3 * _embed(op2x2, q, n))
        return Lk

    all_terms = hamiltonian_terms + collapse_ops
    H_mat = _build_H(all_terms, n_qubits)
    Lk_list = _build_collapse_ops(all_terms, n_qubits)

    # ── Entanglement measures ────────────────────────────────────────────────
    def _entropy(rho_sub):
        evals = np.maximum(0.0, np.linalg.eigvalsh(rho_sub))
        return float(-np.sum(evals * np.log(np.where(evals > 1e-15, evals, 1.0)) * (evals > 1e-15)))

    def _partial_trace_b(rho):
        """Trace over B: ρ_A[ia, ia'] = Σ_ib ρ[ia + ib*da, ia' + ib*da]."""
        rho_a = np.zeros((da, da), dtype=complex)
        for ib in range(db):
            rho_a += rho[ib*da:(ib+1)*da, ib*da:(ib+1)*da]
        return rho_a

    def _partial_trace_a(rho):
        """Trace over A: ρ_B[ib, ib'] = Σ_ia ρ[ia + ib*da, ia + ib'*da]."""
        rho_b = np.zeros((db, db), dtype=complex)
        for ia in range(da):
            rows = np.arange(db) * da + ia
            rho_b += rho[np.ix_(rows, rows)]
        return rho_b

    def _partial_transpose_a(rho):
        """Partial transpose w.r.t. A."""
        out = np.zeros((dim, dim), dtype=complex)
        for ib in range(db):
            for ibp in range(db):
                block = rho[ib*da:(ib+1)*da, ibp*da:(ibp+1)*da]
                out[ib*da:(ib+1)*da, ibp*da:(ibp+1)*da] = block.T
        return out

    def _wootters_concurrence(rho):
        """Wootters concurrence for 2-qubit density matrix."""
        sy_sy = np.array([[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]], dtype=complex)
        rho_tilde = sy_sy @ rho.conj() @ sy_sy
        R = rho @ rho_tilde
        evals = np.linalg.eigvals(R)
        lambdas = np.sort(np.sqrt(np.maximum(0.0, evals.real)))[::-1]
        return float(max(0.0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]))

    def record_frame_pure(psi, step, t):
        """Compute entanglement from pure statevector via SVD."""
        M = psi.reshape(db, da)  # psi[ib*da + ia] → M[ib, ia]
        sv = np.linalg.svd(M, compute_uv=False)  # descending order
        probs = sv**2
        probs = probs[probs > 1e-15]
        entropy_a = float(-np.sum(probs * np.log(probs)))
        mutual_info = 2.0 * entropy_a  # pure state: S_A = S_B
        concurrence = float(2.0 * sv[0] * sv[1]) if (n_a == 1 and n_b == 1 and len(sv) >= 2) else -1.0
        sv_sum = float(np.sum(sv))
        log_neg = float(np.log2(max(1.0, sv_sum)))
        return {
            "step": step, "t": round(t, 10),
            "entropy_a": round(entropy_a, 10),
            "mutual_info": round(mutual_info, 10),
            "concurrence": round(concurrence, 10),
            "log_negativity": round(log_neg, 10),
            "schmidt_values": [round(float(s), 10) for s in sv],
        }

    def record_frame_mixed(rho, step, t):
        """Compute entanglement from density matrix via partial trace."""
        rho_a = _partial_trace_b(rho)
        rho_b = _partial_trace_a(rho)
        entropy_a = _entropy(rho_a)
        entropy_b = _entropy(rho_b)
        entropy_total = _entropy(rho)
        mutual_info = entropy_a + entropy_b - entropy_total
        rho_pt = _partial_transpose_a(rho)
        pt_evals = np.linalg.eigvalsh(rho_pt)
        trace_norm = float(np.sum(np.abs(pt_evals)))
        log_neg = float(np.log2(max(1.0, trace_norm)))
        sv_a = np.sqrt(np.maximum(0.0, np.linalg.eigvalsh(rho_a)))[::-1]
        concurrence = _wootters_concurrence(rho) if n_qubits == 2 else -1.0
        return {
            "step": step, "t": round(t, 10),
            "entropy_a": round(entropy_a, 10),
            "mutual_info": round(mutual_info, 10),
            "concurrence": round(float(concurrence), 10),
            "log_negativity": round(log_neg, 10),
            "schmidt_values": [round(float(s), 10) for s in sv_a],
        }

    total_steps = max(1, round(total_time / dt))
    stride = max(1, total_steps // max(1, samples - 1))

    is_mixed = len(Lk_list) > 0

    if is_mixed:
        # Mixed: Lindblad RK4
        rho = np.zeros((dim, dim), dtype=complex)
        rho[0, 0] = 1.0
        LdagL_list = [L.conj().T @ L for L in Lk_list]

        def lindblad_rhs(r):
            dr = -1j * (H_mat @ r - r @ H_mat)
            for L, LdagL in zip(Lk_list, LdagL_list):
                dr += L @ r @ L.conj().T - 0.5 * (LdagL @ r + r @ LdagL)
            return dr

        def rk4_step(r):
            k1 = lindblad_rhs(r)
            k2 = lindblad_rhs(r + 0.5 * dt * k1)
            k3 = lindblad_rhs(r + 0.5 * dt * k2)
            k4 = lindblad_rhs(r + dt * k3)
            r_new = r + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
            return 0.5 * (r_new + r_new.conj().T)

        frames = [record_frame_mixed(rho, 0, 0.0)]
        for s in range(total_steps):
            rho = rk4_step(rho)
            t_s = (s + 1) * dt
            if (s + 1) % stride == 0 or s + 1 == total_steps:
                frames.append(record_frame_mixed(rho, s + 1, t_s))
    else:
        # Pure: exact matrix exponential
        evals, evecs = np.linalg.eigh(H_mat)
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1.0

        frames = [record_frame_pure(psi, 0, 0.0)]
        U_dt = evecs @ np.diag(np.exp(-1j * evals * dt)) @ evecs.conj().T
        for s in range(total_steps):
            psi = U_dt @ psi
            t_s = (s + 1) * dt
            if (s + 1) % stride == 0 or s + 1 == total_steps:
                frames.append(record_frame_pure(psi, s + 1, t_s))

    return {
        "n_qubits": n_qubits,
        "n_a": n_a,
        "n_b": n_b,
        "is_mixed": is_mixed,
        "source": source,
        "total_time": total_time,
        "frames": frames,
        "hamiltonian_terms": hamiltonian_terms,
        "collapse_ops": collapse_ops,
    }


def _entanglement_page(data: dict) -> str:
    """Render an animated HTML page for bipartite entanglement measures."""
    data_json = json.dumps(data, ensure_ascii=False)
    n_a = data.get("n_a", 1)
    n_b = data.get("n_b", 1)
    n_qubits = data.get("n_qubits", n_a + n_b)
    is_mixed = data.get("is_mixed", False)
    title = f"Entanglement: {n_qubits}-qubit ({n_a}|{n_b} cut)"
    mixed_label = "Mixed (Lindblad)" if is_mixed else "Pure (Schrödinger)"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif;padding:20px}}
h1{{font-size:1.1rem;font-weight:600;margin-bottom:12px;color:#58a6ff}}
.meta{{font-size:0.78rem;color:#8b949e;margin-bottom:16px}}
.controls{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:16px}}
.controls label{{font-size:0.8rem;color:#8b949e}}
button{{background:#21262d;border:1px solid #30363d;color:#e6edf3;
        padding:6px 14px;border-radius:6px;cursor:pointer;font-size:0.82rem}}
button:hover{{background:#30363d}}
input[type=range]{{width:200px;accent-color:#58a6ff}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;max-width:960px}}
.panel{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}}
.panel h2{{font-size:0.85rem;font-weight:600;color:#8b949e;margin-bottom:10px}}
canvas{{display:block;width:100%;height:200px}}
.tick{{font-size:9px;fill:#8b949e}}
.schmidt-bar{{display:flex;gap:3px;align-items:flex-end;height:80px;margin-top:8px}}
.sbar{{background:#58a6ff;border-radius:2px 2px 0 0;min-width:8px;transition:height 0.15s}}
.info{{font-size:0.75rem;color:#8b949e;margin-top:8px}}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">Type: {mixed_label} &nbsp;|&nbsp; Subsystem A: qubits [0,{n_a}) &nbsp;|&nbsp; Subsystem B: qubits [{n_a},{n_qubits})</div>
<div class="controls">
  <button id="btnPlay">&#9654; Play</button>
  <label>Step: <input type="range" id="slider" min="1" max="1" value="1"></label>
  <span id="timeLabel" style="font-size:0.8rem;color:#58a6ff"></span>
</div>
<div class="grid">
  <div class="panel">
    <h2>S(ρ_A) — Entanglement Entropy</h2>
    <canvas id="cvEntropy"></canvas>
  </div>
  <div class="panel">
    <h2>Mutual Information I(A:B)</h2>
    <canvas id="cvMutual"></canvas>
  </div>
  <div class="panel">
    <h2>Concurrence{"" if n_qubits == 2 else " (N/A — not 2-qubit)"} &amp; Log-Negativity</h2>
    <canvas id="cvConc"></canvas>
  </div>
  <div class="panel">
    <h2>Schmidt Spectrum</h2>
    <div class="schmidt-bar" id="schmidtBars"></div>
    <div class="info" id="schmidtInfo"></div>
  </div>
</div>
<script>
const DATA = {data_json};
const frames = DATA.frames;
const N = frames.length;

const slider = document.getElementById("slider");
const timeLabel = document.getElementById("timeLabel");
slider.max = N;

// ── Canvas chart helper ──────────────────────────────────────────────────────
function initCanvas(id) {{
  const cv = document.getElementById(id);
  cv.width = cv.offsetWidth || 440;
  cv.height = 200;
  return cv.getContext("2d");
}}

function drawChart(ctx, series, color, label, yMin, yMax) {{
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const pad = {{l:40,r:12,t:10,b:30}};
  const w = W - pad.l - pad.r, h = H - pad.t - pad.b;
  ctx.clearRect(0, 0, W, H);
  // axes
  ctx.strokeStyle = "#30363d"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l,pad.t); ctx.lineTo(pad.l,pad.t+h); ctx.lineTo(pad.l+w,pad.t+h); ctx.stroke();
  // tick labels
  ctx.fillStyle = "#8b949e"; ctx.font = "9px system-ui"; ctx.textAlign = "right";
  const range = yMax - yMin || 1;
  for (let i = 0; i <= 4; i++) {{
    const v = yMin + (range * i / 4);
    const y = pad.t + h - (v - yMin) / range * h;
    ctx.fillText(v.toFixed(2), pad.l - 3, y + 3);
    ctx.strokeStyle = "#21262d"; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l+w, y); ctx.stroke();
  }}
  // line
  if (series.length < 2) return;
  ctx.strokeStyle = color; ctx.lineWidth = 1.8; ctx.beginPath();
  series.forEach((v, i) => {{
    const x = pad.l + i / (series.length - 1) * w;
    const y = pad.t + h - (v - yMin) / range * h;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }});
  ctx.stroke();
  // label
  ctx.fillStyle = color; ctx.font = "9px system-ui"; ctx.textAlign = "left";
  ctx.fillText(label, pad.l + 2, pad.t + 12);
}}

const entropy_series = frames.map(f => f.entropy_a);
const mutual_series = frames.map(f => f.mutual_info);
const conc_series = frames.map(f => f.concurrence);
const lneg_series = frames.map(f => f.log_negativity);

const e_max = Math.max(...entropy_series, 0.01);
const m_max = Math.max(...mutual_series, 0.01);
const c_max = Math.max(...conc_series.filter(v=>v>=0), ...lneg_series, 0.01);

const ctxE = initCanvas("cvEntropy");
const ctxM = initCanvas("cvMutual");
const ctxC = initCanvas("cvConc");

function drawAll() {{
  drawChart(ctxE, entropy_series, "#58a6ff", "S(ρ_A) [nats]", 0, e_max);
  drawChart(ctxM, mutual_series, "#3fb950", "I(A:B) [nats]", 0, m_max);
  const conc_valid = conc_series[0] >= 0;
  if (conc_valid) {{
    drawChart(ctxC, conc_series, "#f0883e", "Concurrence", 0, c_max);
  }} else {{
    drawChart(ctxC, lneg_series, "#bc8cff", "Log-Neg", 0, c_max);
  }}
}}
drawAll();

// ── Cursor lines ─────────────────────────────────────────────────────────────
function drawCursor(ctx, idx) {{
  const W = ctx.canvas.width, H = ctx.canvas.height;
  const pad = {{l:40,r:12,t:10,b:30}};
  const w = W - pad.l - pad.r;
  const x = pad.l + idx / (N - 1) * w;
  ctx.strokeStyle = "rgba(255,255,255,0.35)"; ctx.lineWidth = 1;
  ctx.setLineDash([3,3]);
  ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, pad.t + H - pad.t - pad.b); ctx.stroke();
  ctx.setLineDash([]);
}}

// ── Schmidt bars ──────────────────────────────────────────────────────────────
function updateSchmidt(f) {{
  const container = document.getElementById("schmidtBars");
  const sv = f.schmidt_values || [];
  const maxSv = Math.max(...sv, 0.01);
  container.innerHTML = sv.map((s, i) => {{
    const h = Math.round(s / maxSv * 76);
    const col = `hsl(${{200 + i*30}},70%,55%)`;
    return `<div class="sbar" style="height:${{h}}px;background:${{col}}" title="σ${{i}}=${{s.toFixed(4)}}"></div>`;
  }}).join("");
  const rank = sv.filter(s => s > 1e-4).length;
  document.getElementById("schmidtInfo").textContent =
    `Schmidt rank ≈ ${{rank}} · max σ = ${{sv[0]?.toFixed(4) ?? "—"}}`;
}}

// ── Render ────────────────────────────────────────────────────────────────────
function render(idx) {{
  const f = frames[idx - 1];
  timeLabel.textContent = `t = ${{f.t.toFixed(3)}}  S = ${{f.entropy_a.toFixed(4)}}  I = ${{f.mutual_info.toFixed(4)}}`;
  drawAll();
  [ctxE, ctxM, ctxC].forEach(c => drawCursor(c, idx - 1));
  updateSchmidt(f);
}}
render(1);
slider.addEventListener("input", () => render(+slider.value));

// ── Play/pause ────────────────────────────────────────────────────────────────
let timer = null;
document.getElementById("btnPlay").addEventListener("click", function() {{
  if (timer) {{ clearInterval(timer); timer = null; this.textContent = "▶ Play"; return; }}
  this.textContent = "⏸ Pause";
  let idx = +slider.value;
  timer = setInterval(() => {{
    idx = idx >= N ? 1 : idx + 1;
    slider.value = idx;
    render(idx);
    if (idx >= N) {{ clearInterval(timer); timer = null; document.getElementById("btnPlay").textContent = "▶ Play"; }}
  }}, 80);
}});
</script>
</body>
</html>"""


def _wigner_frames(
    nx: int,
    ny: int,
    total_time: float,
    dt: float,
    potential_type: str,
    samples: int,
) -> dict:
  """Python reference solver: Wigner quasi-probability and transport observables.

  Evolves a 2D Gaussian wavepacket with split-step FFT and, at each sample step,
  computes W(x, p_x) along y=y_mid, probability current J_x / J_y, and momentum
  expectations.  Returns a dict consumable by _wigner_page().
  """
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("Wigner visualization requires numpy") from ex

  pi = math.pi
  dx = 0.25
  dy = 0.25
  xs = np.array([(j - nx / 2.0) * dx for j in range(nx)])
  ys = np.array([(j - ny / 2.0) * dy for j in range(ny)])
  X, Y = np.meshgrid(xs, ys, indexing="ij")   # shape (nx, ny)

  # Initial Gaussian wavepacket: centred at x0 with initial momentum kx0.
  sigma = 1.5
  kx0 = 1.8
  x0, y0 = -4.0, 0.0
  psi: "np.ndarray" = (
      np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (4.0 * sigma * sigma))
      * np.exp(1j * kx0 * X)
  )
  nrm = float(np.sqrt(np.sum(np.abs(psi) ** 2) * dx * dy))
  psi = psi / nrm

  # Static potential.
  if potential_type == "barrier":
    V = np.where(np.abs(X) < 0.4, 2.5, 0.0).astype(float)
  elif potential_type == "harmonic":
    V = 0.05 * (X * X + Y * Y)
  else:
    V = np.zeros_like(X, dtype=float)

  # Spectral propagators (mass=1, hbar=1).
  kx_arr = 2.0 * pi * np.fft.fftfreq(nx, d=dx)
  ky_arr = 2.0 * pi * np.fft.fftfreq(ny, d=dy)
  KX, KY = np.meshgrid(kx_arr, ky_arr, indexing="ij")
  kinetic_half = np.exp(-1j * 0.5 * dt * (KX * KX + KY * KY))
  potential_full = np.exp(-1j * dt * V)

  n_steps = max(1, int(round(total_time / dt)))
  stride = max(1, n_steps // samples)
  ny_mid = ny // 2
  N = nx  # Wigner grid is square: N x N

  def _record(step: int, t: float, psi_now: "np.ndarray") -> dict:
    psi_row = psi_now[:, ny_mid]  # shape (nx,)

    # Wigner function W(x_j, p_k).
    ls = np.arange(N)
    signs = np.where(ls % 2 == 0, 1.0, -1.0)
    W = np.empty((N, N), dtype=float)
    for j in range(N):
      idx_plus = (j + ls) % N
      idx_minus = (j - ls) % N
      a_prime = psi_row[idx_plus] * np.conj(psi_row[idx_minus]) * signs
      w_fft = np.fft.fft(a_prime)
      W[j, :] = (dx / pi) * np.real(w_fft)

    # Probability current J_x, J_y via spectral differentiation.
    psi_k = np.fft.fft2(psi_now)
    Jx = np.imag(np.conj(psi_now) * np.fft.ifft2(1j * KX * psi_k))  # shape (nx, ny)
    Jy = np.imag(np.conj(psi_now) * np.fft.ifft2(1j * KY * psi_k))

    # Transport: integrate J_x at x=1/4*L and x=3/4*L.
    x_left = N // 4
    x_right = (3 * N) // 4
    transmission = float(np.sum(Jx[x_right, :]) * dy)
    reflection = float(-np.sum(Jx[x_left, :]) * dy)

    # Momentum-space expectation values.
    dens_k = np.abs(psi_k) ** 2
    norm_k = float(np.sum(dens_k))
    inv_nk = 1.0 / norm_k if norm_k > 1e-30 else 0.0
    px_mean = float(np.sum(KX * dens_k)) * inv_nk
    py_mean = float(np.sum(KY * dens_k)) * inv_nk
    p2 = float(np.sum((KX * KX + KY * KY) * dens_k)) * inv_nk
    ke = p2 / 2.0  # mass=1

    # p_min / p_max for Wigner axis (k=0 is leftmost momentum bin).
    p_min = -0.5 * pi / dx
    p_max = (0.5 * N - 1.0) * pi / (N * dx)

    return {
        "step": step,
        "time": float(t),
        "nx": N,
        "np": N,
        "x_min": float(xs[0]),
        "x_max": float(xs[-1]),
        "p_min": p_min,
        "p_max": p_max,
        # W[j,k] row-major → wigner_flat[j*np+k]
        "wigner_flat": W.flatten().tolist(),
        # current_x/y: C++ layout [y*nx + x] = Jx.T.flatten()
        "current_x_flat": Jx.T.flatten().tolist(),
        "current_y_flat": Jy.T.flatten().tolist(),
        "transmission": transmission,
        "reflection": reflection,
        "px_mean": px_mean,
        "py_mean": py_mean,
        "kinetic_energy": ke,
    }

  frames = [_record(0, 0.0, psi)]
  for s in range(1, n_steps + 1):
    # Strang splitting: T/2 → V → T/2.
    psi_k2 = np.fft.fft2(psi) * kinetic_half
    psi = np.fft.ifft2(psi_k2) * potential_full
    psi_k2 = np.fft.fft2(psi) * kinetic_half
    psi = np.fft.ifft2(psi_k2)
    # Normalize.
    nrm2 = float(np.sqrt(np.sum(np.abs(psi) ** 2) * dx * dy))
    if nrm2 > 1e-30:
      psi = psi / nrm2
    if s % stride == 0 or s == n_steps:
      frames.append(_record(s, s * dt, psi))

  return {
      "frames": frames,
      "potential_type": potential_type,
      "nx": nx,
      "ny": ny,
      "x_min": float(xs[0]),
      "x_max": float(xs[-1]),
  }


def _wigner_page(data: dict) -> str:
  """Generate a self-contained HTML page visualising the Wigner function and transport."""
  frames = data["frames"]
  if not frames:
    return "<html><body><p>No Wigner frames to display.</p></body></html>"

  potential_type = data.get("potential_type", "unknown")
  frames_json = json.dumps(frames, separators=(",", ":"))

  # Build per-frame scalar time series for the chart panels.
  times_json = json.dumps([f["time"] for f in frames], separators=(",", ":"))
  px_json    = json.dumps([f["px_mean"] for f in frames], separators=(",", ":"))
  ke_json    = json.dumps([f["kinetic_energy"] for f in frames], separators=(",", ":"))
  trans_json = json.dumps([f["transmission"] for f in frames], separators=(",", ":"))
  refl_json  = json.dumps([f["reflection"] for f in frames], separators=(",", ":"))

  nx_val = frames[0].get("nx", 64)
  np_val = frames[0].get("np", 64)
  x_min  = frames[0].get("x_min", -8.0)
  x_max  = frames[0].get("x_max",  8.0)
  p_min  = frames[0].get("p_min", -6.3)
  p_max  = frames[0].get("p_max",  6.3)

  return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>QuantX – Wigner Function ({potential_type})</title>
<style>
  body{{margin:0;padding:12px 16px;font-family:monospace;font-size:12px;background:#111;color:#ddd}}
  h2{{margin:4px 0 10px;color:#7cf;font-size:14px}}
  .row{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:12px}}
  .panel{{display:flex;flex-direction:column;align-items:center}}
  .panel label{{font-size:11px;color:#aaa;margin-bottom:3px}}
  canvas{{background:#000;border:1px solid #333;display:block}}
  .slider-row{{display:flex;align-items:center;gap:10px;margin-bottom:10px}}
  input[type=range]{{width:380px;accent-color:#7cf}}
  #frame-info{{color:#7cf;min-width:120px}}
  svg{{overflow:visible}}
</style>
</head>
<body>
<h2>Wigner Quasi-Probability  ·  {potential_type} potential</h2>

<div class="slider-row">
  <label>frame:</label>
  <input type="range" id="sl" min="0" max="{len(frames)-1}" value="0" step="1">
  <span id="frame-info">t = 0.000</span>
</div>

<div class="row">
  <div class="panel">
    <label>W(x, p&#x2093;) – Wigner quasi-probability  (red&gt;0 / blue&lt;0)</label>
    <canvas id="cv_wigner" width="{nx_val * 4}" height="{np_val * 4}"></canvas>
  </div>
  <div class="panel">
    <label>J&#x2093;(x, y) – x-probability current  (red&gt;0 / blue&lt;0)</label>
    <canvas id="cv_jx" width="{nx_val * 4}" height="{nx_val * 4}"></canvas>
  </div>
</div>

<div class="row">
  <div class="panel">
    <label>&#x27E8;p&#x2093;&#x27E9;(t) and kinetic energy K(t)</label>
    <svg id="svg_mom" width="340" height="150"></svg>
  </div>
  <div class="panel">
    <label>Transmission T(t) and Reflection R(t)</label>
    <svg id="svg_tr" width="340" height="150"></svg>
  </div>
</div>

<div style="font-size:10px;color:#555;margin-top:4px">
  grid {nx_val}×{np_val} &nbsp;|&nbsp;
  x ∈ [{x_min:.2f}, {x_max:.2f}] &nbsp;|&nbsp;
  p ∈ [{p_min:.2f}, {p_max:.2f}]
</div>

<script>
const FRAMES = {frames_json};
const TIMES  = {times_json};
const PX_MEAN = {px_json};
const KE     = {ke_json};
const TRANS  = {trans_json};
const REFL   = {refl_json};

const cvW  = document.getElementById("cv_wigner");
const ctxW = cvW.getContext("2d");
const cvJ  = document.getElementById("cv_jx");
const ctxJ = cvJ.getContext("2d");

function divergingColor(val, maxAbs) {{
  const t = Math.min(1.0, Math.abs(val) / (maxAbs + 1e-30));
  const bg = 255;
  if (val >= 0) {{
    return [Math.round(bg * (1-t) + 220 * t),
            Math.round(bg * (1-t) + 30  * t),
            Math.round(bg * (1-t) + 30  * t)];
  }} else {{
    return [Math.round(bg * (1-t) + 30  * t),
            Math.round(bg * (1-t) + 30  * t),
            Math.round(bg * (1-t) + 220 * t)];
  }}
}}

function drawWigner(frame) {{
  const nx = frame.nx, np = frame.np;
  const W  = frame.wigner_flat;
  let maxW = 0;
  for (let v of W) maxW = Math.max(maxW, Math.abs(v));
  const off = document.createElement("canvas");
  off.width = nx; off.height = np;
  const offCtx = off.getContext("2d");
  const img = offCtx.createImageData(nx, np);
  for (let iy = 0; iy < np; iy++) {{
    const k = np - 1 - iy;   // flip: p_max at top
    for (let ix = 0; ix < nx; ix++) {{
      const w = W[ix * np + k];
      const [r,g,b] = divergingColor(w, maxW);
      const pi = 4 * (iy * nx + ix);
      img.data[pi]=r; img.data[pi+1]=g; img.data[pi+2]=b; img.data[pi+3]=255;
    }}
  }}
  offCtx.putImageData(img, 0, 0);
  ctxW.imageSmoothingEnabled = false;
  ctxW.drawImage(off, 0, 0, cvW.width, cvW.height);
}}

function drawCurrentX(frame) {{
  const nx = frame.nx, ny_size = frame.current_x_flat.length / nx;
  const J  = frame.current_x_flat;
  let maxJ = 0;
  for (let v of J) maxJ = Math.max(maxJ, Math.abs(v));
  const off = document.createElement("canvas");
  off.width = nx; off.height = ny_size;
  const offCtx = off.getContext("2d");
  const img = offCtx.createImageData(nx, ny_size);
  for (let iy = 0; iy < ny_size; iy++) {{
    const iy_flip = ny_size - 1 - iy;
    for (let ix = 0; ix < nx; ix++) {{
      const j_val = J[iy_flip * nx + ix];  // layout: [y*nx + x]
      const [r,g,b] = divergingColor(j_val, maxJ);
      const pi = 4 * (iy * nx + ix);
      img.data[pi]=r; img.data[pi+1]=g; img.data[pi+2]=b; img.data[pi+3]=255;
    }}
  }}
  offCtx.putImageData(img, 0, 0);
  ctxJ.imageSmoothingEnabled = false;
  ctxJ.drawImage(off, 0, 0, cvJ.width, cvJ.height);
}}

// ── SVG line charts ────────────────────────────────────────────────────────
function makeSVGLine(svgId, series, yLabel) {{
  const svg = document.getElementById(svgId);
  const W = +svg.getAttribute("width"), H = +svg.getAttribute("height");
  const mg = {{l:44,r:8,t:8,b:26}};
  const pw = W - mg.l - mg.r, ph = H - mg.t - mg.b;

  // Compute ranges.
  const allT = TIMES;
  const tMin = allT[0], tMax = allT[allT.length-1];
  let yMin = Infinity, yMax = -Infinity;
  for (const s of series) for (const v of s.vals) {{ yMin=Math.min(yMin,v); yMax=Math.max(yMax,v); }}
  const yPad = (yMax - yMin) * 0.1 || 0.1;
  yMin -= yPad; yMax += yPad;
  const tRange = tMax - tMin || 1, yRange = yMax - yMin;

  function tx(t){{ return mg.l + (t - tMin) / tRange * pw; }}
  function ty(y){{ return mg.t + (1 - (y - yMin) / yRange) * ph; }}

  let html = `<rect x="${{mg.l}}" y="${{mg.t}}" width="${{pw}}" height="${{ph}}" fill="#1a1a2e" stroke="#333"/>`;

  // Grid lines.
  for (let i=0;i<=4;i++) {{
    const yv = yMin + i * yRange/4;
    const yp = ty(yv);
    html += `<line x1="${{mg.l}}" x2="${{mg.l+pw}}" y1="${{yp}}" y2="${{yp}}" stroke="#333" stroke-dasharray="3,3"/>`;
    html += `<text x="${{mg.l-4}}" y="${{yp+4}}" text-anchor="end" fill="#888" font-size="9">${{yv.toFixed(2)}}</text>`;
  }}

  // Zero line if zero is in range.
  if (yMin < 0 && yMax > 0) {{
    html += `<line x1="${{mg.l}}" x2="${{mg.l+pw}}" y1="${{ty(0)}}" y2="${{ty(0)}}" stroke="#555" stroke-width="1"/>`;
  }}

  // Series.
  const colors = ["#f55","#55f","#5f5","#fa0"];
  for (let si=0; si<series.length; si++) {{
    const s = series[si];
    const pts = TIMES.map((t,i)=>`${{tx(t).toFixed(1)}},${{ty(s.vals[i]).toFixed(1)}}`).join(" ");
    html += `<polyline points="${{pts}}" fill="none" stroke="${{colors[si]}}" stroke-width="1.5"/>`;
    html += `<text x="${{mg.l+pw-4}}" y="${{mg.t+14+si*14}}" text-anchor="end" fill="${{colors[si]}}" font-size="9">${{s.name}}</text>`;
  }}

  // Current time marker (updated in render()).
  html += `<line id="${{svgId}}_tm" x1="${{mg.l}}" x2="${{mg.l}}" y1="${{mg.t}}" y2="${{mg.t+ph}}" stroke="#7cf" stroke-width="1" stroke-dasharray="4,3"/>`;

  // Axes.
  html += `<line x1="${{mg.l}}" x2="${{mg.l+pw}}" y1="${{mg.t+ph}}" y2="${{mg.t+ph}}" stroke="#666"/>`;
  html += `<line x1="${{mg.l}}" x2="${{mg.l}}" y1="${{mg.t}}" y2="${{mg.t+ph}}" stroke="#666"/>`;
  html += `<text x="${{mg.l+pw/2}}" y="${{H-4}}" text-anchor="middle" fill="#888" font-size="9">t</text>`;
  html += `<text x="10" y="${{mg.t+ph/2}}" text-anchor="middle" transform="rotate(-90,10,${{mg.t+ph/2}})" fill="#888" font-size="9">${{yLabel}}</text>`;

  svg.innerHTML = html;
}}

makeSVGLine("svg_mom", [{{name:"⟨px⟩", vals:PX_MEAN}}, {{name:"K.E.", vals:KE}}], "energy");
makeSVGLine("svg_tr",  [{{name:"T", vals:TRANS}}, {{name:"R", vals:REFL}}], "flux");

function updateTimeLine(svgId, t) {{
  const svg = document.getElementById(svgId);
  const W = +svg.getAttribute("width");
  const mg = {{l:44,r:8,t:8,b:26}};
  const pw = W - mg.l - mg.r;
  const tMin = TIMES[0], tMax = TIMES[TIMES.length-1], tRange = tMax-tMin||1;
  const xp = (mg.l + (t-tMin)/tRange*pw).toFixed(1);
  const el = document.getElementById(svgId+"_tm");
  if (el) {{ el.setAttribute("x1",xp); el.setAttribute("x2",xp); }}
}}

function render(idx) {{
  const frame = FRAMES[idx];
  document.getElementById("frame-info").textContent = `t = ${{frame.time.toFixed(3)}}`;
  drawWigner(frame);
  drawCurrentX(frame);
  updateTimeLine("svg_mom", frame.time);
  updateTimeLine("svg_tr",  frame.time);
}}

document.getElementById("sl").addEventListener("input", function() {{
  render(+this.value);
}});

render(0);
</script>
</body>
</html>
"""


def _rabi_data(engine, samples: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
  """Collect the simulation and analytic traces for the Rabi demo."""
  pts_sim = []
  pts_ref = []
  for i in range(samples):
    t = math.pi * i / max(1, samples - 1)
    problem = {
        "qubits": 1,
        "equation": "Schrodinger",
        "total_time": t,
        "dt": 0.01,
        "method": "krylov",
        "krylov_dim": 8,
        "hamiltonian_terms": [{"label": "X0", "coefficient": 1.0}],
    }
    result = engine.run(problem, {"enable_checkpoint": False})
    pts_sim.append((t, float(result.get("observables", {}).get("population_1", 0.0))))
    pts_ref.append((t, math.sin(t) ** 2))
  return pts_sim, pts_ref


def _mps_data(engine, samples: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
  """Collect the bond-dimension and truncation traces for the MPS demo."""
  bond = []
  trunc = []
  for i in range(samples):
    t = 2.0 * i / max(1, samples - 1)
    problem = {
        "qubits": 64,
        "equation": "Schrodinger",
        "total_time": t,
        "dt": 0.05,
        "method": "tebd",
        "mps_max_bond": 32,
        "mps_trunc_tol": 1e-6,
        "expected_circuit_depth": 200,
        "hamiltonian_terms": [{"label": "X0X1", "coefficient": 1.0}, {"label": "Z0", "coefficient": 0.2}],
    }
    result = engine.run(problem, {"enable_checkpoint": False})
    obs = result.get("observables", {})
    bond.append((t, float(obs.get("bond_dim", 0.0))))
    trunc.append((t, float(obs.get("truncation_error", 0.0))))
  return bond, trunc


