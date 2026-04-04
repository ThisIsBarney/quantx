from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from .common import (
    _make_engine,
    _resolve_continuous_payload,
    _animated_page,
    _attach_continuous_metadata,
    _user_cli_error,
    _require_list,
    _reshape_flat_to_rows,
    _reshape_flat_points,
    _normalize_engine_export_payload,
)


def _spatial_wavepacket_frames(
    samples_t: int, samples_x: int, potential_type: str, source: str = "python-reference"
) -> dict:
  """Build a 1D reference solution for the spatial wavepacket animation."""
  # Finite-difference + Crank-Nicolson for 1D Schrodinger equation (hbar=m=1).
  x_min, x_max = -12.0, 12.0
  t_min, t_max = 0.0, 4.0
  n = max(80, samples_x)
  nt = max(20, samples_t)
  xs = [x_min + (x_max - x_min) * i / (n - 1) for i in range(n)]
  dx = (x_max - x_min) / (n - 1)
  dt = (t_max - t_min) / (nt - 1)

  # Initial right-moving Gaussian wavepacket.
  x0, sigma0, k0 = -6.0, 0.7, 2.8
  psi = []
  for x in xs:
    amp = math.exp(-((x - x0) ** 2) / (4.0 * sigma0 * sigma0))
    phase = complex(math.cos(k0 * x), math.sin(k0 * x))
    psi.append(amp * phase)

  # Potential profiles for different spatial scenarios.
  if potential_type == "free":
    potential = [0.0 for _ in xs]
  elif potential_type == "harmonic":
    omega = 0.35
    potential = [0.5 * (omega * omega) * (x * x) for x in xs]
  elif potential_type == "well":
    wall = 5.0
    potential = [0.0 if abs(x) <= 6.0 else wall for x in xs]
  else:  # barrier
    v0, vb_w = 0.8, 0.7
    potential = [v0 * math.exp(-(x * x) / (2.0 * vb_w * vb_w)) for x in xs]

  # Tridiagonal coefficients for A psi_{n+1} = d (interior points only).
  off_a = complex(0.0, -dt / (4.0 * dx * dx))
  off_b = complex(0.0, dt / (4.0 * dx * dx))
  diag_a = []
  diag_b = []
  for j in range(1, n - 1):
    h_diag = (1.0 / (dx * dx)) + potential[j]
    diag_a.append(complex(1.0, 0.5 * dt * h_diag))
    diag_b.append(complex(1.0, -0.5 * dt * h_diag))

  def solve_tridiagonal(a: complex, b_diag: list[complex], c: complex, d: list[complex]) -> list[complex]:
    m = len(d)
    cp = [0j] * m
    dp = [0j] * m
    cp[0] = c / b_diag[0]
    dp[0] = d[0] / b_diag[0]
    for i in range(1, m):
      denom = b_diag[i] - a * cp[i - 1]
      cp[i] = (c / denom) if i < m - 1 else 0j
      dp[i] = (d[i] - a * dp[i - 1]) / denom
    x = [0j] * m
    x[-1] = dp[-1]
    for i in range(m - 2, -1, -1):
      x[i] = dp[i] - cp[i] * x[i + 1]
    return x

  def normalize(arr: list[complex]) -> None:
    s = sum((z.real * z.real + z.imag * z.imag) for z in arr) * dx
    if s > 0.0:
      inv = 1.0 / math.sqrt(s)
      for i in range(len(arr)):
        arr[i] *= inv

  normalize(psi)
  frames = []
  n_particles = 56

  for it in range(nt):
    t = t_min + (t_max - t_min) * it / max(1, nt - 1)
    density = [(z.real * z.real + z.imag * z.imag) for z in psi]
    phase = [math.atan2(z.imag, z.real) for z in psi]
    z_norm = sum(density) * dx
    if z_norm > 0.0:
      density = [p / z_norm for p in density]

    exp_x = sum(x * p for x, p in zip(xs, density)) * dx
    cdf, acc = [], 0.0
    for p in density:
      acc += p * dx
      cdf.append(acc)
    cloud = []
    j = 0
    for i in range(n_particles):
      q = (i + 0.5) / n_particles
      while j < len(cdf) - 1 and cdf[j] < q:
        j += 1
      cloud.append(xs[j])
    frames.append({"t": t, "density": density, "phase": phase, "exp_x": exp_x, "cloud": cloud})

    if it == nt - 1:
      break

    # Build RHS d = B * psi_n (Dirichlet boundary psi[0]=psi[-1]=0).
    rhs = []
    for k in range(n - 2):
      j = k + 1
      val = diag_b[k] * psi[j] + off_b * psi[j - 1] + off_b * psi[j + 1]
      rhs.append(val)
    interior = solve_tridiagonal(off_a, diag_a, off_a, rhs)
    psi = [0j] + interior + [0j]
    normalize(psi)

  return {
      "xs": xs,
      "frames": frames,
      "x_min": x_min,
      "x_max": x_max,
      "potential": potential,
      "potential_type": potential_type,
      "source": source,
  }


def _spatial_wavepacket_page(data: dict) -> str:
  """Render the 1D spatial wavepacket animation page."""
  payload = json.dumps(data, ensure_ascii=False)
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>QuantX Animation - Spatial Wavepacket ({data.get("potential_type","barrier")})</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 8px; }}
    button {{ padding: 4px 10px; }}
  </style>
</head>
<body>
  <h1>QuantX Animation - Spatial Wavepacket ({data.get("potential_type","barrier")})</h1>
  <p>1D Schrodinger solver (finite difference + Crank-Nicolson): showing spatial probability density |psi(x,t)|^2, phase color, potential profile, sampled particle cloud, and <b>data source:</b> {data.get("source","python-reference")}.</p>
  <svg id="svg" viewBox="0 0 1000 520" width="1000" height="520" xmlns="http://www.w3.org/2000/svg"></svg>
  <div class="controls">
    <button id="play">Play</button>
    <button id="pause">Pause</button>
    <input id="slider" type="range" min="1" step="1" style="width:520px"/>
    <span id="label"></span>
  </div>
  <script>
  const DATA = {payload};
  const svg = document.getElementById("svg");
  const play = document.getElementById("play");
  const pause = document.getElementById("pause");
  const slider = document.getElementById("slider");
  const label = document.getElementById("label");

  const W=1000,H=520,L=80,R=30,T=40,B=120;
  const PW=W-L-R, PH=H-T-B;
  const xMin=DATA.x_min, xMax=DATA.x_max;
  const yMax=Math.max(...DATA.frames.flatMap(f=>f.density)) * 1.1;
  const vMax=Math.max(1e-12, ...DATA.potential.map(v=>Math.abs(v)));

  const sx=x=>L+(x-xMin)/(xMax-xMin)*PW;
  const sy=y=>T+(1-y/yMax)*PH;

  function el(tag, attrs) {{
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k,v] of Object.entries(attrs||{{}})) e.setAttribute(k,v);
    return e;
  }}
  function phaseColor(phi) {{
    const h = ((phi + Math.PI) / (2*Math.PI)) * 360;
    return `hsl(${{h}},85%,50%)`;
  }}

  svg.appendChild(el("line",{{x1:L,y1:T+PH,x2:L+PW,y2:T+PH,stroke:"#333","stroke-width":"1.5"}}));
  svg.appendChild(el("line",{{x1:L,y1:T,x2:L,y2:T+PH,stroke:"#333","stroke-width":"1.5"}}));
  for(let i=0;i<=5;i++) {{
    const gx=L+PW*i/5, gy=T+PH*i/5;
    svg.appendChild(el("line",{{x1:gx,y1:T,x2:gx,y2:T+PH,stroke:"#ddd"}}));
    svg.appendChild(el("line",{{x1:L,y1:gy,x2:L+PW,y2:gy,stroke:"#ddd"}}));
    const tx=el("text",{{x:gx,y:T+PH+20,"text-anchor":"middle","font-size":"12"}}); tx.textContent=(xMin+(xMax-xMin)*i/5).toPrecision(3); svg.appendChild(tx);
    const ty=el("text",{{x:L-8,y:gy+4,"text-anchor":"end","font-size":"12"}}); ty.textContent=(yMax-yMax*i/5).toPrecision(3); svg.appendChild(ty);
  }}
  const xl=el("text",{{x:L+PW/2,y:H-26,"text-anchor":"middle","font-size":"12"}}); xl.textContent="position x"; svg.appendChild(xl);
  const yl=el("text",{{x:20,y:T+PH/2,transform:`rotate(-90 20 ${{T+PH/2}})`,"text-anchor":"middle","font-size":"12"}}); yl.textContent="|psi(x,t)|^2"; svg.appendChild(yl);

  const probPath = el("path",{{fill:"none",stroke:"#0b84f3","stroke-width":"2.5"}}); svg.appendChild(probPath);
  const potPath = el("path",{{fill:"none",stroke:"#8e44ad","stroke-width":"2","stroke-dasharray":"5 4"}}); svg.appendChild(potPath);
  const expLine = el("line",{{x1:L,y1:T,x2:L,y2:T+PH,stroke:"#f39c12","stroke-width":"2","stroke-dasharray":"6 4"}}); svg.appendChild(expLine);
  const phaseDots = [];
  for(let i=0;i<DATA.xs.length;i+=4) {{
    const c = el("circle",{{cx:L,cy:T+PH,r:2.1,fill:"#999"}});
    phaseDots.push(c); svg.appendChild(c);
  }}
  const cloudDots = [];
  for(let i=0;i<56;i++) {{
    const c = el("circle",{{cx:L,cy:T+PH+55,r:4,fill:"#2e8b57",opacity:"0.78"}});
    cloudDots.push(c); svg.appendChild(c);
  }}
  const cloudAxis = el("line",{{x1:L,y1:T+PH+55,x2:L+PW,y2:T+PH+55,stroke:"#555","stroke-width":"1"}}); svg.appendChild(cloudAxis);
  const cloudLabel = el("text",{{x:L,y:T+PH+84,"font-size":"12"}}); cloudLabel.textContent="sampled position cloud (from |psi|^2)"; svg.appendChild(cloudLabel);
  const potLabel = el("text",{{x:L+PW-6,y:T+16,"text-anchor":"end","font-size":"12",fill:"#8e44ad"}}); potLabel.textContent="potential V(x) (scaled)"; svg.appendChild(potLabel);

  slider.max = String(DATA.frames.length);
  slider.value = "1";

  let timer = null;
  function render(frameIdx) {{
    const f = DATA.frames[frameIdx-1];
    const d = DATA.xs.map((x,i)=>`${{i===0?"M":"L"}}${{sx(x).toFixed(2)}} ${{sy(f.density[i]).toFixed(2)}}`).join(" ");
    probPath.setAttribute("d", d);
    const dp = DATA.xs.map((x,i)=>{{
      const pv = (DATA.potential[i] / vMax) * (0.85 * yMax);
      return `${{i===0?"M":"L"}}${{sx(x).toFixed(2)}} ${{sy(pv).toFixed(2)}}`;
    }}).join(" ");
    potPath.setAttribute("d", dp);
    const ex = sx(f.exp_x);
    expLine.setAttribute("x1", ex); expLine.setAttribute("x2", ex);

    let di = 0;
    for(let i=0;i<DATA.xs.length;i+=4) {{
      phaseDots[di].setAttribute("cx", sx(DATA.xs[i]));
      phaseDots[di].setAttribute("cy", sy(f.density[i]));
      phaseDots[di].setAttribute("fill", phaseColor(f.phase[i]));
      di++;
    }}
    for(let i=0;i<cloudDots.length;i++) {{
      cloudDots[i].setAttribute("cx", sx(f.cloud[i]));
    }}
    label.textContent = `frame ${{frameIdx}}/${{DATA.frames.length}}, t=${{f.t.toFixed(3)}}`;
    slider.value = String(frameIdx);
  }}
  function doPlay() {{
    if (timer) return;
    timer = setInterval(()=>{{
      let f = Number(slider.value);
      f = f >= DATA.frames.length ? 1 : f + 1;
      render(f);
    }}, 85);
  }}
  function doPause() {{
    if(!timer) return;
    clearInterval(timer);
    timer = null;
  }}
  play.onclick = doPlay;
  pause.onclick = doPause;
  slider.oninput = ()=>render(Number(slider.value));
  render(1);
  </script>
</body>
</html>"""


def _spatial_wavepacket_2d_frames(
    samples_t: int,
    nx: int,
    ny: int,
    potential_type: str,
    source: str = "python-reference",
    driving: dict | None = None,
) -> dict:
  """Build a 2D reference solution for the spatial wavepacket animation.

  driving: optional dict with keys type (sine/gaussian_pulse/chirp),
           amplitude, frequency, phase, t_center, t_width, spatial_profile.
  """
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("2D spatial visualization requires numpy") from ex

  x_min, x_max = -8.0, 8.0
  y_min, y_max = -8.0, 8.0
  t_min, t_max = 0.0, 3.0
  nx = max(32, nx)
  ny = max(32, ny)
  nt = max(16, samples_t)

  xs = np.linspace(x_min, x_max, nx)
  ys = np.linspace(y_min, y_max, ny)
  dx = float(xs[1] - xs[0])
  dy = float(ys[1] - ys[0])
  dt = (t_max - t_min) / (nt - 1)

  X, Y = np.meshgrid(xs, ys, indexing="xy")

  # Initial 2D Gaussian packet with momentum.
  x0, y0 = -4.5, -2.0
  sigma = 0.7
  kx0, ky0 = 2.2, 1.1
  envelope = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (4.0 * sigma * sigma))
  phase = np.exp(1j * (kx0 * X + ky0 * Y))
  psi = envelope * phase

  if potential_type == "free":
    V_static = np.zeros_like(X)
  elif potential_type == "harmonic":
    omega = 0.28
    V_static = 0.5 * (omega * omega) * (X * X + Y * Y)
  elif potential_type == "well":
    wall = 5.0
    V_static = np.where((np.abs(X) <= 6.0) & (np.abs(Y) <= 6.0), 0.0, wall)
  else:  # barrier
    V_static = 0.9 * np.exp(-(X * X) / (2.0 * 0.6 * 0.6))

  # Driving field spatial profile.
  has_driving = driving is not None and driving.get("type", "none") != "none" and driving.get("amplitude", 0.0) != 0.0
  if has_driving:
    sp = driving.get("spatial_profile", "uniform")
    if sp == "dipole_x":
      V_drive_spatial = X.copy()
    elif sp == "dipole_y":
      V_drive_spatial = Y.copy()
    else:
      V_drive_spatial = np.ones_like(X)

    def drive_envelope(t: float) -> float:
      A = driving.get("amplitude", 0.0)
      f = driving.get("frequency", 1.0)
      ph = driving.get("phase", 0.0)
      dtype = driving.get("type", "sine")
      if dtype == "sine":
        return A * math.cos(f * t + ph)
      if dtype == "gaussian_pulse":
        tc = driving.get("t_center", 0.0)
        tw = max(1e-30, driving.get("t_width", 1.0))
        return A * math.exp(-0.5 * ((t - tc) / tw) ** 2) * math.cos(f * t + ph)
      if dtype == "chirp":
        tw = driving.get("t_width", 0.0)  # chirp rate α
        return A * math.cos((f + 0.5 * tw * t) * t + ph)
      return 0.0
  else:
    def drive_envelope(t: float) -> float:
      return 0.0
    V_drive_spatial = np.zeros_like(X)

  kx_arr = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
  ky_arr = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
  KX, KY = np.meshgrid(kx_arr, ky_arr, indexing="xy")
  kinetic_phase = np.exp(-1j * 0.5 * (KX * KX + KY * KY) * dt)

  def normalize(a):
    nrm = float(np.sum(np.abs(a) ** 2) * dx * dy)
    if nrm > 0:
      a /= math.sqrt(nrm)
    return a

  psi = normalize(psi)
  frames = []
  dens_max = 1e-12

  for it in range(nt):
    t = t_min + (t_max - t_min) * it / max(1, nt - 1)
    dens = np.abs(psi) ** 2
    dens_max = max(dens_max, float(np.max(dens)))
    exp_x = float(np.sum(X * dens) * dx * dy)
    exp_y = float(np.sum(Y * dens) * dx * dy)
    drv_amp = drive_envelope(t)
    frame: dict = {
        "t": t,
        "density": dens.tolist(),
        "phase": np.angle(psi).tolist(),
        "exp_x": exp_x,
        "exp_y": exp_y,
    }
    if has_driving:
      frame["driving_amplitude"] = drv_amp
    frames.append(frame)
    if it == nt - 1:
      break

    # Strang splitting: V/2 -> T -> V/2 (time-centered potential at t+dt/2)
    t_mid = t + 0.5 * dt
    drv_mid = drive_envelope(t_mid)
    V_total = V_static + drv_mid * V_drive_spatial
    potential_half = np.exp(-1j * V_total * dt * 0.5)
    psi = potential_half * psi
    psi_k = np.fft.fft2(psi)
    psi = np.fft.ifft2(kinetic_phase * psi_k)
    psi = potential_half * psi
    psi = normalize(psi)

  out: dict = {
      "x_min": x_min,
      "x_max": x_max,
      "y_min": y_min,
      "y_max": y_max,
      "potential_type": potential_type,
      "frames": frames,
      "dens_max": dens_max,
      "has_phase": True,
      "source": source,
  }
  if has_driving:
    out["has_driving"] = True
    out["driving"] = driving
  return out


def _spatial_wavepacket_2d_page(data: dict) -> str:
  """Render the 2D spatial wavepacket heatmap animation page with phase toggle and colorbar."""
  payload = json.dumps(data, ensure_ascii=False)
  has_phase = bool(data.get("has_phase"))
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>QuantX Animation - 2D Spatial Wavepacket ({data.get("potential_type","barrier")})</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; background: #f8f8f8; }}
    h1 {{ font-size: 18px; margin-bottom: 4px; }}
    .row {{ display: flex; gap: 12px; align-items: flex-start; margin-top: 8px; }}
    .cv-wrap {{ position: relative; }}
    canvas#cv {{ border: 1px solid #aaa; display: block; }}
    canvas#colorbar {{ border: 1px solid #aaa; display: block; cursor: default; }}
    .cb-labels {{ display: flex; flex-direction: column; justify-content: space-between; font-size: 11px; color: #555; height: 400px; padding: 2px 0; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 8px; flex-wrap: wrap; }}
    button {{ padding: 4px 10px; }}
    select {{ padding: 3px 6px; }}
    label {{ font-size: 13px; }}
  </style>
</head>
<body>
  <h1>QuantX Animation - 2D Spatial Wavepacket ({data.get("potential_type","barrier")})</h1>
  <p style="font-size:13px;color:#555">Split-step Fourier: |ψ(x,y,t)|² heatmap. Source: <b>{data.get("source","python-reference")}</b>.</p>
  <div class="controls">
    <button id="play">Play</button>
    <button id="pause">Pause</button>
    <input id="slider" type="range" min="1" step="1" style="width:400px"/>
    <span id="label" style="font-size:13px"></span>
    {"<label>Color: <select id='colorMode'><option value='density'>Density |ψ|²</option><option value='phase'>Phase arg(ψ)</option></select></label>" if has_phase else ""}
  </div>
  <div class="row">
    <div class="cv-wrap">
      <canvas id="cv" width="512" height="512"></canvas>
    </div>
    <div>
      <canvas id="colorbar" width="28" height="400"></canvas>
    </div>
    <div class="cb-labels" id="cbLabels">
      <span>max</span><span></span><span>0</span>
    </div>
  </div>
  <script>
  const DATA = {payload};
  const cv = document.getElementById("cv");
  const ctx = cv.getContext("2d");
  const slider = document.getElementById("slider");
  const label = document.getElementById("label");
  const cbCanvas = document.getElementById("colorbar");
  const cbCtx = cbCanvas.getContext("2d");
  const cbLabels = document.getElementById("cbLabels");
  slider.max = String(DATA.frames.length);
  slider.value = "1";

  function colorMapDensity(v) {{
    const t = Math.max(0, Math.min(1, v));
    return [Math.floor(255*Math.sqrt(t)), Math.floor(255*t*t), Math.floor(255*(1-t)), 255];
  }}
  function colorMapPhase(phi) {{
    const h = ((phi + Math.PI) / (2*Math.PI)) * 360;
    const s = 0.88, l = 0.52, a = s * Math.min(l, 1-l);
    const f = n => Math.round(255*(l - a*Math.max(-1, Math.min((n+h/30)%12-3, 9-(n+h/30)%12, 1))));
    return [f(0), f(8), f(4), 255];
  }}

  function drawColorbar(mode) {{
    const h = cbCanvas.height;
    for (let y = 0; y < h; y++) {{
      const t = 1 - y/h;
      const rgba = (mode === "phase") ? colorMapPhase((t*2-1)*Math.PI) : colorMapDensity(t);
      cbCtx.fillStyle = `rgb(${{rgba[0]}},${{rgba[1]}},${{rgba[2]}})`;
      cbCtx.fillRect(0, y, cbCanvas.width, 1);
    }}
    if (mode === "phase") {{
      cbLabels.innerHTML = "<span>+π</span><span>0</span><span>−π</span>";
    }} else {{
      const densMax = Math.max(1e-12, DATA.dens_max||0);
      cbLabels.innerHTML = `<span>${{densMax.toExponential(1)}}</span><span></span><span>0</span>`;
    }}
  }}

  function render(frameIdx) {{
    const f = DATA.frames[frameIdx - 1];
    const mode = document.getElementById("colorMode")?.value || "density";
    const densMax = Math.max(1e-12, DATA.dens_max || 0);
    const ny = f.density.length;
    const nx = f.density[0].length;
    const img = ctx.createImageData(nx, ny);
    let p = 0;
    if (mode === "phase" && f.phase && DATA.has_phase) {{
      for (let j = 0; j < ny; j++) {{
        for (let i = 0; i < nx; i++) {{
          const c = colorMapPhase(f.phase[j][i]);
          img.data[p++]=c[0]; img.data[p++]=c[1]; img.data[p++]=c[2]; img.data[p++]=c[3];
        }}
      }}
    }} else {{
      for (let j = 0; j < ny; j++) {{
        for (let i = 0; i < nx; i++) {{
          const c = colorMapDensity(f.density[j][i] / densMax);
          img.data[p++]=c[0]; img.data[p++]=c[1]; img.data[p++]=c[2]; img.data[p++]=c[3];
        }}
      }}
    }}
    const tmp = document.createElement("canvas");
    tmp.width = nx; tmp.height = ny;
    tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(tmp, 0, 0, cv.width, cv.height);

    // Center-of-mass marker
    const sx = x => (x - DATA.x_min) / (DATA.x_max - DATA.x_min) * cv.width;
    const sy = y => cv.height - (y - DATA.y_min) / (DATA.y_max - DATA.y_min) * cv.height;
    ctx.beginPath();
    ctx.arc(sx(f.exp_x), sy(f.exp_y), 6, 0, Math.PI*2);
    ctx.fillStyle = "#fff";
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();

    // Axis tick labels
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.font = "11px sans-serif";
    ctx.fillText(DATA.x_min.toFixed(1), 3, cv.height-4);
    ctx.fillText(DATA.x_max.toFixed(1), cv.width-28, cv.height-4);
    ctx.save(); ctx.translate(12, cv.height/2); ctx.rotate(-Math.PI/2);
    ctx.fillText(`y:[${DATA.y_min.toFixed(1)},${DATA.y_max.toFixed(1)}]`, -40, 0);
    ctx.restore();

    label.textContent = `frame ${{frameIdx}}/${{DATA.frames.length}}, t=${{f.t.toFixed(3)}}`;
    slider.value = String(frameIdx);
    drawColorbar(mode);
  }}

  const colorModeEl = document.getElementById("colorMode");
  if (colorModeEl) colorModeEl.onchange = () => render(Number(slider.value));

  let timer = null;
  function doPlay() {{
    if (timer) return;
    timer = setInterval(() => {{
      let f = Number(slider.value);
      f = f >= DATA.frames.length ? 1 : f + 1;
      render(f);
    }}, 90);
  }}
  function doPause() {{
    if (!timer) return;
    clearInterval(timer);
    timer = null;
  }}
  document.getElementById("play").onclick = doPlay;
  document.getElementById("pause").onclick = doPause;
  slider.oninput = () => render(Number(slider.value));
  render(1);
  </script>
</body>
</html>"""


def _spatial_wavepacket_2d_dirac_frames(
    samples_t: int,
    nx: int,
    ny: int,
    potential_type: str,
    mass: float = 1.0,
    source: str = "python-reference",
) -> dict:
  """Build a 2D (2+1D) Dirac spinor reference solution using split-step Fourier.

  Two-component spinor: (ψ₀ up, ψ₁ down).  Algorithm matches continuous2d_backend.cpp
  evolve_dirac(): V/2 → T_Dirac(k-space) → V/2.
  """
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("Dirac 2D visualization requires numpy") from ex

  x_min, x_max = -8.0, 8.0
  y_min, y_max = -8.0, 8.0
  t_min, t_max = 0.0, 3.0
  nx = max(32, nx)
  ny = max(32, ny)
  nt = max(16, samples_t)

  xs = np.linspace(x_min, x_max, nx)
  ys = np.linspace(y_min, y_max, ny)
  dx = float(xs[1] - xs[0])
  dy = float(ys[1] - ys[0])
  dt = (t_max - t_min) / max(1, nt - 1)

  X, Y = np.meshgrid(xs, ys, indexing="xy")  # shape: (ny, nx)

  # Initial state: Gaussian wavepacket in ψ₀, zero in ψ₁
  x0, y0, sigma = -4.5, -2.0, 0.7
  kx0, ky0 = 2.2, 1.1
  envelope = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (4.0 * sigma ** 2))
  phase_init = np.exp(1j * (kx0 * X + ky0 * Y))
  psi0 = (envelope * phase_init).astype(complex)
  psi1 = np.zeros_like(psi0)

  nrm = float(np.sum(np.abs(psi0) ** 2) * dx * dy)
  if nrm > 0:
    psi0 /= math.sqrt(nrm)

  # Potential (same profiles as _spatial_wavepacket_2d_frames)
  if potential_type == "free":
    V = np.zeros_like(X)
  elif potential_type == "harmonic":
    V = 0.5 * (0.28 ** 2) * (X * X + Y * Y)
  elif potential_type == "well":
    V = np.where((np.abs(X) <= 6.0) & (np.abs(Y) <= 6.0), 0.0, 5.0)
  else:  # barrier
    V = 0.9 * np.exp(-(X * X) / (2.0 * 0.6 ** 2))

  potential_half = np.exp(-1j * V * (dt * 0.5))

  # Kinetic step: precompute Dirac unitary matrices in k-space
  kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
  ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
  KX, KY = np.meshgrid(kx, ky, indexing="xy")  # shape: (ny, nx)
  omega = np.sqrt(KX ** 2 + KY ** 2 + mass ** 2)
  cos_wdt = np.cos(omega * dt)
  sin_wdt_ov_w = np.where(omega > 1e-14, np.sin(omega * dt) / omega, dt * np.ones_like(omega))
  # U₀₀ = cos - i·sin/ω·m,  U₀₁ = -sin/ω·(ky + i·kx)
  # U₁₀ = -sin/ω·(-ky + i·kx), U₁₁ = cos + i·sin/ω·m
  # Matches continuous2d_backend.cpp apply_full_kinetic_dirac_2d() matrix elements
  s = sin_wdt_ov_w
  u00 = cos_wdt - 1j * s * mass
  u01 = -s * KY - 1j * s * KX
  u10 = s * KY - 1j * s * KX
  u11 = cos_wdt + 1j * s * mass

  def apply_kinetic_dirac(p0: np.ndarray, p1: np.ndarray):
    p0k = np.fft.fft2(p0)
    p1k = np.fft.fft2(p1)
    return np.fft.ifft2(u00 * p0k + u01 * p1k), np.fft.ifft2(u10 * p0k + u11 * p1k)

  frames = []
  dens_max = 1e-12

  for it in range(nt):
    t = t_min + (t_max - t_min) * it / max(1, nt - 1)
    dens0 = np.abs(psi0) ** 2
    dens1 = np.abs(psi1) ** 2
    density = dens0 + dens1
    dens_max = max(dens_max, float(np.max(density)))
    exp_x = float(np.sum(X * density) * dx * dy)
    exp_y = float(np.sum(Y * density) * dx * dy)
    frames.append({
        "t": t,
        "density": density.tolist(),
        "phase": np.angle(psi0).tolist(),
        "dirac_s0": dens0.tolist(),
        "dirac_s1": dens1.tolist(),
        "spin_x": (2.0 * np.real(np.conj(psi0) * psi1)).tolist(),
        "spin_y": (2.0 * np.imag(np.conj(psi0) * psi1)).tolist(),
        "spin_z": (dens0 - dens1).tolist(),
        "exp_x": exp_x,
        "exp_y": exp_y,
    })
    if it == nt - 1:
      break

    # Strang splitting: V/2 → T_Dirac → V/2
    psi0 = potential_half * psi0
    psi1 = potential_half * psi1
    psi0, psi1 = apply_kinetic_dirac(psi0, psi1)
    psi0 = potential_half * psi0
    psi1 = potential_half * psi1
    # Renormalise total spinor
    nrm = float(np.sum(np.abs(psi0) ** 2 + np.abs(psi1) ** 2) * dx * dy)
    if nrm > 0:
      inv_sqrt = 1.0 / math.sqrt(nrm)
      psi0 *= inv_sqrt
      psi1 *= inv_sqrt

  return {
      "x_min": x_min,
      "x_max": x_max,
      "y_min": y_min,
      "y_max": y_max,
      "potential_type": potential_type,
      "mass": mass,
      "frames": frames,
      "dens_max": dens_max,
      "has_phase": True,
      "has_dirac": True,
      "source": source,
  }


def _driven_wavepacket_2d_page(data: dict) -> str:
  """Render a driven-potential 2D animation with a driving field overlay panel."""
  payload = json.dumps(data, ensure_ascii=False)
  drv = data.get("driving", {})
  drv_type = drv.get("type", "sine")
  drv_info = f"type={drv_type}, A={drv.get('amplitude',0):.3g}, ω={drv.get('frequency',1):.3g}"
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>QuantX Animation - Driven 2D ({data.get("potential_type","barrier")})</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; background: #f4f6f8; }}
    h1 {{ font-size: 18px; margin-bottom: 4px; }}
    .layout {{ display: flex; gap: 14px; align-items: flex-start; max-width: 880px; }}
    .colorbar-col {{ display: flex; flex-direction: column; align-items: center; gap: 2px; margin-top: 4px; }}
    canvas#colorbar {{ border: 1px solid #aaa; display: block; cursor: default; }}
    .cb-labels {{ display: flex; flex-direction: column; justify-content: space-between;
                  height: 400px; font-size: 11px; color: #555; margin-left: 2px; }}
    .main-col {{ flex: 1; }}
    .row-ctrl {{ display: flex; gap: 10px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 10px; flex-wrap: wrap; }}
    button {{ padding: 4px 10px; }}
    .driving-panel {{ margin-top: 10px; background: #fff; border: 1px solid #ccc; border-radius: 8px;
                      padding: 8px; max-width: 540px; }}
    .driving-panel h3 {{ margin: 0 0 4px; font-size: 13px; color: #333; }}
    canvas#drv_chart {{ display: block; border: 1px solid #bbb; background: #fafafa; }}
    .meta {{ margin-top: 8px; font-size: 12px; color: #444; }}
  </style>
</head>
<body>
  <h1>QuantX Animation – Driven 2D Wavepacket ({data.get("potential_type","barrier")})</h1>
  <p style="font-size:13px;color:#555">Driving field: <b>{drv_info}</b>. Source: <b>{data.get("source","python-reference")}</b>.</p>
  <div class="controls">
    <button id="play">Play</button>
    <button id="pause">Pause</button>
    <input id="slider" type="range" min="1" step="1" style="width:380px"/>
    <span id="label" style="font-size:13px"></span>
    <select id="colorMode" style="font-size:12px;padding:3px 6px">
      <option value="density">Density</option>
      <option value="phase">Phase HSL</option>
    </select>
  </div>
  <div class="layout" style="margin-top:10px">
    <div class="main-col">
      <canvas id="main_canvas" width="540" height="540" style="display:block;border:1px solid #bbb"></canvas>
    </div>
    <div class="colorbar-col">
      <canvas id="colorbar" width="28" height="400"></canvas>
      <div class="cb-labels" id="cbLabels">
        <span id="cb_top"></span><span id="cb_mid"></span><span id="cb_bot"></span>
      </div>
    </div>
  </div>
  <div class="driving-panel">
    <h3>Driving field V_drive(t) = A·envelope(t)·spatial(r)</h3>
    <canvas id="drv_chart" width="540" height="80"></canvas>
  </div>
  <div id="meta" class="meta"></div>
  <script>
  const DATA = {payload};
  const slider = document.getElementById("slider");
  const label  = document.getElementById("label");
  const meta   = document.getElementById("meta");
  const colorModeCtl = document.getElementById("colorMode");
  slider.max = String(DATA.frames.length);
  slider.value = "1";

  function colorMapDensity(v) {{
    const t = Math.max(0, Math.min(1, v));
    return [Math.floor(255*Math.sqrt(t)), Math.floor(255*t*t), Math.floor(255*(1-t)), 255];
  }}
  function colorMapPhase(phi) {{
    const h = ((phi + Math.PI) / (2*Math.PI)) * 360;
    const s = 0.88, l = 0.52, a = s * Math.min(l, 1-l);
    const f = n => Math.round(255*(l - a*Math.max(-1, Math.min((n+h/30)%12-3, 9-(n+h/30)%12, 1))));
    return [f(0), f(8), f(4), 255];
  }}

  function drawColorbar(mode) {{
    const cv = document.getElementById("colorbar");
    const ctx = cv.getContext("2d");
    const h = cv.height;
    for (let y = 0; y < h; y++) {{
      const t = 1 - y / (h - 1);
      let c;
      if (mode === "phase") {{
        c = colorMapPhase((t * 2 - 1) * Math.PI);
      }} else {{
        c = colorMapDensity(t);
      }}
      ctx.fillStyle = `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;
      ctx.fillRect(0, y, cv.width, 1);
    }}
    if (mode === "phase") {{
      document.getElementById("cb_top").textContent = "+π";
      document.getElementById("cb_mid").textContent = "0";
      document.getElementById("cb_bot").textContent = "−π";
    }} else {{
      const mx = DATA.dens_max || 1;
      document.getElementById("cb_top").textContent = mx.toExponential(1);
      document.getElementById("cb_mid").textContent = (mx/2).toExponential(1);
      document.getElementById("cb_bot").textContent = "0";
    }}
  }}

  function drawFrame(fi) {{
    const f = DATA.frames[fi];
    const mode = colorModeCtl.value;
    const cv = document.getElementById("main_canvas");
    const ctx = cv.getContext("2d");
    const densArr = f.density, phaseArr = f.phase;
    const ny = densArr.length, nx = densArr[0].length;
    const dmax = DATA.dens_max || 1;
    // Render at grid resolution into an offscreen canvas, then scale to fill the viewport.
    const off = document.createElement("canvas");
    off.width = nx; off.height = ny;
    const offCtx = off.getContext("2d");
    const img = offCtx.createImageData(nx, ny);
    let p = 0;
    for (let j = 0; j < ny; j++) {{
      for (let i = 0; i < nx; i++) {{
        const c = mode === "phase" ? colorMapPhase(phaseArr[j][i]) : colorMapDensity(densArr[j][i] / dmax);
        img.data[p++]=c[0]; img.data[p++]=c[1]; img.data[p++]=c[2]; img.data[p++]=c[3];
      }}
    }}
    offCtx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, cv.width, cv.height);
    label.textContent = `t = ${{f.t.toFixed(3)}}  ⟨x⟩ = ${{f.exp_x.toFixed(2)}}  ⟨y⟩ = ${{f.exp_y.toFixed(2)}}`;
    if (f.driving_amplitude !== undefined) {{
      meta.textContent = `Drive A(t) = ${{f.driving_amplitude.toExponential(3)}}`;
    }}
    drawDrivingChart(fi);
  }}

  function drawDrivingChart(currentFi) {{
    const cv = document.getElementById("drv_chart");
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const amps = DATA.frames.map(f => f.driving_amplitude !== undefined ? f.driving_amplitude : 0);
    const maxA = Math.max(...amps.map(Math.abs), 1e-12);
    const n = amps.length;
    // Axis
    ctx.strokeStyle = "#ccc"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, H/2); ctx.lineTo(W, H/2); ctx.stroke();
    // Driving curve
    ctx.strokeStyle = "#1565c0"; ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {{
      const x = (i / (n-1)) * W;
      const y = H/2 - (amps[i] / maxA) * (H/2 - 4);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }}
    ctx.stroke();
    // Current time cursor
    const cx = (currentFi / (n-1)) * W;
    ctx.strokeStyle = "#e53935"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
    // Labels
    ctx.fillStyle = "#888"; ctx.font = "10px sans-serif";
    ctx.fillText(`t=${{DATA.frames[0].t.toFixed(1)}}`, 2, H-2);
    ctx.fillText(`t=${{DATA.frames[n-1].t.toFixed(1)}}`, W-28, H-2);
    ctx.fillText(`+${{maxA.toExponential(2)}}`, 2, 11);
    ctx.fillText(`-${{maxA.toExponential(2)}}`, 2, H-12);
  }}

  function render(n) {{
    const fi = Math.max(0, Math.min(n - 1, parseInt(slider.value, 10) - 1));
    drawFrame(fi);
    drawColorbar(colorModeCtl.value);
  }}

  slider.addEventListener("input", () => render(DATA.frames.length));
  colorModeCtl.addEventListener("change", () => render(DATA.frames.length));

  let playing = false, timer = null;
  document.getElementById("play").onclick = () => {{
    if (playing) return;
    playing = true;
    timer = setInterval(() => {{
      let v = parseInt(slider.value, 10) + 1;
      if (v > DATA.frames.length) v = 1;
      slider.value = String(v);
      render(DATA.frames.length);
    }}, 80);
  }};
  document.getElementById("pause").onclick = () => {{ playing = false; clearInterval(timer); }};

  render(DATA.frames.length);
  </script>
</body>
</html>"""


def _spatial_wavepacket_3d_frames(
    samples_t: int,
    nx: int,
    ny: int,
    nz: int,
    potential_type: str,
    boundary: str,
    source: str = "python-reference",
) -> dict:
  """Build a 3D reference solution for the spatial wavepacket animation."""
  try:
    import numpy as np
  except ImportError as ex:
    raise RuntimeError("3D spatial visualization requires numpy") from ex

  x_min, x_max = -6.0, 6.0
  y_min, y_max = -6.0, 6.0
  z_min, z_max = -6.0, 6.0
  t_min, t_max = 0.0, 2.5
  nx = max(16, nx)
  ny = max(16, ny)
  nz = max(16, nz)
  nt = max(10, samples_t)

  xs = np.linspace(x_min, x_max, nx)
  ys = np.linspace(y_min, y_max, ny)
  zs = np.linspace(z_min, z_max, nz)
  dx = float(xs[1] - xs[0])
  dy = float(ys[1] - ys[0])
  dz = float(zs[1] - zs[0])
  dt = (t_max - t_min) / (nt - 1)

  X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

  x0, y0, z0 = -2.0, -1.2, -0.8
  sigma = 0.7
  kx0, ky0, kz0 = 1.8, 1.0, 0.8
  envelope = np.exp(-((X - x0) ** 2 + (Y - y0) ** 2 + (Z - z0) ** 2) / (4.0 * sigma * sigma))
  phase = np.exp(1j * (kx0 * X + ky0 * Y + kz0 * Z))
  psi = envelope * phase

  if potential_type == "free":
    V = np.zeros_like(X)
  elif potential_type == "harmonic":
    omega = 0.25
    V = 0.5 * (omega * omega) * (X * X + Y * Y + Z * Z)
  elif potential_type == "well":
    wall = 5.5
    V = np.where((np.abs(X) <= 4.8) & (np.abs(Y) <= 4.8) & (np.abs(Z) <= 4.8), 0.0, wall)
  else:  # barrier
    V = 0.95 * np.exp(-(X * X) / (2.0 * 0.55 * 0.55))

  kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
  ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
  kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
  KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
  kinetic_phase = np.exp(-1j * 0.5 * (KX * KX + KY * KY + KZ * KZ) * dt)
  potential_half = np.exp(-1j * V * dt * 0.5)

  absorb_mask = np.ones_like(V, dtype=float)
  if boundary == "absorbing":
    width = 4.0
    strength = 4.0
    ix = np.minimum(np.arange(nx), np.arange(nx)[::-1]).reshape(nx, 1, 1)
    iy = np.minimum(np.arange(ny), np.arange(ny)[::-1]).reshape(1, ny, 1)
    iz = np.minimum(np.arange(nz), np.arange(nz)[::-1]).reshape(1, 1, nz)
    edge = np.minimum(np.minimum(ix, iy), iz).astype(float)
    t = np.clip((width - edge) / width, 0.0, 1.0)
    absorb_mask = np.exp(-strength * t * t * dt)

  def normalize(a):
    nrm = float(np.sum(np.abs(a) ** 2) * dx * dy * dz)
    if nrm > 0:
      a /= math.sqrt(nrm)
    return a

  def apply_dirichlet(a):
    a[0, :, :] = 0
    a[-1, :, :] = 0
    a[:, 0, :] = 0
    a[:, -1, :] = 0
    a[:, :, 0] = 0
    a[:, :, -1] = 0
    return a

  psi = normalize(psi)
  if boundary == "dirichlet":
    psi = apply_dirichlet(psi)
    psi = normalize(psi)

  frames = []
  dens_max = 1e-12
  z_mid = nz // 2
  y_mid = ny // 2
  x_mid = nx // 2
  points_max_per_frame = 1400
  points_density_floor_ratio = 0.08

  def sample_points(dens: np.ndarray) -> tuple[list[list[float]], float]:
    vmax = float(np.max(dens))
    if vmax <= 0.0:
      return [], 0.0

    floor = vmax * points_density_floor_ratio
    coords = np.argwhere(dens >= floor)
    if coords.shape[0] == 0:
      coords = np.argwhere(dens > 0.0)
    if coords.shape[0] == 0:
      return [], vmax

    values = dens[coords[:, 0], coords[:, 1], coords[:, 2]]
    if coords.shape[0] > points_max_per_frame:
      keep_idx = np.argpartition(values, -points_max_per_frame)[-points_max_per_frame:]
      coords = coords[keep_idx]
      values = values[keep_idx]

    points = []
    for n in range(coords.shape[0]):
      ix, iy, iz = coords[n]
      points.append(
          [
              round(float(xs[ix]), 4),
              round(float(ys[iy]), 4),
              round(float(zs[iz]), 4),
              round(float(values[n]), 8),
          ]
      )
    return points, vmax

  for it in range(nt):
    t = t_min + (t_max - t_min) * it / max(1, nt - 1)
    dens = np.abs(psi) ** 2
    points, point_vmax = sample_points(dens)
    dens_max = max(dens_max, float(np.max(dens)))
    norm = float(np.sum(dens) * dx * dy * dz)
    exp_x = float(np.sum(X * dens) * dx * dy * dz)
    exp_y = float(np.sum(Y * dens) * dx * dy * dz)
    exp_z = float(np.sum(Z * dens) * dx * dy * dz)
    psi_phase = np.angle(psi)
    frame_entry: dict = {
            "t": t,
            "xy": dens[:, :, z_mid].tolist(),
            "xz": dens[:, y_mid, :].tolist(),
            "yz": dens[x_mid, :, :].tolist(),
            "phase_xy": psi_phase[:, :, z_mid].tolist(),
            "phase_xz": psi_phase[:, y_mid, :].tolist(),
            "phase_yz": psi_phase[x_mid, :, :].tolist(),
            "exp_x": exp_x,
            "exp_y": exp_y,
            "exp_z": exp_z,
            "norm": norm,
            "points": points,
            "point_vmax": point_vmax,
            "point_count": len(points),
    }
    if nx * ny * nz <= 131072:
      frame_entry["density_xyz"] = dens.ravel(order="C").tolist()
    frames.append(frame_entry)
    if it == nt - 1:
      break

    psi = potential_half * psi
    psi_k = np.fft.fftn(psi)
    psi = np.fft.ifftn(kinetic_phase * psi_k)
    psi = potential_half * psi
    if boundary == "absorbing":
      psi = psi * absorb_mask
    if boundary == "dirichlet":
      psi = apply_dirichlet(psi)
    if boundary != "absorbing":
      psi = normalize(psi)

  return {
      "x_min": x_min,
      "x_max": x_max,
      "y_min": y_min,
      "y_max": y_max,
      "z_min": z_min,
      "z_max": z_max,
      "nx": nx,
      "ny": ny,
      "nz": nz,
      "potential_type": potential_type,
      "boundary": boundary,
      "frames": frames,
      "dens_max": dens_max,
      "has_phase": True,
      "has_volume_grid": nx * ny * nz <= 131072,
      "volume_point_fields": ["x", "y", "z", "density"],
      "volume_points_max_per_frame": points_max_per_frame,
      "volume_points_density_floor_ratio": points_density_floor_ratio,
      "source": source,
  }


def _spatial_wavepacket_3d_page(data: dict) -> str:
  """Render the 3D slice-based spatial wavepacket animation page with phase toggle and colorbar."""
  payload = json.dumps(data, ensure_ascii=False)
  has_phase = bool(data.get("has_phase"))
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>QuantX Animation - 3D Spatial Wavepacket ({data.get("potential_type","barrier")}, {data.get("boundary","periodic")})</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; background: #f8f8f8; }}
    h1 {{ font-size: 18px; margin-bottom: 4px; }}
    .row {{ display: flex; gap: 10px; align-items: flex-start; margin-top: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 12px; }}
    .panel {{ border: 1px solid #ccc; border-radius: 8px; padding: 8px; background: #fff; }}
    .panel h3 {{ margin: 0 0 6px; font-size: 13px; }}
    canvas.slice {{ width: 100%; aspect-ratio: 1/1; border: 1px solid #bbb; display: block; }}
    canvas#colorbar {{ border: 1px solid #aaa; display: block; }}
    .cb-labels {{ display: flex; flex-direction: column; justify-content: space-between; font-size: 11px; color: #555; height: 280px; padding: 2px 0; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 8px; flex-wrap: wrap; }}
    button {{ padding: 4px 10px; }}
    select {{ padding: 3px 6px; }}
    .meta {{ margin-top: 8px; font-size: 12px; color: #444; }}
  </style>
</head>
<body>
  <h1>QuantX Animation - 3D Spatial Wavepacket</h1>
  <p style="font-size:13px;color:#555">Split-step Fourier. Orthogonal midplane slices of |ψ|². Source: <b>{data.get("source","python-reference")}</b>.</p>
  <div class="controls">
    <button id="play">Play</button>
    <button id="pause">Pause</button>
    <input id="slider" type="range" min="1" step="1" style="width:400px"/>
    <span id="label" style="font-size:13px"></span>
    {"<label>Color: <select id='colorMode'><option value='density'>Density |ψ|²</option><option value='phase'>Phase arg(ψ)</option></select></label>" if has_phase else ""}
  </div>
  <div class="row">
    <div class="grid">
      <div class="panel"><h3>XY (z-mid)</h3><canvas class="slice" id="cv_xy" width="240" height="240"></canvas></div>
      <div class="panel"><h3>XZ (y-mid)</h3><canvas class="slice" id="cv_xz" width="240" height="240"></canvas></div>
      <div class="panel"><h3>YZ (x-mid)</h3><canvas class="slice" id="cv_yz" width="240" height="240"></canvas></div>
    </div>
    <div>
      <canvas id="colorbar" width="26" height="280"></canvas>
    </div>
    <div class="cb-labels" id="cbLabels">
      <span>max</span><span></span><span>0</span>
    </div>
  </div>
  <div id="meta" class="meta"></div>
  <script>
  const DATA = {payload};
  const slider = document.getElementById("slider");
  const label = document.getElementById("label");
  const meta = document.getElementById("meta");
  const cbCanvas = document.getElementById("colorbar");
  const cbCtx = cbCanvas.getContext("2d");
  const cbLabels = document.getElementById("cbLabels");
  const ctxXY = document.getElementById("cv_xy").getContext("2d");
  const ctxXZ = document.getElementById("cv_xz").getContext("2d");
  const ctxYZ = document.getElementById("cv_yz").getContext("2d");
  slider.max = String(DATA.frames.length);
  slider.value = "1";

  function colorMapDensity(v) {{
    const t = Math.max(0, Math.min(1, v));
    return [Math.floor(255*Math.pow(t,0.55)), Math.floor(255*t*t), Math.floor(255*(1-t)), 255];
  }}
  function colorMapPhase(phi) {{
    const h = ((phi + Math.PI) / (2*Math.PI)) * 360;
    const s = 0.88, l = 0.52, a = s * Math.min(l, 1-l);
    const f = n => Math.round(255*(l - a*Math.max(-1, Math.min((n+h/30)%12-3, 9-(n+h/30)%12, 1))));
    return [f(0), f(8), f(4), 255];
  }}

  function drawColorbar(mode) {{
    const h = cbCanvas.height;
    for (let y = 0; y < h; y++) {{
      const t = 1 - y/h;
      const rgba = (mode === "phase") ? colorMapPhase((t*2-1)*Math.PI) : colorMapDensity(t);
      cbCtx.fillStyle = `rgb(${{rgba[0]}},${{rgba[1]}},${{rgba[2]}})`;
      cbCtx.fillRect(0, y, cbCanvas.width, 1);
    }}
    if (mode === "phase") {{
      cbLabels.innerHTML = "<span>+π</span><span>0</span><span>−π</span>";
    }} else {{
      cbLabels.innerHTML = `<span>${{(DATA.dens_max||0).toExponential(1)}}</span><span></span><span>0</span>`;
    }}
  }}

  function drawSlice(ctx, densArr, phaseArr, mode) {{
    const rows = densArr.length;
    const cols = densArr[0].length;
    const img = ctx.createImageData(cols, rows);
    const densMax = Math.max(1e-12, DATA.dens_max || 0);
    let p = 0;
    if (mode === "phase" && phaseArr && DATA.has_phase) {{
      for (let j = 0; j < rows; j++) {{
        for (let i = 0; i < cols; i++) {{
          const c = colorMapPhase(phaseArr[j][i]);
          img.data[p++]=c[0]; img.data[p++]=c[1]; img.data[p++]=c[2]; img.data[p++]=c[3];
        }}
      }}
    }} else {{
      for (let j = 0; j < rows; j++) {{
        for (let i = 0; i < cols; i++) {{
          const c = colorMapDensity(densArr[j][i] / densMax);
          img.data[p++]=c[0]; img.data[p++]=c[1]; img.data[p++]=c[2]; img.data[p++]=c[3];
        }}
      }}
    }}
    const tmp = document.createElement("canvas");
    tmp.width = cols; tmp.height = rows;
    tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(tmp, 0, 0, ctx.canvas.width, ctx.canvas.height);
  }}

  function render(frameIdx) {{
    const f = DATA.frames[frameIdx - 1];
    const mode = document.getElementById("colorMode")?.value || "density";
    drawSlice(ctxXY, f.xy, f.phase_xy, mode);
    drawSlice(ctxXZ, f.xz, f.phase_xz, mode);
    drawSlice(ctxYZ, f.yz, f.phase_yz, mode);
    label.textContent = `frame ${{frameIdx}}/${{DATA.frames.length}}, t=${{f.t.toFixed(3)}}`;
    meta.textContent = `boundary=${{DATA.boundary}}, potential=${{DATA.potential_type}}, norm=${{(f.norm||0).toFixed(6)}}, <x,y,z>=(${{(f.exp_x||0).toFixed(3)}}, ${{(f.exp_y||0).toFixed(3)}}, ${{(f.exp_z||0).toFixed(3)}})`;
    slider.value = String(frameIdx);
    drawColorbar(mode);
  }}

  const colorModeEl = document.getElementById("colorMode");
  if (colorModeEl) colorModeEl.onchange = () => render(Number(slider.value));

  let timer = null;
  function doPlay() {{
    if (timer) return;
    timer = setInterval(() => {{
      let f = Number(slider.value);
      f = f >= DATA.frames.length ? 1 : f + 1;
      render(f);
    }}, 90);
  }}
  function doPause() {{
    if (!timer) return;
    clearInterval(timer);
    timer = null;
  }}
  document.getElementById("play").onclick = doPlay;
  document.getElementById("pause").onclick = doPause;
  slider.oninput = () => render(Number(slider.value));
  render(1);
  </script>
</body>
</html>"""


def _dirac_spinor_page(data: dict) -> str:
  """Render the Dirac spinor 2D visualization: 4-component panels + spin current arrows."""
  payload = json.dumps(data, ensure_ascii=False)
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>QuantX Animation - 2D Dirac Spinor ({data.get("potential_type","barrier")})</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; background: #f4f6f8; }}
    h1 {{ font-size: 18px; margin-bottom: 4px; }}
    .grid2x2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; max-width: 1100px; }}
    .panel {{ background: #fff; border: 1px solid #ccc; border-radius: 8px; padding: 8px; }}
    .panel h3 {{ margin: 0 0 5px; font-size: 13px; color: #333; }}
    .cv-wrap {{ position: relative; display: inline-block; }}
    canvas.hm {{ display: block; border: 1px solid #bbb; }}
    svg.arrows {{ position: absolute; top: 0; left: 0; pointer-events: none; }}
    .row-ctrl {{ display: flex; gap: 10px; align-items: center; margin-bottom: 8px; flex-wrap: wrap; }}
    .colorbar-row {{ display: flex; gap: 6px; align-items: center; margin-top: 4px; font-size: 11px; color: #555; }}
    canvas.cb {{ border: 1px solid #aaa; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 10px; flex-wrap: wrap; }}
    button {{ padding: 4px 10px; }}
    select {{ padding: 3px 6px; font-size: 12px; }}
    .meta {{ margin-top: 8px; font-size: 12px; color: #444; }}
  </style>
</head>
<body>
  <h1>QuantX Animation - 2D Dirac Spinor ({data.get("potential_type","barrier")})</h1>
  <p style="font-size:13px;color:#555">Split-step Fourier Dirac solver (m={data.get("mass",1.0):.2f}). Source: <b>{data.get("source","python-reference")}</b>.</p>
  <div class="controls">
    <button id="play">Play</button>
    <button id="pause">Pause</button>
    <input id="slider" type="range" min="1" step="1" style="width:380px"/>
    <span id="label" style="font-size:13px"></span>
  </div>
  <div class="grid2x2" style="margin-top:10px">
    <div class="panel">
      <h3>ψ₀ density |ψ₀|²  (up spinor)</h3>
      <div class="cv-wrap"><canvas class="hm" id="cv_s0" width="340" height="340"></canvas></div>
      <div class="colorbar-row"><canvas class="cb" id="cb_s0" width="200" height="12"></canvas><span id="cbl_s0"></span></div>
    </div>
    <div class="panel">
      <h3>ψ₁ density |ψ₁|²  (down spinor)</h3>
      <div class="cv-wrap"><canvas class="hm" id="cv_s1" width="340" height="340"></canvas></div>
      <div class="colorbar-row"><canvas class="cb" id="cb_s1" width="200" height="12"></canvas><span id="cbl_s1"></span></div>
    </div>
    <div class="panel">
      <h3>Total density + phase arg(ψ₀)  <select id="totalMode"><option value="density">Density</option><option value="phase">Phase HSL</option></select></h3>
      <div class="cv-wrap"><canvas class="hm" id="cv_tot" width="340" height="340"></canvas></div>
      <div class="colorbar-row"><canvas class="cb" id="cb_tot" width="200" height="12"></canvas><span id="cbl_tot"></span></div>
    </div>
    <div class="panel">
      <h3>Spin-z density + current arrows</h3>
      <div class="cv-wrap">
        <canvas class="hm" id="cv_spin" width="340" height="340"></canvas>
        <svg class="arrows" id="svg_arrows" width="340" height="340"></svg>
      </div>
      <div class="colorbar-row"><canvas class="cb" id="cb_spin" width="200" height="12"></canvas><span id="cbl_spin"></span></div>
      <div class="colorbar-row"><span style="color:#2166ac">■ spin↓</span>&nbsp;&nbsp;<span style="color:#d6604d">■ spin↑</span>&nbsp;arrows = spin current (J<sub>x</sub>,J<sub>y</sub>)</div>
    </div>
  </div>
  <div id="meta" class="meta"></div>
  <script>
  const DATA = {payload};
  const slider = document.getElementById("slider");
  const label = document.getElementById("label");
  const meta = document.getElementById("meta");
  slider.max = String(DATA.frames.length);
  slider.value = "1";

  function colorMapDensity(v) {{
    const t = Math.max(0, Math.min(1, v));
    return [Math.floor(255*Math.sqrt(t)), Math.floor(255*t*t), Math.floor(255*(1-t)), 255];
  }}
  function colorMapPhase(phi) {{
    const h = ((phi + Math.PI) / (2*Math.PI)) * 360;
    const s = 0.88, l = 0.52, a = s * Math.min(l, 1-l);
    const f = n => Math.round(255*(l - a*Math.max(-1, Math.min((n+h/30)%12-3, 9-(n+h/30)%12, 1))));
    return [f(0), f(8), f(4), 255];
  }}
  // Blue (-1) to white (0) to red (+1) for spin-z
  function colorMapSpinZ(v) {{
    const t = Math.max(-1, Math.min(1, v));
    if (t < 0) {{
      const s = -t;
      return [Math.round(255*(1-s*0.66)), Math.round(255*(1-s*0.58)), 255, 255];
    }}
    return [255, Math.round(255*(1-t*0.62)), Math.round(255*(1-t*0.69)), 255];
  }}

  function drawHeatmap(canvasId, arr2d, colorFn) {{
    const cv = document.getElementById(canvasId);
    const ctx = cv.getContext("2d");
    const ny = arr2d.length, nx = arr2d[0].length;
    const img = ctx.createImageData(nx, ny);
    let p = 0;
    for (let j = 0; j < ny; j++) {{
      for (let i = 0; i < nx; i++) {{
        const c = colorFn(arr2d[j][i]);
        img.data[p++]=c[0]; img.data[p++]=c[1]; img.data[p++]=c[2]; img.data[p++]=c[3];
      }}
    }}
    const tmp = document.createElement("canvas");
    tmp.width = nx; tmp.height = ny;
    tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(tmp, 0, 0, cv.width, cv.height);
  }}

  function drawColorbarH(canvasId, labelId, mode, maxVal) {{
    const cv = document.getElementById(canvasId);
    const ctx = cv.getContext("2d");
    const w = cv.width;
    for (let x = 0; x < w; x++) {{
      const t = x / w;
      let rgba;
      if (mode === "phase") rgba = colorMapPhase((t*2-1)*Math.PI);
      else if (mode === "spinz") rgba = colorMapSpinZ(t*2-1);
      else rgba = colorMapDensity(t);
      ctx.fillStyle = `rgb(${{rgba[0]}},${{rgba[1]}},${{rgba[2]}})`;
      ctx.fillRect(x, 0, 1, cv.height);
    }}
    const lbl = document.getElementById(labelId);
    if (mode === "phase") lbl.textContent = " −π → +π";
    else if (mode === "spinz") lbl.textContent = " −1 → 0 → +1";
    else lbl.textContent = ` 0 → ${{(maxVal||0).toExponential(2)}}`;
  }}

  function drawSpinArrows(f) {{
    const svg = document.getElementById("svg_arrows");
    const W = 340, H = 340;
    svg.setAttribute("width", W); svg.setAttribute("height", H);
    svg.innerHTML = "";
    const ny = f.spin_x.length, nx = f.spin_x[0].length;
    const step = Math.max(4, Math.floor(Math.min(nx, ny) / 18));
    const scaleX = W / nx, scaleY = H / ny;
    const arrowLen = Math.min(scaleX, scaleY) * step * 0.7;
    for (let j = step; j < ny - step; j += step) {{
      for (let i = step; i < nx - step; i += step) {{
        const jx = f.spin_x[j][i], jy = f.spin_y[j][i], sz = f.spin_z[j][i];
        const mag = Math.sqrt(jx*jx + jy*jy);
        if (mag < 1e-10) continue;
        const px = (i + 0.5) * scaleX, py = (j + 0.5) * scaleY;
        const dx = (jx/mag) * arrowLen * Math.min(1, mag*3);
        const dy = (jy/mag) * arrowLen * Math.min(1, mag*3);
        const col = sz > 0 ? `hsl(0,80%,${{40+Math.abs(sz)*20|0}}%)` : `hsl(220,80%,${{40+Math.abs(sz)*20|0}}%)`;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", px); line.setAttribute("y1", py);
        line.setAttribute("x2", px+dx); line.setAttribute("y2", py+dy);
        line.setAttribute("stroke", col);
        line.setAttribute("stroke-width", "1.5");
        line.setAttribute("stroke-linecap", "round");
        svg.appendChild(line);
      }}
    }}
  }}

  function render(frameIdx) {{
    const f = DATA.frames[frameIdx - 1];
    const densMax = Math.max(1e-12, DATA.dens_max||0);
    const totalMode = document.getElementById("totalMode").value;

    // Panel 1: |ψ₀|²
    drawHeatmap("cv_s0", f.dirac_s0, v => colorMapDensity(v/densMax));
    drawColorbarH("cb_s0", "cbl_s0", "density", densMax);

    // Panel 2: |ψ₁|²
    drawHeatmap("cv_s1", f.dirac_s1, v => colorMapDensity(v/densMax));
    drawColorbarH("cb_s1", "cbl_s1", "density", densMax);

    // Panel 3: total density or phase
    if (totalMode === "phase") {{
      drawHeatmap("cv_tot", f.phase, colorMapPhase);
      drawColorbarH("cb_tot", "cbl_tot", "phase", 0);
    }} else {{
      drawHeatmap("cv_tot", f.density, v => colorMapDensity(v/densMax));
      drawColorbarH("cb_tot", "cbl_tot", "density", densMax);
    }}

    // Panel 4: spin-z + arrows
    drawHeatmap("cv_spin", f.spin_z, colorMapSpinZ);
    drawSpinArrows(f);
    drawColorbarH("cb_spin", "cbl_spin", "spinz", 0);

    label.textContent = `frame ${{frameIdx}}/${{DATA.frames.length}}, t=${{f.t.toFixed(3)}}`;
    meta.textContent = `potential=${{DATA.potential_type}}, mass=${{DATA.mass||1}}, <x,y>=(${{f.exp_x.toFixed(3)}},${{f.exp_y.toFixed(3)}})`;
    slider.value = String(frameIdx);
  }}

  document.getElementById("totalMode").onchange = () => render(Number(slider.value));
  let timer = null;
  function doPlay() {{
    if (timer) return;
    timer = setInterval(() => {{
      let f = Number(slider.value);
      f = f >= DATA.frames.length ? 1 : f + 1;
      render(f);
    }}, 90);
  }}
  function doPause() {{ if (!timer) return; clearInterval(timer); timer = null; }}
  document.getElementById("play").onclick = doPlay;
  document.getElementById("pause").onclick = doPause;
  slider.oninput = () => render(Number(slider.value));
  render(1);
  </script>
</body>
</html>"""


