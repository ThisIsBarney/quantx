#!/usr/bin/env python3
"""Generate animated HTML visualizations for QuantX simulations."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _animated_page(title: str, charts: list[dict]) -> str:
  charts_json = json.dumps(charts, ensure_ascii=False)
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; }}
    .chart {{ margin: 24px 0; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 8px; }}
    button {{ padding: 4px 10px; }}
    .legend {{ font-size: 12px; margin-top: 6px; }}
    .legend span {{ margin-right: 16px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Use Play/Pause or drag the slider to inspect time evolution.</p>
  <div id="charts"></div>
  <script>
  const CHARTS = {charts_json};

  function createSvgEl(tag, attrs) {{
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs || {{}})) el.setAttribute(k, v);
    return el;
  }}

  function buildChart(container, cfg, idx) {{
    const width = 900, height = 420;
    const left = 70, right = 20, top = 50, bottom = 60;
    const plotW = width - left - right, plotH = height - top - bottom;
    const allX = cfg.series.flatMap(s => s.points.map(p => p[0]));
    const allY = cfg.series.flatMap(s => s.points.map(p => p[1]));
    let xMin = Math.min(...allX), xMax = Math.max(...allX);
    let yMin = Math.min(...allY), yMax = Math.max(...allY);
    if (Math.abs(xMax - xMin) < 1e-12) xMax = xMin + 1;
    if (Math.abs(yMax - yMin) < 1e-12) yMax = yMin + 1;

    const sx = x => left + (x - xMin) / (xMax - xMin) * plotW;
    const sy = y => top + (1 - (y - yMin) / (yMax - yMin)) * plotH;

    const box = document.createElement("div");
    box.className = "chart";
    const h2 = document.createElement("h2");
    h2.textContent = cfg.title;
    box.appendChild(h2);

    const svg = createSvgEl("svg", {{ viewBox: `0 0 ${{width}} ${{height}}`, width, height }});
    box.appendChild(svg);

    svg.appendChild(createSvgEl("line", {{ x1: left, y1: top + plotH, x2: left + plotW, y2: top + plotH, stroke: "#333", "stroke-width": "1.5" }}));
    svg.appendChild(createSvgEl("line", {{ x1: left, y1: top, x2: left, y2: top + plotH, stroke: "#333", "stroke-width": "1.5" }}));

    for (let i = 0; i <= 5; i++) {{
      const gx = left + plotW * i / 5;
      const gy = top + plotH * i / 5;
      const xv = xMin + (xMax - xMin) * i / 5;
      const yv = yMax - (yMax - yMin) * i / 5;
      svg.appendChild(createSvgEl("line", {{ x1: gx, y1: top, x2: gx, y2: top + plotH, stroke: "#ddd", "stroke-width": "1" }}));
      svg.appendChild(createSvgEl("line", {{ x1: left, y1: gy, x2: left + plotW, y2: gy, stroke: "#ddd", "stroke-width": "1" }}));
      const tx = createSvgEl("text", {{ x: gx, y: top + plotH + 18, "text-anchor": "middle", "font-size": "12" }});
      tx.textContent = xv.toPrecision(3);
      svg.appendChild(tx);
      const ty = createSvgEl("text", {{ x: left - 8, y: gy + 4, "text-anchor": "end", "font-size": "12" }});
      ty.textContent = yv.toPrecision(3);
      svg.appendChild(ty);
    }}

    const xLabel = createSvgEl("text", {{ x: left + plotW / 2, y: height - 18, "text-anchor": "middle", "font-size": "12" }});
    xLabel.textContent = cfg.x_label;
    svg.appendChild(xLabel);
    const yLabel = createSvgEl("text", {{ x: 18, y: top + plotH / 2, transform: `rotate(-90 18 ${{top + plotH / 2}})`, "text-anchor": "middle", "font-size": "12" }});
    yLabel.textContent = cfg.y_label;
    svg.appendChild(yLabel);

    const paths = cfg.series.map(s => {{
      const p = createSvgEl("path", {{ fill: "none", stroke: s.color, "stroke-width": "2" }});
      svg.appendChild(p);
      return p;
    }});

    const controls = document.createElement("div");
    controls.className = "controls";
    const playBtn = document.createElement("button");
    playBtn.textContent = "Play";
    const pauseBtn = document.createElement("button");
    pauseBtn.textContent = "Pause";
    const slider = document.createElement("input");
    slider.type = "range";
    const maxFrames = Math.max(...cfg.series.map(s => s.points.length));
    slider.min = "1";
    slider.max = String(maxFrames);
    slider.value = "1";
    slider.style.width = "420px";
    const frameLabel = document.createElement("span");
    frameLabel.textContent = `frame 1/${{maxFrames}}`;
    controls.append(playBtn, pauseBtn, slider, frameLabel);
    box.appendChild(controls);

    const legend = document.createElement("div");
    legend.className = "legend";
    cfg.series.forEach(s => {{
      const item = document.createElement("span");
      item.innerHTML = `<b style="color:${{s.color}}">${{s.name}}</b>`;
      legend.appendChild(item);
    }});
    box.appendChild(legend);

    container.appendChild(box);

    let timer = null;
    function render(frame) {{
      for (let si = 0; si < cfg.series.length; si++) {{
        const pts = cfg.series[si].points.slice(0, Math.min(frame, cfg.series[si].points.length));
        const d = pts.map((p, i) => `${{i === 0 ? "M" : "L"}}${{sx(p[0]).toFixed(2)}} ${{sy(p[1]).toFixed(2)}}`).join(" ");
        paths[si].setAttribute("d", d);
      }}
      frameLabel.textContent = `frame ${{frame}}/${{maxFrames}}`;
      slider.value = String(frame);
    }}
    function play() {{
      if (timer) return;
      timer = setInterval(() => {{
        let f = Number(slider.value);
        f = f >= maxFrames ? 1 : f + 1;
        render(f);
      }}, 90);
    }}
    function pause() {{
      if (!timer) return;
      clearInterval(timer);
      timer = null;
    }}
    playBtn.onclick = play;
    pauseBtn.onclick = pause;
    slider.oninput = () => render(Number(slider.value));
    render(1);
  }}

  const container = document.getElementById("charts");
  CHARTS.forEach((cfg, idx) => buildChart(container, cfg, idx));
  </script>
</body>
</html>
"""


def _make_engine(root: Path):
  sys.path.insert(0, str(root / "python"))
  from quantx_engine import Engine

  return Engine(str(root / "build-vs" / "Debug" / "quantx.dll"))


def _spatial_wavepacket_frames(samples_t: int, samples_x: int, potential_type: str) -> dict:
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

  return {"xs": xs, "frames": frames, "x_min": x_min, "x_max": x_max, "potential": potential, "potential_type": potential_type}


def _spatial_wavepacket_page(data: dict) -> str:
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
  <p>1D Schrodinger solver (finite difference + Crank-Nicolson): showing spatial probability density |psi(x,t)|^2, phase color, potential profile, and sampled particle cloud.</p>
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


def _spatial_wavepacket_2d_frames(samples_t: int, nx: int, ny: int, potential_type: str) -> dict:
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
    V = np.zeros_like(X)
  elif potential_type == "harmonic":
    omega = 0.28
    V = 0.5 * (omega * omega) * (X * X + Y * Y)
  elif potential_type == "well":
    wall = 5.0
    V = np.where((np.abs(X) <= 6.0) & (np.abs(Y) <= 6.0), 0.0, wall)
  else:  # barrier
    V = 0.9 * np.exp(-(X * X) / (2.0 * 0.6 * 0.6))

  kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
  ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
  KX, KY = np.meshgrid(kx, ky, indexing="xy")
  kinetic_phase = np.exp(-1j * 0.5 * (KX * KX + KY * KY) * dt)
  potential_half = np.exp(-1j * V * dt * 0.5)

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
    frames.append(
        {
            "t": t,
            "density": dens.tolist(),
            "exp_x": exp_x,
            "exp_y": exp_y,
        }
    )
    if it == nt - 1:
      break

    # Strang splitting: V/2 -> T -> V/2
    psi = potential_half * psi
    psi_k = np.fft.fft2(psi)
    psi = np.fft.ifft2(kinetic_phase * psi_k)
    psi = potential_half * psi
    psi = normalize(psi)

  return {
      "x_min": x_min,
      "x_max": x_max,
      "y_min": y_min,
      "y_max": y_max,
      "potential_type": potential_type,
      "frames": frames,
      "dens_max": dens_max,
  }


def _spatial_wavepacket_2d_page(data: dict) -> str:
  payload = json.dumps(data, ensure_ascii=False)
  return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>QuantX Animation - 2D Spatial Wavepacket ({data.get("potential_type","barrier")})</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 20px; }}
    .controls {{ display: flex; gap: 8px; align-items: center; margin-top: 8px; }}
    button {{ padding: 4px 10px; }}
  </style>
</head>
<body>
  <h1>QuantX Animation - 2D Spatial Wavepacket ({data.get("potential_type","barrier")})</h1>
  <p>Numerical 2D Schrodinger evolution (split-step Fourier): heatmap of |psi(x,y,t)|^2 with center-of-mass marker.</p>
  <canvas id="cv" width="640" height="640" style="border:1px solid #ccc"></canvas>
  <div class="controls">
    <button id="play">Play</button>
    <button id="pause">Pause</button>
    <input id="slider" type="range" min="1" step="1" style="width:520px"/>
    <span id="label"></span>
  </div>
  <script>
  const DATA = {payload};
  const cv = document.getElementById("cv");
  const ctx = cv.getContext("2d");
  const slider = document.getElementById("slider");
  const label = document.getElementById("label");
  const play = document.getElementById("play");
  const pause = document.getElementById("pause");
  slider.max = String(DATA.frames.length);
  slider.value = "1";

  function colorMap(v) {{
    const t = Math.max(0, Math.min(1, v));
    const r = Math.floor(255 * Math.sqrt(t));
    const g = Math.floor(255 * (t * t));
    const b = Math.floor(255 * (1 - t));
    return [r, g, b, 255];
  }}

  function render(frameIdx) {{
    const f = DATA.frames[frameIdx - 1];
    const ny = f.density.length;
    const nx = f.density[0].length;
    const img = ctx.createImageData(nx, ny);
    let p = 0;
    for (let j = 0; j < ny; j++) {{
      for (let i = 0; i < nx; i++) {{
        const v = f.density[j][i] / DATA.dens_max;
        const c = colorMap(v);
        img.data[p++] = c[0];
        img.data[p++] = c[1];
        img.data[p++] = c[2];
        img.data[p++] = c[3];
      }}
    }}
    const tmp = document.createElement("canvas");
    tmp.width = nx; tmp.height = ny;
    tmp.getContext("2d").putImageData(img, 0, 0);
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(tmp, 0, 0, cv.width, cv.height);

    const sx = x => (x - DATA.x_min) / (DATA.x_max - DATA.x_min) * cv.width;
    const sy = y => cv.height - (y - DATA.y_min) / (DATA.y_max - DATA.y_min) * cv.height;
    ctx.beginPath();
    ctx.arc(sx(f.exp_x), sy(f.exp_y), 6, 0, Math.PI * 2);
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();

    label.textContent = `frame ${{frameIdx}}/${{DATA.frames.length}}, t=${{f.t.toFixed(3)}}`;
    slider.value = String(frameIdx);
  }}

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
  play.onclick = doPlay;
  pause.onclick = doPause;
  slider.oninput = () => render(Number(slider.value));
  render(1);
  </script>
</body>
</html>"""


def _rabi_data(engine, samples: int) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
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


def main() -> int:
  parser = argparse.ArgumentParser(description="Generate QuantX visualization HTML.")
  parser.add_argument("--scenario", choices=["rabi", "mps", "spatial", "spatial2d", "all"], default="all")
  parser.add_argument("--spatial-potential", choices=["barrier", "free", "harmonic", "well"], default="barrier")
  parser.add_argument("--spatial2d-potential", choices=["barrier", "free", "harmonic", "well"], default="barrier")
  parser.add_argument("--out-dir", default="out/visualizations")
  parser.add_argument("--samples-rabi", type=int, default=80)
  parser.add_argument("--samples-mps", type=int, default=60)
  parser.add_argument("--samples-spatial-t", type=int, default=120)
  parser.add_argument("--samples-spatial-x", type=int, default=220)
  parser.add_argument("--samples-spatial2d-t", type=int, default=90)
  parser.add_argument("--samples-spatial2d-x", type=int, default=72)
  parser.add_argument("--samples-spatial2d-y", type=int, default=72)
  args = parser.parse_args()

  root = Path(__file__).resolve().parents[1]
  out_dir = root / args.out_dir
  out_dir.mkdir(parents=True, exist_ok=True)
  engine = _make_engine(root)
  generated = []

  if args.scenario in {"rabi", "all"}:
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
    spatial_data = _spatial_wavepacket_frames(args.samples_spatial_t, args.samples_spatial_x, args.spatial_potential)
    path = out_dir / f"quantum_spatial_wavepacket_{args.spatial_potential}.html"
    path.write_text(_spatial_wavepacket_page(spatial_data), encoding="utf-8")
    generated.append(path)

  if args.scenario in {"spatial2d", "all"}:
    spatial2d = _spatial_wavepacket_2d_frames(
        args.samples_spatial2d_t, args.samples_spatial2d_x, args.samples_spatial2d_y, args.spatial2d_potential
    )
    path = out_dir / f"quantum_spatial2d_wavepacket_{args.spatial2d_potential}.html"
    path.write_text(_spatial_wavepacket_2d_page(spatial2d), encoding="utf-8")
    generated.append(path)

  for p in generated:
    print(f"wrote visualization: {p}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
