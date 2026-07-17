"""Doped Hubbard cylinder stripe order, from a QX_STRIPES showcase log.

Parses the "PROF i x y n 2sz" lines emitted by qx_hubbard_showcase (QX_STRIPES
mode: hole-doped width-W cylinder, staggered edge pinning at x=0, per-site
profiles measured on the converged block-DMRG ground state), writes
docs/hubbard_stripes.csv and renders docs/hubbard_stripes.svg:

  top    -- rung-averaged hole density 1 - n(x): the charge stripes;
  bottom -- the lattice itself: cell colour = staggered spin
            (-1)^(x+y) <Sz_i> (red/blue AF domains -- the pi phase shift
            across each charge stripe is the stripe-order fingerprint),
            circle radius = per-site hole density.

Run: python benchmarks/plot_stripes.py [logfile] -> docs/hubbard_stripes.svg
"""
import os, re, sys

HERE = os.path.dirname(__file__)
LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "stripes_16x4_chi1024.log")

head, sites = None, {}
for line in open(LOG):
    m = re.match(r"STRIPES Lx=(\d+) W=(\d+) chi=(\d+) U=([\d.]+) holes=(\d+) "
                 r"hpin=([\d.]+) nsw=(\d+) : E0=(-?[\d.]+) \(([\d.]+)s\)", line)
    if m:
        head = dict(Lx=int(m[1]), W=int(m[2]), chi=int(m[3]), U=float(m[4]),
                    holes=int(m[5]), hpin=float(m[6]), nsw=int(m[7]),
                    E0=float(m[8]), wall=float(m[9]))
    m = re.match(r"PROF (\d+) (\d+) (\d+) (-?[\d.]+) (-?[\d.]+)", line)
    if m:
        sites[int(m[1])] = (int(m[2]), int(m[3]), float(m[4]), float(m[5]))
assert head and sites, f"no STRIPES/PROF lines in {LOG}"
Lx, Wd = head["Lx"], head["W"]

with open(os.path.join(HERE, "..", "docs", "hubbard_stripes.csv"), "w", newline="") as f:
    f.write("i,x,y,n,two_sz\n")
    for i in sorted(sites):
        x, y, nv, sv = sites[i]
        f.write(f"{i},{x},{y},{nv:.8f},{sv:.8f}\n")

assert head["holes"] > 0, "stripe figure needs a doped run (holes > 0)"
holex = [0.0] * Lx                                  # rung-averaged hole density
stag = {}                                           # (x,y) -> (-1)^(x+y) <Sz>
holes_site = {}
for i, (x, y, nv, sv) in sites.items():
    holex[x] += (1.0 - nv) / Wd
    stag[(x, y)] = ((-1) ** (x + y)) * 0.5 * sv     # <Sz> = 2Sz/2
    holes_site[(x, y)] = 1.0 - nv

W, H = 760, 560
L0, R0 = 76, 26
PX0, PX1 = L0, W - R0
# top panel: hole density profile
TY0, TY1 = 92, 288
hmax = max(holex) * 1.18
cw = (PX1 - PX0) / Lx
def xctr(x): return PX0 + (x + 0.5) * cw
def typx(v): return TY1 - v / hmax * (TY1 - TY0)

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="white"/>']
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="28" text-anchor="middle" font-size="15.5" font-weight="bold" fill="#111">'
         f'Stripe order in the doped Hubbard cylinder &#8212; {Lx}&#215;{Wd}, U={head["U"]:g}t, '
         f'&#948;=1/{(Lx*Wd)//head["holes"]} hole doping</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="48" text-anchor="middle" font-size="11.5" fill="#555">'
         f'width-{Wd} cylinder (y-periodic), site-split d=4 fast block path, packed (N, 2Sz) sectors; '
         f'&#967;={head["chi"]}, {head["nsw"]} sweeps (DMRG3S mixer), staggered pinning h={head["hpin"]:g} at x=0; '
         f'E&#8320;={head["E0"]:.4f}t ({head["wall"]:.0f}s)</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{TY0-10}" text-anchor="middle" font-size="12.5" fill="#222" font-weight="bold">'
         'rung-averaged hole density  1 &#8722; n(x)</text>')
g = 0.0
while g <= hmax:
    y = typx(g)
    s.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" stroke="#eee"/>')
    s.append(f'<text x="{PX0-7}" y="{y+4:.1f}" text-anchor="end" font-size="10.5" fill="#444">{g:.2f}</text>')
    g += 0.05
mean_h = head["holes"] / (Lx * Wd)
s.append(f'<line x1="{PX0}" y1="{typx(mean_h):.1f}" x2="{PX1}" y2="{typx(mean_h):.1f}" '
         f'stroke="#999" stroke-dasharray="5,4" stroke-width="1.2"/>')
s.append(f'<text x="{PX1-4}" y="{typx(mean_h)-6:.1f}" text-anchor="end" font-size="10" fill="#777">mean &#948;={mean_h:.3f}</text>')
poly = " ".join(f"{xctr(x):.1f},{typx(holex[x]):.1f}" for x in range(Lx))
s.append(f'<polyline points="{poly}" fill="none" stroke="#d62728" stroke-width="2.2"/>')
for x in range(Lx):
    s.append(f'<circle cx="{xctr(x):.1f}" cy="{typx(holex[x]):.1f}" r="3.6" fill="#d62728"/>')
s.append(f'<line x1="{PX0}" y1="{TY0}" x2="{PX0}" y2="{TY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{TY1}" x2="{PX1}" y2="{TY1}" stroke="#444" stroke-width="1.2"/>')

# bottom panel: lattice map, cell colour = staggered <Sz>, circle = hole density
BY0 = TY1 + 64
ch = cw * 0.92
BY1 = BY0 + Wd * ch
smax = max(abs(v) for v in stag.values()) or 1.0
def col(v):                                          # diverging blue-white-red
    r = max(-1.0, min(1.0, v / smax))
    if r >= 0: R, G, B = 255, int(255 * (1 - r * 0.82)), int(255 * (1 - r * 0.82))
    else:      R, G, B = int(255 * (1 + r * 0.82)), int(255 * (1 + r * 0.82)), 255
    return f"rgb({R},{G},{B})"
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{BY0-30}" text-anchor="middle" font-size="12.5" fill="#222" font-weight="bold">'
         'staggered spin (&#8722;1)<tspan baseline-shift="super" font-size="9">x+y</tspan>&#10216;Sz&#10217; '
         '(colour)  and  per-site hole density (circle area)</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{BY0-14}" text-anchor="middle" font-size="10.5" fill="#666">'
         'uniform colour = antiferromagnetic domain; red&#8596;blue flip = the &#960; phase shift at each charge stripe</text>')
hs_max = max(holes_site.values()) or 1.0
for (x, y), v in stag.items():
    cx, cy = PX0 + x * cw, BY0 + (Wd - 1 - y) * ch
    s.append(f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{cw:.1f}" height="{ch:.1f}" '
             f'fill="{col(v)}" stroke="#ccc" stroke-width="0.5"/>')
    rr = 0.40 * ch * (holes_site[(x, y)] / hs_max) ** 0.5
    s.append(f'<circle cx="{cx+cw/2:.1f}" cy="{cy+ch/2:.1f}" r="{rr:.2f}" fill="none" stroke="#222" stroke-width="1.4"/>')
for x in range(Lx):
    s.append(f'<text x="{PX0+(x+0.5)*cw:.1f}" y="{BY1+16:.1f}" text-anchor="middle" font-size="10" fill="#444">{x}</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{BY1+34:.1f}" text-anchor="middle" font-size="12" fill="#222">column x  (cylinder axis; pinned edge at x=0)</text>')
# colour scale legend
lx0, ly0 = PX0, H - 26
for k in range(60):
    v = (k / 59.0) * 2 - 1
    s.append(f'<rect x="{lx0+k*2.2:.1f}" y="{ly0}" width="2.3" height="11" fill="{col(v*smax)}"/>')
s.append(f'<text x="{lx0-6}" y="{ly0+10}" text-anchor="end" font-size="10" fill="#444">&#8722;{smax:.2f}</text>')
s.append(f'<text x="{lx0+60*2.2+6:.1f}" y="{ly0+10}" font-size="10" fill="#444">+{smax:.2f}</text>')
s.append('</svg>')

out = os.path.join(HERE, "..", "docs", "hubbard_stripes.svg")
open(out, "w", encoding="utf-8").write("\n".join(s))
print("wrote", os.path.normpath(out),
      f"| E0={head['E0']:.6f}  wall={head['wall']:.0f}s  max hole(x)={max(holex):.3f}")
