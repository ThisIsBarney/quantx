"""n_f(t) of the tracker's SU(2) hadron circuits, step by step, against the published methods.

Reads docs/referee_su2_hadron_nf_t.csv (from `su2_hadron_quimb.py trajectory-table`) and
docs/referee_su2_hadron_nf.csv (the bond-dimension scan, for the chi = 1600 point at step
20), and writes docs/referee_su2_hadron_nf_t.svg. Two panels share the step axis: n_f(t) for
the MPS and the four published methods, and each method minus the MPS at the steps where the
MPS has converged: the largest-chi run and the one at half its chi (same device) agree to
TOL, and the largest-chi run truncated the norm by less than 5e-5. That is convergence to a stated
tolerance, not an error bound. The numbers the doc quotes come from summaries(). Pure stdlib,
like the other plotters.

Run: python benchmarks/plot_su2_hadron.py
"""
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs")
METHODS = ("TN", "PP_CPU", "PP_GPU", "QPU")
LABELS = {"MPS": "MPS of the circuits (this scan)", "TN": "Tensor network (TDVP of H)",
          "PP_CPU": "Pauli propagation, CPU", "PP_GPU": "Pauli propagation, GPU",
          "QPU": "QPU"}
SHORT = {"TN": "TN", "PP_CPU": "PP, CPU", "PP_GPU": "PP, GPU", "QPU": "QPU"}
# Categorical slots in fixed order (validated: adjacent CVD and normal-vision floors pass on
# the light and dark surfaces; aqua, yellow and magenta sit below 3:1 on light, so the legend
# and the doc's table carry identity, not colour alone).
SLOTS = {"MPS": ("#2a78d6", "#3987e5"), "TN": ("#eb6834", "#d95926"),
         "PP_CPU": ("#1baf7a", "#199e70"), "PP_GPU": ("#eda100", "#c98500"),
         "QPU": ("#e87ba4", "#d55181")}
CONVERGED_NORM2 = 1 - 5e-5
TOL = 5e-4                    # |n_f(chi) - n_f(chi/2)| at a converged step


def load(traj=os.path.join(DOCS, "referee_su2_hadron_nf_t.csv"),
         scan=os.path.join(DOCS, "referee_su2_hadron_nf.csv")):
    with open(traj, newline="") as f:
        rows = list(csv.DictReader(f))
    runs = {}                               # (chi, device) of every n_f_chi<chi>_<device> column
    for k in rows[0]:
        if k.startswith("n_f_chi"):
            c, dev = k[len("n_f_chi"):].split("_")
            runs[(int(c), dev)] = k[len("n_f_"):]
    chi = max(c for c, _ in runs)
    devs = sorted(dev for c, dev in runs if c == chi)
    if len(devs) != 1:
        raise ValueError(f"chi = {chi} has runs on {devs}: plot one of them")
    dev = devs[0]
    lab, half = runs[(chi, dev)], runs.get((chi // 2, dev))
    if half is None:
        raise ValueError(f"no chi = {chi // 2} {dev} run to check chi = {chi} against")
    data = {"steps": [int(r["step"]) for r in rows], "chi": chi,
            "MPS": [float(r[f"n_f_{lab}"]) for r in rows],
            "half": [float(r[f"n_f_{half}"]) for r in rows],
            "converged": [abs(float(r[f"n_f_{lab}"]) - float(r[f"n_f_{half}"])) <= TOL
                          and min(float(r[f"norm2_meson_{lab}"]),
                                  float(r[f"norm2_vacuum_{lab}"])) >= CONVERGED_NORM2
                          for r in rows]}
    # converged means every step up to here, not isolated steps further on
    first_bad = next((i for i, c in enumerate(data["converged"]) if not c), len(rows))
    data["converged"] = [i < first_bad for i in range(len(rows))]
    for m in METHODS:
        data[m] = [float(r[m]) for r in rows]
    with open(scan, newline="") as f:
        runs = {int(r["chi"]): float(r["n_f_20"]) for r in csv.DictReader(f) if r["device"] == "GPU"}
    best = max(runs)
    data["final"] = (best, runs[best], abs(runs[best] - runs[best // 2]))
    return data


def summaries(d):
    """The numbers docs/referee_su2_hadron.md quotes."""
    conv = [i for i, c in enumerate(d["converged"]) if c]
    out = {"converged_through": d["steps"][conv[-1]],
           "half_diff": max(abs(d["MPS"][i] - d["half"][i]) for i in conv)}
    for m in METHODS:
        dev = [d[m][i] - d["MPS"][i] for i in conv]
        k = max(range(len(dev)), key=lambda j: abs(dev[j]))
        above = next((d["steps"][conv[j]] for j in range(len(dev))
                      if all(x > 0 for x in dev[j:])), None)
        out[m] = {"step1": d[m][0] - d["MPS"][0], "max": dev[k],
                  "max_step": d["steps"][conv[k]], "at_last": dev[-1], "above_from": above}
    return out


def _up1(x):
    """x rounded up to one significant figure, as text (3.3e-4 -> '4e-4')."""
    e = math.floor(math.log10(x))
    return f"{math.ceil(x / 10 ** e)}e{e}"


def render(d, path=os.path.join(DOCS, "referee_su2_hadron_nf_t.svg")):
    W, H = 760, 704
    L, R = 64, 600                        # plot area x range (labels to the right)
    A_top, A_bot = 156, 394               # panel A y range
    B_top, B_bot = 468, 654               # panel B y range
    ya = (-1.0, 2.0)
    yb = (-0.25, 0.35)
    steps = d["steps"]
    X = lambda s: L + (s - 0.5) / 20.0 * (R - L)
    YA = lambda v: A_bot - (v - ya[0]) / (ya[1] - ya[0]) * (A_bot - A_top)
    YB = lambda v: B_bot - (v - yb[0]) / (yb[1] - yb[0]) * (B_bot - B_top)
    css = ["svg{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;"
           "--axis:#c3c2b7;--band:#f0efec;"
           + "".join(f"--{k.lower()}:{v[0]};" for k, v in SLOTS.items()) + "}",
           "@media (prefers-color-scheme: dark){svg{--surface:#1a1a19;--ink:#f0efec;"
           "--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--band:#242422;"
           + "".join(f"--{k.lower()}:{v[1]};" for k, v in SLOTS.items()) + "}}",
           "text{font-family:system-ui,-apple-system,'Segoe UI',sans-serif}",
           ".t{fill:var(--ink);font-size:15px;font-weight:600}",
           ".s{fill:var(--ink2);font-size:12px}", ".m{fill:var(--muted);font-size:11px}",
           ".l{fill:var(--ink2);font-size:11.5px}",
           "line.g{stroke:var(--grid);stroke-width:1}", "line.a{stroke:var(--axis);stroke-width:1}",
           ".ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
         "<title id=\"title\">n_f(t) of the SU(2) hadron circuits: an exact MPS against the "
         "published methods</title>",
         "<desc id=\"desc\">Two panels over Trotter steps 1 to 20. The MPS has converged "
         f"through step {summaries(d)['converged_through']}; the classical methods drift above "
         "it. docs/referee_su2_hadron_nf_t.csv holds every value.</desc>",
         "<style>" + "".join(css) + "</style>",
         f'<rect width="{W}" height="{H}" fill="var(--surface)"/>',
         '<text class="t" x="24" y="32">n_f(t) of the tracker\'s SU(2) hadron circuits, '
         'step by step</text>',
         f'<text class="s" x="24" y="52">The MPS (χ = {d["chi"]}) has converged through step '
         f'{summaries(d)["converged_through"]}: χ = {d["chi"] // 2} agrees to '
         f'{_up1(summaries(d)["half_diff"])}, and the norm stays within 5e-5 of 1.</text>',
         '<text class="s" x="24" y="68">The classical methods drift above it; the QPU swings '
         'both ways. The CSV beside this figure holds every value.</text>']
    # legend
    for row, keys in enumerate((("MPS", "TN", "PP_CPU"), ("PP_GPU", "QPU"))):
        lx, ly = 24, 92 + 18 * row
        for k in keys:
            o.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 18}" y2="{ly}" '
                     f'stroke="var(--{k.lower()})" class="ln"/>')
            o.append(f'<text class="l" x="{lx + 24}" y="{ly + 4}">{LABELS[k]}</text>')
            lx += 24 + 6.4 * len(LABELS[k]) + 26
    # panel A
    last_conv = summaries(d)["converged_through"]
    o.append(f'<rect x="{X(last_conv + 0.5):.1f}" y="{A_top}" '
             f'width="{R - X(last_conv + 0.5):.1f}" height="{A_bot - A_top}" fill="var(--band)"/>')
    o.append(f'<text class="m" x="{X(last_conv + 0.5) + 6:.1f}" y="{A_top + 14}">MPS not '
             f'converged</text>')
    for v in (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0):
        cls = "a" if v == 0 else "g"
        o.append(f'<line class="{cls}" x1="{L}" y1="{YA(v):.1f}" x2="{R}" y2="{YA(v):.1f}"/>')
        o.append(f'<text class="m" x="{L - 8}" y="{YA(v) + 4:.1f}" text-anchor="end">{v:g}</text>')
    o.append(f'<text class="s" x="{L}" y="{A_top - 10}">n_f(t), the vacuum-subtracted fermion '
             f'number</text>')

    def path_of(vals, Y, idx):
        return " ".join(f"{'M' if j == 0 else 'L'}{X(steps[i]):.1f},{Y(vals[i]):.1f}"
                        for j, i in enumerate(idx))
    allidx = list(range(len(steps)))
    for k in METHODS:
        o.append(f'<path class="ln" stroke="var(--{k.lower()})" d="{path_of(d[k], YA, allidx)}"/>')
    ex = [i for i in allidx if d["converged"][i]]
    tail = [ex[-1]] + [i for i in allidx if not d["converged"][i] and steps[i] < 20]
    o.append(f'<path class="ln" stroke="var(--mps)" d="{path_of(d["MPS"], YA, ex)}"/>')
    o.append(f'<path class="ln" stroke="var(--mps)" stroke-dasharray="4 4" '
             f'd="{path_of(d["MPS"], YA, tail)}"/>')
    chi_f, v_f, err = d["final"]
    x20 = X(20)
    o.append(f'<line x1="{x20:.1f}" y1="{YA(v_f - err):.1f}" x2="{x20:.1f}" y2="{YA(v_f + err):.1f}" '
             f'stroke="var(--mps)" stroke-width="2"/>')
    o.append(f'<circle cx="{x20:.1f}" cy="{YA(v_f):.1f}" r="4.5" fill="var(--mps)" '
             f'stroke="var(--surface)" stroke-width="2"/>')
    o.append(f'<text class="l" x="{x20 + 12:.1f}" y="{YA(v_f + err) - 22:.1f}">step 20, '
             f'χ = {chi_f}:</text>')
    o.append(f'<text class="l" x="{x20 + 12:.1f}" y="{YA(v_f + err) - 8:.1f}">'
             f'{v_f:.2f} ± {err:.2f}</text>')
    for k in METHODS:                      # end dots at step 20
        o.append(f'<circle cx="{x20:.1f}" cy="{YA(d[k][-1]):.1f}" r="4" fill="var(--{k.lower()})" '
                 f'stroke="var(--surface)" stroke-width="2"/>')
    # panel B: method minus MPS where the MPS is exact
    o.append(f'<text class="s" x="{L}" y="{B_top - 10}">Each method minus the MPS, where the '
             f'MPS has converged (steps 1–{last_conv})</text>')
    for v in (-0.2, -0.1, 0.0, 0.1, 0.2, 0.3):
        cls = "a" if abs(v) < 1e-12 else "g"
        o.append(f'<line class="{cls}" x1="{L}" y1="{YB(v):.1f}" x2="{R}" y2="{YB(v):.1f}"/>')
        o.append(f'<text class="m" x="{L - 8}" y="{YB(v) + 4:.1f}" text-anchor="end">'
                 f'{v:+.1f}</text>' if v else
                 f'<text class="m" x="{L - 8}" y="{YB(v) + 4:.1f}" text-anchor="end">0</text>')
    ends = []
    for k in METHODS:
        dev = [d[k][i] - d["MPS"][i] for i in allidx]
        o.append(f'<path class="ln" stroke="var(--{k.lower()})" d="{path_of(dev, YB, ex)}"/>')
        o.append(f'<circle cx="{X(steps[ex[-1]]):.1f}" cy="{YB(dev[ex[-1]]):.1f}" r="4" '
                 f'fill="var(--{k.lower()})" stroke="var(--surface)" stroke-width="2"/>')
        ends.append([YB(dev[ex[-1]]), k])
    ends.sort()                             # direct labels, pushed apart to 15 px
    for j in range(1, len(ends)):
        ends[j][0] = max(ends[j][0], ends[j - 1][0] + 15)
    for y, k in ends:
        o.append(f'<text class="l" x="{X(steps[ex[-1]]) + 10:.1f}" y="{y + 4:.1f}">{SHORT[k]}</text>')
    for s in (1, 5, 10, 15, 20):            # shared step axis
        for yb_ in (A_bot, B_bot):
            o.append(f'<text class="m" x="{X(s):.1f}" y="{yb_ + 16}" text-anchor="middle">{s}</text>')
    o.append(f'<text class="m" x="{(L + R) / 2:.1f}" y="{B_bot + 34}" text-anchor="middle">'
             f'Trotter step</text>')
    o.append("</svg>")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(o) + "\n")
    return path


if __name__ == "__main__":
    d = load()
    s = summaries(d)
    print(f"MPS converged through step {s['converged_through']}: chi/2 agrees to "
          f"{s['half_diff']:.1e} ({_up1(s['half_diff'])})")
    for m in METHODS:
        v = s[m]
        print(f"{m:7s} step 1 {v['step1']:+.5f}   largest {v['max']:+.4f} at step {v['max_step']}"
              f"   at step {s['converged_through']} {v['at_last']:+.4f}   above from step "
              f"{v['above_from']}")
    print("wrote", render(d))
