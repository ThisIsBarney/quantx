"""n_f(t) of the Quantum Advantage Tracker's SU(2) hadron instance on QuantX's U(1) MPS (M16).

The circuits, their pinned QASM files and the readout are those of
benchmarks/su2_hadron_quimb.py (M8); only the simulator differs. quantx_engine.u1_circuit
fuses each circuit into gates that conserve the number of qubits in |1>, and runs it on the
block-sparse MPS with that number as an exact charge: Q = 60 at any bond dimension, where
quimb's dense MPS keeps it only to its truncation (59.934 at chi = 200).

    python benchmarks/su2_hadron_quantx.py trajectory --chi 1600     # n_f after every step
    python benchmarks/su2_hadron_quantx.py same RUN.json REPEAT.json # bit for bit?
    python benchmarks/su2_hadron_quantx.py check --run RUN.json      # against the scan
    python benchmarks/su2_hadron_quantx.py quimb --gates fused --chi 400 --gpu
    python benchmarks/su2_hadron_quantx.py quimb --gates published --cutoff-mode rel --chi 400
    python benchmarks/su2_hadron_quantx.py table     # docs/referee_su2_hadron_quantx_nf_t.csv
    python benchmarks/su2_hadron_quantx.py summary   # step 20 of every run

`trajectory` measures after every Trotter step on a copy of the state, so step 20 is the
end of the same run. `same` checks that two runs of one bond dimension agree bit for bit:
on a machine whose memory can flip bits, a run is kept only if a second one reproduces it.
`quimb` runs the two controls on quimb's dense MPS:
- `--gates fused` gives it the fused gates instead of the published ones. That separates
  what the fusion changes (one truncation per cluster, not one per CX) from what the
  conserved charge does.
- `--cutoff-mode rel` replaces its open-MPS default, "rsum2", which drops up to cutoff times
  the state's weight at every split.

`check` compares a fresh trajectory, from any machine, with the committed chi = 1600 one,
step by step, and fails past --tol: the reproduction CI job
(.github/workflows/su2-hadron-quantx-repro.yml) runs it on a GitHub runner's CPU.
`table` writes the QuantX runs in the columns of docs/referee_su2_hadron_nf_t.csv, beside
the tracker's published values; benchmarks/plot_su2_hadron.py draws it.

Needs the native library (a platform wheel, or QUANTX_LIB_PATH); --threads caps OpenMP. A
library built with CUDA does its large blocks on the GPU, which the run records as its device.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import su2_hadron_quimb as m8  # noqa: E402  (the pinned circuits, the readout, the tracker)

def gate_lines(text: str) -> list[str]:
    """The circuit's gate statements, one per line in these files, in order."""
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.startswith(("OPENQASM", "include", "qreg", "creg"))]


def layouts(lines: list[str], ends: list[int]) -> list[list[int]]:
    """perm after each of the given source positions: the qubit of the initial layout that
    the SWAP gates have left at each position (as su2_hadron_quimb.trajectory tracks it)."""
    perm, out, k = list(range(2 * m8.N_SITES)), [], 0
    for end in ends:
        for ln in lines[k:end]:
            m = m8.SWAP.match(ln)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                perm[a], perm[b] = perm[b], perm[a]
        k = end
        out.append(list(perm))
    return out


def cusolver_loaded() -> bool:
    """Whether the process has loaded cuSOLVER: the native library was built with CUDA and
    splits its large blocks on the GPU. (backend_info's "gpu" is the state-vector backend's,
    not the block engine's; and the CUDA runtime itself may be linked statically.)"""
    if os.name == "nt":
        import ctypes
        get = ctypes.windll.kernel32.GetModuleHandleW
        get.restype = ctypes.c_void_p
        return any(get(f"cusolver64_{v}.dll") for v in ("10", "11", "12", "13"))
    try:
        return "libcusolver" in pathlib.Path("/proc/self/maps").read_text()
    except OSError:
        return False


def fused(args, name):
    """One circuit, fused; the fused-gate count at the end of each step; the layout there."""
    from quantx_engine import u1_circuit
    if not 1 <= args.steps <= 20:
        sys.exit("--steps must be 1 to 20")
    text = m8.qasm_path(args.qasm_dir, name).read_text()
    t0 = time.time()
    circ = u1_circuit.from_qasm(text)
    fuse_s = time.time() - t0
    lines = gate_lines(text)
    assert len(lines) == circ.source_gates
    n_x = next(i for i, ln in enumerate(lines) if not ln.startswith("x "))
    assert circ.initial_end == n_x, (circ.initial_end, n_x)   # the preparation X gates
    ends = [min(n_x + m8.STEP_GATES * k, len(lines)) for k in range(1, args.steps + 1)]
    return circ, u1_circuit.measure_points(circ, ends), layouts(lines, ends), fuse_s


def finish(out, args, stem):
    out["n_f"] = [float((m8.site_nf(m["z_logical"]) - m8.site_nf(v["z_logical"])).sum())
                  for m, v in zip(out["meson"]["steps"], out["SCV"]["steps"])]
    for k, v in enumerate(out["n_f"], 1):
        print(f"n_f({k:2d}) = {v:+.5f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if args.steps == 20 else f"_steps{args.steps}"
    dest = args.out_dir / f"{stem}_chi{args.chi}{tag}{args.tag}.json"
    dest.write_text(json.dumps(out))
    print("wrote", dest)


def trajectory(args):
    # OpenMP reads its thread count when the library loads, so before the engine is imported.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import numpy as np
    import quantx_engine
    from quantx_engine import u1_circuit

    out = {"engine": "quantx", "chi": args.chi, "cutoff": args.cutoff,
           "threads": args.threads, "steps": args.steps,
           "backend": quantx_engine.backend_info()}
    for name in m8.CIRCUITS:
        circ, when, perms, fuse_s = fused(args, name)
        r = u1_circuit.run(circ, max_bond=args.chi, cutoff=args.cutoff, measure_after=when)
        steps = []
        for k, (meas, perm) in enumerate(zip(r.measurements, perms), 1):
            z_logical = np.empty_like(meas.z)
            z_logical[perm] = meas.z
            steps.append({"step": k, "z_logical": z_logical.tolist(), "norm2": meas.norm2,
                          "max_bond": meas.max_bond,
                          "Q": float((1.0 - meas.z).sum() / 2.0),
                          "layout_restored": perm == list(range(2 * m8.N_SITES))})
        out[name] = {"steps": steps, "ones": r.ones, "fused_gates": len(circ.gates),
                     "max_bond": r.max_bond, "truncation_sum": r.truncation_sum,
                     "truncation_max": r.truncation_max, "seconds": r.wall_time_s,
                     "fuse_seconds": fuse_s}
        print(f"  {name}: chi={args.chi} {r.wall_time_s:.0f} s, max bond {r.max_bond}, "
              f"norm^2 {steps[-1]['norm2']:.6f}, Q = {steps[-1]['Q']:.12f}", flush=True)
    out["device"] = "gpu" if cusolver_loaded() else "cpu"
    finish(out, args, "su2_hadron_quantx")


def quimb_control(args):
    """quimb's dense MPS on the fused or the published gates, under either cutoff rule.

    Two controls on what separates a QuantX run from quimb's. quimb truncates after every gate
    it is given: after each CX of a cluster on the published circuit, once per cluster on the
    fused one. And its open-MPS default, cutoff_mode 'rsum2', drops up to cutoff times the
    weight at every split, where 'rel' drops only singular values below cutoff times the
    largest (QuantX's rule is of that kind).
    """
    if not args.gpu:     # BLAS reads its thread count when it loads
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = str(args.threads)
    import numpy as np
    import quimb as qu
    import quimb.tensor as qtn
    from quimb.tensor.circuit import parse_openqasm2_file

    z_op, gate_opts, backend, to_device = qu.pauli("Z").real, {}, {}, np.asarray
    if args.cutoff_mode != "rsum2":
        gate_opts["cutoff_mode"] = args.cutoff_mode
    if args.gpu:
        import cupy
        z_op, to_device = cupy.asarray(z_op), cupy.asarray
        backend, gate_opts["method"] = {"to_backend": cupy.asarray}, "eig"
    out = {"engine": f"quimb-{args.gates}", "chi": args.chi, "cutoff": args.cutoff,
           "cutoff_mode": args.cutoff_mode, "gpu": args.gpu,
           "threads": None if args.gpu else args.threads, "steps": args.steps}
    for name in m8.CIRCUITS:
        circ, when, perms, _ = fused(args, name)
        mps = qtn.CircuitMPS(2 * m8.N_SITES, max_bond=args.chi, cutoff=args.cutoff,
                             gate_opts=gate_opts, **backend)
        if args.gates == "fused":
            for q, bit in enumerate(circ.initial):
                if bit:
                    mps.apply_gate("X", q)
            todo = [[(to_device(g.u), (g.q,) if g.arity == 1 else (g.q, g.q + 1))
                     for g in circ.gates[a:b]] for a, b in zip([0] + when[:-1], when)]
        else:      # the published gates, step by step, as su2_hadron_quimb.trajectory runs them
            gates = parse_openqasm2_file(str(m8.qasm_path(args.qasm_dir, name)))["gates"]
            mps.apply_gates(gates[:circ.initial_end])
            starts = [circ.initial_end + m8.STEP_GATES * k for k in range(args.steps)]
            todo = [gates[a:min(a + m8.STEP_GATES, len(gates))] for a in starts]
        steps, t0 = [], time.time()
        for k, (block, perm) in enumerate(zip(todo, perms), 1):
            if args.gates == "fused":
                for u, where in block:
                    mps.apply_gate_raw(u, where)
            else:
                mps.apply_gates(block)
            z, norm2 = m8.measure(mps, z_op)
            z_logical = np.empty_like(z)
            z_logical[perm] = z
            steps.append({"step": k, "z_logical": z_logical.tolist(), "norm2": norm2,
                          "Q": float((1.0 - z).sum() / 2.0)})
            print(f"  {name} step {k:2d}: norm^2 {norm2:.9f}, {time.time() - t0:.0f} s",
                  flush=True)
        out[name] = {"steps": steps, "seconds": time.time() - t0}
    finish(out, args, f"su2_hadron_quimb_{args.gates}_{args.cutoff_mode}"
                      f"{'_gpu' if args.gpu else ''}")


def same(args):
    """Exit 1 unless two runs agree bit for bit, in every <Z> of every step."""
    a, b = (json.loads(p.read_text()) for p in (args.a, args.b))
    if (a["chi"], a["cutoff"], a["steps"]) != (b["chi"], b["cutoff"], b["steps"]):
        sys.exit("the two runs differ in chi, cutoff or steps")
    worst = 0.0
    for name in m8.CIRCUITS:
        for sa, sb in zip(a[name]["steps"], b[name]["steps"]):
            worst = max(worst, max(abs(x - y) for x, y in zip(sa["z_logical"], sb["z_logical"])))
    print(f"largest difference in <Z>: {worst:.3e}")
    sys.exit(0 if worst == 0.0 else 1)


def check(args):
    """A trajectory against the committed QuantX scan, step by step; exit 1 past --tol."""
    d = json.loads(args.run.read_text())
    if not d["n_f"]:
        sys.exit(f"{args.run} has no steps")
    with open(args.against, newline="") as f:
        ref = {int(r["step"]): r for r in csv.DictReader(f)}
    if args.column not in ref[1]:
        sys.exit(f"{args.against} has no column {args.column}")
    print(f"step  this run ({d.get('device', '?')}, chi = {d['chi']})  {args.column}   difference")
    worst, non_finite = 0.0, []
    for k, v in enumerate(d["n_f"], 1):
        r = float(ref[k][args.column])
        if not (math.isfinite(v) and math.isfinite(r)):   # max() would pass a NaN by
            non_finite.append(k)
        else:
            worst = max(worst, abs(v - r))
        print(f"{k:4d}  {v:+.6f}  {r:+.6f}  {v - r:+.2e}")
    if non_finite:
        sys.exit(f"non-finite n_f at step(s) {non_finite}: nothing to compare")
    ok = worst <= args.tol
    print(f"largest difference {worst:.2e} over {len(d['n_f'])} steps: "
          f"{'within' if ok else 'PAST'} the tolerance {args.tol:g}")
    if not ok:
        sys.exit(1)


def _runs(out_dir, pattern):
    """Full 20-step runs, one per (chi, device); a repeat must match its run (see `same`)."""
    runs = {}
    for f in sorted(out_dir.glob(pattern)):
        d = json.loads(f.read_text())
        if d["steps"] != 20:
            continue
        dev = d["device"] if "device" in d else ("gpu" if d["gpu"] else "cpu")
        runs.setdefault((d["chi"], dev), d)
    return runs


def table(args):
    """n_f(t) of every QuantX run, in the columns of docs/referee_su2_hadron_nf_t.csv."""
    runs = _runs(args.out_dir, "su2_hadron_quantx_chi*.json")
    keys = sorted(runs)
    print("step," + ",".join(f"n_f_chi{c}_{dev},norm2_meson_chi{c}_{dev},"
                             f"norm2_vacuum_chi{c}_{dev}" for c, dev in keys)
          + ",TN,PP_CPU,PP_GPU,QPU")
    for s in range(1, 21):
        ours = ",".join(f"{runs[k]['n_f'][s - 1]:.6f},{runs[k]['meson']['steps'][s - 1]['norm2']:.6f},"
                        f"{runs[k]['SCV']['steps'][s - 1]['norm2']:.6f}" for k in keys)
        print(f"{s},{ours}," + ",".join(f"{x:.5f}" for x in m8.PUBLISHED[s]))


def summary(args):
    """Step 20 of every run, QuantX's and the quimb controls', with Q and the run time."""
    print("engine,chi,device,n_f_20,norm2_meson,norm2_vacuum,Q_meson_min,Q_meson_max,"
          "seconds_both")
    for pattern in ("su2_hadron_quantx_chi*.json", "su2_hadron_quimb_*_chi*.json"):
        for (c, dev), d in sorted(_runs(args.out_dir, pattern).items()):
            qs = [st["Q"] for st in d["meson"]["steps"]]
            engine = d["engine"] + (f"-{d['cutoff_mode']}" if "cutoff_mode" in d else "")
            print(f"{engine},{c},{dev},{d['n_f'][-1]:.5f},"
                  f"{d['meson']['steps'][-1]['norm2']:.6f},{d['SCV']['steps'][-1]['norm2']:.6f},"
                  f"{min(qs):.9f},{max(qs):.9f},{d['meson']['seconds'] + d['SCV']['seconds']:.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("trajectory")
    qf = sub.add_parser("quimb")
    qf.add_argument("--gates", choices=("fused", "published"), default="fused")
    qf.add_argument("--cutoff-mode", choices=("rsum2", "rel"), default="rsum2",
                    help="quimb's cutoff rule; rsum2 is its open-MPS default")
    qf.add_argument("--gpu", action="store_true")
    for q in (tr, qf):
        q.add_argument("--chi", type=int, required=True)
        q.add_argument("--cutoff", type=float, default=1e-10)
        q.add_argument("--threads", type=int, default=2)
        q.add_argument("--steps", type=int, default=20)
        q.add_argument("--tag", default="", help="appended to the file name, e.g. _repeat")
    ck = sub.add_parser("check")
    ck.add_argument("--run", type=pathlib.Path, required=True)
    ck.add_argument("--against", type=pathlib.Path, default=pathlib.Path(__file__).resolve()
                    .parent.parent / "docs" / "referee_su2_hadron_quantx_nf_t.csv")
    ck.add_argument("--column", default="n_f_chi1600_gpu")
    ck.add_argument("--tol", type=float, default=1e-4)
    sm = sub.add_parser("same")
    sm.add_argument("a", type=pathlib.Path)
    sm.add_argument("b", type=pathlib.Path)
    tb = sub.add_parser("table")
    su = sub.add_parser("summary")
    for q in (tr, qf, tb, su):
        q.add_argument("--qasm-dir", type=pathlib.Path, default=pathlib.Path("su2_hadron_qasm"))
        q.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("su2_hadron_runs"))
    args = p.parse_args()
    {"trajectory": trajectory, "quimb": quimb_control, "check": check, "same": same,
     "table": table, "summary": summary}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
