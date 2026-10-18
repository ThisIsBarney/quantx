"""n_f(20) of the Quantum Advantage Tracker's SU(2) hadron instance, by an MPS of its circuits.

M8 (docs/referee_su2_hadron.md): how large a bond dimension the published circuits need. Each
run simulates the two circuits, ``x_100_meson.qasm`` and its vacuum ``x_100_SCV.qasm`` (120
qubits, 32,072 gates), as a quimb MPS truncated at one bond dimension, and reads the
vacuum-subtracted fermion density after the last step,

    n_f(20) = sum_r [n_f(r)]_meson - [n_f(r)]_vacuum,
    n_f(r) = n_i(r) + n_o(r) on even r,  2 - n_i(r) - n_o(r) on odd r,  n = (1 - <Z>) / 2,

with n_i(r) on qubit r and n_o(r) on qubit 60 + r as the circuits' first gates lay them out
(arXiv:2602.18080). The transpiled circuits' SWAP gates leave the qubits permuted at the end,
so the final <Z> are mapped back through that permutation first; read without it, the sum
stays at its starting value 2 for every bond dimension.

    python benchmarks/su2_hadron_quimb.py run --chi 800 --gpu         # one bond dimension
    python benchmarks/su2_hadron_quimb.py table                       # every saved run
    python benchmarks/su2_hadron_quimb.py trajectory --chi 800 --gpu  # n_f after every step
    python benchmarks/su2_hadron_quimb.py trajectory-table            # with the tracker's

The circuits are 20 blocks of 1,602 gates after their initial X gates, one per Trotter step.
After each of the first 19 blocks the SWAP gates have returned the qubits to their initial
layout; the last block lacks its final 30-SWAP layer, which is the permutation n_f(20) is
read through. `trajectory` measures after every block, on a copy of the state, so it reads
n_f(t) at every step with the same gates and truncations as `run`.

Needs quimb (1.11), and CuPy for --gpu. The QASM files are downloaded into --qasm-dir the
first time, from the tracker's commit a62dc46 (2026-07-07, the files' last change), and
every run checks their SHA-256 against the files the published scan used. --gpu keeps the MPS in complex128 on the GPU and splits each gate
by an eigendecomposition of the Gram matrix (quimb's method "eig"), which cuSOLVER does far
faster than an SVD; it agrees with the CPU's SVD to 3e-3 in n_f(20) at chi = 200. On the CPU,
--threads caps BLAS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import urllib.request

N_SITES = 60
TRACKER_COMMIT = "a62dc4663f47e2c5503e07392479ec14a139d590"
URL = ("https://raw.githubusercontent.com/quantum-advantage-tracker/"
       "quantum-advantage-tracker.github.io/" + TRACKER_COMMIT + "/data/observable-estimations/"
       "circuit-models/su2_hadron_dynamics_lsh/x_100_{}.qasm")
SHA256 = {
    "meson": "28b3d82a2572dac60110c94a0f58a945a117d414e94446866e0d87cc5121b4f9",
    "SCV": "af269dd7aeacfe6c9fef00bdab3bd1d0a641f5ee99be67c56ebf73a15791b6da",
}
CIRCUITS = ("meson", "SCV")


def qasm_path(qasm_dir: pathlib.Path, name: str) -> pathlib.Path:
    path = qasm_dir / f"x_100_{name}.qasm"
    if not path.exists():
        qasm_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL.format(name)}", flush=True)
        urllib.request.urlretrieve(URL.format(name), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256[name]:
        sys.exit(f"{path}: SHA-256 {digest} is not the published scan's {SHA256[name]}; "
                 "delete it to download the pinned version")
    return path


def final_permutation(path: pathlib.Path) -> list[int]:
    """perm[q]: the qubit of the initial layout that the SWAP gates leave at position q."""
    perm = list(range(2 * N_SITES))
    for line in path.read_text().splitlines():
        m = re.match(r"swap\s+q\[(\d+)\],\s*q\[(\d+)\];", line.strip())
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            perm[a], perm[b] = perm[b], perm[a]
    return perm


def site_nf(z):
    import numpy as np
    n = (1.0 - np.asarray(z)) / 2.0
    r = np.arange(N_SITES)
    return np.where(r % 2 == 0, n[:N_SITES] + n[N_SITES:], 2.0 - n[:N_SITES] - n[N_SITES:])


STEP_GATES = 1602            # gates per Trotter step after the initial X gates
# n_f(t) at steps 1-20 from the tracker's README for this instance: the tensor network (the
# Hamiltonian's TDVP), Pauli propagation on CPU and on GPU, and the QPU.
PUBLISHED = {
    1: (1.824222901, 1.825335615, 1.825336, 1.77264257),
    2: (1.343818565, 1.34880517, 1.347871, 1.234719061),
    3: (0.6852977079, 0.698419082, 0.693895, 0.7005505103),
    4: (0.01781852652, 0.04434743353, 0.0317106, 0.1125808617),
    5: (-0.4957689977, -0.4510787869, -0.47774275, -0.3550984939),
    6: (-0.7447406887, -0.6794112084, -0.725069325, -0.4792900767),
    7: (-0.6994223897, -0.6144203611, -0.68012595, -0.405844541),
    8: (-0.4149706965, -0.3149685829, -0.3959514, -0.3160420563),
    9: (-0.009710517986, 0.09777330494, 0.010149475, -0.01701975315),
    10: (0.3744026048, 0.4805343441, 0.394226, 0.1431540203),
    11: (0.6163184396, 0.7136997525, 0.6325666, 0.3540697043),
    12: (0.6514199805, 0.7369940536, 0.6619868, 0.4092782333),
    13: (0.4873750145, 0.5641770138, 0.4951614125, 0.2630479833),
    14: (0.1962856389, 0.2712616284, 0.2069994, 0.1957298805),
    15: (-0.1131674269, -0.03107728475, -0.096355725, 0.00494274374),
    16: (-0.3336240614, -0.2318235614, -0.314319025, -0.08874021367),
    17: (-0.3941897866, -0.2691116404, -0.3787042075, -0.1962503187),
    18: (-0.2807460955, -0.1325140013, -0.272586375, -0.1673053457),
    19: (-0.03330018147, 0.1236920782, -0.035842523, -0.1490836294),
    20: (0.2778348725, 0.4081775866, 0.2456276125, -0.07723426024),
}
SWAP = re.compile(r"swap\s+q\[(\d+)\],\s*q\[(\d+)\];")


def measure(circ, z_op):
    """<Z_q> of every qubit, read on a copy of the state, and the state's squared norm.

    The squared norm is the weight the truncations left. quimb's own fidelity_estimate agrees
    with it at the end of the circuit, but mid-way its recorded orthogonality centre is stale.
    """
    import numpy as np
    psi = circ._psi.copy()
    psi.canonize(0, inplace=True)
    z, norm2 = [], None
    for q in range(psi.L):
        if q > 0:
            psi.canonize(q, cur_orthog=q - 1, inplace=True)
        t = psi[q]
        k = psi.site_ind(q)
        nn = float((t.H @ t).real)
        norm2 = nn if norm2 is None else norm2
        z.append(float((t.H @ t.gate(z_op, k)).real) / nn)
    return np.array(z), norm2


def simulate(path: pathlib.Path, chi: int, cutoff: float, gpu: bool):
    import numpy as np
    import quimb as qu
    import quimb.tensor as qtn
    from quimb.tensor.circuit import parse_openqasm2_file

    z_op = qu.pauli("Z").real
    opts = {}
    if gpu:
        import cupy
        z_op = cupy.asarray(z_op)
        opts = {"to_backend": cupy.asarray, "gate_opts": {"method": "eig"}}
    t0 = time.time()
    gates = parse_openqasm2_file(str(path))["gates"]
    circ = qtn.CircuitMPS(2 * N_SITES, max_bond=chi, cutoff=cutoff, **opts)
    chunk = max(1, len(gates) // 10)
    for i in range(0, len(gates), chunk):
        circ.apply_gates(gates[i:i + chunk])
        print(f"  {path.name} chi={chi}: {min(i + chunk, len(gates))}/{len(gates)} gates, "
              f"{time.time() - t0:.0f} s, max bond {max(circ.psi.bond_sizes())}", flush=True)
    fidelity = float(circ.fidelity_estimate())
    psi = circ.psi
    psi.canonize(0, inplace=True)
    z = []
    for q in range(psi.L):
        if q > 0:
            psi.canonize(q, cur_orthog=q - 1, inplace=True)
        t = psi[q]
        k = psi.site_ind(q)
        z.append(float((t.H @ t.gate(z_op, k)).real / (t.H @ t).real))
    perm = final_permutation(path)
    z = np.array(z)
    z_logical = np.empty_like(z)
    z_logical[perm] = z
    return {"z_physical": z.tolist(), "z_logical": z_logical.tolist(),
            "fidelity_estimate": fidelity, "max_bond": max(psi.bond_sizes()),
            "Q": float((1.0 - z_logical).sum() / 2.0), "seconds": time.time() - t0}


def run(args):
    if not args.gpu:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = str(args.threads)
    out = {"chi": args.chi, "cutoff": args.cutoff, "gpu": args.gpu,
           "threads": None if args.gpu else args.threads}
    for name in CIRCUITS:
        res = simulate(qasm_path(args.qasm_dir, name), args.chi, args.cutoff, args.gpu)
        out[name] = res
        print(f"{name}: chi={args.chi} fidelity estimate {res['fidelity_estimate']:.6f}, "
              f"Q = {res['Q']:.6f}, {res['seconds']:.0f} s", flush=True)
    profile = site_nf(out["meson"]["z_logical"]) - site_nf(out["SCV"]["z_logical"])
    unmapped = site_nf(out["meson"]["z_physical"]) - site_nf(out["SCV"]["z_physical"])
    out["n_f"] = float(profile.sum())
    out["n_f_profile"] = profile.tolist()
    out["n_f_unpermuted"] = float(unmapped.sum())
    print(f"chi={args.chi}: n_f(20) = {out['n_f']:.6f} (read without the permutation: "
          f"{out['n_f_unpermuted']:.6f})")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"su2_hadron_chi{args.chi}{'_gpu' if args.gpu else ''}.json"
    (args.out_dir / name).write_text(json.dumps(out))


def table(args):
    rows = []
    for f in sorted(args.out_dir.glob("su2_hadron_chi*.json")):
        d = json.loads(f.read_text())
        rows.append(d)
    rows.sort(key=lambda d: (d["chi"], not d["gpu"]))
    print("chi,device,fidelity_meson,fidelity_vacuum,Q_meson,Q_vacuum,n_f_20,seconds")
    for d in rows:
        dev = "GPU" if d["gpu"] else f"CPU({d['threads']} threads)"
        print(f"{d['chi']},{dev},{d['meson']['fidelity_estimate']:.4f},"
              f"{d['SCV']['fidelity_estimate']:.4f},{d['meson']['Q']:.5f},{d['SCV']['Q']:.5f},"
              f"{d['n_f']:.4f},{d['meson']['seconds'] + d['SCV']['seconds']:.0f}")


def trajectory(args):
    import numpy as np
    import quimb as qu
    import quimb.tensor as qtn
    from quimb.tensor.circuit import parse_openqasm2_file

    if not args.gpu:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[v] = str(args.threads)
    z_op = qu.pauli("Z").real
    opts = {}
    if args.gpu:
        import cupy
        z_op = cupy.asarray(z_op)
        opts = {"to_backend": cupy.asarray, "gate_opts": {"method": "eig"}}
    out = {"chi": args.chi, "cutoff": args.cutoff, "gpu": args.gpu}
    for name in CIRCUITS:
        path = qasm_path(args.qasm_dir, name)
        gates = parse_openqasm2_file(str(path))["gates"]
        lines = [ln.strip() for ln in path.read_text().splitlines()
                 if ln.strip() and not ln.startswith(("OPENQASM", "include", "qreg", "creg"))]
        assert len(lines) == len(gates)
        n_x = next(i for i, ln in enumerate(lines) if not ln.startswith("x "))
        circ = qtn.CircuitMPS(2 * N_SITES, max_bond=args.chi, cutoff=args.cutoff, **opts)
        circ.apply_gates(gates[:n_x])
        perm, steps, t0 = list(range(2 * N_SITES)), [], time.time()
        for k in range(1, 21):
            lo, hi = n_x + STEP_GATES * (k - 1), min(n_x + STEP_GATES * k, len(gates))
            circ.apply_gates(gates[lo:hi])
            for ln in lines[lo:hi]:
                m = SWAP.match(ln)
                if m:
                    a, b = int(m.group(1)), int(m.group(2))
                    perm[a], perm[b] = perm[b], perm[a]
            z, norm2 = measure(circ, z_op)
            z_logical = np.empty_like(z)
            z_logical[perm] = z
            steps.append({"step": k, "z_logical": z_logical.tolist(), "norm2": norm2,
                          "layout_restored": perm == list(range(2 * N_SITES))})
            print(f"  {name} step {k:2d}: norm^2 {norm2:.6f}, {time.time() - t0:.0f} s",
                  flush=True)
        out[name] = steps
    out["n_f"] = [float((site_nf(m["z_logical"]) - site_nf(v["z_logical"])).sum())
                  for m, v in zip(out["meson"], out["SCV"])]
    for k, v in enumerate(out["n_f"], 1):
        print(f"n_f({k:2d}) = {v:+.5f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = f"su2_hadron_traj_chi{args.chi}{'_gpu' if args.gpu else ''}.json"
    (args.out_dir / name).write_text(json.dumps(out))


def trajectory_table(args):
    runs = {}
    for f in sorted(args.out_dir.glob("su2_hadron_traj_chi*.json")):
        d = json.loads(f.read_text())
        runs.setdefault(d["chi"], d)
    chis = sorted(runs)
    print("step," + ",".join(f"n_f_chi{c},norm2_meson_chi{c},norm2_vacuum_chi{c}" for c in chis)
          + ",TN,PP_CPU,PP_GPU,QPU")
    for k in range(1, 21):
        ours = ",".join(f"{runs[c]['n_f'][k - 1]:.5f},{runs[c]['meson'][k - 1]['norm2']:.6f},"
                        f"{runs[c]['SCV'][k - 1]['norm2']:.6f}" for c in chis)
        print(f"{k},{ours}," + ",".join(f"{x:.5f}" for x in PUBLISHED[k]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--chi", type=int, required=True)
    r.add_argument("--gpu", action="store_true")
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--cutoff", type=float, default=1e-10)
    tr = sub.add_parser("trajectory")
    tr.add_argument("--chi", type=int, required=True)
    tr.add_argument("--gpu", action="store_true")
    tr.add_argument("--threads", type=int, default=2)
    tr.add_argument("--cutoff", type=float, default=1e-10)
    for q in (r, sub.add_parser("table"), tr, sub.add_parser("trajectory-table")):
        q.add_argument("--qasm-dir", type=pathlib.Path, default=pathlib.Path("su2_hadron_qasm"))
        q.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("su2_hadron_runs"))
    args = p.parse_args()
    {"run": run, "table": table, "trajectory": trajectory,
     "trajectory-table": trajectory_table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
