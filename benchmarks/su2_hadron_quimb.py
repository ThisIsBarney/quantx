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

    python benchmarks/su2_hadron_quimb.py run --chi 800 --gpu      # one bond dimension
    python benchmarks/su2_hadron_quimb.py table                    # every saved run

Needs quimb (1.11), and CuPy for --gpu. The QASM files are downloaded from the tracker into
--qasm-dir the first time. --gpu keeps the MPS in complex128 on the GPU and splits each gate
by an eigendecomposition of the Gram matrix (quimb's method "eig"), which cuSOLVER does far
faster than an SVD; it agrees with the CPU's SVD to 3e-3 in n_f(20) at chi = 200. On the CPU,
--threads caps BLAS.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
import urllib.request

N_SITES = 60
URL = ("https://raw.githubusercontent.com/quantum-advantage-tracker/"
       "quantum-advantage-tracker.github.io/main/data/observable-estimations/circuit-models/"
       "su2_hadron_dynamics_lsh/x_100_{}.qasm")
CIRCUITS = ("meson", "SCV")


def qasm_path(qasm_dir: pathlib.Path, name: str) -> pathlib.Path:
    path = qasm_dir / f"x_100_{name}.qasm"
    if not path.exists():
        qasm_dir.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL.format(name)}", flush=True)
        urllib.request.urlretrieve(URL.format(name), path)
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


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--chi", type=int, required=True)
    r.add_argument("--gpu", action="store_true")
    r.add_argument("--threads", type=int, default=2)
    r.add_argument("--cutoff", type=float, default=1e-10)
    for q in (r, sub.add_parser("table")):
        q.add_argument("--qasm-dir", type=pathlib.Path, default=pathlib.Path("su2_hadron_qasm"))
        q.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("su2_hadron_runs"))
    args = p.parse_args()
    {"run": run, "table": table}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
