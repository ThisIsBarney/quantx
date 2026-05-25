"""M169 — QuantX real hardware round-trip demo (IBM Quantum).

Demonstrates the end-to-end workflow for running a QuantX circuit on real
IBM Quantum hardware (or the QASM simulator fallback):

    1. Build a 2-qubit Bell state circuit in QuantX circuit-dict format
    2. Transpile the circuit for the target backend via QASM 3.0
    3. Submit the job to IBM Quantum using :class:`IBMHardwareBackend`
    4. Poll until completion (non-blocking via :class:`HardwareJob`)
    5. Display bitstring counts and compare with the ideal distribution

Requirements
------------
* ``QISKIT_IBM_TOKEN`` environment variable set to a valid IBM Quantum API
  token.  Obtain one at https://quantum.ibm.com/
* ``qiskit>=1.0`` and ``qiskit-ibm-runtime>=0.20`` installed::

      pip install qiskit qiskit-ibm-runtime

  These are **optional** dependencies; the demo exits gracefully when they
  are absent.

Usage
-----
::

    # Run on the default QASM simulator (no real hardware queue)
    python examples/ibm_hardware_demo.py

    # Run on a real device (requires queuing; may take minutes)
    python examples/ibm_hardware_demo.py --backend ibm_sherbrooke --shots 4096

    # Dry-run: build and transpile only (no network request)
    python examples/ibm_hardware_demo.py --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Make the quantx_engine package importable when run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


# ---------------------------------------------------------------------------
# Circuit definition
# ---------------------------------------------------------------------------

def bell_circuit(n_shots: int = 1024) -> dict:
    """Return a QuantX circuit dict that prepares a 2-qubit Bell state |Φ⁺⟩."""
    return {
        "qubits": 2,
        "clbits": 2,
        "shots": n_shots,
        "seed": 42,
        "operations": [
            {"name": "H",     "targets": [0]},
            {"name": "CNOT",  "controls": [0], "targets": [1]},
            {"name": "MEASURE", "targets": [0], "controls": [0]},
            {"name": "MEASURE", "targets": [1], "controls": [1]},
        ],
    }


# ---------------------------------------------------------------------------
# Result display helpers
# ---------------------------------------------------------------------------

def _bar(frac: float, width: int = 30) -> str:
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


def display_counts(counts: dict[str, int], shots: int, title: str = "Results") -> None:
    """Print a simple ASCII-art histogram of bitstring counts."""
    total = sum(counts.values()) or shots
    print(f"\n{title} ({total} shots)")
    print("-" * 50)
    for bitstring, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        frac = count / total
        print(f"  |{bitstring}⟩  {_bar(frac)}  {count:5d}  ({100*frac:5.1f}%)")
    print()

    # Ideal Bell state: |00⟩ and |11⟩ each at 50%
    expected = {"00": 0.5, "11": 0.5}
    print("Ideal Bell state fractions: |00⟩ = 50%, |11⟩ = 50%")
    bell_frac = (counts.get("00", 0) + counts.get("11", 0)) / total
    print(f"  → P(|00⟩ or |11⟩) = {100*bell_frac:.1f}%  "
          f"{'✅ within 5%' if abs(bell_frac - 1.0) < 0.05 else '⚠ check noise'}")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run_demo(
    backend_name: str = "ibmq_qasm_simulator",
    shots: int = 1024,
    dry_run: bool = False,
    poll_interval: float = 5.0,
    max_wait: float = 600.0,
) -> int:
    """Execute the IBM hardware demo.

    Returns
    -------
    int
        0 on success, non-zero on error.
    """
    from quantx_engine.ibm import circuit_to_qasm3

    print("=" * 60)
    print("QuantX IBM Quantum Hardware Round-Trip Demo (M169)")
    print("=" * 60)

    # 1. Build circuit
    print("\n[1/5] Building Bell-state circuit …")
    circuit = bell_circuit(n_shots=shots)
    qasm_src = circuit_to_qasm3(circuit)
    print(f"      Circuit: {circuit['qubits']} qubits, "
          f"{len(circuit['operations'])} operations")
    print(f"      QASM 3.0 snippet: {qasm_src.splitlines()[2]!r}")

    if dry_run:
        print("\n[DRY RUN] QASM source:")
        print(qasm_src)
        print("\nDry-run complete — no job submitted.")
        return 0

    # 2. Connect to IBM backend
    print(f"\n[2/5] Connecting to IBM backend '{backend_name}' …")
    token = os.environ.get("QISKIT_IBM_TOKEN")
    if not token:
        print("ERROR: QISKIT_IBM_TOKEN environment variable not set.")
        print("       Set it to your IBM Quantum API token:")
        print("       export QISKIT_IBM_TOKEN=<your_token>")
        return 1

    try:
        from quantx_engine.hardware import IBMHardwareBackend, HardwareError
    except ImportError as exc:
        print(f"ERROR: Could not import QuantX hardware module: {exc}")
        return 1

    try:
        backend = IBMHardwareBackend(
            token=token,
            instance="ibm-q/open/main",
            backend_name=backend_name,
        )
        print(f"      Connected ✅")
    except Exception as exc:
        print(f"ERROR: Could not connect to IBM backend: {exc}")
        return 1

    # 3. Submit circuit (non-blocking)
    print(f"\n[3/5] Submitting circuit ({shots} shots) …")
    try:
        job = backend.submit(circuit, shots=shots, optimization_level=1)
        print(f"      Job ID: {job.job_id!r}")
        print(f"      Status: {job.status()}")
    except Exception as exc:
        print(f"ERROR: Submission failed: {exc}")
        return 1

    # 4. Poll for completion
    print(f"\n[4/5] Waiting for job to complete "
          f"(polling every {poll_interval:.0f}s, max {max_wait:.0f}s) …")
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        status = job.status()
        print(f"      [{elapsed:5.0f}s]  status = {status}")
        if status == "completed":
            break
        if status == "failed":
            print("ERROR: IBM job failed.")
            return 1
        if elapsed > max_wait:
            print(f"ERROR: Timed out after {max_wait:.0f}s.")
            return 1
        time.sleep(poll_interval)

    # 5. Fetch and display results
    print(f"\n[5/5] Fetching results …")
    try:
        result = job.fetch()
        display_counts(result.counts, shots=result.shots, title="IBM Quantum Results")
    except Exception as exc:
        print(f"ERROR: Could not fetch results: {exc}")
        return 1

    print("Round-trip complete ✅")
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QuantX IBM Quantum real hardware round-trip demo"
    )
    parser.add_argument(
        "--backend",
        default="ibmq_qasm_simulator",
        help="IBM backend name (default: ibmq_qasm_simulator)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of measurement shots (default: 1024)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print the QASM circuit without submitting",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        metavar="SECS",
        help="Polling interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=600.0,
        metavar="SECS",
        help="Maximum wait time for job completion (default: 600)",
    )
    args = parser.parse_args()

    try:
        sys.exit(run_demo(
            backend_name=args.backend,
            shots=args.shots,
            dry_run=args.dry_run,
            poll_interval=args.poll_interval,
            max_wait=args.max_wait,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
