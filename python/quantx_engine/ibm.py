"""
IBM Quantum / Qiskit Runtime connector for the QuantX engine.

Provides IBMQuantumBackend, a thin adapter that:
1. Converts a QuantX native circuit dict to OpenQASM 3.0 source
2. Submits the circuit to IBM Quantum via qiskit-ibm-runtime
3. Converts the IBM job result back to a QuantX-compatible result dict

Requirements (optional dependencies):
    pip install qiskit>=1.0 qiskit-ibm-runtime>=0.20

If these packages are not installed, importing this module is safe; only
instantiation of IBMQuantumBackend will raise ImportError with instructions.

Usage:
    from quantx_engine.ibm import IBMQuantumBackend
    backend = IBMQuantumBackend(token="MY_IBM_TOKEN", instance="ibm-q/open/main")
    result = backend.run_circuit(circuit, shots=1000)
"""
from __future__ import annotations

import json
import os
from typing import Any

__all__ = ["IBMQuantumBackend", "IBMConnectorError", "circuit_to_qasm3"]


class IBMConnectorError(RuntimeError):
    """Raised for IBM connector-specific errors."""


# ── QuantX circuit dict → OpenQASM 3.0 source ────────────────────────────────

# Reverse gate map: QuantX gate name → QASM gate name
_QX_TO_QASM: dict[str, str] = {
    "H": "h", "X": "x", "Y": "y", "Z": "z",
    "S": "s", "T": "t", "Sdg": "sdg", "Tdg": "tdg",
    "CNOT": "cx", "CY": "cy", "CZ": "cz", "CCX": "ccx",
    "RX": "rx", "RY": "ry", "RZ": "rz",
    "P": "p", "U2": "u2", "U3": "u3",
    "SWAP": "swap", "ISWAP": "iswap",
    "I": "id", "SX": "sx", "SXdg": "sxdg",
    "MEASURE": "measure",
    "RESET": "reset",
}


def circuit_to_qasm3(circuit: dict[str, Any]) -> str:
    """Convert a QuantX native circuit dict to an OpenQASM 3.0 string.

    This is the reverse of parse_qasm_to_circuit() in qasm.py.

    Parameters
    ----------
    circuit:
        QuantX circuit dict with keys: qubits, clbits, operations, shots, seed.

    Returns
    -------
    str
        OpenQASM 3.0 source suitable for submission to IBM Quantum.

    Raises
    ------
    IBMConnectorError
        If the circuit contains gates not representable in QASM 3.0.
    """
    n_qubits = int(circuit.get("qubits", 0))
    n_clbits = int(circuit.get("clbits", 0))
    ops = circuit.get("operations", [])

    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', ""]
    if n_qubits > 0:
        lines.append(f"qubit[{n_qubits}] q;")
    if n_clbits > 0:
        lines.append(f"bit[{n_clbits}] c;")
    if n_qubits > 0 or n_clbits > 0:
        lines.append("")

    for op in ops:
        name = str(op.get("name", ""))
        targets = list(op.get("targets", []))
        controls = list(op.get("controls", []))
        params = list(op.get("params", []))

        qasm_name = _QX_TO_QASM.get(name)
        if qasm_name is None:
            raise IBMConnectorError(f"gate {name!r} has no QASM 3.0 equivalent")

        if name == "MEASURE":
            # measure q[t] -> c[c_bit];
            if not targets:
                continue
            q_idx = targets[0]
            c_idx = controls[0] if controls else q_idx
            lines.append(f"c[{c_idx}] = measure q[{q_idx}];")
            continue

        if name == "RESET":
            if targets:
                lines.append(f"reset q[{targets[0]}];")
            continue

        # Build gate line: gname(params) qubits;
        param_str = ""
        if params:
            formatted = ", ".join(f"{p:.17g}" for p in params)
            param_str = f"({formatted})"

        all_qubits = controls + targets
        qubit_str = ", ".join(f"q[{i}]" for i in all_qubits)
        lines.append(f"{qasm_name}{param_str} {qubit_str};")

    return "\n".join(lines) + "\n"


# ── IBM Quantum backend ───────────────────────────────────────────────────────

class IBMQuantumBackend:
    """Adapter that submits QuantX circuits to IBM Quantum via qiskit-ibm-runtime.

    Parameters
    ----------
    token:
        IBM Quantum API token. If not provided, read from the
        ``QISKIT_IBM_TOKEN`` environment variable.
    instance:
        IBM Quantum instance (hub/group/project), e.g. ``"ibm-q/open/main"``.
        Defaults to ``"ibm-q/open/main"``.
    backend_name:
        IBM backend name, e.g. ``"ibm_nairobi"`` or ``"ibmq_qasm_simulator"``.
        Defaults to ``"ibmq_qasm_simulator"``.
    """

    def __init__(
        self,
        token: str | None = None,
        instance: str = "ibm-q/open/main",
        backend_name: str = "ibmq_qasm_simulator",
    ):
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "IBM Quantum connector requires qiskit-ibm-runtime. "
                "Install it with: pip install qiskit>=1.0 qiskit-ibm-runtime>=0.20"
            ) from exc

        self._token = token or os.environ.get("QISKIT_IBM_TOKEN") or ""
        self._instance = instance
        self._backend_name = backend_name
        self._service: Any = None
        self._backend: Any = None

        if self._token:
            self._connect()

    def _connect(self) -> None:
        """Authenticate and select the target backend."""
        from qiskit_ibm_runtime import QiskitRuntimeService  # type: ignore[import]
        try:
            self._service = QiskitRuntimeService(
                channel="ibm_quantum",
                token=self._token,
                instance=self._instance,
            )
            self._backend = self._service.backend(self._backend_name)
        except Exception as exc:
            raise IBMConnectorError(f"IBM Quantum connection failed: {exc}") from exc

    def run_circuit(
        self,
        circuit: dict[str, Any],
        shots: int | None = None,
        optimization_level: int = 1,
    ) -> dict[str, Any]:
        """Submit a QuantX circuit dict to IBM Quantum and return a result dict.

        Parameters
        ----------
        circuit:
            QuantX native circuit dict (from Engine.parse_qasm or run_circuit).
        shots:
            Number of shots. Overrides circuit["shots"] if provided.
        optimization_level:
            Qiskit transpiler optimization level (0–3, default 1).

        Returns
        -------
        dict with keys:
            ok (bool), shots (int), counts (dict[str, int]),
            probabilities (dict[str, float]), backend (str),
            ibm_job_id (str), raw_ibm_result (dict)
        """
        try:
            from qiskit import QuantumCircuit  # type: ignore[import]
            from qiskit.qasm3 import loads as qasm3_loads  # type: ignore[import]
            from qiskit_ibm_runtime import SamplerV2 as Sampler  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "IBM Quantum connector requires qiskit>=1.0 and qiskit-ibm-runtime>=0.20. "
                "Install with: pip install qiskit>=1.0 qiskit-ibm-runtime>=0.20"
            ) from exc

        if self._backend is None:
            raise IBMConnectorError(
                "Backend not connected. Provide token= or set QISKIT_IBM_TOKEN env var."
            )

        n_shots = shots if shots is not None else int(circuit.get("shots", 1000))
        if n_shots <= 0:
            n_shots = 1000

        # Convert QuantX circuit → QASM 3.0 → Qiskit QuantumCircuit
        qasm_src = circuit_to_qasm3(circuit)
        try:
            qc = qasm3_loads(qasm_src)
        except Exception as exc:
            raise IBMConnectorError(f"QASM 3.0 parse failed in Qiskit: {exc}") from exc

        # Transpile and run via SamplerV2
        try:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager  # type: ignore[import]
            pm = generate_preset_pass_manager(
                optimization_level=optimization_level,
                backend=self._backend,
            )
            isa_circuit = pm.run(qc)
            sampler = Sampler(mode=self._backend)
            job = sampler.run([isa_circuit], shots=n_shots)
            # Capture job_id before blocking on result() so it's always available
            job_id: str = getattr(job, "job_id", None) or ""
            if callable(job_id):  # some versions return a method
                try:
                    job_id = job.job_id()
                except Exception:
                    job_id = ""
            ibm_result = job.result()
        except Exception as exc:
            raise IBMConnectorError(f"IBM Quantum job failed: {exc}") from exc

        return self._convert_result(ibm_result, n_shots, job_id=job_id)

    def _convert_result(
        self, ibm_result: Any, shots: int, job_id: str = ""
    ) -> dict[str, Any]:
        """Convert a Qiskit SamplerV2 result to a QuantX-compatible dict."""
        try:
            pub_result = ibm_result[0]
            data = pub_result.data
            # Get counts from the first classical register
            register_names = [k for k in dir(data) if not k.startswith("_")]
            counts_raw: dict[str, int] = {}
            for reg_name in register_names:
                bit_array = getattr(data, reg_name, None)
                if bit_array is not None and hasattr(bit_array, "get_counts"):
                    counts_raw = bit_array.get_counts()
                    break

            total = sum(counts_raw.values()) or shots
            probabilities = {k: v / total for k, v in counts_raw.items()}

            # Build a machine-readable metadata dict from the IBM result object
            # rather than stringifying it, so callers can extract fields reliably.
            raw_metadata: dict[str, Any] = {}
            for attr in ("metadata", "header", "status", "date", "backend_name",
                         "backend_version", "qobj_id", "job_id", "time_taken"):
                val = getattr(ibm_result, attr, None)
                if val is not None:
                    try:
                        # Only include JSON-serialisable scalar/dict values
                        import json as _json
                        _json.dumps(val)
                        raw_metadata[attr] = val
                    except (TypeError, ValueError):
                        raw_metadata[attr] = str(val)

            return {
                "ok": True,
                "shots": shots,
                "counts": counts_raw,
                "probabilities": probabilities,
                "backend": self._backend_name,
                "ibm_job_id": job_id,
                "raw_ibm_result": raw_metadata,
            }
        except Exception as exc:
            raise IBMConnectorError(f"failed to parse IBM result: {exc}") from exc

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def instance(self) -> str:
        return self._instance
