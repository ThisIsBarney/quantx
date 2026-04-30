"""Tests for M98 REST API routes (network, protocols, pulse, advanced)."""
from __future__ import annotations

import math
import sys
import os

import pytest

# Ensure the python SDK is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# /network
# ─────────────────────────────────────────────────────────────────────────────

class TestNetworkRepeaterChain:
    def test_direct_link(self):
        r = client.post("/network/repeater-chain", json={
            "n_hops": 0,
            "segment_length_km": 10.0,
        })
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert 0.5 < d["end_to_end_fidelity"] <= 1.0
        assert d["n_hops"] == 0
        assert isinstance(d["secret_key_rate"], float)

    def test_one_hop_repeater(self):
        r = client.post("/network/repeater-chain", json={
            "n_hops": 1,
            "segment_length_km": 5.0,
            "bsm_fidelity": 0.99,
        })
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["n_hops"] == 1
        assert len(d["hop_fidelities"]) == 2

    def test_negative_hops_returns_400(self):
        r = client.post("/network/repeater-chain", json={"n_hops": -1, "segment_length_km": 10})
        assert r.status_code == 422  # pydantic validation error

    def test_purification_rounds(self):
        r = client.post("/network/repeater-chain", json={
            "n_hops": 0,
            "segment_length_km": 50.0,
            "purification_rounds": 1,
        })
        assert r.status_code == 200

    def test_total_length_km(self):
        r = client.post("/network/repeater-chain", json={
            "n_hops": 2,
            "segment_length_km": 20.0,
        })
        d = r.json()
        assert d["total_length_km"] == pytest.approx(60.0)


class TestNetworkFidelityEndpoints:
    def test_dephasing(self):
        r = client.post("/network/fidelity/dephasing", json={"F": 0.9, "p_deph": 0.1})
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["F_out"] == pytest.approx(0.9 * 0.9 + 0.1 * 0.25, abs=1e-9)

    def test_dephasing_identity(self):
        r = client.post("/network/fidelity/dephasing", json={"F": 0.8, "p_deph": 0.0})
        assert r.json()["F_out"] == pytest.approx(0.8)

    def test_dephasing_full_depolarize(self):
        r = client.post("/network/fidelity/dephasing", json={"F": 0.9, "p_deph": 1.0})
        assert r.json()["F_out"] == pytest.approx(0.25)

    def test_swapping_perfect_pair(self):
        r = client.post("/network/fidelity/swapping", json={"F_A": 1.0, "F_B": 0.8})
        assert r.json()["F_out"] == pytest.approx(0.8)

    def test_swapping_both_perfect(self):
        r = client.post("/network/fidelity/swapping", json={"F_A": 1.0, "F_B": 1.0})
        assert r.json()["F_out"] == pytest.approx(1.0)

    def test_purification_improves_fidelity(self):
        r0 = client.post("/network/fidelity/purification", json={"F": 0.8, "n_rounds": 0})
        r1 = client.post("/network/fidelity/purification", json={"F": 0.8, "n_rounds": 1})
        assert r1.json()["F_out"] > r0.json()["F_out"]

    def test_invalid_F_returns_4xx(self):
        r = client.post("/network/fidelity/dephasing", json={"F": 1.5, "p_deph": 0.1})
        assert r.status_code in (400, 422)


# ─────────────────────────────────────────────────────────────────────────────
# /protocols
# ─────────────────────────────────────────────────────────────────────────────

class TestBB84Route:
    def test_basic(self):
        r = client.post("/protocols/bb84", json={"n_bits": 500, "seed": 0})
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert 0 <= d["qber"] <= 0.5
        assert isinstance(d["eve_detected"], bool)

    def test_no_eve_no_detection(self):
        r = client.post("/protocols/bb84", json={
            "n_bits": 2000, "channel_error_rate": 0.01,
            "eve_intercept_fraction": 0.0, "seed": 1
        })
        d = r.json()
        assert d["eve_detected"] is False

    def test_full_eve_detected(self):
        r = client.post("/protocols/bb84", json={
            "n_bits": 3000, "channel_error_rate": 0.0,
            "eve_intercept_fraction": 1.0, "seed": 7
        })
        d = r.json()
        assert d["eve_detected"] is True

    def test_secure_bits_positive_without_eve(self):
        r = client.post("/protocols/bb84", json={
            "n_bits": 2000, "channel_error_rate": 0.02,
            "eve_intercept_fraction": 0.0, "seed": 5
        })
        assert r.json()["n_secure_bits"] > 0

    def test_invalid_n_bits(self):
        r = client.post("/protocols/bb84", json={"n_bits": 0})
        assert r.status_code == 422


class TestE91Route:
    def test_basic(self):
        r = client.post("/protocols/e91", json={"n_pairs": 2000, "seed": 0})
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert "chsh_s" in d
        assert isinstance(d["bell_violation"], bool)

    def test_high_fidelity_violation(self):
        r = client.post("/protocols/e91", json={
            "n_pairs": 5000, "bell_pair_fidelity": 0.95,
            "channel_error_rate": 0.0, "seed": 1
        })
        d = r.json()
        assert d["bell_violation"] is True
        assert d["chsh_s"] > 2.0

    def test_low_fidelity_no_violation(self):
        # F=0.4 → Werner p=(4*0.4-1)/3≈0.2 → S≈0.57 < 2 → no Bell violation.
        r = client.post("/protocols/e91", json={
            "n_pairs": 5000, "bell_pair_fidelity": 0.4,
            "channel_error_rate": 0.0, "seed": 2
        })
        assert r.status_code == 200
        d = r.json()
        # S ≈ 2√2 * 0.2 ≈ 0.57 → well below the 2.0 threshold
        assert not d["bell_violation"]


class TestTeleportationRoute:
    def test_basic(self):
        r = client.post("/protocols/teleportation", json={
            "n_states": 200, "channel_fidelity": 0.9, "seed": 0
        })
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert 0 <= d["average_fidelity"] <= 1.0

    def test_high_fidelity_above_classical(self):
        r = client.post("/protocols/teleportation", json={
            "n_states": 500, "channel_fidelity": 0.95,
            "bsm_fidelity": 1.0, "seed": 1
        })
        assert r.json()["average_fidelity"] > 2.0 / 3.0

    def test_classical_bits_sent(self):
        r = client.post("/protocols/teleportation", json={"n_states": 50, "seed": 0})
        assert r.json()["classical_bits_sent"] == 100


class TestDenseCodingRoute:
    def test_basic(self):
        r = client.post("/protocols/dense-coding", json={
            "n_messages": 300, "channel_fidelity": 0.95, "seed": 0
        })
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert 0 <= d["channel_capacity"] <= 2.0

    def test_perfect_channel_capacity(self):
        r = client.post("/protocols/dense-coding", json={
            "n_messages": 1000, "channel_fidelity": 1.0, "seed": 0
        })
        d = r.json()
        assert d["channel_capacity"] == pytest.approx(2.0, abs=0.05)

    def test_bits_encoded(self):
        r = client.post("/protocols/dense-coding", json={"n_messages": 200, "seed": 0})
        d = r.json()
        assert d["n_classical_bits_encoded"] == 400
        assert d["n_bits_sent_quantum"] == 200


class TestProtocolUtils:
    def test_key_rate_zero_qber(self):
        r = client.post("/protocols/utils/key-rate", json={"qber": 0.0})
        assert r.status_code == 200
        assert r.json()["secure_key_rate"] == pytest.approx(1.0)

    def test_key_rate_high_qber_zero(self):
        r = client.post("/protocols/utils/key-rate", json={"qber": 0.15})
        assert r.json()["secure_key_rate"] == pytest.approx(0.0)

    def test_teleportation_fidelity_perfect(self):
        r = client.post("/protocols/utils/teleportation-fidelity", json={"channel_fidelity": 1.0})
        d = r.json()
        assert d["average_teleportation_fidelity"] == pytest.approx(1.0)
        assert d["beats_classical"] is True

    def test_teleportation_fidelity_classical_threshold(self):
        r = client.post("/protocols/utils/teleportation-fidelity", json={"channel_fidelity": 0.5})
        d = r.json()
        assert d["average_teleportation_fidelity"] == pytest.approx(2.0 / 3.0, abs=1e-9)
        assert d["beats_classical"] is False


# ─────────────────────────────────────────────────────────────────────────────
# /pulse
# ─────────────────────────────────────────────────────────────────────────────

class TestPulseRoutes:
    def test_pi_pulse_area(self):
        r = client.post("/pulse/pi-pulse", json={"sigma": 25e-9})
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["pi_pulse_area_rad"] > 0

    def test_simulate_no_pulses(self):
        """Empty schedule → state stays |0⟩, Bloch vector ≈ (0,0,1)."""
        r = client.post("/pulse/simulate", json={
            "n_qubits": 1,
            "qubit_frequencies": [0.0],
            "t_total": 100e-9,
            "waveforms": [],
        })
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert len(d["bloch_vectors"]) == 1

    def test_simulate_gaussian_pulse(self):
        r = client.post("/pulse/simulate", json={
            "n_qubits": 1,
            "qubit_frequencies": [0.0],
            "t_total": 200e-9,
            "waveforms": [{
                "waveform_type": "gaussian",
                "t_start": 0,
                "duration": 100e-9,
                "amp": 1e7,
                "sigma": 25e-9,
                "channel_index": 0,
            }],
        })
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert len(d["bloch_vectors"]) == 1

    def test_simulate_constant_pulse(self):
        r = client.post("/pulse/simulate", json={
            "n_qubits": 1,
            "qubit_frequencies": [0.0],
            "t_total": 100e-9,
            "waveforms": [{
                "waveform_type": "constant",
                "t_start": 0,
                "duration": 50e-9,
                "amp": 1e7,
                "channel_index": 0,
            }],
        })
        assert r.status_code == 200

    def test_simulate_invalid_waveform_type(self):
        r = client.post("/pulse/simulate", json={
            "n_qubits": 1,
            "qubit_frequencies": [0.0],
            "t_total": 100e-9,
            "waveforms": [{
                "waveform_type": "magic",
                "t_start": 0,
                "duration": 50e-9,
                "amp": 1.0,
            }],
        })
        assert r.status_code in (400, 422, 500)

    def test_simulate_wrong_freq_length(self):
        r = client.post("/pulse/simulate", json={
            "n_qubits": 2,
            "qubit_frequencies": [0.0],  # wrong length
            "t_total": 100e-9,
        })
        assert r.status_code in (400, 422)


# ─────────────────────────────────────────────────────────────────────────────
# /advanced
# ─────────────────────────────────────────────────────────────────────────────

class TestFloquetRoute:
    def test_basic(self):
        r = client.post("/advanced/floquet/driven-two-level", json={
            "omega_0": 1.0,
            "omega_drive": 1.0,
            "drive_amplitude": 0.1,
            "n_steps": 50,
        })
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert len(d["quasi_energies"]) == 2
        assert "gap" in d

    def test_quasi_energies_symmetric(self):
        r = client.post("/advanced/floquet/driven-two-level", json={
            "omega_0": 2.0,
            "omega_drive": 2.0,
            "drive_amplitude": 0.05,
            "n_steps": 100,
        })
        qe = r.json()["quasi_energies"]
        # Quasi-energies come in ±ε pairs
        assert abs(qe[0] + qe[1]) < 0.5


class TestTDMRGRoute:
    def test_heisenberg(self):
        r = client.post("/advanced/tdmrg/evolve", json={
            "n_sites": 4,
            "model": "heisenberg",
            "J": 1.0,
            "t_total": 0.5,
            "dt": 0.1,
            "chi": 8,
        })
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["n_sites"] == 4
        assert len(d["entanglement_entropy_final"]) == 3  # n_sites - 1

    def test_ising(self):
        r = client.post("/advanced/tdmrg/evolve", json={
            "n_sites": 4,
            "model": "ising",
            "J": 1.0,
            "h": 0.5,
            "t_total": 0.3,
            "dt": 0.05,
            "chi": 8,
        })
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_invalid_model(self):
        r = client.post("/advanced/tdmrg/evolve", json={
            "n_sites": 4, "model": "xy", "J": 1.0, "t_total": 0.5, "dt": 0.1
        })
        assert r.status_code == 400


class TestNoiseAwareVQERoute:
    def _zz_hamiltonian(self, n=2):
        """−ZZ Hamiltonian matrix for n=2 qubits."""
        import numpy as np
        H = np.array([[-1, 0, 0, 0], [0, 1, 0, 0],
                      [0, 0, 1, 0], [0, 0, 0, -1]], dtype=float)
        return H.tolist()

    def test_basic(self):
        r = client.post("/advanced/vqe/noise-aware", json={
            "n_qubits": 2,
            "n_layers": 1,
            "hamiltonian_real": self._zz_hamiltonian(),
            "max_iter": 10,
            "seed": 42,
        })
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert isinstance(d["energy"], float)
        assert isinstance(d["converged"], bool)

    def test_wrong_matrix_size(self):
        r = client.post("/advanced/vqe/noise-aware", json={
            "n_qubits": 2,
            "n_layers": 1,
            "hamiltonian_real": [[1.0, 0.0], [0.0, -1.0]],  # 2×2 wrong for 2 qubits
            "max_iter": 5,
        })
        assert r.status_code in (400, 422)


class TestFaultTolerantRoutes:
    def test_logical_qubit_error_rate(self):
        r = client.post("/advanced/fault-tolerant/logical", json={
            "distance": 3,
            "p_physical": 0.01,
            "n_shots": 100,
            "seed": 0,
        })
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert 0.0 <= d["logical_error_rate"] <= 1.0
        assert d["code_distance"] == 3

    def test_below_threshold_low_error(self):
        """Well below threshold → logical error rate should be very small."""
        r = client.post("/advanced/fault-tolerant/logical", json={
            "distance": 5,
            "p_physical": 0.001,
            "n_shots": 200,
            "seed": 1,
        })
        assert r.json()["logical_error_rate"] < 0.1

    def test_magic_state_basic(self):
        r = client.post("/advanced/fault-tolerant/magic-state", json={
            "p_physical": 0.01,
            "n_attempts": 200,
            "seed": 0,
        })
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        # Distillation must reduce measured error below the theoretical cubic threshold
        assert d["p_output_measured"] <= d["p_input"]
        # Theoretical output p ≈ 35 * p^3 << p_in → theoretical suppression > 1
        assert d["p_output_theoretical"] < d["p_input"]

    def test_magic_state_theoretical_cubic(self):
        """Theoretical p_out ≈ 35 p_in³."""
        p = 0.01
        r = client.post("/advanced/fault-tolerant/magic-state", json={
            "p_physical": p, "n_attempts": 100, "seed": 0
        })
        d = r.json()
        expected = 35 * p ** 3
        assert abs(d["p_output_theoretical"] - expected) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# OpenAPI schema coverage
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenAPISchema:
    def test_new_routes_in_schema(self):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"]
        # All four router prefixes must appear
        assert any(p.startswith("/network") for p in paths)
        assert any(p.startswith("/protocols") for p in paths)
        assert any(p.startswith("/pulse") for p in paths)
        assert any(p.startswith("/advanced") for p in paths)

    def test_legacy_routes_still_present(self):
        r = client.get("/openapi.json")
        paths = r.json()["paths"]
        assert "/health" in paths
        assert "/circuit" in paths
