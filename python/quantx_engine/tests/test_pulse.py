"""Tests for pulse-level simulator — M89."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quantx_engine.contrib.pulse import (
    ArbitraryWaveform,
    ConstantWaveform,
    ControlChannel,
    DRAGWaveform,
    DriveChannel,
    FlatTopWaveform,
    GaussianWaveform,
    MeasureChannel,
    PulseError,
    PulseResult,
    PulseSchedule,
    PulseSimulator,
    QubitSystem,
    pi_pulse_area,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _single_qubit(t1=None, t2=None, detuning=0.0):
    """QubitSystem with 1 qubit; frequencies=1.0, drive on resonance by default."""
    freq = 1.0
    drive = freq - detuning
    return QubitSystem(
        n_qubits=1,
        frequencies=[freq],
        drive_frequencies=[drive],
        t1_times=[t1] if t1 is not None else None,
        t2_times=[t2] if t2 is not None else None,
    )


def _pi_const(T: float) -> ConstantWaveform:
    """ConstantWaveform whose area = π  (π-pulse condition ∫ Ω dt = π)."""
    return ConstantWaveform(duration=T, amp=math.pi / T)


def _pi_gaussian(duration: float, sigma: float) -> GaussianWaveform:
    """GaussianWaveform whose area ≈ π."""
    return GaussianWaveform(duration=duration, amp=pi_pulse_area(sigma), sigma=sigma)


_SIM = PulseSimulator(atol=1e-10, rtol=1e-8)


# ── Channel construction ──────────────────────────────────────────────────────

class TestChannels:
    def test_drive_channel_repr(self):
        assert "0" in repr(DriveChannel(0))

    def test_measure_channel_repr(self):
        assert "MeasureChannel" in repr(MeasureChannel(1))

    def test_control_channel_repr(self):
        assert "ControlChannel" in repr(ControlChannel(2))

    def test_channels_frozen(self):
        ch = DriveChannel(0)
        with pytest.raises((AttributeError, TypeError)):
            ch.qubit = 5  # type: ignore[misc]

    def test_channels_hashable(self):
        d = {DriveChannel(0): "d0", MeasureChannel(0): "m0"}
        assert len(d) == 2


# ── Waveform construction & envelope ─────────────────────────────────────────

class TestWaveforms:
    def test_constant_envelope_inside(self):
        wf = ConstantWaveform(duration=10.0, amp=0.5)
        t = np.array([0.0, 5.0, 10.0])
        assert np.allclose(wf.envelope(t), 0.5)

    def test_constant_envelope_outside_zero(self):
        wf = ConstantWaveform(duration=10.0, amp=1.0)
        assert wf.envelope(np.array([-1.0]))[0] == 0.0
        assert wf.envelope(np.array([11.0]))[0] == 0.0

    def test_constant_duration_zero_raises(self):
        with pytest.raises(PulseError):
            ConstantWaveform(duration=0.0, amp=1.0)

    def test_gaussian_peak_at_center(self):
        wf = GaussianWaveform(duration=20.0, amp=1.0, sigma=3.0)
        t = np.linspace(0, 20, 1000)
        env = wf.envelope(t)
        assert t[np.argmax(env)] == pytest.approx(10.0, abs=0.1)

    def test_gaussian_peak_value(self):
        wf = GaussianWaveform(duration=20.0, amp=2.0, sigma=3.0)
        assert wf.envelope(np.array([10.0]))[0] == pytest.approx(2.0, abs=1e-6)

    def test_gaussian_area(self):
        sigma = 5.0
        amp = pi_pulse_area(sigma)
        wf = GaussianWaveform(duration=60.0, amp=amp, sigma=sigma)
        t = np.linspace(0, 60, 10000)
        area = np.trapezoid(wf.envelope(t), t)
        assert area == pytest.approx(math.pi, abs=1e-3)

    def test_gaussian_sigma_zero_raises(self):
        with pytest.raises(PulseError):
            GaussianWaveform(duration=10.0, amp=1.0, sigma=0.0)

    def test_drag_real_part_is_gaussian(self):
        wf = DRAGWaveform(duration=20.0, amp=1.0, sigma=3.0, beta=0.0)
        t = np.linspace(0, 20, 100)
        env = wf.envelope(t)
        g = GaussianWaveform(duration=20.0, amp=1.0, sigma=3.0)
        assert np.allclose(np.real(env), g.envelope(t))

    def test_drag_imaginary_part_nonzero_for_nonzero_beta(self):
        wf = DRAGWaveform(duration=20.0, amp=1.0, sigma=3.0, beta=1.0)
        t = np.linspace(0.1, 19.9, 100)
        env = wf.envelope(t)
        assert np.any(np.abs(np.imag(env)) > 1e-6)

    def test_flat_top_peak(self):
        wf = FlatTopWaveform(duration=20.0, amp=1.0, rise_time=2.0)
        mid_env = wf.envelope(np.array([10.0]))[0]
        assert mid_env == pytest.approx(1.0, abs=0.01)

    def test_flat_top_edges_below_plateau(self):
        """Edge value at t=0 is well below the plateau value at t=duration/2."""
        wf = FlatTopWaveform(duration=20.0, amp=1.0, rise_time=1.0)
        edge = float(abs(wf.envelope(np.array([0.0]))[0]))
        plateau = float(abs(wf.envelope(np.array([10.0]))[0]))
        assert edge < 0.7 * plateau   # edge significantly below plateau

    def test_arbitrary_waveform_duration(self):
        samples = np.ones(100, dtype=complex)
        wf = ArbitraryWaveform(samples=samples, dt=0.5)
        assert wf.duration == pytest.approx(50.0)

    def test_arbitrary_waveform_interpolation(self):
        samples = np.array([1.0, 2.0, 3.0], dtype=complex)
        wf = ArbitraryWaveform(samples=samples, dt=1.0)
        assert wf.envelope(np.array([0.0]))[0] == pytest.approx(1.0)
        assert wf.envelope(np.array([1.0]))[0] == pytest.approx(2.0)

    def test_arbitrary_waveform_empty_raises(self):
        with pytest.raises(PulseError):
            ArbitraryWaveform(samples=np.array([]))


# ── PulseSchedule ─────────────────────────────────────────────────────────────

class TestPulseSchedule:
    def test_empty_schedule_duration_zero(self):
        s = PulseSchedule()
        assert s.duration == 0.0

    def test_duration_single_instruction(self):
        s = PulseSchedule().add(DriveChannel(0), ConstantWaveform(10.0, 1.0))
        assert s.duration == pytest.approx(10.0)

    def test_duration_with_offset(self):
        s = PulseSchedule().add(DriveChannel(0), ConstantWaveform(10.0, 1.0), t_start=5.0)
        assert s.duration == pytest.approx(15.0)

    def test_duration_parallel_pulses(self):
        s = (
            PulseSchedule()
            .add(DriveChannel(0), ConstantWaveform(10.0, 1.0), t_start=0.0)
            .add(DriveChannel(1), ConstantWaveform(5.0, 1.0), t_start=8.0)
        )
        assert s.duration == pytest.approx(13.0)

    def test_negative_t_start_raises(self):
        with pytest.raises(PulseError):
            PulseSchedule().add(DriveChannel(0), ConstantWaveform(1.0, 1.0), t_start=-1.0)

    def test_instructions_list(self):
        s = PulseSchedule().add(DriveChannel(0), ConstantWaveform(1.0, 1.0))
        assert len(s.instructions) == 1

    def test_fluent_chaining_returns_schedule(self):
        s = PulseSchedule()
        result = s.add(DriveChannel(0), ConstantWaveform(1.0, 1.0))
        assert result is s

    def test_repr_contains_name(self):
        s = PulseSchedule(name="test_sched")
        assert "test_sched" in repr(s)


# ── QubitSystem ───────────────────────────────────────────────────────────────

class TestQubitSystem:
    def test_default_drive_equals_frequencies(self):
        sys = QubitSystem(1, [5.0])
        assert sys.drive_frequencies == [5.0]

    def test_detuning_zero_on_resonance(self):
        sys = QubitSystem(1, [5.0], drive_frequencies=[5.0])
        assert sys.detunings == [0.0]

    def test_detuning_nonzero(self):
        sys = QubitSystem(1, [5.0], drive_frequencies=[4.8])
        assert sys.detunings[0] == pytest.approx(0.2)

    def test_invalid_n_qubits(self):
        with pytest.raises(PulseError):
            QubitSystem(0, [])

    def test_frequencies_length_mismatch(self):
        with pytest.raises(PulseError):
            QubitSystem(2, [5.0])

    def test_two_qubit_system(self):
        sys = QubitSystem(2, [5.0, 5.1], t1_times=[50.0, 50.0])
        assert sys.n_qubits == 2


# ── PulseSimulator — construction errors ─────────────────────────────────────

class TestSimulatorErrors:
    def test_wrong_schedule_type(self):
        with pytest.raises(PulseError):
            _SIM.run({"not": "a schedule"}, _single_qubit())  # type: ignore

    def test_wrong_system_type(self):
        s = PulseSchedule().add(DriveChannel(0), ConstantWaveform(1.0, 1.0))
        with pytest.raises(PulseError):
            _SIM.run(s, {"not": "a system"})  # type: ignore

    def test_zero_duration_schedule(self):
        s = PulseSchedule()   # no instructions → duration = 0
        with pytest.raises(PulseError):
            _SIM.run(s, _single_qubit())

    def test_wrong_initial_state_shape(self):
        s = PulseSchedule().add(DriveChannel(0), _pi_const(math.pi))
        with pytest.raises(PulseError):
            _SIM.run(s, _single_qubit(), initial_state=np.array([1.0, 0.0, 0.0]))


# ── Rabi oscillations ─────────────────────────────────────────────────────────

class TestRabiOscillations:
    """Constant drive on resonance: P₁(t) = sin²(Ω₀ t / 2)."""

    def test_pi_pulse_const_produces_excited_state(self):
        """∫ Ω dt = π → |0⟩ → |1⟩."""
        T = 10.0
        s = PulseSchedule().add(DriveChannel(0), _pi_const(T))
        r = _SIM.run(s, _single_qubit())
        assert r.populations.get("1", 0.0) == pytest.approx(1.0, abs=1e-4)

    def test_two_pi_pulse_returns_ground_state(self):
        """Two consecutive π-pulses → |0⟩."""
        T = 10.0
        s = (
            PulseSchedule()
            .add(DriveChannel(0), _pi_const(T), t_start=0.0)
            .add(DriveChannel(0), _pi_const(T), t_start=T)
        )
        r = _SIM.run(s, _single_qubit())
        assert r.populations.get("0", 0.0) == pytest.approx(1.0, abs=1e-4)

    def test_pi_over_2_pulse_equal_superposition(self):
        """∫ Ω dt = π/2 → equal superposition."""
        T = 5.0
        wf = ConstantWaveform(duration=T, amp=math.pi / (2 * T))   # area = π/2
        s = PulseSchedule().add(DriveChannel(0), wf)
        r = _SIM.run(s, _single_qubit())
        assert r.populations.get("0", 0.0) == pytest.approx(0.5, abs=0.01)
        assert r.populations.get("1", 0.0) == pytest.approx(0.5, abs=0.01)

    def test_rabi_period_correctness(self):
        """Rabi period T_rabi = 2π/Ω₀; check |0⟩ population oscillates."""
        Omega0 = 0.5   # rad/time
        # P₁ at T_pi = π/Ω₀ should be ≈ 1
        T_pi = math.pi / Omega0
        s = PulseSchedule().add(DriveChannel(0), ConstantWaveform(T_pi, Omega0))
        r = _SIM.run(s, _single_qubit())
        assert r.populations.get("1", 0.0) > 0.99

    def test_off_resonance_reduces_inversion(self):
        """Detuned drive: reduced population inversion (incomplete π-pulse)."""
        T = 10.0
        sys_det = _single_qubit(detuning=0.5)   # Δ = 0.5
        s = PulseSchedule().add(DriveChannel(0), _pi_const(T))
        r = _SIM.run(s, sys_det)
        p1 = r.populations.get("1", 0.0)
        assert p1 < 0.95   # detuned → not fully inverted


# ── Gaussian π-pulse ──────────────────────────────────────────────────────────

class TestGaussianPulse:
    def test_gaussian_pi_pulse_excites(self):
        sigma = 3.0
        duration = 6 * sigma          # 6-sigma truncation
        s = PulseSchedule().add(DriveChannel(0), _pi_gaussian(duration, sigma))
        r = _SIM.run(s, _single_qubit())
        assert r.populations.get("1", 0.0) == pytest.approx(1.0, abs=0.01)

    def test_pi_pulse_area_utility(self):
        """pi_pulse_area(sigma) * sigma * sqrt(2π) == π."""
        sigma = 4.0
        amp = pi_pulse_area(sigma)
        area = amp * sigma * math.sqrt(2 * math.pi)
        assert area == pytest.approx(math.pi, abs=1e-10)


# ── DRAG waveform ─────────────────────────────────────────────────────────────

class TestDRAGPulse:
    def test_drag_beta_zero_same_as_gaussian(self):
        """DRAG with beta=0 is identical to Gaussian."""
        sigma = 3.0; T = 6 * sigma
        drag = DRAGWaveform(T, pi_pulse_area(sigma), sigma, beta=0.0)
        s = PulseSchedule().add(DriveChannel(0), drag)
        r = _SIM.run(s, _single_qubit())
        assert r.populations.get("1", 0.0) == pytest.approx(1.0, abs=0.01)

    def test_drag_nonzero_beta_still_achieves_pi_pulse(self):
        """DRAG with nonzero beta on 2-level system still achieves ~π rotation."""
        sigma = 3.0; T = 6 * sigma
        drag = DRAGWaveform(T, pi_pulse_area(sigma), sigma, beta=0.5)
        s = PulseSchedule().add(DriveChannel(0), drag)
        r = _SIM.run(s, _single_qubit())
        # For 2-level system DRAG only adds a Y component; net rotation still ≈ π
        assert r.populations.get("1", 0.0) > 0.90


# ── FlatTop pulse ─────────────────────────────────────────────────────────────

class TestFlatTopPulse:
    def test_flat_top_pi_pulse(self):
        """FlatTop with appropriate area produces π-rotation."""
        T = 20.0; rise = 2.0; amp = 0.7
        # numerically integrate the envelope to get true area
        wf = FlatTopWaveform(T, amp, rise)
        t = np.linspace(0, T, 10000)
        area = float(np.trapezoid(wf.envelope(t), t))
        # Rescale amp to achieve area = π
        pi_amp = amp * math.pi / area
        wf2 = FlatTopWaveform(T, pi_amp, rise)
        s = PulseSchedule().add(DriveChannel(0), wf2)
        r = _SIM.run(s, _single_qubit())
        assert r.populations.get("1", 0.0) == pytest.approx(1.0, abs=0.01)


# ── ArbitraryWaveform pulse ───────────────────────────────────────────────────

class TestArbitraryWaveform:
    def test_arbitrary_const_pi_pulse(self):
        """ArbitraryWaveform replicating a square π-pulse."""
        N = 200; dt = 0.1
        T = N * dt            # = 20
        amp = math.pi / T
        samples = np.full(N, amp, dtype=complex)
        wf = ArbitraryWaveform(samples, dt=dt)
        s = PulseSchedule().add(DriveChannel(0), wf)
        r = _SIM.run(s, _single_qubit())
        assert r.populations.get("1", 0.0) == pytest.approx(1.0, abs=0.02)


# ── T1 relaxation ─────────────────────────────────────────────────────────────

class TestT1Relaxation:
    def test_excited_state_decays(self):
        """Start in |1⟩; after t ≈ T1 population drops to ≈ 1/e."""
        T1 = 50.0
        system = QubitSystem(1, [0.0], t1_times=[T1])
        # initial state |1⟩
        psi1 = np.array([0.0, 1.0], dtype=complex)
        # Empty schedule (no drive); just let system evolve under Lindblad
        s = PulseSchedule().add(DriveChannel(0), ConstantWaveform(T1, 0.0))
        r = _SIM.run(s, system, initial_state=psi1)
        p1 = r.populations.get("1", 0.0)
        expected = math.exp(-1.0)   # e^{-T1/T1}
        assert abs(p1 - expected) < 0.05

    def test_ground_state_stable_under_t1(self):
        """Ground state |0⟩ is stable (T1 damping doesn't create excitation)."""
        T1 = 20.0
        system = QubitSystem(1, [0.0], t1_times=[T1])
        s = PulseSchedule().add(DriveChannel(0), ConstantWaveform(T1, 0.0))
        r = _SIM.run(s, system)
        assert r.populations.get("0", 0.0) > 0.99

    def test_t1_decay_increases_ground_population(self):
        """P₀ after decay > initial P₀ = 0."""
        T1 = 30.0
        system = QubitSystem(1, [0.0], t1_times=[T1])
        psi1 = np.array([0.0, 1.0], dtype=complex)
        s = PulseSchedule().add(DriveChannel(0), ConstantWaveform(T1, 0.0))
        r = _SIM.run(s, system, initial_state=psi1)
        assert r.populations.get("0", 0.0) > 0.3


# ── T2 dephasing ──────────────────────────────────────────────────────────────

class TestT2Dephasing:
    def test_coherence_decays_under_t2(self):
        """⟨X⟩ of |+⟩ state decays toward zero under pure dephasing."""
        T2 = 20.0
        system = QubitSystem(1, [0.0], t2_times=[T2])
        psi_plus = np.array([1, 1], dtype=complex) / math.sqrt(2)
        s = PulseSchedule().add(DriveChannel(0), ConstantWaveform(T2, 0.0))
        r = _SIM.run(s, system, initial_state=psi_plus)
        bx = r.expectation(
            np.array([[0, 1], [1, 0]], dtype=complex)  # single-qubit X
        )
        # ⟨X⟩(T2) ≈ exp(−T2/T2) = 1/e ≈ 0.368, so should be < 0.8
        assert abs(bx) < 0.8


# ── Two-qubit simultaneous pulses ─────────────────────────────────────────────

class TestTwoQubitPulses:
    def _two_qubit_sys(self):
        return QubitSystem(2, [1.0, 1.0])

    def test_independent_pi_pulses_both_excited(self):
        """Simultaneous π-pulses on both qubits: |00⟩ → |11⟩."""
        T = 10.0
        s = (
            PulseSchedule()
            .add(DriveChannel(0), _pi_const(T))
            .add(DriveChannel(1), _pi_const(T))
        )
        r = _SIM.run(s, self._two_qubit_sys())
        assert r.populations.get("11", 0.0) == pytest.approx(1.0, abs=0.01)

    def test_pi_pulse_only_on_qubit_0(self):
        """π on qubit 0 (LSB) only: |00⟩ → |01⟩ (qubit 0 flipped, qubit 1 unchanged)."""
        T = 10.0
        s = PulseSchedule().add(DriveChannel(0), _pi_const(T))
        r = _SIM.run(s, self._two_qubit_sys())
        # Bitstring "01" = format(1, "02b") = q1=0 (MSB), q0=1 (LSB)
        assert r.populations.get("01", 0.0) == pytest.approx(1.0, abs=0.01)

    def test_sequential_pulses(self):
        """π on q0 then π on q1 (sequential): |00⟩ → |11⟩."""
        T = 10.0
        s = (
            PulseSchedule()
            .add(DriveChannel(0), _pi_const(T), t_start=0.0)
            .add(DriveChannel(1), _pi_const(T), t_start=T)
        )
        r = _SIM.run(s, self._two_qubit_sys())
        assert r.populations.get("11", 0.0) == pytest.approx(1.0, abs=0.01)


# ── PulseResult methods ───────────────────────────────────────────────────────

class TestPulseResult:
    def _run_pi(self):
        s = PulseSchedule().add(DriveChannel(0), _pi_const(math.pi))
        return _SIM.run(s, _single_qubit())

    def test_fidelity_with_target_one_state(self):
        r = self._run_pi()
        target = np.array([0.0, 1.0], dtype=complex)  # |1⟩
        assert r.fidelity(target) == pytest.approx(1.0, abs=1e-4)

    def test_fidelity_with_wrong_state_near_zero(self):
        r = self._run_pi()
        target = np.array([1.0, 0.0], dtype=complex)  # |0⟩
        assert r.fidelity(target) < 1e-4

    def test_bloch_vector_excited_state(self):
        r = self._run_pi()
        bx, by, bz = r.bloch_vector(0)
        assert bz == pytest.approx(-1.0, abs=0.01)   # |1⟩ → Bz = -1

    def test_bloch_vector_ground_state(self):
        s = PulseSchedule().add(DriveChannel(0), ConstantWaveform(math.pi, 0.0))
        r = _SIM.run(s, _single_qubit())
        _, _, bz = r.bloch_vector(0)
        assert bz == pytest.approx(1.0, abs=0.01)    # |0⟩ → Bz = +1

    def test_populations_sum_to_one(self):
        r = self._run_pi()
        assert sum(r.populations.values()) == pytest.approx(1.0, abs=1e-6)

    def test_metadata_n_qubits(self):
        r = self._run_pi()
        assert r.metadata["n_qubits"] == 1

    def test_fidelity_unnormalized_target_ok(self):
        r = self._run_pi()
        target = np.array([0.0, 3.0], dtype=complex)  # unnormalized |1⟩
        assert r.fidelity(target) == pytest.approx(1.0, abs=1e-4)

    def test_fidelity_zero_target_raises(self):
        r = self._run_pi()
        with pytest.raises(PulseError):
            r.fidelity(np.array([0.0, 0.0], dtype=complex))


# ── record_states ──────────────────────────────────────────────────────────────

class TestRecordStates:
    def test_record_states_shape(self):
        n_pts = 30
        sim = PulseSimulator(record_states=True, n_record_points=n_pts, atol=1e-8, rtol=1e-6)
        s = PulseSchedule().add(DriveChannel(0), _pi_const(10.0))
        r = sim.run(s, _single_qubit())
        assert r.states_at_times is not None
        assert r.states_at_times.shape == (n_pts, 2)

    def test_first_state_is_ground(self):
        sim = PulseSimulator(record_states=True, n_record_points=20, atol=1e-8, rtol=1e-6)
        s = PulseSchedule().add(DriveChannel(0), _pi_const(10.0))
        r = sim.run(s, _single_qubit())
        assert r.states_at_times is not None
        p0_init = abs(r.states_at_times[0, 0]) ** 2
        assert p0_init == pytest.approx(1.0, abs=0.01)
