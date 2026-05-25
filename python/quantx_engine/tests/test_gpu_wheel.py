"""M168 — Unit tests for the GPU wheel and CuPy backend.

These tests cover the GpuBackend, GpuStatevector, and gpu_info() API.
They are split into two groups:

* **Fallback tests** (default) — run without CuPy; verify that:
    - ``GpuBackend("numpy")`` always works
    - ``GpuBackend("auto")`` silently falls back to numpy when CuPy is absent
    - ``GpuBackend("cupy")`` raises ``GpuError`` when CuPy is absent
    - ``gpu_info()`` returns a well-formed dict in both cases
    - ``GpuStatevector`` works on the numpy backend (CPU fallback path)

* **Real GPU tests** (``pytest.mark.gpu``) — run only on NVIDIA CI runners
  where CuPy is actually installed.  Verify that GpuBackend("cupy") moves
  arrays to the device and back correctly.

Run the fallback tests::

    pytest python/quantx_engine/tests/test_gpu_wheel.py -m "not gpu"

Run real GPU tests (requires cupy-cuda12x)::

    pytest python/quantx_engine/tests/test_gpu_wheel.py -m gpu
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ===========================================================================
# Helpers
# ===========================================================================

def _cupy_absent():
    """Return True when CuPy is NOT installed in this environment."""
    try:
        import cupy  # noqa: F401
        return False
    except ImportError:
        return True


# ===========================================================================
# gpu_info() — shape tests (no CuPy required)
# ===========================================================================

def test_gpu_info_returns_dict():
    """gpu_info() always returns a dict with the expected keys."""
    from quantx_engine.gpu import gpu_info
    info = gpu_info()
    assert isinstance(info, dict)
    assert "cupy_available" in info
    assert isinstance(info["cupy_available"], bool)


def test_gpu_info_without_cupy():
    """gpu_info() reports cupy_available=False when CuPy is absent."""
    from quantx_engine import gpu as gpu_module
    with patch.object(gpu_module, "_HAS_CUPY", False), \
         patch.object(gpu_module, "cp", None):
        from quantx_engine.gpu import gpu_info
        info = gpu_info()
        assert info["cupy_available"] is False


def test_gpu_info_with_mock_cupy():
    """gpu_info() reports device count and version when CuPy is present (mocked)."""
    from quantx_engine import gpu as gpu_module
    mock_cp = MagicMock()
    mock_cp.__version__ = "13.0.0"
    mock_cp.cuda.runtime.getDeviceCount.return_value = 2
    mock_cp.cuda.Device.return_value.__enter__ = lambda s: s
    mock_cp.cuda.Device.return_value.__exit__ = MagicMock(return_value=False)
    mock_cp.cuda.Device.return_value.id = 0
    mock_cp.cuda.runtime.getDeviceProperties.return_value = {"name": b"Tesla T4"}
    mock_cp.cuda.Device.return_value.mem_info = (8 * 2**30, 16 * 2**30)

    with patch.object(gpu_module, "_HAS_CUPY", True), \
         patch.object(gpu_module, "cp", mock_cp):
        from quantx_engine.gpu import gpu_info
        info = gpu_info()
    assert info["cupy_available"] is True
    assert info.get("cupy_version") == "13.0.0"
    assert info.get("n_devices") == 2


# ===========================================================================
# GpuBackend — fallback (numpy) path
# ===========================================================================

def test_gpu_backend_numpy_always_works():
    """GpuBackend('numpy') succeeds regardless of whether CuPy is installed."""
    from quantx_engine.gpu import GpuBackend
    b = GpuBackend("numpy")
    assert b.name == "numpy"
    assert not b.is_gpu()
    assert b.xp is np


def test_gpu_backend_auto_falls_back_to_numpy_without_cupy():
    """GpuBackend('auto') resolves to numpy when CuPy is absent."""
    from quantx_engine import gpu as gpu_module
    with patch.object(gpu_module, "_HAS_CUPY", False), \
         patch.object(gpu_module, "cp", None):
        from quantx_engine.gpu import GpuBackend
        b = GpuBackend("auto")
    assert b.name == "numpy"
    assert not b.is_gpu()


def test_gpu_backend_cupy_raises_without_cupy():
    """GpuBackend('cupy') raises GpuError when CuPy is not installed."""
    from quantx_engine import gpu as gpu_module
    from quantx_engine.gpu import GpuError
    with patch.object(gpu_module, "_HAS_CUPY", False), \
         patch.object(gpu_module, "cp", None):
        from quantx_engine.gpu import GpuBackend
        with pytest.raises(GpuError, match="CuPy is not installed"):
            GpuBackend("cupy")


def test_gpu_backend_unknown_name_raises():
    """GpuBackend with an unknown name raises GpuError."""
    from quantx_engine.gpu import GpuBackend, GpuError
    with pytest.raises(GpuError, match="Unknown backend"):
        GpuBackend("tpu")


def test_gpu_backend_array_numpy():
    """GpuBackend('numpy').array() returns a numpy ndarray."""
    from quantx_engine.gpu import GpuBackend
    b = GpuBackend("numpy")
    arr = b.array([1, 2, 3], dtype=complex)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == complex


def test_gpu_backend_zeros_numpy():
    """GpuBackend('numpy').zeros() returns a zero numpy array."""
    from quantx_engine.gpu import GpuBackend
    b = GpuBackend("numpy")
    z = b.zeros((4,), dtype=complex)
    assert isinstance(z, np.ndarray)
    assert np.all(z == 0)


def test_gpu_backend_to_dict():
    """GpuBackend.to_dict() returns expected keys."""
    from quantx_engine.gpu import GpuBackend
    b = GpuBackend("numpy")
    d = b.to_dict()
    assert "backend" in d
    assert "gpu_available" in d
    assert d["backend"] == "numpy"


def test_gpu_backend_to_numpy_passthrough():
    """to_numpy() on a numpy array returns the same array."""
    from quantx_engine.gpu import GpuBackend
    b = GpuBackend("numpy")
    arr = np.array([1.0 + 0j, 0.0 + 1j])
    result = b.to_numpy(arr)
    np.testing.assert_array_equal(result, arr)


# ===========================================================================
# GpuBackend — mocked CuPy path
# ===========================================================================

def test_gpu_backend_auto_uses_cupy_when_available():
    """GpuBackend('auto') resolves to cupy when CuPy is mocked as present."""
    from quantx_engine import gpu as gpu_module
    mock_cp = MagicMock()
    mock_cp.__version__ = "13.0.0"

    with patch.object(gpu_module, "_HAS_CUPY", True), \
         patch.object(gpu_module, "cp", mock_cp):
        from quantx_engine.gpu import GpuBackend
        b = GpuBackend("auto")
    assert b.name == "cupy"
    assert b.is_gpu()


def test_gpu_backend_to_numpy_converts_cupy_array():
    """to_numpy() calls cp.asnumpy when CuPy array is present (mocked)."""
    from quantx_engine import gpu as gpu_module
    mock_cp = MagicMock()
    mock_cp.__version__ = "13.0.0"
    expected = np.array([1.0, 2.0])
    mock_cp.asnumpy.return_value = expected

    # Make isinstance(arr, cp.ndarray) True
    FakeCuArray = type("ndarray", (), {})
    mock_cp.ndarray = FakeCuArray
    fake_arr = FakeCuArray()

    with patch.object(gpu_module, "_HAS_CUPY", True), \
         patch.object(gpu_module, "cp", mock_cp):
        from quantx_engine.gpu import GpuBackend
        b = GpuBackend("cupy")
        result = b.to_numpy(fake_arr)
    mock_cp.asnumpy.assert_called_once_with(fake_arr)
    np.testing.assert_array_equal(result, expected)


# ===========================================================================
# GpuStatevector — numpy fallback
# ===========================================================================

def test_gpu_statevector_numpy_backend_initialises():
    """GpuStatevector works on the numpy backend (CPU fallback)."""
    from quantx_engine.gpu import GpuStatevector
    sv = GpuStatevector(n_qubits=2, backend="numpy")
    assert sv.n_qubits == 2
    assert sv.backend == "numpy"


def test_gpu_statevector_numpy_state_is_zero():
    """GpuStatevector starts in |0...0> (first amplitude = 1)."""
    from quantx_engine.gpu import GpuStatevector
    sv = GpuStatevector(n_qubits=2, backend="numpy")
    state = sv.state()
    assert abs(state[0] - 1.0) < 1e-12
    assert np.allclose(state[1:], 0)


def test_gpu_statevector_norm_preserved_after_h():
    """Applying H to qubit 0 preserves norm on numpy backend."""
    from quantx_engine.gpu import GpuStatevector
    sv = GpuStatevector(n_qubits=2, backend="numpy")
    sv.apply_gate("H", qubits=0)
    state = sv.state()
    assert abs(np.linalg.norm(state) - 1.0) < 1e-12


# ===========================================================================
# pyproject.toml — [gpu] extra is declared
# ===========================================================================

def test_pyproject_gpu_extra_declared():
    """pyproject.toml declares gpu = ['cupy-cuda12x>=13.0'] as an optional dep."""
    import tomllib
    import pathlib
    pyproject = pathlib.Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    gpu_deps = data["project"]["optional-dependencies"]["gpu"]
    assert any("cupy-cuda12x" in dep for dep in gpu_deps), (
        f"Expected cupy-cuda12x in [gpu] extra, got: {gpu_deps}"
    )


def test_pyproject_gpu_extra_has_version_bound():
    """The cupy-cuda12x dependency has a >=13.0 version bound."""
    import tomllib
    import pathlib
    pyproject = pathlib.Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    gpu_deps = data["project"]["optional-dependencies"]["gpu"]
    cupy_dep = next(d for d in gpu_deps if "cupy-cuda12x" in d)
    assert ">=" in cupy_dep, f"Expected version bound in {cupy_dep!r}"


# ===========================================================================
# Real GPU tests — skipped unless pytest.mark.gpu is selected
# ===========================================================================

@pytest.mark.gpu
def test_real_cupy_backend():
    """GpuBackend('cupy') works on a real CUDA device.

    Run with: pytest -m gpu
    Requires: cupy-cuda12x to be installed and a CUDA 12 device available.
    """
    try:
        import cupy as cp
    except ImportError:
        pytest.skip("cupy not installed")

    from quantx_engine.gpu import GpuBackend
    b = GpuBackend("cupy")
    assert b.is_gpu()
    arr = b.array(np.ones(4, dtype=complex))
    assert isinstance(arr, cp.ndarray)
    back = b.to_numpy(arr)
    np.testing.assert_allclose(back, np.ones(4, dtype=complex))


@pytest.mark.gpu
def test_real_gpu_info_device_count():
    """gpu_info() reports at least one CUDA device on a GPU runner.

    Run with: pytest -m gpu
    """
    try:
        import cupy  # noqa: F401
    except ImportError:
        pytest.skip("cupy not installed")

    from quantx_engine.gpu import gpu_info
    info = gpu_info()
    assert info["cupy_available"] is True
    assert info.get("n_devices", 0) >= 1, "Expected at least one CUDA device"


@pytest.mark.gpu
def test_real_gpu_statevector_bell_state():
    """GpuStatevector produces a correct Bell state on real CUDA hardware."""
    try:
        import cupy  # noqa: F401
    except ImportError:
        pytest.skip("cupy not installed")

    from quantx_engine.gpu import GpuStatevector
    sv = GpuStatevector(n_qubits=2, backend="cupy")
    sv.apply_gate("H", qubits=0)
    sv.apply_gate("CNOT", qubits=[0, 1])
    state = sv.state()
    # Bell state: 1/√2 (|00⟩ + |11⟩)
    assert abs(state[0] - 1 / np.sqrt(2)) < 1e-10
    assert abs(state[1]) < 1e-10
    assert abs(state[2]) < 1e-10
    assert abs(state[3] - 1 / np.sqrt(2)) < 1e-10
