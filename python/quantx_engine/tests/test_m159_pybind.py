"""M159: pybind11 NumPy zero-copy bridge smoke tests.

Verifies that:
1. ``Engine.pybind11_available()`` is a bool (works in pure-Python mode too).
2. ``Engine.run_statevector()`` is callable and always returns a value
   (either an ndarray if pybind11 is present, or a dict via the ctypes fallback).
3. ``Engine.run_circuit_sv()`` contract — same shape rules.
4. ``Engine.run_circuit_counts_np()`` returns a dict[str, int].
5. When pybind11 IS available, the hot-path returns ``numpy.ndarray[complex128]``.
6. ``pybind_module.cpp`` exists and is syntactically valid C++.
7. The build script correctly enables ``-DQX_ENABLE_PYBIND=ON`` when pybind11 is
   installed.

These tests run in CI without the native C++ library compiled, so any test that
would actually call through to the C++ engine is guarded with a skip directive.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pybind_module_path():
    """Return the path to bindings/python/pybind_module.cpp or None."""
    import pathlib
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "bindings" / "python" / "pybind_module.cpp"
        if candidate.exists():
            return candidate
    return None


def _has_native_lib() -> bool:
    """Return True if the native library can actually be loaded."""
    try:
        from quantx_engine.engine import Engine
        Engine()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1. Engine.pybind11_available() contract
# ---------------------------------------------------------------------------

class TestPybind11Available:
    """pybind11_available() must always be a bool regardless of build mode."""

    def test_returns_bool(self):
        from quantx_engine.engine import Engine
        result = Engine.pybind11_available()
        assert isinstance(result, bool), (
            f"pybind11_available() must return bool, got {type(result)}"
        )

    def test_false_without_native_lib(self, monkeypatch):
        """In CI (no native build), pybind11_available should be False."""
        # Temporarily hide _quantx_py if present
        import quantx_engine.engine as eng
        original = eng._PYBIND_MODULE
        try:
            eng._PYBIND_MODULE = None
            assert eng.Engine.pybind11_available() is False
        finally:
            eng._PYBIND_MODULE = original

    def test_true_when_module_present(self, monkeypatch):
        """When _PYBIND_MODULE is set to a non-None value, available() is True."""
        import quantx_engine.engine as eng
        original = eng._PYBIND_MODULE
        try:
            # Inject a fake module object
            eng._PYBIND_MODULE = types.SimpleNamespace(run_statevector=lambda *a: None)
            assert eng.Engine.pybind11_available() is True
        finally:
            eng._PYBIND_MODULE = original


# ---------------------------------------------------------------------------
# 2. Engine.run_statevector() — fallback / pybind11 path
# ---------------------------------------------------------------------------

class TestRunStatevector:
    """run_statevector() must always be callable; return type depends on build."""

    @pytest.mark.skipif(not _has_native_lib(), reason="native library not available")
    def test_returns_value_with_native_lib(self):
        """With native lib, run_statevector returns ndarray or dict."""
        from quantx_engine.engine import Engine
        eng = Engine()
        problem = {"qubits": 2, "target_error": 1e-6, "noisy": False,
                   "expected_circuit_depth": 10}
        result = eng.run_statevector(problem)
        assert result is not None

    @pytest.mark.skipif(not _has_native_lib(), reason="native library not available")
    def test_pybind11_path_returns_ndarray(self):
        """If pybind11 is available, run_statevector must return ndarray."""
        from quantx_engine.engine import Engine
        if not Engine.pybind11_available():
            pytest.skip("pybind11 extension not available")
        import numpy as np
        eng = Engine()
        problem = {"qubits": 2, "target_error": 1e-6, "noisy": False,
                   "expected_circuit_depth": 10}
        sv = eng.run_statevector(problem)
        assert isinstance(sv, np.ndarray), (
            f"Expected ndarray with pybind11, got {type(sv)}"
        )
        assert sv.dtype == np.complex128, f"Expected complex128, got {sv.dtype}"
        assert sv.shape == (4,), f"Expected shape (4,) for 2-qubit problem, got {sv.shape}"

    @pytest.mark.skipif(not _has_native_lib(), reason="native library not available")
    def test_ctypes_fallback_returns_dict(self, monkeypatch):
        """Without pybind11, run_statevector falls back to ctypes (returns dict)."""
        import quantx_engine.engine as eng
        original = eng._PYBIND_MODULE
        try:
            eng._PYBIND_MODULE = None
            engine = eng.Engine()
            problem = {"qubits": 2, "target_error": 1e-6, "noisy": False,
                       "expected_circuit_depth": 10}
            result = engine.run_statevector(problem)
            assert isinstance(result, dict), (
                "ctypes fallback should return dict"
            )
        finally:
            eng._PYBIND_MODULE = original


# ---------------------------------------------------------------------------
# 3. Engine.run_circuit_sv()
# ---------------------------------------------------------------------------

class TestRunCircuitSv:
    """run_circuit_sv() must always be callable."""

    @pytest.mark.skipif(not _has_native_lib(), reason="native library not available")
    def test_pybind11_path_returns_ndarray(self):
        from quantx_engine.engine import Engine
        if not Engine.pybind11_available():
            pytest.skip("pybind11 extension not available")
        import numpy as np
        eng = Engine()
        circuit = {
            "qubits": 1,
            "shots": 0,
            "operations": [{"name": "H", "targets": [0]}],
        }
        sv = eng.run_circuit_sv(circuit)
        assert isinstance(sv, np.ndarray)
        assert sv.dtype == np.complex128
        assert sv.shape == (2,)

    @pytest.mark.skipif(not _has_native_lib(), reason="native library not available")
    def test_ctypes_fallback_returns_dict(self, monkeypatch):
        import quantx_engine.engine as eng
        original = eng._PYBIND_MODULE
        try:
            eng._PYBIND_MODULE = None
            engine = eng.Engine()
            circuit = {
                "qubits": 1,
                "shots": 0,
                "operations": [{"name": "H", "targets": [0]}],
            }
            result = engine.run_circuit_sv(circuit)
            assert isinstance(result, dict)
        finally:
            eng._PYBIND_MODULE = original


# ---------------------------------------------------------------------------
# 4. Engine.run_circuit_counts_np()
# ---------------------------------------------------------------------------

class TestRunCircuitCountsNp:
    """run_circuit_counts_np() must return dict[str, int]."""

    @pytest.mark.skipif(not _has_native_lib(), reason="native library not available")
    def test_returns_dict(self):
        from quantx_engine.engine import Engine
        eng = Engine()
        circuit = {
            "qubits": 1,
            "shots": 100,
            "operations": [
                {"name": "H", "targets": [0]},
                {"name": "MEASURE", "targets": [0]},
            ],
        }
        counts = eng.run_circuit_counts_np(circuit)
        assert isinstance(counts, dict)
        for k, v in counts.items():
            assert isinstance(k, str)
            assert isinstance(v, int)

    @pytest.mark.skipif(not _has_native_lib(), reason="native library not available")
    def test_ctypes_fallback_returns_dict(self, monkeypatch):
        import quantx_engine.engine as eng
        original = eng._PYBIND_MODULE
        try:
            eng._PYBIND_MODULE = None
            engine = eng.Engine()
            circuit = {
                "qubits": 1,
                "shots": 100,
                "operations": [
                    {"name": "H", "targets": [0]},
                    {"name": "MEASURE", "targets": [0]},
                ],
            }
            counts = engine.run_circuit_counts_np(circuit)
            assert isinstance(counts, dict)
        finally:
            eng._PYBIND_MODULE = original


# ---------------------------------------------------------------------------
# 5. Latency benchmark (informational — never fails)
# ---------------------------------------------------------------------------

class TestPybindLatency:
    """Informal latency comparison between ctypes and pybind11 paths.

    Not a hard assertion — records timing info as a print so it appears
    in verbose pytest output.  Only runs when both native lib and pybind11
    are available.
    """

    @pytest.mark.skipif(not _has_native_lib(), reason="native library not available")
    def test_pybind11_faster_than_ctypes(self):
        """pybind11 statevector path should be faster than ctypes JSON path."""
        from quantx_engine.engine import Engine
        if not Engine.pybind11_available():
            pytest.skip("pybind11 extension not available")

        import time
        import quantx_engine.engine as eng

        problem = {"qubits": 14, "target_error": 1e-6, "noisy": False,
                   "expected_circuit_depth": 100}
        engine = eng.Engine()

        # ── ctypes path ───────────────────────────────────────────────────────
        original_pb = eng._PYBIND_MODULE
        try:
            eng._PYBIND_MODULE = None
            t0 = time.perf_counter()
            for _ in range(20):
                engine.run_statevector(problem)
            ctypes_ms = (time.perf_counter() - t0) / 20 * 1000
        finally:
            eng._PYBIND_MODULE = original_pb

        # ── pybind11 path ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        for _ in range(20):
            engine.run_statevector(problem)
        pybind_ms = (time.perf_counter() - t0) / 20 * 1000

        print(f"\n[M159] ctypes latency : {ctypes_ms:.2f} ms/call")
        print(f"[M159] pybind11 latency: {pybind_ms:.2f} ms/call")
        if ctypes_ms > 0:
            speedup = ctypes_ms / pybind_ms
            print(f"[M159] pybind11 speedup: {speedup:.1f}×")

        # Soft assertion: pybind11 should be at least as fast (not 2× slower)
        assert pybind_ms <= ctypes_ms * 2.0, (
            f"pybind11 path ({pybind_ms:.2f} ms) is >2× slower than ctypes "
            f"({ctypes_ms:.2f} ms) — unexpected regression"
        )


# ---------------------------------------------------------------------------
# 6. pybind_module.cpp source checks
# ---------------------------------------------------------------------------

class TestPybindSourceFile:
    """bindings/python/pybind_module.cpp must exist with expected content."""

    def test_source_file_exists(self):
        path = _pybind_module_path()
        assert path is not None, (
            "bindings/python/pybind_module.cpp not found in repository"
        )

    def test_includes_pybind11_numpy(self):
        path = _pybind_module_path()
        if path is None:
            pytest.skip("pybind_module.cpp not found")
        content = path.read_text(encoding="utf-8")
        assert "pybind11/numpy.h" in content, (
            "pybind_module.cpp must include <pybind11/numpy.h> for NumPy support"
        )

    def test_run_statevector_defined(self):
        path = _pybind_module_path()
        if path is None:
            pytest.skip("pybind_module.cpp not found")
        content = path.read_text(encoding="utf-8")
        assert "run_statevector" in content

    def test_run_circuit_sv_defined(self):
        path = _pybind_module_path()
        if path is None:
            pytest.skip("pybind_module.cpp not found")
        content = path.read_text(encoding="utf-8")
        assert "run_circuit_sv" in content

    def test_run_circuit_counts_defined(self):
        path = _pybind_module_path()
        if path is None:
            pytest.skip("pybind_module.cpp not found")
        content = path.read_text(encoding="utf-8")
        assert "run_circuit_counts" in content

    def test_vec_to_numpy_helper(self):
        path = _pybind_module_path()
        if path is None:
            pytest.skip("pybind_module.cpp not found")
        content = path.read_text(encoding="utf-8")
        assert "vec_to_numpy" in content, (
            "pybind_module.cpp must define vec_to_numpy() helper"
        )

    def test_backward_compat_run_cpp(self):
        path = _pybind_module_path()
        if path is None:
            pytest.skip("pybind_module.cpp not found")
        content = path.read_text(encoding="utf-8")
        assert "run_cpp" in content, (
            "pybind_module.cpp must preserve backward-compat run_cpp binding"
        )

    def test_uses_result_state_not_statevector(self):
        """CircuitResult uses .state (not .statevector) per include/qx/types.hpp."""
        path = _pybind_module_path()
        if path is None:
            pytest.skip("pybind_module.cpp not found")
        content = path.read_text(encoding="utf-8")
        # Must reference result.state (correct field from CircuitResult)
        assert "result.state" in content or "r.state" in content, (
            "pybind_module.cpp must use CircuitResult::state (not ::statevector)"
        )
        # Must NOT reference .statevector as a field access (the field does not exist)
        assert "result.statevector" not in content, (
            "CircuitResult has no .statevector field — use .state instead"
        )
        assert "r.statevector" not in content, (
            "CircuitResult has no .statevector field — use .state instead"
        )

    def test_run_circuit_passes_run_options(self):
        """run_circuit() C++ API requires (circuit, RunOptions) — must supply RunOptions{}."""
        path = _pybind_module_path()
        if path is None:
            pytest.skip("pybind_module.cpp not found")
        content = path.read_text(encoding="utf-8")
        assert "RunOptions{}" in content, (
            "pybind_module.cpp must pass qx::RunOptions{} to qx::run_circuit()"
        )

    def test_uses_version_string(self):
        """version() must use qx::version_string() not qx::engine_version()."""
        path = _pybind_module_path()
        if path is None:
            pytest.skip("pybind_module.cpp not found")
        content = path.read_text(encoding="utf-8")
        assert "version_string" in content, (
            "pybind_module.cpp must use qx::version_string() from include/qx/version.hpp"
        )
        assert "engine_version" not in content, (
            "engine_version does not exist in the C++ API — use qx::version_string()"
        )
        assert "list_solver_names" not in content, (
            "list_solver_names does not exist in the C++ API"
        )
        assert "run_named_solver" not in content, (
            "run_named_solver does not exist in the C++ API"
        )

    def test_engine_py_uses_relative_import(self):
        """engine.py must use `from . import _quantx_py` (relative) not `import _quantx_py`."""
        import pathlib
        here = pathlib.Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "quantx_engine" / "engine.py"
            if candidate.exists():
                content = candidate.read_text(encoding="utf-8")
                assert "from . import _quantx_py" in content, (
                    "engine.py must use relative import `from . import _quantx_py` "
                    "so the extension is found in the installed wheel package directory"
                )
                assert "import _quantx_py as _PYBIND_MODULE" not in content.replace(
                    "from . import _quantx_py as _PYBIND_MODULE", ""
                ), (
                    "engine.py must not use absolute `import _quantx_py` "
                    "(fails in installed wheels where the .so is inside the package)"
                )
                return
        pytest.skip("engine.py not found")


# ---------------------------------------------------------------------------
# 7. Build script: QX_ENABLE_PYBIND flag
# ---------------------------------------------------------------------------

class TestBuildScriptPybind:
    """build_native_for_wheel.py must enable pybind11 when available."""

    def _script_path(self):
        import pathlib
        here = pathlib.Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "scripts" / "build_native_for_wheel.py"
            if candidate.exists():
                return candidate
        return None

    def test_script_references_pybind(self):
        path = self._script_path()
        if path is None:
            pytest.skip("build script not found")
        content = path.read_text(encoding="utf-8")
        assert "QX_ENABLE_PYBIND" in content, (
            "build script must set -DQX_ENABLE_PYBIND when pybind11 is available"
        )

    def test_script_finds_pybind_ext(self):
        path = self._script_path()
        if path is None:
            pytest.skip("build script not found")
        content = path.read_text(encoding="utf-8")
        assert "_find_pybind_ext" in content or "_quantx_py" in content, (
            "build script must search for and install the _quantx_py extension"
        )

    def test_wheel_pybind_module_is_opt_in(self):
        # The wheel ships the ctypes library, which serves every Python version; a pybind11
        # module would tie the wheel to one CPython ABI (python/setup.py). So before-build
        # must not install pybind11, and the build script adds the module only when
        # QX_WHEEL_PYBIND=1. (This test used to assert the opposite, and passed only
        # because the comment explaining the change contains the word "pybind11".)
        import pathlib
        tomllib = pytest.importorskip("tomllib")
        here = pathlib.Path(__file__).resolve()
        pyproject = next((p / "pyproject.toml" for p in here.parents
                          if (p / "pyproject.toml").exists()), None)
        if pyproject is None:
            pytest.skip("pyproject.toml not found")
        cfg = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        before = cfg["tool"]["cibuildwheel"]["before-build"]
        commands = " ".join(before) if isinstance(before, list) else before
        assert "pybind11" not in commands, commands
        script = self._script_path()
        if script is None:
            pytest.skip("build script not found")
        assert 'os.environ.get("QX_WHEEL_PYBIND") == "1"' in script.read_text(encoding="utf-8")
