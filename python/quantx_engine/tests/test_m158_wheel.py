"""M158: wheel-packaging smoke tests.

Verifies that:
1. ``Engine._default_library_path()`` honours the QUANTX_LIB_PATH env var.
2. ``Engine._default_library_path()`` returns the wheel-bundled library when a
   file with the correct platform name exists in the package directory.
3. Without either of the above, a string path is still returned (may not exist;
   that is handled by the ``ctypes.CDLL`` call in ``Engine.__init__``).
4. ``backend_info()`` returns a well-formed dict in both native and fallback mode.
5. ``pyproject.toml`` declares native-library globs in ``package-data``.

These tests run in CI without the native C++ library compiled, so they
exercise the *pure-Python fallback path* — i.e. the discovery logic itself
rather than the actual library loading.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _reload_engine():
    """Reload engine module so _default_library_path picks up any monkeypatching."""
    import quantx_engine.engine as eng
    importlib.reload(eng)
    return eng


# ---------------------------------------------------------------------------
# 1. Library path resolution
# ---------------------------------------------------------------------------

class TestDefaultLibraryPath:
    """Unit-test the _default_library_path() search order."""

    def test_returns_string(self, monkeypatch):
        monkeypatch.delenv("QUANTX_LIB_PATH", raising=False)
        from quantx_engine.engine import Engine
        path = Engine._default_library_path()
        assert isinstance(path, str), "must return a string"
        assert len(path) > 0, "must return a non-empty string"

    def test_env_var_takes_priority(self, monkeypatch):
        """QUANTX_LIB_PATH wins over everything else."""
        fake = "/tmp/fake_quantx.so"
        monkeypatch.setenv("QUANTX_LIB_PATH", fake)
        from quantx_engine.engine import Engine
        assert Engine._default_library_path() == fake

    def test_env_var_empty_string_falls_through(self, monkeypatch):
        """An empty QUANTX_LIB_PATH must not be returned (falsy check)."""
        monkeypatch.setenv("QUANTX_LIB_PATH", "")
        from quantx_engine.engine import Engine
        path = Engine._default_library_path()
        # Should fall through to package-dir or dev-build candidate
        assert path != "", "empty env var must not be returned"

    def test_package_dir_checked(self, monkeypatch):
        """A lib placed in the package directory is found before dev-build dirs."""
        import quantx_engine
        pkg_dir = os.path.dirname(os.path.abspath(quantx_engine.__file__))

        lib_name = (
            "quantx.dll" if os.name == "nt" else
            "libquantx.dylib" if sys.platform == "darwin" else
            "libquantx.so"
        )
        monkeypatch.delenv("QUANTX_LIB_PATH", raising=False)

        fake_lib = os.path.join(pkg_dir, lib_name)
        try:
            # Create a dummy file to simulate a wheel-bundled lib
            with open(fake_lib, "wb") as fh:
                fh.write(b"\x7fELF" if sys.platform != "win32" else b"MZ")

            eng = _reload_engine()
            assert eng.Engine._default_library_path() == fake_lib
        finally:
            if os.path.exists(fake_lib):
                os.remove(fake_lib)
            # Reload to clean state
            _reload_engine()

    def test_no_lib_returns_best_guess(self, monkeypatch):
        """When nothing exists, returns a path string ending in the lib name."""
        import tempfile
        monkeypatch.delenv("QUANTX_LIB_PATH", raising=False)
        # Change cwd to a fresh temp dir so no dev-build dirs exist
        tmpdir = tempfile.mkdtemp(prefix="qx_test_")
        try:
            monkeypatch.chdir(tmpdir)
            from quantx_engine.engine import Engine
            path = Engine._default_library_path()
            assert isinstance(path, str)
            expected_suffix = (
                "quantx.dll" if os.name == "nt" else
                "libquantx.dylib" if sys.platform == "darwin" else
                "libquantx.so"
            )
            assert path.endswith(expected_suffix), (
                f"best-guess path should end with {expected_suffix!r}, got {path!r}"
            )
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 2. backend_info() contract
# ---------------------------------------------------------------------------

class TestBackendInfo:
    """backend_info() must return a stable, well-typed dict in all modes."""

    def test_returns_dict(self):
        from quantx_engine import backend_info
        info = backend_info()
        assert isinstance(info, dict)

    def test_required_keys_present(self):
        from quantx_engine import backend_info
        info = backend_info()
        for key in ("native", "gpu", "version", "solvers", "max_qubits"):
            assert key in info, f"missing key: {key!r}"

    def test_native_is_bool(self):
        from quantx_engine import backend_info
        assert isinstance(backend_info()["native"], bool)

    def test_gpu_is_bool(self):
        from quantx_engine import backend_info
        assert isinstance(backend_info()["gpu"], bool)

    def test_version_is_nonempty_str(self):
        from quantx_engine import backend_info
        v = backend_info()["version"]
        assert isinstance(v, str) and v

    def test_solvers_is_list(self):
        from quantx_engine import backend_info
        assert isinstance(backend_info()["solvers"], list)

    def test_max_qubits_is_int_or_none(self):
        from quantx_engine import backend_info
        mq = backend_info()["max_qubits"]
        assert mq is None or isinstance(mq, int)

    def test_pure_python_version_format(self):
        """In pure-Python mode, version must contain 'pure-python-<ver>'."""
        from quantx_engine import backend_info, __version__
        info = backend_info()
        if not info["native"]:
            assert "pure-python" in info["version"], (
                "pure-Python fallback must set version to 'pure-python-<ver>'"
            )
            assert __version__ in info["version"]

    def test_pure_python_solvers_empty(self):
        """In pure-Python mode, solvers list must be empty."""
        from quantx_engine import backend_info
        info = backend_info()
        if not info["native"]:
            assert info["solvers"] == [], (
                "pure-Python fallback must return empty solvers list"
            )

    def test_pure_python_gpu_false(self):
        """In pure-Python mode, gpu must be False."""
        from quantx_engine import backend_info
        info = backend_info()
        if not info["native"]:
            assert info["gpu"] is False


# ---------------------------------------------------------------------------
# 3. pyproject.toml declares native-lib package-data globs
# ---------------------------------------------------------------------------

class TestPyprojectPackageData:
    """pyproject.toml must declare globs for native shared libraries."""

    def _read_pyproject(self) -> str:
        import pathlib
        # Traverse up from this test file to find python/pyproject.toml
        here = pathlib.Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "pyproject.toml"
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
        pytest.skip("pyproject.toml not found in any parent directory")

    def test_package_data_includes_so(self):
        content = self._read_pyproject()
        assert "*.so" in content, "pyproject.toml must include '*.so' in package-data"

    def test_package_data_includes_dll(self):
        content = self._read_pyproject()
        assert "*.dll" in content, "pyproject.toml must include '*.dll' in package-data"

    def test_package_data_includes_dylib(self):
        content = self._read_pyproject()
        assert "*.dylib" in content, "pyproject.toml must include '*.dylib' in package-data"

    def test_cibuildwheel_section_present(self):
        content = self._read_pyproject()
        assert "[tool.cibuildwheel]" in content, (
            "pyproject.toml must contain a [tool.cibuildwheel] section"
        )

    def test_before_build_references_script(self):
        content = self._read_pyproject()
        assert "build_native_for_wheel.py" in content, (
            "cibuildwheel before-build must reference build_native_for_wheel.py"
        )


# ---------------------------------------------------------------------------
# 4. Build script existence and syntax
# ---------------------------------------------------------------------------

class TestBuildScript:
    """scripts/build_native_for_wheel.py must exist and be importable."""

    def _script_path(self):
        import pathlib
        here = pathlib.Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "scripts" / "build_native_for_wheel.py"
            if candidate.exists():
                return candidate
        return None

    def test_script_exists(self):
        path = self._script_path()
        assert path is not None, (
            "scripts/build_native_for_wheel.py must exist in the repo"
        )

    def test_script_is_valid_python(self):
        path = self._script_path()
        if path is None:
            pytest.skip("script not found")
        import ast
        source = path.read_text(encoding="utf-8")
        # Will raise SyntaxError if the script is malformed
        ast.parse(source)

    def test_script_has_main_guard(self):
        path = self._script_path()
        if path is None:
            pytest.skip("script not found")
        content = path.read_text(encoding="utf-8")
        assert '__name__ == "__main__"' in content or \
               "__name__ == '__main__'" in content, \
               "script must have an if __name__ == '__main__' guard"
