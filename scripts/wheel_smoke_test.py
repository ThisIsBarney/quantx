"""Smoke-test an installed wheel: the native engine loads from it, at the version pip installed.

cibuildwheel runs this after installing each platform wheel (python/pyproject.toml,
test-command). It used to be an inline `python -c "..."` spanning several lines, which bash
runs whole but cmd.exe cuts at the first line break: on Windows the check never ran, and the
job passed with no output at all. A file runs the same everywhere.
"""
import quantx_engine
from quantx_engine import backend_info
from quantx_engine.engine import Engine

info = backend_info()
print("native:", info["native"], "| version:", info["version"])
print("pybind11:", Engine.pybind11_available())
assert info["native"], "Native engine not found in wheel"
# The engine once said 0.1.0-dev inside a package that said 0.4.0, inside a distribution that
# said 1.0.0.
assert info["version"] == quantx_engine.__version__, (info["version"], quantx_engine.__version__)
print("wheel smoke test PASSED")
