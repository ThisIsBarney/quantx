"""M7: the experimental modules live in quantx_engine.contrib, and their old paths still work.

Until 0.4.0, ``import quantx_engine.<m>`` gives quantx_engine.contrib.<m> itself, and each
name the package exported before 0.2.0 resolves to the contrib object it named; both warn
with a DeprecationWarning that points at the caller's line. ``import quantx_engine`` loads
the engine API only.
"""
from __future__ import annotations

import importlib
import pathlib
import subprocess
import sys
import warnings

import pytest

import quantx_engine
from quantx_engine import _MOVED_MODULES, _MOVED_NAMES

ENGINE_API = {"__init__", "_deprecation", "engine", "dmrg", "models", "hva", "thermal", "tdvp",
              "structure_factor", "u1_circuit"}
PKG = pathlib.Path(quantx_engine.__file__).parent


def _forget(name):
    """Undo an earlier old-path import, so the next one goes through the alias again."""
    sys.modules.pop(f"quantx_engine.{name}", None)
    quantx_engine.__dict__.pop(name, None)


def test_every_module_outside_the_engine_api_is_in_contrib():
    top = {p.stem for p in PKG.glob("*.py")}
    contrib = {p.stem for p in (PKG / "contrib").glob("*.py")} - {"__init__"}
    assert top == ENGINE_API
    assert contrib == set(_MOVED_MODULES)


def test_all_is_the_engine_api():
    assert set(quantx_engine.__all__) == {"__version__", "backend_info", "Engine",
                                          "CircuitError", "QuantXNativeUnavailable"}


def test_import_loads_no_contrib_module():
    code = ("import sys, quantx_engine; "
            "print([m for m in sys.modules if m.startswith('quantx_engine.contrib')])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "[]"


@pytest.mark.parametrize("name", sorted(_MOVED_MODULES))
def test_old_module_path_is_the_contrib_module(name):
    new = importlib.import_module(f"quantx_engine.contrib.{name}")
    _forget(name)
    with pytest.warns(DeprecationWarning, match=rf"quantx_engine\.contrib\.{name}\b"):
        old = importlib.import_module(f"quantx_engine.{name}")
    assert old is new
    assert new.__name__ == new.__spec__.name == f"quantx_engine.contrib.{name}"
    _forget(name)
    with pytest.warns(DeprecationWarning, match=rf"quantx_engine\.contrib\.{name}\b"):
        assert getattr(quantx_engine, name) is new


@pytest.mark.parametrize("name", sorted(_MOVED_NAMES))
def test_old_top_level_name_is_the_contrib_object(name):
    module, attr = _MOVED_NAMES[name]
    new = getattr(importlib.import_module(f"quantx_engine.contrib.{module}"), attr)
    with pytest.warns(DeprecationWarning, match=rf"contrib\.{module}\.{attr}\b"):
        old = getattr(quantx_engine, name)
    assert old is new


def test_the_warnings_point_at_the_caller():
    _forget("qec")
    _forget("qaoa_optim")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import quantx_engine.qec  # noqa: F401
        from quantx_engine import SurfaceCode  # noqa: F401
        quantx_engine.qaoa_optim  # noqa: B018
    files = [pathlib.Path(w.filename).resolve() for w in caught
             if issubclass(w.category, DeprecationWarning)]
    assert files == [pathlib.Path(__file__).resolve()] * 3, files


def test_an_unknown_name_is_still_an_attribute_error():
    with pytest.raises(AttributeError, match="no_such_name"):
        quantx_engine.no_such_name  # noqa: B018
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("quantx_engine.no_such_module")
