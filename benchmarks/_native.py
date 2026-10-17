"""Finding the native programs that the TeNPy comparison scripts time.

The QuantX side of those comparisons runs C++ programs from a CMake build
(``qx_tebd_bench``, ``qx_hubbard_showcase``), which time the engine without the Python
and JSON layers. They are found under ``QX_BUILD_DIR`` (default ``build/`` at the
repository root), in ``Release/`` for a multi-config generator or at the top for a
single-config one; a program's own variable (``QX_TEBD_BENCH``, ``QX_HUB_SHOWCASE``)
overrides the search with a full path. ``QX_DLL_DIR``, when set, goes first on PATH for
the runs: a Windows build with OpenBLAS or CUDA needs its DLLs found there.

See benchmarks/REPRODUCING.md for what each comparison needs.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def program(name: str, override: str) -> str:
    """The path of the native program ``name``; exits with how to build it if missing."""
    path = os.environ.get(override)
    if path:
        if not os.path.exists(path):
            sys.exit(f"{override}={path} does not exist")
        return path
    build = os.environ.get("QX_BUILD_DIR", os.path.join(ROOT, "build"))
    exe = name + (".exe" if os.name == "nt" else "")
    for sub in ("Release", ""):
        candidate = os.path.join(build, sub, exe)
        if os.path.exists(candidate):
            return candidate
    sys.exit(f"{exe} not found under {build}. Build it with\n"
             f"  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n"
             f"  cmake --build build --config Release --target {name}\n"
             f"or point QX_BUILD_DIR (or {override}) at a build that has it.")


def env(**extra) -> dict:
    """os.environ with QX_DLL_DIR first on PATH (when set) and ``extra`` added."""
    e = dict(os.environ)
    dll = os.environ.get("QX_DLL_DIR")
    if dll:
        e["PATH"] = dll + os.pathsep + e.get("PATH", "")
    e.update({k: str(v) for k, v in extra.items()})
    return e
