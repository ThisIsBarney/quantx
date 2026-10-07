#!/usr/bin/env python3
"""Build the QuantX native shared library and copy it into the Python package directory.

This script is called by the cibuildwheel ``before-build`` hook so that the
compiled ``libquantx.so`` / ``libquantx.dylib`` / ``quantx.dll`` ends up inside
``python/quantx_engine/`` before ``pip`` packages the wheel.  Once the wheel is
installed, :class:`~quantx_engine.engine.Engine` finds the library automatically
(no ``QUANTX_LIB_PATH`` environment variable required).

The ``_quantx_py`` pybind11 extension is built only when ``QX_WHEEL_PYBIND=1``. It is off
by default because it is the one artifact that depends on the Python version, and
cibuildwheel calls this hook once per CPython version while the CMake build directory is
shared between those calls: the cache keeps the first interpreter it saw, so every later
wheel would ship an extension compiled for the wrong Python. The ctypes library, which is
what ``backend_info()["native"]`` reports, does not depend on the Python version at all.

Usage::

    python scripts/build_native_for_wheel.py <dest_pkg_dir>

cibuildwheel before-build (cibuildwheel runs from the repo root, so ``{project}`` is the
repo root and ``{package}`` is ``python/``)::

    python {project}/scripts/build_native_for_wheel.py {package}/quantx_engine
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile


# ── helpers ──────────────────────────────────────────────────────────────────

def _find_cmake() -> str:
    """Return the first cmake / cmake3 executable on PATH."""
    for cmd in ("cmake", "cmake3"):
        if shutil.which(cmd):
            return cmd
    raise RuntimeError(
        "cmake not found on PATH. "
        "Install CMake >= 3.20 before building the native wheel."
    )


def _platform_lib_names() -> list[str]:
    """Return the expected shared-library file names for the current platform."""
    if sys.platform == "win32":
        return ["quantx.dll"]
    if sys.platform == "darwin":
        return ["libquantx.dylib"]
    return ["libquantx.so"]


def _pybind11_available() -> bool:
    """Return True if pybind11 is importable (so we can build _quantx_py)."""
    try:
        import pybind11  # noqa: F401
        return True
    except ImportError:
        return False


def _find_pybind_ext(build_dir: str) -> str | None:
    """Walk build_dir and return the path of the first _quantx_py.*.so/pyd."""
    for root, _dirs, files in os.walk(build_dir):
        for f in files:
            if f.startswith("_quantx_py") and any(
                f.endswith(ext) for ext in (".so", ".pyd", ".dll")
            ):
                return os.path.join(root, f)
    return None


def _find_built_lib(build_dir: str) -> str:
    """Walk build_dir and return the path of the first matching library."""
    lib_names = _platform_lib_names()
    search_dirs = [
        os.path.join(build_dir, "Release"),
        os.path.join(build_dir, "Debug"),
        build_dir,
    ]
    for d in search_dirs:
        for name in lib_names:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p

    # Exhaustive search as a fallback (handles unexpected CMake output dirs)
    for root, _dirs, files in os.walk(build_dir):
        for f in files:
            if f in lib_names:
                return os.path.join(root, f)

    # List everything .so/.dll/.dylib for diagnostics
    found = []
    for root, _dirs, files in os.walk(build_dir):
        for f in files:
            if any(f.endswith(ext) for ext in (".so", ".dll", ".dylib")):
                found.append(os.path.join(root, f))
    msg = "Built native library not found.\n"
    if found:
        msg += "Libraries present in build tree:\n" + "\n".join(f"  {p}" for p in found)
    else:
        msg += "No shared libraries found anywhere in the build tree."
    raise FileNotFoundError(msg)


# ── main ─────────────────────────────────────────────────────────────────────

def build_and_install(dest_dir: str) -> None:
    """Run CMake, build ``qx_capi`` (+ optionally ``_quantx_py``), copy to *dest_dir*."""
    dest_dir = os.path.abspath(dest_dir)

    # The repo root is one level above the scripts/ directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root   = os.path.dirname(script_dir)

    print(f"[build_native_for_wheel] Repo root : {repo_root}")
    print(f"[build_native_for_wheel] Dest dir  : {dest_dir}")

    cmake        = _find_cmake()
    # macOS: cibuildwheel builds arm64 and x86_64 wheels on one host and says which through
    # ARCHFLAGS ("-arch x86_64"), which CMake does not read -- so the x86_64 wheel would carry
    # an arm64 dylib and fail delocate's --require-archs. Pass the arch through explicitly,
    # and give each arch its own build dir so one cache never serves the other.
    osx_archs = []
    if sys.platform == "darwin":
        flags = os.environ.get("ARCHFLAGS", "").split()
        osx_archs = [flags[i + 1] for i, f in enumerate(flags)
                     if f == "-arch" and i + 1 < len(flags)]
    suffix    = ("-" + "-".join(osx_archs)) if osx_archs else ""
    build_dir = os.path.join(tempfile.gettempdir(), "qx_wheel_build" + suffix)
    os.makedirs(build_dir, exist_ok=True)

    n_jobs       = str(os.cpu_count() or 4)
    has_pybind11 = os.environ.get("QX_WHEEL_PYBIND") == "1" and _pybind11_available()

    # ── Configure ────────────────────────────────────────────────────────────
    cfg_cmd = [
        cmake, "-B", build_dir, "-S", repo_root,
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DQX_ENABLE_PYBIND={'ON' if has_pybind11 else 'OFF'}",
        "-DQX_ENABLE_MPI=OFF",
        "-DQX_ENABLE_CUDA=OFF",
    ]
    # A Visual Studio generator picks its target platform itself, and a 32-bit quantx.dll
    # cannot be loaded by 64-bit Python. Say x64 explicitly -- unless the caller chose the
    # generator or platform, where -A would be wrong (Ninja rejects it outright).
    if (sys.platform == "win32" and sys.maxsize > 2**32
            and "CMAKE_GENERATOR" not in os.environ
            and "CMAKE_GENERATOR_PLATFORM" not in os.environ):
        cfg_cmd += ["-A", "x64"]
    if osx_archs:
        cfg_cmd += ["-DCMAKE_OSX_ARCHITECTURES=" + ";".join(osx_archs)]
    # Link the MSVC C++ runtime statically. Otherwise delvewheel vendors whatever
    # msvcp140.dll it finds, and on the CI image that is OLDER (14.40) than the toolset
    # that built quantx.dll (14.51): Microsoft supports only runtime >= toolset, and the
    # vendored copy is renamed, so it wins even where a newer runtime is installed. The
    # OpenMP runtime (vcomp140.dll) has no static form and is still vendored.
    if sys.platform == "win32":
        cfg_cmd += ["-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
                    "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW"]
    print("[build_native_for_wheel] Configure:", " ".join(cfg_cmd))
    subprocess.run(cfg_cmd, check=True)

    # ── Build targets ────────────────────────────────────────────────────────
    targets = ["qx_capi"]
    if has_pybind11:
        targets.append("_quantx_py")
        print("[build_native_for_wheel] pybind11 found -- also building _quantx_py")

    for target in targets:
        build_cmd = [
            cmake, "--build", build_dir,
            "--target", target,
            "--config", "Release",
            "--parallel", n_jobs,
        ]
        print("[build_native_for_wheel] Build:", " ".join(build_cmd))
        subprocess.run(build_cmd, check=True)

    # ── Copy into Python package directory ───────────────────────────────────
    os.makedirs(dest_dir, exist_ok=True)

    # ctypes shared library (qx_capi → quantx.so/.dll/.dylib)
    src = _find_built_lib(build_dir)
    dst = os.path.join(dest_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    print(f"[build_native_for_wheel] Installed: {src} -> {dst}")

    # pybind11 extension module (_quantx_py.*.so / _quantx_py.pyd)
    if has_pybind11:
        ext = _find_pybind_ext(build_dir)
        if ext:
            edst = os.path.join(dest_dir, os.path.basename(ext))
            shutil.copy2(ext, edst)
            print(f"[build_native_for_wheel] Installed: {ext} -> {edst}")
        else:
            print("[build_native_for_wheel] WARNING: _quantx_py extension not found "
                  "after build -- skipping (ctypes path will be used).")


def main() -> None:
    # This script prints paths and ran fine on a Chinese-locale dev box (cp936 can encode
    # the arrow it used to print), then died on the Windows CI runner AFTER a successful
    # 12-minute build: cp1252 cannot encode U+2192, so print() raised UnicodeEncodeError
    # and the hook exited 1. Messages are ASCII now, but paths need not be (a non-ASCII
    # user name), so never let the console encoding fail the build.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    if len(sys.argv) < 2:
        print("Usage: build_native_for_wheel.py <dest_pkg_dir>", file=sys.stderr)
        sys.exit(1)
    build_and_install(sys.argv[1])
    print("[build_native_for_wheel] Done.")


if __name__ == "__main__":
    main()
