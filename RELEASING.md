# Releasing quantx-engine

One workflow publishes to PyPI: `.github/workflows/build-wheels.yml`. **Pushing a `v*` tag
is a PyPI release.** The tag builds the platform wheels and the sdist, smoke-tests them, and
uploads them through trusted publishing. An upload cannot be undone, and PyPI never
accepts the same version twice.

## One-time setup (maintainer)

1. On PyPI, add a trusted publisher for the project `quantx-engine`: owner `ThisIsBarney`,
   repository `quantx`, workflow `build-wheels.yml`, environment `pypi`. A project that is
   not on PyPI yet takes a *pending* publisher with the same fields.
2. On GitHub, create the environment `pypi` (Settings → Environments). With required
   reviewers set there, every upload waits for an explicit approval.

## A stale tag to know about

The repository has an old tag `v0.2.0` from an internal milestone. It corresponds to no
published package (nothing was on PyPI before 0.1.0, and `CHANGELOG.md` lists the
internal milestones as history). It does not affect 0.1.0. Before releasing 0.2.0,
delete it (`git push --delete origin v0.2.0` and `git tag -d v0.2.0`) or skip to 0.3.0:
a tag push publishes, so the release tag must be new.

## Cutting a release

1. **Set the version** in `python/quantx_engine/__init__.py` (`__version__ = "X.Y.Z"`). It
   is the only place the version is written: the wheel metadata, the native engine's
   `qx_version()`, and the docs all read it. Do not add `version =` to
   `python/pyproject.toml`. The version is declared dynamic there, so a static one is a
   build error.
2. Add a `## [X.Y.Z]` section to `CHANGELOG.md`.
3. Merge to `master` and wait for CI to pass. Every push to `master` builds all the wheels
   without publishing, so by now the release build has already run once.
4. Tag and push:
   ```
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
   `build-wheels.yml` builds and smoke-tests. The upload then waits for the `pypi`
   environment's reviewers, if you set any, and runs.
5. Optional: create the GitHub Release by hand. Go to Actions → Release → Run workflow and
   set `publish=true`, `tag=vX.Y.Z`. It attaches the sdist and the CHANGELOG entry.

## What ships

| Platform | Wheel tag | Native engine |
|---|---|---|
| Linux x86_64 | `py3-none-manylinux_2_28_x86_64` | yes, with OpenMP (libgomp bundled) |
| macOS arm64 and x86_64, 10.15+ | `py3-none-macosx_…` | yes, without OpenMP (Apple Clang ships none) |
| Windows AMD64 | `py3-none-win_amd64` | yes; C++ runtime linked statically, OpenMP runtime bundled |
| anything else | built from the sdist: `py3-none-any` | no: pure-Python mode, `backend_info()["native"]` is `False` |

The engine is a shared library loaded through ctypes, independent of the Python version. So
each platform gets one `py3-none-<platform>` wheel, which serves every CPython 3.
cibuildwheel tests that one wheel on 3.10 through 3.13, and the smoke test adds 3.14. The
wheels carry no GPU support and no OpenBLAS.

## Local wheel build

```powershell
.\scripts\build_wheel.ps1
```
