# Releasing quantx-engine

One workflow publishes to PyPI: `.github/workflows/build-wheels.yml`. **Pushing a `v*` tag
is a PyPI release.** The tag builds the platform wheels and the sdist, smoke-tests them, and
uploads them through trusted publishing. An upload cannot be undone, and PyPI never
accepts the same version twice.

## One-time setup (maintainer)

**GitHub** (done 2026-10-16):

- the repository is public;
- the environment `pypi` exists (Settings → Environments). Its required reviewer is the
  maintainer, so every upload waits for an explicit approval, and it deploys `v*` tags
  only;
- a ruleset on `master` requires the `DCO` check and forbids force-pushes and deletion;
  repository admins can bypass it.

**PyPI**, once, by the maintainer:

1. Create an account at https://pypi.org/account/register/ and confirm the email.
2. Turn on two-factor authentication (Account settings → Two factor authentication):
   an authenticator app or a security key. PyPI requires it to publish. Keep the
   recovery codes.
3. Add a *pending* trusted publisher at https://pypi.org/manage/account/publishing/,
   on the GitHub tab:

   | Field | Value |
   |---|---|
   | PyPI Project Name | `quantx-engine` |
   | Owner | `ThisIsBarney` |
   | Repository name | `quantx` |
   | Workflow name | `build-wheels.yml` |
   | Environment name | `pypi` |

   The workflow then uploads with a short-lived token from GitHub; no API token is
   stored anywhere. The project appears on PyPI with the first upload. A pending
   publisher does not reserve the name, so do not leave it pending for long.

## Before the first upload

- **CI must run, and pass, on `master`.** While the repository was private, Actions
  did not start jobs (a billing problem). Check the latest runs on `master` before
  tagging.

The repository's old `v0.2.0` tag, from an internal milestone that published nothing,
was deleted on 2026-10-16 (it pointed at a3c657e). A release tag must be new, because a
tag push publishes.

## Cutting a release

1. **Set the version** in `python/quantx_engine/__init__.py` (`__version__ = "X.Y.Z"`). It
   is the only place the version is written: the wheel metadata, the native engine's
   `qx_version()`, and the docs all read it. Do not add `version =` to
   `python/pyproject.toml`. The version is declared dynamic there, so a static one is a
   build error.
2. Add a `## [X.Y.Z]` section to `CHANGELOG.md`, or, for 0.1.0, replace "unreleased" in
   its heading with the release date. `release.yml` copies that section into the GitHub
   Release.
3. Merge to `master` and wait for CI to pass. Every push to `master` builds all the wheels
   without publishing, so by now the release build has already run once.
4. Tag and push:
   ```
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
   `build-wheels.yml` builds and smoke-tests, which takes about half an hour. The
   upload then waits for the `pypi` environment's approval:
   1. Open Actions → "Build & publish wheels" → the run for the tag.
   2. Click **Review deployments**, tick `pypi`, then **Approve and deploy**.

   The upload refuses to run when the tag and the built files name different
   versions, for example `v0.2.0` pushed on a tree that still says 0.1.0.
5. Check the release: https://pypi.org/project/quantx-engine/ shows it. Then, on one of
   the platforms with a native wheel (Linux x86_64, macOS, Windows AMD64; see "What
   ships"), in a fresh virtual environment, `pip install quantx-engine==X.Y.Z` followed
   by `python -c "import quantx_engine; print(quantx_engine.backend_info())"` prints
   `'native': True`. Elsewhere pip builds the pure-Python sdist, and `False` is correct.
6. Optional: create the GitHub Release by hand. Go to Actions → Release → Run workflow and
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
