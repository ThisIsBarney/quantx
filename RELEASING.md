# Releasing quantx-engine to PyPI

## Prerequisites

- Maintainer access to the GitHub repo
- PyPI and TestPyPI environments configured for OIDC trusted publishing
  (Settings → Environments → `pypi` / `testpypi`, publisher = `ThisIsBarney/quantx`, workflow = `publish-pypi.yml`)

## Cutting a Release

1. **Bump the version** in `python/pyproject.toml`:
   ```
   version = "0.2.0"
   ```

2. **Commit and push to master**:
   ```
   git add python/pyproject.toml
   git commit -m "chore: bump version to 0.2.0"
   git push origin master
   ```
   This triggers `publish-pypi.yml` → `publish-testpypi` job → wheel uploaded to TestPyPI automatically.

3. **Verify on TestPyPI** (optional):
   ```
   pip install -i https://test.pypi.org/simple/ quantx-engine==0.2.0
   ```

4. **Tag the release**:
   ```
   git tag v0.2.0
   git push origin v0.2.0
   ```
   This triggers `publish-pypi.yml` → `publish-pypi` job → wheel uploaded to PyPI.

5. **Create a GitHub Release** (optional but recommended):
   ```
   gh release create v0.2.0 --title "v0.2.0" --notes "..."
   ```

## Installing from PyPI

```bash
pip install quantx-engine
```

The Python SDK is a ctypes wrapper; the native engine binary (`quantx.dll` /
`libquantx.so` / `libquantx.dylib`) must be built separately and its path
configured via the `QUANTX_LIB_PATH` environment variable, or placed in one of
the default search locations (`build/Debug/`).

## Local wheel build

```powershell
.\scripts\build_wheel.ps1
```
