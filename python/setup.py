"""Tag the wheel by what it actually carries.

All metadata lives in pyproject.toml. The package ships its native engine as a prebuilt
library (quantx.dll / libquantx.so / libquantx.dylib) loaded through ctypes: not a
setuptools extension module, and not tied to any Python version or ABI. So:

- With the library in the package (cibuildwheel's before-build hook puts it there), the
  wheel is py3-none-<platform>. One wheel per platform serves every CPython 3, including
  versions released after it was built, and cibuildwheel builds it once and tests that
  same wheel on each Python it lists. The default tagging would have given py3-none-any
  (a wheel claiming to run anywhere while carrying one platform's binary; cibuildwheel
  refuses to build it).
- Without the library (pip building the sdist, on a platform we ship no wheel for), the
  wheel is py3-none-any: pure-Python mode, honestly tagged. It used to come out
  platform-tagged, claiming a native engine it did not contain.
"""
import os

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.dist import Distribution

_PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantx_engine")
_HAS_NATIVE = any(os.path.exists(os.path.join(_PKG, name))
                  for name in ("quantx.dll", "libquantx.so", "libquantx.dylib"))
# The optional pybind11 module (QX_WHEEL_PYBIND=1) IS built against one CPython ABI; a
# wheel carrying it keeps setuptools' cpXY-cpXY tags.
_HAS_PYEXT = os.path.isdir(_PKG) and any(f.startswith("_quantx_py") for f in os.listdir(_PKG))


class _Distribution(Distribution):
    def has_ext_modules(self):
        # True makes the wheel non-pure AND installs the package to platlib -- the wheel's
        # root, beside its binary. Setting only root_is_pure left the package, library
        # included, under .data/purelib/.
        return _HAS_NATIVE or _HAS_PYEXT


class NativeLibWheel(bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()
        if _HAS_NATIVE and not _HAS_PYEXT:
            return "py3", "none", plat
        return python, abi, plat


setup(distclass=_Distribution, cmdclass={"bdist_wheel": NativeLibWheel})
