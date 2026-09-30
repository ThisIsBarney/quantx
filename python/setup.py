"""Mark the wheel as platform-specific.

All metadata lives in pyproject.toml. This file exists for one reason: the package ships a
prebuilt native library (quantx.dll / libquantx.so / libquantx.dylib, loaded through ctypes)
as package data, and has no setuptools extension modules. Without this, setuptools tags the
wheel ``py3-none-any`` -- a wheel that claims to run anywhere while carrying one platform's
binary -- and cibuildwheel refuses to build it at all ("a pure Python wheel was generated").
"""
from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
