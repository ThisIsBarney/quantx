import ctypes
import json
import os
from ctypes import c_char_p, c_void_p


class Engine:
  def __init__(self, lib_path=None):
    self._lib_path = lib_path or self._default_library_path()
    self._lib = ctypes.CDLL(self._lib_path)
    self._lib.qx_run.argtypes = [c_char_p, c_char_p]
    self._lib.qx_run.restype = c_void_p
    self._lib.qx_explain_backend_choice.argtypes = [c_char_p, c_char_p]
    self._lib.qx_explain_backend_choice.restype = c_void_p
    self._lib.qx_free_string.argtypes = [c_void_p]
    self._lib.qx_free_string.restype = None

  def run(self, problem: dict, options: dict | None = None) -> dict:
    options = options or {}
    return self._call(self._lib.qx_run, problem, options)

  def explain(self, problem: dict, options: dict | None = None) -> dict:
    options = options or {}
    return self._call(self._lib.qx_explain_backend_choice, problem, options)

  def _call(self, fn, problem: dict, options: dict) -> dict:
    p = json.dumps(problem).encode("utf-8")
    o = json.dumps(options).encode("utf-8")
    ptr = fn(p, o)
    if not ptr:
      raise RuntimeError("native call returned null pointer")
    try:
      raw = ctypes.cast(ptr, c_char_p).value.decode("utf-8")
      return json.loads(raw)
    finally:
      self._lib.qx_free_string(ptr)

  @staticmethod
  def _default_library_path():
    cwd = os.getcwd()
    if os.name == "nt":
      candidates = [
          os.path.join(cwd, "build", "Debug", "quantx.dll"),
          os.path.join(cwd, "build-vs", "Debug", "quantx.dll"),
      ]
      for p in candidates:
        if os.path.exists(p):
          return p
      return candidates[0]
    if os.uname().sysname == "Darwin":
      return os.path.join(cwd, "build", "libquantx.dylib")
    return os.path.join(cwd, "build", "libquantx.so")
