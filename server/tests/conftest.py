"""pytest configuration — add python/ SDK to sys.path so tests import quantx_engine."""
import os
import sys

# Make `python/` importable from any working directory so the mock-based
# tests work in a clean checkout without installing quantx-engine separately.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
