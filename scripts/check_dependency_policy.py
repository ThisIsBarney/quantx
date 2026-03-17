#!/usr/bin/env python3
"""Check dependency licenses against closed-source-friendly policy."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ALLOW = {"MIT", "BSD-2-Clause", "BSD-3-Clause", "Apache-2.0", "Zlib", "BSL-1.0", "MPL-2.0"}


def main() -> int:
  root = Path(__file__).resolve().parents[1]
  dep_doc = root / "docs" / "DEPENDENCIES.md"
  text = dep_doc.read_text(encoding="utf-8")

  licenses = re.findall(r"- License:\s*([^\n]+)", text)
  if not licenses:
    print("No dependency licenses found in docs/DEPENDENCIES.md", file=sys.stderr)
    return 1

  bad = [x.strip() for x in licenses if x.strip() not in ALLOW]
  if bad:
    print("Dependency policy violation: disallowed license(s):", ", ".join(sorted(set(bad))), file=sys.stderr)
    print("Allowed:", ", ".join(sorted(ALLOW)), file=sys.stderr)
    return 2

  print("Dependency license policy check passed.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
