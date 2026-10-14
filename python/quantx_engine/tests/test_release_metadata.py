"""Release metadata that no build step checks by itself."""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]


def test_license_copies_match():
    # The wheel is built from python/, so the license it ships is python/LICENSE; the
    # repository's is LICENSE at the root. Two copies of one text must not drift.
    root, pkg = ROOT / "LICENSE", ROOT / "python" / "LICENSE"
    if not (root.exists() and pkg.exists()):
        pytest.skip("not running from a source checkout")
    assert root.read_text(encoding="utf-8") == pkg.read_text(encoding="utf-8")


def test_changelog_has_an_entry_for_this_version():
    # RELEASING.md step 2; release.yml lifts the release notes from this section.
    from quantx_engine import __version__
    changelog = ROOT / "CHANGELOG.md"
    if not changelog.exists():
        pytest.skip("not running from a source checkout")
    assert f"## [{__version__}]" in changelog.read_text(encoding="utf-8")
