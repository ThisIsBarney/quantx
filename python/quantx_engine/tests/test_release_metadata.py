"""Release metadata that no build step checks by itself."""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("name", ["LICENSE", "THIRD_PARTY_NOTICES.md"])
def test_license_copies_match(name):
    # The wheel is built from python/, so the files it ships are python/LICENSE and
    # python/THIRD_PARTY_NOTICES.md; the repository's are at the root. Two copies of one
    # text must not drift.
    root, pkg = ROOT / name, ROOT / "python" / name
    if not (root.exists() and pkg.exists()):
        pytest.skip("not running from a source checkout")
    assert root.read_text(encoding="utf-8") == pkg.read_text(encoding="utf-8")


@pytest.mark.parametrize("name, first_line", [
    ("GPL-3.0.txt", "GNU GENERAL PUBLIC LICENSE"),
    ("GCC-exception-3.1.txt", "GCC RUNTIME LIBRARY EXCEPTION"),
])
def test_gpl_texts_for_the_bundled_libgomp_ship(name, first_line):
    # The Linux wheel bundles libgomp, so the GPL and its exception must ship too:
    # pyproject's license-files picks up python/licenses/*.txt.
    text = ROOT / "python" / "licenses" / name
    if not text.exists():
        pytest.skip("not running from a source checkout")
    assert text.read_text(encoding="utf-8").lstrip().startswith(first_line)
    pyproject = (ROOT / "python" / "pyproject.toml").read_text(encoding="utf-8")
    assert '"licenses/*.txt"' in pyproject


def test_changelog_has_an_entry_for_this_version():
    # RELEASING.md step 2; release.yml lifts the release notes from this section.
    from quantx_engine import __version__
    changelog = ROOT / "CHANGELOG.md"
    if not changelog.exists():
        pytest.skip("not running from a source checkout")
    assert f"## [{__version__}]" in changelog.read_text(encoding="utf-8")
