"""Add Colab/Binder badges and pip-install cells to QuantX tutorial notebooks."""
from __future__ import annotations

import json
import os

REPO = "ThisIsBarney/quantx"
NOTEBOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "notebooks")
BRANCH = "master"

NOTEBOOKS = [
    "01_getting_started.ipynb",
    "02_vqe_chemistry.ipynb",
    "03_quantum_error_correction.ipynb",
    "04_quantum_networking.ipynb",
    "05_many_body_mps.ipynb",
    "06_qec_research.ipynb",
    "07_bosonic_qec.ipynb",
]

INSTALL_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Install quantx-engine (needed in Colab / Binder)\n",
        "# Skipped automatically if already installed in the local environment.\n",
        "try:\n",
        "    import quantx_engine  # noqa: F401\n",
        "except ImportError:\n",
        "    import subprocess, sys\n",
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',\n",
        "                           'quantx-engine'])\n",
        "    import quantx_engine  # noqa: F401\n",
        "print('quantx_engine version:', quantx_engine.__version__)",
    ],
}


def make_badges(nb_name: str) -> str:
    nb_path = f"notebooks/{nb_name}"
    binder_url = f"https://mybinder.org/v2/gh/{REPO}/{BRANCH}?filepath={nb_path}"
    colab_url = (
        f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{nb_path}"
    )
    return (
        f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"({colab_url})  "
        f"[![Launch Binder](https://mybinder.org/badge_logo.svg)]({binder_url})\n\n"
    )


def process_notebook(path: str, nb_name: str) -> bool:
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    changed = False

    # 1. Add badges to first markdown cell (after the title line)
    if cells and cells[0]["cell_type"] == "markdown":
        source_text = "".join(cells[0]["source"])
        if "colab.research.google.com" not in source_text:
            badge_line = make_badges(nb_name)
            src = cells[0]["source"]
            # Find first blank line after title or insert after first line
            if src and src[0].startswith("#"):
                cells[0]["source"] = [src[0], "\n", badge_line] + src[1:]
            else:
                cells[0]["source"] = [badge_line] + src
            changed = True
            print(f"  + badges → {nb_name}")

    # 2. Insert install code cell after first markdown cell
    has_install = any(
        "pip install" in "".join(c.get("source", [])) and c["cell_type"] == "code"
        for c in cells[:4]
    )
    if not has_install:
        insert_pos = 1 if cells and cells[0]["cell_type"] == "markdown" else 0
        cells.insert(insert_pos, dict(INSTALL_CELL))  # shallow copy
        changed = True
        print(f"  + install cell → {nb_name}")

    if changed:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)

    return changed


if __name__ == "__main__":
    n_mod = 0
    for nb_name in NOTEBOOKS:
        path = os.path.normpath(os.path.join(NOTEBOOKS_DIR, nb_name))
        if not os.path.exists(path):
            print(f"SKIP (not found): {nb_name}")
            continue
        if process_notebook(path, nb_name):
            n_mod += 1
    print(f"\nDone: modified {n_mod}/{len(NOTEBOOKS)} notebooks.")
