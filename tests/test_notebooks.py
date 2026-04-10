"""Execute every chapter notebook end-to-end as a regression test.

Run everything:

    pytest tests/test_notebooks.py

Run a single chapter:

    pytest tests/test_notebooks.py -k chapter11

Skip notebooks known to be slow or to require external resources by adding
their relative path (POSIX style) to ``SKIP_NOTEBOOKS`` below.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per-cell execution timeout in seconds. Some forecasting notebooks train
# deep models or download large artifacts on the first run.
CELL_TIMEOUT = 1800

# Notebooks to skip. Use paths relative to the repo root, POSIX style.
# Populate this list as you discover notebooks that need GPU, paid APIs,
# or are otherwise impractical to run in CI.
SKIP_NOTEBOOKS: set[str] = set()


def _discover_notebooks() -> list[Path]:
    return sorted(REPO_ROOT.glob("chapter*/*.ipynb"))


NOTEBOOKS = _discover_notebooks()


@pytest.mark.parametrize(
    "notebook_path",
    NOTEBOOKS,
    ids=[p.relative_to(REPO_ROOT).as_posix() for p in NOTEBOOKS],
)
def test_notebook_executes(notebook_path: Path) -> None:
    rel = notebook_path.relative_to(REPO_ROOT).as_posix()
    if rel in SKIP_NOTEBOOKS:
        pytest.skip(f"{rel} is in SKIP_NOTEBOOKS")

    nb = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=CELL_TIMEOUT,
        kernel_name="python3",
        resources={"metadata": {"path": str(notebook_path.parent)}},
    )

    try:
        client.execute()
    except CellExecutionError as exc:
        pytest.fail(f"{rel} failed to execute:\n{exc}")
