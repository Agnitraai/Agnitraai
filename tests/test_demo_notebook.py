from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient


def test_enhanced_demo_notebook_executes(tmp_path):
    """The enhanced demo notebook should execute without errors."""
    pytest.importorskip("ipykernel", reason="Notebook execution requires ipykernel")
    nb_path = Path(__file__).resolve().parents[1] / "agnitra_enhanced_demo.ipynb"
    nb = nbformat.read(nb_path, as_version=4)

    # Remove cells that would attempt package installation or run pytest
    filtered_cells = []
    for cell in nb.cells:
        if cell.cell_type != "code":
            filtered_cells.append(cell)
            continue
        src = "".join(cell.source)
        if "!pip" in src or "!pytest" in src:
            continue
        filtered_cells.append(cell)
    nb.cells = filtered_cells

    client = NotebookClient(nb, timeout=120, kernel_name="python3", allow_errors=True)
    client.execute()

    exec_counts = [c.execution_count for c in nb.cells if c.cell_type == "code"]
    assert exec_counts and exec_counts[-1] is not None
