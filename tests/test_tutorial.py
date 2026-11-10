"""Run the code cells of examples/tutorial.ipynb, so the tutorial keeps working as the API changes."""

import ast
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
pytest.importorskip("IPython")

TUTORIAL = Path(__file__).resolve().parent.parent / "examples" / "tutorial.ipynb"


def _run_cell(source: str, namespace: dict, name: str):
    """Run a cell like Jupyter does and return the value of its last line, if that is an expression."""
    tree = ast.parse(source, name)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    exec(compile(tree, name, "exec"), namespace)  # noqa: S102 (runs our own tutorial)
    if last is not None:
        return eval(compile(ast.Expression(last.value), name, "eval"), namespace)
    return None


def test_tutorial_code_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the tutorial writes muon_event.html
    nb = nbformat.read(TUTORIAL, as_version=4)
    namespace: dict = {}
    n_displays = 0
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        value = _run_cell(cell.source, namespace, f"tutorial cell {i}")
        html = value._repr_html_() if hasattr(value, "_repr_html_") else getattr(value, "data", None)
        if isinstance(html, str) and "TrackVis.mount" in html:
            n_displays += 1
    assert n_displays == 14
    assert (tmp_path / "muon_event.html").stat().st_size > 100_000
