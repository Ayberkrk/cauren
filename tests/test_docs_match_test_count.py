"""Keep the "N tests" figures in README.md/CONTRIBUTING.md honest.

Deliberately dependency-free: it uses `ast.parse` rather than importing
anything, so it counts test functions the same way on every CI leg
regardless of which optional extras (pandas/torch/PyYAML) happen to be
installed there. If someone adds or removes a test without updating the
docs, this fails with the actual current numbers rather than letting the
docs silently drift.
"""

from __future__ import annotations

import ast
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


def _module_level_importorskip(tree: ast.Module) -> bool:
    """Does this file's top level call pytest.importorskip(...)?

    Those are exactly the modules that get skipped as a whole (not
    individual tests) when an optional dependency (pandas/torch/PyYAML)
    isn't installed -- see tests/test_bridge_scour_risk_mapping.py,
    tests/test_checkpoint_selection.py, and
    tests/test_training_feedback_ingestion.py for the pattern.
    """
    for node in tree.body:
        if isinstance(node, ast.Assign):
            call = node.value
        elif isinstance(node, ast.Expr):
            call = node.value
        else:
            continue
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name == "importorskip":
            return True
    return False


def _count_test_functions(tree: ast.Module) -> int:
    return sum(
        1
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
    )


def _collect_counts() -> tuple[int, int]:
    """Returns (minimal_dependency_count, total_count)."""
    total = 0
    minimal = 0
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count = _count_test_functions(tree)
        total += count
        if not _module_level_importorskip(tree):
            minimal += count
    return minimal, total


def test_readme_and_contributing_report_the_actual_test_counts():
    minimal, total = _collect_counts()

    for doc_name in ("README.md", "CONTRIBUTING.md"):
        text = (REPO_ROOT / doc_name).read_text(encoding="utf-8")
        assert f"{minimal} tests" in text, (
            f"{doc_name} doesn't mention '{minimal} tests' (the count with only the minimal "
            f"dependency set installed -- pytest/fastapi/pydantic/httpx). Actual: {minimal} "
            f"minimal, {total} total across tests/. Update {doc_name}'s wording."
        )
        assert f"{total} total" in text, (
            f"{doc_name} doesn't mention '{total} total' (the full count once pandas/torch/"
            f"PyYAML are also installed). Actual: {minimal} minimal, {total} total. "
            f"Update {doc_name}'s wording."
        )
