"""Core production code must not read coordination through the `_conn` seam.

Core reaches into the private Pro coordination manager for read-only
analytics. It previously did so through `mgr._conn` -- the shared WRITE
connection -- which let a reader interleave with an open coordination write
transaction. `get_read_connection()` returns a dedicated read-only
connection instead.

This guard is Core-side and needs no Pro installation: it is pure AST.
"""

from __future__ import annotations

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "omega"
MANAGER_NAMES = {"mgr", "manager", "_mgr"}


def _manager_conn_uses(path: pathlib.Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:  # pragma: no cover
        return []
    hits: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr == "_conn"):
            continue
        base = node.value
        if isinstance(base, ast.Name) and base.id in MANAGER_NAMES:
            hits.append(f"{path.relative_to(SRC)}:{node.lineno} {base.id}._conn")
    return hits


def test_core_reads_coordination_through_the_dedicated_read_api():
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        offenders.extend(_manager_conn_uses(path))
    assert not offenders, (
        "Core must call mgr.get_read_connection() for coordination reads; "
        f"`_conn` consumers found: {offenders}"
    )
