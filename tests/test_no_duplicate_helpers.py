"""Static-analysis guard for the B5 unification.

B5 collapsed several previously-duplicated helpers (kabsch_rmsd,
adjacent_heavy_clash_metrics, pdb_id_from_path, AA3_TO_1, save_pdb)
into a single canonical definition. This test greps the source tree
to make sure nobody re-introduces a second copy, which would silently
desynchronise fixes and features.

The test is intentionally filesystem-based (rather than importing
the modules) so it fails *before* import-time errors, with a clear
error message that points at the offending file and line.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


# Canonical locations (relative to the repo root) for each symbol.
# A symbol is allowed to be defined exactly once *in production code*
# (tests/ are excluded). `save_pdb` is special: it has one canonical
# module-level definition in qtf.utils.pdb *and* a thin delegating
# method on QuantumBiophysicsFolder; both are intentional and are
# whitelisted below.
CANONICAL = {
    "kabsch_rmsd": ["qtf/analysis/stability.py"],
    "adjacent_heavy_clash_metrics": ["qtf/utils/workflow.py"],
    "nonlocal_heavy_clash_metrics": ["qtf/utils/workflow.py"],
    "pdb_id_from_path": ["qtf/utils/workflow.py"],
    "AA3_TO_1": ["qtf/utils/workflow.py"],
    # save_pdb has one module-level definition and one method wrapper.
    "save_pdb_module": ["qtf/utils/pdb.py"],
    "save_pdb_method": ["qtf/core/folder.py"],
}


def _walk_python_files() -> list[Path]:
    """Yield every .py file under the repo root, excluding tests/,
    build artefacts, and caches.

    B5 targets duplicated helper implementations in the importable
    package surface. Repository-root utility scripts (for example,
    cluster launchers) are intentionally out of scope.
    """
    skip_dirs = {
        "tests",
        ".git",
        "__pycache__",
        "build",
        "dist",
        ".eggs",
        "qtf.egg-info",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT / "qtf"):
        # Prune the skip dirs in-place so os.walk doesn't descend.
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(Path(dirpath) / fn)
    return out


def _all_definitions(name: str) -> list[tuple[Path, int, str]]:
    """Find every line that defines ``name``.

    Matches:
      - ``def <name>(...):``  (function or method definition)
      - ``<name> = {``        (module-level dict literal for AA3_TO_1)
      - ``<name> = (``        (tuple-style assignment, used by some helpers)

    Returns a list of (path, line_number, line_text) tuples.
    """
    pat_def = re.compile(rf"^def\s+{re.escape(name)}\s*\(")
    pat_dict = re.compile(rf"^{re.escape(name)}\s*=\s*\{{")
    out: list[tuple[Path, int, str]] = []
    for path in _walk_python_files():
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            if pat_def.match(stripped) or pat_dict.match(stripped):
                out.append((path, i, line.rstrip()))
    return out


def _format_definitions(defs: list[tuple[Path, int, str]]) -> str:
    rel = [(p.relative_to(REPO_ROOT), i, t) for p, i, t in defs]
    return "\n".join(f"  {p}:{i}: {t}" for p, i, t in rel)


@pytest.mark.parametrize(
    "name,canonical",
    [
        ("kabsch_rmsd", CANONICAL["kabsch_rmsd"]),
        ("adjacent_heavy_clash_metrics", CANONICAL["adjacent_heavy_clash_metrics"]),
        ("nonlocal_heavy_clash_metrics", CANONICAL["nonlocal_heavy_clash_metrics"]),
        ("pdb_id_from_path", CANONICAL["pdb_id_from_path"]),
        ("AA3_TO_1", CANONICAL["AA3_TO_1"]),
    ],
)
def test_single_definition(name, canonical):
    """The non-`save_pdb` helpers in B5 must have exactly one
    definition, in the canonical file. This prevents future
    contributors from re-introducing a copy in a script (B5's
    original bug) and silently desynchronising it from the canonical
    implementation."""
    defs = _all_definitions(name)
    assert len(defs) == 1, (
        f"Symbol {name!r} must be defined exactly once, but found "
        f"{len(defs)} definitions:\n{_format_definitions(defs)}"
    )
    canonical_rel = defs[0][0].relative_to(REPO_ROOT)
    assert str(canonical_rel) in canonical, (
        f"Symbol {name!r} must be defined in {canonical}, but found at "
        f"{canonical_rel}:{defs[0][1]}"
    )


def test_save_pdb_canonical_free_function():
    """The module-level `save_pdb` lives only in qtf.utils.pdb."""
    defs = _all_definitions("save_pdb")
    # `save_pdb` is allowed exactly one module-level definition (the
    # free function in qtf.utils.pdb) AND one method definition on
    # QuantumBiophysicsFolder (which delegates to the free function).
    # Both are checked separately below.
    module_level = [
        (p, i, t) for p, i, t in defs
        if not t.lstrip().startswith("def save_pdb(self,")
    ]
    assert len(module_level) == 1, (
        "save_pdb must have exactly one module-level definition, but "
        f"found {len(module_level)}:\n{_format_definitions(module_level)}"
    )
    canonical_rel = module_level[0][0].relative_to(REPO_ROOT)
    assert str(canonical_rel) == "qtf/utils/pdb.py", (
        f"Module-level save_pdb must live in qtf/utils/pdb.py, found "
        f"at {canonical_rel}:{module_level[0][1]}"
    )


def test_save_pdb_folder_method_is_a_wrapper():
    """The folder-method `save_pdb` must exist and must be a thin
    wrapper around the canonical free function (B5: no re-implementation)."""
    defs = _all_definitions("save_pdb")
    methods = [
        (p, i, t) for p, i, t in defs
        if t.lstrip().startswith("def save_pdb(self,")
    ]
    assert len(methods) == 1, (
        "QuantumBiophysicsFolder.save_pdb must be defined exactly once; "
        f"found {len(methods)}:\n{_format_definitions(methods)}"
    )
    method_file = methods[0][0]
    method_line = methods[0][1]
    assert method_file.relative_to(REPO_ROOT) == Path("qtf/core/folder.py")
    # Read a window around the method to confirm it delegates to
    # qtf.utils.pdb.save_pdb (i.e. it is a thin wrapper, not a
    # re-implementation).
    text = method_file.read_text()
    lines = text.splitlines()
    window_start = max(0, method_line - 1)
    window_end = min(len(lines), method_line + 40)
    window = "\n".join(lines[window_start:window_end])
    assert "qtf.utils.pdb" in window or "from qtf.utils.pdb import save_pdb" in window, (
        "QuantumBiophysicsFolder.save_pdb is expected to be a thin "
        "wrapper around qtf.utils.pdb.save_pdb (B5). The window after "
        f"the definition does not mention qtf.utils.pdb:\n{window}"
    )


def test_no_duplicate_helpers_summary():
    """A single end-to-end summary that fails if ANY B5 symbol is
    defined more than the allowed number of times. Useful when a
    contributor adds a new helper and forgets to update this test
    — they can run only this test for a quick scan."""
    violations: list[str] = []

    for name in ("kabsch_rmsd", "adjacent_heavy_clash_metrics",
                 "nonlocal_heavy_clash_metrics", "pdb_id_from_path",
                 "AA3_TO_1"):
        defs = _all_definitions(name)
        if len(defs) != 1:
            violations.append(
                f"{name!r}: expected exactly 1 definition, "
                f"found {len(defs)}\n{_format_definitions(defs)}"
            )

    save_pdb_defs = _all_definitions("save_pdb")
    free = [d for d in save_pdb_defs if not d[2].lstrip().startswith("def save_pdb(self,")]
    method = [d for d in save_pdb_defs if d[2].lstrip().startswith("def save_pdb(self,")]
    if len(free) != 1:
        violations.append(
            "save_pdb (free function): expected exactly 1, "
            f"found {len(free)}\n{_format_definitions(free)}"
        )
    if len(method) != 1:
        violations.append(
            "save_pdb (method): expected exactly 1, "
            f"found {len(method)}\n{_format_definitions(method)}"
        )

    assert not violations, (
        "B5: duplicate helper definitions detected:\n\n"
        + "\n\n".join(violations)
    )


def test_helper_scan_scope_is_package_only():
    """The duplicate-helper guard should only scan importable package code."""
    for path in _walk_python_files():
        rel = path.relative_to(REPO_ROOT)
        assert rel.parts[0] == "qtf"
