"""Portable path handling for serialized QTF artifacts."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


_EMBEDDED_ABSOLUTE_PATH = re.compile(
    r"(?<![:A-Za-z0-9_.-])/(?:[^\s\"'<>]+)"
)


def _portable_string(value: str, base: Path) -> str:
    if not any(character.isspace() for character in value) and os.path.isabs(os.path.expanduser(value)):
        return repo_relative_path(value, base)

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        path_token = token.rstrip(",:;)")
        suffix = token[len(path_token):]
        return repo_relative_path(path_token, base) + suffix

    return _EMBEDDED_ABSOLUTE_PATH.sub(replace, value)


def repository_root(start: str | Path | None = None) -> Path:
    """Return the active repository root, preferring the invocation directory."""

    override = os.environ.get("QTF_REPO_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    origin = Path(start or Path.cwd()).expanduser().resolve()
    if origin.is_file():
        origin = origin.parent
    for candidate in (origin, *origin.parents):
        if (candidate / ".git").exists():
            return candidate
    return Path.cwd().resolve()


def repo_relative_path(value: str | Path, root: str | Path | None = None) -> str:
    """Represent a filesystem path relative to the active repository root."""

    base = repository_root(root)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return Path(os.path.relpath(candidate.resolve(strict=False), base)).as_posix()


def relativize_absolute_paths(value: Any, root: str | Path | None = None) -> Any:
    """Recursively replace absolute filesystem strings with repo-relative ones."""

    base = repository_root(root)
    if isinstance(value, Path):
        return repo_relative_path(value, base)
    if isinstance(value, str):
        return _portable_string(value, base)
    if isinstance(value, dict):
        return {
            str(key): relativize_absolute_paths(item, base)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [relativize_absolute_paths(item, base) for item in value]
    if isinstance(value, tuple):
        return [relativize_absolute_paths(item, base) for item in value]
    return value


def portable_dataframe(frame, root: str | Path | None = None):
    """Return a copy whose absolute string cells are repository-relative."""

    base = repository_root(root)
    result = frame.copy()
    for column in result.columns:
        result[column] = result[column].map(
            lambda item: relativize_absolute_paths(item, base)
        )
    return result


def write_portable_csv(frame, path: str | Path, *, index: bool = False, **kwargs) -> None:
    """Write a dataframe without embedding absolute filesystem paths."""

    portable_dataframe(frame).to_csv(path, index=index, **kwargs)
