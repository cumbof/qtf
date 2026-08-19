"""Recipe loading for the ``qtf fold-simulation`` command."""

from __future__ import annotations

import copy
import json
from importlib import resources
from pathlib import Path
from typing import Any, Mapping, Optional


def _yaml_module():
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("QTF recipes require PyYAML in the active environment.") from exc
    return yaml


def load_recipe_file(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load recipes from a YAML file."""

    yaml = _yaml_module()
    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Recipe file {source} must contain a mapping.")
    _validate_recipe_document(data, source)
    recipes = data.get("recipes")
    if not isinstance(recipes, Mapping):
        raise ValueError(f"Recipe file {source} must define a top-level `recipes` mapping.")
    return {str(name): copy.deepcopy(dict(recipe)) for name, recipe in recipes.items()}


def load_builtin_recipes() -> dict[str, dict[str, Any]]:
    """Load packaged QTF recipes."""

    recipe_root = resources.files("qtf.assets.recipes")
    recipes: dict[str, dict[str, Any]] = {}
    for path in sorted(recipe_root.iterdir(), key=lambda item: item.name):
        if path.name.startswith("_") or path.suffix not in {".yaml", ".yml"}:
            continue
        recipes.update(load_recipe_file(Path(path)))
    return recipes


def load_recipes(recipe_file: Optional[str | Path] = None) -> dict[str, dict[str, Any]]:
    """Load built-in recipes, optionally overridden by a user recipe file."""

    recipes = load_builtin_recipes()
    if recipe_file:
        recipes.update(load_recipe_file(recipe_file))
    return recipes


def resolve_recipe(name: str, recipe_file: Optional[str | Path] = None) -> dict[str, Any]:
    """Return a deep copy of the selected recipe."""

    recipes = load_recipes(recipe_file)
    if name not in recipes:
        available = ", ".join(sorted(recipes)) or "none"
        raise KeyError(f"Unknown recipe {name!r}. Available recipes: {available}")
    recipe = copy.deepcopy(recipes[name])
    recipe.setdefault("name", name)
    return recipe


def _validate_recipe_document(data: Mapping[str, Any], source: Path) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("QTF recipe validation requires jsonschema in the active environment.") from exc

    schema_text = resources.files("qtf.assets.recipes").joinpath("schema.json").read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda error: list(error.absolute_path))
    if not errors:
        return
    first = errors[0]
    location = ".".join(str(part) for part in first.absolute_path) or "<root>"
    raise ValueError(f"Invalid recipe file {source}: {location}: {first.message}")
