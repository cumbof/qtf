"""Shared PHEAT-backed structure metrics for QTF reports."""

from __future__ import annotations

from typing import Iterable


METRIC_ATOM_SETS = ("ca", "backbone", "all-heavy")
PRIMARY_RMSD_ATOM_SET = "all-heavy"
DEFAULT_RMSD_ALIGNMENT_ATOM_SET = "same-as-rmsd"


def normalize_metric_atom_sets(atom_sets: str | Iterable[str] | None = None) -> tuple[str, ...]:
    """Normalize and de-duplicate PHEAT atom-set names."""

    from pheat.metrics import normalize_rmsd_atom_set

    if atom_sets is None:
        values = METRIC_ATOM_SETS
    elif isinstance(atom_sets, str):
        values = [token.strip() for token in atom_sets.split(",") if token.strip()]
    else:
        values = [str(token).strip() for token in atom_sets if str(token).strip()]
    normalized = tuple(dict.fromkeys(normalize_rmsd_atom_set(value) for value in values))
    if not normalized:
        raise ValueError("At least one metric atom set is required.")
    return normalized


def normalize_rmsd_alignment_atom_set(value: str | None) -> str:
    """Normalize the configured RMSD alignment atom-set selector."""

    from pheat.metrics import SAME_AS_RMSD_ALIGNMENT, normalize_rmsd_atom_set

    normalized = str(value or SAME_AS_RMSD_ALIGNMENT).strip().lower().replace("_", "-")
    if normalized == SAME_AS_RMSD_ALIGNMENT:
        return SAME_AS_RMSD_ALIGNMENT
    return normalize_rmsd_atom_set(normalized)


def structure_metric_summary(
    reference,
    target,
    *,
    atom_sets: str | Iterable[str] = METRIC_ATOM_SETS,
    alignment_atom_set: str = DEFAULT_RMSD_ALIGNMENT_ATOM_SET,
) -> dict:
    """Return RMSD metrics for selected atom sets using PHEAT's keyed matching."""

    from pheat.metrics import structure_rmsd

    normalized_atom_sets = normalize_metric_atom_sets(atom_sets)
    normalized_alignment_atom_set = normalize_rmsd_alignment_atom_set(alignment_atom_set)
    metrics = {}
    for atom_set in normalized_atom_sets:
        try:
            payload = structure_rmsd(
                reference,
                target,
                atom_set=atom_set,
                alignment_atom_set=normalized_alignment_atom_set,
            )
            metrics[atom_set] = {"status": "ok", **payload}
        except Exception as exc:
            metrics[atom_set] = {
                "status": "unavailable",
                "error": str(exc),
                "atom_set": atom_set,
                "alignment_atom_set": normalized_alignment_atom_set,
            }
    return metrics


def radius_of_gyration_summary(
    structure,
    *,
    atom_sets: str | Iterable[str] = METRIC_ATOM_SETS,
) -> dict:
    """Return atom-set radius-of-gyration metrics using PHEAT."""

    from pheat.metrics import structure_radius_of_gyration

    normalized_atom_sets = normalize_metric_atom_sets(atom_sets)
    metrics = {}
    for atom_set in normalized_atom_sets:
        try:
            payload = structure_radius_of_gyration(structure, atom_set=atom_set, mode="both")
            metrics[atom_set] = {"status": "ok", **payload}
        except Exception as exc:
            metrics[atom_set] = {"status": "unavailable", "error": str(exc), "atom_set": atom_set}
    return metrics


def radius_of_gyration_delta_summary(before: dict, after: dict, *, atom_sets: str | Iterable[str]) -> dict:
    """Return atom-set radius-of-gyration deltas using existing PHEAT payloads."""

    deltas = {}
    for atom_set in normalize_metric_atom_sets(atom_sets):
        before_payload = before.get(atom_set) or {}
        after_payload = after.get(atom_set) or {}
        if before_payload.get("status") != "ok" or after_payload.get("status") != "ok":
            deltas[atom_set] = {
                "status": "unavailable",
                "error": "radius of gyration is unavailable for one or both structures",
                "atom_set": atom_set,
                "values": {},
                "units": after_payload.get("units") or before_payload.get("units") or "angstrom",
            }
            continue
        values = {}
        before_values = before_payload.get("values") or {}
        after_values = after_payload.get("values") or {}
        for key, after_value in after_values.items():
            before_value = before_values.get(key)
            if before_value is not None and after_value is not None:
                values[key] = float(after_value) - float(before_value)
        deltas[atom_set] = {
            "status": "ok",
            "atom_set": atom_set,
            "mode": after_payload.get("mode") or before_payload.get("mode") or "both",
            "units": after_payload.get("units") or before_payload.get("units") or "angstrom",
            "values": values,
        }
    return deltas
