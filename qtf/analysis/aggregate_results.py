"""Build job-level compatibility indexes from rebuilt per-replica outputs."""

from __future__ import annotations

import csv
import fcntl
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

from qtf.utils.paths import relativize_absolute_paths


def _relative_path(value: Any, root: Path) -> Any:
    if not isinstance(value, str) or not value:
        return value
    candidate = Path(value)
    if not candidate.is_absolute():
        cwd_candidate = (Path.cwd() / candidate).resolve()
        root_candidate = (root / candidate).resolve()
        if cwd_candidate.exists():
            candidate = cwd_candidate
        elif root_candidate.exists():
            candidate = root_candidate
        else:
            return value
    try:
        return os.path.relpath(candidate, root)
    except ValueError:
        return value


def _numeric(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.inf
    return result if math.isfinite(result) else math.inf


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    portable = relativize_absolute_paths(payload)
    temporary.write_text(json.dumps(portable, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(relativize_absolute_paths(rows))
    temporary.replace(path)


def _existing_path(root: Path, *values: Any) -> Path | None:
    for value in values:
        if not value:
            continue
        path = Path(str(value))
        candidates = [path] if path.is_absolute() else [root / path, Path.cwd() / path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
    return None


def _write_multimodel_pdb(path: Path, rows: Iterable[dict[str, Any]], root: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            source = _existing_path(
                root,
                row.get("gromacs_minimized_full_pdb_path"),
                row.get("pdb_path"),
            )
            if source is None:
                continue
            count += 1
            output.write(f"MODEL     {count:4d}\n")
            output.write(
                "REMARK QTF_SOURCE "
                f"replica_id={row.get('replica_id')} "
                f"snapshot_rank_within_replica={row.get('snapshot_rank_within_replica')} "
                f"source={_relative_path(str(source), root)}\n"
            )
            output.write(
                "REMARK QTF_SCORE "
                f"energy={row.get('energy', row.get('snapshot_energy'))} "
                f"rmsd_A={row.get('rmsd_to_reference_A', row.get('snapshot_effective_rmsd_to_reference_A'))}\n"
            )
            for line in source.read_text(encoding="utf-8").splitlines():
                if line[:6].strip().upper() not in {"MODEL", "ENDMDL", "END"}:
                    output.write(line + "\n")
            output.write("ENDMDL\n")
    if count:
        temporary.replace(path)
    else:
        temporary.unlink(missing_ok=True)


def _ensemble_row(result: dict[str, Any], root: Path) -> dict[str, Any]:
    replica_id = int(result.get("replica_id", 0))
    row = {
        "ensemble_rank": 0,
        "ensemble_id": replica_id,
        "replica": replica_id + 1,
        "replica_id": replica_id,
        "seed": result.get("seed"),
        "sequence": result.get("sequence"),
        "recipe": result.get("recipe"),
        "energy": result.get("objective_total", result.get("energy")),
        "energy_model": (
            result.get("objective_model")
            or result.get("result_score_model")
            or result.get("score_model")
        ),
        "score_model": result.get("result_score_model", result.get("score_model")),
        "score_total": result.get("score_total", result.get("primary_score_total")),
        "score_units": result.get("score_units", result.get("primary_score_units")),
        "rmsd_to_reference_A": result.get("rmsd_to_reference_A"),
        "pred_e2e_A": result.get("pred_e2e_A"),
        "pred_rg_A": result.get("pred_rg_A"),
        "ref_e2e_A": result.get("ref_e2e_A"),
        "ref_rg_A": result.get("ref_rg_A"),
        "runtime_s": result.get("runtime_s"),
        "n_qubits": result.get("n_qubits"),
        "n_params": result.get("n_params"),
        "pdb_path": _relative_path(result.get("pdb_path"), root),
        "gromacs_minimized_full_pdb_path": _relative_path(
            result.get("gromacs_refined_pdb_path")
            if result.get("gromacs_refinement_status") == "ok"
            else None,
            root,
        ),
        "gromacs_status": result.get("gromacs_refinement_status"),
        "gromacs_converged": result.get("gromacs_refinement_converged"),
        "gromacs_final_max_force": result.get("gromacs_refinement_final_max_force"),
        "gromacs_potential_kj_mol": result.get("gromacs_potential_kj_mol"),
        "ca_pdb_path": _relative_path(result.get("ca_pdb_path"), root),
        "result_path": _relative_path(
            result.get("_result_path")
            or str(root / "replicas" / f"replica_{replica_id}" / f"replica_{replica_id}_result.json"),
            root,
        ),
        "circuit_params_json_path": _relative_path(result.get("circuit_params_json_path"), root),
        "circuit_params_npz_path": _relative_path(result.get("circuit_params_npz_path"), root),
    }
    for key, value in (result.get("score_terms") or {}).items():
        row[f"score_term_{key}"] = value
    return row


def _snapshot_rows(result: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    replica_id = int(result.get("replica_id", 0))
    rows = []
    for snapshot in result.get("structure_snapshots") or []:
        if snapshot.get("role") != "top_snapshot" or snapshot.get("snapshot_status") != "ok":
            continue
        rows.append(
            {
                "snapshot_global_rank": 0,
                "ensemble_rank": 0,
                "ensemble_id": replica_id,
                "replica": replica_id + 1,
                "replica_id": replica_id,
                "snapshot_rank_within_replica": snapshot.get("snapshot_rank"),
                "snapshot_key": snapshot.get("key"),
                "snapshot_iteration": snapshot.get("iteration"),
                "snapshot_energy": snapshot.get("objective"),
                "snapshot_effective_rmsd_to_reference_A": snapshot.get("rmsd"),
                "snapshot_rmsd_to_reference_A": snapshot.get("rmsd"),
                "snapshot_gromacs_potential_kj_mol": snapshot.get("gromacs_potential_kj_mol"),
                "snapshot_gromacs_potential_kcal_mol": snapshot.get("gromacs_potential_kcal_mol"),
                "gromacs_minimized_full_pdb_path": _relative_path(
                    snapshot.get("gromacs_minimized_full_pdb_path"), root
                ),
                "pdb_path": _relative_path(
                    snapshot.get("viewer_pdb_path") or snapshot.get("pdb_path"), root
                ),
            }
        )
    return rows


def aggregate_job_outputs(job_dir: str | Path) -> dict[str, Any]:
    """Regenerate main-style job indexes from every completed rebuilt replica."""

    root = Path(job_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".qtf_aggregate.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        result_paths = sorted(root.glob("replicas/replica_*/replica_*_result.json"))
        result_paths.extend(sorted(root.glob("replica_*_primary_outputs/replica_*_result.json")))
        result_paths.extend(sorted(root.glob("replica_*_result.json")))
        results = []
        for path in result_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_result_path"] = str(path.resolve())
            results.append(payload)
        ensemble = sorted((_ensemble_row(result, root) for result in results), key=lambda row: _numeric(row["energy"]))
        for rank, row in enumerate(ensemble, start=1):
            row["ensemble_rank"] = rank

        snapshots = []
        ensemble_rank_by_replica = {row["replica_id"]: row["ensemble_rank"] for row in ensemble}
        for result in results:
            snapshots.extend(_snapshot_rows(result, root))
        snapshots.sort(key=lambda row: (_numeric(row["snapshot_energy"]), _numeric(row["snapshot_effective_rmsd_to_reference_A"])))
        for rank, row in enumerate(snapshots, start=1):
            row["snapshot_global_rank"] = rank
            row["ensemble_rank"] = ensemble_rank_by_replica.get(row["replica_id"])

        if ensemble:
            _write_csv(root / "ensemble_ranked.csv", ensemble)
            _write_json(root / "ensemble_ranked.json", ensemble)
            _write_multimodel_pdb(root / "ensemble_ranked.pdb", ensemble, root)
            best = ensemble[0]
            summary = {
                "Sequence": best.get("sequence"),
                "Recipe": best.get("recipe"),
                "Ensemble Size": len(ensemble),
                "Best Objective": best.get("energy"),
                "Objective Model": best.get("energy_model"),
                "Best Score": best.get("score_total"),
                "Score Model": best.get("score_model"),
                "Score Units": best.get("score_units"),
                "Backbone RMSD (Å)": best.get("rmsd_to_reference_A"),
                "End-to-End Dist (Å)": best.get("pred_e2e_A"),
                "Rg (Å)": best.get("pred_rg_A"),
                "Output Dir": ".",
            }
            _write_csv(root / "summary_results.csv", [summary])
            _write_json(root / "summary_results.json", summary)
        if snapshots:
            _write_csv(root / "snapshot_ranked.csv", snapshots)
            _write_json(root / "snapshot_ranked.json", snapshots)
            _write_multimodel_pdb(root / "snapshot_ranked.pdb", snapshots, root)
        return {"replicas": len(ensemble), "snapshots": len(snapshots), "job_dir": str(root)}
