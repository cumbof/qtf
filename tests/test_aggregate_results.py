import csv
import json

from qtf.analysis.aggregate_results import aggregate_job_outputs


def _write_replica(root, replica_id, energy, rmsd):
    primary = root / f"replica_{replica_id}_primary_outputs"
    primary.mkdir()
    pdb = primary / f"replica_{replica_id}.pdb"
    pdb.write_text("ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n")
    result = {
        "replica_id": replica_id,
        "seed": replica_id + 10,
        "sequence": "AA",
        "recipe": "qtf-main-snapshot-equivalent",
        "objective_total": energy,
        "optimizer_objective": "pheat-coarse-protein-folding-v1",
        "score_total": energy,
        "score_units": "arbitrary",
        "result_score_model": "pheat-coarse-protein-folding-v1",
        "rmsd_to_reference_A": rmsd,
        "pdb_path": str(pdb),
        "structure_snapshots": [
            {
                "role": "top_snapshot",
                "snapshot_status": "ok",
                "snapshot_rank": 1,
                "key": f"snapshot_{replica_id}",
                "objective": energy + 1,
                "rmsd": rmsd + 0.1,
                "pdb_path": str(pdb),
            }
        ],
    }
    (primary / f"replica_{replica_id}_result.json").write_text(json.dumps(result))


def test_aggregate_job_outputs_writes_ranked_relative_indexes(tmp_path):
    _write_replica(tmp_path, 0, 5.0, 2.0)
    _write_replica(tmp_path, 1, 2.0, 1.0)

    counts = aggregate_job_outputs(tmp_path)

    assert counts["replicas"] == 2
    assert counts["snapshots"] == 2
    with (tmp_path / "ensemble_ranked.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["replica_id"]) for row in rows] == [1, 0]
    assert rows[0]["pdb_path"] == "replica_1_primary_outputs/replica_1.pdb"
    assert not rows[0]["pdb_path"].startswith("/")
    assert (tmp_path / "ensemble_ranked.json").is_file()
    assert (tmp_path / "ensemble_ranked.pdb").is_file()
    assert (tmp_path / "snapshot_ranked.csv").is_file()
    assert (tmp_path / "snapshot_ranked.json").is_file()
    assert (tmp_path / "snapshot_ranked.pdb").is_file()
    assert (tmp_path / "summary_results.csv").is_file()
    assert (tmp_path / "summary_results.json").is_file()
