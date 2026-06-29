"""
runner.py — one SLURM array task = one replica.

Each replica uses its SLURM array ID as the random seed so every job
explores a different part of the energy landscape.

Outputs (per energy backend, e.g. results/rosetta/):
  <outdir>/<energy_backend>/rmsds.csv               — one row per replica
  <outdir>/<energy_backend>/energies.csv            — one row per optimiser step
  <outdir>/<energy_backend>/best_k/                 — best-K snapshot PDBs (when --top_k > 0)
    replica_<N>_rank<rank>_e<energy>.pdb
  <outdir>/<energy_backend>/best_k_index.csv        — index of all kept snapshots across replicas
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import logging
import time
from pathlib import Path

import numpy as np

from qtf.core.folder import QuantumBiophysicsFolder
from qtf.utils.pdb import get_ground_truth_backbone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SEQUENCE = "YYDPETGTWY"
PDB_ID   = "5AWL"
ANSATZ   = "efficient_su2"
MODE     = "statevector"
SHOTS    = 4096


# ── Helpers ───────────────────────────────────────────────────────────────────

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    n = min(len(P), len(Q))
    P, Q = P[:n], Q[:n]
    U, _, Vt = np.linalg.svd(P.T @ Q)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1., 1., d]) @ U.T
    return float(np.sqrt(np.mean(np.sum((P @ R.T - Q) ** 2, axis=1))))


def extract_ca(coords, labels):
    return np.array([c for c, (_, name, _) in zip(coords, labels) if name == "CA"])


def safe_append(path: Path, row: dict, fieldnames: list[str]) -> None:
    """File-locked CSV append so 400 concurrent jobs don't corrupt the file."""
    # Acquire the lock BEFORE checking whether a header is needed so two
    # jobs that start simultaneously cannot both see an empty file and
    # both write a duplicate header row.
    with open(path, "a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        write_header = path.stat().st_size == 0
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(fh, fcntl.LOCK_UN)


def sort_csv(path: Path, sort_key: str) -> None:
    """Re-sort a shared CSV in-place by *sort_key* (ascending, NaN last).

    File-locked so concurrent jobs cannot read a half-written file.
    The last job to finish leaves the file fully sorted.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with open(path, "r+", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            reader = csv.DictReader(fh)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
            if not rows or sort_key not in fieldnames:
                return
            rows.sort(
                key=lambda r: (
                    float("inf")
                    if r.get(sort_key, "nan") in ("nan", "", "NaN")
                    else float(r[sort_key])
                )
            )
            fh.seek(0)
            fh.truncate()
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica_id",     type=int, required=True)
    parser.add_argument("--max_iter",       type=int, default=2000)
    parser.add_argument("--scout",          type=int, default=20)
    parser.add_argument("--strategy",       type=str, default="random",
                        choices=["random", "helix", "sheet"])
    parser.add_argument("--outdir",         type=str, default="results")
    parser.add_argument("--energy_backend", type=str, default="custom",
                        choices=["custom", "rosetta", "openmm"])
    parser.add_argument("--top_k",          type=int, default=5,
                        help="keep the best-K lowest-energy snapshots per replica "
                             "(0 = disabled, written to <outdir>/<backend>/best_k/)")
    args = parser.parse_args()

    # Results are stored in a per-backend subdirectory so runs for different
    # energy functions never overwrite each other.
    outdir = Path(args.outdir) / args.energy_backend
    outdir.mkdir(parents=True, exist_ok=True)

    log.info("Replica %d | seed=%d | strategy=%s | energy_backend=%s | max_iter=%d | scout=%d",
             args.replica_id, args.replica_id, args.strategy, args.energy_backend,
             args.max_iter, args.scout)

    t0 = time.perf_counter()

    # ── Build folder ──────────────────────────────────────────────────────────
    folder = QuantumBiophysicsFolder(
        SEQUENCE, ansatz=ANSATZ, mode=MODE, shots=SHOTS,
        energy_backend=args.energy_backend,
    )

    # ── Scout: use replica_id as seed so every job starts differently ─────────
    rng = np.random.default_rng(args.replica_id)
    best_e, init_params = float("inf"), None
    for _ in range(args.scout):
        p = rng.uniform(-0.8, 0.8, folder.n_params)
        e = folder.energy_function(p)
        if e < best_e:
            best_e, init_params = e, p

    log.info("Scout done | best_start_energy=%.4f", best_e)

    # ── Fold ──────────────────────────────────────────────────────────────────
    coords, labels, bonds, tracker, final_params, final_energy, best_snapshots = folder.fold(
        max_iter=args.max_iter,
        initial_params=init_params,
        top_k_snapshots=args.top_k,
    )

    elapsed = time.perf_counter() - t0
    log.info("Fold done | final_energy=%.4f | wall=%.1fs", final_energy, elapsed)

    # ── Energy history ────────────────────────────────────────────────────────
    energy_path   = outdir / "energies.csv"
    energy_fields = ["replica_id", "seed", "energy_backend", "stage", "step", "energy"]

    stage_map: dict[int, str] = {}
    if hasattr(tracker, "stage_labels") and tracker.stage_labels:
        boundaries = [s for s, _ in tracker.stage_labels] + [len(tracker.history)]
        for (start, sname), end in zip(tracker.stage_labels, boundaries[1:]):
            for idx in range(start, end):
                stage_map[idx] = sname

    for step, e in enumerate(tracker.history):
        safe_append(energy_path, {
            "replica_id":    args.replica_id,
            "seed":          args.replica_id,
            "energy_backend": args.energy_backend,
            "stage":         stage_map.get(step, "unknown"),
            "step":          step,
            "energy":        f"{e:.6f}",
        }, energy_fields)

    # ── RMSD ──────────────────────────────────────────────────────────────────
    rmsd_path   = outdir / "rmsds.csv"
    rmsd_fields = ["replica_id", "seed", "energy_backend", "strategy", "final_energy",
                   "rmsd_ca_A", "n_pred_ca", "n_true_ca", "wall_s"]

    pred_ca = extract_ca(coords, labels)
    try:
        true_ca = get_ground_truth_backbone(PDB_ID)
        rmsd    = kabsch_rmsd(pred_ca, true_ca)
        log.info("RMSD vs %s: %.3f Å", PDB_ID, rmsd)
    except Exception as exc:
        log.warning("RMSD failed: %s", exc)
        rmsd, true_ca = float("nan"), np.zeros((0, 3))

    safe_append(rmsd_path, {
        "replica_id":    args.replica_id,
        "seed":          args.replica_id,
        "energy_backend": args.energy_backend,
        "strategy":      args.strategy,
        "final_energy":  f"{final_energy:.6f}",
        "rmsd_ca_A":     f"{rmsd:.4f}" if np.isfinite(rmsd) else "nan",
        "n_pred_ca":     len(pred_ca),
        "n_true_ca":     len(true_ca),
        "wall_s":        f"{elapsed:.1f}",
    }, rmsd_fields)

    # ── Best-K snapshots ──────────────────────────────────────────────────────
    if best_snapshots:
        snap_dir = outdir / "best_k"
        snap_dir.mkdir(parents=True, exist_ok=True)

        index_path   = outdir / "best_k_index.csv"
        index_fields = ["replica_id", "rank", "energy", "rmsd_ca_A", "pdb_path"]

        for rank, snap in enumerate(best_snapshots, start=1):
            e_str  = f"{snap['energy']:.4f}".replace("-", "m")
            pdb_name = f"replica_{args.replica_id:04d}_rank{rank}_e{e_str}.pdb"
            pdb_path = snap_dir / pdb_name
            folder.save_pdb(
                snap["coords"],
                snap["labels"],
                filename=str(pdb_path),
                energy=snap["energy"],
                remarks=[
                    f"Best snapshot rank {rank} for replica {args.replica_id} "
                    f"(E={snap['energy']:.4f}, backend={args.energy_backend})"
                ],
                include_hydrogens=False,
            )

            # Per-snapshot Cα RMSD vs ground truth (same reference as the
            # per-replica rmsds.csv above). NaN if ground truth was not
            # available or the alignment failed.
            try:
                if len(true_ca) > 0:
                    snap_ca = extract_ca(snap["coords"], snap["labels"])
                    snap_rmsd = kabsch_rmsd(snap_ca, true_ca)
                else:
                    snap_rmsd = float("nan")
            except Exception as exc:
                log.warning("Snapshot RMSD failed (rank %d): %s", rank, exc)
                snap_rmsd = float("nan")

            safe_append(index_path, {
                "replica_id": args.replica_id,
                "rank":       rank,
                "energy":     f"{snap['energy']:.6f}",
                "rmsd_ca_A":  f"{snap_rmsd:.4f}" if np.isfinite(snap_rmsd) else "nan",
                "pdb_path":   str(pdb_path),
            }, index_fields)

        log.info("Best-K snapshots (K=%d) written to %s", len(best_snapshots), snap_dir)
        # Mirror to stdout so the message is also visible in the SLURM .out
        # file (logging.basicConfig sends INFO records to stderr).
        print(f"Best-K snapshots (K={len(best_snapshots)}) written to {snap_dir}", flush=True)

    # ── Sort CSVs so the final files are ordered regardless of job finish order ─
    # Each job re-sorts after appending; the last job to finish leaves the files
    # fully sorted across all replicas.
    sort_csv(rmsd_path, "rmsd_ca_A")
    sort_csv(energy_path, "energy")
    if best_snapshots:
        sort_csv(outdir / "best_k_index.csv", "energy")

    log.info("Done. Results written to %s", outdir)


if __name__ == "__main__":
    main()
