"""
runner.py — one SLURM array task = one replica.

Each replica uses its SLURM array ID as the random seed so every job
explores a different part of the energy landscape.

Outputs:
  results/rmsds.csv      — one row per replica
  results/energies.csv   — one row per optimiser step
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
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(fh, fcntl.LOCK_UN)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replica_id", type=int, required=True)
    parser.add_argument("--max_iter",   type=int, default=2000)
    parser.add_argument("--scout",      type=int, default=20)
    parser.add_argument("--strategy",   type=str, default="random",
                        choices=["random", "helix", "sheet"])
    parser.add_argument("--outdir",     type=str, default="results")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info("Replica %d | seed=%d | strategy=%s | max_iter=%d | scout=%d",
             args.replica_id, args.replica_id, args.strategy, args.max_iter, args.scout)

    t0 = time.perf_counter()

    # ── Build folder ──────────────────────────────────────────────────────────
    folder = QuantumBiophysicsFolder(
        SEQUENCE, ansatz=ANSATZ, mode=MODE, shots=SHOTS, energy_backend="rosetta"
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
    coords, labels, bonds, tracker, final_params, final_energy = folder.fold(
        max_iter=args.max_iter,
        initial_params=init_params,
    )

    elapsed = time.perf_counter() - t0
    log.info("Fold done | final_energy=%.4f | wall=%.1fs", final_energy, elapsed)

    # ── Energy history ────────────────────────────────────────────────────────
    energy_path   = outdir / "energies.csv"
    energy_fields = ["replica_id", "seed", "stage", "step", "energy"]

    stage_map: dict[int, str] = {}
    if hasattr(tracker, "stage_labels") and tracker.stage_labels:
        boundaries = [s for s, _ in tracker.stage_labels] + [len(tracker.history)]
        for (start, sname), end in zip(tracker.stage_labels, boundaries[1:]):
            for idx in range(start, end):
                stage_map[idx] = sname

    for step, e in enumerate(tracker.history):
        safe_append(energy_path, {
            "replica_id": args.replica_id,
            "seed":       args.replica_id,
            "stage":      stage_map.get(step, "unknown"),
            "step":       step,
            "energy":     f"{e:.6f}",
        }, energy_fields)

    # ── RMSD ──────────────────────────────────────────────────────────────────
    rmsd_path   = outdir / "rmsds.csv"
    rmsd_fields = ["replica_id", "seed", "strategy", "final_energy",
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
        "replica_id":   args.replica_id,
        "seed":         args.replica_id,
        "strategy":     args.strategy,
        "final_energy": f"{final_energy:.6f}",
        "rmsd_ca_A":    f"{rmsd:.4f}" if np.isfinite(rmsd) else "nan",
        "n_pred_ca":    len(pred_ca),
        "n_true_ca":    len(true_ca),
        "wall_s":       f"{elapsed:.1f}",
    }, rmsd_fields)

    log.info("Done. Results written to %s", outdir)


if __name__ == "__main__":
    main()
