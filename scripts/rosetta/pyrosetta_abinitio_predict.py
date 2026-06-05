#!/usr/bin/env python3
"""
PyRosetta fragment-based ab initio prediction driver.

This is much closer to the intended classical Rosetta ab initio workflow than
randomizing torsions and minimizing:
  FASTA/sequence
  -> centroid Pose
  -> ClassicAbinitio with 9-mer and 3-mer fragments
  -> switch to full atom
  -> sidechain repack + FastRelax
  -> ref2015 score + PDB output

You must provide Rosetta fragment files:
  --frag3  aatarget03_05.200_v1_3
  --frag9  aatarget09_05.200_v1_3

Example:
python pyrosetta_abinitio_predict.py \
  --fasta target.fasta \
  --frag3 aatarget03_05.200_v1_3 \
  --frag9 aatarget09_05.200_v1_3 \
  --nstruct 50 \
  --outdir pyrosetta_abinitio_trpcage \
  --out_prefix trpcage
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd


def read_fasta(path: str) -> str:
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_lines.append(line)
    seq = "".join(seq_lines).strip().upper()
    if not seq:
        raise ValueError(f"No sequence found in FASTA: {path}")
    return seq


def init_pyrosetta(extra_flags: str = ""):
    import pyrosetta

    flags = [
        "-mute all",
        "-ignore_unrecognized_res true",
        "-ex1",
        "-ex2aro",
        "-use_input_sc false",
    ]
    if extra_flags:
        flags.append(extra_flags)
    pyrosetta.init(" ".join(flags))
    return pyrosetta


def make_extended_centroid_pose(sequence: str, rosetta):
    import pyrosetta

    pose = rosetta.core.pose.Pose()
    rosetta.core.pose.make_pose_from_sequence(pose, sequence, "fa_standard")

    # Start from a simple extended-ish chain before switching to centroid.
    for i in range(1, pose.size() + 1):
        pose.set_phi(i, -150.0)
        pose.set_psi(i, 150.0)
        pose.set_omega(i, 180.0)

    to_centroid = rosetta.protocols.simple_moves.SwitchResidueTypeSetMover("centroid")
    to_centroid.apply(pose)
    return pose



def load_fragsets(frag3_path: str, frag9_path: str, rosetta):
    frag3_path = Path(frag3_path)
    frag9_path = Path(frag9_path)

    if not frag3_path.exists():
        raise FileNotFoundError(f"3-mer fragment file not found: {frag3_path.resolve()}")
    if not frag9_path.exists():
        raise FileNotFoundError(f"9-mer fragment file not found: {frag9_path.resolve()}")

    if frag3_path.stat().st_size == 0:
        raise ValueError(f"3-mer fragment file is empty: {frag3_path.resolve()}")
    if frag9_path.stat().st_size == 0:
        raise ValueError(f"9-mer fragment file is empty: {frag9_path.resolve()}")

    frag3 = rosetta.core.fragment.ConstantLengthFragSet(3)
    frag3.read_fragment_file(str(frag3_path))

    frag9 = rosetta.core.fragment.ConstantLengthFragSet(9)
    frag9.read_fragment_file(str(frag9_path))

    return frag3, frag9


def run_classic_abinitio(pose, frag3, frag9, rosetta, quick: bool = False):
    movemap = rosetta.core.kinematics.MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(False)
    movemap.set_jump(True)

    abinitio = rosetta.protocols.abinitio.ClassicAbinitio(frag3, frag9, movemap)

    # Optional speed/debug mode.
    if quick:
        # These setters are available in modern PyRosetta builds for some stages.
        # Keep guarded so script remains portable across builds.
        for name, value in [
            ("set_stage1_cycles", 200),
            ("set_stage2_cycles", 200),
            ("set_stage3_cycles", 200),
            ("set_stage4_cycles", 400),
        ]:
            if hasattr(abinitio, name):
                try:
                    getattr(abinitio, name)(value)
                except Exception:
                    pass

    abinitio.init(pose)
    abinitio.apply(pose)
    return pose


def fullatom_repack_relax_score(pose_centroid, rosetta, relax_repeats: int = 3):
    pose = pose_centroid.clone()

    to_fa = rosetta.protocols.simple_moves.SwitchResidueTypeSetMover("fa_standard")
    to_fa.apply(pose)

    scorefxn = rosetta.core.scoring.ScoreFunctionFactory.create_score_function("ref2015")

    # Repack sidechains first.
    task = rosetta.core.pack.task.TaskFactory.create_packer_task(pose)
    task.restrict_to_repacking()
    packer = rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn, task)
    packer.apply(pose)

    # FastRelax is the normal full-atom cleanup step.
    relax = rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(scorefxn)
    try:
        relax.max_iter(200)
    except Exception:
        pass

    # Some builds expose repeats through constructor or setter differently.
    # Applying once is still valid; n repeats handled by loop below.
    for _ in range(max(1, int(relax_repeats))):
        relax.apply(pose)

    fa_score = float(scorefxn(pose))
    return pose, fa_score


def centroid_score(pose_centroid, rosetta, scorefxn_name: str = "score3") -> float:
    scorefxn = rosetta.core.scoring.ScoreFunctionFactory.create_score_function(scorefxn_name)
    return float(scorefxn(pose_centroid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--frag3", required=True)
    ap.add_argument("--frag9", required=True)
    ap.add_argument("--nstruct", type=int, default=10)
    ap.add_argument("--outdir", default="pyrosetta_abinitio_out")
    ap.add_argument("--out_prefix", default="model")
    ap.add_argument("--relax_repeats", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="Reduce ab initio cycles for smoke testing.")
    ap.add_argument("--extra_flags", default="", help="Extra flags passed to pyrosetta.init.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sequence = read_fasta(args.fasta)
    pyrosetta = init_pyrosetta(args.extra_flags)
    from pyrosetta import rosetta

    frag3, frag9 = load_fragsets(args.frag3, args.frag9, rosetta)

    rows = []
    for i in range(1, args.nstruct + 1):
        print(f"--- Abinitio structure {i}/{args.nstruct} ---")

        cen_pose = make_extended_centroid_pose(sequence, rosetta)
        run_classic_abinitio(cen_pose, frag3, frag9, rosetta, quick=args.quick)

        cen_e = centroid_score(cen_pose, rosetta, "score3")

        fa_pose, fa_e = fullatom_repack_relax_score(
            cen_pose,
            rosetta,
            relax_repeats=args.relax_repeats,
        )

        pdb_path = outdir / f"{args.out_prefix}_{i:04d}_E{fa_e:.3f}.pdb"
        fa_pose.dump_pdb(str(pdb_path))

        row = {
            "generation_index": i,
            "pdb_path": str(pdb_path),
            "centroid_score3": cen_e,
            "ref2015": fa_e,
            "sequence": sequence,
            "frag3": str(args.frag3),
            "frag9": str(args.frag9),
            "relax_repeats": args.relax_repeats,
            "quick": bool(args.quick),
        }
        rows.append(row)
        print(f"centroid_score3={cen_e:.3f} ref2015={fa_e:.3f} pdb={pdb_path}")

    df = pd.DataFrame(rows).sort_values("ref2015")
    csv_path = outdir / f"{args.out_prefix}_abinitio_scores.csv"
    json_path = outdir / f"{args.out_prefix}_abinitio_scores.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2))

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
