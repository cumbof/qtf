#!/usr/bin/env python3

import argparse
import os
import pyrosetta
from pyrosetta import rosetta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out_prefix", default="output")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--nstruct", type=int, default=1)
    args = parser.parse_args()

    pyrosetta.init("-mute all")

    # Read sequence
    with open(args.fasta) as f:
        lines = [l.strip() for l in f if not l.startswith(">")]
    sequence = "".join(lines)

    print(f"Sequence: {sequence}")

    # Create score function
    scorefxn = rosetta.core.scoring.get_score_function()  # ref2015

    # Create packer task
    for i in range(args.nstruct):
        print(f"--- Running structure {i+1} ---")

        # 1. Build pose from sequence (extended chain)
        pose = rosetta.core.pose.Pose()
        rosetta.core.pose.make_pose_from_sequence(
            pose, sequence, "fa_standard"
        )

        # 2. Randomize backbone torsions (important!)
        for res in range(1, pose.size() + 1):
            pose.set_phi(res, rosetta.numeric.random.uniform() * 360 - 180)
            pose.set_psi(res, rosetta.numeric.random.uniform() * 360 - 180)
            pose.set_omega(res, 180.0)

        # 3. Repack sidechains
        task = rosetta.core.pack.task.TaskFactory.create_packer_task(pose)
        task.restrict_to_repacking()
        packer = rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn, task)
        packer.apply(pose)

        # 4. Minimize
        movemap = rosetta.core.kinematics.MoveMap()
        movemap.set_bb(True)
        movemap.set_chi(True)

        min_mover = rosetta.protocols.minimization_packing.MinMover()
        min_mover.movemap(movemap)
        min_mover.score_function(scorefxn)
        min_mover.min_type("lbfgs_armijo_nonmonotone")

        min_mover.apply(pose)

        # 5. Score
        energy = scorefxn(pose)
        print(f"Energy: {energy:.3f}")

        # 6. Save
        os.makedirs(args.outdir, exist_ok=True)
        out_file = os.path.join(args.outdir, f"{args.out_prefix}_{i+1}.pdb")
        pose.dump_pdb(out_file)

if __name__ == "__main__":
    main()
