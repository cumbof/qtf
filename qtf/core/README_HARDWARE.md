qtf/core/hardware.py
============================

WHAT IT DOES
------------
Runs one QTF circuit on real quantum hardware, turns the measurement results into
protein torsion angles, and rebuilds the 3D protein structure.

Steps:
  1. Initialize random circuit parameters, or load saved parameters from a
     qtf fold-simulation job when --params-json is supplied.
  2. Rebuild the folder (sequence + ansatz) that produced it.
  3. Bind parameters, add measure_all, transpile for the backend.
  4. Run once on hardware (IBM Runtime) or on AerSimulator.
  5. Convert the shot histogram -> probability vector -> CDF -> angles.
  6. Rebuild heavy-atom coordinates with NERF.
  7. Optionally align to a reference PDB and compute RMSD.
  8. Save the raw hardware PDB and JSON metadata.
  9. By default, refine the rebuilt structure with GROMACS and add raw,
     refined, and effective structural metrics to the JSON.

Command:
  python -m qtf.core.hardware [options]


INPUTS
------
Primary inputs:
  --sequence      Required only when --params-json is omitted.
  --params-json   Optional replica JSON, circuit_parameters manifest/directory,
                  or simulation output directory.
  --replica-id    Zero-based replica ID when selecting from a multi-replica input.
  --outdir        Output directory (default: run_outputs/hardware_fold).

Backend:
  --backend-name       e.g. ibm_torino
  --channel            IBM Runtime channel
  --instance           IBM Runtime instance
  --token              IBM Runtime API token
  --local-simulator    Explicit Aer dry-run instead of an IBM submission
  --shots              Number of shots (default 8192)
  --optimization-level Transpiler level (default 3)

If --backend-name is omitted, IBM Runtime least_busy() selects an operational,
non-simulator backend with enough qubits for the circuit.

Reference / RMSD (optional):
  --reference_pdb              Local PDB file
  --reference_structure        PDB ID to fetch
  --rmsd_mode                  ca | heavy   (default ca)
  --rmsd_residue_scope         core | all   (default core)
  --average_reference_backbone flag

Folder overrides (only used if not stored in the JSON):
  --sequence, --chi_mode, --omega_mode,
  --use_e2e_constraint, --e2e_scale

GROMACS refinement:
  --gromacs              Enabled by default
  --no-gromacs           Skip refinement
  --gromacs-outdir       Defaults to OUTDIR/gromacs_minimized
  --gromacs-forcefield   Defaults to amber99sb-ildn
  --gromacs-water        Defaults to tip3p


OUTPUTS
-------
1. PDB file at OUTDIR/hardware_model.pdb (or --out-pdb)
   Contains the rebuilt heavy-atom structure with REMARK header:
     - backend name and kind (ibm_runtime / aer)
     - shots and unique bitstrings
     - sequence, replica id, ensemble id, timestamp
     - RMSD to reference (if a reference was given)

2. JSON file at OUTDIR/hardware_result.json (or --out-json)
   Metadata: backend, shots, unique bitstrings, n_qubits, n_params,
   total_angles, RMSD value + mode + scope, reference source,
   absolute paths, and timestamp.

3. GROMACS artifacts in OUTDIR/gromacs_minimized (or --gromacs-outdir)
   The minimized PDB, minimization log, and supporting refinement artifacts.

4. Console log
   Progress messages: parameters loaded, folder rebuilt, backend used,
   sampling summary, RMSD value, output paths.


EXAMPLE
-------
Hardware run:
  qtf fold-hardware \
      --params-json ./replica_0_params.json \
      --shots 8192 \
      --outdir ./hardware_result \
      --reference_pdb ./native.pdb

Random-parameter hardware run:
  qtf fold-hardware --sequence YYDPETGTWY --seed 12345

Explicit local dry-run:
  qtf fold-hardware --local-simulator --sequence YYDPETGTWY --seed 12345

Manifest/SLURM-array dispatch:
  scripts/run_hardware_from_manifest.sh manifest.csv 1
  scripts/run_hardware_from_manifest.sh manifest.csv all
