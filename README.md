# QTF: Quantum Torsion Folder

> **Logarithmic-scale variational quantum eigensolver for torsion-space, off-lattice protein structure prediction.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-qtf-orange.svg)](https://pypi.org/project/qtf/)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture at a Glance](#architecture-at-a-glance)
3. [Package Structure](#package-structure)
4. [Dependencies](#dependencies)
5. [Installation](#installation)
6. [CLI](#cli)
7. [Quick Start](#quick-start)
8. [Core Concepts](#core-concepts)
   - [Encoding](#encoding)
   - [Degrees of Freedom](#degrees-of-freedom)
   - [NERF Geometry Builder](#nerf-geometry-builder)
   - [Three-Stage Optimisation](#three-stage-optimisation)
9. [Physics Engine](#physics-engine)
   - [Force Fields](#force-fields)
   - [Energy Terms](#energy-terms)
10. [Ensemble Folding](#ensemble-folding)
    - [Initialisation Strategy](#initialisation-strategy)
    - [Running an Ensemble](#running-an-ensemble)
11. [Ranking and Analysis](#ranking-and-analysis)
    - [EnsembleRanking Statistics](#ensembleranking-statistics)
    - [Convergence Assessment](#convergence-assessment)
    - [Kabsch RMSD](#kabsch-rmsd)
12. [Visualisation](#visualisation)
    - [3-D Structure Viewer](#3-d-structure-viewer)
    - [Energy Landscape](#energy-landscape)
    - [Ranking Dashboard](#ranking-dashboard)
13. [PDB Utilities](#pdb-utilities)
14. [API Reference](#api-reference)
15. [Reproducibility](#reproducibility)
16. [Logging](#logging)
17. [References](#references)
18. [License](#license)

---

## Overview

**QTF** (Quantum Torsion Folder) is a hybrid quantum-classical protein structure prediction package. Unlike traditional lattice-based quantum folding approaches that snap atoms onto a discrete grid, QTF works entirely in **continuous torsion space**: backbone dihedral angles (φ, ) and side-chain rotamer angles (χ₁–χ₅) are the fundamental degrees of freedom.

Rather than assigning one qubit per degree of freedom — which would require hundreds of qubits for realistic proteins — QTF uses only ⌈log₂ N⌉ qubits to represent N continuous angles. The phases of the complex amplitudes in the quantum statevector are extracted and mapped directly to torsion angles in [−π, π]. This allows near-term quantum hardware (or exact simulation) to parameterise rich conformational spaces with a logarithmically small quantum register.

The quantum circuit acts as a **generative model** (the "Actor"), while a physics-based energy function acts as the **critic**. Classical optimisers (COBYLA and SLSQP) close the loop, tuning the circuit parameters until the energy is minimised. Multiple independent runs are managed by an ensemble manager and ranked by a comprehensive statistics module.

---

## Architecture at a Glance

```

                       QUANTUM ACTOR

    θ (parameters)  ──►  EfficientSU2 circuit  ──►  |ψ⟩
                     (⌈log₂ N⌉ qubits)
                             │
                Extract phases of amplitudes
                             │
                    φ₀, ψ₀, χ₁⁰, φ₁, ψ₁, …  (torsion angles)

                             │
                             ▼

                   NERF GEOMETRY BUILDER

  Torsion angles  ──►  3-D Cartesian coordinates (all atoms)
               N, Cα, C, O + full side chains

                             │
                             ▼

                  CLASSICAL CRITIC (Energy)

  Solvation · H-bonds · Electrostatics · Sterics · Disulfide
  Ramachandran · Rotamers · π–π stacking · Geometry integrity

                             │
                             ▼

            CLASSICAL OPTIMISER (COBYLA / SLSQP)

  Stage 1 – Collapse   (COBYLA,  high constraint)
  Stage 2 – Refine     (SLSQP,  high constraint)
  Stage 3 – Relax      (SLSQP,  low constraint)

```

---

## Package Structure

```
qtf/

 __init__.py                  Public API exports (3 core classes) + __version__

 core/
   ├── folder.py                QuantumBiophysicsFolder — main predictor class
   ├── ensemble.py              EnsembleFoldingManager — multi-replica orchestrator
   └── tracker.py              LandscapeTracker — per-evaluation energy logger

 analysis/
   ├── ranking.py              EnsembleRanking — comprehensive statistics & selection
   ├── stability.py             kabsch_rmsd(), StabilityAnalyzer
   └── panel.py                Panel/grid-analysis pipeline

 visualization/
   └── plots.py                plot_structure(), plot_energy_landscape(), plot_ranking()

 utils/
   ├── pdb.py                  save_pdb(), get_ground_truth_backbone(),
   │                           calculate_physics_metrics()
   ├── workflow.py             make_folder(), score_native_structure(),
   │                           gromacs_postprocess_structure(), RMSD utils
   ├── aer_sim.py              GPU/CPU statevector simulation via qiskit-aer
   ├── accelerate.py           Numba-jitted distance matrix & energy kernels
   └── gromacs.py              GROMACS minimisation bridge (optional)

 cli/
   ├── __init__.py             qtf: canonical recipe-driven folding command
   └── make_vmd_trajectory.py  qtf vmd-trajectory: VMD-safe trajectories
```

---

## Dependencies

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| `numpy` | 1.24 | Array mathematics, distance matrices, linear algebra |
| `scipy` | 1.10 | COBYLA and SLSQP optimisers (`scipy.optimize.minimize`) |
| `qiskit` | 1.0 | Ansatz construction; exact statevector simulation |
| `qiskit-aer` | 0.14 | GPU/CPU statevector simulation backend |
| `plotly` | 5.18 | Interactive 3-D and 2-D Plotly figures |
| `pandas` | 2.0 | Per-replica statistics `DataFrame`; CSV export |
| `pheat` | 0.2 | Heavy-atom reconstruction, scoring, external-engine preparation, and report assets |
| `PyYAML` | — | Recipe loading |
| `jsonschema` | — | Recipe and structured-output validation |

Optional extras:

| Extra | Dependencies | Use case |
|-------|-------------|----------|
| `[dev]` | `pytest`, `pytest-cov`, `ruff`, `mypy` | Development, linting, testing |
| `[notebook]` | `jupyter`, `nbformat` | Running `QTF.ipynb` |
| `[workflows]` | `matplotlib`, `mdtraj`, `biopython` | RMSD/reference workflows, PDB I/O, reports, and post-run analysis |
| `[gpu]` | `numba`, `qiskit-aer-gpu` | GPU-accelerated classical & quantum simulation (Linux x86_64) |
| `[docs]` | `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` | Building this documentation site

> [!NOTE]
>
> QTF's default `custom` energy backend works out of the box — no additional dependencies required.
>
> - External physics engines are configured in fold recipes through PHEAT scorer/evaluator options.
> - **OpenMM/PDBFixer** are intentionally not included in QTF's pip extras. Install them via conda-forge when using OpenMM-backed PHEAT evaluators:
>   ```bash
>   conda install -c conda-forge openmm pdbfixer
>   ```
> - **GROMACS 2026 or newer** (used by PHEAT external validation recipes and `gromacs_postprocess_structure()`) must be available as `gmx` or `gmx_mpi`. Because GROMACS is an external executable rather than a Python package, it is not listed in `pyproject.toml`. Install it system-wide on `PATH`, or inside the active conda environment:
>   ```bash
>   conda install -c conda-forge "gromacs>=2026"
>   ```
>   Verify the environment before submitting long jobs:
>   ```bash
>   which gmx
>   gmx --version
>   ```
> - Per-replica HTML reports use PHEAT-managed **Mol\*** browser assets. Node/npm is required only to download the pinned assets; it is not needed to open completed reports because one shared bundle is copied to `report_assets/molstar` in each job directory:
>   ```bash
>   conda install -c conda-forge nodejs
>   pheat molstar install
>   pheat molstar status
>   ```

---

## Installation

**From PyPI (recommended):**

```bash
pip install qtf
```

**Editable install from source (for development or contribution):**

```bash
git clone https://github.com/cumbof/QTF.git
cd QTF
pip install -e ".[dev,workflows]"
```

**Minimum Python version:** 3.9

---

## CLI

QTF installs one canonical command, `qtf`.

### `qtf fold-simulation` — recipe-driven folding

```bash
qtf fold-simulation qtf-main-equivalent \
  --sequence YYDPETGTWY \
  --reference-structure pdb/5AWL.pdb \
  --outdir outputs/example_fold
```

Omitting `--backend` uses exact local statevector extraction. Pass a backend such
as `aer`, `statevector-shots`, or an IBM backend name only when you want sampled
backend execution.

Useful discovery commands:

```bash
qtf fold-simulation --list-recipes
qtf fold-simulation --show-recipe qtf-main-equivalent
```

`qtf fold-simulation` runs named YAML recipes. The built-in main-equivalent recipe uses
PHEAT geometry/scoring through the native `pheat-custom-energy-v1` scorer and reports
RMSD only when a reference structure is provided.

Completed fold runs also save final circuit parameters under
`<outdir>/circuit_parameters/` as per-replica JSON/NPZ files plus a
`circuit_parameters.json` manifest. These files can warm-start later simulator
or hardware runs:

```bash
qtf fold-simulation qtf-main-equivalent \
  --sequence YYDPETGTWY \
  --reference-structure pdb/5AWL.pdb \
  --initial-params outputs/example_fold \
  --initial-params-select best_energy \
  --outdir outputs/warm_start
```

`--initial-params` can point at a previous output directory,
`circuit_parameters/`, `circuit_parameters.json`, or a single JSON/NPZ/NPY
vector file. `--initial-params-select best_energy` selects the lowest raw
QTF/backend objective energy from the previous ensemble, not the GROMACS
post-minimization energy. `best_rmsd` selects the lowest previous reference
RMSD when available; `first` maps replica N to the saved replica N vector.

### `qtf fold-hardware` — cold starts and warm starts

Hardware execution uses the same recipe and structure-rebuilding pipeline as
simulation. If `--params-json` is omitted, QTF creates random circuit
parameters (a hardware cold start). If it is supplied, QTF loads the selected
replica's saved circuit parameters (a hardware warm start). When
`--backend-name` is omitted, Qiskit IBM Runtime selects the least-busy
operational backend; provide a backend name to request a specific device.

Hardware cold start on the least-busy suitable IBM backend:

```bash
qtf fold-hardware \
  --sequence YYDPETGTWY \
  --recipe qtf-main-snapshot-equivalent \
  --channel ibm_quantum_platform \
  --shots 8192 \
  --optimization-level 3 \
  --sampler-max-mitigation \
  --outdir outputs/hardware_cold_start
```

Hardware cold start on a named backend:

```bash
qtf fold-hardware \
  --sequence YYDPETGTWY \
  --recipe qtf-main-snapshot-equivalent \
  --backend-name ibm_torino \
  --channel ibm_quantum_platform \
  --shots 8192 \
  --optimization-level 3 \
  --sampler-max-mitigation \
  --outdir outputs/hardware_cold_start_named
```

Hardware warm start from replica 0 saved by a simulator run:

```bash
qtf fold-hardware \
  --params-json outputs/example_fold/circuit_parameters \
  --replica-id 0 \
  --channel ibm_quantum_platform \
  --shots 8192 \
  --optimization-level 3 \
  --sampler-max-mitigation \
  --outdir outputs/hardware_warm_start
```

For an offline hardware-path test without IBM credentials, add
`--local-simulator`. To skip the default post-run GROMACS refinement, add
`--no-gromacs`. Hardware commands write the rebuilt PDB, result metadata,
transpiled-circuit information, and validation artifacts under `--outdir`.

### `qtf vmd-trajectory` — VMD-compatible multi-model PDBs

```bash
qtf vmd-trajectory --input snapshot_ranked.pdb --output snapshot_ranked_vmd.pdb
```

### External validation

AmberTools, OpenMM, and GROMACS are accessed through PHEAT scorer/evaluator
configuration inside fold recipes, not through separate QTF commands.


---

## Quick Start

```python
import logging
logging.basicConfig(level=logging.INFO)   # optional — enables progress messages

from qtf import QuantumBiophysicsFolder, EnsembleFoldingManager
from qtf.analysis import EnsembleRanking
from qtf.visualization import plot_structure, plot_energy_landscape, plot_ranking
from qtf.utils import get_ground_truth_backbone, save_pdb

# ── 1. Initialise the folder ─────────────────────────────────────────────────
folder = QuantumBiophysicsFolder(
    sequence="YYDPETGTWY",   # Chignolin — a well-studied mini-protein
)

# ── 2. Run an ensemble of independent replicas folding ──────────────────────
manager = EnsembleFoldingManager(folder)
manager.run_ensemble(
    n_runs=5,            # number of independent replicas
    max_iter=2000,       # max optimiser iterations per stage per replica
    scout_attempts=50,   # basin-hopping samples evaluated before each full run
)

# ── 3. (Optional) Fetch experimental ground-truth Cα coordinates ─────────────
true_ca = get_ground_truth_backbone("5AWL", cache_dir="./pdb_cache")

# ── 4. Build the comprehensive ranking ──────────────────────────────────────
ranking = EnsembleRanking.from_ensemble(
    manager.get_results(),
    ground_truth_ca=true_ca,   # omit entirely if no reference is available
)

# ── 5. Print human-readable summary ─────────────────────────────────────────
print(ranking.summary())

# ── 6. Access the two best structures ───────────────────────────────────────
best_e = ranking.best_by_energy   # replica dict — always available
best_r = ranking.best_by_rmsd     # replica dict — None if no ground truth given

# ── 7. Export the best-energy structure to PDB ──────────────────────────────
save_pdb(
    best_e["coords"],
    best_e["labels"],
    folder.sequence,
    filename="best_energy.pdb",
    energy=best_e["energy"],
)

# ── 8. Interactive figures Plotly ─────────────────────────────────────
plot_structure(ranking, ground_truth_ca=true_ca).show()
plot_energy_landscape(ranking).show()
plot_ranking(ranking).show()
```

**Without a ground-truth reference:**

```python
ranking = EnsembleRanking.from_ensemble(manager.get_results())
# ranking.best_by_rmsd  →  None
# ranking.stats_df["rmsd_vs_gt"]  →  all NaN
# The best prediction is still accessible via ranking.best_by_energy
print(ranking.best_by_energy["energy"])
```

---

## Core Concepts

### Encoding

Standard quantum approaches to protein folding assign one or more qubits per torsion angle, requiring O(N) qubits for a protein with N degrees of freedom. QTF instead exploits the **exponential dimensionality** of the quantum state space:

- A circuit of `n_qubits = ⌈log₂ N⌉` qubits produces a statevector with 2^n_qubits complex amplitudes.
- Each complex amplitude `αⱼ = |αⱼ| · e^{iφⱼ}` carries a **phase** φⱼ ∈ (−π, π].
- The first N phases are extracted and used directly as the N torsion angles.

This mapping is **surjective** — every point in (−π, π]^N is reachable by some parameter vector — and **smooth**, enabling gradient-based optimisation via SLSQP.

The circuit depth is set as:

```
n_qubits = max(2, ceil(log2(total_angles)))
reps     = ceil(total_angles / n_qubits) + 2
```

The `reps` heuristic ensures the circuit has enough expressibility to cover the target angle space while remaining shallow enough for efficient classical simulation. The ansatz used is Qiskit's `EfficientSU2` with **circular entanglement** — a hardware-efficient parameterised circuit commonly used in VQE experiments. Each layer applies single-qubit SU(2) rotations followed by a ring of CX (CNOT) entangling gates.

### Degrees of Freedom

For a protein of length L, the degrees of freedom are:

| Type | Count per residue | Description |
|------|------------------|-------------|
| φ (phi) | 1 | N–Cα–C–N backbone dihedral angle |
| ψ (psi) | 1 | Cα–C–N–Cα backbone dihedral angle |
| ω (omega) | 0–1 | Peptide plane twist; exposed as an optimisable DOF only for the residue **preceding** a Pro (cis/trans isomerism) |
| χ₁–χ₅ | 0–5 (residue-dependent) | Side-chain rotamer angles |

**Side-chain torsion counts by residue:**

| Residue | χ angles | Notes |
|---------|----------|-------|
| Gly (G) | 0 | No side chain |
| Ala (A) | 0 | Methyl treated as fixed |
| Ser (S), Cys (C) | 2 | χ₁ + hydroxyl/thiol |
| Thr (T) | 2 | χ₁ + branched χ₂ |
| Val (V) | 2 | χ₁ + branched χ₂ |
| Leu (L), Ile (I) | 3 | χ₁–χ₂ + branch |
| Asp (D), Asn (N) | 2 | χ₁–χ₂ |
| His (H) | 2 | χ₁–χ₂ (imidazole) |
| Pro (P) | 1 | Ring-constrained χ₁ |
| Met (M) | 4 | χ₁–χ₄ |
| Glu (E), Gln (Q), Phe (F), Tyr (Y) | 2–3 | χ₁–χ₂ (+ ring) |
| Trp (W) | 2 | χ₁–χ₂ (indole) |
| Lys (K) | 5 | χ₁–χ₅ |
| Arg (R) | 5 | χ₁–χ₅ |

The `dof_map` attribute of `QuantumBiophysicsFolder` is a list of dicts, one per degree of freedom, each with keys `res` (0-based residue index) and `type` (`"phi"`, `"psi"`, `"chi1"`, `"chi2"`, …). The order is: all DOFs for residue 0, then residue 1, etc.; within each residue, φ comes first, then ψ, then χ angles in alphabetical order of their label.

### NERF Geometry Builder

`build_full_structure(angle_vector)` converts a flat vector of torsion angles into full 3-D Cartesian coordinates using the **Natural Extension Reference Frame (NERF)** algorithm. Given three known atom positions A, B, C and a new atom D defined by its bond length L, bond angle θ, and torsion τ, NERF places D analytically without any iterative refinement:

```
bc_unit = (C − B) / ‖C − B‖
n_unit  = (B−A) × bc_unit / ‖(B−A) × bc_unit‖
bxn     = n_unit × bc_unit

M = [ bc_unit | bxn | n_unit ]     (column-wise 3×3 rotation matrix)

d = [ L · cos(π − θ),
      L · cos(τ) · sin(π − θ),
      L · sin(τ) · sin(π − θ) ]

D = C + M · d
```

The procedure is applied sequentially:

1. **Backbone chain**: N → Cα → C for residue 0, then carbonyl O, then N' → Cα' → C' for residue 1, and so on, using peptide bond geometry (ω = π for trans, 0 for cis-Pro).
2. **Side chains**: Each atom in `SIDE_CHAIN_TOPO` is placed using NERF from its parent. The first side-chain atom (Cβ) is placed using the N–Cα–C bisector plane rather than a torsion, to correctly reproduce the tetrahedral geometry at Cα.

Bond lengths and angles are taken from **Engh & Huber (1991)**, the standard reference for X-ray protein structure refinement parameters. A small numerical guard `+ 1e-9` prevents division by zero for degenerate geometries.

**Outputs of `build_full_structure`:**

- `coords` — `ndarray` of shape `(N_atoms, 3)`, full Cartesian coordinates in Å
- `labels` — list of `(res_id, atom_name, element)` tuples, one per atom
- `bonds` — list of `(idx_a, idx_b)` tuples defining covalent connectivity

### Three-Stage Optimisation

Each folding replica runs through a fixed **curriculum** of three optimisation stages that progressively relax the constraints, mimicking a coarse-to-fine annealing strategy:

| Stage | Optimiser | γ (surface tension) | End-to-end λ | Purpose |
|-------|-----------|--------------------|-----------|----|
| 1 — Collapse | COBYLA | 15.0 | 50.0 | Rapid hydrophobic collapse to a globular state |
| 2 — Refine | SLSQP | 15.0 | 50.0 | Fix local steric clashes while preserving global shape |
| 3 — Relax | SLSQP | 5.0 | 5.0 | Release constraints; H-bonds and electrostatics define the fold |

**COBYLA** (Constrained Optimisation BY Linear Approximations) is derivative-free and well-suited to the noisy, non-convex landscape of Stage 1, where the protein undergoes large conformational changes. **SLSQP** (Sequential Least-SQuares Programming) uses numerically estimated gradients for finer convergence in Stages 2 and 3.

The `LandscapeTracker` object records the energy value at every function evaluation and the iteration index of each stage boundary. This data is what `plot_energy_landscape` visualises.

---

## Physics Engine

### Force Fields

The custom QTF backend uses a single Amber-style partial-charge parameterization:

| Custom parameterization | Backbone N charge | C=O charges | Notes |
|-------------|------------------|------------|-------|
| AMBER ff14SB approx. | −0.42 | +0.60 / −0.57 | Very polar carbonyl; Cα neutral (0.00) |

The custom parameterization uses:
- A common set of ionic/charged-group charges: carboxylate oxygens, guanidinium nitrogens, lysine NZ, etc.
- The Kyte–Doolittle hydrophobicity scale (independent of partial charges).
- Bondi van der Waals radii for heavy atoms.

**Histidine special-casing:** In all force fields, His ND1 and NE2 are assigned −0.4 charges (neutral imidazole tautomer) regardless of the base force field's NE2 value, since NE2 is ambiguous (amide in Gln vs amine in neutral His).

**Terminal neutralisation:** N and C termini have their backbone charges zeroed to approximate N-acetyl / C-amide capping used in experimental short peptide studies.

### Energy Terms

The total energy `E_total` is evaluated at every optimiser step and is a sum of ten terms:

#### 1. End-to-End Constraint

A slack-band harmonic potential keeps the N-terminal and C-terminal Cα atoms near a
**length-aware target distance** that scales with sequence length:

```
target = 4.5 + 0.40 · max(0, N − 5)   (Å)
slack  = 1.5 + 0.05 · N               (Å)

deviation = max(0, |d(Cα_first, Cα_last) − target| − slack)
E_constraint = λ · deviation²
```

`λ` = 50.0 in Stages 1–2, 5.0 in Stage 3 (released to let the protein explore freely).
End-to-end behavior is configured through recipe score options.

#### 2. Implicit Solvent / Hydrophobic Effect (SASA Approximation)

Hydrophobic carbon atoms (on residues Ala, Val, Leu, Ile, Met, Phe, Trp, Pro, Cys) are penalised for being solvent-exposed. A sigmoid-weighted neighbour count approximates the burial fraction:

```
w(d)   = σ(−(d − 6.0))           # soft count of neighbours within ~6 Å
b      = clip(Σ_j w(dᵢⱼ) / 15.0 ,  0, 1)
E_sasa += γ · 30.0 · (1 − b)     # energy penalty for exposure
```

`γ` (the "surface tension") = 15.0 in Stages 1–2 (drives collapse), 5.0 in Stage 3 (allows breathing).

#### 3. Explicit Hydrogen Bonding (N–H···O=C)

For each backbone nitrogen atom, an approximate amide hydrogen position `H` is constructed by bisecting the inward vectors to the adjacent C and Cα:

```
v_h = −(v_NC_unit + v_NCα_unit)
H   = N + 1.01 · v_h / ‖v_h‖
```

For each eligible carbonyl oxygen O (from a residue at least 2 away), the potential is activated when:

- **Distance**: d(H, O) < 3.5 Å
- **Linearity**: cos(∠H–N···O) < −0.4

The energy contribution is:

```
E_hbond = −25.0 · exp(−(d_HO − 2.0)² / 0.5) · (|cos θ| − 0.4) · 2.0
```

The strength −25.0 is deliberately large to strongly drive secondary-structure formation during the collapse stage. All eligible N–O pairs are evaluated in one vectorised pass.

#### 4. Electrostatics (Coulomb's Law)

Pairwise Coulomb interactions between all atom pairs separated by at least 2 residues, with non-negligible partial charges (|q_i · q_j| > 10⁻⁴):

```
E_elec = 332.0637 · qᵢ · qⱼ / (4.0 · rᵢⱼ)
```

The prefactor 332.0637 is the Coulomb constant in kcal mol⁻¹ Å e⁻² and 4.0 is a
uniform implicit-solvent dielectric constant (ε ≈ 4 is the standard choice for
buried environments in coarse force fields; see Warshel & Russell, Q Rev Biophys
1984). The computation is vectorised using NumPy outer products and boolean masking.

#### 5. Disulfide Bonds (Cys–Cys SG Bridges)

When two or more cysteine residues are present, all SG–SG pairs are monitored:

- **Attractive term** (bond formation when d < 3.0 Å):
  ```
  E_bond = −25.0 · exp(−(d_SG − 2.05)² / 0.5)
  ```
  The equilibrium length 2.05 Å matches crystallographic disulfide bonds.

- **Valence penalty** (prevents a single SG from bonding to two partners):
  ```
  saturation = Σⱼ bond_strength(dᵢⱼ)
  E_penalty  = +40.0 · max(0, saturation − 1.0)²
  ```

#### 6. Sterics (Bond-Graph VdW Repulsion)

Steric clashes between heavy-atom pairs are penalised with a softened repulsion term.
Non-bonded pairs are determined via a BFS over the covalent bond graph:

- **1-2 pairs** (directly bonded) and **1-3 pairs** (two bonds apart) are fully excluded.
- **1-4 pairs** (three bonds apart) contribute with a scale factor of **0.35** (AMBER convention).
- All other pairs are included at full weight.

```
term = (σᵢ + σⱼ) / (rᵢⱼ + 0.1))¹²    when rᵢⱼ < σᵢ + σⱼ
```

where σ is the van der Waals radius (Bondi 1964). For extreme overlaps (term > 50), a logarithmic cap prevents gradient explosion during the early collapse stage:

```
term = 50 + log(term − 49)   if term > 50
```

The contribution is scaled by 0.1 to balance against the other energy terms.

#### 7. Ramachandran Bias

Gaussian energy wells in backbone (φ, ψ) dihedral space guide residues towards the two most populated regions of the Ramachandran plot:

- **α-helix well**: φ_target = −1.0 rad (−57°), ψ_target = −0.8 rad (−47°)
- **β-sheet well**: φ_target = −2.3 rad (−132°), ψ_target = +2.4 rad (+138°)
- **Forbidden region penalty**: φ_target = −2.0 rad, ψ_target = +1.0 rad

```
d_helix    = (φ − φ_helix)² + (ψ − ψ_helix)²
d_sheet    = (φ − φ_sheet)² + (ψ − ψ_sheet)²
d_forbid   = (φ − φ_forbid)² + (ψ − ψ_forbid)²

E_rama = −3.0·exp(−d_helix/0.6)
       − 3.0·exp(−d_sheet/0.6)
       + 5.0·exp(−d_forbid/1.0)
```

**Glycine special case**: allowed in both L and D configurations, so four wells are used (helix, sheet, and their mirror images across the origin). The minimum distance to any well is used.

#### 8. Rotamer Preferences (Side-Chain Torsions)

Side-chain χ₁ angles are softly guided towards crystallographically observed preferred rotamer states:

| Residue class | Preferred χ₁ states | Expression |
|---------------|---------------------|------------|
| Val, Ile, Thr | trans (π), gauche⁺ (−60°) | Gaussian wells at ±π and −1.047 |
| Pro | ring pucker (±0.5 rad) | Harmonic wells at ±0.5 |
| Phe, Tyr, Trp, His | trans (π), gauche⁺ (−60°) | Gaussian wells (softer weight −2.0) |
| All others | 3-fold periodic | `1 + cos(3χ₁)` well |

#### 9. Aromatic π–π Stacking

Ring centroids and normal vectors are computed for Phe, Tyr, and Trp. Two stacking geometries are recognised and rewarded:

| Mode | Centroid distance | Normal vector alignment | ΔE |
|------|------------------|------------------------|----|
| T-shaped (edge-to-face) | 4.5–6.0 Å | \|cos θ\| < 0.3 | −4.0 |
| Parallel (face-to-face) | 3.4–4.5 Å | \|cos θ\| > 0.8 | −5.0 |

The energy is `ΔE · exp(−(d − d_opt)²)` centred on the optimal distances (5.0 Å and 3.8 Å respectively).

#### 10. Geometry Integrity

Structural constraints penalise three types of physically impossible geometry.
All penalties use a **Huber loss** — quadratic for small deviations (smooth gradient
signal for the optimiser) and linear for large deviations (prevents a single
badly-placed atom from dominating the landscape):

```
huber(x, δ) = x²            if |x| ≤ δ
            = 2δ|x| − δ²    if |x| > δ      (δ = 1.0 Å by default)
```

The three terms are:

- **Proline ring closure**: The CD–N distance must be 1.47 ± δ Å. Penalty: `+50 · huber(d − 1.47)`.
- **L-chirality**: The signed volume `(N−Cα) × (C−Cα) · (Cβ−Cα)` must be > 1.0. Penalty: `+50 · huber(max(0, 1 − volume))`.
- **Peptide planarity (ω angle)**: The Cα–C–N'–Cα' dihedral must be near 180° (or 0° for Xaa-Pro). Penalty: `+20 · huber(twist)` when the twist exceeds 0.05.

---

## Ensemble Folding

### Initialisation Strategy

Each replica's starting point is determined by a `prime_strategy`:

| Strategy | Description |
|----------|-------------|
| `"random"` (default) | `scout_attempts` random parameter vectors sampled from [−0.8, 0.8]^P; the lowest-energy one is used |
| `"helix"` | Circuit is pre-optimised to output canonical α-helix torsions (φ ≈ −60°, ψ ≈ −45°) before the main optimisation |
| `"sheet"` | Circuit is pre-optimised to output β-sheet torsions (φ ≈ −135°, ψ ≈ 135°) |
| `"mixed"` | Replicas are split evenly between helix and sheet priming |

Seeds are derived deterministically from the protein sequence using SHA-256:

```python
base_seed     = int(sha256(sequence.encode("utf-8")).hexdigest(), 16) % 2**32
replica_seed  = base_seed + seed_offset + i       # i = 0, 1, …, n_runs − 1
```

For SLURM arrays where every task runs a single replica, pass the array task ID
as `--seed-offset` so each job gets a distinct deterministic scouting seed.

This ensures that:
1. Every run on the same sequence always explores the **same set of starting points** (reproducibility).
2. Different replicas use different seeds, so they start from **genuinely different** regions of parameter space (diversity).
3. Results are **fully reproducible** given the same sequence, `n_runs`, and `scout_attempts`.

### Running an Ensemble

```python
from qtf import QuantumBiophysicsFolder, EnsembleFoldingManager

folder = QuantumBiophysicsFolder("ACDEFGHIKLMNPQRSTVWY")
manager = EnsembleFoldingManager(folder)

manager.run_ensemble(
    n_runs=10,                # run 10 independent replicas
    max_iter=3000,            # maximum optimiser iterations per stage, per replica
    scout_attempts=100,       # basin-hopping breadth (higher = better start but slower)
    prime_strategy="random",  # "random" | "helix" | "sheet" | "mixed"
)

# Retrieve results sorted by ascending final energy
results = manager.get_results()

# Retrieve top-k lowest-energy results
top3 = manager.select_top(top_k=3)

# Retrieve top 20% lowest-energy results
top_frac = manager.select_top(top_frac=0.2)
```

Each element of `results` is a `dict` with the following keys:

| Key | Type | Description |
|-----|------|-------------|
| `"id"` | int | Replica index (0-based) |
| `"seed"` | int | Random seed used for this replica |
| `"energy"` | float | Final force-field energy after Stage 3 |
| `"coords"` | ndarray (N_atoms, 3) | All-atom Cartesian coordinates |
| `"labels"` | list | `[(res_id, atom_name, element), …]` |
| `"bonds"` | list | `[(idx_a, idx_b), …]` covalent connectivity |
| `"params"` | ndarray (n_params,) | Final circuit parameter vector |
| `"tracker"` | LandscapeTracker | Full energy history + stage markers |

**Performance guidance:**

| Sequence length | Recommended settings | Approximate wall time (CPU) |
|----------------|---------------------|-----------------------------|
| ≤ 5 residues | `n_runs=3`, `max_iter=500`, `scout_attempts=20` | Minutes |
| 6–10 residues | `n_runs=5`, `max_iter=2000`, `scout_attempts=50` | ~1 hour |
| 11–20 residues | `n_runs=10`, `max_iter=4000`, `scout_attempts=100` | Several hours |

The most expensive step is the Qiskit statevector simulation inside `_get_angles()`, which scales as O(2^n_qubits). For sequences with up to ~20 residues, n_qubits is typically 5–6, keeping simulation tractable on a laptop.

---

## Ranking and Analysis

### EnsembleRanking Statistics

`EnsembleRanking.from_ensemble()` ingests all replica results and computes a comprehensive `stats_df` pandas `DataFrame` with one row per replica and the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `rank_energy` | int | Rank by ascending final energy (1 = lowest = best) |
| `replica_id` | int | Replica index matching `results[i]["id"]` |
| `seed` | int | Random seed used for this replica |
| `energy` | float | Final force-field energy (lower is better) |
| `end_to_end_dist` | float | Euclidean distance Cα[0] → Cα[−1], in Å |
| `radius_of_gyration` | float | Rg = √(mean‖Cα − centroid‖²), in Å |
| `rmsd_vs_gt` | float | Kabsch RMSD of Cα trace vs ground truth (NaN if no GT provided) |
| `rank_rmsd` | int | Rank by ascending RMSD vs GT (NaN if no GT) |
| `mean_rmsd_vs_ensemble` | float | Mean Kabsch RMSD of this replica vs every other replica, in Å |
| `is_best_energy` | bool | `True` for the replica with the lowest final energy |
| `is_best_rmsd` | bool | `True` for the replica with the lowest RMSD vs GT (`False` if no GT) |
| `is_ensemble_centroid` | bool | `True` for the replica with the minimum `mean_rmsd_vs_ensemble` |

The DataFrame is sorted by `rank_energy` ascending (best energy first).

**Practical usage:**

```python
df = ranking.stats_df

Export # ── ─────────────────────────────────────────
df.to_csv("ensemble_stats.csv", index=False)

# ── Inspect top 3 by energy ──────────────────────────
print(df.nsmallest(3, "energy")[["rank_energy", "replica_id", "energy", "rmsd_vs_gt"]])

# ── Correlation between energy and accuracy ──────────────────────────────────
print(df[["energy", "rmsd_vs_gt"]].corr())

# ── Filter near-native structures (within 2 Å of GT) ────────────────────────
near_native = df[df["rmsd_vs_gt"] < 2.0]
print(f"{len(near_native)} / {len(df)} replicas within 2 Å of GT")

# ── Compare compact vs extended structures ───────────────────────────────────
print(df[["radius_of_gyration", "end_to_end_dist"]].describe())

# ── Access the full pairwise matrix RMSD ───────────────────────────
print(ranking.pairwise_rmsd_matrix)   # shape (n_runs, n_runs)
```

**Accessing individual best structures:**

```python
# Best by energy — always available
best_e = ranking.best_by_energy
print(f"Best energy: {best_e['energy']:.4f}")
all_atom_coords = best_e["coords"]   # ndarray, shape (N_atoms, 3)
atom_labels     = best_e["labels"]   # [(res_id, atom_name, element), …]
energy_history  = best_e["tracker"].history   # list of floats

# Best by RMSD vs ground truth — None if no GT was provided
best_r = ranking.best_by_rmsd
if best_r is not None:
    print(f"Best RMSD replica energy: {best_r['energy']:.4f}")
    print(f"This replica's RMSD:      "
          f"{df[df['replica_id'] == best_r['id']]['rmsd_vs_gt'].item():.3f} Å")
```

**Important note:** The best-energy and best-RMSD picks are **independent** and will frequently be different replicas. Low energy does not guarantee low RMSD, and the lowest-RMSD structure may not have the lowest energy. When they coincide, both badges appear on the same trace in the visualisations. When a ground truth is not available (the common case), only the lowest-energy structure is returned.

### Convergence Assessment

`ranking.convergence` is computed from the all-vs-all pairwise Kabsch RMSD matrix of Cα traces:

```python
conv = ranking.convergence
# Keys:
#   "avg_pairwise_rmsd" : float  — mean RMSD over all unique pairs
#   "max_pairwise_rmsd" : float  — maximum RMSD among all pairs
#   "min_nonzero_rmsd"  : float  — smallest non-zero RMSD (most similar pair)
#   "verdict"           : str    — "STABLE" | "FLEXIBLE" | "UNSTABLE"
```

**Verdict thresholds:**

| Verdict | avg pairwise RMSD | Interpretation |
|---------|------------------|----------------|
| `STABLE` | < 2.0 Å | All replicas converged to the same basin; prediction is reliable |
| `FLEXIBLE` | 2.0–4.5 Å | Core fold is conserved but loops or termini vary; moderate confidence |
| `UNSTABLE` | > 4.5 Å | No dominant conformational basin; consider increasing `n_runs` or `max_iter` |

The ensemble centroid (`is_ensemble_centroid`) is the structure that minimises the sum of pairwise RMSD to all others — it is the most "representative" structure of the ensemble. Note that this is **not** necessarily the best prediction; it is simply the one closest to the ensemble mean.

The full pairwise RMSD matrix (shape `(n_runs, n_runs)`, diagonal = 0) is accessible as `ranking.pairwise_rmsd_matrix` for custom downstream analysis (e.g., hierarchical clustering, MDS projection).

### Kabsch RMSD

The Kabsch algorithm finds the rotation matrix **R** that minimises the RMSD between two point sets **P** and **Q** after centring. Given centred versions P_c = P − mean(P) and Q_c = Q − mean(Q):

```
H          = P_c.T @ Q_c
V, S, Wᵀ  = SVD(H)
d          = sign(det(V) · det(Wᵀ))        # ±1: correct for reflections
R          = V · diag(1, 1, d) · Wᵀ        # proper rotation (det = +1)

RMSD       = sqrt(mean(‖P_c @ R − Q_c‖²))
P_aligned  = P_c @ R + mean(Q)             # P superimposed on Q
```

The reflection correction ensures that **R** is a proper rotation (det = +1) rather than an improper rotation / reflection. This is essential when comparing mirror-image conformations.

```python
from qtf.analysis import kabsch_rmsd
import numpy as np

rmsd, pred_aligned = kabsch_rmsd(pred_ca, true_ca)
print(f"Backbone Cα RMSD: {rmsd:.3f} Å")
# pred_aligned is pred_ca rotated and translated to best overlap with true_ca
```

---

## Visualisation

All three visualisation functions return `plotly.graph_objects.Figure` objects. They can be:

- **Displayed interactively** with `.show()` — opens a browser tab, or renders inline in Jupyter/JupyterLab with the plotly extension installed.
- **Exported to HTML** (self-contained, shareable):
  ```python
  fig.write_html("output.html")
  ```
- **Exported to static images** (requires the `kaleido` package: `pip install kaleido`):
  ```python
  fig.write_image("output.png", width=1200, height=800)
  fig.write_image("output.svg")
  fig.write_image("output.pdf")
  ```

### 3-D Structure Viewer

```python
from qtf.visualization import plot_structure

fig = plot_structure(
    ranking,
    ground_truth_ca=true_ca,              # optional (N_residues, 3) array
    ca_label="CA",                         # atom name identifying Cα (default)
    show_all=True,                         # False = show only the two best picks
    title="Chignolin Ensemble vs 5AWL",
)
fig.show()
```

**What is rendered:**

| Element | Colour | Opacity | Description |
|---------|--------|---------|-------------|
| Best-energy replica | Green (`#2ecc71`) | 1.0 | Lowest final energy; labelled ★ |
| Best-RMSD replica | Red (`#e74c3c`) | 1.0 | Lowest RMSD vs GT; labelled ★ |
| Both in same replica | Purple (`#9b59b6`) | 1.0 | Single replica wins on both metrics |
| Other replicas | Grey (`#95a5a6`) | 0.35 | All remaining predictions (when `show_all=True`) |
| Ground truth | Dark navy (`#2c3e50`) | 1.0 | Dashed line; experimental reference |

When a ground truth is provided, all predicted Cα traces are **Kabsch-aligned** to it before rendering, so all structures are shown in the same reference frame. Interactive hover tooltips display energy, Rg, end-to-end distance, and RMSD vs GT for each trace.

### Energy Landscape

```python
from qtf.visualization import plot_energy_landscape

fig = plot_energy_landscape(
    ranking,
    replica_ids=[0, 1, 2],       # optional subset; None = all replicas
    clip_range=(-500.0, 1500.0), # clip extreme values in the display
    title="Optimisation Landscape",
)
fig.show()
```

**What is rendered:**

- **Energy trace** for each selected replica: energy value (y-axis) vs function evaluation step (x-axis).
- The **best-energy replica** is drawn at full opacity in green; all other replicas are drawn faintly (opacity 0.5) in dark navy.
- **Stage boundary markers**: vertical dashed lines at the exact iteration where Stage 1 ends, Stage 2 ends, and Stage 3 begins. Stage 1 = red, Stage 2 = orange, Stage 3 = blue.
- Stage names are not duplicated if multiple replicas share the same stage boundary (since stage boundaries shift based on when the optimiser terminates each stage).
- Hover tooltips show the step index, energy value, final energy, and — if a ground truth was provided — the RMSD vs GT of that replica.

**Interpreting the landscape:**

- A **steep initial drop** in Stage 1 is healthy — it indicates successful hydrophobic collapse.
- **Oscillations** in Stages 2–3 are normal; SLSQP explores locally.
- A **flat final energy** across replicas (all converging to similar values) combined with a `STABLE` convergence verdict is the ideal outcome.
- Replicas with very different final energies suggest the optimiser is stuck in different local minima; increasing `n_runs` or `max_iter` may help.

### Ranking Dashboard

```python
from qtf.visualization import plot_ranking

fig = plot_ranking(
    ranking,
    title="Ensemble Ranking — Chignolin 5AWL",
)
fig.show()
```

**What is rendered:**

- **Top panel — Bar chart**: Final energy of each replica as a vertical bar. Colours match the structure viewer (green = best energy, red = best RMSD, grey = other, purple = both). Interactive hover shows all key statistics.
- **Bottom panel — Statistics table**: Full per-replica table with colour-highlighted rows (green row = best energy, red row = best RMSD, white = other). Displayed columns include rank, energy, Rg, end-to-end distance, RMSD vs GT, mean pairwise RMSD vs ensemble, and the three boolean flags.
- **Footer annotation**: Convergence verdict and average/maximum pairwise RMSD across all replicas.

---

## PDB Utilities

### Saving a Prediction

```python
from qtf.utils import save_pdb

save_pdb(
    coords   = best_e["coords"],      # all-atom coords, shape (N_atoms, 3)
    labels   = best_e["labels"],      # [(res_id, atom_name, element), …]
    sequence = folder.sequence,       # single-letter amino acid string
    filename = "prediction.pdb",      # output path
    energy   = best_e["energy"],      # stored in REMARK 1
)
```

The output is a valid **PDB ATOM record file** with:
- Chain A
- Occupancy 1.00, B-factor 0.00
- Energy stored in `REMARK   1 ENERGY: {value:.3f}`

The file is directly viewable in PyMOL, UCSF Chimera/ChimeraX, VMD, Mol*, or any PDB-compatible structure viewer.

### Loading Ground-Truth Coordinates

```python
from qtf.utils import get_ground_truth_backbone

# Downloads 5AWL.pdb from RCSB on first call; cached thereafter
true_ca = get_ground_truth_backbone("5AWL", cache_dir="./pdb_cache")
# Returns ndarray, shape (N_residues, 3) — Cα atoms only, first MODEL
```

**Behaviour:**
1. Constructs the filename `{cache_dir}/{PDB_ID}.pdb`.
2. If the file does not exist, downloads it from `https://files.rcsb.org/download/{PDB_ID}.pdb`.
3. Parses only lines starting with `ATOM` where columns 12–15 contain `CA`.
4. Stops at the first `ENDMDL` record (picks the first NMR model or the single X-ray model).
5. Returns the Cα coordinates as a float64 ndarray.

**Tip:** For NMR ensembles with many models, this always returns Model 1. For proper NMR ensemble analysis, download the file manually and parse the model you need.

When aligning predictions against a ground truth, note that the sequence lengths may differ (e.g., if the experimental structure has additional residues). Both `EnsembleRanking.from_ensemble` and `plot_structure` truncate to the shorter of the two before computing RMSD or alignment.

### Computing Structural Metrics

```python
from qtf.utils import calculate_physics_metrics
import numpy as np

ca_coords = np.array(...)   # shape (N_residues, 3)
end_to_end, rg = calculate_physics_metrics(ca_coords)
print(f"End-to-end distance : {end_to_end:.2f} Å")
print(f"Radius of gyration  : {rg:.2f} Å")
```

**Definitions:**
- `end_to_end_dist` = ‖coords[0] − coords[−1]‖  (Euclidean distance between first and last Cα)
- `radius_of_gyration` = √(mean‖coords − centroid‖²)  (mass-weighted RMS distance from geometric centre, assuming uniform mass)

---

## API Reference

The full auto-generated API reference is available at
**[https://cumbof.github.io/QTF/api](https://cumbof.github.io/QTF/api)**
or locally by building the docs:

```bash
pip install -e ".[docs]"
mkdocs serve
# Open http://127.0.0.1:8000/api
```

Every public class, method, and function is documented with NumPy-style docstrings
and rendered by `mkdocstrings`. The reference covers:

- **`qtf`** — Package-level exports (`QuantumBiophysicsFolder`,
  `EnsembleFoldingManager`, `LandscapeTracker`)
- **`qtf.core`** — `folder.py`, `ensemble.py`, `tracker.py`
- **`qtf.analysis`** — `ranking.py`, `stability.py`, `panel.py`
- **`qtf.visualization`** — `plots.py`
- **`qtf.utils`** — `pdb.py`, `workflow.py`, `aer_sim.py`, `accelerate.py`, `gromacs.py`

The hand-written API tables formerly in this section have been retired in favour
of the auto-generated docs, which always stay in sync with the source code.

---

## Reproducibility

QTF is designed for **exact reproducibility** given the same sequence and run parameters:

1. **Sequence-derived base seed**: `base_seed = int(SHA-256(sequence)) % 2^32` — the same sequence always produces the same base seed.
2. **Per-replica offsets**: `seed_i = base_seed + i` — each replica uses a unique but deterministic seed.
3. **Modern NumPy RNG**: `numpy.random.default_rng(seed)` is used throughout (not the legacy `np.random.seed` global state).
4. **No global mutable state**: each `QuantumBiophysicsFolder` instance is fully self-contained.

```python
# These two managers will produce byte-identical results
folder1 = QuantumBiophysicsFolder("ACGT")
folder2 = QuantumBiophysicsFolder("ACGT")

m1, m2 = EnsembleFoldingManager(folder1), EnsembleFoldingManager(folder2)
m1.run_ensemble(n_runs=3, max_iter=500)
m2.run_ensemble(n_runs=3, max_iter=500)

for i in range(3):
    assert m1.results[i]["energy"] == m2.results[i]["energy"]  # ✓
```

## Logging

QTF uses Python's standard `logging` module. **No output is printed to stdout by default** — you must add a handler:

```python
import logging

# Recommended: show INFO-level progress
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
```

**Log levels by component:**

| Logger | INFO messages | DEBUG messages |
|--------|--------------|----------------|
| `qtf.core.folder` | Stage transitions, energy at each stage end | Basin-hopping seed and best-start energy |
| `qtf.core.ensemble` | Replica start/finish, final energy per replica | — |
| `qtf.utils.aer_sim` | Aer backend initialisation (GPU/CPU) | Backend creation failures |
| `qtf.analysis.ranking` | — | — |
| `qtf.visualization` | — | — |

Parallel replicas (via `ProcessPoolExecutor`) inherit the root logger
configuration and write to the same terminal; set `max_workers=1` for
sequential in-process execution.

```python
# Show only errors from QTF (suppress progress):
logging.getLogger("qtf").setLevel(logging.ERROR)

# Show everything including per-basin-hop evaluations:
logging.getLogger("qtf").setLevel(logging.DEBUG)

# Fine-grained control:
logging.getLogger("qtf.core.folder").setLevel(logging.DEBUG)
logging.getLogger("qtf.core.ensemble").setLevel(logging.INFO)
```

---

## References

1. **Hydrophobicity Scale**:
   Kyte, J., & Doolittle, R. F. (1982). A simple method for displaying the hydropathic character of a protein. *Journal of Molecular Biology*, 157(1), 105–132. https://doi.org/10.1016/0022-2836(82)90515-0

2. **AMBER Force Field**:

5. **van der Waals Radii**:
   Bondi, A. (1964). van der Waals volumes and radii. *Journal of Physical Chemistry*, 68(3), 441–451. https://doi.org/10.1021/j100785a001

6. **Bond and Angle Parameters**:
   Engh, R. A., & Huber, R. (1991). Accurate bond and angle parameters for X-ray protein structure refinement. *Acta Crystallographica Section A*, 47(4), 392–400. https://doi.org/10.1107/S0108767391001071

7. **NERF Algorithm**:
   Parsons, J., Holmes, J. B., Rojas, J. M., Tsai, J., & Strauss, C. E. M. (2005). Practical conversion from torsion space to Cartesian space for in silico protein synthesis. *Journal of Computational Chemistry*, 26(10), 1063–1068. https://doi.org/10.1002/jcc.20237

8. **Kabsch Algorithm**:
   Kabsch, W. (1978). A discussion of the solution for the best rotation to relate two sets of vectors. *Acta Crystallographica Section A*, 34(5), 827–828. https://doi.org/10.1107/S0567739478001680

9. **EfficientSU2 Ansatz / VQE**:
   Kandala, A., et al. (2017). Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets. *Nature*, 549, 242–246. https://doi.org/10.1038/nature23879

10. **Ramachandran Plot**:
    Ramachandran, G. N., Ramakrishnan, C., & Sasisekharan, V. (1963). Stereochemistry of polypeptide chain configurations. *Journal of Molecular Biology*, 7(1), 95–99. https://doi.org/10.1016/S0022-2836(63)80023-6

11. **COBYLA Optimiser**:
    Powell, M. J. D. (1994). A direct search optimization method that models the objective and constraint functions by linear interpolation. In *Advances in Optimization and Numerical Analysis* (pp. 51–67). Kluwer Academic Publishers.

12. **SLSQP Optimiser**:
    Kraft, D. (1988). *A software package for sequential quadratic programming*. DFVLR-FB 88-28, Deutsche Forschungs- und Versuchsanstalt für Luft- und Raumfahrt.

---

## License

QTF is released under the **MIT License** — see [LICENSE](LICENSE) for the full text.
