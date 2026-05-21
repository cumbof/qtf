# Quick Start

This page walks through a complete prediction run for **Chignolin** (`YYDPETGTWY`,
PDB: [5AWL](https://www.rcsb.org/structure/5AWL)) — a well-characterised 10-residue
β-hairpin mini-protein and a popular benchmark for small-peptide folding methods.

---

## 1. Import the package

```python
import logging
logging.basicConfig(level=logging.INFO)   # optional — enables progress messages

from qtf import QuantumBiophysicsFolder, EnsembleFoldingManager
from qtf.analysis import EnsembleRanking
from qtf.visualization import plot_structure, plot_energy_landscape, plot_ranking
from pheat.pdbio import load_pdb, write_pdb
```

---

## 2. Initialise the folder

```python
folder = QuantumBiophysicsFolder(
    sequence="YYDPETGTWY",   # Chignolin — a well-studied mini-protein
    force_field="amber",     # "charmm" (default) | "amber" | "opls"
)

print(f"Residues      : {folder.n_residues}")
print(f"DoF (angles)  : {folder.total_angles}")
print(f"Qubits        : {folder.n_qubits}")
print(f"Circuit params: {folder.n_params}")
```

The constructor builds the quantum circuit, pre-computes the topology cache for
vectorised energy evaluation, and derives every atom's partial charge, van der Waals
radius, and hydrophobicity class.

---

## 3. Run an ensemble

```python
manager = EnsembleFoldingManager(folder)
manager.run_ensemble(
    n_runs=5,            # number of independent replicas
    max_iter=2000,       # max optimiser iterations per stage per replica
    scout_attempts=50,   # basin-hopping samples evaluated before each full run
)
```

Each replica:

1. Draws `scout_attempts` random parameter vectors and picks the lowest-energy one
   (basin-hopping initialisation).
2. Runs three successive optimisation stages — Collapse (COBYLA) → Refine (SLSQP)
   → Relax (SLSQP) — each stage starting from the result of the previous one.

See [Core Concepts](concepts.md#three-stage-optimisation) for details.

---

## 4. Fetch the ground-truth structure

```python
# Downloads 5AWL.pdb from RCSB on first call; reuses the cached file thereafter.
# cache_dir defaults to a process-wide temp directory; pass an explicit path
# (e.g. "./pdb_cache") if you want a persistent cache in the project.
from pheat.pdbio import load_pdb_ca_by_id

true_ca = load_pdb_ca_by_id("5AWL")
print(f"Ground truth: {len(true_ca)} Cα atoms")
```

This step is **optional**. Omit `ground_truth_ca` in the next step if you have no
experimental reference.

---

## 5. Build the ranking

```python
ranking = EnsembleRanking.from_ensemble(
    manager.get_results(),
    ground_truth_ca=true_ca,   # omit entirely if no reference is available
)
print(ranking.summary())
```

`from_ensemble` computes:

- Per-replica RMSD vs ground truth (Kabsch algorithm)
- Radius of gyration and end-to-end distance for every predicted structure
- All-vs-all pairwise RMSD matrix and convergence verdict

---

## 6. Access the best structures

```python
best_e = ranking.best_by_energy   # always available
best_r = ranking.best_by_rmsd     # None if no ground truth was given

print(f"Best energy : {best_e['energy']:.4f}")
if best_r:
    df = ranking.stats_df
    rmsd = df[df["replica_id"] == best_r["id"]]["rmsd_vs_gt"].item()
    print(f"Best RMSD   : {rmsd:.3f} Å  (replica {best_r['id']})")
```

---

## 7. Export the prediction as PDB

```python
save_pdb(
    best_e["coords"],
    best_e["labels"],
    folder.sequence,
    filename="best_energy.pdb",
    energy=best_e["energy"],
)
```

The resulting file can be opened in PyMOL, UCSF ChimeraX, VMD, Mol*, or any
PDB-compatible viewer.

---

## 8. Interactive visualisations

=== "3-D Backbone Overlay"

    ```python
    plot_structure(ranking, ground_truth_ca=true_ca).show()
    ```

    All predicted Cα traces are Kabsch-aligned to the experimental structure.
    The best-energy replica is highlighted in green, best-RMSD in red.

=== "Energy Landscape"

    ```python
    plot_energy_landscape(ranking).show()
    ```

    Energy vs function-evaluation step for every replica, with vertical markers at
    stage boundaries.

=== "Ranking Dashboard"

    ```python
    plot_ranking(ranking).show()
    ```

    Interactive bar chart and statistics table for all replicas.

All figures are `plotly.graph_objects.Figure` objects and can be saved:

```python
fig.write_html("output.html")          # self-contained HTML
fig.write_image("output.png")          # static PNG (requires kaleido)
```

---

## Without a ground-truth reference

```python
ranking = EnsembleRanking.from_ensemble(manager.get_results())
# ranking.best_by_rmsd       → None
# stats_df["rmsd_vs_gt"]     → all NaN
# best prediction still available via ranking.best_by_energy
print(ranking.best_by_energy["energy"])
```

---

!!! tip "Interactive notebook"
    The repository ships with `QTF.ipynb`, which runs this exact example end-to-end
    using the installed package. Open it with `jupyter notebook QTF.ipynb`.
