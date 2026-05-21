# API Reference

Full auto-generated API documentation for all public classes and functions.

---

## Core

### QuantumBiophysicsFolder

::: qtf.core.folder.QuantumBiophysicsFolder

---

### EnsembleFoldingManager

::: qtf.core.ensemble.EnsembleFoldingManager

---

### LandscapeTracker

::: qtf.core.tracker.LandscapeTracker

---

## Analysis

### EnsembleRanking

::: qtf.analysis.ranking.EnsembleRanking

---

### kabsch_rmsd

Lives in PHEAT — see `pheat.geometry.kabsch_rmsd` / `kabsch_align`.

---

### StabilityAnalyzer

::: qtf.analysis.stability.StabilityAnalyzer

---

## Visualisation

### plot_structure

::: qtf.visualization.plots.plot_structure

---

### plot_energy_landscape

::: qtf.visualization.plots.plot_energy_landscape

---

### plot_ranking

::: qtf.visualization.plots.plot_ranking

---

## Utilities

PDB I/O, Kabsch alignment, RMSD, radius of gyration, and ground-truth fetching live in PHEAT:

- `pheat.pdbio.write_pdb(structure, path, *, remarks=None)` — replaces the former `qtf.utils.pdb.save_pdb`. Pair with `qtf.structures.qtf_structure_to_pheat` to convert QTF `(coords, labels)` into a `HeavyAtomStructure`.
- `pheat.pdbio.load_pdb_ca_by_id(pdb_id, cache_dir=None)` — replaces the former `qtf.utils.pdb.get_ground_truth_backbone`. Returns an `(N, 3)` numpy array of Cα coordinates. Lower-level helpers `pheat.pdbio.load_pdb_by_id` (full `HeavyAtomStructure`) and `pheat.pdbio.fetch_pdb` (path only) are also available. Supports `format="pdb"` or `"cif"`.
- `pheat.geometry.radius_of_gyration(coords)` — replaces the Rg part of the former `qtf.utils.pdb.calculate_physics_metrics`. End-to-end distance is a one-line `numpy.linalg.norm(coords[0] - coords[-1])`.
- `pheat.metrics.pairwise_rmsd_matrix(structures)` and `pheat.metrics.ensemble_rmsd_stats(matrix)` — replace `StabilityAnalyzer.pairwise_rmsd_matrix`. QTF's `StabilityAnalyzer.convergence_summary` now wraps these and adds the STABLE/FLEXIBLE/UNSTABLE verdict.
