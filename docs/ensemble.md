# Ensemble Folding & Ranking

This page covers everything related to running multiple independent replicas and
interpreting the results collectively.

---

## Running an ensemble

```python
from qtf import QuantumBiophysicsFolder, EnsembleFoldingManager

folder  = QuantumBiophysicsFolder("YYDPETGTWY", force_field="amber")
manager = EnsembleFoldingManager(folder)

manager.run_ensemble(
    n_runs=10,           # number of replicas to run
    max_iter=2000,       # max optimiser iterations per stage per replica
    scout_attempts=50,   # basin-hopping samples before each full run
)
```

### Reproducibility

Seeds are generated deterministically from `SHA-256(sequence + replica_index)`, so
the same call always produces the same replicas regardless of platform or environment.
This makes results fully reproducible and shareable.

---

## Accessing raw results

```python
results = manager.get_results()   # list of dicts, one per completed replica
```

Each element is a dictionary with the following schema:

| Key         | Type               | Description |
|:------------|:-------------------|:------------|
| `id`        | `int`              | Zero-based replica index |
| `energy`    | `float`            | Final physics energy |
| `angles`    | `numpy.ndarray`    | All dihedral angles (radians) |
| `coords`    | `numpy.ndarray`    | All-atom Cartesian coordinates (Å) |
| `labels`    | `list[str]`        | Atom labels, one per row of `coords` |
| `tracker`   | `LandscapeTracker` | Full per-step energy trajectory |
| `seed`      | `int`              | Deterministic integer seed for this replica |
| `converged` | `bool`             | `True` if the final optimiser stage reported convergence |

---

## Building a ranking

```python
from qtf.analysis import EnsembleRanking
from pheat.pdbio import load_pdb_ca_by_id

true_ca = load_pdb_ca_by_id("5AWL")   # optional

ranking = EnsembleRanking.from_ensemble(
    results,
    ground_truth_ca=true_ca,   # omit if no reference structure is available
)
```

`from_ensemble` computes per-replica statistics, builds the pairwise RMSD matrix,
and emits a convergence verdict.

---

## Statistics DataFrame

```python
df = ranking.stats_df
print(df.columns.tolist())
```

The DataFrame contains one row per replica with the following columns:

| Column            | Unit | Description |
|:------------------|:-----|:------------|
| `replica_id`      | —    | Zero-based replica index |
| `energy`          | a.u. | Final physics energy (lower is better) |
| `rmsd_vs_gt`      | Å    | Cα RMSD vs ground truth after Kabsch alignment (`NaN` if no ground truth) |
| `rg`              | Å    | Radius of gyration |
| `end_to_end`      | Å    | End-to-end distance (first Cα to last Cα) |
| `converged`       | bool | Whether the final optimiser stage reported convergence |
| `n_evals`         | —    | Total number of energy evaluations performed |

---

## Best structures

```python
best_e = ranking.best_by_energy   # replica with lowest energy
best_r = ranking.best_by_rmsd     # replica with lowest RMSD (None if no ground truth)
```

Both return the full result dict for the corresponding replica (same schema as above).

---

## Convergence verdicts

The `StabilityAnalyzer` computes an all-vs-all pairwise Cα RMSD matrix for the
replicas and emits one of three verdicts, accessible via `ranking.convergence`:

| Verdict       | Meaning |
|:--------------|:--------|
| `"converged"` | Mean pairwise RMSD < 2.0 Å — the ensemble consistently reaches similar structures |
| `"partial"`   | Mean pairwise RMSD 2.0–5.0 Å — some agreement, but multiple basins are present |
| `"diverged"`  | Mean pairwise RMSD > 5.0 Å — no consistent basin; increase `n_runs` or `max_iter` |

---

## Summary table

```python
print(ranking.summary())
```

Prints a compact text table with the most important statistics for every replica,
sorted by energy.

---

## Performance guidance

| Sequence length | Recommended `n_runs` | Recommended `max_iter` |
|:----------------|:---------------------|:-----------------------|
| ≤ 5 residues    | 3–5                  | 500–1 000 |
| 6–10 residues   | 5–10                 | 1 000–2 000 |
| 11–15 residues  | 10–20                | 2 000–4 000 |
| > 15 residues   | 20+                  | 4 000+ |

Increasing `scout_attempts` beyond 100 rarely improves final energy but significantly
reduces the variance across replicas by biasing each run toward a promising region.

!!! warning "Runtime"
    Each replica requires multiple (statevector) circuit evaluations per optimiser
    iteration. A 10-residue protein with `max_iter=2000` and `scout_attempts=50`
    takes roughly 30–120 s per replica on a modern desktop CPU, depending on the
    total number of degrees of freedom.
