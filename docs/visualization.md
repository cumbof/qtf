# Visualisation

QTF ships three Plotly-based interactive figures. All functions accept an
`EnsembleRanking` object and return a `plotly.graph_objects.Figure` that can be
displayed inline (Jupyter), opened in a browser, or saved to HTML / PNG.

---

## plot_structure

Overlays Cα traces for all predicted replicas and — if provided — the experimental
ground-truth backbone.

```python
from qtf.visualization import plot_structure

fig = plot_structure(ranking, ground_truth_ca=true_ca)
fig.show()
```

### Colour coding

| Trace | Colour |
|:------|:-------|
| Ground truth | Red |
| Best energy replica | Green |
| Best RMSD replica (if different from best energy) | Orange |
| Other replicas | Light grey (semi-transparent) |

### Options

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `ground_truth_ca` | `None` | (N, 3) numpy array of ground-truth Cα coordinates |
| `align` | `True` | Kabsch-align all predicted structures to the ground truth before plotting |
| `title` | auto | Plot title |

All predicted Cα traces are aligned to the ground truth (when provided) using the
**Kabsch algorithm** — same alignment used for RMSD computation.

---

## plot_energy_landscape

Shows the energy as a function of optimiser step for every replica, with vertical
dashed lines marking the stage-1→2 and stage-2→3 transitions.

```python
from qtf.visualization import plot_energy_landscape

fig = plot_energy_landscape(ranking)
fig.show()
```

Each replica is drawn as a separate line. Hovering over any point shows the replica
index, iteration count, and exact energy value.

### Options

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `log_y` | `False` | Use a logarithmic y-axis (useful for widely-varying energies) |
| `title` | auto | Plot title |

---

## plot_ranking

Interactive bar chart sorted by energy, with a statistics table appended below the figure.

```python
from qtf.visualization import plot_ranking

fig = plot_ranking(ranking)
fig.show()
```

Bars are coloured by convergence status:

| Colour | Status |
|:-------|:-------|
| Green  | Converged (`converged=True`) |
| Orange | Did not converge |

The table below the chart lists every column from `ranking.stats_df` for quick comparison.

---

## Saving figures

All three functions return a standard `plotly.graph_objects.Figure` and can be saved
with standard Plotly export methods:

```python
# Self-contained interactive HTML
fig.write_html("structure_overlay.html")

# Static PNG (requires the kaleido package)
fig.write_image("structure_overlay.png", width=1200, height=800)

# PDF
fig.write_image("structure_overlay.pdf")
```

Install `kaleido` for static export:

```bash
pip install kaleido
```

---

!!! tip "Inline display"
    In JupyterLab / Jupyter Notebook, `fig.show()` renders the interactive figure
    inline. In VSCode, use the Jupyter extension. For scripts, `fig.show()` opens
    the figure in your default browser.
