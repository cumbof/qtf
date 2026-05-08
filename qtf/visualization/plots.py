"""Interactive Plotly-based visualisations for QTF.

Three main entry points
-----------------------
* :func:`plot_structure` — 3-D backbone overlay (predicted vs ground truth).
* :func:`plot_energy_landscape` — optimisation energy trace with stage markers.
* :func:`plot_ranking` — interactive table / bar chart of ensemble statistics.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qtf.analysis.stability import kabsch_rmsd


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_PALETTE = {
    "best_energy": "#2ecc71",   # green
    "best_rmsd": "#e74c3c",     # red
    "other": "#95a5a6",         # grey
    "ground_truth": "#2c3e50",  # dark navy
    "stage1": "#e74c3c",
    "stage2": "#f39c12",
    "stage3": "#3498db",
    "energy_line": "#2c3e50",
}


# ---------------------------------------------------------------------------
# 3-D structure visualisation
# ---------------------------------------------------------------------------

def plot_structure(
    ranking,
    ground_truth_ca: Optional[np.ndarray] = None,
    ca_label: str = "CA",
    show_all: bool = True,
    title: str = "Predicted Protein Structures",
) -> go.Figure:
    """Interactive 3-D backbone overlay.

    Parameters
    ----------
    ranking:
        :class:`~qtf.analysis.ranking.EnsembleRanking` instance.
    ground_truth_ca:
        Optional ``(N_residues, 3)`` ground-truth Cα coordinates.  When
        provided, all predicted structures are Kabsch-aligned to it before
        display.
    ca_label:
        Atom label used to filter Cα atoms from ``labels`` (default ``"CA"``).
    show_all:
        When *True* all replicas are shown as semi-transparent traces;
        the two best picks are highlighted.
    title:
        Figure title.

    Returns
    -------
    :class:`plotly.graph_objects.Figure`
    """
    fig = go.Figure()

    def _get_ca(result: dict) -> np.ndarray:
        coords = result["coords"]
        labels = result["labels"]
        return np.array([coords[i] for i, lbl in enumerate(labels) if lbl[1] == ca_label])

    df = ranking.stats_df
    best_e_id = int(df[df["is_best_energy"]]["replica_id"].iloc[0])
    best_r_id = (
        int(df[df["is_best_rmsd"]]["replica_id"].iloc[0])
        if ranking.best_by_rmsd is not None
        else None
    )

    # Build replica lookup from the ranking's internal result list
    result_map = {r["id"]: r for r in _collect_results(ranking)}

    reference = ground_truth_ca  # alignment target

    for _, row in df.iterrows():
        rid = int(row["replica_id"])
        ca = _get_ca(result_map[rid])

        if reference is not None:
            n = min(len(ca), len(reference))
            _, ca_aligned = kabsch_rmsd(ca[:n], reference[:n])
        else:
            ca_aligned = ca

        is_best_e = rid == best_e_id
        is_best_r = rid == best_r_id

        if not show_all and not is_best_e and not is_best_r:
            continue

        if is_best_e and is_best_r:
            colour = "#9b59b6"  # purple — same structure is both bests
            name = f"Replica {rid} ★ (best energy + best RMSD)"
            width, opacity = 4, 1.0
        elif is_best_e:
            colour = _PALETTE["best_energy"]
            name = f"Replica {rid} ★ (best energy)"
            width, opacity = 3, 1.0
        elif is_best_r:
            colour = _PALETTE["best_rmsd"]
            name = f"Replica {rid} ★ (best RMSD)"
            width, opacity = 3, 1.0
        else:
            colour = _PALETTE["other"]
            energy_str = f"{row['energy']:.2f}"
            name = f"Replica {rid}  (E={energy_str})"
            width, opacity = 1.5, 0.35

        energy_val = row["energy"]
        rmsd_val = row["rmsd_vs_gt"]
        hover_txt = (
            f"<b>Replica {rid}</b><br>"
            f"Energy: {energy_val:.4f}<br>"
            f"Rg: {row['radius_of_gyration']:.2f} Å<br>"
            f"E2E: {row['end_to_end_dist']:.2f} Å"
            + (f"<br>RMSD vs GT: {rmsd_val:.3f} Å" if not np.isnan(rmsd_val) else "")
        )

        fig.add_trace(
            go.Scatter3d(
                x=ca_aligned[:, 0],
                y=ca_aligned[:, 1],
                z=ca_aligned[:, 2],
                mode="lines+markers",
                line=dict(color=colour, width=width),
                marker=dict(size=3, color=colour, opacity=opacity),
                opacity=opacity,
                name=name,
                hovertemplate=hover_txt + "<extra></extra>",
            )
        )

    # Ground truth
    if ground_truth_ca is not None:
        fig.add_trace(
            go.Scatter3d(
                x=ground_truth_ca[:, 0],
                y=ground_truth_ca[:, 1],
                z=ground_truth_ca[:, 2],
                mode="lines+markers",
                line=dict(color=_PALETTE["ground_truth"], width=4, dash="dash"),
                marker=dict(size=4, color=_PALETTE["ground_truth"]),
                name="Ground Truth",
                hovertemplate="<b>Ground Truth</b><br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        scene=dict(
            xaxis_title="X (Å)",
            yaxis_title="Y (Å)",
            zaxis_title="Z (Å)",
        ),
        legend=dict(itemsizing="constant"),
        template="plotly_white",
        height=700,
    )
    return fig


# ---------------------------------------------------------------------------
# Energy landscape
# ---------------------------------------------------------------------------

def plot_energy_landscape(
    ranking,
    replica_ids: Optional[list[int]] = None,
    clip_range: tuple[float, float] = (-1000.0, 2000.0),
    title: str = "Optimisation Energy Landscape",
) -> go.Figure:
    """Plot energy-vs-evaluation-step traces for selected replicas.

    Parameters
    ----------
    ranking:
        :class:`~qtf.analysis.ranking.EnsembleRanking` instance.
    replica_ids:
        Subset of replica IDs to display.  Defaults to all replicas.
    clip_range:
        ``(min, max)`` range for clipping extreme energy values in the display.
    title:
        Figure title.

    Returns
    -------
    :class:`plotly.graph_objects.Figure`
    """
    df = ranking.stats_df
    results = _collect_results(ranking)
    result_map = {r["id"]: r for r in results}

    if replica_ids is None:
        replica_ids = list(df["replica_id"])

    best_e_id = int(df[df["is_best_energy"]]["replica_id"].iloc[0])

    stage_colours = [_PALETTE["stage1"], _PALETTE["stage2"], _PALETTE["stage3"]]

    fig = go.Figure()
    stage_lines_added: set[str] = set()

    for rid in replica_ids:
        result = result_map.get(rid)
        if result is None:
            continue
        tracker = result["tracker"]
        energies = np.clip(np.array(tracker.history), *clip_range)
        is_best = rid == best_e_id
        row = df[df["replica_id"] == rid].iloc[0]

        colour = _PALETTE["best_energy"] if is_best else _PALETTE["energy_line"]
        width = 2.5 if is_best else 1.0
        opacity = 1.0 if is_best else 0.5
        label = f"Replica {rid}" + (" ★" if is_best else "")

        fig.add_trace(
            go.Scatter(
                x=list(range(len(energies))),
                y=energies.tolist(),
                mode="lines",
                line=dict(color=colour, width=width),
                opacity=opacity,
                name=label,
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    f"Step: %{{x}}<br>"
                    f"Energy: %{{y:.4f}}<br>"
                    f"Final E: {row['energy']:.4f}"
                    + (f"<br>RMSD vs GT: {row['rmsd_vs_gt']:.3f} Å" if not np.isnan(row["rmsd_vs_gt"]) else "")
                    + "<extra></extra>"
                ),
            )
        )

        # Stage markers (add vertical lines only once per stage name)
        for idx_m, (step, stage_name) in enumerate(tracker.stage_markers):
            stage_colour = stage_colours[min(idx_m, len(stage_colours) - 1)]
            line_key = stage_name
            if line_key not in stage_lines_added:
                fig.add_vline(
                    x=step,
                    line=dict(color=stage_colour, width=1.5, dash="dash"),
                    annotation_text=stage_name,
                    annotation_font_color=stage_colour,
                )
                stage_lines_added.add(line_key)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Function Evaluations",
        yaxis_title="Energy (a.u.)",
        legend=dict(itemsizing="constant"),
        template="plotly_white",
        height=500,
    )
    return fig


# ---------------------------------------------------------------------------
# Ranking visualisation
# ---------------------------------------------------------------------------

def plot_ranking(
    ranking,
    title: str = "Ensemble Ranking",
) -> go.Figure:
    """Interactive two-panel figure: ranking bar chart + statistics table.

    Parameters
    ----------
    ranking:
        :class:`~qtf.analysis.ranking.EnsembleRanking` instance.
    title:
        Figure title.

    Returns
    -------
    :class:`plotly.graph_objects.Figure`
    """
    df = ranking.stats_df.copy()
    has_gt = not df["rmsd_vs_gt"].isna().all()

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.55, 0.45],
        specs=[[{"type": "bar"}], [{"type": "table"}]],
        subplot_titles=["Energy by Replica", "Full Statistics Table"],
        vertical_spacing=0.12,
    )

    colours = []
    for _, row in df.iterrows():
        if row["is_best_energy"] and (has_gt and row["is_best_rmsd"]):
            colours.append("#9b59b6")
        elif row["is_best_energy"]:
            colours.append(_PALETTE["best_energy"])
        elif has_gt and row["is_best_rmsd"]:
            colours.append(_PALETTE["best_rmsd"])
        else:
            colours.append(_PALETTE["other"])

    hover_texts = []
    for _, row in df.iterrows():
        ht = (
            f"<b>Replica {int(row['replica_id'])}</b><br>"
            f"Energy rank: {int(row['rank_energy'])}<br>"
            f"Energy: {row['energy']:.4f}<br>"
            f"Rg: {row['radius_of_gyration']:.3f} Å<br>"
            f"E2E: {row['end_to_end_dist']:.3f} Å<br>"
            f"Mean pairwise RMSD: {row['mean_rmsd_vs_ensemble']:.3f} Å"
        )
        if has_gt:
            ht += f"<br>RMSD vs GT: {row['rmsd_vs_gt']:.3f} Å"
        hover_texts.append(ht)

    fig.add_trace(
        go.Bar(
            x=[f"R{int(r)}" for r in df["replica_id"]],
            y=df["energy"].tolist(),
            marker_color=colours,
            hovertemplate=[h + "<extra></extra>" for h in hover_texts],
            name="Energy",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Legend annotations
    legend_items = [
        ("Best by energy", _PALETTE["best_energy"]),
    ]
    if has_gt:
        legend_items.append(("Best by RMSD", _PALETTE["best_rmsd"]))
    legend_items.append(("Other", _PALETTE["other"]))

    # Table
    table_cols = [
        "rank_energy", "replica_id", "energy",
        "radius_of_gyration", "end_to_end_dist",
        "mean_rmsd_vs_ensemble",
    ]
    if has_gt:
        table_cols += ["rmsd_vs_gt", "rank_rmsd"]
    table_cols += ["is_best_energy", "is_best_rmsd", "is_ensemble_centroid"]
    present = [c for c in table_cols if c in df.columns]

    header_labels = [c.replace("_", " ").title() for c in present]

    def _fmt(col: str, vals) -> list:
        out = []
        for v in vals:
            if col in ("energy", "radius_of_gyration", "end_to_end_dist",
                       "mean_rmsd_vs_ensemble", "rmsd_vs_gt"):
                out.append(f"{v:.4f}" if not (isinstance(v, float) and np.isnan(v)) else "—")
            elif col in ("rank_energy", "rank_rmsd", "replica_id", "seed"):
                out.append(str(int(v)) if not (isinstance(v, float) and np.isnan(v)) else "—")
            elif col in ("is_best_energy", "is_best_rmsd", "is_ensemble_centroid"):
                out.append("✓" if v else "")
            else:
                out.append(str(v))
        return out

    cell_values = [_fmt(c, df[c].tolist()) for c in present]

    cell_colours: list[list[str]] = []
    for c in present:
        col_colours = []
        for _, row in df.iterrows():
            if row["is_best_energy"] and (has_gt and row.get("is_best_rmsd", False)):
                col_colours.append("#e8daef")
            elif row["is_best_energy"]:
                col_colours.append("#d5f5e3")
            elif has_gt and row.get("is_best_rmsd", False):
                col_colours.append("#fadbd8")
            else:
                col_colours.append("white")
        cell_colours.append(col_colours)

    fig.add_trace(
        go.Table(
            header=dict(
                values=[f"<b>{h}</b>" for h in header_labels],
                fill_color="#2c3e50",
                font=dict(color="white", size=11),
                align="center",
            ),
            cells=dict(
                values=cell_values,
                fill_color=cell_colours,
                align="center",
                font=dict(size=10),
                height=24,
            ),
        ),
        row=2, col=1,
    )

    # Convergence annotation
    conv = ranking.convergence
    fig.add_annotation(
        text=(
            f"Convergence: {conv['verdict']}  |  "
            f"avg pairwise RMSD = {conv['avg_pairwise_rmsd']:.3f} Å  |  "
            f"max = {conv['max_pairwise_rmsd']:.3f} Å"
        ),
        xref="paper", yref="paper",
        x=0.0, y=-0.02,
        showarrow=False,
        font=dict(size=11, color="#555"),
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white",
        height=800,
        yaxis_title="Energy (a.u.)",
    )
    return fig


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _collect_results(ranking) -> list[dict]:
    """Recover the original list of result dicts from an EnsembleRanking.

    The ranking stores ``best_by_energy`` and ``best_by_rmsd`` directly but
    does not store the full list.  We reconstruct it from the stats DataFrame
    by walking the tracker and coord data stored in each result dict.
    Note: the EnsembleRanking must have been created via ``from_ensemble``
    which keeps a reference to the results list internally.
    """
    # EnsembleRanking.from_ensemble attaches _results for this purpose
    if hasattr(ranking, "_results"):
        return ranking._results
    # Fallback: reconstruct a minimal list from the two stored best dicts
    seen: dict[int, dict] = {}
    for r in [ranking.best_by_energy, ranking.best_by_rmsd]:
        if r is not None:
            seen[r["id"]] = r
    return list(seen.values())
