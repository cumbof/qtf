"""Interactive Plotly-based visualisations for QTF.

Three main entry points
-----------------------
* :func:`plot_structure` — 3-D backbone overlay (predicted vs ground truth).
* :func:`plot_energy_landscape` — optimisation energy trace with stage markers.
* :func:`plot_ranking` — interactive table / bar chart of ensemble statistics.
"""

from __future__ import annotations

from typing import Optional
import warnings

import numpy as np

from pheat.geometry import kabsch_align


def _require_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ModuleNotFoundError as exc:
        raise ImportError(
            "plotly is required for qtf.visualization plots. Install it with `pip install plotly` "
            "or install QTF with visualization/report dependencies."
        ) from exc
    return go, make_subplots


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
    ground_truth_labels: Optional[list] = None,
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
    ground_truth_labels:
        Optional atom labels for the ground-truth structure. When provided,
        Cα traces are aligned by common residue id instead of by truncation.
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
    go, _ = _require_plotly()
    fig = go.Figure()

    def _get_ca(result: dict) -> tuple[np.ndarray, list[int]]:
        coords = result["coords"]
        labels = result["labels"]
        ca_coords = []
        residue_ids = []
        for i, lbl in enumerate(labels):
            if lbl[1] == ca_label:
                ca_coords.append(coords[i])
                residue_ids.append(int(lbl[0]))
        return np.asarray(ca_coords, dtype=float), residue_ids

    def _ground_truth_by_residue() -> dict[int, np.ndarray]:
        if ground_truth_ca is None or ground_truth_labels is None:
            return {}
        gt_ca_labels = [lbl for lbl in ground_truth_labels if lbl[1] == ca_label]
        return {
            int(lbl[0]): np.asarray(ground_truth_ca[idx], dtype=float)
            for idx, lbl in enumerate(gt_ca_labels[: len(ground_truth_ca)])
        }

    gt_by_residue = _ground_truth_by_residue()
    display_ground_truth = None
    warned_truncation = False

    df = ranking.stats_df
    best_e_id = int(df[df["is_best_energy"]]["replica_id"].iloc[0])
    best_r_id = (
        int(df[df["is_best_rmsd"]]["replica_id"].iloc[0])
        if ranking.best_by_rmsd is not None
        else None
    )

    # Build replica lookup from the ranking's internal result list
    result_map = {r["id"]: r for r in _collect_results(ranking)}

    for _, row in df.iterrows():
        rid = int(row["replica_id"])
        ca_raw, ca_residue_ids = _get_ca(result_map[rid])
        ca = ca_raw
        reference = ground_truth_ca

        if ground_truth_ca is not None and gt_by_residue:
            pred_by_residue = {
                residue_id: np.asarray(ca_raw[idx], dtype=float)
                for idx, residue_id in enumerate(ca_residue_ids)
            }
            common = [residue_id for residue_id in ca_residue_ids if residue_id in gt_by_residue]
            if common:
                if (len(common) < len(ca_residue_ids) or len(common) < len(gt_by_residue)) and not warned_truncation:
                    warnings.warn(
                        f"only {len(common)} match ground-truth residue ids; plotting the common C-alpha subset",
                        UserWarning,
                        stacklevel=2,
                    )
                    warned_truncation = True
                ca = np.asarray([pred_by_residue[residue_id] for residue_id in common], dtype=float)
                reference = np.asarray([gt_by_residue[residue_id] for residue_id in common], dtype=float)
                if display_ground_truth is None:
                    display_ground_truth = reference
        elif ground_truth_ca is not None:
            reference = ground_truth_ca

        if reference is not None:
            n = min(len(ca), len(reference))
            ca_aligned = np.asarray(
                kabsch_align(np.asarray(reference[:n], dtype=float).tolist(), ca[:n].tolist()),
                dtype=float,
            )
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
    ground_truth_trace = display_ground_truth if display_ground_truth is not None else ground_truth_ca
    if ground_truth_trace is not None:
        ground_truth_trace = np.asarray(ground_truth_trace, dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=ground_truth_trace[:, 0],
                y=ground_truth_trace[:, 1],
                z=ground_truth_trace[:, 2],
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
    go, _ = _require_plotly()
    df = ranking.stats_df
    results = _collect_results(ranking)
    result_map = {r["id"]: r for r in results}

    if replica_ids is None:
        replica_ids = list(df["replica_id"])

    best_e_id = int(df[df["is_best_energy"]]["replica_id"].iloc[0])

    stage_colours = [_PALETTE["stage1"], _PALETTE["stage2"], _PALETTE["stage3"]]

    # Build a global stage-name → colour map by collecting every unique stage
    # name across all replicas in first-appearance order.  Colouring by the
    # per-replica enumerate index was wrong: a replica that skips an
    # intermediate stage would offset all subsequent stage colours (e.g.
    # Stage3 appearing as stage2 colour).
    seen_stage_names: dict[str, str] = {}  # name → hex colour
    for r in results:
        for _, sname in r["tracker"].stage_markers:
            if sname not in seen_stage_names:
                idx = len(seen_stage_names)
                seen_stage_names[sname] = stage_colours[min(idx, len(stage_colours) - 1)]
    stage_colour_map: dict[str, str] = seen_stage_names

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

        # Stage markers — colour by name, not by per-replica enumerate index.
        for step, stage_name in tracker.stage_markers:
            stage_colour = stage_colour_map.get(stage_name, stage_colours[-1])
            if stage_name not in stage_lines_added:
                fig.add_vline(
                    x=step,
                    line=dict(color=stage_colour, width=1.5, dash="dash"),
                    annotation_text=stage_name,
                    annotation_font_color=stage_colour,
                )
                stage_lines_added.add(stage_name)

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
# Tracker energy landscape
# ---------------------------------------------------------------------------

_PHASE_COLOURS = (
    "#ff6b35",
    "#4ecdc4",
    "#a855f7",
    "#ffd166",
    "#06d6a0",
    "#ef476f",
    "#118ab2",
    "#f78c6b",
)

_PHASE_STATUS_COLOURS = {
    "ok": "#2e7d32",
    "warning": "#f59e0b",
    "error": "#d32f2f",
}


def _tracker_markers(tracker) -> list[tuple[int, str]]:
    markers = getattr(tracker, "phase_markers", None)
    if markers:
        return [(int(step), str(name)) for step, name in markers]
    markers = getattr(tracker, "stage_markers", None) or []
    return [(int(step), str(name)) for step, name in markers]


def _tracker_phase_ranges(markers: list[tuple[int, str]], n_iters: int) -> list[tuple[str, int, int]]:
    if n_iters <= 0:
        return [("Optimization", 0, 0)]
    cleaned = sorted(
        [(max(0, min(int(step), n_iters - 1)), str(name)) for step, name in markers],
        key=lambda item: item[0],
    )
    if not cleaned:
        cleaned = [(0, "Optimization")]
    elif cleaned[0][0] > 0:
        cleaned.insert(0, (0, "Initialization"))

    ranges = []
    for idx, (start, name) in enumerate(cleaned):
        end = cleaned[idx + 1][0] if idx + 1 < len(cleaned) else n_iters
        if end > start:
            ranges.append((name, start, end))
    return ranges or [("Optimization", 0, n_iters)]


def _phase_status_category(success, status, message: str | None) -> str:
    if bool(success):
        return "ok"
    status_text = "" if status is None else str(status).strip()
    message_text = str(message or "").strip().lower()
    if status_text == "9" or "iteration limit" in message_text:
        return "warning"
    if "maximum number of function evaluations" in message_text:
        return "warning"
    if "maxiter" in message_text or "maxfun" in message_text:
        return "warning"
    return "error"


def _phase_metadata_map(phase_results: Optional[list[dict]]) -> tuple[dict, dict]:
    by_range = {}
    by_label = {}
    for phase in phase_results or []:
        label = str(phase.get("label") or phase.get("name") or "")
        status = str(
            phase.get("phase_status")
            or _phase_status_category(phase.get("success"), phase.get("status"), phase.get("message"))
        )
        if status not in _PHASE_STATUS_COLOURS:
            status = "error"
        metadata = {
            "label": label,
            "optimizer": phase.get("optimizer") or "n/a",
            "score_model": phase.get("score_model") or "n/a",
            "status": status,
            "message": phase.get("phase_status_label") or phase.get("message") or status,
        }
        start = phase.get("energy_start_index")
        end = phase.get("energy_end_index")
        if start is not None and end is not None:
            by_range[(int(start), int(end))] = metadata
        if label:
            by_label[label] = metadata
        name = phase.get("name")
        if name:
            by_label[str(name)] = metadata
    return by_range, by_label


def _phase_metadata(name: str, start: int, end: int, by_range: dict, by_label: dict) -> dict:
    return by_range.get((int(start), int(end))) or by_label.get(str(name)) or {
        "label": str(name),
        "optimizer": "n/a",
        "score_model": "n/a",
        "status": "ok",
        "message": "ok",
    }


def _needs_signed_log(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    nonzero = np.abs(finite[finite != 0])
    if nonzero.size == 0:
        return False
    crosses_zero = float(np.min(finite)) < 0 < float(np.max(finite))
    dynamic_range = float(np.max(nonzero) / max(np.min(nonzero), 1e-12))
    return crosses_zero or dynamic_range > 1.0e4


def _signed_log_transform(values: np.ndarray) -> tuple[np.ndarray, float]:
    finite = values[np.isfinite(values)]
    nonzero = np.abs(finite[finite != 0])
    linthresh = max(1.0, float(np.percentile(nonzero, 10))) if nonzero.size else 1.0
    return np.sign(values) * np.log10(1.0 + np.abs(values) / linthresh), linthresh


def plot_tracker_energy_landscape(
    tracker,
    *,
    sequence: str = "",
    forcefield: str = "",
    title: Optional[str] = None,
    phase_results: Optional[list[dict]] = None,
    save_path: Optional[str] = None,
    include_plotlyjs: str | bool = "cdn",
    full_html: bool = True,
) -> go.Figure:
    """Plot a single run's energy trace with phase boundaries.

    This is the per-run companion to :func:`plot_energy_landscape`, which plots
    ranked ensemble traces.  The tracker may expose either ``phase_markers`` or
    the package's existing ``stage_markers`` attribute.
    """
    go, _ = _require_plotly()
    history = np.asarray(getattr(tracker, "history", []) or [], dtype=float)
    markers = _tracker_markers(tracker)
    phase_ranges = _tracker_phase_ranges(markers, len(history))
    by_range, by_label = _phase_metadata_map(phase_results)

    fig = go.Figure()
    if history.size == 0:
        fig.add_annotation(
            text="No energy evaluations recorded",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        y_values = history
        y_title = "Energy (a.u.)"
    else:
        if _needs_signed_log(history):
            y_values, linthresh = _signed_log_transform(history)
            y_title = f"Signed log energy (linthresh={linthresh:.3g})"
        else:
            y_values = history
            y_title = "Energy (a.u.)"

        for idx, (name, start, end) in enumerate(phase_ranges):
            meta = _phase_metadata(name, start, end, by_range, by_label)
            colour = _PHASE_COLOURS[idx % len(_PHASE_COLOURS)]
            status_colour = _PHASE_STATUS_COLOURS.get(meta["status"], _PHASE_STATUS_COLOURS["error"])
            x_segment = list(range(start, end))
            y_segment = y_values[start:end]
            raw_segment = history[start:end]
            phase_label = meta.get("label") or name
            hover = [
                (
                    f"<b>{phase_label}</b><br>"
                    f"Evaluation: {x}<br>"
                    f"Energy: {raw:.6g}<br>"
                    f"Optimizer: {meta['optimizer']}<br>"
                    f"Score: {meta['score_model']}<br>"
                    f"Status: {meta['message']}"
                    "<extra></extra>"
                )
                for x, raw in zip(x_segment, raw_segment)
            ]
            fig.add_trace(
                go.Scatter(
                    x=x_segment,
                    y=y_segment,
                    mode="lines",
                    line=dict(color=colour, width=2.5),
                    name=str(phase_label),
                    hovertemplate=hover,
                )
            )
            if start > 0:
                fig.add_vline(x=start, line=dict(color=colour, width=1, dash="dash"))
            if meta["status"] != "ok":
                fig.add_vrect(
                    x0=start,
                    x1=max(start + 1, end),
                    fillcolor=status_colour,
                    opacity=0.08,
                    line_width=0,
                )
                fig.add_annotation(
                    x=start,
                    y=y_segment[-1] if len(y_segment) else 0,
                    text=meta["status"].upper(),
                    showarrow=True,
                    arrowcolor=status_colour,
                    font=dict(color=status_colour),
                )

    if title is None:
        parts = ["Optimization Energy Landscape"]
        if sequence:
            parts.append(sequence)
        if forcefield:
            parts.append(str(forcefield).upper())
        title = " | ".join(parts)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Function Evaluations",
        yaxis_title=y_title,
        legend=dict(itemsizing="constant"),
        template="plotly_white",
        height=560,
    )
    if save_path is not None:
        import plotly.io as pio

        pio.write_html(fig, file=str(save_path), include_plotlyjs=include_plotlyjs, full_html=full_html)
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
    go, make_subplots = _require_plotly()
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
                out.append(f"{v:.4f}" if not (isinstance(v, (float, np.floating)) and np.isnan(v)) else "—")
            elif col in ("rank_energy", "rank_rmsd", "replica_id", "seed"):
                out.append(str(int(v)) if not (isinstance(v, (float, np.floating)) and np.isnan(v)) else "—")
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
    """Return the full list of replica result dicts from an *EnsembleRanking*.

    Parameters
    ----------
    ranking:
        An :class:`~qtf.analysis.ranking.EnsembleRanking` instance that was
        built via :meth:`~qtf.analysis.ranking.EnsembleRanking.from_ensemble`.
        That constructor stores the complete result list in ``ranking._results``,
        which is the only source this function consults.

    Returns
    -------
    list[dict]
        The full list of replica dicts, each containing at minimum ``id``,
        ``coords``, ``labels``, and ``tracker``.

    Raises
    ------
    ValueError
        If *ranking* has no ``_results`` attribute, or if that attribute is
        empty.  This replaces a previous silent fallback that returned only
        the one or two "best" replica dicts, causing ``KeyError`` in callers
        that iterated over all replica IDs from the stats DataFrame.

        **Fix**: always construct the ranking with::

            ranking = EnsembleRanking.from_ensemble(results, ...)

        where *results* is the list returned by
        :meth:`~qtf.core.ensemble.EnsembleFoldingManager.get_results`.
    """
    results = getattr(ranking, "_results", None)
    if not results:
        raise ValueError(
            "_collect_results: the EnsembleRanking object carries no result "
            "data.  Build the ranking via "
            "EnsembleRanking.from_ensemble(results, ...) where 'results' is "
            "the list returned by EnsembleFoldingManager.get_results()."
        )
    return results
