#!/usr/bin/env python3
"""
QTF Energy Landscape Visualizer
=================================
Plots energy vs iteration across optimization phases.

USAGE:
    # After a fold run, pass the tracker object:
    from qtf_landscape_viz import plot_energy_landscape
    plot_energy_landscape(tracker, sequence="YYDPETGTWY", save_path="landscape.png")

    # Or run standalone with a saved tracker JSON:
    python3 qtf_landscape_viz.py --tracker_json tracker.json
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import argparse
import math
import os
import textwrap


PHASE_COLORS = (
    '#ff6b35',  # orange
    '#4ecdc4',  # teal
    '#a855f7',  # purple
    '#ffd166',  # yellow
    '#06d6a0',  # green
    '#ef476f',  # pink
    '#118ab2',  # blue
    '#f78c6b',  # coral
)

PHASE_STATUS_COLORS = {
    "ok": "#2e7d32",
    "warning": "#f59e0b",
    "error": "#d32f2f",
}


def _compact_energy(value):
    if value is None:
        return "n/a"
    value = float(value)
    if not np.isfinite(value):
        return "n/a"
    abs_value = abs(value)
    if abs_value >= 1.0e4 or (0 < abs_value < 1.0e-2):
        return f"{value:.2e}"
    if abs_value >= 100:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _short_label(value, width=24):
    return textwrap.shorten(str(value), width=width, placeholder="...")


def _phase_status_category(success, status, message):
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


def _phase_status_label(status_name, status, message):
    message_text = str(message or "").strip()
    if status_name == "ok":
        return "ok" if not message_text else f"ok: {message_text}"
    if status_name == "warning":
        if message_text:
            return f"warning: {message_text}"
        return f"warning: status {status}" if status is not None else "warning"
    if message_text:
        return f"error: {message_text}"
    return f"error: status {status}" if status is not None else "error"


def _phase_ranges(markers, n_iters):
    cleaned = []
    for start_iter, name in markers or []:
        start = int(start_iter)
        if start < 0 or start >= n_iters:
            continue
        cleaned.append((start, str(name)))
    cleaned = sorted(cleaned, key=lambda item: item[0])

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


def _phase_color_map(phase_ranges):
    return {
        name: PHASE_COLORS[idx % len(PHASE_COLORS)]
        for idx, (name, _start, _end) in enumerate(phase_ranges)
    }


def _set_dark_axis_style(ax):
    ax.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('#444')
    ax.grid(True, alpha=0.15, color='white')


def _maybe_set_symlog(ax, values, axis='y'):
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(finite) == 0:
        return
    abs_finite = np.abs(finite[finite != 0])
    if len(abs_finite) == 0:
        return
    dynamic_range = float(np.max(abs_finite) / max(np.min(abs_finite), 1e-12))
    crosses_zero = float(np.min(finite)) < 0 < float(np.max(finite))
    if crosses_zero or dynamic_range > 1.0e4:
        linthresh = max(1.0, float(np.percentile(abs_finite, 10)))
        if axis == 'x':
            ax.set_xscale('symlog', linthresh=linthresh)
        else:
            ax.set_yscale('symlog', linthresh=linthresh)


def _phase_summaries(history, phase_ranges, color_by_phase):
    summaries = []
    for name, start, end in phase_ranges:
        seg = history[start:end]
        if len(seg) == 0:
            continue
        summaries.append(
            {
                "name": name,
                "start": start,
                "end": end,
                "evaluations": len(seg),
                "start_energy": float(seg[0]),
                "end_energy": float(seg[-1]),
                "min_energy": float(np.min(seg)),
                "max_energy": float(np.max(seg)),
                "delta": float(seg[0] - seg[-1]),
                "color": color_by_phase.get(name, '#ffffff'),
            }
        )
    return summaries


def _tracker_phase_markers(tracker):
    return tracker.phase_markers


def _phase_metadata_map(phase_results):
    by_range = {}
    by_label = {}
    for phase in phase_results or []:
        start = phase.get("energy_start_index")
        end = phase.get("energy_end_index")
        label = str(phase.get("label") or phase.get("name") or "")
        raw_status = phase.get("status")
        raw_message = str(phase.get("message") or "")
        phase_status = str(
            phase.get("phase_status")
            or _phase_status_category(phase.get("success"), raw_status, raw_message)
        )
        if phase_status not in PHASE_STATUS_COLORS:
            phase_status = "error"
        metadata = {
            "score_model": phase.get("score_model") or "n/a",
            "optimizer": phase.get("optimizer") or "n/a",
            "index": phase.get("index") or "",
            "success": bool(phase.get("success")),
            "status": raw_status,
            "message": raw_message,
            "phase_status": phase_status,
            "phase_status_label": phase.get("phase_status_label")
            or _phase_status_label(phase_status, raw_status, raw_message),
        }
        if start is not None and end is not None:
            by_range[(int(start), int(end))] = metadata
        if label:
            by_label[label] = metadata
    return by_range, by_label


def _phase_metadata(name, start, end, metadata_by_range, metadata_by_label):
    return (
        metadata_by_range.get((int(start), int(end)))
        or metadata_by_label.get(str(name))
        or {
            "score_model": "n/a",
            "optimizer": "n/a",
            "index": "",
            "success": True,
            "status": None,
            "message": "",
            "phase_status": "ok",
            "phase_status_label": "ok",
        }
    )


def _finite_float_array(values):
    finite = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            finite.append(value)
    return np.asarray(finite, dtype=float)


def _needs_signed_log_scale(values):
    finite = _finite_float_array(values)
    if len(finite) == 0:
        return False
    abs_finite = np.abs(finite[finite != 0])
    if len(abs_finite) == 0:
        return False
    dynamic_range = float(np.max(abs_finite) / max(np.min(abs_finite), 1e-12))
    crosses_zero = float(np.min(finite)) < 0 < float(np.max(finite))
    return crosses_zero or dynamic_range > 1.0e4


def _signed_log_linthresh(values):
    finite = _finite_float_array(values)
    abs_finite = np.abs(finite[finite != 0])
    if len(abs_finite) == 0:
        return 1.0
    return max(1.0, float(np.percentile(abs_finite, 10)))


def _signed_log_transform(values, linthresh):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log10(1.0 + np.abs(values) / max(float(linthresh), 1e-12))


def _signed_log_axis_ticks(values, linthresh, max_ticks=11):
    finite = _finite_float_array(values)
    if len(finite) == 0:
        return [], []
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    max_abs = max(abs(min_value), abs(max_value), linthresh)
    start_exp = math.floor(math.log10(max(linthresh, 1e-12)))
    end_exp = math.ceil(math.log10(max_abs))
    powers = [10.0 ** exp for exp in range(start_exp, end_exp + 1)]
    if len(powers) > max_ticks // 2:
        stride = math.ceil(len(powers) / max(1, max_ticks // 2))
        powers = powers[::stride]

    raw_ticks = []
    if min_value < 0:
        raw_ticks.extend([-value for value in reversed(powers) if min_value <= -value <= max_value])
    if min_value <= 0 <= max_value:
        raw_ticks.append(0.0)
    if max_value > 0:
        raw_ticks.extend([value for value in powers if min_value <= value <= max_value])
    raw_ticks.extend([min_value, max_value])

    deduped = []
    for value in sorted(raw_ticks):
        if not deduped or abs(value - deduped[-1]) > 1e-12:
            deduped.append(value)
    tick_values = _signed_log_transform(deduped, linthresh).tolist()
    tick_text = [_compact_energy(value) for value in deduped]
    return tick_values, tick_text


def _plotly_axis_values(values):
    if _needs_signed_log_scale(values):
        linthresh = _signed_log_linthresh(values)
        return _signed_log_transform(values, linthresh), linthresh
    return np.asarray(values, dtype=float), None


def plot_energy_landscape_interactive(
    tracker,
    sequence: str = "",
    forcefield: str = "",
    save_path: str = None,
    title_extra: str = "",
    phase_results=None,
    include_plotlyjs: str = "directory",
    full_html: bool = True,
):
    """
    Build an interactive Plotly version of the energy landscape.

    Raw energy values are preserved in hover data. Axes with very large dynamic
    range or mixed signs use a signed-log display transform.
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise RuntimeError("Interactive energy landscape requires plotly.") from exc

    history = np.asarray(tracker.history, dtype=float)
    markers = _tracker_phase_markers(tracker)
    n_iters = len(history)
    if n_iters == 0:
        print("[WARNING] No energy history to plot.")
        return None

    phase_ranges = _phase_ranges(markers, n_iters)
    color_by_phase = _phase_color_map(phase_ranges)
    summaries = _phase_summaries(history, phase_ranges, color_by_phase)
    metadata_by_range, metadata_by_label = _phase_metadata_map(phase_results)
    for item in summaries:
        metadata = _phase_metadata(
            item["name"],
            item["start"],
            item["end"],
            metadata_by_range,
            metadata_by_label,
        )
        item["metadata"] = metadata
        item["phase_status"] = metadata.get("phase_status") or "ok"
        item["phase_status_label"] = metadata.get("phase_status_label") or "ok"
    transformed_history, history_linthresh = _plotly_axis_values(history)
    deltas = [item["delta"] for item in summaries]
    transformed_deltas, delta_linthresh = _plotly_axis_values(deltas) if deltas else ([], None)

    title = f"QTF Energy Landscape - {sequence or 'sequence'}"
    if forcefield:
        title += f" ({forcefield.upper()})"
    if title_extra:
        title += f"<br><sup>{title_extra}</sup>"

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "table"}]],
        subplot_titles=(
            "Full Energy Landscape",
            "Per-Phase Energy (running minimum)",
            "Energy Change per Phase",
            "Run Summary",
        ),
        horizontal_spacing=0.14,
        vertical_spacing=0.22,
        column_widths=[0.58, 0.42],
        row_heights=[0.58, 0.42],
    )

    for name, start, end in phase_ranges:
        color = color_by_phase.get(name, "#444444")
        label = _short_label(name, width=28)
        seg_energy = history[start:end]
        if len(seg_energy) == 0:
            continue
        global_eval = np.arange(start + 1, end + 1)
        phase_eval = np.arange(1, len(seg_energy) + 1)
        metadata = _phase_metadata(name, start, end, metadata_by_range, metadata_by_label)
        phase_status = metadata.get("phase_status") or "ok"
        phase_status_label = metadata.get("phase_status_label") or "ok"
        phase_status_color = PHASE_STATUS_COLORS.get(phase_status, PHASE_STATUS_COLORS["error"])
        line_dash = "dash" if phase_status == "error" else "dot" if phase_status == "warning" else "solid"
        line_width = 2.4 if phase_status in {"warning", "error"} else 1.6
        customdata = [
            [
                str(name),
                int(phase_index),
                float(raw_energy),
                str(metadata.get("score_model") or "n/a"),
                str(metadata.get("optimizer") or "n/a"),
                str(phase_status_label),
                str(metadata.get("message") or ""),
            ]
            for phase_index, raw_energy in zip(phase_eval, seg_energy)
        ]
        fig.add_trace(
            go.Scatter(
                x=global_eval,
                y=transformed_history[start:end],
                mode="lines",
                name=label,
                legendgroup=label,
                line=dict(color=color, width=line_width, dash=line_dash),
                customdata=customdata,
                hovertemplate=(
                    "Phase: %{customdata[0]}<br>"
                    "Global evaluation: %{x}<br>"
                    "Phase evaluation: %{customdata[1]}<br>"
                    "Energy: %{customdata[2]:.8g}<br>"
                    "Score: %{customdata[3]}<br>"
                    "Optimizer: %{customdata[4]}<br>"
                    "Status: %{customdata[5]}<br>"
                    "Message: %{customdata[6]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_vline(
            x=start + 1,
            line_dash="dash" if phase_status == "ok" else "solid",
            line_color=color if phase_status == "ok" else phase_status_color,
            line_width=1 if phase_status == "ok" else 2.4,
            opacity=0.40 if phase_status == "ok" else 0.85,
            row=1,
            col=1,
        )

        running_min = np.minimum.accumulate(seg_energy)
        transformed_seg = (
            _signed_log_transform(seg_energy, history_linthresh)
            if history_linthresh is not None
            else seg_energy
        )
        transformed_running_min = (
            _signed_log_transform(running_min, history_linthresh)
            if history_linthresh is not None
            else running_min
        )
        zoom_customdata = [
            [str(name), int(phase_index), float(raw_energy), float(run_min), str(phase_status_label)]
            for phase_index, raw_energy, run_min in zip(phase_eval, seg_energy, running_min)
        ]
        fig.add_trace(
            go.Scatter(
                x=phase_eval,
                y=transformed_seg,
                mode="lines",
                name=f"{label} raw",
                legendgroup=label,
                showlegend=False,
                line=dict(color=color, width=1.0, dash=line_dash),
                opacity=0.35,
                customdata=zoom_customdata,
                hovertemplate=(
                    "Phase: %{customdata[0]}<br>"
                    "Phase evaluation: %{x}<br>"
                    "Energy: %{customdata[2]:.8g}<br>"
                    "Status: %{customdata[4]}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=phase_eval,
                y=transformed_running_min,
                mode="lines",
                name=f"{label} running min",
                legendgroup=label,
                line=dict(color=color, width=2.6 if phase_status in {"warning", "error"} else 2.2, dash=line_dash),
                customdata=zoom_customdata,
                hovertemplate=(
                    "Phase: %{customdata[0]}<br>"
                    "Phase evaluation: %{x}<br>"
                    "Running min: %{customdata[3]:.8g}<br>"
                    "Energy: %{customdata[2]:.8g}<br>"
                    "Status: %{customdata[4]}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    if summaries:
        bar_labels = [_short_label(item["name"], width=34) for item in summaries]
        bar_customdata = [
            [
                item["name"],
                item["evaluations"],
                item["start_energy"],
                item["end_energy"],
                item["min_energy"],
                item["max_energy"],
                item["delta"],
                item.get("phase_status_label", "ok"),
            ]
            for item in summaries
        ]
        fig.add_trace(
            go.Bar(
                x=transformed_deltas,
                y=bar_labels,
                orientation="h",
                marker=dict(
                    color=[
                        PHASE_STATUS_COLORS.get(item.get("phase_status"), item["color"])
                        if item.get("phase_status") in {"warning", "error"}
                        else item["color"]
                        for item in summaries
                    ],
                    line=dict(color="#ffffff", width=0.5),
                ),
                customdata=bar_customdata,
                hovertemplate=(
                    "Phase: %{customdata[0]}<br>"
                    "Evaluations: %{customdata[1]}<br>"
                    "Start energy: %{customdata[2]:.8g}<br>"
                    "End energy: %{customdata[3]:.8g}<br>"
                    "Min energy: %{customdata[4]:.8g}<br>"
                    "Max energy: %{customdata[5]:.8g}<br>"
                    "Delta: %{customdata[6]:.8g}<br>"
                    "Status: %{customdata[7]}<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

    table_values = [
        [_short_label(item["name"], width=22) for item in summaries],
        [str(item.get("phase_status") or "ok").upper() for item in summaries],
        [f"{item['evaluations']:,}" for item in summaries],
        [_compact_energy(item["start_energy"]) for item in summaries],
        [_compact_energy(item["end_energy"]) for item in summaries],
        [_compact_energy(item["min_energy"]) for item in summaries],
        [_compact_energy(item["max_energy"]) for item in summaries],
        [_compact_energy(item["delta"]) for item in summaries],
    ]
    status_fill = [
        PHASE_STATUS_COLORS.get(item.get("phase_status"), PHASE_STATUS_COLORS["error"])
        for item in summaries
    ]
    fig.add_trace(
        go.Table(
            columnwidth=[1.7, 0.8, 0.65, 0.8, 0.8, 0.8, 0.8, 0.8],
            header=dict(
                values=["Phase", "Status", "Eval", "Start", "End", "Min", "Max", "Delta"],
                fill_color="#f4f4f4",
                align="left",
                font=dict(color="#1f252d", size=11),
                height=28,
            ),
            cells=dict(
                values=table_values,
                fill_color=[
                    "#ffffff",
                    status_fill,
                    "#ffffff",
                    "#ffffff",
                    "#ffffff",
                    "#ffffff",
                    "#ffffff",
                    "#ffffff",
                ],
                align="left",
                font=dict(
                    color=[
                        "#1f252d",
                        "#ffffff",
                        "#1f252d",
                        "#1f252d",
                        "#1f252d",
                        "#1f252d",
                        "#1f252d",
                        "#1f252d",
                    ],
                    size=10,
                ),
                height=28,
            ),
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Global function evaluation", row=1, col=1)
    fig.update_xaxes(title_text="Evaluation within phase", row=1, col=2)
    fig.update_xaxes(title_text="Energy delta: start - end", row=2, col=1)
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    y_title = "Energy"
    if history_linthresh is not None:
        tickvals, ticktext = _signed_log_axis_ticks(history, history_linthresh)
        y_title = "Energy (signed log display; hover shows raw)"
        fig.update_yaxes(tickvals=tickvals, ticktext=ticktext, row=1, col=1)
        fig.update_yaxes(tickvals=tickvals, ticktext=ticktext, row=1, col=2)
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_yaxes(title_text=y_title, row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    if delta_linthresh is not None:
        tickvals, ticktext = _signed_log_axis_ticks(deltas, delta_linthresh)
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, row=2, col=1)
        fig.update_xaxes(title_text="Energy delta (signed log display; hover shows raw)", row=2, col=1)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        template="plotly_white",
        height=max(980, 860 + 44 * len(summaries)),
        margin=dict(l=90, r=60, t=150, b=150),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="left",
            x=0,
            font=dict(size=11),
            itemwidth=30,
        ),
    )
    fig.update_annotations(font=dict(size=13))
    config = {
        "displaylogo": False,
        "responsive": True,
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"qtf_energy_landscape_{sequence or 'run'}",
            "scale": 2,
        },
    }
    if save_path:
        pio.write_html(
            fig,
            file=save_path,
            include_plotlyjs=include_plotlyjs,
            full_html=full_html,
            config=config,
            auto_open=False,
        )
        print(f"[SAVED] {save_path}")
    return fig


def plot_energy_landscape(
    tracker,
    sequence: str = "",
    forcefield: str = "",
    save_path: str = None,
    show: bool = True,
    title_extra: str = "",
):
    """
    Plot energy landscape from a LandscapeTracker object.

    Parameters
    ----------
    tracker     : LandscapeTracker — from folder.fold()
    sequence    : str — protein sequence for title
    forcefield  : str — force field name for title
    save_path   : str — path to save PNG (None = don't save)
    show        : bool — whether to call plt.show()
    title_extra : str — extra info to append to title (e.g. RMSD)
    """
    history = np.asarray(tracker.history, dtype=float)
    markers = _tracker_phase_markers(tracker)
    n_iters = len(history)

    if n_iters == 0:
        print("[WARNING] No energy history to plot.")
        return

    # ── Figure layout ──────────────────────────────────────────────────────────
    phase_ranges = _phase_ranges(markers, n_iters)
    color_by_phase = _phase_color_map(phase_ranges)
    summaries = _phase_summaries(history, phase_ranges, color_by_phase)
    fig_height = max(11.5, 9.5 + 0.35 * len(summaries))
    fig, axes = plt.subplots(2, 2, figsize=(17, fig_height), constrained_layout=True)
    fig.patch.set_facecolor('#0f0f1a')

    iters = np.arange(n_iters)

    # ── Plot 1: Full landscape ─────────────────────────────────────────────────
    ax1 = axes[0, 0]
    _set_dark_axis_style(ax1)

    for name, start, end in phase_ranges:
        color = color_by_phase.get(name, '#ffffff')
        seg_iters = iters[start:end]
        seg_energy = history[start:end]
        label = _short_label(name)
        ax1.plot(seg_iters, seg_energy, color=color, linewidth=1.2, alpha=0.9, label=label)
        ax1.axvline(x=start, color=color, linestyle='--', alpha=0.4, linewidth=0.8)
        if len(seg_iters) > 0:
            mid = start + (end - start) // 2
            ax1.text(
                mid,
                0.97,
                label,
                transform=ax1.get_xaxis_transform(),
                color=color,
                fontsize=7,
                ha='center',
                va='top',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#101827', edgecolor=color, alpha=0.82),
            )

    _maybe_set_symlog(ax1, history, axis='y')
    ax1.set_xlabel('Function Evaluation', color='white', fontsize=10)
    ax1.set_ylabel('Energy', color='white', fontsize=10)
    ax1.set_title('Full Energy Landscape', color='white', fontsize=12, fontweight='bold')

    # ── Plot 2: Per-phase zoom ─────────────────────────────────────────────────
    ax2 = axes[0, 1]
    _set_dark_axis_style(ax2)

    for name, start, end in phase_ranges:
        color = color_by_phase.get(name, '#ffffff')
        seg_energy = history[start:end]
        seg_iters  = np.arange(len(seg_energy))

        # Running minimum (envelope)
        running_min = np.minimum.accumulate(seg_energy)

        ax2.plot(seg_iters, seg_energy,    color=color, linewidth=0.8,
                 alpha=0.4, label=f'_{name}')
        ax2.plot(seg_iters, running_min,   color=color, linewidth=2.0,
                 alpha=0.95, label=_short_label(name))

    _maybe_set_symlog(ax2, history, axis='y')
    ax2.set_xlabel('Iteration within Phase', color='white', fontsize=10)
    ax2.set_ylabel('Energy', color='white', fontsize=10)
    ax2.set_title('Per-Phase Energy (running minimum)', color='white',
                  fontsize=12, fontweight='bold')
    ax2.legend(facecolor='#2a2a3e', edgecolor='#555',
               labelcolor='white', fontsize=8, loc='best')

    # ── Plot 3: Phase energy change bar chart ──────────────────────────────────
    ax3 = axes[1, 0]
    _set_dark_axis_style(ax3)

    if summaries:
        y = np.arange(len(summaries))
        deltas = [item["delta"] for item in summaries]
        colors = [item["color"] for item in summaries]
        labels = [_short_label(item["name"], width=28) for item in summaries]
        bars = ax3.barh(y, deltas, color=colors, alpha=0.85, edgecolor='white',
                        linewidth=0.5)
        ax3.axvline(0, color='white', alpha=0.45, linewidth=0.8)
        max_abs_delta = max(abs(delta) for delta in deltas) or 1.0
        if min(deltas) < 0 < max(deltas):
            ax3.set_xlim(-1.2 * max_abs_delta, 1.2 * max_abs_delta)
        for bar, delta in zip(bars, deltas):
            offset = 8 if delta >= 0 else -8
            ha = 'left' if delta >= 0 else 'right'
            ax3.annotate(
                _compact_energy(delta),
                xy=(0, bar.get_y() + bar.get_height() / 2),
                xytext=(offset, 0),
                textcoords='offset points',
                ha=ha,
                va='center',
                color='white',
                fontsize=8,
            )
        ax3.set_yticks(y)
        ax3.set_yticklabels(labels, color='white', fontsize=8)
        ax3.invert_yaxis()
        _maybe_set_symlog(ax3, deltas, axis='x')
    else:
        ax3.text(0.5, 0.5, "No phase summaries", transform=ax3.transAxes,
                 ha='center', va='center', color='white')

    ax3.set_xlabel('Energy delta: start - end (positive = lower final energy)', color='white', fontsize=10)
    ax3.set_title('Energy Change per Phase', color='white',
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.15, color='white', axis='x')

    # ── Plot 4: Stats summary ──────────────────────────────────────────────────
    ax4 = axes[1, 1]
    ax4.set_facecolor('#1a1a2e')
    ax4.axis('off')

    summary_text = (
        f"Sequence: {sequence or 'n/a'}    "
        f"Force field: {forcefield.upper() if forcefield else 'n/a'}    "
        f"Evaluations: {n_iters:,}    "
        f"Initial: {_compact_energy(history[0])}    "
        f"Final: {_compact_energy(history[-1])}    "
        f"Drop: {_compact_energy(history[0] - history[-1])}"
    )
    ax4.text(0.0, 0.96, summary_text, transform=ax4.transAxes,
             color='white', fontsize=9, va='top', fontfamily='monospace')

    table_rows = [
        [
            _short_label(item["name"], width=22),
            f"{item['evaluations']:,}",
            _compact_energy(item["start_energy"]),
            _compact_energy(item["end_energy"]),
            _compact_energy(item["min_energy"]),
            _compact_energy(item["max_energy"]),
            _compact_energy(item["delta"]),
        ]
        for item in summaries
    ]
    if table_rows:
        table = ax4.table(
            cellText=table_rows,
            colLabels=["Phase", "Eval", "Start", "End", "Min", "Max", "Delta"],
            loc='center',
            cellLoc='left',
            colWidths=[0.27, 0.08, 0.11, 0.11, 0.11, 0.16, 0.16],
            bbox=[0.0, 0.15 if title_extra else 0.05, 1.0, 0.72 if title_extra else 0.82],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.5)
        table.scale(1.0, 1.35)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#555')
            cell.set_linewidth(0.4)
            cell.get_text().set_color('white')
            cell.set_facecolor('#2a2a3e' if row == 0 else '#1a1a2e')
            if row == 0:
                cell.get_text().set_fontweight('bold')
        for row_idx, item in enumerate(summaries, start=1):
            table[(row_idx, 0)].get_text().set_color(item["color"])

    if title_extra:
        ax4.text(
            0.0,
            0.06,
            textwrap.fill(title_extra, width=78),
            transform=ax4.transAxes,
            color='#cbd5e1',
            fontsize=8,
            va='bottom',
        )

    ax4.set_title('Run Summary', color='white', fontsize=12, fontweight='bold')

    # ── Final layout ───────────────────────────────────────────────────────────
    title = f"QTF Energy Landscape — {sequence}"
    if forcefield:
        title += f" ({forcefield.upper()})"
    fig.suptitle(title, color='white', fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.15,
                    facecolor=fig.get_facecolor())
        print(f"[SAVED] {save_path}")

    if show:
        plt.show()

    plt.close()
    return fig


def plot_ensemble_landscape(results, sequence="", forcefield="", save_path=None):
    """
    Plot energy landscapes for ALL replicas overlaid on one plot.
    Good for visualizing ensemble diversity.

    Parameters
    ----------
    results : list of dicts — from manager.get_ranked_results()
              each dict must have 'tracker', 'energy', 'type', 'id'
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0f0f1a')

    INIT_COLORS = {
        'helix':  '#ff6b35',
        'sheet':  '#4ecdc4',
        'random': '#a855f7',
    }

    # ── Plot 1: All trajectories overlaid ─────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor('#1a1a2e')

    for res in results:
        tracker = res.get('tracker')
        if tracker is None or not tracker.history:
            continue
        history = np.array(tracker.history)
        color   = INIT_COLORS.get(res.get('type', 'random'), '#ffffff')
        running_min = np.minimum.accumulate(history)
        ax1.plot(running_min, color=color, linewidth=1.0, alpha=0.5)

    # Highlight best replica
    best = sorted(results, key=lambda x: x['energy'])[0]
    if best.get('tracker') and best['tracker'].history:
        h = np.minimum.accumulate(best['tracker'].history)
        ax1.plot(h, color='#ffd700', linewidth=2.5, alpha=1.0,
                 label=f"Best (E={best['energy']:.1f})")

    patches = [mpatches.Patch(color=c, label=f'{k} init')
               for k, c in INIT_COLORS.items()]
    patches.append(mpatches.Patch(color='#ffd700', label='Best replica'))
    ax1.legend(handles=patches, facecolor='#2a2a3e',
               edgecolor='#555', labelcolor='white', fontsize=9)
    ax1.set_xlabel('Function Evaluation', color='white', fontsize=10)
    ax1.set_ylabel('Energy (running min)', color='white', fontsize=10)
    ax1.set_title('All Replica Trajectories', color='white',
                  fontsize=12, fontweight='bold')
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('#444')
    ax1.grid(True, alpha=0.15, color='white')

    # ── Plot 2: Final energy distribution ─────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#1a1a2e')

    for init_type, color in INIT_COLORS.items():
        energies = [r['energy'] for r in results
                    if r.get('type') == init_type]
        if energies:
            ax2.hist(energies, bins=15, color=color, alpha=0.6,
                     label=f'{init_type} (n={len(energies)})',
                     edgecolor='white', linewidth=0.3)

    ax2.axvline(best['energy'], color='#ffd700', linewidth=2,
                linestyle='--', label=f"Best: {best['energy']:.1f}")
    ax2.set_xlabel('Final Energy', color='white', fontsize=10)
    ax2.set_ylabel('Count', color='white', fontsize=10)
    ax2.set_title('Final Energy Distribution by Init Strategy',
                  color='white', fontsize=12, fontweight='bold')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_color('#444')
    ax2.grid(True, alpha=0.15, color='white')
    ax2.legend(facecolor='#2a2a3e', edgecolor='#555',
               labelcolor='white', fontsize=9)

    title = f"QTF Ensemble Landscape — {sequence}"
    if forcefield:
        title += f" ({forcefield.upper()})"
    fig.suptitle(title, color='white', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"[SAVED] {save_path}")

    plt.show()
    plt.close()
    return fig


# ── Standalone CLI ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker_json', default=None,
                        help='Path to saved tracker JSON')
    parser.add_argument('--sequence',     default="")
    parser.add_argument('--forcefield',   default="amber")
    parser.add_argument('--save',         default="energy_landscape.png")
    parser.add_argument('--interactive-save', default=None,
                        help='Optional path to save an interactive Plotly HTML landscape')
    parser.add_argument('--show', action='store_true',
                        help='Show the matplotlib figure window after saving')
    args = parser.parse_args()

    if args.tracker_json:
        with open(args.tracker_json) as f:
            data = json.load(f)

        class FakeTracker:
            def __init__(self, d):
                self.history       = d['history']
                self.phase_markers = [tuple(x) for x in d['phase_markers']]
                self.current_iter  = d['current_iter']

        tracker = FakeTracker(data)
        plot_energy_landscape(
            tracker,
            sequence=args.sequence,
            forcefield=args.forcefield,
            save_path=args.save,
            show=args.show,
        )
        if args.interactive_save:
            plot_energy_landscape_interactive(
                tracker,
                sequence=args.sequence,
                forcefield=args.forcefield,
                save_path=args.interactive_save,
            )
    else:
        print("Provide --tracker_json to plot from saved data.")
        print("Or import plot_energy_landscape() in your code.")
