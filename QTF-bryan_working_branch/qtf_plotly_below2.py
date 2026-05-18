#!/usr/bin/env python3
"""
Top replicas below 2.0 Å RMSD — Plotly visualization
Uses statevector RMSD (rmsd_stage3) vs ground truth 5AWL

Run: python3 qtf_plotly_below2.py
"""
import json, os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Load all replicas ──────────────────────────────────────────────────────────
base    = "outputs/no_skip/slurm_YYDPETGTWY_amber"
all_r   = []
below2  = []

for i in range(400):
    p = f"{base}/replica_{i}/replica_{i}_result.json"
    if not os.path.exists(p):
        continue
    r = json.load(open(p))
    s3 = r.get('rmsd_stage3')
    s1 = r.get('rmsd_stage1')
    s2 = r.get('rmsd_stage2')
    e  = r.get('energy')
    if s3 is None:
        continue
    all_r.append({'id': i, 's1': s1, 's2': s2, 's3': s3, 'energy': e})
    if s3 < 2.0:
        below2.append({'id': i, 's1': s1, 's2': s2, 's3': s3, 'energy': e})

below2.sort(key=lambda x: x['s3'])
print(f"Total replicas loaded : {len(all_r)}")
print(f"Below 2.0 Å (S3)      : {len(below2)}")

# ── Colors ─────────────────────────────────────────────────────────────────────
COLORS  = ['#FFD700','#FF6B6B','#4ECDC4','#A855F7','#4CAF50','#FF9800']
IDS     = [r['id'] for r in below2]
S3_VALS = [r['s3'] for r in below2]
S1_VALS = [r['s1'] for r in below2]
S2_VALS = [r['s2'] for r in below2]
E_VALS  = [r['energy'] for r in below2]
LABELS  = [f"Replica #{r['id']}" for r in below2]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        '🏆 Top Replicas Below 2.0 Å — Stage RMSD Comparison',
        '⚡ Energy vs Stage 3 RMSD (all 400 replicas)',
        '📉 Stage Progression — S1 → S2 → S3',
        '📊 Stage 3 RMSD Distribution (all 400 replicas)',
    ],
    vertical_spacing=0.20,
    horizontal_spacing=0.12,
)

# ── Plot 1: Grouped bars S1, S2, S3 for below-2 replicas ──────────────────────
for stage, vals, col in [
    ('Stage 1 (COBYLA)',  S1_VALS, '#ff6b35'),
    ('Stage 2 (SLSQP)',   S2_VALS, '#4ecdc4'),
    ('Stage 3 (Relax)',   S3_VALS, '#a855f7'),
]:
    fig.add_trace(go.Bar(
        name=stage,
        x=LABELS,
        y=vals,
        marker_color=col,
        opacity=0.88,
        text=[f'{v:.3f}Å' for v in vals],
        textposition='outside',
        textfont=dict(size=10, color='white'),
    ), row=1, col=1)

# 2.0 Å target line
fig.add_hline(
    y=2.0, line_dash='dash', line_color='red', line_width=2,
    annotation_text='2.0 Å target',
    annotation_font=dict(color='red', size=11),
    annotation_position='top right',
    row=1, col=1,
)

# ── Plot 2: Energy vs S3 RMSD all replicas ────────────────────────────────────
all_s3 = [r['s3'] for r in all_r]
all_e  = [r['energy'] for r in all_r]
all_id = [r['id'] for r in all_r]

# Clip outliers
e_lo = float(np.percentile(all_e, 2))
e_hi = float(np.percentile(all_e, 98))
filtered = [(s, e, i) for s, e, i in zip(all_s3, all_e, all_id)
            if e_lo <= e <= e_hi]
f_s3, f_e, f_id = zip(*filtered)

fig.add_trace(go.Scatter(
    x=f_s3, y=f_e,
    mode='markers',
    name='All replicas',
    marker=dict(
        size=5,
        color=f_s3,
        colorscale='RdYlGn_r',
        opacity=0.5,
        showscale=True,
        colorbar=dict(
            x=1.02, len=0.45, y=0.78,
            tickfont=dict(color='white'),
            title=dict(text='RMSD (Å)', font=dict(color='white')),
        ),
    ),
    hovertemplate='Replica #%{customdata}<br>RMSD: %{x:.3f} Å<br>Energy: %{y:.1f}',
    customdata=f_id,
), row=1, col=2)

# Highlight below-2 replicas
fig.add_trace(go.Scatter(
    x=S3_VALS, y=E_VALS,
    mode='markers+text',
    name='Below 2.0 Å',
    marker=dict(
        size=16,
        color=COLORS[:len(below2)],
        symbol='star',
        line=dict(width=1.5, color='white'),
    ),
    text=[f"#{i}" for i in IDS],
    textposition='top center',
    textfont=dict(size=10, color='white'),
    hovertemplate='Replica #%{text}<br>RMSD: %{x:.3f} Å<br>Energy: %{y:.1f}',
), row=1, col=2)

# Vertical 2.0 Å line
fig.add_vline(
    x=2.0, line_dash='dash', line_color='red', line_width=1.5,
    annotation_text='2.0 Å', annotation_font=dict(color='red'),
    row=1, col=2,
)

# Pearson r annotation
corr = float(np.corrcoef(f_s3, f_e)[0, 1])
fig.add_annotation(
    x=0.98, y=0.05,
    xref='x2 domain', yref='y2 domain',
    text=f'Pearson r = {corr:.4f}',
    showarrow=False,
    font=dict(size=11, color='#FFD700'),
    xanchor='right', yanchor='bottom',
    bgcolor='rgba(0,0,0,0.4)',
)

# ── Plot 3: Stage progression lines ───────────────────────────────────────────
x_pos  = [0, 1, 2]
stages = ['Stage 1', 'Stage 2', 'Stage 3']

for i, rep in enumerate(below2):
    vals = [rep['s1'], rep['s2'], rep['s3']]
    fig.add_trace(go.Scatter(
        x=x_pos, y=vals,
        mode='lines+markers',
        name=f"Replica #{rep['id']}",
        line=dict(color=COLORS[i], width=2.5),
        marker=dict(size=10, color=COLORS[i],
                    line=dict(width=1, color='white')),
        hovertemplate=(
            f"Replica #{rep['id']}<br>"
            "%{xaxis.ticktext[x]}: %{y:.4f} Å"
        ),
        showlegend=True,
    ), row=2, col=1)

fig.add_hline(
    y=2.0, line_dash='dash', line_color='red', line_width=1.5,
    annotation_text='2.0 Å target',
    annotation_font=dict(color='red', size=10),
    row=2, col=1,
)

fig.update_xaxes(
    tickvals=x_pos, ticktext=stages,
    tickfont=dict(size=11),
    row=2, col=1,
)

# ── Plot 4: Stage 3 RMSD distribution all replicas ────────────────────────────
fig.add_trace(go.Histogram(
    x=all_s3, nbinsx=35,
    marker_color='#a855f7', opacity=0.75,
    name='S3 RMSD all replicas',
    hovertemplate='RMSD: %{x:.3f} Å<br>Count: %{y}',
), row=2, col=2)

# Mark each below-2 replica
for i, rep in enumerate(below2):
    fig.add_vline(
        x=rep['s3'],
        line_color=COLORS[i],
        line_width=2,
        line_dash='dot',
        annotation_text=f"#{rep['id']}<br>{rep['s3']:.3f}Å",
        annotation_font=dict(size=9, color=COLORS[i]),
        annotation_position='top',
        row=2, col=2,
    )

fig.add_vline(
    x=2.0, line_dash='dash', line_color='red', line_width=1.5,
    annotation_text='2.0 Å',
    annotation_font=dict(color='red'),
    row=2, col=2,
)

# ── Layout ─────────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=(
            'QTF — YYDPETGTWY / 5AWL  |  Replicas Below 2.0 Å vs Ground Truth<br>'
            '<sup>Statevector RMSD (Stage 3) | 400 replicas | Amber | Brickwork ansatz</sup>'
        ),
        font=dict(size=17, color='white'),
        x=0.5,
    ),
    template='plotly_dark',
    height=850,
    barmode='group',
    paper_bgcolor='#0f0f1a',
    plot_bgcolor='#1a1a2e',
    font=dict(color='white', family='Arial'),
    legend=dict(
        orientation='h',
        yanchor='bottom', y=-0.18,
        xanchor='center', x=0.5,
        font=dict(size=10),
        bgcolor='rgba(0,0,0,0.3)',
    ),
)

fig.update_yaxes(title_text='RMSD (Å)',  gridcolor='#333', row=1, col=1)
fig.update_yaxes(title_text='Energy',    gridcolor='#333', row=1, col=2)
fig.update_yaxes(title_text='RMSD (Å)',  gridcolor='#333', row=2, col=1)
fig.update_yaxes(title_text='Count',     gridcolor='#333', row=2, col=2)
fig.update_xaxes(title_text='Replica',   gridcolor='#333', row=1, col=1)
fig.update_xaxes(title_text='RMSD (Å)',  gridcolor='#333', row=1, col=2)
fig.update_xaxes(title_text='RMSD (Å)',  gridcolor='#333', row=2, col=2)

# ── Save ───────────────────────────────────────────────────────────────────────
out = 'qtf_below2_visualization.html'
fig.write_html(out)
print(f"\n✅ Saved → {out}")
print("   Open in your browser to interact!")
