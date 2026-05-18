#!/usr/bin/env python3
"""
3D Structure Alignment — ALL 6 replicas below 2.0 Å (true statevector)
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPLICAS = [
    {'id': 98,  's3': 1.7083, 'energy': 419.86},
    {'id': 194, 's3': 1.8664, 'energy':  34.58},
    {'id': 21,  's3': 1.8943, 'energy': -82.65},
    {'id': 260, 's3': 1.9030, 'energy':  42.24},
    {'id': 345, 's3': 1.9316, 'energy':  29.21},
    {'id': 148, 's3': 1.9730, 'energy': -12.59},
]
COLORS   = ['#FFD700','#FF6B6B','#4ECDC4','#A855F7','#4CAF50','#FF9800']
RESIDUES = list("YYDPETGTWY")

def load_ca(pdb_path):
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                coords.append([float(line[30:38]),
                                float(line[38:46]),
                                float(line[46:54])])
    return np.array(coords)

# ── Load ground truth ──────────────────────────────────────────────────────────
true_ca = load_ca('5AWL.pdb')
print(f"Ground truth 5AWL: {len(true_ca)} CA atoms")

# ── Load all statevector PDBs ──────────────────────────────────────────────────
for rep in REPLICAS:
    sv_path = f"outputs/sv_replicas/replica_{rep['id']}/replica_{rep['id']}_sv.pdb"
    sv_ca   = load_ca(sv_path)
    per_res = [float(np.linalg.norm(sv_ca[i] - true_ca[i])) for i in range(10)]
    rmsd    = float(np.sqrt(np.mean([d**2 for d in per_res])))
    rep['sv_ca']   = sv_ca
    rep['per_res'] = per_res
    rep['rmsd']    = rmsd
    print(f"  Replica #{rep['id']}: RMSD={rmsd:.4f} Å")

# ── Subplots: 1 combined + 6 individual = 7 panels ────────────────────────────
titles = (['ALL 6 replicas overlaid vs 5AWL'] +
          [f"Replica #{r['id']} — RMSD {r['rmsd']:.4f} Å  E={r['energy']:.2f}"
           for r in REPLICAS])

n_panels = len(titles)   # 7
n_rows   = 4             # row1=combined, rows2-4=pairs
specs    = [[{'type':'scene'},{'type':'scene'}]] * n_rows

fig = make_subplots(
    rows=n_rows, cols=2,
    specs=specs,
    subplot_titles=titles,
    vertical_spacing=0.05,
)

# ── Panel 1 (row1, col1): All overlaid ────────────────────────────────────────
fig.add_trace(go.Scatter3d(
    x=true_ca[:,0], y=true_ca[:,1], z=true_ca[:,2],
    mode='lines+markers+text',
    name='5AWL ground truth',
    line=dict(color='red', width=7, dash='dash'),
    marker=dict(size=6, color='red'),
    text=[f"{r}{i+1}" for i,r in enumerate(RESIDUES)],
    textposition='top center',
    textfont=dict(size=9, color='#ff8888'),
    legendgroup='truth',
), row=1, col=1)

for i, rep in enumerate(REPLICAS):
    fig.add_trace(go.Scatter3d(
        x=rep['sv_ca'][:,0],
        y=rep['sv_ca'][:,1],
        z=rep['sv_ca'][:,2],
        mode='lines+markers',
        name=f"#{rep['id']} ({rep['rmsd']:.3f}Å)",
        line=dict(color=COLORS[i], width=3),
        marker=dict(size=3, color=COLORS[i]),
        opacity=0.8,
        legendgroup=f"rep{rep['id']}",
    ), row=1, col=1)

# ── Panels 2-7: Individual replicas ───────────────────────────────────────────
panel_positions = [
    (1,2),  # panel 2
    (2,1),  # panel 3
    (2,2),  # panel 4
    (3,1),  # panel 5
    (3,2),  # panel 6
    (4,1),  # panel 7
]

for i, (rep, (p_row, p_col)) in enumerate(zip(REPLICAS, panel_positions)):
    sv   = rep['sv_ca']
    n_ca = len(sv)

    # Ground truth
    fig.add_trace(go.Scatter3d(
        x=true_ca[:n_ca,0], y=true_ca[:n_ca,1], z=true_ca[:n_ca,2],
        mode='lines+markers+text',
        name='5AWL ground truth',
        line=dict(color='red', width=5, dash='dash'),
        marker=dict(size=5, color='red'),
        text=[f"{r}{j+1}" for j,r in enumerate(RESIDUES[:n_ca])],
        textposition='top center',
        textfont=dict(size=9, color='#ff8888'),
        legendgroup='truth',
        showlegend=(i == 0),
    ), row=p_row, col=p_col)

    # Predicted
    fig.add_trace(go.Scatter3d(
        x=sv[:,0], y=sv[:,1], z=sv[:,2],
        mode='lines+markers+text',
        name=f"Replica #{rep['id']}",
        line=dict(color=COLORS[i], width=5),
        marker=dict(size=5, color=COLORS[i]),
        text=[f"{r}{j+1}" for j,r in enumerate(RESIDUES[:n_ca])],
        textposition='bottom center',
        textfont=dict(size=9, color=COLORS[i]),
        legendgroup=f"rep{rep['id']}",
        showlegend=(i == 0 or True),
    ), row=p_row, col=p_col)

    # Gray connector lines per residue
    for j in range(n_ca):
        fig.add_trace(go.Scatter3d(
            x=[sv[j,0], true_ca[j,0]],
            y=[sv[j,1], true_ca[j,1]],
            z=[sv[j,2], true_ca[j,2]],
            mode='lines',
            line=dict(color='#555', width=1.5),
            opacity=0.5,
            showlegend=False,
            hovertemplate=(f"Residue {j+1} ({RESIDUES[j]}): "
                           f"{rep['per_res'][j]:.3f} Å"),
        ), row=p_row, col=p_col)

# ── Layout ─────────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=('QTF — 3D Structure Alignment vs Ground Truth 5AWL<br>'
              '<sup>YYDPETGTWY | Amber | Brickwork ansatz | '
              'TRUE statevector structures | All replicas below 2.0 Å</sup>'),
        font=dict(size=16, color='white'), x=0.5,
    ),
    template='plotly_dark',
    height=560 * n_rows,
    paper_bgcolor='#0f0f1a',
    font=dict(color='white'),
    legend=dict(
        font=dict(size=10),
        bgcolor='rgba(0,0,0,0.3)',
        bordercolor='#444',
    ),
)

# Scene settings
for idx in range(1, n_panels + 1):
    sk = f'scene{idx}' if idx > 1 else 'scene'
    fig.update_layout(**{sk: dict(
        xaxis_title='X (Å)',
        yaxis_title='Y (Å)',
        zaxis_title='Z (Å)',
        aspectmode='data',
        bgcolor='#1a1a2e',
        xaxis=dict(gridcolor='#333', color='white'),
        yaxis=dict(gridcolor='#333', color='white'),
        zaxis=dict(gridcolor='#333', color='white'),
    )})

# ── Save ───────────────────────────────────────────────────────────────────────
out = 'qtf_3d_all_sv_alignment.html'
fig.write_html(out)
print(f"\n✅ Saved → {out}")
print(f"   Open in browser — rotate, zoom, hover!")
print(f"\n   Summary:")
for rep in REPLICAS:
    print(f"   Replica #{rep['id']:>4}  RMSD={rep['rmsd']:.4f} Å  "
          f"worst_res={max(range(10), key=lambda i: rep['per_res'][i])+1}"
          f"({RESIDUES[max(range(10), key=lambda i: rep['per_res'][i])]})"
          f"={max(rep['per_res']):.3f} Å")
