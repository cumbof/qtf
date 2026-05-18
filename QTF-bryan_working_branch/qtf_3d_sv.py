#!/usr/bin/env python3
"""
3D Structure Alignment — TRUE statevector structure vs ground truth 5AWL
Replica 98 — RMSD 1.7083 Å
"""
import numpy as np
import plotly.graph_objects as go

def load_ca(pdb_path):
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                coords.append([float(line[30:38]),
                                float(line[38:46]),
                                float(line[46:54])])
    return np.array(coords)

true_ca = load_ca('5AWL.pdb')
sv_ca   = load_ca('outputs/replica_98_final/replica_98_sv.pdb')

residues = list("YYDPETGTWY")
per_res  = [float(np.linalg.norm(sv_ca[i] - true_ca[i])) for i in range(10)]
rmsd     = float(np.sqrt(np.mean([d**2 for d in per_res])))

fig = go.Figure()

# Ground truth
fig.add_trace(go.Scatter3d(
    x=true_ca[:,0], y=true_ca[:,1], z=true_ca[:,2],
    mode='lines+markers+text',
    name='5AWL (ground truth)',
    line=dict(color='red', width=6, dash='dash'),
    marker=dict(size=6, color='red'),
    text=[f"{r}{i+1}" for i,r in enumerate(residues)],
    textposition='top center',
    textfont=dict(size=10, color='#ff8888'),
))

# Predicted statevector
fig.add_trace(go.Scatter3d(
    x=sv_ca[:,0], y=sv_ca[:,1], z=sv_ca[:,2],
    mode='lines+markers+text',
    name=f'Replica #98 — Statevector (RMSD {rmsd:.4f} Å)',
    line=dict(color='#FFD700', width=6),
    marker=dict(size=6, color='#FFD700'),
    text=[f"{r}{i+1}" for i,r in enumerate(residues)],
    textposition='bottom center',
    textfont=dict(size=10, color='#FFD700'),
))

# Gray lines connecting matched CA atoms
for i in range(10):
    fig.add_trace(go.Scatter3d(
        x=[sv_ca[i,0], true_ca[i,0]],
        y=[sv_ca[i,1], true_ca[i,1]],
        z=[sv_ca[i,2], true_ca[i,2]],
        mode='lines',
        line=dict(color='gray', width=2),
        opacity=0.5,
        showlegend=False,
        hovertemplate=f'Residue {i+1} ({residues[i]}): {per_res[i]:.3f} Å',
    ))

fig.update_layout(
    title=dict(
        text=(f'3D Structure Alignment — Replica #98 vs 5AWL<br>'
              f'<sup>RMSD = {rmsd:.4f} Å | YYDPETGTWY | Amber | '
              f'Statevector (true structure)</sup>'),
        font=dict(size=16, color='white'), x=0.5,
    ),
    template='plotly_dark',
    height=750,
    paper_bgcolor='#0f0f1a',
    font=dict(color='white'),
    scene=dict(
        xaxis_title='X (Å)',
        yaxis_title='Y (Å)',
        zaxis_title='Z (Å)',
        aspectmode='data',
        bgcolor='#1a1a2e',
        xaxis=dict(gridcolor='#333', color='white'),
        yaxis=dict(gridcolor='#333', color='white'),
        zaxis=dict(gridcolor='#333', color='white'),
    ),
    legend=dict(
        font=dict(size=11),
        bgcolor='rgba(0,0,0,0.4)',
        x=0.01, y=0.99,
    ),
    annotations=[dict(
        text=(
            '<br>'.join([
                f"Residue {i+1} ({residues[i]}): {per_res[i]:.3f} Å"
                for i in range(10)
            ])
        ),
        xref='paper', yref='paper',
        x=1.0, y=0.5,
        showarrow=False,
        font=dict(size=10, color='white'),
        align='left',
        bgcolor='rgba(0,0,0,0.4)',
        bordercolor='#444',
    )]
)

out = 'qtf_replica98_sv_alignment.html'
fig.write_html(out)
print(f"✅ Saved → {out}")
print(f"   RMSD = {rmsd:.4f} Å")
print(f"\n   Per-residue distances:")
for i,(r,d) in enumerate(zip(residues, per_res)):
    bar = '█' * int(d * 5)
    print(f"   {i+1:>2} {r}: {d:.3f} Å  {bar}")
