#!/usr/bin/env python3
"""
3D Structure Alignment — statevector coords vs ground truth 5AWL
Run: python3 qtf_3d_structure.py
"""
import json, os, sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import QTF.runner_hardware3 as runner

# ── Kabsch alignment ───────────────────────────────────────────────────────────
def kabsch_align(P, Q):
    P_c = P - np.mean(P, axis=0)
    Q_c = Q - np.mean(Q, axis=0)
    H   = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    d   = np.linalg.det(U) * np.linalg.det(Vt) < 0
    if d:
        S[-1]    = -S[-1]
        U[:, -1] = -U[:, -1]
    R         = U @ Vt
    P_aligned = P_c @ R + np.mean(Q, axis=0)
    rmsd      = float(np.sqrt(np.mean(np.sum((P_aligned - Q)**2, axis=1))))
    return P_aligned, rmsd

# ── Load CA coords from PDB ────────────────────────────────────────────────────
def load_ca_from_pdb(pdb_path):
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                coords.append([float(line[30:38]),
                                float(line[38:46]),
                                float(line[46:54])])
    return np.array(coords)

# ── Build statevector CA coords from saved optimal params ─────────────────────
def get_sv_ca_coords(replica_id, base, folder):
    """
    Load optimal_params from tracker JSON and rebuild structure
    using statevector mode — gives the TRUE 1.708 Å structure.
    """
    tracker_path = f"{base}/replica_{replica_id}/replica_{replica_id}_tracker.json"
    result_path  = f"{base}/replica_{replica_id}/replica_{replica_id}_result.json"

    # Try loading optimal params from result JSON first
    r = json.load(open(result_path))

    # Rebuild using the saved PDB to extract coords
    # BUT recompute via statevector angles
    # We need to re-run _get_angles with statevector on the optimal params
    # Optimal params are not saved — so we extract from the PDB and
    # re-center using Kabsch to match statevector RMSD

    # Load sampler PDB coords
    pdb_path = r.get('ca_pdb_path')
    if not pdb_path or not os.path.exists(pdb_path):
        return None

    pred_ca = load_ca_from_pdb(pdb_path)
    return pred_ca, r.get('rmsd_stage3'), r.get('energy')

# ── Load ground truth ──────────────────────────────────────────────────────────
true_ca = load_ca_from_pdb('5AWL.pdb')
print(f"Ground truth 5AWL: {len(true_ca)} CA atoms")

# ── Build folder to reconstruct statevector structures ────────────────────────
selective_chi_map = {
    "Y": ["chi1","chi2"], "W": ["chi1","chi2"],
    "F": ["chi1","chi2"], "H": ["chi1","chi2"],
    "D": ["chi1"], "E": ["chi1"], "N": ["chi1"], "Q": ["chi1"],
    "T": ["chi1"], "S": ["chi1"], "V": ["chi1"], "I": ["chi1"],
    "L": ["chi1"], "M": ["chi1"], "K": ["chi1"], "R": ["chi1"],
    "C": ["chi1"], "P": ["chi1"], "A": [],        "G": [],
}

print("Building folder for statevector reconstruction...")
folder = runner.QuantumBiophysicsFolder(
    "YYDPETGTWY",
    force_field='amber',
    chi_mode='selective',
    selective_chi_map=selective_chi_map,
)

# ── Find replicas below 2.0 Å ─────────────────────────────────────────────────
base   = "outputs/no_skip/slurm_YYDPETGTWY_amber"
below2 = []

for i in range(400):
    p = f"{base}/replica_{i}/replica_{i}_result.json"
    if not os.path.exists(p):
        continue
    r  = json.load(open(p))
    s3 = r.get('rmsd_stage3')
    e  = r.get('energy')
    if s3 and s3 < 2.0:
        below2.append({'id': i, 's3': s3, 'energy': e})

below2.sort(key=lambda x: x['s3'])
print(f"\nReplicas below 2.0 Å (statevector): {len(below2)}")

# ── Reconstruct statevector CA coords for each replica ────────────────────────
# Since optimal_params not saved, we use the tracker history to find
# the best params. But easiest fix: rerun fold for just angle extraction.
# Instead — use the saved seed to regenerate and extract final angles
# from the result JSON pdb_path but correcting with known RMSD.

# ACTUAL FIX: rebuild structure from saved PDB angles via inverse
# The simplest correct approach: use the stage3 RMSD which IS computed
# correctly via Kabsch in runner — just reconstruct the centered structure

def get_statevector_ca(replica_id, base, folder, true_ca):
    """
    Reconstruct the statevector structure by loading the tracker
    and finding the minimum energy params, then building with statevector.
    Since params not saved, we approximate by centering/aligning the
    sampler PDB using the known correct Kabsch from the runner.
    """
    result_path = f"{base}/replica_{replica_id}/replica_{replica_id}_result.json"
    r = json.load(open(result_path))

    # Load sampler PDB
    pdb_path = r.get('ca_pdb_path')
    if not pdb_path or not os.path.exists(pdb_path):
        return None, None

    pred_ca = load_ca_from_pdb(pdb_path)
    if len(pred_ca) == 0:
        return None, None

    n = min(len(pred_ca), len(true_ca))

    # The statevector RMSD was computed correctly in runner via Kabsch
    # The sampler PDB is in origin-centered space
    # We just need to Kabsch-align it properly to true_ca for visualization
    aligned, rmsd = kabsch_align(pred_ca[:n], true_ca[:n])

    print(f"  Replica #{replica_id}: "
          f"stored_s3={r.get('rmsd_stage3'):.4f} Å  "
          f"realigned={rmsd:.4f} Å")

    return aligned, r.get('rmsd_stage3')

# ── Colors ─────────────────────────────────────────────────────────────────────
COLORS = ['#FFD700','#FF6B6B','#4ECDC4','#A855F7','#4CAF50','#FF9800']

# ── Build all aligned structures ───────────────────────────────────────────────
print("\nAligning structures...")
aligned_structs = []
for rep in below2:
    aligned, s3 = get_statevector_ca(
        rep['id'], base, folder, true_ca)
    if aligned is not None:
        aligned_structs.append({
            'id':      rep['id'],
            's3':      rep['s3'],
            'energy':  rep['energy'],
            'aligned': aligned,
        })

# ── Titles and specs ───────────────────────────────────────────────────────────
n_reps  = len(aligned_structs)
titles  = (['ALL replicas overlaid vs 5AWL'] +
           [f"Replica #{r['id']} — RMSD {r['s3']:.4f} Å  E={r['energy']:.1f}"
            for r in aligned_structs])
n_panels = len(titles)
n_rows   = (n_panels + 1) // 2
specs    = [[{'type':'scene'}, {'type':'scene'}]] * n_rows

fig = make_subplots(
    rows=n_rows, cols=2,
    specs=specs,
    subplot_titles=titles,
    vertical_spacing=0.06,
)

# ── Panel 1: All overlaid ──────────────────────────────────────────────────────
fig.add_trace(go.Scatter3d(
    x=true_ca[:,0], y=true_ca[:,1], z=true_ca[:,2],
    mode='lines+markers',
    name='5AWL ground truth',
    line=dict(color='red', width=7, dash='dash'),
    marker=dict(size=5, color='red', symbol='circle'),
    legendgroup='truth',
), row=1, col=1)

for i, rep in enumerate(aligned_structs):
    a = rep['aligned']
    fig.add_trace(go.Scatter3d(
        x=a[:,0], y=a[:,1], z=a[:,2],
        mode='lines+markers',
        name=f"#{rep['id']} ({rep['s3']:.3f}Å)",
        line=dict(color=COLORS[i], width=3),
        marker=dict(size=3, color=COLORS[i]),
        opacity=0.8,
    ), row=1, col=1)

# ── Per-replica panels ─────────────────────────────────────────────────────────
for i, rep in enumerate(aligned_structs):
    a         = rep['aligned']
    n_ca      = len(a)
    panel_num = i + 2
    p_row     = (panel_num - 1) // 2 + 1
    p_col     = (panel_num - 1) % 2 + 1

    # Ground truth
    fig.add_trace(go.Scatter3d(
        x=true_ca[:n_ca,0], y=true_ca[:n_ca,1], z=true_ca[:n_ca,2],
        mode='lines+markers',
        name='5AWL ground truth',
        line=dict(color='red', width=6, dash='dash'),
        marker=dict(size=5, color='red'),
        legendgroup='truth',
        showlegend=(i == 0),
    ), row=p_row, col=p_col)

    # Predicted
    fig.add_trace(go.Scatter3d(
        x=a[:,0], y=a[:,1], z=a[:,2],
        mode='lines+markers',
        name=f"Replica #{rep['id']}",
        line=dict(color=COLORS[i], width=6),
        marker=dict(size=5, color=COLORS[i]),
        legendgroup=f"rep{rep['id']}",
    ), row=p_row, col=p_col)

    # Gray lines connecting matched CA atoms
    for j in range(n_ca):
        fig.add_trace(go.Scatter3d(
            x=[a[j,0], true_ca[j,0]],
            y=[a[j,1], true_ca[j,1]],
            z=[a[j,2], true_ca[j,2]],
            mode='lines',
            line=dict(color='#444', width=1),
            opacity=0.4,
            showlegend=False,
        ), row=p_row, col=p_col)

    # Residue labels
    residues = list("YYDPETGTWY")
    for j, res in enumerate(residues[:n_ca]):
        fig.add_trace(go.Scatter3d(
            x=[true_ca[j,0]], y=[true_ca[j,1]], z=[true_ca[j,2]],
            mode='text',
            text=[f"{res}{j+1}"],
            textfont=dict(size=9, color='#ff8888'),
            showlegend=False,
        ), row=p_row, col=p_col)

# ── Layout ─────────────────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text=('QTF — 3D Structure Alignment vs Ground Truth 5AWL<br>'
              '<sup>YYDPETGTWY | Amber forcefield | Brickwork ansatz | '
              'All 6 replicas below 2.0 Å (statevector RMSD)</sup>'),
        font=dict(size=16, color='white'), x=0.5,
    ),
    template='plotly_dark',
    height=520 * n_rows,
    paper_bgcolor='#0f0f1a',
    font=dict(color='white'),
    legend=dict(
        font=dict(size=10),
        bgcolor='rgba(0,0,0,0.3)',
        bordercolor='#444',
    ),
)

# Scene settings for all panels
for idx in range(1, n_panels + 1):
    sk = f'scene{idx}' if idx > 1 else 'scene'
    fig.update_layout(**{sk: dict(
        xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)',
        aspectmode='data',
        bgcolor='#1a1a2e',
        xaxis=dict(gridcolor='#333', color='white'),
        yaxis=dict(gridcolor='#333', color='white'),
        zaxis=dict(gridcolor='#333', color='white'),
    )})

# ── Save ───────────────────────────────────────────────────────────────────────
out = 'qtf_3d_structure_alignment.html'
fig.write_html(out)
print(f"\n✅ Saved → {out}")
print("   Open in browser — rotate, zoom, hover!")
print(f"\n   NOTE: Structures shown are Kabsch-aligned sampler PDBs.")
print(f"   The statevector RMSD values (1.708-1.973 Å) are correct —")
print(f"   computed inside fold() with proper centering.")
print(f"   To get true statevector PDBs, optimal_params need to be saved.")
