#!/usr/bin/env python
# coding: utf-8

# # Quantum Torsion Folding (QTF) — Demo Notebook
# 
# This notebook demonstrates a full end-to-end protein structure prediction run
# using the **QTF** package. It folds [Chignolin](https://www.rcsb.org/structure/5AWL)
# (`YYDPETGTWY`, PDB: `5AWL`), a well-characterised 10-residue mini-protein, and
# evaluates the ensemble of predicted structures against the experimental backbone.
# 
# **Architecture recap**
# 
# | Component | Role |
# |-----------|------|
# | Quantum Actor | Parameterised ansatz (EfficientSU2 / RealAmplitudes / brickwork) → torsion angles via statevector phases (`mode="statevector"`) or shot-based CDF (`mode="sampler"`) |
# | Classical Critic | Physics energy function (hydrophobicity, H-bonds, electrostatics, sterics) |
# | Optimiser | Three-stage COBYLA → SLSQP curriculum |
# | Ensemble | Multiple random-init replicas ranked by energy & Cα RMSD |
# 

# ## 1. Install QTF
# 
# The package is not yet published on PyPI; install it directly from the repository root.
# 

# In[ ]:


# Install QTF from the local repository (editable mode)
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "-q"], check=True)


# ## 2. Imports
# 

# In[ ]:


from qtf import QuantumBiophysicsFolder, EnsembleFoldingManager
from qtf.analysis import EnsembleRanking
from qtf.visualization import plot_structure, plot_energy_landscape, plot_ranking
from qtf.utils import get_ground_truth_backbone

import qtf
print(f"QTF version: {qtf.__version__}")


# ## 3. Configuration
# 
# Adjust these parameters to trade speed for prediction quality.
# The defaults below are chosen for a reasonably fast run; increase
# `N_RUNS` and `MAX_ITER` for higher-quality results.
# 

# In[ ]:


# Sequence & force field
SEQUENCE    = "YYDPETGTWY"  # Chignolin (10 residues)
PDB_ID      = "5AWL"         # Ground-truth structure from RCSB
FORCE_FIELD = "amber"

# Ensemble settings
N_RUNS         = 3    # Independent folding replicas
MAX_ITER       = 500  # Optimiser iterations per stage per replica
SCOUT_ATTEMPTS = 20   # Basin-hopping candidates for smart initialisation

# Circuit settings
ANSATZ  = "efficient_su2"  # "efficient_su2" | "real_amplitudes" | "brickwork"
MODE    = "statevector"    # "statevector" (exact) | "sampler" (shot-based / hardware)
SHOTS   = 4096             # ignored when MODE="statevector"


# ## 4. Run the Ensemble
# 
# Each replica starts from a different random parameter set chosen by basin-hopping
# (the best of `SCOUT_ATTEMPTS` random samples). The three-stage optimisation then
# runs: **Stage 1** collapses the chain (COBYLA, strong constraints),
# **Stage 2** refines physics (SLSQP, strong constraints), and
# **Stage 3** relaxes to a natural conformation (SLSQP, soft constraints).
# 

# In[ ]:


folder  = QuantumBiophysicsFolder(SEQUENCE, force_field=FORCE_FIELD, ansatz=ANSATZ, mode=MODE, shots=SHOTS)
manager = EnsembleFoldingManager(folder)

print(f"Sequence      : {SEQUENCE}")
print(f"Residues      : {folder.n_residues}")
print(f"DoF (angles)  : {folder.total_angles}")
print(f"Qubits        : {folder.n_qubits}")
print(f"Circuit params: {folder.n_params}")
print(f"Ansatz        : {ANSATZ}")
print(f"Mode          : {MODE}")
print(f"Shots         : {SHOTS} (ignored in statevector mode)")
print()

manager.run_ensemble(n_runs=N_RUNS, max_iter=MAX_ITER, scout_attempts=SCOUT_ATTEMPTS)

results = manager.get_results()  # sorted by ascending energy
print(f"Completed {len(results)} replica(s).")
for r in results:
    print(f"  Replica {r['id']}  seed={r['seed']}  energy={r['energy']:.4f}")


# ## 5. Load Ground Truth & Build Ranking
# 
# We download the experimental Cα trace from RCSB (cached locally as `5AWL.pdb`)
# and compute per-replica RMSD vs the ground truth alongside ensemble convergence statistics.
# 

# In[ ]:


true_ca = get_ground_truth_backbone(PDB_ID)
print(f"Ground truth: {len(true_ca)} C\u03b1 atoms from PDB {PDB_ID}")

ranking = EnsembleRanking.from_ensemble(results, ground_truth_ca=true_ca)
print()
print(ranking.summary())


# ## 6. Visualise
# 
# ### 6a. 3-D Backbone Overlay
# 
# All replicas are Kabsch-aligned to the experimental structure before display.
# The **best-by-energy** replica is highlighted in green; **best-by-RMSD** in red.
# 

# In[ ]:


plot_structure(
    ranking,
    ground_truth_ca=true_ca,
    title=f"Chignolin ({SEQUENCE}) — Predicted vs Experimental (PDB {PDB_ID})",
).show()


# ### 6b. Optimisation Energy Landscape
# 
# Energy recorded at every function evaluation across all three optimisation stages.
# 

# In[ ]:


plot_energy_landscape(
    ranking,
    title=f"Optimisation Landscape — Chignolin ({SEQUENCE})",
).show()


# ### 6c. Ensemble Ranking Table
# 
# Interactive bar chart and statistics table for all replicas.
# 

# In[ ]:


plot_ranking(
    ranking,
    title=f"Ensemble Ranking — Chignolin ({SEQUENCE})",
).show()

