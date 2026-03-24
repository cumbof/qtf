import numpy as np
import QTF.runner as runner

seq = "YYDPETGTWY"
ff = "amber"

folder = runner.QuantumBiophysicsFolder(sequence=seq, force_field=ff)
folder.current_stage = 3

print("\n--- Hydrophobic/SASA diagnostics ---")
print("total atoms:", len(folder.static_labels))

mh = getattr(folder, "mask_hydrophobic", None)
if mh is None:
    print("mask_hydrophobic: MISSING")
else:
    print("mask_hydrophobic count:", int(np.sum(mh)))

    hyd_labels = [lbl for lbl, flag in zip(folder.static_labels, mh) if flag]
    print("example hydrophobic labels (up to 50):", hyd_labels[:50])

print("sequence:", folder.sequence)

def eval_terms(angle_vec):
    dummy_params = np.zeros(folder.n_params, dtype=float)
    orig_get_angles = folder._get_angles
    try:
        folder._get_angles = lambda _p: angle_vec
        E = float(folder.energy_function(dummy_params, return_terms=True))
        terms = getattr(folder, "last_energy_terms", {}) or {}
        # normalize numpy scalars
        terms = {str(k): float(v) for k, v in terms.items()}
        return E, terms
    finally:
        folder._get_angles = orig_get_angles

# sample a handful of random full-length angle sets
samples = []
for i in range(25):
    ang = np.random.uniform(-np.pi, np.pi, folder.total_angles)
    E, terms = eval_terms(ang)
    samples.append((E, terms))

# print a few
for i, (E, terms) in enumerate(samples[:5]):
    print(f"\nSample {i} E={E:.3f}")
    print(" keys:", sorted(terms.keys()))
    print(" sasa:", terms.get("sasa"), " hbond:", terms.get("hbond"))

# summarize min/max
sasa_vals = [t.get("sasa", 0.0) for _, t in samples]
hbond_vals = [t.get("hbond", 0.0) for _, t in samples]

print("\nSASA min/max:", min(sasa_vals), max(sasa_vals))
print("HBOND min/max:", min(hbond_vals), max(hbond_vals))
