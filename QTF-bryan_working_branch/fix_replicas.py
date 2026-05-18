content = open('qtf_3d_all_sv.py').read()
old = """REPLICAS = [
    {'id': 98,  's3': 1.7083},
    {'id': 194, 's3': 1.8664},
    {'id': 21,  's3': 1.8943},
    {'id': 260, 's3': 1.9030},
    {'id': 345, 's3': 1.9316},
    {'id': 148, 's3': 1.9730},
]"""
new = """REPLICAS = [
    {'id': 98,  's3': 1.7083, 'energy': 419.86},
    {'id': 194, 's3': 1.8664, 'energy':  34.58},
    {'id': 21,  's3': 1.8943, 'energy': -82.65},
    {'id': 260, 's3': 1.9030, 'energy':  42.24},
    {'id': 345, 's3': 1.9316, 'energy':  29.21},
    {'id': 148, 's3': 1.9730, 'energy': -12.59},
]"""
old_titles = """titles = (['ALL 6 replicas overlaid vs 5AWL'] +
          [f"Replica #{r['id']} — RMSD {r['rmsd']:.4f} Å"
           for r in REPLICAS])"""
new_titles = """titles = (['ALL 6 replicas overlaid vs 5AWL'] +
          [f"Replica #{r['id']} — RMSD {r['rmsd']:.4f} Å  E={r['energy']:.2f}"
           for r in REPLICAS])"""
content = content.replace(old, new).replace(old_titles, new_titles)
open('qtf_3d_all_sv.py', 'w').write(content)
print("Done!")
