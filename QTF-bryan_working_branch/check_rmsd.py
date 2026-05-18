import numpy as np

true_ca = []
with open('5AWL.pdb') as f:
    for line in f:
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            true_ca.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
true_ca = np.array(true_ca)

pred_ca = []
with open('outputs/no_skip/slurm_YYDPETGTWY_amber/replica_98/replica_98_ca.pdb') as f:
    for line in f:
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            pred_ca.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
pred_ca = np.array(pred_ca)

print('Ground truth:'); print(true_ca)
print('Predicted:');    print(pred_ca)

P = pred_ca - pred_ca.mean(0)
Q = true_ca - true_ca.mean(0)
H = P.T @ Q
U, S, Vt = np.linalg.svd(H)
d = np.linalg.det(U) * np.linalg.det(Vt) < 0
if d:
    S[-1] = -S[-1]; U[:,-1] = -U[:,-1]
R = U @ Vt
aligned = P @ R + true_ca.mean(0)
rmsd = np.sqrt(np.mean(np.sum((aligned - true_ca)**2, axis=1)))
print(f'Manual RMSD: {rmsd:.4f} A')
for i,(a,t) in enumerate(zip(aligned, true_ca)):
    print(f'  Residue {i+1}: {np.linalg.norm(a-t):.3f} A')
