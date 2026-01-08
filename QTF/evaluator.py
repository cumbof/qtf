import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import urllib.request
import os

def get_ground_truth_backbone(ref_pdb_id):
    url = f"https://files.rcsb.org/download/{ref_pdb_id}.pdb"
    filename = f"{ref_pdb_id}.pdb"
    if not os.path.exists(filename): urllib.request.urlretrieve(url, filename)
    coords_ca = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("ENDMDL"): break
            if line.startswith("ATOM") and "CA" in line[12:16]:
                coords_ca.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(coords_ca)

def calculate_physics_metrics(coords):
    end_to_end = np.linalg.norm(coords[0] - coords[-1])
    centroid = np.mean(coords, axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - centroid)**2, axis=1)))
    return end_to_end, rg

def kabsch_backbone_align(P, Q):
    # Specialized Kabsch for Plotting (Returns Aligned Coords)
    P_c = P - np.mean(P, axis=0)
    Q_c = Q - np.mean(Q, axis=0)
    H = np.dot(P_c.T, Q_c)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T) # R = V U^T
    if np.linalg.det(R) < 0: 
        Vt[2,:] *= -1; R = np.dot(Vt.T, U.T)
    aligned_P = np.dot(P_c, R) + np.mean(Q, axis=0) # Re-center on Q
    diff = aligned_P - Q
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd, aligned_P

