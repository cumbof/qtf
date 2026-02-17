import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import urllib.request
import os

import os
import urllib.request
import numpy as np


def get_ground_truth_backbone(ref_pdb_id, average_backbone=False):
    """
    Fetch backbone CA coordinates for a reference PDB structure.

    Parameters
    ----------
    ref_pdb_id : str
        4-letter PDB ID (e.g., "2JOF").
    average_backbone : True or False
        - False: use only the first MODEL (current behavior).
        - True: average CA coordinates across all MODELS (NMR ensembles).
          For non-NMR PDBs (no MODEL/ENDMDL records), the file is treated as a single model.

    Returns
    -------
    np.ndarray
        Array of shape (N_res, 3) with CA coordinates.
    """
    url = f"https://files.rcsb.org/download/{ref_pdb_id}.pdb"
    filename = f"{ref_pdb_id}.pdb"

    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)

    coords_per_model = []
    current_coords = []
    has_models = False

    with open(filename, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                # Starting a new model; if we somehow had coords, store them
                if current_coords:
                    coords_per_model.append(np.array(current_coords, dtype=float))
                    current_coords = []
                has_models = True

            elif line.startswith("ENDMDL"):
                # End of a model: store and, if using "first", stop here
                if current_coords:
                    coords_per_model.append(np.array(current_coords, dtype=float))
                    current_coords = []

                if average_backbone == False:
                    break

            elif line.startswith("ATOM") and line[12:16] == " CA ":
                # Strictly match CA with proper spacing
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                current_coords.append([x, y, z])

        # If there were no MODEL/ENDMDL records, treat as a single model PDB
        if not has_models and current_coords:
            coords_per_model.append(np.array(current_coords, dtype=float))

    if not coords_per_model:
        raise ValueError(f"No CA coordinates found in PDB {ref_pdb_id}")

    if average_backbone == False:
        return coords_per_model[0]

    if average_backbone== True:
        # Ensure all models have the same number of residues
        lengths = {arr.shape[0] for arr in coords_per_model}
        if len(lengths) != 1:
            raise ValueError(
                f"Models in {ref_pdb_id} have different numbers of CA atoms: {lengths}. "
                "Cannot safely average across models."
            )

        stacked = np.stack(coords_per_model, axis=0)  # (n_models, n_res, 3)
        return np.mean(stacked, axis=0)

    raise ValueError(f"Unknown backbone calculation mode: {average_backbone!r}")

def calculate_physics_metrics(coords):
    end_to_end = np.linalg.norm(coords[0] - coords[-1])
    centroid = np.mean(coords, axis=0)
    rg = np.sqrt(np.mean(np.sum((coords - centroid)**2, axis=1)))
    return end_to_end, rg

