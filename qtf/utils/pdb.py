"""PDB I/O and biophysical utility functions."""

from __future__ import annotations

import os
import urllib.request

import numpy as np


def save_pdb(
    coords: np.ndarray,
    labels: list,
    sequence: str,
    filename: str = "structure.pdb",
    energy: float = 0.0,
) -> None:
    """Write a PDB file from predicted coordinates.

    Parameters
    ----------
    coords:
        Atom coordinate array, shape ``(N_atoms, 3)``.
    labels:
        List of ``(res_id, atom_name, element)`` tuples matching *coords*.
    sequence:
        Single-letter amino acid sequence.
    filename:
        Output file path.
    energy:
        Final energy value stored in a ``REMARK`` record.
    """
    with open(filename, "w") as f:
        f.write(f"REMARK   1 ENERGY: {energy:.3f}\n")
        for k, (pos, (res_id, atom_name, elem)) in enumerate(zip(coords, labels)):
            res_name = sequence[res_id]
            f.write(
                f"ATOM  {k + 1:>5}  {atom_name:<4} {res_name:>3} A{res_id + 1:>4}"
                f"    {pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
                f"  1.00  0.00           {elem:>2}\n"
            )


def get_ground_truth_backbone(pdb_id: str, cache_dir: str = ".") -> np.ndarray:
    """Download (or load from cache) the Cα coordinates of a PDB entry.

    Parameters
    ----------
    pdb_id:
        Four-character PDB identifier (e.g. ``"5AWL"``).
    cache_dir:
        Directory where the downloaded PDB file is stored/read from.

    Returns
    -------
    ndarray, shape (N_residues, 3)
        Cα Cartesian coordinates from the first model in the PDB file.
    """
    pdb_id = pdb_id.upper()
    filename = os.path.join(cache_dir, f"{pdb_id}.pdb")
    if not os.path.exists(filename):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        urllib.request.urlretrieve(url, filename)

    coords_ca: list[list[float]] = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                coords_ca.append([
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                ])
    return np.array(coords_ca)


def calculate_physics_metrics(coords: np.ndarray) -> tuple[float, float]:
    """Compute end-to-end distance and radius of gyration.

    Parameters
    ----------
    coords:
        Coordinate array, shape ``(N, 3)``.

    Returns
    -------
    end_to_end : float
        Euclidean distance between the first and last coordinate.
    radius_of_gyration : float
        Root-mean-square distance of all atoms from the centroid.
    """
    end_to_end = float(np.linalg.norm(coords[0] - coords[-1]))
    centroid = np.mean(coords, axis=0)
    rg = float(np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1))))
    return end_to_end, rg
