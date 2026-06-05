"""PDB I/O and biophysical utility functions."""

from __future__ import annotations

import os
import urllib.request
from typing import List, Optional, Sequence, Union

import numpy as np


_AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def _resolve_resname(
    res_id: int,
    resnames: Union[List[str], dict, None],
    sequence: Optional[str],
) -> str:
    """Pick a 3-letter residue name for a given ``res_id``.

    Precedence:
      1. ``resnames[res_id]`` (dict lookup, then list lookup).
      2. ``_AA1_TO_3[sequence[res_id]]`` if a ``sequence`` is given.
      3. ``"UNK"`` as the final fallback.
    """
    if resnames is not None:
        if isinstance(resnames, dict):
            if res_id in resnames:
                return str(resnames[res_id])
        else:
            try:
                return str(resnames[res_id])
            except (IndexError, TypeError):
                pass
    if sequence is not None and 0 <= res_id < len(sequence):
        return _AA1_TO_3.get(sequence[res_id].upper(), "UNK")
    return "UNK"


def _resolve_resseq(
    res_id: int,
    resseqs: Union[List[int], dict, None],
) -> int:
    """Pick a residue sequence number (PDB column 23-26) for a given
    ``res_id``. Falls back to ``res_id + 1`` so the output is at least
    stable and 1-indexed."""
    if resseqs is None:
        return int(res_id) + 1
    if isinstance(resseqs, dict):
        return int(resseqs.get(res_id, int(res_id) + 1))
    try:
        return int(resseqs[res_id])
    except (IndexError, TypeError):
        return int(res_id) + 1


def _format_atom_line(
    serial: int,
    atom_name: str,
    res_name: str,
    chain_id: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    """Build a single ATOM record that follows the canonical PDB
    column layout (cols 1-30, 31-54, 77-78)."""
    chain = (chain_id or "A")[:1]
    return (
        f"ATOM  {serial:5d} {atom_name:>4} {res_name:>3} {chain:1}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {str(element):>2}\n"
    )


def save_pdb(
    coords: np.ndarray,
    labels: Sequence,
    filename: str = "structure.pdb",
    energy: float = 0.0,
    chain_id: str = "A",
    resseqs: Union[List[int], dict, None] = None,
    resnames: Union[List[str], dict, None] = None,
    remarks: Optional[List[str]] = None,
    include_hydrogens: bool = True,
    sequence: Optional[str] = None,
) -> None:
    """Write a PDB file from predicted coordinates.

    This is the single canonical implementation of ``save_pdb`` for the
    whole QTF project (B5: the previous instance method on
    ``QuantumBiophysicsFolder`` and the module-level function in this
    file were two different signatures doing essentially the same
    thing; they have been unified here).

    Parameters
    ----------
    coords:
        Atom coordinate array, shape ``(N_atoms, 3)``.
    labels:
        Sequence of ``(res_id, atom_name, element)`` tuples matching
        ``coords``.
    filename:
        Output file path. Parent directories are created if missing.
    energy:
        Final energy value stored in a ``REMARK`` record. ``None``
        suppresses the energy remark.
    chain_id:
        Single-character chain identifier written in PDB column 22.
    resseqs:
        Optional mapping ``res_id -> resseq`` (int). May be a dict or
        a list-indexable. ``None`` falls back to ``res_id + 1``.
    resnames:
        Optional mapping ``res_id -> 3-letter residue name``. May be a
        dict or a list-indexable. ``None`` falls back to deriving the
        name from ``sequence`` (``_AA1_TO_3[sequence[res_id]]``) and
        finally to ``"UNK"``.
    remarks:
        Optional list of additional ``REMARK`` strings. The first
        remark slot (REMARK   1) is reserved for the energy; the rest
        of ``remarks`` are written to slots 2+.
    include_hydrogens:
        If ``False``, atoms whose element or atom name starts with
        ``"H"`` are skipped.
    sequence:
        Optional 1-letter amino acid sequence. Used to derive 3-letter
        residue names when ``resnames`` is not supplied.
    """
    coords = np.asarray(coords, dtype=float)
    outdir = os.path.dirname(filename)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(filename, "w") as f:
        if energy is not None:
            f.write(f"REMARK   1 ENERGY: {float(energy):.3f}\n")
        if remarks:
            for idx, remark in enumerate(remarks, start=2):
                f.write(f"REMARK {idx:3d} {remark}\n")

        serial = 1
        for pos, (res_id, atom_name, elem) in zip(coords, labels):
            if (not include_hydrogens) and (
                str(elem).upper() == "H" or str(atom_name).upper().startswith("H")
            ):
                continue
            res_id = int(res_id)
            res_name = _resolve_resname(res_id, resnames, sequence)
            resseq = _resolve_resseq(res_id, resseqs)
            f.write(
                _format_atom_line(
                    serial,
                    str(atom_name),
                    res_name,
                    chain_id,
                    resseq,
                    float(pos[0]),
                    float(pos[1]),
                    float(pos[2]),
                    str(elem),
                )
            )
            serial += 1
        f.write("END\n")


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
