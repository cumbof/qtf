"""PDB I/O and biophysical utility functions."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import urllib.error
import urllib.request
from typing import List, Optional, Sequence, Union

import numpy as np


logger = logging.getLogger(__name__)


_AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


# ---------------------------------------------------------------------------
# Network configuration for PDB downloads
# ---------------------------------------------------------------------------
# B6 hardening: RCSB and the EBI mirror reject unauthenticated
# `urllib` requests that lack a `User-Agent` header (HTTP 403), and
# the default socket timeout is *unbounded*, so a rate-limited
# response can hang the call for several minutes. We set a sensible
# `User-Agent`, a finite `timeout`, and an exponential-backoff retry
# loop. The values below are tuned for the public RCSB
# `https://files.rcsb.org/download/<PDB>.pdb` endpoint, which serves
# the same files as the EBI mirror.

# Keep the version in sync with qtf.__version__.
_USER_AGENT = "QTF/0.3.12 (+https://github.com/cumbof/QTF)"

_PDB_DOWNLOAD_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"

_DEFAULT_DOWNLOAD_TIMEOUT = 15.0   # seconds
_MAX_DOWNLOAD_RETRIES = 3
_RETRY_BACKOFFS_SEC = (0.5, 1.0, 2.0)
_MIN_VALID_PDB_BYTES = 200         # any real PDB file is far larger; this catches truncation
# HTTP status codes that warrant a retry (server-side, transient).
_RETRY_HTTP_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


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


def _download_pdb(
    pdb_id: str,
    target_path: str,
    *,
    timeout: float = _DEFAULT_DOWNLOAD_TIMEOUT,
    max_retries: int = _MAX_DOWNLOAD_RETRIES,
) -> None:
    """Download ``pdb_id`` from RCSB to ``target_path`` with retries.

    B6 hardening:
      * Sets a ``User-Agent`` header (RCSB returns 403 without one).
      * Uses a finite ``timeout`` so a stuck connection does not hang
        the caller for minutes.
      * Forces ``https://``; refuses any other scheme.
      * Retries on transient network errors and on the HTTP status
        codes listed in ``_RETRY_HTTP_STATUSES`` (408, 425, 429,
        5xx). Each retry waits ``_RETRY_BACKOFFS_SEC[i]`` seconds
        before the ``i``-th attempt.
      * Validates the response size: anything smaller than
        ``_MIN_VALID_PDB_BYTES`` bytes is treated as a truncation and
        re-raised (the caller decides whether to retry).
      * Writes the file atomically via a sibling temp file +
        ``os.replace`` so a process crash mid-write cannot leave a
        half-written PDB on disk.
    """
    url = _PDB_DOWNLOAD_URL_TEMPLATE.format(pdb_id=pdb_id)
    if not url.startswith("https://"):
        # Defence in depth: the template already pins https, but a
        # future contributor that changes the template should not be
        # able to silently downgrade to http.
        raise ValueError(
            f"PDB downloads must use HTTPS; got url={url!r}"
        )

    headers = {"User-Agent": _USER_AGENT}
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers=dict(headers))
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            if len(raw) < _MIN_VALID_PDB_BYTES:
                raise RuntimeError(
                    f"PDB {pdb_id} download from {url} truncated: "
                    f"got {len(raw)} bytes (expected at least "
                    f"{_MIN_VALID_PDB_BYTES})"
                )
            # Atomic write: tmp file in the same directory + os.replace.
            target_dir = os.path.dirname(os.path.abspath(target_path)) or "."
            fd, tmp_path = tempfile.mkstemp(
                prefix=f".{pdb_id}_", suffix=".pdb", dir=target_dir
            )
            try:
                with os.fdopen(fd, "wb") as fh:
                    fh.write(raw)
                os.replace(tmp_path, target_path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            return
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in _RETRY_HTTP_STATUSES or attempt == max_retries - 1:
                raise RuntimeError(
                    f"PDB {pdb_id} download failed: HTTP {exc.code} {exc.reason}"
                ) from exc
            wait = _RETRY_BACKOFFS_SEC[min(attempt, len(_RETRY_BACKOFFS_SEC) - 1)]
            logger.warning(
                "PDB %s download attempt %d/%d failed with HTTP %d; "
                "retrying in %.1fs",
                pdb_id, attempt + 1, max_retries, exc.code, wait,
            )
            time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"PDB {pdb_id} download failed after {max_retries} "
                    f"attempts: {exc}"
                ) from exc
            wait = _RETRY_BACKOFFS_SEC[min(attempt, len(_RETRY_BACKOFFS_SEC) - 1)]
            logger.warning(
                "PDB %s download attempt %d/%d failed: %s; retrying in %.1fs",
                pdb_id, attempt + 1, max_retries, exc, wait,
            )
            time.sleep(wait)
    # Defensive fallback (the for-loop always either returns or raises).
    raise RuntimeError(
        f"PDB {pdb_id} download failed after {max_retries} attempts: {last_error}"
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

    Notes
    -----
    Network behaviour (B6):
      * HTTP requests are issued over HTTPS only; plain HTTP is refused.
      * A custom ``User-Agent`` header is sent (RCSB returns 403
        without one).
      * A finite per-attempt ``timeout`` is used; the download
        retries up to 3 times on transient errors (HTTP 408/425/429/
        5xx and network URLErrors) with exponential backoff
        (0.5 s, 1 s, 2 s).
      * The response is validated for size; anything smaller than a
        few hundred bytes is treated as a truncation and the file is
        not cached.
      * The file is written atomically (via a sibling temp file +
        ``os.replace``) so a process crash mid-write cannot leave a
        half-written cache file.
    """
    pdb_id = pdb_id.upper()
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.join(cache_dir, f"{pdb_id}.pdb")
    if not os.path.exists(filename):
        _download_pdb(pdb_id, filename)

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


def calculate_physics_metrics(coords: np.ndarray) -> dict[str, float]:
    """Compute end-to-end distance, radius of gyration, and the RMS bond length.

    Parameters
    ----------
    coords:
        Coordinate array, shape ``(N, 3)``.

    Returns
    -------
    dict[str, float]
        A dictionary with the following keys:

        ``"end_to_end"``
            Euclidean distance between the first and last coordinate.
        ``"radius_of_gyration"``
            Root-mean-square distance of all atoms from the centroid,
            ``sqrt(mean_i |r_i - centroid|^2)``. This is the standard
            textbook definition of the radius of gyration.
        ``"root_mean_square_bond_length"``
            ``sqrt(mean_i |r_{i+1} - r_i|^2)`` over the (N-1)
            consecutive bond vectors. ``0.0`` when ``N < 2``.
            Retained for backward compatibility with an older version
            of this function that incorrectly labelled the same
            expression as the radius of gyration.
    """
    end_to_end = float(np.linalg.norm(coords[0] - coords[-1]))
    centroid = np.mean(coords, axis=0)
    rg = float(np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1))))
    if coords.shape[0] >= 2:
        rms_bond_length = float(
            np.sqrt(
                np.mean(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
            )
        )
    else:
        rms_bond_length = 0.0
    return {
        "end_to_end": end_to_end,
        "radius_of_gyration": rg,
        "root_mean_square_bond_length": rms_bond_length,
    }
