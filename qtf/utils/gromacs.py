import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from qtf.utils.paths import relativize_absolute_paths


def find_gmx() -> Optional[str]:
    """Find a GROMACS executable, preferring the active Python environment."""
    env_bin = Path(sys.executable).resolve().parent
    for candidate in (env_bin / "gmx", env_bin / "gmx_mpi"):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which("gmx") or shutil.which("gmx_mpi")


def parse_pdb_atoms(pdb_path: str) -> Tuple[np.ndarray, List[Tuple[int, str, str]]]:
    """Parse ATOM/HETATM records into QTF-style coords and labels."""
    coords = []
    labels = []
    residue_index: Dict[Tuple[str, int, str], int] = {}
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            chain_id = line[21].strip()
            try:
                resseq = int(line[22:26])
            except ValueError:
                resseq = len(residue_index) + 1
            icode = line[26].strip()
            key = (chain_id, resseq, icode)
            if key not in residue_index:
                residue_index[key] = len(residue_index)
            elem = line[76:78].strip()
            if not elem:
                elem = "".join(c for c in atom_name if c.isalpha())[:1] or "X"
            coords.append([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ])
            labels.append((residue_index[key], atom_name, elem.upper()))
    return np.asarray(coords, dtype=float), labels


def prepare_pdb_for_gromacs(src_pdb: str, dst_pdb: Path) -> None:
    """
    Rewrite ATOM records so each residue is contiguous before pdb2gmx sees it.

    QTF rebuilds some atoms, especially carbonyl O, after the next residue's N.
    Viewers tolerate that, but pdb2gmx treats non-contiguous repeated residue IDs
    as separate residue blocks. This pass preserves coordinates but groups and
    serializes atoms by residue.

    It also normalizes the legacy PRO backbone alias NV to the conventional N,
    allowing pdb2gmx to map the residue to its standard template.
    """
    atom_order = {
        "N": 0,
        "H": 1,
        "H1": 2,
        "H2": 3,
        "H3": 4,
        "CA": 5,
        "HA": 6,
        "C": 90,
        "O": 91,
        "OXT": 92,
    }

    def rewrite_atom_name(line: str, atom_name: str) -> str:
        name_field = f"{atom_name:>4s}"[:4]
        return f"{line[:12]}{name_field}{line[16:]}"

    records = []
    remarks = []
    with open(src_pdb) as handle:
        for original_index, line in enumerate(handle):
            if line.startswith("REMARK"):
                remarks.append(line)
                continue
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21].strip() or "A"
            try:
                resseq = int(line[22:26])
            except ValueError:
                resseq = original_index
            icode = line[26].strip()

            # Some legacy PDB producers emit PRO backbone nitrogen as NV.
            # pdb2gmx expects the conventional backbone name N.
            if res_name == "PRO" and atom_name == "NV":
                atom_name = "N"
                line = rewrite_atom_name(line, atom_name)

            order = atom_order.get(atom_name, 10 + original_index)
            records.append((chain_id, resseq, icode, order, original_index, line))

    records.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    with open(dst_pdb, "w") as out:
        for remark in remarks:
            out.write(remark)
        last_chain = None
        serial = 1
        for chain_id, _resseq, _icode, _order, _orig, line in records:
            if last_chain is not None and chain_id != last_chain:
                out.write("TER\n")
            out.write(f"{line[:6]}{serial:5d}{line[11:]}")
            serial += 1
            last_chain = chain_id
        out.write("TER\nEND\n")


def ca_coords(coords: np.ndarray, labels: List[Tuple[int, str, str]]) -> np.ndarray:
    return np.asarray([coords[i] for i, lbl in enumerate(labels) if lbl[1] == "CA"], dtype=float)


def ring_penetration_metrics(
    coords: np.ndarray,
    labels: List[Tuple[int, str, str]],
    center_segment_threshold_A: float = 0.35,
) -> Dict[str, object]:
    """
    Detect aromatic ring centers that pierce a covalent bond segment elsewhere.

    Endpoint atom distances can look acceptable while the center of a planar ring
    sits directly on a bond segment. That is the TYR1 CA-CB / TYR10 ring failure
    mode this catches.
    """
    coords = np.asarray(coords, dtype=float)
    by_res: Dict[int, Dict[str, int]] = {}
    for i, (res_id, atom_name, _elem) in enumerate(labels):
        by_res.setdefault(int(res_id), {})[str(atom_name)] = i

    aromatic_rings = {
        "Y": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
        "F": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
        "H": [["CG", "ND1", "CD2", "CE1", "NE2"]],
        "W": [
            ["CG", "CD1", "NE1", "CE2", "CD2"],
            ["CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"],
        ],
    }
    standard_bonds = [
        ("N", "CA"),
        ("CA", "C"),
        ("C", "O"),
        ("CA", "CB"),
        ("CB", "CG"),
        ("CG", "CD"),
        ("CG", "CD1"),
        ("CG", "CD2"),
        ("CD", "CE"),
        ("CD1", "CE1"),
        ("CD2", "CE2"),
        ("CE1", "CZ"),
        ("CE2", "CZ"),
    ]

    # Infer one-letter residue names from available ring atom sets.
    rings = []
    for res_id, atoms in by_res.items():
        for aa, ring_defs in aromatic_rings.items():
            if aa == "Y" and "OH" not in atoms:
                continue
            if aa == "F" and "OH" in atoms:
                continue
            for ring_atoms in ring_defs:
                if all(a in atoms for a in ring_atoms):
                    center = np.mean([coords[atoms[a]] for a in ring_atoms], axis=0)
                    rings.append((res_id, aa, ring_atoms, center))

    bonds = []
    for res_id, atoms in by_res.items():
        for a1, a2 in standard_bonds:
            if a1 in atoms and a2 in atoms:
                bonds.append((res_id, a1, a2, atoms[a1], atoms[a2]))
    for res_id in sorted(by_res):
        left = by_res.get(res_id, {})
        right = by_res.get(res_id + 1, {})
        if "C" in left and "N" in right:
            bonds.append((res_id, "C", f"{res_id + 1}:N", left["C"], right["N"]))

    best_dist = float("inf")
    best_pair = ""
    best_t = float("nan")
    for ring_res, aa, _ring_atoms, center in rings:
        for bond_res, a1, a2, i, j in bonds:
            if bond_res == ring_res:
                continue
            p = coords[i]
            q = coords[j]
            pq = q - p
            denom = float(np.dot(pq, pq))
            if denom < 1e-9:
                continue
            t = float(np.dot(center - p, pq) / denom)
            if t <= 0.05 or t >= 0.95:
                continue
            proj = p + t * pq
            d = float(np.linalg.norm(center - proj))
            if d < best_dist:
                best_dist = d
                best_t = t
                best_pair = f"ring {ring_res}:{aa}-bond {bond_res}:{a1}-{a2}"

    if best_dist == float("inf"):
        return {
            "ring_penetration_min_dist_A": np.nan,
            "ring_penetration_flag": False,
            "ring_penetration_pair": "",
            "ring_penetration_segment_t": np.nan,
        }
    return {
        "ring_penetration_min_dist_A": best_dist,
        "ring_penetration_flag": bool(best_dist < float(center_segment_threshold_A)),
        "ring_penetration_pair": best_pair,
        "ring_penetration_segment_t": best_t,
    }


def write_minimization_mdp(path: Path, nsteps: int, emtol: float) -> None:
    path.write_text(
        "\n".join([
            "integrator      = steep",
            f"nsteps          = {int(nsteps)}",
            f"emtol           = {float(emtol)}",
            "emstep          = 0.01",
            "cutoff-scheme   = Verlet",
            "nstlist         = 1",
            "coulombtype     = Cut-off",
            "rcoulomb        = 1.0",
            "vdwtype         = Cut-off",
            "rvdw            = 1.0",
            "pbc             = xyz",
            "constraints     = none",
            "",
        ])
    )


def _run(cmd: List[str], cwd: Path, log_path: Path, input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["GMX_MAXBACKUP"] = "-1"
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    with open(log_path, "a") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.write(proc.stdout or "")
        log.write("\n")
    return proc


def parse_last_xvg_value(path: Path) -> float:
    last = None
    with open(path) as handle:
        for line in handle:
            if line.startswith(("#", "@")) or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                last = float(parts[1])
    return float(last) if last is not None else float("nan")


def parse_gromacs_log_stats(log_path: Path) -> Dict[str, object]:
    txt = log_path.read_text(errors="ignore") if log_path.exists() else ""
    vals = [
        float(x)
        for x in re.findall(
            r"Maximum\s+force\s*=\s*([0-9.eE+-]+)",
            txt,
            flags=re.IGNORECASE,
        )
    ]
    if vals:
        final_max_force = vals[-1]
    else:
        convergence = re.search(
            r"converged\s+to\s+Fmax\s*<\s*([0-9.eE+-]+)",
            txt,
            flags=re.IGNORECASE,
        )
        final_max_force = float(convergence.group(1)) if convergence else float("nan")
    return {
        "gromacs_converged_fmax_lt_100": bool("converged to Fmax < 100" in txt or "converged to Fmax < 100.0" in txt),
        "gromacs_final_max_force": float(final_max_force),
    }


def compact_successful_minimization_dir(workdir: Path, keep_paths: List[Path]) -> None:
    """Keep only compact audit artifacts after a successful minimization."""
    keep = {path.resolve() for path in keep_paths if path.exists()}
    for path in workdir.iterdir():
        if path.resolve() in keep:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def minimize_pdb_with_gromacs(
    pdb_path: str,
    outdir: str,
    *,
    forcefield: str = "amber99sb-ildn",
    water: str = "tip3p",
    nsteps: int = 5000,
    emtol: float = 100.0,
    maxwarn: int = 2,
) -> Dict[str, object]:
    """
    Add hydrogens/topology with pdb2gmx and run a short steepest-descent minimization.

    Returns a stable status dictionary. On success, ``minimized_pdb_path`` points
    to the PDB converted from the minimized GROMACS coordinates.
    """
    gmx = find_gmx()
    workdir = Path(outdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "gromacs_minimize.log"
    log_path.write_text("")

    result: Dict[str, object] = {
        "gromacs_status": "not_run",
        "gromacs_message": "",
        "gromacs_workdir": str(workdir),
        "gromacs_log_path": str(log_path),
        "gromacs_minimized_full_pdb_path": "",
        "gromacs_potential_kj_mol": np.nan,
        "gromacs_potential_kcal_mol": np.nan,
        "gromacs_converged_fmax_lt_100": False,
        "gromacs_converged": False,
        "gromacs_final_max_force": np.nan,
    }

    def finalize() -> Dict[str, object]:
        if log_path.is_file():
            portable_log = relativize_absolute_paths(
                log_path.read_text(encoding="utf-8", errors="replace")
            )
            log_path.write_text(portable_log, encoding="utf-8")
        return result

    if not gmx:
        result["gromacs_status"] = "missing_gmx"
        result["gromacs_message"] = "GROMACS executable not found"
        return finalize()

    input_pdb = Path(pdb_path).resolve()
    prepared_pdb = workdir / "prepared_input.pdb"
    processed_gro = workdir / "processed.gro"
    topol = workdir / "topol.top"
    posre = workdir / "posre.itp"
    boxed_gro = workdir / "boxed.gro"
    mdp = workdir / "minim.mdp"
    minimized_pdb = workdir / "minimized.pdb"

    prepare_pdb_for_gromacs(str(input_pdb), prepared_pdb)

    steps = [
        [
            gmx, "pdb2gmx",
            "-f", str(prepared_pdb),
            "-o", str(processed_gro),
            "-p", str(topol),
            "-i", str(posre),
            "-ff", forcefield,
            "-water", water,
            "-ignh",
        ],
        [
            gmx, "editconf",
            "-f", str(processed_gro),
            "-o", str(boxed_gro),
            "-c",
            "-d", "1.0",
            "-bt", "cubic",
        ],
    ]

    for cmd in steps:
        proc = _run(cmd, workdir, log_path)
        if proc.returncode != 0:
            result["gromacs_status"] = "failed"
            result["gromacs_message"] = f"command failed: {' '.join(cmd[:2])}"
            return finalize()

    write_minimization_mdp(mdp, nsteps=nsteps, emtol=emtol)
    grompp = [
        gmx, "grompp",
        "-f", str(mdp),
        "-c", str(boxed_gro),
        "-p", str(topol),
        "-o", str(workdir / "em.tpr"),
        "-maxwarn", str(int(maxwarn)),
    ]
    proc = _run(grompp, workdir, log_path)
    if proc.returncode != 0:
        result["gromacs_status"] = "failed"
        result["gromacs_message"] = "command failed: gmx grompp"
        return finalize()

    proc = _run([gmx, "mdrun", "-deffnm", "em", "-nt", "1"], workdir, log_path)
    if proc.returncode != 0:
        result["gromacs_status"] = "failed"
        result["gromacs_message"] = "command failed: gmx mdrun"
        return finalize()

    potential_xvg = workdir / "potential.xvg"
    proc = _run(
        [gmx, "energy", "-f", str(workdir / "em.edr"), "-o", str(potential_xvg)],
        workdir,
        log_path,
        input_text="Potential\n0\n",
    )
    if proc.returncode == 0 and potential_xvg.exists():
        potential_kj = parse_last_xvg_value(potential_xvg)
        result["gromacs_potential_kj_mol"] = float(potential_kj)
        result["gromacs_potential_kcal_mol"] = float(potential_kj / 4.184)

    proc = _run([gmx, "editconf", "-f", str(workdir / "em.gro"), "-o", str(minimized_pdb)], workdir, log_path)
    if proc.returncode != 0:
        result["gromacs_status"] = "failed"
        result["gromacs_message"] = "command failed: gmx editconf minimized pdb"
        return finalize()

    command_stats = parse_gromacs_log_stats(log_path)
    mdrun_stats = parse_gromacs_log_stats(workdir / "em.log")
    final_max_force = (
        mdrun_stats["gromacs_final_max_force"]
        if np.isfinite(mdrun_stats["gromacs_final_max_force"])
        else command_stats["gromacs_final_max_force"]
    )
    converged = bool(np.isfinite(final_max_force) and final_max_force <= float(emtol))
    result.update(
        {
            "gromacs_converged_fmax_lt_100": bool(
                np.isfinite(final_max_force) and final_max_force < 100.0
            ),
            "gromacs_converged": converged,
            "gromacs_final_max_force": final_max_force,
        }
    )
    potential_kj = float(result["gromacs_potential_kj_mol"])
    if not np.isfinite(potential_kj) or abs(potential_kj) > 1.0e9:
        result["gromacs_status"] = "failed"
        result["gromacs_message"] = "minimization produced a non-finite or physically invalid potential energy"
        return finalize()
    if not np.isfinite(final_max_force):
        result["gromacs_status"] = "failed"
        result["gromacs_message"] = "minimization produced a non-finite maximum force"
        return finalize()
    if not converged:
        result["gromacs_status"] = "failed"
        result["gromacs_message"] = (
            f"minimization did not converge to the requested Fmax <= {float(emtol):g}; "
            f"final Fmax was {float(final_max_force):g}"
        )
        return finalize()

    result["gromacs_status"] = "ok"
    result["gromacs_message"] = ""
    result["gromacs_minimized_full_pdb_path"] = str(minimized_pdb)
    compact_successful_minimization_dir(workdir, keep_paths=[minimized_pdb, log_path])
    return finalize()


def refine_pdb_with_gromacs(
    pdb_path: str,
    refined_pdb_path: str | Path,
    *,
    log_path: str | Path | None = None,
    forcefield: str = "amber99sb-ildn",
    water: str = "tip3p",
    nsteps: int = 5000,
    emtol: float = 100.0,
    maxwarn: int = 2,
) -> Dict[str, object]:
    """Run minimization in temporary storage and retain only its PDB and log.

    This is the shared artifact-level interface for simulation and hardware
    folds. ``minimize_pdb_with_gromacs`` remains the lower-level work-directory
    implementation used by callers that need the complete working location.
    """

    refined_pdb_path = Path(refined_pdb_path).resolve()
    retained_log_path = (
        Path(log_path).resolve()
        if log_path is not None
        else refined_pdb_path.with_suffix(".log")
    )
    refined_pdb_path.parent.mkdir(parents=True, exist_ok=True)
    retained_log_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="qtf-gromacs-") as workdir:
        result = minimize_pdb_with_gromacs(
            pdb_path,
            workdir,
            forcefield=forcefield,
            water=water,
            nsteps=nsteps,
            emtol=emtol,
            maxwarn=maxwarn,
        )
        generated_log = result.get("gromacs_log_path")
        if generated_log and Path(str(generated_log)).is_file():
            shutil.copy2(str(generated_log), retained_log_path)
            result["gromacs_log_path"] = str(retained_log_path)

        generated_pdb = result.get("gromacs_minimized_full_pdb_path")
        if result.get("gromacs_status") == "ok" and generated_pdb and Path(str(generated_pdb)).is_file():
            shutil.copy2(str(generated_pdb), refined_pdb_path)
            result["gromacs_minimized_full_pdb_path"] = str(refined_pdb_path)

        result["gromacs_workdir"] = None
        return result
