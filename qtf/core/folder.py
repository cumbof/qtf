"""QuantumBiophysicsFolder — hybrid quantum-classical protein structure predictor.

Architecture
------------
1. **Quantum Actor**: a parameterised EfficientSU2 circuit whose statevector
   phases encode backbone/side-chain torsion angles.
2. **Classical Critic**: a physics-based energy function (hydrophobicity, H-bonds,
   electrostatics, sterics, Ramachandran bias, geometry integrity).
3. **Optimisation Loop**: COBYLA + SLSQP in three progressive stages (collapse →
   refine → relax).

References
----------
* Kyte & Doolittle (1982) hydrophobicity scale.
* AMBER ff14SB partial charges (approximate).
* Bondi (1964) van der Waals radii.
* Engh & Huber (1991) bond/angle parameters.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

from qtf.core.tracker import LandscapeTracker
from qtf.utils import gromacs as qtf_gromacs

try:
    import pyrosetta
    from pyrosetta import rosetta as _rosetta
    _PYROSETTA_AVAILABLE = True
except ImportError:
    pyrosetta = None  # type: ignore[assignment]
    _rosetta = None   # type: ignore[assignment]
    _PYROSETTA_AVAILABLE = False


def _load_openmm() -> tuple[tuple, bool]:
    """Best-effort import of :mod:`openmm` and the symbols QTF needs.

    Returns a 2-tuple ``((mm, unit, ForceField, HBonds, Modeller,
    NoCutoff, PDBFile), available)``. The symbols are all ``None`` and
    ``available`` is ``False`` only when :mod:`openmm` is genuinely
    not installed (``ModuleNotFoundError``).

    A non-``ModuleNotFoundError`` exception -- the typical signature
    of a *broken* openmm install (mismatched CUDA runtime, missing
    shared library, failed C++ extension load, etc.) -- is **logged
    and re-raised** so that the user is not misled into a
    ``pip install qtf[workflows]`` loop when the real fix is to
    reinstall the existing openmm package.
    """
    try:
        import openmm as mm
        from openmm import unit
        from openmm.app import (
            ForceField,
            HBonds,
            Modeller,
            NoCutoff,
            PDBFile,
        )
        return (
            (mm, unit, ForceField, HBonds, Modeller, NoCutoff, PDBFile),
            True,
        )
    except ModuleNotFoundError:
        return (None, None, None, None, None, None, None), False
    except Exception as exc:
        logger.error(
            "OpenMM import failed (install is broken, not missing): %s: %s",
            type(exc).__name__,
            exc,
        )
        raise


(
    (_mm, _unit, _ForceField, _HBonds, _Modeller, _NoCutoff, _PDBFile),
    _OPENMM_AVAILABLE,
) = _load_openmm()

_PYROSETTA_INIT_DONE = False

logger = logging.getLogger(__name__)

# Coulomb's law prefactor: 332.0637 kcal mol⁻¹ Å e⁻²
# (charges in elementary charges, distances in Ångströms, energy in kcal/mol)
_COULOMB_PREFACTOR: float = 332.0637
# Uniform implicit-solvent dielectric constant (ε ≈ 4 is the standard choice
# for buried/intermediate environments in coarse force fields; see e.g.
# Warshel & Russell, Q Rev Biophys 1984).
_DIELECTRIC: float = 4.0


# Seed angle (rad) used when calling build_full_structure inside
# _initialize_topology_cache.  All-zero torsions place every backbone atom
# along the same line (collinear), zeroing the cross products in _nerf_step
# and making the reference frame degenerate.  A small non-zero value avoids
# this without affecting the cache output, which discards coordinates and
# only retains atom labels.
_TOPOLOGY_SEED_ANGLE: float = 0.1

# Huber-loss transition threshold for geometry-integrity penalties (Å or Å³).
# Below this threshold each penalty is quadratic (x²), preserving a smooth
# gradient signal for small deviations.  Above it the loss grows only
# linearly (2·δ·|x| − δ²), preventing a single severely distorted bond or
# wrong-chirality centre from dominating the gradient and stalling the
# optimiser.  Value of 1.0 Å corresponds to roughly one bond length of
# distortion before saturation kicks in.
_HUBER_DELTA_GEOM: float = 1.0


class QuantumBiophysicsFolder:
    """Hybrid quantum-classical protein folder."""

    # ------------------------------------------------------------------
    # Defaults for stage-aware attributes.
    #
    # Declaring these at class level (not only in ``__init__``) means
    # that ``folder.energy_function(...)`` is safe to call on an
    # instance whose ``__init__`` has been bypassed -- e.g. by a
    # user-supplied gradient, a checkpoint restart that constructs
    # the folder from a pickled state, or a unit test that exercises
    # the energy function in isolation. Without the class-level
    # default, ``energy_function`` would raise ``AttributeError``
    # when reading ``self.current_stage``.
    # ------------------------------------------------------------------
    current_stage: int = 1  # 1 = helix/sheet-only cost; see fold() for 2, 3.

    # ------------------------------------------------------------------
    # Force-field tables
    # ------------------------------------------------------------------
    _HYDROPHOBICITY: dict[str, float] = {
        "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
        "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
        "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
        "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
    }

    _VDW_RADII: dict[str, float] = {
        "H": 0.6, "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8,
    }

    _LJ_TYPE_PARAMS: dict[str, dict] = {
        "H":           {"radius": 1.20, "epsilon": 0.0157},
        "H_polar":     {"radius": 1.05, "epsilon": 0.0157},
        "C_backbone":  {"radius": 1.75, "epsilon": 0.0700},
        "C_carbonyl":  {"radius": 1.70, "epsilon": 0.0860},
        "C_aliphatic": {"radius": 1.90, "epsilon": 0.1094},
        "C_aromatic":  {"radius": 1.85, "epsilon": 0.1200},
        "N_backbone":  {"radius": 1.65, "epsilon": 0.1700},
        "N_sidechain": {"radius": 1.65, "epsilon": 0.1700},
        "O_carbonyl":  {"radius": 1.60, "epsilon": 0.2100},
        "O_hydroxyl":  {"radius": 1.55, "epsilon": 0.1700},
        "O_carboxyl":  {"radius": 1.60, "epsilon": 0.2100},
        "S_sulfur":    {"radius": 2.00, "epsilon": 0.2500},
        "X":           {"radius": 1.75, "epsilon": 0.1000},
    }

    _SIDE_CHAIN_TOPO: dict[str, list] = {
        "G": [],
        "A": [("CB", "CA", 1.53, 1.91, 2.1)],
        "V": [("CB", "CA", 1.53, 1.91, "chi1"),
              ("CG1", "CB", 1.52, 1.91, "chi2"), ("CG2", "CB", 1.52, 1.91, "chi2_branch")],
        "L": [("CB", "CA", 1.53, 1.91, "chi1"),
              ("CG", "CB", 1.52, 1.91, "chi2"),
              ("CD1", "CG", 1.52, 1.91, "chi3"), ("CD2", "CG", 1.52, 1.91, "chi3_branch")],
        "I": [("CB", "CA", 1.53, 1.91, "chi1"),
              ("CG1", "CB", 1.54, 1.91, "chi2"), ("CD1", "CG1", 1.52, 1.91, "chi3"),
              ("CG2", "CB", 1.54, 1.91, "chi2_branch")],
        "M": [("CB", "CA", 1.53, 1.91, "chi1"), ("CG", "CB", 1.52, 1.91, "chi2"),
              ("SD", "CG", 1.81, 1.91, "chi3"), ("CE", "SD", 1.79, 1.76, "chi4")],
        "P": [("CB", "CA", 1.53, 1.80, "chi1"), ("CG", "CB", 1.50, 1.82, "chi2"),
              ("CD", "CG", 1.52, 1.83, "chi3")],
        "F": [("CB", "CA", 1.53, 1.91, "chi1"), ("CG", "CB", 1.50, 1.91, "chi2"),
              ("CD1", "CG", 1.39, 2.09, 1.57), ("CD2", "CG", 1.39, 2.09, -1.57),
              ("CE1", "CD1", 1.39, 2.09, 3.14), ("CE2", "CD2", 1.39, 2.09, 3.14),
              ("CZ", "CE1", 1.39, 2.09, 0.0)],
        "Y": [("CB", "CA", 1.53, 1.91, "chi1"), ("CG", "CB", 1.50, 1.91, "chi2"),
              ("CD1", "CG", 1.39, 2.09, 1.57), ("CD2", "CG", 1.39, 2.09, -1.57),
              ("CE1", "CD1", 1.39, 2.09, 3.14), ("CE2", "CD2", 1.39, 2.09, 3.14),
              ("CZ", "CE1", 1.39, 2.09, 0.0),
              ("OH", "CZ", 1.37, 2.09, 3.14), ("HH", "OH", 0.96, 1.83, "chi3")],
        "W": [("CB", "CA", 1.53, 1.91, "chi1"), ("CG", "CB", 1.50, 1.91, "chi2"),
              ("CD1", "CG", 1.37, 2.15, 1.0), ("CD2", "CG", 1.43, 2.15, -1.0),
              ("NE1", "CD1", 1.38, 1.90, 3.14), ("HE1", "NE1", 1.01, 2.09, 0.0),
              ("CE2", "CD2", 1.40, 1.90, 0.0), ("CE3", "CD2", 1.40, 2.30, 3.14),
              ("CZ2", "CE2", 1.40, 2.10, 0.0), ("CZ3", "CE3", 1.40, 2.10, 0.0),
              ("CH2", "CZ2", 1.40, 2.10, 0.0)],
        "S": [("CB", "CA", 1.53, 1.91, "chi1"), ("OG", "CB", 1.42, 1.91, "chi2"),
              ("HG", "OG", 0.96, 1.83, "chi3")],
        "T": [("CB", "CA", 1.53, 1.91, "chi1"),
              ("OG1", "CB", 1.43, 1.91, "chi2"), ("HG1", "OG1", 0.96, 1.83, "chi3"),
              ("CG2", "CB", 1.53, 1.91, "chi2_branch")],
        "C": [("CB", "CA", 1.53, 1.91, "chi1"), ("SG", "CB", 1.81, 1.91, "chi2")],
        "D": [("CB", "CA", 1.53, 1.91, "chi1"), ("CG", "CB", 1.52, 1.91, "chi2"),
              ("OD1", "CG", 1.25, 2.0, 1.0), ("OD2", "CG", 1.25, 2.0, -1.0)],
        "N": [("CB", "CA", 1.53, 1.91, "chi1"), ("CG", "CB", 1.52, 1.91, "chi2"),
              ("OD1", "CG", 1.23, 2.09, 0.0), ("ND2", "CG", 1.32, 2.09, 3.14)],
        "E": [("CB", "CA", 1.53, 1.91, "chi1"), ("CG", "CB", 1.52, 1.91, "chi2"),
              ("CD", "CG", 1.52, 1.91, "chi3"), ("OE1", "CD", 1.25, 2.0, 1.0), ("OE2", "CD", 1.25, 2.0, -1.0)],
        "Q": [("CB", "CA", 1.53, 1.91, "chi1"), ("CG", "CB", 1.52, 1.91, "chi2"),
              ("CD", "CG", 1.52, 1.91, "chi3"), ("OE1", "CD", 1.23, 2.09, 0.0), ("NE2", "CD", 1.32, 2.09, 3.14)],
        "K": [("CB", "CA", 1.53, 1.91, "chi1"), ("CG", "CB", 1.52, 1.91, "chi2"),
              ("CD", "CG", 1.52, 1.91, "chi3"), ("CE", "CD", 1.52, 1.91, "chi4"),
              ("NZ", "CE", 1.49, 1.91, "chi5")],
        "R": [("CB", "CA", 1.53, 1.91, "chi1"), ("CG", "CB", 1.52, 1.91, "chi2"),
              ("CD", "CG", 1.52, 1.91, "chi3"), ("NE", "CD", 1.46, 1.91, "chi4"),
              ("CZ", "NE", 1.33, 2.15, "chi5"), ("NH1", "CZ", 1.33, 2.10, 0.0), ("NH2", "CZ", 1.33, 2.10, 3.14)],
        "H": [("CB", "CA", 1.53, 1.91, "chi1"), ("CG", "CB", 1.50, 1.91, "chi2"),
              ("ND1", "CG", 1.38, 2.15, 1.0), ("CD2", "CG", 1.36, 2.15, -1.0),
              ("CE1", "ND1", 1.32, 1.90, 0.0),
              ("NE2", "CD2", 1.32, 1.90, 0.0), ("HE2", "NE2", 1.01, 2.09, 0.0)],
        "DEFAULT": [("CB", "CA", 1.53, 1.91, "chi1")],
    }

    def __init__(
        self,
        sequence: str,
        chi_mode: str = "all",
        selective_chi_map: dict | None = None,
        energy_backend: str | None = None,
        use_e2e_constraint: bool | None = None,
        e2e_scale: float | None = None,
        rosetta_repack: bool | None = None,
        rosetta_fa_min: bool | None = None,
        rosetta_cen_min: bool | None = None,
    ) -> None:
        """
        Parameters
        ----------
        sequence:
            Single-letter amino acid sequence (e.g. ``"MAGTWY"``).
        """
        self.sequence = sequence.upper()
        self.n_residues = len(self.sequence)

        logger.info("Initialising QuantumBiophysicsFolder | seq=%s", self.sequence)

        self.HYDROPHOBICITY = self._HYDROPHOBICITY
        self.VDW_RADII = self._VDW_RADII
        self.SIDE_CHAIN_TOPO = self._SIDE_CHAIN_TOPO

        self.CHARGES = self._build_charges()
        self.chi_mode = chi_mode
        self.selective_chi_map = selective_chi_map or {}
        self.LJ_TYPE_PARAMS = self._LJ_TYPE_PARAMS

        # Empirical peptide backbone geometry (Engh & Huber 1991).
        self.BB_ANGLE_N_CA_C = np.deg2rad(111.4)
        self.BB_ANGLE_CA_C_N = np.deg2rad(118.3)
        self.BB_ANGLE_C_N_CA = np.deg2rad(122.8)
        self.OMEGA_CENTER = np.pi
        self.OMEGA_MIN = np.deg2rad(170.0)
        self.OMEGA_MAX = np.deg2rad(190.0)
        self.OMEGA_HALF_WIDTH = 0.5 * (self.OMEGA_MAX - self.OMEGA_MIN)
        self.fixed_omegas = np.full(max(0, self.n_residues - 1), np.pi, dtype=float)

        # ------------------------------------------------------------------
        # Degrees of freedom
        # ------------------------------------------------------------------
        # Each residue contributes φ and ψ backbone dihedrals plus up to five
        # side-chain χ angles.  In addition, the peptide-bond torsion ω is
        # added as an explicit DOF for every residue **i** whose successor
        # residue **i+1** is proline ("P").  All other ω angles are fixed at
        # π (trans-amide) — deviations from planarity are < 5° in practice
        # and are not worth the extra quantum resource.
        #
        # NOTE — encoding vs optimiser gap
        # The quantum statevector has 2ⁿ complex amplitudes; extracting the
        # first ``total_angles`` phases is sufficient to carry ω_Pro angles
        # without any structural change to the circuit.  However, the current
        # COBYLA optimiser path treats all extracted phases symmetrically and
        # cannot constrain individual DOFs.  The ω_Pro angles therefore enter
        # the optimisation unconstrained in [−π, π].  If a narrow prior is
        # desired (e.g. cis-Pro at ~0 ± 20°), a penalty term should be added
        # to the energy function, or the optimiser should be replaced with one
        # that supports bounded variables.
        self.dof_map: list[dict] = []
        for i, aa in enumerate(self.sequence):
            self.dof_map.append({"res": i, "type": "phi"})
            self.dof_map.append({"res": i, "type": "psi"})
            topo = self.SIDE_CHAIN_TOPO.get(aa, self.SIDE_CHAIN_TOPO["DEFAULT"])
            chis: set[str] = set()
            for atom in topo:
                tor = atom[4]
                if isinstance(tor, str) and "chi" in tor:
                    chis.add(tor)
            for k in self._allowed_chis_for_residue(i, aa, chis):
                self.dof_map.append({"res": i, "type": k})
            # Pre-proline ω is free; add it last so existing φ/ψ/χ indices
            # are unaffected for non-proline-containing sequences.
            if i < self.n_residues - 1 and self.sequence[i + 1] == "P":
                self.dof_map.append({"res": i, "type": "omega"})

        self.total_angles = len(self.dof_map)

        # ------------------------------------------------------------------
        # Quantum circuit
        # ------------------------------------------------------------------
        self.n_qubits = max(2, int(np.ceil(np.log2(self.total_angles))))
        self.reps = int(np.ceil(self.total_angles / self.n_qubits)) + 2
        self.ansatz = efficient_su2(self.n_qubits, reps=self.reps, entanglement="circular")
        self.n_params = self.ansatz.num_parameters

        self.current_stage = 1
        self._cache_initialized = False
        self._initialize_topology_cache()

        # --- Optional stage-3 backends ---
        def _as_bool(value, default=False):
            if value is None:
                return bool(default)
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() not in ("0", "false", "no", "off", "none", "")

        self.stage3_backend = (
            energy_backend or os.getenv("QTF_STAGE3_BACKEND", "custom")
        ).strip().lower()
        if self.stage3_backend not in ("custom", "rosetta", "openmm"):
            raise ValueError("energy_backend must be 'custom', 'rosetta', or 'openmm'")

        self.use_e2e_constraint = _as_bool(
            use_e2e_constraint,
            os.getenv("QTF_USE_E2E_CONSTRAINT", "1").strip().lower()
            not in ("0", "false", "no", "off"),
        )
        self.e2e_scale = float(
            e2e_scale if e2e_scale is not None else os.getenv("QTF_E2E_SCALE", "1.0")
        )
        self.rosetta_flags = os.getenv("QTF_PYROSETTA_FLAGS", "-mute all")
        self.rosetta_centroid_weights = os.getenv("QTF_ROSETTA_CEN_WTS", "cen_std")
        self.rosetta_fullatom_weights = os.getenv("QTF_ROSETTA_FA_WTS", "ref2015")
        self.rosetta_cen_weight = float(os.getenv("QTF_ROSETTA_CEN_WEIGHT", "0.35"))
        self.rosetta_fa_weight = float(os.getenv("QTF_ROSETTA_FA_WEIGHT", "1.0"))
        self.rosetta_do_centroid_min = _as_bool(rosetta_cen_min, False)
        self.rosetta_do_fullatom_min = _as_bool(rosetta_fa_min, False)
        self.rosetta_do_repack = _as_bool(rosetta_repack, False)
        self._rosetta_ready = False
        self._rosetta_scorefxn_cen = None
        self._rosetta_scorefxn_fa = None
        self._last_rosetta_pose = None
        self._last_rosetta_ca = None
        self.openmm_forcefield = os.getenv("QTF_OPENMM_FORCEFIELD", "amber14-all.xml")
        self.openmm_platform = os.getenv("QTF_OPENMM_PLATFORM", "CPU")
        self.openmm_do_minimize = _as_bool(os.getenv("QTF_OPENMM_MINIMIZE", "0"), False)
        self.openmm_max_iterations = int(os.getenv("QTF_OPENMM_MAX_ITERATIONS", "200"))
        self.openmm_tolerance = float(os.getenv("QTF_OPENMM_TOLERANCE", "10.0"))
        self.openmm_ph = float(os.getenv("QTF_OPENMM_PH", "7.0"))
        self._openmm_ready = False
        self._last_openmm_coords = None
        self._last_openmm_labels = None
        self.tracker: LandscapeTracker | None = None
        self.last_energy_terms: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_charges() -> dict[str, float]:
        common: dict[str, float] = {
            "OXT": -1.0,
            "NZ": 1.0, "NH1": 0.5, "NH2": 0.5,
            "OD1": -0.5, "OD2": -0.5, "OE1": -0.5, "OE2": -0.5,
            "ND2": 0.5, "NE2": 0.5,
            "SG": -0.1, "SD": -0.1,
            "HE2": 0.4, "ND1": -0.4,
        }
        amber: dict[str, float] = {
            "N": -0.42, "H": 0.27, "CA": 0.00, "C": 0.60, "O": -0.57,
            "OG": -0.6, "HG": 0.4, "OG1": -0.6, "HG1": 0.4, "OH": -0.5, "HH": 0.4,
            "NE1": -0.4, "HE1": 0.3,
        }
        charges = common.copy()
        charges.update(amber)
        return charges

    def _allowed_chis_for_residue(self, res_idx, aa, available_chis):
        """
        Decide which chi DOFs to expose for a residue.
        """

        available = sorted(set(available_chis), key=lambda x: (len(x), x))

        if self.chi_mode == "all":
            return available

        if self.chi_mode == "chi1_only":
            return [c for c in available if c == "chi1"]

        if self.chi_mode == "selective":
            allowed = self.selective_chi_map.get(aa, ["chi1"])
            allowed = set(allowed)
            return [c for c in available if c in allowed]

        raise ValueError(f"Unknown chi_mode: {self.chi_mode}")

    def _bounded_omega(self, omega):
        """Clamp omega to the allowed trans-peptide band."""
        val = float(omega)
        if abs(val) < 1e-12:
            return float(self.OMEGA_CENTER)
        # MDTraj/PDB torsions are often represented as signed angles near -180.
        # Convert those to the equivalent positive trans angle before enforcing
        # the [170, 190] degree band; e.g. -174 deg means 186 deg, not 174 deg.
        if -self.OMEGA_MAX <= val <= -self.OMEGA_MIN:
            val = (2.0 * np.pi) + val
        return float(np.clip(val, self.OMEGA_MIN, self.OMEGA_MAX))

    def _map_angle_vector_to_physical_ranges(self, angle_vector):
        """
        Map unconstrained circuit phases into physical torsion ranges.

        Phi/psi/chi remain regular signed torsions. Omega is restricted to the
        trans band [170, 190] degrees, so quantum sampling cannot produce
        unphysical peptide twists.
        """
        mapped = np.asarray(angle_vector, dtype=float).copy()
        for j, dof in enumerate(self.dof_map[:len(mapped)]):
            if str(dof.get("type")) == "omega":
                raw = float(np.clip(mapped[j], -np.pi, np.pi))
                mapped[j] = self.OMEGA_CENTER + (raw / np.pi) * self.OMEGA_HALF_WIDTH
                mapped[j] = self._bounded_omega(mapped[j])
        return mapped

    def _angle_dict_from_vector(self, angle_vector):
        angle_dict = {f"{x['res']}_{x['type']}": val for x, val in zip(self.dof_map, angle_vector)}
        for key, val in list(angle_dict.items()):
            if key.endswith("_omega"):
                angle_dict[key] = self._bounded_omega(val)
        return angle_dict

    def _infer_element_from_atom_name(self, atom_name: str) -> str:
        """Infer a PDB element symbol from a compact protein atom name."""
        name = str(atom_name).strip()
        if not name:
            return "X"
        # The topology currently only uses C/N/O/S/H-style atom names.
        first = name[0].upper()
        if first in {"C", "N", "O", "S", "H"}:
            return first
        return "X"

    def build_output_structure(self, angle_vector):
        """
        Build the structure that should be emitted to downstream consumers.

        In Rosetta stage-3 mode, this returns the actual PyRosetta pose scored
        for the supplied torsions so saved PDBs and RMSDs reflect the same
        structure Rosetta evaluated.
        """
        if self.stage3_backend == "rosetta":
            self._score_stage3_rosetta(angle_vector, return_terms=True)
            if self._last_rosetta_pose is not None and self.last_energy_terms.get("rosetta_error", 1.0) == 0.0:
                return self._pose_to_coords_labels_bonds(self._last_rosetta_pose)
        return self.build_full_structure(angle_vector)

    def _assign_lj_type(self, rid: int, atom_name: str, elem: str) -> str:
        """Assign a compact LJ atom type from residue, atom name, and element."""
        aa = self.sequence[int(rid)] if 0 <= int(rid) < self.n_residues else "X"
        name = str(atom_name)
        elem = str(elem)

        if elem == "H" or name.startswith("H"):
            if name in ("H", "HN", "HG", "HG1", "HH", "HE1", "HE2"):
                return "H_polar"
            return "H"

        if elem == "S" or name.startswith("S"):
            return "S_sulfur"

        if elem == "O":
            if name == "O" or name == "OXT":
                return "O_carbonyl"
            if name in ("OD1", "OD2", "OE1", "OE2"):
                return "O_carboxyl"
            return "O_hydroxyl"

        if elem == "N":
            if name == "N":
                return "N_backbone"
            return "N_sidechain"

        if elem == "C":
            if name == "C":
                return "C_carbonyl"
            if name == "CA":
                return "C_backbone"
            if aa in ("F", "Y", "W", "H") and name in {
                "CG", "CD1", "CD2", "CE1", "CE2", "CE3", "CZ", "CZ2", "CZ3", "CH2"
            }:
                return "C_aromatic"
            return "C_aliphatic"

        return "X"

    def _ensure_rosetta(self):
        global _PYROSETTA_INIT_DONE
        if self._rosetta_ready:
            return
        if not _PYROSETTA_AVAILABLE:
            raise RuntimeError("PyRosetta is not installed, but QTF_STAGE3_BACKEND=rosetta was requested.")
        if not _PYROSETTA_INIT_DONE:
            pyrosetta.init(self.rosetta_flags)
            _PYROSETTA_INIT_DONE = True
        self._rosetta_scorefxn_cen = pyrosetta.create_score_function(self.rosetta_centroid_weights)
        try:
            self._rosetta_scorefxn_fa = pyrosetta.create_score_function(self.rosetta_fullatom_weights)
        except Exception:
            self._rosetta_scorefxn_fa = pyrosetta.get_fa_scorefxn()
        self._rosetta_ready = True

    def _ensure_openmm(self):
        if self._openmm_ready:
            return
        if not _OPENMM_AVAILABLE:
            raise ImportError(
                "OpenMM is required for the 'openmm' stage-3 backend but it is "
                "not importable. Install it with `pip install \"qtf[workflows]\"` "
                "(or `conda install -c conda-forge openmm`); if OpenMM is already "
                "installed, the import failure means the install is broken (CUDA "
                "mismatch, missing shared library, failed C++ extension load, etc.) "
                "and the original error was logged at module-import time."
            )
        self._openmm_ready = True

    def _build_rosetta_pose_from_angles(self, angle_vec):
        self._ensure_rosetta()
        pose = pyrosetta.pose_from_sequence(self.sequence, "fa_standard")
        angle_dict = self._angle_dict_from_vector(angle_vec)
        for i in range(1, pose.total_residue()):
            pose.set_omega(i, float(np.rad2deg(angle_dict.get(f"{i-1}_omega", np.pi))))
        for dof, ang in zip(self.dof_map, angle_vec):
            resi = int(dof["res"]) + 1
            t = str(dof["type"])
            if t == "omega":
                continue
            deg = float(np.rad2deg(ang))
            try:
                if t == "phi":
                    pose.set_phi(resi, deg)
                elif t == "psi":
                    pose.set_psi(resi, deg)
                elif t.startswith("chi"):
                    chi_idx = int(t.replace("chi", ""))
                    if chi_idx <= pose.residue(resi).nchi():
                        pose.set_chi(chi_idx, resi, deg)
            except Exception:
                continue
        return pose

    def _build_openmm_input_pdb(self, coords, labels):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
        tmp.close()
        coords = np.asarray(coords, dtype=float)
        labels = list(labels)
        res_ids = [int(lbl[0]) for lbl in labels]
        if labels:
            last_res = max(res_ids)
            atom_names = {str(atom_name).upper() for rid, atom_name, _ in labels if int(rid) == last_res}
            if "OXT" not in atom_names:
                idx_c = idx_o = idx_ca = None
                for i, (rid, atom_name, elem) in enumerate(labels):
                    if int(rid) != last_res:
                        continue
                    an = str(atom_name).upper()
                    if an == "C":
                        idx_c = i
                    elif an == "O":
                        idx_o = i
                    elif an == "CA":
                        idx_ca = i
                if idx_c is not None and idx_o is not None and idx_ca is not None:
                    c = coords[idx_c]
                    o = coords[idx_o]
                    ca = coords[idx_ca]
                    u = o - c
                    nu = np.linalg.norm(u)
                    if nu > 1e-6:
                        u = u / nu
                        v = ca - c
                        nv = np.linalg.norm(v)
                        if nv > 1e-6:
                            v = v / nv
                            n = np.cross(u, v)
                            nn = np.linalg.norm(n)
                            if nn > 1e-6:
                                n = n / nn
                                perp = np.cross(n, u)
                                np_ = np.linalg.norm(perp)
                                if np_ > 1e-6:
                                    perp = perp / np_
                                else:
                                    perp = -u
                                direction = (-0.5 * u) + (0.8660254037844386 * perp)
                                nd = np.linalg.norm(direction)
                                if nd > 1e-6:
                                    direction = direction / nd
                                    oxt = c + 1.24 * direction
                                    coords = np.vstack([coords, oxt])
                                    labels.append((last_res, "OXT", "O"))
        self.save_pdb(
            coords,
            labels,
            filename=tmp.name,
            energy=0.0,
            include_hydrogens=False,
        )
        return tmp.name

    def _update_openmm_output_from_pdb(self, pdb_path):
        if not pdb_path:
            self._last_openmm_coords = None
            self._last_openmm_labels = None
            return
        try:
            coords, labels = qtf_gromacs.parse_pdb_atoms(pdb_path)
            self._last_openmm_coords = coords
            self._last_openmm_labels = labels
        except Exception:
            self._last_openmm_coords = None
            self._last_openmm_labels = None

    def _pose_ca_coords(self, pose):
        ca = []
        for i in range(1, pose.total_residue() + 1):
            rsd = pose.residue(i)
            if rsd.has("CA"):
                xyz = rsd.xyz("CA")
                ca.append([float(xyz.x), float(xyz.y), float(xyz.z)])
        return np.asarray(ca, dtype=float) if ca else np.zeros((0, 3), dtype=float)

    def _pose_to_coords_labels_bonds(self, pose):
        """
        Convert the actual PyRosetta Pose that was scored/refined into the
        runner.py coordinate/label/bond tuple used by downstream QTF code.

        This is intentionally used in Rosetta mode so returned PDB/RMSD
        coordinates match the object that Rosetta scored, repacked, and/or
        minimized. Labels use 0-indexed residue IDs to preserve the existing
        save_pdb() and centroid helpers. Bonds are left empty because QTF
        downstream code treats them as optional metadata.
        """
        coords = []
        labels = []
        for i in range(1, pose.total_residue() + 1):
            rsd = pose.residue(i)
            for j in range(1, rsd.natoms() + 1):
                atom_name = rsd.atom_name(j).strip()
                xyz = rsd.xyz(j)
                elem = "X"
                try:
                    elem = rsd.atom_type(j).element().strip() or atom_name[0]
                except Exception:
                    # Fallback for older PyRosetta builds.
                    elem = atom_name[0] if atom_name else "X"
                coords.append([float(xyz.x), float(xyz.y), float(xyz.z)])
                labels.append((i - 1, atom_name, elem))
        return np.asarray(coords, dtype=float), labels, []

    def _final_output_structure_from_params(self, params):
        """
        Return the final structure for fold(). In custom mode this is the
        original QTF NERF rebuild. In Rosetta mode this forcibly refreshes
        scoring at the final optimizer parameters and returns the actual
        PyRosetta full-atom pose used for scoring/refinement.
        """
        angle_vec = self._get_angles(params)
        if self.stage3_backend == "rosetta":
            # Force one final score call at res_3.x so _last_rosetta_pose is
            # synchronized with the optimizer's final parameter vector.
            self._score_stage3_rosetta(angle_vec, return_terms=True)
            if self._last_rosetta_pose is not None and self.last_energy_terms.get("rosetta_error", 1.0) == 0.0:
                return self._pose_to_coords_labels_bonds(self._last_rosetta_pose)
        if self.stage3_backend == "openmm":
            self._score_stage3_openmm(angle_vec, return_terms=True)
            if self._last_openmm_coords is not None and self.last_energy_terms.get("openmm_error", 1.0) == 0.0:
                return self._last_openmm_coords, self._last_openmm_labels, []
        return self.build_full_structure(angle_vec)

    def _extract_rosetta_terms(self, pose, scorefxn, prefix):
        """Return a stable set of Rosetta term columns, including zeros."""
        _ = scorefxn(pose)
        emap = pose.energies().total_energies()
        names = [
            "fa_atr", "fa_rep", "fa_sol", "lk_ball_wtd", "fa_elec",
            "hbond_sr_bb", "hbond_lr_bb", "hbond_bb_sc", "hbond_sc",
            "rama_prepro", "omega", "p_aa_pp", "fa_dun", "dslf_fa13", "ref",
            "env", "pair", "cbeta", "vdw", "rg", "rama",
        ]
        out = {}
        for name in names:
            val = 0.0
            if hasattr(_rosetta.core.scoring, name):
                st = getattr(_rosetta.core.scoring, name)
                try:
                    val = float(emap[st])
                except Exception:
                    val = 0.0
            out[f"{prefix}_{name}"] = val
        out[f"{prefix}_total"] = float(scorefxn(pose))
        return out

    def _score_stage3_rosetta(self, angle_vec, return_terms=False):
        terms = {"energy_backend_rosetta": 1.0, "energy_backend_custom": 0.0, "energy_backend_openmm": 0.0}
        try:
            self._ensure_rosetta()
            fa_pose = self._build_rosetta_pose_from_angles(angle_vec)

            cen_pose = fa_pose.clone()
            _rosetta.protocols.simple_moves.SwitchResidueTypeSetMover("centroid").apply(cen_pose)
            cen_score = float(self._rosetta_scorefxn_cen(cen_pose))
            terms.update(self._extract_rosetta_terms(cen_pose, self._rosetta_scorefxn_cen, "cen"))

            if self.rosetta_do_centroid_min:
                mm = _rosetta.core.kinematics.MoveMap()
                mm.set_bb(True)
                mm.set_chi(False)
                minmov = _rosetta.protocols.minimization_packing.MinMover()
                minmov.movemap(mm)
                minmov.score_function(self._rosetta_scorefxn_cen)
                minmov.min_type("lbfgs_armijo_nonmonotone")
                minmov.apply(cen_pose)
                cen_score = float(self._rosetta_scorefxn_cen(cen_pose))
                terms.update(self._extract_rosetta_terms(cen_pose, self._rosetta_scorefxn_cen, "cen"))

            if self.rosetta_do_repack:
                task = pyrosetta.standard_packer_task(fa_pose)
                task.restrict_to_repacking()
                task.or_include_current(True)
                packer = _rosetta.protocols.minimization_packing.PackRotamersMover(self._rosetta_scorefxn_fa, task)
                packer.apply(fa_pose)

            if self.rosetta_do_fullatom_min:
                mm = _rosetta.core.kinematics.MoveMap()
                mm.set_bb(False)
                mm.set_chi(True)
                minmov = _rosetta.protocols.minimization_packing.MinMover()
                minmov.movemap(mm)
                minmov.score_function(self._rosetta_scorefxn_fa)
                minmov.min_type("lbfgs_armijo_nonmonotone")
                minmov.apply(fa_pose)

            fa_score = float(self._rosetta_scorefxn_fa(fa_pose))
            terms.update(self._extract_rosetta_terms(fa_pose, self._rosetta_scorefxn_fa, "fa"))
            total = self.rosetta_cen_weight * cen_score + self.rosetta_fa_weight * fa_score
            terms["rosetta_cen_weight"] = float(self.rosetta_cen_weight)
            terms["rosetta_fa_weight"] = float(self.rosetta_fa_weight)
            terms["rosetta_total"] = float(total)
            terms["total"] = float(total)
            terms["rosetta_error"] = 0.0
            self._last_rosetta_pose = fa_pose.clone()
            self._last_rosetta_ca = self._pose_ca_coords(fa_pose)
        except Exception as exc:
            total = 1.0e6
            # Emit stable columns even on failure so CSV/analyzer columns do not
            # become all-empty or disappear in mixed runs.
            for p in ("cen", "fa"):
                for name in [
                    "fa_atr", "fa_rep", "fa_sol", "lk_ball_wtd", "fa_elec",
                    "hbond_sr_bb", "hbond_lr_bb", "hbond_bb_sc", "hbond_sc",
                    "rama_prepro", "omega", "p_aa_pp", "fa_dun", "dslf_fa13", "ref",
                    "env", "pair", "cbeta", "vdw", "rg", "rama", "total",
                ]:
                    terms[f"{p}_{name}"] = 0.0
            terms["rosetta_cen_weight"] = float(self.rosetta_cen_weight)
            terms["rosetta_fa_weight"] = float(self.rosetta_fa_weight)
            terms["rosetta_total"] = float(total)
            terms["total"] = float(total)
            terms["rosetta_error"] = 1.0
            terms["rosetta_message_hash"] = float(abs(hash(str(exc))) % 1000000)
        self.last_energy_terms = {k: float(v) for k, v in terms.items()}
        if self.tracker is not None:
            self.tracker.log(float(total))
        return float(total)

    def _score_stage3_openmm(self, angle_vec, return_terms=False):
        terms = {"energy_backend_openmm": 1.0, "energy_backend_custom": 0.0, "energy_backend_rosetta": 0.0}
        try:
            self._ensure_openmm()
            coords, labels, _ = self.build_full_structure(angle_vec)
            input_pdb = self._build_openmm_input_pdb(coords, labels)
            workdir = tempfile.mkdtemp(prefix="qtf_openmm_")
            prepared_pdb = os.path.join(workdir, "prepared_input.pdb")
            minimized_pdb = os.path.join(workdir, "minimized.pdb")
            result = {
                "openmm_status": "not_run",
                "openmm_message": "",
                "openmm_workdir": workdir,
                "openmm_prepared_pdb_path": prepared_pdb,
                "openmm_minimized_full_pdb_path": "",
                "openmm_potential_kj_mol": np.nan,
                "openmm_potential_kcal_mol": np.nan,
                "openmm_converged": False,
                "openmm_final_max_force": np.nan,
            }

            qtf_gromacs.prepare_pdb_for_gromacs(input_pdb, Path(prepared_pdb))
            pdb = _PDBFile(str(prepared_pdb))
            modeller = _Modeller(pdb.topology, pdb.positions)
            forcefield = _ForceField(self.openmm_forcefield)
            modeller.addHydrogens(forcefield, pH=float(self.openmm_ph))
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=_NoCutoff,
                constraints=_HBonds,
                rigidWater=False,
                removeCMMotion=False,
            )
            integrator = _mm.VerletIntegrator(1.0 * _unit.femtoseconds)
            platform_name = (self.openmm_platform or "CPU").strip() or "CPU"
            try:
                platform = _mm.Platform.getPlatformByName(platform_name)
            except Exception:
                platform = _mm.Platform.getPlatform(0)
            context = _mm.Context(system, integrator, platform)
            context.setPositions(modeller.positions)

            if self.openmm_do_minimize:
                _mm.LocalEnergyMinimizer.minimize(
                    context,
                    tolerance=float(self.openmm_tolerance) * _unit.kilojoule_per_mole / _unit.nanometer,
                    maxIterations=int(self.openmm_max_iterations),
                )

            state = context.getState(getEnergy=True, getPositions=True)
            potential_kj = float(state.getPotentialEnergy().value_in_unit(_unit.kilojoule_per_mole))
            result["openmm_status"] = "ok"
            result["openmm_potential_kj_mol"] = potential_kj
            result["openmm_potential_kcal_mol"] = float(potential_kj / 4.184)
            result["openmm_converged"] = bool(self.openmm_do_minimize)
            if self.openmm_do_minimize:
                with open(minimized_pdb, "w") as handle:
                    _PDBFile.writeFile(modeller.topology, state.getPositions(), handle)
                result["openmm_minimized_full_pdb_path"] = minimized_pdb
            else:
                result["openmm_minimized_full_pdb_path"] = ""

            terms["openmm_potential_kj_mol"] = potential_kj
            terms["openmm_potential_kcal_mol"] = float(potential_kj / 4.184)
            terms["total"] = float(potential_kj)
            terms["openmm_status_ok"] = 1.0
            terms["openmm_minimize"] = 1.0 if self.openmm_do_minimize else 0.0
            terms["openmm_error"] = 0.0
            terms["openmm_forcefield_hash"] = float(abs(hash(self.openmm_forcefield)) % 1000000)
            terms["openmm_platform_hash"] = float(abs(hash(platform_name)) % 1000000)
            terms["openmm_max_iterations"] = float(self.openmm_max_iterations)
            terms["openmm_tolerance"] = float(self.openmm_tolerance)
            terms["openmm_ph"] = float(self.openmm_ph)
            terms["openmm_status_hash"] = float(abs(hash(str(result.get("openmm_status", "")))) % 1000000)
            terms["openmm_message_hash"] = float(abs(hash(str(result.get("openmm_message", "")))) % 1000000)

            if result.get("openmm_minimized_full_pdb_path"):
                self._update_openmm_output_from_pdb(str(result["openmm_minimized_full_pdb_path"]))
            else:
                self._last_openmm_coords = coords
                self._last_openmm_labels = labels

            total = potential_kj
            try:
                os.unlink(input_pdb)
            except OSError:
                pass
        except Exception as exc:
            total = 1.0e6
            terms["openmm_potential_kj_mol"] = float(total)
            terms["openmm_potential_kcal_mol"] = float(total / 4.184)
            terms["total"] = float(total)
            terms["openmm_status_ok"] = 0.0
            terms["openmm_minimize"] = 1.0 if self.openmm_do_minimize else 0.0
            terms["openmm_error"] = 1.0
            terms["openmm_forcefield_hash"] = float(abs(hash(self.openmm_forcefield)) % 1000000)
            terms["openmm_platform_hash"] = float(abs(hash(self.openmm_platform)) % 1000000)
            terms["openmm_max_iterations"] = float(self.openmm_max_iterations)
            terms["openmm_tolerance"] = float(self.openmm_tolerance)
            terms["openmm_ph"] = float(self.openmm_ph)
            terms["openmm_status_hash"] = float(abs(hash(type(exc).__name__)) % 1000000)
            terms["openmm_message_hash"] = float(abs(hash(str(exc))) % 1000000)
            self._last_openmm_coords = None
            self._last_openmm_labels = None
        self.last_energy_terms = {k: float(v) for k, v in terms.items()}
        if self.tracker is not None:
            self.tracker.log(float(total))
        return float(total)

    def _get_angles(self, params: np.ndarray) -> np.ndarray:
        """Map circuit parameters to torsion angles via statevector phases.

        The 2ⁿ complex amplitudes of the statevector each carry a phase in
        ``(-π, π]``.  The first ``total_angles`` phases are used as torsion
        angles after removing the **global phase**.

        A global rotation ``e^{iα}|ψ⟩`` shifts every amplitude phase by α
        uniformly, but is physically unobservable — it would add a spurious
        common offset to every dihedral and waste one degree of freedom.
        The gauge is fixed by subtracting ``angle(ψ₀)`` (the phase of the
        ``|0…0⟩`` basis-state amplitude) from all extracted phases, so that
        ``phases[0]`` is always 0.  The result is wrapped back into
        ``(-π, π]``.

        **Consequence**: the zeroth entry of ``dof_map`` — φ of residue 0,
        the N-terminal φ dihedral — is held at 0 and cannot be optimised.
        This is acceptable: the N-terminal φ has no backbone predecessor and
        is conventionally undefined (or set to a reference value).  The
        optimiser therefore controls ``K − 1`` independent phase degrees of
        freedom for K total torsion angles.
        """
        param_dict = dict(zip(self.ansatz.parameters, params))
        bound_circuit = self.ansatz.assign_parameters(param_dict)
        psi = Statevector(bound_circuit).data
        phases = np.angle(psi)[: self.total_angles]
        # Remove global phase: pin phases[0] to 0 and wrap into (-π, π].
        phases = (phases - np.angle(psi[0]) + np.pi) % (2 * np.pi) - np.pi
        return self._map_angle_vector_to_physical_ranges(phases)

    @staticmethod
    def _nerf_step(
        a: np.ndarray, b: np.ndarray, c: np.ndarray,
        bond_len: float, bond_angle: float, torsion: float,
    ) -> np.ndarray:
        """Natural Extension Reference Frame (NERF) atom placement."""
        bc = c - b
        bc_u = bc / (np.linalg.norm(bc) + 1e-9)
        ab = b - a
        n = np.cross(ab, bc_u)
        n_u = n / (np.linalg.norm(n) + 1e-9)
        bx_n = np.cross(n_u, bc_u)
        M = np.column_stack((bc_u, bx_n, n_u))
        theta_supp = np.pi - bond_angle
        d = np.array([
            bond_len * np.cos(theta_supp),
            bond_len * np.cos(torsion) * np.sin(theta_supp),
            bond_len * np.sin(torsion) * np.sin(theta_supp),
        ])
        return c + (M @ d)

    def build_full_structure(
        self, angle_vector: np.ndarray
    ) -> tuple[np.ndarray, list, list]:
        """Build 3-D Cartesian coordinates from a torsion-angle vector.

        The angle vector is indexed by ``dof_map``.  Most peptide-bond torsion
        angles (ω) are fixed at π (trans-amide); the only exception is the ω
        **preceding a proline residue** (cis-Pro occurs in ~5 % of cases and
        non-planar trans-Pro in another ~5 %).  When the DOF map contains an
        entry ``{"res": i, "type": "omega"}`` the value is read from the angle
        vector and used directly; otherwise ω defaults to π.

        Implementation note — O(1) atom lookup
        ----------------------------------------
        Atom positions are accumulated in a plain Python list ``coords``.
        Earlier versions located backbone atoms via a ``get_idx`` closure that
        scanned ``labels`` in reverse — O(N) per call, making the whole
        function O(N²) in the number of residues.  The current implementation
        maintains ``label_idx``, a ``dict[(res_id, atom_name) → list_index]``
        that is updated whenever an atom is appended through the ``_append``
        helper, giving O(1) lookups throughout.

        Returns
        -------
        coords : ndarray, shape (N_atoms, 3)
        labels : list of (res_id, atom_name, element)
        bonds  : list of (atom_idx_a, atom_idx_b)
        """
        coords: list[np.ndarray] = []
        labels: list[tuple] = []
        bonds: list[tuple] = []

        # O(1) backbone-atom lookup — keeps (res_id, atom_name) → list index.
        # Updated in lock-step with every labels.append() via _append() below.
        label_idx: dict[tuple[int, str], int] = {}

        def _append(res_id: int, atom_name: str, element: str, pos: np.ndarray) -> int:
            """Append one atom and register it in label_idx; return its index."""
            idx = len(coords)
            coords.append(pos)
            labels.append((res_id, atom_name, element))
            label_idx[(res_id, atom_name)] = idx
            return idx

        angle_dict = {f"{x['res']}_{x['type']}": val for x, val in zip(self.dof_map, angle_vector)}

        # Seed backbone frame
        _append(0, "N", "N", np.array([0.0, 0.0, 0.0]))
        _append(0, "CA", "C", np.array([1.46, 0.0, 0.0]))
        _append(0, "C", "C", np.array([1.46 + 1.51 * np.cos(1.9), 1.51 * np.sin(1.9), 0.0]))
        bonds.extend([(0, 1), (1, 2)])

        for i in range(self.n_residues):
            idx_N  = label_idx[(i, "N")]
            idx_CA = label_idx[(i, "CA")]
            idx_C  = label_idx[(i, "C")]

            # Side chain
            topo = self.SIDE_CHAIN_TOPO.get(self.sequence[i], self.SIDE_CHAIN_TOPO["DEFAULT"])
            sc_map: dict[str, int] = {}
            for atom_def in topo:
                name, parent_name, b_len, b_ang, tor_def = atom_def
                elem = self._infer_element_from_atom_name(name)
                if isinstance(tor_def, str) and "chi" in tor_def:
                    t_val = angle_dict.get(f"{i}_{tor_def.replace('_branch', '')}", 0.0)
                    if "branch" in tor_def:
                        t_val += 2.09
                else:
                    t_val = float(tor_def)

                if name == "CB":
                    u_nc = coords[idx_N] - coords[idx_CA]
                    u_cc = coords[idx_C] - coords[idx_CA]
                    n_plane = np.cross(u_nc, u_cc)
                    n_plane /= np.linalg.norm(n_plane) + 1e-9
                    u_mid = -(u_nc + u_cc)
                    u_mid /= np.linalg.norm(u_mid) + 1e-9
                    p_CB = coords[idx_CA] + b_len * (np.cos(0.9) * u_mid + np.sin(0.9) * n_plane)
                    cb_idx = _append(i, name, elem, p_CB)
                    bonds.append((idx_CA, cb_idx))
                    sc_map["CB"] = cb_idx
                else:
                    p_name = "CB"
                    if name.startswith("CD"):
                        p_name = "CG"
                    if name.startswith("CE"):
                        p_name = "CD"
                    if name.startswith("CZ"):
                        p_name = "CE"
                    if name.startswith("NZ"):
                        p_name = "CE"
                    if name.startswith("OE") or name.startswith("OD"):
                        p_name = "CD" if name.startswith("OE") else "CG"
                    if name.startswith("SG"):
                        p_name = "CB"
                    if name.startswith("CG"):
                        p_name = "CB"
                    if name.startswith("CD") and self.sequence[i] == "L":
                        p_name = "CG"
                    if name.startswith("HG") and name != "HG1":
                        p_name = "OG"
                    if name == "HG1":
                        p_name = "OG1"
                    if name == "HH":
                        p_name = "OH"
                    if name == "HE1":
                        p_name = "NE1"
                    if name == "HE2":
                        p_name = "NE2"

                    idx_c = sc_map.get(p_name, -1)
                    if idx_c == -1:
                        idx_c = len(coords) - 1
                    c = coords[idx_c]

                    grandp = "CA" if p_name == "CB" else "CB"
                    if p_name == "OG":
                        grandp = "CB"
                    if p_name == "OG1":
                        grandp = "CB"
                    if p_name == "OH":
                        grandp = "CZ"
                    if p_name == "NE1":
                        grandp = "CD1"
                    if p_name == "NE2":
                        grandp = "CD2"

                    if grandp == "CA":
                        b = coords[idx_CA]
                        a = coords[idx_N]
                    else:
                        b = coords[sc_map.get(grandp, idx_c - 1)]
                        a = coords[idx_CA]

                    new_pos = self._nerf_step(a, b, c, b_len, b_ang, t_val)
                    new_idx = _append(i, name, elem, new_pos)
                    bonds.append((idx_c, new_idx))
                    sc_map[name] = new_idx

            # Carbonyl oxygen
            p_O = self._nerf_step(coords[idx_N], coords[idx_CA], coords[idx_C], 1.23, 2.1, np.pi)
            o_idx = _append(i, "O", "O", p_O)
            bonds.append((idx_C, o_idx))

            # Next residue backbone
            if i < self.n_residues - 1:
                psi = angle_dict.get(f"{i}_psi", -0.5)
                p_next_N = self._nerf_step(coords[idx_N], coords[idx_CA], coords[idx_C], 1.33, 2.0, psi)
                n_idx = _append(i + 1, "N", "N", p_next_N)
                bonds.append((idx_C, n_idx))

                # ω (peptide-bond torsion): π for all residue pairs except
                # when residue i+1 is proline, in which case ω is an explicit
                # DOF and its value is read from angle_dict.
                omega = angle_dict.get(f"{i}_omega", np.pi)
                if f"{i}_omega" not in angle_dict and i < len(self.fixed_omegas):
                    omega = float(self.fixed_omegas[i])
                omega = self._bounded_omega(omega)
                p_next_CA = self._nerf_step(coords[idx_CA], coords[idx_C], p_next_N, 1.46, 2.1, omega)
                ca_idx = _append(i + 1, "CA", "C", p_next_CA)
                bonds.append((n_idx, ca_idx))

                phi = angle_dict.get(f"{i + 1}_phi", -1.0)
                p_next_C = self._nerf_step(coords[idx_C], p_next_N, p_next_CA, 1.51, 1.9, phi)
                c_idx = _append(i + 1, "C", "C", p_next_C)
                bonds.append((ca_idx, c_idx))

        return np.array(coords), labels, bonds

    def _initialize_topology_cache(self) -> None:
        """Pre-compute static atom properties for vectorised energy evaluation."""
        seed = np.full(self.total_angles, _TOPOLOGY_SEED_ANGLE)
        dummy_coords, self.static_labels, static_bonds = self.build_full_structure(seed)
        n_atoms = len(dummy_coords)

        self.atom_to_res = np.array([x[0] for x in self.static_labels], dtype=int)
        self.atom_names = np.array([x[1] for x in self.static_labels])
        self.atom_elems = np.array([x[2] for x in self.static_labels])

        self.q_vector = np.zeros(n_atoms)
        for k, (rid, name, _elem) in enumerate(self.static_labels):
            q = self.CHARGES.get(name, 0.0)
            res_name = self.sequence[rid]
            if res_name == "H":
                if name == "NE2":
                    q = -0.4
                if name == "ND1":
                    q = -0.4
            if rid == 0 or rid == self.n_residues - 1:
                if name in {"N", "CA", "C", "O", "OXT", "H1", "H2", "H3", "H"}:
                    q = 0.0
            self.q_vector[k] = q

        self.lj_type_vector = np.array([
            self._assign_lj_type(rid, name, elem)
            for rid, name, elem in self.static_labels
        ])
        self.vdw_radii_vector = np.array([
            self.LJ_TYPE_PARAMS.get(t, self.LJ_TYPE_PARAMS["X"])["radius"]
            for t in self.lj_type_vector
        ], dtype=float)
        self.lj_epsilon_vector = np.array([
            self.LJ_TYPE_PARAMS.get(t, self.LJ_TYPE_PARAMS["X"])["epsilon"]
            for t in self.lj_type_vector
        ], dtype=float)

        adjacency = [set() for _ in range(n_atoms)]
        for i, j in static_bonds:
            if 0 <= i < n_atoms and 0 <= j < n_atoms:
                adjacency[i].add(j)
                adjacency[j].add(i)

        graph_dist = np.full((n_atoms, n_atoms), 99, dtype=int)
        np.fill_diagonal(graph_dist, 0)
        for i in range(n_atoms):
            frontier = {i}
            visited = {i}
            depth = 0
            while frontier and depth < 3:
                depth += 1
                next_frontier = set()
                for node in frontier:
                    for nbr in adjacency[node]:
                        if nbr in visited:
                            continue
                        visited.add(nbr)
                        if depth < graph_dist[i, nbr]:
                            graph_dist[i, nbr] = depth
                            graph_dist[nbr, i] = depth
                        next_frontier.add(nbr)
                frontier = next_frontier

        offdiag = ~np.eye(n_atoms, dtype=bool)
        self.mask_nonbonded_graph = offdiag & (graph_dist > 3)
        self.mask_14_pairs = offdiag & (graph_dist == 3)

        self.mask_heavy = np.array([not x.startswith("H") for x in self.atom_names], dtype=bool)

        hydro_res_set = set(list("AVLIMFWYPC"))
        self.mask_hydrophobic = np.zeros(n_atoms, dtype=bool)
        for k, (rid, name, elem) in enumerate(self.static_labels):
            aa = self.sequence[rid]
            if aa not in hydro_res_set:
                continue
            if name.startswith("C") and name not in ("C", "CA"):
                self.mask_hydrophobic[k] = True
            elif elem == "S":
                self.mask_hydrophobic[k] = True

        res_diff_matrix = np.abs(self.atom_to_res[:, None] - self.atom_to_res[None, :])
        self.mask_non_bonded = res_diff_matrix >= 2
        self.mask_non_bonded_vdw = self.mask_nonbonded_graph
        self.mask_non_bonded_vdw_14 = self.mask_14_pairs

        self.idx_N_atoms = np.where(self.atom_names == "N")[0]
        self.idx_O_atoms = np.where(self.atom_names == "O")[0]
        self.idx_SG_atoms = np.where(self.atom_names == "SG")[0]
        self._cache_initialized = True

    # ------------------------------------------------------------------
    # Energy function
    # ------------------------------------------------------------------

    def energy_function(self, params: np.ndarray, return_terms: bool = False) -> float:
        """Evaluate physical energy of the structure encoded by *params*."""
        if not self._cache_initialized:
            self._initialize_topology_cache()

        gamma = 15.0
        constraint_strength = 50.0
        if self.current_stage == 3:
            gamma = 5.0
            constraint_strength = 5.0

        angle_vec = self._get_angles(params)
        if self.stage3_backend == "rosetta":
            return self._score_stage3_rosetta(angle_vec, return_terms=return_terms)
        if self.stage3_backend == "openmm":
            return self._score_stage3_openmm(angle_vec, return_terms=return_terms)

        coords, _, _ = self.build_full_structure(angle_vec)
        total_energy = 0.0
        terms = {
            "constraint": 0.0,
            "sasa": 0.0,
            "hbond": 0.0,
            "electrostatics": 0.0,
            "disulfide": 0.0,
            "vdw_repulsion": 0.0,
            "rotamer": 0.0,
            "pi_stacking": 0.0,
            "rama": 0.0,
            "geometry": 0.0,
            "adjacent_heavy_sterics": 0.0,
            "adjacent_backbone_sterics": 0.0,
            "energy_backend_custom": 1.0,
            "energy_backend_rosetta": 0.0,
            "energy_backend_openmm": 0.0,
        }

        def add_term(name: str, value: float) -> None:
            nonlocal total_energy
            v = float(value)
            terms[name] += v
            total_energy += v

        diffs = coords[:, None, :] - coords[None, :, :]
        D = np.sqrt(np.sum(diffs ** 2, axis=-1)) + 1e-9

        ca_indices = [i for i, lbl in enumerate(self.static_labels) if lbl[1] == "CA"]
        if len(ca_indices) >= 2:
            start_ca = coords[ca_indices[0]]
            end_ca = coords[ca_indices[-1]]
            dist_ends = float(np.linalg.norm(start_ca - end_ca))
            target_e2e = float(4.5 + 0.40 * max(0, self.n_residues - 5))
            slack_e2e = float(1.5 + 0.05 * self.n_residues)
            if self.use_e2e_constraint:
                deviation = max(0.0, abs(dist_ends - target_e2e) - slack_e2e)
                add_term("constraint", self.e2e_scale * constraint_strength * (deviation ** 2))

        if np.sum(self.mask_hydrophobic) > 0:
            hydro_dists = D[self.mask_hydrophobic, :]
            weights = 1.0 / (1.0 + np.exp(1.0 * (hydro_dists - 6.0)))
            neighbor_counts = np.sum(weights, axis=1) - 1.0
            burial_fractions = np.clip(neighbor_counts / 15.0, 0.0, 1.0)
            add_term("sasa", np.sum(gamma * 30.0 * (1.0 - burial_fractions)))

        e_hbond = 0.0
        for i_n in self.idx_N_atoms:
            res_d = self.atom_to_res[i_n]
            idx_ca = i_n + 1
            idx_prev_c = i_n - 2
            if idx_prev_c < 0 or self.atom_names[idx_prev_c] != "C":
                pos_h = coords[i_n] + np.array([0, 0, 1.0])
                pos_n = coords[i_n]
            else:
                p_c = coords[idx_prev_c]
                p_n = coords[i_n]
                p_ca = coords[idx_ca]
                v_nc = p_c - p_n
                v_nc /= np.linalg.norm(v_nc)
                v_nca = p_ca - p_n
                v_nca /= np.linalg.norm(v_nca)
                v_h = -(v_nc + v_nca)
                v_h /= np.linalg.norm(v_h)
                pos_h = p_n + v_h * 1.01
                pos_n = p_n
            o_coords = coords[self.idx_O_atoms]
            o_res = self.atom_to_res[self.idx_O_atoms]
            valid_mask = np.abs(o_res - res_d) >= 2
            if not np.any(valid_mask):
                continue
            valid_o_coords = o_coords[valid_mask]
            d_ho = np.linalg.norm(valid_o_coords - pos_h, axis=1)
            close_mask = d_ho < 3.5
            if not np.any(close_mask):
                continue
            final_d_ho = d_ho[close_mask]
            final_o_coords = valid_o_coords[close_mask]
            v_hn = pos_n - pos_h
            v_hn /= np.linalg.norm(v_hn)
            v_ho = final_o_coords - pos_h
            v_ho /= np.linalg.norm(v_ho, axis=1)[:, None]
            angle_cos = np.dot(v_ho, v_hn)
            ang_mask = angle_cos < -0.4
            radial_term = np.exp(-(final_d_ho - 2.0) ** 2 / 0.5)
            angular_term = (np.abs(angle_cos) - 0.4) * 2.0
            e_hbond += np.sum(-25.0 * radial_term * angular_term * ang_mask)
        add_term("hbond", e_hbond)

        add_term("electrostatics", self._electrostatic_energy(D))

        e_disulf = 0.0
        if len(self.idx_SG_atoms) > 1:
            sg_dists = D[np.ix_(self.idx_SG_atoms, self.idx_SG_atoms)]
            sg_mask = np.triu(np.ones_like(sg_dists, dtype=bool), k=1)
            valid_dists = sg_dists[sg_mask]
            bond_strengths = np.exp(-(valid_dists - 2.05) ** 2 / 0.5)
            active_bonds = valid_dists < 3.0
            e_disulf -= np.sum(25.0 * bond_strengths * active_bonds)
            full_strengths = np.exp(-(sg_dists - 2.05) ** 2 / 0.5) * (sg_dists < 3.0)
            np.fill_diagonal(full_strengths, 0.0)
            saturation = np.sum(full_strengths, axis=1)
            overload = saturation - 1.0
            penalty_mask = overload > 0.1
            if np.any(penalty_mask):
                e_disulf += np.sum(40.0 * overload[penalty_mask] ** 2)
        add_term("disulfide", e_disulf)

        Sigma_mat = self.vdw_radii_vector[:, None] + self.vdw_radii_vector[None, :]
        heavy_mat = self.mask_heavy[:, None] & self.mask_heavy[None, :]
        vdw_mask = np.triu(self.mask_non_bonded_vdw & heavy_mat, k=1)
        vdw_14_mask = np.triu(self.mask_non_bonded_vdw_14 & heavy_mat, k=1)

        def _add_vdw(mask: np.ndarray, scale: float) -> None:
            if not np.any(mask):
                return
            r_vdw = D[mask]
            s_vdw = Sigma_mat[mask]
            collision_mask = r_vdw < s_vdw
            if not np.any(collision_mask):
                return
            r_col = r_vdw[collision_mask]
            s_col = s_vdw[collision_mask]
            term = (s_col / (r_col + 0.1)) ** 12
            high_e = term > 50.0
            if np.any(high_e):
                term[high_e] = 50.0 + np.log(term[high_e] - 49.0)
            add_term("vdw_repulsion", scale * np.sum(0.1 * term))

        _add_vdw(vdw_mask, 1.0)
        _add_vdw(vdw_14_mask, 0.35)

        angle_dict = {f"{x['res']}_{x['type']}": val for x, val in zip(self.dof_map, angle_vec)}
        add_term("rotamer", self._calculate_rotamer_energy(angle_dict))
        add_term("pi_stacking", self._calculate_aromatic_quadrupole(coords, self.static_labels, self.atom_to_res))

        e_rama = 0.0
        for i in range(self.n_residues):
            if f"{i}_phi" in angle_dict and f"{i}_psi" in angle_dict:
                phi = angle_dict[f"{i}_phi"]
                psi = angle_dict[f"{i}_psi"]
                aa = self.sequence[i]
                d_helix = (phi - (-1.0)) ** 2 + (psi - (-0.8)) ** 2
                d_sheet = (phi - (-2.3)) ** 2 + (psi - 2.4) ** 2
                if aa == "G":
                    d_helix_L = (phi - 1.0) ** 2 + (psi - 0.8) ** 2
                    d_sheet_L = (phi - 2.3) ** 2 + (psi - (-2.4)) ** 2
                    e_rama += -3.0 * np.exp(-min(d_helix, d_sheet, d_helix_L, d_sheet_L) / 0.6)
                else:
                    d_forbidden = (phi - (-2.0)) ** 2 + (psi - 1.0) ** 2
                    e_rama += -3.0 * np.exp(-d_helix / 0.6) - 3.0 * np.exp(-d_sheet / 0.6) + 5.0 * np.exp(-d_forbidden / 1.0)
        add_term("rama", e_rama)

        add_term("geometry", self._calculate_geometry_integrity(coords, self.static_labels, self.atom_to_res))
        e_local_sterics = self._calculate_adjacent_heavy_sterics(coords, self.static_labels, self.atom_to_res)
        add_term("adjacent_heavy_sterics", e_local_sterics)
        terms["adjacent_backbone_sterics"] = float(e_local_sterics)

        if self.tracker is not None:
            self.tracker.log(total_energy)

        terms["total"] = float(total_energy)
        if return_terms:
            self.last_energy_terms = dict(terms)

        return total_energy

    def _electrostatic_energy(self, D: np.ndarray) -> float:
        """Coulomb electrostatic energy in kcal/mol.

        Uses the standard prefactor 332.0637 kcal mol⁻¹ Å e⁻² and a uniform
        implicit-solvent dielectric ``_DIELECTRIC`` (default 4.0), giving::

            E = 332.0637 · qᵢqⱼ / (4.0 · rᵢⱼ)

        A hard lower bound of 1.0 Å is applied to ``rᵢⱼ`` to prevent
        singularities at unphysically short distances.
        """
        Q_mat = np.outer(self.q_vector, self.q_vector)
        elec_mask = np.triu(self.mask_non_bonded, k=1) & (np.abs(Q_mat) > 0.0001)
        if not np.any(elec_mask):
            return 0.0
        r_elec = np.maximum(D[elec_mask], 1.0)
        return float(np.sum(_COULOMB_PREFACTOR * Q_mat[elec_mask] / (_DIELECTRIC * r_elec)))

    def _calculate_rotamer_energy(self, angle_dict: dict) -> float:
        energy = 0.0
        for i in range(self.n_residues):
            res_name = self.sequence[i]
            key = f"{i}_chi1"
            if key in angle_dict:
                chi = angle_dict[key]
                if res_name in ("V", "I", "T"):
                    energy += -3.0 * (np.exp(-(chi - np.pi) ** 2 / 0.5) + np.exp(-(chi - (-1.047)) ** 2 / 0.5))
                elif res_name == "P":
                    energy += 10.0 * min((chi - (-0.5)) ** 2, (chi - 0.5) ** 2)
                elif res_name in ("W", "F", "Y", "H"):
                    energy += -2.0 * (np.exp(-(chi - np.pi) ** 2 / 0.5) + np.exp(-(chi - (-1.047)) ** 2 / 0.5))
                else:
                    energy += 1.0 * (1.0 + np.cos(3.0 * chi))
        return energy

    def _calculate_aromatic_quadrupole(
        self, coords: np.ndarray, labels: list, atom_to_res_idx: np.ndarray
    ) -> float:
        aromatics: list[tuple] = []
        for r_idx in np.unique(atom_to_res_idx):
            if self.sequence[r_idx] in ("F", "Y", "W"):
                mask = atom_to_res_idx == r_idx
                r_coords = coords[mask]
                r_names = self.atom_names[mask]
                ring_mask = np.isin(r_names, ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"])
                ring_atoms = r_coords[ring_mask]
                if len(ring_atoms) > 2:
                    centroid = np.mean(ring_atoms, axis=0)
                    v1 = ring_atoms[1] - ring_atoms[0]
                    v2 = ring_atoms[2] - ring_atoms[0]
                    normal = np.cross(v1, v2)
                    normal /= np.linalg.norm(normal) + 1e-9
                    aromatics.append((centroid, normal))
        energy_pi = 0.0
        for i in range(len(aromatics)):
            for j in range(i + 1, len(aromatics)):
                c1, n1 = aromatics[i]
                c2, n2 = aromatics[j]
                dist = np.linalg.norm(c1 - c2)
                if dist > 7.0:
                    continue
                alignment = abs(np.dot(n1, n2))
                if alignment < 0.3 and 4.5 < dist < 6.0:
                    energy_pi -= 4.0 * np.exp(-(dist - 5.0) ** 2)
                elif alignment > 0.8 and 3.4 < dist < 4.5:
                    energy_pi -= 5.0 * np.exp(-(dist - 3.8) ** 2)
        return energy_pi

    @staticmethod
    def _huber(x: float, delta: float) -> float:
        """Huber loss: quadratic for |x| ≤ delta, linear beyond.

        The Huber loss is continuously differentiable at the transition point
        |x| = delta (both branches equal delta² and their derivatives equal
        ±2·delta there), so the combined landscape remains smooth.

        For |x| ≤ delta:  L = x²
        For |x| > delta:  L = 2·delta·|x| − delta²

        Parameters
        ----------
        x:
            Residual, e.g. a bond-length deviation in Å or a volume error
            in Å³.
        delta:
            Transition threshold.  Choose to match the scale of tolerable
            distortion; ``_HUBER_DELTA_GEOM`` (1.0 Å) is the default for all
            geometry-integrity terms.

        Returns
        -------
        float
            Non-negative Huber-loss value.
        """
        ax = abs(x)
        if ax <= delta:
            return float(x * x)
        return float(2.0 * delta * ax - delta * delta)

    def _calculate_geometry_integrity(
        self, coords: np.ndarray, labels: list, atom_to_res_idx: np.ndarray, return_terms: bool = False
    ) -> float | tuple[float, dict[str, float]]:
        """Evaluate hard geometry constraints as soft energy penalties."""
        energy = 0.0
        geom_terms = {"pro_ring": 0.0, "chirality": 0.0, "planarity": 0.0}

        res_map: dict[int, dict[str, int]] = {}
        for k, lbl in enumerate(labels):
            r, atom = lbl[0], lbl[1]
            if r not in res_map:
                res_map[r] = {}
            res_map[r][atom] = k
        for r in range(self.n_residues):
            atoms = res_map.get(r, {})
            res_name = self.sequence[r]
            if res_name == "P" and "CD" in atoms and "N" in atoms:
                d = np.linalg.norm(coords[atoms["CD"]] - coords[atoms["N"]])
                dev = d - 1.47
                if abs(dev) > 0.1:
                    penalty = 50.0 * self._huber(dev, _HUBER_DELTA_GEOM)
                    energy += penalty
                    geom_terms["pro_ring"] += penalty
            if all(k in atoms for k in ("CA", "N", "C", "CB")):
                ca = coords[atoms["CA"]]
                n = coords[atoms["N"]]
                c = coords[atoms["C"]]
                cb = coords[atoms["CB"]]
                volume = np.dot(np.cross(n - ca, c - ca), cb - ca)
                if volume < 1.0:
                    penalty = 50.0 * self._huber(1.0 - volume, _HUBER_DELTA_GEOM)
                    energy += penalty
                    geom_terms["chirality"] += penalty
            if r < self.n_residues - 1:
                next_atoms = res_map.get(r + 1, {})
                if all(k in atoms for k in ("C", "CA")) and all(k in next_atoms for k in ("N", "CA")):
                    p1, p2 = coords[atoms["CA"]], coords[atoms["C"]]
                    p3, p4 = coords[next_atoms["N"]], coords[next_atoms["CA"]]
                    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
                    n1 = np.cross(b1, b2)
                    n2 = np.cross(b2, b3)
                    n1_norm = np.linalg.norm(n1)
                    n2_norm = np.linalg.norm(n2)
                    if n1_norm > 1e-8 and n2_norm > 1e-8:
                        n1 /= n1_norm
                        n2 /= n2_norm
                        twist_penalty = 1.0 - abs(np.dot(n1, n2))
                        if twist_penalty > 0.05:
                            penalty = 20.0 * twist_penalty
                            energy += penalty
                            geom_terms["planarity"] += penalty
        if return_terms:
            return energy, geom_terms
        return energy

    def _calculate_adjacent_heavy_sterics(self, coords, labels, atom_to_res_idx, return_terms=False):
        """
        Penalize obvious clashes between heavy atoms on adjacent residues.

        This is intentionally narrower than the full VDW term:
          - it only looks at residue i and i+1
          - it only considers heavy atoms
          - it only activates when atoms are pushed into an unrealistically short range

        The goal is to suppress local backbone overlaps that produce fake bonds in
        viewers, while still allowing legitimate backbone H-bonding geometry.
        """
        coords = np.asarray(coords, dtype=float)
        labels = list(labels)

        res_map = {}
        for k, lbl in enumerate(labels):
            r = int(lbl[0])
            atom = str(lbl[1])
            if r not in res_map:
                res_map[r] = {}
            res_map[r][atom] = k

        scale = float(os.getenv("QTF_LOCAL_HEAVY_STERIC_SCALE", "10.0"))
        min_allowed_A = float(os.getenv("QTF_LOCAL_HEAVY_STERIC_MIN_A", "1.35"))
        threshold_frac = float(os.getenv("QTF_LOCAL_HEAVY_STERIC_FRACTION", "0.55"))
        overlap_width_A = float(os.getenv("QTF_LOCAL_HEAVY_STERIC_WIDTH_A", "0.50"))
        overlap_width_A = max(overlap_width_A, 1e-3)

        energy = 0.0
        terms = {
            "adjacent_heavy_sterics": 0.0,
        }

        for r in range(self.n_residues - 1):
            left = res_map.get(r, {})
            right = res_map.get(r + 1, {})
            for a1, i in left.items():
                if str(labels[i][2]).upper() == "H" or str(labels[i][1]).upper().startswith("H"):
                    continue
                for a2, j in right.items():
                    if str(labels[j][2]).upper() == "H" or str(labels[j][1]).upper().startswith("H"):
                        continue
                    if a1 == "C" and a2 == "N":
                        continue
                    d = float(np.linalg.norm(coords[i] - coords[j]))
                    threshold_A = max(min_allowed_A, threshold_frac * (float(self.vdw_radii_vector[i]) + float(self.vdw_radii_vector[j])))
                    if d >= threshold_A:
                        continue
                    shortfall = threshold_A - d
                    # Quadratic wall with a soft activation range so the penalty
                    # stays mild near the edge but rises quickly for real overlaps.
                    penalty = scale * (shortfall / overlap_width_A) ** 2
                    energy += penalty
                    terms["adjacent_heavy_sterics"] += penalty

        if return_terms:
            return energy, terms
        return energy

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def get_smart_initialization(self, n_attempts: int = 20, seed: int | None = None) -> np.ndarray:
        """Sample random parameter sets and return the one with the lowest energy.

        Parameters
        ----------
        n_attempts:
            Number of random samples to evaluate.
        seed:
            Random seed.  When ``None`` a deterministic seed is derived from
            the protein sequence so results are reproducible per sequence.
        """
        if seed is None:
            seed = int(hashlib.sha256(self.sequence.encode()).hexdigest(), 16) % (2 ** 32)

        rng = np.random.default_rng(seed)
        logger.debug("Scouting %d starting points (seed=%d)", n_attempts, seed)

        best_params: np.ndarray | None = None
        best_energy = float("inf")
        for _ in range(n_attempts):
            trial_params = rng.uniform(-0.8, 0.8, self.n_params)
            e = self.energy_function(trial_params)
            if e < best_energy:
                best_energy = e
                best_params = trial_params

        logger.debug("Best start found: energy=%.2f", best_energy)
        assert best_params is not None
        return best_params

    # ------------------------------------------------------------------
    # Folding
    # ------------------------------------------------------------------

    def fold(
        self,
        max_iter: int = 2000,
        initial_params: np.ndarray | None = None,
        scout_attempts: int | None = None,
    ) -> tuple[np.ndarray, list, list, LandscapeTracker, np.ndarray, float]:
        """Run the three-stage optimisation curriculum.

        Parameters
        ----------
        max_iter:
            Maximum number of energy evaluations allowed for **each** of the
            three optimisation stages (COBYLA collapse, SLSQP refine, SLSQP
            relax).  The default of 2000 is a generous budget suitable for
            peptides up to ~20 residues.
        initial_params:
            Pre-computed circuit parameters to use as the starting point for
            Stage 1.  When *None* (the default) a scouting phase randomly
            samples ``scout_attempts`` parameter vectors and picks the one
            with the lowest energy.
        scout_attempts:
            Number of random parameter vectors evaluated during the scouting
            phase (only used when ``initial_params`` is *None*).  A larger
            value improves the quality of the starting point at a linear cost
            in energy evaluations.

            If *None* (the default) the value is computed as
            ``min(64, max_iter // 10)``, which keeps the scouting budget at
            most 10 % of one optimisation stage while still sampling at least
            a few dozen points.  Pass an explicit integer to override —
            e.g. ``scout_attempts=1`` for the fastest possible run during
            testing, or ``scout_attempts=200`` for a thorough global search.

        Returns
        -------
        coords, labels, bonds, tracker, final_params, final_energy
        """
        logger.info("Starting quantum folding (max_iter=%d)", max_iter)
        self.tracker = LandscapeTracker()

        if initial_params is None:
            n_scout = min(64, max_iter // 10) if scout_attempts is None else scout_attempts
            logger.info("Scouting %d starting points…", n_scout)
            init_params = self.get_smart_initialization(n_attempts=n_scout)
        else:
            init_params = initial_params

        logger.info("Stage 1: Mechanical Collapse (high force)…")
        self.tracker.mark_stage("Stage1")
        self.current_stage = 1
        # COBYLA emits a warning ("Invalid MAXFUN; it should be at
        # least num_vars + 2; it is set to N") whenever ``maxiter`` is
        # below ``n_params + 2``. For small ``max_iter`` budgets (e.g.
        # the smoke-test ``max_iter=10`` used by
        # ``TestFoldScoutBudget``) this floor is violated, the warning
        # leaks into the ensemble output, and the user is left
        # wondering whether the run is broken. Enforce the floor
        # explicitly so the warning is structurally impossible.
        safe_maxiter = max(int(max_iter), int(self.n_params) + 2)
        res_1 = minimize(self.energy_function, init_params, method="COBYLA",
                         options={"maxiter": safe_maxiter, "rhobeg": 1.0})
        logger.info("  Collapse energy: %.2f", res_1.fun)

        logger.info("Stage 2: Physics Refinement (high force)…")
        self.tracker.mark_stage("Stage2")
        self.current_stage = 2
        res_2 = minimize(self.energy_function, res_1.x, method="SLSQP",
                         tol=1e-6, options={"maxiter": max_iter, "disp": False})
        logger.info("  Refinement energy: %.2f", res_2.fun)

        logger.info("Stage 3: Natural Relaxation (releasing constraints)…")
        self.tracker.mark_stage("Stage3")
        self.current_stage = 3
        res_3 = minimize(self.energy_function, res_2.x, method="SLSQP",
                         tol=1e-6, options={"maxiter": max_iter, "disp": False})
        logger.info("  Final energy: %.2f", res_3.fun)

        self.energy_function(res_3.x, return_terms=True)
        coords, labels, bonds = self._final_output_structure_from_params(res_3.x)
        final_energy = self.last_energy_terms.get(
            "total",
            self.last_energy_terms.get(
                "rosetta_total",
                self.last_energy_terms.get("openmm_potential_kj_mol", float(res_3.fun)),
            ),
        )
        return coords, labels, bonds, self.tracker, res_3.x, float(final_energy)

    def compute_sidechain_centroids(self, coords, labels):
        """
        Compute one heavy-atom sidechain centroid per residue from rebuilt coordinates.
        Backbone atoms and hydrogens are excluded.
        """
        backbone_atoms = {'N', 'CA', 'C', 'O', 'OXT'}
        by_residue = {}

        for pos, (res_id, atom_name, elem) in zip(coords, labels):
            if atom_name in backbone_atoms:
                continue
            if atom_name.startswith('H') or elem == 'H':
                continue
            by_residue.setdefault(int(res_id), []).append(np.asarray(pos, dtype=float))

        return {
            rid: np.mean(np.vstack(points), axis=0)
            for rid, points in by_residue.items()
            if points
        }

    def _aa1_to_3(self, aa):
        aa1_to_3 = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
        }
        return aa1_to_3.get(str(aa).upper(), 'UNK')

    def _format_pdb_atom_line(self, serial, atom_name, res_name, chain_id, resseq, x, y, z, element='C'):
        """Backwards-compatible thin wrapper around the canonical
        PDB atom-line formatter in :mod:`qtf.utils.pdb`.

        The implementation now lives in ``qtf.utils.pdb._format_atom_line``;
        this method is kept so external subclasses that override it
        continue to work (B5).
        """
        from qtf.utils.pdb import _format_atom_line as _fmt
        return _fmt(serial, atom_name, res_name, chain_id, resseq, x, y, z, element)

    def save_pdb(self, coords, labels, filename="structure.pdb", energy=0.0, chain_id='A', resseqs=None, resnames=None, remarks=None, include_hydrogens=True):
        """
        Save arbitrary coordinates/labels to a PDB file viewable in PyMOL or Chimera.

        This is a thin wrapper around the canonical
        :func:`qtf.utils.pdb.save_pdb` (B5). The folder method's
        signature is preserved for backward compatibility, but the
        implementation is centralised so fixes and feature work on
        PDB I/O only have to happen in one place.

        Args:
            coords: array-like of shape (N, 3)
            labels: iterable of (res_id, atom_name, element)
            filename: output PDB path
            energy: optional energy remark value
            chain_id: output chain identifier
            resseqs: optional list/dict mapping res_id -> residue number
            resnames: optional list/dict mapping res_id -> residue name (3-letter preferred)
            remarks: optional iterable of additional REMARK strings
            include_hydrogens: if False, omit atoms whose element/name is hydrogen
        """
        from qtf.utils.pdb import save_pdb as _save_pdb
        _save_pdb(
            coords=coords,
            labels=labels,
            filename=filename,
            energy=energy,
            chain_id=chain_id,
            resseqs=resseqs,
            resnames=resnames,
            remarks=remarks,
            include_hydrogens=include_hydrogens,
            sequence=self.sequence,
        )

    def save_reduced_pdb(self, ca_coords, filename="structure_ca.pdb", sidechain_centroids=None, energy=0.0,
                         chain_id='A', resseqs=None, resnames=None):
        """
        Save a reduced PDB containing CA only, or CA plus one sidechain centroid pseudoatom (SC) per residue.
        """
        ca_coords = np.asarray(ca_coords, dtype=float)
        labels = []
        coords_out = []
        n_res = len(ca_coords)

        for i in range(n_res):
            coords_out.append(ca_coords[i])
            labels.append((i, 'CA', 'C'))
            if sidechain_centroids is not None and i in sidechain_centroids:
                sc = np.asarray(sidechain_centroids[i], dtype=float)
                coords_out.append(sc)
                labels.append((i, 'SC', 'C'))

        remarks = [
            'REDUCED REPRESENTATION GENERATED FROM QTF-OPTIMIZED STRUCTURE',
            'CONTENTS: CA ONLY' if sidechain_centroids is None else 'CONTENTS: CA PLUS SIDCHAIN CENTROID PSEUDOATOMS (SC)',
        ]
        self.save_pdb(coords_out, labels, filename=filename, energy=energy, chain_id=chain_id,
                      resseqs=resseqs, resnames=resnames, remarks=remarks)

# ==========================================
# 4. ORCHESTRATOR: ENSEMBLE MANAGER
# ==========================================
