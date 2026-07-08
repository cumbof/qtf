"""QuantumBiophysicsFolder — hybrid quantum-classical protein structure predictor.

This implementation uses the parent-aware topology/rebuild path and custom
energy choices ported from bryan_working_branch:QTF/runner.py, with the main
package orchestration API.


Architecture
------------
1. **Quantum Actor**: a parameterised quantum circuit (EfficientSU2 by
   default) whose statevector phases encode backbone/side-chain torsion
   angles.
2. **Classical Critic**: a physics-based energy function (hydrophobicity, H-bonds,
   electrostatics, sterics, Ramachandran bias, geometry integrity).
3. **Optimisation Loop**: COBYLA + SLSQP in three progressive stages (collapse →
   refine → relax).

References
----------
* Kyte & Doolittle (1982) hydrophobicity scale.
* QTF coarse effective charges, with AMBER ff14SB-inspired backbone/polar values.
* Bondi (1964) van der Waals radii.
* Engh & Huber (1991) bond/angle parameters.
"""

from __future__ import annotations

import hashlib
import heapq
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from qtf.core.tracker import LandscapeTracker
from qtf.utils import gromacs as qtf_gromacs
from qtf.utils.aer_sim import statevector_data as _statevector_data

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

# ---------------------------------------------------------------------------
# Optional Numba acceleration
# ---------------------------------------------------------------------------
try:
    from qtf.utils.accelerate import (
        distance_matrix as _dist_accel,
        electrostatic_energy as _elec_accel,
        vdw_repulsion as _vdw_accel,
        sasa_energy as _sasa_accel,
    )

    _ACCELERATE_AVAILABLE = True
except ImportError:
    _ACCELERATE_AVAILABLE = False

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

    # Hard caps for exposed chi DOFs. These trim redundant terminal torsions
    # while preserving the physically meaningful rotatable bonds for each side
    # chain.
    _CHI_CAP_BY_RESIDUE: dict[str, int] = {
        "C": 1,
        "D": 1,
        "E": 2,
        "S": 1,
        "T": 1,
        "V": 1,
        "I": 2,
        "L": 1,
        "M": 3,
        "K": 4,
        "R": 4,
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
        ansatz: str | None = None,
        mode: str = "statevector",
        backend=None,
        shots: int = 4096,
    ) -> None:
        """
        Parameters
        ----------
        sequence:
            Single-letter amino acid sequence (e.g. ``"MAGTWY"``).
        ansatz:
            Quantum circuit ansatz.  ``None`` (default) → ``EfficientSU2``
            with ``circular`` entanglement.  A string selects a built-in
            Qiskit circuit-library ansatz:

            - ``"efficient_su2"`` (or ``"su2"``)
            - ``"real_amplitudes"`` (or ``"ra"``)
            - ``"brickwork"`` — custom brick-layer entanglement with lower
              circuit depth than circular; suitable for near-term hardware.

            You can also pass a fully constructed
            ``qiskit.circuit.QuantumCircuit`` with ``num_parameters > 0``.
            When a custom circuit is provided the qubit count is taken
            from the circuit (``n_qubits`` = ``circuit.num_qubits``) and
            must be ≥ ``ceil(log2(total_angles))`` so that the statevector
            has enough amplitude slots for all torsion-angle DOFs.
        mode:
            Quantum simulation mode.  ``"statevector"`` (default) extracts
            torsion angles from the complex phases of the full statevector.
            ``"sampler"`` runs a shot-based measurement and derives angles
            from the empirical probability CDF, enabling execution on real
            quantum hardware.
        backend:
            Quantum backend instance (e.g. from ``qiskit_ibm_runtime``).
            Used only when ``mode="sampler"``.  If ``None``, an
            ``AerSimulator`` is used automatically.
        shots:
            Number of measurement shots for ``mode="sampler"``.
            Ignored when ``mode="statevector"``.
        """
        if mode not in ("statevector", "sampler"):
            raise ValueError(
                f"Unknown mode {mode!r}; choose from 'statevector' or 'sampler'"
            )
        self.mode = mode
        self.backend = backend
        self.shots = int(shots)

        self.sequence = sequence.upper()
        self.n_residues = len(self.sequence)

        logger.info("Initialising QuantumBiophysicsFolder | seq=%s | mode=%s", self.sequence, self.mode)

        self.HYDROPHOBICITY = self._HYDROPHOBICITY
        self.VDW_RADII = self._VDW_RADII
        self.SIDE_CHAIN_TOPO = self._SIDE_CHAIN_TOPO

        self.CHARGES = self._build_charges()
        self._chi_mode = chi_mode
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

        self.SIDE_CHAIN_TOPO = {
            'G': [],
            'A': [('CB', 'CA', 1.53, 1.91, 2.1)],

            # Hydrophobic
            'V': [('CB', 'CA', 1.53, 1.91, 2.1),
                  ('CG1', 'CB', 1.52, 1.91, 'chi1'), ('CG2', 'CB', 1.52, 1.91, 'chi1_branch')],
            'L': [('CB', 'CA', 1.53, 1.91, 2.1),
                  ('CG', 'CB', 1.52, 1.91, 'chi1'),
                  ('CD1', 'CG', 1.52, 1.91, 'chi2'), ('CD2', 'CG', 1.52, 1.91, 'chi2_branch')],
            'I': [('CB', 'CA', 1.53, 1.91, 2.1),
                  ('CG1', 'CB', 1.54, 1.91, 'chi1'), ('CD1', 'CG1', 1.52, 1.91, 'chi2'),
                  ('CG2', 'CB', 1.54, 1.91, 'chi1_branch')],
            'M': [('CB', 'CA', 1.53, 1.91, 2.1), ('CG', 'CB', 1.52, 1.91, 'chi1'),
                  ('SD', 'CG', 1.81, 1.91, 'chi2'), ('CE', 'SD', 1.79, 1.76, 'chi3')],
            'P': [('CB', 'CA', 1.53, 1.80, 2.1), ('CG', 'CB', 1.50, 1.82, 'chi1'),
                  ('CD', 'CG', 1.52, 1.83, 'chi2')],

            # Aromatic
            'F': [('CB', 'CA', 1.53, 1.91, 2.1), ('CG', 'CB', 1.50, 1.91, 'chi1'),
                  ('CD1', 'CG', 1.39, 2.09, 'chi2'), ('CD2', 'CG', 1.39, 2.09, -1.57),
                  ('CE1', 'CD1', 1.39, 2.09, 3.14), ('CE2', 'CD2', 1.39, 2.09, 3.14),
                  ('CZ', 'CE1', 1.39, 2.09, 0.0)],
            'Y': [('CB', 'CA', 1.53, 1.91, 2.1), ('CG', 'CB', 1.50, 1.91, 'chi1'),
                  ('CD1', 'CG', 1.39, 2.09, 'chi2'), ('CD2', 'CG', 1.39, 2.09, -1.57),
                  ('CE1', 'CD1', 1.39, 2.09, 3.14), ('CE2', 'CD2', 1.39, 2.09, 3.14),
                  ('CZ', 'CE1', 1.39, 2.09, 0.0),
                  ('OH', 'CZ', 1.37, 2.09, 3.14), ('HH', 'OH', 0.96, 1.83, 0.0)],
            'W': [('CB', 'CA', 1.53, 1.91, 2.1), ('CG', 'CB', 1.50, 1.91, 'chi1'),
                  ('CD1', 'CG', 1.37, 2.15, 'chi2'), ('CD2', 'CG', 1.43, 2.15, -1.0),
                  ('NE1', 'CD1', 1.38, 1.90, 3.14), ('HE1', 'NE1', 1.01, 2.09, 0.0),
                  ('CE2', 'CD2', 1.40, 1.90, 0.0), ('CE3', 'CD2', 1.40, 2.30, 3.14),
                  ('CZ2', 'CE2', 1.40, 2.10, 0.0), ('CZ3', 'CE3', 1.40, 2.10, 0.0),
                  ('CH2', 'CZ2', 1.40, 2.10, 0.0)],

            # Polar / Charged
            'S': [('CB', 'CA', 1.53, 1.91, 2.1), ('OG', 'CB', 1.42, 1.91, 'chi1'),
                  ('HG', 'OG', 0.96, 1.83, 0.0)],
            'T': [('CB', 'CA', 1.53, 1.91, 2.1),
                  ('OG1', 'CB', 1.43, 1.91, 'chi1'), ('HG1', 'OG1', 0.96, 1.83, 0.0),
                  ('CG2', 'CB', 1.53, 1.91, 'chi1_branch')],
            'C': [('CB', 'CA', 1.53, 1.91, 2.1), ('SG', 'CB', 1.81, 1.91, 'chi1')],
            'D': [('CB', 'CA', 1.53, 1.91, 2.1), ('CG', 'CB', 1.52, 1.91, 'chi1'),
                  ('OD1', 'CG', 1.25, 2.0, 'chi2'), ('OD2', 'CG', 1.25, 2.0, 'chi2_branch')],
            'N': [('CB', 'CA', 1.53, 1.91, 2.1), ('CG', 'CB', 1.52, 1.91, 'chi1'),
                  ('OD1', 'CG', 1.23, 2.09, 'chi2'), ('ND2', 'CG', 1.32, 2.09, 'chi2_branch')],
            'E': [('CB', 'CA', 1.53, 1.91, 2.1), ('CG', 'CB', 1.52, 1.91, 'chi1'),
                  ('CD', 'CG', 1.52, 1.91, 'chi2'), ('OE1', 'CD', 1.25, 2.0, 'chi3'), ('OE2', 'CD', 1.25, 2.0, 'chi3_branch')],
            'Q': [('CB', 'CA', 1.53, 1.91, 2.1), ('CG', 'CB', 1.52, 1.91, 'chi1'),
                  ('CD', 'CG', 1.52, 1.91, 'chi2'), ('OE1', 'CD', 1.23, 2.09, 'chi3'), ('NE2', 'CD', 1.32, 2.09, 'chi3_branch')],
            'K': [('CB', 'CA', 1.53, 1.91, 2.1), ('CG', 'CB', 1.52, 1.91, 'chi1'),
                  ('CD', 'CG', 1.52, 1.91, 'chi2'), ('CE', 'CD', 1.52, 1.91, 'chi3'),
                  ('NZ', 'CE', 1.49, 1.91, 'chi4')],
            'R': [('CB', 'CA', 1.53, 1.91, 2.1), ('CG', 'CB', 1.52, 1.91, 'chi1'),
                  ('CD', 'CG', 1.52, 1.91, 'chi2'), ('NE', 'CD', 1.46, 1.91, 'chi3'),
                  ('CZ', 'NE', 1.33, 2.15, 'chi4'), ('NH1', 'CZ', 1.33, 2.10, 0.0), ('NH2', 'CZ', 1.33, 2.10, 3.14)],
            'H': [('CB', 'CA', 1.53, 1.91, 2.1), ('CG', 'CB', 1.50, 1.91, 'chi1'),
                  ('ND1', 'CG', 1.38, 2.15, 'chi2'), ('CD2', 'CG', 1.36, 2.15, -1.0),
                  ('CE1', 'ND1', 1.32, 1.90, 0.0),
                  ('NE2', 'CD2', 1.32, 1.90, 0.0), ('HE2', 'NE2', 1.01, 2.09, 0.0)],

            'DEFAULT': [('CB', 'CA', 1.53, 1.91, 2.1)]
        }

        # --- QUANTUM SETUP ---
        # 1. Map sequence to Degrees of Freedom (DoF)
        self.dof_map = []
        for i, aa in enumerate(self.sequence):
            self.dof_map.append({'res': i, 'type': 'phi'})
            self.dof_map.append({'res': i, 'type': 'psi'})
            if i < self.n_residues - 1:
                self.dof_map.append({'res': i, 'type': 'omega'})

            topo = self.SIDE_CHAIN_TOPO.get(aa, self.SIDE_CHAIN_TOPO['DEFAULT'])
            chis = set()
            for atom in topo:
                atom_name = str(atom[0])
                elem = self._infer_element_from_atom_name(atom_name)
                if elem == 'H' or atom_name.startswith('H'):
                    continue
                tor = atom[4]
                if isinstance(tor, str) and 'chi' in tor:
                    chis.add(tor.replace('_branch', ''))

            allowed_chis = self._allowed_chis_for_residue(i, aa, chis)
            for k in allowed_chis:
                self.dof_map.append({'res': i, 'type': k})

        self._total_angles = len(self.dof_map)

        # ------------------------------------------------------------------
        # Quantum circuit (ansatz)
        # ------------------------------------------------------------------
        min_qubits = max(2, int(np.ceil(np.log2(self.total_angles))))

        if ansatz is None:
            self.n_qubits = min_qubits
            self.reps = int(np.ceil(self.total_angles / self.n_qubits)) + 2
            try:
                from qiskit.circuit.library import efficient_su2
                self.ansatz = efficient_su2(
                    self.n_qubits, reps=self.reps, entanglement="circular",
                )
            except ImportError:
                from qiskit.circuit.library import EfficientSU2
                self.ansatz = EfficientSU2(
                    self.n_qubits, reps=self.reps, entanglement="circular",
                )
        elif isinstance(ansatz, str):
            from qiskit.circuit.library import EfficientSU2, RealAmplitudes
            name = ansatz.lower().replace("-", "_")
            self.n_qubits = min_qubits
            self.reps = int(np.ceil(self.total_angles / self.n_qubits)) + 2
            if name in ("efficient_su2", "su2"):
                self.ansatz = EfficientSU2(
                    self.n_qubits, reps=self.reps, entanglement="circular",
                )
            elif name in ("real_amplitudes", "ra"):
                self.ansatz = RealAmplitudes(
                    self.n_qubits, reps=self.reps, entanglement="circular",
                )
            elif name == "brickwork":
                self.ansatz = self._build_brickwork_ansatz(self.n_qubits, self.reps)
            else:
                raise ValueError(
                    f"Unknown ansatz name {ansatz!r}; "
                    f"choose from 'efficient_su2', 'real_amplitudes', 'brickwork'"
                )
        else:
            from qiskit.circuit import QuantumCircuit
            if not isinstance(ansatz, QuantumCircuit):
                raise TypeError(
                    f"ansatz must be a string, a QuantumCircuit, or None; "
                    f"got {type(ansatz).__name__}"
                )
            if ansatz.num_parameters == 0:
                raise ValueError(
                    "Custom ansatz circuit must have at least one parameter"
                )
            if ansatz.num_qubits < min_qubits:
                raise ValueError(
                    f"Custom ansatz has {ansatz.num_qubits} qubit(s); "
                    f"need at least {min_qubits} for {self.total_angles} angles"
                )
            self.n_qubits = ansatz.num_qubits
            self.ansatz = ansatz

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

        self.stage_backend = (
            energy_backend or "custom"
        ).strip().lower()
        if self.stage_backend not in ("custom", "rosetta", "openmm"):
            raise ValueError("energy_backend must be 'custom', 'rosetta', or 'openmm'")

        if self.stage_backend == "rosetta" and not _PYROSETTA_AVAILABLE:
            raise ImportError(
                "PyRosetta is not installed. Install it with:\n"
                "    pip install pyrosetta "
                "--find-links https://west.rosettacommons.org/pyrosetta/quarterly/release"
            )
        if self.stage_backend == "openmm" and not _OPENMM_AVAILABLE:
            raise ImportError(
                "OpenMM is not installed. Install it with: "
                "conda install -c conda-forge openmm"
            )

        self.use_e2e_constraint = _as_bool(use_e2e_constraint, True)
        self.e2e_scale = float(
            e2e_scale if e2e_scale is not None else "1.0"
        )
        self.rosetta_flags = "-mute all"
        self.rosetta_centroid_weights = "cen_std"
        self.rosetta_fullatom_weights = "ref2015"
        self.rosetta_cen_weight = 0.35
        self.rosetta_fa_weight = 1.0
        self.rosetta_do_centroid_min = _as_bool(rosetta_cen_min, False)
        self.rosetta_do_fullatom_min = _as_bool(rosetta_fa_min, False)
        self.rosetta_do_repack = _as_bool(rosetta_repack, False)
        self._rosetta_ready = False
        self._rosetta_scorefxn_cen = None
        self._rosetta_scorefxn_fa = None
        self._last_rosetta_pose = None
        self._last_rosetta_ca = None
        self.openmm_forcefield = "amber14-all.xml"
        self.openmm_platform = "CPU"
        self.openmm_do_minimize = False
        self.openmm_max_iterations = 200
        self.openmm_tolerance = 10.0
        self.openmm_ph = 7.0
        self._openmm_ready = False
        self._last_openmm_coords = None
        self._last_openmm_labels = None
        self.tracker: LandscapeTracker | None = None
        self.last_energy_terms: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Read-only topological properties
    # ------------------------------------------------------------------

    @property
    def total_angles(self) -> int:
        """Number of torsion-angle degrees of freedom (read-only)."""
        return self._total_angles

    @property
    def chi_mode(self) -> str:
        """Side-chain chi angle sampling mode (read-only)."""
        return self._chi_mode

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_charges() -> dict[str, float]:
        """Return QTF's coarse effective electrostatic charges.

        The custom scorer is mostly heavy-atom based. These charges are
        therefore not a direct all-atom AMBER force-field table. Backbone and
        polar heavy-atom values are AMBER ff14SB-inspired where applicable,
        while charged sidechains use coarse group charges. Charges from the
        explicit polar hydrogens present in the QTF topology are folded into
        their parent heavy atoms, and the hydrogen atoms themselves are assigned
        zero electrostatic charge. Donor hydrogens are still used geometrically
        by H-bond terms, not as separate electrostatic particles. A future
        FF19SB effective-charge model should replace this table with
        residue-template-derived collapsed charges.
        """
        common: dict[str, float] = {
            "OXT": -1.0,
            "NZ": 1.0, "NH1": 0.5, "NH2": 0.5,
            "OD1": -0.5, "OD2": -0.5, "OE1": -0.5, "OE2": -0.5,
            "ND2": 0.5, "NE2": 0.5,
            "SG": -0.1, "SD": -0.1,
            "ND1": -0.4,
        }
        amber_like: dict[str, float] = {
            "N": -0.42, "CA": 0.00, "C": 0.60, "O": -0.57,
            # Explicit donor-H charges from the old table are collapsed into
            # parent heavy atoms: OG/HG, OG1/HG1, OH/HH, NE1/HE1.
            "OG": -0.2, "OG1": -0.2, "OH": -0.1,
            "NE1": -0.1,
        }
        hydrogen: dict[str, float] = {
            "H": 0.0, "HN": 0.0, "H1": 0.0, "H2": 0.0, "H3": 0.0,
            "HA": 0.0, "HA2": 0.0, "HA3": 0.0,
            "HB": 0.0, "HB1": 0.0, "HB2": 0.0, "HB3": 0.0,
            "HG": 0.0, "HG1": 0.0, "HG2": 0.0, "HG3": 0.0,
            "HD": 0.0, "HD1": 0.0, "HD2": 0.0, "HD3": 0.0,
            "HE": 0.0, "HE1": 0.0, "HE2": 0.0, "HE3": 0.0,
            "HZ": 0.0, "HZ1": 0.0, "HZ2": 0.0, "HZ3": 0.0,
            "HH": 0.0, "HH11": 0.0, "HH12": 0.0, "HH21": 0.0, "HH22": 0.0,
        }
        charges = common.copy()
        charges.update(amber_like)
        charges.update(hydrogen)
        return charges

    def _allowed_chis_for_residue(self, res_idx, aa, available_chis):
        """
        Decide which chi DOFs to expose for a residue.
        """

        available = sorted(set(available_chis), key=lambda x: (len(x), x))
        max_chi = self._CHI_CAP_BY_RESIDUE.get(aa)

        def within_cap(chi_name: str) -> bool:
            if max_chi is None:
                return True
            try:
                chi_num = int(str(chi_name).replace("chi", ""))
            except ValueError:
                return False
            return chi_num <= max_chi

        available = [c for c in available if within_cap(c)]

        if self.chi_mode == "all":
            return available

        if self.chi_mode == "chi1_only":
            return [c for c in available if c == "chi1"]

        if self.chi_mode == "selective":
            allowed = self.selective_chi_map.get(aa, ["chi1"])
            allowed = set(allowed)
            return [c for c in available if c in allowed and within_cap(c)]

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

        Phi/psi/chi/omega remain regular signed torsions. Omega is clamped only
        during coordinate rebuild; the energy function scores a separate raw
        omega-window penalty so out-of-window torsions remain visible to the
        optimizer instead of being silently remapped away.
        """
        mapped = np.asarray(angle_vector, dtype=float).copy()
        if len(mapped) != len(self.dof_map):
            logger.warning(
                "_map_angle_vector_to_physical_ranges: angle_vector length %d "
                "does not match dof_map length %d; only the first %d entries "
                "will be remapped",
                len(mapped), len(self.dof_map), min(len(mapped), len(self.dof_map)),
            )
        return mapped

    def _angle_dict_from_vector(self, angle_vector):
        angle_dict = {f"{x['res']}_{x['type']}": val for x, val in zip(self.dof_map, angle_vector)}
        for key, val in list(angle_dict.items()):
            if key.endswith("_omega"):
                angle_dict[key] = self._bounded_omega(val)
        return angle_dict

    def _raw_angle_dict_from_vector(self, angle_vector):
        return {f"{x['res']}_{x['type']}": val for x, val in zip(self.dof_map, angle_vector)}

    def _omega_window_violation(self, omega):
        """Return normalized deviation outside the signed trans omega window."""
        val = float(omega)
        if not np.isfinite(val):
            return 0.0
        # Accept both signed representations of trans peptide omega:
        # +170..+180 and -180..-170 in the optimizer angle domain.
        if self.OMEGA_MIN <= val <= np.pi:
            return 0.0
        if -np.pi <= val <= -self.OMEGA_MIN:
            return 0.0
        if 0.0 <= val < self.OMEGA_MIN:
            dev = self.OMEGA_MIN - val
        elif val > np.pi:
            dev = min(abs(val - np.pi), abs(val - self.OMEGA_MAX))
        elif -self.OMEGA_MIN < val < 0.0:
            dev = self.OMEGA_MIN + val
        else:
            dev = abs(abs(val) - np.pi)
        return float(max(dev, 0.0) / max(self.OMEGA_HALF_WIDTH, 1e-9))

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

        In Rosetta mode, this returns the actual PyRosetta pose scored
        for the supplied torsions so saved PDBs and RMSDs reflect the same
        structure Rosetta evaluated.
        """
        if self.stage_backend == "rosetta":
            self._score_stage_rosetta(angle_vector, return_terms=True)
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
            raise RuntimeError("PyRosetta is not installed, but energy_backend='rosetta' was requested.")
        if not _PYROSETTA_INIT_DONE:
            pyrosetta.init(self.rosetta_flags)
            _PYROSETTA_INIT_DONE = True
        self._rosetta_scorefxn_cen = pyrosetta.create_score_function(self.rosetta_centroid_weights)
        try:
            self._rosetta_scorefxn_fa = pyrosetta.create_score_function(self.rosetta_fullatom_weights)
        except Exception:
            self._rosetta_scorefxn_fa = pyrosetta.get_fa_scorefxn()
        self._rosetta_blank_pose = pyrosetta.pose_from_sequence(self.sequence, "fa_standard")
        self._rosetta_ready = True

    def _ensure_openmm(self):
        if self._openmm_ready:
            return
        if not _OPENMM_AVAILABLE:
            raise ImportError(
                "OpenMM is required for the 'openmm' stage-3 backend but it is "
                "not importable. Install it with `conda install -c conda-forge openmm` "
                "(pip installation is not supported — the C++ backend must be compiled "
                "for your platform by conda); if OpenMM is already installed, the import "
                "failure means the install is broken (CUDA mismatch, missing shared "
                "library, failed C++ extension load, etc.) and the original error was "
                "logged at module-import time."
            )
        self._openmm_ready = True

    def _build_rosetta_pose_from_angles(self, angle_vec):
        self._ensure_rosetta()
        pose = self._rosetta_blank_pose.clone()
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
        if self.stage_backend == "rosetta":
            # Force one final score call so _last_rosetta_pose is
            # synchronized with the optimizer's final parameter vector.
            self._score_stage_rosetta(angle_vec, return_terms=True)
            if self._last_rosetta_pose is not None and self.last_energy_terms.get("rosetta_error", 1.0) == 0.0:
                return self._pose_to_coords_labels_bonds(self._last_rosetta_pose)
        if self.stage_backend == "openmm":
            self._score_stage_openmm(angle_vec, return_terms=True)
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

    def _score_stage_rosetta(self, angle_vec, return_terms=False):
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

    def _score_stage_openmm(self, angle_vec, return_terms=False):
        terms = {"energy_backend_openmm": 1.0, "energy_backend_custom": 0.0, "energy_backend_rosetta": 0.0}
        try:
            self._ensure_openmm()
            coords, labels, _ = self.build_full_structure(angle_vec)
            input_pdb = self._build_openmm_input_pdb(coords, labels)
            try:
                with tempfile.TemporaryDirectory(prefix="qtf_openmm_") as workdir:
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
            finally:
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
        """Map circuit parameters to torsion angles.

        In ``"statevector"`` mode (default) the 2ⁿ complex amplitudes of the
        statevector each carry a phase in ``(-π, π]``.  The first
        ``total_angles`` phases are used as torsion angles after removing the
        **global phase**.

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

        In ``"sampler"`` mode a shot-based measurement is performed; the
        empirical probability distribution is accumulated into a CDF and
        mapped linearly onto ``(-π, π]``.  This mode supports real quantum
        hardware by accepting a ``backend`` instance.
        """
        param_dict = dict(zip(self.ansatz.parameters, params))
        bound_circuit = self.ansatz.assign_parameters(param_dict)

        if self.mode == "sampler":
            return self._get_angles_sampler(bound_circuit)

        # Default: statevector mode
        psi = _statevector_data(bound_circuit)
        phases = np.angle(psi)[: self.total_angles]
        # Remove global phase by subtracting the phase of |0...0⟩.  This pins
        # phases[0] — the N-terminal φ — to 0 and keeps it from being optimised.
        # That is harmless because the seed frame in build_full_structure()
        # places residue 0's backbone independently of φ₀ (see NERF setup at
        # lines 1227-1229).  Wrapping into (-π, π] follows.
        phases = (phases - np.angle(psi[0]) + np.pi) % (2 * np.pi) - np.pi
        return self._map_angle_vector_to_physical_ranges(phases)

    def _get_angles_sampler(self, bound_circuit) -> np.ndarray:
        """Shot-based angle extraction via empirical probability CDF.

        Runs a measurement on ``bound_circuit`` using ``self.backend`` (or an
        ``AerSimulator`` when no backend is provided), accumulates the shot
        counts into a probability vector, and maps the cumulative distribution
        function linearly onto ``(-π, π]`` to produce torsion angles.
        """
        from qiskit import transpile

        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:
            raise ImportError(
                "qiskit-aer is required for sampler mode. "
                "Install it with: pip install qiskit-aer"
            ) from exc

        backend = self.backend if self.backend is not None else AerSimulator()
        n_states = 2 ** self.n_qubits

        qc = bound_circuit.copy()
        qc.measure_all()
        tqc = transpile(qc, backend)
        counts = backend.run(tqc, shots=self.shots).result().get_counts()

        pvec = np.zeros(n_states, dtype=float)
        total = sum(counts.values())
        for bitstring, c in counts.items():
            bs = bitstring.replace(" ", "")[::-1]
            idx = int(bs, 2)
            pvec[idx] += c / total

        # CDF over the linear basis-state index (0 … 2^n-1) maps each
        # cumulative probability to a torsion angle in (-π, π].  The
        # assignment of basis states to torsion-angle slots is arbitrary
        # and has no physical significance — it is just a deterministic
        # way to extract n_angles numbers from the shot distribution.
        cdf = np.cumsum(pvec)
        angles = 2.0 * np.pi * cdf - np.pi
        return self._map_angle_vector_to_physical_ranges(angles[: self.total_angles])

    @staticmethod
    def _build_brickwork_ansatz(n_qubits: int, reps: int):
        """Build a custom brickwork (brick-layer) ansatz circuit.

        Structure per rep:

        - Layer 1: ``Ry`` + ``Rz`` on every qubit.
        - Layer 2: ``CX`` on even pairs (0–1, 2–3, 4–5, …).
        - Layer 3: ``CX`` on odd pairs (1–2, 3–4, 5–6, …).

        This gives full nearest-neighbour connectivity in two CX layers per
        rep — lower serial depth than circular entanglement and therefore
        less noise on current hardware.

        Parameters
        ----------
        n_qubits:
            Number of qubits in the circuit.
        reps:
            Number of repetitions of the rotation + entanglement block.

        Returns
        -------
        qiskit.circuit.QuantumCircuit
            Parameterised circuit with ``2 * n_qubits * (reps + 1)``
            free parameters.
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit import ParameterVector

        n_params_total = 2 * n_qubits * (reps + 1)
        params = ParameterVector("θ", n_params_total)
        qc = QuantumCircuit(n_qubits)
        p_idx = 0

        for _ in range(reps):
            # Single-qubit rotation layer
            for q in range(n_qubits):
                qc.ry(params[p_idx], q); p_idx += 1
                qc.rz(params[p_idx], q); p_idx += 1
            # Even-pair entanglement: (0,1), (2,3), …
            for q in range(0, n_qubits - 1, 2):
                qc.cx(q, q + 1)
            # Odd-pair entanglement: (1,2), (3,4), …
            for q in range(1, n_qubits - 1, 2):
                qc.cx(q, q + 1)

        # Final rotation layer (no entanglement after)
        for q in range(n_qubits):
            qc.ry(params[p_idx], q); p_idx += 1
            qc.rz(params[p_idx], q); p_idx += 1

        return qc

    def _nerf_step(self, a, b, c, bond_len, bond_angle, torsion):
        """
        Natural Extension Reference Frame (NERF)
        This is the standard math for placing atom D given atoms A, B, C.

        Inputs:
        - a, b, c: Coordinates of previous 3 atoms.
        - bond_len: Distance c -> d
        - bond_angle: Angle b-c-d
        - torsion: Dihedral angle a-b-c-d
        """
        bc = c - b; bc_u = bc / (np.linalg.norm(bc) + 1e-9)
        ab = b - a; n = np.cross(ab, bc_u); n_u = n / (np.linalg.norm(n) + 1e-9)
        bx_n = np.cross(n_u, bc_u)

        # Construct rotation matrix column-wise
        M = np.column_stack((bc_u, bx_n, n_u))

        theta_supp = np.pi - bond_angle
        d = np.array([bond_len * np.cos(theta_supp), bond_len * np.cos(torsion) * np.sin(theta_supp), bond_len * np.sin(torsion) * np.sin(theta_supp)])

        return c + (M @ d)
    def build_full_structure(self, angle_vector):
        """
        Constructs the full 3D Cartesian coordinates of the protein from torsions.

        SIDE_CHAIN_TOPO entries are interpreted as:
            (new_atom_name, parent_atom_name, bond_length_A, bond_angle_rad, torsion_spec)

        This version keeps the explicit-parent NERF rebuild for ordinary atoms,
        but uses rigid planar templates for aromatic/ring sidechains (F/Y/W/H).
        The ring template is attached at CB--CG and oriented by the local backbone
        frame plus the available chi2-like angle. This avoids sequentially walking
        around rings with NERF, which can accumulate closure errors and produce
        distorted aromatic bonds/clashes.
        """
        coords = []
        labels = []
        bonds = []

        angle_dict = self._angle_dict_from_vector(angle_vector)

        def add_atom(res_id, atom_name, elem, pos, bonded_to=None):
            idx = len(coords)
            coords.append(np.asarray(pos, dtype=float))
            labels.append((int(res_id), str(atom_name), str(elem)))
            if bonded_to is not None and bonded_to >= 0:
                bonds.append((int(bonded_to), idx))
            return idx

        def unit(v):
            v = np.asarray(v, dtype=float)
            n = np.linalg.norm(v)
            if n < 1e-9:
                return np.zeros_like(v, dtype=float)
            return v / n

        def rotate_about_axis(v, axis, theta):
            """Rodrigues rotation of vector v around unit axis by theta radians."""
            axis = unit(axis)
            v = np.asarray(v, dtype=float)
            return (
                v * np.cos(theta)
                + np.cross(axis, v) * np.sin(theta)
                + axis * np.dot(axis, v) * (1.0 - np.cos(theta))
            )

        def infer_elem(atom_name):
            return self._infer_element_from_atom_name(atom_name)

        # --- 1. INITIALIZE BACKBONE START ---
        add_atom(0, 'N', 'N', np.array([0.0, 0.0, 0.0]))
        add_atom(0, 'CA', 'C', np.array([1.46, 0.0, 0.0]), bonded_to=0)
        add_atom(
            0,
            'C',
            'C',
            np.array([
                1.46 - 1.51 * np.cos(self.BB_ANGLE_N_CA_C),
                1.51 * np.sin(self.BB_ANGLE_N_CA_C),
                0.0,
            ]),
            bonded_to=1,
        )

        def place_rigid_aromatic_template(i, aa, atom_idx):
            """
            Place aromatic/ring heavy atoms as rigid planar fragments.

            Assumes CB and CG have already been built. Coordinates are template
            coordinates in a plane, with CG at (0,0) and the ring extending in
            the +x direction away from CB. The local x-axis is CB->CG. The local
            y-axis is chosen from the CA-CB-CG plane and optionally rotated about
            x by the chi2-like torsion so the ring can flip around the CB-CG bond.
            """
            if 'CB' not in atom_idx or 'CG' not in atom_idx:
                return

            idx_CB = atom_idx['CB']
            idx_CG = atom_idx['CG']
            idx_CA = atom_idx.get('CA', -1)
            cg = coords[idx_CG]
            cb = coords[idx_CB]
            x_axis = unit(cg - cb)
            if np.linalg.norm(x_axis) < 1e-9:
                return

            # Base normal from CA-CB-CG. Fallback to backbone plane if collinear.
            if idx_CA >= 0:
                normal0 = unit(np.cross(coords[idx_CA] - cb, cg - cb))
            else:
                normal0 = np.zeros(3)
            if np.linalg.norm(normal0) < 1e-6:
                idx_N = atom_idx.get('N', -1)
                idx_C = atom_idx.get('C', -1)
                if idx_N >= 0 and idx_C >= 0:
                    normal0 = unit(np.cross(coords[idx_N] - coords[idx_CA], coords[idx_C] - coords[idx_CA]))
            if np.linalg.norm(normal0) < 1e-6:
                # Last resort: choose any vector not parallel to x.
                trial = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(trial, x_axis)) > 0.9:
                    trial = np.array([0.0, 1.0, 0.0])
                normal0 = unit(np.cross(x_axis, trial))

            # Use chi2 as ring rotation around CB-CG when present. This is an
            # approximate mapping: current QTF topology uses CG placement and ring
            # orientation differently from standard force-field internal coords,
            # but this preserves a rotatable aromatic plane without ring walking.
            chi2 = float(angle_dict.get(f"{i}_chi2", 0.0)) + np.pi
            normal = unit(rotate_about_axis(normal0, x_axis, chi2))
            y_axis = unit(np.cross(normal, x_axis))
            if np.linalg.norm(y_axis) < 1e-6:
                return

            def xyz(x, y):
                return cg + float(x) * x_axis + float(y) * y_axis

            def add_template_atom(name, xy, parent):
                if name in atom_idx:
                    return atom_idx[name]
                parent_idx = atom_idx.get(parent, idx_CG)
                new_idx = add_atom(i, name, infer_elem(name), xyz(*xy), bonded_to=parent_idx)
                atom_idx[name] = new_idx
                return new_idx

            # Approximate ideal planar templates. Distances are chosen to preserve
            # local covalent geometry much better than sequential NERF ring closure.
            if aa in ('F', 'Y'):
                b = 1.39
                h = np.sqrt(3.0) * 0.5 * b
                template = {
                    'CD1': (0.5*b,  h),
                    'CD2': (0.5*b, -h),
                    'CE1': (1.5*b,  h),
                    'CE2': (1.5*b, -h),
                    'CZ':  (2.0*b,  0.0),
                }
                parent = {'CD1': 'CG', 'CD2': 'CG', 'CE1': 'CD1', 'CE2': 'CD2', 'CZ': 'CE1'}
                for name in ('CD1', 'CD2', 'CE1', 'CE2', 'CZ'):
                    add_template_atom(name, template[name], parent[name])
                # Add missing CZ-CE2 bond for the ring graph.
                if 'CZ' in atom_idx and 'CE2' in atom_idx:
                    bonds.append((atom_idx['CE2'], atom_idx['CZ']))
                if aa == 'Y':
                    # Phenolic oxygen extends para from CG through CZ.
                    idx_OH = add_template_atom('OH', (2.0*b + 1.37, 0.0), 'CZ')
                    # Optional polar H retained internally, omitted from saved heavy PDBs.
                    if 'HH' in [d[0] for d in self.SIDE_CHAIN_TOPO.get('Y', [])] and 'HH' not in atom_idx:
                        atom_idx['HH'] = add_atom(i, 'HH', 'H', xyz(2.0*b + 1.37 + 0.96, 0.0), bonded_to=idx_OH)
                return

            if aa == 'H':
                # Rough imidazole template, planar and ring-closed.
                template = {
                    'ND1': (0.80,  1.15),
                    'CE1': (2.10,  0.65),
                    'NE2': (2.10, -0.65),
                    'CD2': (0.80, -1.15),
                }
                parent = {'ND1': 'CG', 'CE1': 'ND1', 'NE2': 'CE1', 'CD2': 'CG'}
                for name in ('ND1', 'CE1', 'NE2', 'CD2'):
                    add_template_atom(name, template[name], parent[name])
                if 'CD2' in atom_idx and 'NE2' in atom_idx:
                    bonds.append((atom_idx['CD2'], atom_idx['NE2']))
                if 'HE2' not in atom_idx:
                    atom_idx['HE2'] = add_atom(i, 'HE2', 'H', xyz(2.35, -1.15), bonded_to=atom_idx.get('NE2', idx_CG))
                return

            if aa == 'W':
                # Approximate ideal indole template: five-member ring fused to
                # benzene, expressed in the planar CG-attached frame.
                template = {
                    'CD1': (0.82,  1.06),
                    'NE1': (2.13,  0.65),
                    'CE2': (2.16, -0.73),
                    'CD2': (0.83, -1.16),
                    'CE3': (0.61, -2.54),
                    'CZ3': (1.67, -3.42),
                    'CH2': (2.99, -2.95),
                    'CZ2': (3.23, -1.60),
                }
                parent = {
                    'CD1': 'CG', 'NE1': 'CD1', 'CE2': 'NE1', 'CD2': 'CG',
                    'CE3': 'CD2', 'CZ3': 'CE3', 'CH2': 'CZ3', 'CZ2': 'CH2'
                }
                for name in ('CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ3', 'CH2', 'CZ2'):
                    add_template_atom(name, template[name], parent[name])
                # Fused-ring closure bonds.
                for a, bname in (('CD2', 'CE2'), ('CE2', 'CZ2')):
                    if a in atom_idx and bname in atom_idx:
                        bonds.append((atom_idx[a], atom_idx[bname]))
                if 'HE1' not in atom_idx:
                    atom_idx['HE1'] = add_atom(i, 'HE1', 'H', xyz(2.25, 1.55), bonded_to=atom_idx.get('NE1', idx_CG))
                return

        for i in range(self.n_residues):
            # Per-residue atom index map for the atoms that currently exist.
            atom_idx = {}
            for k, (rid, aname, _elem) in enumerate(labels):
                if int(rid) == i:
                    atom_idx[str(aname)] = k

            idx_N = atom_idx.get('N', -1)
            idx_CA = atom_idx.get('CA', -1)
            idx_C = atom_idx.get('C', -1)

            aa = self.sequence[i]
            topo = self.SIDE_CHAIN_TOPO.get(aa, self.SIDE_CHAIN_TOPO['DEFAULT'])
            parent_map = {str(atom_def[0]): str(atom_def[1]) for atom_def in topo}
            parent_map.update({'N': None, 'CA': 'N', 'C': 'CA', 'O': 'C'})

            def get_idx(name):
                return atom_idx.get(str(name), -1)

            def parent_of(name):
                return parent_map.get(str(name))

            def choose_refs(parent_name):
                """Return indices (a,b,c) for NERF placement of D attached to C=parent."""
                c_idx = get_idx(parent_name)
                if c_idx < 0:
                    return None

                gp = parent_of(parent_name)
                if gp is None:
                    if parent_name == 'N' and idx_CA >= 0 and idx_C >= 0:
                        return idx_C, idx_CA, c_idx
                    return None

                b_idx = get_idx(gp)
                if b_idx < 0:
                    return None

                ggp = parent_of(gp)
                if ggp is None:
                    if gp == 'N' and idx_C >= 0:
                        a_idx = idx_C
                    else:
                        a_idx = idx_N if idx_N >= 0 and idx_N != b_idx else idx_CA
                else:
                    a_idx = get_idx(ggp)

                if a_idx is None or a_idx < 0 or a_idx == b_idx or a_idx == c_idx:
                    if gp == 'CA' and idx_N >= 0:
                        a_idx = idx_N
                    elif gp == 'CB' and idx_CA >= 0:
                        a_idx = idx_CA
                    elif idx_N >= 0 and idx_N not in (b_idx, c_idx):
                        a_idx = idx_N
                    elif idx_CA >= 0 and idx_CA not in (b_idx, c_idx):
                        a_idx = idx_CA
                    else:
                        return None
                return int(a_idx), int(b_idx), int(c_idx)

            # --- 2. SIDECHAIN ---
            aromatic_handled = aa in ('F', 'Y', 'W', 'H')
            aromatic_core_atoms = {'CD1', 'CD2', 'CE1', 'CE2', 'CE3', 'CZ', 'CZ2', 'CZ3', 'CH2', 'ND1', 'NE1', 'NE2', 'HE1', 'HE2', 'OH', 'HH'}

            for atom_def in topo:
                name, parent_name, b_len, b_ang, tor_def = atom_def
                name = str(name)
                parent_name = str(parent_name)
                elem = infer_elem(name)

                if name in atom_idx:
                    continue
                # For aromatic residues, build only CB/CG by NERF and let the
                # rigid template place the ring atoms. This avoids ring walking.
                if aromatic_handled and name in aromatic_core_atoms:
                    continue

                # Determine torsion value. Branches share the same sampled chi but
                # use a fixed phase offset so both branches are distinct.
                if isinstance(tor_def, str) and 'chi' in tor_def:
                    chi_key = tor_def.replace('_branch', '')
                    t_val = angle_dict.get(f"{i}_{chi_key}", 0.0)
                    if 'branch' in tor_def:
                        t_val += 2.09
                else:
                    t_val = float(tor_def)

                # CB placement is tetrahedral from the N-CA-C backbone frame.
                if name == 'CB' and parent_name == 'CA' and idx_N >= 0 and idx_CA >= 0 and idx_C >= 0:
                    u_nc = unit(coords[idx_N] - coords[idx_CA])
                    u_cc = unit(coords[idx_C] - coords[idx_CA])
                    n_plane = unit(np.cross(u_nc, u_cc))
                    u_mid = unit(-(u_nc + u_cc))
                    p_new = coords[idx_CA] + (float(b_len) * (np.cos(0.9)*u_mid + np.sin(0.9)*n_plane))
                    new_idx = add_atom(i, name, elem, p_new, bonded_to=idx_CA)
                    atom_idx[name] = new_idx
                    continue

                refs = choose_refs(parent_name)
                if refs is None:
                    c_idx = get_idx(parent_name)
                    if c_idx < 0:
                        c_idx = len(coords) - 1
                    a_idx = idx_N if idx_N >= 0 else max(0, c_idx - 2)
                    b_idx = idx_CA if idx_CA >= 0 else max(0, c_idx - 1)
                    refs = (a_idx, b_idx, c_idx)

                a_idx, b_idx, c_idx = refs
                p_new = self._nerf_step(coords[a_idx], coords[b_idx], coords[c_idx], float(b_len), float(b_ang), float(t_val))
                new_idx = add_atom(i, name, elem, p_new, bonded_to=c_idx)
                atom_idx[name] = new_idx

            if aromatic_handled:
                place_rigid_aromatic_template(i, aa, atom_idx)

            # --- 3. BACKBONE OXYGEN ---
            # Nonterminal carbonyl oxygens are placed after the next peptide N is
            # known, so they can sit in the CA-C-N peptide plane. The terminal O
            # uses the local frame fallback.
            if i == self.n_residues - 1 and idx_N >= 0 and idx_CA >= 0 and idx_C >= 0 and 'O' not in atom_idx:
                psi_for_oxygen = angle_dict.get(f"{i}_psi", -0.5)
                p_O = self._nerf_step(coords[idx_N], coords[idx_CA], coords[idx_C], 1.23, 2.1, psi_for_oxygen + np.pi)
                add_atom(i, 'O', 'O', p_O, bonded_to=idx_C)

            # --- 4. NEXT RESIDUE BACKBONE ---
            if i < self.n_residues - 1:
                idx_N = get_idx('N')
                idx_CA = get_idx('CA')
                idx_C = get_idx('C')

                psi = angle_dict.get(f"{i}_psi", -0.5)
                p_next_N = self._nerf_step(coords[idx_N], coords[idx_CA], coords[idx_C], 1.33, self.BB_ANGLE_CA_C_N, psi)
                idx_next_N = add_atom(i+1, 'N', 'N', p_next_N, bonded_to=idx_C)

                if 'O' not in atom_idx:
                    u_ca = unit(coords[idx_CA] - coords[idx_C])
                    u_n = unit(p_next_N - coords[idx_C])
                    o_dir = unit(-(u_ca + u_n))
                    if np.linalg.norm(o_dir) < 1e-6:
                        o_dir = unit(np.cross(unit(coords[idx_CA] - coords[idx_C]), unit(p_next_N - coords[idx_C])))
                    p_O = coords[idx_C] + 1.23 * o_dir
                    atom_idx['O'] = add_atom(i, 'O', 'O', p_O, bonded_to=idx_C)

                omega = angle_dict.get(f"{i}_omega", np.pi)
                if f"{i}_omega" not in angle_dict and hasattr(self, "fixed_omegas") and i < len(self.fixed_omegas):
                    omega = float(self.fixed_omegas[i])
                omega = self._bounded_omega(omega)
                p_next_CA = self._nerf_step(coords[idx_CA], coords[idx_C], p_next_N, 1.46, self.BB_ANGLE_C_N_CA, omega)
                idx_next_CA = add_atom(i+1, 'CA', 'C', p_next_CA, bonded_to=idx_next_N)

                phi = angle_dict.get(f"{i+1}_phi", -1.0)
                p_next_C = self._nerf_step(coords[idx_C], p_next_N, p_next_CA, 1.51, self.BB_ANGLE_N_CA_C, phi)
                add_atom(i+1, 'C', 'C', p_next_C, bonded_to=idx_next_CA)

        return np.array(coords), labels, bonds
    def _initialize_topology_cache(self):
        """
        Runs structure builder once to determine static properties of atoms.
        Allows vectorization of Charges, Radii, and Types.
        """
        # Build with dummy zeros to get lists and the reference bond graph.
        dummy_coords, self.static_labels, static_bonds = self.build_full_structure(np.zeros(self.total_angles))
        n_atoms = len(dummy_coords)

        # 1. Map Atom Index -> Residue Index
        self.atom_to_res = np.array([x[0] for x in self.static_labels], dtype=int)
        self.atom_names = np.array([x[1] for x in self.static_labels])
        self.atom_elems = np.array([x[2] for x in self.static_labels])

        # 2. Pre-calculate Charges (Vectorized)
        self.q_vector = np.zeros(n_atoms)
        for k, (rid, name, elem) in enumerate(self.static_labels):
            q = self.CHARGES.get(name, 0.0)
            if elem == "H" or str(name).startswith("H"):
                q = 0.0
            res_name = self.sequence[rid]

            # --- LOGIC PATCH: RESOLVE CHARGE NAME COLLISIONS ---
            # NE2 is ambiguous: It is an Amide (+0.5) in Gln, but an Amine (-0.4) in Neutral His.
            if res_name == 'H':
                # Histidine NE2 carries the old HE2 donor-H charge in the QTF
                # coarse effective model; HE2 itself remains neutral.
                if name == 'NE2': q = 0.0
                if name == 'ND1': q = -0.4

            # Apply Terminal Capping Logic (Neutralize ends usually)
            if rid == 0 or rid == self.n_residues - 1:
                if name in ['N', 'CA', 'C', 'O', 'OXT', 'H1', 'H2', 'H3', 'H']: q = 0.0
            self.q_vector[k] = q

        # 3. Pre-calculate atom-typed LJ radii / epsilons (Vectorized)
        self.lj_type_vector = np.array([
            self._assign_lj_type(rid, name, elem)
            for rid, name, elem in self.static_labels
        ])
        self.vdw_radii_vector = np.array([
            self.LJ_TYPE_PARAMS.get(t, self.LJ_TYPE_PARAMS['X'])['radius']
            for t in self.lj_type_vector
        ], dtype=float)
        self.lj_epsilon_vector = np.array([
            self.LJ_TYPE_PARAMS.get(t, self.LJ_TYPE_PARAMS['X'])['epsilon']
            for t in self.lj_type_vector
        ], dtype=float)

        # 3b. Build a bond-graph topology map so nonbonded terms can exclude
        # true 1-2 / 1-3 pairs and soften 1-4 pairs. This is more physical than
        # residue-index masking and should stop the LJ wall from over-penalizing
        # locally connected native geometry.
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

        # 4. Masks
        # Mask for heavy atoms (not H) - useful for Sterics
        self.mask_heavy = np.array([not x.startswith('H') for x in self.atom_names], dtype=bool)

        # Mask for Hydrophobic atoms (SASA)
        # NOTE: self.sequence is 1-letter codes
        hydro_res_set = set(list("AVLIMFWYPC"))  # include Tyr (Y) + Trp (W)

        self.mask_hydrophobic = np.zeros(n_atoms, dtype=bool)
        for k, (rid, name, elem) in enumerate(self.static_labels):
            aa = self.sequence[rid]

            # Only mark atoms from hydrophobic residues
            if aa not in hydro_res_set:
                continue

            # Prefer sidechain carbons (avoid backbone C/CA), and include sulfur if desired
            if name.startswith("C") and name not in ("C", "CA"):
                self.mask_hydrophobic[k] = True
            elif elem == "S":
                self.mask_hydrophobic[k] = True

        # Keep a conservative residue-separation mask for electrostatics/H-bonds.
        # VDW must remain graph-based only: same-residue and adjacent-residue
        # nonbonded sidechain contacts are exactly where many rebuild clashes
        # show up, so a residue-distance filter would hide them from LJ.
        res_diff_matrix = np.abs(self.atom_to_res[:, None] - self.atom_to_res[None, :])
        self.mask_non_bonded = (res_diff_matrix >= 2)
        self.mask_non_bonded_vdw = self.mask_nonbonded_graph
        self.mask_non_bonded_vdw_14 = self.mask_14_pairs

        # Identify indices for specific calculations to avoid string parsing in loop
        self.idx_N_atoms = np.where(self.atom_names == 'N')[0]
        self.idx_O_atoms = np.where(self.atom_names == 'O')[0]
        self.idx_SG_atoms = np.where(self.atom_names == 'SG')[0]

        self._cache_initialized = True
    def energy_function(self, params, return_terms: bool = False,
                        angle_override: np.ndarray | None = None):
        """
        THE CRITIC (Objective Function)
        Evaluates how "physically good" the protein structure is.
        Lower Energy = Better Fold.

        If return_terms=True, a per-term decomposition is stored in:
            self.last_energy_terms (dict)
        """
        if not self._cache_initialized:
            self._initialize_topology_cache()

        # ==========================================
        # STAGE CONTROLLER (Guided Relaxation)
        # ==========================================
        gamma = 15.0
        constraint_strength = 8.0

        # Stage 3: RELAXATION
        if self.current_stage == 3:
            gamma = 2.5
            constraint_strength = 1.5

        # 1. GENERATION: Get geometry from quantum parameters. Non-custom
        # backends are the selected objective for every stage, not only final
        # relaxation.
        angle_vec = self._get_angles(params) if angle_override is None else angle_override
        if not np.isfinite(angle_vec).all():
            n_bad = int(np.count_nonzero(~np.isfinite(angle_vec)))
            penalty = 1e6 + 1e3 * n_bad
            if return_terms:
                self.last_energy_terms = {"non_finite_penalty": float(penalty), "total": float(penalty)}
            return float(penalty)
        if self.stage_backend == "rosetta":
            return self._score_stage_rosetta(angle_vec, return_terms=return_terms)
        if self.stage_backend == "openmm":
            return self._score_stage_openmm(angle_vec, return_terms=return_terms)

        coords, _, _ = self.build_full_structure(angle_vec)

        terms = {
            "constraint": 0.0,
            "sasa": 0.0,
            "hbond": 0.0,
            "hbond_raw": 0.0,
            "electrostatics": 0.0,
            "disulfide": 0.0,
            "vdw_repulsion": 0.0,
            "vdw_attractive": 0.0,
            "rotamer": 0.0,
            "pi_stacking": 0.0,
            "rama": 0.0,
            "geometry": 0.0,
            "omega": 0.0,
            "omega_window_penalty": 0.0,
            "hard_clash": 0.0,
            "adjacent_heavy_sterics": 0.0,
            "adjacent_backbone_sterics": 0.0,
        }
        total_energy = 0.0

        def add_term(name: str, value: float):
            nonlocal total_energy
            v = float(value)
            terms[name] += v
            total_energy += v

        # --- VECTORIZED DISTANCE MATRIX ---
        diffs = coords[:, None, :] - coords[None, :, :]
        D = np.sqrt(np.sum(diffs**2, axis=-1)) + 1e-9

        # defaults so diagnostics always exist
        neighbor_counts = np.array([0.0], dtype=float)
        burial_fractions = np.array([0.0], dtype=float)

        # 0. END-TO-END BIAS (optional, length-aware, weaker, with slack)
        ca_indices = [i for i, lbl in enumerate(self.static_labels) if lbl[1] == 'CA']
        dist_ends = 0.0
        target_e2e = 0.0
        slack_e2e = 0.0
        if len(ca_indices) >= 2:
            start_ca = coords[ca_indices[0]]
            end_ca = coords[ca_indices[-1]]
            dist_ends = float(np.linalg.norm(start_ca - end_ca))

            # Mild sequence-length-aware prior. It can be disabled with
            # QTF_USE_E2E_CONSTRAINT=0 while preserving diagnostics.
            target_e2e = float(4.5 + 0.40 * max(0, self.n_residues - 5))
            slack_e2e = float(1.5 + 0.05 * self.n_residues)
            if self.use_e2e_constraint:
                deviation = max(0.0, abs(dist_ends - target_e2e) - slack_e2e)
                e_constraint = self.e2e_scale * constraint_strength * (deviation ** 2)
                add_term("constraint", e_constraint)

        # 1. IMPLICIT SOLVENT (SASA)
        if np.sum(self.mask_hydrophobic) > 0:
            hydro_dists = D[self.mask_hydrophobic, :]
            weights = 1.0 / (1.0 + np.exp(1.0 * (hydro_dists - 6.0)))
            neighbor_counts = np.sum(weights, axis=1) - 1.0
            burial_fractions = neighbor_counts / 35.0
            burial_fractions = np.clip(burial_fractions, 0.0, 1.0)
            exposed_area = 30.0 * (1.0 - burial_fractions)
            SASA_SCALE = float(os.getenv("QTF_SASA_SCALE", "0.7"))
            e_sasa = SASA_SCALE*np.sum(gamma * exposed_area)
            add_term("sasa", e_sasa)


        # 2. EXPLICIT H-BONDING
        HBOND_SCALE = float(os.getenv("QTF_HBOND_SCALE", "0.75"))

        e_hbond = 0.0
        for i_n in self.idx_N_atoms:
            res_d = self.atom_to_res[i_n]
            idx_ca = i_n + 1
            idx_prev_c = i_n - 2

            if idx_prev_c < 0 or self.atom_names[idx_prev_c] != 'C':
                pos_h = coords[i_n] + np.array([0,0,1.0]); pos_n = coords[i_n]
            else:
                p_c = coords[idx_prev_c]; p_n = coords[i_n]; p_ca = coords[idx_ca]
                v_nc = p_c - p_n; v_nc /= np.linalg.norm(v_nc)
                v_nca = p_ca - p_n; v_nca /= np.linalg.norm(v_nca)
                v_h = -(v_nc + v_nca); v_h /= np.linalg.norm(v_h)
                pos_h = p_n + v_h * 1.01; pos_n = p_n

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

            v_hn = pos_n - pos_h; v_hn /= np.linalg.norm(v_hn)
            v_ho = final_o_coords - pos_h
            norms = np.linalg.norm(v_ho, axis=1)[:, None]
            v_ho /= norms
            angle_cos = np.dot(v_ho, v_hn)
            ang_mask = angle_cos < -0.4

            radial_term = np.exp(-(final_d_ho - 2.0)**2 / 0.5)
            angular_term = (np.abs(angle_cos) - 0.4) * 2.0
            term = -50.0 * radial_term * angular_term * ang_mask
            e_hbond += np.sum(term)

        e_hbond_scaled = HBOND_SCALE * e_hbond
        terms["hbond_raw"] = float(e_hbond)
        add_term("hbond", e_hbond_scaled)

        # 3. ELECTROSTATICS
        Q_mat = np.outer(self.q_vector, self.q_vector)
        elec_mask = np.triu(self.mask_non_bonded, k=1) & (np.abs(Q_mat) > 0.0001)
        if np.any(elec_mask):
            add_term("electrostatics", self._electrostatic_energy(D))

        # 3b. DISULFIDE
        e_disulf = 0.0
        if len(self.idx_SG_atoms) > 1:
            sg_dists = D[np.ix_(self.idx_SG_atoms, self.idx_SG_atoms)]
            sg_mask = np.triu(np.ones_like(sg_dists, dtype=bool), k=1)
            valid_dists = sg_dists[sg_mask]

            bond_strengths = np.exp(-(valid_dists - 2.05)**2 / 0.5)
            active_bonds = (valid_dists < 3.0)
            e_disulf -= np.sum(25.0 * bond_strengths * active_bonds)

            full_strengths = np.exp(-(sg_dists - 2.05)**2 / 0.5) * (sg_dists < 3.0)
            np.fill_diagonal(full_strengths, 0.0)
            saturation = np.sum(full_strengths, axis=1)
            overload = saturation - 1.0
            penalty_mask = overload > 0.1
            if np.any(penalty_mask):
                e_disulf += np.sum(40.0 * (overload[penalty_mask])**2)

        add_term("disulfide", e_disulf)

        # 4. NONBONDED VDW
        # Next test patch:
        #   - exclude true 1-2 / 1-3 pairs from the bond graph
        #   - treat true 1-4 pairs with reduced weight
        #   - slightly tighten the effective contact distance to avoid overcounting
        #     clashes from coarse rebuilt geometry
        #   - soften the positive LJ wall so a few near-contacts do not dominate
        Sigma_mat = self.vdw_radii_vector[:, None] + self.vdw_radii_vector[None, :]
        Epsilon_mat = np.sqrt(self.lj_epsilon_vector[:, None] * self.lj_epsilon_vector[None, :])
        heavy_mat = self.mask_heavy[:, None] & self.mask_heavy[None, :]
        vdw_mask = np.triu(self.mask_non_bonded_vdw & heavy_mat, k=1)
        vdw_14_mask = np.triu(self.mask_non_bonded_vdw_14 & heavy_mat, k=1)

        # Tunable scales retained so output tables stay comparable to prior runs.
        VDW_REP_SCALE = float(os.getenv("QTF_VDW_REP_SCALE", "0.01"))
        VDW_ATTR_SCALE = float(os.getenv("QTF_VDW_ATTR_SCALE", "0.1"))

        # Internal stabilization defaults for the next diagnostic test.
        LJ_CONTACT_SCALE = 0.95
        LJ_14_SCALE = 0.35
        LJ_REP_CLIP = 25.0
        LJ_ATT_CLIP = -2.5

        def _add_lj_from_mask(mask, pair_scale):
            if not np.any(mask):
                return

            r_vdw = np.maximum(D[mask], 1.2)
            contact_dist = LJ_CONTACT_SCALE * Sigma_mat[mask]
            eps_ij = Epsilon_mat[mask]

            # Interpret the contact distance as the LJ minimum distance r_min,
            # then convert to sigma via r_min = 2^(1/6) * sigma.
            sigma_ij = contact_dist / (2.0 ** (1.0 / 6.0))
            sr6 = (sigma_ij / r_vdw) ** 6
            lj = 4.0 * eps_ij * (sr6**2 - sr6)

            # Soft-cap the repulsive branch. The unmodified wall is too brittle for
            # the current rebuilt geometry and swamps the total score.
            rep_raw = np.clip(lj, 0.0, None)
            rep_term = rep_raw.copy()
            high_rep = rep_raw > LJ_REP_CLIP
            if np.any(high_rep):
                rep_term[high_rep] = LJ_REP_CLIP + np.log1p(rep_raw[high_rep] - LJ_REP_CLIP)
            att_term = np.clip(lj, LJ_ATT_CLIP, 0.0)

            if np.any(rep_term > 0.0):
                add_term("vdw_repulsion", np.sum(pair_scale * VDW_REP_SCALE * rep_term))
            if np.any(att_term < 0.0):
                add_term("vdw_attractive", np.sum(pair_scale * VDW_ATTR_SCALE * att_term))

        _add_lj_from_mask(vdw_mask, 1.0)
        _add_lj_from_mask(vdw_14_mask, LJ_14_SCALE)

        # 4b. HARD CLASH WALL
        # Keep the smoothed LJ term usable for normal contacts, but add a steep
        # guardrail for physically impossible heavy-atom overlaps. This prevents
        # beam search from exploiting clipped LJ repulsion by preserving structures
        # with sub-angstrom nonbonded contacts.
        HARD_CLASH_MIN_A = float(os.getenv("QTF_HARD_CLASH_MIN_A", "1.20"))
        HARD_CLASH_SCALE = float(os.getenv("QTF_HARD_CLASH_SCALE", "5000.0"))
        HARD_CLASH_POWER = float(os.getenv("QTF_HARD_CLASH_POWER", "4.0"))
        HARD_CLASH_14_SCALE = float(os.getenv("QTF_HARD_CLASH_14_SCALE", "0.25"))
        hard_clash_min_dist = float("nan")
        hard_clash_count = 0.0

        def _hard_clash_from_mask(mask, pair_scale):
            if not np.any(mask):
                return 0.0, float("nan"), 0
            r = D[mask]
            finite = np.isfinite(r)
            if not np.any(finite):
                return 0.0, float("nan"), 0
            r = r[finite]
            min_dist = float(np.min(r))
            shortfall = np.clip(HARD_CLASH_MIN_A - r, 0.0, None)
            active = shortfall > 0.0
            if not np.any(active):
                return 0.0, min_dist, 0
            denom = max(HARD_CLASH_MIN_A, 1e-6)
            penalties = HARD_CLASH_SCALE * pair_scale * (shortfall[active] / denom) ** HARD_CLASH_POWER
            return float(np.sum(penalties)), min_dist, int(np.sum(active))

        e_hard_full, min_full, n_full = _hard_clash_from_mask(vdw_mask, 1.0)
        e_hard_14, min_14, n_14 = _hard_clash_from_mask(vdw_14_mask, HARD_CLASH_14_SCALE)
        add_term("hard_clash", e_hard_full + e_hard_14)
        hard_mins = [x for x in (min_full, min_14) if np.isfinite(x)]
        if hard_mins:
            hard_clash_min_dist = float(min(hard_mins))
        hard_clash_count = float(n_full + n_14)

        # LOCALS
        raw_angle_dict = self._raw_angle_dict_from_vector(angle_vec)
        angle_dict = self._angle_dict_from_vector(angle_vec)

        ROTAMER_SCALE = float(os.getenv("QTF_ROTAMER_SCALE", "1.0"))
        PI_STACK_SCALE = float(os.getenv("QTF_PI_STACK_SCALE", "1.0"))

        e_rot = self._calculate_rotamer_energy(angle_dict)
        add_term("rotamer", ROTAMER_SCALE * e_rot)

        e_pi = self._calculate_aromatic_quadrupole(coords, self.static_labels, self.atom_to_res)
        add_term("pi_stacking", PI_STACK_SCALE * e_pi)

        e_rama = 0.0
        for i in range(self.n_residues):
            if f"{i}_phi" in angle_dict and f"{i}_psi" in angle_dict:
                phi = angle_dict[f"{i}_phi"]; psi = angle_dict[f"{i}_psi"]
                aa = self.sequence[i]
                d_helix = (phi - (-1.0))**2 + (psi - (-0.8))**2
                d_sheet = (phi - (-2.3))**2 + (psi - (2.4))**2

                if aa == 'G': # Glycine is flexible
                    d_helix_L = (phi - (1.0))**2 + (psi - (0.8))**2
                    d_sheet_L = (phi - (2.3))**2 + (psi - (-2.4))**2
                    dist_best = min(d_helix, d_sheet, d_helix_L, d_sheet_L)
                    e_rama += -3.0 * np.exp(-dist_best/0.6)
                else:
                    d_forbidden = (phi - (-2.0))**2 + (psi - (1.0))**2
                    term = -3.0 * np.exp(-d_helix/0.6) - 3.0 * np.exp(-d_sheet/0.6) + 5.0 * np.exp(-d_forbidden/1.0)
                    e_rama += term
        add_term("rama", e_rama)

        OMEGA_SCALE = float(os.getenv("QTF_OMEGA_SCALE", "1.0"))
        OMEGA_WINDOW_SCALE = float(os.getenv("QTF_OMEGA_WINDOW_SCALE", "25.0"))
        e_omega = 0.0
        e_omega_window = 0.0
        omega_clamped_count = 0.0
        for i in range(self.n_residues - 1):
            raw_omega = raw_angle_dict.get(f"{i}_omega", np.pi)
            violation = self._omega_window_violation(raw_omega)
            if violation > 0.0:
                omega_clamped_count += 1.0
                e_omega_window += violation ** 2
            omega = self._bounded_omega(raw_omega)
            delta = omega - self.OMEGA_CENTER
            # Mild preference for 180 degrees inside the allowed trans band.
            e_omega += (delta / self.OMEGA_HALF_WIDTH) ** 2
        add_term("omega", OMEGA_SCALE * e_omega)
        add_term("omega_window_penalty", OMEGA_WINDOW_SCALE * e_omega_window)

        e_local_sterics, local_steric_terms = self._calculate_adjacent_heavy_sterics(
            coords, self.static_labels, self.atom_to_res, return_terms=True
        )
        add_term("adjacent_heavy_sterics", e_local_sterics)
        terms["adjacent_backbone_sterics"] = float(e_local_sterics)

        e_geom, geom_subterms = self._calculate_geometry_integrity(
        coords, self.static_labels, self.atom_to_res, return_terms=True
        )
        add_term("geometry", e_geom)

        if self.tracker is not None:
            self.tracker.log(total_energy)

        if return_terms:
            self.last_energy_terms = {
            **terms,

            "energy_backend_custom": 1.0,
            "energy_backend_rosetta": 0.0,
            "energy_backend_openmm": 0.0,
            "use_e2e_constraint": 1.0 if self.use_e2e_constraint else 0.0,
            "omega_window_scale": float(OMEGA_WINDOW_SCALE),
            "omega_clamped_count": float(omega_clamped_count),

            # end-to-end diagnostics
            "e2e_distance": float(dist_ends) if len(ca_indices) >= 2 else 0.0,
            "e2e_target": float(target_e2e) if len(ca_indices) >= 2 else 0.0,
            "e2e_slack": float(slack_e2e) if len(ca_indices) >= 2 else 0.0,
            "vdw_pairs_full": float(np.sum(vdw_mask)) if "vdw_mask" in locals() else 0.0,
            "vdw_pairs_14": float(np.sum(vdw_14_mask)) if "vdw_14_mask" in locals() else 0.0,
            "hard_clash_min_dist": float(hard_clash_min_dist) if np.isfinite(hard_clash_min_dist) else 0.0,
            "hard_clash_count": float(hard_clash_count),
            "hard_clash_min_A": float(HARD_CLASH_MIN_A),
            "hard_clash_scale": float(HARD_CLASH_SCALE),
            "hard_clash_power": float(HARD_CLASH_POWER),
            "hard_clash_14_scale": float(HARD_CLASH_14_SCALE),

            # geometry diagnostics
            "geom_pro_ring": float(geom_subterms["pro_ring"]),
            "geom_chirality": float(geom_subterms["chirality"]),
            "geom_planarity": float(geom_subterms["planarity"]),
            "local_adjacent_heavy_sterics": float(local_steric_terms["adjacent_heavy_sterics"]),
            "local_backbone_sterics": float(local_steric_terms["adjacent_heavy_sterics"]),

            # burial diagnostics
            "burial_mean": float(np.mean(burial_fractions)) if np.sum(self.mask_hydrophobic) > 0 else 0.0,
            "burial_min": float(np.min(burial_fractions)) if np.sum(self.mask_hydrophobic) > 0 else 0.0,
            "burial_max": float(np.max(burial_fractions)) if np.sum(self.mask_hydrophobic) > 0 else 0.0,
            "neighbor_mean": float(np.mean(neighbor_counts)) if np.sum(self.mask_hydrophobic) > 0 else 0.0,
            "neighbor_min": float(np.min(neighbor_counts)) if np.sum(self.mask_hydrophobic) > 0 else 0.0,
            "neighbor_max": float(np.max(neighbor_counts)) if np.sum(self.mask_hydrophobic) > 0 else 0.0,

            "total": float(total_energy),
        }

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

    def _calculate_rotamer_energy(self, angle_dict):
        """
        Rotamer prior for sidechain torsions.

        chi1 is strongest and residue-aware.
        chi2+ are weaker priors that still discourage unphysical placements.
        """
        energy = 0.0

        def wrap_delta(a, b):
            return (a - b + np.pi) % (2.0 * np.pi) - np.pi

        chi_centers = [-1.0471975512, 1.0471975512, 3.1415926536]  # -60, +60, 180 deg

        for i in range(self.n_residues):
            res_name = self.sequence[i]

            # ---- chi1: strongest prior ----
            key1 = f"{i}_chi1"
            if key1 in angle_dict:
                chi = angle_dict[key1]

                if res_name in ['V', 'I', 'T']:
                    # beta-branched residues prefer trans / gauche+
                    d_trans = wrap_delta(chi, np.pi) ** 2
                    d_gplus = wrap_delta(chi, -1.0471975512) ** 2
                    energy += -3.0 * (np.exp(-d_trans / 0.5) + np.exp(-d_gplus / 0.5))

                elif res_name == 'P':
                    d_down = wrap_delta(chi, -0.5) ** 2
                    d_up = wrap_delta(chi, 0.5) ** 2
                    energy += 10.0 * min(d_down, d_up)

                elif res_name in ['W', 'F', 'Y', 'H']:
                    # aromatics: trans/gauche favored, slightly narrower
                    d_trans = wrap_delta(chi, np.pi) ** 2
                    d_gplus = wrap_delta(chi, -1.0471975512) ** 2
                    d_gminus = wrap_delta(chi, 1.0471975512) ** 2
                    energy += -2.0 * (
                        np.exp(-d_trans / 0.45)
                        + 0.8 * np.exp(-d_gplus / 0.45)
                        + 0.8 * np.exp(-d_gminus / 0.45)
                    )

                else:
                    energy += 1.0 * (1.0 + np.cos(3.0 * chi))

            # ---- chi2+ : weaker generic rotamer prior ----
            for chi_idx in (2, 3, 4, 5):
                key = f"{i}_chi{chi_idx}"
                if key not in angle_dict:
                    continue

                chi = angle_dict[key]

                # aromatic chi2 is especially important
                if chi_idx == 2 and res_name in ['W', 'F', 'Y', 'H']:
                    wells = sum(np.exp(-(wrap_delta(chi, c) ** 2) / 0.35) for c in chi_centers)
                    energy += -1.5 * wells
                else:
                    wells = sum(np.exp(-(wrap_delta(chi, c) ** 2) / 0.50) for c in chi_centers)
                    energy += -0.75 * wells

        return energy
    def _calculate_aromatic_quadrupole(self, coords, labels, atom_to_res_idx):
        """
        Calculates stacking energy between aromatic rings (Phe, Tyr, Trp).
        Uses normal vectors to detect parallel stacking.
        """
        aromatics = []
        res_indices = np.unique(atom_to_res_idx)
        for r_idx in res_indices:
            if self.sequence[r_idx] in ['F', 'Y', 'W']:
                mask = (atom_to_res_idx == r_idx)
                r_coords = coords[mask]
                r_names = self.atom_names[mask]

                ring_mask = np.isin(r_names, ['CG','CD1','CD2','CE1','CE2','CZ'])
                ring_atoms = r_coords[ring_mask]

                if len(ring_atoms) > 2:
                    centroid = np.mean(ring_atoms, axis=0)
                    v1 = ring_atoms[1] - ring_atoms[0]
                    v2 = ring_atoms[2] - ring_atoms[0]
                    normal = np.cross(v1, v2); normal /= (np.linalg.norm(normal)+1e-9)
                    aromatics.append((centroid, normal))

        energy_pi = 0.0
        n_aro = len(aromatics)
        if n_aro < 2: return 0.0

        for i in range(n_aro):
            for j in range(i+1, n_aro):
                c1, n1 = aromatics[i]; c2, n2 = aromatics[j]
                dist = np.linalg.norm(c1 - c2)
                if dist > 7.0: continue
                alignment = abs(np.dot(n1, n2))
                # T-stacking vs Parallel Stacking
                if alignment < 0.3 and 4.5 < dist < 6.0:
                     energy_pi -= 4.0 * np.exp(-(dist - 5.0)**2)
                elif alignment > 0.8 and 3.4 < dist < 4.5:
                     energy_pi -= 5.0 * np.exp(-(dist - 3.8)**2)
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
            return float(ax * ax)
        return float(2.0 * delta * ax - delta * delta)

    def _calculate_geometry_integrity(self, coords, labels, atom_to_res_idx, return_terms=False):
        """
        Penalizes physically impossible geometries.
        Optionally returns sub-terms for debugging.
        """
        energy = 0.0

        geom_terms = {
            "pro_ring": 0.0,
            "chirality": 0.0,
            "planarity": 0.0,
        }

        res_map = {}
        for k, lbl in enumerate(labels):
            r = lbl[0]
            atom = lbl[1]
            if r not in res_map:
                res_map[r] = {}
            res_map[r][atom] = k

        for r in range(self.n_residues):
            atoms = res_map.get(r, {})
            res_name = self.sequence[r]

            # PRO ring closure is handled during rebuild in the specific template
            # path when enabled. The scoring function leaves it neutral.
            if res_name == 'P' and 'CD' in atoms and 'N' in atoms:
                pass

            # Chirality check
            if 'CA' in atoms and 'N' in atoms and 'C' in atoms and 'CB' in atoms:
                ca = coords[atoms['CA']]
                n = coords[atoms['N']]
                c = coords[atoms['C']]
                cb = coords[atoms['CB']]
                volume = np.dot(np.cross(n - ca, c - ca), cb - ca)
                if volume < 1.0:
                    penalty = 50.0 * (1.0 - volume) ** 2
                    energy += penalty
                    geom_terms["chirality"] += penalty

            # Peptide planarity
            if r < self.n_residues - 1:
                next_atoms = res_map.get(r + 1, {})
                if 'C' in atoms and 'CA' in atoms and 'N' in next_atoms and 'CA' in next_atoms:
                    idx1, idx2, idx3, idx4 = atoms['CA'], atoms['C'], next_atoms['N'], next_atoms['CA']
                    p1, p2, p3, p4 = coords[idx1], coords[idx2], coords[idx3], coords[idx4]

                    b1 = p2 - p1
                    b2 = p3 - p2
                    b3 = p4 - p3

                    n1 = np.cross(b1, b2)
                    n2 = np.cross(b2, b3)

                    n1_norm = np.linalg.norm(n1)
                    n2_norm = np.linalg.norm(n2)

                    if n1_norm > 1e-8 and n2_norm > 1e-8:
                        n1 /= n1_norm
                        n2 /= n2_norm
                        parallelism = np.dot(n1, n2)

                        next_seq = self.sequence[r + 1]
                        # For peptide planarity, we care that the planes are either parallel OR anti-parallel.
                        # Both correspond to a planar peptide geometry.
                        twist_penalty = 1.0 - abs(parallelism)

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

    # ------------------------------------------------------------------
    # Best-snapshot tracker
    # ------------------------------------------------------------------

    def _build_best_k_tracker(self, k: int):
        """Return a wrapper that records the *k* lowest-energy parameter vectors.

        Usage inside ``fold()``::

            tracker = self._build_best_k_tracker(top_k_snapshots)
            obj = tracker if tracker is not None else self.energy_function
            minimize(obj, ...)
            ...
            best = tracker.best_snapshots if tracker is not None else []
        """
        if k <= 0:
            return None

        # Store (-energy, counter, params) so Python's min-heap gives us
        # (-energy) at the root = most negative = highest actual energy = WORST
        # of the K kept.  The condition `-value > _heap[0][0]` fires when the
        # new energy is lower (better) than the current worst, evicting the
        # worst and inserting the better item — correctly maintaining the K
        # lowest-energy parameter vectors.
        _heap: list[tuple[float, int, np.ndarray]] = []
        _counter: int = 0

        def _tracker(params: np.ndarray, **kwargs) -> float:
            nonlocal _counter
            value = self.energy_function(params, **kwargs)
            if not kwargs.get("return_terms"):
                if len(_heap) < k:
                    heapq.heappush(_heap, (-value, _counter, params.copy()))
                    _counter += 1
                elif -value > _heap[0][0]:
                    # New energy is better (lower) than the worst currently kept.
                    heapq.heappop(_heap)
                    heapq.heappush(_heap, (-value, _counter, params.copy()))
                    _counter += 1
            return value

        _tracker._heap = _heap
        return _tracker

    def fold(
        self,
        max_iter: int = 2000,
        initial_params: np.ndarray | None = None,
        scout_attempts: int | None = None,
        top_k_snapshots: int = 0,
    ) -> tuple[np.ndarray, list, list, LandscapeTracker, np.ndarray, float, list]:
        """Run the optimisation curriculum for the configured energy backend.

        Parameters
        ----------
        max_iter:
            Maximum number of energy evaluations allowed for **each**
            optimisation stage. Custom energy uses three stages (COBYLA
            collapse, SLSQP refine, SLSQP relax). Rosetta/OpenMM use two
            stages (COBYLA, then SLSQP).
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
        coords, labels, bonds, tracker, final_params, final_energy, best_snapshots
            The first six elements are the same as before.  The seventh
            element is a list of dicts (empty when *top_k_snapshots* <= 0)::

                [
                    {
                        "energy": float,
                        "coords": ndarray,
                        "labels": list,
                        "bonds": list,
                    },
                    ...
                ]

            sorted by ascending energy (lowest first).
        """
        logger.info("Starting quantum folding (max_iter=%d)", max_iter)
        self.tracker = LandscapeTracker()

        if initial_params is None:
            n_scout = min(64, max_iter // 10) if scout_attempts is None else scout_attempts
            logger.info("Scouting %d starting points…", n_scout)
            init_params = self.get_smart_initialization(n_attempts=n_scout)
        else:
            init_params = initial_params

        # Wrap the objective function when tracking best snapshots
        best_tracker = self._build_best_k_tracker(top_k_snapshots)
        obj_fn = best_tracker if best_tracker is not None else self.energy_function

        def _with_heartbeat(stage_name: str, objective):
            state = {
                "count": 0,
                "best": float("inf"),
                "last_logged_count": 0,
            }
            heartbeat_interval = max(50, min(200, int(max_iter) // 2 if int(max_iter) > 0 else 50))

            def _wrapped(params, **kwargs):
                value = float(objective(params, **kwargs))
                state["count"] += 1
                improved = value < state["best"]
                if improved:
                    state["best"] = value
                should_log = state["count"] == 1 or improved or (state["count"] - state["last_logged_count"]) >= heartbeat_interval
                if should_log:
                    if improved:
                        message = "  %s progress: eval=%d current=%.2f best=%.2f"
                    else:
                        message = "  %s progress: eval=%d current=%.2f best=%.2f (steady)"
                    logger.info(
                        message,
                        stage_name,
                        state["count"],
                        value,
                        state["best"],
                    )
                    state["last_logged_count"] = state["count"]
                return value

            return _wrapped

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
        res_1 = minimize(_with_heartbeat("Stage 1", obj_fn), init_params, method="COBYLA",
                         options={"maxiter": safe_maxiter, "rhobeg": 1.0})
        logger.info("  Collapse energy: %.2f", res_1.fun)

        logger.info("Stage 2: Physics Refinement (high force)…")
        self.tracker.mark_stage("Stage2")
        self.current_stage = 2
        res_2 = minimize(_with_heartbeat("Stage 2", obj_fn), res_1.x, method="SLSQP",
                         tol=1e-6, options={"maxiter": max_iter, "disp": False})
        logger.info("  Refinement energy: %.2f", res_2.fun)

        final_res = res_2
        if self.stage_backend == "custom":
            logger.info("Stage 3: Natural Relaxation (releasing constraints)…")
            self.tracker.mark_stage("Stage3")
            self.current_stage = 3
            final_res = minimize(_with_heartbeat("Stage 3", obj_fn), res_2.x, method="SLSQP",
                                 tol=1e-6, options={"maxiter": max_iter, "disp": False})
            logger.info("  Final energy: %.2f", final_res.fun)
        else:
            logger.info(
                "Skipping Stage 3 for energy_backend=%s; final energy: %.2f",
                self.stage_backend,
                final_res.fun,
            )

        # Rebuild best-K snapshots (if requested) before the final output
        # rebuild so that last_energy_terms is still valid for the final
        # result.
        best_snapshots: list[dict] = []
        if best_tracker is not None:
            for neg_val, _, params in sorted(best_tracker._heap, key=lambda x: -x[0]):
                energy = -neg_val
                s_coords, s_labels, s_bonds = self._final_output_structure_from_params(params)
                best_snapshots.append({
                    "energy": float(energy),
                    "coords": s_coords,
                    "labels": s_labels,
                    "bonds": s_bonds,
                })

        self.energy_function(final_res.x, return_terms=True)
        coords, labels, bonds = self._final_output_structure_from_params(final_res.x)
        final_energy = self.last_energy_terms.get(
            "total",
            self.last_energy_terms.get(
                "rosetta_total",
                self.last_energy_terms.get("openmm_potential_kj_mol", float(final_res.fun)),
            ),
        )
        return coords, labels, bonds, self.tracker, final_res.x, float(final_energy), best_snapshots

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
