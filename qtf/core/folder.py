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
* CHARMM22 / AMBER ff14SB / OPLS-AA partial charges (approximate).
* Bondi (1964) van der Waals radii.
* Engh & Huber (1991) bond/angle parameters.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from qiskit.circuit.library import EfficientSU2 as efficient_su2
from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit_ibm_runtime import QiskitRuntimeService
    

from qtf.core.tracker import LandscapeTracker


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


#service = QiskitRuntimeService(token="YOUR_TOKEN")
#backend = service.backend("ibm_Miami")


class QuantumBiophysicsFolder:
    """Hybrid quantum-classical protein folder."""

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

    def __init__(self, sequence: str, force_field: str = "charmm") -> None:
        """
        Parameters
        ----------
        sequence:
            Single-letter amino acid sequence (e.g. ``"MAGTWY"``).
        force_field:
            One of ``"charmm"`` (default), ``"amber"``, or ``"opls"``.
        """
        self.sequence = sequence.upper()
        self.n_residues = len(self.sequence)
        self.force_field = force_field.lower()

        logger.info("Initialising QuantumBiophysicsFolder | FF=%s | seq=%s", self.force_field.upper(), self.sequence)

        self.HYDROPHOBICITY = self._HYDROPHOBICITY
        self.VDW_RADII = self._VDW_RADII
        self.SIDE_CHAIN_TOPO = self._SIDE_CHAIN_TOPO

        self.CHARGES = self._build_charges(self.force_field)

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
            for k in sorted(chis):
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
        def build_brickwork_ansatz(n_qubits, reps):
            """
            Custom brickwork (brick-layer) ansatz.
            
            Structure per rep:
            Layer 1: Ry+Rz on ALL qubits
            Layer 2: CX on EVEN pairs  (0-1, 2-3, 4-5, ...)
            Layer 3: CX on ODD pairs   (1-2, 3-4, 5-6, ...)
            
            This gives full connectivity with fewer serial 2Q layers
            than circular entanglement — lower depth, less noise on hardware.
            
            Gate count per rep:
            - 2 * n_qubits single-qubit rotations
            - (n_qubits-1) CX gates split across 2 layers
            """
            # Total params: 2 rotations per qubit per rep + 2 final rotations
            n_params_total = 2 * n_qubits * (reps + 1)
            params = ParameterVector('θ', n_params_total)
            qc     = QuantumCircuit(n_qubits)
            p_idx  = 0

            for rep in range(reps):

                # ── Single-qubit rotation layer ────────────────────────────────
                for q in range(n_qubits):
                    qc.ry(params[p_idx],     q); p_idx += 1
                    qc.rz(params[p_idx],     q); p_idx += 1

                # ── Even layer: CX on (0,1), (2,3), (4,5), ... ────────────────
                for q in range(0, n_qubits - 1, 2):
                    qc.cx(q, q + 1)

                # ── Odd layer: CX on (1,2), (3,4), (5,6), ... ────────────────
                for q in range(1, n_qubits - 1, 2):
                    qc.cx(q, q + 1)

            # ── Final rotation layer (no entanglement after) ───────────────────
            for q in range(n_qubits):
                qc.ry(params[p_idx], q); p_idx += 1
                qc.rz(params[p_idx], q); p_idx += 1

            return qc

        self.ansatz = efficient_su2(self.n_qubits, reps=self.reps, entanglement="circular")
        #self.ansatz   = build_brickwork_ansatz(self.n_qubits, self.reps)
        self.n_params = self.ansatz.num_parameters

        self.current_stage = 1
        self._cache_initialized = False
        self._initialize_topology_cache()
        self.tracker: LandscapeTracker | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_charges(force_field: str) -> dict[str, float]:
        common: dict[str, float] = {
            "OXT": -1.0,
            "NZ": 1.0, "NH1": 0.5, "NH2": 0.5,
            "OD1": -0.5, "OD2": -0.5, "OE1": -0.5, "OE2": -0.5,
            "ND2": 0.5, "NE2": 0.5,
            "SG": -0.1, "SD": -0.1,
            "HE2": 0.4, "ND1": -0.4,
        }
        charmm: dict[str, float] = {
            "N": -0.47, "H": 0.31, "CA": 0.07, "C": 0.51, "O": -0.51,
            "OG": -0.4, "HG": 0.4, "OG1": -0.4, "HG1": 0.4, "OH": -0.4, "HH": 0.4,
            "NE1": -0.3, "HE1": 0.3,
        }
        amber: dict[str, float] = {
            "N": -0.42, "H": 0.27, "CA": 0.00, "C": 0.60, "O": -0.57,
            "OG": -0.6, "HG": 0.4, "OG1": -0.6, "HG1": 0.4, "OH": -0.5, "HH": 0.4,
            "NE1": -0.4, "HE1": 0.3,
        }
        opls: dict[str, float] = {
            "N": -0.50, "H": 0.30, "CA": 0.14, "C": 0.50, "O": -0.50,
            "OG": -0.7, "HG": 0.4, "OG1": -0.7, "HG1": 0.4, "OH": -0.7, "HH": 0.4,
            "NE1": -0.4, "HE1": 0.35,
        }
        charges = common.copy()
        if force_field == "amber":
            charges.update(amber)
        elif force_field == "opls":
            charges.update(opls)
        else:
            if force_field not in ("charmm",):
                logger.warning("Unknown force field '%s'. Defaulting to CHARMM.", force_field)
            charges.update(charmm)
        return charges
    """
    def _get_angles(self, params: np.ndarray) -> np.ndarray:
        Map circuit parameters to torsion angles via statevector phases.

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
        
        param_dict = dict(zip(self.ansatz.parameters, params))
        bound_circuit = self.ansatz.assign_parameters(param_dict)
        psi = Statevector(bound_circuit).data
        phases = np.angle(psi)[: self.total_angles]
        # Remove global phase: pin phases[0] to 0 and wrap into (-π, π].
        phases = (phases - np.angle(psi[0]) + np.pi) % (2 * np.pi) - np.pi
        return phases
        """
    def _get_angles(self, params, mode: str = "statevector", backend=None, shots: int = 4096):
        if self.ansatz is None:
            raise RuntimeError("Qiskit not available.")

        # Bind parameters to circuit
        param_dict    = dict(zip(self.ansatz.parameters, params))
        bound_circuit = self.ansatz.assign_parameters(param_dict)

        # ── MODE 1: STATEVECTOR ────────────────────────────────────────────────
        if mode == "statevector":
            sv     = Statevector(bound_circuit).data
            angles = np.angle(sv)                        # complex amplitudes → angles in [-π, π]
            return angles[:self.total_angles]            # trim to what we need

        # ── MODE 2: SHOT-BASED ─────────────────────────────────────────────────
        if mode == "sampler":
            if backend is None:
                backend = AerSimulator()                 # default noiseless shot sim
            n        = self.n_qubits
            n_states = 2 ** n
            # Measure in Z basis only
            qc = bound_circuit.copy()
            qc.measure_all()
            tqc    = transpile(qc, backend)
            counts = backend.run(tqc, shots=shots).result().get_counts()

            # Counts → probability vector of length 2^n
            pvec  = np.zeros(n_states, dtype=float)
            total = sum(counts.values())
            for bitstring, c in counts.items():
                bs       = bitstring.replace(" ", "")[::-1]   # qubit-0 = LSB
                idx      = int(bs, 2)
                pvec[idx] += c / total

            # CDF of prob vector → mapped to [-π, π]
            cdf    = np.cumsum(pvec)                     # length 2^n, range [0, 1]
            angles = 2.0 * np.pi * cdf - np.pi          # → [-π, π]

            return angles[:self.total_angles]            # trim to what we need
        raise ValueError(f"Unknown mode: '{mode}'. Use 'statevector' or 'sampler'.")

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
                name, elem, b_len, b_ang, tor_def = atom_def
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
                p_next_CA = self._nerf_step(coords[idx_CA], coords[idx_C], p_next_N, 1.46, 2.1, omega)
                ca_idx = _append(i + 1, "CA", "C", p_next_CA)
                bonds.append((n_idx, ca_idx))

                phi = angle_dict.get(f"{i + 1}_phi", -1.0)
                p_next_C = self._nerf_step(coords[idx_C], p_next_N, p_next_CA, 1.51, 1.9, phi)
                c_idx = _append(i + 1, "C", "C", p_next_C)
                bonds.append((ca_idx, c_idx))

        return np.array(coords), labels, bonds

    def _initialize_topology_cache(self) -> None:
        """Pre-compute static atom properties for vectorised energy evaluation.

        A single call to ``build_full_structure`` is made here to harvest the
        atom label list (``static_labels``).  The resulting coordinates are
        **discarded** — only the labels, element symbols, and residue-index
        mapping are retained for use by the energy function.

        The seed torsion vector is ``_TOPOLOGY_SEED_ANGLE`` (0.1 rad) rather
        than all-zeros.  All-zero torsions place every successive backbone
        atom exactly along the same direction, making every cross product in
        ``_nerf_step`` identically zero.  Although the 1e-9 floor in the
        normalisation prevents a division-by-zero crash, the resulting
        reference frames are degenerate and the coordinates are meaningless.
        Using a small non-zero seed is costless (coordinates are unused) and
        removes this brittleness.
        """
        seed = np.full(self.total_angles, _TOPOLOGY_SEED_ANGLE)
        dummy_coords, self.static_labels, _ = self.build_full_structure(seed)
        n_atoms = len(dummy_coords)

        self.atom_to_res = np.array([x[0] for x in self.static_labels], dtype=int)
        self.atom_names = np.array([x[1] for x in self.static_labels])
        self.atom_elems = np.array([x[2] for x in self.static_labels])

        self.q_vector = np.zeros(n_atoms)
        for k, (rid, name, _) in enumerate(self.static_labels):
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

        self.vdw_radii_vector = np.array([self.VDW_RADII.get(x[2], 1.7) for x in self.static_labels])
        self.mask_heavy = np.array([not x.startswith("H") for x in self.atom_names], dtype=bool)

        hydro_res_set = {"A", "V", "L", "I", "M", "F", "W", "P", "C"}
        self.mask_hydrophobic = np.zeros(n_atoms, dtype=bool)
        for k, (rid, name, _) in enumerate(self.static_labels):
            if self.sequence[rid] in hydro_res_set and name.startswith("C"):
                self.mask_hydrophobic[k] = True

        res_diff_matrix = np.abs(self.atom_to_res[:, None] - self.atom_to_res[None, :])
        self.mask_non_bonded = res_diff_matrix >= 2

        self.idx_N_atoms = np.where(self.atom_names == "N")[0]
        self.idx_O_atoms = np.where(self.atom_names == "O")[0]
        self.idx_SG_atoms = np.where(self.atom_names == "SG")[0]
        self._cache_initialized = True

    # ------------------------------------------------------------------
    # Energy function
    # ------------------------------------------------------------------

    def energy_function(self, params: np.ndarray) -> float:
        """Evaluate physical energy of the structure encoded by *params*."""
        if not self._cache_initialized:
            self._initialize_topology_cache()

        gamma = 15.0
        constraint_strength = 50.0
        if self.current_stage == 3:
            gamma = 5.0
            constraint_strength = 5.0

        # Pick ONE of these:
        angle_vec = self._get_angles(params)                                    # statevector (default)
        # angle_vec = self._get_angles(params, mode="sampler", shots=4096)      # shot-based
        # angle_vec = self._get_angles(params, mode="sampler",                  # real hardware
        #                 backend=your_backend, shots=4096)
        coords, _, _ = self.build_full_structure(angle_vec)
        total_energy = 0.0

        diffs = coords[:, None, :] - coords[None, :, :]
        D = np.sqrt(np.sum(diffs ** 2, axis=-1)) + 1e-9

        # End-to-end bias
        ca_indices = [i for i, lbl in enumerate(self.static_labels) if lbl[1] == "CA"]
        if len(ca_indices) >= 2:
            dist_ends = np.linalg.norm(coords[ca_indices[0]] - coords[ca_indices[-1]])
            total_energy += constraint_strength * (dist_ends - 5.5) ** 2

        # Implicit solvent (SASA)
        hydro_dists = D[self.mask_hydrophobic, :]
        weights = 1.0 / (1.0 + np.exp(1.0 * (hydro_dists - 6.0)))
        neighbor_counts = np.sum(weights, axis=1) - 1.0
        burial_fractions = np.clip(neighbor_counts / 15.0, 0.0, 1.0)
        total_energy += np.sum(gamma * 30.0 * (1.0 - burial_fractions))

        # H-bonding
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
        total_energy += e_hbond

        # Electrostatics
        total_energy += self._electrostatic_energy(D)

        # Disulfide bonds
        if len(self.idx_SG_atoms) > 1:
            sg_dists = D[np.ix_(self.idx_SG_atoms, self.idx_SG_atoms)]
            sg_mask = np.triu(np.ones_like(sg_dists, dtype=bool), k=1)
            valid_dists = sg_dists[sg_mask]
            bond_strengths = np.exp(-(valid_dists - 2.05) ** 2 / 0.5)
            active_bonds = valid_dists < 3.0
            total_energy -= np.sum(25.0 * bond_strengths * active_bonds)
            full_strengths = np.exp(-(sg_dists - 2.05) ** 2 / 0.5) * (sg_dists < 3.0)
            np.fill_diagonal(full_strengths, 0.0)
            saturation = np.sum(full_strengths, axis=1)
            overload = saturation - 1.0
            penalty_mask = overload > 0.1
            if np.any(penalty_mask):
                total_energy += np.sum(40.0 * overload[penalty_mask] ** 2)

        # Sterics (softened Lennard-Jones)
        Sigma_mat = self.vdw_radii_vector[:, None] + self.vdw_radii_vector[None, :]
        heavy_mat = self.mask_heavy[:, None] & self.mask_heavy[None, :]
        vdw_mask = np.triu(self.mask_non_bonded & heavy_mat, k=1)
        if np.any(vdw_mask):
            r_vdw = D[vdw_mask]
            s_vdw = Sigma_mat[vdw_mask]
            collision_mask = r_vdw < s_vdw
            if np.any(collision_mask):
                r_col = r_vdw[collision_mask]
                s_col = s_vdw[collision_mask]
                term = (s_col / (r_col + 0.1)) ** 12
                high_e = term > 50.0
                if np.any(high_e):
                    term[high_e] = 50.0 + np.log(term[high_e] - 49.0)
                total_energy += np.sum(0.1 * term)

        # Local terms
        angle_dict = {f"{x['res']}_{x['type']}": val for x, val in zip(self.dof_map, angle_vec)}
        total_energy += self._calculate_rotamer_energy(angle_dict)
        total_energy += self._calculate_aromatic_quadrupole(coords, self.static_labels, self.atom_to_res)

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
                    total_energy += -3.0 * np.exp(-min(d_helix, d_sheet, d_helix_L, d_sheet_L) / 0.6)
                else:
                    d_forbidden = (phi - (-2.0)) ** 2 + (psi - 1.0) ** 2
                    total_energy += -3.0 * np.exp(-d_helix / 0.6) - 3.0 * np.exp(-d_sheet / 0.6) + 5.0 * np.exp(-d_forbidden / 1.0)

        total_energy += self._calculate_geometry_integrity(coords, self.static_labels, self.atom_to_res)

        if self.tracker is not None:
            self.tracker.log(total_energy)

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
        self, coords: np.ndarray, labels: list, atom_to_res_idx: np.ndarray
    ) -> float:
        """Evaluate hard geometry constraints as soft energy penalties.

        Three classes of constraint are checked for each residue:

        1. **Pro ring closure** — the CD–N bond distance in proline must be
           close to 1.47 Å.  Deviations beyond 0.1 Å (the dead zone) are
           penalised.
        2. **Cα chirality** — the scalar triple product
           ``(N−Cα) × (C−Cα) · (Cβ−Cα)`` must be positive (L-amino acid).
           Inversion or collapse of the tetrahedron is penalised whenever
           the volume drops below 1.0 Å³.
        3. **Peptide planarity / twist** — the dihedral formed by consecutive
           Cα–C and N–Cα bonds should be close to trans (180°) for all
           residue pairs except *X*–Pro links where cis is permitted.

        Penalties (1) and (2) previously used unbounded quadratics, which
        caused gradient explosion when a single geometry was far from ideal:
        a bond stretched by 3 Å contributed O(450) kcal/mol and completely
        drowned out all other terms, stalling the optimiser.  Both are now
        replaced by a **Huber loss** with ``_HUBER_DELTA_GEOM`` = 1.0 Å
        (see ``_huber``):

        * Quadratic below δ — gradient proportional to deviation (smooth).
        * Linear above δ  — gradient capped at 2·δ (bounded influence).

        Penalty (3) is already linear and is left unchanged.
        """
        energy = 0.0
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
                    energy += 50.0 * self._huber(dev, _HUBER_DELTA_GEOM)
            if all(k in atoms for k in ("CA", "N", "C", "CB")):
                ca = coords[atoms["CA"]]
                n = coords[atoms["N"]]
                c = coords[atoms["C"]]
                cb = coords[atoms["CB"]]
                volume = np.dot(np.cross(n - ca, c - ca), cb - ca)
                if volume < 1.0:
                    energy += 50.0 * self._huber(1.0 - volume, _HUBER_DELTA_GEOM)
            if r < self.n_residues - 1:
                next_atoms = res_map.get(r + 1, {})
                if all(k in atoms for k in ("C", "CA")) and all(k in next_atoms for k in ("N", "CA")):
                    p1, p2 = coords[atoms["CA"]], coords[atoms["C"]]
                    p3, p4 = coords[next_atoms["N"]], coords[next_atoms["CA"]]
                    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
                    n1 = np.cross(b1, b2)
                    n1 /= np.linalg.norm(n1)
                    n2 = np.cross(b2, b3)
                    n2 /= np.linalg.norm(n2)
                    parallelism = np.dot(n1, n2)
                    twist = (1.0 + parallelism) if self.sequence[r + 1] == "P" else (1.0 - parallelism)
                    if twist > 0.05:
                        energy += 20.0 * twist
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
        res_1 = minimize(self.energy_function, init_params, method="COBYLA",
                         options={"maxiter": max_iter, "rhobeg": 1.0})
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

        # Statevector (default)
        angle_vec = self._get_angles(res_3.x)

        # Shot-based
        # angle_vec = self._get_angles(res_3.x, mode="sampler", shots=4096)

        # Real hardware
        # angle_vec = self._get_angles(res_3.x, mode="sampler", backend=your_backend, shots=4096)

        coords, labels, bonds = self.build_full_structure(angle_vec)
        return coords, labels, bonds, self.tracker, res_3.x, res_3.fun
