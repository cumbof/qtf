import numpy as np
import os
import matplotlib.pyplot as plt
import hashlib
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
try:
    from qiskit.circuit.library import efficient_su2
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
except ImportError:
    efficient_su2 = None
    Statevector = None
    QISKIT_AVAILABLE = False

try:
    import pyrosetta
    from pyrosetta import rosetta
    PYROSETTA_AVAILABLE = True
except ImportError:
    pyrosetta = None
    rosetta = None
    PYROSETTA_AVAILABLE = False

_PYROSETTA_INIT_DONE = False
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

# ==========================================
# 1. UTILITY: TRACKING & LOGGING
# ==========================================
class LandscapeTracker:
    def __init__(self):
        self.history = []
        self.stage_markers = []
        self.current_iter = 0

    def log(self, energy):
        self.history.append(energy)
        self.current_iter += 1

    def mark_stage(self, name):
        self.stage_markers.append((self.current_iter, name))

# ==========================================
# 2. ANALYSIS: STABILITY & CONVERGENCE (KABSCH)
# ==========================================
class StabilityAnalyzer:
    """
    Tools to evaluate if the folding results are structurally consistent.
    Implements the Kabsch algorithm for optimal superposition.
    """
    
    @staticmethod
    def kabsch_rmsd(P, Q):
        """
        Calculates the RMSD between two sets of coordinates P and Q
        after optimally rotating P to align with Q.
        """
        # 1. Centering
        P_centered = P - np.mean(P, axis=0)
        Q_centered = Q - np.mean(Q, axis=0)
        
        # 2. Covariance Matrix
        H = np.dot(P_centered.T, Q_centered)
        
        # 3. SVD
        V, S, Wt = np.linalg.svd(H)
        
        # 4. Rotation Matrix
        d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
        if d: 
            S[-1] = -S[-1]
            V[:, -1] = -V[:, -1]
        
        R = np.dot(V, Wt)
        
        # 5. Rotate and Diff
        P_rotated = np.dot(P_centered, R)
        diff = P_rotated - Q_centered
        rms = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        
        return rms, P_rotated + np.mean(Q, axis=0) # Return RMSD and Aligned Coords

    @staticmethod
    def analyze_convergence(results, top_k=5):
        """
        Calculates pairwise RMSD between the top K lowest energy structures.
        """
        # Sort by Energy (lowest first)
        sorted_results = sorted(results, key=lambda x: x['energy'])
        best_k = sorted_results[:top_k]
        n = len(best_k)
        
        if n < 2: return
        
        print(f"\n--- CONVERGENCE ANALYSIS (Top {n} Structures) ---")
        
        rmsd_matrix = np.zeros((n, n))
        
        # Print Header
        print("      ", end="")
        for i in range(n): print(f" #{i:<4}", end="")
        print("\n" + "-"*60)
        
        for i in range(n):
            print(f"Ref #{i} |", end="")
            for j in range(n):
                if i == j:
                    rmsd_matrix[i, j] = 0.0
                else:
                    # Note: Using all atoms for RMSD. Ideally filter for 'CA' only.
                    rmsd, _ = StabilityAnalyzer.kabsch_rmsd(best_k[i]['coords'], best_k[j]['coords'])
                    rmsd_matrix[i, j] = rmsd
                
                print(f" {rmsd_matrix[i, j]:.2f} ", end="")
            print(f"  (E={best_k[i]['energy']:.1f})")
            
        avg_rmsd = np.sum(rmsd_matrix) / (n*(n-1))
        print(f"\nAverage Pairwise RMSD: {avg_rmsd:.2f} Angstroms")
        
        if avg_rmsd < 2.0:
            print(">>> VERDICT: STABLE. High confidence in prediction.")
        elif avg_rmsd < 4.5:
            print(">>> VERDICT: FLEXIBLE. Core is stable, loops vary.")
        else:
            print(">>> VERDICT: UNSTABLE. No dominant basin found.")

# ==========================================
# 3. CORE: QUANTUM FOLDER
# ==========================================
class QuantumBiophysicsFolder:
    """
    A Hybrid Quantum-Classical Protein Folder.
    
    ARCHITECTURE:
    1. Quantum Actor (The Generator): 
       Uses a parameterized quantum circuit (VQE ansatz) to generate a probability distribution.
       These probabilities are mapped to physical torsion angles (phi, psi, chi).
       
    2. Classical Critic (The Energy Function):
       Constructs the 3D geometry from those angles and evaluates its physical stability
       using a custom force field (hydrophobicity, electrostatics, H-bonds, etc.).
       
    3. Optimization Loop:
       Classical optimizers (COBYLA/SLSQP) tune the quantum circuit parameters to minimize the energy.

    REFERENCES:
    1. Hydrophobicity: Kyte, J., & Doolittle, R. F. (1982). A simple method for displaying the hydropathic character of a protein.
    2. Electrostatics: CHARMM22 / OPLS-AA Force Fields (Approximated partial charges).
    3. VdW Radii: Bondi, A. (1964). Van der Waals volumes and radii.
    4. Topology: Engh, R. A., & Huber, R. (1991). Accurate bond and angle parameters for X-ray protein structure refinement.
    5. Heuristics: Manual tuning ("Quantum Velcro") for folding convergence.
    """

    def __init__(
        self,
        sequence,
        force_field='charmm',
        chi_mode='all',
        selective_chi_map=None,
        energy_backend=None,
        use_e2e_constraint=None,
        e2e_scale=None,
        rosetta_repack=None,
        rosetta_fa_min=None,
        rosetta_cen_min=None,
    ):
        """
        Initialize the folder.
        
        Args:
            sequence (str): Amino acid sequence (e.g., 'MAG').
            force_field (str): 'charmm', 'amber', or 'opls'. Determines partial charges.
        """
        self.sequence = sequence.upper()
        self.n_residues = len(sequence)
        self.force_field = force_field.lower()
        self.chi_mode = chi_mode
        self.selective_chi_map = selective_chi_map or {}
        
        print(f"--- INITIALIZING QUANTUM BIOPHYSICS FOLDER ---")
        print(f"--- FORCE FIELD: {self.force_field.upper()} ---")
        
        # --- PARAMETERS (Force Field Constants) ---

        # 1. HYDROPHOBICITY SCALE
        # Ref: Kyte, J., & Doolittle, R. F. (1982). J. Mol. Biol. 157, 105-132.
        self.HYDROPHOBICITY = {
            'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
            'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
            'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
            'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
        }
        
        # 2. PARTIAL CHARGES (Coulomb's Law)
        # We now support switching between major force field approximations.
        # While coarse-grained, these capture the distinct dipole strengths of each FF.
        
        # BASE (Shared Ions)
        common_charges = {
            'OXT': -1.0, 
            'NZ': 1.0, 'NH1': 0.5, 'NH2': 0.5, 
            'OD1': -0.5, 'OD2': -0.5, 'OE1': -0.5, 'OE2': -0.5,
            'ND2': 0.5, 'NE2': 0.5, 
            'SG': -0.1, 'SD': -0.1,
            'HE2': 0.4, 'ND1': -0.4, # His
        }

        # CHARMM22 (The original default)
        # Strong dipoles, balanced C-alpha.
        charmm_charges = {
            'N': -0.47, 'H': 0.31, 'CA': 0.07, 'C': 0.51, 'O': -0.51,
            'OG': -0.4, 'HG': 0.4, 'OG1': -0.4, 'HG1': 0.4, 'OH': -0.4, 'HH': 0.4,
            'NE1': -0.3, 'HE1': 0.3
        }
        
        # AMBER (ff94/99/14SB approx)
        # N is slightly less negative, C=O is very polar. C-alpha is often neutral.
        amber_charges = {
            'N': -0.42, 'H': 0.27, 'CA': 0.00, 'C': 0.60, 'O': -0.57,
            'OG': -0.6, 'HG': 0.4, 'OG1': -0.6, 'HG1': 0.4, 'OH': -0.5, 'HH': 0.4, # Ambient hydroxyls often stronger
            'NE1': -0.4, 'HE1': 0.3
        }

        # OPLS-AA
        # Very standardized backbone dipoles (+0.5/-0.5).
        opls_charges = {
            'N': -0.50, 'H': 0.30, 'CA': 0.14, 'C': 0.50, 'O': -0.50,
            'OG': -0.7, 'HG': 0.4, 'OG1': -0.7, 'HG1': 0.4, 'OH': -0.7, 'HH': 0.4,
            'NE1': -0.4, 'HE1': 0.35
        }

        # Select Strategy
        self.CHARGES = common_charges.copy()
        if self.force_field == 'amber':
            self.CHARGES.update(amber_charges)
        elif self.force_field == 'opls':
            self.CHARGES.update(opls_charges)
        else: # Default to CHARMM
            if self.force_field != 'charmm':
                print(f" > Warning: Unknown force field '{self.force_field}'. Defaulting to CHARMM.")
            self.CHARGES.update(charmm_charges)
        
        # 3. VAN DER WAALS RADII (Angstroms)
        # Ref: Bondi, A. (1964). J. Phys. Chem. 68, 441-451.
        self.VDW_RADII = {'H': 0.6, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}

        # 3b. Minimal atom-typed Lennard-Jones parameters.
        # These are not a complete AMBER/CHARMM parameter table; they are a
        # pragmatic intermediate model that distinguishes backbone, carbonyl,
        # aliphatic sidechain, aromatic, polar heteroatom, sulfur, and hydrogen
        # environments while preserving the existing output term names.
        # Values are in Angstrom-ish r_min/2 radii and kcal/mol-ish epsilons.
        self.LJ_TYPE_PARAMS = {
            'H':          {'radius': 1.20, 'epsilon': 0.0157},
            'H_polar':    {'radius': 1.05, 'epsilon': 0.0157},
            'C_backbone': {'radius': 1.75, 'epsilon': 0.0700},  # CA-like
            'C_carbonyl': {'radius': 1.70, 'epsilon': 0.0860},
            'C_aliphatic':{'radius': 1.90, 'epsilon': 0.1094},
            'C_aromatic': {'radius': 1.85, 'epsilon': 0.1200},
            'N_backbone': {'radius': 1.65, 'epsilon': 0.1700},
            'N_sidechain':{'radius': 1.65, 'epsilon': 0.1700},
            'O_carbonyl': {'radius': 1.60, 'epsilon': 0.2100},
            'O_hydroxyl': {'radius': 1.55, 'epsilon': 0.1700},
            'O_carboxyl': {'radius': 1.60, 'epsilon': 0.2100},
            'S_sulfur':   {'radius': 2.00, 'epsilon': 0.2500},
            'X':          {'radius': 1.75, 'epsilon': 0.1000},
        }

        # Empirical peptide backbone geometry. The previous rough 1.9/2.0/2.1
        # radian angles noticeably compressed compact loops and created false
        # terminal contacts in rebuilt full-atom PDBs.
        self.BB_ANGLE_N_CA_C = np.deg2rad(111.4)
        self.BB_ANGLE_CA_C_N = np.deg2rad(118.3)
        self.BB_ANGLE_C_N_CA = np.deg2rad(122.8)
        self.OMEGA_CENTER = np.pi
        self.OMEGA_MIN = np.deg2rad(170.0)
        self.OMEGA_MAX = np.deg2rad(190.0)
        self.OMEGA_HALF_WIDTH = 0.5 * (self.OMEGA_MAX - self.OMEGA_MIN)
        self.fixed_omegas = np.full(max(0, self.n_residues - 1), np.pi, dtype=float)
        
        # 4. SIDE CHAIN TOPOLOGY
        # Ref: Engh, R. A., & Huber, R. (1991). Acta Cryst. A47, 392-400.
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
        
        self.total_angles = len(self.dof_map)
        
        # 2. Holographic Encoding
        # We only need log2(N) qubits to represent N angles in terms of statevector capacity.
        # This makes the algorithm scalable.
        self.n_qubits = max(2, int(np.ceil(np.log2(self.total_angles))))
        
        # 3. The Ansatz (The "Actor")
        # 'efficient_su2' is a heuristic circuit often used in VQE.
        # 'reps' controls depth. More reps = more expressive = harder to train.
        self.reps = int(np.ceil(self.total_angles / self.n_qubits)) + 2

        if QISKIT_AVAILABLE:
            self.ansatz = efficient_su2(self.n_qubits, reps=self.reps, entanglement='circular')
            self.n_params = self.ansatz.num_parameters
        else:
            self.ansatz = None
            self.n_params = 1
        
        # --- OPTIONAL STAGE-3 BACKEND ---
        # Preferred control path is explicit constructor args from argparse callers.
        # Environment variables remain as backwards-compatible fallbacks for notebooks
        # and existing shell/grid scripts.
        def _as_bool(value, default=False):
            if value is None:
                return bool(default)
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() not in ("0", "false", "no", "off", "none", "")

        self.stage3_backend = (energy_backend or os.getenv("QTF_STAGE3_BACKEND", "custom")).strip().lower()
        if self.stage3_backend not in ("custom", "rosetta"):
            raise ValueError("energy_backend/stage3_backend must be 'custom' or 'rosetta'")

        self.use_e2e_constraint = _as_bool(
            use_e2e_constraint,
            os.getenv("QTF_USE_E2E_CONSTRAINT", "1").strip().lower() not in ("0", "false", "no", "off")
        )
        self.e2e_scale = float(e2e_scale if e2e_scale is not None else os.getenv("QTF_E2E_SCALE", "1.0"))

        self.rosetta_flags = os.getenv("QTF_PYROSETTA_FLAGS", "-mute all")
        self.rosetta_centroid_weights = os.getenv("QTF_ROSETTA_CEN_WTS", "cen_std")
        self.rosetta_fullatom_weights = os.getenv("QTF_ROSETTA_FA_WTS", "ref2015")
        self.rosetta_cen_weight = float(os.getenv("QTF_ROSETTA_CEN_WEIGHT", "0.35"))
        self.rosetta_fa_weight = float(os.getenv("QTF_ROSETTA_FA_WEIGHT", "1.0"))
        self.rosetta_do_centroid_min = _as_bool(rosetta_cen_min, os.getenv("QTF_ROSETTA_CEN_MIN", "0") == "1")
        self.rosetta_do_fullatom_min = _as_bool(rosetta_fa_min, os.getenv("QTF_ROSETTA_FA_MIN", "0") == "1")
        self.rosetta_do_repack = _as_bool(rosetta_repack, os.getenv("QTF_ROSETTA_REPACK", "0") == "1")
        self._rosetta_ready = False
        self._rosetta_scorefxn_cen = None
        self._rosetta_scorefxn_fa = None
        self._last_rosetta_pose = None
        self._last_rosetta_ca = None

        self.current_stage = 1
        
        # --- PRE-COMPUTE CACHE (Optimization) ---
        # We run the structure builder once with dummy data to figure out which atom is which.
        # This lets us optimize the energy function using Vectorization (NumPy) instead of slow loops.
        self._cache_initialized = False
        self._initialize_topology_cache()
        self.tracker = None  # TRACKER REFERENCE

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

    def _get_angles(self, params):
        """
        THE HOLOGRAPHIC MAPPING
        Maps Circuit Parameters (Theta) -> Torsion Angles (Phi/Psi/Chi).
        """
        if self.ansatz is None or Statevector is None:
            raise RuntimeError(
                "Qiskit is not available. Either install qiskit or override _get_angles()."
            )

        param_dict = dict(zip(self.ansatz.parameters, params))
        bound_circuit = self.ansatz.assign_parameters(param_dict)
        psi = Statevector(bound_circuit).data
        angles = np.angle(psi)[:self.total_angles]
        return self._map_angle_vector_to_physical_ranges(angles)

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

    def build_output_structure(self, angle_vector):
        """
        Build the structure that should be emitted to downstream consumers.

        In Rosetta stage-3 mode, this returns the actual PyRosetta pose scored
        for the supplied torsions so saved PDBs and RMSDs reflect the same
        structure Rosetta evaluated.
        """
        if self.current_stage == 3 and self.stage3_backend == "rosetta":
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
            res_name = self.sequence[rid]
            
            # --- LOGIC PATCH: RESOLVE CHARGE NAME COLLISIONS ---
            # NE2 is ambiguous: It is an Amide (+0.5) in Gln, but an Amine (-0.4) in Neutral His.
            if res_name == 'H':
                if name == 'NE2': q = -0.4 
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

    def _ensure_rosetta(self):
        global _PYROSETTA_INIT_DONE
        if self._rosetta_ready:
            return
        if not PYROSETTA_AVAILABLE:
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
        if self.current_stage == 3 and self.stage3_backend == "rosetta":
            # Force one final score call at res_3.x so _last_rosetta_pose is
            # synchronized with the optimizer's final parameter vector.
            self._score_stage3_rosetta(angle_vec, return_terms=True)
            if self._last_rosetta_pose is not None and self.last_energy_terms.get("rosetta_error", 1.0) == 0.0:
                return self._pose_to_coords_labels_bonds(self._last_rosetta_pose)
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
            if hasattr(rosetta.core.scoring, name):
                st = getattr(rosetta.core.scoring, name)
                try:
                    val = float(emap[st])
                except Exception:
                    val = 0.0
            out[f"{prefix}_{name}"] = val
        out[f"{prefix}_total"] = float(scorefxn(pose))
        return out

    def _score_stage3_rosetta(self, angle_vec, return_terms=False):
        terms = {"energy_backend_rosetta": 1.0, "energy_backend_custom": 0.0}
        try:
            self._ensure_rosetta()
            fa_pose = self._build_rosetta_pose_from_angles(angle_vec)

            cen_pose = fa_pose.clone()
            rosetta.protocols.simple_moves.SwitchResidueTypeSetMover("centroid").apply(cen_pose)
            cen_score = float(self._rosetta_scorefxn_cen(cen_pose))
            terms.update(self._extract_rosetta_terms(cen_pose, self._rosetta_scorefxn_cen, "cen"))

            if self.rosetta_do_centroid_min:
                mm = rosetta.core.kinematics.MoveMap()
                mm.set_bb(True)
                mm.set_chi(False)
                minmov = rosetta.protocols.minimization_packing.MinMover()
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
                packer = rosetta.protocols.minimization_packing.PackRotamersMover(self._rosetta_scorefxn_fa, task)
                packer.apply(fa_pose)

            if self.rosetta_do_fullatom_min:
                mm = rosetta.core.kinematics.MoveMap()
                mm.set_bb(False)
                mm.set_chi(True)
                minmov = rosetta.protocols.minimization_packing.MinMover()
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
            terms["rosetta_error"] = 1.0
            terms["rosetta_message_hash"] = float(abs(hash(str(exc))) % 1000000)
        self.last_energy_terms = {k: float(v) for k, v in terms.items()}
        return float(total)

    def energy_function(self, params, return_terms: bool = False):
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

        # 1. GENERATION: Get Geometry from Quantum Parameters
        angle_vec = self._get_angles(params)
        if self.current_stage == 3 and self.stage3_backend == "rosetta":
            return self._score_stage3_rosetta(angle_vec, return_terms=return_terms)

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
            "hard_clash": 0.0,
            "adjacent_heavy_sterics": 0.0,
            "adjacent_backbone_sterics": 0.0,
            "reference_offset": 0.0,
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
            r_elec = D[elec_mask]
            r_elec = np.maximum(r_elec, 1.0)
            q_prod = Q_mat[elec_mask]
            add_term("electrostatics", np.sum(83.0 * q_prod / (r_elec**2)))

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
        e_omega = 0.0
        for i in range(self.n_residues - 1):
            omega = self._bounded_omega(angle_dict.get(f"{i}_omega", np.pi))
            delta = omega - self.OMEGA_CENTER
            # Inside the enforced 170-190 degree band, this is mild: the edge
            # costs 1.0 * QTF_OMEGA_SCALE per peptide. Values outside the band
            # are clamped before geometry is built, so they cannot be sampled.
            e_omega += (delta / self.OMEGA_HALF_WIDTH) ** 2
        add_term("omega", OMEGA_SCALE * e_omega)

        e_local_sterics, local_steric_terms = self._calculate_adjacent_heavy_sterics(
            coords, self.static_labels, self.atom_to_res, return_terms=True
        )
        add_term("adjacent_heavy_sterics", e_local_sterics)
        terms["adjacent_backbone_sterics"] = float(e_local_sterics)

        e_geom, geom_subterms = self._calculate_geometry_integrity(
        coords, self.static_labels, self.atom_to_res, return_terms=True
        )
        add_term("geometry", e_geom)

        REFERENCE_OFFSET_PER_RESIDUE = float(os.getenv("QTF_REFERENCE_OFFSET_PER_RESIDUE", "-75.0"))
        add_term("reference_offset", REFERENCE_OFFSET_PER_RESIDUE * self.n_residues)

        if self.tracker:
            self.tracker.log(total_energy)

        if return_terms:
            self.last_energy_terms = {
            **terms,

            "energy_backend_custom": 1.0,
            "energy_backend_rosetta": 0.0,
            "use_e2e_constraint": 1.0 if self.use_e2e_constraint else 0.0,
            "reference_offset_per_residue": float(REFERENCE_OFFSET_PER_RESIDUE),

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

    def get_smart_initialization(self, n_attempts=20, seed=None):
        """
        Samples random parameters to find a good starting point (Basin Hopping).
        This avoids getting stuck in high-energy states immediately.
        
        REPRODUCIBILITY:
        Uses a hash of the sequence to seed the random number generator. 
        This ensures that every run with the same sequence starts from the same 
        initial geometry, allowing you to test energy function changes reliably.
        """
        if seed is not None:
            # Create a deterministic seed from the protein sequence
            seed = int(hashlib.sha256(self.sequence.encode('utf-8')).hexdigest(), 16) % (2**32)
        
        rng = np.random.default_rng(seed)
        
        print(f"--- SCOUTING: Checking {n_attempts} starting points ---")
        print(f" > Deterministic Seed: {seed} (Derived from Sequence)")
        
        best_params = None
        best_energy = float('inf')
        for i in range(n_attempts):
            trial_params = rng.uniform(-0.8, 0.8, self.n_params)
            e = self.energy_function(trial_params)
            if e < best_energy:
                best_energy = e
                best_params = trial_params
        print(f" > Best Start Found: Energy {best_energy:.2f}")
        return best_params

    def get_random_parameters(self, seed=None):
        """Return a pure random parameter vector in the ansatz parameter space."""
        rng = np.random.default_rng(seed)
        return rng.uniform(-np.pi, np.pi, self.n_params)

    def fold(self, max_iter=2000, initial_params=None):
        """
        MAIN EXECUTION LOOP
        Runs the 3-stage optimization curriculum.
        """
        print(f"--- STARTING QUANTUM FOLDING ---")
        self.tracker = LandscapeTracker() # Start Tracker

        # Initialization
        if initial_params is None:
            init_params = self.get_random_parameters()
        else:
            init_params = initial_params

        if int(max_iter) <= 0:
            self.tracker.mark_stage("Stage3")
            self.current_stage = 3
            final_energy = float(self.energy_function(init_params))
            coords, labels, bonds = self._final_output_structure_from_params(init_params)
            return coords, labels, bonds, self.tracker, init_params, final_energy

        # STAGE 1: COLLAPSE
        # High Gamma (15.0) + Magnet Active
        # Purpose: Force the protein into a ball rapidly.
        print("Stage 1: Mechanical Collapse (High Force)...")
        self.tracker.mark_stage("Stage1")
        self.current_stage = 1
        res_1 = minimize(self.energy_function, init_params, method='COBYLA', options={'maxiter': max_iter, 'rhobeg': 1.0})
        print(f" > Collapse Energy: {res_1.fun:.2f}")

        # STAGE 2: REFINE
        # High Gamma (15.0) + Magnet Active + Precise Gradient Descent
        # Purpose: Fix local clashes while keeping the globular shape.
        print("Stage 2: Physics Refinement (High Force)...")
        self.tracker.mark_stage("Stage2")
        self.current_stage = 2
        res_2 = minimize(self.energy_function, res_1.x, method='SLSQP', tol=1e-6, options={'maxiter': max_iter, 'disp': True})
        print(f" > Refinement Energy: {res_2.fun:.2f}")
        
        # STAGE 3: RELAX
        # Low Gamma (2.0) + Magnet REMOVED
        # Purpose: Let the protein "breathe". H-bonds and Electrostatics take over
        # to find the native state.
        print("Stage 3: Natural Relaxation (Releasing Constraints)...")
        self.tracker.mark_stage("Stage3")
        self.current_stage = 3 
        res_3 = minimize(self.energy_function, res_2.x, method='SLSQP', tol=1e-6, options={'maxiter': max_iter, 'disp': True})
        print(f" > Relaxation Energy (Final Energy): {res_3.fun:.2f}")
        
        coords, labels, bonds = self._final_output_structure_from_params(res_3.x)
        return coords, labels, bonds, self.tracker, res_3.x, res_3.fun

    def _aa1_to_3(self, aa):
        aa1_to_3 = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
        }
        return aa1_to_3.get(str(aa).upper(), 'UNK')

    def _format_pdb_atom_line(self, serial, atom_name, res_name, chain_id, resseq, x, y, z, element='C'):
        return (
            f"ATOM  {serial:5d} {atom_name:>4} {res_name:>3} {chain_id:1}{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2}\n"
        )

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

    def save_pdb(self, coords, labels, filename="structure.pdb", energy=0.0, chain_id='A', resseqs=None, resnames=None, remarks=None, include_hydrogens=True):
        """
        Save arbitrary coordinates/labels to a PDB file viewable in PyMOL or Chimera.

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
        outdir = os.path.dirname(filename)
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        chain_out = (chain_id or 'A')[:1]
        coords = np.asarray(coords, dtype=float)

        with open(filename, 'w') as f:
            if energy is not None:
                f.write(f"REMARK   1 ENERGY: {float(energy):.3f}\n")
            if remarks:
                for idx, remark in enumerate(remarks, start=2):
                    f.write(f"REMARK {idx:3d} {remark}\n")

            serial = 1
            for pos, (res_id, atom_name, elem) in zip(coords, labels):
                if (not include_hydrogens) and (str(elem).upper() == "H" or str(atom_name).upper().startswith("H")):
                    continue
                res_id = int(res_id)
                if resnames is None:
                    aa = self.sequence[res_id] if 0 <= res_id < len(self.sequence) else 'X'
                    res_name = self._aa1_to_3(aa)
                elif isinstance(resnames, dict):
                    res_name = str(resnames.get(res_id, 'UNK'))
                else:
                    res_name = str(resnames[res_id])

                if resseqs is None:
                    resseq = res_id + 1
                elif isinstance(resseqs, dict):
                    resseq = int(resseqs.get(res_id, res_id + 1))
                else:
                    resseq = int(resseqs[res_id])

                f.write(self._format_pdb_atom_line(
                    serial, atom_name, res_name, chain_out, resseq,
                    float(pos[0]), float(pos[1]), float(pos[2]), str(elem)
                ))
                serial += 1
            f.write('END\n')

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
class EnsembleFoldingManager:
    """
    Manages multiple independent folding runs (Monte Carlo / Multi-Start)
    """
    def __init__(self, folder_instance):
        self.folder = folder_instance
        self.results = [] 
    
    def prime_circuit(self, target_type='helix', seed=42):
        """
        Smart Initialization: Pre-optimizes circuit to output Secondary Structure angles.
        """
        print(f"--- PRIMING CIRCUIT FOR {target_type.upper()} ---")
        
        rng = np.random.default_rng(seed)
        
        if target_type == 'helix':
            t_phi, t_psi = np.deg2rad(-60.0), np.deg2rad(-45.0)
        elif target_type == 'sheet':
            t_phi, t_psi = np.deg2rad(-135.0), np.deg2rad(135.0)
        else:
            return rng.uniform(-0.8, 0.8, self.folder.n_params)

        targets = np.zeros(self.folder.total_angles)
        masks = np.zeros(self.folder.total_angles)
        
        for i, dof in enumerate(self.folder.dof_map):
            if dof['type'] == 'phi': targets[i] = t_phi; masks[i] = 1.0
            elif dof['type'] == 'psi': targets[i] = t_psi; masks[i] = 1.0
            
        def priming_cost(params):
            curr = self.folder._get_angles(params)
            diff = (curr - targets + np.pi) % (2 * np.pi) - np.pi
            return np.sum((diff * masks)**2)

        init_guess = rng.uniform(-0.1, 0.1, self.folder.n_params)
        res = minimize(priming_cost, init_guess, method='COBYLA', options={'maxiter': 200})
        print(f" > Priming Error: {res.fun:.4f}")
        return res.x

    def run_ensemble(self, n_runs=5, max_iter=2000, prime_strategy='mixed'):
        print(f"=== STARTING ENSEMBLE RUN ({n_runs} Trajectories) ===")
        self.results = []

        # Create a deterministic seed from the protein sequence
        # Retrieve the protein sequence from the folder object
        seed = int(hashlib.sha256(self.folder.sequence.encode('utf-8')).hexdigest(), 16) % (2**32)
        
        for i in range(n_runs):
            print(f"\n>> REPLICA {i+1}/{n_runs}")
            
            # Strategy Selection
            if prime_strategy == 'mixed':
                if i % 3 == 0: strat = 'helix'
                elif i % 3 == 1: strat = 'sheet'
                else: strat = 'random'
            else: strat = prime_strategy
                
            # Initialization
            if strat == 'random':
                start_params = self.folder.get_random_parameters(seed=seed+i)
            else:
                start_params = self.prime_circuit(target_type=strat, seed=seed+i)
            
            # Execute Fold
            coords, labels, _, tracker, final_params, final_energy = self.folder.fold(max_iter=max_iter, initial_params=start_params)
            
            print(f" >> Replica {i+1} Final Energy: {final_energy:.2f}")
            self.results.append({
                'id': i, 'seed': seed+i, 'type': strat, 'energy': final_energy,
                'coords': coords, 'labels': labels, 'params': final_params, 'tracker': tracker
            })

    def evaluate_best(self):
        if not self.results: return None, None

        sorted_results = sorted(self.results, key=lambda x: x['energy'])
        best = sorted_results[0]
        
        print(f"\n=== ENSEMBLE EVALUATION ===")
        print(f"Best Candidate: ID {best['id']} (Init: {best['type']})")
        print(f"Lowest Energy:  {best['energy']:.4f}")
        
        if len(self.results) > 1:
            StabilityAnalyzer.analyze_convergence(self.results)

        return best

    def get_ranked_results(self):
        """Return all ensemble results sorted by energy (ascending)."""
        if not self.results:
            return []
        return sorted(self.results, key=lambda x: x['energy'])

    def select_top(self, top_k=None, top_frac=None):
        """
        Select top low-energy structures.

        Parameters
        ----------
        top_k : int | None
            Keep the top_k lowest-energy structures.
        top_frac : float | None
            Keep the top fraction (0<top_frac<=1) of lowest-energy structures.
            If provided, top_frac takes precedence over top_k.

        Returns
        -------
        list[dict]
            Ranked subset of self.results.
        """
        ranked = self.get_ranked_results()
        if not ranked:
            return []
        if top_frac is not None:
            k = max(1, int(np.ceil(len(ranked) * float(top_frac))))
            return ranked[:k]
        if top_k is not None:
            k = max(1, min(int(top_k), len(ranked)))
            return ranked[:k]
        return ranked
