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
        
        # 4. SIDE CHAIN TOPOLOGY
        # Ref: Engh, R. A., & Huber, R. (1991). Acta Cryst. A47, 392-400.
        self.SIDE_CHAIN_TOPO = {
            'G': [],
            'A': [('CB', 'CA', 1.53, 1.91, 2.1)],
            
            # Hydrophobic
            'V': [('CB', 'CA', 1.53, 1.91, 'chi1'), 
                  ('CG1', 'CB', 1.52, 1.91, 'chi2'), ('CG2', 'CB', 1.52, 1.91, 'chi2_branch')],
            'L': [('CB', 'CA', 1.53, 1.91, 'chi1'), 
                  ('CG', 'CB', 1.52, 1.91, 'chi2'), 
                  ('CD1', 'CG', 1.52, 1.91, 'chi3'), ('CD2', 'CG', 1.52, 1.91, 'chi3_branch')],
            'I': [('CB', 'CA', 1.53, 1.91, 'chi1'), 
                  ('CG1', 'CB', 1.54, 1.91, 'chi2'), ('CD1', 'CG1', 1.52, 1.91, 'chi3'),
                  ('CG2', 'CB', 1.54, 1.91, 'chi2_branch')],
            'M': [('CB', 'CA', 1.53, 1.91, 'chi1'), ('CG', 'CB', 1.52, 1.91, 'chi2'),
                  ('SD', 'CG', 1.81, 1.91, 'chi3'), ('CE', 'SD', 1.79, 1.76, 'chi4')],
            'P': [('CB', 'CA', 1.53, 1.80, 'chi1'), ('CG', 'CB', 1.50, 1.82, 'chi2'),
                  ('CD', 'CG', 1.52, 1.83, 'chi3')], 

            # Aromatic
            'F': [('CB', 'CA', 1.53, 1.91, 'chi1'), ('CG', 'CB', 1.50, 1.91, 'chi2'),
                  ('CD1', 'CG', 1.39, 2.09, 1.57), ('CD2', 'CG', 1.39, 2.09, -1.57),
                  ('CE1', 'CD1', 1.39, 2.09, 3.14), ('CE2', 'CD2', 1.39, 2.09, 3.14),
                  ('CZ', 'CE1', 1.39, 2.09, 0.0)],
            'Y': [('CB', 'CA', 1.53, 1.91, 'chi1'), ('CG', 'CB', 1.50, 1.91, 'chi2'),
                  ('CD1', 'CG', 1.39, 2.09, 1.57), ('CD2', 'CG', 1.39, 2.09, -1.57),
                  ('CE1', 'CD1', 1.39, 2.09, 3.14), ('CE2', 'CD2', 1.39, 2.09, 3.14),
                  ('CZ', 'CE1', 1.39, 2.09, 0.0), 
                  ('OH', 'CZ', 1.37, 2.09, 3.14), ('HH', 'OH', 0.96, 1.83, 'chi3')], 
            'W': [('CB', 'CA', 1.53, 1.91, 'chi1'), ('CG', 'CB', 1.50, 1.91, 'chi2'),
                  ('CD1', 'CG', 1.37, 2.15, 1.0), ('CD2', 'CG', 1.43, 2.15, -1.0),
                  ('NE1', 'CD1', 1.38, 1.90, 3.14), ('HE1', 'NE1', 1.01, 2.09, 0.0), 
                  ('CE2', 'CD2', 1.40, 1.90, 0.0), ('CE3', 'CD2', 1.40, 2.30, 3.14), 
                  ('CZ2', 'CE2', 1.40, 2.10, 0.0), ('CZ3', 'CE3', 1.40, 2.10, 0.0), 
                  ('CH2', 'CZ2', 1.40, 2.10, 0.0)], 

            # Polar / Charged
            'S': [('CB', 'CA', 1.53, 1.91, 'chi1'), ('OG', 'CB', 1.42, 1.91, 'chi2'),
                  ('HG', 'OG', 0.96, 1.83, 'chi3')], 
            'T': [('CB', 'CA', 1.53, 1.91, 'chi1'), 
                  ('OG1', 'CB', 1.43, 1.91, 'chi2'), ('HG1', 'OG1', 0.96, 1.83, 'chi3'), 
                  ('CG2', 'CB', 1.53, 1.91, 'chi2_branch')],
            'C': [('CB', 'CA', 1.53, 1.91, 'chi1'), ('SG', 'CB', 1.81, 1.91, 'chi2')],
            'D': [('CB', 'CA', 1.53, 1.91, 'chi1'), ('CG', 'CB', 1.52, 1.91, 'chi2'),
                  ('OD1', 'CG', 1.25, 2.0, 1.0), ('OD2', 'CG', 1.25, 2.0, -1.0)],
            'N': [('CB', 'CA', 1.53, 1.91, 'chi1'), ('CG', 'CB', 1.52, 1.91, 'chi2'),
                  ('OD1', 'CG', 1.23, 2.09, 0.0), ('ND2', 'CG', 1.32, 2.09, 3.14)],
            'E': [('CB', 'CA', 1.53, 1.91, 'chi1'), ('CG', 'CB', 1.52, 1.91, 'chi2'),
                  ('CD', 'CG', 1.52, 1.91, 'chi3'), ('OE1', 'CD', 1.25, 2.0, 1.0), ('OE2', 'CD', 1.25, 2.0, -1.0)],
            'Q': [('CB', 'CA', 1.53, 1.91, 'chi1'), ('CG', 'CB', 1.52, 1.91, 'chi2'),
                  ('CD', 'CG', 1.52, 1.91, 'chi3'), ('OE1', 'CD', 1.23, 2.09, 0.0), ('NE2', 'CD', 1.32, 2.09, 3.14)],
            'K': [('CB', 'CA', 1.53, 1.91, 'chi1'), ('CG', 'CB', 1.52, 1.91, 'chi2'),
                  ('CD', 'CG', 1.52, 1.91, 'chi3'), ('CE', 'CD', 1.52, 1.91, 'chi4'),
                  ('NZ', 'CE', 1.49, 1.91, 'chi5')],
            'R': [('CB', 'CA', 1.53, 1.91, 'chi1'), ('CG', 'CB', 1.52, 1.91, 'chi2'),
                  ('CD', 'CG', 1.52, 1.91, 'chi3'), ('NE', 'CD', 1.46, 1.91, 'chi4'),
                  ('CZ', 'NE', 1.33, 2.15, 'chi5'), ('NH1', 'CZ', 1.33, 2.10, 0.0), ('NH2', 'CZ', 1.33, 2.10, 3.14)],
            'H': [('CB', 'CA', 1.53, 1.91, 'chi1'), ('CG', 'CB', 1.50, 1.91, 'chi2'),
                  ('ND1', 'CG', 1.38, 2.15, 1.0), ('CD2', 'CG', 1.36, 2.15, -1.0),
                  ('CE1', 'ND1', 1.32, 1.90, 0.0), 
                  ('NE2', 'CD2', 1.32, 1.90, 0.0), ('HE2', 'NE2', 1.01, 2.09, 0.0)], 
            
            'DEFAULT': [('CB', 'CA', 1.53, 1.91, 'chi1')]
        }

        # --- QUANTUM SETUP ---
        # 1. Map sequence to Degrees of Freedom (DoF)
        self.dof_map = []
        for i, aa in enumerate(self.sequence):
            self.dof_map.append({'res': i, 'type': 'phi'})
            self.dof_map.append({'res': i, 'type': 'psi'})

            topo = self.SIDE_CHAIN_TOPO.get(aa, self.SIDE_CHAIN_TOPO['DEFAULT'])
            chis = set()
            for atom in topo:
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

        # --- OPTIONAL ROSETTA-BACKED STAGE-3 SCORING ---
        self.stage3_backend = os.getenv("QTF_STAGE3_BACKEND", "rosetta" if PYROSETTA_AVAILABLE else "custom").strip().lower()
        self.rosetta_flags = os.getenv("QTF_PYROSETTA_FLAGS", "-mute all")
        self.rosetta_centroid_weights = os.getenv("QTF_ROSETTA_CEN_WTS", "cen_std")
        self.rosetta_fullatom_weights = os.getenv("QTF_ROSETTA_FA_WTS", "ref2015")
        self.rosetta_cen_weight = float(os.getenv("QTF_ROSETTA_CEN_WEIGHT", "0.35"))
        self.rosetta_fa_weight = float(os.getenv("QTF_ROSETTA_FA_WEIGHT", "1.0"))
        self.rosetta_do_centroid_min = os.getenv("QTF_ROSETTA_CEN_MIN", "0") == "1"
        self.rosetta_do_fullatom_min = os.getenv("QTF_ROSETTA_FA_MIN", "1") == "1"
        self.rosetta_do_repack = os.getenv("QTF_ROSETTA_REPACK", "1") == "1"
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
        return np.angle(psi)[:self.total_angles]

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
        Constructs the full 3D Cartesian coordinates of the protein 
        based on the input torsion angles.
        """
        coords = []; labels = []; bonds = [] 
        
        # Map flat angle vector back to semantic names (e.g., '0_phi', '0_psi')
        angle_dict = {f"{x['res']}_{x['type']}": val for x, val in zip(self.dof_map, angle_vector)}
        
        # --- 1. INITIALIZE BACKBONE START ---
        # First 3 atoms (N, CA, C) are placed statically to establish a frame of reference.
        coords.extend([np.array([0,0,0]), np.array([1.46,0,0]), np.array([1.46 + 1.51*np.cos(1.9), 1.51*np.sin(1.9), 0])])
        labels.extend([(0, 'N', 'N'), (0, 'CA', 'C'), (0, 'C', 'C')])
        bonds.extend([(0,1), (1,2)]) 
        
        # --- 2. MAIN RESIDUE LOOP ---
        for i in range(self.n_residues):
            
            # Helper to find index of atoms in the 'coords' list
            def get_idx(r, n):
                for k in range(len(labels)-1, -1, -1): 
                    if labels[k][0] == r and labels[k][1] == n: return k
                return -1

            idx_N = get_idx(i, 'N'); idx_CA = get_idx(i, 'CA'); idx_C = get_idx(i, 'C')
            
            # --- BUILD SIDE CHAIN ---
            topo = self.SIDE_CHAIN_TOPO.get(self.sequence[i], self.SIDE_CHAIN_TOPO['DEFAULT'])
            sc_map = {} 
            for atom_def in topo:
                name, elem, b_len, b_ang, tor_def = atom_def
                
                # Determine Torsion Value
                t_val = 0.0
                if isinstance(tor_def, str) and 'chi' in tor_def:
                    t_val = angle_dict.get(f"{i}_{tor_def.replace('_branch','')}", 0.0)
                    if 'branch' in tor_def: t_val += 2.09 # Offset for branched chains (Val/Leu)
                else: t_val = tor_def # Fixed angle

                # Special Case: First Sidechain Atom (CB) connects to CA
                # Requires calculating the bisector of N-CA-C to place it correctly.
                if name == 'CB': 
                    u_nc = coords[idx_N] - coords[idx_CA]; u_cc = coords[idx_C] - coords[idx_CA]
                    n_plane = np.cross(u_nc, u_cc); n_plane /= (np.linalg.norm(n_plane)+1e-9)
                    u_mid = -(u_nc + u_cc); u_mid /= (np.linalg.norm(u_mid)+1e-9)
                    p_CB = coords[idx_CA] + (b_len * (np.cos(0.9)*u_mid + np.sin(0.9)*n_plane))
                    coords.append(p_CB); labels.append((i, name, elem)); bonds.append((idx_CA, len(coords)-1))
                    sc_map['CB'] = len(coords) - 1
                else:
                    # General NERF placement for rest of sidechain
                    p_name = 'CB'
                    # Determine Parent (Simple logic for standard topologies)
                    if name.startswith('CD'): p_name = 'CG'
                    if name.startswith('CE'): p_name = 'CD'
                    if name.startswith('CZ'): p_name = 'CE'
                    if name.startswith('NZ'): p_name = 'CE'
                    if name.startswith('OE') or name.startswith('OD'): p_name = 'CD' if name.startswith('OE') else 'CG'
                    if name.startswith('SG'): p_name = 'CB'
                    if name.startswith('CG'): p_name = 'CB'
                    if name.startswith('CD') and self.sequence[i] == 'L': p_name = 'CG'
                    
                    # New Parent Mappings for Explicit Hydrogens
                    if name.startswith('HG') and name != 'HG1': p_name = 'OG'
                    if name == 'HG1': p_name = 'OG1'
                    if name == 'HH': p_name = 'OH'
                    if name == 'HE1': p_name = 'NE1'
                    if name == 'HE2': p_name = 'NE2'

                    idx_c = sc_map.get(p_name, -1)
                    if idx_c == -1: idx_c = len(coords) - 1
                    c = coords[idx_c]
                    
                    grandp = 'CA' if p_name == 'CB' else 'CB'
                    # More specific Grandparent mapping for Hydrogens to ensure correct angle
                    if p_name == 'OG': grandp = 'CB'
                    if p_name == 'OG1': grandp = 'CB'
                    if p_name == 'OH': grandp = 'CZ'
                    if p_name == 'NE1': grandp = 'CD1'
                    if p_name == 'NE2': grandp = 'CD2' # For His

                    if grandp == 'CA': b = coords[idx_CA]; a = coords[idx_N]
                    else: b = coords[sc_map.get(grandp, idx_c-1)]; a = coords[idx_CA]
                    
                    new_pos = self._nerf_step(a, b, c, b_len, b_ang, t_val)
                    coords.append(new_pos); labels.append((i, name, elem)); bonds.append((idx_c, len(coords)-1))
                    sc_map[name] = len(coords) - 1

            # --- BUILD BACKBONE OXYGEN (Carbonyl) ---
            p_O = self._nerf_step(coords[idx_N], coords[idx_CA], coords[idx_C], 1.23, 2.1, np.pi) 
            coords.append(p_O); labels.append((i, 'O', 'O')); bonds.append((idx_C, len(coords)-1))
            
            # --- BUILD NEXT RESIDUE BACKBONE (N, CA, C) ---
            if i < self.n_residues - 1:
                psi = angle_dict.get(f"{i}_psi", -0.5)
                p_next_N = self._nerf_step(coords[idx_N], coords[idx_CA], coords[idx_C], 1.33, 2.0, psi)
                coords.append(p_next_N); labels.append((i+1, 'N', 'N')); bonds.append((idx_C, len(coords)-1))
                
                omega = np.pi # Peptide bond is planar (trans)
                p_next_CA = self._nerf_step(coords[idx_CA], coords[idx_C], p_next_N, 1.46, 2.1, omega)
                coords.append(p_next_CA); labels.append((i+1, 'CA', 'C')); bonds.append((len(coords)-2, len(coords)-1))
                
                phi = angle_dict.get(f"{i+1}_phi", -1.0)
                p_next_C = self._nerf_step(coords[idx_C], p_next_N, p_next_CA, 1.51, 1.9, phi)
                coords.append(p_next_C); labels.append((i+1, 'C', 'C')); bonds.append((len(coords)-2, len(coords)-1))

        return np.array(coords), labels, bonds

    def _initialize_topology_cache(self):
        """
        Runs structure builder once to determine static properties of atoms.
        Allows vectorization of Charges, Radii, and Types.
        """
        # Build with dummy zeros to get lists
        dummy_coords, self.static_labels, _ = self.build_full_structure(np.zeros(self.total_angles))
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
        
        # 3. Pre-calculate VdW Radii (Vectorized)
        self.vdw_radii_vector = np.array([self.VDW_RADII.get(x[2], 1.7) for x in self.static_labels])
        
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
                
        # Mask for Bonded/Neighbor Exclusions (i, i+1 residues)
        # We generally don't calculate VdW/Electrostatics for atoms in the same or adjacent residues
        res_diff_matrix = np.abs(self.atom_to_res[:, None] - self.atom_to_res[None, :])
        self.mask_non_bonded = (res_diff_matrix >= 2) 
        
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
        nres = pose.total_residue()
        for i in range(1, nres):
            pose.set_omega(i, 180.0)
        for dof, ang in zip(self.dof_map, angle_vec):
            resi = int(dof['res']) + 1
            t = str(dof['type'])
            deg = float(np.rad2deg(ang))
            if t == 'phi':
                pose.set_phi(resi, deg)
            elif t == 'psi':
                pose.set_psi(resi, deg)
            elif t.startswith('chi'):
                try:
                    chi_idx = int(t.replace('chi', ''))
                except Exception:
                    continue
                if chi_idx <= pose.residue(resi).nchi():
                    pose.set_chi(chi_idx, resi, deg)
        return pose

    def _pose_ca_coords(self, pose):
        ca = []
        for i in range(1, pose.total_residue() + 1):
            rsd = pose.residue(i)
            if rsd.has("CA"):
                xyz = rsd.xyz("CA")
                ca.append([float(xyz.x), float(xyz.y), float(xyz.z)])
        return np.asarray(ca, dtype=float) if ca else np.zeros((0, 3), dtype=float)

    def _extract_rosetta_terms(self, pose, scorefxn, prefix):
        _ = scorefxn(pose)
        emap = pose.energies().total_energies()
        names = [
            'fa_atr', 'fa_rep', 'fa_sol', 'lk_ball_wtd', 'fa_elec',
            'hbond_sr_bb', 'hbond_lr_bb', 'hbond_bb_sc', 'hbond_sc',
            'rama_prepro', 'omega', 'p_aa_pp', 'fa_dun', 'dslf_fa13', 'ref',
            'env', 'pair', 'cbeta', 'vdw', 'rg', 'rama'
        ]
        out = {}
        for name in names:
            if hasattr(rosetta.core.scoring, name):
                st = getattr(rosetta.core.scoring, name)
                try:
                    val = float(emap[st])
                except Exception:
                    continue
                if abs(val) > 1e-12:
                    out[f"{prefix}_{name}"] = val
        out[f"{prefix}_total"] = float(scorefxn(pose))
        return out

    def _score_stage3_rosetta(self, angle_vec, return_terms=False):
        self._ensure_rosetta()
        terms = {}
        try:
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
                mm.set_bb(True)
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
            self._last_rosetta_pose = fa_pose.clone()
            self._last_rosetta_ca = self._pose_ca_coords(fa_pose)
        except Exception as exc:
            total = 1.0e6
            terms = {"rosetta_error": 1.0, "rosetta_total": float(total), "rosetta_message_hash": float(abs(hash(str(exc))) % 1000000)}
        self.last_energy_terms = terms
        return total

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

        # Stages 1-2 keep the original hand-built landscape for collapse/refinement.
        # Stage 3 can be replaced by Rosetta centroid + all-atom scoring.
        angle_vec = self._get_angles(params)
        if self.current_stage == 3 and self.stage3_backend == 'rosetta':
            return self._score_stage3_rosetta(angle_vec, return_terms=return_terms)

        # ==========================================
        # STAGE CONTROLLER (Guided Relaxation)
        # ==========================================
        gamma = 15.0
        constraint_strength = 50.0

        if self.current_stage == 3:
            gamma = 2.5
            constraint_strength = 5.0

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
        }
        total_energy = 0.0

        def add_term(name: str, value: float):
            nonlocal total_energy
            v = float(value)
            terms[name] += v
            total_energy += v

        diffs = coords[:, None, :] - coords[None, :, :]
        D = np.sqrt(np.sum(diffs**2, axis=-1)) + 1e-9

        neighbor_counts = np.array([0.0], dtype=float)
        burial_fractions = np.array([0.0], dtype=float)

        ca_indices = [i for i, lbl in enumerate(self.static_labels) if lbl[1] == 'CA']
        if len(ca_indices) >= 2:
            start_ca = coords[ca_indices[0]]
            end_ca = coords[ca_indices[-1]]
            dist_ends = np.linalg.norm(start_ca - end_ca)
            e_constraint = constraint_strength * (dist_ends - 5.5)**2
            add_term("constraint", e_constraint)

        hydro_indices = np.where(self.mask_hydrophobic)[0]
        if len(hydro_indices) > 1:
            hydro_coords = coords[hydro_indices]
            hydro_D = squareform(pdist(hydro_coords))
            neighbor_counts = np.sum((hydro_D < 8.5) & (hydro_D > 0), axis=1)
            burial_fractions = np.clip((neighbor_counts - 2.0) / 6.0, 0.0, 1.0)
            hydro_scale = float(os.getenv("QTF_SASA_SCALE", "1.0"))
            e_sasa = hydro_scale * (-gamma * np.sum(self.HYDROPHOBICITY[self.sequence[self.atom_to_res[hydro_indices]]] * burial_fractions))
            add_term("sasa", e_sasa)

        n_idx = self.idx_N_atoms
        o_idx = self.idx_O_atoms
        if len(n_idx) and len(o_idx):
            for n in n_idx:
                for o in o_idx:
                    if abs(self.atom_to_res[n] - self.atom_to_res[o]) < 2:
                        continue
                    d = D[n, o]
                    if d < 3.5:
                        e_hb = -3.0 * np.exp(-((d - 2.9) ** 2) / 0.15)
                        terms["hbond_raw"] += float(e_hb)
                        add_term("hbond", e_hb * float(os.getenv("QTF_HBOND_SCALE", "1.0")))

        qq = np.outer(self.q_vector, self.q_vector)
        elec_mask = self.mask_non_bonded & (np.abs(qq) > 1e-8)
        if np.any(elec_mask):
            e_elec = np.sum(332.0 * qq[elec_mask] / (4.0 * D[elec_mask]))
            add_term("electrostatics", e_elec)

        if len(self.idx_SG_atoms) >= 2:
            sg_coords = coords[self.idx_SG_atoms]
            sg_d = squareform(pdist(sg_coords))
            for i in range(len(self.idx_SG_atoms)):
                for j in range(i + 1, len(self.idx_SG_atoms)):
                    d = sg_d[i, j]
                    if d < 2.5:
                        add_term("disulfide", -20.0 * np.exp(-((d - 2.05) ** 2) / 0.02))

        heavy_idx = np.where(self.mask_heavy)[0]
        if len(heavy_idx) > 1:
            D_h = D[np.ix_(heavy_idx, heavy_idx)]
            rad = self.vdw_radii_vector[heavy_idx]
            sigma = rad[:, None] + rad[None, :]
            mask = self.mask_non_bonded[np.ix_(heavy_idx, heavy_idx)]
            rep = ((sigma / D_h) ** 12)
            attr = ((sigma / D_h) ** 6)
            rep_scale = float(os.getenv("QTF_VDW_REP_SCALE", "1.0"))
            attr_scale = float(os.getenv("QTF_VDW_ATTR_SCALE", "1.0"))
            add_term("vdw_repulsion", rep_scale * np.sum(rep[mask]))
            add_term("vdw_attractive", -attr_scale * np.sum(attr[mask]))

        angle_dict = {f"{x['res']}_{x['type']}": val for x, val in zip(self.dof_map, angle_vec)}
        rot_scale = float(os.getenv("QTF_ROTAMER_SCALE", "1.0"))
        pi_scale = float(os.getenv("QTF_PI_STACK_SCALE", "1.0"))
        add_term("rotamer", rot_scale * self._calculate_rotamer_energy(angle_dict))
        add_term("pi_stacking", pi_scale * self._calculate_aromatic_quadrupole(coords, self.static_labels, self.atom_to_res))

        rama_energy = 0.0
        for i in range(self.n_residues):
            phi = angle_dict.get(f"{i}_phi", 0.0)
            psi = angle_dict.get(f"{i}_psi", 0.0)
            if -2.3 < phi < -0.4 and -1.5 < psi < 0.4:
                rama_energy -= 1.5
            elif -2.8 < phi < -0.8 and 1.8 < psi < 3.2:
                rama_energy -= 1.0
            else:
                rama_energy += 1.5
        add_term("rama", rama_energy)

        geom_energy, _ = self._calculate_geometry_integrity(coords, self.static_labels, self.atom_to_res, return_terms=True)
        add_term("geometry", geom_energy)

        self.last_energy_terms = {k: float(v) for k, v in terms.items()}
        if return_terms:
            return float(total_energy)
        return float(total_energy)

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

            # PRO ring closure penalty
            if res_name == 'P' and 'CD' in atoms and 'N' in atoms:
                d = np.linalg.norm(coords[atoms['CD']] - coords[atoms['N']])
                if abs(d - 1.47) > 0.1:
                    penalty = 50.0 * (d - 1.47) ** 2
                    energy += penalty
                    geom_terms["pro_ring"] += penalty

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

    def fold(self, max_iter=2000, initial_params=None):
        """
        MAIN EXECUTION LOOP
        Runs the 3-stage optimization curriculum.
        """
        print(f"--- STARTING QUANTUM FOLDING ---")
        self.tracker = LandscapeTracker() # Start Tracker

        # Initialization
        if initial_params is None:
            init_params = self.get_smart_initialization(n_attempts=max_iter)
        else:
            init_params = initial_params
        
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
        
        coords, labels, bonds = self.build_full_structure(self._get_angles(res_3.x))
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

    def save_pdb(self, coords, labels, filename="structure.pdb", energy=0.0, chain_id='A', resseqs=None, resnames=None, remarks=None):
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

            for k, (pos, (res_id, atom_name, elem)) in enumerate(zip(coords, labels), start=1):
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
                    k, atom_name, res_name, chain_out, resseq,
                    float(pos[0]), float(pos[1]), float(pos[2]), str(elem)
                ))
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
                start_params = self.folder.get_smart_initialization(n_attempts=50, seed=seed+i)
            else:
                start_params = self.prime_circuit(target_type=strat, seed=seed+i)
            
            # Execute Fold
            coords, _, _, tracker, final_params, final_energy = self.folder.fold(max_iter=max_iter, initial_params=start_params)
            
            print(f" >> Replica {i+1} Final Energy: {final_energy:.2f}")
            self.results.append({
                'id': i, 'seed': seed+i, 'type': strat, 'energy': final_energy,
                'coords': coords, 'params': final_params, 'tracker': tracker
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


