import numpy as np
import os
import hashlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ── Qiskit imports ──────────────────────────────────────────────────────────────
try:
    from qiskit.circuit.library import EfficientSU2 as efficient_su2
    from qiskit.quantum_info import Statevector
    from qiskit import transpile, QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.circuit import ParameterVector
    
    QISKIT_AVAILABLE = True
except ImportError as e:
    print(f"Qiskit import failed: {e}")
    efficient_su2 = None
    Statevector   = None
    AerSimulator  = None
    QISKIT_AVAILABLE = False

from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


# ==============================================================================
# EXECUTION MODE
# ==============================================================================
# Controls how _get_angles works:
#
#   "statevector"  — Classical exact simulation using Statevector.
#                    Used during OPTIMIZATION (fast, gradient-friendly).
#
#   "sampler"      — Hardware-compatible: runs circuit in Z/X/Y bases,
#                    extracts angles from measurement statistics.
#                    Used for FINAL ANGLE EXTRACTION after optimization.
#                    Works with AerSimulator (noisy) or real IBM backend.
#
# The fold() method and EnsembleFoldingManager always optimize in
# "statevector" mode, then switch to "sampler" mode at the very end
# to extract the final angles from hardware or a noise model.
# ==============================================================================

EXECUTION_MODE = "statevector"   # global default; overridden per-call


# ==============================================================================
# 1. UTILITY: TRACKING & LOGGING
# ==============================================================================
class LandscapeTracker:
    def __init__(self):
        self.history      = []
        self.stage_markers = []
        self.current_iter  = 0

    def log(self, energy):
        self.history.append(energy)
        self.current_iter += 1

    def mark_stage(self, name):
        self.stage_markers.append((self.current_iter, name))


# ==============================================================================
# 2. ANALYSIS: STABILITY & CONVERGENCE (KABSCH)
# ==============================================================================
class StabilityAnalyzer:
    """Kabsch-RMSD superposition and convergence analysis."""

    @staticmethod
    def kabsch_rmsd(P, Q):
        P_centered = P - np.mean(P, axis=0)
        Q_centered = Q - np.mean(Q, axis=0)
        H = np.dot(P_centered.T, Q_centered)
        V, S, Wt = np.linalg.svd(H)
        d = (np.linalg.det(V) * np.linalg.det(Wt)) < 0.0
        if d:
            S[-1]    = -S[-1]
            V[:, -1] = -V[:, -1]
        R         = np.dot(V, Wt)
        P_rotated = np.dot(P_centered, R)
        diff      = P_rotated - Q_centered
        rms       = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        return rms, P_rotated + np.mean(Q, axis=0)

    @staticmethod
    def analyze_convergence(results, top_k=5):
        sorted_results = sorted(results, key=lambda x: x['energy'])
        best_k = sorted_results[:top_k]
        n = len(best_k)
        if n < 2:
            return
        print(f"\n--- CONVERGENCE ANALYSIS (Top {n} Structures) ---")
        rmsd_matrix = np.zeros((n, n))
        print("      ", end="")
        for i in range(n):
            print(f" #{i:<4}", end="")
        print("\n" + "-" * 60)
        for i in range(n):
            print(f"Ref #{i} |", end="")
            for j in range(n):
                if i == j:
                    rmsd_matrix[i, j] = 0.0
                else:
                    rmsd, _ = StabilityAnalyzer.kabsch_rmsd(
                        best_k[i]['coords'], best_k[j]['coords'])
                    rmsd_matrix[i, j] = rmsd
                print(f" {rmsd_matrix[i, j]:.2f} ", end="")
            print(f"  (E={best_k[i]['energy']:.1f})")
        avg_rmsd = np.sum(rmsd_matrix) / (n * (n - 1))
        print(f"\nAverage Pairwise RMSD: {avg_rmsd:.2f} Angstroms")
        if avg_rmsd < 2.0:
            print(">>> VERDICT: STABLE. High confidence in prediction.")
        elif avg_rmsd < 4.5:
            print(">>> VERDICT: FLEXIBLE. Core is stable, loops vary.")
        else:
            print(">>> VERDICT: UNSTABLE. No dominant basin found.")


# ==============================================================================
# 3. CORE: QUANTUM FOLDER
# ==============================================================================
class QuantumBiophysicsFolder:
    """
    Hybrid Quantum-Classical Protein Folder.

    HYBRID PIPELINE
    ───────────────
    Phase 1 – Classical optimisation (fast, exact):
        _get_angles(params, mode="statevector")
        Uses Qiskit Statevector to map circuit params → torsion angles
        exactly. The full COBYLA + SLSQP curriculum runs here.
        No IBM jobs are submitted.

    Phase 2 – Hardware angle extraction (once per replica):
        _get_angles(params, mode="sampler", backend=<real or noisy sim>)
        Loads the optimised params into the circuit, runs 3 jobs
        (Z / X / Y measurement bases) on the target backend, and
        decodes torsion angles from measurement statistics.
        This is the only point where IBM hardware is used.
    """

    def __init__(
        self,
        sequence,
        force_field='charmm',
        chi_mode='all',
        selective_chi_map=None,
    ):
        self.sequence          = sequence.upper()
        self.n_residues        = len(sequence)
        self.force_field       = force_field.lower()
        self.chi_mode          = chi_mode
        self.selective_chi_map = selective_chi_map or {}

        print(f"--- INITIALIZING QUANTUM BIOPHYSICS FOLDER (HARDWARE) ---")
        print(f"--- FORCE FIELD: {self.force_field.upper()} ---")

        # ── Force-field parameters (unchanged from original) ────────────────────
        self.HYDROPHOBICITY = {
            'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5,
            'M': 1.9, 'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8,
            'W': -0.9, 'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5,
            'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
        }
        common_charges = {
            'OXT': -1.0, 'NZ': 1.0, 'NH1': 0.5, 'NH2': 0.5,
            'OD1': -0.5, 'OD2': -0.5, 'OE1': -0.5, 'OE2': -0.5,
            'ND2': 0.5, 'NE2': 0.5, 'SG': -0.1, 'SD': -0.1,
            'HE2': 0.4, 'ND1': -0.4,
        }
        charmm_charges = {
            'N': -0.47, 'H': 0.31, 'CA': 0.07, 'C': 0.51, 'O': -0.51,
            'OG': -0.4, 'HG': 0.4, 'OG1': -0.4, 'HG1': 0.4,
            'OH': -0.4, 'HH': 0.4, 'NE1': -0.3, 'HE1': 0.3
        }
        amber_charges = {
            'N': -0.42, 'H': 0.27, 'CA': 0.00, 'C': 0.60, 'O': -0.57,
            'OG': -0.6, 'HG': 0.4, 'OG1': -0.6, 'HG1': 0.4,
            'OH': -0.5, 'HH': 0.4, 'NE1': -0.4, 'HE1': 0.3
        }
        opls_charges = {
            'N': -0.50, 'H': 0.30, 'CA': 0.14, 'C': 0.50, 'O': -0.50,
            'OG': -0.7, 'HG': 0.4, 'OG1': -0.7, 'HG1': 0.4,
            'OH': -0.7, 'HH': 0.4, 'NE1': -0.4, 'HE1': 0.35
        }
        self.CHARGES = common_charges.copy()
        if self.force_field == 'amber':
            self.CHARGES.update(amber_charges)
        elif self.force_field == 'opls':
            self.CHARGES.update(opls_charges)
        else:
            if self.force_field != 'charmm':
                print(f" > Warning: Unknown force field '{self.force_field}'. Defaulting to CHARMM.")
            self.CHARGES.update(charmm_charges)

        self.VDW_RADII = {'H': 0.6, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}

        self.SIDE_CHAIN_TOPO = {
            'G': [], 'A': [('CB', 'CA', 1.53, 1.91, 2.1)],
            'V': [('CB','CA',1.53,1.91,'chi1'),('CG1','CB',1.52,1.91,'chi2'),('CG2','CB',1.52,1.91,'chi2_branch')],
            'L': [('CB','CA',1.53,1.91,'chi1'),('CG','CB',1.52,1.91,'chi2'),('CD1','CG',1.52,1.91,'chi3'),('CD2','CG',1.52,1.91,'chi3_branch')],
            'I': [('CB','CA',1.53,1.91,'chi1'),('CG1','CB',1.54,1.91,'chi2'),('CD1','CG1',1.52,1.91,'chi3'),('CG2','CB',1.54,1.91,'chi2_branch')],
            'M': [('CB','CA',1.53,1.91,'chi1'),('CG','CB',1.52,1.91,'chi2'),('SD','CG',1.81,1.91,'chi3'),('CE','SD',1.79,1.76,'chi4')],
            'P': [('CB','CA',1.53,1.80,'chi1'),('CG','CB',1.50,1.82,'chi2'),('CD','CG',1.52,1.83,'chi3')],
            'F': [('CB','CA',1.53,1.91,'chi1'),('CG','CB',1.50,1.91,'chi2'),('CD1','CG',1.39,2.09,1.57),('CD2','CG',1.39,2.09,-1.57),('CE1','CD1',1.39,2.09,3.14),('CE2','CD2',1.39,2.09,3.14),('CZ','CE1',1.39,2.09,0.0)],
            'Y': [('CB','CA',1.53,1.91,'chi1'),('CG','CB',1.50,1.91,'chi2'),('CD1','CG',1.39,2.09,1.57),('CD2','CG',1.39,2.09,-1.57),('CE1','CD1',1.39,2.09,3.14),('CE2','CD2',1.39,2.09,3.14),('CZ','CE1',1.39,2.09,0.0),('OH','CZ',1.37,2.09,3.14),('HH','OH',0.96,1.83,'chi3')],
            'W': [('CB','CA',1.53,1.91,'chi1'),('CG','CB',1.50,1.91,'chi2'),('CD1','CG',1.37,2.15,1.0),('CD2','CG',1.43,2.15,-1.0),('NE1','CD1',1.38,1.90,3.14),('HE1','NE1',1.01,2.09,0.0),('CE2','CD2',1.40,1.90,0.0),('CE3','CD2',1.40,2.30,3.14),('CZ2','CE2',1.40,2.10,0.0),('CZ3','CE3',1.40,2.10,0.0),('CH2','CZ2',1.40,2.10,0.0)],
            'S': [('CB','CA',1.53,1.91,'chi1'),('OG','CB',1.42,1.91,'chi2'),('HG','OG',0.96,1.83,'chi3')],
            'T': [('CB','CA',1.53,1.91,'chi1'),('OG1','CB',1.43,1.91,'chi2'),('HG1','OG1',0.96,1.83,'chi3'),('CG2','CB',1.53,1.91,'chi2_branch')],
            'C': [('CB','CA',1.53,1.91,'chi1'),('SG','CB',1.81,1.91,'chi2')],
            'D': [('CB','CA',1.53,1.91,'chi1'),('CG','CB',1.52,1.91,'chi2'),('OD1','CG',1.25,2.0,1.0),('OD2','CG',1.25,2.0,-1.0)],
            'N': [('CB','CA',1.53,1.91,'chi1'),('CG','CB',1.52,1.91,'chi2'),('OD1','CG',1.23,2.09,0.0),('ND2','CG',1.32,2.09,3.14)],
            'E': [('CB','CA',1.53,1.91,'chi1'),('CG','CB',1.52,1.91,'chi2'),('CD','CG',1.52,1.91,'chi3'),('OE1','CD',1.25,2.0,1.0),('OE2','CD',1.25,2.0,-1.0)],
            'Q': [('CB','CA',1.53,1.91,'chi1'),('CG','CB',1.52,1.91,'chi2'),('CD','CG',1.52,1.91,'chi3'),('OE1','CD',1.23,2.09,0.0),('NE2','CD',1.32,2.09,3.14)],
            'K': [('CB','CA',1.53,1.91,'chi1'),('CG','CB',1.52,1.91,'chi2'),('CD','CG',1.52,1.91,'chi3'),('CE','CD',1.52,1.91,'chi4'),('NZ','CE',1.49,1.91,'chi5')],
            'R': [('CB','CA',1.53,1.91,'chi1'),('CG','CB',1.52,1.91,'chi2'),('CD','CG',1.52,1.91,'chi3'),('NE','CD',1.46,1.91,'chi4'),('CZ','NE',1.33,2.15,'chi5'),('NH1','CZ',1.33,2.10,0.0),('NH2','CZ',1.33,2.10,3.14)],
            'H': [('CB','CA',1.53,1.91,'chi1'),('CG','CB',1.50,1.91,'chi2'),('ND1','CG',1.38,2.15,1.0),('CD2','CG',1.36,2.15,-1.0),('CE1','ND1',1.32,1.90,0.0),('NE2','CD2',1.32,1.90,0.0),('HE2','NE2',1.01,2.09,0.0)],
            'DEFAULT': [('CB', 'CA', 1.53, 1.91, 'chi1')]
        }

        # ── Degrees of freedom ──────────────────────────────────────────────────
        self.dof_map = []
        for i, aa in enumerate(self.sequence):
            self.dof_map.append({'res': i, 'type': 'phi'})
            self.dof_map.append({'res': i, 'type': 'psi'})
            topo  = self.SIDE_CHAIN_TOPO.get(aa, self.SIDE_CHAIN_TOPO['DEFAULT'])
            chis  = set()
            for atom in topo:
                tor = atom[4]
                if isinstance(tor, str) and 'chi' in tor:
                    chis.add(tor.replace('_branch', ''))
            for k in self._allowed_chis_for_residue(i, aa, chis):
                self.dof_map.append({'res': i, 'type': k})

        self.total_angles = len(self.dof_map)
        self.n_qubits     = max(2, int(np.ceil(np.log2(self.total_angles))))
        # ── Reps calculation ───────────────────────────────────────────────────────
        # Rule of thumb for expressibility vs hardware cost:
        #
        # We need the circuit to express at least total_angles independent values.
        # Each rep adds 2*n_qubits parameters (Ry+Rz per qubit).
        # So minimum reps = ceil(total_angles / (2 * n_qubits))
        #
        # But we also want enough entanglement layers to create correlations
        # between qubits (mimicking how real protein torsions are correlated).
        # Empirically: 3-5 reps is the sweet spot for VQE-style circuits.
        #
        # Too few reps → not expressive enough → optimizer gets stuck
        # Too many reps → barren plateaus → gradients vanish → optimizer gets stuck
        #
        # Hardware constraint: each rep adds ~2*(n_qubits-1) 2Q gates
        # For n=6 qubits: each rep ≈ 10 CX gates → depth grows linearly

        min_reps    = int(np.ceil(self.total_angles / (2 * self.n_qubits)))
        # cap at 6 to avoid barren plateaus and excessive hardware depth
        self.reps   = max(3, min(min_reps, 6))
        #self.reps         = int(np.ceil(self.total_angles / self.n_qubits)) + 2

        if QISKIT_AVAILABLE:
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

            self.ansatz   = build_brickwork_ansatz(self.n_qubits, self.reps)
            self.n_params = self.ansatz.num_parameters
        else:
            self.ansatz   = None
            self.n_params = 1

        self.current_stage    = 1
        self._cache_initialized = False
        self._initialize_topology_cache()
        self.tracker = None

    # ── chi filter ──────────────────────────────────────────────────────────────
    def _allowed_chis_for_residue(self, res_idx, aa, available_chis):
        available = sorted(set(available_chis), key=lambda x: (len(x), x))
        if self.chi_mode == "all":
            return available
        if self.chi_mode == "chi1_only":
            return [c for c in available if c == "chi1"]
        if self.chi_mode == "selective":
            allowed = set(self.selective_chi_map.get(aa, ["chi1"]))
            return [c for c in available if c in allowed]
        raise ValueError(f"Unknown chi_mode: {self.chi_mode}")

    # ──────────────────────────────────────────────────────────────────────────
    # CORE: _get_angles — two modes
    # ──────────────────────────────────────────────────────────────────────────
    def _get_angles(self, params, mode: str = "statevector", backend=None, shots: int = 4096):
        """
        Maps circuit parameters → torsion angles.

        Parameters
        ----------
        params  : np.ndarray   — circuit parameters (trainable)
        mode    : str
            "statevector"  — exact classical simulation (used during optimisation)
            "sampler"      — measurement-based extraction (used for final hardware run)
        backend : Qiskit backend or None
            Only used when mode="sampler".
            None → AerSimulator() (noiseless shot-based sim).
            Pass a real IBM backend for actual hardware execution.
        shots   : int
            Number of shots per basis circuit (only used in sampler mode).
        """
        if self.ansatz is None:
            raise RuntimeError("Qiskit not available.")

        # ── bind params ────────────────────────────────────────────────────────
        param_dict    = dict(zip(self.ansatz.parameters, params))
        bound_circuit = self.ansatz.assign_parameters(param_dict)

        # ══════════════════════════════════════════════════════════════════════
        # MODE 1: STATEVECTOR  (classical optimisation phase)
        # ══════════════════════════════════════════════════════════════════════
        if mode == "statevector":
            if Statevector is None:
                raise RuntimeError("Statevector not available. Install qiskit.")
            sv   = Statevector(bound_circuit).data
            return np.angle(sv)[:self.total_angles]

        # ══════════════════════════════════════════════════════════════════════
        # MODE 2: SAMPLER  (hardware / noisy-sim final extraction)
        # ══════════════════════════════════════════════════════════════════════
        if mode == "sampler":
            if backend is None:
                backend = AerSimulator()

            n        = self.n_qubits
            n_states = 2 ** n   # full probability vector length e.g. 64 for n=6

            # ── helpers ───────────────────────────────────────────────────────
            def _basis_circuit(qc, basis: str):
                c = qc.copy()
                if basis == "X":
                    for q in range(n):
                        c.h(q)
                elif basis == "Y":
                    for q in range(n):
                        c.sdg(q)
                        c.h(q)
                c.measure_all()
                return c

            def _run(qc):
                tqc = transpile(qc, backend)
                job = backend.run(tqc, shots=shots)
                return job.result().get_counts()

            def _counts_to_pvec(counts):
                """
                Convert measurement counts → full probability vector of length 2^n.
                Each bitstring maps to an integer index (qubit-0 = LSB).
                """
                pvec  = np.zeros(n_states, dtype=float)
                total = sum(counts.values())
                for bitstring, c in counts.items():
                    bs  = bitstring.replace(" ", "")[::-1]  # qubit-0 at index 0
                    idx = int(bs, 2)                         # binary → integer
                    pvec[idx] += c / total
                return pvec

            # ── run Z / X / Y → full probability vectors ──────────────────────
            pZ = _counts_to_pvec(_run(_basis_circuit(bound_circuit, "Z")))
            pX = _counts_to_pvec(_run(_basis_circuit(bound_circuit, "X")))
            pY = _counts_to_pvec(_run(_basis_circuit(bound_circuit, "Y")))

            # ── convert probability vectors → angles ───────────────────────────
            # State angles: each state |k> assigned angle 2π*k/N on unit circle
            state_angles = 2.0 * np.pi * np.arange(n_states) / n_states

            def _marginal_angles(pvec):
                """
                Per-qubit marginal: p(qubit_i = 1) = sum of probs where bit i = 1
                Map [0,1] → [-π, π] via:  angle = 2π * p1 - π
                Gives n angles, one per qubit.
                """
                angles = []
                for q in range(n):
                    mask = np.array([(k >> q) & 1 for k in range(n_states)],
                                    dtype=float)
                    p1 = np.dot(pvec, mask)              # P(qubit q = 1)
                    angles.append(2.0 * np.pi * p1 - np.pi)
                return np.array(angles)

            def _circular_mean(pvec):
                """
                Circular mean of the full distribution weighted by state_angles.
                Captures where probability mass is concentrated on the circle.
                angle = atan2(Σ p_k sin(θ_k), Σ p_k cos(θ_k)) → [-π, π]
                """
                s = np.sum(pvec * np.sin(state_angles))
                c = np.sum(pvec * np.cos(state_angles))
                return np.arctan2(s, c)

            def _cdf_angles(pvec):
                """
                Map full 2^n probability vector → 2^n angles via CDF.
                CDF[k] = cumulative probability up to state k → mapped to [-π, π].
                This preserves the full shape of the distribution.
                """
                cdf = np.cumsum(pvec)                    # length 2^n
                return 2.0 * np.pi * cdf - np.pi        # map [0,1] → [-π, π]

            # ── build angle set from all three bases ───────────────────────────
            # Per-qubit marginals: n angles per basis    → 3n total
            mZ = _marginal_angles(pZ)
            mX = _marginal_angles(pX)
            mY = _marginal_angles(pY)

            # Circular means: 1 per basis               → 3 total
            cmZ = _circular_mean(pZ)
            cmX = _circular_mean(pX)
            cmY = _circular_mean(pY)

            # CDF-based angles from Z basis: 2^n angles → richest signal
            cdf_angles = _cdf_angles(pZ)                # length 64 for n=6

            # Cross-basis KL divergence angles           → 2 total
            eps = 1e-12
            kl_ZX = float(np.sum(pZ * np.log((pZ + eps) / (pX + eps))))
            kl_ZY = float(np.sum(pZ * np.log((pZ + eps) / (pY + eps))))
            cross_1 = np.arctan(kl_ZX) * 2.0 - np.pi / 2.0
            cross_2 = np.arctan(kl_ZY) * 2.0 - np.pi / 2.0

            # Stack everything
            # Total: n + n + n + 3 + 2^n + 2 = 3n+5 + 2^n angles
            # For n=6: 3*6+5 + 64 = 87 angles — more than enough for 33 total_angles
            base = np.concatenate([
                mZ,                          # n marginal angles from Z
                mX,                          # n marginal angles from X
                mY,                          # n marginal angles from Y
                [cmZ, cmX, cmY],             # 3 circular means
                cdf_angles,                  # 2^n CDF angles (richest)
                [cross_1, cross_2],          # 2 cross-basis angles
            ])

            # ── clip to [-π, π] ────────────────────────────────────────────────
            base = np.clip(base, -np.pi, np.pi)

            # ── expand to total_angles if needed ───────────────────────────────
            if len(base) == 0:
                base = np.zeros(1, dtype=float)
            if len(base) >= self.total_angles:
                return base[:self.total_angles]

            out = np.zeros(self.total_angles, dtype=float)
            out[:len(base)] = base
            L = len(base)
            for k in range(L, self.total_angles):
                i = k % L
                j = (k * 3 + 1) % L
                m = (k * 7 + 2) % L
                v = (0.60 * base[i]
                     + 0.30 * np.sin(base[j])
                     + 0.10 * np.cos(base[m]))
                out[k] = (v + np.pi) % (2 * np.pi) - np.pi
            return out

        raise ValueError(f"Unknown mode: '{mode}'. Use 'statevector' or 'sampler'.")

    # ──────────────────────────────────────────────────────────────────────────
    # NERF geometry builder (unchanged)
    # ──────────────────────────────────────────────────────────────────────────
    def _nerf_step(self, a, b, c, bond_len, bond_angle, torsion):
        bc   = c - b;  bc_u = bc / (np.linalg.norm(bc) + 1e-9)
        ab   = b - a;  n    = np.cross(ab, bc_u);  n_u = n / (np.linalg.norm(n) + 1e-9)
        bx_n = np.cross(n_u, bc_u)
        M    = np.column_stack((bc_u, bx_n, n_u))
        ts   = np.pi - bond_angle
        d    = np.array([bond_len * np.cos(ts),
                         bond_len * np.cos(torsion) * np.sin(ts),
                         bond_len * np.sin(torsion) * np.sin(ts)])
        return c + (M @ d)

    def build_full_structure(self, angle_vector):
        coords = []; labels = []; bonds = []
        angle_dict = {f"{x['res']}_{x['type']}": val
                      for x, val in zip(self.dof_map, angle_vector)}

        coords.extend([np.array([0, 0, 0]),
                       np.array([1.46, 0, 0]),
                       np.array([1.46 + 1.51 * np.cos(1.9), 1.51 * np.sin(1.9), 0])])
        labels.extend([(0, 'N', 'N'), (0, 'CA', 'C'), (0, 'C', 'C')])
        bonds.extend([(0, 1), (1, 2)])

        for i in range(self.n_residues):
            def get_idx(r, name):
                for k in range(len(labels) - 1, -1, -1):
                    if labels[k][0] == r and labels[k][1] == name:
                        return k
                return -1

            idx_N  = get_idx(i, 'N')
            idx_CA = get_idx(i, 'CA')
            idx_C  = get_idx(i, 'C')

            topo   = self.SIDE_CHAIN_TOPO.get(self.sequence[i], self.SIDE_CHAIN_TOPO['DEFAULT'])
            sc_map = {}
            for atom_def in topo:
                name, elem, b_len, b_ang, tor_def = atom_def
                t_val = 0.0
                if isinstance(tor_def, str) and 'chi' in tor_def:
                    t_val = angle_dict.get(f"{i}_{tor_def.replace('_branch', '')}", 0.0)
                    if 'branch' in tor_def:
                        t_val += 2.09
                else:
                    t_val = tor_def

                if name == 'CB':
                    u_nc  = coords[idx_N]  - coords[idx_CA]
                    u_cc  = coords[idx_C]  - coords[idx_CA]
                    n_pl  = np.cross(u_nc, u_cc); n_pl /= (np.linalg.norm(n_pl) + 1e-9)
                    u_mid = -(u_nc + u_cc);        u_mid /= (np.linalg.norm(u_mid) + 1e-9)
                    p_CB  = coords[idx_CA] + b_len * (np.cos(0.9) * u_mid + np.sin(0.9) * n_pl)
                    coords.append(p_CB); labels.append((i, name, elem)); bonds.append((idx_CA, len(coords) - 1))
                    sc_map['CB'] = len(coords) - 1
                else:
                    p_name = 'CB'
                    if name.startswith('CD'): p_name = 'CG'
                    if name.startswith('CE'): p_name = 'CD'
                    if name.startswith('CZ'): p_name = 'CE'
                    if name.startswith('NZ'): p_name = 'CE'
                    if name.startswith('OE') or name.startswith('OD'):
                        p_name = 'CD' if name.startswith('OE') else 'CG'
                    if name.startswith('SG'): p_name = 'CB'
                    if name.startswith('CG'): p_name = 'CB'
                    if name.startswith('CD') and self.sequence[i] == 'L': p_name = 'CG'
                    if name.startswith('HG') and name != 'HG1': p_name = 'OG'
                    if name == 'HG1':  p_name = 'OG1'
                    if name == 'HH':   p_name = 'OH'
                    if name == 'HE1':  p_name = 'NE1'
                    if name == 'HE2':  p_name = 'NE2'

                    idx_c  = sc_map.get(p_name, len(coords) - 1)
                    c      = coords[idx_c]
                    grandp = 'CA' if p_name == 'CB' else 'CB'
                    if p_name in ('OG', 'OG1'): grandp = 'CB'
                    if p_name == 'OH':           grandp = 'CZ'
                    if p_name == 'NE1':          grandp = 'CD1'
                    if p_name == 'NE2':          grandp = 'CD2'

                    if grandp == 'CA':
                        b = coords[idx_CA]; a = coords[idx_N]
                    else:
                        b = coords[sc_map.get(grandp, idx_c - 1)]; a = coords[idx_CA]

                    new_pos = self._nerf_step(a, b, c, b_len, b_ang, t_val)
                    coords.append(new_pos); labels.append((i, name, elem)); bonds.append((idx_c, len(coords) - 1))
                    sc_map[name] = len(coords) - 1

            p_O = self._nerf_step(coords[idx_N], coords[idx_CA], coords[idx_C], 1.23, 2.1, np.pi)
            coords.append(p_O); labels.append((i, 'O', 'O')); bonds.append((idx_C, len(coords) - 1))

            if i < self.n_residues - 1:
                psi      = angle_dict.get(f"{i}_psi", -0.5)
                p_nN     = self._nerf_step(coords[idx_N], coords[idx_CA], coords[idx_C], 1.33, 2.0, psi)
                coords.append(p_nN); labels.append((i + 1, 'N', 'N')); bonds.append((idx_C, len(coords) - 1))

                p_nCA    = self._nerf_step(coords[idx_CA], coords[idx_C], p_nN, 1.46, 2.1, np.pi)
                coords.append(p_nCA); labels.append((i + 1, 'CA', 'C')); bonds.append((len(coords) - 2, len(coords) - 1))

                phi      = angle_dict.get(f"{i + 1}_phi", -1.0)
                p_nC     = self._nerf_step(coords[idx_C], p_nN, p_nCA, 1.51, 1.9, phi)
                coords.append(p_nC); labels.append((i + 1, 'C', 'C')); bonds.append((len(coords) - 2, len(coords) - 1))

        return np.array(coords), labels, bonds

    # ── topology cache (unchanged) ──────────────────────────────────────────────
    def _initialize_topology_cache(self):
        dummy_coords, self.static_labels, _ = self.build_full_structure(np.zeros(self.total_angles))
        n_atoms = len(dummy_coords)
        self.atom_to_res      = np.array([x[0] for x in self.static_labels], dtype=int)
        self.atom_names       = np.array([x[1] for x in self.static_labels])
        self.atom_elems       = np.array([x[2] for x in self.static_labels])
        self.q_vector         = np.zeros(n_atoms)
        for k, (rid, name, elem) in enumerate(self.static_labels):
            q = self.CHARGES.get(name, 0.0)
            if self.sequence[rid] == 'H':
                if name == 'NE2': q = -0.4
                if name == 'ND1': q = -0.4
            if rid == 0 or rid == self.n_residues - 1:
                if name in ['N', 'CA', 'C', 'O', 'OXT', 'H1', 'H2', 'H3', 'H']:
                    q = 0.0
            self.q_vector[k] = q
        self.vdw_radii_vector   = np.array([self.VDW_RADII.get(x[2], 1.7) for x in self.static_labels])
        self.mask_heavy         = np.array([not x.startswith('H') for x in self.atom_names], dtype=bool)
        hydro_res_set           = set("AVLIMFWYPC")
        self.mask_hydrophobic   = np.zeros(n_atoms, dtype=bool)
        for k, (rid, name, elem) in enumerate(self.static_labels):
            aa = self.sequence[rid]
            if aa not in hydro_res_set:
                continue
            if name.startswith("C") and name not in ("C", "CA"):
                self.mask_hydrophobic[k] = True
            elif elem == "S":
                self.mask_hydrophobic[k] = True
        res_diff                = np.abs(self.atom_to_res[:, None] - self.atom_to_res[None, :])
        self.mask_non_bonded    = (res_diff >= 2)
        self.idx_N_atoms        = np.where(self.atom_names == 'N')[0]
        self.idx_O_atoms        = np.where(self.atom_names == 'O')[0]
        self.idx_SG_atoms       = np.where(self.atom_names == 'SG')[0]
        self._cache_initialized = True

    # ── energy function (always uses statevector internally) ───────────────────
    def energy_function(self, params, return_terms: bool = False):
        """
        Objective function called by the classical optimizer.
        Always uses statevector mode — fast and exact.
        """
        if not self._cache_initialized:
            self._initialize_topology_cache()

        gamma               = 15.0
        constraint_strength = 50.0
        if self.current_stage == 3:
            gamma               = 2.5
            constraint_strength = 5.0

        # Always statevector during optimisation
        angle_vec          = self._get_angles(params, mode="statevector")
        coords, _, _       = self.build_full_structure(angle_vec)

        terms        = {k: 0.0 for k in ["constraint","sasa","hbond","hbond_raw",
                                           "electrostatics","disulfide","vdw_repulsion",
                                           "vdw_attractive","rotamer","pi_stacking",
                                           "rama","geometry"]}
        total_energy = 0.0

        def add_term(name, value):
            nonlocal total_energy
            v = float(value); terms[name] += v; total_energy += v

        diffs = coords[:, None, :] - coords[None, :, :]
        D     = np.sqrt(np.sum(diffs**2, axis=-1)) + 1e-9

        neighbor_counts  = np.array([0.0])
        burial_fractions = np.array([0.0])

        # end-to-end constraint
        ca_idx = [i for i, l in enumerate(self.static_labels) if l[1] == 'CA']
        if len(ca_idx) >= 2:
            d_e2e = np.linalg.norm(coords[ca_idx[0]] - coords[ca_idx[-1]])
            add_term("constraint", constraint_strength * (d_e2e - 5.5) ** 2)

        # SASA
        if np.sum(self.mask_hydrophobic) > 0:
            hd             = D[self.mask_hydrophobic, :]
            w              = 1.0 / (1.0 + np.exp(1.0 * (hd - 6.0)))
            neighbor_counts   = np.sum(w, axis=1) - 1.0
            burial_fractions  = np.clip(neighbor_counts / 35.0, 0.0, 1.0)
            SASA_SCALE     = float(os.getenv("QTF_SASA_SCALE", "0.7"))
            add_term("sasa", SASA_SCALE * np.sum(gamma * 30.0 * (1.0 - burial_fractions)))

        # H-bonds
        HBOND_SCALE = float(os.getenv("QTF_HBOND_SCALE", "0.75"))
        e_hbond = 0.0
        for i_n in self.idx_N_atoms:
            res_d     = self.atom_to_res[i_n]
            idx_ca    = i_n + 1
            idx_prev_c = i_n - 2
            if idx_prev_c < 0 or self.atom_names[idx_prev_c] != 'C':
                pos_h = coords[i_n] + np.array([0, 0, 1.0]); pos_n = coords[i_n]
            else:
                p_c = coords[idx_prev_c]; p_n = coords[i_n]; p_ca = coords[idx_ca]
                v_nc  = p_c - p_n;  v_nc  /= np.linalg.norm(v_nc)
                v_nca = p_ca - p_n; v_nca /= np.linalg.norm(v_nca)
                v_h   = -(v_nc + v_nca); v_h /= np.linalg.norm(v_h)
                pos_h = p_n + v_h * 1.01; pos_n = p_n
            o_coords   = coords[self.idx_O_atoms]
            o_res      = self.atom_to_res[self.idx_O_atoms]
            valid_mask = np.abs(o_res - res_d) >= 2
            if not np.any(valid_mask): continue
            d_ho       = np.linalg.norm(o_coords[valid_mask] - pos_h, axis=1)
            close_mask = d_ho < 3.5
            if not np.any(close_mask): continue
            f_d_ho     = d_ho[close_mask]
            f_o        = o_coords[valid_mask][close_mask]
            v_hn       = pos_n - pos_h; v_hn /= np.linalg.norm(v_hn)
            v_ho       = f_o - pos_h;   v_ho /= np.linalg.norm(v_ho, axis=1)[:, None]
            ac         = np.dot(v_ho, v_hn)
            am         = ac < -0.4
            e_hbond   += np.sum(-50.0 * np.exp(-(f_d_ho - 2.0) ** 2 / 0.5)
                                * (np.abs(ac) - 0.4) * 2.0 * am)
        terms["hbond_raw"] = float(e_hbond)
        add_term("hbond", HBOND_SCALE * e_hbond)

        # Electrostatics
        Q_mat     = np.outer(self.q_vector, self.q_vector)
        elec_mask = np.triu(self.mask_non_bonded, k=1) & (np.abs(Q_mat) > 0.0001)
        if np.any(elec_mask):
            r_e = np.maximum(D[elec_mask], 1.0)
            add_term("electrostatics", np.sum(83.0 * Q_mat[elec_mask] / r_e ** 2))

        # Disulfide
        e_dis = 0.0
        if len(self.idx_SG_atoms) > 1:
            sg_d  = D[np.ix_(self.idx_SG_atoms, self.idx_SG_atoms)]
            sg_m  = np.triu(np.ones_like(sg_d, dtype=bool), k=1)
            vd    = sg_d[sg_m]
            bs    = np.exp(-(vd - 2.05) ** 2 / 0.5)
            e_dis -= np.sum(25.0 * bs * (vd < 3.0))
            full  = np.exp(-(sg_d - 2.05) ** 2 / 0.5) * (sg_d < 3.0)
            np.fill_diagonal(full, 0.0)
            ov    = np.sum(full, axis=1) - 1.0
            pm    = ov > 0.1
            if np.any(pm):
                e_dis += np.sum(40.0 * ov[pm] ** 2)
        add_term("disulfide", e_dis)

        # VdW
        Sig      = self.vdw_radii_vector[:, None] + self.vdw_radii_vector[None, :]
        hm       = self.mask_heavy[:, None] & self.mask_heavy[None, :]
        vdw_mask = np.triu(self.mask_non_bonded & hm, k=1)
        VDW_REP  = float(os.getenv("QTF_VDW_REP_SCALE",  "0.1"))
        VDW_ATT  = float(os.getenv("QTF_VDW_ATTR_SCALE", "0.1"))
        if np.any(vdw_mask):
            rv = D[vdw_mask]; sv = Sig[vdw_mask]; x = sv / (rv + 1e-9)
            cm = rv < sv
            if np.any(cm):
                rt = x[cm] ** 12; he = rt > 50.0
                if np.any(he): rt[he] = 50.0 + np.log(rt[he] - 49.0)
                add_term("vdw_repulsion", np.sum(VDW_REP * rt))
            nm = (rv >= sv) & (rv < 1.5 * sv)
            if np.any(nm):
                at = np.maximum(-(x[nm] ** 6), -2.0)
                add_term("vdw_attractive", np.sum(VDW_ATT * at))

        angle_dict = {f"{x['res']}_{x['type']}": val
                      for x, val in zip(self.dof_map, angle_vec)}
        add_term("rotamer",    float(os.getenv("QTF_ROTAMER_SCALE", "1.0")) * self._calculate_rotamer_energy(angle_dict))
        add_term("pi_stacking",float(os.getenv("QTF_PI_STACK_SCALE","1.0")) * self._calculate_aromatic_quadrupole(coords, self.static_labels, self.atom_to_res))

        e_rama = 0.0
        for i in range(self.n_residues):
            if f"{i}_phi" in angle_dict and f"{i}_psi" in angle_dict:
                phi = angle_dict[f"{i}_phi"]; psi = angle_dict[f"{i}_psi"]
                aa  = self.sequence[i]
                dh  = (phi - (-1.0)) ** 2 + (psi - (-0.8)) ** 2
                ds  = (phi - (-2.3)) ** 2 + (psi - (2.4)) ** 2
                if aa == 'G':
                    dhl = (phi - 1.0) ** 2 + (psi - 0.8) ** 2
                    dsl = (phi - 2.3) ** 2 + (psi - (-2.4)) ** 2
                    e_rama += -3.0 * np.exp(-min(dh, ds, dhl, dsl) / 0.6)
                else:
                    df = (phi - (-2.0)) ** 2 + (psi - 1.0) ** 2
                    e_rama += (-3.0 * np.exp(-dh / 0.6) - 3.0 * np.exp(-ds / 0.6)
                               + 5.0 * np.exp(-df / 1.0))
        add_term("rama", e_rama)

        e_geom, geom_sub = self._calculate_geometry_integrity(
            coords, self.static_labels, self.atom_to_res, return_terms=True)
        add_term("geometry", e_geom)

        if self.tracker:
            self.tracker.log(total_energy)

        if return_terms:
            self.last_energy_terms = {
                **terms,
                "geom_pro_ring":  float(geom_sub["pro_ring"]),
                "geom_chirality": float(geom_sub["chirality"]),
                "geom_planarity": float(geom_sub["planarity"]),
                "burial_mean":    float(np.mean(burial_fractions)) if np.sum(self.mask_hydrophobic) > 0 else 0.0,
                "burial_min":     float(np.min(burial_fractions))  if np.sum(self.mask_hydrophobic) > 0 else 0.0,
                "burial_max":     float(np.max(burial_fractions))  if np.sum(self.mask_hydrophobic) > 0 else 0.0,
                "neighbor_mean":  float(np.mean(neighbor_counts))  if np.sum(self.mask_hydrophobic) > 0 else 0.0,
                "total":          float(total_energy),
            }
        return total_energy

    # ── rotamer, pi-stack, geometry (unchanged) ─────────────────────────────────
    def _calculate_rotamer_energy(self, angle_dict):
        energy = 0.0
        def wd(a, b): return (a - b + np.pi) % (2 * np.pi) - np.pi
        cc = [-1.0471975512, 1.0471975512, 3.1415926536]
        for i in range(self.n_residues):
            rn = self.sequence[i]
            k1 = f"{i}_chi1"
            if k1 in angle_dict:
                chi = angle_dict[k1]
                if rn in ['V','I','T']:
                    energy += -3.0*(np.exp(-wd(chi,np.pi)**2/0.5)+np.exp(-wd(chi,-1.0471975512)**2/0.5))
                elif rn == 'P':
                    energy += 10.0 * min(wd(chi,-0.5)**2, wd(chi,0.5)**2)
                elif rn in ['W','F','Y','H']:
                    energy += -2.0*(np.exp(-wd(chi,np.pi)**2/0.45)
                                    +0.8*np.exp(-wd(chi,-1.0471975512)**2/0.45)
                                    +0.8*np.exp(-wd(chi, 1.0471975512)**2/0.45))
                else:
                    energy += 1.0*(1.0+np.cos(3.0*chi))
            for ci in (2,3,4,5):
                kn = f"{i}_chi{ci}"
                if kn not in angle_dict: continue
                chi = angle_dict[kn]
                if ci == 2 and rn in ['W','F','Y','H']:
                    energy += -1.5*sum(np.exp(-wd(chi,c)**2/0.35) for c in cc)
                else:
                    energy += -0.75*sum(np.exp(-wd(chi,c)**2/0.50) for c in cc)
        return energy

    def _calculate_aromatic_quadrupole(self, coords, labels, atom_to_res_idx):
        aromatics = []
        for r_idx in np.unique(atom_to_res_idx):
            if self.sequence[r_idx] in ['F','Y','W']:
                mask       = (atom_to_res_idx == r_idx)
                r_names    = self.atom_names[mask]
                ring_mask  = np.isin(r_names, ['CG','CD1','CD2','CE1','CE2','CZ'])
                ring_atoms = coords[mask][ring_mask]
                if len(ring_atoms) > 2:
                    centroid = np.mean(ring_atoms, axis=0)
                    v1 = ring_atoms[1]-ring_atoms[0]; v2 = ring_atoms[2]-ring_atoms[0]
                    n  = np.cross(v1,v2); n /= (np.linalg.norm(n)+1e-9)
                    aromatics.append((centroid, n))
        ep = 0.0
        for i in range(len(aromatics)):
            for j in range(i+1, len(aromatics)):
                c1,n1 = aromatics[i]; c2,n2 = aromatics[j]
                d     = np.linalg.norm(c1-c2)
                if d > 7.0: continue
                al    = abs(np.dot(n1,n2))
                if al < 0.3 and 4.5 < d < 6.0: ep -= 4.0*np.exp(-(d-5.0)**2)
                elif al > 0.8 and 3.4 < d < 4.5: ep -= 5.0*np.exp(-(d-3.8)**2)
        return ep

    def _calculate_geometry_integrity(self, coords, labels, atom_to_res_idx, return_terms=False):
        energy = 0.0
        gt     = {"pro_ring": 0.0, "chirality": 0.0, "planarity": 0.0}
        res_map = {}
        for k, lbl in enumerate(labels):
            res_map.setdefault(lbl[0], {})[lbl[1]] = k
        for r in range(self.n_residues):
            atoms = res_map.get(r, {}); rn = self.sequence[r]
            if rn == 'P' and 'CD' in atoms and 'N' in atoms:
                d = np.linalg.norm(coords[atoms['CD']]-coords[atoms['N']])
                if abs(d-1.47) > 0.1:
                    p = 50.0*(d-1.47)**2; energy += p; gt["pro_ring"] += p
            if all(k in atoms for k in ('CA','N','C','CB')):
                ca = coords[atoms['CA']]; n = coords[atoms['N']]
                c  = coords[atoms['C']];  cb = coords[atoms['CB']]
                vol = np.dot(np.cross(n-ca, c-ca), cb-ca)
                if vol < 1.0:
                    p = 50.0*(1.0-vol)**2; energy += p; gt["chirality"] += p
            if r < self.n_residues-1:
                na = res_map.get(r+1, {})
                if all(k in atoms for k in ('C','CA')) and all(k in na for k in ('N','CA')):
                    p1,p2,p3,p4 = (coords[atoms['CA']], coords[atoms['C']],
                                   coords[na['N']], coords[na['CA']])
                    b1=p2-p1; b2=p3-p2; b3=p4-p3
                    n1=np.cross(b1,b2); n2=np.cross(b2,b3)
                    nn1=np.linalg.norm(n1); nn2=np.linalg.norm(n2)
                    if nn1>1e-8 and nn2>1e-8:
                        par = np.dot(n1/nn1, n2/nn2)
                        tp  = 1.0 - abs(par)
                        if tp > 0.05:
                            p = 20.0*tp; energy += p; gt["planarity"] += p
        return (energy, gt) if return_terms else energy

    def fold(self, max_iter=2000, initial_params=None,
            hw_backend=None, hw_shots=4096,
            true_ca=None, skip_stage2=False):

        print(f"--- STARTING QUANTUM FOLDING (HYBRID PIPELINE) ---")
        self.tracker = LandscapeTracker()

        if initial_params is None:
            init_params = self.get_smart_initialization()
        else:
            init_params = initial_params

        # ── Helper: compute RMSD at current params ──────────────────────────────
        def _get_rmsd(params):
            if true_ca is None:
                return None
            angles       = self._get_angles(params, mode="statevector")
            coords, _, _ = self.build_full_structure(angles)
            pred_ca      = np.array([
                coords[i] for i, lbl in enumerate(self.static_labels)
                if lbl[1] == 'CA'
            ])
            n       = min(len(pred_ca), len(true_ca))
            rmsd, _ = StabilityAnalyzer.kabsch_rmsd(pred_ca[:n], true_ca[:n])
            return float(rmsd)

        # ── Stage 1: Collapse (COBYLA) ──────────────────────────────────────────
        print("Stage 1: Mechanical Collapse (COBYLA, statevector)...")
        self.tracker.mark_stage("Stage1"); self.current_stage = 1
        res_1  = minimize(self.energy_function, init_params,
                        method='COBYLA',
                        options={'maxiter': max_iter, 'rhobeg': 1.0})
        rmsd_s1 = _get_rmsd(res_1.x)
        print(f" > Collapse Energy : {res_1.fun:.2f}")
        if rmsd_s1: print(f" > Stage 1 RMSD    : {rmsd_s1:.4f} Å")

        # ── Stage 2: Refine (SLSQP) — optional ─────────────────────────────────
        if skip_stage2:
            print("Stage 2: SKIPPED")
            res_2   = res_1
            rmsd_s2 = None
        else:
            print("Stage 2: Physics Refinement (SLSQP, statevector)...")
            self.tracker.mark_stage("Stage2"); self.current_stage = 2
            res_2   = minimize(self.energy_function, res_1.x,
                            method='SLSQP', tol=1e-6,
                            options={'maxiter': max_iter, 'disp': True})
            rmsd_s2 = _get_rmsd(res_2.x)
            print(f" > Refinement Energy : {res_2.fun:.2f}")
            if rmsd_s2: print(f" > Stage 2 RMSD      : {rmsd_s2:.4f} Å")

        # ── Stage 3: Relax (SLSQP) ─────────────────────────────────────────────
        print("Stage 3: Natural Relaxation (SLSQP, statevector)...")
        self.tracker.mark_stage("Stage3"); self.current_stage = 3
        res_3   = minimize(self.energy_function, res_2.x,
                        method='SLSQP', tol=1e-6,
                        options={'maxiter': max_iter, 'disp': True})
        rmsd_s3 = _get_rmsd(res_3.x)
        print(f" > Relaxation Energy : {res_3.fun:.2f}")
        if rmsd_s3: print(f" > Stage 3 RMSD      : {rmsd_s3:.4f} Å")

        # ── RMSD progress summary ───────────────────────────────────────────────
        if true_ca is not None:
            print(f"\n── RMSD Progress ──────────────────────────")
            print(f"  Stage 1 : {rmsd_s1:.4f} Å" if rmsd_s1 else "  Stage 1 : N/A")
            if not skip_stage2:
                arrow = '↓ better' if rmsd_s2 and rmsd_s1 and rmsd_s2 < rmsd_s1 else '↑ worse'
                print(f"  Stage 2 : {rmsd_s2:.4f} Å  {arrow}" if rmsd_s2 else "  Stage 2 : N/A")
            else:
                print(f"  Stage 2 : skipped")
            arrow = '↓ better' if rmsd_s3 and rmsd_s2 and rmsd_s3 < rmsd_s2 else '↓ better' if rmsd_s3 and rmsd_s1 and rmsd_s3 < rmsd_s1 else '↑ worse'
            print(f"  Stage 3 : {rmsd_s3:.4f} Å  {arrow}" if rmsd_s3 else "  Stage 3 : N/A")
            print(f"──────────────────────────────────────────")

        # ── Store stage RMSDs on folder for retrieval in single_replica.py ──────
        self.stage_rmsds = {'s1': rmsd_s1, 's2': rmsd_s2, 's3': rmsd_s3}

        optimal_params = res_3.x

        # ── Final angle extraction ──────────────────────────────────────────────
        if hw_backend is None:
            print("\n[EXTRACTION] Mode: statevector (classical exact)")
            final_angles = self._get_angles(optimal_params, mode="statevector")
        else:
            backend_name = getattr(hw_backend, 'name', str(hw_backend))
            print(f"\n[EXTRACTION] Mode: sampler → backend='{backend_name}', shots={hw_shots}")
            print("  Submitting 3 circuits (Z / X / Y bases)...")
            final_angles = self._get_angles(optimal_params, mode="sampler",
                                            backend=hw_backend, shots=hw_shots)
            print("  Done. Angles extracted from measurement statistics.")

        coords, labels, bonds = self.build_full_structure(final_angles)
        print(f"\n[RESULT] Final energy: {res_3.fun:.4f}")
        return coords, labels, bonds, self.tracker, optimal_params, res_3.fun

    # ── smart init ──────────────────────────────────────────────────────────────
    def get_smart_initialization(self, n_attempts=20, seed=None):
        """
        Samples random parameters to find a good starting point (Basin Hopping).
        This avoids getting stuck in high-energy states immediately.
        
        REPRODUCIBILITY:
        Uses a hash of the sequence to seed the random number generator. 
        This ensures that every run with the same sequence starts from the same 
        initial geometry, allowing you to test energy function changes reliably.
        """
        if seed is None:
            # Create a deterministic seed from the protein sequence
            seed = int(hashlib.sha256(self.sequence.encode('utf-8')).hexdigest(), 16) % (2**32)
        
        rng = np.random.default_rng(seed)
        
        print(f"--- SCOUTING: Checking {n_attempts} starting points ---")
        print(f" > Deterministic Seed: {seed} (Derived from Sequence)")
        
        best_params = None
        best_energy = float('inf')
        for i in range(n_attempts):
            #trial_params = rng.uniform(-0.8, 0.8, self.n_params)
            trial_params = rng.uniform(-np.pi, np.pi, self.n_params)
            e = self.energy_function(trial_params)
            if e < best_energy:
                best_energy = e
                best_params = trial_params
        print(f" > Best Start Found: Energy {best_energy:.2f}")
        return best_params

    # ── PDB / centroid savers (unchanged) ───────────────────────────────────────
    def compute_sidechain_centroids(self, coords, labels):
        backbone = {'N','CA','C','O','OXT'}
        by_res   = {}
        for pos, (rid, name, elem) in zip(coords, labels):
            if name in backbone or name.startswith('H') or elem == 'H':
                continue
            by_res.setdefault(int(rid), []).append(np.asarray(pos, dtype=float))
        return {rid: np.mean(np.vstack(pts), axis=0)
                for rid, pts in by_res.items() if pts}

    def save_pdb(self, coords, labels, filename="structure.pdb", energy=0.0):
        with open(filename, 'w') as f:
            f.write(f"REMARK   1 ENERGY: {energy:.3f}\n")
            for k, (pos, (rid, name, elem)) in enumerate(zip(coords, labels)):
                rn = self.sequence[rid]
                f.write(f"ATOM  {k+1:>5}  {name:<4} {rn:>3} A{rid+1:>4}    "
                        f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           {elem:>2}\n")

    def save_reduced_pdb(self, ca_coords, filename="reduced.pdb",
                         sidechain_centroids=None, energy=0.0):
        with open(filename, 'w') as f:
            f.write(f"REMARK   1 REDUCED MODEL - CA ONLY\nREMARK   2 ENERGY: {energy:.3f}\n")
            k = 1
            for rid, pos in enumerate(ca_coords):
                rn = self.sequence[rid] if rid < len(self.sequence) else 'UNK'
                f.write(f"ATOM  {k:>5}  CA  {rn:>3} A{rid+1:>4}    "
                        f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           C\n")
                k += 1
            if sidechain_centroids:
                for rid, pos in sorted(sidechain_centroids.items()):
                    rn = self.sequence[rid] if rid < len(self.sequence) else 'UNK'
                    f.write(f"ATOM  {k:>5}  SC  {rn:>3} A{rid+1:>4}    "
                            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00           C\n")
                    k += 1
            f.write("END\n")


# ==============================================================================
# 4. ORCHESTRATOR: ENSEMBLE MANAGER
# ==============================================================================
class EnsembleFoldingManager:
    """
    Manages multiple independent folding replicas.

    hw_backend and hw_shots are forwarded to folder.fold() so that
    each replica's final angles are extracted from the same target backend.
    """

    def __init__(self, folder_instance):
        self.folder  = folder_instance
        self.results = []

    def prime_circuit(self, target_type='helix', seed=42):
        print(f"--- PRIMING CIRCUIT FOR {target_type.upper()} ---")
        rng = np.random.default_rng(seed)
        if target_type == 'helix':
            t_phi, t_psi = np.deg2rad(-60.0), np.deg2rad(-45.0)
        elif target_type == 'sheet':
            t_phi, t_psi = np.deg2rad(-135.0), np.deg2rad(135.0)
        else:
            #return rng.uniform(-0.8, 0.8, self.folder.n_params)
            return rng.uniform(-np.pi, np.pi, self.folder.n_params)

        targets = np.zeros(self.folder.total_angles)
        masks   = np.zeros(self.folder.total_angles)
        for i, dof in enumerate(self.folder.dof_map):
            if dof['type'] == 'phi':  targets[i] = t_phi;  masks[i] = 1.0
            elif dof['type'] == 'psi': targets[i] = t_psi; masks[i] = 1.0

        def cost(params):
            curr = self.folder._get_angles(params, mode="statevector")
            diff = (curr - targets + np.pi) % (2 * np.pi) - np.pi
            return np.sum((diff * masks) ** 2)

        init = rng.uniform(-0.1, 0.1, self.folder.n_params)
        res  = minimize(cost, init, method='COBYLA', options={'maxiter': 200})
        print(f" > Priming Error: {res.fun:.4f}")
        return res.x

    def run_ensemble(self, n_runs=3, max_iter=200, prime_strategy='mixed',
                     hw_backend=None, hw_shots=4096):
        """
        Parameters
        ----------
        hw_backend : Qiskit backend or None
            Passed to each replica's fold() call for final angle extraction.
            None        → statevector (classical only, no hardware)
            AerSimulator() → noisy simulation (test mode)
            real IBM backend → actual hardware
        hw_shots   : int — shots per basis circuit on hardware
        """
        print(f"=== STARTING ENSEMBLE RUN ({n_runs} Trajectories) ===")
        if hw_backend is None:
            print("    Angle extraction: statevector (classical)")
        else:
            bname = getattr(hw_backend, 'name', str(hw_backend))
            print(f"    Angle extraction: sampler on '{bname}' ({hw_shots} shots)")

        self.results = []
        seed = int(hashlib.sha256(
            self.folder.sequence.encode()).hexdigest(), 16) % (2**32)

        for i in range(n_runs):
            print(f"\n>> REPLICA {i+1}/{n_runs}")
            if prime_strategy == 'mixed':
                strat = ['helix','sheet','random'][i % 3]
            else:
                strat = prime_strategy.lower()

            if strat == 'random':
                start = self.folder.get_smart_initialization(n_attempts=50, seed=seed+i)
            else:
                start = self.prime_circuit(target_type=strat, seed=seed+i)

            coords, _, _, tracker, opt_params, final_e = self.folder.fold(
                max_iter=max_iter, initial_params=start,
                hw_backend=hw_backend, hw_shots=hw_shots)

            print(f" >> Replica {i+1} Final Energy: {final_e:.2f}")
            self.results.append({
                'id': i, 'seed': seed+i, 'type': strat,
                'energy': final_e, 'coords': coords,
                'params': opt_params, 'tracker': tracker,
            })

    # ── selection helpers (unchanged) ───────────────────────────────────────────
    def evaluate_best(self):
        if not self.results:
            return None
        sorted_r = sorted(self.results, key=lambda x: x['energy'])
        best     = sorted_r[0]
        print(f"\n=== ENSEMBLE EVALUATION ===")
        print(f"Best Candidate: ID {best['id']} (Init: {best['type']})")
        print(f"Lowest Energy:  {best['energy']:.4f}")
        if len(self.results) > 1:
            StabilityAnalyzer.analyze_convergence(self.results)
        return best

    def get_ranked_results(self):
        return sorted(self.results, key=lambda x: x['energy'])

    def select_top(self, top_k=None, top_frac=None):
        ranked = self.get_ranked_results()
        if not ranked:
            return []
        if top_frac is not None:
            return ranked[:max(1, int(np.ceil(len(ranked) * float(top_frac))))]
        if top_k is not None:
            return ranked[:max(1, min(int(top_k), len(ranked)))]
        return ranked
    
    def select_best_rmsd(self, true_ca, folder):
        """
        Computes RMSD of every replica against ground truth CA coords,
        returns results sorted by RMSD (lowest first).
        
        Parameters
        ----------
        true_ca : np.ndarray  — ground truth CA coordinates
        folder  : QuantumBiophysicsFolder — to extract CA from coords
        """
        print(f"\n=== RMSD RANKING (All {len(self.results)} Replicas vs Ground Truth) ===")
        print(f"{'Replica':>8} {'Init':>8} {'Energy':>10} {'RMSD (Å)':>10}")
        print("-" * 45)

        for res in self.results:
            coords  = res['coords']
            pred_ca = np.array([
                coords[i] for i, lbl in enumerate(folder.static_labels)
                if lbl[1] == 'CA'
            ])
            n = min(len(pred_ca), len(true_ca))
            rmsd, _ = StabilityAnalyzer.kabsch_rmsd(pred_ca[:n], true_ca[:n])
            res['rmsd'] = float(rmsd)

        ranked = sorted(self.results, key=lambda x: x['rmsd'])

        for res in ranked:
            print(f"  #{res['id']:>4}    {res['type']:>8}   {res['energy']:>10.2f}   {res['rmsd']:>8.3f} Å")

        best = ranked[0]
        print(f"\n  Best RMSD : {best['rmsd']:.3f} Å  (Replica #{best['id']}, "
            f"init={best['type']}, energy={best['energy']:.2f})")
        print(f"  Best Energy: {sorted(self.results, key=lambda x: x['energy'])[0]['energy']:.2f} "
            f"(may differ from best RMSD replica)")
        print("=" * 45)

        return ranked