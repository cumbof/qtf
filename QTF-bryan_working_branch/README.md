# QTF: Quantum Torsion Folder
Logarithmic-scale variational quantum eigensolver for torsion-space off-lattice protein structure prediction

__QTF__ (Quantum Torsion Folder) is a hybrid quantum-classical tool designed to predict protein conformations by optimizing torsion angles ($\phi, \psi, \chi$) directly in continuous space. Unlike traditional lattice-based quantum folding methods (which snap atoms to a grid), QTF uses a Holographic Encoding strategy to map $N$ continuous degrees of freedom into $\mathcal{O}(\log N)$ qubits, enabling the simulation of complex backbones and side chains on near-term quantum hardware.

## Key Features

- __Holographic Scaling__: Compresses the search space by mapping quantum statevector phases to physical torsion angles. This allows a small number of qubits to represent complex proteins with hundreds of degrees of freedom.

- __Off-Lattice Geometry__: Uses the NERF (Natural Extension Reference Frame) algorithm to build structures in continuous 3D Cartesian space, avoiding the artifacts of cubic/tetrahedral grids.

- __Full All-Atom Reconstruction__: Explicitly models backbone atoms ($N, C_\alpha, C, O$) and all side-chain atoms (including rotamers) based on IUPAC topology standards.

- __Multi-Force Field Support__: Includes coarse-grained approximations of CHARMM, AMBER, and OPLS partial charge sets.

## Installation

QTF requires Python 3.8+ and standard scientific quantum libraries.

```text
pip install numpy scipy matplotlib qiskit
```

Clone the repository:

```
git clone [https://github.com/cumbof/QTF.git](https://github.com/cumbof/QTF.git)
cd QTF
```
## Quick Start

Predicting the structure of a small peptide (e.g., Chignolin segment or custom sequence) takes just a few lines of code.

```python
# 1. Initialize the folder with a sequence and force field
# Options for force_field: 'charmm' (default), 'amber', 'opls'
folder = QuantumBiophysicsFolder(equence="MAGVLS", force_field="amber")

# 2. Run the folding simulation
# The method uses a multi-stage curriculum (Collapse -> Refine -> Relax)
coords, labels, bonds, _, _, _ = folder.fold(max_iter=2000)

# 3. Save the result to PDB
# Viewable in PyMOL, Chimera, or VMD
folder.save_pdb(coords, labels, filename="prediction_amber.pdb")
```

## Physics Engine & Methodology

QTF uses a classical energy function to guide a VQE Ansatz. The energy landscape is constructed from a blend of physical potentials and algorithmic heuristics:

1. Quantum Generator
   - __Ansatz__: `efficient_su2` (Hardware Efficient Ansatz) with circular entanglement.
   - __Mapping__: The phases of the complex amplitudes in the output statevector are extracted and mapped to the range $[-\pi, \pi]$ to serve as torsion angles.

2. Biophysical Force Field

The total energy $E_{total}$ is minimized by the classical optimizer (COBYLA/SLSQP):

Component | Description
----------|------------
Solvation | Implicit solvent model based on burial fraction (SASA).
Electrostatics | Coulomb interactions with force-field specific partial charges.
H-Bonding | Vectorized orientation-dependent potential ("Super-Glue").
Sterics | Logarithmically softened Lennard-Jones repulsion.
Ramachandran | Gaussian energy wells favoring Helix/Sheet regions.
Magnet Bias | Heuristic: Artificial harmonic constraint to force globular collapse.

## Reproducibility

To compare different settings (e.g., changing the Force Field), QTF guarantees that the initial random distribution is deterministic based on the input.

```python
# These two runs will start from the EXACT same geometry
run1 = QuantumBiophysicsFolder("MVL", force_field="charmm").get_smart_initialization(n_attempts=50)
run2 = QuantumBiophysicsFolder("MVL", force_field="opls").get_smart_initialization(n_attempts=50)
```

## License

QTF is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
