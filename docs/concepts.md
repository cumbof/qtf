# Core Concepts

This page explains the three main ideas that make QTF work: the **quantum-to-angle mapping**,
**NERF geometry**, and the **three-stage optimisation**.

---

## Encoding

### The qubit-scaling problem

Traditional quantum approaches to protein folding assign one bit (or one qubit) per
degree of freedom. A 10-residue chain has ≈30 dihedral angles, which would need 30 qubits
with discrete grid-points — or far more for any useful angular resolution.

QTF instead exploits the fact that a quantum register of **k** qubits has **2ᵏ** complex
amplitudes: the **phases** of those amplitudes form a continuous 2ᵏ-dimensional real
vector, all reachable through a single quantum circuit.

A 10-residue protein with 30 DoF therefore needs only ⌈log₂ 30⌉ = **5 qubits**.

### Degrees of freedom per residue

| Residue type | φ | ψ | χ₁ | χ₂ | χ₃ | χ₄ | χ₅ | Total DoF |
|:-------------|:-:|:-:|:--:|:--:|:--:|:--:|:--:|:---------:|
| Glycine      | ✓ | ✓ | —  | —  | —  | —  | —  | 2 |
| Alanine      | ✓ | ✓ | —  | —  | —  | —  | —  | 2 |
| Serine       | ✓ | ✓ | ✓  | —  | —  | —  | —  | 3 |
| Most residues| ✓ | ✓ | ✓  | ✓  | —  | —  | —  | 4 |
| Arg / Lys    | ✓ | ✓ | ✓  | ✓  | ✓  | ✓  | —  | 6 |
| Arg only     | ✓ | ✓ | ✓  | ✓  | ✓  | ✓  | ✓  | 7 |

### Phase extraction

The `EfficientSU2` ansatz from Qiskit is parameterised by a vector **θ**. Given **θ**,
the quantum circuit is simulated exactly (statevector simulation — no shot noise) and the
phase of each amplitude is extracted:

```
angle_i = arg( ⟨ψ(θ)|computational basis state i⟩ )
```

Phases fall in [−π, π], which maps naturally to dihedral angles in the same range.

---

## NERF Geometry Builder

Once the torsion angles are known, the 3-D Cartesian coordinates of every atom are
reconstructed using **Natural Extension Reference Frame (NERF)**.

NERF places atoms one at a time. Given three already-placed atoms **A**, **B**, **C** and
the next bond length, bond angle, and dihedral angle, the position of atom **D** is:

```
D = M·[d·cos(α), d·sin(α)·cos(τ), d·sin(α)·sin(τ)] + C
```

where **M** is the local reference frame built from **A**, **B**, **C**.

Backbone bond lengths and angles use the QTF geometry tables and Amber-style custom parameterization rather than multiple selectable custom force fields.

---

## Three-Stage Optimisation

QTF uses classical optimisers to minimise the physics energy **E(θ)** with respect to
the circuit parameters **θ** in three successive stages:

| Stage | Algorithm | Purpose |
|:------|:----------|:--------|
| 1 — Collapse | COBYLA (SciPy) | Derivative-free, coarse-grained collapse of the chain — fast convergence from arbitrary initial angles |
| 2 — Refine  | SLSQP (SciPy) | Gradient-based tightening of secondary structure, guided by Ramachandran and H-bond terms |
| 3 — Relax   | SLSQP (SciPy) | Fine-scale relaxation of side chains and sterics, with all ten energy terms active |

### Basin-hopping initialisation

Before the three stages, QTF performs **basin-hopping**: `scout_attempts` random
parameter vectors are evaluated with a single energy call each, and the lowest-energy
parameter vector is selected as the starting point. This dramatically improves the chance
of escaping bad local minima on the first stage.

### Trajectory logging

A `LandscapeTracker` instance records (energy, iteration, optimiser) at every objective
call. The logged data drives the energy-landscape visualisation.

---

## Ensemble Folding

`EnsembleFoldingManager` runs **N independent replicas**, each with:

- a **deterministic, reproducible random seed** derived from SHA-256(sequence + replica_index)
- its own basin-hopping initialisation and three-stage optimisation
- a separate `LandscapeTracker`

Because seeds are deterministic from the sequence string, any result is fully reproducible.

The `get_results()` method returns a list of result dicts, one per completed replica.
The dict schema is:

| Key           | Type                  | Description |
|:--------------|:----------------------|:------------|
| `id`          | `int`                 | Zero-based replica index |
| `energy`      | `float`               | Final energy value |
| `angles`      | `numpy.ndarray`       | All dihedral angles in radians |
| `coords`      | `numpy.ndarray`       | All-atom Cartesian coordinates (Å) |
| `labels`      | `list[str]`           | Atom labels, one per row of `coords` |
| `tracker`     | `LandscapeTracker`    | Full optimisation trajectory |
| `seed`        | `int`                 | Integer seed used for this replica |
| `converged`   | `bool`                | `True` if the final stage reported convergence |
