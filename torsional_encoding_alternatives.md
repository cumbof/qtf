# Torsional Encoding Alternatives

QTF currently uses phase encoding, which can be costly on quantum hardware. For torsional encoding, the main alternatives are ways to represent each torsion with fewer fragile phase degrees of freedom, or to move more of the representation into classical parameters while the quantum circuit samples/searches over them.

| Encoding | Basic idea | Why it may scale better | Main tradeoff |
|---|---|---|---|
| Angle/rotation encoding | Encode torsion values directly as parameterized `RY`, `RZ`, or `RX` rotations | Native to hardware; shallow circuits; no need to recover phases from amplitudes | Optimizer still controls continuous parameters; quantum state may not add much unless circuit correlations matter |
| Discrete bin / binary encoding | Represent each torsion as one of `N` allowed bins using `log2(N)` qubits | Natural for torsions with rotamer-like states; hardware readout is simple bitstrings | Resolution limited by bin count; many torsions still scale as `O(torsions * log bins)` |
| Rotamer/state encoding | Encode allowed residue-specific torsion states rather than arbitrary angles | Protein-aware; dramatically reduces search space for sidechains | Less flexible; needs good residue/rotamer libraries |
| Fourier / sine-cosine encoding | Encode torsion through periodic features like `sin(theta)`, `cos(theta)`, or low Fourier modes | Respects periodicity; avoids angle wrap discontinuities | More classical-feature-like; may need multiple parameters per torsion |
| QAOA-style discrete torsion search | Treat torsion bins/rotamers as combinatorial variables and optimize a cost Hamiltonian | More aligned with quantum hardware sampling; measurements produce candidate structures directly | Requires meaningful mixers/constraints; continuous refinement likely still classical |
| Hybrid coarse-to-fine encoding | Use coarse discrete bins quantumly, then classically/GROMACS/OpenMM refine | Scalable and pragmatic; quantum part searches basins, classical part relaxes geometry | Quantum stage gives approximate basins, not final angles |
| Latent torsion encoding | Encode fewer latent variables that generate correlated torsion patterns | Reduces dimension; can capture backbone motifs | Needs learned or designed mapping; less interpretable |
| Tensor/product ansatz with local torsion registers | Each residue/torsion has a small local register, with entanglers for neighboring coupling | More modular and scalable than global phase readout | Circuit still grows with sequence; entanglement depth must be controlled |

## Recommended Direction

For QTF, the strongest near-term alternative is a discrete torsion/rotamer encoding with classical refinement:

```text
backbone phi/psi:
  coarse bins, e.g. 8-16 states per angle

omega:
  fixed/window/trans by default

sidechain chi:
  residue-specific allowed rotamer bins
  chi=all only exposes physically meaningful states

quantum circuit:
  samples discrete torsion/rotamer assignments

postprocessing:
  rebuild structure
  GROMACS/OpenMM minimize
  rank by effective RMSD or energy
```

This fits the current workflow: the quantum/optimizer stage mostly needs to find a good basin, while GROMACS cleans clashes and local geometry. It also avoids spending expensive hardware resources on high-precision continuous phases that will later be minimized anyway.

For hardware scalability, avoid full continuous phase readout as the main representation. A more scalable direction is:

```text
1. residue-aware discrete bins for torsions
2. low-depth parameterized mixers
3. local/neighborhood coupling
4. classical minimization afterward
```

The most natural near-term replacement is:

```text
binary torsion-bin encoding + rotamer constraints + GROMACS refinement
```

with a possible continuous local refinement step once promising bins are found.
