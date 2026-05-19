# Physics Energy Function

QTF evaluates ten energy terms at every objective call. The total energy is the
(weighted) sum of all active terms. All terms are designed so that **lower is better**
(a more stable, physically plausible structure has a more negative total energy).

---

## Force fields

Three force-field presets are available, each providing bond lengths, bond angles,
partial charges, and van der Waals parameters appropriate for that force field.

| Identifier | Full name | Use case |
|:-----------|:----------|:---------|
| `"charmm"` | CHARMM22 | Default; well-tested for small peptides |
| `"amber"`  | AMBER ff14SB | Modern backbone/side-chain parameters |
| `"opls"`   | OPLS-AA  | United-atom representation; fast |

```python
folder = QuantumBiophysicsFolder("YYDPETGTWY", force_field="amber")
```

---

## Energy terms

### 1. Hydrophobic collapse

Rewards burial of hydrophobic residues (negative ΔG) and penalises exposure of
hydrophilic residues.

```
E_hydro = Σᵢ Σⱼ>ᵢ  hᵢ · hⱼ · f(rᵢⱼ)
```

where `hᵢ` is the hydrophobicity of residue *i* and `f(r)` is a soft-sphere kernel
that is −1 for contacting pairs and 0 for distant pairs.

---

### 2. Hydrogen bonds

Captures N–H···O=C backbone interactions and side-chain H-bonds using a
distance–angle potential:

```
E_hb = Σ  −cos²(θ) · (σ/r)⁶  over (donor, acceptor) pairs within 3.5 Å
```

---

### 3. Electrostatics

Screened Coulomb interaction between partial charges:

```
E_elec = Σᵢ Σⱼ>ᵢ  qᵢ · qⱼ / (4πε₀ · εᵣ · rᵢⱼ)
```

A distance-dependent dielectric `εᵣ = 4 r` is used to implicitly account for
solvent screening.

---

### 4. Sterics (van der Waals)

Lennard-Jones 12-6 potential for all heavy-atom pairs with sequence separation ≥ 2:

```
E_vdw = Σ  ε [(σ/r)¹² − (σ/r)⁶]
```

Pairs within the same residue and 1-2 / 1-3 bonded pairs are excluded.

---

### 5. Ramachandran term

Penalises (φ, ψ) combinations that fall outside favoured or allowed regions of the
Ramachandran map. The penalty function is a sum of Gaussian wells centred on the
canonical secondary-structure regions:

```
E_rama = Σᵢ  1 − Σ_region  exp(−½ [(φᵢ−μ_φ)²/σ_φ² + (ψᵢ−μ_ψ)²/σ_ψ²])
```

Pre-computed Gaussian parameters are stored for α-helix, β-sheet, and PPII regions.

---

### 6. Rotamer term

Side-chain χ angles are penalised when they deviate from known low-energy rotamer wells,
using the Penultimate Rotamer Library values as reference minima.

---

### 7. Compactness / radius of gyration

```
E_compact = Rg²  where  Rg = √(⟨rᵢ − rcom⟩²)
```

This term drives the chain to adopt a compact, globular fold rather than remaining
extended.

---

### 8. Secondary structure propensity

Residue-specific preference for helical vs. strand geometry, derived from
statistical potentials. Applied as a soft bias on (φ, ψ) separately from the
Ramachandran penalty.

---

### 9. π–π stacking

Rewards co-planar, stacked aromatic ring pairs (Phe, Tyr, Trp, His) within 5 Å
with a favourable geometry angle < 30°.

---

### 10. Geometric penalties

Ensures all bond lengths and bond angles remain physically plausible — penalises
deviations from ideal values with a harmonic restraint.

---

## Energy weights

The relative contribution of each term is controlled by internal weights. These are
currently fixed per force field and are **not** user-configurable in the public API.
The default balance was calibrated on short benchmark peptides (≤ 20 residues).

---

!!! info "Units"
    All energy values are in **internal units** (not kcal/mol or kJ/mol).
    Meaningful comparisons are only valid between replicas folded with the **same**
    force field and sequence.
