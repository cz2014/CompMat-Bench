# Dopant geometry, symmetry and spin splitting

## 1. Task

Determine how pair substitution in the prescribed AFM-A MnTe supercell changes its crystallographic families and geometry-dependent spin splitting.

## 2. Shared definitions and inputs

### Inputs

| File | Contents |
|---|---|
| `inputs/MnTe_relaxed.vasp` | Relaxed primitive cell of MnTe in the hexagonal NiAs-type structure, in POSCAR format. |

### System

Treat the supplied cell as the reference that fixes the lattice and the atomic sites. Substitutions replace Te by Se in its 2x2x2 supercell. Positions and lattice remain fixed. Mn has A-type antiferromagnetic order along c.

Conventions for this task (fixed here):

- Determine the space group of a doped structure with all Mn atoms treated as a
  single species, so the magnetic pattern is not folded into the symmetry
  determination.

- Enumerate over the raw set of Te-site pairs, one doped structure per pair; do
  not first reduce the pairs to symmetry-distinct classes.

A family's representative is the lexicographically lowest pair of full-supercell site indices. For this selection, number sites from zero in primitive-site-major order, and order image translations lexicographically over (i,j,k) in {0,1}^3, with components along the primitive lattice vectors.

### Electronic model

Use a full-supercell standard two-centre Slater–Koster model, with the five real Mn d orbitals and the three real p orbitals on each Te/Se. The model is collinear, without spin–orbit coupling, Mn–Mn hopping or crystal-field splitting. In the up-spin block, the Mn onsite energy is lowered by Delta/2 on the positive-moment sublattice and raised by Delta/2 on the other; the down-spin block reverses these shifts. The positive sublattice descends from primitive Mn at z=0.

| Parameter | Value |
|---|---|
| Mn onsite energy before exchange | -1.85 eV |
| Te and Se onsite energies | -4.5 eV |
| Delta | 2.0 eV |
| Mn–ligand (pd sigma, pd pi) | (-1.42, 0.42) eV |
| Ligand–ligand (pp sigma, pp pi) | (1.112, -0.325) eV |
| Hopping multiplier per Se endpoint | 0.5 |
| Mn–ligand radial cutoff | 3.2 angstrom |
| Ligand–ligand radial cutoff | 4.4 angstrom |

The two-centre parameters are constant within the cutoffs, with no distance rescaling. Periodicity is that of the full 2x2x2 cell. Gamma=(0,0,0) and the generic point=(0.137,0.219,0.173) are in its reciprocal basis.

## 3. Deliverable and output definitions

Write `answer.json` as one JSON object with the single top-level key `families`, an array containing one object for every crystallographic space group that occurs in the substitution enumeration. Each object contains exactly the fields below. Sort the array by increasing space-group number.

| JSON field in each family | Physical quantity | Unit / type |
|---|---|---|
| `sg_number` | International space-group number of this family | Integer |
| `count` | Number of raw unordered Te-site pairs belonging to this family | Integer |
| `Gamma_split_eV` | Maximum paired spin splitting of this family's representative at Gamma | eV / number |
| `generic_split_eV` | Maximum paired spin splitting of this family's representative at the generic k point | eV / number |

The spin splitting compares the ascending up- and down-spin eigenvalue lists index by index and takes the maximum absolute difference. The representative and k points are defined in Part 2.
