# Dopant geometry, symmetry and spin splitting

## 1. Task

Determine how pair substitution in the prescribed AFM-A MnTe supercell changes its crystallographic families and geometry-dependent spin splitting.

### Substitution geometry and symmetry

The Mn moments order antiferromagnetically of A type: within each Mn ab-plane the
moments align ferromagnetically, and successive Mn planes along the c axis
alternate in direction.

Build the 2x2x2 supercell of the supplied cell. Consider
replacing a pair of the Te sites by Se.
Enumerate the unordered pairs of Te sites and, for each resulting doped
structure, determine its crystallographic space group.

Group the doped structures by the space group you find, and for each space group
that occurs count how many doped structures fall into it.

### Geometry-dependent spin blocks

For each family that occurs, use its representative geometry to construct the full-cell p/d Slater–Koster Hamiltonian specified in Part 2.

For a bond with displacement vector $\mathbf R$, pointing from the site of the first-named orbital to the site of the second, and direction cosines $(l, m, n) = \mathbf R/|\mathbf R|$ (the $x$, $y$, $z$ components of the unit vector along $\mathbf R$), the hopping amplitude between a magnetic-site $d_{xy}$ orbital and a nonmagnetic-site $p_x$ orbital is

$$\sqrt{3}\, l^2 m\, t_\sigma + m\,(1 - 2 l^2)\, t_\pi,$$

and between two nonmagnetic-site p_x orbitals is

$$l^2\, t_\sigma + (1 - l^2)\, t_\pi.$$

Apply the standard two-centre angular dependence to all d–p and p–p orbital pairs of the full basis; the expressions above illustrate two of those pairs.

Build the full-cell Bloch Hamiltonian $H(\mathbf k)$ in the basis in Part 2. The diagonal entries are the onsite energies defined in Part 2. Each off-diagonal coupling is a sum over the underlying site-to-site bonds of

$$t(\mathbf R)\; e^{\,i\,\mathbf k \cdot \mathbf R}.$$

$t(\mathbf R)$ is the hopping amplitude of the bond, $\mathbf R$ is the bond displacement, and $\mathbf k$ is the wavevector.

For the matrix element in row $i$, column $j$ of
the full-cell basis, $\mathbf R$ points from the site of orbital $i$ to the site
of orbital $j$. Bonds are taken to periodic images of the full supercell
(translations of the partner by integer multiples of its three lattice vectors). Include every
site-to-site bond within the applicable radial cutoff in Part 2, counting every image separately
even where two images of the same partner site fall inside that cutoff;
longer bonds are dropped.

The model has an up-spin block, in which $s = +1$, and a down-spin block, in which $s = -1$. The two blocks share the same hoppings and differ only through $s$ in the onsite terms.

### Spin splitting

At any wavevector, sort each block's eigenvalues in ascending order and pair the up- and down-block eigenvalues index by index; the spin difference at that wavevector is the largest absolute difference over all paired eigenvalues.

Evaluate the two splittings of each family using these geometry-dependent spin blocks at the two k points in Part 2.

The k points in Part 2 are fractional coordinates in the reciprocal basis $(\mathbf b_1, \mathbf b_2, \mathbf b_3)$ defined by $\mathbf b_i \cdot \mathbf a_j = 2\pi\,\delta_{ij}$.

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
