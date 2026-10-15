# CrN exchange interactions and spin waves at a fixed structure

## 1. Task

Determine the four nearest-neighbour exchange parameters and specified spin-wave energies of antiferromagnetic CrN at the supplied structure.

### Ordered ground state and exchange paths

CrN is a rock-salt nitride. Below its ordering temperature the Cr moments form
a double-layered antiferromagnet: ferromagnetic Cr planes of the (110) type of
the cubic parent stack along one direction as up, up, down, down. The ordered
state is orthorhombic. Relative to the cubic parent it carries a shear of the
cell and a displacement of every atom along the stacking direction, one free
internal coordinate per species.

Calculate a converged ground state of the ordered state on the supplied cell,
held fixed, from which the Cr moments are read.

**The exchange paths.** In the supplied cell each Cr has twelve nearest Cr
neighbours. Classify them by the relative orientation of the two moments
(read from the converged static on the supplied cell) and by how many
neighbours of that kind each Cr has; four classes result: four neighbours with
parallel moments, four with antiparallel moments, two with antiparallel
moments, and two with parallel moments.

### Wannier functions and exchange

Execute a two-step calculation that extracts the
isotropic Heisenberg exchange between Cr sites of orthorhombic
antiferromagnetic CrN from first principles: a spin-polarized ground-state
calculation that also builds maximally localized Wannier functions, followed
by a Green's-function evaluation of the pairwise exchange from that
Wannier Hamiltonian.

- `wannier`: a VASP calculation of the ordered ground state on the supplied cell
  that drives the Wannier-function construction through VASP's Wannier90
  library interface, producing the real-space Wannier Hamiltonian and the
  Wannier centres for both spin channels.

- `tb2j`: run TB2J's `wann2J.py` on the outputs of
  the `wannier` step, in that step's directory, to obtain the isotropic
  exchange between Cr sites.

TB2J
reports isotropic exchange in its native convention,

$$
E = -\sum_{i \neq j} J^{\mathrm{TB2J}}_{ij}\, \hat{\mathbf{e}}_i \cdot \hat{\mathbf{e}}_j ,
$$

with $\hat{\mathbf{e}}_i$ unit vectors along the moments and every ordered
pair counted.

Determine the exchange constant of each of the four classes, in meV, in the spin-length convention in Part 2.

Assign every pair to its class from the orientation of the two converged
moments and the pair geometry of that cell.

### Spin waves

Expand the model to quadratic order in spin deviations about the newly calculated
collinear state (Holstein-Primakoff) and obtain the magnon energies
$\hbar\omega(\mathbf{k}) \ge 0$ (the non-negative half of the Bogoliubov
spectrum) by the proper bosonic diagonalization of the
resulting quadratic form. With four magnetic sites per cell there are four
magnon energies at every $\mathbf{k}$; obtain all four, sorted ascending,
repeating degenerate values. Wavevectors are given in fractional coordinates
of the reciprocal lattice of the supplied cell.

Report the branches requested in Part 3.

## 2. Shared definitions and inputs

### Inputs

| File | Contents |
|---|---|
| `inputs/relaxed_cell/POSCAR` | Relaxed reference cell. The Cr stacking axis is its first lattice vector. |
| `inputs/potcars/Cr_pv/POTCAR` | Individual VASP LDA PAW 5.4 potential: PAW Cr_pv 02Aug2007. |
| `inputs/potcars/N/POTCAR` | Individual VASP LDA PAW 5.4 potential: PAW N 21Mar2012. |

### Structure and electronic model

Keep the supplied lattice vectors, atomic positions and site order fixed. The Cr magnetic signs in file order are (+, -, -, +), up to simultaneous reversal.

Use collinear VASP LDA with the Dudarev correction U_eff=4.1 eV on Cr 3d and the supplied PAW 5.4 potentials. Wannier functions span Cr 3d and N 2p separately in each spin channel. Exchange is obtained from the newly calculated Wannier Hamiltonians with the TB2J collinear Green-function method.

### Spin model

The spin Hamiltonian is $H=-2\sum_{i>j}J_{ij}\mathbf S_i\cdot\mathbf S_j$, with spin length S=3/2. The sum counts each unordered pair once; negative J is antiferromagnetic. The first Cr coordination shell consists of the twelve nearest Cr neighbours inherited from the rock-salt Cr sublattice. The model retains the four classes of the first Cr coordination shell, with no further-neighbour exchange, anisotropy or applied field. The expansion is about the ordered state on the supplied geometry.

## 3. Deliverable and output definitions

Write `answer.json` as one JSON object containing exactly the fields below.

| JSON field | Physical quantity | Unit / type |
|---|---|---|
| `J1_spin_meV` | Mean exchange for the first-shell class with four parallel-moment neighbours per Cr | meV / number |
| `J2_spin_meV` | Mean exchange for the first-shell class with four antiparallel-moment neighbours per Cr | meV / number |
| `J3_spin_meV` | Mean exchange for the first-shell class with two antiparallel-moment neighbours per Cr | meV / number |
| `J4_spin_meV` | Mean exchange for the first-shell class with two parallel-moment neighbours per Cr | meV / number |
| `X_omega1_meV` | Lowest magnon energy at X | meV / number |
| `X_omega4_meV` | Highest magnon energy at X | meV / number |
| `U_omega1_meV` | Lowest magnon energy at U | meV / number |
| `T_omega4_meV` | Highest magnon energy at T | meV / number |

Exchange values use the spin-length convention in Part 2. Shells refer to the supplied geometry and site-moment orientations. Magnon energies are the four non-negative branches in ascending order, repeating degeneracies. X=(1/2,0,0), U=(1/2,0,1/2), and T=(0,1/2,1/2) are fractional coordinates in the reciprocal basis of the supplied cell.
