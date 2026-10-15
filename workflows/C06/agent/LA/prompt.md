# CrN exchange interactions and spin waves at a fixed structure

## 1. Task

Determine the four nearest-neighbour exchange parameters and specified spin-wave energies of antiferromagnetic CrN at the supplied structure.

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
