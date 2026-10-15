# Chemical order and magnetic exchange in V2CoAl

## 1. Task

Determine how changing chemical order from L2_1 to X_A affects the nearest-neighbour V–Co exchange couplings in V2CoAl, and which magnetic pair types have the largest coupling magnitude in each ordering.

### Ordered structures and VASP optimization

Construct the full-Heusler and inverse-Heusler orderings of V2CoAl as VASP
POSCAR files. Use a conventional cubic cell with lattice constant
$a = 5.95\,\text{Å}$ and the sublattice occupations in Part 2.

For each newly constructed structure, prepare and execute inputs for a spin-polarized structural
relaxation with the magnetic starting guesses in Part 2, followed by a static total-energy
calculation on the relaxed geometry.

### Required runs

(1) A full relaxation of the ions and cell shape/volume, with a force-based
stopping criterion.

(2) A spin-polarized static calculation with no ionic motion, yielding a
well-converged total energy.

Treat both orderings as metals. Use the specified pseudopotentials and a common,
sufficiently converged numerical accuracy for all runs.

Apply the branch selection in Part 2 and pass the selected relaxed coordinates to ABACUS.

### ABACUS magnetic electronic structure

Prepare and execute inputs for a spin-polarized LCAO single-point SCF calculation of
metallic L2_1 V2CoAl, with site-resolved Mulliken magnetic-moment output.

Prepare and execute inputs for a spin-polarized, collinear LCAO single-point SCF calculation
of metallic XA V2CoAl, with site-resolved Mulliken magnetic-moment output.

Choose numerical settings that give a converged single-point calculation.

Use the selected VASP-relaxed geometry, without relaxation or rescaling.
Use the exact `.upf` and `.orb` files supplied for each species in Part 2.

For L2_1, determine the total magnetic moment per V2CoAl formula unit, in Bohr magnetons,
and the magnetic order of the V/Co system.

For XA, determine the signed site moment per atom for V1 and V2, the total moment per
V2CoAl formula unit, and the magnetic order of the V/Co system. Moments are
in Bohr magnetons and use the spin orientation of the newly computed output.

The magnetic order is ferromagnetic if the two V sublattice moments and the Co
moment are all parallel (same sign), ferrimagnetic if some of them are
antiparallel to the others but the net cell moment does not vanish, and
antiferromagnetic for an antiparallel arrangement with vanishing net cell
moment.

Use these moments and magnetic orders to confirm the magnetic state of each selected branch before computing exchange.

For each selected ABACUS branch, generate the spin-resolved Hamiltonians
`data-HR-sparse_SPIN0.csr` and `data-HR-sparse_SPIN1.csr`, overlap matrix
`data-SR-sparse_SPIN0.csr`, orbital-to-atom table `Orbital`, and SCF log
`running_scf.log`.

### Nearest-neighbour exchange

Use your newly computed ABACUS LCAO Hamiltonians and structures for the L2_1
and XA orderings of V2CoAl. Compute their isotropic nearest-neighbor
Heisenberg exchange couplings $J_{ij}$ using the magnetic-force-theorem
(Green-function) method.

Write and run the computation yourself using `abacus2J.py` on the newly computed data.

Include V and Co as magnetic species. Neglect the small induced Al moment
in the spin model.

The requested $J_{ij}$ use TB2J's native
$J_{\mathrm{iso}}$ convention, without a factor-of-two or spin-magnitude
rescaling.

For each magnetic pair type, obtain the nearest-neighbor coupling in Part 3 from the newly computed exchange bonds.

## 2. Shared definitions and inputs

### Inputs

| File | Contents |
|---|---|
| `inputs/potcars/V_pv/POTCAR` | Individual VASP PAW 5.4 potential: PAW_PBE V_pv 07Sep2000. |
| `inputs/potcars/Co/POTCAR` | Individual VASP PAW 5.4 potential: PAW_PBE Co 06Sep2000. |
| `inputs/potcars/Al/POTCAR` | Individual VASP PAW 5.4 potential: PAW_PBE Al 04Jan2001. |
| `inputs/pseudos/V_ONCV_PBE-1.0.upf` | ABACUS SG15 ONCV PBE pseudopotential for V. |
| `inputs/pseudos/Co_ONCV_PBE-1.0.upf` | ABACUS SG15 ONCV PBE pseudopotential for Co. |
| `inputs/pseudos/Al_ONCV_PBE-1.0.upf` | ABACUS SG15 ONCV PBE pseudopotential for Al. |
| `inputs/orbitals/V_gga_9au_100Ry_4s2p2d1f.orb` | ABACUS numerical orbitals for V, matched to the supplied SG15 pseudopotential. |
| `inputs/orbitals/Co_gga_9au_100Ry_4s2p2d1f.orb` | ABACUS numerical orbitals for Co, matched to the supplied SG15 pseudopotential. |
| `inputs/orbitals/Al_gga_9au_100Ry_4s4p1d.orb` | ABACUS numerical orbitals for Al, matched to the supplied SG15 pseudopotential. |

### Ordered structures

Both V2CoAl structures begin as conventional cubic cells at a=5.95 angstrom. L2_1 has space group Fm-3m, with V on 8c (1/4,1/4,1/4), Co on 4b (1/2,1/2,1/2), and Al on 4a (0,0,0). X_A has space group F-43m, with V1 on 4b (1/2,1/2,1/2), V2 on 4d (3/4,3/4,3/4), Co on 4c (1/4,1/4,1/4), and Al on 4a (0,0,0). Coordinates are fractional conventional-cell representatives. Only these fully ordered endpoints are included.

### Electronic structure and branch selection

Structural optimization uses VASP PAW PBE, with ionic positions and cell free. Magnetic electronic structure on each selected VASP geometry uses ABACUS LCAO PBE. Both are collinear, without Hubbard U or spin–orbit coupling.

At both software stages, the seed set is L2_1 with parallel V and Co, and X_A with both parallel V/Co and V1 antiparallel to V2/Co. Each ordering uses the lowest-energy converged branch within this seed set; VASP selection uses final static energies, and ABACUS selection uses fixed-geometry SCF energies. ABACUS does not relax or rescale the selected VASP geometry.

Exchange uses the collinear magnetic-force-theorem Green-function method on the ABACUS Hamiltonian and overlap matrices, with V and Co as magnetic species. Al is excluded from the spin model.

## 3. Deliverable and output definitions

Write `answer.json` as one JSON object containing exactly the fields below.

| JSON field | Physical quantity | Unit / type |
|---|---|---|
| `L2_1_J_V_Co_meV` | Nearest-neighbour V–Co exchange in L2_1 | meV / number |
| `XA_J_V1_Co_meV` | Nearest-neighbour V1–Co exchange in X_A | meV / number |
| `XA_J_V2_Co_meV` | Nearest-neighbour V2–Co exchange in X_A | meV / number |
| `L2_1_dominant_NN_pairs` | Magnetic pair types with the largest absolute nearest-shell mean exchange in L2_1 | Array of strings |
| `XA_dominant_NN_pairs` | Magnetic pair types with the largest absolute nearest-shell mean exchange in X_A | Array of strings |

The exchange convention is $H=-\sum_{i\ne j}J_{ij}\mathbf S_i\cdot\mathbf S_j$ with unit spins and ordered pairs, so positive J is ferromagnetic. Each reported coupling is the mean over symmetry-equivalent ordered pairs in that pair type's nearest shell, defined by its smallest nonzero intersite distance with a 0.05 angstrom grouping tolerance.

For dominance, compare absolute nearest-shell mean exchanges over all magnetic pair types within each ordering separately. Each array contains the maximizing pair type, or every pair type in an exact tie, sorted lexicographically. A label joins the two site-species labels with a hyphen in lexicographic order. Use V and Co in L2_1; use V1, V2 and Co in X_A. V1 and V2 denote the 4b and 4d sublattices defined in Part 2. Same-species magnetic pairs are included in the comparison.
