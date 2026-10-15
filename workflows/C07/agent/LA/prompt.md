# Chemical order and magnetic exchange in V2CoAl

## 1. Task

Determine how changing chemical order from L2_1 to X_A affects the nearest-neighbour V–Co exchange couplings in V2CoAl, and which magnetic pair types have the largest coupling magnitude in each ordering.

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
