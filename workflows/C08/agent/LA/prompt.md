# Transfer a surface rearrangement between two interatomic potentials

## 1. Task

Determine whether the Ni-island rearrangement found on EAM persists on MACE, and quantify its activation barriers, refined saddle curvature and final-basin energy offset.

## 2. Shared definitions and inputs

### Inputs

| File | Contents |
|---|---|
| `inputs/system.npz` | Reference Ni slab and island: positions (407,3), cell (3,3), and active_idx (7,); lengths in angstrom. |
| `inputs/start_c.npz` | Cartesian displacement_A (7,3), added to the reference positions at active_idx in row order to obtain the search start. |
| `inputs/Ni_u3.eam` | Single-element EAM potential in funcfl format. |
| `inputs/mace_mp0a_medium.model` | MACE foundation-potential checkpoint. |

### Constraints

Only the seven indexed island atoms move; substrate positions and the cell remain fixed. The slab is periodic in its first two lattice directions. The search concerns the single supplied perturbation of the compact island.

### Required potential sequence

Locate the local EAM transition by minimum-mode following from the supplied start, then use the resulting EAM saddle coordinates to initialize a new saddle refinement on MACE. The two EAM-connected minima, re-relaxed on MACE, define the comparison basins for the MACE transition. The final basin is the EAM endpoint other than the reference basin. Basin labels follow the EAM connection on both potentials.

## 3. Deliverable and output definitions

Write `answer.json` as one JSON object containing exactly the fields below.

| JSON field | Physical quantity | Unit / type |
|---|---|---|
| `eam_barrier_eV` | EAM saddle energy minus EAM-relaxed reference-basin energy | eV / number |
| `mace_barrier_eV` | MACE saddle energy minus MACE-relaxed reference-basin energy | eV / number |
| `mace_lambda1_eV_A2` | Algebraically lowest eigenvalue of the MACE saddle Hessian | eV/angstrom^2 / number |
| `mace_lambda2_eV_A2` | Algebraically second-lowest eigenvalue of the MACE saddle Hessian | eV/angstrom^2 / number |
| `mace_n_negative` | Number of negative eigenvalues of the MACE saddle Hessian | Integer |
| `mace_final_offset_eV` | MACE-relaxed final-basin energy minus MACE-relaxed reference-basin energy | eV / number |
| `n_atoms_moved` | Number of island atoms displaced by more than 0.5 angstrom between the two MACE saddle descent endpoints | Integer |
| `same_connection` | Whether the refined MACE saddle connects the two comparison basins defined in Part 2 | Boolean |

Hessians are plain, unweighted energy Hessians with respect to the 21 active Cartesian coordinates. Displacements in n_atoms_moved retain atom identities and use plain Cartesian distance. For same_connection, basin identity may account for permutations of identical island atoms and point symmetries of the fixed substrate anchored at the reference island; translating the island to a different adsorption registry does not identify the basins.
