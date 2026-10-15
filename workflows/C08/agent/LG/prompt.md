# Transfer a surface rearrangement between two interatomic potentials

## 1. Task

Determine whether the Ni-island rearrangement found on EAM persists on MACE, and quantify its activation barriers, refined saddle curvature and final-basin energy offset.

### EAM anchor and endpoints

A compact seven-atom Ni island sits on a Ni(111) slab. From the single
prescribed starting configuration you must locate the nearest first-order
saddle point of the potential-energy surface, certify it, find the two minima
it connects, and report scalar descriptors of the transition.

Starting from the displaced configuration, walk uphill to the nearest
first-order saddle point by minimum-mode following, a standard idea behind
single-ended saddle-point searches on a potential-energy surface. At a
first-order saddle the energy surface curves downward along exactly one
direction, the minimum-curvature mode, and upward along all directions
orthogonal to it. So take the true force and reverse the sign of its
component along the locally lowest-curvature direction. Under this
transformed force the saddle becomes an effective
attractor: ordinary force-following on the transformed force walks uphill
along the soft direction while relaxing in all orthogonal directions, and it
converges to the saddle. The lowest-curvature direction changes along the
path, so it has to be re-estimated as the walk proceeds. How you realise this
walk is your choice.

At the converged EAM saddle, certify its character from the second derivatives of the energy over the
   21 active degrees of freedom (the plain Hessian, no mass weighting):
   obtain the two algebraically lowest eigenvalues (most negative first).
   Any converged evaluation of these second derivatives is acceptable.

Relax downhill on both sides of the saddle (displace slightly along the
   lowest mode in each direction, then minimise). One side returns to
   the reference basin; call the other endpoint the **final state**. A compact
   island at a different adsorption registry is a distinct endpoint.

Relax the reference basin on EAM for the barrier baseline.

### MACE saddle refinement and curvature

On the MACE potential, the EAM saddle is the starting configuration for refining a
saddle point of the potential-energy surface. Your job is to refine and certify it:
measure its energy relative to the MACE-relaxed reference basin, characterize the
curvature of the potential at the refined saddle, and determine its index (the number of
independent downhill directions).

Re-relax both EAM endpoints on MACE as the comparison basins.

Compute the Hessian of the energy with respect to the 21 active degrees of freedom,
   evaluated at the newly refined saddle, by finite differences. Any standard
   finite-difference construction is acceptable, provided you converge your
   result with respect to the step size.

From that Hessian's eigenvalues (sorted algebraically, most negative
   first), report:

   - `mace_lambda1_eV_A2` -- the lowest eigenvalue;
   - `mace_lambda2_eV_A2` -- the second-lowest eigenvalue;
   - `mace_n_negative` -- how many eigenvalues are negative.

### Connection on the new surface

Descend independently from both signs of the refined MACE saddle's unstable mode. Compare these endpoints with the MACE-relaxed EAM endpoints and report the energy and connection observables in Part 3.

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
