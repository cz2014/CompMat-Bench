# Task: locate and certify three saddle points of a Ni island on an EAM surface

A compact seven-atom Ni island sits on a Ni(111) slab. From each of three
prescribed starting configurations you must locate the nearest first-order
saddle point of the potential-energy surface, certify it, find the two minima
it connects, and report scalar descriptors of the transition.

## Environment

You are in a sealed offline workspace. There is no internet. Compute runs on
CPU. Python is available with `numpy`, `scipy`, `torch`, and `ase` installed.
The whole task is minutes of CPU work. Write your result to a single
`answer.json` in the workspace root (schema at the end).

## Staged files

- `system.npz` -- a NumPy archive (positions and cell in angstrom; every atom
  is nickel):
  - `cell` -- shape `(3, 3)`, the simulation cell (periodic in the first two
    lattice directions; the third carries wide vacuum, so its periodicity
    setting does not affect any result).
  - `positions` -- shape `(407, 3)`, the relaxed reference configuration: the
    slab plus the seven-atom island in its compact minimum.
  - `active_idx` -- shape `(7,)`, integer indices of the mobile island atoms.
- `Ni_u3.eam` -- the interatomic potential, a standard single-element EAM
  potential file in funcfl format. Use it as-is.
- `start_a.npz`, `start_b.npz`, `start_c.npz` -- one per run. Each holds one
  key, `displacement_A`, shape `(7, 3)`, a Cartesian displacement in angstrom.
  Adding it to the reference positions at the `active_idx` rows (in order)
  gives that run's starting configuration.

## Fixed conditions

- **Constraint.** Only the seven atoms listed in `active_idx` are mobile; every
  other atom stays fixed at its staged position. All energies, forces, and
  curvatures are taken with respect to these 21 active degrees of freedom.
- **Reference energy.** Every reported energy difference is relative to the
  energy of the staged reference configuration, evaluated under the same
  constraint.

## What to do, per start

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
walk is your choice; the starting displacements are constructed so that any
correct implementation of it converges to the same saddle from a given start.
Converge the saddle until the residual force norm -- the Euclidean norm of
the 21-component force vector over the active degrees of freedom -- is at
most `1e-2 eV/angstrom`.

At the converged saddle:
1. Certify its character from the second derivatives of the energy over the
   21 active degrees of freedom (the plain Hessian, no mass weighting):
   report the two algebraically lowest eigenvalues (most negative first).
   Any converged evaluation of these second derivatives is acceptable.
2. Compute `dE_eV` = E(saddle) - E(reference).
3. Relax downhill on both sides of the saddle (displace slightly along the
   lowest mode in each direction, then minimise; converge each minimum to a
   residual force norm, same definition as above, of at most
   `1e-3 eV/angstrom`). One side returns to
   the reference basin; call the other endpoint the **final state**. A compact
   island at a different adsorption registry is a distinct endpoint.
4. Report `n_atoms_moved`: how many of the seven active atoms are displaced by
   more than `0.5 angstrom` (plain Cartesian distance) between the final
   state and the staged reference.
5. Report `dE_final_eV` = E(final state) - E(reference).

## Output format

Write `answer.json` in the workspace root with exactly this schema (units in
the field names; energies in eV, curvatures in eV/angstrom^2):

```json
{
  "start_a": {"dE_eV": <float>, "lambda1_eV_A2": <float>, "lambda2_eV_A2": <float>,
              "n_atoms_moved": <int>, "dE_final_eV": <float>},
  "start_b": {"dE_eV": <float>, "lambda1_eV_A2": <float>, "lambda2_eV_A2": <float>,
              "n_atoms_moved": <int>, "dE_final_eV": <float>},
  "start_c": {"dE_eV": <float>, "lambda1_eV_A2": <float>, "lambda2_eV_A2": <float>,
              "n_atoms_moved": <int>, "dE_final_eV": <float>}
}
```
