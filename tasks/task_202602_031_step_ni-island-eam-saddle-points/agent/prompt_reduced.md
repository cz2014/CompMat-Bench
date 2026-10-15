# Task: locate and certify three saddle points of a Ni island on an EAM surface

Use the staged EAM potential to locate the nearest first-order saddle by
minimum-mode following from each of the three prescribed starts, and report
the transition descriptors below. The implementation of the search is your choice.

## Inputs and constraints

- `system.npz`: all atoms are Ni. `positions` is the relaxed reference
  configuration; `cell` is the simulation cell. Both use angstrom.
  The first two lattice directions are periodic; either periodicity setting
  in the third direction is permitted because of the wide vacuum.
  Only atoms indexed by `active_idx` are mobile; all other atoms remain fixed.
- `Ni_u3.eam`: the EAM potential, to be used as supplied.
- `start_a.npz`, `start_b.npz`, `start_c.npz`: each `displacement_A` is a
  Cartesian displacement in angstrom relative to the reference, in
  `active_idx` order. All other coordinates remain at their reference values.

## Requested transition descriptors

The saddle residual force norm must be at most `1e-2 eV/angstrom`.
The two downhill-connected minima must each have residual force norm at most
`1e-3 eV/angstrom`. Here the norm is the Euclidean norm over all active
Cartesian force components.

One minimum is in the reference basin. The other is the **final state**;
a compact island at another adsorption registry counts as a distinct endpoint.
All energies refer to the same fixed-atom constraint.

For each start report:

- `dE_eV`: E(saddle) - E(reference);
- `lambda1_eV_A2`, `lambda2_eV_A2`: the two algebraically lowest eigenvalues
  of the saddle's active Cartesian Hessian, without mass weighting,
  most negative first;
- `n_atoms_moved`: the number of active atoms whose plain Cartesian
  displacement from the reference to the final state exceeds `0.5 angstrom`;
- `dE_final_eV`: E(final state) - E(reference).

The workspace is offline and CPU-only. Python with numpy, scipy, torch and
ase is available.

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
