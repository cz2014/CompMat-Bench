# Task: certify two candidate saddle points of a Ni island on a machine-learned potential

A compact seven-atom Ni island sits on a Ni(111) slab. On a machine-learned
interatomic potential, two island configurations have been proposed as
saddle points of the potential-energy surface. Your job is to certify each one:
measure its energy relative to a reference configuration, characterize the
curvature of the potential at it, and determine its index (the number of
independent downhill directions).

## Environment

You are in a sealed offline workspace. There is no internet. Compute runs on CPU.
Python is available with `numpy`, `scipy`, `torch`, `ase`, and `mace-torch`
installed. Write your result to a single `answer.json` in the workspace root (schema at the end).

## Staged files

- `system_cases.npz` -- a NumPy archive with these keys (positions and cell in
  angstrom; every atom is nickel):
  - `cell` -- shape `(3, 3)`, the simulation cell (periodic in the first two
    lattice directions; the third carries wide vacuum, so its periodicity
    setting does not affect any result).
  - `active_idx` -- shape `(7,)`, integer indices of the mobile island atoms.
  - `positions_reference` -- shape `(407, 3)`, the reference configuration.
  - `positions_case_a` -- shape `(407, 3)`, candidate saddle A.
  - `positions_case_b` -- shape `(407, 3)`, candidate saddle B.
- `mace_mp0a_medium.model` -- a MACE foundation-potential checkpoint, loadable
  with the `mace-torch` ASE calculator interface.

## Fixed conditions

- **Potential.** Evaluate energies and forces with the staged checkpoint,
  in double precision (float64) on CPU.
- **Constraint.** Only the seven atoms listed in `active_idx` are mobile; every
  other atom stays fixed at its staged position. All energy differences and all
  curvatures are taken with respect to these 21 active degrees of freedom only.

## What to compute

Treat `positions_case_a` and `positions_case_b` each as a candidate saddle
point. For each case (`case_a`, `case_b`):

1. `dE_eV` -- the energy of the case configuration minus the energy of
   `positions_reference` (both evaluated as staged, with the constraint above).
2. The Hessian of the energy with respect to the 21 active degrees of freedom,
   evaluated at the case configuration, by finite differences. Any standard
   finite-difference construction is acceptable, provided you converge your
   result with respect to the step size.
3. From that Hessian's eigenvalues (sorted algebraically, most negative
   first), report:
   - `lambda1_eV_A2` -- the lowest eigenvalue;
   - `lambda2_eV_A2` -- the second-lowest eigenvalue;
   - `n_negative` -- how many eigenvalues are negative.

## Output format

Write `answer.json` in the workspace root with exactly this flat schema (units
are in the field names; `dE_eV` in eV, eigenvalues in eV/A^2):

```json
{
  "case_a": {
    "dE_eV": <float>,
    "lambda1_eV_A2": <float>,
    "lambda2_eV_A2": <float>,
    "n_negative": <int>
  },
  "case_b": {
    "dE_eV": <float>,
    "lambda1_eV_A2": <float>,
    "lambda2_eV_A2": <float>,
    "n_negative": <int>
  }
}
```
