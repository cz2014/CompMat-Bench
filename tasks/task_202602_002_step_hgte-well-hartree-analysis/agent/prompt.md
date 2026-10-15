# Task: potential asymmetry, charge-neutral Fermi level, and well charge of a gated HgTe quantum well

A 45 nm HgTe quantum well sits between wider-gap (Hg,Cd)Te barrier layers, with
a metallic top gate above the top barrier. The electric field in the bottom
barrier is held fixed (it is set by the sample, not by the gate), while the gate
sets the total carrier density in the well. A self-consistent Hartree
(Schrodinger-Poisson) calculation of this structure was converged at three total
carrier densities: one at the special density where the Hartree potential is
inversion-symmetric about the well center, one p-type doped, and one n-type
doped. Which of the three staged conditions is which is not disclosed.

You are given the converged outputs of those three solves under the three
condition directories described in `README.md` (geometry, grid, and file schemas
are documented there). Your task is to extract, per condition, the three
quantities below from the staged data, and to identify the inversion-symmetric
condition.

## Quantities to report (per condition)

1. **`dU_well_<tag>_meV`** -- the Hartree-potential difference across the well:
   the potential U at the top well interface (z = +22.5 nm) minus U at the
   bottom well interface (z = -22.5 nm), in meV. This is the residual potential
   asymmetry the solve develops across the HgTe layer.

2. **`ef0_<tag>_meV`** -- the charge-neutral Fermi level of that solve's
   converged band structure, in meV.

3. **`n_well_<tag>_cm-2`** -- the net mobile sheet carrier density contained
   in the HgTe layer, $z \in [-22.5, +22.5]$ nm, excluding fixed background
   charge and reported in cm$^{-2}$ with electrons positive.

## The inversion-symmetric condition

Report **`symmetric_condition`** -- a string, one of `"A"`, `"B"`, `"C"`: the
tag of the condition whose converged Hartree potential is inversion-symmetric
about the well center. Decide it from the staged data.

## Condition tags

| tag | condition directory |
|-----|---------------------|
| `A` | `cond_A/` |
| `B` | `cond_B/` |
| `C` | `cond_C/` |

## Output format

Write a single `answer.json` in the workspace root with exactly these fields
(numbers except `symmetric_condition`, units as named):

```json
{
  "dU_well_A_meV": <float>,
  "dU_well_B_meV": <float>,
  "dU_well_C_meV": <float>,
  "ef0_A_meV": <float>,
  "ef0_B_meV": <float>,
  "ef0_C_meV": <float>,
  "n_well_A_cm-2": <float>,
  "n_well_B_cm-2": <float>,
  "n_well_C_cm-2": <float>,
  "symmetric_condition": <string: one of "A", "B", "C">
}
```

All ten fields are required and participate in scoring.
