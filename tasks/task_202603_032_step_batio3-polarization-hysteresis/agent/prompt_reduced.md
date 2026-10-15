# Task: polarization hysteresis of pristine BaTiO3 from a field-stepped molecular-statics loop

Extract the polarization hysteresis, coercive field, remanence and switching
brackets of the staged pristine BaTiO3 molecular-statics loop.

## What you are given

- `field_steps.csv` -- the sequence of field steps: the step label and the
  applied field $E_z$ in MV/cm at that step, in the order the loop was run.
- `states/state_<step>.lmp.gz` -- the relaxed structure at each step, one
  gzip-compressed LAMMPS data file per label in `field_steps.csv`
  (`atom_style full`, metal units). Each ion is a core-shell pair: the core and
  its shell share a molecule id and are joined by a bond, and each carries its
  own charge as listed in the file. Atom types 1, 2, 3 are the Ba, Ti and O
  cores and types 4, 5, 6 the corresponding shells.

## The polarization (fixed for this task)

Use Ti-centred cell dipoles about each Ti core. Include both core and shell,
each with its supplied charge, of the Ti (weight $1$), its six nearest O ions
(weight $1/2$) and its eight nearest Ba ions (weight $1/8$). Positions are
minimum-image displacements from that Ti core. The macroscopic polarization
is the sum of these cell dipoles per box volume, in C/m$^2$.

## What to determine

(1) $P_z$ at every field step, in C/m$^2$, in the order of `field_steps.csv`.

(2) The **coercive field** $E_c$: on each of the two branches of the loop,
    the field at which $P_z$ changes sign, obtained by linear interpolation of
    $E_z$ against $P_z$ between the two consecutive steps that bracket the sign
    change; $E_c$ is the mean of the absolute values of the two branch values.

(3) The **remanent polarization** $P_r$: on each branch, $P_z$ at $E_z = 0$
    by linear interpolation of $P_z$ against $E_z$ between the two consecutive
    steps that bracket zero field; $P_r$ is the mean of the absolute values of
    the two branch values.

(4) The **switching brackets**: for each branch, the applied fields of the two
    consecutive steps between which $P_z$ changes sign, in loop order.

## Output format

Write a single `answer.json` in the workspace root with exactly these fields
(fields in MV/cm, polarizations in C/m$^2$):

```json
{
  "Pz_C_m2": [<float>, ...],
  "Ec_MV_per_cm": <float>,
  "Pr_C_m2": <float>,
  "switching_fields_descending_MV_per_cm": [<float>, <float>],
  "switching_fields_ascending_MV_per_cm": [<float>, <float>]
}
```

`Pz_C_m2` follows `field_steps.csv` row order. Each switching pair contains
the field before, then after, the sign change on the named branch.
