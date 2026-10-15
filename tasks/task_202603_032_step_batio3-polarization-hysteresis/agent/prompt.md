# Task: polarization hysteresis of pristine BaTiO3 from a field-stepped molecular-statics loop

Pristine, defect-free tetragonal BaTiO3, described by a core-shell (shell-model)
interatomic potential, has been driven around one closed hysteresis loop by an
external electric field along $z$: starting from the fully poled state at the
largest positive field, the field was stepped down through zero to the largest
negative field and back up again, and at every field step the atomic positions
and the cell were relaxed by energy minimization (molecular statics, 0 K)
before the next step. The relaxed structure at every step was recorded. From
these structures, the task is to construct the polarization loop $P_z(E_z)$
and extract the coercive field and the remanent polarization that this model
assigns to the pristine material.

## What you are given

- `field_steps.csv` -- the sequence of field steps: the step label and the
  applied field $E_z$ in MV/cm at that step, in the order the loop was run.
- `states/state_<step>.lmp.gz` -- the relaxed structure at each step, one
  gzip-compressed LAMMPS data file per label in `field_steps.csv`
  (`atom_style full`, metal units). Each ion is a core-shell pair: the core and
  its shell share a molecule id and are joined by a bond, and each carries its
  own charge as listed in the file. Atom types 1, 2, 3 are the Ba, Ti and O
  cores and types 4, 5, 6 the corresponding shells.

No simulation is required. The environment provides Python with numpy and
scipy; no molecular-dynamics code is installed.

## The polarization (fixed for this task)

The polarization is built from Ti-centred unit cells. For each Ti ion $k$ the
cell dipole is

$$
\mathbf{u}_k = \sum_{j} w_j \, q_j \left(\mathbf{r}_j - \mathbf{r}_{\mathrm{Ti},k}\right),
$$

where the sum runs over the core and the shell of the Ti ion itself (weight
$w = 1$), of its six nearest O ions (weight $w = 1/2$) and of its eight
nearest Ba ions (weight $w = 1/8$), each particle with its own charge $q_j$
from the data file, and every position entering as the minimum-image
displacement from the Ti core. The macroscopic polarization is

$$
\mathbf{P} = \frac{1}{V} \sum_{k} \mathbf{u}_k
$$

over all Ti cells, with $V$ the box volume. Report polarization in C/m$^2$.

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

- `Pz_C_m2`: item (1), one value per row of `field_steps.csv`, in that order
  (the list has as many entries as the file has steps).
- `Ec_MV_per_cm`: item (2).
- `Pr_C_m2`: item (3).
- `switching_fields_descending_MV_per_cm`: item (4) for the branch on which
  the field decreases; the field before the sign change first, then the field
  after it. `switching_fields_ascending_MV_per_cm`: the same for the branch
  on which the field increases.

All values must be computed from the staged structures; do not guess or
hardcode.
