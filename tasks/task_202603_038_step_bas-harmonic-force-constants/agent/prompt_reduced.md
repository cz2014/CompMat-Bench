# Task: harmonic force constants and zone-centre optical frequencies of boron arsenide from finite-displacement supercell runs

Determine the harmonic force constants and phonon checkpoints from five
512-atom BAs supercell calculations. `runs/reference/` is undisplaced;
`runs/disp_a/` through `runs/disp_d/` each displace one atom.
Each directory contains Quantum ESPRESSO `pw.in` and `pw.out.gz`;
use their geometries, atom order and units as supplied.

Use masses $10.01293695\,\mathrm{u}$ for $^{10}$B, $11.00930536\,\mathrm{u}$ for $^{11}$B, and $74.921595\,\mathrm{u}$
for As; the natural-abundance average B mass is $10.811\,\mathrm{u}$.

Report:

1. For each displaced run, the displaced atom's one-based index in its
   input deck, species, Cartesian displacement axis and signed displacement
   in Å.
2. The maximum acoustic-sum residual over the four displaced runs.
   For each run this residual is the largest absolute Cartesian component
   of the sum over all atoms of force changes relative to the reference, divided by the
   largest absolute force-change component on any atom.
3. The primitive-cell $\Gamma$ optical frequency for pure $^{11}$B, pure $^{10}$B and
   uniform natural-abundance average B mass.
4. The highest frequency of the full supercell $\Gamma$ spectrum for uniform
   natural-abundance average B mass.

Report frequencies as spectroscopic wavenumbers in cm$^{-1}$.
No first-principles calculation is required. Python with NumPy, SciPy and phonopy, and ALAMODE's `alm` and
`anphon`, are available.

## Output format

Write `answer.json` in the workspace root with exactly these fields:

```json
{
  "displaced_runs": {
    "disp_a": {"atom_index": <int>, "species": <string: one of "B", "As">, "axis": <string: one of "x", "y", "z">, "displacement_A": <float>},
    "disp_b": {"atom_index": <int>, "species": <string: one of "B", "As">, "axis": <string: one of "x", "y", "z">, "displacement_A": <float>},
    "disp_c": {"atom_index": <int>, "species": <string: one of "B", "As">, "axis": <string: one of "x", "y", "z">, "displacement_A": <float>},
    "disp_d": {"atom_index": <int>, "species": <string: one of "B", "As">, "axis": <string: one of "x", "y", "z">, "displacement_A": <float>}
  },
  "acoustic_sum_residual_max": <float>,
  "omega_TO_11B_cm1": <float>,
  "omega_TO_10B_cm1": <float>,
  "omega_TO_natural_cm1": <float>,
  "omega_max_supercell_natural_cm1": <float>
}
```
