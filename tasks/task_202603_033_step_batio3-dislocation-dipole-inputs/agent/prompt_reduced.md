# Task: build an edge-dislocation dipole in BaTiO3 and write its LAMMPS relaxation and poling deck

Build a BaTiO3 edge-dislocation dipole and a LAMMPS relaxation/poling
script using `bto_unit.lmp` and `shell_model.yaml`. Submit inputs only.
Use the seed's `atom_style full`, metal units, lattice constant, type,
charge, mass, core-shell pairing and bond conventions, and the potential
exactly as defined in the YAML.

Replicate the seed $40\times80\times1$ along $x,y,z$, preserving its
coordinate origin. Remove the BaO plane nearest the cell centre along $x$
and the TiO2 plane immediately on its $+x$ side, over replica origins
$y\in[L_y/4,3L_y/4)$, where $L_y$ is the built cell length along $y$,
and the full $z$ thickness. Remove whole core-shell
ions only; every remaining coordinate stays ideal, with its core-shell
molecule pairing and bond intact. The built cell is periodic.

The script must:

1. Apply $10\,\mathrm{MV/cm}$ along $+x$ throughout the with-field
   relaxation, including its contribution to the minimized energy.
2. Compress the cell by 1% along $x$ before the first minimization.
   At $0\,\mathrm K$, minimize first with positions free only along $x$,
   then with all positions free, then with positions and all three cell
   lengths free at zero target stress and fixed angles. Alternate the
   cell-and-position and position-only relaxations until the cell converges.
   Every minimization uses a tight force tolerance with the energy stopping
   test disabled.
3. Write `relaxed_wfield.lmp`, remove the field, repeat the
   cell-and-position relaxation and write `relaxed_nofield.lmp`.

## Output

Write `answer.json` in the workspace root with exactly these fields:

```json
{"runs": {"relax": "answer/relax"}}
```

`answer/relax/` contains `relax.in` and the built `input.lmp`.
The script refers to the data file as `input.lmp` and writes the two output
files named above. Keep the seed's atom types, charges, masses,
molecule-pairing and bond types. Do not copy the parameter file into the
submission.
