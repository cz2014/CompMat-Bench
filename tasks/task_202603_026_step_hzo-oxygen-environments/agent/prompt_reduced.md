# Task: oxygen environments and relaxation response of a cation-disordered Hf$_{0.5}$Zr$_{0.5}$O$_2$ supercell

Determine ideal oxygen environments, relaxation displacements, relaxed
cation-shell separation and undoped symmetry from the four HZO/HfO$_2$ structures.

## What you are given

- `ideal_x050.vasp` -- the $x = 0.5$ supercell before relaxation (the
  starting structure, ideal internal coordinates of the parent with the Zr
  substitution applied).
- `relaxed_x050.vasp` -- the same cell after the fixed-lattice relaxation.
- `ideal_hfo2.vasp` -- the undoped HfO$_2$ supercell before relaxation.
- `relaxed_hfo2.vasp` -- the same cell after relaxation.

All four are VASP structure files. Within each pair the atoms are listed in
the same order, so atom $i$ of the relaxed file is atom $i$ of the ideal
file.

## Definitions (fixed for this task)

- **Local cation environment.** For an oxygen site, count the Hf atoms and
  the Zr atoms whose distance from the oxygen, under periodic boundary
  conditions, is at most 2.3 Å.
  The environment of a site is the pair (number of Hf, number of Zr); at
  $x = 0.5$ nine environments exist, five on the four-coordinated sublattice
  and four on the three-coordinated one. Environments are labelled by the
  counts, `4Hf`, `3Hf1Zr`, `2Hf2Zr`, `1Hf3Zr`, `4Zr`, `3Hf`, `2Hf1Zr`,
  `1Hf2Zr`, `3Zr` (a zero count is omitted from the label). **Environments
  are defined on the ideal, unrelaxed coordinates.**
- **Relaxation displacement.** For each atom, the length of the
  minimum-image displacement vector from its ideal position to its relaxed
  position. The maximum of these lengths, and their root mean square, are taken over all
  atoms of the cell.
- **Shell separation.** On the relaxed $x = 0.5$ cell, over all cation-oxygen
  pairs under periodic boundary conditions (Hf and Zr alike): the longest
  cation-oxygen distance that is at most 2.3 Å (the longest bond) and the
  shortest cation-oxygen distance that exceeds 2.3 Å (the closest non-bonded
  cation).
- **Symmetry.** The space group of the relaxed undoped cell, determined with
  a symmetry tolerance of $10^{-3}$ Å on the atomic positions.

## Output format

Write a single `answer.json` in the workspace root with exactly these fields
(lengths in Å, every number unrounded):

```json
{
  "site_environments": [[<float>, <float>, <float>, <int>, <int>], ...],
  "class_counts": {"4Hf": <int>, "3Hf1Zr": <int>, "2Hf2Zr": <int>, "1Hf3Zr": <int>, "4Zr": <int>, "3Hf": <int>, "2Hf1Zr": <int>, "1Hf2Zr": <int>, "3Zr": <int>},
  "displacement_max_doped_A": <float>,
  "displacement_rms_doped_A": <float>,
  "displacement_max_undoped_A": <float>,
  "displacement_rms_undoped_A": <float>,
  "longest_bond_A": <float>,
  "closest_nonbonded_cation_A": <float>,
  "space_group_number_undoped_relaxed": <int>
}
```

`site_environments` contains every oxygen of `ideal_x050.vasp`, in any order:
its three supplied fractional coordinates, then its Hf and Zr counts.
`class_counts` counts these sites by environment. The displacement fields
are the defined max/RMS over all atoms (`doped`: $x = 0.5$; `undoped`: HfO$_2$).
The two distance fields report the defined relaxed shell separation;
the space-group field is the International Tables number.
