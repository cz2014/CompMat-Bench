# Task: oxygen environments and relaxation response of a cation-disordered Hf$_{0.5}$Zr$_{0.5}$O$_2$ supercell

Ferroelectric hafnia-zirconia is a solid solution: in its polar orthorhombic
phase, Hf and Zr share one cation sublattice at random. Oxygen vacancies are
the defects that govern its switching and its leakage, and how easily an
oxygen leaves depends on which cations surround its site. A supercell of the
polar phase at $x = 0.5$, with the cation sites decorated at random, has been
relaxed at fixed lattice within PBE, and so has the undoped HfO$_2$ supercell
of the same size at the same lattice. From the four structures, the task is
to establish the site table a defect study is indexed by, to quantify how
much internal relaxation the cation disorder induces compared with the
ordered parent, to measure how cleanly the first cation shell of an oxygen
separates from the second, and to determine what symmetry the ordered parent
retains after relaxation.

## What you are given

- `ideal_x050.vasp` -- the $x = 0.5$ supercell before relaxation (the
  starting structure, ideal internal coordinates of the parent with the Zr
  substitution applied).
- `relaxed_x050.vasp` -- the same cell after the fixed-lattice relaxation.
- `ideal_hfo2.vasp` -- the undoped HfO$_2$ supercell before relaxation.
- `relaxed_hfo2.vasp` -- the same cell after relaxation.

All four are VASP structure files. Within each pair the atoms are listed in
the same order, so atom $i$ of the relaxed file is atom $i$ of the ideal
file. No simulation is required. The environment provides Python with numpy,
scipy and pymatgen.

## Definitions (fixed for this task)

- **Local cation environment.** For an oxygen site, count the Hf atoms and
  the Zr atoms whose distance from the oxygen, under periodic boundary
  conditions, is at most 2.3 Å. Every oxygen of the parent structure has
  either four such neighbours (the four-coordinated sublattice) or three
  (the three-coordinated sublattice).
  The environment of a site is the pair (number of Hf, number of Zr); at
  $x = 0.5$ nine environments exist, five on the four-coordinated sublattice
  and four on the three-coordinated one. Environments are labelled by the
  counts, `4Hf`, `3Hf1Zr`, `2Hf2Zr`, `1Hf3Zr`, `4Zr`, `3Hf`, `2Hf1Zr`,
  `1Hf2Zr`, `3Zr` (a zero count is omitted from the label). **Environments
  are defined on the ideal, unrelaxed coordinates.**
- **Relaxation displacement.** For each atom, the length of the
  minimum-image displacement vector from its ideal position to its relaxed
  position. The maximum of these lengths, and their root mean square
  (the square root of the mean of the squared lengths), are taken over all
  atoms of the cell.
- **Shell separation.** On the relaxed $x = 0.5$ cell, over all cation-oxygen
  pairs under periodic boundary conditions (Hf and Zr alike): the longest
  cation-oxygen distance that is at most 2.3 Å (the longest bond) and the
  shortest cation-oxygen distance that exceeds 2.3 Å (the closest non-bonded
  cation).
- **Symmetry.** The space group of the relaxed undoped cell, determined with
  a symmetry tolerance of $10^{-3}$ Å on the atomic positions.

## What to determine

(1) The environment of every oxygen of the $x = 0.5$ cell.

(2) How many oxygens fall in each of the nine environments.

(3) The maximum and the root-mean-square relaxation displacement of the
    $x = 0.5$ cell and of the undoped cell.

(4) The shell separation of the relaxed $x = 0.5$ cell: the longest bond
    and the closest non-bonded cation distance.

(5) The space-group number of the relaxed undoped cell.

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

- `site_environments`: item (1), one entry per oxygen of the $x = 0.5$ cell,
  in any order; each entry is the oxygen's three fractional coordinates as
  they appear in `ideal_x050.vasp`, then its Hf count, then its Zr count.
- `class_counts`: item (2), keyed by the nine labels above.
- The four displacement fields: item (3); `doped` is the $x = 0.5$ cell,
  `undoped` the HfO$_2$ cell.
- `longest_bond_A` and `closest_nonbonded_cation_A`: item (4).
- `space_group_number_undoped_relaxed`: item (5), the International Tables
  number.

All values must be computed from the staged structures; do not guess or
hardcode.
