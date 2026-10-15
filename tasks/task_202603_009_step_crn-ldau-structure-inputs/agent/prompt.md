# Task: input files for the LDA+U structural variants of antiferromagnetic CrN

You are preparing density-functional inputs for the magnetically ordered
ground state of chromium nitride and for two reference cells derived from it.

## The system

CrN is a rock-salt nitride. Below its ordering temperature the Cr moments form
a double-layered antiferromagnet: ferromagnetic Cr planes of the (110) type of
the cubic parent stack along one direction as up, up, down, down. The ordered
state is orthorhombic. Relative to the cubic parent it carries a shear of the
cell and a displacement of every atom along the stacking direction, one free
internal coordinate per species. The electronic structure is described with
the local density approximation plus a Hubbard correction of
$U_{\mathrm{eff}} = 4.1$ eV on the Cr 3d states (single effective parameter,
Dudarev form), collinear spin-polarized. In the relaxed ordered cell, the two
neighbouring Cr planes that carry parallel moments are the pair with the larger
plane spacing along the stacking direction, and the antiparallel neighbouring
planes the pair with the smaller spacing; the spacings, not the order in
which the sites are listed, fix where the up-up-down-down sequence begins.

You are given, in this directory:
- `seed_cell/POSCAR` -- the starting reference cell. The Cr stacking axis is
  its first lattice vector. Keep its site order in every submitted cell.
- `relaxed_cell/POSCAR` -- the converged result of the full relaxation
  described under label `relax` below, in the same setting and site order.
- `potcar_catalog.yaml` -- the available pseudopotential catalog.

## The four calculations

- `relax`: the full relaxation of the ordered state (atomic positions, cell
  shape and volume all free), starting from `seed_cell/POSCAR`. The magnetic
  order is imposed from the start and must be the up-up-down-down pattern
  along the stacking axis.
- `static_iii`: a converged ground-state calculation of the ordered state on
  `relaxed_cell/POSCAR`, held fixed, from which the total energy, the band gap
  and the Cr moments are read.
- `static_ii`: the same calculation on a cell that keeps the lattice of
  `relaxed_cell/POSCAR` but has the internal displacements removed, so that
  every atom sits at its ideal rock-salt fractional position.
- `static_i`: the same calculation on a cubic rock-salt cell of the same
  volume as `relaxed_cell/POSCAR`, written in the same eight-atom setting with
  the same axis roles and site order.

Assume the runs are executed with VASP 6.4 and the VASP 5.4 PAW library of
the catalog; you do not run anything here. The three statics differ only in
their cell and use one common set of
electronic settings. Build the `static_ii` and `static_i` cells yourself from
`relaxed_cell/POSCAR`; keep the magnetic decoration of the relaxed cell (the
same sites carry the same moments) in every cell you submit. Do not relax
`static_ii` or `static_i`.

## Output format

Write a manifest `answer.json` in the workspace root mapping the four run
labels to their directories:

```json
{"runs": {
  "relax": "answer/relax",
  "static_iii": "answer/static_iii",
  "static_ii": "answer/static_ii",
  "static_i": "answer/static_i"}}
```

For each label provide a directory `answer/<label>/` containing exactly three
files: `INCAR`, `KPOINTS` and `POSCAR`. Do not submit a POTCAR; the
pseudopotential identities are fixed by `potcar_catalog.yaml` and are not
graded from your files.
