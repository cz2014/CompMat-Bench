# Task: characterize a relaxed edge-dislocation dipole in poled BaTiO3

A BaTiO3 cell containing a pair of $\langle 100 \rangle$ edge dislocations,
described by a core-shell (shell-model) interatomic potential, has been poled
by an external field along one of its axes and relaxed with the field removed.
You are given the relaxed structure. Before its strain and polarization fields
can be studied, the defect itself has to be characterized: where the two cores
sit, how they are terminated, how the removed material partitions the cell,
how close the relaxation has brought cores and shells of neighbouring ions,
and along which direction the cell is polarized.

## What you are given

- `dislocation_cell.lmp` -- the relaxed configuration as a LAMMPS data file
  (`atom_style full`, metal units). Each ion is a core-shell pair: the core
  and its shell share a molecule id and are joined by a bond, and each carries
  its own charge as listed in the file. Atom types 1, 2, 3 are the Ba, Ti and
  O cores and types 4, 5, 6 the corresponding shells.

The dislocation lines run along $z$ and the Burgers vectors are $\pm[100]$: one
BaO-TiO2 double layer normal to $x$ has been removed over part of the cell's
extent along $y$, so the two cores are separated along $y$. Call **region A**
the band of the cell between the two cores in which the double layer is missing
and **region B** the remainder. Each core is named after the plane that
terminates the double layer there: at each end of the removed layer one of its
two planes, BaO or TiO2, reaches one lattice row farther than the other, and
that plane names the core (the **BaO-type** and the **TiO2-type** core).

No simulation is required. The environment provides Python with numpy and
scipy; no molecular-dynamics code is installed.

## What to determine

(1) The **row census**: sort the Ba cores into lattice rows along $y$ and
    report the number of rows, and the number of rows that belong to region A
    (the rows carrying one Ba fewer than the others).

(2) The **two core positions**, from the structure. Take each core as the
    midpoint of the widened Ba-Ba gap in the outermost row of region A at that
    end of the removed layer (the first row, counted from that end, in which
    the Ba column is missing), and label it BaO-type or TiO2-type as defined
    above. Report also the separation of the two cores along $y$ (under
    periodic boundary conditions) divided by the box length along $y$.

(3) The **Ba spacing along $x$** in each region: the mean nearest-neighbour
    Ba-Ba separation along $x$ within a row, averaged over the rows of
    region A, and separately over the rows of region B.

(4) The **closest core-shell contact between different ions**: the smallest
    distance between a Ti core and an O shell in the cell (under periodic
    boundary conditions), which core (BaO-type or TiO2-type) that Ti is nearer
    to, and its in-plane distance from that core.

(5) The **polar axis**: the mean displacement vector of the Ti shells from
    their own cores, averaged over all Ti ions, and the axis and sense of the
    macroscopic polarization that this displacement implies.

## Output format

Write a single `answer.json` in the workspace root with exactly these fields
(lengths in Å):

```json
{
  "n_rows_y": <int>,
  "n_rows_region_A": <int>,
  "core_BaO_xy_A": [<float>, <float>],
  "core_TiO2_xy_A": [<float>, <float>],
  "core_separation_over_Ly": <float>,
  "ba_spacing_x_region_A_A": <float>,
  "ba_spacing_x_region_B_A": <float>,
  "min_Ti_core_O_shell_A": <float>,
  "min_contact_nearest_core": <string: one of "BaO", "TiO2">,
  "min_contact_distance_to_core_A": <float>,
  "Ti_shell_displacement_mean_A": [<float>, <float>, <float>],
  "polarization_direction": <string: one of "+x", "-x", "+y", "-y", "+z", "-z">
}
```

- `n_rows_y`, `n_rows_region_A`: item (1).
- `core_BaO_xy_A`, `core_TiO2_xy_A`: the $(x, y)$ position of each core, item
  (2), in the coordinate frame of the data file; `core_separation_over_Ly`:
  the ratio defined in item (2).
- `ba_spacing_x_region_A_A`, `ba_spacing_x_region_B_A`: item (3).
- `min_Ti_core_O_shell_A`, `min_contact_nearest_core`,
  `min_contact_distance_to_core_A`: item (4).
- `Ti_shell_displacement_mean_A`: item (5), Cartesian components of the mean
  shell-minus-core displacement; `polarization_direction`: item (5).

All values must be computed from the staged structure; do not guess or
hardcode.
