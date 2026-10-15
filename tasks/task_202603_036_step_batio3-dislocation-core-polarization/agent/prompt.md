# Task: local polarization at the two cores of an edge-dislocation dipole in BaTiO3

Tetragonal BaTiO3 containing a pair of $\langle 100 \rangle$ edge dislocations,
described by a core-shell (shell-model) interatomic potential, has been poled by
an external field and then relaxed with the field removed. You are given two such
relaxed, field-free configurations. Each was poled along one of the three
Cartesian axes of the cell before the field was removed; which axis, and whether
the two configurations differ in it, is for you to determine from the structures.

The question is how the local polarization behaves at the two dislocation cores:
whether it is reduced, rotated or reversed there, and whether the two cores, which
terminate on chemically different planes, behave differently.

## What you are given

- `cell_1.lmp`, `cell_2.lmp` -- the two relaxed configurations as LAMMPS data
  files (`atom_style full`, metal units). Each ion is a core-shell pair: the core
  and its shell share a molecule id and are joined by a bond, and each carries
  its own charge as listed in the file. Atom types 1, 2, 3 are the Ba, Ti and O
  cores and types 4, 5, 6 the corresponding shells.

Both cells hold the same dislocation arrangement. The dislocation lines run along
$z$ and the Burgers vectors are $\pm[100]$: one BaO-TiO2 double layer normal to
$x$ has been removed over part of the cell's extent along $y$, so the two cores
are separated along $y$. Following the usual convention for this geometry, call
**region A** the band of the cell between the two cores in which the double layer
is missing and **region B** the remainder. Each core is named after the plane that
terminates the double layer there: at each end of the removed layer one of its
two planes, BaO or TiO2, reaches one lattice row farther than the other, and
that plane names the core (the **BaO-type** and the **TiO2-type** core).

No simulation is required. The environment provides Python with numpy and scipy;
no molecular-dynamics code is installed.

## The local dipole (fixed for this task)

The polarization is resolved per Ti-centred unit cell. For a Ti ion $k$ take as
its cell the polyhedron formed by the Ti core and every O core within 3.1 A of
it under periodic boundary conditions, with periodic images of the same O
counted as separate corners. Assign each ion the single charge
$q = q_{\mathrm{core}} + q_{\mathrm{shell}}$ placed at its core position; the
core-to-shell displacement does not enter. The dipole of the cell is

$$
\mathbf{p}_k = \sum_{j \in \{\mathrm{Ti},\, \mathrm{O\ corners}\}} q_j \left(\mathbf{r}_j - \mathbf{r}_{\mathrm{c}}\right),
$$

where $\mathbf{r}_{\mathrm{c}}$ is the mass-weighted centre of the whole
polyhedron, the central Ti included, with standard atomic masses. The local
polarization is the cell dipole normalized by the volume per formula unit
$V_{\mathrm{cell}}$, defined as the box volume divided by the number of Ti.
The macroscopic polarization is
$\mathbf{P} = (1/V) \sum_k \mathbf{p}_k$ over all Ti cells, with $V$ the box
volume. Report both local and macroscopic polarizations in C/m$^2$.

A Ti whose search returns fewer than six O corners has an incomplete
polyhedron. Such cells are still evaluated and still enter $\mathbf{P}$, but
they are left out of every mean, count and span defined below.

## What to determine, for each configuration

(1) The **poling axis**: the Cartesian axis along which the bulk of the cell is
    polarized. The **polar component** of a cell dipole is its component along
    that axis, with the sign chosen so that the region-B bulk value is positive.

(2) The **two core positions**, from the structure. Take each core as the
    midpoint of the widened Ba-Ba gap in the outermost row of region A at that
    end of the removed layer (the first row, counted from that end, in which the
    Ba column is missing), and label it BaO-type or TiO2-type as defined above.
    Distances from a core below are measured under periodic boundary
    conditions.

(3) The **bulk polarization of each region**: the mean in-plane cell polarization vector
    over the Ti cells of region A, and separately of region B, that lie more
    than 40 A along $y$ from both cores.

(4) The **polar component next to each core**: the mean polar component over the
    Ti cells within 11 A of the core along $x$ and within 8 A along $y$ that lie
    on the side of the core away from region A.

(5) The **reversed cells at each core**: among the Ti cells within 11 A of the
    core along $x$ and 12 A along $y$ (both sides), those whose polar component
    is negative. Report how many there are at each core. Report their extent along
    $x$ for the BaO-type core in both configurations, and for the TiO2-type
    core in `cell_2`: the difference between the largest and the smallest
    $x$ offset from the core among them (report 0.0 when there is none).

(6) The **in-plane components of the macroscopic polarization** $\mathbf{P}$
    of the configuration.

## Output format

Write a single `answer.json` in the workspace root with the following required fields,
one object per configuration, keyed by file name without the extension:

```json
{
  "cell_1": {
    "poling_axis": <string: one of "x", "y", "z">,
    "core_BaO_xy_A": [<float>, <float>],
    "core_TiO2_xy_A": [<float>, <float>],
    "P_bulk_A_C_m2": [<float>, <float>],
    "P_bulk_B_C_m2": [<float>, <float>],
    "p_polar_near_core_BaO_C_m2": <float>,
    "p_polar_near_core_TiO2_C_m2": <float>,
    "n_reversed_BaO": <int>,
    "n_reversed_TiO2": <int>,
    "x_extent_reversed_BaO_A": <float>,
    "P_total_C_m2": [<float>, <float>]
  },
  "cell_2": {
    "poling_axis": <string: one of "x", "y", "z">,
    "core_BaO_xy_A": [<float>, <float>],
    "core_TiO2_xy_A": [<float>, <float>],
    "P_bulk_A_C_m2": [<float>, <float>],
    "P_bulk_B_C_m2": [<float>, <float>],
    "p_polar_near_core_BaO_C_m2": <float>,
    "p_polar_near_core_TiO2_C_m2": <float>,
    "n_reversed_BaO": <int>,
    "n_reversed_TiO2": <int>,
    "x_extent_reversed_BaO_A": <float>,
    "x_extent_reversed_TiO2_A": <float>,
    "P_total_C_m2": [<float>, <float>]
  }
}
```

- `poling_axis`: item (1).
- `core_BaO_xy_A`, `core_TiO2_xy_A`: the $(x, y)$ position of each core in A,
  item (2), in the coordinate frame of the data file.
- `P_bulk_A_C_m2`, `P_bulk_B_C_m2`: item (3), Cartesian $(x, y)$ components.
- `p_polar_near_core_BaO_C_m2`, `p_polar_near_core_TiO2_C_m2`: item (4).
- `n_reversed_BaO`, `n_reversed_TiO2`, `x_extent_reversed_BaO_A`,
  `x_extent_reversed_TiO2_A`: item (5); the extents in A. The TiO2 extent is
  required only for `cell_2`.
- `P_total_C_m2`: item (6), Cartesian $(x, y)$ components.

All values must be computed from the staged structures; do not guess or
hardcode.
