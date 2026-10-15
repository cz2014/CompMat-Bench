# Task: Ba-Ba strain maps around the two cores of an edge-dislocation dipole in BaTiO3

A BaTiO3 cell containing a pair of $\langle 100 \rangle$ edge dislocations,
described by a core-shell interatomic potential, has been poled along $x$ by an
external field and relaxed with that field still applied, so the cell is
tetragonal with its polar axis along $x$. The dislocations impose a strain
field on the lattice around each core, and the question is how that strain is
distributed: its sign pattern on either side of the removed layer, and how
large it becomes next to the two cores.

## What you are given

- `ba_sublattice.lmp` -- the Ba positions of the relaxed cell, as a LAMMPS data
  file with the periodic box. Only Ba is included; the strain is read from the
  Ba sublattice alone.

The dislocation lines run along $z$ and the Burgers vectors are $\pm[100]$: one
BaO-TiO2 double layer normal to $x$ has been removed over part of the cell's
extent along $y$, so the two cores are separated along $y$. Call **region A**
the band of the cell between the two cores in which the double layer is missing
and **region B** the remainder. Each core is named after the plane, BaO or
TiO2, that terminates the removed double layer there. The Ba sublattice alone
cannot show which end is which: take the core at the lower end of region A
(the smaller $y$) as the **BaO-type** core and the core at the upper end as the
**TiO2-type** core.

The reference Ba-Ba spacings of the undislocated cell with the same polar axis
are

$$
d_{xx}^{0} = 4.020297\ \text{Å}, \qquad d_{yy}^{0} = 3.998686\ \text{Å}.
$$

No simulation is required. The environment provides Python with numpy and scipy.

## The local strain (fixed for this task)

For each Ba site $k$ and each in-plane axis $i \in \{x, y\}$, the local Ba-Ba
spacing $d_{ii}(k)$ is defined from the neighbours of $k$ within 4.7 Å under
periodic boundary conditions: among those neighbours whose minimum-image
displacement from $k$ has its largest absolute component along axis $i$, take
the smallest absolute $i$ component of the displacement. That smallest
axis-projected separation is $d_{ii}(k)$; it is not the full neighbour distance
and not an average over the two sides. Report the local engineering strain
$\varepsilon_{ii}(k)$ relative to the specified reference spacing
$d_{ii}^{0}$, in percent.

## What to determine

(1) The **two core positions**, from the structure. Take each core as the
    midpoint of the widened Ba-Ba gap in the outermost row of region A at that
    end of the removed layer (the first row, counted from that end, in which
    the Ba column is missing), and label it BaO-type or TiO2-type as stated
    above. Distances from a core are measured under periodic boundary
    conditions.

(2) The **mean strain of each region**: the mean of $\varepsilon_{xx}$ and of
    $\varepsilon_{yy}$ over all Ba sites of region A, and separately of
    region B.

(3) The **extreme strains**: the largest $\varepsilon_{xx}$ and the smallest
    $\varepsilon_{yy}$ over the whole cell, and the same two extremes taken
    over the Ba sites within 16 Å of each core.

(4) The **threshold counts**: the number of Ba sites with
    $\varepsilon_{xx} > 10.0\,\%$ and the number with
    $\varepsilon_{yy} < -3.5\,\%$, over the whole cell.

## Output format

Write a single `answer.json` in the workspace root with exactly these fields
(strains in percent, positions in Å):

```json
{
  "core_BaO_xy_A": [<float>, <float>],
  "core_TiO2_xy_A": [<float>, <float>],
  "eps_xx_mean_A_pct": <float>,
  "eps_xx_mean_B_pct": <float>,
  "eps_yy_mean_A_pct": <float>,
  "eps_yy_mean_B_pct": <float>,
  "eps_xx_max_pct": <float>,
  "eps_yy_min_pct": <float>,
  "eps_xx_max_near_BaO_pct": <float>,
  "eps_xx_max_near_TiO2_pct": <float>,
  "eps_yy_min_near_BaO_pct": <float>,
  "eps_yy_min_near_TiO2_pct": <float>,
  "n_eps_xx_above_10pct": <int>,
  "n_eps_yy_below_minus3p5pct": <int>
}
```

- `core_BaO_xy_A`, `core_TiO2_xy_A`: the $(x, y)$ position of each core, item
  (1), in the coordinate frame of the data file.
- `eps_xx_mean_A_pct`, `eps_xx_mean_B_pct`, `eps_yy_mean_A_pct`,
  `eps_yy_mean_B_pct`: item (2).
- `eps_xx_max_pct`, `eps_yy_min_pct`: the whole-cell extremes of item (3).
- `eps_xx_max_near_BaO_pct`, `eps_xx_max_near_TiO2_pct`,
  `eps_yy_min_near_BaO_pct`, `eps_yy_min_near_TiO2_pct`: the near-core extremes
  of item (3).
- `n_eps_xx_above_10pct`, `n_eps_yy_below_minus3p5pct`: item (4).

All values must be computed from the staged structure; do not guess or
hardcode.
