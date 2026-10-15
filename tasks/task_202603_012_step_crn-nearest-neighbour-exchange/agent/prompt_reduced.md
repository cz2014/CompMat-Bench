# Task: nearest-neighbour exchange parameters of CrN from Wannier-function calculations

Determine the nearest-neighbour Cr exchange constants from `variant_i/`,
`variant_ii/` and `variant_iii/`. Each provides `POSCAR` and TB2J
`wann2J.py` output `exchange.out`, including converged moments and pair data.
The cells are respectively cubic rock-salt, the relaxed orthorhombic lattice
at ideal rock-salt positions, and the fully relaxed ordered cell, in the
same eight-atom setting and with the same volume.

For each cell report the four nearest-neighbour classes labelled by
relative moment orientation and multiplicity per Cr: parallel/4,
antiparallel/4, antiparallel/2 and parallel/2.

The source convention is

$$E=-\sum_{i\ne j}J^{\mathrm{TB2J}}_{ij}\,
\hat{\mathbf e}_i\cdot\hat{\mathbf e}_j,$$

where $\hat{\mathbf e}$ are unit moment directions. Report exchange in meV
in the convention

$$H=-2\sum_{i>j}J_{ij}\,\mathbf S_i\cdot\mathbf S_j,
\qquad |\mathbf S|=\tfrac32.$$

Use the supplied records only; TB2J is not installed.

## Output format

Write `answer.json` in the workspace root with exactly these fields:

```json
{
  "J_iii_parallel_4_meV": <float>,
  "J_iii_antiparallel_4_meV": <float>,
  "J_iii_antiparallel_2_meV": <float>,
  "J_iii_parallel_2_meV": <float>,
  "J_ii_parallel_4_meV": <float>,
  "J_ii_antiparallel_4_meV": <float>,
  "J_ii_antiparallel_2_meV": <float>,
  "J_ii_parallel_2_meV": <float>,
  "J_i_parallel_4_meV": <float>,
  "J_i_antiparallel_4_meV": <float>,
  "J_i_antiparallel_2_meV": <float>,
  "J_i_parallel_2_meV": <float>
}
```
