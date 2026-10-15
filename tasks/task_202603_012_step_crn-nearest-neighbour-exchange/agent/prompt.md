# Task: nearest-neighbour exchange parameters of CrN from Wannier-function calculations

Chromium nitride orders as a double-layered antiferromagnet: ferromagnetic Cr
planes stack along the first lattice vector of the eight-atom orthorhombic
cell as up, up, down, down. Each Cr has twelve nearest Cr neighbours. They
fall into four classes by the relative orientation of the two moments and by
how many neighbours of that kind each Cr has: four with parallel moments,
four with antiparallel moments, two with antiparallel moments, and two with
parallel moments. The exchange in each class was computed for three cells.

## Provided data

Three directories, one per cell:

- `variant_iii/` -- the fully relaxed orthorhombic ordered cell;
- `variant_ii/` -- the same lattice with the internal displacements removed,
  every atom at its ideal rock-salt fractional position;
- `variant_i/` -- a cubic rock-salt cell of the same volume, in the same
  eight-atom setting.

Each holds the cell as `POSCAR` and `exchange.out`, the output of TB2J's
`wann2J.py` run on the spin-polarized Wannier Hamiltonian of that cell. TB2J
reports isotropic exchange in its native convention,

$$
E = -\sum_{i \neq j} J^{\mathrm{TB2J}}_{ij}\, \hat{\mathbf{e}}_i \cdot \hat{\mathbf{e}}_j ,
$$

with $\hat{\mathbf{e}}_i$ unit vectors along the moments and every ordered
pair counted. TB2J is not installed here; you work from these outputs alone.

## What to report

For each cell, the exchange constant of each of the four classes, in meV, in
the convention

$$
H = -2 \sum_{i > j} J_{ij}\, \mathbf{S}_i \cdot \mathbf{S}_j , \qquad |\mathbf{S}| = \tfrac{3}{2} ,
$$

with each unordered pair counted once and $J_{ij} < 0$ antiferromagnetic.
Assign every pair to its class from the orientation of the two converged
moments and the pair geometry of that cell.

## Output format

Write `answer.json` in the workspace root with exactly this schema (all values
in meV):

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
