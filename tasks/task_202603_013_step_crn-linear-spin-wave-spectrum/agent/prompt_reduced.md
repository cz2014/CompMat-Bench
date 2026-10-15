# Task: linear spin-wave spectrum of antiferromagnetic CrN

Compute the linear spin-wave spectrum of CrN in the collinear magnetic state
specified by `cell/magnetic_state.json`, on the structure in `cell/POSCAR`.
`exchange_table.csv` gives the exchange bonds; its vectors are fractional
coordinates of `cell/POSCAR`, specified componentwise up to sign. Distances
are in Angstrom and exchange constants in meV.

The model contains only those exchange bonds, with

$$
H=-2\sum_{i>j}J_{ij}\,\mathbf S_i\cdot\mathbf S_j,
\qquad |\mathbf S|=\tfrac32.
$$

## Requested spectrum

Report all four non-negative magnon energies in meV at each wavevector below,
ascending and including repeated degenerate values. Wavevectors are fractional
coordinates of the reciprocal lattice of `cell/POSCAR`.

| index | label | $\mathbf{k}$ |
|---|---|---|
| 1 | $\Gamma$ | (0, 0, 0) |
| 2 | X | (1/2, 0, 0) |
| 3 | S | (1/2, 1/2, 0) |
| 4 | Y | (0, 1/2, 0) |
| 5 | Z | (0, 0, 1/2) |
| 6 | U | (1/2, 0, 1/2) |
| 7 | T | (0, 1/2, 1/2) |
| 8 | R | (1/2, 1/2, 1/2) |
| 9 | P | (1/2, 1/4, 1/4) |

`splitting_X_meV` is the highest minus the lowest energy at X.

Python with numpy is available; no spin-wave package is installed.

## Output format

Write `answer.json` in the workspace root with exactly this schema. The outer
list of `energies_meV` has nine entries in the table order; each entry holds
the four energies at that wavevector, ascending.

```json
{
  "energies_meV": [[<float>, <float>, <float>, <float>], ...],
  "splitting_X_meV": <float>
}
```
