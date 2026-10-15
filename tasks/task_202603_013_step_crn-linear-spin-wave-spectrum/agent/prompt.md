# Task: linear spin-wave spectrum of antiferromagnetic CrN

Chromium nitride orders as a double-layered antiferromagnet in an eight-atom
orthorhombic cell: ferromagnetic Cr planes stack along the first lattice
vector as up, up, down, down; the staged magnetic state below fixes which
sites carry which orientation. You will compute its magnon spectrum in linear
spin-wave theory from a four-bond Heisenberg model.

## Provided data

- `cell/POSCAR` -- the relaxed ordered cell.
- `cell/magnetic_state.json` -- the magnetic decoration of `cell/POSCAR`.
- `exchange_table.csv` -- the exchange-bond data. Displacement vectors are
  fractional coordinates of `cell/POSCAR`, specified componentwise up to sign.
  Distances are in Angstrom and exchange constants in meV.

The exchange constants are given in the convention

$$
H = -2 \sum_{i > j} J_{ij}\, \mathbf{S}_i \cdot \mathbf{S}_j , \qquad |\mathbf{S}| = \tfrac{3}{2} ,
$$

with each unordered pair counted once and $J_{ij} < 0$ antiferromagnetic. The
model contains these four bonds and nothing else: no further neighbours, no
anisotropy, no field.

## What to compute

Expand the model to quadratic order in spin deviations about the given
collinear state (Holstein-Primakoff) and obtain the magnon energies
$\hbar\omega(\mathbf{k}) \ge 0$ (the non-negative half of the Bogoliubov
spectrum) by the proper bosonic diagonalization of the
resulting quadratic form. With four magnetic sites per cell there are four
magnon energies at every $\mathbf{k}$; report all four, sorted ascending,
repeating degenerate values. Wavevectors are given in fractional coordinates
of the reciprocal lattice of `cell/POSCAR`.

**(1) Energies at nine wavevectors**, in meV, in this order:

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

**(2) The splitting at X**: the highest minus the lowest of the four energies
at $\mathbf{k} = (1/2, 0, 0)$, in meV.

The environment provides Python with numpy; no spin-wave package is
installed.

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
