# Task: the Neel temperature of antiferromagnetic CrN from a classical Monte Carlo sweep

Chromium nitride orders as a double-layered antiferromagnet: ferromagnetic Cr
planes stack along the first lattice vector of its eight-atom orthorhombic
cell as up, up, down, down. A classical Heisenberg model of the four nearest
Cr-Cr exchange bonds was simulated with VAMPIRE by Monte Carlo on a periodic
supercell at zero field, over a grid of temperatures, with three independent
random-number seeds per temperature.

## Provided data

- `deck/` -- the VAMPIRE deck the sweep ran with: `input`, `material.mat` and
  `unitcell.ucf`. The `input` file is the one used at a single temperature;
  the sweep repeated it over the grid.
- `mc_sweep.csv` -- one row per (seed, temperature) run, with temperature in
  kelvin and VAMPIRE observables averaged over the sampling phase.
  `columns.md` maps observables to their output keywords and material indices.
- `vampire_input_reference.tex` -- the chapter of the VAMPIRE manual that
  documents its input and output keywords. VAMPIRE itself is neither
  installed nor provided here.

## What to report

**(1) The order parameter.** The temperature dependence of the ordered
sublattice moment of the antiferromagnet, normalised to its value at the
lowest temperature of the grid, averaged over the three seeds: one value per
grid temperature, in ascending temperature order.

**(2) The Neel temperature** $T_N$ in kelvin, extracted from that curve.

## Output format

Write `answer.json` in the workspace root with exactly this schema.
`M_stag_norm` has one entry per grid temperature in ascending order.

```json
{
  "M_stag_norm": [<float>, ...],
  "T_N_K": <float>
}
```
