# Task: bonding analysis of the oxygen sublattice of Hf$_{0.5}$Zr$_{0.5}$O$_2$, with and without an oxygen vacancy

Why an oxygen vacancy in hafnia-zirconia prefers one local cation
environment over another is a question about bonding. The converged
wavefunctions of the defect-free $2 \times 2 \times 2$ supercell of Pca2$_1$
Hf$_{0.5}$Zr$_{0.5}$O$_2$ and of the ten vacancy-bearing cells listed in
`cells.csv` have been projected onto a local orbital basis with LOBSTER,
giving the crystal orbital Hamilton population (COHP) of every metal-oxygen
interaction. The task is to read from these projections how strongly each
oxygen environment is bonded, where the deepest antibonding features of the
total curve sit on either side of the Fermi level in the neutral and in the
charged cell, and how the bond strength scales with bond length.

## What you are given

- `cells/<cell>/` -- for each of the eleven cells: the structure LOBSTER
  read (`POSCAR`), the energy-resolved COHP of every metal-oxygen
  interaction of the cell together with their average (`COHPCAR.lobster`),
  and the integrated COHP up to the Fermi level of every interaction
  (`ICOHPLIST.lobster`).
- `cells.csv` -- one row per cell: its name, the environment label and the
  cation coordination of the removed oxygen (empty for the defect-free
  cell), and the charge state.
- `sites.csv` -- the nine representative oxygens of the defect-free cell,
  one per environment: the environment label (the Hf and Zr counts among
  the cations within 2.3 Å of the site), its cation coordination, and its
  fractional coordinates in the defect-free cell.

No simulation is required. The environment provides Python with numpy,
scipy and pymatgen.

## Conventions (fixed for this task)

- **Sign.** Everything is reported as $-\mathrm{COHP}$: bonding is positive,
  antibonding is negative. Energies are relative to the Fermi level, which
  is at 0 eV.
- **Grid.** Curves are analysed on the energy grid of the files, without
  resampling or smoothing.
- **Total curve** of a cell: the plain mean of $-\mathrm{COHP}$ over all
  metal-oxygen interactions of that cell.
- **Partial curve** of an environment: the mean of $-\mathrm{COHP}$ over the
  interactions of that environment's representative oxygen in the
  defect-free cell.
- **Occupied integral** of a curve: its integral from the first grid energy
  of the file up to the Fermi level. For a partial curve this is
  its **bonding integral** $B$.
- **State.** A local minimum of a curve on the grid at which the curve is
  below $-0.001$ (in the units of the files). Its position is the grid
  energy of the minimum and its depth the curve's value there.
- **Integrated COHP.** Items (5) and (6) use the integrated COHP values as
  the files give them (negative for a bonding interaction); the sign
  convention above applies to the curves only.

## What to determine

(1) For every cell: the occupied integral of its total curve.

(2) For every cell: the deepest state of its total curve within the
    occupied window $[-4, 0]$ eV, and the deepest state within the
    unoccupied window $(0, +4]$ eV, whatever the origin of the state; either
    may be absent.

(3) For every environment: the bonding integral $B$ of its partial curve.

(4) For the `4Zr` partial curve: the positions of all of its states within
    $[-4, 0]$ eV.

(5) The sum of the integrated COHP of the interactions of the `4Zr`
    representative oxygen in the defect-free cell.

(6) On the defect-free cell, the slope of a linear fit of the integrated
    COHP against the bond length, over all Hf-O interactions and,
    separately, over all Zr-O interactions.

## Output format

Write a single `answer.json` in the workspace root with exactly these
fields (energies in eV, lengths in Å; curve values and integrals in the
units of the files):

```json
{
  "occupied_integral": {"<cell>": <float>, ...},
  "occupied_state": {"<cell>": [<float>, <float>], ...},
  "unoccupied_state": {"<cell>": [<float>, <float>], ...},
  "partial_bonding_integral": {"4Hf": <float>, "3Hf1Zr": <float>, "2Hf2Zr": <float>, "1Hf3Zr": <float>, "4Zr": <float>, "3Hf": <float>, "2Hf1Zr": <float>, "1Hf2Zr": <float>, "3Zr": <float>},
  "partial_4Zr_states_eV": [<float>, ...],
  "icohp_sum_4Zr_eV": <float>,
  "icohp_slope_eV_per_A": {"Hf": <float>, "Zr": <float>}
}
```

- `occupied_integral`: item (1), keyed by the eleven cell names of
  `cells.csv`.
- `occupied_state`, `unoccupied_state`: item (2), keyed the same way; each
  value is `[position, depth]`, or `null` when the window holds no state.
- `partial_bonding_integral`: item (3).
- `partial_4Zr_states_eV`: item (4), in increasing energy.
- `icohp_sum_4Zr_eV`: item (5).
- `icohp_slope_eV_per_A`: item (6).

All values must be computed from the staged files; do not guess or
hardcode.
