# Task: charge-state thermodynamics of the oxygen vacancy in Hf$_{0.5}$Zr$_{0.5}$O$_2$ across its local cation environments

Oxygen vacancies in the polar orthorhombic phase of hafnia-zirconia exist in
several charge states, and which charge state is stable depends on the
Fermi level and on the local cation environment of the vacated site. The
vacancy has been relaxed in a $2 \times 2 \times 2$ supercell of the
Pca2$_1$ structure at $x = 0.5$, at each of nine representative oxygen
sites, one per environment, in the charge states $q = +2, +1, 0, -1, -2$.
Because a charged defect in a periodic cell interacts with its own images,
each charged energy has to be corrected for finite size before the states
can be compared. The task is to carry the raw calculations through to the
corrected formation energies, to find which environment is the most
favourable in each charge state, and to locate the Fermi levels at which
the stable charge state changes.

## What you are given

- `pristine/` -- the defect-free supercell: its relaxed structure
  (`POSCAR`), the output of its single-point calculation (`OUTCAR.gz`) and
  the electrostatic potential of that calculation on the real-space grid
  (`LOCPOT.gz`).
- `sites.csv` -- the nine representative oxygen sites: the environment
  label (the Hf and Zr counts among the cations within 2.3 Å of the site),
  the cation coordination of the site, and its fractional coordinates in
  the defect-free cell. Every defect cell below was made by removing the
  site of its label from the defect-free cell before relaxation.
- `statics/<label>/q<q>/` -- for each environment label and each charge
  state, the single-point calculation on the relaxed defect geometry: the
  structure (`POSCAR`), the output (`OUTCAR.gz`) and the electrostatic
  potential on the real-space grid (`LOCPOT.gz`). All 45 cells share the
  lattice of the defect-free cell.
- `dielectric/<cell>/` -- five density-functional perturbation theory
  calculations on 12-atom cells of the same phase and lattice, spanning
  the composition from HfO$_2$ to ZrO$_2$; each `OUTCAR.gz` carries the
  static dielectric tensor, electronic and ionic contributions, of its
  cell (`POSCAR`).

All calculations are spin-unpolarised PBE. No simulation is required. The environment provides
Python with numpy, scipy, pymatgen and pymatgen-analysis-defects
(version 2026.3.20).

## Conventions (fixed for this task)

- **Total energies.** The energy of each calculation is its electronic free
  energy at the smearing used, with no extrapolation to zero smearing and no
  removal of the smearing term.
- **Chemical potential.** The oxygen-poor limit, $\mu_\mathrm{O} = -10.09$ eV:
  the absolute chemical potential of the removed oxygen atom, on the same
  energy scale as the staged total energies.
- **Fermi-level reference.** $E_\mathrm{F}$ is measured from the
  valence-band maximum of the defect-free cell, and every formation energy
  is reported at $E_\mathrm{F} = 0$.
- **Finite-size correction.** The charged cells are corrected with the
  Freysoldt-Neugebauer-Van de Walle scheme as implemented in
  `pymatgen-analysis-defects` (`get_freysoldt_correction`, at its default
  model charge and alignment settings), using as its dielectric constant
  the isotropic value $\varepsilon$ of the total static tensor (electronic
  plus ionic), averaged over the five 12-atom cells. The potential-alignment
  term of the scheme is taken as the mean of the alignments obtained along
  the three lattice directions. The neutral cells carry no correction.
- **Lower envelope.** Determine the thermodynamic charge-transition levels
  of the formation-energy lower envelope over all nine environments for
  $E_\mathrm{F} \geq 0$.

## What to determine

(1) $\varepsilon$.

(2) The corrected formation energy of every (environment, charge state)
    pair, at $E_\mathrm{F} = 0$.

(3) For each charge state, the environment with the lowest formation energy.

(4) Every transition level of the lower envelope: the two charge states it
    connects and the Fermi level at which it occurs.

## Output format

Write a single `answer.json` in the workspace root with exactly these fields
(energies in eV, every number unrounded):

```json
{
  "eps_iso": <float>,
  "formation_energies_eV": {
    "4Hf": [<float>, <float>, <float>, <float>, <float>],
    "3Hf1Zr": [<float>, <float>, <float>, <float>, <float>],
    "2Hf2Zr": [<float>, <float>, <float>, <float>, <float>],
    "1Hf3Zr": [<float>, <float>, <float>, <float>, <float>],
    "4Zr": [<float>, <float>, <float>, <float>, <float>],
    "3Hf": [<float>, <float>, <float>, <float>, <float>],
    "2Hf1Zr": [<float>, <float>, <float>, <float>, <float>],
    "1Hf2Zr": [<float>, <float>, <float>, <float>, <float>],
    "3Zr": [<float>, <float>, <float>, <float>, <float>]
  },
  "lowest_environment": [<string>, <string>, <string>, <string>, <string>],
  "transition_levels_eV": [[<int>, <int>, <float>], ...]
}
```

- `eps_iso`: item (1).
- `formation_energies_eV`: item (2); each list holds the five charge states
  in the order $q = +2, +1, 0, -1, -2$.
- `lowest_environment`: item (3), one label per charge state in the same
  order, spelled as in `sites.csv`.
- `transition_levels_eV`: item (4), one entry per transition in order of
  increasing Fermi level; each entry is the charge state below the
  transition, the charge state above it, and the Fermi level.

All values must be computed from the staged calculations; do not guess or
hardcode.
