# Task: input sets for a charged oxygen vacancy in Hf$_{0.5}$Zr$_{0.5}$O$_2$ and for the dielectric tensor its finite-size correction needs

Oxygen vacancies in the polar orthorhombic phase of hafnia-zirconia exist in
several charge states, and which charge state is stable depends on the
Fermi level and on the local cation environment of the vacated site. A
site-resolved formation-energy study therefore relaxes the vacancy in each
charge state at each representative site, and because a charged defect in a
periodic supercell interacts with its own images, every charged energy is
corrected afterwards by a finite-size electrostatic correction whose inputs
are the electrostatic potential of the defect cell and the material's static
dielectric constant. Your job is to prepare the input sets for one such
calculation: the doubly positive vacancy ($q = +2$) at the site whose four
cation neighbours are all Zr, together with the linear-response calculation
of the dielectric tensor. You do NOT run VASP; you only produce the
structures and the input files.

## What you are given (established upstream, use as given)

- `pristine_x050.vasp` -- the defect-free $2 \times 2 \times 2$ supercell of
  Pca2$_1$ Hf$_{0.5}$Zr$_{0.5}$O$_2$, relaxed at fixed lattice. The lattice is
  held fixed in everything that follows.
- `sites.csv` -- the nine representative oxygen sites of that cell, one per
  local cation environment: the environment label, the cation coordination
  of the site, and its fractional coordinates in `pristine_x050.vasp`. The
  environment label counts the Hf and the Zr among the cations within 2.3 Å
  of the site.
- `eps_cell.vasp` -- a relaxed 12-atom cell of the same phase and lattice at
  $x = 0.5$, on which the dielectric tensor is to be computed.
- `potcar_catalog.yaml` -- the PAW potentials to use for Hf, Zr and O, with
  their cutoffs and valence electron counts.

## Conventions to adopt (fixed for this task)

- The defect is a single oxygen vacancy: the `4Zr` site of `sites.csv` is
  removed from `pristine_x050.vasp` and nothing else in the cell is changed.
  The cell you submit is the unrelaxed starting structure of the defect
  calculation.
- Charge state $q = +2$: the cell carries two electrons fewer than the
  neutral vacancy cell.
- All calculations are spin-unpolarised, within PBE.
- The vacancy relaxation is converged until every residual force is below
  $10^{-4}$ eV/Å, with the lattice held fixed.

## What to produce

Three runs.

(1) `relax` -- the fixed-lattice ionic relaxation of the charged vacancy cell.
    POSCAR: the starting cell described above. INCAR and KPOINTS: a
    relaxation of the doubly charged vacancy cell of a wide-gap insulator to
    the stated criterion.

(2) `static` -- the single-point calculation on the relaxed defect geometry
    that the finite-size correction reads. It must write the electrostatic
    potential of the cell to disk (the Hartree and ionic parts, without the
    exchange-correlation part), at the same charge state as the relaxation,
    with no ionic motion. Since the
    relaxed coordinates do not exist yet, submit this run with the same
    starting cell as its POSCAR.

(3) `dfpt` -- the total static dielectric tensor of `eps_cell.vasp`, including
    both electronic and ionic contributions. POSCAR: the staged cell,
    unchanged. The Brillouin-zone sampling
    should be dense enough for a converged dielectric tensor of a 12-atom
    cell.

Beyond the conventions above, every choice in the decks is yours.

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"relax": "answer/relax", "static": "answer/static", "dfpt": "answer/dfpt"}}
```

and three directories `answer/relax/`, `answer/static/`, `answer/dfpt/`, each
containing exactly `INCAR`, `KPOINTS` and `POSCAR`. Do not submit a POTCAR;
the potentials are fixed by the catalog and are not graded from your files.
