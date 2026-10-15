# Task: projection-ready input sets for a bonding analysis of the oxygen sublattice of Hf$_{0.5}$Zr$_{0.5}$O$_2$, with and without a vacancy

Why an oxygen vacancy in hafnia-zirconia prefers one local cation environment
over another is a question about bonding: the answer is read from an
energy-resolved decomposition of the band structure into bonding and
antibonding metal-oxygen contributions, obtained by projecting the converged
plane-wave wavefunctions onto a local atomic-orbital basis with LOBSTER. That
projection is a post-processing step with its own requirements on the
plane-wave calculation that feeds it. Your job is to write the VASP input set
of that feeding calculation and the LOBSTER control file, for the defect-free
cell and for a charged vacancy cell. You do NOT run VASP or LOBSTER; you only
produce the input files.

## What you are given (established upstream, use as given)

- `pristine_x050.vasp` -- the defect-free $2 \times 2 \times 2$ supercell of
  Pca2$_1$ Hf$_{0.5}$Zr$_{0.5}$O$_2$, relaxed at fixed lattice.
- `vacancy_4Zr_q+2.vasp` -- the same cell with one oxygen removed, the site
  whose four cation neighbours are all Zr, relaxed in the charge state
  $q = +2$ (two electrons fewer than the neutral vacancy cell).
- `potcar_catalog.yaml` -- the PAW potentials to use for Hf, Zr and O, with
  their cutoffs and valence electron counts.

- `references/lobster/` -- the LOBSTER 5.1.1 user guide and FAQ, as PDFs
  and searchable text, for offline reference. LOBSTER 5.1.1 is the target
  input-file version; you do not run the program.

Both structures are final: they are used exactly as staged, with no
relaxation and no change of lattice.

## Conventions to adopt (fixed for this task)

- Spin-unpolarised PBE; the material is a wide-gap insulator.
- The projection basis is LOBSTER's `pbeVaspFit2015` set, with the orbitals
  Hf 5d 5p 6s, Zr 4d 4p 4s 5s and O 2p 2s.
- The projected curves are broadened with a Gaussian of width 0.2 eV and
  cover at least 10 eV below and 5 eV above the Fermi level.
- Every metal-oxygen bond of the cell, and nothing else, enters the bonding
  average: the interactions to be computed are all Hf-O and Zr-O contacts
  within the first coordination shell of each cell, which is separated from
  the next shell by a clear gap in the metal-oxygen distance distribution.

## What to produce

Two runs, `pristine` and `vacancy`. Each consists of the plane-wave
single-point calculation on the staged structure, set up so that LOBSTER
can project its result, and the LOBSTER control file for the projection of
that result: INCAR, KPOINTS, POSCAR and `lobsterin`. The `vacancy` run is
at $q = +2$; the `pristine` run is neutral. Beyond the conventions above,
every choice is yours, but remember what the projection needs from the
plane-wave step.

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"pristine": "answer/pristine", "vacancy": "answer/vacancy"}}
```

and two directories `answer/pristine/` and `answer/vacancy/`, each
containing exactly `INCAR`, `KPOINTS`, `POSCAR` and `lobsterin`. Do not
submit a POTCAR; the potentials are fixed by the catalog and are not graded
from your files.
