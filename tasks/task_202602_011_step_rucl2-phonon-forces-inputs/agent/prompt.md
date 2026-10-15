# Task: force-calculation input set for the phonon spectrum of a monolayer 2H-RuCl2 sheet

You are preparing the input files for the harmonic phonon calculation of a
monolayer of 2H-RuCl2, carried out by the finite-displacement (frozen-phonon)
supercell method. In that method the
phonon force constants are assembled from the forces on a set of supercells in
which single atoms are given small displacements; every one of those displaced
supercells is run with the **same** single-point (no ionic motion) electronic
deck, and the forces are differenced. This task is to produce that shared
electronic deck together with the undisplaced supercell it is applied to.

## System (established upstream, use as given)

- Monolayer 2H-RuCl2 is a **ferromagnetic insulator**.
- `POSCAR_unitcell` is the relaxed primitive cell. Use this cell exactly and
  hold it fixed; do not re-relax it.
- `potcar_catalog.yaml` lists the exact VASP 5.4 PAW PBE pseudopotentials to use
  for Ru and Cl, with their plane-wave cutoffs.

## Conventions to adopt (fixed for this task)

These are pinned so the calculation is well defined; adopt them as stated.

- **Supercell size:** build a 4x4x1 supercell of the primitive cell (16 Ru + 32
  Cl, 48 atoms). The single out-of-plane direction is not repeated.
- **Magnetism:** treat the magnetism as collinear ferromagnetic; spin-orbit
  coupling is **not** included (this is a scalar, collinear calculation, not a
  noncollinear one).
- **On-site correction:** apply a Dudarev DFT+U correction of U = 2 eV to the
  Ru 4d shell (no correction on Cl).

## What to produce

A single run, labelled `supercell_forces`, consisting of:

(1) **POSCAR** -- the **undisplaced** 4x4x1 supercell of `POSCAR_unitcell`
    (the perfect, symmetric supercell before any displacement is applied). The
    displaced configurations are generated from this cell by the phonon code and
    are not part of this submission; we grade the base supercell and the deck.

(2) **INCAR** and **KPOINTS** -- the shared single-point electronic deck that
    would be run on each displaced supercell to obtain accurate forces for the
    force constants: a spin-polarized, DFT+U, no-ionic-motion calculation whose
    forces are converged well enough to give a clean, dynamically stable phonon
    spectrum for this ferromagnetic insulator, with a Brillouin-zone sampling
    appropriate to a 48-atom supercell with vacuum along the out-of-plane axis.
    Beyond the pinned conventions above, every choice in the deck is yours.

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"supercell_forces": "answer/supercell_forces"}}
```

and a directory `answer/supercell_forces/` containing exactly three files:
`INCAR`, `KPOINTS`, and `POSCAR` (the undisplaced 4x4x1 supercell). Do not submit
a POTCAR; the pseudopotential identities are fixed by `potcar_catalog.yaml` and
are not graded from your files.
