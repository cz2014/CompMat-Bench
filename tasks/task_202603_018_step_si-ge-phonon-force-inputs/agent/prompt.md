# Task: force-calculation input sets for the ab initio phonon database of bulk Si and Ge

A solver of phonon transport across Si/Ge interfaces needs, for each of the
two bulk materials, a phonon database: the harmonic dispersion and the
three-phonon relaxation times. Both are obtained from interatomic force
constants computed by density-functional theory with the finite-displacement
supercell method: a harmonic set of single-atom displacements and a
third-order set of pair displacements, every displaced supercell of one set
run with the **same** single-point electronic calculation. This task is to
produce, for each material, the shared electronic deck of each of the two
sets together with the undisplaced supercell it is applied to. You do NOT run
VASP; you only produce the input sets.

## System (established upstream, use as given)

- `POSCAR_prim_Si`, `POSCAR_prim_Ge` -- the primitive cell of each material.
  Use them exactly as given. The device model places both materials on one
  common lattice constant, so the Ge cell is deliberately held at the Si
  constant rather than at Ge's own equilibrium value; do not relax either cell
  and do not substitute another constant.
- `potcar_catalog.yaml` -- the exact VASP PAW pseudopotentials to use for Si
  and Ge. They are LDA potentials: the whole database is computed in the
  Perdew–Zunger parametrization of the local-density approximation.
- Si and Ge are non-magnetic semiconductors.

## Conventions to adopt (fixed for this task)

- **Supercell size:** a $4 \times 4 \times 4$ supercell of the primitive cell
  for the harmonic set and for the third-order set alike.

## What to produce

Four runs, labelled `Si_harmonic`, `Si_third_order`, `Ge_harmonic` and
`Ge_third_order`. Each consists of:

(1) **POSCAR** -- the **undisplaced** $4 \times 4 \times 4$ supercell of the
    material's primitive cell. It is graded as a structure; the atom order
    and the origin are yours.

(2) **INCAR** and **KPOINTS** -- the shared single-point electronic deck that
    would be run on each displaced supercell of that set to obtain forces
    accurate enough for the force constants: a no-ionic-motion calculation of
    a non-magnetic semiconductor in the local-density approximation, with a
    Brillouin-zone sampling appropriate to the specified supercell and to the
    number of supercells in the set. Beyond the conventions above, every
    choice in the deck is yours.

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"Si_harmonic": "answer/Si_harmonic", "Si_third_order": "answer/Si_third_order",
          "Ge_harmonic": "answer/Ge_harmonic", "Ge_third_order": "answer/Ge_third_order"}}
```

and, for each label, a directory `answer/<label>/` containing exactly three
files: `INCAR`, `KPOINTS` and `POSCAR` (the undisplaced $4 \times 4 \times 4$
supercell). Do not submit a POTCAR; the pseudopotential identities are fixed
by `potcar_catalog.yaml` and are not graded from your files.
