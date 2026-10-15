# Task: pristine BaTiO3 reference cell and the LAMMPS decks for its field-driven hysteresis loop

Before dislocated BaTiO3 can be studied, the pristine, defect-free material
has to be characterized in the same model: a cubic BaTiO3 cell described by a
core-shell (shell-model) interatomic potential is driven into its metastable
tetragonal state by an external field, and its polarization hysteresis is
traced by stepping the field through one closed loop by molecular statics.
Your job is to build the starting cell and to write the two LAMMPS input
scripts, one that poles it and one that runs the loop. You do NOT run LAMMPS;
you only produce the structure and the input scripts.

## What you are given

- `bto_unit.lmp` -- the cubic BaTiO3 unit cell as a LAMMPS data file
  (`atom_style full`, metal units) in the core-shell representation this
  potential uses. Treat it as the reference that fixes the conventions: the
  lattice constant, the atom types and which of them are cores and shells,
  the per-type charges and masses, the molecule-id pairing of each core with
  its shell, and the bond types. Build from it by replication.
- `shell_model.yaml` -- the interatomic potential: charges, the short-range
  Buckingham terms, the core-shell spring in its own energy form, the Coulomb
  treatment, the cutoff. The parameters are given in the potential's own
  functional forms; expressing them in LAMMPS is part of the task.
- `field_steps.csv` -- the loop: the step label and the applied field $E_z$
  in MV/cm at every step, in the order the loop is run.

## The structure to build

Replicate the unit cell $10 \times 10 \times 10$ along $x$, $y$ and $z$.
Keep the coordinate origin of the seed. Nothing else is changed:
every ion keeps its core-shell pair, its molecule id pairs core with shell,
and every core-shell bond is kept.

## The decks to write

(1) **Poling run** (`pole`). With the potential of `shell_model.yaml` applied
    to the built cell, an external electric field of 10 MV/cm along $+z$ is
    applied (the field enters the minimized energy), and the atomic positions
    together with the three cell lengths are relaxed independently at zero
    target stress, with the cell angles held fixed, by energy minimization at 0 K, the cell-and-position relaxation being repeated in
    alternation with position-only relaxations until the cell has converged.
    Every minimization stops on the force (the energy test is disabled) with
    a tight tolerance. The relaxed poled cell is written as a LAMMPS data
    file `poled.lmp`.

(2) **Loop run** (`loop`). Starting from `poled.lmp`, the field along $z$ is
    set to each value of `field_steps.csv` in turn, in the order of the file.
    At each step the atomic positions and the cell are relaxed under that
    field exactly as in the poling run, starting from the relaxed state of
    the previous step (never from the starting cell), and the relaxed
    structure is written as a LAMMPS data file `state_<step>.lmp`, where
    `<step>` is the step label of the file. The field values must be carried
    into the script (inline, or through a file your script reads that you
    include in the run directory).

## What to submit

The decks as a run manifest plus two run directories:

```text
answer.json                 # {"runs": {"pole": "answer/pole", "loop": "answer/loop"}}
answer/pole/pole.in         # the poling script
answer/pole/input.lmp       # the built 10 x 10 x 10 cell (LAMMPS data file, atom_style full)
answer/loop/loop.in         # the loop script
```

`answer.json` in the workspace root must contain exactly:

```json
{"runs": {"pole": "answer/pole", "loop": "answer/loop"}}
```

`answer/pole/pole.in` reads `answer/pole/input.lmp` (refer to it as
`input.lmp`) and writes `poled.lmp`. `answer/loop/loop.in` reads `poled.lmp`
and writes one `state_<step>.lmp` per step. `answer/pole/input.lmp` must keep
the atom types, charges, masses, molecule ids and bond types of the seed cell
(that is how your particles are identified). Do not copy the parameter file
into the submission.
