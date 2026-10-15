# Task: build an edge-dislocation dipole in BaTiO3 and write its LAMMPS relaxation and poling deck

Ferroelectric switching at extended defects is studied here in an atomistic
model: a pair of $\langle 100 \rangle$ edge dislocations with opposite Burgers
vectors is placed in a periodic cell of BaTiO3 described by a core-shell
(shell-model) interatomic potential, relaxed by molecular statics, and poled
into a tetragonal state by an external field so that its strain and
polarization fields can be studied afterwards. Your job is to build the
starting structure of that cell and to write the LAMMPS input script that
relaxes and poles it along $x$. You do NOT run LAMMPS; you only produce the
structure and the input script.

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

## The structure to build

Replicate the unit cell $40 \times 80 \times 1$ along $x$, $y$ and $z$, so
that the cell is one lattice constant thick along $z$. Then remove one BaO-TiO2 double layer: a
BaO plane normal to $x$ together with the TiO2 plane immediately next to it
on its $+x$ side (the BaO plane is the one at the smaller $x$ of the two),
over 40 consecutive lattice cells along $y$ (half the cell's extent) and over
the full thickness along $z$. Choose the BaO plane nearest the cell centre
along $x$ and the TiO2 plane immediately on its $+x$ side. Along $y$, remove
the replicas in the middle half of the cell, with replica origins in
$[L_y/4, 3L_y/4)$, where $L_y$ is the cell length. Keep the coordinate origin
of the seed.
Nothing else is removed or moved: the remaining
ions stay on their ideal positions, every remaining ion keeps its core-shell
pair, its molecule id pairs core with shell, and every core-shell bond is kept.
This produces the dislocation dipole: the two dislocation lines run along $z$
at the two ends of the removed layer, with Burgers vectors $\pm[100]$.

## The deck to write

Write one LAMMPS input script that, with the potential of `shell_model.yaml`
applied to the built cell:

- poles the cell with an external electric field of 10 MV/cm along $+x$,
  applied for the whole relaxation (the field enters the minimized energy);
- relaxes it in stages, all by energy minimization at 0 K: first the atomic
  positions along $x$ only (positions frozen along $y$ and $z$), then all
  positions, then the positions together with the three cell lengths at zero
  target stress (cell angles held), the cell-and-position relaxation being
  repeated in alternation with position-only relaxations until the cell has
  converged. Before the first minimization the cell is compressed by 1 % along
  $x$ to offset the large negative pressure left by the removed layer.
  Every minimization stops on the force (the energy test is disabled) with a
  tight tolerance;
- writes the relaxed with-field cell as a LAMMPS data file
  `relaxed_wfield.lmp`;
- then removes the field and repeats the cell-and-position relaxation, and
  writes the relaxed field-free cell as `relaxed_nofield.lmp`.

## What to submit

The deck as a run manifest plus one run directory:

```text
answer.json                 # {"runs": {"relax": "answer/relax"}}
answer/relax/relax.in       # the LAMMPS input script
answer/relax/input.lmp      # the built structure (LAMMPS data file, atom_style full)
```

`answer.json` in the workspace root must contain exactly:

```json
{"runs": {"relax": "answer/relax"}}
```

`answer/relax/relax.in` reads `answer/relax/input.lmp` (refer to it as
`input.lmp`) and writes `relaxed_wfield.lmp` and `relaxed_nofield.lmp` as
stated above. `answer/relax/input.lmp` must keep the atom types, charges,
masses, molecule ids and bond types of the seed cell (that is how your
particles are identified). Do not copy the parameter file into the
submission.
