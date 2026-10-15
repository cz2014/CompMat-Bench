# Task: build an R-type WS2/WSe2 moire supercell and its LAMMPS relaxation deck

Build the rigid, zero-twist R-type WS2/WSe2 moire supercell and prepare a
LAMMPS deck for its atomic reconstruction. Submit inputs only; do not run LAMMPS.

## What you are given

- `monolayer_WS2.vasp`, `monolayer_WSe2.vasp`: monolayer seed structures.
- `tmd.sw`: Stillinger-Weber potential.
- `WS2.KC`, `WSe2.KC`, `SSe.KC`: Kolmogorov-Crespi potentials, surface-normal `/z` variant.

## The moire supercell to build

Place a $26\times26$ WS2 supercell below a $25\times25$ WSe2 supercell,
with parallel orientation and zero twist. Their common hexagonal in-plane
side length is the mean of the two natural supercell side lengths. Apply the
corresponding uniform in-plane rescaling to each layer, retaining its staged
internal geometry otherwise. Do not relax or perturb the rigid structure.

The registry anchor is W-over-W at the in-plane origin. The initial W-plane
separation is $6.50\,\text{Å}$. Provide at least $15\,\text{Å}$ of vacuum along z.

## The relaxation deck to write

Use ordinary SW within each complete monolayer. For interlayer KC, include
each W plane and the chalcogen plane facing the interface from its layer.
Couple all cross-layer pairs among these four planes; exclude the outer-facing
chalcogen planes from KC.

Relax only atomic positions at fixed cell, to a tight force criterion well
beyond default tolerances. Disable the energy stopping test.

## What to submit

At the workspace root, submit `POSCAR_moire` as an element-species VASP POSCAR
(W, S, Se; Direct or Cartesian coordinates) and this `answer.json`:

```json
{"runs": {"relax": "answer/relax"}}
```

Under `answer/relax/`, submit `relax.in` and `moire.dat`. The input script
reads `answer/relax/moire.dat` and references the staged potential files.
The data file must describe the same structure as `POSCAR_moire`, with real
element masses in its `Masses` section. Do not copy the supplied potential
contents into the submission.
