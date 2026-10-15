# Task: build a commensurate WS2/WSe2 heterobilayer and its three-step spin-orbit band deck

Build the commensurate bilayer and prepare three VASP input sets for its
layer-projected band structure.

## What you are given

- `monolayer_WS2.vasp`, `monolayer_WSe2.vasp`: monolayer seed structures.
- `potcar_catalog.yaml`: the available VASP 5.4 PAW PBE pseudopotential identities.

## Part 1 -- the commensurate R^X_h bilayer

Build a $1\times1$, zero-twist bilayer with WS2 below WSe2. Its common
hexagonal in-plane lattice constant is the mean of the two seed constants;
both layers are uniformly rescaled in plane to that cell.

The registry is $R^X_h$: the WSe2 chalcogen column lies over the WS2
hexagon center. Initially separate the W planes by $6.30\,\text{Å}$ and
retain each layer's staged chalcogen heights relative to its W plane.
Provide at least $15\,\text{Å}$ of vacuum along z; vacuum height is otherwise free.

## Part 2 -- the three-step VASP deck

Use PBE and converged settings appropriate to each run:

- `relax`: fixed-cell, force-converged ionic relaxation without spin-orbit coupling.
- `static`: self-consistent calculation on the relaxed geometry, with spin-orbit
  coupling and per-site orbital projections.
- `band`: non-self-consistent calculation using the static charge density,
  with spin-orbit coupling and per-site orbital projections, along a path
  containing $\Gamma=(0,0,0)$, $M=(1/2,0,0)$ and $K=(1/3,1/3,0)$
  in reciprocal fractional coordinates.

Use the built bilayer as the `relax` POSCAR. Because this input-only task
provides no relaxation output, the `static` and `band` POSCARs may use the
same built geometry.

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"relax": "answer/relax", "static": "answer/static", "band": "answer/band"}}
```

and, for each label, a directory `answer/<label>/` containing exactly three
files: `INCAR`, `KPOINTS`, and `POSCAR`. Do not submit a POTCAR anywhere; the
pseudopotential identities are fixed by `potcar_catalog.yaml` and are not graded
from your files.
