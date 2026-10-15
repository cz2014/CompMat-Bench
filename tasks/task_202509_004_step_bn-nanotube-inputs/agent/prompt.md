# Task: construct a BN nanotube and prepare relaxation and static inputs

Build an armchair $(8,8)$ single-wall BN nanotube by rolling a honeycomb
sheet whose flat B–N bond length is $1.45\ \text{\AA}$.
`POSCAR_tube_seed` is a reference fixing the cell, tube placement, angular
orientation and species-labeling conventions. Keep its cell and use $y$ as
the tube axis. In the seed reference frame, the tube axis passes through
$x=z=15\ \text{\AA}$. The complete structure is assessed up to a rigid translation;
the seed fixes the relative geometry and orientation, not absolute positions.
`potcar_catalog.yaml` lists the available pseudopotentials.

Prepare two VASP input sets:

(1) `relax`: ionic relaxation at fixed cell, converged on forces.

(2) `static`: a subsequent single-point calculation of the relaxed tube that
writes the electrostatic potential without ionic motion.

Use electronic settings suitable for an isolated nanotube. This is an
input-authoring task: no calculation is required, and no VASP executable or
source is supplied in the task bundle. Submit the same constructed tube as
the `POSCAR` of both runs; when executed, the static run would instead use
the relaxed geometry.

Obtain the self-consistent PBE electronic ground state in both runs. No
precomputed electronic-state files are supplied; the static calculation may
reuse electronic-state output produced by the relaxation.

## Submission

Write `answer.json` in the workspace root:

```json
{"runs": {"relax": "answer/relax", "static": "answer/static"}}
```

Each run directory must contain `INCAR`, `KPOINTS` and `POSCAR`. Do not
submit POTCAR files; pseudopotential identities are fixed by
`potcar_catalog.yaml`.
