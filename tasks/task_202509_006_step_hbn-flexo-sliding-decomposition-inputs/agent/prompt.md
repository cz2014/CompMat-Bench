# Task: prepare inputs for a flexoelectric/sliding decomposition

Prepare a five-run VASP campaign that separates the curvature contribution
of each hBN monolayer from the stacking contribution of their bilayer.

The supplied structures are:

- `mono_I.vasp` and `mono_II.vasp`: unrelaxed monolayers I and II.
- `mono_I_relaxed.vasp` and `mono_II_relaxed.vasp`: their relaxed counterparts.

These files fix the common cell and relative placement.
`potcar_catalog.yaml` lists the available pseudopotentials.

Prepare the following runs:

(1) `mono_I_relax` and `mono_II_relax`: force-converged ionic relaxations of
    the respective unrelaxed monolayers, with the cell fixed.

(2) `mono_I_static` and `mono_II_static`: static calculations on the supplied
    relaxed monolayers, writing the electrostatic potential without ionic motion.

(3) `bilayer_static`: a static electrostatic-potential calculation on the
    rigid union of the two relaxed monolayers at their supplied relative
    positions. Keep the common cell and do not relax this bilayer.

Use consistent electronic settings suitable for these sheets in vacuum.
This is an input-authoring task: no calculation is required, and no VASP
executable or source is supplied in the task bundle.

Obtain the self-consistent electronic ground state using a consistent semilocal
PBE or PBEsol model throughout the campaign. No precomputed electronic-state
files are supplied. Each monolayer static calculation may reuse electronic-state
output produced by its corresponding relaxation.

## Submission

Write `answer.json` in the workspace root:

```json
{"runs": {"mono_I_relax": "answer/mono_I_relax", "mono_II_relax": "answer/mono_II_relax",
          "mono_I_static": "answer/mono_I_static", "mono_II_static": "answer/mono_II_static",
          "bilayer_static": "answer/bilayer_static"}}
```

Each run directory must contain `INCAR`, `KPOINTS` and `POSCAR`. Do not
submit POTCAR files; pseudopotential identities are fixed by
`potcar_catalog.yaml`.
