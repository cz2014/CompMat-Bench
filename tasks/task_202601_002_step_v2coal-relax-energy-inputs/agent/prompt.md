# VASP inputs for structural relaxation and total energies of V2CoAl

You are given `POSCAR_L21` and `POSCAR_XA`, the starting structures for two
orderings of the metallic alloy V2CoAl. `potcar_catalog.yaml` identifies the
VASP 5.4 PAW PBE potentials to use for V, Co, and Al.

For each supplied structure, prepare inputs for a spin-polarized structural
relaxation with a magnetic starting guess, followed by a static total-energy
calculation on the relaxed geometry.

## Required runs

(1) A full relaxation of the ions and cell shape/volume, with a force-based
stopping criterion.

(2) A spin-polarized static calculation with no ionic motion, yielding a
well-converged total energy.

Treat both orderings as metals. Use the specified pseudopotentials and a common,
sufficiently converged numerical accuracy for all four runs.

Use the labels `L21_relax` and `L21_static` for `POSCAR_L21`, and `XA_relax`
and `XA_static` for `POSCAR_XA`.

For each relaxation, submit the corresponding supplied POSCAR. For the static
runs, you may use that same starting POSCAR as a placeholder for the relaxed
geometry. A relaxed geometry is not required at this input-preparation checkpoint.
You do not need to run the calculations.

## Output

Write `answer.json` in the workspace root:

```json
{"runs": {"L21_relax": "answer/L21_relax", "L21_static": "answer/L21_static",
          "XA_relax": "answer/XA_relax", "XA_static": "answer/XA_static"}}
```

Each `answer/<label>/` directory must contain exactly `INCAR`, `KPOINTS`, and
`POSCAR`. Do not submit a POTCAR; its identity is fixed by the supplied catalog.
