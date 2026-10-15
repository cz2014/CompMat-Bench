# ABACUS LCAO inputs for XA V2CoAl with a ferrimagnetic seed

Prepare inputs for a spin-polarized, collinear LCAO single-point SCF calculation
of metallic XA V2CoAl, with site-resolved Mulliken magnetic-moment output.

You are given:

- `POSCAR_relaxed`: the relaxed geometry. Use it as given, without relaxation
  or rescaling.
- `abacus_catalog.yaml`: the pseudopotential and numerical-orbital catalog.
  Use the exact `.upf` and `.orb` identities listed for each species.

Use plain PBE without Hubbard U. Use a ferrimagnetic initial state with the V
4b and 4d sublattices oppositely aligned and a nonzero initial Co moment.
Choose numerical settings that give a converged single-point calculation.

## Output

Write `answer.json` in the workspace root:

```json
{"runs": {"V2CoAl_XA": "answer/V2CoAl_XA"}}
```

Place exactly three ABACUS input files in `answer/V2CoAl_XA/`: `INPUT`, `STRU`,
and `KPT`. Name the catalog's pseudopotential and orbital files in `STRU`; do not
submit their contents. You do not need to run the calculation.
