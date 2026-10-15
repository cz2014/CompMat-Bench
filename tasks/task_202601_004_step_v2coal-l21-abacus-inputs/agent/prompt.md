# ABACUS LCAO inputs for the magnetic moments of L2_1 V2CoAl

Prepare inputs for a spin-polarized LCAO single-point SCF calculation of
metallic L2_1 V2CoAl, with site-resolved Mulliken magnetic-moment output.

You are given:

- `POSCAR_relaxed`: the relaxed geometry. Use it as given, without relaxation
  or rescaling.
- `abacus_catalog.yaml`: the pseudopotential and numerical-orbital catalog.
  Use the exact `.upf` and `.orb` identities listed for each species.

Use plain PBE without Hubbard U and a ferromagnetic initial state for V and Co.
Choose numerical settings that give a converged single-point calculation.

## Output

Write `answer.json` in the workspace root:

```json
{"runs": {"scf": "answer/scf"}}
```

Place exactly three ABACUS input files in `answer/scf/`: `INPUT`, `STRU`, and
`KPT`. Name the catalog's pseudopotential and orbital files in `STRU`; do not
submit their contents. You do not need to run the calculation.
