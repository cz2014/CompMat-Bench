# Task: input files for a bilayer sliding sweep

Prepare input sets to determine how rigid interlayer sliding changes the
vacuum electrostatic-potential step of an hBN bilayer.

`POSCAR_AA` is the reference structure. `potcar_catalog.yaml` lists the
available pseudopotentials. Use the reference cell and interlayer separation
without modification. For the displacement convention, $\mathbf b$ is the
armchair direction and the top layer is the layer at larger $z$.

Generate twelve configurations by translating only the top layer by
$+\nu\mathbf b$, where $\nu=k/12$ for $k=0,\ldots,11$. Keep the bottom layer
fixed. For each configuration, prepare a self-consistent static calculation with no ionic
motion that writes the electrostatic or total local potential and resolves
flat vacuum plateaus above and below the polar bilayer. Use consistent
electronic settings throughout the sweep.

This is a VASP input-authoring task. Prepare files only; no calculation is
required. No VASP executable or source is supplied in this task bundle.

Use the semilocal PBE electronic model.

## Submission

Write `answer.json` in the workspace root:

```json
{"runs": {"nu_0.000": "answer/nu_0.000", "nu_0.083": "answer/nu_0.083",
          "nu_0.167": "answer/nu_0.167", "nu_0.250": "answer/nu_0.250",
          "nu_0.333": "answer/nu_0.333", "nu_0.417": "answer/nu_0.417",
          "nu_0.500": "answer/nu_0.500", "nu_0.583": "answer/nu_0.583",
          "nu_0.667": "answer/nu_0.667", "nu_0.750": "answer/nu_0.750",
          "nu_0.833": "answer/nu_0.833", "nu_0.917": "answer/nu_0.917"}}
```

Each `answer/<label>/` must contain `INCAR`, `KPOINTS` and the corresponding
`POSCAR`. Labels round the displacement to three decimals; construct the
structures at the exact fractions above. Do not submit POTCAR files;
pseudopotential identities are fixed by `potcar_catalog.yaml`.
