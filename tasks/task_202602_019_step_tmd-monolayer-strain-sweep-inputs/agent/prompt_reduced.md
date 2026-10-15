# Task: input files for a biaxial-strain sweep of two 1H TMD monolayers

Prepare VASP inputs for K-point band edges of isolated WS2 and WSe2,
using `monolayer_WS2.vasp`, `monolayer_WSe2.vasp` and the VASP 5.4 PAW PBE
pseudopotentials specified by `potcar_catalog.yaml`.

For each material, build rigid, uniformly biaxially strained cells at
$\varepsilon\in\{-1.5,-1.0,-0.5,0,+0.5,+1.0,+1.5\}\%$ relative to its seed.
The cell height and all atomic z coordinates remain as staged.

For each strained cell prepare:

- `relax`: fixed-cell ionic relaxation of internal coordinates;
- `static`: self-consistent SOC calculation on the relaxed geometry for
  K-point band edges.

Use converged settings and the same fixed strained cell for both runs.
For this input-only submission, the static POSCAR may use the same rigid
strained geometry as the relax POSCAR.

## Output format

Labels are `<layer>_eps_<tag>_<step>`, where layer is `WS2` or `WSe2`,
step is `relax` or `static`, and the tags in ascending strain order are
`m15`, `m10`, `m05`, `p00`, `p05`, `p10`, `p15`.
Submit `answer.json` with a `runs` mapping of all 28 labels to `answer/<label>`.
For example, one entry is:

```json
{"runs": {"WS2_eps_m15_relax": "answer/WS2_eps_m15_relax"}}
```

Each directory contains exactly `INCAR`, `KPOINTS` and `POSCAR`.
Do not submit POTCAR files; their identities are fixed by the catalog.
