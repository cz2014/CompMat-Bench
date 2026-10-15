# Task: input files for a biaxial-strain sweep of two 1H TMD monolayers

Prepare VASP input sets for the strain dependence of K-point band edges in
isolated WS2 and WSe2 monolayers.

## What you are given

- `monolayer_WS2.vasp`, `monolayer_WSe2.vasp`: pristine monolayer seed structures.
- `potcar_catalog.yaml`: the available VASP 5.4 PAW PBE pseudopotential identities.

## Part 1 -- build the 14 strained cells

For each layer, build rigid cells at biaxial in-plane strains

$$
\varepsilon\in\{-1.5,-1.0,-0.5,0,+0.5,+1.0,+1.5\}\%.
$$

Scale the in-plane cell vectors accordingly, retaining the seed fractional
in-plane positions. The cell height and all atomic z coordinates stay as
staged. Internal relaxation belongs to the subsequent run, not this build.

## Part 2 -- two runs per structure

For every strained cell, prepare:

- `relax`: ionic relaxation of internal coordinates at fixed strained cell;
- `static`: self-consistent calculation with spin-orbit coupling on the
  relaxed geometry, for K-point band edges.

Choose converged settings appropriate to each run. Both runs use the same
fixed strained cell. As relaxation outputs are not supplied in this
input-only task, the static POSCAR may use the same rigid strained geometry
as the relax POSCAR.

## Output format

Use labels `<layer>_eps_<tag>_<step>`, with `layer` in `WS2`, `WSe2`, and
`step` in `relax`, `static`. Tags for the seven strains in ascending order
are `m15`, `m10`, `m05`, `p00`, `p05`, `p10`, `p15`.

Submit `answer.json` with a `runs` mapping containing all 28 labels and their
`answer/<label>` directories. For example, the entry for the first WS2
relaxation is:

```json
{"runs": {"WS2_eps_m15_relax": "answer/WS2_eps_m15_relax"}}
```

This example shows one entry; your mapping must contain all 28. Each directory
must contain exactly `INCAR`, `KPOINTS` and `POSCAR` for its label. Do not submit
a POTCAR; pseudopotential identities are fixed by the supplied catalog.
