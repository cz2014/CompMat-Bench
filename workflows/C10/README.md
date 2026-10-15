# C10: Molecular reaction energies and chain propagation

This workflow was built from the authors' own project and is released as a
supplementary workflow without any agent evaluation.

## Graded fields

The submission is `answer.json`. Every field below must lie within
its tolerance of the reference value; other fields are ignored.

| Field | Reference | Absolute tolerance | Relative tolerance |
|---|---|---|---|
| `v_K1_0_sites_per_ps` | 28.2002 | 0.1 | 0.0 |
| `v_K1_5_sites_per_ps` | 59.2804 | 0.1 | 0.0 |
| `v_K2_0_sites_per_ps` | 79.3273 | 0.1 | 0.0 |

## Files

- `agent/LG/` (full guidance): `inputs/HCN.xyz`, `inputs/HNC.xyz`, `prompt.md`
  plus `inputs/potcars/POTCARS_NOT_INCLUDED.md` (VASP POTCAR files, not included)
- `agent/LA/` (reduced guidance): `inputs/HCN.xyz`, `inputs/HNC.xyz`, `prompt.md`
  plus `inputs/potcars/POTCARS_NOT_INCLUDED.md` (VASP POTCAR files, not included)
- `grader/reference.json`: reference values and tolerances; `grader/reference_submission/` scores 1.

## Environment

`environment/` describes the software this workflow needs beyond the shared
workflow runtime: `software_additions.json` lists the added codes,
`uv_project/` pins the Python packages with a uv lock file, and
`system_runtime.md` gives the runtime instructions shown to the agent.
`deployment.json` and `profile.json` record the deployment that passed the
model-free acceptance run in `../runtime/acceptance.json`.

Grade with `python3 workflows/grader.py workflows/C10/grader/reference.json <answer_file>`.
