# C09: Curie temperature and compression response in CuInP2S6

This workflow was built from the authors' own project and is released as a
supplementary workflow without any agent evaluation.

## Graded fields

The submission is `answer.json`. Every field below must lie within
its tolerance of the reference value; other fields are ignored.

| Field | Reference | Absolute tolerance | Relative tolerance |
|---|---|---|---|
| `Tc_unstrained_K` | 363.076 | 58 | 0.0 |
| `slope_K_per_percent_compression` | 54.8564 | 21 | 0.0 |

## Files

- `agent/LG/` (full guidance): `inputs/potential.pb`, `inputs/start.vasp`, `prompt.md`
- `agent/LA/` (reduced guidance): `inputs/potential.pb`, `inputs/start.vasp`, `prompt.md`
- `grader/reference.json`: reference values and tolerances; `grader/reference_submission/` scores 1.

## Environment

`environment/` describes the software this workflow needs beyond the shared
workflow runtime: `software_additions.json` lists the added codes,
`uv_project/` pins the Python packages with a uv lock file, and
`system_runtime.md` gives the runtime instructions shown to the agent.
`deployment.json` and `profile.json` record the deployment that passed the
model-free acceptance run in `../runtime/acceptance.json`.

Grade with `python3 workflows/grader.py workflows/C09/grader/reference.json <answer_file>`.
