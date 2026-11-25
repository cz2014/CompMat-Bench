# Agent trajectories

`manifest.json` lists one row per agent attempt whose score the paper reports:
1940 rows in the full-guidance, reduced-guidance, repeated-trial and workflow
campaigns. It also lists the 34 attempts of the diagnostic run `full94_2.5h`,
which gave Gemini 4 Carbon 2.5 hours on the 34 tasks it did not finish within
one hour; they do not count in the `full94` result. The records themselves
(step logs, submissions and grader reports) are published as one tarball per
campaign and agent on the
[v1.1.0 release page](https://github.com/cz2014/CompMat-Bench/releases/tag/v1.1.0)
beside the benchmark zips, and listed in `CHECKSUMS.sha256` there.

`summary` gives the row counts (`counts`), passes (`passes`) and reruns after
infrastructure failure per campaign and model, the total row count, and the
diagnostic run (`diagnostic_campaigns`).
`rows` holds one object per attempt with these fields:

| Field | Meaning |
|---|---|
| `campaign` | `full94`, `reduced52`, `repeat20x5`, `workflow` or `full94_2.5h`. |
| `model` | `deepseek_v4_1_flash`, `gemini_3_8_flash`, `gpt_5_6_sol`, `codex_gpt_5_6`, `gpt_6_2_astra`, `gpt_6_1_sol`, `gemini_4_argon`, `qwen3_8_max`, `glm_5_3` or `gemini_4_carbon`. |
| `provider_route` | The API route that served the model. |
| `task_or_chain` | The task id, or the workflow id C01 to C08. |
| `guidance` | `full` or `reduced` for single tasks; `LG` (full guidance) or `LA` (reduced guidance) for workflows. |
| `trial` | Run number within the same task and condition. |
| `attempt_id` | Name of the attempt. |
| `session_id` | The agent session id, as it appears in the logs (null when none was recorded). |
| `host` | Where the record was kept: `anvil` (the HPC cluster) or `workstation` (the authors' workstation). |
| `score` | The score the paper reports, 1 (pass) or 0 (fail). |
| `rerun_after_infrastructure_failure` | True when this attempt replaced one that failed for a reason outside the agent. |
| `record_path` | `<campaign>/<model>/<attempt_id>`, the record's directory inside the trajectory tarballs. |
