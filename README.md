# CompMat-Bench

[![arXiv](https://img.shields.io/badge/arXiv-2610.00636-b31b1b.svg)](https://arxiv.org/abs/2610.00636)
[![Licence: CC BY 4.0](https://img.shields.io/badge/Licence-CC_BY_4.0-lightgrey.svg)](LICENSE)
[![Release](https://img.shields.io/github/v/release/cz2014/CompMat-Bench)](https://github.com/cz2014/CompMat-Bench/releases/tag/v1.0.0)

A benchmark of **94 research tasks from 15 peer-reviewed studies** in
computational materials science. Agents prepare simulation inputs, analyze
simulation outputs, and perform calculations toward a study's scientific goal.
Ground truth comes from our own reproductions; submissions are graded by fixed
rules and numerical tolerances, with no LLM judge.

📄 [Paper](https://arxiv.org/abs/2610.00636) ·
⬇️ [Download](https://github.com/cz2014/CompMat-Bench/releases/tag/v1.0.0) ·
🗂️ [Browse tasks](tasks/) · 🤖 [Agent trajectories](trajectories/README.md)

## 📦 What's included

| Resource | Scope |
|---|---|
| **[94 single-step tasks](tasks/)** | 30 input-preparation, 29 output-analysis and 35 run-and-analyze tasks. Expensive simulations are precomputed; agents prepare their inputs, analyze their outputs, or run light calculations themselves. 52 tasks also have reduced-guidance prompts. |
| **[10 workflows](workflows/README.md)** | Agents carry out every step themselves, including the simulations in most workflows, from the first calculation to the final quantities. C01–C08 were evaluated in the paper; C09–C10 are supplementary. |
| **[976 agent trajectories](trajectories/README.md)** | Transcripts, submissions and grades for every attempt reported in the paper, across four agents and three LLMs. |

Tasks cover VASP, Quantum ESPRESSO, ABACUS, Wannier90, LAMMPS, VAMPIRE and LOBSTER.

## 🚀 Quick start: run one task

Use your own agent; this release provides tasks and graders, without an agent
harness. The example below asks an agent to extract the **valley band gaps and
magnetic moment of monolayer RuCl₂** from supplied VASP outputs. No VASP
installation is needed. [Read the example prompt](tasks/task_202602_010_step_rucl2-valley-gaps/agent/prompt.md).

**1. Download and unpack the core package** (about 0.5 GB). It contains all the
inputs for this example. A Git clone contains the prompts and graders, but not
the staged simulation files.

```sh
gh release download v1.0.0 --repo cz2014/CompMat-Bench \
  --pattern compmat_bench_core.zip --dir compmat_bench_release
cd compmat_bench_release
unzip compmat_bench_core.zip
cd compmat_bench
```

You can also download the zip from the [release page](https://github.com/cz2014/CompMat-Bench/releases/tag/v1.0.0).
See the [download guide](docs/download.md) for checksums, the five tasks needing
additional input packages, and trajectory archives.

**2. Check the grader** with the supplied reference submission. This example
needs only Python 3.10 or newer, with no additional Python packages.

```sh
task=tasks/task_202602_010_step_rucl2-valley-gaps
PYTHONDONTWRITEBYTECODE=1 python3 "$task/grader/grader.py" \
  "$task/grader/reference_submission"
```

The last line should be `SCORE: 1`. This checks the grader installation; it does
not run an agent.

**3. Run your agent, then grade its submission.** Make a fresh working copy:

```sh
workspace=$(mktemp -d)
cp -R "$task/agent/." "$workspace/"
```

Give your agent that workspace and its `prompt.md`. It should read the supplied
files and write `answer.json` in the workspace root. Restrict the agent's file
access to its workspace and permitted tools and manuals; keep graders, reference
answers and trajectories outside its accessible environment. Setting the
working directory alone does not enforce this isolation.

After the agent finishes, run the grader outside the agent's environment, from
the same shell used above:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 "$task/grader/grader.py" "$workspace"
```

The grader reports each check and a final `SCORE: 1` (pass) or `SCORE: 0` (fail).
For other tasks, follow their prompts and the submission filename in `task.yaml`.

## ⚙️ Evaluation setup

The paper evaluates single steps and complete workflows, each with full or
reduced methodological guidance. Agents have no internet access and use an
[offline reference library](reflib/README.md).

| Setting | Single-step tasks | Workflows |
|---|---|---|
| Full / reduced prompt | `prompt.md` / `prompt_reduced.md` (52 tasks) | `agent/LG/` / `agent/LA/` |
| Compute in the paper | 4 CPU cores, 16 GiB RAM | One 128-core node |
| Time limit | 1 hour per task | 5 hours per workflow |

See the **[evaluation guide](docs/evaluation.md)** for task packaging, software
versions and grader dependencies, or **[workflow guide](workflows/README.md)**
for the workflow catalogue and execution instructions. Some licensed manuals
and VASP POTCAR files must be supplied separately, as described in those guides.

## 📖 Citation and licence

Please cite [arXiv:2610.00636](https://arxiv.org/abs/2610.00636);
[CITATION.cff](CITATION.cff) provides the citation metadata.

Our tasks, prompts, graders, reference answers and trajectories are licensed
under **[CC BY 4.0](LICENSE)**. Third-party material in `env/`, `reflib/` and
staged task files retains its own licence; see the notices in each directory
and the [reference-library licence list](reflib/README.md).
