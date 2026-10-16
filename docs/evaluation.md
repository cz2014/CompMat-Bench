# Evaluation guide

[Back to the benchmark](../README.md) · [Download](download.md) ·
[Workflows](../workflows/README.md)

## Run a single-step task

Start from the extracted release, not a Git clone: the release includes the
staged inputs. For a complete example, see the [quick start](../README.md#quick-start-run-one-task).

(1) Copy `tasks/<task_id>/agent/` into a fresh workspace. Give the agent the
prompt from `prompt.md` for full guidance, or `prompt_reduced.md` for reduced
guidance where available. Expose only the selected prompt; in a reduced-guidance
run, remove the full-guidance prompt from the working copy before starting.

(2) Run your agent in an isolated environment with the permitted software and
offline manuals. Do not expose `grader/`, reference answers, other task packages
or recorded trajectories. A working-directory setting does not restrict file
access; your harness or container must enforce that boundary. No agent harness
is included in this release.

(3) The agent writes the submission file named by `submission` in `task.yaml`
and any additional files the prompt requires. Run the grader outside the agent's
environment, from the benchmark root (replace the task ID and workspace path):

```sh
task=tasks/YOUR_TASK_ID
workspace=/path/to/agent-workspace
PYTHONDONTWRITEBYTECODE=1 python3 "$task/grader/grader.py" "$workspace"
```

The grader prints its checks followed by `SCORE: 1` or `SCORE: 0`. Numerical
answers are compared with reproduced reference values using task-specific
tolerances; input files are checked against fixed rules. No LLM judge is used.

The reported single-step evaluation used 4 CPU cores, 16 GiB RAM and one hour
per task, with no internet access. Agents prepared inputs or analyzed staged
outputs for expensive simulations, and ran light calculations themselves.
Complete workflows have different compute requirements; see the
[workflow guide](../workflows/README.md).

## Grader environment

Use Python 3.10 or newer. Dependencies vary by task; the base Python packages
listed below cover the common scientific stack. The quick-start example uses
only the Python standard library.

Every task includes a `grader/reference_submission/` that should score 1. Check
a task's grader installation with the same `task` variable:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 "$task/grader/grader.py" \
  "$task/grader/reference_submission"
```

Input-preparation graders do not run the expensive simulations. Two tasks do
require a **Wannier90 input-reader runtime** in the grading environment:
`task_202602_005_step_gete-wannier-inputs` and
`task_202603_011_step_crn-wannier-exchange-inputs`. It parses submitted input using
Wannier90 3.1.0; it does not perform a full Wannier90 calculation. Follow the
[build instructions](../env/env_wannier90_grader/README.md), set
`COMPMAT_WANNIER90_RUNTIME`, and keep this runtime outside the agent environment.

## Agent software environment

Simulation codes available to the agent in the reported evaluation, with
their versions. Single tasks and Workflows give availability in each kind
of run: yes, no, or source (a read-only source tree).

| Code | Version | Single tasks | Workflows |
|---|---|---|---|
| VASP | 6.1.0; 6.4.1 | no | yes |
| Quantum ESPRESSO | 7.5 | no | yes |
| ABACUS | 3.10.1 | no | yes |
| Wannier90 | 3.1.0 | no | yes |
| LAMMPS | 22 Jul 2025, update 4 | yes | yes |
| phono3py, with phonopy | 4.4.0 | yes | yes |
| MACE, with PyTorch | 0.3.16; 2.13.0 (CPU) | yes | yes |
| ALAMODE | 1.4.2 | yes | yes |
| TB2J | 0.9.12.26; 0.9.20 | yes (0.9.12.26); source (0.9.20) | yes |
| kdotpy | 1.4.1 | source | no |
| VAMPIRE | 7.0.0 | source | no |

Base Python packages shared by every run: numpy, scipy, ase, pymatgen, spglib, matplotlib, pandas, h5py, sympy, pyyaml.


This table records the software available in the reported evaluation; it is not
an installation script. The workflow [runtime records](../workflows/runtime/)
provide further version and runtime details.

## Offline manuals and licensed files

The [reference library](../reflib/README.md) contains the redistributable manuals
provided to agents, with origins, versions and licences. VASP wiki pages and the
LOBSTER User's Guide and FAQ were available during evaluation but cannot be
redistributed. The [LOBSTER file list](../reflib/lobster/MANUALS_NOT_INCLUDED.md)
records filenames and checksums for licensed users to restore them.

VASP workflows also require separately supplied POTCAR files. The
[workflow guide](../workflows/README.md#licensed-inputs) identifies the affected
workflows and their file lists.

## Repository layout

| Path | Content |
|---|---|
| [`tasks/`](../tasks/) | One directory per task: metadata, agent files, grader and reference submission. |
| [`graders/`](../graders/) | Shared grading modules; each task's `grader/graderlib/` contains the modules it imports. |
| [`workflows/`](../workflows/README.md) | Full- and reduced-guidance packages, grading references, workflow manifest and runtime records. |
| [`env/`](../env/) | Third-party source bundles and the Wannier90 grader-runtime builder. |
| [`reflib/`](../reflib/README.md) | Offline software manuals and licence notices. |
| [`trajectories/`](../trajectories/README.md) | Agent-attempt manifest and archive documentation. |

A task identifier has the form `task_YYYYMM_NNN_step_<slug>`: the source study's
publication month, a sequence number and a short name. Each `task.yaml` records
the role, cost class, codes, source month and submission filename. The `agent/`
directory holds prompts and staged inputs; `grader/` holds `grader.py`, reference
values and tolerances in `ground_truth.json`, and `reference_submission/`.
