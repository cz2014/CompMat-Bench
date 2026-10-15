# CompMat-Bench

CompMat-Bench is a benchmark of 94 tasks from 15 recently published,
peer-reviewed studies in computational materials science. Each task asks an
agent to complete one step toward the study's scientific goal.

- **No expensive simulation during evaluation.** We reproduced every research
  step in advance. Agents are assessed on preparing the inputs and analyzing
  the outputs of expensive simulations, and run only the light calculations
  themselves.
- **Fixed-rule grading, no LLM judge.** The reproduced inputs and results are
  the ground truth. Submissions are graded by fixed rules with numerical
  tolerances.
- **Four evaluation conditions.** Single tasks and workflows of related tasks,
  each with full or reduced methodological guidance.
- **976 agent trajectories.** The transcripts, submissions and grades
  of every attempt reported in the paper, for comparing agents and analyzing
  their failures.

Paper: [arXiv:2610.00636](https://arxiv.org/abs/2610.00636).

No agent harness is included: any agent that can work in a directory can
be evaluated.

| | Count |
|---|---|
| Tasks | 94: 30 prepare simulation inputs, 64 analyze simulation outputs; 52 also have a reduced-guidance prompt |
| Workflows | 10: C01 to C08 evaluated in the paper; C09 and C10 supplementary, not evaluated |
| Codes | VASP, Quantum ESPRESSO, ABACUS, Wannier90, LAMMPS, VAMPIRE, LOBSTER |
| Agent attempts | 976 across four agents and three LLMs, with transcripts, submissions and grades |

## Download

This repository holds what you read: prompts, graders, reference answers,
manuals and source bundles. The release files hold what you run: the same
tree plus the input files staged for the agent in every task and workflow
(3.2 GB). They are attached to the
[v1.0.0 release](https://github.com/cz2014/CompMat-Bench/releases/tag/v1.0.0)
of this repository, reached from the Releases link on the repository front
page. `CHECKSUMS.sha256`, attached to the same release, lists the sha256 of
every file. The GitHub CLI fetches everything in one command:

```sh
gh release download v1.0.0 --repo cz2014/CompMat-Bench --dir compmat_bench_release
cd compmat_bench_release && sha256sum -c CHECKSUMS.sha256
```

| File | Content | Size |
|---|---|---|
| `compmat_bench_core.zip` | The complete benchmark except the staged inputs of five tasks | 0.5 GB |
| `compmat_bench_heavy_inputs_1.zip`, `compmat_bench_heavy_inputs_2.zip` | The staged inputs of those five tasks | 1.4 GB, 0.8 GB |
| `<campaign>_<model>.tar.gz` (13 files) | Agent trajectories, one tarball per campaign and model (see `trajectories/README.md`) | 0.34 GB |

Every zip unpacks into the same `compmat_bench/` directory: unpack the core
zip, then the heavy-input zips over it. Only these five tasks need a
heavy-input zip:

- `task_202603_028_step_hzo-vacancy-charge-thermodynamics` (`compmat_bench_heavy_inputs_1.zip`)
- `task_202509_007_step_hbn-undulated-step-decomposition` (`compmat_bench_heavy_inputs_2.zip`)
- `task_202601_008_step_v2coal-heisenberg-exchange` (`compmat_bench_heavy_inputs_2.zip`)
- `task_202509_005_step_bn-nanotube-flexoelectric-constant` (`compmat_bench_heavy_inputs_2.zip`)
- `task_202602_014_step_rucl2-magneto-raman-tensor` (`compmat_bench_heavy_inputs_2.zip`)

## Running a task

1. Give the agent `tasks/<task_id>/agent/` as its working directory and the
   prompt from `prompt.md` (full guidance) or `prompt_reduced.md` (reduced
   guidance). The agent must not see `grader/`.
2. The agent writes its submission into the working directory: the file
   named by `submission` in `task.yaml` and, for input-preparation tasks,
   the input files the prompt asks for.
3. Grade the working directory. The grader prints each check and then
   `SCORE: 1` or `SCORE: 0`.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 tasks/<task_id>/grader/grader.py <working_directory>
```

Grading needs Python 3.10 or newer with the packages listed under Software
environment. Two Wannier90 tasks (`task_202602_005_step_gete-wannier-inputs` and `task_202603_011_step_crn-wannier-exchange-inputs`) also need the Wannier90
runtime built by `env/env_wannier90_grader/build_wannier90_grader.py` and
named in `COMPMAT_WANNIER90_RUNTIME`. Every `grader/reference_submission/`
scores 1, which checks an installation. Tasks that prepare inputs for VASP,
Quantum ESPRESSO, ABACUS or Wannier90 do not need those codes: the inputs are
graded without running them.

In the paper the agent had no internet access, the manuals in `reflib/`,
4 CPU cores, 16 GiB of memory, one hour per task and the software listed
below.

## Running a workflow

A workflow package `workflows/Cxx/agent/LG/` (full guidance) or `LA/`
(reduced guidance) is a working directory in the same sense. The agent runs
every calculation itself, including the simulations, and writes the final
quantities the prompt names. Grade with:

```sh
python3 workflows/grader.py workflows/Cxx/grader/reference.json <answer_file>
```

In the paper a workflow had one 128-core node and five hours. Workflows C06,
C07 and C10 run VASP; their POTCAR files are not included, and each
`inputs/potcars/` directory lists them in `POTCARS_NOT_INCLUDED.md` so that
you can place your own copies at those paths.

| Workflow | Title | Evaluated in the paper |
|---|---|---|
| C01 | WS2/WSe2 moire: relaxation, piezoelectric charge and electron potential | yes |
| C02 | Se-pair substitutions: space groups, tight-binding models and spin splittings | yes |
| C03 | BAs: force constants, isotope-disordered dielectric function and near-field transmission | yes |
| C04 | Si/Ge interface: force constants, phonon reflection and specularity | yes |
| C05 | Zn(1-x)Mg(x)O: ordered cells, polarization paths and Landau free energy | yes |
| C06 | CrN: magnetic ground state, Wannier exchange constants and spin waves | yes |
| C07 | V2CoAl orderings: relaxation, magnetic states and exchange couplings | yes |
| C08 | Ni island: EAM saddle, machine-learned refinement and curvature certification | yes |
| C09 | Curie temperature and compression response in CuInP2S6 | no, supplementary |
| C10 | Molecular reaction energies and chain propagation | no, supplementary |

## Layout

| Path | Content |
|---|---|
| `tasks/<task_id>/` | `task.yaml` (role, cost class, codes, source month, submission file name), `agent/` (prompts and staged inputs), `grader/` (`grader.py`, `ground_truth.json` with reference values and tolerances, `reference_submission/`). |
| `graders/` | The shared grading modules; each task's `grader/graderlib/` copies the ones it imports. |
| `workflows/C01` to `C10` | `agent/LG/` and `agent/LA/` packages, `grader/reference.json`; `manifest.json` lists the workflows, `grader.py` grades them, `runtime/` records the software and runtime contract. |
| `env/` | Third-party source bundles staged read-only in some tasks, with their licences. |
| `reflib/` | The offline manuals given to the agent (abacus, alamode, lammps, qe, vampire, wannier90); `reflib/README.md` lists each code's origin, version and licence. The LOBSTER and VASP manuals were available to the agent but are not redistributed; `reflib/lobster/MANUALS_NOT_INCLUDED.md` lists the LOBSTER files and checksums. |
| `trajectories/` | `manifest.json`, one row per attempt, and the README describing the row schema and tarballs. |

Task identifiers read `task_YYYYMM_NNN_step_<slug>`: publication month of
the source study, sequence number, short name.

## Licence and citation

Everything we wrote, including tasks, prompts, graders, reference answers
and trajectories, is licensed under CC BY 4.0 (`LICENSE`). Third-party
material in `env/`, `reflib/` and staged task files keeps its own licence,
included in each directory and summarised in `reflib/README.md`.

Please cite the paper, arXiv:2610.00636; `CITATION.cff` gives the entry.

## Software environment

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
