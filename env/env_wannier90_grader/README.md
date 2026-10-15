# Wannier90 grader runtime

The graders of `task_202602_005_step_gete-wannier-inputs` and
`task_202603_011_step_crn-wannier-exchange-inputs` parse the submitted
Wannier90 input with unmodified Wannier90 3.1.0. This directory holds the
script that builds that reader, `build_wannier90_grader.py`, and the grading
module it takes its pinned version and checksums from,
`grader_wannier90_native.py` (the same file as
`graders/grader_wannier90_native.py`).

The reader runs Wannier90's `param_read` through a small Fortran driver and
exports the projection arrays after `select_projections`. Python then checks
atomic centres, complete independent p/d angular spaces, spin coverage and the
task's restrictions. It does not parse orbital spellings and is not a full
Wannier90 calculation. Random projections are rejected. Grader calls are
offline, single-threaded and limited to 15 seconds.

## Prerequisites

- Linux with gfortran 10 or newer (or another Fortran compiler, given with
  `--fc`) and BLAS/LAPACK. On Debian or Ubuntu with root:
  `apt install gfortran liblapack-dev libblas-dev`. Without root, a
  conda-forge environment works:

  ```sh
  micromamba create -p /path/to/w90env -c conda-forge gfortran openblas liblapack libblas
  ```

  and then pass `--fc /path/to/w90env/bin/gfortran` and
  `--lapack='-L/path/to/w90env/lib -Wl,-rpath,/path/to/w90env/lib -llapack -lblas'`
  to the builder. The finished runtime bundles the libraries it needs, so the
  environment can be removed after the build.
- The official Wannier90 v3.1.0 tarball, downloaded separately:

  ```sh
  curl -sSL -o wannier90-v3.1.0.tar.gz \
    https://github.com/wannier-developers/wannier90/archive/refs/tags/v3.1.0.tar.gz
  ```

  The builder verifies its SHA-256
  `40651a9832eb93dec20a8360dd535262c261c34e13c41b6755fa6915c936b254`.

## Build

The builder finds its files next to itself, so it can run from any directory:

```sh
python3 env/env_wannier90_grader/build_wannier90_grader.py \
  --source /path/to/wannier90-v3.1.0.tar.gz \
  --output /path/to/new-runtime --lapack='-llapack -lblas'
```

The output directory must not exist yet. It receives the driver, the binary,
its non-glibc shared libraries and a hash manifest. The builder makes no
network requests; the build takes about half a minute.

## Use

Point the graders at the runtime:

```sh
export COMPMAT_WANNIER90_RUNTIME=/path/to/new-runtime
```

Without the variable the graders look in `/opt/compmat-grader/wannier90`, so a
read-only bind or copy of the runtime there in a container works as well. The
grader verifies every file of the hash manifest before use; a missing or
changed file is an infrastructure error, not a score of 0. Check the
installation from the `compmat_bench/` directory with the two reference
submissions; each prints `SCORE: 1`:

```sh
python3 tasks/task_202602_005_step_gete-wannier-inputs/grader/grader.py \
  tasks/task_202602_005_step_gete-wannier-inputs/grader/reference_submission
python3 tasks/task_202603_011_step_crn-wannier-exchange-inputs/grader/grader.py \
  tasks/task_202603_011_step_crn-wannier-exchange-inputs/grader/reference_submission
```

Set the variable only in the grading environment, never from a submission,
and do not expose the runtime to the agent being evaluated.

## Licence

Wannier90 is distributed under the GPL. Keep the upstream source and licence
with any redistributed reader, and the licences and sources of the bundled
compiler and OpenBLAS libraries.
