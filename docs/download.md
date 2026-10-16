# Download the benchmark

[Back to the benchmark](../README.md) · [Evaluation guide](evaluation.md)

The Git repository contains prompts, graders, reference answers, manuals and
source bundles. To run tasks or workflows, download the release packages,
which also contain the staged inputs. Agent trajectories are separate downloads.
All files are available on the [v1.0.0 release page](https://github.com/cz2014/CompMat-Bench/releases/tag/v1.0.0).

## Choose a package

| File | Content | Approximate size |
|---|---|---|
| `compmat_bench_core.zip` | Benchmark tree, including staged inputs except those of the five tasks listed below | 0.5 GB |
| `compmat_bench_heavy_inputs_1.zip` | Additional inputs for one task | 1.4 GB |
| `compmat_bench_heavy_inputs_2.zip` | Additional inputs for four tasks | 0.8 GB |
| `<campaign>_<agent>.tar.gz` (13 files) | Agent trajectories, one archive per campaign and agent | 0.34 GB total |

Start with the core package for the [quick-start example](../README.md#-quick-start-run-one-task).
Download both heavy-input packages to use all 94 tasks. Download trajectory
archives only if you want to inspect existing agent runs; their contents and
manifest schema are described in the [trajectory guide](../trajectories/README.md).

## Download and verify all task inputs

With the GitHub CLI:

```sh
gh release download v1.0.0 --repo cz2014/CompMat-Bench \
  --pattern 'compmat_bench_*.zip' --pattern CHECKSUMS.sha256 \
  --dir compmat_bench_release
cd compmat_bench_release
```

`CHECKSUMS.sha256` lists every release asset by name, so after downloading the
whole release `sha256sum -c CHECKSUMS.sha256` checks everything. Select the
three benchmark zip entries to verify just the files downloaded above:

```sh
awk '$2 ~ /^compmat_bench_.*[.]zip$/ {print}' CHECKSUMS.sha256 > benchmark_checksums.sha256
sha256sum -c benchmark_checksums.sha256
```

On macOS, use `shasum -a 256 -c benchmark_checksums.sha256` instead. To check
only the core package, select `$2 == "compmat_bench_core.zip"` in the `awk`
command.

All three zips unpack into the same `compmat_bench/` directory. Unpack the core
first, then add the heavy inputs from the same parent directory:

```sh
unzip compmat_bench_core.zip
unzip compmat_bench_heavy_inputs_1.zip
unzip compmat_bench_heavy_inputs_2.zip
cd compmat_bench
```

## Tasks needing additional inputs

| Task | Additional package |
|---|---|
| `task_202603_028_step_hzo-vacancy-charge-thermodynamics` | `compmat_bench_heavy_inputs_1.zip` |
| `task_202509_007_step_hbn-undulated-step-decomposition` | `compmat_bench_heavy_inputs_2.zip` |
| `task_202601_008_step_v2coal-heisenberg-exchange` | `compmat_bench_heavy_inputs_2.zip` |
| `task_202509_005_step_bn-nanotube-flexoelectric-constant` | `compmat_bench_heavy_inputs_2.zip` |
| `task_202602_014_step_rucl2-magneto-raman-tensor` | `compmat_bench_heavy_inputs_2.zip` |

## Download agent trajectories

To download all 13 archives:

```sh
gh release download v1.0.0 --repo cz2014/CompMat-Bench \
  --pattern '*.tar.gz' --dir compmat_bench_trajectories
```

Use a specific archive filename as the pattern to download one campaign and
agent. The [trajectory manifest](../trajectories/manifest.json) maps each
attempt to its campaign, agent, score and path within the archives. The
release page also carries the manifest and the trajectory documentation as
the assets `trajectory_manifest.json`, `trajectories_README.md`,
`trajectories_DATASHEET.md` and `trajectories_REDACTION_REPORT.md`; they are
the files of this repository's `trajectories/` directory.
