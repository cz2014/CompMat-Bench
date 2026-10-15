# VAMPIRE 7.0.0 (source bundle)

Unmodified source of VAMPIRE, an atomistic/multiscale spin-dynamics and
Monte Carlo code for simulating magnetic materials. Upstream Git repository:
<https://github.com/richard-evans/vampire>. Taken from commit
`525bc27ee44c525aee229570f30f3d4c61d54f66` ("Updates to the manual for v7.0"),
which is the content of the 7.0.0 release tag.

## What was omitted

Relative to the upstream commit, this bundle omits:

- `obj/` (top level) and the three matching build-artifact directories
  `test/unit/obj/`, `test/integration/obj/`, `util/vdc/obj/`. Each contained
  only a placeholder `.gitignore`, no compiled objects or binaries; they are
  omitted because they are build output locations, not source.

Nothing else was removed. The full checkout is 6.2 MB (634 files), well
under the size threshold for trimming figure assets, so `manual/figures/`
and `samples/` are kept intact. There is no `.github/`, CI config, or
Dockerfile in the upstream tree. No
remaining file was altered; all license headers are preserved byte for byte.

## License

VAMPIRE ships two license files, both included unmodified: `license` (the
GNU GPL, version 2) and `BSD_licence` (a 2-clause BSD license, copyright
Richard F L Evans, 2016). Per-file header comments are mixed: most source
files (around 301 of 501 `.cpp`/`.hpp` files) carry a "Free BSD licence"
notice, while a smaller, older set (around 104) still carries a GPL header.
Consult the header of each file for the license that applies to it; both
license texts are reproduced in full in the two files above.

## Building and running

Serial build with a C++ compiler:

    make serial

This produces the `vampire-serial` executable (`make parallel` builds an
MPI version; `make vdc` builds the `util/vdc` visualization/data-conversion
tool; `make` alone builds all three). The manual is LaTeX source under
`manual/` (`manual/vampire-manual.tex` is the top-level document); it is
provided as source only and is not pre-built to PDF in this bundle. Within
it:

- the input file command reference is `manual/InputFileCommandReference/Chapter.tex`,
- the material file command reference is `manual/MaterialFileCommandReference/Chapter.tex`,
- the unit-cell file format is described in `manual/unit-cell-file/Chapter.tex`.

Example input decks are under `samples/`. The package is provided as source
only; no prebuilt binaries are included.
