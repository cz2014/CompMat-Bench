# Task: input files for the DFT+U+SOC valley band structure of monolayer 2H-RuCl2

You are given one structure, `POSCAR_2H_RuCl2`, a relaxed monolayer 2H-RuCl2
cell. Keep its lattice vectors and atomic positions fixed in every run.
`potcar_catalog.yaml` lists the VASP 5.4 PAW PBE pseudopotentials and
cutoffs to use for Ru and Cl.

Prepare inputs for the DFT+U+SOC band structure of this ferromagnetic insulator.

## Fixed physical choices (use exactly these)

- **Correlation:** treat the Ru 4d electrons with DFT+U in the Dudarev
  (rotationally invariant, single effective U) scheme, with an effective
  U = 2 eV on Ru only (no U on Cl).
- **Magnetic order:** ferromagnetic, with the magnetization along the
  out-of-plane (c) axis. Either sign of the out-of-plane moment is acceptable.
- **Band path:** sample the band structure along Gamma -> K -> M -> K' ->
  Gamma, with the two corners taken as the distinct points
  $K = (\tfrac{1}{3}, \tfrac{1}{3}, 0)$ and $K' = (\tfrac{2}{3}, -\tfrac{1}{3}, 0)$
  in the reciprocal basis of this cell.

Beyond these fixed choices, everything else in the two input sets is yours to choose
so that the calculation is sound for this system.

## What to produce

Prepare two input sets that form the standard two-step chain for a spin-orbit
band structure:

(1) **`scf`** -- a self-consistent calculation that converges the charge density
and the ferromagnetic, out-of-plane, spin-orbit-coupled ground state on this
cell, and writes the converged charge density.

(2) **`bands`** -- a band-structure calculation along the high-symmetry path
above, evaluated on the fixed charge density from the `scf` step (a
non-self-consistent band line), carrying the same correlation, magnetic, and
spin-orbit setup.

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"scf": "answer/scf", "bands": "answer/bands"}}
```

and, for each label, a directory `answer/<label>/` containing exactly three
files: `INCAR`, `KPOINTS`, and `POSCAR`. Do not
submit a POTCAR; the pseudopotential identities are fixed by
`potcar_catalog.yaml` and are not graded from your files.
