# Task: build a RuO2/SnTe(110) interface slab and its two-step VASP deck

Prepare the VASP inputs for a commensurate RuO2(001) film on SnTe(110),
using `bulk_RuO2.vasp` and `bulk_SnTe.vasp` as the starting bulk cells.
Treat the system as a metallic RuO2 film on a narrow-gap SnTe substrate,
forming a polar slab.
`potcar_catalog.yaml` specifies the available VASP 5.4 PAW PBE pseudopotentials.

## Required structure

Pair a 1x3 RuO2(001) surface mesh with a 1x2 SnTe(110) surface mesh.
The rectangular in-plane cell is 6.372 A along SnTe [001] (x) and
18.022 A along SnTe [1-10] (y); the film is strained onto the substrate.
The surface normal is z.

The film has 6 atomic layers on 40 SnTe(110) planes, with exactly
160 Sn, 160 Te, 36 Ru and 72 O atoms. In the bottom RuO2 plane next to SnTe,
each Ru's two nearest coplanar O neighbours must lie along y.
The interface gap is about 2.2 A and the vacuum about 20 A.
The film's lateral registry against the substrate is free.
Fix the bottom 20 SnTe planes (160 atoms); all other atoms are free to relax.

Submit the same unrelaxed interface POSCAR for both runs below.

## Required calculations

Both runs use PBE and U_eff = 3.0 eV on Ru d states only, with no Hubbard
correction on Sn, Te or O. Choose converged electronic settings appropriate
to this system and able to resolve a few-meV spin-orbit gap.

Preserve the antiparallel order of the two rutile Ru sublattices. Initially,
use opposite moments of about 1 Bohr magneton on Ru and zero on Sn, Te and O.

- **`relax`**: collinear spin-polarized ionic relaxation without spin-orbit
  coupling, at fixed cell and converged forces, with the fixed atoms above.
- **`static`**: a self-consistent noncollinear spin-orbit single point on the
  relaxed geometry, without ionic motion. Seed moments from the relaxed
  collinear state, with the Ru sublattices antiparallel along a common Neel axis
  and zero moments on Sn, Te and O.

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"relax": "answer/relax", "static": "answer/static"}}
```

and, for each label, a directory `answer/<label>/` containing exactly three
files: `INCAR`, `KPOINTS`, and `POSCAR`. Do not submit a POTCAR; the
pseudopotential identities are fixed by `potcar_catalog.yaml` and are not graded
from your files.
