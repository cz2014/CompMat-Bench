# Task: build a RuO2/SnTe(110) interface slab and its two-step VASP deck

You will construct a commensurate heterostructure of a RuO2 thin film on a
SnTe(110) substrate and prepare the VASP inputs for relaxing it and computing
its spin-orbit electronic structure. RuO2 is a rutile-type metal whose two Ru
sublattices order antiparallel (an altermagnet); SnTe is a rock-salt topological
crystalline insulator. The goal is the input set for studying the magnetic
proximity effect of the RuO2 film on the SnTe surface states.

You are given two bulk unit cells as starting materials:

- `bulk_SnTe.vasp` -- the SnTe starting cell.
- `bulk_RuO2.vasp` -- the RuO2 starting cell.

You are also given `potcar_catalog.yaml`, listing the available VASP 5.4 PAW PBE
pseudopotential identities for the species you will use.

## The interface to build

Build a single slab with the following construction:

1. **Facets and registry.** Construct the specified commensurate interface: a
   RuO2(001) film on a SnTe(110) substrate, pairing a 1x3 RuO2(001) surface mesh
   with a 1x2 SnTe(110) surface mesh.
   The resulting in-plane cell is rectangular,
   6.372 A along [001] (call this the x axis) by 18.022 A along [1-10] (the y
   axis); the film is strained onto the substrate (about 0.5% along x, about
   5.5% along y).
2. **Thickness.** 6 atomic layers of RuO2 stacked on 40 atomic planes of
   SnTe(110). The total cell is 428 atoms: 160 Sn, 160 Te, 36 Ru, 72 O.
   Choose the RuO2 termination so that, in its bottom atomic plane adjacent to
   SnTe, each Ru's two nearest coplanar O neighbors lie along the y axis.
3. **Stacking and vacuum.** The RuO2 film sits on top of the SnTe substrate with
   an interface gap of about 2.2 A, and a vacuum region of about 20 A separates
   the slab from its periodic image along the surface normal (the z axis). The
   lateral registry of the film against the substrate is yours to choose; the
   cell, the counts, the layer order, and the interface gap are fixed.
4. **Fixed region.** Hold the bottom 20 SnTe(110) planes (160 atoms) rigid via
   the POSCAR's selective dynamics; every other atom is free to relax.

Use this exact cell and these exact counts. Submit the same POSCAR for both runs
below (the two steps share one geometry input; the relaxation moves the free
atoms, the static reads the relaxed result -- but for this task you submit the
unrelaxed interface as the POSCAR of both runs).

## The two-step deck

Produce inputs for two runs sharing one physical setup. Choose electronic
settings you judge converged and appropriate to this system, noting what it
is: a metallic oxide film on a narrow-gap semiconductor substrate; a polar
slab (its top and bottom surfaces are inequivalent) with a large vacuum along
z; and a calculation whose downstream observable is a few-meV spin-orbit gap,
so the electronic convergence must be tight enough to resolve that scale. The
available pseudopotentials and their cutoffs are in `potcar_catalog.yaml`; use
the PBE functional. Physics the shared setup must express:

- A Hubbard correction of U_eff = 3.0 eV on the Ru d states only, in both
  runs (no correction on Sn, Te, or O).
- Altermagnetic initial order: give the two rutile Ru sublattices opposite
  initial magnetic moments (about 1 Bohr magneton each, one sublattice +, the
  other -; identify the two sublattices by their rutile positions). Sn, Te and O
  start at zero moment. Do not let the run collapse to a nonmagnetic or
  ferromagnetic state.

The two runs:

- **`relax`** -- a collinear spin-polarized ionic relaxation, no spin-orbit
  coupling. The ions relax at fixed cell (the cell shape and volume do not
  move), converged on the forces; the bottom 20 SnTe planes are held by the
  POSCAR selective dynamics.
- **`static`** -- a single self-consistent static with noncollinear spin-orbit
  coupling on the relaxed geometry (no ionic motion), its magnetic moments
  seeded from the relaxed collinear state: the two Ru sublattices antiparallel
  along a common Neel axis, zero on Sn/Te/O. Keep the Hubbard correction and
  the shared setup above.

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"relax": "answer/relax", "static": "answer/static"}}
```

and, for each label, a directory `answer/<label>/` containing exactly three
files: `INCAR`, `KPOINTS`, and `POSCAR`. Do not submit a POTCAR; the
pseudopotential identities are fixed by `potcar_catalog.yaml` and are not graded
from your files.
