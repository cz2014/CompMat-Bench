# Task: input decks for the pristine MnTe altermagnet pipeline

You are given two files:

- `MnTe_start.vasp`: the starting cell, in VASP POSCAR format. Use it as the
  geometry the cell optimization begins from, and treat it as the reference
  that fixes the conventions: lattice setting, axis orientation, site order.
- `pseudo_catalog.yaml`: the identity and valence of the two pseudopotential
  files to reference in the generated input decks. Refer to them by these file names.

MnTe in this structure is a collinear magnetic semiconductor with a
finite band gap and no spin-orbit coupling. Its manganese atoms carry local
moments that order in the A-type antiferromagnetic pattern: the moments are
aligned within each manganese ab-plane and reverse from one plane to the next
along the c axis, so the moments compensate exactly and the cell has zero net
magnetization.

This magnetic order makes the material an altermagnet: the spin splitting of
the bands depends on the direction in the Brillouin zone. Along the zone path
Gamma-M-K-Gamma-A-L the two spin channels stay degenerate, while along the
two-segment path L-Gamma-L' whose ends are two symmetry-inequivalent L points
the highest valence bands split strongly, with the splitting reversing sign between the
L-Gamma segment and the Gamma-L' segment. Reproducing this direction-dependent
splitting, and its disappearance on the nodal path, is the objective.

Write the complete input decks for the following five-run Quantum ESPRESSO
`pw.x` pipeline. Runs 2 to 5 execute at the geometry the optimization
produces. You are writing all five decks before anything runs, so where a
later deck must carry that geometry explicitly, use your best available
representation of it; the starting cell is acceptable there. Each deck is
judged on specifying the right calculation on the right crystal, not on
anticipating the optimized coordinates.

1. `relax`: optimize both the cell and the ionic positions of the primitive
   cell, starting from the staged structure.
2. `scf`: a self-consistent ground-state run of the optimized primitive cell.
3. `bands`: a band-structure run of the optimized primitive cell, sampling
   an explicit path that covers the nodal segments Gamma-M-K-Gamma-A-L and the
   two-segment path L-Gamma-L' ending at two symmetry-inequivalent L points,
   resolving both spin channels.
4. `scf_sc`: a self-consistent run of the 2 x 2 x 2 supercell of the optimized
   primitive cell.
5. `bands_sc`: a band-structure run of that supercell, sampling the same
   physical L-Gamma-L' line as run 3, now expressed in the supercell
   Brillouin zone.

Facts about this system that you cannot derive without running the
calculation, fixed here:

- An on-site Hubbard correction of 3 eV is applied to the manganese 3d shell.
- The doping study these baselines feed uses the 2 x 2 x 2 supercell; build
  runs 4 and 5 on exactly that supercell.
- The band runs must resolve the two spin channels separately and must include
  enough bands to cover the band-edge region on both sides of the gap.

The scf and its band run share one consistent working layout so the band run
finds the self-consistent data; the supercell scf and its band run likewise
share a layout. Refer to the pseudopotentials by the catalog file names; do
not include pseudopotential file content in your submission.

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"relax": "answer/relax", "scf": "answer/scf",
          "bands": "answer/bands", "scf_sc": "answer/scf_sc",
          "bands_sc": "answer/bands_sc"}}
```

and, for each label, a directory `answer/<label>/` containing exactly one
`pw.x` input file named `pw.in`:

- `answer/relax/pw.in`
- `answer/scf/pw.in`
- `answer/bands/pw.in`
- `answer/scf_sc/pw.in`
- `answer/bands_sc/pw.in`

Do not submit pseudopotential files; their identities are fixed by
`pseudo_catalog.yaml` and are read from your decks, not from copies. Extra
files beyond the prompt's labels are ignored.
