# Task: input decks for the harmonic force constants of a boron arsenide supercell

Prepare self-consistent ground-state input decks for finite-displacement
harmonic force constants of cubic boron arsenide, a semiconductor.

You are given two files:

- `bas_conventional.vasp`: the reference cell at the experimental lattice
  constant. Use it exactly as given, with no relaxation.
- `pseudo_catalog.yaml`: the identity and metadata of the two pseudopotential
  files to reference in the generated input decks. Refer to them by these file names.

## The calculation

Write the complete Quantum ESPRESSO `pw.x` input decks for the displacement
set, at the following specification:

- Supercell: the cubic $4 \times 4 \times 4$ repetition of the given
  conventional cell.
- Exchange-correlation functional: the nonlocal van der Waals density
  functional vdW-DF2 with C09 exchange, used with the pseudopotentials
  specified in the catalog.
- Plane-wave kinetic-energy cutoff for the wavefunctions: 100 Ry.
- Brillouin-zone sampling: the $\Gamma$ point only.
- Electronic self-consistency threshold on the total energy: $10^{-12}$ Ry.
- Forces must be computed and printed for every run.
- One undisplaced reference run, and displaced runs in which exactly one atom
  of the supercell is moved along one Cartesian axis by a small amplitude
  (between 0.005 and 0.03 A) from its equilibrium position, all other atoms
  untouched. The set must include at least one displaced run for each of the
  two species. Displacing each chosen atom both in the positive and in the
  negative direction is optional.
- Give the structure explicitly in every deck (explicit cell vectors and
  atomic coordinates; do not rely on engine-internal Bravais-lattice presets).

No calculation is to be run; only the input decks are graded.

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"reference": "answer/reference", "<displaced label>": "answer/<displaced label>", ...}}
```

with one entry for the undisplaced reference run, labelled exactly
`reference`, and one entry per displaced run under labels of your choice
(letters, digits and underscores). For each label, a directory
`answer/<label>/` must contain exactly one input file named `pw.in` (the
`pw.x` deck of that run).

Do not submit pseudopotential files; their identities are fixed by
`pseudo_catalog.yaml` and are read from your decks, not from copies. Extra
files beyond the labelled decks are ignored.
