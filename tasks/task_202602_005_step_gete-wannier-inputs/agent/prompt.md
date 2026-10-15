# Task: input decks for a fully relativistic Wannier model of rhombohedral GeTe

You are given two files:

- `POSCAR_GeTe`: the experimental reference structure. Use it exactly as
  given in every deck that carries a structure, with no relaxation.
- `pseudo_catalog.yaml`: the available pseudopotential files and their metadata.
  Use the file names listed in the catalog.

The target is a spin-orbit-coupled model of the Rashba-split valence bands
of ferroelectric GeTe, with meV-scale energy accuracy.

Write the complete input decks for the four-step pipeline that turns this cell
into a compact spinor tight-binding model:

1. `scf`: a Quantum ESPRESSO `pw.x` self-consistent ground-state run.
2. `nscf`: a `pw.x` non-self-consistent run providing the wavefunctions and
   eigenvalues that the wannierization consumes on its Brillouin-zone grid
   (see below).
3. `pw2wan`: a `pw2wannier90` run producing, from the nscf wavefunctions, the
   overlap and initial-projection matrices and the spin matrices, the spin
   matrices in formatted (text) form.
4. `wannier`: a `wannier90.x` run producing a 12-band spinor Wannier model
   spanning the Ge p and Te p orbitals. Start the localization from
   atom-centred full p-shell projections on both Ge and Te, using any equivalent
   complete-shell representation. Use species-centred forms (no site coordinates
   or rotated axes). Output the real-space Hamiltonian and the
   Bloch-to-Wannier rotation matrices.

Facts about this system that you cannot derive without running the
calculation, fixed here:

- The Wannier model is built on a 16 x 16 x 16 Brillouin-zone grid.
- The twelve p-derived spinor bands of this system form an isolated group: at
  every point of that grid they are separated by finite energy gaps both from
  all lower-lying (semicore- and s-derived) bands and from all higher
  conduction bands. Build the model from exactly this isolated group.

The four decks must chain: each step must find the files the earlier steps
produced, so the runs share one consistent working layout across the decks.
Give the structure explicitly wherever a deck needs it (explicit cell vectors
plus atomic coordinates; do not rely on engine-internal Bravais-lattice
presets).

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"scf": "answer/scf", "nscf": "answer/nscf",
          "pw2wan": "answer/pw2wan", "wannier": "answer/wannier"}}
```

and, for each label, a directory `answer/<label>/` containing exactly one
input file, named as follows:

- `answer/scf/pw.in` (the `pw.x` scf deck)
- `answer/nscf/pw.in` (the `pw.x` nscf deck)
- `answer/pw2wan/pw2wan.in` (the `pw2wannier90` deck)
- `answer/wannier/wannier90.win` (the `wannier90.x` deck)

Do not submit pseudopotential files; their identities are fixed by
`pseudo_catalog.yaml` and are read from your decks, not from copies. Extra
files beyond the prompt's labels are ignored.
