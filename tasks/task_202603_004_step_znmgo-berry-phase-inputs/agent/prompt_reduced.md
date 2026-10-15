# Task: input decks for the Berry-phase polarization path of wurtzite Zn$_{1-x}$Mg$_x$O

Prepare Quantum ESPRESSO inputs for the fixed-cell Berry-phase path below
from the layered reference to relaxed wurtzite Zn$_{1-x}$Mg$_x$O.

You are given:

- `wz_x000_relaxed.vasp` and `wz_x050_relaxed.vasp` -- the relaxed
  stress-free reference cells at $x = 0$ and $x = 0.5$. Treat each
  as a reference that fixes the conventions: the lattice vectors, atom
  identities and cation decoration are used exactly as given; consistent
  reordering of the atom records is allowed, and the third
  lattice vector is the polar axis, with each apical oxygen at a positive
  axial offset from its paired cation. Do not relax or re-orient anything.
- `pseudo_catalog.yaml` -- the identity and valence of the three
  pseudopotential files to reference in the generated input decks. Refer to them by
  these file names.

## The path (fixed for this task)

The path is parametrized by $\lambda \in [0, 1]$: $\lambda = 1$ is the
relaxed cell exactly as given, $\lambda = 0$ is the layered-hexagonal
reference built from it. Along the path the lattice vectors and every
cation position are frozen, and every oxygen moves along the $c$ axis only.
Pair each oxygen with the nearest cation column in projection onto the basal
plane, then with the periodic cation immediately below it along $c$.
Keep any lateral offset within that pair. Write $u_i$ for the offset
of oxygen $i$ above that cation
along $c$, in units of the wurtzite $c$ spacing (the cation-plane repeat
along the third lattice vector: the full third vector of the primitive
cell, half of it for the doubled supercell). Then

$$
u_i(\lambda) = \tfrac{1}{2} - \lambda\,\bigl(\tfrac{1}{2} - u_i^{\mathrm{relaxed}}\bigr),
$$

## What to write

For each of the three structures

- $x = 0.5$, $\lambda = 0$ (label `x050_lam0000`),
- $x = 0.5$, $\lambda = 0.5$ (label `x050_lam0500`),
- $x = 0$, $\lambda = 0.25$ (label `x000_lam0250`),

write two chained pw.x input decks:

1. a fixed-cell self-consistent ground-state run of that structure
   (label suffix `_scf`);
2. a Berry-phase polarization run along the polar axis that starts from the
   density of run 1 (label suffix `_berry`); the two runs must chain.

The material is an insulator at every point of the path. Give the structure
explicitly in every deck (explicit cell vectors and atomic coordinates).
These decks target Quantum ESPRESSO. For the eventual calculation, assume
the listed pseudopotential files will be placed in a directory of your
choosing, which your decks name. No calculation is run in this task.

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"x050_lam0000_scf": "answer/x050_lam0000_scf",
          "x050_lam0000_berry": "answer/x050_lam0000_berry",
          "x050_lam0500_scf": "answer/x050_lam0500_scf",
          "x050_lam0500_berry": "answer/x050_lam0500_berry",
          "x000_lam0250_scf": "answer/x000_lam0250_scf",
          "x000_lam0250_berry": "answer/x000_lam0250_berry"}}
```

and, for each label, a directory `answer/<label>/` containing exactly one
`pw.x` input file named `pw.in`. Do not submit pseudopotential files; their
identities are fixed by `pseudo_catalog.yaml` and are read from your decks,
not from copies. Extra files beyond the prompt's labels are ignored.
