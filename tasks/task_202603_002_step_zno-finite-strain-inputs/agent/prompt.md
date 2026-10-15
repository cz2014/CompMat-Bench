# Task: input decks for the relaxation and finite-strain elastic and piezoelectric response of wurtzite ZnO

Prepare the finite-strain calculations needed to determine the elastic and
piezoelectric response of insulating wurtzite ZnO.

You are given:

- `ZnO_wurtzite_start.vasp` -- the starting wurtzite ZnO cell for the
  relaxation, in VASP POSCAR format.
- `ZnO_wurtzite_relaxed.vasp` -- the stress-free relaxed cell on which every
  deformation below is built. Treat it as a reference that fixes the
  conventions: lattice vectors, atom identities and the polar axis, which is the
  third lattice vector with $+c$ pointing from a cation to the oxygen
  directly above it. Build the deformed cells from it exactly as given. Consistent
  reordering of atom records is allowed.
- `pseudo_catalog.yaml` -- the identity and valence of the two
  pseudopotential files to reference in the generated input decks. Refer to them by
  these file names.

## The deformations (fixed for this task)

Each deformation is a homogeneous transformation of the lattice vectors of
the relaxed cell with every fractional coordinate kept. With Cartesian axes
$x, y$ in the basal plane and $z$ along the third lattice vector:

- **D1**, in-plane biaxial: $x \to (1 + e)\,x$, $y \to (1 + e)\,y$,
  $z \to z$.
- **D3**, uniaxial along $c$: $x \to x$, $y \to y$, $z \to (1 + e)\,z$.
- **D4**, $zx$ shear: $x \to x + \tfrac{e}{2}\,z$, $z \to z + \tfrac{e}{2}\,x$,
  $y \to y$ (an engineering shear strain $e$).

The rungs to prepare are D1 at $e = +0.008$, D3 at $e = +0.008$ and
$e = -0.008$, and D4 at $e = +0.008$.

## What to write

1. `vcrelax`: a variable-cell relaxation of the starting cell to its
   stress-free state, every cell degree of freedom free.

2. For each of the four rungs, a fixed-cell relaxation of the ions inside
   the deformed cell that reports the stress tensor of the relaxed-ion
   deformed cell (labels `D1_p_relax`, `D3_p_relax`, `D3_m_relax`,
   `D4_p_relax`).

3. For the three rungs D1 ($+0.008$), D3 ($+0.008$) and D3 ($-0.008$), a
   Berry-phase polarization run along the polar axis on the relaxed-ion
   deformed cell that starts from the density of the corresponding
   relaxation (labels `D1_p_berry`, `D3_p_berry`, `D3_m_berry`); the two
   runs of a rung must chain, i.e. the Berry-phase run must find the files
   the relaxation produced. You are writing all decks before anything runs,
   so where the Berry-phase deck must carry the relaxed ionic positions
   explicitly, use your best available representation of them; the
   unrelaxed deformed build is acceptable there.

The material is an insulator. Give the structure explicitly in every deck
(explicit cell vectors plus atomic coordinates; do not rely on
engine-internal Bravais-lattice presets). These decks target Quantum ESPRESSO. For the eventual calculation, assume
the listed pseudopotential files will be placed in a directory of your
choosing, which your decks name. No calculation is run in this task.

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"vcrelax": "answer/vcrelax",
          "D1_p_relax": "answer/D1_p_relax",
          "D1_p_berry": "answer/D1_p_berry",
          "D3_p_relax": "answer/D3_p_relax",
          "D3_p_berry": "answer/D3_p_berry",
          "D3_m_relax": "answer/D3_m_relax",
          "D3_m_berry": "answer/D3_m_berry",
          "D4_p_relax": "answer/D4_p_relax"}}
```

and, for each label, a directory `answer/<label>/` containing exactly one
`pw.x` input file named `pw.in`. Do not submit pseudopotential files; their
identities are fixed by `pseudo_catalog.yaml` and are read from your decks,
not from copies. Extra files beyond the prompt's labels are ignored.
