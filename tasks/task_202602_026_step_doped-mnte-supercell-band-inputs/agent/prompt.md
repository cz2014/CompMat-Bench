# Task: input decks for the band structures of dopant-pair-substituted MnTe supercells

You are given two files:

- `MnTe_relaxed.vasp`: the relaxed primitive cell of MnTe, in VASP POSCAR
  format. Build the supercells described below as
  replicas of this cell, and keep the lattice fixed everywhere (see the doping
  convention below).
- `pseudo_catalog.yaml`: the identity and valence of the four pseudopotential
  files to reference in the generated input decks. Refer to them by these file names.

MnTe in this structure is a magnetic semiconductor. Its Mn moments order in an
A-type antiferromagnetic pattern: the moments are ferromagnetic within each
basal plane and reverse from one Mn plane to the next along the c axis, so the
order is collinear with zero net magnetization. This ordering makes the host an
altermagnet, whose hallmark is a momentum-dependent spin splitting of the bands:
along the line joining the two inequivalent L points of the hexagonal Brillouin
zone through the zone centre, the spin splitting is large and reverses sign
between the two halves of the line, while it stays absent on the zone-centre
symmetry directions.

Substituting a nonmagnetic aliovalent atom for tellurium probes how this
splitting survives when the crystal symmetry is lowered. Build a 2x2x2 supercell
of the staged cell with the A-type antiferromagnetic Mn order, and
study two substituted configurations, replacing a pair of the tellurium sites in
each:

- Configuration `sb_alt`: substitute antimony at the two supercell fractional
  sites (0.166667, 0.333333, 0.125000) and (0.166667, 0.833333, 0.125000).
- Configuration `i_quasi`: substitute iodine at the two supercell fractional
  sites (0.166667, 0.333333, 0.125000) and (0.333333, 0.166667, 0.375000).

An upstream symmetry analysis of the two doped crystals found that the first
pair leaves the crystal in a symmetry that still supports the altermagnetic
spin splitting, while the second pair lowers the symmetry so that this
protection is lost. Antimony and iodine each replace tellurium aliovalently; you
have their valence in the catalog.

Facts about this study that you cannot derive without the upstream work, fixed
here:

- The doping is applied to the fixed relaxed host lattice: the two substituted
  supercells keep the staged cell's lattice, doubled along each axis, and only
  the ions relax. The lattice is not re-optimized per configuration.
- Treat the Mn 3d states with a Hubbard correction of U = 3 eV.
- The dopant-site pairs and their symmetry consequence are the two given above.

For each configuration, write three chained input decks:

1. a fixed-cell ionic relaxation of the doped supercell;
2. a self-consistent run at the relaxed ions;
3. a band-structure run.

You are writing all decks before anything runs, so where the later decks must
carry the relaxed ionic positions explicitly, use your best available
representation of them; the unrelaxed substituted build is acceptable there.
Each deck is judged on specifying the right calculation on the right crystal,
not on anticipating the relaxed coordinates.

The band path is the L-Gamma-L' line of the primitive Brillouin zone, which runs
through two inequivalent L points. The supercell folds this same physical line
into its own Brillouin zone, so the band run's k-points are the folded images of
that primitive line. The self-consistent and band runs of a configuration must
chain: the band run must find the files the self-consistent run produced, so
they share one consistent working layout.

Give the structure explicitly wherever a deck needs it (explicit cell vectors
plus atomic coordinates; do not rely on engine-internal Bravais-lattice presets).

## Output format

Write a manifest `answer.json` in the workspace root:

```json
{"runs": {"relax_sb_alt": "answer/relax_sb_alt",
          "scf_sb_alt": "answer/scf_sb_alt",
          "bands_sb_alt": "answer/bands_sb_alt",
          "relax_i_quasi": "answer/relax_i_quasi",
          "scf_i_quasi": "answer/scf_i_quasi",
          "bands_i_quasi": "answer/bands_i_quasi"}}
```

and, for each label, a directory `answer/<label>/` containing exactly one
`pw.x` input file named `pw.in`:

- `answer/relax_sb_alt/pw.in`, `answer/scf_sb_alt/pw.in`, `answer/bands_sb_alt/pw.in`
- `answer/relax_i_quasi/pw.in`, `answer/scf_i_quasi/pw.in`, `answer/bands_i_quasi/pw.in`

Do not submit pseudopotential files; their identities are fixed by
`pseudo_catalog.yaml` and are read from your decks, not from copies. Extra files
beyond the prompt's labels are ignored.
