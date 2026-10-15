# Spin splitting and magnetism of an altermagnetic MnTe pipeline

You are given a completed set of density-functional calculations for pristine
manganese telluride (MnTe): collinear spin-polarized runs without spin-orbit
coupling, whose magnetic order and geometry are recorded in the staged input
decks.

MnTe with this magnetic order is studied as an altermagnet: a collinear
magnet whose two spin channels may nonetheless split at generic
momenta, with the size and sign of the splitting depending on the direction in
the Brillouin zone. The quantities below probe that splitting along the sampled
paths.

## Provided calculations

The four runs below are the completed pipeline at the optimized geometry. Each
directory holds the input deck (`pw.in`) and the standard output (`pw.out`) of one
finished plane-wave run.

- `runs/scf_prim/` -- self-consistent-field run of the primitive cell.
- `runs/bands_prim/` -- non-self-consistent band run of the primitive cell, giving
  spin-resolved eigenvalues along an explicit k-path listed in the deck.
- `runs/scf_sc/` -- self-consistent-field run of the supercell.
- `runs/bands_sc/` -- non-self-consistent band run of the supercell, giving
  spin-resolved eigenvalues along the folded image of the same primitive path,
  on an explicit k-path listed in the deck.

## Conventions and pinned facts

Facts about this system and this task that you cannot derive from the files alone,
fixed here:

- Spin-channel order: the two spin channels are labelled "first" and "second" in the
  order the outputs report them. Every signed splitting below is
  E(first spin channel) - E(second spin channel).
- Highest valence band: the topmost occupied band per spin channel. The number of
  occupied bands per spin follows from the electron count in the outputs.
- L is (1/2, 0, 1/2); L' is the second, inequivalent L point reached on the last
  segment.

## What to report

- `gap_eV`: the band gap of the primitive cell, evaluated over the primitive band
  path (a path-restricted gap, not a full Brillouin-zone search), from the primitive
  band run.
- `mn_lsm_muB`: the magnitude of the site-projected local magnetic moment of a single
  Mn atom, from the supercell self-consistent run. Every Mn site is equivalent.
- `amss_valence_max_eV`: the maximum of |E(first) - E(second)| taken over all
  valence bands and all sampled points of the full L-Gamma-L' path (primitive run).
- `split_mean_LG_eV`: the mean, over the sampled points of the L-to-Gamma segment, of
  the signed splitting E(first) - E(second) of the highest valence band.
- `split_mean_GLp_eV`: the mean, over the sampled points of the Gamma-to-L' segment,
  of the signed splitting E(first) - E(second) of the highest valence band.
  For both means, each segment's sampled points are the deck's own list for that
  segment, from its zone-boundary endpoint up to and including Gamma; Gamma
  therefore enters each of the two means once.
- `sc_vbm_at_gamma`: whether the valence-band maximum of the supercell band run sits
  at the zone center of the folded path (a boolean).

## Output format

Write a single `answer.json` in the workspace root with exactly these fields:

```json
{
  "gap_eV": <float>,
  "mn_lsm_muB": <float>,
  "amss_valence_max_eV": <float>,
  "split_mean_LG_eV": <float>,
  "split_mean_GLp_eV": <float>,
  "sc_vbm_at_gamma": <bool>
}
```

Numbers only, no prose is read. Report the floats to at least four decimal places.
