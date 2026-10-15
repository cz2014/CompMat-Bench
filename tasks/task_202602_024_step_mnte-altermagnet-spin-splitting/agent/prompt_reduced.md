# Spin splitting and magnetism of an altermagnetic MnTe pipeline

Determine the following from the completed collinear, no-SOC MnTe runs.
`runs/scf_prim/`, `runs/bands_prim/`, `runs/scf_sc/` and `runs/bands_sc/`
contain `pw.in` and `pw.out` for primitive and supercell SCF/band runs.
Geometry, magnetic order and explicit paths are recorded in the decks;
the supercell path is the folded image of the primitive path.

Signed spin splitting is E(first reported spin channel) minus E(second).
L is (1/2,0,1/2); L' is the inequivalent L endpoint of the last segment.

Report:

- `gap_eV`: the primitive-cell gap over its sampled band path;
- `mn_lsm_muB`: the magnitude of one Mn site's projected local moment from
  the supercell SCF run; all Mn sites are equivalent;
- `amss_valence_max_eV`: maximum absolute spin splitting over every valence
  band and sampled point of the primitive $L-\Gamma-L'$ path;
- `split_mean_LG_eV`, `split_mean_GLp_eV`: the mean signed splitting of the
  highest occupied band per spin channel on each respective segment.
  Use the deck's sample lists; $\Gamma$ enters each segment's mean once;
- `sc_vbm_at_gamma`: whether the supercell band run's valence maximum is
  at the zone centre of the folded path.

## Output format

Write `answer.json` in the workspace root with exactly these fields:

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

Report floats to at least four decimal places. No prose is read.
