# Task: extract the reconstruction fields of a relaxed R-type WS2/WSe2 moire

Determine the layer-resolved strain, interlayer spacing, corrugation,
adhesion and domain redistribution of a reconstructed R-type WS2/WSe2 moire.
The pristine lattice constants are $3.17\,\text{Å}$ for WS2 and
$3.30\,\text{Å}$ for WSe2.

## What you are given

- `lammps.dat_min`: relaxed structure.
- `lammps.dat`: rigid pre-relaxation structure.
- `kc.dump`: relaxed-structure LAMMPS dump; `c_pe` is the per-atom
  interlayer KC energy in eV.

All files share atom IDs and types. WS2 is the lower layer, WSe2 the upper.

## Conventions (use exactly these)

Neighbor distances are periodic in the moire cell. Local biaxial strain uses
the mean in-plane distance to the six nearest same-layer W neighbors as the
local lattice constant, relative to the pristine value.

Fields are carried on WS2 W sites. Each is paired with its in-plane-nearest
WSe2 W atom. The paired WSe2 atom supplies the WSe2 strain; the pair's signed
vertical separation (WSe2 minus WS2) is the interlayer distance, and the sum
of its two per-atom KC energies defines the adhesion observable.

Registry is the pair's in-plane offset, WSe2 minus WS2, in the WS2 monolayer
basis (moire vectors divided by 26), modulo 1. The locale coordinates are
$R^h_h=(0,0)$, $R^X_h=(1/3,2/3)$ and $R^M_h=(2/3,1/3)$.
Registry distance is Euclidean on the fractional-coordinate unit torus,
with each difference component in $[-0.5,0.5)$.

A per-locale field value is the mean over WS2 W sites within registry radius
**0.10**. A domain area fraction is the fraction of all WS2 W sites within
radius **0.15**, evaluated separately for rigid and relaxed structures.
Layer corrugation is the peak-to-peak W-plane height variation.

## What to report

Report relaxed per-locale strains for both layers (signed percent),
interlayer distances (Å), and adhesion energies (eV); each layer's
corrugation (Å); and the rigid and relaxed domain fractions at $R^X_h$ and
$R^h_h$.

## Output format

Write `answer.json` in the workspace root with exactly these fields (all numbers,
units as named in the keys):

```json
{
  "eps_WS2_Rhh_pct": <float>,
  "eps_WS2_RXh_pct": <float>,
  "eps_WS2_RMh_pct": <float>,
  "eps_WSe2_Rhh_pct": <float>,
  "eps_WSe2_RXh_pct": <float>,
  "eps_WSe2_RMh_pct": <float>,
  "d_Rhh_A": <float>,
  "d_RXh_A": <float>,
  "d_RMh_A": <float>,
  "corrugation_WS2_A": <float>,
  "corrugation_WSe2_A": <float>,
  "adhesion_Rhh_eV": <float>,
  "adhesion_RXh_eV": <float>,
  "adhesion_RMh_eV": <float>,
  "area_frac_RXh_rigid": <float>,
  "area_frac_RXh_relaxed": <float>,
  "area_frac_Rhh_rigid": <float>,
  "area_frac_Rhh_relaxed": <float>
}
```

The `Rhh`, `RXh`, `RMh` field stems denote $R^h_h$, $R^X_h$, $R^M_h$.
