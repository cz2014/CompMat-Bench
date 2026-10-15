# Task: read the band offsets and layer character at K of a heterobilayer

Determine the layer-resolved band offsets, direct gap, band-edge localization
and alignment class at K for a commensurate $R^X_h$ WS2/WSe2 heterobilayer.

## What you are given

`vasprun.xml.gz`: a VASP non-self-consistent band run with spin-orbit
coupling and per-site orbital projections, following structural relaxation.
Work at $K=(1/3,1/3,0)$ in reciprocal fractional coordinates.

Use band occupations, rather than the recorded Fermi level, to distinguish
occupied and empty states.

## Definitions

A band's host fraction is the fraction of its total site-projection weight
on a layer. Its host is the layer with the larger fraction. A layer's
conduction and valence edges are the lowest conduction and highest valence
states hosted by it, respectively.

## What to report

Let $E_c(L)$ and $E_v(L)$ denote the K-point edges hosted by layer $L$.
Report:

- $\mathrm{CBO}=E_c(\mathrm{WSe_2})-E_c(\mathrm{WS_2})$ and
  $\mathrm{VBO}=E_v(\mathrm{WSe_2})-E_v(\mathrm{WS_2})$, in meV;
- the direct gap at K over all bands, in eV;
- the minimum host fraction among the four layer-resolved edge states;
- the type-I/type-II alignment and the layers hosting the overall conduction
  minimum and valence maximum at K.

## Output format

Write `answer.json` in the workspace root with exactly these fields (numbers where
named, strings for the labels; units as named):

```json
{
  "CBO_meV": <float>,
  "VBO_meV": <float>,
  "direct_gap_at_K_eV": <float>,
  "min_host_f": <float>,
  "alignment": <string: one of "type-I", "type-II">,
  "cbm_layer": <string: one of "WS2", "WSe2">,
  "vbm_layer": <string: one of "WS2", "WSe2">
}
```
