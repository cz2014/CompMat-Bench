# Task: read the band offsets and layer character at K of a heterobilayer

From `vasprun.xml.gz`, the relaxed WS2/WSe2 heterobilayer's SOC band run
with site projections, determine the following at $K=(1/3,1/3,0)$ in
reciprocal fractional coordinates.

A band's host fraction is the fraction of its total site-projection weight
on a layer; its host is the layer with the larger fraction. Layer $L$'s
K-point edges $E_c(L)$ and $E_v(L)$ are its lowest hosted conduction state
and highest hosted valence state, respectively.

Report:

- $\mathrm{CBO}=E_c(\mathrm{WSe_2})-E_c(\mathrm{WS_2})$ and
  $\mathrm{VBO}=E_v(\mathrm{WSe_2})-E_v(\mathrm{WS_2})$, in meV;
- the direct gap at K over all bands, in eV;
- the minimum host fraction among the four layer-resolved edge states;
- the type-I/type-II alignment and the layers hosting the overall
  conduction minimum and valence maximum at K.

## Output format

Write `answer.json` in the workspace root with exactly these fields:

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
