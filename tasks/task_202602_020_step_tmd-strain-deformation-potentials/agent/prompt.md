# Task: strain deformation potentials and the moire band-edge modulation

Quantify the strain response of isolated WS2/WSe2 band edges and their
resulting modulation across a reconstructed R-type moire. Upstream analysis
establishes type-II alignment with weak edge hybridization: use the WS2
conduction edge and WSe2 valence edge, both at $K=(1/3,1/3,0)$ in reciprocal
fractional coordinates.

## What you are given

- `runs/`: 14 spin-orbit static monolayer runs, each with `vasprun.xml.gz`.
  Directory names encode layer and strain in 0.1 percent units: `m15`,
  `m10`, `m05`, `p00`, `p05`, `p10`, `p15` correspond to
  $-1.5,-1.0,-0.5,0,+0.5,+1.0,+1.5\%$.
- `moire_fields.csv`: local biaxial strains in percent and fractional
  interlayer registries, carried on WS2 sites.

Registry coordinates use the WS2 monolayer basis. The locale coordinates
are $R^h_h=(0,0)$, $R^X_h=(1/3,2/3)$ and $R^M_h=(2/3,1/3)$.
Registry distance is Euclidean on the fractional-coordinate unit torus,
with difference components in $[-0.5,0.5)$.

## Reference and fitting conventions

Reference each band edge to the mean of the six lowest K-point eigenvalues
of its own run, not to vacuum. All shifts are relative to zero strain.

Deformation potentials are the linear least-squares slopes over the seven
strains, in meV/%. The modulation model for each edge is the quadratic
least-squares fit of its referenced shift in meV against strain in percent.

## What to compute

The diagonal-profile proxy follows the closed registry path

$$
R^h_h\rightarrow R^X_h\rightarrow R^M_h\rightarrow R^h_h.
$$

Its sampling convention is 9 equally spaced points per segment at fractions
$0,1/9,\ldots,8/9$, plus the final endpoint. At each point, the profile's
strain is the mean within registry radius **0.12**; its edge energy is the
appropriate layer's quadratic model evaluated at that mean strain.
Report each energy profile's peak-to-peak amplitude in meV.

The three conduction locale energies use the WS2 quadratic evaluated at the
mean WS2 strain within registry radius **0.10** of each locale. Report them
in meV and the absolute $R^X_h$–$R^M_h$ energy difference.

Report both deformation-potential slopes along with these modulation quantities.

## Output format

Write `answer.json` in the workspace root with these eight required fields (all numbers,
units as named):

```json
{
  "dp_WS2_conduction_meV_per_pct": <float>,
  "dp_WSe2_valence_meV_per_pct": <float>,
  "amp_conduction_meV": <float>,
  "amp_valence_meV": <float>,
  "Vc_Rhh_meV": <float>,
  "Vc_RXh_meV": <float>,
  "Vc_RMh_meV": <float>,
  "split_RXh_RMh_meV": <float>
}
```
