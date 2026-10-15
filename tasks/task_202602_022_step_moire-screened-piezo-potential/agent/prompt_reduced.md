# Task: screen a moire piezoelectric charge and compute its band-edge potential

Compute the electron band-edge potential energy from the layer-resolved
piezoelectric charge in `piezocharge.csv`. Its header specifies the cell,
position units and charge-density units; rows give sites and locale labels.
Each layer is periodic and cell-neutral.

Treat each layer as an independent 2D sheet in vacuum with Rytova-Keldysh
screening. Its in-plane screening factor is $1+r_0|\mathbf q|$, where

$$r_0=\frac{d_0(\epsilon_\parallel-1)}{2},$$

$d_0=6.3$ Å and $\epsilon_\parallel=14.4$ for WS2 and 15.9 for WSe2.
The cell-average potential is zero. The overall sign of the input charge
is convention-dependent; report all results in one consistent convention.

For each layer report the electron potential-energy peak-to-peak amplitude
and its mean over all sites of each locale ($R^h_h$, $R^X_h$, $R^M_h$), in meV.
Also report the WS2/WSe2 peak-to-peak amplitude ratio.

## Output format

Write `answer.json` in the workspace root with exactly these fields:

```json
{
  "WS2_pp_meV": <float>,
  "WS2_vrp_Rhh_meV": <float>,
  "WS2_vrp_RXh_meV": <float>,
  "WS2_vrp_RMh_meV": <float>,
  "WSe2_pp_meV": <float>,
  "WSe2_vrp_Rhh_meV": <float>,
  "WSe2_vrp_RXh_meV": <float>,
  "WSe2_vrp_RMh_meV": <float>,
  "ratio_pp_WS2_over_WSe2": <float>
}
```
