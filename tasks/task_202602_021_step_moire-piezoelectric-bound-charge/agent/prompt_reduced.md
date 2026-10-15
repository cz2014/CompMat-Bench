# Task: layer-resolved piezoelectric bound charge of a reconstructed moire

Compute the layer-resolved piezoelectric bound charge of a reconstructed
R-type WS2/WSe2 moire, including its amplitude and registry dependence.

## The system and the input files

`lammps.dat_min` and `lammps.dat` are the relaxed and rigid structures,
respectively. Atom IDs correspond between files. In-plane distances and
displacements are periodic in the moire cell.

## The physics model

For each monolayer, the in-plane displacement field of that layer's W
sublattice (relaxed relative to rigid) carries the inhomogeneous strain; the
piezoelectric bound charge density $\rho$ it produces is what you compute, with
the coefficient magnitudes

$$
|e_{11}|=2.74\times10^{-10}\,\mathrm{C/m}\quad(\mathrm{WS_2}),\qquad
|e_{11}|=2.03\times10^{-10}\,\mathrm{C/m}\quad(\mathrm{WSe_2}).
$$

Each monolayer has 3m symmetry. In the data files' Cartesian axes, let
$u_{ij}$ denote Cartesian components of the infinitesimal strain tensor
of the layer's W-sublattice displacement field, and use

$$
\rho = e_{11}\left[\, 2\,\partial_x u_{xy}
      + \partial_y\left(u_{xx} - u_{yy}\right) \right].
$$

Report the signed charge number density $n$ in units of
$10^{12}\,\mathrm{cm}^{-2}$.

Either overall sign convention for $e_{11}$ is allowed, consistently for all
results; the Cartesian axes in the charge relation are fixed.

## The stacking registry

Use relaxed positions. A site's registry is the offset of its nearest
opposite-layer W pair, always WSe2 minus WS2, expressed in the WS2 monolayer
basis (moire vectors divided by 26), modulo 1.

The locale coordinates are $R^h_h=(0,0)$, $R^X_h=(1/3,2/3)$ and
$R^M_h=(2/3,1/3)$. Assign each W site to its nearest locale using Euclidean
distance on the fractional-coordinate unit torus, with each difference
component in $[-0.5,0.5)$.

## What to report, per layer

Report peak $|n|$, cell-average $n$, and signed mean $n$ within each locale,
all in $10^{12}\,\mathrm{cm}^{-2}$. Averages are over the layer's W sites.
Also report the WS2/WSe2 peak-amplitude ratio.

## Output format

Write `answer.json` in the workspace root with exactly these fields (all numbers,
units as named):

```json
{
  "WS2_peak_1e12_cm2": <float>,
  "WSe2_peak_1e12_cm2": <float>,
  "peak_ratio_WS2_over_WSe2": <float>,
  "WS2_net_1e12_cm2": <float>,
  "WSe2_net_1e12_cm2": <float>,
  "WS2_n_Rhh_1e12_cm2": <float>,
  "WS2_n_RXh_1e12_cm2": <float>,
  "WS2_n_RMh_1e12_cm2": <float>,
  "WSe2_n_Rhh_1e12_cm2": <float>,
  "WSe2_n_RXh_1e12_cm2": <float>,
  "WSe2_n_RMh_1e12_cm2": <float>
}
```
