# Task: screen a moire piezoelectric charge and compute its band-edge potential

Compute the electron band-edge potential energy produced by the supplied
layer-resolved piezoelectric charge in a reconstructed WS2/WSe2 moire,
using the independent-sheet screening model below.

## What you are given

`piezocharge.csv` contains the layer-resolved piezoelectric bound charge of
the relaxed moire supercell, with one row per lattice site for both layers.
It includes site positions and registry labels. Positions and cell vectors
use Å; `rho_Cm2` is in
$\mathrm{C/m^2}$. The cell vectors are recorded in the file header. Each
layer's charge is periodic and cell-neutral.

## The screening and electrostatics model (given -- this is the model's setting)

Each monolayer is a polarizable two-dimensional sheet with in-plane
polarizability

$$
\alpha_{\mathrm{2D}}=\frac{d_0(\epsilon_\parallel-1)}{4\pi},
$$

with the static in-plane dielectric constant $\epsilon_\parallel=14.4$ (WS2) / $15.9$
(WSe2) and the layer thickness $d_0=6.3\,\text{Å}$. Solve for the total
in-plane electrostatic potential $\phi$ produced by the bound charge together
with its induced screening charge, treating each layer on its own (each
screened by its own $\epsilon_\parallel$), with the 2D-sheet screened Coulomb kernel
(Rytova-Keldysh form). In SI units, per in-plane Fourier component $\mathbf q$ of the
periodic charge density,

$$
\phi(\mathbf q)=\frac{\rho(\mathbf q)}{2\epsilon_0|\mathbf q|(1+r_0|\mathbf q|)},\qquad
r_0=2\pi\alpha_{\mathrm{2D}}=\frac{d_0(\epsilon_\parallel-1)}{2},
$$

with $\epsilon_0$ the vacuum permittivity. Drop the $\mathbf q=0$ component (the charge is
cell-neutral; only potential variations matter). Convert the potential to the
band-edge potential energy seen by an electron, $V^{\mathrm{RP}}=-e\phi$, and report it
in meV. The overall sign of $V^{\mathrm{RP}}$ follows the sign convention of the input
bound charge, which is fixed only up to an overall sign; report your results in
your own consistent convention.

## What to report, per layer

Report the potential-energy peak-to-peak amplitude and its mean over all
sites of each locale ($R^h_h$, $R^X_h$, $R^M_h$), in meV. Also report the
WS2/WSe2 peak-to-peak amplitude ratio.

## Output format

Write `answer.json` in the workspace root with exactly these fields (all numbers,
units as named):

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
