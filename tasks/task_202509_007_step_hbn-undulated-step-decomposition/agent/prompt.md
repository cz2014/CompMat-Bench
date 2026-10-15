# Task: decompose the potential step of an undulated bilayer

Separate the flexoelectric and sliding contributions to an undulated hBN
bilayer's potential step, and compare them with the upstream analytic model.

`dft/mono_I/`, `dft/mono_II/` and `dft/bilayer/` supply static calculations of
the upper monolayer, lower monolayer and their rigidly combined bilayer.
Each contains the gzip-compressed local potential `LOCPOT.gz` and structure
`POSCAR`. Use $\phi=-\mathrm{LOCPOT}$ for these DFT potentials.

The upstream inputs are `upstream/L1_dVsl_vs_nu_ac.csv` (the flat-bilayer
sliding map), `upstream/L2_nu_ac_vs_y.csv` (the local sliding ratio), and the
flexoelectric constant $V_0=2.420\ \mathrm{V}\,\text{\AA}$. Use these inputs as given.

## Observable convention

The continuum surfaces are independent of $x$. Let $L$ and $L_z$ be the cell
lengths along $y$ and $z$. Define the bilayer mid-surface in the $yz$ section by

$$z_c(y)=\frac{L_z}{2}+h\sin\!\left(\frac{2\pi y}{L}\right),
\qquad h=9\ \text{\AA}.$$

Thus the mid-surface crosses its mean height upward at $y=0$. The monolayer
continuum surfaces are parallel offsets of this curve by
$+d$ for I and $-d$ for II along its upward unit normal, with
$d=1.625\ \text{\AA}$. In parametric form,

$$\mathbf r_c(t)=(t,z_c(t)),\qquad
\mathbf r_{\mathrm I}(t)=\mathbf r_c(t)+d\hat{\mathbf n}_c(t),\qquad
\mathbf r_{\mathrm {II}}(t)=\mathbf r_c(t)-d\hat{\mathbf n}_c(t).$$

Use these continuum surfaces for the local potential observable, rather than
interpolating the atomic-scale height variations. For each structure's profile
at Cartesian coordinate $y$, use the point on its own continuum curve whose
$y$ coordinate equals that value, and its upward normal and local tangent.

At each normal offset $s$, use the $x$-averaged potential, uniformly averaged
over a $4\ \text{\AA}$ straight segment along that local tangent, centered
on the normal line. The local step is the top-minus-bottom intercept difference
at $s=0$ of linear fits over the vacuum intervals
$4.5\le s\le9\ \text{\AA}$ and $-9\le s\le-4.5\ \text{\AA}$.

For the analytic per-layer flexoelectric profile, compute the signed curvature
of the bilayer mid-surface cross-section $z_c(y)$, positive where $z_c''(y)>0$,
and use $V_0\kappa(y)$.
For the analytic sliding profile, evaluate the supplied flat-bilayer map at
the supplied local sliding ratio.

## Quantities and submission

Extract the bilayer and both monolayer DFT step profiles. The sliding
residual is the bilayer step after subtracting both monolayer contributions.
Report these four profiles and the two analytic profiles at $y_j=jL/24$,
$j=0,\ldots,23$, in volts. Report the maximum and minimum residual over these
samples and its cyclic number of sign changes after discarding samples
with absolute residual below $0.02\ \mathrm V$.

Write `answer.json` in the workspace root. Each array contains 24 numbers
ordered by $j$:

```json
{
  "y_over_L": [<float>, ...],
  "dVn_V": [<float>, ...],
  "dVfl_I_V": [<float>, ...],
  "dVfl_II_V": [<float>, ...],
  "dVfl_an_V": [<float>, ...],
  "dVsl_V": [<float>, ...],
  "dVsl_an_V": [<float>, ...],
  "dVsl_max_V": <float>,
  "dVsl_min_V": <float>,
  "sign_changes_per_period": <int>
}
```

`y_over_L` records $j/24$; `dVn_V` is the bilayer step and
`dVfl_I_V`/`dVfl_II_V` are the respective monolayer steps. `dVsl_V` is the
DFT sliding residual. The `_an_V` fields contain the analytic flexoelectric
and sliding profiles, respectively.
