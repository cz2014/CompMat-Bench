# Task: decompose the potential step of an undulated bilayer

Decompose the local potential step of the undulated hBN bilayer into its
monolayer and sliding contributions, and compare the analytic profiles.
`dft/mono_I/`, `dft/mono_II/` and `dft/bilayer/` contain `POSCAR` and
`LOCPOT.gz` for the upper layer, lower layer and rigidly combined bilayer.
Use $\phi=-\mathrm{LOCPOT}$.

The continuum surfaces are independent of $x$. With cell lengths $L$ along
$y$ and $L_z$ along $z$, the bilayer mid-surface is
$$z_c(y)=L_z/2+9\,\text{\AA}\sin(2\pi y/L).$$
The layer-I and layer-II surfaces are its parallel offsets by
$+1.625\,\text{\AA}$ and $-1.625\,\text{\AA}$ along the upward unit normal.
For a profile at Cartesian $y$, use the point on that structure's own
specified continuum curve with this $y$, its upward normal and its local tangent.

At normal offset $s$, average the potential over $x$ and uniformly over a
$4\,\text{\AA}$ straight tangent segment centered on the normal line.
The local step is the top-minus-bottom intercept difference at $s=0$
of linear fits on $4.5\le s\le9\,\text{\AA}$ and
$-9\le s\le-4.5\,\text{\AA}$.

The analytic per-layer flexoelectric step is $V_0\kappa(y)$, with
$V_0=2.420\,\mathrm{V}\,\text{\AA}$ and the signed curvature of $z_c$, positive
where $z_c''>0$. The analytic sliding step uses the flat-bilayer map
`upstream/L1_dVsl_vs_nu_ac.csv` at the local sliding ratios in
`upstream/L2_nu_ac_vs_y.csv`. Use these inputs as supplied.

At $y_j=jL/24$, $j=0,\ldots,23$, report the bilayer step `dVn_V`,
monolayer steps `dVfl_I_V` and `dVfl_II_V`, their sliding residual
`dVsl_V` (bilayer minus both monolayers), and the analytic profiles
`dVfl_an_V` and `dVsl_an_V`, all in volts. Report the residual's maximum,
minimum and cyclic sign-change count over these samples, discarding
samples with absolute residual below $0.02\,\mathrm V$ for the count.

## Output

Write `answer.json` in the workspace root with these fields:

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

Each array contains 24 numbers in increasing $j$; `y_over_L` is $j/24$.
