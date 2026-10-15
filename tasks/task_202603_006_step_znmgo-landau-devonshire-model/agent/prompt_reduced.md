# Task: sixth-order Landau-Devonshire model of stress-free wurtzite Zn$_{1-x}$Mg$_x$O

Fit the room-temperature stress-free bulk model of wurtzite
Zn$_{1-x}$Mg$_x$O over $0\le x\le1$, relative to the nonpolar
layered-hexagonal phase:
$$\Delta f(x,P)=\tfrac12\alpha_0(x-x_c)P^2+\tfrac14\gamma^*P^4+
\tfrac16\omega P^6.$$
$P$ is the $c$-axis polarization magnitude; $\gamma^*$ is the stress-free
fourth-order coefficient. $\gamma^*$ and $\omega$ are composition
independent. $P_{sp}(x)$ is the nonzero global minimizer for $P\ge0$;
$\Delta f(x)$ is the free-energy density at that minimum.

Use all five prescribed polarizations in `psp_vs_x.csv`. At $x=0$, impose
exactly $P_{sp}(0)$ from the file, total static dielectric constant
$\kappa(0)=8.630$ and $\Delta f(0)=-1.683\,\mathrm{GJ\,m^{-3}}$.
The total static relative permittivity $\kappa(x)$ is the sum of the
dimensionless ionic susceptibility of this free energy at $P_{sp}(x)$ and
electronic background $n^2$, with refractive index $n=2$ independent of
composition.

The three pure-ZnO constraints fix $\alpha(0)$, $\gamma^*$ and $\omega$.
Holding them fixed, fit only $\alpha_0$ by equally weighted squared
residuals in $P_{sp}$ over the five compositions, with $x_c$ constrained
by $\alpha(0)=-\alpha_0x_c$. Report the four coefficients and the fitted
model's $\kappa(1)$ and $\Delta f(1)$.

## Output

Write `answer.json` in the workspace root with exactly these fields:

```json
{
  "alpha0_GJ_m_per_C2": <float>,
  "gamma_star_GJ_m5_per_C4": <float>,
  "omega_GJ_m9_per_C6": <float>,
  "xc": <float>,
  "kappa_x1": <float>,
  "df_x1_GJ_per_m3": <float>
}
```

Use $P$ in $\mathrm{C\,m^{-2}}$ and the coefficient units in the field
names; $x_c$ and $\kappa$ are dimensionless. The free-energy difference is
wurtzite minus layered-hexagonal reference. Numpy and scipy are available;
no electronic-structure runs are required.
