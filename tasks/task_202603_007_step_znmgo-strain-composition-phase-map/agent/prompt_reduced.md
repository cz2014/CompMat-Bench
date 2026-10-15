# Task: strain-composition phase map of a coherently strained Zn$_{1-x}$Mg$_x$O film

Determine equal-composition crossings of coherently strained wurtzite and
rocksalt Zn$_{1-x}$Mg$_x$O at $298.15\,\mathrm K$ using
`g_bulk_298K.csv` and `dft_properties_vs_x.csv`. Their headers define units,
composition, molar basis and crystal axes. Smoothly interpolate the
stress-free lattice and elastic properties between tabulated compositions.

Wurtzite is $(0001)$-oriented and rocksalt $(111)$-oriented, with
$[111]_R\parallel[0001]_W$ and $[1\bar10]_R\parallel[11\bar20]_W$.
In the film frame, axes 1 and 2 are in-plane and axis 3 is the growth axis.
Impose $\varepsilon_{11}=\varepsilon_{22}=\varepsilon_s$,
$\varepsilon_{12}=0$ and $\sigma_{i3}=\sigma_{3i}=0$; the remaining strains
are free.

The table gives stiffness coefficients in the crystal axes. Use harmonic
elastic energies with each phase's stiffness tensor expressed in the film
frame and additive eigenstrain kinematics in the common stress-free
rocksalt reference:
$$\boldsymbol\varepsilon_{\mathrm{el},W}=
\boldsymbol\varepsilon-\boldsymbol\varepsilon_{0,W},\qquad
\boldsymbol\varepsilon_{\mathrm{el},R}=\boldsymbol\varepsilon.$$
$\varepsilon_s=0$ matches the stress-free rocksalt $(111)$ cation net at
the same composition. Use the Green-Lagrange strain measure for both
substrate strain and the diagonal rocksalt-to-wurtzite transformation,
whose stretches are
$$\lambda_a=\sqrt2\,a_W/a_R,\qquad
\lambda_c=\sqrt3\,c_W/(2a_R).$$
The in-plane stretches are $\lambda_a$, the growth-axis stretch is
$\lambda_c$, and there is no transformation shear.

Each phase's total molar free energy is its tabulated bulk Gibbs energy
plus its elastic energy density times its own stress-free molar volume,
on the table's molar basis. Define the boundary by equality of these totals
at the same composition.

Report the in-plane transformation strain at $x=0$; the rocksalt $(111)$
biaxial modulus at $x=0$, defined by elastic energy density
$M_R\varepsilon_s^2$ under these boundary conditions; boundary compositions
at $\varepsilon_s=0,+0.02,+0.05$; and the compressive substrate strain at
which the boundary reaches $x=0$. Tensile strain is positive.

## Output

Write `answer.json` in the workspace root with exactly these fields:

```json
{
  "eps0_inplane_x0": <float>,
  "M_R111_x0_GPa": <float>,
  "x_boundary_eps_0": <float>,
  "x_boundary_eps_p0p02": <float>,
  "x_boundary_eps_p0p05": <float>,
  "eps_s_boundary_x0": <float>
}
```

Strains and compositions are dimensionless; $M_R$ is in GPa. Numpy and scipy
are available; use the supplied tables without new electronic-structure
or thermodynamic calculations.
