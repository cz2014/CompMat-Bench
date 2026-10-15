# Task: sixth-order Landau-Devonshire model of stress-free wurtzite Zn$_{1-x}$Mg$_x$O

Wurtzite Zn$_{1-x}$Mg$_x$O is a polar oxide whose spontaneous polarization
along the $c$ axis falls as Mg replaces Zn on the cation sublattice. The
question is what sixth-order Landau-Devonshire free-energy density, written
with the nonpolar layered-hexagonal structure as its reference state,
describes the stress-free bulk single crystal at room temperature over the
whole composition range $0 \le x \le 1$, and what static dielectric constant
and wurtzite-versus-reference energy difference that model implies at the
Mg-rich end.

## What you are given

- `psp_vs_x.csv` -- the prescribed polarization inputs for the effective
  model, based on Berry-phase DFT. Units and composition conventions are in
  the file header.
- For pure ZnO ($x = 0$), two literature values: the total static dielectric
  constant $\kappa(0) = 8.630$, and the free-energy density of the wurtzite
  phase relative to the layered-hexagonal reference,
  $\Delta f(0) = -1.683\ \mathrm{GJ/m^3}$ (negative: wurtzite is the lower).
- The refractive index of pure ZnO, $n = 2$, taken as composition
  independent.

No electronic-structure calculation is required. The environment provides
Python with numpy and scipy.

## The model (fixed for this task)

The polarization is along the $c$ axis only; call its magnitude $P$. Under
stress-free conditions the free-energy density relative to the
layered-hexagonal reference is

$$
\Delta f(x, P) = \tfrac{1}{2}\,\alpha(x)\,P^{2} + \tfrac{1}{4}\,\gamma^{*}\,P^{4}
+ \tfrac{1}{6}\,\omega\,P^{6},
\qquad \alpha(x) = \alpha_0\,(x - x_c),
$$

with $\gamma^{*}$ and $\omega$ composition independent. $\gamma^{*}$ is the
stress-free fourth-order coefficient (the strain-coupling renormalization is
already inside it); it is determined directly, and no electrostrictive
reduction has to be evaluated. Temperature dependence is not considered.
Units: $P$ in C/m$^2$, $\Delta f$ in GJ/m$^3$, $\alpha_0$ in GJ m C$^{-2}$,
$\gamma^{*}$ in GJ m$^{5}$ C$^{-4}$, $\omega$ in GJ m$^{9}$ C$^{-6}$, $x_c$
dimensionless.

The spontaneous polarization $P_{sp}(x)$ is the nonzero polarization at
which $\Delta f(x, P)$ attains its minimum over $P \ge 0$, and $\Delta f(x)$
below means $\Delta f$ evaluated at that minimum. The total static
dielectric constant is the sum of an ionic part, from the curvature of the
free energy at the spontaneous state, and an electronic part, $n^{2}$:

$$
\kappa(x) = n^{2} + \left[\varepsilon_0\,
\frac{\partial^{2} \Delta f}{\partial P^{2}}\Big|_{P_{sp}(x)}\right]^{-1},
$$

with $\varepsilon_0$ the vacuum permittivity in SI units.

## Coefficient constraints and fit (fixed for this task)

The three pure-ZnO inputs, $P_{sp}(0)$, $\kappa(0)$ and $\Delta f(0)$,
are exact constraints fixing $\alpha(0)$, $\gamma^{*}$ and $\omega$.
With those quantities held fixed, $\alpha_0$ is the sole fit parameter:
minimize equally weighted squared residuals in $P_{sp}$ over all five
compositions. $x_c$ is constrained by the stated composition law and fixed $\alpha(0)$;
no additional joint fit or subsequent refit is part of the task.

## What to determine

- The four coefficients $\alpha_0$, $\gamma^{*}$, $\omega$ and $x_c$.
- The model's total static dielectric constant $\kappa$ and its
  $\Delta f$ at $x = 1$, evaluated from the fitted coefficients (not read
  from the data file).

## Output format

Write a single `answer.json` in the workspace root with exactly these fields:

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

- `alpha0_GJ_m_per_C2`, `gamma_star_GJ_m5_per_C4`, `omega_GJ_m9_per_C6`,
  `xc`: the coefficients in the units stated above.
- `kappa_x1`: the total static dielectric constant of the model at $x = 1$
  (dimensionless).
- `df_x1_GJ_per_m3`: $\Delta f$ of the model at $x = 1$, in GJ/m$^3$, with
  the sign convention stated above.

All values must be computed from the staged data and the given inputs; do
not guess or hardcode.
