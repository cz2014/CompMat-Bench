# Task: coupled surface phonon polariton and evanescent energy transmission across a 10 nm vacuum gap in isotope-disordered boron arsenide

Use `eps_nat_bas.npz` to compare natural-abundance BAs's `eps_lorentz`
and `eps_disordered` spectra at $300\,\mathrm{K}$ as two independent cases.
The archive documents the grid, units and array conventions.
The scalar dielectric functions use $e^{-i\omega t}$ with positive
imaginary part. Do not refit, smooth, rescale or extend them.

Two identical semi-infinite half-spaces face each other across a vacuum
gap of width $d=10$ nm. Let $k_0=\omega/c$ be the vacuum wave number, $\beta$ the in-plane
wave vector and $k_{zj}$ the normal component in each medium; choose
$\operatorname{Im}k_{zj}\ge0$ for both vacuum and material.

Report:

1. Every Lorentz-spectrum crossing of $\varepsilon'=-1$, using linear
   interpolation between neighbouring grid points, in ascending cm$^{-1}$
   (empty list if none); and the number of disordered-spectrum grid points
   with $\varepsilon'\le-1$.
2. At every supplied frequency, the p-polarized coupled-gap pole specified by
   $$r_p(\beta)e^{ik_{z0}d}=1,$$
   where $r_p$ is the vacuum-to-material Fresnel reflection coefficient.
   Use $e^{i\beta x-i\omega t}$. Among roots with
   $\operatorname{Re}\beta>k_0$, $\operatorname{Im}\beta\ge0$ and decay into
   both half-spaces under the normal branches above, choose the smallest
   $\operatorname{Im}\beta$, breaking ties by smaller $\operatorname{Re}\beta$.
   Report $\operatorname{Re}\beta/k_0$; if no admissible root exists, report
   $\sqrt{\max(\varepsilon',0)}$. Do this at every grid frequency without
   an $\varepsilon'$ cutoff.
3. Each reported curve's maximum, including fallback values, and the
   Lorentz curve's peak frequency in cm$^{-1}$.
4. The maximum p-polarized evanescent energy-transmission coefficient
   $\xi_p$ for $\beta/k_0>50$, in the standard two-half-space
   fluctuational-electrodynamics convention. Use the file grid or a
   subsample no coarser than $0.1\,\mathrm{cm}^{-1}$; sample $\beta/k_0$ logarithmically
   from 1 to at least 300, with adjacent ratio at most 1.01.

No simulation is required; Python with numpy and scipy is available.

## Output format

Write `answer.json` in the workspace root with exactly these fields:

```json
{
  "crossings_lorentz_cm1": [<float>, ...],
  "n_grid_points_eps_re_le_minus1_disordered": <int>,
  "beta_over_k0_lorentz": [<float>, ...],
  "beta_over_k0_disordered": [<float>, ...],
  "peak_nu_lorentz_cm1": <float>,
  "peak_beta_over_k0_lorentz": <float>,
  "peak_beta_over_k0_disordered": <float>,
  "xi_p_max_above_50_lorentz": <float>,
  "xi_p_max_above_50_disordered": <float>
}
```

Each `beta_over_k0_*` array has one dimensionless value per supplied
`wavenumber_cm1`, in that order. Peak beta ratios and transmission coefficients
are dimensionless.
