# Task: coupled surface phonon polariton and evanescent energy transmission across a 10 nm vacuum gap in isotope-disordered boron arsenide

Cubic boron arsenide at natural isotope abundance carries a large boron mass
variance. Its infrared lattice response can be described in two ways: a
quasiparticle picture, in which the crystal has one damped zone-centre optical
oscillator at the mass-averaged frequency, and an ensemble-averaged
(nonperturbative) treatment, in which the exact response of many random isotope
configurations of a large supercell is averaged. The question is whether the two
descriptions agree on the near-field electrodynamics of a narrow vacuum gap
between two half-spaces of the crystal: whether each description supports a
coupled gap pole with the smallest attenuation, at what in-plane wave vector
it sits, and how much evanescent energy transmission the gap carries at large
in-plane wave vectors.

## What you are given

- `eps_nat_bas.npz` -- the dielectric spectra of natural-abundance BAs at
  300 K: `eps_lorentz` is the quasiparticle description and `eps_disordered`
  is the ensemble-averaged description. Grid, units and array conventions
  are documented in the archive.

  Both spectra are isotropic scalar dielectric functions,
  $\varepsilon = \varepsilon' + i\varepsilon''$ with $\varepsilon'' > 0$,
  for a time dependence $e^{-i\omega t}$. Use them exactly as given: do not
  refit, smooth, rescale or extend them. Every frequency below refers to the
  file's own grid.

No simulation is required. The environment provides Python with numpy and
scipy; no other software is needed.

## Geometry (fixed for this task)

Two semi-infinite half-spaces of the same material, both described by the same
$\varepsilon(\omega)$, face each other across a vacuum gap of width
$d = 10$ nm ($\varepsilon_{\mathrm{gap}} = 1$). $\beta$ is the in-plane wave
vector, $k_0 = \omega / c$ the vacuum wave number, and in medium $j$ the
normal wave-vector component is

$$
k_{zj} = \sqrt{\varepsilon_j k_0^2 - \beta^2}, \qquad \operatorname{Im} k_{zj} \ge 0 ,
$$

the same branch for the vacuum ($\varepsilon_0 = 1$) and for the material.
Treat the two spectra as two independent cases of this one geometry.

## What to determine, for each of the two spectra

(1) **The $\varepsilon' = -1$ crossings of the Lorentz spectrum**: every
    crossing frequency, located by linear interpolation between neighbouring
    grid points, in ascending order. For the disordered spectrum, report the
    **number of grid points** at which $\varepsilon' \le -1$.

(2) **The dispersion of the least-attenuated coupled gap pole**, p polarization, on every grid
    frequency. A bound coupled surface mode of the gap is a pole of the gap's
    p-polarized multiple-reflection response. For this symmetric gap take
    the branch on which the two surface waves are in phase: a complex
    $\beta$ with $\operatorname{Re}\beta > k_0$ at which

    $$
    r_p(\beta) \, e^{i k_{z0} d} = 1 ,
    $$

    where $r_p$ is the vacuum-to-material p-polarized Fresnel reflection
    coefficient and $k_{z0}$ the vacuum normal component (the out-of-phase
    factor, $r_p e^{i k_{z0} d} = -1$, is not asked for). Use the spatial
    convention $e^{i\beta x-i\omega t}$. Among roots of the stated +1 factor
    with $\operatorname{Re}\beta > k_0$, $\operatorname{Im}\beta \ge 0$,
    and decay into both half-spaces under the specified normal branches,
    select the root with the smallest $\operatorname{Im}\beta$. If distinct
    roots share that minimum, select the smaller $\operatorname{Re}\beta$.
    Report $\operatorname{Re}\beta/k_0$. Apply this choice at every supplied
    frequency, without a cutoff on $\operatorname{Re}\varepsilon$. Where no
    admissible root exists, report $\sqrt{\max(\varepsilon',0)}$ instead.

(3) **The peaks of the reported curves** of item (2), including any fallback
    values: the maximum $\operatorname{Re}\beta/k_0$ for each spectrum, and
    the frequency of that maximum for the Lorentz spectrum.

(4) **The evanescent energy transmission coefficient** $\xi_p(\omega, \beta)$
    of the gap for p polarization and $\beta > k_0$, in the standard
    two-half-space fluctuational-electrodynamics convention. Evaluate it
    on a map whose frequency axis is the file grid or a subsample of it with
    step no coarser than 0.1 cm$^{-1}$, and whose $\beta / k_0$ axis is
    logarithmically spaced from 1 to at least 300 with a ratio of at most 1.01
    between neighbouring points. Report the **maximum of $\xi_p$ over the part
    of the map with $\beta / k_0 > 50$**.

## Output format

Write a single `answer.json` in the workspace root with exactly this structure:

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

- `crossings_lorentz_cm1`: item (1), ascending, in cm$^{-1}$; an empty list if
  there is none. `n_grid_points_eps_re_le_minus1_disordered`: item (1), the count.
- `beta_over_k0_*`: item (2), one value per grid frequency, in the order of
  `wavenumber_cm1` (the same length as that array), dimensionless.
- `peak_nu_lorentz_cm1`, `peak_beta_over_k0_*`: item (3).
- `xi_p_max_above_50_*`: item (4), dimensionless.

All values must be computed from the staged file; do not guess or hardcode.
