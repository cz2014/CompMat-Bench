# Task: composition-tuned near-field radiative heat flux across a 10 nm vacuum gap between isotope-engineered boron arsenide half-spaces

Cubic boron arsenide can be grown with any ratio of the two stable boron
isotopes. Mass disorder on the boron sublattice reshapes the infrared lattice
response of the crystal, so two half-spaces of the material facing each other
across a narrow vacuum gap exchange near-field thermal radiation at a rate that
depends on the isotope composition of each side. The question is how the total
and spectral near-field heat flux across a 10 nm gap varies with composition
when both sides share the same composition, and how a fixed receiving side
tunes the flux when only the emitting side's composition is varied.

## What you are given

- `eps_bas_isotope_series.npz` -- the composition-dependent dielectric
  spectra at the two temperatures. The physical data mapping is:
  - `wavenumber_cm1`: the frequency grid $\nu$ in cm$^{-1}$, strictly
    increasing and nonuniform;
  - `n10`: for each composition, the number of $^{10}$B atoms among the 256
    boron sites of the supercell the spectra were computed on; `x`: the same
    as a fraction, $x = n_{10}/256$;
  - `eps_cold_300K`: for each composition (first index, in the order of `n10`),
    the dielectric function of the crystal at 300 K on the grid;
  - `eps_hot_500K`: the same at 500 K;
  - `eps_inf`: the high-frequency dielectric constant.

  The spectra are isotropic scalar dielectric functions,
  $\varepsilon = \varepsilon' + i\varepsilon''$ with $\varepsilon'' > 0$, for
  a time dependence $e^{-i\omega t}$. Use them exactly as given: do not refit,
  smooth, rescale or extend them. Natural isotope abundance is the composition
  with $n_{10} = 51$; the two ends, $n_{10} = 0$ and $n_{10} = 256$, are the
  isotopically pure $^{11}$B and $^{10}$B crystals.

No simulation is required. The environment provides Python with numpy and
scipy; no other software is needed.

## Geometry and thermal state (fixed for this task)

Two semi-infinite half-spaces face each other across a vacuum gap of width
$d = 10$ nm ($\varepsilon_{\mathrm{gap}} = 1$). One half-space, the hot side,
is held at $T_{\mathrm{hot}} = 500$ K and is described by its composition's
`eps_hot_500K` spectrum; the other, the cold side, is held at
$T_{\mathrm{cold}} = 300$ K and is described by its composition's
`eps_cold_300K` spectrum. $\beta$ is the in-plane wave vector,
$k_0 = \omega / c$ the vacuum wave number, and in medium $j$ the normal
wave-vector component is

$$
k_{zj} = \sqrt{\varepsilon_j k_0^2 - \beta^2}, \qquad \operatorname{Im} k_{zj} \ge 0 ,
$$

the same branch in the vacuum and in the two materials.

## The flux (fixed for this task)

Compute the net spectral radiative heat flux $q(\nu)$ from hot to cold per unit
area and per unit wavenumber, using standard fluctuational electrodynamics
for the two half-spaces. Include both polarizations and both propagating
and evanescent contributions. The total flux $Q$ is integrated over the whole staged
frequency grid. Choose the in-plane-wave-vector integration scheme and range
to converge the total flux.

## What to compute

Three series of hot/cold pairs, each over all compositions $i$ of the file:

(1) **symmetric**: hot side composition $i$, cold side composition $i$ (the
    hot side still carries that composition's 500 K spectrum and the cold
    side its 300 K spectrum);
(2) **cold pure $^{11}$B**: hot side composition $i$, cold side $n_{10} = 0$;
(3) **cold natural**: hot side composition $i$, cold side $n_{10} = 51$;

and (4) the **nonresonant baseline**: the same calculation with
$\varepsilon = \varepsilon_\infty$ (the file's `eps_inf`) on both sides at the
same temperatures.

From them:

- the total flux $Q$ of every pair in the three series;
- the composition $n_{10}$ of smallest $Q$ in the symmetric series and the
  composition of largest $Q$ in the cold-natural series;
- in the symmetric series, the frequency of the maximum of $q(\nu)$ over the
  whole grid for the pure $^{11}$B pair and for the pure $^{10}$B pair;
- the ratio of the maximum of $q(\nu)$ for the pure $^{11}$B symmetric pair to
  the maximum of $q(\nu)$ for the natural-abundance symmetric pair;
- for the pair with hot side $n_{10} = 256$ and cold side $n_{10} = 0$, the
  minimum over $680 \le \nu \le 740$ cm$^{-1}$ of the ratio
  $q(\nu) / q_{\mathrm{baseline}}(\nu)$.

## Output format

Write a single `answer.json` in the workspace root with exactly this structure:

```json
{
  "Q_symmetric_W_m2": [<float>, ...],
  "Q_cold_pure11B_W_m2": [<float>, ...],
  "Q_cold_natural_W_m2": [<float>, ...],
  "n10_min_Q_symmetric": <int>,
  "n10_max_Q_cold_natural": <int>,
  "peak_nu_symmetric_pure11B_cm1": <float>,
  "peak_nu_symmetric_pure10B_cm1": <float>,
  "peak_q_ratio_pure11B_over_natural": <float>,
  "min_q_ratio_hot10B_cold11B_over_baseline": <float>
}
```

- `Q_*_W_m2`: one value per composition, in the order of the file's `n10`
  array (the same length as that array), in W m$^{-2}$.
- `n10_min_Q_symmetric`, `n10_max_Q_cold_natural`: the `n10` values at the
  extremes named above.
- `peak_nu_*_cm1`: frequencies in cm$^{-1}$ on the
  file grid.
- `peak_q_ratio_pure11B_over_natural`, `min_q_ratio_hot10B_cold11B_over_baseline`:
  dimensionless.

The same hot/cold pair appears in more than one series. Its repeated total
flux values must agree within a relative difference of $10^{-6}$. Report
total fluxes with at least seven significant figures where practical to avoid
inconsistent rounding; this is a reporting recommendation, not a claim of
seven-digit integration accuracy.

All values must be computed from the staged file; do not guess or hardcode.
