# Task: composition-tuned near-field radiative heat flux across a 10 nm vacuum gap between isotope-engineered boron arsenide half-spaces

Compute net radiative heat flux between BAs half-spaces using
`eps_bas_isotope_series.npz`. Its metadata documents the nonuniform
wavenumber grid, isotope compositions, and scalar dielectric spectra at
$300\,\mathrm{K}$ (`eps_cold_300K`) and $500\,\mathrm{K}$ (`eps_hot_500K`). Use the data as given,
without refitting, smoothing, rescaling or extending it.
The spectra use $e^{-i\omega t}$ with positive imaginary part.

$n_{10}$ counts $^{10}$B atoms among 256 B sites. Natural abundance is $n_{10}=51$;
0 and 256 are pure $^{11}$B and $^{10}$B. Two semi-infinite half-spaces face each
other across $10\,\mathrm{nm}$ of vacuum. The hot side is at $500\,\mathrm{K}$ with its $500\,\mathrm{K}$
spectrum, the cold side at $300\,\mathrm{K}$ with its $300\,\mathrm{K}$ spectrum.
Choose normal wave-vector branches with nonnegative imaginary part in
vacuum and both materials.

## Flux and cases

Use standard two-half-space fluctuational electrodynamics, including both
polarizations and both propagating and evanescent contributions.
$q(\nu)$ is net hot-to-cold flux per unit area and per unit wavenumber.
Integrate total flux $Q$ over the entire supplied frequency grid with
converged in-plane-wave-vector integration.

For every supplied composition i evaluate these hot/cold pairs:

- `symmetric`: i/i;
- `cold_pure11B`: i/0;
- `cold_natural`: i/51.

The nonresonant baseline uses the file's $\varepsilon_\infty$ on both sides
at the same temperatures. Report:

- $Q$ for every pair, in $\mathrm{W\,m^{-2}}$, with each array in the file's `n10` order;
- the $n_{10}$ minimizing symmetric $Q$ and maximizing cold-natural $Q$;
- the frequency of peak $q(\nu)$ over the entire supplied grid for the symmetric pure $^{11}$B and pure
  $^{10}$B pairs, in cm$^{-1}$;
- the ratio of peak $q(\nu)$ for symmetric pure $^{11}$B to symmetric natural BAs;
- for hot $^{10}$B/cold $^{11}$B, the minimum of $q(\nu)/q_{baseline}(\nu)$ over
  $[680,740]\,\mathrm{cm}^{-1}$.

Repeated occurrences of the same pair must have total fluxes agreeing
within relative difference $10^{-6}$. No simulation is required;
Python with numpy and scipy is available.

## Output format

Write `answer.json` in the workspace root with exactly these fields:

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

Composition fields are `n10` values; the two ratios are dimensionless.
