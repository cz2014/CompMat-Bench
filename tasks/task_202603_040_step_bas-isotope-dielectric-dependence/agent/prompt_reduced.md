# Task: isotope-composition dependence of the infrared dielectric function of boron arsenide

Compute the isotope-composition dependence of BAs's isotropic harmonic
infrared dielectric response from `ifc_harmonic_512.npz`. Use its documented
atom order, force-constant layout and units; add no nonanalytic dipole term.

## Model

Born charges are isotropic: B -0.46 e and As +0.46 e.
$\varepsilon_\infty=9.83$. Masses are $^{10}$B $10.01293695\,\mathrm{u}$,
$^{11}$B $11.00930536\,\mathrm{u}$ and As $74.921595\,\mathrm{u}$. Every mode has the same
$300\,\mathrm{K}$ Lorentz linewidth $\gamma=0.29$ cm$^{-1}$, specified as FWHM.

At composition $n_{10}$, exactly $n_{10}$ of the 256 B sites are $^{10}$B,
chosen uniformly without replacement; the other B sites are $^{11}$B and As
is monoisotopic. Average the dielectric response over 256 independent
configurations drawn with your own random generator at each composition.
$x=n_{10}/256$.

For each configuration use the macroscopic harmonic Born-charge dielectric
response with background $\varepsilon_\infty$. The damping denominator of
a mode of angular frequency $\omega_s$ is
$\omega_s^2-\omega^2-i\omega\gamma$, with $\omega$ and $\gamma$ in SI
angular-frequency units. Only positive-frequency, nontranslational modes
contribute. Use the isotropic response, one third of the dielectric tensor's trace.
Let $Z^*_{s,\alpha}$ denote the mass-normalized mode effective charge.
Use a frequency step no coarser than $0.01\,\mathrm{cm}^{-1}$ over $[680,740]\,\mathrm{cm}^{-1}$.
Let $m(n_{10})$ be the minimum real part of the ensemble-mean dielectric
function over this interval.

Report:

- at $n_{10}=0,256$, $m(n_{10})$ and the number of modes whose total
  strength $\sum_\alpha|Z^*_{s,\alpha}|^2$ exceeds $10^{-6}$ times the
  largest mode strength in that configuration;
- $m(n_{10})$ at $n_{10}=13,51,128,251$;
- the smallest $n_{10}$ in [42,51] with $m(n_{10})>0$, and the smallest
  in [243,252] with $m(n_{10})<0$; use -1 if no composition in a bracket
  qualifies. Each evaluated composition uses the ensemble rule above.

No first-principles calculation is required; Python with numpy and scipy
is available.

## Output format

Write `answer.json` in the workspace root with exactly these fields:

```json
{
  "n_active_modes_n10_0": <int>,
  "n_active_modes_n10_256": <int>,
  "min_eps1_n10_0": <float>,
  "min_eps1_n10_13": <float>,
  "min_eps1_n10_51": <float>,
  "min_eps1_n10_128": <float>,
  "min_eps1_n10_251": <float>,
  "min_eps1_n10_256": <float>,
  "n10_first_positive_lower_bracket": <int>,
  "n10_first_negative_upper_bracket": <int>
}
```
