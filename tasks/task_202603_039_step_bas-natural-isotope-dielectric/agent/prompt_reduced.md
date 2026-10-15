# Task: infrared dielectric function of natural-abundance boron arsenide from an isotope-disordered supercell ensemble

Compare the natural-isotope ensemble infrared dielectric response of BAs
with its virtual-crystal baseline, using the supplied harmonic force constants.

## What you are given

- `ifc_harmonic_512.npz` -- the harmonic interatomic force constants of the
  BAs supercell. Use the atom order, index layout and units documented in
  the file. No nonanalytic long-range dipole term is to be added.

## Fixed inputs

- Born effective charges, isotropic: $Z_{\mathrm{B}} = -0.46\,e$,
  $Z_{\mathrm{As}} = +0.46\,e$.
- High-frequency dielectric constant $\varepsilon_\infty = 9.83$.
- Atomic masses: $^{10}$B 10.01293695 u, $^{11}$B 11.00930536 u,
  As 74.921595 u; the natural-abundance average boron mass is 10.811 u.
- Phonon linewidth at 300 K: $\gamma = 0.29$ cm$^{-1}$, a full width at half
  maximum, the same for every mode.
- Natural abundance: the $^{10}$B fraction is $x = 0.199$.

## The model (fixed for this task)

**Isotope configurations.** A configuration assigns exactly 51 of the 256 boron sites to
$^{10}$B, drawn uniformly at random without replacement, and the remaining
boron sites to $^{11}$B; arsenic is monoisotopic. The ensemble is $N = 256$
independent configurations drawn with your own random number generator.

**Lattice dynamics.** Use the harmonic normal modes for each configuration's
isotope masses and the supplied Born charges. Let $Z^*_{s,\alpha}$ be the
mode effective charge in mass-weighted, orthonormal eigenvector normalization.

**Dielectric function.** Use the macroscopic harmonic Born-charge dielectric
response with electronic background $\varepsilon_\infty$. Its Cartesian
diagonal components $\varepsilon_{\alpha\alpha}$ are absolute dimensionless
relative permittivities. Use the supplied supercell volume and SI units for
the response. For a mode of angular frequency $\omega_s$, use the damping
denominator $\omega_s^2-\omega^2-i\omega\gamma$, with $\omega$ and $\gamma$
in angular-frequency units. The isotropic response is one third of the tensor's
trace. The disordered spectrum is its mean over the $N$ configurations.
Evaluate it from 600 to 800 cm$^{-1}$ at a step no coarser than 0.01 cm$^{-1}$.

**Quasiparticle baseline.** The same construction for a single configuration
in which every boron site carries the average mass 10.811 u (the virtual
crystal). Its optically active zone-centre frequency
$\omega_{\mathrm{TO}}^{\mathrm{VC}}$ is the frequency of the modes whose
strength $\sum_\alpha |Z^{*}_{s,\alpha}|^2$ exceeds $10^{-6}$ of the largest
strength (they are degenerate).

In every configuration the three zero-frequency acoustic translations carry
no strength; discard them (with any eigenvalue $\omega_s^2 \le 0$) before
forming the sum.

## What to determine

(1) $\omega_{\mathrm{TO}}^{\mathrm{VC}}$, in cm$^{-1}$.

(2) On the disordered spectrum, the features of $\varepsilon''$ in the window
    680 to 740 cm$^{-1}$, split at 710 cm$^{-1}$ (the point 710 itself
    belongs to the lower half): in each half, the strict local maximum of
    greatest topographic prominence (prominence in the sense of
    `scipy.signal.peak_prominences`), its frequency and its value.

(3) On the disordered spectrum, the features of $\varepsilon'$ in the window
    700 to 740 cm$^{-1}$: the strict local maximum of greatest prominence,
    and, among the strict local minima at higher frequencies than it, the one
    of greatest prominence; their frequencies.

(4) The minimum of $\varepsilon'$ over 680 to 740 cm$^{-1}$, for the
    disordered spectrum and for the baseline.

(5) The ensemble convergence of the disordered spectrum: at every grid point
    in 680 to 740 cm$^{-1}$, the standard error of the mean of the modulus
    $|\varepsilon_{xx}(\omega)|$ over the $N$ configurations divided by that
    mean; report the maximum over the window, in percent.

(6) Two moments of the pooled oscillator strength. Let
    $w_s = \tfrac{1}{3}\sum_\alpha |Z^{*}_{s,\alpha}|^2$ for every mode of
    every configuration with $600 \le \omega_s \le 780$ cm$^{-1}$, pooled
    over the $N$ configurations. Report

    - the strength-weighted mean frequency of the modes above 710 cm$^{-1}$
      minus that of the modes at or below 710 cm$^{-1}$, in cm$^{-1}$;
    - one minus the fraction of that pooled strength (the same set of
      modes) carried by modes within $\pm 1$ cm$^{-1}$ of
      $\omega_{\mathrm{TO}}^{\mathrm{VC}}$.

## Output format

Write a single `answer.json` in the workspace root with exactly this structure:

```json
{
  "omega_TO_vc_cm1": <float>,
  "eps2_max_below_710_cm1": <float>,
  "eps2_max_below_710_value": <float>,
  "eps2_max_above_710_cm1": <float>,
  "eps2_max_above_710_value": <float>,
  "eps1_max_700_740_cm1": <float>,
  "eps1_min_above_max_cm1": <float>,
  "min_eps1_disordered_680_740": <float>,
  "min_eps1_baseline_680_740": <float>,
  "rse_max_xx_percent": <float>,
  "strength_centroid_split_cm1": <float>,
  "off_vc_strength_fraction": <float>
}
```
