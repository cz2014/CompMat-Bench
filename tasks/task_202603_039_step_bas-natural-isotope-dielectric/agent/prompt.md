# Task: infrared dielectric function of natural-abundance boron arsenide from an isotope-disordered supercell ensemble

Boron has two stable isotopes whose masses differ by about ten percent, and
natural boron is roughly one fifth $^{10}$B. In cubic boron arsenide the
zone-centre optical phonon is dominated by boron motion, so isotope mass
disorder acts directly on the infrared lattice response. That response can be
described in two ways: a quasiparticle picture, in which the crystal has one
optical oscillator at the frequency of a virtual crystal with the averaged boron
mass, broadened by a linewidth; and a nonperturbative treatment, in which the
harmonic lattice dynamics of a large supercell with random isotope masses on the
boron sites is diagonalized exactly and the dielectric response is averaged over
many configurations. The question is how the two descriptions differ at natural
abundance: the fine structure of $\varepsilon''$ and $\varepsilon'$ near the
optical phonon, whether $\varepsilon'$ turns negative in each, and how the
oscillator strength is distributed over the disordered modes.

## What you are given

- `ifc_harmonic_512.npz` -- the harmonic interatomic force constants of the
  BAs supercell. Use the atom order, index layout and units documented in
  the file. No nonanalytic long-range dipole term is to be added.

No first-principles calculation is required. The environment provides Python
with numpy and scipy; no other software is needed.

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

**Isotope configurations.** A configuration assigns exactly
$n_{10} = \mathrm{round}(x \times 256) = 51$ of the 256 boron sites to
$^{10}$B, drawn uniformly at random without replacement, and the remaining
boron sites to $^{11}$B; arsenic is monoisotopic. The ensemble is $N = 256$
independent configurations drawn with your own random number generator.

**Lattice dynamics.** For a configuration with masses $m_j$, the dynamical
matrix of the supercell is

$$
D_{j\alpha,k\beta} = \frac{\Phi_{j\alpha,k\beta}}{\sqrt{m_j m_k}} ,
$$

with eigenvalues $\omega_s^2$ and orthonormal eigenvectors $e_{s}$
(components $e_{s,j\alpha}$). The mode effective charge of mode $s$ along the
Cartesian direction $\alpha$ is

$$
Z^{*}_{s,\alpha} = \sum_j \frac{Z_j\, e_{s,j\alpha}}{\sqrt{m_j}} .
$$

**Dielectric function.** For one configuration,

$$
\varepsilon_{\alpha\alpha}(\omega) = \varepsilon_\infty +
\frac{1}{\varepsilon_0 V} \sum_s
\frac{\left|Z^{*}_{s,\alpha}\right|^2}{\omega_s^2 - \omega^2 - i\,\omega\gamma} ,
$$

with $V$ the volume of the supercell and every quantity in SI units ($\gamma$
converted to the same angular-frequency unit as $\omega$ and inserted as
written, not halved). The isotropic dielectric function is
$\varepsilon(\omega) = \tfrac{1}{3}\sum_\alpha \varepsilon_{\alpha\alpha}(\omega)$,
and the **disordered spectrum** is its mean over the $N$ configurations.
Evaluate it on 600 to 800 cm$^{-1}$ with a step no coarser than
0.01 cm$^{-1}$.

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

- `omega_TO_vc_cm1`: item (1).
- `eps2_max_below_710_*`, `eps2_max_above_710_*`: item (2), frequency in
  cm$^{-1}$ and the value of $\varepsilon''$ there.
- `eps1_max_700_740_cm1`, `eps1_min_above_max_cm1`: item (3).
- `min_eps1_disordered_680_740`, `min_eps1_baseline_680_740`: item (4).
- `rse_max_xx_percent`: item (5).
- `strength_centroid_split_cm1`, `off_vc_strength_fraction`: item (6), in
  that order.

All values must be computed from the staged file; do not guess or hardcode.
