# Task: isotope-composition dependence of the infrared dielectric function of boron arsenide

Boron has two stable isotopes whose masses differ by about ten percent, and a
crystal of cubic boron arsenide can be grown at any $^{10}$B fraction $x$
between the two isotopically pure limits. Because the zone-centre optical
phonon is dominated by boron motion, the isotope composition reshapes the
infrared lattice response. The optical-phonon dielectric response determines the frequency range
relevant to surface phonon polaritons. The question
is how this behaviour evolves with composition when the disordered lattice
dynamics is treated nonperturbatively, by diagonalizing the harmonic dynamics of
a large supercell with random isotope masses and averaging over configurations:
at which compositions the minimum of $\varepsilon'$ changes sign, and what the
pure limits look like in the same framework.

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
  As 74.921595 u.
- Phonon linewidth at 300 K: $\gamma = 0.29$ cm$^{-1}$, a full width at half
  maximum, the same for every mode.

## The model (fixed for this task)

**Composition axis and isotope configurations.** The composition is specified
by the integer $n_{10}$, the number of $^{10}$B atoms among the 256 boron
sites of the supercell, $x = n_{10}/256$. A configuration at $n_{10}$ assigns
exactly $n_{10}$ boron sites to $^{10}$B, drawn uniformly at random without
replacement, and the remaining boron sites to $^{11}$B; arsenic is
monoisotopic. At every composition the ensemble is $N = 256$ independent
configurations drawn with your own random number generator (at $n_{10} = 0$
and $n_{10} = 256$ every configuration is the same crystal, so one suffices).

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
and the spectrum of a composition is its mean over the $N$ configurations.
Evaluate it on 680 to 740 cm$^{-1}$ with a step no coarser than
0.01 cm$^{-1}$.

In every configuration the three zero-frequency acoustic translations carry
no strength; discard them (with any eigenvalue $\omega_s^2 \le 0$) before
forming the sum.

**The graded quantity of a composition** is $\min \varepsilon'(\omega)$ over
680 to 740 cm$^{-1}$, written $m(n_{10})$.

## What to determine

(1) At the two pure limits, $n_{10} = 0$ and $n_{10} = 256$: the number of
    optically active modes, that is, the number of modes $s$ whose strength
    $\sum_\alpha |Z^{*}_{s,\alpha}|^2$ exceeds $10^{-6}$ of the largest
    strength in that configuration; and $m(n_{10})$.

(2) $m(n_{10})$ at $n_{10} = 13$, $51$, $128$ and $251$.

(3) The two sign changes of $m(n_{10})$ along the composition axis, each
    known to lie inside a bracket: one inside $42 \le n_{10} \le 51$ and one
    inside $243 \le n_{10} \le 252$. For the lower bracket report the
    smallest $n_{10}$ in it at which $m(n_{10}) > 0$; for the upper bracket
    report the smallest $n_{10}$ in it at which $m(n_{10}) < 0$ (report $-1$
    if no composition in a bracket qualifies). Which compositions inside a
    bracket you evaluate, and in what order, is your choice; every evaluation
    uses the ensemble rule above.

## Output format

Write a single `answer.json` in the workspace root with exactly this structure:

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

- `n_active_modes_n10_0`, `n_active_modes_n10_256`: item (1), counts.
- `min_eps1_n10_<n>`: $m(n_{10})$ at that composition, items (1) and (2),
  dimensionless.
- `n10_first_positive_lower_bracket`, `n10_first_negative_upper_bracket`:
  item (3).

All values must be computed from the staged file; do not guess or hardcode.
