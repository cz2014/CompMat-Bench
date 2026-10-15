# Task: the magneto-Raman tensor of an out-of-plane phonon from displaced-geometry optics runs

Monolayer 2H-RuCl2 is a ferromagnetic insulator with out-of-plane
magnetization. Its low-energy states are massive Dirac fermions at the two
valleys $K$ and $K'$, and the interband gaps of the two valleys differ.

You will compute the Raman tensor of the zone-center out-of-plane optical
phonon (the $A'_1$ mode) and read off its two magneto-Raman observables as
functions of the incident photon energy $\omega$. The Raman tensor is the
derivative of the dynamical (frequency-dependent) dielectric tensor with
respect to the phonon mode coordinate $u$,

$$
R_{ab}(\omega) = \frac{\partial \varepsilon_{ab}(\omega)}{\partial u} ,
$$

evaluated by finite differences over frozen displaced geometries of the mode.
For this out-of-plane mode the Raman tensor is antisymmetric in its off-diagonal
part, $R_{xx} = R_{yy}$ and $R_{xy} = -R_{yx}$, so the independent elements are
$R_{xx}$ and $R_{xy}$.

## Provided data

Five frozen-phonon optics calculations, one per mode amplitude
$u \in \{0.000, +0.010, -0.010, +0.020, -0.020\}$ Angstrom, under `data/`:

    data/u_0p00/   u =  0.000 A
    data/u_p010/   u = +0.010 A     data/u_m010/   u = -0.010 A
    data/u_p020/   u = +0.020 A     data/u_m020/   u = -0.020 A

Each directory holds `WAVEDER` (the optical transition matrix elements written
by the LOPTICS run for that geometry) and `vasprun.opt.xml` (the corresponding
run record: eigenvalues, occupations, k-point weights, and cell). Assemble the
full complex dielectric tensor $\varepsilon_{ab}(\omega)$ for each geometry
from these, apply an electronic decay rate $\eta = 0.05$ eV in the assembly,
entering the transition-energy denominators as a per-transition broadening
(the WAVEDER transition matrix elements are unbroadened; $\eta$ is not a convolution applied
to a finished spectrum), and central-difference to $R_{ab}(\omega)$. Use a
frequency grid fine enough to resolve the resonances (spacing of a few meV or
finer) spanning at least 0 to 0.7 eV. Report every quantity below from the
$\pm 0.010$ Angstrom central-difference pair; the $\pm 0.020$ Angstrom pair is
provided for your own consistency checks only. Overall prefactors (and any
monolayer volume normalization) cancel in every reported quantity, which are
all ratios, phases, or peak positions.

## The two valley gaps

Several quantities below are sampled at the two valley gaps: the direct
interband gap (lowest unoccupied minus highest occupied eigenvalue at the same
k-point) at

$$
K = \left(\tfrac{1}{3}, \tfrac{1}{3}, 0\right), \qquad
K' = \left(\tfrac{2}{3}, -\tfrac{1}{3}, 0\right)
$$

in fractional coordinates of the reciprocal lattice, modulo reciprocal lattice
vectors. Read both from the eigenvalues of the undisplaced ($u = 0.000$) run;
both points lie on its k-point mesh. Call the smaller of the two the low gap
and the larger the high gap, and report them as `gap_low_eV` and `gap_high_eV`.

## What to compute and report

**(1) Circular-polarization intensities.** For the in-plane circular
polarization vectors $\hat{e}_{\pm} = (1, \pm i)/\sqrt{2}$, the backscattering
Raman intensity for incident polarization $\hat{e}_i$ and scattered
polarization $\hat{e}_s$ is

$$
I(\omega) = \left| \hat{e}_s^{*} \cdot R(\omega) \cdot \hat{e}_i \right|^2 .
$$

The two same-handedness channels $I_{LL}(\omega)$ and $I_{RR}(\omega)$ are the
two choices with $\hat{e}_s = \hat{e}_i$. Locate the photon energy at which
each channel peaks, and quantify how strongly each channel is suppressed at
the other channel's peak. Report:
- `resonance_LL_eV`, `resonance_RR_eV`: the photon energy at which each channel
  peaks;
- `extinction_contrast_LL`, `extinction_contrast_RR`: for each channel, its
  intensity at the OTHER channel's resonance divided by its own peak intensity.

**(2) Phase difference.** With $\phi_{xx}(\omega) = \arg R_{xx}(\omega)$ and
$\phi_{xy}(\omega) = \arg R_{xy}(\omega)$, the phase difference
$\phi_{xy} - \phi_{xx}$ sampled at the two valley-gap photon energies. Report,
in units of $\pi$:
- `phase_at_gap_low_over_pi`: $(\phi_{xy} - \phi_{xx})/\pi$ at the low gap;
- `phase_at_gap_high_over_pi`: the same at the high gap;
- `phase_step_over_pi`: the high-gap value minus the low-gap value.

Wrap all three phase quantities to the interval $(-1, 1]$.

**(3) Decay-rate dependence.**
- `phase_by_eta_over_pi`: repeat the phase computation of (2) with the decay
  rate set to $\eta \in \{0.03, 0.05, 0.07, 0.09\}$ eV in turn. For each
  $\eta$ report the pair $[(\phi_{xy} - \phi_{xx})/\pi \text{ at the low gap},
  \; \text{the same at the high gap}]$, each wrapped to $(-1, 1]$; four pairs,
  in the $\eta$ order above.

## Output format

Write `answer.json` in the workspace root with exactly this schema (all energies
in eV; phases in units of $\pi$; contrasts and deviations dimensionless):

```json
{
  "gap_low_eV": <float>,
  "gap_high_eV": <float>,
  "resonance_LL_eV": <float>,
  "resonance_RR_eV": <float>,
  "extinction_contrast_LL": <float>,
  "extinction_contrast_RR": <float>,
  "phase_at_gap_low_over_pi": <float>,
  "phase_at_gap_high_over_pi": <float>,
  "phase_step_over_pi": <float>,
  "phase_by_eta_over_pi": [[<float>, <float>], [<float>, <float>], [<float>, <float>], [<float>, <float>]]
}
```
