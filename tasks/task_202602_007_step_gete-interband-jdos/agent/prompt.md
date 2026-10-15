# Interband JDOS of a Zeeman-split GeTe valence-band model at three doping levels

You are given a Wannier tight-binding model of ferroelectric rhombohedral GeTe whose
Rashba-split bands acquire an additional exchange (Zeeman) splitting, modelling a film
with dilute magnetic doping. The low-energy optical response of such a film depends on
the interband joint density of states (JDOS) within the hole-doped valence manifold.
Your task is to compute the interband JDOS as a function of photon energy for three
samples with different hole densities.

## The provided model

`bundle/` contains the Wannier model data. `wannier_bundle.py` provides the
model interface, documented in its module docstring.

## Physical conditions (given)

These were fixed upstream on this same model; use them as given:

- Zeeman field: h = -50 meV along the [11-2] crystal direction, using the
  supplied module's Zeeman convention.
- Three samples, labelled by their measured hole densities, whose Fermi energies were
  already determined on this bundle's absolute energy scale:

  | hole density p (cm^-3) | E_F (eV, absolute) |
  |---|---|
  | 1.1e21 | 10.794269 |
  | 5.3e20 | 10.897597 |
  | 1.8e19 | 11.122901 |

- Zero temperature (step-function occupations).

## What to compute

For each of the three samples, the interband JDOS at the photon energies 0.050, 0.075,
0.100, 0.125, 0.150, 0.175, 0.200 eV, under the following conventions:

- Include all occupied-to-empty interband transitions of the supplied model,
  with no additional transition-energy cutoff.
- Use Lorentzian broadening with a half-width at half-maximum of 10 meV,
  including only the resonant contribution.
- Use the Gamma-centered, unshifted 60 x 60 x 60 fractional k-mesh
  (points (i, j, k)/60 for i, j, k = 0..59). Report the values on this finite mesh.

Use one consistent normalization for all reported values: per unit cell, in eV^-1.
Grading uses only normalization-independent combinations of the reported values, so a
different overall normalization convention (per volume, a different broadening
prefactor) is not penalized as long as it is applied consistently to all 21 values.

## Answer format

Write a single `answer.json` in the workspace root with exactly:

```json
{
  "jdos_p1.1e21_eVinv": [<float>, <float>, <float>, <float>, <float>, <float>, <float>],
  "jdos_p5.3e20_eVinv": [<float>, <float>, <float>, <float>, <float>, <float>, <float>],
  "jdos_p1.8e19_eVinv": [<float>, <float>, <float>, <float>, <float>, <float>, <float>]
}
```

Each list holds that sample's JDOS at the seven photon energies in the order listed
above (0.050 first, 0.200 last). Report at least six significant digits per value. All
three fields are graded; numbers only, no prose is read.
