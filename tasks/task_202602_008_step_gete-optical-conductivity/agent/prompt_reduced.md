# Low-frequency interband optical conductivity of a Zeeman-split GeTe valence-band model

Compute the real interband optical conductivity of the supplied Zeeman-split
GeTe Wannier model and report the retention and peak ratios defined below.

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

- Zero temperature.

## What to compute

For each of the three samples, sigma_1(omega) for photon energies omega from 0 to
0.30 eV in steps of 0.005 eV (61 grid points), under the following conventions:

- **Interband only.** Include all occupied-to-empty interband transitions of the
  supplied model. Exclude intraband (Drude) contributions.
- **Component.** The diagonal component along the [11-2] crystal direction,
  denoted by $a$ below; $v^a_{ul}$ is the corresponding interband velocity
  matrix element.
- **Spectral weight and broadening.** Take $\sigma_1(\omega)$ as, up to an overall
  constant,

  $$
  \sigma_1(\omega) \propto \sum_{\mathbf{k}} \sum_{l \in \mathrm{occ}}
  \sum_{u \in \mathrm{empty}}
  \frac{\left| v^{a}_{ul}(\mathbf{k}) \right|^2}{E_u(\mathbf{k}) - E_l(\mathbf{k})}
  \left[ L_\gamma\!\big(E_u - E_l - \omega\big)
        + L_\gamma\!\big(E_u - E_l + \omega\big) \right],
  $$

  where $L_\gamma$ is a Lorentzian with HWHM $\gamma=50$ meV.
- **Brillouin-zone sum.** The Gamma-centered, unshifted 56 x 56 x 56 fractional
  k-mesh (points (i, j, k)/56 for i, j, k = 0..55). Report the quantities on this mesh.
- **Normalization.** Use one consistent overall normalization across all three
  samples; the overall constant is unrestricted.

## What to report

From each sample's sigma_1 curve on the grid above:

- **retention** = sigma_1(0.05 eV) / sigma_1(0.20 eV).
- **peak** = the maximum of sigma_1 over the omega grid. Report the two peak ratios
  peak(p = 1.1e21) / peak(p = 1.8e19) and peak(p = 5.3e20) / peak(p = 1.8e19).

## Answer format

Write a single `answer.json` in the workspace root with exactly:

```json
{
  "retention_p1.1e21": <float>,
  "retention_p5.3e20": <float>,
  "retention_p1.8e19": <float>,
  "peak_ratio_p1.1e21": <float>,
  "peak_ratio_p5.3e20": <float>
}
```
