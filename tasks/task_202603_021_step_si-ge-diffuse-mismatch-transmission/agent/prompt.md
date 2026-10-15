# Task: diffuse-mismatch phonon transmission across a Si/Ge interface, against a wave-based reference

A planar interface between crystalline Si and crystalline Ge is studied as a
phonon scatterer. Two descriptions of the transmission across it are compared.
The first is the conventional diffuse mismatch model (DMM), in which the
transmission probability of a phonon depends on its frequency alone and is fixed
by the bulk phonon spectra of the two materials. The second is a wave-based
atomistic Green's function (AGF) scattering calculation of the same interface,
whose mode-resolved transmission has been averaged over the incident modes at
each frequency. The question is how much the DMM transmits in three frequency
bands, and whether, and over what bandwidth, the wave-based result exceeds it
at low frequency.

Throughout, **transmission** means the probability that a phonon incident from
the Si side crosses into Ge.

## What you are given

- `spectrum_Si.csv` and `spectrum_Ge.csv` -- the bulk phonon spectrum of each
  material, tabulated on the same frequency grid: the frequency, the total
  phonon density of states at that frequency and the mean phonon group speed at
  that frequency, all branches combined. Each table runs over its own
  material's spectral range; at a frequency where a table has no row, that
  material has no states. The column headers carry the units. Treat the two
  tables as the fixed spectral input of the model: use the density of states
  and the speed columns as they are, without renormalising the density of
  states or resampling the grid.
- `agf_transmission_Si_incident.csv` -- the wave-based reference: the frequency
  and the AGF mean transmission of phonons incident from the Si side, on the
  same grid.

No simulation is required. The environment provides Python with numpy.

## What to determine

(1) The DMM transmission at every grid frequency the items below need. Use
    the DMM in its full-dispersion form, the form that takes the actual
    frequency-dependent spectrum of each material rather than a Debye
    approximation, with the tabulated density of states and speed of both
    materials standing in for their branch-resolved spectra.

(2) The **mean DMM transmission** over each of three frequency windows,
    $[0.5, 4.5]$, $[4.5, 7.5]$ and $[7.5, 11.5]$ THz: the integral of the
    transmission over the window divided by the window width, the tabulated
    curve being taken as piecewise linear between grid points.

(3) The **bandwidth over which the wave-based reference exceeds the DMM** within
    $[0.5, 7.5]$ THz: the total length, in THz, of the set of frequencies in that
    window at which the reference mean transmission is larger than the DMM
    transmission, both curves being taken as piecewise linear between grid
    points.

## Output format

Write a single `answer.json` in the workspace root with exactly these fields
(the three means are dimensionless probabilities; the bandwidth is in THz):

```json
{
  "mean_T_dmm_0p5_4p5THz": <float>,
  "mean_T_dmm_4p5_7p5THz": <float>,
  "mean_T_dmm_7p5_11p5THz": <float>,
  "agf_above_dmm_bandwidth_THz": <float>
}
```

- `mean_T_dmm_0p5_4p5THz`, `mean_T_dmm_4p5_7p5THz`, `mean_T_dmm_7p5_11p5THz`:
  item (2), one per window.
- `agf_above_dmm_bandwidth_THz`: item (3).

All values must be computed from the staged tables; do not guess or hardcode.
