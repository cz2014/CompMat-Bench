# Task: mode-resolved phonon transmission through a Si/Ge interface from harmonic force constants

Compute the harmonic phonon transmission of a planar Si/Ge(100) interface
between a semi-infinite Si lead on the left and a semi-infinite Ge lead on
the right.

`device_force_constants.hdf5` contains the device's principal-layer
force-constant blocks and atomic masses. Its attributes specify units,
block ordering, coupling directions, layer period and transverse periodicity.
Use these blocks as given, without symmetrisation or sum-rule corrections.
No molecular-dynamics or electronic-structure calculation is required.
Python with numpy, scipy and h5py is available.

## Quantities and accuracy

Let $t$ be the flux-normalized Si-to-Ge transmission matrix, with incoming
propagating Si channels as rows and outgoing propagating Ge channels as columns.
For every incoming channel, transmission plus reflection must sum to one
with absolute error below $10^{-6}$.

At ordinary frequencies $\nu = 1.00, 2.00, 4.00, 5.05, 6.00, 8.00,
10.00, 11.50, 12.00$ THz, report:

- `n_Si`, `n_Ge`: the number of propagating channels in each lead travelling
  towards the interface;
- `Xi`: the sum of transmission probabilities over all incoming Si channels;
- `mean_T_Si`: the mean transmission probability over all incoming Si channels;
- `std_tau_Si`: the population standard deviation of the full spectrum of
  $t t^\dagger$ (all $n_{\mathrm{Si}}$ eigenvalues, including zeros).

## Output format

Write a single `answer.json` in the workspace root with exactly this layout:
one entry per requested frequency, in the order listed above, the frequency
echoed as given (counts are integers; the other three fields are dimensionless
floats):

```json
{
  "results": [
    {
      "frequency_THz": 1.00,
      "n_Si": <int>,
      "n_Ge": <int>,
      "Xi": <float>,
      "mean_T_Si": <float>,
      "std_tau_Si": <float>
    },
    ...
  ]
}
```

All values must be computed from the staged file; do not guess or hardcode.
